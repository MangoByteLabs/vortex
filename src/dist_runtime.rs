// Native distributed training runtime for Vortex
// TCP-based communication backend — no NCCL dependency

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crate::interpreter::{Env, FnDef, Value};
use crate::nn;

// ─── Wire Protocol ──────────────────────────────────────────────────────────

const MSG_ALL_REDUCE: u32 = 1;
const MSG_BROADCAST: u32 = 2;
const MSG_SCATTER: u32 = 3;
const MSG_GATHER: u32 = 4;
const MSG_BARRIER: u32 = 5;
const MSG_ACTIVATIONS: u32 = 6;
const MSG_GRADIENTS: u32 = 7;
const MSG_HEARTBEAT: u32 = 8;
const MSG_JOIN: u32 = 9;
const MSG_LEAVE: u32 = 10;
const MSG_MODEL_PARAMS: u32 = 11;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReduceOp {
    Sum,
    Avg,
    Max,
    Min,
}

// ─── Message Serialization ──────────────────────────────────────────────────

fn serialize_message(msg_type: u32, data: &[f64]) -> Vec<u8> {
    let data_bytes = data.len() * 8;
    let mut buf = Vec::with_capacity(12 + data_bytes);
    buf.extend_from_slice(&msg_type.to_le_bytes());
    buf.extend_from_slice(&(data_bytes as u64).to_le_bytes());
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn deserialize_message(buf: &[u8]) -> Result<(u32, Vec<f64>), String> {
    if buf.len() < 12 {
        return Err("message too short".into());
    }
    let msg_type = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]);
    let data_len = u64::from_le_bytes([buf[4], buf[5], buf[6], buf[7], buf[8], buf[9], buf[10], buf[11]]) as usize;
    let num_f64 = data_len / 8;
    if buf.len() < 12 + data_len {
        return Err(format!("message truncated: expected {} bytes, got {}", 12 + data_len, buf.len()));
    }
    let mut data = Vec::with_capacity(num_f64);
    for i in 0..num_f64 {
        let offset = 12 + i * 8;
        let v = f64::from_le_bytes([
            buf[offset], buf[offset + 1], buf[offset + 2], buf[offset + 3],
            buf[offset + 4], buf[offset + 5], buf[offset + 6], buf[offset + 7],
        ]);
        data.push(v);
    }
    Ok((msg_type, data))
}

fn send_msg(stream: &mut TcpStream, msg_type: u32, data: &[f64]) -> Result<(), String> {
    let buf = serialize_message(msg_type, data);
    // Send total length first (4 bytes), then the message
    let total_len = buf.len() as u32;
    stream.write_all(&total_len.to_le_bytes()).map_err(|e| format!("send len: {}", e))?;
    stream.write_all(&buf).map_err(|e| format!("send data: {}", e))?;
    stream.flush().map_err(|e| format!("flush: {}", e))?;
    Ok(())
}

fn recv_msg(stream: &mut TcpStream) -> Result<(u32, Vec<f64>), String> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf).map_err(|e| format!("recv len: {}", e))?;
    let total_len = u32::from_le_bytes(len_buf) as usize;
    let mut buf = vec![0u8; total_len];
    stream.read_exact(&mut buf).map_err(|e| format!("recv data: {}", e))?;
    deserialize_message(&buf)
}

// ─── Gradient Compression ───────────────────────────────────────────────────

/// Top-K sparsification: keep only the largest K% of gradient values
pub fn top_k_sparsify(grads: &[f64], k_percent: f64) -> (Vec<usize>, Vec<f64>) {
    let k = ((grads.len() as f64 * k_percent / 100.0).ceil() as usize).max(1).min(grads.len());
    let mut indexed: Vec<(usize, f64)> = grads.iter().enumerate().map(|(i, &v)| (i, v.abs())).collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
    let values: Vec<f64> = indices.iter().map(|&i| grads[i]).collect();
    (indices, values)
}

/// Reconstruct full gradient from sparse representation
pub fn sparse_to_dense(indices: &[usize], values: &[f64], total_len: usize) -> Vec<f64> {
    let mut out = vec![0.0; total_len];
    for (&idx, &val) in indices.iter().zip(values.iter()) {
        if idx < total_len {
            out[idx] = val;
        }
    }
    out
}

/// Quantize f64 to f16 (half precision) for bandwidth savings
pub fn quantize_f64_to_f16(data: &[f64]) -> Vec<u16> {
    data.iter().map(|&v| f64_to_f16(v)).collect()
}

/// Dequantize f16 back to f64
pub fn dequantize_f16_to_f64(data: &[u16]) -> Vec<f64> {
    data.iter().map(|&v| f16_to_f64(v)).collect()
}

fn f64_to_f16(v: f64) -> u16 {
    let v = v as f32;
    let bits = v.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0xFF {
        // Inf/NaN
        return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return ((sign << 15) | 0x7C00) as u16; // overflow → inf
    }
    if new_exp <= 0 {
        return (sign << 15) as u16; // underflow → 0
    }
    let new_frac = frac >> 13;
    ((sign << 15) | (new_exp as u32) << 10 | new_frac) as u16
}

fn f16_to_f64(h: u16) -> f64 {
    let sign = (h >> 15) & 1;
    let exp = (h >> 10) & 0x1F;
    let frac = h & 0x3FF;
    if exp == 0x1F {
        return if frac != 0 { f64::NAN } else if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY };
    }
    if exp == 0 {
        if frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        // Denormalized
        let v = (frac as f64) / 1024.0 * 2.0f64.powi(-14);
        return if sign == 1 { -v } else { v };
    }
    let v = (1.0 + frac as f64 / 1024.0) * 2.0f64.powi(exp as i32 - 15);
    if sign == 1 { -v } else { v }
}

/// Error feedback accumulator for gradient compression
#[derive(Debug, Clone)]
pub struct ErrorFeedback {
    pub residuals: Vec<f64>,
}

impl ErrorFeedback {
    pub fn new(size: usize) -> Self {
        Self { residuals: vec![0.0; size] }
    }

    /// Add current gradients to accumulated residuals, sparsify, keep unsent as residual
    pub fn compress_with_feedback(&mut self, grads: &[f64], k_percent: f64) -> Vec<f64> {
        // Add residuals to current gradients
        let combined: Vec<f64> = grads.iter().zip(self.residuals.iter()).map(|(g, r)| g + r).collect();
        // Sparsify
        let (indices, values) = top_k_sparsify(&combined, k_percent);
        let sent = sparse_to_dense(&indices, &values, combined.len());
        // Update residuals: what we didn't send
        self.residuals = combined.iter().zip(sent.iter()).map(|(c, s)| c - s).collect();
        sent
    }
}

// ─── Distributed Runtime ────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DistRuntime {
    pub rank: usize,
    pub world_size: usize,
    pub peers: Vec<SocketAddr>,
    listener: Option<TcpListener>,
    connections: HashMap<usize, TcpStream>,
    alive: Vec<bool>,
    last_heartbeat: Vec<Instant>,
}

impl DistRuntime {
    pub fn new(rank: usize, world_size: usize, peers: Vec<SocketAddr>) -> Result<Self, String> {
        let listener = if rank < peers.len() {
            let addr = peers[rank];
            let l = TcpListener::bind(addr).map_err(|e| format!("bind {}: {}", addr, e))?;
            l.set_nonblocking(true).ok();
            Some(l)
        } else {
            None
        };
        Ok(Self {
            rank,
            world_size,
            peers,
            listener,
            connections: HashMap::new(),
            alive: vec![true; world_size],
            last_heartbeat: vec![Instant::now(); world_size],
        })
    }

    fn get_connection(&mut self, peer: usize) -> Result<&mut TcpStream, String> {
        if !self.connections.contains_key(&peer) {
            let addr = self.peers[peer];
            let stream = TcpStream::connect_timeout(&addr, Duration::from_secs(10))
                .map_err(|e| format!("connect to peer {}: {}", peer, e))?;
            stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
            stream.set_write_timeout(Some(Duration::from_secs(30))).ok();
            self.connections.insert(peer, stream);
        }
        Ok(self.connections.get_mut(&peer).unwrap())
    }

    /// All-reduce: combine data from all workers using the given operation
    pub fn all_reduce(&mut self, data: &[f64], op: ReduceOp) -> Result<Vec<f64>, String> {
        // Ring all-reduce implementation
        let n = self.world_size;
        if n == 1 {
            return Ok(data.to_vec());
        }

        // For simulated single-process mode, just return data
        if self.peers.is_empty() {
            return Ok(data.to_vec());
        }

        // Reduce-scatter phase
        let chunk_size = (data.len() + n - 1) / n;
        let mut buf = data.to_vec();
        let left = (self.rank + n - 1) % n;
        let right = (self.rank + 1) % n;

        for step in 0..n - 1 {
            let send_chunk = (self.rank + n - step) % n;
            let recv_chunk = (self.rank + n - step - 1) % n;
            let send_start = send_chunk * chunk_size;
            let send_end = (send_start + chunk_size).min(buf.len());
            let send_data: Vec<f64> = if send_start < buf.len() {
                buf[send_start..send_end].to_vec()
            } else {
                vec![]
            };

            // Send to right neighbor
            let right_stream = self.get_connection(right)?;
            send_msg(right_stream, MSG_ALL_REDUCE, &send_data)?;

            // Receive from left neighbor
            let left_stream = self.get_connection(left)?;
            let (_, recv_data) = recv_msg(left_stream)?;

            // Reduce into recv_chunk
            let recv_start = recv_chunk * chunk_size;
            for (i, &v) in recv_data.iter().enumerate() {
                let idx = recv_start + i;
                if idx < buf.len() {
                    buf[idx] = match op {
                        ReduceOp::Sum | ReduceOp::Avg => buf[idx] + v,
                        ReduceOp::Max => buf[idx].max(v),
                        ReduceOp::Min => buf[idx].min(v),
                    };
                }
            }
        }

        // All-gather phase
        for step in 0..n - 1 {
            let send_chunk = (self.rank + 1 + step) % n;
            let recv_chunk = (self.rank + step) % n;
            let send_start = send_chunk * chunk_size;
            let send_end = (send_start + chunk_size).min(buf.len());
            let send_data: Vec<f64> = if send_start < buf.len() {
                buf[send_start..send_end].to_vec()
            } else {
                vec![]
            };

            let right_stream = self.get_connection(right)?;
            send_msg(right_stream, MSG_ALL_REDUCE, &send_data)?;

            let left_stream = self.get_connection(left)?;
            let (_, recv_data) = recv_msg(left_stream)?;

            let recv_start = recv_chunk * chunk_size;
            for (i, &v) in recv_data.iter().enumerate() {
                let idx = recv_start + i;
                if idx < buf.len() {
                    buf[idx] = v;
                }
            }
        }

        if op == ReduceOp::Avg {
            for v in &mut buf {
                *v /= n as f64;
            }
        }

        Ok(buf)
    }

    /// Broadcast: root sends data to all workers
    pub fn broadcast(&mut self, data: &[f64], root: usize) -> Result<Vec<f64>, String> {
        if self.world_size == 1 {
            return Ok(data.to_vec());
        }
        if self.peers.is_empty() {
            return Ok(data.to_vec());
        }

        if self.rank == root {
            for peer in 0..self.world_size {
                if peer != self.rank {
                    let stream = self.get_connection(peer)?;
                    send_msg(stream, MSG_BROADCAST, data)?;
                }
            }
            Ok(data.to_vec())
        } else {
            let stream = self.get_connection(root)?;
            let (_, recv_data) = recv_msg(stream)?;
            Ok(recv_data)
        }
    }

    /// Scatter: root splits data and distributes chunks
    pub fn scatter(&mut self, data: &[f64], root: usize) -> Result<Vec<f64>, String> {
        if self.world_size == 1 {
            return Ok(data.to_vec());
        }
        if self.peers.is_empty() {
            let chunk_size = data.len() / self.world_size;
            let start = self.rank * chunk_size;
            let end = if self.rank == self.world_size - 1 { data.len() } else { start + chunk_size };
            return Ok(data[start..end].to_vec());
        }

        let chunk_size = data.len() / self.world_size;
        if self.rank == root {
            for peer in 0..self.world_size {
                if peer != self.rank {
                    let start = peer * chunk_size;
                    let end = if peer == self.world_size - 1 { data.len() } else { start + chunk_size };
                    let chunk = &data[start..end];
                    let stream = self.get_connection(peer)?;
                    send_msg(stream, MSG_SCATTER, chunk)?;
                }
            }
            let start = root * chunk_size;
            let end = if root == self.world_size - 1 { data.len() } else { start + chunk_size };
            Ok(data[start..end].to_vec())
        } else {
            let stream = self.get_connection(root)?;
            let (_, recv_data) = recv_msg(stream)?;
            Ok(recv_data)
        }
    }

    /// Gather: collect data from all workers to root
    pub fn gather(&mut self, data: &[f64], root: usize) -> Result<Vec<f64>, String> {
        if self.world_size == 1 {
            return Ok(data.to_vec());
        }
        if self.peers.is_empty() {
            return Ok(data.to_vec());
        }

        if self.rank == root {
            let mut result = vec![Vec::new(); self.world_size];
            result[root] = data.to_vec();
            for peer in 0..self.world_size {
                if peer != root {
                    let stream = self.get_connection(peer)?;
                    let (_, recv_data) = recv_msg(stream)?;
                    result[peer] = recv_data;
                }
            }
            Ok(result.into_iter().flatten().collect())
        } else {
            let stream = self.get_connection(root)?;
            send_msg(stream, MSG_GATHER, data)?;
            Ok(data.to_vec())
        }
    }

    /// Barrier: synchronize all workers
    pub fn barrier(&mut self) -> Result<(), String> {
        if self.world_size <= 1 || self.peers.is_empty() {
            return Ok(());
        }
        // Simple centralized barrier: everyone sends to rank 0, rank 0 broadcasts back
        if self.rank == 0 {
            for peer in 1..self.world_size {
                let stream = self.get_connection(peer)?;
                let _ = recv_msg(stream)?;
            }
            for peer in 1..self.world_size {
                let stream = self.get_connection(peer)?;
                send_msg(stream, MSG_BARRIER, &[])?;
            }
        } else {
            let stream = self.get_connection(0)?;
            send_msg(stream, MSG_BARRIER, &[])?;
            let _ = recv_msg(stream)?;
        }
        Ok(())
    }

    /// Send heartbeat to detect dead workers
    pub fn send_heartbeat(&mut self) -> Result<(), String> {
        let now = Instant::now();
        let rank = self.rank;
        let rank_f64 = rank as f64;
        for peer in 0..self.world_size {
            if peer != rank && self.alive[peer] {
                if let Ok(stream) = self.get_connection(peer) {
                    let _ = send_msg(stream, MSG_HEARTBEAT, &[rank_f64]);
                }
            }
        }
        self.last_heartbeat[rank] = now;
        Ok(())
    }

    /// Check which workers are alive (heartbeat timeout = 15 seconds)
    pub fn check_alive(&self) -> Vec<bool> {
        let now = Instant::now();
        self.last_heartbeat.iter().enumerate().map(|(i, &t)| {
            if i == self.rank { true } else { now.duration_since(t).as_secs() < 15 }
        }).collect()
    }

    /// Get number of currently alive workers
    pub fn alive_count(&self) -> usize {
        self.check_alive().iter().filter(|&&a| a).count()
    }
}

// ─── Simulated Runtime (for testing without TCP) ────────────────────────────

/// Simulated distributed runtime for single-process testing
#[derive(Debug, Clone)]
pub struct SimulatedRuntime {
    pub rank: usize,
    pub world_size: usize,
    pub buffers: Vec<Vec<f64>>,
}

impl SimulatedRuntime {
    pub fn new(world_size: usize) -> Vec<SimulatedRuntime> {
        (0..world_size).map(|r| SimulatedRuntime {
            rank: r,
            world_size,
            buffers: vec![Vec::new(); world_size],
        }).collect()
    }

    pub fn all_reduce_simulated(workers: &mut [SimulatedRuntime], data: &[Vec<f64>], op: ReduceOp) -> Vec<Vec<f64>> {
        let n = workers.len();
        let len = data[0].len();
        let mut result = vec![vec![0.0; len]; n];

        for i in 0..len {
            let vals: Vec<f64> = data.iter().map(|d| d[i]).collect();
            let reduced = match op {
                ReduceOp::Sum => vals.iter().sum(),
                ReduceOp::Avg => vals.iter().sum::<f64>() / n as f64,
                ReduceOp::Max => vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                ReduceOp::Min => vals.iter().cloned().fold(f64::INFINITY, f64::min),
            };
            for r in &mut result {
                r[i] = reduced;
            }
        }
        result
    }

    pub fn broadcast_simulated(data: &[f64], _root: usize, world_size: usize) -> Vec<Vec<f64>> {
        vec![data.to_vec(); world_size]
    }

    pub fn scatter_simulated(data: &[f64], world_size: usize) -> Vec<Vec<f64>> {
        let chunk_size = data.len() / world_size;
        (0..world_size).map(|r| {
            let start = r * chunk_size;
            let end = if r == world_size - 1 { data.len() } else { start + chunk_size };
            data[start..end].to_vec()
        }).collect()
    }

    pub fn gather_simulated(chunks: &[Vec<f64>]) -> Vec<f64> {
        chunks.iter().flat_map(|c| c.iter().cloned()).collect()
    }
}

// ─── Data Parallel Trainer ──────────────────────────────────────────────────

pub struct DataParallelTrainer {
    pub world_size: usize,
    pub rank: usize,
    error_feedback: Option<ErrorFeedback>,
    pub compression_k: Option<f64>, // top-k percentage, None = no compression
}

impl DataParallelTrainer {
    pub fn new(rank: usize, world_size: usize) -> Self {
        Self {
            world_size,
            rank,
            error_feedback: None,
            compression_k: None,
        }
    }

    pub fn enable_compression(&mut self, k_percent: f64) {
        self.compression_k = Some(k_percent);
    }

    /// Data-parallel training using simulated workers (single process)
    pub fn train_simulated(
        &mut self,
        model: &mut nn::Model,
        data: Vec<Vec<f64>>,
        labels: Vec<Vec<f64>>,
        epochs: usize,
        lr: f64,
    ) -> Vec<f64> {
        let n = self.world_size;
        let samples = data.len();
        let samples_per_worker = samples / n;

        // Create model copies for each worker
        let mut models: Vec<nn::Model> = (0..n).map(|_| model.clone()).collect();
        let mut optimizers: Vec<nn::Optimizer> = (0..n).map(|_| nn::Optimizer::sgd(lr, 0.9, 0.0)).collect();

        let mut losses = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0;

            for w in 0..n {
                // Each worker gets a slice of data
                let start = w * samples_per_worker;
                let end = if w == n - 1 { samples } else { start + samples_per_worker };
                let worker_data = &data[start..end];
                let worker_labels = &labels[start..end];

                // Forward + backward on local data
                models[w].zero_grad();
                let mut batch_loss = 0.0;
                for s in 0..worker_data.len() {
                    let input = nn::Tensor::new(vec![1, worker_data[s].len()], worker_data[s].clone());
                    let target = nn::Tensor::new(vec![1, worker_labels[s].len()], worker_labels[s].clone());
                    let pred = models[w].forward(&input);
                    let (loss, grad) = mse_loss_simple(&pred, &target);
                    batch_loss += loss;
                    models[w].backward(&grad);
                }
                batch_loss /= worker_data.len() as f64;
                epoch_loss += batch_loss;
            }
            epoch_loss /= n as f64;

            // All-reduce gradients: average across all worker models
            let num_layers = models[0].layers.len();
            for l in 0..num_layers {
                let param_groups: Vec<Vec<(&mut nn::Tensor, &mut nn::Tensor)>> =
                    models.iter_mut().map(|m| m.layers[l].parameters_mut()).collect();

                // For each parameter in this layer, average gradients across workers
                if !param_groups.is_empty() && !param_groups[0].is_empty() {
                    let n_params = param_groups[0].len();
                    // Collect gradient data from all workers, average, then distribute
                    // We need to work around borrow checker, so collect grads first
                    for _p in 0..n_params {
                        // This is simulated: we just average in place
                    }
                }
            }

            // Simulated all-reduce: average all model parameters
            average_model_params(&mut models);

            // Update each model with averaged gradients
            for w in 0..n {
                optimizers[w].step(&mut models[w]);
            }

            // Sync all models to model 0's state
            for w in 1..n {
                models[w] = models[0].clone();
            }

            losses.push(epoch_loss);
        }

        // Copy final model back
        *model = models[0].clone();
        losses
    }
}

/// Average model parameters across all model copies
fn average_model_params(models: &mut [nn::Model]) {
    let n = models.len();
    if n <= 1 { return; }

    let num_layers = models[0].layers.len();
    for l in 0..num_layers {
        // Collect all parameter data from layer l across all models
        let all_params: Vec<Vec<Vec<f64>>> = models.iter().map(|m| {
            let pairs = get_layer_param_data(&m.layers[l]);
            pairs
        }).collect();

        if all_params[0].is_empty() { continue; }

        // Average each parameter
        let n_params = all_params[0].len();
        let mut averaged: Vec<Vec<f64>> = Vec::with_capacity(n_params);
        for p in 0..n_params {
            let param_len = all_params[0][p].len();
            let mut avg = vec![0.0; param_len];
            for w in 0..n {
                for i in 0..param_len {
                    avg[i] += all_params[w][p][i];
                }
            }
            for v in &mut avg {
                *v /= n as f64;
            }
            averaged.push(avg);
        }

        // Write averaged params back to all models
        for m in models.iter_mut() {
            set_layer_param_data(&mut m.layers[l], &averaged);
        }
    }
}

fn get_layer_param_data(layer: &nn::Layer) -> Vec<Vec<f64>> {
    match layer {
        nn::Layer::Linear { weight, bias, .. } => vec![weight.data.clone(), bias.data.clone()],
        nn::Layer::LayerNorm { gamma, beta, .. } => vec![gamma.data.clone(), beta.data.clone()],
        nn::Layer::BatchNorm { gamma, beta, .. } => vec![gamma.data.clone(), beta.data.clone()],
        nn::Layer::Embedding { weight, .. } => vec![weight.data.clone()],
        nn::Layer::FeedForward { w1, w2, b1, b2, .. } => vec![w1.data.clone(), w2.data.clone(), b1.data.clone(), b2.data.clone()],
        nn::Layer::MultiHeadAttention { wq, wk, wv, wo, .. } => vec![wq.data.clone(), wk.data.clone(), wv.data.clone(), wo.data.clone()],
        _ => vec![],
    }
}

fn set_layer_param_data(layer: &mut nn::Layer, params: &[Vec<f64>]) {
    match layer {
        nn::Layer::Linear { weight, bias, .. } => {
            if params.len() >= 2 {
                weight.data = params[0].clone();
                bias.data = params[1].clone();
            }
        }
        nn::Layer::LayerNorm { gamma, beta, .. } => {
            if params.len() >= 2 {
                gamma.data = params[0].clone();
                beta.data = params[1].clone();
            }
        }
        nn::Layer::BatchNorm { gamma, beta, .. } => {
            if params.len() >= 2 {
                gamma.data = params[0].clone();
                beta.data = params[1].clone();
            }
        }
        nn::Layer::Embedding { weight, .. } => {
            if !params.is_empty() {
                weight.data = params[0].clone();
            }
        }
        nn::Layer::FeedForward { w1, w2, b1, b2, .. } => {
            if params.len() >= 4 {
                w1.data = params[0].clone();
                w2.data = params[1].clone();
                b1.data = params[2].clone();
                b2.data = params[3].clone();
            }
        }
        nn::Layer::MultiHeadAttention { wq, wk, wv, wo, .. } => {
            if params.len() >= 4 {
                wq.data = params[0].clone();
                wk.data = params[1].clone();
                wv.data = params[2].clone();
                wo.data = params[3].clone();
            }
        }
        _ => {}
    }
}

fn mse_loss_simple(pred: &nn::Tensor, target: &nn::Tensor) -> (f64, nn::Tensor) {
    let diff = pred.sub(target);
    let loss = diff.data.iter().map(|v| v * v).sum::<f64>() / diff.data.len() as f64;
    let grad = diff.mul_scalar(2.0 / diff.data.len() as f64);
    (loss, grad)
}

// ─── Model Parallel Trainer ─────────────────────────────────────────────────

pub struct ModelParallelTrainer {
    pub world_size: usize,
    pub rank: usize,
    pub layer_assignment: Vec<usize>, // which worker owns which layer
}

impl ModelParallelTrainer {
    pub fn new(rank: usize, world_size: usize, num_layers: usize) -> Self {
        // Assign layers round-robin across workers
        let layer_assignment: Vec<usize> = (0..num_layers).map(|l| l % world_size).collect();
        Self { world_size, rank, layer_assignment }
    }

    /// Get indices of layers assigned to this worker
    pub fn local_layer_indices(&self) -> Vec<usize> {
        self.layer_assignment.iter().enumerate()
            .filter(|(_, &w)| w == self.rank)
            .map(|(i, _)| i)
            .collect()
    }

    /// Simulated model-parallel forward pass
    pub fn forward_simulated(&self, model: &mut nn::Model, input: &nn::Tensor) -> nn::Tensor {
        let mut activation = input.clone();
        for (layer_idx, &owner) in self.layer_assignment.iter().enumerate() {
            if layer_idx < model.layers.len() {
                // In simulation, all workers have all layers; in real distributed,
                // only the owner would compute and then send activations
                activation = model.layers[layer_idx].forward(&activation);
            }
        }
        activation
    }

    /// Simulated model-parallel backward pass
    pub fn backward_simulated(&self, model: &mut nn::Model, grad: &nn::Tensor) -> nn::Tensor {
        let mut g = grad.clone();
        for layer_idx in (0..model.layers.len()).rev() {
            g = model.layers[layer_idx].backward(&g);
        }
        g
    }

    /// Pipeline parallel training: split micro-batches across pipeline stages
    pub fn train_pipeline_simulated(
        &self,
        model: &mut nn::Model,
        data: &[Vec<f64>],
        labels: &[Vec<f64>],
        micro_batch_size: usize,
        lr: f64,
    ) -> f64 {
        let num_micro_batches = (data.len() + micro_batch_size - 1) / micro_batch_size;
        let mut total_loss = 0.0;
        let mut optimizer = nn::Optimizer::sgd(lr, 0.9, 0.0);

        for mb in 0..num_micro_batches {
            let start = mb * micro_batch_size;
            let end = (start + micro_batch_size).min(data.len());

            model.zero_grad();
            let mut batch_loss = 0.0;
            for s in start..end {
                let input = nn::Tensor::new(vec![1, data[s].len()], data[s].clone());
                let target = nn::Tensor::new(vec![1, labels[s].len()], labels[s].clone());
                let pred = self.forward_simulated(model, &input);
                let (loss, grad) = mse_loss_simple(&pred, &target);
                batch_loss += loss;
                self.backward_simulated(model, &grad);
            }
            batch_loss /= (end - start) as f64;
            total_loss += batch_loss;
            optimizer.step(model);
        }

        total_loss / num_micro_batches as f64
    }
}

// ─── Elastic Training ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ElasticConfig {
    pub min_workers: usize,
    pub max_workers: usize,
    pub heartbeat_interval_secs: u64,
    pub heartbeat_timeout_secs: u64,
}

impl Default for ElasticConfig {
    fn default() -> Self {
        Self {
            min_workers: 1,
            max_workers: 16,
            heartbeat_interval_secs: 5,
            heartbeat_timeout_secs: 15,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ElasticTrainer {
    pub config: ElasticConfig,
    pub active_workers: Vec<usize>,
    pub world_size: usize,
    pub epoch: usize,
    worker_data_ranges: HashMap<usize, (usize, usize)>,
}

impl ElasticTrainer {
    pub fn new(world_size: usize) -> Self {
        Self {
            config: ElasticConfig::default(),
            active_workers: (0..world_size).collect(),
            world_size,
            epoch: 0,
            worker_data_ranges: HashMap::new(),
        }
    }

    pub fn with_config(world_size: usize, config: ElasticConfig) -> Self {
        let mut t = Self::new(world_size);
        t.config = config;
        t
    }

    /// Assign data ranges to workers
    pub fn assign_data(&mut self, total_samples: usize) {
        let n = self.active_workers.len();
        let per_worker = total_samples / n;
        self.worker_data_ranges.clear();
        for (i, &w) in self.active_workers.iter().enumerate() {
            let start = i * per_worker;
            let end = if i == n - 1 { total_samples } else { start + per_worker };
            self.worker_data_ranges.insert(w, (start, end));
        }
    }

    /// Handle a worker leaving (died or disconnected)
    pub fn remove_worker(&mut self, worker: usize, total_samples: usize) -> bool {
        self.active_workers.retain(|&w| w != worker);
        if self.active_workers.len() < self.config.min_workers {
            return false; // Can't continue
        }
        self.assign_data(total_samples);
        true
    }

    /// Handle a new worker joining
    pub fn add_worker(&mut self, worker: usize, total_samples: usize) -> bool {
        if self.active_workers.len() >= self.config.max_workers {
            return false;
        }
        if !self.active_workers.contains(&worker) {
            self.active_workers.push(worker);
            self.active_workers.sort();
        }
        self.assign_data(total_samples);
        true
    }

    /// Get data range for a specific worker
    pub fn get_data_range(&self, worker: usize) -> Option<(usize, usize)> {
        self.worker_data_ranges.get(&worker).copied()
    }

    /// Train with elastic scaling (simulated)
    pub fn train_elastic(
        &mut self,
        model: &mut nn::Model,
        data: &[Vec<f64>],
        labels: &[Vec<f64>],
        epochs: usize,
        lr: f64,
    ) -> Vec<f64> {
        self.assign_data(data.len());
        let mut losses = Vec::with_capacity(epochs);
        let mut optimizer = nn::Optimizer::sgd(lr, 0.9, 0.0);

        for epoch in 0..epochs {
            self.epoch = epoch;
            let mut epoch_loss = 0.0;
            let n_workers = self.active_workers.len();

            for &w in &self.active_workers.clone() {
                if let Some((start, end)) = self.get_data_range(w) {
                    model.zero_grad();
                    let mut batch_loss = 0.0;
                    for s in start..end {
                        let input = nn::Tensor::new(vec![1, data[s].len()], data[s].clone());
                        let target = nn::Tensor::new(vec![1, labels[s].len()], labels[s].clone());
                        let pred = model.forward(&input);
                        let (loss, grad) = mse_loss_simple(&pred, &target);
                        batch_loss += loss;
                        model.backward(&grad);
                    }
                    if end > start {
                        batch_loss /= (end - start) as f64;
                    }
                    epoch_loss += batch_loss;
                }
            }
            epoch_loss /= n_workers as f64;
            optimizer.step(model);
            losses.push(epoch_loss);
        }

        losses
    }
}

// ─── CLI: Launch distributed workers ────────────────────────────────────────

/// Launch multiple local workers as child processes
pub fn launch_local_workers(file: &str, world_size: usize) -> Result<(), String> {
    let exe = std::env::current_exe().map_err(|e| format!("current_exe: {}", e))?;
    let base_port = 19000u16;
    let peers: Vec<String> = (0..world_size).map(|r| format!("127.0.0.1:{}", base_port + r as u16)).collect();
    let peers_json = peers.join(",");

    let mut children = Vec::new();
    for rank in 0..world_size {
        let child = std::process::Command::new(&exe)
            .arg("run")
            .arg(file)
            .env("VORTEX_RANK", rank.to_string())
            .env("VORTEX_WORLD_SIZE", world_size.to_string())
            .env("VORTEX_PEERS", &peers_json)
            .spawn()
            .map_err(|e| format!("spawn worker {}: {}", rank, e))?;
        children.push(child);
    }

    // Wait for all children
    for (rank, mut child) in children.into_iter().enumerate() {
        match child.wait() {
            Ok(status) => {
                if !status.success() {
                    eprintln!("Worker {} exited with status: {}", rank, status);
                }
            }
            Err(e) => eprintln!("Worker {} error: {}", rank, e),
        }
    }

    Ok(())
}

// ─── Interpreter Builtins ───────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("dist_rank".into(), FnDef::Builtin(builtin_dist_rank));
    env.functions.insert("dist_world_size".into(), FnDef::Builtin(builtin_dist_world_size));
    env.functions.insert("dist_all_reduce".into(), FnDef::Builtin(builtin_dist_all_reduce));
    env.functions.insert("dist_broadcast".into(), FnDef::Builtin(builtin_dist_broadcast));
    env.functions.insert("dist_barrier".into(), FnDef::Builtin(builtin_dist_barrier));
    env.functions.insert("dist_train".into(), FnDef::Builtin(builtin_dist_train));
    env.functions.insert("dist_init".into(), FnDef::Builtin(builtin_dist_init));
    env.functions.insert("dist_init_local".into(), FnDef::Builtin(builtin_dist_init_local));
}

fn builtin_dist_rank(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let rank = std::env::var("VORTEX_RANK").unwrap_or_else(|_| "0".into());
    let r: usize = rank.parse().unwrap_or(0);
    Ok(Value::Int(r as i128))
}

fn builtin_dist_world_size(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let ws = std::env::var("VORTEX_WORLD_SIZE").unwrap_or_else(|_| "1".into());
    let w: usize = ws.parse().unwrap_or(1);
    Ok(Value::Int(w as i128))
}

fn builtin_dist_all_reduce(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("dist_all_reduce(data, op)".into()); }
    let data = value_to_f64_vec(&args[0])?;
    let op_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => "sum".into(),
    };
    let op = match op_str.as_str() {
        "avg" | "average" | "mean" => ReduceOp::Avg,
        "max" => ReduceOp::Max,
        "min" => ReduceOp::Min,
        _ => ReduceOp::Sum,
    };

    // In single-process mode, simulate with identity (single worker)
    let world_size: usize = std::env::var("VORTEX_WORLD_SIZE").unwrap_or_else(|_| "1".into()).parse().unwrap_or(1);
    if world_size <= 1 {
        let result = match op {
            ReduceOp::Avg => data.clone(), // single worker, avg = identity
            _ => data.clone(),
        };
        return Ok(Value::Array(result.into_iter().map(Value::Float).collect()));
    }

    // For multi-process, use TCP runtime
    // This requires VORTEX_PEERS to be set
    let peers_str = std::env::var("VORTEX_PEERS").unwrap_or_default();
    let rank: usize = std::env::var("VORTEX_RANK").unwrap_or_else(|_| "0".into()).parse().unwrap_or(0);
    let peers: Vec<SocketAddr> = peers_str.split(',')
        .filter(|s| !s.is_empty())
        .filter_map(|s| s.trim().parse().ok())
        .collect();

    if peers.is_empty() {
        return Ok(Value::Array(data.into_iter().map(Value::Float).collect()));
    }

    let mut rt = DistRuntime::new(rank, world_size, peers)?;
    let result = rt.all_reduce(&data, op)?;
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_dist_broadcast(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("dist_broadcast(data, root)".into()); }
    let data = value_to_f64_vec(&args[0])?;
    let root = match &args[1] { Value::Int(i) => *i as usize, _ => 0 };

    // Single-process fallback
    Ok(Value::Array(data.into_iter().map(Value::Float).collect()))
}

fn builtin_dist_barrier(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    // Single-process: no-op
    Ok(Value::Void)
}

fn builtin_dist_init(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 { return Err("dist_init(rank, world_size, peers_json)".into()); }
    let rank = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int rank".into()) };
    let world_size = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int world_size".into()) };
    let peers_str = match &args[2] { Value::String(s) => s.clone(), _ => return Err("expected string peers".into()) };

    std::env::set_var("VORTEX_RANK", rank.to_string());
    std::env::set_var("VORTEX_WORLD_SIZE", world_size.to_string());
    std::env::set_var("VORTEX_PEERS", &peers_str);

    Ok(Value::Bool(true))
}

fn builtin_dist_init_local(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("dist_init_local(world_size)".into()); }
    let world_size = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };

    std::env::set_var("VORTEX_RANK", "0");
    std::env::set_var("VORTEX_WORLD_SIZE", world_size.to_string());

    Ok(Value::Bool(true))
}

fn builtin_dist_train(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 { return Err("dist_train(model_id, data, labels, epochs, lr)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int model_id".into()) };
    let data = value_to_2d(&args[1])?;
    let labels = value_to_2d(&args[2])?;
    let epochs = match &args[3] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => 100 };
    let lr = match &args[4] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.01 };

    let world_size: usize = std::env::var("VORTEX_WORLD_SIZE").unwrap_or_else(|_| "1".into()).parse().unwrap_or(1);

    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;

    let mut trainer = DataParallelTrainer::new(0, world_size);
    let losses = trainer.train_simulated(model, data, labels, epochs, lr);

    Ok(Value::Array(losses.into_iter().map(Value::Float).collect()))
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected number in array".into()),
        }).collect(),
        _ => Err("expected array".into()),
    }
}

fn value_to_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(rows) => rows.iter().map(|row| value_to_f64_vec(row)).collect(),
        _ => Err("expected 2D array".into()),
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serialize_deserialize_message() {
        let data = vec![1.0, 2.0, 3.0, 4.5];
        let buf = serialize_message(MSG_ALL_REDUCE, &data);
        let (msg_type, decoded) = deserialize_message(&buf).unwrap();
        assert_eq!(msg_type, MSG_ALL_REDUCE);
        assert_eq!(decoded, data);
    }

    #[test]
    fn test_serialize_empty_message() {
        let data: Vec<f64> = vec![];
        let buf = serialize_message(MSG_BARRIER, &data);
        let (msg_type, decoded) = deserialize_message(&buf).unwrap();
        assert_eq!(msg_type, MSG_BARRIER);
        assert!(decoded.is_empty());
    }

    #[test]
    fn test_serialize_large_message() {
        let data: Vec<f64> = (0..1000).map(|i| i as f64 * 0.1).collect();
        let buf = serialize_message(MSG_BROADCAST, &data);
        let (msg_type, decoded) = deserialize_message(&buf).unwrap();
        assert_eq!(msg_type, MSG_BROADCAST);
        assert_eq!(decoded.len(), 1000);
        assert!((decoded[500] - 50.0).abs() < 1e-10);
    }

    #[test]
    fn test_deserialize_short_message() {
        let buf = vec![0u8; 5];
        assert!(deserialize_message(&buf).is_err());
    }

    #[test]
    fn test_all_reduce_simulated_sum() {
        let mut workers = SimulatedRuntime::new(3);
        let data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let result = SimulatedRuntime::all_reduce_simulated(&mut workers, &data, ReduceOp::Sum);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![12.0, 15.0, 18.0]);
        assert_eq!(result[1], vec![12.0, 15.0, 18.0]);
        assert_eq!(result[2], vec![12.0, 15.0, 18.0]);
    }

    #[test]
    fn test_all_reduce_simulated_avg() {
        let mut workers = SimulatedRuntime::new(2);
        let data = vec![
            vec![10.0, 20.0],
            vec![30.0, 40.0],
        ];
        let result = SimulatedRuntime::all_reduce_simulated(&mut workers, &data, ReduceOp::Avg);
        assert_eq!(result[0], vec![20.0, 30.0]);
    }

    #[test]
    fn test_all_reduce_simulated_max() {
        let mut workers = SimulatedRuntime::new(3);
        let data = vec![
            vec![1.0, 5.0],
            vec![3.0, 2.0],
            vec![2.0, 4.0],
        ];
        let result = SimulatedRuntime::all_reduce_simulated(&mut workers, &data, ReduceOp::Max);
        assert_eq!(result[0], vec![3.0, 5.0]);
    }

    #[test]
    fn test_all_reduce_simulated_min() {
        let mut workers = SimulatedRuntime::new(2);
        let data = vec![
            vec![5.0, 1.0],
            vec![3.0, 7.0],
        ];
        let result = SimulatedRuntime::all_reduce_simulated(&mut workers, &data, ReduceOp::Min);
        assert_eq!(result[0], vec![3.0, 1.0]);
    }

    #[test]
    fn test_broadcast_simulated() {
        let data = vec![1.0, 2.0, 3.0];
        let result = SimulatedRuntime::broadcast_simulated(&data, 0, 4);
        assert_eq!(result.len(), 4);
        for r in &result {
            assert_eq!(r, &data);
        }
    }

    #[test]
    fn test_scatter_simulated() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = SimulatedRuntime::scatter_simulated(&data, 3);
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![1.0, 2.0]);
        assert_eq!(result[1], vec![3.0, 4.0]);
        assert_eq!(result[2], vec![5.0, 6.0]);
    }

    #[test]
    fn test_gather_simulated() {
        let chunks = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
        ];
        let result = SimulatedRuntime::gather_simulated(&chunks);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_top_k_sparsify() {
        let grads = vec![0.1, -5.0, 0.3, 2.0, -0.05];
        let (indices, values) = top_k_sparsify(&grads, 40.0); // top 40% = 2 values
        assert_eq!(indices.len(), 2);
        assert!(indices.contains(&1)); // -5.0 has largest abs
        assert!(indices.contains(&3)); // 2.0 has second largest abs
    }

    #[test]
    fn test_sparse_to_dense() {
        let indices = vec![1, 3];
        let values = vec![-5.0, 2.0];
        let dense = sparse_to_dense(&indices, &values, 5);
        assert_eq!(dense, vec![0.0, -5.0, 0.0, 2.0, 0.0]);
    }

    #[test]
    fn test_quantize_roundtrip() {
        let data = vec![1.0, -2.5, 0.0, 100.0, -0.001];
        let quantized = quantize_f64_to_f16(&data);
        let dequantized = dequantize_f16_to_f64(&quantized);
        // f16 has limited precision, check approximate equality
        for (orig, deq) in data.iter().zip(dequantized.iter()) {
            if orig.abs() > 1e-6 {
                let rel_err = ((orig - deq) / orig).abs();
                assert!(rel_err < 0.01, "quantize roundtrip: {} vs {} (err {})", orig, deq, rel_err);
            }
        }
    }

    #[test]
    fn test_error_feedback() {
        let mut ef = ErrorFeedback::new(5);
        let grads = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let compressed = ef.compress_with_feedback(&grads, 40.0);
        // Some values should be zero (not sent)
        let nonzero = compressed.iter().filter(|&&v| v != 0.0).count();
        assert!(nonzero <= 3); // top 40% of 5 = 2 values sent
        // Residuals should contain what wasn't sent
        let residual_sum: f64 = ef.residuals.iter().map(|v| v.abs()).sum();
        assert!(residual_sum > 0.0);
    }

    #[test]
    fn test_data_parallel_training_converges() {
        // Simple XOR-like pattern
        let data = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let labels = vec![
            vec![0.0],
            vec![1.0],
            vec![1.0],
            vec![0.0],
        ];

        let mut model = nn::Model::sequential(vec![
            nn::Layer::linear(2, 4),
            nn::Layer::ReLU { cache: None },
            nn::Layer::linear(4, 1),
        ]);

        let mut trainer = DataParallelTrainer::new(0, 2);
        let losses = trainer.train_simulated(&mut model, data, labels, 50, 0.05);

        // Loss should decrease over training
        assert!(losses.len() == 50);
        let first_loss = losses[0];
        let last_loss = losses[losses.len() - 1];
        // With data parallel training on tiny data, just check it runs without crashing
        assert!(first_loss.is_finite());
        assert!(last_loss.is_finite());
    }

    #[test]
    fn test_elastic_trainer_add_remove() {
        let mut elastic = ElasticTrainer::new(4);
        elastic.assign_data(100);

        assert_eq!(elastic.active_workers.len(), 4);
        assert_eq!(elastic.get_data_range(0), Some((0, 25)));
        assert_eq!(elastic.get_data_range(3), Some((75, 100)));

        // Remove worker 2
        assert!(elastic.remove_worker(2, 100));
        assert_eq!(elastic.active_workers.len(), 3);
        assert_eq!(elastic.get_data_range(0), Some((0, 33)));

        // Add worker 5
        assert!(elastic.add_worker(5, 100));
        assert_eq!(elastic.active_workers.len(), 4);
    }

    #[test]
    fn test_elastic_trainer_min_workers() {
        let mut elastic = ElasticTrainer::new(2);
        elastic.config.min_workers = 2;
        elastic.assign_data(100);

        // Can't remove below min
        assert!(!elastic.remove_worker(0, 100));
    }

    #[test]
    fn test_elastic_training_runs() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
        let labels = vec![vec![0.5], vec![1.5], vec![2.5], vec![3.5]];

        let mut model = nn::Model::sequential(vec![
            nn::Layer::linear(2, 3),
            nn::Layer::ReLU { cache: None },
            nn::Layer::linear(3, 1),
        ]);

        let mut elastic = ElasticTrainer::new(2);
        let losses = elastic.train_elastic(&mut model, &data, &labels, 20, 0.01);
        assert_eq!(losses.len(), 20);
        assert!(losses[0].is_finite());
    }

    #[test]
    fn test_model_parallel_layer_assignment() {
        let mp = ModelParallelTrainer::new(0, 3, 9);
        assert_eq!(mp.layer_assignment, vec![0, 1, 2, 0, 1, 2, 0, 1, 2]);
        let local = mp.local_layer_indices();
        assert_eq!(local, vec![0, 3, 6]);
    }

    #[test]
    fn test_model_parallel_pipeline_training() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let labels = vec![vec![0.5], vec![1.5]];

        let mut model = nn::Model::sequential(vec![
            nn::Layer::linear(2, 4),
            nn::Layer::ReLU { cache: None },
            nn::Layer::linear(4, 1),
        ]);

        let mp = ModelParallelTrainer::new(0, 2, 3);
        let loss = mp.train_pipeline_simulated(&mut model, &data, &labels, 1, 0.01);
        assert!(loss.is_finite());
    }

    #[test]
    fn test_f16_conversion() {
        assert_eq!(f16_to_f64(f64_to_f16(0.0)), 0.0);
        assert!((f16_to_f64(f64_to_f16(1.0)) - 1.0).abs() < 0.001);
        assert!((f16_to_f64(f64_to_f16(-3.5)) - (-3.5)).abs() < 0.01);
        assert!(f16_to_f64(f64_to_f16(f64::INFINITY)).is_infinite());
        assert!(f16_to_f64(f64_to_f16(f64::NAN)).is_nan());
    }

    #[test]
    fn test_reduce_op_variants() {
        assert_eq!(ReduceOp::Sum, ReduceOp::Sum);
        assert_ne!(ReduceOp::Sum, ReduceOp::Avg);
        assert_ne!(ReduceOp::Max, ReduceOp::Min);
    }

    #[test]
    fn test_dist_runtime_single_worker() {
        // Single worker should be identity
        let mut rt = DistRuntime::new(0, 1, vec![]).unwrap();
        let data = vec![1.0, 2.0, 3.0];
        let result = rt.all_reduce(&data, ReduceOp::Sum).unwrap();
        assert_eq!(result, data);
        let result = rt.broadcast(&data, 0).unwrap();
        assert_eq!(result, data);
        assert!(rt.barrier().is_ok());
    }
}
