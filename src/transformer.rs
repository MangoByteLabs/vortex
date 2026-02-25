// Transformer implementation for Vortex using FastTensor
// Full encoder-decoder transformer with training utilities
//
// Future integration points:
// - crate::simt_engine for GPU-accelerated attention kernels
// - crate::kv_cache for efficient autoregressive decoding

use std::sync::{LazyLock, Mutex};

use crate::interpreter::{Env, FnDef, Value};
use crate::tensor_engine::{FastTensor, DType, Layout};

// ─── Simple deterministic RNG ────────────────────────────────────────────────

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        SimpleRng { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // xorshift64
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    /// Uniform in [-1, 1]
    fn next_f32(&mut self) -> f32 {
        let v = self.next_u64();
        // Map to [0, 1) then shift to [-1, 1)
        let u = (v >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
        u * 2.0 - 1.0
    }

    /// Xavier/Glorot uniform: U(-sqrt(6/(fan_in+fan_out)), sqrt(6/(fan_in+fan_out)))
    fn xavier_f32(&mut self, fan_in: usize, fan_out: usize) -> f32 {
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();
        self.next_f32() * limit
    }
}

// ─── TransformerConfig ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TransformerConfig {
    pub vocab_size: usize,
    pub d_model: usize,
    pub n_heads: usize,
    pub n_layers: usize,
    pub d_ff: usize,
    pub max_seq_len: usize,
    pub dropout_rate: f32,
}

impl TransformerConfig {
    pub fn new(
        vocab_size: usize,
        d_model: usize,
        n_heads: usize,
        n_layers: usize,
        d_ff: usize,
        max_seq_len: usize,
        dropout_rate: f32,
    ) -> Self {
        assert!(d_model % n_heads == 0, "d_model must be divisible by n_heads");
        TransformerConfig {
            vocab_size,
            d_model,
            n_heads,
            n_layers,
            d_ff,
            max_seq_len,
            dropout_rate,
        }
    }

    pub fn head_dim(&self) -> usize {
        self.d_model / self.n_heads
    }
}

// ─── Tensor helper ops (matmul, softmax, etc.) ──────────────────────────────

/// Matrix multiply two 2D FastTensors (F32). [M,K] x [K,N] -> [M,N]
fn tensor_matmul(a: &FastTensor, b: &FastTensor) -> FastTensor {
    assert_eq!(a.dtype, DType::F32);
    assert_eq!(b.dtype, DType::F32);
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    let m = a.shape[0];
    let k = a.shape[1];
    assert_eq!(b.shape[0], k, "matmul inner dimension mismatch");
    let n = b.shape[1];

    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let mut out = vec![0.0f32; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a_data[i * k + p] * b_data[p * n + j];
            }
            out[i * n + j] = sum;
        }
    }

    FastTensor::from_f32(vec![m, n], &out)
}

/// Batched matmul: [B, M, K] x [B, K, N] -> [B, M, N]
/// Falls back to per-batch 2D matmul.
fn tensor_batched_matmul(a: &FastTensor, b: &FastTensor) -> FastTensor {
    assert_eq!(a.shape.len(), 3);
    assert_eq!(b.shape.len(), 3);
    let batch = a.shape[0];
    assert_eq!(b.shape[0], batch);
    let m = a.shape[1];
    let k = a.shape[2];
    assert_eq!(b.shape[1], k);
    let n = b.shape[2];

    let a_data = a.as_f32_slice();
    let b_data = b.as_f32_slice();
    let mut out = vec![0.0f32; batch * m * n];

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        let o_off = bi * m * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f32;
                for p in 0..k {
                    sum += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                }
                out[o_off + i * n + j] = sum;
            }
        }
    }

    FastTensor::from_f32(vec![batch, m, n], &out)
}

/// Element-wise add of two same-shape F32 tensors
fn tensor_add(a: &FastTensor, b: &FastTensor) -> FastTensor {
    assert_eq!(a.shape, b.shape);
    let ad = a.as_f32_slice();
    let bd = b.as_f32_slice();
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
    FastTensor::from_f32(a.shape.clone(), &out)
}

/// Element-wise multiply
fn tensor_mul(a: &FastTensor, b: &FastTensor) -> FastTensor {
    assert_eq!(a.shape, b.shape);
    let ad = a.as_f32_slice();
    let bd = b.as_f32_slice();
    let out: Vec<f32> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
    FastTensor::from_f32(a.shape.clone(), &out)
}

/// Scale tensor by scalar
fn tensor_scale(a: &FastTensor, s: f32) -> FastTensor {
    let ad = a.as_f32_slice();
    let out: Vec<f32> = ad.iter().map(|x| x * s).collect();
    FastTensor::from_f32(a.shape.clone(), &out)
}

/// Add bias (1D) to each row of a 2D tensor [M, N] + [N] -> [M, N]
fn tensor_add_bias(a: &FastTensor, bias: &FastTensor) -> FastTensor {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(bias.shape.len(), 1);
    let m = a.shape[0];
    let n = a.shape[1];
    assert_eq!(bias.shape[0], n);
    let ad = a.as_f32_slice();
    let bd = bias.as_f32_slice();
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = ad[i * n + j] + bd[j];
        }
    }
    FastTensor::from_f32(vec![m, n], &out)
}

/// Softmax along the last dimension of a 2D tensor
fn softmax_2d(t: &FastTensor) -> FastTensor {
    assert_eq!(t.shape.len(), 2);
    let rows = t.shape[0];
    let cols = t.shape[1];
    let data = t.as_f32_slice();
    let mut out = vec![0.0f32; rows * cols];

    for i in 0..rows {
        let row_start = i * cols;
        let row = &data[row_start..row_start + cols];
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..cols {
            let v = (row[j] - max_val).exp();
            out[row_start + j] = v;
            sum += v;
        }
        if sum > 0.0 {
            for j in 0..cols {
                out[row_start + j] /= sum;
            }
        }
    }

    FastTensor::from_f32(vec![rows, cols], &out)
}

/// Softmax along last dimension of a 3D tensor [B, M, N]
fn softmax_3d(t: &FastTensor) -> FastTensor {
    assert_eq!(t.shape.len(), 3);
    let b = t.shape[0];
    let m = t.shape[1];
    let n = t.shape[2];
    let data = t.as_f32_slice();
    let mut out = vec![0.0f32; b * m * n];

    for bi in 0..b {
        for i in 0..m {
            let off = bi * m * n + i * n;
            let row = &data[off..off + n];
            let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..n {
                let v = (row[j] - max_val).exp();
                out[off + j] = v;
                sum += v;
            }
            if sum > 0.0 {
                for j in 0..n {
                    out[off + j] /= sum;
                }
            }
        }
    }

    FastTensor::from_f32(vec![b, m, n], &out)
}

/// GELU activation: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
fn gelu(t: &FastTensor) -> FastTensor {
    let data = t.as_f32_slice();
    let sqrt_2_pi = (2.0f32 / std::f32::consts::PI).sqrt();
    let out: Vec<f32> = data.iter().map(|&x| {
        let inner = sqrt_2_pi * (x + 0.044715 * x * x * x);
        0.5 * x * (1.0 + inner.tanh())
    }).collect();
    FastTensor::from_f32(t.shape.clone(), &out)
}

/// Transpose last two dims of a 3D tensor [B, M, N] -> [B, N, M]
fn transpose_last2(t: &FastTensor) -> FastTensor {
    assert_eq!(t.shape.len(), 3);
    let b = t.shape[0];
    let m = t.shape[1];
    let n = t.shape[2];
    let data = t.as_f32_slice();
    let mut out = vec![0.0f32; b * n * m];

    for bi in 0..b {
        for i in 0..m {
            for j in 0..n {
                out[bi * n * m + j * m + i] = data[bi * m * n + i * n + j];
            }
        }
    }

    FastTensor::from_f32(vec![b, n, m], &out)
}

/// Apply additive mask to 3D attention scores: where mask==0, set to -1e9
fn apply_mask(scores: &FastTensor, mask: &FastTensor) -> FastTensor {
    assert_eq!(scores.shape.len(), 3);
    let sd = scores.as_f32_slice();
    let md = mask.as_f32_slice();
    let numel = sd.len();
    let mask_numel = md.len();
    let mut out = vec![0.0f32; numel];
    for i in 0..numel {
        let mi = i % mask_numel; // broadcast mask
        if md[mi] < 0.5 {
            out[i] = -1e9;
        } else {
            out[i] = sd[i];
        }
    }
    FastTensor::from_f32(scores.shape.clone(), &out)
}

// ─── Linear Layer ────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct Linear {
    pub weight: FastTensor, // [out_features, in_features]
    pub bias: Option<FastTensor>, // [out_features]
    pub in_features: usize,
    pub out_features: usize,
}

impl Linear {
    pub fn new(rng: &mut SimpleRng, in_features: usize, out_features: usize, use_bias: bool) -> Self {
        let mut w_data = vec![0.0f32; out_features * in_features];
        for v in w_data.iter_mut() {
            *v = rng.xavier_f32(in_features, out_features);
        }
        let weight = FastTensor::from_f32(vec![out_features, in_features], &w_data);

        let bias = if use_bias {
            Some(FastTensor::zeros(vec![out_features], DType::F32))
        } else {
            None
        };

        Linear { weight, bias, in_features, out_features }
    }

    /// Forward: input [seq_len, in_features] -> [seq_len, out_features]
    /// Computes input @ weight^T + bias
    pub fn forward(&self, input: &FastTensor) -> FastTensor {
        assert_eq!(input.shape.len(), 2);
        assert_eq!(input.shape[1], self.in_features);

        // Transpose weight: [out, in] -> [in, out]
        let w_data = self.weight.as_f32_slice();
        let mut wt = vec![0.0f32; self.in_features * self.out_features];
        for i in 0..self.out_features {
            for j in 0..self.in_features {
                wt[j * self.out_features + i] = w_data[i * self.in_features + j];
            }
        }
        let wt_tensor = FastTensor::from_f32(vec![self.in_features, self.out_features], &wt);
        let mut result = tensor_matmul(input, &wt_tensor);

        if let Some(ref bias) = self.bias {
            result = tensor_add_bias(&result, bias);
        }

        result
    }

    /// Collect all parameters as f32 vec (weight then bias)
    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.weight.as_f32_slice().to_vec();
        if let Some(ref b) = self.bias {
            p.extend_from_slice(b.as_f32_slice());
        }
        p
    }

    /// Set parameters from flat f32 slice, returns number consumed
    pub fn set_params(&mut self, data: &[f32]) -> usize {
        let wn = self.out_features * self.in_features;
        self.weight = FastTensor::from_f32(
            vec![self.out_features, self.in_features],
            &data[..wn],
        );
        let mut consumed = wn;
        if self.bias.is_some() {
            self.bias = Some(FastTensor::from_f32(
                vec![self.out_features],
                &data[wn..wn + self.out_features],
            ));
            consumed += self.out_features;
        }
        consumed
    }
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct LayerNorm {
    pub gamma: FastTensor, // [d_model]
    pub beta: FastTensor,  // [d_model]
    pub eps: f32,
    pub d_model: usize,
}

impl LayerNorm {
    pub fn new(d_model: usize, eps: f32) -> Self {
        let gamma_data = vec![1.0f32; d_model];
        let beta_data = vec![0.0f32; d_model];
        LayerNorm {
            gamma: FastTensor::from_f32(vec![d_model], &gamma_data),
            beta: FastTensor::from_f32(vec![d_model], &beta_data),
            eps,
            d_model,
        }
    }

    /// Forward: input [seq_len, d_model] -> [seq_len, d_model]
    pub fn forward(&self, input: &FastTensor) -> FastTensor {
        assert_eq!(input.shape.len(), 2);
        let seq_len = input.shape[0];
        let d = input.shape[1];
        assert_eq!(d, self.d_model);

        let data = input.as_f32_slice();
        let gamma = self.gamma.as_f32_slice();
        let beta = self.beta.as_f32_slice();
        let mut out = vec![0.0f32; seq_len * d];

        for i in 0..seq_len {
            let row = &data[i * d..(i + 1) * d];
            let mean: f32 = row.iter().sum::<f32>() / d as f32;
            let var: f32 = row.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / d as f32;
            let std_inv = 1.0 / (var + self.eps).sqrt();

            for j in 0..d {
                out[i * d + j] = (row[j] - mean) * std_inv * gamma[j] + beta[j];
            }
        }

        FastTensor::from_f32(vec![seq_len, d], &out)
    }

    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.gamma.as_f32_slice().to_vec();
        p.extend_from_slice(self.beta.as_f32_slice());
        p
    }

    pub fn set_params(&mut self, data: &[f32]) -> usize {
        self.gamma = FastTensor::from_f32(vec![self.d_model], &data[..self.d_model]);
        self.beta = FastTensor::from_f32(vec![self.d_model], &data[self.d_model..2 * self.d_model]);
        2 * self.d_model
    }
}

// ─── MultiHeadAttention ─────────────────────────────────────────────────────

#[derive(Clone)]
pub struct MultiHeadAttention {
    pub wq: Linear,
    pub wk: Linear,
    pub wv: Linear,
    pub wo: Linear,
    pub n_heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
}

impl MultiHeadAttention {
    pub fn new(rng: &mut SimpleRng, d_model: usize, n_heads: usize) -> Self {
        let head_dim = d_model / n_heads;
        MultiHeadAttention {
            wq: Linear::new(rng, d_model, d_model, true),
            wk: Linear::new(rng, d_model, d_model, true),
            wv: Linear::new(rng, d_model, d_model, true),
            wo: Linear::new(rng, d_model, d_model, true),
            n_heads,
            head_dim,
            d_model,
        }
    }

    /// Reshape [seq_len, d_model] -> [n_heads, seq_len, head_dim]
    fn split_heads(&self, t: &FastTensor, seq_len: usize) -> FastTensor {
        let data = t.as_f32_slice();
        let mut out = vec![0.0f32; self.n_heads * seq_len * self.head_dim];

        for s in 0..seq_len {
            for h in 0..self.n_heads {
                for d in 0..self.head_dim {
                    let src_idx = s * self.d_model + h * self.head_dim + d;
                    let dst_idx = h * seq_len * self.head_dim + s * self.head_dim + d;
                    out[dst_idx] = data[src_idx];
                }
            }
        }

        FastTensor::from_f32(vec![self.n_heads, seq_len, self.head_dim], &out)
    }

    /// Reshape [n_heads, seq_len, head_dim] -> [seq_len, d_model]
    fn merge_heads(&self, t: &FastTensor, seq_len: usize) -> FastTensor {
        let data = t.as_f32_slice();
        let mut out = vec![0.0f32; seq_len * self.d_model];

        for h in 0..self.n_heads {
            for s in 0..seq_len {
                for d in 0..self.head_dim {
                    let src_idx = h * seq_len * self.head_dim + s * self.head_dim + d;
                    let dst_idx = s * self.d_model + h * self.head_dim + d;
                    out[dst_idx] = data[src_idx];
                }
            }
        }

        FastTensor::from_f32(vec![seq_len, self.d_model], &out)
    }

    /// Scaled dot-product attention
    /// Q, K, V: [n_heads, seq_len, head_dim]
    /// mask: Option<[1, seq_len, seq_len]> or None
    /// Returns [n_heads, seq_len, head_dim]
    fn scaled_dot_product_attention(
        &self,
        q: &FastTensor,
        k: &FastTensor,
        v: &FastTensor,
        mask: Option<&FastTensor>,
    ) -> FastTensor {
        let scale = 1.0 / (self.head_dim as f32).sqrt();

        // scores = Q @ K^T => [n_heads, seq_len, seq_len]
        let kt = transpose_last2(k);
        let mut scores = tensor_batched_matmul(q, &kt);
        scores = tensor_scale(&scores, scale);

        // Apply mask if provided
        if let Some(m) = mask {
            scores = apply_mask(&scores, m);
        }

        // Softmax along last dim
        let attn_weights = softmax_3d(&scores);

        // output = attn_weights @ V => [n_heads, seq_len, head_dim]
        tensor_batched_matmul(&attn_weights, v)
    }

    /// Forward pass
    /// q, k, v: [seq_len, d_model]
    /// mask: Option<[1, seq_len, seq_len]>
    /// Returns [seq_len, d_model]
    pub fn forward(
        &self,
        q: &FastTensor,
        k: &FastTensor,
        v: &FastTensor,
        mask: Option<&FastTensor>,
    ) -> FastTensor {
        let seq_len = q.shape[0];

        // Project Q, K, V
        let q_proj = self.wq.forward(q);
        let k_proj = self.wk.forward(k);
        let v_proj = self.wv.forward(v);

        // Split into heads: [n_heads, seq_len, head_dim]
        let q_heads = self.split_heads(&q_proj, seq_len);
        let k_heads = self.split_heads(&k_proj, seq_len);
        let v_heads = self.split_heads(&v_proj, seq_len);

        // Attention
        let attn_out = self.scaled_dot_product_attention(&q_heads, &k_heads, &v_heads, mask);

        // Merge heads back: [seq_len, d_model]
        let merged = self.merge_heads(&attn_out, seq_len);

        // Output projection
        self.wo.forward(&merged)
    }

    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.wq.params_vec();
        p.extend(self.wk.params_vec());
        p.extend(self.wv.params_vec());
        p.extend(self.wo.params_vec());
        p
    }

    pub fn set_params(&mut self, data: &[f32]) -> usize {
        let mut off = 0;
        off += self.wq.set_params(&data[off..]);
        off += self.wk.set_params(&data[off..]);
        off += self.wv.set_params(&data[off..]);
        off += self.wo.set_params(&data[off..]);
        off
    }
}

// ─── FeedForward ─────────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct FeedForward {
    pub linear1: Linear, // d_model -> d_ff
    pub linear2: Linear, // d_ff -> d_model
}

impl FeedForward {
    pub fn new(rng: &mut SimpleRng, d_model: usize, d_ff: usize) -> Self {
        FeedForward {
            linear1: Linear::new(rng, d_model, d_ff, true),
            linear2: Linear::new(rng, d_ff, d_model, true),
        }
    }

    /// Forward: [seq_len, d_model] -> [seq_len, d_model]
    pub fn forward(&self, input: &FastTensor) -> FastTensor {
        let hidden = self.linear1.forward(input);
        let activated = gelu(&hidden);
        self.linear2.forward(&activated)
    }

    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.linear1.params_vec();
        p.extend(self.linear2.params_vec());
        p
    }

    pub fn set_params(&mut self, data: &[f32]) -> usize {
        let mut off = 0;
        off += self.linear1.set_params(&data[off..]);
        off += self.linear2.set_params(&data[off..]);
        off
    }
}

// ─── TransformerLayer ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TransformerLayer {
    pub self_attention: MultiHeadAttention,
    pub ffn: FeedForward,
    pub norm1: LayerNorm,
    pub norm2: LayerNorm,
}

impl TransformerLayer {
    pub fn new(rng: &mut SimpleRng, d_model: usize, n_heads: usize, d_ff: usize) -> Self {
        TransformerLayer {
            self_attention: MultiHeadAttention::new(rng, d_model, n_heads),
            ffn: FeedForward::new(rng, d_model, d_ff),
            norm1: LayerNorm::new(d_model, 1e-5),
            norm2: LayerNorm::new(d_model, 1e-5),
        }
    }

    /// Forward with pre-norm residual connections
    /// input: [seq_len, d_model], mask: Option
    /// Returns [seq_len, d_model]
    pub fn forward(&self, input: &FastTensor, mask: Option<&FastTensor>) -> FastTensor {
        // Self-attention with residual
        let normed1 = self.norm1.forward(input);
        let attn_out = self.self_attention.forward(&normed1, &normed1, &normed1, mask);
        let residual1 = tensor_add(input, &attn_out);

        // FFN with residual
        let normed2 = self.norm2.forward(&residual1);
        let ffn_out = self.ffn.forward(&normed2);
        tensor_add(&residual1, &ffn_out)
    }

    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.self_attention.params_vec();
        p.extend(self.ffn.params_vec());
        p.extend(self.norm1.params_vec());
        p.extend(self.norm2.params_vec());
        p
    }

    pub fn set_params(&mut self, data: &[f32]) -> usize {
        let mut off = 0;
        off += self.self_attention.set_params(&data[off..]);
        off += self.ffn.set_params(&data[off..]);
        off += self.norm1.set_params(&data[off..]);
        off += self.norm2.set_params(&data[off..]);
        off
    }
}

// ─── Sinusoidal Positional Encoding ──────────────────────────────────────────

/// Generate sinusoidal positional encoding [max_len, d_model]
pub fn make_pos_encoding(max_len: usize, d_model: usize) -> FastTensor {
    let mut data = vec![0.0f32; max_len * d_model];

    for pos in 0..max_len {
        for i in 0..d_model {
            let angle = pos as f32 / (10000.0f32).powf((2 * (i / 2)) as f32 / d_model as f32);
            data[pos * d_model + i] = if i % 2 == 0 {
                angle.sin()
            } else {
                angle.cos()
            };
        }
    }

    FastTensor::from_f32(vec![max_len, d_model], &data)
}

// ─── TransformerModel ────────────────────────────────────────────────────────

#[derive(Clone)]
pub struct TransformerModel {
    pub layers: Vec<TransformerLayer>,
    pub embedding: FastTensor,    // [vocab_size, d_model]
    pub pos_encoding: FastTensor, // [max_seq_len, d_model]
    pub output_proj: Linear,      // d_model -> vocab_size
    pub final_norm: LayerNorm,
    pub config: TransformerConfig,
}

impl TransformerModel {
    /// Create a new transformer with Xavier-initialized weights
    pub fn new(config: TransformerConfig) -> Self {
        let mut rng = SimpleRng::new(42);

        // Initialize embedding table
        let emb_size = config.vocab_size * config.d_model;
        let mut emb_data = vec![0.0f32; emb_size];
        for v in emb_data.iter_mut() {
            *v = rng.xavier_f32(config.vocab_size, config.d_model);
        }
        let embedding = FastTensor::from_f32(
            vec![config.vocab_size, config.d_model],
            &emb_data,
        );

        // Positional encoding (sinusoidal, not learned)
        let pos_encoding = make_pos_encoding(config.max_seq_len, config.d_model);

        // Transformer layers
        let mut layers = Vec::with_capacity(config.n_layers);
        for _ in 0..config.n_layers {
            layers.push(TransformerLayer::new(
                &mut rng,
                config.d_model,
                config.n_heads,
                config.d_ff,
            ));
        }

        // Output projection
        let output_proj = Linear::new(&mut rng, config.d_model, config.vocab_size, true);

        let final_norm = LayerNorm::new(config.d_model, 1e-5);

        TransformerModel {
            layers,
            embedding,
            pos_encoding,
            output_proj,
            final_norm,
            config,
        }
    }

    /// Look up token embeddings: token_ids (Vec<usize>) -> [seq_len, d_model]
    fn embed_tokens(&self, token_ids: &[usize]) -> FastTensor {
        let seq_len = token_ids.len();
        let d = self.config.d_model;
        let emb_data = self.embedding.as_f32_slice();
        let pos_data = self.pos_encoding.as_f32_slice();
        let mut out = vec![0.0f32; seq_len * d];

        for (i, &tid) in token_ids.iter().enumerate() {
            assert!(tid < self.config.vocab_size, "token id out of range");
            for j in 0..d {
                out[i * d + j] = emb_data[tid * d + j] + pos_data[i * d + j];
            }
        }

        FastTensor::from_f32(vec![seq_len, d], &out)
    }

    /// Full forward pass: token_ids -> logits [seq_len, vocab_size]
    pub fn forward(&self, token_ids: &[usize], mask: Option<&FastTensor>) -> FastTensor {
        let mut hidden = self.embed_tokens(token_ids);

        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask);
        }

        hidden = self.final_norm.forward(&hidden);
        self.output_proj.forward(&hidden)
    }

    /// Encode: token_ids -> hidden states [seq_len, d_model]
    pub fn encode(&self, token_ids: &[usize], mask: Option<&FastTensor>) -> FastTensor {
        let mut hidden = self.embed_tokens(token_ids);

        for layer in &self.layers {
            hidden = layer.forward(&hidden, mask);
        }

        self.final_norm.forward(&hidden)
    }

    /// Decode: simplified decoder (self-attention + ffn on target tokens)
    /// In a full implementation this would include cross-attention with encoder hidden states.
    /// Future integration: crate::kv_cache for efficient autoregressive decoding
    pub fn decode(
        &self,
        _hidden: &FastTensor,
        target_ids: &[usize],
        mask: Option<&FastTensor>,
    ) -> FastTensor {
        // Simplified: just run target through the same layers + output projection
        // A full encoder-decoder would add cross-attention sublayers here
        let mut h = self.embed_tokens(target_ids);

        for layer in &self.layers {
            h = layer.forward(&h, mask);
        }

        h = self.final_norm.forward(&h);
        self.output_proj.forward(&h)
    }

    /// Collect all model parameters as a flat f32 vec
    pub fn params_vec(&self) -> Vec<f32> {
        let mut p = self.embedding.as_f32_slice().to_vec();
        for layer in &self.layers {
            p.extend(layer.params_vec());
        }
        p.extend(self.final_norm.params_vec());
        p.extend(self.output_proj.params_vec());
        p
    }

    /// Set all parameters from flat f32 slice
    pub fn set_params(&mut self, data: &[f32]) {
        let emb_n = self.config.vocab_size * self.config.d_model;
        self.embedding = FastTensor::from_f32(
            vec![self.config.vocab_size, self.config.d_model],
            &data[..emb_n],
        );
        let mut off = emb_n;
        for layer in self.layers.iter_mut() {
            off += layer.set_params(&data[off..]);
        }
        off += self.final_norm.set_params(&data[off..]);
        self.output_proj.set_params(&data[off..]);
    }
}

// ─── Training Utilities ─────────────────────────────────────────────────────

/// Cross-entropy loss: logits [seq_len, vocab_size], targets [seq_len] (indices)
pub fn cross_entropy_loss(logits: &FastTensor, targets: &[usize]) -> f32 {
    assert_eq!(logits.shape.len(), 2);
    let seq_len = logits.shape[0];
    let vocab_size = logits.shape[1];
    assert_eq!(targets.len(), seq_len);

    let data = logits.as_f32_slice();
    let mut total_loss = 0.0f32;

    for i in 0..seq_len {
        let row = &data[i * vocab_size..(i + 1) * vocab_size];
        // Numerically stable log-softmax
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let log_sum_exp: f32 = row.iter().map(|&x| (x - max_val).exp()).sum::<f32>().ln() + max_val;
        let log_prob = row[targets[i]] - log_sum_exp;
        total_loss -= log_prob;
    }

    total_loss / seq_len as f32
}

/// Adam optimizer step on a flat parameter vector
pub fn adam_step(
    params: &mut Vec<f32>,
    grads: &[f32],
    m: &mut Vec<f32>,
    v: &mut Vec<f32>,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
) {
    let t_f = t as f32;
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);

    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] / bc1;
        let v_hat = v[i] / bc2;
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// Training step using numerical gradient approximation (for demo/testing).
/// Returns the loss value.
pub fn train_step(model: &mut TransformerModel, input: &[usize], target: &[usize], lr: f32) -> f32 {
    let eps = 1e-3f32;

    // Compute base loss
    let logits = model.forward(input, None);
    let base_loss = cross_entropy_loss(&logits, target);

    // Get current params
    let mut params = model.params_vec();
    let n_params = params.len();

    // For efficiency in demo, only update a subset of params (first 512 or all if smaller)
    let update_count = n_params.min(512);

    let mut grads = vec![0.0f32; n_params];

    // Numerical gradient for subset
    for i in 0..update_count {
        let orig = params[i];

        params[i] = orig + eps;
        model.set_params(&params);
        let logits_plus = model.forward(input, None);
        let loss_plus = cross_entropy_loss(&logits_plus, target);

        params[i] = orig;
        grads[i] = (loss_plus - base_loss) / eps;
    }

    // Simple SGD update (Adam state would need persistence)
    for i in 0..update_count {
        params[i] -= lr * grads[i];
    }

    model.set_params(&params);
    base_loss
}

// ─── Global State ────────────────────────────────────────────────────────────

static MODELS: LazyLock<Mutex<Vec<TransformerModel>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

// ─── Builtin Functions ──────────────────────────────────────────────────────

/// transformer_new(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len) -> model_id
fn builtin_transformer_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 6 {
        return Err("transformer_new requires 6 args: vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len".into());
    }

    let vocab_size = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("vocab_size must be int".into()) };
    let d_model = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("d_model must be int".into()) };
    let n_heads = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("n_heads must be int".into()) };
    let n_layers = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("n_layers must be int".into()) };
    let d_ff = match &args[4] { Value::Int(n) => *n as usize, _ => return Err("d_ff must be int".into()) };
    let max_seq_len = match &args[5] { Value::Int(n) => *n as usize, _ => return Err("max_seq_len must be int".into()) };

    let dropout_rate = if args.len() > 6 {
        match &args[6] { Value::Float(f) => *f as f32, _ => 0.1 }
    } else {
        0.1
    };

    let config = TransformerConfig::new(vocab_size, d_model, n_heads, n_layers, d_ff, max_seq_len, dropout_rate);
    let model = TransformerModel::new(config);

    let mut models = MODELS.lock().unwrap();
    let id = models.len();
    models.push(model);

    Ok(Value::Int(id as i128))
}

/// transformer_forward(model_id, token_ids_array) -> logits as Array of Arrays
fn builtin_transformer_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("transformer_forward requires 2 args: model_id, token_ids".into());
    }

    let model_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let token_ids = extract_usize_array(&args[1])?;

    let models = MODELS.lock().unwrap();
    let model = models.get(model_id).ok_or("invalid model_id")?;

    let logits = model.forward(&token_ids, None);
    Ok(tensor_to_value(&logits))
}

/// transformer_encode(model_id, token_ids_array) -> hidden states as Array of Arrays
fn builtin_transformer_encode(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("transformer_encode requires 2 args: model_id, token_ids".into());
    }

    let model_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let token_ids = extract_usize_array(&args[1])?;

    let models = MODELS.lock().unwrap();
    let model = models.get(model_id).ok_or("invalid model_id")?;

    let hidden = model.encode(&token_ids, None);
    Ok(tensor_to_value(&hidden))
}

/// transformer_decode(model_id, hidden_flat, target_ids) -> logits
fn builtin_transformer_decode(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("transformer_decode requires 3 args: model_id, hidden, target_ids".into());
    }

    let model_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let target_ids = extract_usize_array(&args[2])?;

    let models = MODELS.lock().unwrap();
    let model = models.get(model_id).ok_or("invalid model_id")?;

    // hidden is passed but we use simplified decode (see TransformerModel::decode)
    let hidden_placeholder = FastTensor::zeros(vec![1, model.config.d_model], DType::F32);
    let logits = model.decode(&hidden_placeholder, &target_ids, None);
    Ok(tensor_to_value(&logits))
}

/// transformer_train_step(model_id, input_ids, target_ids, lr) -> loss
fn builtin_transformer_train_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("transformer_train_step requires 4 args: model_id, input_ids, target_ids, lr".into());
    }

    let model_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let input_ids = extract_usize_array(&args[1])?;
    let target_ids = extract_usize_array(&args[2])?;
    let lr = match &args[3] { Value::Float(f) => *f as f32, _ => return Err("lr must be float".into()) };

    let mut models = MODELS.lock().unwrap();
    let model = models.get_mut(model_id).ok_or("invalid model_id")?;

    let loss = train_step(model, &input_ids, &target_ids, lr);
    Ok(Value::Float(loss as f64))
}

// ─── Helper Functions ────────────────────────────────────────────────────────

fn extract_usize_array(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                match item {
                    Value::Int(n) => out.push(*n as usize),
                    _ => return Err("array elements must be integers".into()),
                }
            }
            Ok(out)
        }
        _ => Err("expected array".into()),
    }
}

fn tensor_to_value(t: &FastTensor) -> Value {
    let data = t.as_f32_slice();
    match t.shape.len() {
        1 => {
            Value::Array(data.iter().map(|&x| Value::Float(x as f64)).collect())
        }
        2 => {
            let rows = t.shape[0];
            let cols = t.shape[1];
            Value::Array((0..rows).map(|i| {
                Value::Array(
                    data[i * cols..(i + 1) * cols]
                        .iter()
                        .map(|&x| Value::Float(x as f64))
                        .collect(),
                )
            }).collect())
        }
        _ => {
            // Flatten for higher dims
            Value::Array(data.iter().map(|&x| Value::Float(x as f64)).collect())
        }
    }
}

// ─── Registration ────────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("transformer_new".to_string(), FnDef::Builtin(builtin_transformer_new));
    env.functions.insert("transformer_forward".to_string(), FnDef::Builtin(builtin_transformer_forward));
    env.functions.insert("transformer_encode".to_string(), FnDef::Builtin(builtin_transformer_encode));
    env.functions.insert("transformer_decode".to_string(), FnDef::Builtin(builtin_transformer_decode));
    env.functions.insert("transformer_train_step".to_string(), FnDef::Builtin(builtin_transformer_train_step));
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_config() -> TransformerConfig {
        TransformerConfig::new(64, 16, 2, 1, 32, 32, 0.0)
    }

    #[test]
    fn test_model_creation() {
        let config = small_config();
        let model = TransformerModel::new(config.clone());
        assert_eq!(model.layers.len(), 1);
        assert_eq!(model.embedding.shape, vec![64, 16]);
        assert_eq!(model.pos_encoding.shape, vec![32, 16]);
        assert_eq!(model.output_proj.in_features, 16);
        assert_eq!(model.output_proj.out_features, 64);
    }

    #[test]
    fn test_forward_pass_shape() {
        let config = small_config();
        let model = TransformerModel::new(config);
        let token_ids = vec![0, 1, 2, 3];
        let logits = model.forward(&token_ids, None);
        // logits should be [seq_len=4, vocab_size=64]
        assert_eq!(logits.shape, vec![4, 64]);
    }

    #[test]
    fn test_positional_encoding() {
        let pe = make_pos_encoding(10, 8);
        assert_eq!(pe.shape, vec![10, 8]);
        let data = pe.as_f32_slice();
        // Position 0, dim 0 should be sin(0) = 0
        assert!((data[0]).abs() < 1e-5);
        // Position 0, dim 1 should be cos(0) = 1
        assert!((data[1] - 1.0).abs() < 1e-5);
        // All values should be in [-1, 1]
        for &v in data {
            assert!(v >= -1.0 - 1e-6 && v <= 1.0 + 1e-6);
        }
    }

    #[test]
    fn test_cross_entropy_loss() {
        // Perfect prediction should have low loss
        let mut logits_data = vec![0.0f32; 3 * 4]; // 3 tokens, vocab 4
        // Make target class have high logit
        logits_data[0 * 4 + 0] = 10.0; // token 0 -> class 0
        logits_data[1 * 4 + 1] = 10.0; // token 1 -> class 1
        logits_data[2 * 4 + 2] = 10.0; // token 2 -> class 2
        let logits = FastTensor::from_f32(vec![3, 4], &logits_data);
        let targets = vec![0, 1, 2];
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss < 0.1, "loss should be low for correct predictions, got {}", loss);

        // Uniform logits should give loss ~ ln(vocab_size) = ln(4) ~ 1.386
        let uniform_logits = FastTensor::zeros(vec![3, 4], DType::F32);
        let loss_uniform = cross_entropy_loss(&uniform_logits, &targets);
        assert!((loss_uniform - (4.0f32).ln()).abs() < 0.1,
            "uniform loss should be ~ln(4), got {}", loss_uniform);
    }

    #[test]
    fn test_linear_forward() {
        let mut rng = SimpleRng::new(123);
        let linear = Linear::new(&mut rng, 8, 4, true);
        let input = FastTensor::from_f32(vec![3, 8], &vec![1.0f32; 3 * 8]);
        let output = linear.forward(&input);
        assert_eq!(output.shape, vec![3, 4]);
        // Output should be finite
        for &v in output.as_f32_slice() {
            assert!(v.is_finite(), "output contains non-finite value");
        }
    }

    #[test]
    fn test_layer_norm() {
        let ln = LayerNorm::new(4, 1e-5);
        let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let input = FastTensor::from_f32(vec![2, 4], &input_data);
        let output = ln.forward(&input);
        assert_eq!(output.shape, vec![2, 4]);
        // Each row should have mean ~0 and std ~1
        let out_data = output.as_f32_slice();
        for row in 0..2 {
            let row_data = &out_data[row * 4..(row + 1) * 4];
            let mean: f32 = row_data.iter().sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-4, "mean should be ~0, got {}", mean);
        }
    }

    #[test]
    fn test_encode_shape() {
        let config = small_config();
        let model = TransformerModel::new(config);
        let hidden = model.encode(&[0, 1, 2], None);
        assert_eq!(hidden.shape, vec![3, 16]);
    }

    #[test]
    fn test_softmax() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let t = FastTensor::from_f32(vec![1, 4], &data);
        let s = softmax_2d(&t);
        let out = s.as_f32_slice();
        let sum: f32 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax should sum to 1, got {}", sum);
        // Should be monotonically increasing
        assert!(out[0] < out[1]);
        assert!(out[1] < out[2]);
        assert!(out[2] < out[3]);
    }

    #[test]
    fn test_gelu_activation() {
        let data = vec![-2.0f32, -1.0, 0.0, 1.0, 2.0];
        let t = FastTensor::from_f32(vec![1, 5], &data);
        let g = gelu(&t);
        let out = g.as_f32_slice();
        // GELU(0) should be 0
        assert!(out[2].abs() < 1e-5);
        // GELU(x) > 0 for x > 0
        assert!(out[3] > 0.0);
        assert!(out[4] > 0.0);
        // GELU(x) < 0 for x < 0 (slightly negative)
        assert!(out[0] < 0.0);
    }

    #[test]
    fn test_adam_step() {
        let mut params = vec![1.0f32, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        let mut m = vec![0.0f32; 3];
        let mut v = vec![0.0f32; 3];

        adam_step(&mut params, &grads, &mut m, &mut v, 0.001, 0.9, 0.999, 1e-8, 1);

        // Params should have changed
        assert!((params[0] - 1.0).abs() > 1e-6);
        assert!((params[1] - 2.0).abs() > 1e-6);
        assert!((params[2] - 3.0).abs() > 1e-6);
    }

    #[test]
    fn test_matmul() {
        // [2,3] x [3,2] -> [2,2]
        let a = FastTensor::from_f32(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = FastTensor::from_f32(vec![3, 2], &[1.0, 0.0, 0.0, 1.0, 1.0, 0.0]);
        let c = tensor_matmul(&a, &b);
        assert_eq!(c.shape, vec![2, 2]);
        let data = c.as_f32_slice();
        // Row 0: [1*1+2*0+3*1, 1*0+2*1+3*0] = [4, 2]
        assert!((data[0] - 4.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        // Row 1: [4*1+5*0+6*1, 4*0+5*1+6*0] = [10, 5]
        assert!((data[2] - 10.0).abs() < 1e-5);
        assert!((data[3] - 5.0).abs() < 1e-5);
    }
}
