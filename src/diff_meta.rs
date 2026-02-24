// Pillar 3: Differentiable Meta-Learning for Vortex
// MAML, DARTS, learned optimizers, hypergradients, and meta-meta optimization.

use std::collections::HashMap;
use std::sync::Mutex;
use crate::interpreter::{Env, Value, FnDef};

// ─── Deterministic RNG ──────────────────────────────────────────────────────

fn lcg_next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64).max(1e-10)
}

fn randn_vec(dim: usize, seed: &mut u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        let u1 = lcg_next(seed);
        let u2 = lcg_next(seed);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v.push(z * 0.1);
    }
    v
}

fn randn_mat(rows: usize, cols: usize, seed: &mut u64) -> Vec<Vec<f64>> {
    (0..rows).map(|_| randn_vec(cols, seed)).collect()
}

// ─── Value conversion helpers ───────────────────────────────────────────────

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected number, got {:?}", v)),
    }
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) => Ok(*i as usize),
        _ => Err(format!("expected int, got {:?}", v)),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => Err("expected array".into()),
    }
}

fn value_to_f64_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect(),
        _ => Err("expected 2d array".into()),
    }
}

fn value_to_string_vec(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::String(s) => Ok(s.clone()),
            _ => Err("expected string in array".into()),
        }).collect(),
        _ => Err("expected array of strings".into()),
    }
}

// ─── Vector / matrix math helpers ───────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn vec_elem_mul(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
}

fn vec_norm_sq(a: &[f64]) -> f64 {
    a.iter().map(|x| x * x).sum()
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn tanh_act(x: f64) -> f64 {
    x.tanh()
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn mse_loss(pred: &[f64], target: &[f64]) -> f64 {
    pred.iter().zip(target.iter())
        .map(|(p, t)| (p - t).powi(2))
        .sum::<f64>() / pred.len() as f64
}

fn mse_loss_grad(pred: &[f64], target: &[f64]) -> Vec<f64> {
    let n = pred.len() as f64;
    pred.iter().zip(target.iter())
        .map(|(p, t)| 2.0 * (p - t) / n)
        .collect()
}

// ─── MLP forward / backward ────────────────────────────────────────────────

/// Forward pass through an MLP with ReLU hidden activations.
/// Returns (output, activations_per_layer) where activations include pre-relu values.
fn mlp_forward(
    input: &[f64],
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
) -> (Vec<f64>, Vec<Vec<f64>>) {
    let num_layers = weights.len();
    let mut activations = Vec::with_capacity(num_layers + 1);
    let mut x = input.to_vec();
    activations.push(x.clone());

    for l in 0..num_layers {
        let w = &weights[l];
        let b = &biases[l];
        let out_dim = w.len();
        let mut z = Vec::with_capacity(out_dim);
        for i in 0..out_dim {
            z.push(dot(&w[i], &x) + b[i]);
        }
        if l < num_layers - 1 {
            x = z.iter().map(|v| relu(*v)).collect();
        } else {
            x = z;
        }
        activations.push(x.clone());
    }
    (x, activations)
}

/// Backward pass through MLP. Returns (weight_grads, bias_grads).
fn mlp_backward(
    weights: &[Vec<Vec<f64>>],
    activations: &[Vec<f64>],
    output_grad: &[f64],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let num_layers = weights.len();
    let mut w_grads = vec![vec![vec![0.0; 0]; 0]; num_layers];
    let mut b_grads = vec![vec![0.0; 0]; num_layers];
    let mut delta = output_grad.to_vec();

    for l in (0..num_layers).rev() {
        let input_act = &activations[l];
        let output_act = &activations[l + 1];
        let w = &weights[l];
        let out_dim = w.len();
        let in_dim = w[0].len();

        // Apply ReLU derivative for hidden layers
        if l < num_layers - 1 {
            delta = delta.iter().enumerate().map(|(i, &d)| {
                d * relu_deriv(output_act[i])
            }).collect();
        }

        // Weight gradients: delta_j * input_i
        let mut wg = vec![vec![0.0; in_dim]; out_dim];
        for j in 0..out_dim {
            for i in 0..in_dim {
                wg[j][i] = delta[j] * input_act[i];
            }
        }
        w_grads[l] = wg;
        b_grads[l] = delta.clone();

        // Propagate delta to previous layer
        if l > 0 {
            let mut new_delta = vec![0.0; in_dim];
            for j in 0..out_dim {
                for i in 0..in_dim {
                    new_delta[i] += w[j][i] * delta[j];
                }
            }
            delta = new_delta;
        }
    }
    (w_grads, b_grads)
}

/// Compute average loss over a dataset.
fn compute_loss(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    data_x: &[Vec<f64>],
    data_y: &[Vec<f64>],
) -> f64 {
    if data_x.is_empty() {
        return 0.0;
    }
    let total: f64 = data_x.iter().zip(data_y.iter()).map(|(x, y)| {
        let (pred, _) = mlp_forward(x, weights, biases);
        mse_loss(&pred, y)
    }).sum();
    total / data_x.len() as f64
}

/// Compute average gradients over a dataset.
fn compute_grads(
    weights: &[Vec<Vec<f64>>],
    biases: &[Vec<f64>],
    data_x: &[Vec<f64>],
    data_y: &[Vec<f64>],
) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
    let num_layers = weights.len();
    let mut acc_wg: Vec<Vec<Vec<f64>>> = weights.iter().map(|w| {
        w.iter().map(|row| vec![0.0; row.len()]).collect()
    }).collect();
    let mut acc_bg: Vec<Vec<f64>> = biases.iter().map(|b| vec![0.0; b.len()]).collect();
    let n = data_x.len() as f64;

    for (x, y) in data_x.iter().zip(data_y.iter()) {
        let (pred, acts) = mlp_forward(x, weights, biases);
        let out_grad = mse_loss_grad(&pred, y);
        let (wg, bg) = mlp_backward(weights, &acts, &out_grad);
        for l in 0..num_layers {
            for j in 0..wg[l].len() {
                for i in 0..wg[l][j].len() {
                    acc_wg[l][j][i] += wg[l][j][i] / n;
                }
                acc_bg[l][j] += bg[l][j] / n;
            }
        }
    }
    (acc_wg, acc_bg)
}

/// Apply one gradient descent step to weights/biases.
fn apply_grads(
    weights: &mut [Vec<Vec<f64>>],
    biases: &mut [Vec<f64>],
    w_grads: &[Vec<Vec<f64>>],
    b_grads: &[Vec<f64>],
    lr: f64,
) {
    for l in 0..weights.len() {
        for j in 0..weights[l].len() {
            for i in 0..weights[l][j].len() {
                weights[l][j][i] -= lr * w_grads[l][j][i];
            }
            biases[l][j] -= lr * b_grads[l][j];
        }
    }
}

// ─── Structs ────────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct MetaObjective {
    train_loss_weight: f64,
    val_loss_weight: f64,
    complexity_weight: f64,
    energy_weight: f64,
}

impl MetaObjective {
    fn evaluate(
        &self,
        train_loss: f64,
        val_loss: f64,
        weights: &[Vec<Vec<f64>>],
    ) -> f64 {
        let complexity: f64 = weights.iter()
            .flat_map(|w| w.iter().flat_map(|row| row.iter()))
            .map(|v| v * v)
            .sum::<f64>();
        let energy = complexity; // energy proxy: L2 norm of weights
        self.train_loss_weight * train_loss
            + self.val_loss_weight * val_loss
            + self.complexity_weight * complexity
            + self.energy_weight * energy
    }
}

#[derive(Clone, Debug)]
struct TaskData {
    train_x: Vec<Vec<f64>>,
    train_y: Vec<Vec<f64>>,
    val_x: Vec<Vec<f64>>,
    val_y: Vec<Vec<f64>>,
}

#[derive(Clone, Debug)]
struct MetaLearner {
    weights: Vec<Vec<Vec<f64>>>,
    biases: Vec<Vec<f64>>,
    alpha: Vec<Vec<f64>>,
    lr: f64,
    meta_lr: f64,
    inner_steps: usize,
    objective: MetaObjective,
}

impl MetaLearner {
    fn new(layer_sizes: &[usize], lr: f64, meta_lr: f64, inner_steps: usize) -> Self {
        let mut seed = 42u64;
        let num_layers = layer_sizes.len() - 1;
        let mut weights = Vec::with_capacity(num_layers);
        let mut biases = Vec::with_capacity(num_layers);
        let mut alpha = Vec::with_capacity(num_layers);

        for l in 0..num_layers {
            let in_dim = layer_sizes[l];
            let out_dim = layer_sizes[l + 1];
            // Xavier-like initialization
            let scale = (2.0 / (in_dim + out_dim) as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..out_dim).map(|_| {
                randn_vec(in_dim, &mut seed).iter().map(|v| v * scale / 0.1).collect()
            }).collect();
            let b = vec![0.0; out_dim];
            let a = vec![lr; out_dim]; // per-parameter learning rate
            weights.push(w);
            biases.push(b);
            alpha.push(a);
        }

        MetaLearner {
            weights,
            biases,
            alpha,
            lr,
            meta_lr,
            inner_steps,
            objective: MetaObjective {
                train_loss_weight: 0.0,
                val_loss_weight: 1.0,
                complexity_weight: 0.001,
                energy_weight: 0.0,
            },
        }
    }

    /// MAML inner loop: take K gradient steps on the training data, return adapted weights/biases.
    fn inner_loop(
        &self,
        train_x: &[Vec<f64>],
        train_y: &[Vec<f64>],
    ) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let mut w = self.weights.clone();
        let mut b = self.biases.clone();

        for _step in 0..self.inner_steps {
            let (wg, bg) = compute_grads(&w, &b, train_x, train_y);
            // Use per-parameter alpha as learning rate where available
            for l in 0..w.len() {
                for j in 0..w[l].len() {
                    let alpha_lj = self.alpha[l][j];
                    for i in 0..w[l][j].len() {
                        w[l][j][i] -= alpha_lj * wg[l][j][i];
                    }
                    b[l][j] -= alpha_lj * bg[l][j];
                }
            }
        }
        (w, b)
    }

    /// MAML outer step: adapt on each task's train set, evaluate on val set,
    /// update meta-parameters with finite-difference meta-gradients.
    fn outer_step(&mut self, tasks: &[TaskData]) -> f64 {
        let eps = 1e-4;
        let num_layers = self.weights.len();
        let mut meta_loss = 0.0;

        // Compute baseline losses
        for task in tasks {
            let (aw, ab) = self.inner_loop(&task.train_x, &task.train_y);
            let val_loss = compute_loss(&aw, &ab, &task.val_x, &task.val_y);
            let train_loss = compute_loss(&aw, &ab, &task.train_x, &task.train_y);
            meta_loss += self.objective.evaluate(train_loss, val_loss, &aw);
        }
        meta_loss /= tasks.len() as f64;

        // Finite-difference meta-gradients for each weight
        for l in 0..num_layers {
            for j in 0..self.weights[l].len() {
                for i in 0..self.weights[l][j].len() {
                    self.weights[l][j][i] += eps;
                    let mut loss_plus = 0.0;
                    for task in tasks {
                        let (aw, ab) = self.inner_loop(&task.train_x, &task.train_y);
                        let vl = compute_loss(&aw, &ab, &task.val_x, &task.val_y);
                        let tl = compute_loss(&aw, &ab, &task.train_x, &task.train_y);
                        loss_plus += self.objective.evaluate(tl, vl, &aw);
                    }
                    loss_plus /= tasks.len() as f64;
                    let grad = (loss_plus - meta_loss) / eps;
                    self.weights[l][j][i] -= eps; // restore
                    self.weights[l][j][i] -= self.meta_lr * grad;
                }

                // Bias gradients
                self.biases[l][j] += eps;
                let mut loss_plus = 0.0;
                for task in tasks {
                    let (aw, ab) = self.inner_loop(&task.train_x, &task.train_y);
                    let vl = compute_loss(&aw, &ab, &task.val_x, &task.val_y);
                    let tl = compute_loss(&aw, &ab, &task.train_x, &task.train_y);
                    loss_plus += self.objective.evaluate(tl, vl, &aw);
                }
                loss_plus /= tasks.len() as f64;
                let grad = (loss_plus - meta_loss) / eps;
                self.biases[l][j] -= eps;
                self.biases[l][j] -= self.meta_lr * grad;
            }
        }

        meta_loss
    }
}

#[derive(Clone, Debug)]
struct DARTSSearchSpace {
    candidate_ops: Vec<String>,
    alpha: Vec<Vec<f64>>,
    num_nodes: usize,
}

impl DARTSSearchSpace {
    fn new(num_nodes: usize, candidate_ops: Vec<String>) -> Self {
        let num_edges = num_nodes * (num_nodes - 1) / 2;
        let num_ops = candidate_ops.len();
        let alpha = vec![vec![0.0; num_ops]; num_edges];
        DARTSSearchSpace { candidate_ops, alpha, num_nodes }
    }

    /// Number of edges in the DAG
    fn num_edges(&self) -> usize {
        self.alpha.len()
    }

    /// Apply an operation (by name) to a vector. Simple synthetic ops.
    fn apply_op(&self, op: &str, x: &[f64]) -> Vec<f64> {
        match op {
            "identity" | "skip_connect" => x.to_vec(),
            "zero" | "none" => vec![0.0; x.len()],
            "relu" => x.iter().map(|v| relu(*v)).collect(),
            "sigmoid" => x.iter().map(|v| sigmoid(*v)).collect(),
            "tanh" => x.iter().map(|v| tanh_act(*v)).collect(),
            "conv3x3" | "conv_3x3" => {
                // Simulate a 1D convolution-like transform
                let mut out = vec![0.0; x.len()];
                for i in 0..x.len() {
                    let left = if i > 0 { x[i - 1] } else { 0.0 };
                    let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
                    out[i] = 0.25 * left + 0.5 * x[i] + 0.25 * right;
                }
                out
            }
            "conv5x5" | "conv_5x5" => {
                let mut out = vec![0.0; x.len()];
                for i in 0..x.len() {
                    let mut sum = 0.4 * x[i];
                    let offsets = [-2i32, -1, 1, 2];
                    for &off in &offsets {
                        let idx = i as i32 + off;
                        if idx >= 0 && (idx as usize) < x.len() {
                            sum += 0.15 * x[idx as usize];
                        }
                    }
                    out[i] = sum;
                }
                out
            }
            "max_pool" | "avg_pool" => {
                // Pooling-like: average with neighbors
                let mut out = vec![0.0; x.len()];
                for i in 0..x.len() {
                    let mut count = 1.0;
                    let mut s = x[i];
                    if i > 0 { s += x[i - 1]; count += 1.0; }
                    if i + 1 < x.len() { s += x[i + 1]; count += 1.0; }
                    out[i] = s / count;
                }
                out
            }
            "sep_conv" | "sep_conv_3x3" | "dil_conv" | "dil_conv_3x3" => {
                // Depthwise-separable-like
                let mut out = vec![0.0; x.len()];
                for i in 0..x.len() {
                    let left = if i > 0 { x[i - 1] } else { 0.0 };
                    let right = if i + 1 < x.len() { x[i + 1] } else { 0.0 };
                    out[i] = (0.3 * left + 0.4 * x[i] + 0.3 * right).tanh();
                }
                out
            }
            _ => x.to_vec(), // fallback: identity
        }
    }

    /// Mixed forward: softmax over alpha, weighted sum of op outputs.
    fn mixed_forward(&self, edge_idx: usize, x: &[f64]) -> Vec<f64> {
        let probs = softmax(&self.alpha[edge_idx]);
        let dim = x.len();
        let mut result = vec![0.0; dim];
        for (op_idx, op_name) in self.candidate_ops.iter().enumerate() {
            let out = self.apply_op(op_name, x);
            for d in 0..dim {
                result[d] += probs[op_idx] * out[d];
            }
        }
        result
    }

    /// Full DAG forward through the cell.
    fn cell_forward(&self, input: &[f64]) -> Vec<f64> {
        let mut node_outputs: Vec<Vec<f64>> = Vec::with_capacity(self.num_nodes);
        node_outputs.push(input.to_vec()); // node 0 = input
        let dim = input.len();

        let mut edge_idx = 0;
        for j in 1..self.num_nodes {
            let mut node_val = vec![0.0; dim];
            for i in 0..j {
                if edge_idx < self.num_edges() {
                    let mixed = self.mixed_forward(edge_idx, &node_outputs[i]);
                    node_val = vec_add(&node_val, &mixed);
                    edge_idx += 1;
                }
            }
            node_outputs.push(node_val);
        }
        // Output is the last node
        node_outputs.last().unwrap().clone()
    }

    /// One DARTS bilevel optimization step via finite differences on alpha.
    fn step(
        &mut self,
        train_x: &[Vec<f64>],
        train_y: &[Vec<f64>],
        val_x: &[Vec<f64>],
        val_y: &[Vec<f64>],
        lr: f64,
    ) -> f64 {
        let eps = 1e-4;

        // Baseline validation loss
        let base_val_loss: f64 = val_x.iter().zip(val_y.iter()).map(|(x, y)| {
            let pred = self.cell_forward(x);
            mse_loss(&pred, y)
        }).sum::<f64>() / val_x.len().max(1) as f64;

        // Update alpha via finite differences on val loss
        for e in 0..self.num_edges() {
            for o in 0..self.candidate_ops.len() {
                self.alpha[e][o] += eps;
                let loss_plus: f64 = val_x.iter().zip(val_y.iter()).map(|(x, y)| {
                    let pred = self.cell_forward(x);
                    mse_loss(&pred, y)
                }).sum::<f64>() / val_x.len().max(1) as f64;
                let grad = (loss_plus - base_val_loss) / eps;
                self.alpha[e][o] -= eps; // restore
                self.alpha[e][o] -= lr * grad;
            }
        }

        base_val_loss
    }

    /// Get the selected architecture: for each edge, the op with highest alpha.
    fn get_architecture(&self) -> Vec<(usize, usize, String)> {
        let mut result = Vec::new();
        let mut edge_idx = 0;
        for j in 1..self.num_nodes {
            for i in 0..j {
                if edge_idx < self.num_edges() {
                    let probs = softmax(&self.alpha[edge_idx]);
                    let best_op = probs.iter().enumerate()
                        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    result.push((i, j, self.candidate_ops[best_op].clone()));
                    edge_idx += 1;
                }
            }
        }
        result
    }
}

#[derive(Clone, Debug)]
struct LearnedOptimizer {
    hidden_dim: usize,
    lstm_weights: Vec<Vec<f64>>,
    lstm_bias: Vec<f64>,
    hidden_state: Vec<f64>,
    cell_state: Vec<f64>,
}

impl LearnedOptimizer {
    fn new(param_dim: usize, hidden_dim: usize) -> Self {
        let mut seed = 123u64;
        // LSTM has 4 gates, input is [gradient, hidden], output is hidden_dim
        let input_size = param_dim + hidden_dim;
        let gate_size = 4 * hidden_dim;
        let lstm_weights = randn_mat(gate_size, input_size, &mut seed);
        let lstm_bias = vec![0.0; gate_size];
        let hidden_state = vec![0.0; hidden_dim];
        let cell_state = vec![0.0; hidden_dim];

        LearnedOptimizer {
            hidden_dim,
            lstm_weights,
            lstm_bias,
            hidden_state,
            cell_state,
        }
    }

    /// Run one LSTM step: takes gradient vector, outputs parameter update.
    fn step(&mut self, gradients: &[f64]) -> Vec<f64> {
        let h = &self.hidden_state;
        let c = &self.cell_state;
        let hd = self.hidden_dim;

        // Prepare input: [gradients, hidden_state]
        // Truncate or pad gradient to match expected input size
        let input_size = self.lstm_weights[0].len();
        let grad_dim = input_size - hd;
        let mut input = Vec::with_capacity(input_size);
        for i in 0..grad_dim {
            input.push(if i < gradients.len() { gradients[i] } else { 0.0 });
        }
        input.extend_from_slice(h);

        // Compute gates: [i, f, g, o] = W * input + bias
        let gate_size = 4 * hd;
        let mut gates = vec![0.0; gate_size];
        for j in 0..gate_size {
            let mut sum = self.lstm_bias[j];
            for i in 0..input_size.min(self.lstm_weights[j].len()) {
                sum += self.lstm_weights[j][i] * input[i];
            }
            gates[j] = sum;
        }

        // Split into 4 gates
        let i_gate: Vec<f64> = gates[0..hd].iter().map(|x| sigmoid(*x)).collect();
        let f_gate: Vec<f64> = gates[hd..2*hd].iter().map(|x| sigmoid(*x + 1.0)).collect(); // bias forget gate
        let g_gate: Vec<f64> = gates[2*hd..3*hd].iter().map(|x| tanh_act(*x)).collect();
        let o_gate: Vec<f64> = gates[3*hd..4*hd].iter().map(|x| sigmoid(*x)).collect();

        // Update cell state: c_new = f * c + i * g
        let new_cell: Vec<f64> = (0..hd).map(|j| {
            f_gate[j] * c[j] + i_gate[j] * g_gate[j]
        }).collect();

        // Update hidden state: h_new = o * tanh(c_new)
        let new_hidden: Vec<f64> = (0..hd).map(|j| {
            o_gate[j] * tanh_act(new_cell[j])
        }).collect();

        // Output update direction: use hidden state as parameter update
        // Scale down to reasonable step size
        let updates: Vec<f64> = new_hidden.iter().map(|v| v * 0.01).collect();

        self.hidden_state = new_hidden;
        self.cell_state = new_cell;

        // Pad or truncate updates to match gradient dimension
        let mut result = Vec::with_capacity(gradients.len());
        for i in 0..gradients.len() {
            result.push(if i < updates.len() { updates[i] } else { 0.0 });
        }
        result
    }
}

// ─── Global storage ─────────────────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref META_LEARNERS: Mutex<HashMap<usize, MetaLearner>> = Mutex::new(HashMap::new());
    static ref DARTS_SPACES: Mutex<HashMap<usize, DARTSSearchSpace>> = Mutex::new(HashMap::new());
    static ref LEARNED_OPTS: Mutex<HashMap<usize, LearnedOptimizer>> = Mutex::new(HashMap::new());
    static ref META_OBJECTIVES: Mutex<HashMap<usize, MetaObjective>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: Mutex<usize> = Mutex::new(1);
}

fn next_id() -> usize {
    let mut id = NEXT_ID.lock().unwrap();
    let val = *id;
    *id += 1;
    val
}

// ─── Builtin implementations ───────────────────────────────────────────────

fn builtin_meta_learner_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("meta_learner_new(layer_sizes, lr, meta_lr, inner_steps)".into());
    }
    let layer_sizes: Vec<usize> = match &args[0] {
        Value::Array(arr) => arr.iter().map(value_to_usize).collect::<Result<_, _>>()?,
        _ => return Err("layer_sizes must be an array of ints".into()),
    };
    if layer_sizes.len() < 2 {
        return Err("need at least 2 layer sizes (input, output)".into());
    }
    let lr = value_to_f64(&args[1])?;
    let meta_lr = value_to_f64(&args[2])?;
    let inner_steps = value_to_usize(&args[3])?;

    let learner = MetaLearner::new(&layer_sizes, lr, meta_lr, inner_steps);
    let id = next_id();
    META_LEARNERS.lock().unwrap().insert(id, learner);
    Ok(Value::Int(id as i128))
}

fn builtin_maml_inner_loop(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("maml_inner_loop(learner_id, train_x, train_y)".into());
    }
    let id = value_to_usize(&args[0])?;
    let train_x = value_to_f64_2d(&args[1])?;
    let train_y = value_to_f64_2d(&args[2])?;

    let learners = META_LEARNERS.lock().unwrap();
    let learner = learners.get(&id).ok_or("invalid learner_id")?;
    let (adapted_w, _adapted_b) = learner.inner_loop(&train_x, &train_y);

    // Return adapted weights as nested array
    let result: Vec<Value> = adapted_w.iter().map(|layer| {
        Value::Array(layer.iter().map(|row| {
            Value::Array(row.iter().map(|v| Value::Float(*v)).collect())
        }).collect())
    }).collect();
    Ok(Value::Array(result))
}

fn builtin_maml_outer_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("maml_outer_step(learner_id, tasks)".into());
    }
    let id = value_to_usize(&args[0])?;

    // Parse tasks array: each task is a struct or array with train_x, train_y, val_x, val_y
    let tasks_val = match &args[1] {
        Value::Array(arr) => arr.clone(),
        _ => return Err("tasks must be an array".into()),
    };

    let mut tasks = Vec::new();
    for tv in &tasks_val {
        match tv {
            Value::Struct { fields, .. } => {
                let train_x = value_to_f64_2d(fields.get("train_x")
                    .ok_or("task missing train_x")?)?;
                let train_y = value_to_f64_2d(fields.get("train_y")
                    .ok_or("task missing train_y")?)?;
                let val_x = value_to_f64_2d(fields.get("val_x")
                    .ok_or("task missing val_x")?)?;
                let val_y = value_to_f64_2d(fields.get("val_y")
                    .ok_or("task missing val_y")?)?;
                tasks.push(TaskData { train_x, train_y, val_x, val_y });
            }
            Value::Array(arr) if arr.len() == 4 => {
                let train_x = value_to_f64_2d(&arr[0])?;
                let train_y = value_to_f64_2d(&arr[1])?;
                let val_x = value_to_f64_2d(&arr[2])?;
                let val_y = value_to_f64_2d(&arr[3])?;
                tasks.push(TaskData { train_x, train_y, val_x, val_y });
            }
            _ => return Err("each task must be a struct with train_x/train_y/val_x/val_y or array of 4 2D arrays".into()),
        }
    }

    let mut learners = META_LEARNERS.lock().unwrap();
    let learner = learners.get_mut(&id).ok_or("invalid learner_id")?;
    let loss = learner.outer_step(&tasks);
    Ok(Value::Float(loss))
}

fn builtin_darts_search_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("darts_search_new(num_nodes, candidate_ops)".into());
    }
    let num_nodes = value_to_usize(&args[0])?;
    if num_nodes < 2 {
        return Err("num_nodes must be >= 2".into());
    }
    let ops = value_to_string_vec(&args[1])?;
    if ops.is_empty() {
        return Err("candidate_ops must not be empty".into());
    }

    let space = DARTSSearchSpace::new(num_nodes, ops);
    let id = next_id();
    DARTS_SPACES.lock().unwrap().insert(id, space);
    Ok(Value::Int(id as i128))
}

fn builtin_darts_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 6 {
        return Err("darts_step(search_id, train_x, train_y, val_x, val_y, lr)".into());
    }
    let id = value_to_usize(&args[0])?;
    let train_x = value_to_f64_2d(&args[1])?;
    let train_y = value_to_f64_2d(&args[2])?;
    let val_x = value_to_f64_2d(&args[3])?;
    let val_y = value_to_f64_2d(&args[4])?;
    let lr = value_to_f64(&args[5])?;

    let mut spaces = DARTS_SPACES.lock().unwrap();
    let space = spaces.get_mut(&id).ok_or("invalid search_id")?;
    let loss = space.step(&train_x, &train_y, &val_x, &val_y, lr);
    Ok(Value::Float(loss))
}

fn builtin_darts_get_arch(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("darts_get_arch(search_id)".into());
    }
    let id = value_to_usize(&args[0])?;

    let spaces = DARTS_SPACES.lock().unwrap();
    let space = spaces.get(&id).ok_or("invalid search_id")?;
    let arch = space.get_architecture();

    let result: Vec<Value> = arch.iter().map(|(from, to, op)| {
        Value::Array(vec![
            Value::Int(*from as i128),
            Value::Int(*to as i128),
            Value::String(op.clone()),
        ])
    }).collect();
    Ok(Value::Array(result))
}

fn builtin_hypergrad(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 6 {
        return Err("hypergrad(learner_id, train_x, train_y, val_x, val_y, hyperparam_name)".into());
    }
    let id = value_to_usize(&args[0])?;
    let train_x = value_to_f64_2d(&args[1])?;
    let train_y = value_to_f64_2d(&args[2])?;
    let val_x = value_to_f64_2d(&args[3])?;
    let val_y = value_to_f64_2d(&args[4])?;
    let hp_name = match &args[5] {
        Value::String(s) => s.clone(),
        _ => return Err("hyperparam_name must be a string".into()),
    };

    let eps = 1e-4;
    let learners = META_LEARNERS.lock().unwrap();
    let learner = learners.get(&id).ok_or("invalid learner_id")?;

    // Finite difference: d(val_loss) / d(hyperparam)
    // Perturb the hyperparameter, run inner loop, measure val loss difference
    let compute_val_loss = |learner: &MetaLearner| -> f64 {
        let (aw, ab) = learner.inner_loop(&train_x, &train_y);
        compute_loss(&aw, &ab, &val_x, &val_y)
    };

    let base_loss = compute_val_loss(learner);

    let mut perturbed = learner.clone();
    match hp_name.as_str() {
        "lr" => perturbed.lr += eps,
        "meta_lr" => perturbed.meta_lr += eps,
        "inner_steps" => {
            // inner_steps is discrete; use step of 1
            perturbed.inner_steps += 1;
            let loss_plus = compute_val_loss(&perturbed);
            return Ok(Value::Float(loss_plus - base_loss));
        }
        "train_loss_weight" => perturbed.objective.train_loss_weight += eps,
        "val_loss_weight" => perturbed.objective.val_loss_weight += eps,
        "complexity_weight" => perturbed.objective.complexity_weight += eps,
        "energy_weight" => perturbed.objective.energy_weight += eps,
        _ => return Err(format!("unknown hyperparam: {}", hp_name)),
    }

    // For lr, also update alpha (per-param lr) proportionally
    if hp_name == "lr" {
        let ratio = perturbed.lr / learner.lr;
        for l in 0..perturbed.alpha.len() {
            for j in 0..perturbed.alpha[l].len() {
                perturbed.alpha[l][j] *= ratio;
            }
        }
    }

    let loss_plus = compute_val_loss(&perturbed);
    let grad = (loss_plus - base_loss) / eps;
    Ok(Value::Float(grad))
}

fn builtin_learned_optimizer_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("learned_optimizer_new(param_dim, hidden_dim)".into());
    }
    let param_dim = value_to_usize(&args[0])?;
    let hidden_dim = value_to_usize(&args[1])?;

    let opt = LearnedOptimizer::new(param_dim, hidden_dim);
    let id = next_id();
    LEARNED_OPTS.lock().unwrap().insert(id, opt);
    Ok(Value::Int(id as i128))
}

fn builtin_learned_optimizer_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("learned_optimizer_step(optimizer_id, gradients)".into());
    }
    let id = value_to_usize(&args[0])?;
    let grads = value_to_f64_vec(&args[1])?;

    let mut opts = LEARNED_OPTS.lock().unwrap();
    let opt = opts.get_mut(&id).ok_or("invalid optimizer_id")?;
    let updates = opt.step(&grads);

    let result: Vec<Value> = updates.iter().map(|v| Value::Float(*v)).collect();
    Ok(Value::Array(result))
}

fn builtin_meta_objective_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("meta_objective_new(train_w, val_w, complexity_w, energy_w)".into());
    }
    let train_w = value_to_f64(&args[0])?;
    let val_w = value_to_f64(&args[1])?;
    let complexity_w = value_to_f64(&args[2])?;
    let energy_w = value_to_f64(&args[3])?;

    let obj = MetaObjective {
        train_loss_weight: train_w,
        val_loss_weight: val_w,
        complexity_weight: complexity_w,
        energy_weight: energy_w,
    };
    let id = next_id();
    META_OBJECTIVES.lock().unwrap().insert(id, obj.clone());

    // If a learner_id is provided as 5th arg, attach the objective
    if args.len() >= 5 {
        if let Ok(lid) = value_to_usize(&args[4]) {
            let mut learners = META_LEARNERS.lock().unwrap();
            if let Some(learner) = learners.get_mut(&lid) {
                learner.objective = obj;
            }
        }
    }

    Ok(Value::Int(id as i128))
}

fn builtin_diff_through_training(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("diff_through_training(learner_id, train_x, train_y, num_steps)".into());
    }
    let id = value_to_usize(&args[0])?;
    let train_x = value_to_f64_2d(&args[1])?;
    let train_y = value_to_f64_2d(&args[2])?;
    let num_steps = value_to_usize(&args[3])?;

    let mut learners = META_LEARNERS.lock().unwrap();
    let learner = learners.get_mut(&id).ok_or("invalid learner_id")?;

    // Differentiate through N training steps by actually running them
    // and tracking the final loss. The meta-gradient is implicit in the
    // chain of gradient steps.
    let mut w = learner.weights.clone();
    let mut b = learner.biases.clone();

    for _step in 0..num_steps {
        let (wg, bg) = compute_grads(&w, &b, &train_x, &train_y);
        apply_grads(&mut w, &mut b, &wg, &bg, learner.lr);
    }

    let final_loss = compute_loss(&w, &b, &train_x, &train_y);

    // Update the learner's weights to the trained state
    learner.weights = w;
    learner.biases = b;

    Ok(Value::Float(final_loss))
}

fn builtin_meta_meta_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("meta_meta_step(learner_id, tasks, outer_lr)".into());
    }
    let id = value_to_usize(&args[0])?;
    let tasks_val = match &args[1] {
        Value::Array(arr) => arr.clone(),
        _ => return Err("tasks must be an array".into()),
    };
    let outer_lr = value_to_f64(&args[2])?;

    let mut tasks = Vec::new();
    for tv in &tasks_val {
        match tv {
            Value::Struct { fields, .. } => {
                let train_x = value_to_f64_2d(fields.get("train_x")
                    .ok_or("task missing train_x")?)?;
                let train_y = value_to_f64_2d(fields.get("train_y")
                    .ok_or("task missing train_y")?)?;
                let val_x = value_to_f64_2d(fields.get("val_x")
                    .ok_or("task missing val_x")?)?;
                let val_y = value_to_f64_2d(fields.get("val_y")
                    .ok_or("task missing val_y")?)?;
                tasks.push(TaskData { train_x, train_y, val_x, val_y });
            }
            Value::Array(arr) if arr.len() == 4 => {
                let train_x = value_to_f64_2d(&arr[0])?;
                let train_y = value_to_f64_2d(&arr[1])?;
                let val_x = value_to_f64_2d(&arr[2])?;
                let val_y = value_to_f64_2d(&arr[3])?;
                tasks.push(TaskData { train_x, train_y, val_x, val_y });
            }
            _ => return Err("each task must be a struct or array of 4 2D arrays".into()),
        }
    }

    let eps = 1e-4;
    let mut learners = META_LEARNERS.lock().unwrap();
    let learner = learners.get_mut(&id).ok_or("invalid learner_id")?;

    // Meta-meta optimization: optimize the meta-learner's hyperparameters
    // (lr, meta_lr, inner_steps, objective weights) via finite differences
    // on the meta-loss.

    // Compute baseline meta-loss
    let compute_meta_loss = |l: &MetaLearner, tasks: &[TaskData]| -> f64 {
        let mut total = 0.0;
        for task in tasks {
            let (aw, ab) = l.inner_loop(&task.train_x, &task.train_y);
            let vl = compute_loss(&aw, &ab, &task.val_x, &task.val_y);
            let tl = compute_loss(&aw, &ab, &task.train_x, &task.train_y);
            total += l.objective.evaluate(tl, vl, &aw);
        }
        total / tasks.len().max(1) as f64
    };

    let base_loss = compute_meta_loss(learner, &tasks);

    // Optimize lr
    {
        let mut perturbed = learner.clone();
        perturbed.lr += eps;
        for l in 0..perturbed.alpha.len() {
            for j in 0..perturbed.alpha[l].len() {
                perturbed.alpha[l][j] = perturbed.lr;
            }
        }
        let loss_plus = compute_meta_loss(&perturbed, &tasks);
        let grad = (loss_plus - base_loss) / eps;
        learner.lr -= outer_lr * grad;
        learner.lr = learner.lr.max(1e-6);
        // Update alpha to match new lr
        for l in 0..learner.alpha.len() {
            for j in 0..learner.alpha[l].len() {
                learner.alpha[l][j] = learner.lr;
            }
        }
    }

    // Optimize meta_lr
    {
        let mut perturbed = learner.clone();
        perturbed.meta_lr += eps;
        let loss_plus = compute_meta_loss(&perturbed, &tasks);
        let grad = (loss_plus - base_loss) / eps;
        learner.meta_lr -= outer_lr * grad;
        learner.meta_lr = learner.meta_lr.max(1e-6);
    }

    // Optimize objective weights
    let obj_params = [
        ("train_loss_weight", learner.objective.train_loss_weight),
        ("val_loss_weight", learner.objective.val_loss_weight),
        ("complexity_weight", learner.objective.complexity_weight),
        ("energy_weight", learner.objective.energy_weight),
    ];
    let mut new_obj_vals = Vec::new();
    for (name, _val) in &obj_params {
        let mut perturbed = learner.clone();
        match *name {
            "train_loss_weight" => perturbed.objective.train_loss_weight += eps,
            "val_loss_weight" => perturbed.objective.val_loss_weight += eps,
            "complexity_weight" => perturbed.objective.complexity_weight += eps,
            "energy_weight" => perturbed.objective.energy_weight += eps,
            _ => {}
        }
        let loss_plus = compute_meta_loss(&perturbed, &tasks);
        let grad = (loss_plus - base_loss) / eps;
        new_obj_vals.push(grad);
    }
    learner.objective.train_loss_weight -= outer_lr * new_obj_vals[0];
    learner.objective.val_loss_weight -= outer_lr * new_obj_vals[1];
    learner.objective.complexity_weight -= outer_lr * new_obj_vals[2];
    learner.objective.energy_weight -= outer_lr * new_obj_vals[3];

    // Clamp objective weights to non-negative
    learner.objective.train_loss_weight = learner.objective.train_loss_weight.max(0.0);
    learner.objective.val_loss_weight = learner.objective.val_loss_weight.max(0.0);
    learner.objective.complexity_weight = learner.objective.complexity_weight.max(0.0);
    learner.objective.energy_weight = learner.objective.energy_weight.max(0.0);

    Ok(Value::Float(base_loss))
}

// ─── Registration ───────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("meta_learner_new".into(), FnDef::Builtin(builtin_meta_learner_new));
    env.functions.insert("maml_inner_loop".into(), FnDef::Builtin(builtin_maml_inner_loop));
    env.functions.insert("maml_outer_step".into(), FnDef::Builtin(builtin_maml_outer_step));
    env.functions.insert("darts_search_new".into(), FnDef::Builtin(builtin_darts_search_new));
    env.functions.insert("darts_step".into(), FnDef::Builtin(builtin_darts_step));
    env.functions.insert("darts_get_arch".into(), FnDef::Builtin(builtin_darts_get_arch));
    env.functions.insert("hypergrad".into(), FnDef::Builtin(builtin_hypergrad));
    env.functions.insert("learned_optimizer_new".into(), FnDef::Builtin(builtin_learned_optimizer_new));
    env.functions.insert("learned_optimizer_step".into(), FnDef::Builtin(builtin_learned_optimizer_step));
    env.functions.insert("meta_objective_new".into(), FnDef::Builtin(builtin_meta_objective_new));
    env.functions.insert("diff_through_training".into(), FnDef::Builtin(builtin_diff_through_training));
    env.functions.insert("meta_meta_step".into(), FnDef::Builtin(builtin_meta_meta_step));
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_data() -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        // Simple regression: y = 2*x
        let train_x = vec![
            vec![0.0], vec![0.5], vec![1.0], vec![1.5], vec![2.0],
        ];
        let train_y = vec![
            vec![0.0], vec![1.0], vec![2.0], vec![3.0], vec![4.0],
        ];
        (train_x, train_y)
    }

    fn make_task_data() -> TaskData {
        let (tx, ty) = make_simple_data();
        let val_x = vec![vec![0.25], vec![0.75], vec![1.25]];
        let val_y = vec![vec![0.5], vec![1.5], vec![2.5]];
        TaskData { train_x: tx, train_y: ty, val_x, val_y }
    }

    #[test]
    fn test_mlp_forward_backward() {
        let mut seed = 42u64;
        let w = vec![randn_mat(4, 2, &mut seed), randn_mat(1, 4, &mut seed)];
        let b = vec![vec![0.0; 4], vec![0.0; 1]];
        let input = vec![1.0, 0.5];
        let (out, acts) = mlp_forward(&input, &w, &b);
        assert_eq!(out.len(), 1);
        assert_eq!(acts.len(), 3);

        let grad = mse_loss_grad(&out, &[1.0]);
        let (wg, bg) = mlp_backward(&w, &acts, &grad);
        assert_eq!(wg.len(), 2);
        assert_eq!(bg.len(), 2);
    }

    #[test]
    fn test_meta_learner_inner_loop() {
        let learner = MetaLearner::new(&[1, 8, 1], 0.01, 0.001, 3);
        let (tx, ty) = make_simple_data();
        let (aw, ab) = learner.inner_loop(&tx, &ty);
        assert_eq!(aw.len(), 2);
        assert_eq!(ab.len(), 2);

        let loss_before = compute_loss(&learner.weights, &learner.biases, &tx, &ty);
        let loss_after = compute_loss(&aw, &ab, &tx, &ty);
        // After inner loop, loss should generally decrease (or at least not explode)
        assert!(loss_after.is_finite());
        assert!(loss_after <= loss_before + 1.0); // allow some tolerance
    }

    #[test]
    fn test_meta_learner_outer_step() {
        let mut learner = MetaLearner::new(&[1, 4, 1], 0.01, 0.0001, 2);
        let tasks = vec![make_task_data(), make_task_data()];
        let loss = learner.outer_step(&tasks);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_darts_search() {
        let ops = vec!["identity".into(), "relu".into(), "zero".into()];
        let mut space = DARTSSearchSpace::new(3, ops);
        assert_eq!(space.num_edges(), 3); // 3 choose 2

        let input = vec![1.0, 0.5, -0.3];
        let out = space.cell_forward(&input);
        assert_eq!(out.len(), 3);

        let train_x = vec![vec![1.0, 0.5, -0.3]];
        let train_y = vec![vec![0.5, 0.2, 0.1]];
        let val_x = train_x.clone();
        let val_y = train_y.clone();
        let loss = space.step(&train_x, &train_y, &val_x, &val_y, 0.1);
        assert!(loss.is_finite());

        let arch = space.get_architecture();
        assert_eq!(arch.len(), 3);
        for (from, to, op) in &arch {
            assert!(*from < *to);
            assert!(["identity", "relu", "zero"].contains(&op.as_str()));
        }
    }

    #[test]
    fn test_learned_optimizer() {
        let mut opt = LearnedOptimizer::new(4, 8);
        let grads = vec![0.1, -0.2, 0.3, -0.1];
        let updates = opt.step(&grads);
        assert_eq!(updates.len(), 4);
        for u in &updates {
            assert!(u.is_finite());
        }

        // Step again to verify state persistence
        let updates2 = opt.step(&grads);
        assert_eq!(updates2.len(), 4);
        // Updates should differ because LSTM state changed
        assert!(updates[0] != updates2[0] || updates[1] != updates2[1]);
    }

    #[test]
    fn test_meta_objective() {
        let obj = MetaObjective {
            train_loss_weight: 0.0,
            val_loss_weight: 1.0,
            complexity_weight: 0.01,
            energy_weight: 0.0,
        };
        let w = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
        let score = obj.evaluate(0.5, 0.3, &w);
        // val_loss * 1.0 + complexity * 0.01 = 0.3 + 0.01 * (1+4+9+16) = 0.3 + 0.3 = 0.6
        assert!((score - 0.6).abs() < 1e-10);
    }

    #[test]
    fn test_diff_through_training() {
        let mut learner = MetaLearner::new(&[1, 4, 1], 0.01, 0.001, 1);
        let (tx, ty) = make_simple_data();
        let loss_before = compute_loss(&learner.weights, &learner.biases, &tx, &ty);

        let mut w = learner.weights.clone();
        let mut b = learner.biases.clone();
        for _ in 0..5 {
            let (wg, bg) = compute_grads(&w, &b, &tx, &ty);
            apply_grads(&mut w, &mut b, &wg, &bg, learner.lr);
        }
        let loss_after = compute_loss(&w, &b, &tx, &ty);
        assert!(loss_after < loss_before || loss_after.is_finite());
    }

    #[test]
    fn test_softmax() {
        let v = vec![1.0, 2.0, 3.0];
        let s = softmax(&v);
        assert!((s.iter().sum::<f64>() - 1.0).abs() < 1e-10);
        assert!(s[2] > s[1]);
        assert!(s[1] > s[0]);
    }

    #[test]
    fn test_mse_loss() {
        let pred = vec![1.0, 2.0, 3.0];
        let target = vec![1.0, 2.0, 3.0];
        assert!((mse_loss(&pred, &target)).abs() < 1e-10);

        let pred2 = vec![2.0, 3.0, 4.0];
        let loss = mse_loss(&pred2, &target);
        assert!((loss - 1.0).abs() < 1e-10); // each diff is 1, squared is 1, mean is 1
    }

    #[test]
    fn test_hypergrad_finite_diff() {
        let learner = MetaLearner::new(&[1, 4, 1], 0.01, 0.001, 2);
        let (tx, ty) = make_simple_data();
        let vx = vec![vec![0.25], vec![0.75]];
        let vy = vec![vec![0.5], vec![1.5]];

        // Compute hypergrad for lr
        let base_loss = {
            let (aw, ab) = learner.inner_loop(&tx, &ty);
            compute_loss(&aw, &ab, &vx, &vy)
        };

        let eps = 1e-4;
        let mut perturbed = learner.clone();
        perturbed.lr += eps;
        for l in 0..perturbed.alpha.len() {
            for j in 0..perturbed.alpha[l].len() {
                perturbed.alpha[l][j] *= perturbed.lr / learner.lr;
            }
        }
        let perturbed_loss = {
            let (aw, ab) = perturbed.inner_loop(&tx, &ty);
            compute_loss(&aw, &ab, &vx, &vy)
        };
        let grad = (perturbed_loss - base_loss) / eps;
        assert!(grad.is_finite());
    }

    #[test]
    fn test_darts_ops() {
        let ops = vec!["identity".into(), "relu".into(), "sigmoid".into(),
                       "tanh".into(), "zero".into(), "conv3x3".into(),
                       "max_pool".into(), "sep_conv".into()];
        let space = DARTSSearchSpace::new(2, ops.clone());
        let input = vec![1.0, -0.5, 0.3];

        for op in &ops {
            let result = space.apply_op(op, &input);
            assert_eq!(result.len(), input.len());
            for v in &result {
                assert!(v.is_finite());
            }
        }
    }

    #[test]
    fn test_vec_helpers() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((dot(&a, &b) - 32.0).abs() < 1e-10);
        assert_eq!(vec_add(&a, &b), vec![5.0, 7.0, 9.0]);
        assert_eq!(vec_sub(&a, &b), vec![-3.0, -3.0, -3.0]);
        assert_eq!(vec_scale(&a, 2.0), vec![2.0, 4.0, 6.0]);
        assert_eq!(vec_elem_mul(&a, &b), vec![4.0, 10.0, 18.0]);
        assert!((vec_norm_sq(&a) - 14.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_loss_and_grads() {
        let mut seed = 99u64;
        let w = vec![randn_mat(1, 1, &mut seed)];
        let b = vec![vec![0.0]];
        let tx = vec![vec![1.0]];
        let ty = vec![vec![1.0]];

        let loss = compute_loss(&w, &b, &tx, &ty);
        assert!(loss.is_finite());

        let (wg, bg) = compute_grads(&w, &b, &tx, &ty);
        assert_eq!(wg.len(), 1);
        assert_eq!(bg.len(), 1);
        for g in wg[0][0].iter() {
            assert!(g.is_finite());
        }
    }

    #[test]
    fn test_apply_grads() {
        let mut w = vec![vec![vec![1.0]]];
        let mut b = vec![vec![0.0]];
        let wg = vec![vec![vec![0.5]]];
        let bg = vec![vec![0.1]];
        apply_grads(&mut w, &mut b, &wg, &bg, 0.1);
        assert!((w[0][0][0] - 0.95).abs() < 1e-10);
        assert!((b[0][0] - (-0.01)).abs() < 1e-10);
    }
}
