//! Heterogeneous computation: routing tokens to fundamentally different expert architectures.
//!
//! Unlike standard MoE (same MLP with different weights), this module provides experts
//! with different compute types: symbolic ALU, retrieval, convolution, multi-step reasoning,
//! dense MLP, and recurrent (RNN-like).

use std::collections::HashMap;

// ─── ExpertType ──────────────────────────────────────────────────────────────

/// The type of computation an expert performs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertType {
    SymbolicALU,
    RetrievalExpert,
    ConvolutionExpert,
    ReasoningExpert,
    DenseExpert,
    RecurrentExpert,
}

impl std::fmt::Display for ExpertType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

// ─── Expert trait ────────────────────────────────────────────────────────────

pub trait Expert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64>;
    fn param_count(&self) -> usize;
    fn expert_type(&self) -> ExpertType;
    fn reset_state(&mut self);
}

// ─── SymbolicALU ─────────────────────────────────────────────────────────────

/// Exact integer arithmetic expert. Converts f64 inputs to i64, performs
/// operations (add, subtract, multiply, compare, sort, argmax, count_nonzero),
/// and converts back. No floating-point rounding errors.
pub struct SymbolicALUExpert {
    width: usize,
    /// Learned operation selector weights (one per op). During forward, the
    /// operation with the highest weight is applied.
    op_weights: Vec<f64>,
}

impl SymbolicALUExpert {
    pub fn new(width: usize) -> Self {
        // 7 operations: add-adjacent, sub-adjacent, mul-adjacent, compare, sort, argmax, count_nonzero
        Self {
            width,
            op_weights: vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        }
    }

    fn to_ints(input: &[f64]) -> Vec<i64> {
        input.iter().map(|v| v.round() as i64).collect()
    }

    pub fn add(a: &[i64], b: &[i64]) -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
    }

    pub fn subtract(a: &[i64], b: &[i64]) -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
    }

    pub fn multiply(a: &[i64], b: &[i64]) -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| x * y).collect()
    }

    pub fn compare(a: &[i64], b: &[i64]) -> Vec<i64> {
        a.iter().zip(b.iter()).map(|(x, y)| {
            if x > y { 1 } else if x == y { 0 } else { -1 }
        }).collect()
    }

    pub fn sort(vals: &[i64]) -> Vec<i64> {
        let mut v = vals.to_vec();
        v.sort();
        v
    }

    pub fn argmax(vals: &[i64]) -> usize {
        vals.iter().enumerate().max_by_key(|(_, v)| *v).map(|(i, _)| i).unwrap_or(0)
    }

    pub fn count_nonzero(vals: &[i64]) -> usize {
        vals.iter().filter(|v| **v != 0).count()
    }
}

impl Expert for SymbolicALUExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let ints = Self::to_ints(input);
        let n = ints.len();
        // Select operation based on max op_weight
        let op_idx = self.op_weights.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);

        let result_ints: Vec<i64> = match op_idx {
            0 => {
                // Add adjacent pairs, pad with last
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(ints[i] + ints[(i + 1) % n]);
                }
                out
            }
            1 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(ints[i] - ints[(i + 1) % n]);
                }
                out
            }
            2 => {
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    out.push(ints[i] * ints[(i + 1) % n]);
                }
                out
            }
            3 => {
                // Compare adjacent
                let mut out = Vec::with_capacity(n);
                for i in 0..n {
                    let a = ints[i];
                    let b = ints[(i + 1) % n];
                    out.push(if a > b { 1 } else if a == b { 0 } else { -1 });
                }
                out
            }
            4 => Self::sort(&ints),
            5 => {
                // Argmax as one-hot
                let am = Self::argmax(&ints);
                let mut out = vec![0i64; n];
                if n > 0 { out[am] = 1; }
                out
            }
            6 => {
                // Count nonzero broadcast
                let cnt = Self::count_nonzero(&ints) as i64;
                vec![cnt; n]
            }
            _ => ints.clone(),
        };

        result_ints.iter().map(|v| *v as f64).collect()
    }

    fn param_count(&self) -> usize {
        self.op_weights.len()
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::SymbolicALU
    }

    fn reset_state(&mut self) {}
}

// ─── RetrievalExpert ─────────────────────────────────────────────────────────

pub struct RetrievalExpert {
    width: usize,
    /// Key-value memory
    pub memory: Vec<(Vec<f64>, Vec<f64>)>,
    top_k: usize,
}

impl RetrievalExpert {
    pub fn new(width: usize) -> Self {
        Self {
            width,
            memory: Vec::new(),
            top_k: 3,
        }
    }

    pub fn store(&mut self, key: Vec<f64>, value: Vec<f64>) {
        self.memory.push((key, value));
    }

    fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 { return 0.0; }
        dot / (na * nb)
    }

    pub fn retrieve(&self, query: &[f64], top_k: usize) -> Vec<(Vec<f64>, f64)> {
        let mut scored: Vec<(usize, f64)> = self.memory.iter().enumerate()
            .map(|(i, (k, _))| (i, Self::cosine_similarity(query, k)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter()
            .map(|(i, sim)| (self.memory[i].1.clone(), sim))
            .collect()
    }
}

impl Expert for RetrievalExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if self.memory.is_empty() {
            // No memory yet; store this input as identity mapping and return it
            self.store(input.to_vec(), input.to_vec());
            return input.to_vec();
        }
        let results = self.retrieve(input, self.top_k);
        // Weighted average of retrieved values
        let mut output = vec![0.0f64; self.width.max(input.len())];
        let mut total_weight = 0.0f64;
        for (val, sim) in &results {
            let w = sim.max(0.0);
            total_weight += w;
            for (i, v) in val.iter().enumerate() {
                if i < output.len() {
                    output[i] += v * w;
                }
            }
        }
        if total_weight > 1e-12 {
            for v in output.iter_mut() {
                *v /= total_weight;
            }
        }
        output.truncate(input.len());
        output
    }

    fn param_count(&self) -> usize {
        self.memory.len() * self.width * 2
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::RetrievalExpert
    }

    fn reset_state(&mut self) {
        self.memory.clear();
    }
}

// ─── ConvolutionExpert ───────────────────────────────────────────────────────

pub struct ConvolutionExpert {
    width: usize,
    kernel: Vec<f64>,
    bias: f64,
}

impl ConvolutionExpert {
    pub fn new(width: usize, kernel_size: usize) -> Self {
        // Initialize with simple averaging kernel
        let val = 1.0 / kernel_size as f64;
        Self {
            width,
            kernel: vec![val; kernel_size],
            bias: 0.0,
        }
    }
}

impl Expert for ConvolutionExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len();
        let k = self.kernel.len();
        let pad = k / 2;
        let mut output = Vec::with_capacity(n);
        for i in 0..n {
            let mut sum = self.bias;
            for j in 0..k {
                let idx = i as isize + j as isize - pad as isize;
                let val = if idx >= 0 && (idx as usize) < n {
                    input[idx as usize]
                } else {
                    0.0
                };
                sum += val * self.kernel[j];
            }
            output.push(sum);
        }
        output
    }

    fn param_count(&self) -> usize {
        self.kernel.len() + 1 // kernel + bias
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::ConvolutionExpert
    }

    fn reset_state(&mut self) {}
}

// ─── ReasoningExpert ─────────────────────────────────────────────────────────

/// Multi-step chain-of-thought expert. Runs N internal forward passes (using a
/// small MLP) before producing output.
pub struct ReasoningExpert {
    width: usize,
    n_steps: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl ReasoningExpert {
    pub fn new(width: usize, n_steps: usize) -> Self {
        // Simple identity-ish initialization with small perturbations
        let mut weights = Vec::with_capacity(width);
        for i in 0..width {
            let mut row = vec![0.0f64; width];
            row[i] = 1.0; // identity
            // small perturbation
            for j in 0..width {
                row[j] += 0.01 * ((i * 7 + j * 13) % 17) as f64 / 17.0 - 0.005;
            }
            weights.push(row);
        }
        Self {
            width,
            n_steps,
            weights,
            biases: vec![0.0; width],
        }
    }

    fn relu(x: f64) -> f64 {
        x.max(0.0)
    }

    fn single_pass(&self, input: &[f64]) -> Vec<f64> {
        let n = input.len().min(self.width);
        let mut output = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = self.biases[i];
            for j in 0..n {
                sum += input[j] * self.weights[i][j];
            }
            output[i] = Self::relu(sum);
        }
        output
    }
}

impl Expert for ReasoningExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let mut state = input.to_vec();
        for _ in 0..self.n_steps {
            state = self.single_pass(&state);
        }
        state
    }

    fn param_count(&self) -> usize {
        self.width * self.width + self.width
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::ReasoningExpert
    }

    fn reset_state(&mut self) {}
}

// ─── DenseExpert ─────────────────────────────────────────────────────────────

pub struct DenseExpert {
    width: usize,
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl DenseExpert {
    pub fn new(width: usize) -> Self {
        let mut weights = Vec::with_capacity(width);
        for i in 0..width {
            let mut row = vec![0.0f64; width];
            row[i] = 1.0;
            weights.push(row);
        }
        Self {
            width,
            weights,
            biases: vec![0.0; width],
        }
    }
}

impl Expert for DenseExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len().min(self.width);
        let mut output = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = self.biases[i];
            for j in 0..n {
                sum += input[j] * self.weights[i][j];
            }
            output[i] = sum.max(0.0); // ReLU
        }
        output
    }

    fn param_count(&self) -> usize {
        self.width * self.width + self.width
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::DenseExpert
    }

    fn reset_state(&mut self) {}
}

// ─── RecurrentExpert ─────────────────────────────────────────────────────────

pub struct RecurrentExpert {
    width: usize,
    hidden: Vec<f64>,
    w_ih: Vec<Vec<f64>>, // input -> hidden
    w_hh: Vec<Vec<f64>>, // hidden -> hidden
    bias: Vec<f64>,
}

impl RecurrentExpert {
    pub fn new(width: usize) -> Self {
        let init = |i: usize, j: usize| {
            if i == j { 0.5 } else { 0.01 * ((i * 7 + j * 13) % 11) as f64 / 11.0 }
        };
        let w_ih: Vec<Vec<f64>> = (0..width).map(|i| (0..width).map(|j| init(i, j)).collect()).collect();
        let w_hh: Vec<Vec<f64>> = (0..width).map(|i| (0..width).map(|j| if i == j { 0.9 } else { 0.0 }).collect()).collect();
        Self {
            width,
            hidden: vec![0.0; width],
            w_ih,
            w_hh,
            bias: vec![0.0; width],
        }
    }
}

impl Expert for RecurrentExpert {
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let n = input.len().min(self.width);
        let mut new_hidden = vec![0.0f64; n];
        for i in 0..n {
            let mut sum = self.bias[i];
            for j in 0..n {
                sum += input[j] * self.w_ih[i][j];
                sum += self.hidden[j] * self.w_hh[i][j];
            }
            new_hidden[i] = sum.tanh();
        }
        self.hidden = new_hidden.clone();
        new_hidden
    }

    fn param_count(&self) -> usize {
        self.width * self.width * 2 + self.width
    }

    fn expert_type(&self) -> ExpertType {
        ExpertType::RecurrentExpert
    }

    fn reset_state(&mut self) {
        self.hidden = vec![0.0; self.width];
    }
}

// ─── HeterogeneousRouter ────────────────────────────────────────────────────

pub struct HeterogeneousRouter {
    pub router_weights: Vec<Vec<f64>>,
    n_experts: usize,
    top_k: usize,
}

impl HeterogeneousRouter {
    pub fn new(input_dim: usize, n_experts: usize, top_k: usize) -> Self {
        // Initialize router weights with small random-ish values
        let router_weights: Vec<Vec<f64>> = (0..n_experts).map(|e| {
            (0..input_dim).map(|i| {
                0.1 * ((e * 7 + i * 13) % 19) as f64 / 19.0 - 0.05
            }).collect()
        }).collect();
        Self { router_weights, n_experts, top_k }
    }

    fn softmax(logits: &[f64]) -> Vec<f64> {
        let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    /// Gumbel-softmax for differentiable routing.
    /// `temperature` controls sharpness (lower = harder selection).
    fn gumbel_softmax(logits: &[f64], temperature: f64) -> Vec<f64> {
        // Deterministic Gumbel noise approximation using logit index
        let gumbel_noise: Vec<f64> = logits.iter().enumerate().map(|(i, _)| {
            // Pseudo-random but deterministic per-position
            let u = 0.1 + 0.8 * (((i * 2654435761) % 1000) as f64 / 1000.0);
            -((-u.ln()).ln())
        }).collect();

        let perturbed: Vec<f64> = logits.iter().zip(gumbel_noise.iter())
            .map(|(l, g)| (l + g) / temperature)
            .collect();

        Self::softmax(&perturbed)
    }

    pub fn route(&self, input: &[f64]) -> Vec<(usize, f64)> {
        let logits: Vec<f64> = self.router_weights.iter().map(|w| {
            w.iter().zip(input.iter()).map(|(a, b)| a * b).sum()
        }).collect();

        let probs = Self::gumbel_softmax(&logits, 1.0);

        // Top-k selection
        let mut indexed: Vec<(usize, f64)> = probs.into_iter().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        indexed.truncate(self.top_k);
        indexed
    }

    pub fn type_distribution(&self, experts: &[Box<dyn Expert>], batch: &[Vec<f64>]) -> HashMap<ExpertType, usize> {
        let mut counts: HashMap<ExpertType, usize> = HashMap::new();
        for input in batch {
            let routing = self.route(input);
            if let Some((idx, _)) = routing.first() {
                if *idx < experts.len() {
                    let et = experts[*idx].expert_type();
                    *counts.entry(et).or_insert(0) += 1;
                }
            }
        }
        counts
    }
}

// ─── HeteroStats ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct HeteroStats {
    pub type_counts: HashMap<ExpertType, usize>,
    pub total_experts: usize,
    pub total_params: usize,
}

// ─── HeterogeneousLayer ─────────────────────────────────────────────────────

pub struct HeterogeneousLayer {
    pub experts: Vec<Box<dyn Expert>>,
    pub router: HeterogeneousRouter,
    width: usize,
}

impl HeterogeneousLayer {
    pub fn new(width: usize) -> Self {
        Self {
            experts: Vec::new(),
            router: HeterogeneousRouter::new(width, 0, 2),
            width,
        }
    }

    pub fn add_expert(&mut self, expert_type: ExpertType, width: usize) {
        let expert: Box<dyn Expert> = match expert_type {
            ExpertType::SymbolicALU => Box::new(SymbolicALUExpert::new(width)),
            ExpertType::RetrievalExpert => Box::new(RetrievalExpert::new(width)),
            ExpertType::ConvolutionExpert => Box::new(ConvolutionExpert::new(width, 3)),
            ExpertType::ReasoningExpert => Box::new(ReasoningExpert::new(width, 3)),
            ExpertType::DenseExpert => Box::new(DenseExpert::new(width)),
            ExpertType::RecurrentExpert => Box::new(RecurrentExpert::new(width)),
        };
        self.experts.push(expert);
        // Rebuild router with new expert count
        self.router = HeterogeneousRouter::new(self.width, self.experts.len(), 2);
    }

    pub fn remove_expert(&mut self, idx: usize) {
        if idx < self.experts.len() {
            self.experts.remove(idx);
            self.router = HeterogeneousRouter::new(self.width, self.experts.len(), 2);
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        if self.experts.is_empty() {
            return input.to_vec();
        }
        let routing = self.router.route(input);
        let mut output = vec![0.0f64; input.len()];
        let mut total_weight = 0.0f64;
        for (idx, weight) in &routing {
            if *idx < self.experts.len() {
                let expert_output = self.experts[*idx].forward(input);
                total_weight += weight;
                for (i, v) in expert_output.iter().enumerate() {
                    if i < output.len() {
                        output[i] += v * weight;
                    }
                }
            }
        }
        if total_weight > 1e-12 {
            for v in output.iter_mut() {
                *v /= total_weight;
            }
        }
        output
    }

    pub fn forward_batch(&mut self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        // Group by primary expert type for efficiency
        let mut assignments: Vec<(usize, Vec<(usize, f64)>)> = Vec::with_capacity(batch.len());
        for (batch_idx, input) in batch.iter().enumerate() {
            let routing = self.router.route(input);
            assignments.push((batch_idx, routing));
        }

        // Group by primary expert for cache-friendly dispatch
        let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
        for (batch_idx, routing) in &assignments {
            if let Some((expert_idx, _)) = routing.first() {
                groups.entry(*expert_idx).or_default().push(*batch_idx);
            }
        }

        let mut results = vec![vec![]; batch.len()];

        // Process each expert group
        for (_expert_idx, batch_indices) in &groups {
            for &bi in batch_indices {
                let routing = &assignments[bi].1;
                let mut output = vec![0.0f64; batch[bi].len()];
                let mut total_weight = 0.0f64;
                for (eidx, weight) in routing {
                    if *eidx < self.experts.len() {
                        let eo = self.experts[*eidx].forward(&batch[bi]);
                        total_weight += weight;
                        for (i, v) in eo.iter().enumerate() {
                            if i < output.len() {
                                output[i] += v * weight;
                            }
                        }
                    }
                }
                if total_weight > 1e-12 {
                    for v in output.iter_mut() {
                        *v /= total_weight;
                    }
                }
                results[bi] = output;
            }
        }

        // Handle any unrouted items
        for (i, r) in results.iter_mut().enumerate() {
            if r.is_empty() {
                *r = batch[i].clone();
            }
        }

        results
    }

    pub fn stats(&self) -> HeteroStats {
        let mut type_counts: HashMap<ExpertType, usize> = HashMap::new();
        let mut total_params = 0;
        for e in &self.experts {
            *type_counts.entry(e.expert_type()).or_insert(0) += 1;
            total_params += e.param_count();
        }
        HeteroStats {
            type_counts,
            total_experts: self.experts.len(),
            total_params,
        }
    }
}

/// Convenience: create a layer with one expert of each type.
pub fn create_hetero_layer(n_experts_per_type: usize, width: usize) -> HeterogeneousLayer {
    let mut layer = HeterogeneousLayer::new(width);
    let types = [
        ExpertType::SymbolicALU,
        ExpertType::RetrievalExpert,
        ExpertType::ConvolutionExpert,
        ExpertType::ReasoningExpert,
        ExpertType::DenseExpert,
        ExpertType::RecurrentExpert,
    ];
    for et in &types {
        for _ in 0..n_experts_per_type {
            layer.add_expert(*et, width);
        }
    }
    layer
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_alu_exact_arithmetic() {
        let mut expert = SymbolicALUExpert::new(4);
        // Default op is add-adjacent (op_weights[0] = 1.0)
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = expert.forward(&input);
        // add-adjacent: [1+2, 2+3, 3+4, 4+1] = [3, 5, 7, 5]
        assert_eq!(output, vec![3.0, 5.0, 7.0, 5.0]);
    }

    #[test]
    fn test_symbolic_alu_sort() {
        let sorted = SymbolicALUExpert::sort(&[3, 1, 4, 1, 5]);
        assert_eq!(sorted, vec![1, 1, 3, 4, 5]);
    }

    #[test]
    fn test_symbolic_alu_argmax_and_count() {
        assert_eq!(SymbolicALUExpert::argmax(&[1, 5, 3, 2]), 1);
        assert_eq!(SymbolicALUExpert::count_nonzero(&[0, 1, 0, 3, 0]), 2);
    }

    #[test]
    fn test_retrieval_expert_store_and_retrieve() {
        let mut expert = RetrievalExpert::new(3);
        expert.store(vec![1.0, 0.0, 0.0], vec![10.0, 20.0, 30.0]);
        expert.store(vec![0.0, 1.0, 0.0], vec![40.0, 50.0, 60.0]);

        let results = expert.retrieve(&[1.0, 0.0, 0.0], 1);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, vec![10.0, 20.0, 30.0]);
        assert!((results[0].1 - 1.0).abs() < 1e-6); // exact match -> cosine sim 1.0
    }

    #[test]
    fn test_convolution_expert_averaging() {
        let mut expert = ConvolutionExpert::new(5, 3);
        // kernel = [1/3, 1/3, 1/3], so output is local average
        let input = vec![0.0, 0.0, 3.0, 0.0, 0.0];
        let output = expert.forward(&input);
        assert!((output[2] - 1.0).abs() < 1e-6); // center: (0+3+0)/3 = 1.0
        assert!((output[1] - 1.0).abs() < 1e-6); // (0+0+3)/3 = 1.0
        assert!((output[3] - 1.0).abs() < 1e-6); // (3+0+0)/3 = 1.0
    }

    #[test]
    fn test_reasoning_expert_multi_step() {
        let mut expert = ReasoningExpert::new(4, 3);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = expert.forward(&input);
        // After 3 steps through near-identity matrix, output should be non-zero
        assert_eq!(output.len(), 4);
        assert!(output.iter().any(|v| *v != 0.0));
    }

    #[test]
    fn test_recurrent_expert_state() {
        let mut expert = RecurrentExpert::new(3);
        let input = vec![1.0, 0.0, 0.0];
        let out1 = expert.forward(&input);
        let out2 = expert.forward(&input);
        // Second call should differ because hidden state changed
        assert_ne!(out1, out2);
        // After reset, should behave like first call
        expert.reset_state();
        let out3 = expert.forward(&input);
        assert_eq!(out1, out3);
    }

    #[test]
    fn test_heterogeneous_layer_forward() {
        let mut layer = create_hetero_layer(1, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_heterogeneous_layer_batch_and_stats() {
        let mut layer = create_hetero_layer(1, 4);
        let batch = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let outputs = layer.forward_batch(&batch);
        assert_eq!(outputs.len(), 4);
        for o in &outputs {
            assert_eq!(o.len(), 4);
        }

        let stats = layer.stats();
        assert_eq!(stats.total_experts, 6); // 1 per type * 6 types
        assert_eq!(stats.type_counts.len(), 6);
    }

    #[test]
    fn test_router_returns_top_k() {
        let router = HeterogeneousRouter::new(4, 6, 2);
        let routing = router.route(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(routing.len(), 2);
        // Weights should sum to roughly 1 (they're softmax probabilities, top-2)
        let sum: f64 = routing.iter().map(|(_, w)| w).sum();
        assert!(sum > 0.0 && sum <= 1.0 + 1e-6);
    }

    #[test]
    fn test_add_remove_expert() {
        let mut layer = HeterogeneousLayer::new(4);
        assert_eq!(layer.experts.len(), 0);
        layer.add_expert(ExpertType::DenseExpert, 4);
        layer.add_expert(ExpertType::SymbolicALU, 4);
        assert_eq!(layer.experts.len(), 2);
        layer.remove_expert(0);
        assert_eq!(layer.experts.len(), 1);
        assert_eq!(layer.experts[0].expert_type(), ExpertType::SymbolicALU);
    }
}
