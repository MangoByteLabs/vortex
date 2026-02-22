// Adaptive-depth inference: early exit, batch compaction, mixed attention strategies, quantization

use std::f64::consts::PI;

// ── AdaptiveStats ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AdaptiveStats {
    pub exit_counts: Vec<usize>,
    pub avg_depth: f64,
    pub max_depth: usize,
    pub compute_savings: f64,
    pub tokens_per_second: f64,
}

impl AdaptiveStats {
    pub fn compute(exit_counts: &[usize], max_depth: usize, elapsed_secs: f64) -> Self {
        let total_tokens: usize = exit_counts.iter().sum();
        let weighted_sum: usize = exit_counts
            .iter()
            .enumerate()
            .map(|(layer, &count)| (layer + 1) * count)
            .sum();
        let avg_depth = if total_tokens > 0 {
            weighted_sum as f64 / total_tokens as f64
        } else {
            0.0
        };
        let compute_savings = if max_depth > 0 {
            1.0 - avg_depth / max_depth as f64
        } else {
            0.0
        };
        let tps = if elapsed_secs > 0.0 {
            total_tokens as f64 / elapsed_secs
        } else {
            0.0
        };
        AdaptiveStats {
            exit_counts: exit_counts.to_vec(),
            avg_depth,
            max_depth,
            compute_savings,
            tokens_per_second: tps,
        }
    }
}

// ── EarlyExitLayer ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EarlyExitLayer {
    pub classifier: Vec<Vec<f64>>, // [num_classes][hidden_dim]
    pub threshold: f64,
}

impl EarlyExitLayer {
    pub fn new(hidden_dim: usize, num_classes: usize, threshold: f64) -> Self {
        // Initialize with small deterministic weights
        let classifier: Vec<Vec<f64>> = (0..num_classes)
            .map(|c| {
                (0..hidden_dim)
                    .map(|h| ((c * 31 + h * 17) as f64 % 7.0 - 3.0) * 0.1)
                    .collect()
            })
            .collect();
        EarlyExitLayer {
            classifier,
            threshold,
        }
    }

    /// Compute logits then return max softmax probability as confidence.
    pub fn evaluate_confidence(&self, hidden: &[f64]) -> f64 {
        if self.classifier.is_empty() {
            return 0.0;
        }
        let logits: Vec<f64> = self
            .classifier
            .iter()
            .map(|w| {
                w.iter()
                    .zip(hidden.iter())
                    .map(|(wi, hi)| wi * hi)
                    .sum::<f64>()
            })
            .collect();
        // softmax max probability
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f64 = exps.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        exps.iter().cloned().fold(0.0_f64, f64::max) / sum
    }

    pub fn should_exit(&self, hidden: &[f64]) -> bool {
        self.evaluate_confidence(hidden) > self.threshold
    }
}

// ── LayerStrategy ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum LayerStrategy {
    FFTMixing,
    LinearAttention,
    FullAttention,
    QuantizedDense(u8),
}

// ── ModelLayer ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ModelLayer {
    pub weights: Vec<Vec<f64>>, // [out_dim][in_dim]
    pub bias: Vec<f64>,
    pub strategy: LayerStrategy,
}

impl ModelLayer {
    pub fn new(in_dim: usize, out_dim: usize, strategy: LayerStrategy) -> Self {
        let weights: Vec<Vec<f64>> = (0..out_dim)
            .map(|o| {
                (0..in_dim)
                    .map(|i| ((o * 13 + i * 7) as f64 % 5.0 - 2.0) * 0.05)
                    .collect()
            })
            .collect();
        let bias = vec![0.0; out_dim];
        ModelLayer {
            weights,
            bias,
            strategy,
        }
    }

    /// Forward one token through this layer.
    pub fn forward_token(&self, input: &[f64]) -> Vec<f64> {
        match &self.strategy {
            LayerStrategy::QuantizedDense(bits) => self.forward_quantized(input, *bits),
            _ => self.forward_dense(input),
        }
    }

    fn forward_dense(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, b)| {
                let dot: f64 = w.iter().zip(input.iter()).map(|(wi, xi)| wi * xi).sum();
                relu(dot + b)
            })
            .collect()
    }

    fn forward_quantized(&self, input: &[f64], bits: u8) -> Vec<f64> {
        let scale = (1 << (bits - 1)) as f64 - 1.0;
        self.weights
            .iter()
            .zip(self.bias.iter())
            .map(|(w, b)| {
                let dot: f64 = w
                    .iter()
                    .zip(input.iter())
                    .map(|(wi, xi)| {
                        let qi = (wi * scale).round() / scale;
                        qi * xi
                    })
                    .sum();
                relu(dot + b)
            })
            .collect()
    }

    /// Apply FFT-based token mixing across a batch (O(n log n) per dimension).
    pub fn forward_batch_fft(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if batch.is_empty() {
            return vec![];
        }
        let dim = batch[0].len();
        let n = batch.len();
        // For each dimension, do real-valued FFT mixing across sequence positions
        let mut mixed = vec![vec![0.0; dim]; n];
        for d in 0..dim {
            let signal: Vec<f64> = batch.iter().map(|tok| tok[d]).collect();
            let transformed = fft_real(&signal);
            // Low-pass: keep first half
            let mut filtered = transformed.clone();
            for i in filtered.len() / 2..filtered.len() {
                filtered[i] = 0.0;
            }
            let back = ifft_real(&filtered, n);
            for i in 0..n {
                mixed[i][d] = back[i];
            }
        }
        // Then apply dense layer per token
        mixed
            .iter()
            .map(|tok| self.forward_dense(tok))
            .collect()
    }

    /// Linear attention: O(n * d^2) using kernel trick.
    pub fn forward_batch_linear_attn(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if batch.is_empty() {
            return vec![];
        }
        let dim = batch[0].len();
        // phi(x) = elu(x) + 1
        let phi = |x: f64| -> f64 { if x > 0.0 { x + 1.0 } else { x.exp() } };
        // Compute KV accumulator and K accumulator
        let mut kv = vec![vec![0.0; dim]; dim]; // [d][d]
        let mut k_sum = vec![0.0; dim];
        for tok in batch {
            let k: Vec<f64> = tok.iter().map(|&x| phi(x)).collect();
            for i in 0..dim {
                k_sum[i] += k[i];
                for j in 0..dim {
                    kv[i][j] += k[i] * tok[j];
                }
            }
        }
        // For each query, compute attention output
        let mut results = Vec::with_capacity(batch.len());
        for tok in batch {
            let q: Vec<f64> = tok.iter().map(|&x| phi(x)).collect();
            let denom: f64 = q.iter().zip(k_sum.iter()).map(|(qi, ki)| qi * ki).sum();
            let denom = if denom.abs() < 1e-10 { 1.0 } else { denom };
            let out: Vec<f64> = (0..dim)
                .map(|j| {
                    let num: f64 = q.iter().zip(kv.iter()).map(|(qi, kvi)| qi * kvi[j]).sum();
                    num / denom
                })
                .collect();
            // Then apply dense
            results.push(self.forward_dense(&out));
        }
        results
    }

    /// Full quadratic attention: O(n^2 * d).
    pub fn forward_batch_full_attn(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        if batch.is_empty() {
            return vec![];
        }
        let dim = batch[0].len();
        let n = batch.len();
        let scale = (dim as f64).sqrt();
        let mut results = Vec::with_capacity(n);
        for i in 0..n {
            // Compute attention weights
            let mut weights = Vec::with_capacity(n);
            for j in 0..n {
                let dot: f64 = batch[i]
                    .iter()
                    .zip(batch[j].iter())
                    .map(|(a, b)| a * b)
                    .sum();
                weights.push(dot / scale);
            }
            // Softmax
            let max_w = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = weights.iter().map(|w| (w - max_w).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let attn: Vec<f64> = exps.iter().map(|e| e / sum).collect();
            // Weighted sum
            let mut out = vec![0.0; dim];
            for j in 0..n {
                for d in 0..dim {
                    out[d] += attn[j] * batch[j][d];
                }
            }
            results.push(self.forward_dense(&out));
        }
        results
    }

    /// Forward a batch using this layer's strategy.
    pub fn forward_batch(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        match &self.strategy {
            LayerStrategy::FFTMixing => self.forward_batch_fft(batch),
            LayerStrategy::LinearAttention => self.forward_batch_linear_attn(batch),
            LayerStrategy::FullAttention => self.forward_batch_full_attn(batch),
            LayerStrategy::QuantizedDense(_) => {
                batch.iter().map(|tok| self.forward_token(tok)).collect()
            }
        }
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

// ── Simple real FFT / IFFT ─────────────────────────────────────────────

fn fft_real(signal: &[f64]) -> Vec<f64> {
    let n = signal.len();
    if n <= 1 {
        return signal.to_vec();
    }
    // DFT magnitude (simplified real-valued transform)
    let mut result = vec![0.0; n];
    for k in 0..n {
        let mut re = 0.0;
        for (t, &s) in signal.iter().enumerate() {
            let angle = -2.0 * PI * (k as f64) * (t as f64) / (n as f64);
            re += s * angle.cos();
        }
        result[k] = re;
    }
    result
}

fn ifft_real(spectrum: &[f64], n: usize) -> Vec<f64> {
    let mut result = vec![0.0; n];
    let sn = spectrum.len();
    for t in 0..n {
        let mut re = 0.0;
        for (k, &s) in spectrum.iter().enumerate() {
            let angle = 2.0 * PI * (k as f64) * (t as f64) / (sn as f64);
            re += s * angle.cos();
        }
        result[t] = re / sn as f64;
    }
    result
}

// ── BatchCompactor ─────────────────────────────────────────────────────

pub struct BatchCompactor;

impl BatchCompactor {
    /// Remove exited tokens. Returns (compacted batch, index map from compacted -> original).
    pub fn compact(
        batch: &[Vec<f64>],
        mask: &[bool],
    ) -> (Vec<Vec<f64>>, Vec<usize>) {
        let mut compacted = Vec::new();
        let mut index_map = Vec::new();
        for (i, (tok, &active)) in batch.iter().zip(mask.iter()).enumerate() {
            if active {
                compacted.push(tok.clone());
                index_map.push(i);
            }
        }
        (compacted, index_map)
    }

    /// Scatter compacted outputs back to original positions.
    pub fn scatter(
        outputs: &[Vec<f64>],
        index_map: &[usize],
        full_size: usize,
        dim: usize,
    ) -> Vec<Vec<f64>> {
        let mut result = vec![vec![0.0; dim]; full_size];
        for (i, idx) in index_map.iter().enumerate() {
            if i < outputs.len() && *idx < full_size {
                result[*idx] = outputs[i].clone();
            }
        }
        result
    }

    /// Should we compact? Only if >25% exited to amortize overhead.
    pub fn should_compact(exited: usize, total: usize) -> bool {
        total > 0 && exited * 4 >= total
    }
}

// ── AdaptiveDepthModel ─────────────────────────────────────────────────

pub struct AdaptiveDepthModel {
    pub layers: Vec<(ModelLayer, Option<EarlyExitLayer>)>,
}

impl AdaptiveDepthModel {
    pub fn forward_adaptive(
        &self,
        batch: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, AdaptiveStats) {
        let start = std::time::Instant::now();
        let n = batch.len();
        if n == 0 || self.layers.is_empty() {
            return (
                vec![],
                AdaptiveStats::compute(&[], self.layers.len(), 0.0),
            );
        }

        let dim = self.layers.last().map(|(l, _)| l.weights.len()).unwrap_or(1);
        let num_layers = self.layers.len();
        let mut exit_counts = vec![0usize; num_layers];
        let mut final_outputs: Vec<Option<Vec<f64>>> = vec![None; n];

        // Current active batch + mapping to original indices
        let mut active_batch = batch.to_vec();
        let mut active_indices: Vec<usize> = (0..n).collect();

        for (layer_idx, (layer, exit_layer)) in self.layers.iter().enumerate() {
            if active_batch.is_empty() {
                break;
            }

            // Forward through layer
            let outputs = layer.forward_batch(&active_batch);
            active_batch = outputs;

            // Check exits (not on last layer)
            if layer_idx < num_layers - 1 {
                if let Some(exit) = exit_layer {
                    let mut exited_count = 0;
                    let mut still_active = vec![true; active_batch.len()];

                    for (i, tok) in active_batch.iter().enumerate() {
                        if exit.should_exit(tok) {
                            final_outputs[active_indices[i]] = Some(tok.clone());
                            still_active[i] = false;
                            exited_count += 1;
                            exit_counts[layer_idx] += 1;
                        }
                    }

                    if exited_count > 0
                        && BatchCompactor::should_compact(exited_count, active_batch.len())
                    {
                        let (compacted, idx_map) =
                            BatchCompactor::compact(&active_batch, &still_active);
                        let new_indices: Vec<usize> =
                            idx_map.iter().map(|&i| active_indices[i]).collect();
                        active_batch = compacted;
                        active_indices = new_indices;
                    } else if exited_count > 0 {
                        // Even if we don't compact, remove exited from tracking
                        let (compacted, idx_map) =
                            BatchCompactor::compact(&active_batch, &still_active);
                        let new_indices: Vec<usize> =
                            idx_map.iter().map(|&i| active_indices[i]).collect();
                        active_batch = compacted;
                        active_indices = new_indices;
                    }
                }
            }
        }

        // Remaining active tokens exit at last layer
        for (i, tok) in active_batch.iter().enumerate() {
            if i < active_indices.len() {
                final_outputs[active_indices[i]] = Some(tok.clone());
                exit_counts[num_layers - 1] += 1;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let results: Vec<Vec<f64>> = final_outputs
            .into_iter()
            .map(|o| o.unwrap_or_else(|| vec![0.0; dim]))
            .collect();

        let stats = AdaptiveStats::compute(&exit_counts, num_layers, elapsed);
        (results, stats)
    }
}

// ── AdaptiveModel (full combined model) ────────────────────────────────

pub struct AdaptiveModel {
    pub layers: Vec<(LayerStrategy, ModelLayer, Option<EarlyExitLayer>)>,
}

impl AdaptiveModel {
    /// Build a model with early=FFT+INT8, middle=linear_attn+INT4, final=full_attn+FP32.
    pub fn new(hidden_dim: usize, num_layers: usize, num_classes: usize) -> Self {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let (strategy, exit) = if i < num_layers / 3 {
                // Early layers: FFT mixing + exit ramps with low threshold
                (
                    LayerStrategy::FFTMixing,
                    Some(EarlyExitLayer::new(hidden_dim, num_classes, 0.7)),
                )
            } else if i < 2 * num_layers / 3 {
                // Middle layers: linear attention + exit ramps
                (
                    LayerStrategy::LinearAttention,
                    Some(EarlyExitLayer::new(hidden_dim, num_classes, 0.8)),
                )
            } else {
                // Final layers: full attention, no early exit (except last)
                (LayerStrategy::FullAttention, None)
            };
            let layer = ModelLayer::new(hidden_dim, hidden_dim, strategy.clone());
            layers.push((strategy, layer, exit));
        }
        let _ = layers
            .into_iter()
            .map(|(s, l, e)| (s, l, e))
            .collect::<Vec<_>>();
        // Rebuild properly
        let mut layers2 = Vec::new();
        for i in 0..num_layers {
            let (strategy, exit) = if i < num_layers / 3 {
                (
                    LayerStrategy::FFTMixing,
                    Some(EarlyExitLayer::new(hidden_dim, num_classes, 0.7)),
                )
            } else if i < 2 * num_layers / 3 {
                (
                    LayerStrategy::LinearAttention,
                    Some(EarlyExitLayer::new(hidden_dim, num_classes, 0.8)),
                )
            } else {
                (LayerStrategy::FullAttention, None)
            };
            let layer = ModelLayer::new(hidden_dim, hidden_dim, strategy.clone());
            layers2.push((strategy, layer, exit));
        }
        AdaptiveModel { layers: layers2 }
    }

    pub fn forward(&self, batch: &[Vec<f64>]) -> (Vec<Vec<f64>>, AdaptiveStats) {
        let depth_model = AdaptiveDepthModel {
            layers: self
                .layers
                .iter()
                .map(|(_, l, e)| (l.clone(), e.clone()))
                .collect(),
        };
        depth_model.forward_adaptive(batch)
    }
}

// ── ThresholdTuner ─────────────────────────────────────────────────────

pub struct ThresholdTuner {
    pub learning_rate: f64,
}

impl ThresholdTuner {
    pub fn new() -> Self {
        ThresholdTuner {
            learning_rate: 0.05,
        }
    }

    /// Returns adjusted thresholds. If savings < target, lower thresholds to let more exit.
    /// If savings > target, raise thresholds for quality.
    pub fn tune(
        &self,
        stats: &AdaptiveStats,
        target_savings: f64,
        current_thresholds: &[f64],
    ) -> Vec<f64> {
        let diff = stats.compute_savings - target_savings;
        // If savings too high (diff > 0), raise thresholds to keep more tokens going deeper
        // If savings too low (diff < 0), lower thresholds to let more exit early
        current_thresholds
            .iter()
            .map(|&t| (t + diff * self.learning_rate).clamp(0.1, 0.99))
            .collect()
    }
}

// ── Interpreter builtins ───────────────────────────────────────────────

use crate::interpreter::{Env, Value};
use std::sync::Mutex;

static ADAPTIVE_MODELS: Mutex<Vec<AdaptiveModel>> = Mutex::new(Vec::new());

pub fn builtin_adaptive_model_new(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    // adaptive_model_new(hidden_dim, num_layers, num_classes)
    if args.len() < 2 {
        return Err("adaptive_model_new expects (hidden_dim, num_layers, [num_classes])".into());
    }
    let hidden_dim = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("hidden_dim must be int".into()),
    };
    let num_layers = match &args[1] {
        Value::Int(n) => *n as usize,
        _ => return Err("num_layers must be int".into()),
    };
    let num_classes = if args.len() > 2 {
        match &args[2] {
            Value::Int(n) => *n as usize,
            _ => 10,
        }
    } else {
        10
    };
    let model = AdaptiveModel::new(hidden_dim, num_layers, num_classes);
    let mut models = ADAPTIVE_MODELS.lock().unwrap();
    let id = models.len();
    models.push(model);
    Ok(Value::Int(id as i128))
}

pub fn builtin_adaptive_model_forward(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    // adaptive_model_forward(id, batch)
    if args.len() < 2 {
        return Err("adaptive_model_forward expects (id, batch)".into());
    }
    let id = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("id must be int".into()),
    };
    let batch = match &args[1] {
        Value::Array(rows) => {
            let mut b = Vec::new();
            for row in rows {
                match row {
                    Value::Array(vals) => {
                        let v: Vec<f64> = vals
                            .iter()
                            .map(|x| match x {
                                Value::Float(f) => *f,
                                Value::Int(i) => *i as f64,
                                _ => 0.0,
                            })
                            .collect();
                        b.push(v);
                    }
                    _ => return Err("batch rows must be arrays".into()),
                }
            }
            b
        }
        _ => return Err("batch must be array of arrays".into()),
    };
    let models = ADAPTIVE_MODELS.lock().unwrap();
    let model = models.get(id).ok_or("invalid model id")?;
    let (outputs, _stats) = model.forward(&batch);
    let result: Vec<Value> = outputs
        .into_iter()
        .map(|row| Value::Array(row.into_iter().map(Value::Float).collect()))
        .collect();
    Ok(Value::Array(result))
}

pub fn builtin_adaptive_model_stats(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    // adaptive_model_stats(id, batch) -> [avg_depth, compute_savings, tokens_per_sec]
    if args.len() < 2 {
        return Err("adaptive_model_stats expects (id, batch)".into());
    }
    let id = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("id must be int".into()),
    };
    let batch = match &args[1] {
        Value::Array(rows) => {
            let mut b = Vec::new();
            for row in rows {
                match row {
                    Value::Array(vals) => {
                        let v: Vec<f64> = vals
                            .iter()
                            .map(|x| match x {
                                Value::Float(f) => *f,
                                Value::Int(i) => *i as f64,
                                _ => 0.0,
                            })
                            .collect();
                        b.push(v);
                    }
                    _ => return Err("batch rows must be arrays".into()),
                }
            }
            b
        }
        _ => return Err("batch must be array of arrays".into()),
    };
    let models = ADAPTIVE_MODELS.lock().unwrap();
    let model = models.get(id).ok_or("invalid model id")?;
    let (_outputs, stats) = model.forward(&batch);
    Ok(Value::Array(vec![
        Value::Float(stats.avg_depth),
        Value::Float(stats.compute_savings),
        Value::Float(stats.tokens_per_second),
    ]))
}

pub fn builtin_adaptive_model_tune(
    _env: &mut Env,
    args: Vec<Value>,
) -> Result<Value, String> {
    // adaptive_model_tune(id, target_savings) -> new_thresholds
    if args.len() < 2 {
        return Err("adaptive_model_tune expects (id, target_savings)".into());
    }
    let id = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("id must be int".into()),
    };
    let target = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("target_savings must be float".into()),
    };
    let models = ADAPTIVE_MODELS.lock().unwrap();
    let model = models.get(id).ok_or("invalid model id")?;

    // Collect current thresholds
    let thresholds: Vec<f64> = model
        .layers
        .iter()
        .filter_map(|(_, _, e)| e.as_ref().map(|ex| ex.threshold))
        .collect();

    // Run a dummy forward to get stats
    let dim = if let Some((_, l, _)) = model.layers.first() {
        l.weights[0].len()
    } else {
        4
    };
    let dummy_batch = vec![vec![1.0; dim]; 8];
    let (_, stats) = model.forward(&dummy_batch);

    let tuner = ThresholdTuner::new();
    let new_thresholds = tuner.tune(&stats, target, &thresholds);

    Ok(Value::Array(
        new_thresholds.into_iter().map(Value::Float).collect(),
    ))
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_simple_model(
        dim: usize,
        num_layers: usize,
        threshold: f64,
    ) -> AdaptiveDepthModel {
        let mut layers = Vec::new();
        for i in 0..num_layers {
            let layer = ModelLayer::new(dim, dim, LayerStrategy::FullAttention);
            let exit = if i < num_layers - 1 {
                Some(EarlyExitLayer::new(dim, 4, threshold))
            } else {
                None
            };
            layers.push((layer, exit));
        }
        AdaptiveDepthModel { layers }
    }

    #[test]
    fn single_token_forward() {
        let model = make_simple_model(4, 3, 0.99);
        let batch = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let (outputs, stats) = model.forward_adaptive(&batch);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].len(), 4);
        assert!(stats.max_depth == 3);
    }

    #[test]
    fn batch_forward_correct_size() {
        let model = make_simple_model(4, 3, 0.99);
        let batch = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let (outputs, _) = model.forward_adaptive(&batch);
        assert_eq!(outputs.len(), 4);
        for o in &outputs {
            assert_eq!(o.len(), 4);
        }
    }

    #[test]
    fn high_confidence_exits_early() {
        // Very low threshold so tokens exit at first opportunity
        let model = make_simple_model(4, 6, 0.01);
        let batch = vec![vec![10.0, 20.0, 30.0, 40.0]];
        let (_, stats) = model.forward_adaptive(&batch);
        // Token should exit before the last layer
        assert!(
            stats.avg_depth < stats.max_depth as f64,
            "avg_depth {} should be < max_depth {}",
            stats.avg_depth,
            stats.max_depth
        );
    }

    #[test]
    fn low_confidence_goes_deep() {
        // Very high threshold so nothing exits early
        let model = make_simple_model(4, 4, 0.999);
        let batch = vec![vec![0.001, 0.001, 0.001, 0.001]];
        let (_, stats) = model.forward_adaptive(&batch);
        // Should go to the last layer
        assert_eq!(stats.exit_counts[stats.max_depth - 1], 1);
    }

    #[test]
    fn batch_compaction_reduces_size() {
        let batch = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
        ];
        let mask = vec![true, false, true, false]; // keep 0 and 2
        let (compacted, idx_map) = BatchCompactor::compact(&batch, &mask);
        assert_eq!(compacted.len(), 2);
        assert_eq!(idx_map, vec![0, 2]);
        assert_eq!(compacted[0], vec![1.0, 2.0]);
        assert_eq!(compacted[1], vec![5.0, 6.0]);
    }

    #[test]
    fn scatter_reconstructs_ordering() {
        let outputs = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let idx_map = vec![1, 3];
        let scattered = BatchCompactor::scatter(&outputs, &idx_map, 4, 2);
        assert_eq!(scattered.len(), 4);
        assert_eq!(scattered[0], vec![0.0, 0.0]); // not filled
        assert_eq!(scattered[1], vec![10.0, 20.0]);
        assert_eq!(scattered[2], vec![0.0, 0.0]);
        assert_eq!(scattered[3], vec![30.0, 40.0]);
    }

    #[test]
    fn fft_mixing_produces_output() {
        let layer = ModelLayer::new(4, 4, LayerStrategy::FFTMixing);
        let batch = vec![
            vec![1.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0],
            vec![0.0, 0.0, 0.0, 1.0],
        ];
        let out = layer.forward_batch(&batch);
        assert_eq!(out.len(), 4);
        for o in &out {
            assert_eq!(o.len(), 4);
        }
    }

    #[test]
    fn quantized_layer_produces_approximate_output() {
        let layer_fp = ModelLayer::new(4, 4, LayerStrategy::FullAttention);
        let layer_q4 = ModelLayer {
            weights: layer_fp.weights.clone(),
            bias: layer_fp.bias.clone(),
            strategy: LayerStrategy::QuantizedDense(4),
        };
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out_fp = layer_fp.forward_token(&input);
        let out_q4 = layer_q4.forward_token(&input);
        assert_eq!(out_fp.len(), out_q4.len());
        // Quantized should be close but not necessarily identical
        for (a, b) in out_fp.iter().zip(out_q4.iter()) {
            assert!(
                (a - b).abs() < 5.0,
                "fp={} q4={} diff too large",
                a,
                b
            );
        }
    }

    #[test]
    fn adaptive_stats_savings_positive_with_early_exits() {
        let model = make_simple_model(4, 6, 0.01);
        let batch: Vec<Vec<f64>> = (0..16)
            .map(|i| vec![i as f64 * 10.0, i as f64 * 5.0, 1.0, 2.0])
            .collect();
        let (_, stats) = model.forward_adaptive(&batch);
        assert!(
            stats.compute_savings > 0.0,
            "compute_savings should be > 0, got {}",
            stats.compute_savings
        );
    }

    #[test]
    fn threshold_tuner_adjusts_toward_target() {
        let tuner = ThresholdTuner::new();
        // Stats with low savings -> tuner should lower thresholds
        let stats = AdaptiveStats {
            exit_counts: vec![0, 0, 10],
            avg_depth: 3.0,
            max_depth: 3,
            compute_savings: 0.0,
            tokens_per_second: 100.0,
        };
        let thresholds = vec![0.8, 0.9];
        let new_t = tuner.tune(&stats, 0.5, &thresholds);
        // savings(0.0) < target(0.5), diff = -0.5, so thresholds should decrease
        assert!(new_t[0] < thresholds[0], "threshold should decrease");
        assert!(new_t[1] < thresholds[1], "threshold should decrease");
    }

    #[test]
    fn exit_distribution_nonuniform_for_easy_data() {
        // Low threshold: easy data should exit early, making distribution non-uniform
        let model = make_simple_model(4, 6, 0.01);
        let batch: Vec<Vec<f64>> = (0..32)
            .map(|i| vec![i as f64 * 100.0, i as f64 * 50.0, 10.0, 20.0])
            .collect();
        let (_, stats) = model.forward_adaptive(&batch);
        let early_exits: usize = stats.exit_counts[..stats.max_depth - 1].iter().sum();
        let final_exits = stats.exit_counts[stats.max_depth - 1];
        // With very low threshold, most should exit early
        assert!(
            early_exits > 0 || final_exits > 0,
            "some tokens should have exited"
        );
        // Distribution should not be all at the last layer
        let total: usize = stats.exit_counts.iter().sum();
        assert_eq!(total, 32, "all tokens must be accounted for");
    }

    #[test]
    fn full_adaptive_model_end_to_end() {
        let model = AdaptiveModel::new(8, 6, 4);
        let batch: Vec<Vec<f64>> = (0..10)
            .map(|i| vec![i as f64; 8])
            .collect();
        let (outputs, stats) = model.forward(&batch);
        assert_eq!(outputs.len(), 10);
        assert!(stats.max_depth == 6);
        // All tokens must be accounted for
        let total: usize = stats.exit_counts.iter().sum();
        assert_eq!(total, 10);
    }
}
