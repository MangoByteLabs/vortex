/// Multi-timescale processing with clock domains.
///
/// Fast layers process every token, slow layers process every Nth token,
/// with cross-clock buffers for communication between timescales.

// ── ClockDomain ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ClockDomain {
    pub rate: usize,
    pub phase: usize,
}

impl ClockDomain {
    pub fn new(rate: usize, phase: usize) -> Self {
        Self {
            rate: rate.max(1),
            phase: phase % rate.max(1),
        }
    }

    pub fn should_tick(&self, global_step: usize) -> bool {
        (global_step + self.phase) % self.rate == 0
    }
}

// ── AggregationMode & ClockBuffer ────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AggregationMode {
    Mean,
    Max,
    AttentionPool,
    Last,
}

#[derive(Debug, Clone)]
pub struct ClockBuffer {
    pub buffer: Vec<Vec<f64>>,
    pub mode: AggregationMode,
}

impl ClockBuffer {
    pub fn new(mode: AggregationMode) -> Self {
        Self {
            buffer: Vec::new(),
            mode,
        }
    }

    pub fn push(&mut self, value: &[f64]) {
        self.buffer.push(value.to_vec());
    }

    pub fn aggregate(&self) -> Vec<f64> {
        if self.buffer.is_empty() {
            return Vec::new();
        }
        let dim = self.buffer[0].len();
        match self.mode {
            AggregationMode::Mean => {
                let mut acc = vec![0.0; dim];
                for v in &self.buffer {
                    for (i, &x) in v.iter().enumerate() {
                        acc[i] += x;
                    }
                }
                let n = self.buffer.len() as f64;
                acc.iter().map(|x| x / n).collect()
            }
            AggregationMode::Max => {
                let mut acc = vec![f64::NEG_INFINITY; dim];
                for v in &self.buffer {
                    for (i, &x) in v.iter().enumerate() {
                        if x > acc[i] {
                            acc[i] = x;
                        }
                    }
                }
                acc
            }
            AggregationMode::AttentionPool => {
                // Simple attention: softmax over L2 norms, then weighted sum
                let norms: Vec<f64> = self
                    .buffer
                    .iter()
                    .map(|v| v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-12))
                    .collect();
                let max_n = norms.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = norms.iter().map(|n| (n - max_n).exp()).collect();
                let sum_exp: f64 = exps.iter().sum();
                let weights: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();
                let mut acc = vec![0.0; dim];
                for (v, &w) in self.buffer.iter().zip(weights.iter()) {
                    for (i, &x) in v.iter().enumerate() {
                        acc[i] += x * w;
                    }
                }
                acc
            }
            AggregationMode::Last => self.buffer.last().unwrap().clone(),
        }
    }

    pub fn clear(&mut self) {
        self.buffer.clear();
    }
}

// ── TimescaleLayer ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TimescaleLayer {
    pub layer_weights: Vec<Vec<f64>>,
    pub clock: ClockDomain,
    pub cached_output: Option<Vec<f64>>,
    ticks: usize,
    total_calls: usize,
}

impl TimescaleLayer {
    pub fn new(input_dim: usize, output_dim: usize, rate: usize) -> Self {
        // Deterministic small weights for reproducibility
        let mut weights = Vec::with_capacity(output_dim);
        for o in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for i in 0..input_dim {
                row.push(0.01 * ((o * input_dim + i) as f64 % 7.0 - 3.0));
            }
            weights.push(row);
        }
        Self {
            layer_weights: weights,
            clock: ClockDomain::new(rate, 0),
            cached_output: None,
            ticks: 0,
            total_calls: 0,
        }
    }

    pub fn forward(&mut self, input: &[f64], global_step: usize) -> Vec<f64> {
        self.total_calls += 1;
        if self.clock.should_tick(global_step) {
            let out = self.compute(input);
            self.cached_output = Some(out.clone());
            self.ticks += 1;
            out
        } else {
            self.cached_output
                .clone()
                .unwrap_or_else(|| vec![0.0; self.layer_weights.len()])
        }
    }

    fn compute(&self, input: &[f64]) -> Vec<f64> {
        self.layer_weights
            .iter()
            .map(|row| {
                let dot: f64 = row
                    .iter()
                    .zip(input.iter())
                    .map(|(w, x)| w * x)
                    .sum();
                // ReLU
                dot.max(0.0)
            })
            .collect()
    }

    pub fn ticks(&self) -> usize {
        self.ticks
    }

    pub fn total_calls(&self) -> usize {
        self.total_calls
    }
}

// ── MultiscaleModel ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MultiscaleModel {
    pub fast_layers: Vec<TimescaleLayer>,
    pub medium_layers: Vec<TimescaleLayer>,
    pub slow_layers: Vec<TimescaleLayer>,
    pub fast_to_medium: ClockBuffer,
    pub medium_to_slow: ClockBuffer,
    pub slow_to_fast: Vec<f64>,
}

impl MultiscaleModel {
    pub fn new(
        fast_width: usize,
        medium_width: usize,
        slow_width: usize,
        n_layers: usize,
    ) -> Self {
        let mut fast_layers = Vec::new();
        let mut medium_layers = Vec::new();
        let mut slow_layers = Vec::new();

        for i in 0..n_layers {
            let fin = if i == 0 { fast_width } else { fast_width };
            fast_layers.push(TimescaleLayer::new(fin, fast_width, 1));
            medium_layers.push(TimescaleLayer::new(
                if i == 0 { fast_width } else { medium_width },
                medium_width,
                4,
            ));
            slow_layers.push(TimescaleLayer::new(
                if i == 0 { medium_width } else { slow_width },
                slow_width,
                16,
            ));
        }

        Self {
            fast_layers,
            medium_layers,
            slow_layers,
            fast_to_medium: ClockBuffer::new(AggregationMode::Mean),
            medium_to_slow: ClockBuffer::new(AggregationMode::Mean),
            slow_to_fast: vec![0.0; slow_width],
        }
    }

    pub fn forward(&mut self, input: &[f64], step: usize) -> Vec<f64> {
        // Fast path: always runs
        let mut fast_out = input.to_vec();
        // Add slow context to fast input
        if !self.slow_to_fast.is_empty() && fast_out.len() == self.slow_to_fast.len() {
            for (f, &s) in fast_out.iter_mut().zip(self.slow_to_fast.iter()) {
                *f += s * 0.1; // gated residual from slow clock
            }
        }
        for layer in &mut self.fast_layers {
            fast_out = layer.forward(&fast_out, step);
        }

        // Push fast output into cross-clock buffer
        self.fast_to_medium.push(&fast_out);

        // Medium path: every 4th step
        let medium_ticks = self.medium_layers.first().map_or(false, |l| l.clock.should_tick(step));
        if medium_ticks {
            let agg = self.fast_to_medium.aggregate();
            self.fast_to_medium.clear();
            let mut med_out = if agg.is_empty() { fast_out.clone() } else { agg };
            for layer in &mut self.medium_layers {
                med_out = layer.forward(&med_out, step);
            }
            self.medium_to_slow.push(&med_out);
        } else {
            // Still call forward on medium layers (they'll use cache)
            let dummy_in = vec![0.0; self.medium_layers.first().map_or(0, |l| l.layer_weights[0].len())];
            for layer in &mut self.medium_layers {
                let _ = layer.forward(&dummy_in, step);
            }
        }

        // Slow path: every 16th step
        let slow_ticks = self.slow_layers.first().map_or(false, |l| l.clock.should_tick(step));
        if slow_ticks {
            let agg = self.medium_to_slow.aggregate();
            self.medium_to_slow.clear();
            let mut slow_out = if agg.is_empty() {
                vec![0.0; self.slow_layers.first().map_or(0, |l| l.layer_weights[0].len())]
            } else {
                agg
            };
            for layer in &mut self.slow_layers {
                slow_out = layer.forward(&slow_out, step);
            }
            self.slow_to_fast = slow_out;
        } else {
            let dummy_in = vec![0.0; self.slow_layers.first().map_or(0, |l| l.layer_weights[0].len())];
            for layer in &mut self.slow_layers {
                let _ = layer.forward(&dummy_in, step);
            }
        }

        fast_out
    }

    pub fn compute_savings(&self, n_steps: usize) -> f64 {
        let n_fast = self.fast_layers.len();
        let n_medium = self.medium_layers.len();
        let n_slow = self.slow_layers.len();
        let total_layers = n_fast + n_medium + n_slow;
        if total_layers == 0 {
            return 0.0;
        }
        let all_fast_ops = (total_layers * n_steps) as f64;
        let actual_ops = (n_fast * n_steps) as f64
            + (n_medium as f64) * (n_steps as f64 / 4.0).ceil()
            + (n_slow as f64) * (n_steps as f64 / 16.0).ceil();
        1.0 - actual_ops / all_fast_ops
    }

    pub fn stats(&self, n_steps: usize) -> MultiscaleStats {
        let mut ticks_per_layer = Vec::new();
        let mut total_ticks = 0usize;
        let mut total_calls = 0usize;
        for layer in self
            .fast_layers
            .iter()
            .chain(self.medium_layers.iter())
            .chain(self.slow_layers.iter())
        {
            ticks_per_layer.push(layer.ticks());
            total_ticks += layer.ticks();
            total_calls += layer.total_calls();
        }
        let n_layers = ticks_per_layer.len();
        let theoretical_max = n_layers * n_steps;
        let compute_ratio = if theoretical_max > 0 {
            total_ticks as f64 / theoretical_max as f64
        } else {
            0.0
        };
        let cache_hits = if total_calls > 0 {
            (total_calls - total_ticks) as f64 / total_calls as f64
        } else {
            0.0
        };
        MultiscaleStats {
            ticks_per_layer,
            compute_ratio,
            cache_hit_rate: cache_hits,
        }
    }
}

// ── AdaptiveClock ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AdaptiveClock {
    pub rates: Vec<usize>,
}

impl AdaptiveClock {
    pub fn new(n_layers: usize, initial_rate: usize) -> Self {
        Self {
            rates: vec![initial_rate.max(1).min(64); n_layers],
        }
    }

    pub fn adjust_rate(&mut self, layer_idx: usize, input_variance: f64) {
        if layer_idx >= self.rates.len() {
            return;
        }
        // High variance → faster clock (lower rate)
        let current = self.rates[layer_idx] as f64;
        let new_rate = if input_variance > 1.0 {
            (current / 2.0).ceil() as usize
        } else if input_variance > 0.5 {
            current as usize // no change
        } else {
            ((current * 1.5).floor() as usize).min(64)
        };
        self.rates[layer_idx] = new_rate.max(1).min(64);
    }

    pub fn slow_down(&mut self, layer_idx: usize) {
        if layer_idx >= self.rates.len() {
            return;
        }
        self.rates[layer_idx] = (self.rates[layer_idx] * 2).min(64);
    }

    pub fn rate(&self, layer_idx: usize) -> usize {
        self.rates.get(layer_idx).copied().unwrap_or(1)
    }
}

// ── MultiscaleStats ──────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MultiscaleStats {
    pub ticks_per_layer: Vec<usize>,
    pub compute_ratio: f64,
    pub cache_hit_rate: f64,
}

// ── Interpreter builtins ─────────────────────────────────────────────

use crate::interpreter::{Env, Value};

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) => Ok(*i as usize),
        _ => Err("expected integer".into()),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(a) => a
            .iter()
            .map(|x| match x {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("expected numeric array".into()),
            })
            .collect(),
        _ => Err("expected array".into()),
    }
}

pub fn builtin_multiscale_model_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("multiscale_model_new expects 4 args: (fast_width, medium_width, slow_width, n_layers)".into());
    }
    let fw = value_to_usize(&args[0])?;
    let mw = value_to_usize(&args[1])?;
    let sw = value_to_usize(&args[2])?;
    let nl = value_to_usize(&args[3])?;
    let model = MultiscaleModel::new(fw, mw, sw, nl);
    let id = env.next_multiscale_id;
    env.next_multiscale_id += 1;
    env.multiscale_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

pub fn builtin_multiscale_model_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("multiscale_model_forward expects 3 args: (id, input, step)".into());
    }
    let id = value_to_usize(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let step = value_to_usize(&args[2])?;
    let model = env.multiscale_models.get_mut(&id).ok_or("no such multiscale model")?;
    let out = model.forward(&input, step);
    Ok(Value::Array(out.into_iter().map(Value::Float).collect()))
}

pub fn builtin_multiscale_model_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("multiscale_model_stats expects 2 args: (id, n_steps)".into());
    }
    let id = value_to_usize(&args[0])?;
    let n_steps = value_to_usize(&args[1])?;
    let model = env.multiscale_models.get(&id).ok_or("no such multiscale model")?;
    let stats = model.stats(n_steps);
    let savings = model.compute_savings(n_steps);
    Ok(Value::Array(vec![
        Value::Float(stats.compute_ratio),
        Value::Float(stats.cache_hit_rate),
        Value::Float(savings),
    ]))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_clock_domain_rate1() {
        let c = ClockDomain::new(1, 0);
        for i in 0..10 {
            assert!(c.should_tick(i), "rate=1 should tick every step");
        }
    }

    #[test]
    fn test_clock_domain_rate4() {
        let c = ClockDomain::new(4, 0);
        assert!(c.should_tick(0));
        assert!(!c.should_tick(1));
        assert!(!c.should_tick(2));
        assert!(!c.should_tick(3));
        assert!(c.should_tick(4));
        assert!(c.should_tick(16));
    }

    #[test]
    fn test_clock_domain_phase() {
        let c = ClockDomain::new(4, 2);
        assert!(!c.should_tick(0));
        assert!(!c.should_tick(1));
        assert!(c.should_tick(2)); // (2+2)%4==0
        assert!(!c.should_tick(3));
        assert!(!c.should_tick(4));
        assert!(c.should_tick(6)); // (6+2)%4==0
    }

    #[test]
    fn test_clock_buffer_mean() {
        let mut buf = ClockBuffer::new(AggregationMode::Mean);
        buf.push(&[2.0, 4.0]);
        buf.push(&[6.0, 8.0]);
        let agg = buf.aggregate();
        assert!((agg[0] - 4.0).abs() < 1e-9);
        assert!((agg[1] - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_clock_buffer_max() {
        let mut buf = ClockBuffer::new(AggregationMode::Max);
        buf.push(&[1.0, 5.0]);
        buf.push(&[3.0, 2.0]);
        let agg = buf.aggregate();
        assert!((agg[0] - 3.0).abs() < 1e-9);
        assert!((agg[1] - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_clock_buffer_last() {
        let mut buf = ClockBuffer::new(AggregationMode::Last);
        buf.push(&[1.0, 2.0]);
        buf.push(&[10.0, 20.0]);
        let agg = buf.aggregate();
        assert!((agg[0] - 10.0).abs() < 1e-9);
        assert!((agg[1] - 20.0).abs() < 1e-9);
    }

    #[test]
    fn test_timescale_layer_caching() {
        let mut layer = TimescaleLayer::new(4, 4, 4);
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let out0 = layer.forward(&input, 0); // tick
        let out1 = layer.forward(&input, 1); // cached
        let out2 = layer.forward(&input, 2); // cached
        let out4 = layer.forward(&input, 4); // tick
        // Cached outputs should equal tick output
        assert_eq!(out0, out1);
        assert_eq!(out0, out2);
        // Step 4 is a new tick, same input so same result
        assert_eq!(out0, out4);
        assert_eq!(layer.ticks(), 2);
        assert_eq!(layer.total_calls(), 4);
    }

    #[test]
    fn test_multiscale_model_forward() {
        let mut model = MultiscaleModel::new(4, 4, 4, 2);
        let input = vec![1.0, 1.0, 1.0, 1.0];
        // Run 32 steps
        let mut outputs = Vec::new();
        for step in 0..32 {
            outputs.push(model.forward(&input, step));
        }
        // All outputs should have the right dimension
        for out in &outputs {
            assert_eq!(out.len(), 4);
        }
        // Check stats
        let stats = model.stats(32);
        assert!(stats.compute_ratio < 1.0, "should use less than full compute");
        assert!(stats.cache_hit_rate > 0.0, "should have cache hits");
    }

    #[test]
    fn test_compute_savings() {
        let model = MultiscaleModel::new(4, 4, 4, 2);
        let savings = model.compute_savings(64);
        assert!(savings > 0.0, "savings should be positive");
        assert!(savings < 1.0, "savings should be less than 100%");
    }

    #[test]
    fn test_adaptive_clock() {
        let mut ac = AdaptiveClock::new(3, 8);
        assert_eq!(ac.rate(0), 8);
        // High variance should speed up (lower rate)
        ac.adjust_rate(0, 2.0);
        assert!(ac.rate(0) < 8);
        // Slow down
        ac.slow_down(1);
        assert_eq!(ac.rate(1), 16);
        // Max cap
        ac.slow_down(1);
        assert_eq!(ac.rate(1), 32);
        ac.slow_down(1);
        assert_eq!(ac.rate(1), 64);
        ac.slow_down(1);
        assert_eq!(ac.rate(1), 64); // capped
    }

    #[test]
    fn test_clock_buffer_attention_pool() {
        let mut buf = ClockBuffer::new(AggregationMode::AttentionPool);
        buf.push(&[1.0, 0.0]);
        buf.push(&[0.0, 10.0]); // much larger norm → higher weight
        let agg = buf.aggregate();
        // Second vector has larger norm, so result should lean toward it
        assert!(agg[1] > agg[0]);
    }

    #[test]
    fn test_clock_buffer_clear() {
        let mut buf = ClockBuffer::new(AggregationMode::Mean);
        buf.push(&[1.0]);
        buf.clear();
        assert!(buf.buffer.is_empty());
        assert!(buf.aggregate().is_empty());
    }
}
