// Pillar 6: Energy-Efficient Computation for Vortex
// Provides energy budgets, approximate computing, event-driven execution,
// neuromorphic layers (leaky integrate-and-fire), dynamic precision, and energy gating.

use std::collections::HashMap;
use crate::interpreter::{Env, Value, FnDef};

// ─── Deterministic LCG RNG ──────────────────────────────────────────

fn lcg_next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64).max(1e-10)
}

// ─── Helper conversions ─────────────────────────────────────────────

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

fn value_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(format!("expected string, got {:?}", v)),
    }
}

// ─── Core types ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum EnergyPolicy {
    Hard,
    Soft,
    Adaptive,
}

impl EnergyPolicy {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "hard" => Ok(EnergyPolicy::Hard),
            "soft" => Ok(EnergyPolicy::Soft),
            "adaptive" => Ok(EnergyPolicy::Adaptive),
            _ => Err(format!("unknown energy policy: '{}', expected hard/soft/adaptive", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct EnergyBudget {
    pub total_joules: f64,
    pub remaining: f64,
    pub policy: EnergyPolicy,
}

impl EnergyBudget {
    fn new(total_joules: f64, policy: EnergyPolicy) -> Self {
        Self { total_joules, remaining: total_joules, policy }
    }

    /// Attempt to consume energy. Returns true if allowed.
    fn consume(&mut self, joules: f64) -> bool {
        match self.policy {
            EnergyPolicy::Hard => {
                if self.remaining >= joules {
                    self.remaining -= joules;
                    true
                } else {
                    false
                }
            }
            EnergyPolicy::Soft => {
                self.remaining -= joules;
                true
            }
            EnergyPolicy::Adaptive => {
                self.remaining -= joules;
                true
            }
        }
    }

    fn consumed(&self) -> f64 {
        self.total_joules - self.remaining
    }

    fn efficiency(&self) -> f64 {
        if self.total_joules <= 0.0 { return 0.0; }
        (self.consumed() / self.total_joules).min(1.0).max(0.0)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ApproxPrecision {
    Full,
    Half,
    Quarter,
    Binary,
}

impl ApproxPrecision {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "full" => Ok(ApproxPrecision::Full),
            "half" => Ok(ApproxPrecision::Half),
            "quarter" => Ok(ApproxPrecision::Quarter),
            "binary" => Ok(ApproxPrecision::Binary),
            _ => Err(format!("unknown precision: '{}', expected full/half/quarter/binary", s)),
        }
    }

    /// Relative energy cost multiplier
    fn energy_cost(&self) -> f64 {
        match self {
            ApproxPrecision::Full => 1.0,
            ApproxPrecision::Half => 0.5,
            ApproxPrecision::Quarter => 0.25,
            ApproxPrecision::Binary => 0.1,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ApproxConfig {
    pub precision: ApproxPrecision,
    pub skip_threshold: f64,
}

#[derive(Debug, Clone)]
pub struct EventDrivenExecutor {
    pub spike_threshold: f64,
    pub active_neurons: Vec<bool>,
    pub event_queue: Vec<(usize, f64)>,
    pub membrane_potentials: Vec<f64>,
}

impl EventDrivenExecutor {
    fn new(num_neurons: usize, spike_threshold: f64) -> Self {
        Self {
            spike_threshold,
            active_neurons: vec![false; num_neurons],
            event_queue: Vec::new(),
            membrane_potentials: vec![0.0; num_neurons],
        }
    }

    /// Process one time step. Only neurons with input above threshold are computed.
    /// Returns indices of neurons that spiked.
    fn step(&mut self, input_spikes: &[f64]) -> Vec<bool> {
        let n = self.membrane_potentials.len();
        let mut output_spikes = vec![false; n];

        // Clear event queue
        self.event_queue.clear();

        // Only process neurons where input exceeds threshold
        for i in 0..n.min(input_spikes.len()) {
            if input_spikes[i].abs() > self.spike_threshold {
                self.active_neurons[i] = true;
                self.event_queue.push((i, input_spikes[i]));
            }
        }

        // Process only active neurons
        for &(idx, spike_val) in &self.event_queue {
            self.membrane_potentials[idx] += spike_val;
            if self.membrane_potentials[idx] > self.spike_threshold {
                output_spikes[idx] = true;
                // Reset after spike
                self.membrane_potentials[idx] = 0.0;
            }
        }

        // Decay inactive neurons slightly
        for i in 0..n {
            if !self.active_neurons[i] {
                self.membrane_potentials[i] *= 0.9;
            }
            self.active_neurons[i] = false;
        }

        output_spikes
    }
}

#[derive(Debug, Clone)]
pub struct NeuromorphicLayer {
    pub weights: Vec<Vec<f64>>,
    pub thresholds: Vec<f64>,
    pub membrane_potentials: Vec<f64>,
    pub tau_decay: f64,
    pub output_spikes: Vec<bool>,
}

impl NeuromorphicLayer {
    fn new(input_dim: usize, output_dim: usize, tau_decay: f64) -> Self {
        // Initialize weights with small random values using deterministic seed
        let mut seed: u64 = (input_dim as u64).wrapping_mul(7919).wrapping_add(output_dim as u64 * 6271);
        let scale = (2.0 / input_dim as f64).sqrt();
        let weights: Vec<Vec<f64>> = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| {
                        let v = lcg_next(&mut seed);
                        (v - 0.5) * 2.0 * scale
                    })
                    .collect()
            })
            .collect();

        let thresholds = vec![1.0; output_dim];
        let membrane_potentials = vec![0.0; output_dim];

        Self {
            weights,
            thresholds,
            membrane_potentials,
            tau_decay,
            output_spikes: vec![false; output_dim],
        }
    }

    /// Leaky integrate-and-fire forward pass.
    /// 1. Decay membrane potentials: V *= exp(-dt / tau)
    /// 2. Integrate input: V += W * input
    /// 3. Fire if V > threshold, then reset
    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let output_dim = self.weights.len();
        let mut output = vec![0.0; output_dim];

        // Exponential decay
        let decay = (-1.0 / self.tau_decay).exp();

        for i in 0..output_dim {
            // Leak: exponential decay of membrane potential
            self.membrane_potentials[i] *= decay;

            // Integrate: weighted sum of inputs
            let mut sum = 0.0;
            let w = &self.weights[i];
            for j in 0..w.len().min(input.len()) {
                sum += w[j] * input[j];
            }
            self.membrane_potentials[i] += sum;

            // Fire check
            if self.membrane_potentials[i] > self.thresholds[i] {
                self.output_spikes[i] = true;
                output[i] = 1.0;
                // Reset to resting potential
                self.membrane_potentials[i] = 0.0;
            } else {
                self.output_spikes[i] = false;
                output[i] = 0.0;
            }
        }

        output
    }
}

// ─── Approximate matmul implementations ─────────────────────────────

/// Full-precision matrix-vector multiplication
fn matmul_full(weights: &[Vec<f64>], input: &[f64]) -> Vec<f64> {
    weights.iter().map(|row| {
        row.iter().zip(input.iter()).map(|(w, x)| w * x).sum()
    }).collect()
}

/// Half-precision: quantize weights to 16-bit equivalent (round to ~3 decimal digits)
fn matmul_half(weights: &[Vec<f64>], input: &[f64]) -> Vec<f64> {
    weights.iter().map(|row| {
        row.iter().zip(input.iter()).map(|(w, x)| {
            // Simulate fp16 by rounding to ~3-4 significant digits
            let quantized_w = (w * 1024.0).round() / 1024.0;
            let quantized_x = (x * 1024.0).round() / 1024.0;
            quantized_w * quantized_x
        }).sum()
    }).collect()
}

/// Quarter-precision: random projection for dimensionality reduction
fn matmul_quarter(weights: &[Vec<f64>], input: &[f64]) -> Vec<f64> {
    let out_dim = weights.len();
    let in_dim = if out_dim > 0 { weights[0].len() } else { 0 };

    if in_dim == 0 {
        return vec![0.0; out_dim];
    }

    // Use random projection: project input to lower dimension, then approximate
    let proj_dim = (in_dim / 2).max(1);
    let mut seed: u64 = 42;

    // Create random projection matrix
    let scale = (in_dim as f64 / proj_dim as f64).sqrt();
    let mut projected_input = vec![0.0; proj_dim];
    for j in 0..proj_dim {
        for k in 0..in_dim.min(input.len()) {
            let r = lcg_next(&mut seed) - 0.5;
            projected_input[j] += r * input[k];
        }
        projected_input[j] *= scale / (in_dim as f64).sqrt();
    }

    // Approximate output using projected weights
    let mut output = vec![0.0; out_dim];
    for i in 0..out_dim {
        seed = (i as u64).wrapping_mul(7919).wrapping_add(13);
        let mut sum = 0.0;
        for j in 0..proj_dim {
            let w_proj: f64 = (0..((in_dim + proj_dim - 1) / proj_dim).min(in_dim))
                .map(|k| {
                    let idx = (j + k * proj_dim) % in_dim;
                    if idx < weights[i].len() { weights[i][idx] } else { 0.0 }
                })
                .sum::<f64>() / proj_dim as f64;
            sum += w_proj * projected_input[j];
        }
        output[i] = sum * scale;
    }
    output
}

/// Binary precision: use sign of weights only (+1/-1)
fn matmul_binary(weights: &[Vec<f64>], input: &[f64]) -> Vec<f64> {
    let scale_factor = weights.iter().flat_map(|row| row.iter())
        .map(|w| w.abs())
        .sum::<f64>()
        / weights.iter().map(|r| r.len()).sum::<usize>().max(1) as f64;

    weights.iter().map(|row| {
        let sum: f64 = row.iter().zip(input.iter()).map(|(w, x)| {
            let sign = if *w >= 0.0 { 1.0 } else { -1.0 };
            sign * x
        }).sum();
        sum * scale_factor
    }).collect()
}

fn approximate_matmul(weights: &[Vec<f64>], input: &[f64], precision: &ApproxPrecision) -> Vec<f64> {
    match precision {
        ApproxPrecision::Full => matmul_full(weights, input),
        ApproxPrecision::Half => matmul_half(weights, input),
        ApproxPrecision::Quarter => matmul_quarter(weights, input),
        ApproxPrecision::Binary => matmul_binary(weights, input),
    }
}

/// Select precision based on remaining energy fraction
fn select_precision_for_budget(budget: &EnergyBudget) -> ApproxPrecision {
    let frac = budget.remaining / budget.total_joules.max(1e-12);
    if frac > 0.75 {
        ApproxPrecision::Full
    } else if frac > 0.50 {
        ApproxPrecision::Half
    } else if frac > 0.25 {
        ApproxPrecision::Quarter
    } else {
        ApproxPrecision::Binary
    }
}

/// Estimate energy cost of a matmul given dimensions and precision
fn estimate_matmul_energy(out_dim: usize, in_dim: usize, precision: &ApproxPrecision) -> f64 {
    // Base cost: proportional to FLOPs (2 * out * in)
    let base = 2.0 * out_dim as f64 * in_dim as f64 * 1e-9; // nanojoules
    base * precision.energy_cost()
}

// ─── Energy gating ──────────────────────────────────────────────────

/// Learned gating: compute gate = sigmoid(gate_weights * input).
/// If gate < threshold, skip computation (output zeros).
fn energy_gate_forward(
    weights: &[Vec<f64>],
    input: &[f64],
    gate_weights: &[f64],
    threshold: f64,
) -> Vec<f64> {
    // Compute gate value: dot(gate_weights, input) passed through sigmoid
    let gate_input: f64 = gate_weights.iter().zip(input.iter())
        .map(|(g, x)| g * x)
        .sum();
    let gate = 1.0 / (1.0 + (-gate_input).exp());

    if gate < threshold {
        // Skip computation entirely — output zeros
        vec![0.0; weights.len()]
    } else {
        // Scale output by gate value
        let raw = matmul_full(weights, input);
        raw.iter().map(|v| v * gate).collect()
    }
}

// ─── Builtin functions ──────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("energy_budget_new".to_string(), FnDef::Builtin(builtin_energy_budget_new));
    env.functions.insert("energy_budget_remaining".to_string(), FnDef::Builtin(builtin_energy_budget_remaining));
    env.functions.insert("energy_budget_consume".to_string(), FnDef::Builtin(builtin_energy_budget_consume));
    env.functions.insert("energy_aware_forward".to_string(), FnDef::Builtin(builtin_energy_aware_forward));
    env.functions.insert("approximate_matmul".to_string(), FnDef::Builtin(builtin_approximate_matmul));
    env.functions.insert("event_driven_new".to_string(), FnDef::Builtin(builtin_event_driven_new));
    env.functions.insert("event_driven_step".to_string(), FnDef::Builtin(builtin_event_driven_step));
    env.functions.insert("neuromorphic_layer_new".to_string(), FnDef::Builtin(builtin_neuromorphic_layer_new));
    env.functions.insert("neuromorphic_forward".to_string(), FnDef::Builtin(builtin_neuromorphic_forward));
    env.functions.insert("dynamic_precision_forward".to_string(), FnDef::Builtin(builtin_dynamic_precision_forward));
    env.functions.insert("energy_gate".to_string(), FnDef::Builtin(builtin_energy_gate));
    env.functions.insert("energy_profile".to_string(), FnDef::Builtin(builtin_energy_profile));
}

// ─── energy_budget_new(total_joules, policy_str) → id ───────────────

fn builtin_energy_budget_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("energy_budget_new expects 2 args: (total_joules, policy_str)".into());
    }
    let total = value_to_f64(&args[0])?;
    let policy_str = value_to_string(&args[1])?;
    let policy = EnergyPolicy::from_str(&policy_str)?;
    let budget = EnergyBudget::new(total, policy);
    let id = env.next_energy_budget_id;
    env.next_energy_budget_id += 1;
    env.energy_budgets.insert(id, budget);
    Ok(Value::Int(id as i128))
}

// ─── energy_budget_remaining(id) → f64 ─────────────────────────────

fn builtin_energy_budget_remaining(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("energy_budget_remaining expects 1 arg: (id)".into());
    }
    let id = value_to_usize(&args[0])?;
    let budget = env.energy_budgets.get(&id).ok_or("no such energy budget")?;
    Ok(Value::Float(budget.remaining))
}

// ─── energy_budget_consume(id, joules) → bool ──────────────────────

fn builtin_energy_budget_consume(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("energy_budget_consume expects 2 args: (id, joules)".into());
    }
    let id = value_to_usize(&args[0])?;
    let joules = value_to_f64(&args[1])?;
    let budget = env.energy_budgets.get_mut(&id).ok_or("no such energy budget")?;
    let ok = budget.consume(joules);
    Ok(Value::Int(if ok { 1 } else { 0 }))
}

// ─── energy_aware_forward(budget_id, weights, input) → output ──────

fn builtin_energy_aware_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("energy_aware_forward expects 3 args: (budget_id, weights, input)".into());
    }
    let budget_id = value_to_usize(&args[0])?;
    let weights = value_to_f64_2d(&args[1])?;
    let input = value_to_f64_vec(&args[2])?;

    let out_dim = weights.len();
    let in_dim = if out_dim > 0 { weights[0].len() } else { 0 };

    // Auto-select precision based on remaining budget
    let budget = env.energy_budgets.get(&budget_id).ok_or("no such energy budget")?;
    let precision = select_precision_for_budget(budget);
    let cost = estimate_matmul_energy(out_dim, in_dim, &precision);

    // Clone needed info before mutable borrow
    let budget = env.energy_budgets.get_mut(&budget_id).ok_or("no such energy budget")?;
    let allowed = budget.consume(cost);

    let output = if !allowed {
        // Hard budget exceeded: return zeros
        vec![0.0; out_dim]
    } else {
        approximate_matmul(&weights, &input, &precision)
    };

    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

// ─── approximate_matmul(weights, input, precision_str) → output ────

fn builtin_approximate_matmul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let _ = env;
    if args.len() != 3 {
        return Err("approximate_matmul expects 3 args: (weights, input, precision_str)".into());
    }
    let weights = value_to_f64_2d(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let precision_str = value_to_string(&args[2])?;
    let precision = ApproxPrecision::from_str(&precision_str)?;
    let output = approximate_matmul(&weights, &input, &precision);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

// ─── event_driven_new(num_neurons, spike_threshold) → id ───────────

fn builtin_event_driven_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("event_driven_new expects 2 args: (num_neurons, spike_threshold)".into());
    }
    let num = value_to_usize(&args[0])?;
    let threshold = value_to_f64(&args[1])?;
    let executor = EventDrivenExecutor::new(num, threshold);
    let id = env.next_event_driven_id;
    env.next_event_driven_id += 1;
    env.event_driven_executors.insert(id, executor);
    Ok(Value::Int(id as i128))
}

// ─── event_driven_step(id, input_spikes) → output_spikes ───────────

fn builtin_event_driven_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("event_driven_step expects 2 args: (id, input_spikes)".into());
    }
    let id = value_to_usize(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let executor = env.event_driven_executors.get_mut(&id).ok_or("no such event-driven executor")?;
    let spikes = executor.step(&input);
    Ok(Value::Array(spikes.into_iter().map(|b| Value::Int(if b { 1 } else { 0 })).collect()))
}

// ─── neuromorphic_layer_new(input_dim, output_dim, tau_decay) → id ─

fn builtin_neuromorphic_layer_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("neuromorphic_layer_new expects 3 args: (input_dim, output_dim, tau_decay)".into());
    }
    let in_dim = value_to_usize(&args[0])?;
    let out_dim = value_to_usize(&args[1])?;
    let tau = value_to_f64(&args[2])?;
    let layer = NeuromorphicLayer::new(in_dim, out_dim, tau);
    let id = env.next_neuromorphic_id;
    env.next_neuromorphic_id += 1;
    env.neuromorphic_layers.insert(id, layer);
    Ok(Value::Int(id as i128))
}

// ─── neuromorphic_forward(id, input) → output ──────────────────────

fn builtin_neuromorphic_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("neuromorphic_forward expects 2 args: (id, input)".into());
    }
    let id = value_to_usize(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let layer = env.neuromorphic_layers.get_mut(&id).ok_or("no such neuromorphic layer")?;
    let output = layer.forward(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

// ─── dynamic_precision_forward(budget_id, weights, input) → output ─

fn builtin_dynamic_precision_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("dynamic_precision_forward expects 3 args: (budget_id, weights, input)".into());
    }
    let budget_id = value_to_usize(&args[0])?;
    let weights = value_to_f64_2d(&args[1])?;
    let input = value_to_f64_vec(&args[2])?;

    let out_dim = weights.len();
    let in_dim = if out_dim > 0 { weights[0].len() } else { 0 };

    let budget = env.energy_budgets.get(&budget_id).ok_or("no such energy budget")?;
    let precision = select_precision_for_budget(budget);
    let cost = estimate_matmul_energy(out_dim, in_dim, &precision);

    let budget = env.energy_budgets.get_mut(&budget_id).ok_or("no such energy budget")?;
    let allowed = budget.consume(cost);

    let output = if !allowed {
        vec![0.0; out_dim]
    } else {
        approximate_matmul(&weights, &input, &precision)
    };

    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

// ─── energy_gate(weights, input, gate_weights, threshold) → output ─

fn builtin_energy_gate(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let _ = env;
    if args.len() != 4 {
        return Err("energy_gate expects 4 args: (weights, input, gate_weights, threshold)".into());
    }
    let weights = value_to_f64_2d(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let gate_weights = value_to_f64_vec(&args[2])?;
    let threshold = value_to_f64(&args[3])?;
    let output = energy_gate_forward(&weights, &input, &gate_weights, threshold);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

// ─── energy_profile(budget_id) → stats HashMap ─────────────────────

fn builtin_energy_profile(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("energy_profile expects 1 arg: (budget_id)".into());
    }
    let id = value_to_usize(&args[0])?;
    let budget = env.energy_budgets.get(&id).ok_or("no such energy budget")?;
    let total = budget.total_joules;
    let remaining = budget.remaining;
    let consumed = budget.consumed();
    let efficiency = budget.efficiency();

    let mut map = Vec::new();
    map.push(Value::Array(vec![Value::String("total".into()), Value::Float(total)]));
    map.push(Value::Array(vec![Value::String("remaining".into()), Value::Float(remaining)]));
    map.push(Value::Array(vec![Value::String("consumed".into()), Value::Float(consumed)]));
    map.push(Value::Array(vec![Value::String("efficiency".into()), Value::Float(efficiency)]));

    Ok(Value::Array(map))
}

// ─── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_budget_hard() {
        let mut budget = EnergyBudget::new(10.0, EnergyPolicy::Hard);
        assert!(budget.consume(3.0));
        assert!((budget.remaining - 7.0).abs() < 1e-9);
        assert!(budget.consume(7.0));
        assert!((budget.remaining).abs() < 1e-9);
        assert!(!budget.consume(0.1)); // should fail
    }

    #[test]
    fn test_energy_budget_soft() {
        let mut budget = EnergyBudget::new(10.0, EnergyPolicy::Soft);
        assert!(budget.consume(15.0)); // soft allows overdraft
        assert!(budget.remaining < 0.0);
    }

    #[test]
    fn test_energy_budget_adaptive() {
        let mut budget = EnergyBudget::new(10.0, EnergyPolicy::Adaptive);
        assert!(budget.consume(5.0));
        assert!((budget.remaining - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_energy_budget_consumed_efficiency() {
        let mut budget = EnergyBudget::new(100.0, EnergyPolicy::Hard);
        budget.consume(25.0);
        assert!((budget.consumed() - 25.0).abs() < 1e-9);
        assert!((budget.efficiency() - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_approx_precision_from_str() {
        assert_eq!(ApproxPrecision::from_str("full").unwrap(), ApproxPrecision::Full);
        assert_eq!(ApproxPrecision::from_str("half").unwrap(), ApproxPrecision::Half);
        assert_eq!(ApproxPrecision::from_str("quarter").unwrap(), ApproxPrecision::Quarter);
        assert_eq!(ApproxPrecision::from_str("binary").unwrap(), ApproxPrecision::Binary);
        assert!(ApproxPrecision::from_str("foo").is_err());
    }

    #[test]
    fn test_matmul_full() {
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let input = vec![1.0, 1.0];
        let out = matmul_full(&weights, &input);
        assert!((out[0] - 3.0).abs() < 1e-9);
        assert!((out[1] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_matmul_half_close_to_full() {
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let input = vec![1.0, 1.0];
        let full = matmul_full(&weights, &input);
        let half = matmul_half(&weights, &input);
        for (f, h) in full.iter().zip(half.iter()) {
            assert!((f - h).abs() < 0.1, "half should be close to full");
        }
    }

    #[test]
    fn test_matmul_binary_sign_based() {
        let weights = vec![vec![1.0, -2.0], vec![-3.0, 4.0]];
        let input = vec![1.0, 1.0];
        let out = matmul_binary(&weights, &input);
        // Binary uses sign(w)*x, scaled by average |w|
        // signs: [+1, -1], [-1, +1]
        // raw: 1 - 1 = 0, -1 + 1 = 0 => both 0 scaled
        assert!(out.len() == 2);
    }

    #[test]
    fn test_matmul_quarter_produces_output() {
        let weights = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let input = vec![1.0, 1.0, 1.0];
        let out = matmul_quarter(&weights, &input);
        assert_eq!(out.len(), 2);
        // Quarter is approximate, just check it produces finite values
        assert!(out[0].is_finite());
        assert!(out[1].is_finite());
    }

    #[test]
    fn test_energy_cost_ordering() {
        assert!(ApproxPrecision::Full.energy_cost() > ApproxPrecision::Half.energy_cost());
        assert!(ApproxPrecision::Half.energy_cost() > ApproxPrecision::Quarter.energy_cost());
        assert!(ApproxPrecision::Quarter.energy_cost() > ApproxPrecision::Binary.energy_cost());
    }

    #[test]
    fn test_select_precision_for_budget() {
        let b100 = EnergyBudget::new(100.0, EnergyPolicy::Hard);
        assert_eq!(select_precision_for_budget(&b100), ApproxPrecision::Full);

        let mut b60 = EnergyBudget::new(100.0, EnergyPolicy::Hard);
        b60.remaining = 60.0;
        assert_eq!(select_precision_for_budget(&b60), ApproxPrecision::Half);

        let mut b30 = EnergyBudget::new(100.0, EnergyPolicy::Hard);
        b30.remaining = 30.0;
        assert_eq!(select_precision_for_budget(&b30), ApproxPrecision::Quarter);

        let mut b10 = EnergyBudget::new(100.0, EnergyPolicy::Hard);
        b10.remaining = 10.0;
        assert_eq!(select_precision_for_budget(&b10), ApproxPrecision::Binary);
    }

    #[test]
    fn test_event_driven_executor_basic() {
        let mut exec = EventDrivenExecutor::new(4, 0.5);

        // Input below threshold — no spikes
        let spikes = exec.step(&[0.1, 0.1, 0.1, 0.1]);
        assert!(spikes.iter().all(|s| !s));

        // Strong input on neuron 0
        let spikes = exec.step(&[2.0, 0.0, 0.0, 0.0]);
        assert!(spikes[0]); // should spike
    }

    #[test]
    fn test_event_driven_accumulation() {
        let mut exec = EventDrivenExecutor::new(1, 0.3);
        // Step 1: input 0.5 > threshold 0.3, so neuron is processed.
        // membrane becomes 0.5 > 0.3 => spike and reset
        let spikes = exec.step(&[0.5]);
        assert!(spikes[0]);
        assert!((exec.membrane_potentials[0]).abs() < 1e-9);

        // Step 2: input 0.35 > 0.3, processed. membrane = 0.35 > 0.3 => spike
        let spikes = exec.step(&[0.35]);
        assert!(spikes[0]);
    }

    #[test]
    fn test_event_driven_inactive_decay() {
        let mut exec = EventDrivenExecutor::new(2, 1.0);
        exec.membrane_potentials[1] = 0.5;
        // Only neuron 0 gets input above threshold
        let _ = exec.step(&[1.5, 0.0]);
        // Neuron 1 was inactive, so it decays: 0.5 * 0.9 = 0.45
        assert!((exec.membrane_potentials[1] - 0.45).abs() < 1e-9);
    }

    #[test]
    fn test_neuromorphic_layer_creation() {
        let layer = NeuromorphicLayer::new(4, 3, 5.0);
        assert_eq!(layer.weights.len(), 3);
        assert_eq!(layer.weights[0].len(), 4);
        assert_eq!(layer.membrane_potentials.len(), 3);
        assert_eq!(layer.thresholds.len(), 3);
        assert!((layer.tau_decay - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_neuromorphic_lif_forward() {
        let mut layer = NeuromorphicLayer::new(2, 2, 10.0);
        // Set known weights for predictable behavior
        layer.weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        layer.thresholds = vec![0.5, 0.5];

        let out = layer.forward(&[1.0, 0.3]);
        // Neuron 0: membrane += 1.0 > 0.5 => spikes
        assert!((out[0] - 1.0).abs() < 1e-9);
        // Neuron 1: membrane += 0.3 < 0.5 => no spike
        assert!((out[1] - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_neuromorphic_decay() {
        let mut layer = NeuromorphicLayer::new(2, 1, 2.0);
        layer.weights = vec![vec![0.3, 0.0]];
        layer.thresholds = vec![1.0];

        // First step: accumulate 0.3
        let _ = layer.forward(&[1.0, 0.0]);
        let v1 = layer.membrane_potentials[0];
        assert!((v1 - 0.3).abs() < 1e-9);

        // Second step: decayed 0.3 * exp(-1/2) + 0.3
        let _ = layer.forward(&[1.0, 0.0]);
        let decay = (-1.0_f64 / 2.0).exp();
        let expected = 0.3 * decay + 0.3;
        let v2 = layer.membrane_potentials[0];
        assert!((v2 - expected).abs() < 1e-6);
    }

    #[test]
    fn test_neuromorphic_spike_reset() {
        let mut layer = NeuromorphicLayer::new(1, 1, 100.0);
        layer.weights = vec![vec![2.0]];
        layer.thresholds = vec![1.0];

        let out = layer.forward(&[1.0]);
        // 2.0 > 1.0, should spike
        assert!((out[0] - 1.0).abs() < 1e-9);
        // Membrane should be reset
        assert!((layer.membrane_potentials[0]).abs() < 1e-9);
    }

    #[test]
    fn test_energy_gate_skip() {
        let weights = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let input = vec![1.0, 1.0];
        // Gate weights that produce large negative gate => sigmoid < threshold
        let gate_weights = vec![-10.0, -10.0];
        let threshold = 0.5;
        let out = energy_gate_forward(&weights, &input, &gate_weights, threshold);
        // Gate is sigmoid(-20) ≈ 0 < 0.5 => skip
        assert!(out.iter().all(|v| v.abs() < 1e-9));
    }

    #[test]
    fn test_energy_gate_pass() {
        let weights = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let input = vec![2.0, 3.0];
        // Gate weights that produce large positive gate => sigmoid > threshold
        let gate_weights = vec![10.0, 10.0];
        let threshold = 0.5;
        let out = energy_gate_forward(&weights, &input, &gate_weights, threshold);
        // Gate is sigmoid(50) ≈ 1.0 >= 0.5 => compute and scale by ~1.0
        assert!((out[0] - 2.0).abs() < 0.01);
        assert!((out[1] - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_estimate_matmul_energy() {
        let full_cost = estimate_matmul_energy(10, 10, &ApproxPrecision::Full);
        let half_cost = estimate_matmul_energy(10, 10, &ApproxPrecision::Half);
        let quarter_cost = estimate_matmul_energy(10, 10, &ApproxPrecision::Quarter);
        let binary_cost = estimate_matmul_energy(10, 10, &ApproxPrecision::Binary);
        assert!(full_cost > half_cost);
        assert!(half_cost > quarter_cost);
        assert!(quarter_cost > binary_cost);
        assert!(binary_cost > 0.0);
    }

    #[test]
    fn test_energy_policy_from_str() {
        assert!(matches!(EnergyPolicy::from_str("hard").unwrap(), EnergyPolicy::Hard));
        assert!(matches!(EnergyPolicy::from_str("soft").unwrap(), EnergyPolicy::Soft));
        assert!(matches!(EnergyPolicy::from_str("adaptive").unwrap(), EnergyPolicy::Adaptive));
        assert!(EnergyPolicy::from_str("invalid").is_err());
    }

    #[test]
    fn test_approx_config() {
        let config = ApproxConfig {
            precision: ApproxPrecision::Half,
            skip_threshold: 0.01,
        };
        assert_eq!(config.precision, ApproxPrecision::Half);
        assert!((config.skip_threshold - 0.01).abs() < 1e-9);
    }

    #[test]
    fn test_matmul_empty() {
        let weights: Vec<Vec<f64>> = vec![];
        let input: Vec<f64> = vec![];
        let out = matmul_full(&weights, &input);
        assert!(out.is_empty());
    }

    #[test]
    fn test_event_driven_empty_input() {
        let mut exec = EventDrivenExecutor::new(3, 0.5);
        let spikes = exec.step(&[]);
        assert_eq!(spikes.len(), 3);
        assert!(spikes.iter().all(|s| !s));
    }

    #[test]
    fn test_full_budget_lifecycle() {
        let mut budget = EnergyBudget::new(1.0, EnergyPolicy::Hard);
        assert!((budget.total_joules - 1.0).abs() < 1e-9);
        assert!((budget.remaining - 1.0).abs() < 1e-9);
        assert!((budget.consumed()).abs() < 1e-9);
        assert!((budget.efficiency()).abs() < 1e-9);

        budget.consume(0.5);
        assert!((budget.consumed() - 0.5).abs() < 1e-9);
        assert!((budget.efficiency() - 0.5).abs() < 1e-9);

        budget.consume(0.5);
        assert!((budget.consumed() - 1.0).abs() < 1e-9);
        assert!((budget.efficiency() - 1.0).abs() < 1e-9);
    }
}
