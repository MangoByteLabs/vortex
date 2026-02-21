/// Energy-Based Models with Langevin MCMC sampling.
///
/// Implements generative modeling via energy functions and contrastive divergence training.

use std::collections::HashMap;

// ── EnergyFunction trait ──────────────────────────────────────────────

pub trait EnergyFunction {
    fn energy(&self, x: &[f64]) -> f64;
    fn gradient(&self, x: &[f64]) -> Vec<f64>;
}

// ── MLP helpers ───────────────────────────────────────────────────────

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn relu_deriv(x: f64) -> f64 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Simple deterministic PRNG (xorshift64) for reproducible sampling without external deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 { 0xdeadbeef } else { seed })
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform [0, 1)
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    /// Approximate standard normal via Box-Muller.
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    fn range(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

// ── MLPEnergy ─────────────────────────────────────────────────────────

/// Neural network that maps input x -> scalar energy.
pub struct MLPEnergy {
    /// weights[i] is (rows x cols) matrix for layer i
    pub weights: Vec<Vec<Vec<f64>>>,
    /// biases[i] is vector for layer i
    pub biases: Vec<Vec<f64>>,
    pub input_dim: usize,
    pub hidden_dims: Vec<usize>,
}

impl MLPEnergy {
    pub fn new(input_dim: usize, hidden_dims: &[usize]) -> Self {
        let mut rng = Rng::new(42);
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut prev = input_dim;
        for &h in hidden_dims {
            let scale = (2.0 / prev as f64).sqrt();
            let w: Vec<Vec<f64>> = (0..h)
                .map(|_| (0..prev).map(|_| rng.normal() * scale).collect())
                .collect();
            weights.push(w);
            biases.push(vec![0.0; h]);
            prev = h;
        }
        // Final layer: hidden -> 1
        let scale = (2.0 / prev as f64).sqrt();
        let w: Vec<Vec<f64>> = vec![(0..prev).map(|_| rng.normal() * scale).collect()];
        weights.push(w);
        biases.push(vec![0.0; 1]);

        MLPEnergy { weights, biases, input_dim, hidden_dims: hidden_dims.to_vec() }
    }

    /// Forward pass returning (energy_scalar, pre_activations for each hidden layer).
    fn forward_with_cache(&self, x: &[f64]) -> (f64, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let mut activations: Vec<Vec<f64>> = Vec::new(); // post-activation
        let mut pre_acts: Vec<Vec<f64>> = Vec::new();
        let mut current = x.to_vec();
        activations.push(current.clone());

        for i in 0..self.weights.len() {
            let w = &self.weights[i];
            let b = &self.biases[i];
            let mut z = vec![0.0; w.len()];
            for (r, row) in w.iter().enumerate() {
                let mut s = b[r];
                for (c, &wc) in row.iter().enumerate() {
                    s += wc * current[c];
                }
                z[r] = s;
            }
            pre_acts.push(z.clone());
            if i < self.weights.len() - 1 {
                // hidden layer: apply ReLU
                current = z.iter().map(|&v| relu(v)).collect();
            } else {
                // output layer: no activation
                current = z;
            }
            activations.push(current.clone());
        }
        (current[0], pre_acts, activations)
    }

    pub fn forward(&self, x: &[f64]) -> f64 {
        self.forward_with_cache(x).0
    }

    /// Analytical gradient ∂E/∂x via backprop through the MLP.
    pub fn gradient_analytical(&self, x: &[f64]) -> Vec<f64> {
        let (_, pre_acts, activations) = self.forward_with_cache(x);
        let n_layers = self.weights.len();
        // Backprop: dL/d(output) = 1.0
        let mut delta = vec![1.0f64];

        for i in (0..n_layers).rev() {
            // delta is gradient w.r.t. layer i output (pre-activation for output, post-relu for hidden)
            // For hidden layers, multiply by relu derivative
            if i < n_layers - 1 {
                for (j, d) in delta.iter_mut().enumerate() {
                    *d *= relu_deriv(pre_acts[i][j]);
                }
            }
            // Propagate to previous layer: delta_prev = W^T * delta
            let w = &self.weights[i];
            let prev_size = activations[i].len();
            let mut prev_delta = vec![0.0; prev_size];
            for (r, row) in w.iter().enumerate() {
                for (c, &wc) in row.iter().enumerate() {
                    prev_delta[c] += wc * delta[r];
                }
            }
            delta = prev_delta;
        }
        delta
    }

    /// Gradient of energy w.r.t. parameters (weights and biases), for training.
    pub fn param_gradient(&self, x: &[f64]) -> (Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let (_, pre_acts, activations) = self.forward_with_cache(x);
        let n_layers = self.weights.len();
        let mut dw: Vec<Vec<Vec<f64>>> = Vec::new();
        let mut db: Vec<Vec<f64>> = Vec::new();
        // Initialize storage
        for i in 0..n_layers {
            dw.push(vec![vec![0.0; self.weights[i][0].len()]; self.weights[i].len()]);
            db.push(vec![0.0; self.biases[i].len()]);
        }

        let mut delta = vec![1.0f64];
        for i in (0..n_layers).rev() {
            if i < n_layers - 1 {
                for (j, d) in delta.iter_mut().enumerate() {
                    *d *= relu_deriv(pre_acts[i][j]);
                }
            }
            // dW[i] = delta * activation[i]^T, dB[i] = delta
            for (r, d) in delta.iter().enumerate() {
                db[i][r] = *d;
                for (c, &a) in activations[i].iter().enumerate() {
                    dw[i][r][c] = d * a;
                }
            }
            // Propagate
            if i > 0 {
                let w = &self.weights[i];
                let prev_size = activations[i].len();
                let mut prev_delta = vec![0.0; prev_size];
                for (r, row) in w.iter().enumerate() {
                    for (c, &wc) in row.iter().enumerate() {
                        prev_delta[c] += wc * delta[r];
                    }
                }
                delta = prev_delta;
            }
        }
        (dw, db)
    }

    /// Apply parameter update: θ -= lr * grad
    pub fn apply_gradient(&mut self, dw: &[Vec<Vec<f64>>], db: &[Vec<f64>], lr: f64) {
        for i in 0..self.weights.len() {
            for r in 0..self.weights[i].len() {
                self.biases[i][r] -= lr * db[i][r];
                for c in 0..self.weights[i][r].len() {
                    self.weights[i][r][c] -= lr * dw[i][r][c];
                }
            }
        }
    }
}

impl EnergyFunction for MLPEnergy {
    fn energy(&self, x: &[f64]) -> f64 {
        self.forward(x)
    }
    fn gradient(&self, x: &[f64]) -> Vec<f64> {
        self.gradient_analytical(x)
    }
}

// ── LangevinSampler ──────────────────────────────────────────────────

pub struct LangevinSampler {
    pub step_size: f64,
    pub noise_scale: f64,
    pub n_steps: usize,
    pub persistent_chain: Option<Vec<f64>>,
    rng: Rng,
}

impl LangevinSampler {
    pub fn new(step_size: f64, noise_scale: f64, n_steps: usize) -> Self {
        Self { step_size, noise_scale, n_steps, persistent_chain: None, rng: Rng::new(123) }
    }

    /// Run one step of Langevin dynamics: x_{t+1} = x_t - step_size * ∂E/∂x + noise
    fn langevin_step(&mut self, energy_fn: &dyn EnergyFunction, x: &mut Vec<f64>) {
        let grad = energy_fn.gradient(x);
        for (xi, gi) in x.iter_mut().zip(grad.iter()) {
            *xi -= self.step_size * gi + self.noise_scale * self.rng.normal();
        }
    }

    /// Run n_steps of Langevin dynamics from initial point.
    pub fn sample(&mut self, energy_fn: &dyn EnergyFunction, initial: &[f64], n_steps: usize) -> Vec<f64> {
        let mut x = initial.to_vec();
        for _ in 0..n_steps {
            self.langevin_step(energy_fn, &mut x);
        }
        x
    }

    /// Run full MCMC chain collecting samples after burn-in.
    pub fn sample_chain(
        &mut self,
        energy_fn: &dyn EnergyFunction,
        initial: &[f64],
        n_samples: usize,
        burn_in: usize,
    ) -> Vec<Vec<f64>> {
        let mut x = initial.to_vec();
        // Burn-in
        for _ in 0..burn_in {
            self.langevin_step(energy_fn, &mut x);
        }
        // Collect
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            for _ in 0..self.n_steps {
                self.langevin_step(energy_fn, &mut x);
            }
            samples.push(x.clone());
        }
        samples
    }

    /// Sample using persistent chain (PCD).
    pub fn sample_persistent(&mut self, energy_fn: &dyn EnergyFunction, dim: usize, n_steps: usize) -> Vec<f64> {
        let init = match self.persistent_chain.take() {
            Some(c) => c,
            None => (0..dim).map(|_| self.rng.normal()).collect(),
        };
        let result = self.sample(energy_fn, &init, n_steps);
        self.persistent_chain = Some(result.clone());
        result
    }
}

// ── ReplayBuffer ─────────────────────────────────────────────────────

pub struct ReplayBuffer {
    pub buffer: Vec<Vec<f64>>,
    pub max_size: usize,
    pub reinit_probability: f64,
    rng: Rng,
}

impl ReplayBuffer {
    pub fn new(max_size: usize, reinit_probability: f64) -> Self {
        Self { buffer: Vec::new(), max_size, reinit_probability, rng: Rng::new(456) }
    }

    pub fn add(&mut self, sample: Vec<f64>) {
        if self.buffer.len() >= self.max_size {
            let idx = self.rng.range(self.buffer.len());
            self.buffer[idx] = sample;
        } else {
            self.buffer.push(sample);
        }
    }

    pub fn sample_random(&mut self) -> Option<&[f64]> {
        if self.buffer.is_empty() {
            return None;
        }
        let idx = self.rng.range(self.buffer.len());
        Some(&self.buffer[idx])
    }

    /// Get initialization: either from buffer or random, based on reinit_probability.
    pub fn get_initial(&mut self, dim: usize) -> Vec<f64> {
        let r = self.rng.uniform();
        if r < self.reinit_probability || self.buffer.is_empty() {
            // Fresh random init
            (0..dim).map(|_| self.rng.normal()).collect()
        } else {
            self.sample_random().unwrap().to_vec()
        }
    }

    pub fn len(&self) -> usize {
        self.buffer.len()
    }
}

// ── Contrastive Divergence ───────────────────────────────────────────

pub struct ContrastiveDivergence;

impl ContrastiveDivergence {
    /// CD-k: returns (avg_pos_energy, avg_neg_energy, (dw, db) gradient).
    pub fn cd_k(
        model: &MLPEnergy,
        sampler: &mut LangevinSampler,
        buffer: &mut ReplayBuffer,
        positive_data: &[Vec<f64>],
        k_steps: usize,
    ) -> (f64, f64, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let n = positive_data.len().max(1);
        let dim = model.input_dim;

        // Positive phase
        let mut avg_pos = 0.0;
        let mut pos_dw: Vec<Vec<Vec<f64>>> = model.weights.iter()
            .map(|l| vec![vec![0.0; l[0].len()]; l.len()])
            .collect();
        let mut pos_db: Vec<Vec<f64>> = model.biases.iter()
            .map(|b| vec![0.0; b.len()])
            .collect();

        for x in positive_data {
            avg_pos += model.forward(x);
            let (dw, db) = model.param_gradient(x);
            for i in 0..pos_dw.len() {
                for r in 0..pos_dw[i].len() {
                    pos_db[i][r] += db[i][r];
                    for c in 0..pos_dw[i][r].len() {
                        pos_dw[i][r][c] += dw[i][r][c];
                    }
                }
            }
        }
        avg_pos /= n as f64;

        // Negative phase
        let mut avg_neg = 0.0;
        let mut neg_dw: Vec<Vec<Vec<f64>>> = model.weights.iter()
            .map(|l| vec![vec![0.0; l[0].len()]; l.len()])
            .collect();
        let mut neg_db: Vec<Vec<f64>> = model.biases.iter()
            .map(|b| vec![0.0; b.len()])
            .collect();

        for _ in 0..n {
            let init = buffer.get_initial(dim);
            let neg_sample = sampler.sample(model, &init, k_steps);
            avg_neg += model.forward(&neg_sample);
            let (dw, db) = model.param_gradient(&neg_sample);
            for i in 0..neg_dw.len() {
                for r in 0..neg_dw[i].len() {
                    neg_db[i][r] += db[i][r];
                    for c in 0..neg_dw[i][r].len() {
                        neg_dw[i][r][c] += dw[i][r][c];
                    }
                }
            }
            buffer.add(neg_sample);
        }
        avg_neg /= n as f64;

        // Gradient = pos - neg (we want to minimize pos energy, maximize neg energy)
        let mut grad_w = pos_dw;
        let mut grad_b = pos_db;
        for i in 0..grad_w.len() {
            for r in 0..grad_w[i].len() {
                grad_b[i][r] = (grad_b[i][r] - neg_db[i][r]) / n as f64;
                for c in 0..grad_w[i][r].len() {
                    grad_w[i][r][c] = (grad_w[i][r][c] - neg_dw[i][r][c]) / n as f64;
                }
            }
        }

        (avg_pos, avg_neg, grad_w, grad_b)
    }

    /// Persistent CD variant (uses persistent chain in sampler).
    pub fn persistent_cd(
        model: &MLPEnergy,
        sampler: &mut LangevinSampler,
        positive_data: &[Vec<f64>],
        k_steps: usize,
    ) -> (f64, f64, Vec<Vec<Vec<f64>>>, Vec<Vec<f64>>) {
        let n = positive_data.len().max(1);
        let dim = model.input_dim;

        let mut avg_pos = 0.0;
        let mut pos_dw: Vec<Vec<Vec<f64>>> = model.weights.iter()
            .map(|l| vec![vec![0.0; l[0].len()]; l.len()])
            .collect();
        let mut pos_db: Vec<Vec<f64>> = model.biases.iter()
            .map(|b| vec![0.0; b.len()])
            .collect();

        for x in positive_data {
            avg_pos += model.forward(x);
            let (dw, db) = model.param_gradient(x);
            for i in 0..pos_dw.len() {
                for r in 0..pos_dw[i].len() {
                    pos_db[i][r] += db[i][r];
                    for c in 0..pos_dw[i][r].len() {
                        pos_dw[i][r][c] += dw[i][r][c];
                    }
                }
            }
        }
        avg_pos /= n as f64;

        // Negative: use persistent chain
        let neg_sample = sampler.sample_persistent(model, dim, k_steps);
        let avg_neg = model.forward(&neg_sample);
        let (ndw, ndb) = model.param_gradient(&neg_sample);

        let mut grad_w = pos_dw;
        let mut grad_b = pos_db;
        for i in 0..grad_w.len() {
            for r in 0..grad_w[i].len() {
                grad_b[i][r] = grad_b[i][r] / n as f64 - ndb[i][r];
                for c in 0..grad_w[i][r].len() {
                    grad_w[i][r][c] = grad_w[i][r][c] / n as f64 - ndw[i][r][c];
                }
            }
        }

        (avg_pos, avg_neg, grad_w, grad_b)
    }
}

// ── EBMStats ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EBMStats {
    pub avg_pos_energy: f64,
    pub avg_neg_energy: f64,
    pub energy_gap: f64,
    pub sample_quality: f64,
}

// ── EBMModel ─────────────────────────────────────────────────────────

pub struct EBMModel {
    pub energy_fn: MLPEnergy,
    pub sampler: LangevinSampler,
    pub buffer: ReplayBuffer,
    pub stats: EBMStats,
    pub k_steps: usize,
}

impl EBMModel {
    pub fn new(input_dim: usize, hidden_dims: &[usize]) -> Self {
        Self {
            energy_fn: MLPEnergy::new(input_dim, hidden_dims),
            sampler: LangevinSampler::new(0.01, 0.005, 20),
            buffer: ReplayBuffer::new(1000, 0.05),
            stats: EBMStats { avg_pos_energy: 0.0, avg_neg_energy: 0.0, energy_gap: 0.0, sample_quality: 0.0 },
            k_steps: 20,
        }
    }

    /// One training step with CD-k. Returns loss = avg_pos_energy - avg_neg_energy.
    pub fn train_step(&mut self, positive_data: &[Vec<f64>], lr: f64) -> f64 {
        let (pos_e, neg_e, dw, db) = ContrastiveDivergence::cd_k(
            &self.energy_fn, &mut self.sampler, &mut self.buffer,
            positive_data, self.k_steps,
        );
        self.energy_fn.apply_gradient(&dw, &db, lr);

        // Compute sample quality: mean distance of negative samples to nearest positive
        let sample_quality = if !self.buffer.buffer.is_empty() && !positive_data.is_empty() {
            let mut total_dist = 0.0;
            let check = self.buffer.buffer.len().min(10);
            for s in &self.buffer.buffer[..check] {
                let min_d = positive_data.iter().map(|p| {
                    s.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt()
                }).fold(f64::MAX, f64::min);
                total_dist += min_d;
            }
            total_dist / check as f64
        } else {
            0.0
        };

        self.stats = EBMStats {
            avg_pos_energy: pos_e,
            avg_neg_energy: neg_e,
            energy_gap: pos_e - neg_e,
            sample_quality,
        };
        pos_e - neg_e
    }

    /// Generate samples via MCMC.
    pub fn generate(&mut self, n_samples: usize) -> Vec<Vec<f64>> {
        let dim = self.energy_fn.input_dim;
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let init = self.buffer.get_initial(dim);
            let s = self.sampler.sample(&self.energy_fn, &init, self.k_steps * 2);
            samples.push(s);
        }
        samples
    }

    /// Anomaly score (energy value).
    pub fn score(&self, x: &[f64]) -> f64 {
        self.energy_fn.energy(x)
    }
}

// ── Interpreter integration ──────────────────────────────────────────

use crate::interpreter::{Env, Value, FnDef};

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

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("ebm_new".to_string(), FnDef::Builtin(builtin_ebm_new));
    env.functions.insert("ebm_train_step".to_string(), FnDef::Builtin(builtin_ebm_train_step));
    env.functions.insert("ebm_generate".to_string(), FnDef::Builtin(builtin_ebm_generate));
    env.functions.insert("ebm_score".to_string(), FnDef::Builtin(builtin_ebm_score));
}

fn builtin_ebm_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ebm_new expects 2 args: (input_dim, hidden_dims)".into()); }
    let input_dim = value_to_usize(&args[0])?;
    let hidden_dims: Vec<usize> = match &args[1] {
        Value::Array(arr) => arr.iter().map(value_to_usize).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("hidden_dims must be an array".into()),
    };
    let model = EBMModel::new(input_dim, &hidden_dims);
    let id = env.next_ebm_id;
    env.next_ebm_id += 1;
    env.ebm_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_ebm_train_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ebm_train_step expects 3 args: (id, data, lr)".into()); }
    let id = value_to_usize(&args[0])?;
    let data = value_to_f64_2d(&args[1])?;
    let lr = value_to_f64(&args[2])?;
    let model = env.ebm_models.get_mut(&id).ok_or("no such EBM model")?;
    let loss = model.train_step(&data, lr);
    Ok(Value::Float(loss))
}

fn builtin_ebm_generate(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ebm_generate expects 2 args: (id, n_samples)".into()); }
    let id = value_to_usize(&args[0])?;
    let n = value_to_usize(&args[1])?;
    let model = env.ebm_models.get_mut(&id).ok_or("no such EBM model")?;
    let samples = model.generate(n);
    let val = Value::Array(samples.into_iter().map(|s| {
        Value::Array(s.into_iter().map(Value::Float).collect())
    }).collect());
    Ok(val)
}

fn builtin_ebm_score(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ebm_score expects 2 args: (id, input)".into()); }
    let id = value_to_usize(&args[0])?;
    let input = value_to_f64_vec(&args[1])?;
    let model = env.ebm_models.get(&id).ok_or("no such EBM model")?;
    Ok(Value::Float(model.score(&input)))
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mlp_energy_forward() {
        let mlp = MLPEnergy::new(2, &[4]);
        let e = mlp.forward(&[1.0, 0.5]);
        // Just check it returns a finite number
        assert!(e.is_finite());
    }

    #[test]
    fn test_mlp_energy_gradient_finite() {
        let mlp = MLPEnergy::new(3, &[4, 4]);
        let g = mlp.gradient(&[1.0, 2.0, 3.0]);
        assert_eq!(g.len(), 3);
        for &gi in &g {
            assert!(gi.is_finite());
        }
    }

    #[test]
    fn test_gradient_numerical_matches_analytical() {
        let mlp = MLPEnergy::new(2, &[4]);
        let x = vec![0.5, -0.3];
        let analytical = mlp.gradient_analytical(&x);
        let eps = 1e-5;
        for i in 0..x.len() {
            let mut xp = x.clone();
            let mut xm = x.clone();
            xp[i] += eps;
            xm[i] -= eps;
            let numerical = (mlp.forward(&xp) - mlp.forward(&xm)) / (2.0 * eps);
            assert!((analytical[i] - numerical).abs() < 1e-4,
                "dim {}: analytical={} numerical={}", i, analytical[i], numerical);
        }
    }

    #[test]
    fn test_langevin_sampler_moves() {
        let mlp = MLPEnergy::new(2, &[4]);
        let mut sampler = LangevinSampler::new(0.01, 0.005, 10);
        let init = vec![0.0, 0.0];
        let result = sampler.sample(&mlp, &init, 50);
        assert_eq!(result.len(), 2);
        // Should have moved from origin
        let dist: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(dist > 0.0, "sampler should move from initial point");
    }

    #[test]
    fn test_sample_chain() {
        let mlp = MLPEnergy::new(2, &[4]);
        let mut sampler = LangevinSampler::new(0.01, 0.005, 5);
        let chain = sampler.sample_chain(&mlp, &[0.0, 0.0], 10, 5);
        assert_eq!(chain.len(), 10);
        for s in &chain {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn test_replay_buffer() {
        let mut buf = ReplayBuffer::new(5, 0.5);
        assert_eq!(buf.len(), 0);
        for i in 0..10 {
            buf.add(vec![i as f64]);
        }
        assert!(buf.len() <= 5);
        let init = buf.get_initial(2);
        assert_eq!(init.len(), 2);
    }

    #[test]
    fn test_contrastive_divergence() {
        let mlp = MLPEnergy::new(2, &[4]);
        let mut sampler = LangevinSampler::new(0.01, 0.005, 5);
        let mut buffer = ReplayBuffer::new(100, 0.05);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let (pos, neg, dw, db) = ContrastiveDivergence::cd_k(&mlp, &mut sampler, &mut buffer, &data, 5);
        assert!(pos.is_finite());
        assert!(neg.is_finite());
        assert!(!dw.is_empty());
        assert!(!db.is_empty());
    }

    #[test]
    fn test_ebm_model_train_and_generate() {
        let mut model = EBMModel::new(2, &[4]);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        // Run a few training steps
        for _ in 0..3 {
            let loss = model.train_step(&data, 0.01);
            assert!(loss.is_finite());
        }
        let samples = model.generate(5);
        assert_eq!(samples.len(), 5);
        for s in &samples {
            assert_eq!(s.len(), 2);
        }
    }

    #[test]
    fn test_ebm_score() {
        let model = EBMModel::new(2, &[4]);
        let s1 = model.score(&[0.0, 0.0]);
        let s2 = model.score(&[1.0, 1.0]);
        assert!(s1.is_finite());
        assert!(s2.is_finite());
    }

    #[test]
    fn test_persistent_cd() {
        let mlp = MLPEnergy::new(2, &[4]);
        let mut sampler = LangevinSampler::new(0.01, 0.005, 5);
        let data = vec![vec![1.0, 0.0]];
        let (pos, neg, _, _) = ContrastiveDivergence::persistent_cd(&mlp, &mut sampler, &data, 10);
        assert!(pos.is_finite());
        assert!(neg.is_finite());
        // Persistent chain should be set now
        assert!(sampler.persistent_chain.is_some());
    }

    #[test]
    fn test_ebm_stats_updated() {
        let mut model = EBMModel::new(2, &[4]);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        model.train_step(&data, 0.01);
        assert!(model.stats.avg_pos_energy.is_finite());
        assert!(model.stats.avg_neg_energy.is_finite());
        assert!(model.stats.energy_gap.is_finite());
    }
}
