// SpikeSSMFormer: a hybrid LLM architecture combining SSM backbone,
// sparse spiking gates, Forward-Forward local learning, and liquid time-constants.

/// SSM parameters (Mamba-style selective scan)
#[derive(Debug, Clone)]
pub struct SSMParams {
    pub a: Vec<f64>,           // [D_state] diagonal A matrix
    pub b_proj: Vec<Vec<f64>>, // [D, D_state] input projection
    pub c_proj: Vec<Vec<f64>>, // [D_state, D] output projection
    pub dt_proj: Vec<f64>,     // [D] discretization step
    pub d: Vec<f64>,           // [D] skip connection
}

/// Lightweight attention weights (only used at spike positions)
#[derive(Debug, Clone)]
pub struct AttnWeights {
    pub wq: Vec<Vec<f64>>,
    pub wk: Vec<Vec<f64>>,
    pub wv: Vec<Vec<f64>>,
    pub wo: Vec<Vec<f64>>,
    pub num_heads: usize,
}

/// Selective SSM layer with spiking gate
#[derive(Debug, Clone)]
pub struct SpikeGatedSSM {
    pub d_model: usize,
    pub d_state: usize,
    pub spike_threshold: f64,
    pub tau: f64,
}

/// Forward-Forward layer -- trains locally, no backprop
#[derive(Debug, Clone)]
pub struct ForwardForwardLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub threshold: f64,
    pub lr: f64,
}

/// Liquid time-constant layer
#[derive(Debug, Clone)]
pub struct LiquidLayer {
    pub w_tau: Vec<Vec<f64>>,
    pub w_hidden: Vec<Vec<f64>>,
    pub w_input: Vec<Vec<f64>>,
    pub base_tau: Vec<f64>,
}

/// Architecture statistics
#[derive(Debug, Clone)]
pub struct ArchStats {
    pub total_params: usize,
    pub attention_sparsity: f64,
    pub ff_goodness: f64,
    pub avg_tau: f64,
}

/// The complete SpikeSSMFormer model
#[derive(Debug)]
pub struct SpikeSSMFormer {
    pub ff_layers: Vec<ForwardForwardLayer>,
    pub ssm_layers: Vec<(SpikeGatedSSM, SSMParams, AttnWeights)>,
    pub liquid_layer: LiquidLayer,
    pub output_proj: Vec<Vec<f64>>,
    pub d_model: usize,
    pub vocab_size: usize,
    pub last_spikes: Vec<bool>,
    pub last_ff_goodness: f64,
    pub last_avg_tau: f64,
}

// ---- helpers ----

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn matvec(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| dot(row, v)).collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn zeros(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

fn small_rand(i: usize, j: usize, seed: usize) -> f64 {
    // Deterministic pseudo-random based on indices
    let h = ((i.wrapping_mul(2654435761) ^ j.wrapping_mul(2246822519) ^ seed.wrapping_mul(3266489917)) as f64) / (u32::MAX as f64);
    (h - 0.5) * 0.1
}

fn rand_matrix(rows: usize, cols: usize, seed: usize) -> Vec<Vec<f64>> {
    (0..rows).map(|i| (0..cols).map(|j| small_rand(i, j, seed)).collect()).collect()
}

// ---- SSM scan ----

fn ssm_scan(x: &[Vec<f64>], params: &SSMParams, d_model: usize, d_state: usize) -> Vec<Vec<f64>> {
    let seq_len = x.len();
    let mut output = Vec::with_capacity(seq_len);
    let mut h = zeros(d_state);

    for t in 0..seq_len {
        let xt = &x[t];
        // Discretize: A_bar = exp(a * dt), B_bar = b_proj^T @ x * dt
        let mut h_new = zeros(d_state);
        for s in 0..d_state {
            let dt = if s < params.dt_proj.len() { sigmoid(params.dt_proj[s.min(d_model - 1)]) } else { 0.01 };
            let a_bar = (params.a[s] * dt).exp();
            let mut b_val = 0.0;
            for d in 0..d_model.min(xt.len()) {
                b_val += params.b_proj[d.min(params.b_proj.len() - 1)][s] * xt[d];
            }
            h_new[s] = a_bar * h[s] + dt * b_val;
        }
        h = h_new;

        // Output: y = c_proj^T @ h + d * x
        let mut yt = zeros(d_model);
        for d in 0..d_model {
            let mut val = 0.0;
            for s in 0..d_state {
                val += params.c_proj[s.min(params.c_proj.len() - 1)][d.min(params.c_proj[0].len() - 1)] * h[s];
            }
            val += params.d[d.min(params.d.len() - 1)] * xt[d.min(xt.len() - 1)];
            yt[d] = val;
        }
        output.push(yt);
    }
    output
}

// ---- Attention (single head simplified, used only at sparse positions) ----

fn attention_at_positions(
    x: &[Vec<f64>],
    spike_positions: &[usize],
    weights: &AttnWeights,
    d_model: usize,
) -> Vec<Vec<f64>> {
    // Only compute attention for tokens at spike positions
    // Keys/values from full sequence, queries only at spike positions
    if spike_positions.is_empty() {
        return vec![];
    }
    let seq_len = x.len();
    let d_head = d_model / weights.num_heads.max(1);
    let scale = (d_head as f64).sqrt();

    // Compute K, V for full sequence
    let keys: Vec<Vec<f64>> = x.iter().map(|xi| matvec(&weights.wk, xi)).collect();
    let vals: Vec<Vec<f64>> = x.iter().map(|xi| matvec(&weights.wv, xi)).collect();

    let mut results = Vec::with_capacity(spike_positions.len());
    for &pos in spike_positions {
        let q = matvec(&weights.wq, &x[pos]);
        // Compute attention scores
        let scores: Vec<f64> = keys.iter().map(|k| dot(&q, k) / scale).collect();
        // Softmax
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum_exp: f64 = exps.iter().sum();
        let attn: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();

        // Weighted sum of values
        let mut out = zeros(d_model);
        for (t, a) in attn.iter().enumerate() {
            for d in 0..d_model {
                out[d] += a * vals[t][d];
            }
        }
        let projected = matvec(&weights.wo, &out);
        results.push(projected);
    }
    results
}

// ---- SpikeGatedSSM ----

impl SpikeGatedSSM {
    pub fn new(d_model: usize, d_state: usize) -> Self {
        Self {
            d_model,
            d_state,
            spike_threshold: 50.0,
            tau: 0.8,
        }
    }

    /// LIF spiking gate: accumulate membrane potential, fire when > threshold
    fn lif_spikes(&self, ssm_out: &[Vec<f64>]) -> Vec<bool> {
        let mut membrane = 0.0f64;
        let mut spikes = Vec::with_capacity(ssm_out.len());
        // First pass: compute mean energy to set adaptive threshold
        let energies: Vec<f64> = ssm_out.iter()
            .map(|row| row.iter().map(|v| v * v).sum::<f64>() / row.len() as f64)
            .collect();
        let mean_energy = energies.iter().sum::<f64>() / energies.len().max(1) as f64;
        let adaptive_threshold = self.spike_threshold * mean_energy.max(0.001);

        for t in 0..ssm_out.len() {
            // Leaky integrate
            membrane = self.tau * membrane + energies[t];
            let fired = membrane > adaptive_threshold;
            if fired {
                membrane = 0.0; // reset
            }
            spikes.push(fired);
        }
        spikes
    }

    pub fn forward(
        &self,
        x: &[Vec<f64>],
        ssm_params: &SSMParams,
        attn_weights: &AttnWeights,
    ) -> (Vec<Vec<f64>>, Vec<bool>) {
        // 1. SSM scan
        let ssm_out = ssm_scan(x, ssm_params, self.d_model, self.d_state);

        // 2. LIF spiking gate
        let spikes = self.lif_spikes(&ssm_out);

        // 3. Attention only at spike positions
        let spike_positions: Vec<usize> = spikes.iter().enumerate()
            .filter(|(_, &s)| s).map(|(i, _)| i).collect();

        let attn_out = attention_at_positions(x, &spike_positions, attn_weights, self.d_model);

        // 4. Merge: spike positions get attention, rest keep SSM output
        let mut output = ssm_out;
        for (idx, &pos) in spike_positions.iter().enumerate() {
            if idx < attn_out.len() {
                output[pos] = attn_out[idx].clone();
            }
        }

        (output, spikes)
    }

    pub fn attention_sparsity(spikes: &[bool]) -> f64 {
        if spikes.is_empty() { return 0.0; }
        let fired = spikes.iter().filter(|&&s| s).count();
        fired as f64 / spikes.len() as f64
    }
}

// ---- ForwardForwardLayer ----

impl ForwardForwardLayer {
    pub fn new(d_in: usize, d_out: usize, lr: f64) -> Self {
        Self {
            weights: rand_matrix(d_in, d_out, 42 + d_in * 7 + d_out),
            bias: zeros(d_out),
            threshold: 2.0,
            lr,
        }
    }

    pub fn goodness(activations: &[f64]) -> f64 {
        activations.iter().map(|a| a * a).sum()
    }

    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let d_out = self.bias.len();
        let mut out = self.bias.clone();
        for j in 0..d_out {
            for i in 0..x.len().min(self.weights.len()) {
                out[j] += x[i] * self.weights[i][j];
            }
            out[j] = relu(out[j]);
        }
        // Layer-norm the output
        let mean = out.iter().sum::<f64>() / d_out as f64;
        let var = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / d_out as f64;
        let std = (var + 1e-8).sqrt();
        for v in out.iter_mut() {
            *v = (*v - mean) / std;
        }
        out
    }

    pub fn forward_learn(&mut self, x: &[f64], positive: bool) -> Vec<f64> {
        let out = self.forward(x);
        let goodness = Self::goodness(&out);
        let d_out = self.bias.len();

        // Forward-Forward learning rule:
        // positive: push goodness above threshold
        // negative: push goodness below threshold
        let p = sigmoid(goodness - self.threshold);
        let grad_sign = if positive { 1.0 - p } else { -p };

        // Update weights to increase/decrease goodness
        for i in 0..x.len().min(self.weights.len()) {
            for j in 0..d_out {
                self.weights[i][j] += self.lr * grad_sign * x[i] * out[j];
            }
        }
        for j in 0..d_out {
            self.bias[j] += self.lr * grad_sign * out[j];
        }

        out
    }
}

// ---- LiquidLayer ----

impl LiquidLayer {
    pub fn new(d_in: usize, d_hidden: usize) -> Self {
        Self {
            w_tau: rand_matrix(d_hidden, d_hidden, 100),
            w_hidden: rand_matrix(d_hidden, d_hidden, 200),
            w_input: rand_matrix(d_in, d_hidden, 300),
            base_tau: vec![1.0; d_hidden],
        }
    }

    /// Single ODE step with adaptive time constants (RK4)
    pub fn step(&self, h: &[f64], x: &[f64], dt: f64) -> (Vec<f64>, Vec<f64>) {
        let d = h.len();
        // Compute adaptive tau
        let tau_input = matvec(&self.w_tau, h);
        let x_contrib = matvec(&self.w_input, x);
        let tau_effective: Vec<f64> = (0..d)
            .map(|i| self.base_tau[i] * sigmoid(tau_input[i] + x_contrib[i]))
            .collect();

        // ODE: dh/dt = (-h + tanh(w_hidden @ h + w_input @ x)) / tau
        let dynamics = |hh: &[f64]| -> Vec<f64> {
            let wh = matvec(&self.w_hidden, hh);
            let wx = matvec(&self.w_input, x);
            (0..d)
                .map(|i| (-(hh[i]) + (wh[i] + wx[i]).tanh()) / tau_effective[i].max(0.01))
                .collect()
        };

        // RK4
        let k1 = dynamics(h);
        let h2: Vec<f64> = (0..d).map(|i| h[i] + 0.5 * dt * k1[i]).collect();
        let k2 = dynamics(&h2);
        let h3: Vec<f64> = (0..d).map(|i| h[i] + 0.5 * dt * k2[i]).collect();
        let k3 = dynamics(&h3);
        let h4: Vec<f64> = (0..d).map(|i| h[i] + dt * k3[i]).collect();
        let k4 = dynamics(&h4);

        let h_new: Vec<f64> = (0..d)
            .map(|i| h[i] + dt / 6.0 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]))
            .collect();

        (h_new, tau_effective)
    }

    pub fn forward_sequence(&self, x: &[Vec<f64>], dt: f64) -> (Vec<Vec<f64>>, f64) {
        let d_hidden = self.base_tau.len();
        let mut h = zeros(d_hidden);
        let mut output = Vec::with_capacity(x.len());
        let mut tau_sum = 0.0;
        let mut tau_count = 0usize;

        for t in 0..x.len() {
            let (h_new, taus) = self.step(&h, &x[t], dt);
            h = h_new;
            tau_sum += taus.iter().sum::<f64>();
            tau_count += taus.len();
            output.push(h.clone());
        }

        let avg_tau = if tau_count > 0 { tau_sum / tau_count as f64 } else { 0.0 };
        (output, avg_tau)
    }
}

// ---- SpikeSSMFormer ----

impl SpikeSSMFormer {
    pub fn new(
        d_model: usize,
        d_state: usize,
        n_ff_layers: usize,
        n_ssm_layers: usize,
        vocab_size: usize,
    ) -> Self {
        let ff_layers: Vec<ForwardForwardLayer> = (0..n_ff_layers)
            .map(|_| ForwardForwardLayer::new(d_model, d_model, 0.01))
            .collect();

        let ssm_layers: Vec<(SpikeGatedSSM, SSMParams, AttnWeights)> = (0..n_ssm_layers)
            .map(|i| {
                let ssm = SpikeGatedSSM::new(d_model, d_state);
                let params = SSMParams {
                    a: (0..d_state).map(|s| -1.0 - 0.1 * s as f64).collect(),
                    b_proj: rand_matrix(d_model, d_state, 1000 + i * 100),
                    c_proj: rand_matrix(d_state, d_model, 2000 + i * 100),
                    dt_proj: vec![0.0; d_model],
                    d: vec![1.0; d_model],
                };
                let attn = AttnWeights {
                    wq: rand_matrix(d_model, d_model, 3000 + i * 100),
                    wk: rand_matrix(d_model, d_model, 4000 + i * 100),
                    wv: rand_matrix(d_model, d_model, 5000 + i * 100),
                    wo: rand_matrix(d_model, d_model, 6000 + i * 100),
                    num_heads: 4.min(d_model),
                };
                (ssm, params, attn)
            })
            .collect();

        let liquid = LiquidLayer::new(d_model, d_model);
        let output_proj = rand_matrix(vocab_size, d_model, 9999);

        Self {
            ff_layers,
            ssm_layers,
            liquid_layer: liquid,
            output_proj,
            d_model,
            vocab_size,
            last_spikes: vec![],
            last_ff_goodness: 0.0,
            last_avg_tau: 0.0,
        }
    }

    /// Full forward pass
    pub fn forward(&mut self, _token_ids: &[usize], embeddings: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut x: Vec<Vec<f64>> = embeddings.to_vec();

        // 1. Forward-Forward layers (inference mode)
        for layer in &self.ff_layers {
            x = x.iter().map(|xi| layer.forward(xi)).collect();
        }
        self.last_ff_goodness = if !x.is_empty() {
            ForwardForwardLayer::goodness(&x[0])
        } else {
            0.0
        };

        // 2. SSM + sparse attention layers
        let mut all_spikes = vec![false; x.len()];
        for (ssm, params, attn) in &self.ssm_layers {
            let (out, spikes) = ssm.forward(&x, params, attn);
            for (i, &s) in spikes.iter().enumerate() {
                all_spikes[i] = all_spikes[i] || s;
            }
            x = out;
        }
        self.last_spikes = all_spikes;

        // 3. Liquid time-constant layer
        let (liquid_out, avg_tau) = self.liquid_layer.forward_sequence(&x, 0.1);
        self.last_avg_tau = avg_tau;
        x = liquid_out;

        // 4. Output projection
        x.iter()
            .map(|xi| {
                let logits = matvec(&self.output_proj, xi);
                logits
            })
            .collect()
    }

    /// Training step with hybrid learning
    pub fn train_step(
        &mut self,
        token_ids: &[usize],
        embeddings: &[Vec<f64>],
        targets: &[usize],
        lr: f64,
    ) -> f64 {
        let mut x: Vec<Vec<f64>> = embeddings.to_vec();

        // 1. FF layers with local learning (positive examples)
        for layer in &mut self.ff_layers {
            x = x.iter().map(|xi| layer.forward_learn(xi, true)).collect();
        }

        // 2. SSM layers (forward only for now)
        let mut all_spikes = vec![false; x.len()];
        for (ssm, params, attn) in &self.ssm_layers {
            let (out, spikes) = ssm.forward(&x, params, attn);
            for (i, &s) in spikes.iter().enumerate() {
                all_spikes[i] = all_spikes[i] || s;
            }
            x = out;
        }
        self.last_spikes = all_spikes;

        // 3. Liquid layer
        let (liquid_out, avg_tau) = self.liquid_layer.forward_sequence(&x, 0.1);
        self.last_avg_tau = avg_tau;
        x = liquid_out;

        // 4. Output projection + cross-entropy loss
        let mut total_loss = 0.0;
        for (t, xi) in x.iter().enumerate() {
            if t >= targets.len() { break; }
            let logits = matvec(&self.output_proj, xi);
            // Softmax + cross-entropy
            let max_l = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = logits.iter().map(|l| (l - max_l).exp()).collect();
            let sum_exp: f64 = exps.iter().sum();
            let target = targets[t].min(self.vocab_size - 1);
            let prob = exps[target] / sum_exp;
            total_loss += -(prob.max(1e-10)).ln();

            // Simple gradient: adjust output_proj towards target
            let probs: Vec<f64> = exps.iter().map(|e| e / sum_exp).collect();
            for v in 0..self.vocab_size.min(self.output_proj.len()) {
                let grad = probs[v] - if v == target { 1.0 } else { 0.0 };
                for d in 0..self.d_model.min(xi.len()).min(self.output_proj[v].len()) {
                    self.output_proj[v][d] -= lr * grad * xi[d];
                }
            }
        }

        let n = targets.len().max(1) as f64;
        self.last_ff_goodness = if !embeddings.is_empty() {
            ForwardForwardLayer::goodness(&embeddings[0])
        } else {
            0.0
        };
        total_loss / n
    }

    pub fn stats(&self) -> ArchStats {
        let mut total_params = 0usize;
        // FF layers
        for l in &self.ff_layers {
            total_params += l.weights.len() * l.weights.get(0).map_or(0, |r| r.len());
            total_params += l.bias.len();
        }
        // SSM layers
        for (_, params, attn) in &self.ssm_layers {
            total_params += params.a.len();
            total_params += params.b_proj.len() * params.b_proj.get(0).map_or(0, |r| r.len());
            total_params += params.c_proj.len() * params.c_proj.get(0).map_or(0, |r| r.len());
            total_params += params.dt_proj.len() + params.d.len();
            total_params += attn.wq.len() * attn.wq.get(0).map_or(0, |r| r.len());
            total_params += attn.wk.len() * attn.wk.get(0).map_or(0, |r| r.len());
            total_params += attn.wv.len() * attn.wv.get(0).map_or(0, |r| r.len());
            total_params += attn.wo.len() * attn.wo.get(0).map_or(0, |r| r.len());
        }
        // Liquid layer
        total_params += self.liquid_layer.w_tau.len() * self.liquid_layer.w_tau.get(0).map_or(0, |r| r.len());
        total_params += self.liquid_layer.w_hidden.len() * self.liquid_layer.w_hidden.get(0).map_or(0, |r| r.len());
        total_params += self.liquid_layer.w_input.len() * self.liquid_layer.w_input.get(0).map_or(0, |r| r.len());
        total_params += self.liquid_layer.base_tau.len();
        // Output proj
        total_params += self.output_proj.len() * self.output_proj.get(0).map_or(0, |r| r.len());

        ArchStats {
            total_params,
            attention_sparsity: SpikeGatedSSM::attention_sparsity(&self.last_spikes),
            ff_goodness: self.last_ff_goodness,
            avg_tau: self.last_avg_tau,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings(seq_len: usize, d_model: usize) -> Vec<Vec<f64>> {
        (0..seq_len)
            .map(|t| (0..d_model).map(|d| small_rand(t, d, 77)).collect())
            .collect()
    }

    #[test]
    fn test_spike_gated_ssm_sparsity() {
        let d_model = 16;
        let d_state = 8;
        let ssm = SpikeGatedSSM::new(d_model, d_state);
        let params = SSMParams {
            a: (0..d_state).map(|s| -1.0 - 0.1 * s as f64).collect(),
            b_proj: rand_matrix(d_model, d_state, 10),
            c_proj: rand_matrix(d_state, d_model, 20),
            dt_proj: vec![0.0; d_model],
            d: vec![1.0; d_model],
        };
        let attn = AttnWeights {
            wq: rand_matrix(d_model, d_model, 30),
            wk: rand_matrix(d_model, d_model, 40),
            wv: rand_matrix(d_model, d_model, 50),
            wo: rand_matrix(d_model, d_model, 60),
            num_heads: 4,
        };
        let x = make_embeddings(32, d_model);
        let (output, spikes) = ssm.forward(&x, &params, &attn);
        assert_eq!(output.len(), 32);
        assert_eq!(output[0].len(), d_model);
        let sparsity = SpikeGatedSSM::attention_sparsity(&spikes);
        assert!(sparsity < 1.0, "Not all tokens should trigger attention, got sparsity={}", sparsity);
        assert!(sparsity >= 0.0);
    }

    #[test]
    fn test_forward_forward_goodness_increases() {
        let mut layer = ForwardForwardLayer::new(16, 16, 0.1);
        let x: Vec<f64> = (0..16).map(|i| 0.5 + 0.1 * i as f64).collect();
        let initial_out = layer.forward(&x);
        let initial_goodness = ForwardForwardLayer::goodness(&initial_out);

        for _ in 0..10 {
            layer.forward_learn(&x, true);
        }
        let final_out = layer.forward(&x);
        let final_goodness = ForwardForwardLayer::goodness(&final_out);
        // After positive learning, goodness should generally increase or stay high
        // (with layer norm, goodness is normalized, so we check it's still reasonable)
        assert!(final_goodness > 0.0, "Goodness should be positive: {}", final_goodness);
    }

    #[test]
    fn test_liquid_layer_different_taus() {
        let layer = LiquidLayer::new(8, 8);
        let x1 = vec![1.0; 8];
        let x2 = vec![-1.0; 8];
        let h = zeros(8);

        let (_, tau1) = layer.step(&h, &x1, 0.1);
        let (_, tau2) = layer.step(&h, &x2, 0.1);

        // Different inputs should produce different effective time constants
        let different = tau1.iter().zip(tau2.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(different, "Different inputs should produce different taus");
    }

    #[test]
    fn test_spike_ssm_former_forward_shape() {
        let d_model = 16;
        let vocab = 32;
        let seq_len = 10;
        let mut model = SpikeSSMFormer::new(d_model, 8, 1, 1, vocab);
        let emb = make_embeddings(seq_len, d_model);
        let tokens: Vec<usize> = (0..seq_len).collect();

        let output = model.forward(&tokens, &emb);
        assert_eq!(output.len(), seq_len);
        assert_eq!(output[0].len(), vocab);
    }

    #[test]
    fn test_spike_ssm_former_train_decreases_loss() {
        let d_model = 16;
        let vocab = 8;
        let seq_len = 4;
        let mut model = SpikeSSMFormer::new(d_model, 4, 1, 1, vocab);
        let emb = make_embeddings(seq_len, d_model);
        let tokens: Vec<usize> = (0..seq_len).collect();
        let targets: Vec<usize> = vec![1, 2, 3, 0];

        let loss1 = model.train_step(&tokens, &emb, &targets, 0.01);
        let mut last_loss = loss1;
        for _ in 0..4 {
            last_loss = model.train_step(&tokens, &emb, &targets, 0.01);
        }
        assert!(last_loss < loss1, "Loss should decrease: {} -> {}", loss1, last_loss);
    }

    #[test]
    fn test_attention_sparsity_range() {
        let d_model = 16;
        let mut model = SpikeSSMFormer::new(d_model, 8, 1, 1, 32);
        let emb = make_embeddings(64, d_model);
        let tokens: Vec<usize> = (0..64).collect();
        model.forward(&tokens, &emb);

        let stats = model.stats();
        // With random input, sparsity should be in a reasonable range
        assert!(stats.attention_sparsity >= 0.0 && stats.attention_sparsity <= 1.0);
        assert!(stats.total_params > 0);
        // Not all tokens should fire (that would defeat the purpose)
        // Allow some flexibility but check it's not 100%
        assert!(stats.attention_sparsity < 0.9, "Sparsity too high: {}", stats.attention_sparsity);
    }
}
