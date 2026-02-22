/// Probabilistic types: distributions, uncertainty propagation, calibration checking,
/// Bayesian layers, and probabilistic models.

use std::collections::HashMap;
use crate::interpreter::{Env, FnDef, Value};

// ── Deterministic PRNG ──────────────────────────────────────────────

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
    fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn normal(&mut self) -> f64 {
        let u1 = self.uniform().max(1e-15);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// We use a global seed counter so successive calls produce different sequences.
static SEED_CTR: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(42);

fn next_rng() -> Rng {
    let s = SEED_CTR.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    Rng::new(s.wrapping_mul(6364136223846793005).wrapping_add(1))
}

// ── Distribution ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Distribution {
    Normal(f64, f64),           // mean, std
    Uniform(f64, f64),          // low, high
    Bernoulli(f64),             // p
    Categorical(Vec<f64>),      // probs (sum to 1)
    Beta(f64, f64),             // alpha, beta
    Mixture(Vec<(f64, Distribution)>), // weighted components
    Empirical(Vec<f64>),        // samples
}

impl Distribution {
    pub fn sample(&self) -> f64 {
        let mut rng = next_rng();
        self.sample_with(&mut rng)
    }

    fn sample_with(&self, rng: &mut Rng) -> f64 {
        match self {
            Distribution::Normal(mean, std) => mean + std * rng.normal(),
            Distribution::Uniform(lo, hi) => lo + (hi - lo) * rng.uniform(),
            Distribution::Bernoulli(p) => if rng.uniform() < *p { 1.0 } else { 0.0 },
            Distribution::Categorical(probs) => {
                let u = rng.uniform();
                let mut cum = 0.0;
                for (i, p) in probs.iter().enumerate() {
                    cum += p;
                    if u < cum { return i as f64; }
                }
                (probs.len() - 1) as f64
            }
            Distribution::Beta(alpha, beta) => {
                // Use Joehnk's method for small params, otherwise gamma ratio
                let x = gamma_sample(rng, *alpha);
                let y = gamma_sample(rng, *beta);
                if x + y == 0.0 { 0.5 } else { x / (x + y) }
            }
            Distribution::Mixture(components) => {
                let u = rng.uniform();
                let mut cum = 0.0;
                for (w, dist) in components {
                    cum += w;
                    if u < cum {
                        return dist.sample_with(rng);
                    }
                }
                components.last().map(|(_, d)| d.sample_with(rng)).unwrap_or(0.0)
            }
            Distribution::Empirical(samples) => {
                if samples.is_empty() { return 0.0; }
                let idx = (rng.uniform() * samples.len() as f64) as usize;
                samples[idx.min(samples.len() - 1)]
            }
        }
    }

    pub fn log_prob(&self, x: f64) -> f64 {
        match self {
            Distribution::Normal(mean, std) => {
                let z = (x - mean) / std;
                -0.5 * z * z - std.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
            }
            Distribution::Uniform(lo, hi) => {
                if x >= *lo && x <= *hi { -(*hi - *lo).ln() } else { f64::NEG_INFINITY }
            }
            Distribution::Bernoulli(p) => {
                if (x - 1.0).abs() < 1e-9 { p.ln() }
                else if x.abs() < 1e-9 { (1.0 - p).ln() }
                else { f64::NEG_INFINITY }
            }
            Distribution::Categorical(probs) => {
                let i = x.round() as usize;
                if i < probs.len() && probs[i] > 0.0 { probs[i].ln() } else { f64::NEG_INFINITY }
            }
            Distribution::Beta(alpha, beta) => {
                if x <= 0.0 || x >= 1.0 { return f64::NEG_INFINITY; }
                (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - ln_beta(*alpha, *beta)
            }
            Distribution::Mixture(components) => {
                // log-sum-exp
                let log_probs: Vec<f64> = components.iter()
                    .map(|(w, d)| w.ln() + d.log_prob(x))
                    .collect();
                log_sum_exp(&log_probs)
            }
            Distribution::Empirical(samples) => {
                // KDE with bandwidth = 1.06 * std * n^(-1/5)
                if samples.is_empty() { return f64::NEG_INFINITY; }
                let m = self.mean();
                let v = self.variance().sqrt().max(1e-10);
                let n = samples.len() as f64;
                let h = 1.06 * v * n.powf(-0.2);
                let log_probs: Vec<f64> = samples.iter()
                    .map(|s| {
                        let z = (x - s) / h;
                        -0.5 * z * z - h.ln() - 0.5 * (2.0 * std::f64::consts::PI).ln()
                    })
                    .collect();
                log_sum_exp(&log_probs) - n.ln()
            }
        }
    }

    pub fn mean(&self) -> f64 {
        match self {
            Distribution::Normal(m, _) => *m,
            Distribution::Uniform(lo, hi) => (lo + hi) / 2.0,
            Distribution::Bernoulli(p) => *p,
            Distribution::Categorical(probs) => {
                probs.iter().enumerate().map(|(i, p)| i as f64 * p).sum()
            }
            Distribution::Beta(a, b) => a / (a + b),
            Distribution::Mixture(components) => {
                components.iter().map(|(w, d)| w * d.mean()).sum()
            }
            Distribution::Empirical(samples) => {
                if samples.is_empty() { 0.0 }
                else { samples.iter().sum::<f64>() / samples.len() as f64 }
            }
        }
    }

    pub fn variance(&self) -> f64 {
        match self {
            Distribution::Normal(_, s) => s * s,
            Distribution::Uniform(lo, hi) => (hi - lo).powi(2) / 12.0,
            Distribution::Bernoulli(p) => p * (1.0 - p),
            Distribution::Categorical(probs) => {
                let m = self.mean();
                probs.iter().enumerate().map(|(i, p)| p * (i as f64 - m).powi(2)).sum()
            }
            Distribution::Beta(a, b) => (a * b) / ((a + b).powi(2) * (a + b + 1.0)),
            Distribution::Mixture(components) => {
                // law of total variance
                let m = self.mean();
                let mut var = 0.0;
                for (w, d) in components {
                    var += w * (d.variance() + (d.mean() - m).powi(2));
                }
                var
            }
            Distribution::Empirical(samples) => {
                if samples.len() < 2 { return 0.0; }
                let m = self.mean();
                samples.iter().map(|x| (x - m).powi(2)).sum::<f64>() / (samples.len() - 1) as f64
            }
        }
    }

    pub fn entropy(&self) -> f64 {
        match self {
            Distribution::Normal(_, s) => 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E * s * s).ln(),
            Distribution::Uniform(lo, hi) => (hi - lo).ln(),
            Distribution::Bernoulli(p) => {
                if *p <= 0.0 || *p >= 1.0 { 0.0 }
                else { -(p * p.ln() + (1.0 - p) * (1.0 - p).ln()) }
            }
            Distribution::Categorical(probs) => {
                -probs.iter().filter(|p| **p > 0.0).map(|p| p * p.ln()).sum::<f64>()
            }
            Distribution::Beta(a, b) => {
                ln_beta(*a, *b) - (a - 1.0) * digamma(*a) - (b - 1.0) * digamma(*b) + (a + b - 2.0) * digamma(a + b)
            }
            Distribution::Mixture(_) | Distribution::Empirical(_) => {
                // Monte Carlo estimate
                let mut rng = next_rng();
                let n = 1000;
                let mut ent = 0.0;
                for _ in 0..n {
                    let x = self.sample_with(&mut rng);
                    ent -= self.log_prob(x);
                }
                ent / n as f64
            }
        }
    }
}

// ── Math helpers ────────────────────────────────────────────────────

fn log_sum_exp(xs: &[f64]) -> f64 {
    if xs.is_empty() { return f64::NEG_INFINITY; }
    let max = xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max.is_infinite() { return f64::NEG_INFINITY; }
    max + xs.iter().map(|x| (x - max).exp()).sum::<f64>().ln()
}

fn ln_beta(a: f64, b: f64) -> f64 {
    ln_gamma(a) + ln_gamma(b) - ln_gamma(a + b)
}

/// Stirling approximation of ln(Gamma(x))
fn ln_gamma(x: f64) -> f64 {
    if x <= 0.0 { return 0.0; }
    // Lanczos approximation
    let g = 7.0_f64;
    let c = [
        0.99999999999980993,
        676.5203681218851,
        -1259.1392167224028,
        771.32342877765313,
        -176.61502916214059,
        12.507343278686905,
        -0.13857109526572012,
        9.9843695780195716e-6,
        1.5056327351493116e-7,
    ];
    let x = x - 1.0;
    let mut sum = c[0];
    for i in 1..9 {
        sum += c[i] / (x + i as f64);
    }
    let t = x + g + 0.5;
    0.5 * (2.0 * std::f64::consts::PI).ln() + (t).ln() * (x + 0.5) - t + sum.ln()
}

fn digamma(x: f64) -> f64 {
    // Approximation via recurrence + asymptotic
    let mut x = x;
    let mut result = 0.0;
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }
    result += x.ln() - 0.5 / x - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x.powi(4));
    result
}

fn gamma_sample(rng: &mut Rng, alpha: f64) -> f64 {
    // Marsaglia and Tsang's method
    if alpha < 1.0 {
        return gamma_sample(rng, alpha + 1.0) * rng.uniform().powf(1.0 / alpha);
    }
    let d = alpha - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = rng.normal();
        let v = (1.0 + c * x).powi(3);
        if v > 0.0 {
            let u = rng.uniform().max(1e-15);
            if u.ln() < 0.5 * x * x + d - d * v + d * v.ln() {
                return d * v;
            }
        }
    }
}

// ── ProbValue ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProbValue {
    pub value: f64,
    pub distribution: Distribution,
}

impl ProbValue {
    pub fn new(value: f64, distribution: Distribution) -> Self {
        Self { value, distribution }
    }

    pub fn confidence_interval(&self, alpha: f64) -> (f64, f64) {
        // Monte Carlo CI
        let mut rng = next_rng();
        let n = 10000;
        let mut samples: Vec<f64> = (0..n).map(|_| self.distribution.sample_with(&mut rng)).collect();
        samples.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let lo_idx = ((alpha / 2.0) * n as f64) as usize;
        let hi_idx = ((1.0 - alpha / 2.0) * n as f64) as usize;
        (samples[lo_idx.min(n - 1)], samples[hi_idx.min(n - 1)])
    }
}

// ── UncertaintyPropagation ──────────────────────────────────────────

pub struct UncertaintyPropagation;

impl UncertaintyPropagation {
    pub fn add(a: &ProbValue, b: &ProbValue) -> ProbValue {
        // For independent normals, sum is normal
        match (&a.distribution, &b.distribution) {
            (Distribution::Normal(m1, s1), Distribution::Normal(m2, s2)) => {
                ProbValue::new(
                    a.value + b.value,
                    Distribution::Normal(m1 + m2, (s1 * s1 + s2 * s2).sqrt()),
                )
            }
            _ => Self::apply_function(|args| args[0] + args[1], &[a.clone(), b.clone()], 5000),
        }
    }

    pub fn mul(a: &ProbValue, b: &ProbValue) -> ProbValue {
        // For independent normals with small CV, product is approximately normal
        match (&a.distribution, &b.distribution) {
            (Distribution::Normal(m1, s1), Distribution::Normal(m2, s2)) => {
                let new_mean = m1 * m2;
                let new_var = m1 * m1 * s2 * s2 + m2 * m2 * s1 * s1 + s1 * s1 * s2 * s2;
                ProbValue::new(
                    a.value * b.value,
                    Distribution::Normal(new_mean, new_var.sqrt()),
                )
            }
            _ => Self::apply_function(|args| args[0] * args[1], &[a.clone(), b.clone()], 5000),
        }
    }

    pub fn apply_function(f: fn(&[f64]) -> f64, inputs: &[ProbValue], n_samples: usize) -> ProbValue {
        let mut rng = next_rng();
        let mut samples = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let args: Vec<f64> = inputs.iter().map(|pv| pv.distribution.sample_with(&mut rng)).collect();
            samples.push(f(&args));
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        ProbValue::new(mean, Distribution::Empirical(samples))
    }

    pub fn compose(ops: &[ProbValue]) -> ProbValue {
        if ops.is_empty() {
            return ProbValue::new(0.0, Distribution::Normal(0.0, 0.0));
        }
        let mut result = ops[0].clone();
        for op in &ops[1..] {
            result = Self::add(&result, op);
        }
        result
    }
}

// ── CalibrationReport ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CalibrationReport {
    pub ece: f64,
    pub mce: f64,
    pub brier_score: f64,
    pub reliability_diagram: Vec<(f64, f64)>, // (predicted_prob, observed_freq)
}

pub struct CalibrationChecker;

impl CalibrationChecker {
    pub fn check_calibration(predictions: &[ProbValue], outcomes: &[f64]) -> CalibrationReport {
        let n_bins = 10;
        let mut bin_sums = vec![0.0; n_bins];
        let mut bin_counts = vec![0usize; n_bins];
        let mut bin_outcomes = vec![0.0; n_bins];

        let mut brier = 0.0;

        for (pred, outcome) in predictions.iter().zip(outcomes.iter()) {
            // Use the mean as the predicted probability (clamped to [0,1])
            let p = pred.value.clamp(0.0, 1.0);
            let bin = ((p * n_bins as f64) as usize).min(n_bins - 1);
            bin_sums[bin] += p;
            bin_counts[bin] += 1;
            bin_outcomes[bin] += outcome;
            brier += (p - outcome).powi(2);
        }

        let n = predictions.len().max(1) as f64;
        brier /= n;

        let mut ece = 0.0;
        let mut mce = 0.0_f64;
        let mut reliability = Vec::new();

        for i in 0..n_bins {
            if bin_counts[i] > 0 {
                let avg_pred = bin_sums[i] / bin_counts[i] as f64;
                let avg_outcome = bin_outcomes[i] / bin_counts[i] as f64;
                let diff = (avg_pred - avg_outcome).abs();
                ece += diff * bin_counts[i] as f64 / n;
                mce = mce.max(diff);
                reliability.push((avg_pred, avg_outcome));
            }
        }

        CalibrationReport {
            ece,
            mce,
            brier_score: brier,
            reliability_diagram: reliability,
        }
    }

    pub fn is_calibrated(report: &CalibrationReport, threshold: f64) -> bool {
        report.ece < threshold
    }

    pub fn recalibrate(predictions: &[ProbValue], outcomes: &[f64]) -> TemperatureScaling {
        // Simple grid search for optimal temperature
        let logits: Vec<Vec<f64>> = predictions.iter().map(|p| vec![p.value]).collect();
        let labels: Vec<usize> = outcomes.iter().map(|o| if *o > 0.5 { 1 } else { 0 }).collect();
        let t = TemperatureScaling::optimize(&logits, &labels);
        TemperatureScaling { temperature: t }
    }
}

// ── TemperatureScaling ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TemperatureScaling {
    pub temperature: f64,
}

impl TemperatureScaling {
    pub fn apply(logits: &[f64], temperature: f64) -> Vec<f64> {
        let scaled: Vec<f64> = logits.iter().map(|l| l / temperature).collect();
        let max = scaled.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scaled.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        exps.iter().map(|e| e / sum).collect()
    }

    pub fn optimize(predictions: &[Vec<f64>], labels: &[usize]) -> f64 {
        // Grid search over temperature values to minimize NLL
        let mut best_t = 1.0;
        let mut best_nll = f64::INFINITY;

        for t_int in 1..100 {
            let t = t_int as f64 * 0.1;
            let mut nll = 0.0;
            for (logits, &label) in predictions.iter().zip(labels.iter()) {
                let probs = Self::apply(logits, t);
                let idx = label.min(probs.len() - 1);
                nll -= probs[idx].max(1e-15).ln();
            }
            if nll < best_nll {
                best_nll = nll;
                best_t = t;
            }
        }
        best_t
    }
}

// ── BayesianLayer ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BayesianLayer {
    pub weight_means: Vec<Vec<f64>>,
    pub weight_stds: Vec<Vec<f64>>,
    pub bias_means: Vec<f64>,
    pub bias_stds: Vec<f64>,
    pub in_features: usize,
    pub out_features: usize,
}

impl BayesianLayer {
    pub fn new(in_features: usize, out_features: usize) -> Self {
        let scale = (1.0 / in_features as f64).sqrt();
        let mut rng = next_rng();
        let weight_means: Vec<Vec<f64>> = (0..out_features)
            .map(|_| (0..in_features).map(|_| rng.normal() * scale).collect())
            .collect();
        let weight_stds = vec![vec![0.1; in_features]; out_features];
        let bias_means = vec![0.0; out_features];
        let bias_stds = vec![0.1; out_features];
        Self { weight_means, weight_stds, bias_means, bias_stds, in_features, out_features }
    }

    pub fn forward_sample(&self, input: &[f64]) -> ProbValue {
        let mut rng = next_rng();
        let mut outputs = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let mut sum = self.bias_means[o] + self.bias_stds[o] * rng.normal();
            for i in 0..self.in_features.min(input.len()) {
                let w = self.weight_means[o][i] + self.weight_stds[o][i] * rng.normal();
                sum += w * input[i];
            }
            outputs.push(sum);
        }
        let mean = outputs.iter().sum::<f64>() / outputs.len().max(1) as f64;
        ProbValue::new(mean, Distribution::Empirical(outputs))
    }

    pub fn forward_mean(&self, input: &[f64]) -> Vec<f64> {
        let mut outputs = Vec::with_capacity(self.out_features);
        for o in 0..self.out_features {
            let mut sum = self.bias_means[o];
            for i in 0..self.in_features.min(input.len()) {
                sum += self.weight_means[o][i] * input[i];
            }
            outputs.push(sum);
        }
        outputs
    }

    pub fn kl_divergence(&self) -> f64 {
        // KL(N(mu, sigma^2) || N(0, 1)) for each weight
        let mut kl = 0.0;
        for o in 0..self.out_features {
            for i in 0..self.in_features {
                let mu = self.weight_means[o][i];
                let sigma = self.weight_stds[o][i].max(1e-10);
                kl += 0.5 * (mu * mu + sigma * sigma - 1.0 - (sigma * sigma).ln());
            }
            let mu = self.bias_means[o];
            let sigma = self.bias_stds[o].max(1e-10);
            kl += 0.5 * (mu * mu + sigma * sigma - 1.0 - (sigma * sigma).ln());
        }
        kl
    }

    pub fn elbo_loss(&self, output: &ProbValue, target: f64, n_data: usize) -> f64 {
        // ELBO = -log_likelihood + (1/N) * KL
        let nll = -output.distribution.log_prob(target);
        let kl = self.kl_divergence();
        nll + kl / n_data.max(1) as f64
    }
}

// ── ProbabilisticModel ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ProbabilisticModel {
    pub layers: Vec<BayesianLayer>,
}

impl ProbabilisticModel {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for w in layer_sizes.windows(2) {
            layers.push(BayesianLayer::new(w[0], w[1]));
        }
        Self { layers }
    }

    pub fn forward(&self, input: &[f64], n_samples: usize) -> ProbValue {
        let mut all_outputs = Vec::with_capacity(n_samples);
        for _ in 0..n_samples {
            let mut x = input.to_vec();
            for layer in &self.layers {
                let pv = layer.forward_sample(&x);
                // Use the sampled outputs as next input
                match &pv.distribution {
                    Distribution::Empirical(s) => x = s.clone(),
                    _ => x = vec![pv.value],
                }
            }
            // Take mean of final layer output as this sample's prediction
            let out = x.iter().sum::<f64>() / x.len().max(1) as f64;
            all_outputs.push(out);
        }
        let mean = all_outputs.iter().sum::<f64>() / all_outputs.len().max(1) as f64;
        ProbValue::new(mean, Distribution::Empirical(all_outputs))
    }

    pub fn epistemic_uncertainty(&self, input: &[f64], n_samples: usize) -> f64 {
        // Variance of predictions across samples = model uncertainty
        let pv = self.forward(input, n_samples);
        pv.distribution.variance()
    }

    pub fn aleatoric_uncertainty(&self, input: &[f64]) -> f64 {
        // Use mean weights to get a single forward pass, then measure output spread
        let mut x = input.to_vec();
        for layer in &self.layers {
            x = layer.forward_mean(&x);
        }
        // Aleatoric uncertainty from last layer's weight stds
        if let Some(last) = self.layers.last() {
            let mut var = 0.0;
            for o in 0..last.out_features {
                for i in 0..last.in_features.min(input.len()) {
                    var += last.weight_stds[o][i].powi(2) * input.get(i).unwrap_or(&0.0).powi(2);
                }
                var += last.bias_stds[o].powi(2);
            }
            var
        } else {
            0.0
        }
    }

    pub fn predictive_entropy(&self, input: &[f64], n_samples: usize) -> f64 {
        let pv = self.forward(input, n_samples);
        pv.distribution.entropy()
    }
}

// ── Interpreter builtins ────────────────────────────────────────────

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
        Value::Float(f) => Ok(*f as usize),
        _ => Err(format!("expected integer, got {:?}", v)),
    }
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("prob_normal".to_string(), FnDef::Builtin(builtin_prob_normal));
    env.functions.insert("prob_sample".to_string(), FnDef::Builtin(builtin_prob_sample));
    env.functions.insert("prob_mean".to_string(), FnDef::Builtin(builtin_prob_mean));
    env.functions.insert("prob_confidence_interval".to_string(), FnDef::Builtin(builtin_prob_confidence_interval));
    env.functions.insert("bayesian_layer_new".to_string(), FnDef::Builtin(builtin_bayesian_layer_new));
    env.functions.insert("bayesian_forward".to_string(), FnDef::Builtin(builtin_bayesian_forward));
    env.functions.insert("calibration_check".to_string(), FnDef::Builtin(builtin_calibration_check));
}

fn builtin_prob_normal(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("prob_normal expects 2 args: (mean, std)".into()); }
    let mean = value_to_f64(&args[0])?;
    let std = value_to_f64(&args[1])?;
    let pv = ProbValue::new(mean, Distribution::Normal(mean, std));
    let id = env.next_prob_id;
    env.next_prob_id += 1;
    env.prob_values.insert(id, pv);
    Ok(Value::Int(id as i128))
}

fn builtin_prob_sample(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("prob_sample expects 1 arg: (id)".into()); }
    let id = value_to_usize(&args[0])?;
    let pv = env.prob_values.get(&id).ok_or("invalid prob id")?;
    Ok(Value::Float(pv.distribution.sample()))
}

fn builtin_prob_mean(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("prob_mean expects 1 arg: (id)".into()); }
    let id = value_to_usize(&args[0])?;
    let pv = env.prob_values.get(&id).ok_or("invalid prob id")?;
    Ok(Value::Float(pv.distribution.mean()))
}

fn builtin_prob_confidence_interval(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("prob_confidence_interval expects 2 args: (id, alpha)".into()); }
    let id = value_to_usize(&args[0])?;
    let alpha = value_to_f64(&args[1])?;
    let pv = env.prob_values.get(&id).ok_or("invalid prob id")?;
    let (lo, hi) = pv.confidence_interval(alpha);
    Ok(Value::Array(vec![Value::Float(lo), Value::Float(hi)]))
}

fn builtin_bayesian_layer_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("bayesian_layer_new expects 2 args: (in, out)".into()); }
    let in_f = value_to_usize(&args[0])?;
    let out_f = value_to_usize(&args[1])?;
    let layer = BayesianLayer::new(in_f, out_f);
    let id = env.next_prob_layer_id;
    env.next_prob_layer_id += 1;
    env.prob_layers.insert(id, layer);
    Ok(Value::Int(id as i128))
}

fn builtin_bayesian_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("bayesian_forward expects 3 args: (id, input, n_samples)".into()); }
    let id = value_to_usize(&args[0])?;
    let input: Vec<f64> = match &args[1] {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("input must be an array".into()),
    };
    let n_samples = value_to_usize(&args[2])?;

    let layer = env.prob_layers.get(&id).ok_or("invalid layer id")?;
    let mut all_outputs = Vec::new();
    for _ in 0..n_samples {
        let pv = layer.forward_sample(&input);
        all_outputs.push(pv.value);
    }
    let mean = all_outputs.iter().sum::<f64>() / all_outputs.len().max(1) as f64;
    let var = all_outputs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / all_outputs.len().max(1) as f64;
    Ok(Value::Array(vec![Value::Float(mean), Value::Float(var.sqrt())]))
}

fn builtin_calibration_check(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("calibration_check expects 2 args: (predictions, outcomes)".into()); }
    let preds: Vec<f64> = match &args[0] {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("predictions must be an array".into()),
    };
    let outcomes: Vec<f64> = match &args[1] {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("outcomes must be an array".into()),
    };

    let prob_preds: Vec<ProbValue> = preds.iter()
        .map(|&p| ProbValue::new(p, Distribution::Bernoulli(p.clamp(0.0, 1.0))))
        .collect();

    let report = CalibrationChecker::check_calibration(&prob_preds, &outcomes);
    Ok(Value::Array(vec![
        Value::Float(report.ece),
        Value::Float(report.mce),
        Value::Float(report.brier_score),
    ]))
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normal_mean_variance() {
        let d = Distribution::Normal(5.0, 2.0);
        assert!((d.mean() - 5.0).abs() < 1e-10);
        assert!((d.variance() - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_log_prob() {
        let d = Distribution::Normal(0.0, 1.0);
        let lp = d.log_prob(0.0);
        let expected = -0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!((lp - expected).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_distribution() {
        let d = Distribution::Uniform(0.0, 1.0);
        assert!((d.mean() - 0.5).abs() < 1e-10);
        assert!((d.variance() - 1.0 / 12.0).abs() < 1e-10);
        assert!((d.log_prob(0.5) - 0.0).abs() < 1e-10); // ln(1) = 0
        assert!(d.log_prob(1.5).is_infinite());
    }

    #[test]
    fn test_bernoulli_distribution() {
        let d = Distribution::Bernoulli(0.7);
        assert!((d.mean() - 0.7).abs() < 1e-10);
        assert!((d.variance() - 0.21).abs() < 1e-10);
        assert!((d.log_prob(1.0) - 0.7_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_beta_mean() {
        let d = Distribution::Beta(2.0, 5.0);
        assert!((d.mean() - 2.0 / 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_normal_sampling_mean() {
        let d = Distribution::Normal(10.0, 1.0);
        let mut rng = Rng::new(123);
        let samples: Vec<f64> = (0..10000).map(|_| d.sample_with(&mut rng)).collect();
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!((mean - 10.0).abs() < 0.1, "mean was {}", mean);
    }

    #[test]
    fn test_uncertainty_propagation_add() {
        let a = ProbValue::new(3.0, Distribution::Normal(3.0, 1.0));
        let b = ProbValue::new(4.0, Distribution::Normal(4.0, 1.0));
        let result = UncertaintyPropagation::add(&a, &b);
        assert!((result.value - 7.0).abs() < 1e-10);
        // std should be sqrt(2)
        if let Distribution::Normal(m, s) = &result.distribution {
            assert!((m - 7.0).abs() < 1e-10);
            assert!((s - 2.0_f64.sqrt()).abs() < 1e-10);
        } else {
            panic!("expected Normal distribution");
        }
    }

    #[test]
    fn test_uncertainty_propagation_mul() {
        let a = ProbValue::new(3.0, Distribution::Normal(3.0, 0.1));
        let b = ProbValue::new(4.0, Distribution::Normal(4.0, 0.1));
        let result = UncertaintyPropagation::mul(&a, &b);
        assert!((result.value - 12.0).abs() < 1e-10);
    }

    #[test]
    fn test_calibration_perfect() {
        // Perfect calibration: predictions match outcomes
        let preds: Vec<ProbValue> = (0..100).map(|i| {
            let p = (i as f64) / 100.0;
            ProbValue::new(p, Distribution::Bernoulli(p.clamp(0.01, 0.99)))
        }).collect();
        let outcomes: Vec<f64> = preds.iter().map(|p| if p.value > 0.5 { 1.0 } else { 0.0 }).collect();
        let report = CalibrationChecker::check_calibration(&preds, &outcomes);
        // ECE should be relatively low for step-function outcomes
        assert!(report.ece < 0.5, "ECE was {}", report.ece);
        assert!(report.brier_score >= 0.0);
    }

    #[test]
    fn test_temperature_scaling() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = TemperatureScaling::apply(&logits, 1.0);
        // Should sum to 1
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Higher logit => higher prob
        assert!(probs[2] > probs[1]);
        assert!(probs[1] > probs[0]);
        // Higher temperature => more uniform
        let probs_hot = TemperatureScaling::apply(&logits, 10.0);
        assert!((probs_hot[2] - probs_hot[0]).abs() < (probs[2] - probs[0]).abs());
    }

    #[test]
    fn test_bayesian_layer_forward() {
        let layer = BayesianLayer::new(3, 2);
        let input = vec![1.0, 2.0, 3.0];
        let det = layer.forward_mean(&input);
        assert_eq!(det.len(), 2);
        let pv = layer.forward_sample(&input);
        // Should produce a value
        assert!(pv.value.is_finite());
    }

    #[test]
    fn test_bayesian_layer_kl() {
        let layer = BayesianLayer::new(3, 2);
        let kl = layer.kl_divergence();
        // KL should be non-negative
        assert!(kl >= 0.0, "KL was {}", kl);
    }

    #[test]
    fn test_probabilistic_model_forward() {
        let model = ProbabilisticModel::new(&[3, 4, 1]);
        let input = vec![1.0, 0.5, -1.0];
        let pv = model.forward(&input, 20);
        assert!(pv.value.is_finite());
        assert!(pv.distribution.variance() >= 0.0);
    }

    #[test]
    fn test_confidence_interval() {
        let pv = ProbValue::new(0.0, Distribution::Normal(0.0, 1.0));
        let (lo, hi) = pv.confidence_interval(0.05); // 95% CI
        // Should be roughly (-1.96, 1.96)
        assert!(lo < -1.0, "lo was {}", lo);
        assert!(hi > 1.0, "hi was {}", hi);
        assert!(lo < hi);
    }

    #[test]
    fn test_normal_entropy() {
        let d = Distribution::Normal(0.0, 1.0);
        let expected = 0.5 * (2.0 * std::f64::consts::PI * std::f64::consts::E).ln();
        assert!((d.entropy() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_empirical_distribution() {
        let samples = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let d = Distribution::Empirical(samples);
        assert!((d.mean() - 3.0).abs() < 1e-10);
        assert!((d.variance() - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_mixture_distribution() {
        let d = Distribution::Mixture(vec![
            (0.5, Distribution::Normal(0.0, 1.0)),
            (0.5, Distribution::Normal(10.0, 1.0)),
        ]);
        assert!((d.mean() - 5.0).abs() < 1e-10);
        // Variance = E[V] + V[E] = 1.0 + 0.5*25 = 13.5... actually:
        // law of total variance: sum(w*(var + (mean - overall_mean)^2))
        // = 0.5*(1 + 25) + 0.5*(1 + 25) = 26
        assert!((d.variance() - 26.0).abs() < 1e-10);
    }
}
