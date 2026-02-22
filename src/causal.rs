/// Causal Inference primitives: structural causal models, do-calculus,
/// interventions, counterfactuals, and causal discovery.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::{HashMap, HashSet, VecDeque};

// ── Deterministic RNG ────────────────────────────────────────────────

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

// ── CausalGraph ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CausalGraph {
    pub nodes: Vec<String>,
    pub edges: Vec<(usize, usize, f64)>, // (cause, effect, strength)
}

impl CausalGraph {
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }

    pub fn add_node(&mut self, name: &str) -> usize {
        let id = self.nodes.len();
        self.nodes.push(name.to_string());
        id
    }

    pub fn add_edge(&mut self, cause: usize, effect: usize, strength: f64) {
        self.edges.push((cause, effect, strength));
    }

    pub fn parents(&self, node: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.1 == node).map(|e| e.0).collect()
    }

    pub fn children(&self, node: usize) -> Vec<usize> {
        self.edges.iter().filter(|e| e.0 == node).map(|e| e.1).collect()
    }

    pub fn is_ancestor(&self, a: usize, b: usize) -> bool {
        let mut visited = HashSet::new();
        let mut stack = vec![a];
        while let Some(cur) = stack.pop() {
            if cur == b { return true; }
            if visited.insert(cur) {
                for &(src, dst, _) in &self.edges {
                    if src == cur && !visited.contains(&dst) {
                        stack.push(dst);
                    }
                }
            }
        }
        false
    }

    pub fn topological_sort(&self) -> Vec<usize> {
        let n = self.nodes.len();
        let mut in_degree = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for &(src, dst, _) in &self.edges {
            adj[src].push(dst);
            in_degree[dst] += 1;
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| in_degree[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                in_degree[v] -= 1;
                if in_degree[v] == 0 { queue.push_back(v); }
            }
        }
        order
    }

    /// Edge weight from cause -> effect (0.0 if missing)
    pub fn edge_strength(&self, cause: usize, effect: usize) -> f64 {
        self.edges.iter()
            .find(|e| e.0 == cause && e.1 == effect)
            .map(|e| e.2)
            .unwrap_or(0.0)
    }
}

// ── Mechanism ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Mechanism {
    /// value = sum(w_i * parent_i) + bias + noise
    Linear { weights: Vec<f64>, bias: f64 },
    /// Simple single hidden-layer: hidden = relu(W1*input), out = W2*hidden + noise
    Nonlinear { weights: Vec<Vec<f64>> },
    /// value = if (sum(parent_i) > threshold) 1.0 else 0.0  (+ noise)
    Threshold { threshold: f64 },
}

impl Mechanism {
    pub fn evaluate(&self, parent_values: &[f64], noise: f64) -> f64 {
        match self {
            Mechanism::Linear { weights, bias } => {
                let sum: f64 = weights.iter().zip(parent_values).map(|(w, v)| w * v).sum();
                sum + bias + noise
            }
            Mechanism::Nonlinear { weights } => {
                if weights.is_empty() { return noise; }
                // weights[0]: hidden weights (hidden_dim x input_dim flattened as rows)
                // weights[1]: output weights (1 x hidden_dim)
                let hidden_w = &weights[0];
                let n_inputs = parent_values.len().max(1);
                let n_hidden = hidden_w.len() / n_inputs;
                let mut hidden = Vec::with_capacity(n_hidden);
                for h in 0..n_hidden {
                    let mut sum = 0.0;
                    for i in 0..n_inputs.min(parent_values.len()) {
                        sum += hidden_w[h * n_inputs + i] * parent_values[i];
                    }
                    hidden.push(sum.max(0.0)); // relu
                }
                let out_w = if weights.len() > 1 { &weights[1] } else { &hidden };
                let out: f64 = out_w.iter().zip(hidden.iter()).map(|(w, h)| w * h).sum();
                out + noise
            }
            Mechanism::Threshold { threshold } => {
                let sum: f64 = parent_values.iter().sum();
                if sum + noise > *threshold { 1.0 } else { 0.0 }
            }
        }
    }
}

// ── Noise Distribution ───────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum NoiseDist {
    /// Gaussian(mean, std)
    Gaussian(f64, f64),
}

impl NoiseDist {
    fn sample(&self, rng: &mut Rng) -> f64 {
        match self {
            NoiseDist::Gaussian(mean, std) => mean + std * rng.normal(),
        }
    }
}

// ── StructuralCausalModel ────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct StructuralCausalModel {
    pub graph: CausalGraph,
    pub mechanisms: Vec<Mechanism>,
    pub noise_dists: Vec<NoiseDist>,
}

impl StructuralCausalModel {
    pub fn new(graph: CausalGraph) -> Self {
        let n = graph.nodes.len();
        // Default: linear with unit weights, zero bias, std-normal noise
        let mechanisms: Vec<Mechanism> = (0..n).map(|i| {
            let parents = graph.parents(i);
            let weights: Vec<f64> = parents.iter().map(|&p| graph.edge_strength(p, i)).collect();
            Mechanism::Linear { weights, bias: 0.0 }
        }).collect();
        let noise_dists = vec![NoiseDist::Gaussian(0.0, 1.0); n];
        Self { graph, mechanisms, noise_dists }
    }

    pub fn sample(&self) -> Vec<f64> {
        self.sample_with_seed(42)
    }

    pub fn sample_with_seed(&self, seed: u64) -> Vec<f64> {
        let mut rng = Rng::new(seed);
        self.sample_with_rng(&mut rng)
    }

    fn sample_with_rng(&self, rng: &mut Rng) -> Vec<f64> {
        let n = self.graph.nodes.len();
        let mut values = vec![0.0; n];
        let order = self.graph.topological_sort();
        for &node in &order {
            let parents = self.graph.parents(node);
            let parent_vals: Vec<f64> = parents.iter().map(|&p| values[p]).collect();
            let noise = self.noise_dists[node].sample(rng);
            values[node] = self.mechanisms[node].evaluate(&parent_vals, noise);
        }
        values
    }

    /// Rejection sampling for conditional distribution
    pub fn observe(&self, observations: &[(usize, f64)], n_samples: usize, tolerance: f64) -> Vec<f64> {
        let mut rng = Rng::new(123);
        let n = self.graph.nodes.len();
        let mut sum = vec![0.0; n];
        let mut count = 0usize;
        let max_attempts = n_samples * 1000;
        for _ in 0..max_attempts {
            if count >= n_samples { break; }
            let sample = self.sample_with_rng(&mut rng);
            let matches = observations.iter().all(|&(idx, val)| (sample[idx] - val).abs() < tolerance);
            if matches {
                for i in 0..n { sum[i] += sample[i]; }
                count += 1;
            }
        }
        if count == 0 { return vec![f64::NAN; n]; }
        sum.iter().map(|s| s / count as f64).collect()
    }
}

// ── Interventions (do-calculus) ──────────────────────────────────────

/// Returns a mutilated model with incoming edges to intervened nodes removed
pub fn do_intervention(model: &StructuralCausalModel, interventions: &[(usize, f64)]) -> StructuralCausalModel {
    let intervened: HashSet<usize> = interventions.iter().map(|&(i, _)| i).collect();
    let mut new_graph = CausalGraph::new();
    for name in &model.graph.nodes {
        new_graph.add_node(name);
    }
    // Keep only edges not pointing into intervened nodes
    for &(src, dst, w) in &model.graph.edges {
        if !intervened.contains(&dst) {
            new_graph.add_edge(src, dst, w);
        }
    }
    let mut new_model = StructuralCausalModel::new(new_graph.clone());
    // Copy existing mechanisms for non-intervened nodes
    for i in 0..model.graph.nodes.len() {
        if intervened.contains(&i) {
            // Constant mechanism
            let val = interventions.iter().find(|&&(idx, _)| idx == i).unwrap().1;
            new_model.mechanisms[i] = Mechanism::Linear { weights: vec![], bias: val };
            new_model.noise_dists[i] = NoiseDist::Gaussian(0.0, 0.0);
        } else {
            new_model.mechanisms[i] = model.mechanisms[i].clone();
            new_model.noise_dists[i] = model.noise_dists[i].clone();
        }
    }
    new_model
}

/// E[Y | do(X=treat)] - E[Y | do(X=control)]
pub fn average_treatment_effect(
    model: &StructuralCausalModel, treatment: usize, outcome: usize,
    treat_val: f64, control_val: f64, n_samples: usize,
) -> f64 {
    let treat_model = do_intervention(model, &[(treatment, treat_val)]);
    let ctrl_model = do_intervention(model, &[(treatment, control_val)]);
    let mut rng_t = Rng::new(77);
    let mut rng_c = Rng::new(77);
    let mut sum_t = 0.0;
    let mut sum_c = 0.0;
    for _ in 0..n_samples {
        sum_t += treat_model.sample_with_rng(&mut rng_t)[outcome];
        sum_c += ctrl_model.sample_with_rng(&mut rng_c)[outcome];
    }
    (sum_t / n_samples as f64) - (sum_c / n_samples as f64)
}

/// Conditional ATE: ATE restricted to subpopulation where condition holds
pub fn conditional_ate(
    model: &StructuralCausalModel, treatment: usize, outcome: usize,
    treat_val: f64, control_val: f64,
    condition: &[(usize, f64, f64)], // (node, min, max)
    n_samples: usize,
) -> f64 {
    let treat_model = do_intervention(model, &[(treatment, treat_val)]);
    let ctrl_model = do_intervention(model, &[(treatment, control_val)]);
    let mut rng_t = Rng::new(88);
    let mut rng_c = Rng::new(88);
    let mut sum_t = 0.0;
    let mut count_t = 0usize;
    let mut sum_c = 0.0;
    let mut count_c = 0usize;
    for _ in 0..(n_samples * 10) {
        let st = treat_model.sample_with_rng(&mut rng_t);
        if condition.iter().all(|&(n, lo, hi)| st[n] >= lo && st[n] <= hi) {
            sum_t += st[outcome];
            count_t += 1;
        }
        let sc = ctrl_model.sample_with_rng(&mut rng_c);
        if condition.iter().all(|&(n, lo, hi)| sc[n] >= lo && sc[n] <= hi) {
            sum_c += sc[outcome];
            count_c += 1;
        }
    }
    if count_t == 0 || count_c == 0 { return f64::NAN; }
    (sum_t / count_t as f64) - (sum_c / count_c as f64)
}

// ── Counterfactual reasoning ─────────────────────────────────────────

/// Three-step counterfactual: abduction, intervention, prediction
pub fn counterfactual(
    model: &StructuralCausalModel, factual: &[(usize, f64)],
    intervention: &[(usize, f64)],
) -> Vec<f64> {
    let n = model.graph.nodes.len();
    let order = model.graph.topological_sort();

    // Step 1: Abduction - infer noise from factual observations
    // For observed nodes, compute noise = observed - f(parents, 0)
    let mut values = vec![0.0; n];
    let mut noise = vec![0.0; n];
    let factual_map: HashMap<usize, f64> = factual.iter().cloned().collect();

    for &node in &order {
        let parents = model.graph.parents(node);
        let parent_vals: Vec<f64> = parents.iter().map(|&p| values[p]).collect();
        let det_value = model.mechanisms[node].evaluate(&parent_vals, 0.0);
        if let Some(&obs) = factual_map.get(&node) {
            noise[node] = obs - det_value;
            values[node] = obs;
        } else {
            values[node] = det_value;
            noise[node] = 0.0;
        }
    }

    // Step 2: Intervention - create mutilated model
    let int_model = do_intervention(model, intervention);
    let intervened: HashSet<usize> = intervention.iter().map(|&(i, _)| i).collect();

    // Step 3: Prediction - forward with abducted noise
    let mut cf_values = vec![0.0; n];
    let cf_order = int_model.graph.topological_sort();
    for &node in &cf_order {
        if intervened.contains(&node) {
            cf_values[node] = intervention.iter().find(|&&(i, _)| i == node).unwrap().1;
        } else {
            let parents = int_model.graph.parents(node);
            let parent_vals: Vec<f64> = parents.iter().map(|&p| cf_values[p]).collect();
            cf_values[node] = model.mechanisms[node].evaluate(&parent_vals, noise[node]);
        }
    }
    cf_values
}

// ── Causal Discovery ─────────────────────────────────────────────────

pub fn correlation(data: &[Vec<f64>], i: usize, j: usize) -> f64 {
    let n = data.len() as f64;
    if n < 2.0 { return 0.0; }
    let mean_i: f64 = data.iter().map(|r| r[i]).sum::<f64>() / n;
    let mean_j: f64 = data.iter().map(|r| r[j]).sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_i = 0.0;
    let mut var_j = 0.0;
    for row in data {
        let di = row[i] - mean_i;
        let dj = row[j] - mean_j;
        cov += di * dj;
        var_i += di * di;
        var_j += dj * dj;
    }
    if var_i < 1e-15 || var_j < 1e-15 { return 0.0; }
    cov / (var_i.sqrt() * var_j.sqrt())
}

pub fn partial_correlation(data: &[Vec<f64>], i: usize, j: usize, cond: &[usize]) -> f64 {
    if cond.is_empty() {
        return correlation(data, i, j);
    }
    // Residualize i and j on conditioning set via OLS
    let residuals_i = residualize(data, i, cond);
    let residuals_j = residualize(data, j, cond);
    // Correlation of residuals
    let n = residuals_i.len() as f64;
    let mi: f64 = residuals_i.iter().sum::<f64>() / n;
    let mj: f64 = residuals_j.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut vi = 0.0;
    let mut vj = 0.0;
    for k in 0..residuals_i.len() {
        let di = residuals_i[k] - mi;
        let dj = residuals_j[k] - mj;
        cov += di * dj;
        vi += di * di;
        vj += dj * dj;
    }
    if vi < 1e-15 || vj < 1e-15 { return 0.0; }
    cov / (vi.sqrt() * vj.sqrt())
}

fn residualize(data: &[Vec<f64>], target: usize, predictors: &[usize]) -> Vec<f64> {
    let n = data.len();
    let p = predictors.len();
    if n == 0 || p == 0 {
        return data.iter().map(|r| r[target]).collect();
    }
    // Simple OLS: beta = (X'X)^-1 X'y using normal equations
    // For simplicity with small conditioning sets, use iterative approach
    let y: Vec<f64> = data.iter().map(|r| r[target]).collect();
    let mean_y: f64 = y.iter().sum::<f64>() / n as f64;

    // For single predictor, use simple regression
    if p == 1 {
        let x: Vec<f64> = data.iter().map(|r| r[predictors[0]]).collect();
        let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for k in 0..n {
            cov += (x[k] - mean_x) * (y[k] - mean_y);
            var_x += (x[k] - mean_x) * (x[k] - mean_x);
        }
        let beta = if var_x > 1e-15 { cov / var_x } else { 0.0 };
        let alpha = mean_y - beta * mean_x;
        return (0..n).map(|k| y[k] - (alpha + beta * x[k])).collect();
    }

    // Multiple predictors: iterative residualization (sequential)
    let mut residuals = y.clone();
    for &pred in predictors {
        let x: Vec<f64> = data.iter().map(|r| r[pred]).collect();
        let mean_x: f64 = x.iter().sum::<f64>() / n as f64;
        let mean_r: f64 = residuals.iter().sum::<f64>() / n as f64;
        let mut cov = 0.0;
        let mut var_x = 0.0;
        for k in 0..n {
            cov += (x[k] - mean_x) * (residuals[k] - mean_r);
            var_x += (x[k] - mean_x) * (x[k] - mean_x);
        }
        let beta = if var_x > 1e-15 { cov / var_x } else { 0.0 };
        let alpha = mean_r - beta * mean_x;
        for k in 0..n {
            residuals[k] -= alpha + beta * x[k];
            residuals[k] += mean_r; // keep centered on original residual mean
        }
    }
    residuals
}

/// Fisher z-test for independence
pub fn independence_test(r: f64, n: usize, k: usize, alpha: f64) -> bool {
    // z = 0.5 * ln((1+r)/(1-r)), test stat = sqrt(n - k - 3) * |z|
    let r_clamped = r.clamp(-0.9999, 0.9999);
    let z = 0.5 * ((1.0 + r_clamped) / (1.0 - r_clamped)).ln();
    let dof = n as f64 - k as f64 - 3.0;
    if dof <= 0.0 { return true; } // not enough data -> assume independent
    let stat = dof.sqrt() * z.abs();
    // Compare against z critical value (two-tailed)
    // For alpha=0.05, z_crit ~= 1.96; alpha=0.01 -> 2.576
    let z_crit = if alpha <= 0.01 { 2.576 } else if alpha <= 0.05 { 1.96 } else { 1.645 };
    stat < z_crit // true = independent
}

/// PC algorithm: learn causal skeleton from observational data
pub fn pc_algorithm(data: &[Vec<f64>], alpha: f64) -> CausalGraph {
    if data.is_empty() { return CausalGraph::new(); }
    let p = data[0].len();
    let n = data.len();
    let mut graph = CausalGraph::new();
    for i in 0..p {
        graph.add_node(&format!("X{}", i));
    }
    // Start fully connected (undirected = bidirectional)
    let mut adj: Vec<HashSet<usize>> = vec![(0..p).collect(); p];
    for i in 0..p { adj[i].remove(&i); }

    // Remove edges by conditional independence tests with increasing conditioning set size
    let mut max_cond = 0;
    loop {
        let mut removed_any = false;
        for i in 0..p {
            for j in (i + 1)..p {
                if !adj[i].contains(&j) { continue; }
                let neighbors: Vec<usize> = adj[i].iter().filter(|&&x| x != j).cloned().collect();
                if neighbors.len() < max_cond { continue; }
                // Test all conditioning sets of size max_cond
                let subsets = combinations(&neighbors, max_cond);
                for subset in &subsets {
                    let r = partial_correlation(data, i, j, subset);
                    if independence_test(r, n, subset.len(), alpha) {
                        adj[i].remove(&j);
                        adj[j].remove(&i);
                        removed_any = true;
                        break;
                    }
                }
            }
        }
        if !removed_any || max_cond >= p - 2 { break; }
        max_cond += 1;
    }

    // Orient edges (simplified: use correlation magnitude as proxy for direction)
    for i in 0..p {
        for &j in &adj[i].clone() {
            if i < j {
                let r = correlation(data, i, j).abs();
                graph.add_edge(i, j, r);
            }
        }
    }
    graph
}

fn combinations(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if items.len() < k { return vec![]; }
    let mut result = Vec::new();
    for (idx, &item) in items.iter().enumerate() {
        for mut sub in combinations(&items[idx + 1..], k - 1) {
            sub.insert(0, item);
            result.push(sub);
        }
    }
    result
}

// ── DifferentiableSCM ────────────────────────────────────────────────

/// Mean squared error loss
pub fn scm_loss(model: &StructuralCausalModel, data: &[Vec<f64>]) -> f64 {
    let _n = model.graph.nodes.len();
    let order = model.graph.topological_sort();
    let mut total_loss = 0.0;
    for row in data {
        for &node in &order {
            let parents = model.graph.parents(node);
            let parent_vals: Vec<f64> = parents.iter().map(|&p| row[p]).collect();
            let predicted = model.mechanisms[node].evaluate(&parent_vals, 0.0);
            let diff = predicted - row[node];
            total_loss += diff * diff;
        }
    }
    total_loss / data.len() as f64
}

/// Gradient step for linear mechanisms
pub fn update_mechanisms(model: &mut StructuralCausalModel, data: &[Vec<f64>], lr: f64) {
    let order = model.graph.topological_sort();
    let n_data = data.len() as f64;

    for &node in &order {
        let parents = model.graph.parents(node);
        if parents.is_empty() { continue; }

        // Compute gradient for linear mechanism
        if let Mechanism::Linear { ref weights, ref bias } = model.mechanisms[node] {
            let mut grad_w = vec![0.0; weights.len()];
            let mut grad_b = 0.0;

            for row in data {
                let parent_vals: Vec<f64> = parents.iter().map(|&p| row[p]).collect();
                let predicted: f64 = weights.iter().zip(&parent_vals).map(|(w, v)| w * v).sum::<f64>() + bias;
                let err = predicted - row[node];
                for (k, pv) in parent_vals.iter().enumerate() {
                    grad_w[k] += 2.0 * err * pv / n_data;
                }
                grad_b += 2.0 * err / n_data;
            }

            let new_weights: Vec<f64> = weights.iter().zip(&grad_w).map(|(w, g)| w - lr * g).collect();
            let new_bias = bias - lr * grad_b;
            model.mechanisms[node] = Mechanism::Linear { weights: new_weights, bias: new_bias };
        }
    }
}

// ── Interpreter builtins ─────────────────────────────────────────────

fn val_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected number, got {:?}", v)),
    }
}

fn val_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(i) => Ok(*i as usize),
        _ => Err(format!("expected int, got {:?}", v)),
    }
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("causal_model_new".into(), FnDef::Builtin(builtin_causal_model_new));
    env.functions.insert("causal_add_node".into(), FnDef::Builtin(builtin_causal_add_node));
    env.functions.insert("causal_add_edge".into(), FnDef::Builtin(builtin_causal_add_edge));
    env.functions.insert("causal_sample".into(), FnDef::Builtin(builtin_causal_sample));
    env.functions.insert("causal_intervene".into(), FnDef::Builtin(builtin_causal_intervene));
    env.functions.insert("causal_ate".into(), FnDef::Builtin(builtin_causal_ate));
    env.functions.insert("causal_counterfactual".into(), FnDef::Builtin(builtin_causal_counterfactual));
    env.functions.insert("causal_discover".into(), FnDef::Builtin(builtin_causal_discover));
}

fn builtin_causal_model_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() { return Err("causal_model_new takes 0 args".into()); }
    let graph = CausalGraph::new();
    let model = StructuralCausalModel::new(graph);
    let id = env.next_causal_id;
    env.next_causal_id += 1;
    env.causal_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_causal_add_node(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("causal_add_node(model_id, name)".into()); }
    let id = val_to_usize(&args[0])?;
    let name = match &args[1] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let model = env.causal_models.get_mut(&id).ok_or("invalid causal model id")?;
    let node_id = model.graph.add_node(&name);
    // Extend mechanisms and noise dists
    let parents = model.graph.parents(node_id);
    let weights: Vec<f64> = parents.iter().map(|&p| model.graph.edge_strength(p, node_id)).collect();
    model.mechanisms.push(Mechanism::Linear { weights, bias: 0.0 });
    model.noise_dists.push(NoiseDist::Gaussian(0.0, 1.0));
    Ok(Value::Int(node_id as i128))
}

fn builtin_causal_add_edge(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("causal_add_edge(model_id, cause, effect, strength)".into()); }
    let id = val_to_usize(&args[0])?;
    let cause = val_to_usize(&args[1])?;
    let effect = val_to_usize(&args[2])?;
    let strength = val_to_f64(&args[3])?;
    let model = env.causal_models.get_mut(&id).ok_or("invalid causal model id")?;
    model.graph.add_edge(cause, effect, strength);
    // Update mechanism weights for effect node
    let parents = model.graph.parents(effect);
    let weights: Vec<f64> = parents.iter().map(|&p| model.graph.edge_strength(p, effect)).collect();
    if let Some(mech) = model.mechanisms.get_mut(effect) {
        *mech = Mechanism::Linear { weights, bias: 0.0 };
    }
    Ok(Value::Void)
}

fn builtin_causal_sample(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("causal_sample(model_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let model = env.causal_models.get(&id).ok_or("invalid causal model id")?;
    let values = model.sample();
    Ok(Value::Array(values.into_iter().map(Value::Float).collect()))
}

fn builtin_causal_intervene(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("causal_intervene(model_id, node, value)".into()); }
    let id = val_to_usize(&args[0])?;
    let node = val_to_usize(&args[1])?;
    let val = val_to_f64(&args[2])?;
    let model = env.causal_models.get(&id).ok_or("invalid causal model id")?;
    let new_model = do_intervention(model, &[(node, val)]);
    let new_id = env.next_causal_id;
    env.next_causal_id += 1;
    env.causal_models.insert(new_id, new_model);
    Ok(Value::Int(new_id as i128))
}

fn builtin_causal_ate(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("causal_ate(model_id, treatment, outcome, n_samples)".into()); }
    let id = val_to_usize(&args[0])?;
    let treatment = val_to_usize(&args[1])?;
    let outcome = val_to_usize(&args[2])?;
    let n = val_to_usize(&args[3])?;
    let model = env.causal_models.get(&id).ok_or("invalid causal model id")?;
    let ate = average_treatment_effect(model, treatment, outcome, 1.0, 0.0, n);
    Ok(Value::Float(ate))
}

fn builtin_causal_counterfactual(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("causal_counterfactual(model_id, factual, intervention)".into()); }
    let id = val_to_usize(&args[0])?;
    let factual = parse_pairs(&args[1])?;
    let intervention = parse_pairs(&args[2])?;
    let model = env.causal_models.get(&id).ok_or("invalid causal model id")?;
    let result = counterfactual(model, &factual, &intervention);
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_causal_discover(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("causal_discover(data, alpha)".into()); }
    let data = match &args[0] {
        Value::Array(rows) => {
            rows.iter().map(|row| match row {
                Value::Array(vals) => vals.iter().map(val_to_f64).collect::<Result<Vec<_>, _>>(),
                _ => Err("data rows must be arrays".into()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("data must be array of arrays".into()),
    };
    let alpha = val_to_f64(&args[1])?;
    let graph = pc_algorithm(&data, alpha);
    let model = StructuralCausalModel::new(graph);
    let id = env.next_causal_id;
    env.next_causal_id += 1;
    env.causal_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn parse_pairs(v: &Value) -> Result<Vec<(usize, f64)>, String> {
    match v {
        Value::Array(arr) => {
            arr.iter().map(|item| match item {
                Value::Array(pair) if pair.len() == 2 => {
                    Ok((val_to_usize(&pair[0])?, val_to_f64(&pair[1])?))
                }
                Value::Tuple(pair) if pair.len() == 2 => {
                    Ok((val_to_usize(&pair[0])?, val_to_f64(&pair[1])?))
                }
                _ => Err("expected [node, value] pairs".into()),
            }).collect()
        }
        _ => Err("expected array of pairs".into()),
    }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_chain() -> StructuralCausalModel {
        // X -> Y -> Z with weights 2.0 and 3.0
        let mut g = CausalGraph::new();
        let x = g.add_node("X");
        let y = g.add_node("Y");
        let z = g.add_node("Z");
        g.add_edge(x, y, 2.0);
        g.add_edge(y, z, 3.0);
        let mut m = StructuralCausalModel::new(g);
        // Zero noise for deterministic tests
        m.noise_dists = vec![NoiseDist::Gaussian(0.0, 0.0); 3];
        // X is exogenous with bias=1
        m.mechanisms[0] = Mechanism::Linear { weights: vec![], bias: 1.0 };
        m
    }

    #[test]
    fn test_causal_graph_basics() {
        let mut g = CausalGraph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge(a, b, 1.0);
        g.add_edge(b, c, 1.0);
        assert_eq!(g.parents(b), vec![a]);
        assert_eq!(g.children(a), vec![b]);
        assert!(g.is_ancestor(a, c));
        assert!(!g.is_ancestor(c, a));
    }

    #[test]
    fn test_topological_sort() {
        let mut g = CausalGraph::new();
        let a = g.add_node("A");
        let b = g.add_node("B");
        let c = g.add_node("C");
        g.add_edge(a, b, 1.0);
        g.add_edge(a, c, 1.0);
        g.add_edge(b, c, 1.0);
        let order = g.topological_sort();
        assert_eq!(order[0], a);
        let pos_b = order.iter().position(|&x| x == b).unwrap();
        let pos_c = order.iter().position(|&x| x == c).unwrap();
        assert!(pos_b < pos_c);
    }

    #[test]
    fn test_sample_deterministic_chain() {
        let m = simple_chain();
        let vals = m.sample();
        // X=1, Y=2*1=2, Z=3*2=6
        assert_eq!(vals[0], 1.0);
        assert_eq!(vals[1], 2.0);
        assert_eq!(vals[2], 6.0);
    }

    #[test]
    fn test_mechanism_linear() {
        let mech = Mechanism::Linear { weights: vec![2.0, 3.0], bias: 1.0 };
        let val = mech.evaluate(&[1.0, 2.0], 0.5);
        assert!((val - 9.5).abs() < 1e-10); // 2*1 + 3*2 + 1 + 0.5
    }

    #[test]
    fn test_mechanism_threshold() {
        let mech = Mechanism::Threshold { threshold: 1.5 };
        assert_eq!(mech.evaluate(&[1.0, 0.3], 0.0), 0.0); // 1.3 < 1.5
        assert_eq!(mech.evaluate(&[1.0, 1.0], 0.0), 1.0); // 2.0 > 1.5
        assert_eq!(mech.evaluate(&[0.5, 0.5], 0.6), 1.0); // 1.0 + 0.6 > 1.5
    }

    #[test]
    fn test_do_intervention() {
        let m = simple_chain();
        let intervened = do_intervention(&m, &[(1, 5.0)]); // do(Y=5)
        let vals = intervened.sample();
        assert_eq!(vals[0], 1.0); // X unchanged
        assert_eq!(vals[1], 5.0); // Y set to 5
        assert_eq!(vals[2], 15.0); // Z = 3 * 5
    }

    #[test]
    fn test_ate() {
        let m = simple_chain();
        // ATE of Y on Z: do(Y=1) vs do(Y=0), Z = 3*Y so ATE = 3*1 - 3*0 = 3
        let ate = average_treatment_effect(&m, 1, 2, 1.0, 0.0, 100);
        assert!((ate - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_counterfactual() {
        let m = simple_chain();
        // Factual: X=1, Y=2, Z=6. Counterfactual: what if do(Y=10)?
        let cf = counterfactual(&m, &[(0, 1.0), (1, 2.0), (2, 6.0)], &[(1, 10.0)]);
        assert_eq!(cf[0], 1.0); // X unchanged
        assert_eq!(cf[1], 10.0); // Y intervened
        assert_eq!(cf[2], 30.0); // Z = 3 * 10
    }

    #[test]
    fn test_correlation() {
        let data = vec![
            vec![1.0, 2.0],
            vec![2.0, 4.0],
            vec![3.0, 6.0],
        ];
        let r = correlation(&data, 0, 1);
        assert!((r - 1.0).abs() < 1e-10); // perfect correlation
    }

    #[test]
    fn test_independence_test() {
        // High correlation, large n -> not independent
        assert!(!independence_test(0.8, 100, 0, 0.05));
        // Low correlation -> independent
        assert!(independence_test(0.01, 100, 0, 0.05));
    }

    #[test]
    fn test_pc_algorithm() {
        // Generate data from X -> Y (Y = 2*X + noise)
        let data: Vec<Vec<f64>> = (0..200).map(|i| {
            let x = i as f64 * 0.1;
            let y = 2.0 * x + 0.01;
            vec![x, y]
        }).collect();
        let graph = pc_algorithm(&data, 0.05);
        // Should find an edge between X0 and X1
        assert!(!graph.edges.is_empty());
    }

    #[test]
    fn test_scm_loss_and_update() {
        let mut m = simple_chain();
        // Corrupt the weight slightly
        m.mechanisms[1] = Mechanism::Linear { weights: vec![1.5], bias: 0.0 }; // should be 2.0
        let data = vec![
            vec![1.0, 2.0, 6.0],
            vec![2.0, 4.0, 12.0],
        ];
        let loss_before = scm_loss(&m, &data);
        update_mechanisms(&mut m, &data, 0.1);
        let loss_after = scm_loss(&m, &data);
        assert!(loss_after < loss_before, "loss should decrease after gradient step");
    }

    #[test]
    fn test_nonlinear_mechanism() {
        // Simple 2-input, 2-hidden nonlinear mechanism
        let weights = vec![
            vec![1.0, 0.0, 0.0, 1.0], // 2x2 hidden weights
            vec![1.0, 1.0],             // 1x2 output weights
        ];
        let mech = Mechanism::Nonlinear { weights };
        let val = mech.evaluate(&[2.0, 3.0], 0.0);
        // hidden = relu([1*2+0*3, 0*2+1*3]) = [2, 3]
        // out = 1*2 + 1*3 = 5
        assert!((val - 5.0).abs() < 1e-10);
    }
}
