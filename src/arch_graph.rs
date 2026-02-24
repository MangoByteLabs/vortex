// Pillar 2: Architecture as First-Class Data for Vortex
// Represent neural network architectures as mutable, evolvable graphs.
// Supports creation, mutation, crossover, verification, forward passes,
// FLOPs estimation, energy estimation, gradient flow analysis, and serialization.

use std::collections::HashMap;
use crate::interpreter::{Env, Value, FnDef};

// ─── PRNG ────────────────────────────────────────────────────────────────────

struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        XorShift64 { state: if seed == 0 { 0xDEAD_BEEF_CAFE_1234 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }

    fn randn(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }

    fn randn_vec(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.randn() * 0.1).collect()
    }
}

// ─── Helper functions ────────────────────────────────────────────────────────

fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => 0.0,
    }
}

fn value_to_usize(v: &Value) -> usize {
    match v {
        Value::Int(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => 0,
    }
}

fn value_to_f64_vec(v: &Value) -> Vec<f64> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => vec![],
    }
}

fn value_to_f64_2d(v: &Value) -> Vec<Vec<f64>> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect(),
        _ => vec![],
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        _ => format!("{}", v),
    }
}

fn value_to_usize_vec(v: &Value) -> Vec<usize> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_usize).collect(),
        _ => vec![],
    }
}

// ─── Math helpers ────────────────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum.max(1e-12)).collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn mat_vec_mul(mat: &[f64], rows: usize, cols: usize, v: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; rows];
    for r in 0..rows {
        for c in 0..cols.min(v.len()) {
            out[r] += mat[r * cols + c] * v[c];
        }
    }
    out
}

// ─── Architecture graph types ────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum ArchNodeKind {
    Linear { in_features: usize, out_features: usize },
    Conv2d { in_ch: usize, out_ch: usize, kernel: usize },
    Attention { embed_dim: usize, num_heads: usize },
    SSM { state_dim: usize, input_dim: usize },
    Spiking { neurons: usize, threshold: f64 },
    MoE { num_experts: usize, expert_dim: usize, top_k: usize },
    ODEBlock { dim: usize, tol: f64 },
    Residual { subgraph_id: usize },
    ManifoldLayer { kind: String, dim: usize },
    Custom { name: String, params: HashMap<String, f64> },
}

#[derive(Debug, Clone)]
pub struct ArchNode {
    pub id: usize,
    pub kind: ArchNodeKind,
    pub name: String,
    pub specs: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ArchEdge {
    pub from: usize,
    pub to: usize,
    pub shape: Vec<usize>,
}

#[derive(Debug, Clone)]
pub struct ArchGraph {
    pub nodes: HashMap<usize, ArchNode>,
    pub edges: Vec<ArchEdge>,
    pub next_node_id: usize,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone)]
pub enum Mutation {
    AddNode { kind: ArchNodeKind, name: String },
    RemoveNode(usize),
    ReplaceNode { id: usize, kind: ArchNodeKind },
    AddSkip { from: usize, to: usize },
    ChangeParam { node_id: usize, param: String, value: f64 },
    SplitNode(usize),
    MergeNodes(usize, usize),
}

impl ArchGraph {
    fn new() -> Self {
        ArchGraph {
            nodes: HashMap::new(),
            edges: Vec::new(),
            next_node_id: 0,
            metadata: HashMap::new(),
        }
    }

    fn add_node(&mut self, kind: ArchNodeKind, name: String) -> usize {
        let id = self.next_node_id;
        self.next_node_id += 1;
        self.nodes.insert(id, ArchNode { id, kind, name, specs: vec![] });
        id
    }

    fn connect(&mut self, from: usize, to: usize, shape: Vec<usize>) {
        self.edges.push(ArchEdge { from, to, shape });
    }

    fn remove_node(&mut self, node_id: usize) {
        self.nodes.remove(&node_id);
        self.edges.retain(|e| e.from != node_id && e.to != node_id);
    }

    fn node_count(&self) -> usize {
        self.nodes.len()
    }

    fn edge_count(&self) -> usize {
        self.edges.len()
    }

    fn topological_sort(&self) -> Vec<usize> {
        let mut in_degree: HashMap<usize, usize> = HashMap::new();
        for &id in self.nodes.keys() {
            in_degree.insert(id, 0);
        }
        for e in &self.edges {
            if let Some(d) = in_degree.get_mut(&e.to) {
                *d += 1;
            }
        }
        let mut queue: Vec<usize> = in_degree.iter()
            .filter(|(_, &d)| d == 0)
            .map(|(&id, _)| id)
            .collect();
        queue.sort();
        let mut result = Vec::new();
        while let Some(n) = queue.pop() {
            result.push(n);
            let successors: Vec<usize> = self.edges.iter()
                .filter(|e| e.from == n)
                .map(|e| e.to)
                .collect();
            for s in successors {
                if let Some(d) = in_degree.get_mut(&s) {
                    *d -= 1;
                    if *d == 0 {
                        queue.push(s);
                        queue.sort();
                    }
                }
            }
        }
        result
    }

    fn estimate_flops(&self) -> u64 {
        let mut total: u64 = 0;
        let seq_len: u64 = 128; // default assumption
        let spatial: u64 = 32 * 32;
        for node in self.nodes.values() {
            total += match &node.kind {
                ArchNodeKind::Linear { in_features, out_features } =>
                    2 * (*in_features as u64) * (*out_features as u64),
                ArchNodeKind::Conv2d { in_ch, out_ch, kernel, .. } =>
                    2 * (*in_ch as u64) * (*out_ch as u64) * (*kernel as u64) * (*kernel as u64) * spatial,
                ArchNodeKind::Attention { embed_dim, .. } =>
                    4 * (*embed_dim as u64) * (*embed_dim as u64) * seq_len,
                ArchNodeKind::SSM { state_dim, input_dim } =>
                    2 * (*state_dim as u64) * (*input_dim as u64) * seq_len,
                ArchNodeKind::Spiking { neurons, .. } =>
                    3 * (*neurons as u64) * seq_len,
                ArchNodeKind::MoE { num_experts, expert_dim, top_k } =>
                    2 * (*top_k as u64) * (*expert_dim as u64) * (*expert_dim as u64) + (*num_experts as u64) * (*expert_dim as u64),
                ArchNodeKind::ODEBlock { dim, .. } =>
                    10 * 2 * (*dim as u64) * (*dim as u64), // ~10 ODE steps
                ArchNodeKind::Residual { .. } => 0, // subgraph counted separately
                ArchNodeKind::ManifoldLayer { dim, .. } =>
                    3 * (*dim as u64) * (*dim as u64),
                ArchNodeKind::Custom { params, .. } =>
                    *params.get("flops").unwrap_or(&1000.0) as u64,
            };
        }
        total
    }

    fn estimate_energy(&self) -> f64 {
        // Simple model: ~10 pJ per FLOP (approximate GPU energy)
        let flops = self.estimate_flops();
        flops as f64 * 10e-12
    }

    fn analyze_gradient_flow(&self) -> String {
        let order = self.topological_sort();
        let depth = order.len();
        if depth == 0 {
            return "Empty graph - no gradient flow".to_string();
        }

        let mut issues = Vec::new();
        let mut consecutive_linear = 0;
        let mut has_residual = false;
        let mut has_attention = false;

        for &nid in &order {
            if let Some(node) = self.nodes.get(&nid) {
                match &node.kind {
                    ArchNodeKind::Linear { .. } => {
                        consecutive_linear += 1;
                        if consecutive_linear > 5 {
                            issues.push(format!("Warning: {} consecutive linear layers may cause vanishing gradients", consecutive_linear));
                        }
                    }
                    ArchNodeKind::Residual { .. } => {
                        has_residual = true;
                        consecutive_linear = 0;
                    }
                    ArchNodeKind::Attention { .. } => {
                        has_attention = true;
                        consecutive_linear = 0;
                    }
                    _ => { consecutive_linear = 0; }
                }
            }
        }

        if depth > 20 && !has_residual {
            issues.push(format!("Warning: deep network ({} layers) without residual connections", depth));
        }

        // Check for skip connections
        let skip_count = self.edges.iter().filter(|e| {
            let from_pos = order.iter().position(|&x| x == e.from).unwrap_or(0);
            let to_pos = order.iter().position(|&x| x == e.to).unwrap_or(0);
            to_pos > from_pos + 1
        }).count();

        let mut result = format!("Gradient flow analysis: depth={}, nodes={}, skip_connections={}", depth, self.node_count(), skip_count);
        if has_residual {
            result.push_str("; residual connections present (good)");
        }
        if has_attention {
            result.push_str("; attention layers present");
        }
        for issue in &issues {
            result.push_str(&format!("; {}", issue));
        }
        if issues.is_empty() {
            result.push_str("; no issues detected");
        }
        result
    }

    fn to_json(&self) -> String {
        let mut s = String::from("{\n");
        // nodes
        s.push_str("  \"nodes\": [\n");
        let mut node_ids: Vec<usize> = self.nodes.keys().cloned().collect();
        node_ids.sort();
        for (i, &nid) in node_ids.iter().enumerate() {
            let node = &self.nodes[&nid];
            let kind_str = arch_node_kind_to_string(&node.kind);
            let params_str = arch_node_kind_params_json(&node.kind);
            s.push_str(&format!("    {{\"id\": {}, \"kind\": \"{}\", \"name\": \"{}\", \"params\": {{{}}}, \"specs\": {:?}}}",
                nid, kind_str, node.name, params_str, node.specs));
            if i + 1 < node_ids.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("  ],\n");
        // edges
        s.push_str("  \"edges\": [\n");
        for (i, e) in self.edges.iter().enumerate() {
            s.push_str(&format!("    {{\"from\": {}, \"to\": {}, \"shape\": {:?}}}", e.from, e.to, e.shape));
            if i + 1 < self.edges.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("  ],\n");
        // metadata
        s.push_str("  \"metadata\": {\n");
        let meta_keys: Vec<&String> = self.metadata.keys().collect();
        for (i, k) in meta_keys.iter().enumerate() {
            s.push_str(&format!("    \"{}\": \"{}\"", k, self.metadata[*k]));
            if i + 1 < meta_keys.len() { s.push(','); }
            s.push('\n');
        }
        s.push_str("  }\n");
        s.push('}');
        s
    }

    fn forward(&self, input: &[f64], graphs: &HashMap<usize, ArchGraph>) -> Vec<f64> {
        let order = self.topological_sort();
        if order.is_empty() {
            return input.to_vec();
        }
        let mut rng = XorShift64::new(42);
        let mut node_outputs: HashMap<usize, Vec<f64>> = HashMap::new();

        for (idx, &nid) in order.iter().enumerate() {
            let node = match self.nodes.get(&nid) {
                Some(n) => n,
                None => continue,
            };

            // Gather input: either from predecessors or the original input
            let predecessors: Vec<usize> = self.edges.iter()
                .filter(|e| e.to == nid)
                .map(|e| e.from)
                .collect();

            let node_input = if predecessors.is_empty() {
                input.to_vec()
            } else {
                // Sum all predecessor outputs
                let mut combined = vec![0.0; 0];
                for &pred in &predecessors {
                    if let Some(out) = node_outputs.get(&pred) {
                        if combined.is_empty() {
                            combined = out.clone();
                        } else {
                            let len = combined.len().min(out.len());
                            for i in 0..len {
                                combined[i] += out[i];
                            }
                        }
                    }
                }
                if combined.is_empty() { input.to_vec() } else { combined }
            };

            let output = forward_node(node, &node_input, &mut rng, graphs);
            node_outputs.insert(nid, output);
        }

        // Return last node's output
        if let Some(&last) = order.last() {
            node_outputs.remove(&last).unwrap_or_else(|| input.to_vec())
        } else {
            input.to_vec()
        }
    }
}

fn forward_node(node: &ArchNode, input: &[f64], rng: &mut XorShift64, graphs: &HashMap<usize, ArchGraph>) -> Vec<f64> {
    match &node.kind {
        ArchNodeKind::Linear { in_features, out_features } => {
            let inf = *in_features;
            let outf = *out_features;
            // Generate weight matrix
            let weights = rng.randn_vec(inf * outf);
            let bias = rng.randn_vec(outf);
            let mut padded = input.to_vec();
            padded.resize(inf, 0.0);
            let mut out = mat_vec_mul(&weights, outf, inf, &padded);
            for i in 0..outf {
                out[i] += bias[i];
            }
            out
        }
        ArchNodeKind::Conv2d { in_ch, out_ch, kernel } => {
            // Treat input as flat spatial data, apply simplified convolution
            let k = *kernel;
            let spatial = if input.len() > *in_ch { input.len() / in_ch } else { 1 };
            let out_spatial = if spatial >= k { spatial - k + 1 } else { 1 };
            let filter_size = in_ch * k * k;
            let filters: Vec<Vec<f64>> = (0..*out_ch).map(|_| rng.randn_vec(filter_size)).collect();
            let mut output = Vec::with_capacity(out_ch * out_spatial);
            for oc in 0..*out_ch {
                for s in 0..out_spatial {
                    let mut val = 0.0;
                    for ic in 0..*in_ch {
                        for ki in 0..k {
                            let idx = ic * spatial + s + ki;
                            if idx < input.len() && ki < filters[oc].len() {
                                val += input[idx] * filters[oc][ic * k + ki];
                            }
                        }
                    }
                    output.push(val);
                }
            }
            output
        }
        ArchNodeKind::Attention { embed_dim, num_heads } => {
            let d = *embed_dim;
            let h = *num_heads;
            let head_dim = if h > 0 { d / h } else { d };
            let seq_len = if d > 0 { (input.len() + d - 1) / d } else { 1 };
            let seq_len = seq_len.max(1);

            // Generate Q, K, V projection weights
            let wq = rng.randn_vec(d * d);
            let wk = rng.randn_vec(d * d);
            let wv = rng.randn_vec(d * d);

            // Pad input to seq_len * d
            let mut padded = input.to_vec();
            padded.resize(seq_len * d, 0.0);

            // For each position, compute Q, K, V
            let mut queries = vec![vec![0.0; d]; seq_len];
            let mut keys = vec![vec![0.0; d]; seq_len];
            let mut values = vec![vec![0.0; d]; seq_len];

            for t in 0..seq_len {
                let x = &padded[t * d..(t + 1) * d];
                queries[t] = mat_vec_mul(&wq, d, d, x);
                keys[t] = mat_vec_mul(&wk, d, d, x);
                values[t] = mat_vec_mul(&wv, d, d, x);
            }

            // Scaled dot-product attention per head
            let scale = (head_dim as f64).sqrt().max(1.0);
            let mut output = vec![0.0; seq_len * d];

            for head in 0..h {
                let hstart = head * head_dim;
                let hend = (hstart + head_dim).min(d);
                for t in 0..seq_len {
                    // Compute attention scores
                    let mut scores = Vec::with_capacity(seq_len);
                    for s in 0..seq_len {
                        let mut sc = 0.0;
                        for k in hstart..hend {
                            sc += queries[t][k] * keys[s][k];
                        }
                        scores.push(sc / scale);
                    }
                    let attn = softmax(&scores);
                    // Weighted sum of values
                    for k in hstart..hend {
                        let mut val = 0.0;
                        for s in 0..seq_len {
                            val += attn[s] * values[s][k];
                        }
                        output[t * d + k] += val;
                    }
                }
            }
            output
        }
        ArchNodeKind::SSM { state_dim, input_dim } => {
            let sd = *state_dim;
            let id = *input_dim;
            // State-space model: x[t+1] = A*x[t] + B*u[t]; y[t] = C*x[t]
            let a_mat = rng.randn_vec(sd * sd);
            let b_mat = rng.randn_vec(sd * id);
            let c_mat = rng.randn_vec(id * sd);

            let seq_len = if id > 0 { (input.len() + id - 1) / id } else { 1 };
            let seq_len = seq_len.max(1);

            let mut padded = input.to_vec();
            padded.resize(seq_len * id, 0.0);

            let mut state = vec![0.0; sd];
            let mut output = Vec::with_capacity(seq_len * id);

            for t in 0..seq_len {
                let u = &padded[t * id..(t + 1) * id];
                // x = A*x + B*u
                let ax = mat_vec_mul(&a_mat, sd, sd, &state);
                let bu = mat_vec_mul(&b_mat, sd, id, u);
                for i in 0..sd {
                    state[i] = ax[i] * 0.9 + bu[i]; // decay for stability
                }
                // y = C*x
                let y = mat_vec_mul(&c_mat, id, sd, &state);
                output.extend_from_slice(&y);
            }
            output
        }
        ArchNodeKind::Spiking { neurons, threshold } => {
            // Leaky Integrate-and-Fire neuron
            let n = *neurons;
            let thresh = *threshold;
            let seq_len = if n > 0 { (input.len() + n - 1) / n } else { 1 };
            let seq_len = seq_len.max(1);

            let mut padded = input.to_vec();
            padded.resize(seq_len * n, 0.0);

            let weights = rng.randn_vec(n * n);
            let mut membrane = vec![0.0; n];
            let mut output = Vec::with_capacity(seq_len * n);
            let leak = 0.9;

            for t in 0..seq_len {
                let x = &padded[t * n..(t + 1) * n];
                let syn = mat_vec_mul(&weights, n, n, x);
                let mut spikes = vec![0.0; n];
                for i in 0..n {
                    membrane[i] = membrane[i] * leak + syn[i];
                    if membrane[i] > thresh {
                        spikes[i] = 1.0;
                        membrane[i] = 0.0; // reset
                    }
                }
                output.extend_from_slice(&spikes);
            }
            output
        }
        ArchNodeKind::MoE { num_experts, expert_dim, top_k } => {
            let ne = *num_experts;
            let ed = *expert_dim;
            let k = (*top_k).min(ne);

            // Gate: compute scores for each expert
            let gate_weights = rng.randn_vec(ne * input.len().max(1));
            let mut gate_scores = Vec::with_capacity(ne);
            for e in 0..ne {
                let mut score = 0.0;
                for (i, &x) in input.iter().enumerate() {
                    if e * input.len() + i < gate_weights.len() {
                        score += x * gate_weights[e * input.len() + i];
                    }
                }
                gate_scores.push(score);
            }

            // Top-k selection
            let mut indexed: Vec<(usize, f64)> = gate_scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(k);

            let top_scores: Vec<f64> = indexed.iter().map(|(_, s)| *s).collect();
            let top_weights = softmax(&top_scores);

            // Expert forward passes (each expert is a linear layer)
            let mut output = vec![0.0; ed];
            for (idx, &(expert_id, _)) in indexed.iter().enumerate() {
                let expert_w = rng.randn_vec(ed * input.len().max(1));
                let mut expert_out = vec![0.0; ed];
                for r in 0..ed {
                    for (c, &x) in input.iter().enumerate() {
                        if r * input.len() + c < expert_w.len() {
                            expert_out[r] += x * expert_w[r * input.len() + c];
                        }
                    }
                }
                for i in 0..ed {
                    output[i] += top_weights[idx] * expert_out[i];
                }
            }
            output
        }
        ArchNodeKind::ODEBlock { dim, tol } => {
            // Simple Euler ODE solver: dx/dt = f(x) where f is a learned transform
            let d = *dim;
            let _tol = *tol;
            let weights = rng.randn_vec(d * d);
            let mut state = input.to_vec();
            state.resize(d, 0.0);

            let steps = 10;
            let dt = 0.1;
            for _ in 0..steps {
                let deriv = mat_vec_mul(&weights, d, d, &state);
                for i in 0..d {
                    state[i] += dt * deriv[i].tanh(); // tanh for stability
                }
            }
            state
        }
        ArchNodeKind::Residual { subgraph_id } => {
            // Forward through subgraph and add skip connection
            if let Some(subgraph) = graphs.get(subgraph_id) {
                let sub_out = subgraph.forward(input, graphs);
                vec_add_padded(input, &sub_out)
            } else {
                input.to_vec()
            }
        }
        ArchNodeKind::ManifoldLayer { kind, dim } => {
            let d = *dim;
            let mut x = input.to_vec();
            x.resize(d, 0.0);
            match kind.as_str() {
                "sphere" => {
                    // Project onto unit sphere
                    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
                    x.iter().map(|v| v / n).collect()
                }
                "hyperbolic" => {
                    // Poincare ball projection
                    let n = x.iter().map(|v| v * v).sum::<f64>().sqrt();
                    if n >= 1.0 {
                        let scale = 0.99 / n;
                        x.iter().map(|v| v * scale).collect()
                    } else {
                        x
                    }
                }
                _ => {
                    // Default: apply tanh nonlinearity
                    x.iter().map(|v| v.tanh()).collect()
                }
            }
        }
        ArchNodeKind::Custom { params, .. } => {
            // Apply a simple learned transform based on params
            let scale = params.get("scale").copied().unwrap_or(1.0);
            let bias = params.get("bias").copied().unwrap_or(0.0);
            input.iter().map(|&x| x * scale + bias).collect()
        }
    }
}

fn vec_add_padded(a: &[f64], b: &[f64]) -> Vec<f64> {
    let len = a.len().max(b.len());
    let mut out = vec![0.0; len];
    for i in 0..a.len() { out[i] += a[i]; }
    for i in 0..b.len() { out[i] += b[i]; }
    out
}

// ─── String ↔ ArchNodeKind conversion ────────────────────────────────────────

fn parse_node_kind(kind_str: &str, params: &[Value]) -> Result<ArchNodeKind, String> {
    match kind_str {
        "linear" | "Linear" => {
            let inf = if params.len() > 0 { value_to_usize(&params[0]) } else { 64 };
            let outf = if params.len() > 1 { value_to_usize(&params[1]) } else { 64 };
            Ok(ArchNodeKind::Linear { in_features: inf, out_features: outf })
        }
        "conv2d" | "Conv2d" => {
            let inc = if params.len() > 0 { value_to_usize(&params[0]) } else { 3 };
            let outc = if params.len() > 1 { value_to_usize(&params[1]) } else { 16 };
            let k = if params.len() > 2 { value_to_usize(&params[2]) } else { 3 };
            Ok(ArchNodeKind::Conv2d { in_ch: inc, out_ch: outc, kernel: k })
        }
        "attention" | "Attention" => {
            let ed = if params.len() > 0 { value_to_usize(&params[0]) } else { 64 };
            let nh = if params.len() > 1 { value_to_usize(&params[1]) } else { 4 };
            Ok(ArchNodeKind::Attention { embed_dim: ed, num_heads: nh })
        }
        "ssm" | "SSM" => {
            let sd = if params.len() > 0 { value_to_usize(&params[0]) } else { 16 };
            let id = if params.len() > 1 { value_to_usize(&params[1]) } else { 8 };
            Ok(ArchNodeKind::SSM { state_dim: sd, input_dim: id })
        }
        "spiking" | "Spiking" => {
            let n = if params.len() > 0 { value_to_usize(&params[0]) } else { 100 };
            let t = if params.len() > 1 { value_to_f64(&params[1]) } else { 1.0 };
            Ok(ArchNodeKind::Spiking { neurons: n, threshold: t })
        }
        "moe" | "MoE" => {
            let ne = if params.len() > 0 { value_to_usize(&params[0]) } else { 4 };
            let ed = if params.len() > 1 { value_to_usize(&params[1]) } else { 64 };
            let tk = if params.len() > 2 { value_to_usize(&params[2]) } else { 2 };
            Ok(ArchNodeKind::MoE { num_experts: ne, expert_dim: ed, top_k: tk })
        }
        "ode" | "ODEBlock" => {
            let d = if params.len() > 0 { value_to_usize(&params[0]) } else { 32 };
            let t = if params.len() > 1 { value_to_f64(&params[1]) } else { 1e-3 };
            Ok(ArchNodeKind::ODEBlock { dim: d, tol: t })
        }
        "residual" | "Residual" => {
            let sg = if params.len() > 0 { value_to_usize(&params[0]) } else { 0 };
            Ok(ArchNodeKind::Residual { subgraph_id: sg })
        }
        "manifold" | "ManifoldLayer" => {
            let mk = if params.len() > 0 { value_to_string(&params[0]) } else { "sphere".to_string() };
            let d = if params.len() > 1 { value_to_usize(&params[1]) } else { 32 };
            Ok(ArchNodeKind::ManifoldLayer { kind: mk, dim: d })
        }
        "custom" | "Custom" => {
            let name = if params.len() > 0 { value_to_string(&params[0]) } else { "custom_op".to_string() };
            let mut p = HashMap::new();
            // Remaining params are key=value pairs as alternating strings and floats
            let mut i = 1;
            while i + 1 < params.len() {
                let key = value_to_string(&params[i]);
                let val = value_to_f64(&params[i + 1]);
                p.insert(key, val);
                i += 2;
            }
            Ok(ArchNodeKind::Custom { name, params: p })
        }
        _ => Err(format!("Unknown node kind: {}", kind_str)),
    }
}

fn arch_node_kind_to_string(kind: &ArchNodeKind) -> String {
    match kind {
        ArchNodeKind::Linear { .. } => "Linear".into(),
        ArchNodeKind::Conv2d { .. } => "Conv2d".into(),
        ArchNodeKind::Attention { .. } => "Attention".into(),
        ArchNodeKind::SSM { .. } => "SSM".into(),
        ArchNodeKind::Spiking { .. } => "Spiking".into(),
        ArchNodeKind::MoE { .. } => "MoE".into(),
        ArchNodeKind::ODEBlock { .. } => "ODEBlock".into(),
        ArchNodeKind::Residual { .. } => "Residual".into(),
        ArchNodeKind::ManifoldLayer { .. } => "ManifoldLayer".into(),
        ArchNodeKind::Custom { name, .. } => format!("Custom({})", name),
    }
}

fn arch_node_kind_params_json(kind: &ArchNodeKind) -> String {
    match kind {
        ArchNodeKind::Linear { in_features, out_features } =>
            format!("\"in_features\": {}, \"out_features\": {}", in_features, out_features),
        ArchNodeKind::Conv2d { in_ch, out_ch, kernel } =>
            format!("\"in_ch\": {}, \"out_ch\": {}, \"kernel\": {}", in_ch, out_ch, kernel),
        ArchNodeKind::Attention { embed_dim, num_heads } =>
            format!("\"embed_dim\": {}, \"num_heads\": {}", embed_dim, num_heads),
        ArchNodeKind::SSM { state_dim, input_dim } =>
            format!("\"state_dim\": {}, \"input_dim\": {}", state_dim, input_dim),
        ArchNodeKind::Spiking { neurons, threshold } =>
            format!("\"neurons\": {}, \"threshold\": {}", neurons, threshold),
        ArchNodeKind::MoE { num_experts, expert_dim, top_k } =>
            format!("\"num_experts\": {}, \"expert_dim\": {}, \"top_k\": {}", num_experts, expert_dim, top_k),
        ArchNodeKind::ODEBlock { dim, tol } =>
            format!("\"dim\": {}, \"tol\": {}", dim, tol),
        ArchNodeKind::Residual { subgraph_id } =>
            format!("\"subgraph_id\": {}", subgraph_id),
        ArchNodeKind::ManifoldLayer { kind, dim } =>
            format!("\"kind\": \"{}\", \"dim\": {}", kind, dim),
        ArchNodeKind::Custom { name, params } => {
            let mut s = format!("\"name\": \"{}\"", name);
            for (k, v) in params {
                s.push_str(&format!(", \"{}\": {}", k, v));
            }
            s
        }
    }
}

// ─── Global graph storage ────────────────────────────────────────────────────

use std::sync::Mutex;

static GRAPH_STORAGE: std::sync::LazyLock<Mutex<GraphStorage>> =
    std::sync::LazyLock::new(|| Mutex::new(GraphStorage::new()));

struct GraphStorage {
    graphs: HashMap<usize, ArchGraph>,
    next_id: usize,
}

impl GraphStorage {
    fn new() -> Self {
        GraphStorage { graphs: HashMap::new(), next_id: 0 }
    }

    fn insert(&mut self, graph: ArchGraph) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.graphs.insert(id, graph);
        id
    }

    fn get(&self, id: usize) -> Option<&ArchGraph> {
        self.graphs.get(&id)
    }

    fn get_mut(&mut self, id: usize) -> Option<&mut ArchGraph> {
        self.graphs.get_mut(&id)
    }

    fn remove(&mut self, id: usize) -> Option<ArchGraph> {
        self.graphs.remove(&id)
    }

    fn clone_graph(&self, id: usize) -> Option<ArchGraph> {
        self.graphs.get(&id).cloned()
    }
}

// ─── Builtin functions ───────────────────────────────────────────────────────

fn builtin_arch_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let graph = ArchGraph::new();
    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let id = storage.insert(graph);
    Ok(Value::Int(id as i128))
}

fn builtin_arch_add_node(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("arch_add_node requires at least 3 args: graph_id, kind_str, name".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let kind_str = value_to_string(&args[1]);
    let name = value_to_string(&args[2]);
    let extra_params: Vec<Value> = args[3..].to_vec();

    let kind = parse_node_kind(&kind_str, &extra_params)?;

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get_mut(graph_id).ok_or("Graph not found")?;
    let node_id = graph.add_node(kind, name);
    Ok(Value::Int(node_id as i128))
}

fn builtin_arch_connect(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("arch_connect requires at least 3 args: graph_id, from_id, to_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let from_id = value_to_usize(&args[1]);
    let to_id = value_to_usize(&args[2]);
    let shape = if args.len() > 3 { value_to_usize_vec(&args[3]) } else { vec![] };

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get_mut(graph_id).ok_or("Graph not found")?;
    graph.connect(from_id, to_id, shape);
    Ok(Value::Void)
}

fn builtin_arch_remove_node(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("arch_remove_node requires 2 args: graph_id, node_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let node_id = value_to_usize(&args[1]);

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get_mut(graph_id).ok_or("Graph not found")?;
    graph.remove_node(node_id);
    Ok(Value::Void)
}

fn builtin_arch_mutate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("arch_mutate requires at least 2 args: graph_id, mutation_str".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let mutation_str = value_to_string(&args[1]);
    let extra: Vec<Value> = args[2..].to_vec();

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get_mut(graph_id).ok_or("Graph not found")?;

    match mutation_str.as_str() {
        "add_node" => {
            let kind_str = if extra.len() > 0 { value_to_string(&extra[0]) } else { "linear".into() };
            let name = if extra.len() > 1 { value_to_string(&extra[1]) } else { "new_node".into() };
            let kind_params: Vec<Value> = extra[2..].to_vec();
            let kind = parse_node_kind(&kind_str, &kind_params)?;
            graph.add_node(kind, name);
        }
        "remove_node" => {
            let nid = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("remove_node needs node_id".into()); };
            graph.remove_node(nid);
        }
        "replace_node" => {
            let nid = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("replace_node needs node_id".into()); };
            let kind_str = if extra.len() > 1 { value_to_string(&extra[1]) } else { "linear".into() };
            let kind_params: Vec<Value> = extra[2..].to_vec();
            let kind = parse_node_kind(&kind_str, &kind_params)?;
            if let Some(node) = graph.nodes.get_mut(&nid) {
                node.kind = kind;
            } else {
                return Err(format!("Node {} not found", nid));
            }
        }
        "add_skip" => {
            let from = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("add_skip needs from".into()); };
            let to = if extra.len() > 1 { value_to_usize(&extra[1]) } else { return Err("add_skip needs to".into()); };
            graph.connect(from, to, vec![]);
        }
        "change_param" => {
            let nid = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("change_param needs node_id".into()); };
            let param = if extra.len() > 1 { value_to_string(&extra[1]) } else { return Err("change_param needs param name".into()); };
            let value = if extra.len() > 2 { value_to_f64(&extra[2]) } else { return Err("change_param needs value".into()); };
            if let Some(node) = graph.nodes.get_mut(&nid) {
                match &mut node.kind {
                    ArchNodeKind::Linear { ref mut in_features, ref mut out_features } => {
                        match param.as_str() {
                            "in_features" => *in_features = value as usize,
                            "out_features" => *out_features = value as usize,
                            _ => {}
                        }
                    }
                    ArchNodeKind::Conv2d { ref mut in_ch, ref mut out_ch, ref mut kernel } => {
                        match param.as_str() {
                            "in_ch" => *in_ch = value as usize,
                            "out_ch" => *out_ch = value as usize,
                            "kernel" => *kernel = value as usize,
                            _ => {}
                        }
                    }
                    ArchNodeKind::Attention { ref mut embed_dim, ref mut num_heads } => {
                        match param.as_str() {
                            "embed_dim" => *embed_dim = value as usize,
                            "num_heads" => *num_heads = value as usize,
                            _ => {}
                        }
                    }
                    ArchNodeKind::Spiking { ref mut neurons, ref mut threshold } => {
                        match param.as_str() {
                            "neurons" => *neurons = value as usize,
                            "threshold" => *threshold = value,
                            _ => {}
                        }
                    }
                    ArchNodeKind::SSM { ref mut state_dim, ref mut input_dim } => {
                        match param.as_str() {
                            "state_dim" => *state_dim = value as usize,
                            "input_dim" => *input_dim = value as usize,
                            _ => {}
                        }
                    }
                    ArchNodeKind::MoE { ref mut num_experts, ref mut expert_dim, ref mut top_k } => {
                        match param.as_str() {
                            "num_experts" => *num_experts = value as usize,
                            "expert_dim" => *expert_dim = value as usize,
                            "top_k" => *top_k = value as usize,
                            _ => {}
                        }
                    }
                    ArchNodeKind::ODEBlock { ref mut dim, ref mut tol } => {
                        match param.as_str() {
                            "dim" => *dim = value as usize,
                            "tol" => *tol = value,
                            _ => {}
                        }
                    }
                    ArchNodeKind::Custom { ref mut params, .. } => {
                        params.insert(param, value);
                    }
                    _ => {}
                }
            } else {
                return Err(format!("Node {} not found", nid));
            }
        }
        "split_node" => {
            let nid = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("split_node needs node_id".into()); };
            // Split: duplicate the node, insert the copy after it
            if let Some(node) = graph.nodes.get(&nid).cloned() {
                let new_id = graph.next_node_id;
                graph.next_node_id += 1;
                let mut new_node = node.clone();
                new_node.id = new_id;
                new_node.name = format!("{}_split", node.name);
                graph.nodes.insert(new_id, new_node);
                // Redirect outgoing edges from nid to go through new_id
                let outgoing: Vec<usize> = graph.edges.iter()
                    .filter(|e| e.from == nid)
                    .map(|e| e.to)
                    .collect();
                for to in &outgoing {
                    graph.edges.retain(|e| !(e.from == nid && e.to == *to));
                    graph.edges.push(ArchEdge { from: nid, to: new_id, shape: vec![] });
                    graph.edges.push(ArchEdge { from: new_id, to: *to, shape: vec![] });
                }
            }
        }
        "merge_nodes" => {
            let n1 = if extra.len() > 0 { value_to_usize(&extra[0]) } else { return Err("merge_nodes needs first node_id".into()); };
            let n2 = if extra.len() > 1 { value_to_usize(&extra[1]) } else { return Err("merge_nodes needs second node_id".into()); };
            // Merge: keep n1, redirect all edges from/to n2 to n1, remove n2
            let edges_clone = graph.edges.clone();
            for e in &edges_clone {
                if e.from == n2 && e.to != n1 {
                    graph.edges.push(ArchEdge { from: n1, to: e.to, shape: e.shape.clone() });
                }
                if e.to == n2 && e.from != n1 {
                    graph.edges.push(ArchEdge { from: e.from, to: n1, shape: e.shape.clone() });
                }
            }
            graph.edges.retain(|e| e.from != n2 && e.to != n2);
            graph.nodes.remove(&n2);
        }
        _ => return Err(format!("Unknown mutation: {}", mutation_str)),
    }
    Ok(Value::Void)
}

fn builtin_arch_crossover(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("arch_crossover requires 2 args: graph_id1, graph_id2".into());
    }
    let gid1 = value_to_usize(&args[0]);
    let gid2 = value_to_usize(&args[1]);

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let g1 = storage.clone_graph(gid1).ok_or("Graph 1 not found")?;
    let g2 = storage.clone_graph(gid2).ok_or("Graph 2 not found")?;

    let mut child = ArchGraph::new();
    let mut rng = XorShift64::new((gid1 * 31 + gid2 * 17 + 7) as u64);
    let mut id_map1: HashMap<usize, usize> = HashMap::new();
    let mut id_map2: HashMap<usize, usize> = HashMap::new();

    // Take ~half nodes from each parent
    for (&old_id, node) in &g1.nodes {
        if rng.next_f64() < 0.5 {
            let new_id = child.add_node(node.kind.clone(), node.name.clone());
            id_map1.insert(old_id, new_id);
        }
    }
    for (&old_id, node) in &g2.nodes {
        if rng.next_f64() < 0.5 {
            let new_id = child.add_node(node.kind.clone(), node.name.clone());
            id_map2.insert(old_id, new_id);
        }
    }

    // Carry over edges where both endpoints exist in the child
    for e in &g1.edges {
        if let (Some(&nf), Some(&nt)) = (id_map1.get(&e.from), id_map1.get(&e.to)) {
            child.connect(nf, nt, e.shape.clone());
        }
    }
    for e in &g2.edges {
        if let (Some(&nf), Some(&nt)) = (id_map2.get(&e.from), id_map2.get(&e.to)) {
            child.connect(nf, nt, e.shape.clone());
        }
    }

    // Merge metadata
    for (k, v) in &g1.metadata {
        child.metadata.insert(k.clone(), v.clone());
    }
    for (k, v) in &g2.metadata {
        child.metadata.insert(k.clone(), v.clone());
    }
    child.metadata.insert("origin".into(), "crossover".into());

    let id = storage.insert(child);
    Ok(Value::Int(id as i128))
}

fn builtin_arch_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_verify requires at least 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let spec_ids = if args.len() > 1 { value_to_usize_vec(&args[1]) } else { vec![] };

    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;

    let mut results = Vec::new();

    // Check 1: all edges reference existing nodes
    let mut dangling = false;
    for e in &graph.edges {
        if !graph.nodes.contains_key(&e.from) || !graph.nodes.contains_key(&e.to) {
            dangling = true;
        }
    }
    results.push(Value::String(if dangling { "FAIL: dangling edges".into() } else { "PASS: no dangling edges".into() }));

    // Check 2: no self-loops
    let has_self_loop = graph.edges.iter().any(|e| e.from == e.to);
    results.push(Value::String(if has_self_loop { "FAIL: self-loops detected".into() } else { "PASS: no self-loops".into() }));

    // Check 3: graph is connected (weakly)
    if graph.nodes.len() > 1 {
        let mut visited = std::collections::HashSet::new();
        let start = *graph.nodes.keys().next().unwrap();
        let mut stack = vec![start];
        while let Some(n) = stack.pop() {
            if visited.insert(n) {
                for e in &graph.edges {
                    if e.from == n && !visited.contains(&e.to) { stack.push(e.to); }
                    if e.to == n && !visited.contains(&e.from) { stack.push(e.from); }
                }
            }
        }
        let connected = visited.len() == graph.nodes.len();
        results.push(Value::String(if connected { "PASS: graph is connected".into() } else { "WARN: graph has disconnected components".into() }));
    } else {
        results.push(Value::String("PASS: trivially connected".into()));
    }

    // Check 4: topological sort succeeds (DAG check)
    let topo = graph.topological_sort();
    let is_dag = topo.len() == graph.nodes.len();
    results.push(Value::String(if is_dag { "PASS: graph is a DAG".into() } else { "FAIL: graph has cycles".into() }));

    // Check 5: specs validation
    for &spec_id in &spec_ids {
        if let Some(node) = graph.nodes.get(&spec_id) {
            results.push(Value::String(format!("PASS: node {} ({}) exists", spec_id, node.name)));
        } else {
            results.push(Value::String(format!("FAIL: spec node {} not found", spec_id)));
        }
    }

    Ok(Value::Array(results))
}

fn builtin_arch_flops(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_flops requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::Int(graph.estimate_flops() as i128))
}

fn builtin_arch_energy_estimate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_energy_estimate requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::Float(graph.estimate_energy()))
}

fn builtin_arch_analyze_gradient_flow(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_analyze_gradient_flow requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::String(graph.analyze_gradient_flow()))
}

fn builtin_arch_to_json(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_to_json requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::String(graph.to_json()))
}

fn builtin_arch_from_json(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_from_json requires 1 arg: json_str".into());
    }
    let json_str = value_to_string(&args[0]);

    // Simple JSON parser for our format
    let mut graph = ArchGraph::new();

    // Parse nodes
    if let Some(nodes_start) = json_str.find("\"nodes\"") {
        if let Some(arr_start) = json_str[nodes_start..].find('[') {
            let abs_start = nodes_start + arr_start;
            if let Some(arr_end) = find_matching_bracket(&json_str[abs_start..]) {
                let nodes_str = &json_str[abs_start + 1..abs_start + arr_end];
                // Parse individual node objects
                let mut depth = 0;
                let mut obj_start = None;
                for (i, c) in nodes_str.char_indices() {
                    match c {
                        '{' => {
                            if depth == 0 { obj_start = Some(i); }
                            depth += 1;
                        }
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                if let Some(start) = obj_start {
                                    let obj = &nodes_str[start..=i];
                                    parse_node_json(obj, &mut graph);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Parse edges
    if let Some(edges_start) = json_str.find("\"edges\"") {
        if let Some(arr_start) = json_str[edges_start..].find('[') {
            let abs_start = edges_start + arr_start;
            if let Some(arr_end) = find_matching_bracket(&json_str[abs_start..]) {
                let edges_str = &json_str[abs_start + 1..abs_start + arr_end];
                let mut depth = 0;
                let mut obj_start = None;
                for (i, c) in edges_str.char_indices() {
                    match c {
                        '{' => {
                            if depth == 0 { obj_start = Some(i); }
                            depth += 1;
                        }
                        '}' => {
                            depth -= 1;
                            if depth == 0 {
                                if let Some(start) = obj_start {
                                    let obj = &edges_str[start..=i];
                                    parse_edge_json(obj, &mut graph);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    let mut storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let id = storage.insert(graph);
    Ok(Value::Int(id as i128))
}

fn find_matching_bracket(s: &str) -> Option<usize> {
    let open = s.chars().next()?;
    let close = match open { '[' => ']', '{' => '}', _ => return None };
    let mut depth = 0;
    for (i, c) in s.char_indices() {
        if c == open { depth += 1; }
        if c == close {
            depth -= 1;
            if depth == 0 { return Some(i); }
        }
    }
    None
}

fn extract_json_string(obj: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let pos = obj.find(&pattern)?;
    let after = &obj[pos + pattern.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if rest.starts_with('"') {
        let end = rest[1..].find('"')?;
        Some(rest[1..1 + end].to_string())
    } else {
        None
    }
}

fn extract_json_number(obj: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let pos = obj.find(&pattern)?;
    let after = &obj[pos + pattern.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-' && c != 'e' && c != 'E' && c != '+').unwrap_or(rest.len());
    rest[..end].parse::<f64>().ok()
}

fn parse_node_json(obj: &str, graph: &mut ArchGraph) {
    let id = extract_json_number(obj, "id").unwrap_or(0.0) as usize;
    let kind_str = extract_json_string(obj, "kind").unwrap_or_default();
    let name = extract_json_string(obj, "name").unwrap_or_default();

    // Extract params based on kind
    let kind = match kind_str.as_str() {
        "Linear" => {
            let inf = extract_json_number(obj, "in_features").unwrap_or(64.0) as usize;
            let outf = extract_json_number(obj, "out_features").unwrap_or(64.0) as usize;
            ArchNodeKind::Linear { in_features: inf, out_features: outf }
        }
        "Conv2d" => {
            let inc = extract_json_number(obj, "in_ch").unwrap_or(3.0) as usize;
            let outc = extract_json_number(obj, "out_ch").unwrap_or(16.0) as usize;
            let k = extract_json_number(obj, "kernel").unwrap_or(3.0) as usize;
            ArchNodeKind::Conv2d { in_ch: inc, out_ch: outc, kernel: k }
        }
        "Attention" => {
            let ed = extract_json_number(obj, "embed_dim").unwrap_or(64.0) as usize;
            let nh = extract_json_number(obj, "num_heads").unwrap_or(4.0) as usize;
            ArchNodeKind::Attention { embed_dim: ed, num_heads: nh }
        }
        "SSM" => {
            let sd = extract_json_number(obj, "state_dim").unwrap_or(16.0) as usize;
            let id = extract_json_number(obj, "input_dim").unwrap_or(8.0) as usize;
            ArchNodeKind::SSM { state_dim: sd, input_dim: id }
        }
        "Spiking" => {
            let n = extract_json_number(obj, "neurons").unwrap_or(100.0) as usize;
            let t = extract_json_number(obj, "threshold").unwrap_or(1.0);
            ArchNodeKind::Spiking { neurons: n, threshold: t }
        }
        "MoE" => {
            let ne = extract_json_number(obj, "num_experts").unwrap_or(4.0) as usize;
            let ed = extract_json_number(obj, "expert_dim").unwrap_or(64.0) as usize;
            let tk = extract_json_number(obj, "top_k").unwrap_or(2.0) as usize;
            ArchNodeKind::MoE { num_experts: ne, expert_dim: ed, top_k: tk }
        }
        "ODEBlock" => {
            let d = extract_json_number(obj, "dim").unwrap_or(32.0) as usize;
            let t = extract_json_number(obj, "tol").unwrap_or(0.001);
            ArchNodeKind::ODEBlock { dim: d, tol: t }
        }
        "Residual" => {
            let sg = extract_json_number(obj, "subgraph_id").unwrap_or(0.0) as usize;
            ArchNodeKind::Residual { subgraph_id: sg }
        }
        "ManifoldLayer" => {
            let mk = extract_json_string(obj, "kind").unwrap_or("sphere".into());
            let d = extract_json_number(obj, "dim").unwrap_or(32.0) as usize;
            ArchNodeKind::ManifoldLayer { kind: mk, dim: d }
        }
        _ => {
            ArchNodeKind::Custom { name: kind_str.clone(), params: HashMap::new() }
        }
    };

    // Ensure graph next_node_id is ahead
    if id >= graph.next_node_id {
        graph.next_node_id = id + 1;
    }
    graph.nodes.insert(id, ArchNode { id, kind, name, specs: vec![] });
}

fn parse_edge_json(obj: &str, graph: &mut ArchGraph) {
    let from = extract_json_number(obj, "from").unwrap_or(0.0) as usize;
    let to = extract_json_number(obj, "to").unwrap_or(0.0) as usize;
    // shape parsing: simple extraction
    let shape = if let Some(pos) = obj.find("\"shape\"") {
        let after = &obj[pos..];
        if let Some(arr_start) = after.find('[') {
            if let Some(arr_end) = after[arr_start..].find(']') {
                let arr_str = &after[arr_start + 1..arr_start + arr_end];
                arr_str.split(',')
                    .filter_map(|s| s.trim().parse::<usize>().ok())
                    .collect()
            } else { vec![] }
        } else { vec![] }
    } else { vec![] };
    graph.edges.push(ArchEdge { from, to, shape });
}

fn builtin_arch_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("arch_forward requires 2 args: graph_id, input".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let input = value_to_f64_vec(&args[1]);

    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    let all_graphs = &storage.graphs;
    let output = graph.forward(&input, all_graphs);

    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_arch_node_count(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_node_count requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::Int(graph.node_count() as i128))
}

fn builtin_arch_edge_count(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_edge_count requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    Ok(Value::Int(graph.edge_count() as i128))
}

fn builtin_arch_topological_sort(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("arch_topological_sort requires 1 arg: graph_id".into());
    }
    let graph_id = value_to_usize(&args[0]);
    let storage = GRAPH_STORAGE.lock().map_err(|e| e.to_string())?;
    let graph = storage.get(graph_id).ok_or("Graph not found")?;
    let sorted = graph.topological_sort();
    Ok(Value::Array(sorted.into_iter().map(|id| Value::Int(id as i128)).collect()))
}

// ─── Registration ────────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("arch_new".into(), FnDef::Builtin(builtin_arch_new));
    env.functions.insert("arch_add_node".into(), FnDef::Builtin(builtin_arch_add_node));
    env.functions.insert("arch_connect".into(), FnDef::Builtin(builtin_arch_connect));
    env.functions.insert("arch_remove_node".into(), FnDef::Builtin(builtin_arch_remove_node));
    env.functions.insert("arch_mutate".into(), FnDef::Builtin(builtin_arch_mutate));
    env.functions.insert("arch_crossover".into(), FnDef::Builtin(builtin_arch_crossover));
    env.functions.insert("arch_verify".into(), FnDef::Builtin(builtin_arch_verify));
    env.functions.insert("arch_flops".into(), FnDef::Builtin(builtin_arch_flops));
    env.functions.insert("arch_energy_estimate".into(), FnDef::Builtin(builtin_arch_energy_estimate));
    env.functions.insert("arch_analyze_gradient_flow".into(), FnDef::Builtin(builtin_arch_analyze_gradient_flow));
    env.functions.insert("arch_to_json".into(), FnDef::Builtin(builtin_arch_to_json));
    env.functions.insert("arch_from_json".into(), FnDef::Builtin(builtin_arch_from_json));
    env.functions.insert("arch_forward".into(), FnDef::Builtin(builtin_arch_forward));
    env.functions.insert("arch_node_count".into(), FnDef::Builtin(builtin_arch_node_count));
    env.functions.insert("arch_edge_count".into(), FnDef::Builtin(builtin_arch_edge_count));
    env.functions.insert("arch_topological_sort".into(), FnDef::Builtin(builtin_arch_topological_sort));
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift() {
        let mut rng = XorShift64::new(12345);
        let a = rng.next_u64();
        let b = rng.next_u64();
        assert_ne!(a, b);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f < 1.0);
    }

    #[test]
    fn test_graph_basic() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "fc1".into());
        let n1 = g.add_node(ArchNodeKind::Linear { in_features: 8, out_features: 2 }, "fc2".into());
        g.connect(n0, n1, vec![8]);
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.edge_count(), 1);
    }

    #[test]
    fn test_topological_sort() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "a".into());
        let n1 = g.add_node(ArchNodeKind::Linear { in_features: 8, out_features: 4 }, "b".into());
        let n2 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 2 }, "c".into());
        g.connect(n0, n1, vec![]);
        g.connect(n1, n2, vec![]);
        let sorted = g.topological_sort();
        assert_eq!(sorted.len(), 3);
        // n0 must come before n1, n1 before n2
        let pos0 = sorted.iter().position(|&x| x == n0).unwrap();
        let pos1 = sorted.iter().position(|&x| x == n1).unwrap();
        let pos2 = sorted.iter().position(|&x| x == n2).unwrap();
        assert!(pos0 < pos1);
        assert!(pos1 < pos2);
    }

    #[test]
    fn test_remove_node() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "a".into());
        let n1 = g.add_node(ArchNodeKind::Linear { in_features: 8, out_features: 2 }, "b".into());
        g.connect(n0, n1, vec![]);
        g.remove_node(n0);
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.edge_count(), 0);
    }

    #[test]
    fn test_flops() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Linear { in_features: 64, out_features: 32 }, "fc".into());
        let flops = g.estimate_flops();
        assert_eq!(flops, 2 * 64 * 32);
    }

    #[test]
    fn test_energy() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Linear { in_features: 100, out_features: 100 }, "fc".into());
        let energy = g.estimate_energy();
        assert!(energy > 0.0);
    }

    #[test]
    fn test_forward_linear() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 3 }, "fc".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_forward_attention() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Attention { embed_dim: 4, num_heads: 2 }, "attn".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 0.5, -0.3, 0.7, 0.2, 0.8, -0.1, 0.4];
        let output = g.forward(&input, &graphs);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_forward_spiking() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Spiking { neurons: 4, threshold: 0.5 }, "lif".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = g.forward(&input, &graphs);
        assert!(!output.is_empty());
        // All outputs should be 0 or 1 (spikes)
        for &v in &output {
            assert!(v == 0.0 || v == 1.0);
        }
    }

    #[test]
    fn test_forward_ssm() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::SSM { state_dim: 4, input_dim: 2 }, "ssm".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 0.5, 0.3, 0.7];
        let output = g.forward(&input, &graphs);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_forward_moe() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::MoE { num_experts: 3, expert_dim: 4, top_k: 2 }, "moe".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0, 3.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_forward_chain() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "fc1".into());
        let n1 = g.add_node(ArchNodeKind::Linear { in_features: 8, out_features: 2 }, "fc2".into());
        g.connect(n0, n1, vec![]);
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_to_json_roundtrip() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "fc1".into());
        let n1 = g.add_node(ArchNodeKind::Attention { embed_dim: 8, num_heads: 2 }, "attn".into());
        g.connect(n0, n1, vec![8]);
        g.metadata.insert("version".into(), "1".into());

        let json = g.to_json();
        assert!(json.contains("Linear"));
        assert!(json.contains("Attention"));
        assert!(json.contains("fc1"));
    }

    #[test]
    fn test_gradient_flow_analysis() {
        let mut g = ArchGraph::new();
        for i in 0..25 {
            let n = g.add_node(ArchNodeKind::Linear { in_features: 64, out_features: 64 }, format!("fc{}", i));
            if i > 0 {
                g.connect(i - 1, n, vec![]);
            }
        }
        let analysis = g.analyze_gradient_flow();
        assert!(analysis.contains("Warning"));
        assert!(analysis.contains("deep network"));
    }

    #[test]
    fn test_gradient_flow_with_residual() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 64, out_features: 64 }, "fc1".into());
        let n1 = g.add_node(ArchNodeKind::Residual { subgraph_id: 0 }, "res".into());
        let n2 = g.add_node(ArchNodeKind::Linear { in_features: 64, out_features: 64 }, "fc2".into());
        g.connect(n0, n1, vec![]);
        g.connect(n1, n2, vec![]);
        let analysis = g.analyze_gradient_flow();
        assert!(analysis.contains("residual connections present"));
    }

    #[test]
    fn test_verify() {
        let mut g = ArchGraph::new();
        let n0 = g.add_node(ArchNodeKind::Linear { in_features: 4, out_features: 8 }, "fc1".into());
        let n1 = g.add_node(ArchNodeKind::Linear { in_features: 8, out_features: 2 }, "fc2".into());
        g.connect(n0, n1, vec![]);

        let mut storage = GRAPH_STORAGE.lock().unwrap();
        let gid = storage.insert(g);
        drop(storage);

        // Direct test on the graph
        let storage = GRAPH_STORAGE.lock().unwrap();
        let graph = storage.get(gid).unwrap();
        let topo = graph.topological_sort();
        assert_eq!(topo.len(), 2);
    }

    #[test]
    fn test_manifold_sphere() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::ManifoldLayer { kind: "sphere".into(), dim: 3 }, "sphere".into());
        let graphs = HashMap::new();
        let input = vec![3.0, 4.0, 0.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output.len(), 3);
        // Should be unit norm
        let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_manifold_hyperbolic() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::ManifoldLayer { kind: "hyperbolic".into(), dim: 3 }, "hyp".into());
        let graphs = HashMap::new();
        let input = vec![2.0, 2.0, 2.0];
        let output = g.forward(&input, &graphs);
        let norm: f64 = output.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm < 1.0); // must be inside Poincare ball
    }

    #[test]
    fn test_ode_block() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::ODEBlock { dim: 4, tol: 1e-3 }, "ode".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 0.0, -1.0, 0.5];
        let output = g.forward(&input, &graphs);
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_custom_node() {
        let mut params = HashMap::new();
        params.insert("scale".into(), 2.0);
        params.insert("bias".into(), 1.0);
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Custom { name: "my_op".into(), params }, "custom".into());
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0, 3.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output, vec![3.0, 5.0, 7.0]); // 2*x + 1
    }

    #[test]
    fn test_conv2d_forward() {
        let mut g = ArchGraph::new();
        g.add_node(ArchNodeKind::Conv2d { in_ch: 1, out_ch: 2, kernel: 3 }, "conv".into());
        let graphs = HashMap::new();
        let input = vec![1.0; 16]; // 1 channel, 4x4 spatial (flattened)
        let output = g.forward(&input, &graphs);
        assert!(!output.is_empty());
    }

    #[test]
    fn test_parse_node_kind() {
        let kind = parse_node_kind("linear", &[Value::Int(32), Value::Int(16)]).unwrap();
        match kind {
            ArchNodeKind::Linear { in_features, out_features } => {
                assert_eq!(in_features, 32);
                assert_eq!(out_features, 16);
            }
            _ => panic!("Expected Linear"),
        }
    }

    #[test]
    fn test_parse_node_kind_attention() {
        let kind = parse_node_kind("attention", &[Value::Int(128), Value::Int(8)]).unwrap();
        match kind {
            ArchNodeKind::Attention { embed_dim, num_heads } => {
                assert_eq!(embed_dim, 128);
                assert_eq!(num_heads, 8);
            }
            _ => panic!("Expected Attention"),
        }
    }

    #[test]
    fn test_empty_graph_forward() {
        let g = ArchGraph::new();
        let graphs = HashMap::new();
        let input = vec![1.0, 2.0];
        let output = g.forward(&input, &graphs);
        assert_eq!(output, input);
    }

    #[test]
    fn test_graph_metadata() {
        let mut g = ArchGraph::new();
        g.metadata.insert("name".into(), "test_arch".into());
        g.metadata.insert("version".into(), "1.0".into());
        let json = g.to_json();
        assert!(json.contains("test_arch"));
    }

    #[test]
    fn test_mat_vec_mul() {
        // 2x3 matrix times 3-vector
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![1.0, 1.0, 1.0];
        let result = mat_vec_mul(&mat, 2, 3, &v);
        assert_eq!(result.len(), 2);
        assert!((result[0] - 6.0).abs() < 1e-10); // 1+2+3
        assert!((result[1] - 15.0).abs() < 1e-10); // 4+5+6
    }

    #[test]
    fn test_softmax() {
        let v = vec![1.0, 2.0, 3.0];
        let s = softmax(&v);
        assert_eq!(s.len(), 3);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(s[2] > s[1]);
        assert!(s[1] > s[0]);
    }
}
