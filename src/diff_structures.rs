/// Differentiable data structures supporting backpropagation.
///
/// Implements soft/differentiable versions of hash maps, stacks, priority queues,
/// graphs, and FIFO queues using attention-based addressing.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::Mutex;

// ── Global registries ────────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref DIFF_MAPS: Mutex<HashMap<usize, DiffHashMap>> = Mutex::new(HashMap::new());
    static ref DIFF_STACKS: Mutex<HashMap<usize, DiffStack>> = Mutex::new(HashMap::new());
    static ref DIFF_GRAPHS: Mutex<HashMap<usize, DiffGraph>> = Mutex::new(HashMap::new());
    static ref DIFF_PQS: Mutex<HashMap<usize, DiffPriorityQueue>> = Mutex::new(HashMap::new());
    static ref DIFF_QUEUES: Mutex<HashMap<usize, DiffQueue>> = Mutex::new(HashMap::new());
    static ref NEXT_ID: Mutex<usize> = Mutex::new(0);
}

fn next_id() -> usize {
    let mut id = NEXT_ID.lock().unwrap();
    let v = *id;
    *id += 1;
    v
}

// ── SoftAttention helper ─────────────────────────────────────────────

pub struct SoftAttention;

impl SoftAttention {
    /// Compute softmax(query . keys) attention weights.
    pub fn attention_weights(query: &[f64], keys: &[Vec<f64>]) -> Vec<f64> {
        if keys.is_empty() {
            return vec![];
        }
        let logits: Vec<f64> = keys.iter().map(|k| dot(query, k)).collect();
        softmax(&logits)
    }

    /// Weighted sum of values using attention weights.
    pub fn weighted_sum(weights: &[f64], values: &[Vec<f64>]) -> Vec<f64> {
        if values.is_empty() {
            return vec![];
        }
        let dim = values[0].len();
        let mut result = vec![0.0; dim];
        for (w, v) in weights.iter().zip(values.iter()) {
            for (r, x) in result.iter_mut().zip(v.iter()) {
                *r += w * x;
            }
        }
        result
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        return vec![1.0 / logits.len() as f64; logits.len()];
    }
    exps.iter().map(|e| e / sum).collect()
}

// ── DiffHashMap ──────────────────────────────────────────────────────

pub struct DiffHashMap {
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
}

impl DiffHashMap {
    pub fn new() -> Self {
        Self { keys: vec![], values: vec![] }
    }

    /// Soft write: if key is similar to existing, blend; otherwise append.
    pub fn write(&mut self, key: &[f64], value: &[f64]) {
        if self.keys.is_empty() {
            self.keys.push(key.to_vec());
            self.values.push(value.to_vec());
            return;
        }
        // Use cosine similarity to decide blend vs append
        let key_norm = dot(key, key).sqrt().max(1e-12);
        let max_sim = self.keys.iter().map(|k| {
            let k_norm = dot(k, k).sqrt().max(1e-12);
            dot(key, k) / (key_norm * k_norm)
        }).fold(f64::NEG_INFINITY, f64::max);
        let weights = SoftAttention::attention_weights(key, &self.keys);
        if max_sim > 0.8 {
            // Blend into existing entries weighted by attention
            for (i, w) in weights.iter().enumerate() {
                for (v, new_v) in self.values[i].iter_mut().zip(value.iter()) {
                    *v = (1.0 - w) * *v + w * new_v;
                }
            }
        } else {
            self.keys.push(key.to_vec());
            self.values.push(value.to_vec());
        }
    }

    /// Soft read: attention-weighted sum of values.
    pub fn read(&self, query: &[f64]) -> Vec<f64> {
        if self.keys.is_empty() {
            return vec![];
        }
        let weights = SoftAttention::attention_weights(query, &self.keys);
        SoftAttention::weighted_sum(&weights, &self.values)
    }

    /// Soft erase: reduce value magnitudes for matching keys.
    pub fn erase(&mut self, query: &[f64]) {
        if self.keys.is_empty() {
            return;
        }
        let weights = SoftAttention::attention_weights(query, &self.keys);
        for (i, w) in weights.iter().enumerate() {
            for v in self.values[i].iter_mut() {
                *v *= 1.0 - w;
            }
        }
    }

    /// Backward through read: compute gradients for query, keys, values.
    /// Returns (grad_query, grad_keys, grad_values).
    pub fn backward_read(
        &self,
        query: &[f64],
        grad_output: &[f64],
    ) -> (Vec<f64>, Vec<Vec<f64>>, Vec<Vec<f64>>) {
        if self.keys.is_empty() {
            return (vec![0.0; query.len()], vec![], vec![]);
        }
        let weights = SoftAttention::attention_weights(query, &self.keys);
        let n = self.keys.len();
        let dim_v = self.values[0].len();
        let dim_k = query.len();

        // grad_values[i] = weights[i] * grad_output
        let grad_values: Vec<Vec<f64>> = (0..n)
            .map(|i| grad_output.iter().map(|g| weights[i] * g).collect())
            .collect();

        // grad_weights[i] = dot(values[i], grad_output)
        let grad_weights: Vec<f64> = (0..n)
            .map(|i| dot(&self.values[i], grad_output))
            .collect();

        // Jacobian of softmax: d_weights[i]/d_logit[j] = w[i]*(delta_ij - w[j])
        // grad_logits[i] = sum_j grad_weights[j] * w[j] * (delta_ij - w[i])
        //                 = w[i] * (grad_weights[i] - sum_j grad_weights[j] * w[j])
        let weighted_sum_gw: f64 = grad_weights.iter().zip(weights.iter()).map(|(g, w)| g * w).sum();
        let grad_logits: Vec<f64> = (0..n)
            .map(|i| weights[i] * (grad_weights[i] - weighted_sum_gw))
            .collect();

        // logit[i] = dot(query, keys[i])
        // grad_query = sum_i grad_logits[i] * keys[i]
        let mut grad_query = vec![0.0; dim_k];
        for i in 0..n {
            for (gq, k) in grad_query.iter_mut().zip(self.keys[i].iter()) {
                *gq += grad_logits[i] * k;
            }
        }

        // grad_keys[i] = grad_logits[i] * query
        let grad_keys: Vec<Vec<f64>> = (0..n)
            .map(|i| query.iter().map(|q| grad_logits[i] * q).collect())
            .collect();

        (grad_query, grad_keys, grad_values)
    }
}

// ── DiffStack ────────────────────────────────────────────────────────

pub struct DiffStack {
    pub stack: Vec<Vec<f64>>,
    pub push_weights: Vec<f64>,
}

impl DiffStack {
    pub fn new() -> Self {
        Self { stack: vec![], push_weights: vec![] }
    }

    /// Soft push: add element with given strength.
    pub fn push(&mut self, value: &[f64], strength: f64) {
        self.stack.push(value.to_vec());
        self.push_weights.push(strength.clamp(0.0, 1.0));
    }

    /// Soft pop: weighted read from top, reducing weights.
    pub fn pop(&mut self, strength: f64) -> Vec<f64> {
        if self.stack.is_empty() {
            return vec![];
        }
        let strength = strength.clamp(0.0, 1.0);
        let n = self.stack.len();
        let dim = self.stack[0].len();
        let mut result = vec![0.0; dim];
        let mut remaining = strength;

        // Read from top of stack downward
        for i in (0..n).rev() {
            let read_amount = remaining.min(self.push_weights[i]);
            for (r, v) in result.iter_mut().zip(self.stack[i].iter()) {
                *r += read_amount * v;
            }
            self.push_weights[i] -= read_amount;
            remaining -= read_amount;
            if remaining <= 1e-12 {
                break;
            }
        }
        result
    }

    /// Peek at top without modifying weights.
    pub fn peek(&self) -> Vec<f64> {
        if self.stack.is_empty() {
            return vec![];
        }
        let n = self.stack.len();
        let dim = self.stack[0].len();
        let mut result = vec![0.0; dim];
        let mut remaining = 1.0_f64;

        for i in (0..n).rev() {
            let read_amount = remaining.min(self.push_weights[i]);
            for (r, v) in result.iter_mut().zip(self.stack[i].iter()) {
                *r += read_amount * v;
            }
            remaining -= read_amount;
            if remaining <= 1e-12 {
                break;
            }
        }
        result
    }

    /// Backward through pop: distribute gradient to stack elements and their strengths.
    /// Returns Vec<(grad_value, grad_strength)> for each stack element.
    pub fn backward_pop(&self, grad: &[f64]) -> Vec<(Vec<f64>, f64)> {
        let n = self.stack.len();
        if n == 0 {
            return vec![];
        }
        // Each element contributes proportionally to its push_weight from top
        let total_w: f64 = self.push_weights.iter().sum();
        if total_w < 1e-12 {
            return self.stack.iter().map(|s| (vec![0.0; s.len()], 0.0)).collect();
        }
        (0..n)
            .map(|i| {
                let w = self.push_weights[i] / total_w;
                let grad_val: Vec<f64> = grad.iter().map(|g| g * w).collect();
                let grad_str = dot(&self.stack[i], grad) / total_w;
                (grad_val, grad_str)
            })
            .collect()
    }
}

// ── DiffPriorityQueue ────────────────────────────────────────────────

pub struct DiffPriorityQueue {
    pub elements: Vec<(Vec<f64>, f64)>, // (value, priority)
}

impl DiffPriorityQueue {
    pub fn new() -> Self {
        Self { elements: vec![] }
    }

    pub fn insert(&mut self, value: &[f64], priority: f64) {
        self.elements.push((value.to_vec(), priority));
    }

    /// Soft pop max: differentiable extraction of highest priority element.
    pub fn pop_max(&mut self) -> (Vec<f64>, f64) {
        if self.elements.is_empty() {
            return (vec![], 0.0);
        }
        let priorities: Vec<f64> = self.elements.iter().map(|(_, p)| *p).collect();
        let weights = softmax(&priorities);

        let dim = self.elements[0].0.len();
        let mut result = vec![0.0; dim];
        let mut result_priority = 0.0;

        for (i, w) in weights.iter().enumerate() {
            for (r, v) in result.iter_mut().zip(self.elements[i].0.iter()) {
                *r += w * v;
            }
            result_priority += w * self.elements[i].1;
        }

        // Reduce weights of high-priority elements (soft removal)
        for (i, w) in weights.iter().enumerate() {
            self.elements[i].1 *= 1.0 - w;
        }

        (result, result_priority)
    }

    /// Differentiable sort via Sinkhorn normalization.
    /// Returns soft permutation of priorities producing a sorted-like output.
    pub fn differentiable_sort(priorities: &[f64]) -> Vec<f64> {
        let n = priorities.len();
        if n == 0 {
            return vec![];
        }
        let temperature = 0.1;

        // Build cost matrix: C[i][j] = -(priority[i] * rank_score[j])
        // where rank_score[j] = n - j (higher is better)
        let mut log_p = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                log_p[i][j] = priorities[i] * (n - j) as f64 / temperature;
            }
        }

        // Sinkhorn iterations
        for _ in 0..20 {
            // Row normalization
            for i in 0..n {
                let row = softmax(&log_p[i]);
                log_p[i] = row.iter().map(|x| x.ln().max(-30.0)).collect();
            }
            // Column normalization
            for j in 0..n {
                let col: Vec<f64> = (0..n).map(|i| log_p[i][j]).collect();
                let col_norm = softmax(&col);
                for i in 0..n {
                    log_p[i][j] = col_norm[i].ln().max(-30.0);
                }
            }
        }

        // Final row softmax to get permutation matrix
        let perm: Vec<Vec<f64>> = (0..n).map(|i| softmax(&log_p[i])).collect();

        // Apply permutation to get sorted priorities
        // sorted[j] = sum_i perm[i][j] * priorities[i]
        let mut sorted = vec![0.0; n];
        for j in 0..n {
            for i in 0..n {
                sorted[j] += perm[i][j] * priorities[i];
            }
        }
        sorted
    }

    /// Backward through differentiable sort.
    pub fn backward_sort(priorities: &[f64], grad: &[f64]) -> Vec<f64> {
        let n = priorities.len();
        if n == 0 {
            return vec![];
        }
        // Numerical gradient via finite differences
        let eps = 1e-5;
        let mut grad_priorities = vec![0.0; n];
        for i in 0..n {
            let mut p_plus = priorities.to_vec();
            let mut p_minus = priorities.to_vec();
            p_plus[i] += eps;
            p_minus[i] -= eps;
            let s_plus = Self::differentiable_sort(&p_plus);
            let s_minus = Self::differentiable_sort(&p_minus);
            for j in 0..n {
                grad_priorities[i] += grad[j] * (s_plus[j] - s_minus[j]) / (2.0 * eps);
            }
        }
        grad_priorities
    }
}

// ── DiffGraph ────────────────────────────────────────────────────────

pub struct DiffGraph {
    pub nodes: Vec<Vec<f64>>,
    pub edges: Vec<(usize, usize, f64)>, // (from, to, weight)
}

impl DiffGraph {
    pub fn new() -> Self {
        Self { nodes: vec![], edges: vec![] }
    }

    pub fn add_node(&mut self, embedding: &[f64]) -> usize {
        let id = self.nodes.len();
        self.nodes.push(embedding.to_vec());
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push((from, to, weight));
    }

    /// GNN-style message passing: each node aggregates weighted neighbor messages.
    pub fn message_pass(&self, rounds: usize) -> Vec<Vec<f64>> {
        let n = self.nodes.len();
        if n == 0 {
            return vec![];
        }
        let dim = self.nodes[0].len();
        let mut embeddings = self.nodes.clone();

        for _ in 0..rounds {
            let mut new_embeddings = vec![vec![0.0; dim]; n];
            // Accumulate weighted messages + self
            let mut degree = vec![0.0_f64; n];
            for node_idx in 0..n {
                // Self-connection
                for (ne, e) in new_embeddings[node_idx].iter_mut().zip(embeddings[node_idx].iter()) {
                    *ne += e;
                }
                degree[node_idx] += 1.0;
            }
            for &(from, to, w) in &self.edges {
                if from < n && to < n {
                    for (ne, e) in new_embeddings[to].iter_mut().zip(embeddings[from].iter()) {
                        *ne += w * e;
                    }
                    degree[to] += w.abs();
                }
            }
            // Normalize
            for i in 0..n {
                if degree[i] > 0.0 {
                    for v in new_embeddings[i].iter_mut() {
                        *v /= degree[i];
                    }
                }
            }
            // ReLU activation
            for i in 0..n {
                for v in new_embeddings[i].iter_mut() {
                    *v = v.max(0.0);
                }
            }
            embeddings = new_embeddings;
        }
        embeddings
    }

    /// Backward through message passing.
    /// Returns (grad_nodes, grad_edge_weights).
    pub fn backward_message_pass(
        &self,
        grad_nodes: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<f64>) {
        let n = self.nodes.len();
        if n == 0 {
            return (vec![], vec![]);
        }
        let dim = self.nodes[0].len();

        // Compute forward activations for ReLU mask
        let fwd = self.message_pass(1);

        // grad through ReLU
        let mut grad_pre: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                (0..dim)
                    .map(|d| {
                        if fwd[i][d] > 0.0 {
                            grad_nodes[i][d]
                        } else {
                            0.0
                        }
                    })
                    .collect()
            })
            .collect();

        // Compute degree for normalization factor
        let mut degree = vec![1.0_f64; n]; // self-connection
        for &(_, to, w) in &self.edges {
            if to < n {
                degree[to] += w.abs();
            }
        }

        // Scale by 1/degree
        for i in 0..n {
            for v in grad_pre[i].iter_mut() {
                *v /= degree[i];
            }
        }

        // grad_nodes_input: self-connection contribution
        let grad_input: Vec<Vec<f64>> = grad_pre.clone();

        // grad_edge_weights[e] = dot(nodes[from], grad_pre[to]) / degree[to] * degree[to]
        // (degree already applied to grad_pre)
        let grad_edge_weights: Vec<f64> = self
            .edges
            .iter()
            .map(|&(from, to, _)| {
                if from < n && to < n {
                    dot(&self.nodes[from], &grad_pre[to])
                } else {
                    0.0
                }
            })
            .collect();

        (grad_input, grad_edge_weights)
    }
}

// ── DiffQueue (FIFO) ─────────────────────────────────────────────────

pub struct DiffQueue {
    pub queue: Vec<Vec<f64>>,
    pub weights: Vec<f64>,
}

impl DiffQueue {
    pub fn new() -> Self {
        Self { queue: vec![], weights: vec![] }
    }

    pub fn enqueue(&mut self, value: &[f64], strength: f64) {
        self.queue.push(value.to_vec());
        self.weights.push(strength.clamp(0.0, 1.0));
    }

    /// Soft dequeue from front.
    pub fn dequeue(&mut self, strength: f64) -> Vec<f64> {
        if self.queue.is_empty() {
            return vec![];
        }
        let strength = strength.clamp(0.0, 1.0);
        let dim = self.queue[0].len();
        let mut result = vec![0.0; dim];
        let mut remaining = strength;

        for i in 0..self.queue.len() {
            let read_amount = remaining.min(self.weights[i]);
            for (r, v) in result.iter_mut().zip(self.queue[i].iter()) {
                *r += read_amount * v;
            }
            self.weights[i] -= read_amount;
            remaining -= read_amount;
            if remaining <= 1e-12 {
                break;
            }
        }
        result
    }

    /// Backward through dequeue.
    pub fn backward_dequeue(&self, grad: &[f64]) -> Vec<(Vec<f64>, f64)> {
        let n = self.queue.len();
        if n == 0 {
            return vec![];
        }
        let total_w: f64 = self.weights.iter().sum();
        if total_w < 1e-12 {
            return self.queue.iter().map(|q| (vec![0.0; q.len()], 0.0)).collect();
        }
        (0..n)
            .map(|i| {
                let w = self.weights[i] / total_w;
                let grad_val: Vec<f64> = grad.iter().map(|g| g * w).collect();
                let grad_str = dot(&self.queue[i], grad) / total_w;
                (grad_val, grad_str)
            })
            .collect()
    }
}

// ── Interpreter builtins ─────────────────────────────────────────────

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected number, got {:?}", v)),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| value_to_f64(x)).collect(),
        _ => Err("expected array of numbers".into()),
    }
}

fn f64_vec_to_value(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

fn f64_2d_to_value(v: &[Vec<f64>]) -> Value {
    Value::Array(v.iter().map(|row| f64_vec_to_value(row)).collect())
}

// DiffHashMap builtins

fn builtin_diff_map_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let id = next_id();
    DIFF_MAPS.lock().unwrap().insert(id, DiffHashMap::new());
    Ok(Value::Int(id as i128))
}

fn builtin_diff_map_write(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("diff_map_write(id, key, value)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let key = value_to_f64_vec(&args[1])?;
    let value = value_to_f64_vec(&args[2])?;
    DIFF_MAPS.lock().unwrap().get_mut(&id).ok_or("invalid map id")?.write(&key, &value);
    Ok(Value::Void)
}

fn builtin_diff_map_read(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("diff_map_read(id, query)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let query = value_to_f64_vec(&args[1])?;
    let result = DIFF_MAPS.lock().unwrap().get(&id).ok_or("invalid map id")?.read(&query);
    Ok(f64_vec_to_value(&result))
}

// DiffStack builtins

fn builtin_diff_stack_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let id = next_id();
    DIFF_STACKS.lock().unwrap().insert(id, DiffStack::new());
    Ok(Value::Int(id as i128))
}

fn builtin_diff_stack_push(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("diff_stack_push(id, value, strength)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let value = value_to_f64_vec(&args[1])?;
    let strength = value_to_f64(&args[2])?;
    DIFF_STACKS.lock().unwrap().get_mut(&id).ok_or("invalid stack id")?.push(&value, strength);
    Ok(Value::Void)
}

fn builtin_diff_stack_pop(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("diff_stack_pop(id, strength)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let strength = value_to_f64(&args[1])?;
    let result = DIFF_STACKS.lock().unwrap().get_mut(&id).ok_or("invalid stack id")?.pop(strength);
    Ok(f64_vec_to_value(&result))
}

// DiffGraph builtins

fn builtin_diff_graph_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let id = next_id();
    DIFF_GRAPHS.lock().unwrap().insert(id, DiffGraph::new());
    Ok(Value::Int(id as i128))
}

fn builtin_diff_graph_add_node(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("diff_graph_add_node(id, embedding)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let embedding = value_to_f64_vec(&args[1])?;
    let node_id = DIFF_GRAPHS.lock().unwrap().get_mut(&id).ok_or("invalid graph id")?.add_node(&embedding);
    Ok(Value::Int(node_id as i128))
}

fn builtin_diff_graph_add_edge(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("diff_graph_add_edge(id, from, to, weight)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let from = value_to_f64(&args[1])? as usize;
    let to = value_to_f64(&args[2])? as usize;
    let weight = value_to_f64(&args[3])?;
    DIFF_GRAPHS.lock().unwrap().get_mut(&id).ok_or("invalid graph id")?.add_edge(from, to, weight);
    Ok(Value::Void)
}

fn builtin_diff_graph_message_pass(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("diff_graph_message_pass(id, rounds)".into());
    }
    let id = value_to_f64(&args[0])? as usize;
    let rounds = value_to_f64(&args[1])? as usize;
    let result = DIFF_GRAPHS.lock().unwrap().get(&id).ok_or("invalid graph id")?.message_pass(rounds);
    Ok(f64_2d_to_value(&result))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("diff_map_new".to_string(), FnDef::Builtin(builtin_diff_map_new));
    env.functions.insert("diff_map_write".to_string(), FnDef::Builtin(builtin_diff_map_write));
    env.functions.insert("diff_map_read".to_string(), FnDef::Builtin(builtin_diff_map_read));
    env.functions.insert("diff_stack_new".to_string(), FnDef::Builtin(builtin_diff_stack_new));
    env.functions.insert("diff_stack_push".to_string(), FnDef::Builtin(builtin_diff_stack_push));
    env.functions.insert("diff_stack_pop".to_string(), FnDef::Builtin(builtin_diff_stack_pop));
    env.functions.insert("diff_graph_new".to_string(), FnDef::Builtin(builtin_diff_graph_new));
    env.functions.insert("diff_graph_add_node".to_string(), FnDef::Builtin(builtin_diff_graph_add_node));
    env.functions.insert("diff_graph_add_edge".to_string(), FnDef::Builtin(builtin_diff_graph_add_edge));
    env.functions.insert("diff_graph_message_pass".to_string(), FnDef::Builtin(builtin_diff_graph_message_pass));
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax_basic() {
        let w = softmax(&[1.0, 2.0, 3.0]);
        assert_eq!(w.len(), 3);
        let sum: f64 = w.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(w[2] > w[1] && w[1] > w[0]);
    }

    #[test]
    fn test_attention_weights() {
        let query = vec![1.0, 0.0];
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let w = SoftAttention::attention_weights(&query, &keys);
        assert!(w[0] > w[1]); // query matches first key better
    }

    #[test]
    fn test_weighted_sum() {
        let weights = vec![0.7, 0.3];
        let values = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let result = SoftAttention::weighted_sum(&weights, &values);
        assert!((result[0] - 0.7).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
    }

    #[test]
    fn test_diff_hashmap_write_read() {
        let mut map = DiffHashMap::new();
        map.write(&[1.0, 0.0], &[10.0, 20.0]);
        map.write(&[0.0, 1.0], &[30.0, 40.0]);
        // Two distinct keys should both be stored
        assert_eq!(map.keys.len(), 2);
        let result = map.read(&[1.0, 0.0]);
        // Soft read returns attention-weighted blend; verify non-empty and reasonable
        assert_eq!(result.len(), 2);
        // The result should be a weighted average biased toward [10,20]
        // since query [1,0] matches first key better
        let w = SoftAttention::attention_weights(&[1.0, 0.0], &map.keys);
        assert!(w[0] > w[1], "first key should have higher weight: {:?}", w);
    }

    #[test]
    fn test_diff_hashmap_erase() {
        let mut map = DiffHashMap::new();
        map.write(&[1.0, 0.0], &[10.0, 20.0]);
        let before = map.read(&[1.0, 0.0]);
        map.erase(&[1.0, 0.0]);
        let after = map.read(&[1.0, 0.0]);
        // After erase, values should be reduced
        assert!(after[0].abs() < before[0].abs());
    }

    #[test]
    fn test_diff_hashmap_backward_read() {
        let mut map = DiffHashMap::new();
        // Directly populate to ensure two entries
        map.keys.push(vec![1.0, 0.0]);
        map.values.push(vec![10.0, 20.0]);
        map.keys.push(vec![0.0, 1.0]);
        map.values.push(vec![30.0, 40.0]);
        let (gq, gk, gv) = map.backward_read(&[1.0, 0.0], &[1.0, 1.0]);
        assert_eq!(gq.len(), 2);
        assert_eq!(gk.len(), 2);
        assert_eq!(gv.len(), 2);
        // Gradients should be non-zero
        assert!(gq.iter().any(|x| x.abs() > 1e-10));
    }

    #[test]
    fn test_diff_stack_push_pop() {
        let mut stack = DiffStack::new();
        stack.push(&[1.0, 2.0], 1.0);
        stack.push(&[3.0, 4.0], 1.0);
        let result = stack.pop(1.0);
        // Should read from top: [3.0, 4.0]
        assert!((result[0] - 3.0).abs() < 1e-10);
        assert!((result[1] - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_stack_peek() {
        let mut stack = DiffStack::new();
        stack.push(&[1.0, 2.0], 1.0);
        stack.push(&[3.0, 4.0], 1.0);
        let peek1 = stack.peek();
        let peek2 = stack.peek();
        // Peek should be idempotent
        assert!((peek1[0] - peek2[0]).abs() < 1e-10);
    }

    #[test]
    fn test_diff_stack_backward() {
        let mut stack = DiffStack::new();
        stack.push(&[1.0, 2.0], 0.5);
        stack.push(&[3.0, 4.0], 0.5);
        let grads = stack.backward_pop(&[1.0, 1.0]);
        assert_eq!(grads.len(), 2);
        // Both elements should get gradient
        assert!(grads[0].0.iter().any(|x| x.abs() > 1e-10));
        assert!(grads[1].0.iter().any(|x| x.abs() > 1e-10));
    }

    #[test]
    fn test_diff_priority_queue() {
        let mut pq = DiffPriorityQueue::new();
        pq.insert(&[1.0, 0.0], 10.0);
        pq.insert(&[0.0, 1.0], 1.0);
        let (val, pri) = pq.pop_max();
        // Should be closer to high-priority element [1.0, 0.0]
        assert!(val[0] > val[1]);
        assert!(pri > 5.0);
    }

    #[test]
    fn test_differentiable_sort() {
        let sorted = DiffPriorityQueue::differentiable_sort(&[3.0, 1.0, 2.0]);
        assert_eq!(sorted.len(), 3);
        // Highest priority should go to first position (highest rank score)
        assert!(sorted[0] > sorted[2]);
    }

    #[test]
    fn test_backward_sort() {
        let grad = DiffPriorityQueue::backward_sort(&[3.0, 1.0, 2.0], &[1.0, 1.0, 1.0]);
        assert_eq!(grad.len(), 3);
    }

    #[test]
    fn test_diff_graph_message_pass() {
        let mut g = DiffGraph::new();
        g.add_node(&[1.0, 0.0]);
        g.add_node(&[0.0, 1.0]);
        g.add_edge(0, 1, 1.0);
        let result = g.message_pass(1);
        assert_eq!(result.len(), 2);
        // Node 1 should have received message from node 0
        assert!(result[1][0] > 0.0); // got some of node 0's first dim
    }

    #[test]
    fn test_diff_graph_backward() {
        let mut g = DiffGraph::new();
        g.add_node(&[1.0, 0.0]);
        g.add_node(&[0.0, 1.0]);
        g.add_edge(0, 1, 0.5);
        let grad_nodes = vec![vec![1.0, 1.0], vec![1.0, 1.0]];
        let (gn, ge) = g.backward_message_pass(&grad_nodes);
        assert_eq!(gn.len(), 2);
        assert_eq!(ge.len(), 1);
    }

    #[test]
    fn test_diff_queue_fifo() {
        let mut q = DiffQueue::new();
        q.enqueue(&[1.0, 2.0], 1.0);
        q.enqueue(&[3.0, 4.0], 1.0);
        let result = q.dequeue(1.0);
        // Should read from front: [1.0, 2.0]
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_diff_queue_backward() {
        let mut q = DiffQueue::new();
        q.enqueue(&[1.0, 2.0], 0.5);
        q.enqueue(&[3.0, 4.0], 0.5);
        let grads = q.backward_dequeue(&[1.0, 1.0]);
        assert_eq!(grads.len(), 2);
    }

    #[test]
    fn test_empty_structures() {
        let map = DiffHashMap::new();
        assert_eq!(map.read(&[1.0]), Vec::<f64>::new());

        let mut stack = DiffStack::new();
        assert_eq!(stack.pop(1.0), Vec::<f64>::new());
        assert_eq!(stack.peek(), Vec::<f64>::new());

        let mut pq = DiffPriorityQueue::new();
        assert_eq!(pq.pop_max(), (vec![], 0.0));

        let g = DiffGraph::new();
        assert_eq!(g.message_pass(1), Vec::<Vec<f64>>::new());

        let mut q = DiffQueue::new();
        assert_eq!(q.dequeue(1.0), Vec::<f64>::new());
    }
}
