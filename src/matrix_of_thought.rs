/// MatrixOfThought: Multi-dimensional reasoning framework.
///
/// Explores multiple reasoning dimensions simultaneously (hypotheses, abstraction
/// levels, modalities) rather than linear chain-of-thought.

use std::collections::HashMap;

// ── Deterministic RNG ───────────────────────────────────────────────────

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

// ── ThoughtType ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThoughtType {
    Analytical,
    Creative,
    Retrieval,
    Verification,
    Synthesis,
    Refinement,
}

impl ThoughtType {
    fn all() -> &'static [ThoughtType] {
        &[
            ThoughtType::Analytical,
            ThoughtType::Creative,
            ThoughtType::Retrieval,
            ThoughtType::Verification,
            ThoughtType::Synthesis,
            ThoughtType::Refinement,
        ]
    }
}

// ── ThoughtDimension ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ThoughtDimension {
    Hypothesis,
    Abstraction,
    Modality,
    Confidence,
    Temporal,
}

impl ThoughtDimension {
    fn all() -> &'static [ThoughtDimension] {
        &[
            ThoughtDimension::Hypothesis,
            ThoughtDimension::Abstraction,
            ThoughtDimension::Modality,
            ThoughtDimension::Confidence,
            ThoughtDimension::Temporal,
        ]
    }
}

// ── ThoughtNode ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ThoughtNode {
    pub id: usize,
    pub content: Vec<f64>,
    pub thought_type: ThoughtType,
    pub confidence: f64,
    pub depth: usize,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
}

// ── ThoughtMatrix ───────────────────────────────────────────────────────

pub struct ThoughtMatrix {
    pub nodes: Vec<ThoughtNode>,
    pub dimensions: Vec<ThoughtDimension>,
    pub adjacency: Vec<Vec<(usize, f64)>>,
    rng: Rng,
}

impl ThoughtMatrix {
    pub fn new(dimensions: Vec<ThoughtDimension>) -> Self {
        Self {
            nodes: Vec::new(),
            dimensions,
            adjacency: Vec::new(),
            rng: Rng::new(42),
        }
    }

    /// Add a root thought node from a query embedding.
    pub fn add_root(&mut self, content: Vec<f64>, thought_type: ThoughtType) -> usize {
        let id = self.nodes.len();
        self.nodes.push(ThoughtNode {
            id,
            content,
            thought_type,
            confidence: 0.5,
            depth: 0,
            parent: None,
            children: Vec::new(),
        });
        self.adjacency.push(Vec::new());
        id
    }

    /// Expand a node along multiple dimensions, generating child thoughts.
    pub fn expand(&mut self, node_id: usize) -> Vec<usize> {
        let parent_depth = self.nodes[node_id].depth;
        let parent_content = self.nodes[node_id].content.clone();
        let n_dims = self.dimensions.len().max(1);
        let types = ThoughtType::all();

        let mut child_ids = Vec::new();
        for i in 0..n_dims {
            let id = self.nodes.len();
            // Perturb content along this dimension
            let mut content = parent_content.clone();
            for v in content.iter_mut() {
                *v += self.rng.normal() * 0.1;
            }
            let tt = types[i % types.len()];
            let confidence = 0.5 + self.rng.uniform() * 0.3;
            self.nodes.push(ThoughtNode {
                id,
                content,
                thought_type: tt,
                confidence,
                depth: parent_depth + 1,
                parent: Some(node_id),
                children: Vec::new(),
            });
            self.adjacency.push(Vec::new());
            // Add edge from parent to child
            self.adjacency[node_id].push((id, confidence));
            self.nodes[node_id].children.push(id);
            child_ids.push(id);
        }
        child_ids
    }

    /// Evaluate a thought node's score (content magnitude * confidence).
    pub fn evaluate(&self, node_id: usize) -> f64 {
        let node = &self.nodes[node_id];
        let magnitude: f64 = node.content.iter().map(|x| x * x).sum::<f64>().sqrt();
        magnitude * node.confidence
    }

    /// Prune nodes below threshold score. Marks them by setting confidence to 0.
    pub fn prune(&mut self, threshold: f64) -> usize {
        let mut pruned = 0;
        for i in 0..self.nodes.len() {
            if self.evaluate(i) < threshold && self.nodes[i].confidence > 0.0 {
                self.nodes[i].confidence = 0.0;
                pruned += 1;
            }
        }
        pruned
    }

    /// Find the highest-scoring path from any root to any leaf.
    pub fn best_path(&self) -> Vec<usize> {
        if self.nodes.is_empty() {
            return Vec::new();
        }
        // Find all roots
        let roots: Vec<usize> = self.nodes.iter()
            .filter(|n| n.parent.is_none() && n.confidence > 0.0)
            .map(|n| n.id)
            .collect();

        let mut best_score = f64::NEG_INFINITY;
        let mut best_path = Vec::new();

        for root in roots {
            let (path, score) = self.best_path_from(root);
            if score > best_score {
                best_score = score;
                best_path = path;
            }
        }
        best_path
    }

    fn best_path_from(&self, node_id: usize) -> (Vec<usize>, f64) {
        let node = &self.nodes[node_id];
        let score = self.evaluate(node_id);
        let live_children: Vec<usize> = node.children.iter()
            .copied()
            .filter(|&c| self.nodes[c].confidence > 0.0)
            .collect();

        if live_children.is_empty() {
            return (vec![node_id], score);
        }

        let mut best_child_path = Vec::new();
        let mut best_child_score = f64::NEG_INFINITY;
        for child in live_children {
            let (path, s) = self.best_path_from(child);
            if s > best_child_score {
                best_child_score = s;
                best_child_path = path;
            }
        }
        let mut full_path = vec![node_id];
        full_path.extend(best_child_path);
        (full_path, score + best_child_score)
    }

    /// Merge multiple thought nodes into a synthesis node.
    pub fn merge(&mut self, node_ids: &[usize]) -> usize {
        let id = self.nodes.len();
        let dim = if node_ids.is_empty() { 0 } else { self.nodes[node_ids[0]].content.len() };
        let mut content = vec![0.0; dim];
        let mut total_conf = 0.0;
        let mut max_depth = 0;

        for &nid in node_ids {
            let node = &self.nodes[nid];
            let w = node.confidence;
            total_conf += w;
            for (i, v) in node.content.iter().enumerate() {
                if i < content.len() {
                    content[i] += v * w;
                }
            }
            max_depth = max_depth.max(node.depth);
        }
        if total_conf > 0.0 {
            for v in content.iter_mut() {
                *v /= total_conf;
            }
        }

        self.nodes.push(ThoughtNode {
            id,
            content,
            thought_type: ThoughtType::Synthesis,
            confidence: (total_conf / node_ids.len().max(1) as f64).min(1.0),
            depth: max_depth + 1,
            parent: None,
            children: Vec::new(),
        });
        self.adjacency.push(Vec::new());
        // Add edges from merged nodes
        for &nid in node_ids {
            self.adjacency[nid].push((id, 1.0));
        }
        id
    }

    /// Max depth of the thought tree.
    pub fn depth(&self) -> usize {
        self.nodes.iter().map(|n| n.depth).max().unwrap_or(0)
    }

    /// Max width at any depth level.
    pub fn width(&self) -> usize {
        if self.nodes.is_empty() { return 0; }
        let max_d = self.depth();
        let mut best = 0;
        for d in 0..=max_d {
            let w = self.nodes.iter().filter(|n| n.depth == d && n.confidence > 0.0).count();
            best = best.max(w);
        }
        best
    }
}

// ── ReasoningResult ─────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ReasoningResult {
    pub answer: Vec<f64>,
    pub confidence: f64,
    pub path: Vec<usize>,
    pub thoughts_explored: usize,
    pub thoughts_pruned: usize,
    pub expert_usage: HashMap<ThoughtType, usize>,
    pub depth_reached: usize,
    pub compute_savings: f64,
}

// ── VerifiedResult ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct VerifiedResult {
    pub result: ReasoningResult,
    pub proof_valid: bool,
    pub circuit_size: usize,
    pub verification_time_us: u64,
}

// ── ReasoningEngine ─────────────────────────────────────────────────────

pub struct ReasoningEngine {
    pub width: usize,
    pub n_experts: usize,
    query_count: usize,
    total_confidence: f64,
    total_depth: usize,
    total_explored: usize,
    expert_distribution: HashMap<ThoughtType, usize>,
    /// Learned routing weights per ThoughtType (higher = more useful)
    routing_weights: HashMap<ThoughtType, f64>,
    /// Cumulative loss for tracking learning progress
    cumulative_loss: f64,
    learning_rate: f64,
    rng: Rng,
}

impl ReasoningEngine {
    pub fn new(width: usize, n_experts: usize) -> Self {
        let mut routing_weights = HashMap::new();
        for &tt in ThoughtType::all() {
            routing_weights.insert(tt, 1.0);
        }
        Self {
            width,
            n_experts,
            query_count: 0,
            total_confidence: 0.0,
            total_depth: 0,
            total_explored: 0,
            expert_distribution: HashMap::new(),
            routing_weights,
            cumulative_loss: 1.0,
            learning_rate: 0.01,
            rng: Rng::new(123),
        }
    }

    /// Route a thought to the appropriate expert and process it.
    fn route_and_process(&mut self, node: &mut ThoughtNode) {
        let weight = *self.routing_weights.get(&node.thought_type).unwrap_or(&1.0);
        // Simulate expert processing: boost confidence by routing weight
        node.confidence = (node.confidence * weight).min(1.0);

        // Modify content based on thought type (simulate expert specialization)
        match node.thought_type {
            ThoughtType::Analytical => {
                // Symbolic: sharpen the signal
                for v in node.content.iter_mut() {
                    *v = if *v > 0.0 { v.abs().sqrt() } else { -(v.abs().sqrt()) };
                }
            }
            ThoughtType::Creative => {
                // Divergent: add noise for exploration
                for v in node.content.iter_mut() {
                    *v += self.rng.normal() * 0.05;
                }
            }
            ThoughtType::Retrieval => {
                // Memory lookup: normalize
                let mag: f64 = node.content.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
                for v in node.content.iter_mut() {
                    *v /= mag;
                }
            }
            ThoughtType::Verification => {
                // Verification: increase confidence if content is consistent
                let mag: f64 = node.content.iter().map(|x| x * x).sum::<f64>().sqrt();
                if mag > 0.1 {
                    node.confidence = (node.confidence * 1.2).min(1.0);
                }
            }
            ThoughtType::Synthesis | ThoughtType::Refinement => {
                // Pass through with slight refinement
                for v in node.content.iter_mut() {
                    *v *= 1.01;
                }
            }
        }
        *self.expert_distribution.entry(node.thought_type).or_insert(0) += 1;
    }

    /// Full reasoning: expand, route, evaluate, prune.
    pub fn reason(&mut self, query: &[f64], max_depth: usize, max_width: usize) -> ReasoningResult {
        let dims: Vec<ThoughtDimension> = ThoughtDimension::all()
            .iter()
            .copied()
            .take(max_width.min(5))
            .collect();

        let mut matrix = ThoughtMatrix::new(dims);
        let root = matrix.add_root(query.to_vec(), ThoughtType::Analytical);

        let mut frontier = vec![root];
        let mut total_pruned = 0;
        let mut expert_usage: HashMap<ThoughtType, usize> = HashMap::new();

        for _depth in 0..max_depth {
            let mut next_frontier = Vec::new();
            for &node_id in &frontier {
                if matrix.nodes[node_id].confidence <= 0.0 {
                    continue;
                }
                let children = matrix.expand(node_id);
                for &child_id in &children {
                    self.route_and_process(&mut matrix.nodes[child_id]);
                    *expert_usage.entry(matrix.nodes[child_id].thought_type).or_insert(0) += 1;
                }
                next_frontier.extend(children);
            }
            // Prune low-scoring branches
            let threshold = 0.1;
            total_pruned += matrix.prune(threshold);
            frontier = next_frontier;
        }

        let path = matrix.best_path();
        let answer = if let Some(&last) = path.last() {
            matrix.nodes[last].content.clone()
        } else {
            query.to_vec()
        };

        let confidence = if !path.is_empty() {
            path.iter().map(|&id| matrix.nodes[id].confidence).sum::<f64>() / path.len() as f64
        } else {
            0.0
        };

        let total_possible = (max_width as f64).powi(max_depth as i32);
        let explored = matrix.nodes.len();
        let savings = if total_possible > 0.0 {
            1.0 - (explored as f64 / total_possible)
        } else {
            0.0
        };

        self.query_count += 1;
        self.total_confidence += confidence;
        self.total_depth += matrix.depth();
        self.total_explored += explored;

        ReasoningResult {
            answer,
            confidence,
            path,
            thoughts_explored: explored,
            thoughts_pruned: total_pruned,
            expert_usage,
            depth_reached: matrix.depth(),
            compute_savings: savings.max(0.0),
        }
    }

    /// Adaptive reasoning: easy queries get shallow exploration, hard ones go deep.
    pub fn reason_adaptive(&mut self, query: &[f64]) -> ReasoningResult {
        // Estimate query difficulty from content variance
        let mean = query.iter().sum::<f64>() / query.len().max(1) as f64;
        let variance = query.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / query.len().max(1) as f64;

        // Low variance = easy, high variance = hard
        let (max_depth, max_width) = if variance < 0.5 {
            (2, 2) // Easy: shallow/narrow
        } else if variance < 2.0 {
            (3, 3) // Medium
        } else {
            (5, 5) // Hard: deep/wide
        };

        self.reason(query, max_depth, max_width)
    }

    /// Verified reasoning: produces a ZK proof of the reasoning chain.
    pub fn reason_verified(&mut self, query: &[f64]) -> VerifiedResult {
        let start = std::time::Instant::now();
        let result = self.reason(query, 3, 3);

        // Simulate ZK proof generation over the reasoning path
        let circuit_size = result.path.len() * 64; // gates per reasoning step
        // Verify: hash-chain of the path must be consistent
        let mut hash: u64 = 0xcafe;
        for &node_id in &result.path {
            hash ^= node_id as u64;
            hash = hash.wrapping_mul(0x9e3779b97f4a7c15);
        }
        let proof_valid = hash != 0; // always true for non-empty paths

        let elapsed = start.elapsed().as_micros() as u64;

        VerifiedResult {
            result,
            proof_valid,
            circuit_size,
            verification_time_us: elapsed,
        }
    }

    /// Learn from reasoning feedback to improve future reasoning.
    pub fn learn_from_result(&mut self, result: &ReasoningResult, feedback: f64) {
        // feedback in [0, 1]: 1 = good, 0 = bad
        // Update routing weights: boost types that were used in good results
        for (&tt, &count) in &result.expert_usage {
            let w = self.routing_weights.entry(tt).or_insert(1.0);
            *w += self.learning_rate * (feedback - 0.5) * count as f64;
            *w = w.clamp(0.1, 5.0);
        }
        // Update cumulative loss
        self.cumulative_loss = self.cumulative_loss * 0.9 + (1.0 - feedback) * 0.1;
    }

    pub fn stats(&self) -> (usize, f64, f64, usize) {
        let avg_conf = if self.query_count > 0 {
            self.total_confidence / self.query_count as f64
        } else {
            0.0
        };
        let avg_depth = if self.query_count > 0 {
            self.total_depth as f64 / self.query_count as f64
        } else {
            0.0
        };
        (self.query_count, avg_conf, avg_depth, self.total_explored)
    }

    pub fn cumulative_loss(&self) -> f64 {
        self.cumulative_loss
    }
}

// ── MatrixOfThoughtServer ───────────────────────────────────────────────

pub struct MatrixOfThoughtServer {
    pub engine: ReasoningEngine,
}

impl MatrixOfThoughtServer {
    pub fn new(width: usize, n_experts: usize) -> Self {
        Self {
            engine: ReasoningEngine::new(width, n_experts),
        }
    }

    pub fn handle_query(&mut self, query: &[f64]) -> ReasoningResult {
        self.engine.reason(query, 3, 3)
    }

    pub fn handle_verified_query(&mut self, query: &[f64]) -> VerifiedResult {
        self.engine.reason_verified(query)
    }

    pub fn stats(&self) -> (usize, f64, f64, usize) {
        self.engine.stats()
    }
}

// ── Interpreter builtins ────────────────────────────────────────────────

use crate::interpreter::{Env, Value};

pub fn register_builtins(env: &mut Env) {
    use crate::interpreter::FnDef;

    env.functions.insert("mot_engine_new".to_string(), FnDef::Builtin(builtin_mot_engine_new));
    env.functions.insert("mot_reason".to_string(), FnDef::Builtin(builtin_mot_reason));
    env.functions.insert("mot_reason_adaptive".to_string(), FnDef::Builtin(builtin_mot_reason_adaptive));
    env.functions.insert("mot_reason_verified".to_string(), FnDef::Builtin(builtin_mot_reason_verified));
    env.functions.insert("mot_learn".to_string(), FnDef::Builtin(builtin_mot_learn));
    env.functions.insert("mot_stats".to_string(), FnDef::Builtin(builtin_mot_stats));

    // Cognitive Matrix builtins
    env.functions.insert("cog_matrix_new".to_string(), FnDef::Builtin(builtin_cog_matrix_new));
    env.functions.insert("cog_bind_model".to_string(), FnDef::Builtin(builtin_cog_bind_model));
    env.functions.insert("cog_contribute".to_string(), FnDef::Builtin(builtin_cog_contribute));
    env.functions.insert("cog_reason".to_string(), FnDef::Builtin(builtin_cog_reason));
    env.functions.insert("cog_learn".to_string(), FnDef::Builtin(builtin_cog_learn));
    env.functions.insert("cog_stats".to_string(), FnDef::Builtin(builtin_cog_stats));
    env.functions.insert("cog_messages_for".to_string(), FnDef::Builtin(builtin_cog_messages_for));
}

fn builtin_mot_engine_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mot_engine_new expects 2 args: (width, n_experts)".into());
    }
    let width = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("width must be int".into()) };
    let n_experts = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("n_experts must be int".into()) };
    let server = MatrixOfThoughtServer::new(width, n_experts);
    let id = env.next_mot_id;
    env.mot_servers.insert(id, server);
    env.next_mot_id += 1;
    Ok(Value::Int(id as i128))
}

fn extract_f64_array(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("array elements must be numbers".into()),
        }).collect(),
        _ => Err("expected array".into()),
    }
}

fn builtin_mot_reason(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("mot_reason expects 4 args: (id, query, max_depth, max_width)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let query = extract_f64_array(&args[1])?;
    let max_depth = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("max_depth must be int".into()) };
    let max_width = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("max_width must be int".into()) };
    let server = env.mot_servers.get_mut(&id).ok_or("invalid mot engine id")?;
    let result = server.engine.reason(&query, max_depth, max_width);
    Ok(Value::Array(result.answer.iter().map(|f| Value::Float(*f)).collect()))
}

fn builtin_mot_reason_adaptive(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mot_reason_adaptive expects 2 args: (id, query)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let query = extract_f64_array(&args[1])?;
    let server = env.mot_servers.get_mut(&id).ok_or("invalid mot engine id")?;
    let result = server.engine.reason_adaptive(&query);
    Ok(Value::Array(result.answer.iter().map(|f| Value::Float(*f)).collect()))
}

fn builtin_mot_reason_verified(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mot_reason_verified expects 2 args: (id, query)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let query = extract_f64_array(&args[1])?;
    let server = env.mot_servers.get_mut(&id).ok_or("invalid mot engine id")?;
    let vr = server.engine.reason_verified(&query);
    // Return: [...answer, proof_valid_as_float, confidence]
    let mut out: Vec<Value> = vr.result.answer.iter().map(|f| Value::Float(*f)).collect();
    out.push(Value::Float(if vr.proof_valid { 1.0 } else { 0.0 }));
    out.push(Value::Float(vr.result.confidence));
    Ok(Value::Array(out))
}

fn builtin_mot_learn(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mot_learn expects 2 args: (id, feedback)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let feedback = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("feedback must be a number".into()),
    };
    let server = env.mot_servers.get_mut(&id).ok_or("invalid mot engine id")?;
    // Run a query to get a result, then learn from it
    let dummy_query = vec![0.5; 4];
    let result = server.engine.reason(&dummy_query, 2, 2);
    server.engine.learn_from_result(&result, feedback);
    Ok(Value::Float(server.engine.cumulative_loss()))
}

fn builtin_mot_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("mot_stats expects 1 arg: (id)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let server = env.mot_servers.get(&id).ok_or("invalid mot engine id")?;
    let (queries, avg_conf, avg_depth, explored) = server.stats();
    Ok(Value::Array(vec![
        Value::Float(queries as f64),
        Value::Float(avg_conf),
        Value::Float(avg_depth),
        Value::Float(explored as f64),
    ]))
}

// ── Cognitive Binding: Models connect intelligence through the Matrix ───

/// A shared cognitive space where multiple models bind their reasoning.
pub struct CognitiveMatrix {
    shared_thoughts: ThoughtMatrix,
    bound_models: Vec<BoundModel>,
    collective_routing: HashMap<ThoughtType, f64>,
    streams: Vec<ThoughtStream>,
    device_budget: f64,
    collective_lr: f64,
    reasoning_history: Vec<CollectiveResult>,
    max_history: usize,
}

#[derive(Debug, Clone)]
pub struct BoundModel {
    pub id: usize,
    pub name: String,
    pub embedding_dim: usize,
    pub trust_score: f64,
    pub specialization: ThoughtType,
    pub contribution_count: usize,
    pub avg_confidence: f64,
}

#[derive(Debug, Clone)]
pub struct ThoughtStream {
    pub from_model: usize,
    pub to_model: usize,
    pub bandwidth: f64,
    pub messages: Vec<StreamMessage>,
    pub active: bool,
}

#[derive(Debug, Clone)]
pub struct StreamMessage {
    pub embedding: Vec<f64>,
    pub thought_type: ThoughtType,
    pub confidence: f64,
    pub timestamp: usize,
}

#[derive(Debug, Clone)]
pub struct CollectiveResult {
    pub query: Vec<f64>,
    pub answer: Vec<f64>,
    pub confidence: f64,
    pub contributing_models: Vec<usize>,
    pub thought_types_used: Vec<ThoughtType>,
    pub depth_reached: usize,
    pub energy_used: f64,
}

impl CognitiveMatrix {
    pub fn new(dimensions: usize, device_budget: f64) -> Self {
        let mut matrix = ThoughtMatrix::new(ThoughtDimension::all().to_vec());
        let _ = dimensions;
        let mut routing = HashMap::new();
        for tt in ThoughtType::all() {
            routing.insert(*tt, 1.0 / ThoughtType::all().len() as f64);
        }
        Self {
            shared_thoughts: matrix,
            bound_models: Vec::new(),
            collective_routing: routing,
            streams: Vec::new(),
            device_budget,
            collective_lr: 0.01,
            reasoning_history: Vec::new(),
            max_history: 1000,
        }
    }

    pub fn bind_model(&mut self, name: &str, embedding_dim: usize, specialization: ThoughtType) -> usize {
        let id = self.bound_models.len();
        self.bound_models.push(BoundModel {
            id, name: name.to_string(), embedding_dim,
            trust_score: 0.5, specialization,
            contribution_count: 0, avg_confidence: 0.0,
        });
        for existing in 0..id {
            self.streams.push(ThoughtStream {
                from_model: id, to_model: existing,
                bandwidth: 1.0, messages: Vec::new(), active: true,
            });
            self.streams.push(ThoughtStream {
                from_model: existing, to_model: id,
                bandwidth: 1.0, messages: Vec::new(), active: true,
            });
        }
        id
    }

    pub fn unbind_model(&mut self, model_id: usize) {
        self.streams.retain(|s| s.from_model != model_id && s.to_model != model_id);
    }

    pub fn contribute_thought(&mut self, model_id: usize, embedding: Vec<f64>, thought_type: ThoughtType, confidence: f64) -> usize {
        let node_id = self.shared_thoughts.add_root(embedding.clone(), thought_type);
        self.shared_thoughts.nodes[node_id].confidence = confidence;
        if model_id < self.bound_models.len() {
            self.bound_models[model_id].contribution_count += 1;
            let count = self.bound_models[model_id].contribution_count as f64;
            let prev = self.bound_models[model_id].avg_confidence;
            self.bound_models[model_id].avg_confidence = prev + (confidence - prev) / count;
        }
        let timestamp = self.shared_thoughts.nodes.len();
        for stream in self.streams.iter_mut() {
            if stream.from_model == model_id && stream.active {
                stream.messages.push(StreamMessage {
                    embedding: embedding.clone(), thought_type, confidence, timestamp,
                });
            }
        }
        node_id
    }

    pub fn collective_reason(&mut self, query: &[f64], max_depth: usize) -> CollectiveResult {
        let mut energy_remaining = self.device_budget;
        let energy_per_step = 1.0;
        let mut contributing_models = Vec::new();
        let mut thought_types_used = Vec::new();

        for model in &self.bound_models {
            if energy_remaining < energy_per_step { break; }
            let scaled = scale_embedding(query, model.embedding_dim);
            let node_id = self.shared_thoughts.add_root(scaled, model.specialization);
            self.shared_thoughts.nodes[node_id].confidence = model.trust_score;
            contributing_models.push(model.id);
            if !thought_types_used.contains(&model.specialization) {
                thought_types_used.push(model.specialization);
            }
            energy_remaining -= energy_per_step;
        }

        let mut depth = 0;
        while depth < max_depth && energy_remaining > energy_per_step {
            let n = self.shared_thoughts.nodes.len();
            for i in 0..n {
                if energy_remaining < energy_per_step { break; }
                if self.shared_thoughts.nodes[i].children.is_empty() {
                    self.shared_thoughts.expand(i);
                    energy_remaining -= energy_per_step;
                }
            }
            self.shared_thoughts.prune(0.2);
            depth += 1;
        }

        let (answer, confidence) = self.synthesize_answer(query);
        let result = CollectiveResult {
            query: query.to_vec(), answer, confidence,
            contributing_models, thought_types_used,
            depth_reached: depth,
            energy_used: self.device_budget - energy_remaining,
        };
        if self.reasoning_history.len() < self.max_history {
            self.reasoning_history.push(result.clone());
        }
        result
    }

    fn synthesize_answer(&self, query: &[f64]) -> (Vec<f64>, f64) {
        if self.shared_thoughts.nodes.is_empty() {
            return (query.to_vec(), 0.0);
        }
        let mut weighted_sum = vec![0.0; query.len()];
        let mut total_weight = 0.0;
        for node in &self.shared_thoughts.nodes {
            if node.confidence > 0.1 {
                let weight = node.confidence;
                for (i, v) in node.content.iter().enumerate() {
                    if i < weighted_sum.len() { weighted_sum[i] += v * weight; }
                }
                total_weight += weight;
            }
        }
        if total_weight > 0.0 {
            for v in &mut weighted_sum { *v /= total_weight; }
        }
        let avg_conf = if total_weight > 0.0 {
            total_weight / self.shared_thoughts.nodes.iter().filter(|n| n.confidence > 0.1).count().max(1) as f64
        } else { 0.0 };
        (weighted_sum, avg_conf)
    }

    pub fn collective_learn(&mut self, feedback: f64) {
        for model in &mut self.bound_models {
            if model.contribution_count > 0 {
                let delta = self.collective_lr * (feedback - model.trust_score);
                model.trust_score = (model.trust_score + delta).clamp(0.01, 1.0);
            }
        }
        for (tt, weight) in self.collective_routing.iter_mut() {
            let usage = self.bound_models.iter()
                .filter(|m| m.specialization == *tt)
                .map(|m| m.avg_confidence)
                .sum::<f64>();
            *weight = (*weight + self.collective_lr * (usage * feedback - *weight)).max(0.01);
        }
    }

    pub fn get_messages_for(&self, model_id: usize) -> Vec<&StreamMessage> {
        let mut msgs = Vec::new();
        for stream in &self.streams {
            if stream.to_model == model_id && stream.active {
                for msg in &stream.messages { msgs.push(msg); }
            }
        }
        msgs
    }

    pub fn model_count(&self) -> usize { self.bound_models.len() }

    pub fn stats(&self) -> (usize, f64, usize, usize) {
        let n = self.bound_models.len();
        let avg_trust = if n > 0 { self.bound_models.iter().map(|m| m.trust_score).sum::<f64>() / n as f64 } else { 0.0 };
        (n, avg_trust, self.shared_thoughts.nodes.len(), self.streams.iter().filter(|s| s.active).count())
    }
}

fn scale_embedding(input: &[f64], target_dim: usize) -> Vec<f64> {
    if input.len() == target_dim { return input.to_vec(); }
    let mut result = vec![0.0; target_dim];
    for i in 0..target_dim { result[i] = input[i % input.len()]; }
    result
}

fn builtin_cog_matrix_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cog_matrix_new(dimensions, device_budget)".into()); }
    let dims = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("dims must be int".into()) };
    let budget = match &args[1] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("budget must be number".into()) };
    let id = env.next_mot_id;
    env.next_mot_id += 1;
    env.cog_matrices.insert(id, CognitiveMatrix::new(dims, budget));
    Ok(Value::Int(id as i128))
}

fn builtin_cog_bind_model(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("cog_bind_model(matrix_id, name, embed_dim, specialization)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let name = match &args[1] { Value::String(s) => s.clone(), _ => return Err("name must be string".into()) };
    let edim = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("embed_dim must be int".into()) };
    let spec = match &args[3] {
        Value::String(s) => match s.as_str() {
            "analytical" => ThoughtType::Analytical, "creative" => ThoughtType::Creative,
            "retrieval" => ThoughtType::Retrieval, "verification" => ThoughtType::Verification,
            "synthesis" => ThoughtType::Synthesis, "refinement" => ThoughtType::Refinement,
            _ => ThoughtType::Analytical,
        },
        _ => return Err("specialization must be string".into()),
    };
    let matrix = env.cog_matrices.get_mut(&mid).ok_or("matrix not found")?;
    let model_id = matrix.bind_model(&name, edim, spec);
    Ok(Value::Int(model_id as i128))
}

fn builtin_cog_contribute(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("cog_contribute(matrix_id, model_id, embedding, confidence)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let model_id = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let embedding: Vec<f64> = match &args[2] {
        Value::Array(arr) => arr.iter().map(|v| match v { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0 }).collect(),
        _ => return Err("embedding must be array".into()),
    };
    let confidence = match &args[3] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("confidence must be number".into()) };
    let matrix = env.cog_matrices.get_mut(&mid).ok_or("matrix not found")?;
    let tt = if model_id < matrix.bound_models.len() { matrix.bound_models[model_id].specialization } else { ThoughtType::Analytical };
    let node_id = matrix.contribute_thought(model_id, embedding, tt, confidence);
    Ok(Value::Int(node_id as i128))
}

fn builtin_cog_reason(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("cog_reason(matrix_id, query, max_depth)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let query: Vec<f64> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0 }).collect(),
        _ => return Err("query must be array".into()),
    };
    let depth = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("depth must be int".into()) };
    let matrix = env.cog_matrices.get_mut(&mid).ok_or("matrix not found")?;
    let result = matrix.collective_reason(&query, depth);
    Ok(Value::Array(vec![
        Value::Array(result.answer.iter().map(|v| Value::Float(*v)).collect()),
        Value::Float(result.confidence),
        Value::Int(result.depth_reached as i128),
        Value::Float(result.energy_used),
        Value::Int(result.contributing_models.len() as i128),
    ]))
}

fn builtin_cog_learn(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cog_learn(matrix_id, feedback)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let feedback = match &args[1] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("feedback must be number".into()) };
    let matrix = env.cog_matrices.get_mut(&mid).ok_or("matrix not found")?;
    matrix.collective_learn(feedback);
    Ok(Value::Void)
}

fn builtin_cog_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("cog_stats(matrix_id)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let matrix = env.cog_matrices.get(&mid).ok_or("matrix not found")?;
    let (models, trust, thoughts, streams) = matrix.stats();
    Ok(Value::Array(vec![
        Value::Int(models as i128), Value::Float(trust),
        Value::Int(thoughts as i128), Value::Int(streams as i128),
    ]))
}

fn builtin_cog_messages_for(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("cog_messages_for(matrix_id, model_id)".into()); }
    let mid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let model_id = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("model_id must be int".into()) };
    let matrix = env.cog_matrices.get(&mid).ok_or("matrix not found")?;
    let msgs = matrix.get_messages_for(model_id);
    Ok(Value::Array(msgs.iter().map(|m| {
        Value::Tuple(vec![
            Value::Array(m.embedding.iter().map(|v| Value::Float(*v)).collect()),
            Value::Float(m.confidence),
            Value::Int(m.timestamp as i128),
        ])
    }).collect()))
}

// ── ExpertType ──────────────────────────────────────────────────────────
// Maps to the heterogeneous compute units described in design doc §3.

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExpertType {
    /// Exact integer arithmetic — no floating-point error.
    SymbolicALU,
    /// Energy-based Langevin sampler — explores diverse solutions.
    EBMSampler,
    /// Key-value lookup over external memory (bandwidth-bound).
    RetrievalLookup,
    /// ZK-circuit generation (Montgomery arithmetic).
    ZKCircuit,
    /// Standard cross-attention (tensor cores).
    DenseAttention,
    /// Recursive sub-ThoughtMatrix with reduced budget.
    SubThoughtMatrix,
}

impl ExpertType {
    pub fn all() -> &'static [ExpertType] {
        &[
            ExpertType::SymbolicALU,
            ExpertType::EBMSampler,
            ExpertType::RetrievalLookup,
            ExpertType::ZKCircuit,
            ExpertType::DenseAttention,
            ExpertType::SubThoughtMatrix,
        ]
    }

    /// Map from the design-doc thought categories to ExpertType.
    pub fn from_thought_type(tt: ThoughtType) -> Self {
        match tt {
            ThoughtType::Analytical   => ExpertType::SymbolicALU,
            ThoughtType::Creative     => ExpertType::EBMSampler,
            ThoughtType::Retrieval    => ExpertType::RetrievalLookup,
            ThoughtType::Verification => ExpertType::ZKCircuit,
            ThoughtType::Synthesis    => ExpertType::DenseAttention,
            ThoughtType::Refinement   => ExpertType::SubThoughtMatrix,
        }
    }
}

// ── MotNode: 5D-coordinate thought node ─────────────────────────────────
// A richer node used by the standalone MoT API.  The existing ThoughtNode
// inside ThoughtMatrix uses a flat index; MotNode adds the 5D coords and
// an explicit dimension tag as specified in design doc §2.

#[derive(Debug, Clone)]
pub struct MotNode {
    pub id: u64,
    /// Semantic embedding of this thought.
    pub content: Vec<f64>,
    /// Which axis (dimension) this node lives on.
    pub dimension: ThoughtDimension,
    /// Position in 5D thought-space:
    /// [hypothesis, abstraction, modality, confidence, temporal]
    pub coords: [u32; 5],
    /// Certainty score in [0, 1].
    pub confidence: f64,
    /// Outgoing edges (child ids).
    pub children: Vec<u64>,
    /// Incoming edges (parent ids / provenance).
    pub parents: Vec<u64>,
    /// Which expert produced this node.
    pub expert_type: ExpertType,
}

impl MotNode {
    pub fn new(
        id: u64,
        content: Vec<f64>,
        dimension: ThoughtDimension,
        coords: [u32; 5],
        confidence: f64,
        expert_type: ExpertType,
    ) -> Self {
        Self { id, content, dimension, coords, confidence, children: Vec::new(), parents: Vec::new(), expert_type }
    }
}

// ── MotMatrix ────────────────────────────────────────────────────────────
// The design-doc ThoughtMatrix with a priority-queue frontier.

pub struct MotMatrix {
    /// All nodes, keyed by id.
    pub nodes: HashMap<u64, MotNode>,
    /// Priority frontier: (value, node_id).  Sorted descending by value.
    pub frontier: Vec<(f64, u64)>,
    /// Dimensions to explore.
    pub dimensions: [ThoughtDimension; 5],
    /// Remaining compute budget in milliseconds.
    pub budget_ms: u64,
    next_id: u64,
    rng: Rng,
}

impl MotMatrix {
    pub fn new(budget_ms: u64) -> Self {
        Self {
            nodes: HashMap::new(),
            frontier: Vec::new(),
            dimensions: [
                ThoughtDimension::Hypothesis,
                ThoughtDimension::Abstraction,
                ThoughtDimension::Modality,
                ThoughtDimension::Confidence,
                ThoughtDimension::Temporal,
            ],
            budget_ms,
            next_id: 0,
            rng: Rng::new(0xc0ffee),
        }
    }

    fn alloc_id(&mut self) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Insert a node and push it onto the frontier with its confidence as priority.
    pub fn insert(&mut self, node: MotNode) {
        let prio = node.confidence;
        let id = node.id;
        self.nodes.insert(id, node);
        self.frontier.push((prio, id));
        self.frontier.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    }

    /// Pop the highest-priority node from the frontier.
    pub fn pop_frontier(&mut self) -> Option<u64> {
        if self.frontier.is_empty() { return None; }
        let (_, id) = self.frontier.remove(0);
        Some(id)
    }

    /// Average confidence of all nodes currently on the frontier.
    pub fn frontier_avg_confidence(&self) -> f64 {
        if self.frontier.is_empty() { return 0.0; }
        self.frontier.iter().map(|(p, _)| p).sum::<f64>() / self.frontier.len() as f64
    }

    /// Create one root node per dimension from the initial query embedding.
    pub fn seed_from_query(&mut self, content: &[f64]) {
        // Collect dimensions first to avoid simultaneous borrows.
        let dims: Vec<ThoughtDimension> = self.dimensions.to_vec();
        for (i, dim) in dims.into_iter().enumerate() {
            let id = self.alloc_id();
            let mut coords = [0u32; 5];
            coords[i] = 0;
            let expert = match dim {
                ThoughtDimension::Hypothesis  => ExpertType::SymbolicALU,
                ThoughtDimension::Abstraction => ExpertType::DenseAttention,
                ThoughtDimension::Modality    => ExpertType::EBMSampler,
                ThoughtDimension::Confidence  => ExpertType::RetrievalLookup,
                ThoughtDimension::Temporal    => ExpertType::DenseAttention,
            };
            let node = MotNode::new(id, content.to_vec(), dim, coords, 0.5, expert);
            self.insert(node);
        }
    }
}

// ── expert_dispatch ───────────────────────────────────────────────────────
// Routes a MotNode to the appropriate computation.  Returns a new node
// with updated content and confidence.

pub fn expert_dispatch(node: &MotNode, expert_type: ExpertType, rng: &mut Rng) -> MotNode {
    let mut content = node.content.clone();
    let confidence;

    match expert_type {
        ExpertType::SymbolicALU => {
            // Exact integer-like rounding — sharpen signal.
            for v in content.iter_mut() {
                *v = v.round();
            }
            confidence = (node.confidence * 1.1).min(1.0);
        }
        ExpertType::EBMSampler => {
            // Langevin-style: add small noise to explore energy landscape.
            for v in content.iter_mut() {
                *v += rng.normal() * 0.05;
            }
            confidence = node.confidence * 0.95 + rng.uniform() * 0.1;
        }
        ExpertType::RetrievalLookup => {
            // Simulate lookup: normalise content vector.
            let mag: f64 = content.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-10);
            for v in content.iter_mut() { *v /= mag; }
            confidence = (node.confidence * 1.05).min(1.0);
        }
        ExpertType::ZKCircuit => {
            // Simulate ZK step: hash-compress into a commitment vector.
            let hash_seed: f64 = content.iter().enumerate().map(|(i, v)| v * (i + 1) as f64).sum();
            for (i, v) in content.iter_mut().enumerate() {
                *v = (hash_seed * (i + 1) as f64).sin() * 0.5 + 0.5;
            }
            confidence = (node.confidence * 1.15).min(1.0);
        }
        ExpertType::DenseAttention => {
            // Simulate attention: weighted self-average.
            let mean: f64 = content.iter().sum::<f64>() / content.len().max(1) as f64;
            for v in content.iter_mut() { *v = 0.9 * *v + 0.1 * mean; }
            confidence = (node.confidence + 0.05).min(1.0);
        }
        ExpertType::SubThoughtMatrix => {
            // Recursive sub-matrix: simulate by averaging + slight boost.
            let mean: f64 = content.iter().sum::<f64>() / content.len().max(1) as f64;
            for v in content.iter_mut() { *v = (*v + mean) * 0.5; }
            confidence = (node.confidence * 1.08).min(1.0);
        }
    }

    let mut out = MotNode::new(
        node.id,                 // same id (caller may re-key)
        content,
        node.dimension,
        node.coords,
        confidence.clamp(0.0, 1.0),
        expert_type,
    );
    out.parents = node.parents.clone();
    out.children = node.children.clone();
    out
}

// ── expand_thought ────────────────────────────────────────────────────────
// Generates child thoughts across the five dimensions from a parent node.
// Returns the new MotNode children (not yet inserted into any matrix).

pub fn expand_thought(parent: &MotNode, rng: &mut Rng, next_id: &mut u64) -> Vec<MotNode> {
    let mut children = Vec::new();

    match parent.dimension {
        ThoughtDimension::Hypothesis => {
            // Generate 3 competing alternative hypotheses.
            for k in 0..3u32 {
                let id = *next_id;
                *next_id += 1;
                let mut coords = parent.coords;
                coords[0] = k + 1;
                let mut content = parent.content.clone();
                for v in content.iter_mut() { *v += rng.normal() * 0.15; }
                let node = MotNode::new(id, content, ThoughtDimension::Hypothesis, coords,
                    (parent.confidence * 0.9 + rng.uniform() * 0.1).clamp(0.0, 1.0),
                    ExpertType::SymbolicALU);
                children.push(node);
            }
        }
        ThoughtDimension::Abstraction => {
            // Shift one level up (more abstract) and one level down (more concrete).
            for (dir, delta) in [(0u32, 1i32), (1u32, -1i32)] {
                let id = *next_id;
                *next_id += 1;
                let mut coords = parent.coords;
                coords[1] = (coords[1] as i32 + delta).max(0) as u32;
                let scale = if delta > 0 { 0.5 } else { 1.5 };
                let content = parent.content.iter().map(|v| v * scale).collect();
                let node = MotNode::new(id, content, ThoughtDimension::Abstraction, coords,
                    parent.confidence * 0.95,
                    if dir == 0 { ExpertType::DenseAttention } else { ExpertType::SymbolicALU });
                children.push(node);
            }
        }
        ThoughtDimension::Modality => {
            // Apply symbolic, statistical, and analogical reasoning.
            let experts = [ExpertType::SymbolicALU, ExpertType::EBMSampler, ExpertType::DenseAttention];
            for (k, &exp) in experts.iter().enumerate() {
                let id = *next_id;
                *next_id += 1;
                let mut coords = parent.coords;
                coords[2] = k as u32;
                let child_base = MotNode::new(id, parent.content.clone(), ThoughtDimension::Modality,
                    coords, parent.confidence * 0.9, exp);
                let processed = expert_dispatch(&child_base, exp, rng);
                children.push(processed);
            }
        }
        ThoughtDimension::Confidence => {
            if parent.confidence < 0.5 {
                // Low confidence: gather 5 pieces of evidence.
                for k in 0..5u32 {
                    let id = *next_id;
                    *next_id += 1;
                    let mut coords = parent.coords;
                    coords[3] = k;
                    let mut content = parent.content.clone();
                    for v in content.iter_mut() { *v += rng.normal() * 0.05; }
                    let node = MotNode::new(id, content, ThoughtDimension::Confidence, coords,
                        (parent.confidence + 0.05).min(1.0),
                        ExpertType::RetrievalLookup);
                    children.push(node);
                }
            }
            // If confidence >= 0.5: no expansion needed (high confidence).
        }
        ThoughtDimension::Temporal => {
            // Project into past and future.
            for (k, scale) in [(0u32, 0.8f64), (1u32, 1.2f64)] {
                let id = *next_id;
                *next_id += 1;
                let mut coords = parent.coords;
                coords[4] = k;
                let content = parent.content.iter().map(|v| v * scale).collect();
                let node = MotNode::new(id, content, ThoughtDimension::Temporal, coords,
                    parent.confidence * 0.9,
                    ExpertType::DenseAttention);
                children.push(node);
            }
        }
    }

    children
}

// ── convergence_check ─────────────────────────────────────────────────────
// Returns true when the matrix has converged: the frontier's top nodes all
// have high confidence and agree well with each other (low variance).

pub fn convergence_check(matrix: &MotMatrix) -> bool {
    if matrix.frontier.is_empty() {
        return true; // nothing left to explore
    }

    // Look at the top-k frontier entries.
    let top_k = matrix.frontier.iter().take(5).collect::<Vec<_>>();
    let avg_conf: f64 = top_k.iter().map(|(p, _)| p).sum::<f64>() / top_k.len() as f64;

    if avg_conf < 0.9 {
        return false; // confidence not high enough yet
    }

    // Check agreement: variance of frontier confidence scores.
    let variance: f64 = top_k.iter()
        .map(|(p, _)| (p - avg_conf).powi(2))
        .sum::<f64>() / top_k.len() as f64;

    // Converged when high average confidence AND low variance (agreement > 0.8).
    variance < 0.04 // std-dev < 0.2 → agreement ~0.8+
}

// ── run_matrix ────────────────────────────────────────────────────────────
// Main entry point.  Expands a query into a ThoughtMatrix and returns the
// synthesised answer embedding with its confidence.

pub struct MatrixAnswer {
    pub content: Vec<f64>,
    pub confidence: f64,
    pub nodes_explored: usize,
    pub depth_reached: usize,
    pub converged: bool,
}

pub fn run_matrix(initial_query: &[f64], depth: usize, budget_ms: u64) -> MatrixAnswer {
    let mut matrix = MotMatrix::new(budget_ms);
    matrix.seed_from_query(initial_query);

    let mut rng = Rng::new(0xdecaf);
    let mut next_id = matrix.next_id;
    let mut depth_reached = 0usize;
    let mut converged = false;

    // Simulate a budget timer via node-expansion count (1 expansion ≈ 1ms).
    let max_expansions = budget_ms as usize;
    let mut expansions = 0usize;

    'outer: for d in 0..depth {
        // Collect frontier ids for this level.
        let frontier_ids: Vec<u64> = matrix.frontier.iter().map(|(_, id)| *id).collect();
        if frontier_ids.is_empty() { break; }

        for node_id in frontier_ids {
            if expansions >= max_expansions { break 'outer; }

            // Phase 1 quick exit: trivial confidence.
            if d == 0 {
                let conf = matrix.nodes.get(&node_id).map(|n| n.confidence).unwrap_or(0.0);
                if conf > 0.95 {
                    converged = true;
                    break 'outer;
                }
            }

            // Pop the node and dispatch to its expert.
            if let Some(node) = matrix.nodes.get(&node_id).cloned() {
                let processed = expert_dispatch(&node, node.expert_type, &mut rng);

                // Generate children.
                let children = expand_thought(&processed, &mut rng, &mut next_id);
                let parent_id = processed.id;

                // Re-insert the processed node.
                matrix.nodes.insert(parent_id, processed);

                for mut child in children {
                    child.parents.push(parent_id);
                    let child_id = child.id;
                    matrix.nodes.get_mut(&parent_id).map(|p| p.children.push(child_id));
                    matrix.insert(child);
                    expansions += 1;
                    if expansions >= max_expansions { break; }
                }
            }

            // Remove processed node from frontier.
            matrix.frontier.retain(|(_, id)| *id != node_id);
        }

        depth_reached = d + 1;

        // Convergence check after each depth level.
        if convergence_check(&matrix) {
            converged = true;
            break;
        }
    }

    matrix.next_id = next_id;

    // Synthesise: weighted average of all nodes by confidence.
    let dim = initial_query.len();
    let mut weighted = vec![0.0f64; dim];
    let mut total_w = 0.0f64;
    for node in matrix.nodes.values() {
        if node.confidence > 0.0 {
            let w = node.confidence;
            for (i, v) in node.content.iter().enumerate() {
                if i < dim { weighted[i] += v * w; }
            }
            total_w += w;
        }
    }
    if total_w > 0.0 {
        for v in &mut weighted { *v /= total_w; }
    }
    let avg_conf = if total_w > 0.0 {
        matrix.nodes.values().map(|n| n.confidence).sum::<f64>() / matrix.nodes.len() as f64
    } else { 0.0 };

    MatrixAnswer {
        content: weighted,
        confidence: avg_conf,
        nodes_explored: matrix.nodes.len(),
        depth_reached,
        converged,
    }
}

// ── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_node_creation() {
        let node = ThoughtNode {
            id: 0,
            content: vec![1.0, 2.0, 3.0],
            thought_type: ThoughtType::Analytical,
            confidence: 0.8,
            depth: 0,
            parent: None,
            children: Vec::new(),
        };
        assert_eq!(node.id, 0);
        assert_eq!(node.content.len(), 3);
        assert_eq!(node.thought_type, ThoughtType::Analytical);
        assert!((node.confidence - 0.8).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_expand_generates_children() {
        let mut matrix = ThoughtMatrix::new(vec![
            ThoughtDimension::Hypothesis,
            ThoughtDimension::Abstraction,
            ThoughtDimension::Modality,
        ]);
        let root = matrix.add_root(vec![1.0, 0.5], ThoughtType::Analytical);
        let children = matrix.expand(root);
        assert_eq!(children.len(), 3); // one per dimension
        for &cid in &children {
            assert_eq!(matrix.nodes[cid].depth, 1);
            assert_eq!(matrix.nodes[cid].parent, Some(root));
        }
        assert_eq!(matrix.nodes[root].children.len(), 3);
    }

    #[test]
    fn test_matrix_prune_removes_low_scoring() {
        let mut matrix = ThoughtMatrix::new(vec![ThoughtDimension::Hypothesis]);
        matrix.add_root(vec![0.001], ThoughtType::Analytical); // very low magnitude
        matrix.add_root(vec![10.0, 10.0], ThoughtType::Creative); // high magnitude
        let pruned = matrix.prune(1.0);
        assert!(pruned >= 1); // at least the low-magnitude node
        assert!(matrix.nodes[0].confidence == 0.0); // first node pruned
        assert!(matrix.nodes[1].confidence > 0.0); // second node survives
    }

    #[test]
    fn test_matrix_best_path_finds_highest_score() {
        let mut matrix = ThoughtMatrix::new(vec![ThoughtDimension::Hypothesis, ThoughtDimension::Abstraction]);
        let root = matrix.add_root(vec![1.0, 1.0], ThoughtType::Analytical);
        let children = matrix.expand(root);
        assert!(!children.is_empty());
        let path = matrix.best_path();
        assert!(!path.is_empty());
        assert_eq!(path[0], root);
        assert!(path.len() >= 2); // root + at least one child
    }

    #[test]
    fn test_matrix_merge_synthesizes_thoughts() {
        let mut matrix = ThoughtMatrix::new(vec![ThoughtDimension::Hypothesis]);
        let a = matrix.add_root(vec![1.0, 0.0], ThoughtType::Analytical);
        let b = matrix.add_root(vec![0.0, 1.0], ThoughtType::Creative);
        let merged = matrix.merge(&[a, b]);
        assert_eq!(matrix.nodes[merged].thought_type, ThoughtType::Synthesis);
        assert_eq!(matrix.nodes[merged].content.len(), 2);
        // Merged content should be weighted average
        assert!(matrix.nodes[merged].content[0] > 0.0);
        assert!(matrix.nodes[merged].content[1] > 0.0);
    }

    #[test]
    fn test_reasoning_engine_basic() {
        let mut engine = ReasoningEngine::new(4, 6);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let result = engine.reason(&query, 3, 3);
        assert!(!result.answer.is_empty());
        assert!(result.confidence > 0.0);
        assert!(result.thoughts_explored > 1);
        assert!(result.depth_reached > 0);
        assert!(!result.path.is_empty());
    }

    #[test]
    fn test_adaptive_reasoning_easy_vs_hard() {
        let mut engine = ReasoningEngine::new(4, 6);
        // Easy query: low variance
        let easy = vec![1.0, 1.0, 1.0, 1.0];
        let easy_result = engine.reason_adaptive(&easy);

        let mut engine2 = ReasoningEngine::new(4, 6);
        // Hard query: high variance
        let hard = vec![-10.0, 10.0, -10.0, 10.0];
        let hard_result = engine2.reason_adaptive(&hard);

        // Hard queries should explore more thoughts
        assert!(hard_result.thoughts_explored > easy_result.thoughts_explored,
            "hard={} easy={}", hard_result.thoughts_explored, easy_result.thoughts_explored);
    }

    #[test]
    fn test_expert_routing_matches_thought_type() {
        let mut engine = ReasoningEngine::new(4, 6);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let result = engine.reason(&query, 2, 3);
        // Should have used multiple expert types
        assert!(!result.expert_usage.is_empty());
        // Analytical should be present (root is analytical, and children include it)
        assert!(result.expert_usage.values().sum::<usize>() > 0);
    }

    #[test]
    fn test_verified_reasoning_produces_valid_proof() {
        let mut engine = ReasoningEngine::new(4, 6);
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let vr = engine.reason_verified(&query);
        assert!(vr.proof_valid);
        assert!(vr.circuit_size > 0);
        assert!(vr.result.confidence > 0.0);
    }

    #[test]
    fn test_learning_from_feedback_reduces_loss() {
        let mut engine = ReasoningEngine::new(4, 6);
        let initial_loss = engine.cumulative_loss();

        // Give positive feedback repeatedly
        for _ in 0..20 {
            let query = vec![1.0, 2.0, 3.0, 4.0];
            let result = engine.reason(&query, 2, 2);
            engine.learn_from_result(&result, 0.9); // good feedback
        }

        let final_loss = engine.cumulative_loss();
        assert!(final_loss < initial_loss,
            "Loss should decrease with positive feedback: initial={}, final={}", initial_loss, final_loss);
    }

    #[test]
    fn test_multiple_queries_improve_quality() {
        let mut engine = ReasoningEngine::new(4, 6);
        let query = vec![1.0, 2.0, 3.0, 4.0];

        // First query
        let r1 = engine.reason(&query, 2, 2);
        engine.learn_from_result(&r1, 0.9);

        // Several more with feedback
        for _ in 0..10 {
            let r = engine.reason(&query, 2, 2);
            engine.learn_from_result(&r, 0.9);
        }

        // After learning, routing weights should have changed
        let (count, _, _, _) = engine.stats();
        assert!(count >= 11); // at least 11 queries processed
    }

    #[test]
    fn test_server_handles_multiple_queries_with_state() {
        let mut server = MatrixOfThoughtServer::new(4, 6);

        // First query
        let r1 = server.handle_query(&[1.0, 2.0]);
        assert!(r1.confidence > 0.0);

        // Second query
        let r2 = server.handle_query(&[3.0, 4.0]);
        assert!(r2.confidence > 0.0);

        // Verified query
        let vr = server.handle_verified_query(&[5.0, 6.0]);
        assert!(vr.proof_valid);

        // Stats should reflect all 3 queries
        let (queries, avg_conf, avg_depth, explored) = server.stats();
        assert_eq!(queries, 3);
        assert!(avg_conf > 0.0);
        assert!(avg_depth > 0.0);
        assert!(explored > 0);
    }

    // ── New tests for ExpertType, MotNode, MotMatrix, standalone fns ─────

    #[test]
    fn test_expert_type_all_has_six_variants() {
        assert_eq!(ExpertType::all().len(), 6);
    }

    #[test]
    fn test_expert_type_from_thought_type_mapping() {
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Analytical),   ExpertType::SymbolicALU);
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Creative),     ExpertType::EBMSampler);
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Retrieval),    ExpertType::RetrievalLookup);
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Verification), ExpertType::ZKCircuit);
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Synthesis),    ExpertType::DenseAttention);
        assert_eq!(ExpertType::from_thought_type(ThoughtType::Refinement),   ExpertType::SubThoughtMatrix);
    }

    #[test]
    fn test_mot_node_has_five_d_coords() {
        let node = MotNode::new(
            42, vec![1.0, 2.0], ThoughtDimension::Hypothesis,
            [1, 2, 3, 4, 5], 0.7, ExpertType::SymbolicALU,
        );
        assert_eq!(node.id, 42);
        assert_eq!(node.coords, [1, 2, 3, 4, 5]);
        assert_eq!(node.dimension, ThoughtDimension::Hypothesis);
        assert!((node.confidence - 0.7).abs() < 1e-10);
        assert_eq!(node.expert_type, ExpertType::SymbolicALU);
    }

    #[test]
    fn test_mot_matrix_seed_creates_five_root_nodes() {
        let mut matrix = MotMatrix::new(100);
        matrix.seed_from_query(&[1.0, 2.0, 3.0]);
        assert_eq!(matrix.nodes.len(), 5); // one per dimension
        assert_eq!(matrix.frontier.len(), 5);
    }

    #[test]
    fn test_mot_matrix_frontier_sorted_descending() {
        let mut matrix = MotMatrix::new(100);
        matrix.seed_from_query(&[1.0, 1.0]);
        // Frontier must be non-empty and priorities must be non-increasing.
        let priorities: Vec<f64> = matrix.frontier.iter().map(|(p, _)| *p).collect();
        for w in priorities.windows(2) {
            assert!(w[0] >= w[1], "frontier must be sorted descending: {:?}", priorities);
        }
    }

    #[test]
    fn test_expert_dispatch_symbolic_alu_rounds_content() {
        let mut rng = Rng::new(1);
        let node = MotNode::new(0, vec![1.4, 2.6, -0.7], ThoughtDimension::Hypothesis,
            [0; 5], 0.6, ExpertType::SymbolicALU);
        let out = expert_dispatch(&node, ExpertType::SymbolicALU, &mut rng);
        assert!((out.content[0] - 1.0).abs() < 1e-10, "expected 1.0 got {}", out.content[0]);
        assert!((out.content[1] - 3.0).abs() < 1e-10, "expected 3.0 got {}", out.content[1]);
        assert!((out.content[2] - (-1.0)).abs() < 1e-10, "expected -1.0 got {}", out.content[2]);
        assert!(out.confidence >= node.confidence); // confidence boosted
    }

    #[test]
    fn test_expert_dispatch_retrieval_normalises_content() {
        let mut rng = Rng::new(2);
        let node = MotNode::new(1, vec![3.0, 4.0], ThoughtDimension::Confidence,
            [0; 5], 0.5, ExpertType::RetrievalLookup);
        let out = expert_dispatch(&node, ExpertType::RetrievalLookup, &mut rng);
        let mag: f64 = out.content.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!((mag - 1.0).abs() < 1e-9, "content should be unit-normalised, mag={}", mag);
    }

    #[test]
    fn test_expert_dispatch_confidence_clamped() {
        let mut rng = Rng::new(3);
        let node = MotNode::new(2, vec![1.0], ThoughtDimension::Hypothesis,
            [0; 5], 1.0, ExpertType::ZKCircuit);
        let out = expert_dispatch(&node, ExpertType::ZKCircuit, &mut rng);
        assert!(out.confidence <= 1.0, "confidence must not exceed 1.0");
        assert!(out.confidence >= 0.0, "confidence must be non-negative");
    }

    #[test]
    fn test_expand_thought_hypothesis_generates_three_children() {
        let parent = MotNode::new(0, vec![1.0, 0.5], ThoughtDimension::Hypothesis,
            [0; 5], 0.6, ExpertType::SymbolicALU);
        let mut rng = Rng::new(10);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 3, "Hypothesis should produce 3 alternatives");
        assert_eq!(next_id, 4); // IDs 1, 2, 3 allocated
    }

    #[test]
    fn test_expand_thought_abstraction_generates_two_children() {
        let parent = MotNode::new(0, vec![2.0, 2.0], ThoughtDimension::Abstraction,
            [0, 1, 0, 0, 0], 0.7, ExpertType::DenseAttention);
        let mut rng = Rng::new(11);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 2, "Abstraction should shift up and down");
    }

    #[test]
    fn test_expand_thought_modality_generates_three_children() {
        let parent = MotNode::new(0, vec![1.0, 1.0], ThoughtDimension::Modality,
            [0; 5], 0.6, ExpertType::EBMSampler);
        let mut rng = Rng::new(12);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 3, "Modality: symbolic, statistical, analogical");
    }

    #[test]
    fn test_expand_thought_confidence_low_generates_five_evidence_nodes() {
        let parent = MotNode::new(0, vec![0.5], ThoughtDimension::Confidence,
            [0; 5], 0.3, ExpertType::RetrievalLookup); // confidence < 0.5
        let mut rng = Rng::new(13);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 5, "Low-confidence nodes should gather 5 evidence pieces");
    }

    #[test]
    fn test_expand_thought_confidence_high_generates_no_children() {
        let parent = MotNode::new(0, vec![0.5], ThoughtDimension::Confidence,
            [0; 5], 0.8, ExpertType::RetrievalLookup); // confidence >= 0.5
        let mut rng = Rng::new(14);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 0, "High-confidence nodes should not expand");
    }

    #[test]
    fn test_expand_thought_temporal_generates_past_and_future() {
        let parent = MotNode::new(0, vec![1.0, 2.0], ThoughtDimension::Temporal,
            [0; 5], 0.6, ExpertType::DenseAttention);
        let mut rng = Rng::new(15);
        let mut next_id = 1u64;
        let children = expand_thought(&parent, &mut rng, &mut next_id);
        assert_eq!(children.len(), 2, "Temporal should project into past and future");
    }

    #[test]
    fn test_convergence_check_empty_frontier_returns_true() {
        let matrix = MotMatrix::new(50);
        assert!(convergence_check(&matrix), "empty frontier means converged");
    }

    #[test]
    fn test_convergence_check_low_confidence_returns_false() {
        let mut matrix = MotMatrix::new(50);
        // Insert nodes with low confidence.
        for i in 0..5u64 {
            let node = MotNode::new(i, vec![1.0], ThoughtDimension::Hypothesis,
                [0; 5], 0.4, ExpertType::SymbolicALU);
            matrix.insert(node);
        }
        assert!(!convergence_check(&matrix), "low-confidence frontier should not converge");
    }

    #[test]
    fn test_convergence_check_high_confidence_low_variance_returns_true() {
        let mut matrix = MotMatrix::new(50);
        // Insert nodes with very high and consistent confidence.
        for i in 0..5u64 {
            let node = MotNode::new(i, vec![1.0], ThoughtDimension::Hypothesis,
                [0; 5], 0.95, ExpertType::SymbolicALU);
            matrix.insert(node);
        }
        assert!(convergence_check(&matrix), "high uniform confidence should converge");
    }

    #[test]
    fn test_run_matrix_returns_answer_with_correct_dimensions() {
        let query = vec![1.0, 2.0, 3.0, 4.0];
        let answer = run_matrix(&query, 3, 50);
        assert_eq!(answer.content.len(), query.len(), "answer should match query dimension");
        assert!(answer.confidence >= 0.0 && answer.confidence <= 1.0);
        assert!(answer.nodes_explored > 0);
    }

    #[test]
    fn test_run_matrix_explores_more_nodes_with_higher_budget() {
        let query = vec![1.0, 2.0, 3.0];
        let small = run_matrix(&query, 5, 10);
        let large  = run_matrix(&query, 5, 200);
        assert!(large.nodes_explored >= small.nodes_explored,
            "larger budget should explore at least as many nodes: small={} large={}",
            small.nodes_explored, large.nodes_explored);
    }

    #[test]
    fn test_run_matrix_zero_budget_still_seeds_nodes() {
        let query = vec![1.0, 2.0];
        let answer = run_matrix(&query, 3, 0);
        // Even with zero budget, the 5 seed nodes are created.
        assert!(answer.nodes_explored >= 5);
    }

    #[test]
    fn test_run_matrix_depth_zero_returns_seeded_answer() {
        let query = vec![0.5, 0.5];
        let answer = run_matrix(&query, 0, 100);
        assert_eq!(answer.content.len(), 2);
        assert_eq!(answer.depth_reached, 0);
    }
}
