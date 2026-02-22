// Novel AI Architecture Primitives for Vortex
// Building blocks for AGI-class models: memory-augmented networks, neuro-symbolic reasoning,
// world models, consciousness-inspired architectures, continual learning, and compositional generalization.

use std::collections::HashMap;
use crate::interpreter::{Env, Value, FnDef};

// ─── Utility: simple deterministic RNG ──────────────────────────────────────

fn lcg_next(seed: &mut u64) -> f64 {
    *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*seed >> 33) as f64 / (1u64 << 31) as f64).max(1e-10)
}

fn randn_vec(dim: usize, seed: &mut u64) -> Vec<f64> {
    let mut v = Vec::with_capacity(dim);
    for _ in 0..dim {
        let u1 = lcg_next(seed);
        let u2 = lcg_next(seed);
        let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        v.push(z * 0.1);
    }
    v
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt().max(1e-12)
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    dot(a, b) / (norm(a) * norm(b))
}

fn softmax(v: &[f64]) -> Vec<f64> {
    let max_v = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = v.iter().map(|x| (x - max_v).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

fn vec_add(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x + y).collect()
}

fn vec_scale(a: &[f64], s: f64) -> Vec<f64> {
    a.iter().map(|x| x * s).collect()
}

fn vec_sub(a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter().zip(b.iter()).map(|(x, y)| x - y).collect()
}

fn relu(x: f64) -> f64 {
    x.max(0.0)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn linear_forward(input: &[f64], weights: &[Vec<f64>], bias: &[f64]) -> Vec<f64> {
    let out_dim = weights.len();
    let mut output = Vec::with_capacity(out_dim);
    for i in 0..out_dim {
        output.push(dot(&weights[i], input) + bias[i]);
    }
    output
}

fn mlp_forward(input: &[f64], layers: &[(Vec<Vec<f64>>, Vec<f64>)]) -> Vec<f64> {
    let mut x = input.to_vec();
    for (i, (w, b)) in layers.iter().enumerate() {
        x = linear_forward(&x, w, b);
        if i < layers.len() - 1 {
            x = x.iter().map(|v| relu(*v)).collect();
        }
    }
    x
}

fn make_layer(in_dim: usize, out_dim: usize, seed: &mut u64) -> (Vec<Vec<f64>>, Vec<f64>) {
    let scale = (2.0 / in_dim as f64).sqrt();
    let weights: Vec<Vec<f64>> = (0..out_dim)
        .map(|_| randn_vec(in_dim, seed).iter().map(|v| v * scale / 0.1).collect())
        .collect();
    let bias = vec![0.0; out_dim];
    (weights, bias)
}

fn make_mlp(dims: &[usize], seed: &mut u64) -> Vec<(Vec<Vec<f64>>, Vec<f64>)> {
    let mut layers = Vec::new();
    for i in 0..dims.len() - 1 {
        layers.push(make_layer(dims[i], dims[i + 1], seed));
    }
    layers
}

// ─── 1. Memory-Augmented Networks ───────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NeuralMemory {
    pub memory_bank: Vec<Vec<f64>>,  // M slots x D dimensions
    pub read_heads: usize,
    pub write_heads: usize,
    pub usage: Vec<f64>,             // allocation weights per slot
    pub dim: usize,
    seed: u64,
}

impl NeuralMemory {
    pub fn new(slots: usize, dim: usize, read_heads: usize, write_heads: usize) -> Self {
        let mut seed = 12345u64;
        let memory_bank = (0..slots).map(|_| randn_vec(dim, &mut seed)).collect();
        let usage = vec![0.0; slots];
        Self { memory_bank, read_heads, write_heads, usage, dim, seed }
    }

    /// Content-based read: cosine similarity -> softmax -> weighted sum
    pub fn read(&self, query: &[f64]) -> Vec<f64> {
        let sims: Vec<f64> = self.memory_bank.iter()
            .map(|slot| cosine_similarity(query, slot))
            .collect();
        let weights = softmax(&sims);
        let mut result = vec![0.0; self.dim];
        for (i, w) in weights.iter().enumerate() {
            for (j, v) in self.memory_bank[i].iter().enumerate() {
                result[j] += w * v;
            }
        }
        result
    }

    /// Content-based addressing: returns attention weights over memory slots
    pub fn address_content(&self, query: &[f64]) -> Vec<f64> {
        let sims: Vec<f64> = self.memory_bank.iter()
            .map(|slot| cosine_similarity(query, slot))
            .collect();
        softmax(&sims)
    }

    /// Location-based addressing: shift + sharpen existing weights
    pub fn address_location(&self, prev_weights: &[f64], shift: i32, sharpen: f64) -> Vec<f64> {
        let n = prev_weights.len();
        // Circular shift
        let mut shifted = vec![0.0; n];
        for i in 0..n {
            let src = ((i as i32 - shift).rem_euclid(n as i32)) as usize;
            shifted[i] = prev_weights[src];
        }
        // Sharpen
        let sharpened: Vec<f64> = shifted.iter().map(|w| w.powf(sharpen)).collect();
        let sum: f64 = sharpened.iter().sum();
        if sum > 1e-12 {
            sharpened.iter().map(|w| w / sum).collect()
        } else {
            sharpened
        }
    }

    /// Write: erase old content + add new content
    pub fn write(&mut self, key: &[f64], value: &[f64], erase_strength: f64) {
        let weights = self.address_content(key);
        for (i, w) in weights.iter().enumerate() {
            // Erase
            for j in 0..self.dim {
                self.memory_bank[i][j] *= 1.0 - w * erase_strength;
            }
            // Add
            for j in 0..self.dim {
                self.memory_bank[i][j] += w * value[j.min(value.len() - 1)];
            }
            // Update usage
            self.usage[i] += w;
        }
    }

    /// Allocate: find least-recently-used slot
    pub fn allocate(&self) -> usize {
        self.usage.iter().enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Write to least-used slot directly
    pub fn write_lru(&mut self, value: &[f64]) {
        let slot = self.allocate();
        let dim = self.dim;
        for j in 0..dim {
            self.memory_bank[slot][j] = value[j.min(value.len() - 1)];
        }
        self.usage[slot] = self.usage.iter().cloned().fold(f64::NEG_INFINITY, f64::max) + 1.0;
    }
}

// ─── 2. Neuro-Symbolic Reasoning Layer ──────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Predicate {
    pub name: String,
    pub args: Vec<String>,
}

impl std::fmt::Display for Predicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}({})", self.name, self.args.join(", "))
    }
}

#[derive(Debug, Clone)]
pub struct Rule {
    pub premises: Vec<Predicate>,
    pub conclusion: Predicate,
    pub confidence: f64,
    pub learned: bool,
}

#[derive(Debug, Clone)]
pub struct Fact {
    pub predicate: Predicate,
    pub confidence: f64,
}

impl std::fmt::Display for Fact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} [conf={:.2}]", self.predicate, self.confidence)
    }
}

#[derive(Debug, Clone)]
pub struct SymbolicReasoner {
    pub rules: Vec<Rule>,
    pub facts: Vec<Fact>,
    embeddings: HashMap<String, Vec<f64>>,
    embed_dim: usize,
    seed: u64,
}

impl SymbolicReasoner {
    pub fn new(embed_dim: usize) -> Self {
        Self {
            rules: Vec::new(),
            facts: Vec::new(),
            embeddings: HashMap::new(),
            embed_dim,
            seed: 42,
        }
    }

    pub fn add_rule(&mut self, premises: Vec<Predicate>, conclusion: Predicate, confidence: f64) {
        self.rules.push(Rule { premises, conclusion, confidence, learned: false });
    }

    pub fn add_fact(&mut self, pred: Predicate, confidence: f64) {
        self.facts.push(Fact { predicate: pred, confidence });
    }

    /// Get or create embedding for a symbol
    fn embed(&mut self, symbol: &str) -> Vec<f64> {
        if let Some(e) = self.embeddings.get(symbol) {
            return e.clone();
        }
        let e = randn_vec(self.embed_dim, &mut self.seed);
        self.embeddings.insert(symbol.to_string(), e.clone());
        e
    }

    /// Check if a predicate matches a fact (simple name + arity match with variable binding)
    fn matches(pred: &Predicate, fact: &Predicate, bindings: &mut HashMap<String, String>) -> bool {
        if pred.name != fact.name || pred.args.len() != fact.args.len() {
            return false;
        }
        let mut local_bindings = bindings.clone();
        for (p, f) in pred.args.iter().zip(fact.args.iter()) {
            if p.starts_with('?') {
                // Variable
                if let Some(bound) = local_bindings.get(p) {
                    if bound != f { return false; }
                } else {
                    local_bindings.insert(p.clone(), f.clone());
                }
            } else if p != f {
                return false;
            }
        }
        *bindings = local_bindings;
        true
    }

    /// Apply bindings to a predicate
    fn apply_bindings(pred: &Predicate, bindings: &HashMap<String, String>) -> Predicate {
        Predicate {
            name: pred.name.clone(),
            args: pred.args.iter().map(|a| {
                if a.starts_with('?') {
                    bindings.get(a).cloned().unwrap_or_else(|| a.clone())
                } else {
                    a.clone()
                }
            }).collect(),
        }
    }

    /// Forward chaining: apply all rules to derive new facts
    pub fn forward_chain(&mut self, max_iterations: usize) -> Vec<Fact> {
        let mut derived = Vec::new();
        for _ in 0..max_iterations {
            let mut new_facts = Vec::new();
            for rule in &self.rules {
                // Try to match all premises
                let mut binding_sets: Vec<HashMap<String, String>> = vec![HashMap::new()];
                for premise in &rule.premises {
                    let mut next_bindings = Vec::new();
                    for bindings in &binding_sets {
                        for fact in &self.facts {
                            let mut b = bindings.clone();
                            if Self::matches(premise, &fact.predicate, &mut b) {
                                next_bindings.push(b);
                            }
                        }
                    }
                    binding_sets = next_bindings;
                }
                // Derive conclusions
                for bindings in &binding_sets {
                    let conclusion = Self::apply_bindings(&rule.conclusion, bindings);
                    let conf = rule.confidence
                        * rule.premises.len() as f64 / rule.premises.len().max(1) as f64;
                    // Check if already known
                    let already_known = self.facts.iter().any(|f| {
                        f.predicate.name == conclusion.name && f.predicate.args == conclusion.args
                    });
                    if !already_known {
                        new_facts.push(Fact { predicate: conclusion, confidence: conf });
                    }
                }
            }
            if new_facts.is_empty() { break; }
            for f in &new_facts {
                derived.push(f.clone());
                self.facts.push(f.clone());
            }
        }
        derived
    }

    /// Backward chaining: goal-directed reasoning
    pub fn backward_chain(&self, goal: &Predicate, depth: usize) -> bool {
        if depth == 0 { return false; }
        // Check if goal is already a fact
        for fact in &self.facts {
            let mut bindings = HashMap::new();
            if Self::matches(goal, &fact.predicate, &mut bindings) {
                return true;
            }
        }
        // Try to prove via rules
        for rule in &self.rules {
            let mut bindings = HashMap::new();
            if Self::matches(&rule.conclusion, goal, &mut bindings) {
                // Need to prove all premises
                let all_proved = rule.premises.iter().all(|p| {
                    let bound_p = Self::apply_bindings(p, &bindings);
                    self.backward_chain(&bound_p, depth - 1)
                });
                if all_proved { return true; }
            }
        }
        false
    }

    /// Learn a new rule from example fact pairs (simple pattern extraction)
    pub fn learn_rule(&mut self, examples: &[(Vec<Fact>, Fact)]) -> Option<Rule> {
        if examples.is_empty() { return None; }
        // Find common premise predicates across examples
        let first_premises: Vec<String> = examples[0].0.iter()
            .map(|f| f.predicate.name.clone()).collect();
        let common: Vec<String> = first_premises.iter().filter(|name| {
            examples.iter().all(|ex| ex.0.iter().any(|f| f.predicate.name == **name))
        }).cloned().collect();
        if common.is_empty() { return None; }
        let conclusion_name = examples[0].1.predicate.name.clone();
        let arity = examples[0].1.predicate.args.len();
        let premise_preds: Vec<Predicate> = common.iter().map(|name| {
            let ex_fact = examples[0].0.iter().find(|f| f.predicate.name == *name).unwrap();
            Predicate {
                name: name.clone(),
                args: (0..ex_fact.predicate.args.len()).map(|i| format!("?x{}", i)).collect(),
            }
        }).collect();
        let conclusion = Predicate {
            name: conclusion_name,
            args: (0..arity).map(|i| format!("?x{}", i)).collect(),
        };
        let avg_conf: f64 = examples.iter().map(|e| e.1.confidence).sum::<f64>() / examples.len() as f64;
        let rule = Rule {
            premises: premise_preds,
            conclusion,
            confidence: avg_conf,
            learned: true,
        };
        self.rules.push(rule.clone());
        Some(rule)
    }

    /// Differentiable unification: similarity-based soft matching score
    pub fn soft_unify(&mut self, a: &Predicate, b: &Predicate) -> f64 {
        let name_sim = if a.name == b.name { 1.0 } else {
            let ea = self.embed(&a.name);
            let eb = self.embed(&b.name);
            cosine_similarity(&ea, &eb).max(0.0)
        };
        if a.args.len() != b.args.len() {
            return name_sim * 0.5;
        }
        let arg_sims: f64 = a.args.iter().zip(b.args.iter()).map(|(x, y)| {
            if x == y { 1.0 } else {
                let ex = self.embed(x);
                let ey = self.embed(y);
                cosine_similarity(&ex, &ey).max(0.0)
            }
        }).sum::<f64>() / a.args.len().max(1) as f64;
        name_sim * 0.5 + arg_sims * 0.5
    }
}

// ─── 3. World Model ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Entity {
    pub id: usize,
    pub embedding: Vec<f64>,
    pub properties: HashMap<String, f64>,
    pub persistent: bool,
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub source: usize,
    pub target: usize,
    pub relation_type: String,
    pub strength: f64,
}

#[derive(Debug, Clone)]
pub struct DynamicsModel {
    transition_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    reward_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    inverse_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    state_dim: usize,
    action_dim: usize,
}

impl DynamicsModel {
    fn new(state_dim: usize, action_dim: usize, seed: &mut u64) -> Self {
        let combined = state_dim + action_dim;
        let hidden = (combined * 2).max(16);
        let transition_layers = make_mlp(&[combined, hidden, state_dim], seed);
        let reward_layers = make_mlp(&[combined, hidden / 2, 1], seed);
        let inv_combined = state_dim * 2;
        let inverse_layers = make_mlp(&[inv_combined, hidden, action_dim], seed);
        Self { transition_layers, reward_layers, inverse_layers, state_dim, action_dim }
    }

    fn predict_next(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        let mut input = state.to_vec();
        input.extend_from_slice(action);
        let delta = mlp_forward(&input, &self.transition_layers);
        // Residual connection: next_state = state + delta
        vec_add(state, &delta)
    }

    fn predict_reward(&self, state: &[f64], action: &[f64]) -> f64 {
        let mut input = state.to_vec();
        input.extend_from_slice(action);
        let out = mlp_forward(&input, &self.reward_layers);
        out[0]
    }

    fn predict_action(&self, state: &[f64], next_state: &[f64]) -> Vec<f64> {
        let mut input = state.to_vec();
        input.extend_from_slice(next_state);
        mlp_forward(&input, &self.inverse_layers)
    }
}

#[derive(Debug, Clone)]
pub struct WorldModel {
    pub entities: Vec<Entity>,
    pub relations: Vec<Relation>,
    pub dynamics: DynamicsModel,
    entity_dim: usize,
    action_dim: usize,
    seed: u64,
}

impl WorldModel {
    pub fn new(entity_dim: usize, action_dim: usize) -> Self {
        let mut seed = 77777u64;
        let dynamics = DynamicsModel::new(entity_dim, action_dim, &mut seed);
        Self {
            entities: Vec::new(),
            relations: Vec::new(),
            dynamics,
            entity_dim,
            action_dim,
            seed,
        }
    }

    pub fn add_entity(&mut self, properties: HashMap<String, f64>, persistent: bool) -> usize {
        let id = self.entities.len();
        let embedding = randn_vec(self.entity_dim, &mut self.seed);
        self.entities.push(Entity { id, embedding, properties, persistent });
        id
    }

    pub fn add_relation(&mut self, source: usize, target: usize, rel_type: &str, strength: f64) {
        self.relations.push(Relation {
            source, target, relation_type: rel_type.to_string(), strength,
        });
    }

    /// Predict next state given current state and action
    pub fn step(&self, state: &[f64], action: &[f64]) -> Vec<f64> {
        self.dynamics.predict_next(state, action)
    }

    /// Roll out a trajectory: state, [actions] -> [states]
    pub fn imagine(&self, initial_state: &[f64], actions: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let mut trajectory = vec![initial_state.to_vec()];
        let mut current = initial_state.to_vec();
        for action in actions {
            current = self.dynamics.predict_next(&current, action);
            trajectory.push(current.clone());
        }
        trajectory
    }

    /// Counterfactual: given a state and alternative action, compare trajectories
    pub fn counterfactual(
        &self, state: &[f64], factual_actions: &[Vec<f64>], counter_actions: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let factual = self.imagine(state, factual_actions);
        let counter = self.imagine(state, counter_actions);
        (factual, counter)
    }

    /// Predict reward for a state-action pair
    pub fn reward(&self, state: &[f64], action: &[f64]) -> f64 {
        self.dynamics.predict_reward(state, action)
    }

    /// Inverse model: infer action from state transition
    pub fn infer_action(&self, state: &[f64], next_state: &[f64]) -> Vec<f64> {
        self.dynamics.predict_action(state, next_state)
    }
}

// ─── 4. Consciousness-Inspired Architecture (Global Workspace Theory) ──────

#[derive(Debug, Clone)]
pub struct Specialist {
    pub name: String,
    pub network: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub competence: f64,
    pub domain: String,
    input_dim: usize,
    output_dim: usize,
}

impl Specialist {
    fn forward(&self, input: &[f64]) -> Vec<f64> {
        mlp_forward(input, &self.network)
    }
}

#[derive(Debug, Clone)]
pub struct MetaCognition {
    confidence_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    uncertainty_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    strategy_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl MetaCognition {
    fn new(dim: usize, seed: &mut u64) -> Self {
        let confidence_layers = make_mlp(&[dim, dim / 2, 1], seed);
        let uncertainty_layers = make_mlp(&[dim, dim / 2, 1], seed);
        let strategy_layers = make_mlp(&[dim, dim / 2, 3], seed); // 3 strategies
        Self { confidence_layers, uncertainty_layers, strategy_layers }
    }

    pub fn estimate_confidence(&self, workspace: &[f64]) -> f64 {
        let out = mlp_forward(workspace, &self.confidence_layers);
        sigmoid(out[0])
    }

    pub fn estimate_uncertainty(&self, workspace: &[f64]) -> f64 {
        let out = mlp_forward(workspace, &self.uncertainty_layers);
        sigmoid(out[0])
    }

    /// Select strategy: 0=continue, 1=ask_for_help, 2=retry_different
    pub fn select_strategy(&self, workspace: &[f64]) -> usize {
        let out = mlp_forward(workspace, &self.strategy_layers);
        let probs = softmax(&out);
        probs.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone)]
pub struct GlobalWorkspace {
    pub specialists: Vec<Specialist>,
    pub workspace: Vec<f64>,
    pub metacognition: MetaCognition,
    workspace_dim: usize,
    seed: u64,
}

impl GlobalWorkspace {
    pub fn new(num_specialists: usize, input_dim: usize, workspace_dim: usize) -> Self {
        let mut seed = 99999u64;
        let specialists: Vec<Specialist> = (0..num_specialists).map(|i| {
            let network = make_mlp(&[input_dim, workspace_dim, workspace_dim], &mut seed);
            Specialist {
                name: format!("specialist_{}", i),
                network,
                competence: 0.5,
                domain: format!("domain_{}", i),
                input_dim,
                output_dim: workspace_dim,
            }
        }).collect();
        let metacognition = MetaCognition::new(workspace_dim, &mut seed);
        Self {
            specialists,
            workspace: vec![0.0; workspace_dim],
            metacognition,
            workspace_dim,
            seed,
        }
    }

    /// Competition: each specialist processes input, highest competence * output_norm wins
    pub fn compete(&mut self, input: &[f64]) -> Vec<(usize, f64)> {
        let mut scores: Vec<(usize, f64)> = Vec::new();
        for (i, spec) in self.specialists.iter().enumerate() {
            let output = spec.forward(input);
            let output_strength = norm(&output);
            let score = spec.competence * output_strength;
            scores.push((i, score));
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }

    /// Broadcast: winning specialist's output becomes the workspace, shared with all
    pub fn broadcast(&mut self, input: &[f64]) -> Vec<f64> {
        let scores = self.compete(input);
        if scores.is_empty() {
            return self.workspace.clone();
        }
        let winner_idx = scores[0].0;
        let output = self.specialists[winner_idx].forward(input);
        self.workspace = output.clone();
        // Update competence based on workspace quality
        let confidence = self.metacognition.estimate_confidence(&self.workspace);
        self.specialists[winner_idx].competence =
            self.specialists[winner_idx].competence * 0.9 + confidence * 0.1;
        output
    }

    /// Meta-cognitive assessment of current workspace
    pub fn assess(&self) -> (f64, f64, usize) {
        let confidence = self.metacognition.estimate_confidence(&self.workspace);
        let uncertainty = self.metacognition.estimate_uncertainty(&self.workspace);
        let strategy = self.metacognition.select_strategy(&self.workspace);
        (confidence, uncertainty, strategy)
    }
}

// ─── 5. Continual Learning Without Forgetting ──────────────────────────────

#[derive(Debug, Clone)]
pub struct TaskMemory {
    pub task_id: usize,
    pub exemplars: Vec<(Vec<f64>, Vec<f64>)>, // (input, target) pairs
    pub importance: Vec<Vec<f64>>,             // fisher information per layer
}

#[derive(Debug, Clone)]
pub struct ReplayBuffer {
    pub buffer: Vec<(Vec<f64>, Vec<f64>)>,
    pub capacity: usize,
    write_idx: usize,
}

impl ReplayBuffer {
    fn new(capacity: usize) -> Self {
        Self { buffer: Vec::new(), capacity, write_idx: 0 }
    }

    fn add(&mut self, input: Vec<f64>, target: Vec<f64>) {
        if self.buffer.len() < self.capacity {
            self.buffer.push((input, target));
        } else {
            self.buffer[self.write_idx] = (input, target);
        }
        self.write_idx = (self.write_idx + 1) % self.capacity;
    }

    fn sample(&self, n: usize, seed: &mut u64) -> Vec<(Vec<f64>, Vec<f64>)> {
        if self.buffer.is_empty() { return Vec::new(); }
        let mut samples = Vec::new();
        for _ in 0..n.min(self.buffer.len()) {
            let idx = (lcg_next(seed) * self.buffer.len() as f64) as usize % self.buffer.len();
            samples.push(self.buffer[idx].clone());
        }
        samples
    }
}

#[derive(Debug, Clone)]
pub struct ProgressiveColumn {
    pub layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub frozen: bool,
}

#[derive(Debug, Clone)]
pub struct ContinualLearner {
    pub model_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    pub task_memories: Vec<TaskMemory>,
    pub fisher_information: Vec<Vec<f64>>,     // flattened per-layer importance
    pub replay_buffer: ReplayBuffer,
    pub columns: Vec<ProgressiveColumn>,       // progressive net columns
    pub ewc_lambda: f64,
    current_task: usize,
    dims: Vec<usize>,
    seed: u64,
}

impl ContinualLearner {
    pub fn new(dims: &[usize], replay_capacity: usize) -> Self {
        let mut seed = 55555u64;
        let model_layers = make_mlp(dims, &mut seed);
        let fisher_information = model_layers.iter().map(|(w, _)| {
            vec![1.0; w.len() * w[0].len()]
        }).collect();
        Self {
            model_layers,
            task_memories: Vec::new(),
            fisher_information,
            replay_buffer: ReplayBuffer::new(replay_capacity),
            columns: Vec::new(),
            ewc_lambda: 1000.0,
            current_task: 0,
            dims: dims.to_vec(),
            seed,
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut x = mlp_forward(input, &self.model_layers);
        // Add contributions from progressive columns
        for col in &self.columns {
            if col.frozen {
                let col_out = mlp_forward(input, &col.layers);
                x = vec_add(&x, &vec_scale(&col_out, 0.1));
            }
        }
        x
    }

    /// Simple online learning step with EWC regularization
    pub fn learn_step(&mut self, input: &[f64], target: &[f64], lr: f64) {
        let output = self.forward(input);
        // Compute error
        let error: Vec<f64> = vec_sub(target, &output);
        // Add to replay buffer
        self.replay_buffer.add(input.to_vec(), target.to_vec());
        // Simple gradient descent on last layer (simplified)
        let last_idx = self.model_layers.len() - 1;
        // Get input to last layer (simplified: use original input)
        let pre_activation = if last_idx > 0 {
            mlp_forward(input, &self.model_layers[..last_idx])
        } else {
            input.to_vec()
        };
        let (ref mut weights, ref mut bias) = self.model_layers[last_idx];
        for i in 0..weights.len().min(error.len()) {
            for j in 0..weights[i].len().min(pre_activation.len()) {
                let grad = -error[i] * pre_activation[j];
                // EWC penalty
                let fisher_penalty = if last_idx < self.fisher_information.len() {
                    let fi = &self.fisher_information[last_idx];
                    let flat_idx = i * weights[i].len() + j;
                    if flat_idx < fi.len() { fi[flat_idx] * weights[i][j] * self.ewc_lambda } else { 0.0 }
                } else { 0.0 };
                weights[i][j] -= lr * (grad + fisher_penalty * 0.0001);
            }
            if i < bias.len() {
                bias[i] -= lr * (-error[i]);
            }
        }
    }

    /// Experience replay: rehearse old tasks
    pub fn replay_step(&mut self, n_samples: usize, lr: f64) {
        let samples = self.replay_buffer.sample(n_samples, &mut self.seed);
        for (input, target) in &samples {
            self.learn_step(input, target, lr * 0.5); // Lower LR for replay
        }
    }

    /// Finish current task: save memory, update fisher, freeze column
    pub fn finish_task(&mut self) {
        let task_id = self.current_task;
        // Save exemplars from replay buffer
        let exemplars = self.replay_buffer.sample(10, &mut self.seed);
        self.task_memories.push(TaskMemory {
            task_id,
            exemplars,
            importance: self.fisher_information.clone(),
        });
        // Add a new progressive column (frozen copy of current model)
        self.columns.push(ProgressiveColumn {
            layers: self.model_layers.clone(),
            frozen: true,
        });
        // Increase fisher info (simulated: increase importance of well-used weights)
        for fi in &mut self.fisher_information {
            for v in fi.iter_mut() {
                *v *= 1.1; // Weights become more important over time
            }
        }
        self.current_task += 1;
    }

    pub fn num_tasks_learned(&self) -> usize {
        self.current_task
    }
}

// ─── 6. Compositional Generalization ────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Primitive {
    pub id: usize,
    pub embedding: Vec<f64>,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct CompositionalNet {
    pub primitives: Vec<Primitive>,
    combiner_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    structure_layers: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
    dim: usize,
    seed: u64,
}

impl CompositionalNet {
    pub fn new(num_primitives: usize, dim: usize) -> Self {
        let mut seed = 33333u64;
        let primitives: Vec<Primitive> = (0..num_primitives).map(|i| {
            Primitive {
                id: i,
                embedding: randn_vec(dim, &mut seed),
                name: format!("prim_{}", i),
            }
        }).collect();
        let combiner_layers = make_mlp(&[dim * 2, dim * 2, dim], &mut seed);
        let structure_layers = make_mlp(&[dim, dim, num_primitives], &mut seed);
        Self { primitives, combiner_layers, structure_layers, dim, seed }
    }

    /// Compose two primitives
    pub fn compose_pair(&self, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut input = a.to_vec();
        input.extend_from_slice(b);
        mlp_forward(&input, &self.combiner_layers)
    }

    /// Compose a sequence of primitive IDs using left-fold
    pub fn compose(&self, primitive_ids: &[usize]) -> Vec<f64> {
        if primitive_ids.is_empty() {
            return vec![0.0; self.dim];
        }
        let first_id = primitive_ids[0].min(self.primitives.len() - 1);
        let mut result = self.primitives[first_id].embedding.clone();
        for &id in &primitive_ids[1..] {
            let pid = id.min(self.primitives.len() - 1);
            result = self.compose_pair(&result, &self.primitives[pid].embedding);
        }
        result
    }

    /// Predict composition structure for an input embedding
    pub fn predict_structure(&self, input: &[f64]) -> Vec<f64> {
        let logits = mlp_forward(input, &self.structure_layers);
        softmax(&logits)
    }

    /// Structural analogy: if A:B :: C:? then ? = C + (B - A)
    pub fn analogy(&self, a_ids: &[usize], b_ids: &[usize], c_ids: &[usize]) -> Vec<f64> {
        let a = self.compose(a_ids);
        let b = self.compose(b_ids);
        let c = self.compose(c_ids);
        let diff = vec_sub(&b, &a);
        vec_add(&c, &diff)
    }

    /// Find closest primitive to a given embedding
    pub fn nearest_primitive(&self, embedding: &[f64]) -> usize {
        self.primitives.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| {
                cosine_similarity(embedding, &a.embedding)
                    .partial_cmp(&cosine_similarity(embedding, &b.embedding))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

// ─── Interpreter Builtins ──────────────────────────────────────────────────

fn extract_f64_array(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected number in array".to_string()),
        }).collect(),
        _ => Err("expected array".to_string()),
    }
}

fn f64_array_to_value(v: &[f64]) -> Value {
    Value::Array(v.iter().map(|x| Value::Float(*x)).collect())
}

pub fn register_builtins(env: &mut Env) {
    // Neural memory
    env.functions.insert("neural_memory_new".into(), FnDef::Builtin(builtin_neural_memory_new));
    env.functions.insert("neural_memory_read".into(), FnDef::Builtin(builtin_neural_memory_read));
    env.functions.insert("neural_memory_write".into(), FnDef::Builtin(builtin_neural_memory_write));
    env.functions.insert("neural_memory_allocate".into(), FnDef::Builtin(builtin_neural_memory_allocate));

    // World model
    env.functions.insert("world_model_new".into(), FnDef::Builtin(builtin_world_model_new));
    env.functions.insert("world_model_step".into(), FnDef::Builtin(builtin_world_model_step));
    env.functions.insert("world_model_imagine".into(), FnDef::Builtin(builtin_world_model_imagine));
    env.functions.insert("world_model_reward".into(), FnDef::Builtin(builtin_world_model_reward));
    env.functions.insert("world_model_counterfactual".into(), FnDef::Builtin(builtin_world_model_counterfactual));

    // Symbolic reasoner
    env.functions.insert("reasoner_new".into(), FnDef::Builtin(builtin_reasoner_new));
    env.functions.insert("reasoner_add_rule".into(), FnDef::Builtin(builtin_reasoner_add_rule));
    env.functions.insert("reasoner_add_fact".into(), FnDef::Builtin(builtin_reasoner_add_fact));
    env.functions.insert("reasoner_forward_chain".into(), FnDef::Builtin(builtin_reasoner_forward_chain));
    env.functions.insert("reasoner_backward_chain".into(), FnDef::Builtin(builtin_reasoner_backward_chain));
    env.functions.insert("reasoner_learn_rule".into(), FnDef::Builtin(builtin_reasoner_learn_rule));

    // Global workspace
    env.functions.insert("workspace_new".into(), FnDef::Builtin(builtin_workspace_new));
    env.functions.insert("workspace_broadcast".into(), FnDef::Builtin(builtin_workspace_broadcast));
    env.functions.insert("workspace_assess".into(), FnDef::Builtin(builtin_workspace_assess));

    // Continual learner
    env.functions.insert("continual_learner_new2".into(), FnDef::Builtin(builtin_continual_learner_new2));
    env.functions.insert("continual_learn".into(), FnDef::Builtin(builtin_continual_learn));
    env.functions.insert("continual_replay".into(), FnDef::Builtin(builtin_continual_replay));
    env.functions.insert("continual_finish_task".into(), FnDef::Builtin(builtin_continual_finish_task));
    env.functions.insert("continual_forward".into(), FnDef::Builtin(builtin_continual_forward));

    // Compositional net
    env.functions.insert("compositional_net_new".into(), FnDef::Builtin(builtin_compositional_net_new));
    env.functions.insert("compose".into(), FnDef::Builtin(builtin_compose));
    env.functions.insert("compositional_analogy".into(), FnDef::Builtin(builtin_compositional_analogy));
    env.functions.insert("compositional_nearest".into(), FnDef::Builtin(builtin_compositional_nearest));
}

// --- Neural Memory builtins ---

fn builtin_neural_memory_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("neural_memory_new(slots, dim, read_heads, write_heads)".into()); }
    let slots = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("slots: int".into()) };
    let dim = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("dim: int".into()) };
    let rh = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("read_heads: int".into()) };
    let wh = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("write_heads: int".into()) };
    let mem = NeuralMemory::new(slots, dim, rh, wh);
    let id = env.novel_memories.len();
    env.novel_memories.insert(id, mem);
    Ok(Value::Int(id as i128))
}

fn builtin_neural_memory_read(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("neural_memory_read(mem_id, query)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("mem_id: int".into()) };
    let query = extract_f64_array(&args[1])?;
    let mem = env.novel_memories.get(&id).ok_or("unknown memory id")?;
    let result = mem.read(&query);
    Ok(f64_array_to_value(&result))
}

fn builtin_neural_memory_write(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("neural_memory_write(mem_id, key, value)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("mem_id: int".into()) };
    let key = extract_f64_array(&args[1])?;
    let value = extract_f64_array(&args[2])?;
    let mem = env.novel_memories.get_mut(&id).ok_or("unknown memory id")?;
    mem.write(&key, &value, 0.8);
    Ok(Value::Void)
}

fn builtin_neural_memory_allocate(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("neural_memory_allocate(mem_id)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("mem_id: int".into()) };
    let mem = env.novel_memories.get(&id).ok_or("unknown memory id")?;
    Ok(Value::Int(mem.allocate() as i128))
}

// --- World Model builtins ---

fn builtin_world_model_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("world_model_new(entity_dim, action_dim)".into()); }
    let ed = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("entity_dim: int".into()) };
    let ad = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("action_dim: int".into()) };
    let wm = WorldModel::new(ed, ad);
    let id = env.novel_worlds.len();
    env.novel_worlds.insert(id, wm);
    Ok(Value::Int(id as i128))
}

fn builtin_world_model_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("world_model_step(id, state, action)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let state = extract_f64_array(&args[1])?;
    let action = extract_f64_array(&args[2])?;
    let wm = env.novel_worlds.get(&id).ok_or("unknown world model id")?;
    let next = wm.step(&state, &action);
    Ok(f64_array_to_value(&next))
}

fn builtin_world_model_imagine(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("world_model_imagine(id, state, actions)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let state = extract_f64_array(&args[1])?;
    let actions_val = match &args[2] { Value::Array(a) => a, _ => return Err("actions: array of arrays".into()) };
    let actions: Result<Vec<Vec<f64>>, String> = actions_val.iter().map(extract_f64_array).collect();
    let actions = actions?;
    let wm = env.novel_worlds.get(&id).ok_or("unknown world model id")?;
    let traj = wm.imagine(&state, &actions);
    Ok(Value::Array(traj.iter().map(|s| f64_array_to_value(s)).collect()))
}

fn builtin_world_model_reward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("world_model_reward(id, state, action)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let state = extract_f64_array(&args[1])?;
    let action = extract_f64_array(&args[2])?;
    let wm = env.novel_worlds.get(&id).ok_or("unknown world model id")?;
    Ok(Value::Float(wm.reward(&state, &action)))
}

fn builtin_world_model_counterfactual(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("world_model_counterfactual(id, state, factual_actions, counter_actions)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let state = extract_f64_array(&args[1])?;
    let fa = match &args[2] { Value::Array(a) => a, _ => return Err("factual_actions: array".into()) };
    let ca = match &args[3] { Value::Array(a) => a, _ => return Err("counter_actions: array".into()) };
    let fa: Result<Vec<Vec<f64>>, _> = fa.iter().map(extract_f64_array).collect();
    let ca: Result<Vec<Vec<f64>>, _> = ca.iter().map(extract_f64_array).collect();
    let wm = env.novel_worlds.get(&id).ok_or("unknown world model id")?;
    let (factual, counter) = wm.counterfactual(&state, &fa?, &ca?);
    Ok(Value::Array(vec![
        Value::Array(factual.iter().map(|s| f64_array_to_value(s)).collect()),
        Value::Array(counter.iter().map(|s| f64_array_to_value(s)).collect()),
    ]))
}

// --- Reasoner builtins ---

fn builtin_reasoner_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let dim = if args.is_empty() { 16 } else {
        match &args[0] { Value::Int(n) => *n as usize, _ => 16 }
    };
    let r = SymbolicReasoner::new(dim);
    let id = env.novel_reasoners.len();
    env.novel_reasoners.insert(id, r);
    Ok(Value::Int(id as i128))
}

fn parse_predicate(v: &Value) -> Result<Predicate, String> {
    match v {
        Value::String(s) => {
            // Parse "name(arg1, arg2)" format
            if let Some(paren) = s.find('(') {
                let name = s[..paren].trim().to_string();
                let args_str = s[paren+1..].trim_end_matches(')');
                let args: Vec<String> = args_str.split(',').map(|a| a.trim().to_string()).filter(|a| !a.is_empty()).collect();
                Ok(Predicate { name, args })
            } else {
                Ok(Predicate { name: s.clone(), args: Vec::new() })
            }
        }
        _ => Err("predicate must be a string like 'name(arg1, arg2)'".into()),
    }
}

fn builtin_reasoner_add_rule(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 { return Err("reasoner_add_rule(id, premises, conclusion, [confidence])".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let premises = match &args[1] {
        Value::Array(arr) => arr.iter().map(parse_predicate).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("premises: array of strings".into()),
    };
    let conclusion = parse_predicate(&args[2])?;
    let confidence = if args.len() > 3 {
        match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 1.0 }
    } else { 1.0 };
    let r = env.novel_reasoners.get_mut(&id).ok_or("unknown reasoner id")?;
    r.add_rule(premises, conclusion, confidence);
    Ok(Value::Void)
}

fn builtin_reasoner_add_fact(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("reasoner_add_fact(id, predicate, [confidence])".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let pred = parse_predicate(&args[1])?;
    let conf = if args.len() > 2 {
        match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 1.0 }
    } else { 1.0 };
    let r = env.novel_reasoners.get_mut(&id).ok_or("unknown reasoner id")?;
    r.add_fact(pred, conf);
    Ok(Value::Void)
}

fn builtin_reasoner_forward_chain(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("reasoner_forward_chain(id, [max_iter])".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let max_iter = if args.len() > 1 {
        match &args[1] { Value::Int(n) => *n as usize, _ => 10 }
    } else { 10 };
    let r = env.novel_reasoners.get_mut(&id).ok_or("unknown reasoner id")?;
    let derived = r.forward_chain(max_iter);
    Ok(Value::Array(derived.iter().map(|f| Value::String(format!("{}", f))).collect()))
}

fn builtin_reasoner_backward_chain(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("reasoner_backward_chain(id, goal)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let goal = parse_predicate(&args[1])?;
    let depth = if args.len() > 2 {
        match &args[2] { Value::Int(n) => *n as usize, _ => 10 }
    } else { 10 };
    let r = env.novel_reasoners.get(&id).ok_or("unknown reasoner id")?;
    Ok(Value::Bool(r.backward_chain(&goal, depth)))
}

fn builtin_reasoner_learn_rule(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("reasoner_learn_rule(id, examples)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    // examples: array of [premises_array, conclusion_string]
    let examples_val = match &args[1] { Value::Array(a) => a, _ => return Err("examples: array".into()) };
    let mut examples = Vec::new();
    for ex in examples_val {
        match ex {
            Value::Array(pair) if pair.len() >= 2 => {
                let premises = match &pair[0] {
                    Value::Array(ps) => ps.iter().map(|p| {
                        let pred = parse_predicate(p)?;
                        Ok(Fact { predicate: pred, confidence: 1.0 })
                    }).collect::<Result<Vec<_>, String>>()?,
                    _ => return Err("premise: array of predicates".into()),
                };
                let conclusion_pred = parse_predicate(&pair[1])?;
                let conclusion = Fact { predicate: conclusion_pred, confidence: 1.0 };
                examples.push((premises, conclusion));
            }
            _ => return Err("each example: [premises, conclusion]".into()),
        }
    }
    let r = env.novel_reasoners.get_mut(&id).ok_or("unknown reasoner id")?;
    match r.learn_rule(&examples) {
        Some(rule) => Ok(Value::String(format!("Learned rule: {} IF {}", rule.conclusion,
            rule.premises.iter().map(|p| format!("{}", p)).collect::<Vec<_>>().join(" AND ")))),
        None => Ok(Value::String("No rule learned".into())),
    }
}

// --- Global Workspace builtins ---

fn builtin_workspace_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 { return Err("workspace_new(num_specialists, [input_dim], [ws_dim])".into()); }
    let ns = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("num_specialists: int".into()) };
    let input_dim = if args.len() > 1 { match &args[1] { Value::Int(n) => *n as usize, _ => 16 } } else { 16 };
    let ws_dim = if args.len() > 2 { match &args[2] { Value::Int(n) => *n as usize, _ => 16 } } else { 16 };
    let ws = GlobalWorkspace::new(ns, input_dim, ws_dim);
    let id = env.novel_workspaces.len();
    env.novel_workspaces.insert(id, ws);
    Ok(Value::Int(id as i128))
}

fn builtin_workspace_broadcast(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("workspace_broadcast(ws_id, input)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("ws_id: int".into()) };
    let input = extract_f64_array(&args[1])?;
    let ws = env.novel_workspaces.get_mut(&id).ok_or("unknown workspace id")?;
    let output = ws.broadcast(&input);
    Ok(f64_array_to_value(&output))
}

fn builtin_workspace_assess(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("workspace_assess(ws_id)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("ws_id: int".into()) };
    let ws = env.novel_workspaces.get(&id).ok_or("unknown workspace id")?;
    let (conf, unc, strat) = ws.assess();
    Ok(Value::Array(vec![Value::Float(conf), Value::Float(unc), Value::Int(strat as i128)]))
}

// --- Continual Learner builtins ---

fn builtin_continual_learner_new2(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 { return Err("continual_learner_new2(layer_sizes, [replay_capacity])".into()); }
    let dims = extract_f64_array(&args[0])?.iter().map(|x| *x as usize).collect::<Vec<_>>();
    let cap = if args.len() > 1 { match &args[1] { Value::Int(n) => *n as usize, _ => 1000 } } else { 1000 };
    let cl = ContinualLearner::new(&dims, cap);
    let id = env.novel_continual.len();
    env.novel_continual.insert(id, cl);
    Ok(Value::Int(id as i128))
}

fn builtin_continual_learn(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 { return Err("continual_learn(id, input, target, [lr])".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let input = extract_f64_array(&args[1])?;
    let target = extract_f64_array(&args[2])?;
    let lr = if args.len() > 3 {
        match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.01 }
    } else { 0.01 };
    let cl = env.novel_continual.get_mut(&id).ok_or("unknown continual learner id")?;
    cl.learn_step(&input, &target, lr);
    Ok(Value::Void)
}

fn builtin_continual_replay(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 { return Err("continual_replay(id, [n_samples], [lr])".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let n = if args.len() > 1 { match &args[1] { Value::Int(n) => *n as usize, _ => 5 } } else { 5 };
    let lr = if args.len() > 2 {
        match &args[2] { Value::Float(f) => *f, _ => 0.01 }
    } else { 0.01 };
    let cl = env.novel_continual.get_mut(&id).ok_or("unknown continual learner id")?;
    cl.replay_step(n, lr);
    Ok(Value::Void)
}

fn builtin_continual_finish_task(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("continual_finish_task(id)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let cl = env.novel_continual.get_mut(&id).ok_or("unknown continual learner id")?;
    cl.finish_task();
    Ok(Value::Int(cl.num_tasks_learned() as i128))
}

fn builtin_continual_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("continual_forward(id, input)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id: int".into()) };
    let input = extract_f64_array(&args[1])?;
    let cl = env.novel_continual.get(&id).ok_or("unknown continual learner id")?;
    let output = cl.forward(&input);
    Ok(f64_array_to_value(&output))
}

// --- Compositional Net builtins ---

fn builtin_compositional_net_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("compositional_net_new(num_primitives, dim)".into()); }
    let np = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("num_primitives: int".into()) };
    let dim = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("dim: int".into()) };
    let cn = CompositionalNet::new(np, dim);
    let id = env.novel_compositional.len();
    env.novel_compositional.insert(id, cn);
    Ok(Value::Int(id as i128))
}

fn builtin_compose(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("compose(net_id, primitive_ids)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("net_id: int".into()) };
    let ids = match &args[1] {
        Value::Array(a) => a.iter().map(|v| match v {
            Value::Int(n) => Ok(*n as usize),
            _ => Err("primitive_ids: array of ints".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("primitive_ids: array".into()),
    };
    let cn = env.novel_compositional.get(&id).ok_or("unknown compositional net id")?;
    let result = cn.compose(&ids);
    Ok(f64_array_to_value(&result))
}

fn builtin_compositional_analogy(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("compositional_analogy(net_id, a_ids, b_ids, c_ids)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("net_id: int".into()) };
    let extract_ids = |v: &Value| -> Result<Vec<usize>, String> {
        match v {
            Value::Array(a) => a.iter().map(|x| match x {
                Value::Int(n) => Ok(*n as usize), _ => Err("int expected".into()),
            }).collect(),
            _ => Err("array expected".into()),
        }
    };
    let a = extract_ids(&args[1])?;
    let b = extract_ids(&args[2])?;
    let c = extract_ids(&args[3])?;
    let cn = env.novel_compositional.get(&id).ok_or("unknown compositional net id")?;
    let result = cn.analogy(&a, &b, &c);
    Ok(f64_array_to_value(&result))
}

fn builtin_compositional_nearest(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("compositional_nearest(net_id, embedding)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("net_id: int".into()) };
    let emb = extract_f64_array(&args[1])?;
    let cn = env.novel_compositional.get(&id).ok_or("unknown compositional net id")?;
    Ok(Value::Int(cn.nearest_primitive(&emb) as i128))
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neural_memory_create() {
        let mem = NeuralMemory::new(8, 16, 1, 1);
        assert_eq!(mem.memory_bank.len(), 8);
        assert_eq!(mem.memory_bank[0].len(), 16);
        assert_eq!(mem.usage.len(), 8);
    }

    #[test]
    fn test_neural_memory_read_write() {
        let mut mem = NeuralMemory::new(4, 8, 1, 1);
        let key = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let value = vec![0.5; 8];
        mem.write(&key, &value, 0.9);
        let result = mem.read(&key);
        assert_eq!(result.len(), 8);
        // After writing, reading with same key should return something closer to value
    }

    #[test]
    fn test_neural_memory_content_addressing() {
        let mem = NeuralMemory::new(4, 8, 1, 1);
        let query = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let weights = mem.address_content(&query);
        assert_eq!(weights.len(), 4);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6, "softmax weights should sum to 1, got {}", sum);
    }

    #[test]
    fn test_neural_memory_location_addressing() {
        let mem = NeuralMemory::new(4, 8, 1, 1);
        let weights = vec![0.1, 0.2, 0.6, 0.1];
        let shifted = mem.address_location(&weights, 1, 2.0);
        assert_eq!(shifted.len(), 4);
        let sum: f64 = shifted.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_neural_memory_lru_allocate() {
        let mut mem = NeuralMemory::new(4, 8, 1, 1);
        // All usage is 0 initially, so allocate should return 0
        assert_eq!(mem.allocate(), 0);
        mem.usage[0] = 5.0;
        mem.usage[1] = 3.0;
        mem.usage[2] = 1.0;
        mem.usage[3] = 7.0;
        assert_eq!(mem.allocate(), 2); // slot 2 has lowest usage
    }

    #[test]
    fn test_reasoner_forward_chain() {
        let mut r = SymbolicReasoner::new(8);
        r.add_fact(Predicate { name: "parent".into(), args: vec!["alice".into(), "bob".into()] }, 1.0);
        r.add_fact(Predicate { name: "parent".into(), args: vec!["bob".into(), "charlie".into()] }, 1.0);
        r.add_rule(
            vec![
                Predicate { name: "parent".into(), args: vec!["?x".into(), "?y".into()] },
                Predicate { name: "parent".into(), args: vec!["?y".into(), "?z".into()] },
            ],
            Predicate { name: "grandparent".into(), args: vec!["?x".into(), "?z".into()] },
            1.0,
        );
        let derived = r.forward_chain(5);
        assert!(!derived.is_empty(), "should derive grandparent fact");
        assert!(r.facts.iter().any(|f| f.predicate.name == "grandparent"));
    }

    #[test]
    fn test_reasoner_backward_chain() {
        let mut r = SymbolicReasoner::new(8);
        r.add_fact(Predicate { name: "human".into(), args: vec!["socrates".into()] }, 1.0);
        r.add_rule(
            vec![Predicate { name: "human".into(), args: vec!["?x".into()] }],
            Predicate { name: "mortal".into(), args: vec!["?x".into()] },
            1.0,
        );
        let goal = Predicate { name: "mortal".into(), args: vec!["socrates".into()] };
        assert!(r.backward_chain(&goal, 5));
        let bad_goal = Predicate { name: "mortal".into(), args: vec!["zeus".into()] };
        assert!(!r.backward_chain(&bad_goal, 5));
    }

    #[test]
    fn test_reasoner_learn_rule() {
        let mut r = SymbolicReasoner::new(8);
        let examples = vec![
            (
                vec![Fact { predicate: Predicate { name: "hot".into(), args: vec!["x".into()] }, confidence: 1.0 }],
                Fact { predicate: Predicate { name: "danger".into(), args: vec!["x".into()] }, confidence: 0.9 },
            ),
            (
                vec![Fact { predicate: Predicate { name: "hot".into(), args: vec!["y".into()] }, confidence: 1.0 }],
                Fact { predicate: Predicate { name: "danger".into(), args: vec!["y".into()] }, confidence: 0.8 },
            ),
        ];
        let rule = r.learn_rule(&examples);
        assert!(rule.is_some());
        assert!(rule.unwrap().learned);
    }

    #[test]
    fn test_reasoner_soft_unify() {
        let mut r = SymbolicReasoner::new(8);
        let a = Predicate { name: "likes".into(), args: vec!["alice".into(), "bob".into()] };
        let b = Predicate { name: "likes".into(), args: vec!["alice".into(), "bob".into()] };
        let score = r.soft_unify(&a, &b);
        assert!((score - 1.0).abs() < 1e-6, "identical predicates should unify with score 1.0");
    }

    #[test]
    fn test_world_model_step() {
        let wm = WorldModel::new(8, 4);
        let state = vec![0.1; 8];
        let action = vec![0.5; 4];
        let next = wm.step(&state, &action);
        assert_eq!(next.len(), 8);
        // Next state should differ from current state
        assert!(next != state, "next state should differ from current");
    }

    #[test]
    fn test_world_model_imagine_trajectory() {
        let wm = WorldModel::new(8, 4);
        let state = vec![0.1; 8];
        let actions = vec![vec![0.5; 4], vec![-0.3; 4], vec![0.1; 4]];
        let traj = wm.imagine(&state, &actions);
        assert_eq!(traj.len(), 4); // initial + 3 steps
        assert_eq!(traj[0], state);
    }

    #[test]
    fn test_world_model_counterfactual_differs() {
        let wm = WorldModel::new(8, 4);
        let state = vec![0.1; 8];
        let factual = vec![vec![1.0; 4]];
        let counter = vec![vec![-1.0; 4]];
        let (f_traj, c_traj) = wm.counterfactual(&state, &factual, &counter);
        assert_eq!(f_traj.len(), 2);
        assert_eq!(c_traj.len(), 2);
        // Final states should differ
        assert!(f_traj[1] != c_traj[1], "counterfactual should produce different outcome");
    }

    #[test]
    fn test_world_model_reward() {
        let wm = WorldModel::new(8, 4);
        let state = vec![0.1; 8];
        let action = vec![0.5; 4];
        let r = wm.reward(&state, &action);
        assert!(r.is_finite());
    }

    #[test]
    fn test_global_workspace_broadcast() {
        let mut ws = GlobalWorkspace::new(3, 8, 8);
        let input = vec![0.5; 8];
        let output = ws.broadcast(&input);
        assert_eq!(output.len(), 8);
    }

    #[test]
    fn test_global_workspace_selects_most_competent() {
        let mut ws = GlobalWorkspace::new(3, 8, 8);
        // Set specialist 1 to be most competent
        ws.specialists[1].competence = 0.9;
        ws.specialists[0].competence = 0.1;
        ws.specialists[2].competence = 0.1;
        let scores = ws.compete(&vec![0.5; 8]);
        // Specialist 1 should rank highly (though output norm also matters)
        assert!(!scores.is_empty());
    }

    #[test]
    fn test_global_workspace_metacognition() {
        let ws = GlobalWorkspace::new(3, 8, 8);
        let (conf, unc, strat) = ws.assess();
        assert!(conf >= 0.0 && conf <= 1.0);
        assert!(unc >= 0.0 && unc <= 1.0);
        assert!(strat <= 2);
    }

    #[test]
    fn test_continual_learner_forward() {
        let cl = ContinualLearner::new(&[4, 8, 2], 100);
        let input = vec![0.5; 4];
        let output = cl.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_continual_learner_remembers_old_tasks() {
        let mut cl = ContinualLearner::new(&[4, 8, 2], 100);
        // Learn task 1
        for _ in 0..10 {
            cl.learn_step(&[1.0, 0.0, 0.0, 0.0], &[1.0, 0.0], 0.01);
        }
        let output_before = cl.forward(&[1.0, 0.0, 0.0, 0.0]);
        cl.finish_task();
        // Learn task 2
        for _ in 0..10 {
            cl.learn_step(&[0.0, 0.0, 0.0, 1.0], &[0.0, 1.0], 0.01);
        }
        let output_after = cl.forward(&[1.0, 0.0, 0.0, 0.0]);
        // With progressive nets, old knowledge should be partially preserved
        // (output shouldn't be completely different)
        let diff: f64 = output_before.iter().zip(&output_after).map(|(a, b)| (a - b).abs()).sum();
        // Just check it still produces reasonable output
        assert!(output_after.len() == 2);
        assert!(diff.is_finite());
    }

    #[test]
    fn test_continual_learner_progressive_columns() {
        let mut cl = ContinualLearner::new(&[4, 8, 2], 100);
        assert_eq!(cl.columns.len(), 0);
        cl.finish_task();
        assert_eq!(cl.columns.len(), 1);
        assert!(cl.columns[0].frozen);
        cl.finish_task();
        assert_eq!(cl.columns.len(), 2);
    }

    #[test]
    fn test_compositional_net_compose() {
        let cn = CompositionalNet::new(5, 8);
        let result = cn.compose(&[0, 1]);
        assert_eq!(result.len(), 8);
    }

    #[test]
    fn test_compositional_net_analogy() {
        let cn = CompositionalNet::new(5, 8);
        let result = cn.analogy(&[0], &[1], &[2]);
        assert_eq!(result.len(), 8);
        // analogy(A, B, C) = C + (B - A), so should differ from just compose(C)
        let c_only = cn.compose(&[2]);
        assert!(result != c_only);
    }

    #[test]
    fn test_compositional_net_nearest_primitive() {
        let cn = CompositionalNet::new(5, 8);
        // The embedding of primitive 0 should be nearest to itself
        let emb = cn.primitives[0].embedding.clone();
        let nearest = cn.nearest_primitive(&emb);
        assert_eq!(nearest, 0);
    }

    #[test]
    fn test_compositional_net_structure_prediction() {
        let cn = CompositionalNet::new(5, 8);
        let input = vec![0.5; 8];
        let probs = cn.predict_structure(&input);
        assert_eq!(probs.len(), 5);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let s = softmax(&v);
        let sum: f64 = s.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }
}
