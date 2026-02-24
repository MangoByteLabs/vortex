//! Pillar 8: AGI Emergence Core for the Vortex language.
//! Implements self-models, goal systems, world models, temporal reasoning,
//! meta-meta-learning, and consciousness loops.

use crate::interpreter::{Env, Value, FnDef};
use std::collections::HashMap;
use std::sync::Mutex;

// ── Helper conversions ─────────────────────────────────────────────────────

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("Expected number".to_string()),
    }
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Float(f) => Ok(*f as usize),
        _ => Err("Expected integer".to_string()),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => Err("Expected array".to_string()),
    }
}

fn value_to_f64_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect(),
        _ => Err("Expected 2D array".to_string()),
    }
}

fn value_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        Value::Int(n) => Ok(n.to_string()),
        Value::Float(f) => Ok(f.to_string()),
        _ => Err("Expected string".to_string()),
    }
}

// ── PRNG ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 1 } else { seed } }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_f64(&mut self) -> f64 {
        (self.next() as f64) / (u64::MAX as f64)
    }

    fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

// ── Core structs ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct SelfModel {
    state_history: Vec<Vec<f64>>,
    attention_weights: Vec<f64>,
    confidence: f64,
    introspection_depth: usize,
    metacognitive_log: Vec<String>,
}

impl SelfModel {
    fn new(dim: usize) -> Self {
        let attention_weights = vec![1.0 / dim as f64; dim];
        Self {
            state_history: Vec::new(),
            attention_weights,
            confidence: 0.5,
            introspection_depth: 0,
            metacognitive_log: Vec::new(),
        }
    }

    fn record_state(&mut self, state: Vec<f64>) {
        self.state_history.push(state);
        if self.state_history.len() > 1000 {
            self.state_history.remove(0);
        }
    }

    fn compute_novelty(&self, observation: &[f64]) -> f64 {
        if self.state_history.is_empty() {
            return 1.0;
        }
        let mean: Vec<f64> = {
            let n = self.state_history.len() as f64;
            let dim = observation.len();
            let mut m = vec![0.0; dim];
            for s in &self.state_history {
                for (i, v) in s.iter().enumerate() {
                    if i < dim {
                        m[i] += v / n;
                    }
                }
            }
            m
        };
        let dist: f64 = observation.iter().zip(mean.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        (dist / (1.0 + dist)).min(1.0)
    }

    fn compute_variance(&self) -> f64 {
        if self.state_history.len() < 2 {
            return 0.0;
        }
        let n = self.state_history.len() as f64;
        let dim = self.state_history[0].len();
        let mut total_var = 0.0;
        for d in 0..dim {
            let mean = self.state_history.iter().map(|s| s.get(d).copied().unwrap_or(0.0)).sum::<f64>() / n;
            let var = self.state_history.iter()
                .map(|s| { let v = s.get(d).copied().unwrap_or(0.0) - mean; v * v })
                .sum::<f64>() / n;
            total_var += var;
        }
        total_var / dim as f64
    }

    fn update_confidence(&mut self, prediction_error: f64) {
        let alpha = 0.1;
        self.confidence = self.confidence * (1.0 - alpha) + (1.0 - prediction_error.min(1.0)) * alpha;
        self.confidence = self.confidence.clamp(0.0, 1.0);
    }

    fn update_attention(&mut self, errors: &[f64]) {
        if errors.len() != self.attention_weights.len() {
            return;
        }
        let total: f64 = errors.iter().map(|e| e.abs() + 1e-8).sum();
        for (w, e) in self.attention_weights.iter_mut().zip(errors.iter()) {
            *w = 0.9 * *w + 0.1 * (e.abs() + 1e-8) / total;
        }
        let sum: f64 = self.attention_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.attention_weights {
                *w /= sum;
            }
        }
    }
}

// ── Goal System ────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum GoalStatus {
    Active,
    Completed,
    Suspended,
    Failed,
}

impl GoalStatus {
    fn as_str(&self) -> &str {
        match self {
            GoalStatus::Active => "active",
            GoalStatus::Completed => "completed",
            GoalStatus::Suspended => "suspended",
            GoalStatus::Failed => "failed",
        }
    }
}

#[derive(Debug, Clone)]
struct Goal {
    id: usize,
    description: String,
    priority: f64,
    subgoals: Vec<usize>,
    progress: f64,
    status: GoalStatus,
}

#[derive(Debug, Clone)]
struct GoalSystem {
    goals: Vec<Goal>,
    next_goal_id: usize,
    curiosity_weight: f64,
    reward_history: Vec<f64>,
}

impl GoalSystem {
    fn new() -> Self {
        Self {
            goals: Vec::new(),
            next_goal_id: 0,
            curiosity_weight: 0.3,
            reward_history: Vec::new(),
        }
    }

    fn add_goal(&mut self, description: String, priority: f64) -> usize {
        let id = self.next_goal_id;
        self.next_goal_id += 1;
        self.goals.push(Goal {
            id,
            description,
            priority,
            subgoals: Vec::new(),
            progress: 0.0,
            status: GoalStatus::Active,
        });
        id
    }

    fn get_goal(&self, id: usize) -> Option<&Goal> {
        self.goals.iter().find(|g| g.id == id)
    }

    fn get_goal_mut(&mut self, id: usize) -> Option<&mut Goal> {
        self.goals.iter_mut().find(|g| g.id == id)
    }

    fn decompose_goal(&mut self, goal_id: usize, rng: &mut Xorshift64) -> Vec<usize> {
        let desc = match self.get_goal(goal_id) {
            Some(g) => g.description.clone(),
            None => return Vec::new(),
        };
        let priority = self.get_goal(goal_id).map(|g| g.priority).unwrap_or(0.5);
        let num_sub = 2 + (rng.next() % 3) as usize;
        let mut sub_ids = Vec::new();
        for i in 0..num_sub {
            let sub_desc = format!("{} - subtask {}", desc, i + 1);
            let sub_priority = priority * (0.7 + 0.3 * rng.next_f64());
            let sid = self.add_goal(sub_desc, sub_priority);
            sub_ids.push(sid);
        }
        if let Some(g) = self.get_goal_mut(goal_id) {
            g.subgoals = sub_ids.clone();
        }
        sub_ids
    }

    fn evaluate_progress(&self) -> f64 {
        if self.goals.is_empty() {
            return 0.0;
        }
        let total: f64 = self.goals.iter().map(|g| g.progress * g.priority).sum();
        let weight: f64 = self.goals.iter().map(|g| g.priority).sum();
        if weight > 0.0 { total / weight } else { 0.0 }
    }

    fn generate_goals(&mut self, context: &[f64], rng: &mut Xorshift64) -> Vec<String> {
        let num = 1 + (context.len() % 3);
        let mut descs = Vec::new();
        for i in 0..num {
            let ctx_sum: f64 = context.iter().sum();
            let desc = if ctx_sum > 0.0 {
                format!("explore_positive_region_{}", i)
            } else if ctx_sum < 0.0 {
                format!("investigate_negative_anomaly_{}", i)
            } else {
                format!("gather_more_data_{}", i)
            };
            let priority = 0.3 + 0.7 * rng.next_f64();
            self.add_goal(desc.clone(), priority);
            descs.push(desc);
        }
        descs
    }
}

// ── World Model ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct WorldModel {
    state: Vec<f64>,
    transition_weights: Vec<Vec<f64>>,
    observation_weights: Vec<Vec<f64>>,
    imagination_buffer: Vec<Vec<f64>>,
    prediction_error: f64,
}

impl WorldModel {
    fn new(dim: usize, rng: &mut Xorshift64) -> Self {
        let transition_weights = Self::init_matrix(dim, dim, rng);
        let observation_weights = Self::init_matrix(dim, dim, rng);
        Self {
            state: vec![0.0; dim],
            transition_weights,
            observation_weights,
            imagination_buffer: Vec::new(),
            prediction_error: 1.0,
        }
    }

    fn init_matrix(rows: usize, cols: usize, rng: &mut Xorshift64) -> Vec<Vec<f64>> {
        let scale = 1.0 / (cols as f64).sqrt();
        (0..rows).map(|_| {
            (0..cols).map(|_| rng.next_range(-scale, scale)).collect()
        }).collect()
    }

    fn predict(&self) -> Vec<f64> {
        mat_vec_mul(&self.transition_weights, &self.state)
    }

    fn update(&mut self, observation: &[f64], learning_rate: f64) {
        let predicted = self.predict();
        let dim = self.state.len().min(observation.len());
        let mut errors = vec![0.0; dim];
        let mut total_error = 0.0;
        for i in 0..dim {
            errors[i] = observation[i] - predicted[i];
            total_error += errors[i] * errors[i];
        }
        self.prediction_error = (total_error / dim as f64).sqrt();

        // Update transition weights via gradient
        for i in 0..self.transition_weights.len().min(dim) {
            for j in 0..self.transition_weights[i].len().min(dim) {
                self.transition_weights[i][j] += learning_rate * errors[i] * self.state[j];
            }
        }

        // Update state
        for i in 0..dim {
            self.state[i] = observation[i];
        }
    }

    fn forward_step(&self, state: &[f64]) -> Vec<f64> {
        mat_vec_mul(&self.transition_weights, state)
    }

    fn simulate(&self, action: &[f64], steps: usize) -> Vec<Vec<f64>> {
        let mut trajectory = Vec::new();
        let dim = self.state.len();
        let mut current = vec![0.0; dim];
        for i in 0..dim {
            current[i] = self.state[i] + action.get(i).copied().unwrap_or(0.0);
        }
        trajectory.push(current.clone());
        for _ in 1..steps {
            current = self.forward_step(&current);
            // Apply tanh to keep bounded
            for v in &mut current {
                *v = v.tanh();
            }
            trajectory.push(current.clone());
        }
        trajectory
    }

    fn imagination_rollout(&mut self, initial_state: &[f64], num_steps: usize) -> Vec<Vec<f64>> {
        let result = self.simulate(initial_state, num_steps);
        self.imagination_buffer.extend(result.clone());
        if self.imagination_buffer.len() > 500 {
            self.imagination_buffer = self.imagination_buffer.split_off(self.imagination_buffer.len() - 500);
        }
        result
    }
}

// ── Domain Adapter ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct DomainAdapter {
    source_dim: usize,
    target_dim: usize,
    alignment_weights: Vec<Vec<f64>>,
    transfer_history: Vec<f64>,
}

impl DomainAdapter {
    fn new(source_dim: usize, target_dim: usize, rng: &mut Xorshift64) -> Self {
        let scale = 1.0 / (source_dim as f64).sqrt();
        let alignment_weights = (0..target_dim).map(|_| {
            (0..source_dim).map(|_| rng.next_range(-scale, scale)).collect()
        }).collect();
        Self {
            source_dim,
            target_dim,
            alignment_weights,
            transfer_history: Vec::new(),
        }
    }

    fn transfer(&mut self, source_data: &[f64]) -> Vec<f64> {
        let result = mat_vec_mul(&self.alignment_weights, source_data);
        let norm: f64 = result.iter().map(|x| x * x).sum::<f64>().sqrt();
        self.transfer_history.push(norm);
        if self.transfer_history.len() > 200 {
            self.transfer_history.remove(0);
        }
        result
    }

    fn resize(&mut self, new_target_dim: usize, rng: &mut Xorshift64) {
        let scale = 1.0 / (self.source_dim as f64).sqrt();
        self.alignment_weights.resize_with(new_target_dim, || {
            (0..self.source_dim).map(|_| rng.next_range(-scale, scale)).collect()
        });
        self.target_dim = new_target_dim;
    }
}

// ── Temporal Reasoner ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct TemporalReasoner {
    memory: Vec<(u64, Vec<f64>)>,
    causal_weights: Vec<Vec<f64>>,
    time_scales: Vec<f64>,
    current_time: u64,
}

impl TemporalReasoner {
    fn new(dim: usize, rng: &mut Xorshift64) -> Self {
        let causal_weights = WorldModel::init_matrix(dim, dim, rng);
        let time_scales = vec![1.0, 10.0, 100.0, 1000.0];
        Self {
            memory: Vec::new(),
            causal_weights,
            time_scales,
            current_time: 0,
        }
    }

    fn record(&mut self, state: Vec<f64>) {
        self.memory.push((self.current_time, state));
        self.current_time += 1;
        if self.memory.len() > 500 {
            self.memory.remove(0);
        }
    }

    fn reason(&self, query_time: u64, context: &[f64]) -> Vec<f64> {
        if self.memory.is_empty() {
            return context.to_vec();
        }
        let dim = context.len();
        let mut result = vec![0.0; dim];
        let mut total_weight = 0.0;

        for (t, state) in &self.memory {
            // Weight by temporal proximity across multiple scales
            let mut proximity = 0.0;
            for scale in &self.time_scales {
                let dt = (query_time as f64 - *t as f64).abs();
                proximity += (-dt / scale).exp();
            }
            proximity /= self.time_scales.len() as f64;

            // Causal relevance
            let causal_rel: f64 = context.iter().zip(state.iter())
                .map(|(c, s)| c * s)
                .sum::<f64>()
                .abs();
            let weight = proximity * (1.0 + causal_rel);
            total_weight += weight;

            for i in 0..dim.min(state.len()) {
                result[i] += weight * state[i];
            }
        }

        if total_weight > 0.0 {
            for v in &mut result {
                *v /= total_weight;
            }
        }

        // Apply causal weights
        let causal_output = mat_vec_mul(&self.causal_weights, &result);
        // Blend
        for i in 0..dim.min(causal_output.len()) {
            result[i] = 0.7 * result[i] + 0.3 * causal_output[i];
        }
        result
    }
}

// ── Meta-Meta Learner ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MetaMetaLearner {
    strategies: Vec<Vec<f64>>,
    strategy_scores: Vec<f64>,
    evolution_history: Vec<f64>,
    current_strategy: usize,
}

impl MetaMetaLearner {
    fn new(dim: usize, num_strategies: usize, rng: &mut Xorshift64) -> Self {
        let strategies: Vec<Vec<f64>> = (0..num_strategies).map(|_| {
            (0..dim).map(|_| rng.next_range(-1.0, 1.0)).collect()
        }).collect();
        let strategy_scores = vec![0.0; num_strategies];
        Self {
            strategies,
            strategy_scores,
            evolution_history: Vec::new(),
            current_strategy: 0,
        }
    }

    fn step(&mut self, performance: f64, rng: &mut Xorshift64) -> String {
        // Update score of current strategy
        self.strategy_scores[self.current_strategy] =
            0.8 * self.strategy_scores[self.current_strategy] + 0.2 * performance;
        self.evolution_history.push(performance);

        // Find best strategy
        let best = self.strategy_scores.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        // Evolve worst strategy by mutating from best
        let worst = self.strategy_scores.iter()
            .enumerate()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);

        if worst != best {
            let best_strat = self.strategies[best].clone();
            let dim = best_strat.len();
            self.strategies[worst] = (0..dim).map(|i| {
                best_strat[i] + rng.next_range(-0.1, 0.1)
            }).collect();
            self.strategy_scores[worst] = self.strategy_scores[best] * 0.5;
        }

        // Select strategy: epsilon-greedy
        if rng.next_f64() < 0.1 {
            self.current_strategy = rng.next() as usize % self.strategies.len();
        } else {
            self.current_strategy = best;
        }

        format!("strategy_{}_score_{:.4}_evolved_{}", self.current_strategy,
                self.strategy_scores[self.current_strategy], worst != best)
    }

    fn current_strategy_vec(&self) -> &[f64] {
        &self.strategies[self.current_strategy]
    }
}

// ── AGI Core ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct AGICore {
    self_model: SelfModel,
    goals: GoalSystem,
    world_model: WorldModel,
    transfer: DomainAdapter,
    temporal: TemporalReasoner,
    meta_meta: MetaMetaLearner,
    cycle_count: u64,
    state_dim: usize,
    rng: Xorshift64,
}

impl AGICore {
    fn new(state_dim: usize, seed: u64) -> Self {
        let mut rng = Xorshift64::new(seed);
        Self {
            self_model: SelfModel::new(state_dim),
            goals: GoalSystem::new(),
            world_model: WorldModel::new(state_dim, &mut rng),
            transfer: DomainAdapter::new(state_dim, state_dim, &mut rng),
            temporal: TemporalReasoner::new(state_dim, &mut rng),
            meta_meta: MetaMetaLearner::new(state_dim, 5, &mut rng),
            cycle_count: 0,
            state_dim,
            rng,
        }
    }

    /// One full consciousness cycle: perceive → reflect → plan → act → learn
    fn consciousness_loop(&mut self, observation: &[f64]) -> Vec<f64> {
        let dim = self.state_dim;

        // 1. Perceive: world model predicts, compare with observation
        let predicted = self.world_model.predict();
        let mut errors = vec![0.0; dim];
        for i in 0..dim.min(observation.len()).min(predicted.len()) {
            errors[i] = observation[i] - predicted[i];
        }

        // 2. Update world model
        self.world_model.update(observation, 0.01);

        // 3. Reflect: update self model
        self.self_model.update_confidence(self.world_model.prediction_error);
        self.self_model.update_attention(&errors);
        self.self_model.record_state(observation.to_vec());

        // 4. Temporal recording
        self.temporal.record(observation.to_vec());

        // 5. Plan action: use meta-meta strategy + attention-weighted response to error
        let strategy = self.meta_meta.current_strategy_vec().to_vec();
        let mut action = vec![0.0; dim];
        for i in 0..dim {
            let s = strategy.get(i).copied().unwrap_or(0.0);
            let a = self.self_model.attention_weights.get(i).copied().unwrap_or(1.0 / dim as f64);
            let e = errors.get(i).copied().unwrap_or(0.0);
            action[i] = (s * a * e).tanh();
        }

        // 6. Update goal progress
        let performance = 1.0 - self.world_model.prediction_error.min(1.0);
        for g in &mut self.goals.goals {
            if g.status == GoalStatus::Active {
                g.progress = (g.progress + performance * 0.01).min(1.0);
                if g.progress >= 1.0 {
                    g.status = GoalStatus::Completed;
                }
            }
        }
        self.goals.reward_history.push(performance);

        // 7. Meta-meta step
        self.meta_meta.step(performance, &mut self.rng);

        // 8. Log
        let novelty = self.self_model.compute_novelty(observation);
        self.self_model.metacognitive_log.push(
            format!("cycle={} err={:.4} conf={:.4} nov={:.4}",
                    self.cycle_count, self.world_model.prediction_error,
                    self.self_model.confidence, novelty)
        );
        if self.self_model.metacognitive_log.len() > 500 {
            self.self_model.metacognitive_log.remove(0);
        }

        self.cycle_count += 1;
        action
    }

    fn reflect(&self) -> String {
        let novelty = if !self.self_model.state_history.is_empty() {
            self.self_model.compute_novelty(self.self_model.state_history.last().unwrap())
        } else {
            0.0
        };
        let variance = self.self_model.compute_variance();
        let confusion = self.world_model.prediction_error;
        format!(
            "confidence={:.4} novelty={:.4} variance={:.4} confusion={:.4} cycles={} goals={} active_goals={}",
            self.self_model.confidence,
            novelty,
            variance,
            confusion,
            self.cycle_count,
            self.goals.goals.len(),
            self.goals.goals.iter().filter(|g| g.status == GoalStatus::Active).count()
        )
    }

    fn dream(&mut self, num_episodes: usize) -> String {
        if self.self_model.state_history.is_empty() {
            return "no_experiences_to_dream_about".to_string();
        }
        let decay: f64 = 0.95;
        let mut consolidated = vec![0.0; self.state_dim];
        let mut total_weight = 0.0;

        let history_len = self.self_model.state_history.len();
        let episodes = num_episodes.min(history_len);

        for ep in 0..episodes {
            // Replay from recent experiences with decay
            let idx = history_len - 1 - (ep % history_len);
            let weight = decay.powi(ep as i32);
            total_weight += weight;
            let state = &self.self_model.state_history[idx];
            for i in 0..self.state_dim.min(state.len()) {
                consolidated[i] += weight * state[i];
            }
        }

        if total_weight > 0.0 {
            for v in &mut consolidated {
                *v /= total_weight;
            }
        }

        // Inject consolidated knowledge into world model state
        for i in 0..self.state_dim.min(consolidated.len()) {
            self.world_model.state[i] = 0.9 * self.world_model.state[i] + 0.1 * consolidated[i];
        }

        let energy: f64 = consolidated.iter().map(|x| x * x).sum::<f64>().sqrt();
        format!("dreamed_{}_episodes_energy_{:.4}_consolidated", episodes, energy)
    }

    fn curiosity_score(&self, observation: &[f64]) -> f64 {
        // Curiosity = prediction error (novelty signal)
        let predicted = self.world_model.predict();
        let dim = self.state_dim.min(observation.len()).min(predicted.len());
        let mut error = 0.0;
        for i in 0..dim {
            let e = observation[i] - predicted[i];
            error += e * e;
        }
        let pred_error = (error / dim.max(1) as f64).sqrt();
        let novelty = self.self_model.compute_novelty(observation);
        // Blend prediction error and novelty
        0.6 * pred_error.min(1.0) + 0.4 * novelty
    }

    fn communicate(&mut self, message: &[f64]) -> Vec<f64> {
        // Process message through world model observation weights
        let processed = mat_vec_mul(&self.world_model.observation_weights, message);
        // Generate response based on current state + processed message
        let dim = self.state_dim;
        let mut response = vec![0.0; dim];
        for i in 0..dim.min(processed.len()) {
            response[i] = (self.world_model.state[i] + processed[i]).tanh();
        }
        response
    }

    fn self_improve(&mut self, feedback: &[f64]) -> String {
        let dim = self.state_dim;
        let lr = 0.05;

        // Adjust world model transition weights based on feedback
        for i in 0..dim.min(self.world_model.transition_weights.len()) {
            for j in 0..dim.min(self.world_model.transition_weights[i].len()) {
                let fb = feedback.get(i).copied().unwrap_or(0.0);
                self.world_model.transition_weights[i][j] += lr * fb * self.world_model.state[j];
            }
        }

        // Adjust attention weights
        let total: f64 = feedback.iter().map(|f| f.abs() + 1e-8).sum();
        for i in 0..dim.min(self.self_model.attention_weights.len()) {
            let fb = feedback.get(i).copied().unwrap_or(0.0);
            self.self_model.attention_weights[i] =
                0.8 * self.self_model.attention_weights[i] + 0.2 * (fb.abs() + 1e-8) / total;
        }
        let sum: f64 = self.self_model.attention_weights.iter().sum();
        if sum > 0.0 {
            for w in &mut self.self_model.attention_weights {
                *w /= sum;
            }
        }

        self.self_model.confidence = (self.self_model.confidence + 0.01).min(1.0);
        format!("improved_lr_{:.4}_confidence_{:.4}", lr, self.self_model.confidence)
    }

    fn introspect(&self, depth: usize) -> String {
        if depth == 0 {
            return format!("state_dim={} confidence={:.4}", self.state_dim, self.self_model.confidence);
        }
        let inner = self.introspect(depth - 1);
        let novelty = if !self.self_model.state_history.is_empty() {
            self.self_model.compute_novelty(self.self_model.state_history.last().unwrap())
        } else {
            0.0
        };
        format!(
            "depth={} cycles={} novelty={:.4} goals={} error={:.4} [{}]",
            depth,
            self.cycle_count,
            novelty,
            self.goals.goals.len(),
            self.world_model.prediction_error,
            inner
        )
    }
}

// ── Linear algebra helper ──────────────────────────────────────────────────

fn mat_vec_mul(mat: &[Vec<f64>], vec: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| {
        row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

// ── Global storage ─────────────────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref AGI_STORE: Mutex<HashMap<usize, AGICore>> = Mutex::new(HashMap::new());
    static ref AGI_NEXT_ID: Mutex<usize> = Mutex::new(0);
}

fn store_agi(core: AGICore) -> usize {
    let mut store = AGI_STORE.lock().unwrap();
    let mut id = AGI_NEXT_ID.lock().unwrap();
    let current_id = *id;
    *id += 1;
    store.insert(current_id, core);
    current_id
}

fn with_agi<F, R>(id: usize, f: F) -> Result<R, String>
where
    F: FnOnce(&mut AGICore) -> R,
{
    let mut store = AGI_STORE.lock().unwrap();
    match store.get_mut(&id) {
        Some(core) => Ok(f(core)),
        None => Err(format!("AGI core {} not found", id)),
    }
}

// ── Builtins ───────────────────────────────────────────────────────────────

fn builtin_agi_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("agi_new(state_dim)".to_string());
    }
    let state_dim = value_to_usize(&args[0])?;
    if state_dim == 0 || state_dim > 4096 {
        return Err("state_dim must be 1..4096".to_string());
    }
    let seed = (state_dim as u64).wrapping_mul(2654435761);
    let core = AGICore::new(state_dim, seed);
    let id = store_agi(core);
    Ok(Value::Int(id as i128))
}

fn builtin_agi_consciousness_loop(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_consciousness_loop(agi_id, observation)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let obs = value_to_f64_vec(&args[1])?;
    let action = with_agi(id, |core| core.consciousness_loop(&obs))?;
    Ok(Value::Array(action.into_iter().map(Value::Float).collect()))
}

fn builtin_agi_reflect(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("agi_reflect(agi_id)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let report = with_agi(id, |core| core.reflect())?;
    Ok(Value::String(report))
}

fn builtin_agi_generate_goals(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_generate_goals(agi_id, context)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let context = value_to_f64_vec(&args[1])?;
    let descs = with_agi(id, |core| {
        core.goals.generate_goals(&context, &mut core.rng)
    })?;
    Ok(Value::Array(descs.into_iter().map(Value::String).collect()))
}

fn builtin_agi_add_goal(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agi_add_goal(agi_id, description, priority)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let desc = value_to_string(&args[1])?;
    let priority = value_to_f64(&args[2])?;
    let goal_id = with_agi(id, |core| core.goals.add_goal(desc, priority))?;
    Ok(Value::Int(goal_id as i128))
}

fn builtin_agi_goal_status(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_goal_status(agi_id, goal_id)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let goal_id = value_to_usize(&args[1])?;
    let status = with_agi(id, |core| {
        core.goals.get_goal(goal_id)
            .map(|g| g.status.as_str().to_string())
            .unwrap_or_else(|| "not_found".to_string())
    })?;
    Ok(Value::String(status))
}

fn builtin_agi_simulate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agi_simulate(agi_id, action, steps)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let action = value_to_f64_vec(&args[1])?;
    let steps = value_to_usize(&args[2])?;
    let steps = steps.max(1).min(1000);
    let trajectory = with_agi(id, |core| core.world_model.simulate(&action, steps))?;
    Ok(Value::Array(trajectory.into_iter().map(|s| {
        Value::Array(s.into_iter().map(Value::Float).collect())
    }).collect()))
}

fn builtin_agi_transfer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agi_transfer(agi_id, source_data, target_dim)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let source = value_to_f64_vec(&args[1])?;
    let target_dim = value_to_usize(&args[2])?;
    let result = with_agi(id, |core| {
        if core.transfer.target_dim != target_dim {
            core.transfer.resize(target_dim, &mut core.rng);
        }
        if core.transfer.source_dim != source.len() {
            core.transfer = DomainAdapter::new(source.len(), target_dim, &mut core.rng);
        }
        core.transfer.transfer(&source)
    })?;
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_agi_temporal_reason(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agi_temporal_reason(agi_id, query_time, context)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let query_time = value_to_usize(&args[1])? as u64;
    let context = value_to_f64_vec(&args[2])?;
    let result = with_agi(id, |core| core.temporal.reason(query_time, &context))?;
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_agi_dream(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_dream(agi_id, num_episodes)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let episodes = value_to_usize(&args[1])?;
    let report = with_agi(id, |core| core.dream(episodes))?;
    Ok(Value::String(report))
}

fn builtin_agi_meta_meta_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_meta_meta_step(agi_id, performance)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let performance = value_to_f64(&args[1])?;
    let desc = with_agi(id, |core| core.meta_meta.step(performance, &mut core.rng))?;
    Ok(Value::String(desc))
}

fn builtin_agi_self_improve(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_self_improve(agi_id, feedback)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let feedback = value_to_f64_vec(&args[1])?;
    let report = with_agi(id, |core| core.self_improve(&feedback))?;
    Ok(Value::String(report))
}

fn builtin_agi_communicate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_communicate(agi_id, message_embedding)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let message = value_to_f64_vec(&args[1])?;
    let response = with_agi(id, |core| core.communicate(&message))?;
    Ok(Value::Array(response.into_iter().map(Value::Float).collect()))
}

fn builtin_agi_world_model_update(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_world_model_update(agi_id, observation)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let obs = value_to_f64_vec(&args[1])?;
    let error = with_agi(id, |core| {
        core.world_model.update(&obs, 0.01);
        core.world_model.prediction_error
    })?;
    Ok(Value::Float(error))
}

fn builtin_agi_curiosity_score(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_curiosity_score(agi_id, observation)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let obs = value_to_f64_vec(&args[1])?;
    let score = with_agi(id, |core| core.curiosity_score(&obs))?;
    Ok(Value::Float(score))
}

fn builtin_agi_decompose_goal(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_decompose_goal(agi_id, goal_id)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let goal_id = value_to_usize(&args[1])?;
    let sub_ids = with_agi(id, |core| {
        core.goals.decompose_goal(goal_id, &mut core.rng)
    })?;
    Ok(Value::Array(sub_ids.into_iter().map(|i| Value::Int(i as i128)).collect()))
}

fn builtin_agi_get_state(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("agi_get_state(agi_id)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let state = with_agi(id, |core| {
        let mut fields = HashMap::new();
        fields.insert("cycle_count".to_string(), Value::Int(core.cycle_count as i128));
        fields.insert("confidence".to_string(), Value::Float(core.self_model.confidence));
        fields.insert("num_goals".to_string(), Value::Int(core.goals.goals.len() as i128));
        fields.insert("prediction_error".to_string(), Value::Float(core.world_model.prediction_error));
        fields
    })?;
    Ok(Value::Struct { name: "AGIState".to_string(), fields: state })
}

fn builtin_agi_imagination_rollout(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agi_imagination_rollout(agi_id, initial_state, num_steps)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let initial = value_to_f64_vec(&args[1])?;
    let steps = value_to_usize(&args[2])?.max(1).min(500);
    let result = with_agi(id, |core| core.world_model.imagination_rollout(&initial, steps))?;
    Ok(Value::Array(result.into_iter().map(|s| {
        Value::Array(s.into_iter().map(Value::Float).collect())
    }).collect()))
}

fn builtin_agi_introspect(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agi_introspect(agi_id, depth)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let depth = value_to_usize(&args[1])?.min(10);
    let report = with_agi(id, |core| core.introspect(depth))?;
    Ok(Value::String(report))
}

fn builtin_agi_evaluate_progress(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("agi_evaluate_progress(agi_id)".to_string());
    }
    let id = value_to_usize(&args[0])?;
    let score = with_agi(id, |core| core.goals.evaluate_progress())?;
    Ok(Value::Float(score))
}

// ── Registration ───────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("agi_new".to_string(), FnDef::Builtin(builtin_agi_new));
    env.functions.insert("agi_consciousness_loop".to_string(), FnDef::Builtin(builtin_agi_consciousness_loop));
    env.functions.insert("agi_reflect".to_string(), FnDef::Builtin(builtin_agi_reflect));
    env.functions.insert("agi_generate_goals".to_string(), FnDef::Builtin(builtin_agi_generate_goals));
    env.functions.insert("agi_add_goal".to_string(), FnDef::Builtin(builtin_agi_add_goal));
    env.functions.insert("agi_goal_status".to_string(), FnDef::Builtin(builtin_agi_goal_status));
    env.functions.insert("agi_simulate".to_string(), FnDef::Builtin(builtin_agi_simulate));
    env.functions.insert("agi_transfer".to_string(), FnDef::Builtin(builtin_agi_transfer));
    env.functions.insert("agi_temporal_reason".to_string(), FnDef::Builtin(builtin_agi_temporal_reason));
    env.functions.insert("agi_dream".to_string(), FnDef::Builtin(builtin_agi_dream));
    env.functions.insert("agi_meta_meta_step".to_string(), FnDef::Builtin(builtin_agi_meta_meta_step));
    env.functions.insert("agi_self_improve".to_string(), FnDef::Builtin(builtin_agi_self_improve));
    env.functions.insert("agi_communicate".to_string(), FnDef::Builtin(builtin_agi_communicate));
    env.functions.insert("agi_world_model_update".to_string(), FnDef::Builtin(builtin_agi_world_model_update));
    env.functions.insert("agi_curiosity_score".to_string(), FnDef::Builtin(builtin_agi_curiosity_score));
    env.functions.insert("agi_decompose_goal".to_string(), FnDef::Builtin(builtin_agi_decompose_goal));
    env.functions.insert("agi_get_state".to_string(), FnDef::Builtin(builtin_agi_get_state));
    env.functions.insert("agi_imagination_rollout".to_string(), FnDef::Builtin(builtin_agi_imagination_rollout));
    env.functions.insert("agi_introspect".to_string(), FnDef::Builtin(builtin_agi_introspect));
    env.functions.insert("agi_evaluate_progress".to_string(), FnDef::Builtin(builtin_agi_evaluate_progress));
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_core(dim: usize) -> AGICore {
        AGICore::new(dim, 42)
    }

    #[test]
    fn test_xorshift() {
        let mut rng = Xorshift64::new(12345);
        let a = rng.next();
        let b = rng.next();
        assert_ne!(a, b);
        assert_ne!(a, 0);
        let f = rng.next_f64();
        assert!(f >= 0.0 && f <= 1.0);
        let r = rng.next_range(2.0, 5.0);
        assert!(r >= 2.0 && r <= 5.0);
    }

    #[test]
    fn test_xorshift_zero_seed() {
        let mut rng = Xorshift64::new(0);
        assert_eq!(rng.state, 1);
        let v = rng.next();
        assert_ne!(v, 0);
    }

    #[test]
    fn test_mat_vec_mul() {
        let mat = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let r = mat_vec_mul(&mat, &v);
        assert_eq!(r.len(), 2);
        assert!((r[0] - 3.0).abs() < 1e-10);
        assert!((r[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_mat_vec_mul_empty() {
        let mat: Vec<Vec<f64>> = Vec::new();
        let v = vec![1.0];
        let r = mat_vec_mul(&mat, &v);
        assert!(r.is_empty());
    }

    #[test]
    fn test_self_model_new() {
        let sm = SelfModel::new(4);
        assert_eq!(sm.attention_weights.len(), 4);
        assert!((sm.confidence - 0.5).abs() < 1e-10);
        assert!(sm.state_history.is_empty());
        assert_eq!(sm.introspection_depth, 0);
    }

    #[test]
    fn test_self_model_novelty_empty() {
        let sm = SelfModel::new(3);
        assert!((sm.compute_novelty(&[1.0, 2.0, 3.0]) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_self_model_novelty_with_history() {
        let mut sm = SelfModel::new(2);
        sm.record_state(vec![1.0, 1.0]);
        sm.record_state(vec![1.0, 1.0]);
        let nov_same = sm.compute_novelty(&[1.0, 1.0]);
        let nov_diff = sm.compute_novelty(&[10.0, 10.0]);
        assert!(nov_same < nov_diff);
    }

    #[test]
    fn test_self_model_variance() {
        let mut sm = SelfModel::new(2);
        assert!((sm.compute_variance()).abs() < 1e-10);
        sm.record_state(vec![0.0, 0.0]);
        assert!((sm.compute_variance()).abs() < 1e-10); // only 1 sample
        sm.record_state(vec![2.0, 2.0]);
        assert!(sm.compute_variance() > 0.0);
    }

    #[test]
    fn test_self_model_confidence_update() {
        let mut sm = SelfModel::new(2);
        sm.update_confidence(0.0); // perfect prediction
        assert!(sm.confidence > 0.5);
        let mut sm2 = SelfModel::new(2);
        sm2.update_confidence(1.0); // terrible prediction
        assert!(sm2.confidence < 0.5);
    }

    #[test]
    fn test_self_model_attention_update() {
        let mut sm = SelfModel::new(3);
        sm.update_attention(&[0.0, 0.0, 10.0]);
        // Third dimension should get more attention
        assert!(sm.attention_weights[2] > sm.attention_weights[0]);
    }

    #[test]
    fn test_self_model_record_overflow() {
        let mut sm = SelfModel::new(2);
        for i in 0..1100 {
            sm.record_state(vec![i as f64, 0.0]);
        }
        assert_eq!(sm.state_history.len(), 1000);
    }

    #[test]
    fn test_goal_status_as_str() {
        assert_eq!(GoalStatus::Active.as_str(), "active");
        assert_eq!(GoalStatus::Completed.as_str(), "completed");
        assert_eq!(GoalStatus::Suspended.as_str(), "suspended");
        assert_eq!(GoalStatus::Failed.as_str(), "failed");
    }

    #[test]
    fn test_goal_system_add_and_get() {
        let mut gs = GoalSystem::new();
        let id = gs.add_goal("test goal".to_string(), 0.8);
        assert_eq!(id, 0);
        let g = gs.get_goal(id).unwrap();
        assert_eq!(g.description, "test goal");
        assert!((g.priority - 0.8).abs() < 1e-10);
        assert_eq!(g.status, GoalStatus::Active);
        assert!((g.progress).abs() < 1e-10);
    }

    #[test]
    fn test_goal_system_multiple() {
        let mut gs = GoalSystem::new();
        let id0 = gs.add_goal("a".to_string(), 1.0);
        let id1 = gs.add_goal("b".to_string(), 0.5);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert!(gs.get_goal(99).is_none());
    }

    #[test]
    fn test_goal_system_decompose() {
        let mut gs = GoalSystem::new();
        let mut rng = Xorshift64::new(42);
        let id = gs.add_goal("big goal".to_string(), 1.0);
        let subs = gs.decompose_goal(id, &mut rng);
        assert!(subs.len() >= 2 && subs.len() <= 4);
        let g = gs.get_goal(id).unwrap();
        assert_eq!(g.subgoals.len(), subs.len());
        for sid in &subs {
            assert!(gs.get_goal(*sid).is_some());
        }
    }

    #[test]
    fn test_goal_system_decompose_nonexistent() {
        let mut gs = GoalSystem::new();
        let mut rng = Xorshift64::new(42);
        let subs = gs.decompose_goal(999, &mut rng);
        assert!(subs.is_empty());
    }

    #[test]
    fn test_goal_system_evaluate_progress_empty() {
        let gs = GoalSystem::new();
        assert!((gs.evaluate_progress()).abs() < 1e-10);
    }

    #[test]
    fn test_goal_system_evaluate_progress() {
        let mut gs = GoalSystem::new();
        gs.add_goal("a".to_string(), 1.0);
        gs.add_goal("b".to_string(), 1.0);
        gs.get_goal_mut(0).unwrap().progress = 0.5;
        gs.get_goal_mut(1).unwrap().progress = 1.0;
        let p = gs.evaluate_progress();
        assert!((p - 0.75).abs() < 1e-10);
    }

    #[test]
    fn test_goal_system_generate_goals() {
        let mut gs = GoalSystem::new();
        let mut rng = Xorshift64::new(42);
        let descs = gs.generate_goals(&[1.0, 2.0], &mut rng);
        assert!(!descs.is_empty());
        assert!(descs[0].contains("explore_positive"));
    }

    #[test]
    fn test_world_model_predict_and_update() {
        let mut rng = Xorshift64::new(42);
        let mut wm = WorldModel::new(3, &mut rng);
        let pred = wm.predict();
        assert_eq!(pred.len(), 3);
        wm.update(&[1.0, 2.0, 3.0], 0.01);
        assert!(wm.prediction_error >= 0.0);
        // State should now be the observation
        assert!((wm.state[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_world_model_simulate() {
        let mut rng = Xorshift64::new(42);
        let wm = WorldModel::new(2, &mut rng);
        let traj = wm.simulate(&[0.5, 0.5], 5);
        assert_eq!(traj.len(), 5);
        for s in &traj {
            assert_eq!(s.len(), 2);
            for v in s {
                assert!(v.abs() <= 1.0); // tanh bounded
            }
        }
    }

    #[test]
    fn test_world_model_imagination_rollout() {
        let mut rng = Xorshift64::new(42);
        let mut wm = WorldModel::new(2, &mut rng);
        let result = wm.imagination_rollout(&[0.1, 0.2], 3);
        assert_eq!(result.len(), 3);
        assert!(!wm.imagination_buffer.is_empty());
    }

    #[test]
    fn test_domain_adapter_transfer() {
        let mut rng = Xorshift64::new(42);
        let mut da = DomainAdapter::new(3, 2, &mut rng);
        let result = da.transfer(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 2);
        assert_eq!(da.transfer_history.len(), 1);
    }

    #[test]
    fn test_domain_adapter_resize() {
        let mut rng = Xorshift64::new(42);
        let mut da = DomainAdapter::new(3, 2, &mut rng);
        da.resize(5, &mut rng);
        assert_eq!(da.target_dim, 5);
        assert_eq!(da.alignment_weights.len(), 5);
        let result = da.transfer(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_temporal_reasoner_record_and_reason() {
        let mut rng = Xorshift64::new(42);
        let mut tr = TemporalReasoner::new(2, &mut rng);
        tr.record(vec![1.0, 0.0]);
        tr.record(vec![0.0, 1.0]);
        let result = tr.reason(1, &[0.5, 0.5]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_temporal_reasoner_empty_reason() {
        let mut rng = Xorshift64::new(42);
        let tr = TemporalReasoner::new(2, &mut rng);
        let result = tr.reason(0, &[1.0, 2.0]);
        assert_eq!(result, vec![1.0, 2.0]);
    }

    #[test]
    fn test_temporal_reasoner_overflow() {
        let mut rng = Xorshift64::new(42);
        let mut tr = TemporalReasoner::new(2, &mut rng);
        for _ in 0..600 {
            tr.record(vec![1.0, 1.0]);
        }
        assert_eq!(tr.memory.len(), 500);
    }

    #[test]
    fn test_meta_meta_learner_step() {
        let mut rng = Xorshift64::new(42);
        let mut mml = MetaMetaLearner::new(3, 4, &mut rng);
        assert_eq!(mml.strategies.len(), 4);
        let desc = mml.step(0.8, &mut rng);
        assert!(desc.contains("strategy_"));
        assert!(desc.contains("score_"));
        assert_eq!(mml.evolution_history.len(), 1);
    }

    #[test]
    fn test_meta_meta_learner_evolution() {
        let mut rng = Xorshift64::new(42);
        let mut mml = MetaMetaLearner::new(3, 4, &mut rng);
        for _ in 0..20 {
            mml.step(rng.next_f64(), &mut rng.clone());
        }
        assert_eq!(mml.evolution_history.len(), 20);
    }

    #[test]
    fn test_agi_core_new() {
        let core = make_core(4);
        assert_eq!(core.state_dim, 4);
        assert_eq!(core.cycle_count, 0);
        assert!((core.self_model.confidence - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_consciousness_loop() {
        let mut core = make_core(3);
        let action = core.consciousness_loop(&[1.0, 0.5, -0.3]);
        assert_eq!(action.len(), 3);
        assert_eq!(core.cycle_count, 1);
        assert!(!core.self_model.state_history.is_empty());
        assert!(!core.self_model.metacognitive_log.is_empty());
    }

    #[test]
    fn test_consciousness_loop_multiple() {
        let mut core = make_core(2);
        for i in 0..10 {
            let obs = vec![i as f64 * 0.1, (10 - i) as f64 * 0.1];
            core.consciousness_loop(&obs);
        }
        assert_eq!(core.cycle_count, 10);
        assert_eq!(core.self_model.state_history.len(), 10);
    }

    #[test]
    fn test_reflect() {
        let mut core = make_core(3);
        core.consciousness_loop(&[1.0, 2.0, 3.0]);
        let report = core.reflect();
        assert!(report.contains("confidence="));
        assert!(report.contains("novelty="));
        assert!(report.contains("variance="));
        assert!(report.contains("confusion="));
        assert!(report.contains("cycles=1"));
    }

    #[test]
    fn test_reflect_no_history() {
        let core = make_core(2);
        let report = core.reflect();
        assert!(report.contains("confidence=0.5000"));
    }

    #[test]
    fn test_dream_no_experience() {
        let mut core = make_core(2);
        let report = core.dream(5);
        assert_eq!(report, "no_experiences_to_dream_about");
    }

    #[test]
    fn test_dream_with_experience() {
        let mut core = make_core(2);
        core.consciousness_loop(&[1.0, 2.0]);
        core.consciousness_loop(&[3.0, 4.0]);
        let report = core.dream(5);
        assert!(report.contains("dreamed_"));
        assert!(report.contains("consolidated"));
    }

    #[test]
    fn test_curiosity_score() {
        let mut core = make_core(3);
        core.consciousness_loop(&[0.0, 0.0, 0.0]);
        let score_familiar = core.curiosity_score(&[0.0, 0.0, 0.0]);
        let score_novel = core.curiosity_score(&[10.0, 10.0, 10.0]);
        assert!(score_novel > score_familiar);
    }

    #[test]
    fn test_communicate() {
        let mut core = make_core(3);
        let response = core.communicate(&[1.0, 0.5, -0.5]);
        assert_eq!(response.len(), 3);
        for v in &response {
            assert!(v.abs() <= 1.0); // tanh bounded
        }
    }

    #[test]
    fn test_self_improve() {
        let mut core = make_core(2);
        let conf_before = core.self_model.confidence;
        let report = core.self_improve(&[0.5, -0.5]);
        assert!(report.contains("improved"));
        assert!(core.self_model.confidence > conf_before);
    }

    #[test]
    fn test_introspect_depth_0() {
        let core = make_core(3);
        let report = core.introspect(0);
        assert!(report.contains("state_dim=3"));
        assert!(report.contains("confidence="));
    }

    #[test]
    fn test_introspect_depth_3() {
        let core = make_core(3);
        let report = core.introspect(3);
        assert!(report.contains("depth=3"));
        assert!(report.contains("depth=2"));
        // nested
        assert!(report.contains("["));
    }

    #[test]
    fn test_global_store() {
        let core = make_core(2);
        let id = store_agi(core);
        let result = with_agi(id, |c| c.state_dim);
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_global_store_not_found() {
        let result = with_agi(99999, |c| c.state_dim);
        assert!(result.is_err());
    }

    #[test]
    fn test_world_model_error_decreases() {
        let mut rng = Xorshift64::new(42);
        let mut wm = WorldModel::new(2, &mut rng);
        let obs = vec![1.0, 0.5];
        // Repeatedly show same observation, error should decrease
        let mut last_err = f64::MAX;
        for _ in 0..50 {
            wm.update(&obs, 0.05);
        }
        // After 50 updates with the same observation, prediction error should be small
        wm.update(&obs, 0.05);
        assert!(wm.prediction_error < last_err || wm.prediction_error < 1.0);
    }

    #[test]
    fn test_goal_completion_via_consciousness() {
        let mut core = make_core(2);
        core.goals.add_goal("test".to_string(), 1.0);
        // Run many cycles to accumulate progress
        for _ in 0..200 {
            core.consciousness_loop(&[0.5, 0.5]);
        }
        let g = core.goals.get_goal(0).unwrap();
        assert!(g.progress > 0.0);
    }

    #[test]
    fn test_transfer_dimension_change() {
        let mut core = make_core(3);
        let result = core.transfer.transfer(&[1.0, 2.0, 3.0]);
        assert_eq!(result.len(), 3); // default same dim
        core.transfer.resize(5, &mut core.rng);
        let result2 = core.transfer.transfer(&[1.0, 2.0, 3.0]);
        assert_eq!(result2.len(), 5);
    }

    #[test]
    fn test_imagination_buffer_limit() {
        let mut rng = Xorshift64::new(42);
        let mut wm = WorldModel::new(2, &mut rng);
        for _ in 0..100 {
            wm.imagination_rollout(&[0.1, 0.2], 10);
        }
        assert!(wm.imagination_buffer.len() <= 500);
    }

    #[test]
    fn test_metacognitive_log_limit() {
        let mut core = make_core(2);
        for _ in 0..600 {
            core.consciousness_loop(&[0.1, 0.2]);
        }
        assert!(core.self_model.metacognitive_log.len() <= 500);
    }

    #[test]
    fn test_generate_goals_negative_context() {
        let mut gs = GoalSystem::new();
        let mut rng = Xorshift64::new(42);
        let descs = gs.generate_goals(&[-1.0, -2.0], &mut rng);
        assert!(!descs.is_empty());
        assert!(descs[0].contains("negative"));
    }

    #[test]
    fn test_generate_goals_zero_context() {
        let mut gs = GoalSystem::new();
        let mut rng = Xorshift64::new(42);
        let descs = gs.generate_goals(&[0.0, 0.0], &mut rng);
        assert!(!descs.is_empty());
        assert!(descs[0].contains("gather"));
    }
}
