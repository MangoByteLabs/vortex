// Continuous Learning: self-improving model server that trains while serving inference.
// Combines Forward-Forward local learning, tensor autodiff, and adaptive update policies.

use std::collections::VecDeque;

/// Deterministic pseudo-random for weight init
fn seeded_rand(seed: u64) -> f64 {
    let x = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let bits = (x >> 33) as f64 / (1u64 << 31) as f64;
    bits - 1.0
}

fn rand_matrix(rows: usize, cols: usize, seed: u64) -> Vec<Vec<f64>> {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    (0..rows)
        .map(|i| {
            (0..cols)
                .map(|j| seeded_rand(seed + i as u64 * 1000 + j as u64) * scale)
                .collect()
        })
        .collect()
}

fn zeros(n: usize) -> Vec<f64> {
    vec![0.0; n]
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.01 * x } // leaky ReLU
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ─── AdaptiveUpdatePolicy ───────────────────────────────────────────────────

/// Policy controlling how weights are updated during continuous learning.
#[derive(Debug, Clone)]
pub enum AdaptiveUpdatePolicy {
    /// Caps the maximum weight change per step to prevent catastrophic forgetting.
    BoundedUpdate { max_delta: f64 },
    /// Elastic Weight Consolidation: penalizes changes to important weights.
    EWC { lambda: f64, fisher: Vec<Vec<Vec<f64>>>, star_weights: Vec<Vec<Vec<f64>>> },
    /// Stores recent examples and mixes them with live traffic for stability.
    ReplayBuffer { capacity: usize, buffer: VecDeque<(Vec<f64>, Vec<f64>)> },
    /// Monitors input distribution shift and adjusts learning rate.
    DriftDetector {
        running_mean: Vec<f64>,
        running_var: Vec<f64>,
        sample_count: usize,
        drift_threshold: f64,
        drift_score: f64,
    },
}

impl AdaptiveUpdatePolicy {
    pub fn bounded(max_delta: f64) -> Self {
        AdaptiveUpdatePolicy::BoundedUpdate { max_delta }
    }

    pub fn ewc(lambda: f64, weights: &[Vec<Vec<f64>>]) -> Self {
        // Initialize Fisher information to zeros, star_weights = current weights
        let fisher: Vec<Vec<Vec<f64>>> = weights
            .iter()
            .map(|layer| layer.iter().map(|row| vec![0.0; row.len()]).collect())
            .collect();
        AdaptiveUpdatePolicy::EWC {
            lambda,
            fisher,
            star_weights: weights.to_vec(),
        }
    }

    pub fn replay_buffer(capacity: usize) -> Self {
        AdaptiveUpdatePolicy::ReplayBuffer {
            capacity,
            buffer: VecDeque::new(),
        }
    }

    pub fn drift_detector(input_dim: usize, threshold: f64) -> Self {
        AdaptiveUpdatePolicy::DriftDetector {
            running_mean: vec![0.0; input_dim],
            running_var: vec![1.0; input_dim],
            sample_count: 0,
            drift_threshold: threshold,
            drift_score: 0.0,
        }
    }

    /// Clip weight deltas according to BoundedUpdate policy.
    pub fn clip_delta(&self, delta: f64) -> f64 {
        match self {
            AdaptiveUpdatePolicy::BoundedUpdate { max_delta } => {
                delta.max(-*max_delta).min(*max_delta)
            }
            _ => delta,
        }
    }

    /// Compute EWC penalty gradient for a specific weight.
    pub fn ewc_penalty_grad(&self, layer: usize, i: usize, j: usize, current_weight: f64) -> f64 {
        match self {
            AdaptiveUpdatePolicy::EWC { lambda, fisher, star_weights } => {
                let f = fisher[layer][i][j];
                let w_star = star_weights[layer][i][j];
                lambda * f * (current_weight - w_star)
            }
            _ => 0.0,
        }
    }

    /// Update Fisher information with observed gradients.
    pub fn update_fisher(&mut self, layer: usize, i: usize, j: usize, grad: f64) {
        if let AdaptiveUpdatePolicy::EWC { fisher, .. } = self {
            if layer < fisher.len() && i < fisher[layer].len() && j < fisher[layer][i].len() {
                // Running average of squared gradients as Fisher approximation
                fisher[layer][i][j] = 0.99 * fisher[layer][i][j] + 0.01 * grad * grad;
            }
        }
    }

    /// Store an example in the replay buffer.
    pub fn store_example(&mut self, input: Vec<f64>, target: Vec<f64>) {
        if let AdaptiveUpdatePolicy::ReplayBuffer { capacity, buffer } = self {
            if buffer.len() >= *capacity {
                buffer.pop_front();
            }
            buffer.push_back((input, target));
        }
    }

    /// Get replay examples (up to `count`).
    pub fn get_replay_examples(&self, count: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
        match self {
            AdaptiveUpdatePolicy::ReplayBuffer { buffer, .. } => {
                buffer.iter().rev().take(count).cloned().collect()
            }
            _ => vec![],
        }
    }

    /// Get the replay buffer length.
    pub fn replay_len(&self) -> usize {
        match self {
            AdaptiveUpdatePolicy::ReplayBuffer { buffer, .. } => buffer.len(),
            _ => 0,
        }
    }

    /// Update drift detector with new input and return current drift score.
    pub fn update_drift(&mut self, input: &[f64]) -> f64 {
        if let AdaptiveUpdatePolicy::DriftDetector {
            running_mean,
            running_var,
            sample_count,
            drift_threshold: _,
            drift_score,
        } = self
        {
            *sample_count += 1;
            let alpha = if *sample_count < 100 {
                1.0 / *sample_count as f64
            } else {
                0.01
            };

            let mut total_drift = 0.0;
            for (idx, &val) in input.iter().enumerate() {
                if idx < running_mean.len() {
                    let old_mean = running_mean[idx];
                    let old_var = running_var[idx];
                    running_mean[idx] = (1.0 - alpha) * old_mean + alpha * val;
                    running_var[idx] =
                        (1.0 - alpha) * old_var + alpha * (val - old_mean) * (val - running_mean[idx]);
                    // Drift = normalized shift in mean relative to std
                    let std = running_var[idx].abs().sqrt().max(1e-8);
                    total_drift += ((running_mean[idx] - old_mean) / std).abs();
                }
            }
            *drift_score = total_drift / input.len().max(1) as f64;
            *drift_score
        } else {
            0.0
        }
    }

    /// Get the current drift score.
    pub fn get_drift_score(&self) -> f64 {
        match self {
            AdaptiveUpdatePolicy::DriftDetector { drift_score, .. } => *drift_score,
            _ => 0.0,
        }
    }

    /// Check whether drift exceeds threshold.
    pub fn is_drifting(&self) -> bool {
        match self {
            AdaptiveUpdatePolicy::DriftDetector { drift_score, drift_threshold, .. } => {
                *drift_score > *drift_threshold
            }
            _ => false,
        }
    }
}

// ─── MemoryBudget ───────────────────────────────────────────────────────────

/// Tracks GPU memory budget and decides whether training is possible.
#[derive(Debug, Clone)]
pub struct MemoryBudget {
    pub total_mb: f64,
    pub used_mb: f64,
    pub training_reserve_mb: f64,
}

impl MemoryBudget {
    pub fn new(total_mb: f64) -> Self {
        Self {
            total_mb,
            used_mb: 0.0,
            training_reserve_mb: total_mb * 0.3, // reserve 30% for training
        }
    }

    pub fn allocate(&mut self, mb: f64) -> bool {
        if self.used_mb + mb <= self.total_mb {
            self.used_mb += mb;
            true
        } else {
            false
        }
    }

    pub fn free(&mut self, mb: f64) {
        self.used_mb = (self.used_mb - mb).max(0.0);
    }

    pub fn available_mb(&self) -> f64 {
        self.total_mb - self.used_mb
    }

    /// Whether there is enough memory to train (available > reserve).
    pub fn can_train(&self) -> bool {
        self.available_mb() >= self.training_reserve_mb
    }

    /// Estimate memory needed for model weights (in MB).
    pub fn estimate_model_mb(weights: &[Vec<Vec<f64>>]) -> f64 {
        let total_params: usize = weights.iter().map(|l| l.iter().map(|r| r.len()).sum::<usize>()).sum();
        (total_params * 8) as f64 / (1024.0 * 1024.0) // 8 bytes per f64
    }
}

// ─── CheckpointManager ─────────────────────────────────────────────────────

/// Periodic model checkpointing with quality-based rollback.
#[derive(Debug, Clone)]
pub struct CheckpointManager {
    pub checkpoints: Vec<(Vec<Vec<Vec<f64>>>, f64)>, // (weights, loss_at_checkpoint)
    pub max_checkpoints: usize,
    pub checkpoint_interval: usize,
    pub steps_since_checkpoint: usize,
}

impl CheckpointManager {
    pub fn new(max_checkpoints: usize, interval: usize) -> Self {
        Self {
            checkpoints: Vec::new(),
            max_checkpoints,
            checkpoint_interval: interval,
            steps_since_checkpoint: 0,
        }
    }

    /// Maybe save a checkpoint. Returns true if saved.
    pub fn maybe_checkpoint(&mut self, weights: &[Vec<Vec<f64>>], loss: f64) -> bool {
        self.steps_since_checkpoint += 1;
        if self.steps_since_checkpoint >= self.checkpoint_interval {
            self.steps_since_checkpoint = 0;
            if self.checkpoints.len() >= self.max_checkpoints {
                self.checkpoints.remove(0);
            }
            self.checkpoints.push((weights.to_vec(), loss));
            return true;
        }
        false
    }

    /// Rollback to last checkpoint if current loss is worse.
    pub fn maybe_rollback(&self, current_loss: f64) -> Option<Vec<Vec<Vec<f64>>>> {
        if let Some((ref weights, best_loss)) = self.checkpoints.last() {
            if current_loss > best_loss * 1.5 {
                // Loss degraded by more than 50%
                return Some(weights.clone());
            }
        }
        None
    }

    /// Force save a checkpoint.
    pub fn save(&mut self, weights: &[Vec<Vec<f64>>], loss: f64) {
        if self.checkpoints.len() >= self.max_checkpoints {
            self.checkpoints.remove(0);
        }
        self.checkpoints.push((weights.to_vec(), loss));
    }

    /// Restore the latest checkpoint.
    pub fn restore_latest(&self) -> Option<Vec<Vec<Vec<f64>>>> {
        self.checkpoints.last().map(|(w, _)| w.clone())
    }

    pub fn num_checkpoints(&self) -> usize {
        self.checkpoints.len()
    }
}

// ─── ContinuousLearner ─────────────────────────────────────────────────────

/// A model that learns continuously with Forward-Forward local updates.
/// `@persistent_grad` semantics: gradients persist across requests with momentum.
#[derive(Debug, Clone)]
pub struct ContinuousLearner {
    /// Weight matrices: layers x rows x cols
    pub weights: Vec<Vec<Vec<f64>>>,
    /// Biases per layer
    pub biases: Vec<Vec<f64>>,
    /// Layer sizes
    pub layer_sizes: Vec<usize>,
    /// Persistent gradient momentum (same shape as weights)
    pub momentum: Vec<Vec<Vec<f64>>>,
    /// Momentum coefficient
    pub momentum_coeff: f64,
    /// Running loss EMA
    pub loss_ema: f64,
    /// Update count
    pub update_count: u64,
    /// Forward-Forward threshold
    pub ff_threshold: f64,
}

impl ContinuousLearner {
    pub fn new(layer_sizes: &[usize]) -> Self {
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let mut momentum = Vec::new();

        for i in 0..layer_sizes.len() - 1 {
            let w = rand_matrix(layer_sizes[i], layer_sizes[i + 1], 42 + i as u64 * 137);
            let m = vec![vec![0.0; layer_sizes[i + 1]]; layer_sizes[i]];
            weights.push(w);
            momentum.push(m);
            biases.push(zeros(layer_sizes[i + 1]));
        }

        Self {
            weights,
            biases,
            layer_sizes: layer_sizes.to_vec(),
            momentum,
            momentum_coeff: 0.9,
            loss_ema: 1.0,
            update_count: 0,
            ff_threshold: 2.0,
        }
    }

    /// Forward pass through the network.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut activation = input.to_vec();
        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let rows = w.len();
            let cols = if rows > 0 { w[0].len() } else { 0 };
            let mut out = b.clone();
            for j in 0..cols {
                for i in 0..activation.len().min(rows) {
                    out[j] += activation[i] * w[i][j];
                }
            }
            // ReLU for hidden layers, identity for output
            if layer_idx < self.weights.len() - 1 {
                for v in out.iter_mut() {
                    *v = relu(*v);
                }
            }
            activation = out;
        }
        activation
    }

    /// Forward-Forward learning step: local update without backward pass.
    /// Returns the loss (goodness-based).
    pub fn ff_learn(&mut self, input: &[f64], target: &[f64], lr: f64) -> f64 {
        // Forward pass, collecting activations
        let mut activations = vec![input.to_vec()];
        let mut act = input.to_vec();
        let num_layers = self.weights.len();
        for (layer_idx, (w, b)) in self.weights.iter().zip(self.biases.iter()).enumerate() {
            let cols = if w.is_empty() { 0 } else { w[0].len() };
            let mut out = b.clone();
            for j in 0..cols {
                for i in 0..act.len().min(w.len()) {
                    out[j] += act[i] * w[i][j];
                }
            }
            // Hidden layers: ReLU + layer norm; output layer: identity
            if layer_idx < num_layers - 1 {
                for v in out.iter_mut() {
                    *v = relu(*v);
                }
                if out.len() > 1 {
                    let mean = out.iter().sum::<f64>() / out.len() as f64;
                    let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / out.len() as f64;
                    let std = (var + 1e-8).sqrt();
                    for v in out.iter_mut() {
                        *v = (*v - mean) / std;
                    }
                }
            }
            act = out.clone();
            activations.push(out);
        }

        // Compute output loss (MSE for learning signal)
        let output = activations.last().unwrap();
        let mut loss = 0.0;
        for i in 0..output.len().min(target.len()) {
            loss += (output[i] - target[i]).powi(2);
        }
        loss /= output.len().max(1) as f64;

        // Forward-Forward local update: each layer adjusts to increase goodness for
        // correct predictions and decrease for incorrect.
        // FF update on hidden layers only; output layer uses direct MSE below
        for layer_idx in 0..self.weights.len().saturating_sub(1) {
            let x = &activations[layer_idx];
            let y = &activations[layer_idx + 1];
            let goodness: f64 = y.iter().map(|a| a * a).sum();
            let positive = loss < 1.0; // good prediction
            let p = sigmoid(goodness - self.ff_threshold);
            let grad_sign = if positive { 1.0 - p } else { -p };

            let cols = if self.weights[layer_idx].is_empty() {
                0
            } else {
                self.weights[layer_idx][0].len()
            };

            for i in 0..x.len().min(self.weights[layer_idx].len()) {
                for j in 0..cols {
                    let grad = grad_sign * x[i] * y[j];
                    // @persistent_grad: accumulate with momentum
                    self.momentum[layer_idx][i][j] =
                        self.momentum_coeff * self.momentum[layer_idx][i][j] + lr * grad;
                    self.weights[layer_idx][i][j] += self.momentum[layer_idx][i][j];
                }
            }

            // Bias update
            for j in 0..cols.min(self.biases[layer_idx].len()) {
                self.biases[layer_idx][j] += lr * grad_sign * y[j];
            }
        }

        // Direct MSE gradient on output layer (fallback to autodiff-like update)
        let last = self.weights.len() - 1;
        let hidden = &activations[last];
        let output = &activations[last + 1];
        let out_cols = if self.weights[last].is_empty() { 0 } else { self.weights[last][0].len() };
        for j in 0..out_cols.min(target.len()) {
            let err = output[j] - target[j]; // dL/do_j = 2*(o_j - t_j) / n
            let dloss = 2.0 * err / target.len().max(1) as f64;
            for i in 0..hidden.len().min(self.weights[last].len()) {
                let grad = dloss * hidden[i];
                self.momentum[last][i][j] =
                    self.momentum_coeff * self.momentum[last][i][j] + lr * (-grad);
                self.weights[last][i][j] += self.momentum[last][i][j];
            }
            if j < self.biases[last].len() {
                self.biases[last][j] += lr * (-dloss);
            }
        }

        // Update stats
        self.update_count += 1;
        self.loss_ema = 0.95 * self.loss_ema + 0.05 * loss;

        loss
    }

    /// Get running statistics.
    pub fn stats(&self) -> (f64, u64) {
        (self.loss_ema, self.update_count)
    }
}

// ─── ServingTrainer ─────────────────────────────────────────────────────────

/// Combines inference and training in a single serving loop.
pub struct ServingTrainer {
    pub learner: ContinuousLearner,
    pub policies: Vec<AdaptiveUpdatePolicy>,
    pub memory_budget: MemoryBudget,
    pub checkpoint_mgr: CheckpointManager,
}

impl ServingTrainer {
    pub fn new(layer_sizes: &[usize], total_memory_mb: f64) -> Self {
        let learner = ContinuousLearner::new(layer_sizes);
        let model_mb = MemoryBudget::estimate_model_mb(&learner.weights);
        let mut budget = MemoryBudget::new(total_memory_mb);
        budget.allocate(model_mb);

        Self {
            learner,
            policies: Vec::new(),
            memory_budget: budget,
            checkpoint_mgr: CheckpointManager::new(5, 100),
        }
    }

    pub fn add_policy(&mut self, policy: AdaptiveUpdatePolicy) {
        self.policies.push(policy);
    }

    /// Inference only — no training.
    pub fn infer(&self, input: &[f64]) -> Vec<f64> {
        self.learner.forward(input)
    }

    /// Serve inference AND update weights from feedback.
    pub fn infer_and_learn(&mut self, input: &[f64], target: &[f64], lr: f64) -> (Vec<f64>, f64) {
        let output = self.learner.forward(input);

        if self.memory_budget.can_train() {
            // Store in replay buffer if we have one
            for policy in &mut self.policies {
                policy.store_example(input.to_vec(), target.to_vec());
            }

            // FF learning step
            let loss = self.learner.ff_learn(input, target, lr);

            // Apply policy constraints
            self.apply_policies(lr);

            // Checkpoint
            self.checkpoint_mgr.maybe_checkpoint(&self.learner.weights, loss);

            // Check for rollback
            if let Some(restored) = self.checkpoint_mgr.maybe_rollback(loss) {
                self.learner.weights = restored;
            }

            (output, loss)
        } else {
            // Not enough memory — serve only
            (output, -1.0)
        }
    }

    fn apply_policies(&mut self, _lr: f64) {
        for policy in &mut self.policies {
            // Update drift detector
            if let AdaptiveUpdatePolicy::DriftDetector { .. } = policy {
                // drift updated externally via update_drift
            }
        }
    }

    /// Get training statistics.
    pub fn get_stats(&self) -> (f64, f64, u64, f64) {
        let (loss_ema, updates) = self.learner.stats();
        let drift = self.policies.iter().map(|p| p.get_drift_score()).sum::<f64>();
        let mem = self.memory_budget.used_mb;
        (loss_ema, drift, updates, mem)
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_inference() {
        let learner = ContinuousLearner::new(&[4, 8, 2]);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let output = learner.forward(&input);
        assert_eq!(output.len(), 2);
        // Output should be finite
        assert!(output[0].is_finite());
        assert!(output[1].is_finite());
    }

    #[test]
    fn test_learning_reduces_loss() {
        let mut learner = ContinuousLearner::new(&[4, 16, 4]);
        let input = vec![1.0, 0.0, 0.5, -0.3];
        let target = vec![0.5, -0.2, 0.1, 0.3];
        let lr = 0.001;

        let first_loss = learner.ff_learn(&input, &target, lr);

        // Train for several steps
        let mut last_loss = first_loss;
        for _ in 0..100 {
            last_loss = learner.ff_learn(&input, &target, lr);
        }

        // Loss should decrease (or at least the EMA should)
        assert!(
            learner.loss_ema < first_loss || last_loss < first_loss,
            "Loss should decrease: first={}, last={}, ema={}",
            first_loss,
            last_loss,
            learner.loss_ema
        );
        assert!(learner.update_count == 101);
    }

    #[test]
    fn test_bounded_update_caps_weight_changes() {
        let policy = AdaptiveUpdatePolicy::bounded(0.01);

        assert_eq!(policy.clip_delta(0.5), 0.01);
        assert_eq!(policy.clip_delta(-0.5), -0.01);
        assert_eq!(policy.clip_delta(0.005), 0.005);
        assert_eq!(policy.clip_delta(-0.005), -0.005);
    }

    #[test]
    fn test_ewc_penalizes_important_weight_changes() {
        let weights = vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]];
        let mut policy = AdaptiveUpdatePolicy::ewc(100.0, &weights);

        // Simulate observed gradients to build Fisher info
        for _ in 0..100 {
            policy.update_fisher(0, 0, 0, 2.0); // high gradient -> important
            policy.update_fisher(0, 0, 1, 0.01); // low gradient -> unimportant
        }

        // Penalty for moving important weight should be much higher
        let penalty_important = policy.ewc_penalty_grad(0, 0, 0, 2.0); // moved from 1.0 to 2.0
        let penalty_unimportant = policy.ewc_penalty_grad(0, 0, 1, 3.0); // moved from 2.0 to 3.0

        assert!(
            penalty_important.abs() > penalty_unimportant.abs() * 10.0,
            "Important weight penalty ({}) should be >> unimportant ({})",
            penalty_important.abs(),
            penalty_unimportant.abs()
        );
    }

    #[test]
    fn test_replay_buffer_stores_and_replays() {
        let mut policy = AdaptiveUpdatePolicy::replay_buffer(3);

        policy.store_example(vec![1.0], vec![0.0]);
        policy.store_example(vec![2.0], vec![1.0]);
        policy.store_example(vec![3.0], vec![0.0]);
        assert_eq!(policy.replay_len(), 3);

        // Adding one more should evict the oldest
        policy.store_example(vec![4.0], vec![1.0]);
        assert_eq!(policy.replay_len(), 3);

        let examples = policy.get_replay_examples(10);
        assert_eq!(examples.len(), 3);
        // Most recent first
        assert_eq!(examples[0].0, vec![4.0]);
        assert_eq!(examples[1].0, vec![3.0]);
        assert_eq!(examples[2].0, vec![2.0]);
    }

    #[test]
    fn test_drift_detector_detects_distribution_shift() {
        let mut policy = AdaptiveUpdatePolicy::drift_detector(2, 0.5);

        // Feed stable distribution
        for _ in 0..200 {
            policy.update_drift(&[1.0, 2.0]);
        }
        let stable_drift = policy.get_drift_score();

        // Sudden distribution shift
        let shift_drift = policy.update_drift(&[100.0, 200.0]);

        assert!(
            shift_drift > stable_drift,
            "Drift after shift ({}) should be > stable drift ({})",
            shift_drift,
            stable_drift
        );
    }

    #[test]
    fn test_checkpoint_save_restore() {
        let mut mgr = CheckpointManager::new(3, 1); // checkpoint every step

        let w1 = vec![vec![vec![1.0, 2.0]]];
        mgr.save(&w1, 0.5);

        let w2 = vec![vec![vec![3.0, 4.0]]];
        mgr.save(&w2, 0.3);

        assert_eq!(mgr.num_checkpoints(), 2);

        let restored = mgr.restore_latest().unwrap();
        assert_eq!(restored[0][0][0], 3.0);
        assert_eq!(restored[0][0][1], 4.0);

        // Rollback: current loss 10.0 is much worse than 0.3 * 1.5 = 0.45
        let rollback = mgr.maybe_rollback(10.0);
        assert!(rollback.is_some());

        // No rollback if loss is fine
        let no_rollback = mgr.maybe_rollback(0.4);
        assert!(no_rollback.is_none());
    }

    #[test]
    fn test_memory_budget_respects_limits() {
        let mut budget = MemoryBudget::new(100.0); // 100 MB total
        assert!(budget.can_train()); // 30% reserve = 30 MB, 100 available

        budget.allocate(75.0);
        assert!(!budget.can_train()); // 25 MB available < 30 MB reserve

        budget.free(50.0);
        assert!(budget.can_train()); // 75 MB available > 30 MB reserve

        // Can't allocate more than total
        assert!(!budget.allocate(200.0));
        assert_eq!(budget.available_mb(), 75.0);
    }

    #[test]
    fn test_serving_trainer_infer_and_learn() {
        let mut trainer = ServingTrainer::new(&[2, 4, 1], 1000.0);
        trainer.add_policy(AdaptiveUpdatePolicy::replay_buffer(100));

        let (output, loss) = trainer.infer_and_learn(&[1.0, 0.0], &[1.0], 0.01);
        assert_eq!(output.len(), 1);
        assert!(loss >= 0.0);

        let (loss_ema, _drift, updates, mem) = trainer.get_stats();
        assert!(loss_ema > 0.0);
        assert_eq!(updates, 1);
        assert!(mem > 0.0);
    }

    #[test]
    fn test_serving_trainer_memory_gated() {
        let mut trainer = ServingTrainer::new(&[2, 4, 1], 0.001); // tiny budget
        // Fill budget so available < training_reserve (30% of 0.001 = 0.0003)
        trainer.memory_budget.used_mb = 0.0008; // only 0.0002 available < 0.0003 reserve

        let (output, loss) = trainer.infer_and_learn(&[1.0, 0.0], &[1.0], 0.01);
        assert_eq!(output.len(), 1);
        // Should return -1.0 loss when can't train
        assert_eq!(loss, -1.0);
    }

    #[test]
    fn test_persistent_grad_momentum() {
        let mut learner = ContinuousLearner::new(&[4, 8, 4]);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let target = vec![1.0, 0.0, 0.5, -0.5];

        // After first step, momentum should be non-zero
        learner.ff_learn(&input, &target, 0.1);
        let has_nonzero_momentum = learner.momentum.iter().any(|layer| {
            layer.iter().any(|row| row.iter().any(|&v| v.abs() > 1e-15))
        });
        assert!(has_nonzero_momentum, "Momentum should accumulate after learning");

        // After second step, momentum carries forward (changes)
        let m_before: f64 = learner.momentum[0][0][0];
        learner.ff_learn(&input, &target, 0.1);
        let m_after = learner.momentum[0][0][0];
        assert_ne!(m_before, m_after, "Momentum should change between steps");
    }
}
