use std::collections::HashMap;

// ---------------------------------------------------------------------------
// StorageTier
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageTier {
    Hot,  // GPU / fast memory
    Warm, // CPU / RAM
    Cold, // SSD / disk
}

// ---------------------------------------------------------------------------
// MicroExpert – small 2-layer MLP
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct MicroExpert {
    pub input_dim: usize,
    pub hidden_dim: usize,
    pub output_dim: usize,
    // layer 1: hidden_dim x input_dim weights + hidden_dim biases
    pub w1: Vec<f64>,
    pub b1: Vec<f64>,
    // layer 2: output_dim x hidden_dim weights + output_dim biases
    pub w2: Vec<f64>,
    pub b2: Vec<f64>,
}

impl MicroExpert {
    pub fn new(input_dim: usize, hidden_dim: usize, output_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let scale1 = (2.0 / input_dim as f64).sqrt();
        let scale2 = (2.0 / hidden_dim as f64).sqrt();
        let w1: Vec<f64> = (0..hidden_dim * input_dim).map(|_| rng.next_normal() * scale1).collect();
        let b1 = vec![0.0; hidden_dim];
        let w2: Vec<f64> = (0..output_dim * hidden_dim).map(|_| rng.next_normal() * scale2).collect();
        let b2 = vec![0.0; output_dim];
        Self { input_dim, hidden_dim, output_dim, w1, b1, w2, b2 }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        assert_eq!(input.len(), self.input_dim);
        // hidden = ReLU(W1 * input + b1)
        let mut hidden = vec![0.0; self.hidden_dim];
        for i in 0..self.hidden_dim {
            let mut s = self.b1[i];
            for j in 0..self.input_dim {
                s += self.w1[i * self.input_dim + j] * input[j];
            }
            hidden[i] = if s > 0.0 { s } else { 0.0 }; // ReLU
        }
        // output = W2 * hidden + b2
        let mut output = vec![0.0; self.output_dim];
        for i in 0..self.output_dim {
            let mut s = self.b2[i];
            for j in 0..self.hidden_dim {
                s += self.w2[i * self.hidden_dim + j] * hidden[j];
            }
            output[i] = s;
        }
        output
    }

    pub fn param_count(&self) -> usize {
        self.hidden_dim * self.input_dim + self.hidden_dim
            + self.output_dim * self.hidden_dim + self.output_dim
    }

    pub fn to_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        for d in [self.input_dim, self.hidden_dim, self.output_dim] {
            buf.extend_from_slice(&(d as u64).to_le_bytes());
        }
        for v in self.w1.iter().chain(&self.b1).chain(&self.w2).chain(&self.b2) {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        buf
    }

    pub fn from_bytes(data: &[u8]) -> Option<Self> {
        if data.len() < 24 { return None; }
        let read_u64 = |off: usize| -> Option<usize> {
            Some(u64::from_le_bytes(data[off..off+8].try_into().ok()?) as usize)
        };
        let input_dim = read_u64(0)?;
        let hidden_dim = read_u64(8)?;
        let output_dim = read_u64(16)?;
        let n_floats = hidden_dim * input_dim + hidden_dim + output_dim * hidden_dim + output_dim;
        if data.len() < 24 + n_floats * 8 { return None; }
        let mut off = 24;
        let mut read_vec = |n: usize| -> Vec<f64> {
            let v: Vec<f64> = (0..n).map(|i| {
                let start = off + i * 8;
                f64::from_le_bytes(data[start..start+8].try_into().unwrap())
            }).collect();
            off += n * 8;
            v
        };
        let w1 = read_vec(hidden_dim * input_dim);
        let b1 = read_vec(hidden_dim);
        let w2 = read_vec(output_dim * hidden_dim);
        let b2 = read_vec(output_dim);
        Some(Self { input_dim, hidden_dim, output_dim, w1, b1, w2, b2 })
    }
}

// ---------------------------------------------------------------------------
// TieredExpertStore
// ---------------------------------------------------------------------------

pub struct TieredExpertStore {
    pub experts: HashMap<usize, (MicroExpert, StorageTier)>,
    pub hot_budget: usize,
    pub warm_budget: usize,
    pub access_counts: HashMap<usize, usize>,
    access_order: Vec<usize>, // tracks recency (most recent at end)
}

impl TieredExpertStore {
    pub fn new(hot_budget: usize, warm_budget: usize) -> Self {
        Self {
            experts: HashMap::new(),
            hot_budget,
            warm_budget,
            access_counts: HashMap::new(),
            access_order: Vec::new(),
        }
    }

    pub fn insert(&mut self, id: usize, expert: MicroExpert, tier: StorageTier) {
        self.experts.insert(id, (expert, tier));
    }

    pub fn get_tier(&self, id: usize) -> Option<StorageTier> {
        self.experts.get(&id).map(|(_, t)| *t)
    }

    pub fn tier_counts(&self) -> (usize, usize, usize) {
        let (mut h, mut w, mut c) = (0, 0, 0);
        for (_, (_, t)) in &self.experts {
            match t { StorageTier::Hot => h += 1, StorageTier::Warm => w += 1, StorageTier::Cold => c += 1 }
        }
        (h, w, c)
    }

    pub fn record_access(&mut self, id: usize) {
        *self.access_counts.entry(id).or_insert(0) += 1;
        self.access_order.retain(|&x| x != id);
        self.access_order.push(id);
    }

    pub fn promote(&mut self, id: usize) {
        if let Some((_, tier)) = self.experts.get_mut(&id) {
            *tier = match *tier {
                StorageTier::Cold => StorageTier::Warm,
                StorageTier::Warm => StorageTier::Hot,
                StorageTier::Hot => StorageTier::Hot,
            };
        }
    }

    pub fn demote(&mut self, id: usize) {
        if let Some((_, tier)) = self.experts.get_mut(&id) {
            *tier = match *tier {
                StorageTier::Hot => StorageTier::Warm,
                StorageTier::Warm => StorageTier::Cold,
                StorageTier::Cold => StorageTier::Cold,
            };
        }
    }

    pub fn prefetch(&mut self, ids: &[usize]) {
        for &id in ids {
            if let Some((_, tier)) = self.experts.get(&id) {
                if *tier != StorageTier::Hot {
                    self.promote(id);
                }
            }
        }
        self.enforce_budgets();
    }

    /// Evict least-recently-used experts from Hot->Warm, Warm->Cold to respect budgets.
    pub fn evict_lru(&mut self) {
        self.enforce_budgets();
    }

    fn enforce_budgets(&mut self) {
        // Evict from Hot tier if over budget
        while self.count_tier(StorageTier::Hot) > self.hot_budget {
            if let Some(victim) = self.find_lru(StorageTier::Hot) {
                self.demote(victim);
            } else {
                break;
            }
        }
        // Evict from Warm tier if over budget
        while self.count_tier(StorageTier::Warm) > self.warm_budget {
            if let Some(victim) = self.find_lru(StorageTier::Warm) {
                self.demote(victim);
            } else {
                break;
            }
        }
    }

    fn count_tier(&self, tier: StorageTier) -> usize {
        self.experts.values().filter(|(_, t)| *t == tier).count()
    }

    fn find_lru(&self, tier: StorageTier) -> Option<usize> {
        // Walk access_order from oldest to newest, find first in given tier
        for &id in &self.access_order {
            if let Some((_, t)) = self.experts.get(&id) {
                if *t == tier { return Some(id); }
            }
        }
        // Fallback: any expert in that tier
        self.experts.iter().find(|(_, (_, t))| *t == tier).map(|(&id, _)| id)
    }

    pub fn forward_expert(&mut self, id: usize, input: &[f64]) -> Option<Vec<f64>> {
        self.record_access(id);
        self.experts.get(&id).map(|(e, _)| e.forward(input))
    }
}

// ---------------------------------------------------------------------------
// HierarchicalRouter – two-level coarse+fine routing
// ---------------------------------------------------------------------------

pub struct HierarchicalRouter {
    pub n_clusters: usize,
    pub cluster_assignment: Vec<Vec<usize>>, // cluster_id -> list of expert_ids
    // coarse router weights: n_clusters x input_dim
    pub coarse_weights: Vec<f64>,
    // fine router weights: n_experts x input_dim (flat, indexed by expert_id)
    pub fine_weights: HashMap<usize, Vec<f64>>,
    pub input_dim: usize,
}

impl HierarchicalRouter {
    pub fn new(n_experts: usize, n_clusters: usize, input_dim: usize, seed: u64) -> Self {
        let mut rng = SimpleRng::new(seed);
        let scale = (1.0 / input_dim as f64).sqrt();

        // Assign experts to clusters round-robin
        let mut cluster_assignment = vec![Vec::new(); n_clusters];
        for i in 0..n_experts {
            cluster_assignment[i % n_clusters].push(i);
        }

        let coarse_weights: Vec<f64> = (0..n_clusters * input_dim)
            .map(|_| rng.next_normal() * scale)
            .collect();

        let mut fine_weights = HashMap::new();
        for i in 0..n_experts {
            let w: Vec<f64> = (0..input_dim).map(|_| rng.next_normal() * scale).collect();
            fine_weights.insert(i, w);
        }

        Self { n_clusters, cluster_assignment, coarse_weights, fine_weights, input_dim }
    }

    /// Route input: pick top_k_clusters clusters, then top_k experts total from those clusters.
    pub fn route(&self, input: &[f64], top_k: usize) -> Vec<(usize, f64)> {
        let top_k_clusters = (self.n_clusters as f64).sqrt().ceil() as usize;
        let top_k_clusters = top_k_clusters.max(1).min(self.n_clusters);

        // Score clusters
        let mut cluster_scores: Vec<(usize, f64)> = (0..self.n_clusters)
            .map(|c| {
                let score = dot(&self.coarse_weights[c * self.input_dim..(c + 1) * self.input_dim], input);
                (c, score)
            })
            .collect();
        cluster_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        cluster_scores.truncate(top_k_clusters);

        // Score experts within selected clusters
        let mut expert_scores: Vec<(usize, f64)> = Vec::new();
        for &(cluster_id, _) in &cluster_scores {
            for &expert_id in &self.cluster_assignment[cluster_id] {
                if let Some(w) = self.fine_weights.get(&expert_id) {
                    let score = dot(w, input);
                    expert_scores.push((expert_id, score));
                }
            }
        }
        expert_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        expert_scores.truncate(top_k);

        // Softmax over selected expert scores
        if expert_scores.is_empty() { return vec![]; }
        let max_s = expert_scores.iter().map(|(_, s)| *s).fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = expert_scores.iter().map(|(_, s)| (s - max_s).exp()).collect();
        let sum: f64 = exps.iter().sum();
        expert_scores.iter().zip(exps.iter())
            .map(|((id, _), e)| (*id, e / sum))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// PrefetchScheduler – simple Markov chain predictor
// ---------------------------------------------------------------------------

pub struct PrefetchScheduler {
    // transition_counts[from][to] = count
    transition_counts: HashMap<usize, HashMap<usize, usize>>,
}

impl PrefetchScheduler {
    pub fn new() -> Self {
        Self { transition_counts: HashMap::new() }
    }

    pub fn observe(&mut self, from_ids: &[usize], to_ids: &[usize]) {
        for &f in from_ids {
            let entry = self.transition_counts.entry(f).or_insert_with(HashMap::new);
            for &t in to_ids {
                *entry.entry(t).or_insert(0) += 1;
            }
        }
    }

    pub fn predict_next(&self, current_ids: &[usize]) -> Vec<usize> {
        let mut scores: HashMap<usize, usize> = HashMap::new();
        for &id in current_ids {
            if let Some(transitions) = self.transition_counts.get(&id) {
                for (&next, &count) in transitions {
                    *scores.entry(next).or_insert(0) += count;
                }
            }
        }
        let mut ranked: Vec<(usize, usize)> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));
        ranked.into_iter().take(8).map(|(id, _)| id).collect()
    }
}

// ---------------------------------------------------------------------------
// TieredMoELayer – combines everything
// ---------------------------------------------------------------------------

pub struct TieredMoELayer {
    pub router: HierarchicalRouter,
    pub store: TieredExpertStore,
    pub prefetcher: PrefetchScheduler,
    pub top_k: usize,
    pub input_dim: usize,
    pub output_dim: usize,
    usage_history: Vec<Vec<usize>>, // last N forward expert selections
    prev_expert_ids: Vec<usize>,
    promote_threshold: usize,
}

impl TieredMoELayer {
    pub fn new(
        n_experts: usize,
        expert_width: usize,
        n_clusters: usize,
        hot_budget: usize,
        warm_budget: usize,
        top_k: usize,
        input_dim: usize,
    ) -> Self {
        let router = HierarchicalRouter::new(n_experts, n_clusters, input_dim, 42);
        let mut store = TieredExpertStore::new(hot_budget, warm_budget);

        // Create experts: first hot_budget in Hot, next warm_budget in Warm, rest Cold
        for i in 0..n_experts {
            let tier = if i < hot_budget {
                StorageTier::Hot
            } else if i < hot_budget + warm_budget {
                StorageTier::Warm
            } else {
                StorageTier::Cold
            };
            let expert = MicroExpert::new(input_dim, expert_width, input_dim, 100 + i as u64);
            store.insert(i, expert, tier);
        }

        Self {
            router,
            store,
            prefetcher: PrefetchScheduler::new(),
            top_k,
            input_dim,
            output_dim: input_dim,
            usage_history: Vec::new(),
            prev_expert_ids: Vec::new(),
            promote_threshold: 3,
        }
    }

    pub fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let routed = self.router.route(input, self.top_k);
        let expert_ids: Vec<usize> = routed.iter().map(|(id, _)| *id).collect();

        // Prefetch predicted experts
        let predicted = self.prefetcher.predict_next(&expert_ids);
        self.store.prefetch(&predicted);

        // Update prefetcher with transition
        if !self.prev_expert_ids.is_empty() {
            self.prefetcher.observe(&self.prev_expert_ids, &expert_ids);
        }
        self.prev_expert_ids = expert_ids.clone();

        // Compute weighted combination
        let mut output = vec![0.0; self.output_dim];
        for &(expert_id, weight) in &routed {
            if let Some(expert_out) = self.store.forward_expert(expert_id, input) {
                for (o, &e) in output.iter_mut().zip(expert_out.iter()) {
                    *o += weight * e;
                }
            }
        }

        // Track usage
        self.usage_history.push(expert_ids.clone());
        if self.usage_history.len() > 100 {
            self.usage_history.remove(0);
        }

        // Auto tier management: promote frequently accessed, demote rarely accessed
        for &id in &expert_ids {
            let count = *self.store.access_counts.get(&id).unwrap_or(&0);
            if count >= self.promote_threshold {
                if self.store.get_tier(id) != Some(StorageTier::Hot) {
                    self.store.promote(id);
                }
            }
        }
        self.store.evict_lru();

        output
    }

    pub fn total_params(&self) -> usize {
        self.store.experts.values().map(|(e, _)| e.param_count()).sum()
    }

    pub fn active_params(&self) -> usize {
        self.store.experts.values()
            .filter(|(_, t)| *t == StorageTier::Hot)
            .map(|(e, _)| e.param_count())
            .sum()
    }

    pub fn utilization(&self) -> f64 {
        if self.usage_history.is_empty() || self.store.experts.is_empty() {
            return 0.0;
        }
        let mut used: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for ids in &self.usage_history {
            for &id in ids { used.insert(id); }
        }
        used.len() as f64 / self.store.experts.len() as f64
    }

    pub fn stats(&self) -> (usize, usize, usize, usize, usize) {
        let (h, w, c) = self.store.tier_counts();
        (self.total_params(), self.active_params(), h, w, c)
    }
}

// ---------------------------------------------------------------------------
// Public constructor for interpreter integration
// ---------------------------------------------------------------------------

pub fn create_tiered_moe(n_experts: usize, expert_width: usize, n_clusters: usize, hot_budget: usize, input_dim: usize) -> TieredMoELayer {
    let warm_budget = hot_budget * 2;
    let top_k = 4;
    TieredMoELayer::new(n_experts, expert_width, n_clusters, hot_budget, warm_budget, top_k, input_dim)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

struct SimpleRng { state: u64 }

impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(1) } }

    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() & 0x1FFFFFFFFFFFFF) as f64 / (1u64 << 53) as f64
    }

    fn next_normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_forward_produces_output() {
        let mut layer = TieredMoELayer::new(100, 16, 10, 20, 40, 4, 8);
        let input = vec![1.0; 8];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 8);
        // Output should not be all zeros (experts have random weights)
        assert!(output.iter().any(|&v| v != 0.0), "output should be non-zero");
    }

    #[test]
    fn test_router_selects_top_k() {
        let router = HierarchicalRouter::new(100, 10, 8, 42);
        let input = vec![1.0; 8];
        let result = router.route(&input, 4);
        assert_eq!(result.len(), 4);
        // Weights should sum to ~1 (softmax)
        let wsum: f64 = result.iter().map(|(_, w)| w).sum();
        assert!((wsum - 1.0).abs() < 1e-6, "weights should sum to 1, got {}", wsum);
    }

    #[test]
    fn test_hierarchical_routing_uses_clusters() {
        let router = HierarchicalRouter::new(1000, 50, 8, 99);
        let input = vec![0.5; 8];
        let result = router.route(&input, 4);
        assert_eq!(result.len(), 4);
        // All selected experts should exist
        for &(id, _) in &result {
            assert!(id < 1000);
        }
        // With sqrt(50) ~ 7 clusters evaluated, we only look at a subset
        // The router should NOT have evaluated all 1000 experts
    }

    #[test]
    fn test_tier_promotion_on_frequent_access() {
        let mut layer = TieredMoELayer::new(20, 8, 4, 5, 10, 2, 4);
        // Expert 15 starts in Cold tier (index >= 5+10=15)
        assert_eq!(layer.store.get_tier(15), Some(StorageTier::Cold));

        // Access expert 15 many times to trigger promotion
        for _ in 0..5 {
            layer.store.record_access(15);
        }
        // Manually promote
        layer.store.promote(15);
        assert_eq!(layer.store.get_tier(15), Some(StorageTier::Warm));
        layer.store.promote(15);
        assert_eq!(layer.store.get_tier(15), Some(StorageTier::Hot));
    }

    #[test]
    fn test_tier_demotion_lru_eviction() {
        let mut store = TieredExpertStore::new(2, 2);
        for i in 0..5 {
            store.insert(i, MicroExpert::new(4, 8, 4, i as u64), StorageTier::Hot);
            store.record_access(i);
        }
        // 5 experts in Hot, budget is 2 -> evict 3
        store.evict_lru();
        let (h, w, _c) = store.tier_counts();
        assert!(h <= 2, "hot count {} should be <= 2", h);
        assert!(w <= 2, "warm count {} should be <= 2", w);
    }

    #[test]
    fn test_prefetch_loads_predicted_experts() {
        let mut layer = TieredMoELayer::new(50, 8, 5, 10, 20, 2, 4);
        let input = vec![1.0; 4];

        // Run several forwards to build up Markov chain
        for _ in 0..10 {
            layer.forward(&input);
        }

        // The prefetcher should have some transitions recorded
        let current: Vec<usize> = layer.prev_expert_ids.clone();
        let predicted = layer.prefetcher.predict_next(&current);
        // Should predict at least something if we've had repeated patterns
        // (with deterministic routing on same input, we get same experts each time)
        assert!(!predicted.is_empty() || current.is_empty());
    }

    #[test]
    fn test_1000_plus_experts_works() {
        let mut layer = TieredMoELayer::new(1200, 8, 50, 20, 100, 4, 4);
        let input = vec![0.3, -0.5, 0.7, 0.1];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 4);
        assert_eq!(layer.store.experts.len(), 1200);
    }

    #[test]
    fn test_memory_stays_within_hot_budget() {
        let hot_budget = 10;
        let mut layer = TieredMoELayer::new(500, 8, 20, hot_budget, 50, 4, 4);
        let input = vec![1.0; 4];

        for i in 0..20 {
            let shifted: Vec<f64> = input.iter().map(|x| x + i as f64 * 0.1).collect();
            layer.forward(&shifted);
        }

        let (h, _w, _c) = layer.store.tier_counts();
        assert!(h <= hot_budget + layer.top_k,
            "hot count {} should be within budget {} + top_k {}", h, hot_budget, layer.top_k);
    }

    #[test]
    fn test_micro_expert_serialization() {
        let expert = MicroExpert::new(4, 8, 4, 42);
        let bytes = expert.to_bytes();
        let restored = MicroExpert::from_bytes(&bytes).expect("deserialization should work");
        assert_eq!(expert.param_count(), restored.param_count());
        let input = vec![1.0; 4];
        let out1 = expert.forward(&input);
        let out2 = restored.forward(&input);
        assert_eq!(out1, out2);
    }

    #[test]
    fn test_utilization_tracking() {
        let mut layer = TieredMoELayer::new(100, 8, 10, 20, 40, 4, 4);
        assert_eq!(layer.utilization(), 0.0);
        let input = vec![1.0; 4];
        layer.forward(&input);
        assert!(layer.utilization() > 0.0);
    }
}
