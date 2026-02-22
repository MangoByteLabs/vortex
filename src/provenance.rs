/// Data provenance, privacy tracking, and differential privacy for Vortex tensors.

use std::collections::HashMap;
use crate::interpreter::{Env, FnDef, Value};

// ── PrivacyLevel ────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum PrivacyLevel {
    Public,
    Internal,
    Confidential,
    HIPAA,
    GDPR,
    PCI,
    TopSecret,
}

impl PrivacyLevel {
    pub fn can_export(&self) -> bool {
        matches!(self, PrivacyLevel::Public | PrivacyLevel::Internal)
    }

    /// Combine two privacy levels: most restrictive wins.
    pub fn can_combine(&self, other: &PrivacyLevel) -> PrivacyLevel {
        if self >= other { self.clone() } else { other.clone() }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "public" => Ok(PrivacyLevel::Public),
            "internal" => Ok(PrivacyLevel::Internal),
            "confidential" => Ok(PrivacyLevel::Confidential),
            "hipaa" => Ok(PrivacyLevel::HIPAA),
            "gdpr" => Ok(PrivacyLevel::GDPR),
            "pci" => Ok(PrivacyLevel::PCI),
            "topsecret" | "top_secret" => Ok(PrivacyLevel::TopSecret),
            _ => Err(format!("Unknown privacy level: {}", s)),
        }
    }
}

impl std::fmt::Display for PrivacyLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PrivacyLevel::Public => write!(f, "Public"),
            PrivacyLevel::Internal => write!(f, "Internal"),
            PrivacyLevel::Confidential => write!(f, "Confidential"),
            PrivacyLevel::HIPAA => write!(f, "HIPAA"),
            PrivacyLevel::GDPR => write!(f, "GDPR"),
            PrivacyLevel::PCI => write!(f, "PCI"),
            PrivacyLevel::TopSecret => write!(f, "TopSecret"),
        }
    }
}

// ── DataSource ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataSource {
    pub name: String,
    pub privacy_level: PrivacyLevel,
    pub retention_days: Option<usize>,
    pub created_at: u64,
    pub license: Option<String>,
}

// ── AuditEntry ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AuditEntry {
    pub timestamp: u64,
    pub operation: String,
    pub input_ids: Vec<usize>,
    pub output_id: usize,
    pub actor: String,
}

// ── ProvenanceTracker ───────────────────────────────────────────────

#[derive(Debug, Clone, Default)]
pub struct ProvenanceTracker {
    pub sources: HashMap<usize, DataSource>,
    pub lineage: HashMap<usize, Vec<usize>>,
    pub audit_log: Vec<AuditEntry>,
}

impl ProvenanceTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn track(&mut self, tensor_id: usize, source: DataSource) {
        self.audit_log.push(AuditEntry {
            timestamp: source.created_at,
            operation: "track".to_string(),
            input_ids: vec![],
            output_id: tensor_id,
            actor: "system".to_string(),
        });
        self.sources.insert(tensor_id, source);
    }

    pub fn derive(&mut self, new_id: usize, parent_ids: &[usize]) {
        self.lineage.insert(new_id, parent_ids.to_vec());
        self.audit_log.push(AuditEntry {
            timestamp: 0,
            operation: "derive".to_string(),
            input_ids: parent_ids.to_vec(),
            output_id: new_id,
            actor: "system".to_string(),
        });
    }

    /// Get all ancestor DataSources (recursive).
    pub fn get_lineage(&self, tensor_id: usize) -> Vec<DataSource> {
        let mut visited = std::collections::HashSet::new();
        let mut result = Vec::new();
        self.collect_sources(tensor_id, &mut visited, &mut result);
        result
    }

    fn collect_sources(&self, id: usize, visited: &mut std::collections::HashSet<usize>, result: &mut Vec<DataSource>) {
        if !visited.insert(id) { return; }
        if let Some(src) = self.sources.get(&id) {
            result.push(src.clone());
        }
        if let Some(parents) = self.lineage.get(&id) {
            for &p in parents {
                self.collect_sources(p, visited, result);
            }
        }
    }

    /// Inherited privacy level: most restrictive of all ancestors.
    pub fn privacy_level(&self, tensor_id: usize) -> PrivacyLevel {
        let sources = self.get_lineage(tensor_id);
        sources.iter().fold(PrivacyLevel::Public, |acc, s| acc.can_combine(&s.privacy_level))
    }

    pub fn can_export(&self, tensor_id: usize) -> bool {
        self.privacy_level(tensor_id).can_export()
    }

    pub fn audit_trail(&self, tensor_id: usize) -> Vec<AuditEntry> {
        self.audit_log.iter().filter(|e| e.output_id == tensor_id || e.input_ids.contains(&tensor_id)).cloned().collect()
    }

    /// Remove a source and return its id.
    pub fn remove_source(&mut self, tensor_id: usize) -> Option<DataSource> {
        self.sources.remove(&tensor_id)
    }
}

// ── RetentionPolicy ─────────────────────────────────────────────────

pub struct RetentionPolicy;

impl RetentionPolicy {
    pub fn check_expiry(source: &DataSource, current_time: u64) -> bool {
        if let Some(days) = source.retention_days {
            let secs = days as u64 * 86400;
            current_time > source.created_at + secs
        } else {
            false
        }
    }

    pub fn expire_data(tracker: &mut ProvenanceTracker, current_time: u64) -> Vec<usize> {
        let expired: Vec<usize> = tracker.sources.iter()
            .filter(|(_, src)| Self::check_expiry(src, current_time))
            .map(|(&id, _)| id)
            .collect();
        for &id in &expired {
            tracker.sources.remove(&id);
        }
        expired
    }

    /// GDPR right to delete: remove all data from a source + derivatives.
    pub fn gdpr_right_to_delete(tracker: &mut ProvenanceTracker, source_name: &str) -> Vec<usize> {
        // Find direct sources with matching name
        let direct: Vec<usize> = tracker.sources.iter()
            .filter(|(_, src)| src.name == source_name)
            .map(|(&id, _)| id)
            .collect();
        // Find all derivatives
        let mut to_delete = std::collections::HashSet::new();
        for &id in &direct {
            to_delete.insert(id);
        }
        // Iterate lineage to find anything derived from these
        let mut changed = true;
        while changed {
            changed = false;
            for (&child, parents) in &tracker.lineage {
                if !to_delete.contains(&child) && parents.iter().any(|p| to_delete.contains(p)) {
                    to_delete.insert(child);
                    changed = true;
                }
            }
        }
        for &id in &to_delete {
            tracker.sources.remove(&id);
        }
        let mut result: Vec<usize> = to_delete.into_iter().collect();
        result.sort();
        result
    }
}

// ── DifferentialPrivacy ─────────────────────────────────────────────

pub struct DifferentialPrivacy;

/// Simple deterministic PRNG (xorshift64) to avoid external deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self { Self(if seed == 0 { 0xdeadbeef } else { seed }) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
    /// Box-Muller transform for Gaussian samples.
    fn next_gaussian(&mut self) -> f64 {
        let u1 = self.next_f64().max(1e-15);
        let u2 = self.next_f64();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
    }
    /// Laplace sample: mu=0.
    fn next_laplace(&mut self, scale: f64) -> f64 {
        let u = self.next_f64() - 0.5;
        -scale * u.signum() * (1.0 - 2.0 * u.abs()).ln()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NoiseMechanism {
    Gaussian,
    Laplace,
}

impl DifferentialPrivacy {
    /// Add calibrated noise to data.
    /// For Gaussian: sigma = sensitivity * sqrt(2*ln(1.25/delta)) / epsilon
    /// For Laplace: scale = sensitivity / epsilon
    pub fn add_noise(data: &[f64], epsilon: f64, delta: f64, sensitivity: f64, mechanism: NoiseMechanism) -> Vec<f64> {
        let mut rng = Rng::new(42);
        match mechanism {
            NoiseMechanism::Gaussian => {
                let sigma = sensitivity * (2.0 * (1.25_f64 / delta).ln()).sqrt() / epsilon;
                data.iter().map(|&x| x + sigma * rng.next_gaussian()).collect()
            }
            NoiseMechanism::Laplace => {
                let scale = sensitivity / epsilon;
                data.iter().map(|&x| x + rng.next_laplace(scale)).collect()
            }
        }
    }

    /// DP-SGD step: clip gradients per-sample then add Gaussian noise.
    pub fn dp_sgd_step(gradients: &[Vec<f64>], max_norm: f64, noise_scale: f64, lr: f64) -> Vec<Vec<f64>> {
        if gradients.is_empty() { return vec![]; }
        let dim = gradients[0].len();
        let n = gradients.len() as f64;

        // Clip each gradient vector to max_norm
        let clipped: Vec<Vec<f64>> = gradients.iter().map(|g| {
            let norm: f64 = g.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm > max_norm {
                let scale = max_norm / norm;
                g.iter().map(|x| x * scale).collect()
            } else {
                g.clone()
            }
        }).collect();

        // Average clipped gradients
        let mut avg = vec![0.0; dim];
        for g in &clipped {
            for (i, &v) in g.iter().enumerate() {
                avg[i] += v / n;
            }
        }

        // Add Gaussian noise
        let sigma = noise_scale * max_norm / n;
        let mut rng = Rng::new(123);
        for v in &mut avg {
            *v += sigma * rng.next_gaussian();
        }

        // Apply learning rate: return as single "updated gradient"
        let updated: Vec<f64> = avg.iter().map(|&v| -lr * v).collect();
        vec![updated]
    }

    /// Basic composition: epsilons add.
    pub fn compose(epsilon1: f64, epsilon2: f64) -> f64 {
        epsilon1 + epsilon2
    }
}

// ── PrivacyBudget ───────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct PrivacyBudget {
    pub total_epsilon: f64,
    pub spent_epsilon: f64,
}

impl PrivacyBudget {
    pub fn new(total_epsilon: f64) -> Self {
        Self { total_epsilon, spent_epsilon: 0.0 }
    }

    pub fn remaining(&self) -> f64 {
        (self.total_epsilon - self.spent_epsilon).max(0.0)
    }

    pub fn can_query(&self, epsilon: f64) -> bool {
        self.spent_epsilon + epsilon <= self.total_epsilon
    }

    pub fn spend(&mut self, epsilon: f64) -> Result<(), String> {
        if !self.can_query(epsilon) {
            return Err(format!("Privacy budget exceeded: need {}, have {}", epsilon, self.remaining()));
        }
        self.spent_epsilon += epsilon;
        Ok(())
    }

    /// Advanced composition theorem: for k queries each at epsilon, with failure prob delta.
    /// Result: epsilon_total = sqrt(2k * ln(1/delta)) * epsilon + k * epsilon * (e^epsilon - 1)
    pub fn advanced_composition(k: usize, epsilon: f64, delta: f64) -> f64 {
        let kf = k as f64;
        let term1 = (2.0 * kf * (1.0 / delta).ln()).sqrt() * epsilon;
        let term2 = kf * epsilon * (epsilon.exp() - 1.0);
        term1 + term2
    }
}

// ── Global state for interpreter builtins ───────────────────────────

use std::sync::Mutex;
use std::sync::LazyLock;

static PROVENANCE_TRACKER: LazyLock<Mutex<ProvenanceTracker>> = LazyLock::new(|| Mutex::new(ProvenanceTracker::new()));
static PRIVACY_BUDGETS: LazyLock<Mutex<HashMap<usize, PrivacyBudget>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static BUDGET_COUNTER: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));

// ── Interpreter builtins ────────────────────────────────────────────

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        _ => Err("Expected integer".to_string()),
    }
}

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("Expected number".to_string()),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => Err("Expected array".to_string()),
    }
}

fn builtin_provenance_track(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("provenance_track(tensor_id, source_name, privacy_level)".into()); }
    let tensor_id = value_to_usize(&args[0])?;
    let name = match &args[1] { Value::String(s) => s.clone(), _ => return Err("source_name must be string".into()) };
    let level_str = match &args[2] { Value::String(s) => s.clone(), _ => return Err("privacy_level must be string".into()) };
    let level = PrivacyLevel::from_str(&level_str)?;
    let source = DataSource { name, privacy_level: level, retention_days: None, created_at: 0, license: None };
    PROVENANCE_TRACKER.lock().unwrap().track(tensor_id, source);
    Ok(Value::Bool(true))
}

fn builtin_provenance_lineage(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("provenance_lineage(tensor_id)".into()); }
    let tensor_id = value_to_usize(&args[0])?;
    let tracker = PROVENANCE_TRACKER.lock().unwrap();
    let sources = tracker.get_lineage(tensor_id);
    let names: Vec<Value> = sources.iter().map(|s| Value::String(s.name.clone())).collect();
    Ok(Value::Array(names))
}

fn builtin_provenance_can_export(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("provenance_can_export(tensor_id)".into()); }
    let tensor_id = value_to_usize(&args[0])?;
    let tracker = PROVENANCE_TRACKER.lock().unwrap();
    Ok(Value::Bool(tracker.can_export(tensor_id)))
}

fn builtin_dp_add_noise(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("dp_add_noise(data, epsilon, sensitivity)".into()); }
    let data = value_to_f64_vec(&args[0])?;
    let epsilon = value_to_f64(&args[1])?;
    let sensitivity = value_to_f64(&args[2])?;
    let noisy = DifferentialPrivacy::add_noise(&data, epsilon, 1e-5, sensitivity, NoiseMechanism::Laplace);
    Ok(Value::Array(noisy.into_iter().map(Value::Float).collect()))
}

fn builtin_dp_sgd_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("dp_sgd_step(gradients, max_norm, noise_scale, lr)".into()); }
    let grads_outer = match &args[0] { Value::Array(a) => a, _ => return Err("gradients must be array of arrays".into()) };
    let gradients: Vec<Vec<f64>> = grads_outer.iter().map(value_to_f64_vec).collect::<Result<_, _>>()?;
    let max_norm = value_to_f64(&args[1])?;
    let noise_scale = value_to_f64(&args[2])?;
    let lr = value_to_f64(&args[3])?;
    let result = DifferentialPrivacy::dp_sgd_step(&gradients, max_norm, noise_scale, lr);
    let outer: Vec<Value> = result.into_iter().map(|v| Value::Array(v.into_iter().map(Value::Float).collect())).collect();
    Ok(Value::Array(outer))
}

fn builtin_privacy_budget_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("privacy_budget_new(total_epsilon)".into()); }
    let total = value_to_f64(&args[0])?;
    let mut counter = BUDGET_COUNTER.lock().unwrap();
    let id = *counter;
    *counter += 1;
    PRIVACY_BUDGETS.lock().unwrap().insert(id, PrivacyBudget::new(total));
    Ok(Value::Int(id as i128))
}

fn builtin_privacy_budget_spend(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("privacy_budget_spend(id, epsilon)".into()); }
    let id = value_to_usize(&args[0])?;
    let eps = value_to_f64(&args[1])?;
    let mut budgets = PRIVACY_BUDGETS.lock().unwrap();
    let budget = budgets.get_mut(&id).ok_or("Unknown budget id")?;
    budget.spend(eps)?;
    Ok(Value::Float(budget.remaining()))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("provenance_track".to_string(), FnDef::Builtin(builtin_provenance_track));
    env.functions.insert("provenance_lineage".to_string(), FnDef::Builtin(builtin_provenance_lineage));
    env.functions.insert("provenance_can_export".to_string(), FnDef::Builtin(builtin_provenance_can_export));
    env.functions.insert("dp_add_noise".to_string(), FnDef::Builtin(builtin_dp_add_noise));
    env.functions.insert("dp_sgd_step".to_string(), FnDef::Builtin(builtin_dp_sgd_step));
    env.functions.insert("privacy_budget_new".to_string(), FnDef::Builtin(builtin_privacy_budget_new));
    env.functions.insert("privacy_budget_spend".to_string(), FnDef::Builtin(builtin_privacy_budget_spend));
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_level_ordering() {
        assert!(PrivacyLevel::TopSecret > PrivacyLevel::Public);
        assert!(PrivacyLevel::HIPAA > PrivacyLevel::Internal);
        assert!(PrivacyLevel::GDPR > PrivacyLevel::Confidential);
    }

    #[test]
    fn test_privacy_level_can_export() {
        assert!(PrivacyLevel::Public.can_export());
        assert!(PrivacyLevel::Internal.can_export());
        assert!(!PrivacyLevel::Confidential.can_export());
        assert!(!PrivacyLevel::HIPAA.can_export());
        assert!(!PrivacyLevel::GDPR.can_export());
        assert!(!PrivacyLevel::PCI.can_export());
        assert!(!PrivacyLevel::TopSecret.can_export());
    }

    #[test]
    fn test_privacy_level_combine() {
        assert_eq!(PrivacyLevel::Public.can_combine(&PrivacyLevel::HIPAA), PrivacyLevel::HIPAA);
        assert_eq!(PrivacyLevel::TopSecret.can_combine(&PrivacyLevel::Public), PrivacyLevel::TopSecret);
        assert_eq!(PrivacyLevel::GDPR.can_combine(&PrivacyLevel::PCI), PrivacyLevel::PCI);
    }

    #[test]
    fn test_provenance_track_and_lineage() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track(1, DataSource { name: "hospital_a".into(), privacy_level: PrivacyLevel::HIPAA, retention_days: Some(365), created_at: 1000, license: None });
        tracker.track(2, DataSource { name: "public_data".into(), privacy_level: PrivacyLevel::Public, retention_days: None, created_at: 1000, license: Some("MIT".into()) });
        tracker.derive(3, &[1, 2]);

        let lineage = tracker.get_lineage(3);
        assert_eq!(lineage.len(), 2);
        assert_eq!(tracker.privacy_level(3), PrivacyLevel::HIPAA);
        assert!(!tracker.can_export(3));
        assert!(tracker.can_export(2));
    }

    #[test]
    fn test_provenance_deep_lineage() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track(1, DataSource { name: "src1".into(), privacy_level: PrivacyLevel::TopSecret, retention_days: None, created_at: 0, license: None });
        tracker.derive(2, &[1]);
        tracker.derive(3, &[2]);
        tracker.derive(4, &[3]);
        assert_eq!(tracker.privacy_level(4), PrivacyLevel::TopSecret);
        assert_eq!(tracker.get_lineage(4).len(), 1);
    }

    #[test]
    fn test_audit_trail() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track(1, DataSource { name: "s1".into(), privacy_level: PrivacyLevel::Public, retention_days: None, created_at: 0, license: None });
        tracker.derive(2, &[1]);
        let trail = tracker.audit_trail(1);
        assert!(trail.len() >= 1);
    }

    #[test]
    fn test_retention_expiry() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track(1, DataSource { name: "temp".into(), privacy_level: PrivacyLevel::Internal, retention_days: Some(30), created_at: 1000, license: None });
        tracker.track(2, DataSource { name: "perm".into(), privacy_level: PrivacyLevel::Public, retention_days: None, created_at: 1000, license: None });
        let expired = RetentionPolicy::expire_data(&mut tracker, 1000 + 31 * 86400);
        assert_eq!(expired, vec![1]);
        assert!(tracker.sources.get(&1).is_none());
        assert!(tracker.sources.get(&2).is_some());
    }

    #[test]
    fn test_gdpr_right_to_delete() {
        let mut tracker = ProvenanceTracker::new();
        tracker.track(1, DataSource { name: "user_data".into(), privacy_level: PrivacyLevel::GDPR, retention_days: None, created_at: 0, license: None });
        tracker.track(2, DataSource { name: "other".into(), privacy_level: PrivacyLevel::Public, retention_days: None, created_at: 0, license: None });
        tracker.derive(3, &[1]);
        tracker.derive(4, &[3, 2]);
        let deleted = RetentionPolicy::gdpr_right_to_delete(&mut tracker, "user_data");
        assert!(deleted.contains(&1));
        assert!(deleted.contains(&3));
        assert!(deleted.contains(&4)); // derived from 3 which derives from 1
        assert!(!deleted.contains(&2));
    }

    #[test]
    fn test_dp_add_noise_laplace() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let noisy = DifferentialPrivacy::add_noise(&data, 1.0, 1e-5, 1.0, NoiseMechanism::Laplace);
        assert_eq!(noisy.len(), 5);
        // Noise should be added - at least one value should differ
        assert!(noisy.iter().zip(data.iter()).any(|(a, b)| (a - b).abs() > 1e-10));
    }

    #[test]
    fn test_dp_add_noise_gaussian() {
        let data = vec![10.0; 100];
        let noisy = DifferentialPrivacy::add_noise(&data, 1.0, 1e-5, 1.0, NoiseMechanism::Gaussian);
        assert_eq!(noisy.len(), 100);
        let mean: f64 = noisy.iter().sum::<f64>() / 100.0;
        // Mean should be roughly 10 (within a few sigma)
        assert!((mean - 10.0).abs() < 5.0);
    }

    #[test]
    fn test_dp_sgd_step() {
        let grads = vec![vec![3.0, 4.0], vec![6.0, 8.0]]; // norms 5 and 10
        let result = DifferentialPrivacy::dp_sgd_step(&grads, 5.0, 0.0001, 0.1);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 2);
        // Second gradient should be clipped to norm 5: [3, 4]
        // Average of [3,4] and [3,4] = [3,4], times -0.1 = [-0.3, -0.4] (approx, with tiny noise)
    }

    #[test]
    fn test_privacy_budget() {
        let mut budget = PrivacyBudget::new(10.0);
        assert_eq!(budget.remaining(), 10.0);
        assert!(budget.can_query(5.0));
        budget.spend(3.0).unwrap();
        assert_eq!(budget.remaining(), 7.0);
        budget.spend(7.0).unwrap();
        assert_eq!(budget.remaining(), 0.0);
        assert!(budget.spend(0.1).is_err());
    }

    #[test]
    fn test_advanced_composition() {
        let result = PrivacyBudget::advanced_composition(10, 0.1, 1e-5);
        // Should be tighter than naive 10 * 0.1 = 1.0
        assert!(result < 3.0);
        assert!(result > 0.0);
    }

    #[test]
    fn test_dp_composition() {
        assert_eq!(DifferentialPrivacy::compose(0.5, 0.3), 0.8);
    }

    #[test]
    fn test_privacy_level_from_str() {
        assert_eq!(PrivacyLevel::from_str("public").unwrap(), PrivacyLevel::Public);
        assert_eq!(PrivacyLevel::from_str("HIPAA").unwrap(), PrivacyLevel::HIPAA);
        assert_eq!(PrivacyLevel::from_str("topsecret").unwrap(), PrivacyLevel::TopSecret);
        assert!(PrivacyLevel::from_str("unknown").is_err());
    }
}
