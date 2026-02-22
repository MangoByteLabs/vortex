// Metabolic computing: programs that optimize their own energy consumption
// with energy budgets, carbon-aware scheduling, and energy-per-operation profiling.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. EnergyProfile – per-operation energy costs in picojoules
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EnergyProfile {
    pub fp32_add: f64,
    pub fp32_mul: f64,
    pub fp32_fma: f64,
    pub fp16_fma: f64,
    pub int8_fma: f64,
    pub int4_fma: f64,
    pub dram_read: f64,
    pub sram_read: f64,
    pub register_read: f64,
    pub network_send_byte: f64,
}

impl EnergyProfile {
    pub fn h100() -> Self {
        Self {
            fp32_add: 0.9,
            fp32_mul: 3.7,
            fp32_fma: 4.6,
            fp16_fma: 1.1,
            int8_fma: 0.3,
            int4_fma: 0.08,
            dram_read: 19200.0,   // ~19.2 nJ per 32-byte cache line
            sram_read: 51.2,
            register_read: 1.0,
            network_send_byte: 640.0,
        }
    }

    pub fn a100() -> Self {
        Self {
            fp32_add: 1.2,
            fp32_mul: 4.5,
            fp32_fma: 5.7,
            fp16_fma: 1.4,
            int8_fma: 0.4,
            int4_fma: 0.12,
            dram_read: 25600.0,
            sram_read: 64.0,
            register_read: 1.2,
            network_send_byte: 800.0,
        }
    }

    pub fn rtx4090() -> Self {
        Self {
            fp32_add: 1.0,
            fp32_mul: 4.0,
            fp32_fma: 5.0,
            fp16_fma: 1.2,
            int8_fma: 0.35,
            int4_fma: 0.09,
            dram_read: 22400.0,
            sram_read: 57.0,
            register_read: 1.1,
            network_send_byte: 720.0,
        }
    }

    pub fn cpu_x86() -> Self {
        Self {
            fp32_add: 3.1,
            fp32_mul: 12.0,
            fp32_fma: 15.0,
            fp16_fma: 7.5,
            int8_fma: 2.0,
            int4_fma: 1.5,
            dram_read: 51200.0,
            sram_read: 128.0,
            register_read: 0.5,
            network_send_byte: 1200.0,
        }
    }

    /// Return per-fma cost for a given dtype string
    pub fn fma_cost(&self, dtype: &str) -> f64 {
        match dtype {
            "fp32" | "FP32" => self.fp32_fma,
            "fp16" | "FP16" => self.fp16_fma,
            "int8" | "INT8" => self.int8_fma,
            "int4" | "INT4" => self.int4_fma,
            _ => self.fp32_fma,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. EnergyBudget
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct EnergyBudget {
    pub total_joules: f64,
    pub spent_joules: f64,
    useful_compute_joules: f64,
}

impl EnergyBudget {
    pub fn new(total_joules: f64) -> Self {
        Self { total_joules, spent_joules: 0.0, useful_compute_joules: 0.0 }
    }

    pub fn remaining(&self) -> f64 {
        (self.total_joules - self.spent_joules).max(0.0)
    }

    pub fn can_afford(&self, cost: f64) -> bool {
        self.spent_joules + cost <= self.total_joules
    }

    pub fn spend(&mut self, cost: f64) -> Result<(), String> {
        if !self.can_afford(cost) {
            return Err(format!(
                "energy budget exceeded: need {} J but only {} J remaining",
                cost,
                self.remaining()
            ));
        }
        self.spent_joules += cost;
        self.useful_compute_joules += cost;
        Ok(())
    }

    pub fn efficiency(&self) -> f64 {
        if self.spent_joules == 0.0 { return 0.0; }
        self.useful_compute_joules / self.spent_joules
    }
}

// ---------------------------------------------------------------------------
// 3. OperationCost – estimate energy for common operations
// ---------------------------------------------------------------------------

pub struct OperationCost;

impl OperationCost {
    /// MatMul C[m,n] = A[m,k] * B[k,n] requires 2*m*n*k FMAs
    pub fn matmul_cost(m: usize, n: usize, k: usize, dtype: &str, profile: &EnergyProfile) -> f64 {
        let fmas = 2.0 * m as f64 * n as f64 * k as f64;
        let compute = fmas * profile.fma_cost(dtype);
        // Memory: read A + B, write C (in elements, assume 4 bytes for fp32 etc)
        let bytes_read = (m * k + k * n) as f64 * dtype_bytes(dtype);
        let bytes_write = (m * n) as f64 * dtype_bytes(dtype);
        let mem = (bytes_read + bytes_write) / 32.0 * profile.dram_read;
        compute + mem
    }

    /// Multi-head attention cost: Q*K^T + softmax + attn*V
    pub fn attention_cost(seq_len: usize, d_model: usize, dtype: &str, profile: &EnergyProfile) -> f64 {
        // QK^T: [seq, d] x [d, seq] => matmul(seq, seq, d)
        let qk = Self::matmul_cost(seq_len, seq_len, d_model, dtype, profile);
        // softmax: ~5 ops per element
        let softmax = (seq_len * seq_len) as f64 * 5.0 * profile.fp32_fma;
        // attn * V: [seq, seq] x [seq, d]
        let av = Self::matmul_cost(seq_len, d_model, seq_len, dtype, profile);
        qk + softmax + av
    }

    /// Convolution cost
    pub fn conv_cost(batch: usize, channels: usize, spatial: usize, kernel: usize, profile: &EnergyProfile) -> f64 {
        let output_spatial = if spatial >= kernel { spatial - kernel + 1 } else { 1 };
        let fmas = 2.0 * batch as f64 * channels as f64 * output_spatial as f64 * kernel as f64 * kernel as f64;
        let compute = fmas * profile.fp32_fma;
        let mem = (batch * channels * spatial) as f64 * 4.0 / 32.0 * profile.dram_read;
        compute + mem
    }

    /// Generic layer cost by type name
    pub fn layer_cost(layer_type: &str, dims: &[usize], profile: &EnergyProfile) -> f64 {
        match layer_type {
            "matmul" | "linear" => {
                let (m, n, k) = (
                    dims.first().copied().unwrap_or(1),
                    dims.get(1).copied().unwrap_or(1),
                    dims.get(2).copied().unwrap_or(1),
                );
                Self::matmul_cost(m, n, k, "fp32", profile)
            }
            "attention" => {
                let seq = dims.first().copied().unwrap_or(128);
                let d = dims.get(1).copied().unwrap_or(64);
                Self::attention_cost(seq, d, "fp32", profile)
            }
            "conv" => {
                let b = dims.first().copied().unwrap_or(1);
                let c = dims.get(1).copied().unwrap_or(64);
                let s = dims.get(2).copied().unwrap_or(32);
                let k = dims.get(3).copied().unwrap_or(3);
                Self::conv_cost(b, c, s, k, profile)
            }
            "relu" | "gelu" | "layernorm" => {
                let elems: f64 = dims.iter().map(|&d| d as f64).product();
                elems * profile.fp32_mul * 3.0
            }
            _ => {
                // Unknown layer: rough estimate
                let elems: f64 = dims.iter().map(|&d| d as f64).product();
                elems * profile.fp32_fma
            }
        }
    }

    /// Total model cost across all layers
    pub fn model_cost(layers: &[(&str, Vec<usize>)], profile: &EnergyProfile) -> f64 {
        layers.iter().map(|(ty, dims)| Self::layer_cost(ty, dims, profile)).sum()
    }
}

fn dtype_bytes(dtype: &str) -> f64 {
    match dtype {
        "fp32" | "FP32" => 4.0,
        "fp16" | "FP16" => 2.0,
        "int8" | "INT8" => 1.0,
        "int4" | "INT4" => 0.5,
        _ => 4.0,
    }
}

// ---------------------------------------------------------------------------
// 4. EnergyAwareScheduler
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct ScheduleDecision {
    pub operation: String,
    pub dtype: String,
    pub skip: bool,
    pub reason: String,
}

pub struct EnergyAwareScheduler;

impl EnergyAwareScheduler {
    pub fn schedule(
        operations: &[(&str, Vec<usize>)],
        budget: &EnergyBudget,
        profile: &EnergyProfile,
    ) -> Vec<ScheduleDecision> {
        let remaining = budget.remaining();
        let total_ops = operations.len();
        let mut decisions = Vec::new();

        for (i, (op, dims)) in operations.iter().enumerate() {
            let remaining_ops = total_ops - i;
            let per_op_budget = remaining / remaining_ops.max(1) as f64;

            let fp32_cost = OperationCost::layer_cost(op, dims, profile);
            let fp16_cost = fp32_cost * 0.25; // fp16 roughly 1/4 energy
            let int8_cost = fp32_cost * 0.07;
            let int4_cost = fp32_cost * 0.02;

            let (dtype, skip, reason) = if per_op_budget >= fp32_cost {
                ("FP32".to_string(), false, "sufficient budget for full precision".to_string())
            } else if per_op_budget >= fp16_cost {
                ("FP16".to_string(), false, "reduced to FP16 to fit budget".to_string())
            } else if per_op_budget >= int8_cost {
                ("INT8".to_string(), false, "reduced to INT8 to fit budget".to_string())
            } else if per_op_budget >= int4_cost {
                ("INT4".to_string(), false, "reduced to INT4 to fit budget".to_string())
            } else {
                // Can't afford even INT4 — skip non-essential layers
                let essential = matches!(*op, "attention" | "matmul" | "linear");
                if essential {
                    ("INT4".to_string(), false, "essential layer kept at INT4".to_string())
                } else {
                    ("INT4".to_string(), true, "skipped: budget exhausted".to_string())
                }
            };

            decisions.push(ScheduleDecision {
                operation: op.to_string(),
                dtype,
                skip,
                reason,
            });
        }
        decisions
    }

    pub fn adaptive_precision(remaining_budget: f64, remaining_ops: usize) -> String {
        if remaining_ops == 0 { return "FP32".to_string(); }
        let per_op = remaining_budget / remaining_ops as f64;
        if per_op > 1e6 {
            "FP32".to_string()
        } else if per_op > 1e5 {
            "FP16".to_string()
        } else if per_op > 1e4 {
            "INT8".to_string()
        } else {
            "INT4".to_string()
        }
    }
}

// ---------------------------------------------------------------------------
// 5. CarbonTracker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct CarbonReport {
    pub total_joules: f64,
    pub total_carbon_g: f64,
    pub equivalent_km_driving: f64,
    pub equivalent_phone_charges: f64,
    pub green_fraction: f64,
}

#[derive(Debug, Clone)]
pub struct CarbonTracker {
    pub carbon_intensity: f64, // gCO2/kWh
    pub total_carbon_g: f64,
    total_joules: f64,
    green_joules: f64,
    green_threshold: f64, // gCO2/kWh considered "green"
}

impl CarbonTracker {
    pub fn new(carbon_intensity: f64) -> Self {
        Self {
            carbon_intensity,
            total_carbon_g: 0.0,
            total_joules: 0.0,
            green_joules: 0.0,
            green_threshold: 100.0, // below 100 gCO2/kWh is green
        }
    }

    pub fn energy_to_carbon(&self, joules: f64) -> f64 {
        // joules -> kWh: divide by 3_600_000
        // kWh * gCO2/kWh = gCO2
        joules / 3_600_000.0 * self.carbon_intensity
    }

    pub fn track(&mut self, joules: f64) {
        let carbon = self.energy_to_carbon(joules);
        self.total_carbon_g += carbon;
        self.total_joules += joules;
        if self.is_green_period() {
            self.green_joules += joules;
        }
    }

    pub fn is_green_period(&self) -> bool {
        self.carbon_intensity < self.green_threshold
    }

    /// Schedule compute during green periods given a deadline.
    /// Returns Vec<(start_hour, duration_hours)> windows.
    /// Simple model: assume intensity is sinusoidal with min at 2pm (solar peak).
    pub fn schedule_for_green(&self, deadline_hours: f64, compute_hours: f64) -> Vec<(f64, f64)> {
        if compute_hours >= deadline_hours {
            return vec![(0.0, deadline_hours)];
        }
        let mut windows = Vec::new();
        let mut scheduled = 0.0;
        // Simulate hour-by-hour; assume intensity follows pattern:
        // base_intensity * (1 + 0.5 * cos((hour - 14) * pi / 12))
        // Low at hour 14 (solar peak), high at hour 2 (night)
        let mut hour = 0.0_f64;
        while hour < deadline_hours && scheduled < compute_hours {
            let intensity = self.carbon_intensity * (1.0 + 0.5 * ((hour - 14.0) * std::f64::consts::PI / 12.0).cos());
            if intensity < self.green_threshold {
                let chunk = (compute_hours - scheduled).min(1.0);
                windows.push((hour, chunk));
                scheduled += chunk;
            }
            hour += 1.0;
        }
        // If we couldn't fill all green hours, schedule remaining at end
        if scheduled < compute_hours {
            windows.push((hour.min(deadline_hours - (compute_hours - scheduled)), compute_hours - scheduled));
        }
        windows
    }

    pub fn carbon_report(&self) -> CarbonReport {
        let green_fraction = if self.total_joules > 0.0 {
            self.green_joules / self.total_joules
        } else {
            0.0
        };
        CarbonReport {
            total_joules: self.total_joules,
            total_carbon_g: self.total_carbon_g,
            // Average car emits ~120 gCO2/km
            equivalent_km_driving: self.total_carbon_g / 120.0,
            // Charging a phone uses ~0.012 kWh => ~5.4 gCO2 at 450 gCO2/kWh average
            equivalent_phone_charges: self.total_carbon_g / 5.4,
            green_fraction,
        }
    }
}

// ---------------------------------------------------------------------------
// 7. EnergyOptimizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub enum Optimization {
    Quantize { layer: usize, from_dtype: String, to_dtype: String },
    Prune { layer: usize, fraction: f64 },
    SkipLayer { layer: usize },
    FuseOps { layer1: usize, layer2: usize },
    ReduceDim { layer: usize, new_dim: usize },
}

pub struct EnergyOptimizer;

impl EnergyOptimizer {
    pub fn optimize_model(
        layers: &[(&str, Vec<usize>)],
        target_reduction: f64,
        profile: &EnergyProfile,
    ) -> Vec<Optimization> {
        let original_cost = OperationCost::model_cost(layers, profile);
        let target_cost = original_cost * (1.0 - target_reduction);
        let mut optimizations = Vec::new();
        let mut current_cost = original_cost;

        // Strategy: greedily apply optimizations from cheapest quality loss to most
        // 1. Quantize non-attention layers to FP16
        for (i, (ty, _dims)) in layers.iter().enumerate() {
            if current_cost <= target_cost { break; }
            if *ty != "attention" {
                let layer_cost = OperationCost::layer_cost(ty, _dims, profile);
                current_cost -= layer_cost * 0.75; // FP16 saves ~75%
                optimizations.push(Optimization::Quantize {
                    layer: i,
                    from_dtype: "FP32".to_string(),
                    to_dtype: "FP16".to_string(),
                });
            }
        }

        // 2. Quantize attention layers to FP16
        for (i, (ty, dims)) in layers.iter().enumerate() {
            if current_cost <= target_cost { break; }
            if *ty == "attention" {
                let layer_cost = OperationCost::layer_cost(ty, dims, profile);
                current_cost -= layer_cost * 0.75;
                optimizations.push(Optimization::Quantize {
                    layer: i,
                    from_dtype: "FP32".to_string(),
                    to_dtype: "FP16".to_string(),
                });
            }
        }

        // 3. Fuse adjacent compatible layers
        for i in 0..layers.len().saturating_sub(1) {
            if current_cost <= target_cost { break; }
            let (t1, _) = &layers[i];
            let (t2, _) = &layers[i + 1];
            if (*t1 == "matmul" || *t1 == "linear") && (*t2 == "relu" || *t2 == "gelu") {
                current_cost -= original_cost * 0.02; // ~2% saving from fusion
                optimizations.push(Optimization::FuseOps { layer1: i, layer2: i + 1 });
            }
        }

        // 4. Prune large layers
        for (i, (_ty, dims)) in layers.iter().enumerate() {
            if current_cost <= target_cost { break; }
            let size: usize = dims.iter().product();
            if size > 10000 {
                let layer_cost = OperationCost::layer_cost(_ty, dims, profile);
                current_cost -= layer_cost * 0.3;
                optimizations.push(Optimization::Prune { layer: i, fraction: 0.3 });
            }
        }

        // 5. Skip non-essential layers
        for (i, (ty, dims)) in layers.iter().enumerate() {
            if current_cost <= target_cost { break; }
            if *ty == "relu" || *ty == "gelu" || *ty == "layernorm" {
                let layer_cost = OperationCost::layer_cost(ty, dims, profile);
                current_cost -= layer_cost;
                optimizations.push(Optimization::SkipLayer { layer: i });
            }
        }

        // 6. ReduceDim as last resort
        for (i, (_ty, dims)) in layers.iter().enumerate() {
            if current_cost <= target_cost { break; }
            if dims.len() >= 2 {
                let new_dim = dims[1] / 2;
                let layer_cost = OperationCost::layer_cost(_ty, dims, profile);
                current_cost -= layer_cost * 0.5;
                optimizations.push(Optimization::ReduceDim { layer: i, new_dim });
            }
        }

        optimizations
    }

    pub fn estimated_quality_loss(optimizations: &[Optimization]) -> f64 {
        let mut loss = 0.0;
        for opt in optimizations {
            loss += match opt {
                Optimization::Quantize { to_dtype, .. } => match to_dtype.as_str() {
                    "FP16" => 0.005,
                    "INT8" => 0.02,
                    "INT4" => 0.08,
                    _ => 0.01,
                },
                Optimization::Prune { fraction, .. } => fraction * 0.15,
                Optimization::SkipLayer { .. } => 0.03,
                Optimization::FuseOps { .. } => 0.0,
                Optimization::ReduceDim { .. } => 0.05,
            };
        }
        loss.min(1.0)
    }

    /// Pareto frontier: returns (energy_fraction, quality_fraction) pairs
    pub fn pareto_frontier(
        layers: &[(&str, Vec<usize>)],
        profile: &EnergyProfile,
    ) -> Vec<(f64, f64)> {
        let mut points = Vec::new();
        // No optimization
        points.push((1.0, 1.0));
        // Various reduction targets
        for &target in &[0.1, 0.2, 0.3, 0.5, 0.7, 0.9] {
            let opts = Self::optimize_model(layers, target, profile);
            let quality = 1.0 - Self::estimated_quality_loss(&opts);
            let energy = 1.0 - target;
            points.push((energy, quality));
        }
        points
    }
}

// ---------------------------------------------------------------------------
// Interpreter builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("energy_profile_h100".to_string(), FnDef::Builtin(builtin_energy_profile_h100));
    env.functions.insert("energy_matmul_cost".to_string(), FnDef::Builtin(builtin_energy_matmul_cost));
    env.functions.insert("energy_budget_new".to_string(), FnDef::Builtin(builtin_energy_budget_new));
    env.functions.insert("energy_budget_spend".to_string(), FnDef::Builtin(builtin_energy_budget_spend));
    env.functions.insert("energy_model_cost".to_string(), FnDef::Builtin(builtin_energy_model_cost));
    env.functions.insert("carbon_track".to_string(), FnDef::Builtin(builtin_carbon_track));
    env.functions.insert("energy_optimize".to_string(), FnDef::Builtin(builtin_energy_optimize));
}

// Global stores for interpreter state (keyed by id)
use std::sync::Mutex;
use std::sync::LazyLock;

static PROFILES: LazyLock<Mutex<HashMap<usize, EnergyProfile>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static PROFILE_COUNTER: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));
static BUDGETS: LazyLock<Mutex<HashMap<usize, EnergyBudget>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static BUDGET_COUNTER: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));
static TRACKER: LazyLock<Mutex<CarbonTracker>> = LazyLock::new(|| Mutex::new(CarbonTracker::new(450.0)));

fn builtin_energy_profile_h100(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() { return Err("energy_profile_h100 takes 0 args".into()); }
    let mut counter = PROFILE_COUNTER.lock().unwrap();
    let id = *counter;
    *counter += 1;
    PROFILES.lock().unwrap().insert(id, EnergyProfile::h100());
    Ok(Value::Int(id as i128))
}

fn builtin_energy_matmul_cost(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 { return Err("energy_matmul_cost(m, n, k, dtype, profile_id)".into()); }
    let m = match &args[0] { Value::Int(v) => *v as usize, _ => return Err("m must be int".into()) };
    let n = match &args[1] { Value::Int(v) => *v as usize, _ => return Err("n must be int".into()) };
    let k = match &args[2] { Value::Int(v) => *v as usize, _ => return Err("k must be int".into()) };
    let dtype = match &args[3] { Value::String(s) => s.clone(), _ => return Err("dtype must be string".into()) };
    let pid = match &args[4] { Value::Int(v) => *v as usize, _ => return Err("profile_id must be int".into()) };
    let profiles = PROFILES.lock().unwrap();
    let profile = profiles.get(&pid).ok_or("unknown profile id")?;
    let cost = OperationCost::matmul_cost(m, n, k, &dtype, profile);
    Ok(Value::Float(cost))
}

fn builtin_energy_budget_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("energy_budget_new(joules)".into()); }
    let joules = match &args[0] {
        Value::Float(v) => *v,
        Value::Int(v) => *v as f64,
        _ => return Err("joules must be numeric".into()),
    };
    let mut counter = BUDGET_COUNTER.lock().unwrap();
    let id = *counter;
    *counter += 1;
    BUDGETS.lock().unwrap().insert(id, EnergyBudget::new(joules));
    Ok(Value::Int(id as i128))
}

fn builtin_energy_budget_spend(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("energy_budget_spend(id, joules)".into()); }
    let id = match &args[0] { Value::Int(v) => *v as usize, _ => return Err("id must be int".into()) };
    let joules = match &args[1] {
        Value::Float(v) => *v,
        Value::Int(v) => *v as f64,
        _ => return Err("joules must be numeric".into()),
    };
    let mut budgets = BUDGETS.lock().unwrap();
    let budget = budgets.get_mut(&id).ok_or("unknown budget id")?;
    budget.spend(joules)?;
    Ok(Value::Float(budget.remaining()))
}

fn builtin_energy_model_cost(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("energy_model_cost(layer_specs, profile_id)".into()); }
    let layers = parse_layer_specs(&args[0])?;
    let pid = match &args[1] { Value::Int(v) => *v as usize, _ => return Err("profile_id must be int".into()) };
    let profiles = PROFILES.lock().unwrap();
    let profile = profiles.get(&pid).ok_or("unknown profile id")?;
    let layer_refs: Vec<(&str, Vec<usize>)> = layers.iter().map(|(s, d)| (s.as_str(), d.clone())).collect();
    let cost = OperationCost::model_cost(&layer_refs, profile);
    Ok(Value::Float(cost))
}

fn builtin_carbon_track(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("carbon_track(joules)".into()); }
    let joules = match &args[0] {
        Value::Float(v) => *v,
        Value::Int(v) => *v as f64,
        _ => return Err("joules must be numeric".into()),
    };
    let mut tracker = TRACKER.lock().unwrap();
    let carbon = tracker.energy_to_carbon(joules);
    tracker.track(joules);
    Ok(Value::Float(carbon))
}

fn builtin_energy_optimize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("energy_optimize(layer_specs, target_reduction, profile_id)".into()); }
    let layers = parse_layer_specs(&args[0])?;
    let target = match &args[1] {
        Value::Float(v) => *v,
        Value::Int(v) => *v as f64,
        _ => return Err("target_reduction must be numeric".into()),
    };
    let pid = match &args[2] { Value::Int(v) => *v as usize, _ => return Err("profile_id must be int".into()) };
    let profiles = PROFILES.lock().unwrap();
    let profile = profiles.get(&pid).ok_or("unknown profile id")?;
    let layer_refs: Vec<(&str, Vec<usize>)> = layers.iter().map(|(s, d)| (s.as_str(), d.clone())).collect();
    let opts = EnergyOptimizer::optimize_model(&layer_refs, target, profile);
    // Return as array of strings describing each optimization
    let descriptions: Vec<Value> = opts.iter().map(|o| Value::String(format!("{:?}", o))).collect();
    Ok(Value::Array(descriptions))
}

fn parse_layer_specs(val: &Value) -> Result<Vec<(String, Vec<usize>)>, String> {
    match val {
        Value::Array(arr) => {
            let mut layers = Vec::new();
            for item in arr {
                match item {
                    Value::Array(pair) if pair.len() == 2 => {
                        let name = match &pair[0] {
                            Value::String(s) => s.clone(),
                            _ => return Err("layer name must be string".into()),
                        };
                        let dims = match &pair[1] {
                            Value::Array(d) => d.iter().map(|v| match v {
                                Value::Int(i) => Ok(*i as usize),
                                _ => Err("dim must be int".to_string()),
                            }).collect::<Result<Vec<_>, String>>()?,
                            _ => return Err("layer dims must be array".into()),
                        };
                        layers.push((name, dims));
                    }
                    _ => return Err("each layer spec must be [name, [dims]]".into()),
                }
            }
            Ok(layers)
        }
        _ => Err("layer_specs must be an array".into()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_energy_profile_h100_values() {
        let p = EnergyProfile::h100();
        assert!(p.fp32_fma > p.fp16_fma);
        assert!(p.fp16_fma > p.int8_fma);
        assert!(p.int8_fma > p.int4_fma);
        assert!(p.dram_read > p.sram_read);
        assert!(p.sram_read > p.register_read);
    }

    #[test]
    fn test_energy_profile_all_hardware() {
        let h = EnergyProfile::h100();
        let a = EnergyProfile::a100();
        let r = EnergyProfile::rtx4090();
        let c = EnergyProfile::cpu_x86();
        // H100 should be most efficient for FP16
        assert!(h.fp16_fma < a.fp16_fma);
        // CPU should be least efficient
        assert!(c.fp32_fma > h.fp32_fma);
        assert!(r.fp32_fma > 0.0);
    }

    #[test]
    fn test_energy_budget_basic() {
        let mut b = EnergyBudget::new(10.0);
        assert_eq!(b.remaining(), 10.0);
        assert!(b.can_afford(5.0));
        assert!(b.spend(3.0).is_ok());
        assert_eq!(b.remaining(), 7.0);
        assert!(!b.can_afford(8.0));
        assert!(b.spend(8.0).is_err());
    }

    #[test]
    fn test_energy_budget_efficiency() {
        let mut b = EnergyBudget::new(100.0);
        b.spend(50.0).unwrap();
        assert!((b.efficiency() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matmul_cost_scales_with_size() {
        let p = EnergyProfile::h100();
        let small = OperationCost::matmul_cost(32, 32, 32, "fp32", &p);
        let large = OperationCost::matmul_cost(64, 64, 64, "fp32", &p);
        assert!(large > small * 4.0); // cubic scaling
    }

    #[test]
    fn test_matmul_cost_dtype_affects_energy() {
        let p = EnergyProfile::h100();
        let fp32 = OperationCost::matmul_cost(128, 128, 128, "fp32", &p);
        let fp16 = OperationCost::matmul_cost(128, 128, 128, "fp16", &p);
        let int8 = OperationCost::matmul_cost(128, 128, 128, "int8", &p);
        assert!(fp32 > fp16);
        assert!(fp16 > int8);
    }

    #[test]
    fn test_attention_cost_positive() {
        let p = EnergyProfile::h100();
        let cost = OperationCost::attention_cost(512, 64, "fp32", &p);
        assert!(cost > 0.0);
    }

    #[test]
    fn test_model_cost_sums_layers() {
        let p = EnergyProfile::h100();
        let layers: Vec<(&str, Vec<usize>)> = vec![
            ("matmul", vec![128, 128, 128]),
            ("relu", vec![128, 128]),
            ("matmul", vec![128, 64, 128]),
        ];
        let total = OperationCost::model_cost(&layers, &p);
        let sum: f64 = layers.iter().map(|(t, d)| OperationCost::layer_cost(t, d, &p)).sum();
        assert!((total - sum).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_respects_budget() {
        let p = EnergyProfile::h100();
        let ops: Vec<(&str, Vec<usize>)> = vec![
            ("matmul", vec![1024, 1024, 1024]),
            ("relu", vec![1024, 1024]),
            ("attention", vec![512, 64]),
        ];
        // Tiny budget forces lower precision
        let budget = EnergyBudget::new(1.0);
        let decisions = EnergyAwareScheduler::schedule(&ops, &budget, &p);
        assert_eq!(decisions.len(), 3);
        // With tiny budget, should not all be FP32
        let has_low_precision = decisions.iter().any(|d| d.dtype != "FP32");
        assert!(has_low_precision);
    }

    #[test]
    fn test_carbon_tracker() {
        let mut ct = CarbonTracker::new(50.0); // very green grid
        assert!(ct.is_green_period());
        ct.track(3_600_000.0); // 1 kWh
        assert!((ct.total_carbon_g - 50.0).abs() < 0.01);
        let report = ct.carbon_report();
        assert!(report.equivalent_km_driving > 0.0);
        assert!(report.green_fraction > 0.0);
    }

    #[test]
    fn test_carbon_energy_to_carbon() {
        let ct = CarbonTracker::new(400.0);
        // 1 kWh = 3_600_000 J => 400 gCO2
        let carbon = ct.energy_to_carbon(3_600_000.0);
        assert!((carbon - 400.0).abs() < 0.01);
    }

    #[test]
    fn test_optimizer_produces_optimizations() {
        let p = EnergyProfile::h100();
        let layers: Vec<(&str, Vec<usize>)> = vec![
            ("matmul", vec![256, 256, 256]),
            ("relu", vec![256, 256]),
            ("attention", vec![128, 64]),
            ("layernorm", vec![256]),
        ];
        let opts = EnergyOptimizer::optimize_model(&layers, 0.5, &p);
        assert!(!opts.is_empty());
        let loss = EnergyOptimizer::estimated_quality_loss(&opts);
        assert!(loss > 0.0 && loss <= 1.0);
    }

    #[test]
    fn test_pareto_frontier() {
        let p = EnergyProfile::h100();
        let layers: Vec<(&str, Vec<usize>)> = vec![
            ("matmul", vec![128, 128, 128]),
            ("attention", vec![64, 32]),
        ];
        let frontier = EnergyOptimizer::pareto_frontier(&layers, &p);
        assert!(frontier.len() >= 2);
        // First point should be (1.0, 1.0) - no optimization
        assert!((frontier[0].0 - 1.0).abs() < 1e-10);
        assert!((frontier[0].1 - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_adaptive_precision() {
        assert_eq!(EnergyAwareScheduler::adaptive_precision(1e8, 10), "FP32");
        assert_eq!(EnergyAwareScheduler::adaptive_precision(1e7, 10), "FP16");
        assert_eq!(EnergyAwareScheduler::adaptive_precision(1e6, 10), "INT8");
        assert_eq!(EnergyAwareScheduler::adaptive_precision(100.0, 10), "INT4");
    }

    #[test]
    fn test_schedule_for_green() {
        let ct = CarbonTracker::new(50.0); // green grid
        let windows = ct.schedule_for_green(24.0, 4.0);
        assert!(!windows.is_empty());
        let total_scheduled: f64 = windows.iter().map(|(_, d)| d).sum();
        assert!((total_scheduled - 4.0).abs() < 0.01);
    }
}
