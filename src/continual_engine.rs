// Continual Learning Engine: EWC, PackNet, Progressive Nets, Experience Replay
// Enables models to learn new tasks without catastrophic forgetting.

use crate::interpreter::{Value, Env, FnDef};
use std::sync::{LazyLock, Mutex};

// --- State storage ---

struct EwcState {
    lambda: f64,
    // Per-task: (optimal_params, fisher_diagonal)
    tasks: Vec<(Vec<f64>, Vec<f64>)>,
}

struct PackNetState {
    total_params: usize,
    // task_id -> binary mask (true = allocated to that task)
    masks: Vec<Vec<bool>>,
    frozen: Vec<bool>, // which tasks are frozen
}

struct ReplayBuffer {
    capacity: usize,
    buffer: Vec<Value>,
}

struct ProgNetColumn {
    dims: Vec<usize>,      // layer dimensions
    weights: Vec<Vec<f64>>, // flattened weight matrices per layer
    lateral: Vec<Vec<f64>>, // lateral adapter weights from previous columns
}

struct ProgNet {
    base_dims: Vec<usize>,
    columns: Vec<ProgNetColumn>,
}

static EWC_STATES: LazyLock<Mutex<Vec<EwcState>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static PACKNET_STATES: LazyLock<Mutex<Vec<PackNetState>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static REPLAY_BUFFERS: LazyLock<Mutex<Vec<ReplayBuffer>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static PROG_NETS: LazyLock<Mutex<Vec<ProgNet>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// --- Helpers ---

fn extract_f64_array(val: &Value) -> Result<Vec<f64>, String> {
    match val {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("Expected numeric array".into()),
        }).collect(),
        _ => Err("Expected array".into()),
    }
}

fn extract_usize_array(val: &Value) -> Result<Vec<usize>, String> {
    match val {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            _ => Err("Expected int array".into()),
        }).collect(),
        _ => Err("Expected array".into()),
    }
}

// --- EWC builtins ---

fn builtin_ewc_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("ewc_create(lambda)".into()); }
    let lambda = match &args[0] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("lambda must be numeric".into()),
    };
    let mut states = EWC_STATES.lock().unwrap();
    let id = states.len();
    states.push(EwcState { lambda, tasks: Vec::new() });
    Ok(Value::Int(id as i128))
}

fn builtin_ewc_register_task(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ewc_register_task(id, params, fisher_diag)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let params = extract_f64_array(&args[1])?;
    let fisher = extract_f64_array(&args[2])?;
    if params.len() != fisher.len() {
        return Err("params and fisher_diag must have same length".into());
    }
    let mut states = EWC_STATES.lock().unwrap();
    let st = states.get_mut(id).ok_or("Invalid EWC id")?;
    st.tasks.push((params, fisher));
    Ok(Value::Int(st.tasks.len() as i128 - 1))
}

fn builtin_ewc_penalty(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ewc_penalty(id, current_params)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let current = extract_f64_array(&args[1])?;
    let states = EWC_STATES.lock().unwrap();
    let st = states.get(id).ok_or("Invalid EWC id")?;
    let mut penalty = 0.0;
    for (opt_params, fisher) in &st.tasks {
        if current.len() != opt_params.len() {
            return Err("Parameter dimension mismatch".into());
        }
        for i in 0..current.len() {
            let diff = current[i] - opt_params[i];
            penalty += fisher[i] * diff * diff;
        }
    }
    penalty *= st.lambda / 2.0;
    Ok(Value::Float(penalty))
}

// --- PackNet builtins ---

fn builtin_packnet_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("packnet_create(total_params)".into()); }
    let total = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("total_params must be int".into()) };
    let mut states = PACKNET_STATES.lock().unwrap();
    let id = states.len();
    states.push(PackNetState { total_params: total, masks: Vec::new(), frozen: Vec::new() });
    Ok(Value::Int(id as i128))
}

fn builtin_packnet_prune(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("packnet_prune(id, task_id, mask_threshold, magnitudes)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let _task_id = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("task_id must be int".into()) };
    let threshold = match &args[2] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("threshold must be numeric".into()),
    };
    let magnitudes = extract_f64_array(&args[3])?;
    let mut states = PACKNET_STATES.lock().unwrap();
    let st = states.get_mut(id).ok_or("Invalid PackNet id")?;
    if magnitudes.len() != st.total_params {
        return Err(format!("Expected {} magnitudes, got {}", st.total_params, magnitudes.len()));
    }
    // Build mask: true if magnitude >= threshold AND not already frozen by another task
    let mut mask = vec![false; st.total_params];
    for i in 0..st.total_params {
        let already_taken = st.masks.iter().enumerate().any(|(tid, m)| {
            tid < st.frozen.len() && st.frozen[tid] && m[i]
        });
        mask[i] = !already_taken && magnitudes[i] >= threshold;
    }
    let result = mask.iter().map(|&b| Value::Bool(b)).collect();
    st.masks.push(mask);
    st.frozen.push(false);
    Ok(Value::Array(result))
}

fn builtin_packnet_freeze(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("packnet_freeze(id, task_id)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let task_id = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("task_id must be int".into()) };
    let mut states = PACKNET_STATES.lock().unwrap();
    let st = states.get_mut(id).ok_or("Invalid PackNet id")?;
    if task_id >= st.frozen.len() { return Err("Invalid task_id".into()); }
    st.frozen[task_id] = true;
    Ok(Value::Bool(true))
}

// --- Experience Replay builtins ---

fn builtin_replay_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("replay_create(capacity)".into()); }
    let cap = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("capacity must be int".into()) };
    let mut bufs = REPLAY_BUFFERS.lock().unwrap();
    let id = bufs.len();
    bufs.push(ReplayBuffer { capacity: cap, buffer: Vec::new() });
    Ok(Value::Int(id as i128))
}

fn builtin_replay_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("replay_add(id, experience)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let exp = args[1].clone();
    let mut bufs = REPLAY_BUFFERS.lock().unwrap();
    let buf = bufs.get_mut(id).ok_or("Invalid replay buffer id")?;
    if buf.buffer.len() >= buf.capacity {
        // Reservoir-style: replace random element using simple modular index
        let idx = buf.buffer.len() % buf.capacity;
        buf.buffer[idx] = exp;
    } else {
        buf.buffer.push(exp);
    }
    Ok(Value::Bool(true))
}

fn builtin_replay_sample(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("replay_sample(id, batch_size)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let batch = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("batch_size must be int".into()) };
    let bufs = REPLAY_BUFFERS.lock().unwrap();
    let buf = bufs.get(id).ok_or("Invalid replay buffer id")?;
    if buf.buffer.is_empty() { return Ok(Value::Array(vec![])); }
    // Deterministic pseudo-random sampling using a simple LCG seeded by buffer len
    let mut seed: u64 = buf.buffer.len() as u64 * 2654435761;
    let mut samples = Vec::with_capacity(batch);
    for _ in 0..batch {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let idx = (seed >> 33) as usize % buf.buffer.len();
        samples.push(buf.buffer[idx].clone());
    }
    Ok(Value::Array(samples))
}

// --- Progressive Nets builtins ---

fn builtin_prognet_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("prognet_create(base_dims)".into()); }
    let dims = extract_usize_array(&args[0])?;
    if dims.len() < 2 { return Err("Need at least input and output dims".into()); }
    // Create first column with zero-initialized weights
    let mut weights = Vec::new();
    for i in 0..dims.len() - 1 {
        weights.push(vec![0.0; dims[i] * dims[i + 1]]);
    }
    let col = ProgNetColumn { dims: dims.clone(), weights, lateral: Vec::new() };
    let mut nets = PROG_NETS.lock().unwrap();
    let id = nets.len();
    nets.push(ProgNet { base_dims: dims, columns: vec![col] });
    Ok(Value::Int(id as i128))
}

fn builtin_prognet_add_column(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("prognet_add_column(id)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let mut nets = PROG_NETS.lock().unwrap();
    let net = nets.get_mut(id).ok_or("Invalid prognet id")?;
    let dims = &net.base_dims;
    let num_prev = net.columns.len();
    let mut weights = Vec::new();
    let mut lateral = Vec::new();
    for i in 0..dims.len() - 1 {
        weights.push(vec![0.0; dims[i] * dims[i + 1]]);
        // Lateral connections from each previous column's layer i output
        lateral.push(vec![0.1; num_prev * dims[i + 1]]);
    }
    let col_id = net.columns.len();
    net.columns.push(ProgNetColumn { dims: dims.clone(), weights, lateral });
    Ok(Value::Int(col_id as i128))
}

fn builtin_prognet_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("prognet_forward(id, column, input)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let col_idx = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("column must be int".into()) };
    let input = extract_f64_array(&args[2])?;
    let nets = PROG_NETS.lock().unwrap();
    let net = nets.get(id).ok_or("Invalid prognet id")?;
    if col_idx >= net.columns.len() { return Err("Invalid column index".into()); }
    let dims = &net.base_dims;
    if input.len() != dims[0] { return Err(format!("Expected input dim {}", dims[0])); }
    // Forward pass through the column with lateral connections
    let col = &net.columns[col_idx];
    let mut activation = input;
    for layer in 0..dims.len() - 1 {
        let out_dim = dims[layer + 1];
        let mut output = vec![0.0; out_dim];
        // Main weights: matrix-vector multiply
        for j in 0..out_dim {
            let mut sum = 0.0;
            for k in 0..activation.len() {
                sum += col.weights[layer][k * out_dim + j] * activation[k];
            }
            output[j] = sum;
        }
        // Add lateral adapter contributions (scaled sum from previous columns)
        if col_idx > 0 && !col.lateral.is_empty() {
            for j in 0..out_dim {
                let mut lat_sum = 0.0;
                for p in 0..col_idx {
                    let lat_idx = p * out_dim + j;
                    if lat_idx < col.lateral[layer].len() {
                        lat_sum += col.lateral[layer][lat_idx];
                    }
                }
                output[j] += lat_sum;
            }
        }
        // ReLU activation
        for v in &mut output { if *v < 0.0 { *v = 0.0; } }
        activation = output;
    }
    Ok(Value::Array(activation.into_iter().map(Value::Float).collect()))
}

// --- Drift detection ---

fn builtin_continual_detect_drift(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("continual_detect_drift(old_dist, new_dist)".into()); }
    let old = extract_f64_array(&args[0])?;
    let new = extract_f64_array(&args[1])?;
    if old.len() != new.len() { return Err("Distributions must have same length".into()); }
    // KL divergence: sum p(x) * ln(p(x)/q(x))
    let mut kl = 0.0;
    for i in 0..old.len() {
        let p = old[i].max(1e-10);
        let q = new[i].max(1e-10);
        kl += p * (p / q).ln();
    }
    Ok(Value::Float(kl))
}

// --- Registration ---

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("ewc_create".into(), FnDef::Builtin(builtin_ewc_create));
    env.functions.insert("ewc_register_task".into(), FnDef::Builtin(builtin_ewc_register_task));
    env.functions.insert("ewc_penalty".into(), FnDef::Builtin(builtin_ewc_penalty));
    env.functions.insert("packnet_create".into(), FnDef::Builtin(builtin_packnet_create));
    env.functions.insert("packnet_prune".into(), FnDef::Builtin(builtin_packnet_prune));
    env.functions.insert("packnet_freeze".into(), FnDef::Builtin(builtin_packnet_freeze));
    env.functions.insert("replay_create".into(), FnDef::Builtin(builtin_replay_create));
    env.functions.insert("replay_add".into(), FnDef::Builtin(builtin_replay_add));
    env.functions.insert("replay_sample".into(), FnDef::Builtin(builtin_replay_sample));
    env.functions.insert("prognet_create".into(), FnDef::Builtin(builtin_prognet_create));
    env.functions.insert("prognet_add_column".into(), FnDef::Builtin(builtin_prognet_add_column));
    env.functions.insert("prognet_forward".into(), FnDef::Builtin(builtin_prognet_forward));
    env.functions.insert("continual_detect_drift".into(), FnDef::Builtin(builtin_continual_detect_drift));
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env { Env::new() }

    fn as_float(v: &Value) -> f64 {
        match v { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => panic!("Expected numeric") }
    }
    fn as_bool(v: &Value) -> bool {
        match v { Value::Bool(b) => *b, _ => panic!("Expected bool") }
    }
    fn as_array(v: &Value) -> &Vec<Value> {
        match v { Value::Array(a) => a, _ => panic!("Expected array") }
    }

    #[test]
    fn test_ewc_create_and_penalty_no_tasks() {
        let mut e = env();
        let id = builtin_ewc_create(&mut e, vec![Value::Float(1.0)]).unwrap();
        let penalty = builtin_ewc_penalty(&mut e, vec![id.clone(), Value::Array(vec![Value::Float(5.0)])]).unwrap();
        assert!((as_float(&penalty) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_ewc_penalty_computation() {
        let mut e = env();
        let id = builtin_ewc_create(&mut e, vec![Value::Float(2.0)]).unwrap();
        let params = Value::Array(vec![Value::Float(1.0), Value::Float(2.0)]);
        let fisher = Value::Array(vec![Value::Float(1.0), Value::Float(1.0)]);
        builtin_ewc_register_task(&mut e, vec![id.clone(), params, fisher]).unwrap();
        let current = Value::Array(vec![Value::Float(2.0), Value::Float(4.0)]);
        let pen = builtin_ewc_penalty(&mut e, vec![id, current]).unwrap();
        assert!((as_float(&pen) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_packnet_prune_and_freeze() {
        let mut e = env();
        let id = builtin_packnet_create(&mut e, vec![Value::Int(4)]).unwrap();
        let mags = Value::Array(vec![Value::Float(0.5), Value::Float(0.1), Value::Float(0.8), Value::Float(0.3)]);
        let mask = builtin_packnet_prune(&mut e, vec![id.clone(), Value::Int(0), Value::Float(0.4), mags]).unwrap();
        let m = as_array(&mask);
        assert!(as_bool(&m[0]));
        assert!(!as_bool(&m[1]));
        assert!(as_bool(&m[2]));
        assert!(!as_bool(&m[3]));
        builtin_packnet_freeze(&mut e, vec![id, Value::Int(0)]).unwrap();
    }

    #[test]
    fn test_packnet_frozen_exclusion() {
        let mut e = env();
        let id = builtin_packnet_create(&mut e, vec![Value::Int(3)]).unwrap();
        let mags1 = Value::Array(vec![Value::Float(0.9), Value::Float(0.9), Value::Float(0.1)]);
        builtin_packnet_prune(&mut e, vec![id.clone(), Value::Int(0), Value::Float(0.5), mags1]).unwrap();
        builtin_packnet_freeze(&mut e, vec![id.clone(), Value::Int(0)]).unwrap();
        let mags2 = Value::Array(vec![Value::Float(0.9), Value::Float(0.9), Value::Float(0.9)]);
        let mask2 = builtin_packnet_prune(&mut e, vec![id, Value::Int(1), Value::Float(0.5), mags2]).unwrap();
        let m = as_array(&mask2);
        assert!(!as_bool(&m[0]));
        assert!(!as_bool(&m[1]));
        assert!(as_bool(&m[2]));
    }

    #[test]
    fn test_replay_buffer() {
        let mut e = env();
        let id = builtin_replay_create(&mut e, vec![Value::Int(3)]).unwrap();
        for i in 0..5 {
            builtin_replay_add(&mut e, vec![id.clone(), Value::Int(i)]).unwrap();
        }
        let batch = builtin_replay_sample(&mut e, vec![id, Value::Int(2)]).unwrap();
        assert_eq!(as_array(&batch).len(), 2);
    }

    #[test]
    fn test_replay_empty_sample() {
        let mut e = env();
        let id = builtin_replay_create(&mut e, vec![Value::Int(10)]).unwrap();
        let batch = builtin_replay_sample(&mut e, vec![id, Value::Int(5)]).unwrap();
        assert!(as_array(&batch).is_empty());
    }

    #[test]
    fn test_prognet_create_and_forward() {
        let mut e = env();
        let dims = Value::Array(vec![Value::Int(2), Value::Int(3), Value::Int(1)]);
        let id = builtin_prognet_create(&mut e, vec![dims]).unwrap();
        let input = Value::Array(vec![Value::Float(1.0), Value::Float(1.0)]);
        let out = builtin_prognet_forward(&mut e, vec![id, Value::Int(0), input]).unwrap();
        let o = as_array(&out);
        assert_eq!(o.len(), 1);
        assert!((as_float(&o[0])).abs() < 1e-10);
    }

    #[test]
    fn test_drift_detection() {
        let mut e = env();
        let same = Value::Array(vec![Value::Float(0.5), Value::Float(0.5)]);
        let kl = builtin_continual_detect_drift(&mut e, vec![same.clone(), same]).unwrap();
        assert!(as_float(&kl).abs() < 1e-6);
        let p = Value::Array(vec![Value::Float(0.9), Value::Float(0.1)]);
        let q = Value::Array(vec![Value::Float(0.1), Value::Float(0.9)]);
        let kl2 = builtin_continual_detect_drift(&mut e, vec![p, q]).unwrap();
        assert!(as_float(&kl2) > 0.0);
    }
}
