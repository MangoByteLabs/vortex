// agent_protocol.rs — Multi-agent coordination, NAS, and reasoning primitives
use crate::interpreter::{Value, Env, FnDef};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ── Data structures ──────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Message {
    from: usize,
    content: String,
}

#[derive(Clone, Debug)]
struct Agent {
    id: usize,
    name: String,
    role: String,
    inbox: Vec<Message>,
    state: HashMap<String, String>,
}

#[derive(Clone, Debug)]
struct ArchRecord {
    arch: Vec<String>,
    score: f64,
}

#[derive(Clone, Debug)]
struct SearchSpace {
    id: usize,
    layers: Vec<Vec<String>>,
    records: Vec<ArchRecord>,
}

// ── Global state ─────────────────────────────────────────────────────────────

static AGENTS: LazyLock<Mutex<Vec<Agent>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static NAS_SPACES: LazyLock<Mutex<Vec<SearchSpace>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ── Helpers ──────────────────────────────────────────────────────────────────

fn val_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Int(i) => i.to_string(),
        Value::Float(f) => f.to_string(),
        other => format!("{:?}", other),
    }
}

fn val_to_string_vec(v: &Value) -> Result<Vec<String>, String> {
    match v {
        Value::Array(arr) => Ok(arr.iter().map(|x| val_to_string(x)).collect()),
        _ => Err("expected array".into()),
    }
}

// ── Agent builtins ───────────────────────────────────────────────────────────

fn builtin_agent_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("agent_create(name, role)".into());
    }
    let name = val_to_string(&args[0]);
    let role = val_to_string(&args[1]);
    let mut agents = AGENTS.lock().unwrap();
    let id = agents.len();
    agents.push(Agent { id, name, role, inbox: Vec::new(), state: HashMap::new() });
    Ok(Value::Int(id as i128))
}

fn builtin_agent_send(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("agent_send(from_id, to_id, message)".into());
    }
    let from_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("from_id must be int".into()) };
    let to_id = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("to_id must be int".into()) };
    let content = val_to_string(&args[2]);
    let mut agents = AGENTS.lock().unwrap();
    if to_id >= agents.len() { return Err(format!("agent {} not found", to_id)); }
    if from_id >= agents.len() { return Err(format!("agent {} not found", from_id)); }
    agents[to_id].inbox.push(Message { from: from_id, content });
    Ok(Value::Bool(true))
}

fn builtin_agent_receive(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("agent_receive(id)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let mut agents = AGENTS.lock().unwrap();
    if id >= agents.len() { return Err(format!("agent {} not found", id)); }
    if agents[id].inbox.is_empty() {
        return Ok(Value::String("None".into()));
    }
    let msg = agents[id].inbox.remove(0);
    let mut map = HashMap::new();
    map.insert("from".to_string(), Value::Int(msg.from as i128));
    map.insert("content".to_string(), Value::String(msg.content));
    Ok(Value::Struct { name: "Message".into(), fields: map })
}

fn builtin_agent_broadcast(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("agent_broadcast(from_id, message)".into());
    }
    let from_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("from_id must be int".into()) };
    let content = val_to_string(&args[1]);
    let mut agents = AGENTS.lock().unwrap();
    if from_id >= agents.len() { return Err(format!("agent {} not found", from_id)); }
    let count = agents.len();
    for i in 0..count {
        if i != from_id {
            agents[i].inbox.push(Message { from: from_id, content: content.clone() });
        }
    }
    Ok(Value::Int((count - 1) as i128))
}

// ── NAS builtins ─────────────────────────────────────────────────────────────

fn builtin_nas_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("nas_create(search_space_array)".into());
    }
    let layers: Vec<Vec<String>> = match &args[0] {
        Value::Array(outer) => {
            outer.iter().map(|v| val_to_string_vec(v)).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("search_space must be array of arrays".into()),
    };
    let mut spaces = NAS_SPACES.lock().unwrap();
    let id = spaces.len();
    spaces.push(SearchSpace { id, layers, records: Vec::new() });
    Ok(Value::Int(id as i128))
}

fn builtin_nas_random_arch(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("nas_random_arch(space_id)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let spaces = NAS_SPACES.lock().unwrap();
    if id >= spaces.len() { return Err(format!("search space {} not found", id)); }
    let space = &spaces[id];
    // Pseudo-random: pick from each layer based on layer index + record count
    let seed = space.records.len();
    let arch: Vec<Value> = space.layers.iter().enumerate().map(|(i, opts)| {
        let pick = (seed + i * 7 + 3) % opts.len();
        Value::String(opts[pick].clone())
    }).collect();
    Ok(Value::Array(arch))
}

fn builtin_nas_evaluate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("nas_evaluate(space_id, arch, score)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let arch = val_to_string_vec(&args[1])?;
    let score = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("score must be number".into()) };
    let mut spaces = NAS_SPACES.lock().unwrap();
    if id >= spaces.len() { return Err(format!("search space {} not found", id)); }
    spaces[id].records.push(ArchRecord { arch, score });
    Ok(Value::Int(spaces[id].records.len() as i128))
}

fn builtin_nas_best(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("nas_best(space_id)".into());
    }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let spaces = NAS_SPACES.lock().unwrap();
    if id >= spaces.len() { return Err(format!("search space {} not found", id)); }
    let records = &spaces[id].records;
    if records.is_empty() { return Ok(Value::String("None".into())); }
    let best = records.iter().max_by(|a, b| a.score.partial_cmp(&b.score).unwrap()).unwrap();
    let arch_vals: Vec<Value> = best.arch.iter().map(|s| Value::String(s.clone())).collect();
    let mut fields = HashMap::new();
    fields.insert("arch".into(), Value::Array(arch_vals));
    fields.insert("score".into(), Value::Float(best.score));
    Ok(Value::Struct { name: "NASResult".into(), fields })
}

// ── Reasoning builtins ───────────────────────────────────────────────────────

fn builtin_chain_of_thought(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("chain_of_thought(steps_array)".into());
    }
    let steps = val_to_string_vec(&args[0])?;
    if steps.is_empty() { return Err("steps must be non-empty".into()); }
    let trace: Vec<Value> = steps.iter().enumerate().map(|(i, s)| {
        let mut fields = HashMap::new();
        fields.insert("step".into(), Value::Int((i + 1) as i128));
        fields.insert("thought".into(), Value::String(s.clone()));
        fields.insert("cumulative_confidence".into(), Value::Float(1.0 - 0.05 * i as f64));
        Value::Struct { name: "CoTStep".into(), fields }
    }).collect();
    let mut result = HashMap::new();
    result.insert("steps".into(), Value::Array(trace));
    result.insert("total_steps".into(), Value::Int(steps.len() as i128));
    let final_conf = 1.0 - 0.05 * (steps.len() - 1) as f64;
    result.insert("final_confidence".into(), Value::Float(final_conf.max(0.1)));
    Ok(Value::Struct { name: "CoTTrace".into(), fields: result })
}

fn builtin_tree_of_thought(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("tree_of_thought(branches_array)".into());
    }
    let branches = val_to_string_vec(&args[0])?;
    if branches.is_empty() { return Err("branches must be non-empty".into()); }
    // Score each branch by length heuristic (longer = more detailed = higher score)
    let scored: Vec<(f64, &String)> = branches.iter().map(|b| {
        let score = (b.len() as f64).ln().min(5.0) / 5.0;
        (score, b)
    }).collect();
    let best = scored.iter().max_by(|a, b| a.0.partial_cmp(&b.0).unwrap()).unwrap();
    let evaluated: Vec<Value> = scored.iter().map(|(score, text)| {
        let mut f = HashMap::new();
        f.insert("branch".into(), Value::String(text.to_string()));
        f.insert("score".into(), Value::Float(*score));
        Value::Struct { name: "Branch".into(), fields: f }
    }).collect();
    let mut result = HashMap::new();
    result.insert("branches".into(), Value::Array(evaluated));
    result.insert("best".into(), Value::String(best.1.clone()));
    result.insert("best_score".into(), Value::Float(best.0));
    Ok(Value::Struct { name: "ToTResult".into(), fields: result })
}

fn builtin_self_reflect(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("self_reflect(claim, evidence_array)".into());
    }
    let claim = val_to_string(&args[0]);
    let evidence = val_to_string_vec(&args[1])?;
    // Consistency: fraction of evidence items that share words with claim
    let claim_words: Vec<&str> = claim.split_whitespace().collect();
    let mut supporting = 0usize;
    for e in &evidence {
        let e_words: Vec<&str> = e.split_whitespace().collect();
        if e_words.iter().any(|w| claim_words.contains(w)) {
            supporting += 1;
        }
    }
    let consistency = if evidence.is_empty() { 0.0 } else { supporting as f64 / evidence.len() as f64 };
    let mut fields = HashMap::new();
    fields.insert("claim".into(), Value::String(claim));
    fields.insert("evidence_count".into(), Value::Int(evidence.len() as i128));
    fields.insert("supporting".into(), Value::Int(supporting as i128));
    fields.insert("consistency".into(), Value::Float(consistency));
    Ok(Value::Struct { name: "ReflectionResult".into(), fields })
}

// ── Registration ─────────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("agent_create".into(), FnDef::Builtin(builtin_agent_create));
    env.functions.insert("agent_send".into(), FnDef::Builtin(builtin_agent_send));
    env.functions.insert("agent_receive".into(), FnDef::Builtin(builtin_agent_receive));
    env.functions.insert("agent_broadcast".into(), FnDef::Builtin(builtin_agent_broadcast));
    env.functions.insert("nas_create".into(), FnDef::Builtin(builtin_nas_create));
    env.functions.insert("nas_random_arch".into(), FnDef::Builtin(builtin_nas_random_arch));
    env.functions.insert("nas_evaluate".into(), FnDef::Builtin(builtin_nas_evaluate));
    env.functions.insert("nas_best".into(), FnDef::Builtin(builtin_nas_best));
    env.functions.insert("chain_of_thought".into(), FnDef::Builtin(builtin_chain_of_thought));
    env.functions.insert("tree_of_thought".into(), FnDef::Builtin(builtin_tree_of_thought));
    env.functions.insert("self_reflect".into(), FnDef::Builtin(builtin_self_reflect));
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: LazyLock<Mutex<()>> = LazyLock::new(|| Mutex::new(()));

    fn reset_state() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        AGENTS.lock().unwrap().clear();
        NAS_SPACES.lock().unwrap().clear();
        guard
    }

    fn assert_int(v: &Value, expected: i128) {
        match v { Value::Int(i) => assert_eq!(*i, expected), _ => panic!("expected Int({}), got {:?}", expected, v) }
    }
    fn assert_str(v: &Value, expected: &str) {
        match v { Value::String(s) => assert_eq!(s, expected), _ => panic!("expected String({}), got {:?}", expected, v) }
    }
    fn assert_float(v: &Value, expected: f64) {
        match v { Value::Float(f) => assert!((f - expected).abs() < 1e-9), _ => panic!("expected Float({}), got {:?}", expected, v) }
    }

    #[test]
    fn test_agent_create() {
        let _g = reset_state();
        let mut env = Env::new();
        let r = builtin_agent_create(&mut env, vec![Value::String("Alice".into()), Value::String("planner".into())]).unwrap();
        assert!(matches!(r, Value::Int(_)));
        let r2 = builtin_agent_create(&mut env, vec![Value::String("Bob".into()), Value::String("coder".into())]).unwrap();
        assert!(matches!(r2, Value::Int(_)));
        // Second agent should have a higher id
        if let (Value::Int(a), Value::Int(b)) = (&r, &r2) { assert!(b > a); }
    }

    #[test]
    fn test_agent_send_receive() {
        let _g = reset_state();
        let mut env = Env::new();
        builtin_agent_create(&mut env, vec![Value::String("A".into()), Value::String("r".into())]).unwrap();
        builtin_agent_create(&mut env, vec![Value::String("B".into()), Value::String("r".into())]).unwrap();
        builtin_agent_send(&mut env, vec![Value::Int(0), Value::Int(1), Value::String("hello".into())]).unwrap();
        let msg = builtin_agent_receive(&mut env, vec![Value::Int(1)]).unwrap();
        match msg {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Message");
                assert_str(&fields["content"], "hello");
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn test_agent_receive_empty() {
        let _g = reset_state();
        let mut env = Env::new();
        builtin_agent_create(&mut env, vec![Value::String("X".into()), Value::String("r".into())]).unwrap();
        let r = builtin_agent_receive(&mut env, vec![Value::Int(0)]).unwrap();
        assert_str(&r, "None");
    }

    #[test]
    fn test_agent_broadcast() {
        let _g = reset_state();
        let mut env = Env::new();
        for name in &["A", "B", "C"] {
            builtin_agent_create(&mut env, vec![Value::String(name.to_string()), Value::String("r".into())]).unwrap();
        }
        let count = builtin_agent_broadcast(&mut env, vec![Value::Int(0), Value::String("hi all".into())]).unwrap();
        assert_int(&count, 2);
        assert_str(&builtin_agent_receive(&mut env, vec![Value::Int(0)]).unwrap(), "None");
        assert!(matches!(builtin_agent_receive(&mut env, vec![Value::Int(1)]).unwrap(), Value::Struct { .. }));
        assert!(matches!(builtin_agent_receive(&mut env, vec![Value::Int(2)]).unwrap(), Value::Struct { .. }));
    }

    #[test]
    fn test_nas_create_and_random() {
        let _g = reset_state();
        let mut env = Env::new();
        let space = Value::Array(vec![
            Value::Array(vec![Value::String("conv3".into()), Value::String("conv5".into())]),
            Value::Array(vec![Value::String("relu".into()), Value::String("gelu".into()), Value::String("silu".into())]),
        ]);
        let id = builtin_nas_create(&mut env, vec![space]).unwrap();
        assert!(matches!(id, Value::Int(_)));
        let arch = builtin_nas_random_arch(&mut env, vec![id.clone()]).unwrap();
        match arch {
            Value::Array(v) => assert_eq!(v.len(), 2),
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_nas_evaluate_and_best() {
        let _g = reset_state();
        let mut env = Env::new();
        let space = Value::Array(vec![
            Value::Array(vec![Value::String("a".into()), Value::String("b".into())]),
        ]);
        builtin_nas_create(&mut env, vec![space]).unwrap();
        builtin_nas_evaluate(&mut env, vec![Value::Int(0), Value::Array(vec![Value::String("a".into())]), Value::Float(0.8)]).unwrap();
        builtin_nas_evaluate(&mut env, vec![Value::Int(0), Value::Array(vec![Value::String("b".into())]), Value::Float(0.95)]).unwrap();
        let best = builtin_nas_best(&mut env, vec![Value::Int(0)]).unwrap();
        match best {
            Value::Struct { fields, .. } => {
                assert_float(&fields["score"], 0.95);
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn test_chain_of_thought() {
        let _g = reset_state();
        let mut env = Env::new();
        let steps = Value::Array(vec![
            Value::String("observe input".into()),
            Value::String("identify pattern".into()),
            Value::String("conclude".into()),
        ]);
        let r = builtin_chain_of_thought(&mut env, vec![steps]).unwrap();
        match r {
            Value::Struct { name, fields } => {
                assert_eq!(name, "CoTTrace");
                assert_int(&fields["total_steps"], 3);
            }
            _ => panic!("expected struct"),
        }
    }

    #[test]
    fn test_self_reflect() {
        let _g = reset_state();
        let mut env = Env::new();
        let claim = Value::String("the model is accurate".into());
        let evidence = Value::Array(vec![
            Value::String("the model scores 95%".into()),
            Value::String("unrelated fact".into()),
        ]);
        let r = builtin_self_reflect(&mut env, vec![claim, evidence]).unwrap();
        match r {
            Value::Struct { fields, .. } => {
                assert_int(&fields["evidence_count"], 2);
                if let Value::Float(c) = fields["consistency"] { assert!(c > 0.0); }
            }
            _ => panic!("expected struct"),
        }
    }
}
