// Model merging (SLERP/TIES/DARE) + RAG vector store for Vortex
use crate::interpreter::{Value, Env, FnDef};
use std::sync::{LazyLock, Mutex};

// ── RAG Vector Store ──────────────────────────────────────────────

#[derive(Clone, Debug)]
struct Document {
    text: String,
    embedding: Vec<f64>,
}

#[derive(Clone, Debug, Default)]
struct VectorStore {
    docs: Vec<Option<Document>>, // Option for soft-delete
}

static STORES: LazyLock<Mutex<Vec<VectorStore>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ── Helper: extract Vec<f64> from Value ───────────────────────────

fn extract_floats(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected numeric array".into()),
        }).collect(),
        _ => Err("expected array of numbers".into()),
    }
}

fn floats_to_value(v: Vec<f64>) -> Value {
    Value::Array(v.into_iter().map(Value::Float).collect())
}

// ── Math helpers ──────────────────────────────────────────────────

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let d = norm(a) * norm(b);
    if d == 0.0 { 0.0 } else { dot(a, b) / d }
}

// ── SLERP ─────────────────────────────────────────────────────────

fn slerp(a: &[f64], b: &[f64], t: f64) -> Vec<f64> {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return a.iter().zip(b.iter()).map(|(x, y)| x * (1.0 - t) + y * t).collect();
    }
    let cos_omega = (dot(a, b) / (na * nb)).clamp(-1.0, 1.0);
    if cos_omega.abs() > 0.9995 {
        // Nearly parallel — fall back to lerp
        return a.iter().zip(b.iter()).map(|(x, y)| x * (1.0 - t) + y * t).collect();
    }
    let omega = cos_omega.acos();
    let sin_omega = omega.sin();
    let ca = ((1.0 - t) * omega).sin() / sin_omega;
    let cb = (t * omega).sin() / sin_omega;
    a.iter().zip(b.iter()).map(|(x, y)| x * ca + y * cb).collect()
}

fn builtin_merge_slerp(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("merge_slerp(weights_a, weights_b, t)".into()); }
    let a = extract_floats(&args[0])?;
    let b = extract_floats(&args[1])?;
    let t = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t must be numeric".into()) };
    if a.len() != b.len() { return Err("weight vectors must have same length".into()); }
    Ok(floats_to_value(slerp(&a, &b, t)))
}

// ── TIES merge ────────────────────────────────────────────────────

fn builtin_merge_ties(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("merge_ties(models_weights, threshold)".into()); }
    let models = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| extract_floats(v)).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("expected array of weight vectors".into()),
    };
    let threshold = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("threshold must be numeric".into()) };
    if models.is_empty() { return Err("need at least one model".into()); }
    let dim = models[0].len();
    let n = models.len() as f64;

    let mut result = vec![0.0; dim];
    for i in 0..dim {
        // Trim: zero out values below threshold
        let trimmed: Vec<f64> = models.iter().map(|m| {
            if m[i].abs() < threshold { 0.0 } else { m[i] }
        }).collect();
        // Elect sign: majority sign
        let pos: usize = trimmed.iter().filter(|&&x| x > 0.0).count();
        let neg: usize = trimmed.iter().filter(|&&x| x < 0.0).count();
        let sign: f64 = if pos >= neg { 1.0 } else { -1.0 };
        // Disjoint merge: average only agreeing values
        let mut sum = 0.0;
        let mut cnt = 0.0;
        for &v in &trimmed {
            if (v > 0.0 && sign > 0.0) || (v < 0.0 && sign < 0.0) {
                sum += v;
                cnt += 1.0;
            }
        }
        result[i] = if cnt > 0.0 { sum / n } else { 0.0 };
    }
    Ok(floats_to_value(result))
}

// ── DARE merge ────────────────────────────────────────────────────

fn builtin_merge_dare(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("merge_dare(models_weights, drop_rate)".into()); }
    let models = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| extract_floats(v)).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("expected array of weight vectors".into()),
    };
    let drop_rate = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("drop_rate must be numeric".into()) };
    if models.is_empty() { return Err("need at least one model".into()); }
    let dim = models[0].len();
    let n = models.len();
    let rescale = if drop_rate < 1.0 { 1.0 / (1.0 - drop_rate) } else { 1.0 };

    // Deterministic drop using simple hash-like pattern for reproducibility
    let mut result = vec![0.0; dim];
    for i in 0..dim {
        let mut sum = 0.0;
        for (mi, m) in models.iter().enumerate() {
            let hash = ((i * 2654435761 + mi * 40503) & 0xFFFF) as f64 / 65535.0;
            if hash >= drop_rate {
                sum += m[i] * rescale;
            }
        }
        result[i] = sum / n as f64;
    }
    Ok(floats_to_value(result))
}

// ── Linear merge ──────────────────────────────────────────────────

fn builtin_merge_linear(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("merge_linear(models_weights, coefficients)".into()); }
    let models = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| extract_floats(v)).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("expected array of weight vectors".into()),
    };
    let coeffs = extract_floats(&args[1])?;
    if models.len() != coeffs.len() { return Err("models and coefficients must have same length".into()); }
    if models.is_empty() { return Err("need at least one model".into()); }
    let dim = models[0].len();

    let mut result = vec![0.0; dim];
    for (m, &c) in models.iter().zip(coeffs.iter()) {
        if m.len() != dim { return Err("all weight vectors must have same length".into()); }
        for i in 0..dim {
            result[i] += m[i] * c;
        }
    }
    Ok(floats_to_value(result))
}

// ── RAG builtins ──────────────────────────────────────────────────

fn builtin_rag_create(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut stores = STORES.lock().map_err(|e| e.to_string())?;
    let id = stores.len();
    stores.push(VectorStore::default());
    Ok(Value::Int(id as i128))
}

fn builtin_rag_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("rag_add(store_id, text, embedding)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("store_id must be int".into()) };
    let text = match &args[1] { Value::String(s) => s.clone(), _ => return Err("text must be string".into()) };
    let embedding = extract_floats(&args[2])?;
    let mut stores = STORES.lock().map_err(|e| e.to_string())?;
    let store = stores.get_mut(id).ok_or("invalid store_id")?;
    let doc_idx = store.docs.len();
    store.docs.push(Some(Document { text, embedding }));
    Ok(Value::Int(doc_idx as i128))
}

fn builtin_rag_query(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("rag_query(store_id, query_embedding, top_k)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("store_id must be int".into()) };
    let query = extract_floats(&args[1])?;
    let top_k = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("top_k must be int".into()) };
    let stores = STORES.lock().map_err(|e| e.to_string())?;
    let store = stores.get(id).ok_or("invalid store_id")?;

    let mut scored: Vec<(f64, &Document)> = store.docs.iter()
        .filter_map(|d| d.as_ref())
        .map(|d| (cosine_similarity(&query, &d.embedding), d))
        .collect();
    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(top_k);

    let results: Vec<Value> = scored.into_iter().map(|(score, doc)| {
        Value::Array(vec![Value::String(doc.text.clone()), Value::Float(score)])
    }).collect();
    Ok(Value::Array(results))
}

fn builtin_rag_remove(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("rag_remove(store_id, doc_index)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("store_id must be int".into()) };
    let idx = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("doc_index must be int".into()) };
    let mut stores = STORES.lock().map_err(|e| e.to_string())?;
    let store = stores.get_mut(id).ok_or("invalid store_id")?;
    if idx >= store.docs.len() || store.docs[idx].is_none() {
        return Err("document not found".into());
    }
    store.docs[idx] = None;
    Ok(Value::Bool(true))
}

fn builtin_rag_count(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("rag_count(store_id)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("store_id must be int".into()) };
    let stores = STORES.lock().map_err(|e| e.to_string())?;
    let store = stores.get(id).ok_or("invalid store_id")?;
    let count = store.docs.iter().filter(|d| d.is_some()).count();
    Ok(Value::Int(count as i128))
}

// ── Registration ──────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("merge_slerp".into(), FnDef::Builtin(builtin_merge_slerp));
    env.functions.insert("merge_ties".into(), FnDef::Builtin(builtin_merge_ties));
    env.functions.insert("merge_dare".into(), FnDef::Builtin(builtin_merge_dare));
    env.functions.insert("merge_linear".into(), FnDef::Builtin(builtin_merge_linear));
    env.functions.insert("rag_create".into(), FnDef::Builtin(builtin_rag_create));
    env.functions.insert("rag_add".into(), FnDef::Builtin(builtin_rag_add));
    env.functions.insert("rag_query".into(), FnDef::Builtin(builtin_rag_query));
    env.functions.insert("rag_remove".into(), FnDef::Builtin(builtin_rag_remove));
    env.functions.insert("rag_count".into(), FnDef::Builtin(builtin_rag_count));
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env { Env::new() }

    #[test]
    fn test_slerp_midpoint() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let r = slerp(&a, &b, 0.5);
        let n = norm(&r);
        assert!((n - 1.0).abs() < 1e-10, "slerp should preserve unit norm");
        assert!((r[0] - r[1]).abs() < 1e-10, "midpoint should be symmetric");
    }

    #[test]
    fn test_slerp_endpoints() {
        let a = vec![3.0, 4.0];
        let b = vec![1.0, 2.0];
        let r0 = slerp(&a, &b, 0.0);
        let r1 = slerp(&a, &b, 1.0);
        for i in 0..2 { assert!((r0[i] - a[i]).abs() < 1e-10); }
        for i in 0..2 { assert!((r1[i] - b[i]).abs() < 1e-10); }
    }

    #[test]
    fn test_merge_linear() {
        let mut e = env();
        let result = builtin_merge_linear(&mut e, vec![
            Value::Array(vec![
                Value::Array(vec![Value::Float(1.0), Value::Float(2.0)]),
                Value::Array(vec![Value::Float(3.0), Value::Float(4.0)]),
            ]),
            Value::Array(vec![Value::Float(0.5), Value::Float(0.5)]),
        ]).unwrap();
        let floats = extract_floats(&result).unwrap();
        assert!((floats[0] - 2.0).abs() < 1e-10);
        assert!((floats[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ties_zeros_below_threshold() {
        let mut e = env();
        let result = builtin_merge_ties(&mut e, vec![
            Value::Array(vec![
                Value::Array(vec![Value::Float(0.01), Value::Float(5.0)]),
                Value::Array(vec![Value::Float(-0.01), Value::Float(3.0)]),
            ]),
            Value::Float(0.1),
        ]).unwrap();
        let floats = extract_floats(&result).unwrap();
        assert!((floats[0]).abs() < 1e-10, "below-threshold values should be zeroed");
        assert!(floats[1] > 0.0, "above-threshold values should survive");
    }

    #[test]
    fn test_dare_output_length() {
        let mut e = env();
        let result = builtin_merge_dare(&mut e, vec![
            Value::Array(vec![
                Value::Array(vec![Value::Float(1.0), Value::Float(2.0), Value::Float(3.0)]),
                Value::Array(vec![Value::Float(4.0), Value::Float(5.0), Value::Float(6.0)]),
            ]),
            Value::Float(0.3),
        ]).unwrap();
        let floats = extract_floats(&result).unwrap();
        assert_eq!(floats.len(), 3);
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rag_create_and_count() {
        let mut e = env();
        let id = builtin_rag_create(&mut e, vec![]).unwrap();
        let count = builtin_rag_count(&mut e, vec![id.clone()]).unwrap();
        match count { Value::Int(0) => {}, other => panic!("expected Int(0), got {:?}", other) }
    }

    #[test]
    fn test_rag_add_query_remove() {
        let mut e = env();
        let id = builtin_rag_create(&mut e, vec![]).unwrap();
        // Add two docs
        builtin_rag_add(&mut e, vec![
            id.clone(), Value::String("hello world".into()),
            Value::Array(vec![Value::Float(1.0), Value::Float(0.0)]),
        ]).unwrap();
        builtin_rag_add(&mut e, vec![
            id.clone(), Value::String("goodbye".into()),
            Value::Array(vec![Value::Float(0.0), Value::Float(1.0)]),
        ]).unwrap();
        // Query close to first doc
        let results = builtin_rag_query(&mut e, vec![
            id.clone(),
            Value::Array(vec![Value::Float(0.9), Value::Float(0.1)]),
            Value::Int(1),
        ]).unwrap();
        if let Value::Array(arr) = &results {
            assert_eq!(arr.len(), 1);
            if let Value::Array(pair) = &arr[0] {
                match &pair[0] { Value::String(s) => assert_eq!(s, "hello world"), other => panic!("expected string, got {:?}", other) }
            }
        }
        // Remove and check count
        builtin_rag_remove(&mut e, vec![id.clone(), Value::Int(0)]).unwrap();
        let count = builtin_rag_count(&mut e, vec![id.clone()]).unwrap();
        match count { Value::Int(1) => {}, other => panic!("expected Int(1), got {:?}", other) }
    }
}
