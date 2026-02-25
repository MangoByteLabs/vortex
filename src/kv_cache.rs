//! KV Cache with paged attention and continuous batching for LLM inference.

use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};
use crate::interpreter::{Env, Value, FnDef};

#[derive(Debug, Clone)]
pub struct KVBlock {
    pub keys: Vec<Vec<f64>>,
    pub values: Vec<Vec<f64>>,
    pub seq_len: usize,
    pub capacity: usize,
}

impl KVBlock {
    pub fn new(capacity: usize, head_dim: usize) -> Self {
        Self { keys: Vec::with_capacity(capacity), values: Vec::with_capacity(capacity), seq_len: 0, capacity }
    }
    pub fn append(&mut self, key: Vec<f64>, value: Vec<f64>) -> bool {
        if self.seq_len >= self.capacity { return false; }
        self.keys.push(key);
        self.values.push(value);
        self.seq_len += 1;
        true
    }
}

#[derive(Debug)]
pub struct PagedKVCache {
    blocks: Vec<KVBlock>,
    block_size: usize,
    num_heads: usize,
    head_dim: usize,
    num_layers: usize,
    block_tables: HashMap<(usize, usize), Vec<usize>>, // (layer, seq_id) -> block indices
    free_blocks: Vec<usize>,
    prefix_cache: HashMap<u64, Vec<usize>>,
}

impl PagedKVCache {
    pub fn new(num_layers: usize, num_heads: usize, head_dim: usize, block_size: usize, max_blocks: usize) -> Self {
        let blocks: Vec<KVBlock> = (0..max_blocks).map(|_| KVBlock::new(block_size, head_dim)).collect();
        let free_blocks: Vec<usize> = (0..max_blocks).rev().collect();
        Self { blocks, block_size, num_heads, head_dim, num_layers, block_tables: HashMap::new(), free_blocks, prefix_cache: HashMap::new() }
    }

    pub fn allocate_block(&mut self) -> Option<usize> { self.free_blocks.pop() }

    pub fn free_block(&mut self, id: usize) {
        self.blocks[id] = KVBlock::new(self.block_size, self.head_dim);
        self.free_blocks.push(id);
    }

    pub fn append_kv(&mut self, layer: usize, seq_id: usize, key: Vec<f64>, value: Vec<f64>) -> Result<(), String> {
        let needs_block = {
            let table = self.block_tables.entry((layer, seq_id)).or_insert_with(Vec::new);
            table.is_empty() || self.blocks[*table.last().unwrap()].seq_len >= self.block_size
        };
        if needs_block {
            let bid = self.allocate_block().ok_or("KV cache out of blocks")?;
            self.block_tables.entry((layer, seq_id)).or_insert_with(Vec::new).push(bid);
        }
        let table = self.block_tables.get(&(layer, seq_id)).unwrap();
        let last_block = *table.last().unwrap();
        self.blocks[last_block].append(key, value);
        Ok(())
    }

    pub fn get_kv(&self, layer: usize, seq_id: usize) -> (Vec<Vec<f64>>, Vec<Vec<f64>>) {
        let table = match self.block_tables.get(&(layer, seq_id)) {
            Some(t) => t, None => return (vec![], vec![]),
        };
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for &bid in table {
            keys.extend(self.blocks[bid].keys.iter().cloned());
            values.extend(self.blocks[bid].values.iter().cloned());
        }
        (keys, values)
    }

    pub fn free_seq(&mut self, seq_id: usize) {
        let to_free: Vec<(usize, usize)> = self.block_tables.keys().filter(|(_, s)| *s == seq_id).cloned().collect();
        for key in to_free {
            if let Some(blocks) = self.block_tables.remove(&key) {
                for bid in blocks { self.free_block(bid); }
            }
        }
    }

    pub fn used_blocks(&self) -> usize { self.blocks.len() - self.free_blocks.len() }
    pub fn free_block_count(&self) -> usize { self.free_blocks.len() }
}

pub fn paged_attention(query: &[f64], keys: &[Vec<f64>], values: &[Vec<f64>], head_dim: usize) -> Vec<f64> {
    if keys.is_empty() { return vec![0.0; head_dim]; }
    let scale = 1.0 / (head_dim as f64).sqrt();
    let scores: Vec<f64> = keys.iter().map(|k| {
        k.iter().zip(query.iter()).map(|(ki, qi)| ki * qi).sum::<f64>() * scale
    }).collect();
    // softmax
    let max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exp: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
    let sum: f64 = exp.iter().sum();
    let weights: Vec<f64> = exp.iter().map(|e| e / sum).collect();
    // weighted sum of values
    let mut out = vec![0.0; head_dim];
    for (w, v) in weights.iter().zip(values.iter()) {
        for (o, vi) in out.iter_mut().zip(v.iter()) { *o += w * vi; }
    }
    out
}

// ---- Continuous Batching ----

#[derive(Debug, Clone)]
pub struct BatchSlot {
    pub seq_id: usize,
    pub tokens: Vec<u64>,
    pub position: usize,
    pub is_prefill: bool,
    pub is_done: bool,
}

#[derive(Debug)]
pub struct ContinuousBatcher {
    pub slots: Vec<Option<BatchSlot>>,
    pub max_batch_size: usize,
    pub next_seq_id: usize,
}

impl ContinuousBatcher {
    pub fn new(max_batch_size: usize) -> Self {
        Self { slots: vec![None; max_batch_size], max_batch_size, next_seq_id: 0 }
    }

    pub fn add_request(&mut self, tokens: Vec<u64>) -> Option<usize> {
        for (i, slot) in self.slots.iter_mut().enumerate() {
            if slot.is_none() {
                let seq_id = self.next_seq_id;
                self.next_seq_id += 1;
                *slot = Some(BatchSlot { seq_id, tokens, position: 0, is_prefill: true, is_done: false });
                return Some(i);
            }
        }
        None
    }

    pub fn remove_request(&mut self, slot_id: usize) {
        if slot_id < self.slots.len() { self.slots[slot_id] = None; }
    }

    pub fn num_active(&self) -> usize { self.slots.iter().filter(|s| s.is_some()).count() }
}

// ---- Speculative decoding verification ----

pub fn spec_verify(draft_tokens: &[u32], draft_probs: &[Vec<f64>], target_probs: &[Vec<f64>]) -> Vec<u32> {
    let mut accepted = Vec::new();
    for (i, &token) in draft_tokens.iter().enumerate() {
        if i >= draft_probs.len() || i >= target_probs.len() { break; }
        let t = token as usize;
        let p_draft = draft_probs[i].get(t).copied().unwrap_or(0.001);
        let p_target = target_probs[i].get(t).copied().unwrap_or(0.0);
        if p_target >= p_draft {
            accepted.push(token);
        } else {
            let accept_prob = p_target / p_draft;
            // Deterministic for reproducibility: accept if ratio > 0.5
            if accept_prob > 0.5 { accepted.push(token); } else { break; }
        }
    }
    accepted
}

// ---- Storage ----

static CACHES: LazyLock<Mutex<Vec<PagedKVCache>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static BATCHERS: LazyLock<Mutex<Vec<ContinuousBatcher>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ---- Builtins ----

fn builtin_kv_cache_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let nl = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 12 };
    let nh = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 8 };
    let hd = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => 64 };
    let bs = match args.get(3) { Some(Value::Int(n)) => *n as usize, _ => 16 };
    let mb = match args.get(4) { Some(Value::Int(n)) => *n as usize, _ => 256 };
    let mut cs = CACHES.lock().unwrap();
    let id = cs.len();
    cs.push(PagedKVCache::new(nl, nh, hd, bs, mb));
    Ok(Value::Int(id as i128))
}

fn builtin_kv_cache_append(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let cid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need cache_id".into()) };
    let layer = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need layer".into()) };
    let seq_id = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need seq_id".into()) };
    let key = extract_f64(args.get(3).ok_or("need key")?)?;
    let value = extract_f64(args.get(4).ok_or("need value")?)?;
    let mut cs = CACHES.lock().unwrap();
    let c = cs.get_mut(cid).ok_or("invalid cache")?;
    c.append_kv(layer, seq_id, key, value)?;
    Ok(Value::Bool(true))
}

fn builtin_kv_cache_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let cid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need cache_id".into()) };
    let query = extract_f64(args.get(1).ok_or("need query")?)?;
    let layer = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need layer".into()) };
    let seq_id = match args.get(3) { Some(Value::Int(n)) => *n as usize, _ => return Err("need seq_id".into()) };
    let cs = CACHES.lock().unwrap();
    let c = cs.get(cid).ok_or("invalid cache")?;
    let (keys, values) = c.get_kv(layer, seq_id);
    let head_dim = c.head_dim;
    let out = paged_attention(&query, &keys, &values, head_dim);
    Ok(Value::Array(out.iter().map(|f| Value::Float(*f)).collect()))
}

fn builtin_kv_cache_stats(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let cid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need cache_id".into()) };
    let cs = CACHES.lock().unwrap();
    let c = cs.get(cid).ok_or("invalid cache")?;
    let mut fields = HashMap::new();
    fields.insert("used_blocks".to_string(), Value::Int(c.used_blocks() as i128));
    fields.insert("free_blocks".to_string(), Value::Int(c.free_block_count() as i128));
    fields.insert("block_size".to_string(), Value::Int(c.block_size as i128));
    fields.insert("num_seqs".to_string(), Value::Int(c.block_tables.len() as i128));
    Ok(Value::Struct { name: "KVCacheStats".to_string(), fields })
}

fn builtin_kv_cache_free_seq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let cid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need cache_id".into()) };
    let seq_id = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need seq_id".into()) };
    let mut cs = CACHES.lock().unwrap();
    let c = cs.get_mut(cid).ok_or("invalid cache")?;
    c.free_seq(seq_id);
    Ok(Value::Bool(true))
}

fn builtin_batcher_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let max_batch = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 32 };
    let mut bs = BATCHERS.lock().unwrap();
    let id = bs.len();
    bs.push(ContinuousBatcher::new(max_batch));
    Ok(Value::Int(id as i128))
}

fn builtin_batcher_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let bid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need batcher_id".into()) };
    let tokens = match args.get(1) { Some(Value::Array(arr)) => arr.iter().map(|v| match v { Value::Int(n) => *n as u64, _ => 0 }).collect(), _ => vec![] };
    let mut bs = BATCHERS.lock().unwrap();
    let b = bs.get_mut(bid).ok_or("invalid batcher")?;
    match b.add_request(tokens) {
        Some(slot) => Ok(Value::Int(slot as i128)),
        None => Err("batcher full".into()),
    }
}

fn builtin_batcher_active(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let bid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need batcher_id".into()) };
    let bs = BATCHERS.lock().unwrap();
    let b = bs.get(bid).ok_or("invalid batcher")?;
    Ok(Value::Int(b.num_active() as i128))
}

fn builtin_batcher_remove(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let bid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need batcher_id".into()) };
    let slot = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need slot_id".into()) };
    let mut bs = BATCHERS.lock().unwrap();
    let b = bs.get_mut(bid).ok_or("invalid batcher")?;
    b.remove_request(slot);
    Ok(Value::Bool(true))
}

fn builtin_spec_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let draft_tokens: Vec<u32> = match args.get(0) { Some(Value::Array(a)) => a.iter().map(|v| match v { Value::Int(n) => *n as u32, _ => 0 }).collect(), _ => vec![] };
    let draft_probs = extract_prob_matrix(args.get(1))?;
    let target_probs = extract_prob_matrix(args.get(2))?;
    let accepted = spec_verify(&draft_tokens, &draft_probs, &target_probs);
    Ok(Value::Array(accepted.iter().map(|t| Value::Int(*t as i128)).collect()))
}

fn builtin_kv_cache_memory(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let cid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need cache_id".into()) };
    let cs = CACHES.lock().unwrap();
    let c = cs.get(cid).ok_or("invalid cache")?;
    let bytes = c.used_blocks() * c.block_size * c.head_dim * 2 * 8; // keys + values, f64
    Ok(Value::Int(bytes as i128))
}

fn extract_f64(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x { Value::Float(f) => Ok(*f), Value::Int(n) => Ok(*n as f64), _ => Err("expected number".into()) }).collect(),
        _ => Err("expected array".into()),
    }
}

fn extract_prob_matrix(v: Option<&Value>) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Some(Value::Array(rows)) => rows.iter().map(|r| extract_f64(r)).collect(),
        _ => Ok(vec![]),
    }
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("kv_cache_new".to_string(), FnDef::Builtin(builtin_kv_cache_new));
    env.functions.insert("kv_cache_append".to_string(), FnDef::Builtin(builtin_kv_cache_append));
    env.functions.insert("kv_cache_attention".to_string(), FnDef::Builtin(builtin_kv_cache_attention));
    env.functions.insert("kv_cache_stats".to_string(), FnDef::Builtin(builtin_kv_cache_stats));
    env.functions.insert("kv_cache_free_seq".to_string(), FnDef::Builtin(builtin_kv_cache_free_seq));
    env.functions.insert("kv_cache_memory".to_string(), FnDef::Builtin(builtin_kv_cache_memory));
    env.functions.insert("batcher_new".to_string(), FnDef::Builtin(builtin_batcher_new));
    env.functions.insert("batcher_add".to_string(), FnDef::Builtin(builtin_batcher_add));
    env.functions.insert("batcher_active".to_string(), FnDef::Builtin(builtin_batcher_active));
    env.functions.insert("batcher_remove".to_string(), FnDef::Builtin(builtin_batcher_remove));
    env.functions.insert("spec_verify".to_string(), FnDef::Builtin(builtin_spec_verify));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kv_block_append() {
        let mut b = KVBlock::new(4, 2);
        assert!(b.append(vec![1.0, 2.0], vec![3.0, 4.0]));
        assert_eq!(b.seq_len, 1);
    }

    #[test]
    fn test_kv_block_full() {
        let mut b = KVBlock::new(2, 1);
        assert!(b.append(vec![1.0], vec![1.0]));
        assert!(b.append(vec![2.0], vec![2.0]));
        assert!(!b.append(vec![3.0], vec![3.0]));
    }

    #[test]
    fn test_paged_cache_basic() {
        let mut c = PagedKVCache::new(1, 1, 2, 4, 10);
        c.append_kv(0, 0, vec![1.0, 0.0], vec![1.0, 2.0]).unwrap();
        c.append_kv(0, 0, vec![0.0, 1.0], vec![3.0, 4.0]).unwrap();
        let (keys, vals) = c.get_kv(0, 0);
        assert_eq!(keys.len(), 2);
        assert_eq!(vals.len(), 2);
    }

    #[test]
    fn test_paged_attention() {
        let query = vec![1.0, 0.0];
        let keys = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let values = vec![vec![10.0, 20.0], vec![30.0, 40.0]];
        let out = paged_attention(&query, &keys, &values, 2);
        assert_eq!(out.len(), 2);
        assert!(out[0] > 10.0); // should weight first value more
    }

    #[test]
    fn test_free_seq() {
        let mut c = PagedKVCache::new(1, 1, 2, 4, 10);
        c.append_kv(0, 0, vec![1.0, 0.0], vec![1.0, 2.0]).unwrap();
        let used_before = c.used_blocks();
        c.free_seq(0);
        assert!(c.free_block_count() > 10 - used_before);
    }

    #[test]
    fn test_batcher() {
        let mut b = ContinuousBatcher::new(4);
        assert_eq!(b.num_active(), 0);
        b.add_request(vec![1, 2, 3]);
        assert_eq!(b.num_active(), 1);
        b.add_request(vec![4, 5]);
        assert_eq!(b.num_active(), 2);
        b.remove_request(0);
        assert_eq!(b.num_active(), 1);
    }

    #[test]
    fn test_spec_verify_all_accept() {
        let tokens = vec![1, 2, 3];
        let draft = vec![vec![0.0, 0.9, 0.1], vec![0.0, 0.1, 0.9], vec![0.0, 0.1, 0.9]];
        let target = vec![vec![0.0, 0.95, 0.05], vec![0.0, 0.05, 0.95], vec![0.0, 0.05, 0.95]];
        let accepted = spec_verify(&tokens, &draft, &target);
        assert!(accepted.len() >= 2, "should accept most tokens, got {}", accepted.len());
    }

    #[test]
    fn test_spec_verify_reject() {
        let tokens = vec![1];
        let draft = vec![vec![0.0, 0.9, 0.1]];
        let target = vec![vec![0.0, 0.1, 0.9]]; // target disagrees
        let accepted = spec_verify(&tokens, &draft, &target);
        assert!(accepted.len() < 1); // should reject
    }

    #[test]
    fn test_cache_stats() {
        let mut c = PagedKVCache::new(1, 1, 2, 4, 10);
        assert_eq!(c.used_blocks(), 0);
        c.append_kv(0, 0, vec![1.0, 0.0], vec![1.0, 2.0]).unwrap();
        assert_eq!(c.used_blocks(), 1);
    }

    #[test]
    fn test_multi_layer_cache() {
        let mut c = PagedKVCache::new(2, 1, 2, 4, 20);
        c.append_kv(0, 0, vec![1.0, 0.0], vec![1.0, 2.0]).unwrap();
        c.append_kv(1, 0, vec![0.0, 1.0], vec![3.0, 4.0]).unwrap();
        let (k0, _) = c.get_kv(0, 0);
        let (k1, _) = c.get_kv(1, 0);
        assert_eq!(k0[0], vec![1.0, 0.0]);
        assert_eq!(k1[0], vec![0.0, 1.0]);
    }
}
