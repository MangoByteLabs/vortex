//! BPE (Byte Pair Encoding) tokenizer for the Vortex language runtime.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use crate::interpreter::{Value, Env, FnDef};

static TOKENIZERS: LazyLock<Mutex<Vec<BpeTokenizer>>> = LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub vocab: HashMap<String, u32>,
    pub inverse_vocab: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        Self { vocab: HashMap::new(), inverse_vocab: HashMap::new(), merges: Vec::new() }
    }

    /// Train BPE from a text corpus up to the given vocab_size.
    pub fn train(text: &str, vocab_size: usize) -> Self {
        let mut tok = Self::new();
        // Seed vocab with individual bytes
        for b in 0u8..=255 {
            let s = String::from(b as char);
            let id = b as u32;
            tok.vocab.insert(s.clone(), id);
            tok.inverse_vocab.insert(id, s);
        }
        // Split text into words (whitespace-delimited), represent each as Vec<String> of chars
        let mut words: Vec<(Vec<String>, usize)> = Vec::new();
        let mut freq: HashMap<Vec<String>, usize> = HashMap::new();
        for word in text.split_whitespace() {
            let syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            *freq.entry(syms).or_insert(0) += 1;
        }
        for (syms, count) in &freq {
            words.push((syms.clone(), *count));
        }

        let mut next_id = 256u32;
        while next_id < vocab_size as u32 {
            // Count pairs
            let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();
            for (syms, count) in &words {
                for i in 0..syms.len().saturating_sub(1) {
                    let pair = (syms[i].clone(), syms[i + 1].clone());
                    *pair_counts.entry(pair).or_insert(0) += count;
                }
            }
            if pair_counts.is_empty() {
                break;
            }
            let best = pair_counts.into_iter().max_by_key(|&(_, c)| c).unwrap();
            let (ref a, ref b) = best.0;
            let merged = format!("{}{}", a, b);
            tok.merges.push((a.clone(), b.clone()));
            tok.vocab.insert(merged.clone(), next_id);
            tok.inverse_vocab.insert(next_id, merged.clone());
            // Apply merge to all words
            for (syms, _) in &mut words {
                let mut i = 0;
                while i + 1 < syms.len() {
                    if syms[i] == *a && syms[i + 1] == *b {
                        syms[i] = merged.clone();
                        syms.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
            next_id += 1;
        }
        // Ensure all remaining symbols in words are in vocab
        for (syms, _) in &words {
            for s in syms {
                if !tok.vocab.contains_key(s) {
                    tok.vocab.insert(s.clone(), next_id);
                    tok.inverse_vocab.insert(next_id, s.clone());
                    next_id += 1;
                }
            }
        }
        tok
    }

    /// Apply learned merges to tokenize a string, returning token ids.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut result = Vec::new();
        for word in text.split_whitespace() {
            let mut syms: Vec<String> = word.chars().map(|c| c.to_string()).collect();
            for (a, b) in &self.merges {
                let merged = format!("{}{}", a, b);
                let mut i = 0;
                while i + 1 < syms.len() {
                    if syms[i] == *a && syms[i + 1] == *b {
                        syms[i] = merged.clone();
                        syms.remove(i + 1);
                    } else {
                        i += 1;
                    }
                }
            }
            for s in &syms {
                if let Some(&id) = self.vocab.get(s) {
                    result.push(id);
                }
            }
        }
        result
    }

    /// Decode token ids back to a string.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut parts = Vec::new();
        for id in ids {
            if let Some(s) = self.inverse_vocab.get(id) {
                parts.push(s.as_str());
            }
        }
        parts.join("")
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn to_json(&self) -> String {
        let merges_arr: Vec<String> = self.merges.iter()
            .map(|(a, b)| format!("[\"{}\",\"{}\"]", escape_json(a), escape_json(b)))
            .collect();
        let vocab_arr: Vec<String> = self.vocab.iter()
            .map(|(k, v)| format!("\"{}\":{}", escape_json(k), v))
            .collect();
        format!("{{\"vocab\":{{{}}},\"merges\":[{}]}}", vocab_arr.join(","), merges_arr.join(","))
    }

    pub fn from_json(json: &str) -> Result<Self, String> {
        // Minimal JSON parser for our known format
        let json = json.trim();
        let vocab_start = json.find("\"vocab\":{").ok_or("missing vocab")? + 9;
        let mut depth = 1;
        let mut vocab_end = vocab_start;
        for (i, c) in json[vocab_start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => { depth -= 1; if depth == 0 { vocab_end = vocab_start + i; break; } }
                _ => {}
            }
        }
        let vocab_str = &json[vocab_start..vocab_end];
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();
        // Parse "key":value pairs (handling escaped quotes)
        fn find_unescaped_quote(s: &str) -> Option<usize> {
            let mut i = 0;
            let bytes = s.as_bytes();
            while i < bytes.len() {
                if bytes[i] == b'\\' { i += 2; continue; }
                if bytes[i] == b'"' { return Some(i); }
                i += 1;
            }
            None
        }
        fn unescape(s: &str) -> String {
            let mut out = String::new();
            let mut chars = s.chars();
            while let Some(c) = chars.next() {
                if c == '\\' {
                    match chars.next() {
                        Some('"') => out.push('"'),
                        Some('\\') => out.push('\\'),
                        Some('n') => out.push('\n'),
                        Some('r') => out.push('\r'),
                        Some('t') => out.push('\t'),
                        Some('u') => {
                            let hex: String = chars.by_ref().take(4).collect();
                            if let Ok(code) = u32::from_str_radix(&hex, 16) {
                                if let Some(ch) = char::from_u32(code) { out.push(ch); }
                            }
                        }
                        Some(x) => { out.push('\\'); out.push(x); }
                        None => out.push('\\'),
                    }
                } else {
                    out.push(c);
                }
            }
            out
        }
        let mut rest = vocab_str;
        while let Some(qs) = rest.find('"') {
            rest = &rest[qs + 1..];
            let qe = find_unescaped_quote(rest).ok_or("bad vocab key")?;
            let key = unescape(&rest[..qe]);
            rest = &rest[qe + 1..];
            let cs = rest.find(':').ok_or("bad vocab sep")?;
            rest = &rest[cs + 1..];
            let ve = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            let val: u32 = rest[..ve].trim().parse().map_err(|e| format!("{}", e))?;
            vocab.insert(key.clone(), val);
            inverse_vocab.insert(val, key);
            rest = &rest[ve..];
        }
        // Parse merges
        let merges_start = json.find("\"merges\":[").ok_or("missing merges")? + 10;
        let merges_end = json[merges_start..].rfind(']').map(|i| merges_start + i).ok_or("bad merges")?;
        let merges_str = &json[merges_start..merges_end];
        let mut merges = Vec::new();
        let mut mrest = merges_str;
        while let Some(bs) = mrest.find('[') {
            mrest = &mrest[bs + 1..];
            let be = mrest.find(']').ok_or("bad merge")?;
            let inner = &mrest[..be];
            let parts: Vec<&str> = inner.split(',').collect();
            if parts.len() == 2 {
                let a = parts[0].trim().trim_matches('"').to_string();
                let b = parts[1].trim().trim_matches('"').to_string();
                merges.push((a, b));
            }
            mrest = &mrest[be + 1..];
        }
        Ok(Self { vocab, inverse_vocab, merges })
    }
}

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => { out.push_str(&format!("\\u{:04x}", c as u32)); }
            _ => out.push(c),
        }
    }
    out
}

fn get_tokenizer(id: usize) -> Result<BpeTokenizer, String> {
    let store = TOKENIZERS.lock().map_err(|e| format!("lock: {}", e))?;
    store.get(id).cloned().ok_or_else(|| format!("invalid tokenizer id: {}", id))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("bpe_train".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_train(text, vocab_size)".into()); }
        let text = match &args[0] { Value::String(s) => s.clone(), _ => return Err("text must be string".into()) };
        let vs = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("vocab_size must be int".into()) };
        let tok = BpeTokenizer::train(&text, vs);
        let mut store = TOKENIZERS.lock().map_err(|e| format!("{}", e))?;
        let id = store.len();
        store.push(tok);
        Ok(Value::Int(id as i128))
    }));

    env.functions.insert("bpe_encode".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_encode(id, text)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let text = match &args[1] { Value::String(s) => s.clone(), _ => return Err("text must be string".into()) };
        let tok = get_tokenizer(id)?;
        let ids = tok.encode(&text);
        Ok(Value::Array(ids.into_iter().map(|i| Value::Int(i as i128)).collect()))
    }));

    env.functions.insert("bpe_decode".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_decode(id, token_ids)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let ids: Vec<u32> = match &args[1] {
            Value::Array(arr) => arr.iter().map(|v| match v {
                Value::Int(n) => Ok(*n as u32),
                _ => Err("token ids must be ints".to_string()),
            }).collect::<Result<Vec<_>, _>>()?,
            _ => return Err("token_ids must be array".into()),
        };
        let tok = get_tokenizer(id)?;
        Ok(Value::String(tok.decode(&ids)))
    }));

    env.functions.insert("bpe_vocab_size".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 1 { return Err("bpe_vocab_size(id)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let tok = get_tokenizer(id)?;
        Ok(Value::Int(tok.vocab_size() as i128))
    }));

    env.functions.insert("bpe_save".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_save(id, path)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
        let tok = get_tokenizer(id)?;
        std::fs::write(&path, tok.to_json()).map_err(|e| format!("write: {}", e))?;
        Ok(Value::Bool(true))
    }));

    env.functions.insert("bpe_load".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 1 { return Err("bpe_load(path)".into()); }
        let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
        let data = std::fs::read_to_string(&path).map_err(|e| format!("read: {}", e))?;
        let tok = BpeTokenizer::from_json(&data)?;
        let mut store = TOKENIZERS.lock().map_err(|e| format!("{}", e))?;
        let id = store.len();
        store.push(tok);
        Ok(Value::Int(id as i128))
    }));

    env.functions.insert("bpe_tokenize_batch".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_tokenize_batch(id, texts)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let texts: Vec<String> = match &args[1] {
            Value::Array(arr) => arr.iter().map(|v| match v {
                Value::String(s) => Ok(s.clone()),
                _ => Err("texts must be strings".to_string()),
            }).collect::<Result<Vec<_>, _>>()?,
            _ => return Err("texts must be array".into()),
        };
        let tok = get_tokenizer(id)?;
        let result: Vec<Value> = texts.iter().map(|t| {
            let ids = tok.encode(t);
            Value::Array(ids.into_iter().map(|i| Value::Int(i as i128)).collect())
        }).collect();
        Ok(Value::Array(result))
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_text() -> &'static str {
        "the cat sat on the mat the cat ate the rat the bat sat on the hat"
    }

    #[test]
    fn test_train_creates_vocab() {
        let tok = BpeTokenizer::train(sample_text(), 270);
        assert!(tok.vocab_size() >= 256);
        assert!(!tok.merges.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let tok = BpeTokenizer::train(sample_text(), 280);
        // Roundtrip on a word from training corpus (no spaces — encode splits on whitespace)
        let ids = tok.encode("cat sat");
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "catsat");
    }

    #[test]
    fn test_encode_returns_ids() {
        let tok = BpeTokenizer::train(sample_text(), 270);
        let ids = tok.encode("the cat");
        assert!(!ids.is_empty());
        for id in &ids {
            assert!(tok.inverse_vocab.contains_key(id));
        }
    }

    #[test]
    fn test_vocab_size() {
        let tok = BpeTokenizer::train("ab ab ab cd cd", 260);
        assert!(tok.vocab_size() >= 256);
    }

    #[test]
    fn test_json_roundtrip() {
        let tok = BpeTokenizer::train(sample_text(), 270);
        let json = tok.to_json();
        let tok2 = BpeTokenizer::from_json(&json).unwrap();
        assert_eq!(tok.merges.len(), tok2.merges.len());
        assert_eq!(tok.vocab.len(), tok2.vocab.len());
    }

    #[test]
    fn test_save_load() {
        let tok = BpeTokenizer::train(sample_text(), 270);
        let path = "/tmp/vortex_bpe_test.json";
        std::fs::write(path, tok.to_json()).unwrap();
        let tok2 = BpeTokenizer::from_json(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert_eq!(tok.merges.len(), tok2.merges.len());
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_empty_input() {
        let tok = BpeTokenizer::train("", 270);
        let ids = tok.encode("");
        assert!(ids.is_empty());
    }

    #[test]
    fn test_batch_encode() {
        let tok = BpeTokenizer::train(sample_text(), 270);
        let texts = vec!["the cat", "sat on"];
        let results: Vec<Vec<u32>> = texts.iter().map(|t| tok.encode(t)).collect();
        assert_eq!(results.len(), 2);
        assert!(!results[0].is_empty());
        assert!(!results[1].is_empty());
    }
}
