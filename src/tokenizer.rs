//! BPE (Byte Pair Encoding) tokenizer for the Vortex language runtime.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use crate::interpreter::{Value, Env, FnDef};

static TOKENIZERS: LazyLock<Mutex<Vec<BpeTokenizer>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static HF_TOKENIZERS: LazyLock<Mutex<Vec<HfTokenizer>>> = LazyLock::new(|| Mutex::new(Vec::new()));

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
        // Find vocab_end by scanning, skipping over JSON strings to avoid false { } matches
        let mut depth = 1i32;
        let mut vocab_end = vocab_start;
        let bytes = json.as_bytes();
        let mut i = vocab_start;
        while i < bytes.len() && depth > 0 {
            match bytes[i] {
                b'"' => {
                    // Skip string
                    i += 1;
                    while i < bytes.len() {
                        if bytes[i] == b'\\' { i += 2; continue; }
                        if bytes[i] == b'"' { i += 1; break; }
                        i += 1;
                    }
                    continue;
                }
                b'{' => { depth += 1; }
                b'}' => {
                    depth -= 1;
                    if depth == 0 { vocab_end = i; break; }
                }
                _ => {}
            }
            i += 1;
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

// ─── HuggingFace tokenizer.json support ───

#[derive(Debug, Clone)]
pub struct HfTokenizer {
    pub vocab: HashMap<String, u32>,
    pub inverse_vocab: HashMap<u32, String>,
    pub merges: Vec<(String, String)>,
}

impl HfTokenizer {
    /// Parse a HuggingFace tokenizer.json format string.
    /// Expects {"model": {"vocab": {"token": id, ...}, "merges": ["a b", ...]}}
    pub fn from_hf_json(json: &str) -> Result<Self, String> {
        let json = json.trim();
        // Find "model" object
        let model_key = json.find("\"model\"").ok_or("missing \"model\" key")?;
        let model_brace = json[model_key..].find('{').ok_or("missing model object")? + model_key;

        // Find "vocab" inside model
        let vocab_key = json[model_brace..].find("\"vocab\"")
            .ok_or("missing \"vocab\" in model")? + model_brace;
        let vocab_brace = json[vocab_key..].find('{').ok_or("missing vocab object")? + vocab_key;
        let vocab_start = vocab_brace + 1;
        let mut depth = 1;
        let mut vocab_end = vocab_start;
        for (i, c) in json[vocab_start..].char_indices() {
            match c {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 { vocab_end = vocab_start + i; break; }
                }
                _ => {}
            }
        }
        let vocab_str = &json[vocab_start..vocab_end];

        // Parse vocab entries "token": id
        let mut vocab = HashMap::new();
        let mut inverse_vocab = HashMap::new();
        let mut rest = vocab_str;
        while let Some(qs) = rest.find('"') {
            rest = &rest[qs + 1..];
            // Find closing quote (handle escapes)
            let mut qe = 0;
            let bytes = rest.as_bytes();
            let mut bi = 0;
            while bi < bytes.len() {
                if bytes[bi] == b'\\' { bi += 2; continue; }
                if bytes[bi] == b'"' { qe = bi; break; }
                bi += 1;
            }
            let key = rest[..qe].replace("\\\"", "\"").replace("\\\\", "\\");
            rest = &rest[qe + 1..];
            let cs = rest.find(':').ok_or("bad vocab separator")?;
            rest = &rest[cs + 1..];
            let ve = rest.find(|c: char| c == ',' || c == '}').unwrap_or(rest.len());
            let val: u32 = rest[..ve].trim().parse().map_err(|e| format!("vocab val: {}", e))?;
            inverse_vocab.insert(val, key.clone());
            vocab.insert(key, val);
            rest = &rest[ve..];
        }

        // Find "merges" array inside model
        let merges_key = json[model_brace..].find("\"merges\"")
            .ok_or("missing \"merges\" in model")? + model_brace;
        let merges_bracket = json[merges_key..].find('[').ok_or("missing merges array")? + merges_key;
        let merges_start = merges_bracket + 1;
        // Find matching ]
        let mut depth = 1;
        let mut merges_end = merges_start;
        for (i, c) in json[merges_start..].char_indices() {
            match c {
                '[' => depth += 1,
                ']' => {
                    depth -= 1;
                    if depth == 0 { merges_end = merges_start + i; break; }
                }
                _ => {}
            }
        }
        let merges_str = &json[merges_start..merges_end];

        // Parse merge strings: "a b" format
        let mut merges = Vec::new();
        let mut mrest = merges_str;
        while let Some(qs) = mrest.find('"') {
            mrest = &mrest[qs + 1..];
            let qe = mrest.find('"').ok_or("bad merge string")?;
            let merge_str = &mrest[..qe];
            // Split on first space
            if let Some(sp) = merge_str.find(' ') {
                let a = merge_str[..sp].to_string();
                let b = merge_str[sp + 1..].to_string();
                merges.push((a, b));
            }
            mrest = &mrest[qe + 1..];
        }

        Ok(Self { vocab, inverse_vocab, merges })
    }

    /// Apply BPE merges to encode text into token ids.
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
}

// ─── GPT-2 byte-level BPE utilities ───

/// GPT-2 byte-to-unicode mapping. Bytes 33..=126, 161..=172, 174..=255 map to
/// themselves as chars; remaining 0..=32, 127..=160, 173 map to 256+.
pub fn gpt2_byte_encoder() -> HashMap<u8, char> {
    let mut map = HashMap::new();
    let mut offset = 0u32;
    for b in 0u8..=255 {
        let direct = matches!(b, 33..=126 | 161..=172 | 174..=255);
        if direct {
            map.insert(b, b as char);
        } else {
            map.insert(b, char::from_u32(256 + offset).unwrap());
            offset += 1;
        }
    }
    map
}

/// Inverse of gpt2_byte_encoder.
pub fn gpt2_byte_decoder() -> HashMap<char, u8> {
    gpt2_byte_encoder().into_iter().map(|(b, c)| (c, b)).collect()
}

/// GPT-2 pre-tokenization: split text into pieces matching the GPT-2 pattern.
/// Splits on: contractions ('s, 't, 're, 've, 'm, 'll, 'd), letter runs,
/// digit runs, non-letter-non-digit-non-space runs, and space-prefixed runs.
/// Implemented as a state machine (no regex dependency).
pub fn gpt2_pretokenize(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < chars.len() {
        // Check for contractions: 's 't 're 've 'm 'll 'd
        if chars[i] == '\'' && i + 1 < chars.len() {
            let rest = &chars[i + 1..];
            let contraction = if rest.len() >= 2
                && (rest[0] == 'l' || rest[0] == 'L')
                && (rest[1] == 'l' || rest[1] == 'L')
            {
                Some(3) // 'll
            } else if rest.len() >= 2
                && (rest[0] == 'r' || rest[0] == 'R')
                && (rest[1] == 'e' || rest[1] == 'E')
            {
                Some(3) // 're
            } else if rest.len() >= 2
                && (rest[0] == 'v' || rest[0] == 'V')
                && (rest[1] == 'e' || rest[1] == 'E')
            {
                Some(3) // 've
            } else if !rest.is_empty()
                && (rest[0] == 's' || rest[0] == 'S'
                    || rest[0] == 't' || rest[0] == 'T'
                    || rest[0] == 'm' || rest[0] == 'M'
                    || rest[0] == 'd' || rest[0] == 'D')
            {
                Some(2)
            } else {
                None
            };
            if let Some(len) = contraction {
                tokens.push(chars[i..i + len].iter().collect());
                i += len;
                continue;
            }
        }

        // Space-prefixed sequence: space followed by non-space chars
        if chars[i] == ' ' && i + 1 < chars.len() && chars[i + 1] != ' ' {
            let start = i;
            i += 1; // consume the space
            // Consume following non-space chars of the same category
            if chars[i].is_alphabetic() {
                while i < chars.len() && chars[i].is_alphabetic() {
                    i += 1;
                }
            } else if chars[i].is_ascii_digit() {
                while i < chars.len() && chars[i].is_ascii_digit() {
                    i += 1;
                }
            } else if !chars[i].is_alphanumeric() && chars[i] != ' ' {
                while i < chars.len() && !chars[i].is_alphanumeric() && chars[i] != ' ' {
                    i += 1;
                }
            } else {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // Letter run
        if chars[i].is_alphabetic() {
            let start = i;
            while i < chars.len() && chars[i].is_alphabetic() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // Digit run
        if chars[i].is_ascii_digit() {
            let start = i;
            while i < chars.len() && chars[i].is_ascii_digit() {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // Non-alphanumeric, non-space run
        if !chars[i].is_alphanumeric() && chars[i] != ' ' {
            let start = i;
            while i < chars.len() && !chars[i].is_alphanumeric() && chars[i] != ' ' {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }

        // Single space (not followed by non-space, or trailing)
        tokens.push(chars[i].to_string());
        i += 1;
    }
    tokens
}

/// GPT-2 style BPE encode: pretokenize, byte-encode each piece, then apply BPE merges.
pub fn bpe_encode_gpt2(text: &str, tokenizer_id: usize) -> Result<Vec<u32>, String> {
    let store = HF_TOKENIZERS.lock().map_err(|e| format!("lock: {}", e))?;
    let tok = store.get(tokenizer_id)
        .ok_or_else(|| format!("invalid hf tokenizer id: {}", tokenizer_id))?;
    let byte_enc = gpt2_byte_encoder();

    let pieces = gpt2_pretokenize(text);
    let mut result = Vec::new();
    for piece in &pieces {
        // Convert piece bytes to GPT-2 unicode representation
        let encoded_piece: String = piece.bytes().map(|b| byte_enc[&b]).collect();
        // Tokenize using BPE
        let mut syms: Vec<String> = encoded_piece.chars().map(|c| c.to_string()).collect();
        for (a, b) in &tok.merges {
            let merged = format!("{}{}", a, b);
            let mut j = 0;
            while j + 1 < syms.len() {
                if syms[j] == *a && syms[j + 1] == *b {
                    syms[j] = merged.clone();
                    syms.remove(j + 1);
                } else {
                    j += 1;
                }
            }
        }
        for s in &syms {
            if let Some(&id) = tok.vocab.get(s) {
                result.push(id);
            }
        }
    }
    Ok(result)
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

    env.functions.insert("tokenizer_load_hf".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 1 { return Err("tokenizer_load_hf(json_string)".into()); }
        let json = match &args[0] { Value::String(s) => s.clone(), _ => return Err("json must be string".into()) };
        let tok = HfTokenizer::from_hf_json(&json)?;
        let mut store = HF_TOKENIZERS.lock().map_err(|e| format!("{}", e))?;
        let id = store.len();
        store.push(tok);
        Ok(Value::Int(id as i128))
    }));

    env.functions.insert("tokenizer_from_file".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 1 { return Err("tokenizer_from_file(path)".into()); }
        let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
        let data = std::fs::read_to_string(&path).map_err(|e| format!("read: {}", e))?;
        let tok = HfTokenizer::from_hf_json(&data)?;
        let mut store = HF_TOKENIZERS.lock().map_err(|e| format!("{}", e))?;
        let id = store.len();
        store.push(tok);
        Ok(Value::Int(id as i128))
    }));

    env.functions.insert("bpe_encode_gpt2".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("bpe_encode_gpt2(tokenizer_id, text)".into()); }
        let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
        let text = match &args[1] { Value::String(s) => s.clone(), _ => return Err("text must be string".into()) };
        let ids = bpe_encode_gpt2(&text, id)?;
        Ok(Value::Array(ids.into_iter().map(|i| Value::Int(i as i128)).collect()))
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
    fn test_hf_tokenizer_from_json() {
        let json = r#"{
            "model": {
                "type": "BPE",
                "vocab": {"a": 0, "b": 1, "c": 2, "ab": 3},
                "merges": ["a b"]
            }
        }"#;
        let tok = HfTokenizer::from_hf_json(json).unwrap();
        assert_eq!(tok.vocab.len(), 4);
        assert_eq!(tok.vocab["ab"], 3);
        assert_eq!(tok.merges.len(), 1);
        assert_eq!(tok.merges[0], ("a".to_string(), "b".to_string()));
        // encode: "ab" should merge to single token
        let ids = tok.encode("ab");
        assert_eq!(ids, vec![3]);
        // decode roundtrip
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, "ab");
    }

    #[test]
    fn test_gpt2_byte_encoder_decoder_roundtrip() {
        let enc = gpt2_byte_encoder();
        let dec = gpt2_byte_decoder();
        assert_eq!(enc.len(), 256);
        assert_eq!(dec.len(), 256);
        // Every byte roundtrips
        for b in 0u8..=255 {
            let c = enc[&b];
            assert_eq!(dec[&c], b);
        }
        // Direct-mapped bytes map to themselves
        assert_eq!(enc[&65], 'A');
        assert_eq!(enc[&48], '0');
    }

    #[test]
    fn test_gpt2_pretokenize() {
        let tokens = gpt2_pretokenize("Hello world");
        assert_eq!(tokens, vec!["Hello", " world"]);

        let tokens = gpt2_pretokenize("I'm happy");
        assert_eq!(tokens, vec!["I", "'m", " happy"]);

        let tokens = gpt2_pretokenize("test123 foo");
        assert!(tokens.contains(&"test".to_string()));
        assert!(tokens.contains(&"123".to_string()));

        let tokens = gpt2_pretokenize("don't");
        assert!(tokens.contains(&"don".to_string()));
        assert!(tokens.contains(&"'t".to_string()));
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
