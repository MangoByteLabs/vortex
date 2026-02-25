use crate::interpreter::{Env, Value, FnDef};
use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

// ---------------------------------------------------------------------------
// String interning global state
// ---------------------------------------------------------------------------

static INTERN_TABLE: LazyLock<Mutex<(Vec<String>, HashMap<String, usize>)>> =
    LazyLock::new(|| Mutex::new((Vec::new(), HashMap::new())));

// ---------------------------------------------------------------------------
// FNV-1a hash
// ---------------------------------------------------------------------------

fn fnv1a(bytes: &[u8]) -> u64 {
    let mut hash: u64 = 0xcbf29ce484222325;
    for &b in bytes {
        hash ^= b as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

// ---------------------------------------------------------------------------
// String processing builtins
// ---------------------------------------------------------------------------

fn builtin_str_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            let arr = s.as_bytes().iter().map(|&b| Value::Int(b as i128)).collect();
            Ok(Value::Array(arr))
        }
        _ => Err("str_bytes expects (String)".into()),
    }
}

fn builtin_str_from_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::Array(arr)] => {
            let mut bytes = Vec::with_capacity(arr.len());
            for v in arr {
                match v {
                    Value::Int(i) => {
                        if *i < 0 || *i > 255 {
                            return Err(format!("str_from_bytes: byte value {} out of range", i));
                        }
                        bytes.push(*i as u8);
                    }
                    _ => return Err("str_from_bytes expects Array[Int]".into()),
                }
            }
            match String::from_utf8(bytes) {
                Ok(s) => Ok(Value::String(s)),
                Err(e) => Err(format!("str_from_bytes: invalid UTF-8: {}", e)),
            }
        }
        _ => Err("str_from_bytes expects (Array[Int])".into()),
    }
}

fn builtin_str_char_at(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s), Value::Int(idx)] => {
            let idx = *idx as usize;
            match s.chars().nth(idx) {
                Some(c) => Ok(Value::String(c.to_string())),
                None => Err(format!("str_char_at: index {} out of bounds for len {}", idx, s.chars().count())),
            }
        }
        _ => Err("str_char_at expects (String, Int)".into()),
    }
}

fn builtin_str_substr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s), Value::Int(start), Value::Int(len)] => {
            let start = *start as usize;
            let len = *len as usize;
            let result: String = s.chars().skip(start).take(len).collect();
            Ok(Value::String(result))
        }
        _ => Err("str_substr expects (String, Int, Int)".into()),
    }
}

fn builtin_str_find(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s), Value::String(needle)] => {
            match s.find(needle.as_str()) {
                Some(idx) => Ok(Value::Option(Some(Box::new(Value::Int(idx as i128))))),
                None => Ok(Value::Option(None)),
            }
        }
        _ => Err("str_find expects (String, String)".into()),
    }
}

fn builtin_str_split_at(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s), Value::Int(idx)] => {
            let idx = *idx as usize;
            if idx > s.len() {
                return Err(format!("str_split_at: index {} out of bounds for byte len {}", idx, s.len()));
            }
            if !s.is_char_boundary(idx) {
                return Err(format!("str_split_at: index {} is not a char boundary", idx));
            }
            let (a, b) = s.split_at(idx);
            Ok(Value::Array(vec![Value::String(a.to_string()), Value::String(b.to_string())]))
        }
        _ => Err("str_split_at expects (String, Int)".into()),
    }
}

fn builtin_str_is_ascii_alpha(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic())))
        }
        _ => Err("str_is_ascii_alpha expects (String)".into()),
    }
}

fn builtin_str_is_ascii_digit(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_digit())))
        }
        _ => Err("str_is_ascii_digit expects (String)".into()),
    }
}

fn builtin_str_is_ascii_whitespace(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            Ok(Value::Bool(!s.is_empty() && s.chars().all(|c| c.is_ascii_whitespace())))
        }
        _ => Err("str_is_ascii_whitespace expects (String)".into()),
    }
}

fn builtin_str_to_chars(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            let arr = s.chars().map(|c| Value::String(c.to_string())).collect();
            Ok(Value::Array(arr))
        }
        _ => Err("str_to_chars expects (String)".into()),
    }
}

fn builtin_str_from_chars(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::Array(arr)] => {
            let mut result = String::with_capacity(arr.len());
            for v in arr {
                match v {
                    Value::String(s) => result.push_str(s),
                    _ => return Err("str_from_chars expects Array[String]".into()),
                }
            }
            Ok(Value::String(result))
        }
        _ => Err("str_from_chars expects (Array[String])".into()),
    }
}

// ---------------------------------------------------------------------------
// String interning builtins
// ---------------------------------------------------------------------------

fn builtin_str_intern(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            let mut table = INTERN_TABLE.lock().unwrap();
            let (ref mut vec, ref mut map) = *table;
            if let Some(&id) = map.get(s) {
                Ok(Value::Int(id as i128))
            } else {
                let id = vec.len();
                vec.push(s.clone());
                map.insert(s.clone(), id);
                Ok(Value::Int(id as i128))
            }
        }
        _ => Err("str_intern expects (String)".into()),
    }
}

fn builtin_str_lookup(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::Int(id)] => {
            let id = *id as usize;
            let table = INTERN_TABLE.lock().unwrap();
            match table.0.get(id) {
                Some(s) => Ok(Value::String(s.clone())),
                None => Err(format!("str_lookup: id {} not found", id)),
            }
        }
        _ => Err("str_lookup expects (Int)".into()),
    }
}

fn builtin_str_hash(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(s)] => {
            let h = fnv1a(s.as_bytes());
            Ok(Value::Int(h as i128))
        }
        _ => Err("str_hash expects (String)".into()),
    }
}

// ---------------------------------------------------------------------------
// Simple regex builtins
// ---------------------------------------------------------------------------

fn builtin_regex_match(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(pattern), Value::String(text)] => {
            Ok(Value::Bool(simple_regex_match(pattern, text)))
        }
        _ => Err("regex_match expects (String, String)".into()),
    }
}

fn simple_regex_match(pattern: &str, text: &str) -> bool {
    let anchor_start = pattern.starts_with('^');
    let anchor_end = pattern.ends_with('$');

    let inner = {
        let s = if anchor_start { &pattern[1..] } else { pattern };
        if anchor_end && !s.is_empty() { &s[..s.len() - 1] } else { s }
    };

    // Handle .* wildcard: split pattern on first ".*" and check parts
    if let Some(pos) = inner.find(".*") {
        let before = &inner[..pos];
        let after = &inner[pos + 2..];
        return match (anchor_start, anchor_end) {
            (true, true) => text.starts_with(before) && text.ends_with(after),
            (true, false) => {
                text.starts_with(before) && text[before.len()..].contains(after)
            }
            (false, true) => {
                text.ends_with(after) && text[..text.len() - after.len()].contains(before)
            }
            (false, false) => {
                if let Some(i) = text.find(before) {
                    text[i + before.len()..].contains(after)
                } else {
                    false
                }
            }
        };
    }

    // No wildcard — literal matching
    match (anchor_start, anchor_end) {
        (true, true) => text == inner,
        (true, false) => text.starts_with(inner),
        (false, true) => text.ends_with(inner),
        (false, false) => text.contains(inner),
    }
}

fn builtin_regex_find_all(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.as_slice() {
        [Value::String(pattern), Value::String(text)] => {
            let mut results = Vec::new();
            if pattern.is_empty() {
                return Ok(Value::Array(results));
            }
            let mut start = 0;
            while start <= text.len() {
                if let Some(pos) = text[start..].find(pattern.as_str()) {
                    results.push(Value::String(pattern.clone()));
                    start += pos + pattern.len();
                } else {
                    break;
                }
            }
            Ok(Value::Array(results))
        }
        _ => Err("regex_find_all expects (String, String)".into()),
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    // String processing
    env.functions.insert("str_bytes".into(), FnDef::Builtin(builtin_str_bytes));
    env.functions.insert("str_from_bytes".into(), FnDef::Builtin(builtin_str_from_bytes));
    env.functions.insert("str_char_at".into(), FnDef::Builtin(builtin_str_char_at));
    env.functions.insert("str_substr".into(), FnDef::Builtin(builtin_str_substr));
    env.functions.insert("str_find".into(), FnDef::Builtin(builtin_str_find));
    env.functions.insert("str_split_at".into(), FnDef::Builtin(builtin_str_split_at));
    env.functions.insert("str_is_ascii_alpha".into(), FnDef::Builtin(builtin_str_is_ascii_alpha));
    env.functions.insert("str_is_ascii_digit".into(), FnDef::Builtin(builtin_str_is_ascii_digit));
    env.functions.insert("str_is_ascii_whitespace".into(), FnDef::Builtin(builtin_str_is_ascii_whitespace));
    env.functions.insert("str_to_chars".into(), FnDef::Builtin(builtin_str_to_chars));
    env.functions.insert("str_from_chars".into(), FnDef::Builtin(builtin_str_from_chars));

    // String interning
    env.functions.insert("str_intern".into(), FnDef::Builtin(builtin_str_intern));
    env.functions.insert("str_lookup".into(), FnDef::Builtin(builtin_str_lookup));
    env.functions.insert("str_hash".into(), FnDef::Builtin(builtin_str_hash));

    // Regex
    env.functions.insert("regex_match".into(), FnDef::Builtin(builtin_regex_match));
    env.functions.insert("regex_find_all".into(), FnDef::Builtin(builtin_regex_find_all));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env {
        Env::new()
    }

    fn s(v: &str) -> Value {
        Value::String(v.to_string())
    }

    fn int(v: i128) -> Value {
        Value::Int(v)
    }

    #[test]
    fn test_str_bytes_roundtrip() {
        let mut e = env();
        let bytes = builtin_str_bytes(&mut e, vec![s("hello")]).unwrap();
        let back = builtin_str_from_bytes(&mut e, vec![bytes]).unwrap();
        match back { Value::String(ref v) => assert_eq!(v, "hello"), ref other => panic!("expected String, got {:?}", other) }
    }

    #[test]
    fn test_str_char_at_ascii() {
        let mut e = env();
        let r = builtin_str_char_at(&mut e, vec![s("hello"), int(1)]).unwrap();
        match r { Value::String(ref v) => assert_eq!(v, "e"), ref other => panic!("expected String, got {:?}", other) }
    }

    #[test]
    fn test_str_char_at_multibyte() {
        let mut e = env();
        // "café" — 'é' is at char index 3
        let r = builtin_str_char_at(&mut e, vec![s("café"), int(3)]).unwrap();
        match r { Value::String(ref v) => assert_eq!(v, "é"), ref other => panic!("expected String, got {:?}", other) }
    }

    #[test]
    fn test_str_substr() {
        let mut e = env();
        let r = builtin_str_substr(&mut e, vec![s("hello world"), int(6), int(5)]).unwrap();
        match r { Value::String(ref v) => assert_eq!(v, "world"), ref other => panic!("expected String, got {:?}", other) }
    }

    #[test]
    fn test_str_find_found() {
        let mut e = env();
        let r = builtin_str_find(&mut e, vec![s("hello world"), s("world")]).unwrap();
        match r { Value::Option(Some(ref inner)) => match **inner { Value::Int(v) => assert_eq!(v, 6), ref other => panic!("expected Int, got {:?}", other) }, ref other => panic!("expected Option(Some), got {:?}", other) }
    }

    #[test]
    fn test_str_find_not_found() {
        let mut e = env();
        let r = builtin_str_find(&mut e, vec![s("hello"), s("xyz")]).unwrap();
        assert!(matches!(r, Value::Option(None)));
    }

    #[test]
    fn test_str_is_ascii_alpha() {
        let mut e = env();
        match builtin_str_is_ascii_alpha(&mut e, vec![s("abc")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool(true), got {:?}", other) }
        match builtin_str_is_ascii_alpha(&mut e, vec![s("ab3")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool(false), got {:?}", other) }
        match builtin_str_is_ascii_alpha(&mut e, vec![s("")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool(false), got {:?}", other) }
    }

    #[test]
    fn test_str_is_ascii_digit() {
        let mut e = env();
        match builtin_str_is_ascii_digit(&mut e, vec![s("123")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool(true), got {:?}", other) }
        match builtin_str_is_ascii_digit(&mut e, vec![s("12a")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool(false), got {:?}", other) }
    }

    #[test]
    fn test_str_is_ascii_whitespace() {
        let mut e = env();
        match builtin_str_is_ascii_whitespace(&mut e, vec![s(" \t\n")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool(true), got {:?}", other) }
        match builtin_str_is_ascii_whitespace(&mut e, vec![s(" x ")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool(false), got {:?}", other) }
    }

    #[test]
    fn test_str_to_chars_from_chars_roundtrip() {
        let mut e = env();
        let chars = builtin_str_to_chars(&mut e, vec![s("café")]).unwrap();
        let back = builtin_str_from_chars(&mut e, vec![chars]).unwrap();
        match back { Value::String(ref v) => assert_eq!(v, "café"), ref other => panic!("expected String, got {:?}", other) }
    }

    #[test]
    fn test_str_intern_lookup_roundtrip() {
        let mut e = env();
        let id = builtin_str_intern(&mut e, vec![s("test_intern_val")]).unwrap();
        let looked = builtin_str_lookup(&mut e, vec![id.clone()]).unwrap();
        match looked { Value::String(ref v) => assert_eq!(v, "test_intern_val"), ref other => panic!("expected String, got {:?}", other) }
        // Interning again returns same id
        let id2 = builtin_str_intern(&mut e, vec![s("test_intern_val")]).unwrap();
        match (&id, &id2) { (Value::Int(a), Value::Int(b)) => assert_eq!(a, b), _ => panic!("expected Int values") }
    }

    #[test]
    fn test_str_hash_consistency() {
        let mut e = env();
        let h1 = builtin_str_hash(&mut e, vec![s("hello")]).unwrap();
        let h2 = builtin_str_hash(&mut e, vec![s("hello")]).unwrap();
        match (&h1, &h2) { (Value::Int(a), Value::Int(b)) => assert_eq!(a, b), _ => panic!("expected Int values") }
        let h3 = builtin_str_hash(&mut e, vec![s("world")]).unwrap();
        match (&h1, &h3) { (Value::Int(a), Value::Int(b)) => assert_ne!(a, b), _ => panic!("expected Int values") }
    }

    #[test]
    fn test_regex_match_basic() {
        let mut e = env();
        // Exact substring
        match builtin_regex_match(&mut e, vec![s("ell"), s("hello")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
        // Anchored start
        match builtin_regex_match(&mut e, vec![s("^hel"), s("hello")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
        match builtin_regex_match(&mut e, vec![s("^ell"), s("hello")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool, got {:?}", other) }
        // Anchored end
        match builtin_regex_match(&mut e, vec![s("llo$"), s("hello")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
        // Both anchors (exact)
        match builtin_regex_match(&mut e, vec![s("^hello$"), s("hello")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
        match builtin_regex_match(&mut e, vec![s("^hell$"), s("hello")]).unwrap() { Value::Bool(v) => assert!(!v), other => panic!("expected Bool, got {:?}", other) }
        // Wildcard
        match builtin_regex_match(&mut e, vec![s("hel.*rld"), s("hello world")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
        match builtin_regex_match(&mut e, vec![s("^hel.*rld$"), s("hello world")]).unwrap() { Value::Bool(v) => assert!(v), other => panic!("expected Bool, got {:?}", other) }
    }

    #[test]
    fn test_regex_find_all_literal() {
        let mut e = env();
        let r = builtin_regex_find_all(&mut e, vec![s("ab"), s("ababcab")]).unwrap();
        match r {
            Value::Array(arr) => assert_eq!(arr.len(), 3),
            _ => panic!("expected array"),
        }
        // Empty pattern
        let r2 = builtin_regex_find_all(&mut e, vec![s(""), s("hello")]).unwrap();
        match r2 {
            Value::Array(arr) => assert_eq!(arr.len(), 0),
            _ => panic!("expected array"),
        }
    }
}
