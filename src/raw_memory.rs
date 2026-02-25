// raw_memory.rs — Raw memory access builtins for Vortex
//
// Provides unsafe memory allocation, read/write of primitive types,
// bulk operations, pointer arithmetic, and size queries.

use crate::interpreter::{Value, Env, FnDef};

use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

// Global allocation tracker: ptr -> size
static ALLOCATIONS: LazyLock<Mutex<HashMap<usize, usize>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn realloc(ptr: *mut u8, size: usize) -> *mut u8;
    fn memcpy(dst: *mut u8, src: *const u8, n: usize) -> *mut u8;
    fn memset(s: *mut u8, c: i32, n: usize) -> *mut u8;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_ptr(v: &Value) -> Result<usize, String> {
    match v {
        Value::Pointer(p) => Ok(*p),
        Value::Int(i) => Ok(*i as usize),
        _ => Err(format!("expected Pointer, got {:?}", v)),
    }
}

fn extract_int(v: &Value) -> Result<i128, String> {
    match v {
        Value::Int(i) => Ok(*i),
        _ => Err(format!("expected Int, got {:?}", v)),
    }
}

fn extract_float(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("expected Float, got {:?}", v)),
    }
}

fn extract_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(format!("expected String, got {:?}", v)),
    }
}

// ---------------------------------------------------------------------------
// Allocation builtins
// ---------------------------------------------------------------------------

fn builtin_mem_alloc(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("mem_alloc expects 1 argument (size)".into());
    }
    let size = extract_int(&args[0])? as usize;
    if size == 0 {
        return Err("mem_alloc: size must be > 0".into());
    }
    let ptr = unsafe { malloc(size) };
    if ptr.is_null() {
        return Err("mem_alloc: allocation failed".into());
    }
    let addr = ptr as usize;
    ALLOCATIONS.lock().unwrap().insert(addr, size);
    Ok(Value::Pointer(addr))
}

fn builtin_mem_free(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("mem_free expects 1 argument (ptr)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let mut allocs = ALLOCATIONS.lock().unwrap();
    if !allocs.contains_key(&addr) {
        return Err(format!("mem_free: pointer 0x{:x} not tracked (double-free or invalid)", addr));
    }
    allocs.remove(&addr);
    unsafe { free(addr as *mut u8) };
    Ok(Value::Void)
}

fn builtin_mem_realloc(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_realloc expects 2 arguments (ptr, new_size)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let new_size = extract_int(&args[1])? as usize;
    if new_size == 0 {
        return Err("mem_realloc: new_size must be > 0".into());
    }
    let mut allocs = ALLOCATIONS.lock().unwrap();
    if !allocs.contains_key(&addr) {
        return Err(format!("mem_realloc: pointer 0x{:x} not tracked", addr));
    }
    let new_ptr = unsafe { realloc(addr as *mut u8, new_size) };
    if new_ptr.is_null() {
        return Err("mem_realloc: reallocation failed".into());
    }
    let new_addr = new_ptr as usize;
    allocs.remove(&addr);
    allocs.insert(new_addr, new_size);
    Ok(Value::Pointer(new_addr))
}

// ---------------------------------------------------------------------------
// Read primitives
// ---------------------------------------------------------------------------

macro_rules! read_builtin {
    ($name:ident, $ty:ty, Int) => {
        fn $name(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            if args.len() < 1 || args.len() > 2 {
                return Err(format!("{} expects 1-2 arguments (ptr [, offset])", stringify!($name)));
            }
            let mut addr = extract_ptr(&args[0])?;
            if args.len() == 2 {
                let offset = match &args[1] { Value::Int(n) => *n as usize, Value::Pointer(p) => *p, _ => return Err(format!("{}: offset must be int", stringify!($name))) };
                addr += offset;
            }
            let val = unsafe { std::ptr::read_unaligned(addr as *const $ty) };
            Ok(Value::Int(val as i128))
        }
    };
    ($name:ident, $ty:ty, Float) => {
        fn $name(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            if args.len() < 1 || args.len() > 2 {
                return Err(format!("{} expects 1-2 arguments (ptr [, offset])", stringify!($name)));
            }
            let mut addr = extract_ptr(&args[0])?;
            if args.len() == 2 {
                let offset = match &args[1] { Value::Int(n) => *n as usize, Value::Pointer(p) => *p, _ => return Err(format!("{}: offset must be int", stringify!($name))) };
                addr += offset;
            }
            let val = unsafe { std::ptr::read_unaligned(addr as *const $ty) };
            Ok(Value::Float(val as f64))
        }
    };
}

read_builtin!(builtin_mem_read_u8,  u8,  Int);
read_builtin!(builtin_mem_read_u16, u16, Int);
read_builtin!(builtin_mem_read_u32, u32, Int);
read_builtin!(builtin_mem_read_u64, u64, Int);
read_builtin!(builtin_mem_read_i8,  i8,  Int);
read_builtin!(builtin_mem_read_i16, i16, Int);
read_builtin!(builtin_mem_read_i32, i32, Int);
read_builtin!(builtin_mem_read_i64, i64, Int);
read_builtin!(builtin_mem_read_f32, f32, Float);
read_builtin!(builtin_mem_read_f64, f64, Float);

// ---------------------------------------------------------------------------
// Write primitives
// ---------------------------------------------------------------------------

macro_rules! write_builtin {
    ($name:ident, $ty:ty, Int) => {
        fn $name(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            if args.len() < 2 || args.len() > 3 {
                return Err(format!("{} expects 2-3 arguments (ptr, val) or (ptr, offset, val)", stringify!($name)));
            }
            let mut addr = extract_ptr(&args[0])?;
            let val_idx = if args.len() == 3 {
                let offset = match &args[1] { Value::Int(n) => *n as usize, Value::Pointer(p) => *p, _ => return Err(format!("{}: offset must be int", stringify!($name))) };
                addr += offset;
                2
            } else { 1 };
            let val = extract_int(&args[val_idx])? as $ty;
            unsafe { std::ptr::write_unaligned(addr as *mut $ty, val) };
            Ok(Value::Void)
        }
    };
    ($name:ident, $ty:ty, Float) => {
        fn $name(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            if args.len() < 2 || args.len() > 3 {
                return Err(format!("{} expects 2-3 arguments (ptr, val) or (ptr, offset, val)", stringify!($name)));
            }
            let mut addr = extract_ptr(&args[0])?;
            let val_idx = if args.len() == 3 {
                let offset = match &args[1] { Value::Int(n) => *n as usize, Value::Pointer(p) => *p, _ => return Err(format!("{}: offset must be int", stringify!($name))) };
                addr += offset;
                2
            } else { 1 };
            let val = extract_float(&args[val_idx])? as $ty;
            unsafe { std::ptr::write_unaligned(addr as *mut $ty, val) };
            Ok(Value::Void)
        }
    };
}

write_builtin!(builtin_mem_write_u8,  u8,  Int);
write_builtin!(builtin_mem_write_u16, u16, Int);
write_builtin!(builtin_mem_write_u32, u32, Int);
write_builtin!(builtin_mem_write_u64, u64, Int);
write_builtin!(builtin_mem_write_i8,  i8,  Int);
write_builtin!(builtin_mem_write_i16, i16, Int);
write_builtin!(builtin_mem_write_i32, i32, Int);
write_builtin!(builtin_mem_write_i64, i64, Int);
write_builtin!(builtin_mem_write_f32, f32, Float);
write_builtin!(builtin_mem_write_f64, f64, Float);

// ---------------------------------------------------------------------------
// Bulk operations
// ---------------------------------------------------------------------------

fn builtin_mem_copy(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("mem_copy expects 3 arguments (dst, src, len)".into());
    }
    let dst = extract_ptr(&args[0])?;
    let src = extract_ptr(&args[1])?;
    let len = extract_int(&args[2])? as usize;
    unsafe { memcpy(dst as *mut u8, src as *const u8, len) };
    Ok(Value::Void)
}

fn builtin_mem_zero(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_zero expects 2 arguments (ptr, len)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let len = extract_int(&args[1])? as usize;
    unsafe { memset(addr as *mut u8, 0, len) };
    Ok(Value::Void)
}

fn builtin_mem_read_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_read_bytes expects 2 arguments (ptr, len)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let len = extract_int(&args[1])? as usize;
    let mut result = Vec::with_capacity(len);
    for i in 0..len {
        let byte = unsafe { std::ptr::read_unaligned((addr + i) as *const u8) };
        result.push(Value::Int(byte as i128));
    }
    Ok(Value::Array(result))
}

fn builtin_mem_write_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_write_bytes expects 2 arguments (ptr, bytes)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let bytes = match &args[1] {
        Value::Array(arr) => arr,
        _ => return Err("mem_write_bytes: second argument must be Array".into()),
    };
    for (i, v) in bytes.iter().enumerate() {
        let byte = extract_int(v)? as u8;
        unsafe { std::ptr::write_unaligned((addr + i) as *mut u8, byte) };
    }
    Ok(Value::Void)
}

fn builtin_mem_read_string(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_read_string expects 2 arguments (ptr, len)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let len = extract_int(&args[1])? as usize;
    let mut bytes = Vec::with_capacity(len);
    for i in 0..len {
        let byte = unsafe { std::ptr::read_unaligned((addr + i) as *const u8) };
        bytes.push(byte);
    }
    let s = std::string::String::from_utf8(bytes)
        .map_err(|e| format!("mem_read_string: invalid UTF-8: {}", e))?;
    Ok(Value::String(s))
}

fn builtin_mem_write_string(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("mem_write_string expects 2 arguments (ptr, string)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let s = extract_string(&args[1])?;
    let bytes = s.as_bytes();
    for (i, &byte) in bytes.iter().enumerate() {
        unsafe { std::ptr::write_unaligned((addr + i) as *mut u8, byte) };
    }
    Ok(Value::Void)
}

// ---------------------------------------------------------------------------
// Pointer arithmetic
// ---------------------------------------------------------------------------

fn builtin_ptr_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ptr_add expects 2 arguments (ptr, offset)".into());
    }
    let addr = extract_ptr(&args[0])?;
    let offset = extract_int(&args[1])?;
    let result = if offset >= 0 {
        addr.wrapping_add(offset as usize)
    } else {
        addr.wrapping_sub((-offset) as usize)
    };
    Ok(Value::Pointer(result))
}

fn builtin_ptr_to_int(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("ptr_to_int expects 1 argument (ptr)".into());
    }
    let addr = extract_ptr(&args[0])?;
    Ok(Value::Int(addr as i128))
}

fn builtin_int_to_ptr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("int_to_ptr expects 1 argument (addr)".into());
    }
    let addr = extract_int(&args[0])? as usize;
    Ok(Value::Pointer(addr))
}

// ---------------------------------------------------------------------------
// Utility
// ---------------------------------------------------------------------------

fn builtin_mem_size_of(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("mem_size_of expects 1 argument (type_name)".into());
    }
    let name = extract_string(&args[0])?;
    let size = match name.as_str() {
        "u8"  | "i8"  => 1,
        "u16" | "i16" => 2,
        "u32" | "i32" | "f32" => 4,
        "u64" | "i64" | "f64" => 8,
        "ptr" => std::mem::size_of::<usize>(),
        _ => return Err(format!("mem_size_of: unknown type '{}'", name)),
    };
    Ok(Value::Int(size as i128))
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    // Allocation
    env.functions.insert("mem_alloc".to_string(),   FnDef::Builtin(builtin_mem_alloc));
    env.functions.insert("mem_free".to_string(),    FnDef::Builtin(builtin_mem_free));
    env.functions.insert("mem_realloc".to_string(), FnDef::Builtin(builtin_mem_realloc));

    // Read primitives
    env.functions.insert("mem_read_u8".to_string(),  FnDef::Builtin(builtin_mem_read_u8));
    env.functions.insert("mem_read_u16".to_string(), FnDef::Builtin(builtin_mem_read_u16));
    env.functions.insert("mem_read_u32".to_string(), FnDef::Builtin(builtin_mem_read_u32));
    env.functions.insert("mem_read_u64".to_string(), FnDef::Builtin(builtin_mem_read_u64));
    env.functions.insert("mem_read_i8".to_string(),  FnDef::Builtin(builtin_mem_read_i8));
    env.functions.insert("mem_read_i16".to_string(), FnDef::Builtin(builtin_mem_read_i16));
    env.functions.insert("mem_read_i32".to_string(), FnDef::Builtin(builtin_mem_read_i32));
    env.functions.insert("mem_read_i64".to_string(), FnDef::Builtin(builtin_mem_read_i64));
    env.functions.insert("mem_read_f32".to_string(), FnDef::Builtin(builtin_mem_read_f32));
    env.functions.insert("mem_read_f64".to_string(), FnDef::Builtin(builtin_mem_read_f64));

    // Write primitives
    env.functions.insert("mem_write_u8".to_string(),  FnDef::Builtin(builtin_mem_write_u8));
    env.functions.insert("mem_write_u16".to_string(), FnDef::Builtin(builtin_mem_write_u16));
    env.functions.insert("mem_write_u32".to_string(), FnDef::Builtin(builtin_mem_write_u32));
    env.functions.insert("mem_write_u64".to_string(), FnDef::Builtin(builtin_mem_write_u64));
    env.functions.insert("mem_write_i8".to_string(),  FnDef::Builtin(builtin_mem_write_i8));
    env.functions.insert("mem_write_i16".to_string(), FnDef::Builtin(builtin_mem_write_i16));
    env.functions.insert("mem_write_i32".to_string(), FnDef::Builtin(builtin_mem_write_i32));
    env.functions.insert("mem_write_i64".to_string(), FnDef::Builtin(builtin_mem_write_i64));
    env.functions.insert("mem_write_f32".to_string(), FnDef::Builtin(builtin_mem_write_f32));
    env.functions.insert("mem_write_f64".to_string(), FnDef::Builtin(builtin_mem_write_f64));

    // Bulk operations
    env.functions.insert("mem_copy".to_string(),         FnDef::Builtin(builtin_mem_copy));
    env.functions.insert("mem_zero".to_string(),         FnDef::Builtin(builtin_mem_zero));
    env.functions.insert("mem_read_bytes".to_string(),   FnDef::Builtin(builtin_mem_read_bytes));
    env.functions.insert("mem_write_bytes".to_string(),  FnDef::Builtin(builtin_mem_write_bytes));
    env.functions.insert("mem_read_string".to_string(),  FnDef::Builtin(builtin_mem_read_string));
    env.functions.insert("mem_write_string".to_string(), FnDef::Builtin(builtin_mem_write_string));

    // Pointer arithmetic
    env.functions.insert("ptr_add".to_string(),    FnDef::Builtin(builtin_ptr_add));
    env.functions.insert("ptr_to_int".to_string(), FnDef::Builtin(builtin_ptr_to_int));
    env.functions.insert("int_to_ptr".to_string(), FnDef::Builtin(builtin_int_to_ptr));

    // Utility
    env.functions.insert("mem_size_of".to_string(), FnDef::Builtin(builtin_mem_size_of));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal Env for testing builtins.
    fn test_env() -> Env {
        Env::new()
    }

    /// Helper: allocate `size` bytes via the builtin, return the pointer address.
    fn alloc(env: &mut Env, size: i128) -> usize {
        match builtin_mem_alloc(env, vec![Value::Int(size)]).unwrap() {
            Value::Pointer(p) => p,
            other => panic!("expected Pointer, got {:?}", other),
        }
    }

    #[test]
    fn test_alloc_free_roundtrip() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 64);
        assert!(ptr != 0);
        assert!(ALLOCATIONS.lock().unwrap().contains_key(&ptr));

        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        assert!(!ALLOCATIONS.lock().unwrap().contains_key(&ptr));
    }

    #[test]
    fn test_double_free_detected() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 32);
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();

        let result = builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("not tracked"));
    }

    #[test]
    fn test_read_write_u8() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 1);
        builtin_mem_write_u8(&mut env, vec![Value::Pointer(ptr), Value::Int(42)]).unwrap();
        let val = builtin_mem_read_u8(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 42), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_u16() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 2);
        builtin_mem_write_u16(&mut env, vec![Value::Pointer(ptr), Value::Int(1234)]).unwrap();
        let val = builtin_mem_read_u16(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 1234), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_u32() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 4);
        builtin_mem_write_u32(&mut env, vec![Value::Pointer(ptr), Value::Int(70000)]).unwrap();
        let val = builtin_mem_read_u32(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 70000), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_u64() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 8);
        let big: i128 = 1 << 40;
        builtin_mem_write_u64(&mut env, vec![Value::Pointer(ptr), Value::Int(big)]).unwrap();
        let val = builtin_mem_read_u64(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, big), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_i8() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 1);
        builtin_mem_write_i8(&mut env, vec![Value::Pointer(ptr), Value::Int(-42)]).unwrap();
        let val = builtin_mem_read_i8(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, -42), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_i16() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 2);
        builtin_mem_write_i16(&mut env, vec![Value::Pointer(ptr), Value::Int(-1234)]).unwrap();
        let val = builtin_mem_read_i16(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, -1234), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_i32() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 4);
        builtin_mem_write_i32(&mut env, vec![Value::Pointer(ptr), Value::Int(-70000)]).unwrap();
        let val = builtin_mem_read_i32(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, -70000), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_i64() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 8);
        let neg: i128 = -(1 << 40);
        builtin_mem_write_i64(&mut env, vec![Value::Pointer(ptr), Value::Int(neg)]).unwrap();
        let val = builtin_mem_read_i64(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, neg), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_f32() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 4);
        builtin_mem_write_f32(&mut env, vec![Value::Pointer(ptr), Value::Float(3.14)]).unwrap();
        let val = builtin_mem_read_f32(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val {
            Value::Float(f) => assert!((f - 3.14).abs() < 0.001),
            _ => panic!("expected Float"),
        }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_read_write_f64() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 8);
        builtin_mem_write_f64(&mut env, vec![Value::Pointer(ptr), Value::Float(2.718281828)]).unwrap();
        let val = builtin_mem_read_f64(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val {
            Value::Float(f) => assert!((f - 2.718281828).abs() < 1e-9),
            _ => panic!("expected Float"),
        }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_pointer_arithmetic() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 16);

        // Write values at offsets
        builtin_mem_write_u8(&mut env, vec![Value::Pointer(ptr), Value::Int(10)]).unwrap();

        let ptr1 = match builtin_ptr_add(&mut env, vec![Value::Pointer(ptr), Value::Int(4)]).unwrap() {
            Value::Pointer(p) => p,
            _ => panic!("expected Pointer"),
        };
        assert_eq!(ptr1, ptr + 4);

        builtin_mem_write_u32(&mut env, vec![Value::Pointer(ptr1), Value::Int(99999)]).unwrap();
        let val = builtin_mem_read_u32(&mut env, vec![Value::Pointer(ptr1)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 99999), other => panic!("expected Int, got {:?}", other) }

        // ptr_to_int / int_to_ptr roundtrip
        let int_val = builtin_ptr_to_int(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        let ptr_back = builtin_int_to_ptr(&mut env, vec![int_val]).unwrap();
        match ptr_back { Value::Pointer(v) => assert_eq!(v, ptr), other => panic!("expected Pointer, got {:?}", other) }

        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_mem_copy() {
        let mut env = test_env();
        let src = alloc(&mut env, 4);
        let dst = alloc(&mut env, 4);

        builtin_mem_write_u32(&mut env, vec![Value::Pointer(src), Value::Int(0xDEADBEEF)]).unwrap();
        builtin_mem_copy(&mut env, vec![Value::Pointer(dst), Value::Pointer(src), Value::Int(4)]).unwrap();
        let val = builtin_mem_read_u32(&mut env, vec![Value::Pointer(dst)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 0xDEADBEEF), other => panic!("expected Int, got {:?}", other) }

        builtin_mem_free(&mut env, vec![Value::Pointer(src)]).unwrap();
        builtin_mem_free(&mut env, vec![Value::Pointer(dst)]).unwrap();
    }

    #[test]
    fn test_mem_read_write_string() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 64);
        let msg = "Hello, Vortex!";

        builtin_mem_write_string(&mut env, vec![
            Value::Pointer(ptr),
            Value::String(msg.to_string()),
        ]).unwrap();

        let val = builtin_mem_read_string(&mut env, vec![
            Value::Pointer(ptr),
            Value::Int(msg.len() as i128),
        ]).unwrap();
        match val { Value::String(s) => assert_eq!(s, msg), other => panic!("expected String, got {:?}", other) }

        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_mem_read_write_bytes() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 4);
        let bytes = vec![Value::Int(1), Value::Int(2), Value::Int(3), Value::Int(4)];

        builtin_mem_write_bytes(&mut env, vec![Value::Pointer(ptr), Value::Array(bytes.clone())]).unwrap();
        let result = builtin_mem_read_bytes(&mut env, vec![Value::Pointer(ptr), Value::Int(4)]).unwrap();
        match result {
            Value::Array(arr) => {
                assert_eq!(arr.len(), 4);
                for (i, v) in arr.iter().enumerate() {
                    match v { Value::Int(val) => assert_eq!(*val, (i + 1) as i128), other => panic!("expected Int, got {:?}", other) }
                }
            }
            other => panic!("expected Array, got {:?}", other),
        }

        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_mem_zero() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 4);
        builtin_mem_write_u32(&mut env, vec![Value::Pointer(ptr), Value::Int(0xFFFFFFFF)]).unwrap();
        builtin_mem_zero(&mut env, vec![Value::Pointer(ptr), Value::Int(4)]).unwrap();
        let val = builtin_mem_read_u32(&mut env, vec![Value::Pointer(ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 0), other => panic!("expected Int, got {:?}", other) }
        builtin_mem_free(&mut env, vec![Value::Pointer(ptr)]).unwrap();
    }

    #[test]
    fn test_mem_realloc() {
        let mut env = test_env();
        let ptr = alloc(&mut env, 8);
        builtin_mem_write_u32(&mut env, vec![Value::Pointer(ptr), Value::Int(42)]).unwrap();

        let new_ptr = match builtin_mem_realloc(&mut env, vec![Value::Pointer(ptr), Value::Int(64)]).unwrap() {
            Value::Pointer(p) => p,
            _ => panic!("expected Pointer"),
        };
        // Original data should be preserved
        let val = builtin_mem_read_u32(&mut env, vec![Value::Pointer(new_ptr)]).unwrap();
        match val { Value::Int(v) => assert_eq!(v, 42), other => panic!("expected Int, got {:?}", other) }

        assert!(ALLOCATIONS.lock().unwrap().contains_key(&new_ptr));
        assert!(!ALLOCATIONS.lock().unwrap().contains_key(&ptr) || ptr == new_ptr);

        builtin_mem_free(&mut env, vec![Value::Pointer(new_ptr)]).unwrap();
    }

    #[test]
    fn test_mem_size_of() {
        let mut env = test_env();
        match builtin_mem_size_of(&mut env, vec![Value::String("u8".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 1), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("u16".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 2), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("u32".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 4), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("u64".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 8), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("f32".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 4), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("f64".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 8), other => panic!("expected Int, got {:?}", other) }
        match builtin_mem_size_of(&mut env, vec![Value::String("ptr".into())]).unwrap() { Value::Int(v) => assert_eq!(v, 8), other => panic!("expected Int, got {:?}", other) }
        assert!(builtin_mem_size_of(&mut env, vec![Value::String("foo".into())]).is_err());
    }
}
