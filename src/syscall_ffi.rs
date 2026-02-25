// syscall_ffi.rs — Raw Linux syscalls + FFI module for Vortex
//
// Provides direct access to Linux system calls and dynamic library loading
// (dlopen/dlsym/dlclose) from Vortex programs. All pointers are represented
// Pointer-returning builtins use Value::Pointer(usize).

use crate::interpreter::{Value, Env, FnDef};
use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};

// ---------------------------------------------------------------------------
// extern "C" declarations
// ---------------------------------------------------------------------------

extern "C" {
    fn syscall(num: i64, ...) -> i64;
    fn dlopen(filename: *const u8, flags: i32) -> *mut std::ffi::c_void;
    fn dlsym(handle: *mut std::ffi::c_void, symbol: *const u8) -> *mut std::ffi::c_void;
    fn dlclose(handle: *mut std::ffi::c_void) -> i32;
    fn dlerror() -> *const u8;
}

// ---------------------------------------------------------------------------
// Global state — track dlopen handles
// ---------------------------------------------------------------------------

static FFI_HANDLES: LazyLock<Mutex<HashMap<usize, String>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

// ---------------------------------------------------------------------------
// Helper: extract i64 from Value::Int
// ---------------------------------------------------------------------------

fn val_to_i64(v: &Value) -> Result<i64, String> {
    match v {
        Value::Int(n) => Ok(*n as i64),
        Value::Pointer(p) => Ok(*p as i64),
        _ => Err(format!("expected Int, got {:?}", v)),
    }
}

fn val_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Pointer(p) => Ok(*p),
        _ => Err(format!("expected Int or Pointer, got {:?}", v)),
    }
}

fn val_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(format!("expected String, got {:?}", v)),
    }
}

fn val_to_int_array(v: &Value) -> Result<Vec<i64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(val_to_i64).collect(),
        _ => Err(format!("expected Array of Int, got {:?}", v)),
    }
}

// ---------------------------------------------------------------------------
// Syscall builtins
// ---------------------------------------------------------------------------

/// Generic syscall: syscall(num, [arg1, arg2, ...])
fn builtin_syscall(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("syscall: expected at least 1 argument (syscall number)".into());
    }
    let num = val_to_i64(&args[0])?;

    // Remaining args can be individual Ints or a single Array of Ints
    let sargs: Vec<i64> = if args.len() == 2 {
        match &args[1] {
            Value::Array(_) => val_to_int_array(&args[1])?,
            _ => args[1..].iter().map(val_to_i64).collect::<Result<Vec<_>, _>>()?,
        }
    } else {
        args[1..].iter().map(val_to_i64).collect::<Result<Vec<_>, _>>()?
    };

    if sargs.len() > 6 {
        return Err("syscall: at most 6 arguments supported".into());
    }

    let ret = unsafe {
        match sargs.len() {
            0 => syscall(num),
            1 => syscall(num, sargs[0]),
            2 => syscall(num, sargs[0], sargs[1]),
            3 => syscall(num, sargs[0], sargs[1], sargs[2]),
            4 => syscall(num, sargs[0], sargs[1], sargs[2], sargs[3]),
            5 => syscall(num, sargs[0], sargs[1], sargs[2], sargs[3], sargs[4]),
            6 => syscall(num, sargs[0], sargs[1], sargs[2], sargs[3], sargs[4], sargs[5]),
            _ => unreachable!(),
        }
    };
    Ok(Value::Int(ret as i128))
}

/// syscall0(num) -> Int
fn builtin_syscall0(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("syscall0: expected 1 argument".into());
    }
    let num = val_to_i64(&args[0])?;
    let ret = unsafe { syscall(num) };
    Ok(Value::Int(ret as i128))
}

/// syscall1(num, a1) -> Int
fn builtin_syscall1(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("syscall1: expected 2 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let ret = unsafe { syscall(num, a1) };
    Ok(Value::Int(ret as i128))
}

/// syscall2(num, a1, a2) -> Int
fn builtin_syscall2(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("syscall2: expected 3 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let a2 = val_to_i64(&args[2])?;
    let ret = unsafe { syscall(num, a1, a2) };
    Ok(Value::Int(ret as i128))
}

/// syscall3(num, a1, a2, a3) -> Int
fn builtin_syscall3(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("syscall3: expected 4 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let a2 = val_to_i64(&args[2])?;
    let a3 = val_to_i64(&args[3])?;
    let ret = unsafe { syscall(num, a1, a2, a3) };
    Ok(Value::Int(ret as i128))
}

/// syscall4(num, a1, a2, a3, a4) -> Int
fn builtin_syscall4(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 {
        return Err("syscall4: expected 5 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let a2 = val_to_i64(&args[2])?;
    let a3 = val_to_i64(&args[3])?;
    let a4 = val_to_i64(&args[4])?;
    let ret = unsafe { syscall(num, a1, a2, a3, a4) };
    Ok(Value::Int(ret as i128))
}

/// syscall5(num, a1, a2, a3, a4, a5) -> Int
fn builtin_syscall5(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 {
        return Err("syscall5: expected 6 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let a2 = val_to_i64(&args[2])?;
    let a3 = val_to_i64(&args[3])?;
    let a4 = val_to_i64(&args[4])?;
    let a5 = val_to_i64(&args[5])?;
    let ret = unsafe { syscall(num, a1, a2, a3, a4, a5) };
    Ok(Value::Int(ret as i128))
}

/// syscall6(num, a1, a2, a3, a4, a5, a6) -> Int
fn builtin_syscall6(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 7 {
        return Err("syscall6: expected 7 arguments".into());
    }
    let num = val_to_i64(&args[0])?;
    let a1 = val_to_i64(&args[1])?;
    let a2 = val_to_i64(&args[2])?;
    let a3 = val_to_i64(&args[3])?;
    let a4 = val_to_i64(&args[4])?;
    let a5 = val_to_i64(&args[5])?;
    let a6 = val_to_i64(&args[6])?;
    let ret = unsafe { syscall(num, a1, a2, a3, a4, a5, a6) };
    Ok(Value::Int(ret as i128))
}

// ---------------------------------------------------------------------------
// FFI builtins
// ---------------------------------------------------------------------------

/// ffi_open(path: String, flags?: Int) -> Int (pointer as int)
///
/// Opens a shared library. Pass "" or empty string to get the current process handle.
/// Default flags = RTLD_LAZY (1).
fn builtin_ffi_open(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() || args.len() > 2 {
        return Err("ffi_open: expected 1-2 arguments (path, flags?)".into());
    }
    let path = val_to_string(&args[0])?;
    let flags = if args.len() == 2 {
        val_to_i64(&args[1])? as i32
    } else {
        1 // RTLD_LAZY
    };

    let c_path: *const u8 = if path.is_empty() {
        std::ptr::null()
    } else {
        let mut bytes = path.as_bytes().to_vec();
        bytes.push(0); // null-terminate
        let ptr = bytes.as_ptr();
        std::mem::forget(bytes); // intentional leak for C interop
        ptr
    };

    let handle = unsafe { dlopen(c_path, flags) };
    if handle.is_null() {
        let err_msg = unsafe {
            let e = dlerror();
            if e.is_null() {
                "unknown dlopen error".to_string()
            } else {
                let mut len = 0;
                let mut p = e;
                while *p != 0 {
                    len += 1;
                    p = p.add(1);
                }
                String::from_utf8_lossy(std::slice::from_raw_parts(e, len)).to_string()
            }
        };
        return Err(format!("ffi_open: {}", err_msg));
    }

    let addr = handle as usize;
    FFI_HANDLES
        .lock()
        .unwrap()
        .insert(addr, path);

    Ok(Value::Pointer(addr))
}

/// ffi_sym(handle: Pointer, name: String) -> Pointer (function pointer)
fn builtin_ffi_sym(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ffi_sym: expected 2 arguments (handle, name)".into());
    }
    let handle_addr = val_to_usize(&args[0])?;
    let name = val_to_string(&args[1])?;

    let handle = handle_addr as *mut std::ffi::c_void;

    // Null-terminate the symbol name
    let mut name_bytes = name.as_bytes().to_vec();
    name_bytes.push(0);

    // Clear previous errors
    unsafe { dlerror(); }

    let sym = unsafe { dlsym(handle, name_bytes.as_ptr()) };

    // Check for error (dlsym can legitimately return NULL)
    let err = unsafe { dlerror() };
    if !err.is_null() {
        let err_msg = unsafe {
            let mut len = 0;
            let mut p = err;
            while *p != 0 {
                len += 1;
                p = p.add(1);
            }
            String::from_utf8_lossy(std::slice::from_raw_parts(err, len)).to_string()
        };
        return Err(format!("ffi_sym: {}", err_msg));
    }

    Ok(Value::Pointer(sym as usize))
}

/// ffi_close(handle: Int) -> Int
fn builtin_ffi_close(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("ffi_close: expected 1 argument (handle)".into());
    }
    let handle_addr = val_to_usize(&args[0])?;
    let handle = handle_addr as *mut std::ffi::c_void;

    let ret = unsafe { dlclose(handle) };

    FFI_HANDLES.lock().unwrap().remove(&handle_addr);

    Ok(Value::Int(ret as i128))
}

/// ffi_call(fn_ptr: Int, args: Array[Int]) -> Int
///
/// Call a C function pointer with up to 8 i64 arguments, returning i64.
fn builtin_ffi_call(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ffi_call: expected 2 arguments (fn_ptr, args_array)".into());
    }
    let fn_addr = val_to_usize(&args[0])?;
    let call_args = val_to_int_array(&args[1])?;

    if call_args.len() > 8 {
        return Err("ffi_call: at most 8 arguments supported".into());
    }
    if fn_addr == 0 {
        return Err("ffi_call: null function pointer".into());
    }

    let ret: i64 = unsafe {
        match call_args.len() {
            0 => {
                let f: extern "C" fn() -> i64 = std::mem::transmute(fn_addr);
                f()
            }
            1 => {
                let f: extern "C" fn(i64) -> i64 = std::mem::transmute(fn_addr);
                f(call_args[0])
            }
            2 => {
                let f: extern "C" fn(i64, i64) -> i64 = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1])
            }
            3 => {
                let f: extern "C" fn(i64, i64, i64) -> i64 = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2])
            }
            4 => {
                let f: extern "C" fn(i64, i64, i64, i64) -> i64 = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2], call_args[3])
            }
            5 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2], call_args[3], call_args[4])
            }
            6 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                )
            }
            7 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                    call_args[6],
                )
            }
            8 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64 =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                    call_args[6], call_args[7],
                )
            }
            _ => unreachable!(),
        }
    };

    Ok(Value::Int(ret as i128))
}

/// ffi_call_ptr(fn_ptr: Int, args: Array[Int]) -> Int (pointer result)
///
/// Same as ffi_call but the return value is treated as a pointer (usize).
fn builtin_ffi_call_ptr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ffi_call_ptr: expected 2 arguments (fn_ptr, args_array)".into());
    }
    let fn_addr = val_to_usize(&args[0])?;
    let call_args = val_to_int_array(&args[1])?;

    if call_args.len() > 8 {
        return Err("ffi_call_ptr: at most 8 arguments supported".into());
    }
    if fn_addr == 0 {
        return Err("ffi_call_ptr: null function pointer".into());
    }

    let ret: usize = unsafe {
        match call_args.len() {
            0 => {
                let f: extern "C" fn() -> usize = std::mem::transmute(fn_addr);
                f()
            }
            1 => {
                let f: extern "C" fn(i64) -> usize = std::mem::transmute(fn_addr);
                f(call_args[0])
            }
            2 => {
                let f: extern "C" fn(i64, i64) -> usize = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1])
            }
            3 => {
                let f: extern "C" fn(i64, i64, i64) -> usize = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2])
            }
            4 => {
                let f: extern "C" fn(i64, i64, i64, i64) -> usize = std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2], call_args[3])
            }
            5 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64) -> usize =
                    std::mem::transmute(fn_addr);
                f(call_args[0], call_args[1], call_args[2], call_args[3], call_args[4])
            }
            6 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64) -> usize =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                )
            }
            7 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> usize =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                    call_args[6],
                )
            }
            8 => {
                let f: extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> usize =
                    std::mem::transmute(fn_addr);
                f(
                    call_args[0], call_args[1], call_args[2],
                    call_args[3], call_args[4], call_args[5],
                    call_args[6], call_args[7],
                )
            }
            _ => unreachable!(),
        }
    };

    Ok(Value::Pointer(ret))
}

// ---------------------------------------------------------------------------
// Utility builtins
// ---------------------------------------------------------------------------

/// ffi_handles() -> Array — list currently open FFI handle addresses
fn builtin_ffi_handles(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let handles = FFI_HANDLES.lock().unwrap();
    let entries: Vec<Value> = handles
        .iter()
        .map(|(addr, path)| {
            Value::Array(vec![
                Value::Pointer(*addr),
                Value::String(path.clone()),
            ])
        })
        .collect();
    Ok(Value::Array(entries))
}

/// ptr_to_string(ptr: Int, len: Int) -> String — read len bytes from a raw pointer
fn builtin_ptr_to_string(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ptr_to_string: expected 2 arguments (ptr, len)".into());
    }
    let ptr = val_to_usize(&args[0])?;
    let len = val_to_usize(&args[1])?;

    if ptr == 0 {
        return Err("ptr_to_string: null pointer".into());
    }
    if len > 1_048_576 {
        return Err("ptr_to_string: length exceeds 1MB safety limit".into());
    }

    let s = unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const u8, len);
        String::from_utf8_lossy(slice).to_string()
    };
    Ok(Value::String(s))
}

/// ptr_to_cstring(ptr: Int) -> String — read a null-terminated C string
fn builtin_ptr_to_cstring(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("ptr_to_cstring: expected 1 argument (ptr)".into());
    }
    let ptr = val_to_usize(&args[0])?;
    if ptr == 0 {
        return Err("ptr_to_cstring: null pointer".into());
    }

    let s = unsafe {
        let cstr = std::ffi::CStr::from_ptr(ptr as *const std::ffi::c_char);
        cstr.to_string_lossy().to_string()
    };
    Ok(Value::String(s))
}

/// ptr_read_i64(ptr: Int, offset: Int) -> Int — read an i64 at ptr + offset*8
fn builtin_ptr_read_i64(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ptr_read_i64: expected 2 arguments (ptr, offset)".into());
    }
    let ptr = val_to_usize(&args[0])?;
    let offset = val_to_i64(&args[1])? as usize;

    if ptr == 0 {
        return Err("ptr_read_i64: null pointer".into());
    }

    let val = unsafe {
        let p = (ptr as *const i64).add(offset);
        *p
    };
    Ok(Value::Int(val as i128))
}

/// ptr_write_i64(ptr: Int, offset: Int, value: Int) -> Void
fn builtin_ptr_write_i64(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("ptr_write_i64: expected 3 arguments (ptr, offset, value)".into());
    }
    let ptr = val_to_usize(&args[0])?;
    let offset = val_to_i64(&args[1])? as usize;
    let value = val_to_i64(&args[2])?;

    if ptr == 0 {
        return Err("ptr_write_i64: null pointer".into());
    }

    unsafe {
        let p = (ptr as *mut i64).add(offset);
        *p = value;
    }
    Ok(Value::Void)
}

/// alloc_bytes(size: Int) -> Int — allocate a buffer, return pointer as int
fn builtin_alloc_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("alloc_bytes: expected 1 argument (size)".into());
    }
    let size = val_to_usize(&args[0])?;
    if size == 0 || size > 1_073_741_824 {
        return Err("alloc_bytes: size must be 1..1GB".into());
    }

    let layout = std::alloc::Layout::from_size_align(size, 8)
        .map_err(|e| format!("alloc_bytes: {}", e))?;
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    if ptr.is_null() {
        return Err("alloc_bytes: allocation failed".into());
    }
    Ok(Value::Pointer(ptr as usize))
}

/// free_bytes(ptr: Pointer, size: Int) -> Void — free a previously allocated buffer
fn builtin_free_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("free_bytes: expected 2 arguments (ptr, size)".into());
    }
    let ptr = val_to_usize(&args[0])?;
    let size = val_to_usize(&args[1])?;

    if ptr == 0 {
        return Err("free_bytes: null pointer".into());
    }

    let layout = std::alloc::Layout::from_size_align(size, 8)
        .map_err(|e| format!("free_bytes: {}", e))?;
    unsafe { std::alloc::dealloc(ptr as *mut u8, layout); }
    Ok(Value::Void)
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    // Syscall builtins
    env.functions.insert("syscall".to_string(), FnDef::Builtin(builtin_syscall));
    env.functions.insert("syscall0".to_string(), FnDef::Builtin(builtin_syscall0));
    env.functions.insert("syscall1".to_string(), FnDef::Builtin(builtin_syscall1));
    env.functions.insert("syscall2".to_string(), FnDef::Builtin(builtin_syscall2));
    env.functions.insert("syscall3".to_string(), FnDef::Builtin(builtin_syscall3));
    env.functions.insert("syscall4".to_string(), FnDef::Builtin(builtin_syscall4));
    env.functions.insert("syscall5".to_string(), FnDef::Builtin(builtin_syscall5));
    env.functions.insert("syscall6".to_string(), FnDef::Builtin(builtin_syscall6));

    // FFI builtins
    env.functions.insert("ffi_open".to_string(), FnDef::Builtin(builtin_ffi_open));
    env.functions.insert("ffi_sym".to_string(), FnDef::Builtin(builtin_ffi_sym));
    env.functions.insert("ffi_close".to_string(), FnDef::Builtin(builtin_ffi_close));
    env.functions.insert("ffi_call".to_string(), FnDef::Builtin(builtin_ffi_call));
    env.functions.insert("ffi_call_ptr".to_string(), FnDef::Builtin(builtin_ffi_call_ptr));

    // Utility builtins
    env.functions.insert("ffi_handles".to_string(), FnDef::Builtin(builtin_ffi_handles));
    env.functions.insert("ptr_to_string".to_string(), FnDef::Builtin(builtin_ptr_to_string));
    env.functions.insert("ptr_to_cstring".to_string(), FnDef::Builtin(builtin_ptr_to_cstring));
    env.functions.insert("ptr_read_i64".to_string(), FnDef::Builtin(builtin_ptr_read_i64));
    env.functions.insert("ptr_write_i64".to_string(), FnDef::Builtin(builtin_ptr_write_i64));
    env.functions.insert("alloc_bytes".to_string(), FnDef::Builtin(builtin_alloc_bytes));
    env.functions.insert("free_bytes".to_string(), FnDef::Builtin(builtin_free_bytes));

    // -----------------------------------------------------------------------
    // Syscall number constants (x86_64 Linux)
    // -----------------------------------------------------------------------
    let syscall_constants: &[(&str, i128)] = &[
        ("SYS_READ", 0),
        ("SYS_WRITE", 1),
        ("SYS_OPEN", 2),
        ("SYS_CLOSE", 3),
        ("SYS_STAT", 4),
        ("SYS_FSTAT", 5),
        ("SYS_LSEEK", 8),
        ("SYS_MMAP", 9),
        ("SYS_MPROTECT", 10),
        ("SYS_MUNMAP", 11),
        ("SYS_BRK", 12),
        ("SYS_IOCTL", 16),
        ("SYS_PIPE", 22),
        ("SYS_SELECT", 23),
        ("SYS_GETPID", 39),
        ("SYS_SOCKET", 41),
        ("SYS_CONNECT", 42),
        ("SYS_ACCEPT", 43),
        ("SYS_SENDTO", 44),
        ("SYS_RECVFROM", 45),
        ("SYS_BIND", 49),
        ("SYS_LISTEN", 50),
        ("SYS_FORK", 57),
        ("SYS_EXECVE", 59),
        ("SYS_EXIT", 60),
        ("SYS_WAIT4", 61),
        ("SYS_KILL", 62),
        ("SYS_GETUID", 102),
        ("SYS_GETGID", 104),
        ("SYS_GETTID", 186),
        ("SYS_CLOCK_GETTIME", 228),
        ("SYS_EPOLL_WAIT", 232),
        ("SYS_EPOLL_CTL", 233),
        ("SYS_EPOLL_CREATE1", 291),
    ];

    for (name, val) in syscall_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // File open flags
    // -----------------------------------------------------------------------
    let file_constants: &[(&str, i128)] = &[
        ("O_RDONLY", 0),
        ("O_WRONLY", 1),
        ("O_RDWR", 2),
        ("O_CREAT", 64),
        ("O_TRUNC", 512),
        ("O_APPEND", 1024),
        ("O_NONBLOCK", 2048),
    ];

    for (name, val) in file_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // Memory protection flags
    // -----------------------------------------------------------------------
    let prot_constants: &[(&str, i128)] = &[
        ("PROT_NONE", 0),
        ("PROT_READ", 1),
        ("PROT_WRITE", 2),
        ("PROT_EXEC", 4),
    ];

    for (name, val) in prot_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // mmap flags
    // -----------------------------------------------------------------------
    let map_constants: &[(&str, i128)] = &[
        ("MAP_SHARED", 1),
        ("MAP_PRIVATE", 2),
        ("MAP_FIXED", 16),
        ("MAP_ANONYMOUS", 32),
    ];

    for (name, val) in map_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // dlopen flags
    // -----------------------------------------------------------------------
    let rtld_constants: &[(&str, i128)] = &[
        ("RTLD_LAZY", 1),
        ("RTLD_NOW", 2),
        ("RTLD_GLOBAL", 256),
    ];

    for (name, val) in rtld_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // Socket constants
    // -----------------------------------------------------------------------
    let socket_constants: &[(&str, i128)] = &[
        ("AF_UNIX", 1),
        ("AF_INET", 2),
        ("AF_INET6", 10),
        ("SOCK_STREAM", 1),
        ("SOCK_DGRAM", 2),
        ("SOCK_RAW", 3),
    ];

    for (name, val) in socket_constants {
        env.define(name, Value::Int(*val));
    }

    // -----------------------------------------------------------------------
    // Standard file descriptors
    // -----------------------------------------------------------------------
    env.define("STDIN_FILENO", Value::Int(0));
    env.define("STDOUT_FILENO", Value::Int(1));
    env.define("STDERR_FILENO", Value::Int(2));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a minimal Env for testing
    fn test_env() -> Env {
        let mut env = Env::new();
        register_builtins(&mut env);
        env
    }

    #[test]
    fn test_syscall_getpid() {
        let mut env = test_env();
        // SYS_GETPID = 39
        let result = builtin_syscall0(&mut env, vec![Value::Int(39)]).unwrap();
        match result {
            Value::Int(pid) => assert!(pid > 0, "getpid should return > 0, got {}", pid),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn test_syscall_getpid_generic() {
        let mut env = test_env();
        let result = builtin_syscall(&mut env, vec![Value::Int(39)]).unwrap();
        match result {
            Value::Int(pid) => assert!(pid > 0),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn test_syscall_clock_gettime() {
        let mut env = test_env();
        // Allocate a buffer for struct timespec (16 bytes)
        let buf = builtin_alloc_bytes(&mut env, vec![Value::Int(16)]).unwrap();
        let buf_addr = match buf {
            Value::Pointer(a) => a as i128,
            _ => panic!("expected Pointer"),
        };

        // syscall2(SYS_CLOCK_GETTIME=228, CLOCK_REALTIME=0, buf)
        let result = builtin_syscall2(
            &mut env,
            vec![Value::Int(228), Value::Int(0), Value::Int(buf_addr)],
        )
        .unwrap();

        match result {
            Value::Int(ret) => assert_eq!(ret, 0, "clock_gettime should return 0"),
            _ => panic!("expected Int"),
        }

        // Read seconds from the buffer
        let secs = builtin_ptr_read_i64(&mut env, vec![Value::Int(buf_addr), Value::Int(0)])
            .unwrap();
        match secs {
            Value::Int(s) => assert!(s > 1_000_000_000, "timestamp should be recent, got {}", s),
            _ => panic!("expected Int"),
        }

        // Clean up
        let _ = builtin_free_bytes(&mut env, vec![Value::Int(buf_addr), Value::Int(16)]);
    }

    #[test]
    fn test_ffi_open_close_current_process() {
        let mut env = test_env();

        // dlopen(NULL) returns handle to current process
        let handle = builtin_ffi_open(&mut env, vec![Value::String("".into())]).unwrap();
        let addr = match handle {
            Value::Pointer(a) => {
                assert!(a != 0, "handle should not be null");
                a
            }
            _ => panic!("expected Pointer"),
        };

        // Look up a known symbol: "printf"
        let sym = builtin_ffi_sym(
            &mut env,
            vec![Value::Pointer(addr), Value::String("printf".into())],
        )
        .unwrap();
        match sym {
            Value::Pointer(s) => assert!(s != 0, "printf symbol should not be null"),
            _ => panic!("expected Pointer"),
        }

        // Close
        let ret = builtin_ffi_close(&mut env, vec![Value::Pointer(addr)]).unwrap();
        match ret {
            Value::Int(r) => assert_eq!(r, 0),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn test_ffi_open_libc() {
        let mut env = test_env();

        // Try libc.so.6
        let handle = builtin_ffi_open(
            &mut env,
            vec![Value::String("libc.so.6".into())],
        );

        // This may fail in some environments; skip if so
        if let Ok(Value::Pointer(addr)) = handle {
            assert!(addr != 0);

            let sym = builtin_ffi_sym(
                &mut env,
                vec![Value::Pointer(addr), Value::String("getpid".into())],
            )
            .unwrap();

            match sym {
                Value::Pointer(s) => assert!(s != 0),
                _ => panic!("expected Pointer"),
            }

            let _ = builtin_ffi_close(&mut env, vec![Value::Pointer(addr)]);
        }
    }

    #[test]
    fn test_constants_registered() {
        let env = test_env();

        // Verify a few constants via env.get
        let checks: &[(&str, i128)] = &[
            ("SYS_READ", 0),
            ("SYS_WRITE", 1),
            ("SYS_GETPID", 39),
            ("SYS_EXIT", 60),
            ("O_RDONLY", 0),
            ("O_CREAT", 64),
            ("PROT_READ", 1),
            ("MAP_PRIVATE", 2),
            ("MAP_ANONYMOUS", 32),
            ("RTLD_LAZY", 1),
            ("AF_INET", 2),
            ("SOCK_STREAM", 1),
            ("STDIN_FILENO", 0),
            ("STDOUT_FILENO", 1),
            ("STDERR_FILENO", 2),
        ];

        for (name, expected) in checks {
            match env.get(name) {
                Some(Value::Int(v)) => {
                    assert_eq!(v, *expected, "constant {} should be {}, got {}", name, expected, v);
                }
                other => panic!("constant {} not found or wrong type: {:?}", name, other),
            }
        }
    }

    #[test]
    fn test_syscall_generic_with_array() {
        let mut env = test_env();
        // SYS_GETPID via generic syscall with empty array
        let result = builtin_syscall(
            &mut env,
            vec![Value::Int(39), Value::Array(vec![])],
        )
        .unwrap();
        match result {
            Value::Int(pid) => assert!(pid > 0),
            _ => panic!("expected Int"),
        }
    }

    #[test]
    fn test_alloc_and_read_write() {
        let mut env = test_env();

        let buf = builtin_alloc_bytes(&mut env, vec![Value::Int(64)]).unwrap();
        let addr = match buf {
            Value::Pointer(a) => a as i128,
            _ => panic!("expected Pointer"),
        };

        // Write a value
        builtin_ptr_write_i64(
            &mut env,
            vec![Value::Int(addr), Value::Int(0), Value::Int(42)],
        )
        .unwrap();

        // Read it back
        let val = builtin_ptr_read_i64(&mut env, vec![Value::Int(addr), Value::Int(0)]).unwrap();
        match val {
            Value::Int(v) => assert_eq!(v, 42),
            _ => panic!("expected Int"),
        }

        // Write at offset 1
        builtin_ptr_write_i64(
            &mut env,
            vec![Value::Int(addr), Value::Int(1), Value::Int(99)],
        )
        .unwrap();

        let val2 = builtin_ptr_read_i64(&mut env, vec![Value::Int(addr), Value::Int(1)]).unwrap();
        match val2 {
            Value::Int(v) => assert_eq!(v, 99),
            _ => panic!("expected Int"),
        }

        let _ = builtin_free_bytes(&mut env, vec![Value::Int(addr), Value::Int(64)]);
    }

    #[test]
    fn test_ffi_handles_tracking() {
        let mut env = test_env();

        let handle = builtin_ffi_open(&mut env, vec![Value::String("".into())]).unwrap();
        let addr = match handle {
            Value::Pointer(a) => a,
            _ => panic!("expected Pointer"),
        };

        // Check handles list
        let handles = builtin_ffi_handles(&mut env, vec![]).unwrap();
        match handles {
            Value::Array(arr) => {
                assert!(!arr.is_empty(), "should have at least one handle");
            }
            _ => panic!("expected Array"),
        }

        let _ = builtin_ffi_close(&mut env, vec![Value::Pointer(addr)]);
    }
}
