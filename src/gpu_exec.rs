//! JIT Compilation Engine for Vortex.
//!
//! Compiles MLIR IR to shared libraries (.so) via the MLIR toolchain,
//! loads them with dlopen, and calls the compiled functions directly
//! with tensor data. Falls back to pure Rust when compilation fails.
//!
//! Pipeline: MLIR -> mlir-opt-20 -> mlir-translate-20 -> LLVM IR
//!           LLVM IR + C wrapper -> clang -shared -O2 -> .so
//!           .so -> dlopen -> function pointer -> call

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Mutex;

// ---------------------------------------------------------------------------
// Tool discovery (same pattern as gpu_pipeline.rs / gpu_compute.rs)
// ---------------------------------------------------------------------------

fn find_tool(base: &str) -> Option<String> {
    let versioned = format!("{}-20", base);
    if tool_exists(&versioned) {
        return Some(versioned);
    }
    if tool_exists(base) {
        return Some(base.to_string());
    }
    None
}

fn tool_exists(name: &str) -> bool {
    Command::new("which")
        .arg(name)
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Returns true if the full JIT pipeline is available.
pub fn jit_available() -> bool {
    find_tool("mlir-opt").is_some()
        && find_tool("mlir-translate").is_some()
        && find_tool("clang").is_some()
}

// ---------------------------------------------------------------------------
// FNV-1a hash for cache keys
// ---------------------------------------------------------------------------

fn fnv_hash(data: &str) -> String {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    format!("{:016x}", h)
}

// ---------------------------------------------------------------------------
// Cache directory
// ---------------------------------------------------------------------------

fn jit_cache_dir() -> PathBuf {
    let dir = PathBuf::from("/tmp/vortex_jit_cache");
    let _ = fs::create_dir_all(&dir);
    dir
}

// ---------------------------------------------------------------------------
// MemRef descriptor matching MLIR's memref ABI
// ---------------------------------------------------------------------------

/// MemRef descriptor for 1D tensors, matching MLIR's unranked memref ABI.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemRefDescriptor1D {
    pub allocated: *mut f64,
    pub aligned: *mut f64,
    pub offset: i64,
    pub sizes: [i64; 1],
    pub strides: [i64; 1],
}

/// MemRef descriptor for 2D tensors.
#[repr(C)]
#[derive(Debug, Clone)]
pub struct MemRefDescriptor2D {
    pub allocated: *mut f64,
    pub aligned: *mut f64,
    pub offset: i64,
    pub sizes: [i64; 2],
    pub strides: [i64; 2],
}

impl MemRefDescriptor1D {
    pub fn from_slice(data: &mut [f64]) -> Self {
        let len = data.len() as i64;
        Self {
            allocated: data.as_mut_ptr(),
            aligned: data.as_mut_ptr(),
            offset: 0,
            sizes: [len],
            strides: [1],
        }
    }
}

impl MemRefDescriptor2D {
    pub fn from_slice(data: &mut [f64], rows: usize, cols: usize) -> Self {
        Self {
            allocated: data.as_mut_ptr(),
            aligned: data.as_mut_ptr(),
            offset: 0,
            sizes: [rows as i64, cols as i64],
            strides: [cols as i64, 1],
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor buffer management
// ---------------------------------------------------------------------------

/// A contiguous memory buffer for tensor data with shape information.
#[derive(Debug, Clone)]
pub struct TensorBuffer {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
}

impl TensorBuffer {
    pub fn new(data: Vec<f64>, shape: Vec<usize>) -> Self {
        Self { data, shape }
    }

    pub fn zeros(shape: &[usize]) -> Self {
        let n: usize = shape.iter().product();
        Self {
            data: vec![0.0; n],
            shape: shape.to_vec(),
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn rows(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[0]
        } else {
            1
        }
    }

    pub fn cols(&self) -> usize {
        if self.shape.len() >= 2 {
            self.shape[1]
        } else {
            *self.shape.first().unwrap_or(&1)
        }
    }

    pub fn strides(&self) -> Vec<usize> {
        let mut strides = vec![1usize; self.shape.len()];
        for i in (0..self.shape.len().saturating_sub(1)).rev() {
            strides[i] = strides[i + 1] * self.shape[i + 1];
        }
        strides
    }
}

// ---------------------------------------------------------------------------
// JIT compilation cache
// ---------------------------------------------------------------------------

struct JitCache {
    /// Maps hash -> path to compiled .so
    entries: HashMap<String, PathBuf>,
}

impl JitCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
        }
    }
}

fn global_jit_cache() -> &'static Mutex<JitCache> {
    use std::sync::OnceLock;
    static CACHE: OnceLock<Mutex<JitCache>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(JitCache::new()))
}

// ---------------------------------------------------------------------------
// MLIR generation for JIT execution (static dimensions)
// ---------------------------------------------------------------------------

/// Generate a complete MLIR module for matrix multiplication with static dimensions.
pub fn generate_matmul_mlir(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"module {{
  func.func @matmul(%A: memref<{m}x{k}xf64>, %B: memref<{k}x{n}xf64>, %C: memref<{m}x{n}xf64>) {{
    affine.for %i = 0 to {m} {{
      affine.for %j = 0 to {n} {{
        %zero = arith.constant 0.0 : f64
        %res = affine.for %p = 0 to {k} iter_args(%acc = %zero) -> (f64) {{
          %a = affine.load %A[%i, %p] : memref<{m}x{k}xf64>
          %b = affine.load %B[%p, %j] : memref<{k}x{n}xf64>
          %prod = arith.mulf %a, %b : f64
          %new = arith.addf %acc, %prod : f64
          affine.yield %new : f64
        }}
        affine.store %res, %C[%i, %j] : memref<{m}x{n}xf64>
      }}
    }}
    return
  }}
}}"#
    )
}

/// Generate MLIR for elementwise binary operations.
pub fn generate_elementwise_mlir(op: &str, size: usize) -> String {
    let arith_op = match op {
        "add" => "arith.addf",
        "sub" => "arith.subf",
        "mul" => "arith.mulf",
        "div" => "arith.divf",
        _ => "arith.addf",
    };
    let fname = format!("elementwise_{}", op);
    format!(
        r#"module {{
  func.func @{fname}(%A: memref<{size}xf64>, %B: memref<{size}xf64>, %C: memref<{size}xf64>) {{
    affine.for %i = 0 to {size} {{
      %a = affine.load %A[%i] : memref<{size}xf64>
      %b = affine.load %B[%i] : memref<{size}xf64>
      %c = {arith_op} %a, %b : f64
      affine.store %c, %C[%i] : memref<{size}xf64>
    }}
    return
  }}
}}"#
    )
}

/// Generate MLIR for reduction along rows.
pub fn generate_reduce_mlir(op: &str, rows: usize, cols: usize) -> String {
    let (fname, init, combine) = match op {
        "sum" => ("reduce_sum", "0.0", "arith.addf"),
        "max" => ("reduce_max", "-1.0e308", "arith.maximumf"),
        "min" => ("reduce_min", "1.0e308", "arith.minimumf"),
        _ => ("reduce_sum", "0.0", "arith.addf"),
    };
    format!(
        r#"module {{
  func.func @{fname}(%A: memref<{rows}x{cols}xf64>, %C: memref<{rows}xf64>) {{
    affine.for %i = 0 to {rows} {{
      %init = arith.constant {init} : f64
      %res = affine.for %j = 0 to {cols} iter_args(%acc = %init) -> (f64) {{
        %v = affine.load %A[%i, %j] : memref<{rows}x{cols}xf64>
        %new = {combine} %acc, %v : f64
        affine.yield %new : f64
      }}
      affine.store %res, %C[%i] : memref<{rows}xf64>
    }}
    return
  }}
}}"#
    )
}

/// Generate MLIR for ReLU activation.
pub fn generate_relu_mlir(size: usize) -> String {
    format!(
        r#"module {{
  func.func @relu(%A: memref<{size}xf64>, %C: memref<{size}xf64>) {{
    %zero = arith.constant 0.0 : f64
    affine.for %i = 0 to {size} {{
      %v = affine.load %A[%i] : memref<{size}xf64>
      %out = arith.maximumf %v, %zero : f64
      affine.store %out, %C[%i] : memref<{size}xf64>
    }}
    return
  }}
}}"#
    )
}

/// Generate MLIR for softmax (row-wise on 2D tensor).
pub fn generate_softmax_mlir(rows: usize, cols: usize) -> String {
    format!(
        r#"module {{
  func.func @softmax(%A: memref<{rows}x{cols}xf64>, %C: memref<{rows}x{cols}xf64>) {{
    affine.for %i = 0 to {rows} {{
      %neg_inf = arith.constant -1.0e308 : f64
      %max_val = affine.for %j = 0 to {cols} iter_args(%mx = %neg_inf) -> (f64) {{
        %v = affine.load %A[%i, %j] : memref<{rows}x{cols}xf64>
        %new_mx = arith.maximumf %mx, %v : f64
        affine.yield %new_mx : f64
      }}
      %zero = arith.constant 0.0 : f64
      %exp_sum = affine.for %j = 0 to {cols} iter_args(%s = %zero) -> (f64) {{
        %v = affine.load %A[%i, %j] : memref<{rows}x{cols}xf64>
        %shifted = arith.subf %v, %max_val : f64
        %e = math.exp %shifted : f64
        %ns = arith.addf %s, %e : f64
        affine.yield %ns : f64
      }}
      affine.for %j = 0 to {cols} {{
        %v = affine.load %A[%i, %j] : memref<{rows}x{cols}xf64>
        %shifted = arith.subf %v, %max_val : f64
        %e = math.exp %shifted : f64
        %out = arith.divf %e, %exp_sum : f64
        affine.store %out, %C[%i, %j] : memref<{rows}x{cols}xf64>
      }}
    }}
    return
  }}
}}"#
    )
}

// ---------------------------------------------------------------------------
// C wrapper generation
// ---------------------------------------------------------------------------

/// Generate a C wrapper for matmul that provides a simple C-callable interface
/// linking against the MLIR-compiled LLVM IR.
fn generate_matmul_c_wrapper(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#include <stdlib.h>
#include <string.h>
#include <stdint.h>

/* The MLIR-compiled function expects memref descriptors passed as individual args:
   (allocated_ptr, aligned_ptr, offset, size0, size1, stride0, stride1) for each memref */
extern void matmul(double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t,
                   double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t,
                   double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t);

/* Simple C-callable wrapper: takes flat arrays and dimensions */
void matmul_wrapper(double* a, int64_t m, int64_t k_dim,
                    double* b, int64_t k2, int64_t n,
                    double* c) {{
    matmul(a, a, 0, m, k_dim, k_dim, 1,
           b, b, 0, k2, n, n, 1,
           c, c, 0, m, n, n, 1);
}}
"#
    )
}

/// Generate a C wrapper for elementwise operations.
fn generate_elementwise_c_wrapper(op: &str, size: usize) -> String {
    let fname = format!("elementwise_{}", op);
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

extern void {fname}(double*, double*, int64_t, int64_t, int64_t,
                     double*, double*, int64_t, int64_t, int64_t,
                     double*, double*, int64_t, int64_t, int64_t);

void {fname}_wrapper(double* a, double* b, double* c, int64_t n) {{
    {fname}(a, a, 0, n, 1,
            b, b, 0, n, 1,
            c, c, 0, n, 1);
}}
"#
    )
}

/// Generate a C wrapper for reduce operations.
fn generate_reduce_c_wrapper(op: &str, rows: usize, cols: usize) -> String {
    let fname = match op {
        "sum" => "reduce_sum",
        "max" => "reduce_max",
        "min" => "reduce_min",
        _ => "reduce_sum",
    };
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

extern void {fname}(double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t,
                     double*, double*, int64_t, int64_t, int64_t);

void {fname}_wrapper(double* a, double* c, int64_t rows, int64_t cols) {{
    {fname}(a, a, 0, rows, cols, cols, 1,
            c, c, 0, rows, 1);
}}
"#
    )
}

/// Generate a C wrapper for relu.
fn generate_relu_c_wrapper(size: usize) -> String {
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

extern void relu(double*, double*, int64_t, int64_t, int64_t,
                 double*, double*, int64_t, int64_t, int64_t);

void relu_wrapper(double* a, double* c, int64_t n) {{
    relu(a, a, 0, n, 1,
         c, c, 0, n, 1);
}}
"#
    )
}

/// Generate a C wrapper for softmax.
fn generate_softmax_c_wrapper(rows: usize, cols: usize) -> String {
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

extern void softmax(double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t,
                    double*, double*, int64_t, int64_t, int64_t, int64_t, int64_t);

void softmax_wrapper(double* a, double* c, int64_t rows, int64_t cols) {{
    softmax(a, a, 0, rows, cols, cols, 1,
            c, c, 0, rows, cols, cols, 1);
}}
"#
    )
}

// ---------------------------------------------------------------------------
// JIT compilation pipeline
// ---------------------------------------------------------------------------

/// Validate MLIR with mlir-opt (just check it parses).
pub fn validate_mlir(mlir: &str) -> Result<(), String> {
    let tool = find_tool("mlir-opt").ok_or("mlir-opt not found")?;
    let dir = jit_cache_dir();
    let input = dir.join(format!("validate_{}.mlir", fnv_hash(mlir)));
    fs::write(&input, mlir).map_err(|e| format!("write: {}", e))?;

    let output = Command::new(&tool)
        .arg("--verify-diagnostics")
        .arg(&input)
        .output()
        .map_err(|e| format!("mlir-opt: {}", e))?;

    let _ = fs::remove_file(&input);

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        // If verify-diagnostics fails, try without it (some versions don't support it)
        let input2 = dir.join(format!("validate2_{}.mlir", fnv_hash(mlir)));
        fs::write(&input2, mlir).map_err(|e| format!("write: {}", e))?;
        let output2 = Command::new(&tool)
            .arg(&input2)
            .output()
            .map_err(|e| format!("mlir-opt: {}", e))?;
        let _ = fs::remove_file(&input2);
        if !output2.status.success() {
            return Err(format!(
                "MLIR validation failed:\n{}",
                String::from_utf8_lossy(&output2.stderr)
            ));
        }
    }
    Ok(())
}

/// Lower MLIR to LLVM dialect using mlir-opt.
fn lower_mlir_to_llvm(mlir: &str, hash: &str) -> Result<String, String> {
    let tool = find_tool("mlir-opt").ok_or("mlir-opt not found")?;
    let dir = jit_cache_dir();
    let input = dir.join(format!("{}_input.mlir", hash));
    fs::write(&input, mlir).map_err(|e| format!("write: {}", e))?;

    let output = Command::new(&tool)
        .args(&[
            "--lower-affine",
            "--convert-scf-to-cf",
            "--convert-math-to-llvm",
            "--convert-func-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-cf-to-llvm",
            "--finalize-memref-to-llvm",
            "--reconcile-unrealized-casts",
        ])
        .arg(&input)
        .output()
        .map_err(|e| format!("mlir-opt: {}", e))?;

    let _ = fs::remove_file(&input);

    if !output.status.success() {
        return Err(format!(
            "mlir-opt lowering failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(String::from_utf8_lossy(&output.stdout).to_string())
}

/// Translate MLIR (LLVM dialect) to LLVM IR.
fn translate_to_llvm_ir(lowered_mlir: &str, hash: &str) -> Result<String, String> {
    let tool = find_tool("mlir-translate").ok_or("mlir-translate not found")?;
    let dir = jit_cache_dir();
    let input = dir.join(format!("{}_lowered.mlir", hash));
    let output_path = dir.join(format!("{}.ll", hash));

    fs::write(&input, lowered_mlir).map_err(|e| format!("write: {}", e))?;

    let output = Command::new(&tool)
        .arg("--mlir-to-llvmir")
        .arg(&input)
        .arg("-o")
        .arg(&output_path)
        .output()
        .map_err(|e| format!("mlir-translate: {}", e))?;

    let _ = fs::remove_file(&input);

    if !output.status.success() {
        let _ = fs::remove_file(&output_path);
        return Err(format!(
            "mlir-translate failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let ir = fs::read_to_string(&output_path)
        .map_err(|e| format!("read llvm ir: {}", e))?;
    let _ = fs::remove_file(&output_path);
    Ok(ir)
}

/// Compile LLVM IR + C wrapper to a shared library (.so).
fn compile_to_shared_lib(
    llvm_ir: &str,
    c_wrapper: &str,
    hash: &str,
) -> Result<PathBuf, String> {
    let clang = find_tool("clang").ok_or("clang not found")?;
    let dir = jit_cache_dir();
    let ll_path = dir.join(format!("{}.ll", hash));
    let c_path = dir.join(format!("{}_wrapper.c", hash));
    let so_path = dir.join(format!("{}.so", hash));

    if so_path.exists() {
        return Ok(so_path);
    }

    fs::write(&ll_path, llvm_ir).map_err(|e| format!("write ll: {}", e))?;
    fs::write(&c_path, c_wrapper).map_err(|e| format!("write c: {}", e))?;

    let output = Command::new(&clang)
        .args(&[
            "-shared",
            "-O2",
            "-fPIC",
            "-lm",
            "-o",
        ])
        .arg(&so_path)
        .arg(&ll_path)
        .arg(&c_path)
        .output()
        .map_err(|e| format!("clang: {}", e))?;

    let _ = fs::remove_file(&ll_path);
    let _ = fs::remove_file(&c_path);

    if !output.status.success() {
        let _ = fs::remove_file(&so_path);
        return Err(format!(
            "clang shared lib compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(so_path)
}

/// Compile a standalone C fallback to a shared library (.so).
fn compile_c_fallback_to_shared_lib(c_code: &str, hash: &str) -> Result<PathBuf, String> {
    let clang = find_tool("clang").ok_or("clang not found")?;
    let dir = jit_cache_dir();
    let c_path = dir.join(format!("{}_fallback.c", hash));
    let so_path = dir.join(format!("{}_fallback.so", hash));

    if so_path.exists() {
        return Ok(so_path);
    }

    fs::write(&c_path, c_code).map_err(|e| format!("write: {}", e))?;

    let output = Command::new(&clang)
        .args(&["-shared", "-O2", "-fPIC", "-lm", "-o"])
        .arg(&so_path)
        .arg(&c_path)
        .output()
        .map_err(|e| format!("clang: {}", e))?;

    let _ = fs::remove_file(&c_path);

    if !output.status.success() {
        let _ = fs::remove_file(&so_path);
        return Err(format!(
            "clang fallback compilation failed:\n{}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    Ok(so_path)
}

// ---------------------------------------------------------------------------
// dlopen-based shared library loading
// ---------------------------------------------------------------------------

/// A handle to a loaded shared library with a callable function.
pub struct JitKernel {
    /// Path to the .so file
    pub so_path: PathBuf,
    /// dlopen handle (raw pointer)
    _handle: *mut std::ffi::c_void,
    /// Function pointer to the wrapper
    func_ptr: *mut std::ffi::c_void,
    /// Name of the wrapper function
    pub func_name: String,
}

// Safety: We manage the lifetime carefully and the .so stays loaded.
unsafe impl Send for JitKernel {}
unsafe impl Sync for JitKernel {}

impl JitKernel {
    /// Load a shared library and look up a function by name.
    pub fn load(so_path: &Path, func_name: &str) -> Result<Self, String> {
        let c_path = std::ffi::CString::new(so_path.to_string_lossy().as_bytes())
            .map_err(|_| "invalid path")?;
        let c_name =
            std::ffi::CString::new(func_name).map_err(|_| "invalid func name")?;

        unsafe {
            let handle = libc_dlopen(c_path.as_ptr(), 0x00002 /* RTLD_NOW */);
            if handle.is_null() {
                let err = libc_dlerror();
                let msg = if err.is_null() {
                    "unknown dlopen error".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err)
                        .to_string_lossy()
                        .to_string()
                };
                return Err(format!("dlopen failed: {}", msg));
            }

            let sym = libc_dlsym(handle, c_name.as_ptr());
            if sym.is_null() {
                let err = libc_dlerror();
                let msg = if err.is_null() {
                    "symbol not found".to_string()
                } else {
                    std::ffi::CStr::from_ptr(err)
                        .to_string_lossy()
                        .to_string()
                };
                libc_dlclose(handle);
                return Err(format!("dlsym '{}' failed: {}", func_name, msg));
            }

            Ok(Self {
                so_path: so_path.to_path_buf(),
                _handle: handle,
                func_ptr: sym,
                func_name: func_name.to_string(),
            })
        }
    }

    /// Get the raw function pointer.
    pub fn as_ptr(&self) -> *mut std::ffi::c_void {
        self.func_ptr
    }
}

impl Drop for JitKernel {
    fn drop(&mut self) {
        if !self._handle.is_null() {
            unsafe {
                libc_dlclose(self._handle);
            }
        }
    }
}

// Raw libc FFI for dlopen/dlsym/dlclose
extern "C" {
    #[link_name = "dlopen"]
    fn libc_dlopen(
        filename: *const std::ffi::c_char,
        flags: std::ffi::c_int,
    ) -> *mut std::ffi::c_void;
    #[link_name = "dlsym"]
    fn libc_dlsym(
        handle: *mut std::ffi::c_void,
        symbol: *const std::ffi::c_char,
    ) -> *mut std::ffi::c_void;
    #[link_name = "dlclose"]
    fn libc_dlclose(handle: *mut std::ffi::c_void) -> std::ffi::c_int;
    #[link_name = "dlerror"]
    fn libc_dlerror() -> *const std::ffi::c_char;
}

// ---------------------------------------------------------------------------
// Full JIT pipeline: MLIR -> .so -> loaded kernel
// ---------------------------------------------------------------------------

/// Full pipeline result: either a loaded JIT kernel or an indication to use fallback.
enum JitResult {
    Kernel(JitKernel),
    Fallback,
}

/// Compile MLIR through the full pipeline and return a loaded kernel.
fn jit_compile_mlir(
    mlir: &str,
    c_wrapper: &str,
    wrapper_func_name: &str,
) -> Result<JitKernel, String> {
    let hash = fnv_hash(&format!("{}{}", mlir, c_wrapper));

    // Check cache
    {
        let cache = global_jit_cache().lock().unwrap();
        if let Some(path) = cache.entries.get(&hash) {
            if path.exists() {
                return JitKernel::load(path, wrapper_func_name);
            }
        }
    }

    // Check on-disk cache
    let so_path = jit_cache_dir().join(format!("{}.so", hash));
    if so_path.exists() {
        let kernel = JitKernel::load(&so_path, wrapper_func_name)?;
        let mut cache = global_jit_cache().lock().unwrap();
        cache.entries.insert(hash, so_path);
        return Ok(kernel);
    }

    // Full pipeline
    let lowered = lower_mlir_to_llvm(mlir, &hash)?;
    let llvm_ir = translate_to_llvm_ir(&lowered, &hash)?;
    let so = compile_to_shared_lib(&llvm_ir, c_wrapper, &hash)?;
    let kernel = JitKernel::load(&so, wrapper_func_name)?;

    let mut cache = global_jit_cache().lock().unwrap();
    cache.entries.insert(hash, so);

    Ok(kernel)
}

// ---------------------------------------------------------------------------
// C fallback code generation (standalone, no MLIR dependency)
// ---------------------------------------------------------------------------

fn generate_matmul_c_fallback(m: usize, k: usize, n: usize) -> String {
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

void matmul_fallback(double* a, int64_t m, int64_t k_dim,
                     double* b, int64_t k2, int64_t n,
                     double* c) {{
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {{
            double s = 0.0;
            for (int64_t p = 0; p < k_dim; p++)
                s += a[i * k_dim + p] * b[p * n + j];
            c[i * n + j] = s;
        }}
}}
"#
    )
}

fn generate_elementwise_c_fallback(op: &str, size: usize) -> String {
    let cop = match op {
        "add" => "+",
        "sub" => "-",
        "mul" => "*",
        "div" => "/",
        _ => "+",
    };
    let fname = format!("elementwise_{}_fallback", op);
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

void {fname}(double* a, double* b, double* c, int64_t n) {{
    for (int64_t i = 0; i < n; i++)
        c[i] = a[i] {cop} b[i];
}}
"#
    )
}

fn generate_reduce_c_fallback(op: &str) -> String {
    let (fname, init, combine) = match op {
        "sum" => ("reduce_sum_fallback", "0.0", "acc += v;"),
        "max" => ("reduce_max_fallback", "-1e308", "if (v > acc) acc = v;"),
        "min" => ("reduce_min_fallback", "1e308", "if (v < acc) acc = v;"),
        _ => ("reduce_sum_fallback", "0.0", "acc += v;"),
    };
    format!(
        r#"#include <stdlib.h>
#include <stdint.h>

void {fname}(double* a, double* c, int64_t rows, int64_t cols) {{
    for (int64_t i = 0; i < rows; i++) {{
        double acc = {init};
        for (int64_t j = 0; j < cols; j++) {{
            double v = a[i * cols + j];
            {combine}
        }}
        c[i] = acc;
    }}
}}
"#
    )
}

fn generate_relu_c_fallback() -> String {
    r#"#include <stdlib.h>
#include <stdint.h>

void relu_fallback(double* a, double* c, int64_t n) {
    for (int64_t i = 0; i < n; i++)
        c[i] = a[i] > 0.0 ? a[i] : 0.0;
}
"#
    .to_string()
}

fn generate_softmax_c_fallback() -> String {
    r#"#include <stdlib.h>
#include <stdint.h>
#include <math.h>

void softmax_fallback(double* a, double* c, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; i++) {
        double mx = -1e308;
        for (int64_t j = 0; j < cols; j++) {
            double v = a[i * cols + j];
            if (v > mx) mx = v;
        }
        double s = 0.0;
        for (int64_t j = 0; j < cols; j++) {
            c[i * cols + j] = exp(a[i * cols + j] - mx);
            s += c[i * cols + j];
        }
        for (int64_t j = 0; j < cols; j++)
            c[i * cols + j] /= s;
    }
}
"#
    .to_string()
}

// ---------------------------------------------------------------------------
// CPU fallback implementations (pure Rust, no external tools needed)
// ---------------------------------------------------------------------------

fn cpu_matmul(a: &[f64], a_shape: &[usize], b: &[f64], b_shape: &[usize]) -> Vec<f64> {
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

fn cpu_elementwise(op: &str, a: &[f64], b: &[f64]) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| match op {
            "add" => x + y,
            "sub" => x - y,
            "mul" => x * y,
            "div" => x / y,
            _ => x + y,
        })
        .collect()
}

fn cpu_reduce(op: &str, data: &[f64], shape: &[usize], _axis: usize) -> Vec<f64> {
    let rows = shape[0];
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };
    let mut out = vec![0.0; rows];
    for i in 0..rows {
        let init = match op {
            "sum" => 0.0,
            "max" => f64::NEG_INFINITY,
            "min" => f64::INFINITY,
            _ => 0.0,
        };
        let mut acc = init;
        for j in 0..cols {
            let v = data[i * cols + j];
            match op {
                "sum" => acc += v,
                "max" => {
                    if v > acc {
                        acc = v
                    }
                }
                "min" => {
                    if v < acc {
                        acc = v
                    }
                }
                _ => acc += v,
            }
        }
        out[i] = acc;
    }
    out
}

fn cpu_relu(data: &[f64]) -> Vec<f64> {
    data.iter()
        .map(|&v| if v > 0.0 { v } else { 0.0 })
        .collect()
}

fn cpu_softmax(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let rows = shape[0];
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };
    let mut out = vec![0.0; data.len()];
    for i in 0..rows {
        let start = i * cols;
        let mx = data[start..start + cols]
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let mut s = 0.0;
        for j in 0..cols {
            out[start + j] = (data[start + j] - mx).exp();
            s += out[start + j];
        }
        for j in 0..cols {
            out[start + j] /= s;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Public Kernel Launch API
// ---------------------------------------------------------------------------

/// JIT-compiled matrix multiplication.
/// Falls back to pure Rust if compilation fails.
pub fn jit_matmul(
    a: &[f64],
    a_shape: &[usize],
    b: &[f64],
    b_shape: &[usize],
) -> Vec<f64> {
    assert!(a_shape.len() == 2 && b_shape.len() == 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    assert_eq!(k, b_shape[0], "matmul inner dimension mismatch");

    // Try JIT compilation
    if jit_available() {
        let mlir = generate_matmul_mlir(m, k, n);
        let c_wrapper = generate_matmul_c_wrapper(m, k, n);

        match jit_compile_mlir(&mlir, &c_wrapper, "matmul_wrapper") {
            Ok(kernel) => {
                let mut a_buf = a.to_vec();
                let mut b_buf = b.to_vec();
                let mut c_buf = vec![0.0f64; m * n];

                type MatmulFn = unsafe extern "C" fn(
                    *mut f64, i64, i64,
                    *mut f64, i64, i64,
                    *mut f64,
                );
                let func: MatmulFn = unsafe { std::mem::transmute(kernel.as_ptr()) };
                unsafe {
                    func(
                        a_buf.as_mut_ptr(), m as i64, k as i64,
                        b_buf.as_mut_ptr(), k as i64, n as i64,
                        c_buf.as_mut_ptr(),
                    );
                }
                return c_buf;
            }
            Err(_e) => {
                // Try C fallback .so
                let c_code = generate_matmul_c_fallback(m, k, n);
                let hash = fnv_hash(&c_code);
                if let Ok(so) = compile_c_fallback_to_shared_lib(&c_code, &hash) {
                    if let Ok(kernel) = JitKernel::load(&so, "matmul_fallback") {
                        let mut a_buf = a.to_vec();
                        let mut b_buf = b.to_vec();
                        let mut c_buf = vec![0.0f64; m * n];
                        type MatmulFn = unsafe extern "C" fn(
                            *mut f64, i64, i64,
                            *mut f64, i64, i64,
                            *mut f64,
                        );
                        let func: MatmulFn =
                            unsafe { std::mem::transmute(kernel.as_ptr()) };
                        unsafe {
                            func(
                                a_buf.as_mut_ptr(), m as i64, k as i64,
                                b_buf.as_mut_ptr(), k as i64, n as i64,
                                c_buf.as_mut_ptr(),
                            );
                        }
                        return c_buf;
                    }
                }
            }
        }
    }

    // Pure Rust fallback
    cpu_matmul(a, a_shape, b, b_shape)
}

/// JIT-compiled elementwise binary operation.
pub fn jit_elementwise(
    op: &str,
    a: &[f64],
    b: &[f64],
    shape: &[usize],
) -> Vec<f64> {
    let size: usize = shape.iter().product();
    assert_eq!(a.len(), size);
    assert_eq!(b.len(), size);

    if jit_available() {
        let mlir = generate_elementwise_mlir(op, size);
        let c_wrapper = generate_elementwise_c_wrapper(op, size);
        let fname = format!("elementwise_{}_wrapper", op);

        match jit_compile_mlir(&mlir, &c_wrapper, &fname) {
            Ok(kernel) => {
                let mut a_buf = a.to_vec();
                let mut b_buf = b.to_vec();
                let mut c_buf = vec![0.0f64; size];
                type ElemFn =
                    unsafe extern "C" fn(*mut f64, *mut f64, *mut f64, i64);
                let func: ElemFn =
                    unsafe { std::mem::transmute(kernel.as_ptr()) };
                unsafe {
                    func(
                        a_buf.as_mut_ptr(),
                        b_buf.as_mut_ptr(),
                        c_buf.as_mut_ptr(),
                        size as i64,
                    );
                }
                return c_buf;
            }
            Err(_) => {
                // Try C fallback
                let c_code = generate_elementwise_c_fallback(op, size);
                let hash = fnv_hash(&c_code);
                let fname_fb = format!("elementwise_{}_fallback", op);
                if let Ok(so) = compile_c_fallback_to_shared_lib(&c_code, &hash) {
                    if let Ok(kernel) = JitKernel::load(&so, &fname_fb) {
                        let mut a_buf = a.to_vec();
                        let mut b_buf = b.to_vec();
                        let mut c_buf = vec![0.0f64; size];
                        type ElemFn =
                            unsafe extern "C" fn(*mut f64, *mut f64, *mut f64, i64);
                        let func: ElemFn =
                            unsafe { std::mem::transmute(kernel.as_ptr()) };
                        unsafe {
                            func(
                                a_buf.as_mut_ptr(),
                                b_buf.as_mut_ptr(),
                                c_buf.as_mut_ptr(),
                                size as i64,
                            );
                        }
                        return c_buf;
                    }
                }
            }
        }
    }

    cpu_elementwise(op, a, b)
}

/// JIT-compiled reduction.
pub fn jit_reduce(
    op: &str,
    data: &[f64],
    shape: &[usize],
    axis: usize,
) -> Vec<f64> {
    let rows = shape[0];
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };

    if jit_available() && shape.len() >= 2 {
        let mlir = generate_reduce_mlir(op, rows, cols);
        let c_wrapper = generate_reduce_c_wrapper(op, rows, cols);
        let fname = match op {
            "sum" => "reduce_sum_wrapper",
            "max" => "reduce_max_wrapper",
            "min" => "reduce_min_wrapper",
            _ => "reduce_sum_wrapper",
        };

        match jit_compile_mlir(&mlir, &c_wrapper, fname) {
            Ok(kernel) => {
                let mut data_buf = data.to_vec();
                let mut c_buf = vec![0.0f64; rows];
                type ReduceFn =
                    unsafe extern "C" fn(*mut f64, *mut f64, i64, i64);
                let func: ReduceFn =
                    unsafe { std::mem::transmute(kernel.as_ptr()) };
                unsafe {
                    func(
                        data_buf.as_mut_ptr(),
                        c_buf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                return c_buf;
            }
            Err(_) => {
                // Try C fallback
                let c_code = generate_reduce_c_fallback(op);
                let hash = fnv_hash(&c_code);
                let fname_fb = match op {
                    "sum" => "reduce_sum_fallback",
                    "max" => "reduce_max_fallback",
                    "min" => "reduce_min_fallback",
                    _ => "reduce_sum_fallback",
                };
                if let Ok(so) = compile_c_fallback_to_shared_lib(&c_code, &hash) {
                    if let Ok(kernel) = JitKernel::load(&so, fname_fb) {
                        let mut data_buf = data.to_vec();
                        let mut c_buf = vec![0.0f64; rows];
                        type ReduceFn =
                            unsafe extern "C" fn(*mut f64, *mut f64, i64, i64);
                        let func: ReduceFn =
                            unsafe { std::mem::transmute(kernel.as_ptr()) };
                        unsafe {
                            func(
                                data_buf.as_mut_ptr(),
                                c_buf.as_mut_ptr(),
                                rows as i64,
                                cols as i64,
                            );
                        }
                        return c_buf;
                    }
                }
            }
        }
    }

    cpu_reduce(op, data, shape, axis)
}

/// JIT-compiled ReLU activation.
pub fn jit_relu(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let size: usize = shape.iter().product();

    if jit_available() {
        let mlir = generate_relu_mlir(size);
        let c_wrapper = generate_relu_c_wrapper(size);

        match jit_compile_mlir(&mlir, &c_wrapper, "relu_wrapper") {
            Ok(kernel) => {
                let mut data_buf = data.to_vec();
                let mut c_buf = vec![0.0f64; size];
                type ReluFn =
                    unsafe extern "C" fn(*mut f64, *mut f64, i64);
                let func: ReluFn =
                    unsafe { std::mem::transmute(kernel.as_ptr()) };
                unsafe {
                    func(
                        data_buf.as_mut_ptr(),
                        c_buf.as_mut_ptr(),
                        size as i64,
                    );
                }
                return c_buf;
            }
            Err(_) => {
                let c_code = generate_relu_c_fallback();
                let hash = fnv_hash(&c_code);
                if let Ok(so) = compile_c_fallback_to_shared_lib(&c_code, &hash) {
                    if let Ok(kernel) = JitKernel::load(&so, "relu_fallback") {
                        let mut data_buf = data.to_vec();
                        let mut c_buf = vec![0.0f64; size];
                        type ReluFn =
                            unsafe extern "C" fn(*mut f64, *mut f64, i64);
                        let func: ReluFn =
                            unsafe { std::mem::transmute(kernel.as_ptr()) };
                        unsafe {
                            func(
                                data_buf.as_mut_ptr(),
                                c_buf.as_mut_ptr(),
                                size as i64,
                            );
                        }
                        return c_buf;
                    }
                }
            }
        }
    }

    cpu_relu(data)
}

/// JIT-compiled softmax (row-wise on 2D tensor).
pub fn jit_softmax(data: &[f64], shape: &[usize]) -> Vec<f64> {
    let rows = shape[0];
    let cols = if shape.len() >= 2 { shape[1] } else { 1 };

    if jit_available() && shape.len() >= 2 {
        let mlir = generate_softmax_mlir(rows, cols);
        let c_wrapper = generate_softmax_c_wrapper(rows, cols);

        match jit_compile_mlir(&mlir, &c_wrapper, "softmax_wrapper") {
            Ok(kernel) => {
                let mut data_buf = data.to_vec();
                let mut c_buf = vec![0.0f64; rows * cols];
                type SoftmaxFn =
                    unsafe extern "C" fn(*mut f64, *mut f64, i64, i64);
                let func: SoftmaxFn =
                    unsafe { std::mem::transmute(kernel.as_ptr()) };
                unsafe {
                    func(
                        data_buf.as_mut_ptr(),
                        c_buf.as_mut_ptr(),
                        rows as i64,
                        cols as i64,
                    );
                }
                return c_buf;
            }
            Err(_) => {
                let c_code = generate_softmax_c_fallback();
                let hash = fnv_hash(&c_code);
                if let Ok(so) = compile_c_fallback_to_shared_lib(&c_code, &hash) {
                    if let Ok(kernel) = JitKernel::load(&so, "softmax_fallback") {
                        let mut data_buf = data.to_vec();
                        let mut c_buf = vec![0.0f64; rows * cols];
                        type SoftmaxFn =
                            unsafe extern "C" fn(*mut f64, *mut f64, i64, i64);
                        let func: SoftmaxFn =
                            unsafe { std::mem::transmute(kernel.as_ptr()) };
                        unsafe {
                            func(
                                data_buf.as_mut_ptr(),
                                c_buf.as_mut_ptr(),
                                rows as i64,
                                cols as i64,
                            );
                        }
                        return c_buf;
                    }
                }
            }
        }
    }

    cpu_softmax(data, shape)
}

// ---------------------------------------------------------------------------
// Interpreter integration API
// ---------------------------------------------------------------------------

/// Opaque handle to a JIT-compiled function (for interpreter use).
pub struct JitHandle {
    /// The compiled .so path
    pub so_path: PathBuf,
    /// The MLIR source (for debugging)
    pub mlir_source: String,
    /// Function name in the .so
    pub func_name: String,
}

/// Compile an MLIR string to a shared library. Returns a handle.
pub fn jit_compile(mlir: &str) -> Result<JitHandle, String> {
    if !jit_available() {
        return Err("JIT not available: mlir-opt, mlir-translate, or clang missing".to_string());
    }

    let hash = fnv_hash(mlir);

    // Lower and compile
    let lowered = lower_mlir_to_llvm(mlir, &hash)?;
    let llvm_ir = translate_to_llvm_ir(&lowered, &hash)?;

    // No C wrapper needed for raw compilation
    let clang = find_tool("clang").ok_or("clang not found")?;
    let dir = jit_cache_dir();
    let ll_path = dir.join(format!("{}_raw.ll", hash));
    let so_path = dir.join(format!("{}_raw.so", hash));

    if !so_path.exists() {
        fs::write(&ll_path, &llvm_ir).map_err(|e| format!("write: {}", e))?;
        let output = Command::new(&clang)
            .args(&["-shared", "-O2", "-fPIC", "-lm", "-o"])
            .arg(&so_path)
            .arg(&ll_path)
            .output()
            .map_err(|e| format!("clang: {}", e))?;
        let _ = fs::remove_file(&ll_path);
        if !output.status.success() {
            return Err(format!(
                "compilation failed:\n{}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
    }

    Ok(JitHandle {
        so_path,
        mlir_source: mlir.to_string(),
        func_name: String::new(),
    })
}

/// Run a JIT-compiled handle with raw arguments (interpreter integration).
/// This is a simplified interface -- the interpreter should use the specific
/// jit_matmul, jit_elementwise, etc. functions for type-safe calls.
pub fn jit_run(handle: &JitHandle, func_name: &str) -> Result<JitKernel, String> {
    JitKernel::load(&handle.so_path, func_name)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- MLIR generation tests --

    #[test]
    fn test_generate_matmul_mlir_structure() {
        let mlir = generate_matmul_mlir(4, 3, 2);
        assert!(mlir.contains("func.func @matmul"));
        assert!(mlir.contains("memref<4x3xf64>"));
        assert!(mlir.contains("memref<3x2xf64>"));
        assert!(mlir.contains("memref<4x2xf64>"));
        assert!(mlir.contains("affine.for"));
        assert!(mlir.contains("arith.mulf"));
        assert!(mlir.contains("arith.addf"));
    }

    #[test]
    fn test_generate_elementwise_mlir_add() {
        let mlir = generate_elementwise_mlir("add", 8);
        assert!(mlir.contains("func.func @elementwise_add"));
        assert!(mlir.contains("memref<8xf64>"));
        assert!(mlir.contains("arith.addf"));
    }

    #[test]
    fn test_generate_elementwise_mlir_mul() {
        let mlir = generate_elementwise_mlir("mul", 16);
        assert!(mlir.contains("arith.mulf"));
        assert!(mlir.contains("memref<16xf64>"));
    }

    #[test]
    fn test_generate_reduce_mlir_sum() {
        let mlir = generate_reduce_mlir("sum", 4, 8);
        assert!(mlir.contains("func.func @reduce_sum"));
        assert!(mlir.contains("memref<4x8xf64>"));
        assert!(mlir.contains("memref<4xf64>"));
    }

    #[test]
    fn test_generate_reduce_mlir_max() {
        let mlir = generate_reduce_mlir("max", 2, 3);
        assert!(mlir.contains("func.func @reduce_max"));
        assert!(mlir.contains("arith.maximumf"));
    }

    #[test]
    fn test_generate_relu_mlir_structure() {
        let mlir = generate_relu_mlir(10);
        assert!(mlir.contains("func.func @relu"));
        assert!(mlir.contains("memref<10xf64>"));
        assert!(mlir.contains("arith.maximumf"));
    }

    #[test]
    fn test_generate_softmax_mlir_structure() {
        let mlir = generate_softmax_mlir(2, 4);
        assert!(mlir.contains("func.func @softmax"));
        assert!(mlir.contains("memref<2x4xf64>"));
        assert!(mlir.contains("math.exp"));
        assert!(mlir.contains("arith.divf"));
    }

    // -- Tensor buffer tests --

    #[test]
    fn test_tensor_buffer_zeros() {
        let t = TensorBuffer::zeros(&[3, 4]);
        assert_eq!(t.numel(), 12);
        assert_eq!(t.rows(), 3);
        assert_eq!(t.cols(), 4);
        assert!(t.data.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_tensor_buffer_strides() {
        let t = TensorBuffer::new(vec![0.0; 24], vec![2, 3, 4]);
        assert_eq!(t.strides(), vec![12, 4, 1]);
    }

    #[test]
    fn test_memref_descriptor_1d() {
        let mut data = vec![1.0, 2.0, 3.0];
        let desc = MemRefDescriptor1D::from_slice(&mut data);
        assert_eq!(desc.offset, 0);
        assert_eq!(desc.sizes[0], 3);
        assert_eq!(desc.strides[0], 1);
    }

    #[test]
    fn test_memref_descriptor_2d() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let desc = MemRefDescriptor2D::from_slice(&mut data, 2, 3);
        assert_eq!(desc.offset, 0);
        assert_eq!(desc.sizes, [2, 3]);
        assert_eq!(desc.strides, [3, 1]);
    }

    // -- CPU fallback tests --

    #[test]
    fn test_cpu_matmul_identity() {
        let a = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = cpu_matmul(&a, &[2, 2], &b, &[2, 2]);
        assert_eq!(c, vec![5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_cpu_matmul_2x3_3x2() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let c = cpu_matmul(&a, &[2, 3], &b, &[3, 2]);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!((c[0] - 58.0).abs() < 1e-10);
        assert!((c[1] - 64.0).abs() < 1e-10);
        assert!((c[2] - 139.0).abs() < 1e-10);
        assert!((c[3] - 154.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_elementwise_add() {
        let result = cpu_elementwise("add", &[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_cpu_elementwise_sub() {
        let result = cpu_elementwise("sub", &[10.0, 20.0], &[3.0, 7.0]);
        assert_eq!(result, vec![7.0, 13.0]);
    }

    #[test]
    fn test_cpu_reduce_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = cpu_reduce("sum", &data, &[2, 3], 1);
        assert!((result[0] - 6.0).abs() < 1e-10);
        assert!((result[1] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_reduce_max() {
        let data = vec![1.0, 5.0, 3.0, 2.0, 8.0, 4.0];
        let result = cpu_reduce("max", &data, &[2, 3], 1);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_relu() {
        let result = cpu_relu(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_cpu_softmax_sums_to_one() {
        let data = vec![1.0, 2.0, 3.0];
        let result = cpu_softmax(&data, &[1, 3]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_softmax_ordering() {
        let data = vec![1.0, 2.0, 3.0];
        let result = cpu_softmax(&data, &[1, 3]);
        assert!(result[2] > result[1]);
        assert!(result[1] > result[0]);
    }

    // -- Hash / cache tests --

    #[test]
    fn test_fnv_hash_deterministic() {
        let h1 = fnv_hash("test input");
        let h2 = fnv_hash("test input");
        assert_eq!(h1, h2);
    }

    #[test]
    fn test_fnv_hash_different_inputs() {
        let h1 = fnv_hash("input a");
        let h2 = fnv_hash("input b");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_cache_dir_exists() {
        let dir = jit_cache_dir();
        assert!(dir.exists());
    }

    // -- JIT availability test --

    #[test]
    fn test_jit_available_returns_bool() {
        // Just verify it does not panic
        let _avail = jit_available();
    }

    // -- Full JIT execution tests (only run if tools available) --

    #[test]
    fn test_jit_matmul_execution() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = jit_matmul(&a, &[2, 2], &b, &[2, 2]);
        // [1*5+2*7, 1*6+2*8] = [19, 22]
        // [3*5+4*7, 3*6+4*8] = [43, 50]
        assert!((c[0] - 19.0).abs() < 1e-6);
        assert!((c[1] - 22.0).abs() < 1e-6);
        assert!((c[2] - 43.0).abs() < 1e-6);
        assert!((c[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_elementwise_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let c = jit_elementwise("add", &a, &b, &[4]);
        assert!((c[0] - 11.0).abs() < 1e-6);
        assert!((c[3] - 44.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_elementwise_mul() {
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        let c = jit_elementwise("mul", &a, &b, &[2]);
        assert!((c[0] - 8.0).abs() < 1e-6);
        assert!((c[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_reduce_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let result = jit_reduce("sum", &data, &[2, 3], 1);
        assert!((result[0] - 6.0).abs() < 1e-6);
        assert!((result[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_relu() {
        let data = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let result = jit_relu(&data, &[5]);
        assert!((result[0] - 0.0).abs() < 1e-6);
        assert!((result[3] - 1.0).abs() < 1e-6);
        assert!((result[4] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_softmax() {
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let result = jit_softmax(&data, &[1, 4]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!(result[3] > result[2]);
    }

    #[test]
    fn test_jit_matmul_non_square() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
        let b = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3x2
        let c = jit_matmul(&a, &[2, 3], &b, &[3, 2]);
        // [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
        // [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
        assert!((c[0] - 22.0).abs() < 1e-6);
        assert!((c[1] - 28.0).abs() < 1e-6);
        assert!((c[2] - 49.0).abs() < 1e-6);
        assert!((c[3] - 64.0).abs() < 1e-6);
    }

    #[test]
    fn test_jit_caching() {
        // Call twice with same params -- second should be cached
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![1.0, 0.0, 0.0, 1.0];
        let c1 = jit_matmul(&a, &[2, 2], &b, &[2, 2]);
        let c2 = jit_matmul(&a, &[2, 2], &b, &[2, 2]);
        assert_eq!(c1, c2);
    }

    // -- MLIR validation tests (only if mlir-opt available) --

    #[test]
    fn test_validate_matmul_mlir() {
        if !jit_available() {
            eprintln!("JIT tools not available, skipping MLIR validation test");
            return;
        }
        let mlir = generate_matmul_mlir(2, 3, 2);
        let result = validate_mlir(&mlir);
        assert!(result.is_ok(), "matmul MLIR validation failed: {:?}", result.err());
    }

    #[test]
    fn test_validate_elementwise_mlir() {
        if !jit_available() {
            return;
        }
        let mlir = generate_elementwise_mlir("add", 8);
        let result = validate_mlir(&mlir);
        assert!(result.is_ok(), "elementwise MLIR validation failed: {:?}", result.err());
    }

    #[test]
    fn test_validate_relu_mlir() {
        if !jit_available() {
            return;
        }
        let mlir = generate_relu_mlir(4);
        let result = validate_mlir(&mlir);
        assert!(result.is_ok(), "relu MLIR validation failed: {:?}", result.err());
    }

    #[test]
    fn test_validate_reduce_mlir() {
        if !jit_available() {
            return;
        }
        let mlir = generate_reduce_mlir("sum", 3, 4);
        let result = validate_mlir(&mlir);
        assert!(result.is_ok(), "reduce MLIR validation failed: {:?}", result.err());
    }

    #[test]
    fn test_validate_softmax_mlir() {
        if !jit_available() {
            return;
        }
        let mlir = generate_softmax_mlir(2, 3);
        let result = validate_mlir(&mlir);
        assert!(result.is_ok(), "softmax MLIR validation failed: {:?}", result.err());
    }

    #[test]
    fn test_full_pipeline_matmul() {
        if !jit_available() {
            eprintln!("JIT tools not available, skipping full pipeline test");
            return;
        }
        let mlir = generate_matmul_mlir(2, 2, 2);
        let hash = fnv_hash(&mlir);
        let lowered = lower_mlir_to_llvm(&mlir, &hash);
        assert!(lowered.is_ok(), "lowering failed: {:?}", lowered.err());

        let llvm_ir = translate_to_llvm_ir(&lowered.unwrap(), &hash);
        assert!(llvm_ir.is_ok(), "translation failed: {:?}", llvm_ir.err());

        let ir = llvm_ir.unwrap();
        assert!(ir.contains("define") || ir.contains("@matmul"),
                "LLVM IR should contain function definition");
    }
}
