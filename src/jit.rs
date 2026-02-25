//! JIT compilation engine for @gpu annotated functions.
//!
//! Compiles Vortex functions through the MLIR pipeline to native shared libraries,
//! then executes them via a subprocess runner. Falls back to the interpreter if
//! any tool in the pipeline is missing.

use crate::ast::*;
use crate::codegen;
use std::collections::HashMap;
use std::fmt::Write as FmtWrite;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::process::{Command, Output};
use std::time::Duration;

/// Represents a compiled native function
#[derive(Debug, Clone)]
pub struct CompiledFn {
    /// Path to the compiled shared library
    pub lib_path: PathBuf,
    /// The exported C function name
    pub fn_name: String,
    /// Parameter types
    pub param_types: Vec<JitType>,
    /// Return type
    pub return_type: JitType,
    /// Path to the runner executable
    pub runner_path: PathBuf,
}

/// JIT type representation for marshalling values
#[derive(Debug, Clone, PartialEq)]
pub enum JitType {
    I64,
    F64,
    Bool,
    Void,
}

impl JitType {
    /// C type name for code generation
    fn c_type(&self) -> &'static str {
        match self {
            JitType::I64 => "long long",
            JitType::F64 => "double",
            JitType::Bool => "int",
            JitType::Void => "void",
        }
    }

    /// printf format specifier
    fn printf_fmt(&self) -> &'static str {
        match self {
            JitType::I64 => "%lld",
            JitType::F64 => "%.17g",
            JitType::Bool => "%d",
            JitType::Void => "",
        }
    }
}

/// The JIT compilation engine
pub struct JitEngine {
    /// Cache of compiled functions: fn_name -> CompiledFn
    cache: HashMap<String, CompiledFn>,
    /// Working directory for compilation artifacts
    compile_dir: PathBuf,
    /// Whether MLIR tools are available
    tools_available: Option<bool>,
}

impl JitEngine {
    pub fn new() -> Self {
        let compile_dir = std::env::temp_dir().join("vortex_jit");
        let _ = fs::create_dir_all(&compile_dir);
        Self {
            cache: HashMap::new(),
            compile_dir,
            tools_available: None,
        }
    }

    /// Check if the required MLIR tools and clang are available
    pub fn check_tools(&mut self) -> bool {
        if let Some(available) = self.tools_available {
            return available;
        }

        let mlir_opt = find_tool("mlir-opt");
        let mlir_translate = find_tool("mlir-translate");
        let clang = find_tool("clang");

        let available = mlir_opt.is_some() && mlir_translate.is_some() && clang.is_some();
        self.tools_available = Some(available);
        available
    }

    /// Get a cached compiled function, or None
    pub fn get_cached(&self, name: &str) -> Option<&CompiledFn> {
        self.cache.get(name)
    }

    /// Compile a Vortex function to a native shared library
    pub fn compile_function(&mut self, func: &Function) -> Result<CompiledFn, String> {
        // Check cache first
        if let Some(compiled) = self.cache.get(&func.name.name) {
            return Ok(compiled.clone());
        }

        if !self.check_tools() {
            return Err("MLIR tools (mlir-opt, mlir-translate) or clang not found".to_string());
        }

        let fn_name = &func.name.name;
        let work_dir = self.compile_dir.join(fn_name);
        let _ = fs::create_dir_all(&work_dir);

        // 1. Determine types
        let param_types = resolve_param_types(&func.params);
        let return_type = resolve_return_type(&func.ret_type);

        // 2. Generate MLIR IR for this function
        let mlir_ir = generate_function_mlir(func, &param_types, &return_type)?;
        let mlir_path = work_dir.join("input.mlir");
        fs::write(&mlir_path, &mlir_ir)
            .map_err(|e| format!("failed to write MLIR: {}", e))?;

        // 3. Lower through mlir-opt
        let lowered_path = work_dir.join("lowered.mlir");
        let mlir_opt = find_tool("mlir-opt").unwrap();
        let opt_result = command_output_timeout(
            {
                let mut c = Command::new(&mlir_opt);
                c.arg(&mlir_path)
                    .arg("--canonicalize")
                    .arg("--cse")
                    .arg("--convert-scf-to-cf")
                    .arg("--convert-func-to-llvm")
                    .arg("--convert-arith-to-llvm")
                    .arg("--convert-cf-to-llvm")
                    .arg("--finalize-memref-to-llvm")
                    .arg("--reconcile-unrealized-casts")
                    .arg("-o")
                    .arg(&lowered_path);
                c
            }
        )
        .map_err(|e| format!("mlir-opt failed to execute: {}", e))?;

        if !opt_result.status.success() {
            let stderr = String::from_utf8_lossy(&opt_result.stderr);
            return Err(format!("mlir-opt failed:\n{}", stderr));
        }

        // 4. Translate to LLVM IR
        let llvm_path = work_dir.join("output.ll");
        let mlir_translate = find_tool("mlir-translate").unwrap();
        let translate_result = command_output_timeout(
            {
                let mut c = Command::new(&mlir_translate);
                c.arg("--mlir-to-llvmir")
                    .arg(&lowered_path)
                    .arg("-o")
                    .arg(&llvm_path);
                c
            }
        )
        .map_err(|e| format!("mlir-translate failed to execute: {}", e))?;

        if !translate_result.status.success() {
            let stderr = String::from_utf8_lossy(&translate_result.stderr);
            return Err(format!("mlir-translate failed:\n{}", stderr));
        }

        // 5. Compile to shared object
        let so_path = work_dir.join(format!("lib{}.so", fn_name));
        let clang = find_tool("clang").unwrap();
        let clang_result = command_output_timeout(
            {
                let mut c = Command::new(&clang);
                c.arg("-shared")
                    .arg("-O2")
                    .arg("-o")
                    .arg(&so_path)
                    .arg(&llvm_path);
                c
            }
        )
        .map_err(|e| format!("clang failed to execute: {}", e))?;

        if !clang_result.status.success() {
            let stderr = String::from_utf8_lossy(&clang_result.stderr);
            return Err(format!("clang failed:\n{}", stderr));
        }

        // 6. Generate a runner program
        let runner_path = generate_runner(&work_dir, fn_name, &so_path, &param_types, &return_type)?;

        let compiled = CompiledFn {
            lib_path: so_path,
            fn_name: fn_name.clone(),
            param_types,
            return_type,
            runner_path,
        };

        self.cache.insert(fn_name.clone(), compiled.clone());
        Ok(compiled)
    }

    /// Execute a compiled function with the given arguments
    pub fn execute(&self, compiled: &CompiledFn, args: &[f64]) -> Result<JitResult, String> {
        let str_args: Vec<String> = args.iter().map(|a| format!("{}", a)).collect();

        let output = command_output_timeout(
            {
                let mut c = Command::new(&compiled.runner_path);
                c.args(&str_args);
                c
            }
        )
        .map_err(|e| format!("runner execution failed: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("JIT execution failed:\n{}", stderr));
        }

        let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();

        // Parse result based on return type
        match &compiled.return_type {
            JitType::F64 => {
                let val: f64 = stdout.parse()
                    .map_err(|e| format!("failed to parse result '{}': {}", stdout, e))?;
                Ok(JitResult::Float(val))
            }
            JitType::I64 => {
                let val: i64 = stdout.parse()
                    .map_err(|e| format!("failed to parse result '{}': {}", stdout, e))?;
                Ok(JitResult::Int(val))
            }
            JitType::Bool => {
                let val: i32 = stdout.parse()
                    .map_err(|e| format!("failed to parse result '{}': {}", stdout, e))?;
                Ok(JitResult::Bool(val != 0))
            }
            JitType::Void => Ok(JitResult::Void),
        }
    }
}

/// Result from a JIT-compiled function execution
#[derive(Debug, Clone)]
pub enum JitResult {
    Int(i64),
    Float(f64),
    Bool(bool),
    Void,
}

/// Resolve parameter types from AST type annotations
fn resolve_param_types(params: &[Param]) -> Vec<JitType> {
    params.iter().map(|p| resolve_type_expr(&p.ty)).collect()
}

/// Resolve return type from AST
fn resolve_return_type(ret_ty: &Option<TypeExpr>) -> JitType {
    match ret_ty {
        Some(ty) => resolve_type_expr(ty),
        None => JitType::Void,
    }
}

/// Map a Vortex type expression to a JIT type
fn resolve_type_expr(ty: &TypeExpr) -> JitType {
    match &ty.kind {
        TypeExprKind::Named(ident) => match ident.name.as_str() {
            "f32" | "f64" | "float" => JitType::F64,
            "i8" | "i16" | "i32" | "i64" | "u8" | "u16" | "u32" | "u64"
            | "int" | "isize" | "usize" => JitType::I64,
            "bool" => JitType::Bool,
            _ => JitType::F64, // default to f64 for unknown types
        },
        _ => JitType::F64,
    }
}

/// Generate MLIR IR for a single function
fn generate_function_mlir(
    func: &Function,
    param_types: &[JitType],
    return_type: &JitType,
) -> Result<String, String> {
    // Build a minimal program with just this function and generate MLIR via codegen
    let program = Program {
        items: vec![Item {
            kind: ItemKind::Function(func.clone()),
            span: func.name.span,
            is_pub: false,
        }],
    };
    let mlir = codegen::generate_mlir(&program);

    // If codegen produced empty or trivial output, generate manually
    if mlir.trim().is_empty() || !mlir.contains("func") {
        return generate_manual_mlir(func, param_types, return_type);
    }

    Ok(mlir)
}

/// Generate MLIR manually for simple functions the codegen might not handle
fn generate_manual_mlir(
    func: &Function,
    param_types: &[JitType],
    return_type: &JitType,
) -> Result<String, String> {
    let mut out = String::new();

    let fn_name = &func.name.name;
    let ret_mlir = jit_type_to_mlir(return_type);
    let params_mlir: Vec<String> = param_types
        .iter()
        .enumerate()
        .map(|(i, t)| format!("%arg{}: {}", i, jit_type_to_mlir(t)))
        .collect();

    writeln!(out, "module {{").unwrap();
    writeln!(
        out,
        "  func.func @{}({}) -> {} {{",
        fn_name,
        params_mlir.join(", "),
        ret_mlir
    )
    .unwrap();

    // Generate body from AST - handle simple cases
    let body_ir = generate_body_mlir(&func.body, param_types, return_type, &func.params)?;
    write!(out, "{}", body_ir).unwrap();

    writeln!(out, "  }}").unwrap();
    writeln!(out, "}}").unwrap();

    Ok(out)
}

/// Generate MLIR for a function body (handles simple expressions)
fn generate_body_mlir(
    block: &Block,
    param_types: &[JitType],
    return_type: &JitType,
    params: &[Param],
) -> Result<String, String> {
    let mut out = String::new();
    let mut ssa_counter = 0;
    let mut var_map: HashMap<String, (String, &JitType)> = HashMap::new();

    // Map parameter names to SSA values
    for (i, param) in params.iter().enumerate() {
        var_map.insert(
            param.name.name.clone(),
            (format!("%arg{}", i), &param_types[i]),
        );
    }

    // Process statements
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Return(Some(expr)) => {
                let (val, ir) = emit_expr_mlir(expr, &var_map, &mut ssa_counter, return_type)?;
                write!(out, "{}", ir).unwrap();
                writeln!(out, "    func.return {} : {}", val, jit_type_to_mlir(return_type)).unwrap();
            }
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                let (val, ir) = emit_expr_mlir(value, &var_map, &mut ssa_counter, return_type)?;
                write!(out, "{}", ir).unwrap();
                var_map.insert(name.name.clone(), (val, return_type));
            }
            _ => {}
        }
    }

    // Handle trailing expression as implicit return
    if let Some(expr) = &block.expr {
        let (val, ir) = emit_expr_mlir(expr, &var_map, &mut ssa_counter, return_type)?;
        write!(out, "{}", ir).unwrap();
        writeln!(out, "    func.return {} : {}", val, jit_type_to_mlir(return_type)).unwrap();
    }

    Ok(out)
}

/// Emit MLIR for an expression, returning (ssa_value_name, ir_text)
fn emit_expr_mlir(
    expr: &Expr,
    var_map: &HashMap<String, (String, &JitType)>,
    counter: &mut usize,
    _ctx_type: &JitType,
) -> Result<(String, String), String> {
    match &expr.kind {
        ExprKind::IntLiteral(n) => {
            let name = fresh_ssa(counter);
            let ir = format!("    {} = arith.constant {} : i64\n", name, n);
            Ok((name, ir))
        }
        ExprKind::FloatLiteral(n) => {
            let name = fresh_ssa(counter);
            // Format with enough precision
            let ir = format!("    {} = arith.constant {:.17e} : f64\n", name, n);
            Ok((name, ir))
        }
        ExprKind::BoolLiteral(b) => {
            let name = fresh_ssa(counter);
            let val = if *b { 1 } else { 0 };
            let ir = format!("    {} = arith.constant {} : i1\n", name, val);
            Ok((name, ir))
        }
        ExprKind::Ident(ident) => {
            if let Some((ssa, _)) = var_map.get(&ident.name) {
                Ok((ssa.clone(), String::new()))
            } else {
                Err(format!("undefined variable in JIT context: {}", ident.name))
            }
        }
        ExprKind::Binary { lhs, op, rhs } => {
            let (lhs_val, lhs_ir) = emit_expr_mlir(lhs, var_map, counter, _ctx_type)?;
            let (rhs_val, rhs_ir) = emit_expr_mlir(rhs, var_map, counter, _ctx_type)?;
            let result = fresh_ssa(counter);

            let op_str = match op {
                BinOp::Add => "arith.addf",
                BinOp::Sub => "arith.subf",
                BinOp::Mul => "arith.mulf",
                BinOp::Div => "arith.divf",
                _ => return Err(format!("unsupported binary op in JIT: {:?}", op)),
            };

            let ir = format!(
                "{}{}    {} = {} {}, {} : f64\n",
                lhs_ir, rhs_ir, result, op_str, lhs_val, rhs_val
            );
            Ok((result, ir))
        }
        ExprKind::Unary { op, expr: inner } => {
            let (inner_val, inner_ir) = emit_expr_mlir(inner, var_map, counter, _ctx_type)?;
            match op {
                UnaryOp::Neg => {
                    let zero = fresh_ssa(counter);
                    let result = fresh_ssa(counter);
                    let ir = format!(
                        "{}    {} = arith.constant 0.0 : f64\n    {} = arith.subf {}, {} : f64\n",
                        inner_ir, zero, result, zero, inner_val
                    );
                    Ok((result, ir))
                }
                _ => Err(format!("unsupported unary op in JIT: {:?}", op)),
            }
        }
        ExprKind::If { cond, then_block, else_block } => {
            // For simple if/else, we can use scf.if
            let (cond_val, cond_ir) = emit_expr_mlir(cond, var_map, counter, _ctx_type)?;

            // Simplified: just return an error for complex conditionals in manual mode
            // The codegen path should handle these
            Err("complex if/else not supported in manual JIT MLIR generation; use codegen path".to_string())
        }
        _ => Err(format!("unsupported expression in JIT MLIR generation: {:?}", std::mem::discriminant(&expr.kind))),
    }
}

fn fresh_ssa(counter: &mut usize) -> String {
    let name = format!("%{}", counter);
    *counter += 1;
    name
}

fn jit_type_to_mlir(ty: &JitType) -> &'static str {
    match ty {
        JitType::I64 => "i64",
        JitType::F64 => "f64",
        JitType::Bool => "i1",
        JitType::Void => "()",
    }
}

/// Generate a C runner program that loads the .so and calls the function
fn generate_runner(
    work_dir: &PathBuf,
    fn_name: &str,
    so_path: &PathBuf,
    param_types: &[JitType],
    return_type: &JitType,
) -> Result<PathBuf, String> {
    let runner_c_path = work_dir.join("runner.c");
    let runner_bin_path = work_dir.join("runner");

    let mut code = String::new();
    writeln!(code, "#include <stdio.h>").unwrap();
    writeln!(code, "#include <stdlib.h>").unwrap();
    writeln!(code, "#include <dlfcn.h>").unwrap();
    writeln!(code, "").unwrap();

    // Function pointer typedef
    let ret_c = return_type.c_type();
    let params_c: Vec<String> = param_types.iter().map(|t| t.c_type().to_string()).collect();
    writeln!(
        code,
        "typedef {} (*fn_t)({});",
        ret_c,
        if params_c.is_empty() {
            "void".to_string()
        } else {
            params_c.join(", ")
        }
    )
    .unwrap();

    writeln!(code, "").unwrap();
    writeln!(code, "int main(int argc, char *argv[]) {{").unwrap();

    // Load shared library
    writeln!(
        code,
        "    void *lib = dlopen(\"{}\", RTLD_NOW);",
        so_path.to_string_lossy()
    )
    .unwrap();
    writeln!(code, "    if (!lib) {{ fprintf(stderr, \"dlopen: %s\\n\", dlerror()); return 1; }}").unwrap();

    // Get function pointer
    writeln!(
        code,
        "    fn_t func = (fn_t)dlsym(lib, \"{}\");",
        fn_name
    )
    .unwrap();
    writeln!(code, "    if (!func) {{ fprintf(stderr, \"dlsym: %s\\n\", dlerror()); return 1; }}").unwrap();

    // Parse args from command line
    for (i, pt) in param_types.iter().enumerate() {
        match pt {
            JitType::F64 => {
                writeln!(code, "    double arg{} = atof(argv[{}]);", i, i + 1).unwrap();
            }
            JitType::I64 => {
                writeln!(code, "    long long arg{} = atoll(argv[{}]);", i, i + 1).unwrap();
            }
            JitType::Bool => {
                writeln!(code, "    int arg{} = atoi(argv[{}]);", i, i + 1).unwrap();
            }
            JitType::Void => {}
        }
    }

    // Call the function
    let arg_list: Vec<String> = (0..param_types.len()).map(|i| format!("arg{}", i)).collect();
    match return_type {
        JitType::Void => {
            writeln!(code, "    func({});", arg_list.join(", ")).unwrap();
        }
        _ => {
            writeln!(
                code,
                "    {} result = func({});",
                ret_c,
                arg_list.join(", ")
            )
            .unwrap();
            writeln!(
                code,
                "    printf(\"{}\\n\", result);",
                return_type.printf_fmt()
            )
            .unwrap();
        }
    }

    writeln!(code, "    dlclose(lib);").unwrap();
    writeln!(code, "    return 0;").unwrap();
    writeln!(code, "}}").unwrap();

    fs::write(&runner_c_path, &code)
        .map_err(|e| format!("failed to write runner.c: {}", e))?;

    // Compile the runner
    let clang = find_tool("clang").unwrap();
    let result = command_output_timeout(
        {
            let mut c = Command::new(&clang);
            c.arg("-o")
                .arg(&runner_bin_path)
                .arg(&runner_c_path)
                .arg("-ldl")
                .arg("-O2");
            c
        }
    )
    .map_err(|e| format!("clang runner compilation failed: {}", e))?;

    if !result.status.success() {
        let stderr = String::from_utf8_lossy(&result.stderr);
        return Err(format!("runner compilation failed:\n{}", stderr));
    }

    Ok(runner_bin_path)
}

/// Run a `Command` capturing stdout/stderr with a 30-second timeout.
/// This prevents tests from hanging indefinitely when external tools stall.
fn command_output_timeout(mut cmd: Command) -> Result<Output, String> {
    let mut child = cmd
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .map_err(|e| format!("failed to spawn: {}", e))?;

    let timeout = Duration::from_secs(30);
    let start = std::time::Instant::now();

    loop {
        match child.try_wait().map_err(|e| format!("wait error: {}", e))? {
            Some(_) => break,
            None => {
                if start.elapsed() >= timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return Err("process timed out after 30 seconds".to_string());
                }
                std::thread::sleep(Duration::from_millis(50));
            }
        }
    }

    child
        .wait_with_output()
        .map_err(|e| format!("failed to collect output: {}", e))
}

/// Find a tool by trying versioned and unversioned names.
/// Uses native PATH lookup (no subprocess spawning) for speed.
fn find_tool(base_name: &str) -> Option<String> {
    // Try versioned names first (e.g., mlir-opt-20, mlir-opt-19, etc.)
    for version in (14..=22).rev() {
        let versioned = format!("{}-{}", base_name, version);
        if tool_exists_in_path(&versioned) {
            return Some(versioned);
        }
    }
    // Try unversioned
    if tool_exists_in_path(base_name) {
        return Some(base_name.to_string());
    }
    None
}

/// Check if a tool exists in PATH without spawning a subprocess.
fn tool_exists_in_path(name: &str) -> bool {
    if let Ok(path_var) = std::env::var("PATH") {
        for dir in path_var.split(':') {
            let candidate = std::path::Path::new(dir).join(name);
            if candidate.is_file() {
                return true;
            }
        }
    }
    false
}

fn tool_exists(name: &str) -> bool {
    tool_exists_in_path(name)
}

/// Check if a function has the @gpu or @jit annotation
pub fn has_gpu_annotation(func: &Function) -> bool {
    func.annotations.iter().any(|a| matches!(a, Annotation::Gpu | Annotation::Jit))
}

/// Check if a function has the @distributed annotation
pub fn has_distributed_annotation(func: &Function) -> bool {
    func.annotations.iter().any(|a| matches!(a, Annotation::Distributed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser;

    fn parse_program(src: &str) -> Program {
        let tokens = lexer::lex(src);
        parser::parse(tokens, 0).expect("parse failed")
    }

    #[test]
    fn test_parse_gpu_annotation() {
        let prog = parse_program("@gpu\nfn add(a: f64, b: f64) -> f64 { return a + b }");
        assert_eq!(prog.items.len(), 1);
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert!(has_gpu_annotation(f));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_parse_jit_annotation() {
        let prog = parse_program("@jit\nfn mul(a: f64, b: f64) -> f64 { return a * b }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert!(has_gpu_annotation(f));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_parse_inline_annotation() {
        let prog = parse_program("@inline\nfn sq(x: f64) -> f64 { return x * x }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert!(f.annotations.iter().any(|a| matches!(a, Annotation::Inline)));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_parse_distributed_annotation() {
        let prog = parse_program("@distributed\nfn work(x: f64) -> f64 { return x }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert!(has_distributed_annotation(f));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_parse_multiple_annotations() {
        let prog = parse_program("@gpu\n@inline\nfn add(a: f64, b: f64) -> f64 { return a + b }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert_eq!(f.annotations.len(), 2);
            assert!(has_gpu_annotation(f));
            assert!(f.annotations.iter().any(|a| matches!(a, Annotation::Inline)));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_parse_custom_annotation() {
        let prog = parse_program("@myattr\nfn foo(x: f64) -> f64 { return x }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            assert!(f.annotations.iter().any(|a| matches!(a, Annotation::Custom(n, _) if n == "myattr")));
        } else {
            panic!("expected function");
        }
    }

    #[test]
    fn test_gpu_annotation_with_existing_code() {
        // Ensure @gpu doesn't break normal functions
        let prog = parse_program(
            "@gpu\nfn add(a: f64, b: f64) -> f64 { return a + b }\nfn main() { let x = add(1.0, 2.0) }",
        );
        assert_eq!(prog.items.len(), 2);
    }

    #[test]
    fn test_jit_type_resolution() {
        assert_eq!(resolve_type_expr(&TypeExpr {
            kind: TypeExprKind::Named(Ident::new("f64".to_string(), crate::lexer::Span::new(0, 3))),
            span: crate::lexer::Span::new(0, 3),
        }), JitType::F64);

        assert_eq!(resolve_type_expr(&TypeExpr {
            kind: TypeExprKind::Named(Ident::new("i64".to_string(), crate::lexer::Span::new(0, 3))),
            span: crate::lexer::Span::new(0, 3),
        }), JitType::I64);

        assert_eq!(resolve_type_expr(&TypeExpr {
            kind: TypeExprKind::Named(Ident::new("bool".to_string(), crate::lexer::Span::new(0, 4))),
            span: crate::lexer::Span::new(0, 4),
        }), JitType::Bool);
    }

    #[test]
    fn test_jit_engine_creation() {
        let engine = JitEngine::new();
        assert!(engine.cache.is_empty());
        assert!(engine.compile_dir.to_string_lossy().contains("vortex_jit"));
    }

    #[test]
    fn test_find_tool() {
        // clang should be available per task description
        let clang = find_tool("clang");
        // Don't assert -- may not be available in test env
        if clang.is_some() {
            assert!(clang.unwrap().contains("clang"));
        }
    }

    #[test]
    fn test_mlir_generation_simple_add() {
        let prog = parse_program("@gpu\nfn add(a: f64, b: f64) -> f64 { return a + b }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            let param_types = resolve_param_types(&f.params);
            let return_type = resolve_return_type(&f.ret_type);
            let mlir = generate_function_mlir(f, &param_types, &return_type);
            assert!(mlir.is_ok());
            let mlir_text = mlir.unwrap();
            assert!(mlir_text.contains("func"));
            assert!(mlir_text.contains("add"));
        }
    }

    #[test]
    fn test_jit_compile_and_execute() {
        let prog = parse_program("@gpu\nfn jit_add(a: f64, b: f64) -> f64 { return a + b }");
        let mut engine = JitEngine::new();
        if !engine.check_tools() {
            eprintln!("Skipping JIT compile test: tools not available");
            return;
        }
        if let ItemKind::Function(f) = &prog.items[0].kind {
            match engine.compile_function(f) {
                Ok(compiled) => {
                    assert!(compiled.lib_path.exists());
                    assert!(compiled.runner_path.exists());

                    // Execute
                    let result = engine.execute(&compiled, &[3.0, 4.0]);
                    match result {
                        Ok(JitResult::Float(v)) => {
                            assert!((v - 7.0).abs() < 1e-10, "expected 7.0, got {}", v);
                        }
                        Ok(other) => panic!("expected Float result, got {:?}", other),
                        Err(e) => panic!("execution failed: {}", e),
                    }
                }
                Err(e) => panic!("compilation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_jit_cache_hit() {
        let prog = parse_program("@gpu\nfn cached_add(a: f64, b: f64) -> f64 { return a + b }");
        let mut engine = JitEngine::new();
        if !engine.check_tools() {
            eprintln!("Skipping cache test: tools not available");
            return;
        }
        if let ItemKind::Function(f) = &prog.items[0].kind {
            // Compile first time
            let compiled1 = engine.compile_function(f);
            assert!(compiled1.is_ok());

            // Second call should hit cache
            let compiled2 = engine.compile_function(f);
            assert!(compiled2.is_ok());

            // Paths should be the same (cached)
            assert_eq!(
                compiled1.unwrap().lib_path,
                compiled2.unwrap().lib_path
            );
        }
    }

    #[test]
    fn test_jit_multiply() {
        let prog = parse_program("@gpu\nfn jit_mul(a: f64, b: f64) -> f64 { return a * b }");
        let mut engine = JitEngine::new();
        if !engine.check_tools() {
            eprintln!("Skipping JIT multiply test: tools not available");
            return;
        }
        if let ItemKind::Function(f) = &prog.items[0].kind {
            match engine.compile_function(f) {
                Ok(compiled) => {
                    let result = engine.execute(&compiled, &[3.0, 4.0]);
                    match result {
                        Ok(JitResult::Float(v)) => {
                            assert!((v - 12.0).abs() < 1e-10);
                        }
                        Ok(other) => panic!("expected Float, got {:?}", other),
                        Err(e) => panic!("execution failed: {}", e),
                    }
                }
                Err(e) => panic!("compilation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_jit_dot_product_step() {
        let prog = parse_program(
            "@gpu\nfn dot_step(a: f64, b: f64, acc: f64) -> f64 { return acc + a * b }",
        );
        let mut engine = JitEngine::new();
        if !engine.check_tools() {
            eprintln!("Skipping dot product test: tools not available");
            return;
        }
        if let ItemKind::Function(f) = &prog.items[0].kind {
            match engine.compile_function(f) {
                Ok(compiled) => {
                    let result = engine.execute(&compiled, &[3.0, 4.0, 10.0]);
                    match result {
                        Ok(JitResult::Float(v)) => {
                            assert!((v - 22.0).abs() < 1e-10, "expected 22.0, got {}", v);
                        }
                        Ok(other) => panic!("expected Float, got {:?}", other),
                        Err(e) => panic!("execution failed: {}", e),
                    }
                }
                Err(e) => panic!("compilation failed: {}", e),
            }
        }
    }

    #[test]
    fn test_fallback_when_tools_missing() {
        // Create engine with a bogus compile dir to simulate failure
        let mut engine = JitEngine::new();
        engine.tools_available = Some(false); // Force tools unavailable

        let prog = parse_program("@gpu\nfn fb(a: f64) -> f64 { return a }");
        if let ItemKind::Function(f) = &prog.items[0].kind {
            let result = engine.compile_function(f);
            assert!(result.is_err());
            assert!(result.unwrap_err().contains("not found"));
        }
    }

    #[test]
    fn test_annotation_display() {
        assert_eq!(format!("{}", Annotation::Gpu), "@gpu");
        assert_eq!(format!("{}", Annotation::Jit), "@jit");
        assert_eq!(format!("{}", Annotation::Inline), "@inline");
        assert_eq!(format!("{}", Annotation::Distributed), "@distributed");
        assert_eq!(format!("{}", Annotation::Custom("myattr".to_string(), vec![])), "@myattr");
    }

    #[test]
    fn test_jit_type_c_types() {
        assert_eq!(JitType::F64.c_type(), "double");
        assert_eq!(JitType::I64.c_type(), "long long");
        assert_eq!(JitType::Bool.c_type(), "int");
        assert_eq!(JitType::Void.c_type(), "void");
    }

    #[test]
    fn test_generate_runner_code() {
        // Just verify runner generation doesn't crash
        let work_dir = std::env::temp_dir().join("vortex_jit_test_runner");
        let _ = fs::create_dir_all(&work_dir);
        let so_path = work_dir.join("libtest.so");

        let result = generate_runner(
            &work_dir,
            "test_fn",
            &so_path,
            &[JitType::F64, JitType::F64],
            &JitType::F64,
        );

        // May fail if clang isn't available, but shouldn't panic
        if find_tool("clang").is_some() {
            // runner.c should at least be written
            assert!(work_dir.join("runner.c").exists());
        }

        let _ = fs::remove_dir_all(&work_dir);
    }
}
