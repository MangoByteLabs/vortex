//! MCP (Model Context Protocol) server for Vortex.
//!
//! Exposes Vortex language tools (run, typecheck, codegen, etc.) over
//! JSON-RPC 2.0 on stdin/stdout so that AI agents can use Vortex as a tool.

use crate::ast::Program;
use crate::codegen;
use crate::interpreter;
use crate::lexer;
use crate::parser;
use crate::typeck;

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Write};
use std::time::Instant;

// ── Session state ──────────────────────────────────────────────────────

/// Persistent state across MCP tool calls.
pub struct SessionState {
    /// Named model registry (model_id -> parameter count).
    pub models: HashMap<String, usize>,
    /// Command history: (tool_name, summary).
    pub history: Vec<(String, String)>,
    /// Monotonic counter for generating model ids.
    model_counter: u64,
}

impl SessionState {
    pub fn new() -> Self {
        Self {
            models: HashMap::new(),
            history: Vec::new(),
            model_counter: 0,
        }
    }

    fn next_model_id(&mut self) -> String {
        self.model_counter += 1;
        format!("model_{}", self.model_counter)
    }
}

// ── Tool descriptors ───────────────────────────────────────────────────

fn tool_descriptors() -> Value {
    json!([
        {
            "name": "vortex_run",
            "description": "Execute a Vortex program and return its stdout output.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "Vortex source code" },
                    "file": { "type": "string", "description": "Path to a .vx file" }
                }
            }
        },
        {
            "name": "vortex_typecheck",
            "description": "Type-check Vortex code. Returns type errors or OK.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": { "type": "string", "description": "Vortex source code" }
                },
                "required": ["code"]
            }
        },
        {
            "name": "vortex_codegen",
            "description": "Generate MLIR IR from Vortex source.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code":   { "type": "string", "description": "Vortex source code" },
                    "target": { "type": "string", "enum": ["cpu","nvidia","amd"], "description": "Target backend" }
                },
                "required": ["code"]
            }
        },
        {
            "name": "vortex_train",
            "description": "Train a model defined in Vortex code.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code":   { "type": "string" },
                    "epochs": { "type": "integer", "default": 10 },
                    "lr":     { "type": "number",  "default": 0.001 }
                },
                "required": ["code"]
            }
        },
        {
            "name": "vortex_infer",
            "description": "Run inference on a trained model.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "model_id": { "type": "string" },
                    "input":    { "type": "array", "items": { "type": "number" } }
                },
                "required": ["model_id", "input"]
            }
        },
        {
            "name": "vortex_benchmark",
            "description": "Benchmark a Vortex program.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code":       { "type": "string" },
                    "iterations": { "type": "integer", "default": 100 }
                },
                "required": ["code"]
            }
        },
        {
            "name": "vortex_explain",
            "description": "Explain what a Vortex program does (AST summary, types, signatures).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "code": { "type": "string" }
                },
                "required": ["code"]
            }
        },
        {
            "name": "vortex_gpu_status",
            "description": "Query GPU resource status.",
            "inputSchema": { "type": "object", "properties": {} }
        },
        {
            "name": "vortex_create_model",
            "description": "Create a model from an architecture description.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "architecture": { "type": "string", "enum": ["transformer","ssm","hybrid","ebm"] },
                    "config":       { "type": "object" }
                },
                "required": ["architecture"]
            }
        },
        {
            "name": "vortex_list_builtins",
            "description": "List all available Vortex builtin functions.",
            "inputSchema": { "type": "object", "properties": {} }
        }
    ])
}

// ── Helpers ─────────────────────────────────────────────────────────────

/// Parse Vortex source into AST, returning a user-friendly error string on failure.
fn parse_vortex(code: &str) -> Result<Program, String> {
    let tokens = lexer::lex(code);
    parser::parse(tokens, 0).map_err(|diags| {
        diags
            .iter()
            .map(|d| d.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })
}

fn get_code(params: &Value) -> Result<String, String> {
    if let Some(code) = params.get("code").and_then(Value::as_str) {
        Ok(code.to_string())
    } else if let Some(path) = params.get("file").and_then(Value::as_str) {
        std::fs::read_to_string(path).map_err(|e| format!("Cannot read file: {}", e))
    } else {
        Err("Missing 'code' or 'file' parameter".into())
    }
}

// ── Tool implementations ───────────────────────────────────────────────

fn tool_run(params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let program = parse_vortex(&code)?;
    let output = interpreter::interpret(&program)?;
    Ok(json!({ "output": output.join("\n") }))
}

fn tool_typecheck(params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let program = parse_vortex(&code)?;
    match typeck::check(&program, 0) {
        Ok(()) => Ok(json!({ "status": "OK" })),
        Err(diags) => {
            let errors: Vec<String> = diags.iter().map(|d| d.message.clone()).collect();
            Ok(json!({ "status": "error", "errors": errors }))
        }
    }
}

fn tool_codegen(params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let _target = params
        .get("target")
        .and_then(Value::as_str)
        .unwrap_or("cpu");
    let program = parse_vortex(&code)?;
    let mlir = codegen::generate_mlir(&program);
    Ok(json!({ "mlir": mlir }))
}

fn tool_train(params: &Value, state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let epochs = params.get("epochs").and_then(Value::as_i64).unwrap_or(10) as usize;
    let lr = params.get("lr").and_then(Value::as_f64).unwrap_or(0.001);

    // Validate the program first
    let program = parse_vortex(&code)?;
    interpreter::interpret(&program).map_err(|e| format!("Program error: {}", e))?;

    // Simulate training (real GPU training would go through MLIR pipeline)
    let mut losses: Vec<f64> = Vec::new();
    let mut loss = 1.0_f64;
    for _ in 0..epochs {
        loss *= 1.0 - lr;
        losses.push(loss);
    }

    let model_id = state.next_model_id();
    state.models.insert(model_id.clone(), 0);

    Ok(json!({
        "model_id": model_id,
        "epochs": epochs,
        "lr": lr,
        "losses": losses,
        "final_loss": loss
    }))
}

fn tool_infer(params: &Value, state: &mut SessionState) -> Result<Value, String> {
    let model_id = params
        .get("model_id")
        .and_then(Value::as_str)
        .ok_or("Missing model_id")?;
    let input = params
        .get("input")
        .ok_or("Missing input")?;

    if !state.models.contains_key(model_id) {
        return Err(format!("Unknown model: {}", model_id));
    }

    // Placeholder inference – echo input through an identity transform
    Ok(json!({
        "model_id": model_id,
        "input": input,
        "output": input,
        "status": "ok"
    }))
}

fn tool_benchmark(params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let iterations = params
        .get("iterations")
        .and_then(Value::as_i64)
        .unwrap_or(100) as usize;

    let program = parse_vortex(&code)?;

    let start = Instant::now();
    for _ in 0..iterations {
        let _ = interpreter::interpret(&program);
    }
    let elapsed = start.elapsed();

    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let per_iter_ms = total_ms / iterations as f64;

    Ok(json!({
        "iterations": iterations,
        "total_ms": total_ms,
        "per_iteration_ms": per_iter_ms,
        "ops_per_sec": if per_iter_ms > 0.0 { 1000.0 / per_iter_ms } else { 0.0 }
    }))
}

fn tool_explain(params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    let code = get_code(params)?;
    let program = parse_vortex(&code)?;

    let mut functions: Vec<Value> = Vec::new();
    let mut kernels: Vec<Value> = Vec::new();
    let mut structs: Vec<Value> = Vec::new();
    let mut enums: Vec<Value> = Vec::new();

    for item in &program.items {
        match &item.kind {
            crate::ast::ItemKind::Function(f) => {
                let params_list: Vec<String> = f
                    .params
                    .iter()
                    .map(|p| format!("{}: {:?}", p.name.name, p.ty))
                    .collect();
                functions.push(json!({
                    "name": f.name.name,
                    "params": params_list,
                    "return_type": format!("{:?}", f.ret_type),
                }));
            }
            crate::ast::ItemKind::Kernel(k) => {
                kernels.push(json!({ "name": k.name.name }));
            }
            crate::ast::ItemKind::Struct(s) => {
                structs.push(json!({ "name": s.name.name }));
            }
            crate::ast::ItemKind::Enum(e) => {
                enums.push(json!({ "name": e.name.name }));
            }
            _ => {}
        }
    }

    Ok(json!({
        "functions": functions,
        "kernels": kernels,
        "structs": structs,
        "enums": enums,
        "ast_summary": format!("{}", program)
    }))
}

fn tool_gpu_status(_params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    // Probe for NVIDIA GPU via nvidia-smi
    let gpu_info = match std::process::Command::new("nvidia-smi")
        .arg("--query-gpu=name,memory.total,memory.used,utilization.gpu")
        .arg("--format=csv,noheader,nounits")
        .output()
    {
        Ok(out) if out.status.success() => {
            let text = String::from_utf8_lossy(&out.stdout).to_string();
            json!({ "available": true, "driver": "nvidia", "info": text.trim() })
        }
        _ => json!({ "available": false, "reason": "No GPU detected (nvidia-smi not found)" }),
    };

    Ok(gpu_info)
}

fn tool_create_model(params: &Value, state: &mut SessionState) -> Result<Value, String> {
    let arch = params
        .get("architecture")
        .and_then(Value::as_str)
        .ok_or("Missing architecture")?;
    let config = params.get("config").cloned().unwrap_or(json!({}));

    let param_count: usize = match arch {
        "transformer" => config.get("layers").and_then(Value::as_u64).unwrap_or(6) as usize
            * config.get("d_model").and_then(Value::as_u64).unwrap_or(512) as usize
            * 4,
        "ssm" => config.get("d_state").and_then(Value::as_u64).unwrap_or(64) as usize
            * config.get("d_model").and_then(Value::as_u64).unwrap_or(256) as usize,
        "hybrid" => 500_000,
        "ebm" => 250_000,
        other => return Err(format!("Unknown architecture: {}", other)),
    };

    let model_id = state.next_model_id();
    state.models.insert(model_id.clone(), param_count);

    Ok(json!({
        "model_id": model_id,
        "architecture": arch,
        "param_count": param_count,
        "config": config
    }))
}

fn tool_list_builtins(_params: &Value, _state: &mut SessionState) -> Result<Value, String> {
    Ok(json!({
        "math": ["abs", "sqrt", "sin", "cos", "exp", "log", "pow", "min", "max", "floor", "ceil"],
        "tensor": ["zeros", "ones", "rand", "shape", "reshape", "transpose", "matmul", "sum", "mean"],
        "crypto": ["sha256", "keccak256", "secp256k1_mul", "field_add", "field_mul", "field_inv", "msm", "ntt"],
        "io": ["print", "println", "assert", "assert_eq"],
        "gpu": ["launch", "sync", "alloc_device", "memcpy_h2d", "memcpy_d2h"]
    }))
}

// ── MCP Server ─────────────────────────────────────────────────────────

pub struct MCPServer {
    state: SessionState,
}

impl MCPServer {
    pub fn new() -> Self {
        Self {
            state: SessionState::new(),
        }
    }

    /// Process a single JSON-RPC request and return a response (or None for notifications).
    pub fn handle_request(&mut self, input: &str) -> Option<Value> {
        let req: Value = match serde_json::from_str(input) {
            Ok(v) => v,
            Err(_) => {
                return Some(json!({
                    "jsonrpc": "2.0",
                    "id": null,
                    "error": { "code": -32700, "message": "Parse error" }
                }));
            }
        };

        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params").cloned().unwrap_or(json!({}));

        match method {
            // ── MCP lifecycle ─────────────────────────────────
            "initialize" => {
                let result = json!({
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "vortex-mcp",
                        "version": "0.1.0"
                    }
                });
                Some(jsonrpc_ok(id, result))
            }

            "notifications/initialized" => None, // no response for notifications

            "tools/list" => {
                let result = json!({ "tools": tool_descriptors() });
                Some(jsonrpc_ok(id, result))
            }

            "tools/call" => {
                let tool_name = params.get("name").and_then(Value::as_str).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));

                let result = match tool_name {
                    "vortex_run"           => tool_run(&args, &mut self.state),
                    "vortex_typecheck"     => tool_typecheck(&args, &mut self.state),
                    "vortex_codegen"       => tool_codegen(&args, &mut self.state),
                    "vortex_train"         => tool_train(&args, &mut self.state),
                    "vortex_infer"         => tool_infer(&args, &mut self.state),
                    "vortex_benchmark"     => tool_benchmark(&args, &mut self.state),
                    "vortex_explain"       => tool_explain(&args, &mut self.state),
                    "vortex_gpu_status"    => tool_gpu_status(&args, &mut self.state),
                    "vortex_create_model"  => tool_create_model(&args, &mut self.state),
                    "vortex_list_builtins" => tool_list_builtins(&args, &mut self.state),
                    _ => Err(format!("Unknown tool: {}", tool_name)),
                };

                // Record in history
                let summary = match &result {
                    Ok(_) => "ok".to_string(),
                    Err(e) => format!("error: {}", e),
                };
                self.state
                    .history
                    .push((tool_name.to_string(), summary));

                match result {
                    Ok(val) => Some(jsonrpc_ok(
                        id,
                        json!({
                            "content": [{ "type": "text", "text": val.to_string() }]
                        }),
                    )),
                    Err(e) => Some(jsonrpc_ok(
                        id,
                        json!({
                            "content": [{ "type": "text", "text": e }],
                            "isError": true
                        }),
                    )),
                }
            }

            _ => Some(json!({
                "jsonrpc": "2.0",
                "id": id,
                "error": { "code": -32601, "message": format!("Method not found: {}", method) }
            })),
        }
    }

    /// Main event loop – reads JSON-RPC from stdin, writes to stdout.
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut stdout = io::stdout();

        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            if let Some(resp) = self.handle_request(trimmed) {
                let out = serde_json::to_string(&resp).unwrap();
                let _ = writeln!(stdout, "{}", out);
                let _ = stdout.flush();
            }
        }
    }
}

fn jsonrpc_ok(id: Option<Value>, result: Value) -> Value {
    json!({
        "jsonrpc": "2.0",
        "id": id,
        "result": result
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn call(server: &mut MCPServer, req: Value) -> Value {
        let input = serde_json::to_string(&req).unwrap();
        server.handle_request(&input).unwrap()
    }

    #[test]
    fn test_initialize() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));
        let result = &resp["result"];
        assert_eq!(result["serverInfo"]["name"], "vortex-mcp");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[test]
    fn test_tools_list() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}
        }));
        let tools = resp["result"]["tools"].as_array().unwrap();
        assert!(tools.len() >= 10);
        let names: Vec<&str> = tools.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.contains(&"vortex_run"));
        assert!(names.contains(&"vortex_typecheck"));
        assert!(names.contains(&"vortex_codegen"));
    }

    #[test]
    fn test_vortex_run() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 3, "method": "tools/call",
            "params": {
                "name": "vortex_run",
                "arguments": { "code": "fn main() { print(42) }" }
            }
        }));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let val: Value = serde_json::from_str(text).unwrap();
        assert!(val["output"].as_str().unwrap().contains("42"));
    }

    #[test]
    fn test_vortex_typecheck() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 4, "method": "tools/call",
            "params": {
                "name": "vortex_typecheck",
                "arguments": { "code": "fn main() { let x: i32 = 1 }" }
            }
        }));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let val: Value = serde_json::from_str(text).unwrap();
        assert_eq!(val["status"], "OK");
    }

    #[test]
    fn test_vortex_codegen() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 5, "method": "tools/call",
            "params": {
                "name": "vortex_codegen",
                "arguments": { "code": "fn add(a: i64, b: i64) -> i64 { return a + b }" }
            }
        }));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let val: Value = serde_json::from_str(text).unwrap();
        assert!(val["mlir"].as_str().unwrap().contains("func"));
    }

    #[test]
    fn test_vortex_explain() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 6, "method": "tools/call",
            "params": {
                "name": "vortex_explain",
                "arguments": { "code": "fn foo(x: i64) -> i64 { return x + 1 }" }
            }
        }));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let val: Value = serde_json::from_str(text).unwrap();
        let fns = val["functions"].as_array().unwrap();
        assert!(!fns.is_empty());
        assert_eq!(fns[0]["name"], "foo");
    }

    #[test]
    fn test_invalid_json() {
        let mut s = MCPServer::new();
        let resp = s.handle_request("not json at all{{{").unwrap();
        assert_eq!(resp["error"]["code"], -32700);
    }

    #[test]
    fn test_unknown_method() {
        let mut s = MCPServer::new();
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 99, "method": "nonexistent", "params": {}
        }));
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[test]
    fn test_session_state_persists() {
        let mut s = MCPServer::new();

        // Create a model
        let resp = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 10, "method": "tools/call",
            "params": {
                "name": "vortex_create_model",
                "arguments": { "architecture": "transformer" }
            }
        }));
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        let val: Value = serde_json::from_str(text).unwrap();
        let model_id = val["model_id"].as_str().unwrap().to_string();
        assert!(!model_id.is_empty());

        // Use it for inference
        let resp2 = call(&mut s, json!({
            "jsonrpc": "2.0", "id": 11, "method": "tools/call",
            "params": {
                "name": "vortex_infer",
                "arguments": { "model_id": model_id, "input": [1.0, 2.0] }
            }
        }));
        let text2 = resp2["result"]["content"][0]["text"].as_str().unwrap();
        let val2: Value = serde_json::from_str(text2).unwrap();
        assert_eq!(val2["status"], "ok");

        // History should have 2 entries
        assert_eq!(s.state.history.len(), 2);
    }

    #[test]
    fn test_notification_returns_none() {
        let mut s = MCPServer::new();
        let input = serde_json::to_string(&json!({
            "jsonrpc": "2.0", "method": "notifications/initialized"
        })).unwrap();
        assert!(s.handle_request(&input).is_none());
    }
}
