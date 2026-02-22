//! LSP (Language Server Protocol) server for Vortex.
//!
//! Implements JSON-RPC 2.0 over stdin/stdout with support for:
//! - `initialize` / `initialized`
//! - `textDocument/didOpen`, `textDocument/didChange`
//! - `textDocument/hover`
//! - `textDocument/definition`
//! - `shutdown` / `exit`

use crate::lsp::{self, SymbolTable, VortexAnalyzer};
use crate::lexer;
use crate::parser;

use serde_json::{json, Value};
use std::collections::HashMap;
use std::io::{self, BufRead, Read, Write};

/// An open document tracked by the server.
struct Document {
    uri: String,
    text: String,
    version: i64,
}

pub struct LspServer {
    documents: HashMap<String, Document>,
    initialized: bool,
    shutdown_requested: bool,
}

impl LspServer {
    pub fn new() -> Self {
        Self {
            documents: HashMap::new(),
            initialized: false,
            shutdown_requested: false,
        }
    }

    /// Run the LSP server over stdin/stdout using Content-Length framing.
    pub fn run(&mut self) {
        let stdin = io::stdin();
        let mut stdout = io::stdout();
        let mut reader = stdin.lock();

        loop {
            // Read headers (Content-Length framed)
            let content_length = match read_content_length(&mut reader) {
                Some(len) => len,
                None => break, // EOF
            };

            // Read body
            let mut body = vec![0u8; content_length];
            if reader.read_exact(&mut body).is_err() {
                break;
            }
            let body_str = match String::from_utf8(body) {
                Ok(s) => s,
                Err(_) => continue,
            };

            let request: Value = match serde_json::from_str(&body_str) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let responses = self.handle_message(&request);
            for resp in responses {
                send_message(&mut stdout, &resp);
            }

            if self.shutdown_requested {
                break;
            }
        }
    }

    /// Handle a single JSON-RPC message, returning zero or more responses.
    pub fn handle_message(&mut self, msg: &Value) -> Vec<Value> {
        let method = msg.get("method").and_then(Value::as_str).unwrap_or("");
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(json!({}));

        match method {
            "initialize" => {
                self.initialized = true;
                let result = json!({
                    "capabilities": {
                        "textDocumentSync": 1,
                        "hoverProvider": true,
                        "definitionProvider": true,
                        "diagnosticProvider": {
                            "interFileDependencies": false,
                            "workspaceDiagnostics": false
                        }
                    },
                    "serverInfo": {
                        "name": "vortex-lsp",
                        "version": "0.1.0"
                    }
                });
                vec![jsonrpc_ok(id, result)]
            }
            "initialized" => {
                // Notification, no response
                vec![]
            }
            "shutdown" => {
                self.shutdown_requested = true;
                vec![jsonrpc_ok(id, json!(null))]
            }
            "exit" => {
                self.shutdown_requested = true;
                vec![]
            }
            "textDocument/didOpen" => {
                if let Some(td) = params.get("textDocument") {
                    let uri = td.get("uri").and_then(Value::as_str).unwrap_or("").to_string();
                    let text = td.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                    let version = td.get("version").and_then(Value::as_i64).unwrap_or(0);
                    self.documents.insert(uri.clone(), Document {
                        uri: uri.clone(),
                        text: text.clone(),
                        version,
                    });
                    // Publish diagnostics
                    let diags = self.compute_diagnostics(&uri, &text);
                    vec![json!({
                        "jsonrpc": "2.0",
                        "method": "textDocument/publishDiagnostics",
                        "params": {
                            "uri": uri,
                            "diagnostics": diags
                        }
                    })]
                } else {
                    vec![]
                }
            }
            "textDocument/didChange" => {
                if let Some(td) = params.get("textDocument") {
                    let uri = td.get("uri").and_then(Value::as_str).unwrap_or("").to_string();
                    let version = td.get("version").and_then(Value::as_i64).unwrap_or(0);
                    // Full sync: take last content change
                    if let Some(changes) = params.get("contentChanges").and_then(Value::as_array) {
                        if let Some(change) = changes.last() {
                            let text = change.get("text").and_then(Value::as_str).unwrap_or("").to_string();
                            self.documents.insert(uri.clone(), Document {
                                uri: uri.clone(),
                                text: text.clone(),
                                version,
                            });
                            let diags = self.compute_diagnostics(&uri, &text);
                            return vec![json!({
                                "jsonrpc": "2.0",
                                "method": "textDocument/publishDiagnostics",
                                "params": {
                                    "uri": uri,
                                    "diagnostics": diags
                                }
                            })];
                        }
                    }
                }
                vec![]
            }
            "textDocument/hover" => {
                let uri = params.pointer("/textDocument/uri").and_then(Value::as_str).unwrap_or("");
                let line = params.pointer("/position/line").and_then(Value::as_u64).unwrap_or(0) as usize;
                let col = params.pointer("/position/character").and_then(Value::as_u64).unwrap_or(0) as usize;

                if let Some(doc) = self.documents.get(uri) {
                    let tokens = lexer::lex(&doc.text);
                    if let Ok(program) = parser::parse(tokens, 0) {
                        let table = SymbolTable::from_program(&program.items, &doc.text, uri);
                        // LSP uses 0-based lines, our symbol table uses 1-based
                        if let Some(sym) = table.symbol_at(&doc.text, line + 1, col + 1) {
                            let detail = sym.detail.as_deref().unwrap_or("");
                            let contents = format!("**{}** `{}`\n\n{}", sym.kind, sym.name, detail);
                            return vec![jsonrpc_ok(id, json!({
                                "contents": {
                                    "kind": "markdown",
                                    "value": contents
                                }
                            }))];
                        }
                    }
                }
                vec![jsonrpc_ok(id, json!(null))]
            }
            "textDocument/definition" => {
                let uri = params.pointer("/textDocument/uri").and_then(Value::as_str).unwrap_or("");
                let line = params.pointer("/position/line").and_then(Value::as_u64).unwrap_or(0) as usize;
                let col = params.pointer("/position/character").and_then(Value::as_u64).unwrap_or(0) as usize;

                if let Some(doc) = self.documents.get(uri) {
                    let tokens = lexer::lex(&doc.text);
                    if let Ok(program) = parser::parse(tokens, 0) {
                        let table = SymbolTable::from_program(&program.items, &doc.text, uri);
                        if let Some(sym) = table.symbol_at(&doc.text, line + 1, col + 1) {
                            // Find the definition of the symbol
                            if let Some(def) = table.find_definition(&sym.name) {
                                return vec![jsonrpc_ok(id, json!({
                                    "uri": def.file,
                                    "range": {
                                        "start": { "line": def.line - 1, "character": def.col - 1 },
                                        "end": { "line": def.line - 1, "character": def.col - 1 }
                                    }
                                }))];
                            }
                        }
                    }
                }
                vec![jsonrpc_ok(id, json!(null))]
            }
            _ => {
                // Unknown method — if it has an id, respond with method not found
                if let Some(id) = id {
                    vec![json!({
                        "jsonrpc": "2.0",
                        "id": id,
                        "error": {
                            "code": -32601,
                            "message": format!("Method not found: {}", method)
                        }
                    })]
                } else {
                    vec![]
                }
            }
        }
    }

    fn compute_diagnostics(&self, uri: &str, text: &str) -> Vec<Value> {
        let filename = uri_to_filename(uri);
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze(text, &filename);

        analyzer.diagnostics.iter().map(|d| {
            let severity = match d.severity {
                lsp::Severity::Error => 1,
                lsp::Severity::Warning => 2,
                lsp::Severity::Info => 3,
                lsp::Severity::Hint => 4,
            };
            json!({
                "range": {
                    "start": { "line": d.line.saturating_sub(1), "character": d.col.saturating_sub(1) },
                    "end": { "line": d.end_line.saturating_sub(1), "character": d.end_col.saturating_sub(1) }
                },
                "severity": severity,
                "code": d.code,
                "source": "vortex",
                "message": d.message
            })
        }).collect()
    }
}

fn uri_to_filename(uri: &str) -> String {
    uri.strip_prefix("file://").unwrap_or(uri).to_string()
}

fn read_content_length(reader: &mut impl BufRead) -> Option<usize> {
    let mut content_length: Option<usize> = None;
    loop {
        let mut header_line = String::new();
        match reader.read_line(&mut header_line) {
            Ok(0) => return None, // EOF
            Ok(_) => {}
            Err(_) => return None,
        }
        let trimmed = header_line.trim();
        if trimmed.is_empty() {
            // End of headers
            return content_length;
        }
        if let Some(val) = trimmed.strip_prefix("Content-Length:") {
            if let Ok(len) = val.trim().parse::<usize>() {
                content_length = Some(len);
            }
        }
    }
}

fn send_message(stdout: &mut impl Write, msg: &Value) {
    let body = serde_json::to_string(msg).unwrap();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    let _ = stdout.write_all(header.as_bytes());
    let _ = stdout.write_all(body.as_bytes());
    let _ = stdout.flush();
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

    #[test]
    fn test_lsp_initialize_returns_capabilities() {
        let mut server = LspServer::new();
        let req = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "capabilities": {}
            }
        });
        let responses = server.handle_message(&req);
        assert_eq!(responses.len(), 1);
        let result = &responses[0]["result"];
        assert!(result.get("capabilities").is_some());
        let caps = &result["capabilities"];
        assert_eq!(caps["hoverProvider"], json!(true));
        assert_eq!(caps["definitionProvider"], json!(true));
        assert!(result.get("serverInfo").is_some());
        assert_eq!(result["serverInfo"]["name"], "vortex-lsp");
    }

    #[test]
    fn test_lsp_diagnostics_for_errors() {
        let mut server = LspServer::new();
        // Initialize first
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));

        // Open a document with a parse error
        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.vx",
                    "languageId": "vortex",
                    "version": 1,
                    "text": "fn {"
                }
            }
        }));
        assert_eq!(responses.len(), 1);
        let diags = responses[0]["params"]["diagnostics"].as_array().unwrap();
        assert!(!diags.is_empty(), "expected diagnostics for invalid code");
        // Severity 1 = Error
        assert_eq!(diags[0]["severity"], 1);
    }

    #[test]
    fn test_lsp_diagnostics_clean_file() {
        let mut server = LspServer::new();
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));

        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///clean.vx",
                    "languageId": "vortex",
                    "version": 1,
                    "text": "fn add(a: f32, b: f32) -> f32 {\n    return a + b\n}"
                }
            }
        }));
        let diags = responses[0]["params"]["diagnostics"].as_array().unwrap();
        assert!(diags.is_empty(), "clean file should have no diagnostics");
    }

    #[test]
    fn test_lsp_hover() {
        let mut server = LspServer::new();
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));
        server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.vx",
                    "languageId": "vortex",
                    "version": 1,
                    "text": "fn add(a: f32, b: f32) -> f32 {\n    return a + b\n}"
                }
            }
        }));

        // Hover over "add" at line 0, col 3
        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "textDocument/hover",
            "params": {
                "textDocument": { "uri": "file:///test.vx" },
                "position": { "line": 0, "character": 3 }
            }
        }));
        assert_eq!(responses.len(), 1);
        let result = &responses[0]["result"];
        assert!(result.get("contents").is_some(), "hover should return contents");
    }

    #[test]
    fn test_lsp_definition() {
        let mut server = LspServer::new();
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));
        server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.vx",
                    "languageId": "vortex",
                    "version": 1,
                    "text": "fn add(a: f32, b: f32) -> f32 {\n    return a + b\n}"
                }
            }
        }));

        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "textDocument/definition",
            "params": {
                "textDocument": { "uri": "file:///test.vx" },
                "position": { "line": 0, "character": 3 }
            }
        }));
        assert_eq!(responses.len(), 1);
        // Should get a location back
        let result = &responses[0]["result"];
        if !result.is_null() {
            assert!(result.get("uri").is_some());
            assert!(result.get("range").is_some());
        }
    }

    #[test]
    fn test_lsp_shutdown() {
        let mut server = LspServer::new();
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));
        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 2, "method": "shutdown"
        }));
        assert_eq!(responses.len(), 1);
        assert!(server.shutdown_requested);
    }

    #[test]
    fn test_lsp_didchange_updates_diagnostics() {
        let mut server = LspServer::new();
        server.handle_message(&json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}
        }));
        // Open clean file
        server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": "file:///test.vx",
                    "languageId": "vortex",
                    "version": 1,
                    "text": "fn main() {}"
                }
            }
        }));
        // Change to broken code
        let responses = server.handle_message(&json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didChange",
            "params": {
                "textDocument": { "uri": "file:///test.vx", "version": 2 },
                "contentChanges": [{ "text": "fn {" }]
            }
        }));
        assert_eq!(responses.len(), 1);
        let diags = responses[0]["params"]["diagnostics"].as_array().unwrap();
        assert!(!diags.is_empty(), "broken code should have diagnostics after change");
    }
}
