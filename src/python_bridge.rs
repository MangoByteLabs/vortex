/// Python communicates with Vortex via JSON over stdin/stdout.
/// This is the simplest interop that works without external dependencies.

use crate::interpreter::Value;
use crate::{codegen, interpreter, lexer, parser, typeck};
use codespan_reporting::files::SimpleFiles;
use std::io::{self, BufRead, Write};

pub struct PythonBridge {
    pub mode: BridgeMode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BridgeMode {
    Server,
}

#[derive(Debug, Clone, PartialEq)]
pub enum BridgeCommand {
    Eval { source: String },
    RunFile { path: String },
    Parse { source: String },
    Check { source: String },
    Codegen { source: String },
    CallFunction { name: String, args: Vec<String> },
    Quit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BridgeResponse {
    pub success: bool,
    pub result: Option<String>,
    pub error: Option<String>,
    pub diagnostics: Vec<String>,
}

impl PythonBridge {
    pub fn new() -> Self {
        PythonBridge {
            mode: BridgeMode::Server,
        }
    }

    /// Run the bridge server: read JSON commands from stdin, write responses to stdout.
    pub fn serve(&mut self) {
        let stdin = io::stdin();
        let stdout = io::stdout();
        for line in stdin.lock().lines() {
            let line = match line {
                Ok(l) => l,
                Err(_) => break,
            };
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let cmd = match Self::parse_command(trimmed) {
                Ok(c) => c,
                Err(e) => {
                    let resp = BridgeResponse {
                        success: false,
                        result: None,
                        error: Some(e),
                        diagnostics: vec![],
                    };
                    let mut out = stdout.lock();
                    let _ = writeln!(out, "{}", Self::format_response(&resp));
                    let _ = out.flush();
                    continue;
                }
            };
            if cmd == BridgeCommand::Quit {
                let resp = BridgeResponse {
                    success: true,
                    result: Some("bye".to_string()),
                    error: None,
                    diagnostics: vec![],
                };
                let mut out = stdout.lock();
                let _ = writeln!(out, "{}", Self::format_response(&resp));
                let _ = out.flush();
                break;
            }
            let resp = self.execute(cmd);
            let mut out = stdout.lock();
            let _ = writeln!(out, "{}", Self::format_response(&resp));
            let _ = out.flush();
        }
    }

    /// Parse a JSON command string into a BridgeCommand.
    pub fn parse_command(json: &str) -> Result<BridgeCommand, String> {
        // Minimal JSON parsing without serde — we parse the subset we need.
        let json = json.trim();
        if !json.starts_with('{') || !json.ends_with('}') {
            return Err("Expected JSON object".to_string());
        }
        let command = extract_json_string(json, "command")
            .ok_or_else(|| "Missing 'command' field".to_string())?;
        match command.as_str() {
            "eval" => {
                let source = extract_json_string(json, "source")
                    .ok_or_else(|| "Missing 'source' field".to_string())?;
                Ok(BridgeCommand::Eval { source })
            }
            "run_file" => {
                let path = extract_json_string(json, "path")
                    .ok_or_else(|| "Missing 'path' field".to_string())?;
                Ok(BridgeCommand::RunFile { path })
            }
            "parse" => {
                let source = extract_json_string(json, "source")
                    .ok_or_else(|| "Missing 'source' field".to_string())?;
                Ok(BridgeCommand::Parse { source })
            }
            "check" => {
                let source = extract_json_string(json, "source")
                    .ok_or_else(|| "Missing 'source' field".to_string())?;
                Ok(BridgeCommand::Check { source })
            }
            "codegen" => {
                let source = extract_json_string(json, "source")
                    .ok_or_else(|| "Missing 'source' field".to_string())?;
                Ok(BridgeCommand::Codegen { source })
            }
            "call_function" => {
                let name = extract_json_string(json, "name")
                    .ok_or_else(|| "Missing 'name' field".to_string())?;
                let args = extract_json_string_array(json, "args").unwrap_or_default();
                Ok(BridgeCommand::CallFunction { name, args })
            }
            "quit" => Ok(BridgeCommand::Quit),
            other => Err(format!("Unknown command: {}", other)),
        }
    }

    /// Execute a command and return a response.
    pub fn execute(&mut self, cmd: BridgeCommand) -> BridgeResponse {
        match cmd {
            BridgeCommand::Eval { source } => self.do_eval(&source),
            BridgeCommand::RunFile { path } => {
                match std::fs::read_to_string(&path) {
                    Ok(source) => self.do_eval(&source),
                    Err(e) => BridgeResponse {
                        success: false,
                        result: None,
                        error: Some(format!("Cannot read file '{}': {}", path, e)),
                        diagnostics: vec![],
                    },
                }
            }
            BridgeCommand::Parse { source } => self.do_parse(&source),
            BridgeCommand::Check { source } => self.do_check(&source),
            BridgeCommand::Codegen { source } => self.do_codegen(&source),
            BridgeCommand::CallFunction { name, args } => self.do_call(&name, &args),
            BridgeCommand::Quit => BridgeResponse {
                success: true,
                result: Some("bye".to_string()),
                error: None,
                diagnostics: vec![],
            },
        }
    }

    fn do_eval(&self, source: &str) -> BridgeResponse {
        let tokens = lexer::lex(source);
        let mut files = SimpleFiles::new();
        let file_id = files.add("<bridge>".to_string(), source.to_string());
        match parser::parse(tokens, file_id) {
            Ok(program) => match interpreter::interpret(&program) {
                Ok(output) => BridgeResponse {
                    success: true,
                    result: Some(output.join("\n")),
                    error: None,
                    diagnostics: vec![],
                },
                Err(e) => BridgeResponse {
                    success: false,
                    result: None,
                    error: Some(e),
                    diagnostics: vec![],
                },
            },
            Err(diags) => BridgeResponse {
                success: false,
                result: None,
                error: Some("Parse error".to_string()),
                diagnostics: diags.iter().map(|d| format!("{:?}", d.message)).collect(),
            },
        }
    }

    fn do_parse(&self, source: &str) -> BridgeResponse {
        let tokens = lexer::lex(source);
        let mut files = SimpleFiles::new();
        let file_id = files.add("<bridge>".to_string(), source.to_string());
        match parser::parse(tokens, file_id) {
            Ok(program) => BridgeResponse {
                success: true,
                result: Some(format!("{}", program)),
                error: None,
                diagnostics: vec![],
            },
            Err(diags) => BridgeResponse {
                success: false,
                result: None,
                error: Some("Parse error".to_string()),
                diagnostics: diags.iter().map(|d| format!("{:?}", d.message)).collect(),
            },
        }
    }

    fn do_check(&self, source: &str) -> BridgeResponse {
        let tokens = lexer::lex(source);
        let mut files = SimpleFiles::new();
        let file_id = files.add("<bridge>".to_string(), source.to_string());
        match parser::parse(tokens, file_id) {
            Ok(program) => match typeck::check(&program, file_id) {
                Ok(()) => BridgeResponse {
                    success: true,
                    result: Some("Type check passed".to_string()),
                    error: None,
                    diagnostics: vec![],
                },
                Err(diags) => BridgeResponse {
                    success: false,
                    result: None,
                    error: Some("Type check failed".to_string()),
                    diagnostics: diags.iter().map(|d| format!("{:?}", d.message)).collect(),
                },
            },
            Err(diags) => BridgeResponse {
                success: false,
                result: None,
                error: Some("Parse error".to_string()),
                diagnostics: diags.iter().map(|d| format!("{:?}", d.message)).collect(),
            },
        }
    }

    fn do_codegen(&self, source: &str) -> BridgeResponse {
        let tokens = lexer::lex(source);
        let mut files = SimpleFiles::new();
        let file_id = files.add("<bridge>".to_string(), source.to_string());
        match parser::parse(tokens, file_id) {
            Ok(program) => {
                let mlir = codegen::generate_mlir(&program);
                BridgeResponse {
                    success: true,
                    result: Some(mlir),
                    error: None,
                    diagnostics: vec![],
                }
            }
            Err(diags) => BridgeResponse {
                success: false,
                result: None,
                error: Some("Parse error".to_string()),
                diagnostics: diags.iter().map(|d| format!("{:?}", d.message)).collect(),
            },
        }
    }

    fn do_call(&self, name: &str, args: &[String]) -> BridgeResponse {
        // Build a source string that defines the function call
        let args_str = args.join(", ");
        let source = format!("{}({})", name, args_str);
        // We can't call a standalone function without its definition,
        // so return an informative error for now.
        // In practice, users would eval source that includes the function + the call.
        BridgeResponse {
            success: false,
            result: None,
            error: Some(format!(
                "call_function requires a full program context. Use eval with source that defines '{}' and calls it. Attempted: {}",
                name, source
            )),
            diagnostics: vec![],
        }
    }

    /// Format a BridgeResponse as a JSON string.
    pub fn format_response(resp: &BridgeResponse) -> String {
        let success = if resp.success { "true" } else { "false" };
        let result = match &resp.result {
            Some(r) => format!("\"{}\"", escape_json(r)),
            None => "null".to_string(),
        };
        let error = match &resp.error {
            Some(e) => format!("\"{}\"", escape_json(e)),
            None => "null".to_string(),
        };
        let diags: Vec<String> = resp
            .diagnostics
            .iter()
            .map(|d| format!("\"{}\"", escape_json(d)))
            .collect();
        let diags_str = format!("[{}]", diags.join(", "));
        format!(
            "{{\"success\": {}, \"result\": {}, \"error\": {}, \"diagnostics\": {}}}",
            success, result, error, diags_str
        )
    }

    /// Convert a Vortex Value to a JSON string representation.
    pub fn value_to_json(val: &Value) -> String {
        match val {
            Value::Int(n) => format!("{}", n),
            Value::Float(f) => format!("{}", f),
            Value::Bool(b) => format!("{}", b),
            Value::String(s) => format!("\"{}\"", escape_json(s)),
            Value::Array(elems) => {
                let items: Vec<String> = elems.iter().map(Self::value_to_json).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Tuple(elems) => {
                let items: Vec<String> = elems.iter().map(Self::value_to_json).collect();
                format!("[{}]", items.join(", "))
            }
            Value::Void => "null".to_string(),
            Value::Struct { name, fields } => {
                let mut parts = vec![format!("\"__type\": \"{}\"", escape_json(name))];
                for (k, v) in fields {
                    parts.push(format!("\"{}\": {}", escape_json(k), Self::value_to_json(v)));
                }
                format!("{{{}}}", parts.join(", "))
            }
            _ => format!("\"{}\"", escape_json(&format!("{}", val))),
        }
    }

    /// Convert a JSON string to a Vortex Value (basic types only).
    pub fn json_to_value(json: &str) -> Result<Value, String> {
        let json = json.trim();
        if json == "null" {
            return Ok(Value::Void);
        }
        if json == "true" {
            return Ok(Value::Bool(true));
        }
        if json == "false" {
            return Ok(Value::Bool(false));
        }
        if json.starts_with('"') && json.ends_with('"') {
            let inner = &json[1..json.len() - 1];
            return Ok(Value::String(unescape_json(inner)));
        }
        if json.starts_with('[') && json.ends_with(']') {
            let inner = &json[1..json.len() - 1].trim();
            if inner.is_empty() {
                return Ok(Value::Array(vec![]));
            }
            let items = split_json_array(inner);
            let mut vals = Vec::new();
            for item in items {
                vals.push(Self::json_to_value(item.trim())?);
            }
            return Ok(Value::Array(vals));
        }
        // Try integer
        if let Ok(n) = json.parse::<i128>() {
            return Ok(Value::Int(n));
        }
        // Try float
        if let Ok(f) = json.parse::<f64>() {
            return Ok(Value::Float(f));
        }
        Err(format!("Cannot convert JSON to Value: {}", json))
    }
}

// --- JSON utility helpers (no serde dependency) ---

fn escape_json(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c => out.push(c),
        }
    }
    out
}

fn unescape_json(s: &str) -> String {
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
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// Extract a string value for a given key from a JSON object (simple implementation).
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    // Skip whitespace and colon
    let after_colon = after_key.trim_start().strip_prefix(':')?;
    let after_colon = after_colon.trim_start();
    if after_colon.starts_with('"') {
        // Find the closing quote, handling escapes
        let content = &after_colon[1..];
        let mut end = 0;
        let mut escaped = false;
        for (i, c) in content.char_indices() {
            if escaped {
                escaped = false;
                continue;
            }
            if c == '\\' {
                escaped = true;
                continue;
            }
            if c == '"' {
                end = i;
                break;
            }
        }
        Some(unescape_json(&content[..end]))
    } else {
        None
    }
}

/// Extract a string array for a given key from a JSON object.
fn extract_json_string_array(json: &str, key: &str) -> Option<Vec<String>> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    if !after_colon.starts_with('[') {
        return None;
    }
    // Find matching ]
    let mut depth = 0;
    let mut end = 0;
    for (i, c) in after_colon.char_indices() {
        match c {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    end = i;
                    break;
                }
            }
            _ => {}
        }
    }
    let inner = &after_colon[1..end].trim();
    if inner.is_empty() {
        return Some(vec![]);
    }
    let items = split_json_array(inner);
    let mut result = Vec::new();
    for item in items {
        let item = item.trim();
        if item.starts_with('"') && item.ends_with('"') {
            result.push(unescape_json(&item[1..item.len() - 1]));
        } else {
            result.push(item.to_string());
        }
    }
    Some(result)
}

/// Split a JSON array's inner content by commas (respecting nesting).
fn split_json_array(s: &str) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0;
    let mut in_string = false;
    let mut escaped = false;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if escaped {
            escaped = false;
            continue;
        }
        if c == '\\' && in_string {
            escaped = true;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
            continue;
        }
        if in_string {
            continue;
        }
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                items.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    items.push(&s[start..]);
    items
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_command_eval() {
        let json = r#"{"command": "eval", "source": "let x = 42\nprintln(to_string(x))"}"#;
        let cmd = PythonBridge::parse_command(json).unwrap();
        match cmd {
            BridgeCommand::Eval { source } => {
                assert!(source.contains("let x = 42"));
            }
            _ => panic!("Expected Eval command"),
        }
    }

    #[test]
    fn test_parse_command_invalid_json() {
        let result = PythonBridge::parse_command("not json");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_command_missing_command() {
        let result = PythonBridge::parse_command(r#"{"source": "hello"}"#);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_command_quit() {
        let cmd = PythonBridge::parse_command(r#"{"command": "quit"}"#).unwrap();
        assert_eq!(cmd, BridgeCommand::Quit);
    }

    #[test]
    fn test_value_to_json_int() {
        assert_eq!(PythonBridge::value_to_json(&Value::Int(42)), "42");
    }

    #[test]
    fn test_value_to_json_float() {
        let j = PythonBridge::value_to_json(&Value::Float(3.14));
        assert!(j.starts_with("3.14"));
    }

    #[test]
    fn test_value_to_json_bool() {
        assert_eq!(PythonBridge::value_to_json(&Value::Bool(true)), "true");
        assert_eq!(PythonBridge::value_to_json(&Value::Bool(false)), "false");
    }

    #[test]
    fn test_value_to_json_string() {
        assert_eq!(
            PythonBridge::value_to_json(&Value::String("hello".to_string())),
            "\"hello\""
        );
    }

    #[test]
    fn test_value_to_json_array() {
        let arr = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(PythonBridge::value_to_json(&arr), "[1, 2, 3]");
    }

    #[test]
    fn test_json_to_value_roundtrip() {
        // Int
        let v = PythonBridge::json_to_value("42").unwrap();
        assert!(matches!(v, Value::Int(42)));

        // Float
        let v = PythonBridge::json_to_value("3.14").unwrap();
        assert!(matches!(v, Value::Float(f) if (f - 3.14).abs() < 1e-10));

        // Bool
        let v = PythonBridge::json_to_value("true").unwrap();
        assert!(matches!(v, Value::Bool(true)));

        // String
        let v = PythonBridge::json_to_value("\"hello\"").unwrap();
        assert!(matches!(v, Value::String(ref s) if s == "hello"));

        // Null
        let v = PythonBridge::json_to_value("null").unwrap();
        assert!(matches!(v, Value::Void));

        // Array
        let v = PythonBridge::json_to_value("[1, 2, 3]").unwrap();
        assert!(matches!(v, Value::Array(ref a) if a.len() == 3));
    }

    #[test]
    fn test_execute_eval() {
        let mut bridge = PythonBridge::new();
        let resp = bridge.execute(BridgeCommand::Eval {
            source: "fn main() {\n  println(to_string(1 + 2))\n}".to_string(),
        });
        assert!(resp.success);
        assert!(resp.result.is_some());
    }

    #[test]
    fn test_execute_parse() {
        let mut bridge = PythonBridge::new();
        let resp = bridge.execute(BridgeCommand::Parse {
            source: "fn add(a: int, b: int) -> int {\n  return a + b\n}".to_string(),
        });
        assert!(resp.success);
        let result = resp.result.unwrap();
        assert!(!result.is_empty());
        assert!(result.contains("add"));
    }

    #[test]
    fn test_format_response_valid_json() {
        let resp = BridgeResponse {
            success: true,
            result: Some("hello".to_string()),
            error: None,
            diagnostics: vec![],
        };
        let json = PythonBridge::format_response(&resp);
        assert!(json.contains("\"success\": true"));
        assert!(json.contains("\"result\": \"hello\""));
        assert!(json.contains("\"error\": null"));
        assert!(json.contains("\"diagnostics\": []"));
        // Verify it starts/ends as JSON object
        assert!(json.starts_with('{'));
        assert!(json.ends_with('}'));
    }
}
