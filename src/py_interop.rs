/// Python interop for Vortex — subprocess-based bridge to Python3.
///
/// Spawns a Python3 subprocess running an embedded helper script that accepts
/// JSON commands on stdin and returns JSON results on stdout. Supports:
/// - eval/exec Python code
/// - import modules
/// - get/set tensors (numpy/torch arrays)
/// - call Python functions
/// - load PyTorch models
///
/// All communication is JSON over pipes — no shared memory, no FFI.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::{Arc, Mutex};

/// The Python helper script embedded as a string constant.
/// It runs in the subprocess, accepting JSON commands on stdin.
const PYTHON_HELPER: &str = r#"
import sys, json, importlib, traceback

state = {}

def serialize(obj):
    """Convert a Python object to a JSON-safe representation."""
    if obj is None:
        return None
    if isinstance(obj, (int, float, bool, str)):
        return obj
    if isinstance(obj, (list, tuple)):
        return [serialize(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): serialize(v) for k, v in obj.items()}
    # numpy array
    try:
        import numpy as np
        if isinstance(obj, np.ndarray):
            return {"__tensor__": True, "data": obj.flatten().tolist(), "shape": list(obj.shape), "dtype": str(obj.dtype)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
    except ImportError:
        pass
    # torch tensor
    try:
        import torch
        if isinstance(obj, torch.Tensor):
            arr = obj.detach().cpu().numpy()
            return {"__tensor__": True, "data": arr.flatten().tolist(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
    except ImportError:
        pass
    return str(obj)

while True:
    try:
        line = sys.stdin.readline()
    except Exception:
        break
    if not line:
        break
    line = line.strip()
    if not line:
        continue
    try:
        cmd = json.loads(line)
    except json.JSONDecodeError as e:
        print(json.dumps({"error": f"JSON parse error: {e}"}), flush=True)
        continue

    try:
        cmd_type = cmd.get("type", "")
        if cmd_type == "eval":
            result = eval(cmd["code"], state)
            print(json.dumps({"ok": serialize(result)}), flush=True)
        elif cmd_type == "exec":
            exec(cmd["code"], state)
            print(json.dumps({"ok": None}), flush=True)
        elif cmd_type == "import":
            mod = importlib.import_module(cmd["module"])
            alias = cmd.get("as", cmd["module"])
            state[alias] = mod
            print(json.dumps({"ok": True}), flush=True)
        elif cmd_type == "get_tensor":
            obj = eval(cmd["name"], state)
            # Convert torch tensor to numpy
            try:
                import torch
                if isinstance(obj, torch.Tensor):
                    obj = obj.detach().cpu().numpy()
            except ImportError:
                pass
            import numpy as np
            if isinstance(obj, np.ndarray):
                print(json.dumps({"ok": {"data": obj.flatten().tolist(), "shape": list(obj.shape), "dtype": str(obj.dtype)}}), flush=True)
            else:
                print(json.dumps({"error": f"'{cmd['name']}' is not a tensor/array, got {type(obj).__name__}"}), flush=True)
        elif cmd_type == "set_tensor":
            import numpy as np
            arr = np.array(cmd["data"], dtype=np.float64).reshape(cmd["shape"])
            state[cmd["name"]] = arr
            print(json.dumps({"ok": True}), flush=True)
        elif cmd_type == "call":
            fn = eval(cmd["func"], state)
            args = cmd.get("args", [])
            kwargs = cmd.get("kwargs", {})
            result = fn(*args, **kwargs)
            print(json.dumps({"ok": serialize(result)}), flush=True)
        elif cmd_type == "get_var":
            obj = eval(cmd["name"], state)
            print(json.dumps({"ok": serialize(obj)}), flush=True)
        elif cmd_type == "set_var":
            state[cmd["name"]] = cmd["value"]
            print(json.dumps({"ok": True}), flush=True)
        elif cmd_type == "load_model":
            import torch
            sd = torch.load(cmd["path"], map_location="cpu", weights_only=True)
            if hasattr(sd, 'state_dict'):
                sd = sd.state_dict()
            weights = {}
            for k, v in sd.items():
                arr = v.detach().cpu().numpy()
                weights[k] = {"data": arr.flatten().tolist(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
            print(json.dumps({"ok": weights}), flush=True)
        elif cmd_type == "ping":
            print(json.dumps({"ok": "pong"}), flush=True)
        elif cmd_type == "shutdown":
            print(json.dumps({"ok": "bye"}), flush=True)
            break
        else:
            print(json.dumps({"error": f"Unknown command type: {cmd_type}"}), flush=True)
    except Exception:
        print(json.dumps({"error": traceback.format_exc()}), flush=True)
"#;

/// Manages a Python3 subprocess for interop.
pub struct PythonInterop {
    process: Option<Child>,
    stdin: Option<ChildStdin>,
    stdout: Option<BufReader<ChildStdout>>,
    initialized: bool,
}

/// Shared, thread-safe handle to a PythonInterop instance.
pub type SharedPython = Arc<Mutex<PythonInterop>>;

impl PythonInterop {
    /// Check if python3 is available on the system.
    pub fn python_available() -> bool {
        Command::new("python3")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }

    /// Start a new Python3 subprocess running the helper script.
    pub fn start() -> Result<Self, String> {
        if !Self::python_available() {
            return Err("python3 is not available on this system".to_string());
        }

        let mut child = Command::new("python3")
            .arg("-u") // unbuffered
            .arg("-c")
            .arg(PYTHON_HELPER)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| format!("Failed to spawn python3: {}", e))?;

        let stdin = child.stdin.take().ok_or("Failed to get stdin")?;
        let stdout = child.stdout.take().ok_or("Failed to get stdout")?;

        let mut interop = PythonInterop {
            process: Some(child),
            stdin: Some(stdin),
            stdout: Some(BufReader::new(stdout)),
            initialized: false,
        };

        // Verify the subprocess is alive with a ping
        match interop.send_command(r#"{"type":"ping"}"#) {
            Ok(resp) => {
                if resp.contains("pong") {
                    interop.initialized = true;
                    Ok(interop)
                } else {
                    Err(format!("Python subprocess ping failed: {}", resp))
                }
            }
            Err(e) => Err(format!("Python subprocess not responding: {}", e)),
        }
    }

    /// Send a raw JSON command and read the JSON response line.
    fn send_command(&mut self, json_cmd: &str) -> Result<String, String> {
        let stdin = self.stdin.as_mut().ok_or("Python subprocess stdin closed")?;
        let stdout = self.stdout.as_mut().ok_or("Python subprocess stdout closed")?;

        writeln!(stdin, "{}", json_cmd)
            .map_err(|e| format!("Failed to write to Python: {}", e))?;
        stdin
            .flush()
            .map_err(|e| format!("Failed to flush Python stdin: {}", e))?;

        let mut line = String::new();
        stdout
            .read_line(&mut line)
            .map_err(|e| format!("Failed to read from Python: {}", e))?;

        if line.is_empty() {
            return Err("Python subprocess closed unexpectedly".to_string());
        }

        Ok(line.trim().to_string())
    }

    /// Parse a JSON response, extracting either the "ok" value or the "error".
    fn parse_response(resp: &str) -> Result<String, String> {
        // Minimal JSON parsing: look for "ok" or "error" keys
        let trimmed = resp.trim();
        if trimmed.starts_with('{') {
            if let Some(err_start) = find_json_key(trimmed, "error") {
                let err_val = extract_json_value(trimmed, err_start);
                if err_val != "null" {
                    return Err(unquote(&err_val));
                }
            }
            if let Some(ok_start) = find_json_key(trimmed, "ok") {
                let ok_val = extract_json_value(trimmed, ok_start);
                return Ok(ok_val);
            }
        }
        Err(format!("Unexpected response from Python: {}", resp))
    }

    /// Evaluate a Python expression and return the serialized result.
    pub fn eval(&mut self, code: &str) -> Result<String, String> {
        let cmd = format!(
            r#"{{"type":"eval","code":{}}}"#,
            json_string(code)
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)
    }

    /// Execute Python statements (no return value).
    pub fn exec(&mut self, code: &str) -> Result<(), String> {
        let cmd = format!(
            r#"{{"type":"exec","code":{}}}"#,
            json_string(code)
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)?;
        Ok(())
    }

    /// Import a Python module with an optional alias.
    pub fn import(&mut self, module: &str, alias: &str) -> Result<(), String> {
        let cmd = format!(
            r#"{{"type":"import","module":{},"as":{}}}"#,
            json_string(module),
            json_string(alias)
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)?;
        Ok(())
    }

    /// Get a tensor (numpy array or torch tensor) from Python state.
    /// Returns (flattened data, shape).
    pub fn get_tensor(&mut self, name: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
        let cmd = format!(
            r#"{{"type":"get_tensor","name":{}}}"#,
            json_string(name)
        );
        let resp = self.send_command(&cmd)?;
        let ok_val = Self::parse_response(&resp)?;
        // Parse the tensor object: {"data": [...], "shape": [...], "dtype": "..."}
        parse_tensor_json(&ok_val)
    }

    /// Set a tensor in Python state from Vortex data.
    pub fn set_tensor(
        &mut self,
        name: &str,
        data: &[f64],
        shape: &[usize],
    ) -> Result<(), String> {
        let data_str: Vec<String> = data.iter().map(|x| format!("{}", x)).collect();
        let shape_str: Vec<String> = shape.iter().map(|x| format!("{}", x)).collect();
        let cmd = format!(
            r#"{{"type":"set_tensor","name":{},"data":[{}],"shape":[{}]}}"#,
            json_string(name),
            data_str.join(","),
            shape_str.join(",")
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)?;
        Ok(())
    }

    /// Call a Python function with JSON-serializable arguments.
    pub fn call(&mut self, func: &str, args: &[Value]) -> Result<String, String> {
        let args_json: Vec<String> = args.iter().map(|v| value_to_py_json(v)).collect();
        let cmd = format!(
            r#"{{"type":"call","func":{},"args":[{}]}}"#,
            json_string(func),
            args_json.join(",")
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)
    }

    /// Get a variable from Python state.
    pub fn get_var(&mut self, name: &str) -> Result<String, String> {
        let cmd = format!(
            r#"{{"type":"get_var","name":{}}}"#,
            json_string(name)
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)
    }

    /// Set a variable in Python state.
    pub fn set_var(&mut self, name: &str, value: &Value) -> Result<(), String> {
        let cmd = format!(
            r#"{{"type":"set_var","name":{},"value":{}}}"#,
            json_string(name),
            value_to_py_json(value)
        );
        let resp = self.send_command(&cmd)?;
        Self::parse_response(&resp)?;
        Ok(())
    }

    /// Load a PyTorch model checkpoint and return weight tensors.
    /// Returns a map of layer_name -> (data, shape).
    pub fn load_pytorch_weights(
        &mut self,
        path: &str,
    ) -> Result<HashMap<String, (Vec<f64>, Vec<usize>)>, String> {
        let cmd = format!(
            r#"{{"type":"load_model","path":{}}}"#,
            json_string(path)
        );
        let resp = self.send_command(&cmd)?;
        let ok_val = Self::parse_response(&resp)?;
        parse_weights_json(&ok_val)
    }

    /// Gracefully shut down the Python subprocess.
    pub fn stop(&mut self) {
        if self.initialized {
            let _ = self.send_command(r#"{"type":"shutdown"}"#);
            self.initialized = false;
        }
        // Drop stdin to signal EOF
        self.stdin.take();
        if let Some(mut child) = self.process.take() {
            let _ = child.wait();
        }
    }

    /// Check if the interop is active.
    pub fn is_active(&self) -> bool {
        self.initialized
    }
}

impl Drop for PythonInterop {
    fn drop(&mut self) {
        self.stop();
    }
}

// ---------------------------------------------------------------------------
// JSON helpers (no serde dependency)
// ---------------------------------------------------------------------------

/// Escape a string for JSON.
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 8);
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Produce a JSON string literal (with quotes).
fn json_string(s: &str) -> String {
    format!("\"{}\"", json_escape(s))
}

/// Convert a Vortex Value to a JSON value for sending to Python.
fn value_to_py_json(val: &Value) -> String {
    match val {
        Value::Int(n) => format!("{}", n),
        Value::Float(f) => {
            if f.is_nan() {
                "null".to_string()
            } else if f.is_infinite() {
                if *f > 0.0 { "1e308".to_string() } else { "-1e308".to_string() }
            } else {
                format!("{}", f)
            }
        }
        Value::Bool(b) => if *b { "true".to_string() } else { "false".to_string() },
        Value::String(s) => json_string(s),
        Value::Array(elems) => {
            let items: Vec<String> = elems.iter().map(|v| value_to_py_json(v)).collect();
            format!("[{}]", items.join(","))
        }
        Value::Tuple(elems) => {
            let items: Vec<String> = elems.iter().map(|v| value_to_py_json(v)).collect();
            format!("[{}]", items.join(","))
        }
        Value::HashMap(map) => {
            let items: Vec<String> = map
                .iter()
                .map(|(k, v)| format!("{}:{}", json_string(k), value_to_py_json(v)))
                .collect();
            format!("{{{}}}", items.join(","))
        }
        Value::Void => "null".to_string(),
        _ => json_string(&format!("{}", val)),
    }
}

/// Find the byte offset of the value for a given key in a JSON object string.
fn find_json_key(json: &str, key: &str) -> Option<usize> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)?;
    let after = &json[pos + needle.len()..];
    let colon = after.find(':')?;
    Some(pos + needle.len() + colon + 1)
}

/// Extract a JSON value starting at the given byte offset.
/// Returns the raw JSON substring for the value.
fn extract_json_value(json: &str, start: usize) -> String {
    let s = json[start..].trim_start();
    let _offset = json.len() - json[start..].len() + (json[start..].len() - s.len());
    if s.is_empty() {
        return "null".to_string();
    }
    let first = s.as_bytes()[0];
    match first {
        b'"' => {
            // String: find closing quote
            let mut i = 1;
            let bytes = s.as_bytes();
            while i < bytes.len() {
                if bytes[i] == b'\\' {
                    i += 2;
                    continue;
                }
                if bytes[i] == b'"' {
                    return s[..i + 1].to_string();
                }
                i += 1;
            }
            s.to_string()
        }
        b'{' | b'[' => {
            // Object or array: find matching brace
            let open = first;
            let close = if first == b'{' { b'}' } else { b']' };
            let mut depth = 0;
            let mut in_str = false;
            let mut esc = false;
            let bytes = s.as_bytes();
            for i in 0..bytes.len() {
                if esc {
                    esc = false;
                    continue;
                }
                if bytes[i] == b'\\' && in_str {
                    esc = true;
                    continue;
                }
                if bytes[i] == b'"' {
                    in_str = !in_str;
                    continue;
                }
                if in_str {
                    continue;
                }
                if bytes[i] == open {
                    depth += 1;
                } else if bytes[i] == close {
                    depth -= 1;
                    if depth == 0 {
                        return s[..i + 1].to_string();
                    }
                }
            }
            s.to_string()
        }
        _ => {
            // Number, bool, null: read until comma, brace, or end
            let end = s
                .find(|c: char| c == ',' || c == '}' || c == ']')
                .unwrap_or(s.len());
            s[..end].trim().to_string()
        }
    }
}

/// Remove surrounding quotes from a JSON string value.
fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.starts_with('"') && s.ends_with('"') && s.len() >= 2 {
        let inner = &s[1..s.len() - 1];
        // Unescape
        let mut out = String::new();
        let mut chars = inner.chars();
        while let Some(c) = chars.next() {
            if c == '\\' {
                match chars.next() {
                    Some('n') => out.push('\n'),
                    Some('r') => out.push('\r'),
                    Some('t') => out.push('\t'),
                    Some('"') => out.push('"'),
                    Some('\\') => out.push('\\'),
                    Some(o) => {
                        out.push('\\');
                        out.push(o);
                    }
                    None => out.push('\\'),
                }
            } else {
                out.push(c);
            }
        }
        out
    } else {
        s.to_string()
    }
}

/// Parse a tensor JSON object: {"data": [...], "shape": [...], "dtype": "..."}
fn parse_tensor_json(json: &str) -> Result<(Vec<f64>, Vec<usize>), String> {
    let json = json.trim();
    // Extract data array
    let data_start = find_json_key(json, "data")
        .ok_or_else(|| "Missing 'data' in tensor response".to_string())?;
    let data_val = extract_json_value(json, data_start);
    let data = parse_number_array(&data_val)?;

    // Extract shape array
    let shape_start = find_json_key(json, "shape")
        .ok_or_else(|| "Missing 'shape' in tensor response".to_string())?;
    let shape_val = extract_json_value(json, shape_start);
    let shape: Vec<usize> = parse_number_array(&shape_val)?
        .iter()
        .map(|x| *x as usize)
        .collect();

    Ok((data, shape))
}

/// Parse a JSON array of numbers.
fn parse_number_array(json: &str) -> Result<Vec<f64>, String> {
    let json = json.trim();
    if !json.starts_with('[') || !json.ends_with(']') {
        return Err(format!("Expected JSON array, got: {}", json));
    }
    let inner = json[1..json.len() - 1].trim();
    if inner.is_empty() {
        return Ok(vec![]);
    }
    let mut result = Vec::new();
    for item in split_top_level(inner, ',') {
        let item = item.trim();
        if item.is_empty() {
            continue;
        }
        let val: f64 = item
            .parse()
            .map_err(|e| format!("Cannot parse '{}' as number: {}", item, e))?;
        result.push(val);
    }
    Ok(result)
}

/// Split string by delimiter at the top level (not inside brackets/braces/strings).
fn split_top_level(s: &str, delim: char) -> Vec<&str> {
    let mut items = Vec::new();
    let mut depth = 0;
    let mut in_str = false;
    let mut esc = false;
    let mut start = 0;
    for (i, c) in s.char_indices() {
        if esc {
            esc = false;
            continue;
        }
        if c == '\\' && in_str {
            esc = true;
            continue;
        }
        if c == '"' {
            in_str = !in_str;
            continue;
        }
        if in_str {
            continue;
        }
        match c {
            '[' | '{' => depth += 1,
            ']' | '}' => depth -= 1,
            c2 if c2 == delim && depth == 0 => {
                items.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    items.push(&s[start..]);
    items
}

/// Parse weights JSON: {"layer_name": {"data": [...], "shape": [...], "dtype": "..."}, ...}
fn parse_weights_json(json: &str) -> Result<HashMap<String, (Vec<f64>, Vec<usize>)>, String> {
    let json = json.trim();
    if !json.starts_with('{') || !json.ends_with('}') {
        return Err("Expected JSON object for weights".to_string());
    }
    let inner = &json[1..json.len() - 1];
    let mut weights = HashMap::new();

    // Split by top-level key-value pairs
    let pairs = split_top_level(inner, ',');
    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        // Find key
        if !pair.starts_with('"') {
            continue;
        }
        let key_end = pair[1..].find('"').map(|i| i + 1);
        let key_end = match key_end {
            Some(e) => e,
            None => continue,
        };
        let key = pair[1..key_end].to_string();
        // Find colon
        let rest = &pair[key_end + 1..];
        let colon = match rest.find(':') {
            Some(c) => c,
            None => continue,
        };
        let val_str = rest[colon + 1..].trim();
        match parse_tensor_json(val_str) {
            Ok((data, shape)) => {
                weights.insert(key, (data, shape));
            }
            Err(_) => continue,
        }
    }
    Ok(weights)
}

/// Convert a Python JSON response back to a Vortex Value.
pub fn py_json_to_value(json: &str) -> Value {
    let json = json.trim();
    if json == "null" || json == "None" {
        return Value::Void;
    }
    if json == "true" || json == "True" {
        return Value::Bool(true);
    }
    if json == "false" || json == "False" {
        return Value::Bool(false);
    }
    if json.starts_with('"') && json.ends_with('"') {
        return Value::String(unquote(json));
    }
    if json.starts_with('[') && json.ends_with(']') {
        let inner = json[1..json.len() - 1].trim();
        if inner.is_empty() {
            return Value::Array(vec![]);
        }
        let items = split_top_level(inner, ',');
        let vals: Vec<Value> = items.iter().map(|s| py_json_to_value(s.trim())).collect();
        return Value::Array(vals);
    }
    if json.starts_with('{') && json.ends_with('}') {
        // Check if it's a tensor
        if json.contains("\"__tensor__\"") {
            if let Ok((data, shape)) = parse_tensor_json(json) {
                let data_vals: Vec<Value> = data.iter().map(|x| Value::Float(*x)).collect();
                let mut fields = HashMap::new();
                fields.insert("data".to_string(), Value::Array(data_vals));
                fields.insert(
                    "shape".to_string(),
                    Value::Array(shape.iter().map(|s| Value::Int(*s as i128)).collect()),
                );
                return Value::Struct {
                    name: "Tensor".to_string(),
                    fields,
                };
            }
        }
        // Generic dict -> HashMap
        let inner = &json[1..json.len() - 1];
        let pairs = split_top_level(inner, ',');
        let mut map = HashMap::new();
        for pair in pairs {
            let pair = pair.trim();
            if pair.is_empty() {
                continue;
            }
            let colon = match pair.find(':') {
                Some(c) => c,
                None => continue,
            };
            let key = unquote(pair[..colon].trim());
            let val = py_json_to_value(pair[colon + 1..].trim());
            map.insert(key, val);
        }
        return Value::HashMap(map);
    }
    // Try integer
    if let Ok(n) = json.parse::<i128>() {
        return Value::Int(n);
    }
    // Try float
    if let Ok(f) = json.parse::<f64>() {
        return Value::Float(f);
    }
    Value::String(json.to_string())
}

// ---------------------------------------------------------------------------
// Global Python instance (lazy, shared)
// ---------------------------------------------------------------------------

use std::sync::OnceLock;

static GLOBAL_PYTHON: OnceLock<Mutex<Option<PythonInterop>>> = OnceLock::new();

fn get_or_start_python() -> Result<std::sync::MutexGuard<'static, Option<PythonInterop>>, String> {
    let mtx = GLOBAL_PYTHON.get_or_init(|| Mutex::new(None));
    let mut guard = mtx.lock().map_err(|e| format!("Python mutex poisoned: {}", e))?;
    if guard.is_none() {
        *guard = Some(PythonInterop::start()?);
    }
    Ok(guard)
}

fn with_python<F, R>(f: F) -> Result<R, String>
where
    F: FnOnce(&mut PythonInterop) -> Result<R, String>,
{
    let mut guard = get_or_start_python()?;
    let py = guard
        .as_mut()
        .ok_or_else(|| "Python not initialized".to_string())?;
    f(py)
}

// ---------------------------------------------------------------------------
// Interpreter builtins
// ---------------------------------------------------------------------------

pub fn builtin_py_available(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Bool(PythonInterop::python_available()))
}

pub fn builtin_py_import(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_import requires at least 1 argument: module name".to_string());
    }
    let module = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_import: module name must be a string".to_string()),
    };
    let alias = if args.len() > 1 {
        match &args[1] {
            Value::String(s) => s.clone(),
            _ => module.clone(),
        }
    } else {
        module.clone()
    };
    with_python(|py| py.import(&module, &alias))?;
    Ok(Value::Void)
}

pub fn builtin_py_eval(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_eval requires 1 argument: code string".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_eval: code must be a string".to_string()),
    };
    let result = with_python(|py| py.eval(&code))?;
    Ok(py_json_to_value(&result))
}

pub fn builtin_py_exec(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_exec requires 1 argument: code string".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_exec: code must be a string".to_string()),
    };
    with_python(|py| py.exec(&code))?;
    Ok(Value::Void)
}

pub fn builtin_py_call(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_call requires at least 1 argument: function name".to_string());
    }
    let func = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_call: function name must be a string".to_string()),
    };
    let call_args = if args.len() > 1 { &args[1..] } else { &[] };
    let result = with_python(|py| py.call(&func, call_args))?;
    Ok(py_json_to_value(&result))
}

pub fn builtin_py_get(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_get requires 1 argument: variable name".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_get: name must be a string".to_string()),
    };
    let result = with_python(|py| py.get_var(&name))?;
    Ok(py_json_to_value(&result))
}

pub fn builtin_py_set(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("py_set requires 2 arguments: name, value".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_set: name must be a string".to_string()),
    };
    with_python(|py| py.set_var(&name, &args[1]))?;
    Ok(Value::Void)
}

pub fn builtin_py_tensor_get(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_tensor_get requires 1 argument: tensor name/expression".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_tensor_get: name must be a string".to_string()),
    };
    let (data, shape) = with_python(|py| py.get_tensor(&name))?;
    let data_vals: Vec<Value> = data.iter().map(|x| Value::Float(*x)).collect();
    let shape_vals: Vec<Value> = shape.iter().map(|s| Value::Int(*s as i128)).collect();
    let mut fields = HashMap::new();
    fields.insert("data".to_string(), Value::Array(data_vals));
    fields.insert("shape".to_string(), Value::Array(shape_vals));
    Ok(Value::Struct {
        name: "Tensor".to_string(),
        fields,
    })
}

pub fn builtin_py_tensor_set(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("py_tensor_set requires 3 arguments: name, data (array), shape (array)".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_tensor_set: name must be a string".to_string()),
    };
    let data: Vec<f64> = match &args[1] {
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Float(f) => Ok(*f),
                Value::Int(n) => Ok(*n as f64),
                _ => Err(format!("py_tensor_set: data element not a number: {}", v)),
            })
            .collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("py_tensor_set: data must be an array".to_string()),
    };
    let shape: Vec<usize> = match &args[2] {
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Int(n) => Ok(*n as usize),
                _ => Err("py_tensor_set: shape elements must be integers".to_string()),
            })
            .collect::<Result<Vec<usize>, String>>()?,
        _ => return Err("py_tensor_set: shape must be an array".to_string()),
    };
    with_python(|py| py.set_tensor(&name, &data, &shape))?;
    Ok(Value::Void)
}

pub fn builtin_py_load_model(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("py_load_model requires 1 argument: model path".to_string());
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("py_load_model: path must be a string".to_string()),
    };
    let weights = with_python(|py| py.load_pytorch_weights(&path))?;
    // Convert to a Vortex HashMap of Tensor structs
    let mut map = HashMap::new();
    for (name, (data, shape)) in weights {
        let data_vals: Vec<Value> = data.iter().map(|x| Value::Float(*x)).collect();
        let shape_vals: Vec<Value> = shape.iter().map(|s| Value::Int(*s as i128)).collect();
        let mut fields = HashMap::new();
        fields.insert("data".to_string(), Value::Array(data_vals));
        fields.insert("shape".to_string(), Value::Array(shape_vals));
        map.insert(
            name,
            Value::Struct {
                name: "Tensor".to_string(),
                fields,
            },
        );
    }
    Ok(Value::HashMap(map))
}

/// Register all Python interop builtins into the interpreter environment.
pub fn register_builtins(env: &mut Env) {
    env.functions.insert("py_available".to_string(), FnDef::Builtin(builtin_py_available));
    env.functions.insert("py_import".to_string(), FnDef::Builtin(builtin_py_import));
    env.functions.insert("py_eval".to_string(), FnDef::Builtin(builtin_py_eval));
    env.functions.insert("py_exec".to_string(), FnDef::Builtin(builtin_py_exec));
    env.functions.insert("py_call".to_string(), FnDef::Builtin(builtin_py_call));
    env.functions.insert("py_get".to_string(), FnDef::Builtin(builtin_py_get));
    env.functions.insert("py_set".to_string(), FnDef::Builtin(builtin_py_set));
    env.functions.insert("py_tensor_get".to_string(), FnDef::Builtin(builtin_py_tensor_get));
    env.functions.insert("py_tensor_set".to_string(), FnDef::Builtin(builtin_py_tensor_set));
    env.functions.insert("py_load_model".to_string(), FnDef::Builtin(builtin_py_load_model));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_python_available() {
        // Just check the function runs without panicking
        let _available = PythonInterop::python_available();
    }

    #[test]
    fn test_json_escape() {
        assert_eq!(json_escape("hello"), "hello");
        assert_eq!(json_escape("he\"llo"), "he\\\"llo");
        assert_eq!(json_escape("line\nnew"), "line\\nnew");
    }

    #[test]
    fn test_json_string() {
        assert_eq!(json_string("hello"), "\"hello\"");
        assert_eq!(json_string("he\"llo"), "\"he\\\"llo\"");
    }

    #[test]
    fn test_value_to_py_json() {
        assert_eq!(value_to_py_json(&Value::Int(42)), "42");
        assert_eq!(value_to_py_json(&Value::Float(3.14)), "3.14");
        assert_eq!(value_to_py_json(&Value::Bool(true)), "true");
        assert_eq!(value_to_py_json(&Value::String("hi".to_string())), "\"hi\"");
        assert_eq!(value_to_py_json(&Value::Void), "null");
    }

    #[test]
    fn test_value_to_py_json_array() {
        let arr = Value::Array(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        assert_eq!(value_to_py_json(&arr), "[1,2,3]");
    }

    #[test]
    fn test_py_json_to_value_primitives() {
        assert!(matches!(py_json_to_value("null"), Value::Void));
        assert!(matches!(py_json_to_value("true"), Value::Bool(true)));
        assert!(matches!(py_json_to_value("false"), Value::Bool(false)));
        assert!(matches!(py_json_to_value("42"), Value::Int(42)));
        assert!(matches!(py_json_to_value("3.14"), Value::Float(f) if (f - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_py_json_to_value_string() {
        match py_json_to_value("\"hello\"") {
            Value::String(s) => assert_eq!(s, "hello"),
            other => panic!("Expected String, got {:?}", other),
        }
    }

    #[test]
    fn test_py_json_to_value_array() {
        match py_json_to_value("[1, 2, 3]") {
            Value::Array(a) => assert_eq!(a.len(), 3),
            other => panic!("Expected Array, got {:?}", other),
        }
    }

    #[test]
    fn test_py_json_to_value_empty_array() {
        match py_json_to_value("[]") {
            Value::Array(a) => assert_eq!(a.len(), 0),
            other => panic!("Expected empty Array, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_number_array() {
        let result = parse_number_array("[1.0, 2.5, 3.0]").unwrap();
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-10);
        assert!((result[1] - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_parse_number_array_empty() {
        let result = parse_number_array("[]").unwrap();
        assert_eq!(result.len(), 0);
    }

    #[test]
    fn test_parse_tensor_json() {
        let json = r#"{"data": [1.0, 2.0, 3.0, 4.0], "shape": [2, 2], "dtype": "float64"}"#;
        let (data, shape) = parse_tensor_json(json).unwrap();
        assert_eq!(data.len(), 4);
        assert_eq!(shape, vec![2, 2]);
    }

    #[test]
    fn test_unquote() {
        assert_eq!(unquote("\"hello\""), "hello");
        assert_eq!(unquote("\"he\\nllo\""), "he\nllo");
        assert_eq!(unquote("plain"), "plain");
    }

    #[test]
    fn test_find_json_key() {
        let json = r#"{"ok": 42, "error": null}"#;
        assert!(find_json_key(json, "ok").is_some());
        assert!(find_json_key(json, "error").is_some());
        assert!(find_json_key(json, "missing").is_none());
    }

    #[test]
    fn test_extract_json_value_number() {
        let json = r#"{"ok": 42, "error": null}"#;
        let start = find_json_key(json, "ok").unwrap();
        let val = extract_json_value(json, start);
        assert_eq!(val.trim(), "42");
    }

    #[test]
    fn test_extract_json_value_string() {
        let json = r#"{"ok": "hello", "x": 1}"#;
        let start = find_json_key(json, "ok").unwrap();
        let val = extract_json_value(json, start);
        assert_eq!(val, "\"hello\"");
    }

    #[test]
    fn test_extract_json_value_object() {
        let json = r#"{"ok": {"data": [1,2], "shape": [2]}, "x": 1}"#;
        let start = find_json_key(json, "ok").unwrap();
        let val = extract_json_value(json, start);
        assert!(val.starts_with('{'));
        assert!(val.contains("data"));
    }

    #[test]
    fn test_split_top_level() {
        let items = split_top_level("1, 2, 3", ',');
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_split_top_level_nested() {
        let items = split_top_level("[1,2], 3, {\"a\": 1}", ',');
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_parse_response_ok() {
        let resp = r#"{"ok": 42}"#;
        let result = PythonInterop::parse_response(resp).unwrap();
        assert_eq!(result.trim(), "42");
    }

    #[test]
    fn test_parse_response_error() {
        let resp = r#"{"error": "something went wrong"}"#;
        let result = PythonInterop::parse_response(resp);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("something went wrong"));
    }

    // Integration tests that require python3
    #[test]
    fn test_python_start_and_eval() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        let result = py.eval("1 + 1").unwrap();
        assert_eq!(result.trim(), "2");
        py.stop();
    }

    #[test]
    fn test_python_exec_and_eval() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        py.exec("x = 42").unwrap();
        let result = py.eval("x").unwrap();
        assert_eq!(result.trim(), "42");
        py.stop();
    }

    #[test]
    fn test_python_import() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        // math is always available
        py.import("math", "math").unwrap();
        let result = py.eval("math.pi").unwrap();
        let pi: f64 = result.trim().parse().unwrap();
        assert!((pi - std::f64::consts::PI).abs() < 1e-10);
        py.stop();
    }

    #[test]
    fn test_python_error_handling() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        let result = py.eval("undefined_variable");
        assert!(result.is_err());
        // Should still be usable after error
        let result2 = py.eval("1 + 1").unwrap();
        assert_eq!(result2.trim(), "2");
        py.stop();
    }

    #[test]
    fn test_python_call_function() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        py.exec("def add(a, b): return a + b").unwrap();
        let result = py.call("add", &[Value::Int(3), Value::Int(4)]).unwrap();
        assert_eq!(result.trim(), "7");
        py.stop();
    }

    #[test]
    fn test_python_multiline_exec() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        py.exec("def factorial(n):\n    if n <= 1: return 1\n    return n * factorial(n-1)")
            .unwrap();
        let result = py.eval("factorial(5)").unwrap();
        assert_eq!(result.trim(), "120");
        py.stop();
    }

    #[test]
    fn test_python_set_get_tensor() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        // Check if numpy is available
        if py.import("numpy", "np").is_err() {
            eprintln!("Skipping tensor test: numpy not available");
            py.stop();
            return;
        }
        // Set tensor
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let shape = vec![2, 3];
        py.set_tensor("test_arr", &data, &shape).unwrap();

        // Get it back
        let (got_data, got_shape) = py.get_tensor("test_arr").unwrap();
        assert_eq!(got_shape, vec![2, 3]);
        assert_eq!(got_data.len(), 6);
        for (a, b) in data.iter().zip(got_data.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
        py.stop();
    }

    #[test]
    fn test_python_graceful_shutdown() {
        if !PythonInterop::python_available() {
            eprintln!("Skipping: python3 not available");
            return;
        }
        let mut py = PythonInterop::start().unwrap();
        assert!(py.is_active());
        py.stop();
        assert!(!py.is_active());
    }

    #[test]
    fn test_builtin_py_available() {
        let mut env = Env::new();
        let result = builtin_py_available(&mut env, vec![]).unwrap();
        assert!(matches!(result, Value::Bool(_)));
    }

    #[test]
    fn test_py_json_to_value_tensor() {
        let json = r#"{"__tensor__": true, "data": [1.0, 2.0], "shape": [2], "dtype": "float64"}"#;
        match py_json_to_value(json) {
            Value::Struct { name, fields } => {
                assert_eq!(name, "Tensor");
                assert!(fields.contains_key("data"));
                assert!(fields.contains_key("shape"));
            }
            other => panic!("Expected Struct, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_weights_json() {
        let json = r#"{"layer1": {"data": [1.0, 2.0], "shape": [2], "dtype": "float64"}, "layer2": {"data": [3.0], "shape": [1], "dtype": "float64"}}"#;
        let weights = parse_weights_json(json).unwrap();
        assert_eq!(weights.len(), 2);
        assert!(weights.contains_key("layer1"));
        assert!(weights.contains_key("layer2"));
        let (data, shape) = &weights["layer1"];
        assert_eq!(data.len(), 2);
        assert_eq!(shape, &vec![2]);
    }

    #[test]
    fn test_value_roundtrip_json() {
        // Int roundtrip
        let v = Value::Int(42);
        let json = value_to_py_json(&v);
        let back = py_json_to_value(&json);
        assert!(matches!(back, Value::Int(42)));

        // Float roundtrip
        let v = Value::Float(3.14);
        let json = value_to_py_json(&v);
        let back = py_json_to_value(&json);
        assert!(matches!(back, Value::Float(f) if (f - 3.14).abs() < 1e-10));

        // Bool roundtrip
        let v = Value::Bool(true);
        let json = value_to_py_json(&v);
        let back = py_json_to_value(&json);
        assert!(matches!(back, Value::Bool(true)));

        // String roundtrip
        let v = Value::String("hello world".to_string());
        let json = value_to_py_json(&v);
        let back = py_json_to_value(&json);
        match back {
            Value::String(s) => assert_eq!(s, "hello world"),
            _ => panic!("Expected string"),
        }
    }
}
