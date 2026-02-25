use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::io::{BufRead, BufReader, Read as IoRead, Write as IoWrite};
use std::net::TcpListener;
use std::net::TcpStream;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread::{self, JoinHandle};

// ---------------------------------------------------------------------------
// InferenceServer
// ---------------------------------------------------------------------------

struct InferenceServer {
    listener: Option<Arc<TcpListener>>,
    port: u16,
    running: Arc<AtomicBool>,
    model_name: String,
    thread_handles: Vec<JoinHandle<()>>,
}

impl InferenceServer {
    fn new() -> Self {
        Self {
            listener: None,
            port: 0,
            running: Arc::new(AtomicBool::new(false)),
            model_name: String::from("default"),
            thread_handles: Vec::new(),
        }
    }
}

static SERVER: LazyLock<Mutex<Option<InferenceServer>>> =
    LazyLock::new(|| Mutex::new(None));

// ---------------------------------------------------------------------------
// HTTP primitives
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
struct HttpRequest {
    method: String,
    path: String,
    headers: HashMap<String, String>,
    body: String,
}

fn parse_request(stream: &mut TcpStream) -> Result<HttpRequest, String> {
    let mut reader = BufReader::new(stream.try_clone().map_err(|e| e.to_string())?);

    // Request line
    let mut request_line = String::new();
    reader
        .read_line(&mut request_line)
        .map_err(|e| e.to_string())?;
    let request_line = request_line.trim_end().to_string();
    let parts: Vec<&str> = request_line.splitn(3, ' ').collect();
    if parts.len() < 2 {
        return Err("Malformed request line".into());
    }
    let method = parts[0].to_string();
    let path = parts[1].to_string();

    // Headers
    let mut headers = HashMap::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).map_err(|e| e.to_string())?;
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            break;
        }
        if let Some((key, val)) = trimmed.split_once(':') {
            headers.insert(
                key.trim().to_lowercase(),
                val.trim().to_string(),
            );
        }
    }

    // Body
    let body = if let Some(cl) = headers.get("content-length") {
        let len: usize = cl.parse().unwrap_or(0);
        if len > 0 {
            let mut buf = vec![0u8; len];
            reader.read_exact(&mut buf).map_err(|e| e.to_string())?;
            String::from_utf8_lossy(&buf).to_string()
        } else {
            String::new()
        }
    } else {
        String::new()
    };

    Ok(HttpRequest {
        method,
        path,
        headers,
        body,
    })
}

fn send_response(
    stream: &mut TcpStream,
    status: u16,
    headers: &[(&str, &str)],
    body: &str,
) {
    let reason = match status {
        200 => "OK",
        404 => "Not Found",
        400 => "Bad Request",
        500 => "Internal Server Error",
        _ => "Unknown",
    };
    let mut resp = format!("HTTP/1.1 {} {}\r\n", status, reason);
    for (k, v) in headers {
        resp.push_str(&format!("{}: {}\r\n", k, v));
    }
    if !headers.iter().any(|(k, _)| k.eq_ignore_ascii_case("content-length")) {
        resp.push_str(&format!("Content-Length: {}\r\n", body.len()));
    }
    resp.push_str("\r\n");
    resp.push_str(body);
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn send_chunked_start(stream: &mut TcpStream, status: u16) {
    let reason = match status {
        200 => "OK",
        _ => "Unknown",
    };
    let resp = format!(
        "HTTP/1.1 {} {}\r\nTransfer-Encoding: chunked\r\nContent-Type: application/json\r\n\r\n",
        status, reason
    );
    let _ = stream.write_all(resp.as_bytes());
    let _ = stream.flush();
}

fn send_chunk(stream: &mut TcpStream, data: &str) {
    let chunk = format!("{:x}\r\n{}\r\n", data.len(), data);
    let _ = stream.write_all(chunk.as_bytes());
    let _ = stream.flush();
}

fn send_chunk_end(stream: &mut TcpStream) {
    let _ = stream.write_all(b"0\r\n\r\n");
    let _ = stream.flush();
}

// ---------------------------------------------------------------------------
// Minimal JSON helpers (no serde dependency)
// ---------------------------------------------------------------------------

fn json_get_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    // skip whitespace and colon
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();
    if rest.starts_with('"') {
        let rest = &rest[1..];
        let end = rest.find('"')?;
        Some(rest[..end].to_string())
    } else {
        None
    }
}

fn json_get_number(json: &str, key: &str) -> Option<f64> {
    let pattern = format!("\"{}\"", key);
    let idx = json.find(&pattern)?;
    let rest = &json[idx + pattern.len()..];
    let rest = rest.trim_start();
    let rest = rest.strip_prefix(':')?;
    let rest = rest.trim_start();
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '.' && c != '-').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ---------------------------------------------------------------------------
// Route handling
// ---------------------------------------------------------------------------

fn handle_connection(mut stream: TcpStream, model_name: &str) {
    let req = match parse_request(&mut stream) {
        Ok(r) => r,
        Err(_) => {
            send_response(&mut stream, 400, &[("Content-Type", "application/json")], "{\"error\":\"bad request\"}");
            return;
        }
    };
    route_request(&mut stream, &req, model_name);
}

fn route_request(stream: &mut TcpStream, req: &HttpRequest, model_name: &str) {
    match (req.method.as_str(), req.path.as_str()) {
        ("GET", "/health") => {
            let body = format!(
                "{{\"status\":\"ok\",\"model\":\"{}\"}}",
                model_name
            );
            send_response(
                stream,
                200,
                &[("Content-Type", "application/json")],
                &body,
            );
        }
        ("GET", "/info") => {
            let body = "{\"server\":\"vortex-inference\",\"version\":\"0.1.0\"}";
            send_response(
                stream,
                200,
                &[("Content-Type", "application/json")],
                body,
            );
        }
        ("POST", "/generate") => {
            handle_generate(stream, req);
        }
        _ => {
            send_response(
                stream,
                404,
                &[("Content-Type", "application/json")],
                "{\"error\":\"not found\"}",
            );
        }
    }
}

fn handle_generate(stream: &mut TcpStream, req: &HttpRequest) {
    let prompt = json_get_string(&req.body, "prompt").unwrap_or_default();
    let _max_tokens = json_get_number(&req.body, "max_tokens").unwrap_or(64.0) as usize;
    let _temperature = json_get_number(&req.body, "temperature").unwrap_or(1.0);

    // Placeholder generation: echo prompt + marker tokens via chunked encoding
    send_chunked_start(stream, 200);

    let tokens = vec![
        prompt.clone(),
        " [generated]".to_string(),
    ];

    for tok in &tokens {
        let chunk_body = format!(
            "{{\"token\":\"{}\",\"finished\":false}}",
            tok.replace('\\', "\\\\").replace('"', "\\\"")
        );
        send_chunk(stream, &chunk_body);
    }

    // Final chunk indicating completion
    send_chunk(stream, "{\"token\":\"\",\"finished\":true}");
    send_chunk_end(stream);
}

// ---------------------------------------------------------------------------
// Server lifecycle
// ---------------------------------------------------------------------------

fn start_server(port: u16) -> Result<(), String> {
    let mut guard = SERVER.lock().map_err(|e| e.to_string())?;
    if let Some(ref srv) = *guard {
        if srv.running.load(Ordering::SeqCst) {
            return Err("Server already running".into());
        }
    }

    let addr = format!("0.0.0.0:{}", port);
    let listener = TcpListener::bind(&addr).map_err(|e| format!("Bind failed: {}", e))?;
    listener
        .set_nonblocking(true)
        .map_err(|e| format!("set_nonblocking: {}", e))?;
    let listener = Arc::new(listener);

    let running = Arc::new(AtomicBool::new(true));
    let model_name = guard
        .as_ref()
        .map(|s| s.model_name.clone())
        .unwrap_or_else(|| "default".into());

    let mut handles = Vec::new();
    for _ in 0..4 {
        let l = Arc::clone(&listener);
        let r = Arc::clone(&running);
        let mn = model_name.clone();
        let h = thread::spawn(move || {
            worker_loop(l, r, mn);
        });
        handles.push(h);
    }

    *guard = Some(InferenceServer {
        listener: Some(listener),
        port,
        running,
        model_name,
        thread_handles: handles,
    });

    Ok(())
}

fn worker_loop(listener: Arc<TcpListener>, running: Arc<AtomicBool>, model_name: String) {
    while running.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((stream, _addr)) => {
                // Set blocking for this connection
                let _ = stream.set_nonblocking(false);
                handle_connection(stream, &model_name);
            }
            Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                thread::sleep(std::time::Duration::from_millis(50));
            }
            Err(_) => {
                break;
            }
        }
    }
}

fn stop_server() -> Result<(), String> {
    let mut guard = SERVER.lock().map_err(|e| e.to_string())?;
    if let Some(mut srv) = guard.take() {
        srv.running.store(false, Ordering::SeqCst);
        // Drop listener to unblock accepts
        srv.listener.take();
        for h in srv.thread_handles.drain(..) {
            let _ = h.join();
        }
        Ok(())
    } else {
        Err("Server not running".into())
    }
}

fn set_model_name(name: String) -> Result<(), String> {
    let mut guard = SERVER.lock().map_err(|e| e.to_string())?;
    match guard.as_mut() {
        Some(srv) => {
            srv.model_name = name;
            Ok(())
        }
        None => {
            // Create placeholder so the name persists for next start
            let mut srv = InferenceServer::new();
            srv.model_name = name;
            *guard = Some(srv);
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert(
        "server_start".to_string(),
        FnDef::Builtin(|_env, args| {
            let port = match args.first() {
                Some(Value::Int(n)) => *n as u16,
                _ => return Err("server_start expects an integer port".into()),
            };
            start_server(port)?;
            Ok(Value::String(format!("Server started on port {}", port)))
        }),
    );

    env.functions.insert(
        "server_stop".to_string(),
        FnDef::Builtin(|_env, _args| {
            stop_server()?;
            Ok(Value::String("Server stopped".into()))
        }),
    );

    env.functions.insert(
        "server_set_model".to_string(),
        FnDef::Builtin(|_env, args| {
            let name = match args.first() {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("server_set_model expects a string name".into()),
            };
            set_model_name(name.clone())?;
            Ok(Value::String(format!("Model set to {}", name)))
        }),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_request_basic() {
        use std::io::Write;
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let mut client = TcpStream::connect(addr).unwrap();
            client
                .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
                .unwrap();
            client.flush().unwrap();
        });

        let (mut stream, _) = listener.accept().unwrap();
        let req = parse_request(&mut stream).unwrap();
        assert_eq!(req.method, "GET");
        assert_eq!(req.path, "/health");
        assert_eq!(req.headers.get("host").unwrap(), "localhost");
        assert_eq!(req.body, "");
        handle.join().unwrap();
    }

    #[test]
    fn test_parse_request_with_body() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();
        let body_text = "{\"prompt\":\"hello\"}";

        let handle = std::thread::spawn(move || {
            let mut client = TcpStream::connect(addr).unwrap();
            let msg = format!(
                "POST /generate HTTP/1.1\r\nContent-Length: {}\r\n\r\n{}",
                body_text.len(),
                body_text
            );
            client.write_all(msg.as_bytes()).unwrap();
            client.flush().unwrap();
        });

        let (mut stream, _) = listener.accept().unwrap();
        let req = parse_request(&mut stream).unwrap();
        assert_eq!(req.method, "POST");
        assert_eq!(req.path, "/generate");
        assert_eq!(req.body, body_text);
        handle.join().unwrap();
    }

    #[test]
    fn test_send_response_format() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            send_response(
                &mut stream,
                200,
                &[("Content-Type", "application/json")],
                "{\"ok\":true}",
            );
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let mut buf = Vec::new();
        std::thread::sleep(std::time::Duration::from_millis(100));
        client.set_read_timeout(Some(std::time::Duration::from_millis(500))).unwrap();
        let _ = client.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        assert!(text.starts_with("HTTP/1.1 200 OK\r\n"));
        assert!(text.contains("Content-Type: application/json"));
        assert!(text.contains("{\"ok\":true}"));
        handle.join().unwrap();
    }

    #[test]
    fn test_health_endpoint() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let (stream, _) = listener.accept().unwrap();
            let _ = stream.set_nonblocking(false);
            handle_connection(stream, "test-model");
        });

        let mut client = TcpStream::connect(addr).unwrap();
        client
            .write_all(b"GET /health HTTP/1.1\r\nHost: localhost\r\nContent-Length: 0\r\n\r\n")
            .unwrap();
        client.flush().unwrap();

        let mut buf = Vec::new();
        client.set_read_timeout(Some(std::time::Duration::from_millis(500))).unwrap();
        let _ = client.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("200 OK"));
        assert!(text.contains("\"status\":\"ok\""));
        assert!(text.contains("\"model\":\"test-model\""));
        handle.join().unwrap();
    }

    #[test]
    fn test_chunked_encoding() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = listener.local_addr().unwrap();

        let handle = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            send_chunked_start(&mut stream, 200);
            send_chunk(&mut stream, "hello");
            send_chunk(&mut stream, "world");
            send_chunk_end(&mut stream);
        });

        let mut client = TcpStream::connect(addr).unwrap();
        let mut buf = Vec::new();
        client.set_read_timeout(Some(std::time::Duration::from_millis(500))).unwrap();
        let _ = client.read_to_end(&mut buf);
        let text = String::from_utf8_lossy(&buf);
        assert!(text.contains("Transfer-Encoding: chunked"));
        // chunk size for "hello" is 5 => "5\r\nhello\r\n"
        assert!(text.contains("5\r\nhello\r\n"));
        assert!(text.contains("5\r\nworld\r\n"));
        assert!(text.contains("0\r\n\r\n"));
        handle.join().unwrap();
    }

    #[test]
    fn test_json_helpers() {
        let json = r#"{"prompt":"hi there","max_tokens":128,"temperature":0.7}"#;
        assert_eq!(json_get_string(json, "prompt").unwrap(), "hi there");
        assert_eq!(json_get_number(json, "max_tokens").unwrap(), 128.0);
        assert!((json_get_number(json, "temperature").unwrap() - 0.7).abs() < 1e-9);
        assert!(json_get_string(json, "missing").is_none());
    }
}
