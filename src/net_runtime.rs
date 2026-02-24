/// Net Runtime: TCP/UDP sockets, HTTP client for Vortex.
use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream, UdpSocket};
use std::time::Duration;

pub struct NetConnection {
    pub id: usize,
    pub kind: ConnKind,
    pub remote: String,
    pub local: String,
}

pub enum ConnKind {
    TcpClient(TcpStream),
    TcpServer(TcpListener),
    Udp(UdpSocket),
}

pub struct NetRuntime {
    pub connections: HashMap<usize, NetConnection>,
    next_id: usize,
}

impl NetRuntime {
    pub fn new() -> Self {
        Self { connections: HashMap::new(), next_id: 1 }
    }

    pub fn tcp_connect(&mut self, addr: &str) -> Result<usize, String> {
        let stream = TcpStream::connect(addr).map_err(|e| format!("tcp connect: {}", e))?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        let local = stream.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let id = self.next_id;
        self.next_id += 1;
        self.connections.insert(id, NetConnection { id, kind: ConnKind::TcpClient(stream), remote: addr.to_string(), local });
        Ok(id)
    }

    pub fn tcp_listen(&mut self, addr: &str) -> Result<usize, String> {
        let listener = TcpListener::bind(addr).map_err(|e| format!("tcp bind: {}", e))?;
        let local = listener.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let id = self.next_id;
        self.next_id += 1;
        self.connections.insert(id, NetConnection { id, kind: ConnKind::TcpServer(listener), remote: String::new(), local });
        Ok(id)
    }

    pub fn tcp_accept(&mut self, server_id: usize) -> Result<usize, String> {
        // Extract listener ref carefully to avoid borrow issues
        let listener_local = {
            match self.connections.get(&server_id) {
                Some(NetConnection { kind: ConnKind::TcpServer(_), .. }) => {},
                _ => return Err("not a tcp server".into()),
            }
            // We need to get the raw fd or clone - use a different approach
            server_id
        };
        let conn = self.connections.get(&listener_local).unwrap();
        let listener = match &conn.kind {
            ConnKind::TcpServer(l) => l,
            _ => unreachable!(),
        };
        let (stream, addr) = listener.accept().map_err(|e| format!("accept: {}", e))?;
        let local = stream.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let id = self.next_id;
        self.next_id += 1;
        self.connections.insert(id, NetConnection { id, kind: ConnKind::TcpClient(stream), remote: addr.to_string(), local });
        Ok(id)
    }

    pub fn tcp_send(&mut self, conn_id: usize, data: &[u8]) -> Result<usize, String> {
        match self.connections.get_mut(&conn_id) {
            Some(NetConnection { kind: ConnKind::TcpClient(ref mut s), .. }) => {
                s.write_all(data).map_err(|e| format!("send: {}", e))?;
                Ok(data.len())
            }
            _ => Err("not a tcp client".into()),
        }
    }

    pub fn tcp_recv(&mut self, conn_id: usize, max_bytes: usize) -> Result<Vec<u8>, String> {
        match self.connections.get_mut(&conn_id) {
            Some(NetConnection { kind: ConnKind::TcpClient(ref mut s), .. }) => {
                let mut buf = vec![0u8; max_bytes];
                let n = s.read(&mut buf).map_err(|e| format!("recv: {}", e))?;
                buf.truncate(n);
                Ok(buf)
            }
            _ => Err("not a tcp client".into()),
        }
    }

    pub fn udp_bind(&mut self, addr: &str) -> Result<usize, String> {
        let sock = UdpSocket::bind(addr).map_err(|e| format!("udp bind: {}", e))?;
        let local = sock.local_addr().map(|a| a.to_string()).unwrap_or_default();
        let id = self.next_id;
        self.next_id += 1;
        self.connections.insert(id, NetConnection { id, kind: ConnKind::Udp(sock), remote: String::new(), local });
        Ok(id)
    }

    pub fn udp_send_to(&mut self, conn_id: usize, data: &[u8], addr: &str) -> Result<usize, String> {
        match self.connections.get(&conn_id) {
            Some(NetConnection { kind: ConnKind::Udp(ref s), .. }) => {
                s.send_to(data, addr).map_err(|e| format!("udp send: {}", e))
            }
            _ => Err("not a udp socket".into()),
        }
    }

    pub fn udp_recv_from(&mut self, conn_id: usize, max_bytes: usize) -> Result<(Vec<u8>, String), String> {
        match self.connections.get(&conn_id) {
            Some(NetConnection { kind: ConnKind::Udp(ref s), .. }) => {
                let mut buf = vec![0u8; max_bytes];
                let (n, addr) = s.recv_from(&mut buf).map_err(|e| format!("udp recv: {}", e))?;
                buf.truncate(n);
                Ok((buf, addr.to_string()))
            }
            _ => Err("not a udp socket".into()),
        }
    }

    pub fn close(&mut self, conn_id: usize) -> Result<(), String> {
        self.connections.remove(&conn_id).ok_or_else(|| "connection not found".to_string())?;
        Ok(())
    }

    pub fn http_get(&mut self, url: &str) -> Result<String, String> {
        let (host, port, path) = parse_url(url)?;
        let addr = format!("{}:{}", host, port);
        let mut stream = TcpStream::connect(&addr).map_err(|e| format!("connect: {}", e))?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        let request = format!("GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n", path, host);
        stream.write_all(request.as_bytes()).map_err(|e| format!("write: {}", e))?;
        let mut response = String::new();
        stream.read_to_string(&mut response).map_err(|e| format!("read: {}", e))?;
        if let Some(idx) = response.find("\r\n\r\n") {
            Ok(response[idx + 4..].to_string())
        } else {
            Ok(response)
        }
    }

    pub fn http_post(&mut self, url: &str, body: &str, content_type: &str) -> Result<String, String> {
        let (host, port, path) = parse_url(url)?;
        let addr = format!("{}:{}", host, port);
        let mut stream = TcpStream::connect(&addr).map_err(|e| format!("connect: {}", e))?;
        stream.set_read_timeout(Some(Duration::from_secs(30))).ok();
        let request = format!(
            "POST {} HTTP/1.1\r\nHost: {}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            path, host, content_type, body.len(), body
        );
        stream.write_all(request.as_bytes()).map_err(|e| format!("write: {}", e))?;
        let mut response = String::new();
        stream.read_to_string(&mut response).map_err(|e| format!("read: {}", e))?;
        if let Some(idx) = response.find("\r\n\r\n") {
            Ok(response[idx + 4..].to_string())
        } else {
            Ok(response)
        }
    }
}

fn parse_url(url: &str) -> Result<(String, u16, String), String> {
    let url = url.strip_prefix("http://").unwrap_or(url);
    let (host_port, path) = if let Some(idx) = url.find('/') {
        (&url[..idx], url[idx..].to_string())
    } else {
        (url, "/".to_string())
    };
    let (host, port) = if let Some(idx) = host_port.find(':') {
        (&host_port[..idx], host_port[idx+1..].parse::<u16>().map_err(|_| "bad port".to_string())?)
    } else {
        (host_port, 80u16)
    };
    Ok((host.to_string(), port, path))
}

// ── Builtins ────────────────────────────────────────────────────────

use crate::interpreter::{Env, Value, FnDef};

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("net_tcp_connect".to_string(), FnDef::Builtin(builtin_tcp_connect));
    env.functions.insert("net_tcp_listen".to_string(), FnDef::Builtin(builtin_tcp_listen));
    env.functions.insert("net_tcp_accept".to_string(), FnDef::Builtin(builtin_tcp_accept));
    env.functions.insert("net_tcp_send".to_string(), FnDef::Builtin(builtin_tcp_send));
    env.functions.insert("net_tcp_recv".to_string(), FnDef::Builtin(builtin_tcp_recv));
    env.functions.insert("net_udp_bind".to_string(), FnDef::Builtin(builtin_udp_bind));
    env.functions.insert("net_udp_send_to".to_string(), FnDef::Builtin(builtin_udp_send_to));
    env.functions.insert("net_udp_recv_from".to_string(), FnDef::Builtin(builtin_udp_recv_from));
    env.functions.insert("net_close".to_string(), FnDef::Builtin(builtin_net_close));
    env.functions.insert("net_http_get".to_string(), FnDef::Builtin(builtin_http_get));
    env.functions.insert("net_http_post".to_string(), FnDef::Builtin(builtin_http_post));
}

fn builtin_tcp_connect(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_tcp_connect(addr)".into()); }
    let addr = match &args[0] { Value::String(s) => s.clone(), _ => return Err("addr must be string".into()) };
    let id = env.net_runtime.tcp_connect(&addr)?;
    Ok(Value::Int(id as i128))
}

fn builtin_tcp_listen(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_tcp_listen(addr)".into()); }
    let addr = match &args[0] { Value::String(s) => s.clone(), _ => return Err("addr must be string".into()) };
    let id = env.net_runtime.tcp_listen(&addr)?;
    Ok(Value::Int(id as i128))
}

fn builtin_tcp_accept(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_tcp_accept(server_id)".into()); }
    let sid = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let id = env.net_runtime.tcp_accept(sid)?;
    Ok(Value::Int(id as i128))
}

fn builtin_tcp_send(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("net_tcp_send(id, data)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let data = match &args[1] { Value::String(s) => s.as_bytes().to_vec(), _ => return Err("data must be string".into()) };
    let n = env.net_runtime.tcp_send(id, &data)?;
    Ok(Value::Int(n as i128))
}

fn builtin_tcp_recv(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("net_tcp_recv(id, max_bytes)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let max = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("max must be int".into()) };
    let data = env.net_runtime.tcp_recv(id, max)?;
    Ok(Value::String(String::from_utf8_lossy(&data).to_string()))
}

fn builtin_udp_bind(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_udp_bind(addr)".into()); }
    let addr = match &args[0] { Value::String(s) => s.clone(), _ => return Err("addr must be string".into()) };
    let id = env.net_runtime.udp_bind(&addr)?;
    Ok(Value::Int(id as i128))
}

fn builtin_udp_send_to(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("net_udp_send_to(id, data, addr)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let data = match &args[1] { Value::String(s) => s.as_bytes().to_vec(), _ => return Err("data must be string".into()) };
    let addr = match &args[2] { Value::String(s) => s.clone(), _ => return Err("addr must be string".into()) };
    let n = env.net_runtime.udp_send_to(id, &data, &addr)?;
    Ok(Value::Int(n as i128))
}

fn builtin_udp_recv_from(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("net_udp_recv_from(id, max_bytes)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    let max = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("max must be int".into()) };
    let (data, addr) = env.net_runtime.udp_recv_from(id, max)?;
    Ok(Value::Tuple(vec![
        Value::String(String::from_utf8_lossy(&data).to_string()),
        Value::String(addr),
    ]))
}

fn builtin_net_close(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_close(id)".into()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("id must be int".into()) };
    env.net_runtime.close(id)?;
    Ok(Value::Void)
}

fn builtin_http_get(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("net_http_get(url)".into()); }
    let url = match &args[0] { Value::String(s) => s.clone(), _ => return Err("url must be string".into()) };
    let body = env.net_runtime.http_get(&url)?;
    Ok(Value::String(body))
}

fn builtin_http_post(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("net_http_post(url, body, content_type)".into()); }
    let url = match &args[0] { Value::String(s) => s.clone(), _ => return Err("url must be string".into()) };
    let body = match &args[1] { Value::String(s) => s.clone(), _ => return Err("body must be string".into()) };
    let ct = match &args[2] { Value::String(s) => s.clone(), _ => return Err("content_type must be string".into()) };
    let resp = env.net_runtime.http_post(&url, &body, &ct)?;
    Ok(Value::String(resp))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_net_runtime_new() {
        let rt = NetRuntime::new();
        assert_eq!(rt.next_id, 1);
    }

    #[test]
    fn test_parse_url_simple() {
        let (h, p, path) = parse_url("http://example.com/api").unwrap();
        assert_eq!(h, "example.com");
        assert_eq!(p, 80);
        assert_eq!(path, "/api");
    }

    #[test]
    fn test_parse_url_with_port() {
        let (h, p, path) = parse_url("http://localhost:8080/test").unwrap();
        assert_eq!(h, "localhost");
        assert_eq!(p, 8080);
        assert_eq!(path, "/test");
    }

    #[test]
    fn test_parse_url_no_path() {
        let (h, p, path) = parse_url("example.com").unwrap();
        assert_eq!(h, "example.com");
        assert_eq!(p, 80);
        assert_eq!(path, "/");
    }

    #[test]
    fn test_udp_bind_localhost() {
        let mut rt = NetRuntime::new();
        let id = rt.udp_bind("127.0.0.1:0").unwrap();
        assert_eq!(id, 1);
        rt.close(id).unwrap();
    }

    #[test]
    fn test_close_nonexistent() {
        let mut rt = NetRuntime::new();
        assert!(rt.close(999).is_err());
    }

    #[test]
    fn test_tcp_loopback() {
        let mut rt = NetRuntime::new();
        let srv = rt.tcp_listen("127.0.0.1:0").unwrap();
        let local_addr = rt.connections.get(&srv).unwrap().local.clone();
        let client = rt.tcp_connect(&local_addr).unwrap();
        let accepted = rt.tcp_accept(srv).unwrap();
        rt.tcp_send(client, b"hello").unwrap();
        let data = rt.tcp_recv(accepted, 1024).unwrap();
        assert_eq!(data, b"hello");
        rt.close(client).unwrap();
        rt.close(accepted).unwrap();
        rt.close(srv).unwrap();
    }

    #[test]
    fn test_udp_loopback() {
        let mut rt = NetRuntime::new();
        let s1 = rt.udp_bind("127.0.0.1:0").unwrap();
        let s2 = rt.udp_bind("127.0.0.1:0").unwrap();
        let addr2 = rt.connections.get(&s2).unwrap().local.clone();
        rt.udp_send_to(s1, b"ping", &addr2).unwrap();
        let (data, _from) = rt.udp_recv_from(s2, 1024).unwrap();
        assert_eq!(data, b"ping");
        rt.close(s1).unwrap();
        rt.close(s2).unwrap();
    }
}
