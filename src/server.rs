//! Vortex Server Runtime — persistent processes that maintain state across requests.
//!
//! Supports inference servers, streaming processors, actor systems, and generic
//! event-driven Vortex programs that "stay alive" indefinitely.

use crate::ast::*;
use crate::interpreter::{Env, FnDef, Value, eval_block};
use crate::lexer;
use crate::parser;
use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Core server
// ---------------------------------------------------------------------------

/// A Vortex program that stays alive and accepts requests.
pub struct VortexServer {
    /// The interpreter environment (persists across requests)
    env: Env,
    /// Registered request handlers: route -> function_name
    handlers: HashMap<String, String>,
    /// Timer handlers: timer_name -> (interval_ms, function_name)
    timers: HashMap<String, (u64, String)>,
    /// Stream handlers: stream_name -> function_name
    streams: HashMap<String, String>,
    /// Server state
    state: ServerState,
    /// Event loop queue
    event_queue: VecDeque<Event>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ServerState {
    Running,
    Paused,
    Stopping,
}

#[derive(Clone, Debug)]
pub enum Event {
    /// HTTP-like request
    Request {
        route: String,
        payload: String,
        response_tx: usize,
    },
    /// Timer tick
    Timer {
        name: String,
        interval_ms: u64,
    },
    /// Data stream event
    StreamData {
        stream_name: String,
        data: Vec<f64>,
    },
    /// Checkpoint signal
    Checkpoint {
        path: String,
    },
    /// Shutdown
    Shutdown,
}

#[derive(Clone, Debug)]
pub struct Response {
    pub status: u16,
    pub body: String,
}

/// Load source into an Env: lex, parse, register all definitions.
fn load_source_into_env(env: &mut Env, source: &str) -> Result<(), String> {
    let tokens = lexer::lex(source);
    let program = parser::parse(tokens, 0).map_err(|diags| {
        diags
            .iter()
            .map(|d| format!("{:?}", d.message))
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    register_program(env, &program);
    Ok(())
}

/// Register all functions / structs / enums / impls from a parsed program.
fn register_program(env: &mut Env, program: &Program) {
    for item in &program.items {
        match &item.kind {
            ItemKind::Function(func) => {
                let params: Vec<String> =
                    func.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    func.name.name.clone(),
                    FnDef::User {
                        params,
                        body: func.body.clone(),
                    },
                );
            }
            ItemKind::Kernel(kernel) => {
                let params: Vec<String> =
                    kernel.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    kernel.name.name.clone(),
                    FnDef::User {
                        params,
                        body: kernel.body.clone(),
                    },
                );
            }
            ItemKind::Struct(s) => {
                let field_names: Vec<String> =
                    s.fields.iter().map(|f| f.name.name.clone()).collect();
                env.struct_defs
                    .insert(s.name.name.clone(), field_names);
            }
            ItemKind::Impl(impl_block) => {
                let type_name = match &impl_block.target.kind {
                    TypeExprKind::Named(id) => id.name.clone(),
                    _ => continue,
                };
                for method_item in &impl_block.methods {
                    if let ItemKind::Function(func) = &method_item.kind {
                        let params: Vec<String> =
                            func.params.iter().map(|p| p.name.name.clone()).collect();
                        let qualified = format!("{}::{}", type_name, func.name.name);
                        env.functions.insert(
                            qualified,
                            FnDef::User {
                                params,
                                body: func.body.clone(),
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

/// Call a named Vortex function with the given argument values.
fn call_vortex_fn(env: &mut Env, fn_name: &str, args: Vec<Value>) -> Result<Value, String> {
    let func = env
        .functions
        .get(fn_name)
        .cloned()
        .ok_or_else(|| format!("function '{}' not found", fn_name))?;
    match func {
        FnDef::User { params, body } => {
            env.push_scope();
            for (param, val) in params.iter().zip(args.into_iter()) {
                env.define(param, val);
            }
            let result = eval_block(env, &body)?;
            env.pop_scope();
            match result {
                Value::Return(v) => Ok(*v),
                other => Ok(other),
            }
        }
        FnDef::Builtin(f) => f(env, args),
        FnDef::GradWrapper { fn_name, order: _ } => {
            crate::interpreter::eval_block(env, &crate::ast::Block { stmts: vec![], expr: None, span: crate::ast::Span::new(0, 0) })
        }
    }
}

impl VortexServer {
    pub fn new() -> Self {
        Self {
            env: Env::new(),
            handlers: HashMap::new(),
            timers: HashMap::new(),
            streams: HashMap::new(),
            state: ServerState::Running,
            event_queue: VecDeque::new(),
        }
    }

    /// Load a Vortex program and initialise the environment.
    pub fn load_program(&mut self, source: &str) -> Result<(), String> {
        load_source_into_env(&mut self.env, source)
    }

    /// Register a handler: when `route` is hit, call the named Vortex function.
    pub fn register_handler(&mut self, route: &str, function_name: &str) {
        self.handlers
            .insert(route.to_string(), function_name.to_string());
    }

    /// Register a periodic timer.
    pub fn register_timer(&mut self, name: &str, interval_ms: u64, function_name: &str) {
        self.timers
            .insert(name.to_string(), (interval_ms, function_name.to_string()));
    }

    /// Register a stream processor.
    pub fn register_stream(&mut self, stream_name: &str, function_name: &str) {
        self.streams
            .insert(stream_name.to_string(), function_name.to_string());
    }

    /// Process one event and optionally return a response.
    pub fn process_event(&mut self, event: &Event) -> Option<Response> {
        match event {
            Event::Request { route, payload, .. } => {
                if let Some(fn_name) = self.handlers.get(route).cloned() {
                    let arg = Value::String(payload.clone());
                    match call_vortex_fn(&mut self.env, &fn_name, vec![arg]) {
                        Ok(val) => Some(Response {
                            status: 200,
                            body: format!("{}", val),
                        }),
                        Err(e) => Some(Response {
                            status: 500,
                            body: e,
                        }),
                    }
                } else {
                    Some(Response {
                        status: 404,
                        body: format!("no handler for route '{}'", route),
                    })
                }
            }
            Event::Timer { name, .. } => {
                if let Some((_, fn_name)) = self.timers.get(name).cloned() {
                    let _ = call_vortex_fn(&mut self.env, &fn_name, vec![]);
                }
                None
            }
            Event::StreamData { stream_name, data } => {
                if let Some(fn_name) = self.streams.get(stream_name).cloned() {
                    let arr = Value::Array(data.iter().map(|&v| Value::Float(v)).collect());
                    let _ = call_vortex_fn(&mut self.env, &fn_name, vec![arr]);
                }
                None
            }
            Event::Checkpoint { path } => {
                let _ = self.checkpoint(path);
                None
            }
            Event::Shutdown => {
                self.state = ServerState::Stopping;
                None
            }
        }
    }

    /// Run the event loop (blocking). Drains the queue until Shutdown.
    pub fn run(&mut self) {
        while self.state == ServerState::Running {
            if let Some(event) = self.event_queue.pop_front() {
                self.process_event(&event);
                if self.state == ServerState::Stopping {
                    break;
                }
            } else {
                break; // no more events
            }
        }
    }

    /// Push an event into the queue.
    pub fn push_event(&mut self, event: Event) {
        self.event_queue.push_back(event);
    }

    /// Get server state.
    pub fn state(&self) -> &ServerState {
        &self.state
    }

    /// Checkpoint: serialise variable state to a simple text format.
    pub fn checkpoint(&self, path: &str) -> Result<(), String> {
        let mut lines = Vec::new();
        // Serialise the top-level scope variables
        if let Some(scope) = self.env.scopes.first() {
            for (k, v) in scope {
                lines.push(format!("{}={}", k, v));
            }
        }
        std::fs::write(path, lines.join("\n"))
            .map_err(|e| format!("checkpoint write error: {}", e))
    }

    /// Restore from a checkpoint file. Re-creates the server with the saved
    /// top-level variable bindings.
    pub fn restore(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("checkpoint read error: {}", e))?;
        let mut server = Self::new();
        for line in content.lines() {
            if let Some((k, v)) = line.split_once('=') {
                // Attempt to parse as int, float, bool, or string
                let val = if let Ok(n) = v.parse::<i128>() {
                    Value::Int(n)
                } else if let Ok(f) = v.parse::<f64>() {
                    Value::Float(f)
                } else if v == "true" {
                    Value::Bool(true)
                } else if v == "false" {
                    Value::Bool(false)
                } else {
                    Value::String(v.to_string())
                };
                server.env.define(k, val);
            }
        }
        Ok(server)
    }

    /// Direct access to the environment (for tests / advanced use).
    pub fn env(&self) -> &Env {
        &self.env
    }

    /// Mutable access to the environment.
    pub fn env_mut(&mut self) -> &mut Env {
        &mut self.env
    }
}

// ---------------------------------------------------------------------------
// Inference server
// ---------------------------------------------------------------------------

pub struct InferenceRequest {
    pub id: usize,
    pub input: Vec<f64>,
    pub shape: Vec<usize>,
    pub timestamp_us: u64,
}

pub struct InferenceResult {
    pub id: usize,
    pub output: Vec<f64>,
    pub shape: Vec<usize>,
    pub latency_us: u64,
}

/// An inference server for serving ML model predictions.
pub struct InferenceServer {
    server: VortexServer,
    model_name: String,
    /// Batching: collect requests and process together
    batch_queue: Vec<InferenceRequest>,
    batch_size: usize,
    batch_timeout_ms: u64,
    /// Statistics
    requests_served: usize,
    total_latency_us: u64,
    next_id: usize,
}

impl InferenceServer {
    pub fn new(model_source: &str, batch_size: usize) -> Result<Self, String> {
        let mut server = VortexServer::new();
        server.load_program(model_source)?;
        Ok(Self {
            server,
            model_name: "model".to_string(),
            batch_queue: Vec::new(),
            batch_size,
            batch_timeout_ms: 100,
            requests_served: 0,
            total_latency_us: 0,
            next_id: 0,
        })
    }

    /// Submit a single inference request synchronously.
    pub fn infer(&mut self, input: &[f64], shape: &[usize]) -> InferenceResult {
        let start = Instant::now();
        let input_val = Value::Array(input.iter().map(|&v| Value::Float(v)).collect());

        let output_val = call_vortex_fn(&mut self.server.env, "predict", vec![input_val])
            .unwrap_or(Value::Array(vec![]));

        let output = match &output_val {
            Value::Array(arr) => arr
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => 0.0,
                })
                .collect(),
            Value::Float(f) => vec![*f],
            Value::Int(n) => vec![*n as f64],
            _ => vec![],
        };

        let latency = start.elapsed().as_micros() as u64;
        self.requests_served += 1;
        self.total_latency_us += latency;

        InferenceResult {
            id: self.next_id,
            output,
            shape: shape.to_vec(),
            latency_us: latency,
        }
    }

    /// Submit a batch of requests.
    pub fn infer_batch(&mut self, requests: Vec<InferenceRequest>) -> Vec<InferenceResult> {
        requests
            .into_iter()
            .map(|req| {
                let mut res = self.infer(&req.input, &req.shape);
                res.id = req.id;
                res
            })
            .collect()
    }

    /// Enqueue a request for dynamic batching. Returns request id.
    pub fn submit_request(&mut self, input: &[f64], shape: &[usize]) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.batch_queue.push(InferenceRequest {
            id,
            input: input.to_vec(),
            shape: shape.to_vec(),
            timestamp_us: 0,
        });
        id
    }

    /// Flush the batch queue, processing all pending requests.
    pub fn flush_batch(&mut self) -> Vec<InferenceResult> {
        let batch: Vec<InferenceRequest> = self.batch_queue.drain(..).collect();
        self.infer_batch(batch)
    }

    pub fn avg_latency_us(&self) -> f64 {
        if self.requests_served == 0 {
            0.0
        } else {
            self.total_latency_us as f64 / self.requests_served as f64
        }
    }

    pub fn throughput_rps(&self) -> f64 {
        if self.total_latency_us == 0 {
            0.0
        } else {
            self.requests_served as f64 / (self.total_latency_us as f64 / 1_000_000.0)
        }
    }

    pub fn requests_served(&self) -> usize {
        self.requests_served
    }

    pub fn batch_queue_len(&self) -> usize {
        self.batch_queue.len()
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

// ---------------------------------------------------------------------------
// Streaming processor
// ---------------------------------------------------------------------------

pub struct SlidingWindow {
    pub data: VecDeque<Vec<f64>>,
    pub window_size: usize,
    pub stride: usize,
    offset: usize,
}

/// Process continuous data streams with Vortex.
pub struct StreamProcessor {
    server: VortexServer,
    /// Sliding window buffer for each stream
    windows: HashMap<String, SlidingWindow>,
    /// stream_name -> handler function name
    stream_handlers: HashMap<String, String>,
}

impl StreamProcessor {
    pub fn new(source: &str) -> Result<Self, String> {
        let mut server = VortexServer::new();
        server.load_program(source)?;
        Ok(Self {
            server,
            windows: HashMap::new(),
            stream_handlers: HashMap::new(),
        })
    }

    /// Register a stream with windowed processing.
    pub fn register_stream(
        &mut self,
        name: &str,
        window_size: usize,
        stride: usize,
        handler: &str,
    ) {
        self.windows.insert(
            name.to_string(),
            SlidingWindow {
                data: VecDeque::new(),
                window_size,
                stride,
                offset: 0,
            },
        );
        self.stream_handlers
            .insert(name.to_string(), handler.to_string());
    }

    /// Push new data to a stream. Returns results from any windows that triggered.
    pub fn push_data(&mut self, stream_name: &str, data: &[f64]) -> Vec<Vec<f64>> {
        let mut results = Vec::new();

        let handler = match self.stream_handlers.get(stream_name) {
            Some(h) => h.clone(),
            None => return results,
        };

        // Insert data into window
        match self.windows.get_mut(stream_name) {
            Some(w) => {
                w.data.push_back(data.to_vec());
                w.offset += 1;
            }
            None => return results,
        };

        // Collect windows to process, then process them separately
        loop {
            let win_data = {
                let window = match self.windows.get(stream_name) {
                    Some(w) => w,
                    None => break,
                };
                if window.data.len() < window.window_size {
                    break;
                }
                window.data.iter().take(window.window_size).cloned().collect::<Vec<_>>()
            };

            let result = self.process_window(&handler, &win_data);
            results.push(result);

            // Advance by stride
            let stride = self.windows.get(stream_name).map(|w| w.stride).unwrap_or(1);
            if let Some(window) = self.windows.get_mut(stream_name) {
                for _ in 0..stride {
                    if window.data.is_empty() {
                        break;
                    }
                    window.data.pop_front();
                }
            }
        }

        results
    }

    fn process_window(&mut self, handler: &str, window: &[Vec<f64>]) -> Vec<f64> {
        // Flatten all window rows into a single array
        let flat: Vec<Value> = window
            .iter()
            .flat_map(|row| row.iter().map(|&v| Value::Float(v)))
            .collect();
        let arg = Value::Array(flat);

        match call_vortex_fn(&mut self.server.env, handler, vec![arg]) {
            Ok(Value::Array(arr)) => arr
                .iter()
                .map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Int(n) => *n as f64,
                    _ => 0.0,
                })
                .collect(),
            Ok(Value::Float(f)) => vec![f],
            Ok(Value::Int(n)) => vec![n as f64],
            _ => vec![],
        }
    }
}

// ---------------------------------------------------------------------------
// Actor model
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Message {
    pub from: String,
    pub tag: String,
    pub payload: String,
}

/// Actor: a long-lived Vortex computation with a mailbox.
pub struct Actor {
    pub name: String,
    env: Env,
    mailbox: VecDeque<Message>,
    state: HashMap<String, Value>,
}

impl Actor {
    pub fn new(name: &str, source: &str) -> Result<Self, String> {
        let mut env = Env::new();
        load_source_into_env(&mut env, source)?;
        Ok(Self {
            name: name.to_string(),
            env,
            mailbox: VecDeque::new(),
            state: HashMap::new(),
        })
    }

    /// Send a message to this actor.
    pub fn send(&mut self, msg: Message) {
        self.mailbox.push_back(msg);
    }

    /// Process next message in mailbox.
    pub fn receive(&mut self) -> Option<Response> {
        let msg = self.mailbox.pop_front()?;

        // Expose message fields to the Vortex environment
        self.env.define("msg_from", Value::String(msg.from.clone()));
        self.env.define("msg_tag", Value::String(msg.tag.clone()));
        self.env
            .define("msg_payload", Value::String(msg.payload.clone()));

        // Expose actor state
        for (k, v) in &self.state {
            self.env.define(k, v.clone());
        }

        // Call handle_message if it exists, otherwise handle_{tag}
        let fn_name = if self.env.functions.contains_key("handle_message") {
            "handle_message".to_string()
        } else {
            format!("handle_{}", msg.tag)
        };

        let result = call_vortex_fn(
            &mut self.env,
            &fn_name,
            vec![Value::String(msg.payload)],
        );

        // After handling, read back any state variables the handler may have set
        for k in self.state.keys().cloned().collect::<Vec<_>>() {
            if let Some(v) = self.env.get(&k) {
                self.state.insert(k, v);
            }
        }

        match result {
            Ok(val) => Some(Response {
                status: 200,
                body: format!("{}", val),
            }),
            Err(e) => Some(Response {
                status: 500,
                body: e,
            }),
        }
    }

    /// Get actor's internal state.
    pub fn get_state(&self, key: &str) -> Option<&Value> {
        self.state.get(key)
    }

    /// Set actor's state.
    pub fn set_state(&mut self, key: &str, value: Value) {
        self.state.insert(key.to_string(), value.clone());
        self.env.define(key, value);
    }
}

/// Actor system: manages multiple actors.
pub struct ActorSystem {
    actors: HashMap<String, Actor>,
}

impl ActorSystem {
    pub fn new() -> Self {
        Self {
            actors: HashMap::new(),
        }
    }

    pub fn spawn(&mut self, name: &str, source: &str) -> Result<(), String> {
        let actor = Actor::new(name, source)?;
        self.actors.insert(name.to_string(), actor);
        Ok(())
    }

    pub fn send(&mut self, to: &str, msg: Message) {
        if let Some(actor) = self.actors.get_mut(to) {
            actor.send(msg);
        }
    }

    /// Process one message per actor. Returns (actor_name, response) pairs.
    pub fn tick(&mut self) -> Vec<(String, Response)> {
        let mut results = Vec::new();
        let names: Vec<String> = self.actors.keys().cloned().collect();
        for name in names {
            if let Some(actor) = self.actors.get_mut(&name) {
                if let Some(resp) = actor.receive() {
                    results.push((name, resp));
                }
            }
        }
        results
    }

    pub fn get_actor(&self, name: &str) -> Option<&Actor> {
        self.actors.get(name)
    }

    pub fn get_actor_mut(&mut self, name: &str) -> Option<&mut Actor> {
        self.actors.get_mut(name)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal Vortex program with a handler function
    const HANDLER_PROGRAM: &str = "fn handle_request(payload: String) -> String {\nreturn \"got: \" + payload\n}\nfn main() {\nlet x = 0\n}";

    // A program with mutable state — counter is set in the env externally
    const STATEFUL_PROGRAM: &str = "fn increment(payload: String) -> i64 {\ncounter = counter + 1\nreturn counter\n}\nfn main() {\nlet x = 0\n}";

    // A trivial predict function for inference tests
    const PREDICT_PROGRAM: &str = "fn predict(input: [f64]) -> [f64] {\nreturn input\n}\nfn main() {\nlet x = 0\n}";

    // Stream handler: identity - returns the data it receives
    const STREAM_PROGRAM: &str = "fn process_window(data: [f64]) -> [f64] {\nreturn data\n}\nfn main() {\nlet x = 0\n}";

    // Actor handler
    const ACTOR_PROGRAM: &str = "fn handle_message(payload: String) -> String {\nreturn \"echo: \" + payload\n}\nfn main() {\nlet x = 0\n}";

    #[test]
    fn test_server_load_and_handle_request() {
        let mut srv = VortexServer::new();
        srv.load_program(HANDLER_PROGRAM).unwrap();
        srv.register_handler("/hello", "handle_request");

        let resp = srv
            .process_event(&Event::Request {
                route: "/hello".into(),
                payload: "world".into(),
                response_tx: 0,
            })
            .unwrap();

        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "got: world");
    }

    #[test]
    fn test_server_state_persists_across_requests() {
        let mut srv = VortexServer::new();
        srv.load_program(STATEFUL_PROGRAM).unwrap();
        srv.register_handler("/inc", "increment");

        // Define counter in top-level scope
        srv.env_mut().define("counter", Value::Int(0));

        let resp1 = srv
            .process_event(&Event::Request {
                route: "/inc".into(),
                payload: "".into(),
                response_tx: 0,
            })
            .unwrap();
        let resp2 = srv
            .process_event(&Event::Request {
                route: "/inc".into(),
                payload: "".into(),
                response_tx: 0,
            })
            .unwrap();

        // Counter should increment across requests
        assert_eq!(resp1.body, "1");
        assert_eq!(resp2.body, "2");
    }

    #[test]
    fn test_inference_single() {
        let mut inf = InferenceServer::new(PREDICT_PROGRAM, 4).unwrap();
        let result = inf.infer(&[1.0, 2.0, 3.0], &[3]);
        assert_eq!(result.output, vec![1.0, 2.0, 3.0]);
        assert_eq!(inf.requests_served(), 1);
    }

    #[test]
    fn test_inference_batch() {
        let mut inf = InferenceServer::new(PREDICT_PROGRAM, 4).unwrap();
        let requests = vec![
            InferenceRequest {
                id: 0,
                input: vec![1.0],
                shape: vec![1],
                timestamp_us: 0,
            },
            InferenceRequest {
                id: 1,
                input: vec![2.0],
                shape: vec![1],
                timestamp_us: 0,
            },
        ];
        let results = inf.infer_batch(requests);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].output, vec![1.0]);
        assert_eq!(results[1].output, vec![2.0]);
    }

    #[test]
    fn test_inference_dynamic_batching_flush() {
        let mut inf = InferenceServer::new(PREDICT_PROGRAM, 2).unwrap();
        let id0 = inf.submit_request(&[10.0], &[1]);
        let id1 = inf.submit_request(&[20.0], &[1]);
        assert_eq!(inf.batch_queue_len(), 2);

        let results = inf.flush_batch();
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].id, id0);
        assert_eq!(results[1].id, id1);
        assert_eq!(inf.batch_queue_len(), 0);
    }

    #[test]
    fn test_stream_window_triggers() {
        let mut sp = StreamProcessor::new(STREAM_PROGRAM).unwrap();
        sp.register_stream("sensor", 3, 3, "process_window");

        // Push 3 data points — should trigger one window
        let r1 = sp.push_data("sensor", &[1.0, 2.0]);
        assert!(r1.is_empty()); // not full yet
        let r2 = sp.push_data("sensor", &[3.0, 4.0]);
        assert!(r2.is_empty());
        let r3 = sp.push_data("sensor", &[5.0, 6.0]);
        assert_eq!(r3.len(), 1); // window full, triggered
        // process_window returns identity of flat array: [1,2,3,4,5,6]
        assert_eq!(r3[0], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stream_overlapping_windows() {
        let mut sp = StreamProcessor::new(STREAM_PROGRAM).unwrap();
        // window_size=3, stride=1 => overlapping
        sp.register_stream("feed", 3, 1, "process_window");

        sp.push_data("feed", &[1.0]);
        sp.push_data("feed", &[2.0]);
        let r = sp.push_data("feed", &[3.0]);
        assert_eq!(r.len(), 1); // first window

        // Push one more — should trigger another window because stride=1
        let r2 = sp.push_data("feed", &[4.0]);
        assert_eq!(r2.len(), 1);
    }

    #[test]
    fn test_actor_send_receive() {
        let mut actor = Actor::new("echo", ACTOR_PROGRAM).unwrap();
        actor.send(Message {
            from: "test".into(),
            tag: "ping".into(),
            payload: "hello".into(),
        });
        let resp = actor.receive().unwrap();
        assert_eq!(resp.status, 200);
        assert_eq!(resp.body, "echo: hello");
    }

    #[test]
    fn test_actor_state_persists() {
        let mut actor = Actor::new("counter", ACTOR_PROGRAM).unwrap();
        actor.set_state("count", Value::Int(0));

        // State persists
        assert!(matches!(actor.get_state("count"), Some(Value::Int(0))));
        actor.set_state("count", Value::Int(42));
        assert!(matches!(actor.get_state("count"), Some(Value::Int(42))));
    }

    #[test]
    fn test_actor_system_two_actors() {
        let mut sys = ActorSystem::new();
        sys.spawn("a", ACTOR_PROGRAM).unwrap();
        sys.spawn("b", ACTOR_PROGRAM).unwrap();

        sys.send(
            "a",
            Message {
                from: "b".into(),
                tag: "greet".into(),
                payload: "hi from b".into(),
            },
        );
        sys.send(
            "b",
            Message {
                from: "a".into(),
                tag: "greet".into(),
                payload: "hi from a".into(),
            },
        );

        let results = sys.tick();
        assert_eq!(results.len(), 2);
        // Both actors should have responded
        let bodies: Vec<&str> = results.iter().map(|(_, r)| r.body.as_str()).collect();
        assert!(bodies.contains(&"echo: hi from b"));
        assert!(bodies.contains(&"echo: hi from a"));
    }

    #[test]
    fn test_checkpoint_save_and_restore() {
        let mut srv = VortexServer::new();
        srv.env.define("counter", Value::Int(42));
        srv.env.define("name", Value::String("test".into()));

        let path = "/tmp/vortex_test_checkpoint.txt";
        srv.checkpoint(path).unwrap();

        let restored = VortexServer::restore(path).unwrap();
        match restored.env().get("counter") {
            Some(Value::Int(42)) => {}
            other => panic!("expected Int(42), got {:?}", other),
        }
        match restored.env().get("name") {
            Some(Value::String(s)) if s == "test" => {}
            other => panic!("expected String(\"test\"), got {:?}", other),
        }

        // Cleanup
        let _ = std::fs::remove_file(path);
    }
}
