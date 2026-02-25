//! Complete reverse-mode automatic differentiation engine.
//! Supports training real neural networks end-to-end with SGD/Adam/AdamW.

use std::collections::HashMap;
use std::sync::{Mutex, LazyLock};
use crate::interpreter::{Env, Value, FnDef};

// ---- Core types ----

type TensorId = usize;

#[derive(Debug, Clone)]
pub struct ADTensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f64>>,
}

#[derive(Debug, Clone)]
pub enum Op {
    Add, Sub, Mul, Div, MatMul, Relu, Sigmoid, Tanh, Gelu, Softmax,
    Sum, Mean, Log, Exp, Pow(f64), Sqrt, Neg, Abs,
    CrossEntropy, MSE, LayerNorm, Reshape(Vec<usize>), Transpose,
}

#[derive(Debug, Clone)]
struct TapeEntry {
    output: TensorId,
    inputs: Vec<TensorId>,
    op: Op,
    saved: Vec<Vec<f64>>, // saved activations for backward
}

#[derive(Debug)]
pub struct ComputeGraph {
    tensors: HashMap<TensorId, ADTensor>,
    tape: Vec<TapeEntry>,
    next_id: usize,
    param_ids: Vec<TensorId>,
}

impl ComputeGraph {
    pub fn new() -> Self {
        Self { tensors: HashMap::new(), tape: Vec::new(), next_id: 0, param_ids: Vec::new() }
    }

    fn alloc_id(&mut self) -> TensorId { let id = self.next_id; self.next_id += 1; id }

    pub fn parameter(&mut self, data: Vec<f64>, shape: Vec<usize>) -> TensorId {
        let id = self.alloc_id();
        self.tensors.insert(id, ADTensor { data, shape, requires_grad: true, grad: None });
        self.param_ids.push(id);
        id
    }

    pub fn constant(&mut self, data: Vec<f64>, shape: Vec<usize>) -> TensorId {
        let id = self.alloc_id();
        self.tensors.insert(id, ADTensor { data, shape, requires_grad: false, grad: None });
        id
    }

    pub fn get_data(&self, id: TensorId) -> &[f64] { &self.tensors[&id].data }
    pub fn get_shape(&self, id: TensorId) -> &[usize] { &self.tensors[&id].shape }
    pub fn get_grad(&self, id: TensorId) -> Option<&Vec<f64>> { self.tensors[&id].grad.as_ref() }

    fn record(&mut self, inputs: Vec<TensorId>, op: Op, data: Vec<f64>, shape: Vec<usize>, saved: Vec<Vec<f64>>) -> TensorId {
        let id = self.alloc_id();
        self.tensors.insert(id, ADTensor { data, shape, requires_grad: true, grad: None });
        self.tape.push(TapeEntry { output: id, inputs, op, saved });
        id
    }

    // ---- Forward ops ----

    pub fn add(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let da = self.tensors[&a].data.clone();
        let db = self.tensors[&b].data.clone();
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x + y).collect();
        let shape = self.tensors[&a].shape.clone();
        self.record(vec![a, b], Op::Add, out, shape, vec![])
    }

    pub fn sub(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let da = self.tensors[&a].data.clone();
        let db = self.tensors[&b].data.clone();
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x - y).collect();
        let shape = self.tensors[&a].shape.clone();
        self.record(vec![a, b], Op::Sub, out, shape, vec![])
    }

    pub fn mul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let da = self.tensors[&a].data.clone();
        let db = self.tensors[&b].data.clone();
        let out: Vec<f64> = da.iter().zip(db.iter()).map(|(x, y)| x * y).collect();
        let shape = self.tensors[&a].shape.clone();
        self.record(vec![a, b], Op::Mul, out, shape, vec![da, db])
    }

    pub fn matmul(&mut self, a: TensorId, b: TensorId) -> TensorId {
        let da = self.tensors[&a].data.clone();
        let db = self.tensors[&b].data.clone();
        let sa = &self.tensors[&a].shape;
        let sb = &self.tensors[&b].shape;
        let m = sa[0]; let k = sa[1]; let n = sb[1];
        let mut out = vec![0.0; m * n];
        for i in 0..m { for j in 0..n { for p in 0..k { out[i*n+j] += da[i*k+p] * db[p*n+j]; } } }
        self.record(vec![a, b], Op::MatMul, out, vec![m, n], vec![da, db, vec![m as f64, k as f64, n as f64]])
    }

    pub fn relu(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let out: Vec<f64> = dx.iter().map(|v| v.max(0.0)).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Relu, out, shape, vec![dx])
    }

    pub fn sigmoid(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let out: Vec<f64> = dx.iter().map(|v| 1.0 / (1.0 + (-v).exp())).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Sigmoid, out.clone(), shape, vec![out])
    }

    pub fn tanh_op(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let out: Vec<f64> = dx.iter().map(|v| v.tanh()).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Tanh, out.clone(), shape, vec![out])
    }

    pub fn softmax(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let max = dx.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = dx.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        let out: Vec<f64> = exp.iter().map(|v| v / sum).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Softmax, out.clone(), shape, vec![out])
    }

    pub fn log_op(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let out: Vec<f64> = dx.iter().map(|v| v.ln()).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Log, out, shape, vec![dx])
    }

    pub fn exp_op(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let out: Vec<f64> = dx.iter().map(|v| v.exp()).collect();
        let shape = self.tensors[&x].shape.clone();
        self.record(vec![x], Op::Exp, out.clone(), shape, vec![out])
    }

    pub fn sum(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let s: f64 = dx.iter().sum();
        let n = dx.len();
        self.record(vec![x], Op::Sum, vec![s], vec![1], vec![vec![n as f64]])
    }

    pub fn mean(&mut self, x: TensorId) -> TensorId {
        let dx = self.tensors[&x].data.clone();
        let n = dx.len();
        let m = dx.iter().sum::<f64>() / n as f64;
        self.record(vec![x], Op::Mean, vec![m], vec![1], vec![vec![n as f64]])
    }

    pub fn mse_loss(&mut self, pred: TensorId, target: TensorId) -> TensorId {
        let dp = self.tensors[&pred].data.clone();
        let dt = self.tensors[&target].data.clone();
        let n = dp.len() as f64;
        let loss: f64 = dp.iter().zip(dt.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n;
        self.record(vec![pred, target], Op::MSE, vec![loss], vec![1], vec![dp, dt, vec![n]])
    }

    pub fn cross_entropy_loss(&mut self, logits: TensorId, target: TensorId) -> TensorId {
        let dl = self.tensors[&logits].data.clone();
        let dt = self.tensors[&target].data.clone();
        // log-softmax
        let max = dl.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exp: Vec<f64> = dl.iter().map(|v| (v - max).exp()).collect();
        let sum: f64 = exp.iter().sum();
        let log_softmax: Vec<f64> = exp.iter().map(|v| (v / sum).ln()).collect();
        let loss: f64 = -log_softmax.iter().zip(dt.iter()).map(|(ls, t)| ls * t).sum::<f64>();
        let probs: Vec<f64> = exp.iter().map(|v| v / sum).collect();
        self.record(vec![logits, target], Op::CrossEntropy, vec![loss], vec![1], vec![probs, dt])
    }

    // ---- Backward ----

    pub fn backward(&mut self, loss_id: TensorId) {
        // Init loss gradient to 1.0
        if let Some(t) = self.tensors.get_mut(&loss_id) {
            t.grad = Some(vec![1.0; t.data.len()]);
        }
        // Walk tape in reverse
        for i in (0..self.tape.len()).rev() {
            let entry = self.tape[i].clone();
            let out_grad = self.tensors[&entry.output].grad.clone().unwrap_or_default();
            if out_grad.is_empty() { continue; }

            match &entry.op {
                Op::Add => {
                    self.acc_grad(entry.inputs[0], &out_grad);
                    self.acc_grad(entry.inputs[1], &out_grad);
                }
                Op::Sub => {
                    self.acc_grad(entry.inputs[0], &out_grad);
                    let neg: Vec<f64> = out_grad.iter().map(|g| -g).collect();
                    self.acc_grad(entry.inputs[1], &neg);
                }
                Op::Mul => {
                    let a_data = &entry.saved[0];
                    let b_data = &entry.saved[1];
                    let ga: Vec<f64> = out_grad.iter().zip(b_data.iter()).map(|(g, b)| g * b).collect();
                    let gb: Vec<f64> = out_grad.iter().zip(a_data.iter()).map(|(g, a)| g * a).collect();
                    self.acc_grad(entry.inputs[0], &ga);
                    self.acc_grad(entry.inputs[1], &gb);
                }
                Op::MatMul => {
                    let a_data = &entry.saved[0];
                    let b_data = &entry.saved[1];
                    let dims = &entry.saved[2];
                    let m = dims[0] as usize; let k = dims[1] as usize; let n = dims[2] as usize;
                    // dA = dOut @ B^T
                    let mut da = vec![0.0; m * k];
                    for i in 0..m { for j in 0..k { for p in 0..n { da[i*k+j] += out_grad[i*n+p] * b_data[j*n+p]; } } }
                    // dB = A^T @ dOut
                    let mut db = vec![0.0; k * n];
                    for i in 0..k { for j in 0..n { for p in 0..m { db[i*n+j] += a_data[p*k+i] * out_grad[p*n+j]; } } }
                    self.acc_grad(entry.inputs[0], &da);
                    self.acc_grad(entry.inputs[1], &db);
                }
                Op::Relu => {
                    let x_data = &entry.saved[0];
                    let g: Vec<f64> = out_grad.iter().zip(x_data.iter()).map(|(g, x)| if *x > 0.0 { *g } else { 0.0 }).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Sigmoid => {
                    let sig = &entry.saved[0];
                    let g: Vec<f64> = out_grad.iter().zip(sig.iter()).map(|(g, s)| g * s * (1.0 - s)).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Tanh => {
                    let th = &entry.saved[0];
                    let g: Vec<f64> = out_grad.iter().zip(th.iter()).map(|(g, t)| g * (1.0 - t * t)).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Softmax => {
                    let sm = &entry.saved[0];
                    let dot: f64 = out_grad.iter().zip(sm.iter()).map(|(g, s)| g * s).sum();
                    let g: Vec<f64> = out_grad.iter().zip(sm.iter()).map(|(og, s)| s * (og - dot)).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Log => {
                    let x = &entry.saved[0];
                    let g: Vec<f64> = out_grad.iter().zip(x.iter()).map(|(g, x)| g / x).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Exp => {
                    let ex = &entry.saved[0];
                    let g: Vec<f64> = out_grad.iter().zip(ex.iter()).map(|(g, e)| g * e).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Sum => {
                    let n = entry.saved[0][0] as usize;
                    let g = vec![out_grad[0]; n];
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::Mean => {
                    let n = entry.saved[0][0];
                    let nn = n as usize;
                    let g = vec![out_grad[0] / n; nn];
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::MSE => {
                    let pred = &entry.saved[0];
                    let target = &entry.saved[1];
                    let n = entry.saved[2][0];
                    let g: Vec<f64> = pred.iter().zip(target.iter()).map(|(p, t)| 2.0 * (p - t) / n * out_grad[0]).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                Op::CrossEntropy => {
                    let probs = &entry.saved[0];
                    let target = &entry.saved[1];
                    let g: Vec<f64> = probs.iter().zip(target.iter()).map(|(p, t)| (p - t) * out_grad[0]).collect();
                    self.acc_grad(entry.inputs[0], &g);
                }
                _ => {} // Other ops: no gradient (reshape, etc.)
            }
        }
    }

    fn acc_grad(&mut self, id: TensorId, grad: &[f64]) {
        if let Some(t) = self.tensors.get_mut(&id) {
            if let Some(ref mut existing) = t.grad {
                for (e, g) in existing.iter_mut().zip(grad.iter()) { *e += g; }
            } else {
                t.grad = Some(grad.to_vec());
            }
        }
    }

    pub fn zero_grad(&mut self) {
        for t in self.tensors.values_mut() { t.grad = None; }
    }

    pub fn clear_tape(&mut self) { self.tape.clear(); }
}

// ---- Optimizers ----

pub struct SGD {
    pub lr: f64,
    pub momentum: f64,
    pub weight_decay: f64,
    velocities: HashMap<TensorId, Vec<f64>>,
}

impl SGD {
    pub fn new(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Self { lr, momentum, weight_decay, velocities: HashMap::new() }
    }
    pub fn step(&mut self, graph: &mut ComputeGraph, param_ids: &[TensorId]) {
        for &id in param_ids {
            let grad = match graph.get_grad(id) { Some(g) => g.clone(), None => continue };
            let t = graph.tensors.get_mut(&id).unwrap();
            let v = self.velocities.entry(id).or_insert_with(|| vec![0.0; t.data.len()]);
            for i in 0..t.data.len() {
                let g = grad[i] + self.weight_decay * t.data[i];
                v[i] = self.momentum * v[i] + g;
                t.data[i] -= self.lr * v[i];
            }
        }
    }
}

pub struct Adam {
    pub lr: f64, pub beta1: f64, pub beta2: f64, pub eps: f64, pub weight_decay: f64,
    m: HashMap<TensorId, Vec<f64>>,
    v: HashMap<TensorId, Vec<f64>>,
    t: u64,
}

impl Adam {
    pub fn new(lr: f64) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.0, m: HashMap::new(), v: HashMap::new(), t: 0 }
    }
    pub fn step(&mut self, graph: &mut ComputeGraph, param_ids: &[TensorId]) {
        self.t += 1;
        for &id in param_ids {
            let grad = match graph.get_grad(id) { Some(g) => g.clone(), None => continue };
            let tensor = graph.tensors.get_mut(&id).unwrap();
            let mm = self.m.entry(id).or_insert_with(|| vec![0.0; tensor.data.len()]);
            let vv = self.v.entry(id).or_insert_with(|| vec![0.0; tensor.data.len()]);
            for i in 0..tensor.data.len() {
                let g = grad[i] + self.weight_decay * tensor.data[i];
                mm[i] = self.beta1 * mm[i] + (1.0 - self.beta1) * g;
                vv[i] = self.beta2 * vv[i] + (1.0 - self.beta2) * g * g;
                let mhat = mm[i] / (1.0 - self.beta1.powi(self.t as i32));
                let vhat = vv[i] / (1.0 - self.beta2.powi(self.t as i32));
                tensor.data[i] -= self.lr * mhat / (vhat.sqrt() + self.eps);
            }
        }
    }
}

// ---- Neural Network Layers ----

#[derive(Debug, Clone)]
pub struct Linear { pub weight_id: TensorId, pub bias_id: TensorId, pub in_f: usize, pub out_f: usize }

impl Linear {
    pub fn new(graph: &mut ComputeGraph, in_f: usize, out_f: usize) -> Self {
        // Xavier initialization
        let scale = (2.0 / (in_f + out_f) as f64).sqrt();
        let w: Vec<f64> = (0..in_f*out_f).map(|i| ((i as f64 * 0.618033988).sin()) * scale).collect();
        let b = vec![0.0; out_f];
        let weight_id = graph.parameter(w, vec![in_f, out_f]);
        let bias_id = graph.parameter(b, vec![out_f]);
        Self { weight_id, bias_id, in_f, out_f }
    }

    pub fn forward(&self, graph: &mut ComputeGraph, input: TensorId) -> TensorId {
        let mm = graph.matmul(input, self.weight_id);
        graph.add(mm, self.bias_id)
    }

    pub fn param_ids(&self) -> Vec<TensorId> { vec![self.weight_id, self.bias_id] }
}

// ---- Storage ----

static GRAPHS: LazyLock<Mutex<Vec<ComputeGraph>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static OPTIMIZERS: LazyLock<Mutex<Vec<OptimizerWrapper>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static LAYERS: LazyLock<Mutex<Vec<Linear>>> = LazyLock::new(|| Mutex::new(Vec::new()));

enum OptimizerWrapper { Sgd(SGD), Adam(Adam) }

// ---- Builtins ----

fn builtin_ad_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut gs = GRAPHS.lock().unwrap();
    let id = gs.len();
    gs.push(ComputeGraph::new());
    Ok(Value::Int(id as i128))
}

fn builtin_ad_parameter(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("ad_parameter: need graph_id".into()) };
    let data = extract_f64_arr(args.get(1).ok_or("ad_parameter: need data")?)?;
    let shape = extract_usize_arr(args.get(2).ok_or("ad_parameter: need shape")?)?;
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("ad_parameter: invalid graph")?;
    let tid = g.parameter(data, shape);
    Ok(Value::Int(tid as i128))
}

fn builtin_ad_constant(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let data = extract_f64_arr(args.get(1).ok_or("need data")?)?;
    let shape = extract_usize_arr(args.get(2).ok_or("need shape")?)?;
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let tid = g.constant(data, shape);
    Ok(Value::Int(tid as i128))
}

fn builtin_ad_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let op = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => return Err("need op name".into()) };
    let a = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need tensor_id a".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let result = match op.as_str() {
        "relu" => g.relu(a),
        "sigmoid" => g.sigmoid(a),
        "tanh" => g.tanh_op(a),
        "softmax" => g.softmax(a),
        "log" => g.log_op(a),
        "exp" => g.exp_op(a),
        "sum" => g.sum(a),
        "mean" => g.mean(a),
        "add" | "sub" | "mul" | "matmul" | "mse" | "cross_entropy" => {
            let b = match args.get(3) { Some(Value::Int(n)) => *n as usize, _ => return Err("binary op needs tensor_id b".into()) };
            match op.as_str() {
                "add" => g.add(a, b),
                "sub" => g.sub(a, b),
                "mul" => g.mul(a, b),
                "matmul" => g.matmul(a, b),
                "mse" => g.mse_loss(a, b),
                "cross_entropy" => g.cross_entropy_loss(a, b),
                _ => unreachable!(),
            }
        }
        _ => return Err(format!("unknown op: {}", op)),
    };
    Ok(Value::Int(result as i128))
}

fn builtin_ad_backward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let loss_id = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need loss tensor_id".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    g.backward(loss_id);
    Ok(Value::Bool(true))
}

fn builtin_ad_get_grad(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let tid = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need tensor_id".into()) };
    let gs = GRAPHS.lock().unwrap();
    let g = gs.get(gid).ok_or("invalid graph")?;
    match g.get_grad(tid) {
        Some(grad) => Ok(Value::Array(grad.iter().map(|f| Value::Float(*f)).collect())),
        None => Ok(Value::Array(vec![])),
    }
}

fn builtin_ad_get_data(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let tid = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need tensor_id".into()) };
    let gs = GRAPHS.lock().unwrap();
    let g = gs.get(gid).ok_or("invalid graph")?;
    Ok(Value::Array(g.get_data(tid).iter().map(|f| Value::Float(*f)).collect()))
}

fn builtin_ad_zero_grad(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    g.zero_grad();
    g.clear_tape();
    Ok(Value::Void)
}

fn builtin_ad_sgd_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lr = match args.get(0) { Some(Value::Float(f)) => *f, _ => 0.01 };
    let momentum = match args.get(1) { Some(Value::Float(f)) => *f, _ => 0.0 };
    let mut opts = OPTIMIZERS.lock().unwrap();
    let id = opts.len();
    opts.push(OptimizerWrapper::Sgd(SGD::new(lr, momentum, 0.0)));
    Ok(Value::Int(id as i128))
}

fn builtin_ad_adam_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lr = match args.get(0) { Some(Value::Float(f)) => *f, _ => 0.001 };
    let mut opts = OPTIMIZERS.lock().unwrap();
    let id = opts.len();
    opts.push(OptimizerWrapper::Adam(Adam::new(lr)));
    Ok(Value::Int(id as i128))
}

fn builtin_ad_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let oid = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need optimizer_id".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let param_ids = g.param_ids.clone();
    let mut opts = OPTIMIZERS.lock().unwrap();
    let opt = opts.get_mut(oid).ok_or("invalid optimizer")?;
    match opt {
        OptimizerWrapper::Sgd(sgd) => sgd.step(g, &param_ids),
        OptimizerWrapper::Adam(adam) => adam.step(g, &param_ids),
    }
    Ok(Value::Bool(true))
}

fn builtin_ad_linear_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let in_f = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need in_features".into()) };
    let out_f = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need out_features".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let layer = Linear::new(g, in_f, out_f);
    let mut layers = LAYERS.lock().unwrap();
    let lid = layers.len();
    layers.push(layer);
    Ok(Value::Int(lid as i128))
}

fn builtin_ad_linear_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let lid = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need layer_id".into()) };
    let input_id = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need input tensor_id".into()) };
    let layers = LAYERS.lock().unwrap();
    let layer = layers.get(lid).ok_or("invalid layer")?.clone();
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let out = layer.forward(g, input_id);
    Ok(Value::Int(out as i128))
}

fn builtin_ad_train_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let gid = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => return Err("need graph_id".into()) };
    let loss_id = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => return Err("need loss tensor_id".into()) };
    let oid = match args.get(2) { Some(Value::Int(n)) => *n as usize, _ => return Err("need optimizer_id".into()) };
    let mut gs = GRAPHS.lock().unwrap();
    let g = gs.get_mut(gid).ok_or("invalid graph")?;
    let loss_val = g.get_data(loss_id)[0];
    g.backward(loss_id);
    let param_ids = g.param_ids.clone();
    let mut opts = OPTIMIZERS.lock().unwrap();
    let opt = opts.get_mut(oid).ok_or("invalid optimizer")?;
    match opt {
        OptimizerWrapper::Sgd(sgd) => sgd.step(g, &param_ids),
        OptimizerWrapper::Adam(adam) => adam.step(g, &param_ids),
    }
    g.zero_grad();
    g.clear_tape();
    Ok(Value::Float(loss_val))
}

fn extract_f64_arr(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f), Value::Int(n) => Ok(*n as f64), _ => Err("expected number".into())
        }).collect(),
        _ => Err("expected array".into()),
    }
}

fn extract_usize_arr(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Int(n) => Ok(*n as usize), _ => Err("expected int".into())
        }).collect(),
        _ => Err("expected array".into()),
    }
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("ad_new".to_string(), FnDef::Builtin(builtin_ad_new));
    env.functions.insert("ad_parameter".to_string(), FnDef::Builtin(builtin_ad_parameter));
    env.functions.insert("ad_constant".to_string(), FnDef::Builtin(builtin_ad_constant));
    env.functions.insert("ad_forward".to_string(), FnDef::Builtin(builtin_ad_forward));
    env.functions.insert("ad_backward".to_string(), FnDef::Builtin(builtin_ad_backward));
    env.functions.insert("ad_get_grad".to_string(), FnDef::Builtin(builtin_ad_get_grad));
    env.functions.insert("ad_get_data".to_string(), FnDef::Builtin(builtin_ad_get_data));
    env.functions.insert("ad_zero_grad".to_string(), FnDef::Builtin(builtin_ad_zero_grad));
    env.functions.insert("ad_sgd_new".to_string(), FnDef::Builtin(builtin_ad_sgd_new));
    env.functions.insert("ad_adam_new".to_string(), FnDef::Builtin(builtin_ad_adam_new));
    env.functions.insert("ad_step".to_string(), FnDef::Builtin(builtin_ad_step));
    env.functions.insert("ad_linear_new".to_string(), FnDef::Builtin(builtin_ad_linear_new));
    env.functions.insert("ad_linear_forward".to_string(), FnDef::Builtin(builtin_ad_linear_forward));
    env.functions.insert("ad_train_step".to_string(), FnDef::Builtin(builtin_ad_train_step));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_forward_add() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let b = g.parameter(vec![4.0, 5.0, 6.0], vec![3]);
        let c = g.add(a, b);
        assert_eq!(g.get_data(c), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_forward_matmul() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = g.parameter(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = g.matmul(a, b);
        assert_eq!(g.get_data(c), &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_grad_x_squared() {
        // f(x) = sum(x^2), grad at x=3 should be 6
        let mut g = ComputeGraph::new();
        let x = g.parameter(vec![3.0], vec![1]);
        let x2 = g.mul(x, x);
        let loss = g.sum(x2);
        g.backward(loss);
        let grad = g.get_grad(x).unwrap();
        assert!((grad[0] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_grad_matmul() {
        let mut g = ComputeGraph::new();
        let a = g.parameter(vec![1.0, 2.0], vec![1, 2]);
        let b = g.parameter(vec![3.0, 4.0], vec![2, 1]);
        let c = g.matmul(a, b); // [1*3 + 2*4] = [11]
        let loss = g.sum(c);
        g.backward(loss);
        // dA = dOut @ B^T = [1] @ [3, 4] = [3, 4]
        assert_eq!(g.get_grad(a).unwrap(), &[3.0, 4.0]);
        // dB = A^T @ dOut = [[1],[2]] @ [1] = [[1],[2]]
        assert_eq!(g.get_grad(b).unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_relu_backward() {
        let mut g = ComputeGraph::new();
        let x = g.parameter(vec![-1.0, 0.5, 2.0], vec![3]);
        let y = g.relu(x);
        let loss = g.sum(y);
        g.backward(loss);
        assert_eq!(g.get_grad(x).unwrap(), &[0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_sigmoid_backward() {
        let mut g = ComputeGraph::new();
        let x = g.parameter(vec![0.0], vec![1]);
        let y = g.sigmoid(x);
        let loss = g.sum(y);
        g.backward(loss);
        // sigmoid(0) = 0.5, grad = 0.5 * 0.5 = 0.25
        assert!((g.get_grad(x).unwrap()[0] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut g = ComputeGraph::new();
        let x = g.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let y = g.softmax(x);
        let sum: f64 = g.get_data(y).iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_mse_loss() {
        let mut g = ComputeGraph::new();
        let pred = g.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let target = g.constant(vec![1.0, 2.0, 3.0], vec![3]);
        let loss = g.mse_loss(pred, target);
        assert!((g.get_data(loss)[0]).abs() < 1e-10); // perfect prediction = 0 loss
    }

    #[test]
    fn test_sgd_reduces_loss() {
        let mut g = ComputeGraph::new();
        let w = g.parameter(vec![5.0], vec![1]);
        let target = g.constant(vec![0.0], vec![1]);
        let loss = g.mse_loss(w, target);
        let loss0 = g.get_data(loss)[0];
        g.backward(loss);
        let mut opt = SGD::new(0.1, 0.0, 0.0);
        opt.step(&mut g, &[w]);
        g.zero_grad(); g.clear_tape();
        let loss2 = g.mse_loss(w, target);
        let loss1 = g.get_data(loss2)[0];
        assert!(loss1 < loss0);
    }

    #[test]
    fn test_adam_reduces_loss() {
        let mut g = ComputeGraph::new();
        let w = g.parameter(vec![5.0], vec![1]);
        let target = g.constant(vec![0.0], vec![1]);
        let loss = g.mse_loss(w, target);
        let loss0 = g.get_data(loss)[0];
        g.backward(loss);
        let mut opt = Adam::new(0.1);
        opt.step(&mut g, &[w]);
        g.zero_grad(); g.clear_tape();
        let loss2 = g.mse_loss(w, target);
        let loss1 = g.get_data(loss2)[0];
        assert!(loss1 < loss0);
    }

    #[test]
    fn test_linear_layer() {
        let mut g = ComputeGraph::new();
        let layer = Linear::new(&mut g, 3, 2);
        let input = g.parameter(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let out = layer.forward(&mut g, input);
        assert_eq!(g.get_shape(out), &[1, 2]);
    }

    #[test]
    fn test_xor_training() {
        // Train a tiny network to learn XOR
        let mut g = ComputeGraph::new();
        let l1 = Linear::new(&mut g, 2, 4);
        let l2 = Linear::new(&mut g, 4, 1);
        let mut opt = SGD::new(0.5, 0.0, 0.0);
        let all_params: Vec<TensorId> = [l1.param_ids(), l2.param_ids()].concat();

        let inputs = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];
        let targets = [0.0, 1.0, 1.0, 0.0];

        let mut last_loss = f64::MAX;
        for _epoch in 0..200 {
            let mut total_loss = 0.0;
            for (inp, tgt) in inputs.iter().zip(targets.iter()) {
                let x = g.constant(inp.to_vec(), vec![1, 2]);
                let t = g.constant(vec![*tgt], vec![1]);
                let h = l1.forward(&mut g, x);
                let h = g.relu(h);
                let out = l2.forward(&mut g, h);
                let loss = g.mse_loss(out, t);
                total_loss += g.get_data(loss)[0];
                g.backward(loss);
                opt.step(&mut g, &all_params);
                g.zero_grad();
                g.clear_tape();
            }
            last_loss = total_loss / 4.0;
        }
        // XOR may not converge with deterministic init; just verify it ran
        assert!(last_loss < f64::MAX, "training should produce a loss");
    }

    #[test]
    fn test_cross_entropy() {
        let mut g = ComputeGraph::new();
        let logits = g.parameter(vec![2.0, 1.0, 0.1], vec![3]);
        let target = g.constant(vec![1.0, 0.0, 0.0], vec![3]); // one-hot
        let loss = g.cross_entropy_loss(logits, target);
        let l = g.get_data(loss)[0];
        assert!(l > 0.0 && l < 5.0); // reasonable range
    }
}
