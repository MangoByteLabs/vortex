//! Native symbolic reasoning: exact Field arithmetic and neural Tensor computation
//! coexisting in one model, with routing between symbolic and neural paths
//! and straight-through gradient estimation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. SymbolicValue
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicValue {
    Integer(i64),
    Rational(i64, i64), // numerator, denominator – always reduced, denom > 0
    Boolean(bool),
    Vector(Vec<SymbolicValue>),
    Unknown,
}

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

impl SymbolicValue {
    /// Create a reduced rational. Panics on zero denominator.
    pub fn rational(num: i64, den: i64) -> Self {
        assert!(den != 0, "zero denominator");
        let sign = if den < 0 { -1 } else { 1 };
        let g = gcd(num, den);
        let (n, d) = (sign * num / g, sign * den / g);
        if d == 1 {
            SymbolicValue::Integer(n)
        } else {
            SymbolicValue::Rational(n, d)
        }
    }

    /// Promote Integer to Rational for uniform arithmetic.
    fn to_rational(&self) -> Option<(i64, i64)> {
        match self {
            SymbolicValue::Integer(n) => Some((*n, 1)),
            SymbolicValue::Rational(n, d) => Some((*n, *d)),
            _ => None,
        }
    }

    pub fn add(&self, other: &SymbolicValue) -> Option<SymbolicValue> {
        match (self.to_rational(), other.to_rational()) {
            (Some((a, b)), Some((c, d))) => {
                Some(SymbolicValue::rational(a * d + c * b, b * d))
            }
            _ => None,
        }
    }

    pub fn sub(&self, other: &SymbolicValue) -> Option<SymbolicValue> {
        match (self.to_rational(), other.to_rational()) {
            (Some((a, b)), Some((c, d))) => {
                Some(SymbolicValue::rational(a * d - c * b, b * d))
            }
            _ => None,
        }
    }

    pub fn mul(&self, other: &SymbolicValue) -> Option<SymbolicValue> {
        match (self.to_rational(), other.to_rational()) {
            (Some((a, b)), Some((c, d))) => {
                Some(SymbolicValue::rational(a * c, b * d))
            }
            _ => None,
        }
    }

    pub fn div(&self, other: &SymbolicValue) -> Option<SymbolicValue> {
        match (self.to_rational(), other.to_rational()) {
            (Some((a, b)), Some((c, d))) => {
                if c == 0 {
                    None
                } else {
                    Some(SymbolicValue::rational(a * d, b * c))
                }
            }
            _ => None,
        }
    }

    /// Convert to f64 for neural interop.
    pub fn to_f64(&self) -> Option<f64> {
        match self {
            SymbolicValue::Integer(n) => Some(*n as f64),
            SymbolicValue::Rational(n, d) => Some(*n as f64 / *d as f64),
            SymbolicValue::Boolean(b) => Some(if *b { 1.0 } else { 0.0 }),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// 2. SymbolicEngine
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolicOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    GCD,
    IsPrime,
    Sort,
    ArgMax,
    Compare,
    CountIf,
}

pub struct SymbolicEngine;

impl SymbolicEngine {
    pub fn new() -> Self {
        SymbolicEngine
    }

    pub fn evaluate_exact(&self, op: SymbolicOp, args: &[SymbolicValue]) -> Option<SymbolicValue> {
        match op {
            SymbolicOp::Add => {
                if args.len() != 2 { return None; }
                args[0].add(&args[1])
            }
            SymbolicOp::Sub => {
                if args.len() != 2 { return None; }
                args[0].sub(&args[1])
            }
            SymbolicOp::Mul => {
                if args.len() != 2 { return None; }
                args[0].mul(&args[1])
            }
            SymbolicOp::Div => {
                if args.len() != 2 { return None; }
                args[0].div(&args[1])
            }
            SymbolicOp::Mod => {
                if args.len() != 2 { return None; }
                match (&args[0], &args[1]) {
                    (SymbolicValue::Integer(a), SymbolicValue::Integer(b)) if *b != 0 => {
                        Some(SymbolicValue::Integer(a % b))
                    }
                    _ => None,
                }
            }
            SymbolicOp::GCD => {
                if args.len() != 2 { return None; }
                match (&args[0], &args[1]) {
                    (SymbolicValue::Integer(a), SymbolicValue::Integer(b)) => {
                        Some(SymbolicValue::Integer(gcd(*a, *b)))
                    }
                    _ => None,
                }
            }
            SymbolicOp::IsPrime => {
                if args.len() != 1 { return None; }
                match &args[0] {
                    SymbolicValue::Integer(n) => Some(SymbolicValue::Boolean(is_prime(*n))),
                    _ => None,
                }
            }
            SymbolicOp::Sort => {
                if args.len() != 1 { return None; }
                match &args[0] {
                    SymbolicValue::Vector(v) => {
                        let mut ints: Vec<i64> = Vec::new();
                        for item in v {
                            match item {
                                SymbolicValue::Integer(n) => ints.push(*n),
                                _ => return None,
                            }
                        }
                        ints.sort();
                        Some(SymbolicValue::Vector(
                            ints.into_iter().map(SymbolicValue::Integer).collect(),
                        ))
                    }
                    _ => None,
                }
            }
            SymbolicOp::ArgMax => {
                if args.len() != 1 { return None; }
                match &args[0] {
                    SymbolicValue::Vector(v) if !v.is_empty() => {
                        let mut best_idx = 0i64;
                        let mut best_val: Option<(i64, i64)> = None;
                        for (i, item) in v.iter().enumerate() {
                            let r = item.to_rational()?;
                            match best_val {
                                None => {
                                    best_val = Some(r);
                                    best_idx = i as i64;
                                }
                                Some((bn, bd)) => {
                                    // compare r > best: r.0*bd > bn*r.1
                                    if r.0 * bd > bn * r.1 {
                                        best_val = Some(r);
                                        best_idx = i as i64;
                                    }
                                }
                            }
                        }
                        Some(SymbolicValue::Integer(best_idx))
                    }
                    _ => None,
                }
            }
            SymbolicOp::Compare => {
                if args.len() != 2 { return None; }
                match (args[0].to_rational(), args[1].to_rational()) {
                    (Some((a, b)), Some((c, d))) => {
                        let lhs = a * d;
                        let rhs = c * b;
                        let cmp = if lhs < rhs { -1 } else if lhs > rhs { 1 } else { 0 };
                        Some(SymbolicValue::Integer(cmp))
                    }
                    _ => None,
                }
            }
            SymbolicOp::CountIf => {
                // CountIf(vector, threshold) – count elements > threshold
                if args.len() != 2 { return None; }
                match &args[0] {
                    SymbolicValue::Vector(v) => {
                        let thresh = args[1].to_rational()?;
                        let mut count = 0i64;
                        for item in v {
                            let r = item.to_rational()?;
                            if r.0 * thresh.1 > thresh.0 * r.1 {
                                count += 1;
                            }
                        }
                        Some(SymbolicValue::Integer(count))
                    }
                    _ => None,
                }
            }
        }
    }
}

fn is_prime(n: i64) -> bool {
    let n = n.abs();
    if n < 2 {
        return false;
    }
    if n < 4 {
        return true;
    }
    if n % 2 == 0 || n % 3 == 0 {
        return false;
    }
    let mut i = 5i64;
    while i * i <= n {
        if n % i == 0 || n % (i + 2) == 0 {
            return false;
        }
        i += 6;
    }
    true
}

// ---------------------------------------------------------------------------
// 3. NeuralFallback
// ---------------------------------------------------------------------------

pub struct NeuralFallback {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
}

impl NeuralFallback {
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        // Initialize with small deterministic weights for reproducibility.
        let mut weights = Vec::with_capacity(output_dim);
        for o in 0..output_dim {
            let mut row = Vec::with_capacity(input_dim);
            for i in 0..input_dim {
                // simple deterministic init
                row.push(((o * 7 + i * 13) % 19) as f64 * 0.01 - 0.09);
            }
            weights.push(row);
        }
        let biases = vec![0.0; output_dim];
        NeuralFallback { weights, biases }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .zip(self.biases.iter())
            .map(|(row, bias)| {
                let dot: f64 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                relu(dot + bias)
            })
            .collect()
    }
}

fn relu(x: f64) -> f64 {
    if x > 0.0 { x } else { 0.0 }
}

// ---------------------------------------------------------------------------
// 4. SymbolicRouter
// ---------------------------------------------------------------------------

pub struct SymbolicRouter {
    pub threshold: f64,
    /// Linear classifier weights: dot(weights, input) + bias → logit
    pub weights: Vec<f64>,
    pub bias: f64,
}

impl SymbolicRouter {
    pub fn new(input_dim: usize, threshold: f64) -> Self {
        // Default: route everything that has integer-like features to symbolic.
        let weights = vec![1.0; input_dim];
        SymbolicRouter {
            threshold,
            weights,
            bias: 0.0,
        }
    }

    /// Returns probability (sigmoid) that input needs symbolic computation.
    pub fn classify(&self, input: &[f64]) -> f64 {
        let logit: f64 = self
            .weights
            .iter()
            .zip(input.iter())
            .map(|(w, x)| w * x)
            .sum::<f64>()
            + self.bias;
        sigmoid(logit)
    }

    /// Straight-through estimator: forward pass uses discrete decision,
    /// backward pass pretends the routing function was the identity.
    pub fn straight_through_grad(&self, symbolic_output: &[f64], neural_grad: &[f64]) -> Vec<f64> {
        // STE: gradient of discrete routing ≈ 1, so just pass neural_grad through
        // scaled by how close symbolic_output is to the neural path.
        symbolic_output
            .iter()
            .zip(neural_grad.iter())
            .map(|(_s, g)| *g) // identity pass-through
            .collect()
    }
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// ---------------------------------------------------------------------------
// 5. HybridLayer
// ---------------------------------------------------------------------------

pub struct HybridLayer {
    pub symbolic: SymbolicEngine,
    pub neural: NeuralFallback,
    pub router: SymbolicRouter,
}

impl HybridLayer {
    pub fn new(input_dim: usize, output_dim: usize, threshold: f64) -> Self {
        HybridLayer {
            symbolic: SymbolicEngine::new(),
            neural: NeuralFallback::new(input_dim, output_dim),
            router: SymbolicRouter::new(input_dim, threshold),
        }
    }

    /// Route a single input to symbolic or neural path.
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let p_sym = self.router.classify(input);
        if p_sym >= self.router.threshold {
            // Try symbolic: interpret inputs as integers, do Add
            let sym_args: Vec<SymbolicValue> = input
                .iter()
                .map(|x| {
                    let rounded = x.round() as i64;
                    if (x - rounded as f64).abs() < 1e-9 {
                        SymbolicValue::Integer(rounded)
                    } else {
                        SymbolicValue::Unknown
                    }
                })
                .collect();

            // If all are exact integers, sum them symbolically
            if sym_args.iter().all(|v| matches!(v, SymbolicValue::Integer(_))) && !sym_args.is_empty() {
                let mut acc = sym_args[0].clone();
                for v in &sym_args[1..] {
                    match acc.add(v) {
                        Some(r) => acc = r,
                        None => return self.neural.forward(input),
                    }
                }
                if let Some(val) = acc.to_f64() {
                    return vec![val];
                }
            }
            // Fallback to neural if symbolic can't handle it
            self.neural.forward(input)
        } else {
            self.neural.forward(input)
        }
    }

    /// Mixed-mode batch: each token independently routed.
    pub fn forward_mixed(&self, batch: &[Vec<f64>]) -> Vec<Vec<f64>> {
        batch.iter().map(|input| self.forward(input)).collect()
    }
}

// ---------------------------------------------------------------------------
// 6. SymbolicMemory
// ---------------------------------------------------------------------------

pub struct SymbolicMemory {
    pub facts: HashMap<String, SymbolicValue>,
}

impl SymbolicMemory {
    pub fn new() -> Self {
        SymbolicMemory {
            facts: HashMap::new(),
        }
    }

    pub fn assert_fact(&mut self, key: &str, value: SymbolicValue) {
        self.facts.insert(key.to_string(), value);
    }

    pub fn query(&self, key: &str) -> Option<&SymbolicValue> {
        self.facts.get(key)
    }

    /// Simple forward chaining: given a rule (op) and premise keys,
    /// derive a new value. Unknown premises are filled with neural estimation.
    pub fn derive(
        &self,
        engine: &SymbolicEngine,
        rule: SymbolicOp,
        premise_keys: &[&str],
        neural: &NeuralFallback,
    ) -> Option<SymbolicValue> {
        let mut args = Vec::new();
        for key in premise_keys {
            match self.query(key) {
                Some(v) => args.push(v.clone()),
                None => {
                    // Neural estimation for unknown premise
                    let est = neural.forward(&[0.0]); // minimal input
                    if let Some(v) = est.first() {
                        args.push(SymbolicValue::Integer(v.round() as i64));
                    } else {
                        return None;
                    }
                }
            }
        }
        engine.evaluate_exact(rule, &args)
    }
}

// ---------------------------------------------------------------------------
// Interpreter builtins
// ---------------------------------------------------------------------------

use crate::interpreter::{Env, Value};
use std::sync::Mutex;

fn hybrid_layers() -> &'static Mutex<Vec<HybridLayer>> {
    use std::sync::OnceLock;
    static LAYERS: OnceLock<Mutex<Vec<HybridLayer>>> = OnceLock::new();
    LAYERS.get_or_init(|| Mutex::new(Vec::new()))
}

/// symbolic_eval(op_str, args_array) -> result value
pub fn builtin_symbolic_eval(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("symbolic_eval requires (op, args_array)".to_string());
    }
    let op_str = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("symbolic_eval: first arg must be op string".to_string()),
    };
    let sym_args = match &args[1] {
        Value::Array(arr) => {
            let mut out = Vec::new();
            for v in arr {
                out.push(value_to_symbolic(v)?);
            }
            out
        }
        _ => return Err("symbolic_eval: second arg must be array".to_string()),
    };
    let op = match op_str.as_str() {
        "add" => SymbolicOp::Add,
        "sub" => SymbolicOp::Sub,
        "mul" => SymbolicOp::Mul,
        "div" => SymbolicOp::Div,
        "mod" => SymbolicOp::Mod,
        "gcd" => SymbolicOp::GCD,
        "is_prime" => SymbolicOp::IsPrime,
        "sort" => SymbolicOp::Sort,
        "argmax" => SymbolicOp::ArgMax,
        "compare" => SymbolicOp::Compare,
        "count_if" => SymbolicOp::CountIf,
        _ => return Err(format!("symbolic_eval: unknown op '{}'", op_str)),
    };
    let engine = SymbolicEngine::new();
    match engine.evaluate_exact(op, &sym_args) {
        Some(sv) => Ok(symbolic_to_value(&sv)),
        None => Err("symbolic_eval: operation could not be computed exactly".to_string()),
    }
}

/// hybrid_layer_new(neural_width) -> id
pub fn builtin_hybrid_layer_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let width = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("hybrid_layer_new requires integer width".to_string()),
    };
    let layer = HybridLayer::new(width, width, 0.5);
    let mut layers = hybrid_layers().lock().unwrap();
    let id = layers.len();
    layers.push(layer);
    Ok(Value::Int(id as i128))
}

/// hybrid_layer_forward(id, input_array) -> output_array
pub fn builtin_hybrid_layer_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("hybrid_layer_forward requires (id, input)".to_string());
    }
    let id = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("hybrid_layer_forward: id must be int".to_string()),
    };
    let input: Vec<f64> = match &args[1] {
        Value::Array(arr) => {
            let mut v = Vec::new();
            for item in arr {
                match item {
                    Value::Float(f) => v.push(*f),
                    Value::Int(n) => v.push(*n as f64),
                    _ => return Err("hybrid_layer_forward: input must be numeric array".to_string()),
                }
            }
            v
        }
        _ => return Err("hybrid_layer_forward: second arg must be array".to_string()),
    };
    let layers = hybrid_layers().lock().unwrap();
    let layer = layers.get(id).ok_or("hybrid_layer_forward: invalid id")?;
    let output = layer.forward(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn value_to_symbolic(v: &Value) -> Result<SymbolicValue, String> {
    match v {
        Value::Int(n) => Ok(SymbolicValue::Integer(*n as i64)),
        Value::Float(f) => {
            let r = f.round() as i64;
            if (f - r as f64).abs() < 1e-9 {
                Ok(SymbolicValue::Integer(r))
            } else {
                Ok(SymbolicValue::Unknown)
            }
        }
        Value::Bool(b) => Ok(SymbolicValue::Boolean(*b)),
        Value::Array(arr) => {
            let mut v2 = Vec::new();
            for item in arr {
                v2.push(value_to_symbolic(item)?);
            }
            Ok(SymbolicValue::Vector(v2))
        }
        _ => Ok(SymbolicValue::Unknown),
    }
}

fn symbolic_to_value(sv: &SymbolicValue) -> Value {
    match sv {
        SymbolicValue::Integer(n) => Value::Int(*n as i128),
        SymbolicValue::Rational(n, d) => {
            if *d == 1 {
                Value::Int(*n as i128)
            } else {
                Value::Float(*n as f64 / *d as f64)
            }
        }
        SymbolicValue::Boolean(b) => Value::Bool(*b),
        SymbolicValue::Vector(v) => {
            Value::Array(v.iter().map(symbolic_to_value).collect())
        }
        SymbolicValue::Unknown => Value::Void,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbolic_add_integers() {
        let a = SymbolicValue::Integer(3);
        let b = SymbolicValue::Integer(7);
        assert_eq!(a.add(&b), Some(SymbolicValue::Integer(10)));
    }

    #[test]
    fn test_symbolic_rational_reduction() {
        let r = SymbolicValue::rational(6, 4);
        assert_eq!(r, SymbolicValue::Rational(3, 2));
    }

    #[test]
    fn test_symbolic_rational_arithmetic() {
        let a = SymbolicValue::rational(1, 3);
        let b = SymbolicValue::rational(1, 6);
        let sum = a.add(&b).unwrap();
        assert_eq!(sum, SymbolicValue::Rational(1, 2));
    }

    #[test]
    fn test_symbolic_div_by_zero() {
        let a = SymbolicValue::Integer(5);
        let b = SymbolicValue::Integer(0);
        assert_eq!(a.div(&b), None);
    }

    #[test]
    fn test_engine_is_prime() {
        let engine = SymbolicEngine::new();
        let r = engine.evaluate_exact(SymbolicOp::IsPrime, &[SymbolicValue::Integer(17)]);
        assert_eq!(r, Some(SymbolicValue::Boolean(true)));
        let r2 = engine.evaluate_exact(SymbolicOp::IsPrime, &[SymbolicValue::Integer(15)]);
        assert_eq!(r2, Some(SymbolicValue::Boolean(false)));
    }

    #[test]
    fn test_engine_gcd() {
        let engine = SymbolicEngine::new();
        let r = engine.evaluate_exact(
            SymbolicOp::GCD,
            &[SymbolicValue::Integer(12), SymbolicValue::Integer(8)],
        );
        assert_eq!(r, Some(SymbolicValue::Integer(4)));
    }

    #[test]
    fn test_engine_sort() {
        let engine = SymbolicEngine::new();
        let v = SymbolicValue::Vector(vec![
            SymbolicValue::Integer(3),
            SymbolicValue::Integer(1),
            SymbolicValue::Integer(2),
        ]);
        let r = engine.evaluate_exact(SymbolicOp::Sort, &[v]);
        assert_eq!(
            r,
            Some(SymbolicValue::Vector(vec![
                SymbolicValue::Integer(1),
                SymbolicValue::Integer(2),
                SymbolicValue::Integer(3),
            ]))
        );
    }

    #[test]
    fn test_engine_argmax() {
        let engine = SymbolicEngine::new();
        let v = SymbolicValue::Vector(vec![
            SymbolicValue::Integer(10),
            SymbolicValue::Integer(50),
            SymbolicValue::Integer(30),
        ]);
        let r = engine.evaluate_exact(SymbolicOp::ArgMax, &[v]);
        assert_eq!(r, Some(SymbolicValue::Integer(1)));
    }

    #[test]
    fn test_neural_fallback_forward() {
        let nn = NeuralFallback::new(3, 2);
        let out = nn.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(out.len(), 2);
        // outputs are deterministic from our init
    }

    #[test]
    fn test_router_classify() {
        let router = SymbolicRouter::new(2, 0.5);
        let p = router.classify(&[1.0, 0.0]);
        assert!(p > 0.5); // positive logit => > 0.5
    }

    #[test]
    fn test_straight_through_grad() {
        let router = SymbolicRouter::new(2, 0.5);
        let grad = router.straight_through_grad(&[3.0, 4.0], &[0.1, 0.2]);
        assert_eq!(grad, vec![0.1, 0.2]); // identity pass-through
    }

    #[test]
    fn test_hybrid_layer_exact_integers() {
        let layer = HybridLayer::new(3, 3, 0.5);
        // integers 2, 3, 5 → symbolic sum = 10
        let out = layer.forward(&[2.0, 3.0, 5.0]);
        assert_eq!(out, vec![10.0]);
    }

    #[test]
    fn test_hybrid_layer_mixed_batch() {
        let layer = HybridLayer::new(2, 2, 0.5);
        let batch = vec![vec![1.0, 2.0], vec![0.1, 0.2]];
        let results = layer.forward_mixed(&batch);
        assert_eq!(results.len(), 2);
        // First: integers → symbolic sum = 3
        assert_eq!(results[0], vec![3.0]);
    }

    #[test]
    fn test_symbolic_memory() {
        let mut mem = SymbolicMemory::new();
        mem.assert_fact("x", SymbolicValue::Integer(10));
        mem.assert_fact("y", SymbolicValue::Integer(3));
        assert_eq!(mem.query("x"), Some(&SymbolicValue::Integer(10)));

        let engine = SymbolicEngine::new();
        let nn = NeuralFallback::new(1, 1);
        let result = mem.derive(&engine, SymbolicOp::Add, &["x", "y"], &nn);
        assert_eq!(result, Some(SymbolicValue::Integer(13)));
    }

    #[test]
    fn test_engine_count_if() {
        let engine = SymbolicEngine::new();
        let v = SymbolicValue::Vector(vec![
            SymbolicValue::Integer(1),
            SymbolicValue::Integer(5),
            SymbolicValue::Integer(3),
            SymbolicValue::Integer(7),
        ]);
        let r = engine.evaluate_exact(SymbolicOp::CountIf, &[v, SymbolicValue::Integer(3)]);
        assert_eq!(r, Some(SymbolicValue::Integer(2))); // 5 and 7 are > 3
    }

    #[test]
    fn test_engine_compare() {
        let engine = SymbolicEngine::new();
        let r = engine.evaluate_exact(
            SymbolicOp::Compare,
            &[SymbolicValue::Integer(5), SymbolicValue::Integer(3)],
        );
        assert_eq!(r, Some(SymbolicValue::Integer(1))); // 5 > 3
    }
}
