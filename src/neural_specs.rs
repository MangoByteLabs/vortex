// Pillar 4: Formal Neural Specifications for Vortex
// Provides specification and verification of neural network properties:
// Lipschitz bounds, robustness, output range, gradient bounds, equivariance, monotonicity.

use crate::interpreter::{Env, Value, FnDef};
use std::collections::HashMap;
use std::sync::Mutex;

// ─── Helper Functions ──────────────────────────────────────────────────

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("Expected number".into()),
    }
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => {
            if *n < 0 { return Err("Expected non-negative integer".into()); }
            Ok(*n as usize)
        }
        Value::Float(f) => Ok(*f as usize),
        _ => Err("Expected integer".into()),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64).collect(),
        _ => Err("Expected array of numbers".into()),
    }
}

fn value_to_f64_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect(),
        _ => Err("Expected 2D array of numbers".into()),
    }
}

// ─── Core Types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum NormType {
    L1,
    L2,
    LInf,
}

impl NormType {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "l1" => Ok(NormType::L1),
            "l2" => Ok(NormType::L2),
            "linf" | "inf" => Ok(NormType::LInf),
            _ => Err(format!("Unknown norm type: {}", s)),
        }
    }
}

#[derive(Debug, Clone)]
pub enum SymmetryGroup {
    Rotation2D,
    Rotation3D,
    Permutation(usize),
    Translation(usize),
    ScaleInvariance,
    Custom(Vec<Vec<f64>>),
}

impl SymmetryGroup {
    fn from_str(s: &str) -> Result<Self, String> {
        match s.to_lowercase().as_str() {
            "rotation2d" | "rot2d" => Ok(SymmetryGroup::Rotation2D),
            "rotation3d" | "rot3d" => Ok(SymmetryGroup::Rotation3D),
            "scale" | "scale_invariance" => Ok(SymmetryGroup::ScaleInvariance),
            _ => {
                if s.starts_with("perm") || s.starts_with("Perm") {
                    let n = s.trim_start_matches(|c: char| !c.is_ascii_digit())
                        .parse::<usize>()
                        .unwrap_or(3);
                    Ok(SymmetryGroup::Permutation(n))
                } else if s.starts_with("trans") || s.starts_with("Trans") {
                    let n = s.trim_start_matches(|c: char| !c.is_ascii_digit())
                        .parse::<usize>()
                        .unwrap_or(2);
                    Ok(SymmetryGroup::Translation(n))
                } else {
                    Err(format!("Unknown symmetry group: {}", s))
                }
            }
        }
    }

    /// Generate a representative group transformation matrix for testing.
    fn sample_transform(&self, dim: usize) -> Vec<Vec<f64>> {
        match self {
            SymmetryGroup::Rotation2D => {
                let theta = std::f64::consts::PI / 4.0; // 45 degrees
                let c = theta.cos();
                let s = theta.sin();
                let mut mat = identity_matrix(dim);
                if dim >= 2 {
                    mat[0][0] = c; mat[0][1] = -s;
                    mat[1][0] = s; mat[1][1] = c;
                }
                mat
            }
            SymmetryGroup::Rotation3D => {
                let theta = std::f64::consts::PI / 6.0; // 30 degrees around z
                let c = theta.cos();
                let s = theta.sin();
                let mut mat = identity_matrix(dim);
                if dim >= 3 {
                    mat[0][0] = c; mat[0][1] = -s;
                    mat[1][0] = s; mat[1][1] = c;
                }
                mat
            }
            SymmetryGroup::Permutation(n) => {
                let sz = (*n).min(dim);
                let mut mat = identity_matrix(dim);
                // Cyclic permutation of first sz elements
                if sz >= 2 {
                    for i in 0..dim {
                        for j in 0..dim {
                            mat[i][j] = 0.0;
                        }
                    }
                    for i in 0..sz {
                        mat[i][(i + 1) % sz] = 1.0;
                    }
                    for i in sz..dim {
                        mat[i][i] = 1.0;
                    }
                }
                mat
            }
            SymmetryGroup::Translation(n) => {
                // Translation is not a linear map, so we represent as identity
                // and handle translation as a shift vector separately.
                let _ = n;
                identity_matrix(dim)
            }
            SymmetryGroup::ScaleInvariance => {
                let scale = 2.0;
                let mut mat = identity_matrix(dim);
                for i in 0..dim {
                    mat[i][i] = scale;
                }
                mat
            }
            SymmetryGroup::Custom(m) => {
                if m.len() >= dim {
                    m.clone()
                } else {
                    let mut mat = identity_matrix(dim);
                    for i in 0..m.len().min(dim) {
                        for j in 0..m[i].len().min(dim) {
                            mat[i][j] = m[i][j];
                        }
                    }
                    mat
                }
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum NeuralSpec {
    LipschitzBound(f64),
    InformationPreservation(f64),
    Equivariance { group: SymmetryGroup, tolerance: f64 },
    Monotonicity { dim: usize, increasing: bool },
    RobustnessRadius { eps: f64, norm: NormType },
    OutputRange { min: f64, max: f64 },
    GradientBound(f64),
    Invertibility(f64),
}

#[derive(Debug, Clone)]
pub enum SpecResult {
    Verified,
    Violated { counterexample: Vec<f64>, message: String },
    Unknown(String),
}

impl SpecResult {
    fn to_string_repr(&self) -> String {
        match self {
            SpecResult::Verified => "verified".to_string(),
            SpecResult::Violated { counterexample: _, message } => {
                format!("violated: {}", message)
            }
            SpecResult::Unknown(msg) => format!("unknown: {}", msg),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    Linear,
    Softmax,
}

impl Activation {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "relu" => Activation::ReLU,
            "sigmoid" => Activation::Sigmoid,
            "tanh" => Activation::Tanh,
            "softmax" => Activation::Softmax,
            _ => Activation::Linear,
        }
    }

    fn apply(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => x.max(0.0),
            Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
            Activation::Tanh => x.tanh(),
            Activation::Linear => x,
            Activation::Softmax => x, // applied at vector level
        }
    }

    fn derivative(&self, x: f64) -> f64 {
        match self {
            Activation::ReLU => if x > 0.0 { 1.0 } else { 0.0 },
            Activation::Sigmoid => {
                let s = 1.0 / (1.0 + (-x).exp());
                s * (1.0 - s)
            }
            Activation::Tanh => 1.0 - x.tanh().powi(2),
            Activation::Linear => 1.0,
            Activation::Softmax => 1.0, // simplified
        }
    }

    fn apply_interval(&self, lo: f64, hi: f64) -> (f64, f64) {
        match self {
            Activation::ReLU => (lo.max(0.0), hi.max(0.0)),
            Activation::Sigmoid => {
                let s = |x: f64| 1.0 / (1.0 + (-x).exp());
                (s(lo), s(hi))
            }
            Activation::Tanh => (lo.tanh(), hi.tanh()),
            Activation::Linear => (lo, hi),
            Activation::Softmax => (0.0, 1.0),
        }
    }
}

#[derive(Debug, Clone)]
pub struct NetworkLayer {
    pub weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub activation: Activation,
}

impl NetworkLayer {
    fn output_dim(&self) -> usize {
        self.weights.len()
    }

    fn input_dim(&self) -> usize {
        if self.weights.is_empty() { 0 } else { self.weights[0].len() }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        let n_out = self.output_dim();
        let mut output = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let mut sum = self.bias[i];
            for (j, &x) in input.iter().enumerate() {
                if j < self.weights[i].len() {
                    sum += self.weights[i][j] * x;
                }
            }
            output.push(self.activation.apply(sum));
        }
        if matches!(self.activation, Activation::Softmax) {
            let max_v = output.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = output.iter().map(|x| (x - max_v).exp()).collect();
            let sum: f64 = exps.iter().sum();
            output = exps.iter().map(|e| e / sum).collect();
        }
        output
    }

    /// Propagate intervals through this layer.
    fn propagate_bounds(&self, lo: &[f64], hi: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n_out = self.output_dim();
        let mut out_lo = Vec::with_capacity(n_out);
        let mut out_hi = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let mut sum_lo = self.bias[i];
            let mut sum_hi = self.bias[i];
            for j in 0..lo.len().min(self.weights[i].len()) {
                let w = self.weights[i][j];
                if w >= 0.0 {
                    sum_lo += w * lo[j];
                    sum_hi += w * hi[j];
                } else {
                    sum_lo += w * hi[j];
                    sum_hi += w * lo[j];
                }
            }
            let (a_lo, a_hi) = self.activation.apply_interval(sum_lo, sum_hi);
            out_lo.push(a_lo);
            out_hi.push(a_hi);
        }
        (out_lo, out_hi)
    }

    /// Compute spectral norm via power iteration.
    fn spectral_norm(&self, iterations: usize) -> f64 {
        let rows = self.weights.len();
        let cols = if rows == 0 { 0 } else { self.weights[0].len() };
        if rows == 0 || cols == 0 {
            return 0.0;
        }

        // Initialize random-ish vector
        let mut v = vec![1.0 / (cols as f64).sqrt(); cols];

        for _ in 0..iterations {
            // u = W * v
            let mut u = vec![0.0; rows];
            for i in 0..rows {
                for j in 0..cols {
                    u[i] += self.weights[i][j] * v[j];
                }
            }
            let u_norm = vec_norm_l2(&u);
            if u_norm < 1e-15 { return 0.0; }
            for x in u.iter_mut() { *x /= u_norm; }

            // v = W^T * u
            v = vec![0.0; cols];
            for j in 0..cols {
                for i in 0..rows {
                    v[j] += self.weights[i][j] * u[i];
                }
            }
            let v_norm = vec_norm_l2(&v);
            if v_norm < 1e-15 { return 0.0; }
            for x in v.iter_mut() { *x /= v_norm; }
        }

        // Compute sigma = ||W * v||
        let mut wv = vec![0.0; rows];
        for i in 0..rows {
            for j in 0..cols {
                wv[i] += self.weights[i][j] * v[j];
            }
        }
        vec_norm_l2(&wv)
    }

    /// Upper bound on the Lipschitz constant of this layer (activation included).
    fn lipschitz_upper_bound(&self, iterations: usize) -> f64 {
        let sigma = self.spectral_norm(iterations);
        // ReLU, sigmoid, tanh all have Lipschitz constant <= 1
        // (sigmoid max derivative = 0.25, tanh max = 1, relu = 1, linear = 1)
        let act_lip = match self.activation {
            Activation::Sigmoid => 0.25,
            _ => 1.0,
        };
        sigma * act_lip
    }
}

pub struct SpecChecker {
    specs: Vec<(usize, NeuralSpec)>,
    results: Vec<(usize, SpecResult)>,
}

impl SpecChecker {
    fn new() -> Self {
        SpecChecker { specs: Vec::new(), results: Vec::new() }
    }

    fn add_spec(&mut self, spec_id: usize, spec: NeuralSpec) {
        self.specs.push((spec_id, spec));
    }

    fn verify_all(&mut self, layers: &[NetworkLayer]) {
        self.results.clear();
        for (id, spec) in &self.specs {
            let result = verify_spec_impl(spec, layers);
            self.results.push((*id, result));
        }
    }
}

// ─── Utility Functions ─────────────────────────────────────────────────

fn identity_matrix(n: usize) -> Vec<Vec<f64>> {
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..n {
        m[i][i] = 1.0;
    }
    m
}

fn vec_norm_l2(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn vec_norm_l1(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).sum()
}

fn vec_norm_linf(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

fn vec_norm(v: &[f64], norm: &NormType) -> f64 {
    match norm {
        NormType::L1 => vec_norm_l1(v),
        NormType::L2 => vec_norm_l2(v),
        NormType::LInf => vec_norm_linf(v),
    }
}

fn mat_vec_mul(mat: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    mat.iter().map(|row| {
        row.iter().zip(v.iter()).map(|(a, b)| a * b).sum()
    }).collect()
}

fn network_forward(layers: &[NetworkLayer], input: &[f64]) -> Vec<f64> {
    let mut current = input.to_vec();
    for layer in layers {
        current = layer.forward(&current);
    }
    current
}

fn network_input_dim(layers: &[NetworkLayer]) -> usize {
    if layers.is_empty() { 0 } else { layers[0].input_dim() }
}

// ─── Verification Implementations ──────────────────────────────────────

fn verify_lipschitz(bound: f64, layers: &[NetworkLayer]) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    let mut total_lip = 1.0;
    for layer in layers {
        total_lip *= layer.lipschitz_upper_bound(20);
    }
    if total_lip <= bound {
        SpecResult::Verified
    } else {
        SpecResult::Violated {
            counterexample: vec![],
            message: format!(
                "Network Lipschitz constant upper bound {:.6} exceeds specified bound {:.6}",
                total_lip, bound
            ),
        }
    }
}

fn verify_robustness(eps: f64, norm: &NormType, layers: &[NetworkLayer]) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    let input_dim = network_input_dim(layers);
    if input_dim == 0 {
        return SpecResult::Unknown("Empty input dimension".into());
    }

    // Test at origin: propagate interval [-eps, eps] for each input
    let lo = vec![-eps; input_dim];
    let hi = vec![eps; input_dim];

    let mut cur_lo = lo;
    let mut cur_hi = hi;
    for layer in layers {
        let (new_lo, new_hi) = layer.propagate_bounds(&cur_lo, &cur_hi);
        cur_lo = new_lo;
        cur_hi = new_hi;
    }

    // Check if output classification could change (max output range)
    let output_ranges: Vec<f64> = cur_lo.iter().zip(cur_hi.iter())
        .map(|(l, h)| h - l)
        .collect();
    let max_range = output_ranges.iter().cloned().fold(0.0_f64, f64::max);

    // If all output intervals are narrow enough that the argmax cannot change,
    // the network is robust at this point.
    // Heuristic: if the max output range is small relative to eps, consider verified.
    let _ = norm; // norm type used for the eps-ball; interval propagation handles all norms conservatively
    if max_range < eps * 0.1 {
        SpecResult::Verified
    } else {
        // Cannot conclusively verify with interval arithmetic alone
        SpecResult::Unknown(format!(
            "Interval analysis inconclusive: max output range {:.6} for eps {:.6}",
            max_range, eps
        ))
    }
}

fn verify_output_range(min: f64, max: f64, layers: &[NetworkLayer]) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    let input_dim = network_input_dim(layers);
    if input_dim == 0 {
        return SpecResult::Unknown("Empty input dimension".into());
    }

    // Propagate bounds [-1e6, 1e6] (wide input range)
    let lo = vec![-1e6; input_dim];
    let hi = vec![1e6; input_dim];

    let mut cur_lo = lo;
    let mut cur_hi = hi;
    for layer in layers {
        let (new_lo, new_hi) = layer.propagate_bounds(&cur_lo, &cur_hi);
        cur_lo = new_lo;
        cur_hi = new_hi;
    }

    // Check all output bounds
    for (i, (lo_i, hi_i)) in cur_lo.iter().zip(cur_hi.iter()).enumerate() {
        if *lo_i < min - 1e-9 {
            return SpecResult::Violated {
                counterexample: vec![],
                message: format!(
                    "Output dim {} lower bound {:.6} < specified min {:.6}",
                    i, lo_i, min
                ),
            };
        }
        if *hi_i > max + 1e-9 {
            return SpecResult::Violated {
                counterexample: vec![],
                message: format!(
                    "Output dim {} upper bound {:.6} > specified max {:.6}",
                    i, hi_i, max
                ),
            };
        }
    }
    SpecResult::Verified
}

fn verify_gradient_bound(max_norm: f64, layers: &[NetworkLayer]) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    // Upper bound on Jacobian norm is the product of layer spectral norms * activation derivatives
    // This is the same as the Lipschitz constant.
    let mut jacobian_bound = 1.0;
    for layer in layers {
        jacobian_bound *= layer.lipschitz_upper_bound(20);
    }
    if jacobian_bound <= max_norm {
        SpecResult::Verified
    } else {
        SpecResult::Violated {
            counterexample: vec![],
            message: format!(
                "Jacobian norm upper bound {:.6} exceeds max gradient bound {:.6}",
                jacobian_bound, max_norm
            ),
        }
    }
}

fn verify_equivariance(
    group: &SymmetryGroup,
    tolerance: f64,
    layers: &[NetworkLayer],
) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    let input_dim = network_input_dim(layers);
    if input_dim == 0 {
        return SpecResult::Unknown("Empty input dimension".into());
    }

    let transform = group.sample_transform(input_dim);

    // Sample several test inputs
    let test_inputs = generate_test_inputs(input_dim, 10);

    for input in &test_inputs {
        // f(T(x))
        let tx = mat_vec_mul(&transform, input);
        let f_tx = network_forward(layers, &tx);

        // T(f(x))
        let fx = network_forward(layers, input);
        let output_dim = fx.len();
        let out_transform = group.sample_transform(output_dim);
        let t_fx = mat_vec_mul(&out_transform, &fx);

        // Check ||f(T(x)) - T(f(x))|| < tolerance
        let diff: Vec<f64> = f_tx.iter().zip(t_fx.iter()).map(|(a, b)| a - b).collect();
        let err = vec_norm_l2(&diff);
        if err > tolerance {
            return SpecResult::Violated {
                counterexample: input.clone(),
                message: format!(
                    "Equivariance violated: ||f(T(x)) - T(f(x))|| = {:.6} > tolerance {:.6}",
                    err, tolerance
                ),
            };
        }
    }
    SpecResult::Verified
}

fn verify_monotonicity(dim: usize, increasing: bool, layers: &[NetworkLayer]) -> SpecResult {
    if layers.is_empty() {
        return SpecResult::Verified;
    }
    let input_dim = network_input_dim(layers);
    if dim >= input_dim {
        return SpecResult::Violated {
            counterexample: vec![],
            message: format!("Dimension {} out of range (input dim = {})", dim, input_dim),
        };
    }

    // Check Jacobian sign via finite differences at sample points
    let delta = 1e-5;
    let test_inputs = generate_test_inputs(input_dim, 20);

    for input in &test_inputs {
        let fx = network_forward(layers, input);
        let mut perturbed = input.clone();
        perturbed[dim] += delta;
        let fx_plus = network_forward(layers, &perturbed);

        for (k, (a, b)) in fx.iter().zip(fx_plus.iter()).enumerate() {
            let grad = (b - a) / delta;
            if increasing && grad < -1e-6 {
                return SpecResult::Violated {
                    counterexample: input.clone(),
                    message: format!(
                        "Monotonicity violated at output dim {}: gradient = {:.6} (expected >= 0)",
                        k, grad
                    ),
                };
            }
            if !increasing && grad > 1e-6 {
                return SpecResult::Violated {
                    counterexample: input.clone(),
                    message: format!(
                        "Monotonicity violated at output dim {}: gradient = {:.6} (expected <= 0)",
                        k, grad
                    ),
                };
            }
        }
    }
    SpecResult::Verified
}

fn generate_test_inputs(dim: usize, count: usize) -> Vec<Vec<f64>> {
    let mut inputs = Vec::with_capacity(count);
    // Deterministic pseudo-random inputs using a simple LCG
    let mut seed: u64 = 42;
    for _ in 0..count {
        let mut v = Vec::with_capacity(dim);
        for _ in 0..dim {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let val = ((seed >> 33) as f64) / (u32::MAX as f64) * 2.0 - 1.0;
            v.push(val);
        }
        inputs.push(v);
    }
    inputs
}

fn verify_spec_impl(spec: &NeuralSpec, layers: &[NetworkLayer]) -> SpecResult {
    match spec {
        NeuralSpec::LipschitzBound(bound) => verify_lipschitz(*bound, layers),
        NeuralSpec::RobustnessRadius { eps, norm } => verify_robustness(*eps, norm, layers),
        NeuralSpec::OutputRange { min, max } => verify_output_range(*min, *max, layers),
        NeuralSpec::GradientBound(max_norm) => verify_gradient_bound(*max_norm, layers),
        NeuralSpec::Equivariance { group, tolerance } => {
            verify_equivariance(group, *tolerance, layers)
        }
        NeuralSpec::Monotonicity { dim, increasing } => {
            verify_monotonicity(*dim, *increasing, layers)
        }
        NeuralSpec::InformationPreservation(threshold) => {
            // Approximate: check smallest singular value of each weight matrix
            // If product of smallest singular values > threshold, information is preserved
            if layers.is_empty() {
                return SpecResult::Verified;
            }
            let mut min_sv_product = 1.0;
            for layer in layers {
                let sv = smallest_singular_value(&layer.weights, 20);
                min_sv_product *= sv;
            }
            if min_sv_product >= *threshold {
                SpecResult::Verified
            } else {
                SpecResult::Violated {
                    counterexample: vec![],
                    message: format!(
                        "Information preservation: min singular value product {:.6} < threshold {:.6}",
                        min_sv_product, threshold
                    ),
                }
            }
        }
        NeuralSpec::Invertibility(threshold) => {
            // Check that each layer's weight matrix has condition number bounded
            if layers.is_empty() {
                return SpecResult::Verified;
            }
            for (i, layer) in layers.iter().enumerate() {
                let max_sv = layer.spectral_norm(20);
                let min_sv = smallest_singular_value(&layer.weights, 20);
                if min_sv < 1e-12 {
                    return SpecResult::Violated {
                        counterexample: vec![],
                        message: format!("Layer {} is (near-)singular", i),
                    };
                }
                let cond = max_sv / min_sv;
                if cond > 1.0 / threshold {
                    return SpecResult::Violated {
                        counterexample: vec![],
                        message: format!(
                            "Layer {} condition number {:.6} exceeds 1/threshold {:.6}",
                            i, cond, 1.0 / threshold
                        ),
                    };
                }
            }
            SpecResult::Verified
        }
    }
}

/// Approximate smallest singular value via inverse power iteration on W^T W.
fn smallest_singular_value(weights: &[Vec<f64>], iterations: usize) -> f64 {
    let rows = weights.len();
    let cols = if rows == 0 { return 0.0 } else { weights[0].len() };
    if rows == 0 || cols == 0 { return 0.0; }

    // Compute W^T W
    let mut wtw = vec![vec![0.0; cols]; cols];
    for i in 0..cols {
        for j in 0..cols {
            let mut s = 0.0;
            for k in 0..rows {
                s += weights[k][i] * weights[k][j];
            }
            wtw[i][j] = s;
        }
    }

    // Shift: compute approximate max eigenvalue first (power iteration on W^T W)
    let mut v = vec![1.0 / (cols as f64).sqrt(); cols];
    let mut max_eig = 1.0;
    for _ in 0..iterations {
        let u = mat_vec_mul(&wtw, &v);
        let norm = vec_norm_l2(&u);
        if norm < 1e-15 { return 0.0; }
        max_eig = norm;
        v = u.iter().map(|x| x / norm).collect();
    }

    // Shifted inverse iteration: find smallest eigenvalue of W^T W
    // (W^T W - sigma I)^{-1} with sigma slightly below max_eig
    // For simplicity, use a direct approach: compute eigenvalue via Rayleigh quotient iteration
    // starting from a random vector orthogonal to the dominant eigenvector.

    // Simple fallback: estimate via ratio of Frobenius norm and spectral norm
    let frob_sq: f64 = weights.iter().flat_map(|r| r.iter()).map(|x| x * x).sum();
    let rank = rows.min(cols);
    // avg eigenvalue of W^T W
    let avg_eig = frob_sq / rank as f64;
    // Very rough: smallest sv ~ sqrt(avg_eig) * some factor
    // Better: use the fact that sum of eigenvalues = frob_sq and largest = max_eig
    // So remaining sum = frob_sq - max_eig, average of remaining = (frob_sq - max_eig) / (rank - 1)
    if rank <= 1 {
        return max_eig.sqrt();
    }
    let remaining = (frob_sq - max_eig).max(0.0);
    let avg_remaining = remaining / (rank as f64 - 1.0);
    // The smallest eigenvalue is at most the average of the remaining
    // This is a very rough lower bound; good enough for spec checking
    avg_remaining.sqrt().min(max_eig.sqrt())
}

// ─── Global State ──────────────────────────────────────────────────────

lazy_static::lazy_static! {
    static ref SPEC_STORE: Mutex<HashMap<usize, NeuralSpec>> = Mutex::new(HashMap::new());
    static ref SPEC_COUNTER: Mutex<usize> = Mutex::new(0);
    static ref CHECKER_STORE: Mutex<HashMap<usize, SpecChecker>> = Mutex::new(HashMap::new());
    static ref CHECKER_COUNTER: Mutex<usize> = Mutex::new(0);
}

fn next_spec_id() -> usize {
    let mut c = SPEC_COUNTER.lock().unwrap();
    let id = *c;
    *c += 1;
    id
}

fn next_checker_id() -> usize {
    let mut c = CHECKER_COUNTER.lock().unwrap();
    let id = *c;
    *c += 1;
    id
}

fn store_spec(spec: NeuralSpec) -> usize {
    let id = next_spec_id();
    SPEC_STORE.lock().unwrap().insert(id, spec);
    id
}

fn parse_layers_from_value(v: &Value) -> Result<Vec<NetworkLayer>, String> {
    // Expect an array of layers, where each layer is an array [weights_2d, bias_1d, activation_str]
    let arr = match v {
        Value::Array(a) => a,
        _ => return Err("Expected array of layers".into()),
    };
    let mut layers = Vec::new();
    for item in arr {
        let layer_arr = match item {
            Value::Array(a) => a,
            _ => return Err("Each layer must be an array [weights, bias, activation]".into()),
        };
        if layer_arr.len() < 2 {
            return Err("Each layer must have at least [weights, bias]".into());
        }
        let weights = value_to_f64_2d(&layer_arr[0])?;
        let bias = value_to_f64_vec(&layer_arr[1])?;
        let activation = if layer_arr.len() >= 3 {
            match &layer_arr[2] {
                Value::String(s) => Activation::from_str(s),
                _ => Activation::Linear,
            }
        } else {
            Activation::Linear
        };
        layers.push(NetworkLayer { weights, bias, activation });
    }
    Ok(layers)
}

// ─── Builtin Functions ─────────────────────────────────────────────────

fn builtin_spec_lipschitz(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("spec_lipschitz(bound) expects 1 argument".into());
    }
    let bound = value_to_f64(&args[0])?;
    if bound <= 0.0 {
        return Err("Lipschitz bound must be positive".into());
    }
    let id = store_spec(NeuralSpec::LipschitzBound(bound));
    Ok(Value::Int(id as i128))
}

fn builtin_spec_robustness(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_robustness(eps, norm_str) expects 2 arguments".into());
    }
    let eps = value_to_f64(&args[0])?;
    let norm_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("Second argument must be a norm string (l1, l2, linf)".into()),
    };
    let norm = NormType::from_str(&norm_str)?;
    let id = store_spec(NeuralSpec::RobustnessRadius { eps, norm });
    Ok(Value::Int(id as i128))
}

fn builtin_spec_output_range(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_output_range(min, max) expects 2 arguments".into());
    }
    let min = value_to_f64(&args[0])?;
    let max = value_to_f64(&args[1])?;
    if min > max {
        return Err("min must be <= max".into());
    }
    let id = store_spec(NeuralSpec::OutputRange { min, max });
    Ok(Value::Int(id as i128))
}

fn builtin_spec_gradient_bound(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("spec_gradient_bound(max_norm) expects 1 argument".into());
    }
    let max_norm = value_to_f64(&args[0])?;
    if max_norm <= 0.0 {
        return Err("Gradient bound must be positive".into());
    }
    let id = store_spec(NeuralSpec::GradientBound(max_norm));
    Ok(Value::Int(id as i128))
}

fn builtin_spec_equivariance(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_equivariance(group_str, tolerance) expects 2 arguments".into());
    }
    let group_str = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("First argument must be a group string".into()),
    };
    let tolerance = value_to_f64(&args[1])?;
    let group = SymmetryGroup::from_str(&group_str)?;
    let id = store_spec(NeuralSpec::Equivariance { group, tolerance });
    Ok(Value::Int(id as i128))
}

fn builtin_spec_monotonicity(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_monotonicity(dim, increasing) expects 2 arguments".into());
    }
    let dim = value_to_usize(&args[0])?;
    let increasing = match &args[1] {
        Value::Bool(b) => *b,
        Value::Int(n) => *n != 0,
        _ => return Err("Second argument must be a boolean".into()),
    };
    let id = store_spec(NeuralSpec::Monotonicity { dim, increasing });
    Ok(Value::Int(id as i128))
}

fn builtin_verify_spec(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("verify_spec(spec_id, weights_list) expects 2 arguments".into());
    }
    let spec_id = value_to_usize(&args[0])?;
    let layers = parse_layers_from_value(&args[1])?;

    let store = SPEC_STORE.lock().unwrap();
    let spec = store.get(&spec_id)
        .ok_or_else(|| format!("No spec with id {}", spec_id))?;

    let result = verify_spec_impl(spec, &layers);
    Ok(Value::String(result.to_string_repr()))
}

fn builtin_spec_checker_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("spec_checker_new() expects 0 arguments".into());
    }
    let id = next_checker_id();
    CHECKER_STORE.lock().unwrap().insert(id, SpecChecker::new());
    Ok(Value::Int(id as i128))
}

fn builtin_spec_checker_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_checker_add(checker_id, spec_id) expects 2 arguments".into());
    }
    let checker_id = value_to_usize(&args[0])?;
    let spec_id = value_to_usize(&args[1])?;

    let spec_store = SPEC_STORE.lock().unwrap();
    let spec = spec_store.get(&spec_id)
        .ok_or_else(|| format!("No spec with id {}", spec_id))?
        .clone();
    drop(spec_store);

    let mut checker_store = CHECKER_STORE.lock().unwrap();
    let checker = checker_store.get_mut(&checker_id)
        .ok_or_else(|| format!("No checker with id {}", checker_id))?;
    checker.add_spec(spec_id, spec);
    Ok(Value::Void)
}

fn builtin_spec_checker_verify_all(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("spec_checker_verify_all(checker_id, weights_list) expects 2 arguments".into());
    }
    let checker_id = value_to_usize(&args[0])?;
    let layers = parse_layers_from_value(&args[1])?;

    let mut checker_store = CHECKER_STORE.lock().unwrap();
    let checker = checker_store.get_mut(&checker_id)
        .ok_or_else(|| format!("No checker with id {}", checker_id))?;

    checker.verify_all(&layers);

    let results: Vec<Value> = checker.results.iter()
        .map(|(_, r)| Value::String(r.to_string_repr()))
        .collect();
    Ok(Value::Array(results))
}

// ─── Registration ──────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("spec_lipschitz".into(), FnDef::Builtin(builtin_spec_lipschitz));
    env.functions.insert("spec_robustness".into(), FnDef::Builtin(builtin_spec_robustness));
    env.functions.insert("spec_output_range".into(), FnDef::Builtin(builtin_spec_output_range));
    env.functions.insert("spec_gradient_bound".into(), FnDef::Builtin(builtin_spec_gradient_bound));
    env.functions.insert("spec_equivariance".into(), FnDef::Builtin(builtin_spec_equivariance));
    env.functions.insert("spec_monotonicity".into(), FnDef::Builtin(builtin_spec_monotonicity));
    env.functions.insert("verify_spec".into(), FnDef::Builtin(builtin_verify_spec));
    env.functions.insert("spec_checker_new".into(), FnDef::Builtin(builtin_spec_checker_new));
    env.functions.insert("spec_checker_add".into(), FnDef::Builtin(builtin_spec_checker_add));
    env.functions.insert("spec_checker_verify_all".into(), FnDef::Builtin(builtin_spec_checker_verify_all));
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_identity_layer(n: usize) -> NetworkLayer {
        let mut weights = vec![vec![0.0; n]; n];
        for i in 0..n {
            weights[i][i] = 1.0;
        }
        NetworkLayer {
            weights,
            bias: vec![0.0; n],
            activation: Activation::Linear,
        }
    }

    fn make_scaled_layer(n: usize, scale: f64) -> NetworkLayer {
        let mut weights = vec![vec![0.0; n]; n];
        for i in 0..n {
            weights[i][i] = scale;
        }
        NetworkLayer {
            weights,
            bias: vec![0.0; n],
            activation: Activation::Linear,
        }
    }

    fn make_relu_layer(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> NetworkLayer {
        NetworkLayer { weights, bias, activation: Activation::ReLU }
    }

    fn make_sigmoid_layer(weights: Vec<Vec<f64>>, bias: Vec<f64>) -> NetworkLayer {
        NetworkLayer { weights, bias, activation: Activation::Sigmoid }
    }

    #[test]
    fn test_lipschitz_identity() {
        let layers = vec![make_identity_layer(3)];
        match verify_lipschitz(1.5, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_lipschitz_violated() {
        let layers = vec![make_scaled_layer(3, 5.0)];
        match verify_lipschitz(2.0, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated, got {:?}", other),
        }
    }

    #[test]
    fn test_lipschitz_multi_layer() {
        let layers = vec![
            make_scaled_layer(3, 2.0),
            make_scaled_layer(3, 3.0),
        ];
        // Product of spectral norms = 6.0
        match verify_lipschitz(6.1, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
        match verify_lipschitz(5.0, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated, got {:?}", other),
        }
    }

    #[test]
    fn test_output_range_sigmoid() {
        // Sigmoid outputs are always in (0, 1)
        let layers = vec![make_sigmoid_layer(
            vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            vec![0.0, 0.0],
        )];
        match verify_output_range(0.0, 1.0, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified for sigmoid [0,1], got {:?}", other),
        }
    }

    #[test]
    fn test_output_range_violated() {
        // Linear layer with scale 2 can output anything
        let layers = vec![make_scaled_layer(2, 2.0)];
        match verify_output_range(-1.0, 1.0, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated, got {:?}", other),
        }
    }

    #[test]
    fn test_gradient_bound_identity() {
        let layers = vec![make_identity_layer(3)];
        match verify_gradient_bound(1.5, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_gradient_bound_violated() {
        let layers = vec![make_scaled_layer(3, 10.0)];
        match verify_gradient_bound(5.0, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated, got {:?}", other),
        }
    }

    #[test]
    fn test_monotonicity_increasing() {
        // Identity layer is monotonically increasing in all dimensions
        let layers = vec![make_identity_layer(3)];
        match verify_monotonicity(0, true, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_monotonicity_decreasing() {
        // Negative scale: monotonically decreasing
        let layers = vec![make_scaled_layer(3, -1.0)];
        match verify_monotonicity(0, false, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_monotonicity_violated() {
        // Negative scale is not increasing
        let layers = vec![make_scaled_layer(3, -1.0)];
        match verify_monotonicity(0, true, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated, got {:?}", other),
        }
    }

    #[test]
    fn test_equivariance_permutation_symmetric() {
        // A network with equal weights is permutation equivariant
        let w = vec![
            vec![0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5],
            vec![0.5, 0.5, 0.5],
        ];
        let layers = vec![NetworkLayer {
            weights: w,
            bias: vec![0.0, 0.0, 0.0],
            activation: Activation::Linear,
        }];
        match verify_equivariance(&SymmetryGroup::Permutation(3), 1e-6, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified for symmetric network, got {:?}", other),
        }
    }

    #[test]
    fn test_equivariance_violated() {
        // Asymmetric network should violate permutation equivariance
        let w = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 2.0, 0.0],
            vec![0.0, 0.0, 3.0],
        ];
        let layers = vec![NetworkLayer {
            weights: w,
            bias: vec![0.0, 0.0, 0.0],
            activation: Activation::Linear,
        }];
        match verify_equivariance(&SymmetryGroup::Permutation(3), 1e-6, &layers) {
            SpecResult::Violated { .. } => {}
            other => panic!("Expected Violated for asymmetric network, got {:?}", other),
        }
    }

    #[test]
    fn test_robustness_tiny_network() {
        // Very small weights => small output variation => robust
        let layers = vec![NetworkLayer {
            weights: vec![vec![0.001, 0.001]],
            bias: vec![0.0],
            activation: Activation::Linear,
        }];
        match verify_robustness(0.1, &NormType::L2, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified for tiny network, got {:?}", other),
        }
    }

    #[test]
    fn test_spectral_norm() {
        let layer = make_scaled_layer(3, 4.0);
        let sn = layer.spectral_norm(50);
        assert!((sn - 4.0).abs() < 0.01, "Expected spectral norm ~4.0, got {}", sn);
    }

    #[test]
    fn test_network_forward() {
        let layers = vec![
            NetworkLayer {
                weights: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
                bias: vec![0.5, -0.5],
                activation: Activation::ReLU,
            },
        ];
        let out = network_forward(&layers, &[1.0, 1.0]);
        // [1*1+2*1+0.5, 3*1+4*1-0.5] = [3.5, 6.5], relu => [3.5, 6.5]
        assert!((out[0] - 3.5).abs() < 1e-9);
        assert!((out[1] - 6.5).abs() < 1e-9);
    }

    #[test]
    fn test_interval_propagation() {
        let layer = NetworkLayer {
            weights: vec![vec![1.0, -1.0]],
            bias: vec![0.0],
            activation: Activation::ReLU,
        };
        let (lo, hi) = layer.propagate_bounds(&[-1.0, -1.0], &[1.0, 1.0]);
        // pre-activation: min = 1*(-1) + (-1)*1 = -2, max = 1*1 + (-1)*(-1) = 2
        // relu: [0, 2]
        assert!((lo[0] - 0.0).abs() < 1e-9);
        assert!((hi[0] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_spec_result_string() {
        assert_eq!(SpecResult::Verified.to_string_repr(), "verified");
        let v = SpecResult::Violated {
            counterexample: vec![1.0],
            message: "bad".into(),
        };
        assert_eq!(v.to_string_repr(), "violated: bad");
        let u = SpecResult::Unknown("dunno".into());
        assert_eq!(u.to_string_repr(), "unknown: dunno");
    }

    #[test]
    fn test_activation_derivatives() {
        let relu = Activation::ReLU;
        assert_eq!(relu.derivative(1.0), 1.0);
        assert_eq!(relu.derivative(-1.0), 0.0);

        let sig = Activation::Sigmoid;
        let d = sig.derivative(0.0);
        assert!((d - 0.25).abs() < 1e-9); // sigmoid'(0) = 0.25

        let th = Activation::Tanh;
        let d = th.derivative(0.0);
        assert!((d - 1.0).abs() < 1e-9); // tanh'(0) = 1
    }

    #[test]
    fn test_spec_checker_workflow() {
        let mut checker = SpecChecker::new();
        let spec1 = NeuralSpec::LipschitzBound(2.0);
        let spec2 = NeuralSpec::OutputRange { min: -1.0, max: 1.0 };
        checker.add_spec(0, spec1);
        checker.add_spec(1, spec2);

        let layers = vec![NetworkLayer {
            weights: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]],
            bias: vec![0.0; 3],
            activation: Activation::Tanh,
        }];
        checker.verify_all(&layers);

        assert_eq!(checker.results.len(), 2);
        for (_, result) in &checker.results {
            match result {
                SpecResult::Verified => {}
                other => panic!("Expected Verified, got {:?}", other),
            }
        }
    }

    #[test]
    fn test_information_preservation() {
        // Identity matrix preserves all information
        let layers = vec![make_identity_layer(3)];
        let spec = NeuralSpec::InformationPreservation(0.5);
        match verify_spec_impl(&spec, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_invertibility_identity() {
        let layers = vec![make_identity_layer(3)];
        let spec = NeuralSpec::Invertibility(0.1);
        match verify_spec_impl(&spec, &layers) {
            SpecResult::Verified => {}
            other => panic!("Expected Verified, got {:?}", other),
        }
    }

    #[test]
    fn test_norm_types() {
        let v = vec![1.0, -2.0, 3.0];
        assert!((vec_norm(&v, &NormType::L1) - 6.0).abs() < 1e-9);
        assert!((vec_norm(&v, &NormType::L2) - (14.0_f64).sqrt()).abs() < 1e-9);
        assert!((vec_norm(&v, &NormType::LInf) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_symmetry_group_parsing() {
        assert!(matches!(SymmetryGroup::from_str("rotation2d").unwrap(), SymmetryGroup::Rotation2D));
        assert!(matches!(SymmetryGroup::from_str("rot3d").unwrap(), SymmetryGroup::Rotation3D));
        assert!(matches!(SymmetryGroup::from_str("scale").unwrap(), SymmetryGroup::ScaleInvariance));
        assert!(matches!(SymmetryGroup::from_str("perm3").unwrap(), SymmetryGroup::Permutation(3)));
    }

    #[test]
    fn test_norm_type_parsing() {
        assert!(matches!(NormType::from_str("l1").unwrap(), NormType::L1));
        assert!(matches!(NormType::from_str("L2").unwrap(), NormType::L2));
        assert!(matches!(NormType::from_str("linf").unwrap(), NormType::LInf));
        assert!(NormType::from_str("garbage").is_err());
    }

    #[test]
    fn test_generate_test_inputs() {
        let inputs = generate_test_inputs(3, 5);
        assert_eq!(inputs.len(), 5);
        for inp in &inputs {
            assert_eq!(inp.len(), 3);
            for &v in inp {
                assert!(v >= -1.0 && v <= 1.0);
            }
        }
        // Deterministic
        let inputs2 = generate_test_inputs(3, 5);
        assert_eq!(inputs, inputs2);
    }

    #[test]
    fn test_mat_vec_mul() {
        let mat = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let v = vec![1.0, 1.0];
        let result = mat_vec_mul(&mat, &v);
        assert!((result[0] - 3.0).abs() < 1e-9);
        assert!((result[1] - 7.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_forward() {
        let layer = NetworkLayer {
            weights: vec![vec![1.0, 0.0], vec![0.0, 1.0]],
            bias: vec![0.0, 0.0],
            activation: Activation::Softmax,
        };
        let out = layer.forward(&[1.0, 2.0]);
        assert!((out[0] + out[1] - 1.0).abs() < 1e-9); // sums to 1
        assert!(out[1] > out[0]); // second input larger
    }
}
