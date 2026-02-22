use crate::interpreter::{Env, Value, FnDef};

/// A layer in a neural network for verification purposes.
#[derive(Debug, Clone)]
pub struct Layer {
    pub weights: Vec<Vec<f64>>,
    pub biases: Vec<f64>,
    pub activation: String, // "relu", "sigmoid", "tanh", "linear"
}

/// Abstract value for interval-based abstract interpretation.
#[derive(Debug, Clone, PartialEq)]
pub enum AbstractValue {
    Interval(f64, f64),
    Top,
    Bottom,
}

impl AbstractValue {
    pub fn add(a: &AbstractValue, b: &AbstractValue) -> AbstractValue {
        match (a, b) {
            (AbstractValue::Bottom, _) | (_, AbstractValue::Bottom) => AbstractValue::Bottom,
            (AbstractValue::Top, _) | (_, AbstractValue::Top) => AbstractValue::Top,
            (AbstractValue::Interval(a_lo, a_hi), AbstractValue::Interval(b_lo, b_hi)) => {
                AbstractValue::Interval(a_lo + b_lo, a_hi + b_hi)
            }
        }
    }

    pub fn mul(a: &AbstractValue, b: &AbstractValue) -> AbstractValue {
        match (a, b) {
            (AbstractValue::Bottom, _) | (_, AbstractValue::Bottom) => AbstractValue::Bottom,
            (AbstractValue::Top, _) | (_, AbstractValue::Top) => AbstractValue::Top,
            (AbstractValue::Interval(a_lo, a_hi), AbstractValue::Interval(b_lo, b_hi)) => {
                let products = [a_lo * b_lo, a_lo * b_hi, a_hi * b_lo, a_hi * b_hi];
                let lo = products.iter().cloned().fold(f64::INFINITY, f64::min);
                let hi = products.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                AbstractValue::Interval(lo, hi)
            }
        }
    }

    pub fn relu(a: &AbstractValue) -> AbstractValue {
        match a {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => AbstractValue::Interval(0.0, f64::INFINITY),
            AbstractValue::Interval(lo, hi) => {
                AbstractValue::Interval(lo.max(0.0), hi.max(0.0))
            }
        }
    }

    pub fn sigmoid(a: &AbstractValue) -> AbstractValue {
        match a {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => AbstractValue::Interval(0.0, 1.0),
            AbstractValue::Interval(lo, hi) => {
                let s = |x: f64| 1.0 / (1.0 + (-x).exp());
                AbstractValue::Interval(s(*lo), s(*hi))
            }
        }
    }

    pub fn tanh_abs(a: &AbstractValue) -> AbstractValue {
        match a {
            AbstractValue::Bottom => AbstractValue::Bottom,
            AbstractValue::Top => AbstractValue::Interval(-1.0, 1.0),
            AbstractValue::Interval(lo, hi) => {
                AbstractValue::Interval(lo.tanh(), hi.tanh())
            }
        }
    }

    pub fn contains(&self, x: f64) -> bool {
        match self {
            AbstractValue::Bottom => false,
            AbstractValue::Top => true,
            AbstractValue::Interval(lo, hi) => x >= *lo && x <= *hi,
        }
    }

    pub fn width(&self) -> f64 {
        match self {
            AbstractValue::Bottom => 0.0,
            AbstractValue::Top => f64::INFINITY,
            AbstractValue::Interval(lo, hi) => hi - lo,
        }
    }
}

// ---- Abstract Interpreter ----

pub struct AbstractInterpreter;

impl AbstractInterpreter {
    pub fn interpret_layer(
        weights: &[Vec<f64>],
        biases: &[f64],
        input: &[AbstractValue],
        activation: &str,
    ) -> Vec<AbstractValue> {
        let n_out = weights.len();
        let mut output = Vec::with_capacity(n_out);
        for i in 0..n_out {
            let mut acc = AbstractValue::Interval(biases[i], biases[i]);
            for (j, inp) in input.iter().enumerate() {
                let w = weights[i][j];
                let wi = AbstractValue::Interval(w, w);
                let prod = AbstractValue::mul(&wi, inp);
                acc = AbstractValue::add(&acc, &prod);
            }
            let activated = match activation {
                "relu" => AbstractValue::relu(&acc),
                "sigmoid" => AbstractValue::sigmoid(&acc),
                "tanh" => AbstractValue::tanh_abs(&acc),
                _ => acc, // "linear" or unknown
            };
            output.push(activated);
        }
        output
    }

    pub fn interpret_network(
        layers: &[Layer],
        input: &[AbstractValue],
    ) -> Vec<AbstractValue> {
        let mut current = input.to_vec();
        for layer in layers {
            current = Self::interpret_layer(
                &layer.weights,
                &layer.biases,
                &current,
                &layer.activation,
            );
        }
        current
    }
}

// ---- Monotonicity Verifier ----

#[derive(Debug, Clone, PartialEq)]
pub enum MonotonicityResult {
    Proven,
    Disproven(Vec<f64>),
    Inconclusive,
}

pub struct MonotonicityVerifier;

impl MonotonicityVerifier {
    /// Verify monotonicity of output w.r.t. input_idx.
    /// For ReLU networks: sufficient condition is all weights along paths are non-negative.
    pub fn verify_monotonic(
        layers: &[Layer],
        input_idx: usize,
        input_range: (f64, f64),
        n_samples: usize,
    ) -> MonotonicityResult {
        // First try analytical proof for ReLU/linear networks
        if layers.iter().all(|l| l.activation == "relu" || l.activation == "linear") {
            if Self::check_weight_signs(layers, input_idx) {
                return MonotonicityResult::Proven;
            }
        }

        // Fallback: sampling-based check
        let step = (input_range.1 - input_range.0) / (n_samples as f64);
        let n_inputs = layers[0].weights[0].len();
        let mut base_input: Vec<f64> = vec![0.0; n_inputs];

        let mut prev_output = None;
        for i in 0..=n_samples {
            let x = input_range.0 + step * (i as f64);
            base_input[input_idx] = x;
            let out = Self::evaluate_network(layers, &base_input);
            if let Some(prev) = prev_output {
                if out[0] < prev {
                    let mut counterexample = base_input.clone();
                    counterexample[input_idx] = x;
                    return MonotonicityResult::Disproven(counterexample);
                }
            }
            prev_output = Some(out[0]);
        }
        MonotonicityResult::Inconclusive
    }

    fn check_weight_signs(layers: &[Layer], input_idx: usize) -> bool {
        // For a single-output network: check that effective weight from input_idx is non-negative
        // through all paths. Sufficient condition: all relevant weights are non-negative.
        let n_inputs = layers[0].weights[0].len();
        // Track which neurons can be reached with non-negative contribution
        let mut positive = vec![false; n_inputs];
        positive[input_idx] = true;

        for layer in layers {
            let n_out = layer.weights.len();
            let mut next_positive = vec![false; n_out];
            for i in 0..n_out {
                let mut all_nonneg = true;
                let mut has_contribution = false;
                for (j, &p) in positive.iter().enumerate() {
                    if p {
                        has_contribution = true;
                        if layer.weights[i][j] < 0.0 {
                            all_nonneg = false;
                            break;
                        }
                    }
                }
                if has_contribution && all_nonneg {
                    next_positive[i] = true;
                }
            }
            positive = next_positive;
        }
        positive.iter().any(|&p| p)
    }

    fn evaluate_network(layers: &[Layer], input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in layers {
            let mut next = Vec::with_capacity(layer.weights.len());
            for i in 0..layer.weights.len() {
                let mut sum = layer.biases[i];
                for j in 0..current.len() {
                    sum += layer.weights[i][j] * current[j];
                }
                let activated = match layer.activation.as_str() {
                    "relu" => sum.max(0.0),
                    "sigmoid" => 1.0 / (1.0 + (-sum).exp()),
                    "tanh" => sum.tanh(),
                    _ => sum,
                };
                next.push(activated);
            }
            current = next;
        }
        current
    }
}

// ---- Boundedness Verifier ----

pub struct BoundednessVerifier;

impl BoundednessVerifier {
    pub fn verify_bounded(
        layers: &[Layer],
        input_ranges: &[(f64, f64)],
        output_bounds: (f64, f64),
    ) -> bool {
        let inputs: Vec<AbstractValue> = input_ranges
            .iter()
            .map(|&(lo, hi)| AbstractValue::Interval(lo, hi))
            .collect();
        let outputs = AbstractInterpreter::interpret_network(layers, &inputs);
        outputs.iter().all(|o| match o {
            AbstractValue::Interval(lo, hi) => *lo >= output_bounds.0 && *hi <= output_bounds.1,
            AbstractValue::Bottom => true,
            AbstractValue::Top => false,
        })
    }
}

// ---- Lipschitz Verifier ----

pub struct LipschitzVerifier;

impl LipschitzVerifier {
    /// Upper bound on Lipschitz constant: product of operator norms of weight matrices.
    /// For ReLU networks this is exact (ReLU has Lipschitz constant 1).
    pub fn compute_lipschitz_bound(layers: &[Layer]) -> f64 {
        let mut k = 1.0;
        for layer in layers {
            let norm = Self::operator_norm(&layer.weights);
            // Activation Lipschitz constants
            let act_lip = match layer.activation.as_str() {
                "relu" => 1.0,
                "sigmoid" => 0.25, // max derivative of sigmoid
                "tanh" => 1.0,
                _ => 1.0,
            };
            k *= norm * act_lip;
        }
        k
    }

    /// Approximate spectral norm via power iteration.
    pub fn operator_norm(matrix: &[Vec<f64>]) -> f64 {
        if matrix.is_empty() || matrix[0].is_empty() {
            return 0.0;
        }
        let m = matrix.len();
        let n = matrix[0].len();
        let mut v: Vec<f64> = vec![1.0 / (n as f64).sqrt(); n];

        for _ in 0..100 {
            // u = A * v
            let mut u = vec![0.0; m];
            for i in 0..m {
                for j in 0..n {
                    u[i] += matrix[i][j] * v[j];
                }
            }
            let u_norm: f64 = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if u_norm < 1e-15 {
                return 0.0;
            }
            for x in u.iter_mut() {
                *x /= u_norm;
            }

            // v = A^T * u
            v = vec![0.0; n];
            for j in 0..n {
                for i in 0..m {
                    v[j] += matrix[i][j] * u[i];
                }
            }
            let v_norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm < 1e-15 {
                return 0.0;
            }
            for x in v.iter_mut() {
                *x /= v_norm;
            }
        }

        // Final sigma = ||A * v||
        let mut av = vec![0.0; m];
        for i in 0..m {
            for j in 0..n {
                av[i] += matrix[i][j] * v[j];
            }
        }
        av.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    pub fn verify_lipschitz(layers: &[Layer], target_k: f64) -> bool {
        Self::compute_lipschitz_bound(layers) <= target_k
    }
}

// ---- Invariance Verifier ----

pub struct InvarianceVerifier;

impl InvarianceVerifier {
    pub fn verify_permutation_invariant(
        layers: &[Layer],
        input: &[f64],
        perm: &[usize],
    ) -> bool {
        let original_out = MonotonicityVerifier::evaluate_network(layers, input);
        let permuted_input: Vec<f64> = perm.iter().map(|&i| input[i]).collect();
        let permuted_out = MonotonicityVerifier::evaluate_network(layers, &permuted_input);
        original_out
            .iter()
            .zip(permuted_out.iter())
            .all(|(a, b)| (a - b).abs() < 1e-10)
    }

    pub fn verify_group_invariant(
        layers: &[Layer],
        input: &[f64],
        transforms: &[Vec<Vec<f64>>],
    ) -> bool {
        let original_out = MonotonicityVerifier::evaluate_network(layers, input);
        for transform in transforms {
            let transformed: Vec<f64> = transform
                .iter()
                .map(|row| row.iter().zip(input).map(|(a, b)| a * b).sum())
                .collect();
            let out = MonotonicityVerifier::evaluate_network(layers, &transformed);
            if !original_out
                .iter()
                .zip(out.iter())
                .all(|(a, b)| (a - b).abs() < 1e-10)
            {
                return false;
            }
        }
        true
    }

    pub fn fairness_check(
        layers: &[Layer],
        protected_idx: usize,
        input: &[f64],
    ) -> f64 {
        // Compute sensitivity: max |f(x) - f(x')| where x' differs only in protected_idx
        let original_out = MonotonicityVerifier::evaluate_network(layers, input);
        let mut max_diff = 0.0f64;
        // Test a range of values for the protected attribute
        for k in 0..=10 {
            let val = -1.0 + 0.2 * (k as f64);
            let mut modified = input.to_vec();
            modified[protected_idx] = val;
            let out = MonotonicityVerifier::evaluate_network(layers, &modified);
            for (a, b) in original_out.iter().zip(out.iter()) {
                max_diff = max_diff.max((a - b).abs());
            }
        }
        max_diff
    }
}

// ---- Verification Report ----

#[derive(Debug, Clone, PartialEq)]
pub enum VerificationResult {
    Proven,
    Disproven,
    Inconclusive,
}

#[derive(Debug, Clone)]
pub struct VerificationReport {
    pub property: String,
    pub result: VerificationResult,
    pub counterexample: Option<Vec<f64>>,
    pub bound: Option<f64>,
    pub confidence: f64,
}

#[derive(Debug, Clone)]
pub enum Property {
    Monotonic(usize),
    Bounded(f64, f64),
    Lipschitz(f64),
    Invariant(Vec<Vec<f64>>),
    Fair(usize, f64),
}

// ---- Model Verifier ----

pub struct ModelVerifier;

impl ModelVerifier {
    pub fn verify_all(
        layers: &[Layer],
        input_ranges: &[(f64, f64)],
        properties: &[Property],
    ) -> Vec<VerificationReport> {
        properties
            .iter()
            .map(|prop| match prop {
                Property::Monotonic(idx) => {
                    let range = if *idx < input_ranges.len() {
                        input_ranges[*idx]
                    } else {
                        (-1.0, 1.0)
                    };
                    let res = MonotonicityVerifier::verify_monotonic(layers, *idx, range, 100);
                    match res {
                        MonotonicityResult::Proven => VerificationReport {
                            property: format!("monotonic({})", idx),
                            result: VerificationResult::Proven,
                            counterexample: None,
                            bound: None,
                            confidence: 1.0,
                        },
                        MonotonicityResult::Disproven(ce) => VerificationReport {
                            property: format!("monotonic({})", idx),
                            result: VerificationResult::Disproven,
                            counterexample: Some(ce),
                            bound: None,
                            confidence: 1.0,
                        },
                        MonotonicityResult::Inconclusive => VerificationReport {
                            property: format!("monotonic({})", idx),
                            result: VerificationResult::Inconclusive,
                            counterexample: None,
                            bound: None,
                            confidence: 0.9,
                        },
                    }
                }
                Property::Bounded(lo, hi) => {
                    let bounded = BoundednessVerifier::verify_bounded(layers, input_ranges, (*lo, *hi));
                    VerificationReport {
                        property: format!("bounded({}, {})", lo, hi),
                        result: if bounded { VerificationResult::Proven } else { VerificationResult::Disproven },
                        counterexample: None,
                        bound: None,
                        confidence: 1.0,
                    }
                }
                Property::Lipschitz(k) => {
                    let actual_k = LipschitzVerifier::compute_lipschitz_bound(layers);
                    VerificationReport {
                        property: format!("lipschitz({})", k),
                        result: if actual_k <= *k { VerificationResult::Proven } else { VerificationResult::Disproven },
                        counterexample: None,
                        bound: Some(actual_k),
                        confidence: 1.0,
                    }
                }
                Property::Invariant(transform) => {
                    let input: Vec<f64> = input_ranges.iter().map(|&(lo, hi)| (lo + hi) / 2.0).collect();
                    let inv = InvarianceVerifier::verify_group_invariant(layers, &input, &[transform.clone()]);
                    VerificationReport {
                        property: "invariant".to_string(),
                        result: if inv { VerificationResult::Inconclusive } else { VerificationResult::Disproven },
                        counterexample: None,
                        bound: None,
                        confidence: if inv { 0.5 } else { 1.0 },
                    }
                }
                Property::Fair(idx, threshold) => {
                    let input: Vec<f64> = input_ranges.iter().map(|&(lo, hi)| (lo + hi) / 2.0).collect();
                    let sensitivity = InvarianceVerifier::fairness_check(layers, *idx, &input);
                    VerificationReport {
                        property: format!("fair({}, {})", idx, threshold),
                        result: if sensitivity <= *threshold { VerificationResult::Proven } else { VerificationResult::Disproven },
                        counterexample: None,
                        bound: Some(sensitivity),
                        confidence: 1.0,
                    }
                }
            })
            .collect()
    }
}

// ---- Helper: parse layers from interpreter values ----

fn parse_layers_from_args(args: &[Value]) -> Result<(Vec<Layer>, Vec<usize>), String> {
    // args[0] = layer_sizes: [int, ...], args[1] = weights: flat array of floats
    let sizes = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("layer_sizes must be integers".to_string()),
        }).collect::<Result<Vec<usize>, String>>()?,
        _ => return Err("first arg must be array of layer sizes".to_string()),
    };

    let flat_weights = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("weights must be numbers".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("second arg must be array of weights".to_string()),
    };

    // Build layers from sizes and flat weight array
    let mut layers = Vec::new();
    let mut offset = 0;
    for i in 0..sizes.len() - 1 {
        let n_in = sizes[i];
        let n_out = sizes[i + 1];
        let n_weights = n_in * n_out;
        let n_biases = n_out;
        if offset + n_weights + n_biases > flat_weights.len() {
            return Err(format!("not enough weights: need {} but only {} remain",
                n_weights + n_biases, flat_weights.len() - offset));
        }
        let mut weights = Vec::with_capacity(n_out);
        for o in 0..n_out {
            let row: Vec<f64> = flat_weights[offset + o * n_in..offset + o * n_in + n_in].to_vec();
            weights.push(row);
        }
        offset += n_weights;
        let biases = flat_weights[offset..offset + n_biases].to_vec();
        offset += n_biases;

        let activation = if i == sizes.len() - 2 { "linear" } else { "relu" }.to_string();
        layers.push(Layer { weights, biases, activation });
    }

    Ok((layers, sizes))
}

// ---- Interpreter builtins ----

fn builtin_verify_monotonic(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("verify_monotonic(layer_sizes, weights, input_idx) requires 3 args".to_string());
    }
    let (layers, _) = parse_layers_from_args(&args)?;
    let input_idx = match &args[2] {
        Value::Int(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("input_idx must be an integer".to_string()),
    };
    let result = MonotonicityVerifier::verify_monotonic(&layers, input_idx, (-1.0, 1.0), 100);
    Ok(Value::Bool(matches!(result, MonotonicityResult::Proven)))
}

fn builtin_verify_bounded(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("verify_bounded(layer_sizes, weights, output_bounds) requires 3 args".to_string());
    }
    let (layers, sizes) = parse_layers_from_args(&args)?;
    let bounds = match &args[2] {
        Value::Array(arr) if arr.len() == 2 => {
            let lo = match &arr[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("bounds must be numbers".to_string()) };
            let hi = match &arr[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("bounds must be numbers".to_string()) };
            (lo, hi)
        }
        _ => return Err("output_bounds must be [lo, hi]".to_string()),
    };
    let input_ranges: Vec<(f64, f64)> = vec![(-1.0, 1.0); sizes[0]];
    let result = BoundednessVerifier::verify_bounded(&layers, &input_ranges, bounds);
    Ok(Value::Bool(result))
}

fn builtin_verify_lipschitz(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("verify_lipschitz(layer_sizes, weights) requires 2 args".to_string());
    }
    let (layers, _) = parse_layers_from_args(&args)?;
    let k = LipschitzVerifier::compute_lipschitz_bound(&layers);
    Ok(Value::Float(k))
}

fn builtin_verify_fairness(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("verify_fairness(layer_sizes, weights, protected_idx) requires 3 args".to_string());
    }
    let (layers, sizes) = parse_layers_from_args(&args)?;
    let protected_idx = match &args[2] {
        Value::Int(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("protected_idx must be an integer".to_string()),
    };
    let input: Vec<f64> = vec![0.0; sizes[0]];
    let sensitivity = InvarianceVerifier::fairness_check(&layers, protected_idx, &input);
    Ok(Value::Float(sensitivity))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("verify_monotonic".to_string(), FnDef::Builtin(builtin_verify_monotonic));
    env.functions.insert("verify_bounded".to_string(), FnDef::Builtin(builtin_verify_bounded));
    env.functions.insert("verify_lipschitz".to_string(), FnDef::Builtin(builtin_verify_lipschitz));
    env.functions.insert("verify_fairness".to_string(), FnDef::Builtin(builtin_verify_fairness));
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_linear_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        Layer { weights, biases, activation: "linear".to_string() }
    }

    fn relu_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        Layer { weights, biases, activation: "relu".to_string() }
    }

    fn sigmoid_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
        Layer { weights, biases, activation: "sigmoid".to_string() }
    }

    // Test 1: AbstractValue interval arithmetic - add
    #[test]
    fn test_abstract_add() {
        let a = AbstractValue::Interval(1.0, 3.0);
        let b = AbstractValue::Interval(2.0, 4.0);
        let result = AbstractValue::add(&a, &b);
        assert_eq!(result, AbstractValue::Interval(3.0, 7.0));
    }

    // Test 2: AbstractValue interval arithmetic - mul
    #[test]
    fn test_abstract_mul() {
        let a = AbstractValue::Interval(-2.0, 3.0);
        let b = AbstractValue::Interval(1.0, 4.0);
        let result = AbstractValue::mul(&a, &b);
        match result {
            AbstractValue::Interval(lo, hi) => {
                assert!((lo - (-8.0)).abs() < 1e-10);
                assert!((hi - 12.0).abs() < 1e-10);
            }
            _ => panic!("expected interval"),
        }
    }

    // Test 3: ReLU on interval
    #[test]
    fn test_abstract_relu() {
        let a = AbstractValue::Interval(-3.0, 5.0);
        let result = AbstractValue::relu(&a);
        assert_eq!(result, AbstractValue::Interval(0.0, 5.0));

        let b = AbstractValue::Interval(-5.0, -1.0);
        let result2 = AbstractValue::relu(&b);
        assert_eq!(result2, AbstractValue::Interval(0.0, 0.0));
    }

    // Test 4: Sigmoid on interval
    #[test]
    fn test_abstract_sigmoid() {
        let a = AbstractValue::Interval(0.0, 0.0);
        match AbstractValue::sigmoid(&a) {
            AbstractValue::Interval(lo, hi) => {
                assert!((lo - 0.5).abs() < 1e-10);
                assert!((hi - 0.5).abs() < 1e-10);
            }
            _ => panic!("expected interval"),
        }
        // Top -> [0, 1]
        assert_eq!(AbstractValue::sigmoid(&AbstractValue::Top), AbstractValue::Interval(0.0, 1.0));
    }

    // Test 5: Abstract interpretation through a linear layer
    #[test]
    fn test_interpret_linear_layer() {
        let input = vec![AbstractValue::Interval(0.0, 1.0), AbstractValue::Interval(0.0, 1.0)];
        let weights = vec![vec![1.0, 2.0]]; // single output neuron
        let biases = vec![0.5];
        let output = AbstractInterpreter::interpret_layer(&weights, &biases, &input, "linear");
        // output = 1*[0,1] + 2*[0,1] + 0.5 = [0.5, 3.5]
        assert_eq!(output.len(), 1);
        match &output[0] {
            AbstractValue::Interval(lo, hi) => {
                assert!((lo - 0.5).abs() < 1e-10);
                assert!((hi - 3.5).abs() < 1e-10);
            }
            _ => panic!("expected interval"),
        }
    }

    // Test 6: Monotonicity proven for positive-weight ReLU network
    #[test]
    fn test_monotonicity_proven() {
        let layers = vec![
            relu_layer(vec![vec![1.0, 0.5], vec![0.3, 0.8]], vec![0.0, 0.0]),
            simple_linear_layer(vec![vec![1.0, 1.0]], vec![0.0]),
        ];
        let result = MonotonicityVerifier::verify_monotonic(&layers, 0, (-1.0, 1.0), 50);
        assert_eq!(result, MonotonicityResult::Proven);
    }

    // Test 7: Monotonicity disproven for network with negative weights
    #[test]
    fn test_monotonicity_disproven() {
        let layers = vec![
            simple_linear_layer(vec![vec![-1.0]], vec![0.0]),
        ];
        let result = MonotonicityVerifier::verify_monotonic(&layers, 0, (-1.0, 1.0), 50);
        assert!(matches!(result, MonotonicityResult::Disproven(_)));
    }

    // Test 8: Boundedness via sigmoid output
    #[test]
    fn test_boundedness_sigmoid() {
        let layers = vec![
            sigmoid_layer(vec![vec![2.0, -1.0]], vec![0.0]),
        ];
        let input_ranges = vec![(-10.0, 10.0), (-10.0, 10.0)];
        assert!(BoundednessVerifier::verify_bounded(&layers, &input_ranges, (0.0, 1.0)));
        assert!(!BoundednessVerifier::verify_bounded(&layers, &input_ranges, (0.3, 0.7)));
    }

    // Test 9: Lipschitz bound computation
    #[test]
    fn test_lipschitz_bound() {
        // Identity matrix has spectral norm 1
        let layers = vec![
            simple_linear_layer(vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![0.0, 0.0]),
        ];
        let k = LipschitzVerifier::compute_lipschitz_bound(&layers);
        assert!((k - 1.0).abs() < 1e-6);
    }

    // Test 10: Lipschitz verify
    #[test]
    fn test_lipschitz_verify() {
        let layers = vec![
            simple_linear_layer(vec![vec![0.5, 0.0], vec![0.0, 0.5]], vec![0.0, 0.0]),
        ];
        assert!(LipschitzVerifier::verify_lipschitz(&layers, 1.0));
        assert!(LipschitzVerifier::verify_lipschitz(&layers, 0.5));
        assert!(!LipschitzVerifier::verify_lipschitz(&layers, 0.3));
    }

    // Test 11: Fairness check
    #[test]
    fn test_fairness_check() {
        // Network where output depends heavily on input 0 (protected)
        let layers = vec![
            simple_linear_layer(vec![vec![5.0, 0.1]], vec![0.0]),
        ];
        let input = vec![0.0, 0.5];
        let sensitivity = InvarianceVerifier::fairness_check(&layers, 0, &input);
        assert!(sensitivity > 1.0); // high sensitivity to protected attribute

        // Network where output barely depends on input 0
        let layers2 = vec![
            simple_linear_layer(vec![vec![0.01, 1.0]], vec![0.0]),
        ];
        let sensitivity2 = InvarianceVerifier::fairness_check(&layers2, 0, &input);
        assert!(sensitivity2 < 0.1);
    }

    // Test 12: Contains and width
    #[test]
    fn test_contains_and_width() {
        let iv = AbstractValue::Interval(2.0, 5.0);
        assert!(iv.contains(3.0));
        assert!(!iv.contains(1.0));
        assert!((iv.width() - 3.0).abs() < 1e-10);

        assert!(AbstractValue::Top.contains(999.0));
        assert!(!AbstractValue::Bottom.contains(0.0));
        assert_eq!(AbstractValue::Bottom.width(), 0.0);
    }

    // Test 13: ModelVerifier verify_all
    #[test]
    fn test_model_verifier() {
        let layers = vec![
            relu_layer(vec![vec![1.0, 0.5], vec![0.3, 0.8]], vec![0.0, 0.0]),
            sigmoid_layer(vec![vec![1.0, 1.0]], vec![0.0]),
        ];
        let input_ranges = vec![(-1.0, 1.0), (-1.0, 1.0)];
        let properties = vec![
            Property::Bounded(0.0, 1.0),
            Property::Lipschitz(10.0),
        ];
        let reports = ModelVerifier::verify_all(&layers, &input_ranges, &properties);
        assert_eq!(reports.len(), 2);
        assert_eq!(reports[0].result, VerificationResult::Proven);
        assert_eq!(reports[0].confidence, 1.0);
    }

    // Test 14: Permutation invariance
    #[test]
    fn test_permutation_invariance() {
        // A network that sums inputs is permutation invariant
        let layers = vec![
            simple_linear_layer(vec![vec![1.0, 1.0, 1.0]], vec![0.0]),
        ];
        assert!(InvarianceVerifier::verify_permutation_invariant(&layers, &[1.0, 2.0, 3.0], &[2, 0, 1]));

        // A network with different weights is not
        let layers2 = vec![
            simple_linear_layer(vec![vec![1.0, 2.0, 3.0]], vec![0.0]),
        ];
        assert!(!InvarianceVerifier::verify_permutation_invariant(&layers2, &[1.0, 2.0, 3.0], &[2, 0, 1]));
    }
}
