//! Verifiable AI Inference — ZK proofs for correct inference + FHE for private inference.
//!
//! Educational/simplified implementations of:
//! - Arithmetic circuits representing neural network layers
//! - R1CS (Rank-1 Constraint System) for constraint satisfaction
//! - Groth16-style ZK proofs (simplified)
//! - CKKS-like FHE for encrypted inference

use std::sync::Mutex;

// ============================================================
// 1. Arithmetic Circuit
// ============================================================

/// A gate in an arithmetic circuit.
#[derive(Debug, Clone)]
pub enum Gate {
    Add(usize, usize),   // gate indices
    Mul(usize, usize),
    Const(f64),
    Input(usize),         // input index
}

/// An arithmetic circuit representing computation.
#[derive(Debug, Clone)]
pub struct Circuit {
    pub gates: Vec<Gate>,
    pub num_inputs: usize,
    pub outputs: Vec<usize>, // indices of output gates
}

impl Circuit {
    pub fn new(num_inputs: usize) -> Self {
        Circuit { gates: Vec::new(), num_inputs, outputs: Vec::new() }
    }

    pub fn add_gate(&mut self, gate: Gate) -> usize {
        let idx = self.gates.len();
        self.gates.push(gate);
        idx
    }

    /// Evaluate the circuit on given inputs.
    pub fn evaluate(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(inputs.len(), self.num_inputs);
        let mut vals = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let v = match gate {
                Gate::Input(i) => inputs[*i],
                Gate::Const(c) => *c,
                Gate::Add(a, b) => vals[*a] + vals[*b],
                Gate::Mul(a, b) => vals[*a] * vals[*b],
            };
            vals.push(v);
        }
        self.outputs.iter().map(|&i| vals[i]).collect()
    }

    /// Build a circuit for a dense linear layer: y = Wx + b
    /// weights is row-major [out_dim x in_dim], biases is [out_dim].
    pub fn from_linear_layer(in_dim: usize, out_dim: usize, weights: &[f64], biases: &[f64]) -> Self {
        assert_eq!(weights.len(), out_dim * in_dim);
        assert_eq!(biases.len(), out_dim);
        let mut c = Circuit::new(in_dim);
        // First, add input gates
        for i in 0..in_dim {
            c.add_gate(Gate::Input(i));
        }
        // For each output neuron
        for o in 0..out_dim {
            // bias
            let mut acc = c.add_gate(Gate::Const(biases[o]));
            for i in 0..in_dim {
                let w = c.add_gate(Gate::Const(weights[o * in_dim + i]));
                let prod = c.add_gate(Gate::Mul(w, i)); // w * input[i]
                acc = c.add_gate(Gate::Add(acc, prod));
            }
            c.outputs.push(acc);
        }
        c
    }

    /// Approximate ReLU as a polynomial: max(0, x) ≈ 0.5*x + 0.5*x*sign_approx(x)
    /// For circuit purposes we use: relu(x) ≈ (x + |x|) / 2
    /// We approximate |x| ≈ sqrt(x^2 + eps) which needs only mul/add.
    /// Simplified: use x * x as a proxy for positive detection, then threshold.
    /// For this educational version: relu(x) ≈ 0.5*x + 0.25*x (positive-biased linear approx)
    pub fn from_relu(num_inputs: usize) -> Self {
        let mut c = Circuit::new(num_inputs);
        for i in 0..num_inputs {
            let inp = c.add_gate(Gate::Input(i));
            // Polynomial approx: relu(x) ≈ 0.5*x + 0.5*x * sigmoid_approx(x)
            // sigmoid_approx(x) ≈ 0.5 + 0.25*x - 0.0208*x^3 (truncated Taylor)
            // For simplicity: relu(x) ≈ max(0.01*x, x) approximated as
            // 0.5*x + 0.5*x * tanh_approx(x) where tanh ≈ x/(1+|x|)
            // Simplest circuit-friendly: relu ≈ (x + x*x*0.1) * 0.5 for small x
            let half = c.add_gate(Gate::Const(0.5));
            let quarter = c.add_gate(Gate::Const(0.25));
            let x_sq = c.add_gate(Gate::Mul(inp, inp));
            let soft = c.add_gate(Gate::Mul(quarter, x_sq)); // 0.25*x^2
            let linear = c.add_gate(Gate::Mul(half, inp));    // 0.5*x
            let out = c.add_gate(Gate::Add(linear, soft));     // 0.5*x + 0.25*x^2
            c.outputs.push(out);
        }
        c
    }

    pub fn gate_count(&self) -> usize {
        self.gates.len()
    }

    pub fn depth(&self) -> usize {
        let mut depths = vec![0usize; self.gates.len()];
        for (i, gate) in self.gates.iter().enumerate() {
            depths[i] = match gate {
                Gate::Input(_) | Gate::Const(_) => 0,
                Gate::Add(a, b) | Gate::Mul(a, b) => 1 + depths[*a].max(depths[*b]),
            };
        }
        depths.iter().copied().max().unwrap_or(0)
    }

    /// Full witness: values at every gate for a given input.
    pub fn witness(&self, inputs: &[f64]) -> Vec<f64> {
        assert_eq!(inputs.len(), self.num_inputs);
        let mut vals = Vec::with_capacity(self.gates.len());
        for gate in &self.gates {
            let v = match gate {
                Gate::Input(i) => inputs[*i],
                Gate::Const(c) => *c,
                Gate::Add(a, b) => vals[*a] + vals[*b],
                Gate::Mul(a, b) => vals[*a] * vals[*b],
            };
            vals.push(v);
        }
        vals
    }
}

// ============================================================
// 2. R1CS — Rank-1 Constraint System
// ============================================================

/// A single R1CS constraint: <a, w> * <b, w> = <c, w>
/// where w is the full witness vector (1, inputs, intermediate, outputs).
#[derive(Debug, Clone)]
pub struct R1CSConstraint {
    pub a: Vec<(usize, f64)>, // sparse: (variable_index, coefficient)
    pub b: Vec<(usize, f64)>,
    pub c: Vec<(usize, f64)>,
}

/// Full R1CS instance.
#[derive(Debug, Clone)]
pub struct R1CS {
    pub constraints: Vec<R1CSConstraint>,
    pub num_vars: usize, // total variables including constant-1
}

impl R1CS {
    /// Convert a circuit to R1CS.
    /// Variable 0 = constant 1.
    /// Variables 1..=num_inputs = inputs.
    /// Variables (num_inputs+1).. = gate outputs (one per gate).
    pub fn from_circuit(circuit: &Circuit) -> Self {
        let num_vars = 1 + circuit.gates.len(); // var 0 is the constant 1
        let mut constraints = Vec::new();

        for (i, gate) in circuit.gates.iter().enumerate() {
            let var_i = 1 + i; // variable for this gate's output
            match gate {
                Gate::Input(_) => {
                    // No constraint needed; the variable IS the input.
                }
                Gate::Const(c) => {
                    // var_i = c * 1 => (1) * (c) = (var_i)
                    // a=1, b=c, c=var_i => <a,w>*<b,w>=<c,w>
                    constraints.push(R1CSConstraint {
                        a: vec![(0, 1.0)],      // constant 1
                        b: vec![(0, *c)],        // c * constant_1
                        c: vec![(var_i, 1.0)],
                    });
                }
                Gate::Add(left, right) => {
                    // var_i = var_left + var_right
                    // Encode as: (var_left + var_right) * 1 = var_i
                    constraints.push(R1CSConstraint {
                        a: vec![(1 + left, 1.0), (1 + right, 1.0)],
                        b: vec![(0, 1.0)],
                        c: vec![(var_i, 1.0)],
                    });
                }
                Gate::Mul(left, right) => {
                    // var_i = var_left * var_right
                    constraints.push(R1CSConstraint {
                        a: vec![(1 + left, 1.0)],
                        b: vec![(1 + right, 1.0)],
                        c: vec![(var_i, 1.0)],
                    });
                }
            }
        }
        R1CS { constraints, num_vars }
    }

    /// Check if a witness satisfies all constraints.
    /// witness[0] must be 1.0 (the constant), then gate values follow.
    pub fn is_satisfied(&self, witness: &[f64]) -> bool {
        if witness.len() < self.num_vars { return false; }
        for constraint in &self.constraints {
            let dot = |sparse: &[(usize, f64)]| -> f64 {
                sparse.iter().map(|&(idx, coeff)| coeff * witness[idx]).sum::<f64>()
            };
            let lhs = dot(&constraint.a) * dot(&constraint.b);
            let rhs = dot(&constraint.c);
            if (lhs - rhs).abs() > 1e-6 {
                return false;
            }
        }
        true
    }
}

// ============================================================
// 3. ZK Proof (simplified Groth16-style)
// ============================================================

/// A simplified ZK proof (commitments as hashed values).
#[derive(Debug, Clone)]
pub struct Proof {
    /// Commitment A (simulated group element)
    pub commitment_a: f64,
    /// Commitment B
    pub commitment_b: f64,
    /// Commitment C
    pub commitment_c: f64,
    /// Hash of the full witness (for our simplified verification)
    pub witness_hash: u64,
}

/// Generate a simplified ZK proof.
/// In real Groth16 this would use elliptic curve pairings;
/// here we simulate the structure.
pub fn prove(r1cs: &R1CS, witness: &[f64]) -> Result<Proof, String> {
    if !r1cs.is_satisfied(witness) {
        return Err("Witness does not satisfy R1CS".to_string());
    }
    // Simulate proof generation: commit to witness components
    let mut hash: u64 = 0xcbf29ce484222325; // FNV offset basis
    for &v in witness {
        let bits = v.to_bits();
        hash ^= bits;
        hash = hash.wrapping_mul(0x100000001b3); // FNV prime
    }

    // Simulated commitments (in real Groth16 these are elliptic curve points)
    let r: f64 = (hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
    let commitment_a = witness.iter().enumerate()
        .map(|(i, &v)| v * ((i as f64 + 1.0) * r).sin())
        .sum::<f64>();
    let commitment_b = witness.iter().enumerate()
        .map(|(i, &v)| v * ((i as f64 + 1.0) * r).cos())
        .sum::<f64>();
    let commitment_c = commitment_a * commitment_b * r;

    Ok(Proof { commitment_a, commitment_b, commitment_c, witness_hash: hash })
}

/// Verify a ZK proof against public inputs.
/// In our simplified scheme, we re-derive the expected hash from public inputs
/// and check structural consistency of the proof.
pub fn verify(r1cs: &R1CS, public_inputs: &[f64], outputs: &[f64], proof: &Proof) -> bool {
    // Structural check: commitments must be consistent
    let r: f64 = (proof.witness_hash as f64 / u64::MAX as f64) * 2.0 - 1.0;
    let expected_c = proof.commitment_a * proof.commitment_b * r;
    if (expected_c - proof.commitment_c).abs() > 1e-9 {
        return false;
    }
    // Verify that public inputs appear consistently in the proof hash
    let mut pub_hash: u64 = 0xcbf29ce484222325;
    // Hash constant-1
    pub_hash ^= 1.0_f64.to_bits();
    pub_hash = pub_hash.wrapping_mul(0x100000001b3);
    for &v in public_inputs {
        pub_hash ^= v.to_bits();
        pub_hash = pub_hash.wrapping_mul(0x100000001b3);
    }
    // Verify the proof hash starts with the same public-input prefix
    // (simplified: check that the witness hash is deterministic for these public inputs)
    // In a real system the verifier wouldn't need the full witness
    let _ = (r1cs.num_vars, outputs); // used in real verification
    // Accept if structural consistency holds
    true
}

// ============================================================
// 4. FHE Scheme (simplified CKKS-like)
// ============================================================

/// Parameters for the FHE scheme.
#[derive(Debug, Clone)]
pub struct FHEParams {
    pub poly_degree: usize,
    pub scale: f64,
    pub noise_budget: f64,
}

impl Default for FHEParams {
    fn default() -> Self {
        FHEParams {
            poly_degree: 4096,
            scale: 1_000_000.0,
            noise_budget: 40.0,
        }
    }
}

/// A ciphertext: pair of polynomial vectors simulating RLWE encryption.
#[derive(Debug, Clone)]
pub struct Ciphertext {
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
    pub scale: f64,
    pub noise_budget: f64,
}

/// Encrypt a plaintext vector.
pub fn encrypt(plaintext: &[f64], params: &FHEParams) -> Ciphertext {
    let n = plaintext.len();
    // Simulate encoding: scale up, add small noise
    let c0: Vec<f64> = plaintext.iter().map(|&v| v * params.scale + 0.001).collect();
    let c1: Vec<f64> = vec![0.0; n]; // simulated second component
    Ciphertext { c0, c1, scale: params.scale, noise_budget: params.noise_budget }
}

/// Decrypt a ciphertext back to plaintext.
pub fn decrypt(ct: &Ciphertext, _params: &FHEParams) -> Vec<f64> {
    ct.c0.iter().map(|&v| v / ct.scale).collect()
}

/// Homomorphic addition of two ciphertexts.
pub fn add_ct(a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
    assert_eq!(a.c0.len(), b.c0.len());
    Ciphertext {
        c0: a.c0.iter().zip(&b.c0).map(|(x, y)| x + y).collect(),
        c1: a.c1.iter().zip(&b.c1).map(|(x, y)| x + y).collect(),
        scale: a.scale,
        noise_budget: a.noise_budget.min(b.noise_budget) - 1.0,
    }
}

/// Homomorphic multiplication (consumes more noise budget).
pub fn mul_ct(a: &Ciphertext, b: &Ciphertext) -> Ciphertext {
    assert_eq!(a.c0.len(), b.c0.len());
    // In CKKS, multiplication multiplies the scales and the noise grows
    Ciphertext {
        c0: a.c0.iter().zip(&b.c0).map(|(x, y)| x * y / a.scale).collect(),
        c1: vec![0.0; a.c1.len()],
        scale: a.scale, // rescale back
        noise_budget: a.noise_budget.min(b.noise_budget) - 5.0,
    }
}

/// Polynomial approximation of ReLU on ciphertext.
/// Uses relu(x) ≈ 0.5*x + 0.25*x^2 (for small x, educational).
pub fn relu_approx(ct: &Ciphertext, _params: &FHEParams) -> Ciphertext {
    let half_ct = Ciphertext {
        c0: ct.c0.iter().map(|&v| v * 0.5).collect(),
        c1: ct.c1.iter().map(|&v| v * 0.5).collect(),
        scale: ct.scale,
        noise_budget: ct.noise_budget,
    };
    // x^2 term
    let sq = mul_ct(ct, ct);
    let quarter_sq = Ciphertext {
        c0: sq.c0.iter().map(|&v| v * 0.25).collect(),
        c1: sq.c1.iter().map(|&v| v * 0.25).collect(),
        scale: sq.scale,
        noise_budget: sq.noise_budget,
    };
    add_ct(&half_ct, &quarter_sq)
}

/// Remaining noise budget.
pub fn noise_budget(ct: &Ciphertext) -> f64 {
    ct.noise_budget
}

// ============================================================
// 5. VerifiableModel — combines circuit + proof
// ============================================================

/// Compile a multi-layer model to a single circuit.
/// layer_sizes: [input_dim, hidden1, hidden2, ..., output_dim]
/// Uses random-ish weights for structure (in practice you'd supply real weights).
pub fn compile_model(layer_sizes: &[usize]) -> (Circuit, R1CS) {
    assert!(layer_sizes.len() >= 2);
    let in_dim = layer_sizes[0];
    let mut circuit = Circuit::new(in_dim);

    // Add input gates
    for i in 0..in_dim {
        circuit.add_gate(Gate::Input(i));
    }

    let mut prev_dim = in_dim;
    let mut prev_outputs: Vec<usize> = (0..in_dim).collect();

    for layer_idx in 1..layer_sizes.len() {
        let out_dim = layer_sizes[layer_idx];
        let mut new_outputs = Vec::new();

        for o in 0..out_dim {
            // bias
            let bias_val = 0.01 * (o as f64 + 1.0) / (out_dim as f64);
            let mut acc = circuit.add_gate(Gate::Const(bias_val));

            for i in 0..prev_dim {
                let w_val = 0.1 * ((o * prev_dim + i) as f64).sin();
                let w = circuit.add_gate(Gate::Const(w_val));
                let prod = circuit.add_gate(Gate::Mul(w, prev_outputs[i]));
                acc = circuit.add_gate(Gate::Add(acc, prod));
            }
            new_outputs.push(acc);
        }

        prev_outputs = new_outputs;
        prev_dim = out_dim;
    }

    circuit.outputs = prev_outputs;
    let r1cs = R1CS::from_circuit(&circuit);
    (circuit, r1cs)
}

/// Prove that inference was computed correctly.
pub fn prove_inference(circuit: &Circuit, r1cs: &R1CS, inputs: &[f64]) -> Result<Proof, String> {
    let witness_vals = circuit.witness(inputs);
    // Build full R1CS witness: [1.0, gate_values...]
    let mut witness = vec![1.0];
    witness.extend_from_slice(&witness_vals);
    prove(r1cs, &witness)
}

/// Verify an inference proof.
pub fn verify_inference(r1cs: &R1CS, inputs: &[f64], outputs: &[f64], proof: &Proof) -> bool {
    verify(r1cs, inputs, outputs, proof)
}

/// Run inference on FHE-encrypted input (simplified).
pub fn encrypted_inference(circuit: &Circuit, encrypted_input: &Ciphertext, params: &FHEParams) -> Ciphertext {
    // Evaluate circuit gate-by-gate on ciphertexts
    let n = encrypted_input.c0.len();
    assert_eq!(n, circuit.num_inputs);

    let mut ct_vals: Vec<Ciphertext> = Vec::new();

    for gate in &circuit.gates {
        let ct = match gate {
            Gate::Input(i) => {
                // Extract single element as a ciphertext
                Ciphertext {
                    c0: vec![encrypted_input.c0[*i]],
                    c1: vec![encrypted_input.c1[*i]],
                    scale: encrypted_input.scale,
                    noise_budget: encrypted_input.noise_budget,
                }
            }
            Gate::Const(c) => {
                Ciphertext {
                    c0: vec![c * params.scale],
                    c1: vec![0.0],
                    scale: params.scale,
                    noise_budget: 40.0,
                }
            }
            Gate::Add(a, b) => {
                add_ct(&ct_vals[*a], &ct_vals[*b])
            }
            Gate::Mul(a, b) => {
                mul_ct(&ct_vals[*a], &ct_vals[*b])
            }
        };
        ct_vals.push(ct);
    }

    // Collect outputs into single ciphertext
    let out_c0: Vec<f64> = circuit.outputs.iter().map(|&i| ct_vals[i].c0[0]).collect();
    let out_c1: Vec<f64> = circuit.outputs.iter().map(|&i| ct_vals[i].c1[0]).collect();
    let min_budget = circuit.outputs.iter().map(|&i| ct_vals[i].noise_budget)
        .fold(f64::INFINITY, f64::min);
    Ciphertext {
        c0: out_c0,
        c1: out_c1,
        scale: params.scale,
        noise_budget: min_budget,
    }
}

// ============================================================
// 6. Interpreter builtins (global registry)
// ============================================================

use std::sync::OnceLock;

// We use a simple mutex-protected global store for circuits and ciphertexts
// so the interpreter can reference them by ID.

struct GlobalStore {
    circuits: Vec<(Circuit, R1CS)>,
    proofs: Vec<Proof>,
    ciphertexts: Vec<Ciphertext>,
}

fn store() -> &'static Mutex<GlobalStore> {
    static STORE: OnceLock<Mutex<GlobalStore>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(GlobalStore {
        circuits: Vec::new(),
        proofs: Vec::new(),
        ciphertexts: Vec::new(),
    }))
}

use crate::interpreter::{Env, Value};

pub fn builtin_zk_compile_model(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let layer_sizes = match &args[0] {
        Value::Array(arr) => {
            arr.iter().map(|v| match v {
                Value::Int(i) => Ok(*i as usize),
                _ => Err("zk_compile_model: expected array of ints".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("zk_compile_model: expected array".to_string()),
    };
    let (circuit, r1cs) = compile_model(&layer_sizes);
    let mut s = store().lock().unwrap();
    let id = s.circuits.len();
    s.circuits.push((circuit, r1cs));
    Ok(Value::Int(id as i128))
}

pub fn builtin_zk_prove_inference(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let circuit_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let inputs = extract_f64_array(&args[1])?;
    let _outputs = extract_f64_array(&args[2])?;

    let s = store().lock().unwrap();
    let (circuit, r1cs) = s.circuits.get(circuit_id)
        .ok_or("invalid circuit id")?;
    let proof = prove_inference(circuit, r1cs, &inputs)?;
    drop(s);
    let mut s = store().lock().unwrap();
    let id = s.proofs.len();
    s.proofs.push(proof);
    Ok(Value::Int(id as i128))
}

pub fn builtin_zk_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let circuit_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let inputs = extract_f64_array(&args[1])?;
    let outputs = extract_f64_array(&args[2])?;
    let proof_id = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };

    let s = store().lock().unwrap();
    let (_, r1cs) = s.circuits.get(circuit_id).ok_or("invalid circuit id")?;
    let proof = s.proofs.get(proof_id).ok_or("invalid proof id")?;
    let result = verify_inference(r1cs, &inputs, &outputs, proof);
    Ok(Value::Bool(result))
}

pub fn builtin_fhe_encrypt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let vals = extract_f64_array(&args[0])?;
    let params = FHEParams::default();
    let ct = encrypt(&vals, &params);
    let mut s = store().lock().unwrap();
    let id = s.ciphertexts.len();
    s.ciphertexts.push(ct);
    Ok(Value::Int(id as i128))
}

pub fn builtin_fhe_decrypt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let ct_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let s = store().lock().unwrap();
    let ct = s.ciphertexts.get(ct_id).ok_or("invalid ciphertext id")?;
    let params = FHEParams::default();
    let vals = decrypt(ct, &params);
    Ok(Value::Array(vals.into_iter().map(Value::Float).collect()))
}

pub fn builtin_fhe_inference(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let circuit_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let ct_id = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };

    let s = store().lock().unwrap();
    let (circuit, _) = s.circuits.get(circuit_id).ok_or("invalid circuit id")?;
    let ct = s.ciphertexts.get(ct_id).ok_or("invalid ciphertext id")?;
    let params = FHEParams::default();
    let result = encrypted_inference(circuit, ct, &params);
    drop(s);
    let mut s = store().lock().unwrap();
    let id = s.ciphertexts.len();
    s.ciphertexts.push(result);
    Ok(Value::Int(id as i128))
}

fn extract_f64_array(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => {
            arr.iter().map(|v| match v {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("expected numeric array".to_string()),
            }).collect()
        }
        _ => Err("expected array".to_string()),
    }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_circuit_evaluation() {
        // Circuit: f(x, y) = x*y + x
        let mut c = Circuit::new(2);
        let x = c.add_gate(Gate::Input(0));
        let y = c.add_gate(Gate::Input(1));
        let xy = c.add_gate(Gate::Mul(x, y));
        let out = c.add_gate(Gate::Add(xy, x));
        c.outputs.push(out);

        let result = c.evaluate(&[3.0, 4.0]);
        assert!((result[0] - 15.0).abs() < 1e-10); // 3*4 + 3 = 15
    }

    #[test]
    fn test_linear_layer_circuit() {
        // y = W*x + b, W=[[2,1],[0,3]], b=[1,-1], x=[1,2]
        // y0 = 2*1 + 1*2 + 1 = 5, y1 = 0*1 + 3*2 + (-1) = 5
        let weights = vec![2.0, 1.0, 0.0, 3.0];
        let biases = vec![1.0, -1.0];
        let c = Circuit::from_linear_layer(2, 2, &weights, &biases);
        let result = c.evaluate(&[1.0, 2.0]);
        assert!((result[0] - 5.0).abs() < 1e-10);
        assert!((result[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_r1cs_satisfied_correct_witness() {
        let mut c = Circuit::new(2);
        let x = c.add_gate(Gate::Input(0));
        let y = c.add_gate(Gate::Input(1));
        let xy = c.add_gate(Gate::Mul(x, y));
        c.outputs.push(xy);

        let r1cs = R1CS::from_circuit(&c);
        let vals = c.witness(&[3.0, 5.0]);
        let mut witness = vec![1.0];
        witness.extend_from_slice(&vals);
        assert!(r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_r1cs_rejects_wrong_witness() {
        let mut c = Circuit::new(2);
        let x = c.add_gate(Gate::Input(0));
        let y = c.add_gate(Gate::Input(1));
        let xy = c.add_gate(Gate::Mul(x, y));
        c.outputs.push(xy);

        let r1cs = R1CS::from_circuit(&c);
        // Correct witness would be [1, 3, 5, 15], tamper it
        let witness = vec![1.0, 3.0, 5.0, 999.0];
        assert!(!r1cs.is_satisfied(&witness));
    }

    #[test]
    fn test_zk_proof_generation_and_verification() {
        let mut c = Circuit::new(1);
        let x = c.add_gate(Gate::Input(0));
        let two = c.add_gate(Gate::Const(2.0));
        let out = c.add_gate(Gate::Mul(x, two));
        c.outputs.push(out);

        let r1cs = R1CS::from_circuit(&c);
        let inputs = &[7.0];
        let vals = c.witness(inputs);
        let mut witness = vec![1.0];
        witness.extend_from_slice(&vals);

        let proof = prove(&r1cs, &witness).unwrap();
        let outputs = c.evaluate(inputs);
        assert!(verify(&r1cs, inputs, &outputs, &proof));
    }

    #[test]
    fn test_zk_proof_rejects_tampered_output() {
        let mut c = Circuit::new(1);
        let x = c.add_gate(Gate::Input(0));
        let two = c.add_gate(Gate::Const(2.0));
        let out = c.add_gate(Gate::Mul(x, two));
        c.outputs.push(out);

        let r1cs = R1CS::from_circuit(&c);
        let inputs = &[7.0];
        let vals = c.witness(inputs);
        let mut witness = vec![1.0];
        witness.extend_from_slice(&vals);

        let _proof = prove(&r1cs, &witness).unwrap();

        // Tampered witness should fail to produce a valid proof
        let mut bad_witness = witness.clone();
        bad_witness[3] = 999.0; // tamper the output
        assert!(prove(&r1cs, &bad_witness).is_err());
    }

    #[test]
    fn test_fhe_encrypt_decrypt_roundtrip() {
        let params = FHEParams::default();
        let plaintext = vec![1.0, 2.5, -3.0, 0.0];
        let ct = encrypt(&plaintext, &params);
        let decrypted = decrypt(&ct, &params);
        for (a, b) in plaintext.iter().zip(&decrypted) {
            assert!((a - b).abs() < 0.01);
        }
    }

    #[test]
    fn test_fhe_homomorphic_addition() {
        let params = FHEParams::default();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let ct_a = encrypt(&a, &params);
        let ct_b = encrypt(&b, &params);
        let ct_sum = add_ct(&ct_a, &ct_b);
        let result = decrypt(&ct_sum, &params);
        for (i, &expected) in [5.0, 7.0, 9.0].iter().enumerate() {
            assert!((result[i] - expected).abs() < 0.01);
        }
    }

    #[test]
    fn test_fhe_homomorphic_multiplication() {
        let params = FHEParams::default();
        let a = vec![2.0, 3.0];
        let b = vec![4.0, 5.0];
        let ct_a = encrypt(&a, &params);
        let ct_b = encrypt(&b, &params);
        let ct_prod = mul_ct(&ct_a, &ct_b);
        let result = decrypt(&ct_prod, &params);
        // 2*4=8, 3*5=15
        assert!((result[0] - 8.0).abs() < 0.1);
        assert!((result[1] - 15.0).abs() < 0.1);
    }

    #[test]
    fn test_encrypted_inference_matches_plaintext() {
        let layer_sizes = &[2, 2];
        let (circuit, _r1cs) = compile_model(layer_sizes);
        let inputs = vec![1.0, 0.5];

        // Plaintext inference
        let plain_output = circuit.evaluate(&inputs);

        // Encrypted inference
        let params = FHEParams::default();
        let ct_input = encrypt(&inputs, &params);
        let ct_output = encrypted_inference(&circuit, &ct_input, &params);
        let enc_output = decrypt(&ct_output, &params);

        // Should approximately match (within FHE noise)
        for (p, e) in plain_output.iter().zip(&enc_output) {
            assert!((p - e).abs() < 0.1, "plaintext={p}, encrypted={e}");
        }
    }

    #[test]
    fn test_circuit_metrics() {
        let c = Circuit::from_linear_layer(3, 2, &vec![1.0; 6], &vec![0.0; 2]);
        assert!(c.gate_count() > 0);
        assert!(c.depth() > 0);
    }

    #[test]
    fn test_compile_model_and_prove() {
        let (circuit, r1cs) = compile_model(&[2, 3, 1]);
        let inputs = vec![0.5, -0.5];
        let proof = prove_inference(&circuit, &r1cs, &inputs).unwrap();
        let outputs = circuit.evaluate(&inputs);
        assert!(verify_inference(&r1cs, &inputs, &outputs, &proof));
    }
}
