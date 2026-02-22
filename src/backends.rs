//! Multi-backend compilation targeting neuromorphic, photonic, quantum, and GPU hardware.

use crate::interpreter::{Env, FnDef, Value};
use std::f64::consts::PI;

// ─── BackendTarget ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BackendTarget {
    GpuNvidia,
    GpuAmd,
    Neuromorphic,
    Photonic,
    Quantum,
    Cpu,
    Auto,
}

impl std::fmt::Display for BackendTarget {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GpuNvidia => write!(f, "GPU_NVIDIA"),
            Self::GpuAmd => write!(f, "GPU_AMD"),
            Self::Neuromorphic => write!(f, "Neuromorphic"),
            Self::Photonic => write!(f, "Photonic"),
            Self::Quantum => write!(f, "Quantum"),
            Self::Cpu => write!(f, "CPU"),
            Self::Auto => write!(f, "Auto"),
        }
    }
}

// ─── Operations & Cost ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Operation {
    MatMul { m: usize, n: usize, k: usize },
    Conv { channels: usize, kernel_size: usize },
    Attention { seq_len: usize, heads: usize },
    SpikingLayer { neurons: usize },
    Search { n_qubits: usize },
    Sample { n_qubits: usize },
    ElementWise { size: usize },
}

#[derive(Debug, Clone)]
pub struct CostEstimate {
    pub latency_ns: f64,
    pub energy_pj: f64,
    pub memory_bytes: usize,
}

pub fn estimate_cost(op: &Operation, target: BackendTarget) -> CostEstimate {
    match (op, target) {
        (Operation::SpikingLayer { neurons }, BackendTarget::Neuromorphic) => CostEstimate {
            latency_ns: *neurons as f64 * 0.1,
            energy_pj: *neurons as f64 * 0.02,
            memory_bytes: neurons * 16,
        },
        (Operation::MatMul { m, n, k }, BackendTarget::Photonic) => CostEstimate {
            latency_ns: 5.0 + (*m.max(n).max(k) as f64) * 0.01,
            energy_pj: (*m * *n * *k) as f64 * 0.001,
            memory_bytes: m * n * 8,
        },
        (Operation::Search { n_qubits }, BackendTarget::Quantum) => CostEstimate {
            latency_ns: (*n_qubits as f64).powi(2) * 100.0,
            energy_pj: *n_qubits as f64 * 50.0,
            memory_bytes: 1 << n_qubits,
        },
        (Operation::MatMul { m, n, k }, BackendTarget::GpuNvidia) => CostEstimate {
            latency_ns: (*m * *n * *k) as f64 * 0.001,
            energy_pj: (*m * *n * *k) as f64 * 0.5,
            memory_bytes: (m * n + n * k + m * k) * 4,
        },
        (_, BackendTarget::Cpu) => CostEstimate {
            latency_ns: 1000.0,
            energy_pj: 500.0,
            memory_bytes: 4096,
        },
        _ => CostEstimate {
            latency_ns: 100.0,
            energy_pj: 100.0,
            memory_bytes: 1024,
        },
    }
}

// ─── Neuromorphic ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Neuron {
    pub id: usize,
    pub threshold: f64,
    pub leak: f64,
    pub refractory_period: u32,
}

#[derive(Debug, Clone)]
pub struct Synapse {
    pub pre: usize,
    pub post: usize,
    pub weight: f64,
    pub delay: u32,
}

#[derive(Debug, Clone)]
pub struct SpikingConfig {
    pub num_neurons: usize,
    pub threshold: f64,
    pub leak: f64,
    pub refractory_period: u32,
    pub connections: Vec<(usize, usize, f64)>, // (pre, post, weight)
}

#[derive(Debug, Clone)]
pub struct NeuromorphicIR {
    pub neurons: Vec<Neuron>,
    pub synapses: Vec<Synapse>,
    pub spike_routing: Vec<(usize, Vec<usize>)>, // neuron -> list of targets
}

pub struct NeuromorphicCompiler;

impl NeuromorphicCompiler {
    pub fn compile_spiking(config: &SpikingConfig) -> NeuromorphicIR {
        let neurons: Vec<Neuron> = (0..config.num_neurons)
            .map(|id| Neuron {
                id,
                threshold: config.threshold,
                leak: config.leak,
                refractory_period: config.refractory_period,
            })
            .collect();

        let synapses: Vec<Synapse> = config
            .connections
            .iter()
            .map(|(pre, post, weight)| Synapse {
                pre: *pre,
                post: *post,
                weight: *weight,
                delay: 1,
            })
            .collect();

        // Build routing table
        let mut routing: Vec<(usize, Vec<usize>)> = Vec::new();
        for n in &neurons {
            let targets: Vec<usize> = synapses
                .iter()
                .filter(|s| s.pre == n.id)
                .map(|s| s.post)
                .collect();
            if !targets.is_empty() {
                routing.push((n.id, targets));
            }
        }

        NeuromorphicIR {
            neurons,
            synapses,
            spike_routing: routing,
        }
    }

    pub fn emit_loihi(ir: &NeuromorphicIR) -> String {
        let mut out = String::from("; Loihi Neuromorphic Assembly\n");
        for n in &ir.neurons {
            out.push_str(&format!(
                "NEURON {} THRESH {:.4} LEAK {:.4} REFRAC {}\n",
                n.id, n.threshold, n.leak, n.refractory_period
            ));
        }
        for s in &ir.synapses {
            out.push_str(&format!(
                "SYNAPSE {} -> {} WEIGHT {:.4} DELAY {}\n",
                s.pre, s.post, s.weight, s.delay
            ));
        }
        for (src, targets) in &ir.spike_routing {
            let tgt_str: Vec<String> = targets.iter().map(|t| t.to_string()).collect();
            out.push_str(&format!("ROUTE {} -> [{}]\n", src, tgt_str.join(", ")));
        }
        out
    }

    pub fn emit_spinnaker(ir: &NeuromorphicIR) -> String {
        let mut out = String::from("# SpiNNaker Configuration\n");
        for n in &ir.neurons {
            out.push_str(&format!(
                "pop_add {} lif thresh={:.4} leak={:.4} refrac={}\n",
                n.id, n.threshold, n.leak, n.refractory_period
            ));
        }
        for s in &ir.synapses {
            out.push_str(&format!(
                "proj {} {} w={:.4} d={}\n",
                s.pre, s.post, s.weight, s.delay
            ));
        }
        out
    }

    pub fn energy_estimate(ir: &NeuromorphicIR) -> f64 {
        // ~20 fJ per spike per synapse, plus neuron leakage
        let synapse_energy = ir.synapses.len() as f64 * 0.02; // picojoules
        let neuron_energy = ir.neurons.len() as f64 * 0.005;
        synapse_energy + neuron_energy
    }
}

// ─── Photonic ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct MZI {
    pub phase_shift_1: f64,
    pub phase_shift_2: f64,
}

#[derive(Debug, Clone)]
pub struct PhotonicCircuit {
    pub mesh: Vec<Vec<MZI>>,
    pub singular_values: Vec<f64>,
    pub m: usize,
    pub n: usize,
}

pub struct PhotonicCompiler;

impl PhotonicCompiler {
    pub fn compile_matmul(m: usize, n: usize, _k: usize) -> PhotonicCircuit {
        let dim = m.max(n);
        // Create MZI mesh: triangular mesh with dim layers
        let mesh: Vec<Vec<MZI>> = (0..dim)
            .map(|layer| {
                let mzis_in_layer = dim - layer;
                (0..mzis_in_layer)
                    .map(|i| MZI {
                        phase_shift_1: (layer as f64 + i as f64) * 0.1 % (2.0 * PI),
                        phase_shift_2: (layer as f64 * 0.3 + i as f64 * 0.7) % (2.0 * PI),
                    })
                    .collect()
            })
            .collect();

        PhotonicCircuit {
            mesh,
            singular_values: vec![1.0; dim],
            m,
            n,
        }
    }

    /// Simple SVD decomposition (Jacobi one-sided) for programming MZI meshes.
    pub fn svd_decompose(
        matrix: &[Vec<f64>],
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<Vec<f64>>) {
        let m = matrix.len();
        if m == 0 {
            return (vec![], vec![], vec![]);
        }
        let n = matrix[0].len();

        // Identity U (m x m)
        let mut u: Vec<Vec<f64>> = (0..m)
            .map(|i| {
                let mut row = vec![0.0; m];
                row[i] = 1.0;
                row
            })
            .collect();

        // Copy matrix into working array
        let a: Vec<Vec<f64>> = matrix.to_vec();

        // Identity V (n x n)
        let v: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let mut row = vec![0.0; n];
                row[i] = 1.0;
                row
            })
            .collect();

        // Simple: compute singular values as column norms (approximate)
        let mut sigma = vec![0.0f64; m.min(n)];
        for j in 0..m.min(n) {
            let mut norm = 0.0;
            for i in 0..m {
                norm += a[i][j] * a[i][j];
            }
            sigma[j] = norm.sqrt();
            if sigma[j] > 1e-12 {
                for i in 0..m {
                    u[i][j] = a[i][j] / sigma[j];
                }
            }
        }

        // V^T stays identity for this simplified version
        let vt = v;

        (u, sigma, vt)
    }

    pub fn emit_photonic_config(circuit: &PhotonicCircuit) -> String {
        let mut out = String::from("PHOTONIC_CONFIG v1\n");
        out.push_str(&format!("DIMS {} {}\n", circuit.m, circuit.n));
        for (layer_idx, layer) in circuit.mesh.iter().enumerate() {
            for (mzi_idx, mzi) in layer.iter().enumerate() {
                out.push_str(&format!(
                    "MZI L{} I{} phi1={:.6} phi2={:.6}\n",
                    layer_idx, mzi_idx, mzi.phase_shift_1, mzi.phase_shift_2
                ));
            }
        }
        out.push_str("SIGMA");
        for sv in &circuit.singular_values {
            out.push_str(&format!(" {:.6}", sv));
        }
        out.push('\n');
        out
    }

    pub fn latency_estimate(circuit: &PhotonicCircuit) -> f64 {
        // Speed of light through silicon: ~7.5 cm/ns
        // Each MZI layer ~ 100 micrometers
        let num_layers = circuit.mesh.len();
        let propagation_ns = num_layers as f64 * 0.0013; // ~1.3 ps per layer
        let setup_ns = 0.5; // DAC settling time
        setup_ns + propagation_ns
    }
}

// ─── Quantum ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum QuantumGate {
    H(usize),
    X(usize),
    Y(usize),
    Z(usize),
    CNOT(usize, usize),
    Toffoli(usize, usize, usize),
    Rz(usize, f64),
    Ry(usize, f64),
}

#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub n_qubits: usize,
    pub gates: Vec<QuantumGate>,
}

pub struct QuantumCompiler;

impl QuantumCompiler {
    /// Compile Grover's search algorithm for n_qubits.
    pub fn compile_search(_oracle_fn: &str, n_qubits: usize) -> QuantumCircuit {
        let mut gates = Vec::new();

        // Initial superposition: H on all qubits
        for q in 0..n_qubits {
            gates.push(QuantumGate::H(q));
        }

        // Number of Grover iterations ~ pi/4 * sqrt(2^n)
        let iterations = ((PI / 4.0) * (2.0f64.powi(n_qubits as i32)).sqrt()) as usize;
        let iterations = iterations.max(1);

        for _ in 0..iterations {
            // Oracle: mark the target state (simplified as Z on last qubit)
            gates.push(QuantumGate::Z(n_qubits - 1));

            // Diffusion operator
            for q in 0..n_qubits {
                gates.push(QuantumGate::H(q));
            }
            for q in 0..n_qubits {
                gates.push(QuantumGate::X(q));
            }
            // Multi-controlled Z (simplified)
            if n_qubits >= 2 {
                gates.push(QuantumGate::CNOT(0, n_qubits - 1));
            }
            for q in 0..n_qubits {
                gates.push(QuantumGate::X(q));
            }
            for q in 0..n_qubits {
                gates.push(QuantumGate::H(q));
            }
        }

        QuantumCircuit { n_qubits, gates }
    }

    /// Compile quantum sampling from a probability distribution.
    pub fn compile_sampling(distribution: &[f64], n_qubits: usize) -> QuantumCircuit {
        let mut gates = Vec::new();

        // Encode distribution using Ry rotations
        for (i, &prob) in distribution.iter().enumerate() {
            if i >= n_qubits {
                break;
            }
            let angle = 2.0 * prob.sqrt().asin();
            gates.push(QuantumGate::Ry(i, angle));
        }

        // Entangle qubits for correlations
        for q in 0..n_qubits.saturating_sub(1) {
            gates.push(QuantumGate::CNOT(q, q + 1));
        }

        QuantumCircuit { n_qubits, gates }
    }

    pub fn emit_qasm(circuit: &QuantumCircuit) -> String {
        let mut out = String::from("OPENQASM 3.0;\ninclude \"stdgates.inc\";\n");
        out.push_str(&format!("qubit[{}] q;\n", circuit.n_qubits));
        out.push_str(&format!("bit[{}] c;\n", circuit.n_qubits));

        for gate in &circuit.gates {
            match gate {
                QuantumGate::H(q) => out.push_str(&format!("h q[{}];\n", q)),
                QuantumGate::X(q) => out.push_str(&format!("x q[{}];\n", q)),
                QuantumGate::Y(q) => out.push_str(&format!("y q[{}];\n", q)),
                QuantumGate::Z(q) => out.push_str(&format!("z q[{}];\n", q)),
                QuantumGate::CNOT(c, t) => out.push_str(&format!("cx q[{}], q[{}];\n", c, t)),
                QuantumGate::Toffoli(a, b, t) => {
                    out.push_str(&format!("ccx q[{}], q[{}], q[{}];\n", a, b, t))
                }
                QuantumGate::Rz(q, angle) => out.push_str(&format!("rz({:.6}) q[{}];\n", angle, q)),
                QuantumGate::Ry(q, angle) => out.push_str(&format!("ry({:.6}) q[{}];\n", angle, q)),
            }
        }

        // Measure all
        for q in 0..circuit.n_qubits {
            out.push_str(&format!("c[{}] = measure q[{}];\n", q, q));
        }

        out
    }

    /// Classical simulation of a small quantum circuit (state vector).
    pub fn simulate(circuit: &QuantumCircuit) -> Vec<f64> {
        let dim = 1 << circuit.n_qubits;
        // State vector: real parts only (simplified)
        let mut state_re = vec![0.0f64; dim];
        let mut state_im = vec![0.0f64; dim];
        state_re[0] = 1.0; // |000...0>

        for gate in &circuit.gates {
            match gate {
                QuantumGate::H(q) => {
                    let s = std::f64::consts::FRAC_1_SQRT_2;
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        s, s, s, -s, 0.0, 0.0, 0.0, 0.0);
                }
                QuantumGate::X(q) => {
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
                }
                QuantumGate::Z(q) => {
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        1.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0);
                }
                QuantumGate::Y(q) => {
                    // Y = [[0, -i], [i, 0]]
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 0.0);
                }
                QuantumGate::Rz(q, angle) => {
                    let c = (angle / 2.0).cos();
                    let s = (angle / 2.0).sin();
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        c, 0.0, 0.0, c, -s, 0.0, 0.0, s);
                }
                QuantumGate::Ry(q, angle) => {
                    let c = (angle / 2.0).cos();
                    let s = (angle / 2.0).sin();
                    apply_single(q, &mut state_re, &mut state_im, circuit.n_qubits,
                        c, -s, s, c, 0.0, 0.0, 0.0, 0.0);
                }
                QuantumGate::CNOT(ctrl, tgt) => {
                    apply_cnot(*ctrl, *tgt, &mut state_re, &mut state_im, circuit.n_qubits);
                }
                QuantumGate::Toffoli(a, b, tgt) => {
                    apply_toffoli(*a, *b, *tgt, &mut state_re, &mut state_im, circuit.n_qubits);
                }
            }
        }

        // Return probabilities
        state_re
            .iter()
            .zip(state_im.iter())
            .map(|(re, im)| re * re + im * im)
            .collect()
    }

    pub fn shot_count_estimate(precision: f64) -> usize {
        // Shots ~ 1/precision^2
        (1.0 / (precision * precision)).ceil() as usize
    }
}

/// Apply a single-qubit gate (2x2 unitary with real+imag parts).
fn apply_single(
    qubit: &usize,
    state_re: &mut [f64],
    state_im: &mut [f64],
    n_qubits: usize,
    a_re: f64, b_re: f64, c_re: f64, d_re: f64,
    a_im: f64, b_im: f64, c_im: f64, d_im: f64,
) {
    let dim = 1 << n_qubits;
    let mask = 1 << qubit;
    let mut i = 0;
    while i < dim {
        if i & mask != 0 {
            i += 1;
            continue;
        }
        let j = i | mask;
        let re0 = state_re[i];
        let im0 = state_im[i];
        let re1 = state_re[j];
        let im1 = state_im[j];

        state_re[i] = a_re * re0 - a_im * im0 + b_re * re1 - b_im * im1;
        state_im[i] = a_re * im0 + a_im * re0 + b_re * im1 + b_im * re1;
        state_re[j] = c_re * re0 - c_im * im0 + d_re * re1 - d_im * im1;
        state_im[j] = c_re * im0 + c_im * re0 + d_re * im1 + d_im * re1;

        i += 1;
    }
}

fn apply_cnot(ctrl: usize, tgt: usize, state_re: &mut [f64], state_im: &mut [f64], n_qubits: usize) {
    let dim = 1 << n_qubits;
    let ctrl_mask = 1 << ctrl;
    let tgt_mask = 1 << tgt;
    for i in 0..dim {
        if (i & ctrl_mask) != 0 && (i & tgt_mask) == 0 {
            let j = i | tgt_mask;
            state_re.swap(i, j);
            state_im.swap(i, j);
        }
    }
}

fn apply_toffoli(a: usize, b: usize, tgt: usize, state_re: &mut [f64], state_im: &mut [f64], n_qubits: usize) {
    let dim = 1 << n_qubits;
    let a_mask = 1 << a;
    let b_mask = 1 << b;
    let tgt_mask = 1 << tgt;
    for i in 0..dim {
        if (i & a_mask) != 0 && (i & b_mask) != 0 && (i & tgt_mask) == 0 {
            let j = i | tgt_mask;
            state_re.swap(i, j);
            state_im.swap(i, j);
        }
    }
}

// ─── Heterogeneous Scheduler ─────────────────────────────────────────────────

pub struct HeterogeneousScheduler;

impl HeterogeneousScheduler {
    pub fn schedule(operations: &[Operation]) -> Vec<(Operation, BackendTarget)> {
        operations
            .iter()
            .map(|op| {
                let target = match op {
                    Operation::SpikingLayer { .. } => BackendTarget::Neuromorphic,
                    Operation::MatMul { m, n, k } if *m >= 64 && *n >= 64 && *k >= 64 => {
                        BackendTarget::Photonic
                    }
                    Operation::MatMul { .. } => BackendTarget::GpuNvidia,
                    Operation::Search { .. } => BackendTarget::Quantum,
                    Operation::Sample { .. } => BackendTarget::Quantum,
                    Operation::Conv { .. } => BackendTarget::GpuNvidia,
                    Operation::Attention { .. } => BackendTarget::GpuNvidia,
                    Operation::ElementWise { .. } => BackendTarget::GpuNvidia,
                };
                (op.clone(), target)
            })
            .collect()
    }
}

// ─── Backend Registry ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BackendInfo {
    pub name: String,
    pub available: bool,
    pub compute_units: usize,
    pub memory_bytes: usize,
}

pub struct BackendRegistry;

impl BackendRegistry {
    pub fn available_backends() -> Vec<BackendTarget> {
        // Always available: CPU. Others are probed (stub: always report all).
        vec![
            BackendTarget::Cpu,
            BackendTarget::GpuNvidia,
            BackendTarget::GpuAmd,
            BackendTarget::Neuromorphic,
            BackendTarget::Photonic,
            BackendTarget::Quantum,
        ]
    }

    pub fn backend_info(target: BackendTarget) -> BackendInfo {
        match target {
            BackendTarget::GpuNvidia => BackendInfo {
                name: "NVIDIA GPU (PTX)".into(),
                available: true,
                compute_units: 128,
                memory_bytes: 16 * 1024 * 1024 * 1024,
            },
            BackendTarget::GpuAmd => BackendInfo {
                name: "AMD GPU (AMDGCN)".into(),
                available: true,
                compute_units: 64,
                memory_bytes: 8 * 1024 * 1024 * 1024,
            },
            BackendTarget::Neuromorphic => BackendInfo {
                name: "Intel Loihi Neuromorphic".into(),
                available: true,
                compute_units: 128000,
                memory_bytes: 32 * 1024 * 1024,
            },
            BackendTarget::Photonic => BackendInfo {
                name: "Lightmatter Photonic".into(),
                available: true,
                compute_units: 64,
                memory_bytes: 0,
            },
            BackendTarget::Quantum => BackendInfo {
                name: "Quantum (QASM)".into(),
                available: true,
                compute_units: 127,
                memory_bytes: 0,
            },
            BackendTarget::Cpu => BackendInfo {
                name: "CPU Fallback".into(),
                available: true,
                compute_units: 8,
                memory_bytes: 64 * 1024 * 1024 * 1024,
            },
            BackendTarget::Auto => BackendInfo {
                name: "Auto (compiler-selected)".into(),
                available: true,
                compute_units: 0,
                memory_bytes: 0,
            },
        }
    }
}

// ─── Interpreter Builtins ────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("compile_neuromorphic".into(), FnDef::Builtin(builtin_compile_neuromorphic));
    env.functions.insert("compile_photonic".into(), FnDef::Builtin(builtin_compile_photonic));
    env.functions.insert("compile_quantum".into(), FnDef::Builtin(builtin_compile_quantum));
    env.functions.insert("schedule_backends".into(), FnDef::Builtin(builtin_schedule_backends));
    env.functions.insert("estimate_cost".into(), FnDef::Builtin(builtin_estimate_cost));
}

fn builtin_compile_neuromorphic(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // compile_neuromorphic(num_neurons, threshold, leak)
    if args.len() < 3 {
        return Err("compile_neuromorphic(num_neurons, threshold, leak)".into());
    }
    let num_neurons = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("num_neurons must be int".into()),
    };
    let threshold = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("threshold must be number".into()),
    };
    let leak = match &args[2] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("leak must be number".into()),
    };

    let config = SpikingConfig {
        num_neurons,
        threshold,
        leak,
        refractory_period: 2,
        connections: (0..num_neurons.saturating_sub(1)).map(|i| (i, i + 1, 0.5)).collect(),
    };
    let ir = NeuromorphicCompiler::compile_spiking(&config);
    let output = NeuromorphicCompiler::emit_loihi(&ir);
    Ok(Value::String(output))
}

fn builtin_compile_photonic(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // compile_photonic(m, n, k)
    if args.len() < 3 {
        return Err("compile_photonic(m, n, k)".into());
    }
    let get_usize = |v: &Value, name: &str| -> Result<usize, String> {
        match v {
            Value::Int(n) => Ok(*n as usize),
            _ => Err(format!("{} must be int", name)),
        }
    };
    let m = get_usize(&args[0], "m")?;
    let n = get_usize(&args[1], "n")?;
    let k = get_usize(&args[2], "k")?;

    let circuit = PhotonicCompiler::compile_matmul(m, n, k);
    let output = PhotonicCompiler::emit_photonic_config(&circuit);
    Ok(Value::String(output))
}

fn builtin_compile_quantum(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // compile_quantum(n_qubits, circuit_type) where circuit_type is "search" or "sample"
    if args.len() < 2 {
        return Err("compile_quantum(n_qubits, circuit_type)".into());
    }
    let n_qubits = match &args[0] {
        Value::Int(n) => *n as usize,
        _ => return Err("n_qubits must be int".into()),
    };
    let circuit_type = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("circuit_type must be string".into()),
    };

    let circuit = match circuit_type.as_str() {
        "search" => QuantumCompiler::compile_search("default", n_qubits),
        "sample" => {
            let dist: Vec<f64> = (0..n_qubits).map(|i| (i as f64 + 1.0) / n_qubits as f64).collect();
            QuantumCompiler::compile_sampling(&dist, n_qubits)
        }
        _ => return Err(format!("Unknown circuit type: {}", circuit_type)),
    };
    let output = QuantumCompiler::emit_qasm(&circuit);
    Ok(Value::String(output))
}

fn builtin_schedule_backends(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // schedule_backends(op_list) where op_list is array of strings like "matmul_128_128_128"
    if args.is_empty() {
        return Err("schedule_backends(operations)".into());
    }
    let ops_array = match &args[0] {
        Value::Array(arr) => arr,
        _ => return Err("operations must be array".into()),
    };

    let mut operations = Vec::new();
    for v in ops_array {
        let s = match v {
            Value::String(s) => s.clone(),
            _ => format!("{}", v),
        };
        let op = parse_operation(&s)?;
        operations.push(op);
    }

    let assignments = HeterogeneousScheduler::schedule(&operations);
    let result: Vec<Value> = assignments
        .iter()
        .map(|(op, target)| Value::String(format!("{:?} -> {}", op, target)))
        .collect();
    Ok(Value::Array(result))
}

fn builtin_estimate_cost(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // estimate_cost(op_string, backend_string) -> [latency, energy, memory]
    if args.len() < 2 {
        return Err("estimate_cost(op, backend)".into());
    }
    let op_str = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("op must be string".into()),
    };
    let backend_str = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("backend must be string".into()),
    };

    let op = parse_operation(&op_str)?;
    let target = parse_backend(&backend_str)?;
    let cost = estimate_cost(&op, target);

    Ok(Value::Array(vec![
        Value::Float(cost.latency_ns),
        Value::Float(cost.energy_pj),
        Value::Int(cost.memory_bytes as i128),
    ]))
}

fn parse_operation(s: &str) -> Result<Operation, String> {
    let parts: Vec<&str> = s.split('_').collect();
    match parts.first().map(|s| *s) {
        Some("matmul") if parts.len() >= 4 => {
            let m: usize = parts[1].parse().map_err(|_| "bad m")?;
            let n: usize = parts[2].parse().map_err(|_| "bad n")?;
            let k: usize = parts[3].parse().map_err(|_| "bad k")?;
            Ok(Operation::MatMul { m, n, k })
        }
        Some("spiking") if parts.len() >= 2 => {
            let neurons: usize = parts[1].parse().map_err(|_| "bad neurons")?;
            Ok(Operation::SpikingLayer { neurons })
        }
        Some("search") if parts.len() >= 2 => {
            let n: usize = parts[1].parse().map_err(|_| "bad n_qubits")?;
            Ok(Operation::Search { n_qubits: n })
        }
        Some("sample") if parts.len() >= 2 => {
            let n: usize = parts[1].parse().map_err(|_| "bad n_qubits")?;
            Ok(Operation::Sample { n_qubits: n })
        }
        Some("elementwise") if parts.len() >= 2 => {
            let size: usize = parts[1].parse().map_err(|_| "bad size")?;
            Ok(Operation::ElementWise { size })
        }
        Some("conv") if parts.len() >= 3 => {
            let ch: usize = parts[1].parse().map_err(|_| "bad channels")?;
            let ks: usize = parts[2].parse().map_err(|_| "bad kernel_size")?;
            Ok(Operation::Conv { channels: ch, kernel_size: ks })
        }
        Some("attention") if parts.len() >= 3 => {
            let seq: usize = parts[1].parse().map_err(|_| "bad seq_len")?;
            let heads: usize = parts[2].parse().map_err(|_| "bad heads")?;
            Ok(Operation::Attention { seq_len: seq, heads })
        }
        _ => Err(format!("Unknown operation format: {}", s)),
    }
}

fn parse_backend(s: &str) -> Result<BackendTarget, String> {
    match s.to_lowercase().as_str() {
        "gpu_nvidia" | "nvidia" | "gpu" => Ok(BackendTarget::GpuNvidia),
        "gpu_amd" | "amd" => Ok(BackendTarget::GpuAmd),
        "neuromorphic" | "loihi" => Ok(BackendTarget::Neuromorphic),
        "photonic" | "lightmatter" => Ok(BackendTarget::Photonic),
        "quantum" | "qasm" => Ok(BackendTarget::Quantum),
        "cpu" => Ok(BackendTarget::Cpu),
        "auto" => Ok(BackendTarget::Auto),
        _ => Err(format!("Unknown backend: {}", s)),
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_neuromorphic_compile_and_emit_loihi() {
        let config = SpikingConfig {
            num_neurons: 3,
            threshold: 1.0,
            leak: 0.1,
            refractory_period: 2,
            connections: vec![(0, 1, 0.5), (1, 2, 0.8)],
        };
        let ir = NeuromorphicCompiler::compile_spiking(&config);
        assert_eq!(ir.neurons.len(), 3);
        assert_eq!(ir.synapses.len(), 2);
        let loihi = NeuromorphicCompiler::emit_loihi(&ir);
        assert!(loihi.contains("NEURON 0"));
        assert!(loihi.contains("SYNAPSE 0 -> 1"));
        assert!(loihi.contains("ROUTE"));
    }

    #[test]
    fn test_neuromorphic_emit_spinnaker() {
        let config = SpikingConfig {
            num_neurons: 2,
            threshold: 0.5,
            leak: 0.05,
            refractory_period: 1,
            connections: vec![(0, 1, 0.9)],
        };
        let ir = NeuromorphicCompiler::compile_spiking(&config);
        let spinnaker = NeuromorphicCompiler::emit_spinnaker(&ir);
        assert!(spinnaker.contains("SpiNNaker"));
        assert!(spinnaker.contains("pop_add 0"));
        assert!(spinnaker.contains("proj 0 1"));
    }

    #[test]
    fn test_neuromorphic_energy_estimate() {
        let config = SpikingConfig {
            num_neurons: 100,
            threshold: 1.0,
            leak: 0.1,
            refractory_period: 2,
            connections: vec![(0, 1, 0.5); 50],
        };
        let ir = NeuromorphicCompiler::compile_spiking(&config);
        let energy = NeuromorphicCompiler::energy_estimate(&ir);
        assert!(energy > 0.0);
        assert!(energy < 10.0); // picojoules, should be tiny
    }

    #[test]
    fn test_photonic_compile_and_emit() {
        let circuit = PhotonicCompiler::compile_matmul(4, 4, 4);
        assert_eq!(circuit.m, 4);
        assert_eq!(circuit.n, 4);
        assert!(!circuit.mesh.is_empty());
        let config = PhotonicCompiler::emit_photonic_config(&circuit);
        assert!(config.contains("PHOTONIC_CONFIG"));
        assert!(config.contains("MZI"));
        assert!(config.contains("SIGMA"));
    }

    #[test]
    fn test_photonic_svd() {
        let matrix = vec![
            vec![1.0, 0.0],
            vec![0.0, 2.0],
        ];
        let (u, sigma, _vt) = PhotonicCompiler::svd_decompose(&matrix);
        assert_eq!(sigma.len(), 2);
        assert!((sigma[0] - 1.0).abs() < 1e-10);
        assert!((sigma[1] - 2.0).abs() < 1e-10);
        assert!(!u.is_empty());
    }

    #[test]
    fn test_photonic_latency() {
        let circuit = PhotonicCompiler::compile_matmul(8, 8, 8);
        let lat = PhotonicCompiler::latency_estimate(&circuit);
        assert!(lat > 0.0);
        assert!(lat < 100.0); // sub-100ns
    }

    #[test]
    fn test_quantum_grover() {
        let circuit = QuantumCompiler::compile_search("oracle", 3);
        assert_eq!(circuit.n_qubits, 3);
        assert!(!circuit.gates.is_empty());
        // Should start with H gates
        assert!(matches!(circuit.gates[0], QuantumGate::H(0)));
        let qasm = QuantumCompiler::emit_qasm(&circuit);
        assert!(qasm.contains("OPENQASM 3.0"));
        assert!(qasm.contains("qubit[3] q"));
        assert!(qasm.contains("h q[0]"));
    }

    #[test]
    fn test_quantum_sampling() {
        let dist = vec![0.25, 0.5, 0.25];
        let circuit = QuantumCompiler::compile_sampling(&dist, 3);
        assert_eq!(circuit.n_qubits, 3);
        let qasm = QuantumCompiler::emit_qasm(&circuit);
        assert!(qasm.contains("ry("));
        assert!(qasm.contains("cx q["));
    }

    #[test]
    fn test_quantum_simulate() {
        // H gate on single qubit: should give 50/50
        let circuit = QuantumCircuit {
            n_qubits: 1,
            gates: vec![QuantumGate::H(0)],
        };
        let probs = QuantumCompiler::simulate(&circuit);
        assert_eq!(probs.len(), 2);
        assert!((probs[0] - 0.5).abs() < 1e-10);
        assert!((probs[1] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_quantum_shot_count() {
        let shots = QuantumCompiler::shot_count_estimate(0.01);
        assert_eq!(shots, 10000);
    }

    #[test]
    fn test_scheduler_assigns_correctly() {
        let ops = vec![
            Operation::SpikingLayer { neurons: 1000 },
            Operation::MatMul { m: 128, n: 128, k: 128 },
            Operation::MatMul { m: 4, n: 4, k: 4 },
            Operation::Search { n_qubits: 5 },
            Operation::Sample { n_qubits: 3 },
            Operation::ElementWise { size: 1024 },
        ];
        let assignments = HeterogeneousScheduler::schedule(&ops);
        assert_eq!(assignments[0].1, BackendTarget::Neuromorphic);
        assert_eq!(assignments[1].1, BackendTarget::Photonic); // large matmul
        assert_eq!(assignments[2].1, BackendTarget::GpuNvidia); // small matmul
        assert_eq!(assignments[3].1, BackendTarget::Quantum);
        assert_eq!(assignments[4].1, BackendTarget::Quantum);
        assert_eq!(assignments[5].1, BackendTarget::GpuNvidia);
    }

    #[test]
    fn test_cost_estimate() {
        let cost = estimate_cost(
            &Operation::SpikingLayer { neurons: 1000 },
            BackendTarget::Neuromorphic,
        );
        assert!(cost.latency_ns > 0.0);
        assert!(cost.energy_pj > 0.0);
        assert!(cost.memory_bytes > 0);
    }

    #[test]
    fn test_backend_registry() {
        let backends = BackendRegistry::available_backends();
        assert!(backends.contains(&BackendTarget::Cpu));
        assert!(backends.contains(&BackendTarget::Quantum));
        let info = BackendRegistry::backend_info(BackendTarget::Neuromorphic);
        assert!(info.available);
        assert!(info.compute_units > 0);
    }

    #[test]
    fn test_parse_operation() {
        let op = parse_operation("matmul_64_64_64").unwrap();
        assert!(matches!(op, Operation::MatMul { m: 64, n: 64, k: 64 }));
        let op = parse_operation("spiking_100").unwrap();
        assert!(matches!(op, Operation::SpikingLayer { neurons: 100 }));
    }

    #[test]
    fn test_backend_display() {
        assert_eq!(format!("{}", BackendTarget::GpuNvidia), "GPU_NVIDIA");
        assert_eq!(format!("{}", BackendTarget::Neuromorphic), "Neuromorphic");
        assert_eq!(format!("{}", BackendTarget::Quantum), "Quantum");
    }
}
