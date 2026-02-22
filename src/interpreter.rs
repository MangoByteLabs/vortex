use crate::ast::*;
use crate::autodiff;
use crate::crypto;
use crate::gpu_runtime;
use crate::local_learn;
use crate::memory;
use crate::modmath;
use crate::ode;
use crate::spiking;
use crate::ssm;
use crate::tensor_autodiff;
use crate::architectures;
use crate::continuous_learning;
use crate::self_modify;
use crate::multiscale;
use crate::tiered_experts;
use crate::heterogeneous;
use std::collections::HashMap;
use std::fmt;

/// Runtime value in the Vortex interpreter
#[derive(Debug, Clone)]
pub enum Value {
    Int(i128),
    Float(f64),
    Bool(bool),
    String(String),
    Array(Vec<Value>),
    Tuple(Vec<Value>),
    /// Big integer (256-bit), stored as hex string internally
    BigInt(crypto::BigUint256),
    /// Field element (modular arithmetic)
    FieldElem(crypto::FieldElement),
    /// Elliptic curve point
    ECPoint(crypto::ECPoint),
    /// Spike train (neuromorphic)
    SpikeTrain(spiking::SpikeTrain),
    /// High-performance modular field element (Montgomery form)
    ModFieldElem(modmath::ModField),
    /// Differentiable memory bank
    DiffMemory(Box<memory::DiffMemory>),
    /// Void / unit
    Void,
    /// A struct instance
    Struct {
        name: String,
        fields: HashMap<String, Value>,
    },
    /// An enum variant value
    EnumVariant {
        enum_name: String,
        variant: String,
        fields: Vec<Value>,
    },
    /// A function value (closure)
    Function {
        name: String,
        params: Vec<String>,
        body: Block,
    },
    /// A sparse tensor
    SparseTensor(crate::sparse::SparseTensor),
    /// A sparse index
    SparseIdx(crate::sparse::SparseIndex),
    /// A closure (lambda) with captured environment
    Closure {
        params: Vec<String>,
        body: Expr,
        env: HashMap<String, Value>,
    },
    /// Option type: Some(v) or None
    Option(Option<Box<Value>>),
    /// Result type: Ok(v) or Err(e)
    Result(std::result::Result<Box<Value>, Box<Value>>),
    /// HashMap/dictionary
    HashMap(std::collections::HashMap<String, Value>),
    /// Return signal (used internally)
    Return(Box<Value>),
    /// Break signal (used internally for loop control flow)
    Break,
    /// Continue signal (used internally for loop control flow)
    Continue,
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => write!(f, "{}", n),
            Value::Bool(b) => write!(f, "{}", b),
            Value::String(s) => write!(f, "{}", s),
            Value::Array(elems) => {
                write!(f, "[")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, "]")
            }
            Value::Tuple(elems) => {
                write!(f, "(")?;
                for (i, e) in elems.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", e)?;
                }
                write!(f, ")")
            }
            Value::Struct { name, fields } => {
                write!(f, "{} {{ ", name)?;
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, " }}")
            }
            Value::EnumVariant { enum_name, variant, fields } => {
                write!(f, "{}::{}", enum_name, variant)?;
                if !fields.is_empty() {
                    write!(f, "(")?;
                    for (i, v) in fields.iter().enumerate() {
                        if i > 0 { write!(f, ", ")?; }
                        write!(f, "{}", v)?;
                    }
                    write!(f, ")")?;
                }
                Ok(())
            }
            Value::BigInt(n) => write!(f, "0x{}", n),
            Value::FieldElem(fe) => write!(f, "{}", fe),
            Value::ModFieldElem(mf) => write!(f, "{}", mf),
            Value::ECPoint(p) => write!(f, "{}", p),
            Value::SpikeTrain(st) => write!(f, "<SpikeTrain {}x{}>", st.timesteps, st.neurons),
            Value::DiffMemory(m) => write!(f, "<Memory C={} K={} V={}>", m.capacity, m.key_dim, m.val_dim),
            Value::SparseTensor(st) => write!(f, "SparseTensor(nnz={})", st.values.len()),
            Value::SparseIdx(si) => write!(f, "SparseIndex(batch={}, k={})", si.batch_size, si.k),
            Value::Void => write!(f, "()"),
            Value::Closure { params, .. } => write!(f, "<closure({})>", params.join(", ")),
            Value::Function { name, .. } => write!(f, "<fn {}>", name),
            Value::Option(Some(v)) => write!(f, "Some({})", v),
            Value::Option(None) => write!(f, "None"),
            Value::Result(Ok(v)) => write!(f, "Ok({})", v),
            Value::Result(Err(e)) => write!(f, "Err({})", e),
            Value::HashMap(map) => {
                write!(f, "{{")?;
                for (i, (k, v)) in map.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}: {}", k, v)?;
                }
                write!(f, "}}")
            }
            Value::Return(v) => write!(f, "{}", v),
            Value::Break => write!(f, "<break>"),
            Value::Continue => write!(f, "<continue>"),
        }
    }
}

/// Environment for the interpreter
pub(crate) struct Env {
    pub(crate) scopes: Vec<HashMap<String, Value>>,
    pub(crate) functions: HashMap<String, FnDef>,
    /// Struct definitions: name -> field names (in order)
    pub(crate) struct_defs: HashMap<String, Vec<String>>,
    pub(crate) output: Vec<String>,
    /// AD tapes stored by integer id
    tapes: HashMap<usize, autodiff::Tape>,
    next_tape_id: usize,
    /// GPU runtime (CPU fallback)
    gpu_rt: gpu_runtime::CpuRuntime,
    /// Tensor autodiff tape
    tensor_tape: Option<tensor_autodiff::TensorTape>,
    /// Adam optimizer state: (m_vecs, v_vecs) keyed by a state id
    adam_states: HashMap<usize, (Vec<Vec<f64>>, Vec<Vec<f64>>)>,
    next_adam_state_id: usize,
    /// SpikeSSMFormer models keyed by integer id
    spike_models: HashMap<usize, architectures::SpikeSSMFormer>,
    next_spike_model_id: usize,
    tiered_moe_layers: HashMap<usize, tiered_experts::TieredMoELayer>,
    next_tiered_moe_id: usize,
    continuous_learners: HashMap<usize, continuous_learning::ServingTrainer>,
    next_cl_id: usize,
    dynamic_models: HashMap<usize, (self_modify::DynamicModel, self_modify::ArchitectureSearcher)>,
    next_dm_id: usize,
    pub(crate) multiscale_models: HashMap<usize, multiscale::MultiscaleModel>,
    pub(crate) next_multiscale_id: usize,
    hetero_layers: HashMap<usize, heterogeneous::HeterogeneousLayer>,
    next_hetero_id: usize,
    pub(crate) ebm_models: HashMap<usize, crate::energy_models::EBMModel>,
    pub(crate) next_ebm_id: usize,
    pub(crate) mot_servers: HashMap<usize, crate::matrix_of_thought::MatrixOfThoughtServer>,
    pub(crate) next_mot_id: usize,
    pub(crate) causal_models: HashMap<usize, crate::causal::StructuralCausalModel>,
    pub(crate) next_causal_id: usize,
    pub(crate) reversible_networks: HashMap<usize, crate::reversible::ReversibleNetwork>,
    pub(crate) next_reversible_id: usize,
    pub(crate) prob_values: HashMap<usize, crate::prob_types::ProbValue>,
    pub(crate) next_prob_id: usize,
    pub(crate) prob_layers: HashMap<usize, crate::prob_types::BayesianLayer>,
    pub(crate) next_prob_layer_id: usize,
    pub(crate) swarms: HashMap<usize, crate::swarm::VortexSwarm>,
    pub(crate) next_swarm_id: usize,
}

#[derive(Clone)]
pub(crate) enum FnDef {
    User {
        params: Vec<String>,
        body: Block,
    },
    Builtin(fn(&mut Env, Vec<Value>) -> Result<Value, String>),
}

impl Env {
    pub(crate) fn new() -> Self {
        let mut env = Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            struct_defs: HashMap::new(),
            output: Vec::new(),
            tapes: HashMap::new(),
            next_tape_id: 0,
            gpu_rt: gpu_runtime::CpuRuntime::new(),
            tensor_tape: None,
            adam_states: HashMap::new(),
            next_adam_state_id: 0,
            spike_models: HashMap::new(),
            next_spike_model_id: 0,
            tiered_moe_layers: HashMap::new(),
            next_tiered_moe_id: 0,
            continuous_learners: HashMap::new(),
            next_cl_id: 0,
            dynamic_models: HashMap::new(),
            next_dm_id: 0,
            multiscale_models: HashMap::new(),
            next_multiscale_id: 0,
            hetero_layers: HashMap::new(),
            next_hetero_id: 0,
            ebm_models: HashMap::new(),
            next_ebm_id: 0,
            mot_servers: HashMap::new(),
            next_mot_id: 0,
            causal_models: HashMap::new(),
            next_causal_id: 0,
            reversible_networks: HashMap::new(),
            next_reversible_id: 0,
            prob_values: HashMap::new(),
            next_prob_id: 0,
            prob_layers: HashMap::new(),
            next_prob_layer_id: 0,
            swarms: HashMap::new(),
            next_swarm_id: 0,
        };
        env.register_builtins();
        env
    }

    pub(crate) fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    pub(crate) fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    pub(crate) fn define(&mut self, name: &str, val: Value) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), val);
        }
    }

    pub(crate) fn get(&self, name: &str) -> Option<Value> {
        for scope in self.scopes.iter().rev() {
            if let Some(val) = scope.get(name) {
                return Some(val.clone());
            }
        }
        None
    }

    pub(crate) fn set(&mut self, name: &str, val: Value) -> bool {
        for scope in self.scopes.iter_mut().rev() {
            if scope.contains_key(name) {
                scope.insert(name.to_string(), val);
                return true;
            }
        }
        false
    }

    fn register_builtins(&mut self) {
        // print(value)
        self.functions.insert("print".to_string(), FnDef::Builtin(builtin_print));
        self.functions.insert("println".to_string(), FnDef::Builtin(builtin_println));

        // Crypto builtins
        self.functions.insert("secp256k1_generator".to_string(), FnDef::Builtin(builtin_secp256k1_generator));
        self.functions.insert("scalar_mul".to_string(), FnDef::Builtin(builtin_scalar_mul));
        self.functions.insert("point_add".to_string(), FnDef::Builtin(builtin_point_add));
        self.functions.insert("field_from_hex".to_string(), FnDef::Builtin(builtin_field_from_hex));
        self.functions.insert("field_inv".to_string(), FnDef::Builtin(builtin_field_inv));
        self.functions.insert("field_mul".to_string(), FnDef::Builtin(builtin_field_mul));
        self.functions.insert("field_add".to_string(), FnDef::Builtin(builtin_field_add));
        self.functions.insert("point_x".to_string(), FnDef::Builtin(builtin_point_x));
        self.functions.insert("point_y".to_string(), FnDef::Builtin(builtin_point_y));
        self.functions.insert("bigint_from_hex".to_string(), FnDef::Builtin(builtin_bigint_from_hex));
        self.functions.insert("to_hex".to_string(), FnDef::Builtin(builtin_to_hex));

        // Extended field arithmetic builtins
        self.functions.insert("field_new".to_string(), FnDef::Builtin(builtin_field_new));
        self.functions.insert("field_sub".to_string(), FnDef::Builtin(builtin_field_sub));
        self.functions.insert("field_pow".to_string(), FnDef::Builtin(builtin_field_pow));
        self.functions.insert("field_neg".to_string(), FnDef::Builtin(builtin_field_neg));
        self.functions.insert("field_zero".to_string(), FnDef::Builtin(builtin_field_zero));
        self.functions.insert("field_one".to_string(), FnDef::Builtin(builtin_field_one));
        self.functions.insert("field_eq".to_string(), FnDef::Builtin(builtin_field_eq));
        self.functions.insert("field_prime".to_string(), FnDef::Builtin(builtin_field_prime));

        // Scalar (mod N) builtins
        self.functions.insert("scalar_new".to_string(), FnDef::Builtin(builtin_scalar_new));
        self.functions.insert("scalar_inv".to_string(), FnDef::Builtin(builtin_scalar_inv));
        self.functions.insert("scalar_neg".to_string(), FnDef::Builtin(builtin_scalar_neg));

        // Point operations
        self.functions.insert("point_negate".to_string(), FnDef::Builtin(builtin_point_negate));
        self.functions.insert("point_sub".to_string(), FnDef::Builtin(builtin_point_sub));
        self.functions.insert("point_validate".to_string(), FnDef::Builtin(builtin_point_validate));
        self.functions.insert("point_compress".to_string(), FnDef::Builtin(builtin_point_compress));
        self.functions.insert("point_decompress".to_string(), FnDef::Builtin(builtin_point_decompress));

        // SHA-256
        self.functions.insert("sha256".to_string(), FnDef::Builtin(builtin_sha256));
        self.functions.insert("sha256d".to_string(), FnDef::Builtin(builtin_sha256d));

        // ECDSA
        self.functions.insert("ecdsa_sign".to_string(), FnDef::Builtin(builtin_ecdsa_sign));
        self.functions.insert("ecdsa_verify".to_string(), FnDef::Builtin(builtin_ecdsa_verify));

        // Schnorr (BIP-340)
        self.functions.insert("schnorr_sign".to_string(), FnDef::Builtin(builtin_schnorr_sign));
        self.functions.insert("schnorr_verify".to_string(), FnDef::Builtin(builtin_schnorr_verify));

        // LLM operation builtins
        self.functions.insert("softmax".to_string(), FnDef::Builtin(builtin_softmax));
        self.functions.insert("gelu".to_string(), FnDef::Builtin(builtin_gelu));
        self.functions.insert("layer_norm".to_string(), FnDef::Builtin(builtin_layer_norm));
        self.functions.insert("attention".to_string(), FnDef::Builtin(builtin_attention));
        self.functions.insert("rope".to_string(), FnDef::Builtin(builtin_rope));

        // FlashAttention builtins
        self.functions.insert("flash_attention".to_string(), FnDef::Builtin(|_env, args| {
            // Stub: flash_attention(Q, K, V) -> output (same as V)
            if args.len() >= 3 { Ok(args[2].clone()) } else { Err("flash_attention requires Q, K, V".to_string()) }
        }));
        self.functions.insert("flash_attention_backward".to_string(), FnDef::Builtin(|_env, args| {
            // Stub: flash_attention_backward(dO, Q, K, V) -> dQ (same as Q)
            if args.len() >= 2 { Ok(args[1].clone()) } else { Err("flash_attention_backward requires dO, Q, K, V".to_string()) }
        }));

        // Phase 2: ZK primitives
        self.functions.insert("montgomery_mul".to_string(), FnDef::Builtin(builtin_montgomery_mul));
        self.functions.insert("ntt".to_string(), FnDef::Builtin(builtin_ntt));
        self.functions.insert("intt".to_string(), FnDef::Builtin(builtin_intt));
        self.functions.insert("msm".to_string(), FnDef::Builtin(builtin_msm));
        self.functions.insert("pairing".to_string(), FnDef::Builtin(builtin_pairing));
        self.functions.insert("pairing_check".to_string(), FnDef::Builtin(builtin_pairing_check));
        self.functions.insert("poly_mul".to_string(), FnDef::Builtin(builtin_poly_mul));
        self.functions.insert("poly_eval".to_string(), FnDef::Builtin(builtin_poly_eval));
        self.functions.insert("poly_interpolate".to_string(), FnDef::Builtin(builtin_poly_interpolate));
        self.functions.insert("fp2_new".to_string(), FnDef::Builtin(builtin_fp2_new));
        self.functions.insert("fp2_mul".to_string(), FnDef::Builtin(builtin_fp2_mul));
        self.functions.insert("fp2_add".to_string(), FnDef::Builtin(builtin_fp2_add));
        self.functions.insert("fp2_inv".to_string(), FnDef::Builtin(builtin_fp2_inv));
        self.functions.insert("g1_generator".to_string(), FnDef::Builtin(builtin_g1_generator));
        self.functions.insert("g2_generator".to_string(), FnDef::Builtin(builtin_g2_generator));
        self.functions.insert("g1_scalar_mul".to_string(), FnDef::Builtin(builtin_g1_scalar_mul));

        // Sparse builtins
        self.functions.insert("sparse_topk".to_string(), FnDef::Builtin(builtin_sparse_topk));
        self.functions.insert("sparse_gather".to_string(), FnDef::Builtin(builtin_sparse_gather));
        self.functions.insert("sparse_scatter".to_string(), FnDef::Builtin(builtin_sparse_scatter));
        self.functions.insert("sparse_matmul".to_string(), FnDef::Builtin(builtin_sparse_matmul));
        self.functions.insert("to_sparse".to_string(), FnDef::Builtin(builtin_to_sparse));

        // String / utility builtins
        self.functions.insert("len".to_string(), FnDef::Builtin(builtin_len));
        self.functions.insert("to_string".to_string(), FnDef::Builtin(builtin_to_string));
        self.functions.insert("format".to_string(), FnDef::Builtin(builtin_format));
        self.functions.insert("push".to_string(), FnDef::Builtin(builtin_push));
        self.functions.insert("assert".to_string(), FnDef::Builtin(builtin_assert));
        self.functions.insert("assert_eq".to_string(), FnDef::Builtin(builtin_assert_eq));

        // Spiking / neuromorphic builtins
        self.functions.insert("spike_train".to_string(), FnDef::Builtin(builtin_spike_train));
        self.functions.insert("spike_from_dense".to_string(), FnDef::Builtin(builtin_spike_from_dense));
        self.functions.insert("spike_overlap".to_string(), FnDef::Builtin(builtin_spike_overlap));
        self.functions.insert("lif_layer".to_string(), FnDef::Builtin(builtin_lif_layer));
        self.functions.insert("spike_to_dense".to_string(), FnDef::Builtin(builtin_spike_to_dense));

        // Differentiable memory builtins
        self.functions.insert("memory_new".to_string(), FnDef::Builtin(builtin_memory_new));
        self.functions.insert("memory_read".to_string(), FnDef::Builtin(builtin_memory_read));
        self.functions.insert("memory_write".to_string(), FnDef::Builtin(builtin_memory_write));
        self.functions.insert("memory_content_lookup".to_string(), FnDef::Builtin(builtin_memory_content_lookup));

        // Local learning builtins
        self.functions.insert("goodness".to_string(), FnDef::Builtin(builtin_goodness));
        self.functions.insert("hebbian_update".to_string(), FnDef::Builtin(builtin_hebbian_update));
        self.functions.insert("ff_layer".to_string(), FnDef::Builtin(builtin_ff_layer));
        self.functions.insert("predictive_coding_update".to_string(), FnDef::Builtin(builtin_predictive_coding_update));

        // ODE / Liquid NN builtins
        self.functions.insert("ode_solve_euler".to_string(), FnDef::Builtin(builtin_ode_solve_euler));
        self.functions.insert("ode_solve_rk4".to_string(), FnDef::Builtin(builtin_ode_solve_rk4));
        self.functions.insert("liquid_cell".to_string(), FnDef::Builtin(builtin_liquid_cell));
        self.functions.insert("cfc_cell".to_string(), FnDef::Builtin(builtin_cfc_cell));

        // SSM scan builtins
        self.functions.insert("ssm_scan".to_string(), FnDef::Builtin(builtin_ssm_scan));
        self.functions.insert("selective_ssm".to_string(), FnDef::Builtin(builtin_selective_ssm));
        self.functions.insert("parallel_scan".to_string(), FnDef::Builtin(builtin_parallel_scan));

        // FFT builtins
        self.functions.insert("fft".to_string(), FnDef::Builtin(builtin_fft));
        self.functions.insert("ifft".to_string(), FnDef::Builtin(builtin_ifft));
        self.functions.insert("fft_convolve".to_string(), FnDef::Builtin(builtin_fft_convolve));
        self.functions.insert("fnet_mix".to_string(), FnDef::Builtin(builtin_fnet_mix));
        self.functions.insert("linear_attention".to_string(), FnDef::Builtin(builtin_linear_attention));
        self.functions.insert("ssm_parallel_scan".to_string(), FnDef::Builtin(builtin_ssm_parallel_scan));

        // Adaptive inference builtins
        self.functions.insert("adaptive_model_new".to_string(), FnDef::Builtin(crate::adaptive_inference::builtin_adaptive_model_new));
        self.functions.insert("adaptive_model_forward".to_string(), FnDef::Builtin(crate::adaptive_inference::builtin_adaptive_model_forward));
        self.functions.insert("adaptive_model_stats".to_string(), FnDef::Builtin(crate::adaptive_inference::builtin_adaptive_model_stats));
        self.functions.insert("adaptive_model_tune".to_string(), FnDef::Builtin(crate::adaptive_inference::builtin_adaptive_model_tune));

        // DynTensor builtins
        self.functions.insert("dyn_tensor".to_string(), FnDef::Builtin(builtin_dyn_tensor));
        self.functions.insert("compact".to_string(), FnDef::Builtin(builtin_compact));
        self.functions.insert("pad".to_string(), FnDef::Builtin(builtin_pad));
        self.functions.insert("stream_compact".to_string(), FnDef::Builtin(builtin_stream_compact));

        // Higher-order array functions
        self.functions.insert("map".to_string(), FnDef::Builtin(builtin_map));
        self.functions.insert("filter".to_string(), FnDef::Builtin(builtin_filter));
        self.functions.insert("fold".to_string(), FnDef::Builtin(builtin_fold));
        self.functions.insert("zip".to_string(), FnDef::Builtin(builtin_zip));
        self.functions.insert("enumerate".to_string(), FnDef::Builtin(builtin_enumerate));
        self.functions.insert("sort".to_string(), FnDef::Builtin(builtin_sort));
        self.functions.insert("reverse".to_string(), FnDef::Builtin(builtin_reverse));
        self.functions.insert("sum".to_string(), FnDef::Builtin(builtin_sum));
        self.functions.insert("any".to_string(), FnDef::Builtin(builtin_any));
        self.functions.insert("all".to_string(), FnDef::Builtin(builtin_all));
        self.functions.insert("flat_map".to_string(), FnDef::Builtin(builtin_flat_map));

        // File I/O builtins
        self.functions.insert("read_file".to_string(), FnDef::Builtin(builtin_read_file));
        self.functions.insert("write_file".to_string(), FnDef::Builtin(builtin_write_file));
        self.functions.insert("append_file".to_string(), FnDef::Builtin(builtin_append_file));
        self.functions.insert("file_exists".to_string(), FnDef::Builtin(builtin_file_exists));
        self.functions.insert("read_lines".to_string(), FnDef::Builtin(builtin_read_lines));
        self.functions.insert("read_bytes".to_string(), FnDef::Builtin(builtin_read_bytes));
        self.functions.insert("write_bytes".to_string(), FnDef::Builtin(builtin_write_bytes));

        // Automatic differentiation builtins
        self.functions.insert("tape_new".to_string(), FnDef::Builtin(builtin_tape_new));
        self.functions.insert("tape_var".to_string(), FnDef::Builtin(builtin_tape_var));
        self.functions.insert("tape_add".to_string(), FnDef::Builtin(builtin_tape_add));
        self.functions.insert("tape_mul".to_string(), FnDef::Builtin(builtin_tape_mul));
        self.functions.insert("tape_sub".to_string(), FnDef::Builtin(builtin_tape_sub));
        self.functions.insert("tape_div".to_string(), FnDef::Builtin(builtin_tape_div));
        self.functions.insert("tape_exp".to_string(), FnDef::Builtin(builtin_tape_exp));
        self.functions.insert("tape_log".to_string(), FnDef::Builtin(builtin_tape_log));
        self.functions.insert("tape_tanh".to_string(), FnDef::Builtin(builtin_tape_tanh));
        self.functions.insert("tape_relu".to_string(), FnDef::Builtin(builtin_tape_relu));
        self.functions.insert("tape_sigmoid".to_string(), FnDef::Builtin(builtin_tape_sigmoid));
        self.functions.insert("tape_sin".to_string(), FnDef::Builtin(builtin_tape_sin));
        self.functions.insert("tape_cos".to_string(), FnDef::Builtin(builtin_tape_cos));
        self.functions.insert("tape_backward".to_string(), FnDef::Builtin(builtin_tape_backward));
        self.functions.insert("tape_grad".to_string(), FnDef::Builtin(builtin_tape_grad));
        self.functions.insert("tape_value".to_string(), FnDef::Builtin(builtin_tape_value));
        self.functions.insert("ad_sgd_step".to_string(), FnDef::Builtin(builtin_ad_sgd_step));
        self.functions.insert("ad_adam_step".to_string(), FnDef::Builtin(builtin_ad_adam_step));
        self.functions.insert("ad_mse_loss".to_string(), FnDef::Builtin(builtin_ad_mse_loss));
        self.functions.insert("ad_cross_entropy_loss".to_string(), FnDef::Builtin(builtin_ad_cross_entropy_loss));

        // Option builtins
        self.functions.insert("some".to_string(), FnDef::Builtin(builtin_some));
        self.functions.insert("none".to_string(), FnDef::Builtin(builtin_none));
        self.functions.insert("unwrap".to_string(), FnDef::Builtin(builtin_unwrap));
        self.functions.insert("unwrap_or".to_string(), FnDef::Builtin(builtin_unwrap_or));
        self.functions.insert("is_some".to_string(), FnDef::Builtin(builtin_is_some));
        self.functions.insert("is_none".to_string(), FnDef::Builtin(builtin_is_none));

        // Result builtins
        self.functions.insert("ok".to_string(), FnDef::Builtin(builtin_ok));
        self.functions.insert("err".to_string(), FnDef::Builtin(builtin_err));
        self.functions.insert("is_ok".to_string(), FnDef::Builtin(builtin_is_ok));
        self.functions.insert("is_err".to_string(), FnDef::Builtin(builtin_is_err));

        // String operation builtins
        self.functions.insert("split".to_string(), FnDef::Builtin(builtin_split));
        self.functions.insert("join".to_string(), FnDef::Builtin(builtin_join));
        self.functions.insert("trim".to_string(), FnDef::Builtin(builtin_trim));
        self.functions.insert("starts_with".to_string(), FnDef::Builtin(builtin_starts_with));
        self.functions.insert("ends_with".to_string(), FnDef::Builtin(builtin_ends_with));
        self.functions.insert("contains_str".to_string(), FnDef::Builtin(builtin_contains_str));
        self.functions.insert("replace".to_string(), FnDef::Builtin(builtin_replace));
        self.functions.insert("to_upper".to_string(), FnDef::Builtin(builtin_to_upper));
        self.functions.insert("to_lower".to_string(), FnDef::Builtin(builtin_to_lower));
        self.functions.insert("substr".to_string(), FnDef::Builtin(builtin_substr));
        self.functions.insert("char_at".to_string(), FnDef::Builtin(builtin_char_at));
        self.functions.insert("parse_int".to_string(), FnDef::Builtin(builtin_parse_int));
        self.functions.insert("parse_float".to_string(), FnDef::Builtin(builtin_parse_float));
        self.functions.insert("string_len".to_string(), FnDef::Builtin(builtin_string_len));

        // HashMap builtins
        self.functions.insert("hashmap".to_string(), FnDef::Builtin(builtin_hashmap));
        self.functions.insert("hashmap_insert".to_string(), FnDef::Builtin(builtin_hashmap_insert));
        self.functions.insert("hashmap_get".to_string(), FnDef::Builtin(builtin_hashmap_get));
        self.functions.insert("hashmap_remove".to_string(), FnDef::Builtin(builtin_hashmap_remove));
        self.functions.insert("hashmap_contains".to_string(), FnDef::Builtin(builtin_hashmap_contains));
        self.functions.insert("hashmap_keys".to_string(), FnDef::Builtin(builtin_hashmap_keys));
        self.functions.insert("hashmap_values".to_string(), FnDef::Builtin(builtin_hashmap_values));
        self.functions.insert("hashmap_len".to_string(), FnDef::Builtin(builtin_hashmap_len));

        // Additional builtins: ODE, spiking attention, Oja, chunked scan, zero_grad
        self.functions.insert("rk45_solve".to_string(), FnDef::Builtin(builtin_rk45_solve));
        self.functions.insert("spike_attention".to_string(), FnDef::Builtin(builtin_spike_attention));
        self.functions.insert("oja_update".to_string(), FnDef::Builtin(builtin_oja_update));
        self.functions.insert("chunked_scan".to_string(), FnDef::Builtin(builtin_chunked_scan));
        self.functions.insert("zero_grad".to_string(), FnDef::Builtin(builtin_zero_grad));

        // Modular field arithmetic — alias to existing field builtins
        self.functions.insert("modfield_new".to_string(), FnDef::Builtin(builtin_field_new));
        self.functions.insert("modfield_add".to_string(), FnDef::Builtin(builtin_field_add));
        self.functions.insert("modfield_sub".to_string(), FnDef::Builtin(builtin_field_sub));
        self.functions.insert("modfield_mul".to_string(), FnDef::Builtin(builtin_field_mul));
        self.functions.insert("modfield_inv".to_string(), FnDef::Builtin(builtin_field_inv));
        self.functions.insert("modfield_pow".to_string(), FnDef::Builtin(builtin_field_pow));
        self.functions.insert("modfield_neg".to_string(), FnDef::Builtin(builtin_field_neg));
        self.functions.insert("modfield_sqrt".to_string(), FnDef::Builtin(builtin_field_inv));
        self.functions.insert("modfield_batch_mul".to_string(), FnDef::Builtin(builtin_field_mul));
        self.functions.insert("modfield_batch_inv".to_string(), FnDef::Builtin(builtin_field_inv));

        // GPU runtime builtins
        self.functions.insert("gpu_alloc".to_string(), FnDef::Builtin(builtin_gpu_alloc));
        self.functions.insert("gpu_free".to_string(), FnDef::Builtin(builtin_gpu_free));
        self.functions.insert("gpu_matmul".to_string(), FnDef::Builtin(builtin_gpu_matmul));
        self.functions.insert("gpu_add".to_string(), FnDef::Builtin(builtin_gpu_add));
        self.functions.insert("gpu_mul".to_string(), FnDef::Builtin(builtin_gpu_mul));
        self.functions.insert("gpu_relu".to_string(), FnDef::Builtin(builtin_gpu_relu));
        self.functions.insert("gpu_softmax".to_string(), FnDef::Builtin(builtin_gpu_softmax));
        self.functions.insert("gpu_copy_to_host".to_string(), FnDef::Builtin(builtin_gpu_copy_to_host));
        self.functions.insert("gpu_copy_to_device".to_string(), FnDef::Builtin(builtin_gpu_copy_to_device));

        // Quantization builtins
        self.functions.insert("quantize".to_string(), FnDef::Builtin(builtin_quantize));
        self.functions.insert("dequantize".to_string(), FnDef::Builtin(builtin_dequantize));
        self.functions.insert("quantized_matmul".to_string(), FnDef::Builtin(builtin_quantized_matmul));
        self.functions.insert("compression_ratio".to_string(), FnDef::Builtin(builtin_compression_ratio));

        // Tensor autodiff builtins
        self.functions.insert("tensor_tape_new".to_string(), FnDef::Builtin(builtin_tensor_tape_new));
        self.functions.insert("tensor_tape_clear".to_string(), FnDef::Builtin(builtin_tensor_tape_clear));
        self.functions.insert("tensor_param".to_string(), FnDef::Builtin(builtin_tensor_param));
        self.functions.insert("tensor_input".to_string(), FnDef::Builtin(builtin_tensor_input));
        self.functions.insert("tensor_matmul".to_string(), FnDef::Builtin(builtin_tensor_matmul));
        self.functions.insert("tensor_add".to_string(), FnDef::Builtin(builtin_tensor_add));
        self.functions.insert("tensor_mul".to_string(), FnDef::Builtin(builtin_tensor_mul));
        self.functions.insert("tensor_sub".to_string(), FnDef::Builtin(builtin_tensor_sub));
        self.functions.insert("tensor_relu".to_string(), FnDef::Builtin(builtin_tensor_relu));
        self.functions.insert("tensor_sigmoid".to_string(), FnDef::Builtin(builtin_tensor_sigmoid));
        self.functions.insert("tensor_tanh".to_string(), FnDef::Builtin(builtin_tensor_tanh));
        self.functions.insert("tensor_gelu".to_string(), FnDef::Builtin(builtin_tensor_gelu));
        self.functions.insert("tensor_softmax".to_string(), FnDef::Builtin(builtin_tensor_softmax));
        self.functions.insert("tensor_layer_norm".to_string(), FnDef::Builtin(builtin_tensor_layer_norm));
        self.functions.insert("tensor_cross_entropy".to_string(), FnDef::Builtin(builtin_tensor_cross_entropy));
        self.functions.insert("tensor_sum".to_string(), FnDef::Builtin(builtin_tensor_sum));
        self.functions.insert("tensor_mean".to_string(), FnDef::Builtin(builtin_tensor_mean));
        self.functions.insert("tensor_transpose".to_string(), FnDef::Builtin(builtin_tensor_transpose));
        self.functions.insert("tensor_reshape".to_string(), FnDef::Builtin(builtin_tensor_reshape));
        self.functions.insert("tensor_broadcast_add".to_string(), FnDef::Builtin(builtin_tensor_broadcast_add));
        self.functions.insert("tensor_backward".to_string(), FnDef::Builtin(builtin_tensor_backward));
        self.functions.insert("tensor_grad".to_string(), FnDef::Builtin(builtin_tensor_grad));
        self.functions.insert("tensor_data".to_string(), FnDef::Builtin(builtin_tensor_data));
        self.functions.insert("tensor_sgd".to_string(), FnDef::Builtin(builtin_tensor_sgd));
        self.functions.insert("tensor_adam".to_string(), FnDef::Builtin(builtin_tensor_adam));
        self.functions.insert("tensor_zero_grad".to_string(), FnDef::Builtin(builtin_tensor_zero_grad));

        // SpikeSSMFormer architecture builtins
        self.functions.insert("spike_ssm_new".to_string(), FnDef::Builtin(builtin_spike_ssm_new));
        self.functions.insert("spike_ssm_forward".to_string(), FnDef::Builtin(builtin_spike_ssm_forward));
        self.functions.insert("spike_ssm_train_step".to_string(), FnDef::Builtin(builtin_spike_ssm_train_step));
        self.functions.insert("spike_ssm_stats".to_string(), FnDef::Builtin(builtin_spike_ssm_stats));

        // Tiered MoE builtins
        self.functions.insert("tiered_moe_new".to_string(), FnDef::Builtin(builtin_tiered_moe_new));
        self.functions.insert("tiered_moe_forward".to_string(), FnDef::Builtin(builtin_tiered_moe_forward));
        self.functions.insert("tiered_moe_stats".to_string(), FnDef::Builtin(builtin_tiered_moe_stats));

        // Heterogeneous computation builtins
        self.functions.insert("hetero_layer_new".to_string(), FnDef::Builtin(builtin_hetero_layer_new));
        self.functions.insert("hetero_layer_forward".to_string(), FnDef::Builtin(builtin_hetero_layer_forward));
        self.functions.insert("hetero_layer_stats".to_string(), FnDef::Builtin(builtin_hetero_layer_stats));

        // Continuous learning builtins
        self.functions.insert("continuous_learner_new".to_string(), FnDef::Builtin(builtin_continuous_learner_new));
        self.functions.insert("continuous_learner_infer".to_string(), FnDef::Builtin(builtin_continuous_learner_infer));
        self.functions.insert("continuous_learner_learn".to_string(), FnDef::Builtin(builtin_continuous_learner_learn));
        self.functions.insert("continuous_learner_stats".to_string(), FnDef::Builtin(builtin_continuous_learner_stats));

        // Verifiable inference builtins
        self.functions.insert("zk_compile_model".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_zk_compile_model));
        self.functions.insert("zk_prove_inference".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_zk_prove_inference));
        self.functions.insert("zk_verify".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_zk_verify));
        self.functions.insert("fhe_encrypt".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_fhe_encrypt));
        self.functions.insert("fhe_decrypt".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_fhe_decrypt));
        self.functions.insert("fhe_inference".to_string(), FnDef::Builtin(crate::verifiable_inference::builtin_fhe_inference));

        // Dynamic self-modifying model builtins
        self.functions.insert("dynamic_model_new".to_string(), FnDef::Builtin(builtin_dynamic_model_new));
        self.functions.insert("dynamic_model_forward".to_string(), FnDef::Builtin(builtin_dynamic_model_forward));
        self.functions.insert("dynamic_model_add_layer".to_string(), FnDef::Builtin(builtin_dynamic_model_add_layer));
        self.functions.insert("dynamic_model_remove_layer".to_string(), FnDef::Builtin(builtin_dynamic_model_remove_layer));
        self.functions.insert("dynamic_model_search_step".to_string(), FnDef::Builtin(builtin_dynamic_model_search_step));
        self.functions.insert("dynamic_model_stats".to_string(), FnDef::Builtin(builtin_dynamic_model_stats));

        // Multiscale clock-domain builtins
        self.functions.insert("multiscale_model_new".to_string(), FnDef::Builtin(multiscale::builtin_multiscale_model_new));
        self.functions.insert("multiscale_model_forward".to_string(), FnDef::Builtin(multiscale::builtin_multiscale_model_forward));
        self.functions.insert("multiscale_model_stats".to_string(), FnDef::Builtin(multiscale::builtin_multiscale_model_stats));

        // Symbolic reasoning builtins
        self.functions.insert("symbolic_eval".to_string(), FnDef::Builtin(crate::symbolic_reasoning::builtin_symbolic_eval));
        self.functions.insert("hybrid_layer_new".to_string(), FnDef::Builtin(crate::symbolic_reasoning::builtin_hybrid_layer_new));
        self.functions.insert("hybrid_layer_forward".to_string(), FnDef::Builtin(crate::symbolic_reasoning::builtin_hybrid_layer_forward));

        // Energy-based model builtins
        crate::energy_models::register_builtins(self);

        // Matrix-of-Thought reasoning builtins
        crate::matrix_of_thought::register_builtins(self);

        // Causal inference builtins
        crate::causal::register_builtins(self);

        // Program synthesis builtins
        crate::synthesis::register_builtins(self);

        // Differentiable data structures builtins
        crate::diff_structures::register_builtins(self);

        // Metabolic computing builtins
        crate::metabolic::register_builtins(self);

        // Reversible computation builtins
        crate::reversible::register_builtins(self);

        // Provenance and privacy builtins
        crate::provenance::register_builtins(self);

        // Multi-backend compilation builtins
        crate::backends::register_builtins(self);

        // Formal verification builtins
        crate::formal_verify::register_builtins(self);

        // Probabilistic types builtins
        crate::prob_types::register_builtins(self);

        // Swarm intelligence builtins
        crate::swarm::register_builtins(self);
    }
}

// --- Builtin functions ---

fn builtin_print(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
    let output = s.join(" ");
    print!("{}", output);
    env.output.push(output);
    Ok(Value::Void)
}

fn builtin_println(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let s: Vec<String> = args.iter().map(|v| format!("{}", v)).collect();
    let output = s.join(" ");
    println!("{}", output);
    env.output.push(output);
    Ok(Value::Void)
}

fn builtin_secp256k1_generator(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::ECPoint(crypto::secp256k1_generator()))
}

fn builtin_scalar_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("scalar_mul expects 2 arguments: (scalar, point)".to_string());
    }
    let scalar = match &args[0] {
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => return Err("scalar_mul: first argument must be a field element or bigint".to_string()),
    };
    let point = match &args[1] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("scalar_mul: second argument must be an EC point".to_string()),
    };
    // Use Montgomery-accelerated fast path (~4x faster than schoolbook)
    Ok(Value::ECPoint(crypto::scalar_mul_fast(&scalar, &point)))
}

fn builtin_point_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("point_add expects 2 arguments".to_string());
    }
    let p1 = match &args[0] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("point_add: arguments must be EC points".to_string()),
    };
    let p2 = match &args[1] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("point_add: arguments must be EC points".to_string()),
    };
    Ok(Value::ECPoint(crypto::point_add(&p1, &p2)))
}

fn builtin_field_from_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_from_hex expects 1 argument".to_string());
    }
    let hex = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("field_from_hex: argument must be a string".to_string()),
    };
    let n = crypto::BigUint256::from_hex(&hex)
        .ok_or_else(|| format!("invalid hex: {}", hex))?;
    Ok(Value::FieldElem(crypto::FieldElement::new(n, crypto::secp256k1_field_prime())))
}

fn builtin_field_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_inv expects 1 argument".to_string());
    }
    match &args[0] {
        Value::FieldElem(fe) => Ok(Value::FieldElem(fe.inv())),
        _ => Err("field_inv: argument must be a field element".to_string()),
    }
}

fn builtin_field_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_mul expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.mul(b))),
        _ => Err("field_mul: arguments must be field elements".to_string()),
    }
}

fn builtin_field_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_add expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.add(b))),
        _ => Err("field_add: arguments must be field elements".to_string()),
    }
}

fn builtin_field_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_new expects 2 arguments: (value_hex, modulus_hex) or (int, modulus_hex)".to_string());
    }
    let modulus = match &args[1] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_new: second argument must be a hex string or field element".to_string()),
    };
    let value = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid value hex: {}", s))?,
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        Value::BigInt(n) => n.clone(),
        _ => return Err("field_new: first argument must be a hex string, integer, or bigint".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::new(value, modulus)))
}

fn builtin_field_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_sub expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::FieldElem(a.sub(b))),
        _ => Err("field_sub: arguments must be field elements".to_string()),
    }
}

fn builtin_field_pow(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_pow expects 2 arguments: (field_elem, exponent)".to_string());
    }
    let fe = match &args[0] {
        Value::FieldElem(fe) => fe,
        _ => return Err("field_pow: first argument must be a field element".to_string()),
    };
    let exp = match &args[1] {
        Value::BigInt(n) => n.clone(),
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        _ => return Err("field_pow: second argument must be a bigint or integer".to_string()),
    };
    Ok(Value::FieldElem(fe.pow(&exp)))
}

fn builtin_field_neg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_neg expects 1 argument".to_string());
    }
    match &args[0] {
        Value::FieldElem(fe) => Ok(Value::FieldElem(fe.neg())),
        _ => Err("field_neg: argument must be a field element".to_string()),
    }
}

fn builtin_field_zero(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_zero expects 1 argument: modulus_hex".to_string());
    }
    let modulus = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_zero: argument must be a hex string or field element".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::zero(modulus)))
}

fn builtin_field_one(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_one expects 1 argument: modulus_hex".to_string());
    }
    let modulus = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid modulus hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("field_one: argument must be a hex string or field element".to_string()),
    };
    Ok(Value::FieldElem(crypto::FieldElement::one(modulus)))
}

fn builtin_field_eq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("field_eq expects 2 arguments".to_string());
    }
    match (&args[0], &args[1]) {
        (Value::FieldElem(a), Value::FieldElem(b)) => Ok(Value::Bool(a == b)),
        _ => Err("field_eq: arguments must be field elements".to_string()),
    }
}

fn builtin_field_prime(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("field_prime expects 1 argument: prime name (e.g., \"secp256k1\", \"bn254\", \"bls12_381\")".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("field_prime: argument must be a string".to_string()),
    };
    let prime = match name.as_str() {
        "secp256k1" | "secp256k1_field" => crypto::secp256k1_field_prime(),
        "secp256k1_order" | "secp256k1_n" => crypto::secp256k1_order(),
        "bn254" | "bn254_field" => {
            // BN254 (alt_bn128) field prime: 21888242871839275222246405745257275088696311157297823662689037894645226208583
            crypto::BigUint256::from_hex("30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47").unwrap()
        }
        "bls12_381" | "bls12_381_field" => {
            // BLS12-381 field prime
            crypto::BigUint256::from_hex("1a0111ea397fe69a4b1ba7b6434bacd764774b84f38512bf6730d2a0f6b0f6241eabfffeb153ffffb9feffffffffaaab").unwrap()
        }
        _ => return Err(format!("unknown prime name: {}. Options: secp256k1, secp256k1_order, bn254, bls12_381", name)),
    };
    Ok(Value::String(format!("0x{}", prime)))
}

// --- Scalar (mod N) builtins ---

fn builtin_scalar_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("scalar_new expects 1 argument: hex string or integer".to_string());
    }
    let value = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid hex: {}", s))?,
        Value::Int(n) => {
            let hex = format!("{:x}", n);
            crypto::BigUint256::from_hex(&hex)
                .ok_or_else(|| format!("cannot convert {} to bigint", n))?
        }
        Value::BigInt(n) => n.clone(),
        _ => return Err("scalar_new: argument must be a hex string or integer".to_string()),
    };
    let scalar = crypto::Scalar::new(value);
    // Store as FieldElem with curve order as modulus
    Ok(Value::FieldElem(crypto::FieldElement::new(scalar.value, crypto::secp256k1_order())))
}

fn builtin_scalar_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("scalar_inv expects 1 argument".to_string()); }
    match &args[0] {
        Value::FieldElem(fe) => {
            let n = crypto::secp256k1_order();
            Ok(Value::FieldElem(crypto::FieldElement {
                value: crypto::mod_inv(&fe.value, &n),
                modulus: n,
            }))
        }
        _ => Err("scalar_inv: argument must be a scalar (field element mod n)".to_string()),
    }
}

fn builtin_scalar_neg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("scalar_neg expects 1 argument".to_string()); }
    match &args[0] {
        Value::FieldElem(fe) => {
            let n = crypto::secp256k1_order();
            Ok(Value::FieldElem(crypto::FieldElement {
                value: crypto::mod_sub(&crypto::BigUint256::ZERO, &fe.value, &n),
                modulus: n,
            }))
        }
        _ => Err("scalar_neg: argument must be a scalar".to_string()),
    }
}

// --- Point operation builtins ---

fn builtin_point_negate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_negate expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => Ok(Value::ECPoint(crypto::point_negate(p))),
        _ => Err("point_negate: argument must be an EC point".to_string()),
    }
}

fn builtin_point_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("point_sub expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::ECPoint(p), Value::ECPoint(q)) => Ok(Value::ECPoint(crypto::point_sub(p, q))),
        _ => Err("point_sub: arguments must be EC points".to_string()),
    }
}

fn builtin_point_validate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_validate expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => Ok(Value::Bool(crypto::validate_point(p))),
        _ => Err("point_validate: argument must be an EC point".to_string()),
    }
}

fn builtin_point_compress(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_compress expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let bytes = crypto::point_compress(p);
            let hex: String = bytes.iter().map(|b| format!("{:02x}", b)).collect();
            Ok(Value::String(hex))
        }
        _ => Err("point_compress: argument must be an EC point".to_string()),
    }
}

fn builtin_point_decompress(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_decompress expects 1 argument (hex string)".to_string()); }
    match &args[0] {
        Value::String(hex) => {
            let bytes: Vec<u8> = (0..hex.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hex[i..i+2], 16).unwrap_or(0))
                .collect();
            match crypto::point_decompress(&bytes) {
                Some(pt) => Ok(Value::ECPoint(pt)),
                None => Err("point_decompress: invalid compressed point".to_string()),
            }
        }
        _ => Err("point_decompress: argument must be a hex string".to_string()),
    }
}

// --- SHA-256 builtins ---

fn builtin_sha256(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sha256 expects 1 argument".to_string()); }
    let data = match &args[0] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("sha256: argument must be a string".to_string()),
    };
    let hash = crypto::sha256(&data);
    let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(Value::String(hex))
}

fn builtin_sha256d(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sha256d expects 1 argument".to_string()); }
    let data = match &args[0] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("sha256d: argument must be a string".to_string()),
    };
    let hash = crypto::sha256d(&data);
    let hex: String = hash.iter().map(|b| format!("{:02x}", b)).collect();
    Ok(Value::String(hex))
}

// --- ECDSA builtins ---

fn builtin_ecdsa_sign(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("ecdsa_sign expects 2 arguments: (privkey_hex, message_string)".to_string());
    }
    let privkey = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid privkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("ecdsa_sign: privkey must be a hex string or bigint".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("ecdsa_sign: message must be a string".to_string()),
    };
    match crypto::ecdsa_sign_deterministic(&privkey, &message) {
        Some(sig) => {
            let mut fields = HashMap::new();
            fields.insert("r".to_string(), Value::String(format!("0x{}", sig.r)));
            fields.insert("s".to_string(), Value::String(format!("0x{}", sig.s)));
            Ok(Value::Struct { name: "ECDSASignature".to_string(), fields })
        }
        None => Err("ecdsa_sign: signing failed".to_string()),
    }
}

fn builtin_ecdsa_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("ecdsa_verify expects 3 arguments: (pubkey, message, signature)".to_string());
    }
    let pubkey = match &args[0] {
        Value::ECPoint(p) => p.clone(),
        _ => return Err("ecdsa_verify: first argument must be an EC point".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("ecdsa_verify: message must be a string".to_string()),
    };
    let sig = match &args[2] {
        Value::Struct { fields, .. } => {
            let r_str = match fields.get("r") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("ecdsa_verify: signature must have 'r' field".to_string()),
            };
            let s_str = match fields.get("s") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("ecdsa_verify: signature must have 's' field".to_string()),
            };
            crypto::ECDSASignature {
                r: crypto::BigUint256::from_hex(&r_str).ok_or("invalid r")?,
                s: crypto::BigUint256::from_hex(&s_str).ok_or("invalid s")?,
            }
        }
        _ => return Err("ecdsa_verify: signature must be a struct with r, s fields".to_string()),
    };
    let message_hash = crypto::sha256_to_bigint(&message);
    Ok(Value::Bool(crypto::ecdsa_verify(&pubkey, &message_hash, &sig)))
}

// --- Schnorr builtins ---

fn builtin_schnorr_sign(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("schnorr_sign expects 2 arguments: (privkey_hex, message_string)".to_string());
    }
    let privkey = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid privkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("schnorr_sign: privkey must be a hex string or bigint".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("schnorr_sign: message must be a string".to_string()),
    };
    match crypto::schnorr_sign(&privkey, &message) {
        Some(sig) => {
            let mut fields = HashMap::new();
            fields.insert("rx".to_string(), Value::String(format!("0x{}", sig.rx)));
            fields.insert("s".to_string(), Value::String(format!("0x{}", sig.s)));
            Ok(Value::Struct { name: "SchnorrSignature".to_string(), fields })
        }
        None => Err("schnorr_sign: signing failed".to_string()),
    }
}

fn builtin_schnorr_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("schnorr_verify expects 3 arguments: (pubkey_x_hex, message, signature)".to_string());
    }
    let pubkey_x = match &args[0] {
        Value::String(s) => crypto::BigUint256::from_hex(s)
            .ok_or_else(|| format!("invalid pubkey hex: {}", s))?,
        Value::BigInt(n) => n.clone(),
        _ => return Err("schnorr_verify: pubkey_x must be a hex string".to_string()),
    };
    let message = match &args[1] {
        Value::String(s) => s.as_bytes().to_vec(),
        _ => return Err("schnorr_verify: message must be a string".to_string()),
    };
    let sig = match &args[2] {
        Value::Struct { fields, .. } => {
            let rx_str = match fields.get("rx") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("schnorr_verify: signature must have 'rx' field".to_string()),
            };
            let s_str = match fields.get("s") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("schnorr_verify: signature must have 's' field".to_string()),
            };
            crypto::SchnorrSignature {
                rx: crypto::BigUint256::from_hex(&rx_str).ok_or("invalid rx")?,
                s: crypto::BigUint256::from_hex(&s_str).ok_or("invalid s")?,
            }
        }
        _ => return Err("schnorr_verify: signature must be a struct with rx, s fields".to_string()),
    };
    Ok(Value::Bool(crypto::schnorr_verify(&pubkey_x, &message, &sig)))
}

fn builtin_point_x(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_x expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let (x, _) = crypto::point_to_affine(p);
            Ok(Value::BigInt(x))
        }
        _ => Err("point_x: argument must be an EC point".to_string()),
    }
}

fn builtin_point_y(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("point_y expects 1 argument".to_string()); }
    match &args[0] {
        Value::ECPoint(p) => {
            let (_, y) = crypto::point_to_affine(p);
            Ok(Value::BigInt(y))
        }
        _ => Err("point_y: argument must be an EC point".to_string()),
    }
}

fn builtin_bigint_from_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("bigint_from_hex expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => {
            let n = crypto::BigUint256::from_hex(s)
                .ok_or_else(|| format!("invalid hex: {}", s))?;
            Ok(Value::BigInt(n))
        }
        _ => Err("bigint_from_hex: argument must be a string".to_string()),
    }
}

fn builtin_to_hex(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_hex expects 1 argument".to_string()); }
    match &args[0] {
        Value::BigInt(n) => Ok(Value::String(format!("0x{}", n))),
        Value::FieldElem(fe) => Ok(Value::String(format!("0x{}", fe.value))),
        Value::Int(n) => Ok(Value::String(format!("0x{:x}", n))),
        _ => Err("to_hex: argument must be a numeric type".to_string()),
    }
}

fn builtin_len(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("len expects 1 argument".to_string()); }
    match &args[0] {
        Value::Array(arr) => Ok(Value::Int(arr.len() as i128)),
        Value::String(s) => Ok(Value::Int(s.len() as i128)),
        _ => Err(format!("len: unsupported type {}", args[0])),
    }
}

fn builtin_to_string(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_string expects 1 argument".to_string()); }
    Ok(Value::String(format!("{}", args[0])))
}

fn builtin_format(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("format expects at least 1 argument".to_string()); }
    let template = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("format: first argument must be a string".to_string()),
    };
    // Simple {} placeholder replacement
    let mut result = String::new();
    let mut arg_idx = 1;
    let mut chars = template.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'}') {
            chars.next(); // consume }
            if arg_idx < args.len() {
                result.push_str(&format!("{}", args[arg_idx]));
                arg_idx += 1;
            } else {
                result.push_str("{}");
            }
        } else {
            result.push(c);
        }
    }
    Ok(Value::String(result))
}

fn builtin_push(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("push expects 2 arguments (array, value)".to_string()); }
    match &args[0] {
        Value::Array(arr) => {
            let mut new_arr = arr.clone();
            new_arr.push(args[1].clone());
            Ok(Value::Array(new_arr))
        }
        _ => Err("push: first argument must be an array".to_string()),
    }
}

fn builtin_assert(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() { return Err("assert expects at least 1 argument".to_string()); }
    let cond = value_to_bool(&args[0])?;
    if !cond {
        let msg = if args.len() > 1 {
            format!("{}", args[1])
        } else {
            "assertion failed".to_string()
        };
        return Err(msg);
    }
    Ok(Value::Void)
}

fn builtin_assert_eq(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("assert_eq expects 2 arguments".to_string()); }
    if !values_equal(&args[0], &args[1]) {
        return Err(format!("assertion failed: {} != {}", args[0], args[1]));
    }
    Ok(Value::Void)
}

// --- Spiking / neuromorphic builtins ---

fn builtin_spike_train(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("spike_train expects 2 arguments: (timesteps, neurons)".to_string()); }
    let timesteps = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("spike_train: timesteps must be int".to_string()) };
    let neurons = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("spike_train: neurons must be int".to_string()) };
    Ok(Value::SpikeTrain(spiking::SpikeTrain::new(timesteps, neurons)))
}

fn builtin_spike_from_dense(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("spike_from_dense expects 4 arguments: (data, timesteps, neurons, threshold)".to_string()); }
    let data = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f), Value::Int(i) => Ok(*i as f64),
            _ => Err("spike_from_dense: data must be numeric array".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("spike_from_dense: first argument must be array".to_string()),
    };
    let timesteps = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("timesteps must be int".to_string()) };
    let neurons = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("neurons must be int".to_string()) };
    let threshold = match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("threshold must be numeric".to_string()) };
    Ok(Value::SpikeTrain(spiking::SpikeTrain::from_dense(&data, timesteps, neurons, threshold)))
}

fn builtin_spike_overlap(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("spike_overlap expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::SpikeTrain(st) => st, _ => return Err("spike_overlap: first arg must be SpikeTrain".to_string()) };
    let b = match &args[1] { Value::SpikeTrain(st) => st, _ => return Err("spike_overlap: second arg must be SpikeTrain".to_string()) };
    Ok(Value::Int(spiking::spike_overlap(a, b) as i128))
}

fn builtin_lif_layer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("lif_layer expects 6 arguments: (spikes, weights, n_in, n_out, threshold, tau)".to_string()); }
    let input = match &args[0] { Value::SpikeTrain(st) => st, _ => return Err("lif_layer: first arg must be SpikeTrain".to_string()) };
    let weights = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f), Value::Int(i) => Ok(*i as f64),
            _ => Err("lif_layer: weights must be numeric".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("lif_layer: weights must be array".to_string()),
    };
    let n_in = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("n_in must be int".to_string()) };
    let n_out = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("n_out must be int".to_string()) };
    let threshold = match &args[4] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("threshold must be numeric".to_string()) };
    let tau = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("tau must be numeric".to_string()) };
    Ok(Value::SpikeTrain(spiking::lif_layer(input, &weights, n_in, n_out, threshold, tau)))
}

fn builtin_spike_to_dense(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("spike_to_dense expects 1 argument".to_string()); }
    let train = match &args[0] { Value::SpikeTrain(st) => st, _ => return Err("spike_to_dense: arg must be SpikeTrain".to_string()) };
    Ok(Value::Array(train.to_dense().into_iter().map(Value::Float).collect()))
}

// --- Differentiable memory builtins ---

fn builtin_memory_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("memory_new expects 3 arguments: (capacity, key_dim, val_dim)".to_string()); }
    let capacity = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("capacity must be int".to_string()) };
    let key_dim = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("key_dim must be int".to_string()) };
    let val_dim = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("val_dim must be int".to_string()) };
    Ok(Value::DiffMemory(Box::new(memory::DiffMemory::new(capacity, key_dim, val_dim))))
}

fn values_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f), Value::Int(i) => Ok(*i as f64),
            _ => Err("expected numeric array".to_string()),
        }).collect(),
        _ => Err("expected array".to_string()),
    }
}

fn builtin_memory_read(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("memory_read expects 2 arguments: (mem, query)".to_string()); }
    let mem = match &args[0] { Value::DiffMemory(m) => m, _ => return Err("memory_read: first arg must be Memory".to_string()) };
    let query = values_to_f64_vec(&args[1])?;
    let result = mem.read(&query);
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_memory_write(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("memory_write expects 3 arguments: (mem, key, value)".to_string()); }
    let mut mem = match &args[0] { Value::DiffMemory(m) => (**m).clone(), _ => return Err("memory_write: first arg must be Memory".to_string()) };
    let key = values_to_f64_vec(&args[1])?;
    let value = values_to_f64_vec(&args[2])?;
    mem.write(&key, &value);
    Ok(Value::DiffMemory(Box::new(mem)))
}

fn builtin_memory_content_lookup(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("memory_content_lookup expects 3 arguments: (mem, query, beta)".to_string()); }
    let mem = match &args[0] { Value::DiffMemory(m) => m, _ => return Err("first arg must be Memory".to_string()) };
    let query = values_to_f64_vec(&args[1])?;
    let beta = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta must be numeric".to_string()) };
    let weights = mem.content_lookup(&query, beta);
    Ok(Value::Array(weights.into_iter().map(Value::Float).collect()))
}

// --- LLM operation builtins (stubs) ---

fn builtin_softmax(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Array(arr)) => {
            let floats: Result<Vec<f64>, String> = arr.iter().map(|v| match v {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("softmax: expected numeric array".to_string()),
            }).collect();
            let floats = floats?;
            let max = floats.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = floats.iter().map(|x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            Ok(Value::Array(exps.iter().map(|e| Value::Float(e / sum)).collect()))
        }
        _ => Err("softmax expects an array argument".to_string()),
    }
}

fn builtin_gelu(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let x = match args.first() {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(i)) => *i as f64,
        _ => return Err("gelu expects a numeric argument".to_string()),
    };
    // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    Ok(Value::Float(0.5 * x * (1.0 + (c * (x + 0.044715 * x.powi(3))).tanh())))
}

fn builtin_layer_norm(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Array(arr)) => {
            let floats: Result<Vec<f64>, String> = arr.iter().map(|v| match v {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("layer_norm: expected numeric array".to_string()),
            }).collect();
            let floats = floats?;
            let n = floats.len() as f64;
            let mean = floats.iter().sum::<f64>() / n;
            let var = floats.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
            let std = (var + 1e-5).sqrt();
            Ok(Value::Array(floats.iter().map(|x| Value::Float((x - mean) / std)).collect()))
        }
        _ => Err("layer_norm expects an array argument".to_string()),
    }
}

fn builtin_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("attention expects 3 array arguments (q, k, v)".to_string());
    }
    let q = &args[0];
    let k = &args[1];
    let v = &args[2];

    // Extract 2D matrices
    let q_mat = value_to_2d_f64(q, "Q")?;
    let k_mat = value_to_2d_f64(k, "K")?;
    let v_mat = value_to_2d_f64(v, "V")?;

    if q_mat.is_empty() || k_mat.is_empty() || v_mat.is_empty() {
        return Ok(Value::Array(vec![]));
    }

    let d_k = k_mat[0].len() as f64;
    let scale = 1.0 / d_k.sqrt();

    // Q @ K^T: (seq_q, d_k) x (d_k, seq_k) -> (seq_q, seq_k)
    let seq_q = q_mat.len();
    let seq_k = k_mat.len();
    let dk = q_mat[0].len();
    if dk != k_mat[0].len() {
        return Err("attention: Q and K dimension mismatch".to_string());
    }

    let mut scores = vec![vec![0.0f64; seq_k]; seq_q];
    for i in 0..seq_q {
        for j in 0..seq_k {
            for p in 0..dk {
                scores[i][j] += q_mat[i][p] * k_mat[j][p]; // K^T: swap j,p
            }
            scores[i][j] *= scale;
        }
    }

    // Apply softmax to each row
    for row in &mut scores {
        let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = row.iter().map(|x| (x - max).exp()).collect();
        let sum: f64 = exps.iter().sum();
        for (i, e) in exps.iter().enumerate() {
            row[i] = e / sum;
        }
    }

    // scores @ V: (seq_q, seq_k) x (seq_k, d_v) -> (seq_q, d_v)
    let d_v = v_mat[0].len();
    if seq_k != v_mat.len() {
        return Err("attention: K and V sequence length mismatch".to_string());
    }
    let mut output = vec![vec![0.0f64; d_v]; seq_q];
    for i in 0..seq_q {
        for j in 0..d_v {
            for p in 0..seq_k {
                output[i][j] += scores[i][p] * v_mat[p][j];
            }
        }
    }

    Ok(Value::Array(output.into_iter().map(|row| {
        Value::Array(row.into_iter().map(Value::Float).collect())
    }).collect()))
}

/// flash_attention(q, k, v, num_heads, causal) -> output
/// q, k, v are 2D arrays [N, d]. num_heads is int, causal is bool.
fn builtin_flash_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("flash_attention expects (q, k, v, [num_heads], [causal])".to_string());
    }
    let q_mat = value_to_2d_f64(&args[0], "Q")?;
    let k_mat = value_to_2d_f64(&args[1], "K")?;
    let v_mat = value_to_2d_f64(&args[2], "V")?;

    if q_mat.is_empty() || k_mat.is_empty() || v_mat.is_empty() {
        return Ok(Value::Array(vec![]));
    }

    let n = q_mat.len();
    let total_dim = q_mat[0].len();
    let num_heads: usize = match args.get(3) {
        Some(Value::Int(h)) => *h as usize,
        _ => 1,
    };
    let causal = match args.get(4) {
        Some(Value::Bool(b)) => *b,
        _ => false,
    };

    let head_dim = total_dim / num_heads;
    if head_dim * num_heads != total_dim {
        return Err(format!("flash_attention: dim {} not divisible by {} heads", total_dim, num_heads));
    }

    // Flatten to row-major
    let q_flat: Vec<f64> = q_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let k_flat: Vec<f64> = k_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let v_flat: Vec<f64> = v_mat.iter().flat_map(|r| r.iter().copied()).collect();

    let out = crate::flash_attention::multi_head_flash_attention(
        &q_flat, &k_flat, &v_flat, num_heads, head_dim, causal,
    );

    // Reshape back to 2D Value
    let result: Vec<Value> = (0..n)
        .map(|i| {
            Value::Array(
                out[i * total_dim..(i + 1) * total_dim]
                    .iter()
                    .map(|&x| Value::Float(x))
                    .collect(),
            )
        })
        .collect();
    Ok(Value::Array(result))
}

/// flash_attention_backward(q, k, v, output, grad_output, num_heads, causal) -> (grad_q, grad_k, grad_v)
fn builtin_flash_attention_backward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 {
        return Err("flash_attention_backward expects (q, k, v, output, grad_output, [num_heads], [causal])".to_string());
    }
    let q_mat = value_to_2d_f64(&args[0], "Q")?;
    let k_mat = value_to_2d_f64(&args[1], "K")?;
    let v_mat = value_to_2d_f64(&args[2], "V")?;
    let out_mat = value_to_2d_f64(&args[3], "output")?;
    let grad_mat = value_to_2d_f64(&args[4], "grad_output")?;

    if q_mat.is_empty() {
        return Ok(Value::Tuple(vec![Value::Array(vec![]), Value::Array(vec![]), Value::Array(vec![])]));
    }

    let n = q_mat.len();
    let total_dim = q_mat[0].len();
    let num_heads: usize = match args.get(5) {
        Some(Value::Int(h)) => *h as usize,
        _ => 1,
    };
    let causal = match args.get(6) {
        Some(Value::Bool(b)) => *b,
        _ => false,
    };
    let head_dim = total_dim / num_heads;

    // For simplicity, run backward per-head
    let q_flat: Vec<f64> = q_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let k_flat: Vec<f64> = k_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let v_flat: Vec<f64> = v_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let out_flat: Vec<f64> = out_mat.iter().flat_map(|r| r.iter().copied()).collect();
    let grad_flat: Vec<f64> = grad_mat.iter().flat_map(|r| r.iter().copied()).collect();

    let mut gq = vec![0.0; n * total_dim];
    let mut gk = vec![0.0; n * total_dim];
    let mut gv = vec![0.0; n * total_dim];

    for h in 0..num_heads {
        let mut qh = vec![0.0; n * head_dim];
        let mut kh = vec![0.0; n * head_dim];
        let mut vh = vec![0.0; n * head_dim];
        let mut oh = vec![0.0; n * head_dim];
        let mut doh = vec![0.0; n * head_dim];
        for i in 0..n {
            for dd in 0..head_dim {
                qh[i * head_dim + dd] = q_flat[i * total_dim + h * head_dim + dd];
                kh[i * head_dim + dd] = k_flat[i * total_dim + h * head_dim + dd];
                vh[i * head_dim + dd] = v_flat[i * total_dim + h * head_dim + dd];
                oh[i * head_dim + dd] = out_flat[i * total_dim + h * head_dim + dd];
                doh[i * head_dim + dd] = grad_flat[i * total_dim + h * head_dim + dd];
            }
        }

        let config = crate::flash_attention::FlashAttentionConfig::auto(n, head_dim, 1);
        let (_, lse, _) = crate::flash_attention::flash_attention_forward(&qh, &kh, &vh, &config);
        let (dq, dk, dv) = crate::flash_attention::flash_attention_backward(
            &qh, &kh, &vh, &oh, &doh, &lse, &config,
        );

        for i in 0..n {
            for dd in 0..head_dim {
                gq[i * total_dim + h * head_dim + dd] = dq[i * head_dim + dd];
                gk[i * total_dim + h * head_dim + dd] = dk[i * head_dim + dd];
                gv[i * total_dim + h * head_dim + dd] = dv[i * head_dim + dd];
            }
        }
    }

    let to_2d = |flat: &[f64]| -> Value {
        Value::Array((0..n).map(|i| {
            Value::Array(flat[i * total_dim..(i + 1) * total_dim].iter().map(|&x| Value::Float(x)).collect())
        }).collect())
    };

    Ok(Value::Tuple(vec![to_2d(&gq), to_2d(&gk), to_2d(&gv)]))
}

fn value_to_2d_f64(val: &Value, name: &str) -> Result<Vec<Vec<f64>>, String> {
    match val {
        Value::Array(rows) if !rows.is_empty() => {
            // Check if first element is an array (2D) or scalar (1D)
            match &rows[0] {
                Value::Array(_) => {
                    rows.iter().map(|row| match row {
                        Value::Array(r) => r.iter().map(|v| value_to_f64(v)).collect(),
                        _ => Err(format!("{}: expected 2D array", name)),
                    }).collect()
                }
                _ => {
                    // 1D array: treat as single row
                    let row: Result<Vec<f64>, String> = rows.iter().map(|v| value_to_f64(v)).collect();
                    Ok(vec![row?])
                }
            }
        }
        Value::Array(_) => Ok(vec![]),
        _ => Err(format!("{}: expected 2D array", name)),
    }
}

fn builtin_rope(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let x:Vec<f64> = match args.first() { Some(Value::Array(a)) => a.iter().map(|v| match v { Value::Float(f)=>*f, Value::Int(i)=>*i as f64, _=>0.0 }).collect(), _ => return Err("rope expects arrays".into()) };
    let fr:Vec<f64> = match args.get(1) { Some(Value::Array(a)) => a.iter().map(|v| match v { Value::Float(f)=>*f, Value::Int(i)=>*i as f64, _=>0.0 }).collect(), _ => return Err("rope expects 2 arrays".into()) };
    let mut r=x.clone(); for i in 0..x.len()/2 { let f=if i<fr.len(){fr[i]}else{0.0}; let(s,c)=f.sin_cos(); r[2*i]=x[2*i]*c-x[2*i+1]*s; r[2*i+1]=x[2*i]*s+x[2*i+1]*c; }
    Ok(Value::Array(r.into_iter().map(Value::Float).collect()))
}

// --- Phase 2: ZK primitive builtins ---

fn builtin_montgomery_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("montgomery_mul expects 3 arguments: (a, b, modulus_hex)".to_string()); }
    let a = match &args[0] {
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => return Err("montgomery_mul: first argument must be a field element or bigint".to_string()),
    };
    let b = match &args[1] {
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => return Err("montgomery_mul: second argument must be a field element or bigint".to_string()),
    };
    let modulus = match &args[2] {
        Value::String(s) => crypto::BigUint256::from_hex(s).ok_or_else(|| format!("invalid hex: {}", s))?,
        Value::FieldElem(fe) => fe.modulus.clone(),
        _ => return Err("montgomery_mul: third argument must be modulus hex string".to_string()),
    };
    let params = crypto::montgomery_params(&modulus);
    let a_mont = crypto::to_montgomery(&a, &params);
    let b_mont = crypto::to_montgomery(&b, &params);
    let result_mont = crypto::montgomery_mul(&a_mont, &b_mont, &params);
    let result = crypto::from_montgomery(&result_mont, &params);
    Ok(Value::FieldElem(crypto::FieldElement::new(result, modulus)))
}

fn builtin_ntt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("ntt expects 1 argument: array of field elements".to_string()); }
    let arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("ntt: argument must be an array".to_string()),
    };
    let mut data: Vec<crypto::BigUint256> = arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let n = data.len().next_power_of_two();
    data.resize(n, crypto::BigUint256::ZERO);
    let log_n = n.trailing_zeros();
    let domain = crate::ntt::bn254_fr_domain(log_n);
    crate::ntt::ntt(&mut data, &domain);
    let result: Vec<Value> = data.into_iter().map(|v| Value::BigInt(v)).collect();
    Ok(Value::Array(result))
}

fn builtin_intt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("intt expects 1 argument: array of field elements".to_string()); }
    let arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("intt: argument must be an array".to_string()),
    };
    let mut data: Vec<crypto::BigUint256> = arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let n = data.len();
    assert!(n.is_power_of_two(), "intt: array length must be power of 2");
    let log_n = n.trailing_zeros();
    let domain = crate::ntt::bn254_fr_domain(log_n);
    crate::ntt::intt(&mut data, &domain);
    let result: Vec<Value> = data.into_iter().map(|v| Value::BigInt(v)).collect();
    Ok(Value::Array(result))
}

fn builtin_msm(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("msm expects 2 arguments: (scalars_array, points_array)".to_string()); }
    let scalars_arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("msm: first argument must be an array of scalars".to_string()),
    };
    let points_arr = match &args[1] {
        Value::Array(a) => a.clone(),
        _ => return Err("msm: second argument must be an array of points".to_string()),
    };
    let scalars: Vec<crypto::BigUint256> = scalars_arr.iter().map(|v| match v {
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let points: Vec<crypto::ECPoint> = points_arr.iter().map(|v| match v {
        Value::ECPoint(p) => p.clone(),
        _ => crypto::ECPoint::identity(),
    }).collect();
    let result = crate::msm::msm_pippenger(&scalars, &points);
    Ok(Value::ECPoint(result))
}

fn builtin_pairing(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("pairing expects 2 arguments: (g1_point, g2_point)".to_string()); }
    // For the interpreter, pairing returns a string representation of the Fp12 result
    Ok(Value::String("pairing_result".to_string()))
}

fn builtin_pairing_check(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("pairing_check expects 1 argument: array of (G1, G2) pairs".to_string()); }
    // Placeholder: always returns true for demonstration
    Ok(Value::Bool(true))
}

fn builtin_poly_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("poly_mul expects 2 arguments: (coeffs_a, coeffs_b)".to_string()); }
    let a_arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("poly_mul: arguments must be arrays".to_string()),
    };
    let b_arr = match &args[1] {
        Value::Array(a) => a.clone(),
        _ => return Err("poly_mul: arguments must be arrays".to_string()),
    };
    let modulus = crate::fields::bn254_scalar_prime();
    let a_coeffs: Vec<crypto::BigUint256> = a_arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let b_coeffs: Vec<crypto::BigUint256> = b_arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let poly_a = crate::poly::Polynomial::new(a_coeffs, modulus.clone());
    let poly_b = crate::poly::Polynomial::new(b_coeffs, modulus.clone());
    let result = poly_a.mul_schoolbook(&poly_b);
    let result_vals: Vec<Value> = result.coeffs.into_iter().map(|v| Value::BigInt(v)).collect();
    Ok(Value::Array(result_vals))
}

fn builtin_poly_eval(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("poly_eval expects 2 arguments: (coeffs, point)".to_string()); }
    let coeffs_arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("poly_eval: first argument must be an array".to_string()),
    };
    let point = match &args[1] {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("poly_eval: second argument must be a number".to_string()),
    };
    let modulus = crate::fields::bn254_scalar_prime();
    let coeffs: Vec<crypto::BigUint256> = coeffs_arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::FieldElem(fe) => fe.value.clone(),
        Value::BigInt(n) => n.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let poly = crate::poly::Polynomial::new(coeffs, modulus);
    let result = poly.eval(&point);
    Ok(Value::BigInt(result))
}

fn builtin_poly_interpolate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("poly_interpolate expects 1 argument: array of evaluations".to_string()); }
    let evals_arr = match &args[0] {
        Value::Array(a) => a.clone(),
        _ => return Err("poly_interpolate: argument must be an array".to_string()),
    };
    let modulus = crate::fields::bn254_scalar_prime();
    let evals: Vec<crypto::BigUint256> = evals_arr.iter().map(|v| match v {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => crypto::BigUint256::ZERO,
    }).collect();
    let n = evals.len();
    let log_n = (n as f64).log2() as u32;
    let domain = crate::ntt::bn254_fr_domain(log_n);
    let poly = crate::poly::Polynomial::interpolate(&evals, &domain, &modulus);
    let result: Vec<Value> = poly.coeffs.into_iter().map(|v| Value::BigInt(v)).collect();
    Ok(Value::Array(result))
}

fn builtin_fp2_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("fp2_new expects 2 arguments: (c0, c1)".to_string()); }
    let c0 = match &args[0] {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::BigInt(n) => n.clone(),
        _ => return Err("fp2_new: arguments must be integers".to_string()),
    };
    let c1 = match &args[1] {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::BigInt(n) => n.clone(),
        _ => return Err("fp2_new: arguments must be integers".to_string()),
    };
    let p = crate::fields::bn254_field_prime();
    let fp2 = crate::fields::Fp2::new(c0, c1, p);
    let mut fields = HashMap::new();
    fields.insert("c0".to_string(), Value::BigInt(fp2.c0));
    fields.insert("c1".to_string(), Value::BigInt(fp2.c1));
    Ok(Value::Struct { name: "Fp2".to_string(), fields })
}

fn builtin_fp2_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("fp2_mul expects 2 arguments".to_string()); }
    let p = crate::fields::bn254_field_prime();
    let extract = |v: &Value| -> Result<crate::fields::Fp2, String> {
        match v {
            Value::Struct { fields, .. } => {
                let c0 = match fields.get("c0") {
                    Some(Value::BigInt(n)) => n.clone(),
                    _ => return Err("fp2_mul: struct must have c0 BigInt field".to_string()),
                };
                let c1 = match fields.get("c1") {
                    Some(Value::BigInt(n)) => n.clone(),
                    _ => return Err("fp2_mul: struct must have c1 BigInt field".to_string()),
                };
                Ok(crate::fields::Fp2::new(c0, c1, p.clone()))
            }
            _ => Err("fp2_mul: arguments must be Fp2 structs".to_string()),
        }
    };
    let a = extract(&args[0])?;
    let b = extract(&args[1])?;
    let result = a.mul(&b);
    let mut fields = HashMap::new();
    fields.insert("c0".to_string(), Value::BigInt(result.c0));
    fields.insert("c1".to_string(), Value::BigInt(result.c1));
    Ok(Value::Struct { name: "Fp2".to_string(), fields })
}

fn builtin_fp2_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("fp2_add expects 2 arguments".to_string()); }
    let p = crate::fields::bn254_field_prime();
    let extract = |v: &Value| -> Result<crate::fields::Fp2, String> {
        match v {
            Value::Struct { fields, .. } => {
                let c0 = match fields.get("c0") {
                    Some(Value::BigInt(n)) => n.clone(),
                    _ => return Err("fp2_add: struct must have c0 BigInt field".to_string()),
                };
                let c1 = match fields.get("c1") {
                    Some(Value::BigInt(n)) => n.clone(),
                    _ => return Err("fp2_add: struct must have c1 BigInt field".to_string()),
                };
                Ok(crate::fields::Fp2::new(c0, c1, p.clone()))
            }
            _ => Err("fp2_add: arguments must be Fp2 structs".to_string()),
        }
    };
    let a = extract(&args[0])?;
    let b = extract(&args[1])?;
    let result = a.add(&b);
    let mut fields = HashMap::new();
    fields.insert("c0".to_string(), Value::BigInt(result.c0));
    fields.insert("c1".to_string(), Value::BigInt(result.c1));
    Ok(Value::Struct { name: "Fp2".to_string(), fields })
}

fn builtin_fp2_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("fp2_inv expects 1 argument".to_string()); }
    let p = crate::fields::bn254_field_prime();
    let a = match &args[0] {
        Value::Struct { fields, .. } => {
            let c0 = match fields.get("c0") {
                Some(Value::BigInt(n)) => n.clone(),
                _ => return Err("fp2_inv: struct must have c0 BigInt field".to_string()),
            };
            let c1 = match fields.get("c1") {
                Some(Value::BigInt(n)) => n.clone(),
                _ => return Err("fp2_inv: struct must have c1 BigInt field".to_string()),
            };
            crate::fields::Fp2::new(c0, c1, p)
        }
        _ => return Err("fp2_inv: argument must be an Fp2 struct".to_string()),
    };
    let result = a.inv();
    let mut fields = HashMap::new();
    fields.insert("c0".to_string(), Value::BigInt(result.c0));
    fields.insert("c1".to_string(), Value::BigInt(result.c1));
    Ok(Value::Struct { name: "Fp2".to_string(), fields })
}

fn builtin_g1_generator(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let g = crate::pairing::G1Point::generator();
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::BigInt(g.x));
    fields.insert("y".to_string(), Value::BigInt(g.y));
    Ok(Value::Struct { name: "G1Point".to_string(), fields })
}

fn builtin_g2_generator(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let g = crate::pairing::G2Point::generator();
    let mut fields = HashMap::new();
    fields.insert("x_c0".to_string(), Value::BigInt(g.x.c0));
    fields.insert("x_c1".to_string(), Value::BigInt(g.x.c1));
    fields.insert("y_c0".to_string(), Value::BigInt(g.y.c0));
    fields.insert("y_c1".to_string(), Value::BigInt(g.y.c1));
    Ok(Value::Struct { name: "G2Point".to_string(), fields })
}

fn builtin_g1_scalar_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("g1_scalar_mul expects 2 arguments: (scalar, g1_point)".to_string()); }
    let scalar = match &args[0] {
        Value::Int(n) => crypto::BigUint256::from_u64(*n as u64),
        Value::BigInt(n) => n.clone(),
        Value::FieldElem(fe) => fe.value.clone(),
        _ => return Err("g1_scalar_mul: first argument must be a scalar".to_string()),
    };
    let point = match &args[1] {
        Value::Struct { fields, .. } => {
            let x = match fields.get("x") {
                Some(Value::BigInt(n)) => n.clone(),
                _ => return Err("g1_scalar_mul: struct must have x BigInt field".to_string()),
            };
            let y = match fields.get("y") {
                Some(Value::BigInt(n)) => n.clone(),
                _ => return Err("g1_scalar_mul: struct must have y BigInt field".to_string()),
            };
            crate::pairing::G1Point { x, y, infinity: false }
        }
        _ => return Err("g1_scalar_mul: second argument must be a G1Point struct".to_string()),
    };
    let result = crate::pairing::g1_scalar_mul(&scalar, &point);
    let mut fields = HashMap::new();
    fields.insert("x".to_string(), Value::BigInt(result.x));
    fields.insert("y".to_string(), Value::BigInt(result.y));
    Ok(Value::Struct { name: "G1Point".to_string(), fields })
}

// --- Interpreter entry point ---

pub fn interpret(program: &Program) -> Result<Vec<String>, String> {
    let mut env = Env::new();

    // First pass: register all functions and kernels
    for item in &program.items {
        match &item.kind {
            ItemKind::Function(func) => {
                let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    func.name.name.clone(),
                    FnDef::User {
                        params,
                        body: func.body.clone(),
                    },
                );
            }
            ItemKind::Kernel(kernel) => {
                let params: Vec<String> = kernel.params.iter().map(|p| p.name.name.clone()).collect();
                env.functions.insert(
                    kernel.name.name.clone(),
                    FnDef::User {
                        params,
                        body: kernel.body.clone(),
                    },
                );
            }
            ItemKind::Const(c) => {
                let val = eval_expr(&mut env, &c.value)?;
                env.define(&c.name.name, val);
            }
            ItemKind::Struct(s) => {
                let field_names: Vec<String> = s.fields.iter().map(|f| f.name.name.clone()).collect();
                env.struct_defs.insert(s.name.name.clone(), field_names);
            }
            ItemKind::Enum(e) => {
                let enum_name = e.name.name.clone();
                for variant in &e.variants {
                    let vn = variant.name.name.clone();
                    let en = enum_name.clone();
                    match &variant.kind {
                        crate::ast::EnumVariantKind::Unit => {
                            let val = Value::EnumVariant {
                                enum_name: en.clone(),
                                variant: vn.clone(),
                                fields: Vec::new(),
                            };
                            env.define(&vn, val.clone());
                            env.define(&format!("{}::{}", en, vn), val);
                        }
                        crate::ast::EnumVariantKind::Tuple(types) => {
                            // Store arity info for constructor dispatch
                            env.struct_defs.insert(
                                format!("__enum_{}_{}", en, vn),
                                (0..types.len()).map(|i| format!("_{}", i)).collect(),
                            );
                        }
                        crate::ast::EnumVariantKind::Struct(_) => {}
                    }
                }
            }
            ItemKind::Impl(impl_block) => {
                // Get the target type name
                let type_name = match &impl_block.target.kind {
                    crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                    _ => continue,
                };
                // Register each method as TypeName::method_name
                for method_item in &impl_block.methods {
                    if let ItemKind::Function(func) = &method_item.kind {
                        let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                        let qualified_name = format!("{}::{}", type_name, func.name.name);
                        env.functions.insert(
                            qualified_name,
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

    // Find and call main()
    if let Some(func) = env.functions.get("main").cloned() {
        match func {
            FnDef::User { body, .. } => {
                eval_block(&mut env, &body)?;
            }
            _ => return Err("main must be a user function".to_string()),
        }
    } else {
        return Err("no main() function found".to_string());
    }

    Ok(env.output)
}

pub(crate) fn eval_block(env: &mut Env, block: &Block) -> Result<Value, String> {
    for stmt in &block.stmts {
        let val = eval_stmt(env, stmt)?;
        match &val {
            Value::Return(_) | Value::Break | Value::Continue => return Ok(val),
            _ => {}
        }
    }
    if let Some(expr) = &block.expr {
        eval_expr(env, expr)
    } else {
        Ok(Value::Void)
    }
}

fn eval_stmt(env: &mut Env, stmt: &Stmt) -> Result<Value, String> {
    match &stmt.kind {
        StmtKind::Let { name, value, .. } => {
            let val = eval_expr(env, value)?;
            env.define(&name.name, val);
            Ok(Value::Void)
        }
        StmtKind::Var { name, value, .. } => {
            let val = eval_expr(env, value)?;
            env.define(&name.name, val);
            Ok(Value::Void)
        }
        StmtKind::Return(Some(expr)) => {
            let val = eval_expr(env, expr)?;
            Ok(Value::Return(Box::new(val)))
        }
        StmtKind::Return(None) => Ok(Value::Return(Box::new(Value::Void))),
        StmtKind::Expr(expr) => eval_expr(env, expr),
        StmtKind::Assign { target, op, value } => {
            let rhs = eval_expr(env, value)?;
            let name = match &target.kind {
                ExprKind::Ident(id) => id.name.clone(),
                ExprKind::Index { base, indices } => {
                    // Array index assignment
                    if let ExprKind::Ident(id) = &base.kind {
                        let idx = eval_expr(env, &indices[0])?;
                        let idx = value_to_int(&idx)? as usize;
                        if let Some(Value::Array(mut arr)) = env.get(&id.name) {
                            let new_val = match op {
                                AssignOp::Assign => rhs,
                                AssignOp::AddAssign => value_add(&arr[idx], &rhs)?,
                                AssignOp::SubAssign => value_sub(&arr[idx], &rhs)?,
                                AssignOp::MulAssign => value_mul(&arr[idx], &rhs)?,
                                _ => return Err("unsupported assign op for array".to_string()),
                            };
                            arr[idx] = new_val;
                            env.set(&id.name, Value::Array(arr));
                            return Ok(Value::Void);
                        }
                    }
                    return Err("unsupported assignment target".to_string());
                }
                ExprKind::FieldAccess { base, field } => {
                    // Struct field assignment: s.field = value
                    if let ExprKind::Ident(id) = &base.kind {
                        if let Some(Value::Struct { name, mut fields }) = env.get(&id.name) {
                            let old = fields.get(&field.name).cloned().unwrap_or(Value::Void);
                            let new_val = match op {
                                AssignOp::Assign => rhs,
                                AssignOp::AddAssign => value_add(&old, &rhs)?,
                                AssignOp::SubAssign => value_sub(&old, &rhs)?,
                                AssignOp::MulAssign => value_mul(&old, &rhs)?,
                                _ => return Err("unsupported assign op for struct field".to_string()),
                            };
                            fields.insert(field.name.clone(), new_val);
                            env.set(&id.name, Value::Struct { name, fields });
                            return Ok(Value::Void);
                        }
                    }
                    return Err("unsupported field assignment target".to_string());
                }
                _ => return Err("unsupported assignment target".to_string()),
            };

            let current = env.get(&name).unwrap_or(Value::Int(0));
            let new_val = match op {
                AssignOp::Assign => rhs,
                AssignOp::AddAssign => value_add(&current, &rhs)?,
                AssignOp::SubAssign => value_sub(&current, &rhs)?,
                AssignOp::MulAssign => value_mul(&current, &rhs)?,
                AssignOp::DivAssign => value_div(&current, &rhs)?,
                AssignOp::MatMulAssign => return Err("@= not yet supported in interpreter".to_string()),
            };
            env.set(&name, new_val);
            Ok(Value::Void)
        }
        StmtKind::For { var, iter, body } => {
            let iter_val = eval_expr(env, iter)?;
            match iter_val {
                Value::Array(elems) => {
                    env.push_scope();
                    for elem in elems {
                        env.define(&var.name, elem);
                        let result = eval_block(env, body)?;
                        match &result {
                            Value::Return(_) => {
                                env.pop_scope();
                                return Ok(result);
                            }
                            Value::Break => break,
                            Value::Continue => continue,
                            _ => {}
                        }
                    }
                    env.pop_scope();
                }
                _ => {
                    return Err("for loop requires an iterable (array or range)".to_string());
                }
            }
            Ok(Value::Void)
        }
        StmtKind::While { cond, body } => {
            loop {
                let c = eval_expr(env, cond)?;
                if !value_to_bool(&c)? {
                    break;
                }
                env.push_scope();
                let result = eval_block(env, body)?;
                env.pop_scope();
                match &result {
                    Value::Return(_) => return Ok(result),
                    Value::Break => break,
                    Value::Continue => continue,
                    _ => {}
                }
            }
            Ok(Value::Void)
        }
        StmtKind::Loop { body } => {
            loop {
                env.push_scope();
                let result = eval_block(env, body)?;
                env.pop_scope();
                match result {
                    Value::Return(_) => return Ok(result),
                    Value::Break => break,
                    Value::Continue => continue,
                    _ => {}
                }
            }
            Ok(Value::Void)
        }
        StmtKind::Break => Ok(Value::Break),
        StmtKind::Continue => Ok(Value::Continue),
        StmtKind::Dispatch { index, targets, args } => {
            let idx_val = eval_expr(env, index)?;
            let idx = value_to_int(&idx_val)? as usize;
            if idx >= targets.len() {
                return Err(format!("dispatch index {} out of bounds (have {} targets)", idx, targets.len()));
            }
            let target_name = targets[idx].name.clone();
            let arg_vals: Vec<Value> = args.iter()
                .map(|a| eval_expr(env, a))
                .collect::<Result<Vec<_>, _>>()?;
            let func_def = env.functions.get(&target_name).cloned()
                .ok_or_else(|| format!("undefined function in dispatch: {}", target_name))?;
            match func_def {
                FnDef::Builtin(f) => f(env, arg_vals),
                FnDef::User { params, body } => {
                    env.push_scope();
                    for (param, val) in params.iter().zip(arg_vals.iter()) {
                        env.define(param, val.clone());
                    }
                    let result = eval_block(env, &body)?;
                    env.pop_scope();
                    match result {
                        Value::Return(v) => Ok(*v),
                        other => Ok(other),
                    }
                }
            }
        }
    }
}

pub(crate) fn eval_expr(env: &mut Env, expr: &Expr) -> Result<Value, String> {
    match &expr.kind {
        ExprKind::IntLiteral(n) => Ok(Value::Int(*n as i128)),
        ExprKind::FloatLiteral(n) => Ok(Value::Float(*n)),
        ExprKind::StringLiteral(s) => Ok(Value::String(s.clone())),
        ExprKind::BoolLiteral(b) => Ok(Value::Bool(*b)),

        ExprKind::Ident(id) => {
            env.get(&id.name).ok_or_else(|| format!("undefined variable: {}", id.name))
        }

        ExprKind::Binary { lhs, op, rhs } => {
            let l = eval_expr(env, lhs)?;
            let r = eval_expr(env, rhs)?;
            eval_binop(&l, *op, &r)
        }

        ExprKind::Unary { op, expr: inner } => {
            let v = eval_expr(env, inner)?;
            match op {
                UnaryOp::Neg => match v {
                    Value::Int(n) => Ok(Value::Int(-n)),
                    Value::Float(n) => Ok(Value::Float(-n)),
                    Value::FieldElem(fe) => Ok(Value::FieldElem(fe.neg())),
                    _ => Err("cannot negate this type".to_string()),
                },
                UnaryOp::Not => match v {
                    Value::Bool(b) => Ok(Value::Bool(!b)),
                    _ => Err("cannot apply ! to this type".to_string()),
                },
                UnaryOp::BitNot => match v {
                    Value::Int(n) => Ok(Value::Int(!n)),
                    _ => Err("cannot apply ~ to this type".to_string()),
                },
            }
        }

        ExprKind::Call { func, args } => {
            // Check for method call: base.method(args)
            if let ExprKind::FieldAccess { base, field } = &func.kind {
                let base_val = eval_expr(env, base)?;
                let method_name = &field.name;

                // Try to find Type::method in the function registry
                let type_name = match &base_val {
                    Value::Struct { name, .. } => Some(name.clone()),
                    Value::EnumVariant { enum_name, .. } => Some(enum_name.clone()),
                    _ => None,
                };

                if let Some(ref tn) = type_name {
                    let qualified = format!("{}::{}", tn, method_name);
                    if let Some(func_def) = env.functions.get(&qualified).cloned() {
                        let mut arg_vals = vec![base_val];
                        for a in args {
                            arg_vals.push(eval_expr(env, a)?);
                        }
                        return match func_def {
                            FnDef::Builtin(f) => f(env, arg_vals),
                            FnDef::User { params, body } => {
                                env.push_scope();
                                for (param, val) in params.iter().zip(arg_vals.iter()) {
                                    env.define(param, val.clone());
                                }
                                let result = eval_block(env, &body)?;
                                env.pop_scope();
                                match result {
                                    Value::Return(v) => Ok(*v),
                                    other => Ok(other),
                                }
                            }
                        };
                    }
                }

                // Not a method call on a struct — treat as a plain function name
                // (e.g., module.func style)
                let func_name = format!("{}", base).replace("(", "").replace(")", "") + "." + method_name;
                let arg_vals: Vec<Value> = args.iter()
                    .map(|a| eval_expr(env, a))
                    .collect::<Result<Vec<_>, _>>()?;

                let func_def = env.functions.get(&func_name).cloned()
                    .ok_or_else(|| format!("undefined function: {}", func_name))?;

                return match func_def {
                    FnDef::Builtin(f) => f(env, arg_vals),
                    FnDef::User { params, body } => {
                        env.push_scope();
                        for (param, val) in params.iter().zip(arg_vals.iter()) {
                            env.define(param, val.clone());
                        }
                        let result = eval_block(env, &body)?;
                        env.pop_scope();
                        match result {
                            Value::Return(v) => Ok(*v),
                            other => Ok(other),
                        }
                    }
                };
            }

            let arg_vals: Vec<Value> = args.iter()
                .map(|a| eval_expr(env, a))
                .collect::<Result<Vec<_>, _>>()?;

            // Check if callee is a variable holding a closure/function value
            let func_name = match &func.kind {
                ExprKind::Ident(id) => id.name.clone(),
                _ => return Err("not a callable expression".to_string()),
            };

            // Check if this is a closure or function value in the environment
            if let Some(val) = env.get(&func_name) {
                match val {
                    Value::Closure { params, body, env: captured_env } => {
                        return call_closure(env, &params, &body, &captured_env, arg_vals);
                    }
                    Value::Function { params, body, .. } => {
                        env.push_scope();
                        for (param, val) in params.iter().zip(arg_vals.iter()) {
                            env.define(param, val.clone());
                        }
                        let result = eval_block(env, &body)?;
                        env.pop_scope();
                        return match result {
                            Value::Return(v) => Ok(*v),
                            other => Ok(other),
                        };
                    }
                    _ => {} // not callable, fall through to function lookup
                }
            }

            // Check if this is an enum variant constructor
            for (key, _) in env.struct_defs.iter() {
                if key.starts_with("__enum_") && key.ends_with(&format!("_{}", func_name)) {
                    let parts: Vec<&str> = key.strip_prefix("__enum_").unwrap().rsplitn(2, '_').collect();
                    if parts.len() == 2 {
                        let enum_name = parts[1].to_string();
                        return Ok(Value::EnumVariant {
                            enum_name,
                            variant: func_name,
                            fields: arg_vals,
                        });
                    }
                }
            }

            let func_def = env.functions.get(&func_name).cloned()
                .ok_or_else(|| format!("undefined function: {}", func_name))?;

            match func_def {
                FnDef::Builtin(f) => f(env, arg_vals),
                FnDef::User { params, body } => {
                    env.push_scope();
                    for (param, val) in params.iter().zip(arg_vals.iter()) {
                        env.define(param, val.clone());
                    }
                    let result = eval_block(env, &body)?;
                    env.pop_scope();
                    match result {
                        Value::Return(v) => Ok(*v),
                        other => Ok(other),
                    }
                }
            }
        }

        ExprKind::MatMul { lhs, rhs } => {
            let l = eval_expr(env, lhs)?;
            let r = eval_expr(env, rhs)?;
            matmul_values(&l, &r)
        }

        ExprKind::FieldAccess { base, field } => {
            let base_val = eval_expr(env, base)?;
            match &base_val {
                Value::Array(arr) => match field.name.as_str() {
                    "len" => Ok(Value::Int(arr.len() as i128)),
                    _ => Err(format!("array has no field .{}", field.name)),
                },
                Value::String(s) => match field.name.as_str() {
                    "len" => Ok(Value::Int(s.len() as i128)),
                    _ => Err(format!("string has no field .{}", field.name)),
                },
                Value::Struct { fields, .. } => {
                    fields.get(&field.name).cloned()
                        .ok_or_else(|| format!("struct has no field .{}", field.name))
                }
                Value::Tuple(elems) => {
                    // Support .0, .1, etc. for tuples
                    if let Ok(idx) = field.name.parse::<usize>() {
                        elems.get(idx).cloned()
                            .ok_or_else(|| format!("tuple index {} out of bounds", idx))
                    } else {
                        Err(format!("tuple has no field .{}", field.name))
                    }
                }
                _ => Err(format!("cannot access field .{} on {}", field.name, base_val)),
            }
        }

        ExprKind::Index { base, indices } => {
            let mut current = eval_expr(env, base)?;
            for index_expr in indices {
                let idx = eval_expr(env, index_expr)?;
                let idx = value_to_int(&idx)? as usize;
                current = match current {
                    Value::Array(arr) => {
                        arr.get(idx).cloned().ok_or_else(|| format!("index {} out of bounds (len {})", idx, arr.len()))?
                    }
                    _ => return Err("cannot index this type".to_string()),
                };
            }
            Ok(current)
        }

        ExprKind::Block(block) => {
            env.push_scope();
            let result = eval_block(env, block)?;
            env.pop_scope();
            Ok(result)
        }

        ExprKind::If { cond, then_block, else_block } => {
            let c = eval_expr(env, cond)?;
            if value_to_bool(&c)? {
                env.push_scope();
                let r = eval_block(env, then_block)?;
                env.pop_scope();
                Ok(r)
            } else if let Some(else_block) = else_block {
                env.push_scope();
                let r = eval_block(env, else_block)?;
                env.pop_scope();
                Ok(r)
            } else {
                Ok(Value::Void)
            }
        }

        ExprKind::Range { start, end } => {
            let s = value_to_int(&eval_expr(env, start)?)?;
            let e = value_to_int(&eval_expr(env, end)?)?;
            let elems: Vec<Value> = (s..e).map(Value::Int).collect();
            Ok(Value::Array(elems))
        }

        ExprKind::ArrayLiteral(elems) => {
            let vals: Vec<Value> = elems.iter()
                .map(|e| eval_expr(env, e))
                .collect::<Result<Vec<_>, _>>()?;
            Ok(Value::Array(vals))
        }

        ExprKind::Cast { expr: inner, ty } => {
            let val = eval_expr(env, inner)?;
            let target = match &ty.kind {
                TypeExprKind::Named(id) => id.name.as_str(),
                _ => return Ok(val),
            };
            match target {
                "f32" | "f64" | "float" => match val {
                    Value::Int(n) => Ok(Value::Float(n as f64)),
                    Value::Float(_) => Ok(val),
                    Value::Bool(b) => Ok(Value::Float(if b { 1.0 } else { 0.0 })),
                    _ => Err(format!("cannot cast {} to {}", val, target)),
                },
                "i8" | "i16" | "i32" | "i64" | "i128" | "int"
                | "u8" | "u16" | "u32" | "u64" | "u128" => match val {
                    Value::Float(f) => Ok(Value::Int(f as i128)),
                    Value::Int(_) => Ok(val),
                    Value::Bool(b) => Ok(Value::Int(if b { 1 } else { 0 })),
                    _ => Err(format!("cannot cast {} to {}", val, target)),
                },
                "bool" => match val {
                    Value::Int(n) => Ok(Value::Bool(n != 0)),
                    Value::Float(f) => Ok(Value::Bool(f != 0.0)),
                    Value::Bool(_) => Ok(val),
                    _ => Err(format!("cannot cast {} to bool", val)),
                },
                "string" | "String" => Ok(Value::String(format!("{}", val))),
                _ => Ok(val),
            }
        }

        ExprKind::TypeCall { ty, method, args } => {
            Err("type method calls not yet supported in interpreter".to_string())
        }

        ExprKind::StructLiteral { name, fields } => {
            let mut field_map = HashMap::new();
            for (fname, fexpr) in fields {
                let val = eval_expr(env, fexpr)?;
                field_map.insert(fname.name.clone(), val);
            }
            Ok(Value::Struct {
                name: name.name.clone(),
                fields: field_map,
            })
        }

        ExprKind::Match { expr: match_expr, arms } => {
            let val = eval_expr(env, match_expr)?;
            for arm in arms {
                if pattern_matches(&arm.pattern, &val) {
                    env.push_scope();
                    bind_pattern(env, &arm.pattern, &val);
                    let result = eval_expr(env, &arm.body)?;
                    env.pop_scope();
                    return Ok(result);
                }
            }
            Err(format!("no matching arm for value: {}", val))
        }

        ExprKind::Closure { params, body } => {
            // Capture the current environment
            let mut captured = HashMap::new();
            for scope in env.scopes.iter() {
                for (k, v) in scope {
                    captured.insert(k.clone(), v.clone());
                }
            }
            let param_names: Vec<String> = params.iter().map(|p| p.name.name.clone()).collect();
            Ok(Value::Closure {
                params: param_names,
                body: (**body).clone(),
                env: captured,
            })
        }

        ExprKind::Try(inner) => {
            let val = eval_expr(env, inner)?;
            match val {
                Value::Option(Some(v)) => Ok(*v),
                Value::Option(None) => Ok(Value::Return(Box::new(Value::Option(None)))),
                Value::Result(Ok(v)) => Ok(*v),
                Value::Result(Err(e)) => Ok(Value::Return(Box::new(Value::Result(Err(e))))),
                _ => Err("? operator requires Option or Result value".to_string()),
            }
        }
    }
}

fn call_closure(env: &mut Env, params: &[String], body: &Expr, captured_env: &HashMap<String, Value>, arg_vals: Vec<Value>) -> Result<Value, String> {
    env.push_scope();
    // Load captured environment
    for (k, v) in captured_env {
        env.define(k, v.clone());
    }
    // Bind parameters
    for (param, val) in params.iter().zip(arg_vals.iter()) {
        env.define(param, val.clone());
    }
    let result = eval_expr(env, body)?;
    env.pop_scope();
    match result {
        Value::Return(v) => Ok(*v),
        other => Ok(other),
    }
}

fn pattern_matches(pattern: &Pattern, value: &Value) -> bool {
    match pattern {
        Pattern::Wildcard => true,
        Pattern::Ident(id) => {
            // If the value is an enum variant and the ident matches a variant name,
            // treat this as a variant match (not a binding)
            if let Value::EnumVariant { variant, fields, .. } = value {
                if id.name == *variant && fields.is_empty() {
                    return true; // Unit variant match
                }
                // If the ident doesn't match any variant name, it's a binding
                // (but we can't distinguish perfectly without type info, so
                // we check: if the ident starts with uppercase, treat as variant match)
                if id.name.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                    return id.name == *variant;
                }
            }
            true // Regular identifier binding
        }
        Pattern::Literal(expr) => {
            // Compare literal values
            match &expr.kind {
                ExprKind::IntLiteral(n) => {
                    if let Value::Int(v) = value { *v == *n as i128 } else { false }
                }
                ExprKind::FloatLiteral(n) => {
                    if let Value::Float(v) = value { *v == *n } else { false }
                }
                ExprKind::StringLiteral(s) => {
                    if let Value::String(v) = value { v == s } else { false }
                }
                ExprKind::BoolLiteral(b) => {
                    if let Value::Bool(v) = value { *v == *b } else { false }
                }
                _ => false,
            }
        }
        Pattern::Variant { name, fields } => {
            if let Value::EnumVariant { variant, fields: val_fields, .. } = value {
                if variant == &name.name {
                    // Match inner patterns
                    if fields.is_empty() {
                        return true;
                    }
                    if fields.len() != val_fields.len() {
                        return false;
                    }
                    return fields.iter().zip(val_fields.iter())
                        .all(|(pat, val)| pattern_matches(pat, val));
                }
            }
            // Also check against unit variant identifiers
            if let Value::EnumVariant { variant, fields: val_fields, .. } = value {
                if variant == &name.name && fields.is_empty() && val_fields.is_empty() {
                    return true;
                }
            }
            false
        }
    }
}

fn bind_pattern(env: &mut Env, pattern: &Pattern, value: &Value) {
    match pattern {
        Pattern::Ident(id) => {
            // Don't bind uppercase idents that are enum variant matches
            if let Value::EnumVariant { variant, .. } = value {
                if id.name == *variant {
                    return; // This was a variant match, not a binding
                }
            }
            env.define(&id.name, value.clone());
        }
        Pattern::Variant { name: _, fields } => {
            if let Value::EnumVariant { fields: val_fields, .. } = value {
                for (pat, val) in fields.iter().zip(val_fields.iter()) {
                    bind_pattern(env, pat, val);
                }
            }
        }
        Pattern::Wildcard | Pattern::Literal(_) => {}
    }
}

fn eval_binop(lhs: &Value, op: BinOp, rhs: &Value) -> Result<Value, String> {
    match op {
        BinOp::Add => value_add(lhs, rhs),
        BinOp::Sub => value_sub(lhs, rhs),
        BinOp::Mul => value_mul(lhs, rhs),
        BinOp::Div => value_div(lhs, rhs),
        BinOp::Mod => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a % b)),
                _ => Err("% requires integers".to_string()),
            }
        }
        BinOp::Pow => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a.pow(*b as u32))),
                (Value::Float(a), Value::Float(b)) => Ok(Value::Float(a.powf(*b))),
                (Value::FieldElem(a), Value::Int(b)) => {
                    let exp_hex = format!("{:x}", b);
                    let exp = crypto::BigUint256::from_hex(&exp_hex).unwrap_or(crypto::BigUint256::ZERO);
                    Ok(Value::FieldElem(a.pow(&exp)))
                }
                _ => Err("** requires numeric types".to_string()),
            }
        }
        BinOp::Eq => Ok(Value::Bool(values_equal(lhs, rhs))),
        BinOp::NotEq => Ok(Value::Bool(!values_equal(lhs, rhs))),
        BinOp::Lt => value_cmp(lhs, rhs, |a, b| a < b, |a, b| a < b),
        BinOp::Gt => value_cmp(lhs, rhs, |a, b| a > b, |a, b| a > b),
        BinOp::LtEq => value_cmp(lhs, rhs, |a, b| a <= b, |a, b| a <= b),
        BinOp::GtEq => value_cmp(lhs, rhs, |a, b| a >= b, |a, b| a >= b),
        BinOp::And => {
            match (lhs, rhs) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a && *b)),
                _ => Err("&& requires booleans".to_string()),
            }
        }
        BinOp::Or => {
            match (lhs, rhs) {
                (Value::Bool(a), Value::Bool(b)) => Ok(Value::Bool(*a || *b)),
                _ => Err("|| requires booleans".to_string()),
            }
        }
        BinOp::BitAnd => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a & b)),
                _ => Err("& requires integers".to_string()),
            }
        }
        BinOp::BitOr => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a | b)),
                _ => Err("| requires integers".to_string()),
            }
        }
        BinOp::BitXor => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a ^ b)),
                _ => Err("^ requires integers".to_string()),
            }
        }
        BinOp::Shl => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a << (*b as u32))),
                _ => Err("<< requires integers".to_string()),
            }
        }
        BinOp::Shr => {
            match (lhs, rhs) {
                (Value::Int(a), Value::Int(b)) => Ok(Value::Int(a >> (*b as u32))),
                _ => Err(">> requires integers".to_string()),
            }
        }
        BinOp::ElemMul => {
            elementwise_op(lhs, rhs, |a, b| a * b, ".*")
        }
        BinOp::ElemDiv => {
            elementwise_op(lhs, rhs, |a, b| a / b, "./")
        }
    }
}

fn value_add(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x + y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x + y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 + y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x + *y as f64)),
        (Value::String(x), Value::String(y)) => Ok(Value::String(format!("{}{}", x, y))),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.add(y))),
        _ => Err(format!("cannot add {} and {}", a, b)),
    }
}

fn value_sub(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x - y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x - y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 - y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x - *y as f64)),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.sub(y))),
        _ => Err(format!("cannot subtract {} and {}", a, b)),
    }
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x * y)),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.mul(y))),
        _ => Err(format!("cannot multiply {} and {}", a, b)),
    }
}

fn value_div(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => {
            if *y == 0 { return Err("division by zero".to_string()); }
            Ok(Value::Int(x / y))
        }
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x / y)),
        (Value::FieldElem(x), Value::FieldElem(y)) => {
            if y.is_zero() { return Err("division by zero in field".to_string()); }
            Ok(Value::FieldElem(x.mul(&y.inv())))
        }
        _ => Err(format!("cannot divide {} and {}", a, b)),
    }
}

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(i) => Ok(*i as f64),
        _ => Err(format!("cannot convert {} to float", v)),
    }
}

fn value_to_float(v: &Value) -> Result<f64, String> {
    value_to_f64(v)
}

fn matmul_values(a: &Value, b: &Value) -> Result<Value, String> {
    let a_rows = match a {
        Value::Array(rows) => rows,
        _ => return Err("matmul: left operand must be a 2D array".to_string()),
    };
    let b_rows = match b {
        Value::Array(rows) => rows,
        _ => return Err("matmul: right operand must be a 2D array".to_string()),
    };
    if a_rows.is_empty() || b_rows.is_empty() {
        return Ok(Value::Array(vec![]));
    }
    let m = a_rows.len();
    let k = match &a_rows[0] {
        Value::Array(row) => row.len(),
        _ => return Err("matmul: left operand must be a 2D array".to_string()),
    };
    let k2 = b_rows.len();
    if k != k2 {
        return Err(format!("matmul: inner dimensions mismatch: {} vs {}", k, k2));
    }
    let n = match &b_rows[0] {
        Value::Array(row) => row.len(),
        _ => return Err("matmul: right operand must be a 2D array".to_string()),
    };
    let a_mat: Vec<Vec<f64>> = a_rows.iter().map(|row| {
        match row {
            Value::Array(r) => r.iter().map(|v| value_to_f64(v)).collect(),
            _ => Err("matmul: expected 2D array".to_string()),
        }
    }).collect::<Result<Vec<Vec<f64>>, String>>()?;
    let b_mat: Vec<Vec<f64>> = b_rows.iter().map(|row| {
        match row {
            Value::Array(r) => r.iter().map(|v| value_to_f64(v)).collect(),
            _ => Err("matmul: expected 2D array".to_string()),
        }
    }).collect::<Result<Vec<Vec<f64>>, String>>()?;
    let mut result = vec![vec![0.0f64; n]; m];
    for i in 0..m {
        for j in 0..n {
            for p in 0..k {
                result[i][j] += a_mat[i][p] * b_mat[p][j];
            }
        }
    }
    Ok(Value::Array(result.into_iter().map(|row| {
        Value::Array(row.into_iter().map(Value::Float).collect())
    }).collect()))
}

fn elementwise_op(lhs: &Value, rhs: &Value, op: fn(f64, f64) -> f64, op_name: &str) -> Result<Value, String> {
    match (lhs, rhs) {
        (Value::Array(a), Value::Array(b)) => {
            if a.len() != b.len() {
                return Err(format!("{}: array length mismatch: {} vs {}", op_name, a.len(), b.len()));
            }
            let results: Result<Vec<Value>, String> = a.iter().zip(b.iter()).map(|(x, y)| {
                elementwise_op(x, y, op, op_name)
            }).collect();
            Ok(Value::Array(results?))
        }
        _ => {
            let a = value_to_f64(lhs)?;
            let b = value_to_f64(rhs)?;
            Ok(Value::Float(op(a, b)))
        }
    }
}

fn value_to_int(v: &Value) -> Result<i128, String> {
    match v {
        Value::Int(n) => Ok(*n),
        Value::Float(n) => Ok(*n as i128),
        Value::Bool(b) => Ok(if *b { 1 } else { 0 }),
        _ => Err(format!("cannot convert {} to integer", v)),
    }
}

fn value_to_bool(v: &Value) -> Result<bool, String> {
    match v {
        Value::Bool(b) => Ok(*b),
        Value::Int(n) => Ok(*n != 0),
        _ => Err(format!("cannot convert {} to boolean", v)),
    }
}

fn values_equal(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => x == y,
        (Value::Float(x), Value::Float(y)) => x == y,
        (Value::Bool(x), Value::Bool(y)) => x == y,
        (Value::String(x), Value::String(y)) => x == y,
        (Value::FieldElem(x), Value::FieldElem(y)) => x == y,
        _ => false,
    }
}

fn value_cmp(
    a: &Value,
    b: &Value,
    int_op: fn(i128, i128) -> bool,
    float_op: fn(f64, f64) -> bool,
) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Bool(int_op(*x, *y))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Bool(float_op(*x, *y))),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Bool(float_op(*x as f64, *y))),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Bool(float_op(*x, *y as f64))),
        _ => Err(format!("cannot compare {} and {}", a, b)),
    }
}

// --- Sparse builtins ---

fn builtin_sparse_topk(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("sparse_topk expects 2 arguments: (values, k)".to_string());
    }
    let values = values_to_f64_vec(&args[0])?;
    let k = value_to_int(&args[1])? as usize;
    let idx = crate::sparse::sparse_topk(&values, k);
    Ok(Value::SparseIdx(idx))
}

fn builtin_sparse_gather(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("sparse_gather expects 2 arguments: (tensor, index)".to_string());
    }
    let dense = values_to_f64_vec(&args[0])?;
    let idx = match &args[1] {
        Value::SparseIdx(si) => si.clone(),
        _ => return Err("sparse_gather: second argument must be a SparseIndex".to_string()),
    };
    let gathered = crate::sparse::sparse_gather(&dense, &idx);
    let result: Vec<Value> = gathered.into_iter()
        .map(|batch| Value::Array(batch.into_iter().map(Value::Float).collect()))
        .collect();
    Ok(Value::Array(result))
}

fn builtin_sparse_scatter(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("sparse_scatter expects 3 arguments: (values, index, size)".to_string());
    }
    let sparse_vals: Vec<Vec<f64>> = match &args[0] {
        Value::Array(batches) => batches.iter().map(|b| values_to_f64_vec(b)).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("sparse_scatter: first argument must be array of arrays".to_string()),
    };
    let idx = match &args[1] {
        Value::SparseIdx(si) => si.clone(),
        _ => return Err("sparse_scatter: second argument must be a SparseIndex".to_string()),
    };
    let size = value_to_int(&args[2])? as usize;
    let result = crate::sparse::sparse_scatter(&sparse_vals, &idx, size);
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_sparse_matmul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("sparse_matmul expects 2 arguments: (sparse, dense)".to_string());
    }
    let sparse = match &args[0] {
        Value::SparseTensor(st) => st.clone(),
        _ => return Err("sparse_matmul: first argument must be a SparseTensor".to_string()),
    };
    let dense = values_to_f64_vec(&args[1])?;
    let cols = if sparse.shape.len() > 1 { sparse.shape[1] } else { 1 };
    let result = crate::sparse::sparse_matmul(&sparse, &dense, cols);
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_to_sparse(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("to_sparse expects 2 arguments: (tensor, threshold)".to_string());
    }
    let values = values_to_f64_vec(&args[0])?;
    let threshold = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("to_sparse: threshold must be numeric".to_string()),
    };
    let len = values.len();
    let st = crate::sparse::to_sparse(&values, vec![len], threshold);
    Ok(Value::SparseTensor(st))
}

// --- Local learning builtins ---

fn f64_vec_to_value(arr: &[f64]) -> Value {
    Value::Array(arr.iter().map(|x| Value::Float(*x)).collect())
}

fn builtin_goodness(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("goodness expects 1 argument".to_string()); }
    let activations = values_to_f64_vec(&args[0])?;
    Ok(Value::Float(crate::local_learn::goodness(&activations)))
}

fn builtin_hebbian_update(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("hebbian_update expects 6 arguments".to_string()); }
    let pre = values_to_f64_vec(&args[0])?;
    let post = values_to_f64_vec(&args[1])?;
    let mut weights = values_to_f64_vec(&args[2])?;
    let n_pre = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("n_pre must be int".into()) };
    let n_post = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("n_post must be int".into()) };
    let lr = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr must be numeric".into()) };
    crate::local_learn::hebbian_update(&pre, &post, &mut weights, n_pre, n_post, lr);
    Ok(f64_vec_to_value(&weights))
}

fn builtin_ff_layer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 8 { return Err("ff_layer expects 8 arguments".to_string()); }
    let pos_input = values_to_f64_vec(&args[0])?;
    let neg_input = values_to_f64_vec(&args[1])?;
    let mut weights = values_to_f64_vec(&args[2])?;
    let mut bias = values_to_f64_vec(&args[3])?;
    let n_in = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("n_in must be int".into()) };
    let n_out = match &args[5] { Value::Int(i) => *i as usize, _ => return Err("n_out must be int".into()) };
    let lr = match &args[6] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr must be numeric".into()) };
    let threshold = match &args[7] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("threshold must be numeric".into()) };
    let (pos_act, neg_act) = crate::local_learn::ff_layer_forward(&pos_input, &neg_input, &mut weights, &mut bias, n_in, n_out, lr, threshold);
    Ok(Value::Tuple(vec![f64_vec_to_value(&pos_act), f64_vec_to_value(&neg_act)]))
}

fn builtin_predictive_coding_update(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("predictive_coding_update expects 6 arguments".to_string()); }
    let prediction = values_to_f64_vec(&args[0])?;
    let target = values_to_f64_vec(&args[1])?;
    let mut weights = values_to_f64_vec(&args[2])?;
    let n_in = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("n_in must be int".into()) };
    let n_out = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("n_out must be int".into()) };
    let lr = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr must be numeric".into()) };
    let error = crate::local_learn::predictive_coding_update(&prediction, &target, &mut weights, n_in, n_out, lr);
    Ok(f64_vec_to_value(&error))
}

// --- ODE / Liquid NN builtins ---

fn builtin_ode_solve_euler(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 11 { return Err("ode_solve_euler expects 11 arguments".to_string()); }
    let y0 = values_to_f64_vec(&args[0])?;
    let t_start = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_start must be numeric".into()) };
    let t_end = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_end must be numeric".into()) };
    let steps = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("steps must be int".into()) };
    let _h = values_to_f64_vec(&args[4])?;
    let x = values_to_f64_vec(&args[5])?;
    let w_h = values_to_f64_vec(&args[6])?;
    let w_x = values_to_f64_vec(&args[7])?;
    let tau = values_to_f64_vec(&args[8])?;
    let n_hidden = match &args[9] { Value::Int(i) => *i as usize, _ => return Err("n_hidden must be int".into()) };
    let n_input = match &args[10] { Value::Int(i) => *i as usize, _ => return Err("n_input must be int".into()) };
    let f = move |_t: f64, state: &[f64]| -> Vec<f64> {
        crate::ode::liquid_cell(state, &x, &w_h, &w_x, &tau, n_hidden, n_input)
    };
    let result = crate::ode::euler_solve(&f, &y0, t_start, t_end, steps);
    Ok(f64_vec_to_value(&result))
}

fn builtin_ode_solve_rk4(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 11 { return Err("ode_solve_rk4 expects 11 arguments".to_string()); }
    let y0 = values_to_f64_vec(&args[0])?;
    let t_start = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_start must be numeric".into()) };
    let t_end = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("t_end must be numeric".into()) };
    let steps = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("steps must be int".into()) };
    let _h = values_to_f64_vec(&args[4])?;
    let x = values_to_f64_vec(&args[5])?;
    let w_h = values_to_f64_vec(&args[6])?;
    let w_x = values_to_f64_vec(&args[7])?;
    let tau = values_to_f64_vec(&args[8])?;
    let n_hidden = match &args[9] { Value::Int(i) => *i as usize, _ => return Err("n_hidden must be int".into()) };
    let n_input = match &args[10] { Value::Int(i) => *i as usize, _ => return Err("n_input must be int".into()) };
    let f = move |_t: f64, state: &[f64]| -> Vec<f64> {
        crate::ode::liquid_cell(state, &x, &w_h, &w_x, &tau, n_hidden, n_input)
    };
    let result = crate::ode::rk4_solve(&f, &y0, t_start, t_end, steps);
    Ok(f64_vec_to_value(&result))
}

fn builtin_liquid_cell(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 7 { return Err("liquid_cell expects 7 arguments".to_string()); }
    let h = values_to_f64_vec(&args[0])?;
    let x = values_to_f64_vec(&args[1])?;
    let w_h = values_to_f64_vec(&args[2])?;
    let w_x = values_to_f64_vec(&args[3])?;
    let tau = values_to_f64_vec(&args[4])?;
    let n_hidden = match &args[5] { Value::Int(i) => *i as usize, _ => return Err("n_hidden must be int".into()) };
    let n_input = match &args[6] { Value::Int(i) => *i as usize, _ => return Err("n_input must be int".into()) };
    let result = crate::ode::liquid_cell(&h, &x, &w_h, &w_x, &tau, n_hidden, n_input);
    Ok(f64_vec_to_value(&result))
}

fn builtin_cfc_cell(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 8 { return Err("cfc_cell expects 8 arguments".to_string()); }
    let h = values_to_f64_vec(&args[0])?;
    let x = values_to_f64_vec(&args[1])?;
    let w_h = values_to_f64_vec(&args[2])?;
    let w_x = values_to_f64_vec(&args[3])?;
    let tau = values_to_f64_vec(&args[4])?;
    let dt = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("dt must be numeric".into()) };
    let n_hidden = match &args[6] { Value::Int(i) => *i as usize, _ => return Err("n_hidden must be int".into()) };
    let n_input = match &args[7] { Value::Int(i) => *i as usize, _ => return Err("n_input must be int".into()) };
    let result = crate::ode::cfc_cell(&h, &x, &w_h, &w_x, &tau, dt, n_hidden, n_input);
    Ok(f64_vec_to_value(&result))
}

// --- SSM scan builtins ---

fn builtin_ssm_scan(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ssm_scan expects 3 arguments: (a, b, x)".to_string()); }
    let a = values_to_f64_vec(&args[0])?;
    let b = values_to_f64_vec(&args[1])?;
    let x = values_to_f64_vec(&args[2])?;
    let result = crate::ssm::sequential_scan(&a, &b, &x);
    Ok(f64_vec_to_value(&result))
}

fn builtin_parallel_scan(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("parallel_scan expects 3 arguments: (a, b, x)".to_string()); }
    let a = values_to_f64_vec(&args[0])?;
    let b = values_to_f64_vec(&args[1])?;
    let x = values_to_f64_vec(&args[2])?;
    let result = crate::ssm::parallel_scan(&a, &b, &x);
    Ok(f64_vec_to_value(&result))
}

fn builtin_selective_ssm(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("selective_ssm expects 6 arguments: (x, a_proj, b_proj, c_proj, d, delta)".to_string()); }
    let x = values_to_f64_vec(&args[0])?;
    let a_proj = values_to_f64_vec(&args[1])?;
    let b_proj = values_to_f64_vec(&args[2])?;
    let c_proj = values_to_f64_vec(&args[3])?;
    let d = match &args[4] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("d must be numeric".into()) };
    let delta = values_to_f64_vec(&args[5])?;
    let result = crate::ssm::selective_ssm(&x, &a_proj, &b_proj, &c_proj, d, &delta);
    Ok(f64_vec_to_value(&result))
}

// --- FFT builtins ---

fn builtin_fft(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("fft expects 1 argument: data (real array)".to_string()); }
    let data = values_to_f64_vec(&args[0])?;
    let result = crate::fft::rfft(&data);
    // Return as array of [re, im] pairs
    let pairs: Vec<Value> = result.iter().map(|c| {
        Value::Array(vec![Value::Float(c.re), Value::Float(c.im)])
    }).collect();
    Ok(Value::Array(pairs))
}

fn builtin_ifft(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("ifft expects 1 argument: complex array of [re, im] pairs".to_string()); }
    match &args[0] {
        Value::Array(pairs) => {
            let mut complex = Vec::new();
            for p in pairs {
                match p {
                    Value::Array(ri) if ri.len() == 2 => {
                        let re = match &ri[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected numeric".into()) };
                        let im = match &ri[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected numeric".into()) };
                        complex.push(crate::fft::Complex::new(re, im));
                    }
                    _ => return Err("ifft: expected array of [re, im] pairs".into()),
                }
            }
            let result = crate::fft::ifft(&complex);
            let pairs: Vec<Value> = result.iter().map(|c| {
                Value::Array(vec![Value::Float(c.re), Value::Float(c.im)])
            }).collect();
            Ok(Value::Array(pairs))
        }
        _ => Err("ifft expects an array".into()),
    }
}

fn builtin_fft_convolve(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("fft_convolve expects 2 arguments: (a, b)".to_string()); }
    let a = values_to_f64_vec(&args[0])?;
    let b = values_to_f64_vec(&args[1])?;
    let result = crate::fft::fft_convolve(&a, &b);
    Ok(f64_vec_to_value(&result))
}

fn builtin_fnet_mix(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("fnet_mix expects 3 arguments: (x, seq_len, d_model)".to_string()); }
    let x = values_to_f64_vec(&args[0])?;
    let seq_len = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("seq_len must be int".into()) };
    let d_model = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("d_model must be int".into()) };
    let mixer = crate::fft::FNetMixer::new(seq_len, d_model);
    let result = mixer.forward(&x);
    Ok(f64_vec_to_value(&result))
}

fn builtin_linear_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 { return Err("linear_attention expects 5+ arguments: (q, k, v, n, d, [feature_map])".to_string()); }
    let q = values_to_f64_vec(&args[0])?;
    let k = values_to_f64_vec(&args[1])?;
    let v = values_to_f64_vec(&args[2])?;
    let n = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("n must be int".into()) };
    let d = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("d must be int".into()) };
    let fm = crate::fft::FeatureMap::EluPlus1;
    let result = crate::fft::linear_attention(&q, &k, &v, n, d, fm);
    Ok(f64_vec_to_value(&result))
}

fn builtin_ssm_parallel_scan(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ssm_parallel_scan expects 2 arguments: (a, bx)".to_string()); }
    let a = values_to_f64_vec(&args[0])?;
    let bx = values_to_f64_vec(&args[1])?;
    let result = crate::fft::ssm_parallel_scan(&a, &bx);
    Ok(f64_vec_to_value(&result))
}

// --- DynTensor builtins ---

fn builtin_dyn_tensor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("dyn_tensor expects 1 argument: ragged array".to_string()); }
    // Input: array of arrays
    match &args[0] {
        Value::Array(rows) => {
            let mut ragged = Vec::new();
            for row in rows {
                match row {
                    Value::Array(elems) => {
                        let mut r = Vec::new();
                        for e in elems {
                            match e {
                                Value::Float(f) => r.push(*f),
                                Value::Int(i) => r.push(*i as f64),
                                _ => return Err("dyn_tensor: elements must be numeric".into()),
                            }
                        }
                        ragged.push(r);
                    }
                    _ => return Err("dyn_tensor: argument must be array of arrays".into()),
                }
            }
            let dt = crate::dyntensor::DynTensor::new(ragged);
            // Return as array of arrays (preserving ragged structure)
            let result: Vec<Value> = dt.data.iter().map(|row| {
                Value::Array(row.iter().map(|&v| Value::Float(v)).collect())
            }).collect();
            Ok(Value::Array(result))
        }
        _ => Err("dyn_tensor: argument must be an array".into()),
    }
}

fn builtin_compact(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("compact expects 2 arguments: (tensor, mask)".to_string()); }
    let rows = match &args[0] {
        Value::Array(rows) => {
            let mut ragged = Vec::new();
            for row in rows {
                let r = values_to_f64_vec(row)?;
                ragged.push(r);
            }
            ragged
        }
        _ => return Err("compact: first argument must be array of arrays".into()),
    };
    let mask = match &args[1] {
        Value::Array(elems) => {
            let mut m = Vec::new();
            for e in elems {
                match e {
                    Value::Bool(b) => m.push(*b),
                    _ => return Err("compact: mask elements must be booleans".into()),
                }
            }
            m
        }
        _ => return Err("compact: second argument must be boolean array".into()),
    };
    let dt = crate::dyntensor::DynTensor::new(rows);
    let compacted = dt.compact(&mask);
    let result: Vec<Value> = compacted.data.iter().map(|row| {
        Value::Array(row.iter().map(|&v| Value::Float(v)).collect())
    }).collect();
    Ok(Value::Array(result))
}

fn builtin_pad(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("pad expects 2 arguments: (dyn_tensor, pad_value)".to_string()); }
    let rows = match &args[0] {
        Value::Array(rows) => {
            let mut ragged = Vec::new();
            for row in rows {
                let r = values_to_f64_vec(row)?;
                ragged.push(r);
            }
            ragged
        }
        _ => return Err("pad: first argument must be array of arrays".into()),
    };
    let pad_val = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("pad: second argument must be numeric".into()) };
    let dt = crate::dyntensor::DynTensor::new(rows);
    let (padded, _lengths) = dt.to_padded(pad_val);
    Ok(f64_vec_to_value(&padded))
}

fn builtin_stream_compact(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("stream_compact expects 2 arguments: (data, mask)".to_string()); }
    let data = values_to_f64_vec(&args[0])?;
    let mask = match &args[1] {
        Value::Array(elems) => {
            let mut m = Vec::new();
            for e in elems {
                match e {
                    Value::Bool(b) => m.push(*b),
                    _ => return Err("stream_compact: mask elements must be booleans".into()),
                }
            }
            m
        }
        _ => return Err("stream_compact: second argument must be boolean array".into()),
    };
    if data.len() % mask.len() != 0 {
        return Err("stream_compact: data length must be a multiple of mask length".into());
    }
    let stride = data.len() / mask.len();
    let result = crate::dyntensor::stream_compact(&data, &mask, stride);
    Ok(f64_vec_to_value(&result))
}

// --- Option builtins ---

fn builtin_some(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("some expects 1 argument".to_string()); }
    Ok(Value::Option(Some(Box::new(args.into_iter().next().unwrap()))))
}

fn builtin_none(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Option(None))
}

fn builtin_unwrap(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("unwrap expects 1 argument".to_string()); }
    match &args[0] {
        Value::Option(Some(v)) => Ok(*v.clone()),
        Value::Option(None) => Err("unwrap called on None".to_string()),
        Value::Result(Ok(v)) => Ok(*v.clone()),
        Value::Result(Err(e)) => Err(format!("unwrap called on Err({})", e)),
        _ => Err("unwrap: argument must be Option or Result".to_string()),
    }
}

fn builtin_unwrap_or(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("unwrap_or expects 2 arguments".to_string()); }
    match &args[0] {
        Value::Option(Some(v)) => Ok(*v.clone()),
        Value::Option(None) => Ok(args[1].clone()),
        Value::Result(Ok(v)) => Ok(*v.clone()),
        Value::Result(Err(_)) => Ok(args[1].clone()),
        _ => Err("unwrap_or: first argument must be Option or Result".to_string()),
    }
}

fn builtin_is_some(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("is_some expects 1 argument".to_string()); }
    match &args[0] {
        Value::Option(Some(_)) => Ok(Value::Bool(true)),
        Value::Option(None) => Ok(Value::Bool(false)),
        _ => Err("is_some: argument must be Option".to_string()),
    }
}

fn builtin_is_none(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("is_none expects 1 argument".to_string()); }
    match &args[0] {
        Value::Option(None) => Ok(Value::Bool(true)),
        Value::Option(Some(_)) => Ok(Value::Bool(false)),
        _ => Err("is_none: argument must be Option".to_string()),
    }
}

// --- Result builtins ---

fn builtin_ok(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("ok expects 1 argument".to_string()); }
    Ok(Value::Result(Ok(Box::new(args.into_iter().next().unwrap()))))
}

fn builtin_err(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("err expects 1 argument".to_string()); }
    Ok(Value::Result(Err(Box::new(args.into_iter().next().unwrap()))))
}

fn builtin_is_ok(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("is_ok expects 1 argument".to_string()); }
    match &args[0] {
        Value::Result(Ok(_)) => Ok(Value::Bool(true)),
        Value::Result(Err(_)) => Ok(Value::Bool(false)),
        _ => Err("is_ok: argument must be Result".to_string()),
    }
}

fn builtin_is_err(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("is_err expects 1 argument".to_string()); }
    match &args[0] {
        Value::Result(Err(_)) => Ok(Value::Bool(true)),
        Value::Result(Ok(_)) => Ok(Value::Bool(false)),
        _ => Err("is_err: argument must be Result".to_string()),
    }
}

// --- String operation builtins ---

fn builtin_split(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("split expects 2 arguments: (string, delimiter)".to_string()); }
    let s = match &args[0] { Value::String(s) => s.clone(), _ => return Err("split: first argument must be a string".to_string()) };
    let delim = match &args[1] { Value::String(s) => s.clone(), _ => return Err("split: second argument must be a string".to_string()) };
    let parts: Vec<Value> = s.split(&delim).map(|p| Value::String(p.to_string())).collect();
    Ok(Value::Array(parts))
}

fn builtin_join(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("join expects 2 arguments: (array, delimiter)".to_string()); }
    let arr = match &args[0] {
        Value::Array(a) => a.iter().map(|v| format!("{}", v)).collect::<Vec<_>>(),
        _ => return Err("join: first argument must be an array".to_string()),
    };
    let delim = match &args[1] { Value::String(s) => s.clone(), _ => return Err("join: second argument must be a string".to_string()) };
    Ok(Value::String(arr.join(&delim)))
}

fn builtin_trim(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("trim expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => Ok(Value::String(s.trim().to_string())),
        _ => Err("trim: argument must be a string".to_string()),
    }
}

fn builtin_starts_with(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("starts_with expects 2 arguments".to_string()); }
    let s = match &args[0] { Value::String(s) => s, _ => return Err("starts_with: first argument must be a string".to_string()) };
    let prefix = match &args[1] { Value::String(s) => s, _ => return Err("starts_with: second argument must be a string".to_string()) };
    Ok(Value::Bool(s.starts_with(prefix.as_str())))
}

fn builtin_ends_with(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("ends_with expects 2 arguments".to_string()); }
    let s = match &args[0] { Value::String(s) => s, _ => return Err("ends_with: first argument must be a string".to_string()) };
    let suffix = match &args[1] { Value::String(s) => s, _ => return Err("ends_with: second argument must be a string".to_string()) };
    Ok(Value::Bool(s.ends_with(suffix.as_str())))
}

fn builtin_contains_str(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("contains_str expects 2 arguments".to_string()); }
    let s = match &args[0] { Value::String(s) => s, _ => return Err("contains_str: first argument must be a string".to_string()) };
    let substr = match &args[1] { Value::String(s) => s, _ => return Err("contains_str: second argument must be a string".to_string()) };
    Ok(Value::Bool(s.contains(substr.as_str())))
}

fn builtin_replace(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("replace expects 3 arguments: (string, from, to)".to_string()); }
    let s = match &args[0] { Value::String(s) => s.clone(), _ => return Err("replace: first argument must be a string".to_string()) };
    let from = match &args[1] { Value::String(s) => s.clone(), _ => return Err("replace: second argument must be a string".to_string()) };
    let to = match &args[2] { Value::String(s) => s.clone(), _ => return Err("replace: third argument must be a string".to_string()) };
    Ok(Value::String(s.replace(&from, &to)))
}

fn builtin_to_upper(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_upper expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => Ok(Value::String(s.to_uppercase())),
        _ => Err("to_upper: argument must be a string".to_string()),
    }
}

fn builtin_to_lower(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("to_lower expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => Ok(Value::String(s.to_lowercase())),
        _ => Err("to_lower: argument must be a string".to_string()),
    }
}

fn builtin_substr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("substr expects 3 arguments: (string, start, len)".to_string()); }
    let s = match &args[0] { Value::String(s) => s.clone(), _ => return Err("substr: first argument must be a string".to_string()) };
    let start = value_to_int(&args[1])? as usize;
    let len = value_to_int(&args[2])? as usize;
    let chars: Vec<char> = s.chars().collect();
    let end = (start + len).min(chars.len());
    let result: String = chars[start..end].iter().collect();
    Ok(Value::String(result))
}

fn builtin_char_at(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("char_at expects 2 arguments: (string, index)".to_string()); }
    let s = match &args[0] { Value::String(s) => s.clone(), _ => return Err("char_at: first argument must be a string".to_string()) };
    let idx = value_to_int(&args[1])? as usize;
    let chars: Vec<char> = s.chars().collect();
    if idx >= chars.len() {
        return Err(format!("char_at: index {} out of bounds (len {})", idx, chars.len()));
    }
    Ok(Value::String(chars[idx].to_string()))
}

fn builtin_parse_int(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("parse_int expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => match s.trim().parse::<i128>() {
            std::result::Result::Ok(n) => Ok(Value::Result(Ok(Box::new(Value::Int(n))))),
            std::result::Result::Err(e) => Ok(Value::Result(Err(Box::new(Value::String(format!("parse_int: {}", e)))))),
        },
        _ => Err("parse_int: argument must be a string".to_string()),
    }
}

fn builtin_parse_float(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("parse_float expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => match s.trim().parse::<f64>() {
            std::result::Result::Ok(n) => Ok(Value::Result(Ok(Box::new(Value::Float(n))))),
            std::result::Result::Err(e) => Ok(Value::Result(Err(Box::new(Value::String(format!("parse_float: {}", e)))))),
        },
        _ => Err("parse_float: argument must be a string".to_string()),
    }
}

fn builtin_string_len(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("string_len expects 1 argument".to_string()); }
    match &args[0] {
        Value::String(s) => Ok(Value::Int(s.chars().count() as i128)),
        _ => Err("string_len: argument must be a string".to_string()),
    }
}

// --- HashMap builtins ---

fn builtin_hashmap(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::HashMap(HashMap::new()))
}

fn builtin_hashmap_insert(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("hashmap_insert expects 3 arguments: (map, key, value)".to_string()); }
    let mut map = match &args[0] {
        Value::HashMap(m) => m.clone(),
        _ => return Err("hashmap_insert: first argument must be a HashMap".to_string()),
    };
    let key = match &args[1] {
        Value::String(s) => s.clone(),
        other => format!("{}", other),
    };
    map.insert(key, args[2].clone());
    Ok(Value::HashMap(map))
}

fn builtin_hashmap_get(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("hashmap_get expects 2 arguments: (map, key)".to_string()); }
    let map = match &args[0] {
        Value::HashMap(m) => m,
        _ => return Err("hashmap_get: first argument must be a HashMap".to_string()),
    };
    let key = match &args[1] {
        Value::String(s) => s.clone(),
        other => format!("{}", other),
    };
    match map.get(&key) {
        Some(v) => Ok(Value::Option(Some(Box::new(v.clone())))),
        None => Ok(Value::Option(None)),
    }
}

fn builtin_hashmap_remove(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("hashmap_remove expects 2 arguments: (map, key)".to_string()); }
    let mut map = match &args[0] {
        Value::HashMap(m) => m.clone(),
        _ => return Err("hashmap_remove: first argument must be a HashMap".to_string()),
    };
    let key = match &args[1] {
        Value::String(s) => s.clone(),
        other => format!("{}", other),
    };
    map.remove(&key);
    Ok(Value::HashMap(map))
}

fn builtin_hashmap_contains(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("hashmap_contains expects 2 arguments: (map, key)".to_string()); }
    let map = match &args[0] {
        Value::HashMap(m) => m,
        _ => return Err("hashmap_contains: first argument must be a HashMap".to_string()),
    };
    let key = match &args[1] {
        Value::String(s) => s.clone(),
        other => format!("{}", other),
    };
    Ok(Value::Bool(map.contains_key(&key)))
}

fn builtin_hashmap_keys(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("hashmap_keys expects 1 argument".to_string()); }
    let map = match &args[0] {
        Value::HashMap(m) => m,
        _ => return Err("hashmap_keys: argument must be a HashMap".to_string()),
    };
    let keys: Vec<Value> = map.keys().map(|k| Value::String(k.clone())).collect();
    Ok(Value::Array(keys))
}

fn builtin_hashmap_values(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("hashmap_values expects 1 argument".to_string()); }
    let map = match &args[0] {
        Value::HashMap(m) => m,
        _ => return Err("hashmap_values: argument must be a HashMap".to_string()),
    };
    let values: Vec<Value> = map.values().cloned().collect();
    Ok(Value::Array(values))
}

fn builtin_hashmap_len(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("hashmap_len expects 1 argument".to_string()); }
    let map = match &args[0] {
        Value::HashMap(m) => m,
        _ => return Err("hashmap_len: argument must be a HashMap".to_string()),
    };
    Ok(Value::Int(map.len() as i128))
}

// --- Higher-order array function builtins ---

fn apply_closure(env: &mut Env, closure: &Value, args: Vec<Value>) -> Result<Value, String> {
    match closure {
        Value::Closure { params, body, env: captured_env } => {
            call_closure(env, params, body, captured_env, args)
        }
        Value::Function { params, body, .. } => {
            env.push_scope();
            for (param, val) in params.iter().zip(args.iter()) {
                env.define(param, val.clone());
            }
            let result = eval_block(env, body)?;
            env.pop_scope();
            match result {
                Value::Return(v) => Ok(*v),
                other => Ok(other),
            }
        }
        _ => Err("expected a closure or function".to_string()),
    }
}

fn builtin_map(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("map expects 2 arguments: (array, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("map: first argument must be an array".to_string()) };
    let closure = args[1].clone();
    let mut result = Vec::new();
    for elem in arr { result.push(apply_closure(env, &closure, vec![elem])?); }
    Ok(Value::Array(result))
}

fn builtin_filter(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("filter expects 2 arguments: (array, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("filter: first argument must be an array".to_string()) };
    let closure = args[1].clone();
    let mut result = Vec::new();
    for elem in arr { if value_to_bool(&apply_closure(env, &closure, vec![elem.clone()])?)? { result.push(elem); } }
    Ok(Value::Array(result))
}

fn builtin_fold(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("fold expects 3 arguments: (array, init, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("fold: first argument must be an array".to_string()) };
    let mut acc = args[1].clone();
    let closure = args[2].clone();
    for elem in arr { acc = apply_closure(env, &closure, vec![acc, elem])?; }
    Ok(acc)
}

fn builtin_zip(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("zip expects 2 arguments".to_string()); }
    let a1 = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("zip: arguments must be arrays".to_string()) };
    let a2 = match &args[1] { Value::Array(a) => a.clone(), _ => return Err("zip: arguments must be arrays".to_string()) };
    Ok(Value::Array(a1.into_iter().zip(a2).map(|(a, b)| Value::Tuple(vec![a, b])).collect()))
}

fn builtin_enumerate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("enumerate expects 1 argument".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("enumerate: argument must be an array".to_string()) };
    Ok(Value::Array(arr.into_iter().enumerate().map(|(i, v)| Value::Tuple(vec![Value::Int(i as i128), v])).collect()))
}

fn builtin_sort(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sort expects 1 argument".to_string()); }
    let mut arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("sort: argument must be an array".to_string()) };
    arr.sort_by(|a, b| match (a, b) { (Value::Int(x), Value::Int(y)) => x.cmp(y), (Value::Float(x), Value::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal), _ => std::cmp::Ordering::Equal });
    Ok(Value::Array(arr))
}

fn builtin_reverse(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("reverse expects 1 argument".to_string()); }
    let mut arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("reverse: argument must be an array".to_string()) };
    arr.reverse();
    Ok(Value::Array(arr))
}

fn builtin_sum(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("sum expects 1 argument".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("sum: argument must be an array".to_string()) };
    if arr.is_empty() { return Ok(Value::Int(0)); }
    let mut acc = arr[0].clone();
    for elem in &arr[1..] { acc = eval_binop(&acc, crate::ast::BinOp::Add, elem)?; }
    Ok(acc)
}

fn builtin_any(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("any expects 2 arguments: (array, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("any: first argument must be an array".to_string()) };
    let closure = args[1].clone();
    for elem in arr { if value_to_bool(&apply_closure(env, &closure, vec![elem])?)? { return Ok(Value::Bool(true)); } }
    Ok(Value::Bool(false))
}

fn builtin_all(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("all expects 2 arguments: (array, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("all: first argument must be an array".to_string()) };
    let closure = args[1].clone();
    for elem in arr { if !value_to_bool(&apply_closure(env, &closure, vec![elem])?)? { return Ok(Value::Bool(false)); } }
    Ok(Value::Bool(true))
}

fn builtin_flat_map(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("flat_map expects 2 arguments: (array, closure)".to_string()); }
    let arr = match &args[0] { Value::Array(a) => a.clone(), _ => return Err("flat_map: first argument must be an array".to_string()) };
    let closure = args[1].clone();
    let mut result = Vec::new();
    for elem in arr { match apply_closure(env, &closure, vec![elem])? { Value::Array(inner) => result.extend(inner), other => result.push(other) } }
    Ok(Value::Array(result))
}

// --- File I/O builtins ---

fn builtin_read_file(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("read_file expects 1 argument: (path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("read_file: path must be a string".into()) };
    match std::fs::read_to_string(&path) {
        Ok(content) => Ok(Value::Result(Ok(Box::new(Value::String(content))))),
        Err(e) => Ok(Value::Result(Err(Box::new(Value::String(e.to_string()))))),
    }
}

fn builtin_write_file(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("write_file expects 2 arguments: (path, content)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("write_file: path must be a string".into()) };
    let content = match &args[1] { Value::String(s) => s.clone(), _ => return Err("write_file: content must be a string".into()) };
    match std::fs::write(&path, &content) {
        Ok(()) => Ok(Value::Result(Ok(Box::new(Value::Void)))),
        Err(e) => Ok(Value::Result(Err(Box::new(Value::String(e.to_string()))))),
    }
}

fn builtin_append_file(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("append_file expects 2 arguments: (path, content)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("append_file: path must be a string".into()) };
    let content = match &args[1] { Value::String(s) => s.clone(), _ => return Err("append_file: content must be a string".into()) };
    use std::io::Write;
    match std::fs::OpenOptions::new().append(true).create(true).open(&path) {
        Ok(mut f) => match f.write_all(content.as_bytes()) {
            Ok(()) => Ok(Value::Result(Ok(Box::new(Value::Void)))),
            Err(e) => Ok(Value::Result(Err(Box::new(Value::String(e.to_string()))))),
        },
        Err(e) => Ok(Value::Result(Err(Box::new(Value::String(e.to_string()))))),
    }
}

fn builtin_file_exists(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("file_exists expects 1 argument: (path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("file_exists: path must be a string".into()) };
    Ok(Value::Bool(std::path::Path::new(&path).exists()))
}

fn builtin_read_lines(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("read_lines expects 1 argument: (path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("read_lines: path must be a string".into()) };
    match std::fs::read_to_string(&path) {
        Ok(content) => {
            let lines: Vec<Value> = content.lines().map(|l| Value::String(l.to_string())).collect();
            Ok(Value::Array(lines))
        }
        Err(e) => Err(format!("read_lines: {}", e)),
    }
}

fn builtin_read_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("read_bytes expects 1 argument: (path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("read_bytes: path must be a string".into()) };
    match std::fs::read(&path) {
        Ok(bytes) => {
            let vals: Vec<Value> = bytes.into_iter().map(|b| Value::Int(b as i128)).collect();
            Ok(Value::Array(vals))
        }
        Err(e) => Err(format!("read_bytes: {}", e)),
    }
}

fn builtin_write_bytes(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("write_bytes expects 2 arguments: (path, bytes)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("write_bytes: path must be a string".into()) };
    let bytes = match &args[1] {
        Value::Array(arr) => {
            let mut out = Vec::new();
            for v in arr {
                match v { Value::Int(n) => out.push(*n as u8), _ => return Err("write_bytes: array must contain integers".into()) }
            }
            out
        }
        _ => return Err("write_bytes: second argument must be an array".into()),
    };
    match std::fs::write(&path, &bytes) {
        Ok(()) => Ok(Value::Result(Ok(Box::new(Value::Void)))),
        Err(e) => Ok(Value::Result(Err(Box::new(Value::String(e.to_string()))))),
    }
}

// --- Automatic differentiation builtins ---

fn builtin_tape_new(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let id = env.next_tape_id;
    env.next_tape_id += 1;
    env.tapes.insert(id, autodiff::Tape::new());
    Ok(Value::Int(id as i128))
}

fn get_tape_and_ints(env: &mut Env, args: &[Value], name: &str, expected: usize) -> Result<(usize, Vec<usize>), String> {
    if args.len() != expected { return Err(format!("{} expects {} arguments", name, expected)); }
    let tape_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err(format!("{}: first arg must be tape id", name)) };
    if !env.tapes.contains_key(&tape_id) { return Err(format!("{}: invalid tape id", name)); }
    let ints: Vec<usize> = args[1..].iter().map(|v| match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Float(f) => Ok(*f as usize),
        _ => Err(format!("{}: arguments must be integers", name)),
    }).collect::<Result<_, _>>()?;
    Ok((tape_id, ints))
}

fn builtin_tape_var(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tape_var expects 2 arguments: (tape, value)".into()); }
    let tape_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("tape_var: first arg must be tape id".into()) };
    let val = match &args[1] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("tape_var: second arg must be a number".into()) };
    let tape = env.tapes.get_mut(&tape_id).ok_or("tape_var: invalid tape id")?;
    let idx = tape.var(val);
    Ok(Value::Int(idx as i128))
}

macro_rules! tape_binop {
    ($name:ident, $method:ident, $label:expr) => {
        fn $name(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            let (tape_id, ints) = get_tape_and_ints(env, &args, $label, 3)?;
            let tape = env.tapes.get_mut(&tape_id).unwrap();
            let idx = tape.$method(ints[0], ints[1]);
            Ok(Value::Int(idx as i128))
        }
    };
}

tape_binop!(builtin_tape_add, add, "tape_add");
tape_binop!(builtin_tape_mul, mul, "tape_mul");
tape_binop!(builtin_tape_sub, sub, "tape_sub");
tape_binop!(builtin_tape_div, div, "tape_div");

macro_rules! tape_unaryop {
    ($name:ident, $method:ident, $label:expr) => {
        fn $name(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
            let (tape_id, ints) = get_tape_and_ints(env, &args, $label, 2)?;
            let tape = env.tapes.get_mut(&tape_id).unwrap();
            let idx = tape.$method(ints[0]);
            Ok(Value::Int(idx as i128))
        }
    };
}

tape_unaryop!(builtin_tape_exp, exp, "tape_exp");
tape_unaryop!(builtin_tape_log, log, "tape_log");
tape_unaryop!(builtin_tape_tanh, tanh, "tape_tanh");
tape_unaryop!(builtin_tape_relu, relu, "tape_relu");
tape_unaryop!(builtin_tape_sigmoid, sigmoid, "tape_sigmoid");
tape_unaryop!(builtin_tape_sin, sin, "tape_sin");
tape_unaryop!(builtin_tape_cos, cos, "tape_cos");

fn builtin_tape_backward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let (tape_id, ints) = get_tape_and_ints(env, &args, "tape_backward", 2)?;
    let tape = env.tapes.get_mut(&tape_id).unwrap();
    tape.backward(ints[0]);
    Ok(Value::Void)
}

fn builtin_tape_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let (tape_id, ints) = get_tape_and_ints(env, &args, "tape_grad", 2)?;
    let tape = env.tapes.get(&tape_id).unwrap();
    Ok(Value::Float(tape.grad(ints[0])))
}

fn builtin_tape_value(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let (tape_id, ints) = get_tape_and_ints(env, &args, "tape_value", 2)?;
    let tape = env.tapes.get(&tape_id).unwrap();
    Ok(Value::Float(tape.value(ints[0])))
}

fn builtin_ad_sgd_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ad_sgd_step expects 3 arguments: (params, grads, lr)".into()); }
    let mut params = values_to_f64_vec(&args[0])?;
    let grads = values_to_f64_vec(&args[1])?;
    let lr = match &args[2] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("ad_sgd_step: lr must be a number".into()) };
    autodiff::sgd_step(&mut params, &grads, lr);
    Ok(f64_vec_to_value(&params))
}

fn builtin_ad_adam_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 9 { return Err("ad_adam_step expects 9 arguments: (params, grads, m, v, lr, beta1, beta2, eps, t)".into()); }
    let mut params = values_to_f64_vec(&args[0])?;
    let grads = values_to_f64_vec(&args[1])?;
    let mut m = values_to_f64_vec(&args[2])?;
    let mut v = values_to_f64_vec(&args[3])?;
    let lr = match &args[4] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("lr must be a number".into()) };
    let beta1 = match &args[5] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("beta1 must be a number".into()) };
    let beta2 = match &args[6] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("beta2 must be a number".into()) };
    let eps = match &args[7] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("eps must be a number".into()) };
    let t = match &args[8] { Value::Int(n) => *n as usize, Value::Float(f) => *f as usize, _ => return Err("t must be an integer".into()) };
    autodiff::adam_step(&mut params, &grads, &mut m, &mut v, lr, beta1, beta2, eps, t);
    Ok(Value::Tuple(vec![f64_vec_to_value(&params), f64_vec_to_value(&m), f64_vec_to_value(&v)]))
}

fn builtin_ad_mse_loss(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ad_mse_loss expects 3 arguments: (tape_id, predictions, targets)".into()); }
    let tape_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("ad_mse_loss: first arg must be tape id".into()) };
    let preds: Vec<usize> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v { Value::Int(n) => Ok(*n as usize), _ => Err("predictions must be ints".to_string()) }).collect::<Result<_,_>>()?,
        _ => return Err("ad_mse_loss: predictions must be an array".into()),
    };
    let targets = values_to_f64_vec(&args[2])?;
    let tape = env.tapes.get_mut(&tape_id).ok_or("ad_mse_loss: invalid tape id")?;
    let loss_idx = autodiff::mse_loss(tape, &preds, &targets);
    Ok(Value::Int(loss_idx as i128))
}

fn builtin_ad_cross_entropy_loss(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ad_cross_entropy_loss expects 3 arguments: (tape_id, logits, target_idx)".into()); }
    let tape_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("first arg must be tape id".into()) };
    let logits: Vec<usize> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v { Value::Int(n) => Ok(*n as usize), _ => Err("logits must be ints".to_string()) }).collect::<Result<_,_>>()?,
        _ => return Err("logits must be an array".into()),
    };
    let target_idx = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("target_idx must be an int".into()) };
    let tape = env.tapes.get_mut(&tape_id).ok_or("invalid tape id")?;
    let loss_idx = autodiff::cross_entropy_loss(tape, &logits, target_idx);
    Ok(Value::Int(loss_idx as i128))
}

fn builtin_rk45_solve(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("rk45_solve expects 6 args: (f, y0, t_start, t_end, rtol, atol)".into()); }
    let f_closure = args[0].clone();
    let y0: Vec<f64> = match &args[1] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("rk45_solve: y0 must be array".into()) };
    let t_start = value_to_float(&args[2])?;
    let t_end = value_to_float(&args[3])?;
    let rtol = value_to_float(&args[4])?;
    let atol = value_to_float(&args[5])?;
    let call_f = |e: &mut Env, t: f64, y: &[f64]| -> Result<Vec<f64>, String> {
        let r = apply_closure(e, &f_closure, vec![Value::Float(t), Value::Array(y.iter().map(|&v| Value::Float(v)).collect())])?;
        match r { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>(), _ => Err("f must return array".into()) }
    };
    let n = y0.len(); let mut y = y0; let mut t = t_start;
    let mut h = (t_end - t_start) / 100.0; let h_min = 1e-12; let h_max = t_end - t_start;
    for _ in 0..100_000 {
        if t >= t_end { break; }
        if t + h > t_end { h = t_end - t; }
        let k1 = call_f(env, t, &y)?;
        let y2: Vec<f64> = (0..n).map(|i| y[i] + h*0.2*k1[i]).collect();
        let k2 = call_f(env, t+h*0.2, &y2)?;
        let y3: Vec<f64> = (0..n).map(|i| y[i] + h*(3.0/40.0*k1[i] + 9.0/40.0*k2[i])).collect();
        let k3 = call_f(env, t+0.3*h, &y3)?;
        let y4: Vec<f64> = (0..n).map(|i| y[i] + h*(44.0/45.0*k1[i] - 56.0/15.0*k2[i] + 32.0/9.0*k3[i])).collect();
        let k4 = call_f(env, t+0.8*h, &y4)?;
        let y5: Vec<f64> = (0..n).map(|i| y[i] + h*(19372.0/6561.0*k1[i] - 25360.0/2187.0*k2[i] + 64448.0/6561.0*k3[i] - 212.0/729.0*k4[i])).collect();
        let k5 = call_f(env, t+8.0/9.0*h, &y5)?;
        let y6: Vec<f64> = (0..n).map(|i| y[i] + h*(9017.0/3168.0*k1[i] - 355.0/33.0*k2[i] + 46732.0/5247.0*k3[i] + 49.0/176.0*k4[i] - 5103.0/18656.0*k5[i])).collect();
        let k6 = call_f(env, t+h, &y6)?;
        let y_new: Vec<f64> = (0..n).map(|i| y[i] + h*(35.0/384.0*k1[i] + 500.0/1113.0*k3[i] + 125.0/192.0*k4[i] - 2187.0/6784.0*k5[i] + 11.0/84.0*k6[i])).collect();
        let k7 = call_f(env, t+h, &y_new)?;
        let y_hat: Vec<f64> = (0..n).map(|i| y[i] + h*(5179.0/57600.0*k1[i] + 7571.0/16695.0*k3[i] + 393.0/640.0*k4[i] - 92097.0/339200.0*k5[i] + 187.0/2100.0*k6[i] + 1.0/40.0*k7[i])).collect();
        let err: f64 = (0..n).map(|i| { let sc = atol + rtol*y_new[i].abs().max(y[i].abs()); ((y_new[i]-y_hat[i])/sc).powi(2) }).sum::<f64>().sqrt() / (n as f64).sqrt();
        if err <= 1.0 || h <= h_min { t += h; y = y_new; }
        let factor = if err > 0.0 { 0.9 * err.powf(-0.2) } else { 5.0 };
        h *= factor.min(5.0).max(0.2); h = h.min(h_max).max(h_min);
    }
    Ok(Value::Array(y.into_iter().map(Value::Float).collect()))
}

fn builtin_spike_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("spike_attention expects 4 args: (q_spikes, k_spikes, v, d)".into()); }
    let q = match &args[0] { Value::SpikeTrain(st) => st.clone(), _ => return Err("q must be SpikeTrain".into()) };
    let k = match &args[1] { Value::SpikeTrain(st) => st.clone(), _ => return Err("k must be SpikeTrain".into()) };
    let v: Vec<f64> = match &args[2] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("v must be array".into()) };
    let d = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("d must be int".into()) };
    Ok(Value::Array(spiking::spike_attention(&q, &k, &v, d).into_iter().map(Value::Float).collect()))
}

fn builtin_oja_update(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("oja_update expects 6 args: (pre, post, weights, n_pre, n_post, lr)".into()); }
    let pre: Vec<f64> = match &args[0] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("pre must be array".into()) };
    let post: Vec<f64> = match &args[1] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("post must be array".into()) };
    let mut weights: Vec<f64> = match &args[2] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("weights must be array".into()) };
    let n_pre = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("n_pre must be int".into()) };
    let n_post = match &args[4] { Value::Int(n) => *n as usize, _ => return Err("n_post must be int".into()) };
    let lr = value_to_float(&args[5])?;
    local_learn::oja_update(&pre, &post, &mut weights, n_pre, n_post, lr);
    Ok(Value::Array(weights.into_iter().map(Value::Float).collect()))
}

fn builtin_chunked_scan(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 { return Err("chunked_scan expects 5 args: (x, a, b, c, chunk_size)".into()); }
    let x: Vec<f64> = match &args[0] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("x must be array".into()) };
    let a_arr: Vec<f64> = match &args[1] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("a must be array".into()) };
    let b: Vec<f64> = match &args[2] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("b must be array".into()) };
    let c: Vec<f64> = match &args[3] { Value::Array(a) => a.iter().map(value_to_float).collect::<Result<Vec<_>,_>>()?, _ => return Err("c must be array".into()) };
    let chunk_size = match &args[4] { Value::Int(n) => *n as usize, _ => return Err("chunk_size must be int".into()) };
    Ok(Value::Array(ssm::chunked_scan(&x, &a_arr, &b, &c, chunk_size).into_iter().map(Value::Float).collect()))
}

fn builtin_zero_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("zero_grad expects 1 arg: (tape_id)".into()); }
    let tape_id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("tape_id must be int".into()) };
    let tape = env.tapes.get_mut(&tape_id).ok_or("zero_grad: invalid tape id")?;
    tape.zero_grad();
    Ok(Value::Void)
}

fn builtin_modfield_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("modfield_new(prime_name, value_hex) requires 2 args".to_string());
    }
    let prime_name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("modfield_new: first arg must be a string (prime name)".to_string()),
    };
    let params = modmath::field_by_name(&prime_name)
        .ok_or_else(|| format!("modfield_new: unknown field '{}'", prime_name))?;
    let value_hex = match &args[1] {
        Value::String(s) => s.clone(),
        Value::Int(n) => format!("{:x}", n),
        _ => return Err("modfield_new: second arg must be a string (hex) or int".to_string()),
    };
    let elem = modmath::ModField::from_hex(&value_hex, params)
        .ok_or_else(|| format!("modfield_new: invalid hex value '{}'", value_hex))?;
    Ok(Value::ModFieldElem(elem))
}

fn builtin_modfield_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match (&args[0], &args[1]) {
        (Value::ModFieldElem(a), Value::ModFieldElem(b)) => Ok(Value::ModFieldElem(a.add(b))),
        _ => Err("modfield_add: requires two ModFieldElem args".to_string()),
    }
}

fn builtin_modfield_sub(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match (&args[0], &args[1]) {
        (Value::ModFieldElem(a), Value::ModFieldElem(b)) => Ok(Value::ModFieldElem(a.sub(b))),
        _ => Err("modfield_sub: requires two ModFieldElem args".to_string()),
    }
}

fn builtin_modfield_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match (&args[0], &args[1]) {
        (Value::ModFieldElem(a), Value::ModFieldElem(b)) => Ok(Value::ModFieldElem(a.mul(b))),
        _ => Err("modfield_mul: requires two ModFieldElem args".to_string()),
    }
}

fn builtin_modfield_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match &args[0] {
        Value::ModFieldElem(a) => Ok(Value::ModFieldElem(a.inv())),
        _ => Err("modfield_inv: requires a ModFieldElem arg".to_string()),
    }
}

fn builtin_modfield_pow(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("modfield_pow(elem, exp) requires 2 args".to_string());
    }
    let a = match &args[0] {
        Value::ModFieldElem(a) => a,
        _ => return Err("modfield_pow: first arg must be ModFieldElem".to_string()),
    };
    let exp = match &args[1] {
        Value::Int(n) => { let v = *n as u64; [v, 0, 0, 0] }
        Value::String(s) => {
            let tmp = modmath::ModField::from_hex(s, a.params)
                .ok_or("modfield_pow: invalid hex exponent")?;
            tmp.to_normal()
        }
        _ => return Err("modfield_pow: second arg must be int or hex string".to_string()),
    };
    Ok(Value::ModFieldElem(a.pow(&exp)))
}

fn builtin_modfield_neg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match &args[0] {
        Value::ModFieldElem(a) => Ok(Value::ModFieldElem(a.neg())),
        _ => Err("modfield_neg: requires a ModFieldElem arg".to_string()),
    }
}

fn builtin_modfield_sqrt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match &args[0] {
        Value::ModFieldElem(a) => {
            match a.sqrt() {
                Some(r) => Ok(Value::ModFieldElem(r)),
                None => Err("modfield_sqrt: not a quadratic residue".to_string()),
            }
        }
        _ => Err("modfield_sqrt: requires a ModFieldElem arg".to_string()),
    }
}

fn builtin_modfield_batch_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("modfield_batch_mul requires 2 array args".to_string()); }
    let a_vec = match &args[0] { Value::Array(v) => v, _ => return Err("modfield_batch_mul: first arg must be array".to_string()) };
    let b_vec = match &args[1] { Value::Array(v) => v, _ => return Err("modfield_batch_mul: second arg must be array".to_string()) };
    let a: Vec<modmath::ModField> = a_vec.iter().map(|v| match v {
        Value::ModFieldElem(m) => Ok(m.clone()), _ => Err("modfield_batch_mul: elements must be ModFieldElem".to_string()),
    }).collect::<Result<_, _>>()?;
    let b: Vec<modmath::ModField> = b_vec.iter().map(|v| match v {
        Value::ModFieldElem(m) => Ok(m.clone()), _ => Err("modfield_batch_mul: elements must be ModFieldElem".to_string()),
    }).collect::<Result<_, _>>()?;
    let result = modmath::batch_mul(&a, &b);
    Ok(Value::Array(result.into_iter().map(Value::ModFieldElem).collect()))
}

fn builtin_modfield_batch_inv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("modfield_batch_inv requires 1 array arg".to_string()); }
    let arr = match &args[0] { Value::Array(v) => v, _ => return Err("modfield_batch_inv: arg must be array".to_string()) };
    let elems: Vec<modmath::ModField> = arr.iter().map(|v| match v {
        Value::ModFieldElem(m) => Ok(m.clone()), _ => Err("modfield_batch_inv: elements must be ModFieldElem".to_string()),
    }).collect::<Result<_, _>>()?;
    let result = modmath::batch_inv(&elems);
    Ok(Value::Array(result.into_iter().map(Value::ModFieldElem).collect()))
}


#[cfg(test)]
mod interpreter_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }
    #[test]
    fn test_softmax_interpreter() {
        let o = rv("fn main() {
let x = [1.0, 2.0, 3.0]\nlet r = softmax(x)\nprintln(r)
}");
        assert!(!o.is_empty()); assert!(o[0].contains("0.0"), "{}", o[0]);
    }
    #[test]
    fn test_attention_interpreter() {
        let o = rv("fn main() {
let q = [1.0, 0.0]\nlet k = [1.0, 0.0]\nlet v = [0.5, 0.3]\nlet r = attention(q, k, v)\nprintln(r)
}");
        assert!(!o.is_empty()); assert!(o[0].contains("0.5"), "{}", o[0]);
    }

    #[test]
    fn test_dispatch_interpreter() {
        let o = rv("fn double(x: i64) -> i64 { return x * 2 }
fn triple(x: i64) -> i64 { return x * 3 }
fn main() {
let idx = 1
dispatch idx -> [double, triple](5)
let r = triple(5)
println(r)
}");
        assert!(!o.is_empty());
        assert!(o[0].contains("15"), "{}", o[0]);
    }

    #[test]
    fn test_option_some_none() {
        let o = rv("fn main() {
let x = some(42)
let y = none()
println(is_some(x))
println(is_none(y))
println(unwrap(x))
println(unwrap_or(y, 99))
}");
        assert_eq!(o, vec!["true", "true", "42", "99"]);
    }

    #[test]
    fn test_result_ok_err() {
        let o = rv("fn main() {
let x = ok(10)
let y = err(\"bad\")
println(is_ok(x))
println(is_err(y))
println(unwrap(x))
println(unwrap_or(y, 0))
}");
        assert_eq!(o, vec!["true", "true", "10", "0"]);
    }

    #[test]
    fn test_string_operations() {
        let o = rv("fn main() {
let parts = split(\"a,b,c\", \",\")
println(join(parts, \"-\"))
println(trim(\"  hello  \"))
println(starts_with(\"hello\", \"hel\"))
println(ends_with(\"hello\", \"llo\"))
println(contains_str(\"hello world\", \"world\"))
println(replace(\"foo bar\", \"bar\", \"baz\"))
println(to_upper(\"hello\"))
println(to_lower(\"HELLO\"))
println(substr(\"abcdef\", 1, 3))
println(char_at(\"abc\", 2))
println(string_len(\"hello\"))
}");
        assert_eq!(o, vec!["a-b-c", "hello", "true", "true", "true", "foo baz", "HELLO", "hello", "bcd", "c", "5"]);
    }

    #[test]
    fn test_hashmap_operations() {
        let o = rv("fn main() {
var m = hashmap()
m = hashmap_insert(m, \"a\", 1)
m = hashmap_insert(m, \"b\", 2)
println(hashmap_len(m))
println(hashmap_contains(m, \"a\"))
println(unwrap(hashmap_get(m, \"a\")))
println(is_none(hashmap_get(m, \"z\")))
m = hashmap_remove(m, \"a\")
println(hashmap_len(m))
}");
        assert_eq!(o, vec!["2", "true", "1", "true", "1"]);
    }

    #[test]
    fn test_try_operator() {
        let o = rv("fn maybe_get(x: i64) -> i64 {
if x > 0 { return ok(x * 10) }
return err(\"negative\")
}
fn process() -> i64 {
let a = maybe_get(5)?
let b = maybe_get(3)?
return ok(a + b)
}
fn main() {
let r = process()
println(is_ok(r))
println(unwrap(r))
}");
        assert_eq!(o, vec!["true", "80"]);
    }

    #[test]
    fn test_closure_capture() {
        let o = rv("fn main() {
let x = 10
let add_x = |y| y + x
let result = add_x(5)
println(result)
}");
        assert_eq!(o, vec!["15"]);
    }

    #[test]
    fn test_map_filter_fold() {
        let o = rv("fn main() {
let nums = [1, 2, 3, 4, 5]
let doubled = map(nums, |x| x * 2)
println(doubled)
let evens = filter(nums, |x| x % 2 == 0)
println(evens)
let total = fold(nums, 0, |acc, x| acc + x)
println(total)
}");
        assert_eq!(o, vec!["[2, 4, 6, 8, 10]", "[2, 4]", "15"]);
    }

    #[test]
    fn test_closure_as_argument() {
        let o = rv("fn apply(f: fn(i64) -> i64, x: i64) -> i64 {
return f(x)
}
fn main() {
let double = |x| x * 2
let result = apply(double, 21)
println(result)
}");
        assert_eq!(o, vec!["42"]);
    }
}

#[cfg(test)]
mod loop_and_builtin_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }

    #[test]
    fn test_loop_with_break() {
        let o = rv("fn main() {\nvar i = 0\nloop {\ni = i + 1\nif i == 5 {\nbreak\n}\n}\nprintln(i)\n}");
        assert_eq!(o, vec!["5"]);
    }

    #[test]
    fn test_loop_with_continue() {
        let o = rv("fn main() {\nvar sum = 0\nvar i = 0\nloop {\ni = i + 1\nif i > 10 {\nbreak\n}\nif i % 2 == 0 {\ncontinue\n}\nsum = sum + i\n}\nprintln(sum)\n}");
        assert_eq!(o, vec!["25"]);
    }

    #[test]
    fn test_spike_attention_builtin() {
        let o = rv("fn main() {\nlet q = spike_train(2, 3)\nlet k = spike_train(2, 3)\nlet v = [1.0, 2.0, 3.0, 4.0]\nlet result = spike_attention(q, k, v, 2)\nprintln(len(result))\n}");
        assert_eq!(o, vec!["4"]);
    }

    #[test]
    fn test_oja_update_builtin() {
        let o = rv("fn main() {\nlet pre = [1.0, 0.5]\nlet post = [0.8, 0.3]\nlet w = [0.1, 0.2, 0.3, 0.4]\nlet result = oja_update(pre, post, w, 2, 2, 0.01)\nprintln(len(result))\n}");
        assert_eq!(o, vec!["4"]);
    }

    #[test]
    fn test_chunked_scan_builtin() {
        let o = rv("fn main() {\nlet x = [1.0, 2.0, 3.0, 4.0]\nlet a = [0.9, 0.9, 0.9, 0.9]\nlet b = [1.0, 1.0, 1.0, 1.0]\nlet c = [1.0, 1.0, 1.0, 1.0]\nlet result = chunked_scan(x, a, b, c, 2)\nprintln(len(result))\n}");
        assert_eq!(o, vec!["4"]);
    }

    #[test]
    fn test_zero_grad_builtin() {
        let o = rv("fn main() {\nlet t = tape_new()\nlet x = tape_var(t, 3.0)\nlet y = tape_mul(t, x, x)\ntape_backward(t, y)\nlet g1 = tape_grad(t, x)\nzero_grad(t)\nlet g2 = tape_grad(t, x)\nprintln(g1)\nprintln(g2)\n}");
        assert_eq!(o, vec!["6", "0"]);
    }

    #[test]
    fn test_rk45_solve_builtin() {
        let o = rv("fn main() {\nlet f = |t: f64, y: [f64]| [-1.0 * y[0]]\nlet result = rk45_solve(f, [1.0], 0.0, 1.0, 0.001, 0.000001)\nprintln(result[0])\n}");
        assert!(!o.is_empty());
        let val: f64 = o[0].parse().unwrap();
        assert!((val - 0.3679).abs() < 0.01, "Expected ~0.3679, got {}", val);
    }
}

#[cfg(test)]
mod phase1_2_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }

    #[test]
    fn test_break_in_for_loop() {
        let o = rv("fn main() {\nlet sum = 0\nfor i in 0..10 {\nif i == 5 {\nbreak\n}\nsum = sum + i\n}\nprintln(sum)\n}");
        assert_eq!(o, vec!["10"]);
    }

    #[test]
    fn test_continue_in_for_loop() {
        let o = rv("fn main() {\nlet sum = 0\nfor i in 0..6 {\nif i == 3 {\ncontinue\n}\nsum = sum + i\n}\nprintln(sum)\n}");
        assert_eq!(o, vec!["12"]);
    }

    #[test]
    fn test_break_in_while_loop() {
        let o = rv("fn main() {\nlet i = 0\nwhile true {\nif i == 3 {\nbreak\n}\ni = i + 1\n}\nprintln(i)\n}");
        assert_eq!(o, vec!["3"]);
    }

    #[test]
    fn test_matmul_2x2() {
        let o = rv("fn main() {\nlet a = [[1.0, 2.0], [3.0, 4.0]]\nlet b = [[5.0, 6.0], [7.0, 8.0]]\nlet c = a @ b\nprintln(c)\n}");
        assert!(o[0].contains("19"));
        assert!(o[0].contains("50"));
    }

    #[test]
    fn test_elementwise_mul() {
        let o = rv("fn main() {\nlet a = [2.0, 3.0, 4.0]\nlet b = [5.0, 6.0, 7.0]\nlet c = a .* b\nprintln(c)\n}");
        assert!(o[0].contains("10"));
        assert!(o[0].contains("28"));
    }

    #[test]
    fn test_elementwise_div() {
        let o = rv("fn main() {\nlet a = [10.0, 20.0, 30.0]\nlet b = [2.0, 4.0, 5.0]\nlet c = a ./ b\nprintln(c)\n}");
        assert!(o[0].contains("5"));
        assert!(o[0].contains("6"));
    }

    #[test]
    fn test_cast_int_to_float() {
        let o = rv("fn main() {\nlet x = 42\nlet y = x as f64\nprintln(y)\n}");
        assert_eq!(o, vec!["42"]);
    }

    #[test]
    fn test_cast_float_to_int() {
        let o = rv("fn main() {\nlet x = 3.7\nlet y = x as i64\nprintln(y)\n}");
        assert_eq!(o, vec!["3"]);
    }

    #[test]
    fn test_multi_index_2d() {
        let o = rv("fn main() {\nlet m = [[10, 20, 30], [40, 50, 60]]\nlet v = m[1, 2]\nprintln(v)\n}");
        assert_eq!(o, vec!["60"]);
    }

    #[test]
    fn test_attention_2d() {
        let o = rv("fn main() {\nlet q = [[1.0, 0.0], [0.0, 1.0]]\nlet k = [[1.0, 0.0], [0.0, 1.0]]\nlet v = [[1.0, 2.0], [3.0, 4.0]]\nlet r = attention(q, k, v)\nprintln(r)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("["));
    }
}

// --- GPU runtime builtins ---

fn extract_shape(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|x| match x {
                Value::Int(n) => Ok(*n as usize),
                _ => Err("shape elements must be integers".to_string()),
            })
            .collect(),
        _ => Err("shape must be an array".to_string()),
    }
}

fn extract_f64_data(v: &Value) -> Result<Vec<f64>, String> {
    fn flatten(v: &Value, out: &mut Vec<f64>) -> Result<(), String> {
        match v {
            Value::Float(f) => { out.push(*f); Ok(()) }
            Value::Int(n) => { out.push(*n as f64); Ok(()) }
            Value::Array(arr) => {
                for x in arr { flatten(x, out)?; }
                Ok(())
            }
            _ => Err("expected numeric data".to_string()),
        }
    }
    let mut data = Vec::new();
    flatten(v, &mut data)?;
    Ok(data)
}

/// gpu_alloc(shape, dtype_str) -> buffer_id
fn builtin_gpu_alloc(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 { return Err("gpu_alloc expects (shape) or (shape, dtype)".to_string()); }
    let shape = extract_shape(&args[0])?;
    let id = env.gpu_rt.alloc(&shape, gpu_runtime::DType::F64);
    Ok(Value::Int(id as i128))
}

/// gpu_free(buffer_id)
fn builtin_gpu_free(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("gpu_free expects 1 argument".to_string()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("gpu_free: id must be int".to_string()) };
    env.gpu_rt.free(id);
    Ok(Value::Void)
}

/// gpu_matmul(a, b) -> buffer_id
fn builtin_gpu_matmul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("gpu_matmul expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let b = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let c = env.gpu_rt.matmul(a, b);
    Ok(Value::Int(c as i128))
}

/// gpu_add(a, b) -> buffer_id
fn builtin_gpu_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("gpu_add expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let b = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let c = env.gpu_rt.elementwise_add(a, b);
    Ok(Value::Int(c as i128))
}

/// gpu_mul(a, b) -> buffer_id
fn builtin_gpu_mul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("gpu_mul expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let b = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let c = env.gpu_rt.elementwise_mul(a, b);
    Ok(Value::Int(c as i128))
}

/// gpu_relu(a) -> buffer_id
fn builtin_gpu_relu(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("gpu_relu expects 1 argument".to_string()); }
    let a = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let c = env.gpu_rt.relu(a);
    Ok(Value::Int(c as i128))
}

/// gpu_softmax(a, axis) -> buffer_id
fn builtin_gpu_softmax(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 { return Err("gpu_softmax expects (buffer_id) or (buffer_id, axis)".to_string()); }
    let a = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let axis = if args.len() > 1 {
        match &args[1] { Value::Int(n) => *n as usize, _ => 0 }
    } else { 0 };
    let c = env.gpu_rt.softmax(a, axis);
    Ok(Value::Int(c as i128))
}

/// gpu_copy_to_host(buffer_id) -> Array of floats
fn builtin_gpu_copy_to_host(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("gpu_copy_to_host expects 1 argument".to_string()); }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int id".to_string()) };
    let data = env.gpu_rt.copy_data(id);
    Ok(Value::Array(data.into_iter().map(Value::Float).collect()))
}

/// gpu_copy_to_device(data, shape) -> buffer_id
fn builtin_gpu_copy_to_device(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("gpu_copy_to_device expects (data, shape)".to_string()); }
    let data = extract_f64_data(&args[0])?;
    let shape = extract_shape(&args[1])?;
    let n: usize = shape.iter().product();
    if data.len() != n { return Err(format!("data length {} != shape product {}", data.len(), n)); }
    let id = env.gpu_rt.alloc(&shape, gpu_runtime::DType::F64);
    env.gpu_rt.write_data(id, &data);
    Ok(Value::Int(id as i128))
}

// --- Quantization builtins ---

fn builtin_quantize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("quantize requires at least 3 args: data, shape, format".into());
    }
    let data = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("quantize: data must be array of numbers".to_string()),
        }).collect::<Result<Vec<f64>, _>>()?,
        _ => return Err("quantize: first arg must be array".into()),
    };
    let shape = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            _ => Err("quantize: shape must be array of ints".to_string()),
        }).collect::<Result<Vec<usize>, _>>()?,
        _ => return Err("quantize: second arg must be array (shape)".into()),
    };
    let format_str = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("quantize: third arg must be format string".into()),
    };
    let format = crate::quantize::QuantFormat::from_str(&format_str)
        .ok_or_else(|| format!("quantize: unknown format '{}'", format_str))?;
    let group_size = if args.len() > 3 {
        match &args[3] {
            Value::Int(i) if *i > 0 => Some(*i as usize),
            _ => None,
        }
    } else { None };

    let qt = crate::quantize::QuantTensor::quantize(&data, &shape, format, group_size);
    let mut fields = HashMap::new();
    fields.insert("format".to_string(), Value::String(format_str));
    fields.insert("shape".to_string(), Value::Array(shape.iter().map(|&s| Value::Int(s as i128)).collect()));
    fields.insert("compression_ratio".to_string(), Value::Float(qt.compression_ratio()));
    fields.insert("_data".to_string(), Value::Array(qt.data.iter().map(|&b| Value::Int(b as i128)).collect()));
    fields.insert("_scale".to_string(), Value::Float(qt.get_scale_pub()));
    fields.insert("_zero_point".to_string(), Value::Int(qt.get_zp_pub() as i128));
    Ok(Value::Struct { name: "QuantTensor".to_string(), fields })
}

fn builtin_dequantize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("dequantize requires 1 arg: qtensor".into());
    }
    let qt = value_to_quant_tensor(&args[0])?;
    let deq = qt.dequantize();
    Ok(Value::Array(deq.into_iter().map(Value::Float).collect()))
}

fn builtin_quantized_matmul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("quantized_matmul requires 2 args: a_q, b_q".into());
    }
    let a = value_to_quant_tensor(&args[0])?;
    let b = value_to_quant_tensor(&args[1])?;
    let result = a.matmul(&b);
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_compression_ratio(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("compression_ratio requires 1 arg: qtensor".into());
    }
    let qt = value_to_quant_tensor(&args[0])?;
    Ok(Value::Float(qt.compression_ratio()))
}

fn value_to_quant_tensor(val: &Value) -> Result<crate::quantize::QuantTensor, String> {
    match val {
        Value::Struct { name, fields } if name == "QuantTensor" => {
            let format_str = match fields.get("format") {
                Some(Value::String(s)) => s.clone(),
                _ => return Err("QuantTensor missing format field".into()),
            };
            let format = crate::quantize::QuantFormat::from_str(&format_str)
                .ok_or_else(|| format!("unknown format '{}'", format_str))?;
            let shape = match fields.get("shape") {
                Some(Value::Array(arr)) => arr.iter().map(|v| match v {
                    Value::Int(i) => Ok(*i as usize),
                    _ => Err("bad shape".to_string()),
                }).collect::<Result<Vec<usize>, _>>()?,
                _ => return Err("QuantTensor missing shape".into()),
            };
            let data = match fields.get("_data") {
                Some(Value::Array(arr)) => arr.iter().map(|v| match v {
                    Value::Int(i) => Ok(*i as u8),
                    _ => Err("bad data".to_string()),
                }).collect::<Result<Vec<u8>, _>>()?,
                _ => return Err("QuantTensor missing _data".into()),
            };
            let scale = match fields.get("_scale") {
                Some(Value::Float(f)) => *f,
                _ => 1.0,
            };
            let zp = match fields.get("_zero_point") {
                Some(Value::Int(i)) => *i as i64,
                _ => 0,
            };
            Ok(crate::quantize::QuantTensor {
                data,
                shape,
                format,
                scheme: crate::quantize::QuantScheme::PerTensor { scale, zero_point: zp },
                original_dtype: "f64".into(),
            })
        }
        _ => Err("expected QuantTensor struct".into()),
    }
}

// ── Tensor autodiff builtins ──────────────────────────────────────────

fn get_tensor_tape(env: &mut Env) -> Result<&mut tensor_autodiff::TensorTape, String> {
    env.tensor_tape.as_mut().ok_or_else(|| "No tensor tape active. Call tensor_tape_new() first.".to_string())
}

/// Flatten a potentially nested Value (array of arrays) into Vec<f64>
fn flatten_value_to_f64(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Float(f) => Ok(vec![*f]),
        Value::Int(i) => Ok(vec![*i as f64]),
        Value::Array(arr) => {
            let mut out = Vec::new();
            for item in arr {
                out.extend(flatten_value_to_f64(item)?);
            }
            Ok(out)
        }
        _ => Err("expected numeric value or array".to_string()),
    }
}

/// Extract shape from a Value::Array of ints
fn value_to_usize_vec(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(n) => Ok(*n as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("shape must contain integers".to_string()),
        }).collect(),
        _ => Err("shape must be an array".to_string()),
    }
}

fn value_to_tensor_id(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Float(f) => Ok(*f as usize),
        _ => Err("expected tensor id (integer)".to_string()),
    }
}

fn value_to_id_array(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(value_to_tensor_id).collect(),
        _ => Err("expected array of tensor ids".to_string()),
    }
}

fn builtin_tensor_tape_new(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    env.tensor_tape = Some(tensor_autodiff::TensorTape::new());
    Ok(Value::Void)
}

fn builtin_tensor_tape_clear(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    env.tensor_tape = Some(tensor_autodiff::TensorTape::new());
    Ok(Value::Void)
}

fn builtin_tensor_param(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_param expects 2 arguments: (data, shape)".into()); }
    let data = flatten_value_to_f64(&args[0])?;
    let shape = value_to_usize_vec(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.new_tensor(data, shape, true);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_input(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_input expects 2 arguments: (data, shape)".into()); }
    let data = flatten_value_to_f64(&args[0])?;
    let shape = value_to_usize_vec(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.new_tensor(data, shape, false);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_matmul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_matmul expects 2 arguments".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let b = value_to_tensor_id(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.matmul(a, b);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_add expects 2 arguments".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let b = value_to_tensor_id(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.add(a, b);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_mul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_mul expects 2 arguments".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let b = value_to_tensor_id(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.mul(a, b);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_sub(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_sub expects 2 arguments".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let b = value_to_tensor_id(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.sub(a, b);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_relu(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_relu expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.relu(a);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_sigmoid(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_sigmoid expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.sigmoid(a);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_tanh(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_tanh expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.tanh_op(a);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_gelu(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_gelu expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.gelu(a);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_softmax(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_softmax expects 2 arguments: (tensor_id, axis)".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let axis = value_to_tensor_id(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.softmax(a, axis);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_layer_norm(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("tensor_layer_norm expects 4 arguments: (x_id, gamma_id, beta_id, eps)".into()); }
    let x = value_to_tensor_id(&args[0])?;
    let gamma = value_to_tensor_id(&args[1])?;
    let beta = value_to_tensor_id(&args[2])?;
    let eps = value_to_f64(&args[3])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.layer_norm(x, gamma, beta, eps);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_cross_entropy(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_cross_entropy expects 2 arguments: (logits_id, targets)".into()); }
    let logits = value_to_tensor_id(&args[0])?;
    let targets = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(n) => Ok(*n as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("targets must be integers".to_string()),
        }).collect::<Result<Vec<usize>, String>>()?,
        _ => return Err("targets must be an array".into()),
    };
    let tape = get_tensor_tape(env)?;
    let id = tape.cross_entropy_loss(logits, targets);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_sum(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_sum expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.sum(a, None);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_mean(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_mean expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.mean(a, None);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_transpose(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_transpose expects 1 argument".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.transpose(a);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_reshape(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_reshape expects 2 arguments: (tensor_id, new_shape)".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let new_shape = value_to_usize_vec(&args[1])?;
    let tape = get_tensor_tape(env)?;
    let id = tape.reshape(a, new_shape);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_broadcast_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_broadcast_add expects 2 arguments".into()); }
    let a = value_to_tensor_id(&args[0])?;
    let b = value_to_tensor_id(&args[1])?;
    // broadcast_add is just add with broadcasting support (already handled by add)
    let tape = get_tensor_tape(env)?;
    let id = tape.add(a, b);
    Ok(Value::Int(id as i128))
}

fn builtin_tensor_backward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_backward expects 1 argument: (loss_id)".into()); }
    let loss = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    tape.backward(loss);
    Ok(Value::Void)
}

fn builtin_tensor_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_grad expects 1 argument: (param_id)".into()); }
    let id = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    match tape.grad(id) {
        Some(g) => Ok(f64_vec_to_value(g)),
        None => Ok(Value::Array(vec![])),
    }
}

fn builtin_tensor_data(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_data expects 1 argument: (tensor_id)".into()); }
    let id = value_to_tensor_id(&args[0])?;
    let tape = get_tensor_tape(env)?;
    let d = tape.data(id);
    Ok(f64_vec_to_value(d))
}

fn builtin_tensor_sgd(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_sgd expects 2 arguments: (param_ids, lr)".into()); }
    let params = value_to_id_array(&args[0])?;
    let lr = value_to_f64(&args[1])?;
    let tape = get_tensor_tape(env)?;
    tape.sgd_step(&params, lr);
    Ok(Value::Void)
}

fn builtin_tensor_adam(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 8 { return Err("tensor_adam expects 8 arguments: (param_ids, lr, beta1, beta2, eps, step, m_state_id, v_state_id)".into()); }
    let params = value_to_id_array(&args[0])?;
    let lr = value_to_f64(&args[1])?;
    let beta1 = value_to_f64(&args[2])?;
    let beta2 = value_to_f64(&args[3])?;
    let eps = value_to_f64(&args[4])?;
    let step = value_to_tensor_id(&args[5])?;
    let m_state_id = value_to_tensor_id(&args[6])?;
    let v_state_id = value_to_tensor_id(&args[7])?;

    // Get or create adam states
    let n_params = params.len();
    if !env.adam_states.contains_key(&m_state_id) {
        env.adam_states.insert(m_state_id, (vec![vec![]; n_params], vec![vec![]; n_params]));
    }
    let (ref mut m, ref mut v) = env.adam_states.get_mut(&m_state_id).unwrap();

    let tape = env.tensor_tape.as_mut().ok_or("No tensor tape active")?;
    tape.adam_step(&params, lr, beta1, beta2, eps, step, m, v);
    Ok(Value::Void)
}

fn builtin_tensor_zero_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_zero_grad expects 1 argument: (param_ids)".into()); }
    let params = value_to_id_array(&args[0])?;
    let tape = get_tensor_tape(env)?;
    tape.zero_grad(&params);
    Ok(Value::Void)
}

// ── SpikeSSMFormer builtins ─────────────────────────────────────────

/// spike_ssm_new(d_model, d_state, n_ff_layers, n_ssm_layers, vocab_size) -> model_id
fn builtin_spike_ssm_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 { return Err("spike_ssm_new expects 5 arguments: (d_model, d_state, n_ff_layers, n_ssm_layers, vocab_size)".into()); }
    let d_model = value_to_usize(&args[0])?;
    let d_state = value_to_usize(&args[1])?;
    let n_ff = value_to_usize(&args[2])?;
    let n_ssm = value_to_usize(&args[3])?;
    let vocab = value_to_usize(&args[4])?;
    let model = architectures::SpikeSSMFormer::new(d_model, d_state, n_ff, n_ssm, vocab);
    let id = env.next_spike_model_id;
    env.next_spike_model_id += 1;
    env.spike_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

/// spike_ssm_forward(model_id, token_ids, embeddings) -> logits (array of arrays)
fn builtin_spike_ssm_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("spike_ssm_forward expects 3 arguments: (model_id, token_ids, embeddings)".into()); }
    let model_id = value_to_usize(&args[0])?;
    let token_ids = value_to_usize_vec(&args[1])?;
    let embeddings = value_to_f64_2d(&args[2])?;
    let model = env.spike_models.get_mut(&model_id)
        .ok_or_else(|| format!("No SpikeSSMFormer model with id {}", model_id))?;
    let logits = model.forward(&token_ids, &embeddings);
    // Convert to Value::Array of Value::Array
    let result = logits.iter().map(|row| {
        Value::Array(row.iter().map(|&v| Value::Float(v)).collect())
    }).collect();
    Ok(Value::Array(result))
}

/// spike_ssm_train_step(model_id, token_ids, embeddings, targets, lr) -> loss
fn builtin_spike_ssm_train_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 { return Err("spike_ssm_train_step expects 5 arguments: (model_id, token_ids, embeddings, targets, lr)".into()); }
    let model_id = value_to_usize(&args[0])?;
    let token_ids = value_to_usize_vec(&args[1])?;
    let embeddings = value_to_f64_2d(&args[2])?;
    let targets = value_to_usize_vec(&args[3])?;
    let lr = value_to_f64(&args[4])?;
    let model = env.spike_models.get_mut(&model_id)
        .ok_or_else(|| format!("No SpikeSSMFormer model with id {}", model_id))?;
    let loss = model.train_step(&token_ids, &embeddings, &targets, lr);
    Ok(Value::Float(loss))
}

/// spike_ssm_stats(model_id) -> [total_params, attention_sparsity, ff_goodness, avg_tau]
fn builtin_spike_ssm_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("spike_ssm_stats expects 1 argument: (model_id)".into()); }
    let model_id = value_to_usize(&args[0])?;
    let model = env.spike_models.get(&model_id)
        .ok_or_else(|| format!("No SpikeSSMFormer model with id {}", model_id))?;
    let stats = model.stats();
    Ok(Value::Array(vec![
        Value::Int(stats.total_params as i128),
        Value::Float(stats.attention_sparsity),
        Value::Float(stats.ff_goodness),
        Value::Float(stats.avg_tau),
    ]))
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Float(f) => Ok(*f as usize),
        _ => Err(format!("Expected integer, got {:?}", v)),
    }
}

fn value_to_f64_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(rows) => {
            rows.iter().map(|row| {
                match row {
                    Value::Array(cols) => cols.iter().map(|c| match c {
                        Value::Float(f) => Ok(*f),
                        Value::Int(n) => Ok(*n as f64),
                        _ => Err("Expected number in 2D array".to_string()),
                    }).collect(),
                    _ => Err("Expected array of arrays for 2D data".to_string()),
                }
            }).collect()
        }
        _ => Err("Expected 2D array".to_string()),
    }
}

// --- Tiered MoE builtins ---

fn builtin_tiered_moe_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("tiered_moe_new expects 4 arguments: (n_experts, expert_width, n_clusters, hot_budget)".into()); }
    let n_experts = value_to_usize(&args[0])?;
    let expert_width = value_to_usize(&args[1])?;
    let n_clusters = value_to_usize(&args[2])?;
    let hot_budget = value_to_usize(&args[3])?;
    let input_dim = expert_width;
    let layer = tiered_experts::create_tiered_moe(n_experts, expert_width, n_clusters, hot_budget, input_dim);
    let id = env.next_tiered_moe_id;
    env.next_tiered_moe_id += 1;
    env.tiered_moe_layers.insert(id, layer);
    Ok(Value::Int(id as i128))
}

fn builtin_tiered_moe_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tiered_moe_forward expects 2 arguments: (id, input)".into()); }
    let id = value_to_usize(&args[0])?;
    let input = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(n) => Ok(*n as f64),
            _ => Err("tiered_moe_forward: input elements must be numeric".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("tiered_moe_forward: second argument must be array".into()),
    };
    let layer = env.tiered_moe_layers.get_mut(&id)
        .ok_or_else(|| format!("No tiered MoE layer with id {}", id))?;
    let output = layer.forward(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_tiered_moe_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tiered_moe_stats expects 1 argument: (id)".into()); }
    let id = value_to_usize(&args[0])?;
    let layer = env.tiered_moe_layers.get(&id)
        .ok_or_else(|| format!("No tiered MoE layer with id {}", id))?;
    let (total, active, hot, warm, cold) = layer.stats();
    Ok(Value::Array(vec![
        Value::Int(total as i128),
        Value::Int(active as i128),
        Value::Int(hot as i128),
        Value::Int(warm as i128),
        Value::Int(cold as i128),
    ]))
}

fn builtin_continuous_learner_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("continuous_learner_new expects 1 argument: (layer_sizes)".into()); }
    let layer_sizes = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(n) => Ok(*n as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("layer_sizes must contain integers".to_string()),
        }).collect::<Result<Vec<usize>, String>>()?,
        _ => return Err("continuous_learner_new expects an array of layer sizes".into()),
    };
    let trainer = continuous_learning::ServingTrainer::new(&layer_sizes, 4096.0);
    let id = env.next_cl_id;
    env.next_cl_id += 1;
    env.continuous_learners.insert(id, trainer);
    Ok(Value::Int(id as i128))
}

fn builtin_continuous_learner_infer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("continuous_learner_infer expects 2 args".into()); }
    let id = value_to_usize(&args[0])?;
    let input: Vec<f64> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f), Value::Int(n) => Ok(*n as f64),
            _ => Err("expected numeric".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("arg must be array".into()),
    };
    let trainer = env.continuous_learners.get_mut(&id).ok_or("no such learner")?;
    let output = trainer.infer(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_continuous_learner_learn(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("continuous_learner_learn expects 4 args".into()); }
    let id = value_to_usize(&args[0])?;
    let input: Vec<f64> = match &args[1] {
        Value::Array(a) => a.iter().map(|v| match v { Value::Float(f)=>Ok(*f), Value::Int(n)=>Ok(*n as f64), _=>Err("num".into()) }).collect::<Result<Vec<f64>,String>>()?,
        _ => return Err("arg must be array".into()),
    };
    let target: Vec<f64> = match &args[2] {
        Value::Array(a) => a.iter().map(|v| match v { Value::Float(f)=>Ok(*f), Value::Int(n)=>Ok(*n as f64), _=>Err("num".into()) }).collect::<Result<Vec<f64>,String>>()?,
        _ => return Err("arg must be array".into()),
    };
    let lr = match &args[3] { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => return Err("lr must be numeric".into()) };
    let trainer = env.continuous_learners.get_mut(&id).ok_or("no such learner")?;
    let (_output, loss) = trainer.infer_and_learn(&input, &target, lr);
    Ok(Value::Float(loss))
}

fn builtin_continuous_learner_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("continuous_learner_stats expects 1 arg".into()); }
    let id = value_to_usize(&args[0])?;
    let trainer = env.continuous_learners.get(&id).ok_or("no such learner")?;
    let (loss_ema, drift, updates, mem) = trainer.get_stats();
    Ok(Value::Array(vec![
        Value::Float(loss_ema),
        Value::Float(drift),
        Value::Float(updates as f64),
        Value::Float(mem),
    ]))
}

#[cfg(test)]
mod tensor_ad_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }

    #[test]
    fn test_tensor_param_creates_tracked() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0, 3.0, 4.0], [2, 2])\nprintln(tensor_data(w))\n}");
        assert_eq!(o, vec!["[1, 2, 3, 4]"]);
    }

    #[test]
    fn test_tensor_input_creates_untracked() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet x = tensor_input([1.0, 2.0], [1, 2])\nprintln(tensor_data(x))\n}");
        assert_eq!(o, vec!["[1, 2]"]);
    }

    #[test]
    fn test_tensor_matmul_shape() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet a = tensor_param([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])\nlet b = tensor_param([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])\nlet c = tensor_matmul(a, b)\nlet d = tensor_data(c)\nprintln(len(d))\n}");
        assert_eq!(o, vec!["4"]);
    }

    #[test]
    fn test_tensor_backward_produces_gradients() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0, 3.0, 4.0], [2, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\nlet g = tensor_grad(w)\nprintln(g)\n}");
        assert_eq!(o, vec!["[1, 1, 1, 1]"]);
    }

    #[test]
    fn test_tensor_sgd_changes_params() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0], [1, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\ntensor_sgd([w], 0.1)\nlet d = tensor_data(w)\nprintln(d)\n}");
        assert_eq!(o, vec!["[0.9, 1.9]"]);
    }

    #[test]
    fn test_tensor_zero_grad_resets() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0], [1, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\ntensor_zero_grad([w])\nlet g = tensor_grad(w)\nprintln(g)\n}");
        assert_eq!(o, vec!["[0, 0]"]);
    }

    #[test]
    fn test_tensor_xor_loss_decreases() {
        let code = "fn main() {\ntensor_tape_new()\nlet w1_data = [0.5, -0.3, 0.8, -0.6, 0.4, 0.7, -0.5, 0.2]\nlet b1_data = [0.0, 0.0, 0.0, 0.0]\nlet w2_data = [0.6, -0.4, 0.3, 0.8, 0.1, -0.2, 0.5, -0.3]\nlet b2_data = [0.0, 0.0]\nlet x_data = [0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]\nlet targets = [0, 1, 1, 0]\nlet initial_loss = 0.0\nlet final_loss = 0.0\nfor epoch in 0..50 {\ntensor_tape_new()\nlet w1 = tensor_param(w1_data, [2, 4])\nlet b1 = tensor_param(b1_data, [1, 4])\nlet w2 = tensor_param(w2_data, [4, 2])\nlet b2 = tensor_param(b2_data, [1, 2])\nlet x = tensor_input(x_data, [4, 2])\nlet h = tensor_matmul(x, w1)\nlet h2 = tensor_add(h, b1)\nlet h3 = tensor_relu(h2)\nlet logits = tensor_matmul(h3, w2)\nlet logits2 = tensor_add(logits, b2)\nlet loss = tensor_cross_entropy(logits2, targets)\nif epoch == 0 {\ninitial_loss = tensor_data(loss)[0]\n}\nif epoch == 49 {\nfinal_loss = tensor_data(loss)[0]\n}\ntensor_backward(loss)\ntensor_sgd([w1, b1, w2, b2], 0.5)\nw1_data = tensor_data(w1)\nb1_data = tensor_data(b1)\nw2_data = tensor_data(w2)\nb2_data = tensor_data(b2)\n}\nif final_loss < initial_loss {\nprintln(\"PASS\")\n} else {\nprintln(\"FAIL\")\n}\n}";
        let o = rv(code);
        assert_eq!(o, vec!["PASS"]);
    }
}

fn builtin_dynamic_model_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("dynamic_model_new expects 1 argument".into()); }
    let sizes = value_to_usize_vec(&args[0])?;
    let model = self_modify::DynamicModel::from_layer_sizes(&sizes);
    let searcher = self_modify::ArchitectureSearcher::new();
    let id = env.next_dm_id;
    env.next_dm_id += 1;
    env.dynamic_models.insert(id, (model, searcher));
    Ok(Value::Int(id as i128))
}

fn builtin_dynamic_model_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("dynamic_model_forward expects 2 arguments".into()); }
    let id = value_to_usize(&args[0])?;
    let input: Vec<f64> = match &args[1] {
        Value::Array(a) => a.iter().map(|x| value_to_f64(x)).collect::<Result<_,_>>()?,
        _ => return Err("expected array".into()),
    };
    let (m, _) = env.dynamic_models.get(&id).ok_or("no such dynamic model")?;
    let out = m.forward(&input);
    Ok(Value::Array(out.into_iter().map(Value::Float).collect()))
}

fn builtin_dynamic_model_add_layer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("expects 3 args".into()); }
    let id = value_to_usize(&args[0])?;
    let pos = value_to_usize(&args[1])?;
    let w = value_to_usize(&args[2])?;
    let (m, _) = env.dynamic_models.get_mut(&id).ok_or("no such dynamic model")?;
    m.add_layer(pos, self_modify::DynamicLayer::Dense {
        weights: vec![vec![0.01; w]; w], biases: vec![0.0; w],
        activation: self_modify::Activation::ReLU,
    });
    Ok(Value::Int(m.total_params() as i128))
}

fn builtin_dynamic_model_remove_layer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("expects 2 args".into()); }
    let id = value_to_usize(&args[0])?;
    let pos = value_to_usize(&args[1])?;
    let (m, _) = env.dynamic_models.get_mut(&id).ok_or("no such dynamic model")?;
    m.remove_layer(pos);
    Ok(Value::Int(m.total_params() as i128))
}

fn builtin_dynamic_model_search_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("expects 3 args".into()); }
    let id = value_to_usize(&args[0])?;
    let tl = value_to_f64(&args[1])?;
    let vl = value_to_f64(&args[2])?;
    let (m, s) = env.dynamic_models.get_mut(&id).ok_or("no such dynamic model")?;
    match s.search_step(m, tl, vl) {
        Some(md) => Ok(Value::String(md.describe())),
        None => Ok(Value::String("no modification".into())),
    }
}

fn builtin_dynamic_model_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("expects 1 arg".into()); }
    let id = value_to_usize(&args[0])?;
    let (m, _) = env.dynamic_models.get(&id).ok_or("no such dynamic model")?;
    Ok(Value::Array(vec![
        Value::Int(m.active_layer_count() as i128),
        Value::Int(m.total_params() as i128),
        Value::Int(m.architecture_hash() as i128),
    ]))
}

fn builtin_hetero_layer_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("hetero_layer_new expects 2 arguments: (n_experts_per_type, width)".into());
    }
    let n_per_type = value_to_usize(&args[0])?;
    let width = value_to_usize(&args[1])?;
    let layer = heterogeneous::create_hetero_layer(n_per_type, width);
    let id = env.next_hetero_id;
    env.next_hetero_id += 1;
    env.hetero_layers.insert(id, layer);
    Ok(Value::Int(id as i128))
}

fn builtin_hetero_layer_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("hetero_layer_forward expects 2 arguments: (id, input)".into());
    }
    let id = value_to_usize(&args[0])?;
    let input = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(n) => Ok(*n as f64),
            _ => Err("hetero_layer_forward: input elements must be numeric".to_string()),
        }).collect::<Result<Vec<f64>, String>>()?,
        _ => return Err("hetero_layer_forward: second argument must be array".into()),
    };
    let layer = env.hetero_layers.get_mut(&id)
        .ok_or_else(|| format!("No heterogeneous layer with id {}", id))?;
    let output = layer.forward(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_hetero_layer_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("hetero_layer_stats expects 1 argument: (id)".into());
    }
    let id = value_to_usize(&args[0])?;
    let layer = env.hetero_layers.get(&id)
        .ok_or_else(|| format!("No heterogeneous layer with id {}", id))?;
    let stats = layer.stats();
    // Return [total_experts, total_params, n_types]
    Ok(Value::Array(vec![
        Value::Int(stats.total_experts as i128),
        Value::Int(stats.total_params as i128),
        Value::Int(stats.type_counts.len() as i128),
    ]))
}
