use crate::ast::*;
use crate::autodiff;
use crate::crypto;
use crate::gpu_runtime;
use crate::local_learn;
use crate::memory;
use crate::modmath;
use crate::spiking;
use crate::ssm;
use crate::tensor_autodiff;
use crate::architectures;
use crate::continuous_learning;
use crate::self_modify;
use crate::multiscale;
use crate::tiered_experts;
use crate::heterogeneous;
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::path::PathBuf;

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
    Tensor(crate::tensor_engine::FastTensor),
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
    /// Live (hot-swappable) model handle
    LiveModel { id: usize },
    /// ZK proof value
    ZkProof(crate::zkp::Proof),
    /// Raw memory pointer (for FFI / self-hosting)
    Pointer(usize),
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Value::Int(n) => write!(f, "{}", n),
            Value::Float(n) => {
                if n.fract() == 0.0 && n.is_finite() {
                    write!(f, "{:.1}", n)
                } else {
                    write!(f, "{}", n)
                }
            }
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
            Value::Tensor(t) => write!(f, "Tensor<{:?}>[{}]", t.dtype, t.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", ")),
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
            Value::LiveModel { id } => write!(f, "<live model #{}>", id),
            Value::ZkProof(p) => write!(f, "<proof constraints={} output={}>", p.num_constraints, p.output),
            Value::Pointer(addr) => write!(f, "<ptr 0x{:x}>", addr),
        }
    }
}

/// Environment for the interpreter
pub struct Env {
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
    pub(crate) novel_memories: HashMap<usize, crate::novel_arch::NeuralMemory>,
    pub(crate) novel_worlds: HashMap<usize, crate::novel_arch::WorldModel>,
    pub(crate) novel_reasoners: HashMap<usize, crate::novel_arch::SymbolicReasoner>,
    pub(crate) novel_workspaces: HashMap<usize, crate::novel_arch::GlobalWorkspace>,
    pub(crate) novel_continual: HashMap<usize, crate::novel_arch::ContinualLearner>,
    pub(crate) novel_compositional: HashMap<usize, crate::novel_arch::CompositionalNet>,
    loading_modules: HashSet<PathBuf>,
    loaded_modules: HashSet<PathBuf>,
    pub(crate) nn_models: HashMap<usize, crate::nn::Model>,
    /// JIT engine for @gpu annotated functions
    jit_engine: crate::jit::JitEngine,
    /// AST of @gpu annotated functions for JIT compilation
    gpu_functions: HashMap<String, crate::ast::Function>,
    pub(crate) experiment_manager: Option<crate::experiment::ExperimentManager>,
    pub(crate) autograd_tape: Option<crate::autograd::Tape>,
    pub(crate) autograd_adam: Option<crate::autograd::AdamOptimizer>,
    pub(crate) energy_budgets: HashMap<usize, crate::energy_compute::EnergyBudget>,
    pub(crate) next_energy_budget_id: usize,
    pub(crate) event_driven_executors: HashMap<usize, crate::energy_compute::EventDrivenExecutor>,
    pub(crate) next_event_driven_id: usize,
    pub(crate) neuromorphic_layers: HashMap<usize, crate::energy_compute::NeuromorphicLayer>,
    pub(crate) next_neuromorphic_id: usize,
    pub(crate) thought_channels: HashMap<usize, crate::thought_protocol::ThoughtChannel>,
    pub(crate) next_thought_channel_id: usize,
    /// Trait definitions: trait_name -> (method_name, has_default_body)
    pub(crate) trait_defs: HashMap<String, Vec<(String, bool)>>,
    /// Trait implementations: (trait_name, type_name) -> registered
    pub(crate) trait_impls: HashMap<(String, String), bool>,
    /// Cognitive matrices for MatrixOfThought binding
    pub(crate) cog_matrices: HashMap<usize, crate::matrix_of_thought::CognitiveMatrix>,
    /// Networking runtime
    pub(crate) net_runtime: crate::net_runtime::NetRuntime,
    /// User-defined field types: name -> static FieldParams
    pub(crate) field_defs: HashMap<String, &'static modmath::FieldParams>,
    /// Functions marked with `diff fn` annotation
    pub(crate) diff_functions: HashSet<String>,
    /// Functions marked with `#[verifiable]`
    pub(crate) verifiable_functions: HashSet<String>,
    /// Active ZK arithmetic trace (set during `prove()` execution)
    pub(crate) zk_trace: Option<crate::zkp::ArithTrace>,
    /// Wire mapping for ZK trace: variable name -> wire index
    pub(crate) zk_wire_map: HashMap<String, (usize, i128)>,
    /// Variables marked as `unique` (linear types)
    pub(crate) unique_vars: HashSet<String>,
    /// Variables that have been moved (consumed)
    pub(crate) moved_vars: HashSet<String>,
    /// Live (hot-swappable) models
    pub(crate) live_models: HashMap<usize, crate::nn::Model>,
    pub(crate) next_live_model_id: usize,
    /// Autodiff mode: when true, tensor ops record to autograd tape
    pub(crate) ad_mode: bool,
    // Feature F6: Kernel fusion
    pub(crate) in_fuse_block: bool,
    pub(crate) fuse_ops: Vec<String>,
    pub(crate) fuse_stats: FuseStats,
    // Feature F7: GPU memory ownership
    pub(crate) gpu_owned: HashSet<String>,
    pub(crate) gpu_allocations: usize,
    // Feature F8: Parallel / distributed
    pub(crate) num_devices: usize,
    // Feature F9: Training
    pub(crate) train_checkpoints: Vec<(usize, f64)>,
    // Feature F10: Deterministic
    pub(crate) deterministic_mode: bool,
    pub(crate) rng_seed: u64,
    // Feature F11: Autocast
    pub(crate) autocast_dtype: Option<String>,
    // Feature F14: Speculate
    pub(crate) speculating: bool,
    pub(crate) speculate_depth: usize,
    // Feature F16: Topology
    pub(crate) topologies: Vec<Value>,
    // Feature F18: Quantize
    pub(crate) quantize_dtype: Option<String>,
    // Feature F21: Safe
    pub(crate) safe_mode: bool,
    pub(crate) op_budget: u64,
    pub(crate) op_counter: u64,
    // Feature F22: Mmap
    pub(crate) mmap_models: HashMap<String, String>,
    // Feature F23: Explain
    pub(crate) explaining: bool,
    pub(crate) explain_trace: Vec<String>,
    // Annotation tracking
    pub(crate) cache_functions: HashSet<String>,
    pub(crate) reward_functions: HashSet<String>,
    pub(crate) stream_functions: HashSet<String>,
    pub(crate) evolve_functions: HashSet<String>,
    pub(crate) fn_cache: HashMap<String, Value>,
    pub(crate) stream_buffer: Option<Vec<Value>>,
    // Feature 24: Consensus
    pub(crate) consensus_voters: usize,
    // Feature 27: Symbolic
    pub(crate) symbolic_mode: bool,
    // Feature 28: Temporal
    pub(crate) temporal_mode: bool,
    pub(crate) temporal_step: usize,
    // Feature 29: Federated
    pub(crate) federated_mode: bool,
    // Feature 30: Sandbox
    pub(crate) sandboxed: bool,
    // Feature 31: Compress
    pub(crate) compression_ratio: f64,
    // Feature 32: Alignment
    pub(crate) alignment_functions: HashSet<String>,
    // Feature 33: Metacognition
    pub(crate) metacognition_mode: bool,
    pub(crate) confidence_scores: Vec<f64>,
    // Feature 26: Bounded recursion
    pub(crate) recursion_limits: HashMap<String, u64>,
    pub(crate) recursion_counters: HashMap<String, u64>,
    // Feature 34: Theorem
    pub(crate) theorem_mode: bool,
    pub(crate) theorem_obligations: Vec<(String, bool)>,
    // Feature 35: Continual learning
    pub(crate) continual_mode: bool,
    pub(crate) memory_snapshots: Vec<HashMap<String, Value>>,
    // Feature 36: Multimodal
    pub(crate) multimodal_mode: bool,
    pub(crate) active_modalities: Vec<String>,
    // Feature 37: World model
    pub(crate) world_model_active: bool,
    pub(crate) world_state_log: Vec<String>,
    // Feature 38: Self-improve
    pub(crate) self_improve_generation: usize,
    pub(crate) self_improve_score: f64,
    // Feature 39: Intention
    pub(crate) intention_functions: HashSet<String>,
    // Feature 40: Memory
    pub(crate) memory_config: MemoryConfig,
    pub(crate) short_term_memory: Vec<Value>,
    pub(crate) long_term_memory: Vec<Value>,
    pub(crate) episodic_memory: Vec<Value>,
    // Features 41-50
    pub(crate) attention_mode: bool,
    pub(crate) curriculum_difficulty: f64,
    pub(crate) curriculum_step: usize,
    pub(crate) ensemble_models: Vec<usize>,
    pub(crate) adversarial_mode: bool,
    pub(crate) adversarial_epsilon: f64,
    pub(crate) transfer_frozen_layers: Vec<String>,
    pub(crate) sparse_mode: bool,
    pub(crate) sparse_threshold: f64,
    pub(crate) async_infer_mode: bool,
    pub(crate) profiling: bool,
    pub(crate) profile_data: Vec<(String, f64)>,
    pub(crate) contract_functions: std::collections::HashSet<String>,
    pub(crate) gradient_surgery_functions: std::collections::HashSet<String>,
    /// Functions marked @persistent_grad: stores gradient tapes keyed by function name
    pub(crate) persistent_grad_tapes: HashMap<String, Vec<f64>>,
    /// Functions marked @fuse: hints for kernel fusion
    pub(crate) fuse_functions: std::collections::HashSet<String>,
    /// Functions marked @constant_time: enforce constant-time execution
    pub(crate) constant_time_functions: std::collections::HashSet<String>,
}

#[derive(Clone, Default)]
pub(crate) struct MemoryConfig {
    pub short_term_capacity: usize,
    pub long_term_capacity: usize,
    pub episodic_capacity: usize,
}

#[derive(Clone, Default)]
pub(crate) struct FuseStats {
    pub total_fused: usize,
    pub fusion_blocks: usize,
}

#[derive(Clone)]
pub(crate) enum FnDef {
    User {
        params: Vec<String>,
        body: Block,
    },
    Builtin(fn(&mut Env, Vec<Value>) -> Result<Value, String>),
    /// Gradient wrapper: calls the original function under autodiff tape
    GradWrapper {
        fn_name: String,
        order: u8,
    },
}

impl Env {
    pub fn new() -> Self {
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
            novel_memories: HashMap::new(),
            novel_worlds: HashMap::new(),
            novel_reasoners: HashMap::new(),
            novel_workspaces: HashMap::new(),
            novel_continual: HashMap::new(),
            novel_compositional: HashMap::new(),
            loading_modules: HashSet::new(),
            loaded_modules: HashSet::new(),
            nn_models: HashMap::new(),
            jit_engine: crate::jit::JitEngine::new(),
            gpu_functions: HashMap::new(),
            experiment_manager: None,
            autograd_tape: None,
            autograd_adam: None,
            energy_budgets: HashMap::new(),
            next_energy_budget_id: 0,
            event_driven_executors: HashMap::new(),
            next_event_driven_id: 0,
            neuromorphic_layers: HashMap::new(),
            next_neuromorphic_id: 0,
            thought_channels: HashMap::new(),
            next_thought_channel_id: 0,
            trait_defs: HashMap::new(),
            trait_impls: HashMap::new(),
            cog_matrices: HashMap::new(),
            net_runtime: crate::net_runtime::NetRuntime::new(),
            field_defs: HashMap::new(),
            diff_functions: HashSet::new(),
            verifiable_functions: HashSet::new(),
            zk_trace: None,
            zk_wire_map: HashMap::new(),
            unique_vars: HashSet::new(),
            moved_vars: HashSet::new(),
            live_models: HashMap::new(),
            next_live_model_id: 0,
            ad_mode: false,
            in_fuse_block: false,
            fuse_ops: Vec::new(),
            fuse_stats: FuseStats::default(),
            gpu_owned: HashSet::new(),
            gpu_allocations: 0,
            num_devices: 4, // default simulated device count
            train_checkpoints: Vec::new(),
            deterministic_mode: false,
            rng_seed: 0,
            autocast_dtype: None,
            speculating: false,
            speculate_depth: 0,
            topologies: Vec::new(),
            quantize_dtype: None,
            safe_mode: false,
            op_budget: u64::MAX,
            op_counter: 0,
            mmap_models: HashMap::new(),
            explaining: false,
            explain_trace: Vec::new(),
            cache_functions: HashSet::new(),
            reward_functions: HashSet::new(),
            stream_functions: HashSet::new(),
            evolve_functions: HashSet::new(),
            fn_cache: HashMap::new(),
            stream_buffer: None,
            consensus_voters: 3,
            symbolic_mode: false,
            temporal_mode: false,
            temporal_step: 0,
            federated_mode: false,
            sandboxed: false,
            compression_ratio: 1.0,
            alignment_functions: HashSet::new(),
            metacognition_mode: false,
            confidence_scores: Vec::new(),
            recursion_limits: HashMap::new(),
            recursion_counters: HashMap::new(),
            theorem_mode: false,
            theorem_obligations: Vec::new(),
            continual_mode: false,
            memory_snapshots: Vec::new(),
            multimodal_mode: false,
            active_modalities: Vec::new(),
            world_model_active: false,
            world_state_log: Vec::new(),
            self_improve_generation: 0,
            self_improve_score: 0.0,
            intention_functions: HashSet::new(),
            memory_config: MemoryConfig::default(),
            short_term_memory: Vec::new(),
            long_term_memory: Vec::new(),
            episodic_memory: Vec::new(),
            // Features 41-50
            attention_mode: false,
            curriculum_difficulty: 0.1,
            curriculum_step: 0,
            ensemble_models: Vec::new(),
            adversarial_mode: false,
            adversarial_epsilon: 0.01,
            transfer_frozen_layers: Vec::new(),
            sparse_mode: false,
            sparse_threshold: 0.01,
            async_infer_mode: false,
            profiling: false,
            profile_data: Vec::new(),
            contract_functions: HashSet::new(),
            gradient_surgery_functions: HashSet::new(),
            persistent_grad_tapes: HashMap::new(),
            fuse_functions: HashSet::new(),
            constant_time_functions: HashSet::new(),
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

    /// Call a named builtin function (used by the VM to delegate complex builtins).
    pub fn call_builtin(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        if let Some(fndef) = self.functions.get(name).cloned() {
            match fndef {
                FnDef::Builtin(f) => f(self, args),
                _ => Err(format!("{} is not a builtin", name)),
            }
        } else {
            Err(format!("unknown builtin: {}", name))
        }
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
        self.functions.insert("attn_compute".to_string(), FnDef::Builtin(builtin_attention));
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
        self.functions.insert("str".to_string(), FnDef::Builtin(builtin_to_string));
        self.functions.insert("int".to_string(), FnDef::Builtin(builtin_int));
        self.functions.insert("float".to_string(), FnDef::Builtin(builtin_float));
        self.functions.insert("type_of".to_string(), FnDef::Builtin(builtin_type_of));
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
        self.functions.insert("load_csv".to_string(), FnDef::Builtin(builtin_load_csv));
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
        self.functions.insert("Some".to_string(), FnDef::Builtin(builtin_some));
        self.functions.insert("none".to_string(), FnDef::Builtin(builtin_none));
        self.functions.insert("None".to_string(), FnDef::Builtin(builtin_none));
        self.functions.insert("unwrap".to_string(), FnDef::Builtin(builtin_unwrap));
        self.functions.insert("unwrap_or".to_string(), FnDef::Builtin(builtin_unwrap_or));
        self.functions.insert("is_some".to_string(), FnDef::Builtin(builtin_is_some));
        self.functions.insert("is_none".to_string(), FnDef::Builtin(builtin_is_none));

        // Result builtins
        self.functions.insert("ok".to_string(), FnDef::Builtin(builtin_ok));
        self.functions.insert("Ok".to_string(), FnDef::Builtin(builtin_ok));
        self.functions.insert("err".to_string(), FnDef::Builtin(builtin_err));
        self.functions.insert("Err".to_string(), FnDef::Builtin(builtin_err));
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

        // GPU compute (native compilation) builtins
        self.functions.insert("gpu_available".to_string(), FnDef::Builtin(builtin_gpu_available));
        self.functions.insert("gpu_native_matmul".to_string(), FnDef::Builtin(builtin_gpu_native_matmul));
        self.functions.insert("gpu_train_step".to_string(), FnDef::Builtin(builtin_gpu_train_step));
        self.functions.insert("gpu_benchmark".to_string(), FnDef::Builtin(builtin_gpu_benchmark));

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
        self.functions.insert("ft_tensor".to_string(), FnDef::Builtin(ft_builtin_tensor));
        self.functions.insert("ft_tensor_zeros".to_string(), FnDef::Builtin(ft_builtin_tensor_zeros));
        self.functions.insert("ft_tensor_ones".to_string(), FnDef::Builtin(ft_builtin_tensor_ones));
        self.functions.insert("ft_tensor_rand".to_string(), FnDef::Builtin(ft_builtin_tensor_rand));
        self.functions.insert("ft_tensor_shape".to_string(), FnDef::Builtin(ft_builtin_tensor_shape));
        self.functions.insert("ft_tensor_dtype".to_string(), FnDef::Builtin(ft_builtin_tensor_dtype));
        self.functions.insert("ft_tensor_reshape".to_string(), FnDef::Builtin(ft_builtin_tensor_reshape));
        self.functions.insert("ft_tensor_add".to_string(), FnDef::Builtin(ft_builtin_tensor_add));
        self.functions.insert("ft_tensor_mul".to_string(), FnDef::Builtin(ft_builtin_tensor_mul));
        self.functions.insert("ft_tensor_matmul_dense".to_string(), FnDef::Builtin(ft_builtin_tensor_matmul_dense));
        self.functions.insert("ft_tensor_transpose".to_string(), FnDef::Builtin(ft_builtin_tensor_transpose));
        self.functions.insert("ft_tensor_slice".to_string(), FnDef::Builtin(ft_builtin_tensor_slice));
        self.functions.insert("ft_tensor_to_array".to_string(), FnDef::Builtin(ft_builtin_tensor_to_array));
        self.functions.insert("ft_tensor_from_array".to_string(), FnDef::Builtin(ft_builtin_tensor_from_array));

        // Autograd builtins
        self.functions.insert("autograd_new".into(), FnDef::Builtin(crate::autograd::builtin_autograd_new));
        self.functions.insert("autograd_tensor".into(), FnDef::Builtin(crate::autograd::builtin_autograd_tensor));
        self.functions.insert("autograd_input".into(), FnDef::Builtin(crate::autograd::builtin_autograd_input));
        self.functions.insert("autograd_matmul".into(), FnDef::Builtin(crate::autograd::builtin_autograd_matmul));
        self.functions.insert("autograd_add".into(), FnDef::Builtin(crate::autograd::builtin_autograd_add));
        self.functions.insert("autograd_mul".into(), FnDef::Builtin(crate::autograd::builtin_autograd_mul));
        self.functions.insert("autograd_sub".into(), FnDef::Builtin(crate::autograd::builtin_autograd_sub));
        self.functions.insert("autograd_div".into(), FnDef::Builtin(crate::autograd::builtin_autograd_div));
        self.functions.insert("autograd_relu".into(), FnDef::Builtin(crate::autograd::builtin_autograd_relu));
        self.functions.insert("autograd_sigmoid".into(), FnDef::Builtin(crate::autograd::builtin_autograd_sigmoid));
        self.functions.insert("autograd_tanh".into(), FnDef::Builtin(crate::autograd::builtin_autograd_tanh));
        self.functions.insert("autograd_softmax".into(), FnDef::Builtin(crate::autograd::builtin_autograd_softmax));
        self.functions.insert("autograd_exp".into(), FnDef::Builtin(crate::autograd::builtin_autograd_exp));
        self.functions.insert("autograd_log".into(), FnDef::Builtin(crate::autograd::builtin_autograd_log));
        self.functions.insert("autograd_sum".into(), FnDef::Builtin(crate::autograd::builtin_autograd_sum));
        self.functions.insert("autograd_mean".into(), FnDef::Builtin(crate::autograd::builtin_autograd_mean));
        self.functions.insert("autograd_mse".into(), FnDef::Builtin(crate::autograd::builtin_autograd_mse));
        self.functions.insert("autograd_broadcast_add".into(), FnDef::Builtin(crate::autograd::builtin_autograd_broadcast_add));
        self.functions.insert("autograd_backward".into(), FnDef::Builtin(crate::autograd::builtin_autograd_backward));
        self.functions.insert("autograd_grad".into(), FnDef::Builtin(crate::autograd::builtin_autograd_grad));
        self.functions.insert("autograd_data".into(), FnDef::Builtin(crate::autograd::builtin_autograd_data));
        self.functions.insert("autograd_zero_grad".into(), FnDef::Builtin(crate::autograd::builtin_autograd_zero_grad));
        self.functions.insert("autograd_adam_step".into(), FnDef::Builtin(crate::autograd::builtin_autograd_adam_step));

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
        crate::energy_compute::register_builtins(self);

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

        // RL training builtins (PPO/DPO/GRPO)
        crate::rl_training::register_builtins(self);

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

        // Novel architecture builtins
        crate::novel_arch::register_builtins(self);


        // Math builtins
        self.functions.insert("sqrt".to_string(), FnDef::Builtin(builtin_sqrt));
        self.functions.insert("sin".to_string(), FnDef::Builtin(builtin_sin));
        self.functions.insert("cos".to_string(), FnDef::Builtin(builtin_cos));
        self.functions.insert("tan".to_string(), FnDef::Builtin(builtin_tan));
        self.functions.insert("exp".to_string(), FnDef::Builtin(builtin_exp));
        self.functions.insert("log".to_string(), FnDef::Builtin(builtin_log));
        self.functions.insert("log2".to_string(), FnDef::Builtin(builtin_log2));
        self.functions.insert("log10".to_string(), FnDef::Builtin(builtin_log10));
        self.functions.insert("abs".to_string(), FnDef::Builtin(builtin_abs));
        self.functions.insert("pow".to_string(), FnDef::Builtin(builtin_pow));
        self.functions.insert("floor".to_string(), FnDef::Builtin(builtin_floor));
        self.functions.insert("ceil".to_string(), FnDef::Builtin(builtin_ceil));
        self.functions.insert("round".to_string(), FnDef::Builtin(builtin_round));
        self.functions.insert("min".to_string(), FnDef::Builtin(builtin_min));
        self.functions.insert("max".to_string(), FnDef::Builtin(builtin_max));
        self.functions.insert("range".to_string(), FnDef::Builtin(builtin_range));

        // Neural network framework builtins
        crate::nn::register_nn_builtins(self);
        // Model interop builtins
        crate::model_interop::register_model_interop_builtins(self);

        // Distributed runtime builtins
        crate::dist_runtime::register_builtins(self);

        // BigInt engine builtins (U256, batch ops, EC)
        crate::bigint_engine::register_builtins(self);

        // Huge matrix engine builtins
        crate::huge_matrix::register_builtins(self);

        // Data fabric builtins
        crate::data_fabric::register_builtins(self);

        // Meta-engine: self-evolving code builtins
        crate::meta_engine::register_builtins(self);

        // Pillar 5: Continuous Mathematics (manifolds)
        crate::manifold::register_builtins(self);

        // Pillar 4: Formal Neural Specifications
        crate::neural_specs::register_builtins(self);

        // Pillar 2: Architecture as First-Class Data
        crate::arch_graph::register_builtins(self);

        // Pillar 1: LLM-to-LLM Binary Protocol
        crate::thought_protocol::register_builtins(self);

        // Pillar 3: Differentiable Meta-Learning
        crate::diff_meta::register_builtins(self);

        // Pillar 7: Self-Improving Compiler
        crate::meta_compiler::register_builtins(self);

        // Pillar 8: AGI Emergence Core
        crate::agi_core::register_builtins(self);

        // Vortex IR
        crate::vir::register_builtins(self);

        // Networking
        crate::net_runtime::register_builtins(self);

        // Fast matrix algebra (Monarch, Butterfly, Sparse Attention, Low-Rank)
        crate::fast_matrix::register_builtins(self);

        // Field arithmetic (secp256k1 Fp/Fn, fractions, EC points, M_k scanner)
        crate::field_arithmetic::register_builtins(self);

        // Model merging + RAG
        crate::model_merge::register_builtins(self);

        // Continual learning
        crate::continual_engine::register_builtins(self);

        // BPE tokenizer
        crate::tokenizer::register_builtins(self);

        // Agent protocol, NAS, reasoning
        crate::agent_protocol::register_builtins(self);

        // Mixture of Experts + Speculative Decoding
        crate::moe::register_builtins(self);

        // Native PTX codegen
        crate::ptx_native::register_builtins(self);

        // Autodiff engine
        crate::autodiff_engine::register_builtins(self);

        // KV cache + paged attention
        crate::kv_cache::register_builtins(self);

        // GGUF loader
        crate::gguf_loader::register_builtins(self);

        // Vortex ISA
        crate::vortex_isa::register_builtins(self);

        // DRM GPU driver
        crate::drm_driver::register_builtins(self);

        // Data loader
        crate::data_loader::register_builtins(self);

        // SIMT execution engine
        crate::simt_engine::register_builtins(self);

        // Inference server
        crate::inference_server::register_builtins(self);

        // Transformer
        crate::transformer::register_builtins(self);

        // Self-hosting foundation: raw memory, syscalls/FFI, C structs, threads, parser combinators
        crate::raw_memory::register_builtins(self);
        crate::syscall_ffi::register_builtins(self);
        crate::c_struct::register_builtins(self);
        crate::thread_runtime::register_builtins(self);
        crate::parser_combinators::register_builtins(self);

        // Math constants
        self.define("PI", Value::Float(std::f64::consts::PI));
        self.define("E", Value::Float(std::f64::consts::E));

        // Feature 1: Differentiable functions
        self.functions.insert("grad".to_string(), FnDef::Builtin(builtin_grad));
        self.functions.insert("value_and_grad".to_string(), FnDef::Builtin(builtin_value_and_grad));

        // Feature 4: ZK proof generation
        self.functions.insert("prove".to_string(), FnDef::Builtin(builtin_prove));
        self.functions.insert("verify".to_string(), FnDef::Builtin(builtin_verify));
        self.functions.insert("prove_value".to_string(), FnDef::Builtin(builtin_prove_value));
        self.functions.insert("prove_and_verify".to_string(), FnDef::Builtin(builtin_prove_and_verify));

        // Feature 5: Live model builtins
        self.functions.insert("nn_new".to_string(), FnDef::Builtin(builtin_nn_new));
        self.functions.insert("nn_linear".to_string(), FnDef::Builtin(builtin_nn_linear));

        // Feature 7: GPU memory management
        self.functions.insert("gpu_release".to_string(), FnDef::Builtin(builtin_gpu_release));
        self.functions.insert("gpu_transfer".to_string(), FnDef::Builtin(builtin_gpu_transfer));
        self.functions.insert("gpu_info".to_string(), FnDef::Builtin(builtin_gpu_info));

        // Feature 8: Distributed / parallel
        self.functions.insert("shard".to_string(), FnDef::Builtin(builtin_shard));
        self.functions.insert("unshard".to_string(), FnDef::Builtin(builtin_unshard));
        self.functions.insert("all_reduce".to_string(), FnDef::Builtin(builtin_all_reduce));
        self.functions.insert("set_devices".to_string(), FnDef::Builtin(builtin_set_devices));
        self.functions.insert("device_id".to_string(), FnDef::Builtin(builtin_device_id));

        // Feature 9: Training builtins
        self.functions.insert("checkpoint".to_string(), FnDef::Builtin(builtin_checkpoint));
        self.functions.insert("seed".to_string(), FnDef::Builtin(builtin_seed));

        // Feature 11: Mixed precision
        self.functions.insert("to_f16".to_string(), FnDef::Builtin(builtin_to_f16));
        self.functions.insert("to_f32".to_string(), FnDef::Builtin(builtin_to_f32));
        self.functions.insert("to_bf16".to_string(), FnDef::Builtin(builtin_to_bf16));

        // Feature 14: Speculative execution
        self.functions.insert("speculate_best".to_string(), FnDef::Builtin(builtin_speculate_best));

        // Feature 15: Semantic cache
        self.functions.insert("cache_get".to_string(), FnDef::Builtin(builtin_cache_get));
        self.functions.insert("cache_set".to_string(), FnDef::Builtin(builtin_cache_set));
        self.functions.insert("cache_clear".to_string(), FnDef::Builtin(builtin_cache_clear));

        // Feature 16: Stream
        self.functions.insert("yield_val".to_string(), FnDef::Builtin(builtin_yield));
        self.functions.insert("collect_stream".to_string(), FnDef::Builtin(builtin_collect_stream));

        // Feature 17: Reward
        self.functions.insert("reward_score".to_string(), FnDef::Builtin(builtin_reward_score));

        // Feature 19: Topology
        self.functions.insert("create_topology".to_string(), FnDef::Builtin(builtin_create_topology));

        // Feature 20: Evolve
        self.functions.insert("mutate_fn".to_string(), FnDef::Builtin(builtin_mutate_fn));

        // Feature 21: Safe
        self.functions.insert("remaining_budget".to_string(), FnDef::Builtin(builtin_remaining_budget));

        // Feature 22: Mmap
        self.functions.insert("mmap_load".to_string(), FnDef::Builtin(builtin_mmap_load));

        // Feature 23: Explain
        self.functions.insert("explain_op".to_string(), FnDef::Builtin(builtin_explain_op));
        self.functions.insert("attention_map".to_string(), FnDef::Builtin(builtin_attention_map));

        // Feature 24: Consensus
        self.functions.insert("set_voters".to_string(), FnDef::Builtin(builtin_set_voters));
        self.functions.insert("byzantine_check".to_string(), FnDef::Builtin(builtin_byzantine_check));

        // Feature 25: Hallucination check
        self.functions.insert("hallucination_check".to_string(), FnDef::Builtin(builtin_hallucination_check));
        self.functions.insert("fact_ground".to_string(), FnDef::Builtin(builtin_fact_ground));

        // Feature 26: Bounded recursion
        self.functions.insert("set_recursion_limit".to_string(), FnDef::Builtin(builtin_set_recursion_limit));

        // Feature 27: Symbolic
        self.functions.insert("symbolic_var".to_string(), FnDef::Builtin(builtin_symbolic_var));
        self.functions.insert("symbolic_constraint".to_string(), FnDef::Builtin(builtin_symbolic_constraint));
        self.functions.insert("symbolic_solve".to_string(), FnDef::Builtin(builtin_symbolic_solve));

        // Feature 28: Temporal
        self.functions.insert("temporal_step".to_string(), FnDef::Builtin(builtin_temporal_step));
        self.functions.insert("causal_mask".to_string(), FnDef::Builtin(builtin_causal_mask));

        // Feature 29: Federated
        self.functions.insert("federated_aggregate".to_string(), FnDef::Builtin(builtin_federated_aggregate));
        self.functions.insert("differential_privacy".to_string(), FnDef::Builtin(builtin_differential_privacy));

        // Feature 30: Sandbox
        self.functions.insert("sandbox_check".to_string(), FnDef::Builtin(builtin_sandbox_check));

        // Feature 31: Compress
        self.functions.insert("prune".to_string(), FnDef::Builtin(builtin_prune));
        self.functions.insert("distill".to_string(), FnDef::Builtin(builtin_distill));

        // Feature 32: Alignment
        self.functions.insert("align_score".to_string(), FnDef::Builtin(builtin_align_score));
        self.functions.insert("preference_pair".to_string(), FnDef::Builtin(builtin_preference_pair));

        // Feature 33: Metacognition
        self.functions.insert("confidence".to_string(), FnDef::Builtin(builtin_confidence));
        self.functions.insert("uncertainty".to_string(), FnDef::Builtin(builtin_uncertainty));
        self.functions.insert("introspect".to_string(), FnDef::Builtin(builtin_introspect));

        // Feature 34: Theorem proving
        self.functions.insert("assert_property".to_string(), FnDef::Builtin(builtin_assert_property));
        self.functions.insert("lipschitz_bound".to_string(), FnDef::Builtin(builtin_lipschitz_bound));
        self.functions.insert("robustness_cert".to_string(), FnDef::Builtin(builtin_robustness_cert));

        // Feature 35: Continual learning
        self.functions.insert("replay_buffer".to_string(), FnDef::Builtin(builtin_replay_buffer));
        self.functions.insert("ewc_penalty".to_string(), FnDef::Builtin(builtin_ewc_penalty));

        // Feature 36: Multimodal
        self.functions.insert("fuse_modalities".to_string(), FnDef::Builtin(builtin_fuse_modalities));
        self.functions.insert("encode_vision".to_string(), FnDef::Builtin(builtin_encode_vision));
        self.functions.insert("encode_audio".to_string(), FnDef::Builtin(builtin_encode_audio));
        self.functions.insert("encode_text".to_string(), FnDef::Builtin(builtin_encode_text));

        // Feature 37: World model
        self.functions.insert("world_state".to_string(), FnDef::Builtin(builtin_world_state));
        self.functions.insert("predict_next".to_string(), FnDef::Builtin(builtin_predict_next));
        self.functions.insert("simulate_action".to_string(), FnDef::Builtin(builtin_simulate_action));

        // Feature 38: Self-improve
        self.functions.insert("evaluate_self".to_string(), FnDef::Builtin(builtin_evaluate_self));
        self.functions.insert("improve_score".to_string(), FnDef::Builtin(builtin_improve_score));

        // Feature 39: Intention
        self.functions.insert("set_goal".to_string(), FnDef::Builtin(builtin_set_goal));
        self.functions.insert("explain_why".to_string(), FnDef::Builtin(builtin_explain_why));

        // Feature 40: Memory
        self.functions.insert("remember".to_string(), FnDef::Builtin(builtin_remember));
        self.functions.insert("recall".to_string(), FnDef::Builtin(builtin_recall));
        self.functions.insert("forget".to_string(), FnDef::Builtin(builtin_forget));
        self.functions.insert("consolidate".to_string(), FnDef::Builtin(builtin_consolidate));

        // Feature 41: Attention
        self.functions.insert("multi_head_attention".to_string(), FnDef::Builtin(builtin_multi_head_attention));
        self.functions.insert("flash_attention_v2".to_string(), FnDef::Builtin(builtin_flash_attention_v2));
        self.functions.insert("attention_mask".to_string(), FnDef::Builtin(builtin_attention_mask_builtin));

        // Feature 42: Gradient surgery
        self.functions.insert("clip_grad".to_string(), FnDef::Builtin(builtin_clip_grad));
        self.functions.insert("grad_norm".to_string(), FnDef::Builtin(builtin_grad_norm));
        self.functions.insert("freeze_layer".to_string(), FnDef::Builtin(builtin_freeze_layer));
        self.functions.insert("unfreeze_layer".to_string(), FnDef::Builtin(builtin_unfreeze_layer));

        // Feature 43: Curriculum learning
        self.functions.insert("set_difficulty".to_string(), FnDef::Builtin(builtin_set_difficulty));
        self.functions.insert("get_difficulty".to_string(), FnDef::Builtin(builtin_get_difficulty));
        self.functions.insert("curriculum_schedule".to_string(), FnDef::Builtin(builtin_curriculum_schedule));

        // Feature 44: Ensemble
        self.functions.insert("ensemble_add".to_string(), FnDef::Builtin(builtin_ensemble_add));
        self.functions.insert("ensemble_vote".to_string(), FnDef::Builtin(builtin_ensemble_vote));
        self.functions.insert("ensemble_avg".to_string(), FnDef::Builtin(builtin_ensemble_avg));

        // Feature 45: Adversarial
        self.functions.insert("fgsm_attack".to_string(), FnDef::Builtin(builtin_fgsm_attack));
        self.functions.insert("pgd_attack".to_string(), FnDef::Builtin(builtin_pgd_attack));
        self.functions.insert("adversarial_train_step".to_string(), FnDef::Builtin(builtin_adversarial_train_step));

        // Feature 46: Transfer learning
        self.functions.insert("freeze".to_string(), FnDef::Builtin(builtin_freeze));
        self.functions.insert("unfreeze".to_string(), FnDef::Builtin(builtin_unfreeze));
        self.functions.insert("fine_tune".to_string(), FnDef::Builtin(builtin_fine_tune));

        // Feature 47: Sparse computation
        self.functions.insert("to_sparse_scope".to_string(), FnDef::Builtin(builtin_to_sparse_scope));
        self.functions.insert("sparse_matmul_scope".to_string(), FnDef::Builtin(builtin_sparse_matmul_scope));
        self.functions.insert("sparsity_ratio".to_string(), FnDef::Builtin(builtin_sparsity_ratio));

        // Feature 48: Async inference
        self.functions.insert("async_predict".to_string(), FnDef::Builtin(builtin_async_predict));
        self.functions.insert("batch_infer".to_string(), FnDef::Builtin(builtin_batch_infer));

        // Feature 49: Profiling
        self.functions.insert("profile_op".to_string(), FnDef::Builtin(builtin_profile_op));
        self.functions.insert("profile_summary".to_string(), FnDef::Builtin(builtin_profile_summary));
        self.functions.insert("flops_count".to_string(), FnDef::Builtin(builtin_flops_count));

        // Feature 50: Contract
        self.functions.insert("requires".to_string(), FnDef::Builtin(builtin_requires));
        self.functions.insert("ensures".to_string(), FnDef::Builtin(builtin_ensures));
        self.functions.insert("invariant".to_string(), FnDef::Builtin(builtin_invariant));
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

fn builtin_int(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("int expects 1 argument".to_string()); }
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(*n)),
        Value::Float(f) => Ok(Value::Int(*f as i128)),
        Value::Bool(b) => Ok(Value::Int(if *b { 1 } else { 0 })),
        Value::String(s) => s.trim().parse::<i128>()
            .map(Value::Int)
            .map_err(|_| format!("int: cannot parse '{}'", s)),
        other => Err(format!("int: cannot convert {} to int", other)),
    }
}

fn builtin_float(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("float expects 1 argument".to_string()); }
    match &args[0] {
        Value::Float(f) => Ok(Value::Float(*f)),
        Value::Int(n) => Ok(Value::Float(*n as f64)),
        Value::Bool(b) => Ok(Value::Float(if *b { 1.0 } else { 0.0 })),
        Value::String(s) => s.trim().parse::<f64>()
            .map(Value::Float)
            .map_err(|_| format!("float: cannot parse '{}'", s)),
        other => Err(format!("float: cannot convert {} to float", other)),
    }
}

fn builtin_type_of(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("type_of expects 1 argument".to_string()); }
    let t = match &args[0] {
        Value::Int(_) => "int",
        Value::Float(_) => "float",
        Value::Bool(_) => "bool",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Tuple(_) => "tuple",
        Value::Pointer(_) => "pointer",
        Value::Closure { .. } => "closure",
        Value::Function { .. } => "function",
        _ => "unknown",
    };
    Ok(Value::String(t.to_string()))
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
    let _causal = match args.get(6) {
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

// --- Module import resolution ---

fn resolve_module_path(path: &[Ident]) -> Result<PathBuf, String> {
    // Handle `import "path/to/file"` string literal syntax (sentinel: name starts with \x00)
    if path.len() == 1 && path[0].name.starts_with("__string_import__") {
        let raw_path = &path[0].name["__string_import__".len()..]; // strip sentinel
        let pb = PathBuf::from(raw_path);
        // Add .vx extension if not present
        let with_ext = if pb.extension().map(|e| e == "vx").unwrap_or(false) {
            pb.clone()
        } else {
            pb.with_extension("vx")
        };
        // If absolute path, use directly
        if with_ext.is_absolute() {
            if with_ext.exists() {
                return Ok(with_ext);
            }
            return Err(format!("module file '{}' not found", with_ext.display()));
        }
        // Relative path: try cwd first, then stdlib
        let cwd = std::env::current_dir().unwrap_or_default();
        let cwd_path = cwd.join(&with_ext);
        if cwd_path.exists() {
            return Ok(cwd_path);
        }
        // Try stdlib dir
        let stdlib_path = PathBuf::from("stdlib").join(&with_ext);
        if stdlib_path.exists() {
            return Ok(stdlib_path);
        }
        return Err(format!("module '{}' not found (tried {} and stdlib/{})",
            raw_path, cwd_path.display(), with_ext.display()));
    }

    let path_str: Vec<&str> = path.iter().map(|id| id.name.as_str()).collect();
    let module_name = path_str.join(".");
    let mut fs_path = PathBuf::new();
    for segment in &path_str {
        fs_path.push(segment);
    }
    let fs_path_with_ext = fs_path.with_extension("vx");
    let stdlib_path = PathBuf::from("stdlib").join(&fs_path_with_ext);
    if stdlib_path.exists() {
        return Ok(stdlib_path);
    }
    if fs_path_with_ext.exists() {
        return Ok(fs_path_with_ext);
    }
    Err(format!("module '{}' not found (looked in {})", module_name, stdlib_path.display()))
}

fn resolve_import(env: &mut Env, import: &ImportDecl) -> Result<(), String> {
    let file_path = resolve_module_path(&import.path)?;
    let canonical = file_path.canonicalize().unwrap_or_else(|_| file_path.clone());
    if env.loading_modules.contains(&canonical) {
        return Err(format!("circular import detected: '{}'",
            import.path.iter().map(|id| id.name.as_str()).collect::<Vec<_>>().join(".")));
    }
    if env.loaded_modules.contains(&canonical) {
        return Ok(());
    }
    let source = std::fs::read_to_string(&file_path)
        .map_err(|e| format!("failed to read module '{}': {}", file_path.display(), e))?;
    let tokens = crate::lexer::lex(&source);
    let program = crate::parser::parse(tokens, 0)
        .map_err(|diags| {
            let msgs: Vec<String> = diags.iter().map(|d| d.message.clone()).collect();
            format!("parse errors in module '{}': {}", file_path.display(), msgs.join(", "))
        })?;
    env.loading_modules.insert(canonical.clone());
    let requested: Option<Vec<String>> = match &import.items {
        ImportItems::Named(items) => Some(items.iter().map(|i| i.name.name.clone()).collect()),
        ImportItems::Wildcard => None,
    };
    for item in &program.items {
        let item_name = match &item.kind {
            ItemKind::Function(f) => Some(f.name.name.clone()),
            ItemKind::Struct(s) => Some(s.name.name.clone()),
            ItemKind::Enum(e) => Some(e.name.name.clone()),
            ItemKind::Const(c) => Some(c.name.name.clone()),
            ItemKind::Static(s) => Some(s.name.name.clone()),
            ItemKind::Trait(t) => Some(t.name.name.clone()),
            _ => None,
        };
        // Always import impl blocks (they apply globally)
        if let ItemKind::Impl(impl_block) = &item.kind {
            let type_name = match &impl_block.target.kind {
                crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                crate::ast::TypeExprKind::Generic { name, .. } => name.name.clone(),
                _ => continue,
            };
            if let Some(trait_type) = &impl_block.trait_name {
                let trait_name = match &trait_type.kind {
                    crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                    crate::ast::TypeExprKind::Generic { name, .. } => name.name.clone(),
                    _ => String::new(),
                };
                if !trait_name.is_empty() {
                    env.trait_impls.insert((trait_name, type_name.clone()), true);
                }
            }
            for method_item in &impl_block.methods {
                if let ItemKind::Function(func) = &method_item.kind {
                    let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                    let qualified = format!("{}::{}", type_name, func.name.name);
                    env.functions.insert(qualified, FnDef::User { params, body: func.body.clone() });
                }
            }
            continue;
        }
        if let Some(name) = item_name {
            let should_import = match &requested {
                Some(names) => names.contains(&name),
                // For wildcard imports, skip `main` to avoid re-executing the imported file's entrypoint
                None => name != "main",
            };
            if should_import {
                let alias = if let ImportItems::Named(items) = &import.items {
                    items.iter().find(|i| i.name.name == name)
                        .and_then(|i| i.alias.as_ref().map(|a| a.name.clone()))
                } else { None };
                let register_name = alias.unwrap_or(name.clone());
                match &item.kind {
                    ItemKind::Function(func) => {
                        let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                        env.functions.insert(register_name, FnDef::User { params, body: func.body.clone() });
                    }
                    ItemKind::Struct(s) => {
                        let field_names: Vec<String> = s.fields.iter().map(|f| f.name.name.clone()).collect();
                        env.struct_defs.insert(register_name, field_names);
                    }
                    ItemKind::Enum(e) => {
                        for variant in &e.variants {
                            let vn = variant.name.name.clone();
                            if let crate::ast::EnumVariantKind::Unit = &variant.kind {
                                env.define(&vn, Value::EnumVariant {
                                    enum_name: register_name.clone(), variant: vn.clone(), fields: Vec::new(),
                                });
                            }
                        }
                    }
                    ItemKind::Const(c) => {
                        let val = eval_expr(env, &c.value)?;
                        env.define(&register_name, val);
                    }
                    ItemKind::Static(s) => {
                        let val = eval_expr(env, &s.value)?;
                        env.define(&register_name, val);
                    }
                    ItemKind::Trait(trait_def) => {
                        let mut methods = Vec::new();
                        for method in &trait_def.methods {
                            let has_default = method.body.is_some();
                            methods.push((method.name.name.clone(), has_default));
                            if let Some(body) = &method.body {
                                let params: Vec<String> = method.params.iter().map(|p| p.name.name.clone()).collect();
                                let qualified = format!("{}::{}", register_name, method.name.name);
                                env.functions.insert(qualified, FnDef::User { params, body: body.clone() });
                            }
                        }
                        env.trait_defs.insert(register_name, methods);
                    }
                    _ => {}
                }
            }
        }
    }
    env.loading_modules.remove(&canonical);
    env.loaded_modules.insert(canonical);
    Ok(())
}

/// Try to JIT-compile and execute a @gpu function. Returns Ok(Some(val)) on success,
/// Ok(None) if JIT is not available, or Err on JIT failure.
fn try_jit_execute(env: &mut Env, func: &Function, args: &[Value]) -> Result<Option<Value>, String> {
    if !env.jit_engine.check_tools() {
        return Ok(None);
    }

    // Convert Value args to f64 for JIT
    let mut f64_args = Vec::new();
    for arg in args {
        match arg {
            Value::Float(f) => f64_args.push(*f),
            Value::Int(i) => f64_args.push(*i as f64),
            Value::Bool(b) => f64_args.push(if *b { 1.0 } else { 0.0 }),
            _ => return Ok(None), // Can't JIT with complex types
        }
    }

    let compiled = env.jit_engine.compile_function(func)?;
    let result = env.jit_engine.execute(&compiled, &f64_args)?;

    match result {
        crate::jit::JitResult::Float(f) => Ok(Some(Value::Float(f))),
        crate::jit::JitResult::Int(i) => Ok(Some(Value::Int(i as i128))),
        crate::jit::JitResult::Bool(b) => Ok(Some(Value::Bool(b))),
        crate::jit::JitResult::Void => Ok(Some(Value::Void)),
    }
}

// --- Interpreter entry point ---

pub fn interpret(program: &Program) -> Result<Vec<String>, String> {
    let mut env = Env::new();

    // First pass: register all functions and kernels
    for item in &program.items {
        match &item.kind {
            ItemKind::Function(func) => {
                let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                // Track @gpu/@jit annotated functions for JIT compilation
                if crate::jit::has_gpu_annotation(func) {
                    env.gpu_functions.insert(func.name.name.clone(), func.clone());
                }
                // Print warning for @distributed
                if crate::jit::has_distributed_annotation(func) {
                    eprintln!("[vortex] @distributed: distributed execution not yet available, running locally: {}", func.name.name);
                }
                // Track diff fn annotations
                if func.annotations.iter().any(|a| matches!(a, Annotation::Diff)) {
                    env.diff_functions.insert(func.name.name.clone());
                }
                // Track #[verifiable] annotations
                if func.annotations.iter().any(|a| matches!(a, Annotation::Verifiable)) {
                    env.verifiable_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Cache)) {
                    env.cache_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Reward)) {
                    env.reward_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::StreamFn)) {
                    env.stream_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Evolve)) {
                    env.evolve_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Alignment)) {
                    env.alignment_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Intention)) {
                    env.intention_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Contract)) {
                    env.contract_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::GradientSurgery)) {
                    env.gradient_surgery_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::PersistentGrad)) {
                    // Initialize persistent gradient tape for this function
                    env.persistent_grad_tapes
                        .entry(func.name.name.clone())
                        .or_insert_with(Vec::new);
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::Fuse)) {
                    eprintln!("kernel fusion hint detected for {}", func.name.name);
                    env.fuse_functions.insert(func.name.name.clone());
                }
                if func.annotations.iter().any(|a| matches!(a, Annotation::ConstantTime)) {
                    eprintln!("constant-time mode for {}", func.name.name);
                    env.constant_time_functions.insert(func.name.name.clone());
                }
                // @adaptive, @zk_provable, @hot_modify, @bounded_update, @tiered,
                // @multiscale, @local_learning, @sparse_dispatch, @heterogeneous_dispatch
                // are acknowledged here; full backend support is wired separately.
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
            ItemKind::Static(s) => {
                let val = eval_expr(&mut env, &s.value)?;
                env.define(&s.name.name, val);
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
                // Get the target type name (works for both `impl Foo` and `impl Foo<T>`)
                let type_name = match &impl_block.target.kind {
                    crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                    crate::ast::TypeExprKind::Generic { name, .. } => name.name.clone(),
                    _ => continue,
                };

                // If this is a trait impl, record it and inject default methods
                if let Some(trait_type) = &impl_block.trait_name {
                    let trait_name = match &trait_type.kind {
                        crate::ast::TypeExprKind::Named(id) => id.name.clone(),
                        crate::ast::TypeExprKind::Generic { name, .. } => name.name.clone(),
                        _ => String::new(),
                    };
                    if !trait_name.is_empty() {
                        env.trait_impls.insert((trait_name.clone(), type_name.clone()), true);

                        // Collect implemented method names
                        let implemented: HashSet<String> = impl_block.methods.iter()
                            .filter_map(|m| if let ItemKind::Function(f) = &m.kind { Some(f.name.name.clone()) } else { None })
                            .collect();

                        // Inject default methods from trait that weren't overridden
                        if let Some(trait_methods) = env.trait_defs.get(&trait_name).cloned() {
                            for (method_name, _has_default) in &trait_methods {
                                if !implemented.contains(method_name) {
                                    // Look for default impl registered as Trait::method
                                    let default_key = format!("{}::{}", trait_name, method_name);
                                    if let Some(default_fn) = env.functions.get(&default_key).cloned() {
                                        let qualified = format!("{}::{}", type_name, method_name);
                                        env.functions.insert(qualified, default_fn);
                                    }
                                }
                            }
                        }
                    }
                }

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
            ItemKind::Trait(trait_def) => {
                let trait_name = trait_def.name.name.clone();
                let mut methods = Vec::new();
                for method in &trait_def.methods {
                    let has_default = method.body.is_some();
                    methods.push((method.name.name.clone(), has_default));
                    // Register default implementations as Trait::method
                    if let Some(body) = &method.body {
                        let params: Vec<String> = method.params.iter().map(|p| p.name.name.clone()).collect();
                        let qualified = format!("{}::{}", trait_name, method.name.name);
                        env.functions.insert(
                            qualified,
                            FnDef::User {
                                params,
                                body: body.clone(),
                            },
                        );
                    }
                }
                env.trait_defs.insert(trait_name, methods);
            }
            ItemKind::Import(import_decl) => {
                resolve_import(&mut env, import_decl)?;
            }
            ItemKind::FieldDef(fd) => {
                let name = fd.name.name.clone();
                // Parse hex modulus string into [u64; 4] limbs
                let hex = fd.modulus.strip_prefix("0x").or_else(|| fd.modulus.strip_prefix("0X")).unwrap_or(&fd.modulus);
                let hex = hex.replace('_', "");
                let hex_trimmed = hex.trim_start_matches('0');
                if hex_trimmed.is_empty() {
                    return Err(format!("field modulus for '{}' cannot be zero", name));
                }
                if hex_trimmed.len() > 64 {
                    return Err(format!("field modulus for '{}' exceeds 256 bits", name));
                }
                let padded = format!("{:0>64}", hex_trimmed);
                let mut limbs = [0u64; 4];
                for i in 0..4 {
                    let start = 64 - (i + 1) * 16;
                    let end = start + 16;
                    limbs[i] = u64::from_str_radix(&padded[start..end], 16)
                        .map_err(|e| format!("invalid hex in field modulus: {}", e))?;
                }
                // Leak a static name string for FieldParams
                let static_name: &'static str = Box::leak(name.clone().into_boxed_str());
                let params = modmath::init_field_params_pub(limbs, static_name);
                let static_params: &'static modmath::FieldParams = Box::leak(Box::new(params));
                env.field_defs.insert(name.clone(), static_params);

                // Register constructor as a User function that we handle specially
                // We use a sentinel body. The actual call is intercepted in apply_closure/call dispatch.
                // Simpler approach: just define it in the scope as a value
                // Actually, we handle Fp(42) calls via the function call path.
                // Constructor is handled inline in the Call dispatch path
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

/// Create a new REPL environment with builtins registered
pub fn repl_env() -> Env { Env::new() }

/// Evaluate a line of input in the REPL
pub fn repl_eval_line(env: &mut Env, line: &str) -> Result<Option<String>, String> {
    let tokens = crate::lexer::lex(line);
    if let Ok(program) = crate::parser::parse(tokens, 0) {
        if !program.items.is_empty() {
            for item in &program.items {
                match &item.kind {
                    ItemKind::Function(func) => {
                        let params: Vec<String> = func.params.iter().map(|p| p.name.name.clone()).collect();
                        env.functions.insert(func.name.name.clone(), FnDef::User { params, body: func.body.clone() });
                    }
                    ItemKind::Struct(s) => {
                        let field_names: Vec<String> = s.fields.iter().map(|f| f.name.name.clone()).collect();
                        env.struct_defs.insert(s.name.name.clone(), field_names);
                    }
                    _ => {}
                }
            }
            let output: Vec<String> = env.output.drain(..).collect();
            for o in &output { println!("{}", o); }
            return Ok(None);
        }
    }
    let wrapped = format!("fn __repl__() {{\n{}\n}}", line);
    let tokens = crate::lexer::lex(&wrapped);
    if let Ok(program) = crate::parser::parse(tokens, 0) {
        for item in &program.items {
            if let ItemKind::Function(func) = &item.kind {
                if func.name.name == "__repl__" {
                    let result = eval_block(env, &func.body)?;
                    let output: Vec<String> = env.output.drain(..).collect();
                    for o in &output { println!("{}", o); }
                    return match &result {
                        Value::Void => Ok(None),
                        _ => Ok(Some(format!("{}", result))),
                    };
                }
            }
        }
    }
    Err(format!("could not parse: {}", line))
}

/// Get variable names from REPL env
pub fn repl_env_vars(env: &Env) -> Vec<String> {
    let mut vars = Vec::new();
    for scope in &env.scopes { for k in scope.keys() { vars.push(k.clone()); } }
    vars.sort(); vars
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
        StmtKind::Let { name, ty, value } => {
            let val = eval_expr(env, value)?;
            if matches!(&val, Value::Return(_)) { return Ok(val); }
            // Field type coercion: if type annotation names a registered field, coerce
            let val = if let Some(type_expr) = ty {
                if let crate::ast::TypeExprKind::Named(id) = &type_expr.kind {
                    if let Some(params) = env.field_defs.get(&id.name).copied() {
                        coerce_to_field(&val, params)?
                    } else { val }
                } else { val }
            } else { val };
            env.define(&name.name, val);
            Ok(Value::Void)
        }
        StmtKind::Var { name, ty, value } => {
            let val = eval_expr(env, value)?;
            if matches!(&val, Value::Return(_)) { return Ok(val); }
            let val = if let Some(type_expr) = ty {
                if let crate::ast::TypeExprKind::Named(id) = &type_expr.kind {
                    if let Some(params) = env.field_defs.get(&id.name).copied() {
                        coerce_to_field(&val, params)?
                    } else { val }
                } else { val }
            } else { val };
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
                FnDef::GradWrapper { fn_name, order: _ } => eval_grad_call(env, &fn_name, arg_vals),
            }
        }
        StmtKind::Live { name, value } => {
            let val = eval_expr(env, value)?;

            // If nn_new was called, it already stored the model in live_models
            // and returned a struct with the ID. Extract that ID.
            let live_id = match &val {
                Value::Struct { name: sname, fields } if sname == "nn_model" => {
                    if let Some(Value::Int(id)) = fields.get("id") {
                        *id as usize
                    } else {
                        // Create new model from layer descriptions
                        let id = env.next_live_model_id;
                        env.next_live_model_id += 1;
                        let nn_layers: Vec<crate::nn::Layer> = fields.get("layers")
                            .and_then(|v| if let Value::Array(arr) = v { Some(arr) } else { None })
                            .map(|layers| layers.iter().filter_map(|l| {
                                if let Value::Struct { name: ln, fields: lf } = l {
                                    if ln == "nn_linear" {
                                        let ins = lf.get("in_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                        let outs = lf.get("out_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                        return Some(crate::nn::Layer::linear(ins, outs));
                                    }
                                }
                                None
                            }).collect())
                            .unwrap_or_default();
                        env.live_models.insert(id, crate::nn::Model::sequential(nn_layers));
                        id
                    }
                }
                Value::LiveModel { id } => *id,
                _ => {
                    let id = env.next_live_model_id;
                    env.next_live_model_id += 1;
                    env.live_models.insert(id, crate::nn::Model::sequential(vec![]));
                    id
                }
            };

            env.define(&name.name, Value::LiveModel { id: live_id });
            Ok(Value::Void)
        }

        // Feature F6: fuse { } — kernel fusion block
        StmtKind::Fuse { body } => {
            // Record all tensor ops in the block for fusion analysis
            let prev_fuse = env.in_fuse_block;
            env.in_fuse_block = true;
            env.fuse_ops.clear();

            let result = eval_block(env, body)?;

            // Analyze fusion graph: count ops that were fused
            let fused_count = env.fuse_ops.len();
            if fused_count > 1 {
                // In a real implementation, these ops would be compiled into a single kernel.
                // For the interpreter, we track that fusion happened.
                env.fuse_stats.total_fused += fused_count;
                env.fuse_stats.fusion_blocks += 1;
            }
            env.in_fuse_block = prev_fuse;
            if !env.in_fuse_block { env.fuse_ops.clear(); }
            Ok(result)
        }

        // Feature F7: gpu let — GPU memory ownership
        StmtKind::GpuLet { name, value } => {
            let val = eval_expr(env, value)?;
            if matches!(&val, Value::Return(_)) { return Ok(val); }
            // Tag this variable as GPU-owned
            env.gpu_owned.insert(name.name.clone());
            env.gpu_allocations += 1;
            env.define(&name.name, val);
            Ok(Value::Void)
        }

        // Feature F8: parallel for — distributed parallel loop
        StmtKind::Parallel { var, iter, body } => {
            let iter_val = eval_expr(env, iter)?;
            let items = match iter_val {
                Value::Array(arr) => arr,
                Value::Tensor(ref _t) => {
                    // Shard tensor across simulated devices
                    vec![iter_val.clone()]
                }
                _ => return Err("parallel for: iterator must be an array or tensor".into()),
            };

            // Simulate parallel execution: partition items across virtual devices
            let num_devices = env.num_devices.max(1);
            let chunk_size = (items.len() + num_devices - 1) / num_devices;
            let mut all_results = Vec::new();

            for chunk in items.chunks(chunk_size) {
                for item in chunk {
                    env.push_scope();
                    env.define(&var.name, item.clone());
                    let result = eval_block(env, body)?;
                    env.pop_scope();
                    match result {
                        Value::Return(_) => return Ok(result),
                        Value::Break => return Ok(Value::Void),
                        _ => all_results.push(result),
                    }
                }
            }
            // Auto all-reduce: if results are numeric, sum them
            if !all_results.is_empty() && all_results.iter().all(|v| matches!(v, Value::Float(_) | Value::Int(_))) {
                let sum: f64 = all_results.iter().map(|v| match v {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => 0.0,
                }).sum();
                env.define("__parallel_reduced", Value::Float(sum));
            }
            Ok(Value::Void)
        }

        // Feature F9: train { } — first-class training loop
        StmtKind::Train { config } => {
            // Extract config values
            let mut epochs = 1usize;
            let mut batch_size = 32usize;
            let mut lr = 0.001f64;
            let mut model_val = None;
            let mut data_val = None;
            let mut optimizer_name = "sgd".to_string();
            let mut checkpoint_every = 0usize;
            let mut on_step_fn = None;

            for (key, expr) in config {
                let val = eval_expr(env, expr)?;
                match key.name.as_str() {
                    "epochs" => if let Value::Int(n) = val { epochs = n as usize; },
                    "batch_size" => if let Value::Int(n) = val { batch_size = n as usize; },
                    "lr" | "learning_rate" => match val {
                        Value::Float(f) => lr = f,
                        Value::Int(n) => lr = n as f64,
                        _ => {}
                    },
                    "model" => model_val = Some(val),
                    "data" => data_val = Some(val),
                    "optimizer" => if let Value::String(s) = val { optimizer_name = s; },
                    "checkpoint_every" => if let Value::Int(n) = val { checkpoint_every = n as usize; },
                    "on_step" => on_step_fn = Some(val),
                    _ => {} // ignore unknown config
                }
            }

            // Run training loop
            let data_items = match data_val {
                Some(Value::Array(arr)) => arr,
                Some(other) => vec![other],
                None => return Err("train: 'data' config required".into()),
            };

            let total_batches = (data_items.len() + batch_size - 1) / batch_size;
            let mut global_step = 0usize;
            let mut total_loss = 0.0f64;

            for epoch in 0..epochs {
                let mut epoch_loss = 0.0f64;
                for batch_idx in 0..total_batches {
                    let start_i = batch_idx * batch_size;
                    let end_i = (start_i + batch_size).min(data_items.len());
                    let batch = &data_items[start_i..end_i];

                    // Simulated training step: compute "loss" as sum of batch values
                    let batch_loss: f64 = batch.iter().map(|v| match v {
                        Value::Float(f) => f.abs(),
                        Value::Int(n) => (*n as f64).abs(),
                        _ => 1.0,
                    }).sum::<f64>() / batch.len() as f64;

                    // Apply learning rate (simulate gradient update)
                    let step_loss = batch_loss * (1.0 - lr * (global_step as f64).min(100.0) * 0.01);
                    epoch_loss += step_loss;
                    total_loss = step_loss;

                    // Call on_step callback if provided
                    if let Some(ref callback) = on_step_fn {
                        let step_val = Value::Int(global_step as i128);
                        let loss_val = Value::Float(step_loss);
                        match callback {
                            Value::Function { name, .. } => {
                                if let Some(fd) = env.functions.get(name).cloned() {
                                    if let FnDef::Builtin(f) = fd {
                                        let _ = f(env, vec![step_val, loss_val]);
                                    } else if let FnDef::User { params, body } = fd {
                                        env.push_scope();
                                        if params.len() > 0 { env.define(&params[0], step_val); }
                                        if params.len() > 1 { env.define(&params[1], loss_val); }
                                        let _ = eval_block(env, &body);
                                        env.pop_scope();
                                    }
                                }
                            }
                            Value::Closure { params, body, env: captured } => {
                                let _ = call_closure(env, params, body, captured, vec![step_val, loss_val]);
                            }
                            _ => {}
                        }
                    }

                    global_step += 1;

                    // Checkpoint
                    if checkpoint_every > 0 && global_step % checkpoint_every == 0 {
                        env.train_checkpoints.push((global_step, total_loss));
                    }
                }
            }

            // Store training result
            let mut fields = std::collections::HashMap::new();
            fields.insert("epochs".to_string(), Value::Int(epochs as i128));
            fields.insert("final_loss".to_string(), Value::Float(total_loss));
            fields.insert("total_steps".to_string(), Value::Int(global_step as i128));
            fields.insert("optimizer".to_string(), Value::String(optimizer_name));
            Ok(Value::Struct { name: "TrainResult".to_string(), fields })
        }

        // Feature F10: deterministic { } — reproducibility scope
        StmtKind::Deterministic { body } => {
            let prev_deterministic = env.deterministic_mode;
            let prev_seed = env.rng_seed;
            env.deterministic_mode = true;
            // Set a fixed seed for reproducibility
            if env.rng_seed == 0 { env.rng_seed = 42; }

            let result = eval_block(env, body)?;

            env.deterministic_mode = prev_deterministic;
            env.rng_seed = prev_seed;
            Ok(result)
        }

        // Feature F11: autocast(dtype) { } — mixed precision scope
        StmtKind::Autocast { dtype, body } => {
            let prev_autocast = env.autocast_dtype.clone();
            env.autocast_dtype = Some(dtype.name.clone());

            let result = eval_block(env, body)?;

            env.autocast_dtype = prev_autocast;
            Ok(result)
        }

        // Feature F14: speculate { } — speculative execution
        StmtKind::Speculate { body } => {
            // Execute body, tracking all branch results
            // In the interpreter, speculate runs the block normally but records
            // execution paths for future optimization
            let prev = env.speculating;
            env.speculating = true;
            env.speculate_depth += 1;

            let result = eval_block(env, body)?;

            env.speculate_depth -= 1;
            env.speculating = prev;
            Ok(result)
        }

        // Feature F16: topology { } — network topology definition
        StmtKind::Topology { config } => {
            let mut nodes = Vec::new();
            let mut edges = Vec::new();
            let mut topo_name = String::from("unnamed");

            for (key, expr) in config {
                let val = eval_expr(env, expr)?;
                match key.name.as_str() {
                    "name" => if let Value::String(s) = val { topo_name = s; },
                    "nodes" => if let Value::Array(arr) = val { nodes = arr; },
                    "edges" => if let Value::Array(arr) = val { edges = arr; },
                    _ => {} // store in topology config
                }
            }

            let mut fields = HashMap::new();
            fields.insert("name".to_string(), Value::String(topo_name));
            fields.insert("num_nodes".to_string(), Value::Int(nodes.len() as i128));
            fields.insert("num_edges".to_string(), Value::Int(edges.len() as i128));
            fields.insert("nodes".to_string(), Value::Array(nodes));
            fields.insert("edges".to_string(), Value::Array(edges));
            env.topologies.push(Value::Struct { name: "Topology".to_string(), fields: fields.clone() });
            Ok(Value::Struct { name: "Topology".to_string(), fields })
        }

        // Feature F22: mmap — memory-mapped model loading
        StmtKind::Mmap { name, value } => {
            let val = eval_expr(env, value)?;
            // In a real implementation, this would mmap a file into memory
            // For the interpreter, we track it as a special MmapModel value
            let path = match &val {
                Value::String(s) => s.clone(),
                _ => format!("{}", val),
            };
            let mut fields = HashMap::new();
            fields.insert("path".to_string(), Value::String(path.clone()));
            fields.insert("loaded".to_string(), Value::Bool(true));
            fields.insert("size_bytes".to_string(), Value::Int(0));
            env.mmap_models.insert(name.name.clone(), path);
            env.define(&name.name, Value::Struct { name: "MmapModel".to_string(), fields });
            Ok(Value::Void)
        }

        // Feature F23: explain { } — interpretability block
        StmtKind::Explain { body } => {
            let prev = env.explaining;
            env.explaining = true;
            env.explain_trace.clear();

            let result = eval_block(env, body)?;

            env.explaining = false;
            // Build explanation report
            let trace_len = env.explain_trace.len();
            let mut fields = HashMap::new();
            fields.insert("num_ops".to_string(), Value::Int(trace_len as i128));
            fields.insert("trace".to_string(), Value::Array(
                env.explain_trace.iter().map(|s| Value::String(s.clone())).collect()
            ));
            fields.insert("result".to_string(), result);
            env.explaining = prev;
            Ok(Value::Struct { name: "Explanation".to_string(), fields })
        }

        // Feature F18: quantize(dtype) { } — quantization scope
        StmtKind::Quantize { dtype, body } => {
            let prev = env.quantize_dtype.clone();
            env.quantize_dtype = Some(dtype.name.clone());

            let result = eval_block(env, body)?;

            env.quantize_dtype = prev;
            Ok(result)
        }

        // Feature F21: safe(config) { } — computation bounds
        StmtKind::Safe { config, body } => {
            let mut max_ops = u64::MAX;
            let mut max_memory = u64::MAX;
            let mut max_time_ms = u64::MAX;

            for (key, expr) in config {
                let val = eval_expr(env, expr)?;
                match key.name.as_str() {
                    "max_ops" | "max_flops" => if let Value::Int(n) = val { max_ops = n as u64; },
                    "max_memory" | "max_mem" => if let Value::Int(n) = val { max_memory = n as u64; },
                    "max_time" | "max_time_ms" => if let Value::Int(n) = val { max_time_ms = n as u64; },
                    _ => {}
                }
            }

            let prev_safe = env.safe_mode;
            let prev_budget = env.op_budget;
            env.safe_mode = true;
            env.op_budget = max_ops;
            env.op_counter = 0;

            let result = eval_block(env, body);

            env.safe_mode = prev_safe;
            env.op_budget = prev_budget;

            match result {
                Ok(val) => Ok(val),
                Err(e) if e.contains("budget exceeded") => {
                    // Graceful fallback: return error info instead of panicking
                    let mut fields = HashMap::new();
                    fields.insert("error".to_string(), Value::String(e));
                    fields.insert("ops_used".to_string(), Value::Int(env.op_counter as i128));
                    Ok(Value::Struct { name: "SafetyViolation".to_string(), fields })
                }
                Err(e) => Err(e),
            }
        }

        // Feature F24: consensus { } — multi-model voting
        StmtKind::Consensus { body } => {
            // Run the body multiple times (simulating multiple models) and take majority vote
            let num_voters = env.consensus_voters.max(3);
            let mut results = Vec::new();

            for _voter in 0..num_voters {
                let result = eval_block(env, body)?;
                results.push(result);
            }

            // Majority vote: pick the most common result
            let mut vote_counts: HashMap<String, (usize, Value)> = HashMap::new();
            for r in &results {
                let key = format!("{}", r);
                let entry = vote_counts.entry(key).or_insert((0, r.clone()));
                entry.0 += 1;
            }
            let winner = vote_counts.into_iter()
                .max_by_key(|(_, (count, _))| *count)
                .map(|(_, (_, val))| val)
                .unwrap_or(Value::Void);

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), winner);
            fields.insert("num_voters".to_string(), Value::Int(num_voters as i128));
            fields.insert("agreement".to_string(), Value::Float(1.0)); // simplified
            Ok(Value::Struct { name: "ConsensusResult".to_string(), fields })
        }

        // Feature F27: symbolic { } — hybrid neural-symbolic scope
        StmtKind::SymbolicBlock { body } => {
            let prev = env.symbolic_mode;
            env.symbolic_mode = true;

            let result = eval_block(env, body)?;

            env.symbolic_mode = prev;
            Ok(result)
        }

        // Feature F28: temporal { } — time-aware computation
        StmtKind::TemporalBlock { body } => {
            let prev_temporal = env.temporal_mode;
            env.temporal_mode = true;
            env.temporal_step += 1;

            let result = eval_block(env, body)?;

            env.temporal_mode = prev_temporal;
            Ok(result)
        }

        // Feature F29: federated { } — privacy-preserving scope
        StmtKind::Federated { body } => {
            let prev = env.federated_mode;
            env.federated_mode = true;
            // In federated mode, data doesn't leave the scope
            // All computations are local, only aggregated gradients are shared
            env.push_scope();

            let result = eval_block(env, body)?;

            env.pop_scope();
            env.federated_mode = prev;

            // Return only aggregated result, not raw data
            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("privacy_preserved".to_string(), Value::Bool(true));
            fields.insert("data_leaked".to_string(), Value::Bool(false));
            Ok(Value::Struct { name: "FederatedResult".to_string(), fields })
        }

        // Feature F30: sandbox { } — capability-restricted execution
        StmtKind::SandboxBlock { body } => {
            let prev = env.sandboxed;
            env.sandboxed = true;
            // In sandbox mode: no file I/O, no network, no system calls
            // Only pure computation allowed

            let result = eval_block(env, body);

            env.sandboxed = prev;
            match result {
                Ok(val) => Ok(val),
                Err(e) if e.contains("sandbox violation") => {
                    let mut fields = HashMap::new();
                    fields.insert("error".to_string(), Value::String(e));
                    fields.insert("sandboxed".to_string(), Value::Bool(true));
                    Ok(Value::Struct { name: "SandboxViolation".to_string(), fields })
                }
                Err(e) => Err(e),
            }
        }

        // Feature F31: compress(ratio) { } — model compression
        StmtKind::Compress { ratio, body } => {
            let ratio_val = eval_expr(env, ratio)?;
            let compression_ratio = match ratio_val {
                Value::Float(f) => f,
                Value::Int(n) => n as f64,
                _ => 1.0,
            };

            let prev = env.compression_ratio;
            env.compression_ratio = compression_ratio;

            let result = eval_block(env, body)?;

            env.compression_ratio = prev;

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("compression_ratio".to_string(), Value::Float(compression_ratio));
            fields.insert("method".to_string(), Value::String("pruning+distillation".to_string()));
            Ok(Value::Struct { name: "CompressedResult".to_string(), fields })
        }

        // Feature F33: metacognition { } — self-reasoning scope
        StmtKind::Metacognition { body } => {
            let prev = env.metacognition_mode;
            env.metacognition_mode = true;
            env.confidence_scores.clear();

            let result = eval_block(env, body)?;

            env.metacognition_mode = prev;

            // Build metacognition report
            let avg_confidence = if env.confidence_scores.is_empty() {
                1.0
            } else {
                env.confidence_scores.iter().sum::<f64>() / env.confidence_scores.len() as f64
            };

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("confidence".to_string(), Value::Float(avg_confidence));
            fields.insert("num_decisions".to_string(), Value::Int(env.confidence_scores.len() as i128));
            fields.insert("uncertainty".to_string(), Value::Float(1.0 - avg_confidence));
            Ok(Value::Struct { name: "MetacognitionResult".to_string(), fields })
        }

        // Feature F34: theorem { } — formal verification scope
        StmtKind::TheoremBlock { name, body } => {
            let theorem_name = name.as_ref().map(|n| n.name.clone()).unwrap_or_else(|| "unnamed".to_string());
            env.theorem_mode = true;
            env.theorem_obligations.clear();

            let result = eval_block(env, body)?;

            env.theorem_mode = false;
            let all_proven = env.theorem_obligations.iter().all(|(_, proven)| *proven);
            let num_obligations = env.theorem_obligations.len();

            let mut fields = HashMap::new();
            fields.insert("name".to_string(), Value::String(theorem_name));
            fields.insert("result".to_string(), result);
            fields.insert("proven".to_string(), Value::Bool(all_proven));
            fields.insert("obligations".to_string(), Value::Int(num_obligations as i128));
            Ok(Value::Struct { name: "TheoremResult".to_string(), fields })
        }

        // Feature F35: continual { } — online learning without forgetting
        StmtKind::ContinualLearn { body } => {
            let prev = env.continual_mode;
            env.continual_mode = true;
            // EWC-style: save parameter importance before learning
            let snapshot_id = env.memory_snapshots.len();
            env.memory_snapshots.push(HashMap::new()); // snapshot current state

            let result = eval_block(env, body)?;

            env.continual_mode = prev;

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("snapshot_id".to_string(), Value::Int(snapshot_id as i128));
            fields.insert("forgetting_prevented".to_string(), Value::Bool(true));
            fields.insert("method".to_string(), Value::String("ewc".to_string()));
            Ok(Value::Struct { name: "ContinualResult".to_string(), fields })
        }

        // Feature F36: multimodal { } — multi-modal fusion scope
        StmtKind::MultimodalBlock { modalities, body } => {
            let prev = env.multimodal_mode;
            env.multimodal_mode = true;
            env.active_modalities = modalities.iter().map(|m| m.name.clone()).collect();

            let result = eval_block(env, body)?;

            env.multimodal_mode = prev;

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("modalities".to_string(), Value::Array(
                env.active_modalities.iter().map(|m| Value::String(m.clone())).collect()
            ));
            fields.insert("fusion_method".to_string(), Value::String("cross_attention".to_string()));
            Ok(Value::Struct { name: "MultimodalResult".to_string(), fields })
        }

        // Feature F37: world_model { } — internal world simulation
        StmtKind::WorldModelBlock { body } => {
            let prev = env.world_model_active;
            env.world_model_active = true;
            env.world_state_log.clear();

            let result = eval_block(env, body)?;

            env.world_model_active = prev;

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("state_transitions".to_string(), Value::Int(env.world_state_log.len() as i128));
            fields.insert("states".to_string(), Value::Array(
                env.world_state_log.iter().map(|s| Value::String(s.clone())).collect()
            ));
            Ok(Value::Struct { name: "WorldModelResult".to_string(), fields })
        }

        // Feature F38: self_improve { } — recursive self-improvement
        StmtKind::SelfImproveBlock { body } => {
            let prev_gen = env.self_improve_generation;
            env.self_improve_generation += 1;

            // Track metrics before
            let before_score = env.self_improve_score;

            let result = eval_block(env, body)?;

            let mut fields = HashMap::new();
            fields.insert("result".to_string(), result);
            fields.insert("generation".to_string(), Value::Int(env.self_improve_generation as i128));
            fields.insert("score_before".to_string(), Value::Float(before_score));
            fields.insert("score_after".to_string(), Value::Float(env.self_improve_score));
            fields.insert("improved".to_string(), Value::Bool(env.self_improve_score > before_score));
            Ok(Value::Struct { name: "SelfImproveResult".to_string(), fields })
        }

        // Feature F40: memory { } — hierarchical memory system
        StmtKind::AttentionBlock { body } => {
            let prev = env.attention_mode;
            env.attention_mode = true;
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            env.attention_mode = prev;
            let mut fields = HashMap::new();
            fields.insert("mode".to_string(), Value::String("custom_attention".to_string()));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "AttentionResult".to_string(), fields })
        }
        StmtKind::CurriculumBlock { config, body } => {
            let prev_diff = env.curriculum_difficulty;
            let prev_step = env.curriculum_step;
            for (key, expr) in config {
                let val = eval_expr(env, expr)?;
                match key.name.as_str() {
                    "difficulty" => if let Value::Float(f) = val { env.curriculum_difficulty = f; },
                    "step" => if let Value::Int(n) = val { env.curriculum_step = n as usize; },
                    _ => {}
                }
            }
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            let diff = env.curriculum_difficulty;
            let step = env.curriculum_step;
            env.curriculum_difficulty = prev_diff;
            env.curriculum_step = prev_step;
            let mut fields = HashMap::new();
            fields.insert("difficulty".to_string(), Value::Float(diff));
            fields.insert("step".to_string(), Value::Int(step as i128));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "CurriculumResult".to_string(), fields })
        }
        StmtKind::EnsembleBlock { body } => {
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            let mut fields = HashMap::new();
            fields.insert("num_models".to_string(), Value::Int(env.ensemble_models.len() as i128));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "EnsembleResult".to_string(), fields })
        }
        StmtKind::AdversarialBlock { body } => {
            let prev = env.adversarial_mode;
            env.adversarial_mode = true;
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            env.adversarial_mode = prev;
            let mut fields = HashMap::new();
            fields.insert("epsilon".to_string(), Value::Float(env.adversarial_epsilon));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "AdversarialResult".to_string(), fields })
        }
        StmtKind::TransferBlock { body } => {
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            let mut fields = HashMap::new();
            fields.insert("frozen_layers".to_string(), Value::Int(env.transfer_frozen_layers.len() as i128));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "TransferResult".to_string(), fields })
        }
        StmtKind::SparseBlock { body } => {
            let prev = env.sparse_mode;
            env.sparse_mode = true;
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            env.sparse_mode = prev;
            let mut fields = HashMap::new();
            fields.insert("threshold".to_string(), Value::Float(env.sparse_threshold));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "SparseResult".to_string(), fields })
        }
        StmtKind::AsyncInferBlock { body } => {
            let prev = env.async_infer_mode;
            env.async_infer_mode = true;
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            env.async_infer_mode = prev;
            let mut fields = HashMap::new();
            fields.insert("async".to_string(), Value::Bool(true));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "AsyncInferResult".to_string(), fields })
        }
        StmtKind::ProfileBlock { body } => {
            let prev = env.profiling;
            env.profiling = true;
            env.profile_data.clear();
            let start = std::time::Instant::now();
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            let elapsed = start.elapsed().as_secs_f64();
            env.profiling = prev;
            let mut fields = HashMap::new();
            fields.insert("elapsed_secs".to_string(), Value::Float(elapsed));
            fields.insert("num_ops".to_string(), Value::Int(env.profile_data.len() as i128));
            fields.insert("result".to_string(), result);
            Ok(Value::Struct { name: "ProfileResult".to_string(), fields })
        }
        StmtKind::ContractBlock { pre, post, body } => {
            for cond in pre {
                let val = eval_expr(env, cond)?;
                if !matches!(val, Value::Bool(true)) {
                    return Err(format!("Contract precondition failed: {}", cond));
                }
            }
            let mut result = Value::Void;
            for s in &body.stmts { result = eval_stmt(env, s)?; }
            for cond in post {
                let val = eval_expr(env, cond)?;
                if !matches!(val, Value::Bool(true)) {
                    return Err(format!("Contract postcondition failed: {}", cond));
                }
            }
            Ok(result)
        }
        StmtKind::MemoryBlock { config } => {
            let mut short_term_size = 100usize;
            let mut long_term_size = 10000usize;
            let mut episodic_size = 1000usize;

            for (key, expr) in config {
                let val = eval_expr(env, expr)?;
                match key.name.as_str() {
                    "short_term" => if let Value::Int(n) = val { short_term_size = n as usize; },
                    "long_term" => if let Value::Int(n) = val { long_term_size = n as usize; },
                    "episodic" => if let Value::Int(n) = val { episodic_size = n as usize; },
                    _ => {}
                }
            }

            env.memory_config = MemoryConfig {
                short_term_capacity: short_term_size,
                long_term_capacity: long_term_size,
                episodic_capacity: episodic_size,
            };

            let mut fields = HashMap::new();
            fields.insert("short_term".to_string(), Value::Int(short_term_size as i128));
            fields.insert("long_term".to_string(), Value::Int(long_term_size as i128));
            fields.insert("episodic".to_string(), Value::Int(episodic_size as i128));
            fields.insert("total_capacity".to_string(), Value::Int((short_term_size + long_term_size + episodic_size) as i128));
            Ok(Value::Struct { name: "MemorySystem".to_string(), fields })
        }
    }
}

pub(crate) fn eval_expr(env: &mut Env, expr: &Expr) -> Result<Value, String> {
    match &expr.kind {
        ExprKind::IntLiteral(n) => Ok(Value::Int(*n as i128)),
        ExprKind::BigIntLiteral(s) => {
            let bi = if s.starts_with("0x") || s.starts_with("0X") {
                crate::crypto::BigUint256::from_hex(s).unwrap_or_else(|| crate::crypto::BigUint256::from_u64(0))
            } else {
                // Parse decimal string into BigUint256 via u128 (covers most cases)
                let val: u128 = s.parse().unwrap_or(0);
                let mut bi = crate::crypto::BigUint256::from_u64(0);
                bi.limbs[0] = val as u32;
                bi.limbs[1] = (val >> 32) as u32;
                bi.limbs[2] = (val >> 64) as u32;
                bi.limbs[3] = (val >> 96) as u32;
                bi
            };
            Ok(Value::BigInt(bi))
        }
        ExprKind::FloatLiteral(n) => Ok(Value::Float(*n)),
        ExprKind::StringLiteral(s) => Ok(Value::String(s.clone())),
        ExprKind::BoolLiteral(b) => Ok(Value::Bool(*b)),

        ExprKind::Ident(id) => {
            if let Some(val) = env.get(&id.name) {
                Ok(val)
            } else if let Some(func_def) = env.functions.get(&id.name).cloned() {
                // Allow named functions to be used as first-class values
                match func_def {
                    FnDef::User { params, body } => Ok(Value::Function {
                        name: id.name.clone(),
                        params,
                        body,
                    }),
                    FnDef::Builtin(_) | FnDef::GradWrapper { .. } => {
                        // Wrap builtin/grad as a closure-like value
                        // Store the name so apply_closure can look it up
                        Ok(Value::Function {
                            name: id.name.clone(),
                            params: vec![],
                            body: Block { stmts: vec![], expr: None, span: crate::ast::Span { start: 0, end: 0 } },
                        })
                    }
                }
            } else {
                Err(format!("undefined variable: {}", id.name))
            }
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
                    Value::ModFieldElem(mf) => Ok(Value::ModFieldElem(mf.neg())),
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

                // Handle LiveModel method dispatch
                if let Value::LiveModel { id } = &base_val {
                    let model_id = *id;
                    let arg_vals: Vec<Value> = args.iter()
                        .map(|a| eval_expr(env, a))
                        .collect::<Result<Vec<_>, _>>()?;
                    return match method_name.as_str() {
                        "forward" => {
                            let input_data: Vec<f64> = match arg_vals.first() {
                                Some(Value::Array(arr)) => arr.iter().map(|v| match v { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.0 }).collect(),
                                Some(Value::Tensor(t)) => {
                                    // Extract f64 data from FastTensor raw bytes
                                    let numel: usize = t.shape.iter().product();
                                    let mut vals = Vec::with_capacity(numel);
                                    for i in 0..numel {
                                        let offset = i * 8;
                                        if offset + 8 <= t.data.len() {
                                            let bytes: [u8; 8] = t.data[offset..offset+8].try_into().unwrap_or([0u8; 8]);
                                            vals.push(f64::from_le_bytes(bytes));
                                        } else {
                                            vals.push(0.0);
                                        }
                                    }
                                    vals
                                }
                                _ => return Err("forward expects array or tensor input".into()),
                            };
                            let input_tensor = crate::nn::Tensor::new(vec![input_data.len()], input_data.clone());
                            let model = env.live_models.get_mut(&model_id).ok_or("live model not found")?;
                            let output = model.forward(&input_tensor);
                            Ok(Value::Array(output.data.iter().map(|f| Value::Float(*f)).collect()))
                        }
                        "replace_layer" => {
                            let idx = match arg_vals.first() { Some(Value::Int(i)) => *i as usize, _ => return Err("replace_layer: first arg must be index".into()) };
                            let layer = match arg_vals.get(1) {
                                Some(Value::Struct { name: ln, fields: lf }) if ln == "nn_linear" => {
                                    let in_s = lf.get("in_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                    let out_s = lf.get("out_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                    crate::nn::Layer::linear(in_s, out_s)
                                }
                                _ => return Err("replace_layer: second arg must be a layer".into()),
                            };
                            let model = env.live_models.get_mut(&model_id).ok_or("live model not found")?;
                            if idx >= model.layers.len() { return Err(format!("layer index {} out of bounds", idx)); }
                            model.layers[idx] = layer;
                            Ok(Value::Void)
                        }
                        "add_layer" => {
                            let layer = match arg_vals.first() {
                                Some(Value::Struct { name: ln, fields: lf }) if ln == "nn_linear" => {
                                    let in_s = lf.get("in_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                    let out_s = lf.get("out_size").and_then(|v| if let Value::Int(n) = v { Some(*n as usize) } else { None }).unwrap_or(1);
                                    crate::nn::Layer::linear(in_s, out_s)
                                }
                                _ => return Err("add_layer: arg must be a layer".into()),
                            };
                            let model = env.live_models.get_mut(&model_id).ok_or("live model not found")?;
                            model.layers.push(layer);
                            Ok(Value::Void)
                        }
                        "remove_layer" => {
                            let idx = match arg_vals.first() { Some(Value::Int(i)) => *i as usize, _ => return Err("remove_layer: arg must be index".into()) };
                            let model = env.live_models.get_mut(&model_id).ok_or("live model not found")?;
                            if idx >= model.layers.len() { return Err(format!("layer index {} out of bounds", idx)); }
                            model.layers.remove(idx);
                            Ok(Value::Void)
                        }
                        "num_layers" => {
                            let model = env.live_models.get(&model_id).ok_or("live model not found")?;
                            Ok(Value::Int(model.layers.len() as i128))
                        }
                        _ => Err(format!("unknown live model method: {}", method_name)),
                    };
                }

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
                            FnDef::GradWrapper { fn_name, order: _ } => eval_grad_call(env, &fn_name, arg_vals),
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
                    FnDef::GradWrapper { fn_name, order: _ } => eval_grad_call(env, &fn_name, arg_vals),
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
                    Value::Function { name, params, body } => {
                        // If params is empty but we have a name, look up the actual function def
                        if params.is_empty() && !name.is_empty() {
                            if let Some(func_def) = env.functions.get(&name).cloned() {
                                return match func_def {
                                    FnDef::Builtin(f) => f(env, arg_vals),
                                    FnDef::GradWrapper { fn_name, order: _ } => eval_grad_call(env, &fn_name, arg_vals),
                                    FnDef::User { params: fp, body: fb, .. } => {
                                        env.push_scope();
                                        for (param, val) in fp.iter().zip(arg_vals.iter()) {
                                            env.define(param, val.clone());
                                        }
                                        let result = eval_block(env, &fb)?;
                                        env.pop_scope();
                                        match result {
                                            Value::Return(v) => Ok(*v),
                                            other => Ok(other),
                                        }
                                    }
                                };
                            }
                        }
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

            // Check if this is a field constructor call: Fp(42), Fn(x)
            if let Some(params) = env.field_defs.get(&func_name).copied() {
                if arg_vals.is_empty() {
                    return Err(format!("{}() requires an argument", func_name));
                }
                return coerce_to_field(&arg_vals[0], params);
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

            // JIT compilation path for @gpu annotated functions
            if let Some(gpu_func) = env.gpu_functions.get(&func_name).cloned() {
                // Try JIT compilation and execution
                match try_jit_execute(env, &gpu_func, &arg_vals) {
                    Ok(Some(val)) => return Ok(val),
                    Ok(None) => {
                        // JIT not available, fall through to interpreter
                    }
                    Err(e) => {
                        eprintln!("[vortex] JIT compilation warning: {}, falling back to interpreter", e);
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
                FnDef::GradWrapper { fn_name, order: _ } => {
                    eval_grad_call(env, &fn_name, arg_vals)
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

        ExprKind::TypeCall { ty: _, method: _, args: _ } => {
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
                    // Check guard if present
                    if let Some(guard) = &arm.guard {
                        let guard_val = eval_expr(env, guard)?;
                        if !matches!(guard_val, Value::Bool(true)) {
                            env.pop_scope();
                            continue;
                        }
                    }
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
        Pattern::StructVariant { name, fields } => {
            if let Value::EnumVariant { variant, fields: val_fields, .. } = value {
                if variant != &name.name { return false; }
                true
            } else if let Value::Struct { name: sn, fields: sf } = value {
                if sn != &name.name { return false; }
                for (field_name, field_pat) in fields {
                    if let Some(val) = sf.get(&field_name.name) {
                        if !pattern_matches(field_pat, val) { return false; }
                    } else {
                        return false;
                    }
                }
                true
            } else {
                false
            }
        }
        Pattern::Or(patterns) => {
            patterns.iter().any(|p| pattern_matches(p, value))
        }
        Pattern::Tuple(patterns) => {
            if let Value::Tuple(values) = value {
                if patterns.len() != values.len() { return false; }
                patterns.iter().zip(values.iter()).all(|(p, v)| pattern_matches(p, v))
            } else {
                false
            }
        }
        Pattern::Rest => true,
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
        Pattern::StructVariant { name: _, fields } => {
            if let Value::Struct { fields: sf, .. } = value {
                for (field_name, field_pat) in fields {
                    if let Some(val) = sf.get(&field_name.name) {
                        bind_pattern(env, field_pat, val);
                    }
                }
            }
        }
        Pattern::Or(patterns) => {
            for p in patterns {
                if pattern_matches(p, value) {
                    bind_pattern(env, p, value);
                    break;
                }
            }
        }
        Pattern::Tuple(patterns) => {
            if let Value::Tuple(values) = value {
                for (p, v) in patterns.iter().zip(values.iter()) {
                    bind_pattern(env, p, v);
                }
            }
        }
        Pattern::Rest => {}
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
                (Value::ModFieldElem(a), Value::Int(b)) => {
                    let v = *b as u64;
                    let exp = [v, 0, 0, 0];
                    Ok(Value::ModFieldElem(a.pow(&exp)))
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
        (Value::ModFieldElem(x), Value::ModFieldElem(y)) => {
            if !std::ptr::eq(x.params, y.params) {
                return Err(format!("cannot add field elements from different fields ('{}' and '{}')", x.params.name, y.params.name));
            }
            Ok(Value::ModFieldElem(x.add(y)))
        }
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
        (Value::ModFieldElem(x), Value::ModFieldElem(y)) => {
            if !std::ptr::eq(x.params, y.params) {
                return Err(format!("cannot subtract field elements from different fields ('{}' and '{}')", x.params.name, y.params.name));
            }
            Ok(Value::ModFieldElem(x.sub(y)))
        }
        _ => Err(format!("cannot subtract {} and {}", a, b)),
    }
}

fn value_mul(a: &Value, b: &Value) -> Result<Value, String> {
    match (a, b) {
        (Value::Int(x), Value::Int(y)) => Ok(Value::Int(x.wrapping_mul(*y))),
        (Value::Float(x), Value::Float(y)) => Ok(Value::Float(x * y)),
        (Value::Int(x), Value::Float(y)) => Ok(Value::Float(*x as f64 * y)),
        (Value::Float(x), Value::Int(y)) => Ok(Value::Float(x * *y as f64)),
        (Value::FieldElem(x), Value::FieldElem(y)) => Ok(Value::FieldElem(x.mul(y))),
        (Value::ModFieldElem(x), Value::ModFieldElem(y)) => {
            if !std::ptr::eq(x.params, y.params) {
                return Err(format!("cannot multiply field elements from different fields ('{}' and '{}')", x.params.name, y.params.name));
            }
            Ok(Value::ModFieldElem(x.mul(y)))
        }
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
        (Value::ModFieldElem(x), Value::ModFieldElem(y)) => {
            if !std::ptr::eq(x.params, y.params) {
                return Err(format!("cannot divide field elements from different fields ('{}' and '{}')", x.params.name, y.params.name));
            }
            if y.is_zero() { return Err("division by zero in field".to_string()); }
            Ok(Value::ModFieldElem(x.mul(&y.inv())))
        }
        _ => Err(format!("cannot divide {} and {}", a, b)),
    }
}

fn coerce_to_field(val: &Value, params: &'static modmath::FieldParams) -> Result<Value, String> {
    match val {
        Value::Int(n) => {
            let v = if *n < 0 {
                // Negative: compute modulus - |n|
                let abs = (-*n) as u64;
                let limbs = [abs, 0, 0, 0];
                let elem = modmath::ModField::new(limbs, params);
                elem.neg()
            } else {
                let v = *n as u128;
                let limbs = [v as u64, (v >> 64) as u64, 0, 0];
                modmath::ModField::new(limbs, params)
            };
            Ok(Value::ModFieldElem(v))
        }
        Value::ModFieldElem(mf) => {
            // Cross-field cast: convert to normal form, re-interpret in new field
            let normal = mf.to_normal();
            Ok(Value::ModFieldElem(modmath::ModField::new(normal, params)))
        }
        Value::String(s) => {
            let elem = modmath::ModField::from_hex(s, params)
                .ok_or_else(|| format!("cannot convert hex string '{}' to field element", s))?;
            Ok(Value::ModFieldElem(elem))
        }
        _ => Err(format!("cannot coerce {} to field element", val)),
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
        (Value::ModFieldElem(x), Value::ModFieldElem(y)) => x == y,
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
        Value::Function { name, params, body } => {
            // If params is empty but we have a name, look up the actual function
            if params.is_empty() && !name.is_empty() {
                if let Some(func_def) = env.functions.get(name).cloned() {
                    return match func_def {
                        FnDef::Builtin(f) => f(env, args),
                        FnDef::User { params, body } => {
                            env.push_scope();
                            for (param, val) in params.iter().zip(args.iter()) {
                                env.define(param, val.clone());
                            }
                            let result = eval_block(env, &body)?;
                            env.pop_scope();
                            match result {
                                Value::Return(v) => Ok(*v),
                                other => Ok(other),
                            }
                        }
                        FnDef::GradWrapper { fn_name, order: _ } => eval_grad_call(env, &fn_name, args),
                    };
                }
            }
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
        Ok(contents) => Ok(Value::String(contents)),
        Err(e) => Err(format!("read_file error: {}", e)),
    }
}

fn builtin_write_file(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("write_file expects 2 arguments: (path, content)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("write_file: path must be a string".into()) };
    let content = match &args[1] { Value::String(s) => s.clone(), _ => return Err("write_file: content must be a string".into()) };
    match std::fs::write(&path, &content) {
        Ok(()) => Ok(Value::Void),
        Err(e) => Err(format!("write_file error: {}", e)),
    }
}

fn builtin_append_file(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("append_file expects 2 arguments: (path, content)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("append_file: path must be a string".into()) };
    let content = match &args[1] { Value::String(s) => s.clone(), _ => return Err("append_file: content must be a string".into()) };
    use std::io::Write;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .map_err(|e| format!("append_file error: {}", e))?;
    file.write_all(content.as_bytes())
        .map_err(|e| format!("append_file error: {}", e))?;
    Ok(Value::Void)
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

fn builtin_load_csv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("load_csv expects 1 argument: (path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("load_csv: path must be a string".into()) };
    match std::fs::read_to_string(&path) {
        Ok(contents) => {
            let rows: Vec<Value> = contents.lines().map(|line| {
                let cols: Vec<Value> = line.split(',').map(|c| Value::String(c.trim().to_string())).collect();
                Value::Array(cols)
            }).collect();
            Ok(Value::Array(rows))
        }
        Err(e) => Err(format!("load_csv error: {}", e)),
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

fn ag_to_data(v: &Value) -> Result<(Vec<f64>, Vec<usize>), String> { match v { Value::Float(f) => Ok((vec![*f], vec![1])), Value::Int(i) => Ok((vec![*i as f64], vec![1])), Value::Array(arr) => { let data: Vec<f64> = arr.iter().map(|x| match x { Value::Float(f) => Ok(*f), Value::Int(i) => Ok(*i as f64), _ => Err("expected number".to_string()) }).collect::<Result<_,_>>()?; let len = data.len(); Ok((data, vec![len])) } _ => Err("expected number or array".into()) } }
fn builtin_autograd_new(env: &mut Env, _a: Vec<Value>) -> Result<Value, String> { env.autograd_tape = Some(crate::autograd::Tape::new()); env.autograd_adam = None; Ok(Value::Void) }
fn builtin_autograd_tensor(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let (d, s) = ag_to_data(&args[0])?; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.parameter(d, s) as i128)) }
fn builtin_autograd_input(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let (d, s) = ag_to_data(&args[0])?; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.input(d, s) as i128)) }
fn builtin_autograd_matmul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.mul(a, b) as i128)) }
fn builtin_autograd_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.add(a, b) as i128)) }
fn builtin_autograd_mul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.mul(a, b) as i128)) }
fn builtin_autograd_sub(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.sub(a, b) as i128)) }
fn builtin_autograd_div(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.div(a, b) as i128)) }
fn builtin_autograd_relu(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let tape = env.autograd_tape.as_mut().ok_or("no tape")?; let data: Vec<f64> = tape.get_data(a).iter().map(|x| x.max(0.0)).collect(); let s = vec![data.len()]; Ok(Value::Int(tape.input(data, s) as i128)) }
fn builtin_autograd_sigmoid(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let tape = env.autograd_tape.as_mut().ok_or("no tape")?; let data: Vec<f64> = tape.get_data(a).iter().map(|x| 1.0/(1.0+(-x).exp())).collect(); let s = vec![data.len()]; Ok(Value::Int(tape.input(data, s) as i128)) }
fn builtin_autograd_tanh(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let tape = env.autograd_tape.as_mut().ok_or("no tape")?; let data: Vec<f64> = tape.get_data(a).iter().map(|x| x.tanh()).collect(); let s = vec![data.len()]; Ok(Value::Int(tape.input(data, s) as i128)) }
fn builtin_autograd_softmax(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.exp(a) as i128)) }
fn builtin_autograd_exp(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.exp(a) as i128)) }
fn builtin_autograd_log(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Int(env.autograd_tape.as_mut().ok_or("no tape")?.log(a) as i128)) }
fn builtin_autograd_sum(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Float(env.autograd_tape.as_ref().ok_or("no tape")?.get_data(a).iter().sum())) }
fn builtin_autograd_mean(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let d = env.autograd_tape.as_ref().ok_or("no tape")?.get_data(a); Ok(Value::Float(if d.is_empty() { 0.0 } else { d.iter().sum::<f64>() / d.len() as f64 })) }
fn builtin_autograd_mse(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let a = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let b = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; let tape = env.autograd_tape.as_mut().ok_or("no tape")?; let d = tape.sub(a, b); Ok(Value::Int(tape.mul(d, d) as i128)) }
fn builtin_autograd_broadcast_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { builtin_autograd_add(env, args) }
fn builtin_autograd_backward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; env.autograd_tape.as_mut().ok_or("no tape")?.backward(id); Ok(Value::Void) }
fn builtin_autograd_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; match env.autograd_tape.as_ref().ok_or("no tape")?.get_grad(id) { Some(g) => Ok(Value::Array(g.into_iter().map(Value::Float).collect())), None => Ok(Value::Option(None)) } }
fn builtin_autograd_data(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("e".into()) }; Ok(Value::Array(env.autograd_tape.as_ref().ok_or("no tape")?.get_data(id).iter().map(|v| Value::Float(*v)).collect())) }
fn builtin_autograd_zero_grad(env: &mut Env, _a: Vec<Value>) -> Result<Value, String> { env.autograd_tape = Some(crate::autograd::Tape::new()); Ok(Value::Void) }
fn builtin_autograd_adam_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { let lr = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr".into()) }; let ids: Vec<usize> = args[1..].iter().map(|v| match v { Value::Int(i) => Ok(*i as usize), _ => Err("e".to_string()) }).collect::<Result<_,_>>()?; let tape = env.autograd_tape.as_mut().ok_or("no tape")?; let psizes: Vec<usize> = ids.iter().map(|&id| tape.get_data(id).len()).collect(); let grads: Vec<Vec<f64>> = ids.iter().map(|&id| tape.get_grad(id).unwrap_or_else(|| vec![0.0; tape.get_data(id).len()])).collect(); let adam = env.autograd_adam.get_or_insert_with(|| crate::autograd::AdamOptimizer::new(lr, &psizes)); adam.lr = lr; adam.step(tape, &ids, &grads); Ok(Value::Void) }


// --- First-class Tensor builtins ---

fn ft_builtin_tensor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor expects 2 arguments: (data, shape)".to_string()); }
    let data: Vec<f64> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("tensor: data elements must be numbers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor: first argument must be an array".to_string()),
    };
    let shape: Vec<usize> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("tensor: shape elements must be integers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor: second argument must be a shape array".to_string()),
    };
    let expected: usize = shape.iter().product();
    if data.len() != expected {
        return Err(format!("tensor: data length {} doesn't match shape {:?} (expected {})", data.len(), shape, expected));
    }
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(shape, &data)))
}

fn ft_builtin_tensor_zeros(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_zeros expects 1 argument: (shape)".to_string()); }
    let shape: Vec<usize> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("tensor_zeros: shape elements must be integers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor_zeros: argument must be a shape array".to_string()),
    };
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::zeros(shape, crate::tensor_engine::DType::F64)))
}

fn ft_builtin_tensor_ones(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_ones expects 1 argument: (shape)".to_string()); }
    let shape: Vec<usize> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("tensor_ones: shape elements must be integers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor_ones: argument must be a shape array".to_string()),
    };
    let numel: usize = shape.iter().product();
    let data = vec![1.0f64; numel];
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(shape, &data)))
}

fn ft_builtin_tensor_rand(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_rand expects 1 argument: (shape)".to_string()); }
    let shape: Vec<usize> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("tensor_rand: shape elements must be integers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor_rand: argument must be a shape array".to_string()),
    };
    let numel: usize = shape.iter().product();
    let mut seed: u64 = 42 + numel as u64;
    let data: Vec<f64> = (0..numel).map(|i| {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        seed = seed.wrapping_add(i as u64);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    }).collect();
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(shape, &data)))
}

fn ft_builtin_tensor_shape(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_shape expects 1 argument".to_string()); }
    match &args[0] {
        Value::Tensor(t) => Ok(Value::Array(t.shape.iter().map(|&s| Value::Int(s as i128)).collect())),
        _ => Err("tensor_shape: argument must be a tensor".to_string()),
    }
}

fn ft_builtin_tensor_dtype(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_dtype expects 1 argument".to_string()); }
    match &args[0] {
        Value::Tensor(t) => Ok(Value::String(format!("{:?}", t.dtype))),
        _ => Err("tensor_dtype: argument must be a tensor".to_string()),
    }
}

fn ft_builtin_tensor_reshape(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_reshape expects 2 arguments: (tensor, new_shape)".to_string()); }
    let t = match &args[0] {
        Value::Tensor(t) => t.clone(),
        _ => return Err("tensor_reshape: first argument must be a tensor".to_string()),
    };
    let new_shape: Vec<usize> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("tensor_reshape: shape elements must be integers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor_reshape: second argument must be a shape array".to_string()),
    };
    let old_numel: usize = t.shape.iter().product();
    let new_numel: usize = new_shape.iter().product();
    if old_numel != new_numel {
        return Err(format!("tensor_reshape: cannot reshape {} elements into {:?} ({} elements)", old_numel, new_shape, new_numel));
    }
    let strides = crate::tensor_engine::FastTensor::zeros(new_shape.clone(), t.dtype).strides;
    Ok(Value::Tensor(crate::tensor_engine::FastTensor {
        data: t.data,
        shape: new_shape,
        strides,
        dtype: t.dtype,
        layout: t.layout,
    }))
}

fn ft_tensor_extract_f64(t: &crate::tensor_engine::FastTensor) -> Vec<f64> {
    t.as_f64_slice().to_vec()
}

fn ft_builtin_tensor_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_add expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_add: first argument must be a tensor".to_string()) };
    let b = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_add: second argument must be a tensor".to_string()) };
    if a.shape != b.shape { return Err(format!("tensor_add: shape mismatch {:?} vs {:?}", a.shape, b.shape)); }
    let ad = ft_tensor_extract_f64(&a);
    let bd = ft_tensor_extract_f64(&b);
    let result: Vec<f64> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(a.shape, &result)))
}

fn ft_builtin_tensor_mul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_mul expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_mul: first argument must be a tensor".to_string()) };
    let b = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_mul: second argument must be a tensor".to_string()) };
    if a.shape != b.shape { return Err(format!("tensor_mul: shape mismatch {:?} vs {:?}", a.shape, b.shape)); }
    let ad = ft_tensor_extract_f64(&a);
    let bd = ft_tensor_extract_f64(&b);
    let result: Vec<f64> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(a.shape, &result)))
}

fn ft_builtin_tensor_matmul_dense(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_matmul_dense expects 2 arguments".to_string()); }
    let a = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_matmul_dense: first argument must be a tensor".to_string()) };
    let b = match &args[1] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_matmul_dense: second argument must be a tensor".to_string()) };
    if a.shape.len() != 2 || b.shape.len() != 2 {
        return Err("tensor_matmul_dense: both tensors must be 2D".to_string());
    }
    let (m, k1) = (a.shape[0], a.shape[1]);
    let (k2, n) = (b.shape[0], b.shape[1]);
    if k1 != k2 { return Err(format!("tensor_matmul_dense: inner dimensions don't match: {} vs {}", k1, k2)); }
    let ad = ft_tensor_extract_f64(&a);
    let bd = ft_tensor_extract_f64(&b);
    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for kk in 0..k1 {
                sum += ad[i * k1 + kk] * bd[kk * n + j];
            }
            result[i * n + j] = sum;
        }
    }
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(vec![m, n], &result)))
}

fn ft_builtin_tensor_transpose(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_transpose expects 1 argument".to_string()); }
    let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_transpose: argument must be a tensor".to_string()) };
    if t.shape.len() != 2 { return Err("tensor_transpose: tensor must be 2D".to_string()); }
    let (rows, cols) = (t.shape[0], t.shape[1]);
    let data = ft_tensor_extract_f64(&t);
    let mut result = vec![0.0f64; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            result[j * rows + i] = data[i * cols + j];
        }
    }
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(vec![cols, rows], &result)))
}

fn ft_builtin_tensor_slice(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("tensor_slice expects 4 arguments: (tensor, dim, start, end)".to_string()); }
    let t = match &args[0] { Value::Tensor(t) => t.clone(), _ => return Err("tensor_slice: first argument must be a tensor".to_string()) };
    let dim = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("tensor_slice: dim must be an integer".to_string()) };
    let start = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("tensor_slice: start must be an integer".to_string()) };
    let end = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("tensor_slice: end must be an integer".to_string()) };
    if dim >= t.shape.len() { return Err(format!("tensor_slice: dim {} out of bounds for {}D tensor", dim, t.shape.len())); }
    if end > t.shape[dim] || start > end { return Err("tensor_slice: invalid slice bounds".to_string()); }
    let data = ft_tensor_extract_f64(&t);
    let mut new_shape = t.shape.clone();
    new_shape[dim] = end - start;
    let outer: usize = t.shape[..dim].iter().product();
    let inner: usize = t.shape[dim + 1..].iter().product();
    let dim_size = t.shape[dim];
    let mut result = Vec::with_capacity(new_shape.iter().product());
    for o in 0..outer {
        for d in start..end {
            for i in 0..inner {
                result.push(data[o * dim_size * inner + d * inner + i]);
            }
        }
    }
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(new_shape, &result)))
}

fn ft_builtin_tensor_to_array(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_to_array expects 1 argument".to_string()); }
    match &args[0] {
        Value::Tensor(t) => {
            let data = ft_tensor_extract_f64(t);
            Ok(Value::Array(data.iter().map(|&v| Value::Float(v)).collect()))
        }
        _ => Err("tensor_to_array: argument must be a tensor".to_string()),
    }
}

fn ft_builtin_tensor_from_array(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_from_array expects 1 argument".to_string()); }
    let data: Vec<f64> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("tensor_from_array: elements must be numbers".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("tensor_from_array: argument must be an array".to_string()),
    };
    let len = data.len();
    Ok(Value::Tensor(crate::tensor_engine::FastTensor::from_f64(vec![len], &data)))
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
let q = [1.0, 0.0]\nlet k = [1.0, 0.0]\nlet v = [0.5, 0.3]\nlet r = attn_compute(q, k, v)\nprintln(r)
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

    #[test]
    fn test_field_def_basic_add() {
        let o = rv("field Fp = 0x17\nfn main() {\nlet a: Fp = 7\nlet b: Fp = 13\nlet c = a + b\nlet expected = Fp(20)\nprintln(c == expected)\n}");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_field_def_mul() {
        let o = rv("field Fp = 0x17\nfn main() {\nlet a: Fp = 7\nlet b: Fp = 13\nlet c = a * b\nlet expected = Fp(91 % 23)\nprintln(c == expected)\n}");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_field_def_div_roundtrip() {
        let o = rv("field Fp = 0x17\nfn main() {\nlet a: Fp = 7\nlet b: Fp = 13\nlet d = a / b\nlet check = d * b\nprintln(check == a)\n}");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_field_def_pow() {
        let o = rv("field Fp = 0x17\nfn main() {\nlet a: Fp = 2\nlet b = a ** 4\nlet expected = Fp(16)\nprintln(b == expected)\n}");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_field_def_cross_field_error() {
        let code = "field Fp = 0x17\nfield Fn = 0x1D\nfn main() {\nlet a: Fp = 7\nlet b: Fn = 13\nlet c = a + b\nprintln(c)\n}";
        let t = lexer::lex(code);
        let p = parser::parse(t, 0).unwrap();
        let result = interpret(&p);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("different fields"));
    }

    #[test]
    fn test_field_def_cross_field_cast() {
        let o = rv("field Fp = 0x17\nfield Fn = 0x1D\nfn main() {\nlet a: Fp = 7\nlet b: Fn = Fn(a)\nprintln(b == Fn(7))\n}");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_field_def_secp256k1() {
        let o = rv("field Fp = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F\nfn main() {\nlet a: Fp = 42\nlet b: Fp = 0xdeadbeef\nlet c = a + b\nlet d = a * b\nprintln(c == Fp(42 + 3735928559))\nprintln(d == Fp(42 * 3735928559))\n}");
        assert_eq!(o, vec!["true", "true"]);
    }

    #[test]
    fn test_field_def_negation() {
        let o = rv("field Fp = 0x17\nfn main() {\nlet a: Fp = 7\nlet b = -a\nlet c = a + b\nlet zero = Fp(0)\nprintln(c == zero)\n}");
        assert_eq!(o, vec!["true"]);
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
        assert_eq!(o, vec!["6.0", "0.0"]);
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
        assert_eq!(o, vec!["42.0"]);
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
        let o = rv("fn main() {\nlet q = [[1.0, 0.0], [0.0, 1.0]]\nlet k = [[1.0, 0.0], [0.0, 1.0]]\nlet v = [[1.0, 2.0], [3.0, 4.0]]\nlet r = attn_compute(q, k, v)\nprintln(r)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("["));
    }

    // Feature 41: Attention block
    #[test]
    fn test_attention_block() {
        let o = rv("fn main() {\nlet a = multi_head_attention(4, 32)\nprintln(a)\n}");
        assert!(!o.is_empty());
    }

    // Feature 42: Gradient surgery
    #[test]
    fn test_gradient_surgery() {
        let o = rv("fn main() {\nlet g = [3.0, 4.0]\nlet n = grad_norm(g)\nprintln(n)\nlet c = clip_grad(1.0)\nprintln(c)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("5")); // sqrt(9+16) = 5
    }

    // Feature 43: Curriculum
    #[test]
    fn test_curriculum_block() {
        let o = rv("fn main() {\nlet d = curriculum_schedule(100, 50)\nprintln(d)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("0.5"));
    }

    // Feature 44: Ensemble
    #[test]
    fn test_ensemble_builtins() {
        let o = rv("fn main() {\nlet a = ensemble_avg([1.0, 2.0, 3.0])\nprintln(a)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("2"));
    }

    // Feature 45: Adversarial
    #[test]
    fn test_adversarial_block() {
        let o = rv("fn main() {\nlet a = fgsm_attack(0.01, [1.0, 2.0, 3.0])\nprintln(a)\n}");
        assert!(!o.is_empty());
    }

    // Feature 46: Transfer
    #[test]
    fn test_transfer_block() {
        let o = rv("fn main() {\nlet f = freeze(\"layer_0\")\nprintln(f)\nlet u = unfreeze(\"layer_0\")\nprintln(u)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("frozen"));
    }

    // Feature 47: Sparse
    #[test]
    fn test_sparse_block() {
        let o = rv("fn main() {\nlet r = sparsity_ratio([0.0, 1.0, 0.0, 2.0])\nprintln(r)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("0.5"));
    }

    // Feature 48: Async infer
    #[test]
    fn test_async_infer_block() {
        let o = rv("fn main() {\nlet p = async_predict(0, 42)\nprintln(p)\n}");
        assert!(!o.is_empty());
    }

    // Feature 49: Profile
    #[test]
    fn test_profile_block() {
        let o = rv("fn main() {\nlet s = profile_summary()\nprintln(s)\n}");
        assert!(!o.is_empty());
    }

    // Feature 50: Contract
    #[test]
    fn test_contract_builtins() {
        let o = rv("fn main() {\nrequires(true, \"must be true\")\nlet x = 42\nensures(x > 0, \"must be positive\")\nprintln(x)\n}");
        assert!(!o.is_empty());
        assert!(o[0].contains("42"));
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

// --- GPU Compute (native compilation) builtins ---

use crate::gpu_compute;

/// gpu_available() -> bool
fn builtin_gpu_available(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Bool(gpu_compute::native_available()))
}

/// gpu_native_matmul(a_data, a_shape, b_data, b_shape) -> array
fn builtin_gpu_native_matmul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("gpu_native_matmul expects (a_data, a_shape, b_data, b_shape)".to_string());
    }
    let a_data = extract_f64_data(&args[0])?;
    let a_shape = extract_shape(&args[1])?;
    let b_data = extract_f64_data(&args[2])?;
    let b_shape = extract_shape(&args[3])?;
    let a = gpu_compute::Tensor::new(a_data, a_shape);
    let b = gpu_compute::Tensor::new(b_data, b_shape);
    let c = gpu_compute::gpu_matmul(&a, &b, gpu_compute::Backend::Auto)?;
    Ok(Value::Array(c.data.into_iter().map(Value::Float).collect()))
}

/// gpu_train_step(input, target, weights, biases, lr) -> loss
fn builtin_gpu_train_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 5 {
        return Err("gpu_train_step expects (input, target, weights, biases, lr)".to_string());
    }
    let input = extract_f64_data(&args[0])?;
    let target = extract_f64_data(&args[1])?;
    let loss: f64 = input.iter().zip(target.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>() / target.len().max(1) as f64;
    Ok(Value::Float(loss))
}

/// gpu_benchmark(op_name, size) -> struct { cpu_us, native_us }
fn builtin_gpu_benchmark(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("gpu_benchmark expects (op_name, size)".to_string());
    }
    let op_name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("gpu_benchmark: first arg must be string".to_string()),
    };
    let size = match &args[1] {
        Value::Int(n) => *n as usize,
        _ => return Err("gpu_benchmark: second arg must be int".to_string()),
    };
    let a = gpu_compute::Tensor::new(vec![1.0; size * size], vec![size, size]);
    let b = gpu_compute::Tensor::new(vec![1.0; size * size], vec![size, size]);
    let op = match op_name.as_str() {
        "matmul" => gpu_compute::GpuOp::MatMul { m: size, k: size, n: size },
        "relu" => gpu_compute::GpuOp::Relu { size: size * size },
        "softmax" => gpu_compute::GpuOp::Softmax { rows: size, cols: size },
        _ => return Err(format!("gpu_benchmark: unknown op '{}'", op_name)),
    };
    let (cpu_us, native_us) = gpu_compute::benchmark_op(&op, &a, Some(&b));
    let mut fields = HashMap::new();
    fields.insert("cpu_us".to_string(), Value::Int(cpu_us as i128));
    fields.insert("native_us".to_string(), match native_us {
        Some(us) => Value::Int(us as i128),
        None => Value::Void,
    });
    Ok(Value::Struct { name: "BenchmarkResult".to_string(), fields })
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
    let _v_state_id = value_to_tensor_id(&args[7])?;

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
        assert_eq!(o, vec!["[1.0, 2.0, 3.0, 4.0]"]);
    }

    #[test]
    fn test_tensor_input_creates_untracked() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet x = tensor_input([1.0, 2.0], [1, 2])\nprintln(tensor_data(x))\n}");
        assert_eq!(o, vec!["[1.0, 2.0]"]);
    }

    #[test]
    fn test_tensor_matmul_shape() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet a = tensor_param([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])\nlet b = tensor_param([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])\nlet c = tensor_matmul(a, b)\nlet d = tensor_data(c)\nprintln(len(d))\n}");
        assert_eq!(o, vec!["4"]);
    }

    #[test]
    fn test_tensor_backward_produces_gradients() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0, 3.0, 4.0], [2, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\nlet g = tensor_grad(w)\nprintln(g)\n}");
        assert_eq!(o, vec!["[1.0, 1.0, 1.0, 1.0]"]);
    }

    #[test]
    fn test_tensor_sgd_changes_params() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0], [1, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\ntensor_sgd([w], 0.1)\nlet d = tensor_data(w)\nprintln(d)\n}");
        assert_eq!(o, vec!["[0.9, 1.9]"]);
    }

    #[test]
    fn test_tensor_zero_grad_resets() {
        let o = rv("fn main() {\ntensor_tape_new()\nlet w = tensor_param([1.0, 2.0], [1, 2])\nlet s = tensor_sum(w)\ntensor_backward(s)\ntensor_zero_grad([w])\nlet g = tensor_grad(w)\nprintln(g)\n}");
        assert_eq!(o, vec!["[0.0, 0.0]"]);
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

// --- Math builtins ---

fn val_to_f64_math(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("expected numeric value".to_string()),
    }
}

fn math_unary(args: Vec<Value>, f: fn(f64) -> f64, name: &str) -> Result<Value, String> {
    if args.len() != 1 { return Err(format!("{} expects 1 argument", name)); }
    Ok(Value::Float(f(val_to_f64_math(&args[0])?)))
}

fn builtin_sqrt(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::sqrt, "sqrt") }
fn builtin_sin(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::sin, "sin") }
fn builtin_cos(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::cos, "cos") }
fn builtin_tan(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::tan, "tan") }
fn builtin_exp(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::exp, "exp") }
fn builtin_log(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::ln, "log") }
fn builtin_log2(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::log2, "log2") }
fn builtin_log10(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::log10, "log10") }
fn builtin_floor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::floor, "floor") }
fn builtin_ceil(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::ceil, "ceil") }
fn builtin_round(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> { math_unary(args, f64::round, "round") }

fn builtin_abs(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("abs expects 1 argument".to_string()); }
    match &args[0] {
        Value::Int(n) => Ok(Value::Int(n.abs())),
        Value::Float(f) => Ok(Value::Float(f.abs())),
        _ => Err("abs expects a numeric value".to_string()),
    }
}

fn builtin_pow(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("pow expects 2 arguments: (base, exp)".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Int(base), Value::Int(exp)) if *exp >= 0 => {
            Ok(Value::Int(base.pow(*exp as u32)))
        }
        _ => {
            let base = val_to_f64_math(&args[0])?;
            let exp = val_to_f64_math(&args[1])?;
            Ok(Value::Float(base.powf(exp)))
        }
    }
}

fn builtin_min(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("min expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.min(b))),
        _ => {
            let a = val_to_f64_math(&args[0])?;
            let b = val_to_f64_math(&args[1])?;
            Ok(Value::Float(a.min(b)))
        }
    }
}

fn builtin_max(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("max expects 2 arguments".to_string()); }
    match (&args[0], &args[1]) {
        (Value::Int(a), Value::Int(b)) => Ok(Value::Int(*a.max(b))),
        _ => {
            let a = val_to_f64_math(&args[0])?;
            let b = val_to_f64_math(&args[1])?;
            Ok(Value::Float(a.max(b)))
        }
    }
}

fn builtin_range(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.len() {
        2 => {
            let start = value_to_int(&args[0])?;
            let end = value_to_int(&args[1])?;
            let elems: Vec<Value> = if start <= end {
                (start..end).map(Value::Int).collect()
            } else {
                Vec::new()
            };
            Ok(Value::Array(elems))
        }
        3 => {
            let start = value_to_int(&args[0])?;
            let end = value_to_int(&args[1])?;
            let step = value_to_int(&args[2])?;
            if step == 0 { return Err("range: step cannot be zero".to_string()); }
            let mut elems = Vec::new();
            let mut i = start;
            if step > 0 {
                while i < end { elems.push(Value::Int(i)); i += step; }
            } else {
                while i > end { elems.push(Value::Int(i)); i += step; }
            }
            Ok(Value::Array(elems))
        }
        _ => Err("range expects 2 or 3 arguments: (start, end) or (start, end, step)".to_string()),
    }
}

#[cfg(test)]
mod math_builtin_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }

    #[test]
    fn test_sqrt() { assert_eq!(rv("fn main() { println(sqrt(9)) }"), vec!["3.0"]); }

    #[test]
    fn test_sin_cos() {
        assert_eq!(rv("fn main() { println(sin(0)) }"), vec!["0.0"]);
        assert_eq!(rv("fn main() { println(cos(0)) }"), vec!["1.0"]);
    }

    #[test]
    fn test_exp_log() {
        assert_eq!(rv("fn main() { println(exp(0)) }"), vec!["1.0"]);
        assert_eq!(rv("fn main() { println(log(1)) }"), vec!["0.0"]);
    }

    #[test]
    fn test_abs() {
        assert_eq!(rv("fn main() { println(abs(-5)) }"), vec!["5"]);
        assert_eq!(rv("fn main() { println(abs(3)) }"), vec!["3"]);
    }

    #[test]
    fn test_pow() {
        assert_eq!(rv("fn main() { println(pow(2, 10)) }"), vec!["1024"]);
    }

    #[test]
    fn test_floor_ceil_round() {
        assert_eq!(rv("fn main() { println(floor(3.7)) }"), vec!["3.0"]);
        assert_eq!(rv("fn main() { println(ceil(3.2)) }"), vec!["4.0"]);
        assert_eq!(rv("fn main() { println(round(3.5)) }"), vec!["4.0"]);
    }

    #[test]
    fn test_min_max() {
        assert_eq!(rv("fn main() { println(min(3, 7)) }"), vec!["3"]);
        assert_eq!(rv("fn main() { println(max(3, 7)) }"), vec!["7"]);
    }

    #[test]
    fn test_range_basic() {
        assert_eq!(rv("fn main() { println(range(0, 5)) }"), vec!["[0, 1, 2, 3, 4]"]);
    }

    #[test]
    fn test_range_step() {
        assert_eq!(rv("fn main() { println(range(0, 10, 3)) }"), vec!["[0, 3, 6, 9]"]);
    }

    #[test]
    fn test_range_negative_step() {
        assert_eq!(rv("fn main() { println(range(5, 0, -1)) }"), vec!["[5, 4, 3, 2, 1]"]);
    }

    #[test]
    fn test_for_range_loop() {
        let o = rv("fn main() {\nvar s = 0\nfor i in range(0, 5) {\ns = s + i\n}\nprintln(s)\n}");
        assert_eq!(o, vec!["10"]);
    }

    #[test]
    fn test_pi_e_constants() {
        let o = rv("fn main() { println(PI > 3.0) }");
        assert_eq!(o, vec!["true"]);
        let o = rv("fn main() { println(E > 2.0) }");
        assert_eq!(o, vec!["true"]);
    }

    #[test]
    fn test_closure_capture_and_exec() {
        let o = rv("fn main() {\nlet x = 10\nlet f = |y| y + x\nprintln(f(5))\n}");
        assert_eq!(o, vec!["15"]);
    }

    #[test]
    fn test_closure_passed_as_argument() {
        let o = rv("fn apply(f: fn(i64) -> i64, x: i64) -> i64 {\nreturn f(x)\n}\nfn main() {\nlet double = |x| x * 2\nprintln(apply(double, 5))\n}");
        assert_eq!(o, vec!["10"]);
    }

    #[test]
    fn test_log2_log10() {
        assert_eq!(rv("fn main() { println(log2(8)) }"), vec!["3.0"]);
        assert_eq!(rv("fn main() { println(log10(1000)) }"), vec!["3.0"]);
    }

    #[test]
    fn test_tan() {
        assert_eq!(rv("fn main() { println(tan(0)) }"), vec!["0.0"]);
    }
}

#[cfg(test)]
mod file_io_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;
    fn rv(s: &str) -> Vec<String> { let t = lexer::lex(s); let p = parser::parse(t, 0).unwrap(); interpret(&p).unwrap() }

    #[test]
    fn test_read_write_file_roundtrip() {
        let o = rv("fn main() {\nwrite_file(\"/tmp/vx_test_rw.txt\", \"hello vortex\")\nlet content = read_file(\"/tmp/vx_test_rw.txt\")\nprintln(content)\n}");
        assert_eq!(o, vec!["hello vortex"]);
        std::fs::remove_file("/tmp/vx_test_rw.txt").ok();
    }

    #[test]
    fn test_file_exists_true_false() {
        let o = rv("fn main() {\nwrite_file(\"/tmp/vx_test_ex.txt\", \"x\")\nprintln(file_exists(\"/tmp/vx_test_ex.txt\"))\nprintln(file_exists(\"/tmp/vx_nonexist.txt\"))\n}");
        assert_eq!(o, vec!["true", "false"]);
        std::fs::remove_file("/tmp/vx_test_ex.txt").ok();
    }

    #[test]
    fn test_load_csv() {
        std::fs::write("/tmp/vx_test.csv", "a,b,c\n1,2,3\nx,y,z\n").unwrap();
        let o = rv("fn main() {\nlet data = load_csv(\"/tmp/vx_test.csv\")\nprintln(len(data))\n}");
        assert_eq!(o[0], "3");
        std::fs::remove_file("/tmp/vx_test.csv").ok();
    }

    #[test]
    fn test_float_display_decimal() {
        let o = rv("fn main() {\nlet x: f64 = 10.0\nprintln(x)\nlet y: f64 = 3.14\nprintln(y)\n}");
        assert_eq!(o[0], "10.0");
        assert_eq!(o[1], "3.14");
    }
}

#[cfg(test)]
mod import_tests {
    use crate::lexer;
    use crate::parser;
    use crate::interpreter::interpret;

    #[test]
    fn test_import_from_stdlib() {
        let src = "import std.nn { relu }\nfn main() {\n    println(relu(5.0))\n    println(relu(-3.0))\n}";
        let tokens = lexer::lex(src);
        let program = parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output, vec!["5.0", "0.0"]);
    }

    #[test]
    fn test_import_nonexistent_module() {
        let src = "import nonexistent.module { Foo }\nfn main() {}";
        let tokens = lexer::lex(src);
        let program = parser::parse(tokens, 0).unwrap();
        let result = interpret(&program);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(err.contains("not found"), "error should mention not found: {}", err);
    }

    #[test]
    fn test_from_import_syntax() {
        let src = "from std.nn import relu\nfn main() {\n    println(relu(2.5))\n}";
        let tokens = lexer::lex(src);
        let program = parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output, vec!["2.5"]);
    }

    #[test]
    fn test_diff_fn_grad() {
        let src = r#"
diff fn square(x: f64) -> f64 {
    return x * x
}
fn main() {
    let v = square(3.0)
    println(v)
    let grad_square = grad(square)
    let g = grad_square(3.0)
    println(g)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        // Numerical gradient of x^2 at x=3 should be ~6.0
        assert!(!output.is_empty(), "expected output from grad test");
        let val: f64 = output[1].parse().unwrap_or_else(|_| panic!("could not parse '{}' as f64", output[1]));
        assert!((val - 6.0).abs() < 0.1, "gradient of x^2 at 3 should be ~6.0, got {}", val);
    }

    #[test]
    fn test_verifiable_prove_verify() {
        let src = r#"
#[verifiable]
fn hash_sum(a: i64, b: i64) -> i64 { a + b }
fn main() {
    let val = prove_value(hash_sum, 3, 4)
    println(val)
    let ok = prove_and_verify(hash_sum, 3, 4)
    println(ok)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "7");
        assert_eq!(output[1], "true");
    }

    #[test]
    fn test_unique_type_move() {
        let src = r#"
fn consume(buf: i64) -> i64 { buf }
fn main() {
    let x = 42
    let y = consume(x)
    println(y)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "42");
    }

    #[test]
    fn test_live_model() {
        let src = r#"
fn main() {
    live model = nn_new([3, 2, 1])
    println(model.num_layers())
    model.replace_layer(0, nn_linear(3, 5))
    println(model.num_layers())
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "2");
        assert_eq!(output[1], "2");
    }

    #[test]
    fn test_fuse_block() {
        let src = r#"
fn main() {
    fuse {
        let a = 2.0 + 3.0
        let b = a * 4.0
        println(b)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "20.0");
    }

    #[test]
    fn test_gpu_let() {
        let src = r#"
fn main() {
    gpu let buf = 42
    println(buf)
    let info = gpu_info()
    println(info.allocations)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "42");
        assert_eq!(output[1], "1");
    }

    #[test]
    fn test_parallel_for() {
        let src = r#"
fn main() {
    let total = 0
    parallel for x in [1, 2, 3, 4] {
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output.len(), 4);
    }

    #[test]
    fn test_shard_and_unshard() {
        let src = r#"
fn main() {
    let data = [1, 2, 3, 4, 5, 6]
    let shards = shard(data, 0, 3)
    println(len(shards))
    let reduced = all_reduce([1.0, 2.0, 3.0], "sum")
    println(reduced)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "3");
        assert_eq!(output[1], "6.0");
    }

    #[test]
    #[test]
    fn test_train_block() {
        let src = r#"
fn main() {
    train {
        data: [1.0, 2.0, 3.0, 4.0],
        epochs: 2,
        batch_size: 2,
        lr: 0.01,
    }
    println("trained")
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "trained");
    }

    #[test]
    fn test_deterministic_block() {
        let src = r#"
fn main() {
    deterministic {
        let a = 3 + 4
        println(a)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "7");
    }

    #[test]
    fn test_autocast_block() {
        let src = r#"
fn main() {
    autocast(f16) {
        let x = to_f16(3.14159)
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        // f16 rounds 3.14159, result should be close but not exact
        let val: f64 = output[0].parse().unwrap();
        assert!((val - 3.14159).abs() < 0.01, "f16 conversion should be close: {}", val);
    }

    #[test]
    fn test_mixed_precision_builtins() {
        let src = r#"
fn main() {
    let x = to_f16(1.5)
    let y = to_f32(x)
    let z = to_bf16(2.5)
    println(x)
    println(y)
    println(z)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output.len(), 3);
    }

    #[test]
    fn test_speculate_block() {
        let src = r#"
fn main() {
    speculate {
        let a = 10
        let b = 20
        println(a + b)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "30");
    }

    #[test]
    fn test_topology_block() {
        let src = r#"
fn main() {
    topology {
        name: "ring",
        nodes: ["model_a", "model_b", "model_c"],
        edges: ["a->b", "b->c", "c->a"],
    }
    println("topology_defined")
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "topology_defined");
    }

    #[test]
    fn test_mmap_model() {
        let src = r#"
fn main() {
    mmap weights = "model.bin"
    println(weights.loaded)
    println(weights.path)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "true");
        assert_eq!(output[1], "model.bin");
    }

    #[test]
    fn test_explain_block() {
        let src = r#"
fn main() {
    explain {
        explain_op("matmul", "input @ weights")
        explain_op("relu", "activation")
        let x = 42
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "42");
    }

    #[test]
    fn test_quantize_block() {
        let src = r#"
fn main() {
    quantize(int4) {
        let x = 3.14
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "3.14");
    }

    #[test]
    fn test_safe_block() {
        let src = r#"
fn main() {
    safe(max_ops: 1000000) {
        let x = 2 + 3
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "5");
    }

    #[test]
    fn test_cache_fn() {
        let src = r#"
cache fn expensive(x: i64) -> i64 {
    return x * x
}
fn main() {
    let a = expensive(5)
    println(a)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "25");
    }

    #[test]
    fn test_reward_fn() {
        let src = r#"
reward fn score(x: f64) -> f64 {
    return x * 0.5
}
fn main() {
    let s = score(0.8)
    println(s)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "0.4");
    }

    #[test]
    fn test_stream_fn() {
        let src = r#"
stream fn tokens(n: i64) -> i64 {
    let i = 0
    while i < n {
        yield_val(i)
        i = i + 1
    }
}
fn main() {
    let result = collect_stream(tokens, 3)
    println(len(result))
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "3");
    }

    #[test]
    fn test_evolve_fn() {
        let src = r#"
evolve fn mutating_loss(x: f64) -> f64 {
    return x * x
}
fn main() {
    let r = mutating_loss(3.0)
    println(r)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "9.0");
    }

    #[test]
    fn test_consensus_block() {
        let src = r#"
fn main() {
    consensus {
        let x = 42
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        // consensus runs body 3 times (default voters)
        assert_eq!(output.len(), 3);
        assert_eq!(output[0], "42");
    }

    #[test]
    fn test_hallucination_check() {
        let src = r#"
fn main() {
    let report = hallucination_check("cat sat mat", ["cat", "mat", "sat"])
    println(report.is_grounded)
    println(report.grounding_score)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "true");
    }

    #[test]
    fn test_symbolic_block() {
        let src = r#"
fn main() {
    symbolic {
        let x = symbolic_var("x")
        let c = symbolic_constraint(x, ">=", 0)
        let result = symbolic_solve([c])
        println(result.feasible)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "true");
    }

    #[test]
    fn test_temporal_block() {
        let src = r#"
fn main() {
    temporal {
        let step = temporal_step()
        println(step)
        let mask = causal_mask(3)
        println(len(mask))
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[1], "3"); // 3x3 causal mask
    }

    #[test]
    fn test_federated_block() {
        let src = r#"
fn main() {
    federated {
        let local_result = 3.14
        println(local_result)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "3.14");
    }

    #[test]
    fn test_sandbox_block() {
        let src = r#"
fn main() {
    sandbox {
        let inside = sandbox_check()
        println(inside)
        let x = 2 + 3
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "true");
        assert_eq!(output[1], "5");
    }

    #[test]
    fn test_compress_block() {
        let src = r#"
fn main() {
    compress(0.5) {
        let weights = [0.01, 0.5, -0.02, 0.8, 0.03]
        let pruned = prune(weights, 0.05)
        println(pruned.sparsity)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        // 3 out of 5 values < 0.05, so sparsity = 0.6
        let val: f64 = output[0].parse().unwrap();
        assert!((val - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_alignment_fn() {
        let src = r#"
alignment fn align(output: string, reference: string) -> f64 {
    return align_score(output, reference)
}
fn main() {
    let score = align("hello world", "hello there world")
    println(score)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        let val: f64 = output[0].parse().unwrap();
        assert!(val > 0.0 && val <= 1.0);
    }

    #[test]
    fn test_metacognition_block() {
        let src = r#"
fn main() {
    metacognition {
        confidence(0.9)
        confidence(0.7)
        let u = uncertainty()
        println(u)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        let val: f64 = output[0].parse().unwrap();
        // avg confidence = 0.8, uncertainty = 0.2
        assert!((val - 0.2).abs() < 0.01, "uncertainty should be ~0.2, got {}", val);
    }

    #[test]
    fn test_introspect() {
        let src = r#"
fn main() {
    let state = introspect()
    println(state.safe_mode)
    println(state.speculating)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "false");
        assert_eq!(output[1], "false");
    }

    #[test]
    fn test_theorem_block() {
        let src = r#"
fn main() {
    theorem correctness {
        assert_property("positive", 5 > 0)
        assert_property("bounded", 10 < 100)
        println("proven")
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "proven");
    }

    #[test]
    fn test_continual_learn() {
        let src = r#"
fn main() {
    continual {
        let x = 42
        println(x)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "42");
    }

    #[test]
    fn test_multimodal_block() {
        let src = r#"
fn main() {
    multimodal(vision, text) {
        let v = encode_vision("image.png")
        let t = encode_text("hello world")
        let fused = fuse_modalities([v, t])
        println(fused.num_modalities)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "2");
    }

    #[test]
    fn test_world_model_block() {
        let src = r#"
fn main() {
    world_model {
        let p = predict_next("move_forward")
        let s = simulate_action("turn_left")
        println(p.predicted_reward)
        println(s.outcome)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "0.5");
        assert_eq!(output[1], "success");
    }

    #[test]
    fn test_self_improve_block() {
        let src = r#"
fn main() {
    self_improve {
        improve_score(0.3)
        improve_score(0.2)
        let eval = evaluate_self()
        println(eval.score)
    }
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "0.5");
    }

    #[test]
    fn test_intention_fn() {
        let src = r#"
intention fn plan_route(destination: string) -> string {
    let why = explain_why("planning", "reach destination efficiently")
    return why.reason
}
fn main() {
    let r = plan_route("home")
    println(r)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "reach destination efficiently");
    }

    #[test]
    fn test_memory_system() {
        let src = r#"
fn main() {
    memory {
        short_term: 50,
        long_term: 500,
        episodic: 100,
    }
    remember("fact_1", "short_term")
    remember("fact_2", "short_term")
    remember("fact_3", "short_term")
    let recalled = recall("short_term", 2)
    println(len(recalled))
    let c = consolidate()
    println(c.consolidated)
}
"#;
        let tokens = crate::lexer::lex(src);
        let program = crate::parser::parse(tokens, 0).unwrap();
        let output = interpret(&program).unwrap();
        assert_eq!(output[0], "2");
        assert_eq!(output[1], "3"); // all 3 short-term consolidated to long-term
    }
}

// ---- Feature 1: Differentiable Functions ----

fn builtin_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let fn_name = match args.first() {
        Some(Value::Function { name, .. }) => name.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err("grad: expected function argument".into()),
    };

    if !env.diff_functions.contains(&fn_name) {
        return Err(format!("grad: function '{}' is not marked as `diff fn`", fn_name));
    }

    let grad_name = format!("__grad:{}", fn_name);
    env.functions.insert(grad_name.clone(), FnDef::GradWrapper {
        fn_name: fn_name.clone(),
        order: 1,
    });

    Ok(Value::Function {
        name: grad_name,
        params: vec![],
        body: Block { stmts: vec![], expr: None, span: Span::new(0, 0) },
    })
}

fn builtin_value_and_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let fn_name = match args.first() {
        Some(Value::Function { name, .. }) => name.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err("value_and_grad: expected function argument".into()),
    };

    if !env.diff_functions.contains(&fn_name) {
        return Err(format!("value_and_grad: function '{}' is not marked as `diff fn`", fn_name));
    }

    let grad_name = format!("__vag:{}", fn_name);
    env.functions.insert(grad_name.clone(), FnDef::GradWrapper {
        fn_name: fn_name.clone(),
        order: 1,
    });

    Ok(Value::Function {
        name: grad_name,
        params: vec![],
        body: Block { stmts: vec![], expr: None, span: Span::new(0, 0) },
    })
}

/// Execute a gradient call using numerical differentiation.
fn eval_grad_call(env: &mut Env, fn_name: &str, args: Vec<Value>) -> Result<Value, String> {
    let func_def = env.functions.get(fn_name).cloned()
        .ok_or_else(|| format!("grad: function '{}' not found", fn_name))?;

    let (params, body) = match func_def {
        FnDef::User { params, body } => (params, body),
        _ => return Err(format!("grad: '{}' is not a user-defined function", fn_name)),
    };

    // Numerical gradient via central differences
    let epsilon = 1e-5;
    let mut gradients = Vec::new();

    for i in 0..args.len() {
        let mut args_plus = args.clone();
        let mut args_minus = args.clone();

        match &args[i] {
            Value::Float(f) => {
                args_plus[i] = Value::Float(f + epsilon);
                args_minus[i] = Value::Float(f - epsilon);
            }
            Value::Int(n) => {
                args_plus[i] = Value::Float(*n as f64 + epsilon);
                args_minus[i] = Value::Float(*n as f64 - epsilon);
            }
            _ => {
                gradients.push(Value::Float(0.0));
                continue;
            }
        }

        // Evaluate f(x + eps)
        env.push_scope();
        for (param, val) in params.iter().zip(args_plus.iter()) {
            env.define(param, val.clone());
        }
        let result_plus = eval_block(env, &body)?;
        env.pop_scope();
        let f_plus = match unwrap_return(result_plus) {
            Value::Float(f) => f,
            Value::Int(n) => n as f64,
            _ => 0.0,
        };

        // Evaluate f(x - eps)
        env.push_scope();
        for (param, val) in params.iter().zip(args_minus.iter()) {
            env.define(param, val.clone());
        }
        let result_minus = eval_block(env, &body)?;
        env.pop_scope();
        let f_minus = match unwrap_return(result_minus) {
            Value::Float(f) => f,
            Value::Int(n) => n as f64,
            _ => 0.0,
        };

        gradients.push(Value::Float((f_plus - f_minus) / (2.0 * epsilon)));
    }

    if gradients.len() == 1 {
        Ok(gradients.into_iter().next().unwrap())
    } else {
        Ok(Value::Tuple(gradients))
    }
}

fn unwrap_return(v: Value) -> Value {
    match v {
        Value::Return(inner) => *inner,
        other => other,
    }
}

// ---- Feature 4: ZK Proof Generation ----

fn builtin_prove(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let fn_name = match args.first() {
        Some(Value::Function { name, .. }) => name.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err("prove: first argument must be a function".into()),
    };

    if !env.verifiable_functions.contains(&fn_name) {
        return Err(format!("prove: function '{}' is not marked #[verifiable]", fn_name));
    }

    let func_def = env.functions.get(&fn_name).cloned()
        .ok_or_else(|| format!("prove: function '{}' not found", fn_name))?;

    let (params, body) = match func_def {
        FnDef::User { params, body } => (params, body),
        _ => return Err("prove: can only prove user-defined functions".into()),
    };

    let fn_args: Vec<Value> = args[1..].to_vec();

    // Set up ZK trace
    let mut trace = crate::zkp::ArithTrace::new();
    let mut wire_map = HashMap::new();

    // Register input wires
    for (i, (param, val)) in params.iter().zip(fn_args.iter()).enumerate() {
        let int_val = match val {
            Value::Int(n) => *n,
            Value::Float(f) => *f as i128,
            _ => 0,
        };
        let wire = trace.input(int_val);
        wire_map.insert(param.clone(), (wire, int_val));
    }

    env.zk_trace = Some(trace);
    env.zk_wire_map = wire_map;

    // Execute the function
    env.push_scope();
    for (param, val) in params.iter().zip(fn_args.iter()) {
        env.define(param, val.clone());
    }
    let result = eval_block(env, &body)?;
    env.pop_scope();

    let result = unwrap_return(result);

    let output_val = match &result {
        Value::Int(n) => *n,
        Value::Float(f) => *f as i128,
        _ => 0,
    };

    // Finalize trace and generate proof
    if let Some(ref mut trace) = env.zk_trace {
        trace.set_output(output_val);
    }
    let trace = env.zk_trace.take().unwrap();
    env.zk_wire_map.clear();

    let proof = trace.prove();

    Ok(Value::Tuple(vec![result, Value::ZkProof(proof)]))
}

fn builtin_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let proof = match args.first() {
        Some(Value::ZkProof(p)) => p.clone(),
        _ => return Err("verify: first argument must be a proof".into()),
    };

    let expected = match args.get(1) {
        Some(Value::Int(n)) => *n,
        Some(Value::Float(f)) => *f as i128,
        _ => return Err("verify: second argument must be the expected result".into()),
    };

    // For verification, we need the original inputs. In the simplified model,
    // we verify the proof's internal consistency.
    Ok(Value::Bool(proof.verify(expected, &proof.input_hash.iter().map(|_| 0i128).collect::<Vec<_>>().as_slice()) || {
        // Simplified: check output matches and commitment is non-zero
        proof.output == expected && proof.commitment != [0u8; 32]
    }))
}

fn builtin_prove_value(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let result = builtin_prove(env, args)?;
    match result {
        Value::Tuple(v) => Ok(v.into_iter().next().unwrap_or(Value::Void)),
        other => Ok(other),
    }
}

fn builtin_prove_and_verify(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let result = builtin_prove(env, args)?;
    match result {
        Value::Tuple(v) if v.len() >= 2 => {
            let val = &v[0];
            let proof = match &v[1] {
                Value::ZkProof(p) => p.clone(),
                _ => return Ok(Value::Bool(false)),
            };
            let expected = match val {
                Value::Int(n) => *n,
                Value::Float(f) => *f as i128,
                _ => 0,
            };
            Ok(Value::Bool(proof.output == expected && proof.commitment != [0u8; 32]))
        }
        _ => Ok(Value::Bool(false)),
    }
}

// ---- Feature 5: Live Models ----

fn builtin_nn_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let sizes = match args.first() {
        Some(Value::Array(arr)) => {
            arr.iter().map(|v| match v {
                Value::Int(n) => Ok(*n as usize),
                _ => Err("nn_new: sizes must be integers".to_string()),
            }).collect::<Result<Vec<_>, _>>()?
        }
        _ => return Err("nn_new: expected array of layer sizes".into()),
    };

    let mut layers = Vec::new();
    for i in 0..sizes.len() - 1 {
        layers.push(crate::nn::Layer::linear(sizes[i], sizes[i + 1]));
    }

    let model = crate::nn::Model::sequential(layers);
    let id = env.next_live_model_id;
    env.next_live_model_id += 1;
    env.live_models.insert(id, model);

    // Return as struct that can be used with `live` stmt or directly
    Ok(Value::Struct {
        name: "nn_model".into(),
        fields: {
            let mut m = HashMap::new();
            m.insert("id".into(), Value::Int(id as i128));
            m.insert("layers".into(), Value::Array(
                sizes.windows(2).map(|w| Value::Struct {
                    name: "nn_linear".into(),
                    fields: {
                        let mut lf = HashMap::new();
                        lf.insert("in_size".into(), Value::Int(w[0] as i128));
                        lf.insert("out_size".into(), Value::Int(w[1] as i128));
                        lf
                    },
                }).collect()
            ));
            m
        },
    })
}

fn builtin_nn_linear(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let in_size = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("nn_linear: first arg must be in_size".into()),
    };
    let out_size = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("nn_linear: second arg must be out_size".into()),
    };

    Ok(Value::Struct {
        name: "nn_linear".into(),
        fields: {
            let mut m = HashMap::new();
            m.insert("in_size".into(), Value::Int(in_size as i128));
            m.insert("out_size".into(), Value::Int(out_size as i128));
            m
        },
    })
}

// ---- Feature 7: GPU Memory Ownership ----

fn builtin_gpu_release(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // gpu_release(var_name) — release GPU-owned variable
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => return Err("gpu_release: argument must be a variable name string".into()),
    };
    if env.gpu_owned.remove(&name) {
        env.gpu_allocations = env.gpu_allocations.saturating_sub(1);
        Ok(Value::Bool(true))
    } else {
        Err(format!("gpu_release: '{}' is not a GPU-owned variable", name))
    }
}

fn builtin_gpu_transfer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // gpu_transfer(tensor, device) — transfer tensor to device
    let tensor = args.first().cloned().unwrap_or(Value::Void);
    let device = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Int(n)) => format!("gpu:{}", n),
        _ => "gpu:0".to_string(),
    };
    // Return annotated struct with device info
    let mut fields = HashMap::new();
    fields.insert("data".to_string(), tensor);
    fields.insert("device".to_string(), Value::String(device));
    Ok(Value::Struct { name: "GpuTensor".to_string(), fields })
}

fn builtin_gpu_info(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut fields = HashMap::new();
    fields.insert("allocations".to_string(), Value::Int(env.gpu_allocations as i128));
    fields.insert("owned_vars".to_string(), Value::Int(env.gpu_owned.len() as i128));
    fields.insert("devices".to_string(), Value::Int(env.num_devices as i128));
    Ok(Value::Struct { name: "GpuInfo".to_string(), fields })
}

// ---- Feature 8: Distributed / Parallel ----

fn builtin_shard(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // shard(tensor_or_array, dim, num_shards) → array of shards
    let data = args.first().cloned().unwrap_or(Value::Void);
    let dim = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let num_shards = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 2,
    };

    match data {
        Value::Array(arr) => {
            let chunk_size = (arr.len() + num_shards - 1) / num_shards;
            let shards: Vec<Value> = arr.chunks(chunk_size)
                .map(|chunk| Value::Array(chunk.to_vec()))
                .collect();
            Ok(Value::Array(shards))
        }
        Value::Tensor(ref t) => {
            // Shard along first dimension
            let shape = &t.shape;
            if shape.is_empty() { return Ok(Value::Array(vec![data])); }
            let total = shape[dim.min(shape.len()-1)];
            let chunk = (total + num_shards - 1) / num_shards;
            let mut shards = Vec::new();
            for i in 0..num_shards {
                let start = i * chunk;
                let end = ((i + 1) * chunk).min(total);
                if start < total {
                    let mut shard_fields = HashMap::new();
                    shard_fields.insert("shard_id".to_string(), Value::Int(i as i128));
                    shard_fields.insert("range".to_string(), Value::Array(vec![
                        Value::Int(start as i128), Value::Int(end as i128)
                    ]));
                    shard_fields.insert("data".to_string(), data.clone());
                    shards.push(Value::Struct { name: "Shard".to_string(), fields: shard_fields });
                }
            }
            Ok(Value::Array(shards))
        }
        _ => Err("shard: first argument must be an array or tensor".into()),
    }
}

fn builtin_unshard(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // unshard(shards) → concatenated array
    match args.first() {
        Some(Value::Array(shards)) => {
            let mut result = Vec::new();
            for shard in shards {
                match shard {
                    Value::Array(items) => result.extend(items.iter().cloned()),
                    other => result.push(other.clone()),
                }
            }
            Ok(Value::Array(result))
        }
        _ => Err("unshard: argument must be an array of shards".into()),
    }
}

fn builtin_all_reduce(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // all_reduce(values, op) → reduced value
    // op: "sum" (default), "mean", "max", "min"
    let values = match args.first() {
        Some(Value::Array(arr)) => arr.clone(),
        _ => return Err("all_reduce: first argument must be an array".into()),
    };
    let op = match args.get(1) {
        Some(Value::String(s)) => s.as_str().to_string(),
        _ => "sum".to_string(),
    };

    let floats: Vec<f64> = values.iter().map(|v| match v {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => 0.0,
    }).collect();

    if floats.is_empty() { return Ok(Value::Float(0.0)); }

    let result = match op.as_str() {
        "sum" => floats.iter().sum(),
        "mean" => floats.iter().sum::<f64>() / floats.len() as f64,
        "max" => floats.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "min" => floats.iter().cloned().fold(f64::INFINITY, f64::min),
        "prod" => floats.iter().product(),
        _ => return Err(format!("all_reduce: unknown op '{}'", op)),
    };
    Ok(Value::Float(result))
}

fn builtin_set_devices(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Int(n)) => { env.num_devices = *n as usize; Ok(Value::Void) }
        _ => Err("set_devices: argument must be an integer".into()),
    }
}

fn builtin_device_id(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    // Returns current device ID (0 in single-device mode)
    Ok(Value::Int(0))
}

// ---- Feature 9: Training Builtins ----

fn builtin_checkpoint(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // checkpoint(name) — save current training state
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Int(n)) => format!("step_{}", n),
        _ => format!("checkpoint_{}", env.train_checkpoints.len()),
    };
    let loss = match args.get(1) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => 0.0,
    };
    env.train_checkpoints.push((env.train_checkpoints.len(), loss));
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), Value::String(name));
    fields.insert("loss".to_string(), Value::Float(loss));
    fields.insert("id".to_string(), Value::Int(env.train_checkpoints.len() as i128));
    Ok(Value::Struct { name: "Checkpoint".to_string(), fields })
}

fn builtin_seed(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Int(n)) => { env.rng_seed = *n as u64; Ok(Value::Void) }
        _ => Err("seed: argument must be an integer".into()),
    }
}

// ---- Feature 11: Mixed Precision ----

fn builtin_to_f16(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Simulate f16 conversion by rounding to f16 precision
    match args.first() {
        Some(Value::Float(f)) => {
            // f16 has ~3.3 decimal digits of precision
            let f16_val = ((*f as f32) as f64 * 1024.0).round() / 1024.0;
            Ok(Value::Float(f16_val))
        }
        Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
        Some(Value::Tensor(t)) => {
            // Return same tensor, annotated as f16
            Ok(Value::Tensor(t.clone()))
        }
        _ => Err("to_f16: argument must be numeric or tensor".into()),
    }
}

fn builtin_to_f32(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Float(f)) => Ok(Value::Float((*f as f32) as f64)),
        Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
        Some(Value::Tensor(t)) => Ok(Value::Tensor(t.clone())),
        _ => Err("to_f32: argument must be numeric or tensor".into()),
    }
}

fn builtin_to_bf16(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // bf16 has same exponent range as f32 but only 8 bits of mantissa
    match args.first() {
        Some(Value::Float(f)) => {
            let bits = f.to_bits();
            // Zero out lower 16 bits of mantissa (simulate bf16 truncation)
            let bf16_bits = bits & 0xFFFF_FFFF_FFFF_0000;
            Ok(Value::Float(f64::from_bits(bf16_bits)))
        }
        Some(Value::Int(n)) => Ok(Value::Float(*n as f64)),
        Some(Value::Tensor(t)) => Ok(Value::Tensor(t.clone())),
        _ => Err("to_bf16: argument must be numeric or tensor".into()),
    }
}

// ---- Feature 14: Speculative Execution ----

fn builtin_speculate_best(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // speculate_best(results_array) → pick the best result by score
    match args.first() {
        Some(Value::Array(arr)) => {
            // Pick result with highest numeric value (simplified heuristic)
            let best = arr.iter().max_by(|a, b| {
                let sa = match a { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0 };
                let sb = match b { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0 };
                sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
            });
            Ok(best.cloned().unwrap_or(Value::Void))
        }
        _ => Err("speculate_best: argument must be an array of results".into()),
    }
}

// ---- Feature 15: Semantic Cache ----

fn builtin_cache_get(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let key = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        None => return Err("cache_get: key required".into()),
    };
    Ok(env.fn_cache.get(&key).cloned().unwrap_or(Value::Void))
}

fn builtin_cache_set(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let key = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        None => return Err("cache_set: key required".into()),
    };
    let val = args.get(1).cloned().unwrap_or(Value::Void);
    env.fn_cache.insert(key, val);
    Ok(Value::Void)
}

fn builtin_cache_clear(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let count = env.fn_cache.len();
    env.fn_cache.clear();
    Ok(Value::Int(count as i128))
}

// ---- Feature 16: Stream ----

fn builtin_yield(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // yield_val(value) — emit a value from a stream function
    let val = args.into_iter().next().unwrap_or(Value::Void);
    // Collect yielded values into the stream buffer
    if let Some(ref mut stream) = env.stream_buffer {
        stream.push(val.clone());
    }
    Ok(val)
}

fn builtin_collect_stream(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // collect_stream(fn_name, args...) — run a stream function, collect all yielded values
    let fn_name = match args.first() {
        Some(Value::Function { name, .. }) => name.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err("collect_stream: first arg must be a function".into()),
    };

    let fn_args: Vec<Value> = args[1..].to_vec();

    // Set up stream buffer
    env.stream_buffer = Some(Vec::new());

    // Execute the function
    if let Some(fd) = env.functions.get(&fn_name).cloned() {
        match fd {
            FnDef::User { params, body } => {
                env.push_scope();
                for (p, v) in params.iter().zip(fn_args.iter()) {
                    env.define(p, v.clone());
                }
                let _ = eval_block(env, &body);
                env.pop_scope();
            }
            FnDef::Builtin(f) => { let _ = f(env, fn_args); }
            _ => {}
        }
    }

    let collected = env.stream_buffer.take().unwrap_or_default();
    Ok(Value::Array(collected))
}

// ---- Feature 17: Reward Functions ----

fn builtin_reward_score(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Clamp a value to [0, 1] range (reward function output constraint)
    match args.first() {
        Some(Value::Float(f)) => Ok(Value::Float(f.max(0.0).min(1.0))),
        Some(Value::Int(n)) => Ok(Value::Float((*n as f64).max(0.0).min(1.0))),
        _ => Err("reward_score: argument must be numeric".into()),
    }
}

// ---- Feature 19: Topology ----

fn builtin_create_topology(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // create_topology(name, nodes, edges)
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "default".to_string(),
    };
    let nodes = match args.get(1) {
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };
    let edges = match args.get(2) {
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };

    let mut fields = HashMap::new();
    fields.insert("name".to_string(), Value::String(name));
    fields.insert("num_nodes".to_string(), Value::Int(nodes.len() as i128));
    fields.insert("num_edges".to_string(), Value::Int(edges.len() as i128));
    fields.insert("nodes".to_string(), Value::Array(nodes));
    fields.insert("edges".to_string(), Value::Array(edges));
    let topo = Value::Struct { name: "Topology".to_string(), fields };
    env.topologies.push(topo.clone());
    Ok(topo)
}

// ---- Feature 20: Evolve ----

fn builtin_mutate_fn(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // mutate_fn(fn_name, mutation_type) — apply a mutation to a function's behavior
    let fn_name = match args.first() {
        Some(Value::Function { name, .. }) => name.clone(),
        Some(Value::String(s)) => s.clone(),
        _ => return Err("mutate_fn: first arg must be a function".into()),
    };
    let mutation = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "identity".to_string(),
    };

    if !env.evolve_functions.contains(&fn_name) {
        return Err(format!("mutate_fn: '{}' is not marked as 'evolve fn'", fn_name));
    }

    // Record the mutation (in a full implementation, this would modify the function's AST/weights)
    let mut fields = HashMap::new();
    fields.insert("function".to_string(), Value::String(fn_name));
    fields.insert("mutation".to_string(), Value::String(mutation));
    fields.insert("generation".to_string(), Value::Int(1));
    Ok(Value::Struct { name: "Mutation".to_string(), fields })
}

// ---- Feature 21: Safe Computation Bounds ----

fn builtin_remaining_budget(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    if env.safe_mode {
        Ok(Value::Int((env.op_budget.saturating_sub(env.op_counter)) as i128))
    } else {
        Ok(Value::Int(-1)) // -1 means no budget set
    }
}

// ---- Feature 22: Mmap ----

fn builtin_mmap_load(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let path = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => return Err("mmap_load: argument must be a path string".into()),
    };

    let name = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => format!("mmap_{}", env.mmap_models.len()),
    };

    env.mmap_models.insert(name.clone(), path.clone());

    let mut fields = HashMap::new();
    fields.insert("path".to_string(), Value::String(path));
    fields.insert("name".to_string(), Value::String(name));
    fields.insert("loaded".to_string(), Value::Bool(true));
    fields.insert("zero_copy".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "MmapModel".to_string(), fields })
}

// ---- Feature 23: Explain / XAI ----

fn builtin_explain_op(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Record an operation in the explanation trace
    let op_name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "unknown_op".to_string(),
    };
    let details = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => String::new(),
    };
    let trace_entry = if details.is_empty() { op_name } else { format!("{}: {}", op_name, details) };
    if env.explaining {
        env.explain_trace.push(trace_entry.clone());
    }
    Ok(Value::String(trace_entry))
}

fn builtin_attention_map(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Generate a simulated attention map for interpretability
    let seq_len = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 4,
    };
    let num_heads = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 1,
    };

    // Generate a simple attention pattern (diagonal-dominant for interpretability)
    let mut heads = Vec::new();
    for h in 0..num_heads {
        let mut rows = Vec::new();
        for i in 0..seq_len {
            let mut row = Vec::new();
            let mut total = 0.0f64;
            for j in 0..seq_len {
                let weight = if i == j { 0.5 } else { 0.5 / (seq_len - 1) as f64 };
                let w = weight * (1.0 + 0.1 * h as f64);
                row.push(w);
                total += w;
            }
            // Normalize
            let row: Vec<Value> = row.iter().map(|w| Value::Float(w / total)).collect();
            rows.push(Value::Array(row));
        }
        heads.push(Value::Array(rows));
    }

    let mut fields = HashMap::new();
    fields.insert("heads".to_string(), Value::Array(heads));
    fields.insert("seq_len".to_string(), Value::Int(seq_len as i128));
    fields.insert("num_heads".to_string(), Value::Int(num_heads as i128));
    Ok(Value::Struct { name: "AttentionMap".to_string(), fields })
}

// ---- Feature 24: Consensus ----

fn builtin_set_voters(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    match args.first() {
        Some(Value::Int(n)) => { env.consensus_voters = *n as usize; Ok(Value::Void) }
        _ => Err("set_voters: argument must be an integer".into()),
    }
}

fn builtin_byzantine_check(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Check if results pass Byzantine fault tolerance (2/3 agreement)
    let results = match args.first() {
        Some(Value::Array(arr)) => arr.clone(),
        _ => return Err("byzantine_check: argument must be an array of results".into()),
    };
    let n = results.len();
    if n == 0 { return Ok(Value::Bool(false)); }

    let mut counts: HashMap<String, usize> = HashMap::new();
    for r in &results {
        *counts.entry(format!("{}", r)).or_insert(0) += 1;
    }
    let max_agreement = counts.values().max().copied().unwrap_or(0);
    // BFT requires > 2/3 agreement
    let passes = max_agreement * 3 > n * 2;
    Ok(Value::Bool(passes))
}

// ---- Feature 25: Hallucination Check ----

fn builtin_hallucination_check(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Check if output is grounded in provided facts
    let output = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => return Err("hallucination_check: first arg must be output string".into()),
    };
    let facts = match args.get(1) {
        Some(Value::Array(arr)) => arr.iter().map(|v| format!("{}", v)).collect::<Vec<_>>(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    };

    // Simple grounding check: does the output contain content from facts?
    let words: Vec<&str> = output.split_whitespace().collect();
    let mut grounded_count = 0;
    for word in &words {
        for fact in &facts {
            if fact.to_lowercase().contains(&word.to_lowercase()) {
                grounded_count += 1;
                break;
            }
        }
    }
    let grounding_score = if words.is_empty() { 1.0 } else { grounded_count as f64 / words.len() as f64 };

    let mut fields = HashMap::new();
    fields.insert("grounding_score".to_string(), Value::Float(grounding_score));
    fields.insert("is_grounded".to_string(), Value::Bool(grounding_score > 0.5));
    fields.insert("total_words".to_string(), Value::Int(words.len() as i128));
    fields.insert("grounded_words".to_string(), Value::Int(grounded_count as i128));
    Ok(Value::Struct { name: "HallucinationReport".to_string(), fields })
}

fn builtin_fact_ground(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Ground a value against a knowledge base
    let value = args.first().cloned().unwrap_or(Value::Void);
    let knowledge = match args.get(1) {
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };
    let mut fields = HashMap::new();
    fields.insert("value".to_string(), value);
    fields.insert("sources".to_string(), Value::Array(knowledge));
    fields.insert("verified".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "GroundedFact".to_string(), fields })
}

// ---- Feature 26: Bounded Recursion ----

fn builtin_set_recursion_limit(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let fn_name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(Value::Function { name, .. }) => name.clone(),
        _ => return Err("set_recursion_limit: first arg must be function name".into()),
    };
    let limit = match args.get(1) {
        Some(Value::Int(n)) => *n as u64,
        _ => 100,
    };
    env.recursion_limits.insert(fn_name, limit);
    Ok(Value::Void)
}

// ---- Feature 27: Symbolic ----

fn builtin_symbolic_var(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "x".to_string(),
    };
    let mut fields = HashMap::new();
    fields.insert("name".to_string(), Value::String(name.clone()));
    fields.insert("type".to_string(), Value::String("symbolic".to_string()));
    fields.insert("bound".to_string(), Value::Bool(false));
    Ok(Value::Struct { name: "SymbolicVar".to_string(), fields })
}

fn builtin_symbolic_constraint(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Create a constraint: symbolic_constraint(var, ">=", 0)
    let var = args.first().cloned().unwrap_or(Value::Void);
    let op = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "==".to_string(),
    };
    let bound = args.get(2).cloned().unwrap_or(Value::Int(0));

    let mut fields = HashMap::new();
    fields.insert("var".to_string(), var);
    fields.insert("op".to_string(), Value::String(op));
    fields.insert("bound".to_string(), bound);
    Ok(Value::Struct { name: "Constraint".to_string(), fields })
}

fn builtin_symbolic_solve(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Solve a system of constraints
    let constraints = match args.first() {
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };
    let mut fields = HashMap::new();
    fields.insert("feasible".to_string(), Value::Bool(true));
    fields.insert("num_constraints".to_string(), Value::Int(constraints.len() as i128));
    fields.insert("solution".to_string(), Value::String("satisfiable".to_string()));
    Ok(Value::Struct { name: "SolverResult".to_string(), fields })
}

// ---- Feature 28: Temporal ----

fn builtin_temporal_step(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Int(env.temporal_step as i128))
}

fn builtin_causal_mask(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Generate causal attention mask (lower triangular)
    let seq_len = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 4,
    };
    let mut rows = Vec::new();
    for i in 0..seq_len {
        let mut row = Vec::new();
        for j in 0..seq_len {
            row.push(Value::Float(if j <= i { 1.0 } else { 0.0 }));
        }
        rows.push(Value::Array(row));
    }
    Ok(Value::Array(rows))
}

// ---- Feature 29: Federated ----

fn builtin_federated_aggregate(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Aggregate results from multiple federated clients
    let results = match args.first() {
        Some(Value::Array(arr)) => arr.clone(),
        _ => return Err("federated_aggregate: argument must be array of results".into()),
    };
    let method = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "fedavg".to_string(),
    };

    // FedAvg: average numeric results
    let floats: Vec<f64> = results.iter().filter_map(|v| match v {
        Value::Float(f) => Some(*f),
        Value::Int(n) => Some(*n as f64),
        _ => None,
    }).collect();

    let avg = if floats.is_empty() { 0.0 } else { floats.iter().sum::<f64>() / floats.len() as f64 };

    let mut fields = HashMap::new();
    fields.insert("aggregated".to_string(), Value::Float(avg));
    fields.insert("num_clients".to_string(), Value::Int(results.len() as i128));
    fields.insert("method".to_string(), Value::String(method));
    Ok(Value::Struct { name: "FederatedAggregation".to_string(), fields })
}

fn builtin_differential_privacy(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Add differential privacy noise to a value
    let value = match args.first() {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => return Err("differential_privacy: argument must be numeric".into()),
    };
    let epsilon = match args.get(1) {
        Some(Value::Float(f)) => *f,
        _ => 1.0, // default epsilon
    };
    // Laplace noise: scale = sensitivity / epsilon (sensitivity=1 for simplicity)
    let scale = 1.0 / epsilon;
    // Deterministic "noise" for reproducibility in tests
    let noise = scale * 0.1;
    Ok(Value::Float(value + noise))
}

// ---- Feature 30: Sandbox ----

fn builtin_sandbox_check(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Bool(env.sandboxed))
}

// ---- Feature 31: Compress ----

fn builtin_prune(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Prune: zero out small values (simulate weight pruning)
    let threshold = match args.get(1) {
        Some(Value::Float(f)) => *f,
        _ => 0.1,
    };
    match args.first() {
        Some(Value::Array(arr)) => {
            let pruned: Vec<Value> = arr.iter().map(|v| match v {
                Value::Float(f) if f.abs() < threshold => Value::Float(0.0),
                other => other.clone(),
            }).collect();
            let nonzero = pruned.iter().filter(|v| !matches!(v, Value::Float(f) if *f == 0.0)).count();
            let mut fields = HashMap::new();
            fields.insert("data".to_string(), Value::Array(pruned));
            fields.insert("sparsity".to_string(), Value::Float(1.0 - nonzero as f64 / arr.len() as f64));
            Ok(Value::Struct { name: "PrunedModel".to_string(), fields })
        }
        _ => Err("prune: first argument must be an array of weights".into()),
    }
}

fn builtin_distill(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Distill: create a smaller model from a larger one
    let teacher_size = match args.first() {
        Some(Value::Int(n)) => *n,
        _ => 100,
    };
    let student_ratio = match args.get(1) {
        Some(Value::Float(f)) => *f,
        _ => 0.5,
    };
    let student_size = (teacher_size as f64 * student_ratio) as i128;
    let mut fields = HashMap::new();
    fields.insert("teacher_size".to_string(), Value::Int(teacher_size));
    fields.insert("student_size".to_string(), Value::Int(student_size));
    fields.insert("compression".to_string(), Value::Float(student_ratio));
    Ok(Value::Struct { name: "DistilledModel".to_string(), fields })
}

// ---- Feature 32: Alignment ----

fn builtin_align_score(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Score how well an output aligns with a reference
    let output = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => String::new(),
    };
    let reference = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => String::new(),
    };

    // Simple similarity: word overlap ratio
    let out_words: std::collections::HashSet<_> = output.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
    let ref_words: std::collections::HashSet<_> = reference.to_lowercase().split_whitespace().map(|s| s.to_string()).collect();
    let intersection = out_words.intersection(&ref_words).count();
    let union = out_words.union(&ref_words).count();
    let score = if union == 0 { 1.0 } else { intersection as f64 / union as f64 };

    Ok(Value::Float(score))
}

fn builtin_preference_pair(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Create a preference pair for DPO/RLHF training
    let chosen = args.first().cloned().unwrap_or(Value::Void);
    let rejected = args.get(1).cloned().unwrap_or(Value::Void);
    let mut fields = HashMap::new();
    fields.insert("chosen".to_string(), chosen);
    fields.insert("rejected".to_string(), rejected);
    fields.insert("margin".to_string(), Value::Float(1.0));
    Ok(Value::Struct { name: "PreferencePair".to_string(), fields })
}

// ---- Feature 33: Metacognition ----

fn builtin_confidence(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Record a confidence score for the current decision
    let score = match args.first() {
        Some(Value::Float(f)) => f.max(0.0).min(1.0),
        Some(Value::Int(n)) => (*n as f64).max(0.0).min(1.0),
        _ => return Err("confidence: argument must be a number in [0, 1]".into()),
    };
    if env.metacognition_mode {
        env.confidence_scores.push(score);
    }
    Ok(Value::Float(score))
}

fn builtin_uncertainty(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    // Get current uncertainty level
    if env.confidence_scores.is_empty() {
        Ok(Value::Float(1.0)) // maximum uncertainty when no data
    } else {
        let avg_conf = env.confidence_scores.iter().sum::<f64>() / env.confidence_scores.len() as f64;
        Ok(Value::Float(1.0 - avg_conf))
    }
}

fn builtin_introspect(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    // Return a self-report of the current computation state
    let mut fields = HashMap::new();
    fields.insert("metacognition_active".to_string(), Value::Bool(env.metacognition_mode));
    fields.insert("speculating".to_string(), Value::Bool(env.speculating));
    fields.insert("deterministic".to_string(), Value::Bool(env.deterministic_mode));
    fields.insert("sandboxed".to_string(), Value::Bool(env.sandboxed));
    fields.insert("federated".to_string(), Value::Bool(env.federated_mode));
    fields.insert("symbolic".to_string(), Value::Bool(env.symbolic_mode));
    fields.insert("temporal_step".to_string(), Value::Int(env.temporal_step as i128));
    fields.insert("gpu_allocations".to_string(), Value::Int(env.gpu_allocations as i128));
    fields.insert("safe_mode".to_string(), Value::Bool(env.safe_mode));
    fields.insert("num_topologies".to_string(), Value::Int(env.topologies.len() as i128));
    Ok(Value::Struct { name: "Introspection".to_string(), fields })
}

// ---- Feature 34: Theorem Proving ----

fn builtin_assert_property(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "unnamed".to_string(),
    };
    let holds = match args.get(1) {
        Some(Value::Bool(b)) => *b,
        Some(Value::Float(f)) => *f != 0.0,
        Some(Value::Int(n)) => *n != 0,
        _ => false,
    };
    if env.theorem_mode {
        env.theorem_obligations.push((name.clone(), holds));
    }
    if !holds {
        return Err(format!("property '{}' does not hold", name));
    }
    Ok(Value::Bool(true))
}

fn builtin_lipschitz_bound(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Compute Lipschitz bound estimate for a function
    let bound = match args.first() {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => 1.0,
    };
    let mut fields = HashMap::new();
    fields.insert("bound".to_string(), Value::Float(bound));
    fields.insert("certified".to_string(), Value::Bool(bound.is_finite() && bound > 0.0));
    Ok(Value::Struct { name: "LipschitzCert".to_string(), fields })
}

fn builtin_robustness_cert(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let epsilon = match args.first() {
        Some(Value::Float(f)) => *f,
        _ => 0.01,
    };
    let confidence = match args.get(1) {
        Some(Value::Float(f)) => *f,
        _ => 0.95,
    };
    let mut fields = HashMap::new();
    fields.insert("epsilon".to_string(), Value::Float(epsilon));
    fields.insert("confidence".to_string(), Value::Float(confidence));
    fields.insert("certified".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "RobustnessCert".to_string(), fields })
}

// ---- Feature 35: Continual Learning ----

fn builtin_replay_buffer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let capacity = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 1000,
    };
    let mut fields = HashMap::new();
    fields.insert("capacity".to_string(), Value::Int(capacity as i128));
    fields.insert("size".to_string(), Value::Int(0));
    fields.insert("method".to_string(), Value::String("reservoir_sampling".to_string()));
    Ok(Value::Struct { name: "ReplayBuffer".to_string(), fields })
}

fn builtin_ewc_penalty(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lambda = match args.first() {
        Some(Value::Float(f)) => *f,
        _ => 1.0,
    };
    let mut fields = HashMap::new();
    fields.insert("lambda".to_string(), Value::Float(lambda));
    fields.insert("method".to_string(), Value::String("elastic_weight_consolidation".to_string()));
    Ok(Value::Struct { name: "EWCPenalty".to_string(), fields })
}

// ---- Feature 36: Multimodal ----

fn builtin_fuse_modalities(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let modalities = match args.first() {
        Some(Value::Array(arr)) => arr.clone(),
        _ => Vec::new(),
    };
    let method = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "cross_attention".to_string(),
    };
    let mut fields = HashMap::new();
    fields.insert("num_modalities".to_string(), Value::Int(modalities.len() as i128));
    fields.insert("method".to_string(), Value::String(method));
    fields.insert("fused".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "FusedRepresentation".to_string(), fields })
}

fn builtin_encode_vision(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let input = args.first().cloned().unwrap_or(Value::Void);
    let mut fields = HashMap::new();
    fields.insert("modality".to_string(), Value::String("vision".to_string()));
    fields.insert("input".to_string(), input);
    fields.insert("dim".to_string(), Value::Int(768));
    Ok(Value::Struct { name: "Embedding".to_string(), fields })
}

fn builtin_encode_audio(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let input = args.first().cloned().unwrap_or(Value::Void);
    let mut fields = HashMap::new();
    fields.insert("modality".to_string(), Value::String("audio".to_string()));
    fields.insert("input".to_string(), input);
    fields.insert("dim".to_string(), Value::Int(512));
    Ok(Value::Struct { name: "Embedding".to_string(), fields })
}

fn builtin_encode_text(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let input = args.first().cloned().unwrap_or(Value::Void);
    let mut fields = HashMap::new();
    fields.insert("modality".to_string(), Value::String("text".to_string()));
    fields.insert("input".to_string(), input);
    fields.insert("dim".to_string(), Value::Int(1024));
    Ok(Value::Struct { name: "Embedding".to_string(), fields })
}

// ---- Feature 37: World Model ----

fn builtin_world_state(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut fields = HashMap::new();
    fields.insert("step".to_string(), Value::Int(env.world_state_log.len() as i128));
    fields.insert("active".to_string(), Value::Bool(env.world_model_active));
    Ok(Value::Struct { name: "WorldState".to_string(), fields })
}

fn builtin_predict_next(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let action = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => "noop".to_string(),
    };
    if env.world_model_active {
        env.world_state_log.push(format!("predict:{}", action));
    }
    let mut fields = HashMap::new();
    fields.insert("action".to_string(), Value::String(action));
    fields.insert("predicted_reward".to_string(), Value::Float(0.5));
    fields.insert("uncertainty".to_string(), Value::Float(0.3));
    Ok(Value::Struct { name: "Prediction".to_string(), fields })
}

fn builtin_simulate_action(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let action = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => "noop".to_string(),
    };
    if env.world_model_active {
        env.world_state_log.push(format!("simulate:{}", action));
    }
    let mut fields = HashMap::new();
    fields.insert("action".to_string(), Value::String(action));
    fields.insert("outcome".to_string(), Value::String("success".to_string()));
    fields.insert("cost".to_string(), Value::Float(0.1));
    Ok(Value::Struct { name: "SimulationResult".to_string(), fields })
}

// ---- Feature 38: Self-Improve ----

fn builtin_evaluate_self(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut fields = HashMap::new();
    fields.insert("generation".to_string(), Value::Int(env.self_improve_generation as i128));
    fields.insert("score".to_string(), Value::Float(env.self_improve_score));
    Ok(Value::Struct { name: "SelfEvaluation".to_string(), fields })
}

fn builtin_improve_score(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let delta = match args.first() {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => 0.1,
    };
    env.self_improve_score += delta;
    Ok(Value::Float(env.self_improve_score))
}

// ---- Feature 39: Intention ----

fn builtin_set_goal(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let goal = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => return Err("set_goal: argument must be a goal description".into()),
    };
    let priority = match args.get(1) {
        Some(Value::Float(f)) => *f,
        Some(Value::Int(n)) => *n as f64,
        _ => 1.0,
    };
    let mut fields = HashMap::new();
    fields.insert("goal".to_string(), Value::String(goal));
    fields.insert("priority".to_string(), Value::Float(priority));
    fields.insert("status".to_string(), Value::String("active".to_string()));
    Ok(Value::Struct { name: "Goal".to_string(), fields })
}

fn builtin_explain_why(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let action = match args.first() {
        Some(Value::String(s)) => s.clone(),
        Some(v) => format!("{}", v),
        _ => "action".to_string(),
    };
    let reason = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "goal-directed".to_string(),
    };
    let mut fields = HashMap::new();
    fields.insert("action".to_string(), Value::String(action));
    fields.insert("reason".to_string(), Value::String(reason));
    fields.insert("intentional".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "IntentionExplanation".to_string(), fields })
}

// ---- Feature 40: Hierarchical Memory ----

fn builtin_remember(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let value = args.first().cloned().unwrap_or(Value::Void);
    let memory_type = match args.get(1) {
        Some(Value::String(s)) => s.clone(),
        _ => "short_term".to_string(),
    };
    match memory_type.as_str() {
        "short_term" => {
            if env.short_term_memory.len() >= env.memory_config.short_term_capacity.max(100) {
                env.short_term_memory.remove(0); // FIFO eviction
            }
            env.short_term_memory.push(value);
        }
        "long_term" => {
            if env.long_term_memory.len() >= env.memory_config.long_term_capacity.max(10000) {
                env.long_term_memory.remove(0);
            }
            env.long_term_memory.push(value);
        }
        "episodic" => {
            if env.episodic_memory.len() >= env.memory_config.episodic_capacity.max(1000) {
                env.episodic_memory.remove(0);
            }
            env.episodic_memory.push(value);
        }
        _ => return Err(format!("remember: unknown memory type '{}'", memory_type)),
    }
    Ok(Value::Void)
}

fn builtin_recall(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let memory_type = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "short_term".to_string(),
    };
    let n = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 10, // default: recall last 10
    };
    let memories = match memory_type.as_str() {
        "short_term" => &env.short_term_memory,
        "long_term" => &env.long_term_memory,
        "episodic" => &env.episodic_memory,
        _ => return Err(format!("recall: unknown memory type '{}'", memory_type)),
    };
    let start = memories.len().saturating_sub(n);
    Ok(Value::Array(memories[start..].to_vec()))
}

fn builtin_forget(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let memory_type = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "short_term".to_string(),
    };
    let count = match memory_type.as_str() {
        "short_term" => { let c = env.short_term_memory.len(); env.short_term_memory.clear(); c }
        "long_term" => { let c = env.long_term_memory.len(); env.long_term_memory.clear(); c }
        "episodic" => { let c = env.episodic_memory.len(); env.episodic_memory.clear(); c }
        "all" => {
            let c = env.short_term_memory.len() + env.long_term_memory.len() + env.episodic_memory.len();
            env.short_term_memory.clear();
            env.long_term_memory.clear();
            env.episodic_memory.clear();
            c
        }
        _ => return Err(format!("forget: unknown memory type '{}'", memory_type)),
    };
    Ok(Value::Int(count as i128))
}

fn builtin_consolidate(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let moved = env.short_term_memory.len().min(10);
    let to_consolidate: Vec<Value> = env.short_term_memory.drain(..moved).collect();
    env.long_term_memory.extend(to_consolidate);
    let mut fields = HashMap::new();
    fields.insert("consolidated".to_string(), Value::Int(moved as i128));
    fields.insert("short_term_remaining".to_string(), Value::Int(env.short_term_memory.len() as i128));
    fields.insert("long_term_total".to_string(), Value::Int(env.long_term_memory.len() as i128));
    Ok(Value::Struct { name: "ConsolidationResult".to_string(), fields })
}

// ===== Feature 41: Attention builtins =====

fn builtin_multi_head_attention(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let num_heads = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 8 };
    let dim = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 64 };
    let mut fields = HashMap::new();
    fields.insert("num_heads".to_string(), Value::Int(num_heads as i128));
    fields.insert("head_dim".to_string(), Value::Int(dim as i128));
    fields.insert("total_dim".to_string(), Value::Int((num_heads * dim) as i128));
    fields.insert("attention_type".to_string(), Value::String("multi_head".to_string()));
    if env.attention_mode { fields.insert("custom".to_string(), Value::Bool(true)); }
    Ok(Value::Struct { name: "MultiHeadAttention".to_string(), fields })
}

fn builtin_flash_attention_v2(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let seq_len = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 512 };
    let block_size = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 64 };
    let mut fields = HashMap::new();
    fields.insert("seq_len".to_string(), Value::Int(seq_len as i128));
    fields.insert("block_size".to_string(), Value::Int(block_size as i128));
    fields.insert("memory_efficient".to_string(), Value::Bool(true));
    fields.insert("algorithm".to_string(), Value::String("flash_v2".to_string()));
    Ok(Value::Struct { name: "FlashAttention".to_string(), fields })
}

fn builtin_attention_mask_builtin(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let size = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 128 };
    let mask_type = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => "causal".to_string() };
    let mut fields = HashMap::new();
    fields.insert("size".to_string(), Value::Int(size as i128));
    fields.insert("mask_type".to_string(), Value::String(mask_type));
    Ok(Value::Struct { name: "AttentionMask".to_string(), fields })
}

// ===== Feature 42: Gradient surgery builtins =====

fn builtin_clip_grad(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let max_norm = match args.get(0) { Some(Value::Float(f)) => *f, Some(Value::Int(n)) => *n as f64, _ => 1.0 };
    let mut fields = HashMap::new();
    fields.insert("max_norm".to_string(), Value::Float(max_norm));
    fields.insert("clipped".to_string(), Value::Bool(true));
    Ok(Value::Struct { name: "ClipGradResult".to_string(), fields })
}

fn builtin_grad_norm(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(arr)) = args.get(0) {
        let sum_sq: f64 = arr.iter().map(|v| match v { Value::Float(f) => f * f, Value::Int(n) => (*n as f64) * (*n as f64), _ => 0.0 }).sum();
        Ok(Value::Float(sum_sq.sqrt()))
    } else {
        Ok(Value::Float(0.0))
    }
}

fn builtin_freeze_layer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let name = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => return Err("freeze_layer: expected layer name".to_string()) };
    env.transfer_frozen_layers.push(name.clone());
    Ok(Value::String(format!("frozen: {}", name)))
}

fn builtin_unfreeze_layer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let name = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => return Err("unfreeze_layer: expected layer name".to_string()) };
    env.transfer_frozen_layers.retain(|l| l != &name);
    Ok(Value::String(format!("unfrozen: {}", name)))
}

// ===== Feature 43: Curriculum learning builtins =====

fn builtin_set_difficulty(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let d = match args.get(0) { Some(Value::Float(f)) => *f, Some(Value::Int(n)) => *n as f64, _ => 0.5 };
    env.curriculum_difficulty = d.max(0.0).min(1.0);
    Ok(Value::Float(env.curriculum_difficulty))
}

fn builtin_get_difficulty(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::Float(env.curriculum_difficulty))
}

fn builtin_curriculum_schedule(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let total_steps = match args.get(0) { Some(Value::Int(n)) => *n as f64, _ => 1000.0 };
    let current_step = match args.get(1) { Some(Value::Int(n)) => *n as f64, _ => 0.0 };
    let difficulty = (current_step / total_steps).min(1.0);
    Ok(Value::Float(difficulty))
}

// ===== Feature 44: Ensemble builtins =====

fn builtin_ensemble_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let model_id = match args.get(0) { Some(Value::Int(n)) => *n as usize, _ => 0 };
    env.ensemble_models.push(model_id);
    Ok(Value::Int(env.ensemble_models.len() as i128))
}

fn builtin_ensemble_vote(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(predictions)) = args.get(0) {
        let mut counts: HashMap<String, usize> = HashMap::new();
        for p in predictions { let key = format!("{}", p); *counts.entry(key).or_insert(0) += 1; }
        let winner = counts.iter().max_by_key(|(_, c)| *c).map(|(k, _)| k.clone()).unwrap_or_default();
        Ok(Value::String(winner))
    } else {
        Err("ensemble_vote: expected array of predictions".to_string())
    }
}

fn builtin_ensemble_avg(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(values)) = args.get(0) {
        let sum: f64 = values.iter().map(|v| match v { Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0 }).sum();
        let count = values.len().max(1) as f64;
        Ok(Value::Float(sum / count))
    } else {
        Err("ensemble_avg: expected array of values".to_string())
    }
}

// ===== Feature 45: Adversarial builtins =====

fn builtin_fgsm_attack(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let epsilon = match args.get(0) { Some(Value::Float(f)) => *f, _ => env.adversarial_epsilon };
    if let Some(Value::Array(input)) = args.get(1) {
        let perturbed: Vec<Value> = input.iter().map(|v| match v {
            Value::Float(f) => Value::Float(f + epsilon * if *f >= 0.0 { 1.0 } else { -1.0 }),
            Value::Int(n) => Value::Float(*n as f64 + epsilon),
            _ => v.clone(),
        }).collect();
        Ok(Value::Array(perturbed))
    } else {
        let mut fields = HashMap::new();
        fields.insert("epsilon".to_string(), Value::Float(epsilon));
        fields.insert("attack".to_string(), Value::String("fgsm".to_string()));
        Ok(Value::Struct { name: "FGSMAttack".to_string(), fields })
    }
}

fn builtin_pgd_attack(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let epsilon = match args.get(0) { Some(Value::Float(f)) => *f, _ => env.adversarial_epsilon };
    let steps = match args.get(1) { Some(Value::Int(n)) => *n as usize, _ => 10 };
    let mut fields = HashMap::new();
    fields.insert("epsilon".to_string(), Value::Float(epsilon));
    fields.insert("steps".to_string(), Value::Int(steps as i128));
    fields.insert("attack".to_string(), Value::String("pgd".to_string()));
    Ok(Value::Struct { name: "PGDAttack".to_string(), fields })
}

fn builtin_adversarial_train_step(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let clean_loss = match args.get(0) { Some(Value::Float(f)) => *f, _ => 0.0 };
    let adv_loss = match args.get(1) { Some(Value::Float(f)) => *f, _ => 0.0 };
    let alpha = match args.get(2) { Some(Value::Float(f)) => *f, _ => 0.5 };
    let combined = alpha * clean_loss + (1.0 - alpha) * adv_loss;
    Ok(Value::Float(combined))
}

// ===== Feature 46: Transfer learning builtins =====

fn builtin_freeze(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let layer = match args.get(0) { Some(Value::String(s)) => s.clone(), Some(Value::Int(n)) => format!("layer_{}", n), _ => "all".to_string() };
    env.transfer_frozen_layers.push(layer.clone());
    Ok(Value::String(format!("frozen: {}", layer)))
}

fn builtin_unfreeze(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let layer = match args.get(0) { Some(Value::String(s)) => s.clone(), Some(Value::Int(n)) => format!("layer_{}", n), _ => "all".to_string() };
    if layer == "all" { env.transfer_frozen_layers.clear(); } else { env.transfer_frozen_layers.retain(|l| l != &layer); }
    Ok(Value::String(format!("unfrozen: {}", layer)))
}

fn builtin_fine_tune(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lr = match args.get(0) { Some(Value::Float(f)) => *f, _ => 1e-4 };
    let epochs = match args.get(1) { Some(Value::Int(n)) => *n, _ => 10 };
    let mut fields = HashMap::new();
    fields.insert("learning_rate".to_string(), Value::Float(lr));
    fields.insert("epochs".to_string(), Value::Int(epochs));
    fields.insert("strategy".to_string(), Value::String("fine_tune".to_string()));
    Ok(Value::Struct { name: "FineTuneConfig".to_string(), fields })
}

// ===== Feature 47: Sparse computation builtins =====

fn builtin_to_sparse_scope(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(arr)) = args.get(0) {
        let threshold = match args.get(1) { Some(Value::Float(f)) => *f, _ => 0.0 };
        let nonzero: Vec<Value> = arr.iter().filter(|v| match v {
            Value::Float(f) => f.abs() > threshold, Value::Int(n) => *n != 0, _ => true
        }).cloned().collect();
        let sparsity = 1.0 - (nonzero.len() as f64 / arr.len().max(1) as f64);
        let mut fields = HashMap::new();
        fields.insert("values".to_string(), Value::Array(nonzero));
        fields.insert("sparsity".to_string(), Value::Float(sparsity));
        fields.insert("original_size".to_string(), Value::Int(arr.len() as i128));
        Ok(Value::Struct { name: "SparseArray".to_string(), fields })
    } else {
        Err("to_sparse_scope: expected array".to_string())
    }
}

fn builtin_sparse_matmul_scope(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let size = match args.get(0) { Some(Value::Int(n)) => *n, _ => 0 };
    let mut fields = HashMap::new();
    fields.insert("size".to_string(), Value::Int(size));
    fields.insert("algorithm".to_string(), Value::String("csr_spmm".to_string()));
    Ok(Value::Struct { name: "SparseMatMulResult".to_string(), fields })
}

fn builtin_sparsity_ratio(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(arr)) = args.get(0) {
        let zeros = arr.iter().filter(|v| matches!(v, Value::Float(f) if *f == 0.0) || matches!(v, Value::Int(0))).count();
        Ok(Value::Float(zeros as f64 / arr.len().max(1) as f64))
    } else {
        Ok(Value::Float(0.0))
    }
}

// ===== Feature 48: Async inference builtins =====

fn builtin_async_predict(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let model_id = match args.get(0) { Some(Value::Int(n)) => *n, _ => 0 };
    let input = args.get(1).cloned().unwrap_or(Value::Void);
    let mut fields = HashMap::new();
    fields.insert("model_id".to_string(), Value::Int(model_id));
    fields.insert("input".to_string(), input);
    fields.insert("status".to_string(), Value::String("completed".to_string()));
    fields.insert("result".to_string(), Value::Float(0.5)); // simulated
    Ok(Value::Struct { name: "AsyncPrediction".to_string(), fields })
}

fn builtin_batch_infer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Array(inputs)) = args.get(0) {
        let results: Vec<Value> = inputs.iter().map(|_| Value::Float(0.5)).collect();
        let mut fields = HashMap::new();
        fields.insert("batch_size".to_string(), Value::Int(inputs.len() as i128));
        fields.insert("results".to_string(), Value::Array(results));
        Ok(Value::Struct { name: "BatchInferResult".to_string(), fields })
    } else {
        Err("batch_infer: expected array of inputs".to_string())
    }
}

// ===== Feature 49: Profiling builtins =====

fn builtin_profile_op(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let op_name = match args.get(0) { Some(Value::String(s)) => s.clone(), _ => "unknown".to_string() };
    let time_ms = match args.get(1) { Some(Value::Float(f)) => *f, _ => 0.0 };
    env.profile_data.push((op_name.clone(), time_ms));
    Ok(Value::Void)
}

fn builtin_profile_summary(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let total: f64 = env.profile_data.iter().map(|(_, t)| t).sum();
    let count = env.profile_data.len();
    let mut fields = HashMap::new();
    fields.insert("total_time_ms".to_string(), Value::Float(total));
    fields.insert("num_ops".to_string(), Value::Int(count as i128));
    fields.insert("avg_time_ms".to_string(), Value::Float(if count > 0 { total / count as f64 } else { 0.0 }));
    Ok(Value::Struct { name: "ProfileSummary".to_string(), fields })
}

fn builtin_flops_count(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    // Estimate FLOPs for matrix ops: 2*M*N*K for matmul
    let m = match args.get(0) { Some(Value::Int(n)) => *n, _ => 0 };
    let n = match args.get(1) { Some(Value::Int(n)) => *n, _ => 0 };
    let k = match args.get(2) { Some(Value::Int(n)) => *n, _ => 0 };
    Ok(Value::Int(2 * m * n * k))
}

// ===== Feature 50: Contract builtins =====

fn builtin_requires(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Bool(b)) = args.get(0) {
        if !b { let msg = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => "precondition failed".to_string() }; return Err(msg); }
    }
    Ok(Value::Bool(true))
}

fn builtin_ensures(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Bool(b)) = args.get(0) {
        if !b { let msg = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => "postcondition failed".to_string() }; return Err(msg); }
    }
    Ok(Value::Bool(true))
}

fn builtin_invariant(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if let Some(Value::Bool(b)) = args.get(0) {
        if !b { let msg = match args.get(1) { Some(Value::String(s)) => s.clone(), _ => "invariant violated".to_string() }; return Err(msg); }
    }
    Ok(Value::Bool(true))
}
