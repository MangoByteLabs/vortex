//! Vortex Intermediate Representation (VIR)
//!
//! An SSA-form IR designed specifically for AI/tensor workloads. VIR is
//! tensor-aware, self-contained (no external dependencies), and suitable
//! for lowering to PTX, LLVM IR, or native code.

use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::interpreter::{Env, FnDef, Value};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Every SSA value is identified by a unique integer.
pub type VirId = usize;

/// Reduction operation for `Reduce` instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

impl fmt::Display for ReduceOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ReduceOp::Sum => write!(f, "sum"),
            ReduceOp::Max => write!(f, "max"),
            ReduceOp::Min => write!(f, "min"),
            ReduceOp::Mean => write!(f, "mean"),
        }
    }
}

/// Fused activation for FusedLinear operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FusedActivation {
    None,
    Relu,
    Gelu,
    Silu,
    Tanh,
    Sigmoid,
}

impl fmt::Display for FusedActivation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FusedActivation::None => write!(f, "none"),
            FusedActivation::Relu => write!(f, "relu"),
            FusedActivation::Gelu => write!(f, "gelu"),
            FusedActivation::Silu => write!(f, "silu"),
            FusedActivation::Tanh => write!(f, "tanh"),
            FusedActivation::Sigmoid => write!(f, "sigmoid"),
        }
    }
}

/// Quantization scheme for compressed tensor operations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantScheme {
    /// Symmetric: value = quantized * scale
    Symmetric,
    /// Asymmetric: value = (quantized - zero_point) * scale
    Asymmetric,
    /// Group quantization: separate scale per group of N elements
    Group(usize),
}

impl fmt::Display for QuantScheme {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QuantScheme::Symmetric => write!(f, "symmetric"),
            QuantScheme::Asymmetric => write!(f, "asymmetric"),
            QuantScheme::Group(n) => write!(f, "group({})", n),
        }
    }
}

/// A dimension in a tensor shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum VirDim {
    /// Compile-time known size.
    Fixed(usize),
    /// Symbolic (named) dimension — resolved at specialization time.
    Symbolic(String),
    /// Fully dynamic — only known at runtime.
    Dynamic,
}

impl fmt::Display for VirDim {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VirDim::Fixed(n) => write!(f, "{}", n),
            VirDim::Symbolic(s) => write!(f, "{}", s),
            VirDim::Dynamic => write!(f, "?"),
        }
    }
}

/// VIR type system — scalar, pointer, aggregate, tensor, and function types.
#[derive(Debug, Clone, PartialEq)]
pub enum VirType {
    Void,
    Bool,
    I4,  // 4-bit integer for quantized models
    I8,
    I16,
    I32,
    I64,
    I128,
    F16,
    F32,
    F64,
    Ptr(Box<VirType>),
    Array(Box<VirType>, usize),
    Tensor {
        elem: Box<VirType>,
        shape: Vec<VirDim>,
    },
    Fn {
        params: Vec<VirType>,
        ret: Box<VirType>,
    },
    Struct(String, Vec<(String, VirType)>),
}

impl VirType {
    /// Returns true when the type is a floating point type.
    pub fn is_float(&self) -> bool {
        matches!(self, VirType::F16 | VirType::F32 | VirType::F64)
    }

    /// Returns true when the type is an integer type.
    pub fn is_int(&self) -> bool {
        matches!(
            self,
            VirType::I4 | VirType::I8 | VirType::I16 | VirType::I32 | VirType::I64 | VirType::I128
        )
    }

    /// Returns true when the type is a tensor.
    pub fn is_tensor(&self) -> bool {
        matches!(self, VirType::Tensor { .. })
    }

    /// Tensor element type (or None).
    pub fn tensor_elem(&self) -> Option<&VirType> {
        match self {
            VirType::Tensor { elem, .. } => Some(elem),
            _ => None,
        }
    }
}

impl fmt::Display for VirType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VirType::Void => write!(f, "void"),
            VirType::Bool => write!(f, "bool"),
            VirType::I4 => write!(f, "i4"),
            VirType::I8 => write!(f, "i8"),
            VirType::I16 => write!(f, "i16"),
            VirType::I32 => write!(f, "i32"),
            VirType::I64 => write!(f, "i64"),
            VirType::I128 => write!(f, "i128"),
            VirType::F16 => write!(f, "f16"),
            VirType::F32 => write!(f, "f32"),
            VirType::F64 => write!(f, "f64"),
            VirType::Ptr(inner) => write!(f, "*{}", inner),
            VirType::Array(elem, n) => write!(f, "[{}; {}]", elem, n),
            VirType::Tensor { elem, shape } => {
                write!(f, "tensor<{}; [", elem)?;
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", d)?;
                }
                write!(f, "]>")
            }
            VirType::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            VirType::Struct(name, _) => write!(f, "%{}", name),
        }
    }
}

// ---------------------------------------------------------------------------
// SSA instructions
// ---------------------------------------------------------------------------

/// A single SSA instruction.
#[derive(Debug, Clone)]
pub struct VirInst {
    pub id: VirId,
    pub kind: VirInstKind,
    pub ty: VirType,
}

/// The payload of an SSA instruction.
#[derive(Debug, Clone)]
pub enum VirInstKind {
    // -- Constants --
    ConstInt(i128),
    ConstFloat(f64),
    ConstBool(bool),

    // -- Arithmetic --
    Add(VirId, VirId),
    Sub(VirId, VirId),
    Mul(VirId, VirId),
    Div(VirId, VirId),
    Mod(VirId, VirId),
    Neg(VirId),

    // -- Comparison --
    Eq(VirId, VirId),
    Ne(VirId, VirId),
    Lt(VirId, VirId),
    Le(VirId, VirId),
    Gt(VirId, VirId),
    Ge(VirId, VirId),

    // -- Logic --
    And(VirId, VirId),
    Or(VirId, VirId),
    Not(VirId),

    // -- Bitwise --
    BitAnd(VirId, VirId),
    BitOr(VirId, VirId),
    BitXor(VirId, VirId),
    Shl(VirId, VirId),
    Shr(VirId, VirId),
    BitNot(VirId),

    // -- Memory --
    Alloca(VirType),
    Load(VirId),
    Store(VirId, VirId), // ptr, value
    GetElementPtr(VirId, Vec<VirId>),

    // -- Struct --
    StructCreate(String, Vec<(String, VirId)>),
    StructGet(VirId, String),
    StructSet(VirId, String, VirId),

    // -- Tensor operations (native to VIR) --
    TensorCreate {
        data: VirId,
        shape: Vec<VirId>,
        dtype: VirType,
    },
    TensorLoad(VirId, Vec<VirId>),
    TensorStore(VirId, Vec<VirId>, VirId),
    TensorShape(VirId, usize),
    TensorReshape(VirId, Vec<VirId>),
    TensorSlice {
        tensor: VirId,
        dim: usize,
        start: VirId,
        end: VirId,
    },

    // -- High-level tensor ops (for fusion) --
    MatMul(VirId, VirId),
    Conv2d {
        input: VirId,
        weight: VirId,
        stride: [usize; 2],
        padding: [usize; 2],
    },
    Reduce {
        tensor: VirId,
        op: ReduceOp,
        axis: usize,
    },
    Broadcast(VirId, Vec<VirId>),
    Transpose(VirId, Vec<usize>),

    // -- Activation functions (fused for efficiency) --
    Relu(VirId),
    Sigmoid(VirId),
    Tanh(VirId),
    Gelu(VirId),
    Softmax(VirId, usize),
    LayerNorm { input: VirId, gamma: VirId, beta: VirId, eps: f64 },
    RMSNorm { input: VirId, gamma: VirId, eps: f64 },

    // -- Fused high-performance operations --
    /// Flash Attention: O(N) memory instead of O(N^2). Tiled, fused.
    FlashAttention {
        q: VirId, k: VirId, v: VirId,
        mask: Option<VirId>,
        scale: f64,
        causal: bool,
    },
    /// Fused matmul + bias + activation in one kernel launch
    FusedLinear {
        input: VirId, weight: VirId, bias: Option<VirId>,
        activation: FusedActivation,
    },
    /// Fused multi-head attention (Q/K/V projection + attention + output projection)
    FusedMHA {
        input: VirId, wq: VirId, wk: VirId, wv: VirId, wo: VirId,
        num_heads: usize, head_dim: usize, causal: bool,
    },
    /// Speculative decoding: draft with small model, verify with large model
    SpecDecode {
        draft_logits: VirId, target_logits: VirId,
        draft_tokens: VirId, temperature: f64,
    },
    /// Quantize tensor to lower precision
    Quantize { input: VirId, target_dtype: VirType, scheme: QuantScheme },
    /// Dequantize tensor back to float
    Dequantize { input: VirId, scale: VirId, zero_point: Option<VirId> },
    /// Quantized matmul (INT4/INT8 with accumulation in FP32)
    QMatMul { a: VirId, b: VirId, a_scale: VirId, b_scale: VirId },

    // -- Control flow --
    Phi(Vec<(VirId, usize)>),
    Call(String, Vec<VirId>),

    // -- Cast --
    Cast(VirId, VirType),

    // -- Function parameter --
    Param(usize),
}

impl VirInstKind {
    /// Return the set of VirIds that this instruction reads.
    pub fn uses(&self) -> Vec<VirId> {
        match self {
            VirInstKind::ConstInt(_)
            | VirInstKind::ConstFloat(_)
            | VirInstKind::ConstBool(_)
            | VirInstKind::Alloca(_)
            | VirInstKind::Param(_) => vec![],

            VirInstKind::Add(a, b)
            | VirInstKind::Sub(a, b)
            | VirInstKind::Mul(a, b)
            | VirInstKind::Div(a, b)
            | VirInstKind::Mod(a, b)
            | VirInstKind::Eq(a, b)
            | VirInstKind::Ne(a, b)
            | VirInstKind::Lt(a, b)
            | VirInstKind::Le(a, b)
            | VirInstKind::Gt(a, b)
            | VirInstKind::Ge(a, b)
            | VirInstKind::And(a, b)
            | VirInstKind::Or(a, b)
            | VirInstKind::BitAnd(a, b)
            | VirInstKind::BitOr(a, b)
            | VirInstKind::BitXor(a, b)
            | VirInstKind::Shl(a, b)
            | VirInstKind::Shr(a, b)
            | VirInstKind::MatMul(a, b) => vec![*a, *b],

            VirInstKind::Store(p, v) => vec![*p, *v],

            VirInstKind::Neg(a)
            | VirInstKind::Not(a)
            | VirInstKind::BitNot(a)
            | VirInstKind::Load(a)
            | VirInstKind::Relu(a)
            | VirInstKind::Sigmoid(a)
            | VirInstKind::Tanh(a)
            | VirInstKind::Gelu(a) => vec![*a],

            VirInstKind::Softmax(a, _) => vec![*a],
            VirInstKind::TensorShape(a, _) => vec![*a],
            VirInstKind::Cast(a, _) => vec![*a],

            VirInstKind::GetElementPtr(base, idxs) => {
                let mut v = vec![*base];
                v.extend(idxs);
                v
            }
            VirInstKind::StructCreate(_, fields) => fields.iter().map(|(_, id)| *id).collect(),
            VirInstKind::StructGet(s, _) => vec![*s],
            VirInstKind::StructSet(s, _, v) => vec![*s, *v],

            VirInstKind::TensorCreate { data, shape, .. } => {
                let mut v = vec![*data];
                v.extend(shape);
                v
            }
            VirInstKind::TensorLoad(t, idxs) => {
                let mut v = vec![*t];
                v.extend(idxs);
                v
            }
            VirInstKind::TensorStore(t, idxs, val) => {
                let mut v = vec![*t];
                v.extend(idxs);
                v.push(*val);
                v
            }
            VirInstKind::TensorReshape(t, dims) => {
                let mut v = vec![*t];
                v.extend(dims);
                v
            }
            VirInstKind::TensorSlice {
                tensor,
                start,
                end,
                ..
            } => vec![*tensor, *start, *end],

            VirInstKind::Conv2d { input, weight, .. } => vec![*input, *weight],
            VirInstKind::Reduce { tensor, .. } => vec![*tensor],
            VirInstKind::Broadcast(t, dims) => {
                let mut v = vec![*t];
                v.extend(dims);
                v
            }
            VirInstKind::Transpose(t, _) => vec![*t],

            VirInstKind::LayerNorm { input, gamma, beta, .. } => vec![*input, *gamma, *beta],
            VirInstKind::RMSNorm { input, gamma, .. } => vec![*input, *gamma],
            VirInstKind::FlashAttention { q, k, v, mask, .. } => {
                let mut r = vec![*q, *k, *v];
                if let Some(m) = mask { r.push(*m); }
                r
            }
            VirInstKind::FusedLinear { input, weight, bias, .. } => {
                let mut r = vec![*input, *weight];
                if let Some(b) = bias { r.push(*b); }
                r
            }
            VirInstKind::FusedMHA { input, wq, wk, wv, wo, .. } => vec![*input, *wq, *wk, *wv, *wo],
            VirInstKind::SpecDecode { draft_logits, target_logits, draft_tokens, .. } => {
                vec![*draft_logits, *target_logits, *draft_tokens]
            }
            VirInstKind::Quantize { input, .. } => vec![*input],
            VirInstKind::Dequantize { input, scale, zero_point } => {
                let mut r = vec![*input, *scale];
                if let Some(zp) = zero_point { r.push(*zp); }
                r
            }
            VirInstKind::QMatMul { a, b, a_scale, b_scale } => vec![*a, *b, *a_scale, *b_scale],

            VirInstKind::Phi(pairs) => pairs.iter().map(|(id, _)| *id).collect(),
            VirInstKind::Call(_, args) => args.clone(),
        }
    }
}

impl fmt::Display for VirInst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "  %{}: {} = ", self.id, self.ty)?;
        match &self.kind {
            VirInstKind::ConstInt(v) => write!(f, "const {}", v),
            VirInstKind::ConstFloat(v) => write!(f, "const {:.6}", v),
            VirInstKind::ConstBool(v) => write!(f, "const {}", v),
            VirInstKind::Add(a, b) => write!(f, "add %{}, %{}", a, b),
            VirInstKind::Sub(a, b) => write!(f, "sub %{}, %{}", a, b),
            VirInstKind::Mul(a, b) => write!(f, "mul %{}, %{}", a, b),
            VirInstKind::Div(a, b) => write!(f, "div %{}, %{}", a, b),
            VirInstKind::Mod(a, b) => write!(f, "mod %{}, %{}", a, b),
            VirInstKind::Neg(a) => write!(f, "neg %{}", a),
            VirInstKind::Eq(a, b) => write!(f, "eq %{}, %{}", a, b),
            VirInstKind::Ne(a, b) => write!(f, "ne %{}, %{}", a, b),
            VirInstKind::Lt(a, b) => write!(f, "lt %{}, %{}", a, b),
            VirInstKind::Le(a, b) => write!(f, "le %{}, %{}", a, b),
            VirInstKind::Gt(a, b) => write!(f, "gt %{}, %{}", a, b),
            VirInstKind::Ge(a, b) => write!(f, "ge %{}, %{}", a, b),
            VirInstKind::And(a, b) => write!(f, "and %{}, %{}", a, b),
            VirInstKind::Or(a, b) => write!(f, "or %{}, %{}", a, b),
            VirInstKind::Not(a) => write!(f, "not %{}", a),
            VirInstKind::BitAnd(a, b) => write!(f, "bitand %{}, %{}", a, b),
            VirInstKind::BitOr(a, b) => write!(f, "bitor %{}, %{}", a, b),
            VirInstKind::BitXor(a, b) => write!(f, "bitxor %{}, %{}", a, b),
            VirInstKind::Shl(a, b) => write!(f, "shl %{}, %{}", a, b),
            VirInstKind::Shr(a, b) => write!(f, "shr %{}, %{}", a, b),
            VirInstKind::BitNot(a) => write!(f, "bitnot %{}", a),
            VirInstKind::Alloca(ty) => write!(f, "alloca {}", ty),
            VirInstKind::Load(p) => write!(f, "load %{}", p),
            VirInstKind::Store(p, v) => write!(f, "store %{}, %{}", p, v),
            VirInstKind::GetElementPtr(base, idxs) => {
                write!(f, "gep %{}", base)?;
                for idx in idxs {
                    write!(f, ", %{}", idx)?;
                }
                Ok(())
            }
            VirInstKind::StructCreate(name, fields) => {
                write!(f, "struct.create %{} {{", name)?;
                for (i, (fname, fid)) in fields.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " {}: %{}", fname, fid)?;
                }
                write!(f, " }}")
            }
            VirInstKind::StructGet(s, field) => write!(f, "struct.get %{}, .{}", s, field),
            VirInstKind::StructSet(s, field, v) => {
                write!(f, "struct.set %{}, .{}, %{}", s, field, v)
            }
            VirInstKind::TensorCreate { data, shape, dtype } => {
                write!(f, "tensor.create %{}, shape=[", data)?;
                for (i, s) in shape.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", s)?;
                }
                write!(f, "], dtype={}", dtype)
            }
            VirInstKind::TensorLoad(t, idxs) => {
                write!(f, "tensor.load %{}[", t)?;
                for (i, idx) in idxs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", idx)?;
                }
                write!(f, "]")
            }
            VirInstKind::TensorStore(t, idxs, v) => {
                write!(f, "tensor.store %{}[", t)?;
                for (i, idx) in idxs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", idx)?;
                }
                write!(f, "] = %{}", v)
            }
            VirInstKind::TensorShape(t, dim) => write!(f, "tensor.shape %{}, dim={}", t, dim),
            VirInstKind::TensorReshape(t, dims) => {
                write!(f, "tensor.reshape %{}, [", t)?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", d)?;
                }
                write!(f, "]")
            }
            VirInstKind::TensorSlice {
                tensor,
                dim,
                start,
                end,
            } => write!(
                f,
                "tensor.slice %{}, dim={}, %{}..%{}",
                tensor, dim, start, end
            ),
            VirInstKind::MatMul(a, b) => write!(f, "matmul %{}, %{}", a, b),
            VirInstKind::Conv2d {
                input,
                weight,
                stride,
                padding,
            } => write!(
                f,
                "conv2d %{}, %{}, stride=[{}, {}], pad=[{}, {}]",
                input, weight, stride[0], stride[1], padding[0], padding[1]
            ),
            VirInstKind::Reduce { tensor, op, axis } => {
                write!(f, "reduce.{} %{}, axis={}", op, tensor, axis)
            }
            VirInstKind::Broadcast(t, dims) => {
                write!(f, "broadcast %{}, [", t)?;
                for (i, d) in dims.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", d)?;
                }
                write!(f, "]")
            }
            VirInstKind::Transpose(t, perm) => {
                write!(f, "transpose %{}, [", t)?;
                for (i, p) in perm.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", p)?;
                }
                write!(f, "]")
            }
            VirInstKind::Relu(a) => write!(f, "relu %{}", a),
            VirInstKind::Sigmoid(a) => write!(f, "sigmoid %{}", a),
            VirInstKind::Tanh(a) => write!(f, "tanh %{}", a),
            VirInstKind::Gelu(a) => write!(f, "gelu %{}", a),
            VirInstKind::Softmax(a, axis) => write!(f, "softmax %{}, axis={}", a, axis),
            VirInstKind::LayerNorm { input, gamma, beta, eps } => {
                write!(f, "layernorm %{}, gamma=%{}, beta=%{}, eps={}", input, gamma, beta, eps)
            }
            VirInstKind::RMSNorm { input, gamma, eps } => {
                write!(f, "rmsnorm %{}, gamma=%{}, eps={}", input, gamma, eps)
            }
            VirInstKind::FlashAttention { q, k, v, mask, scale, causal } => {
                write!(f, "flash_attention q=%{}, k=%{}, v=%{}", q, k, v)?;
                if let Some(m) = mask { write!(f, ", mask=%{}", m)?; }
                write!(f, ", scale={}, causal={}", scale, causal)
            }
            VirInstKind::FusedLinear { input, weight, bias, activation } => {
                write!(f, "fused_linear %{} @ %{}", input, weight)?;
                if let Some(b) = bias { write!(f, " + %{}", b)?; }
                write!(f, ", act={}", activation)
            }
            VirInstKind::FusedMHA { input, wq, wk, wv, wo, num_heads, head_dim, causal } => {
                write!(f, "fused_mha %{}, wq=%{}, wk=%{}, wv=%{}, wo=%{}, heads={}, dim={}, causal={}",
                    input, wq, wk, wv, wo, num_heads, head_dim, causal)
            }
            VirInstKind::SpecDecode { draft_logits, target_logits, draft_tokens, temperature } => {
                write!(f, "spec_decode draft=%{}, target=%{}, tokens=%{}, temp={}",
                    draft_logits, target_logits, draft_tokens, temperature)
            }
            VirInstKind::Quantize { input, target_dtype, scheme } => {
                write!(f, "quantize %{} to {}, scheme={}", input, target_dtype, scheme)
            }
            VirInstKind::Dequantize { input, scale, zero_point } => {
                write!(f, "dequantize %{}, scale=%{}", input, scale)?;
                if let Some(zp) = zero_point { write!(f, ", zp=%{}", zp)?; }
                Ok(())
            }
            VirInstKind::QMatMul { a, b, a_scale, b_scale } => {
                write!(f, "qmatmul %{} @ %{}, scales=(%{}, %{})", a, b, a_scale, b_scale)
            }
            VirInstKind::Phi(pairs) => {
                write!(f, "phi")?;
                for (i, (val, bb)) in pairs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ",")?;
                    }
                    write!(f, " [%{}, bb{}]", val, bb)?;
                }
                Ok(())
            }
            VirInstKind::Call(name, args) => {
                write!(f, "call @{}(", name)?;
                for (i, a) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "%{}", a)?;
                }
                write!(f, ")")
            }
            VirInstKind::Cast(v, ty) => write!(f, "cast %{} to {}", v, ty),
            VirInstKind::Param(idx) => write!(f, "param {}", idx),
        }
    }
}

// ---------------------------------------------------------------------------
// Basic blocks and terminators
// ---------------------------------------------------------------------------

/// A basic block — a straight-line sequence of instructions ending with a
/// terminator that transfers control to another block (or returns).
#[derive(Debug, Clone)]
pub struct BasicBlock {
    pub id: usize,
    pub label: String,
    pub insts: Vec<VirInst>,
    pub terminator: Terminator,
}

impl fmt::Display for BasicBlock {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "bb{} ({}):", self.id, self.label)?;
        for inst in &self.insts {
            writeln!(f, "{}", inst)?;
        }
        write!(f, "  {}", self.terminator)
    }
}

/// A block terminator — exactly one of these ends every basic block.
#[derive(Debug, Clone)]
pub enum Terminator {
    Return(Option<VirId>),
    Branch(usize),
    CondBranch {
        cond: VirId,
        true_bb: usize,
        false_bb: usize,
    },
    Unreachable,
}

impl fmt::Display for Terminator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Terminator::Return(None) => write!(f, "ret void"),
            Terminator::Return(Some(id)) => write!(f, "ret %{}", id),
            Terminator::Branch(bb) => write!(f, "br bb{}", bb),
            Terminator::CondBranch {
                cond,
                true_bb,
                false_bb,
            } => write!(f, "br %{}, bb{}, bb{}", cond, true_bb, false_bb),
            Terminator::Unreachable => write!(f, "unreachable"),
        }
    }
}

// ---------------------------------------------------------------------------
// Functions
// ---------------------------------------------------------------------------

/// A VIR function — a list of basic blocks in SSA form.
#[derive(Debug, Clone)]
pub struct VirFunction {
    pub name: String,
    pub params: Vec<(String, VirType)>,
    pub ret_type: VirType,
    pub blocks: Vec<BasicBlock>,
    pub is_kernel: bool,
    next_id: usize,
    current_block: usize,
}

impl VirFunction {
    pub fn new(name: &str, params: Vec<(&str, VirType)>, ret: VirType) -> Self {
        let params: Vec<(String, VirType)> =
            params.into_iter().map(|(n, t)| (n.to_string(), t)).collect();
        let entry = BasicBlock {
            id: 0,
            label: "entry".to_string(),
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
        };
        let mut func = VirFunction {
            name: name.to_string(),
            params: params.clone(),
            ret_type: ret,
            blocks: vec![entry],
            is_kernel: false,
            next_id: 0,
            current_block: 0,
        };
        // Emit Param instructions for each parameter in the entry block.
        for (i, (_, ty)) in params.iter().enumerate() {
            func.emit(VirInstKind::Param(i), ty.clone());
        }
        func
    }

    /// Create a new basic block and return its id.
    pub fn add_block(&mut self, label: &str) -> usize {
        let id = self.blocks.len();
        self.blocks.push(BasicBlock {
            id,
            label: label.to_string(),
            insts: Vec::new(),
            terminator: Terminator::Unreachable,
        });
        id
    }

    /// Set the insertion point to the given block.
    pub fn set_block(&mut self, block_id: usize) {
        assert!(block_id < self.blocks.len(), "invalid block id");
        self.current_block = block_id;
    }

    /// Emit an instruction in the current block and return its SSA id.
    pub fn emit(&mut self, kind: VirInstKind, ty: VirType) -> VirId {
        let id = self.next_id;
        self.next_id += 1;
        let inst = VirInst {
            id,
            kind,
            ty,
        };
        self.blocks[self.current_block].insts.push(inst);
        id
    }

    // -- Convenience helpers --------------------------------------------------

    pub fn const_i64(&mut self, val: i64) -> VirId {
        self.emit(VirInstKind::ConstInt(val as i128), VirType::I64)
    }

    pub fn const_f64(&mut self, val: f64) -> VirId {
        self.emit(VirInstKind::ConstFloat(val), VirType::F64)
    }

    pub fn const_bool(&mut self, val: bool) -> VirId {
        self.emit(VirInstKind::ConstBool(val), VirType::Bool)
    }

    pub fn add(&mut self, a: VirId, b: VirId) -> VirId {
        self.emit(VirInstKind::Add(a, b), VirType::I64)
    }

    pub fn sub(&mut self, a: VirId, b: VirId) -> VirId {
        self.emit(VirInstKind::Sub(a, b), VirType::I64)
    }

    pub fn mul(&mut self, a: VirId, b: VirId) -> VirId {
        self.emit(VirInstKind::Mul(a, b), VirType::I64)
    }

    pub fn matmul(&mut self, a: VirId, b: VirId, out_shape: Vec<VirDim>) -> VirId {
        let ty = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: out_shape,
        };
        self.emit(VirInstKind::MatMul(a, b), ty)
    }

    pub fn call(&mut self, name: &str, args: Vec<VirId>, ret_ty: VirType) -> VirId {
        self.emit(VirInstKind::Call(name.to_string(), args), ret_ty)
    }

    pub fn ret(&mut self, val: Option<VirId>) {
        self.blocks[self.current_block].terminator = Terminator::Return(val);
    }

    pub fn br(&mut self, target: usize) {
        self.blocks[self.current_block].terminator = Terminator::Branch(target);
    }

    pub fn cond_br(&mut self, cond: VirId, t: usize, f: usize) {
        self.blocks[self.current_block].terminator = Terminator::CondBranch {
            cond,
            true_bb: t,
            false_bb: f,
        };
    }

    /// Lookup the instruction that produced a given VirId.
    pub fn get_inst(&self, id: VirId) -> Option<&VirInst> {
        for bb in &self.blocks {
            for inst in &bb.insts {
                if inst.id == id {
                    return Some(inst);
                }
            }
        }
        None
    }

    /// Collect all defined VirIds.
    fn defined_ids(&self) -> HashSet<VirId> {
        let mut ids = HashSet::new();
        for bb in &self.blocks {
            for inst in &bb.insts {
                ids.insert(inst.id);
            }
        }
        ids
    }

    /// Collect the set of VirIds that are *used* by instructions or terminators.
    fn used_ids(&self) -> HashSet<VirId> {
        let mut used = HashSet::new();
        for bb in &self.blocks {
            for inst in &bb.insts {
                for u in inst.kind.uses() {
                    used.insert(u);
                }
            }
            match &bb.terminator {
                Terminator::Return(Some(id)) => {
                    used.insert(*id);
                }
                Terminator::CondBranch { cond, .. } => {
                    used.insert(*cond);
                }
                _ => {}
            }
        }
        used
    }
}

impl fmt::Display for VirFunction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_kernel {
            write!(f, "kernel ")?;
        }
        write!(f, "fn @{}(", self.name)?;
        for (i, (name, ty)) in self.params.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}: {}", name, ty)?;
        }
        writeln!(f, ") -> {} {{", self.ret_type)?;
        for bb in &self.blocks {
            writeln!(f, "{}", bb)?;
        }
        write!(f, "}}")
    }
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------

/// A VIR module — the top-level container analogous to a compilation unit.
#[derive(Debug, Clone)]
pub struct VirModule {
    pub name: String,
    pub functions: Vec<VirFunction>,
    pub structs: Vec<(String, Vec<(String, VirType)>)>,
    pub globals: Vec<(String, VirType, Option<VirId>)>,
}

impl VirModule {
    pub fn new(name: &str) -> Self {
        VirModule {
            name: name.to_string(),
            functions: Vec::new(),
            structs: Vec::new(),
            globals: Vec::new(),
        }
    }

    /// Add a function to the module.
    pub fn add_function(&mut self, func: VirFunction) {
        self.functions.push(func);
    }

    /// Pretty-print the entire module to a human-readable string.
    pub fn to_text(&self) -> String {
        let mut s = String::new();
        s.push_str(&format!("; VIR Module: {}\n\n", self.name));

        // Struct declarations
        for (name, fields) in &self.structs {
            s.push_str(&format!("struct %{} {{\n", name));
            for (fname, fty) in fields {
                s.push_str(&format!("  {}: {},\n", fname, fty));
            }
            s.push_str("}\n\n");
        }

        // Globals
        for (name, ty, init) in &self.globals {
            match init {
                Some(id) => s.push_str(&format!("global @{}: {} = %{}\n", name, ty, id)),
                None => s.push_str(&format!("global @{}: {}\n", name, ty)),
            }
        }
        if !self.globals.is_empty() {
            s.push('\n');
        }

        // Functions
        for func in &self.functions {
            s.push_str(&format!("{}\n\n", func));
        }

        s
    }
}

impl fmt::Display for VirModule {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.to_text())
    }
}

// ---------------------------------------------------------------------------
// AST to VIR lowering
// ---------------------------------------------------------------------------

/// Lowers a Vortex AST (`crate::ast::Program`) into a `VirModule`.
pub struct VirLowering {
    module: VirModule,
    current_fn: Option<usize>,
    var_map: HashMap<String, VirId>,
}

impl VirLowering {
    pub fn new(module_name: &str) -> Self {
        VirLowering {
            module: VirModule::new(module_name),
            current_fn: None,
            var_map: HashMap::new(),
        }
    }

    /// Lower an entire Vortex program.
    pub fn lower_program(&mut self, program: &crate::ast::Program) -> Result<VirModule, String> {
        for item in &program.items {
            self.lower_item(item)?;
        }
        Ok(self.module.clone())
    }

    fn lower_item(&mut self, item: &crate::ast::Item) -> Result<(), String> {
        match &item.kind {
            crate::ast::ItemKind::Function(func) => self.lower_function(func, false),
            crate::ast::ItemKind::Kernel(kernel) => {
                // Convert Kernel -> function lowering with is_kernel=true
                let func = crate::ast::Function {
                    name: kernel.name.clone(),
                    generics: kernel.generics.clone(),
                    params: kernel.params.clone(),
                    ret_type: kernel.ret_type.clone(),
                    where_clause: kernel.where_clause.clone(),
                    body: kernel.body.clone(),
                    annotations: kernel.annotations.clone(),
                };
                self.lower_function(&func, true)
            }
            crate::ast::ItemKind::Struct(sdef) => {
                let fields: Vec<(String, VirType)> = sdef
                    .fields
                    .iter()
                    .map(|f| (f.name.name.clone(), self.lower_type_expr(&f.ty)))
                    .collect();
                self.module.structs.push((sdef.name.name.clone(), fields));
                Ok(())
            }
            crate::ast::ItemKind::Const(cdecl) => {
                let ty = match &cdecl.ty {
                    Some(te) => self.lower_type_expr(te),
                    None => VirType::I64,
                };
                self.module
                    .globals
                    .push((cdecl.name.name.clone(), ty, None));
                Ok(())
            }
            // Skip other items for now
            _ => Ok(()),
        }
    }

    fn lower_function(
        &mut self,
        func: &crate::ast::Function,
        is_kernel: bool,
    ) -> Result<(), String> {
        let params: Vec<(&str, VirType)> = func
            .params
            .iter()
            .map(|p| (p.name.name.as_str(), self.lower_type_expr(&p.ty)))
            .collect();
        let ret = match &func.ret_type {
            Some(te) => self.lower_type_expr(te),
            None => VirType::Void,
        };

        let mut vir_fn = VirFunction::new(&func.name.name, params, ret);
        vir_fn.is_kernel = is_kernel;

        let fn_idx = self.module.functions.len();
        self.module.functions.push(vir_fn);
        self.current_fn = Some(fn_idx);

        // Map parameter names to their SSA ids.
        self.var_map.clear();
        for (i, p) in func.params.iter().enumerate() {
            self.var_map.insert(p.name.name.clone(), i);
        }

        // Lower body
        self.lower_block(&func.body)?;

        // If the last block has no explicit return, add one.
        let f = &mut self.module.functions[fn_idx];
        let last_bb = f.blocks.len() - 1;
        if matches!(f.blocks[last_bb].terminator, Terminator::Unreachable) {
            f.blocks[last_bb].terminator = Terminator::Return(None);
        }

        self.current_fn = None;
        Ok(())
    }

    fn cur_fn(&mut self) -> &mut VirFunction {
        let idx = self.current_fn.expect("no current function");
        &mut self.module.functions[idx]
    }

    fn lower_block(&mut self, block: &crate::ast::Block) -> Result<Option<VirId>, String> {
        for stmt in &block.stmts {
            self.lower_stmt(stmt)?;
        }
        if let Some(expr) = &block.expr {
            let id = self.lower_expr(expr)?;
            Ok(Some(id))
        } else {
            Ok(None)
        }
    }

    fn lower_stmt(&mut self, stmt: &crate::ast::Stmt) -> Result<(), String> {
        match &stmt.kind {
            crate::ast::StmtKind::Let { name, ty, value } | crate::ast::StmtKind::Var { name, ty, value } => {
                let val_id = self.lower_expr(value)?;
                self.var_map.insert(name.name.clone(), val_id);
                Ok(())
            }
            crate::ast::StmtKind::Return(opt_expr) => {
                let id = match opt_expr {
                    Some(e) => Some(self.lower_expr(e)?),
                    None => None,
                };
                self.cur_fn().ret(id);
                Ok(())
            }
            crate::ast::StmtKind::Expr(expr) => {
                self.lower_expr(expr)?;
                Ok(())
            }
            crate::ast::StmtKind::Assign { target, value, .. } => {
                let val_id = self.lower_expr(value)?;
                // Simple case: assign to identifier
                if let crate::ast::ExprKind::Ident(ident) = &target.kind {
                    self.var_map.insert(ident.name.clone(), val_id);
                }
                Ok(())
            }
            crate::ast::StmtKind::For { var, iter, body } => {
                // Lower as a simple counted loop for Range expressions
                let header_bb = self.cur_fn().add_block("for.header");
                let body_bb = self.cur_fn().add_block("for.body");
                let exit_bb = self.cur_fn().add_block("for.exit");

                // Lower the iterator expression
                if let crate::ast::ExprKind::Range { start, end } = &iter.kind {
                    let start_id = self.lower_expr(start)?;
                    let end_id = self.lower_expr(end)?;
                    self.cur_fn().br(header_bb);

                    // Header: phi + cond
                    self.cur_fn().set_block(header_bb);
                    let phi_id = self.cur_fn().emit(
                        VirInstKind::Phi(vec![(start_id, 0), (0, body_bb)]), // placeholder
                        VirType::I64,
                    );
                    self.var_map.insert(var.name.clone(), phi_id);
                    let cond = self.cur_fn().emit(VirInstKind::Lt(phi_id, end_id), VirType::Bool);
                    self.cur_fn().cond_br(cond, body_bb, exit_bb);

                    // Body
                    self.cur_fn().set_block(body_bb);
                    self.lower_block(body)?;
                    let one = self.cur_fn().const_i64(1);
                    let next = self.cur_fn().add(phi_id, one);
                    // Patch phi with correct increment id
                    let fn_ref = self.cur_fn();
                    for inst in &mut fn_ref.blocks[header_bb].insts {
                        if inst.id == phi_id {
                            if let VirInstKind::Phi(ref mut pairs) = inst.kind {
                                pairs[1].0 = next;
                            }
                        }
                    }
                    self.cur_fn().br(header_bb);

                    self.cur_fn().set_block(exit_bb);
                } else {
                    // Fallback: just lower body once
                    self.cur_fn().br(body_bb);
                    self.cur_fn().set_block(body_bb);
                    self.lower_block(body)?;
                    self.cur_fn().br(exit_bb);
                    self.cur_fn().set_block(exit_bb);
                }
                Ok(())
            }
            crate::ast::StmtKind::While { cond, body } => {
                let header_bb = self.cur_fn().add_block("while.header");
                let body_bb = self.cur_fn().add_block("while.body");
                let exit_bb = self.cur_fn().add_block("while.exit");

                self.cur_fn().br(header_bb);
                self.cur_fn().set_block(header_bb);
                let cond_id = self.lower_expr(cond)?;
                self.cur_fn().cond_br(cond_id, body_bb, exit_bb);

                self.cur_fn().set_block(body_bb);
                self.lower_block(body)?;
                self.cur_fn().br(header_bb);

                self.cur_fn().set_block(exit_bb);
                Ok(())
            }
            crate::ast::StmtKind::Loop { body } => {
                let loop_bb = self.cur_fn().add_block("loop.body");
                let exit_bb = self.cur_fn().add_block("loop.exit");

                self.cur_fn().br(loop_bb);
                self.cur_fn().set_block(loop_bb);
                self.lower_block(body)?;
                // If terminator is still unreachable, loop back
                let f = self.cur_fn();
                let cur = f.current_block;
                if matches!(f.blocks[cur].terminator, Terminator::Unreachable) {
                    f.blocks[cur].terminator = Terminator::Branch(loop_bb);
                }

                self.cur_fn().set_block(exit_bb);
                Ok(())
            }
            crate::ast::StmtKind::Break | crate::ast::StmtKind::Continue => {
                // In a real compiler these would manipulate loop exit/header blocks.
                // Stub for now.
                Ok(())
            }
            crate::ast::StmtKind::Dispatch { .. } => Ok(()),
        }
    }

    fn lower_expr(&mut self, expr: &crate::ast::Expr) -> Result<VirId, String> {
        match &expr.kind {
            crate::ast::ExprKind::IntLiteral(n) => {
                let id = self.cur_fn().emit(VirInstKind::ConstInt(*n as i128), VirType::I64);
                Ok(id)
            }
            crate::ast::ExprKind::FloatLiteral(v) => {
                let id = self.cur_fn().const_f64(*v);
                Ok(id)
            }
            crate::ast::ExprKind::BoolLiteral(v) => {
                let id = self.cur_fn().const_bool(*v);
                Ok(id)
            }
            crate::ast::ExprKind::StringLiteral(_) => {
                // Strings are not natively supported in VIR; emit as i64(0) placeholder
                let id = self.cur_fn().const_i64(0);
                Ok(id)
            }
            crate::ast::ExprKind::Ident(ident) => {
                match self.var_map.get(&ident.name) {
                    Some(id) => Ok(*id),
                    None => {
                        // Unknown variable — emit a zero constant as fallback
                        let id = self.cur_fn().const_i64(0);
                        Ok(id)
                    }
                }
            }
            crate::ast::ExprKind::Binary { lhs, op, rhs } => {
                let l = self.lower_expr(lhs)?;
                let r = self.lower_expr(rhs)?;
                let kind = match op {
                    crate::ast::BinOp::Add => VirInstKind::Add(l, r),
                    crate::ast::BinOp::Sub => VirInstKind::Sub(l, r),
                    crate::ast::BinOp::Mul | crate::ast::BinOp::ElemMul => VirInstKind::Mul(l, r),
                    crate::ast::BinOp::Div | crate::ast::BinOp::ElemDiv => VirInstKind::Div(l, r),
                    crate::ast::BinOp::Mod => VirInstKind::Mod(l, r),
                    crate::ast::BinOp::Pow => {
                        // No native pow; emit as a call
                        return Ok(self.cur_fn().call("__pow", vec![l, r], VirType::F64));
                    }
                    crate::ast::BinOp::Eq => VirInstKind::Eq(l, r),
                    crate::ast::BinOp::NotEq => VirInstKind::Ne(l, r),
                    crate::ast::BinOp::Lt => VirInstKind::Lt(l, r),
                    crate::ast::BinOp::Gt => VirInstKind::Gt(l, r),
                    crate::ast::BinOp::LtEq => VirInstKind::Le(l, r),
                    crate::ast::BinOp::GtEq => VirInstKind::Ge(l, r),
                    crate::ast::BinOp::And => VirInstKind::And(l, r),
                    crate::ast::BinOp::Or => VirInstKind::Or(l, r),
                    crate::ast::BinOp::BitAnd => VirInstKind::BitAnd(l, r),
                    crate::ast::BinOp::BitOr => VirInstKind::BitOr(l, r),
                    crate::ast::BinOp::BitXor => VirInstKind::BitXor(l, r),
                    crate::ast::BinOp::Shl => VirInstKind::Shl(l, r),
                    crate::ast::BinOp::Shr => VirInstKind::Shr(l, r),
                };
                let ty = match op {
                    crate::ast::BinOp::Eq
                    | crate::ast::BinOp::NotEq
                    | crate::ast::BinOp::Lt
                    | crate::ast::BinOp::Gt
                    | crate::ast::BinOp::LtEq
                    | crate::ast::BinOp::GtEq
                    | crate::ast::BinOp::And
                    | crate::ast::BinOp::Or => VirType::Bool,
                    _ => VirType::I64,
                };
                Ok(self.cur_fn().emit(kind, ty))
            }
            crate::ast::ExprKind::Unary { op, expr: inner } => {
                let a = self.lower_expr(inner)?;
                let (kind, ty) = match op {
                    crate::ast::UnaryOp::Neg => (VirInstKind::Neg(a), VirType::I64),
                    crate::ast::UnaryOp::Not => (VirInstKind::Not(a), VirType::Bool),
                    crate::ast::UnaryOp::BitNot => (VirInstKind::BitNot(a), VirType::I64),
                };
                Ok(self.cur_fn().emit(kind, ty))
            }
            crate::ast::ExprKind::Call { func, args } => {
                let arg_ids: Vec<VirId> = args
                    .iter()
                    .map(|a| self.lower_expr(a))
                    .collect::<Result<_, _>>()?;
                let name = match &func.kind {
                    crate::ast::ExprKind::Ident(id) => id.name.clone(),
                    _ => "__indirect_call".to_string(),
                };
                Ok(self.cur_fn().call(&name, arg_ids, VirType::I64))
            }
            crate::ast::ExprKind::MatMul { lhs, rhs } => {
                let l = self.lower_expr(lhs)?;
                let r = self.lower_expr(rhs)?;
                let ty = VirType::Tensor {
                    elem: Box::new(VirType::F32),
                    shape: vec![VirDim::Dynamic, VirDim::Dynamic],
                };
                Ok(self.cur_fn().emit(VirInstKind::MatMul(l, r), ty))
            }
            crate::ast::ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond_id = self.lower_expr(cond)?;
                let then_bb = self.cur_fn().add_block("if.then");
                let else_bb = self.cur_fn().add_block("if.else");
                let merge_bb = self.cur_fn().add_block("if.merge");

                self.cur_fn().cond_br(cond_id, then_bb, else_bb);

                // Then
                self.cur_fn().set_block(then_bb);
                let then_val = self.lower_block(then_block)?;
                let then_exit = self.cur_fn().current_block;
                self.cur_fn().br(merge_bb);

                // Else
                self.cur_fn().set_block(else_bb);
                let else_val = if let Some(eb) = else_block {
                    self.lower_block(eb)?
                } else {
                    None
                };
                let else_exit = self.cur_fn().current_block;
                self.cur_fn().br(merge_bb);

                // Merge
                self.cur_fn().set_block(merge_bb);
                match (then_val, else_val) {
                    (Some(tv), Some(ev)) => {
                        let phi = self.cur_fn().emit(
                            VirInstKind::Phi(vec![(tv, then_exit), (ev, else_exit)]),
                            VirType::I64,
                        );
                        Ok(phi)
                    }
                    (Some(tv), None) => Ok(tv),
                    _ => Ok(self.cur_fn().const_i64(0)),
                }
            }
            crate::ast::ExprKind::Block(block) => {
                match self.lower_block(block)? {
                    Some(id) => Ok(id),
                    None => Ok(self.cur_fn().const_i64(0)),
                }
            }
            crate::ast::ExprKind::Index { base, indices } => {
                let base_id = self.lower_expr(base)?;
                let idx_ids: Vec<VirId> = indices
                    .iter()
                    .map(|i| self.lower_expr(i))
                    .collect::<Result<_, _>>()?;
                Ok(self.cur_fn().emit(VirInstKind::TensorLoad(base_id, idx_ids), VirType::F32))
            }
            crate::ast::ExprKind::FieldAccess { base, field } => {
                let base_id = self.lower_expr(base)?;
                Ok(self.cur_fn().emit(
                    VirInstKind::StructGet(base_id, field.name.clone()),
                    VirType::I64,
                ))
            }
            crate::ast::ExprKind::Cast { expr: inner, ty } => {
                let a = self.lower_expr(inner)?;
                let target = self.lower_type_expr(ty);
                Ok(self.cur_fn().emit(VirInstKind::Cast(a, target.clone()), target))
            }
            crate::ast::ExprKind::ArrayLiteral(_)
            | crate::ast::ExprKind::Range { .. }
            | crate::ast::ExprKind::StructLiteral { .. }
            | crate::ast::ExprKind::Match { .. }
            | crate::ast::ExprKind::Closure { .. }
            | crate::ast::ExprKind::TypeCall { .. }
            | crate::ast::ExprKind::Try(_) => {
                // Unsupported in VIR lowering for now — emit zero placeholder.
                Ok(self.cur_fn().const_i64(0))
            }
        }
    }

    /// Convert an AST type expression into a VIR type.
    fn lower_type_expr(&self, te: &crate::ast::TypeExpr) -> VirType {
        match &te.kind {
            crate::ast::TypeExprKind::Named(ident) => match ident.name.as_str() {
                "void" | "()" => VirType::Void,
                "bool" => VirType::Bool,
                "i8" => VirType::I8,
                "i16" => VirType::I16,
                "i32" | "int" => VirType::I32,
                "i64" => VirType::I64,
                "i128" => VirType::I128,
                "f16" => VirType::F16,
                "f32" | "float" => VirType::F32,
                "f64" => VirType::F64,
                other => VirType::Struct(other.to_string(), vec![]),
            },
            crate::ast::TypeExprKind::Generic { name, args } => {
                if name.name == "Tensor" {
                    let elem = if let Some(crate::ast::TypeArg::Type(te)) = args.first() {
                        Box::new(self.lower_type_expr(te))
                    } else {
                        Box::new(VirType::F32)
                    };
                    let shape = if let Some(crate::ast::TypeArg::Shape(dims)) = args.get(1) {
                        dims.iter()
                            .map(|d| match d {
                                crate::ast::ShapeDim::Lit(n) => VirDim::Fixed(*n as usize),
                                crate::ast::ShapeDim::Ident(id) => {
                                    VirDim::Symbolic(id.name.clone())
                                }
                                crate::ast::ShapeDim::Dynamic => VirDim::Dynamic,
                                crate::ast::ShapeDim::Expr(_) => VirDim::Dynamic,
                            })
                            .collect()
                    } else {
                        vec![VirDim::Dynamic]
                    };
                    VirType::Tensor { elem, shape }
                } else {
                    VirType::Struct(name.name.clone(), vec![])
                }
            }
            crate::ast::TypeExprKind::Array { elem, .. } => {
                VirType::Array(Box::new(self.lower_type_expr(elem)), 0)
            }
            crate::ast::TypeExprKind::Ref { inner, .. } => {
                VirType::Ptr(Box::new(self.lower_type_expr(inner)))
            }
            crate::ast::TypeExprKind::Fn { params, ret } => VirType::Fn {
                params: params.iter().map(|p| self.lower_type_expr(p)).collect(),
                ret: Box::new(self.lower_type_expr(ret)),
            },
            _ => VirType::I64, // fallback
        }
    }
}

// ---------------------------------------------------------------------------
// Optimization passes
// ---------------------------------------------------------------------------

/// Constant folding — evaluate constant expressions at compile time.
pub fn constant_fold(module: &mut VirModule) {
    for func in &mut module.functions {
        // Build a map of id -> constant value
        let mut int_consts: HashMap<VirId, i128> = HashMap::new();
        let mut float_consts: HashMap<VirId, f64> = HashMap::new();
        let mut bool_consts: HashMap<VirId, bool> = HashMap::new();

        for bb in &mut func.blocks {
            let mut new_insts = Vec::with_capacity(bb.insts.len());
            for inst in bb.insts.drain(..) {
                match &inst.kind {
                    VirInstKind::ConstInt(v) => {
                        int_consts.insert(inst.id, *v);
                        new_insts.push(inst);
                    }
                    VirInstKind::ConstFloat(v) => {
                        float_consts.insert(inst.id, *v);
                        new_insts.push(inst);
                    }
                    VirInstKind::ConstBool(v) => {
                        bool_consts.insert(inst.id, *v);
                        new_insts.push(inst);
                    }
                    VirInstKind::Add(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            let result = va.wrapping_add(vb);
                            int_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstInt(result),
                                ty: inst.ty,
                            });
                        } else if let (Some(&va), Some(&vb)) =
                            (float_consts.get(a), float_consts.get(b))
                        {
                            let result = va + vb;
                            float_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstFloat(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Sub(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            let result = va.wrapping_sub(vb);
                            int_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstInt(result),
                                ty: inst.ty,
                            });
                        } else if let (Some(&va), Some(&vb)) =
                            (float_consts.get(a), float_consts.get(b))
                        {
                            let result = va - vb;
                            float_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstFloat(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Mul(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            let result = va.wrapping_mul(vb);
                            int_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstInt(result),
                                ty: inst.ty,
                            });
                        } else if let (Some(&va), Some(&vb)) =
                            (float_consts.get(a), float_consts.get(b))
                        {
                            let result = va * vb;
                            float_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstFloat(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Div(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            if vb != 0 {
                                let result = va / vb;
                                int_consts.insert(inst.id, result);
                                new_insts.push(VirInst {
                                    id: inst.id,
                                    kind: VirInstKind::ConstInt(result),
                                    ty: inst.ty,
                                });
                            } else {
                                new_insts.push(inst);
                            }
                        } else if let (Some(&va), Some(&vb)) =
                            (float_consts.get(a), float_consts.get(b))
                        {
                            let result = va / vb;
                            float_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstFloat(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Neg(a) => {
                        if let Some(&va) = int_consts.get(a) {
                            let result = -va;
                            int_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstInt(result),
                                ty: inst.ty,
                            });
                        } else if let Some(&va) = float_consts.get(a) {
                            let result = -va;
                            float_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstFloat(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Not(a) => {
                        if let Some(&va) = bool_consts.get(a) {
                            let result = !va;
                            bool_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstBool(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Eq(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            let result = va == vb;
                            bool_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstBool(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    VirInstKind::Lt(a, b) => {
                        if let (Some(&va), Some(&vb)) = (int_consts.get(a), int_consts.get(b)) {
                            let result = va < vb;
                            bool_consts.insert(inst.id, result);
                            new_insts.push(VirInst {
                                id: inst.id,
                                kind: VirInstKind::ConstBool(result),
                                ty: inst.ty,
                            });
                        } else {
                            new_insts.push(inst);
                        }
                    }
                    _ => {
                        new_insts.push(inst);
                    }
                }
            }
            bb.insts = new_insts;
        }
    }
}

/// Dead code elimination — remove instructions whose results are never used.
pub fn dead_code_eliminate(module: &mut VirModule) {
    for func in &mut module.functions {
        let used = func.used_ids();
        for bb in &mut func.blocks {
            bb.insts.retain(|inst| {
                // Keep side-effecting instructions always
                match &inst.kind {
                    VirInstKind::Store(_, _)
                    | VirInstKind::TensorStore(_, _, _)
                    | VirInstKind::Call(_, _)
                    | VirInstKind::Param(_) => true,
                    _ => used.contains(&inst.id),
                }
            });
        }
    }
}

/// Tensor fusion — fuse sequential tensor operations into combined ops.
///
/// Currently supports:
/// - MatMul + Relu  -> fused matmul+relu (represented as MatMul followed by Relu
///   with a marker comment, but we collapse the Relu into a Call to __fused_matmul_relu)
/// - MatMul + Sigmoid -> fused_matmul_sigmoid
/// - MatMul + Gelu    -> fused_matmul_gelu
pub fn tensor_fusion(module: &mut VirModule) {
    for func in &mut module.functions {
        // Map from VirId -> index of the instruction that produced it, per block
        for bb in &mut func.blocks {
            let mut producer: HashMap<VirId, usize> = HashMap::new();
            for (i, inst) in bb.insts.iter().enumerate() {
                producer.insert(inst.id, i);
            }

            // Identify fusion candidates
            let mut fuse_targets: Vec<(usize, usize, String)> = Vec::new(); // (matmul_idx, act_idx, fused_name)

            for (i, inst) in bb.insts.iter().enumerate() {
                let (input_id, fused_name) = match &inst.kind {
                    VirInstKind::Relu(a) => (*a, "__fused_matmul_relu"),
                    VirInstKind::Sigmoid(a) => (*a, "__fused_matmul_sigmoid"),
                    VirInstKind::Gelu(a) => (*a, "__fused_matmul_gelu"),
                    VirInstKind::Tanh(a) => (*a, "__fused_matmul_tanh"),
                    _ => continue,
                };
                if let Some(&mm_idx) = producer.get(&input_id) {
                    if matches!(bb.insts[mm_idx].kind, VirInstKind::MatMul(_, _)) {
                        fuse_targets.push((mm_idx, i, fused_name.to_string()));
                    }
                }
            }

            // Apply fusions (replace activation with fused call, keep matmul operands)
            for (mm_idx, act_idx, fused_name) in fuse_targets.into_iter().rev() {
                let (a, b) = match &bb.insts[mm_idx].kind {
                    VirInstKind::MatMul(a, b) => (*a, *b),
                    _ => unreachable!(),
                };
                let fused_ty = bb.insts[act_idx].ty.clone();
                let fused_id = bb.insts[act_idx].id;

                // Replace activation with fused call
                bb.insts[act_idx] = VirInst {
                    id: fused_id,
                    kind: VirInstKind::Call(fused_name, vec![a, b]),
                    ty: fused_ty,
                };

                // Remove the original matmul (if its result is only used by the activation)
                // Check if anyone else uses the matmul result
                let mm_id = bb.insts[mm_idx].id;
                let mut other_uses = false;
                for (i, inst) in bb.insts.iter().enumerate() {
                    if i == act_idx {
                        continue;
                    }
                    if inst.kind.uses().contains(&mm_id) {
                        other_uses = true;
                        break;
                    }
                }
                if !other_uses {
                    bb.insts.remove(mm_idx);
                }
            }
        }
    }
}

/// Common subexpression elimination — replace duplicate computations with
/// references to the first occurrence.
pub fn common_subexpression_eliminate(module: &mut VirModule) {
    for func in &mut module.functions {
        for bb in &mut func.blocks {
            // Key: a canonical string representation of an instruction's operation.
            let mut seen: HashMap<String, VirId> = HashMap::new();
            // Mapping from eliminated id -> canonical id
            let mut replacements: HashMap<VirId, VirId> = HashMap::new();

            // First pass: identify duplicates
            for inst in &bb.insts {
                let key = cse_key(&inst.kind);
                if let Some(key) = key {
                    if let Some(&first_id) = seen.get(&key) {
                        replacements.insert(inst.id, first_id);
                    } else {
                        seen.insert(key, inst.id);
                    }
                }
            }

            if replacements.is_empty() {
                continue;
            }

            // Second pass: rewrite uses
            for inst in &mut bb.insts {
                rewrite_uses(&mut inst.kind, &replacements);
            }

            // Third pass: remove dead duplicates
            let replaced_ids: HashSet<VirId> = replacements.keys().copied().collect();
            bb.insts.retain(|inst| !replaced_ids.contains(&inst.id));
        }
    }
}

/// Generate a canonical key for CSE. Returns None for instructions that
/// should not be CSE'd (side-effecting, etc).
fn cse_key(kind: &VirInstKind) -> Option<String> {
    match kind {
        VirInstKind::ConstInt(v) => Some(format!("ci:{}", v)),
        VirInstKind::ConstFloat(v) => Some(format!("cf:{:?}", v.to_bits())),
        VirInstKind::ConstBool(v) => Some(format!("cb:{}", v)),
        VirInstKind::Add(a, b) => Some(format!("add:{},{}", a, b)),
        VirInstKind::Sub(a, b) => Some(format!("sub:{},{}", a, b)),
        VirInstKind::Mul(a, b) => Some(format!("mul:{},{}", a, b)),
        VirInstKind::Div(a, b) => Some(format!("div:{},{}", a, b)),
        VirInstKind::Mod(a, b) => Some(format!("mod:{},{}", a, b)),
        VirInstKind::Neg(a) => Some(format!("neg:{}", a)),
        VirInstKind::Eq(a, b) => Some(format!("eq:{},{}", a, b)),
        VirInstKind::Ne(a, b) => Some(format!("ne:{},{}", a, b)),
        VirInstKind::Lt(a, b) => Some(format!("lt:{},{}", a, b)),
        VirInstKind::Le(a, b) => Some(format!("le:{},{}", a, b)),
        VirInstKind::Gt(a, b) => Some(format!("gt:{},{}", a, b)),
        VirInstKind::Ge(a, b) => Some(format!("ge:{},{}", a, b)),
        VirInstKind::And(a, b) => Some(format!("and:{},{}", a, b)),
        VirInstKind::Or(a, b) => Some(format!("or:{},{}", a, b)),
        VirInstKind::Not(a) => Some(format!("not:{}", a)),
        VirInstKind::BitAnd(a, b) => Some(format!("ba:{},{}", a, b)),
        VirInstKind::BitOr(a, b) => Some(format!("bo:{},{}", a, b)),
        VirInstKind::BitXor(a, b) => Some(format!("bx:{},{}", a, b)),
        VirInstKind::Shl(a, b) => Some(format!("shl:{},{}", a, b)),
        VirInstKind::Shr(a, b) => Some(format!("shr:{},{}", a, b)),
        VirInstKind::BitNot(a) => Some(format!("bn:{}", a)),
        VirInstKind::MatMul(a, b) => Some(format!("mm:{},{}", a, b)),
        VirInstKind::Relu(a) => Some(format!("relu:{}", a)),
        VirInstKind::Sigmoid(a) => Some(format!("sig:{}", a)),
        VirInstKind::Tanh(a) => Some(format!("tanh:{}", a)),
        VirInstKind::Gelu(a) => Some(format!("gelu:{}", a)),
        VirInstKind::Softmax(a, ax) => Some(format!("sm:{}:{}", a, ax)),
        VirInstKind::Cast(a, ty) => Some(format!("cast:{}:{}", a, ty)),
        VirInstKind::TensorShape(a, d) => Some(format!("ts:{}:{}", a, d)),
        // Side-effecting / non-CSE-able:
        _ => None,
    }
}

/// Rewrite all VirId uses in an instruction according to the replacement map.
fn rewrite_uses(kind: &mut VirInstKind, map: &HashMap<VirId, VirId>) {
    fn r(id: &mut VirId, map: &HashMap<VirId, VirId>) {
        if let Some(&new) = map.get(id) {
            *id = new;
        }
    }

    match kind {
        VirInstKind::Add(a, b)
        | VirInstKind::Sub(a, b)
        | VirInstKind::Mul(a, b)
        | VirInstKind::Div(a, b)
        | VirInstKind::Mod(a, b)
        | VirInstKind::Eq(a, b)
        | VirInstKind::Ne(a, b)
        | VirInstKind::Lt(a, b)
        | VirInstKind::Le(a, b)
        | VirInstKind::Gt(a, b)
        | VirInstKind::Ge(a, b)
        | VirInstKind::And(a, b)
        | VirInstKind::Or(a, b)
        | VirInstKind::BitAnd(a, b)
        | VirInstKind::BitOr(a, b)
        | VirInstKind::BitXor(a, b)
        | VirInstKind::Shl(a, b)
        | VirInstKind::Shr(a, b)
        | VirInstKind::MatMul(a, b) => {
            r(a, map);
            r(b, map);
        }
        VirInstKind::Store(a, b) => {
            r(a, map);
            r(b, map);
        }
        VirInstKind::Neg(a)
        | VirInstKind::Not(a)
        | VirInstKind::BitNot(a)
        | VirInstKind::Load(a)
        | VirInstKind::Relu(a)
        | VirInstKind::Sigmoid(a)
        | VirInstKind::Tanh(a)
        | VirInstKind::Gelu(a) => {
            r(a, map);
        }
        VirInstKind::Softmax(a, _) | VirInstKind::TensorShape(a, _) => {
            r(a, map);
        }
        VirInstKind::Cast(a, _) => {
            r(a, map);
        }
        VirInstKind::Call(_, args) => {
            for a in args {
                r(a, map);
            }
        }
        VirInstKind::Phi(pairs) => {
            for (v, _) in pairs {
                r(v, map);
            }
        }
        VirInstKind::GetElementPtr(base, idxs) => {
            r(base, map);
            for i in idxs {
                r(i, map);
            }
        }
        VirInstKind::StructCreate(_, fields) => {
            for (_, v) in fields {
                r(v, map);
            }
        }
        VirInstKind::StructGet(s, _) => {
            r(s, map);
        }
        VirInstKind::StructSet(s, _, v) => {
            r(s, map);
            r(v, map);
        }
        VirInstKind::TensorCreate { data, shape, .. } => {
            r(data, map);
            for s in shape {
                r(s, map);
            }
        }
        VirInstKind::TensorLoad(t, idxs) => {
            r(t, map);
            for i in idxs {
                r(i, map);
            }
        }
        VirInstKind::TensorStore(t, idxs, v) => {
            r(t, map);
            for i in idxs {
                r(i, map);
            }
            r(v, map);
        }
        VirInstKind::TensorReshape(t, dims) => {
            r(t, map);
            for d in dims {
                r(d, map);
            }
        }
        VirInstKind::TensorSlice {
            tensor,
            start,
            end,
            ..
        } => {
            r(tensor, map);
            r(start, map);
            r(end, map);
        }
        VirInstKind::Conv2d { input, weight, .. } => {
            r(input, map);
            r(weight, map);
        }
        VirInstKind::Reduce { tensor, .. } => {
            r(tensor, map);
        }
        VirInstKind::Broadcast(t, dims) => {
            r(t, map);
            for d in dims {
                r(d, map);
            }
        }
        VirInstKind::Transpose(t, _) => {
            r(t, map);
        }
        VirInstKind::LayerNorm { input, gamma, beta, .. } => {
            r(input, map); r(gamma, map); r(beta, map);
        }
        VirInstKind::RMSNorm { input, gamma, .. } => {
            r(input, map); r(gamma, map);
        }
        VirInstKind::FlashAttention { q, k, v, mask, .. } => {
            r(q, map); r(k, map); r(v, map);
            if let Some(m) = mask { r(m, map); }
        }
        VirInstKind::FusedLinear { input, weight, bias, .. } => {
            r(input, map); r(weight, map);
            if let Some(b) = bias { r(b, map); }
        }
        VirInstKind::FusedMHA { input, wq, wk, wv, wo, .. } => {
            r(input, map); r(wq, map); r(wk, map); r(wv, map); r(wo, map);
        }
        VirInstKind::SpecDecode { draft_logits, target_logits, draft_tokens, .. } => {
            r(draft_logits, map); r(target_logits, map); r(draft_tokens, map);
        }
        VirInstKind::Quantize { input, .. } => { r(input, map); }
        VirInstKind::Dequantize { input, scale, zero_point } => {
            r(input, map); r(scale, map);
            if let Some(zp) = zero_point { r(zp, map); }
        }
        VirInstKind::QMatMul { a, b, a_scale, b_scale } => {
            r(a, map); r(b, map); r(a_scale, map); r(b_scale, map);
        }
        VirInstKind::Alloca(_)
        | VirInstKind::ConstInt(_)
        | VirInstKind::ConstFloat(_)
        | VirInstKind::ConstBool(_)
        | VirInstKind::Param(_) => {}
    }
}

/// Run all optimization passes in the standard order.
/// Advanced kernel fusion: fuse matmul + bias + activation into FusedLinear.
/// Also fuse Q/K/V projections + attention into FusedMHA.
pub fn advanced_fusion(module: &mut VirModule) {
    for func in &mut module.functions {
        for bb in &mut func.blocks {
            let mut producer: HashMap<VirId, usize> = HashMap::new();
            for (i, inst) in bb.insts.iter().enumerate() {
                producer.insert(inst.id, i);
            }

            // Pass 1: Fuse matmul + add (bias) + activation → FusedLinear
            let mut fused_linear_targets: Vec<(usize, Option<usize>, Option<usize>, FusedActivation)> = Vec::new();
            for (i, inst) in bb.insts.iter().enumerate() {
                // Look for activation(add(matmul(x, w), bias)) or activation(matmul(x, w))
                let (inner_id, act) = match &inst.kind {
                    VirInstKind::Relu(a) => (*a, FusedActivation::Relu),
                    VirInstKind::Gelu(a) => (*a, FusedActivation::Gelu),
                    VirInstKind::Sigmoid(a) => (*a, FusedActivation::Sigmoid),
                    VirInstKind::Tanh(a) => (*a, FusedActivation::Tanh),
                    _ => continue,
                };
                // Check if inner is an Add (bias)
                if let Some(&add_idx) = producer.get(&inner_id) {
                    if let VirInstKind::Add(mm_id, bias_id) = &bb.insts[add_idx].kind {
                        if let Some(&mm_idx) = producer.get(mm_id) {
                            if matches!(bb.insts[mm_idx].kind, VirInstKind::MatMul(_, _)) {
                                fused_linear_targets.push((mm_idx, Some(add_idx), Some(i), act));
                                continue;
                            }
                        }
                    }
                }
                // Check if inner is directly a MatMul (no bias)
                if let Some(&mm_idx) = producer.get(&inner_id) {
                    if matches!(bb.insts[mm_idx].kind, VirInstKind::MatMul(_, _)) {
                        fused_linear_targets.push((mm_idx, None, Some(i), act));
                    }
                }
            }

            // Apply FusedLinear fusions (reverse order to preserve indices)
            for (mm_idx, add_idx, act_idx, activation) in fused_linear_targets.into_iter().rev() {
                let (input, weight) = match &bb.insts[mm_idx].kind {
                    VirInstKind::MatMul(a, b) => (*a, *b),
                    _ => continue,
                };
                let bias = add_idx.and_then(|ai| {
                    if let VirInstKind::Add(_, b) = &bb.insts[ai].kind {
                        Some(*b)
                    } else { None }
                });
                let final_idx = act_idx.unwrap_or(add_idx.unwrap_or(mm_idx));
                let fused_ty = bb.insts[final_idx].ty.clone();
                let fused_id = bb.insts[final_idx].id;

                bb.insts[final_idx] = VirInst {
                    id: fused_id,
                    kind: VirInstKind::FusedLinear { input, weight, bias, activation },
                    ty: fused_ty,
                };

                // Remove intermediates if not used elsewhere
                let mut to_remove = Vec::new();
                if let Some(ai) = add_idx {
                    let add_id = bb.insts[ai].id;
                    let used = bb.insts.iter().enumerate().any(|(j, inst)| j != final_idx && inst.kind.uses().contains(&add_id));
                    if !used { to_remove.push(ai); }
                }
                let mm_id_val = bb.insts[mm_idx].id;
                let mm_used = bb.insts.iter().enumerate().any(|(j, inst)| {
                    j != final_idx && add_idx.map_or(true, |ai| j != ai) && inst.kind.uses().contains(&mm_id_val)
                });
                if !mm_used { to_remove.push(mm_idx); }
                to_remove.sort_unstable();
                to_remove.dedup();
                for idx in to_remove.into_iter().rev() {
                    bb.insts.remove(idx);
                }
            }
        }
    }
}

/// Fuse attention patterns: detect softmax(Q@K^T / sqrt(d)) @ V and replace with FlashAttention.
pub fn attention_fusion(module: &mut VirModule) {
    for func in &mut module.functions {
        for bb in &mut func.blocks {
            let mut producer: HashMap<VirId, usize> = HashMap::new();
            for (i, inst) in bb.insts.iter().enumerate() {
                producer.insert(inst.id, i);
            }

            // Look for pattern: matmul(softmax(matmul(Q, K_t)), V)
            let mut attention_targets: Vec<(usize, VirId, VirId, VirId)> = Vec::new(); // (final_idx, q, k, v)
            for (i, inst) in bb.insts.iter().enumerate() {
                // outer matmul: attn_weights @ V
                if let VirInstKind::MatMul(weights_id, v) = &inst.kind {
                    if let Some(&softmax_idx) = producer.get(weights_id) {
                        if let VirInstKind::Softmax(scores_id, _) = &bb.insts[softmax_idx].kind {
                            if let Some(&inner_mm_idx) = producer.get(scores_id) {
                                if let VirInstKind::MatMul(q, k_t) = &bb.insts[inner_mm_idx].kind {
                                    attention_targets.push((i, *q, *k_t, *v));
                                }
                            }
                        }
                    }
                }
            }

            // Replace with FlashAttention
            for (final_idx, q, k, v) in attention_targets.into_iter().rev() {
                let fused_ty = bb.insts[final_idx].ty.clone();
                let fused_id = bb.insts[final_idx].id;
                bb.insts[final_idx] = VirInst {
                    id: fused_id,
                    kind: VirInstKind::FlashAttention {
                        q, k, v,
                        mask: None,
                        scale: 1.0, // caller should set proper scale
                        causal: false,
                    },
                    ty: fused_ty,
                };
            }
        }
    }
}

pub fn optimize(module: &mut VirModule) {
    constant_fold(module);
    common_subexpression_eliminate(module);
    dead_code_eliminate(module);
    tensor_fusion(module);
    advanced_fusion(module);
    attention_fusion(module);
    dead_code_eliminate(module); // clean up after fusion
}

// ---------------------------------------------------------------------------
// Builtin registration
// ---------------------------------------------------------------------------

/// Register the `vir_compile` builtin into the Vortex interpreter environment.
pub fn register_builtins(env: &mut Env) {
    env.functions
        .insert("vir_compile".into(), FnDef::Builtin(builtin_vir_compile));
    env.functions
        .insert("vir_dump".into(), FnDef::Builtin(builtin_vir_dump));
    env.functions
        .insert("vir_optimize".into(), FnDef::Builtin(builtin_vir_optimize));
}

/// `vir_compile(module_name)` — creates a new VIR module and returns its name.
fn builtin_vir_compile(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let name = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "unnamed".to_string(),
    };
    let module = VirModule::new(&name);
    Ok(Value::String(format!(
        "VIR module '{}' created ({} functions)",
        module.name,
        module.functions.len()
    )))
}

/// `vir_dump()` — returns a sample VIR text for demonstration.
fn builtin_vir_dump(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut module = VirModule::new("demo");
    let mut f = VirFunction::new(
        "add_tensors",
        vec![
            (
                "a",
                VirType::Tensor {
                    elem: Box::new(VirType::F32),
                    shape: vec![VirDim::Symbolic("N".into()), VirDim::Symbolic("M".into())],
                },
            ),
            (
                "b",
                VirType::Tensor {
                    elem: Box::new(VirType::F32),
                    shape: vec![VirDim::Symbolic("N".into()), VirDim::Symbolic("M".into())],
                },
            ),
        ],
        VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![VirDim::Symbolic("N".into()), VirDim::Symbolic("M".into())],
        },
    );
    // %0 = param 0 (a), %1 = param 1 (b)  — already emitted by new()
    let result = f.emit(VirInstKind::Add(0, 1), VirType::Tensor {
        elem: Box::new(VirType::F32),
        shape: vec![VirDim::Symbolic("N".into()), VirDim::Symbolic("M".into())],
    });
    f.ret(Some(result));
    module.add_function(f);
    Ok(Value::String(module.to_text()))
}

/// `vir_optimize(text)` — placeholder that returns info about available passes.
fn builtin_vir_optimize(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    Ok(Value::String(
        "VIR optimization passes: constant_fold, dead_code_eliminate, tensor_fusion, common_subexpression_eliminate".to_string(),
    ))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vir_construction_and_text() {
        let mut module = VirModule::new("test_mod");
        let mut f = VirFunction::new(
            "simple_add",
            vec![("x", VirType::I64), ("y", VirType::I64)],
            VirType::I64,
        );
        // params are %0 and %1
        let sum = f.add(0, 1);
        f.ret(Some(sum));
        module.add_function(f);

        let text = module.to_text();
        assert!(text.contains("fn @simple_add"));
        assert!(text.contains("add %0, %1"));
        assert!(text.contains("ret %"));
    }

    #[test]
    fn test_constant_folding() {
        let mut module = VirModule::new("cf_test");
        let mut f = VirFunction::new("fold_me", vec![], VirType::I64);

        let a = f.const_i64(10);
        let b = f.const_i64(20);
        let c = f.add(a, b);
        f.ret(Some(c));

        module.add_function(f);
        constant_fold(&mut module);

        // The add should have been folded into const 30
        let func = &module.functions[0];
        let last_bb = &func.blocks[0];
        let folded = last_bb.insts.iter().find(|i| i.id == c).unwrap();
        match &folded.kind {
            VirInstKind::ConstInt(30) => {} // success
            other => panic!("expected ConstInt(30), got {:?}", other),
        }
    }

    #[test]
    fn test_constant_folding_float() {
        let mut module = VirModule::new("cf_float");
        let mut f = VirFunction::new("fold_float", vec![], VirType::F64);

        let a = f.const_f64(2.5);
        let b = f.const_f64(3.5);
        let c = f.emit(VirInstKind::Mul(a, b), VirType::F64);
        f.ret(Some(c));

        module.add_function(f);
        constant_fold(&mut module);

        let func = &module.functions[0];
        let folded = func.blocks[0].insts.iter().find(|i| i.id == c).unwrap();
        match &folded.kind {
            VirInstKind::ConstFloat(v) => {
                assert!((v - 8.75).abs() < 1e-10);
            }
            other => panic!("expected ConstFloat(8.75), got {:?}", other),
        }
    }

    #[test]
    fn test_dead_code_elimination() {
        let mut module = VirModule::new("dce_test");
        let mut f = VirFunction::new("dead_code", vec![], VirType::I64);

        let a = f.const_i64(1);
        let b = f.const_i64(2);
        let _dead = f.add(a, b); // result never used
        let c = f.const_i64(42);
        f.ret(Some(c));

        module.add_function(f);
        dead_code_eliminate(&mut module);

        let func = &module.functions[0];
        // The add instruction should have been removed.
        // Only the const 42 and possibly consts 1 and 2 remain
        // (1 and 2 are also dead since the add is removed).
        // After a second pass they would be removed too, but one pass
        // only removes immediate dead code.
        let has_add = func.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(&i.kind, VirInstKind::Add(_, _)));
        assert!(!has_add, "add should be eliminated");
    }

    #[test]
    fn test_tensor_fusion() {
        let mut module = VirModule::new("fusion_test");
        let tensor_ty = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![VirDim::Fixed(4), VirDim::Fixed(4)],
        };
        let mut f = VirFunction::new(
            "matmul_relu",
            vec![("a", tensor_ty.clone()), ("b", tensor_ty.clone())],
            tensor_ty.clone(),
        );

        let mm = f.emit(VirInstKind::MatMul(0, 1), tensor_ty.clone());
        let act = f.emit(VirInstKind::Relu(mm), tensor_ty.clone());
        f.ret(Some(act));

        module.add_function(f);
        tensor_fusion(&mut module);

        let func = &module.functions[0];
        // The relu should have been replaced with a fused call
        let has_fused = func.blocks[0].insts.iter().any(|i| {
            matches!(&i.kind, VirInstKind::Call(name, _) if name == "__fused_matmul_relu")
        });
        assert!(has_fused, "matmul+relu should be fused");

        // The standalone matmul should be gone
        let has_matmul = func.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(&i.kind, VirInstKind::MatMul(_, _)));
        assert!(!has_matmul, "standalone matmul should be removed after fusion");
    }

    #[test]
    fn test_cse() {
        let mut module = VirModule::new("cse_test");
        let mut f = VirFunction::new("cse", vec![("x", VirType::I64)], VirType::I64);

        // Two identical adds
        let a = f.add(0, 0);
        let b = f.add(0, 0); // duplicate of a
        let c = f.emit(VirInstKind::Add(a, b), VirType::I64);
        f.ret(Some(c));

        module.add_function(f);
        common_subexpression_eliminate(&mut module);

        let func = &module.functions[0];
        // b should have been eliminated and c should reference a twice
        let c_inst = func.blocks[0].insts.iter().find(|i| i.id == c).unwrap();
        if let VirInstKind::Add(l, r) = &c_inst.kind {
            assert_eq!(l, r, "CSE should make both operands point to the same value");
            assert_eq!(*l, a);
        } else {
            panic!("expected Add");
        }
    }

    #[test]
    fn test_basic_block_control_flow() {
        let mut module = VirModule::new("cfg_test");
        let mut f = VirFunction::new("branch", vec![("cond", VirType::Bool)], VirType::I64);

        let then_bb = f.add_block("then");
        let else_bb = f.add_block("else");
        let merge_bb = f.add_block("merge");

        f.cond_br(0, then_bb, else_bb);

        f.set_block(then_bb);
        let v1 = f.const_i64(1);
        f.br(merge_bb);

        f.set_block(else_bb);
        let v2 = f.const_i64(2);
        f.br(merge_bb);

        f.set_block(merge_bb);
        let phi = f.emit(
            VirInstKind::Phi(vec![(v1, then_bb), (v2, else_bb)]),
            VirType::I64,
        );
        f.ret(Some(phi));

        module.add_function(f);

        let text = module.to_text();
        assert!(text.contains("bb1 (then):"));
        assert!(text.contains("bb2 (else):"));
        assert!(text.contains("phi"));
    }

    #[test]
    fn test_tensor_types() {
        let ty = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![
                VirDim::Fixed(128),
                VirDim::Symbolic("N".into()),
                VirDim::Dynamic,
            ],
        };
        let text = format!("{}", ty);
        assert_eq!(text, "tensor<f32; [128, N, ?]>");
    }

    #[test]
    fn test_kernel_function() {
        let mut module = VirModule::new("gpu");
        let tensor_ty = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![VirDim::Symbolic("N".into())],
        };
        let mut f = VirFunction::new("vector_add", vec![("a", tensor_ty.clone()), ("b", tensor_ty.clone())], tensor_ty);
        f.is_kernel = true;
        let sum = f.emit(VirInstKind::Add(0, 1), VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![VirDim::Symbolic("N".into())],
        });
        f.ret(Some(sum));
        module.add_function(f);

        let text = module.to_text();
        assert!(text.contains("kernel fn @vector_add"));
    }

    #[test]
    fn test_vir_inst_uses() {
        let inst = VirInstKind::Add(3, 5);
        assert_eq!(inst.uses(), vec![3, 5]);

        let inst2 = VirInstKind::ConstInt(42);
        assert!(inst2.uses().is_empty());

        let inst3 = VirInstKind::Call("foo".into(), vec![1, 2, 3]);
        assert_eq!(inst3.uses(), vec![1, 2, 3]);

        let inst4 = VirInstKind::Phi(vec![(10, 0), (20, 1)]);
        assert_eq!(inst4.uses(), vec![10, 20]);
    }

    #[test]
    fn test_vir_module_display() {
        let mut module = VirModule::new("display_test");
        module.structs.push((
            "Point".to_string(),
            vec![
                ("x".to_string(), VirType::F64),
                ("y".to_string(), VirType::F64),
            ],
        ));
        module.globals.push(("PI".to_string(), VirType::F64, None));

        let text = module.to_text();
        assert!(text.contains("struct %Point"));
        assert!(text.contains("x: f64"));
        assert!(text.contains("global @PI: f64"));
    }

    #[test]
    fn test_optimize_pipeline() {
        let mut module = VirModule::new("opt_test");
        let mut f = VirFunction::new("pipeline", vec![], VirType::I64);

        let a = f.const_i64(5);
        let b = f.const_i64(10);
        let c = f.add(a, b); // should fold to 15
        let _dead = f.const_i64(999); // dead code
        f.ret(Some(c));

        module.add_function(f);
        optimize(&mut module);

        let func = &module.functions[0];
        let c_inst = func.blocks[0].insts.iter().find(|i| i.id == c).unwrap();
        assert!(
            matches!(&c_inst.kind, VirInstKind::ConstInt(15)),
            "pipeline should fold 5+10 to 15"
        );
        // Dead code (999) should be gone
        let has_999 = func.blocks[0]
            .insts
            .iter()
            .any(|i| matches!(&i.kind, VirInstKind::ConstInt(999)));
        assert!(!has_999, "dead constant should be eliminated");
    }

    #[test]
    fn test_vir_type_properties() {
        assert!(VirType::F32.is_float());
        assert!(VirType::F64.is_float());
        assert!(!VirType::I64.is_float());
        assert!(VirType::I32.is_int());
        assert!(!VirType::F32.is_int());

        let t = VirType::Tensor {
            elem: Box::new(VirType::F16),
            shape: vec![VirDim::Fixed(8)],
        };
        assert!(t.is_tensor());
        assert_eq!(t.tensor_elem(), Some(&VirType::F16));
        assert_eq!(VirType::I64.tensor_elem(), None);
    }

    #[test]
    fn test_lowering_simple_function() {
        use crate::ast::*;
        use crate::lexer::Span;

        let span = Span { start: 0, end: 0 };
        let program = Program {
            items: vec![Item {
                kind: ItemKind::Function(Function {
                    name: Ident::new("add".into(), span),
                    generics: vec![],
                    params: vec![
                        Param {
                            name: Ident::new("a".into(), span),
                            ty: TypeExpr {
                                kind: TypeExprKind::Named(Ident::new("i64".into(), span)),
                                span,
                            },
                            default: None,
                            span,
                        },
                        Param {
                            name: Ident::new("b".into(), span),
                            ty: TypeExpr {
                                kind: TypeExprKind::Named(Ident::new("i64".into(), span)),
                                span,
                            },
                            default: None,
                            span,
                        },
                    ],
                    ret_type: Some(TypeExpr {
                        kind: TypeExprKind::Named(Ident::new("i64".into(), span)),
                        span,
                    }),
                    where_clause: vec![],
                    body: Block {
                        stmts: vec![],
                        expr: Some(Box::new(Expr {
                            kind: ExprKind::Binary {
                                lhs: Box::new(Expr {
                                    kind: ExprKind::Ident(Ident::new("a".into(), span)),
                                    span,
                                }),
                                op: BinOp::Add,
                                rhs: Box::new(Expr {
                                    kind: ExprKind::Ident(Ident::new("b".into(), span)),
                                    span,
                                }),
                            },
                            span,
                        })),
                        span,
                    },
                    annotations: vec![],
                }),
                span,
                is_pub: false,
            }],
        };

        let mut lowering = VirLowering::new("test");
        let module = lowering.lower_program(&program).unwrap();

        assert_eq!(module.functions.len(), 1);
        assert_eq!(module.functions[0].name, "add");
        let text = module.to_text();
        assert!(text.contains("add %0, %1"));
    }

    #[test]
    fn test_reduce_op_display() {
        assert_eq!(format!("{}", ReduceOp::Sum), "sum");
        assert_eq!(format!("{}", ReduceOp::Max), "max");
        assert_eq!(format!("{}", ReduceOp::Min), "min");
        assert_eq!(format!("{}", ReduceOp::Mean), "mean");
    }

    #[test]
    fn test_vir_dim_display() {
        assert_eq!(format!("{}", VirDim::Fixed(42)), "42");
        assert_eq!(format!("{}", VirDim::Symbolic("N".into())), "N");
        assert_eq!(format!("{}", VirDim::Dynamic), "?");
    }

    #[test]
    fn test_terminator_display() {
        assert_eq!(format!("{}", Terminator::Return(None)), "ret void");
        assert_eq!(format!("{}", Terminator::Return(Some(5))), "ret %5");
        assert_eq!(format!("{}", Terminator::Branch(2)), "br bb2");
        assert_eq!(
            format!(
                "{}",
                Terminator::CondBranch {
                    cond: 3,
                    true_bb: 1,
                    false_bb: 2
                }
            ),
            "br %3, bb1, bb2"
        );
        assert_eq!(format!("{}", Terminator::Unreachable), "unreachable");
    }

    #[test]
    fn test_struct_operations() {
        let mut module = VirModule::new("struct_test");
        module.structs.push((
            "Vec2".to_string(),
            vec![
                ("x".to_string(), VirType::F64),
                ("y".to_string(), VirType::F64),
            ],
        ));

        let mut f = VirFunction::new("make_vec", vec![], VirType::Struct("Vec2".to_string(), vec![]));
        let x = f.const_f64(1.0);
        let y = f.const_f64(2.0);
        let s = f.emit(
            VirInstKind::StructCreate("Vec2".to_string(), vec![
                ("x".to_string(), x),
                ("y".to_string(), y),
            ]),
            VirType::Struct("Vec2".to_string(), vec![]),
        );
        let _gx = f.emit(VirInstKind::StructGet(s, "x".to_string()), VirType::F64);
        f.ret(Some(s));

        module.add_function(f);
        let text = module.to_text();
        assert!(text.contains("struct.create"));
        assert!(text.contains("struct.get"));
    }

    #[test]
    fn test_conv2d_and_reduce() {
        let mut module = VirModule::new("conv_test");
        let tensor4d = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![
                VirDim::Fixed(1),
                VirDim::Fixed(3),
                VirDim::Fixed(224),
                VirDim::Fixed(224),
            ],
        };
        let weight = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![
                VirDim::Fixed(64),
                VirDim::Fixed(3),
                VirDim::Fixed(3),
                VirDim::Fixed(3),
            ],
        };
        let out = VirType::Tensor {
            elem: Box::new(VirType::F32),
            shape: vec![
                VirDim::Fixed(1),
                VirDim::Fixed(64),
                VirDim::Fixed(222),
                VirDim::Fixed(222),
            ],
        };
        let mut f = VirFunction::new(
            "conv_reduce",
            vec![("input", tensor4d.clone()), ("w", weight)],
            VirType::Tensor {
                elem: Box::new(VirType::F32),
                shape: vec![VirDim::Fixed(1), VirDim::Fixed(64)],
            },
        );

        let conv = f.emit(
            VirInstKind::Conv2d {
                input: 0,
                weight: 1,
                stride: [1, 1],
                padding: [0, 0],
            },
            out,
        );
        let reduced = f.emit(
            VirInstKind::Reduce {
                tensor: conv,
                op: ReduceOp::Mean,
                axis: 2,
            },
            VirType::Tensor {
                elem: Box::new(VirType::F32),
                shape: vec![VirDim::Fixed(1), VirDim::Fixed(64)],
            },
        );
        f.ret(Some(reduced));
        module.add_function(f);

        let text = module.to_text();
        assert!(text.contains("conv2d"));
        assert!(text.contains("reduce.mean"));
        assert!(text.contains("stride=[1, 1]"));
    }
}
