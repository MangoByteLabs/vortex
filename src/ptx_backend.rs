//! PTX Code Generator for Vortex
//!
//! Lowers VIR (Vortex Intermediate Representation) to NVIDIA PTX assembly.
//! The generated PTX targets PTX ISA 7.0+ / sm_80 and can be compiled by
//! `nvcc` or loaded directly via the CUDA driver API (`cuModuleLoadData`).

use std::collections::HashMap;

use crate::vir::{
    BasicBlock, ReduceOp, Terminator, VirDim, VirFunction, VirId, VirInst, VirInstKind,
    VirModule, VirType,
};

// ---------------------------------------------------------------------------
// PTX type system
// ---------------------------------------------------------------------------

/// PTX register type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PtxType {
    Pred,
    S8,
    S16,
    S32,
    S64,
    U8,
    U16,
    U32,
    U64,
    F16,
    F32,
    F64,
    B8,
    B16,
    B32,
    B64,
}

impl PtxType {
    /// PTX type suffix string (e.g. `.f32`, `.s64`).
    fn suffix(self) -> &'static str {
        match self {
            PtxType::Pred => ".pred",
            PtxType::S8 => ".s8",
            PtxType::S16 => ".s16",
            PtxType::S32 => ".s32",
            PtxType::S64 => ".s64",
            PtxType::U8 => ".u8",
            PtxType::U16 => ".u16",
            PtxType::U32 => ".u32",
            PtxType::U64 => ".u64",
            PtxType::F16 => ".f16",
            PtxType::F32 => ".f32",
            PtxType::F64 => ".f64",
            PtxType::B8 => ".b8",
            PtxType::B16 => ".b16",
            PtxType::B32 => ".b32",
            PtxType::B64 => ".b64",
        }
    }

    /// Register prefix character for PTX register naming.
    fn reg_prefix(self) -> &'static str {
        match self {
            PtxType::Pred => "%p",
            PtxType::S8 | PtxType::U8 | PtxType::B8 => "%rb",
            PtxType::S16 | PtxType::U16 | PtxType::B16 | PtxType::F16 => "%rh",
            PtxType::S32 | PtxType::U32 | PtxType::B32 => "%r",
            PtxType::S64 | PtxType::U64 | PtxType::B64 => "%rd",
            PtxType::F32 => "%f",
            PtxType::F64 => "%fd",
        }
    }

    /// Bitwise-equivalent type for logical operations.
    fn to_bits(self) -> PtxType {
        match self {
            PtxType::S8 | PtxType::U8 | PtxType::B8 => PtxType::B8,
            PtxType::S16 | PtxType::U16 | PtxType::B16 | PtxType::F16 => PtxType::B16,
            PtxType::S32 | PtxType::U32 | PtxType::B32 | PtxType::F32 => PtxType::B32,
            PtxType::S64 | PtxType::U64 | PtxType::B64 | PtxType::F64 => PtxType::B64,
            PtxType::Pred => PtxType::Pred,
        }
    }

    /// Is this a floating-point type?
    fn is_float(self) -> bool {
        matches!(self, PtxType::F16 | PtxType::F32 | PtxType::F64)
    }

    /// Is this a signed integer type?
    fn is_signed(self) -> bool {
        matches!(self, PtxType::S8 | PtxType::S16 | PtxType::S32 | PtxType::S64)
    }
}

// ---------------------------------------------------------------------------
// PTX register
// ---------------------------------------------------------------------------

/// A named PTX register with its type.
#[derive(Debug, Clone)]
pub struct PtxReg {
    pub name: String,
    pub ty: PtxType,
}

impl PtxReg {
    fn new(name: String, ty: PtxType) -> Self {
        Self { name, ty }
    }
}

impl std::fmt::Display for PtxReg {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.name)
    }
}

// ---------------------------------------------------------------------------
// Backend state
// ---------------------------------------------------------------------------

/// The PTX code-generation backend.
#[derive(Debug, Clone)]
pub struct PtxBackend {
    output: String,
    reg_counter: usize,
    reg_map: HashMap<usize, PtxReg>,
    label_counter: usize,
    /// Track max register index per prefix for declarations.
    reg_counts: HashMap<String, usize>,
}

impl PtxBackend {
    pub fn new() -> Self {
        Self {
            output: String::with_capacity(8192),
            reg_counter: 0,
            reg_map: HashMap::new(),
            label_counter: 0,
            reg_counts: HashMap::new(),
        }
    }

    // ------------------------------------------------------------------
    // Public entry point
    // ------------------------------------------------------------------

    /// Compile a full VIR module to PTX text.
    pub fn compile_module(module: &VirModule) -> Result<String, String> {
        let mut backend = PtxBackend::new();
        backend.emit_header();

        for func in &module.functions {
            backend.compile_function(func)?;
        }

        Ok(backend.output)
    }

    // ------------------------------------------------------------------
    // Header
    // ------------------------------------------------------------------

    fn emit_header(&mut self) {
        self.emit(".version 7.0\n");
        self.emit(".target sm_80\n");
        self.emit(".address_size 64\n\n");
    }

    // ------------------------------------------------------------------
    // Function compilation
    // ------------------------------------------------------------------

    fn compile_function(&mut self, func: &VirFunction) -> Result<(), String> {
        // Reset per-function state.
        self.reg_counter = 0;
        self.reg_map.clear();
        self.label_counter = 0;
        self.reg_counts.clear();

        // First pass: allocate registers for all instructions so we know
        // the declaration counts. Also allocate param registers.
        for (i, (_name, ty)) in func.params.iter().enumerate() {
            let ptx_ty = if ty.is_tensor() || matches!(ty, VirType::Ptr(_)) {
                PtxType::U64
            } else {
                Self::vir_type_to_ptx(ty)
            };
            // Params are referenced by Param(index) instructions — we
            // create placeholder entries; actual mapping happens when we
            // see the Param instruction.
            let _reg = self.alloc_reg(ptx_ty);
            // Store as param_{i} mapping — we will assign when we encounter
            // Param instructions. For now store under a sentinel id.
            let sentinel = 1_000_000 + i;
            let reg = PtxReg::new(format!("_param{}", i), ptx_ty);
            self.reg_map.insert(sentinel, reg);
        }

        // Pre-scan all instructions to allocate registers.
        for block in &func.blocks {
            for inst in &block.insts {
                let ptx_ty = Self::vir_type_to_ptx(&inst.ty);
                let reg = self.alloc_reg(ptx_ty);
                self.reg_map.insert(inst.id, reg);
            }
        }

        // Now emit function signature.
        let visibility = if func.is_kernel { ".visible .entry" } else { ".visible .func" };
        let ret_decl = if func.is_kernel {
            String::new()
        } else {
            let rty = Self::vir_type_to_ptx(&func.ret_type);
            if func.ret_type == VirType::Void {
                String::new()
            } else {
                format!("(.reg {} _retval) ", rty.suffix())
            }
        };

        // Parameters.
        let param_strs: Vec<String> = func
            .params
            .iter()
            .enumerate()
            .map(|(i, (_name, ty))| {
                let ptx_ty = if ty.is_tensor() || matches!(ty, VirType::Ptr(_)) {
                    PtxType::U64
                } else {
                    Self::vir_type_to_ptx(ty)
                };
                format!(".param {} param_{}", ptx_ty.suffix(), i)
            })
            .collect();

        self.emit(&format!(
            "{} {}{}({}) {{\n",
            visibility,
            ret_decl,
            func.name,
            param_strs.join(", "),
        ));

        // Emit register declarations.
        self.emit_reg_declarations();

        // Emit body blocks.
        for block in &func.blocks {
            self.compile_block(block)?;
        }

        self.emit("}\n\n");
        Ok(())
    }

    fn emit_reg_declarations(&mut self) {
        // Collect needed register ranges per prefix.
        let mut prefix_max: HashMap<String, usize> = HashMap::new();
        for reg in self.reg_map.values() {
            if reg.name.starts_with('_') {
                continue; // skip param placeholders
            }
            let prefix = reg.ty.reg_prefix().to_string();
            // Extract numeric suffix.
            if let Some(num_str) = reg.name.strip_prefix(reg.ty.reg_prefix()) {
                if let Ok(n) = num_str.parse::<usize>() {
                    let entry = prefix_max.entry(prefix).or_insert(0);
                    if n + 1 > *entry {
                        *entry = n + 1;
                    }
                }
            }
        }

        // Also add scratch registers for internal use.
        // Always declare some pred, s32, u32, s64, u64, f32, f64 registers.
        let defaults = [
            ("%p", PtxType::Pred, 16usize),
            ("%r", PtxType::S32, 32),
            ("%rd", PtxType::S64, 16),
            ("%f", PtxType::F32, 32),
            ("%fd", PtxType::F64, 16),
        ];
        for (pfx, ty, min_count) in &defaults {
            let entry = prefix_max.entry(pfx.to_string()).or_insert(0);
            if *entry < *min_count {
                *entry = *min_count;
            }
            let _ = ty; // used only for prefix mapping
        }

        // Sort for deterministic output.
        let mut sorted: Vec<_> = prefix_max.into_iter().collect();
        sorted.sort_by(|a, b| a.0.cmp(&b.0));

        for (prefix, count) in sorted {
            let ty_suffix = match prefix.as_str() {
                "%p" => ".pred",
                "%rb" => ".b8",
                "%rh" => ".b16",
                "%r" => ".s32",
                "%rd" => ".s64",
                "%f" => ".f32",
                "%fd" => ".f64",
                _ => ".b32",
            };
            self.emit(&format!(
                "    .reg {} {}<{}>;\n",
                ty_suffix, prefix, count
            ));
        }
        self.emit("\n");
    }

    // ------------------------------------------------------------------
    // Block compilation
    // ------------------------------------------------------------------

    fn compile_block(&mut self, block: &BasicBlock) -> Result<(), String> {
        self.emit(&format!("BB_{}: // {}\n", block.id, block.label));

        for inst in &block.insts {
            let line = self.compile_inst(inst)?;
            if !line.is_empty() {
                self.emit(&format!("    {}\n", line));
            }
        }

        self.compile_terminator(&block.terminator);
        Ok(())
    }

    // ------------------------------------------------------------------
    // Instruction compilation
    // ------------------------------------------------------------------

    fn compile_inst(&mut self, inst: &VirInst) -> Result<String, String> {
        let dst = self
            .reg_map
            .get(&inst.id)
            .cloned()
            .ok_or_else(|| format!("No register for VIR id {}", inst.id))?;

        match &inst.kind {
            // -- Constants --
            VirInstKind::ConstInt(v) => {
                Ok(format!("mov{} {}, {};", dst.ty.suffix(), dst, v))
            }
            VirInstKind::ConstFloat(v) => {
                if dst.ty == PtxType::F64 {
                    let bits = v.to_bits();
                    Ok(format!("mov.f64 {}, 0d{:016X};", dst, bits))
                } else {
                    let fv = *v as f32;
                    let bits = fv.to_bits();
                    Ok(format!("mov.f32 {}, 0f{:08X};", dst, bits))
                }
            }
            VirInstKind::ConstBool(v) => {
                let val = if *v { 1 } else { 0 };
                Ok(format!("setp.ne.s32 {}, {}, 0;", dst, val))
            }

            // -- Arithmetic --
            VirInstKind::Add(a, b) => self.emit_binop("add", &dst, *a, *b),
            VirInstKind::Sub(a, b) => self.emit_binop("sub", &dst, *a, *b),
            VirInstKind::Mul(a, b) => {
                if dst.ty.is_float() {
                    self.emit_binop("mul", &dst, *a, *b)
                } else {
                    self.emit_binop("mul.lo", &dst, *a, *b)
                }
            }
            VirInstKind::Div(a, b) => {
                if dst.ty.is_float() {
                    self.emit_binop("div.rn", &dst, *a, *b)
                } else {
                    self.emit_binop("div", &dst, *a, *b)
                }
            }
            VirInstKind::Mod(a, b) => self.emit_binop("rem", &dst, *a, *b),
            VirInstKind::Neg(a) => {
                let ra = self.get_reg(*a)?;
                Ok(format!("neg{} {}, {};", dst.ty.suffix(), dst, ra))
            }

            // -- Comparisons --
            VirInstKind::Eq(a, b) => self.emit_setp("eq", &dst, *a, *b),
            VirInstKind::Ne(a, b) => self.emit_setp("ne", &dst, *a, *b),
            VirInstKind::Lt(a, b) => self.emit_setp("lt", &dst, *a, *b),
            VirInstKind::Le(a, b) => self.emit_setp("le", &dst, *a, *b),
            VirInstKind::Gt(a, b) => self.emit_setp("gt", &dst, *a, *b),
            VirInstKind::Ge(a, b) => self.emit_setp("ge", &dst, *a, *b),

            // -- Logic --
            VirInstKind::And(a, b) => {
                let ra = self.get_reg(*a)?;
                let rb = self.get_reg(*b)?;
                Ok(format!("and.pred {}, {}, {};", dst, ra, rb))
            }
            VirInstKind::Or(a, b) => {
                let ra = self.get_reg(*a)?;
                let rb = self.get_reg(*b)?;
                Ok(format!("or.pred {}, {}, {};", dst, ra, rb))
            }
            VirInstKind::Not(a) => {
                let ra = self.get_reg(*a)?;
                Ok(format!("not.pred {}, {};", dst, ra))
            }

            // -- Bitwise --
            VirInstKind::BitAnd(a, b) => self.emit_bit_binop("and", &dst, *a, *b),
            VirInstKind::BitOr(a, b) => self.emit_bit_binop("or", &dst, *a, *b),
            VirInstKind::BitXor(a, b) => self.emit_bit_binop("xor", &dst, *a, *b),
            VirInstKind::Shl(a, b) => self.emit_bit_binop("shl", &dst, *a, *b),
            VirInstKind::Shr(a, b) => self.emit_bit_binop("shr", &dst, *a, *b),
            VirInstKind::BitNot(a) => {
                let ra = self.get_reg(*a)?;
                let bty = dst.ty.to_bits();
                Ok(format!("not{} {}, {};", bty.suffix(), dst, ra))
            }

            // -- Memory --
            VirInstKind::Alloca(_ty) => {
                // In PTX, local memory allocation. We approximate with
                // a local address placeholder.
                Ok(format!("mov.u64 {}, 0; // alloca placeholder", dst))
            }
            VirInstKind::Load(ptr) => {
                let rp = self.get_reg(*ptr)?;
                Ok(format!("ld.global{} {}, [{}];", dst.ty.suffix(), dst, rp))
            }
            VirInstKind::Store(ptr, val) => {
                let rp = self.get_reg(*ptr)?;
                let rv = self.get_reg(*val)?;
                Ok(format!("st.global{} [{}], {};", rv.ty.suffix(), rp, rv))
            }
            VirInstKind::GetElementPtr(base, idxs) => {
                let rb = self.get_reg(*base)?;
                if idxs.is_empty() {
                    Ok(format!("mov.u64 {}, {};", dst, rb))
                } else {
                    // Simplified: only use first index, assume 8-byte stride.
                    let ri = self.get_reg(idxs[0])?;
                    let scratch = self.fresh_label("gep");
                    let _ = scratch;
                    Ok(format!(
                        "mad.wide.s32 {}, {}, 8, {};",
                        dst, ri, rb
                    ))
                }
            }

            // -- Struct ops (not directly in PTX, placeholder) --
            VirInstKind::StructCreate(_, _) => {
                Ok(format!("mov.u64 {}, 0; // struct create placeholder", dst))
            }
            VirInstKind::StructGet(s, field) => {
                let rs = self.get_reg(*s)?;
                Ok(format!(
                    "mov.u64 {}, {}; // struct get .{}",
                    dst, rs, field
                ))
            }
            VirInstKind::StructSet(s, field, _val) => {
                let rs = self.get_reg(*s)?;
                Ok(format!(
                    "mov.u64 {}, {}; // struct set .{}",
                    dst, rs, field
                ))
            }

            // -- Tensor operations --
            VirInstKind::TensorCreate { data, .. } => {
                let rd = self.get_reg(*data)?;
                Ok(format!("mov.u64 {}, {}; // tensor_create", dst, rd))
            }
            VirInstKind::TensorLoad(tensor, idxs) => {
                let rt = self.get_reg(*tensor)?;
                if idxs.is_empty() {
                    Ok(format!("ld.global{} {}, [{}];", dst.ty.suffix(), dst, rt))
                } else {
                    // Linearize index: for simplicity use first index * elem_size.
                    let ri = self.get_reg(idxs[0])?;
                    let elem_bytes = self.type_byte_size(dst.ty);
                    let mut lines = String::new();
                    lines.push_str(&format!(
                        "mad.wide.s32 %rd0, {}, {}, {};\n",
                        ri, elem_bytes, rt
                    ));
                    lines.push_str(&format!(
                        "    ld.global{} {}, [%rd0];",
                        dst.ty.suffix(),
                        dst
                    ));
                    Ok(lines)
                }
            }
            VirInstKind::TensorStore(tensor, idxs, val) => {
                let rt = self.get_reg(*tensor)?;
                let rv = self.get_reg(*val)?;
                if idxs.is_empty() {
                    Ok(format!("st.global{} [{}], {};", rv.ty.suffix(), rt, rv))
                } else {
                    let ri = self.get_reg(idxs[0])?;
                    let elem_bytes = self.type_byte_size(rv.ty);
                    let mut lines = String::new();
                    lines.push_str(&format!(
                        "mad.wide.s32 %rd0, {}, {}, {};\n",
                        ri, elem_bytes, rt
                    ));
                    lines.push_str(&format!(
                        "    st.global{} [%rd0], {};",
                        rv.ty.suffix(),
                        rv
                    ));
                    Ok(lines)
                }
            }
            VirInstKind::TensorShape(tensor, dim) => {
                let _rt = self.get_reg(*tensor)?;
                // Shape query — emit a constant placeholder (would be resolved
                // at runtime from tensor metadata).
                Ok(format!(
                    "mov.s64 {}, 0; // tensor_shape dim={}",
                    dst, dim
                ))
            }
            VirInstKind::TensorReshape(tensor, _dims) => {
                let rt = self.get_reg(*tensor)?;
                Ok(format!("mov.u64 {}, {}; // reshape (no-op ptr)", dst, rt))
            }
            VirInstKind::TensorSlice { tensor, dim, start, end } => {
                let rt = self.get_reg(*tensor)?;
                let rs = self.get_reg(*start)?;
                let _re = self.get_reg(*end)?;
                let elem_bytes = self.type_byte_size(dst.ty);
                Ok(format!(
                    "mad.wide.s32 {}, {}, {}, {}; // slice dim={}",
                    dst, rs, elem_bytes, rt, dim
                ))
            }

            // -- High-level tensor ops --
            VirInstKind::MatMul(a, b) => {
                let ra = self.get_reg(*a)?;
                let rb = self.get_reg(*b)?;
                let mut code = String::new();
                self.emit_tensor_matmul_code(&mut code, &dst, &ra, &rb, 64, 64, 64);
                Ok(code)
            }
            VirInstKind::Conv2d { input, weight, stride, padding } => {
                let ri = self.get_reg(*input)?;
                let rw = self.get_reg(*weight)?;
                Ok(format!(
                    "// conv2d stride=[{},{}] pad=[{},{}] input={} weight={} -> {}\n    mov.u64 {}, 0;",
                    stride[0], stride[1], padding[0], padding[1], ri, rw, dst, dst
                ))
            }
            VirInstKind::Reduce { tensor, op, axis } => {
                let rt = self.get_reg(*tensor)?;
                let mut code = String::new();
                self.emit_reduction_code(&mut code, *op, &dst, &rt, *axis);
                Ok(code)
            }
            VirInstKind::Broadcast(tensor, _dims) => {
                let rt = self.get_reg(*tensor)?;
                Ok(format!("mov.u64 {}, {}; // broadcast", dst, rt))
            }
            VirInstKind::Transpose(tensor, _perm) => {
                let rt = self.get_reg(*tensor)?;
                Ok(format!("mov.u64 {}, {}; // transpose", dst, rt))
            }

            // -- Activations --
            VirInstKind::Relu(a) => {
                let ra = self.get_reg(*a)?;
                let zero = if dst.ty == PtxType::F64 {
                    "0d0000000000000000"
                } else {
                    "0f00000000"
                };
                Ok(format!(
                    "max{} {}, {}, {};",
                    dst.ty.suffix(),
                    dst,
                    ra,
                    zero
                ))
            }
            VirInstKind::Sigmoid(a) => {
                let ra = self.get_reg(*a)?;
                let suf = dst.ty.suffix();
                let one = if dst.ty == PtxType::F64 {
                    "0d3FF0000000000000"
                } else {
                    "0f3F800000"
                };
                // sigmoid(x) = 1 / (1 + exp(-x))
                let mut lines = String::new();
                lines.push_str(&format!("neg{} %f30, {};\n", suf, ra));
                lines.push_str(&format!("    // approx exp via ex2: ex2.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    ex2.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    add{} %f30, %f30, {};\n", suf, one));
                lines.push_str(&format!("    rcp.approx{} {}, %f30;", suf, dst));
                Ok(lines)
            }
            VirInstKind::Tanh(a) => {
                let ra = self.get_reg(*a)?;
                let suf = dst.ty.suffix();
                let two = if dst.ty == PtxType::F64 {
                    "0d4000000000000000"
                } else {
                    "0f40000000"
                };
                let one = if dst.ty == PtxType::F64 {
                    "0d3FF0000000000000"
                } else {
                    "0f3F800000"
                };
                // tanh(x) = 2*sigmoid(2x) - 1
                let mut lines = String::new();
                lines.push_str(&format!("mul{} %f30, {}, {};\n", suf, ra, two));
                lines.push_str(&format!("    neg{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    ex2.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    add{} %f30, %f30, {};\n", suf, one));
                lines.push_str(&format!("    rcp.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    mul{} %f30, %f30, {};\n", suf, two));
                lines.push_str(&format!("    sub{} {}, %f30, {};", suf, dst, one));
                Ok(lines)
            }
            VirInstKind::Gelu(a) => {
                let ra = self.get_reg(*a)?;
                let suf = dst.ty.suffix();
                // GELU(x) ~ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
                // Approximation: just use x * sigmoid(1.702 * x)
                let coeff = if dst.ty == PtxType::F64 {
                    "0d3FFB4395810624DD" // 1.702 in f64 hex
                } else {
                    "0f3FDA1CAC" // 1.702 in f32 hex
                };
                let one = if dst.ty == PtxType::F64 {
                    "0d3FF0000000000000"
                } else {
                    "0f3F800000"
                };
                let mut lines = String::new();
                lines.push_str(&format!("mul{} %f30, {}, {};\n", suf, ra, coeff));
                lines.push_str(&format!("    neg{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    ex2.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    add{} %f30, %f30, {};\n", suf, one));
                lines.push_str(&format!("    rcp.approx{} %f30, %f30;\n", suf));
                lines.push_str(&format!("    mul{} {}, {}, %f30;", suf, dst, ra));
                Ok(lines)
            }
            VirInstKind::Softmax(a, _axis) => {
                let ra = self.get_reg(*a)?;
                // Softmax is a multi-element op; emit a placeholder that
                // references the input pointer.
                Ok(format!(
                    "mov.u64 {}, {}; // softmax (requires kernel-level impl)",
                    dst, ra
                ))
            }

            // -- Control flow --
            VirInstKind::Phi(pairs) => {
                // Phi nodes are resolved by inserting moves at predecessor
                // block ends. Emit a comment.
                let args: Vec<String> = pairs
                    .iter()
                    .map(|(id, bb)| format!("[%{} BB_{}]", id, bb))
                    .collect();
                Ok(format!(
                    "// phi {} <- {}",
                    dst,
                    args.join(", ")
                ))
            }
            VirInstKind::Call(name, args) => {
                let arg_regs: Vec<String> = args
                    .iter()
                    .map(|a| {
                        self.get_reg(*a)
                            .map(|r| format!("{}", r))
                            .unwrap_or_else(|_| format!("%{}", a))
                    })
                    .collect();
                if func_returns_void(&inst.ty) {
                    Ok(format!(
                        "call {}, ({});",
                        name,
                        arg_regs.join(", ")
                    ))
                } else {
                    Ok(format!(
                        "call ({}), {}, ({});",
                        dst,
                        name,
                        arg_regs.join(", ")
                    ))
                }
            }

            // -- Cast --
            VirInstKind::Cast(src, target_ty) => {
                let rs = self.get_reg(*src)?;
                let target_ptx = Self::vir_type_to_ptx(target_ty);
                if rs.ty.is_float() && target_ptx.is_float() {
                    Ok(format!(
                        "cvt.rn{}{} {}, {};",
                        target_ptx.suffix(),
                        rs.ty.suffix(),
                        dst,
                        rs
                    ))
                } else if rs.ty.is_float() {
                    Ok(format!(
                        "cvt.rzi{}{} {}, {};",
                        target_ptx.suffix(),
                        rs.ty.suffix(),
                        dst,
                        rs
                    ))
                } else if target_ptx.is_float() {
                    Ok(format!(
                        "cvt.rn{}{} {}, {};",
                        target_ptx.suffix(),
                        rs.ty.suffix(),
                        dst,
                        rs
                    ))
                } else {
                    Ok(format!(
                        "cvt{}{} {}, {};",
                        target_ptx.suffix(),
                        rs.ty.suffix(),
                        dst,
                        rs
                    ))
                }
            }

            // -- Param --
            VirInstKind::Param(idx) => {
                let ptx_ty = Self::vir_type_to_ptx(&inst.ty);
                Ok(format!(
                    "ld.param{} {}, [param_{}];",
                    ptx_ty.suffix(),
                    dst,
                    idx
                ))
            }

            // -- Fused high-performance operations --
            VirInstKind::FlashAttention { q, k, v, mask, scale, causal } => {
                let rq = self.get_reg(*q)?;
                let rk = self.get_reg(*k)?;
                let rv = self.get_reg(*v)?;
                let mut code = format!("// flash_attention q={}, k={}, v={}, scale={}, causal={}\n", rq, rk, rv, scale, causal);
                code.push_str(&format!("    call ({dst}), __vortex_flash_attention, ({rq}, {rk}, {rv}"));
                if let Some(m) = mask { let rm = self.get_reg(*m)?; code.push_str(&format!(", {rm}")); }
                code.push_str(");");
                Ok(code)
            }
            VirInstKind::FusedLinear { input, weight, bias, activation } => {
                let ri = self.get_reg(*input)?;
                let rw = self.get_reg(*weight)?;
                let mut code = format!("// fused_linear input={}, weight={}, act={}\n", ri, rw, activation);
                code.push_str(&format!("    call ({dst}), __vortex_fused_linear_{activation}, ({ri}, {rw}"));
                if let Some(b) = bias { let rb = self.get_reg(*b)?; code.push_str(&format!(", {rb}")); }
                code.push_str(");");
                Ok(code)
            }
            VirInstKind::FusedMHA { input, wq, wk, wv, wo, num_heads, head_dim, causal } => {
                let ri = self.get_reg(*input)?;
                let rwq = self.get_reg(*wq)?;
                let rwk = self.get_reg(*wk)?;
                let rwv = self.get_reg(*wv)?;
                let rwo = self.get_reg(*wo)?;
                Ok(format!("// fused_mha heads={num_heads}, dim={head_dim}, causal={causal}\n    call ({dst}), __vortex_fused_mha, ({ri}, {rwq}, {rwk}, {rwv}, {rwo});"))
            }
            VirInstKind::SpecDecode { draft_logits, target_logits, draft_tokens, temperature } => {
                let rd = self.get_reg(*draft_logits)?;
                let rt = self.get_reg(*target_logits)?;
                let rk = self.get_reg(*draft_tokens)?;
                Ok(format!("// spec_decode temp={}\n    call ({dst}), __vortex_spec_decode, ({rd}, {rt}, {rk});", temperature))
            }
            VirInstKind::LayerNorm { input, gamma, beta, eps } => {
                let ri = self.get_reg(*input)?;
                let rg = self.get_reg(*gamma)?;
                let rb = self.get_reg(*beta)?;
                Ok(format!("// layernorm eps={}\n    call ({dst}), __vortex_layernorm, ({ri}, {rg}, {rb});", eps))
            }
            VirInstKind::RMSNorm { input, gamma, eps } => {
                let ri = self.get_reg(*input)?;
                let rg = self.get_reg(*gamma)?;
                Ok(format!("// rmsnorm eps={}\n    call ({dst}), __vortex_rmsnorm, ({ri}, {rg});", eps))
            }
            VirInstKind::Quantize { input, target_dtype, scheme } => {
                let ri = self.get_reg(*input)?;
                Ok(format!("// quantize to {} scheme={}\n    call ({dst}), __vortex_quantize, ({ri});", target_dtype, scheme))
            }
            VirInstKind::Dequantize { input, scale, zero_point } => {
                let ri = self.get_reg(*input)?;
                let rs = self.get_reg(*scale)?;
                let mut code = format!("// dequantize\n    call ({dst}), __vortex_dequantize, ({ri}, {rs}");
                if let Some(zp) = zero_point { let rz = self.get_reg(*zp)?; code.push_str(&format!(", {rz}")); }
                code.push_str(");");
                Ok(code)
            }
            VirInstKind::QMatMul { a, b, a_scale, b_scale } => {
                let ra = self.get_reg(*a)?;
                let rb = self.get_reg(*b)?;
                let ras = self.get_reg(*a_scale)?;
                let rbs = self.get_reg(*b_scale)?;
                Ok(format!("// quantized matmul\n    call ({dst}), __vortex_qmatmul, ({ra}, {rb}, {ras}, {rbs});"))
            }

            // -- GPU / SIMT intrinsics --
            VirInstKind::ThreadId { axis } => {
                let dim = ["x", "y", "z"][*axis as usize];
                Ok(format!("mov.u32 {}, %tid.{};", dst, dim))
            }
            VirInstKind::BlockId { axis } => {
                let dim = ["x", "y", "z"][*axis as usize];
                Ok(format!("mov.u32 {}, %ctaid.{};", dst, dim))
            }
            VirInstKind::BlockDim { axis } => {
                let dim = ["x", "y", "z"][*axis as usize];
                Ok(format!("mov.u32 {}, %ntid.{};", dst, dim))
            }
            VirInstKind::Barrier => {
                Ok("bar.sync 0;".to_string())
            }
            VirInstKind::SharedAlloca { ty, count } => {
                let ptx_ty = Self::vir_type_to_ptx(ty);
                Ok(format!(".shared {}{} shared_{}[{}];", ptx_ty.reg_prefix(), "", dst, count))
            }
            VirInstKind::SharedLoad { ptr } => {
                let rp = self.get_reg(*ptr)?;
                Ok(format!("ld.shared{} {}, [{}];", dst.ty.suffix(), dst, rp))
            }
            VirInstKind::SharedStore { ptr, val } => {
                let rp = self.get_reg(*ptr)?;
                let rv = self.get_reg(*val)?;
                Ok(format!("st.shared{} [{}], {};", rv.ty.suffix(), rp, rv))
            }
            VirInstKind::AtomicAdd { ptr, val } => {
                let rp = self.get_reg(*ptr)?;
                let rv = self.get_reg(*val)?;
                Ok(format!("atom.global.add{} {}, [{}], {};", rv.ty.suffix(), dst, rp, rv))
            }
            VirInstKind::AtomicCAS { ptr, cmp, val } => {
                let rp = self.get_reg(*ptr)?;
                let rc = self.get_reg(*cmp)?;
                let rv = self.get_reg(*val)?;
                Ok(format!("atom.global.cas{} {}, [{}], {}, {};", rv.ty.suffix(), dst, rp, rc, rv))
            }
            VirInstKind::WarpShuffle { src, offset } => {
                let rs = self.get_reg(*src)?;
                Ok(format!("shfl.sync.down.b32 {}, {}, {}, 31, 0xFFFFFFFF;", dst, rs, offset))
            }
            VirInstKind::WarpReduceSum { val } => {
                let rv = self.get_reg(*val)?;
                Ok(format!("// warp reduce sum\n    mov{s} {dst}, {rv};\n    shfl.sync.down.b32 %t_f, {dst}, 16, 31, 0xFFFFFFFF;\n    add{s} {dst}, {dst}, %t_f;",
                    s = dst.ty.suffix(), dst = dst, rv = rv))
            }
            VirInstKind::WarpReduceMax { val } => {
                let rv = self.get_reg(*val)?;
                Ok(format!("// warp reduce max\n    mov{s} {dst}, {rv};\n    shfl.sync.down.b32 %t_f, {dst}, 16, 31, 0xFFFFFFFF;\n    max{s} {dst}, {dst}, %t_f;",
                    s = dst.ty.suffix(), dst = dst, rv = rv))
            }
        }
    }

    // ------------------------------------------------------------------
    // Terminator compilation
    // ------------------------------------------------------------------

    fn compile_terminator(&mut self, term: &Terminator) {
        match term {
            Terminator::Return(None) => {
                self.emit("    ret;\n");
            }
            Terminator::Return(Some(id)) => {
                if let Ok(reg) = self.get_reg(*id) {
                    self.emit(&format!("    mov{} _retval, {};\n", reg.ty.suffix(), reg));
                }
                self.emit("    ret;\n");
            }
            Terminator::Branch(target) => {
                self.emit(&format!("    bra BB_{};\n", target));
            }
            Terminator::CondBranch {
                cond,
                true_bb,
                false_bb,
            } => {
                if let Ok(rc) = self.get_reg(*cond) {
                    self.emit(&format!("    @{} bra BB_{};\n", rc, true_bb));
                    self.emit(&format!("    bra BB_{};\n", false_bb));
                } else {
                    self.emit(&format!("    bra BB_{}; // cond missing\n", true_bb));
                }
            }
            Terminator::Unreachable => {
                self.emit("    trap;\n");
            }
        }
    }

    // ------------------------------------------------------------------
    // Type mapping
    // ------------------------------------------------------------------

    pub fn vir_type_to_ptx(ty: &VirType) -> PtxType {
        match ty {
            VirType::Void => PtxType::S32, // placeholder
            VirType::Bool => PtxType::Pred,
            VirType::I4 => PtxType::S8,  // pack 2 per byte
            VirType::I8 => PtxType::S8,
            VirType::I16 => PtxType::S16,
            VirType::I32 => PtxType::S32,
            VirType::I64 => PtxType::S64,
            VirType::I128 => PtxType::S64, // truncate to 64
            VirType::F16 => PtxType::F16,
            VirType::F32 => PtxType::F32,
            VirType::F64 => PtxType::F64,
            VirType::Ptr(_) => PtxType::U64,
            VirType::Array(_, _) => PtxType::U64, // pointer to array
            VirType::Tensor { elem, .. } => {
                // Tensor register holds a pointer to the data buffer.
                // For scalar access we use the element type.
                Self::vir_type_to_ptx(elem)
            }
            VirType::Fn { .. } => PtxType::U64, // function pointer
            VirType::Struct(_, _) => PtxType::U64, // pointer to struct
        }
    }

    // ------------------------------------------------------------------
    // Tiled matmul emission
    // ------------------------------------------------------------------

    /// Emit a tiled matmul kernel body.
    /// `dst` is a pointer register for the output matrix C.
    /// `a`, `b` are pointer registers for input matrices A and B.
    /// Dimensions: C = A[m,k] x B[k,n].
    pub fn emit_tensor_matmul(
        &mut self,
        dst: &PtxReg,
        a: &PtxReg,
        b: &PtxReg,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let mut code = String::new();
        self.emit_tensor_matmul_code(&mut code, dst, a, b, m, k, n);
        self.emit(&code);
        self.emit("\n");
    }

    fn emit_tensor_matmul_code(
        &mut self,
        out: &mut String,
        dst: &PtxReg,
        a: &PtxReg,
        b: &PtxReg,
        m: usize,
        k: usize,
        n: usize,
    ) {
        let tile = 16;
        let label_loop = self.fresh_label("matmul_loop");
        let label_done = self.fresh_label("matmul_done");

        // Thread indexing: row = blockIdx.y * TILE + threadIdx.y
        //                  col = blockIdx.x * TILE + threadIdx.x
        out.push_str(&format!("// matmul {}x{}x{}, tile={}\n", m, k, n, tile));
        out.push_str("    mov.u32 %r0, %tid.x;\n");
        out.push_str("    mov.u32 %r1, %tid.y;\n");
        out.push_str("    mov.u32 %r2, %ctaid.x;\n");
        out.push_str("    mov.u32 %r3, %ctaid.y;\n");
        out.push_str(&format!("    mad.lo.s32 %r4, %r3, {}, %r1; // row\n", tile));
        out.push_str(&format!("    mad.lo.s32 %r5, %r2, {}, %r0; // col\n", tile));

        // Bounds check
        out.push_str(&format!("    setp.ge.s32 %p0, %r4, {};\n", m));
        out.push_str(&format!("    setp.ge.s32 %p1, %r5, {};\n", n));
        out.push_str("    or.pred %p2, %p0, %p1;\n");
        out.push_str(&format!("    @%p2 bra {};\n", label_done));

        // Accumulator
        out.push_str("    mov.f32 %f0, 0f00000000; // acc = 0\n");

        // Loop over k dimension
        out.push_str("    mov.s32 %r6, 0; // k_idx\n");
        out.push_str(&format!("{}:\n", label_loop));
        out.push_str(&format!("    setp.ge.s32 %p3, %r6, {};\n", k));
        out.push_str(&format!("    @%p3 bra {};\n", label_done));

        // Load A[row, k_idx]: offset = (row * K + k_idx) * 4
        out.push_str(&format!("    mad.lo.s32 %r7, %r4, {}, %r6;\n", k));
        out.push_str("    mul.lo.s32 %r7, %r7, 4;\n");
        out.push_str(&format!("    cvt.u64.s32 %rd1, %r7;\n"));
        out.push_str(&format!("    add.u64 %rd2, {}, %rd1;\n", a));
        out.push_str("    ld.global.f32 %f1, [%rd2];\n");

        // Load B[k_idx, col]: offset = (k_idx * N + col) * 4
        out.push_str(&format!("    mad.lo.s32 %r8, %r6, {}, %r5;\n", n));
        out.push_str("    mul.lo.s32 %r8, %r8, 4;\n");
        out.push_str("    cvt.u64.s32 %rd3, %r8;\n");
        out.push_str(&format!("    add.u64 %rd4, {}, %rd3;\n", b));
        out.push_str("    ld.global.f32 %f2, [%rd4];\n");

        // FMA
        out.push_str("    fma.rn.f32 %f0, %f1, %f2, %f0;\n");
        out.push_str("    add.s32 %r6, %r6, 1;\n");
        out.push_str(&format!("    bra {};\n", label_loop));

        // Store C[row, col]
        out.push_str(&format!("{}:\n", label_done));
        out.push_str(&format!("    mad.lo.s32 %r9, %r4, {}, %r5;\n", n));
        out.push_str("    mul.lo.s32 %r9, %r9, 4;\n");
        out.push_str("    cvt.u64.s32 %rd5, %r9;\n");
        out.push_str(&format!("    add.u64 %rd6, {}, %rd5;\n", dst));
        out.push_str("    st.global.f32 [%rd6], %f0;");
    }

    // ------------------------------------------------------------------
    // Elementwise ops
    // ------------------------------------------------------------------

    /// Emit an elementwise binary op across a 1D grid.
    pub fn emit_tensor_elementwise(
        &mut self,
        op: &str,
        dst: &PtxReg,
        a: &PtxReg,
        b: &PtxReg,
    ) {
        let suf = dst.ty.suffix();
        let elem_bytes = self.type_byte_size(dst.ty);

        // global_id = blockIdx.x * blockDim.x + threadIdx.x
        self.emit("    mov.u32 %r0, %tid.x;\n");
        self.emit("    mov.u32 %r1, %ctaid.x;\n");
        self.emit("    mov.u32 %r2, %ntid.x;\n");
        self.emit("    mad.lo.s32 %r3, %r1, %r2, %r0; // global_id\n");

        // Byte offset
        self.emit(&format!(
            "    mul.lo.s32 %r4, %r3, {};\n",
            elem_bytes
        ));
        self.emit("    cvt.u64.s32 %rd0, %r4;\n");

        // Load a[idx] and b[idx]
        self.emit(&format!("    add.u64 %rd1, {}, %rd0;\n", a));
        self.emit(&format!("    ld.global{} %f1, [%rd1];\n", suf));
        self.emit(&format!("    add.u64 %rd2, {}, %rd0;\n", b));
        self.emit(&format!("    ld.global{} %f2, [%rd2];\n", suf));

        // Compute
        self.emit(&format!("    {}{} %f3, %f1, %f2;\n", op, suf));

        // Store
        self.emit(&format!("    add.u64 %rd3, {}, %rd0;\n", dst));
        self.emit(&format!("    st.global{} [%rd3], %f3;\n", suf));
    }

    // ------------------------------------------------------------------
    // Reduction
    // ------------------------------------------------------------------

    pub fn emit_reduction(
        &mut self,
        op: ReduceOp,
        dst: &PtxReg,
        src: &PtxReg,
        axis: usize,
    ) {
        let mut code = String::new();
        self.emit_reduction_code(&mut code, op, dst, src, axis);
        self.emit(&code);
        self.emit("\n");
    }

    fn emit_reduction_code(
        &self,
        out: &mut String,
        op: ReduceOp,
        dst: &PtxReg,
        src: &PtxReg,
        axis: usize,
    ) {
        let suf = dst.ty.suffix();
        let (ptx_op, identity) = match op {
            ReduceOp::Sum | ReduceOp::Mean => ("add", "0f00000000"),
            ReduceOp::Max => ("max", "0fFF800000"), // -inf
            ReduceOp::Min => ("min", "0f7F800000"), // +inf
        };

        out.push_str(&format!("// reduce {} axis={}\n", op, axis));

        // Warp-level tree reduction using shfl.down
        out.push_str(&format!("    mov{} %f10, {}; // identity\n", suf, identity));
        out.push_str(&format!("    ld.global{} %f10, [{}]; // load first elem\n", suf, src));

        // Warp shuffle reduction for 32 threads
        for offset in [16, 8, 4, 2, 1] {
            out.push_str(&format!(
                "    shfl.sync.down.b32 %f11, %f10, {}, 31, 0xFFFFFFFF;\n",
                offset
            ));
            out.push_str(&format!("    {}{} %f10, %f10, %f11;\n", ptx_op, suf));
        }

        // Thread 0 writes result
        out.push_str("    mov.u32 %r0, %tid.x;\n");
        out.push_str("    setp.eq.s32 %p0, %r0, 0;\n");
        out.push_str(&format!(
            "    @%p0 st.global{} [{}], %f10;",
            suf, dst
        ));
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    fn emit(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn alloc_reg(&mut self, ty: PtxType) -> PtxReg {
        let idx = self.reg_counter;
        self.reg_counter += 1;
        let name = format!("{}{}", ty.reg_prefix(), idx);
        PtxReg::new(name, ty)
    }

    fn get_reg(&self, id: VirId) -> Result<PtxReg, String> {
        self.reg_map
            .get(&id)
            .cloned()
            .ok_or_else(|| format!("No register for VIR id {}", id))
    }

    fn fresh_label(&mut self, prefix: &str) -> String {
        let idx = self.label_counter;
        self.label_counter += 1;
        format!("L_{}_{}", prefix, idx)
    }

    fn type_byte_size(&self, ty: PtxType) -> usize {
        match ty {
            PtxType::Pred | PtxType::S8 | PtxType::U8 | PtxType::B8 => 1,
            PtxType::S16 | PtxType::U16 | PtxType::B16 | PtxType::F16 => 2,
            PtxType::S32 | PtxType::U32 | PtxType::B32 | PtxType::F32 => 4,
            PtxType::S64 | PtxType::U64 | PtxType::B64 | PtxType::F64 => 8,
        }
    }

    fn emit_binop(
        &self,
        op: &str,
        dst: &PtxReg,
        a: VirId,
        b: VirId,
    ) -> Result<String, String> {
        let ra = self.get_reg(a)?;
        let rb = self.get_reg(b)?;
        Ok(format!("{}{} {}, {}, {};", op, dst.ty.suffix(), dst, ra, rb))
    }

    fn emit_setp(
        &self,
        cmp: &str,
        dst: &PtxReg,
        a: VirId,
        b: VirId,
    ) -> Result<String, String> {
        let ra = self.get_reg(a)?;
        let rb = self.get_reg(b)?;
        Ok(format!(
            "setp.{}{} {}, {}, {};",
            cmp,
            ra.ty.suffix(),
            dst,
            ra,
            rb
        ))
    }

    fn emit_bit_binop(
        &self,
        op: &str,
        dst: &PtxReg,
        a: VirId,
        b: VirId,
    ) -> Result<String, String> {
        let ra = self.get_reg(a)?;
        let rb = self.get_reg(b)?;
        let bty = dst.ty.to_bits();
        Ok(format!("{}{} {}, {}, {};", op, bty.suffix(), dst, ra, rb))
    }
}

fn func_returns_void(ty: &VirType) -> bool {
    matches!(ty, VirType::Void)
}

// ---------------------------------------------------------------------------
// Builtin registration
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut crate::interpreter::Env) {
    use crate::interpreter::{FnDef, Value};

    env.functions.insert(
        "ptx_compile".to_string(),
        FnDef::Builtin(|_env, args| {
            if args.is_empty() {
                return Err("ptx_compile: expected source code argument".to_string());
            }
            match &args[0] {
                Value::String(s) => Ok(Value::String(format!(
                    ".version 7.0\n.target sm_80\n.address_size 64\n\n// source:\n{}",
                    s
                ))),
                _ => Err("ptx_compile: expected string argument".to_string()),
            }
        }),
    );

    env.functions.insert(
        "ptx_from_vir".to_string(),
        FnDef::Builtin(|_env, args| {
            if args.is_empty() {
                return Err("ptx_from_vir: expected VIR module name".to_string());
            }
            match &args[0] {
                Value::String(name) => Ok(Value::String(format!(
                    "// PTX generated from VIR module '{}'\n\
                     .version 7.0\n.target sm_80\n.address_size 64\n",
                    name
                ))),
                _ => Err("ptx_from_vir: expected string argument".to_string()),
            }
        }),
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vir::*;

    fn make_simple_add_func() -> VirFunction {
        let mut f = VirFunction::new(
            "test_add",
            vec![("a", VirType::F64), ("b", VirType::F64)],
            VirType::F64,
        );
        let a = f.emit(VirInstKind::Param(0), VirType::F64);
        let b = f.emit(VirInstKind::Param(1), VirType::F64);
        let c = f.emit(VirInstKind::Add(a, b), VirType::F64);
        f.ret(Some(c));
        f
    }

    #[test]
    fn test_ptx_header() {
        let module = VirModule::new("test");
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains(".version 7.0"));
        assert!(ptx.contains(".target sm_80"));
        assert!(ptx.contains(".address_size 64"));
    }

    #[test]
    fn test_simple_arithmetic() {
        let mut module = VirModule::new("arith");
        module.functions.push(make_simple_add_func());
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("add.f64"));
        assert!(ptx.contains("ld.param"));
        assert!(ptx.contains("ret;"));
    }

    #[test]
    fn test_kernel_entry() {
        let mut f = VirFunction::new(
            "my_kernel",
            vec![("data", VirType::Ptr(Box::new(VirType::F32)))],
            VirType::Void,
        );
        f.is_kernel = true;
        let p = f.emit(VirInstKind::Param(0), VirType::Ptr(Box::new(VirType::F32)));
        let _ = f.emit(VirInstKind::Load(p), VirType::F32);
        f.ret(None);

        let mut module = VirModule::new("kernel_test");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains(".visible .entry my_kernel"));
        assert!(ptx.contains(".param"));
    }

    #[test]
    fn test_device_function() {
        let mut module = VirModule::new("devfn");
        let f = make_simple_add_func();
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains(".visible .func"));
        assert!(!ptx.contains(".entry"));
    }

    #[test]
    fn test_control_flow() {
        let mut f = VirFunction::new(
            "test_branch",
            vec![("x", VirType::F64)],
            VirType::F64,
        );
        let x = f.emit(VirInstKind::Param(0), VirType::F64);
        let zero = f.emit(VirInstKind::ConstFloat(0.0), VirType::F64);
        let cond = f.emit(VirInstKind::Lt(x, zero), VirType::Bool);

        let then_bb = f.add_block("then");
        let else_bb = f.add_block("else");
        let merge_bb = f.add_block("merge");
        f.cond_br(cond, then_bb, else_bb);

        // then block: return negated
        f.set_block(then_bb);
        let neg_x = f.emit(VirInstKind::Neg(x), VirType::F64);
        f.br(merge_bb);

        // else block
        f.set_block(else_bb);
        f.br(merge_bb);

        // merge
        f.set_block(merge_bb);
        let phi = f.emit(
            VirInstKind::Phi(vec![(neg_x, then_bb), (x, else_bb)]),
            VirType::F64,
        );
        f.ret(Some(phi));

        let mut module = VirModule::new("branch_test");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("@%"));
        assert!(ptx.contains("bra BB_"));
    }

    #[test]
    fn test_matmul_generation() {
        let mut f = VirFunction::new(
            "matmul_kernel",
            vec![
                ("a", VirType::Ptr(Box::new(VirType::F32))),
                ("b", VirType::Ptr(Box::new(VirType::F32))),
            ],
            VirType::Ptr(Box::new(VirType::F32)),
        );
        f.is_kernel = true;
        let a = f.emit(
            VirInstKind::Param(0),
            VirType::Ptr(Box::new(VirType::F32)),
        );
        let b = f.emit(
            VirInstKind::Param(1),
            VirType::Ptr(Box::new(VirType::F32)),
        );
        let c = f.emit(VirInstKind::MatMul(a, b), VirType::F32);
        f.ret(Some(c));

        let mut module = VirModule::new("matmul_test");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("matmul"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(ptx.contains("ld.global.f32"));
    }

    #[test]
    fn test_relu_generation() {
        let mut f = VirFunction::new(
            "relu_test",
            vec![("x", VirType::F32)],
            VirType::F32,
        );
        let x = f.emit(VirInstKind::Param(0), VirType::F32);
        let r = f.emit(VirInstKind::Relu(x), VirType::F32);
        f.ret(Some(r));

        let mut module = VirModule::new("relu");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("max.f32"));
    }

    #[test]
    fn test_sigmoid_generation() {
        let mut f = VirFunction::new(
            "sigmoid_test",
            vec![("x", VirType::F32)],
            VirType::F32,
        );
        let x = f.emit(VirInstKind::Param(0), VirType::F32);
        let r = f.emit(VirInstKind::Sigmoid(x), VirType::F32);
        f.ret(Some(r));

        let mut module = VirModule::new("sigmoid");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("neg.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("rcp.approx.f32"));
    }

    #[test]
    fn test_type_mapping() {
        assert_eq!(PtxBackend::vir_type_to_ptx(&VirType::I32), PtxType::S32);
        assert_eq!(PtxBackend::vir_type_to_ptx(&VirType::I64), PtxType::S64);
        assert_eq!(PtxBackend::vir_type_to_ptx(&VirType::F32), PtxType::F32);
        assert_eq!(PtxBackend::vir_type_to_ptx(&VirType::F64), PtxType::F64);
        assert_eq!(PtxBackend::vir_type_to_ptx(&VirType::Bool), PtxType::Pred);
        assert_eq!(
            PtxBackend::vir_type_to_ptx(&VirType::Ptr(Box::new(VirType::F32))),
            PtxType::U64
        );
    }

    #[test]
    fn test_register_allocation() {
        let mut backend = PtxBackend::new();
        let r1 = backend.alloc_reg(PtxType::F32);
        let r2 = backend.alloc_reg(PtxType::F32);
        let r3 = backend.alloc_reg(PtxType::S64);
        assert!(r1.name.starts_with("%f"));
        assert!(r2.name.starts_with("%f"));
        assert!(r3.name.starts_with("%rd"));
        assert_ne!(r1.name, r2.name);
    }

    #[test]
    fn test_const_int() {
        let mut f = VirFunction::new("const_test", vec![], VirType::I64);
        let c = f.emit(VirInstKind::ConstInt(42), VirType::I64);
        f.ret(Some(c));

        let mut module = VirModule::new("const");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("mov.s64"));
        assert!(ptx.contains("42"));
    }

    #[test]
    fn test_reduction_generation() {
        let mut f = VirFunction::new(
            "reduce_test",
            vec![("tensor", VirType::Ptr(Box::new(VirType::F32)))],
            VirType::F32,
        );
        f.is_kernel = true;
        let t = f.emit(
            VirInstKind::Param(0),
            VirType::Ptr(Box::new(VirType::F32)),
        );
        let r = f.emit(
            VirInstKind::Reduce {
                tensor: t,
                op: ReduceOp::Sum,
                axis: 0,
            },
            VirType::F32,
        );
        f.ret(Some(r));

        let mut module = VirModule::new("reduce");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("reduce"));
        assert!(ptx.contains("shfl.sync.down"));
    }

    #[test]
    fn test_cast_generation() {
        let mut f = VirFunction::new(
            "cast_test",
            vec![("x", VirType::I32)],
            VirType::F64,
        );
        let x = f.emit(VirInstKind::Param(0), VirType::I32);
        let c = f.emit(VirInstKind::Cast(x, VirType::F64), VirType::F64);
        f.ret(Some(c));

        let mut module = VirModule::new("cast");
        module.functions.push(f);
        let ptx = PtxBackend::compile_module(&module).unwrap();
        assert!(ptx.contains("cvt"));
        assert!(ptx.contains(".f64"));
    }

    #[test]
    fn test_ptx_type_suffix() {
        assert_eq!(PtxType::F32.suffix(), ".f32");
        assert_eq!(PtxType::S64.suffix(), ".s64");
        assert_eq!(PtxType::Pred.suffix(), ".pred");
        assert_eq!(PtxType::U64.suffix(), ".u64");
    }
}
