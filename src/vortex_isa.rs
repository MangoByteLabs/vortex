// vortex_isa.rs — Vortex GPU Instruction Set Architecture
// Binary ISA compiled from VIR, with .vxb format and hardware translation stubs.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// VortexOpcode — all GPU ISA opcodes as u8
// ---------------------------------------------------------------------------

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VortexOpcode {
    // Arithmetic — floating point
    FADD = 0x01,
    FSUB = 0x02,
    FMUL = 0x03,
    FDIV = 0x04,
    FNEG = 0x05,
    FABS = 0x06,
    FSQRT = 0x07,
    FEXP = 0x08,
    FLOG = 0x09,
    FMAX = 0x0A,
    FMIN = 0x0B,
    FMA = 0x0C,

    // Arithmetic — integer
    IADD = 0x10,
    ISUB = 0x11,
    IMUL = 0x12,

    // Comparison — float
    FEQ = 0x20,
    FLT = 0x21,
    FLE = 0x22,
    FGT = 0x23,
    FGE = 0x24,

    // Comparison — integer
    IEQ = 0x28,
    ILT = 0x29,

    // Logic / bitwise
    AND = 0x30,
    OR = 0x31,
    XOR = 0x32,
    SHL = 0x33,
    SHR = 0x34,
    NOT = 0x35,

    // Memory
    LD_GLOBAL = 0x40,
    ST_GLOBAL = 0x41,
    LD_SHARED = 0x42,
    ST_SHARED = 0x43,
    LD_CONST = 0x44,

    // SIMT intrinsics
    THREAD_ID_X = 0x50,
    THREAD_ID_Y = 0x51,
    THREAD_ID_Z = 0x52,
    BLOCK_ID_X = 0x53,
    BLOCK_ID_Y = 0x54,
    BLOCK_ID_Z = 0x55,
    BLOCK_DIM_X = 0x56,
    BLOCK_DIM_Y = 0x57,
    BLOCK_DIM_Z = 0x58,
    BARRIER = 0x59,
    WARP_SHUFFLE = 0x5A,
    WARP_REDUCE_SUM = 0x5B,
    WARP_REDUCE_MAX = 0x5C,
    WARP_VOTE_ALL = 0x5D,
    WARP_VOTE_ANY = 0x5E,

    // Control flow
    BRANCH = 0x60,
    BRANCH_COND = 0x61,
    CALL = 0x62,
    RET = 0x63,

    // Type conversion
    F2I = 0x70,
    I2F = 0x71,
    F16_TO_F32 = 0x72,
    F32_TO_F16 = 0x73,

    // Tensor fused ops
    TILED_MATMUL = 0x80,
    FUSED_ATTENTION = 0x81,
    FUSED_LAYERNORM = 0x82,
    FUSED_GELU = 0x83,

    // No-op / padding
    NOP = 0x00,
}

impl VortexOpcode {
    pub fn from_u8(v: u8) -> Option<VortexOpcode> {
        match v {
            0x00 => Some(VortexOpcode::NOP),
            0x01 => Some(VortexOpcode::FADD),
            0x02 => Some(VortexOpcode::FSUB),
            0x03 => Some(VortexOpcode::FMUL),
            0x04 => Some(VortexOpcode::FDIV),
            0x05 => Some(VortexOpcode::FNEG),
            0x06 => Some(VortexOpcode::FABS),
            0x07 => Some(VortexOpcode::FSQRT),
            0x08 => Some(VortexOpcode::FEXP),
            0x09 => Some(VortexOpcode::FLOG),
            0x0A => Some(VortexOpcode::FMAX),
            0x0B => Some(VortexOpcode::FMIN),
            0x0C => Some(VortexOpcode::FMA),
            0x10 => Some(VortexOpcode::IADD),
            0x11 => Some(VortexOpcode::ISUB),
            0x12 => Some(VortexOpcode::IMUL),
            0x20 => Some(VortexOpcode::FEQ),
            0x21 => Some(VortexOpcode::FLT),
            0x22 => Some(VortexOpcode::FLE),
            0x23 => Some(VortexOpcode::FGT),
            0x24 => Some(VortexOpcode::FGE),
            0x28 => Some(VortexOpcode::IEQ),
            0x29 => Some(VortexOpcode::ILT),
            0x30 => Some(VortexOpcode::AND),
            0x31 => Some(VortexOpcode::OR),
            0x32 => Some(VortexOpcode::XOR),
            0x33 => Some(VortexOpcode::SHL),
            0x34 => Some(VortexOpcode::SHR),
            0x35 => Some(VortexOpcode::NOT),
            0x40 => Some(VortexOpcode::LD_GLOBAL),
            0x41 => Some(VortexOpcode::ST_GLOBAL),
            0x42 => Some(VortexOpcode::LD_SHARED),
            0x43 => Some(VortexOpcode::ST_SHARED),
            0x44 => Some(VortexOpcode::LD_CONST),
            0x50 => Some(VortexOpcode::THREAD_ID_X),
            0x51 => Some(VortexOpcode::THREAD_ID_Y),
            0x52 => Some(VortexOpcode::THREAD_ID_Z),
            0x53 => Some(VortexOpcode::BLOCK_ID_X),
            0x54 => Some(VortexOpcode::BLOCK_ID_Y),
            0x55 => Some(VortexOpcode::BLOCK_ID_Z),
            0x56 => Some(VortexOpcode::BLOCK_DIM_X),
            0x57 => Some(VortexOpcode::BLOCK_DIM_Y),
            0x58 => Some(VortexOpcode::BLOCK_DIM_Z),
            0x59 => Some(VortexOpcode::BARRIER),
            0x5A => Some(VortexOpcode::WARP_SHUFFLE),
            0x5B => Some(VortexOpcode::WARP_REDUCE_SUM),
            0x5C => Some(VortexOpcode::WARP_REDUCE_MAX),
            0x5D => Some(VortexOpcode::WARP_VOTE_ALL),
            0x5E => Some(VortexOpcode::WARP_VOTE_ANY),
            0x60 => Some(VortexOpcode::BRANCH),
            0x61 => Some(VortexOpcode::BRANCH_COND),
            0x62 => Some(VortexOpcode::CALL),
            0x63 => Some(VortexOpcode::RET),
            0x70 => Some(VortexOpcode::F2I),
            0x71 => Some(VortexOpcode::I2F),
            0x72 => Some(VortexOpcode::F16_TO_F32),
            0x73 => Some(VortexOpcode::F32_TO_F16),
            0x80 => Some(VortexOpcode::TILED_MATMUL),
            0x81 => Some(VortexOpcode::FUSED_ATTENTION),
            0x82 => Some(VortexOpcode::FUSED_LAYERNORM),
            0x83 => Some(VortexOpcode::FUSED_GELU),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            VortexOpcode::NOP => "NOP",
            VortexOpcode::FADD => "FADD",
            VortexOpcode::FSUB => "FSUB",
            VortexOpcode::FMUL => "FMUL",
            VortexOpcode::FDIV => "FDIV",
            VortexOpcode::FNEG => "FNEG",
            VortexOpcode::FABS => "FABS",
            VortexOpcode::FSQRT => "FSQRT",
            VortexOpcode::FEXP => "FEXP",
            VortexOpcode::FLOG => "FLOG",
            VortexOpcode::FMAX => "FMAX",
            VortexOpcode::FMIN => "FMIN",
            VortexOpcode::FMA => "FMA",
            VortexOpcode::IADD => "IADD",
            VortexOpcode::ISUB => "ISUB",
            VortexOpcode::IMUL => "IMUL",
            VortexOpcode::FEQ => "FEQ",
            VortexOpcode::FLT => "FLT",
            VortexOpcode::FLE => "FLE",
            VortexOpcode::FGT => "FGT",
            VortexOpcode::FGE => "FGE",
            VortexOpcode::IEQ => "IEQ",
            VortexOpcode::ILT => "ILT",
            VortexOpcode::AND => "AND",
            VortexOpcode::OR => "OR",
            VortexOpcode::XOR => "XOR",
            VortexOpcode::SHL => "SHL",
            VortexOpcode::SHR => "SHR",
            VortexOpcode::NOT => "NOT",
            VortexOpcode::LD_GLOBAL => "LD_GLOBAL",
            VortexOpcode::ST_GLOBAL => "ST_GLOBAL",
            VortexOpcode::LD_SHARED => "LD_SHARED",
            VortexOpcode::ST_SHARED => "ST_SHARED",
            VortexOpcode::LD_CONST => "LD_CONST",
            VortexOpcode::THREAD_ID_X => "THREAD_ID_X",
            VortexOpcode::THREAD_ID_Y => "THREAD_ID_Y",
            VortexOpcode::THREAD_ID_Z => "THREAD_ID_Z",
            VortexOpcode::BLOCK_ID_X => "BLOCK_ID_X",
            VortexOpcode::BLOCK_ID_Y => "BLOCK_ID_Y",
            VortexOpcode::BLOCK_ID_Z => "BLOCK_ID_Z",
            VortexOpcode::BLOCK_DIM_X => "BLOCK_DIM_X",
            VortexOpcode::BLOCK_DIM_Y => "BLOCK_DIM_Y",
            VortexOpcode::BLOCK_DIM_Z => "BLOCK_DIM_Z",
            VortexOpcode::BARRIER => "BARRIER",
            VortexOpcode::WARP_SHUFFLE => "WARP_SHUFFLE",
            VortexOpcode::WARP_REDUCE_SUM => "WARP_REDUCE_SUM",
            VortexOpcode::WARP_REDUCE_MAX => "WARP_REDUCE_MAX",
            VortexOpcode::WARP_VOTE_ALL => "WARP_VOTE_ALL",
            VortexOpcode::WARP_VOTE_ANY => "WARP_VOTE_ANY",
            VortexOpcode::BRANCH => "BRANCH",
            VortexOpcode::BRANCH_COND => "BRANCH_COND",
            VortexOpcode::CALL => "CALL",
            VortexOpcode::RET => "RET",
            VortexOpcode::F2I => "F2I",
            VortexOpcode::I2F => "I2F",
            VortexOpcode::F16_TO_F32 => "F16_TO_F32",
            VortexOpcode::F32_TO_F16 => "F32_TO_F16",
            VortexOpcode::TILED_MATMUL => "TILED_MATMUL",
            VortexOpcode::FUSED_ATTENTION => "FUSED_ATTENTION",
            VortexOpcode::FUSED_LAYERNORM => "FUSED_LAYERNORM",
            VortexOpcode::FUSED_GELU => "FUSED_GELU",
        }
    }
}

// ---------------------------------------------------------------------------
// Instruction encoding: 32-bit word
//   [31:24] opcode  [23:16] dst_reg  [15:8] src1_reg  [7:0] src2_reg
// ---------------------------------------------------------------------------

pub fn encode_instruction(opcode: VortexOpcode, dst: u8, src1: u8, src2: u8) -> u32 {
    ((opcode as u32) << 24) | ((dst as u32) << 16) | ((src1 as u32) << 8) | (src2 as u32)
}

pub fn decode_instruction(word: u32) -> (u8, u8, u8, u8) {
    let opcode = ((word >> 24) & 0xFF) as u8;
    let dst = ((word >> 16) & 0xFF) as u8;
    let src1 = ((word >> 8) & 0xFF) as u8;
    let src2 = (word & 0xFF) as u8;
    (opcode, dst, src1, src2)
}

// ---------------------------------------------------------------------------
// VxbKernel / VxbModule — binary module representation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
pub struct VxbKernel {
    pub name: String,
    pub num_regs: u32,
    pub workgroup_size: [u32; 3],
    pub instructions: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct VxbModule {
    pub kernels: Vec<VxbKernel>,
    pub shared_mem_size: usize,
}

impl VxbModule {
    pub fn new(shared_mem_size: usize) -> Self {
        VxbModule {
            kernels: Vec::new(),
            shared_mem_size,
        }
    }

    pub fn add_kernel(&mut self, kernel: VxbKernel) {
        self.kernels.push(kernel);
    }

    pub fn find_kernel(&self, name: &str) -> Option<&VxbKernel> {
        self.kernels.iter().find(|k| k.name == name)
    }
}

// ---------------------------------------------------------------------------
// VXB binary format magic / version
// ---------------------------------------------------------------------------

const VXB_MAGIC: [u8; 4] = [b'V', b'X', b'B', 0];
const VXB_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// VortexIsaCompiler — compile, serialize, deserialize
// ---------------------------------------------------------------------------

pub struct VortexIsaCompiler;

impl VortexIsaCompiler {
    /// Compile a kernel from a sequence of encoded instruction words.
    pub fn compile_kernel(
        name: &str,
        instructions: Vec<u32>,
        num_regs: u32,
        workgroup_size: [u32; 3],
    ) -> VxbKernel {
        VxbKernel {
            name: name.to_string(),
            num_regs,
            workgroup_size,
            instructions,
        }
    }

    /// Compile a kernel from opcode tuples (opcode, dst, src1, src2).
    pub fn compile_from_opcodes(
        name: &str,
        ops: &[(VortexOpcode, u8, u8, u8)],
        num_regs: u32,
        workgroup_size: [u32; 3],
    ) -> VxbKernel {
        let instructions = ops
            .iter()
            .map(|&(op, d, s1, s2)| encode_instruction(op, d, s1, s2))
            .collect();
        Self::compile_kernel(name, instructions, num_regs, workgroup_size)
    }

    /// Serialize a VxbModule to .vxb binary format.
    pub fn serialize(module: &VxbModule) -> Vec<u8> {
        let mut buf: Vec<u8> = Vec::new();

        // Header: magic(4) + version(4) + num_kernels(4) + shared_mem_size(4)
        buf.extend_from_slice(&VXB_MAGIC);
        buf.extend_from_slice(&VXB_VERSION.to_le_bytes());
        buf.extend_from_slice(&(module.kernels.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(module.shared_mem_size as u32).to_le_bytes());

        for kernel in &module.kernels {
            let name_bytes = kernel.name.as_bytes();
            buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(name_bytes);
            buf.extend_from_slice(&kernel.num_regs.to_le_bytes());
            buf.extend_from_slice(&(kernel.instructions.len() as u32).to_le_bytes());
            for &ws in &kernel.workgroup_size {
                buf.extend_from_slice(&ws.to_le_bytes());
            }
            for &inst in &kernel.instructions {
                buf.extend_from_slice(&inst.to_le_bytes());
            }
        }

        buf
    }

    /// Deserialize a VxbModule from .vxb binary data.
    pub fn deserialize(data: &[u8]) -> Result<VxbModule, String> {
        if data.len() < 16 {
            return Err("VXB data too short for header".to_string());
        }

        // Magic
        if &data[0..4] != &VXB_MAGIC {
            return Err("Invalid VXB magic".to_string());
        }

        let version = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
        if version != VXB_VERSION {
            return Err(format!("Unsupported VXB version: {}", version));
        }

        let num_kernels = u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize;
        let shared_mem_size =
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;

        let mut module = VxbModule::new(shared_mem_size);
        let mut pos = 16usize;

        for _ in 0..num_kernels {
            // name_len + name
            if pos + 4 > data.len() {
                return Err("Unexpected EOF reading kernel name_len".to_string());
            }
            let name_len =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;

            if pos + name_len > data.len() {
                return Err("Unexpected EOF reading kernel name".to_string());
            }
            let name = String::from_utf8(data[pos..pos + name_len].to_vec())
                .map_err(|e| format!("Invalid kernel name UTF-8: {}", e))?;
            pos += name_len;

            // num_regs(4) + num_instructions(4) + workgroup_size(12)
            if pos + 20 > data.len() {
                return Err("Unexpected EOF reading kernel metadata".to_string());
            }
            let num_regs =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
            pos += 4;
            let num_instructions =
                u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
                    as usize;
            pos += 4;

            let mut workgroup_size = [0u32; 3];
            for ws in workgroup_size.iter_mut() {
                *ws =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                pos += 4;
            }

            // Instructions
            if pos + num_instructions * 4 > data.len() {
                return Err("Unexpected EOF reading kernel instructions".to_string());
            }
            let mut instructions = Vec::with_capacity(num_instructions);
            for _ in 0..num_instructions {
                let word =
                    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]]);
                instructions.push(word);
                pos += 4;
            }

            module.add_kernel(VxbKernel {
                name,
                num_regs,
                workgroup_size,
                instructions,
            });
        }

        Ok(module)
    }

    /// Disassemble instructions into human-readable text.
    pub fn disassemble(kernel: &VxbKernel) -> String {
        let mut out = format!("kernel {} (regs={}, wg=[{},{},{}]):\n",
            kernel.name, kernel.num_regs,
            kernel.workgroup_size[0], kernel.workgroup_size[1], kernel.workgroup_size[2]);
        for (i, &word) in kernel.instructions.iter().enumerate() {
            let (op, dst, s1, s2) = decode_instruction(word);
            let opname = VortexOpcode::from_u8(op)
                .map(|o| o.name())
                .unwrap_or("UNKNOWN");
            out.push_str(&format!("  {:04}: {} r{}, r{}, r{}\n", i, opname, dst, s1, s2));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Pre-compiled kernel library (12 kernels)
// ---------------------------------------------------------------------------

fn build_matmul_kernel() -> VxbKernel {
    // Tiled matmul: load tiles, multiply-accumulate, store
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::THREAD_ID_Y, 1, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 2, 0, 0),
        (VortexOpcode::BLOCK_ID_Y, 3, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 4, 0, 0),
        (VortexOpcode::IMUL, 5, 2, 4),   // global_x = block_id * block_dim
        (VortexOpcode::IADD, 6, 5, 0),   // + thread_id
        (VortexOpcode::IMUL, 7, 3, 4),
        (VortexOpcode::IADD, 8, 7, 1),
        (VortexOpcode::LD_GLOBAL, 10, 6, 0),  // load A tile
        (VortexOpcode::LD_GLOBAL, 11, 8, 0),  // load B tile
        (VortexOpcode::LD_SHARED, 12, 10, 0),
        (VortexOpcode::LD_SHARED, 13, 11, 0),
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::TILED_MATMUL, 14, 12, 13),
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::ST_GLOBAL, 14, 6, 8),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("matmul", &ops, 16, [16, 16, 1])
}

fn build_softmax_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::WARP_REDUCE_MAX, 6, 5, 0),   // max for numerical stability
        (VortexOpcode::FSUB, 7, 5, 6),
        (VortexOpcode::FEXP, 8, 7, 0),
        (VortexOpcode::WARP_REDUCE_SUM, 9, 8, 0),
        (VortexOpcode::FDIV, 10, 8, 9),
        (VortexOpcode::ST_GLOBAL, 10, 4, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("softmax", &ops, 12, [256, 1, 1])
}

fn build_layernorm_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::FUSED_LAYERNORM, 6, 5, 0),
        (VortexOpcode::ST_GLOBAL, 6, 4, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("layernorm", &ops, 8, [256, 1, 1])
}

fn build_attention_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::THREAD_ID_Y, 1, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 2, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 3, 0, 0),
        (VortexOpcode::IMUL, 4, 2, 3),
        (VortexOpcode::IADD, 5, 4, 0),
        (VortexOpcode::LD_GLOBAL, 6, 5, 0),   // Q
        (VortexOpcode::LD_GLOBAL, 7, 5, 1),   // K
        (VortexOpcode::LD_GLOBAL, 8, 5, 2),   // V
        (VortexOpcode::FUSED_ATTENTION, 9, 6, 7),
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::ST_GLOBAL, 9, 5, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("attention", &ops, 12, [64, 1, 1])
}

fn build_relu_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::FMAX, 6, 5, 0),   // max(x, 0) — r0 assumed zero-init
        (VortexOpcode::ST_GLOBAL, 6, 4, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("relu", &ops, 8, [256, 1, 1])
}

fn build_gelu_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::FUSED_GELU, 6, 5, 0),
        (VortexOpcode::ST_GLOBAL, 6, 4, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("gelu", &ops, 8, [256, 1, 1])
}

fn build_silu_kernel() -> VxbKernel {
    // SiLU(x) = x * sigmoid(x) = x * (1 / (1 + exp(-x)))
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::FNEG, 6, 5, 0),
        (VortexOpcode::FEXP, 7, 6, 0),
        // sigmoid approx via 1/(1+exp(-x)): use FDIV with const 1 in r15
        (VortexOpcode::LD_CONST, 15, 0, 0),   // load 1.0
        (VortexOpcode::FADD, 8, 15, 7),
        (VortexOpcode::FDIV, 9, 15, 8),
        (VortexOpcode::FMUL, 10, 5, 9),
        (VortexOpcode::ST_GLOBAL, 10, 4, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("silu", &ops, 16, [256, 1, 1])
}

fn build_saxpy_kernel() -> VxbKernel {
    // y = a*x + y
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_CONST, 5, 0, 0),    // scalar a
        (VortexOpcode::LD_GLOBAL, 6, 4, 0),   // x[i]
        (VortexOpcode::LD_GLOBAL, 7, 4, 1),   // y[i]
        (VortexOpcode::FMA, 8, 5, 6),         // a * x[i] + y[i] (fused)
        (VortexOpcode::ST_GLOBAL, 8, 4, 1),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("saxpy", &ops, 10, [256, 1, 1])
}

fn build_reduce_sum_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::LD_SHARED, 6, 0, 5),   // store to shared
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::WARP_REDUCE_SUM, 7, 6, 0),
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::ST_GLOBAL, 7, 1, 0),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("reduce_sum", &ops, 8, [256, 1, 1])
}

fn build_transpose_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::THREAD_ID_Y, 1, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 2, 0, 0),
        (VortexOpcode::BLOCK_ID_Y, 3, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 4, 0, 0),
        (VortexOpcode::IMUL, 5, 2, 4),
        (VortexOpcode::IADD, 6, 5, 0),   // col
        (VortexOpcode::IMUL, 7, 3, 4),
        (VortexOpcode::IADD, 8, 7, 1),   // row
        (VortexOpcode::LD_GLOBAL, 9, 8, 6),   // src[row][col]
        (VortexOpcode::ST_SHARED, 9, 0, 1),
        (VortexOpcode::BARRIER, 0, 0, 0),
        (VortexOpcode::LD_SHARED, 10, 1, 0),
        (VortexOpcode::ST_GLOBAL, 10, 6, 8),  // dst[col][row]
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("transpose", &ops, 12, [16, 16, 1])
}

fn build_elementwise_add_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::LD_GLOBAL, 6, 4, 1),
        (VortexOpcode::FADD, 7, 5, 6),
        (VortexOpcode::ST_GLOBAL, 7, 4, 2),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("elementwise_add", &ops, 8, [256, 1, 1])
}

fn build_elementwise_mul_kernel() -> VxbKernel {
    let ops = vec![
        (VortexOpcode::THREAD_ID_X, 0, 0, 0),
        (VortexOpcode::BLOCK_ID_X, 1, 0, 0),
        (VortexOpcode::BLOCK_DIM_X, 2, 0, 0),
        (VortexOpcode::IMUL, 3, 1, 2),
        (VortexOpcode::IADD, 4, 3, 0),
        (VortexOpcode::LD_GLOBAL, 5, 4, 0),
        (VortexOpcode::LD_GLOBAL, 6, 4, 1),
        (VortexOpcode::FMUL, 7, 5, 6),
        (VortexOpcode::ST_GLOBAL, 7, 4, 2),
        (VortexOpcode::RET, 0, 0, 0),
    ];
    VortexIsaCompiler::compile_from_opcodes("elementwise_mul", &ops, 8, [256, 1, 1])
}

fn build_precompiled_module() -> VxbModule {
    let mut module = VxbModule::new(49152); // 48 KiB shared memory
    module.add_kernel(build_matmul_kernel());
    module.add_kernel(build_softmax_kernel());
    module.add_kernel(build_layernorm_kernel());
    module.add_kernel(build_attention_kernel());
    module.add_kernel(build_relu_kernel());
    module.add_kernel(build_gelu_kernel());
    module.add_kernel(build_silu_kernel());
    module.add_kernel(build_saxpy_kernel());
    module.add_kernel(build_reduce_sum_kernel());
    module.add_kernel(build_transpose_kernel());
    module.add_kernel(build_elementwise_add_kernel());
    module.add_kernel(build_elementwise_mul_kernel());
    module
}

static PRECOMPILED_MODULE: LazyLock<Mutex<VxbModule>> =
    LazyLock::new(|| Mutex::new(build_precompiled_module()));

// ---------------------------------------------------------------------------
// Hardware translation stubs
// ---------------------------------------------------------------------------

/// Trait for translating VXB instructions to target hardware ISA.
pub trait HardwareTranslator {
    /// Target name (e.g. "AMD GCN", "NVIDIA NVC0", "CPU").
    fn target_name(&self) -> &str;

    /// Translate a VxbKernel into target-specific binary.
    fn translate(&self, kernel: &VxbKernel) -> Result<Vec<u8>, String>;

    /// Emit dispatch metadata for the given workgroup configuration.
    fn emit_dispatch(&self, workgroup_size: [u32; 3], grid_size: [u32; 3]) -> Result<Vec<u8>, String>;
}

// --- AMD GCN stub ---

pub struct AmdGcnTranslator;

impl HardwareTranslator for AmdGcnTranslator {
    fn target_name(&self) -> &str {
        "AMD GCN"
    }

    fn translate(&self, kernel: &VxbKernel) -> Result<Vec<u8>, String> {
        // Stub: map VXB opcodes to GCN instruction packets
        let mut gcn_binary = Vec::new();
        // GCN header: program descriptor (256 bytes placeholder)
        gcn_binary.extend_from_slice(&[0u8; 256]);

        for &word in &kernel.instructions {
            let (opcode_byte, dst, src1, src2) = decode_instruction(word);
            let opcode = VortexOpcode::from_u8(opcode_byte);
            match opcode {
                Some(VortexOpcode::FADD) => {
                    // V_ADD_F32: encoding 0x02 in VOP2
                    gcn_binary.extend_from_slice(&[0x02, dst, src1, src2]);
                }
                Some(VortexOpcode::FMUL) => {
                    // V_MUL_F32: encoding 0x08
                    gcn_binary.extend_from_slice(&[0x08, dst, src1, src2]);
                }
                Some(VortexOpcode::BARRIER) => {
                    // S_BARRIER
                    gcn_binary.extend_from_slice(&[0xBF, 0x8A, 0x07, 0x00]);
                }
                _ => {
                    // Pass through as NOP placeholder
                    gcn_binary.extend_from_slice(&word.to_le_bytes());
                }
            }
        }

        Ok(gcn_binary)
    }

    fn emit_dispatch(&self, workgroup_size: [u32; 3], grid_size: [u32; 3]) -> Result<Vec<u8>, String> {
        let mut pkt = Vec::with_capacity(64);
        // AQL dispatch packet header (stub)
        pkt.extend_from_slice(&[0x02, 0x00]); // packet type = dispatch
        for &gs in &grid_size {
            pkt.extend_from_slice(&gs.to_le_bytes());
        }
        for &ws in &workgroup_size {
            pkt.extend_from_slice(&ws.to_le_bytes());
        }
        Ok(pkt)
    }
}

// --- NVIDIA NVC0 stub ---

pub struct NvidiaNvc0Translator;

impl HardwareTranslator for NvidiaNvc0Translator {
    fn target_name(&self) -> &str {
        "NVIDIA NVC0"
    }

    fn translate(&self, kernel: &VxbKernel) -> Result<Vec<u8>, String> {
        let mut sass_binary = Vec::new();
        // NVC0 header placeholder (128 bytes)
        sass_binary.extend_from_slice(&[0u8; 128]);

        for &word in &kernel.instructions {
            let (opcode_byte, dst, src1, src2) = decode_instruction(word);
            let opcode = VortexOpcode::from_u8(opcode_byte);
            match opcode {
                Some(VortexOpcode::FADD) => {
                    // FADD Rd, Rs1, Rs2 — SASS-like encoding
                    let sass_word: u64 = 0x5C58_0000_0000_0000
                        | ((dst as u64) << 16)
                        | ((src1 as u64) << 8)
                        | (src2 as u64);
                    sass_binary.extend_from_slice(&sass_word.to_le_bytes());
                }
                Some(VortexOpcode::FMUL) => {
                    let sass_word: u64 = 0x5C68_0000_0000_0000
                        | ((dst as u64) << 16)
                        | ((src1 as u64) << 8)
                        | (src2 as u64);
                    sass_binary.extend_from_slice(&sass_word.to_le_bytes());
                }
                Some(VortexOpcode::BARRIER) => {
                    // BAR.SYNC 0
                    sass_binary.extend_from_slice(&0xF0A8_0000_0000_0000u64.to_le_bytes());
                }
                _ => {
                    // Encode as 8-byte NOP with embedded VXB word
                    let nop: u64 = 0x5090_0000_0000_0000 | (word as u64);
                    sass_binary.extend_from_slice(&nop.to_le_bytes());
                }
            }
        }

        Ok(sass_binary)
    }

    fn emit_dispatch(&self, workgroup_size: [u32; 3], grid_size: [u32; 3]) -> Result<Vec<u8>, String> {
        let mut launch = Vec::new();
        // CUDA launch params (stub)
        for &gs in &grid_size {
            launch.extend_from_slice(&gs.to_le_bytes());
        }
        for &ws in &workgroup_size {
            launch.extend_from_slice(&ws.to_le_bytes());
        }
        Ok(launch)
    }
}

// --- CPU fallback (interpret directly) ---

pub struct CpuFallbackTranslator;

impl CpuFallbackTranslator {
    /// Interpret a kernel on the CPU for a single thread.
    /// regs: mutable register file, global_mem: simulated global memory.
    pub fn interpret_single_thread(
        kernel: &VxbKernel,
        regs: &mut Vec<f64>,
        global_mem: &mut HashMap<usize, f64>,
        thread_id: [u32; 3],
        block_id: [u32; 3],
        block_dim: [u32; 3],
    ) -> Result<(), String> {
        let mut pc = 0usize;
        let num_inst = kernel.instructions.len();

        while pc < num_inst {
            let word = kernel.instructions[pc];
            let (opcode_byte, dst, src1, src2) = decode_instruction(word);
            let d = dst as usize;
            let s1 = src1 as usize;
            let s2 = src2 as usize;

            // Ensure register file is large enough
            let max_r = d.max(s1).max(s2) + 1;
            if regs.len() < max_r {
                regs.resize(max_r, 0.0);
            }

            match VortexOpcode::from_u8(opcode_byte) {
                Some(VortexOpcode::FADD) => regs[d] = regs[s1] + regs[s2],
                Some(VortexOpcode::FSUB) => regs[d] = regs[s1] - regs[s2],
                Some(VortexOpcode::FMUL) => regs[d] = regs[s1] * regs[s2],
                Some(VortexOpcode::FDIV) => {
                    if regs[s2] == 0.0 {
                        return Err("Division by zero".to_string());
                    }
                    regs[d] = regs[s1] / regs[s2];
                }
                Some(VortexOpcode::FNEG) => regs[d] = -regs[s1],
                Some(VortexOpcode::FABS) => regs[d] = regs[s1].abs(),
                Some(VortexOpcode::FSQRT) => regs[d] = regs[s1].sqrt(),
                Some(VortexOpcode::FEXP) => regs[d] = regs[s1].exp(),
                Some(VortexOpcode::FLOG) => regs[d] = regs[s1].ln(),
                Some(VortexOpcode::FMAX) => regs[d] = regs[s1].max(regs[s2]),
                Some(VortexOpcode::FMIN) => regs[d] = regs[s1].min(regs[s2]),
                Some(VortexOpcode::FMA) => regs[d] = regs[s1].mul_add(regs[s2], regs[d]),
                Some(VortexOpcode::IADD) => regs[d] = ((regs[s1] as i64) + (regs[s2] as i64)) as f64,
                Some(VortexOpcode::ISUB) => regs[d] = ((regs[s1] as i64) - (regs[s2] as i64)) as f64,
                Some(VortexOpcode::IMUL) => regs[d] = ((regs[s1] as i64) * (regs[s2] as i64)) as f64,
                Some(VortexOpcode::FEQ) => regs[d] = if regs[s1] == regs[s2] { 1.0 } else { 0.0 },
                Some(VortexOpcode::FLT) => regs[d] = if regs[s1] < regs[s2] { 1.0 } else { 0.0 },
                Some(VortexOpcode::FLE) => regs[d] = if regs[s1] <= regs[s2] { 1.0 } else { 0.0 },
                Some(VortexOpcode::FGT) => regs[d] = if regs[s1] > regs[s2] { 1.0 } else { 0.0 },
                Some(VortexOpcode::FGE) => regs[d] = if regs[s1] >= regs[s2] { 1.0 } else { 0.0 },
                Some(VortexOpcode::IEQ) => regs[d] = if (regs[s1] as i64) == (regs[s2] as i64) { 1.0 } else { 0.0 },
                Some(VortexOpcode::ILT) => regs[d] = if (regs[s1] as i64) < (regs[s2] as i64) { 1.0 } else { 0.0 },
                Some(VortexOpcode::AND) => regs[d] = ((regs[s1] as u64) & (regs[s2] as u64)) as f64,
                Some(VortexOpcode::OR) => regs[d] = ((regs[s1] as u64) | (regs[s2] as u64)) as f64,
                Some(VortexOpcode::XOR) => regs[d] = ((regs[s1] as u64) ^ (regs[s2] as u64)) as f64,
                Some(VortexOpcode::SHL) => regs[d] = ((regs[s1] as u64) << (regs[s2] as u32)) as f64,
                Some(VortexOpcode::SHR) => regs[d] = ((regs[s1] as u64) >> (regs[s2] as u32)) as f64,
                Some(VortexOpcode::NOT) => regs[d] = (!(regs[s1] as u64)) as f64,
                Some(VortexOpcode::LD_GLOBAL) => {
                    let addr = regs[s1] as usize;
                    regs[d] = *global_mem.get(&addr).unwrap_or(&0.0);
                }
                Some(VortexOpcode::ST_GLOBAL) => {
                    let addr = regs[s1] as usize;
                    global_mem.insert(addr, regs[d]);
                }
                Some(VortexOpcode::LD_SHARED) | Some(VortexOpcode::ST_SHARED) |
                Some(VortexOpcode::LD_CONST) => {
                    // Shared/const memory treated as global in CPU fallback
                    let addr = regs[s1] as usize;
                    regs[d] = *global_mem.get(&addr).unwrap_or(&0.0);
                }
                Some(VortexOpcode::THREAD_ID_X) => regs[d] = thread_id[0] as f64,
                Some(VortexOpcode::THREAD_ID_Y) => regs[d] = thread_id[1] as f64,
                Some(VortexOpcode::THREAD_ID_Z) => regs[d] = thread_id[2] as f64,
                Some(VortexOpcode::BLOCK_ID_X) => regs[d] = block_id[0] as f64,
                Some(VortexOpcode::BLOCK_ID_Y) => regs[d] = block_id[1] as f64,
                Some(VortexOpcode::BLOCK_ID_Z) => regs[d] = block_id[2] as f64,
                Some(VortexOpcode::BLOCK_DIM_X) => regs[d] = block_dim[0] as f64,
                Some(VortexOpcode::BLOCK_DIM_Y) => regs[d] = block_dim[1] as f64,
                Some(VortexOpcode::BLOCK_DIM_Z) => regs[d] = block_dim[2] as f64,
                Some(VortexOpcode::BARRIER) => { /* no-op on single thread */ }
                Some(VortexOpcode::WARP_SHUFFLE) => { /* identity on single thread */ }
                Some(VortexOpcode::WARP_REDUCE_SUM) => regs[d] = regs[s1],
                Some(VortexOpcode::WARP_REDUCE_MAX) => regs[d] = regs[s1],
                Some(VortexOpcode::WARP_VOTE_ALL) => regs[d] = regs[s1],
                Some(VortexOpcode::WARP_VOTE_ANY) => regs[d] = regs[s1],
                Some(VortexOpcode::BRANCH) => {
                    pc = dst as usize;
                    continue;
                }
                Some(VortexOpcode::BRANCH_COND) => {
                    if regs[s1] != 0.0 {
                        pc = dst as usize;
                        continue;
                    }
                }
                Some(VortexOpcode::CALL) => { /* stub: no call stack in single-thread mode */ }
                Some(VortexOpcode::RET) => break,
                Some(VortexOpcode::F2I) => regs[d] = (regs[s1] as i64) as f64,
                Some(VortexOpcode::I2F) => regs[d] = regs[s1], // already f64
                Some(VortexOpcode::F16_TO_F32) => regs[d] = regs[s1], // passthrough in f64 mode
                Some(VortexOpcode::F32_TO_F16) => regs[d] = regs[s1],
                Some(VortexOpcode::TILED_MATMUL) => {
                    // Stub: single element multiply-accumulate
                    regs[d] = regs[s1] * regs[s2] + regs[d];
                }
                Some(VortexOpcode::FUSED_ATTENTION) => {
                    // Stub: Q * K scaled
                    regs[d] = regs[s1] * regs[s2];
                }
                Some(VortexOpcode::FUSED_LAYERNORM) => {
                    // Stub: identity
                    regs[d] = regs[s1];
                }
                Some(VortexOpcode::FUSED_GELU) => {
                    // GELU(x) ~ x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
                    let x = regs[s1];
                    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                    regs[d] = 0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh());
                }
                Some(VortexOpcode::NOP) => {}
                None => {
                    return Err(format!("Unknown opcode: 0x{:02X} at pc={}", opcode_byte, pc));
                }
            }

            pc += 1;
        }

        Ok(())
    }
}

impl HardwareTranslator for CpuFallbackTranslator {
    fn target_name(&self) -> &str {
        "CPU Fallback"
    }

    fn translate(&self, kernel: &VxbKernel) -> Result<Vec<u8>, String> {
        // For CPU fallback, return the raw VXB instructions as bytes
        let mut buf = Vec::with_capacity(kernel.instructions.len() * 4);
        for &inst in &kernel.instructions {
            buf.extend_from_slice(&inst.to_le_bytes());
        }
        Ok(buf)
    }

    fn emit_dispatch(&self, workgroup_size: [u32; 3], grid_size: [u32; 3]) -> Result<Vec<u8>, String> {
        // Return thread counts for CPU scheduling
        let total_threads = (grid_size[0] * workgroup_size[0])
            * (grid_size[1] * workgroup_size[1])
            * (grid_size[2] * workgroup_size[2]);
        Ok(total_threads.to_le_bytes().to_vec())
    }
}

// ---------------------------------------------------------------------------
// Builtins — bridge to Vortex runtime
// ---------------------------------------------------------------------------

/// vxb_compile(name: String, opcodes: Array<Array<Int>>, num_regs: Int, wg_x: Int, wg_y: Int, wg_z: Int) -> String
fn builtin_vxb_compile(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 6 {
        return Err("vxb_compile requires (name, opcodes, num_regs, wg_x, wg_y, wg_z)".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("vxb_compile: name must be a string".to_string()),
    };
    let opcodes_arr = match &args[1] {
        Value::Array(a) => a.clone(),
        _ => return Err("vxb_compile: opcodes must be an array of [op, dst, src1, src2]".to_string()),
    };
    let num_regs = match &args[2] {
        Value::Int(n) => *n as u32,
        _ => return Err("vxb_compile: num_regs must be Int".to_string()),
    };
    let wg_x = match &args[3] { Value::Int(n) => *n as u32, _ => return Err("wg_x must be Int".to_string()) };
    let wg_y = match &args[4] { Value::Int(n) => *n as u32, _ => return Err("wg_y must be Int".to_string()) };
    let wg_z = match &args[5] { Value::Int(n) => *n as u32, _ => return Err("wg_z must be Int".to_string()) };

    let mut instructions = Vec::new();
    for val in &opcodes_arr {
        match val {
            Value::Array(inner) if inner.len() == 4 => {
                let op = match &inner[0] { Value::Int(n) => *n as u8, _ => return Err("opcode must be Int".to_string()) };
                let dst = match &inner[1] { Value::Int(n) => *n as u8, _ => return Err("dst must be Int".to_string()) };
                let s1 = match &inner[2] { Value::Int(n) => *n as u8, _ => return Err("src1 must be Int".to_string()) };
                let s2 = match &inner[3] { Value::Int(n) => *n as u8, _ => return Err("src2 must be Int".to_string()) };
                instructions.push(encode_instruction(
                    VortexOpcode::from_u8(op).ok_or_else(|| format!("Unknown opcode: {}", op))?,
                    dst, s1, s2,
                ));
            }
            _ => return Err("Each instruction must be [opcode, dst, src1, src2]".to_string()),
        }
    }

    let kernel = VortexIsaCompiler::compile_kernel(&name, instructions, num_regs, [wg_x, wg_y, wg_z]);

    // Store in precompiled module
    let mut module = PRECOMPILED_MODULE.lock().unwrap();
    let info = format!("kernel '{}': {} instructions, {} regs, wg=[{},{},{}]",
        kernel.name, kernel.instructions.len(), kernel.num_regs,
        wg_x, wg_y, wg_z);
    module.add_kernel(kernel);

    Ok(Value::String(info))
}

/// vxb_serialize() -> Array<Int>  — serialize the current module to bytes
fn builtin_vxb_serialize(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let module = PRECOMPILED_MODULE.lock().unwrap();
    let bytes = VortexIsaCompiler::serialize(&module);
    let arr = bytes.into_iter().map(|b| Value::Int(b as i128)).collect();
    Ok(Value::Array(arr))
}

/// vxb_deserialize(data: Array<Int>) -> String — deserialize and replace module
fn builtin_vxb_deserialize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("vxb_deserialize requires a byte array".to_string());
    }
    let data_arr = match &args[0] {
        Value::Array(a) => a,
        _ => return Err("vxb_deserialize: argument must be Array<Int>".to_string()),
    };
    let bytes: Vec<u8> = data_arr
        .iter()
        .map(|v| match v {
            Value::Int(n) => Ok(*n as u8),
            _ => Err("Each byte must be Int".to_string()),
        })
        .collect::<Result<Vec<u8>, String>>()?;

    let new_module = VortexIsaCompiler::deserialize(&bytes)?;
    let summary = format!(
        "Loaded VXB module: {} kernels, shared_mem={}",
        new_module.kernels.len(),
        new_module.shared_mem_size
    );

    let mut module = PRECOMPILED_MODULE.lock().unwrap();
    *module = new_module;

    Ok(Value::String(summary))
}

/// vxb_list_kernels() -> Array<String>
fn builtin_vxb_list_kernels(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let module = PRECOMPILED_MODULE.lock().unwrap();
    let names: Vec<Value> = module
        .kernels
        .iter()
        .map(|k| Value::String(k.name.clone()))
        .collect();
    Ok(Value::Array(names))
}

/// vxb_kernel_info(name: String) -> String — disassembly and metadata
fn builtin_vxb_kernel_info(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("vxb_kernel_info requires a kernel name".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("vxb_kernel_info: name must be string".to_string()),
    };
    let module = PRECOMPILED_MODULE.lock().unwrap();
    match module.find_kernel(&name) {
        Some(kernel) => {
            let disasm = VortexIsaCompiler::disassemble(kernel);
            Ok(Value::String(disasm))
        }
        None => Err(format!("Kernel '{}' not found", name)),
    }
}

/// vxb_precompiled() -> Array<String> — list just the 12 standard precompiled kernels
fn builtin_vxb_precompiled(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let names = vec![
        "matmul", "softmax", "layernorm", "attention",
        "relu", "gelu", "silu", "saxpy",
        "reduce_sum", "transpose", "elementwise_add", "elementwise_mul",
    ];
    let arr = names.into_iter().map(|n| Value::String(n.to_string())).collect();
    Ok(Value::Array(arr))
}

// ---------------------------------------------------------------------------
// register_builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("vxb_compile".to_string(), FnDef::Builtin(builtin_vxb_compile));
    env.functions.insert("vxb_serialize".to_string(), FnDef::Builtin(builtin_vxb_serialize));
    env.functions.insert("vxb_deserialize".to_string(), FnDef::Builtin(builtin_vxb_deserialize));
    env.functions.insert("vxb_list_kernels".to_string(), FnDef::Builtin(builtin_vxb_list_kernels));
    env.functions.insert("vxb_kernel_info".to_string(), FnDef::Builtin(builtin_vxb_kernel_info));
    env.functions.insert("vxb_precompiled".to_string(), FnDef::Builtin(builtin_vxb_precompiled));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_instruction_encoding_roundtrip() {
        let cases = vec![
            (VortexOpcode::FADD, 1, 2, 3),
            (VortexOpcode::TILED_MATMUL, 14, 12, 13),
            (VortexOpcode::BARRIER, 0, 0, 0),
            (VortexOpcode::BRANCH_COND, 255, 128, 64),
            (VortexOpcode::NOP, 0, 0, 0),
        ];
        for (op, d, s1, s2) in cases {
            let word = encode_instruction(op, d, s1, s2);
            let (got_op, got_d, got_s1, got_s2) = decode_instruction(word);
            assert_eq!(got_op, op as u8);
            assert_eq!(got_d, d);
            assert_eq!(got_s1, s1);
            assert_eq!(got_s2, s2);
        }
    }

    #[test]
    fn test_serialize_deserialize_roundtrip() {
        let mut module = VxbModule::new(4096);
        let kernel = VortexIsaCompiler::compile_from_opcodes(
            "test_add",
            &[
                (VortexOpcode::THREAD_ID_X, 0, 0, 0),
                (VortexOpcode::LD_GLOBAL, 1, 0, 0),
                (VortexOpcode::LD_GLOBAL, 2, 0, 1),
                (VortexOpcode::FADD, 3, 1, 2),
                (VortexOpcode::ST_GLOBAL, 3, 0, 2),
                (VortexOpcode::RET, 0, 0, 0),
            ],
            4,
            [256, 1, 1],
        );
        module.add_kernel(kernel);

        let bytes = VortexIsaCompiler::serialize(&module);
        let restored = VortexIsaCompiler::deserialize(&bytes).unwrap();

        assert_eq!(restored.shared_mem_size, 4096);
        assert_eq!(restored.kernels.len(), 1);
        assert_eq!(restored.kernels[0].name, "test_add");
        assert_eq!(restored.kernels[0].num_regs, 4);
        assert_eq!(restored.kernels[0].workgroup_size, [256, 1, 1]);
        assert_eq!(restored.kernels[0].instructions.len(), 6);

        // Verify instruction-level equality
        let orig_mod = VxbModule::new(4096);
        let orig_kernel = VortexIsaCompiler::compile_from_opcodes(
            "test_add",
            &[
                (VortexOpcode::THREAD_ID_X, 0, 0, 0),
                (VortexOpcode::LD_GLOBAL, 1, 0, 0),
                (VortexOpcode::LD_GLOBAL, 2, 0, 1),
                (VortexOpcode::FADD, 3, 1, 2),
                (VortexOpcode::ST_GLOBAL, 3, 0, 2),
                (VortexOpcode::RET, 0, 0, 0),
            ],
            4,
            [256, 1, 1],
        );
        assert_eq!(restored.kernels[0].instructions, orig_kernel.instructions);
    }

    #[test]
    fn test_precompiled_kernel_listing() {
        let module = PRECOMPILED_MODULE.lock().unwrap();
        let names: Vec<&str> = module.kernels.iter().map(|k| k.name.as_str()).collect();
        assert!(names.contains(&"matmul"));
        assert!(names.contains(&"softmax"));
        assert!(names.contains(&"layernorm"));
        assert!(names.contains(&"attention"));
        assert!(names.contains(&"relu"));
        assert!(names.contains(&"gelu"));
        assert!(names.contains(&"silu"));
        assert!(names.contains(&"saxpy"));
        assert!(names.contains(&"reduce_sum"));
        assert!(names.contains(&"transpose"));
        assert!(names.contains(&"elementwise_add"));
        assert!(names.contains(&"elementwise_mul"));
        assert!(names.len() >= 12);
    }

    #[test]
    fn test_disassemble() {
        let kernel = VortexIsaCompiler::compile_from_opcodes(
            "tiny",
            &[
                (VortexOpcode::FADD, 2, 0, 1),
                (VortexOpcode::RET, 0, 0, 0),
            ],
            3,
            [1, 1, 1],
        );
        let text = VortexIsaCompiler::disassemble(&kernel);
        assert!(text.contains("FADD"));
        assert!(text.contains("RET"));
        assert!(text.contains("tiny"));
    }

    #[test]
    fn test_cpu_fallback_simple_add() {
        let kernel = VortexIsaCompiler::compile_from_opcodes(
            "add_test",
            &[
                (VortexOpcode::FADD, 2, 0, 1),
                (VortexOpcode::RET, 0, 0, 0),
            ],
            3,
            [1, 1, 1],
        );
        let mut regs = vec![3.0, 4.0, 0.0];
        let mut mem = HashMap::new();
        CpuFallbackTranslator::interpret_single_thread(
            &kernel, &mut regs, &mut mem,
            [0, 0, 0], [0, 0, 0], [1, 1, 1],
        ).unwrap();
        assert!((regs[2] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_cpu_fallback_gelu() {
        let kernel = VortexIsaCompiler::compile_from_opcodes(
            "gelu_test",
            &[
                (VortexOpcode::FUSED_GELU, 1, 0, 0),
                (VortexOpcode::RET, 0, 0, 0),
            ],
            2,
            [1, 1, 1],
        );
        let mut regs = vec![1.0, 0.0];
        let mut mem = HashMap::new();
        CpuFallbackTranslator::interpret_single_thread(
            &kernel, &mut regs, &mut mem,
            [0, 0, 0], [0, 0, 0], [1, 1, 1],
        ).unwrap();
        // GELU(1.0) ~ 0.8412
        assert!((regs[1] - 0.8412).abs() < 0.01);
    }

    #[test]
    fn test_invalid_magic() {
        let bad_data = vec![0u8; 16];
        assert!(VortexIsaCompiler::deserialize(&bad_data).is_err());
    }

    #[test]
    fn test_hardware_translator_names() {
        let amd = AmdGcnTranslator;
        assert_eq!(amd.target_name(), "AMD GCN");
        let nv = NvidiaNvc0Translator;
        assert_eq!(nv.target_name(), "NVIDIA NVC0");
        let cpu = CpuFallbackTranslator;
        assert_eq!(cpu.target_name(), "CPU Fallback");
    }

    #[test]
    fn test_opcode_from_u8_roundtrip() {
        for byte in 0..=0xFF {
            if let Some(op) = VortexOpcode::from_u8(byte) {
                assert_eq!(op as u8, byte);
            }
        }
    }
}
