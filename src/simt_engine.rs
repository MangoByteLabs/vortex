// simt_engine.rs — SIMT execution engine for Vortex
// Runs .vxb kernels on CPU (simulated SIMT) or dispatches to GPU via drm_driver.

use crate::interpreter::{Env, FnDef, Value};
use std::sync::{LazyLock, Mutex};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const LANES_PER_WARP: usize = 32;
const WARPS_PER_BLOCK: usize = 4;
const LANES_PER_BLOCK: usize = LANES_PER_WARP * WARPS_PER_BLOCK; // 128
const REGISTER_FILE_SIZE: usize = 64;
const DEFAULT_SHARED_MEM_SIZE: usize = 4096; // f32 elements

// ---------------------------------------------------------------------------
// VXB Opcodes (mirrors vortex_isa encoding — top 8 bits of u32 instruction)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Opcode {
    // Arithmetic
    Add = 0x01,
    Sub = 0x02,
    Mul = 0x03,
    Div = 0x04,
    Fma = 0x05,
    Neg = 0x06,
    Sqrt = 0x07,
    Exp = 0x08,
    Log = 0x09,
    Max = 0x0A,
    Min = 0x0B,

    // Memory
    LoadGlobal = 0x10,
    StoreGlobal = 0x11,
    LoadShared = 0x12,
    StoreShared = 0x13,
    LoadReg = 0x14,

    // SIMT intrinsics
    ShuffleDown = 0x20,
    ReduceSum = 0x21,
    ReduceMax = 0x22,
    Broadcast = 0x23,
    VoteAll = 0x24,
    VoteAny = 0x25,
    Barrier = 0x26,

    // Control flow
    BranchIf = 0x30,
    BranchIfNot = 0x31,
    Jump = 0x32,
    Converge = 0x33,
    Halt = 0x3F,

    // Fused ops
    TiledMatmul = 0x40,
    FusedAttention = 0x41,
    FusedLayerNorm = 0x42,
    FusedSoftmax = 0x43,
    FusedGelu = 0x44,
    FusedRmsNorm = 0x45,
}

impl Opcode {
    fn from_u8(v: u8) -> Option<Opcode> {
        match v {
            0x01 => Some(Opcode::Add),
            0x02 => Some(Opcode::Sub),
            0x03 => Some(Opcode::Mul),
            0x04 => Some(Opcode::Div),
            0x05 => Some(Opcode::Fma),
            0x06 => Some(Opcode::Neg),
            0x07 => Some(Opcode::Sqrt),
            0x08 => Some(Opcode::Exp),
            0x09 => Some(Opcode::Log),
            0x0A => Some(Opcode::Max),
            0x0B => Some(Opcode::Min),
            0x10 => Some(Opcode::LoadGlobal),
            0x11 => Some(Opcode::StoreGlobal),
            0x12 => Some(Opcode::LoadShared),
            0x13 => Some(Opcode::StoreShared),
            0x14 => Some(Opcode::LoadReg),
            0x20 => Some(Opcode::ShuffleDown),
            0x21 => Some(Opcode::ReduceSum),
            0x22 => Some(Opcode::ReduceMax),
            0x23 => Some(Opcode::Broadcast),
            0x24 => Some(Opcode::VoteAll),
            0x25 => Some(Opcode::VoteAny),
            0x26 => Some(Opcode::Barrier),
            0x30 => Some(Opcode::BranchIf),
            0x31 => Some(Opcode::BranchIfNot),
            0x32 => Some(Opcode::Jump),
            0x33 => Some(Opcode::Converge),
            0x3F => Some(Opcode::Halt),
            0x40 => Some(Opcode::TiledMatmul),
            0x41 => Some(Opcode::FusedAttention),
            0x42 => Some(Opcode::FusedLayerNorm),
            0x43 => Some(Opcode::FusedSoftmax),
            0x44 => Some(Opcode::FusedGelu),
            0x45 => Some(Opcode::FusedRmsNorm),
            _ => None,
        }
    }
}

/// Decode a 32-bit VXB instruction: [opcode:8][dst:8][src1:8][src2:8]
fn decode_instruction(inst: u32) -> (u8, usize, usize, usize) {
    let opcode = ((inst >> 24) & 0xFF) as u8;
    let dst = ((inst >> 16) & 0xFF) as usize;
    let src1 = ((inst >> 8) & 0xFF) as usize;
    let src2 = (inst & 0xFF) as usize;
    (opcode, dst, src1, src2)
}

/// Encode a VXB instruction from parts.
pub fn encode_instruction(opcode: u8, dst: u8, src1: u8, src2: u8) -> u32 {
    ((opcode as u32) << 24) | ((dst as u32) << 16) | ((src1 as u32) << 8) | (src2 as u32)
}

// ---------------------------------------------------------------------------
// SIMTLane
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SIMTLane {
    pub register_file: [f32; REGISTER_FILE_SIZE],
    pub active: bool,
    pub token_pos: usize,
    pub sequence_id: usize,
    pub step: usize,
}

impl SIMTLane {
    pub fn new() -> Self {
        SIMTLane {
            register_file: [0.0; REGISTER_FILE_SIZE],
            active: true,
            token_pos: 0,
            sequence_id: 0,
            step: 0,
        }
    }

    pub fn inactive() -> Self {
        let mut lane = Self::new();
        lane.active = false;
        lane
    }
}

impl Default for SIMTLane {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SIMTWarp (32 lanes)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SIMTWarp {
    pub lanes: Vec<SIMTLane>, // length 32
    pub divergence_stack: Vec<u32>, // bitmask stack for divergence
    pub pc: usize, // program counter
}

impl SIMTWarp {
    pub fn new() -> Self {
        let lanes = (0..LANES_PER_WARP).map(|_| SIMTLane::new()).collect();
        SIMTWarp {
            lanes,
            divergence_stack: Vec::new(),
            pc: 0,
        }
    }

    /// Get the active mask as a u32 bitmask.
    pub fn active_mask(&self) -> u32 {
        let mut mask = 0u32;
        for (i, lane) in self.lanes.iter().enumerate() {
            if lane.active {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// Set active lanes from a bitmask.
    pub fn set_active_mask(&mut self, mask: u32) {
        for (i, lane) in self.lanes.iter_mut().enumerate() {
            lane.active = (mask >> i) & 1 != 0;
        }
    }

    /// Shuffle-down: each active lane reads from lane + delta.
    pub fn shuffle_down(&mut self, reg: usize, delta: usize) {
        let snapshot: Vec<f32> = self.lanes.iter().map(|l| l.register_file[reg]).collect();
        for i in 0..LANES_PER_WARP {
            if self.lanes[i].active {
                let src = i + delta;
                if src < LANES_PER_WARP {
                    self.lanes[i].register_file[reg] = snapshot[src];
                }
            }
        }
    }

    /// Reduce-sum across all active lanes for a given register, result broadcast to lane 0.
    pub fn reduce_sum(&mut self, reg: usize) -> f32 {
        let sum: f32 = self.lanes.iter()
            .filter(|l| l.active)
            .map(|l| l.register_file[reg])
            .sum();
        if self.lanes[0].active {
            self.lanes[0].register_file[reg] = sum;
        }
        sum
    }

    /// Reduce-max across all active lanes for a given register.
    pub fn reduce_max(&mut self, reg: usize) -> f32 {
        let max_val = self.lanes.iter()
            .filter(|l| l.active)
            .map(|l| l.register_file[reg])
            .fold(f32::NEG_INFINITY, f32::max);
        if self.lanes[0].active {
            self.lanes[0].register_file[reg] = max_val;
        }
        max_val
    }

    /// Broadcast value from src_lane's register to all active lanes.
    pub fn broadcast(&mut self, reg: usize, src_lane: usize) {
        if src_lane >= LANES_PER_WARP {
            return;
        }
        let val = self.lanes[src_lane].register_file[reg];
        for lane in self.lanes.iter_mut() {
            if lane.active {
                lane.register_file[reg] = val;
            }
        }
    }

    /// Vote-all: returns true iff all active lanes have register[reg] != 0.
    pub fn vote_all(&self, reg: usize) -> bool {
        self.lanes.iter()
            .filter(|l| l.active)
            .all(|l| l.register_file[reg] != 0.0)
    }

    /// Vote-any: returns true if any active lane has register[reg] != 0.
    pub fn vote_any(&self, reg: usize) -> bool {
        self.lanes.iter()
            .filter(|l| l.active)
            .any(|l| l.register_file[reg] != 0.0)
    }

    /// Push current active mask for divergence, narrow to predicate-true lanes.
    pub fn push_divergence(&mut self, predicate_reg: usize) {
        let current = self.active_mask();
        self.divergence_stack.push(current);
        // Narrow: only lanes where predicate != 0 stay active
        for lane in self.lanes.iter_mut() {
            if lane.active && lane.register_file[predicate_reg] == 0.0 {
                lane.active = false;
            }
        }
    }

    /// Flip to the else branch: active = saved & !current_active.
    pub fn flip_divergence(&mut self) {
        if let Some(&saved) = self.divergence_stack.last() {
            let current = self.active_mask();
            let flipped = saved & !current;
            self.set_active_mask(flipped);
        }
    }

    /// Reconverge: pop divergence stack, restore saved mask.
    pub fn reconverge(&mut self) {
        if let Some(saved) = self.divergence_stack.pop() {
            self.set_active_mask(saved);
        }
    }
}

impl Default for SIMTWarp {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SIMTBlock (4 warps = 128 lanes)
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SIMTBlock {
    pub warps: Vec<SIMTWarp>, // length 4
    pub shared_memory: Vec<f32>,
    pub barrier_count: usize,
    pub block_idx: [usize; 3],
}

impl SIMTBlock {
    pub fn new() -> Self {
        SIMTBlock {
            warps: (0..WARPS_PER_BLOCK).map(|_| SIMTWarp::new()).collect(),
            shared_memory: vec![0.0; DEFAULT_SHARED_MEM_SIZE],
            barrier_count: 0,
            block_idx: [0, 0, 0],
        }
    }

    /// Barrier synchronization: all warps must reach barrier before any proceed.
    pub fn barrier_sync(&mut self) {
        self.barrier_count += 1;
        // In CPU simulation, warps execute sequentially, so barrier is implicitly
        // satisfied. We track the count for diagnostics.
    }

    /// Get a flat lane index → (warp_idx, lane_idx).
    pub fn flat_to_warp_lane(flat: usize) -> (usize, usize) {
        (flat / LANES_PER_WARP, flat % LANES_PER_WARP)
    }

    /// Get a lane by flat index.
    pub fn lane_mut(&mut self, flat: usize) -> &mut SIMTLane {
        let (w, l) = Self::flat_to_warp_lane(flat);
        &mut self.warps[w].lanes[l]
    }

    /// Get a lane by flat index (immutable).
    pub fn lane(&self, flat: usize) -> &SIMTLane {
        let (w, l) = Self::flat_to_warp_lane(flat);
        &self.warps[w].lanes[l]
    }
}

impl Default for SIMTBlock {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SIMTGrid
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct SIMTGrid {
    pub blocks: Vec<SIMTBlock>,
    pub grid_dim: [usize; 3],
    pub block_dim: [usize; 3],
}

impl SIMTGrid {
    pub fn new(grid_dim: [usize; 3], block_dim: [usize; 3]) -> Self {
        let num_blocks = grid_dim[0] * grid_dim[1] * grid_dim[2];
        let mut blocks: Vec<SIMTBlock> = (0..num_blocks).map(|_| SIMTBlock::new()).collect();
        // Assign block indices
        let mut idx = 0;
        for z in 0..grid_dim[2] {
            for y in 0..grid_dim[1] {
                for x in 0..grid_dim[0] {
                    if idx < blocks.len() {
                        blocks[idx].block_idx = [x, y, z];
                        idx += 1;
                    }
                }
            }
        }
        SIMTGrid { blocks, grid_dim, block_dim }
    }

    /// Total number of lanes across all blocks.
    pub fn total_lanes(&self) -> usize {
        self.blocks.len() * LANES_PER_BLOCK
    }
}

// ---------------------------------------------------------------------------
// VXB Interpreter — execute kernel instructions on the SIMT grid
// ---------------------------------------------------------------------------

pub fn execute_vxb(
    kernel_instructions: &[u32],
    grid: &mut SIMTGrid,
    global_memory: &mut Vec<f32>,
) {
    let max_steps = kernel_instructions.len() * 1000; // safety limit
    for block in grid.blocks.iter_mut() {
        for wi in 0..block.warps.len() {
            block.warps[wi].pc = 0;
            let mut steps = 0;
            loop {
                if block.warps[wi].pc >= kernel_instructions.len() || steps >= max_steps {
                    break;
                }
                let inst = kernel_instructions[block.warps[wi].pc];
                let (op_byte, dst, src1, src2) = decode_instruction(inst);
                let opcode = Opcode::from_u8(op_byte);

                match opcode {
                    Some(Opcode::Halt) => break,

                    // --- Arithmetic ---
                    Some(Opcode::Add) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] =
                                    lane.register_file[src1] + lane.register_file[src2];
                            }
                        }
                    }
                    Some(Opcode::Sub) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] =
                                    lane.register_file[src1] - lane.register_file[src2];
                            }
                        }
                    }
                    Some(Opcode::Mul) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] =
                                    lane.register_file[src1] * lane.register_file[src2];
                            }
                        }
                    }
                    Some(Opcode::Div) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                let divisor = lane.register_file[src2];
                                lane.register_file[dst] = if divisor != 0.0 {
                                    lane.register_file[src1] / divisor
                                } else {
                                    f32::NAN
                                };
                            }
                        }
                    }
                    Some(Opcode::Fma) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = lane.register_file[src1]
                                    .mul_add(lane.register_file[src2], lane.register_file[dst]);
                            }
                        }
                    }
                    Some(Opcode::Neg) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = -lane.register_file[src1];
                            }
                        }
                    }
                    Some(Opcode::Sqrt) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = lane.register_file[src1].sqrt();
                            }
                        }
                    }
                    Some(Opcode::Exp) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = lane.register_file[src1].exp();
                            }
                        }
                    }
                    Some(Opcode::Log) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = lane.register_file[src1].ln();
                            }
                        }
                    }
                    Some(Opcode::Max) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] =
                                    lane.register_file[src1].max(lane.register_file[src2]);
                            }
                        }
                    }
                    Some(Opcode::Min) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] =
                                    lane.register_file[src1].min(lane.register_file[src2]);
                            }
                        }
                    }

                    // --- Memory ---
                    Some(Opcode::LoadGlobal) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                let addr = lane.register_file[src1] as usize;
                                lane.register_file[dst] = if addr < global_memory.len() {
                                    global_memory[addr]
                                } else {
                                    0.0
                                };
                            }
                        }
                    }
                    Some(Opcode::StoreGlobal) => {
                        for lane in block.warps[wi].lanes.iter() {
                            if lane.active {
                                let addr = lane.register_file[src1] as usize;
                                let val = lane.register_file[src2];
                                if addr < global_memory.len() {
                                    global_memory[addr] = val;
                                }
                            }
                        }
                    }
                    Some(Opcode::LoadShared) => {
                        // Collect addresses first to avoid double borrow
                        let loads: Vec<(usize, usize)> = block.warps[wi].lanes.iter()
                            .enumerate()
                            .filter(|(_, l)| l.active)
                            .map(|(i, l)| (i, l.register_file[src1] as usize))
                            .collect();
                        for (lane_idx, addr) in loads {
                            let val = if addr < block.shared_memory.len() {
                                block.shared_memory[addr]
                            } else {
                                0.0
                            };
                            block.warps[wi].lanes[lane_idx].register_file[dst] = val;
                        }
                    }
                    Some(Opcode::StoreShared) => {
                        let stores: Vec<(usize, f32)> = block.warps[wi].lanes.iter()
                            .filter(|l| l.active)
                            .map(|l| (l.register_file[src1] as usize, l.register_file[src2]))
                            .collect();
                        for (addr, val) in stores {
                            if addr < block.shared_memory.len() {
                                block.shared_memory[addr] = val;
                            }
                        }
                    }
                    Some(Opcode::LoadReg) => {
                        let imm = ((src1 as u32) << 8) | (src2 as u32);
                        let val = imm as f32;
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = val;
                            }
                        }
                    }

                    // --- SIMT intrinsics ---
                    Some(Opcode::ShuffleDown) => {
                        block.warps[wi].shuffle_down(dst, src1);
                    }
                    Some(Opcode::ReduceSum) => {
                        block.warps[wi].reduce_sum(dst);
                    }
                    Some(Opcode::ReduceMax) => {
                        block.warps[wi].reduce_max(dst);
                    }
                    Some(Opcode::Broadcast) => {
                        block.warps[wi].broadcast(dst, src1);
                    }
                    Some(Opcode::VoteAll) => {
                        let result = if block.warps[wi].vote_all(src1) { 1.0 } else { 0.0 };
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = result;
                            }
                        }
                    }
                    Some(Opcode::VoteAny) => {
                        let result = if block.warps[wi].vote_any(src1) { 1.0 } else { 0.0 };
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                lane.register_file[dst] = result;
                            }
                        }
                    }
                    Some(Opcode::Barrier) => {
                        block.barrier_count += 1;
                    }

                    // --- Control flow ---
                    Some(Opcode::BranchIf) => {
                        block.warps[wi].push_divergence(src1);
                    }
                    Some(Opcode::BranchIfNot) => {
                        block.warps[wi].flip_divergence();
                    }
                    Some(Opcode::Jump) => {
                        let target = (dst << 8) | src1;
                        if target < kernel_instructions.len() {
                            block.warps[wi].pc = target;
                            steps += 1;
                            continue;
                        }
                    }
                    Some(Opcode::Converge) => {
                        block.warps[wi].reconverge();
                    }

                    // --- Fused ops (CPU fast paths) ---
                    Some(Opcode::TiledMatmul) => {
                        cpu_fused_tiled_matmul(&mut block.shared_memory, global_memory, dst, src1, src2);
                    }
                    Some(Opcode::FusedAttention) => {
                        cpu_fused_attention(&mut block.shared_memory, global_memory, dst, src1, src2);
                    }
                    Some(Opcode::FusedLayerNorm) => {
                        cpu_fused_layernorm(&mut block.shared_memory, dst, src1, src2);
                    }
                    Some(Opcode::FusedSoftmax) => {
                        cpu_fused_softmax(&mut block.shared_memory, dst, src1);
                    }
                    Some(Opcode::FusedGelu) => {
                        for lane in block.warps[wi].lanes.iter_mut() {
                            if lane.active {
                                let x = lane.register_file[src1];
                                let c = 0.7978845608_f32;
                                let inner = c * (x + 0.044715 * x * x * x);
                                lane.register_file[dst] = 0.5 * x * (1.0 + inner.tanh());
                            }
                        }
                    }
                    Some(Opcode::FusedRmsNorm) => {
                        cpu_fused_rmsnorm(&mut block.shared_memory, dst, src1, src2);
                    }

                    None => {
                        // Unknown opcode — skip
                    }
                }

                block.warps[wi].pc += 1;
                steps += 1;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// CPU fused-op fast paths
// ---------------------------------------------------------------------------

/// Tiled matmul: reads from global_memory, writes result to shared_memory.
/// dst = base offset in shared_memory for output
/// src1 = base offset in global_memory for matrix A
/// src2 = base offset in global_memory for matrix B
/// Tile size = block lanes (128), assumes square tiles for simplicity.
fn cpu_fused_tiled_matmul(
    smem: &mut Vec<f32>,
    global_memory: &mut Vec<f32>,
    out_base: usize,
    a_base: usize,
    b_base: usize,
) {
    let tile = 16; // 16x16 tile
    for i in 0..tile {
        for j in 0..tile {
            let mut acc = 0.0f32;
            for k in 0..tile {
                let a_idx = a_base + i * tile + k;
                let b_idx = b_base + k * tile + j;
                let a_val = if a_idx < global_memory.len() { global_memory[a_idx] } else { 0.0 };
                let b_val = if b_idx < global_memory.len() { global_memory[b_idx] } else { 0.0 };
                acc = a_val.mul_add(b_val, acc);
            }
            let out_idx = out_base + i * tile + j;
            if out_idx < smem.len() {
                smem[out_idx] = acc;
            }
        }
    }
}

/// Fused attention: Q*K^T / sqrt(d) → softmax → * V.
/// Operates on shared memory tiles. dst=output base, src1=QK base, src2=V base.
fn cpu_fused_attention(
    smem: &mut Vec<f32>,
    global_memory: &mut Vec<f32>,
    out_base: usize,
    qk_base: usize,
    v_base: usize,
) {
    let seq_len = 16;
    let d_head = 16;

    // Step 1: scores = Q*K^T / sqrt(d)
    let scale = 1.0 / (d_head as f32).sqrt();
    let mut scores = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0f32;
            for k in 0..d_head {
                let q_idx = qk_base + i * d_head + k;
                let k_idx = qk_base + seq_len * d_head + j * d_head + k;
                let q = if q_idx < global_memory.len() { global_memory[q_idx] } else { 0.0 };
                let kv = if k_idx < global_memory.len() { global_memory[k_idx] } else { 0.0 };
                dot += q * kv;
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }

    // Step 2: softmax per row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let max_val = scores[row_start..row_start + seq_len]
            .iter()
            .cloned()
            .fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for j in 0..seq_len {
            scores[row_start + j] = (scores[row_start + j] - max_val).exp();
            sum += scores[row_start + j];
        }
        if sum > 0.0 {
            for j in 0..seq_len {
                scores[row_start + j] /= sum;
            }
        }
    }

    // Step 3: output = scores * V
    for i in 0..seq_len {
        for j in 0..d_head {
            let mut acc = 0.0f32;
            for k in 0..seq_len {
                let v_idx = v_base + k * d_head + j;
                let v = if v_idx < global_memory.len() { global_memory[v_idx] } else { 0.0 };
                acc += scores[i * seq_len + k] * v;
            }
            let idx = out_base + i * d_head + j;
            if idx < smem.len() {
                smem[idx] = acc;
            }
        }
    }
}

/// Fused layer-norm on shared_memory[src1..src1+src2] → writes to shared_memory[dst..].
fn cpu_fused_layernorm(
    smem: &mut Vec<f32>,
    dst: usize,
    src_base: usize,
    len: usize,
) {
    let n = len.min(smem.len().saturating_sub(src_base));
    if n == 0 {
        return;
    }
    let mean: f32 = smem[src_base..src_base + n].iter().sum::<f32>() / n as f32;
    let var: f32 = smem[src_base..src_base + n]
        .iter()
        .map(|x| (x - mean) * (x - mean))
        .sum::<f32>()
        / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in 0..n {
        let out_idx = dst + i;
        if out_idx < smem.len() {
            smem[out_idx] = (smem[src_base + i] - mean) * inv_std;
        }
    }
}

/// Fused softmax on shared_memory[src1..src1+dst] (dst = length).
fn cpu_fused_softmax(smem: &mut Vec<f32>, len: usize, src_base: usize) {
    let n = len.min(smem.len().saturating_sub(src_base));
    if n == 0 {
        return;
    }
    let max_val = smem[src_base..src_base + n]
        .iter()
        .cloned()
        .fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in 0..n {
        smem[src_base + i] = (smem[src_base + i] - max_val).exp();
        sum += smem[src_base + i];
    }
    if sum > 0.0 {
        for i in 0..n {
            smem[src_base + i] /= sum;
        }
    }
}

/// Fused RMS-norm on shared_memory.
fn cpu_fused_rmsnorm(
    smem: &mut Vec<f32>,
    dst: usize,
    src_base: usize,
    len: usize,
) {
    let n = len.min(smem.len().saturating_sub(src_base));
    if n == 0 {
        return;
    }
    let rms: f32 = (smem[src_base..src_base + n]
        .iter()
        .map(|x| x * x)
        .sum::<f32>()
        / n as f32
        + 1e-5)
        .sqrt();
    let inv_rms = 1.0 / rms;
    for i in 0..n {
        let out_idx = dst + i;
        if out_idx < smem.len() {
            smem[out_idx] = smem[src_base + i] * inv_rms;
        }
    }
}

// ---------------------------------------------------------------------------
// LLM SIMT Batch Scheduler
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct SequenceState {
    tokens: Vec<u32>,
    position: usize,
    finished: bool,
    draft_tokens: Vec<u32>, // speculative decoding
    expert_id: Option<usize>,
}

#[derive(Clone)]
pub struct SIMTBatchScheduler {
    sequences: Vec<SequenceState>,
    next_seq_id: usize,
    pending_queue: Vec<Vec<u32>>,
    max_batch_size: usize,
}

impl SIMTBatchScheduler {
    pub fn new(max_batch_size: usize) -> Self {
        SIMTBatchScheduler {
            sequences: Vec::new(),
            next_seq_id: 0,
            pending_queue: Vec::new(),
            max_batch_size,
        }
    }

    /// Submit a new token sequence for processing. Returns sequence ID.
    pub fn submit(&mut self, tokens: Vec<u32>) -> usize {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        if self.sequences.len() < self.max_batch_size {
            self.sequences.push(SequenceState {
                tokens,
                position: 0,
                finished: false,
                draft_tokens: Vec::new(),
                expert_id: None,
            });
        } else {
            self.pending_queue.push(tokens);
        }
        seq_id
    }

    /// Advance all active sequences one decode step.
    /// In real use this would invoke the model; here we simulate by advancing position.
    pub fn step(&mut self, grid: &mut SIMTGrid) {
        for (i, seq) in self.sequences.iter_mut().enumerate() {
            if seq.finished {
                continue;
            }
            seq.position += 1;
            if seq.position >= seq.tokens.len() {
                seq.finished = true;
            }
            // Map sequence → block, position → lane within block
            if i < grid.blocks.len() {
                let block = &mut grid.blocks[i];
                for warp in block.warps.iter_mut() {
                    for lane in warp.lanes.iter_mut() {
                        lane.sequence_id = i;
                        lane.step = seq.position;
                        if lane.token_pos < seq.tokens.len() {
                            lane.register_file[0] = seq.tokens[lane.token_pos] as f32;
                        }
                    }
                }
            }
        }
    }

    /// Speculative decoding step: lanes split for draft vs verify.
    pub fn speculative_step(&mut self, grid: &mut SIMTGrid) {
        for (i, seq) in self.sequences.iter_mut().enumerate() {
            if seq.finished {
                continue;
            }
            // Generate draft tokens (simulate: copy next 4 tokens as draft)
            let draft_count = 4.min(seq.tokens.len().saturating_sub(seq.position));
            seq.draft_tokens.clear();
            for d in 0..draft_count {
                let pos = seq.position + d;
                if pos < seq.tokens.len() {
                    seq.draft_tokens.push(seq.tokens[pos]);
                }
            }
            // Map to grid: first half of warp lanes = draft, second half = verify
            if i < grid.blocks.len() {
                let block = &mut grid.blocks[i];
                for warp in block.warps.iter_mut() {
                    for (lane_idx, lane) in warp.lanes.iter_mut().enumerate() {
                        if lane_idx < LANES_PER_WARP / 2 {
                            // Draft lane
                            lane.active = true;
                            lane.token_pos = seq.position + (lane_idx % draft_count.max(1));
                        } else {
                            // Verify lane — active only if we have drafts to verify
                            lane.active = !seq.draft_tokens.is_empty();
                            let draft_idx = (lane_idx - LANES_PER_WARP / 2) % draft_count.max(1);
                            if draft_idx < seq.draft_tokens.len() {
                                lane.register_file[0] = seq.draft_tokens[draft_idx] as f32;
                            }
                        }
                    }
                }
            }
            // Accept drafts (in simulation, accept all)
            seq.position += draft_count;
            if seq.position >= seq.tokens.len() {
                seq.finished = true;
            }
        }
    }

    /// MoE dispatch: route lanes to expert warps based on router scores.
    pub fn moe_dispatch(
        &mut self,
        grid: &mut SIMTGrid,
        router_scores: &[f32],
        num_experts: usize,
    ) {
        if num_experts == 0 {
            return;
        }
        // For each sequence, pick top expert from router scores
        for (i, seq) in self.sequences.iter_mut().enumerate() {
            if seq.finished {
                continue;
            }
            let expert = if i < router_scores.len() {
                // Pick the expert with highest score for this sequence
                let base = i * num_experts;
                let end = (base + num_experts).min(router_scores.len());
                if base < end {
                    let (max_idx, _) = router_scores[base..end]
                        .iter()
                        .enumerate()
                        .fold((0, f32::NEG_INFINITY), |(mi, mv), (idx, &v)| {
                            if v > mv { (idx, v) } else { (mi, mv) }
                        });
                    max_idx
                } else {
                    0
                }
            } else {
                i % num_experts
            };
            seq.expert_id = Some(expert);

            // Map expert to warp index within block
            if i < grid.blocks.len() {
                let warp_idx = expert % WARPS_PER_BLOCK;
                let block = &mut grid.blocks[i];
                // Deactivate all warps, activate only the expert warp
                for (wi, warp) in block.warps.iter_mut().enumerate() {
                    let should_active = wi == warp_idx;
                    for lane in warp.lanes.iter_mut() {
                        lane.active = should_active;
                    }
                }
            }
        }
    }

    /// Rebalance: reassign finished sequence lanes to pending sequences (continuous batching).
    pub fn rebalance(&mut self) {
        let mut freed = Vec::new();
        for (i, seq) in self.sequences.iter().enumerate() {
            if seq.finished {
                freed.push(i);
            }
        }
        for slot in freed {
            if let Some(tokens) = self.pending_queue.pop() {
                self.sequences[slot] = SequenceState {
                    tokens,
                    position: 0,
                    finished: false,
                    draft_tokens: Vec::new(),
                    expert_id: None,
                };
            }
        }
    }

    pub fn active_count(&self) -> usize {
        self.sequences.iter().filter(|s| !s.finished).count()
    }

    pub fn pending_count(&self) -> usize {
        self.pending_queue.len()
    }

    pub fn total_sequences(&self) -> usize {
        self.sequences.len()
    }
}

// ---------------------------------------------------------------------------
// GPU Dispatch Path (stub — uses drm_driver when available)
// ---------------------------------------------------------------------------

fn gpu_available() -> bool {
    // Check if drm_driver module exposes a device.
    // For now, always return false (CPU SIMT simulation).
    false
}

#[allow(dead_code)]
fn gpu_dispatch(
    _kernel: &[u32],
    _global_memory: &[f32],
    _grid_dim: [usize; 3],
    _block_dim: [usize; 3],
) -> Result<Vec<f32>, String> {
    Err("GPU dispatch not available — using CPU SIMT simulation".into())
}

// ---------------------------------------------------------------------------
// SIMTRuntime — global state
// ---------------------------------------------------------------------------

pub struct SIMTRuntime {
    pub grid: SIMTGrid,
    pub scheduler: SIMTBatchScheduler,
    pub global_memory: Vec<f32>,
    pub initialized: bool,
}

impl SIMTRuntime {
    fn new() -> Self {
        SIMTRuntime {
            grid: SIMTGrid::new([1, 1, 1], [128, 1, 1]),
            scheduler: SIMTBatchScheduler::new(32),
            global_memory: vec![0.0; 65536],
            initialized: false,
        }
    }
}

static SIMT_RUNTIME: LazyLock<Mutex<SIMTRuntime>> =
    LazyLock::new(|| Mutex::new(SIMTRuntime::new()));

// ---------------------------------------------------------------------------
// Builtins
// ---------------------------------------------------------------------------

fn builtin_simt_init(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let grid_x = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 1,
    };
    let grid_y = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 1,
    };
    let grid_z = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 1,
    };
    let block_size = match args.get(3) {
        Some(Value::Int(n)) => *n as usize,
        _ => 128,
    };
    let mem_size = match args.get(4) {
        Some(Value::Int(n)) => *n as usize,
        _ => 65536,
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    rt.grid = SIMTGrid::new([grid_x, grid_y, grid_z], [block_size, 1, 1]);
    rt.global_memory = vec![0.0; mem_size];
    rt.scheduler = SIMTBatchScheduler::new(grid_x * grid_y * grid_z);
    rt.initialized = true;

    let total = rt.grid.total_lanes();
    Ok(Value::String(format!(
        "SIMT initialized: {}x{}x{} grid, {} lanes, {} global mem, backend={}",
        grid_x, grid_y, grid_z, total, mem_size,
        if gpu_available() { "GPU" } else { "CPU-SIMT" }
    )))
}

fn builtin_simt_launch(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let kernel: Vec<u32> = match args.first() {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::Int(n) => *n as u32,
                _ => 0,
            })
            .collect(),
        _ => return Err("simt_launch: first arg must be array of instruction u32s".into()),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    if !rt.initialized {
        return Err("simt_launch: runtime not initialized, call simt_init first".into());
    }

    let rt_ref = &mut *rt;
    if gpu_available() {
        match gpu_dispatch(&kernel, &rt_ref.global_memory, rt_ref.grid.grid_dim, rt_ref.grid.block_dim) {
            Ok(result_mem) => {
                rt_ref.global_memory = result_mem;
            }
            Err(_) => {
                execute_vxb(&kernel, &mut rt_ref.grid, &mut rt_ref.global_memory);
            }
        }
    } else {
        execute_vxb(&kernel, &mut rt_ref.grid, &mut rt_ref.global_memory);
    }

    Ok(Value::String("kernel executed".into()))
}

fn builtin_simt_matmul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let m = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 16,
    };
    let n = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 16,
    };
    let k = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 16,
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    if !rt.initialized {
        return Err("simt_matmul: runtime not initialized".into());
    }

    // Perform matmul on global_memory: A at offset 0, B at offset m*k, C at offset m*k + k*n
    let a_base = 0;
    let b_base = m * k;
    let c_base = b_base + k * n;

    // Ensure enough memory
    let needed = c_base + m * n;
    if rt.global_memory.len() < needed {
        rt.global_memory.resize(needed, 0.0);
    }

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for kk in 0..k {
                acc += rt.global_memory[a_base + i * k + kk]
                    * rt.global_memory[b_base + kk * n + j];
            }
            rt.global_memory[c_base + i * n + j] = acc;
        }
    }

    Ok(Value::String(format!("matmul {}x{}x{} complete", m, n, k)))
}

fn builtin_simt_softmax(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let offset = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let len = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("simt_softmax: need offset and length".into()),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let mem = &mut rt.global_memory;
    let end = (offset + len).min(mem.len());
    if offset >= end {
        return Err("simt_softmax: invalid range".into());
    }

    let max_val = mem[offset..end].iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for i in offset..end {
        mem[i] = (mem[i] - max_val).exp();
        sum += mem[i];
    }
    if sum > 0.0 {
        for i in offset..end {
            mem[i] /= sum;
        }
    }

    Ok(Value::String(format!("softmax applied to {} elements", end - offset)))
}

fn builtin_simt_attention(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let seq_len = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 16,
    };
    let d_head = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 16,
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    if !rt.initialized {
        return Err("simt_attention: runtime not initialized".into());
    }

    // Use first block for attention compute
    if rt.grid.blocks.is_empty() {
        return Err("simt_attention: no blocks in grid".into());
    }

    let qk_base = 0;
    let v_base = seq_len * d_head * 2; // V starts after Q and K
    let out_base = 0;
    let rt_ref = &mut *rt;
    cpu_fused_attention(&mut rt_ref.grid.blocks[0].shared_memory, &mut rt_ref.global_memory, out_base, qk_base, v_base);

    Ok(Value::String(format!(
        "attention computed: seq_len={}, d_head={}",
        seq_len, d_head
    )))
}

fn builtin_simt_layernorm(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let offset = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let len = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("simt_layernorm: need offset and length".into()),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let mem = &mut rt.global_memory;
    let end = (offset + len).min(mem.len());
    if offset >= end {
        return Ok(Value::String("layernorm: empty range".into()));
    }
    let n = end - offset;
    let mean: f32 = mem[offset..end].iter().sum::<f32>() / n as f32;
    let var: f32 = mem[offset..end].iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + 1e-5).sqrt();
    for i in offset..end {
        mem[i] = (mem[i] - mean) * inv_std;
    }

    Ok(Value::String(format!("layernorm applied to {} elements", n)))
}

fn builtin_simt_submit_sequence(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let tokens: Vec<u32> = match args.first() {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::Int(n) => *n as u32,
                _ => 0,
            })
            .collect(),
        _ => return Err("simt_submit_sequence: need array of token ints".into()),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let seq_id = rt.scheduler.submit(tokens);
    Ok(Value::Int(seq_id as i128))
}

fn builtin_simt_step(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let rt_ref = &mut *rt;
    rt_ref.scheduler.step(&mut rt_ref.grid);
    let active = rt_ref.scheduler.active_count();
    let pending = rt_ref.scheduler.pending_count();
    Ok(Value::String(format!(
        "step complete: {} active, {} pending",
        active, pending
    )))
}

fn builtin_simt_speculative_step(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let rt_ref = &mut *rt;
    rt_ref.scheduler.speculative_step(&mut rt_ref.grid);
    let active = rt_ref.scheduler.active_count();
    Ok(Value::String(format!(
        "speculative step complete: {} active sequences",
        active
    )))
}

fn builtin_simt_moe_dispatch(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let router_scores: Vec<f32> = match args.first() {
        Some(Value::Array(arr)) => arr
            .iter()
            .map(|v| match v {
                Value::Float(f) => *f as f32,
                Value::Int(n) => *n as f32,
                _ => 0.0,
            })
            .collect(),
        _ => return Err("simt_moe_dispatch: first arg must be array of scores".into()),
    };
    let num_experts = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => return Err("simt_moe_dispatch: second arg must be num_experts int".into()),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    let rt_ref = &mut *rt;
    rt_ref.scheduler.moe_dispatch(&mut rt_ref.grid, &router_scores, num_experts);
    Ok(Value::String(format!(
        "MoE dispatch: {} experts, {} sequences",
        num_experts,
        rt_ref.scheduler.active_count()
    )))
}

fn builtin_simt_warp_reduce(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let block_idx = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let warp_idx = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let reg = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let op = match args.get(3) {
        Some(Value::String(s)) => s.clone(),
        _ => "sum".into(),
    };

    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    if block_idx >= rt.grid.blocks.len() {
        return Err("simt_warp_reduce: block index out of range".into());
    }
    let block = &mut rt.grid.blocks[block_idx];
    if warp_idx >= block.warps.len() {
        return Err("simt_warp_reduce: warp index out of range".into());
    }
    let warp = &mut block.warps[warp_idx];
    let result = match op.as_str() {
        "sum" => warp.reduce_sum(reg),
        "max" => warp.reduce_max(reg),
        _ => return Err(format!("simt_warp_reduce: unknown op '{}'", op)),
    };

    Ok(Value::Float(result as f64))
}

fn builtin_simt_lane_info(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let block_idx = match args.first() {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };
    let flat_lane = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 0,
    };

    let rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    if block_idx >= rt.grid.blocks.len() {
        return Err("simt_lane_info: block out of range".into());
    }
    let block = &rt.grid.blocks[block_idx];
    if flat_lane >= LANES_PER_BLOCK {
        return Err("simt_lane_info: lane out of range".into());
    }
    let lane = block.lane(flat_lane);
    let (warp_idx, lane_idx) = SIMTBlock::flat_to_warp_lane(flat_lane);

    let mut fields = std::collections::HashMap::new();
    fields.insert("warp".into(), Value::Int(warp_idx as i128));
    fields.insert("lane".into(), Value::Int(lane_idx as i128));
    fields.insert("active".into(), Value::Bool(lane.active));
    fields.insert("token_pos".into(), Value::Int(lane.token_pos as i128));
    fields.insert("sequence_id".into(), Value::Int(lane.sequence_id as i128));
    fields.insert("step".into(), Value::Int(lane.step as i128));
    fields.insert(
        "r0".into(),
        Value::Float(lane.register_file[0] as f64),
    );

    Ok(Value::Struct {
        name: "SIMTLaneInfo".into(),
        fields,
    })
}

fn builtin_simt_benchmark(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let op = match args.first() {
        Some(Value::String(s)) => s.clone(),
        _ => "matmul".into(),
    };
    let size = match args.get(1) {
        Some(Value::Int(n)) => *n as usize,
        _ => 256,
    };
    let iterations = match args.get(2) {
        Some(Value::Int(n)) => *n as usize,
        _ => 10,
    };

    let start = std::time::Instant::now();
    match op.as_str() {
        "matmul" => {
            let mut a = vec![0.5f32; size * size];
            let mut b = vec![0.3f32; size * size];
            let mut c = vec![0.0f32; size * size];
            for _ in 0..iterations {
                for i in 0..size {
                    for j in 0..size {
                        let mut acc = 0.0f32;
                        for k in 0..size {
                            acc += a[i * size + k] * b[k * size + j];
                        }
                        c[i * size + j] = acc;
                    }
                }
                // Prevent optimization
                a[0] = c[0];
                b[0] = c[size - 1];
            }
        }
        "softmax" => {
            let mut data = vec![1.0f32; size];
            for _ in 0..iterations {
                let max_val = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0f32;
                for v in data.iter_mut() {
                    *v = (*v - max_val).exp();
                    sum += *v;
                }
                if sum > 0.0 {
                    for v in data.iter_mut() {
                        *v /= sum;
                    }
                }
            }
        }
        _ => return Err(format!("simt_benchmark: unknown op '{}'", op)),
    }
    let elapsed = start.elapsed();
    let flops = match op.as_str() {
        "matmul" => 2.0 * (size as f64).powi(3) * iterations as f64,
        "softmax" => 3.0 * size as f64 * iterations as f64,
        _ => 0.0,
    };
    let gflops = flops / elapsed.as_secs_f64() / 1e9;

    Ok(Value::String(format!(
        "benchmark {}: {}x{} x{} iters in {:.3}ms ({:.2} GFLOPS)",
        op,
        size,
        size,
        iterations,
        elapsed.as_secs_f64() * 1000.0,
        gflops
    )))
}

fn builtin_simt_rebalance(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mut rt = SIMT_RUNTIME.lock().map_err(|e| e.to_string())?;
    rt.scheduler.rebalance();
    let active = rt.scheduler.active_count();
    let pending = rt.scheduler.pending_count();
    Ok(Value::String(format!(
        "rebalance complete: {} active, {} pending",
        active, pending
    )))
}

// ---------------------------------------------------------------------------
// Register all builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("simt_init".into(), FnDef::Builtin(builtin_simt_init));
    env.functions.insert("simt_launch".into(), FnDef::Builtin(builtin_simt_launch));
    env.functions.insert("simt_matmul".into(), FnDef::Builtin(builtin_simt_matmul));
    env.functions.insert("simt_softmax".into(), FnDef::Builtin(builtin_simt_softmax));
    env.functions.insert("simt_attention".into(), FnDef::Builtin(builtin_simt_attention));
    env.functions.insert("simt_layernorm".into(), FnDef::Builtin(builtin_simt_layernorm));
    env.functions.insert(
        "simt_submit_sequence".into(),
        FnDef::Builtin(builtin_simt_submit_sequence),
    );
    env.functions.insert("simt_step".into(), FnDef::Builtin(builtin_simt_step));
    env.functions.insert(
        "simt_speculative_step".into(),
        FnDef::Builtin(builtin_simt_speculative_step),
    );
    env.functions.insert(
        "simt_moe_dispatch".into(),
        FnDef::Builtin(builtin_simt_moe_dispatch),
    );
    env.functions.insert(
        "simt_warp_reduce".into(),
        FnDef::Builtin(builtin_simt_warp_reduce),
    );
    env.functions.insert("simt_lane_info".into(), FnDef::Builtin(builtin_simt_lane_info));
    env.functions.insert("simt_benchmark".into(), FnDef::Builtin(builtin_simt_benchmark));
    env.functions.insert("simt_rebalance".into(), FnDef::Builtin(builtin_simt_rebalance));
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_warp_reduce_sum() {
        let mut warp = SIMTWarp::new();
        // Set register 0 of each lane to its lane index
        for (i, lane) in warp.lanes.iter_mut().enumerate() {
            lane.register_file[0] = i as f32;
        }
        let sum = warp.reduce_sum(0);
        // Sum of 0..31 = 31*32/2 = 496
        assert!((sum - 496.0).abs() < 1e-3, "reduce_sum got {}", sum);
        assert!((warp.lanes[0].register_file[0] - 496.0).abs() < 1e-3);
    }

    #[test]
    fn test_warp_reduce_max() {
        let mut warp = SIMTWarp::new();
        for (i, lane) in warp.lanes.iter_mut().enumerate() {
            lane.register_file[1] = (i as f32) * 2.0 - 30.0;
        }
        let max_val = warp.reduce_max(1);
        // max = 31*2 - 30 = 32
        assert!((max_val - 32.0).abs() < 1e-3, "reduce_max got {}", max_val);
    }

    #[test]
    fn test_warp_shuffle_down() {
        let mut warp = SIMTWarp::new();
        for (i, lane) in warp.lanes.iter_mut().enumerate() {
            lane.register_file[0] = i as f32;
        }
        warp.shuffle_down(0, 1);
        // Lane 0 should now have value 1, lane 1 should have 2, etc.
        assert!((warp.lanes[0].register_file[0] - 1.0).abs() < 1e-6);
        assert!((warp.lanes[1].register_file[0] - 2.0).abs() < 1e-6);
        // Lane 31 should still have 31 (no source beyond)
        assert!((warp.lanes[31].register_file[0] - 31.0).abs() < 1e-6);
    }

    #[test]
    fn test_warp_vote() {
        let mut warp = SIMTWarp::new();
        for lane in warp.lanes.iter_mut() {
            lane.register_file[0] = 1.0;
        }
        assert!(warp.vote_all(0));
        assert!(warp.vote_any(0));

        warp.lanes[5].register_file[0] = 0.0;
        assert!(!warp.vote_all(0));
        assert!(warp.vote_any(0));

        for lane in warp.lanes.iter_mut() {
            lane.register_file[0] = 0.0;
        }
        assert!(!warp.vote_all(0));
        assert!(!warp.vote_any(0));
    }

    #[test]
    fn test_warp_broadcast() {
        let mut warp = SIMTWarp::new();
        warp.lanes[7].register_file[2] = 42.0;
        warp.broadcast(2, 7);
        for lane in &warp.lanes {
            assert!((lane.register_file[2] - 42.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_barrier_sync() {
        let mut block = SIMTBlock::new();
        assert_eq!(block.barrier_count, 0);
        block.barrier_sync();
        assert_eq!(block.barrier_count, 1);
        block.barrier_sync();
        assert_eq!(block.barrier_count, 2);
    }

    #[test]
    fn test_divergence() {
        let mut warp = SIMTWarp::new();
        // Set predicate: even lanes = 1.0, odd lanes = 0.0
        for (i, lane) in warp.lanes.iter_mut().enumerate() {
            lane.register_file[0] = if i % 2 == 0 { 1.0 } else { 0.0 };
        }
        warp.push_divergence(0);
        // Only even lanes should be active
        for (i, lane) in warp.lanes.iter().enumerate() {
            assert_eq!(lane.active, i % 2 == 0, "lane {} after push_divergence", i);
        }

        // Flip to else branch
        warp.flip_divergence();
        for (i, lane) in warp.lanes.iter().enumerate() {
            assert_eq!(lane.active, i % 2 != 0, "lane {} after flip", i);
        }

        // Reconverge
        warp.reconverge();
        for lane in &warp.lanes {
            assert!(lane.active, "all lanes active after reconverge");
        }
    }

    #[test]
    fn test_vxb_arithmetic_kernel() {
        // Simple kernel: r2 = r0 + r1, then halt
        let kernel = vec![
            encode_instruction(Opcode::Add as u8, 2, 0, 1),
            encode_instruction(Opcode::Halt as u8, 0, 0, 0),
        ];

        let mut grid = SIMTGrid::new([1, 1, 1], [128, 1, 1]);
        let mut global_memory = vec![0.0f32; 256];

        // Set r0=3.0, r1=7.0 for all lanes
        for block in grid.blocks.iter_mut() {
            for warp in block.warps.iter_mut() {
                for lane in warp.lanes.iter_mut() {
                    lane.register_file[0] = 3.0;
                    lane.register_file[1] = 7.0;
                }
            }
        }

        execute_vxb(&kernel, &mut grid, &mut global_memory);

        // r2 should be 10.0 for all lanes
        for block in &grid.blocks {
            for warp in &block.warps {
                for (i, lane) in warp.lanes.iter().enumerate() {
                    assert!(
                        (lane.register_file[2] - 10.0).abs() < 1e-6,
                        "lane {} r2 = {}, expected 10.0",
                        i,
                        lane.register_file[2]
                    );
                }
            }
        }
    }

    #[test]
    fn test_vxb_mul_sub_chain() {
        let kernel = vec![
            encode_instruction(Opcode::Mul as u8, 2, 0, 1), // r2 = r0 * r1
            encode_instruction(Opcode::Sub as u8, 3, 2, 0), // r3 = r2 - r0
            encode_instruction(Opcode::Halt as u8, 0, 0, 0),
        ];

        let mut grid = SIMTGrid::new([1, 1, 1], [128, 1, 1]);
        let mut global_memory = vec![0.0f32; 64];

        for block in grid.blocks.iter_mut() {
            for warp in block.warps.iter_mut() {
                for lane in warp.lanes.iter_mut() {
                    lane.register_file[0] = 5.0;
                    lane.register_file[1] = 4.0;
                }
            }
        }

        execute_vxb(&kernel, &mut grid, &mut global_memory);

        for block in &grid.blocks {
            for warp in &block.warps {
                for lane in &warp.lanes {
                    assert!((lane.register_file[2] - 20.0).abs() < 1e-6);
                    assert!((lane.register_file[3] - 15.0).abs() < 1e-6);
                }
            }
        }
    }

    #[test]
    fn test_vxb_global_memory() {
        // Load from global memory, add, store back
        let kernel = vec![
            encode_instruction(Opcode::LoadGlobal as u8, 1, 0, 0), // r1 = global[r0]
            encode_instruction(Opcode::Add as u8, 2, 1, 1),        // r2 = r1 + r1
            encode_instruction(Opcode::StoreGlobal as u8, 0, 0, 2), // global[r0] = r2
            encode_instruction(Opcode::Halt as u8, 0, 0, 0),
        ];

        let mut grid = SIMTGrid::new([1, 1, 1], [128, 1, 1]);
        let mut global_memory = vec![0.0f32; 256];
        global_memory[0] = 21.0;

        // Deactivate all lanes except lane 0 of warp 0 to avoid multi-warp overwrites
        for block in grid.blocks.iter_mut() {
            for (wi, warp) in block.warps.iter_mut().enumerate() {
                for (li, lane) in warp.lanes.iter_mut().enumerate() {
                    lane.register_file[0] = 0.0;
                    lane.active = wi == 0 && li == 0;
                }
            }
        }

        execute_vxb(&kernel, &mut grid, &mut global_memory);
        // global[0] should be 42.0 (21 * 2)
        assert!((global_memory[0] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_scheduler_submit_and_step() {
        let mut scheduler = SIMTBatchScheduler::new(4);
        let mut grid = SIMTGrid::new([4, 1, 1], [128, 1, 1]);

        let id0 = scheduler.submit(vec![10, 20, 30, 40]);
        let id1 = scheduler.submit(vec![100, 200]);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
        assert_eq!(scheduler.active_count(), 2);

        scheduler.step(&mut grid);
        assert_eq!(scheduler.active_count(), 2); // both still have tokens

        scheduler.step(&mut grid);
        // seq1 (len=2) should be finished now (position=2 >= len)
        assert_eq!(scheduler.active_count(), 1);

        scheduler.step(&mut grid);
        scheduler.step(&mut grid);
        // seq0 (len=4) should be finished now
        assert_eq!(scheduler.active_count(), 0);
    }

    #[test]
    fn test_scheduler_rebalance() {
        let mut scheduler = SIMTBatchScheduler::new(2);
        let mut grid = SIMTGrid::new([2, 1, 1], [128, 1, 1]);

        scheduler.submit(vec![1, 2]);
        scheduler.submit(vec![3, 4]);
        // This should go to pending queue (max_batch_size=2)
        scheduler.submit(vec![5, 6, 7]);
        assert_eq!(scheduler.pending_count(), 1);

        // Finish first two
        scheduler.step(&mut grid);
        scheduler.step(&mut grid);
        assert_eq!(scheduler.active_count(), 0);

        // Rebalance should pull from pending
        scheduler.rebalance();
        assert_eq!(scheduler.active_count(), 1); // one rebalanced in
        assert_eq!(scheduler.pending_count(), 0);
    }

    #[test]
    fn test_grid_creation() {
        let grid = SIMTGrid::new([2, 3, 1], [128, 1, 1]);
        assert_eq!(grid.blocks.len(), 6);
        assert_eq!(grid.total_lanes(), 6 * LANES_PER_BLOCK);
        assert_eq!(grid.blocks[0].block_idx, [0, 0, 0]);
        assert_eq!(grid.blocks[1].block_idx, [1, 0, 0]);
        assert_eq!(grid.blocks[2].block_idx, [0, 1, 0]);
    }

    #[test]
    fn test_encode_decode_instruction() {
        let inst = encode_instruction(0x01, 5, 10, 20);
        let (op, dst, src1, src2) = decode_instruction(inst);
        assert_eq!(op, 0x01);
        assert_eq!(dst, 5);
        assert_eq!(src1, 10);
        assert_eq!(src2, 20);
    }
}
