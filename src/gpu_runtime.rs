//! GPU Runtime Abstraction Layer for Vortex.
//!
//! Phase 3: The layer between compiled MLIR and actual GPU execution.
//! Provides backend-agnostic buffer management, kernel launch configuration,
//! a full CPU fallback runtime for testing, and a memory pool allocator.

use std::collections::{BTreeMap, HashMap};

// ---------------------------------------------------------------------------
// 1. Backend Abstraction
// ---------------------------------------------------------------------------

/// Supported GPU backends.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GpuBackend {
    Cuda,
    Rocm,
    Vulkan,
    Cpu, // fallback for testing
}

impl std::fmt::Display for GpuBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GpuBackend::Cuda => write!(f, "CUDA"),
            GpuBackend::Rocm => write!(f, "ROCm"),
            GpuBackend::Vulkan => write!(f, "Vulkan"),
            GpuBackend::Cpu => write!(f, "CPU"),
        }
    }
}

/// Trait describing a GPU device's capabilities.
pub trait GpuDevice {
    fn name(&self) -> &str;
    fn memory_total(&self) -> usize;
    fn memory_free(&self) -> usize;
    fn compute_capability(&self) -> (u32, u32);
    fn max_threads_per_block(&self) -> u32;
    fn max_shared_memory(&self) -> usize;
    fn warp_size(&self) -> u32;
}

/// A CPU-based "device" used when no real GPU is available.
pub struct CpuDevice {
    pub total_mem: usize,
    pub used_mem: usize,
}

impl CpuDevice {
    pub fn new() -> Self {
        Self {
            total_mem: 16 * 1024 * 1024 * 1024, // 16 GB virtual
            used_mem: 0,
        }
    }
}

impl GpuDevice for CpuDevice {
    fn name(&self) -> &str {
        "Vortex CPU Fallback"
    }
    fn memory_total(&self) -> usize {
        self.total_mem
    }
    fn memory_free(&self) -> usize {
        self.total_mem - self.used_mem
    }
    fn compute_capability(&self) -> (u32, u32) {
        (0, 0)
    }
    fn max_threads_per_block(&self) -> u32 {
        1024
    }
    fn max_shared_memory(&self) -> usize {
        164 * 1024
    }
    fn warp_size(&self) -> u32 {
        1
    }
}

// ---------------------------------------------------------------------------
// 2. DType & Buffer Management
// ---------------------------------------------------------------------------

/// Data types supported by Vortex GPU buffers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    F64,
    F32,
    F16,
    BF16,
    I64,
    I32,
    I16,
    I8,
    U64,
    U32,
    U16,
    U8,
    Bool,
    // Quantized / low-precision types -- Vortex innovation
    Int4x2,  // two INT4 packed in one byte
    FP8E4M3, // 8-bit float (e4m3)
    FP8E5M2, // 8-bit float (e5m2)
}

impl DType {
    /// Size in bytes of a single element (for packed types, size of the pack unit).
    pub fn size_bytes(&self) -> usize {
        match self {
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
            DType::Int4x2 => 1,  // 2 elements per byte
            DType::FP8E4M3 | DType::FP8E5M2 => 1,
        }
    }

    /// How many logical elements fit in one `size_bytes()` unit.
    pub fn elements_per_unit(&self) -> usize {
        match self {
            DType::Int4x2 => 2,
            _ => 1,
        }
    }
}

/// A GPU buffer (or CPU-backed buffer in fallback mode).
#[derive(Debug, Clone)]
pub struct GpuBuffer {
    pub id: usize,
    pub size: usize, // size in bytes
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub device: GpuBackend,
    /// Actual data storage for CPU fallback
    pub cpu_data: Option<Vec<u8>>,
}

impl GpuBuffer {
    /// Allocate a new buffer.
    pub fn alloc(shape: &[usize], dtype: DType, backend: GpuBackend) -> Self {
        let n_elements: usize = shape.iter().product();
        let size = (n_elements + dtype.elements_per_unit() - 1)
            / dtype.elements_per_unit()
            * dtype.size_bytes();
        let cpu_data = if backend == GpuBackend::Cpu {
            Some(vec![0u8; size])
        } else {
            None
        };
        Self {
            id: 0, // caller must assign
            size,
            dtype,
            shape: shape.to_vec(),
            device: backend,
            cpu_data,
        }
    }

    /// Create a buffer from f64 data (stored as F64 on CPU backend).
    pub fn from_f64(data: &[f64], shape: &[usize], backend: GpuBackend) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(data.len(), n, "data length must match shape product");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        Self {
            id: 0,
            size: bytes.len(),
            dtype: DType::F64,
            shape: shape.to_vec(),
            device: backend,
            cpu_data: Some(bytes),
        }
    }

    /// Read back as f64 (only works for F64 dtype with cpu_data).
    pub fn to_f64(&self) -> Vec<f64> {
        let data = self.cpu_data.as_ref().expect("no cpu_data in buffer");
        data.chunks_exact(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    /// Copy contents to another buffer.
    pub fn copy_to(&self, dst: &mut GpuBuffer) {
        assert_eq!(self.size, dst.size, "buffer sizes must match for copy");
        if let (Some(src), Some(d)) = (&self.cpu_data, &mut dst.cpu_data) {
            d.copy_from_slice(src);
        }
    }

    /// Reshape (must preserve total element count).
    pub fn reshape(&mut self, new_shape: &[usize]) {
        let old_n: usize = self.shape.iter().product();
        let new_n: usize = new_shape.iter().product();
        assert_eq!(old_n, new_n, "reshape must preserve element count");
        self.shape = new_shape.to_vec();
    }

    /// Total size in bytes.
    pub fn size_bytes(&self) -> usize {
        self.size
    }

    /// Total number of logical elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}

// ---------------------------------------------------------------------------
// 3. Kernel Launch Abstraction
// ---------------------------------------------------------------------------

/// A scalar value that can be passed as a kernel argument.
#[derive(Debug, Clone)]
pub enum ScalarValue {
    F64(f64),
    F32(f32),
    I64(i64),
    I32(i32),
    U64(u64),
    U32(u32),
}

/// An argument to a GPU kernel.
#[derive(Debug, Clone)]
pub enum KernelArg {
    BufferId(usize),
    Scalar(ScalarValue),
}

/// A kernel ready to launch.
#[derive(Debug, Clone)]
pub struct KernelLaunch {
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: usize,
    pub args: Vec<KernelArg>,
}

/// Launch configuration helper.
#[derive(Debug, Clone)]
pub struct LaunchConfig {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub shared_mem: usize,
}

impl LaunchConfig {
    /// Auto-configure for a 1D workload of `n` elements.
    pub fn auto_1d(n: usize, block_size: u32) -> Self {
        let grid_x = ((n as u32) + block_size - 1) / block_size;
        Self {
            grid: [grid_x, 1, 1],
            block: [block_size, 1, 1],
            shared_mem: 0,
        }
    }

    /// Auto-configure for a 2D workload (rows x cols).
    pub fn auto_2d(rows: usize, cols: usize) -> Self {
        let bx = 16u32;
        let by = 16u32;
        let gx = ((cols as u32) + bx - 1) / bx;
        let gy = ((rows as u32) + by - 1) / by;
        Self {
            grid: [gx, gy, 1],
            block: [bx, by, 1],
            shared_mem: 0,
        }
    }

    /// Auto-configure for matrix multiply (M x K) @ (K x N).
    pub fn auto_matmul(m: usize, n: usize, _k: usize) -> Self {
        let bx = 16u32;
        let by = 16u32;
        let gx = ((n as u32) + bx - 1) / bx;
        let gy = ((m as u32) + by - 1) / by;
        Self {
            grid: [gx, gy, 1],
            block: [bx, by, 1],
            shared_mem: (bx * by * 8) as usize, // tile of f64
        }
    }
}

// ---------------------------------------------------------------------------
// 4. CPU Fallback Runtime
// ---------------------------------------------------------------------------

/// CPU-based runtime that executes tensor ops without a GPU.
pub struct CpuRuntime {
    buffers: HashMap<usize, GpuBuffer>,
    next_id: usize,
}

impl CpuRuntime {
    pub fn new() -> Self {
        Self {
            buffers: HashMap::new(),
            next_id: 1,
        }
    }

    fn assign_id(&mut self) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Allocate a zeroed buffer, return its id.
    pub fn alloc(&mut self, shape: &[usize], dtype: DType) -> usize {
        let id = self.assign_id();
        let mut buf = GpuBuffer::alloc(shape, dtype, GpuBackend::Cpu);
        buf.id = id;
        self.buffers.insert(id, buf);
        id
    }

    /// Free a buffer.
    pub fn free(&mut self, id: usize) {
        self.buffers.remove(&id);
    }

    /// Get a buffer reference.
    pub fn get(&self, id: usize) -> &GpuBuffer {
        self.buffers.get(&id).expect("buffer not found")
    }

    /// Read f64 data from a buffer.
    pub fn copy_data(&self, id: usize) -> Vec<f64> {
        self.get(id).to_f64()
    }

    /// Write f64 data into a buffer.
    pub fn write_data(&mut self, id: usize, data: &[f64]) {
        let buf = self.buffers.get_mut(&id).expect("buffer not found");
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        if let Some(ref mut cpu) = buf.cpu_data {
            cpu[..bytes.len()].copy_from_slice(&bytes);
        }
    }

    // -- Tensor operations --

    fn read_f64_pair(&self, a: usize, b: usize) -> (Vec<f64>, Vec<f64>, Vec<usize>, Vec<usize>) {
        let ba = self.get(a);
        let bb = self.get(b);
        (ba.to_f64(), bb.to_f64(), ba.shape.clone(), bb.shape.clone())
    }

    /// Matrix multiply: (M x K) @ (K x N) -> (M x N)
    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let (ad, bd, ash, bsh) = self.read_f64_pair(a, b);
        assert_eq!(ash.len(), 2, "matmul requires 2D tensors");
        assert_eq!(bsh.len(), 2, "matmul requires 2D tensors");
        let (m, k1) = (ash[0], ash[1]);
        let (k2, n) = (bsh[0], bsh[1]);
        assert_eq!(k1, k2, "inner dimensions must match");

        let mut out = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for p in 0..k1 {
                    sum += ad[i * k1 + p] * bd[p * n + j];
                }
                out[i * n + j] = sum;
            }
        }
        let id = self.alloc(&[m, n], DType::F64);
        self.write_data(id, &out);
        id
    }

    /// Element-wise addition.
    pub fn elementwise_add(&mut self, a: usize, b: usize) -> usize {
        let (ad, bd, shape, _) = self.read_f64_pair(a, b);
        assert_eq!(ad.len(), bd.len());
        let out: Vec<f64> = ad.iter().zip(bd.iter()).map(|(x, y)| x + y).collect();
        let id = self.alloc(&shape, DType::F64);
        self.write_data(id, &out);
        id
    }

    /// Element-wise multiplication.
    pub fn elementwise_mul(&mut self, a: usize, b: usize) -> usize {
        let (ad, bd, shape, _) = self.read_f64_pair(a, b);
        assert_eq!(ad.len(), bd.len());
        let out: Vec<f64> = ad.iter().zip(bd.iter()).map(|(x, y)| x * y).collect();
        let id = self.alloc(&shape, DType::F64);
        self.write_data(id, &out);
        id
    }

    /// ReLU activation.
    pub fn relu(&mut self, a: usize) -> usize {
        let buf = self.get(a);
        let data = buf.to_f64();
        let shape = buf.shape.clone();
        let out: Vec<f64> = data.iter().map(|&x| if x > 0.0 { x } else { 0.0 }).collect();
        let id = self.alloc(&shape, DType::F64);
        self.write_data(id, &out);
        id
    }

    /// Softmax along a given axis.
    pub fn softmax(&mut self, a: usize, axis: usize) -> usize {
        let buf = self.get(a);
        let data = buf.to_f64();
        let shape = buf.shape.clone();

        if shape.len() == 1 || (shape.len() == 2 && axis == 1) {
            // Row-wise softmax for 2D or full softmax for 1D
            let cols = if shape.len() == 1 { shape[0] } else { shape[1] };
            let rows = if shape.len() == 1 { 1 } else { shape[0] };
            let mut out = vec![0.0f64; data.len()];
            for r in 0..rows {
                let start = r * cols;
                let row = &data[start..start + cols];
                let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
                let sum: f64 = exps.iter().sum();
                for c in 0..cols {
                    out[start + c] = exps[c] / sum;
                }
            }
            let id = self.alloc(&shape, DType::F64);
            self.write_data(id, &out);
            id
        } else {
            // Fallback: flatten softmax
            let max = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = data.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let out: Vec<f64> = exps.iter().map(|e| e / sum).collect();
            let id = self.alloc(&shape, DType::F64);
            self.write_data(id, &out);
            id
        }
    }

    /// Layer normalization: y = gamma * (x - mean) / sqrt(var + eps) + beta
    pub fn layer_norm(&mut self, x: usize, gamma: usize, beta: usize, eps: f64) -> usize {
        let xd = self.get(x).to_f64();
        let gd = self.get(gamma).to_f64();
        let bd = self.get(beta).to_f64();
        let shape = self.get(x).shape.clone();
        let n = xd.len() as f64;
        let mean = xd.iter().sum::<f64>() / n;
        let var = xd.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n;
        let std = (var + eps).sqrt();
        let out: Vec<f64> = xd
            .iter()
            .enumerate()
            .map(|(i, &v)| gd[i % gd.len()] * (v - mean) / std + bd[i % bd.len()])
            .collect();
        let id = self.alloc(&shape, DType::F64);
        self.write_data(id, &out);
        id
    }

    /// Cross-entropy loss: -sum(log(softmax(logits)[target])) / batch
    pub fn cross_entropy(&mut self, logits: usize, targets: &[usize]) -> f64 {
        let buf = self.get(logits);
        let data = buf.to_f64();
        let shape = &buf.shape;
        assert_eq!(shape.len(), 2, "cross_entropy requires 2D logits");
        let (rows, cols) = (shape[0], shape[1]);
        assert_eq!(rows, targets.len());

        let mut loss = 0.0;
        for r in 0..rows {
            let start = r * cols;
            let row = &data[start..start + cols];
            let max = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|&x| (x - max).exp()).collect();
            let sum: f64 = exps.iter().sum();
            let log_softmax = (exps[targets[r]] / sum).ln();
            loss -= log_softmax;
        }
        loss / rows as f64
    }

    /// Transpose a 2D tensor.
    pub fn transpose(&mut self, a: usize) -> usize {
        let buf = self.get(a);
        let data = buf.to_f64();
        let shape = buf.shape.clone();
        assert_eq!(shape.len(), 2, "transpose requires 2D tensor");
        let (rows, cols) = (shape[0], shape[1]);
        let mut out = vec![0.0f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                out[j * rows + i] = data[i * cols + j];
            }
        }
        let id = self.alloc(&[cols, rows], DType::F64);
        self.write_data(id, &out);
        id
    }
}

// ---------------------------------------------------------------------------
// 5. Memory Pool / Arena Allocator
// ---------------------------------------------------------------------------

/// Statistics about pool usage.
#[derive(Debug, Clone)]
pub struct PoolStats {
    pub total_allocated: usize,
    pub peak_allocated: usize,
    pub free_blocks: usize,
    pub reuse_count: usize,
}

/// A simple memory pool that caches freed buffers by size for reuse.
pub struct MemoryPool {
    free_blocks: BTreeMap<usize, Vec<usize>>, // size -> list of buffer ids
    total_allocated: usize,
    peak_allocated: usize,
    reuse_count: usize,
}

impl MemoryPool {
    pub fn new() -> Self {
        Self {
            free_blocks: BTreeMap::new(),
            total_allocated: 0,
            peak_allocated: 0,
            reuse_count: 0,
        }
    }

    /// Allocate a buffer of `size` bytes, reusing a freed block if available.
    pub fn alloc(&mut self, size: usize, runtime: &mut CpuRuntime) -> usize {
        // Check for a reusable block of the same size
        if let Some(ids) = self.free_blocks.get_mut(&size) {
            if let Some(id) = ids.pop() {
                if ids.is_empty() {
                    self.free_blocks.remove(&size);
                }
                self.reuse_count += 1;
                return id;
            }
        }
        // No reusable block; allocate fresh
        let n_elements = size / 8; // assume F64 for simplicity
        let shape = if n_elements > 0 { vec![n_elements] } else { vec![1] };
        let id = runtime.alloc(&shape, DType::F64);
        self.total_allocated += size;
        if self.total_allocated > self.peak_allocated {
            self.peak_allocated = self.total_allocated;
        }
        id
    }

    /// Return a buffer to the pool for later reuse.
    pub fn free(&mut self, id: usize, size: usize) {
        self.free_blocks.entry(size).or_default().push(id);
    }

    /// Free everything (e.g., end of training step).
    pub fn reset(&mut self) {
        self.free_blocks.clear();
        self.total_allocated = 0;
    }

    /// Get pool statistics.
    pub fn stats(&self) -> PoolStats {
        let free_count: usize = self.free_blocks.values().map(|v| v.len()).sum();
        PoolStats {
            total_allocated: self.total_allocated,
            peak_allocated: self.peak_allocated,
            free_blocks: free_count,
            reuse_count: self.reuse_count,
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_free_copy_roundtrip() {
        let mut rt = CpuRuntime::new();
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let id = rt.alloc(&[4], DType::F64);
        rt.write_data(id, &data);
        let out = rt.copy_data(id);
        assert_eq!(data, out);
        rt.free(id);
    }

    #[test]
    fn test_matmul_cpu_reference() {
        let mut rt = CpuRuntime::new();
        // [2x3] @ [3x2] = [2x2]
        let a = rt.alloc(&[2, 3], DType::F64);
        rt.write_data(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = rt.alloc(&[3, 2], DType::F64);
        rt.write_data(b, &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = rt.matmul(a, b);
        let out = rt.copy_data(c);
        // Row 0: 1*7+2*9+3*11 = 58, 1*8+2*10+3*12 = 64
        // Row 1: 4*7+5*9+6*11 = 139, 4*8+5*10+6*12 = 154
        assert_eq!(out, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let mut rt = CpuRuntime::new();
        let id = rt.alloc(&[5], DType::F64);
        rt.write_data(id, &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let s = rt.softmax(id, 0);
        let out = rt.copy_data(s);
        let sum: f64 = out.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "softmax should sum to 1, got {}", sum);
        // All values should be positive
        assert!(out.iter().all(|&v| v > 0.0));
    }

    #[test]
    fn test_softmax_2d_rows_sum_to_one() {
        let mut rt = CpuRuntime::new();
        let id = rt.alloc(&[2, 3], DType::F64);
        rt.write_data(id, &[1.0, 2.0, 3.0, 10.0, 20.0, 30.0]);
        let s = rt.softmax(id, 1);
        let out = rt.copy_data(s);
        let row0_sum: f64 = out[0..3].iter().sum();
        let row1_sum: f64 = out[3..6].iter().sum();
        assert!((row0_sum - 1.0).abs() < 1e-10);
        assert!((row1_sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_memory_pool_reuse() {
        let mut rt = CpuRuntime::new();
        let mut pool = MemoryPool::new();

        let id1 = pool.alloc(64, &mut rt);
        pool.free(id1, 64);
        let id2 = pool.alloc(64, &mut rt);
        // Should reuse the same buffer
        assert_eq!(id1, id2);
        assert_eq!(pool.stats().reuse_count, 1);
    }

    #[test]
    fn test_memory_pool_different_sizes_no_reuse() {
        let mut rt = CpuRuntime::new();
        let mut pool = MemoryPool::new();

        let id1 = pool.alloc(64, &mut rt);
        pool.free(id1, 64);
        let id2 = pool.alloc(128, &mut rt);
        // Different size, should NOT reuse
        assert_ne!(id1, id2);
        assert_eq!(pool.stats().reuse_count, 0);
    }

    #[test]
    fn test_auto_launch_config_1d() {
        let cfg = LaunchConfig::auto_1d(1000, 256);
        assert_eq!(cfg.block, [256, 1, 1]);
        assert_eq!(cfg.grid[0], 4); // ceil(1000/256)
    }

    #[test]
    fn test_auto_launch_config_2d() {
        let cfg = LaunchConfig::auto_2d(100, 200);
        assert_eq!(cfg.block, [16, 16, 1]);
        assert_eq!(cfg.grid[0], 13); // ceil(200/16)
        assert_eq!(cfg.grid[1], 7);  // ceil(100/16)
    }

    #[test]
    fn test_auto_launch_config_matmul() {
        let cfg = LaunchConfig::auto_matmul(64, 128, 32);
        assert_eq!(cfg.block, [16, 16, 1]);
        assert_eq!(cfg.grid[0], 8);  // ceil(128/16)
        assert_eq!(cfg.grid[1], 4);  // ceil(64/16)
        assert!(cfg.shared_mem > 0);
    }

    #[test]
    fn test_dtype_sizes() {
        assert_eq!(DType::F64.size_bytes(), 8);
        assert_eq!(DType::F32.size_bytes(), 4);
        assert_eq!(DType::F16.size_bytes(), 2);
        assert_eq!(DType::BF16.size_bytes(), 2);
        assert_eq!(DType::I64.size_bytes(), 8);
        assert_eq!(DType::I32.size_bytes(), 4);
        assert_eq!(DType::I8.size_bytes(), 1);
        assert_eq!(DType::Bool.size_bytes(), 1);
        assert_eq!(DType::Int4x2.size_bytes(), 1);
        assert_eq!(DType::Int4x2.elements_per_unit(), 2);
        assert_eq!(DType::FP8E4M3.size_bytes(), 1);
        assert_eq!(DType::FP8E5M2.size_bytes(), 1);
    }

    #[test]
    fn test_gpu_buffer_from_f64_roundtrip() {
        let data = vec![3.14, 2.71, 1.41];
        let buf = GpuBuffer::from_f64(&data, &[3], GpuBackend::Cpu);
        let out = buf.to_f64();
        assert_eq!(data, out);
    }

    #[test]
    fn test_gpu_buffer_reshape() {
        let mut buf = GpuBuffer::alloc(&[2, 3], DType::F64, GpuBackend::Cpu);
        assert_eq!(buf.num_elements(), 6);
        buf.reshape(&[3, 2]);
        assert_eq!(buf.shape, vec![3, 2]);
        assert_eq!(buf.num_elements(), 6);
    }

    #[test]
    fn test_relu() {
        let mut rt = CpuRuntime::new();
        let id = rt.alloc(&[5], DType::F64);
        rt.write_data(id, &[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let r = rt.relu(id);
        let out = rt.copy_data(r);
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_elementwise_add() {
        let mut rt = CpuRuntime::new();
        let a = rt.alloc(&[3], DType::F64);
        rt.write_data(a, &[1.0, 2.0, 3.0]);
        let b = rt.alloc(&[3], DType::F64);
        rt.write_data(b, &[10.0, 20.0, 30.0]);
        let c = rt.elementwise_add(a, b);
        assert_eq!(rt.copy_data(c), vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn test_elementwise_mul() {
        let mut rt = CpuRuntime::new();
        let a = rt.alloc(&[3], DType::F64);
        rt.write_data(a, &[2.0, 3.0, 4.0]);
        let b = rt.alloc(&[3], DType::F64);
        rt.write_data(b, &[5.0, 6.0, 7.0]);
        let c = rt.elementwise_mul(a, b);
        assert_eq!(rt.copy_data(c), vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_transpose() {
        let mut rt = CpuRuntime::new();
        let a = rt.alloc(&[2, 3], DType::F64);
        rt.write_data(a, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = rt.transpose(a);
        let out = rt.copy_data(t);
        assert_eq!(out, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
        assert_eq!(rt.get(t).shape, vec![3, 2]);
    }

    #[test]
    fn test_cross_entropy() {
        let mut rt = CpuRuntime::new();
        let logits = rt.alloc(&[2, 3], DType::F64);
        rt.write_data(logits, &[2.0, 1.0, 0.1, 0.1, 1.0, 2.0]);
        let loss = rt.cross_entropy(logits, &[0, 2]);
        // Should be a positive finite number
        assert!(loss > 0.0 && loss.is_finite());
    }

    #[test]
    fn test_layer_norm() {
        let mut rt = CpuRuntime::new();
        let x = rt.alloc(&[4], DType::F64);
        rt.write_data(x, &[1.0, 2.0, 3.0, 4.0]);
        let g = rt.alloc(&[4], DType::F64);
        rt.write_data(g, &[1.0, 1.0, 1.0, 1.0]);
        let b = rt.alloc(&[4], DType::F64);
        rt.write_data(b, &[0.0, 0.0, 0.0, 0.0]);
        let out_id = rt.layer_norm(x, g, b, 1e-5);
        let out = rt.copy_data(out_id);
        // With gamma=1, beta=0, output should be zero-mean
        let mean: f64 = out.iter().sum::<f64>() / out.len() as f64;
        assert!(mean.abs() < 1e-10, "layer_norm output should be zero-mean");
    }

    #[test]
    fn test_cpu_device_trait() {
        let dev = CpuDevice::new();
        assert_eq!(dev.name(), "Vortex CPU Fallback");
        assert!(dev.memory_total() > 0);
        assert_eq!(dev.compute_capability(), (0, 0));
        assert_eq!(dev.max_threads_per_block(), 1024);
        assert!(dev.max_shared_memory() > 0);
        assert_eq!(dev.warp_size(), 1);
    }

    #[test]
    fn test_memory_pool_stats() {
        let mut rt = CpuRuntime::new();
        let mut pool = MemoryPool::new();
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 0);
        assert_eq!(stats.peak_allocated, 0);

        let _id = pool.alloc(1024, &mut rt);
        let stats = pool.stats();
        assert_eq!(stats.total_allocated, 1024);
        assert_eq!(stats.peak_allocated, 1024);
    }

    #[test]
    fn test_memory_pool_reset() {
        let mut rt = CpuRuntime::new();
        let mut pool = MemoryPool::new();
        let id = pool.alloc(64, &mut rt);
        pool.free(id, 64);
        assert_eq!(pool.stats().free_blocks, 1);
        pool.reset();
        assert_eq!(pool.stats().free_blocks, 0);
        assert_eq!(pool.stats().total_allocated, 0);
    }
}
