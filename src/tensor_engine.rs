// Fast native tensor backend for Vortex
// Pure Rust — no external deps, no CUDA
// Tiled matmul, manual SIMD-style unrolling, memory pools, multi-threading

use std::sync::{Arc, Mutex, atomic::{AtomicUsize, Ordering}};
use std::thread;

// ─── Dtype ───────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DType {
    F16,
    BF16,
    F32,
    F64,
    I32,
    I64,
    Bool,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F16 | DType::BF16 => 2,
            DType::F32 | DType::I32 => 4,
            DType::F64 | DType::I64 => 8,
            DType::Bool => 1,
        }
    }
}

// ─── Layout ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Layout {
    RowMajor,
    ColMajor,
}

// ─── f16 / bf16 bit manipulation ─────────────────────────────────────────────

pub fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf/NaN
        return (sign | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 {
        return (sign | 0x7C00) as u16; // overflow -> Inf
    }
    if new_exp <= 0 {
        if new_exp < -10 {
            return sign as u16; // underflow -> 0
        }
        let frac = (frac | 0x800000) >> (1 - new_exp);
        return (sign | (frac >> 13)) as u16;
    }
    (sign | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let frac = (h & 0x3FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        // subnormal
        let mut e = 0i32;
        let mut f = frac;
        while f & 0x400 == 0 {
            f <<= 1;
            e += 1;
        }
        let exp32 = (127 - 15 - e) as u32;
        let frac32 = (f & 0x3FF) << 13;
        return f32::from_bits(sign | (exp32 << 23) | frac32);
    }
    if exp == 31 {
        let bits = sign | 0x7F800000 | (frac << 13);
        return f32::from_bits(bits);
    }
    let exp32 = exp + 127 - 15;
    f32::from_bits(sign | (exp32 << 23) | (frac << 13))
}

pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round to nearest even
    let rounding_bias = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

// ─── Quantization ────────────────────────────────────────────────────────────

pub struct QuantParams {
    pub scale: f32,
    pub zero_point: i32,
}

pub fn quantize_f32_to_i8(data: &[f32]) -> (Vec<i8>, QuantParams) {
    if data.is_empty() {
        return (vec![], QuantParams { scale: 1.0, zero_point: 0 });
    }
    let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let scale = (max - min) / 255.0;
    let scale = if scale == 0.0 { 1.0 } else { scale };
    let zero_point = (-128.0 - min / scale).round() as i32;
    let quantized: Vec<i8> = data
        .iter()
        .map(|&x| {
            let q = (x / scale).round() as i32 + zero_point;
            q.clamp(-128, 127) as i8
        })
        .collect();
    (quantized, QuantParams { scale, zero_point })
}

pub fn dequantize_i8_to_f32(data: &[i8], params: &QuantParams) -> Vec<f32> {
    data.iter()
        .map(|&q| (q as i32 - params.zero_point) as f32 * params.scale)
        .collect()
}

// ─── FastTensor ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct FastTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub dtype: DType,
    pub layout: Layout,
}

fn compute_strides(shape: &[usize], layout: Layout) -> Vec<usize> {
    let ndim = shape.len();
    if ndim == 0 {
        return vec![];
    }
    let mut strides = vec![0usize; ndim];
    match layout {
        Layout::RowMajor => {
            strides[ndim - 1] = 1;
            for i in (0..ndim - 1).rev() {
                strides[i] = strides[i + 1] * shape[i + 1];
            }
        }
        Layout::ColMajor => {
            strides[0] = 1;
            for i in 1..ndim {
                strides[i] = strides[i - 1] * shape[i - 1];
            }
        }
    }
    strides
}

impl FastTensor {
    pub fn zeros(shape: Vec<usize>, dtype: DType) -> Self {
        let numel: usize = shape.iter().product();
        let strides = compute_strides(&shape, Layout::RowMajor);
        FastTensor {
            data: vec![0u8; numel * dtype.size_bytes()],
            shape,
            strides,
            dtype,
            layout: Layout::RowMajor,
        }
    }

    pub fn from_f64(shape: Vec<usize>, vals: &[f64]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(vals.len(), numel);
        let strides = compute_strides(&shape, Layout::RowMajor);
        let mut data = vec![0u8; numel * 8];
        for (i, &v) in vals.iter().enumerate() {
            let bytes = v.to_le_bytes();
            data[i * 8..i * 8 + 8].copy_from_slice(&bytes);
        }
        FastTensor {
            data,
            shape,
            strides,
            dtype: DType::F64,
            layout: Layout::RowMajor,
        }
    }

    pub fn from_f32(shape: Vec<usize>, vals: &[f32]) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(vals.len(), numel);
        let strides = compute_strides(&shape, Layout::RowMajor);
        let mut data = vec![0u8; numel * 4];
        for (i, &v) in vals.iter().enumerate() {
            let bytes = v.to_le_bytes();
            data[i * 4..i * 4 + 4].copy_from_slice(&bytes);
        }
        FastTensor {
            data,
            shape,
            strides,
            dtype: DType::F32,
            layout: Layout::RowMajor,
        }
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn as_f64_slice(&self) -> &[f64] {
        assert_eq!(self.dtype, DType::F64);
        let ptr = self.data.as_ptr() as *const f64;
        unsafe { std::slice::from_raw_parts(ptr, self.numel()) }
    }

    pub fn as_f64_slice_mut(&mut self) -> &mut [f64] {
        assert_eq!(self.dtype, DType::F64);
        let ptr = self.data.as_mut_ptr() as *mut f64;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.numel()) }
    }

    pub fn as_f32_slice(&self) -> &[f32] {
        assert_eq!(self.dtype, DType::F32);
        let ptr = self.data.as_ptr() as *const f32;
        unsafe { std::slice::from_raw_parts(ptr, self.numel()) }
    }

    pub fn as_f32_slice_mut(&mut self) -> &mut [f32] {
        assert_eq!(self.dtype, DType::F32);
        let ptr = self.data.as_mut_ptr() as *mut f32;
        unsafe { std::slice::from_raw_parts_mut(ptr, self.numel()) }
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self.dtype {
            DType::F64 => self.as_f64_slice().to_vec(),
            DType::F32 => self.as_f32_slice().iter().map(|&x| x as f64).collect(),
            _ => panic!("to_f64_vec: unsupported dtype {:?}", self.dtype),
        }
    }

    /// Cast to a different dtype
    pub fn cast(&self, target: DType) -> FastTensor {
        if self.dtype == target {
            return self.clone();
        }
        let src = self.to_f64_vec();
        match target {
            DType::F64 => FastTensor::from_f64(self.shape.clone(), &src),
            DType::F32 => {
                let f: Vec<f32> = src.iter().map(|&x| x as f32).collect();
                FastTensor::from_f32(self.shape.clone(), &f)
            }
            DType::F16 => {
                let numel = self.numel();
                let mut data = vec![0u8; numel * 2];
                for (i, &v) in src.iter().enumerate() {
                    let h = f32_to_f16(v as f32);
                    data[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                }
                FastTensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    dtype: DType::F16,
                    layout: self.layout,
                }
            }
            DType::BF16 => {
                let numel = self.numel();
                let mut data = vec![0u8; numel * 2];
                for (i, &v) in src.iter().enumerate() {
                    let h = f32_to_bf16(v as f32);
                    data[i * 2..i * 2 + 2].copy_from_slice(&h.to_le_bytes());
                }
                FastTensor {
                    data,
                    shape: self.shape.clone(),
                    strides: self.strides.clone(),
                    dtype: DType::BF16,
                    layout: self.layout,
                }
            }
            _ => panic!("cast: unsupported target {:?}", target),
        }
    }
}

// ─── Memory Pool ─────────────────────────────────────────────────────────────

struct PoolEntry {
    buf: Vec<u8>,
    in_use: bool,
}

pub struct MemoryPool {
    pools: Mutex<Vec<PoolEntry>>,
    total_allocated: AtomicUsize,
    total_reused: AtomicUsize,
}

impl MemoryPool {
    pub fn new() -> Self {
        MemoryPool {
            pools: Mutex::new(Vec::new()),
            total_allocated: AtomicUsize::new(0),
            total_reused: AtomicUsize::new(0),
        }
    }

    /// Acquire a buffer of at least `size` bytes (zeroed)
    pub fn acquire(&self, size: usize) -> Vec<u8> {
        let mut pools = self.pools.lock().unwrap();
        // Find a free buffer that's big enough
        for entry in pools.iter_mut() {
            if !entry.in_use && entry.buf.len() >= size {
                entry.in_use = true;
                self.total_reused.fetch_add(1, Ordering::Relaxed);
                let mut buf = entry.buf.clone();
                for b in buf.iter_mut().take(size) {
                    *b = 0;
                }
                return buf;
            }
        }
        // Allocate new
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        let buf = vec![0u8; size];
        pools.push(PoolEntry {
            buf: buf.clone(),
            in_use: true,
        });
        buf
    }

    /// Release a buffer back to the pool
    pub fn release(&self, buf: Vec<u8>) {
        let mut pools = self.pools.lock().unwrap();
        for entry in pools.iter_mut() {
            if entry.in_use && entry.buf.len() == buf.len() {
                entry.in_use = false;
                entry.buf = buf;
                return;
            }
        }
        // Not tracked, just add it
        pools.push(PoolEntry {
            buf,
            in_use: false,
        });
    }

    pub fn stats(&self) -> (usize, usize) {
        (
            self.total_allocated.load(Ordering::Relaxed),
            self.total_reused.load(Ordering::Relaxed),
        )
    }

    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        pools.clear();
    }
}

// Global memory pool
fn global_pool() -> &'static MemoryPool {
    use std::sync::OnceLock;
    static POOL: OnceLock<MemoryPool> = OnceLock::new();
    POOL.get_or_init(MemoryPool::new)
}

// ─── Thread pool ─────────────────────────────────────────────────────────────

static THREAD_COUNT: AtomicUsize = AtomicUsize::new(0);

pub fn set_thread_count(n: usize) {
    THREAD_COUNT.store(n, Ordering::Relaxed);
}

fn get_thread_count() -> usize {
    let n = THREAD_COUNT.load(Ordering::Relaxed);
    if n == 0 {
        // auto-detect
        thread::available_parallelism()
            .map(|p| p.get())
            .unwrap_or(4)
    } else {
        n
    }
}

// ─── Tiled Matmul (f64) ─────────────────────────────────────────────────────

const TILE: usize = 32;

/// Tiled cache-friendly matmul: C[m,n] = A[m,k] * B[k,n]
/// a, b, c are row-major flat arrays.
fn tiled_matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    // Zero output
    for v in c.iter_mut() {
        *v = 0.0;
    }

    // Transpose B for cache-friendly access on inner dimension
    let mut bt = vec![0.0f64; k * n];
    for i in 0..k {
        for j in 0..n {
            bt[j * k + i] = b[i * n + j];
        }
    }

    // Tiled multiplication
    let ti = TILE;
    let tj = TILE;
    let tk = TILE;

    for ii in (0..m).step_by(ti) {
        let i_end = (ii + ti).min(m);
        for jj in (0..n).step_by(tj) {
            let j_end = (jj + tj).min(n);
            for kk in (0..k).step_by(tk) {
                let k_end = (kk + tk).min(k);
                // Micro-kernel: process tile
                for i in ii..i_end {
                    let a_row = i * k;
                    let c_row = i * n;
                    for j in jj..j_end {
                        let bt_row = j * k;
                        let mut sum = 0.0f64;
                        // Manual 4x unroll
                        let mut p = kk;
                        let p_end4 = kk + ((k_end - kk) / 4) * 4;
                        while p < p_end4 {
                            sum += a[a_row + p] * bt[bt_row + p]
                                + a[a_row + p + 1] * bt[bt_row + p + 1]
                                + a[a_row + p + 2] * bt[bt_row + p + 2]
                                + a[a_row + p + 3] * bt[bt_row + p + 3];
                            p += 4;
                        }
                        while p < k_end {
                            sum += a[a_row + p] * bt[bt_row + p];
                            p += 1;
                        }
                        c[c_row + j] += sum;
                    }
                }
            }
        }
    }
}

/// Multi-threaded tiled matmul
fn mt_matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let nthreads = get_thread_count().min(m);
    if nthreads <= 1 || m < 64 {
        let mut c = vec![0.0f64; m * n];
        tiled_matmul_f64(a, b, &mut c, m, k, n);
        return c;
    }

    let a_arc = Arc::new(a.to_vec());
    let b_arc = Arc::new(b.to_vec());
    let rows_per = (m + nthreads - 1) / nthreads;

    let handles: Vec<_> = (0..nthreads)
        .map(|t| {
            let a_ref = Arc::clone(&a_arc);
            let b_ref = Arc::clone(&b_arc);
            let start = t * rows_per;
            let end = ((t + 1) * rows_per).min(m);
            let k_ = k;
            let n_ = n;
            thread::spawn(move || {
                if start >= end {
                    return (start, vec![]);
                }
                let rows = end - start;
                let a_sub = &a_ref[start * k_..(end * k_)];
                let mut c_sub = vec![0.0f64; rows * n_];
                tiled_matmul_f64(a_sub, &b_ref, &mut c_sub, rows, k_, n_);
                (start, c_sub)
            })
        })
        .collect();

    let mut c = vec![0.0f64; m * n];
    for h in handles {
        let (start, data) = h.join().unwrap();
        if !data.is_empty() {
            let offset = start * n;
            c[offset..offset + data.len()].copy_from_slice(&data);
        }
    }
    c
}

// ─── Public matmul API ───────────────────────────────────────────────────────

/// Fast matmul for FastTensor (2D only)
pub fn fast_matmul(a: &FastTensor, b: &FastTensor) -> FastTensor {
    assert_eq!(a.shape.len(), 2);
    assert_eq!(b.shape.len(), 2);
    assert_eq!(a.shape[1], b.shape[0], "matmul dim mismatch");
    let m = a.shape[0];
    let k = a.shape[1];
    let n = b.shape[1];

    let a_f64 = a.to_f64_vec();
    let b_f64 = b.to_f64_vec();
    let c_data = mt_matmul_f64(&a_f64, &b_f64, m, k, n);
    FastTensor::from_f64(vec![m, n], &c_data)
}

/// Integration with nn.rs — drop-in replacement
pub fn fast_tensor_matmul(
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    assert_eq!(a_shape.len(), 2);
    assert_eq!(b_shape.len(), 2);
    let m = a_shape[0];
    let k = a_shape[1];
    let n = b_shape[1];
    assert_eq!(k, b_shape[0]);
    let c = mt_matmul_f64(a_data, b_data, m, k, n);
    (c, vec![m, n])
}

/// Batched matmul: [B,M,K] x [K,N] -> [B,M,N]
pub fn fast_tensor_batched_matmul(
    a_data: &[f64],
    a_shape: &[usize],
    b_data: &[f64],
    b_shape: &[usize],
) -> (Vec<f64>, Vec<usize>) {
    assert_eq!(a_shape.len(), 3);
    assert_eq!(b_shape.len(), 2);
    let batch = a_shape[0];
    let m = a_shape[1];
    let k = a_shape[2];
    let n = b_shape[1];
    assert_eq!(k, b_shape[0]);

    let mut out = vec![0.0f64; batch * m * n];
    for bi in 0..batch {
        let a_slice = &a_data[bi * m * k..(bi + 1) * m * k];
        let mut c_slice = vec![0.0f64; m * n];
        tiled_matmul_f64(a_slice, b_data, &mut c_slice, m, k, n);
        out[bi * m * n..(bi + 1) * m * n].copy_from_slice(&c_slice);
    }
    (out, vec![batch, m, n])
}

// ─── Elementwise Operations ──────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum ElemOp {
    Add,
    Sub,
    Mul,
    Div,
}

/// Vectorized elementwise on f64 slices (same shape)
fn elementwise_f64(op: ElemOp, a: &[f64], b: &[f64], out: &mut [f64]) {
    let n = a.len();
    debug_assert_eq!(n, b.len());
    debug_assert_eq!(n, out.len());

    // Process 4 at a time
    let n4 = (n / 4) * 4;
    match op {
        ElemOp::Add => {
            let mut i = 0;
            while i < n4 {
                out[i] = a[i] + b[i];
                out[i + 1] = a[i + 1] + b[i + 1];
                out[i + 2] = a[i + 2] + b[i + 2];
                out[i + 3] = a[i + 3] + b[i + 3];
                i += 4;
            }
            for j in n4..n {
                out[j] = a[j] + b[j];
            }
        }
        ElemOp::Sub => {
            let mut i = 0;
            while i < n4 {
                out[i] = a[i] - b[i];
                out[i + 1] = a[i + 1] - b[i + 1];
                out[i + 2] = a[i + 2] - b[i + 2];
                out[i + 3] = a[i + 3] - b[i + 3];
                i += 4;
            }
            for j in n4..n {
                out[j] = a[j] - b[j];
            }
        }
        ElemOp::Mul => {
            let mut i = 0;
            while i < n4 {
                out[i] = a[i] * b[i];
                out[i + 1] = a[i + 1] * b[i + 1];
                out[i + 2] = a[i + 2] * b[i + 2];
                out[i + 3] = a[i + 3] * b[i + 3];
                i += 4;
            }
            for j in n4..n {
                out[j] = a[j] * b[j];
            }
        }
        ElemOp::Div => {
            let mut i = 0;
            while i < n4 {
                out[i] = a[i] / b[i];
                out[i + 1] = a[i + 1] / b[i + 1];
                out[i + 2] = a[i + 2] / b[i + 2];
                out[i + 3] = a[i + 3] / b[i + 3];
                i += 4;
            }
            for j in n4..n {
                out[j] = a[j] / b[j];
            }
        }
    }
}

/// Multi-threaded elementwise for large tensors
fn mt_elementwise_f64(op: ElemOp, a: &[f64], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    if n < 10000 {
        let mut out = vec![0.0f64; n];
        elementwise_f64(op, a, b, &mut out);
        return out;
    }

    let nthreads = get_thread_count();
    let chunk = (n + nthreads - 1) / nthreads;
    let a_arc = Arc::new(a.to_vec());
    let b_arc = Arc::new(b.to_vec());

    let handles: Vec<_> = (0..nthreads)
        .map(|t| {
            let a_ref = Arc::clone(&a_arc);
            let b_ref = Arc::clone(&b_arc);
            let start = t * chunk;
            let end = ((t + 1) * chunk).min(n);
            thread::spawn(move || {
                if start >= end {
                    return (start, vec![]);
                }
                let len = end - start;
                let mut out = vec![0.0f64; len];
                elementwise_f64(op, &a_ref[start..end], &b_ref[start..end], &mut out);
                (start, out)
            })
        })
        .collect();

    let mut result = vec![0.0f64; n];
    for h in handles {
        let (start, data) = h.join().unwrap();
        if !data.is_empty() {
            result[start..start + data.len()].copy_from_slice(&data);
        }
    }
    result
}

/// Elementwise with broadcasting support
pub fn fast_elementwise(op: ElemOp, a: &FastTensor, b: &FastTensor) -> FastTensor {
    let a_data = a.to_f64_vec();
    let b_data = b.to_f64_vec();

    if a.shape == b.shape {
        let out = mt_elementwise_f64(op, &a_data, &b_data);
        return FastTensor::from_f64(a.shape.clone(), &out);
    }

    // Broadcasting: b is 1D, last dim matches
    if b.shape.len() == 1 && a.shape.last() == Some(&b.shape[0]) {
        let cols = b.shape[0];
        let n = a_data.len();
        let mut out = vec![0.0f64; n];
        for i in 0..n {
            let bv = b_data[i % cols];
            out[i] = match op {
                ElemOp::Add => a_data[i] + bv,
                ElemOp::Sub => a_data[i] - bv,
                ElemOp::Mul => a_data[i] * bv,
                ElemOp::Div => a_data[i] / bv,
            };
        }
        return FastTensor::from_f64(a.shape.clone(), &out);
    }

    // Scalar broadcast: b is single element
    if b.numel() == 1 {
        let bv = b_data[0];
        let out: Vec<f64> = a_data
            .iter()
            .map(|&av| match op {
                ElemOp::Add => av + bv,
                ElemOp::Sub => av - bv,
                ElemOp::Mul => av * bv,
                ElemOp::Div => av / bv,
            })
            .collect();
        return FastTensor::from_f64(a.shape.clone(), &out);
    }

    panic!(
        "Cannot broadcast shapes {:?} and {:?}",
        a.shape, b.shape
    );
}

// ─── Reduce Operations ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
pub enum ReduceOp {
    Sum,
    Max,
    Min,
    Mean,
}

/// Reduce along a given axis
pub fn fast_reduce(op: ReduceOp, t: &FastTensor, axis: usize) -> FastTensor {
    assert!(axis < t.shape.len(), "axis out of bounds");
    let data = t.to_f64_vec();
    let ndim = t.shape.len();
    let axis_size = t.shape[axis];

    // Compute output shape
    let mut out_shape: Vec<usize> = t.shape.clone();
    out_shape.remove(axis);
    if out_shape.is_empty() {
        out_shape.push(1);
    }
    let out_numel: usize = out_shape.iter().product();

    // Compute stride for the axis
    let stride: usize = t.shape[axis + 1..].iter().product();
    let outer: usize = t.shape[..axis].iter().product();
    let inner: usize = if axis + 1 < ndim {
        t.shape[axis + 1..].iter().product()
    } else {
        1
    };

    let mut out = vec![0.0f64; out_numel];

    for o in 0..outer {
        for i in 0..inner {
            let init = match op {
                ReduceOp::Sum | ReduceOp::Mean => 0.0,
                ReduceOp::Max => f64::NEG_INFINITY,
                ReduceOp::Min => f64::INFINITY,
            };
            let mut acc = init;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                let v = data[idx];
                acc = match op {
                    ReduceOp::Sum | ReduceOp::Mean => acc + v,
                    ReduceOp::Max => acc.max(v),
                    ReduceOp::Min => acc.min(v),
                };
            }
            if matches!(op, ReduceOp::Mean) {
                acc /= axis_size as f64;
            }
            out[o * inner + i] = acc;
        }
    }

    FastTensor::from_f64(out_shape, &out)
}

// ─── Transpose ───────────────────────────────────────────────────────────────

/// Cache-oblivious transpose for 2D tensors
pub fn fast_transpose(t: &FastTensor) -> FastTensor {
    assert_eq!(t.shape.len(), 2);
    let rows = t.shape[0];
    let cols = t.shape[1];
    let data = t.to_f64_vec();
    let mut out = vec![0.0f64; rows * cols];

    // Tiled transpose for cache friendliness
    const BLK: usize = 32;
    for ii in (0..rows).step_by(BLK) {
        let i_end = (ii + BLK).min(rows);
        for jj in (0..cols).step_by(BLK) {
            let j_end = (jj + BLK).min(cols);
            for i in ii..i_end {
                for j in jj..j_end {
                    out[j * rows + i] = data[i * cols + j];
                }
            }
        }
    }
    FastTensor::from_f64(vec![cols, rows], &out)
}

// ─── Softmax ─────────────────────────────────────────────────────────────────

/// Numerically stable softmax along given axis
pub fn fast_softmax(t: &FastTensor, axis: usize) -> FastTensor {
    assert!(axis < t.shape.len());
    let data = t.to_f64_vec();
    let mut out = data.clone();

    let axis_size = t.shape[axis];
    let inner: usize = t.shape[axis + 1..].iter().product::<usize>().max(1);
    let outer: usize = t.shape[..axis].iter().product::<usize>().max(1);

    for o in 0..outer {
        for i in 0..inner {
            // Find max
            let mut mx = f64::NEG_INFINITY;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                mx = mx.max(data[idx]);
            }
            // Exp and sum
            let mut sum = 0.0f64;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                let e = (data[idx] - mx).exp();
                out[idx] = e;
                sum += e;
            }
            // Normalize
            let inv = 1.0 / sum;
            for a in 0..axis_size {
                let idx = o * axis_size * inner + a * inner + i;
                out[idx] *= inv;
            }
        }
    }

    FastTensor::from_f64(t.shape.clone(), &out)
}

// ─── Layer Norm ──────────────────────────────────────────────────────────────

/// Fused layer normalization: (x - mean) / sqrt(var + eps) * gamma + beta
pub fn fast_layer_norm(
    t: &FastTensor,
    gamma: &FastTensor,
    beta: &FastTensor,
    eps: f64,
) -> FastTensor {
    assert_eq!(t.shape.len(), 2);
    let rows = t.shape[0];
    let cols = t.shape[1];
    assert_eq!(gamma.numel(), cols);
    assert_eq!(beta.numel(), cols);

    let data = t.to_f64_vec();
    let g = gamma.to_f64_vec();
    let b = beta.to_f64_vec();
    let mut out = vec![0.0f64; rows * cols];

    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        // Fused: compute mean and variance in one pass
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for &v in row {
            sum += v;
            sum_sq += v * v;
        }
        let mean = sum / cols as f64;
        let var = sum_sq / cols as f64 - mean * mean;
        let inv_std = 1.0 / (var + eps).sqrt();

        let out_row = &mut out[r * cols..(r + 1) * cols];
        for j in 0..cols {
            out_row[j] = (row[j] - mean) * inv_std * g[j] + b[j];
        }
    }

    FastTensor::from_f64(t.shape.clone(), &out)
}

// ─── Conv2d (im2col + matmul) ────────────────────────────────────────────────

/// 2D convolution via im2col + matmul
/// input: [N, C_in, H, W]
/// kernel: [C_out, C_in, KH, KW]
/// Returns: [N, C_out, H_out, W_out]
pub fn fast_conv2d(
    input: &FastTensor,
    kernel: &FastTensor,
    stride: usize,
    padding: usize,
) -> FastTensor {
    assert_eq!(input.shape.len(), 4);
    assert_eq!(kernel.shape.len(), 4);

    let batch = input.shape[0];
    let c_in = input.shape[1];
    let h = input.shape[2];
    let w = input.shape[3];
    let c_out = kernel.shape[0];
    assert_eq!(kernel.shape[1], c_in);
    let kh = kernel.shape[2];
    let kw = kernel.shape[3];

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;

    let in_data = input.to_f64_vec();
    let k_data = kernel.to_f64_vec();

    // Reshape kernel to [C_out, C_in*KH*KW]
    let col_len = c_in * kh * kw;

    let mut output = vec![0.0f64; batch * c_out * h_out * w_out];

    for n in 0..batch {
        // im2col: build column matrix [col_len, h_out*w_out]
        let patches = h_out * w_out;
        let mut col = vec![0.0f64; col_len * patches];

        for c in 0..c_in {
            for ky in 0..kh {
                for kx in 0..kw {
                    let col_row = c * kh * kw + ky * kw + kx;
                    for oy in 0..h_out {
                        for ox in 0..w_out {
                            let iy = oy * stride + ky;
                            let ix = ox * stride + kx;
                            let iy = iy as isize - padding as isize;
                            let ix = ix as isize - padding as isize;
                            let val = if iy >= 0
                                && iy < h as isize
                                && ix >= 0
                                && ix < w as isize
                            {
                                in_data[n * c_in * h * w
                                    + c * h * w
                                    + iy as usize * w
                                    + ix as usize]
                            } else {
                                0.0
                            };
                            col[col_row * patches + oy * w_out + ox] = val;
                        }
                    }
                }
            }
        }

        // matmul: kernel[C_out, col_len] x col[col_len, patches] -> [C_out, patches]
        let mut out_slice = vec![0.0f64; c_out * patches];
        tiled_matmul_f64(&k_data, &col, &mut out_slice, c_out, col_len, patches);

        output[n * c_out * h_out * w_out..(n + 1) * c_out * h_out * w_out]
            .copy_from_slice(&out_slice);
    }

    FastTensor::from_f64(vec![batch, c_out, h_out, w_out], &output)
}

// ─── Benchmarks ──────────────────────────────────────────────────────────────

/// Naive matmul for benchmarking comparison
fn naive_matmul_f64(a: &[f64], b: &[f64], c: &mut [f64], m: usize, k: usize, n: usize) {
    for v in c.iter_mut() {
        *v = 0.0;
    }
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0f64;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
}

pub fn bench_matmul(size: usize) -> String {
    let n = size;
    let mut a = vec![0.0f64; n * n];
    let mut b = vec![0.0f64; n * n];
    // Fill with pseudo-random data
    let mut seed: u64 = 12345;
    for v in a.iter_mut().chain(b.iter_mut()) {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
    }

    // Naive
    let t0 = std::time::Instant::now();
    let mut c_naive = vec![0.0f64; n * n];
    naive_matmul_f64(&a, &b, &mut c_naive, n, n, n);
    let naive_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Tiled
    let t1 = std::time::Instant::now();
    let mut c_tiled = vec![0.0f64; n * n];
    tiled_matmul_f64(&a, &b, &mut c_tiled, n, n, n);
    let tiled_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // MT tiled
    let t2 = std::time::Instant::now();
    let _c_mt = mt_matmul_f64(&a, &b, n, n, n);
    let mt_ms = t2.elapsed().as_secs_f64() * 1000.0;

    format!(
        "Matmul {}x{}: naive={:.2}ms tiled={:.2}ms mt={:.2}ms | speedup: tiled={:.1}x mt={:.1}x",
        n,
        n,
        naive_ms,
        tiled_ms,
        mt_ms,
        naive_ms / tiled_ms.max(0.001),
        naive_ms / mt_ms.max(0.001)
    )
}

pub fn bench_elementwise(size: usize) -> String {
    let n = size;
    let mut a = vec![0.0f64; n];
    let mut b = vec![0.0f64; n];
    let mut seed: u64 = 54321;
    for v in a.iter_mut().chain(b.iter_mut()) {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = (seed >> 33) as f64 / (1u64 << 31) as f64;
    }

    // Naive
    let t0 = std::time::Instant::now();
    let _naive: Vec<f64> = a.iter().zip(&b).map(|(x, y)| x + y).collect();
    let naive_ms = t0.elapsed().as_secs_f64() * 1000.0;

    // Fast
    let t1 = std::time::Instant::now();
    let mut out = vec![0.0f64; n];
    elementwise_f64(ElemOp::Add, &a, &b, &mut out);
    let fast_ms = t1.elapsed().as_secs_f64() * 1000.0;

    // MT
    let t2 = std::time::Instant::now();
    let _mt = mt_elementwise_f64(ElemOp::Add, &a, &b);
    let mt_ms = t2.elapsed().as_secs_f64() * 1000.0;

    format!(
        "Elementwise add {} elems: naive={:.2}ms fast={:.2}ms mt={:.2}ms | speedup: fast={:.1}x mt={:.1}x",
        n,
        naive_ms,
        fast_ms,
        mt_ms,
        naive_ms / fast_ms.max(0.001),
        naive_ms / mt_ms.max(0.001)
    )
}

// ─── Interpreter builtins ────────────────────────────────────────────────────

/// Called from interpreter: tensor_fast_matmul(a_flat, a_shape, b_flat, b_shape)
pub fn builtin_fast_matmul(
    a_data: Vec<f64>,
    a_shape: Vec<usize>,
    b_data: Vec<f64>,
    b_shape: Vec<usize>,
) -> (Vec<f64>, Vec<usize>) {
    if a_shape.len() == 3 && b_shape.len() == 2 {
        fast_tensor_batched_matmul(&a_data, &a_shape, &b_data, &b_shape)
    } else {
        fast_tensor_matmul(&a_data, &a_shape, &b_data, &b_shape)
    }
}

/// Called from interpreter: tensor_benchmark(op, size)
pub fn builtin_benchmark(op: &str, size: usize) -> String {
    match op {
        "matmul" => bench_matmul(size),
        "elementwise" => bench_elementwise(size),
        _ => format!("Unknown benchmark op: {}", op),
    }
}

/// Called from interpreter: tensor_set_threads(n)
pub fn builtin_set_threads(n: usize) {
    set_thread_count(n);
}

// ─── USE_FAST_BACKEND flag ───────────────────────────────────────────────────

use std::sync::atomic::AtomicBool;

static USE_FAST_BACKEND: AtomicBool = AtomicBool::new(true);

pub fn set_use_fast_backend(enabled: bool) {
    USE_FAST_BACKEND.store(enabled, Ordering::Relaxed);
}

pub fn use_fast_backend() -> bool {
    USE_FAST_BACKEND.load(Ordering::Relaxed)
}

// ─── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| (x - y).abs() < tol)
    }

    #[test]
    fn test_tiled_matmul_small() {
        // 2x3 * 3x2 = 2x2
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let mut c = vec![0.0; 4];
        tiled_matmul_f64(&a, &b, &mut c, 2, 3, 2);
        // [1*7+2*9+3*11, 1*8+2*10+3*12] = [58, 64]
        // [4*7+5*9+6*11, 4*8+5*10+6*12] = [139, 154]
        assert!(approx_eq(&c, &[58.0, 64.0, 139.0, 154.0], 1e-10));
    }

    #[test]
    fn test_tiled_matmul_1x1() {
        let a = vec![3.0];
        let b = vec![5.0];
        let mut c = vec![0.0];
        tiled_matmul_f64(&a, &b, &mut c, 1, 1, 1);
        assert!((c[0] - 15.0).abs() < 1e-10);
    }

    #[test]
    fn test_tiled_matmul_1xn() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0; 1];
        // 1x3 * 3x1 = 1x1 => 1*4+2*5+3*6 = 32
        tiled_matmul_f64(&a, &b, &mut c, 1, 3, 1);
        assert!((c[0] - 32.0).abs() < 1e-10);
    }

    #[test]
    fn test_tiled_matmul_nx1() {
        // 3x1 * 1x3 = 3x3
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut c = vec![0.0; 9];
        tiled_matmul_f64(&a, &b, &mut c, 3, 1, 3);
        let expected = vec![4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
        assert!(approx_eq(&c, &expected, 1e-10));
    }

    #[test]
    fn test_tiled_matmul_nonsquare() {
        // 2x4 * 4x3 = 2x3
        let a: Vec<f64> = (1..=8).map(|x| x as f64).collect();
        let b: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let mut c = vec![0.0; 6];
        tiled_matmul_f64(&a, &b, &mut c, 2, 4, 3);
        // Row 0: 1*1+2*4+3*7+4*10 = 1+8+21+40=70, 1*2+2*5+3*8+4*11=2+10+24+44=80, ...=90
        assert!((c[0] - 70.0).abs() < 1e-10);
        assert!((c[1] - 80.0).abs() < 1e-10);
        assert!((c[2] - 90.0).abs() < 1e-10);
    }

    #[test]
    fn test_fast_matmul_matches_naive() {
        let n = 64;
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        let mut seed: u64 = 999;
        for v in a.iter_mut().chain(b.iter_mut()) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        }
        let mut c_naive = vec![0.0f64; n * n];
        naive_matmul_f64(&a, &b, &mut c_naive, n, n, n);
        let mut c_tiled = vec![0.0f64; n * n];
        tiled_matmul_f64(&a, &b, &mut c_tiled, n, n, n);
        assert!(approx_eq(&c_naive, &c_tiled, 1e-8));
    }

    #[test]
    fn test_mt_matmul_matches() {
        let n = 100;
        let mut a = vec![0.0f64; n * n];
        let mut b = vec![0.0f64; n * n];
        let mut seed: u64 = 42;
        for v in a.iter_mut().chain(b.iter_mut()) {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *v = (seed >> 33) as f64 / (1u64 << 31) as f64 - 0.5;
        }
        let mut c_ref = vec![0.0f64; n * n];
        tiled_matmul_f64(&a, &b, &mut c_ref, n, n, n);
        let c_mt = mt_matmul_f64(&a, &b, n, n, n);
        assert!(approx_eq(&c_ref, &c_mt, 1e-8));
    }

    #[test]
    fn test_fast_tensor_matmul_api() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let (c, shape) = fast_tensor_matmul(&a, &[2, 3], &[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);
        assert_eq!(shape, vec![2, 2]);
        assert!(approx_eq(&c, &[58.0, 64.0, 139.0, 154.0], 1e-10));
    }

    #[test]
    fn test_elementwise_add() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let mut out = vec![0.0; 5];
        elementwise_f64(ElemOp::Add, &a, &b, &mut out);
        assert!(approx_eq(&out, &[11.0, 22.0, 33.0, 44.0, 55.0], 1e-10));
    }

    #[test]
    fn test_elementwise_mul() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![10.0, 10.0, 10.0, 10.0];
        let mut out = vec![0.0; 4];
        elementwise_f64(ElemOp::Mul, &a, &b, &mut out);
        assert!(approx_eq(&out, &[20.0, 30.0, 40.0, 50.0], 1e-10));
    }

    #[test]
    fn test_broadcast_elementwise() {
        let a = FastTensor::from_f64(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = FastTensor::from_f64(vec![3], &[10.0, 20.0, 30.0]);
        let c = fast_elementwise(ElemOp::Add, &a, &b);
        assert_eq!(c.shape, vec![2, 3]);
        let vals = c.to_f64_vec();
        assert!(approx_eq(&vals, &[11.0, 22.0, 33.0, 14.0, 25.0, 36.0], 1e-10));
    }

    #[test]
    fn test_reduce_sum() {
        let t = FastTensor::from_f64(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = fast_reduce(ReduceOp::Sum, &t, 1);
        assert_eq!(r.shape, vec![2]);
        let vals = r.to_f64_vec();
        assert!(approx_eq(&vals, &[6.0, 15.0], 1e-10));
    }

    #[test]
    fn test_reduce_mean() {
        let t = FastTensor::from_f64(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let r = fast_reduce(ReduceOp::Mean, &t, 0);
        assert_eq!(r.shape, vec![3]);
        let vals = r.to_f64_vec();
        assert!(approx_eq(&vals, &[2.5, 3.5, 4.5], 1e-10));
    }

    #[test]
    fn test_reduce_max() {
        let t = FastTensor::from_f64(vec![2, 3], &[1.0, 5.0, 3.0, 4.0, 2.0, 6.0]);
        let r = fast_reduce(ReduceOp::Max, &t, 1);
        let vals = r.to_f64_vec();
        assert!(approx_eq(&vals, &[5.0, 6.0], 1e-10));
    }

    #[test]
    fn test_transpose() {
        let t = FastTensor::from_f64(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let tr = fast_transpose(&t);
        assert_eq!(tr.shape, vec![3, 2]);
        let vals = tr.to_f64_vec();
        assert!(approx_eq(&vals, &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0], 1e-10));
    }

    #[test]
    fn test_softmax() {
        let t = FastTensor::from_f64(vec![1, 3], &[1.0, 2.0, 3.0]);
        let s = fast_softmax(&t, 1);
        let vals = s.to_f64_vec();
        let sum: f64 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        // Values should be monotonically increasing
        assert!(vals[0] < vals[1] && vals[1] < vals[2]);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        // Large values shouldn't cause overflow
        let t = FastTensor::from_f64(vec![1, 3], &[1000.0, 1001.0, 1002.0]);
        let s = fast_softmax(&t, 1);
        let vals = s.to_f64_vec();
        let sum: f64 = vals.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(vals.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_layer_norm() {
        let t = FastTensor::from_f64(vec![2, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let gamma = FastTensor::from_f64(vec![4], &[1.0, 1.0, 1.0, 1.0]);
        let beta = FastTensor::from_f64(vec![4], &[0.0, 0.0, 0.0, 0.0]);
        let ln = fast_layer_norm(&t, &gamma, &beta, 1e-5);
        let vals = ln.to_f64_vec();
        // Each row should have mean ~0, std ~1
        let row0: Vec<f64> = vals[0..4].to_vec();
        let mean: f64 = row0.iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_f16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 0.5, 65504.0, -65504.0, 0.001];
        for &v in &values {
            let h = f32_to_f16(v);
            let back = f16_to_f32(h);
            let tol = if v.abs() > 1000.0 { v.abs() * 0.001 } else { 0.01 };
            assert!(
                (back - v).abs() < tol,
                "f16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_bf16_roundtrip() {
        let values = [0.0f32, 1.0, -1.0, 3.14, 100.0, -100.0];
        for &v in &values {
            let b = f32_to_bf16(v);
            let back = bf16_to_f32(b);
            assert!(
                (back - v).abs() < v.abs() * 0.02 + 0.01,
                "bf16 roundtrip failed for {}: got {}",
                v,
                back
            );
        }
    }

    #[test]
    fn test_quantize_dequantize() {
        let data = vec![0.0f32, 0.5, 1.0, -1.0, -0.5];
        let (q, params) = quantize_f32_to_i8(&data);
        let deq = dequantize_i8_to_f32(&q, &params);
        for (i, (&orig, &recovered)) in data.iter().zip(deq.iter()).enumerate() {
            assert!(
                (orig - recovered).abs() < 0.05,
                "quantize roundtrip failed at {}: {} vs {}",
                i,
                orig,
                recovered
            );
        }
    }

    #[test]
    fn test_memory_pool() {
        let pool = MemoryPool::new();
        let buf1 = pool.acquire(1024);
        assert_eq!(buf1.len(), 1024);
        pool.release(buf1);
        let buf2 = pool.acquire(1024);
        assert_eq!(buf2.len(), 1024);
        let (alloc, reused) = pool.stats();
        assert_eq!(alloc, 1);
        assert_eq!(reused, 1);
    }

    #[test]
    fn test_cast_f64_to_f32() {
        let t = FastTensor::from_f64(vec![3], &[1.0, 2.0, 3.0]);
        let t32 = t.cast(DType::F32);
        assert_eq!(t32.dtype, DType::F32);
        let vals: Vec<f64> = t32.to_f64_vec();
        assert!(approx_eq(&vals, &[1.0, 2.0, 3.0], 1e-5));
    }

    #[test]
    fn test_conv2d_basic() {
        // 1 batch, 1 channel, 4x4 input, 1 filter 2x2, stride 1, no padding
        let input = FastTensor::from_f64(
            vec![1, 1, 4, 4],
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0, 16.0,
            ],
        );
        let kernel = FastTensor::from_f64(vec![1, 1, 2, 2], &[1.0, 0.0, 0.0, 1.0]);
        let out = fast_conv2d(&input, &kernel, 1, 0);
        assert_eq!(out.shape, vec![1, 1, 3, 3]);
        let vals = out.to_f64_vec();
        // [0,0]: 1*1+2*0+5*0+6*1 = 7
        assert!((vals[0] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_thread_safety_matmul() {
        // Run multiple matmuls in parallel to test thread safety
        let handles: Vec<_> = (0..4)
            .map(|_| {
                thread::spawn(|| {
                    let a = vec![1.0f64; 16];
                    let b = vec![1.0f64; 16];
                    let (c, shape) = fast_tensor_matmul(&a, &[4, 4], &b, &[4, 4]);
                    assert_eq!(shape, vec![4, 4]);
                    // Each element should be 4.0 (sum of 4 ones)
                    assert!(c.iter().all(|&v| (v - 4.0).abs() < 1e-10));
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_global_pool() {
        let pool = global_pool();
        let buf = pool.acquire(256);
        assert_eq!(buf.len(), 256);
        pool.release(buf);
    }

    #[test]
    fn test_batched_matmul() {
        // [2,2,3] x [3,2] -> [2,2,2]
        let a = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // batch 0
            7.0, 8.0, 9.0, 10.0, 11.0, 12.0, // batch 1
        ];
        let b = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0]; // 3x2
        let (c, shape) = fast_tensor_batched_matmul(&a, &[2, 2, 3], &b, &[3, 2]);
        assert_eq!(shape, vec![2, 2, 2]);
        // batch 0, row 0: 1*1+2*0+3*1=4, 1*0+2*1+3*1=5
        assert!((c[0] - 4.0).abs() < 1e-10);
        assert!((c[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_set_threads() {
        set_thread_count(2);
        assert_eq!(get_thread_count(), 2);
        set_thread_count(0); // reset to auto
    }

    #[test]
    fn test_strides_row_major() {
        let s = compute_strides(&[2, 3, 4], Layout::RowMajor);
        assert_eq!(s, vec![12, 4, 1]);
    }

    #[test]
    fn test_strides_col_major() {
        let s = compute_strides(&[2, 3, 4], Layout::ColMajor);
        assert_eq!(s, vec![1, 2, 6]);
    }
}
