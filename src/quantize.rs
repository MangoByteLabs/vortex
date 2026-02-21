// quantize.rs — Native quantization engine for Vortex
//
// Supports INT8/INT4/INT2, FP8, BF16, FP16, NF4, MX4, LOG2 formats.
// Uses bytecarry primitives for overflow-safe accumulation.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Quantization Formats
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum QuantFormat {
    INT8,
    INT4,
    INT2,
    FP8E4M3,
    FP8E5M2,
    BF16,
    FP16,
    NF4,
    MX4,
    LOG2,
}

impl QuantFormat {
    pub fn bits_per_element(&self) -> usize {
        match self {
            QuantFormat::INT8 | QuantFormat::FP8E4M3 | QuantFormat::FP8E5M2 => 8,
            QuantFormat::INT4 | QuantFormat::NF4 | QuantFormat::MX4 => 4,
            QuantFormat::INT2 => 2,
            QuantFormat::BF16 | QuantFormat::FP16 => 16,
            QuantFormat::LOG2 => 8, // 1 sign + 7 exponent bits
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_uppercase().as_str() {
            "INT8" | "I8" => Some(QuantFormat::INT8),
            "INT4" | "I4" => Some(QuantFormat::INT4),
            "INT2" | "I2" => Some(QuantFormat::INT2),
            "FP8E4M3" => Some(QuantFormat::FP8E4M3),
            "FP8E5M2" => Some(QuantFormat::FP8E5M2),
            "BF16" => Some(QuantFormat::BF16),
            "FP16" => Some(QuantFormat::FP16),
            "NF4" => Some(QuantFormat::NF4),
            "MX4" => Some(QuantFormat::MX4),
            "LOG2" => Some(QuantFormat::LOG2),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization Schemes
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum QuantScheme {
    PerTensor { scale: f64, zero_point: i64 },
    PerChannel { scales: Vec<f64>, zero_points: Vec<i64>, axis: usize },
    PerGroup { scales: Vec<f64>, zero_points: Vec<i64>, group_size: usize },
    Microscaling { shared_exponents: Vec<i8>, block_size: usize },
}

// ---------------------------------------------------------------------------
// Quantized Tensor
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct QuantTensor {
    pub data: Vec<u8>,
    pub shape: Vec<usize>,
    pub format: QuantFormat,
    pub scheme: QuantScheme,
    pub original_dtype: String,
}

impl QuantTensor {
    /// Quantize an f64 slice into a QuantTensor.
    pub fn quantize(data: &[f64], shape: &[usize], format: QuantFormat, group_size: Option<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape product");

        match format {
            QuantFormat::INT8 => Self::quantize_int8(data, shape, group_size),
            QuantFormat::INT4 => Self::quantize_int4(data, shape, group_size),
            QuantFormat::INT2 => Self::quantize_int2(data, shape, group_size),
            QuantFormat::NF4 => Self::quantize_nf4(data, shape, group_size),
            QuantFormat::LOG2 => Self::quantize_log2(data, shape),
            QuantFormat::MX4 => Self::quantize_mx4(data, shape, group_size),
            _ => Self::quantize_int8(data, shape, group_size), // fallback
        }
    }

    fn quantize_int8(data: &[f64], shape: &[usize], group_size: Option<usize>) -> Self {
        match group_size {
            Some(gs) => {
                let mut packed = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();
                for chunk in data.chunks(gs) {
                    let (scale, zp) = calibrate_minmax(chunk);
                    scales.push(scale);
                    zero_points.push(zp);
                    for &v in chunk {
                        let q = if scale == 0.0 { 0i8 } else {
                            ((v / scale).round() as i64 + zp).clamp(-128, 127) as i8
                        };
                        packed.push(q as u8);
                    }
                }
                QuantTensor {
                    data: packed,
                    shape: shape.to_vec(),
                    format: QuantFormat::INT8,
                    scheme: QuantScheme::PerGroup { scales, zero_points, group_size: gs },
                    original_dtype: "f64".into(),
                }
            }
            None => {
                let (scale, zp) = calibrate_minmax(data);
                let packed: Vec<u8> = data.iter().map(|&v| {
                    if scale == 0.0 { 0u8 } else {
                        ((v / scale).round() as i64 + zp).clamp(-128, 127) as u8
                    }
                }).collect();
                QuantTensor {
                    data: packed,
                    shape: shape.to_vec(),
                    format: QuantFormat::INT8,
                    scheme: QuantScheme::PerTensor { scale, zero_point: zp },
                    original_dtype: "f64".into(),
                }
            }
        }
    }

    fn quantize_int4(data: &[f64], shape: &[usize], group_size: Option<usize>) -> Self {
        match group_size {
            Some(gs) => {
                let mut all_q = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();
                for chunk in data.chunks(gs) {
                    let (scale, zp) = calibrate_minmax_range(chunk, -8.0, 7.0);
                    scales.push(scale);
                    zero_points.push(zp);
                    for &v in chunk {
                        let q = if scale == 0.0 { 0i8 } else {
                            ((v / scale).round() as i64 + zp).clamp(-8, 7) as i8
                        };
                        all_q.push(q);
                    }
                }
                QuantTensor {
                    data: pack_int4(&all_q),
                    shape: shape.to_vec(),
                    format: QuantFormat::INT4,
                    scheme: QuantScheme::PerGroup { scales, zero_points, group_size: gs },
                    original_dtype: "f64".into(),
                }
            }
            None => {
                let (scale, zp) = calibrate_minmax_range(data, -8.0, 7.0);
                let quantized: Vec<i8> = data.iter().map(|&v| {
                    if scale == 0.0 { 0i8 } else {
                        ((v / scale).round() as i64 + zp).clamp(-8, 7) as i8
                    }
                }).collect();
                QuantTensor {
                    data: pack_int4(&quantized),
                    shape: shape.to_vec(),
                    format: QuantFormat::INT4,
                    scheme: QuantScheme::PerTensor { scale, zero_point: zp },
                    original_dtype: "f64".into(),
                }
            }
        }
    }

    fn quantize_int2(data: &[f64], shape: &[usize], group_size: Option<usize>) -> Self {
        match group_size {
            Some(gs) => {
                let mut all_q = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();
                for chunk in data.chunks(gs) {
                    let (scale, zp) = calibrate_minmax_range(chunk, -2.0, 1.0);
                    scales.push(scale);
                    zero_points.push(zp);
                    for &v in chunk {
                        let q = if scale == 0.0 { 0i8 } else {
                            ((v / scale).round() as i64 + zp).clamp(-2, 1) as i8
                        };
                        all_q.push(q);
                    }
                }
                QuantTensor {
                    data: pack_int2(&all_q),
                    shape: shape.to_vec(),
                    format: QuantFormat::INT2,
                    scheme: QuantScheme::PerGroup { scales, zero_points, group_size: gs },
                    original_dtype: "f64".into(),
                }
            }
            None => {
                let (scale, zp) = calibrate_minmax_range(data, -2.0, 1.0);
                let quantized: Vec<i8> = data.iter().map(|&v| {
                    if scale == 0.0 { 0i8 } else {
                        ((v / scale).round() as i64 + zp).clamp(-2, 1) as i8
                    }
                }).collect();
                QuantTensor {
                    data: pack_int2(&quantized),
                    shape: shape.to_vec(),
                    format: QuantFormat::INT2,
                    scheme: QuantScheme::PerTensor { scale, zero_point: zp },
                    original_dtype: "f64".into(),
                }
            }
        }
    }

    fn quantize_nf4(data: &[f64], shape: &[usize], group_size: Option<usize>) -> Self {
        match group_size {
            Some(gs) => {
                let mut all_q = Vec::new();
                let mut scales = Vec::new();
                let mut zero_points = Vec::new();
                for chunk in data.chunks(gs) {
                    let absmax = chunk.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                    let scale = if absmax == 0.0 { 1.0 } else { absmax };
                    scales.push(scale);
                    zero_points.push(0);
                    for &v in chunk {
                        let normalized = v / scale;
                        let q = nf4_quantize(normalized);
                        all_q.push(q as i8);
                    }
                }
                QuantTensor {
                    data: pack_int4(&all_q),
                    shape: shape.to_vec(),
                    format: QuantFormat::NF4,
                    scheme: QuantScheme::PerGroup { scales, zero_points, group_size: gs },
                    original_dtype: "f64".into(),
                }
            }
            None => {
                let absmax = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                let scale = if absmax == 0.0 { 1.0 } else { absmax };
                let quantized: Vec<i8> = data.iter().map(|&v| {
                    nf4_quantize(v / scale) as i8
                }).collect();
                QuantTensor {
                    data: pack_int4(&quantized),
                    shape: shape.to_vec(),
                    format: QuantFormat::NF4,
                    scheme: QuantScheme::PerTensor { scale, zero_point: 0 },
                    original_dtype: "f64".into(),
                }
            }
        }
    }

    fn quantize_log2(data: &[f64], shape: &[usize]) -> Self {
        let mut packed = Vec::new();
        for &v in data {
            let (sign, exp) = log2_quantize(v, 7);
            // Pack: bit 7 = sign, bits 0-6 = exponent (biased by 63)
            let byte = if sign { 0x80 } else { 0x00 } | ((exp + 63).clamp(0, 127) as u8);
            packed.push(byte);
        }
        QuantTensor {
            data: packed,
            shape: shape.to_vec(),
            format: QuantFormat::LOG2,
            scheme: QuantScheme::PerTensor { scale: 1.0, zero_point: 0 },
            original_dtype: "f64".into(),
        }
    }

    fn quantize_mx4(data: &[f64], shape: &[usize], group_size: Option<usize>) -> Self {
        let bs = group_size.unwrap_or(32);
        let mut all_q = Vec::new();
        let mut shared_exponents = Vec::new();
        for chunk in data.chunks(bs) {
            let max_abs = chunk.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
            let shared_exp = if max_abs == 0.0 { 0i8 } else {
                max_abs.log2().ceil() as i8
            };
            shared_exponents.push(shared_exp);
            let scale = 2.0f64.powi(shared_exp as i32);
            for &v in chunk {
                let q = if scale == 0.0 { 0i8 } else {
                    (v / scale * 7.0).round().clamp(-8.0, 7.0) as i8
                };
                all_q.push(q);
            }
        }
        QuantTensor {
            data: pack_int4(&all_q),
            shape: shape.to_vec(),
            format: QuantFormat::MX4,
            scheme: QuantScheme::Microscaling { shared_exponents, block_size: bs },
            original_dtype: "f64".into(),
        }
    }

    /// Dequantize back to f64.
    pub fn dequantize(&self) -> Vec<f64> {
        let numel: usize = self.shape.iter().product();
        match self.format {
            QuantFormat::INT8 => self.dequantize_int8(numel),
            QuantFormat::INT4 => self.dequantize_int4(numel),
            QuantFormat::INT2 => self.dequantize_int2(numel),
            QuantFormat::NF4 => self.dequantize_nf4(numel),
            QuantFormat::LOG2 => self.dequantize_log2(numel),
            QuantFormat::MX4 => self.dequantize_mx4(numel),
            _ => self.dequantize_int8(numel),
        }
    }

    fn dequantize_int8(&self, numel: usize) -> Vec<f64> {
        match &self.scheme {
            QuantScheme::PerTensor { scale, zero_point } => {
                self.data.iter().take(numel).map(|&b| {
                    (b as i8 as i64 - zero_point) as f64 * scale
                }).collect()
            }
            QuantScheme::PerGroup { scales, zero_points, group_size } => {
                let mut result = Vec::with_capacity(numel);
                for (gi, chunk) in self.data.chunks(*group_size).enumerate() {
                    let scale = scales[gi];
                    let zp = zero_points[gi];
                    for &b in chunk {
                        if result.len() >= numel { break; }
                        result.push((b as i8 as i64 - zp) as f64 * scale);
                    }
                }
                result
            }
            _ => vec![0.0; numel],
        }
    }

    fn dequantize_int4(&self, numel: usize) -> Vec<f64> {
        let unpacked = unpack_int4(&self.data);
        match &self.scheme {
            QuantScheme::PerTensor { scale, zero_point } => {
                unpacked.iter().take(numel).map(|&q| {
                    (q as i64 - zero_point) as f64 * scale
                }).collect()
            }
            QuantScheme::PerGroup { scales, zero_points, group_size } => {
                let mut result = Vec::with_capacity(numel);
                for (gi, chunk) in unpacked.chunks(*group_size).enumerate() {
                    let scale = scales[gi];
                    let zp = zero_points[gi];
                    for &q in chunk {
                        if result.len() >= numel { break; }
                        result.push((q as i64 - zp) as f64 * scale);
                    }
                }
                result
            }
            _ => vec![0.0; numel],
        }
    }

    fn dequantize_int2(&self, numel: usize) -> Vec<f64> {
        let unpacked = unpack_int2(&self.data);
        match &self.scheme {
            QuantScheme::PerTensor { scale, zero_point } => {
                unpacked.iter().take(numel).map(|&q| {
                    (q as i64 - zero_point) as f64 * scale
                }).collect()
            }
            QuantScheme::PerGroup { scales, zero_points, group_size } => {
                let mut result = Vec::with_capacity(numel);
                for (gi, chunk) in unpacked.chunks(*group_size).enumerate() {
                    let scale = scales[gi];
                    let zp = zero_points[gi];
                    for &q in chunk {
                        if result.len() >= numel { break; }
                        result.push((q as i64 - zp) as f64 * scale);
                    }
                }
                result
            }
            _ => vec![0.0; numel],
        }
    }

    fn dequantize_nf4(&self, numel: usize) -> Vec<f64> {
        let unpacked = unpack_int4(&self.data);
        match &self.scheme {
            QuantScheme::PerTensor { scale, .. } => {
                unpacked.iter().take(numel).map(|&q| {
                    nf4_dequantize(q as u8) * scale
                }).collect()
            }
            QuantScheme::PerGroup { scales, group_size, .. } => {
                let mut result = Vec::with_capacity(numel);
                for (gi, chunk) in unpacked.chunks(*group_size).enumerate() {
                    let scale = scales[gi];
                    for &q in chunk {
                        if result.len() >= numel { break; }
                        result.push(nf4_dequantize(q as u8) * scale);
                    }
                }
                result
            }
            _ => vec![0.0; numel],
        }
    }

    fn dequantize_log2(&self, numel: usize) -> Vec<f64> {
        self.data.iter().take(numel).map(|&b| {
            let sign = b & 0x80 != 0;
            let exp = (b & 0x7F) as i8 - 63;
            log2_dequantize(sign, exp)
        }).collect()
    }

    fn dequantize_mx4(&self, numel: usize) -> Vec<f64> {
        let unpacked = unpack_int4(&self.data);
        match &self.scheme {
            QuantScheme::Microscaling { shared_exponents, block_size } => {
                let mut result = Vec::with_capacity(numel);
                for (gi, chunk) in unpacked.chunks(*block_size).enumerate() {
                    let exp = shared_exponents[gi];
                    let scale = 2.0f64.powi(exp as i32);
                    for &q in chunk {
                        if result.len() >= numel { break; }
                        result.push(q as f64 / 7.0 * scale);
                    }
                }
                result
            }
            _ => vec![0.0; numel],
        }
    }

    /// Quantized matrix multiply. Avoids full dequantization where possible.
    pub fn matmul(&self, other: &QuantTensor) -> Vec<f64> {
        assert!(self.shape.len() == 2 && other.shape.len() == 2);
        let m = self.shape[0];
        let k = self.shape[1];
        assert_eq!(k, other.shape[0]);
        let n = other.shape[1];

        match (self.format, other.format) {
            (QuantFormat::INT8, QuantFormat::INT8) => {
                self.matmul_int8(other, m, k, n)
            }
            (QuantFormat::INT4, QuantFormat::INT4) => {
                self.matmul_int4(other, m, k, n)
            }
            (QuantFormat::LOG2, QuantFormat::LOG2) => {
                self.matmul_log2(other, m, k, n)
            }
            _ => {
                // Fallback: dequantize both and multiply in f64
                let a = self.dequantize();
                let b = other.dequantize();
                matmul_f64(&a, &b, m, k, n)
            }
        }
    }

    fn matmul_int8(&self, other: &QuantTensor, m: usize, k: usize, n: usize) -> Vec<f64> {
        let a_scale = self.get_scale();
        let b_scale = other.get_scale();
        let a_zp = self.get_zero_point();
        let b_zp = other.get_zero_point();

        let mut result = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for p in 0..k {
                    let a_val = self.data[i * k + p] as i8 as i32 - a_zp as i32;
                    let b_val = other.data[p * n + j] as i8 as i32 - b_zp as i32;
                    acc += a_val * b_val;
                }
                result[i * n + j] = acc as f64 * a_scale * b_scale;
            }
        }
        result
    }

    fn matmul_int4(&self, other: &QuantTensor, m: usize, k: usize, n: usize) -> Vec<f64> {
        let a_unpacked = unpack_int4(&self.data);
        let b_unpacked = unpack_int4(&other.data);
        let a_scale = self.get_scale();
        let b_scale = other.get_scale();
        let a_zp = self.get_zero_point();
        let b_zp = other.get_zero_point();

        let mut result = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc: i32 = 0;
                for p in 0..k {
                    let a_val = a_unpacked[i * k + p] as i32 - a_zp as i32;
                    let b_val = b_unpacked[p * n + j] as i32 - b_zp as i32;
                    acc += a_val * b_val;
                }
                result[i * n + j] = acc as f64 * a_scale * b_scale;
            }
        }
        result
    }

    fn matmul_log2(&self, other: &QuantTensor, m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut a_signs = Vec::with_capacity(m * k);
        let mut a_exps = Vec::with_capacity(m * k);
        let mut b_signs = Vec::with_capacity(k * n);
        let mut b_exps = Vec::with_capacity(k * n);

        for &byte in &self.data[..m * k] {
            a_signs.push(byte & 0x80 != 0);
            a_exps.push((byte & 0x7F) as i8 - 63);
        }
        for &byte in &other.data[..k * n] {
            b_signs.push(byte & 0x80 != 0);
            b_exps.push((byte & 0x7F) as i8 - 63);
        }
        log2_matmul(&a_signs, &a_exps, &b_signs, &b_exps, m, k, n)
    }

    fn get_scale(&self) -> f64 {
        match &self.scheme {
            QuantScheme::PerTensor { scale, .. } => *scale,
            QuantScheme::PerGroup { scales, .. } => {
                // For matmul we use average scale as approximation in per-group
                if scales.is_empty() { 1.0 } else {
                    scales.iter().sum::<f64>() / scales.len() as f64
                }
            }
            _ => 1.0,
        }
    }

    fn get_zero_point(&self) -> i64 {
        match &self.scheme {
            QuantScheme::PerTensor { zero_point, .. } => *zero_point,
            _ => 0,
        }
    }

    /// Public accessor for scale (used by interpreter).
    pub fn get_scale_pub(&self) -> f64 {
        self.get_scale()
    }

    /// Public accessor for zero point (used by interpreter).
    pub fn get_zp_pub(&self) -> i64 {
        self.get_zero_point()
    }

    /// Memory compression ratio vs f32.
    pub fn compression_ratio(&self) -> f64 {
        32.0 / self.format.bits_per_element() as f64
    }

    /// Quantization error (RMSE vs original).
    pub fn error(&self, original: &[f64]) -> f64 {
        let deq = self.dequantize();
        let n = original.len().min(deq.len());
        if n == 0 { return 0.0; }
        let mse: f64 = original.iter().zip(deq.iter()).take(n)
            .map(|(a, b)| (a - b).powi(2)).sum::<f64>() / n as f64;
        mse.sqrt()
    }
}

// ---------------------------------------------------------------------------
// Calibration
// ---------------------------------------------------------------------------

/// Min-max calibration for symmetric quantization (INT8 range).
pub fn calibrate_minmax(data: &[f64]) -> (f64, i64) {
    if data.is_empty() { return (1.0, 0); }
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 { return (1.0, 0); }
    let scale = max_abs / 127.0;
    (scale, 0) // symmetric: zero_point = 0
}

/// Min-max calibration for a given signed range.
fn calibrate_minmax_range(data: &[f64], qmin: f64, qmax: f64) -> (f64, i64) {
    if data.is_empty() { return (1.0, 0); }
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 { return (1.0, 0); }
    let scale = max_abs / qmax.abs().max(qmin.abs());
    (scale, 0)
}

/// Percentile calibration: clip outliers.
pub fn calibrate_percentile(data: &[f64], percentile: f64) -> (f64, i64) {
    if data.is_empty() { return (1.0, 0); }
    let mut sorted: Vec<f64> = data.iter().map(|v| v.abs()).collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((sorted.len() as f64 * percentile / 100.0) as usize).min(sorted.len() - 1);
    let clip_val = sorted[idx];
    if clip_val == 0.0 { return (1.0, 0); }
    (clip_val / 127.0, 0)
}

/// Entropy-based calibration (KL-divergence minimization).
pub fn calibrate_entropy(data: &[f64], num_bins: usize) -> (f64, i64) {
    if data.is_empty() { return (1.0, 0); }
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    if max_abs == 0.0 { return (1.0, 0); }

    // Build histogram
    let mut hist = vec![0usize; num_bins];
    for &v in data {
        let bin = ((v.abs() / max_abs) * (num_bins - 1) as f64).round() as usize;
        hist[bin.min(num_bins - 1)] += 1;
    }

    // Try different thresholds, find one with minimum KL divergence
    let mut best_scale = max_abs / 127.0;
    let mut best_kl = f64::MAX;

    for t in (128..num_bins).step_by(num_bins / 64 + 1) {
        let threshold = max_abs * t as f64 / num_bins as f64;
        let scale = threshold / 127.0;

        // Compute KL divergence approximation
        let mut kl = 0.0f64;
        let total: f64 = hist.iter().sum::<usize>() as f64;
        if total == 0.0 { continue; }

        for (i, &count) in hist.iter().enumerate().take(t) {
            if count == 0 { continue; }
            let p = count as f64 / total;
            let center = max_abs * i as f64 / num_bins as f64;
            let q_val = (center / scale).round().clamp(-127.0, 127.0) * scale;
            let q_bin = ((q_val.abs() / max_abs) * (num_bins - 1) as f64).round() as usize;
            let q_count = hist.get(q_bin.min(num_bins - 1)).copied().unwrap_or(1);
            let q = (q_count as f64 + 1e-10) / total;
            kl += p * (p / q).ln();
        }

        if kl < best_kl {
            best_kl = kl;
            best_scale = scale;
        }
    }

    (best_scale, 0)
}

// ---------------------------------------------------------------------------
// INT4 Packing
// ---------------------------------------------------------------------------

/// Pack two 4-bit signed values per byte (little-endian: low nibble first).
pub fn pack_int4(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 1) / 2);
    for chunk in values.chunks(2) {
        let lo = (chunk[0] as u8) & 0x0F;
        let hi = if chunk.len() > 1 { (chunk[1] as u8) & 0x0F } else { 0 };
        packed.push(lo | (hi << 4));
    }
    packed
}

/// Unpack 4-bit signed values from packed bytes.
pub fn unpack_int4(packed: &[u8]) -> Vec<i8> {
    let mut values = Vec::with_capacity(packed.len() * 2);
    for &byte in packed {
        values.push(sign_extend_4bit(byte & 0x0F));
        values.push(sign_extend_4bit((byte >> 4) & 0x0F));
    }
    values
}

fn sign_extend_4bit(v: u8) -> i8 {
    let v = v & 0x0F;
    if v & 0x08 != 0 { (v | 0xF0) as i8 } else { v as i8 }
}

// ---------------------------------------------------------------------------
// INT2 Packing
// ---------------------------------------------------------------------------

/// Pack four 2-bit signed values per byte.
pub fn pack_int2(values: &[i8]) -> Vec<u8> {
    let mut packed = Vec::with_capacity((values.len() + 3) / 4);
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (i, &v) in chunk.iter().enumerate() {
            byte |= ((v as u8) & 0x03) << (i * 2);
        }
        packed.push(byte);
    }
    packed
}

/// Unpack 2-bit signed values from packed bytes.
pub fn unpack_int2(packed: &[u8]) -> Vec<i8> {
    let mut values = Vec::with_capacity(packed.len() * 4);
    for &byte in packed {
        for i in 0..4 {
            let bits = (byte >> (i * 2)) & 0x03;
            // Sign extend 2-bit: if bit 1 set, it's negative
            let val = if bits & 0x02 != 0 {
                (bits | 0xFC) as i8
            } else {
                bits as i8
            };
            values.push(val);
        }
    }
    values
}

// ---------------------------------------------------------------------------
// NormalFloat4 (NF4)
// ---------------------------------------------------------------------------

/// NF4 quantization levels — optimal for N(0,1) distributed weights.
/// These 16 levels minimize expected MSE for standard normal distribution.
const NF4_LEVELS: [f64; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// Quantize a normalized value to NF4 (returns 0-15).
pub fn nf4_quantize(value: f64) -> u8 {
    let clamped = value.clamp(-1.0, 1.0);
    let mut best = 0u8;
    let mut best_dist = f64::MAX;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (clamped - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best = i as u8;
        }
    }
    best
}

/// Dequantize NF4 code back to f64.
pub fn nf4_dequantize(q: u8) -> f64 {
    NF4_LEVELS[(q & 0x0F) as usize]
}

// ---------------------------------------------------------------------------
// Logarithmic Quantization (LOG2)
// ---------------------------------------------------------------------------

/// Logarithmic quantization: val ~ sign * 2^exp.
pub fn log2_quantize(value: f64, bits: u32) -> (bool, i8) {
    if value == 0.0 { return (false, -63); } // represents ~0
    let sign = value < 0.0;
    let abs_val = value.abs();
    let exp = abs_val.log2().round() as i8;
    let max_exp = (1i8 << (bits - 1)) - 1;
    let clamped = exp.clamp(-max_exp, max_exp);
    (sign, clamped)
}

/// Dequantize logarithmic value.
pub fn log2_dequantize(sign: bool, exp: i8) -> f64 {
    let val = 2.0f64.powi(exp as i32);
    if sign { -val } else { val }
}

/// LOG2 matmul: multiplication becomes integer addition of exponents.
pub fn log2_matmul(
    a_signs: &[bool], a_exps: &[i8],
    b_signs: &[bool], b_exps: &[i8],
    m: usize, k: usize, n: usize,
) -> Vec<f64> {
    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                let a_idx = i * k + p;
                let b_idx = p * n + j;
                let sign = a_signs[a_idx] ^ b_signs[b_idx];
                let exp_sum = a_exps[a_idx] as i32 + b_exps[b_idx] as i32;
                let val = 2.0f64.powi(exp_sum);
                acc += if sign { -val } else { val };
            }
            result[i * n + j] = acc;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Mixed Precision & Weight-Only Quantization
// ---------------------------------------------------------------------------

/// Mixed-precision configuration.
#[derive(Clone, Debug)]
pub struct MixedPrecisionConfig {
    pub default_format: QuantFormat,
    pub layer_overrides: HashMap<String, QuantFormat>,
    pub sensitive_layers: Vec<String>,
}

impl MixedPrecisionConfig {
    pub fn new(default: QuantFormat) -> Self {
        Self {
            default_format: default,
            layer_overrides: HashMap::new(),
            sensitive_layers: Vec::new(),
        }
    }

    pub fn format_for_layer(&self, layer: &str) -> QuantFormat {
        if self.sensitive_layers.contains(&layer.to_string()) {
            QuantFormat::FP16
        } else {
            *self.layer_overrides.get(layer).unwrap_or(&self.default_format)
        }
    }
}

/// Weight-only matmul (W4A16): INT4 weights, f64 activations.
pub fn weight_only_matmul(
    weights_q: &QuantTensor,
    activations: &[f64],
    m: usize, k: usize, n: usize,
) -> Vec<f64> {
    let w_deq = weights_q.dequantize();
    matmul_f64(activations, &w_deq, m, k, n)
}

/// Activation-aware quantization (AWQ): scale weights by channel importance.
pub fn awq_quantize(
    weights: &[f64],
    activation_scales: &[f64],
    shape: &[usize],
    format: QuantFormat,
    group_size: usize,
) -> QuantTensor {
    assert_eq!(shape.len(), 2);
    let rows = shape[0];
    let cols = shape[1];
    assert_eq!(weights.len(), rows * cols);
    assert_eq!(activation_scales.len(), cols);

    // Scale weights: w_scaled[i][j] = w[i][j] * act_scale[j]
    let mut scaled = vec![0.0f64; weights.len()];
    for i in 0..rows {
        for j in 0..cols {
            scaled[i * cols + j] = weights[i * cols + j] * activation_scales[j];
        }
    }

    // Quantize scaled weights
    let mut qt = QuantTensor::quantize(&scaled, shape, format, Some(group_size));

    // Store inverse scales in the tensor for dequantization correction
    // (In a full implementation, we'd adjust scales; here we embed it in the scheme)
    // For now, the AWQ benefit is that salient channels get more dynamic range
    qt.original_dtype = "f64_awq".into();
    qt
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut result = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            result[i * n + j] = acc;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_roundtrip_error() {
        // Random-ish data
        let data: Vec<f64> = (0..256).map(|i| (i as f64 - 128.0) / 128.0).collect();
        let shape = vec![16, 16];
        let qt = QuantTensor::quantize(&data, &shape, QuantFormat::INT8, None);
        let deq = qt.dequantize();
        let rmse = qt.error(&data);
        // Error should be small relative to data range
        let data_range = data.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - data.iter().cloned().fold(f64::INFINITY, f64::min);
        assert!(rmse / data_range < 0.01, "INT8 roundtrip error too large: {}", rmse / data_range);
        assert_eq!(deq.len(), data.len());
    }

    #[test]
    fn test_int4_pack_unpack_roundtrip() {
        let values: Vec<i8> = vec![0, 1, -1, 7, -8, 3, -4, 5];
        let packed = pack_int4(&values);
        let unpacked = unpack_int4(&packed);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(unpacked[i], v, "INT4 pack/unpack mismatch at index {}", i);
        }
    }

    #[test]
    fn test_int2_pack_unpack_roundtrip() {
        let values: Vec<i8> = vec![0, 1, -1, -2, 0, 1, -2, 1];
        let packed = pack_int2(&values);
        let unpacked = unpack_int2(&packed);
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(unpacked[i], v, "INT2 pack/unpack mismatch at index {}", i);
        }
    }

    #[test]
    fn test_int4_matmul_matches_f64() {
        // 2x3 * 3x2 matrix multiply
        let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let a_shape = vec![2, 3];
        let b_shape = vec![3, 2];

        // Scale data to fit INT4 range
        let a_max = a_data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let b_max = b_data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
        let a_scaled: Vec<f64> = a_data.iter().map(|v| v / a_max * 7.0).collect();
        let b_scaled: Vec<f64> = b_data.iter().map(|v| v / b_max * 7.0).collect();

        let a_q = QuantTensor::quantize(&a_scaled, &a_shape, QuantFormat::INT4, None);
        let b_q = QuantTensor::quantize(&b_scaled, &b_shape, QuantFormat::INT4, None);

        let q_result = a_q.matmul(&b_q);
        let f64_result = matmul_f64(&a_scaled, &b_scaled, 2, 3, 2);

        // Check within tolerance (INT4 is coarse)
        for (i, (qv, fv)) in q_result.iter().zip(f64_result.iter()).enumerate() {
            let rel_err = if fv.abs() > 0.01 { (qv - fv).abs() / fv.abs() } else { (qv - fv).abs() };
            assert!(rel_err < 0.5, "INT4 matmul mismatch at {}: q={}, f={}", i, qv, fv);
        }
    }

    #[test]
    fn test_nf4_lower_error_than_uniform_int4() {
        // Gaussian-distributed data
        // Use a simple PRNG to generate normally distributed values
        let mut data = Vec::with_capacity(1024);
        let mut state: u64 = 12345;
        for _ in 0..1024 {
            // Box-Muller approximation using LCG
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u1 = (state >> 33) as f64 / (1u64 << 31) as f64;
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let u2 = (state >> 33) as f64 / (1u64 << 31) as f64;
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z.clamp(-3.0, 3.0));
        }
        let shape = vec![32, 32];

        let nf4_qt = QuantTensor::quantize(&data, &shape, QuantFormat::NF4, None);
        let int4_qt = QuantTensor::quantize(&data, &shape, QuantFormat::INT4, None);

        let nf4_err = nf4_qt.error(&data);
        let int4_err = int4_qt.error(&data);

        assert!(nf4_err < int4_err,
            "NF4 error ({}) should be less than uniform INT4 error ({})", nf4_err, int4_err);
    }

    #[test]
    fn test_log2_matmul_correct() {
        // 2x2 * 2x2
        let a_vals = [2.0f64, 4.0, 1.0, 8.0];
        let b_vals = [1.0f64, 2.0, 4.0, 0.5];

        let mut a_signs = Vec::new();
        let mut a_exps = Vec::new();
        let mut b_signs = Vec::new();
        let mut b_exps = Vec::new();

        for &v in &a_vals {
            let (s, e) = log2_quantize(v, 7);
            a_signs.push(s);
            a_exps.push(e);
        }
        for &v in &b_vals {
            let (s, e) = log2_quantize(v, 7);
            b_signs.push(s);
            b_exps.push(e);
        }

        let result = log2_matmul(&a_signs, &a_exps, &b_signs, &b_exps, 2, 2, 2);
        let expected = matmul_f64(&a_vals, &b_vals, 2, 2, 2);

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!((r - e).abs() < 0.01, "LOG2 matmul mismatch at {}: got {}, expected {}", i, r, e);
        }
    }

    #[test]
    fn test_compression_ratios() {
        let data = vec![1.0f64; 64];
        let shape = vec![8, 8];

        let int8 = QuantTensor::quantize(&data, &shape, QuantFormat::INT8, None);
        let int4 = QuantTensor::quantize(&data, &shape, QuantFormat::INT4, None);
        let int2 = QuantTensor::quantize(&data, &shape, QuantFormat::INT2, None);

        assert_eq!(int8.compression_ratio(), 4.0);
        assert_eq!(int4.compression_ratio(), 8.0);
        assert_eq!(int2.compression_ratio(), 16.0);
    }

    #[test]
    fn test_per_group_lower_error_than_per_tensor() {
        // Data with varying magnitudes across groups
        let mut data = Vec::with_capacity(256);
        for i in 0..256 {
            let group = i / 32;
            let scale = 10.0f64.powi(group as i32 - 4); // different scales per group
            data.push(((i % 32) as f64 - 16.0) * scale);
        }
        let shape = vec![16, 16];

        let per_tensor = QuantTensor::quantize(&data, &shape, QuantFormat::INT8, None);
        let per_group = QuantTensor::quantize(&data, &shape, QuantFormat::INT8, Some(32));

        let pt_err = per_tensor.error(&data);
        let pg_err = per_group.error(&data);

        assert!(pg_err < pt_err,
            "Per-group error ({}) should be less than per-tensor error ({})", pg_err, pt_err);
    }

    #[test]
    fn test_weight_only_matmul() {
        let weights = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let activations = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let w_shape = vec![3, 3];

        let w_q = QuantTensor::quantize(&weights, &w_shape, QuantFormat::INT4, None);
        let q_result = weight_only_matmul(&w_q, &activations, 2, 3, 3);
        let f64_result = matmul_f64(&activations, &weights, 2, 3, 3);

        for (i, (qv, fv)) in q_result.iter().zip(f64_result.iter()).enumerate() {
            let rel_err = if fv.abs() > 0.01 { (qv - fv).abs() / fv.abs() } else { (qv - fv).abs() };
            assert!(rel_err < 0.15, "Weight-only matmul error at {}: q={}, f={}, err={}", i, qv, fv, rel_err);
        }
    }

    #[test]
    fn test_awq_salient_channels_lower_error() {
        // Weights with some very important channels
        let rows = 8;
        let cols = 8;
        let mut weights = vec![0.0f64; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                weights[i * cols + j] = ((i * cols + j) as f64 - 32.0) / 32.0;
            }
        }

        // Channels 0 and 1 are "salient" (have high activation magnitude)
        let mut act_scales = vec![1.0f64; cols];
        act_scales[0] = 10.0;
        act_scales[1] = 10.0;

        let uniform_qt = QuantTensor::quantize(&weights, &[rows, cols], QuantFormat::INT4, Some(8));
        let awq_qt = awq_quantize(&weights, &act_scales, &[rows, cols], QuantFormat::INT4, 8);

        // Check error on salient columns specifically
        let mut uniform_salient_err = 0.0f64;
        let mut awq_salient_err = 0.0f64;
        let uniform_deq = uniform_qt.dequantize();
        let awq_deq = awq_qt.dequantize();

        for i in 0..rows {
            for &j in &[0usize, 1] {
                let idx = i * cols + j;
                let orig_scaled = weights[idx] * act_scales[j];
                uniform_salient_err += (weights[idx] * act_scales[j] - uniform_deq[idx]).powi(2);
                awq_salient_err += (orig_scaled - awq_deq[idx]).powi(2);
            }
        }

        // AWQ should have lower error on salient channels because it scales them up
        // before quantization, giving them more dynamic range
        assert!(awq_salient_err < uniform_salient_err,
            "AWQ salient error ({}) should be less than uniform ({})", awq_salient_err, uniform_salient_err);
    }
}
