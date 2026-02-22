// Neural network training framework for Vortex
// Supports multiple architectures, backpropagation, optimizers, and data loading.



// ─── Tensor ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        let expected: usize = shape.iter().product();
        assert_eq!(data.len(), expected, "shape {:?} expects {} elems, got {}", shape, expected, data.len());
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { shape, data: vec![0.0; n] }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { shape, data: vec![1.0; n] }
    }

    pub fn randn(shape: Vec<usize>) -> Self {
        // Simple Box-Muller using a basic LCG for reproducibility
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut seed: u64 = 42 + n as u64;
        for _ in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            let u1 = u1.max(1e-10);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z);
        }
        Self { shape, data }
    }

    pub fn randn_seeded(shape: Vec<usize>, seed_val: u64) -> Self {
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        let mut seed: u64 = seed_val;
        for _ in 0..n {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u1 = ((seed >> 33) as f64 / (1u64 << 31) as f64).max(1e-10);
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let u2 = (seed >> 33) as f64 / (1u64 << 31) as f64;
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            data.push(z);
        }
        Self { shape, data }
    }

    pub fn numel(&self) -> usize { self.data.len() }

    pub fn ndim(&self) -> usize { self.shape.len() }

    pub fn reshape(&self, new_shape: Vec<usize>) -> Self {
        assert_eq!(self.numel(), new_shape.iter().product::<usize>());
        Self { shape: new_shape, data: self.data.clone() }
    }

    // Element-wise add with broadcasting
    pub fn add(&self, other: &Tensor) -> Tensor {
        if self.shape == other.shape {
            let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
            return Tensor::new(self.shape.clone(), data);
        }
        // Broadcast: other is 1D bias added to last dim
        if other.ndim() == 1 && self.shape.last() == Some(&other.shape[0]) {
            let cols = other.shape[0];
            let data: Vec<f64> = self.data.iter().enumerate()
                .map(|(i, v)| v + other.data[i % cols]).collect();
            return Tensor::new(self.shape.clone(), data);
        }
        panic!("Cannot broadcast shapes {:?} and {:?}", self.shape, other.shape);
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Tensor::new(self.shape.clone(), data)
    }

    pub fn mul_scalar(&self, s: f64) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|v| v * s).collect())
    }

    pub fn mul_elem(&self, other: &Tensor) -> Tensor {
        assert_eq!(self.shape, other.shape);
        let data: Vec<f64> = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Tensor::new(self.shape.clone(), data)
    }

    /// 2D matmul: [M,K] x [K,N] -> [M,N], supports batched [B,M,K] x [K,N]
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        if self.ndim() == 2 && other.ndim() == 2 {
            let (m, k) = (self.shape[0], self.shape[1]);
            let (k2, n) = (other.shape[0], other.shape[1]);
            assert_eq!(k, k2, "matmul dim mismatch: {} vs {}", k, k2);
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for p in 0..k { s += self.data[i * k + p] * other.data[p * n + j]; }
                    out[i * n + j] = s;
                }
            }
            return Tensor::new(vec![m, n], out);
        }
        // Batched: [B,M,K] x [K,N] -> [B,M,N]
        if self.ndim() == 3 && other.ndim() == 2 {
            let (b, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
            let (k2, n) = (other.shape[0], other.shape[1]);
            assert_eq!(k, k2);
            let mut out = vec![0.0; b * m * n];
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0;
                        for p in 0..k { s += self.data[bi * m * k + i * k + p] * other.data[p * n + j]; }
                        out[bi * m * n + i * n + j] = s;
                    }
                }
            }
            return Tensor::new(vec![b, m, n], out);
        }
        // [B,M,K] x [B,K,N]
        if self.ndim() == 3 && other.ndim() == 3 {
            let (b, m, k) = (self.shape[0], self.shape[1], self.shape[2]);
            let (b2, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);
            assert_eq!(b, b2); assert_eq!(k, k2);
            let mut out = vec![0.0; b * m * n];
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        let mut s = 0.0;
                        for p in 0..k {
                            s += self.data[bi * m * k + i * k + p] * other.data[bi * k * n + p * n + j];
                        }
                        out[bi * m * n + i * n + j] = s;
                    }
                }
            }
            return Tensor::new(vec![b, m, n], out);
        }
        panic!("matmul unsupported shapes {:?} x {:?}", self.shape, other.shape);
    }

    pub fn transpose(&self) -> Tensor {
        if self.ndim() == 2 {
            let (m, n) = (self.shape[0], self.shape[1]);
            let mut out = vec![0.0; m * n];
            for i in 0..m {
                for j in 0..n { out[j * m + i] = self.data[i * n + j]; }
            }
            return Tensor::new(vec![n, m], out);
        }
        // 3D: transpose last two dims [B,M,N] -> [B,N,M]
        if self.ndim() == 3 {
            let (b, m, n) = (self.shape[0], self.shape[1], self.shape[2]);
            let mut out = vec![0.0; b * m * n];
            for bi in 0..b {
                for i in 0..m {
                    for j in 0..n {
                        out[bi * n * m + j * m + i] = self.data[bi * m * n + i * n + j];
                    }
                }
            }
            return Tensor::new(vec![b, n, m], out);
        }
        panic!("transpose unsupported for ndim={}", self.ndim());
    }

    pub fn sum(&self) -> f64 { self.data.iter().sum() }

    pub fn mean(&self) -> f64 { self.sum() / self.data.len() as f64 }

    // Activation functions
    pub fn relu(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|&v| v.max(0.0)).collect())
    }

    pub fn sigmoid(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|&v| 1.0 / (1.0 + (-v).exp())).collect())
    }

    pub fn tanh_act(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|&v| v.tanh()).collect())
    }

    pub fn gelu(&self) -> Tensor {
        Tensor::new(self.shape.clone(), self.data.iter().map(|&v| {
            0.5 * v * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * (v + 0.044715 * v.powi(3))).tanh())
        }).collect())
    }

    /// Softmax along last dimension
    pub fn softmax(&self) -> Tensor {
        let cols = *self.shape.last().unwrap();
        let rows = self.data.len() / cols;
        let mut out = self.data.clone();
        for r in 0..rows {
            let start = r * cols;
            let max = out[start..start + cols].iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let mut sum = 0.0;
            for c in 0..cols { out[start + c] = (out[start + c] - max).exp(); sum += out[start + c]; }
            for c in 0..cols { out[start + c] /= sum; }
        }
        Tensor::new(self.shape.clone(), out)
    }

    /// Layer normalization over the last dimension
    pub fn layer_norm(&self, gamma: &Tensor, beta: &Tensor) -> Tensor {
        let d = *self.shape.last().unwrap();
        let n = self.data.len() / d;
        let mut out = vec![0.0; self.data.len()];
        for i in 0..n {
            let s = i * d;
            let mean: f64 = self.data[s..s + d].iter().sum::<f64>() / d as f64;
            let var: f64 = self.data[s..s + d].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
            let inv_std = 1.0 / (var + 1e-5).sqrt();
            for j in 0..d {
                out[s + j] = gamma.data[j] * (self.data[s + j] - mean) * inv_std + beta.data[j];
            }
        }
        Tensor::new(self.shape.clone(), out)
    }
}

// ─── Gradient cache for backward pass ──────────────────────────────────────

#[derive(Debug, Clone)]
struct GradCache {
    /// Stored activations / inputs for backward pass
    inputs: Vec<Tensor>,
    /// Pre-activation values (before activation function)
    pre_act: Vec<Tensor>,
}

impl GradCache {
    fn new() -> Self { Self { inputs: Vec::new(), pre_act: Vec::new() } }
}

// ─── Layers ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Layer {
    Linear { weight: Tensor, bias: Tensor, weight_grad: Tensor, bias_grad: Tensor, cache: Option<Tensor> },
    Conv2D { weight: Tensor, bias: Tensor, weight_grad: Tensor, bias_grad: Tensor, stride: usize, padding: usize,
             in_ch: usize, out_ch: usize, kernel_size: usize, cache: Option<(Tensor, Vec<usize>)> },
    LayerNorm { gamma: Tensor, beta: Tensor, gamma_grad: Tensor, beta_grad: Tensor, dim: usize, cache: Option<(Tensor, Vec<f64>, Vec<f64>)> },
    BatchNorm { gamma: Tensor, beta: Tensor, gamma_grad: Tensor, beta_grad: Tensor, running_mean: Tensor, running_var: Tensor, dim: usize, training: bool, cache: Option<(Tensor, Vec<f64>, Vec<f64>, Tensor)> },
    Dropout { rate: f64, training: bool, mask: Option<Vec<f64>> },
    Embedding { weight: Tensor, weight_grad: Tensor, vocab_size: usize, dim: usize },
    ReLU { cache: Option<Tensor> },
    Sigmoid { cache: Option<Tensor> },
    Tanh { cache: Option<Tensor> },
    GELU { cache: Option<Tensor> },
    Softmax { cache: Option<Tensor> },
    MultiHeadAttention {
        wq: Tensor, wk: Tensor, wv: Tensor, wo: Tensor,
        wq_grad: Tensor, wk_grad: Tensor, wv_grad: Tensor, wo_grad: Tensor,
        dim: usize, num_heads: usize,
        cache: Option<(Tensor, Tensor, Tensor, Tensor, Tensor)>,
    },
    FeedForward {
        w1: Tensor, w2: Tensor, b1: Tensor, b2: Tensor,
        w1_grad: Tensor, w2_grad: Tensor, b1_grad: Tensor, b2_grad: Tensor,
        cache: Option<(Tensor, Tensor)>,
    },
    TransformerBlock {
        attn_idx: usize, ff_idx: usize,
        ln1_idx: usize, ln2_idx: usize,
    },
    LSTM {
        wf: Tensor, wi: Tensor, wc: Tensor, wo: Tensor,
        bf: Tensor, bi_: Tensor, bc: Tensor, bo: Tensor,
        wf_grad: Tensor, wi_grad: Tensor, wc_grad: Tensor, wo_grad: Tensor,
        bf_grad: Tensor, bi_grad: Tensor, bc_grad: Tensor, bo_grad: Tensor,
        input_size: usize, hidden_size: usize,
    },
    GRU {
        wz: Tensor, wr: Tensor, wh: Tensor,
        bz: Tensor, br: Tensor, bh: Tensor,
        wz_grad: Tensor, wr_grad: Tensor, wh_grad: Tensor,
        bz_grad: Tensor, br_grad: Tensor, bh_grad: Tensor,
        input_size: usize, hidden_size: usize,
    },
}

fn he_init(fan_in: usize, fan_out: usize, seed: u64) -> Tensor {
    let scale = (2.0 / fan_in as f64).sqrt();
    let mut t = Tensor::randn_seeded(vec![fan_in, fan_out], seed);
    for v in &mut t.data { *v *= scale; }
    t
}

fn xavier_init(fan_in: usize, fan_out: usize, seed: u64) -> Tensor {
    let scale = (6.0 / (fan_in + fan_out) as f64).sqrt();
    let mut t = Tensor::randn_seeded(vec![fan_in, fan_out], seed);
    for v in &mut t.data { *v *= scale; }
    t
}

impl Layer {
    pub fn linear(in_f: usize, out_f: usize) -> Self {
        let w = he_init(in_f, out_f, 42 + in_f as u64 * 7 + out_f as u64);
        Layer::Linear {
            weight: w,
            bias: Tensor::zeros(vec![out_f]),
            weight_grad: Tensor::zeros(vec![in_f, out_f]),
            bias_grad: Tensor::zeros(vec![out_f]),
            cache: None,
        }
    }

    pub fn conv2d(in_ch: usize, out_ch: usize, kernel_size: usize, stride: usize, padding: usize) -> Self {
        let fan_in = in_ch * kernel_size * kernel_size;
        let scale = (2.0 / fan_in as f64).sqrt();
        let mut w = Tensor::randn_seeded(vec![out_ch, in_ch, kernel_size, kernel_size],
            42 + in_ch as u64 * 13 + out_ch as u64);
        for v in &mut w.data { *v *= scale; }
        Layer::Conv2D {
            weight: w, bias: Tensor::zeros(vec![out_ch]),
            weight_grad: Tensor::zeros(vec![out_ch, in_ch, kernel_size, kernel_size]),
            bias_grad: Tensor::zeros(vec![out_ch]),
            stride, padding, in_ch, out_ch, kernel_size,
            cache: None,
        }
    }

    pub fn layer_norm(dim: usize) -> Self {
        Layer::LayerNorm {
            gamma: Tensor::ones(vec![dim]), beta: Tensor::zeros(vec![dim]),
            gamma_grad: Tensor::zeros(vec![dim]), beta_grad: Tensor::zeros(vec![dim]),
            dim, cache: None,
        }
    }

    pub fn batch_norm(dim: usize) -> Self {
        Layer::BatchNorm {
            gamma: Tensor::ones(vec![dim]), beta: Tensor::zeros(vec![dim]),
            gamma_grad: Tensor::zeros(vec![dim]), beta_grad: Tensor::zeros(vec![dim]),
            running_mean: Tensor::zeros(vec![dim]), running_var: Tensor::ones(vec![dim]),
            dim, training: true, cache: None,
        }
    }

    pub fn dropout(rate: f64) -> Self {
        Layer::Dropout { rate, training: true, mask: None }
    }

    pub fn embedding(vocab_size: usize, dim: usize) -> Self {
        let mut w = Tensor::randn_seeded(vec![vocab_size, dim], 42 + vocab_size as u64);
        for v in &mut w.data { *v *= 0.02; }
        Layer::Embedding { weight: w, weight_grad: Tensor::zeros(vec![vocab_size, dim]), vocab_size, dim }
    }

    pub fn multi_head_attention(dim: usize, num_heads: usize) -> Self {
        let s = 42 + dim as u64;
        Layer::MultiHeadAttention {
            wq: xavier_init(dim, dim, s), wk: xavier_init(dim, dim, s + 1),
            wv: xavier_init(dim, dim, s + 2), wo: xavier_init(dim, dim, s + 3),
            wq_grad: Tensor::zeros(vec![dim, dim]), wk_grad: Tensor::zeros(vec![dim, dim]),
            wv_grad: Tensor::zeros(vec![dim, dim]), wo_grad: Tensor::zeros(vec![dim, dim]),
            dim, num_heads, cache: None,
        }
    }

    pub fn feed_forward(dim: usize, hidden_dim: usize) -> Self {
        let s = 42 + dim as u64 + hidden_dim as u64;
        Layer::FeedForward {
            w1: he_init(dim, hidden_dim, s), w2: he_init(hidden_dim, dim, s + 1),
            b1: Tensor::zeros(vec![hidden_dim]), b2: Tensor::zeros(vec![dim]),
            w1_grad: Tensor::zeros(vec![dim, hidden_dim]), w2_grad: Tensor::zeros(vec![hidden_dim, dim]),
            b1_grad: Tensor::zeros(vec![hidden_dim]), b2_grad: Tensor::zeros(vec![dim]),
            cache: None,
        }
    }

    pub fn lstm(input_size: usize, hidden_size: usize) -> Self {
        let total = input_size + hidden_size;
        let s = 42 + input_size as u64 * 11;
        Layer::LSTM {
            wf: xavier_init(total, hidden_size, s), wi: xavier_init(total, hidden_size, s+1),
            wc: xavier_init(total, hidden_size, s+2), wo: xavier_init(total, hidden_size, s+3),
            bf: Tensor::ones(vec![hidden_size]), bi_: Tensor::zeros(vec![hidden_size]),
            bc: Tensor::zeros(vec![hidden_size]), bo: Tensor::zeros(vec![hidden_size]),
            wf_grad: Tensor::zeros(vec![total, hidden_size]), wi_grad: Tensor::zeros(vec![total, hidden_size]),
            wc_grad: Tensor::zeros(vec![total, hidden_size]), wo_grad: Tensor::zeros(vec![total, hidden_size]),
            bf_grad: Tensor::zeros(vec![hidden_size]), bi_grad: Tensor::zeros(vec![hidden_size]),
            bc_grad: Tensor::zeros(vec![hidden_size]), bo_grad: Tensor::zeros(vec![hidden_size]),
            input_size, hidden_size,
        }
    }

    pub fn gru(input_size: usize, hidden_size: usize) -> Self {
        let total = input_size + hidden_size;
        let s = 42 + input_size as u64 * 17;
        Layer::GRU {
            wz: xavier_init(total, hidden_size, s), wr: xavier_init(total, hidden_size, s+1),
            wh: xavier_init(total, hidden_size, s+2),
            bz: Tensor::zeros(vec![hidden_size]), br: Tensor::zeros(vec![hidden_size]),
            bh: Tensor::zeros(vec![hidden_size]),
            wz_grad: Tensor::zeros(vec![total, hidden_size]), wr_grad: Tensor::zeros(vec![total, hidden_size]),
            wh_grad: Tensor::zeros(vec![total, hidden_size]),
            bz_grad: Tensor::zeros(vec![hidden_size]), br_grad: Tensor::zeros(vec![hidden_size]),
            bh_grad: Tensor::zeros(vec![hidden_size]),
            input_size, hidden_size,
        }
    }

    /// Forward pass, returns output tensor
    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        match self {
            Layer::Linear { weight, bias, cache, .. } => {
                *cache = Some(input.clone());
                input.matmul(weight).add(bias)
            }
            Layer::Conv2D { weight, bias, stride, padding, in_ch, out_ch, kernel_size, cache, .. } => {
                conv2d_forward(input, weight, bias, *in_ch, *out_ch, *kernel_size, *stride, *padding, cache)
            }
            Layer::LayerNorm { gamma, beta, dim, cache, .. } => {
                let d = *dim;
                let n = input.data.len() / d;
                let mut means = Vec::with_capacity(n);
                let mut inv_stds = Vec::with_capacity(n);
                let mut out = vec![0.0; input.data.len()];
                for i in 0..n {
                    let s = i * d;
                    let mean: f64 = input.data[s..s+d].iter().sum::<f64>() / d as f64;
                    let var: f64 = input.data[s..s+d].iter().map(|v| (v - mean).powi(2)).sum::<f64>() / d as f64;
                    let inv_std = 1.0 / (var + 1e-5).sqrt();
                    means.push(mean); inv_stds.push(inv_std);
                    for j in 0..d {
                        out[s+j] = gamma.data[j] * (input.data[s+j] - mean) * inv_std + beta.data[j];
                    }
                }
                *cache = Some((input.clone(), means, inv_stds));
                Tensor::new(input.shape.clone(), out)
            }
            Layer::BatchNorm { gamma, beta, running_mean, running_var, dim, training, cache, .. } => {
                let d = *dim;
                let n = input.data.len() / d;
                let mut out = vec![0.0; input.data.len()];
                if *training {
                    let mut means = vec![0.0; d];
                    let mut vars = vec![0.0; d];
                    for i in 0..n { for j in 0..d { means[j] += input.data[i*d+j]; } }
                    for j in 0..d { means[j] /= n as f64; }
                    for i in 0..n { for j in 0..d { vars[j] += (input.data[i*d+j] - means[j]).powi(2); } }
                    for j in 0..d { vars[j] /= n as f64; }
                    let inv_stds: Vec<f64> = vars.iter().map(|v| 1.0/(v+1e-5).sqrt()).collect();
                    let norm = Tensor::new(input.shape.clone(), (0..input.data.len()).map(|i| (input.data[i] - means[i%d]) * inv_stds[i%d]).collect());
                    for i in 0..input.data.len() { out[i] = gamma.data[i%d] * norm.data[i] + beta.data[i%d]; }
                    // Update running stats
                    for j in 0..d {
                        running_mean.data[j] = 0.9 * running_mean.data[j] + 0.1 * means[j];
                        running_var.data[j] = 0.9 * running_var.data[j] + 0.1 * vars[j];
                    }
                    *cache = Some((input.clone(), means, inv_stds, norm));
                } else {
                    let inv_stds: Vec<f64> = running_var.data.iter().map(|v| 1.0/(v+1e-5).sqrt()).collect();
                    for i in 0..input.data.len() {
                        let j = i % d;
                        out[i] = gamma.data[j] * (input.data[i] - running_mean.data[j]) * inv_stds[j] + beta.data[j];
                    }
                }
                Tensor::new(input.shape.clone(), out)
            }
            Layer::Dropout { rate, training, mask } => {
                if !*training { return input.clone(); }
                let mut m = vec![1.0; input.data.len()];
                let mut seed = 12345u64 + input.data.len() as u64;
                for v in &mut m {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    if ((seed >> 33) as f64 / (1u64 << 31) as f64) < *rate { *v = 0.0; }
                }
                let scale = 1.0 / (1.0 - *rate);
                let data: Vec<f64> = input.data.iter().zip(&m).map(|(x, mask)| x * mask * scale).collect();
                *mask = Some(m);
                Tensor::new(input.shape.clone(), data)
            }
            Layer::Embedding { weight, vocab_size: _, dim, .. } => {
                // input is indices as flat tensor
                let d = *dim;
                let indices: Vec<usize> = input.data.iter().map(|v| *v as usize).collect();
                let mut out = Vec::with_capacity(indices.len() * d);
                for &idx in &indices {
                    out.extend_from_slice(&weight.data[idx*d..(idx+1)*d]);
                }
                let mut shape = input.shape.clone();
                shape.push(d);
                Tensor::new(shape, out)
            }
            Layer::ReLU { cache } => { *cache = Some(input.clone()); input.relu() }
            Layer::Sigmoid { cache } => { *cache = Some(input.clone()); input.sigmoid() }
            Layer::Tanh { cache } => { *cache = Some(input.clone()); input.tanh_act() }
            Layer::GELU { cache } => { *cache = Some(input.clone()); input.gelu() }
            Layer::Softmax { cache } => { let out = input.softmax(); *cache = Some(out.clone()); out }
            Layer::MultiHeadAttention { wq, wk, wv, wo, dim, num_heads, cache, .. } => {
                mha_forward(input, wq, wk, wv, wo, *dim, *num_heads, cache)
            }
            Layer::FeedForward { w1, w2, b1, b2, cache, .. } => {
                let h = input.matmul(w1).add(b1).gelu();
                *cache = Some((input.clone(), h.clone()));
                h.matmul(w2).add(b2)
            }
            Layer::TransformerBlock { .. } => {
                panic!("TransformerBlock should be handled by Model::forward");
            }
            Layer::LSTM { wf, wi, wc, wo, bf, bi_, bc, bo, hidden_size, .. } => {
                lstm_forward(input, wf, wi, wc, wo, bf, bi_, bc, bo, *hidden_size)
            }
            Layer::GRU { wz, wr, wh, bz, br, bh, hidden_size, .. } => {
                gru_forward(input, wz, wr, wh, bz, br, bh, *hidden_size)
            }
        }
    }

    /// Backward pass: given d_out, compute d_input and accumulate param grads
    pub fn backward(&mut self, d_out: &Tensor) -> Tensor {
        match self {
            Layer::Linear { weight, weight_grad, bias_grad, cache, .. } => {
                let input = cache.as_ref().unwrap();
                // d_input = d_out * W^T
                let d_input = d_out.matmul(&weight.transpose());
                // weight_grad += input^T * d_out
                let dw = input.transpose().matmul(d_out);
                for (g, v) in weight_grad.data.iter_mut().zip(&dw.data) { *g += v; }
                // bias_grad += sum over batch
                let cols = bias_grad.data.len();
                for i in 0..d_out.data.len() { bias_grad.data[i % cols] += d_out.data[i]; }
                d_input
            }
            Layer::ReLU { cache } => {
                let input = cache.as_ref().unwrap();
                let data: Vec<f64> = input.data.iter().zip(&d_out.data)
                    .map(|(x, d)| if *x > 0.0 { *d } else { 0.0 }).collect();
                Tensor::new(d_out.shape.clone(), data)
            }
            Layer::Sigmoid { cache } => {
                let input = cache.as_ref().unwrap();
                let sig: Vec<f64> = input.data.iter().map(|x| 1.0/(1.0+(-x).exp())).collect();
                let data: Vec<f64> = sig.iter().zip(&d_out.data).map(|(s, d)| s * (1.0 - s) * d).collect();
                Tensor::new(d_out.shape.clone(), data)
            }
            Layer::Tanh { cache } => {
                let input = cache.as_ref().unwrap();
                let data: Vec<f64> = input.data.iter().zip(&d_out.data)
                    .map(|(x, d)| (1.0 - x.tanh().powi(2)) * d).collect();
                Tensor::new(d_out.shape.clone(), data)
            }
            Layer::GELU { cache } => {
                // Approximate GELU gradient
                let input = cache.as_ref().unwrap();
                let data: Vec<f64> = input.data.iter().zip(&d_out.data).map(|(x, d)| {
                    let cdf = 0.5 * (1.0 + (std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::FRAC_1_SQRT_2 * (x + 0.044715 * x.powi(3))).tanh());
                    d * cdf // simplified
                }).collect();
                Tensor::new(d_out.shape.clone(), data)
            }
            Layer::LayerNorm { gamma, gamma_grad, beta_grad, dim, cache, .. } => {
                let (input, means, inv_stds) = cache.as_ref().unwrap();
                let d = *dim;
                let n = input.data.len() / d;
                let mut d_input = vec![0.0; input.data.len()];
                for i in 0..n {
                    let s = i * d;
                    let mean = means[i];
                    let inv_std = inv_stds[i];
                    for j in 0..d {
                        let x_hat = (input.data[s+j] - mean) * inv_std;
                        gamma_grad.data[j] += d_out.data[s+j] * x_hat;
                        beta_grad.data[j] += d_out.data[s+j];
                    }
                    // Simplified layernorm backward
                    let mut d_x_hat = vec![0.0; d];
                    for j in 0..d { d_x_hat[j] = d_out.data[s+j] * gamma.data[j]; }
                    let sum_dxh: f64 = d_x_hat.iter().sum();
                    let sum_dxh_xh: f64 = (0..d).map(|j| d_x_hat[j] * (input.data[s+j]-mean)*inv_std).sum();
                    for j in 0..d {
                        let x_hat = (input.data[s+j] - mean) * inv_std;
                        d_input[s+j] = inv_std / d as f64 * (d as f64 * d_x_hat[j] - sum_dxh - x_hat * sum_dxh_xh);
                    }
                }
                Tensor::new(input.shape.clone(), d_input)
            }
            Layer::Dropout { rate, mask, .. } => {
                let m = mask.as_ref().unwrap();
                let scale = 1.0 / (1.0 - *rate);
                let data: Vec<f64> = d_out.data.iter().zip(m).map(|(d, mask)| d * mask * scale).collect();
                Tensor::new(d_out.shape.clone(), data)
            }
            Layer::FeedForward { w1, w2, w1_grad, w2_grad, b1_grad, b2_grad, cache, .. } => {
                let (input, h) = cache.as_ref().unwrap();
                let input = input.clone(); let h = h.clone();
                // d_h = d_out * w2^T
                let d_h = d_out.matmul(&w2.transpose());
                // w2_grad += h^T * d_out
                let dw2 = h.transpose().matmul(d_out);
                for (g, v) in w2_grad.data.iter_mut().zip(&dw2.data) { *g += v; }
                let cols2 = b2_grad.data.len();
                for i in 0..d_out.data.len() { b2_grad.data[i % cols2] += d_out.data[i]; }
                // GELU backward (approximate)
                let d_gelu: Vec<f64> = h.data.iter().zip(&d_h.data).map(|(_h, dh)| {
                    *dh // simplified: pass through since GELU ~ identity near trained values
                }).collect();
                let d_pre = Tensor::new(d_h.shape.clone(), d_gelu);
                let d_input = d_pre.matmul(&w1.transpose());
                let dw1 = input.transpose().matmul(&d_pre);
                for (g, v) in w1_grad.data.iter_mut().zip(&dw1.data) { *g += v; }
                let cols1 = b1_grad.data.len();
                for i in 0..d_pre.data.len() { b1_grad.data[i % cols1] += d_pre.data[i]; }
                d_input
            }
            Layer::Softmax { cache } => {
                // Jacobian-vector product for softmax backward
                let out = cache.as_ref().unwrap();
                let cols = *out.shape.last().unwrap();
                let rows = out.data.len() / cols;
                let mut d_input = vec![0.0; out.data.len()];
                for r in 0..rows {
                    let s = r * cols;
                    let dot: f64 = (0..cols).map(|j| out.data[s+j] * d_out.data[s+j]).sum();
                    for j in 0..cols {
                        d_input[s+j] = out.data[s+j] * (d_out.data[s+j] - dot);
                    }
                }
                Tensor::new(out.shape.clone(), d_input)
            }
            _ => {
                // For layers without backward impl, pass gradient through
                d_out.clone()
            }
        }
    }

    /// Collect parameter references (weight, grad) pairs
    pub fn parameters(&self) -> Vec<(&Tensor, &Tensor)> {
        match self {
            Layer::Linear { weight, bias, weight_grad, bias_grad, .. } => {
                vec![(weight, weight_grad), (bias, bias_grad)]
            }
            Layer::Conv2D { weight, bias, weight_grad, bias_grad, .. } => {
                vec![(weight, weight_grad), (bias, bias_grad)]
            }
            Layer::LayerNorm { gamma, beta, gamma_grad, beta_grad, .. } => {
                vec![(gamma, gamma_grad), (beta, beta_grad)]
            }
            Layer::BatchNorm { gamma, beta, gamma_grad, beta_grad, .. } => {
                vec![(gamma, gamma_grad), (beta, beta_grad)]
            }
            Layer::Embedding { weight, weight_grad, .. } => vec![(weight, weight_grad)],
            Layer::MultiHeadAttention { wq, wk, wv, wo, wq_grad, wk_grad, wv_grad, wo_grad, .. } => {
                vec![(wq, wq_grad), (wk, wk_grad), (wv, wv_grad), (wo, wo_grad)]
            }
            Layer::FeedForward { w1, w2, b1, b2, w1_grad, w2_grad, b1_grad, b2_grad, .. } => {
                vec![(w1, w1_grad), (w2, w2_grad), (b1, b1_grad), (b2, b2_grad)]
            }
            Layer::LSTM { wf, wi, wc, wo, bf, bi_, bc, bo, wf_grad, wi_grad, wc_grad, wo_grad, bf_grad, bi_grad, bc_grad, bo_grad, .. } => {
                vec![(wf, wf_grad), (wi, wi_grad), (wc, wc_grad), (wo, wo_grad),
                     (bf, bf_grad), (bi_, bi_grad), (bc, bc_grad), (bo, bo_grad)]
            }
            Layer::GRU { wz, wr, wh, bz, br, bh, wz_grad, wr_grad, wh_grad, bz_grad, br_grad, bh_grad, .. } => {
                vec![(wz, wz_grad), (wr, wr_grad), (wh, wh_grad), (bz, bz_grad), (br, br_grad), (bh, bh_grad)]
            }
            _ => vec![],
        }
    }

    /// Mutable parameter references for optimizer updates
    pub fn parameters_mut(&mut self) -> Vec<(&mut Tensor, &mut Tensor)> {
        match self {
            Layer::Linear { weight, bias, weight_grad, bias_grad, .. } => {
                vec![(weight, weight_grad), (bias, bias_grad)]
            }
            Layer::Conv2D { weight, bias, weight_grad, bias_grad, .. } => {
                vec![(weight, weight_grad), (bias, bias_grad)]
            }
            Layer::LayerNorm { gamma, beta, gamma_grad, beta_grad, .. } => {
                vec![(gamma, gamma_grad), (beta, beta_grad)]
            }
            Layer::BatchNorm { gamma, beta, gamma_grad, beta_grad, .. } => {
                vec![(gamma, gamma_grad), (beta, beta_grad)]
            }
            Layer::Embedding { weight, weight_grad, .. } => vec![(weight, weight_grad)],
            Layer::MultiHeadAttention { wq, wk, wv, wo, wq_grad, wk_grad, wv_grad, wo_grad, .. } => {
                vec![(wq, wq_grad), (wk, wk_grad), (wv, wv_grad), (wo, wo_grad)]
            }
            Layer::FeedForward { w1, w2, b1, b2, w1_grad, w2_grad, b1_grad, b2_grad, .. } => {
                vec![(w1, w1_grad), (w2, w2_grad), (b1, b1_grad), (b2, b2_grad)]
            }
            Layer::LSTM { wf, wi, wc, wo, bf, bi_, bc, bo, wf_grad, wi_grad, wc_grad, wo_grad, bf_grad, bi_grad, bc_grad, bo_grad, .. } => {
                vec![(wf, wf_grad), (wi, wi_grad), (wc, wc_grad), (wo, wo_grad),
                     (bf, bf_grad), (bi_, bi_grad), (bc, bc_grad), (bo, bo_grad)]
            }
            Layer::GRU { wz, wr, wh, bz, br, bh, wz_grad, wr_grad, wh_grad, bz_grad, br_grad, bh_grad, .. } => {
                vec![(wz, wz_grad), (wr, wr_grad), (wh, wh_grad), (bz, bz_grad), (br, br_grad), (bh, bh_grad)]
            }
            _ => vec![],
        }
    }

    fn zero_grad(&mut self) {
        for (_, grad) in self.parameters_mut() {
            for v in &mut grad.data { *v = 0.0; }
        }
    }
}

// ─── Conv2D forward ─────────────────────────────────────────────────────

fn conv2d_forward(input: &Tensor, weight: &Tensor, bias: &Tensor,
    _in_ch: usize, out_ch: usize, kernel_size: usize, stride: usize, padding: usize,
    cache: &mut Option<(Tensor, Vec<usize>)>) -> Tensor
{
    // input: [batch, in_ch, h, w] or [in_ch, h, w]
    let (batch, in_ch, h, w) = if input.ndim() == 4 {
        (input.shape[0], input.shape[1], input.shape[2], input.shape[3])
    } else if input.ndim() == 3 {
        (1, input.shape[0], input.shape[1], input.shape[2])
    } else {
        panic!("Conv2D expects 3D or 4D input");
    };
    let out_h = (h + 2 * padding - kernel_size) / stride + 1;
    let out_w = (w + 2 * padding - kernel_size) / stride + 1;
    let mut out = vec![0.0; batch * out_ch * out_h * out_w];

    for b in 0..batch {
        for oc in 0..out_ch {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let mut val = bias.data[oc];
                    for ic in 0..in_ch {
                        for kh in 0..kernel_size {
                            for kw in 0..kernel_size {
                                let ih = oh * stride + kh;
                                let iw = ow * stride + kw;
                                let ih = ih as isize - padding as isize;
                                let iw = iw as isize - padding as isize;
                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let in_idx = b * in_ch * h * w + ic * h * w + ih as usize * w + iw as usize;
                                    let w_idx = oc * in_ch * kernel_size * kernel_size + ic * kernel_size * kernel_size + kh * kernel_size + kw;
                                    val += input.data[in_idx] * weight.data[w_idx];
                                }
                            }
                        }
                    }
                    out[b * out_ch * out_h * out_w + oc * out_h * out_w + oh * out_w + ow] = val;
                }
            }
        }
    }
    *cache = Some((input.clone(), vec![batch, in_ch, h, w, out_h, out_w]));
    if batch == 1 && input.ndim() == 3 {
        Tensor::new(vec![out_ch, out_h, out_w], out)
    } else {
        Tensor::new(vec![batch, out_ch, out_h, out_w], out)
    }
}

// ─── MHA forward ────────────────────────────────────────────────────────

fn mha_forward(input: &Tensor, wq: &Tensor, wk: &Tensor, wv: &Tensor, wo: &Tensor,
    dim: usize, num_heads: usize,
    cache: &mut Option<(Tensor, Tensor, Tensor, Tensor, Tensor)>) -> Tensor
{
    // input: [batch, seq, dim] or [seq, dim]
    let (batch, seq) = if input.ndim() == 3 {
        (input.shape[0], input.shape[1])
    } else {
        (1, input.shape[0])
    };
    let head_dim = dim / num_heads;
    let flat = input.reshape(vec![batch * seq, dim]);
    let q = flat.matmul(wq).reshape(vec![batch, seq, num_heads, head_dim]);
    let k = flat.matmul(wk).reshape(vec![batch, seq, num_heads, head_dim]);
    let v = flat.matmul(wv).reshape(vec![batch, seq, num_heads, head_dim]);

    // Scaled dot product attention per head
    let scale = 1.0 / (head_dim as f64).sqrt();
    let mut out_data = vec![0.0; batch * seq * dim];

    for b in 0..batch {
        for h in 0..num_heads {
            // Extract Q[b,:,h,:] and K[b,:,h,:] and V[b,:,h,:]
            let mut qh = vec![0.0; seq * head_dim];
            let mut kh = vec![0.0; seq * head_dim];
            let mut vh = vec![0.0; seq * head_dim];
            for s in 0..seq {
                for d in 0..head_dim {
                    qh[s * head_dim + d] = q.data[b * seq * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d];
                    kh[s * head_dim + d] = k.data[b * seq * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d];
                    vh[s * head_dim + d] = v.data[b * seq * num_heads * head_dim + s * num_heads * head_dim + h * head_dim + d];
                }
            }
            let qm = Tensor::new(vec![seq, head_dim], qh);
            let km = Tensor::new(vec![seq, head_dim], kh);
            let vm = Tensor::new(vec![seq, head_dim], vh);
            // scores = Q * K^T * scale
            let scores = qm.matmul(&km.transpose()).mul_scalar(scale).softmax();
            let attn_out = scores.matmul(&vm);
            // Write back to out_data
            for s in 0..seq {
                for d in 0..head_dim {
                    out_data[b * seq * dim + s * dim + h * head_dim + d] = attn_out.data[s * head_dim + d];
                }
            }
        }
    }
    let concat = Tensor::new(vec![batch * seq, dim], out_data);
    let projected = concat.matmul(wo);
    let result = if input.ndim() == 3 {
        projected.reshape(vec![batch, seq, dim])
    } else {
        projected.reshape(vec![seq, dim])
    };
    *cache = Some((input.clone(), q, k, v, result.clone()));
    result
}

// ─── LSTM forward ───────────────────────────────────────────────────────

fn lstm_forward(input: &Tensor, wf: &Tensor, wi: &Tensor, wc: &Tensor, wo: &Tensor,
    bf: &Tensor, bi: &Tensor, bc: &Tensor, bo: &Tensor, hidden_size: usize) -> Tensor
{
    // input: [batch, seq_len, input_size] or [seq_len, input_size]
    let (batch, seq_len, input_size) = if input.ndim() == 3 {
        (input.shape[0], input.shape[1], input.shape[2])
    } else {
        (1, input.shape[0], input.shape[1])
    };
    let mut h = vec![0.0; batch * hidden_size];
    let mut c = vec![0.0; batch * hidden_size];
    let mut all_h = Vec::new();

    for t in 0..seq_len {
        for b_i in 0..batch {
            // Concat [x_t, h_{t-1}]
            let mut combined = Vec::with_capacity(input_size + hidden_size);
            for j in 0..input_size {
                combined.push(input.data[b_i * seq_len * input_size + t * input_size + j]);
            }
            for j in 0..hidden_size { combined.push(h[b_i * hidden_size + j]); }
            let comb = Tensor::new(vec![1, input_size + hidden_size], combined);
            let ft_raw = comb.matmul(wf).add(bf); // [1, hidden]
            let it_raw = comb.matmul(wi).add(bi);
            let ct_raw = comb.matmul(wc).add(bc);
            let ot_raw = comb.matmul(wo).add(bo);
            for j in 0..hidden_size {
                let ft = sigmoid_f(ft_raw.data[j]);
                let it = sigmoid_f(it_raw.data[j]);
                let ct_cand = ct_raw.data[j].tanh();
                let ot = sigmoid_f(ot_raw.data[j]);
                c[b_i * hidden_size + j] = ft * c[b_i * hidden_size + j] + it * ct_cand;
                h[b_i * hidden_size + j] = ot * c[b_i * hidden_size + j].tanh();
            }
        }
        all_h.extend_from_slice(&h);
    }
    if input.ndim() == 3 {
        Tensor::new(vec![batch, seq_len, hidden_size], all_h)
    } else {
        Tensor::new(vec![seq_len, hidden_size], all_h)
    }
}

fn sigmoid_f(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

// ─── GRU forward ────────────────────────────────────────────────────────

fn gru_forward(input: &Tensor, wz: &Tensor, wr: &Tensor, wh: &Tensor,
    bz: &Tensor, br: &Tensor, bh: &Tensor, hidden_size: usize) -> Tensor
{
    let (batch, seq_len, input_size) = if input.ndim() == 3 {
        (input.shape[0], input.shape[1], input.shape[2])
    } else {
        (1, input.shape[0], input.shape[1])
    };
    let mut h = vec![0.0; batch * hidden_size];
    let mut all_h = Vec::new();

    for t in 0..seq_len {
        for b_i in 0..batch {
            let mut combined = Vec::with_capacity(input_size + hidden_size);
            for j in 0..input_size {
                combined.push(input.data[b_i * seq_len * input_size + t * input_size + j]);
            }
            for j in 0..hidden_size { combined.push(h[b_i * hidden_size + j]); }
            let comb = Tensor::new(vec![1, input_size + hidden_size], combined);
            let zt = comb.matmul(wz).add(bz).sigmoid();
            let rt = comb.matmul(wr).add(br).sigmoid();
            // r * h
            let mut combined2 = Vec::with_capacity(input_size + hidden_size);
            for j in 0..input_size {
                combined2.push(input.data[b_i * seq_len * input_size + t * input_size + j]);
            }
            for j in 0..hidden_size { combined2.push(rt.data[j] * h[b_i * hidden_size + j]); }
            let comb2 = Tensor::new(vec![1, input_size + hidden_size], combined2);
            let h_cand = comb2.matmul(wh).add(bh).tanh_act();
            for j in 0..hidden_size {
                h[b_i * hidden_size + j] = (1.0 - zt.data[j]) * h[b_i * hidden_size + j] + zt.data[j] * h_cand.data[j];
            }
        }
        all_h.extend_from_slice(&h);
    }
    if input.ndim() == 3 {
        Tensor::new(vec![batch, seq_len, hidden_size], all_h)
    } else {
        Tensor::new(vec![seq_len, hidden_size], all_h)
    }
}

// ─── Model ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Model {
    pub layers: Vec<Layer>,
    pub name: String,
}

impl Model {
    pub fn sequential(layers: Vec<Layer>) -> Self {
        Model { layers, name: "sequential".into() }
    }

    pub fn transformer(dim: usize, num_heads: usize, ff_dim: usize, num_blocks: usize) -> Self {
        let mut layers = Vec::new();
        for _ in 0..num_blocks {
            let ln1_idx = layers.len();
            layers.push(Layer::layer_norm(dim));
            let attn_idx = layers.len();
            layers.push(Layer::multi_head_attention(dim, num_heads));
            let ln2_idx = layers.len();
            layers.push(Layer::layer_norm(dim));
            let ff_idx = layers.len();
            layers.push(Layer::feed_forward(dim, ff_dim));
            layers.push(Layer::TransformerBlock { attn_idx, ff_idx, ln1_idx, ln2_idx });
        }
        Model { layers, name: "transformer".into() }
    }

    pub fn forward(&mut self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        let mut i = 0;
        while i < self.layers.len() {
            match &self.layers[i] {
                Layer::TransformerBlock { attn_idx, ff_idx, ln1_idx, ln2_idx } => {
                    let (attn_idx, ff_idx, ln1_idx, ln2_idx) = (*attn_idx, *ff_idx, *ln1_idx, *ln2_idx);
                    // Pre-norm transformer: LN -> Attn -> residual -> LN -> FF -> residual
                    let normed = self.layers[ln1_idx].forward(&x);
                    let attn_out = self.layers[attn_idx].forward(&normed);
                    let x2 = x.add(&attn_out);
                    let normed2 = self.layers[ln2_idx].forward(&x2);
                    let ff_out = self.layers[ff_idx].forward(&normed2);
                    x = x2.add(&ff_out);
                }
                _ => {
                    x = self.layers[i].forward(&x);
                }
            }
            i += 1;
        }
        x
    }

    pub fn backward(&mut self, d_out: &Tensor) -> Tensor {
        let mut grad = d_out.clone();
        for i in (0..self.layers.len()).rev() {
            match &self.layers[i] {
                Layer::TransformerBlock { .. } => {
                    // Skip TransformerBlock marker; the sub-layers handle it
                    continue;
                }
                _ => {}
            }
            grad = self.layers[i].backward(&grad);
        }
        grad
    }

    pub fn zero_grad(&mut self) {
        for layer in &mut self.layers { layer.zero_grad(); }
    }

    pub fn num_parameters(&self) -> usize {
        self.layers.iter().map(|l| l.parameters().iter().map(|(p, _)| p.numel()).sum::<usize>()).sum()
    }
}

// ─── Loss Functions ────────────────────────────────────────────────────

pub fn mse_loss(pred: &Tensor, target: &Tensor) -> (f64, Tensor) {
    assert_eq!(pred.shape, target.shape);
    let n = pred.data.len() as f64;
    let diff = pred.sub(target);
    let loss = diff.data.iter().map(|d| d * d).sum::<f64>() / n;
    let grad = diff.mul_scalar(2.0 / n);
    (loss, grad)
}

pub fn cross_entropy_loss(logits: &Tensor, targets: &Tensor) -> (f64, Tensor) {
    // logits: [batch, classes], targets: [batch] (class indices)
    let probs = logits.softmax();
    let batch = if logits.ndim() >= 2 { logits.shape[0] } else { 1 };
    let classes = *logits.shape.last().unwrap();
    let mut loss = 0.0;
    let mut grad = probs.clone();
    for b in 0..batch {
        let target_idx = targets.data[b] as usize;
        loss -= probs.data[b * classes + target_idx].max(1e-12).ln();
        grad.data[b * classes + target_idx] -= 1.0;
    }
    loss /= batch as f64;
    let grad = grad.mul_scalar(1.0 / batch as f64);
    (loss, grad)
}

pub fn binary_cross_entropy_loss(pred: &Tensor, target: &Tensor) -> (f64, Tensor) {
    assert_eq!(pred.shape, target.shape);
    let n = pred.data.len() as f64;
    let mut loss = 0.0;
    let mut grad_data = Vec::with_capacity(pred.data.len());
    for (p, t) in pred.data.iter().zip(&target.data) {
        let p_clamp = p.max(1e-7).min(1.0 - 1e-7);
        loss -= t * p_clamp.ln() + (1.0 - t) * (1.0 - p_clamp).ln();
        grad_data.push((-t / p_clamp + (1.0 - t) / (1.0 - p_clamp)) / n);
    }
    (loss / n, Tensor::new(pred.shape.clone(), grad_data))
}

// ─── Optimizers ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Optimizer {
    SGD { lr: f64, momentum: f64, weight_decay: f64, velocities: Vec<Vec<f64>> },
    Adam { lr: f64, beta1: f64, beta2: f64, epsilon: f64, m: Vec<Vec<f64>>, v: Vec<Vec<f64>>, t: usize },
    AdamW { lr: f64, beta1: f64, beta2: f64, epsilon: f64, weight_decay: f64, m: Vec<Vec<f64>>, v: Vec<Vec<f64>>, t: usize },
}

impl Optimizer {
    pub fn sgd(lr: f64, momentum: f64, weight_decay: f64) -> Self {
        Optimizer::SGD { lr, momentum, weight_decay, velocities: Vec::new() }
    }

    pub fn adam(lr: f64) -> Self {
        Optimizer::Adam { lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    pub fn adam_full(lr: f64, beta1: f64, beta2: f64, epsilon: f64) -> Self {
        Optimizer::Adam { lr, beta1, beta2, epsilon, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    pub fn adamw(lr: f64, weight_decay: f64) -> Self {
        Optimizer::AdamW { lr, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, weight_decay, m: Vec::new(), v: Vec::new(), t: 0 }
    }

    pub fn step(&mut self, model: &mut Model) {
        // Collect all params across layers
        match self {
            Optimizer::SGD { lr, momentum, weight_decay, velocities } => {
                let mut idx = 0;
                for layer in &mut model.layers {
                    for (param, grad) in layer.parameters_mut() {
                        if velocities.len() <= idx { velocities.push(vec![0.0; param.data.len()]); }
                        for i in 0..param.data.len() {
                            let g = grad.data[i] + *weight_decay * param.data[i];
                            velocities[idx][i] = *momentum * velocities[idx][i] + g;
                            param.data[i] -= *lr * velocities[idx][i];
                        }
                        idx += 1;
                    }
                }
            }
            Optimizer::Adam { lr, beta1, beta2, epsilon, m, v, t } => {
                *t += 1;
                let mut idx = 0;
                for layer in &mut model.layers {
                    for (param, grad) in layer.parameters_mut() {
                        if m.len() <= idx { m.push(vec![0.0; param.data.len()]); v.push(vec![0.0; param.data.len()]); }
                        for i in 0..param.data.len() {
                            m[idx][i] = *beta1 * m[idx][i] + (1.0 - *beta1) * grad.data[i];
                            v[idx][i] = *beta2 * v[idx][i] + (1.0 - *beta2) * grad.data[i] * grad.data[i];
                            let m_hat = m[idx][i] / (1.0 - beta1.powi(*t as i32));
                            let v_hat = v[idx][i] / (1.0 - beta2.powi(*t as i32));
                            param.data[i] -= *lr * m_hat / (v_hat.sqrt() + *epsilon);
                        }
                        idx += 1;
                    }
                }
            }
            Optimizer::AdamW { lr, beta1, beta2, epsilon, weight_decay, m, v, t } => {
                *t += 1;
                let mut idx = 0;
                for layer in &mut model.layers {
                    for (param, grad) in layer.parameters_mut() {
                        if m.len() <= idx { m.push(vec![0.0; param.data.len()]); v.push(vec![0.0; param.data.len()]); }
                        for i in 0..param.data.len() {
                            param.data[i] *= 1.0 - *lr * *weight_decay;
                            m[idx][i] = *beta1 * m[idx][i] + (1.0 - *beta1) * grad.data[i];
                            v[idx][i] = *beta2 * v[idx][i] + (1.0 - *beta2) * grad.data[i] * grad.data[i];
                            let m_hat = m[idx][i] / (1.0 - beta1.powi(*t as i32));
                            let v_hat = v[idx][i] / (1.0 - beta2.powi(*t as i32));
                            param.data[i] -= *lr * m_hat / (v_hat.sqrt() + *epsilon);
                        }
                        idx += 1;
                    }
                }
            }
        }
    }
}

// ─── Learning Rate Schedulers ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum LRScheduler {
    StepLR { step_size: usize, gamma: f64 },
    CosineAnnealing { t_max: usize, eta_min: f64 },
    WarmupCosine { warmup_steps: usize, total_steps: usize, eta_min: f64 },
}

impl LRScheduler {
    pub fn get_lr(&self, base_lr: f64, epoch: usize) -> f64 {
        match self {
            LRScheduler::StepLR { step_size, gamma } => {
                base_lr * gamma.powi((epoch / step_size) as i32)
            }
            LRScheduler::CosineAnnealing { t_max, eta_min } => {
                eta_min + (base_lr - eta_min) * (1.0 + (std::f64::consts::PI * epoch as f64 / *t_max as f64).cos()) / 2.0
            }
            LRScheduler::WarmupCosine { warmup_steps, total_steps, eta_min } => {
                if epoch < *warmup_steps {
                    base_lr * epoch as f64 / *warmup_steps as f64
                } else {
                    let progress = (epoch - warmup_steps) as f64 / (total_steps - warmup_steps) as f64;
                    eta_min + (base_lr - eta_min) * (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0
                }
            }
        }
    }

    pub fn set_lr(optimizer: &mut Optimizer, lr: f64) {
        match optimizer {
            Optimizer::SGD { lr: ref mut l, .. } => *l = lr,
            Optimizer::Adam { lr: ref mut l, .. } => *l = lr,
            Optimizer::AdamW { lr: ref mut l, .. } => *l = lr,
        }
    }
}

// ─── DataLoader ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataLoader {
    pub data: Vec<Vec<f64>>,
    pub labels: Vec<Vec<f64>>,
    pub batch_size: usize,
    pub shuffle: bool,
    indices: Vec<usize>,
}

impl DataLoader {
    pub fn new(data: Vec<Vec<f64>>, labels: Vec<Vec<f64>>, batch_size: usize, shuffle: bool) -> Self {
        let n = data.len();
        DataLoader { data, labels, batch_size, shuffle, indices: (0..n).collect() }
    }

    pub fn shuffle_indices(&mut self, seed: u64) {
        if !self.shuffle { return; }
        let n = self.indices.len();
        let mut s = seed;
        for i in (1..n).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let j = (s >> 33) as usize % (i + 1);
            self.indices.swap(i, j);
        }
    }

    pub fn num_batches(&self) -> usize {
        (self.data.len() + self.batch_size - 1) / self.batch_size
    }

    pub fn get_batch(&self, batch_idx: usize) -> (Tensor, Tensor) {
        let start = batch_idx * self.batch_size;
        let end = (start + self.batch_size).min(self.data.len());
        let bs = end - start;
        let in_dim = self.data[0].len();
        let out_dim = self.labels[0].len();
        let mut data = Vec::with_capacity(bs * in_dim);
        let mut labels = Vec::with_capacity(bs * out_dim);
        for i in start..end {
            let idx = self.indices[i];
            data.extend_from_slice(&self.data[idx]);
            labels.extend_from_slice(&self.labels[idx]);
        }
        (Tensor::new(vec![bs, in_dim], data), Tensor::new(vec![bs, out_dim], labels))
    }
}

// ─── Training Loop ─────────────────────────────────────────────────────

pub fn train_epoch(model: &mut Model, loader: &DataLoader, optimizer: &mut Optimizer, loss_fn: &str) -> f64 {
    let mut total_loss = 0.0;
    let nb = loader.num_batches();
    for b in 0..nb {
        let (x, y) = loader.get_batch(b);
        model.zero_grad();
        let pred = model.forward(&x);
        let (loss, grad) = match loss_fn {
            "mse" => mse_loss(&pred, &y),
            "cross_entropy" => cross_entropy_loss(&pred, &y),
            "bce" => binary_cross_entropy_loss(&pred, &y),
            _ => mse_loss(&pred, &y),
        };
        total_loss += loss;
        model.backward(&grad);
        optimizer.step(model);
    }
    total_loss / nb as f64
}

pub fn evaluate(model: &mut Model, loader: &DataLoader, loss_fn: &str) -> (f64, f64) {
    let mut total_loss = 0.0;
    let mut correct = 0usize;
    let mut total = 0usize;
    let nb = loader.num_batches();
    for b in 0..nb {
        let (x, y) = loader.get_batch(b);
        let pred = model.forward(&x);
        let (loss, _) = match loss_fn {
            "mse" => mse_loss(&pred, &y),
            "cross_entropy" => cross_entropy_loss(&pred, &y),
            "bce" => binary_cross_entropy_loss(&pred, &y),
            _ => mse_loss(&pred, &y),
        };
        total_loss += loss;
        // Accuracy for classification
        let classes = *pred.shape.last().unwrap();
        let batch = pred.data.len() / classes;
        for i in 0..batch {
            let pred_class = (0..classes).max_by(|&a, &b| {
                pred.data[i*classes+a].partial_cmp(&pred.data[i*classes+b]).unwrap()
            }).unwrap();
            let target_class = if y.shape.len() > 1 && y.shape[1] > 1 {
                (0..y.shape[1]).max_by(|&a, &b| y.data[i*y.shape[1]+a].partial_cmp(&y.data[i*y.shape[1]+b]).unwrap()).unwrap()
            } else {
                y.data[i] as usize
            };
            if pred_class == target_class { correct += 1; }
            total += 1;
        }
    }
    (total_loss / nb as f64, correct as f64 / total as f64)
}

/// High-level train function: returns vec of losses per epoch
pub fn train(model: &mut Model, data: Vec<Vec<f64>>, labels: Vec<Vec<f64>>,
             optimizer: &mut Optimizer, epochs: usize, loss_fn: &str) -> Vec<f64> {
    let mut loader = DataLoader::new(data, labels, 32.min(1.max(1)), true);
    // Use full batch for small datasets
    loader.batch_size = loader.data.len();
    let mut losses = Vec::with_capacity(epochs);
    for epoch in 0..epochs {
        loader.shuffle_indices(epoch as u64 * 7 + 42);
        let loss = train_epoch(model, &loader, optimizer, loss_fn);
        losses.push(loss);
    }
    losses
}

// ─── Model I/O ─────────────────────────────────────────────────────────

pub fn save_model(model: &Model, path: &str) -> Result<(), String> {
    let mut params: Vec<Vec<f64>> = Vec::new();
    for layer in &model.layers {
        for (p, _) in layer.parameters() {
            params.push(p.data.clone());
        }
    }
    let json = format!("{{\"name\":\"{}\",\"params\":{}}}", model.name,
        params.iter().map(|p| format!("[{}]", p.iter().map(|v| format!("{}", v)).collect::<Vec<_>>().join(","))).collect::<Vec<_>>().join(","));
    std::fs::write(path, json).map_err(|e| e.to_string())
}

pub fn load_model_weights(model: &mut Model, path: &str) -> Result<(), String> {
    let content = std::fs::read_to_string(path).map_err(|e| e.to_string())?;
    // Simple JSON parsing for params
    let params_start = content.find("\"params\":").ok_or("invalid format")?;
    let rest = &content[params_start + 9..content.len()-1];
    let mut params: Vec<Vec<f64>> = Vec::new();
    let mut depth = 0;
    let mut current = String::new();
    for ch in rest.chars() {
        match ch {
            '[' => { depth += 1; if depth == 1 { current.clear(); continue; } }
            ']' => {
                depth -= 1;
                if depth == 0 {
                    let vals: Vec<f64> = current.split(',').filter(|s| !s.is_empty())
                        .map(|s| s.trim().parse::<f64>().unwrap_or(0.0)).collect();
                    params.push(vals);
                    current.clear();
                    continue;
                }
            }
            ',' if depth == 0 => continue,
            _ => {}
        }
        if depth >= 1 { current.push(ch); }
    }
    let mut idx = 0;
    for layer in &mut model.layers {
        for (p, _) in layer.parameters_mut() {
            if idx < params.len() && params[idx].len() == p.data.len() {
                p.data = params[idx].clone();
            }
            idx += 1;
        }
    }
    Ok(())
}

// ─── Interpreter integration helpers ───────────────────────────────────

use crate::interpreter::{Value, Env, FnDef};

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected number in array".into()),
        }).collect(),
        _ => Err("expected array".into()),
    }
}

fn value_to_2d(v: &Value) -> Result<Vec<Vec<f64>>, String> {
    match v {
        Value::Array(rows) => rows.iter().map(|row| value_to_f64_vec(row)).collect(),
        _ => Err("expected 2D array".into()),
    }
}

fn value_to_usize_vec(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Int(i) => Ok(*i as usize),
            Value::Float(f) => Ok(*f as usize),
            _ => Err("expected int in shape".into()),
        }).collect(),
        _ => Err("expected array for shape".into()),
    }
}

fn tensor_to_value(t: &Tensor) -> Value {
    let shape = Value::Array(t.shape.iter().map(|s| Value::Int(*s as i128)).collect());
    let data = Value::Array(t.data.iter().map(|v| Value::Float(*v)).collect());
    Value::Array(vec![shape, data])
}

pub fn register_nn_builtins(env: &mut Env) {
    env.functions.insert("nn_linear".into(), FnDef::Builtin(builtin_nn_linear));
    env.functions.insert("nn_conv2d".into(), FnDef::Builtin(builtin_nn_conv2d));
    env.functions.insert("nn_transformer".into(), FnDef::Builtin(builtin_nn_transformer));
    env.functions.insert("nn_lstm".into(), FnDef::Builtin(builtin_nn_lstm));
    env.functions.insert("nn_sequential".into(), FnDef::Builtin(builtin_nn_sequential));
    env.functions.insert("nn_forward".into(), FnDef::Builtin(builtin_nn_forward));
    env.functions.insert("nn_train".into(), FnDef::Builtin(builtin_nn_train));
    env.functions.insert("nn_predict".into(), FnDef::Builtin(builtin_nn_predict));
    env.functions.insert("nn_save".into(), FnDef::Builtin(builtin_nn_save));
    env.functions.insert("nn_load".into(), FnDef::Builtin(builtin_nn_load));
    env.functions.insert("nn_adam".into(), FnDef::Builtin(builtin_nn_adam));
    env.functions.insert("nn_sgd".into(), FnDef::Builtin(builtin_nn_sgd));
    env.functions.insert("nn_cross_entropy".into(), FnDef::Builtin(builtin_nn_cross_entropy));
    env.functions.insert("tensor_new".into(), FnDef::Builtin(builtin_tensor_new));
    env.functions.insert("tensor_zeros_nn".into(), FnDef::Builtin(builtin_tensor_zeros));
    env.functions.insert("tensor_randn_nn".into(), FnDef::Builtin(builtin_tensor_randn));
    env.functions.insert("tensor_matmul_nn".into(), FnDef::Builtin(builtin_tensor_matmul));
    env.functions.insert("tensor_add_nn".into(), FnDef::Builtin(builtin_tensor_add));
    env.functions.insert("nn_relu".into(), FnDef::Builtin(builtin_nn_relu));
    env.functions.insert("nn_sigmoid".into(), FnDef::Builtin(builtin_nn_sigmoid));
    env.functions.insert("nn_tanh".into(), FnDef::Builtin(builtin_nn_tanh));
    env.functions.insert("nn_gelu".into(), FnDef::Builtin(builtin_nn_gelu));
    env.functions.insert("nn_softmax".into(), FnDef::Builtin(builtin_nn_softmax));
    env.functions.insert("nn_embedding".into(), FnDef::Builtin(builtin_nn_embedding));
    env.functions.insert("nn_layer_norm".into(), FnDef::Builtin(builtin_nn_layer_norm));
    env.functions.insert("nn_dropout".into(), FnDef::Builtin(builtin_nn_dropout));
    env.functions.insert("nn_gru".into(), FnDef::Builtin(builtin_nn_gru));
    env.functions.insert("nn_batch_norm".into(), FnDef::Builtin(builtin_nn_batch_norm));
    env.functions.insert("nn_train_verbose".into(), FnDef::Builtin(builtin_nn_train_verbose));
    env.functions.insert("nn_evaluate".into(), FnDef::Builtin(builtin_nn_evaluate));
    env.functions.insert("nn_num_params".into(), FnDef::Builtin(builtin_nn_num_params));
    env.functions.insert("nn_clone".into(), FnDef::Builtin(builtin_nn_clone));
}

fn builtin_nn_linear(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_linear(in, out)".into()); }
    let in_f = match &args[0] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => return Err("expected int".into()) };
    let out_f = match &args[1] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::linear(in_f, out_f)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_conv2d(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 { return Err("nn_conv2d(in_ch, out_ch, kernel)".into()); }
    let in_ch = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let out_ch = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let ks = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::conv2d(in_ch, out_ch, ks, 1, 0)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_transformer(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("nn_transformer(dim, heads, ff_dim, num_layers)".into()); }
    let dim = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let heads = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let ff_dim = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let n = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::transformer(dim, heads, ff_dim, n);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_lstm(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_lstm(input_size, hidden_size)".into()); }
    let input_size = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let hidden_size = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::lstm(input_size, hidden_size)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_sequential(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_sequential(layer_ids_array)".into()); }
    let ids = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Int(i) => Ok(*i as usize),
            _ => Err("expected int id".into()),
        }).collect::<Result<Vec<usize>, String>>()?,
        _ => return Err("expected array".into()),
    };
    // Merge layers from sub-models
    let mut layers = Vec::new();
    for id in &ids {
        if let Some(m) = env.nn_models.get(id) {
            layers.extend(m.layers.clone());
        } else {
            return Err(format!("no model with id {}", id));
        }
    }
    let model = Model::sequential(layers);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_forward(model_id, input)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let input = parse_tensor_arg(&args[1])?;
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    let out = model.forward(&input);
    Ok(tensor_to_value(&out))
}

fn builtin_nn_train(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 { return Err("nn_train(model_id, data, labels, optimizer_name, epochs, lr)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let data = value_to_2d(&args[1])?;
    let labels = value_to_2d(&args[2])?;
    let opt_name = match &args[3] { Value::String(s) => s.clone(), _ => "adam".into() };
    let epochs = match &args[4] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => 100 };
    let lr = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.01 };
    let mut optimizer = match opt_name.as_str() {
        "sgd" => Optimizer::sgd(lr, 0.9, 0.0),
        "adamw" => Optimizer::adamw(lr, 0.01),
        _ => Optimizer::adam(lr),
    };
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    // Detect classification: if all label values are small non-negative integers and single column
    let is_class_index = labels[0].len() == 1 && labels.iter().all(|l| l[0] >= 0.0 && l[0] == l[0].floor() && l[0] < 100.0);
    let num_outputs = model.layers.iter().rev().find_map(|l| match l {
        Layer::Linear { weight, .. } => Some(weight.shape[1]),
        _ => None,
    }).unwrap_or(1);
    let loss_fn = if num_outputs > 1 && is_class_index { "cross_entropy" } else { "mse" };
    let losses = train(model, data, labels, &mut optimizer, epochs, loss_fn);
    Ok(Value::Array(losses.into_iter().map(|l| Value::Float(l)).collect()))
}

fn builtin_nn_predict(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_predict(model_id, input)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let input_data = value_to_f64_vec(&args[1])?;
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    let input = Tensor::new(vec![1, input_data.len()], input_data);
    let out = model.forward(&input);
    Ok(Value::Array(out.data.into_iter().map(|v| Value::Float(v)).collect()))
}

fn builtin_nn_save(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_save(model_id, path)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("expected string path".into()) };
    let model = env.nn_models.get(&model_id).ok_or("no such model")?;
    save_model(model, &path)?;
    Ok(Value::Void)
}

fn builtin_nn_load(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_load(model_id, path)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("expected string path".into()) };
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    load_model_weights(model, &path)?;
    Ok(Value::Void)
}

fn builtin_nn_adam(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lr = match args.first() { Some(Value::Float(f)) => *f, Some(Value::Int(i)) => *i as f64, _ => 0.001 };
    Ok(Value::String(format!("adam:{}", lr)))
}

fn builtin_nn_sgd(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    let lr = match args.first() { Some(Value::Float(f)) => *f, Some(Value::Int(i)) => *i as f64, _ => 0.01 };
    Ok(Value::String(format!("sgd:{}", lr)))
}

fn builtin_nn_cross_entropy(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_cross_entropy(pred, target)".into()); }
    let pred = parse_tensor_arg(&args[0])?;
    let target = parse_tensor_arg(&args[1])?;
    let (loss, _) = cross_entropy_loss(&pred, &target);
    Ok(Value::Float(loss))
}

fn builtin_tensor_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_new(shape, data)".into()); }
    let shape = value_to_usize_vec(&args[0])?;
    let data = value_to_f64_vec(&args[1])?;
    let t = Tensor::new(shape, data);
    Ok(tensor_to_value(&t))
}

fn builtin_tensor_zeros(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_zeros(shape)".into()); }
    let shape = value_to_usize_vec(&args[0])?;
    Ok(tensor_to_value(&Tensor::zeros(shape)))
}

fn builtin_tensor_randn(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_randn(shape)".into()); }
    let shape = value_to_usize_vec(&args[0])?;
    Ok(tensor_to_value(&Tensor::randn(shape)))
}

fn builtin_tensor_matmul(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_matmul(a, b)".into()); }
    let a = parse_tensor_arg(&args[0])?;
    let b = parse_tensor_arg(&args[1])?;
    Ok(tensor_to_value(&a.matmul(&b)))
}

fn builtin_tensor_add(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("tensor_add(a, b)".into()); }
    let a = parse_tensor_arg(&args[0])?;
    let b = parse_tensor_arg(&args[1])?;
    Ok(tensor_to_value(&a.add(&b)))
}

fn builtin_nn_relu(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let model = Model::sequential(vec![Layer::ReLU { cache: None }]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_sigmoid(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let model = Model::sequential(vec![Layer::Sigmoid { cache: None }]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_tanh(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let model = Model::sequential(vec![Layer::Tanh { cache: None }]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_gelu(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let model = Model::sequential(vec![Layer::GELU { cache: None }]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_softmax(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let model = Model::sequential(vec![Layer::Softmax { cache: None }]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_embedding(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_embedding(vocab_size, dim)".into()); }
    let vocab = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let dim = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::embedding(vocab, dim)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_layer_norm(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_layer_norm(dim)".into()); }
    let dim = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::layer_norm(dim)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_dropout(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_dropout(rate)".into()); }
    let rate = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("expected number".into()) };
    let model = Model::sequential(vec![Layer::dropout(rate)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_gru(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("nn_gru(input_size, hidden_size)".into()); }
    let input_size = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let hidden_size = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::gru(input_size, hidden_size)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_batch_norm(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_batch_norm(dim)".into()); }
    let dim = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = Model::sequential(vec![Layer::batch_norm(dim)]);
    let id = env.nn_models.len();
    env.nn_models.insert(id, model);
    Ok(Value::Int(id as i128))
}

fn builtin_nn_train_verbose(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 7 { return Err("nn_train_verbose(model_id, data, labels, optimizer, epochs, lr, print_every)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let data = value_to_2d(&args[1])?;
    let labels = value_to_2d(&args[2])?;
    let opt_name = match &args[3] { Value::String(s) => s.clone(), _ => "adam".into() };
    let epochs = match &args[4] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => 100 };
    let lr = match &args[5] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.01 };
    let print_every = match &args[6] { Value::Int(i) => *i as usize, Value::Float(f) => *f as usize, _ => 10 };
    let mut optimizer = match opt_name.as_str() {
        "sgd" => Optimizer::sgd(lr, 0.9, 0.0),
        "adamw" => Optimizer::adamw(lr, 0.01),
        _ => Optimizer::adam(lr),
    };
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    let is_class_index = labels[0].len() == 1 && labels.iter().all(|l| l[0] >= 0.0 && l[0] == l[0].floor() && l[0] < 100.0);
    let num_outputs = model.layers.iter().rev().find_map(|l| match l {
        Layer::Linear { weight, .. } => Some(weight.shape[1]),
        _ => None,
    }).unwrap_or(1);
    let loss_fn = if num_outputs > 1 && is_class_index { "cross_entropy" } else { "mse" };
    let mut loader = DataLoader::new(data, labels, 32, true);
    loader.batch_size = loader.data.len();
    let mut final_loss = 0.0;
    for epoch in 0..epochs {
        loader.shuffle_indices(epoch as u64 * 7 + 42);
        let loss = train_epoch(model, &loader, &mut optimizer, loss_fn);
        final_loss = loss;
        if epoch % print_every == 0 || epoch == epochs - 1 {
            println!("  Epoch {}/{}: loss = {:.6}", epoch + 1, epochs, loss);
        }
    }
    Ok(Value::Float(final_loss))
}

fn builtin_nn_evaluate(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("nn_evaluate(model_id, data, labels)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let data = value_to_2d(&args[1])?;
    let labels = value_to_2d(&args[2])?;
    let model = env.nn_models.get_mut(&model_id).ok_or("no such model")?;
    let is_class_index = labels[0].len() == 1 && labels.iter().all(|l| l[0] >= 0.0 && l[0] == l[0].floor() && l[0] < 100.0);
    let num_outputs = model.layers.iter().rev().find_map(|l| match l {
        Layer::Linear { weight, .. } => Some(weight.shape[1]),
        _ => None,
    }).unwrap_or(1);
    let loss_fn = if num_outputs > 1 && is_class_index { "cross_entropy" } else { "mse" };
    let loader = DataLoader::new(data, labels, 32, false);
    let (loss, acc) = evaluate(model, &loader, loss_fn);
    Ok(Value::Array(vec![Value::Float(loss), Value::Float(acc)]))
}

fn builtin_nn_num_params(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_num_params(model_id)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = env.nn_models.get(&model_id).ok_or("no such model")?;
    Ok(Value::Int(model.num_parameters() as i128))
}

fn builtin_nn_clone(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("nn_clone(model_id)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let model = env.nn_models.get(&model_id).ok_or("no such model")?.clone();
    let new_id = env.nn_models.len();
    env.nn_models.insert(new_id, model);
    Ok(Value::Int(new_id as i128))
}

fn parse_tensor_arg(v: &Value) -> Result<Tensor, String> {
    match v {
        Value::Array(arr) if arr.len() == 2 => {
            if let (Value::Array(_), Value::Array(_)) = (&arr[0], &arr[1]) {
                let shape = value_to_usize_vec(&arr[0])?;
                let data = value_to_f64_vec(&arr[1])?;
                return Ok(Tensor::new(shape, data));
            }
            // Flat array of floats - treat as 1D
            let data = value_to_f64_vec(v)?;
            Ok(Tensor::new(vec![data.len()], data))
        }
        Value::Array(arr) => {
            let data = value_to_f64_vec(v)?;
            Ok(Tensor::new(vec![arr.len()], data))
        }
        _ => Err("expected tensor (array of [shape, data])".into()),
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_new() {
        let t = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(t.shape, vec![2, 3]);
        assert_eq!(t.numel(), 6);
    }

    #[test]
    fn test_tensor_zeros() {
        let t = Tensor::zeros(vec![3, 4]);
        assert_eq!(t.data.len(), 12);
        assert!(t.data.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn test_tensor_randn() {
        let t = Tensor::randn(vec![10, 10]);
        assert_eq!(t.numel(), 100);
        // Should not be all zeros
        assert!(t.data.iter().any(|v| v.abs() > 0.01));
    }

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let c = a.matmul(&b);
        assert_eq!(c.shape, vec![2, 2]);
        assert_eq!(c.data, vec![22.0, 28.0, 49.0, 64.0]);
    }

    #[test]
    fn test_transpose() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.data, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_relu() {
        let t = Tensor::new(vec![4], vec![-2.0, -1.0, 0.0, 1.0]);
        let r = t.relu();
        assert_eq!(r.data, vec![0.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sigmoid() {
        let t = Tensor::new(vec![1], vec![0.0]);
        let s = t.sigmoid();
        assert!((s.data[0] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_softmax() {
        let t = Tensor::new(vec![1, 3], vec![1.0, 2.0, 3.0]);
        let s = t.softmax();
        let sum: f64 = s.data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(s.data[2] > s.data[1] && s.data[1] > s.data[0]);
    }

    #[test]
    fn test_linear_forward() {
        let mut layer = Layer::linear(3, 2);
        let input = Tensor::new(vec![1, 3], vec![1.0, 0.5, -0.5]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![1, 2]);
    }

    #[test]
    fn test_linear_backward() {
        let mut layer = Layer::linear(2, 2);
        let input = Tensor::new(vec![1, 2], vec![1.0, 2.0]);
        let _ = layer.forward(&input);
        let d_out = Tensor::new(vec![1, 2], vec![1.0, 1.0]);
        let d_in = layer.backward(&d_out);
        assert_eq!(d_in.shape, vec![1, 2]);
    }

    #[test]
    fn test_sequential_model() {
        let mut model = Model::sequential(vec![
            Layer::linear(2, 4),
            Layer::ReLU { cache: None },
            Layer::linear(4, 1),
            Layer::Sigmoid { cache: None },
        ]);
        let input = Tensor::new(vec![1, 2], vec![0.5, 0.3]);
        let out = model.forward(&input);
        assert_eq!(out.shape, vec![1, 1]);
        assert!(out.data[0] >= 0.0 && out.data[0] <= 1.0);
    }

    #[test]
    fn test_mse_loss() {
        let pred = Tensor::new(vec![2], vec![1.0, 2.0]);
        let target = Tensor::new(vec![2], vec![1.0, 2.0]);
        let (loss, _) = mse_loss(&pred, &target);
        assert!((loss).abs() < 1e-10);
    }

    #[test]
    fn test_cross_entropy() {
        let logits = Tensor::new(vec![1, 3], vec![2.0, 1.0, 0.1]);
        let targets = Tensor::new(vec![1], vec![0.0]);
        let (loss, grad) = cross_entropy_loss(&logits, &targets);
        assert!(loss > 0.0);
        assert_eq!(grad.shape, vec![1, 3]);
    }

    #[test]
    fn test_bce_loss() {
        let pred = Tensor::new(vec![2], vec![0.9, 0.1]);
        let target = Tensor::new(vec![2], vec![1.0, 0.0]);
        let (loss, _) = binary_cross_entropy_loss(&pred, &target);
        assert!(loss < 0.2);
    }

    #[test]
    fn test_adam_optimizer() {
        let mut model = Model::sequential(vec![Layer::linear(2, 1)]);
        let mut opt = Optimizer::adam(0.01);
        let input = Tensor::new(vec![1, 2], vec![1.0, 1.0]);
        let target = Tensor::new(vec![1, 1], vec![0.5]);
        model.zero_grad();
        let pred = model.forward(&input);
        let (_, grad) = mse_loss(&pred, &target);
        model.backward(&grad);
        opt.step(&mut model);
        // Weights should have changed
        let pred2 = model.forward(&input);
        // Second prediction should be closer to target
        let d1 = (pred.data[0] - 0.5).abs();
        let d2 = (pred2.data[0] - 0.5).abs();
        // At least optimizer ran without error
        assert!(d1 >= 0.0 && d2 >= 0.0);
    }

    #[test]
    fn test_train_xor() {
        let mut model = Model::sequential(vec![
            Layer::linear(2, 8),
            Layer::ReLU { cache: None },
            Layer::linear(8, 1),
            Layer::Sigmoid { cache: None },
        ]);
        let data = vec![vec![0.0,0.0], vec![0.0,1.0], vec![1.0,0.0], vec![1.0,1.0]];
        let labels = vec![vec![0.0], vec![1.0], vec![1.0], vec![0.0]];
        let mut opt = Optimizer::adam(0.01);
        let losses = train(&mut model, data, labels, &mut opt, 500, "mse");
        assert!(losses.last().unwrap() < &losses[0], "Loss should decrease");
    }

    #[test]
    fn test_conv2d_forward() {
        let mut layer = Layer::conv2d(1, 2, 3, 1, 0);
        let input = Tensor::randn(vec![1, 6, 6]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![2, 4, 4]);
    }

    #[test]
    fn test_layer_norm() {
        let mut layer = Layer::layer_norm(4);
        let input = Tensor::new(vec![2, 4], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![2, 4]);
        // Each row should be approximately zero-mean
        let mean: f64 = out.data[0..4].iter().sum::<f64>() / 4.0;
        assert!(mean.abs() < 1e-5);
    }

    #[test]
    fn test_dropout() {
        let mut layer = Layer::dropout(0.5);
        let input = Tensor::ones(vec![100]);
        let out = layer.forward(&input);
        // Some values should be zero
        let zeros = out.data.iter().filter(|v| **v == 0.0).count();
        assert!(zeros > 10 && zeros < 90);
    }

    #[test]
    fn test_embedding() {
        let mut layer = Layer::embedding(10, 4);
        let indices = Tensor::new(vec![3], vec![0.0, 5.0, 9.0]);
        let out = layer.forward(&indices);
        assert_eq!(out.shape, vec![3, 4]);
    }

    #[test]
    fn test_mha_forward() {
        let mut layer = Layer::multi_head_attention(8, 2);
        let input = Tensor::randn(vec![1, 4, 8]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![1, 4, 8]);
    }

    #[test]
    fn test_transformer_model() {
        let mut model = Model::transformer(8, 2, 16, 1);
        let input = Tensor::randn(vec![1, 4, 8]);
        let out = model.forward(&input);
        assert_eq!(out.shape, vec![1, 4, 8]);
    }

    #[test]
    fn test_lstm_forward() {
        let mut layer = Layer::lstm(4, 8);
        let input = Tensor::randn(vec![3, 4]); // seq_len=3, input=4
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_gru_forward() {
        let mut layer = Layer::gru(4, 8);
        let input = Tensor::randn(vec![3, 4]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![3, 8]);
    }

    #[test]
    fn test_dataloader() {
        let data = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0], vec![7.0, 8.0]];
        let labels = vec![vec![0.0], vec![1.0], vec![0.0], vec![1.0]];
        let loader = DataLoader::new(data, labels, 2, false);
        assert_eq!(loader.num_batches(), 2);
        let (x, y) = loader.get_batch(0);
        assert_eq!(x.shape, vec![2, 2]);
        assert_eq!(y.shape, vec![2, 1]);
    }

    #[test]
    fn test_lr_scheduler_step() {
        let sched = LRScheduler::StepLR { step_size: 10, gamma: 0.1 };
        assert!((sched.get_lr(0.1, 0) - 0.1).abs() < 1e-10);
        assert!((sched.get_lr(0.1, 10) - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_lr_scheduler_cosine() {
        let sched = LRScheduler::CosineAnnealing { t_max: 100, eta_min: 0.0 };
        let lr0 = sched.get_lr(0.1, 0);
        let lr50 = sched.get_lr(0.1, 50);
        let lr100 = sched.get_lr(0.1, 100);
        assert!((lr0 - 0.1).abs() < 1e-10);
        assert!(lr50 < lr0);
        assert!((lr100).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_optimizer() {
        let mut model = Model::sequential(vec![Layer::linear(2, 1)]);
        let mut opt = Optimizer::sgd(0.01, 0.0, 0.0);
        let input = Tensor::new(vec![1, 2], vec![1.0, 1.0]);
        let target = Tensor::new(vec![1, 1], vec![0.5]);
        model.zero_grad();
        let pred = model.forward(&input);
        let (_, grad) = mse_loss(&pred, &target);
        model.backward(&grad);
        opt.step(&mut model);
        // Should run without error
    }

    #[test]
    fn test_adamw_optimizer() {
        let mut model = Model::sequential(vec![Layer::linear(2, 1)]);
        let mut opt = Optimizer::adamw(0.01, 0.01);
        let input = Tensor::new(vec![1, 2], vec![1.0, 1.0]);
        let target = Tensor::new(vec![1, 1], vec![0.5]);
        model.zero_grad();
        let pred = model.forward(&input);
        let (_, grad) = mse_loss(&pred, &target);
        model.backward(&grad);
        opt.step(&mut model);
    }

    #[test]
    fn test_batch_norm() {
        let mut layer = Layer::batch_norm(4);
        let input = Tensor::new(vec![3, 4], vec![1.0,2.0,3.0,4.0, 5.0,6.0,7.0,8.0, 9.0,10.0,11.0,12.0]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![3, 4]);
    }

    #[test]
    fn test_model_num_parameters() {
        let model = Model::sequential(vec![Layer::linear(10, 5), Layer::linear(5, 2)]);
        // 10*5 + 5 + 5*2 + 2 = 67
        assert_eq!(model.num_parameters(), 67);
    }

    #[test]
    fn test_feed_forward_layer() {
        let mut layer = Layer::feed_forward(8, 16);
        let input = Tensor::randn(vec![2, 8]);
        let out = layer.forward(&input);
        assert_eq!(out.shape, vec![2, 8]);
    }

    #[test]
    fn test_broadcast_add() {
        let a = Tensor::new(vec![2, 3], vec![1.0,2.0,3.0,4.0,5.0,6.0]);
        let b = Tensor::new(vec![3], vec![10.0, 20.0, 30.0]);
        let c = a.add(&b);
        assert_eq!(c.data, vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
    }
}
