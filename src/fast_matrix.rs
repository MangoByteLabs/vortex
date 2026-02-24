/// Fast Matrix Algebra for AI: Monarch, Butterfly, Sparse Attention, Low-Rank
///
/// Achieves sub-quadratic complexity for matrix operations that dominate AI workloads:
/// - Monarch matrices: O(n^1.5) matmul via block-diagonal + permutation decomposition
/// - Butterfly transforms: O(n log n) matmul via log(n) sparse factors
/// - Sparse attention patterns: O(kn) where k << n
/// - Auto low-rank: O(rn) where r = detected rank << n
/// - Combined pipeline: automatic selection of fastest path

use std::collections::HashMap;

// ─── Dense Matrix Primitives ────────────────────────────────────────────

/// Row-major dense matrix
#[derive(Debug, Clone)]
pub struct DenseMatrix {
    pub data: Vec<f64>,
    pub rows: usize,
    pub cols: usize,
}

impl DenseMatrix {
    pub fn new(rows: usize, cols: usize) -> Self {
        Self { data: vec![0.0; rows * cols], rows, cols }
    }

    pub fn from_data(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols);
        Self { data, rows, cols }
    }

    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Self::new(n, n);
        for i in 0..n { m.set(i, i, 1.0); }
        m
    }

    pub fn naive_matmul(&self, other: &DenseMatrix) -> DenseMatrix {
        assert_eq!(self.cols, other.rows);
        let mut result = DenseMatrix::new(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.get(i, k);
                for j in 0..other.cols {
                    let idx = i * other.cols + j;
                    result.data[idx] += a * other.get(k, j);
                }
            }
        }
        result
    }

    /// Frobenius norm
    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Transpose
    pub fn transpose(&self) -> DenseMatrix {
        let mut t = DenseMatrix::new(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                t.set(j, i, self.get(i, j));
            }
        }
        t
    }

    /// Matrix-vector multiply
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(self.cols, v.len());
        let mut result = vec![0.0; self.rows];
        for i in 0..self.rows {
            for j in 0..self.cols {
                result[i] += self.get(i, j) * v[j];
            }
        }
        result
    }
}

// ─── Monarch Matrix ─────────────────────────────────────────────────────
// W ≈ P₂ · BlockDiag(B₁..Bk) · P₁ · BlockDiag(A₁..Ak)
// Complexity: O(n^1.5) instead of O(n²)
// Reference: Dao et al., "Monarch: Expressive Structured Matrices for Efficient and Accurate Training"

/// A Monarch matrix of size n×n decomposed into two block-diagonal layers with permutations.
/// n must be a perfect square: n = b², where b is the block size.
#[derive(Debug, Clone)]
pub struct MonarchMatrix {
    /// Block size (sqrt of matrix dimension)
    pub block_size: usize,
    /// Number of blocks = block_size
    pub num_blocks: usize,
    /// First layer: num_blocks blocks, each block_size × block_size
    pub blocks_a: Vec<DenseMatrix>,
    /// Second layer: num_blocks blocks, each block_size × block_size
    pub blocks_b: Vec<DenseMatrix>,
}

impl MonarchMatrix {
    /// Create a Monarch matrix for dimension n (must be perfect square)
    pub fn new(n: usize) -> Self {
        let b = (n as f64).sqrt() as usize;
        assert_eq!(b * b, n, "Monarch matrix dimension must be a perfect square");
        let blocks_a = (0..b).map(|_| DenseMatrix::identity(b)).collect();
        let blocks_b = (0..b).map(|_| DenseMatrix::identity(b)).collect();
        Self { block_size: b, num_blocks: b, blocks_a, blocks_b }
    }

    /// Create with random initialization
    pub fn random(n: usize, seed: u64) -> Self {
        let b = (n as f64).sqrt() as usize;
        assert_eq!(b * b, n);
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        let mut make_block = || -> DenseMatrix {
            let scale = 1.0 / (b as f64).sqrt();
            DenseMatrix::from_data(b, b,
                (0..b*b).map(|_| next_f64() * scale).collect())
        };
        let blocks_a = (0..b).map(|_| make_block()).collect();
        let blocks_b = (0..b).map(|_| make_block()).collect();
        Self { block_size: b, num_blocks: b, blocks_a, blocks_b }
    }

    /// Matrix dimension
    pub fn dim(&self) -> usize { self.block_size * self.num_blocks }

    /// Apply Monarch matrix to vector x. O(n^1.5) instead of O(n²).
    ///
    /// Steps:
    /// 1. Reshape x into (num_blocks, block_size) matrix
    /// 2. Apply block-diagonal A (each block operates on its row)
    /// 3. Transpose (permutation P₁)
    /// 4. Apply block-diagonal B
    /// 5. Transpose back (permutation P₂)
    /// 6. Flatten to output vector
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        let n = self.dim();
        assert_eq!(x.len(), n);
        let b = self.block_size;

        // Step 1: Reshape to (num_blocks, block_size)
        // Step 2: Apply block-diagonal A
        let mut after_a = vec![0.0; n];
        for block_idx in 0..self.num_blocks {
            let block = &self.blocks_a[block_idx];
            let offset = block_idx * b;
            for i in 0..b {
                let mut sum = 0.0;
                for j in 0..b {
                    sum += block.get(i, j) * x[offset + j];
                }
                after_a[offset + i] = sum;
            }
        }

        // Step 3: Transpose permutation (view as b×b matrix, transpose)
        let mut permuted = vec![0.0; n];
        for i in 0..b {
            for j in 0..b {
                permuted[j * b + i] = after_a[i * b + j];
            }
        }

        // Step 4: Apply block-diagonal B
        let mut after_b = vec![0.0; n];
        for block_idx in 0..self.num_blocks {
            let block = &self.blocks_b[block_idx];
            let offset = block_idx * b;
            for i in 0..b {
                let mut sum = 0.0;
                for j in 0..b {
                    sum += block.get(i, j) * permuted[offset + j];
                }
                after_b[offset + i] = sum;
            }
        }

        // Step 5: Transpose back
        let mut output = vec![0.0; n];
        for i in 0..b {
            for j in 0..b {
                output[j * b + i] = after_b[i * b + j];
            }
        }

        output
    }

    /// Compute full dense matrix (for verification). O(n²) - only for testing.
    pub fn to_dense(&self) -> DenseMatrix {
        let n = self.dim();
        let mut result = DenseMatrix::new(n, n);
        for col in 0..n {
            let mut e = vec![0.0; n];
            e[col] = 1.0;
            let out = self.forward(&e);
            for row in 0..n {
                result.set(row, col, out[row]);
            }
        }
        result
    }

    /// FLOPs for Monarch forward: 2 * num_blocks * block_size² = 2n^1.5
    pub fn flops(&self) -> usize {
        2 * self.num_blocks * self.block_size * self.block_size
    }
}

// ─── Butterfly Matrix ───────────────────────────────────────────────────
// Product of log(n) sparse factors, each with O(n) nonzeros.
// Total: O(n log n) matmul.
// Reference: Dao et al., "Kaleidoscope: An Efficient, Learnable Representation For All Structured Linear Maps"

/// A butterfly factor: pairs of 2×2 blocks along the diagonal.
/// For dimension n and stride s, element i is paired with element i XOR s.
#[derive(Debug, Clone)]
pub struct ButterflyFactor {
    /// Diagonal elements (scale factors)
    pub diag: Vec<f64>,
    /// Off-diagonal elements (mixing factors)
    pub off_diag: Vec<f64>,
    /// Stride for this butterfly level
    pub stride: usize,
}

/// Full butterfly transform: log(n) factors composed.
#[derive(Debug, Clone)]
pub struct ButterflyMatrix {
    pub factors: Vec<ButterflyFactor>,
    pub dim: usize,
}

impl ButterflyMatrix {
    /// Create identity butterfly matrix for dimension n (must be power of 2)
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "Butterfly dimension must be power of 2");
        let num_levels = (n as f64).log2() as usize;
        let factors = (0..num_levels).map(|level| {
            ButterflyFactor {
                diag: vec![1.0; n],
                off_diag: vec![0.0; n],
                stride: 1 << level,
            }
        }).collect();
        Self { factors, dim: n }
    }

    /// Create with random initialization
    pub fn random(n: usize, seed: u64) -> Self {
        assert!(n.is_power_of_two());
        let num_levels = (n as f64).log2() as usize;
        let mut rng = seed;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };
        let scale = 1.0 / (n as f64).sqrt();
        let factors = (0..num_levels).map(|level| {
            ButterflyFactor {
                diag: (0..n).map(|_| next_f64() * scale).collect(),
                off_diag: (0..n).map(|_| next_f64() * scale).collect(),
                stride: 1 << level,
            }
        }).collect();
        Self { factors, dim: n }
    }

    /// Apply butterfly transform to vector. O(n log n).
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.dim);
        let mut current = x.to_vec();

        for factor in &self.factors {
            let mut next = vec![0.0; self.dim];
            let s = factor.stride;
            for i in 0..self.dim {
                let partner = i ^ s; // XOR gives butterfly partner
                next[i] = factor.diag[i] * current[i] + factor.off_diag[i] * current[partner];
            }
            current = next;
        }

        current
    }

    /// FLOPs: n * log(n) * 2 (one mul + one fma per element per level)
    pub fn flops(&self) -> usize {
        self.dim * self.factors.len() * 2
    }
}

// ─── Sparse Attention Patterns ──────────────────────────────────────────
// Instead of full n×n attention, use structured sparsity patterns.
// Complexity: O(kn) where k = window + random + global tokens

#[derive(Debug, Clone)]
pub enum AttentionPattern {
    /// Full dense attention - O(n²)
    Dense,
    /// Local sliding window - each token attends to w neighbors
    SlidingWindow { window_size: usize },
    /// Strided attention - attend every s-th token
    Strided { stride: usize, window_size: usize },
    /// Random attention - each token attends to k random tokens + local window
    Random { num_random: usize, window_size: usize },
    /// Global + local: some tokens attend to all, rest attend locally
    GlobalLocal { num_global: usize, window_size: usize },
    /// Longformer-style: combination of local + global + dilated
    Longformer { window_size: usize, dilated_window: usize, num_global: usize },
}

impl AttentionPattern {
    /// Generate attention mask as sparse index pairs (query_idx, key_idx).
    /// Returns sorted pairs for cache-friendly access.
    pub fn generate_mask(&self, seq_len: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();

        match self {
            AttentionPattern::Dense => {
                for i in 0..seq_len {
                    for j in 0..seq_len {
                        pairs.push((i, j));
                    }
                }
            }
            AttentionPattern::SlidingWindow { window_size } => {
                let w = *window_size;
                for i in 0..seq_len {
                    let start = if i >= w / 2 { i - w / 2 } else { 0 };
                    let end = (i + w / 2 + 1).min(seq_len);
                    for j in start..end {
                        pairs.push((i, j));
                    }
                }
            }
            AttentionPattern::Strided { stride, window_size } => {
                let w = *window_size;
                let s = *stride;
                for i in 0..seq_len {
                    // Local window
                    let start = if i >= w / 2 { i - w / 2 } else { 0 };
                    let end = (i + w / 2 + 1).min(seq_len);
                    for j in start..end {
                        pairs.push((i, j));
                    }
                    // Strided
                    let mut j = i % s;
                    while j < seq_len {
                        pairs.push((i, j));
                        j += s;
                    }
                }
            }
            AttentionPattern::Random { num_random, window_size } => {
                let w = *window_size;
                let k = *num_random;
                let mut rng: u64 = 42;
                for i in 0..seq_len {
                    // Local window
                    let start = if i >= w / 2 { i - w / 2 } else { 0 };
                    let end = (i + w / 2 + 1).min(seq_len);
                    for j in start..end {
                        pairs.push((i, j));
                    }
                    // Random tokens
                    for _ in 0..k {
                        rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
                        let j = (rng as usize) % seq_len;
                        pairs.push((i, j));
                    }
                }
            }
            AttentionPattern::GlobalLocal { num_global, window_size } => {
                let g = *num_global;
                let w = *window_size;
                for i in 0..seq_len {
                    if i < g {
                        // Global tokens attend to all
                        for j in 0..seq_len { pairs.push((i, j)); }
                    } else {
                        // Local tokens attend to global + local window
                        for j in 0..g { pairs.push((i, j)); }
                        let start = if i >= w / 2 { i - w / 2 } else { 0 };
                        let end = (i + w / 2 + 1).min(seq_len);
                        for j in start..end { pairs.push((i, j)); }
                    }
                    // All tokens are attended to by global tokens
                    for j in 0..g { pairs.push((i, j)); }
                }
            }
            AttentionPattern::Longformer { window_size, dilated_window, num_global } => {
                let w = *window_size;
                let dw = *dilated_window;
                let g = *num_global;
                for i in 0..seq_len {
                    // Global
                    if i < g {
                        for j in 0..seq_len { pairs.push((i, j)); }
                        continue;
                    }
                    for j in 0..g { pairs.push((i, j)); }
                    // Local window
                    let start = if i >= w / 2 { i - w / 2 } else { 0 };
                    let end = (i + w / 2 + 1).min(seq_len);
                    for j in start..end { pairs.push((i, j)); }
                    // Dilated window
                    for d in 1..=dw {
                        if i >= d * 2 { pairs.push((i, i - d * 2)); }
                        if i + d * 2 < seq_len { pairs.push((i, i + d * 2)); }
                    }
                }
            }
        }

        // Deduplicate and sort
        pairs.sort();
        pairs.dedup();
        pairs
    }

    /// Approximate FLOPs per sequence position
    pub fn flops_per_position(&self, seq_len: usize) -> usize {
        match self {
            AttentionPattern::Dense => seq_len,
            AttentionPattern::SlidingWindow { window_size } => *window_size,
            AttentionPattern::Strided { stride, window_size } => window_size + seq_len / stride,
            AttentionPattern::Random { num_random, window_size } => window_size + num_random,
            AttentionPattern::GlobalLocal { num_global, window_size } => num_global + window_size,
            AttentionPattern::Longformer { window_size, dilated_window, num_global } => {
                num_global + window_size + dilated_window * 2
            }
        }
    }

    /// Compute sparse attention: Q, K, V are (seq_len, head_dim) matrices
    pub fn sparse_attention(&self, q: &DenseMatrix, k: &DenseMatrix, v: &DenseMatrix, scale: f64) -> DenseMatrix {
        let seq_len = q.rows;
        let head_dim = q.cols;
        assert_eq!(k.rows, seq_len);
        assert_eq!(v.rows, seq_len);
        let mask = self.generate_mask(seq_len);

        // Compute attention scores only for masked positions
        // Group by query index for softmax
        let mut output = DenseMatrix::new(seq_len, head_dim);
        let mut i = 0;
        while i < mask.len() {
            let qi = mask[i].0;
            // Collect all keys for this query
            let mut scores = Vec::new();
            let mut key_indices = Vec::new();
            let mut j = i;
            while j < mask.len() && mask[j].0 == qi {
                let kj = mask[j].1;
                // Dot product Q[qi] · K[kj]
                let mut dot = 0.0;
                for d in 0..head_dim {
                    dot += q.get(qi, d) * k.get(kj, d);
                }
                scores.push(dot * scale);
                key_indices.push(kj);
                j += 1;
            }

            // Softmax
            let max_score = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_scores: Vec<f64> = scores.iter().map(|s| (s - max_score).exp()).collect();
            let sum_exp: f64 = exp_scores.iter().sum();

            // Weighted sum of values
            for (idx, &kj) in key_indices.iter().enumerate() {
                let weight = exp_scores[idx] / sum_exp;
                for d in 0..head_dim {
                    let curr = output.get(qi, d);
                    output.set(qi, d, curr + weight * v.get(kj, d));
                }
            }

            i = j;
        }

        output
    }
}

// ─── Low-Rank Decomposition ─────────────────────────────────────────────
// Detect and exploit low-rank structure: W ≈ U·V where U is n×r, V is r×m
// Matmul becomes O(r(n+m)) instead of O(nm)

/// Low-rank factorization of a matrix
#[derive(Debug, Clone)]
pub struct LowRankMatrix {
    /// Left factor: (rows × rank)
    pub u: DenseMatrix,
    /// Right factor: (rank × cols)
    pub v: DenseMatrix,
    pub rank: usize,
}

impl LowRankMatrix {
    /// Compute rank-r approximation using randomized SVD (Halko et al.)
    /// This is the standard algorithm used in practice: O(nmr) for rank-r approx.
    pub fn from_dense(matrix: &DenseMatrix, target_rank: usize) -> Self {
        let m = matrix.rows;
        let n = matrix.cols;
        let r = target_rank.min(m).min(n);

        // Step 1: Random projection - generate n×r random matrix
        let mut rng: u64 = 12345;
        let mut next_f64 = || -> f64 {
            rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
            (rng as f64) / (u64::MAX as f64) * 2.0 - 1.0
        };

        let mut omega = DenseMatrix::new(n, r);
        for i in 0..n {
            for j in 0..r {
                omega.set(i, j, next_f64());
            }
        }

        // Step 2: Y = A * Omega (m × r)
        let y = matrix.naive_matmul(&omega);

        // Step 3: QR factorization of Y via modified Gram-Schmidt
        let q = gram_schmidt(&y, r);

        // Step 4: B = Q^T * A (r × n)
        let qt = q.transpose();
        let b = qt.naive_matmul(matrix);

        Self { u: q, v: b, rank: r }
    }

    /// Apply low-rank matrix to vector. O(r * (rows + cols)) instead of O(rows * cols).
    pub fn forward(&self, x: &[f64]) -> Vec<f64> {
        // First multiply V * x (rank × 1)
        let vx = self.v.matvec(x);
        // Then multiply U * (V*x) (rows × 1)
        self.u.matvec(&vx)
    }

    /// Reconstruct full dense matrix (for verification)
    pub fn to_dense(&self) -> DenseMatrix {
        self.u.naive_matmul(&self.v)
    }

    /// Relative approximation error: ||A - UV||_F / ||A||_F
    pub fn error(&self, original: &DenseMatrix) -> f64 {
        let approx = self.to_dense();
        let mut diff_norm_sq = 0.0;
        for i in 0..original.rows {
            for j in 0..original.cols {
                let d = original.get(i, j) - approx.get(i, j);
                diff_norm_sq += d * d;
            }
        }
        diff_norm_sq.sqrt() / original.frobenius_norm()
    }

    /// Detect effective rank of a matrix (number of significant singular values)
    pub fn detect_rank(matrix: &DenseMatrix, tolerance: f64) -> usize {
        // Power iteration to estimate singular values
        let n = matrix.cols;
        let m = matrix.rows;
        let ata = matrix.transpose().naive_matmul(matrix); // n×n

        let mut rank = 0;
        let mut residual = ata.clone();
        let orig_norm = ata.frobenius_norm();

        for _ in 0..n.min(m) {
            // Power iteration for largest singular value of residual
            let mut v = vec![1.0 / (n as f64).sqrt(); n];
            for _ in 0..20 {
                let av = residual.matvec(&v);
                let norm: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();
                if norm < 1e-15 { break; }
                v = av.iter().map(|x| x / norm).collect();
            }

            let av = residual.matvec(&v);
            let sigma: f64 = av.iter().map(|x| x * x).sum::<f64>().sqrt();

            if sigma / orig_norm < tolerance { break; }
            rank += 1;

            // Deflate: subtract rank-1 component
            for i in 0..n {
                for j in 0..n {
                    let val = residual.get(i, j) - sigma * v[i] * v[j];
                    residual.set(i, j, val);
                }
            }
        }

        rank.max(1)
    }

    pub fn flops(&self) -> usize {
        // U*x: rows*rank, V*x: rank*cols
        self.u.rows * self.rank + self.rank * self.v.cols
    }
}

/// Modified Gram-Schmidt orthogonalization
fn gram_schmidt(a: &DenseMatrix, num_cols: usize) -> DenseMatrix {
    let m = a.rows;
    let r = num_cols.min(a.cols);
    let mut q = DenseMatrix::new(m, r);

    for j in 0..r {
        // Copy column j
        let mut col: Vec<f64> = (0..m).map(|i| a.get(i, j)).collect();

        // Orthogonalize against previous columns
        for k in 0..j {
            let mut dot = 0.0;
            for i in 0..m {
                dot += q.get(i, k) * col[i];
            }
            for i in 0..m {
                col[i] -= dot * q.get(i, k);
            }
        }

        // Normalize
        let norm: f64 = col.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-15 {
            for i in 0..m {
                q.set(i, j, col[i] / norm);
            }
        }
    }

    q
}

// ─── Auto-Select Engine ─────────────────────────────────────────────────
// Automatically choose the fastest path based on matrix properties

#[derive(Debug, Clone)]
pub enum MatrixStrategy {
    Dense,
    Monarch,
    Butterfly,
    LowRank { rank: usize },
}

/// Analyze a matrix and recommend the fastest multiplication strategy
pub fn auto_select_strategy(matrix: &DenseMatrix) -> MatrixStrategy {
    let n = matrix.rows;
    let m = matrix.cols;

    // Check if square and perfect square (Monarch candidate)
    if n == m {
        let b = (n as f64).sqrt() as usize;
        if b * b == n && n >= 16 {
            return MatrixStrategy::Monarch;
        }

        // Check if power of 2 (Butterfly candidate)
        if n.is_power_of_two() && n >= 8 {
            return MatrixStrategy::Butterfly;
        }
    }

    // Check effective rank
    let effective_rank = LowRankMatrix::detect_rank(matrix, 0.01);
    if effective_rank < n.min(m) / 4 {
        return MatrixStrategy::LowRank { rank: effective_rank };
    }

    MatrixStrategy::Dense
}

// ─── Builtins ───────────────────────────────────────────────────────────

use crate::interpreter::{Env, Value, FnDef};

fn value_to_matrix(v: &Value) -> Result<DenseMatrix, String> {
    match v {
        Value::Array(rows) => {
            if rows.is_empty() { return Err("empty matrix".into()); }
            let first_row = match &rows[0] {
                Value::Array(r) => r,
                _ => return Err("matrix must be array of arrays".into()),
            };
            let cols = first_row.len();
            let num_rows = rows.len();
            let mut data = Vec::with_capacity(num_rows * cols);
            for row in rows {
                match row {
                    Value::Array(r) => {
                        if r.len() != cols { return Err("rows must have equal length".into()); }
                        for v in r {
                            match v {
                                Value::Float(f) => data.push(*f),
                                Value::Int(n) => data.push(*n as f64),
                                _ => return Err("matrix elements must be numbers".into()),
                            }
                        }
                    }
                    _ => return Err("matrix must be array of arrays".into()),
                }
            }
            Ok(DenseMatrix::from_data(num_rows, cols, data))
        }
        _ => Err("expected matrix (array of arrays)".into()),
    }
}

fn matrix_to_value(m: &DenseMatrix) -> Value {
    Value::Array((0..m.rows).map(|i| {
        Value::Array((0..m.cols).map(|j| Value::Float(m.get(i, j))).collect())
    }).collect())
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("monarch_new".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 1 { return Err("monarch_new(n)".into()); }
        let n = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("n must be int".into()) };
        let m = MonarchMatrix::random(n, 42);
        // Return as nested array representation
        Ok(Value::Array(vec![
            Value::Int(m.block_size as i128),
            Value::Int(m.num_blocks as i128),
            Value::Int(m.flops() as i128),
        ]))
    }));

    env.functions.insert("monarch_forward".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 2 { return Err("monarch_forward(n, x)".into()); }
        let n = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("n must be int".into()) };
        let x: Vec<f64> = match &args[1] {
            Value::Array(arr) => arr.iter().map(|v| match v {
                Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0
            }).collect(),
            _ => return Err("x must be array".into()),
        };
        let m = MonarchMatrix::random(n, 42);
        let result = m.forward(&x);
        Ok(Value::Array(result.iter().map(|v| Value::Float(*v)).collect()))
    }));

    env.functions.insert("butterfly_forward".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 2 { return Err("butterfly_forward(n, x)".into()); }
        let n = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("n must be int".into()) };
        let x: Vec<f64> = match &args[1] {
            Value::Array(arr) => arr.iter().map(|v| match v {
                Value::Float(f) => *f, Value::Int(n) => *n as f64, _ => 0.0
            }).collect(),
            _ => return Err("x must be array".into()),
        };
        let bf = ButterflyMatrix::random(n, 42);
        let result = bf.forward(&x);
        Ok(Value::Array(result.iter().map(|v| Value::Float(*v)).collect()))
    }));

    env.functions.insert("sparse_attention".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 4 { return Err("sparse_attention(q, k, v, window_size)".into()); }
        let q = value_to_matrix(&args[0])?;
        let k = value_to_matrix(&args[1])?;
        let v = value_to_matrix(&args[2])?;
        let window = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("window must be int".into()) };
        let pattern = AttentionPattern::SlidingWindow { window_size: window };
        let scale = 1.0 / (q.cols as f64).sqrt();
        let out = pattern.sparse_attention(&q, &k, &v, scale);
        Ok(matrix_to_value(&out))
    }));

    env.functions.insert("low_rank_approx".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 2 { return Err("low_rank_approx(matrix, rank)".into()); }
        let m = value_to_matrix(&args[0])?;
        let r = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("rank must be int".into()) };
        let lr = LowRankMatrix::from_dense(&m, r);
        let approx = lr.to_dense();
        Ok(matrix_to_value(&approx))
    }));

    env.functions.insert("detect_rank".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 1 { return Err("detect_rank(matrix)".into()); }
        let m = value_to_matrix(&args[0])?;
        let rank = LowRankMatrix::detect_rank(&m, 0.01);
        Ok(Value::Int(rank as i128))
    }));

    env.functions.insert("auto_matrix_strategy".to_string(), FnDef::Builtin(|_env, args| {
        if args.len() != 1 { return Err("auto_matrix_strategy(matrix)".into()); }
        let m = value_to_matrix(&args[0])?;
        let strategy = auto_select_strategy(&m);
        let name = match strategy {
            MatrixStrategy::Dense => "dense",
            MatrixStrategy::Monarch => "monarch",
            MatrixStrategy::Butterfly => "butterfly",
            MatrixStrategy::LowRank { .. } => "low_rank",
        };
        Ok(Value::String(name.to_string()))
    }));
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_matmul() {
        let a = DenseMatrix::from_data(2, 2, vec![1.0, 2.0, 3.0, 4.0]);
        let b = DenseMatrix::from_data(2, 2, vec![5.0, 6.0, 7.0, 8.0]);
        let c = a.naive_matmul(&b);
        assert_eq!(c.get(0, 0), 19.0);
        assert_eq!(c.get(0, 1), 22.0);
        assert_eq!(c.get(1, 0), 43.0);
        assert_eq!(c.get(1, 1), 50.0);
    }

    #[test]
    fn test_monarch_identity() {
        let m = MonarchMatrix::new(16); // 4×4 blocks
        let x: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let y = m.forward(&x);
        // Identity monarch should preserve input (up to permutation)
        let sum_in: f64 = x.iter().sum();
        let sum_out: f64 = y.iter().sum();
        assert!((sum_in - sum_out).abs() < 1e-10, "Sum should be preserved");
    }

    #[test]
    fn test_monarch_random_produces_output() {
        let m = MonarchMatrix::random(16, 42);
        let x = vec![1.0; 16];
        let y = m.forward(&x);
        assert_eq!(y.len(), 16);
        // Output should not be all zeros
        assert!(y.iter().any(|v| v.abs() > 1e-10));
    }

    #[test]
    fn test_monarch_flops_subquadratic() {
        let m = MonarchMatrix::new(256); // 16×16 blocks
        // Monarch flops: 2 * n^1.5 = 2 * 256^1.5 = 8192
        // Dense flops: n^2 = 65536
        assert!(m.flops() < 256 * 256, "Monarch should be sub-quadratic: {} < {}", m.flops(), 256*256);
    }

    #[test]
    fn test_butterfly_identity() {
        let bf = ButterflyMatrix::new(8);
        let x: Vec<f64> = (0..8).map(|i| i as f64).collect();
        let y = bf.forward(&x);
        for i in 0..8 {
            assert!((x[i] - y[i]).abs() < 1e-10, "Identity butterfly should preserve input");
        }
    }

    #[test]
    fn test_butterfly_random_produces_output() {
        let bf = ButterflyMatrix::random(16, 42);
        let x = vec![1.0; 16];
        let y = bf.forward(&x);
        assert_eq!(y.len(), 16);
        assert!(y.iter().any(|v| v.abs() > 1e-10));
    }

    #[test]
    fn test_butterfly_flops_nlogn() {
        let bf = ButterflyMatrix::random(1024, 42);
        let nlogn = 1024 * 10 * 2; // n * log2(n) * 2
        assert!(bf.flops() <= nlogn + 1000, "Butterfly should be O(n log n): {} <= {}", bf.flops(), nlogn);
    }

    #[test]
    fn test_sliding_window_attention() {
        let seq_len = 8;
        let head_dim = 4;
        let q = DenseMatrix::from_data(seq_len, head_dim,
            (0..seq_len*head_dim).map(|i| (i as f64) * 0.1).collect());
        let k = q.clone();
        let v = q.clone();
        let pattern = AttentionPattern::SlidingWindow { window_size: 3 };
        let out = pattern.sparse_attention(&q, &k, &v, 0.5);
        assert_eq!(out.rows, seq_len);
        assert_eq!(out.cols, head_dim);
        // Output should not be all zeros
        assert!(out.data.iter().any(|v| v.abs() > 1e-10));
    }

    #[test]
    fn test_sliding_window_fewer_pairs_than_dense() {
        let dense = AttentionPattern::Dense;
        let window = AttentionPattern::SlidingWindow { window_size: 5 };
        let dense_pairs = dense.generate_mask(100);
        let window_pairs = window.generate_mask(100);
        assert!(window_pairs.len() < dense_pairs.len(),
            "Window should have fewer pairs: {} < {}", window_pairs.len(), dense_pairs.len());
    }

    #[test]
    fn test_global_local_attention() {
        let pattern = AttentionPattern::GlobalLocal { num_global: 2, window_size: 3 };
        let pairs = pattern.generate_mask(10);
        // Global tokens (0, 1) should attend to all 10 positions
        let global0_keys: Vec<_> = pairs.iter().filter(|(q, _)| *q == 0).collect();
        assert!(global0_keys.len() >= 10, "Global token should attend to all");
    }

    #[test]
    fn test_low_rank_approximation() {
        // Create a rank-2 matrix: each row is a linear combination of [1,0,0,0] and [0,1,0,0]
        let mut m = DenseMatrix::new(8, 4);
        for i in 0..8 {
            m.set(i, 0, (i as f64) * 1.0);
            m.set(i, 1, (i as f64) * 0.5);
        }
        let lr = LowRankMatrix::from_dense(&m, 2);
        let err = lr.error(&m);
        assert!(err < 0.1, "Rank-2 approximation of rank-2 matrix should be good: err={}", err);
    }

    #[test]
    fn test_detect_rank_low_rank_matrix() {
        // Rank-1 matrix: outer product of two vectors
        let mut m = DenseMatrix::new(8, 8);
        for i in 0..8 {
            for j in 0..8 {
                m.set(i, j, (i as f64 + 1.0) * (j as f64 + 1.0));
            }
        }
        let rank = LowRankMatrix::detect_rank(&m, 0.01);
        assert!(rank <= 2, "Rank-1 matrix should have detected rank <= 2, got {}", rank);
    }

    #[test]
    fn test_low_rank_forward() {
        let m = DenseMatrix::from_data(4, 4, vec![
            1.0, 2.0, 3.0, 4.0,
            2.0, 4.0, 6.0, 8.0,
            3.0, 6.0, 9.0, 12.0,
            4.0, 8.0, 12.0, 16.0,
        ]);
        let lr = LowRankMatrix::from_dense(&m, 1);
        let x = vec![1.0, 0.0, 0.0, 0.0];
        let y_approx = lr.forward(&x);
        let y_exact = m.matvec(&x);
        // Should be close for rank-1 matrix
        let err: f64 = y_approx.iter().zip(&y_exact).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        let norm: f64 = y_exact.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(err / norm < 0.5, "Low-rank forward should approximate well: err/norm={}", err/norm);
    }

    #[test]
    fn test_auto_strategy_monarch() {
        // 16×16 = perfect square → Monarch
        let m = DenseMatrix::new(16, 16);
        let strategy = auto_select_strategy(&m);
        assert!(matches!(strategy, MatrixStrategy::Monarch));
    }

    #[test]
    fn test_auto_strategy_butterfly() {
        // 8×8 = power of 2 but not perfect square → Butterfly
        let m = DenseMatrix::new(8, 8);
        let strategy = auto_select_strategy(&m);
        assert!(matches!(strategy, MatrixStrategy::Butterfly));
    }

    #[test]
    fn test_longformer_pattern() {
        let pattern = AttentionPattern::Longformer {
            window_size: 3, dilated_window: 2, num_global: 1,
        };
        let pairs = pattern.generate_mask(20);
        assert!(!pairs.is_empty());
        // Global token 0 should attend to all
        let global_keys: Vec<_> = pairs.iter().filter(|(q, _)| *q == 0).collect();
        assert!(global_keys.len() >= 20);
    }

    #[test]
    fn test_gram_schmidt_orthogonal() {
        let a = DenseMatrix::from_data(4, 2, vec![
            1.0, 0.5,
            0.0, 1.0,
            1.0, 0.0,
            0.0, 0.5,
        ]);
        let q = gram_schmidt(&a, 2);
        // Check orthogonality: Q^T * Q ≈ I
        let qtq = q.transpose().naive_matmul(&q);
        assert!((qtq.get(0, 0) - 1.0).abs() < 1e-10);
        assert!((qtq.get(1, 1) - 1.0).abs() < 1e-10);
        assert!((qtq.get(0, 1)).abs() < 1e-10);
    }
}
