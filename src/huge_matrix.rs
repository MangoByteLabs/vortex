/// Huge matrix engine for Vortex.
/// Sparse matrices (CSR/CSC/COO/BlockSparse), out-of-core disk tensors,
/// blocked/tiled matmul, Strassen's algorithm, flash/linear/sliding-window attention,
/// biological sequence alignment (Smith-Waterman, Needleman-Wunsch),
/// and optimized convolution (Winograd, FFT, depthwise, dilated).

use std::path::PathBuf;
use std::fs::{File, OpenOptions};
use std::io::{Read as IoRead, Write as IoWrite, Seek, SeekFrom};

// ─── Sparse Matrix Formats ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum SparseFormat {
    CSR { row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Vec<f64> },
    CSC { col_ptr: Vec<usize>, row_idx: Vec<usize>, values: Vec<f64> },
    COO { rows: Vec<usize>, cols: Vec<usize>, values: Vec<f64> },
    BlockSparse { blocks: Vec<(usize, usize, Vec<f64>)>, block_size: usize },
}

#[derive(Debug, Clone)]
pub struct SparseMatrix {
    pub format: SparseFormat,
    pub rows: usize,
    pub cols: usize,
    pub nnz: usize,
}

impl SparseMatrix {
    /// Create a CSR sparse matrix from raw components.
    pub fn from_csr(rows: usize, cols: usize, row_ptr: Vec<usize>, col_idx: Vec<usize>, values: Vec<f64>) -> Self {
        let nnz = values.len();
        SparseMatrix { format: SparseFormat::CSR { row_ptr, col_idx, values }, rows, cols, nnz }
    }

    /// Create a COO sparse matrix.
    pub fn from_coo(rows: usize, cols: usize, row_indices: Vec<usize>, col_indices: Vec<usize>, values: Vec<f64>) -> Self {
        let nnz = values.len();
        SparseMatrix { format: SparseFormat::COO { rows: row_indices, cols: col_indices, values }, rows, cols, nnz }
    }

    /// Create a CSC sparse matrix.
    pub fn from_csc(rows: usize, cols: usize, col_ptr: Vec<usize>, row_idx: Vec<usize>, values: Vec<f64>) -> Self {
        let nnz = values.len();
        SparseMatrix { format: SparseFormat::CSC { col_ptr, row_idx, values }, rows, cols, nnz }
    }

    /// Convert to COO format.
    pub fn to_coo(&self) -> SparseMatrix {
        match &self.format {
            SparseFormat::COO { .. } => self.clone(),
            SparseFormat::CSR { row_ptr, col_idx, values } => {
                let mut r = Vec::with_capacity(self.nnz);
                let mut c = Vec::with_capacity(self.nnz);
                for i in 0..self.rows {
                    for j in row_ptr[i]..row_ptr[i + 1] {
                        r.push(i);
                        c.push(col_idx[j]);
                    }
                }
                SparseMatrix::from_coo(self.rows, self.cols, r, c, values.clone())
            }
            SparseFormat::CSC { col_ptr, row_idx, values } => {
                let mut r = Vec::with_capacity(self.nnz);
                let mut c = Vec::with_capacity(self.nnz);
                for j in 0..self.cols {
                    for idx in col_ptr[j]..col_ptr[j + 1] {
                        r.push(row_idx[idx]);
                        c.push(j);
                    }
                }
                SparseMatrix::from_coo(self.rows, self.cols, r, c, values.clone())
            }
            SparseFormat::BlockSparse { blocks, block_size } => {
                let mut r = Vec::new();
                let mut c = Vec::new();
                let mut v = Vec::new();
                let bs = *block_size;
                for (br, bc, data) in blocks {
                    for bi in 0..bs {
                        for bj in 0..bs {
                            let val = data[bi * bs + bj];
                            if val != 0.0 {
                                r.push(br * bs + bi);
                                c.push(bc * bs + bj);
                                v.push(val);
                            }
                        }
                    }
                }
                let nnz = v.len();
                SparseMatrix { format: SparseFormat::COO { rows: r, cols: c, values: v }, rows: self.rows, cols: self.cols, nnz }
            }
        }
    }

    /// Convert to CSR format.
    pub fn to_csr(&self) -> SparseMatrix {
        let coo = self.to_coo();
        if let SparseFormat::COO { rows: r, cols: c, values: v } = &coo.format {
            // Sort by row then column
            let mut entries: Vec<(usize, usize, f64)> = r.iter().zip(c.iter()).zip(v.iter())
                .map(|((&ri, &ci), &vi)| (ri, ci, vi)).collect();
            entries.sort_by_key(|e| (e.0, e.1));

            let mut row_ptr = vec![0usize; self.rows + 1];
            let mut col_idx = Vec::with_capacity(self.nnz);
            let mut values = Vec::with_capacity(self.nnz);
            for (ri, ci, vi) in &entries {
                row_ptr[ri + 1] += 1;
                col_idx.push(*ci);
                values.push(*vi);
            }
            for i in 1..=self.rows {
                row_ptr[i] += row_ptr[i - 1];
            }
            SparseMatrix::from_csr(self.rows, self.cols, row_ptr, col_idx, values)
        } else {
            unreachable!()
        }
    }

    /// Convert to CSC format.
    pub fn to_csc(&self) -> SparseMatrix {
        let coo = self.to_coo();
        if let SparseFormat::COO { rows: r, cols: c, values: v } = &coo.format {
            let mut entries: Vec<(usize, usize, f64)> = r.iter().zip(c.iter()).zip(v.iter())
                .map(|((&ri, &ci), &vi)| (ri, ci, vi)).collect();
            entries.sort_by_key(|e| (e.1, e.0));

            let mut col_ptr = vec![0usize; self.cols + 1];
            let mut row_idx = Vec::with_capacity(self.nnz);
            let mut values = Vec::with_capacity(self.nnz);
            for (ri, ci, vi) in &entries {
                col_ptr[ci + 1] += 1;
                row_idx.push(*ri);
                values.push(*vi);
            }
            for j in 1..=self.cols {
                col_ptr[j] += col_ptr[j - 1];
            }
            SparseMatrix::from_csc(self.rows, self.cols, col_ptr, row_idx, values)
        } else {
            unreachable!()
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> SparseMatrix {
        let coo = self.to_coo();
        if let SparseFormat::COO { rows: r, cols: c, values: v } = coo.format {
            SparseMatrix::from_coo(self.cols, self.rows, c, r, v)
        } else {
            unreachable!()
        }
    }

    /// Convert to dense (row-major).
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = vec![0.0; self.rows * self.cols];
        let coo = self.to_coo();
        if let SparseFormat::COO { rows: r, cols: c, values: v } = &coo.format {
            for i in 0..v.len() {
                out[r[i] * self.cols + c[i]] += v[i];
            }
        }
        out
    }

    /// Create from dense with threshold.
    pub fn from_dense(data: &[f64], rows: usize, cols: usize, threshold: f64) -> Self {
        let mut ri = Vec::new();
        let mut ci = Vec::new();
        let mut vi = Vec::new();
        for r in 0..rows {
            for c in 0..cols {
                let val = data[r * cols + c];
                if val.abs() > threshold {
                    ri.push(r);
                    ci.push(c);
                    vi.push(val);
                }
            }
        }
        SparseMatrix::from_coo(rows, cols, ri, ci, vi)
    }
}

/// Sparse-dense matrix multiply (SpMM): sparse(M,K) * dense(K,N) -> dense(M,N).
pub fn sparse_dense_matmul(sparse: &SparseMatrix, dense: &[f64], dense_cols: usize) -> Vec<f64> {
    let m = sparse.rows;
    let mut result = vec![0.0; m * dense_cols];
    let csr = sparse.to_csr();
    if let SparseFormat::CSR { row_ptr, col_idx, values } = &csr.format {
        for i in 0..m {
            for idx in row_ptr[i]..row_ptr[i + 1] {
                let k = col_idx[idx];
                let v = values[idx];
                for j in 0..dense_cols {
                    result[i * dense_cols + j] += v * dense[k * dense_cols + j];
                }
            }
        }
    }
    result
}

/// Sparse-sparse multiply (SpGEMM): A(M,K) * B(K,N) -> C(M,N) in COO.
pub fn sparse_sparse_matmul(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    assert_eq!(a.cols, b.rows, "SpGEMM dimension mismatch");
    let a_csr = a.to_csr();
    let b_csr = b.to_csr();
    let mut rows_out = Vec::new();
    let mut cols_out = Vec::new();
    let mut vals_out = Vec::new();

    if let (SparseFormat::CSR { row_ptr: a_rp, col_idx: a_ci, values: a_v },
            SparseFormat::CSR { row_ptr: b_rp, col_idx: b_ci, values: b_v }) = (&a_csr.format, &b_csr.format) {
        for i in 0..a.rows {
            let mut acc = std::collections::HashMap::new();
            for a_idx in a_rp[i]..a_rp[i + 1] {
                let k = a_ci[a_idx];
                let a_val = a_v[a_idx];
                for b_idx in b_rp[k]..b_rp[k + 1] {
                    let j = b_ci[b_idx];
                    *acc.entry(j).or_insert(0.0) += a_val * b_v[b_idx];
                }
            }
            for (j, val) in acc {
                if val.abs() > 1e-15 {
                    rows_out.push(i);
                    cols_out.push(j);
                    vals_out.push(val);
                }
            }
        }
    }
    SparseMatrix::from_coo(a.rows, b.cols, rows_out, cols_out, vals_out)
}

/// Sparse addition: A + B (both must have same dimensions).
pub fn sparse_add(a: &SparseMatrix, b: &SparseMatrix) -> SparseMatrix {
    assert_eq!(a.rows, b.rows);
    assert_eq!(a.cols, b.cols);
    let a_coo = a.to_coo();
    let b_coo = b.to_coo();
    let mut acc = std::collections::HashMap::new();
    if let SparseFormat::COO { rows: ar, cols: ac, values: av } = &a_coo.format {
        for i in 0..av.len() {
            *acc.entry((ar[i], ac[i])).or_insert(0.0) += av[i];
        }
    }
    if let SparseFormat::COO { rows: br, cols: bc, values: bv } = &b_coo.format {
        for i in 0..bv.len() {
            *acc.entry((br[i], bc[i])).or_insert(0.0) += bv[i];
        }
    }
    let mut r = Vec::new();
    let mut c = Vec::new();
    let mut v = Vec::new();
    for ((ri, ci), val) in acc {
        if val.abs() > 1e-15 {
            r.push(ri);
            c.push(ci);
            v.push(val);
        }
    }
    SparseMatrix::from_coo(a.rows, a.cols, r, c, v)
}

// ─── Out-of-Core DiskTensor ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F64,
    F32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self { DType::F64 => 8, DType::F32 => 4 }
    }
}

#[derive(Debug, Clone)]
pub struct DiskTensor {
    pub path: PathBuf,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub chunk_size: usize,
}

impl DiskTensor {
    /// Create a new DiskTensor backed by a file, initialized to zero.
    pub fn zeros(path: PathBuf, shape: Vec<usize>, chunk_size: usize) -> std::io::Result<Self> {
        let total: usize = shape.iter().product();
        let mut f = File::create(&path)?;
        let zero = [0u8; 8];
        for _ in 0..total {
            f.write_all(&zero)?;
        }
        f.flush()?;
        Ok(DiskTensor { path, shape, dtype: DType::F64, chunk_size })
    }

    /// Write a dense vector to the file.
    pub fn write_data(&self, data: &[f64]) -> std::io::Result<()> {
        let mut f = File::create(&self.path)?;
        for &val in data {
            f.write_all(&val.to_le_bytes())?;
        }
        f.flush()
    }

    /// Read a chunk of rows starting at `row_offset`.
    pub fn read_chunk(&self, row_offset: usize, num_rows: usize) -> std::io::Result<Vec<f64>> {
        let cols = if self.shape.len() >= 2 { self.shape[1] } else { 1 };
        let elems = num_rows * cols;
        let byte_offset = (row_offset * cols * 8) as u64;
        let mut f = File::open(&self.path)?;
        f.seek(SeekFrom::Start(byte_offset))?;
        let mut buf = vec![0u8; elems * 8];
        let n = f.read(&mut buf)?;
        let elems_read = n / 8;
        let mut result = Vec::with_capacity(elems_read);
        for i in 0..elems_read {
            let bytes: [u8; 8] = buf[i * 8..(i + 1) * 8].try_into().unwrap();
            result.push(f64::from_le_bytes(bytes));
        }
        Ok(result)
    }

    /// Streaming reduction (sum or max) along axis 0.
    pub fn reduce_sum(&self) -> std::io::Result<Vec<f64>> {
        let rows = self.shape[0];
        let cols = if self.shape.len() >= 2 { self.shape[1] } else { 1 };
        let mut acc = vec![0.0; cols];
        let mut offset = 0;
        while offset < rows {
            let chunk_rows = (self.chunk_size).min(rows - offset);
            let chunk = self.read_chunk(offset, chunk_rows)?;
            for r in 0..chunk_rows {
                for c in 0..cols {
                    if r * cols + c < chunk.len() {
                        acc[c] += chunk[r * cols + c];
                    }
                }
            }
            offset += chunk_rows;
        }
        Ok(acc)
    }
}

/// Out-of-core blocked matmul: A(M,K) * B(K,N) -> C(M,N), all on disk.
pub fn disk_matmul(a: &DiskTensor, b: &DiskTensor, output_path: PathBuf) -> std::io::Result<DiskTensor> {
    let m = a.shape[0];
    let k = if a.shape.len() >= 2 { a.shape[1] } else { 1 };
    let n = if b.shape.len() >= 2 { b.shape[1] } else { 1 };
    let chunk = a.chunk_size.max(1);

    let out = DiskTensor::zeros(output_path.clone(), vec![m, n], chunk)?;

    // Read all of B into memory (B should fit; if not, this is a simplification)
    let b_data = b.read_chunk(0, b.shape[0])?;

    let mut row = 0;
    while row < m {
        let rows_this = chunk.min(m - row);
        let a_chunk = a.read_chunk(row, rows_this)?;
        let mut c_chunk = vec![0.0f64; rows_this * n];
        for i in 0..rows_this {
            for p in 0..k {
                let a_val = a_chunk[i * k + p];
                if a_val == 0.0 { continue; }
                for j in 0..n {
                    c_chunk[i * n + j] += a_val * b_data[p * n + j];
                }
            }
        }
        // Write chunk to output
        let byte_offset = (row * n * 8) as u64;
        let mut f = OpenOptions::new().write(true).open(&output_path)?;
        f.seek(SeekFrom::Start(byte_offset))?;
        for &val in &c_chunk {
            f.write_all(&val.to_le_bytes())?;
        }
        f.flush()?;
        row += rows_this;
    }
    Ok(out)
}

// ─── Blocked/Tiled Dense Matrix ─────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BlockedMatrix {
    pub blocks: Vec<Vec<f64>>,
    pub block_rows: usize,
    pub block_cols: usize,
    pub total_rows: usize,
    pub total_cols: usize,
}

impl BlockedMatrix {
    /// Create from dense row-major data with given block size.
    pub fn from_dense(data: &[f64], rows: usize, cols: usize, block_size: usize) -> Self {
        let br = (rows + block_size - 1) / block_size;
        let bc = (cols + block_size - 1) / block_size;
        let mut blocks = Vec::with_capacity(br * bc);
        for bi in 0..br {
            for bj in 0..bc {
                let mut block = vec![0.0; block_size * block_size];
                for i in 0..block_size {
                    let row = bi * block_size + i;
                    if row >= rows { break; }
                    for j in 0..block_size {
                        let col = bj * block_size + j;
                        if col >= cols { break; }
                        block[i * block_size + j] = data[row * cols + col];
                    }
                }
                blocks.push(block);
            }
        }
        BlockedMatrix { blocks, block_rows: br, block_cols: bc, total_rows: rows, total_cols: cols }
    }

    /// Convert back to dense.
    pub fn to_dense(&self, block_size: usize) -> Vec<f64> {
        let mut out = vec![0.0; self.total_rows * self.total_cols];
        for bi in 0..self.block_rows {
            for bj in 0..self.block_cols {
                let block = &self.blocks[bi * self.block_cols + bj];
                for i in 0..block_size {
                    let row = bi * block_size + i;
                    if row >= self.total_rows { break; }
                    for j in 0..block_size {
                        let col = bj * block_size + j;
                        if col >= self.total_cols { break; }
                        out[row * self.total_cols + col] = block[i * block_size + j];
                    }
                }
            }
        }
        out
    }
}

/// Blocked matmul: C = A * B, both blocked with same block_size.
pub fn blocked_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize, block_size: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    let bs = block_size;
    for bi in (0..m).step_by(bs) {
        for bj in (0..n).step_by(bs) {
            for bk in (0..k).step_by(bs) {
                let i_end = (bi + bs).min(m);
                let j_end = (bj + bs).min(n);
                let k_end = (bk + bs).min(k);
                for i in bi..i_end {
                    for p in bk..k_end {
                        let a_val = a[i * k + p];
                        for j in bj..j_end {
                            c[i * n + j] += a_val * b[p * n + j];
                        }
                    }
                }
            }
        }
    }
    c
}

/// Strassen's algorithm for square matrix multiply. Crossover to naive at threshold.
pub fn strassen_matmul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    if n <= 64 {
        return naive_matmul(a, b, n, n, n);
    }
    let h = n / 2;
    if n % 2 != 0 {
        // Pad to even
        let nn = n + 1;
        let mut ap = vec![0.0; nn * nn];
        let mut bp = vec![0.0; nn * nn];
        for i in 0..n {
            for j in 0..n {
                ap[i * nn + j] = a[i * n + j];
                bp[i * nn + j] = b[i * n + j];
            }
        }
        let cp = strassen_matmul(&ap, &bp, nn);
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                c[i * n + j] = cp[i * nn + j];
            }
        }
        return c;
    }

    let sub = |mat: &[f64], r0: usize, c0: usize| -> Vec<f64> {
        let mut s = vec![0.0; h * h];
        for i in 0..h { for j in 0..h { s[i * h + j] = mat[(r0 + i) * n + c0 + j]; } }
        s
    };
    let add_m = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x + y).collect()
    };
    let sub_m = |a: &[f64], b: &[f64]| -> Vec<f64> {
        a.iter().zip(b).map(|(x, y)| x - y).collect()
    };

    let a11 = sub(a, 0, 0); let a12 = sub(a, 0, h);
    let a21 = sub(a, h, 0); let a22 = sub(a, h, h);
    let b11 = sub(b, 0, 0); let b12 = sub(b, 0, h);
    let b21 = sub(b, h, 0); let b22 = sub(b, h, h);

    let m1 = strassen_matmul(&add_m(&a11, &a22), &add_m(&b11, &b22), h);
    let m2 = strassen_matmul(&add_m(&a21, &a22), &b11, h);
    let m3 = strassen_matmul(&a11, &sub_m(&b12, &b22), h);
    let m4 = strassen_matmul(&a22, &sub_m(&b21, &b11), h);
    let m5 = strassen_matmul(&add_m(&a11, &a12), &b22, h);
    let m6 = strassen_matmul(&sub_m(&a21, &a11), &add_m(&b11, &b12), h);
    let m7 = strassen_matmul(&sub_m(&a12, &a22), &add_m(&b21, &b22), h);

    let c11 = add_m(&sub_m(&add_m(&m1, &m4), &m5), &m7);
    let c12 = add_m(&m3, &m5);
    let c21 = add_m(&m2, &m4);
    let c22 = add_m(&sub_m(&add_m(&m1, &m3), &m2), &m6);

    let mut c = vec![0.0; n * n];
    for i in 0..h {
        for j in 0..h {
            c[i * n + j] = c11[i * h + j];
            c[i * n + h + j] = c12[i * h + j];
            c[(h + i) * n + j] = c21[i * h + j];
            c[(h + i) * n + h + j] = c22[i * h + j];
        }
    }
    c
}

fn naive_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += av * b[p * n + j];
            }
        }
    }
    c
}

/// Auto-select best matmul algorithm.
pub fn huge_matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    if m == k && k == n && m >= 512 {
        strassen_matmul(a, b, m)
    } else if m >= 64 || n >= 64 || k >= 64 {
        blocked_matmul(a, b, m, k, n, 64)
    } else {
        naive_matmul(a, b, m, k, n)
    }
}

// ─── Attention Mechanisms ────────────────────────────────────────────────────

/// Flash attention: O(N) memory, tiled computation.
/// Q, K, V are (seq_len, d) in row-major. Returns output (seq_len, d).
pub fn flash_attention(q: &[f64], k: &[f64], v: &[f64], seq_len: usize, d: usize, block_size: usize) -> Vec<f64> {
    let bs = block_size.max(1).min(seq_len);
    let mut output = vec![0.0; seq_len * d];
    let mut row_max = vec![f64::NEG_INFINITY; seq_len];
    let mut row_sum = vec![0.0; seq_len];

    let num_blocks = (seq_len + bs - 1) / bs;

    for bj in 0..num_blocks {
        let j_start = bj * bs;
        let j_end = (j_start + bs).min(seq_len);

        for i in 0..seq_len {
            // Compute attention scores for Q[i] against K[j_start..j_end]
            let mut scores = Vec::with_capacity(j_end - j_start);
            for j in j_start..j_end {
                let mut dot = 0.0;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                dot /= (d as f64).sqrt();
                scores.push(dot);
            }

            // Online softmax update
            let block_max = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let new_max = row_max[i].max(block_max);

            // Rescale previous accumulator
            if row_sum[i] > 0.0 {
                let scale = (row_max[i] - new_max).exp();
                row_sum[i] *= scale;
                for dd in 0..d {
                    output[i * d + dd] *= scale;
                }
            }

            // Add new block contribution
            let mut block_sum = 0.0;
            for (idx, &s) in scores.iter().enumerate() {
                let w = (s - new_max).exp();
                block_sum += w;
                let j = j_start + idx;
                for dd in 0..d {
                    output[i * d + dd] += w * v[j * d + dd];
                }
            }

            row_max[i] = new_max;
            row_sum[i] += block_sum;
        }
    }

    // Normalize
    for i in 0..seq_len {
        if row_sum[i] > 0.0 {
            for dd in 0..d {
                output[i * d + dd] /= row_sum[i];
            }
        }
    }
    output
}

/// Naive attention for reference: softmax(Q*K^T / sqrt(d)) * V.
pub fn naive_attention(q: &[f64], k: &[f64], v: &[f64], seq_len: usize, d: usize) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let mut output = vec![0.0; seq_len * d];
    for i in 0..seq_len {
        // Compute scores
        let mut scores = vec![0.0; seq_len];
        let mut max_s = f64::NEG_INFINITY;
        for j in 0..seq_len {
            let mut dot = 0.0;
            for dd in 0..d {
                dot += q[i * d + dd] * k[j * d + dd];
            }
            scores[j] = dot * scale;
            if scores[j] > max_s { max_s = scores[j]; }
        }
        // Softmax
        let mut sum = 0.0;
        for j in 0..seq_len {
            scores[j] = (scores[j] - max_s).exp();
            sum += scores[j];
        }
        for j in 0..seq_len {
            scores[j] /= sum;
        }
        // Weighted sum
        for j in 0..seq_len {
            for dd in 0..d {
                output[i * d + dd] += scores[j] * v[j * d + dd];
            }
        }
    }
    output
}

/// Linear attention using random Fourier features.
/// Approximates softmax attention in O(N*d) time/memory.
pub fn linear_attention(q: &[f64], k: &[f64], v: &[f64], seq_len: usize, d: usize) -> Vec<f64> {
    // Use ELU+1 feature map: phi(x) = elu(x) + 1
    let phi = |x: f64| -> f64 { if x >= 0.0 { x + 1.0 } else { x.exp() } };

    // Compute phi(K)^T * V -> (d, d) matrix
    let mut kv = vec![0.0; d * d];
    let mut k_sum = vec![0.0; d]; // sum of phi(K) for normalization
    for j in 0..seq_len {
        let mut phi_k = vec![0.0; d];
        for dd in 0..d {
            phi_k[dd] = phi(k[j * d + dd]);
            k_sum[dd] += phi_k[dd];
        }
        for dd1 in 0..d {
            for dd2 in 0..d {
                kv[dd1 * d + dd2] += phi_k[dd1] * v[j * d + dd2];
            }
        }
    }

    // For each query, compute phi(Q) * (phi(K)^T * V) / (phi(Q) * sum(phi(K)))
    let mut output = vec![0.0; seq_len * d];
    for i in 0..seq_len {
        let mut phi_q = vec![0.0; d];
        for dd in 0..d {
            phi_q[dd] = phi(q[i * d + dd]);
        }
        let mut denom = 0.0;
        for dd in 0..d {
            denom += phi_q[dd] * k_sum[dd];
        }
        if denom.abs() < 1e-10 { denom = 1e-10; }
        for dd2 in 0..d {
            let mut val = 0.0;
            for dd1 in 0..d {
                val += phi_q[dd1] * kv[dd1 * d + dd2];
            }
            output[i * d + dd2] = val / denom;
        }
    }
    output
}

/// Sliding window attention: each query only attends to a window of keys.
pub fn sliding_window_attention(q: &[f64], k: &[f64], v: &[f64], seq_len: usize, d: usize, window_size: usize) -> Vec<f64> {
    let scale = 1.0 / (d as f64).sqrt();
    let half_w = window_size / 2;
    let mut output = vec![0.0; seq_len * d];

    for i in 0..seq_len {
        let j_start = if i >= half_w { i - half_w } else { 0 };
        let j_end = (i + half_w + 1).min(seq_len);

        let mut scores = Vec::with_capacity(j_end - j_start);
        let mut max_s = f64::NEG_INFINITY;
        for j in j_start..j_end {
            let mut dot = 0.0;
            for dd in 0..d {
                dot += q[i * d + dd] * k[j * d + dd];
            }
            let s = dot * scale;
            if s > max_s { max_s = s; }
            scores.push(s);
        }
        let mut sum = 0.0;
        for s in &mut scores {
            *s = (*s - max_s).exp();
            sum += *s;
        }
        for s in &mut scores { *s /= sum; }

        for (idx, j) in (j_start..j_end).enumerate() {
            for dd in 0..d {
                output[i * d + dd] += scores[idx] * v[j * d + dd];
            }
        }
    }
    output
}

// ─── Biological Sequence Alignment ──────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct AlignmentResult {
    pub score: f64,
    pub aligned_a: String,
    pub aligned_b: String,
}

#[derive(Debug, Clone)]
pub struct ScoringMatrix {
    pub match_score: f64,
    pub mismatch_penalty: f64,
    pub gap_penalty: f64,
}

impl Default for ScoringMatrix {
    fn default() -> Self {
        ScoringMatrix { match_score: 2.0, mismatch_penalty: -1.0, gap_penalty: -1.0 }
    }
}

/// Smith-Waterman local alignment.
pub fn smith_waterman(seq_a: &str, seq_b: &str, scoring: &ScoringMatrix) -> AlignmentResult {
    let a: Vec<char> = seq_a.chars().collect();
    let b: Vec<char> = seq_b.chars().collect();
    let m = a.len();
    let n = b.len();
    let mut h = vec![vec![0.0f64; n + 1]; m + 1];
    let mut max_score = 0.0;
    let mut max_i = 0;
    let mut max_j = 0;

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] { scoring.match_score } else { scoring.mismatch_penalty };
            h[i][j] = (h[i - 1][j - 1] + s)
                .max(h[i - 1][j] + scoring.gap_penalty)
                .max(h[i][j - 1] + scoring.gap_penalty)
                .max(0.0);
            if h[i][j] > max_score {
                max_score = h[i][j];
                max_i = i;
                max_j = j;
            }
        }
    }

    // Traceback
    let mut aligned_a = String::new();
    let mut aligned_b = String::new();
    let mut i = max_i;
    let mut j = max_j;
    while i > 0 && j > 0 && h[i][j] > 0.0 {
        let s = if a[i - 1] == b[j - 1] { scoring.match_score } else { scoring.mismatch_penalty };
        if (h[i][j] - (h[i - 1][j - 1] + s)).abs() < 1e-10 {
            aligned_a.insert(0, a[i - 1]);
            aligned_b.insert(0, b[j - 1]);
            i -= 1;
            j -= 1;
        } else if (h[i][j] - (h[i - 1][j] + scoring.gap_penalty)).abs() < 1e-10 {
            aligned_a.insert(0, a[i - 1]);
            aligned_b.insert(0, '-');
            i -= 1;
        } else {
            aligned_a.insert(0, '-');
            aligned_b.insert(0, b[j - 1]);
            j -= 1;
        }
    }

    AlignmentResult { score: max_score, aligned_a, aligned_b }
}

/// Needleman-Wunsch global alignment.
pub fn needleman_wunsch(seq_a: &str, seq_b: &str, scoring: &ScoringMatrix) -> AlignmentResult {
    let a: Vec<char> = seq_a.chars().collect();
    let b: Vec<char> = seq_b.chars().collect();
    let m = a.len();
    let n = b.len();
    let mut h = vec![vec![0.0f64; n + 1]; m + 1];

    for i in 1..=m { h[i][0] = h[i - 1][0] + scoring.gap_penalty; }
    for j in 1..=n { h[0][j] = h[0][j - 1] + scoring.gap_penalty; }

    for i in 1..=m {
        for j in 1..=n {
            let s = if a[i - 1] == b[j - 1] { scoring.match_score } else { scoring.mismatch_penalty };
            h[i][j] = (h[i - 1][j - 1] + s)
                .max(h[i - 1][j] + scoring.gap_penalty)
                .max(h[i][j - 1] + scoring.gap_penalty);
        }
    }

    let mut aligned_a = String::new();
    let mut aligned_b = String::new();
    let mut i = m;
    let mut j = n;
    while i > 0 || j > 0 {
        if i > 0 && j > 0 {
            let s = if a[i - 1] == b[j - 1] { scoring.match_score } else { scoring.mismatch_penalty };
            if (h[i][j] - (h[i - 1][j - 1] + s)).abs() < 1e-10 {
                aligned_a.insert(0, a[i - 1]);
                aligned_b.insert(0, b[j - 1]);
                i -= 1;
                j -= 1;
                continue;
            }
        }
        if i > 0 && (h[i][j] - (h[i - 1][j] + scoring.gap_penalty)).abs() < 1e-10 {
            aligned_a.insert(0, a[i - 1]);
            aligned_b.insert(0, '-');
            i -= 1;
        } else {
            aligned_a.insert(0, '-');
            aligned_b.insert(0, b[j - 1]);
            j -= 1;
        }
    }

    AlignmentResult { score: h[m][n], aligned_a, aligned_b }
}

/// Pairwise distance matrix for a set of sequences (edit distance).
pub fn pairwise_distance(sequences: &[&str]) -> Vec<Vec<f64>> {
    let n = sequences.len();
    let mut dist = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in i + 1..n {
            let d = edit_distance(sequences[i], sequences[j]) as f64;
            dist[i][j] = d;
            dist[j][i] = d;
        }
    }
    dist
}

fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let m = a.len();
    let n = b.len();
    let mut dp = vec![vec![0usize; n + 1]; m + 1];
    for i in 0..=m { dp[i][0] = i; }
    for j in 0..=n { dp[0][j] = j; }
    for i in 1..=m {
        for j in 1..=n {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            dp[i][j] = (dp[i - 1][j] + 1).min(dp[i][j - 1] + 1).min(dp[i - 1][j - 1] + cost);
        }
    }
    dp[m][n]
}

/// Neighbor-joining tree construction. Returns tree as list of (left, right, distance) merges.
pub fn neighbor_joining(dist_matrix: &[Vec<f64>]) -> Vec<(usize, usize, f64)> {
    let n = dist_matrix.len();
    if n < 2 { return vec![]; }

    let mut d: Vec<Vec<f64>> = dist_matrix.to_vec();
    let mut active: Vec<usize> = (0..n).collect();
    let mut merges = Vec::new();
    let mut next_id = n;

    while active.len() > 2 {
        let m = active.len();
        // Compute Q matrix
        let row_sums: Vec<f64> = (0..m).map(|i| (0..m).map(|j| d[active[i]][active[j]]).sum::<f64>()).collect();
        let mut min_q = f64::INFINITY;
        let mut min_i = 0;
        let mut min_j = 1;
        for i in 0..m {
            for j in i + 1..m {
                let q = (m as f64 - 2.0) * d[active[i]][active[j]] - row_sums[i] - row_sums[j];
                if q < min_q {
                    min_q = q;
                    min_i = i;
                    min_j = j;
                }
            }
        }
        let ai = active[min_i];
        let aj = active[min_j];
        let dist_ij = d[ai][aj];
        merges.push((ai, aj, dist_ij));

        // Add new node distances
        let new_id = next_id;
        next_id += 1;
        // Extend distance matrix
        let new_size = d.len() + 1;
        for row in &mut d { row.resize(new_size, 0.0); }
        d.push(vec![0.0; new_size]);

        for &ak in &active {
            if ak == ai || ak == aj { continue; }
            let new_d = (d[ai][ak] + d[aj][ak] - dist_ij) / 2.0;
            d[new_id][ak] = new_d;
            d[ak][new_id] = new_d;
        }

        active.retain(|&x| x != ai && x != aj);
        active.push(new_id);
    }

    if active.len() == 2 {
        merges.push((active[0], active[1], d[active[0]][active[1]]));
    }
    merges
}

// ─── Convolution Engine ─────────────────────────────────────────────────────

/// Winograd F(2,3) convolution for 3x3 kernels.
/// Input: (H, W), Kernel: (3, 3). Output: (H-2, W-2).
pub fn winograd_conv2d(input: &[f64], ih: usize, iw: usize, kernel: &[f64]) -> Vec<f64> {
    if ih < 3 || iw < 3 { return vec![]; }
    let oh = ih - 2;
    let ow = iw - 2;
    let mut output = vec![0.0; oh * ow];

    // Winograd transform matrices for F(2,3)
    // BT = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
    // G = [[1,0,0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0,0,1]]
    // AT = [[1,1,1,0],[0,1,-1,-1]]

    // Transform kernel: U = G * g * G^T (done once)
    let g = kernel;
    let mut u = [0.0f64; 16]; // 4x4
    // G * g (4x3 * 3x3 = 4x3)
    let mut gg = [0.0f64; 12];
    // G rows: [1,0,0], [0.5,0.5,0.5], [0.5,-0.5,0.5], [0,0,1]
    let g_mat = [[1.0,0.0,0.0],[0.5,0.5,0.5],[0.5,-0.5,0.5],[0.0,0.0,1.0]];
    for i in 0..4 {
        for j in 0..3 {
            let mut s = 0.0;
            for k in 0..3 { s += g_mat[i][k] * g[k * 3 + j]; }
            gg[i * 3 + j] = s;
        }
    }
    // gg * G^T (4x3 * 3x4 = 4x4)
    let gt = [[1.0,0.5,0.5,0.0],[0.0,0.5,-0.5,0.0],[0.0,0.5,0.5,1.0]];
    for i in 0..4 {
        for j in 0..4 {
            let mut s = 0.0;
            for k in 0..3 { s += gg[i * 3 + k] * gt[k][j]; }
            u[i * 4 + j] = s;
        }
    }

    // Process 2x2 output tiles
    for ti in (0..oh).step_by(2) {
        for tj in (0..ow).step_by(2) {
            // Extract 4x4 input tile
            let mut d = [0.0f64; 16];
            for di in 0..4 {
                for dj in 0..4 {
                    let r = ti + di;
                    let c = tj + dj;
                    if r < ih && c < iw { d[di * 4 + dj] = input[r * iw + c]; }
                }
            }

            // BT * d * B
            // BT rows: [1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]
            let mut bd = [0.0f64; 16];
            for j in 0..4 {
                bd[0 * 4 + j] = d[0 * 4 + j] - d[2 * 4 + j];
                bd[1 * 4 + j] = d[1 * 4 + j] + d[2 * 4 + j];
                bd[2 * 4 + j] = -d[1 * 4 + j] + d[2 * 4 + j];
                bd[3 * 4 + j] = d[1 * 4 + j] - d[3 * 4 + j];
            }
            let mut v_tile = [0.0f64; 16];
            for i in 0..4 {
                v_tile[i * 4 + 0] = bd[i * 4 + 0] - bd[i * 4 + 2];
                v_tile[i * 4 + 1] = bd[i * 4 + 1] + bd[i * 4 + 2];
                v_tile[i * 4 + 2] = -bd[i * 4 + 1] + bd[i * 4 + 2];
                v_tile[i * 4 + 3] = bd[i * 4 + 1] - bd[i * 4 + 3];
            }

            // Element-wise multiply
            let mut m_tile = [0.0f64; 16];
            for idx in 0..16 { m_tile[idx] = v_tile[idx] * u[idx]; }

            // AT * m * A
            // AT rows: [1,1,1,0],[0,1,-1,-1]
            let mut am = [0.0f64; 8]; // 2x4
            for j in 0..4 {
                am[0 * 4 + j] = m_tile[0 * 4 + j] + m_tile[1 * 4 + j] + m_tile[2 * 4 + j];
                am[1 * 4 + j] = m_tile[1 * 4 + j] - m_tile[2 * 4 + j] - m_tile[3 * 4 + j];
            }
            let mut out_tile = [0.0f64; 4]; // 2x2
            for i in 0..2 {
                out_tile[i * 2 + 0] = am[i * 4 + 0] + am[i * 4 + 1] + am[i * 4 + 2];
                out_tile[i * 2 + 1] = am[i * 4 + 1] - am[i * 4 + 2] - am[i * 4 + 3];
            }

            for di in 0..2 {
                for dj in 0..2 {
                    let r = ti + di;
                    let c = tj + dj;
                    if r < oh && c < ow {
                        output[r * ow + c] = out_tile[di * 2 + dj];
                    }
                }
            }
        }
    }
    output
}

/// Naive 2D convolution for reference.
pub fn naive_conv2d(input: &[f64], ih: usize, iw: usize, kernel: &[f64], kh: usize, kw: usize) -> Vec<f64> {
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut output = vec![0.0; oh * ow];
    for i in 0..oh {
        for j in 0..ow {
            let mut sum = 0.0;
            for ki in 0..kh {
                for kj in 0..kw {
                    sum += input[(i + ki) * iw + (j + kj)] * kernel[ki * kw + kj];
                }
            }
            output[i * ow + j] = sum;
        }
    }
    output
}

/// FFT-based convolution for large kernels (1D).
pub fn fft_conv(input: &[f64], kernel: &[f64]) -> Vec<f64> {
    let n = input.len();
    let k = kernel.len();
    if k == 0 || n == 0 { return vec![]; }
    let out_len = n + k - 1;
    // Pad to power of 2
    let fft_len = out_len.next_power_of_two();

    let mut a_re = vec![0.0; fft_len];
    let mut a_im = vec![0.0; fft_len];
    let mut b_re = vec![0.0; fft_len];
    let mut b_im = vec![0.0; fft_len];

    for i in 0..n { a_re[i] = input[i]; }
    for i in 0..k { b_re[i] = kernel[i]; }

    fft_inplace(&mut a_re, &mut a_im, false);
    fft_inplace(&mut b_re, &mut b_im, false);

    // Pointwise multiply
    for i in 0..fft_len {
        let re = a_re[i] * b_re[i] - a_im[i] * b_im[i];
        let im = a_re[i] * b_im[i] + a_im[i] * b_re[i];
        a_re[i] = re;
        a_im[i] = im;
    }

    fft_inplace(&mut a_re, &mut a_im, true);

    a_re.truncate(out_len);
    a_re
}

fn fft_inplace(re: &mut [f64], im: &mut [f64], inverse: bool) {
    let n = re.len();
    if n <= 1 { return; }
    // Bit-reversal permutation
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 { j ^= bit; bit >>= 1; }
        j ^= bit;
        if i < j { re.swap(i, j); im.swap(i, j); }
    }
    // Cooley-Tukey
    let mut len = 2;
    while len <= n {
        let half = len / 2;
        let angle = if inverse { 2.0 * std::f64::consts::PI / len as f64 } else { -2.0 * std::f64::consts::PI / len as f64 };
        let w_re = angle.cos();
        let w_im = angle.sin();
        let mut i = 0;
        while i < n {
            let mut wr = 1.0;
            let mut wi = 0.0;
            for k in 0..half {
                let tr = wr * re[i + half + k] - wi * im[i + half + k];
                let ti = wr * im[i + half + k] + wi * re[i + half + k];
                re[i + half + k] = re[i + k] - tr;
                im[i + half + k] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
                let new_wr = wr * w_re - wi * w_im;
                wi = wr * w_im + wi * w_re;
                wr = new_wr;
            }
            i += len;
        }
        len <<= 1;
    }
    if inverse {
        let inv_n = 1.0 / n as f64;
        for i in 0..n { re[i] *= inv_n; im[i] *= inv_n; }
    }
}

/// Depthwise separable convolution (1D per channel).
pub fn depthwise_conv2d(input: &[f64], ih: usize, iw: usize, channels: usize, kernel: &[f64], kh: usize, kw: usize) -> Vec<f64> {
    let oh = ih - kh + 1;
    let ow = iw - kw + 1;
    let mut output = vec![0.0; channels * oh * ow];
    for ch in 0..channels {
        let k_off = ch * kh * kw;
        for i in 0..oh {
            for j in 0..ow {
                let mut sum = 0.0;
                for ki in 0..kh {
                    for kj in 0..kw {
                        sum += input[ch * ih * iw + (i + ki) * iw + (j + kj)] * kernel[k_off + ki * kw + kj];
                    }
                }
                output[ch * oh * ow + i * ow + j] = sum;
            }
        }
    }
    output
}

/// Dilated (atrous) convolution.
pub fn dilated_conv(input: &[f64], ih: usize, iw: usize, kernel: &[f64], kh: usize, kw: usize, dilation: usize) -> Vec<f64> {
    let eff_kh = (kh - 1) * dilation + 1;
    let eff_kw = (kw - 1) * dilation + 1;
    if ih < eff_kh || iw < eff_kw { return vec![]; }
    let oh = ih - eff_kh + 1;
    let ow = iw - eff_kw + 1;
    let mut output = vec![0.0; oh * ow];
    for i in 0..oh {
        for j in 0..ow {
            let mut sum = 0.0;
            for ki in 0..kh {
                for kj in 0..kw {
                    sum += input[(i + ki * dilation) * iw + (j + kj * dilation)] * kernel[ki * kw + kj];
                }
            }
            output[i * ow + j] = sum;
        }
    }
    output
}

// ─── Interpreter Builtins ──────────────────────────────────────────────────────

use crate::interpreter::{Env, Value, FnDef};

fn extract_f64_array(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => {
            arr.iter().map(|x| match x {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("expected numeric array".to_string()),
            }).collect()
        }
        _ => Err("expected array".to_string()),
    }
}

fn f64_to_value_array(data: &[f64]) -> Value {
    Value::Array(data.iter().map(|&x| Value::Float(x)).collect())
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("hm_sparse_from_dense".into(), FnDef::Builtin(|_env, args| {
        // args: data_array, rows, cols, threshold
        if args.len() < 4 { return Err("hm_sparse_from_dense(data, rows, cols, threshold)".into()); }
        let data = extract_f64_array(&args[0])?;
        let rows = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("rows must be int".into()) };
        let cols = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("cols must be int".into()) };
        let threshold = match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.0 };
        let sparse = SparseMatrix::from_dense(&data, rows, cols, threshold);
        Ok(Value::Int(sparse.nnz as i128))
    }));

    env.functions.insert("hm_sparse_matmul".into(), FnDef::Builtin(|_env, args| {
        // args: sparse_data, sparse_rows_idx, sparse_cols_idx, sp_rows, sp_cols, dense_data, dense_cols
        if args.len() < 7 { return Err("hm_sparse_matmul requires 7 args".into()); }
        let values = extract_f64_array(&args[0])?;
        let row_idx: Vec<usize> = extract_f64_array(&args[1])?.iter().map(|&x| x as usize).collect();
        let col_idx: Vec<usize> = extract_f64_array(&args[2])?.iter().map(|&x| x as usize).collect();
        let sp_rows = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("sp_rows must be int".into()) };
        let sp_cols = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("sp_cols must be int".into()) };
        let dense = extract_f64_array(&args[5])?;
        let dense_cols = match &args[6] { Value::Int(i) => *i as usize, _ => return Err("dense_cols must be int".into()) };
        let sparse = SparseMatrix::from_coo(sp_rows, sp_cols, row_idx, col_idx, values);
        let result = sparse_dense_matmul(&sparse, &dense, dense_cols);
        Ok(f64_to_value_array(&result))
    }));

    env.functions.insert("hm_sparse_nnz".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 4 { return Err("hm_sparse_nnz(data, rows, cols, threshold)".into()); }
        let data = extract_f64_array(&args[0])?;
        let rows = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("rows must be int".into()) };
        let cols = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("cols must be int".into()) };
        let threshold = match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => 0.0 };
        let sparse = SparseMatrix::from_dense(&data, rows, cols, threshold);
        Ok(Value::Int(sparse.nnz as i128))
    }));

    env.functions.insert("hm_flash_attention".into(), FnDef::Builtin(|_env, args| {
        // args: Q, K, V, seq_len, d, block_size
        if args.len() < 6 { return Err("hm_flash_attention(Q, K, V, seq_len, d, block_size)".into()); }
        let q = extract_f64_array(&args[0])?;
        let k = extract_f64_array(&args[1])?;
        let v = extract_f64_array(&args[2])?;
        let seq_len = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("seq_len must be int".into()) };
        let d = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("d must be int".into()) };
        let bs = match &args[5] { Value::Int(i) => *i as usize, _ => return Err("block_size must be int".into()) };
        let out = flash_attention(&q, &k, &v, seq_len, d, bs);
        Ok(f64_to_value_array(&out))
    }));

    env.functions.insert("hm_linear_attention".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 5 { return Err("hm_linear_attention(Q, K, V, seq_len, d)".into()); }
        let q = extract_f64_array(&args[0])?;
        let k = extract_f64_array(&args[1])?;
        let v = extract_f64_array(&args[2])?;
        let seq_len = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("seq_len must be int".into()) };
        let d = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("d must be int".into()) };
        let out = linear_attention(&q, &k, &v, seq_len, d);
        Ok(f64_to_value_array(&out))
    }));

    env.functions.insert("hm_huge_matmul".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 5 { return Err("hm_huge_matmul(a, b, m, k, n)".into()); }
        let a = extract_f64_array(&args[0])?;
        let b = extract_f64_array(&args[1])?;
        let m = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("m must be int".into()) };
        let k = match &args[3] { Value::Int(i) => *i as usize, _ => return Err("k must be int".into()) };
        let n = match &args[4] { Value::Int(i) => *i as usize, _ => return Err("n must be int".into()) };
        let out = huge_matmul(&a, &b, m, k, n);
        Ok(f64_to_value_array(&out))
    }));

    env.functions.insert("hm_smith_waterman".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("hm_smith_waterman(seq_a, seq_b)".into()); }
        let a = match &args[0] { Value::String(s) => s.clone(), _ => return Err("seq_a must be string".into()) };
        let b = match &args[1] { Value::String(s) => s.clone(), _ => return Err("seq_b must be string".into()) };
        let scoring = ScoringMatrix::default();
        let result = smith_waterman(&a, &b, &scoring);
        Ok(Value::Float(result.score))
    }));

    env.functions.insert("hm_needleman_wunsch".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("hm_needleman_wunsch(seq_a, seq_b)".into()); }
        let a = match &args[0] { Value::String(s) => s.clone(), _ => return Err("seq_a must be string".into()) };
        let b = match &args[1] { Value::String(s) => s.clone(), _ => return Err("seq_b must be string".into()) };
        let scoring = ScoringMatrix::default();
        let result = needleman_wunsch(&a, &b, &scoring);
        Ok(Value::Float(result.score))
    }));

    env.functions.insert("hm_winograd_conv".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 4 { return Err("hm_winograd_conv(input, ih, iw, kernel_3x3)".into()); }
        let input = extract_f64_array(&args[0])?;
        let ih = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("ih must be int".into()) };
        let iw = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("iw must be int".into()) };
        let kernel = extract_f64_array(&args[3])?;
        let out = winograd_conv2d(&input, ih, iw, &kernel);
        Ok(f64_to_value_array(&out))
    }));

    env.functions.insert("hm_fft_conv".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 2 { return Err("hm_fft_conv(input, kernel)".into()); }
        let input = extract_f64_array(&args[0])?;
        let kernel = extract_f64_array(&args[1])?;
        let out = fft_conv(&input, &kernel);
        Ok(f64_to_value_array(&out))
    }));

    env.functions.insert("hm_disk_tensor_new".into(), FnDef::Builtin(|_env, args| {
        if args.len() < 3 { return Err("hm_disk_tensor_new(path, rows, cols)".into()); }
        let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
        let rows = match &args[1] { Value::Int(i) => *i as usize, _ => return Err("rows must be int".into()) };
        let cols = match &args[2] { Value::Int(i) => *i as usize, _ => return Err("cols must be int".into()) };
        DiskTensor::zeros(PathBuf::from(&path), vec![rows, cols], 64).map_err(|e| e.to_string())?;
        Ok(Value::String(path))
    }));
}

// ─── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coo_to_csr_roundtrip() {
        let m = SparseMatrix::from_coo(3, 3,
            vec![0, 0, 1, 2], vec![0, 2, 1, 0], vec![1.0, 2.0, 3.0, 4.0]);
        let csr = m.to_csr();
        let back = csr.to_coo();
        let dense_orig = m.to_dense();
        let dense_back = back.to_dense();
        assert_eq!(dense_orig, dense_back);
    }

    #[test]
    fn test_coo_to_csc_roundtrip() {
        let m = SparseMatrix::from_coo(3, 3,
            vec![0, 1, 2], vec![0, 1, 2], vec![5.0, 6.0, 7.0]);
        let csc = m.to_csc();
        let back = csc.to_coo();
        assert_eq!(m.to_dense(), back.to_dense());
    }

    #[test]
    fn test_csr_to_csc() {
        let m = SparseMatrix::from_coo(2, 3,
            vec![0, 0, 1, 1], vec![0, 2, 1, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let csr = m.to_csr();
        let csc = csr.to_csc();
        assert_eq!(m.to_dense(), csc.to_dense());
    }

    #[test]
    fn test_sparse_dense_matmul() {
        // A = [[1,0],[0,2]], B = [[3,4],[5,6]]
        let a = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 2.0]);
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let c = sparse_dense_matmul(&a, &b, 2);
        assert_eq!(c, vec![3.0, 4.0, 10.0, 12.0]);
    }

    #[test]
    fn test_sparse_sparse_matmul() {
        // I * I = I (2x2 identity)
        let a = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 1.0]);
        let b = SparseMatrix::from_coo(2, 2, vec![0, 1], vec![0, 1], vec![1.0, 1.0]);
        let c = sparse_sparse_matmul(&a, &b);
        let dense = c.to_dense();
        assert_eq!(dense, vec![1.0, 0.0, 0.0, 1.0]);
    }

    #[test]
    fn test_sparse_add() {
        let a = SparseMatrix::from_coo(2, 2, vec![0], vec![0], vec![1.0]);
        let b = SparseMatrix::from_coo(2, 2, vec![1], vec![1], vec![2.0]);
        let c = sparse_add(&a, &b);
        let d = c.to_dense();
        assert_eq!(d, vec![1.0, 0.0, 0.0, 2.0]);
    }

    #[test]
    fn test_sparse_transpose() {
        let m = SparseMatrix::from_coo(2, 3, vec![0, 1], vec![1, 2], vec![5.0, 7.0]);
        let t = m.transpose();
        assert_eq!(t.rows, 3);
        assert_eq!(t.cols, 2);
        let d = t.to_dense();
        assert_eq!(d[1 * 2 + 0], 5.0); // (1,0) in transposed
        assert_eq!(d[2 * 2 + 1], 7.0); // (2,1) in transposed
    }

    #[test]
    fn test_sparse_from_dense() {
        let data = vec![0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 3.0, 0.0, 0.0];
        let s = SparseMatrix::from_dense(&data, 3, 3, 0.5);
        assert_eq!(s.nnz, 3);
        assert_eq!(s.to_dense(), data);
    }

    #[test]
    fn test_disk_tensor_write_read() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_disk_tensor.bin");
        let dt = DiskTensor::zeros(path.clone(), vec![4, 3], 2).unwrap();
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        dt.write_data(&data).unwrap();
        let chunk = dt.read_chunk(1, 2).unwrap(); // rows 1-2
        assert_eq!(chunk, vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_disk_reduce_sum() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_disk_reduce.bin");
        let dt = DiskTensor::zeros(path.clone(), vec![3, 2], 2).unwrap();
        dt.write_data(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let sum = dt.reduce_sum().unwrap();
        assert_eq!(sum, vec![9.0, 12.0]);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_disk_matmul() {
        let dir = std::env::temp_dir();
        let pa = dir.join("test_dm_a.bin");
        let pb = dir.join("test_dm_b.bin");
        let pc = dir.join("test_dm_c.bin");
        let a = DiskTensor::zeros(pa.clone(), vec![2, 2], 1).unwrap();
        let b = DiskTensor::zeros(pb.clone(), vec![2, 2], 2).unwrap();
        a.write_data(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        b.write_data(&[5.0, 6.0, 7.0, 8.0]).unwrap();
        let c = disk_matmul(&a, &b, pc.clone()).unwrap();
        let result = c.read_chunk(0, 2).unwrap();
        assert_eq!(result, vec![19.0, 22.0, 43.0, 50.0]);
        std::fs::remove_file(pa).ok();
        std::fs::remove_file(pb).ok();
        std::fs::remove_file(pc).ok();
    }

    #[test]
    fn test_blocked_matmul() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let c_blocked = blocked_matmul(&a, &b, 3, 3, 3, 2);
        let c_naive = naive_matmul(&a, &b, 3, 3, 3);
        for i in 0..9 {
            assert!((c_blocked[i] - c_naive[i]).abs() < 1e-10, "mismatch at {}", i);
        }
    }

    #[test]
    fn test_strassen_matches_naive() {
        let n = 8;
        let a: Vec<f64> = (0..n * n).map(|i| (i as f64) * 0.1).collect();
        let b: Vec<f64> = (0..n * n).map(|i| ((n * n - i) as f64) * 0.1).collect();
        let c_strassen = strassen_matmul(&a, &b, n);
        let c_naive = naive_matmul(&a, &b, n, n, n);
        for i in 0..n * n {
            assert!((c_strassen[i] - c_naive[i]).abs() < 1e-6, "mismatch at {}: {} vs {}", i, c_strassen[i], c_naive[i]);
        }
    }

    #[test]
    fn test_blocked_matrix_roundtrip() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let bm = BlockedMatrix::from_dense(&data, 2, 3, 2);
        let back = bm.to_dense(2);
        assert_eq!(data, back);
    }

    #[test]
    fn test_flash_attention_matches_naive() {
        let seq_len = 8;
        let d = 4;
        let q: Vec<f64> = (0..seq_len * d).map(|i| (i as f64) * 0.1).collect();
        let k: Vec<f64> = (0..seq_len * d).map(|i| ((seq_len * d - i) as f64) * 0.1).collect();
        let v: Vec<f64> = (0..seq_len * d).map(|i| (i as f64) * 0.05).collect();

        let naive = naive_attention(&q, &k, &v, seq_len, d);
        let flash = flash_attention(&q, &k, &v, seq_len, d, 2);

        for i in 0..seq_len * d {
            assert!((naive[i] - flash[i]).abs() < 1e-6,
                "flash attention mismatch at {}: naive={} flash={}", i, naive[i], flash[i]);
        }
    }

    #[test]
    fn test_linear_attention_output_reasonable() {
        let seq_len = 4;
        let d = 3;
        let q = vec![1.0; seq_len * d];
        let k = vec![1.0; seq_len * d];
        let v: Vec<f64> = (0..seq_len * d).map(|i| i as f64).collect();
        let out = linear_attention(&q, &k, &v, seq_len, d);
        // Should produce finite, non-zero output
        assert!(out.iter().all(|x| x.is_finite()));
        assert!(out.iter().any(|x| *x != 0.0));
    }

    #[test]
    fn test_sliding_window_attention() {
        let seq_len = 6;
        let d = 2;
        let q = vec![1.0; seq_len * d];
        let k = vec![1.0; seq_len * d];
        let v: Vec<f64> = (0..seq_len * d).map(|i| i as f64).collect();
        let out = sliding_window_attention(&q, &k, &v, seq_len, d, 3);
        assert_eq!(out.len(), seq_len * d);
        assert!(out.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn test_smith_waterman() {
        let scoring = ScoringMatrix::default();
        let result = smith_waterman("ACGTACGT", "CGTACG", &scoring);
        assert!(result.score > 0.0);
        assert!(!result.aligned_a.is_empty());
    }

    #[test]
    fn test_needleman_wunsch() {
        let scoring = ScoringMatrix::default();
        let result = needleman_wunsch("ACGT", "ACGT", &scoring);
        assert_eq!(result.score, 8.0); // 4 matches * 2.0
        assert_eq!(result.aligned_a, "ACGT");
        assert_eq!(result.aligned_b, "ACGT");
    }

    #[test]
    fn test_pairwise_distance() {
        let seqs = vec!["ACGT", "ACGT", "TTTT"];
        let d = pairwise_distance(&seqs);
        assert_eq!(d[0][1], 0.0); // identical
        assert!(d[0][2] > 0.0);   // different
        assert_eq!(d[0][2], d[2][0]); // symmetric
    }

    #[test]
    fn test_neighbor_joining() {
        let dist = vec![
            vec![0.0, 2.0, 4.0],
            vec![2.0, 0.0, 3.0],
            vec![4.0, 3.0, 0.0],
        ];
        let merges = neighbor_joining(&dist);
        assert!(!merges.is_empty());
    }

    #[test]
    fn test_winograd_conv_matches_naive() {
        let ih = 5;
        let iw = 5;
        let input: Vec<f64> = (0..ih * iw).map(|i| i as f64).collect();
        let kernel = vec![1.0, 0.0, -1.0, 2.0, 0.0, -2.0, 1.0, 0.0, -1.0];
        let naive = naive_conv2d(&input, ih, iw, &kernel, 3, 3);
        let wino = winograd_conv2d(&input, ih, iw, &kernel);
        assert_eq!(naive.len(), wino.len());
        for i in 0..naive.len() {
            assert!((naive[i] - wino[i]).abs() < 1e-6,
                "winograd mismatch at {}: naive={} wino={}", i, naive[i], wino[i]);
        }
    }

    #[test]
    fn test_fft_conv() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let kernel = vec![1.0, 0.0, -1.0];
        let result = fft_conv(&input, &kernel);
        // Expected: conv([1,2,3,4,5], [1,0,-1]) = [1,2,2,2,2,-4,-5]
        let expected = vec![1.0, 2.0, 2.0, 2.0, 2.0, -4.0, -5.0];
        assert_eq!(result.len(), expected.len());
        for i in 0..expected.len() {
            assert!((result[i] - expected[i]).abs() < 1e-10,
                "fft_conv mismatch at {}: {} vs {}", i, result[i], expected[i]);
        }
    }

    #[test]
    fn test_depthwise_conv() {
        // 2 channels, 3x3 input, 2x2 kernel each
        let input = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,  // ch0
            9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0,  // ch1
        ];
        let kernel = vec![
            1.0, 0.0, 0.0, 1.0,  // ch0: sum of diagonal
            0.0, 1.0, 1.0, 0.0,  // ch1: sum of anti-diagonal
        ];
        let out = depthwise_conv2d(&input, 3, 3, 2, &kernel, 2, 2);
        assert_eq!(out.len(), 2 * 2 * 2); // 2 channels * 2x2 output
    }

    #[test]
    fn test_dilated_conv() {
        let input: Vec<f64> = (0..25).map(|i| i as f64).collect(); // 5x5
        let kernel = vec![1.0, 0.0, 0.0, 1.0]; // 2x2
        let out = dilated_conv(&input, 5, 5, &kernel, 2, 2, 2);
        // effective kernel size: 3x3, so output: 3x3
        assert_eq!(out.len(), 9);
    }

    #[test]
    fn test_huge_matmul_small() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let c = huge_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }
}
