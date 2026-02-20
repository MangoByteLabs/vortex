/// Sparse tensor runtime for Vortex.

/// A sparse tensor in COO-like format: stores only non-zero elements.
#[derive(Debug, Clone)]
pub struct SparseTensor {
    pub indices: Vec<usize>,
    pub values: Vec<f64>,
    pub shape: Vec<usize>,
}

/// A sparse index structure for top-k / MoE routing.
#[derive(Debug, Clone)]
pub struct SparseIndex {
    pub batch_size: usize,
    pub k: usize,
    pub indices: Vec<Vec<usize>>,
}

/// Select the top-k indices from a flat array of values.
pub fn sparse_topk(values: &[f64], k: usize) -> SparseIndex {
    let k = k.min(values.len());
    let mut indexed: Vec<(usize, f64)> = values.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_indices: Vec<usize> = indexed[..k].iter().map(|(i, _)| *i).collect();
    SparseIndex {
        batch_size: 1,
        k,
        indices: vec![top_indices],
    }
}

/// Gather values from a dense array using a SparseIndex.
pub fn sparse_gather(dense: &[f64], idx: &SparseIndex) -> Vec<Vec<f64>> {
    idx.indices
        .iter()
        .map(|batch_indices| {
            batch_indices
                .iter()
                .map(|&i| if i < dense.len() { dense[i] } else { 0.0 })
                .collect()
        })
        .collect()
}

/// Scatter sparse values back into a dense array of the given size.
pub fn sparse_scatter(sparse_vals: &[Vec<f64>], idx: &SparseIndex, size: usize) -> Vec<f64> {
    let mut result = vec![0.0; size];
    for (batch_indices, batch_values) in idx.indices.iter().zip(sparse_vals.iter()) {
        for (&i, &v) in batch_indices.iter().zip(batch_values.iter()) {
            if i < size {
                result[i] += v;
            }
        }
    }
    result
}

/// Sparse-dense matrix multiply: sparse (stored as COO) times dense column-major.
/// `dense` is a flat array of shape [rows x cols], row-major.
/// `cols` is the number of columns in the dense matrix.
pub fn sparse_matmul(sparse: &SparseTensor, dense: &[f64], cols: usize) -> Vec<f64> {
    let rows = if sparse.shape.is_empty() { 0 } else { sparse.shape[0] };
    let mut result = vec![0.0; rows * cols];
    for (&idx, &val) in sparse.indices.iter().zip(sparse.values.iter()) {
        let row = idx / (if sparse.shape.len() > 1 { sparse.shape[1] } else { 1 });
        let col_s = idx % (if sparse.shape.len() > 1 { sparse.shape[1] } else { 1 });
        for c in 0..cols {
            if col_s * cols + c < dense.len() {
                result[row * cols + c] += val * dense[col_s * cols + c];
            }
        }
    }
    result
}

/// Convert a dense tensor to sparse by zeroing out values below a threshold.
pub fn to_sparse(values: &[f64], shape: Vec<usize>, threshold: f64) -> SparseTensor {
    let mut indices = Vec::new();
    let mut sparse_values = Vec::new();
    for (i, &v) in values.iter().enumerate() {
        if v.abs() >= threshold {
            indices.push(i);
            sparse_values.push(v);
        }
    }
    SparseTensor {
        indices,
        values: sparse_values,
        shape,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sparse_topk() {
        let values = vec![0.1, 0.9, 0.3, 0.7, 0.5];
        let idx = sparse_topk(&values, 3);
        assert_eq!(idx.k, 3);
        assert_eq!(idx.indices.len(), 1);
        let top = &idx.indices[0];
        assert_eq!(top.len(), 3);
        // Top 3 values are 0.9 (idx 1), 0.7 (idx 3), 0.5 (idx 4)
        assert!(top.contains(&1));
        assert!(top.contains(&3));
        assert!(top.contains(&4));
    }

    #[test]
    fn test_sparse_gather_scatter() {
        let dense = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let idx = SparseIndex {
            batch_size: 1,
            k: 3,
            indices: vec![vec![0, 2, 4]],
        };
        let gathered = sparse_gather(&dense, &idx);
        assert_eq!(gathered, vec![vec![10.0, 30.0, 50.0]]);
        let scattered = sparse_scatter(&gathered, &idx, 5);
        assert_eq!(scattered, vec![10.0, 0.0, 30.0, 0.0, 50.0]);
    }

    #[test]
    fn test_sparse_matmul() {
        // 2x2 identity matrix in sparse form
        let sparse = SparseTensor {
            indices: vec![0, 3], // (0,0) and (1,1)
            values: vec![1.0, 1.0],
            shape: vec![2, 2],
        };
        let dense = vec![5.0, 6.0, 7.0, 8.0]; // 2x2 row-major
        let result = sparse_matmul(&sparse, &dense, 2);
        // Identity * dense = dense
        assert_eq!(result, vec![5.0, 6.0, 7.0, 8.0]);
    }
}
