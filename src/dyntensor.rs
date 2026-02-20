/// Dynamic/ragged tensor support for variable-length computation.
///
/// Enables variable-length sequences (e.g., MoE routing, speculative decoding,
/// variable-length attention) without requiring padding to a fixed shape.

/// A ragged/dynamic tensor where one dimension is variable.
#[derive(Debug, Clone)]
pub struct DynTensor {
    /// Ragged data: each row can have a different length.
    pub data: Vec<Vec<f64>>,
    /// Maximum row length across all rows.
    pub max_len: usize,
}

impl DynTensor {
    /// Create a new DynTensor from ragged data.
    pub fn new(data: Vec<Vec<f64>>) -> Self {
        let max_len = data.iter().map(|row| row.len()).max().unwrap_or(0);
        DynTensor { data, max_len }
    }

    /// Create a DynTensor from dense (rectangular) data.
    pub fn from_dense(data: &[f64], rows: usize, cols: usize) -> Self {
        assert_eq!(data.len(), rows * cols);
        let mut ragged = Vec::with_capacity(rows);
        for r in 0..rows {
            ragged.push(data[r * cols..(r + 1) * cols].to_vec());
        }
        DynTensor {
            data: ragged,
            max_len: cols,
        }
    }

    /// Pad all rows to max_len with pad_value, returning (flat_data, lengths).
    pub fn to_padded(&self, pad_value: f64) -> (Vec<f64>, Vec<usize>) {
        let lengths: Vec<usize> = self.data.iter().map(|row| row.len()).collect();
        let mut flat = Vec::with_capacity(self.data.len() * self.max_len);
        for row in &self.data {
            flat.extend_from_slice(row);
            for _ in row.len()..self.max_len {
                flat.push(pad_value);
            }
        }
        (flat, lengths)
    }

    /// Remove rows where mask is false (stream compaction on rows).
    pub fn compact(&self, mask: &[bool]) -> DynTensor {
        assert_eq!(mask.len(), self.data.len());
        let data: Vec<Vec<f64>> = self
            .data
            .iter()
            .zip(mask.iter())
            .filter(|(_, &m)| m)
            .map(|(row, _)| row.clone())
            .collect();
        DynTensor::new(data)
    }

    /// Number of rows.
    pub fn num_rows(&self) -> usize {
        self.data.len()
    }

    /// Total number of elements across all rows.
    pub fn total_elements(&self) -> usize {
        self.data.iter().map(|row| row.len()).sum()
    }
}

/// Stream compaction: remove inactive elements based on mask.
/// Elements are grouped by stride; entire groups are kept or removed.
pub fn stream_compact(data: &[f64], mask: &[bool], stride: usize) -> Vec<f64> {
    assert!(stride > 0);
    let num_groups = mask.len();
    assert_eq!(data.len(), num_groups * stride);

    let offsets = prefix_sum(mask);
    let total = offsets.last().copied().unwrap_or(0);
    let mut result = vec![0.0; total * stride];

    for (i, &m) in mask.iter().enumerate() {
        if m {
            let dst_start = if i == 0 { 0 } else { offsets[i - 1] } * stride;
            let src_start = i * stride;
            result[dst_start..dst_start + stride]
                .copy_from_slice(&data[src_start..src_start + stride]);
        }
    }

    result
}

/// Exclusive prefix sum on a boolean mask, returning cumulative counts.
/// Result[i] = number of true values in mask[0..=i].
pub fn prefix_sum(mask: &[bool]) -> Vec<usize> {
    let mut result = Vec::with_capacity(mask.len());
    let mut sum = 0usize;
    for &m in mask {
        if m {
            sum += 1;
        }
        result.push(sum);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dyntensor_compact() {
        let dt = DynTensor::new(vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0],
            vec![6.0],
            vec![7.0, 8.0, 9.0, 10.0],
        ]);
        assert_eq!(dt.num_rows(), 4);
        assert_eq!(dt.max_len, 4);
        assert_eq!(dt.total_elements(), 10);

        let mask = vec![true, false, true, false];
        let compacted = dt.compact(&mask);
        assert_eq!(compacted.num_rows(), 2);
        assert_eq!(compacted.data[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(compacted.data[1], vec![6.0]);
        assert_eq!(compacted.max_len, 3);
    }

    #[test]
    fn test_dyntensor_pad() {
        let dt = DynTensor::new(vec![
            vec![1.0, 2.0],
            vec![3.0],
            vec![4.0, 5.0, 6.0],
        ]);
        assert_eq!(dt.max_len, 3);

        let (padded, lengths) = dt.to_padded(0.0);
        assert_eq!(lengths, vec![2, 1, 3]);
        assert_eq!(padded, vec![
            1.0, 2.0, 0.0,  // row 0 padded
            3.0, 0.0, 0.0,  // row 1 padded
            4.0, 5.0, 6.0,  // row 2 no padding needed
        ]);
    }

    #[test]
    fn test_stream_compact() {
        let data = vec![
            1.0, 2.0,   // group 0
            3.0, 4.0,   // group 1
            5.0, 6.0,   // group 2
            7.0, 8.0,   // group 3
        ];
        let mask = vec![true, false, true, true];
        let stride = 2;

        let result = stream_compact(&data, &mask, stride);
        assert_eq!(result, vec![1.0, 2.0, 5.0, 6.0, 7.0, 8.0]);

        // Verify prefix_sum
        let ps = prefix_sum(&mask);
        assert_eq!(ps, vec![1, 1, 2, 3]);
    }
}
