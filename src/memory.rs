/// Differentiable memory bank (Neural Turing Machine / DNC style)
#[derive(Debug, Clone)]
pub struct DiffMemory {
    pub keys: Vec<f64>,   // C * K flattened
    pub values: Vec<f64>, // C * V flattened
    pub capacity: usize,  // C
    pub key_dim: usize,   // K
    pub val_dim: usize,   // V
    pub usage: Vec<f64>,  // C usage scores for eviction
}

impl DiffMemory {
    pub fn new(capacity: usize, key_dim: usize, val_dim: usize) -> Self {
        DiffMemory {
            keys: vec![0.0; capacity * key_dim],
            values: vec![0.0; capacity * val_dim],
            capacity,
            key_dim,
            val_dim,
            usage: vec![0.0; capacity],
        }
    }

    /// Soft read: attention-weighted retrieval
    pub fn read(&self, query: &[f64]) -> Vec<f64> {
        // weights = softmax(query @ keys.T)
        let weights = self.content_lookup(query, 1.0);
        // return weights @ values
        let mut result = vec![0.0; self.val_dim];
        for c in 0..self.capacity {
            for v in 0..self.val_dim {
                result[v] += weights[c] * self.values[c * self.val_dim + v];
            }
        }
        result
    }

    /// Soft write: attention-weighted update
    pub fn write(&mut self, key: &[f64], value: &[f64]) {
        // addr = softmax(key @ self.keys.T)
        let addr = self.content_lookup(key, 1.0);
        // self.values += outer(addr, value)
        for c in 0..self.capacity {
            for v in 0..self.val_dim {
                if v < value.len() {
                    self.values[c * self.val_dim + v] += addr[c] * value[v];
                }
            }
            // Update keys similarly
            for k in 0..self.key_dim {
                if k < key.len() {
                    self.keys[c * self.key_dim + k] += addr[c] * key[k];
                }
            }
            // Update usage scores
            self.usage[c] += addr[c];
        }
    }

    /// Content-based addressing with cosine similarity and sharpening factor beta
    pub fn content_lookup(&self, query: &[f64], beta: f64) -> Vec<f64> {
        let mut similarities = Vec::with_capacity(self.capacity);

        let q_norm = dot_norm(query);

        for c in 0..self.capacity {
            let key_slice = &self.keys[c * self.key_dim..(c + 1) * self.key_dim];
            let k_norm = dot_norm(key_slice);
            let dot: f64 = query
                .iter()
                .zip(key_slice.iter())
                .map(|(a, b)| a * b)
                .sum();

            let denom = q_norm * k_norm;
            let cosine = if denom > 1e-12 { dot / denom } else { 0.0 };
            similarities.push(cosine * beta);
        }

        softmax(&similarities)
    }

    /// Least-recently-used write to free slot
    pub fn write_lru(&mut self, key: &[f64], value: &[f64]) {
        // Find slot with lowest usage, overwrite
        let mut min_idx = 0;
        let mut min_usage = f64::INFINITY;
        for c in 0..self.capacity {
            if self.usage[c] < min_usage {
                min_usage = self.usage[c];
                min_idx = c;
            }
        }

        // Overwrite that slot
        for k in 0..self.key_dim {
            if k < key.len() {
                self.keys[min_idx * self.key_dim + k] = key[k];
            }
        }
        for v in 0..self.val_dim {
            if v < value.len() {
                self.values[min_idx * self.val_dim + v] = value[v];
            }
        }
        self.usage[min_idx] = 1.0;
    }
}

fn dot_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    if logits.is_empty() {
        return vec![];
    }
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum > 0.0 {
        exps.iter().map(|e| e / sum).collect()
    } else {
        vec![1.0 / logits.len() as f64; logits.len()]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_read_write() {
        let mut mem = DiffMemory::new(4, 3, 2);
        let key = vec![1.0, 0.0, 0.0];
        let value = vec![5.0, 10.0];

        // Write a value
        mem.write_lru(&key, &value);

        // Read it back with the same key
        let result = mem.read(&key);
        // The written slot should dominate since other slots are zero
        // The LRU write put it at slot 0, cosine similarity with [1,0,0] is 1.0 for that slot
        // After softmax, slot 0 should get highest weight
        assert!(result[0] > 1.0, "Expected significant retrieval, got {:?}", result);
    }

    #[test]
    fn test_memory_content_lookup() {
        let mut mem = DiffMemory::new(3, 2, 2);
        // Set up distinct keys
        mem.keys = vec![
            1.0, 0.0, // slot 0
            0.0, 1.0, // slot 1
            0.7, 0.7, // slot 2
        ];

        let query = vec![1.0, 0.0];
        let weights = mem.content_lookup(&query, 10.0); // high beta for sharp attention

        // Slot 0 should have highest weight (cosine similarity = 1.0)
        assert!(
            weights[0] > weights[1],
            "Slot 0 should dominate: {:?}",
            weights
        );
        assert!(
            weights[0] > weights[2],
            "Slot 0 should beat slot 2: {:?}",
            weights
        );
    }

    #[test]
    fn test_memory_lru() {
        let mut mem = DiffMemory::new(3, 2, 2);

        // Fill all slots
        mem.write_lru(&[1.0, 0.0], &[1.0, 1.0]);
        mem.write_lru(&[0.0, 1.0], &[2.0, 2.0]);
        mem.write_lru(&[1.0, 1.0], &[3.0, 3.0]);

        // All slots used: usage = [1.0, 1.0, 1.0]
        // Now bump usage of slot 1 to make slot 0 the LRU
        mem.usage[1] = 5.0;
        mem.usage[2] = 3.0;

        // Write new data - should evict slot 0 (lowest usage = 1.0)
        mem.write_lru(&[0.5, 0.5], &[99.0, 88.0]);

        // Slot 0 should now have the new values
        assert_eq!(mem.values[0], 99.0);
        assert_eq!(mem.values[1], 88.0);
        assert_eq!(mem.keys[0], 0.5);
        assert_eq!(mem.keys[1], 0.5);
    }
}
