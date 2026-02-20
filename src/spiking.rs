/// Bitpacked spike train: T timesteps, N neurons, 1 bit each
/// Stored as Vec<u64> for efficient bitwise operations
#[derive(Debug, Clone)]
pub struct SpikeTrain {
    pub timesteps: usize,
    pub neurons: usize,
    pub bits: Vec<u64>, // ceil(T * N / 64) u64s
}

impl SpikeTrain {
    pub fn new(timesteps: usize, neurons: usize) -> Self {
        let total_bits = timesteps * neurons;
        let n_words = (total_bits + 63) / 64;
        SpikeTrain {
            timesteps,
            neurons,
            bits: vec![0u64; n_words],
        }
    }

    pub fn from_dense(data: &[f64], timesteps: usize, neurons: usize, threshold: f64) -> Self {
        let mut train = Self::new(timesteps, neurons);
        for t in 0..timesteps {
            for n in 0..neurons {
                let idx = t * neurons + n;
                if idx < data.len() && data[idx] > threshold {
                    train.set(t, n, true);
                }
            }
        }
        train
    }

    pub fn get(&self, t: usize, n: usize) -> bool {
        let bit_idx = t * self.neurons + n;
        let word = bit_idx / 64;
        let bit = bit_idx % 64;
        if word < self.bits.len() {
            (self.bits[word] >> bit) & 1 == 1
        } else {
            false
        }
    }

    pub fn set(&mut self, t: usize, n: usize, val: bool) {
        let bit_idx = t * self.neurons + n;
        let word = bit_idx / 64;
        let bit = bit_idx % 64;
        if word < self.bits.len() {
            if val {
                self.bits[word] |= 1u64 << bit;
            } else {
                self.bits[word] &= !(1u64 << bit);
            }
        }
    }

    /// Get all neurons at timestep t
    pub fn step(&self, t: usize) -> Vec<bool> {
        (0..self.neurons).map(|n| self.get(t, n)).collect()
    }

    /// Expand to 0.0/1.0 dense representation
    pub fn to_dense(&self) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.timesteps * self.neurons);
        for t in 0..self.timesteps {
            for n in 0..self.neurons {
                out.push(if self.get(t, n) { 1.0 } else { 0.0 });
            }
        }
        out
    }

    /// Fraction of zeros
    pub fn sparsity(&self) -> f64 {
        let total = self.timesteps * self.neurons;
        if total == 0 {
            return 1.0;
        }
        let ones: usize = self.bits.iter().map(|w| w.count_ones() as usize).sum();
        // Discount any padding bits beyond total
        let padding_bits = self.bits.len() * 64 - total;
        // All padding bits are zero, so ones count is accurate
        let _ = padding_bits;
        1.0 - (ones as f64 / total as f64)
    }
}

/// Bitwise overlap scoring (for SDR/HTM)
pub fn spike_overlap(a: &SpikeTrain, b: &SpikeTrain) -> usize {
    // popcount(a AND b) - single instruction per 64 bits on GPU
    a.bits
        .iter()
        .zip(b.bits.iter())
        .map(|(x, y)| (x & y).count_ones() as usize)
        .sum()
}

/// Leaky Integrate-and-Fire neuron layer
pub fn lif_layer(
    input: &SpikeTrain,
    weights: &[f64],
    n_in: usize,
    n_out: usize,
    threshold: f64,
    tau: f64,
) -> SpikeTrain {
    let t = input.timesteps;
    let mut output = SpikeTrain::new(t, n_out);
    let mut potential = vec![0.0; n_out];

    for step in 0..t {
        let spikes = input.step(step);
        for j in 0..n_out {
            potential[j] *= tau; // leak
            for i in 0..n_in {
                if i < spikes.len() && spikes[i] {
                    potential[j] += weights[i * n_out + j];
                }
            }
            if potential[j] > threshold {
                output.set(step, j, true);
                potential[j] = 0.0; // reset
            }
        }
    }
    output
}

/// Spike-driven attention: only attend to neurons that fired
pub fn spike_attention(
    q_spikes: &SpikeTrain,
    k_spikes: &SpikeTrain,
    v: &[f64],
    d: usize,
) -> Vec<f64> {
    // Sparse attention: only compute for active (spiking) positions
    let t = q_spikes.timesteps;
    let n = q_spikes.neurons;
    let mut output = vec![0.0; t * d];

    for step in 0..t {
        let q_active = q_spikes.step(step);
        let k_active = k_spikes.step(step);

        // Count active keys for normalization
        let active_count: f64 = k_active.iter().filter(|&&b| b).count() as f64;
        if active_count == 0.0 {
            continue;
        }

        for qi in 0..n {
            if !q_active[qi] {
                continue;
            }
            // Uniform attention over active keys
            for ki in 0..n {
                if !k_active[ki] {
                    continue;
                }
                let weight = 1.0 / active_count;
                for dd in 0..d {
                    if ki * d + dd < v.len() {
                        output[step * d + dd] += weight * v[ki * d + dd];
                    }
                }
            }
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_spike_train_basic() {
        let mut train = SpikeTrain::new(4, 8);
        assert_eq!(train.get(0, 0), false);

        train.set(0, 3, true);
        train.set(2, 5, true);
        train.set(3, 7, true);

        assert_eq!(train.get(0, 3), true);
        assert_eq!(train.get(2, 5), true);
        assert_eq!(train.get(3, 7), true);
        assert_eq!(train.get(0, 0), false);
        assert_eq!(train.get(1, 3), false);

        // 3 ones out of 32 total bits
        let sp = train.sparsity();
        assert!((sp - (1.0 - 3.0 / 32.0)).abs() < 1e-10);
    }

    #[test]
    fn test_spike_overlap() {
        let mut a = SpikeTrain::new(1, 8);
        let mut b = SpikeTrain::new(1, 8);

        a.set(0, 0, true);
        a.set(0, 2, true);
        a.set(0, 4, true);

        b.set(0, 0, true);
        b.set(0, 1, true);
        b.set(0, 4, true);

        // Overlap at positions 0 and 4
        assert_eq!(spike_overlap(&a, &b), 2);
    }

    #[test]
    fn test_lif_layer() {
        // 4 timesteps, 2 input neurons, 1 output neuron
        let mut input = SpikeTrain::new(4, 2);
        // Fire neuron 0 at every timestep
        input.set(0, 0, true);
        input.set(1, 0, true);
        input.set(2, 0, true);
        input.set(3, 0, true);

        // weights: 2 inputs -> 1 output, weight = 0.6
        let weights = vec![0.6, 0.3];
        // threshold=1.0, tau=0.5 (leak factor)
        let output = lif_layer(&input, &weights, 2, 1, 1.0, 0.5);

        // step 0: potential = 0*0.5 + 0.6 = 0.6 (no fire)
        // step 1: potential = 0.6*0.5 + 0.6 = 0.9 (no fire)
        // step 2: potential = 0.9*0.5 + 0.6 = 1.05 > 1.0 (FIRE, reset to 0)
        // step 3: potential = 0*0.5 + 0.6 = 0.6 (no fire)
        assert_eq!(output.get(0, 0), false);
        assert_eq!(output.get(1, 0), false);
        assert_eq!(output.get(2, 0), true);
        assert_eq!(output.get(3, 0), false);
    }

    #[test]
    fn test_spike_from_dense() {
        let data = vec![0.1, 0.8, 0.3, 0.9, 0.05, 0.7];
        let train = SpikeTrain::from_dense(&data, 2, 3, 0.5);

        // timestep 0: [0.1, 0.8, 0.3] -> [false, true, false]
        // timestep 1: [0.9, 0.05, 0.7] -> [true, false, true]
        assert_eq!(train.get(0, 0), false);
        assert_eq!(train.get(0, 1), true);
        assert_eq!(train.get(0, 2), false);
        assert_eq!(train.get(1, 0), true);
        assert_eq!(train.get(1, 1), false);
        assert_eq!(train.get(1, 2), true);

        // Verify round-trip via to_dense
        let dense = train.to_dense();
        assert_eq!(dense, vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0]);
    }
}
