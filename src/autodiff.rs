/// Reverse-mode automatic differentiation (backpropagation) via a computation tape.

/// A node in the computation graph
#[derive(Clone, Debug)]
pub struct TapeEntry {
    pub value: f64,
    pub grad: f64,
    pub deps: Vec<(usize, f64)>, // (parent_index, local_gradient)
}

/// The computation tape for reverse-mode AD
pub struct Tape {
    pub entries: Vec<TapeEntry>,
}

impl Tape {
    pub fn new() -> Self {
        Tape {
            entries: Vec::new(),
        }
    }

    /// Create a variable (leaf node)
    pub fn var(&mut self, value: f64) -> usize {
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value,
            grad: 0.0,
            deps: vec![],
        });
        idx
    }

    /// Addition
    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let val = self.entries[a].value + self.entries[b].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: val,
            grad: 0.0,
            deps: vec![(a, 1.0), (b, 1.0)],
        });
        idx
    }

    /// Multiplication
    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let va = self.entries[a].value;
        let vb = self.entries[b].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: va * vb,
            grad: 0.0,
            deps: vec![(a, vb), (b, va)],
        });
        idx
    }

    /// Subtraction
    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let val = self.entries[a].value - self.entries[b].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: val,
            grad: 0.0,
            deps: vec![(a, 1.0), (b, -1.0)],
        });
        idx
    }

    /// Division
    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let va = self.entries[a].value;
        let vb = self.entries[b].value;
        let idx = self.entries.len();
        // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
        self.entries.push(TapeEntry {
            value: va / vb,
            grad: 0.0,
            deps: vec![(a, 1.0 / vb), (b, -va / (vb * vb))],
        });
        idx
    }

    /// Power: a^n (n is a constant)
    pub fn pow(&mut self, a: usize, n: f64) -> usize {
        let va = self.entries[a].value;
        let idx = self.entries.len();
        // d(a^n)/da = n * a^(n-1)
        self.entries.push(TapeEntry {
            value: va.powf(n),
            grad: 0.0,
            deps: vec![(a, n * va.powf(n - 1.0))],
        });
        idx
    }

    /// Exp
    pub fn exp(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let result = va.exp();
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: result,
            grad: 0.0,
            deps: vec![(a, result)],
        });
        idx
    }

    /// Log (natural)
    pub fn log(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: va.ln(),
            grad: 0.0,
            deps: vec![(a, 1.0 / va)],
        });
        idx
    }

    /// Tanh
    pub fn tanh(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let t = va.tanh();
        let idx = self.entries.len();
        // d(tanh(a))/da = 1 - tanh(a)^2
        self.entries.push(TapeEntry {
            value: t,
            grad: 0.0,
            deps: vec![(a, 1.0 - t * t)],
        });
        idx
    }

    /// ReLU
    pub fn relu(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let idx = self.entries.len();
        let (val, grad) = if va > 0.0 { (va, 1.0) } else { (0.0, 0.0) };
        self.entries.push(TapeEntry {
            value: val,
            grad: 0.0,
            deps: vec![(a, grad)],
        });
        idx
    }

    /// Sigmoid
    pub fn sigmoid(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let s = 1.0 / (1.0 + (-va).exp());
        let idx = self.entries.len();
        // d(sigmoid(a))/da = sigmoid(a) * (1 - sigmoid(a))
        self.entries.push(TapeEntry {
            value: s,
            grad: 0.0,
            deps: vec![(a, s * (1.0 - s))],
        });
        idx
    }

    /// Sin
    pub fn sin(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: va.sin(),
            grad: 0.0,
            deps: vec![(a, va.cos())],
        });
        idx
    }

    /// Cos
    pub fn cos(&mut self, a: usize) -> usize {
        let va = self.entries[a].value;
        let idx = self.entries.len();
        self.entries.push(TapeEntry {
            value: va.cos(),
            grad: 0.0,
            deps: vec![(a, -va.sin())],
        });
        idx
    }

    /// Sum of array of tape indices
    pub fn sum(&mut self, indices: &[usize]) -> usize {
        let val: f64 = indices.iter().map(|&i| self.entries[i].value).sum();
        let idx = self.entries.len();
        let deps: Vec<(usize, f64)> = indices.iter().map(|&i| (i, 1.0)).collect();
        self.entries.push(TapeEntry {
            value: val,
            grad: 0.0,
            deps,
        });
        idx
    }

    /// Dot product of two vectors of tape indices
    pub fn dot(&mut self, a: &[usize], b: &[usize]) -> usize {
        assert_eq!(a.len(), b.len(), "dot: vectors must be same length");
        let products: Vec<usize> = a
            .iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| self.mul(ai, bi))
            .collect();
        self.sum(&products)
    }

    /// Backward pass: compute all gradients
    pub fn backward(&mut self, output: usize) {
        self.entries[output].grad = 1.0;
        for i in (0..=output).rev() {
            let grad = self.entries[i].grad;
            let deps = self.entries[i].deps.clone();
            for (parent_idx, local_grad) in deps {
                self.entries[parent_idx].grad += grad * local_grad;
            }
        }
    }

    /// Get gradient of a variable
    pub fn grad(&self, var_idx: usize) -> f64 {
        self.entries[var_idx].grad
    }

    /// Get value of a node
    pub fn value(&self, idx: usize) -> f64 {
        self.entries[idx].value
    }

    /// Reset all gradients to zero
    pub fn zero_grad(&mut self) {
        for entry in &mut self.entries {
            entry.grad = 0.0;
        }
    }
}

/// SGD optimizer step
pub fn sgd_step(params: &mut [f64], grads: &[f64], lr: f64) {
    for (p, g) in params.iter_mut().zip(grads.iter()) {
        *p -= lr * g;
    }
}

/// Adam optimizer step
pub fn adam_step(
    params: &mut [f64],
    grads: &[f64],
    m: &mut [f64],
    v: &mut [f64],
    lr: f64,
    beta1: f64,
    beta2: f64,
    eps: f64,
    t: usize,
) {
    let t_f = t as f64;
    for i in 0..params.len() {
        m[i] = beta1 * m[i] + (1.0 - beta1) * grads[i];
        v[i] = beta2 * v[i] + (1.0 - beta2) * grads[i] * grads[i];
        let m_hat = m[i] / (1.0 - beta1.powf(t_f));
        let v_hat = v[i] / (1.0 - beta2.powf(t_f));
        params[i] -= lr * m_hat / (v_hat.sqrt() + eps);
    }
}

/// MSE loss on tape
pub fn mse_loss(tape: &mut Tape, predictions: &[usize], targets: &[f64]) -> usize {
    assert_eq!(predictions.len(), targets.len());
    let n = predictions.len() as f64;
    let diffs: Vec<usize> = predictions
        .iter()
        .zip(targets.iter())
        .map(|(&pred, &target)| {
            let t = tape.var(target);
            let diff = tape.sub(pred, t);
            tape.pow(diff, 2.0)
        })
        .collect();
    let total = tape.sum(&diffs);
    let n_node = tape.var(n);
    tape.div(total, n_node)
}

/// Cross-entropy loss: -log(softmax(logits)[target_idx])
pub fn cross_entropy_loss(tape: &mut Tape, logits: &[usize], target_idx: usize) -> usize {
    // Numerically stable softmax: subtract max
    let max_val = logits
        .iter()
        .map(|&i| tape.value(i))
        .fold(f64::NEG_INFINITY, f64::max);
    let max_node = tape.var(max_val);
    let shifted: Vec<usize> = logits.iter().map(|&l| tape.sub(l, max_node)).collect();
    let exp_vals: Vec<usize> = shifted.iter().map(|&s| tape.exp(s)).collect();
    let sum_exp = tape.sum(&exp_vals);
    let log_sum = tape.log(sum_exp);
    let log_softmax_target = tape.sub(shifted[target_idx], log_sum);
    // Negate
    let neg_one = tape.var(-1.0);
    tape.mul(neg_one, log_softmax_target)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_gradient() {
        // d/dx(x^2) = 2x, at x=3 -> grad=6
        let mut tape = Tape::new();
        let x = tape.var(3.0);
        let y = tape.pow(x, 2.0);
        tape.backward(y);
        assert!((tape.grad(x) - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_chain_rule() {
        // d/dx(sin(x^2)) = 2x*cos(x^2), at x=1 -> 2*cos(1)
        let mut tape = Tape::new();
        let x = tape.var(1.0);
        let x2 = tape.pow(x, 2.0);
        let y = tape.sin(x2);
        tape.backward(y);
        let expected = 2.0 * 1.0_f64.cos();
        assert!((tape.grad(x) - expected).abs() < 1e-10);
    }

    #[test]
    fn test_product_rule() {
        // d/dx(x*y) = y, d/dy(x*y) = x, at x=3, y=5
        let mut tape = Tape::new();
        let x = tape.var(3.0);
        let y = tape.var(5.0);
        let z = tape.mul(x, y);
        tape.backward(z);
        assert!((tape.grad(x) - 5.0).abs() < 1e-10);
        assert!((tape.grad(y) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_backward_mlp() {
        // Simple 2-layer: y = relu(w2 * relu(w1 * x + b1) + b2)
        let mut tape = Tape::new();
        let x = tape.var(1.0);
        let w1 = tape.var(0.5);
        let b1 = tape.var(0.1);
        let w2 = tape.var(0.3);
        let b2 = tape.var(-0.1);

        let h1 = tape.mul(w1, x);
        let h2 = tape.add(h1, b1);
        let h3 = tape.relu(h2);
        let o1 = tape.mul(w2, h3);
        let o2 = tape.add(o1, b2);
        let out = tape.relu(o2);

        tape.backward(out);
        // All gradients should be finite
        assert!(tape.grad(w1).is_finite());
        assert!(tape.grad(w2).is_finite());
        assert!(tape.grad(b1).is_finite());
        assert!(tape.grad(b2).is_finite());
        // Value should be relu(0.3 * relu(0.5*1+0.1) + (-0.1)) = relu(0.3*0.6-0.1) = relu(0.08) = 0.08
        assert!((tape.value(out) - 0.08).abs() < 1e-10);
    }

    #[test]
    fn test_sgd_step() {
        let mut params = vec![1.0, 2.0, 3.0];
        let grads = vec![0.1, 0.2, 0.3];
        sgd_step(&mut params, &grads, 0.1);
        assert!((params[0] - 0.99).abs() < 1e-10);
        assert!((params[1] - 1.98).abs() < 1e-10);
        assert!((params[2] - 2.97).abs() < 1e-10);
    }

    #[test]
    fn test_adam_step() {
        let mut params = vec![1.0];
        let grads = vec![1.0];
        let mut m = vec![0.0];
        let mut v = vec![0.0];
        // Run a few steps and check convergence toward 0
        for t in 1..=100 {
            adam_step(&mut params, &grads, &mut m, &mut v, 0.01, 0.9, 0.999, 1e-8, t);
        }
        assert!(params[0] < 1.0, "Adam should decrease param with positive gradient");
    }

    #[test]
    fn test_mse_loss() {
        let mut tape = Tape::new();
        let p1 = tape.var(1.0);
        let p2 = tape.var(2.0);
        let loss = mse_loss(&mut tape, &[p1, p2], &[1.0, 1.0]);
        tape.backward(loss);
        // MSE = ((1-1)^2 + (2-1)^2)/2 = 0.5
        assert!((tape.value(loss) - 0.5).abs() < 1e-10);
        // d(MSE)/d(p1) = 2*(1-1)/2 = 0, d(MSE)/d(p2) = 2*(2-1)/2 = 1
        assert!((tape.grad(p1) - 0.0).abs() < 1e-10);
        assert!((tape.grad(p2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_relu_gradient() {
        let mut tape = Tape::new();
        // Positive input
        let x = tape.var(2.0);
        let y = tape.relu(x);
        tape.backward(y);
        assert!((tape.grad(x) - 1.0).abs() < 1e-10);

        // Negative input
        let mut tape2 = Tape::new();
        let x2 = tape2.var(-2.0);
        let y2 = tape2.relu(x2);
        tape2.backward(y2);
        assert!((tape2.grad(x2) - 0.0).abs() < 1e-10);
    }
}
