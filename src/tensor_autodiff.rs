/// Tensor-level reverse-mode automatic differentiation engine.
///
/// Extends the scalar tape in `autodiff.rs` to support tensor operations
/// including matmul, activations, softmax, layer norm, and cross-entropy loss.

use std::f64::consts::PI;

/// A tensor value that may or may not be tracked for gradients
#[derive(Clone, Debug)]
pub struct Tensor {
    pub data: Vec<f64>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: Option<Vec<f64>>,
    tape_id: Option<usize>,
}

impl Tensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Operations recorded on the tensor tape
#[derive(Clone, Debug)]
pub enum TensorOp {
    // Binary
    MatMul { lhs: usize, rhs: usize, m: usize, k: usize, n: usize },
    Add { lhs: usize, rhs: usize },
    Mul { lhs: usize, rhs: usize },
    Sub { lhs: usize, rhs: usize },
    Div { lhs: usize, rhs: usize },

    // Unary
    Relu { input: usize },
    Sigmoid { input: usize },
    Tanh { input: usize },
    Gelu { input: usize },
    Exp { input: usize },
    Log { input: usize },
    Sqrt { input: usize },
    Neg { input: usize },

    // Reductions
    Sum { input: usize, axis: Option<usize> },
    Mean { input: usize, axis: Option<usize> },
    Max { input: usize, axis: Option<usize> },

    // Shape
    Reshape { input: usize, new_shape: Vec<usize> },
    Transpose { input: usize },

    // Neural network ops
    Softmax { input: usize, axis: usize },
    LayerNorm { input: usize, gamma: usize, beta: usize, eps: f64 },
    CrossEntropyLoss { logits: usize, targets: Vec<usize> },

    // Broadcast
    BroadcastAdd { tensor: usize, bias: usize },

    // Leaf
    Leaf,
}

pub struct TensorTape {
    pub tensors: Vec<Tensor>,
    pub ops: Vec<TensorOp>,
}

// ── Helper functions ────────────────────────────────────────────────

/// Raw 2D matrix multiply: A[m,k] @ B[k,n] -> C[m,n]
pub fn matmul_2d(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut s = 0.0;
            for p in 0..k {
                s += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = s;
        }
    }
    c
}

/// Compute broadcast-compatible output shape
pub fn broadcast_shapes(a: &[usize], b: &[usize]) -> Vec<usize> {
    let max_len = a.len().max(b.len());
    let mut result = vec![0usize; max_len];
    for i in 0..max_len {
        let da = if i < max_len - a.len() { 1 } else { a[i - (max_len - a.len())] };
        let db = if i < max_len - b.len() { 1 } else { b[i - (max_len - b.len())] };
        assert!(da == db || da == 1 || db == 1, "incompatible shapes for broadcast");
        result[i] = da.max(db);
    }
    result
}

/// Sum out broadcast dimensions: reduce grad from grad_shape to target_shape
pub fn reduce_grad_to_shape(grad: &[f64], grad_shape: &[usize], target_shape: &[usize]) -> Vec<f64> {
    if grad_shape == target_shape {
        return grad.to_vec();
    }
    let target_numel: usize = target_shape.iter().product();
    // Pad target_shape on left with 1s
    let ndim = grad_shape.len();
    let mut padded = vec![1usize; ndim];
    let offset = ndim - target_shape.len();
    for i in 0..target_shape.len() {
        padded[offset + i] = target_shape[i];
    }

    let mut result = vec![0.0; target_numel];
    let grad_numel: usize = grad_shape.iter().product();

    for flat_idx in 0..grad_numel {
        // Convert flat_idx to multi-dim index in grad_shape
        let mut remaining = flat_idx;
        // Compute from right to left
        let mut coords = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            coords[d] = remaining % grad_shape[d];
            remaining /= grad_shape[d];
        }
        // Map to target index
        let mut tf = 0;
        let mut ts = 1;
        for d in (0..ndim).rev() {
            let c = if padded[d] == 1 { 0 } else { coords[d] };
            tf += c * ts;
            ts *= padded[d];
        }
        result[tf] += grad[flat_idx];
    }
    result
}

fn elementwise_broadcast(a_data: &[f64], a_shape: &[usize], b_data: &[f64], b_shape: &[usize], out_shape: &[usize]) -> (Vec<f64>, Vec<f64>) {
    // Return expanded a and b to out_shape
    let numel: usize = out_shape.iter().product();
    let ndim = out_shape.len();
    let a_pad = pad_shape(a_shape, ndim);
    let b_pad = pad_shape(b_shape, ndim);

    let mut a_exp = vec![0.0; numel];
    let mut b_exp = vec![0.0; numel];

    for i in 0..numel {
        let mut remaining = i;
        let mut coords = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            coords[d] = remaining % out_shape[d];
            remaining /= out_shape[d];
        }
        // compute flat indices
        let mut af = 0;
        let mut as_ = 1;
        for d in (0..ndim).rev() {
            let c = if a_pad[d] == 1 { 0 } else { coords[d] };
            af += c * as_;
            as_ *= a_pad[d];
        }
        let mut bf = 0;
        let mut bs = 1;
        for d in (0..ndim).rev() {
            let c = if b_pad[d] == 1 { 0 } else { coords[d] };
            bf += c * bs;
            bs *= b_pad[d];
        }
        a_exp[i] = a_data[af];
        b_exp[i] = b_data[bf];
    }
    (a_exp, b_exp)
}

fn pad_shape(shape: &[usize], ndim: usize) -> Vec<usize> {
    let mut padded = vec![1usize; ndim];
    let offset = ndim - shape.len();
    for i in 0..shape.len() {
        padded[offset + i] = shape[i];
    }
    padded
}

// ── TensorTape implementation ───────────────────────────────────────

impl TensorTape {
    pub fn new() -> Self {
        TensorTape { tensors: Vec::new(), ops: Vec::new() }
    }

    pub fn new_tensor(&mut self, data: Vec<f64>, shape: Vec<usize>, requires_grad: bool) -> usize {
        let numel: usize = shape.iter().product();
        assert_eq!(data.len(), numel, "data length must match shape");
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            data,
            shape,
            requires_grad,
            grad: None,
            tape_id: Some(id),
        });
        self.ops.push(TensorOp::Leaf);
        id
    }

    fn push(&mut self, data: Vec<f64>, shape: Vec<usize>, op: TensorOp) -> usize {
        let id = self.tensors.len();
        self.tensors.push(Tensor {
            data,
            shape,
            requires_grad: false,
            grad: None,
            tape_id: Some(id),
        });
        self.ops.push(op);
        id
    }

    // ── Forward ops ─────────────────────────────────────────────────

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let a_shape = &self.tensors[a].shape;
        let b_shape = &self.tensors[b].shape;
        assert_eq!(a_shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(b_shape.len(), 2, "matmul requires 2D tensors");
        let m = a_shape[0];
        let k = a_shape[1];
        assert_eq!(b_shape[0], k, "matmul inner dims must match");
        let n = b_shape[1];
        let data = matmul_2d(&self.tensors[a].data, &self.tensors[b].data, m, k, n);
        self.push(data, vec![m, n], TensorOp::MatMul { lhs: a, rhs: b, m, k, n })
    }

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.tensors[a].shape.clone();
        let b_shape = self.tensors[b].shape.clone();
        let out_shape = broadcast_shapes(&a_shape, &b_shape);
        let _numel: usize = out_shape.iter().product();

        let (a_exp, b_exp) = elementwise_broadcast(
            &self.tensors[a].data, &a_shape,
            &self.tensors[b].data, &b_shape,
            &out_shape,
        );
        let data: Vec<f64> = a_exp.iter().zip(b_exp.iter()).map(|(x, y)| x + y).collect();
        self.push(data, out_shape, TensorOp::Add { lhs: a, rhs: b })
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.tensors[a].shape.clone();
        let b_shape = self.tensors[b].shape.clone();
        let out_shape = broadcast_shapes(&a_shape, &b_shape);
        let (a_exp, b_exp) = elementwise_broadcast(
            &self.tensors[a].data, &a_shape,
            &self.tensors[b].data, &b_shape,
            &out_shape,
        );
        let data: Vec<f64> = a_exp.iter().zip(b_exp.iter()).map(|(x, y)| x - y).collect();
        self.push(data, out_shape, TensorOp::Sub { lhs: a, rhs: b })
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.tensors[a].shape.clone();
        let b_shape = self.tensors[b].shape.clone();
        let out_shape = broadcast_shapes(&a_shape, &b_shape);
        let (a_exp, b_exp) = elementwise_broadcast(
            &self.tensors[a].data, &a_shape,
            &self.tensors[b].data, &b_shape,
            &out_shape,
        );
        let data: Vec<f64> = a_exp.iter().zip(b_exp.iter()).map(|(x, y)| x * y).collect();
        self.push(data, out_shape, TensorOp::Mul { lhs: a, rhs: b })
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let a_shape = self.tensors[a].shape.clone();
        let b_shape = self.tensors[b].shape.clone();
        let out_shape = broadcast_shapes(&a_shape, &b_shape);
        let (a_exp, b_exp) = elementwise_broadcast(
            &self.tensors[a].data, &a_shape,
            &self.tensors[b].data, &b_shape,
            &out_shape,
        );
        let data: Vec<f64> = a_exp.iter().zip(b_exp.iter()).map(|(x, y)| x / y).collect();
        self.push(data, out_shape, TensorOp::Div { lhs: a, rhs: b })
    }

    pub fn relu(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| x.max(0.0)).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Relu { input: a })
    }

    pub fn sigmoid(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| 1.0 / (1.0 + (-x).exp())).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Sigmoid { input: a })
    }

    pub fn tanh_op(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| x.tanh()).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Tanh { input: a })
    }

    pub fn gelu(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| {
            0.5 * x * (1.0 + (((2.0 / PI).sqrt()) * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Gelu { input: a })
    }

    pub fn exp(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| x.exp()).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Exp { input: a })
    }

    pub fn log(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| x.ln()).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Log { input: a })
    }

    pub fn sqrt(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| x.sqrt()).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Sqrt { input: a })
    }

    pub fn neg(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.tensors[a].data.iter().map(|&x| -x).collect();
        let shape = self.tensors[a].shape.clone();
        self.push(data, shape, TensorOp::Neg { input: a })
    }

    pub fn sum(&mut self, a: usize, axis: Option<usize>) -> usize {
        let a_shape = &self.tensors[a].shape;
        let a_data = &self.tensors[a].data;
        match axis {
            None => {
                let s: f64 = a_data.iter().sum();
                self.push(vec![s], vec![1], TensorOp::Sum { input: a, axis: None })
            }
            Some(ax) => {
                let ndim = a_shape.len();
                assert!(ax < ndim);
                let mut out_shape: Vec<usize> = a_shape.to_vec();
                out_shape.remove(ax);
                if out_shape.is_empty() { out_shape.push(1); }
                let out_numel: usize = out_shape.iter().product();
                let mut result = vec![0.0; out_numel];

                let numel: usize = a_shape.iter().product();
                for i in 0..numel {
                    let mut remaining = i;
                    let mut coords = vec![0usize; ndim];
                    for d in (0..ndim).rev() {
                        coords[d] = remaining % a_shape[d];
                        remaining /= a_shape[d];
                    }
                    // Output index: remove axis
                    let mut out_coords: Vec<usize> = coords.clone();
                    out_coords.remove(ax);
                    if out_coords.is_empty() { out_coords.push(0); }
                    let mut out_idx = 0;
                    let mut stride = 1;
                    for d in (0..out_shape.len()).rev() {
                        out_idx += out_coords[d] * stride;
                        stride *= out_shape[d];
                    }
                    result[out_idx] += a_data[i];
                }
                self.push(result, out_shape, TensorOp::Sum { input: a, axis: Some(ax) })
            }
        }
    }

    pub fn mean(&mut self, a: usize, axis: Option<usize>) -> usize {
        let a_shape = &self.tensors[a].shape;
        let a_data = &self.tensors[a].data;
        match axis {
            None => {
                let n = a_data.len() as f64;
                let s: f64 = a_data.iter().sum();
                self.push(vec![s / n], vec![1], TensorOp::Mean { input: a, axis: None })
            }
            Some(ax) => {
                let ndim = a_shape.len();
                assert!(ax < ndim);
                let axis_size = a_shape[ax] as f64;
                let mut out_shape: Vec<usize> = a_shape.to_vec();
                out_shape.remove(ax);
                if out_shape.is_empty() { out_shape.push(1); }
                let out_numel: usize = out_shape.iter().product();
                let mut result = vec![0.0; out_numel];

                let numel: usize = a_shape.iter().product();
                for i in 0..numel {
                    let mut remaining = i;
                    let mut coords = vec![0usize; ndim];
                    for d in (0..ndim).rev() {
                        coords[d] = remaining % a_shape[d];
                        remaining /= a_shape[d];
                    }
                    let mut out_coords = coords.clone();
                    out_coords.remove(ax);
                    if out_coords.is_empty() { out_coords.push(0); }
                    let mut out_idx = 0;
                    let mut stride = 1;
                    for d in (0..out_shape.len()).rev() {
                        out_idx += out_coords[d] * stride;
                        stride *= out_shape[d];
                    }
                    result[out_idx] += a_data[i] / axis_size;
                }
                self.push(result, out_shape, TensorOp::Mean { input: a, axis: Some(ax) })
            }
        }
    }

    pub fn transpose(&mut self, a: usize) -> usize {
        let a_shape = &self.tensors[a].shape;
        assert_eq!(a_shape.len(), 2, "transpose requires 2D");
        let (rows, cols) = (a_shape[0], a_shape[1]);
        let a_data = &self.tensors[a].data;
        let mut data = vec![0.0; rows * cols];
        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = a_data[i * cols + j];
            }
        }
        self.push(data, vec![cols, rows], TensorOp::Transpose { input: a })
    }

    pub fn reshape(&mut self, a: usize, new_shape: Vec<usize>) -> usize {
        let numel: usize = new_shape.iter().product();
        assert_eq!(self.tensors[a].data.len(), numel, "reshape: numel mismatch");
        let data = self.tensors[a].data.clone();
        let ns = new_shape.clone();
        self.push(data, ns, TensorOp::Reshape { input: a, new_shape })
    }

    pub fn softmax(&mut self, a: usize, axis: usize) -> usize {
        let a_shape = self.tensors[a].shape.clone();
        let a_data = self.tensors[a].data.clone();
        let ndim = a_shape.len();
        assert!(axis < ndim);
        let numel: usize = a_shape.iter().product();
        let mut result = vec![0.0; numel];

        // Iterate over all "other" indices, compute softmax along axis
        let axis_size = a_shape[axis];
        let outer: usize = a_shape[..axis].iter().product();
        let inner: usize = a_shape[axis + 1..].iter().product();

        for o in 0..outer {
            for i in 0..inner {
                // Find max for numerical stability
                let mut max_val = f64::NEG_INFINITY;
                for a_idx in 0..axis_size {
                    let flat = o * axis_size * inner + a_idx * inner + i;
                    max_val = max_val.max(a_data[flat]);
                }
                let mut sum_exp = 0.0;
                for a_idx in 0..axis_size {
                    let flat = o * axis_size * inner + a_idx * inner + i;
                    let e = (a_data[flat] - max_val).exp();
                    result[flat] = e;
                    sum_exp += e;
                }
                for a_idx in 0..axis_size {
                    let flat = o * axis_size * inner + a_idx * inner + i;
                    result[flat] /= sum_exp;
                }
            }
        }
        self.push(result, a_shape, TensorOp::Softmax { input: a, axis })
    }

    pub fn layer_norm(&mut self, x: usize, gamma: usize, beta: usize, eps: f64) -> usize {
        // x: [..., D], gamma: [D], beta: [D]
        let x_shape = self.tensors[x].shape.clone();
        let x_data = self.tensors[x].data.clone();
        let gamma_data = self.tensors[gamma].data.clone();
        let beta_data = self.tensors[beta].data.clone();
        let d = *x_shape.last().unwrap();
        assert_eq!(gamma_data.len(), d);
        assert_eq!(beta_data.len(), d);
        let numel = x_data.len();
        let num_vecs = numel / d;
        let mut result = vec![0.0; numel];

        for v in 0..num_vecs {
            let start = v * d;
            let slice = &x_data[start..start + d];
            let mean: f64 = slice.iter().sum::<f64>() / d as f64;
            let var: f64 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / d as f64;
            let std = (var + eps).sqrt();
            for i in 0..d {
                result[start + i] = (slice[i] - mean) / std * gamma_data[i] + beta_data[i];
            }
        }
        self.push(result, x_shape, TensorOp::LayerNorm { input: x, gamma, beta, eps })
    }

    pub fn cross_entropy_loss(&mut self, logits: usize, targets: Vec<usize>) -> usize {
        // logits: [B, C], targets: [B] (class indices)
        let logits_shape = self.tensors[logits].shape.clone();
        let logits_data = self.tensors[logits].data.clone();
        assert_eq!(logits_shape.len(), 2);
        let batch = logits_shape[0];
        let classes = logits_shape[1];
        assert_eq!(targets.len(), batch);

        let mut total_loss = 0.0;
        for b in 0..batch {
            let row = &logits_data[b * classes..(b + 1) * classes];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|&x| (x - max_val).exp()).sum();
            let log_softmax = row[targets[b]] - max_val - exp_sum.ln();
            total_loss -= log_softmax;
        }
        total_loss /= batch as f64;
        let tgts = targets.clone();
        self.push(vec![total_loss], vec![1], TensorOp::CrossEntropyLoss { logits, targets: tgts })
    }

    // ── Backward ────────────────────────────────────────────────────

    pub fn backward(&mut self, loss_id: usize) {
        // Initialize loss gradient
        let n = self.tensors.len();
        // Allocate grads for all tensors
        for i in 0..n {
            let numel = self.tensors[i].data.len();
            if self.tensors[i].grad.is_none() {
                self.tensors[i].grad = Some(vec![0.0; numel]);
            }
        }
        self.tensors[loss_id].grad = Some(vec![1.0; self.tensors[loss_id].data.len()]);

        for i in (0..n).rev() {
            let grad = self.tensors[i].grad.clone().unwrap();
            let op = self.ops[i].clone();
            match op {
                TensorOp::Leaf => {}

                TensorOp::MatMul { lhs, rhs, m, k, n } => {
                    // dL/dA = dL/dC @ B^T
                    let b_data = &self.tensors[rhs].data;
                    let a_data = &self.tensors[lhs].data;
                    // B^T: [n, k]
                    let mut bt = vec![0.0; k * n];
                    for ii in 0..k { for jj in 0..n { bt[jj * k + ii] = b_data[ii * n + jj]; } }
                    let da = matmul_2d(&grad, &bt, m, n, k);
                    // dL/dB = A^T @ dL/dC
                    let mut at = vec![0.0; k * m];
                    for ii in 0..m { for jj in 0..k { at[jj * m + ii] = a_data[ii * k + jj]; } }
                    let db = matmul_2d(&at, &grad, k, m, n);

                    add_grad(&mut self.tensors[lhs].grad, &da);
                    add_grad(&mut self.tensors[rhs].grad, &db);
                }

                TensorOp::Add { lhs, rhs } => {
                    let out_shape = self.tensors[i].shape.clone();
                    let lhs_shape = self.tensors[lhs].shape.clone();
                    let rhs_shape = self.tensors[rhs].shape.clone();
                    let da = reduce_grad_to_shape(&grad, &out_shape, &lhs_shape);
                    let db = reduce_grad_to_shape(&grad, &out_shape, &rhs_shape);
                    add_grad(&mut self.tensors[lhs].grad, &da);
                    add_grad(&mut self.tensors[rhs].grad, &db);
                }

                TensorOp::Sub { lhs, rhs } => {
                    let out_shape = self.tensors[i].shape.clone();
                    let lhs_shape = self.tensors[lhs].shape.clone();
                    let rhs_shape = self.tensors[rhs].shape.clone();
                    let da = reduce_grad_to_shape(&grad, &out_shape, &lhs_shape);
                    let neg_grad: Vec<f64> = grad.iter().map(|&g| -g).collect();
                    let db = reduce_grad_to_shape(&neg_grad, &out_shape, &rhs_shape);
                    add_grad(&mut self.tensors[lhs].grad, &da);
                    add_grad(&mut self.tensors[rhs].grad, &db);
                }

                TensorOp::Mul { lhs, rhs } => {
                    let out_shape = self.tensors[i].shape.clone();
                    let lhs_shape = self.tensors[lhs].shape.clone();
                    let rhs_shape = self.tensors[rhs].shape.clone();
                    let (a_exp, b_exp) = elementwise_broadcast(
                        &self.tensors[lhs].data, &lhs_shape,
                        &self.tensors[rhs].data, &rhs_shape,
                        &out_shape,
                    );
                    let da_full: Vec<f64> = grad.iter().zip(b_exp.iter()).map(|(g, b)| g * b).collect();
                    let db_full: Vec<f64> = grad.iter().zip(a_exp.iter()).map(|(g, a)| g * a).collect();
                    let da = reduce_grad_to_shape(&da_full, &out_shape, &lhs_shape);
                    let db = reduce_grad_to_shape(&db_full, &out_shape, &rhs_shape);
                    add_grad(&mut self.tensors[lhs].grad, &da);
                    add_grad(&mut self.tensors[rhs].grad, &db);
                }

                TensorOp::Div { lhs, rhs } => {
                    let out_shape = self.tensors[i].shape.clone();
                    let lhs_shape = self.tensors[lhs].shape.clone();
                    let rhs_shape = self.tensors[rhs].shape.clone();
                    let (a_exp, b_exp) = elementwise_broadcast(
                        &self.tensors[lhs].data, &lhs_shape,
                        &self.tensors[rhs].data, &rhs_shape,
                        &out_shape,
                    );
                    // d(a/b)/da = 1/b, d(a/b)/db = -a/b^2
                    let da_full: Vec<f64> = grad.iter().zip(b_exp.iter()).map(|(g, b)| g / b).collect();
                    let db_full: Vec<f64> = grad.iter().zip(a_exp.iter().zip(b_exp.iter()))
                        .map(|(g, (a, b))| -g * a / (b * b)).collect();
                    let da = reduce_grad_to_shape(&da_full, &out_shape, &lhs_shape);
                    let db = reduce_grad_to_shape(&db_full, &out_shape, &rhs_shape);
                    add_grad(&mut self.tensors[lhs].grad, &da);
                    add_grad(&mut self.tensors[rhs].grad, &db);
                }

                TensorOp::Relu { input } => {
                    let input_data = &self.tensors[input].data;
                    let dg: Vec<f64> = grad.iter().zip(input_data.iter())
                        .map(|(g, &x)| if x > 0.0 { *g } else { 0.0 }).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Sigmoid { input } => {
                    let out_data = &self.tensors[i].data;
                    let dg: Vec<f64> = grad.iter().zip(out_data.iter())
                        .map(|(g, &s)| g * s * (1.0 - s)).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Tanh { input } => {
                    let out_data = &self.tensors[i].data;
                    let dg: Vec<f64> = grad.iter().zip(out_data.iter())
                        .map(|(g, &t)| g * (1.0 - t * t)).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Gelu { input } => {
                    // Approximate GELU gradient
                    let x_data = &self.tensors[input].data;
                    let dg: Vec<f64> = grad.iter().zip(x_data.iter()).map(|(g, &x)| {
                        let c = (2.0 / PI).sqrt();
                        let inner = c * (x + 0.044715 * x.powi(3));
                        let tanh_val = inner.tanh();
                        let sech2 = 1.0 - tanh_val * tanh_val;
                        let d_inner = c * (1.0 + 3.0 * 0.044715 * x * x);
                        g * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
                    }).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Exp { input } => {
                    let out_data = &self.tensors[i].data;
                    let dg: Vec<f64> = grad.iter().zip(out_data.iter())
                        .map(|(g, &e)| g * e).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Log { input } => {
                    let input_data = &self.tensors[input].data;
                    let dg: Vec<f64> = grad.iter().zip(input_data.iter())
                        .map(|(g, &x)| g / x).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Sqrt { input } => {
                    let out_data = &self.tensors[i].data;
                    let dg: Vec<f64> = grad.iter().zip(out_data.iter())
                        .map(|(g, &s)| g / (2.0 * s)).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Neg { input } => {
                    let dg: Vec<f64> = grad.iter().map(|g| -g).collect();
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Sum { input, axis } => {
                    let input_shape = self.tensors[input].shape.clone();
                    let input_numel: usize = input_shape.iter().product();
                    match axis {
                        None => {
                            // Broadcast scalar grad to full shape
                            let g = grad[0];
                            let dg = vec![g; input_numel];
                            add_grad(&mut self.tensors[input].grad, &dg);
                        }
                        Some(ax) => {
                            // Broadcast grad back along axis
                            let ndim = input_shape.len();
                            let mut dg = vec![0.0; input_numel];
                            let out_shape = self.tensors[i].shape.clone();
                            for idx in 0..input_numel {
                                let mut remaining = idx;
                                let mut coords = vec![0usize; ndim];
                                for d in (0..ndim).rev() {
                                    coords[d] = remaining % input_shape[d];
                                    remaining /= input_shape[d];
                                }
                                let mut out_coords = coords.clone();
                                out_coords.remove(ax);
                                if out_coords.is_empty() { out_coords.push(0); }
                                let mut out_idx = 0;
                                let mut stride = 1;
                                for d in (0..out_shape.len()).rev() {
                                    out_idx += out_coords[d] * stride;
                                    stride *= out_shape[d];
                                }
                                dg[idx] = grad[out_idx];
                            }
                            add_grad(&mut self.tensors[input].grad, &dg);
                        }
                    }
                }

                TensorOp::Mean { input, axis } => {
                    let input_shape = self.tensors[input].shape.clone();
                    let input_numel: usize = input_shape.iter().product();
                    match axis {
                        None => {
                            let n = input_numel as f64;
                            let g = grad[0] / n;
                            let dg = vec![g; input_numel];
                            add_grad(&mut self.tensors[input].grad, &dg);
                        }
                        Some(ax) => {
                            let ndim = input_shape.len();
                            let axis_size = input_shape[ax] as f64;
                            let mut dg = vec![0.0; input_numel];
                            let out_shape = self.tensors[i].shape.clone();
                            for idx in 0..input_numel {
                                let mut remaining = idx;
                                let mut coords = vec![0usize; ndim];
                                for d in (0..ndim).rev() {
                                    coords[d] = remaining % input_shape[d];
                                    remaining /= input_shape[d];
                                }
                                let mut out_coords = coords.clone();
                                out_coords.remove(ax);
                                if out_coords.is_empty() { out_coords.push(0); }
                                let mut out_idx = 0;
                                let mut stride = 1;
                                for d in (0..out_shape.len()).rev() {
                                    out_idx += out_coords[d] * stride;
                                    stride *= out_shape[d];
                                }
                                dg[idx] = grad[out_idx] / axis_size;
                            }
                            add_grad(&mut self.tensors[input].grad, &dg);
                        }
                    }
                }

                TensorOp::Max { input, axis: _ } => {
                    // Not computing backward for max for now (used rarely alone)
                    let _ = input;
                }

                TensorOp::Reshape { input, new_shape: _ } => {
                    // Gradient passes through with same reshaping back
                    let _input_shape = self.tensors[input].shape.clone();
                    // grad has out shape, just reshape back
                    let dg = grad.clone(); // same data, different logical shape
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Transpose { input } => {
                    // Transpose the gradient back
                    let out_shape = self.tensors[i].shape.clone();
                    let (rows, cols) = (out_shape[0], out_shape[1]);
                    let mut dg = vec![0.0; grad.len()];
                    for r in 0..rows {
                        for c in 0..cols {
                            dg[c * rows + r] = grad[r * cols + c];
                        }
                    }
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::Softmax { input, axis } => {
                    // Jacobian-vector product: ds_i = s_i * (dL_i - sum_j(dL_j * s_j))
                    let out_data = self.tensors[i].data.clone();
                    let shape = self.tensors[i].shape.clone();
                    let _ndim = shape.len();
                    let axis_size = shape[axis];
                    let outer: usize = shape[..axis].iter().product();
                    let inner: usize = shape[axis + 1..].iter().product();
                    let mut dg = vec![0.0; out_data.len()];

                    for o in 0..outer {
                        for ii in 0..inner {
                            let mut dot = 0.0;
                            for a in 0..axis_size {
                                let flat = o * axis_size * inner + a * inner + ii;
                                dot += grad[flat] * out_data[flat];
                            }
                            for a in 0..axis_size {
                                let flat = o * axis_size * inner + a * inner + ii;
                                dg[flat] = out_data[flat] * (grad[flat] - dot);
                            }
                        }
                    }
                    add_grad(&mut self.tensors[input].grad, &dg);
                }

                TensorOp::LayerNorm { input, gamma, beta, eps } => {
                    let x_shape = self.tensors[input].shape.clone();
                    let x_data = self.tensors[input].data.clone();
                    let gamma_data = self.tensors[gamma].data.clone();
                    let d = *x_shape.last().unwrap();
                    let num_vecs = x_data.len() / d;

                    let mut dx = vec![0.0; x_data.len()];
                    let mut dgamma = vec![0.0; d];
                    let mut dbeta = vec![0.0; d];

                    for v in 0..num_vecs {
                        let start = v * d;
                        let x_slice = &x_data[start..start + d];
                        let g_slice = &grad[start..start + d];

                        let mean: f64 = x_slice.iter().sum::<f64>() / d as f64;
                        let var: f64 = x_slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / d as f64;
                        let std = (var + eps).sqrt();
                        let inv_std = 1.0 / std;

                        let xhat: Vec<f64> = x_slice.iter().map(|&x| (x - mean) * inv_std).collect();

                        for j in 0..d {
                            dgamma[j] += g_slice[j] * xhat[j];
                            dbeta[j] += g_slice[j];
                        }

                        // dx computation (standard layernorm backward)
                        let mut dxhat = vec![0.0; d];
                        for j in 0..d {
                            dxhat[j] = g_slice[j] * gamma_data[j];
                        }

                        let dxhat_sum: f64 = dxhat.iter().sum();
                        let dxhat_xhat_sum: f64 = dxhat.iter().zip(xhat.iter()).map(|(a, b)| a * b).sum();

                        let d_f = d as f64;
                        for j in 0..d {
                            dx[start + j] = inv_std / d_f * (d_f * dxhat[j] - dxhat_sum - xhat[j] * dxhat_xhat_sum);
                        }
                    }

                    add_grad(&mut self.tensors[input].grad, &dx);
                    add_grad(&mut self.tensors[gamma].grad, &dgamma);
                    add_grad(&mut self.tensors[beta].grad, &dbeta);
                }

                TensorOp::CrossEntropyLoss { logits, targets } => {
                    let logits_shape = self.tensors[logits].shape.clone();
                    let logits_data = self.tensors[logits].data.clone();
                    let batch = logits_shape[0];
                    let classes = logits_shape[1];
                    let mut dlogits = vec![0.0; batch * classes];

                    for b in 0..batch {
                        let row = &logits_data[b * classes..(b + 1) * classes];
                        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = row.iter().map(|&x| (x - max_val).exp()).collect();
                        let sum_exp: f64 = exps.iter().sum();
                        for c in 0..classes {
                            let softmax_c = exps[c] / sum_exp;
                            let target_val = if c == targets[b] { 1.0 } else { 0.0 };
                            dlogits[b * classes + c] = grad[0] * (softmax_c - target_val) / batch as f64;
                        }
                    }
                    add_grad(&mut self.tensors[logits].grad, &dlogits);
                }

                TensorOp::BroadcastAdd { tensor, bias } => {
                    let out_shape = self.tensors[i].shape.clone();
                    let t_shape = self.tensors[tensor].shape.clone();
                    let b_shape = self.tensors[bias].shape.clone();
                    let dt = reduce_grad_to_shape(&grad, &out_shape, &t_shape);
                    let db = reduce_grad_to_shape(&grad, &out_shape, &b_shape);
                    add_grad(&mut self.tensors[tensor].grad, &dt);
                    add_grad(&mut self.tensors[bias].grad, &db);
                }
            }
        }
    }

    // ── Optimizers ──────────────────────────────────────────────────

    pub fn sgd_step(&mut self, params: &[usize], lr: f64) {
        for &p in params {
            if let Some(ref g) = self.tensors[p].grad.clone() {
                for (d, gv) in self.tensors[p].data.iter_mut().zip(g.iter()) {
                    *d -= lr * gv;
                }
            }
        }
    }

    pub fn adam_step(
        &mut self,
        params: &[usize],
        lr: f64,
        beta1: f64,
        beta2: f64,
        eps: f64,
        t: usize,
        m: &mut Vec<Vec<f64>>,
        v: &mut Vec<Vec<f64>>,
    ) {
        let t_f = t as f64;
        for (idx, &p) in params.iter().enumerate() {
            if let Some(ref g) = self.tensors[p].grad.clone() {
                let n = g.len();
                if m[idx].is_empty() { m[idx] = vec![0.0; n]; }
                if v[idx].is_empty() { v[idx] = vec![0.0; n]; }
                for i in 0..n {
                    m[idx][i] = beta1 * m[idx][i] + (1.0 - beta1) * g[i];
                    v[idx][i] = beta2 * v[idx][i] + (1.0 - beta2) * g[i] * g[i];
                    let m_hat = m[idx][i] / (1.0 - beta1.powf(t_f));
                    let v_hat = v[idx][i] / (1.0 - beta2.powf(t_f));
                    self.tensors[p].data[i] -= lr * m_hat / (v_hat.sqrt() + eps);
                }
            }
        }
    }

    pub fn zero_grad(&mut self, params: &[usize]) {
        for &p in params {
            if let Some(ref mut g) = self.tensors[p].grad {
                for v in g.iter_mut() {
                    *v = 0.0;
                }
            }
        }
    }

    // ── Accessors ───────────────────────────────────────────────────

    pub fn data(&self, id: usize) -> &[f64] {
        &self.tensors[id].data
    }

    pub fn grad(&self, id: usize) -> Option<&[f64]> {
        self.tensors[id].grad.as_deref()
    }

    pub fn shape(&self, id: usize) -> &[usize] {
        &self.tensors[id].shape
    }
}

fn add_grad(target: &mut Option<Vec<f64>>, src: &[f64]) {
    if let Some(ref mut t) = target {
        for (a, b) in t.iter_mut().zip(src.iter()) {
            *a += b;
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: &[f64], b: &[f64], tol: f64) -> bool {
        a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    fn numerical_grad(
        tape_fn: &dyn Fn(&mut TensorTape, &[Vec<f64>]) -> (usize, Vec<usize>),
        param_data: &[Vec<f64>],
        _param_shapes: &[Vec<usize>],
        eps: f64,
    ) -> Vec<Vec<f64>> {
        let mut grads = Vec::new();
        for p in 0..param_data.len() {
            let mut pg = vec![0.0; param_data[p].len()];
            for i in 0..param_data[p].len() {
                // f(x + eps)
                let mut data_plus = param_data.to_vec();
                data_plus[p][i] += eps;
                let mut tape_plus = TensorTape::new();
                let (loss_plus, _) = tape_fn(&mut tape_plus, &data_plus);
                let lp = tape_plus.data(loss_plus)[0];

                // f(x - eps)
                let mut data_minus = param_data.to_vec();
                data_minus[p][i] -= eps;
                let mut tape_minus = TensorTape::new();
                let (loss_minus, _) = tape_fn(&mut tape_minus, &data_minus);
                let lm = tape_minus.data(loss_minus)[0];

                pg[i] = (lp - lm) / (2.0 * eps);
            }
            grads.push(pg);
        }
        grads
    }

    #[test]
    fn test_matmul_forward() {
        let mut tape = TensorTape::new();
        // A: 2x3, B: 3x4
        let a = tape.new_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let b = tape.new_tensor(
            vec![1.0, 2.0, 3.0, 4.0,
                 5.0, 6.0, 7.0, 8.0,
                 9.0, 10.0, 11.0, 12.0],
            vec![3, 4], true,
        );
        let c = tape.matmul(a, b);
        assert_eq!(tape.shape(c), &[2, 4]);
        // Row 0: [1*1+2*5+3*9, 1*2+2*6+3*10, 1*3+2*7+3*11, 1*4+2*8+3*12] = [38, 44, 50, 56]
        // Row 1: [4*1+5*5+6*9, 4*2+5*6+6*10, 4*3+5*7+6*11, 4*4+5*8+6*12] = [83, 98, 113, 128]
        assert!(approx_eq(tape.data(c), &[38.0, 44.0, 50.0, 56.0, 83.0, 98.0, 113.0, 128.0], 1e-10));
    }

    #[test]
    fn test_matmul_backward() {
        let a_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data = vec![0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
        let param_data = vec![a_data.clone(), b_data.clone()];
        let param_shapes = vec![vec![2, 3], vec![3, 2]];

        let build = |tape: &mut TensorTape, data: &[Vec<f64>]| -> (usize, Vec<usize>) {
            let a = tape.new_tensor(data[0].clone(), vec![2, 3], true);
            let b = tape.new_tensor(data[1].clone(), vec![3, 2], true);
            let c = tape.matmul(a, b);
            let loss = tape.sum(c, None);
            (loss, vec![a, b])
        };

        // Analytical
        let mut tape = TensorTape::new();
        let (loss, params) = build(&mut tape, &param_data);
        tape.backward(loss);
        let ga = tape.grad(params[0]).unwrap().to_vec();
        let gb = tape.grad(params[1]).unwrap().to_vec();

        // Numerical
        let ng = numerical_grad(&build, &param_data, &param_shapes, 1e-5);
        assert!(approx_eq(&ga, &ng[0], 1e-4), "matmul dA mismatch: {:?} vs {:?}", ga, ng[0]);
        assert!(approx_eq(&gb, &ng[1], 1e-4), "matmul dB mismatch: {:?} vs {:?}", gb, ng[1]);
    }

    #[test]
    fn test_linear_layer_backward() {
        // y = x @ W + b
        let x_data = vec![1.0, 2.0, 3.0];    // [1, 3]
        let w_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [3, 2]
        let b_data = vec![0.01, 0.02]; // [2]
        let param_data = vec![x_data.clone(), w_data.clone(), b_data.clone()];

        let build = |tape: &mut TensorTape, data: &[Vec<f64>]| -> (usize, Vec<usize>) {
            let x = tape.new_tensor(data[0].clone(), vec![1, 3], true);
            let w = tape.new_tensor(data[1].clone(), vec![3, 2], true);
            let b = tape.new_tensor(data[2].clone(), vec![1, 2], true);
            let xw = tape.matmul(x, w);
            let y = tape.add(xw, b);
            let loss = tape.sum(y, None);
            (loss, vec![x, w, b])
        };

        let mut tape = TensorTape::new();
        let (loss, params) = build(&mut tape, &param_data);
        tape.backward(loss);

        let gw = tape.grad(params[1]).unwrap().to_vec();
        let gb = tape.grad(params[2]).unwrap().to_vec();

        let ng = numerical_grad(&build, &param_data, &[vec![1, 3], vec![3, 2], vec![1, 2]], 1e-5);
        assert!(approx_eq(&gw, &ng[1], 1e-4), "linear dW: {:?} vs {:?}", gw, ng[1]);
        assert!(approx_eq(&gb, &ng[2], 1e-4), "linear db: {:?} vs {:?}", gb, ng[2]);
    }

    #[test]
    fn test_softmax() {
        let mut tape = TensorTape::new();
        let a = tape.new_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2], true);
        let s = tape.softmax(a, 1);
        let out = tape.data(s).to_vec();
        // Check rows sum to 1
        assert!((out[0] + out[1] - 1.0).abs() < 1e-10);
        assert!((out[2] + out[3] - 1.0).abs() < 1e-10);

        // Check gradients via numerical
        let build = |tape: &mut TensorTape, data: &[Vec<f64>]| -> (usize, Vec<usize>) {
            let a = tape.new_tensor(data[0].clone(), vec![2, 2], true);
            let s = tape.softmax(a, 1);
            let loss = tape.sum(s, None);
            (loss, vec![a])
        };
        let param_data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let mut t = TensorTape::new();
        let (loss, params) = build(&mut t, &param_data);
        t.backward(loss);
        let ga = t.grad(params[0]).unwrap().to_vec();
        let ng = numerical_grad(&build, &param_data, &[vec![2, 2]], 1e-5);
        assert!(approx_eq(&ga, &ng[0], 1e-4), "softmax grad: {:?} vs {:?}", ga, ng[0]);
    }

    #[test]
    fn test_layer_norm() {
        let mut tape = TensorTape::new();
        let x = tape.new_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3], true);
        let gamma = tape.new_tensor(vec![1.0, 1.0, 1.0], vec![3], true);
        let beta = tape.new_tensor(vec![0.0, 0.0, 0.0], vec![3], true);
        let ln = tape.layer_norm(x, gamma, beta, 1e-5);
        let out = tape.data(ln).to_vec();
        // Each row should have mean ~ 0, std ~ 1
        let row0_mean = (out[0] + out[1] + out[2]) / 3.0;
        let row1_mean = (out[3] + out[4] + out[5]) / 3.0;
        assert!(row0_mean.abs() < 1e-5, "LN mean: {}", row0_mean);
        assert!(row1_mean.abs() < 1e-5, "LN mean: {}", row1_mean);

        // Check gradients
        let build = |tape: &mut TensorTape, data: &[Vec<f64>]| -> (usize, Vec<usize>) {
            let x = tape.new_tensor(data[0].clone(), vec![2, 3], true);
            let g = tape.new_tensor(data[1].clone(), vec![3], true);
            let b = tape.new_tensor(data[2].clone(), vec![3], true);
            let ln = tape.layer_norm(x, g, b, 1e-5);
            let loss = tape.sum(ln, None);
            (loss, vec![x, g, b])
        };
        let param_data = vec![
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ];
        let mut t = TensorTape::new();
        let (loss, params) = build(&mut t, &param_data);
        t.backward(loss);
        let gx = t.grad(params[0]).unwrap().to_vec();
        let gg = t.grad(params[1]).unwrap().to_vec();
        let ng = numerical_grad(&build, &param_data, &[vec![2, 3], vec![3], vec![3]], 1e-5);
        assert!(approx_eq(&gx, &ng[0], 1e-3), "LN dx: {:?} vs {:?}", gx, ng[0]);
        assert!(approx_eq(&gg, &ng[1], 1e-3), "LN dgamma: {:?} vs {:?}", gg, ng[1]);
    }

    #[test]
    fn test_cross_entropy() {
        let mut tape = TensorTape::new();
        let logits = tape.new_tensor(vec![2.0, 1.0, 0.1, 0.1, 2.0, 1.0], vec![2, 3], true);
        let loss = tape.cross_entropy_loss(logits, vec![0, 1]);
        let l = tape.data(loss)[0];
        // Manual: for row 0 target=0, for row 1 target=1
        // Row 0: log_softmax[0] = 2.0 - log(e^2 + e^1 + e^0.1)
        let r0_lse = (2.0_f64.exp() + 1.0_f64.exp() + 0.1_f64.exp()).ln();
        let r0_loss = -(2.0 - r0_lse);
        let r1_lse = (0.1_f64.exp() + 2.0_f64.exp() + 1.0_f64.exp()).ln();
        let r1_loss = -(2.0 - r1_lse);
        let expected = (r0_loss + r1_loss) / 2.0;
        assert!((l - expected).abs() < 1e-10, "CE loss: {} vs {}", l, expected);
    }

    #[test]
    fn test_mlp_xor() {
        // 2-layer MLP on XOR: inputs [0,0],[0,1],[1,0],[1,1] -> targets [0,1,1,0]
        let inputs = vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ];
        let targets = vec![0.0, 1.0, 1.0, 0.0];

        // W1: [2,4], b1: [1,4], W2: [4,1], b2: [1,1]
        // Fixed seed initialization
        let mut w1_data = vec![0.5, -0.3, 0.8, -0.6, 0.4, 0.7, -0.5, 0.2];
        let mut b1_data = vec![0.0, 0.0, 0.0, 0.0];
        let mut w2_data = vec![0.6, -0.4, 0.3, 0.8];
        let mut b2_data = vec![0.0];

        let lr = 0.5;
        let mut initial_loss = 0.0;
        let mut final_loss = 0.0;

        for step in 0..100 {
            let mut tape = TensorTape::new();
            let w1 = tape.new_tensor(w1_data.clone(), vec![2, 4], true);
            let b1 = tape.new_tensor(b1_data.clone(), vec![1, 4], true);
            let w2 = tape.new_tensor(w2_data.clone(), vec![4, 1], true);
            let b2 = tape.new_tensor(b2_data.clone(), vec![1, 1], true);

            let _total_loss_val = 0.0;
            let mut loss_ids = Vec::new();

            for (inp, &tgt) in inputs.iter().zip(targets.iter()) {
                let x = tape.new_tensor(inp.clone(), vec![1, 2], false);
                let h = tape.matmul(x, w1);
                let h = tape.add(h, b1);
                let h = tape.relu(h);
                let o = tape.matmul(h, w2);
                let o = tape.add(o, b2);
                // MSE: (o - target)^2
                let tgt_t = tape.new_tensor(vec![tgt], vec![1, 1], false);
                let diff = tape.sub(o, tgt_t);
                let sq = tape.mul(diff, diff);
                loss_ids.push(sq);
            }

            // Sum all losses
            // Manually sum the scalar losses
            let mut total = loss_ids[0];
            for i in 1..loss_ids.len() {
                total = tape.add(total, loss_ids[i]);
            }
            // Mean
            let four = tape.new_tensor(vec![4.0], vec![1, 1], false);
            let loss = tape.div(total, four);

            let loss_val = tape.data(loss)[0];
            if step == 0 { initial_loss = loss_val; }
            if step == 99 { final_loss = loss_val; }

            tape.backward(loss);

            // SGD update
            let gw1 = tape.grad(w1).unwrap().to_vec();
            let gb1 = tape.grad(b1).unwrap().to_vec();
            let gw2 = tape.grad(w2).unwrap().to_vec();
            let gb2 = tape.grad(b2).unwrap().to_vec();

            for (p, g) in w1_data.iter_mut().zip(gw1.iter()) { *p -= lr * g; }
            for (p, g) in b1_data.iter_mut().zip(gb1.iter()) { *p -= lr * g; }
            for (p, g) in w2_data.iter_mut().zip(gw2.iter()) { *p -= lr * g; }
            for (p, g) in b2_data.iter_mut().zip(gb2.iter()) { *p -= lr * g; }
        }

        assert!(final_loss < initial_loss, "Loss should decrease: {} -> {}", initial_loss, final_loss);
        assert!(final_loss < 0.1, "Final loss should be small: {}", final_loss);
    }

    #[test]
    fn test_transformer_block_gradients() {
        // Self-attention + FFN block, verify gradients match numerical
        // Simplified: seq_len=2, d_model=3, d_ff=4
        let seq_len = 2;
        let d_model = 3;

        let x_data: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]; // [2,3]
        let wq_data: Vec<f64> = vec![0.1, -0.1, 0.2, -0.2, 0.3, -0.3, 0.1, 0.2, -0.1]; // [3,3]
        let wk_data: Vec<f64> = vec![-0.1, 0.2, -0.2, 0.1, -0.3, 0.3, 0.2, -0.1, 0.1]; // [3,3]
        let wv_data: Vec<f64> = vec![0.2, -0.1, 0.1, -0.1, 0.2, -0.2, 0.3, 0.1, -0.1]; // [3,3]

        let param_data = vec![x_data.clone(), wq_data.clone(), wk_data.clone(), wv_data.clone()];

        let build = |tape: &mut TensorTape, data: &[Vec<f64>]| -> (usize, Vec<usize>) {
            let x = tape.new_tensor(data[0].clone(), vec![seq_len, d_model], true);
            let wq = tape.new_tensor(data[1].clone(), vec![d_model, d_model], true);
            let wk = tape.new_tensor(data[2].clone(), vec![d_model, d_model], true);
            let wv = tape.new_tensor(data[3].clone(), vec![d_model, d_model], true);

            // Q = X @ Wq, K = X @ Wk, V = X @ Wv
            let q = tape.matmul(x, wq);
            let k = tape.matmul(x, wk);
            let v = tape.matmul(x, wv);

            // Attention scores = softmax(Q @ K^T / sqrt(d))
            let kt = tape.transpose(k);
            let scores = tape.matmul(q, kt);
            // Scale by 1/sqrt(d_model) via element-wise mul with scalar
            let scale_val = 1.0 / (d_model as f64).sqrt();
            let scale = tape.new_tensor(vec![scale_val; seq_len * seq_len], vec![seq_len, seq_len], false);
            let scores = tape.mul(scores, scale);
            let attn = tape.softmax(scores, 1);

            // Output = attn @ V
            let out = tape.matmul(attn, v);
            let loss = tape.sum(out, None);
            (loss, vec![x, wq, wk, wv])
        };

        let mut tape = TensorTape::new();
        let (loss, params) = build(&mut tape, &param_data);
        tape.backward(loss);

        let shapes = vec![vec![seq_len, d_model], vec![d_model, d_model], vec![d_model, d_model], vec![d_model, d_model]];
        let ng = numerical_grad(&build, &param_data, &shapes, 1e-5);

        for (i, name) in ["x", "Wq", "Wk", "Wv"].iter().enumerate() {
            let ag = tape.grad(params[i]).unwrap();
            let max_diff = ag.iter().zip(ng[i].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f64, f64::max);
            assert!(max_diff < 1e-3, "Transformer {} grad max_diff={}: analytical={:?} numerical={:?}", name, max_diff, ag, ng[i]);
        }
    }
}
