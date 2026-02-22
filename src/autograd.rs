/// Tape-based automatic differentiation engine for Vortex.
///
/// Implements a Wengert list (computation tape) that records forward operations
/// and replays them in reverse to compute gradients via the chain rule.
/// Supports tensor operations, neural network primitives, and higher-order gradients.

use std::cell::RefCell;
use std::rc::Rc;
use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Op enum — every operation the tape can record
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub enum Op {
    // Leaf nodes
    Parameter,
    Input,
    Constant,

    // Unary
    Neg,
    Exp,
    Log,
    Sqrt,
    Abs,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Softmax { axis: usize },
    Sin,
    Cos,
    Sum { axis: Option<usize> },
    Mean { axis: Option<usize> },
    Reshape { new_shape: Vec<usize> },
    Transpose,

    // Binary
    Add,
    Sub,
    Mul,
    Div,
    MatMul,
    Pow,

    // Neural net
    Conv2d { stride: usize, padding: usize, kh: usize, kw: usize },
    LayerNorm { eps: f64, normalized_shape: usize },
    Dropout { rate: f64, training: bool, mask: Vec<bool> },
    Embedding { vocab_size: usize },
    CrossEntropy,
    Mse,
    BroadcastAdd,
}

// ---------------------------------------------------------------------------
// TapeNode — single entry in the computation tape
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TapeNode {
    pub id: usize,
    pub op: Op,
    pub inputs: Vec<usize>,
    pub output_shape: Vec<usize>,
    pub output_data: Vec<f64>,
    pub grad: Option<Vec<f64>>,
    pub requires_grad: bool,
}

// ---------------------------------------------------------------------------
// Tape — the Wengert list
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct Tape {
    pub nodes: Vec<TapeNode>,
}

impl Tape {
    pub fn new() -> Self {
        Tape { nodes: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    fn push_node(&mut self, op: Op, inputs: Vec<usize>, shape: Vec<usize>, data: Vec<f64>, requires_grad: bool) -> usize {
        let id = self.nodes.len();
        self.nodes.push(TapeNode {
            id,
            op,
            inputs,
            output_shape: shape,
            output_data: data,
            grad: None,
            requires_grad,
        });
        id
    }

    // ----- Leaf constructors -----

    pub fn parameter(&mut self, data: Vec<f64>, shape: Vec<usize>) -> usize {
        self.push_node(Op::Parameter, vec![], shape, data, true)
    }

    pub fn input(&mut self, data: Vec<f64>, shape: Vec<usize>) -> usize {
        self.push_node(Op::Input, vec![], shape, data, false)
    }

    pub fn constant(&mut self, data: Vec<f64>, shape: Vec<usize>) -> usize {
        self.push_node(Op::Constant, vec![], shape, data, false)
    }

    // ----- Unary ops -----

    pub fn neg(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| -x).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Neg, vec![a], shape, data, rg)
    }

    pub fn exp(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.exp()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Exp, vec![a], shape, data, rg)
    }

    pub fn log(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.ln()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Log, vec![a], shape, data, rg)
    }

    pub fn sqrt(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.sqrt()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Sqrt, vec![a], shape, data, rg)
    }

    pub fn abs(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.abs()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Abs, vec![a], shape, data, rg)
    }

    pub fn sin(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.sin()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Sin, vec![a], shape, data, rg)
    }

    pub fn cos(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.cos()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Cos, vec![a], shape, data, rg)
    }

    pub fn relu(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.max(0.0)).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Relu, vec![a], shape, data, rg)
    }

    pub fn sigmoid(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Sigmoid, vec![a], shape, data, rg)
    }

    pub fn tanh_op(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| x.tanh()).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Tanh, vec![a], shape, data, rg)
    }

    pub fn gelu(&mut self, a: usize) -> usize {
        let data: Vec<f64> = self.nodes[a].output_data.iter().map(|x| {
            0.5 * x * (1.0 + ((2.0 / PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
        }).collect();
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Gelu, vec![a], shape, data, rg)
    }

    pub fn softmax(&mut self, a: usize, axis: usize) -> usize {
        let shape = self.nodes[a].output_shape.clone();
        let input = &self.nodes[a].output_data;
        let data = softmax_forward(input, &shape, axis);
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Softmax { axis }, vec![a], shape, data, rg)
    }

    pub fn sum(&mut self, a: usize, axis: Option<usize>) -> usize {
        let input = &self.nodes[a].output_data;
        let in_shape = self.nodes[a].output_shape.clone();
        let (data, out_shape) = reduce_sum(input, &in_shape, axis);
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Sum { axis }, vec![a], out_shape, data, rg)
    }

    pub fn mean(&mut self, a: usize, axis: Option<usize>) -> usize {
        let input = &self.nodes[a].output_data;
        let in_shape = self.nodes[a].output_shape.clone();
        let (data, out_shape) = reduce_mean(input, &in_shape, axis);
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Mean { axis }, vec![a], out_shape, data, rg)
    }

    pub fn reshape(&mut self, a: usize, new_shape: Vec<usize>) -> usize {
        let data = self.nodes[a].output_data.clone();
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Reshape { new_shape: new_shape.clone() }, vec![a], new_shape, data, rg)
    }

    pub fn transpose(&mut self, a: usize) -> usize {
        let in_shape = &self.nodes[a].output_shape;
        assert!(in_shape.len() == 2, "transpose requires 2D tensor");
        let (rows, cols) = (in_shape[0], in_shape[1]);
        let input = &self.nodes[a].output_data;
        let mut data = vec![0.0; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                data[c * rows + r] = input[r * cols + c];
            }
        }
        let rg = self.nodes[a].requires_grad;
        self.push_node(Op::Transpose, vec![a], vec![cols, rows], data, rg)
    }

    // ----- Binary ops -----

    pub fn add(&mut self, a: usize, b: usize) -> usize {
        let (data, shape) = broadcast_binary(&self.nodes[a], &self.nodes[b], |x, y| x + y);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::Add, vec![a, b], shape, data, rg)
    }

    pub fn sub(&mut self, a: usize, b: usize) -> usize {
        let (data, shape) = broadcast_binary(&self.nodes[a], &self.nodes[b], |x, y| x - y);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::Sub, vec![a, b], shape, data, rg)
    }

    pub fn mul(&mut self, a: usize, b: usize) -> usize {
        let (data, shape) = broadcast_binary(&self.nodes[a], &self.nodes[b], |x, y| x * y);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::Mul, vec![a, b], shape, data, rg)
    }

    pub fn div(&mut self, a: usize, b: usize) -> usize {
        let (data, shape) = broadcast_binary(&self.nodes[a], &self.nodes[b], |x, y| x / y);
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::Div, vec![a, b], shape, data, rg)
    }

    pub fn pow(&mut self, a: usize, b: usize) -> usize {
        let (data, shape) = broadcast_binary(&self.nodes[a], &self.nodes[b], |x, y| x.powf(y));
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::Pow, vec![a, b], shape, data, rg)
    }

    pub fn matmul(&mut self, a: usize, b: usize) -> usize {
        let sa = &self.nodes[a].output_shape;
        let sb = &self.nodes[b].output_shape;
        assert!(sa.len() == 2 && sb.len() == 2, "matmul requires 2D tensors");
        let (m, k) = (sa[0], sa[1]);
        let (k2, n) = (sb[0], sb[1]);
        assert_eq!(k, k2, "matmul inner dimensions must match");
        let ad = &self.nodes[a].output_data;
        let bd = &self.nodes[b].output_data;
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for p in 0..k {
                    s += ad[i * k + p] * bd[p * n + j];
                }
                data[i * n + j] = s;
            }
        }
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::MatMul, vec![a, b], vec![m, n], data, rg)
    }

    pub fn broadcast_add(&mut self, a: usize, b: usize) -> usize {
        // a is [m, n], b is [n] — add bias
        let sa = &self.nodes[a].output_shape;
        let sb = &self.nodes[b].output_shape;
        assert!(sa.len() == 2 && sb.len() == 1 && sa[1] == sb[0],
                "broadcast_add: a must be [m,n], b must be [n]");
        let (m, n) = (sa[0], sa[1]);
        let ad = &self.nodes[a].output_data;
        let bd = &self.nodes[b].output_data;
        let mut data = vec![0.0; m * n];
        for i in 0..m {
            for j in 0..n {
                data[i * n + j] = ad[i * n + j] + bd[j];
            }
        }
        let rg = self.nodes[a].requires_grad || self.nodes[b].requires_grad;
        self.push_node(Op::BroadcastAdd, vec![a, b], vec![m, n], data, rg)
    }

    // ----- Neural net ops -----

    pub fn layer_norm(&mut self, a: usize, gamma: usize, beta: usize, eps: f64) -> usize {
        let shape = self.nodes[a].output_shape.clone();
        assert!(shape.len() == 2);
        let (batch, dim) = (shape[0], shape[1]);
        let input = &self.nodes[a].output_data;
        let g = &self.nodes[gamma].output_data;
        let b = &self.nodes[beta].output_data;
        let mut data = vec![0.0; batch * dim];
        for i in 0..batch {
            let row = &input[i * dim..(i + 1) * dim];
            let mean: f64 = row.iter().sum::<f64>() / dim as f64;
            let var: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
            let std = (var + eps).sqrt();
            for j in 0..dim {
                data[i * dim + j] = g[j] * (row[j] - mean) / std + b[j];
            }
        }
        let rg = self.nodes[a].requires_grad || self.nodes[gamma].requires_grad || self.nodes[beta].requires_grad;
        self.push_node(Op::LayerNorm { eps, normalized_shape: dim }, vec![a, gamma, beta], shape, data, rg)
    }

    pub fn dropout(&mut self, a: usize, rate: f64, training: bool) -> usize {
        let data_in = &self.nodes[a].output_data;
        let shape = self.nodes[a].output_shape.clone();
        let rg = self.nodes[a].requires_grad;
        if !training || rate == 0.0 {
            let mask = vec![true; data_in.len()];
            let data = data_in.clone();
            return self.push_node(Op::Dropout { rate, training, mask }, vec![a], shape, data, rg);
        }
        // Deterministic "random" mask based on index for reproducibility in tests
        let scale = 1.0 / (1.0 - rate);
        let mut mask = Vec::with_capacity(data_in.len());
        let mut data = Vec::with_capacity(data_in.len());
        for (i, &x) in data_in.iter().enumerate() {
            let keep = ((i * 2654435761) % 1000) as f64 / 1000.0 >= rate;
            mask.push(keep);
            data.push(if keep { x * scale } else { 0.0 });
        }
        self.push_node(Op::Dropout { rate, training, mask }, vec![a], shape, data, rg)
    }

    pub fn cross_entropy(&mut self, logits: usize, targets: usize) -> usize {
        // logits: [batch, classes], targets: [batch] with class indices as floats
        let ls = &self.nodes[logits].output_shape;
        assert!(ls.len() == 2);
        let (batch, classes) = (ls[0], ls[1]);
        let ld = &self.nodes[logits].output_data;
        let td = &self.nodes[targets].output_data;

        let mut loss = 0.0;
        for i in 0..batch {
            let row = &ld[i * classes..(i + 1) * classes];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exp_sum: f64 = row.iter().map(|x| (x - max_val).exp()).sum();
            let log_sum = max_val + exp_sum.ln();
            let target_class = td[i] as usize;
            loss += log_sum - row[target_class];
        }
        loss /= batch as f64;

        let rg = self.nodes[logits].requires_grad;
        self.push_node(Op::CrossEntropy, vec![logits, targets], vec![1], vec![loss], rg)
    }

    pub fn mse(&mut self, pred: usize, target: usize) -> usize {
        let pd = &self.nodes[pred].output_data;
        let td = &self.nodes[target].output_data;
        let n = pd.len();
        let loss: f64 = pd.iter().zip(td.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>() / n as f64;
        let rg = self.nodes[pred].requires_grad || self.nodes[target].requires_grad;
        self.push_node(Op::Mse, vec![pred, target], vec![1], vec![loss], rg)
    }

    pub fn conv2d(&mut self, input: usize, kernel: usize, stride: usize, padding: usize) -> usize {
        // input: [batch, c_in, h, w], kernel: [c_out, c_in, kh, kw]
        let is = &self.nodes[input].output_shape;
        let ks = &self.nodes[kernel].output_shape;
        assert!(is.len() == 4 && ks.len() == 4);
        let (batch, c_in, h, w) = (is[0], is[1], is[2], is[3]);
        let (c_out, kc, kh, kw) = (ks[0], ks[1], ks[2], ks[3]);
        assert_eq!(c_in, kc);
        let oh = (h + 2 * padding - kh) / stride + 1;
        let ow = (w + 2 * padding - kw) / stride + 1;
        let id = &self.nodes[input].output_data;
        let kd = &self.nodes[kernel].output_data;

        let mut data = vec![0.0; batch * c_out * oh * ow];
        for b in 0..batch {
            for co in 0..c_out {
                for i in 0..oh {
                    for j in 0..ow {
                        let mut val = 0.0;
                        for ci in 0..c_in {
                            for ki in 0..kh {
                                for kj in 0..kw {
                                    let hi = i * stride + ki;
                                    let wj = j * stride + kj;
                                    let hi = hi as isize - padding as isize;
                                    let wj = wj as isize - padding as isize;
                                    if hi >= 0 && hi < h as isize && wj >= 0 && wj < w as isize {
                                        let in_idx = b * c_in * h * w + ci * h * w + hi as usize * w + wj as usize;
                                        let k_idx = co * kc * kh * kw + ci * kh * kw + ki * kw + kj;
                                        val += id[in_idx] * kd[k_idx];
                                    }
                                }
                            }
                        }
                        data[b * c_out * oh * ow + co * oh * ow + i * ow + j] = val;
                    }
                }
            }
        }
        let rg = self.nodes[input].requires_grad || self.nodes[kernel].requires_grad;
        self.push_node(Op::Conv2d { stride, padding, kh, kw }, vec![input, kernel], vec![batch, c_out, oh, ow], data, rg)
    }

    // ----- Backward pass -----

    pub fn zero_grad(&mut self) {
        for node in &mut self.nodes {
            node.grad = None;
        }
    }

    pub fn backward(&mut self, output_node: usize) {
        let n = self.nodes.len();
        assert!(output_node < n);

        // Initialize output gradient to ones
        let out_len = self.nodes[output_node].output_data.len();
        self.nodes[output_node].grad = Some(vec![1.0; out_len]);

        // Reverse-mode: walk tape backwards
        for i in (0..=output_node).rev() {
            let grad_out = match &self.nodes[i].grad {
                Some(g) => g.clone(),
                None => continue,
            };

            if !self.nodes[i].requires_grad && self.nodes[i].inputs.is_empty() {
                continue;
            }

            let op = self.nodes[i].op.clone();
            let inputs = self.nodes[i].inputs.clone();

            match op {
                Op::Parameter | Op::Input | Op::Constant => {}

                Op::Neg => {
                    let a = inputs[0];
                    let g: Vec<f64> = grad_out.iter().map(|x| -x).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Exp => {
                    let a = inputs[0];
                    let out = self.nodes[i].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(out.iter()).map(|(go, o)| go * o).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Log => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| go / x).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Sqrt => {
                    let a = inputs[0];
                    let out = self.nodes[i].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(out.iter()).map(|(go, o)| go / (2.0 * o)).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Abs => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| {
                        if *x > 0.0 { *go } else if *x < 0.0 { -go } else { 0.0 }
                    }).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Sin => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| go * x.cos()).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Cos => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| -go * x.sin()).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Relu => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| {
                        if *x > 0.0 { *go } else { 0.0 }
                    }).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Sigmoid => {
                    let a = inputs[0];
                    let out = self.nodes[i].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(out.iter()).map(|(go, s)| go * s * (1.0 - s)).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Tanh => {
                    let a = inputs[0];
                    let out = self.nodes[i].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(out.iter()).map(|(go, t)| go * (1.0 - t * t)).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Gelu => {
                    let a = inputs[0];
                    let inp = self.nodes[a].output_data.clone();
                    let g: Vec<f64> = grad_out.iter().zip(inp.iter()).map(|(go, x)| {
                        let k = (2.0 / PI).sqrt();
                        let inner = k * (x + 0.044715 * x.powi(3));
                        let tanh_val = inner.tanh();
                        let sech2 = 1.0 - tanh_val * tanh_val;
                        let d_inner = k * (1.0 + 3.0 * 0.044715 * x * x);
                        go * (0.5 * (1.0 + tanh_val) + 0.5 * x * sech2 * d_inner)
                    }).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Softmax { axis } => {
                    let a = inputs[0];
                    let out = self.nodes[i].output_data.clone();
                    let shape = self.nodes[i].output_shape.clone();
                    let g = softmax_backward(&grad_out, &out, &shape, axis);
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Sum { axis } => {
                    let a = inputs[0];
                    let in_shape = self.nodes[a].output_shape.clone();
                    let g = sum_backward(&grad_out, &in_shape, axis);
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Mean { axis } => {
                    let a = inputs[0];
                    let in_shape = self.nodes[a].output_shape.clone();
                    let count = match axis {
                        Some(ax) => in_shape[ax] as f64,
                        None => in_shape.iter().product::<usize>() as f64,
                    };
                    let sum_grad = sum_backward(&grad_out, &in_shape, axis);
                    let g: Vec<f64> = sum_grad.iter().map(|x| x / count).collect();
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Reshape { new_shape: _ } => {
                    let a = inputs[0];
                    let in_shape = self.nodes[a].output_shape.clone();
                    // Just reshape the gradient back
                    let g = grad_out.clone(); // data is same, just shape differs
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Transpose => {
                    let a = inputs[0];
                    let out_shape = self.nodes[i].output_shape.clone();
                    let (rows, cols) = (out_shape[0], out_shape[1]);
                    // Transpose gradient back
                    let mut g = vec![0.0; rows * cols];
                    for r in 0..rows {
                        for c in 0..cols {
                            g[c * rows + r] = grad_out[r * cols + c];
                        }
                    }
                    accumulate_grad(&mut self.nodes[a], &g);
                }

                Op::Add => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let ga = reduce_broadcast_grad(&grad_out, &self.nodes[i].output_shape, &self.nodes[a].output_shape);
                    let gb = reduce_broadcast_grad(&grad_out, &self.nodes[i].output_shape, &self.nodes[b].output_shape);
                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::Sub => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let ga = reduce_broadcast_grad(&grad_out, &self.nodes[i].output_shape, &self.nodes[a].output_shape);
                    let neg_go: Vec<f64> = grad_out.iter().map(|x| -x).collect();
                    let gb = reduce_broadcast_grad(&neg_go, &self.nodes[i].output_shape, &self.nodes[b].output_shape);
                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::Mul => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let ad = self.nodes[a].output_data.clone();
                    let bd = self.nodes[b].output_data.clone();
                    // Need to handle broadcasting
                    let out_shape = self.nodes[i].output_shape.clone();
                    let a_shape = self.nodes[a].output_shape.clone();
                    let b_shape = self.nodes[b].output_shape.clone();

                    let ga_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let b_idx = broadcast_index(idx, &out_shape, &b_shape);
                        go * bd[b_idx]
                    }).collect();
                    let gb_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let a_idx = broadcast_index(idx, &out_shape, &a_shape);
                        go * ad[a_idx]
                    }).collect();
                    let ga = reduce_broadcast_grad(&ga_full, &out_shape, &a_shape);
                    let gb = reduce_broadcast_grad(&gb_full, &out_shape, &b_shape);
                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::Div => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let ad = self.nodes[a].output_data.clone();
                    let bd = self.nodes[b].output_data.clone();
                    let out_shape = self.nodes[i].output_shape.clone();
                    let a_shape = self.nodes[a].output_shape.clone();
                    let b_shape = self.nodes[b].output_shape.clone();

                    let ga_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let b_idx = broadcast_index(idx, &out_shape, &b_shape);
                        go / bd[b_idx]
                    }).collect();
                    let gb_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let a_idx = broadcast_index(idx, &out_shape, &a_shape);
                        let b_idx = broadcast_index(idx, &out_shape, &b_shape);
                        -go * ad[a_idx] / (bd[b_idx] * bd[b_idx])
                    }).collect();
                    let ga = reduce_broadcast_grad(&ga_full, &out_shape, &a_shape);
                    let gb = reduce_broadcast_grad(&gb_full, &out_shape, &b_shape);
                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::Pow => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let ad = self.nodes[a].output_data.clone();
                    let bd = self.nodes[b].output_data.clone();
                    let out = self.nodes[i].output_data.clone();
                    let out_shape = self.nodes[i].output_shape.clone();
                    let a_shape = self.nodes[a].output_shape.clone();
                    let b_shape = self.nodes[b].output_shape.clone();

                    // d/da (a^b) = b * a^(b-1)
                    let ga_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let a_idx = broadcast_index(idx, &out_shape, &a_shape);
                        let b_idx = broadcast_index(idx, &out_shape, &b_shape);
                        go * bd[b_idx] * ad[a_idx].powf(bd[b_idx] - 1.0)
                    }).collect();
                    // d/db (a^b) = a^b * ln(a)
                    let gb_full: Vec<f64> = grad_out.iter().enumerate().map(|(idx, go)| {
                        let a_idx = broadcast_index(idx, &out_shape, &a_shape);
                        go * out[idx] * ad[a_idx].ln()
                    }).collect();
                    let ga = reduce_broadcast_grad(&ga_full, &out_shape, &a_shape);
                    let gb = reduce_broadcast_grad(&gb_full, &out_shape, &b_shape);
                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::MatMul => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let sa = self.nodes[a].output_shape.clone();
                    let sb = self.nodes[b].output_shape.clone();
                    let (m, k) = (sa[0], sa[1]);
                    let n = sb[1];
                    let ad = self.nodes[a].output_data.clone();
                    let bd = self.nodes[b].output_data.clone();

                    // grad_a = grad_out @ b.T
                    let mut ga = vec![0.0; m * k];
                    for ii in 0..m {
                        for jj in 0..k {
                            let mut s = 0.0;
                            for pp in 0..n {
                                s += grad_out[ii * n + pp] * bd[jj * n + pp];
                            }
                            ga[ii * k + jj] = s;
                        }
                    }

                    // grad_b = a.T @ grad_out
                    let mut gb = vec![0.0; k * n];
                    for ii in 0..k {
                        for jj in 0..n {
                            let mut s = 0.0;
                            for pp in 0..m {
                                s += ad[pp * k + ii] * grad_out[pp * n + jj];
                            }
                            gb[ii * n + jj] = s;
                        }
                    }

                    accumulate_grad(&mut self.nodes[a], &ga);
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::BroadcastAdd => {
                    let (a, b) = (inputs[0], inputs[1]);
                    let sa = self.nodes[a].output_shape.clone();
                    let (m, n) = (sa[0], sa[1]);

                    // grad_a = grad_out (same shape)
                    accumulate_grad(&mut self.nodes[a], &grad_out);

                    // grad_b = sum over rows
                    let mut gb = vec![0.0; n];
                    for ii in 0..m {
                        for jj in 0..n {
                            gb[jj] += grad_out[ii * n + jj];
                        }
                    }
                    accumulate_grad(&mut self.nodes[b], &gb);
                }

                Op::LayerNorm { eps, normalized_shape } => {
                    let (a, gamma, beta) = (inputs[0], inputs[1], inputs[2]);
                    let shape = self.nodes[i].output_shape.clone();
                    let (batch, dim) = (shape[0], shape[1]);
                    let inp = self.nodes[a].output_data.clone();
                    let g = self.nodes[gamma].output_data.clone();

                    let mut grad_input = vec![0.0; batch * dim];
                    let mut grad_gamma = vec![0.0; dim];
                    let mut grad_beta = vec![0.0; dim];

                    for bi in 0..batch {
                        let row = &inp[bi * dim..(bi + 1) * dim];
                        let mean: f64 = row.iter().sum::<f64>() / dim as f64;
                        let var: f64 = row.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / dim as f64;
                        let std = (var + eps).sqrt();
                        let n_f = dim as f64;

                        let go_row = &grad_out[bi * dim..(bi + 1) * dim];

                        // Accumulate gamma and beta grads
                        for j in 0..dim {
                            let x_hat = (row[j] - mean) / std;
                            grad_gamma[j] += go_row[j] * x_hat;
                            grad_beta[j] += go_row[j];
                        }

                        // Input gradient (3-term formula)
                        let mut dx_hat = vec![0.0; dim];
                        for j in 0..dim {
                            dx_hat[j] = go_row[j] * g[j];
                        }
                        let dx_hat_sum: f64 = dx_hat.iter().sum();
                        let dx_hat_x_sum: f64 = dx_hat.iter().enumerate()
                            .map(|(j, &dxh)| dxh * (row[j] - mean))
                            .sum();

                        for j in 0..dim {
                            grad_input[bi * dim + j] = (1.0 / std) * (
                                dx_hat[j] - dx_hat_sum / n_f - (row[j] - mean) * dx_hat_x_sum / (n_f * (var + eps))
                            );
                        }
                    }

                    accumulate_grad(&mut self.nodes[a], &grad_input);
                    accumulate_grad(&mut self.nodes[gamma], &grad_gamma);
                    accumulate_grad(&mut self.nodes[beta], &grad_beta);
                }

                Op::Dropout { rate, training, ref mask } => {
                    let a = inputs[0];
                    if !training || rate == 0.0 {
                        accumulate_grad(&mut self.nodes[a], &grad_out);
                    } else {
                        let scale = 1.0 / (1.0 - rate);
                        let g: Vec<f64> = grad_out.iter().zip(mask.iter()).map(|(go, &keep)| {
                            if keep { go * scale } else { 0.0 }
                        }).collect();
                        accumulate_grad(&mut self.nodes[a], &g);
                    }
                }

                Op::CrossEntropy => {
                    let logits = inputs[0];
                    let targets_id = inputs[1];
                    let ls = self.nodes[logits].output_shape.clone();
                    let (batch, classes) = (ls[0], ls[1]);
                    let ld = self.nodes[logits].output_data.clone();
                    let td = self.nodes[targets_id].output_data.clone();

                    let mut g = vec![0.0; batch * classes];
                    for bi in 0..batch {
                        let row = &ld[bi * classes..(bi + 1) * classes];
                        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                        let exps: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
                        let sum: f64 = exps.iter().sum();
                        let target_class = td[bi] as usize;
                        for j in 0..classes {
                            let softmax_j = exps[j] / sum;
                            g[bi * classes + j] = grad_out[0] * (softmax_j - if j == target_class { 1.0 } else { 0.0 }) / batch as f64;
                        }
                    }
                    accumulate_grad(&mut self.nodes[logits], &g);
                }

                Op::Mse => {
                    let (pred, target) = (inputs[0], inputs[1]);
                    let pd = self.nodes[pred].output_data.clone();
                    let td = self.nodes[target].output_data.clone();
                    let n = pd.len() as f64;
                    let g: Vec<f64> = pd.iter().zip(td.iter()).map(|(p, t)| {
                        grad_out[0] * 2.0 * (p - t) / n
                    }).collect();
                    accumulate_grad(&mut self.nodes[pred], &g);
                    // Also propagate to target if it requires grad
                    let g_t: Vec<f64> = pd.iter().zip(td.iter()).map(|(p, t)| {
                        grad_out[0] * 2.0 * (t - p) / n
                    }).collect();
                    accumulate_grad(&mut self.nodes[target], &g_t);
                }

                Op::Conv2d { stride, padding, kh, kw } => {
                    let (input, kernel) = (inputs[0], inputs[1]);
                    let is = self.nodes[input].output_shape.clone();
                    let ks = self.nodes[kernel].output_shape.clone();
                    let (batch, c_in, h, w) = (is[0], is[1], is[2], is[3]);
                    let (c_out, _, _, _) = (ks[0], ks[1], ks[2], ks[3]);
                    let oh = (h + 2 * padding - kh) / stride + 1;
                    let ow = (w + 2 * padding - kw) / stride + 1;
                    let id = self.nodes[input].output_data.clone();
                    let kd = self.nodes[kernel].output_data.clone();

                    // Gradient w.r.t. input
                    let mut g_input = vec![0.0; batch * c_in * h * w];
                    // Gradient w.r.t. kernel
                    let mut g_kernel = vec![0.0; c_out * c_in * kh * kw];

                    for b in 0..batch {
                        for co in 0..c_out {
                            for oi in 0..oh {
                                for oj in 0..ow {
                                    let go = grad_out[b * c_out * oh * ow + co * oh * ow + oi * ow + oj];
                                    for ci in 0..c_in {
                                        for ki in 0..kh {
                                            for kj in 0..kw {
                                                let hi = (oi * stride + ki) as isize - padding as isize;
                                                let wj = (oj * stride + kj) as isize - padding as isize;
                                                if hi >= 0 && hi < h as isize && wj >= 0 && wj < w as isize {
                                                    let in_idx = b * c_in * h * w + ci * h * w + hi as usize * w + wj as usize;
                                                    let k_idx = co * c_in * kh * kw + ci * kh * kw + ki * kw + kj;
                                                    g_input[in_idx] += go * kd[k_idx];
                                                    g_kernel[k_idx] += go * id[in_idx];
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }

                    accumulate_grad(&mut self.nodes[input], &g_input);
                    accumulate_grad(&mut self.nodes[kernel], &g_kernel);
                }

                Op::Embedding { .. } => {
                    // Embedding grad: scatter gradients to the looked-up rows
                    // (not commonly differentiable in this simple engine)
                }
            }
        }
    }

    pub fn get_grad(&self, node_id: usize) -> Option<Vec<f64>> {
        self.nodes[node_id].grad.clone()
    }

    pub fn get_data(&self, node_id: usize) -> &[f64] {
        &self.nodes[node_id].output_data
    }

    pub fn get_shape(&self, node_id: usize) -> &[usize] {
        &self.nodes[node_id].output_shape
    }

    /// Update parameter data in-place (for optimizer steps)
    pub fn update_data(&mut self, node_id: usize, new_data: Vec<f64>) {
        self.nodes[node_id].output_data = new_data;
    }
}

// ---------------------------------------------------------------------------
// TrackedTensor — high-level API with shared tape
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct TrackedTensor {
    pub tape: Rc<RefCell<Tape>>,
    pub node_id: usize,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
}

impl TrackedTensor {
    pub fn parameter(tape: &Rc<RefCell<Tape>>, data: Vec<f64>, shape: Vec<usize>) -> Self {
        let node_id = tape.borrow_mut().parameter(data, shape.clone());
        TrackedTensor { tape: Rc::clone(tape), node_id, shape, requires_grad: true }
    }

    pub fn input(tape: &Rc<RefCell<Tape>>, data: Vec<f64>, shape: Vec<usize>) -> Self {
        let node_id = tape.borrow_mut().input(data, shape.clone());
        TrackedTensor { tape: Rc::clone(tape), node_id, shape, requires_grad: false }
    }

    pub fn constant(tape: &Rc<RefCell<Tape>>, data: Vec<f64>, shape: Vec<usize>) -> Self {
        let node_id = tape.borrow_mut().constant(data, shape.clone());
        TrackedTensor { tape: Rc::clone(tape), node_id, shape, requires_grad: false }
    }

    pub fn data(&self) -> Vec<f64> {
        self.tape.borrow().get_data(self.node_id).to_vec()
    }

    pub fn grad(&self) -> Option<Vec<f64>> {
        self.tape.borrow().get_grad(self.node_id)
    }

    fn unary(&self, f: impl FnOnce(&mut Tape, usize) -> usize) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = f(&mut t, self.node_id);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    fn binary(&self, other: &TrackedTensor, f: impl FnOnce(&mut Tape, usize, usize) -> usize) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = f(&mut t, self.node_id, other.node_id);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    pub fn neg(&self) -> TrackedTensor { self.unary(|t, a| t.neg(a)) }
    pub fn exp(&self) -> TrackedTensor { self.unary(|t, a| t.exp(a)) }
    pub fn log(&self) -> TrackedTensor { self.unary(|t, a| t.log(a)) }
    pub fn sqrt(&self) -> TrackedTensor { self.unary(|t, a| t.sqrt(a)) }
    pub fn abs(&self) -> TrackedTensor { self.unary(|t, a| t.abs(a)) }
    pub fn sin(&self) -> TrackedTensor { self.unary(|t, a| t.sin(a)) }
    pub fn cos(&self) -> TrackedTensor { self.unary(|t, a| t.cos(a)) }
    pub fn relu(&self) -> TrackedTensor { self.unary(|t, a| t.relu(a)) }
    pub fn sigmoid(&self) -> TrackedTensor { self.unary(|t, a| t.sigmoid(a)) }
    pub fn tanh_op(&self) -> TrackedTensor { self.unary(|t, a| t.tanh_op(a)) }
    pub fn gelu(&self) -> TrackedTensor { self.unary(|t, a| t.gelu(a)) }
    pub fn transpose(&self) -> TrackedTensor { self.unary(|t, a| t.transpose(a)) }

    pub fn softmax(&self, axis: usize) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = t.softmax(self.node_id, axis);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    pub fn sum(&self, axis: Option<usize>) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = t.sum(self.node_id, axis);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    pub fn mean(&self, axis: Option<usize>) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = t.mean(self.node_id, axis);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    pub fn reshape(&self, new_shape: Vec<usize>) -> TrackedTensor {
        let mut t = self.tape.borrow_mut();
        let id = t.reshape(self.node_id, new_shape);
        let shape = t.nodes[id].output_shape.clone();
        let rg = t.nodes[id].requires_grad;
        TrackedTensor { tape: Rc::clone(&self.tape), node_id: id, shape, requires_grad: rg }
    }

    pub fn add(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.add(a, b)) }
    pub fn sub(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.sub(a, b)) }
    pub fn mul(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.mul(a, b)) }
    pub fn div(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.div(a, b)) }
    pub fn pow(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.pow(a, b)) }
    pub fn matmul(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.matmul(a, b)) }
    pub fn broadcast_add(&self, other: &TrackedTensor) -> TrackedTensor { self.binary(other, |t, a, b| t.broadcast_add(a, b)) }

    pub fn mse(&self, target: &TrackedTensor) -> TrackedTensor { self.binary(target, |t, a, b| t.mse(a, b)) }
    pub fn cross_entropy(&self, targets: &TrackedTensor) -> TrackedTensor { self.binary(targets, |t, a, b| t.cross_entropy(a, b)) }
}

// ---------------------------------------------------------------------------
// grad() — functional API
// ---------------------------------------------------------------------------

/// Compute gradients of a scalar-valued function w.r.t. parameters.
/// Returns gradient vectors for each parameter.
pub fn grad(
    f: impl Fn(&[TrackedTensor]) -> TrackedTensor,
    params: &[TrackedTensor],
) -> Vec<Vec<f64>> {
    let output = f(params);
    output.tape.borrow_mut().backward(output.node_id);
    params.iter().map(|p| {
        p.tape.borrow().get_grad(p.node_id).unwrap_or_else(|| vec![0.0; p.shape.iter().product()])
    }).collect()
}

/// Compute second derivatives via nested differentiation.
/// For scalar functions f: R^n -> R, computes the diagonal of the Hessian.
pub fn hessian_diagonal(
    f: impl Fn(&[TrackedTensor]) -> TrackedTensor,
    params: &[Vec<f64>],
    shapes: &[Vec<usize>],
    eps: f64,
) -> Vec<Vec<f64>> {
    // Use finite differences on the gradient for higher-order derivatives
    let tape = Rc::new(RefCell::new(Tape::new()));
    let base_params: Vec<TrackedTensor> = params.iter().zip(shapes.iter())
        .map(|(d, s)| TrackedTensor::parameter(&tape, d.clone(), s.clone()))
        .collect();
    let base_grads = grad(&f, &base_params);

    let mut hessian_diag = Vec::new();
    for (pi, (param_data, shape)) in params.iter().zip(shapes.iter()).enumerate() {
        let mut diag = vec![0.0; param_data.len()];
        for j in 0..param_data.len() {
            // f(x + eps*e_j)
            let tape_p = Rc::new(RefCell::new(Tape::new()));
            let mut perturbed = params.to_vec();
            perturbed[pi][j] += eps;
            let pp: Vec<TrackedTensor> = perturbed.iter().zip(shapes.iter())
                .map(|(d, s)| TrackedTensor::parameter(&tape_p, d.clone(), s.clone()))
                .collect();
            let grads_p = grad(&f, &pp);
            diag[j] = (grads_p[pi][j] - base_grads[pi][j]) / eps;
        }
        hessian_diag.push(diag);
    }
    hessian_diag
}

// ---------------------------------------------------------------------------
// Adam optimizer
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct AdamOptimizer {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub m: Vec<Vec<f64>>,
    pub v: Vec<Vec<f64>>,
    pub t: usize,
}

impl AdamOptimizer {
    pub fn new(lr: f64, param_sizes: &[usize]) -> Self {
        AdamOptimizer {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            m: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            v: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            t: 0,
        }
    }

    pub fn new_with_betas(lr: f64, beta1: f64, beta2: f64, eps: f64, param_sizes: &[usize]) -> Self {
        AdamOptimizer {
            lr, beta1, beta2, eps,
            m: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            v: param_sizes.iter().map(|&s| vec![0.0; s]).collect(),
            t: 0,
        }
    }

    pub fn step(&mut self, tape: &mut Tape, param_ids: &[usize], grads: &[Vec<f64>]) {
        self.t += 1;
        let t = self.t as f64;
        for (i, (&pid, g)) in param_ids.iter().zip(grads.iter()).enumerate() {
            for j in 0..g.len() {
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * g[j];
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * g[j] * g[j];
                let m_hat = self.m[i][j] / (1.0 - self.beta1.powf(t));
                let v_hat = self.v[i][j] / (1.0 - self.beta2.powf(t));
                tape.nodes[pid].output_data[j] -= self.lr * m_hat / (v_hat.sqrt() + self.eps);
            }
        }
    }
}

/// Simple SGD optimizer
pub struct SgdOptimizer {
    pub lr: f64,
}

impl SgdOptimizer {
    pub fn new(lr: f64) -> Self { SgdOptimizer { lr } }

    pub fn step(&self, tape: &mut Tape, param_ids: &[usize], grads: &[Vec<f64>]) {
        for (&pid, g) in param_ids.iter().zip(grads.iter()) {
            for j in 0..g.len() {
                tape.nodes[pid].output_data[j] -= self.lr * g[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn accumulate_grad(node: &mut TapeNode, grad: &[f64]) {
    match &mut node.grad {
        Some(existing) => {
            for (e, g) in existing.iter_mut().zip(grad.iter()) {
                *e += g;
            }
        }
        None => {
            node.grad = Some(grad.to_vec());
        }
    }
}

fn softmax_forward(input: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    assert!(shape.len() == 2);
    let (rows, cols) = (shape[0], shape[1]);
    let mut output = vec![0.0; input.len()];

    if axis == 1 || (axis == 0 && rows == 1) {
        // Softmax along last axis (most common)
        let dim = if axis == 1 { cols } else { rows };
        let outer = if axis == 1 { rows } else { 1 };
        let inner = if axis == 1 { cols } else { rows };

        for i in 0..outer {
            let start = i * inner;
            let row = &input[start..start + inner];
            let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let exps: Vec<f64> = row.iter().map(|x| (x - max_val).exp()).collect();
            let sum: f64 = exps.iter().sum();
            for j in 0..inner {
                output[start + j] = exps[j] / sum;
            }
        }
    } else {
        // axis == 0: softmax along rows
        for j in 0..cols {
            let mut max_val = f64::NEG_INFINITY;
            for i in 0..rows {
                max_val = max_val.max(input[i * cols + j]);
            }
            let mut sum = 0.0;
            for i in 0..rows {
                let e = (input[i * cols + j] - max_val).exp();
                output[i * cols + j] = e;
                sum += e;
            }
            for i in 0..rows {
                output[i * cols + j] /= sum;
            }
        }
    }
    output
}

fn softmax_backward(grad_out: &[f64], output: &[f64], shape: &[usize], axis: usize) -> Vec<f64> {
    assert!(shape.len() == 2);
    let (rows, cols) = (shape[0], shape[1]);
    let mut grad_input = vec![0.0; grad_out.len()];

    if axis == 1 {
        for i in 0..rows {
            let start = i * cols;
            let s = &output[start..start + cols];
            let go = &grad_out[start..start + cols];
            // Jacobian-vector product: diag(s) - s*s^T
            let dot: f64 = s.iter().zip(go.iter()).map(|(si, gi)| si * gi).sum();
            for j in 0..cols {
                grad_input[start + j] = s[j] * (go[j] - dot);
            }
        }
    } else {
        // axis == 0
        for j in 0..cols {
            let mut dot = 0.0;
            for i in 0..rows {
                dot += output[i * cols + j] * grad_out[i * cols + j];
            }
            for i in 0..rows {
                grad_input[i * cols + j] = output[i * cols + j] * (grad_out[i * cols + j] - dot);
            }
        }
    }
    grad_input
}

fn reduce_sum(input: &[f64], shape: &[usize], axis: Option<usize>) -> (Vec<f64>, Vec<usize>) {
    match axis {
        None => {
            let s: f64 = input.iter().sum();
            (vec![s], vec![1])
        }
        Some(ax) => {
            assert!(ax < shape.len());
            let mut out_shape = shape.to_vec();
            out_shape[ax] = 1;
            let out_size: usize = out_shape.iter().product();
            let mut out = vec![0.0; out_size];

            // General n-dim sum along axis
            let total: usize = shape.iter().product();
            let stride: usize = shape[ax + 1..].iter().product();
            let ax_size = shape[ax];
            let outer: usize = shape[..ax].iter().product();

            for o in 0..outer {
                for s in 0..stride {
                    let mut sum = 0.0;
                    for a in 0..ax_size {
                        sum += input[o * ax_size * stride + a * stride + s];
                    }
                    out[o * stride + s] = sum;
                }
            }
            (out, out_shape)
        }
    }
}

fn reduce_mean(input: &[f64], shape: &[usize], axis: Option<usize>) -> (Vec<f64>, Vec<usize>) {
    let (sum, out_shape) = reduce_sum(input, shape, axis);
    let count = match axis {
        None => shape.iter().product::<usize>() as f64,
        Some(ax) => shape[ax] as f64,
    };
    let mean: Vec<f64> = sum.iter().map(|x| x / count).collect();
    (mean, out_shape)
}

fn sum_backward(grad_out: &[f64], in_shape: &[usize], axis: Option<usize>) -> Vec<f64> {
    let in_size: usize = in_shape.iter().product();
    match axis {
        None => {
            // Scalar output -> broadcast grad to all elements
            vec![grad_out[0]; in_size]
        }
        Some(ax) => {
            let stride: usize = in_shape[ax + 1..].iter().product();
            let ax_size = in_shape[ax];
            let outer: usize = in_shape[..ax].iter().product();
            let mut g = vec![0.0; in_size];
            for o in 0..outer {
                for s in 0..stride {
                    let go_val = grad_out[o * stride + s];
                    for a in 0..ax_size {
                        g[o * ax_size * stride + a * stride + s] = go_val;
                    }
                }
            }
            g
        }
    }
}

fn broadcast_binary(a: &TapeNode, b: &TapeNode, f: impl Fn(f64, f64) -> f64) -> (Vec<f64>, Vec<usize>) {
    let sa = &a.output_shape;
    let sb = &b.output_shape;

    if sa == sb {
        let data: Vec<f64> = a.output_data.iter().zip(b.output_data.iter()).map(|(x, y)| f(*x, *y)).collect();
        return (data, sa.clone());
    }

    // Broadcast: pad shorter shape with 1s on the left
    let max_ndim = sa.len().max(sb.len());
    let mut sa_pad = vec![1usize; max_ndim];
    let mut sb_pad = vec![1usize; max_ndim];
    for i in 0..sa.len() { sa_pad[max_ndim - sa.len() + i] = sa[i]; }
    for i in 0..sb.len() { sb_pad[max_ndim - sb.len() + i] = sb[i]; }

    let mut out_shape = vec![0usize; max_ndim];
    for i in 0..max_ndim {
        assert!(sa_pad[i] == sb_pad[i] || sa_pad[i] == 1 || sb_pad[i] == 1,
                "shapes not broadcastable");
        out_shape[i] = sa_pad[i].max(sb_pad[i]);
    }

    let out_size: usize = out_shape.iter().product();
    let mut data = vec![0.0; out_size];

    for idx in 0..out_size {
        let a_idx = broadcast_index(idx, &out_shape, sa);
        let b_idx = broadcast_index(idx, &out_shape, sb);
        data[idx] = f(a.output_data[a_idx], b.output_data[b_idx]);
    }

    (data, out_shape)
}

/// Map a flat index in output_shape to a flat index in target_shape (with broadcasting)
fn broadcast_index(flat_idx: usize, out_shape: &[usize], target_shape: &[usize]) -> usize {
    let nd = out_shape.len();
    let tnd = target_shape.len();
    if nd == tnd && out_shape == target_shape {
        return flat_idx;
    }

    // Decompose flat_idx into multi-index in out_shape
    let mut remaining = flat_idx;
    let mut multi = vec![0usize; nd];
    for i in (0..nd).rev() {
        multi[i] = remaining % out_shape[i];
        remaining /= out_shape[i];
    }

    // Map to target multi-index (clamp dimensions that are 1 in target)
    let offset = if nd > tnd { nd - tnd } else { 0 };
    let mut target_flat = 0;
    let mut stride = 1;
    for i in (0..tnd).rev() {
        let out_i = if i + offset < nd { multi[i + offset] } else { 0 };
        let t_dim = target_shape[i];
        let mapped = if t_dim == 1 { 0 } else { out_i };
        target_flat += mapped * stride;
        stride *= t_dim;
    }
    target_flat
}

/// Reduce a gradient from output_shape back to target_shape by summing over broadcast dims
fn reduce_broadcast_grad(grad: &[f64], out_shape: &[usize], target_shape: &[usize]) -> Vec<f64> {
    if out_shape == target_shape {
        return grad.to_vec();
    }

    let target_size: usize = target_shape.iter().product();
    let mut reduced = vec![0.0; target_size];
    let out_size: usize = out_shape.iter().product();

    for idx in 0..out_size {
        let t_idx = broadcast_index(idx, out_shape, target_shape);
        reduced[t_idx] += grad[idx];
    }
    reduced
}

/// Numerical gradient check via finite differences
pub fn numerical_grad(
    f: impl Fn(&[Vec<f64>]) -> f64,
    params: &[Vec<f64>],
    eps: f64,
) -> Vec<Vec<f64>> {
    let mut grads = Vec::new();
    for i in 0..params.len() {
        let mut g = vec![0.0; params[i].len()];
        for j in 0..params[i].len() {
            let mut p_plus = params.to_vec();
            let mut p_minus = params.to_vec();
            p_plus[i][j] += eps;
            p_minus[i][j] -= eps;
            g[j] = (f(&p_plus) - f(&p_minus)) / (2.0 * eps);
        }
        grads.push(g);
    }
    grads
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn approx_eq(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    fn assert_grad_close(analytical: &[f64], numerical: &[f64], tol: f64, label: &str) {
        assert_eq!(analytical.len(), numerical.len(), "{}: length mismatch", label);
        for (i, (a, n)) in analytical.iter().zip(numerical.iter()).enumerate() {
            assert!(approx_eq(*a, *n, tol),
                "{}: element {} differs: analytical={}, numerical={}", label, i, a, n);
        }
    }

    // ---- Test 1: Add gradient ----
    #[test]
    fn test_add_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let b = tape.parameter(vec![4.0, 5.0, 6.0], vec![3]);
        let c = tape.add(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);
        let ga = tape.get_grad(a).unwrap();
        let gb = tape.get_grad(b).unwrap();
        assert_eq!(ga, vec![1.0, 1.0, 1.0]);
        assert_eq!(gb, vec![1.0, 1.0, 1.0]);
    }

    // ---- Test 2: Mul gradient ----
    #[test]
    fn test_mul_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![2.0, 3.0], vec![2]);
        let b = tape.parameter(vec![4.0, 5.0], vec![2]);
        let c = tape.mul(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);
        assert_eq!(tape.get_grad(a).unwrap(), vec![4.0, 5.0]); // d/da = b
        assert_eq!(tape.get_grad(b).unwrap(), vec![2.0, 3.0]); // d/db = a
    }

    // ---- Test 3: MatMul gradient ----
    #[test]
    fn test_matmul_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = tape.parameter(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = tape.matmul(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);

        // Numerical check
        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![2, 2]);
            let bi = t.parameter(params[1].clone(), vec![2, 2]);
            let ci = t.matmul(ai, bi);
            t.get_data(ci).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-4, "matmul grad_a");
        assert_grad_close(&tape.get_grad(b).unwrap(), &ng[1], 1e-4, "matmul grad_b");
    }

    // ---- Test 4: Relu gradient ----
    #[test]
    fn test_relu_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![-1.0, 0.5, -0.3, 2.0], vec![4]);
        let r = tape.relu(a);
        let s = tape.sum(r, None);
        tape.backward(s);
        assert_eq!(tape.get_grad(a).unwrap(), vec![0.0, 1.0, 0.0, 1.0]);
    }

    // ---- Test 5: Sigmoid gradient ----
    #[test]
    fn test_sigmoid_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![0.0, 1.0, -1.0], vec![3]);
        let sig = tape.sigmoid(a);
        let s = tape.sum(sig, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![3]);
            let si = t.sigmoid(ai);
            t.get_data(si).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![0.0, 1.0, -1.0]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-4, "sigmoid");
    }

    // ---- Test 6: Tanh gradient ----
    #[test]
    fn test_tanh_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![0.0, 0.5, -0.5], vec![3]);
        let t_op = tape.tanh_op(a);
        let s = tape.sum(t_op, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![3]);
            let ti = t.tanh_op(ai);
            t.get_data(ti).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![0.0, 0.5, -0.5]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-4, "tanh");
    }

    // ---- Test 7: Exp gradient ----
    #[test]
    fn test_exp_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![0.0, 1.0, -1.0], vec![3]);
        let e = tape.exp(a);
        let s = tape.sum(e, None);
        tape.backward(s);
        let ga = tape.get_grad(a).unwrap();
        // d/dx exp(x) = exp(x)
        assert!(approx_eq(ga[0], 1.0, 1e-10));
        assert!(approx_eq(ga[1], 1.0_f64.exp(), 1e-10));
        assert!(approx_eq(ga[2], (-1.0_f64).exp(), 1e-10));
    }

    // ---- Test 8: Log gradient ----
    #[test]
    fn test_log_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 0.5], vec![3]);
        let l = tape.log(a);
        let s = tape.sum(l, None);
        tape.backward(s);
        let ga = tape.get_grad(a).unwrap();
        assert!(approx_eq(ga[0], 1.0, 1e-10));
        assert!(approx_eq(ga[1], 0.5, 1e-10));
        assert!(approx_eq(ga[2], 2.0, 1e-10));
    }

    // ---- Test 9: Chain rule through multiple ops ----
    #[test]
    fn test_chain_rule() {
        // f(x) = sum(relu(x * 2 + 1))
        let mut tape = Tape::new();
        let x = tape.parameter(vec![-1.0, 0.5, 2.0], vec![3]);
        let two = tape.constant(vec![2.0, 2.0, 2.0], vec![3]);
        let one = tape.constant(vec![1.0, 1.0, 1.0], vec![3]);
        let x2 = tape.mul(x, two);
        let x2p1 = tape.add(x2, one);
        let r = tape.relu(x2p1);
        let s = tape.sum(r, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let xi = t.parameter(params[0].clone(), vec![3]);
            let twoi = t.constant(vec![2.0, 2.0, 2.0], vec![3]);
            let onei = t.constant(vec![1.0, 1.0, 1.0], vec![3]);
            let x2i = t.mul(xi, twoi);
            let x2p1i = t.add(x2i, onei);
            let ri = t.relu(x2p1i);
            t.get_data(ri).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![-1.0, 0.5, 2.0]], 1e-5);
        assert_grad_close(&tape.get_grad(x).unwrap(), &ng[0], 1e-4, "chain");
    }

    // ---- Test 10: Diamond pattern (tensor used twice) ----
    #[test]
    fn test_diamond_pattern() {
        // f(x) = sum(x * x)  => grad = 2*x
        let mut tape = Tape::new();
        let x = tape.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let x2 = tape.mul(x, x);
        let s = tape.sum(x2, None);
        tape.backward(s);
        assert_eq!(tape.get_grad(x).unwrap(), vec![2.0, 4.0, 6.0]);
    }

    // ---- Test 11: Softmax gradient ----
    #[test]
    fn test_softmax_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm = tape.softmax(a, 1);
        let s = tape.sum(sm, None);
        tape.backward(s);

        // Sum of softmax = 1 always, so grad should be ~0
        let ga = tape.get_grad(a).unwrap();
        for g in &ga {
            assert!(g.abs() < 1e-10, "softmax sum grad should be ~0, got {}", g);
        }

        // Now test with a target: pick out one element
        let mut tape2 = Tape::new();
        let a2 = tape2.parameter(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let sm2 = tape2.softmax(a2, 1);
        // Multiply by a mask to select element 0
        let mask = tape2.constant(vec![1.0, 0.0, 0.0], vec![1, 3]);
        let selected = tape2.mul(sm2, mask);
        let s2 = tape2.sum(selected, None);
        tape2.backward(s2);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![1, 3]);
            let si = t.softmax(ai, 1);
            let mi = t.constant(vec![1.0, 0.0, 0.0], vec![1, 3]);
            let sel = t.mul(si, mi);
            t.get_data(sel).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![1.0, 2.0, 3.0]], 1e-5);
        assert_grad_close(&tape2.get_grad(a2).unwrap(), &ng[0], 1e-4, "softmax selective");
    }

    // ---- Test 12: Conv2d gradient ----
    #[test]
    fn test_conv2d_grad() {
        let mut tape = Tape::new();
        // 1 batch, 1 channel, 3x3 input
        let input = tape.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], vec![1, 1, 3, 3]);
        // 1 out channel, 1 in channel, 2x2 kernel
        let kernel = tape.parameter(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let out = tape.conv2d(input, kernel, 1, 0);
        let s = tape.sum(out, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let inp = t.parameter(params[0].clone(), vec![1, 1, 3, 3]);
            let ker = t.parameter(params[1].clone(), vec![1, 1, 2, 2]);
            let o = t.conv2d(inp, ker, 1, 0);
            t.get_data(o).iter().sum()
        };
        let ng = numerical_grad(f,
            &[vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
              vec![1.0, 0.0, 0.0, 1.0]], 1e-5);
        assert_grad_close(&tape.get_grad(input).unwrap(), &ng[0], 1e-3, "conv2d input");
        assert_grad_close(&tape.get_grad(kernel).unwrap(), &ng[1], 1e-3, "conv2d kernel");
    }

    // ---- Test 13: Div gradient ----
    #[test]
    fn test_div_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![6.0, 8.0], vec![2]);
        let b = tape.parameter(vec![2.0, 4.0], vec![2]);
        let c = tape.div(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![2]);
            let bi = t.parameter(params[1].clone(), vec![2]);
            let ci = t.div(ai, bi);
            t.get_data(ci).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![6.0, 8.0], vec![2.0, 4.0]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-4, "div grad_a");
        assert_grad_close(&tape.get_grad(b).unwrap(), &ng[1], 1e-4, "div grad_b");
    }

    // ---- Test 14: Pow gradient ----
    #[test]
    fn test_pow_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![2.0, 3.0], vec![2]);
        let b = tape.parameter(vec![3.0, 2.0], vec![2]);
        let c = tape.pow(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![2]);
            let bi = t.parameter(params[1].clone(), vec![2]);
            let ci = t.pow(ai, bi);
            t.get_data(ci).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![2.0, 3.0], vec![3.0, 2.0]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-3, "pow grad_a");
        assert_grad_close(&tape.get_grad(b).unwrap(), &ng[1], 1e-3, "pow grad_b");
    }

    // ---- Test 15: Mean gradient ----
    #[test]
    fn test_mean_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let m = tape.mean(a, None);
        tape.backward(m);
        let ga = tape.get_grad(a).unwrap();
        for g in &ga {
            assert!(approx_eq(*g, 0.25, 1e-10));
        }
    }

    // ---- Test 16: TrackedTensor API ----
    #[test]
    fn test_tracked_tensor_api() {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let x = TrackedTensor::parameter(&tape, vec![1.0, 2.0, 3.0], vec![3]);
        let y = TrackedTensor::parameter(&tape, vec![4.0, 5.0, 6.0], vec![3]);
        let z = x.mul(&y);
        let s = z.sum(None);
        tape.borrow_mut().backward(s.node_id);
        assert_eq!(x.grad().unwrap(), vec![4.0, 5.0, 6.0]);
        assert_eq!(y.grad().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    // ---- Test 17: grad() functional API ----
    #[test]
    fn test_grad_function() {
        let tape = Rc::new(RefCell::new(Tape::new()));
        let x = TrackedTensor::parameter(&tape, vec![2.0, 3.0], vec![2]);
        let grads = grad(|params| {
            let sq = params[0].mul(&params[0]);
            sq.sum(None)
        }, &[x]);
        assert_eq!(grads[0], vec![4.0, 6.0]);
    }

    // ---- Test 18: Adam optimizer reduces loss ----
    #[test]
    fn test_adam_reduces_loss() {
        let mut tape = Tape::new();
        let w = tape.parameter(vec![0.5, -0.5, 0.3, -0.3], vec![2, 2]);
        let x = tape.input(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let target = tape.input(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

        let pred = tape.matmul(x, w);
        let loss = tape.mse(pred, target);
        let initial_loss = tape.get_data(loss)[0];

        tape.backward(loss);
        let g = tape.get_grad(w).unwrap();

        let mut opt = AdamOptimizer::new(0.1, &[4]);
        opt.step(&mut tape, &[w], &[g]);

        // Re-evaluate with updated weights
        let mut tape2 = Tape::new();
        let w2 = tape2.parameter(tape.get_data(w).to_vec(), vec![2, 2]);
        let x2 = tape2.input(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);
        let t2 = tape2.input(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);
        let pred2 = tape2.matmul(x2, w2);
        let loss2 = tape2.mse(pred2, t2);
        let new_loss = tape2.get_data(loss2)[0];

        assert!(new_loss < initial_loss, "Adam should reduce loss: {} >= {}", new_loss, initial_loss);
    }

    // ---- Test 19: Higher-order gradient (Hessian diagonal) ----
    #[test]
    fn test_hessian_diagonal() {
        // f(x) = x^2, so f''(x) = 2
        let f = |params: &[TrackedTensor]| -> TrackedTensor {
            let sq = params[0].mul(&params[0]);
            sq.sum(None)
        };
        let hess = hessian_diagonal(f, &[vec![3.0]], &[vec![1]], 1e-4);
        assert!(approx_eq(hess[0][0], 2.0, 1e-2), "Hessian of x^2 should be 2, got {}", hess[0][0]);
    }

    // ---- Test 20: Layer norm gradient ----
    #[test]
    fn test_layer_norm_grad() {
        let mut tape = Tape::new();
        let x = tape.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let gamma = tape.parameter(vec![1.0, 1.0, 1.0], vec![3]);
        let beta = tape.parameter(vec![0.0, 0.0, 0.0], vec![3]);
        let ln = tape.layer_norm(x, gamma, beta, 1e-5);
        let s = tape.sum(ln, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let xi = t.parameter(params[0].clone(), vec![2, 3]);
            let gi = t.parameter(params[1].clone(), vec![3]);
            let bi = t.parameter(params[2].clone(), vec![3]);
            let li = t.layer_norm(xi, gi, bi, 1e-5);
            t.get_data(li).iter().sum()
        };
        let ng = numerical_grad(f, &[
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![1.0, 1.0, 1.0],
            vec![0.0, 0.0, 0.0],
        ], 1e-4);
        assert_grad_close(&tape.get_grad(x).unwrap(), &ng[0], 1e-2, "layernorm x");
        assert_grad_close(&tape.get_grad(gamma).unwrap(), &ng[1], 1e-2, "layernorm gamma");
        assert_grad_close(&tape.get_grad(beta).unwrap(), &ng[2], 1e-2, "layernorm beta");
    }

    // ---- Test 21: Gelu gradient ----
    #[test]
    fn test_gelu_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![-1.0, 0.0, 1.0, 2.0], vec![4]);
        let g = tape.gelu(a);
        let s = tape.sum(g, None);
        tape.backward(s);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let ai = t.parameter(params[0].clone(), vec![4]);
            let gi = t.gelu(ai);
            t.get_data(gi).iter().sum()
        };
        let ng = numerical_grad(f, &[vec![-1.0, 0.0, 1.0, 2.0]], 1e-5);
        assert_grad_close(&tape.get_grad(a).unwrap(), &ng[0], 1e-3, "gelu");
    }

    // ---- Test 22: Cross entropy gradient ----
    #[test]
    fn test_cross_entropy_grad() {
        let mut tape = Tape::new();
        let logits = tape.parameter(vec![1.0, 2.0, 3.0, 1.0, 3.0, 2.0], vec![2, 3]);
        let targets = tape.input(vec![2.0, 0.0], vec![2]); // class indices
        let loss = tape.cross_entropy(logits, targets);
        tape.backward(loss);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let li = t.parameter(params[0].clone(), vec![2, 3]);
            let ti = t.input(vec![2.0, 0.0], vec![2]);
            let lo = t.cross_entropy(li, ti);
            t.get_data(lo)[0]
        };
        let ng = numerical_grad(f, &[vec![1.0, 2.0, 3.0, 1.0, 3.0, 2.0]], 1e-5);
        assert_grad_close(&tape.get_grad(logits).unwrap(), &ng[0], 1e-3, "cross_entropy");
    }

    // ---- Test 23: MSE gradient ----
    #[test]
    fn test_mse_grad() {
        let mut tape = Tape::new();
        let pred = tape.parameter(vec![1.0, 2.0, 3.0], vec![3]);
        let target = tape.input(vec![1.5, 2.5, 2.5], vec![3]);
        let loss = tape.mse(pred, target);
        tape.backward(loss);

        let f = |params: &[Vec<f64>]| -> f64 {
            let mut t = Tape::new();
            let pi = t.parameter(params[0].clone(), vec![3]);
            let ti = t.input(vec![1.5, 2.5, 2.5], vec![3]);
            let lo = t.mse(pi, ti);
            t.get_data(lo)[0]
        };
        let ng = numerical_grad(f, &[vec![1.0, 2.0, 3.0]], 1e-5);
        assert_grad_close(&tape.get_grad(pred).unwrap(), &ng[0], 1e-4, "mse");
    }

    // ---- Test 24: Sub gradient ----
    #[test]
    fn test_sub_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![5.0, 3.0], vec![2]);
        let b = tape.parameter(vec![1.0, 2.0], vec![2]);
        let c = tape.sub(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);
        assert_eq!(tape.get_grad(a).unwrap(), vec![1.0, 1.0]);
        assert_eq!(tape.get_grad(b).unwrap(), vec![-1.0, -1.0]);
    }

    // ---- Test 25: Transpose gradient ----
    #[test]
    fn test_transpose_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t = tape.transpose(a);
        let s = tape.sum(t, None);
        tape.backward(s);
        // Gradient of sum through transpose should be all 1s
        assert_eq!(tape.get_grad(a).unwrap(), vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    // ---- Test 26: Broadcast add gradient ----
    #[test]
    fn test_broadcast_add_grad() {
        let mut tape = Tape::new();
        let a = tape.parameter(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = tape.parameter(vec![0.1, 0.2, 0.3], vec![3]);
        let c = tape.broadcast_add(a, b);
        let s = tape.sum(c, None);
        tape.backward(s);
        assert_eq!(tape.get_grad(a).unwrap(), vec![1.0; 6]);
        assert_eq!(tape.get_grad(b).unwrap(), vec![2.0, 2.0, 2.0]); // summed over 2 rows
    }

    // ---- Test 27: Training loop with autograd ----
    #[test]
    fn test_training_loop() {
        // Simple linear regression: y = 2*x + 1
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![3.0, 5.0, 7.0, 9.0];

        let mut w_data = vec![0.0];
        let mut b_data = vec![0.0];
        let lr = 0.05;

        for _ in 0..500 {
            let mut tape = Tape::new();
            let w = tape.parameter(w_data.clone(), vec![1]);
            let b = tape.parameter(b_data.clone(), vec![1]);

            let mut total_loss = 0.0;
            // Manual loop over data points
            for (&x_val, &y_val) in xs.iter().zip(ys.iter()) {
                let x = tape.input(vec![x_val], vec![1]);
                let y_true = tape.input(vec![y_val], vec![1]);
                let wx = tape.mul(w, x);
                let pred = tape.add(wx, b);
                let diff = tape.sub(pred, y_true);
                let sq = tape.mul(diff, diff);
                total_loss += tape.get_data(sq)[0];
            }
            // We just need the gradient direction, not the exact loss node
            // Use the last sq as proxy — the gradients accumulated from the diamond pattern
            // Actually, let's use a proper approach: build the full computation
            let mut tape2 = Tape::new();
            let w2 = tape2.parameter(w_data.clone(), vec![1, 1]);
            let b2 = tape2.parameter(b_data.clone(), vec![1]);
            let x_all = tape2.input(xs.clone(), vec![4, 1]);
            let y_all = tape2.input(ys.clone(), vec![4, 1]);
            let pred = tape2.matmul(x_all, w2);
            let pred_b = tape2.broadcast_add(pred, b2);
            let loss = tape2.mse(pred_b, y_all);
            tape2.backward(loss);

            let gw = tape2.get_grad(w2).unwrap();
            let gb = tape2.get_grad(b2).unwrap();

            w_data[0] -= lr * gw[0];
            b_data[0] -= lr * gb[0];
        }

        assert!(approx_eq(w_data[0], 2.0, 0.1), "w should be ~2.0, got {}", w_data[0]);
        assert!(approx_eq(b_data[0], 1.0, 0.1), "b should be ~1.0, got {}", b_data[0]);
    }
}

// ---------------------------------------------------------------------------
// Interpreter builtins (exported for use by interpreter.rs)
// ---------------------------------------------------------------------------

use crate::interpreter::{Env, Value};

fn flatten_f64(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Float(f) => Ok(vec![*f]),
        Value::Int(i) => Ok(vec![*i as f64]),
        Value::Array(arr) => { let mut out = Vec::new(); for item in arr { out.extend(flatten_f64(item)?); } Ok(out) }
        _ => Err("expected numeric data".into()),
    }
}
fn to_usize_vec(v: &Value) -> Result<Vec<usize>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|v| match v { Value::Int(n) => Ok(*n as usize), Value::Float(f) => Ok(*f as usize), _ => Err("shape: integers".into()) }).collect(),
        _ => Err("shape: array".into()),
    }
}
fn to_id(v: &Value) -> Result<usize, String> { match v { Value::Int(n) => Ok(*n as usize), Value::Float(f) => Ok(*f as usize), _ => Err("expected id".into()) } }
fn to_id_array(v: &Value) -> Result<Vec<usize>, String> { match v { Value::Array(arr) => arr.iter().map(to_id).collect(), _ => Err("expected array of ids".into()) } }
fn tape(env: &mut Env) -> Result<&mut Tape, String> { env.autograd_tape.as_mut().ok_or_else(|| "No autograd tape. Call autograd_new() first.".into()) }

pub fn builtin_autograd_new(env: &mut Env, _a: Vec<Value>) -> Result<Value, String> { env.autograd_tape = Some(Tape::new()); env.autograd_adam = None; Ok(Value::Void) }
pub fn builtin_autograd_tensor(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() < 2 || args.len() > 3 { return Err("autograd_tensor: (data, shape, [rg])".into()); } let data = flatten_f64(&args[0])?; let shape = to_usize_vec(&args[1])?; let rg = if args.len() == 3 { match &args[2] { Value::Bool(b) => *b, _ => true } } else { true }; let t = tape(env)?; let id = if rg { t.parameter(data, shape) } else { t.input(data, shape) }; Ok(Value::Int(id as i128)) }
pub fn builtin_autograd_input(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("(data, shape)".into()); } let data = flatten_f64(&args[0])?; let shape = to_usize_vec(&args[1])?; Ok(Value::Int(tape(env)?.input(data, shape) as i128)) }
pub fn builtin_autograd_matmul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.matmul(a, b) as i128)) }
pub fn builtin_autograd_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.add(a, b) as i128)) }
pub fn builtin_autograd_mul(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.mul(a, b) as i128)) }
pub fn builtin_autograd_sub(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.sub(a, b) as i128)) }
pub fn builtin_autograd_div(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.div(a, b) as i128)) }
pub fn builtin_autograd_relu(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.relu(to_id(&args[0])?) as i128)) }
pub fn builtin_autograd_sigmoid(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.sigmoid(to_id(&args[0])?) as i128)) }
pub fn builtin_autograd_tanh(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.tanh_op(to_id(&args[0])?) as i128)) }
pub fn builtin_autograd_softmax(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, ax) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.softmax(a, ax) as i128)) }
pub fn builtin_autograd_exp(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.exp(to_id(&args[0])?) as i128)) }
pub fn builtin_autograd_log(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.log(to_id(&args[0])?) as i128)) }
pub fn builtin_autograd_sum(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.sum(to_id(&args[0])?, None) as i128)) }
pub fn builtin_autograd_mean(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } Ok(Value::Int(tape(env)?.mean(to_id(&args[0])?, None) as i128)) }
pub fn builtin_autograd_mse(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.mse(a, b) as i128)) }
pub fn builtin_autograd_broadcast_add(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 2 { return Err("2 args".into()); } let (a, b) = (to_id(&args[0])?, to_id(&args[1])?); Ok(Value::Int(tape(env)?.broadcast_add(a, b) as i128)) }
pub fn builtin_autograd_backward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } tape(env)?.backward(to_id(&args[0])?); Ok(Value::Void) }
pub fn builtin_autograd_grad(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } let n = to_id(&args[0])?; match tape(env)?.get_grad(n) { Some(g) => Ok(Value::Array(g.into_iter().map(Value::Float).collect())), None => Ok(Value::Array(vec![])) } }
pub fn builtin_autograd_data(env: &mut Env, args: Vec<Value>) -> Result<Value, String> { if args.len() != 1 { return Err("1 arg".into()); } let d = tape(env)?.get_data(to_id(&args[0])?).to_vec(); Ok(Value::Array(d.into_iter().map(Value::Float).collect())) }
pub fn builtin_autograd_zero_grad(env: &mut Env, _a: Vec<Value>) -> Result<Value, String> { tape(env)?.zero_grad(); Ok(Value::Void) }
pub fn builtin_autograd_adam_step(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("(param_ids, lr)".into()); }
    let param_ids = to_id_array(&args[0])?;
    let lr = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lr: number".into()) };
    let t = env.autograd_tape.as_ref().ok_or("No autograd tape")?;
    let grads: Vec<Vec<f64>> = param_ids.iter().map(|&p| t.get_grad(p).unwrap_or_else(|| vec![0.0; t.get_data(p).len()])).collect();
    let sizes: Vec<usize> = param_ids.iter().map(|&p| t.get_data(p).len()).collect();
    if env.autograd_adam.is_none() { env.autograd_adam = Some(AdamOptimizer::new(lr, &sizes)); }
    let adam = env.autograd_adam.as_mut().unwrap();
    adam.lr = lr;
    let t = env.autograd_tape.as_mut().unwrap();
    adam.step(t, &param_ids, &grads);
    Ok(Value::Void)
}
