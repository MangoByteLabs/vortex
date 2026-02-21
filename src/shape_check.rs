//! Compile-time tensor shape checking — Vortex's dependent type system for tensors.
//!
//! `Tensor<f32, [B, T, D]>` shapes are checked at compile time:
//! - `[B, T, D] @ [D, H]` -> `[B, T, H]`
//! - `[B, T, D] @ [H, D]` -> ShapeError (inner dims don't match)
//! - `[B, T, D] + [D]` -> `[B, T, D]` (broadcast)

use crate::typeck::Dim;
use std::fmt;

/// Arithmetic expression over shape dimensions.
#[derive(Debug, Clone, PartialEq)]
pub enum ShapeExpr {
    Dim(Dim),
    Add(Box<ShapeExpr>, Box<ShapeExpr>),
    Mul(Box<ShapeExpr>, Box<ShapeExpr>),
    Div(Box<ShapeExpr>, Box<ShapeExpr>),
}

/// Errors from shape checking.
#[derive(Debug, Clone)]
pub enum ShapeError {
    DimMismatch {
        expected: Dim,
        got: Dim,
        context: String,
    },
    RankMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    ProductMismatch {
        from: Vec<Dim>,
        to: Vec<Dim>,
    },
    InvalidAxis {
        axis: usize,
        rank: usize,
    },
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ShapeError::DimMismatch { expected, got, context } => {
                write!(f, "shape dimension mismatch in {}: expected `{}`, got `{}`",
                    context, dim_display(expected), dim_display(got))
            }
            ShapeError::RankMismatch { expected, got, context } => {
                write!(f, "rank mismatch in {}: expected {}, got {}", context, expected, got)
            }
            ShapeError::ProductMismatch { from, to } => {
                write!(f, "reshape product mismatch: {} vs {}",
                    dims_display(from), dims_display(to))
            }
            ShapeError::InvalidAxis { axis, rank } => {
                write!(f, "invalid axis {} for tensor of rank {}", axis, rank)
            }
        }
    }
}

fn dim_display(d: &Dim) -> String {
    match d {
        Dim::Lit(n) => n.to_string(),
        Dim::Sym(s) => s.clone(),
        Dim::Dynamic => "?".to_string(),
    }
}

fn dims_display(dims: &[Dim]) -> String {
    let parts: Vec<String> = dims.iter().map(dim_display).collect();
    format!("[{}]", parts.join(", "))
}

/// Shape checker for compile-time tensor dimension verification.
pub struct ShapeChecker;

impl ShapeChecker {
    /// Check if two dimensions are symbolically equal.
    pub fn dims_equal(a: &Dim, b: &Dim) -> bool {
        match (a, b) {
            (Dim::Lit(x), Dim::Lit(y)) => x == y,
            (Dim::Sym(x), Dim::Sym(y)) => x == y,
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            _ => false,
        }
    }

    /// MatMul: [.., M, K] @ [.., K, N] -> [.., M, N]
    /// Supports 2D and batched (3D+) with batch dimension broadcasting.
    pub fn check_matmul(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        if lhs.len() < 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                got: lhs.len(),
                context: "matmul lhs".to_string(),
            });
        }
        if rhs.len() < 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                got: rhs.len(),
                context: "matmul rhs".to_string(),
            });
        }

        let lhs_k = &lhs[lhs.len() - 1];
        let rhs_k = &rhs[rhs.len() - 2];

        if !Self::dims_equal(lhs_k, rhs_k) {
            return Err(ShapeError::DimMismatch {
                expected: lhs_k.clone(),
                got: rhs_k.clone(),
                context: "matmul inner dimensions".to_string(),
            });
        }

        let m = &lhs[lhs.len() - 2];
        let n = &rhs[rhs.len() - 1];

        // Broadcast batch dimensions
        let lhs_batch = &lhs[..lhs.len() - 2];
        let rhs_batch = &rhs[..rhs.len() - 2];
        let batch = Self::broadcast_batch(lhs_batch, rhs_batch)?;

        let mut result = batch;
        result.push(m.clone());
        result.push(n.clone());
        Ok(result)
    }

    /// Broadcast batch dimensions (all dims except last 2).
    fn broadcast_batch(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        if lhs.is_empty() {
            return Ok(rhs.to_vec());
        }
        if rhs.is_empty() {
            return Ok(lhs.to_vec());
        }

        let max_len = lhs.len().max(rhs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let l_idx = if i < max_len - lhs.len() { None } else { Some(i - (max_len - lhs.len())) };
            let r_idx = if i < max_len - rhs.len() { None } else { Some(i - (max_len - rhs.len())) };

            match (l_idx.map(|j| &lhs[j]), r_idx.map(|j| &rhs[j])) {
                (Some(l), Some(r)) => {
                    if Self::dims_equal(l, r) {
                        result.push(l.clone());
                    } else if matches!(l, Dim::Lit(1)) {
                        result.push(r.clone());
                    } else if matches!(r, Dim::Lit(1)) {
                        result.push(l.clone());
                    } else {
                        return Err(ShapeError::DimMismatch {
                            expected: l.clone(),
                            got: r.clone(),
                            context: "batch dimension broadcast".to_string(),
                        });
                    }
                }
                (Some(l), None) => result.push(l.clone()),
                (None, Some(r)) => result.push(r.clone()),
                (None, None) => unreachable!(),
            }
        }
        Ok(result)
    }

    /// Add/Sub: shapes must match or broadcast (NumPy-style).
    pub fn check_broadcast(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        let max_len = lhs.len().max(rhs.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let l_idx = if i < max_len - lhs.len() { None } else { Some(i - (max_len - lhs.len())) };
            let r_idx = if i < max_len - rhs.len() { None } else { Some(i - (max_len - rhs.len())) };

            match (l_idx.map(|j| &lhs[j]), r_idx.map(|j| &rhs[j])) {
                (Some(l), Some(r)) => {
                    if Self::dims_equal(l, r) {
                        result.push(l.clone());
                    } else if matches!(l, Dim::Lit(1)) {
                        result.push(r.clone());
                    } else if matches!(r, Dim::Lit(1)) {
                        result.push(l.clone());
                    } else {
                        return Err(ShapeError::DimMismatch {
                            expected: l.clone(),
                            got: r.clone(),
                            context: "broadcast".to_string(),
                        });
                    }
                }
                (Some(l), None) => result.push(l.clone()),
                (None, Some(r)) => result.push(r.clone()),
                (None, None) => unreachable!(),
            }
        }
        Ok(result)
    }

    /// Softmax: preserves shape, validates axis.
    pub fn check_softmax(input: &[Dim], axis: usize) -> Result<Vec<Dim>, ShapeError> {
        if axis >= input.len() {
            return Err(ShapeError::InvalidAxis { axis, rank: input.len() });
        }
        Ok(input.to_vec())
    }

    /// LayerNorm: last dim of x must match gamma shape.
    pub fn check_layer_norm(x: &[Dim], gamma: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        if gamma.len() != 1 {
            return Err(ShapeError::RankMismatch {
                expected: 1,
                got: gamma.len(),
                context: "layer_norm gamma".to_string(),
            });
        }
        if x.is_empty() {
            return Err(ShapeError::RankMismatch {
                expected: 1,
                got: 0,
                context: "layer_norm input".to_string(),
            });
        }
        let last = &x[x.len() - 1];
        if !Self::dims_equal(last, &gamma[0]) {
            return Err(ShapeError::DimMismatch {
                expected: last.clone(),
                got: gamma[0].clone(),
                context: "layer_norm last dim vs gamma".to_string(),
            });
        }
        Ok(x.to_vec())
    }

    /// Transpose: swap last two dims.
    pub fn check_transpose(input: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        if input.len() < 2 {
            return Err(ShapeError::RankMismatch {
                expected: 2,
                got: input.len(),
                context: "transpose".to_string(),
            });
        }
        let mut result = input.to_vec();
        let n = result.len();
        result.swap(n - 2, n - 1);
        Ok(result)
    }

    /// Reshape: symbolic product check. Concrete dims must have matching products.
    /// If any symbolic dim is present, we allow the reshape (can't verify at compile time).
    pub fn check_reshape(input: &[Dim], target: &[Dim]) -> Result<Vec<Dim>, ShapeError> {
        let input_prod = Self::concrete_product(input);
        let target_prod = Self::concrete_product(target);

        match (input_prod, target_prod) {
            (Some(a), Some(b)) if a != b => {
                Err(ShapeError::ProductMismatch {
                    from: input.to_vec(),
                    to: target.to_vec(),
                })
            }
            _ => Ok(target.to_vec()),
        }
    }

    /// Compute concrete product of all dims, or None if any is symbolic/dynamic.
    fn concrete_product(dims: &[Dim]) -> Option<u64> {
        let mut prod = 1u64;
        for d in dims {
            match d {
                Dim::Lit(n) => prod *= n,
                _ => return None,
            }
        }
        Some(prod)
    }

    /// Simplify a shape expression: D*1 -> D, 4*8 -> 32, etc.
    pub fn simplify(expr: &ShapeExpr) -> Dim {
        match expr {
            ShapeExpr::Dim(d) => d.clone(),
            ShapeExpr::Add(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x + y),
                    _ => Dim::Dynamic, // can't simplify symbolic add
                }
            }
            ShapeExpr::Mul(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x * y),
                    (Dim::Lit(1), other) | (other, Dim::Lit(1)) => other.clone(),
                    (Dim::Lit(0), _) | (_, Dim::Lit(0)) => Dim::Lit(0),
                    _ => Dim::Dynamic,
                }
            }
            ShapeExpr::Div(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) if *y != 0 => Dim::Lit(x / y),
                    (d, Dim::Lit(1)) => d.clone(),
                    _ => Dim::Dynamic,
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sym(s: &str) -> Dim { Dim::Sym(s.to_string()) }
    fn lit(n: u64) -> Dim { Dim::Lit(n) }

    // Test 1: [B, T, D] @ [D, H] -> [B, T, H]
    #[test]
    fn test_matmul_batched_2d() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("D"), sym("H")];
        let result = ShapeChecker::check_matmul(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("H")]);
    }

    // Test 2: [B, T, D] @ [H, D] -> ShapeError
    #[test]
    fn test_matmul_inner_dim_mismatch() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("H"), sym("D")];
        let err = ShapeChecker::check_matmul(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, ShapeError::DimMismatch { .. }));
    }

    // Test 3: [B, T, D] + [D] -> [B, T, D] (broadcast)
    #[test]
    fn test_broadcast_trailing() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("D")];
        let result = ShapeChecker::check_broadcast(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("D")]);
    }

    // Test 4: [B, T, D] + [B, T, D] -> [B, T, D]
    #[test]
    fn test_broadcast_same_shape() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("B"), sym("T"), sym("D")];
        let result = ShapeChecker::check_broadcast(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("D")]);
    }

    // Test 5: [B, T, D] + [B, T, H] -> ShapeError
    #[test]
    fn test_broadcast_mismatch() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("B"), sym("T"), sym("H")];
        let err = ShapeChecker::check_broadcast(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, ShapeError::DimMismatch { .. }));
    }

    // Test 6: transpose [B, T, D] -> [B, D, T]
    #[test]
    fn test_transpose() {
        let input = vec![sym("B"), sym("T"), sym("D")];
        let result = ShapeChecker::check_transpose(&input).unwrap();
        assert_eq!(result, vec![sym("B"), sym("D"), sym("T")]);
    }

    // Test 7: softmax axis 2 on [B, T, D] -> [B, T, D]
    #[test]
    fn test_softmax() {
        let input = vec![sym("B"), sym("T"), sym("D")];
        let result = ShapeChecker::check_softmax(&input, 2).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("D")]);
    }

    // Test 8: concrete [4, 128] @ [128, 64] -> [4, 64]
    #[test]
    fn test_matmul_concrete() {
        let lhs = vec![lit(4), lit(128)];
        let rhs = vec![lit(128), lit(64)];
        let result = ShapeChecker::check_matmul(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![lit(4), lit(64)]);
    }

    // Test 9: mixed [B, 128] @ [128, D] -> [B, D]
    #[test]
    fn test_matmul_mixed() {
        let lhs = vec![sym("B"), lit(128)];
        let rhs = vec![lit(128), sym("D")];
        let result = ShapeChecker::check_matmul(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("D")]);
    }

    // Test 10: simplify D * 1 -> D, 4 * 8 -> 32
    #[test]
    fn test_simplify() {
        let expr1 = ShapeExpr::Mul(
            Box::new(ShapeExpr::Dim(sym("D"))),
            Box::new(ShapeExpr::Dim(lit(1))),
        );
        assert_eq!(ShapeChecker::simplify(&expr1), sym("D"));

        let expr2 = ShapeExpr::Mul(
            Box::new(ShapeExpr::Dim(lit(4))),
            Box::new(ShapeExpr::Dim(lit(8))),
        );
        assert_eq!(ShapeChecker::simplify(&expr2), lit(32));
    }

    // Test: concrete matmul mismatch
    #[test]
    fn test_matmul_concrete_mismatch() {
        let lhs = vec![lit(4), lit(128)];
        let rhs = vec![lit(64), lit(32)];
        let err = ShapeChecker::check_matmul(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, ShapeError::DimMismatch { .. }));
    }

    // Test: layer norm
    #[test]
    fn test_layer_norm() {
        let x = vec![sym("B"), sym("T"), sym("D")];
        let gamma = vec![sym("D")];
        let result = ShapeChecker::check_layer_norm(&x, &gamma).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("D")]);
    }

    // Test: layer norm mismatch
    #[test]
    fn test_layer_norm_mismatch() {
        let x = vec![sym("B"), sym("T"), sym("D")];
        let gamma = vec![sym("H")];
        let err = ShapeChecker::check_layer_norm(&x, &gamma).unwrap_err();
        assert!(matches!(err, ShapeError::DimMismatch { .. }));
    }

    // Test: reshape concrete
    #[test]
    fn test_reshape_ok() {
        let input = vec![lit(4), lit(8)];
        let target = vec![lit(2), lit(16)];
        let result = ShapeChecker::check_reshape(&input, &target).unwrap();
        assert_eq!(result, vec![lit(2), lit(16)]);
    }

    // Test: reshape product mismatch
    #[test]
    fn test_reshape_mismatch() {
        let input = vec![lit(4), lit(8)];
        let target = vec![lit(3), lit(16)];
        let err = ShapeChecker::check_reshape(&input, &target).unwrap_err();
        assert!(matches!(err, ShapeError::ProductMismatch { .. }));
    }

    // Test: softmax invalid axis
    #[test]
    fn test_softmax_invalid_axis() {
        let input = vec![sym("B"), sym("T")];
        let err = ShapeChecker::check_softmax(&input, 5).unwrap_err();
        assert!(matches!(err, ShapeError::InvalidAxis { .. }));
    }

    // Test: 2D matmul [M, K] @ [K, N] -> [M, N]
    #[test]
    fn test_matmul_2d() {
        let lhs = vec![sym("M"), sym("K")];
        let rhs = vec![sym("K"), sym("N")];
        let result = ShapeChecker::check_matmul(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("M"), sym("N")]);
    }

    // Test: simplify add
    #[test]
    fn test_simplify_add() {
        let expr = ShapeExpr::Add(
            Box::new(ShapeExpr::Dim(lit(10))),
            Box::new(ShapeExpr::Dim(lit(20))),
        );
        assert_eq!(ShapeChecker::simplify(&expr), lit(30));
    }

    // Test: simplify div
    #[test]
    fn test_simplify_div() {
        let expr = ShapeExpr::Div(
            Box::new(ShapeExpr::Dim(lit(32))),
            Box::new(ShapeExpr::Dim(lit(4))),
        );
        assert_eq!(ShapeChecker::simplify(&expr), lit(8));

        // D / 1 -> D
        let expr2 = ShapeExpr::Div(
            Box::new(ShapeExpr::Dim(sym("D"))),
            Box::new(ShapeExpr::Dim(lit(1))),
        );
        assert_eq!(ShapeChecker::simplify(&expr2), sym("D"));
    }
}
