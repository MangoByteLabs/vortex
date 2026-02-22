//! Compile-time tensor shape checking — Vortex's dependent type system for tensors.
//!
//! This module performs whole-program shape inference by walking the AST after
//! type checking.  It tracks tensor shapes through variable bindings, function
//! calls, and arithmetic, then reports mismatches with rich, span-annotated
//! error messages.
//!
//! # Shape type system
//!
//! Every expression is assigned a `ShapeType`:
//! - `Scalar(kind)` — non-tensor scalar
//! - `Tensor { elem, shape }` — tensor with element type and shape dims
//! - `Function { params, ret }` — for first-class function values
//! - `Unknown` — not yet determined / not relevant
//!
//! Dimension expressions (`DimExpr`) can be concrete literals, symbolic
//! variables, binary ops, or wildcards.  A constraint solver unifies
//! symbolic dimensions during checking (e.g. `N` unifies with `768`).

use crate::ast::*;
use crate::lexer::Span;
use crate::typeck::Dim;
use std::collections::HashMap;
use std::fmt;

// ───────────────────────────── Shape types ──────────────────────────────

/// Scalar element types understood by the shape checker.
#[derive(Debug, Clone, PartialEq)]
pub enum ScalarKind {
    I64,
    F64,
    F32,
    F16,
    Bool,
    String,
    Other(std::string::String),
}

/// The shape-level type assigned to every expression.
#[derive(Debug, Clone)]
pub enum ShapeType {
    Scalar(ScalarKind),
    Tensor {
        elem: ScalarKind,
        shape: Vec<DimExpr>,
    },
    Function {
        params: Vec<ShapeType>,
        ret: Box<ShapeType>,
    },
    Unknown,
}

/// A dimension expression that may be concrete, symbolic, or compound.
#[derive(Debug, Clone, PartialEq)]
pub enum DimExpr {
    /// Concrete literal dimension, e.g. 768
    Lit(u64),
    /// Symbolic variable, e.g. `N`, `batch`
    Var(std::string::String),
    /// Binary operation on dimensions
    BinOp(Box<DimExpr>, DimOp, Box<DimExpr>),
    /// Wildcard — matches any dimension
    Any,
}

/// Binary operations over dimensions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DimOp {
    Add,
    Sub,
    Mul,
    Div,
}

impl fmt::Display for DimExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DimExpr::Lit(n) => write!(f, "{}", n),
            DimExpr::Var(s) => write!(f, "{}", s),
            DimExpr::BinOp(a, op, b) => {
                let sym = match op {
                    DimOp::Add => "+",
                    DimOp::Sub => "-",
                    DimOp::Mul => "*",
                    DimOp::Div => "/",
                };
                write!(f, "({} {} {})", a, sym, b)
            }
            DimExpr::Any => write!(f, "_"),
        }
    }
}

fn dims_display(dims: &[DimExpr]) -> std::string::String {
    let parts: Vec<std::string::String> = dims.iter().map(|d| format!("{}", d)).collect();
    format!("[{}]", parts.join(", "))
}

fn shape_type_display(st: &ShapeType) -> std::string::String {
    match st {
        ShapeType::Scalar(k) => format!("{:?}", k),
        ShapeType::Tensor { elem, shape } => {
            format!("Tensor<{:?}, {}>", elem, dims_display(shape))
        }
        ShapeType::Function { .. } => "fn(...)".to_string(),
        ShapeType::Unknown => "unknown".to_string(),
    }
}

// ───────────────────────────── Errors ──────────────────────────────────

/// A shape error with location and optional hint.
#[derive(Debug, Clone)]
pub struct ShapeError {
    pub message: std::string::String,
    pub span: Span,
    pub hint: Option<std::string::String>,
}

impl fmt::Display for ShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shape error: {}", self.message)?;
        if let Some(h) = &self.hint {
            write!(f, "\n  = hint: {}", h)?;
        }
        Ok(())
    }
}

// ───────────────────────────── Constraints ──────────────────────────────

/// A constraint that two dimension expressions must be equal.
#[derive(Debug, Clone)]
pub struct DimConstraint {
    pub left: DimExpr,
    pub right: DimExpr,
    pub reason: std::string::String,
    pub span: Span,
}

// ───────────────────────────── Checker ──────────────────────────────────

/// The main shape checker.  Walks the AST and infers / checks tensor shapes.
pub struct ShapeChecker {
    /// Variable → shape-type environment (scoped).
    scopes: Vec<HashMap<std::string::String, ShapeType>>,
    /// Function signatures: name → (param shapes, return shape).
    functions: HashMap<std::string::String, (Vec<ShapeType>, ShapeType)>,
    /// Symbolic dimension bindings discovered during unification.
    dim_bindings: HashMap<std::string::String, DimExpr>,
    /// Accumulated constraints (for reporting).
    pub constraints: Vec<DimConstraint>,
    /// Accumulated errors.
    pub errors: Vec<ShapeError>,
}

impl ShapeChecker {
    pub fn new() -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            dim_bindings: HashMap::new(),
            constraints: Vec::new(),
            errors: Vec::new(),
        }
    }

    // ── scope management ──

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, ty: ShapeType) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<&ShapeType> {
        for scope in self.scopes.iter().rev() {
            if let Some(t) = scope.get(name) {
                return Some(t);
            }
        }
        None
    }

    // ── dimension unification ──

    /// Resolve a DimExpr through current bindings.
    fn resolve_dim(&self, d: &DimExpr) -> DimExpr {
        match d {
            DimExpr::Var(name) => {
                if let Some(bound) = self.dim_bindings.get(name) {
                    self.resolve_dim(bound)
                } else {
                    d.clone()
                }
            }
            DimExpr::BinOp(a, op, b) => {
                let ra = self.resolve_dim(a);
                let rb = self.resolve_dim(b);
                // Try to evaluate if both are literals
                if let (DimExpr::Lit(x), DimExpr::Lit(y)) = (&ra, &rb) {
                    let result = match op {
                        DimOp::Add => Some(x + y),
                        DimOp::Sub => x.checked_sub(*y),
                        DimOp::Mul => Some(x * y),
                        DimOp::Div => {
                            if *y != 0 {
                                Some(x / y)
                            } else {
                                None
                            }
                        }
                    };
                    if let Some(val) = result {
                        return DimExpr::Lit(val);
                    }
                }
                DimExpr::BinOp(Box::new(ra), *op, Box::new(rb))
            }
            _ => d.clone(),
        }
    }

    /// Unify two dimension expressions.  Returns Ok(()) on success, Err(msg) on mismatch.
    pub fn unify_dim(&mut self, a: &DimExpr, b: &DimExpr) -> Result<(), std::string::String> {
        let ra = self.resolve_dim(a);
        let rb = self.resolve_dim(b);

        match (&ra, &rb) {
            (DimExpr::Any, _) | (_, DimExpr::Any) => Ok(()),
            (DimExpr::Lit(x), DimExpr::Lit(y)) => {
                if x == y {
                    Ok(())
                } else {
                    Err(format!("dimension mismatch: {} vs {}", x, y))
                }
            }
            (DimExpr::Var(a_name), DimExpr::Var(b_name)) if a_name == b_name => Ok(()),
            (DimExpr::Var(name), other) | (other, DimExpr::Var(name)) => {
                self.dim_bindings
                    .insert(name.clone(), other.clone());
                Ok(())
            }
            (DimExpr::Lit(_), DimExpr::BinOp(..)) | (DimExpr::BinOp(..), DimExpr::Lit(_)) => {
                // Can't verify symbolic bin-ops against literals at compile time — allow
                Ok(())
            }
            (DimExpr::BinOp(..), DimExpr::BinOp(..)) => {
                // Both are compound — allow (conservative)
                Ok(())
            }
        }
    }

    /// Unify two shapes element-wise.
    fn unify_shapes(
        &mut self,
        a: &[DimExpr],
        b: &[DimExpr],
        span: Span,
        context: &str,
    ) -> Result<Vec<DimExpr>, ()> {
        if a.len() != b.len() {
            self.errors.push(ShapeError {
                message: format!(
                    "rank mismatch in {}: expected {}, got {}",
                    context,
                    a.len(),
                    b.len()
                ),
                span,
                hint: Some(format!(
                    "left shape is {}, right shape is {}",
                    dims_display(a),
                    dims_display(b)
                )),
            });
            return Err(());
        }
        let mut result = Vec::with_capacity(a.len());
        for (i, (da, db)) in a.iter().zip(b.iter()).enumerate() {
            if let Err(msg) = self.unify_dim(da, db) {
                self.errors.push(ShapeError {
                    message: format!("{} at dimension {}: {}", context, i, msg),
                    span,
                    hint: Some(format!(
                        "left shape {}, right shape {}",
                        dims_display(a),
                        dims_display(b)
                    )),
                });
                return Err(());
            }
            // Prefer the more concrete of the two
            result.push(self.resolve_dim(da));
        }
        Ok(result)
    }

    // ── broadcasting ──

    /// NumPy-style broadcasting of two shapes.
    pub fn broadcast(
        &mut self,
        a: &[DimExpr],
        b: &[DimExpr],
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        let max_len = a.len().max(b.len());
        let mut result = Vec::with_capacity(max_len);

        for i in 0..max_len {
            let ad = if i < max_len - a.len() {
                None
            } else {
                Some(&a[i - (max_len - a.len())])
            };
            let bd = if i < max_len - b.len() {
                None
            } else {
                Some(&b[i - (max_len - b.len())])
            };

            match (ad, bd) {
                (Some(l), Some(r)) => {
                    let rl = self.resolve_dim(l);
                    let rr = self.resolve_dim(r);
                    if rl == rr || matches!((&rl, &rr), (DimExpr::Any, _) | (_, DimExpr::Any)) {
                        result.push(rl);
                    } else if rl == DimExpr::Lit(1) {
                        result.push(rr);
                    } else if rr == DimExpr::Lit(1) {
                        result.push(rl);
                    } else {
                        // Try unification (symbolic vars)
                        if self.unify_dim(l, r).is_ok() {
                            result.push(self.resolve_dim(l));
                        } else {
                            self.errors.push(ShapeError {
                                message: format!(
                                    "cannot broadcast shapes {} and {}",
                                    dims_display(a),
                                    dims_display(b)
                                ),
                                span,
                                hint: Some(format!(
                                    "dimension {} is {} vs {} — not broadcastable",
                                    i, rl, rr
                                )),
                            });
                            return Err(());
                        }
                    }
                }
                (Some(l), None) => result.push(self.resolve_dim(l)),
                (None, Some(r)) => result.push(self.resolve_dim(r)),
                (None, None) => unreachable!(),
            }
        }
        Ok(result)
    }

    // ── matmul ──

    /// Check matrix multiply: [..., M, K] @ [..., K, N] → [..., M, N]
    pub fn check_matmul(
        &mut self,
        a: &[DimExpr],
        b: &[DimExpr],
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        if a.len() < 2 {
            self.errors.push(ShapeError {
                message: format!("matmul lhs must be at least rank 2, got rank {}", a.len()),
                span,
                hint: None,
            });
            return Err(());
        }
        if b.len() < 2 {
            self.errors.push(ShapeError {
                message: format!("matmul rhs must be at least rank 2, got rank {}", b.len()),
                span,
                hint: None,
            });
            return Err(());
        }

        let lhs_k = &a[a.len() - 1];
        let rhs_k = &b[b.len() - 2];

        if let Err(msg) = self.unify_dim(lhs_k, rhs_k) {
            let lk = self.resolve_dim(lhs_k);
            let rk = self.resolve_dim(rhs_k);
            self.errors.push(ShapeError {
                message: format!("shape mismatch in matrix multiply (@): {}", msg),
                span,
                hint: Some(format!(
                    "input has shape {}, weights has shape {}\n  = expected inner dimension {}, found {}\n  = hint: did you mean weights with shape [{}, {}]?",
                    dims_display(a),
                    dims_display(b),
                    lk, rk, lk,
                    &b[b.len() - 1],
                )),
            });
            return Err(());
        }

        let m = &a[a.len() - 2];
        let n = &b[b.len() - 1];

        // Broadcast batch dimensions
        let a_batch = &a[..a.len() - 2];
        let b_batch = &b[..b.len() - 2];
        let mut batch = if a_batch.is_empty() && b_batch.is_empty() {
            vec![]
        } else if a_batch.is_empty() {
            b_batch.to_vec()
        } else if b_batch.is_empty() {
            a_batch.to_vec()
        } else {
            self.broadcast(a_batch, b_batch, span)?
        };

        batch.push(self.resolve_dim(m));
        batch.push(self.resolve_dim(n));
        Ok(batch)
    }

    // ── transpose ──

    pub fn check_transpose(&self, input: &[DimExpr]) -> Result<Vec<DimExpr>, std::string::String> {
        if input.len() < 2 {
            return Err(format!(
                "transpose requires rank >= 2, got {}",
                input.len()
            ));
        }
        let mut result = input.to_vec();
        let n = result.len();
        result.swap(n - 2, n - 1);
        Ok(result)
    }

    // ── reshape ──

    pub fn check_reshape(
        &mut self,
        input: &[DimExpr],
        target: &[DimExpr],
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        let in_prod = concrete_product(input);
        let tgt_prod = concrete_product(target);
        match (in_prod, tgt_prod) {
            (Some(a), Some(b)) if a != b => {
                self.errors.push(ShapeError {
                    message: format!(
                        "reshape total elements mismatch: {} ({}) vs {} ({})",
                        dims_display(input),
                        a,
                        dims_display(target),
                        b
                    ),
                    span,
                    hint: None,
                });
                Err(())
            }
            _ => Ok(target.to_vec()),
        }
    }

    // ── concat ──

    pub fn check_concat(
        &mut self,
        a: &[DimExpr],
        b: &[DimExpr],
        axis: usize,
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        if a.len() != b.len() {
            self.errors.push(ShapeError {
                message: format!(
                    "concat rank mismatch: {} vs {}",
                    a.len(),
                    b.len()
                ),
                span,
                hint: None,
            });
            return Err(());
        }
        if axis >= a.len() {
            self.errors.push(ShapeError {
                message: format!("concat axis {} out of range for rank {}", axis, a.len()),
                span,
                hint: None,
            });
            return Err(());
        }
        let mut result = Vec::with_capacity(a.len());
        for i in 0..a.len() {
            if i == axis {
                // Sum the dims along concat axis
                let sum = match (&a[i], &b[i]) {
                    (DimExpr::Lit(x), DimExpr::Lit(y)) => DimExpr::Lit(x + y),
                    (x, y) => DimExpr::BinOp(Box::new(x.clone()), DimOp::Add, Box::new(y.clone())),
                };
                result.push(sum);
            } else {
                if let Err(msg) = self.unify_dim(&a[i], &b[i]) {
                    self.errors.push(ShapeError {
                        message: format!(
                            "concat dimension {} mismatch (not the concat axis): {}",
                            i, msg
                        ),
                        span,
                        hint: None,
                    });
                    return Err(());
                }
                result.push(self.resolve_dim(&a[i]));
            }
        }
        Ok(result)
    }

    // ── nn_linear ──

    /// nn_linear(in_features, out_features) applied to input [batch, in_features] → [batch, out_features]
    pub fn check_nn_linear(
        &mut self,
        input: &[DimExpr],
        in_features: &DimExpr,
        out_features: &DimExpr,
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        if input.is_empty() {
            self.errors.push(ShapeError {
                message: "nn_linear input must be at least rank 1".to_string(),
                span,
                hint: None,
            });
            return Err(());
        }
        let last = &input[input.len() - 1];
        if let Err(msg) = self.unify_dim(last, in_features) {
            self.errors.push(ShapeError {
                message: format!(
                    "nn_linear: input last dim {} doesn't match in_features {}: {}",
                    last, in_features, msg
                ),
                span,
                hint: Some(format!(
                    "input shape is {}, expected last dim to be {}",
                    dims_display(input),
                    in_features
                )),
            });
            return Err(());
        }
        let mut result = input[..input.len() - 1].to_vec();
        result.push(out_features.clone());
        Ok(result)
    }

    // ── conv2d ──

    /// Conv2d: input [N, C_in, H, W], kernel [C_out, C_in, kH, kW], stride, padding
    /// → output [N, C_out, H', W'] where H' = (H + 2*pad - kH)/stride + 1
    pub fn check_conv2d(
        &mut self,
        input: &[DimExpr],
        c_out: &DimExpr,
        c_in: &DimExpr,
        kernel_h: &DimExpr,
        kernel_w: &DimExpr,
        stride: u64,
        padding: u64,
        span: Span,
    ) -> Result<Vec<DimExpr>, ()> {
        if input.len() != 4 {
            self.errors.push(ShapeError {
                message: format!("conv2d expects rank-4 input [N,C,H,W], got rank {}", input.len()),
                span,
                hint: None,
            });
            return Err(());
        }

        // Check C_in matches
        if let Err(msg) = self.unify_dim(&input[1], c_in) {
            self.errors.push(ShapeError {
                message: format!("conv2d: input channels {} vs kernel channels {}: {}", input[1], c_in, msg),
                span,
                hint: None,
            });
            return Err(());
        }

        let out_h = conv_output_dim(&input[2], kernel_h, stride, padding);
        let out_w = conv_output_dim(&input[3], kernel_w, stride, padding);

        Ok(vec![input[0].clone(), c_out.clone(), out_h, out_w])
    }

    // ── AST walking ──

    /// Top-level entry: check shapes for an entire program.
    pub fn check_program(&mut self, program: &Program) {
        // Pass 1: collect function signatures
        for item in &program.items {
            if let ItemKind::Function(func) = &item.kind {
                let params: Vec<ShapeType> = func
                    .params
                    .iter()
                    .map(|p| self.type_expr_to_shape(&p.ty))
                    .collect();
                let ret = func
                    .ret_type
                    .as_ref()
                    .map(|t| self.type_expr_to_shape(t))
                    .unwrap_or(ShapeType::Unknown);
                self.functions
                    .insert(func.name.name.clone(), (params, ret));
            }
        }

        // Pass 2: check each function body
        for item in &program.items {
            match &item.kind {
                ItemKind::Function(func) => self.check_function(func),
                ItemKind::Kernel(k) => self.check_kernel(k),
                _ => {}
            }
        }
    }

    fn check_function(&mut self, func: &Function) {
        self.push_scope();
        for param in &func.params {
            let sty = self.type_expr_to_shape(&param.ty);
            self.define(&param.name.name, sty);
        }
        self.check_block(&func.body);

        // If function has a return type annotation with a tensor shape, check that
        // the body's return shape matches
        if let Some(ret_ty) = &func.ret_type {
            let expected = self.type_expr_to_shape(ret_ty);
            if let Some(body_expr) = &func.body.expr {
                let actual = self.infer_expr(body_expr);
                self.check_shape_compat(&expected, &actual, body_expr.span);
            }
        }
        self.pop_scope();
    }

    fn check_kernel(&mut self, kernel: &Kernel) {
        self.push_scope();
        for param in &kernel.params {
            let sty = self.type_expr_to_shape(&param.ty);
            self.define(&param.name.name, sty);
        }
        self.check_block(&kernel.body);
        self.pop_scope();
    }

    fn check_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        if let Some(expr) = &block.expr {
            let _ = self.infer_expr(expr);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, ty, value } | StmtKind::Var { name, ty, value } => {
                let val_shape = self.infer_expr(value);
                if let Some(ty_expr) = ty {
                    let annotated = self.type_expr_to_shape(ty_expr);
                    self.check_shape_compat(&annotated, &val_shape, stmt.span);
                }
                self.define(&name.name, val_shape);
            }
            StmtKind::Expr(expr) => {
                let _ = self.infer_expr(expr);
            }
            StmtKind::Return(Some(expr)) => {
                let _ = self.infer_expr(expr);
            }
            StmtKind::Assign { target: _, op: _, value } => {
                let _ = self.infer_expr(value);
            }
            StmtKind::For { var, iter, body } => {
                let _ = self.infer_expr(iter);
                self.push_scope();
                self.define(&var.name, ShapeType::Scalar(ScalarKind::I64));
                self.check_block(body);
                self.pop_scope();
            }
            StmtKind::While { cond, body } => {
                let _ = self.infer_expr(cond);
                self.push_scope();
                self.check_block(body);
                self.pop_scope();
            }
            StmtKind::Loop { body } => {
                self.push_scope();
                self.check_block(body);
                self.pop_scope();
            }
            _ => {}
        }
    }

    /// Infer the shape type of an expression.
    pub fn infer_expr(&mut self, expr: &Expr) -> ShapeType {
        match &expr.kind {
            ExprKind::IntLiteral(_) => ShapeType::Scalar(ScalarKind::I64),
            ExprKind::FloatLiteral(_) => ShapeType::Scalar(ScalarKind::F64),
            ExprKind::StringLiteral(_) => ShapeType::Scalar(ScalarKind::String),
            ExprKind::BoolLiteral(_) => ShapeType::Scalar(ScalarKind::Bool),
            ExprKind::Ident(id) => self
                .lookup(&id.name)
                .cloned()
                .unwrap_or(ShapeType::Unknown),

            ExprKind::MatMul { lhs, rhs } => {
                let lt = self.infer_expr(lhs);
                let rt = self.infer_expr(rhs);
                match (&lt, &rt) {
                    (
                        ShapeType::Tensor {
                            elem: le,
                            shape: ls,
                        },
                        ShapeType::Tensor { shape: rs, .. },
                    ) => match self.check_matmul(ls, rs, expr.span) {
                        Ok(result) => ShapeType::Tensor {
                            elem: le.clone(),
                            shape: result,
                        },
                        Err(()) => ShapeType::Unknown,
                    },
                    _ => ShapeType::Unknown,
                }
            }

            ExprKind::Binary { lhs, op, rhs } => {
                let lt = self.infer_expr(lhs);
                let rt = self.infer_expr(rhs);
                match op {
                    BinOp::Add | BinOp::Sub | BinOp::ElemMul | BinOp::ElemDiv => {
                        match (&lt, &rt) {
                            (
                                ShapeType::Tensor {
                                    elem: le,
                                    shape: ls,
                                },
                                ShapeType::Tensor { shape: rs, .. },
                            ) => {
                                if matches!(op, BinOp::ElemMul | BinOp::ElemDiv) {
                                    // Elementwise: exact or broadcast
                                    match self.broadcast(ls, rs, expr.span) {
                                        Ok(result) => ShapeType::Tensor {
                                            elem: le.clone(),
                                            shape: result,
                                        },
                                        Err(()) => ShapeType::Unknown,
                                    }
                                } else {
                                    // Add/Sub: broadcast
                                    match self.broadcast(ls, rs, expr.span) {
                                        Ok(result) => ShapeType::Tensor {
                                            elem: le.clone(),
                                            shape: result,
                                        },
                                        Err(()) => ShapeType::Unknown,
                                    }
                                }
                            }
                            _ => lt, // scalar arithmetic
                        }
                    }
                    BinOp::Mul | BinOp::Div => {
                        // Could be scalar * tensor (scaling)
                        match (&lt, &rt) {
                            (ShapeType::Tensor { .. }, ShapeType::Scalar(_)) => lt,
                            (ShapeType::Scalar(_), ShapeType::Tensor { .. }) => rt,
                            (
                                ShapeType::Tensor {
                                    elem: le,
                                    shape: ls,
                                },
                                ShapeType::Tensor { shape: rs, .. },
                            ) => match self.broadcast(ls, rs, expr.span) {
                                Ok(result) => ShapeType::Tensor {
                                    elem: le.clone(),
                                    shape: result,
                                },
                                Err(()) => ShapeType::Unknown,
                            },
                            _ => lt,
                        }
                    }
                    _ => ShapeType::Unknown,
                }
            }

            ExprKind::Call { func, args } => {
                self.infer_call(func, args, expr.span)
            }

            ExprKind::ArrayLiteral(elems) => {
                if elems.is_empty() {
                    return ShapeType::Unknown;
                }
                let first = self.infer_expr(&elems[0]);
                // If elements are tensors, we get a higher-rank tensor
                match &first {
                    ShapeType::Tensor { elem, shape } => {
                        let mut new_shape = vec![DimExpr::Lit(elems.len() as u64)];
                        new_shape.extend(shape.iter().cloned());
                        ShapeType::Tensor {
                            elem: elem.clone(),
                            shape: new_shape,
                        }
                    }
                    _ => ShapeType::Unknown,
                }
            }

            ExprKind::Index { base, indices } => {
                let bt = self.infer_expr(base);
                // Indexing reduces rank
                match bt {
                    ShapeType::Tensor { elem, shape } => {
                        if indices.len() >= shape.len() {
                            ShapeType::Scalar(elem)
                        } else {
                            ShapeType::Tensor {
                                elem,
                                shape: shape[indices.len()..].to_vec(),
                            }
                        }
                    }
                    _ => ShapeType::Unknown,
                }
            }

            ExprKind::Block(block) => {
                self.push_scope();
                for s in &block.stmts {
                    self.check_stmt(s);
                }
                let result = block
                    .expr
                    .as_ref()
                    .map(|e| self.infer_expr(e))
                    .unwrap_or(ShapeType::Unknown);
                self.pop_scope();
                result
            }

            ExprKind::If {
                cond,
                then_block,
                else_block: _,
            } => {
                let _ = self.infer_expr(cond);
                self.push_scope();
                self.check_block(then_block);
                let result = then_block
                    .expr
                    .as_ref()
                    .map(|e| self.infer_expr(e))
                    .unwrap_or(ShapeType::Unknown);
                self.pop_scope();
                result
            }

            ExprKind::FieldAccess { base, .. } => {
                let _ = self.infer_expr(base);
                ShapeType::Unknown
            }

            ExprKind::Cast { expr: inner, .. } => self.infer_expr(inner),
            ExprKind::Unary { expr: inner, .. } => self.infer_expr(inner),

            _ => ShapeType::Unknown,
        }
    }

    /// Infer shape for a function/method call.
    fn infer_call(&mut self, func_expr: &Expr, args: &[Expr], span: Span) -> ShapeType {
        let arg_shapes: Vec<ShapeType> = args.iter().map(|a| self.infer_expr(a)).collect();

        // Check if it's a named function
        let func_name = match &func_expr.kind {
            ExprKind::Ident(id) => Some(id.name.clone()),
            ExprKind::FieldAccess { base: _, field } => Some(field.name.clone()),
            _ => None,
        };

        if let Some(name) = &func_name {
            match name.as_str() {
                "transpose" => {
                    if let Some(ShapeType::Tensor { elem, shape }) = arg_shapes.first() {
                        match self.check_transpose(shape) {
                            Ok(result) => {
                                return ShapeType::Tensor {
                                    elem: elem.clone(),
                                    shape: result,
                                };
                            }
                            Err(msg) => {
                                self.errors.push(ShapeError {
                                    message: msg,
                                    span,
                                    hint: None,
                                });
                                return ShapeType::Unknown;
                            }
                        }
                    }
                }

                "reshape" => {
                    if arg_shapes.len() >= 2 {
                        if let ShapeType::Tensor { elem, shape: in_shape } = &arg_shapes[0] {
                            // Second arg should be an array literal giving target shape
                            if let ExprKind::ArrayLiteral(dims) = &args[1].kind {
                                let target: Vec<DimExpr> = dims
                                    .iter()
                                    .map(|d| self.expr_to_dim(d))
                                    .collect();
                                match self.check_reshape(in_shape, &target, span) {
                                    Ok(result) => {
                                        return ShapeType::Tensor {
                                            elem: elem.clone(),
                                            shape: result,
                                        };
                                    }
                                    Err(()) => return ShapeType::Unknown,
                                }
                            }
                        }
                    }
                }

                "concat" => {
                    if arg_shapes.len() >= 3 {
                        if let (
                            ShapeType::Tensor { elem, shape: sa },
                            ShapeType::Tensor { shape: sb, .. },
                        ) = (&arg_shapes[0], &arg_shapes[1])
                        {
                            // Third arg is axis
                            let axis = match &args[2].kind {
                                ExprKind::IntLiteral(n) => *n as usize,
                                _ => 0,
                            };
                            match self.check_concat(sa, sb, axis, span) {
                                Ok(result) => {
                                    return ShapeType::Tensor {
                                        elem: elem.clone(),
                                        shape: result,
                                    };
                                }
                                Err(()) => return ShapeType::Unknown,
                            }
                        }
                    }
                }

                "nn_linear" => {
                    // nn_linear(input, in_features, out_features)
                    if arg_shapes.len() >= 1 && args.len() >= 3 {
                        if let ShapeType::Tensor { elem, shape } = &arg_shapes[0] {
                            let in_f = self.expr_to_dim(&args[1]);
                            let out_f = self.expr_to_dim(&args[2]);
                            match self.check_nn_linear(shape, &in_f, &out_f, span) {
                                Ok(result) => {
                                    return ShapeType::Tensor {
                                        elem: elem.clone(),
                                        shape: result,
                                    };
                                }
                                Err(()) => return ShapeType::Unknown,
                            }
                        }
                    }
                }

                "softmax" => {
                    if let Some(ShapeType::Tensor { elem, shape }) = arg_shapes.first() {
                        let axis = if args.len() >= 2 {
                            match &args[1].kind {
                                ExprKind::IntLiteral(n) => *n as usize,
                                _ => shape.len() - 1,
                            }
                        } else {
                            shape.len().saturating_sub(1)
                        };
                        if axis >= shape.len() {
                            self.errors.push(ShapeError {
                                message: format!(
                                    "softmax axis {} out of range for rank {}",
                                    axis,
                                    shape.len()
                                ),
                                span,
                                hint: None,
                            });
                            return ShapeType::Unknown;
                        }
                        return ShapeType::Tensor {
                            elem: elem.clone(),
                            shape: shape.clone(),
                        };
                    }
                }

                "layer_norm" => {
                    if arg_shapes.len() >= 2 {
                        if let (
                            ShapeType::Tensor { elem, shape: xs },
                            ShapeType::Tensor { shape: gs, .. },
                        ) = (&arg_shapes[0], &arg_shapes[1])
                        {
                            if gs.len() == 1 && !xs.is_empty() {
                                if let Err(msg) =
                                    self.unify_dim(&xs[xs.len() - 1], &gs[0])
                                {
                                    self.errors.push(ShapeError {
                                        message: format!(
                                            "layer_norm: last dim vs gamma: {}",
                                            msg
                                        ),
                                        span,
                                        hint: None,
                                    });
                                }
                            }
                            return ShapeType::Tensor {
                                elem: elem.clone(),
                                shape: xs.clone(),
                            };
                        }
                    }
                }

                _ => {}
            }

            // Look up user-defined function signature
            if let Some((param_shapes, ret_shape)) = self.functions.get(name).cloned() {
                // Unify argument shapes with parameter shapes
                for (i, (ps, as_)) in param_shapes.iter().zip(arg_shapes.iter()).enumerate() {
                    self.check_shape_compat_detail(ps, as_, span, &format!("argument {} of '{}'", i, name));
                }
                return ret_shape;
            }
        }

        ShapeType::Unknown
    }

    /// Check that actual shape is compatible with expected.
    fn check_shape_compat(&mut self, expected: &ShapeType, actual: &ShapeType, span: Span) {
        self.check_shape_compat_detail(expected, actual, span, "type annotation");
    }

    fn check_shape_compat_detail(
        &mut self,
        expected: &ShapeType,
        actual: &ShapeType,
        span: Span,
        context: &str,
    ) {
        match (expected, actual) {
            (
                ShapeType::Tensor {
                    shape: es, ..
                },
                ShapeType::Tensor {
                    shape: as_, ..
                },
            ) => {
                let _ = self.unify_shapes(es, as_, span, context);
            }
            (ShapeType::Unknown, _) | (_, ShapeType::Unknown) => {}
            _ => {}
        }
    }

    // ── helpers ──

    /// Convert a type expression from the AST into a ShapeType.
    fn type_expr_to_shape(&self, ty: &TypeExpr) -> ShapeType {
        match &ty.kind {
            TypeExprKind::Named(ident) => match ident.name.as_str() {
                "i64" | "i32" | "i16" | "i8" => ShapeType::Scalar(ScalarKind::I64),
                "f64" => ShapeType::Scalar(ScalarKind::F64),
                "f32" => ShapeType::Scalar(ScalarKind::F32),
                "f16" => ShapeType::Scalar(ScalarKind::F16),
                "bool" => ShapeType::Scalar(ScalarKind::Bool),
                "String" | "str" => ShapeType::Scalar(ScalarKind::String),
                _ => ShapeType::Unknown,
            },
            TypeExprKind::Generic { name, args } if name.name == "Tensor" => {
                let elem = args.first().map(|a| match a {
                    TypeArg::Type(te) => match &te.kind {
                        TypeExprKind::Named(id) => match id.name.as_str() {
                            "f32" => ScalarKind::F32,
                            "f64" => ScalarKind::F64,
                            "f16" => ScalarKind::F16,
                            "i64" => ScalarKind::I64,
                            "bool" => ScalarKind::Bool,
                            other => ScalarKind::Other(other.to_string()),
                        },
                        _ => ScalarKind::F64,
                    },
                    _ => ScalarKind::F64,
                }).unwrap_or(ScalarKind::F64);

                let shape = args.iter().find_map(|a| match a {
                    TypeArg::Shape(dims) => Some(
                        dims.iter()
                            .map(|d| self.shape_dim_to_dim_expr(d))
                            .collect::<Vec<_>>(),
                    ),
                    _ => None,
                }).unwrap_or_default();

                ShapeType::Tensor { elem, shape }
            }
            TypeExprKind::Fn { params, ret } => ShapeType::Function {
                params: params.iter().map(|p| self.type_expr_to_shape(p)).collect(),
                ret: Box::new(self.type_expr_to_shape(ret)),
            },
            _ => ShapeType::Unknown,
        }
    }

    fn shape_dim_to_dim_expr(&self, sd: &ShapeDim) -> DimExpr {
        match sd {
            ShapeDim::Lit(n) => DimExpr::Lit(*n),
            ShapeDim::Ident(id) => DimExpr::Var(id.name.clone()),
            ShapeDim::Dynamic => DimExpr::Any,
            ShapeDim::Expr(e) => self.expr_to_dim(e),
        }
    }

    fn expr_to_dim(&self, expr: &Expr) -> DimExpr {
        match &expr.kind {
            ExprKind::IntLiteral(n) => DimExpr::Lit(*n as u64),
            ExprKind::Ident(id) => DimExpr::Var(id.name.clone()),
            ExprKind::Binary { lhs, op, rhs } => {
                let dl = self.expr_to_dim(lhs);
                let dr = self.expr_to_dim(rhs);
                let dim_op = match op {
                    BinOp::Add => DimOp::Add,
                    BinOp::Sub => DimOp::Sub,
                    BinOp::Mul => DimOp::Mul,
                    BinOp::Div => DimOp::Div,
                    _ => return DimExpr::Any,
                };
                DimExpr::BinOp(Box::new(dl), dim_op, Box::new(dr))
            }
            _ => DimExpr::Any,
        }
    }
}

// ── Free helpers ──

fn concrete_product(dims: &[DimExpr]) -> Option<u64> {
    let mut prod = 1u64;
    for d in dims {
        match d {
            DimExpr::Lit(n) => prod *= n,
            _ => return None,
        }
    }
    Some(prod)
}

fn conv_output_dim(input: &DimExpr, kernel: &DimExpr, stride: u64, padding: u64) -> DimExpr {
    match (input, kernel) {
        (DimExpr::Lit(h), DimExpr::Lit(k)) => {
            DimExpr::Lit((h + 2 * padding - k) / stride + 1)
        }
        _ => DimExpr::Any, // symbolic — can't compute
    }
}

// ── Legacy compatibility (used by typeck.rs) ──
// The old ShapeChecker struct with static methods, kept for backward compat.

/// Legacy shape error (used by old code paths).
#[derive(Debug, Clone)]
pub enum LegacyShapeError {
    DimMismatch {
        expected: Dim,
        got: Dim,
        context: std::string::String,
    },
    RankMismatch {
        expected: usize,
        got: usize,
        context: std::string::String,
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

impl fmt::Display for LegacyShapeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LegacyShapeError::DimMismatch { expected, got, context } => {
                write!(f, "shape dimension mismatch in {}: expected `{}`, got `{}`",
                    context, legacy_dim_display(expected), legacy_dim_display(got))
            }
            LegacyShapeError::RankMismatch { expected, got, context } => {
                write!(f, "rank mismatch in {}: expected {}, got {}", context, expected, got)
            }
            LegacyShapeError::ProductMismatch { from, to } => {
                write!(f, "reshape product mismatch: {} vs {}",
                    legacy_dims_display(from), legacy_dims_display(to))
            }
            LegacyShapeError::InvalidAxis { axis, rank } => {
                write!(f, "invalid axis {} for tensor of rank {}", axis, rank)
            }
        }
    }
}

fn legacy_dim_display(d: &Dim) -> std::string::String {
    match d {
        Dim::Lit(n) => n.to_string(),
        Dim::Sym(s) => s.clone(),
        Dim::Dynamic => "?".to_string(),
    }
}

fn legacy_dims_display(dims: &[Dim]) -> std::string::String {
    let parts: Vec<std::string::String> = dims.iter().map(legacy_dim_display).collect();
    format!("[{}]", parts.join(", "))
}

/// Legacy static shape checker (used by typeck.rs).
pub struct LegacyShapeChecker;

impl LegacyShapeChecker {
    pub fn dims_equal(a: &Dim, b: &Dim) -> bool {
        match (a, b) {
            (Dim::Lit(x), Dim::Lit(y)) => x == y,
            (Dim::Sym(x), Dim::Sym(y)) => x == y,
            (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
            _ => false,
        }
    }

    pub fn check_matmul(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        if lhs.len() < 2 {
            return Err(LegacyShapeError::RankMismatch {
                expected: 2, got: lhs.len(), context: "matmul lhs".to_string(),
            });
        }
        if rhs.len() < 2 {
            return Err(LegacyShapeError::RankMismatch {
                expected: 2, got: rhs.len(), context: "matmul rhs".to_string(),
            });
        }
        let lhs_k = &lhs[lhs.len() - 1];
        let rhs_k = &rhs[rhs.len() - 2];
        if !Self::dims_equal(lhs_k, rhs_k) {
            return Err(LegacyShapeError::DimMismatch {
                expected: lhs_k.clone(), got: rhs_k.clone(),
                context: "matmul inner dimensions".to_string(),
            });
        }
        let m = &lhs[lhs.len() - 2];
        let n = &rhs[rhs.len() - 1];
        let lhs_batch = &lhs[..lhs.len() - 2];
        let rhs_batch = &rhs[..rhs.len() - 2];
        let batch = Self::broadcast_batch(lhs_batch, rhs_batch)?;
        let mut result = batch;
        result.push(m.clone());
        result.push(n.clone());
        Ok(result)
    }

    fn broadcast_batch(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        if lhs.is_empty() { return Ok(rhs.to_vec()); }
        if rhs.is_empty() { return Ok(lhs.to_vec()); }
        let max_len = lhs.len().max(rhs.len());
        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let l_idx = if i < max_len - lhs.len() { None } else { Some(i - (max_len - lhs.len())) };
            let r_idx = if i < max_len - rhs.len() { None } else { Some(i - (max_len - rhs.len())) };
            match (l_idx.map(|j| &lhs[j]), r_idx.map(|j| &rhs[j])) {
                (Some(l), Some(r)) => {
                    if Self::dims_equal(l, r) { result.push(l.clone()); }
                    else if matches!(l, Dim::Lit(1)) { result.push(r.clone()); }
                    else if matches!(r, Dim::Lit(1)) { result.push(l.clone()); }
                    else {
                        return Err(LegacyShapeError::DimMismatch {
                            expected: l.clone(), got: r.clone(),
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

    pub fn check_broadcast(lhs: &[Dim], rhs: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        let max_len = lhs.len().max(rhs.len());
        let mut result = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let l_idx = if i < max_len - lhs.len() { None } else { Some(i - (max_len - lhs.len())) };
            let r_idx = if i < max_len - rhs.len() { None } else { Some(i - (max_len - rhs.len())) };
            match (l_idx.map(|j| &lhs[j]), r_idx.map(|j| &rhs[j])) {
                (Some(l), Some(r)) => {
                    if Self::dims_equal(l, r) { result.push(l.clone()); }
                    else if matches!(l, Dim::Lit(1)) { result.push(r.clone()); }
                    else if matches!(r, Dim::Lit(1)) { result.push(l.clone()); }
                    else {
                        return Err(LegacyShapeError::DimMismatch {
                            expected: l.clone(), got: r.clone(),
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

    pub fn check_softmax(input: &[Dim], axis: usize) -> Result<Vec<Dim>, LegacyShapeError> {
        if axis >= input.len() {
            return Err(LegacyShapeError::InvalidAxis { axis, rank: input.len() });
        }
        Ok(input.to_vec())
    }

    pub fn check_layer_norm(x: &[Dim], gamma: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        if gamma.len() != 1 {
            return Err(LegacyShapeError::RankMismatch { expected: 1, got: gamma.len(), context: "layer_norm gamma".to_string() });
        }
        if x.is_empty() {
            return Err(LegacyShapeError::RankMismatch { expected: 1, got: 0, context: "layer_norm input".to_string() });
        }
        let last = &x[x.len() - 1];
        if !Self::dims_equal(last, &gamma[0]) {
            return Err(LegacyShapeError::DimMismatch { expected: last.clone(), got: gamma[0].clone(), context: "layer_norm last dim vs gamma".to_string() });
        }
        Ok(x.to_vec())
    }

    pub fn check_transpose(input: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        if input.len() < 2 {
            return Err(LegacyShapeError::RankMismatch { expected: 2, got: input.len(), context: "transpose".to_string() });
        }
        let mut result = input.to_vec();
        let n = result.len();
        result.swap(n - 2, n - 1);
        Ok(result)
    }

    pub fn check_reshape(input: &[Dim], target: &[Dim]) -> Result<Vec<Dim>, LegacyShapeError> {
        let input_prod = Self::concrete_product(input);
        let target_prod = Self::concrete_product(target);
        match (input_prod, target_prod) {
            (Some(a), Some(b)) if a != b => {
                Err(LegacyShapeError::ProductMismatch { from: input.to_vec(), to: target.to_vec() })
            }
            _ => Ok(target.to_vec()),
        }
    }

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

    pub fn simplify(expr: &crate::shape_check::ShapeExpr) -> Dim {
        match expr {
            crate::shape_check::ShapeExpr::Dim(d) => d.clone(),
            crate::shape_check::ShapeExpr::Add(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x + y),
                    _ => Dim::Dynamic,
                }
            }
            crate::shape_check::ShapeExpr::Mul(a, b) => {
                let a = Self::simplify(a);
                let b = Self::simplify(b);
                match (&a, &b) {
                    (Dim::Lit(x), Dim::Lit(y)) => Dim::Lit(x * y),
                    (Dim::Lit(1), other) | (other, Dim::Lit(1)) => other.clone(),
                    (Dim::Lit(0), _) | (_, Dim::Lit(0)) => Dim::Lit(0),
                    _ => Dim::Dynamic,
                }
            }
            crate::shape_check::ShapeExpr::Div(a, b) => {
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

// ── Public entry point ──

/// Run shape checking on a parsed program.  Returns Ok(()) if no shape errors,
/// or Err with the list of errors found.
pub fn check_shapes(program: &Program) -> Result<(), Vec<ShapeError>> {
    let mut checker = ShapeChecker::new();
    checker.check_program(program);
    if checker.errors.is_empty() {
        Ok(())
    } else {
        Err(checker.errors)
    }
}

// ───────────────────────────── Tests ──────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn var(s: &str) -> DimExpr {
        DimExpr::Var(s.to_string())
    }
    fn lit(n: u64) -> DimExpr {
        DimExpr::Lit(n)
    }
    fn span() -> Span {
        Span::new(0, 0)
    }

    // Legacy compat tests using Dim
    fn sym(s: &str) -> Dim { Dim::Sym(s.to_string()) }
    fn dlit(n: u64) -> Dim { Dim::Lit(n) }

    // ── Test 1: valid matmul [M, K] @ [K, N] → [M, N] ──
    #[test]
    fn test_matmul_valid_2d() {
        let mut c = ShapeChecker::new();
        let lhs = vec![var("M"), var("K")];
        let rhs = vec![var("K"), var("N")];
        let result = c.check_matmul(&lhs, &rhs, span()).unwrap();
        assert_eq!(result, vec![var("M"), var("N")]);
        assert!(c.errors.is_empty());
    }

    // ── Test 2: invalid matmul inner dim mismatch ──
    #[test]
    fn test_matmul_invalid_inner() {
        let mut c = ShapeChecker::new();
        let lhs = vec![lit(4), lit(768)];
        let rhs = vec![lit(512), lit(3072)];
        assert!(c.check_matmul(&lhs, &rhs, span()).is_err());
        assert!(!c.errors.is_empty());
        assert!(c.errors[0].message.contains("matrix multiply"));
    }

    // ── Test 3: broadcasting [B,T,D] + [D] → [B,T,D] ──
    #[test]
    fn test_broadcast_trailing() {
        let mut c = ShapeChecker::new();
        let lhs = vec![var("B"), var("T"), var("D")];
        let rhs = vec![var("D")];
        let result = c.broadcast(&lhs, &rhs, span()).unwrap();
        assert_eq!(result.len(), 3);
    }

    // ── Test 4: broadcast [M,1] + [1,N] → [M,N] ──
    #[test]
    fn test_broadcast_expand() {
        let mut c = ShapeChecker::new();
        let lhs = vec![var("M"), lit(1)];
        let rhs = vec![lit(1), var("N")];
        let result = c.broadcast(&lhs, &rhs, span()).unwrap();
        assert_eq!(result, vec![var("M"), var("N")]);
    }

    // ── Test 5: symbolic unification ──
    #[test]
    fn test_unify_var_lit() {
        let mut c = ShapeChecker::new();
        assert!(c.unify_dim(&var("N"), &lit(768)).is_ok());
        assert_eq!(c.resolve_dim(&var("N")), lit(768));
    }

    // ── Test 6: concrete unify mismatch ──
    #[test]
    fn test_unify_lit_mismatch() {
        let mut c = ShapeChecker::new();
        assert!(c.unify_dim(&lit(768), &lit(512)).is_err());
    }

    // ── Test 7: batched matmul [B,T,D] @ [D,H] → [B,T,H] ──
    #[test]
    fn test_matmul_batched() {
        let mut c = ShapeChecker::new();
        let lhs = vec![var("B"), var("T"), var("D")];
        let rhs = vec![var("D"), var("H")];
        let result = c.check_matmul(&lhs, &rhs, span()).unwrap();
        assert_eq!(result, vec![var("B"), var("T"), var("H")]);
    }

    // ── Test 8: transpose [B,T,D] → [B,D,T] ──
    #[test]
    fn test_transpose() {
        let c = ShapeChecker::new();
        let input = vec![var("B"), var("T"), var("D")];
        let result = c.check_transpose(&input).unwrap();
        assert_eq!(result, vec![var("B"), var("D"), var("T")]);
    }

    // ── Test 9: reshape valid ──
    #[test]
    fn test_reshape_valid() {
        let mut c = ShapeChecker::new();
        let input = vec![lit(4), lit(8)];
        let target = vec![lit(2), lit(16)];
        let result = c.check_reshape(&input, &target, span()).unwrap();
        assert_eq!(result, vec![lit(2), lit(16)]);
    }

    // ── Test 10: reshape product mismatch ──
    #[test]
    fn test_reshape_mismatch() {
        let mut c = ShapeChecker::new();
        let input = vec![lit(4), lit(8)];
        let target = vec![lit(3), lit(16)];
        assert!(c.check_reshape(&input, &target, span()).is_err());
    }

    // ── Test 11: concat along axis ──
    #[test]
    fn test_concat_valid() {
        let mut c = ShapeChecker::new();
        let a = vec![lit(4), lit(10)];
        let b = vec![lit(4), lit(20)];
        let result = c.check_concat(&a, &b, 1, span()).unwrap();
        assert_eq!(result, vec![lit(4), lit(30)]);
    }

    // ── Test 12: concat rank mismatch ──
    #[test]
    fn test_concat_rank_mismatch() {
        let mut c = ShapeChecker::new();
        let a = vec![lit(4), lit(10)];
        let b = vec![lit(4), lit(10), lit(5)];
        assert!(c.check_concat(&a, &b, 0, span()).is_err());
    }

    // ── Test 13: nn_linear ──
    #[test]
    fn test_nn_linear_valid() {
        let mut c = ShapeChecker::new();
        let input = vec![var("batch"), lit(768)];
        let result = c
            .check_nn_linear(&input, &lit(768), &lit(3072), span())
            .unwrap();
        assert_eq!(result, vec![var("batch"), lit(3072)]);
    }

    // ── Test 14: nn_linear input dim mismatch ──
    #[test]
    fn test_nn_linear_mismatch() {
        let mut c = ShapeChecker::new();
        let input = vec![var("batch"), lit(512)];
        assert!(c.check_nn_linear(&input, &lit(768), &lit(3072), span()).is_err());
    }

    // ── Test 15: conv2d output shape ──
    #[test]
    fn test_conv2d_output() {
        let mut c = ShapeChecker::new();
        // input [1, 3, 32, 32], kernel 3x3, stride 1, padding 0
        let input = vec![lit(1), lit(3), lit(32), lit(32)];
        let result = c
            .check_conv2d(&input, &lit(64), &lit(3), &lit(3), &lit(3), 1, 0, span())
            .unwrap();
        // output: [1, 64, 30, 30]
        assert_eq!(result, vec![lit(1), lit(64), lit(30), lit(30)]);
    }

    // ── Test 16: conv2d channel mismatch ──
    #[test]
    fn test_conv2d_channel_mismatch() {
        let mut c = ShapeChecker::new();
        let input = vec![lit(1), lit(3), lit(32), lit(32)];
        assert!(c
            .check_conv2d(&input, &lit(64), &lit(16), &lit(3), &lit(3), 1, 0, span())
            .is_err());
    }

    // ── Test 17: shape through multiple operations ──
    #[test]
    fn test_multi_op_pipeline() {
        let mut c = ShapeChecker::new();
        // [B, 768] @ [768, 3072] → [B, 3072]
        let h = c
            .check_matmul(
                &[var("B"), lit(768)],
                &[lit(768), lit(3072)],
                span(),
            )
            .unwrap();
        assert_eq!(h, vec![var("B"), lit(3072)]);
        // [B, 3072] @ [3072, 768] → [B, 768]
        let out = c
            .check_matmul(&h, &[lit(3072), lit(768)], span())
            .unwrap();
        assert_eq!(out, vec![var("B"), lit(768)]);
    }

    // ── Test 18: wildcard dimension ──
    #[test]
    fn test_wildcard_unify() {
        let mut c = ShapeChecker::new();
        assert!(c.unify_dim(&DimExpr::Any, &lit(768)).is_ok());
        assert!(c.unify_dim(&lit(768), &DimExpr::Any).is_ok());
    }

    // ── Test 19: dim binop resolution ──
    #[test]
    fn test_dim_binop_resolve() {
        let c = ShapeChecker::new();
        let expr = DimExpr::BinOp(Box::new(lit(4)), DimOp::Mul, Box::new(lit(8)));
        assert_eq!(c.resolve_dim(&expr), lit(32));
    }

    // ── Test 20: error message quality ──
    #[test]
    fn test_error_message_has_hint() {
        let mut c = ShapeChecker::new();
        let _ = c.check_matmul(&[lit(4), lit(768)], &[lit(512), lit(3072)], span());
        assert!(!c.errors.is_empty());
        let err = &c.errors[0];
        assert!(err.hint.is_some());
        assert!(err.hint.as_ref().unwrap().contains("768"));
    }

    // ── Test 21: conv2d with padding ──
    #[test]
    fn test_conv2d_with_padding() {
        let mut c = ShapeChecker::new();
        // input [1, 3, 32, 32], kernel 3x3, stride 1, padding 1 → [1, 64, 32, 32]
        let result = c
            .check_conv2d(&input_nchw(), &lit(64), &lit(3), &lit(3), &lit(3), 1, 1, span())
            .unwrap();
        assert_eq!(result, vec![lit(1), lit(64), lit(32), lit(32)]);
    }

    // ── Test 22: broadcast same shape ──
    #[test]
    fn test_broadcast_same() {
        let mut c = ShapeChecker::new();
        let s = vec![lit(4), lit(8)];
        let result = c.broadcast(&s, &s, span()).unwrap();
        assert_eq!(result, vec![lit(4), lit(8)]);
    }

    // ── Test 23: broadcast mismatch ──
    #[test]
    fn test_broadcast_mismatch() {
        let mut c = ShapeChecker::new();
        let a = vec![lit(4), lit(3)];
        let b = vec![lit(4), lit(5)];
        assert!(c.broadcast(&a, &b, span()).is_err());
    }

    // ── Test 24: concat non-concat dims must match ──
    #[test]
    fn test_concat_non_axis_mismatch() {
        let mut c = ShapeChecker::new();
        let a = vec![lit(4), lit(10)];
        let b = vec![lit(5), lit(20)];
        assert!(c.check_concat(&a, &b, 1, span()).is_err());
    }

    // ── Test 25: function return shape matches annotation ──
    #[test]
    fn test_unify_shapes_match() {
        let mut c = ShapeChecker::new();
        let expected = vec![var("B"), lit(768)];
        let actual = vec![var("B"), lit(768)];
        assert!(c.unify_shapes(&expected, &actual, span(), "return").is_ok());
    }

    // ── Legacy compat tests ──

    #[test]
    fn test_legacy_matmul_batched_2d() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("D"), sym("H")];
        let result = LegacyShapeChecker::check_matmul(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("H")]);
    }

    #[test]
    fn test_legacy_matmul_inner_dim_mismatch() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("H"), sym("D")];
        let err = LegacyShapeChecker::check_matmul(&lhs, &rhs).unwrap_err();
        assert!(matches!(err, LegacyShapeError::DimMismatch { .. }));
    }

    #[test]
    fn test_legacy_broadcast_trailing() {
        let lhs = vec![sym("B"), sym("T"), sym("D")];
        let rhs = vec![sym("D")];
        let result = LegacyShapeChecker::check_broadcast(&lhs, &rhs).unwrap();
        assert_eq!(result, vec![sym("B"), sym("T"), sym("D")]);
    }

    #[test]
    fn test_legacy_transpose() {
        let input = vec![sym("B"), sym("T"), sym("D")];
        let result = LegacyShapeChecker::check_transpose(&input).unwrap();
        assert_eq!(result, vec![sym("B"), sym("D"), sym("T")]);
    }

    #[test]
    fn test_legacy_reshape_ok() {
        let input = vec![dlit(4), dlit(8)];
        let target = vec![dlit(2), dlit(16)];
        let result = LegacyShapeChecker::check_reshape(&input, &target).unwrap();
        assert_eq!(result, vec![dlit(2), dlit(16)]);
    }

    #[test]
    fn test_legacy_reshape_mismatch() {
        let input = vec![dlit(4), dlit(8)];
        let target = vec![dlit(3), dlit(16)];
        let err = LegacyShapeChecker::check_reshape(&input, &target).unwrap_err();
        assert!(matches!(err, LegacyShapeError::ProductMismatch { .. }));
    }

    #[test]
    fn test_legacy_simplify() {
        let expr1 = crate::shape_check::ShapeExpr::Mul(
            Box::new(crate::shape_check::ShapeExpr::Dim(sym("D"))),
            Box::new(crate::shape_check::ShapeExpr::Dim(dlit(1))),
        );
        assert_eq!(LegacyShapeChecker::simplify(&expr1), sym("D"));

        let expr2 = crate::shape_check::ShapeExpr::Mul(
            Box::new(crate::shape_check::ShapeExpr::Dim(dlit(4))),
            Box::new(crate::shape_check::ShapeExpr::Dim(dlit(8))),
        );
        assert_eq!(LegacyShapeChecker::simplify(&expr2), dlit(32));
    }

    // Helper
    fn input_nchw() -> Vec<DimExpr> {
        vec![lit(1), lit(3), lit(32), lit(32)]
    }
}
