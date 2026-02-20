use crate::ast::*;
use crate::lexer::Span;
use codespan_reporting::diagnostic::{Diagnostic, Label};
use std::collections::HashMap;

type FileId = usize;

/// The resolved type of an expression
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// Scalar types: u8, u16, u32, u64, u128, i8, i16, i32, i64, f16, f32, f64, bool
    Scalar(ScalarType),
    /// Tensor type with element type and shape
    Tensor {
        elem: Box<Type>,
        shape: Vec<Dim>,
    },
    /// Array type: [T; N]
    Array {
        elem: Box<Type>,
        size: u64,
    },
    /// A named type that we haven't resolved yet (for forward references, user types)
    Named(String),
    /// Function type
    Fn {
        params: Vec<Type>,
        ret: Box<Type>,
    },
    /// Void (no return value)
    Void,
    /// Type variable (for inference)
    Var(usize),
    /// Error type (for recovery)
    Error,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ScalarType {
    U8, U16, U32, U64, U128,
    I8, I16, I32, I64,
    F16, F32, F64,
    Bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Dim {
    Lit(u64),
    Sym(String),
    Dynamic,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Type::Scalar(s) => write!(f, "{}", s),
            Type::Tensor { elem, shape } => {
                write!(f, "Tensor<{}, [", elem)?;
                for (i, d) in shape.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    match d {
                        Dim::Lit(n) => write!(f, "{}", n)?,
                        Dim::Sym(s) => write!(f, "{}", s)?,
                        Dim::Dynamic => write!(f, "?")?,
                    }
                }
                write!(f, "]>")
            }
            Type::Array { elem, size } => write!(f, "[{}; {}]", elem, size),
            Type::Named(name) => write!(f, "{}", name),
            Type::Fn { params, ret } => {
                write!(f, "fn(")?;
                for (i, p) in params.iter().enumerate() {
                    if i > 0 { write!(f, ", ")?; }
                    write!(f, "{}", p)?;
                }
                write!(f, ") -> {}", ret)
            }
            Type::Void => write!(f, "void"),
            Type::Var(n) => write!(f, "?{}", n),
            Type::Error => write!(f, "<error>"),
        }
    }
}

impl std::fmt::Display for ScalarType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ScalarType::U8 => write!(f, "u8"),
            ScalarType::U16 => write!(f, "u16"),
            ScalarType::U32 => write!(f, "u32"),
            ScalarType::U64 => write!(f, "u64"),
            ScalarType::U128 => write!(f, "u128"),
            ScalarType::I8 => write!(f, "i8"),
            ScalarType::I16 => write!(f, "i16"),
            ScalarType::I32 => write!(f, "i32"),
            ScalarType::I64 => write!(f, "i64"),
            ScalarType::F16 => write!(f, "f16"),
            ScalarType::F32 => write!(f, "f32"),
            ScalarType::F64 => write!(f, "f64"),
            ScalarType::Bool => write!(f, "bool"),
        }
    }
}

/// Environment for type checking
struct TypeEnv {
    /// Stack of scopes, each mapping names to types
    scopes: Vec<HashMap<String, Type>>,
    /// Function signatures: name -> (param types, return type)
    functions: HashMap<String, (Vec<Type>, Type)>,
    /// Type definitions: name -> Type
    type_defs: HashMap<String, Type>,
    /// Generic type parameters in scope
    type_params: HashMap<String, Vec<String>>, // name -> trait bounds
    /// Errors collected during type checking
    errors: Vec<Diagnostic<FileId>>,
    file_id: FileId,
    /// Counter for type variables
    next_var: usize,
}

impl TypeEnv {
    fn new(file_id: FileId) -> Self {
        Self {
            scopes: vec![HashMap::new()],
            functions: HashMap::new(),
            type_defs: HashMap::new(),
            type_params: HashMap::new(),
            errors: Vec::new(),
            file_id,
            next_var: 0,
        }
    }

    fn push_scope(&mut self) {
        self.scopes.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.scopes.pop();
    }

    fn define(&mut self, name: &str, ty: Type) {
        if let Some(scope) = self.scopes.last_mut() {
            scope.insert(name.to_string(), ty);
        }
    }

    fn lookup(&self, name: &str) -> Option<&Type> {
        for scope in self.scopes.iter().rev() {
            if let Some(ty) = scope.get(name) {
                return Some(ty);
            }
        }
        None
    }

    fn fresh_var(&mut self) -> Type {
        let v = self.next_var;
        self.next_var += 1;
        Type::Var(v)
    }

    fn error(&mut self, span: Span, msg: String) {
        self.errors.push(
            Diagnostic::error()
                .with_message(&msg)
                .with_labels(vec![Label::primary(self.file_id, span.start..span.end)]),
        );
    }

    fn type_mismatch(&mut self, span: Span, expected: &Type, got: &Type) {
        self.error(span, format!("type mismatch: expected `{}`, found `{}`", expected, got));
    }
}

/// Main entry point for type checking
pub fn check(program: &Program, file_id: FileId) -> Result<(), Vec<Diagnostic<FileId>>> {
    let mut env = TypeEnv::new(file_id);

    // First pass: collect all function/kernel signatures
    for item in &program.items {
        collect_item_signature(&mut env, item);
    }

    // Second pass: type check bodies
    for item in &program.items {
        check_item(&mut env, item);
    }

    if env.errors.is_empty() {
        Ok(())
    } else {
        Err(env.errors)
    }
}

fn collect_item_signature(env: &mut TypeEnv, item: &Item) {
    match &item.kind {
        ItemKind::Function(func) => {
            let param_types: Vec<Type> = func.params.iter()
                .map(|p| resolve_type_expr(env, &p.ty))
                .collect();
            let ret_type = func.ret_type.as_ref()
                .map(|t| resolve_type_expr(env, t))
                .unwrap_or(Type::Void);
            env.functions.insert(func.name.name.clone(), (param_types, ret_type));
        }
        ItemKind::Kernel(kernel) => {
            let param_types: Vec<Type> = kernel.params.iter()
                .map(|p| resolve_type_expr(env, &p.ty))
                .collect();
            let ret_type = kernel.ret_type.as_ref()
                .map(|t| resolve_type_expr(env, t))
                .unwrap_or(Type::Void);
            env.functions.insert(kernel.name.name.clone(), (param_types, ret_type));
        }
        ItemKind::Const(c) => {
            if let Some(ty) = &c.ty {
                let resolved = resolve_type_expr(env, ty);
                env.define(&c.name.name, resolved);
            }
        }
        ItemKind::TypeAlias(alias) => {
            let resolved = resolve_type_expr(env, &alias.value);
            env.type_defs.insert(alias.name.name.clone(), resolved);
        }
        _ => {}
    }
}

fn check_item(env: &mut TypeEnv, item: &Item) {
    match &item.kind {
        ItemKind::Function(func) => check_function(env, func),
        ItemKind::Kernel(kernel) => check_kernel(env, kernel),
        ItemKind::Struct(s) => check_struct(env, s),
        _ => {}
    }
}

fn check_function(env: &mut TypeEnv, func: &Function) {
    env.push_scope();

    // Register generic type params
    for g in &func.generics {
        match &g.kind {
            GenericParamKind::Type { bounds } => {
                let bound_names: Vec<String> = bounds.iter()
                    .map(|b| format!("{}", b))
                    .collect();
                env.type_params.insert(g.name.name.clone(), bound_names);
            }
            GenericParamKind::Const { ty } => {
                let resolved = resolve_type_expr(env, ty);
                env.define(&g.name.name, resolved);
            }
        }
    }

    // Register parameters
    for param in &func.params {
        let ty = resolve_type_expr(env, &param.ty);
        env.define(&param.name.name, ty);
    }

    let expected_ret = func.ret_type.as_ref()
        .map(|t| resolve_type_expr(env, t))
        .unwrap_or(Type::Void);

    // Check body
    check_block(env, &func.body, &expected_ret);

    env.pop_scope();
}

fn check_kernel(env: &mut TypeEnv, kernel: &Kernel) {
    env.push_scope();

    // Register generic type params
    for g in &kernel.generics {
        match &g.kind {
            GenericParamKind::Type { bounds } => {
                let bound_names: Vec<String> = bounds.iter()
                    .map(|b| format!("{}", b))
                    .collect();
                env.type_params.insert(g.name.name.clone(), bound_names);
            }
            GenericParamKind::Const { ty } => {
                let resolved = resolve_type_expr(env, ty);
                env.define(&g.name.name, resolved);
            }
        }
    }

    // Register parameters
    for param in &kernel.params {
        let ty = resolve_type_expr(env, &param.ty);
        env.define(&param.name.name, ty);
    }

    let expected_ret = kernel.ret_type.as_ref()
        .map(|t| resolve_type_expr(env, t))
        .unwrap_or(Type::Void);

    check_block(env, &kernel.body, &expected_ret);

    env.pop_scope();
}

fn check_struct(env: &mut TypeEnv, s: &StructDef) {
    // Check that all field types are valid
    for field in &s.fields {
        let _ = resolve_type_expr(env, &field.ty);
    }
}

fn check_block(env: &mut TypeEnv, block: &Block, expected_ret: &Type) {
    for stmt in &block.stmts {
        check_stmt(env, stmt, expected_ret);
    }

    if let Some(expr) = &block.expr {
        let ty = infer_expr(env, expr);
        if !types_compatible(expected_ret, &ty) && *expected_ret != Type::Void {
            env.type_mismatch(expr.span, expected_ret, &ty);
        }
    }
}

fn check_stmt(env: &mut TypeEnv, stmt: &Stmt, expected_ret: &Type) {
    match &stmt.kind {
        StmtKind::Let { name, ty, value } => {
            let value_ty = infer_expr(env, value);
            if let Some(annotated_ty) = ty {
                let expected = resolve_type_expr(env, annotated_ty);
                if !types_compatible(&expected, &value_ty) {
                    env.type_mismatch(value.span, &expected, &value_ty);
                }
                env.define(&name.name, expected);
            } else {
                env.define(&name.name, value_ty);
            }
        }
        StmtKind::Var { name, ty, value } => {
            let value_ty = infer_expr(env, value);
            if let Some(annotated_ty) = ty {
                let expected = resolve_type_expr(env, annotated_ty);
                if !types_compatible(&expected, &value_ty) {
                    env.type_mismatch(value.span, &expected, &value_ty);
                }
                env.define(&name.name, expected);
            } else {
                env.define(&name.name, value_ty);
            }
        }
        StmtKind::Return(Some(expr)) => {
            let ty = infer_expr(env, expr);
            if !types_compatible(expected_ret, &ty) && *expected_ret != Type::Void {
                env.type_mismatch(expr.span, expected_ret, &ty);
            }
        }
        StmtKind::Return(None) => {
            if *expected_ret != Type::Void {
                env.error(stmt.span, format!("expected return value of type `{}`", expected_ret));
            }
        }
        StmtKind::Expr(expr) => {
            let _ = infer_expr(env, expr);
        }
        StmtKind::Assign { target, op: _, value } => {
            let target_ty = infer_expr(env, target);
            let value_ty = infer_expr(env, value);
            if !types_compatible(&target_ty, &value_ty) {
                env.type_mismatch(value.span, &target_ty, &value_ty);
            }
        }
        StmtKind::For { var, iter, body } => {
            env.push_scope();
            let iter_ty = infer_expr(env, iter);
            // For range iteration, the loop variable gets the element type
            let var_ty = match &iter_ty {
                Type::Tensor { elem, .. } => *elem.clone(),
                _ => Type::Scalar(ScalarType::I64), // default for ranges
            };
            env.define(&var.name, var_ty);
            check_block(env, body, &Type::Void);
            env.pop_scope();
        }
        StmtKind::While { cond, body } => {
            let cond_ty = infer_expr(env, cond);
            if !types_compatible(&Type::Scalar(ScalarType::Bool), &cond_ty) {
                env.type_mismatch(cond.span, &Type::Scalar(ScalarType::Bool), &cond_ty);
            }
            env.push_scope();
            check_block(env, body, &Type::Void);
            env.pop_scope();
        }
        StmtKind::Break | StmtKind::Continue => {}
    }
}

fn infer_expr(env: &mut TypeEnv, expr: &Expr) -> Type {
    match &expr.kind {
        ExprKind::IntLiteral(_) => Type::Scalar(ScalarType::I64),
        ExprKind::FloatLiteral(_) => Type::Scalar(ScalarType::F64),
        ExprKind::StringLiteral(_) => Type::Named("String".to_string()),
        ExprKind::BoolLiteral(_) => Type::Scalar(ScalarType::Bool),

        ExprKind::Ident(id) => {
            if let Some(ty) = env.lookup(&id.name) {
                ty.clone()
            } else if env.type_params.contains_key(&id.name) {
                // Generic type parameter — treat as its own type
                Type::Named(id.name.clone())
            } else {
                env.error(id.span, format!("undefined variable `{}`", id.name));
                Type::Error
            }
        }

        ExprKind::Binary { lhs, op, rhs } => {
            let lhs_ty = infer_expr(env, lhs);
            let rhs_ty = infer_expr(env, rhs);

            match op {
                BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                    if !types_compatible(&lhs_ty, &rhs_ty) {
                        env.type_mismatch(expr.span, &lhs_ty, &rhs_ty);
                    }
                    Type::Scalar(ScalarType::Bool)
                }
                BinOp::And | BinOp::Or => {
                    if lhs_ty != Type::Scalar(ScalarType::Bool) {
                        env.type_mismatch(lhs.span, &Type::Scalar(ScalarType::Bool), &lhs_ty);
                    }
                    if rhs_ty != Type::Scalar(ScalarType::Bool) {
                        env.type_mismatch(rhs.span, &Type::Scalar(ScalarType::Bool), &rhs_ty);
                    }
                    Type::Scalar(ScalarType::Bool)
                }
                BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::Mod | BinOp::Pow => {
                    check_numeric_op(env, expr.span, &lhs_ty, &rhs_ty)
                }
                BinOp::ElemMul | BinOp::ElemDiv => {
                    // Elementwise ops require tensor operands
                    match (&lhs_ty, &rhs_ty) {
                        (Type::Tensor { .. }, Type::Tensor { .. }) => lhs_ty,
                        _ => {
                            env.error(expr.span, "elementwise operations require tensor operands".to_string());
                            Type::Error
                        }
                    }
                }
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    if !types_compatible(&lhs_ty, &rhs_ty) {
                        env.type_mismatch(expr.span, &lhs_ty, &rhs_ty);
                    }
                    lhs_ty
                }
            }
        }

        ExprKind::Unary { op, expr: inner } => {
            let ty = infer_expr(env, inner);
            match op {
                UnaryOp::Neg => {
                    if !is_numeric(&ty) {
                        env.error(inner.span, format!("cannot negate type `{}`", ty));
                    }
                    ty
                }
                UnaryOp::Not => {
                    if ty != Type::Scalar(ScalarType::Bool) {
                        env.error(inner.span, format!("cannot apply `!` to type `{}`", ty));
                    }
                    Type::Scalar(ScalarType::Bool)
                }
                UnaryOp::BitNot => ty,
            }
        }

        ExprKind::Call { func, args } => {
            // Check if it's a known function
            if let ExprKind::Ident(id) = &func.kind {
                if let Some((param_types, ret_type)) = env.functions.get(&id.name).cloned() {
                    // Check argument count
                    if args.len() != param_types.len() {
                        env.error(
                            expr.span,
                            format!(
                                "function `{}` expects {} arguments, got {}",
                                id.name, param_types.len(), args.len()
                            ),
                        );
                    } else {
                        // Check argument types
                        for (arg, expected) in args.iter().zip(param_types.iter()) {
                            let arg_ty = infer_expr(env, arg);
                            if !types_compatible(expected, &arg_ty) {
                                env.type_mismatch(
                                    arg.span,
                                    expected,
                                    &arg_ty,
                                );
                            }
                        }
                    }
                    return ret_type;
                }
            }

            // Unknown function — infer args and return unknown
            for arg in args {
                let _ = infer_expr(env, arg);
            }
            env.fresh_var()
        }

        ExprKind::MatMul { lhs, rhs } => {
            let lhs_ty = infer_expr(env, lhs);
            let rhs_ty = infer_expr(env, rhs);

            match (&lhs_ty, &rhs_ty) {
                (
                    Type::Tensor { elem: elem_l, shape: shape_l },
                    Type::Tensor { elem: elem_r, shape: shape_r },
                ) => {
                    if elem_l != elem_r {
                        env.error(
                            expr.span,
                            format!("matmul element type mismatch: `{}` vs `{}`", elem_l, elem_r),
                        );
                    }
                    // For 2D: [M, K] @ [K, N] -> [M, N]
                    if shape_l.len() == 2 && shape_r.len() == 2 {
                        Type::Tensor {
                            elem: elem_l.clone(),
                            shape: vec![shape_l[0].clone(), shape_r[1].clone()],
                        }
                    } else {
                        // For now, just return a tensor with inferred shape
                        lhs_ty
                    }
                }
                _ => {
                    env.error(expr.span, "matmul requires tensor operands".to_string());
                    Type::Error
                }
            }
        }

        ExprKind::FieldAccess { base, field: _ } => {
            let _ = infer_expr(env, base);
            // For now we don't resolve struct fields; return unknown
            env.fresh_var()
        }

        ExprKind::Index { base, indices } => {
            let base_ty = infer_expr(env, base);
            for idx in indices {
                let _ = infer_expr(env, idx);
            }
            match &base_ty {
                Type::Tensor { elem, shape } => {
                    if indices.len() == shape.len() {
                        // Full indexing: returns element type
                        *elem.clone()
                    } else {
                        // Partial indexing: returns a tensor with fewer dimensions
                        base_ty
                    }
                }
                Type::Array { elem, .. } => *elem.clone(),
                _ => env.fresh_var(),
            }
        }

        ExprKind::Block(block) => {
            env.push_scope();
            for stmt in &block.stmts {
                check_stmt(env, stmt, &Type::Void);
            }
            let ty = if let Some(expr) = &block.expr {
                infer_expr(env, expr)
            } else {
                Type::Void
            };
            env.pop_scope();
            ty
        }

        ExprKind::If { cond, then_block, else_block } => {
            let cond_ty = infer_expr(env, cond);
            if !types_compatible(&Type::Scalar(ScalarType::Bool), &cond_ty) {
                env.type_mismatch(cond.span, &Type::Scalar(ScalarType::Bool), &cond_ty);
            }

            env.push_scope();
            let then_ty = if let Some(expr) = &then_block.expr {
                for stmt in &then_block.stmts {
                    check_stmt(env, stmt, &Type::Void);
                }
                infer_expr(env, expr)
            } else {
                for stmt in &then_block.stmts {
                    check_stmt(env, stmt, &Type::Void);
                }
                Type::Void
            };
            env.pop_scope();

            if let Some(else_block) = else_block {
                env.push_scope();
                let else_ty = if let Some(expr) = &else_block.expr {
                    for stmt in &else_block.stmts {
                        check_stmt(env, stmt, &Type::Void);
                    }
                    infer_expr(env, expr)
                } else {
                    for stmt in &else_block.stmts {
                        check_stmt(env, stmt, &Type::Void);
                    }
                    Type::Void
                };
                env.pop_scope();

                if !types_compatible(&then_ty, &else_ty) {
                    env.error(
                        expr.span,
                        format!("if/else branches have incompatible types: `{}` vs `{}`", then_ty, else_ty),
                    );
                }
                then_ty
            } else {
                Type::Void
            }
        }

        ExprKind::Range { start, end } => {
            let start_ty = infer_expr(env, start);
            let end_ty = infer_expr(env, end);
            if !types_compatible(&start_ty, &end_ty) {
                env.type_mismatch(expr.span, &start_ty, &end_ty);
            }
            Type::Named("Range".to_string())
        }

        ExprKind::ArrayLiteral(elems) => {
            if elems.is_empty() {
                return Type::Array {
                    elem: Box::new(env.fresh_var()),
                    size: 0,
                };
            }
            let first_ty = infer_expr(env, &elems[0]);
            for elem in &elems[1..] {
                let ty = infer_expr(env, elem);
                if !types_compatible(&first_ty, &ty) {
                    env.type_mismatch(elem.span, &first_ty, &ty);
                }
            }
            Type::Array {
                elem: Box::new(first_ty),
                size: elems.len() as u64,
            }
        }

        ExprKind::TypeCall { .. } => env.fresh_var(),

        ExprKind::Cast { expr: inner, ty } => {
            let _ = infer_expr(env, inner);
            resolve_type_expr(env, ty)
        }

        ExprKind::StructLiteral { fields, .. } => {
            for (_, fexpr) in fields {
                infer_expr(env, fexpr);
            }
            env.fresh_var() // TODO: resolve to actual struct type
        }

        ExprKind::Match { expr: match_expr, arms } => {
            infer_expr(env, match_expr);
            let mut result_ty = env.fresh_var();
            for arm in arms {
                let arm_ty = infer_expr(env, &arm.body);
                result_ty = arm_ty;
            }
            result_ty
        }
    }
}

fn check_numeric_op(env: &mut TypeEnv, span: Span, lhs: &Type, rhs: &Type) -> Type {
    // Both tensors of same shape — elementwise op
    if let (Type::Tensor { .. }, Type::Tensor { .. }) = (lhs, rhs) {
        if !types_compatible(lhs, rhs) {
            env.type_mismatch(span, lhs, rhs);
        }
        return lhs.clone();
    }

    // Both scalars
    if is_numeric(lhs) && is_numeric(rhs) {
        if !types_compatible(lhs, rhs) {
            env.type_mismatch(span, lhs, rhs);
        }
        return lhs.clone();
    }

    // Scalar + tensor broadcasting
    if is_numeric(lhs) && matches!(rhs, Type::Tensor { .. }) {
        return rhs.clone();
    }
    if matches!(lhs, Type::Tensor { .. }) && is_numeric(rhs) {
        return lhs.clone();
    }

    // Named types (might be field elements or other numeric types)
    if matches!(lhs, Type::Named(_)) || matches!(rhs, Type::Named(_)) {
        return lhs.clone();
    }

    // Type variables
    if matches!(lhs, Type::Var(_)) || matches!(rhs, Type::Var(_)) {
        return lhs.clone();
    }

    if matches!(lhs, Type::Error) || matches!(rhs, Type::Error) {
        return Type::Error;
    }

    env.error(span, format!("cannot perform arithmetic on `{}` and `{}`", lhs, rhs));
    Type::Error
}

fn is_numeric(ty: &Type) -> bool {
    matches!(
        ty,
        Type::Scalar(
            ScalarType::U8 | ScalarType::U16 | ScalarType::U32 | ScalarType::U64 | ScalarType::U128 |
            ScalarType::I8 | ScalarType::I16 | ScalarType::I32 | ScalarType::I64 |
            ScalarType::F16 | ScalarType::F32 | ScalarType::F64
        )
    )
}

fn types_compatible(expected: &Type, got: &Type) -> bool {
    // Type variables are compatible with anything
    if matches!(expected, Type::Var(_)) || matches!(got, Type::Var(_)) {
        return true;
    }
    // Error types are compatible with anything (to avoid cascading errors)
    if matches!(expected, Type::Error) || matches!(got, Type::Error) {
        return true;
    }
    // Named types that match generic type parameters are compatible
    if matches!(expected, Type::Named(_)) || matches!(got, Type::Named(_)) {
        return true; // Lenient for now — full resolution in Phase 1
    }

    match (expected, got) {
        (Type::Scalar(a), Type::Scalar(b)) => {
            a == b || are_numeric_compatible(a, b)
        }
        (
            Type::Tensor { elem: e1, shape: s1 },
            Type::Tensor { elem: e2, shape: s2 },
        ) => {
            types_compatible(e1, e2) && shapes_compatible(s1, s2)
        }
        (Type::Array { elem: e1, size: s1 }, Type::Array { elem: e2, size: s2 }) => {
            s1 == s2 && types_compatible(e1, e2)
        }
        (Type::Void, Type::Void) => true,
        _ => false,
    }
}

/// Check if two scalar types are compatible for implicit conversion.
/// For Phase 0, we allow: f64 literals ↔ f32, i64 literals ↔ i32/u32
fn are_numeric_compatible(a: &ScalarType, b: &ScalarType) -> bool {
    use ScalarType::*;
    matches!(
        (a, b),
        // Float widening/narrowing for literals
        (F32, F64) | (F64, F32) |
        // Integer widening/narrowing for literals
        (I32, I64) | (I64, I32) |
        (U32, I64) | (I64, U32) |
        (U32, U64) | (U64, U32) |
        (I32, U32) | (U32, I32)
    )
}

fn shapes_compatible(a: &[Dim], b: &[Dim]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    a.iter().zip(b.iter()).all(|(d1, d2)| match (d1, d2) {
        (Dim::Lit(n1), Dim::Lit(n2)) => n1 == n2,
        (Dim::Sym(_), _) | (_, Dim::Sym(_)) => true, // Symbolic dims are compatible
        (Dim::Dynamic, _) | (_, Dim::Dynamic) => true,
    })
}

fn resolve_type_expr(env: &mut TypeEnv, ty: &TypeExpr) -> Type {
    match &ty.kind {
        TypeExprKind::Named(id) => {
            match id.name.as_str() {
                "u8" => Type::Scalar(ScalarType::U8),
                "u16" => Type::Scalar(ScalarType::U16),
                "u32" => Type::Scalar(ScalarType::U32),
                "u64" => Type::Scalar(ScalarType::U64),
                "u128" => Type::Scalar(ScalarType::U128),
                "i8" => Type::Scalar(ScalarType::I8),
                "i16" => Type::Scalar(ScalarType::I16),
                "i32" => Type::Scalar(ScalarType::I32),
                "i64" => Type::Scalar(ScalarType::I64),
                "f16" => Type::Scalar(ScalarType::F16),
                "f32" => Type::Scalar(ScalarType::F32),
                "f64" => Type::Scalar(ScalarType::F64),
                "bool" => Type::Scalar(ScalarType::Bool),
                "void" => Type::Void,
                name => {
                    if let Some(resolved) = env.type_defs.get(name) {
                        resolved.clone()
                    } else {
                        Type::Named(name.to_string())
                    }
                }
            }
        }
        TypeExprKind::Generic { name, args } => {
            if name.name == "Tensor" {
                // Tensor<elem_type, [shape]>
                let elem = if let Some(TypeArg::Type(t)) = args.first() {
                    resolve_type_expr(env, t)
                } else {
                    Type::Error
                };
                let shape = if let Some(TypeArg::Shape(dims)) = args.get(1) {
                    dims.iter().map(|d| match d {
                        ShapeDim::Lit(n) => Dim::Lit(*n),
                        ShapeDim::Ident(id) => Dim::Sym(id.name.clone()),
                        ShapeDim::Dynamic => Dim::Dynamic,
                        ShapeDim::Expr(_) => Dim::Dynamic, // TODO: evaluate
                    }).collect()
                } else {
                    Vec::new()
                };
                Type::Tensor {
                    elem: Box::new(elem),
                    shape,
                }
            } else {
                Type::Named(name.name.clone())
            }
        }
        TypeExprKind::Array { elem, size } => {
            let elem_ty = resolve_type_expr(env, elem);
            let size_val = match &size.kind {
                ExprKind::IntLiteral(n) => *n as u64,
                _ => 0, // TODO: evaluate const exprs
            };
            Type::Array {
                elem: Box::new(elem_ty),
                size: size_val,
            }
        }
        TypeExprKind::Ref { inner, .. } => {
            resolve_type_expr(env, inner)
        }
        TypeExprKind::Tuple(types) => {
            // For now, just use Named for tuples
            Type::Named(format!("({})", types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ")))
        }
        TypeExprKind::Shape(_) => {
            Type::Named("Shape".to_string())
        }
        TypeExprKind::Fn { params, ret } => {
            let param_types: Vec<Type> = params.iter()
                .map(|p| resolve_type_expr(env, p))
                .collect();
            let ret_type = resolve_type_expr(env, ret);
            Type::Fn {
                params: param_types,
                ret: Box::new(ret_type),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser;

    fn check_ok(source: &str) {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).expect("parse failed");
        check(&program, 0).expect("type check failed");
    }

    fn check_err(source: &str) -> Vec<Diagnostic<usize>> {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).expect("parse failed");
        check(&program, 0).expect_err("expected type error")
    }

    #[test]
    fn test_simple_function_passes() {
        check_ok("fn add(a: f32, b: f32) -> f32 { return a + b }");
    }

    #[test]
    fn test_let_binding_type_inferred() {
        check_ok(
            "fn test() {
                let x = 42
                let y = x + 1
            }",
        );
    }

    #[test]
    fn test_type_mismatch_return() {
        let errs = check_err(
            "fn test() -> f32 {
                return true
            }",
        );
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_undefined_variable() {
        let errs = check_err(
            "fn test() -> f32 {
                return undefined_var
            }",
        );
        assert!(!errs.is_empty());
    }

    #[test]
    fn test_kernel_tensor_params() {
        check_ok(
            "kernel vec_add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> {
                return a + b
            }",
        );
    }

    #[test]
    fn test_matmul_type() {
        check_ok(
            "fn test(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
                return a @ b
            }",
        );
    }

    #[test]
    fn test_for_loop_typing() {
        check_ok(
            "fn test() {
                for i in 0..10 {
                    let x = i + 1
                }
            }",
        );
    }

    #[test]
    fn test_if_else_typing() {
        check_ok(
            "fn test(x: f32) -> f32 {
                if x > 0.0 { x } else { -x }
            }",
        );
    }

    #[test]
    fn test_function_call_arg_count() {
        let errs = check_err(
            "fn add(a: f32, b: f32) -> f32 { return a + b }
             fn test() { add(1.0) }",
        );
        assert!(!errs.is_empty());
    }
}
