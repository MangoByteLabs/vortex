use crate::ast::*;
use crate::lexer::Span;
use crate::shape_checker::LegacyShapeChecker as ShapeChecker;
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

/// Registered enum variant info for type checking
#[derive(Debug, Clone)]
struct EnumVariantInfo {
    /// The parent enum name
    enum_name: String,
    /// Field types for this variant (empty for unit variants)
    field_types: Vec<Type>,
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
    /// Enum definitions: enum_name -> list of variant names
    enum_variants: HashMap<String, Vec<String>>,
    /// Variant info: variant_name -> EnumVariantInfo
    variant_info: HashMap<String, EnumVariantInfo>,
    /// Struct field definitions: struct_name -> [(field_name, field_type)]
    struct_fields: HashMap<String, Vec<(String, Type)>>,
    /// Substitution map for type variable unification
    substitutions: HashMap<usize, Type>,
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
            enum_variants: HashMap::new(),
            variant_info: HashMap::new(),
            struct_fields: HashMap::new(),
            substitutions: HashMap::new(),
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

    fn resolve(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(resolved) = self.substitutions.get(id) {
                    self.resolve(resolved)
                } else {
                    ty.clone()
                }
            }
            _ => ty.clone(),
        }
    }

    /// Unify two types, updating the substitution map.
    /// Returns the unified type on success, or None on failure (and records an error).
    fn unify(&mut self, a: &Type, b: &Type, span: Span) -> Option<Type> {
        let a = self.resolve_deep(a);
        let b = self.resolve_deep(b);

        // Error types unify with anything (avoid cascading errors)
        if matches!(&a, Type::Error) || matches!(&b, Type::Error) {
            return Some(if matches!(&a, Type::Error) { b } else { a });
        }

        match (&a, &b) {
            // Type variable on the left -> bind
            (Type::Var(id), _) => {
                self.bind(*id, &b);
                Some(b)
            }
            // Type variable on the right -> bind
            (_, Type::Var(id)) => {
                self.bind(*id, &a);
                Some(a)
            }

            // Same concrete types
            (Type::Scalar(sa), Type::Scalar(sb)) => {
                if sa == sb {
                    Some(a)
                } else if are_numeric_compatible(sa, sb) {
                    Some(wider_numeric_type(sa, sb))
                } else {
                    self.type_mismatch(span, &a, &b);
                    None
                }
            }
            (Type::Void, Type::Void) => Some(Type::Void),
            (Type::Named(na), Type::Named(nb)) if na == nb => Some(a),

            // Named types: be lenient only against Var/Error (handled above) and
            // other Named types of the same name (handled above).
            // Named vs Scalar/Array/Tensor/Void/Fn is a mismatch.
            // Named vs Named (different) is also a mismatch.
            (Type::Named(_), _) | (_, Type::Named(_)) => {
                self.type_mismatch(span, &a, &b);
                None
            }

            // Tensor types
            (
                Type::Tensor { elem: e1, shape: s1 },
                Type::Tensor { elem: e2, shape: s2 },
            ) => {
                let elem_unified = self.unify(e1, e2, span);
                if shapes_compatible(s1, s2) {
                    elem_unified.map(|e| Type::Tensor {
                        elem: Box::new(e),
                        shape: s1.clone(),
                    })
                } else {
                    self.type_mismatch(span, &a, &b);
                    None
                }
            }

            // Array types
            (Type::Array { elem: e1, size: s1 }, Type::Array { elem: e2, size: s2 }) => {
                if s1 == s2 {
                    self.unify(e1, e2, span).map(|e| Type::Array {
                        elem: Box::new(e),
                        size: *s1,
                    })
                } else {
                    self.type_mismatch(span, &a, &b);
                    None
                }
            }

            // Function types
            (Type::Fn { params: p1, ret: r1 }, Type::Fn { params: p2, ret: r2 }) => {
                if p1.len() != p2.len() {
                    self.type_mismatch(span, &a, &b);
                    return None;
                }
                let mut unified_params = Vec::new();
                for (pa, pb) in p1.iter().zip(p2.iter()) {
                    match self.unify(pa, pb, span) {
                        Some(u) => unified_params.push(u),
                        None => return None,
                    }
                }
                self.unify(r1, r2, span).map(|r| Type::Fn {
                    params: unified_params,
                    ret: Box::new(r),
                })
            }

            _ => {
                self.type_mismatch(span, &a, &b);
                None
            }
        }
    }

    /// Try to unify without reporting errors. Returns Some on success.
    fn try_unify(&mut self, a: &Type, b: &Type) -> Option<Type> {
        let a = self.resolve_deep(a);
        let b = self.resolve_deep(b);

        if matches!(&a, Type::Error) || matches!(&b, Type::Error) {
            return Some(if matches!(&a, Type::Error) { b } else { a });
        }

        match (&a, &b) {
            (Type::Var(id), _) => { self.bind(*id, &b); Some(b) }
            (_, Type::Var(id)) => { self.bind(*id, &a); Some(a) }
            (Type::Scalar(sa), Type::Scalar(sb)) => {
                if sa == sb { Some(a) }
                else if are_numeric_compatible(sa, sb) { Some(wider_numeric_type(sa, sb)) }
                else { None }
            }
            (Type::Void, Type::Void) => Some(Type::Void),
            (Type::Named(na), Type::Named(nb)) if na == nb => Some(a),
            (Type::Named(_), _) | (_, Type::Named(_)) => None,
            (
                Type::Tensor { elem: e1, shape: s1 },
                Type::Tensor { elem: e2, shape: s2 },
            ) => {
                if shapes_compatible(s1, s2) {
                    self.try_unify(e1, e2).map(|e| Type::Tensor { elem: Box::new(e), shape: s1.clone() })
                } else { None }
            }
            (Type::Array { elem: e1, size: s1 }, Type::Array { elem: e2, size: s2 }) => {
                if s1 == s2 { self.try_unify(e1, e2).map(|e| Type::Array { elem: Box::new(e), size: *s1 }) }
                else { None }
            }
            _ => None,
        }
    }

    /// Follow substitution chains deeply: resolve Var inside compound types too
    fn resolve_deep(&self, ty: &Type) -> Type {
        match ty {
            Type::Var(id) => {
                if let Some(resolved) = self.substitutions.get(id) {
                    self.resolve_deep(resolved)
                } else {
                    ty.clone()
                }
            }
            Type::Array { elem, size } => Type::Array {
                elem: Box::new(self.resolve_deep(elem)),
                size: *size,
            },
            Type::Tensor { elem, shape } => Type::Tensor {
                elem: Box::new(self.resolve_deep(elem)),
                shape: shape.clone(),
            },
            Type::Fn { params, ret } => Type::Fn {
                params: params.iter().map(|p| self.resolve_deep(p)).collect(),
                ret: Box::new(self.resolve_deep(ret)),
            },
            _ => ty.clone(),
        }
    }

    /// Bind a type variable (with occurs check)
    fn bind(&mut self, id: usize, ty: &Type) {
        if self.occurs(id, ty) {
            return;
        }
        self.substitutions.insert(id, ty.clone());
    }

    /// Occurs check: does Var(id) appear in ty?
    fn occurs(&self, id: usize, ty: &Type) -> bool {
        match self.resolve_deep(ty) {
            Type::Var(other_id) => id == other_id,
            Type::Array { elem, .. } => self.occurs(id, &elem),
            Type::Tensor { elem, .. } => self.occurs(id, &elem),
            Type::Fn { params, ret } => {
                params.iter().any(|p| self.occurs(id, p)) || self.occurs(id, &ret)
            }
            _ => false,
        }
    }

    fn error(&mut self, span: Span, msg: String) {
        self.errors.push(
            Diagnostic::error()
                .with_message(&msg)
                .with_labels(vec![Label::primary(self.file_id, span.start..span.end)]),
        );
    }

    fn warning(&mut self, span: Span, msg: String) {
        self.errors.push(
            Diagnostic::warning()
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
        ItemKind::Struct(s) => {
            let fields: Vec<(String, Type)> = s.fields.iter()
                .map(|f| (f.name.name.clone(), resolve_type_expr(env, &f.ty)))
                .collect();
            env.struct_fields.insert(s.name.name.clone(), fields);
        }
        ItemKind::Enum(e) => {
            let enum_name = e.name.name.clone();
            let mut variant_names = Vec::new();
            for variant in &e.variants {
                let vn = variant.name.name.clone();
                variant_names.push(vn.clone());
                let field_types = match &variant.kind {
                    EnumVariantKind::Unit => {
                        // Register unit variants as variables with the enum type
                        env.define(&vn, Type::Named(enum_name.clone()));
                        Vec::new()
                    }
                    EnumVariantKind::Tuple(types) => {
                        types.iter().map(|t| resolve_type_expr(env, t)).collect()
                    }
                    EnumVariantKind::Struct(fields) => {
                        fields.iter().map(|f| resolve_type_expr(env, &f.ty)).collect()
                    }
                };
                env.variant_info.insert(vn, EnumVariantInfo {
                    enum_name: enum_name.clone(),
                    field_types,
                });
            }
            env.enum_variants.insert(enum_name, variant_names);
        }
        _ => {}
    }
}

fn check_item(env: &mut TypeEnv, item: &Item) {
    match &item.kind {
        ItemKind::Function(func) => check_function(env, func),
        ItemKind::Kernel(kernel) => check_kernel(env, kernel),
        ItemKind::Struct(s) => check_struct(env, s),
        ItemKind::Enum(e) => check_enum(env, e),
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

fn check_enum(env: &mut TypeEnv, e: &EnumDef) {
    for variant in &e.variants {
        match &variant.kind {
            EnumVariantKind::Unit => {}
            EnumVariantKind::Tuple(types) => {
                for ty in types {
                    let _ = resolve_type_expr(env, ty);
                }
            }
            EnumVariantKind::Struct(fields) => {
                for field in fields {
                    let _ = resolve_type_expr(env, &field.ty);
                }
            }
        }
    }
}

fn check_block(env: &mut TypeEnv, block: &Block, expected_ret: &Type) {
    for stmt in &block.stmts {
        check_stmt(env, stmt, expected_ret);
    }

    if let Some(expr) = &block.expr {
        let ty = infer_expr(env, expr);
        if *expected_ret != Type::Void {
            env.unify(expected_ret, &ty, expr.span);
        }
    }
}

fn check_stmt(env: &mut TypeEnv, stmt: &Stmt, expected_ret: &Type) {
    match &stmt.kind {
        StmtKind::Let { name, ty, value } => {
            let value_ty = infer_expr(env, value);
            if let Some(annotated_ty) = ty {
                let expected = resolve_type_expr(env, annotated_ty);
                let unified = env.unify(&expected, &value_ty, value.span);
                env.define(&name.name, unified.unwrap_or(expected));
            } else {
                // Resolve any type variables that got unified
                let resolved = env.resolve_deep(&value_ty);
                env.define(&name.name, resolved);
            }
        }
        StmtKind::Var { name, ty, value } => {
            let value_ty = infer_expr(env, value);
            if let Some(annotated_ty) = ty {
                let expected = resolve_type_expr(env, annotated_ty);
                let unified = env.unify(&expected, &value_ty, value.span);
                env.define(&name.name, unified.unwrap_or(expected));
            } else {
                let resolved = env.resolve_deep(&value_ty);
                env.define(&name.name, resolved);
            }
        }
        StmtKind::Return(Some(expr)) => {
            let ty = infer_expr(env, expr);
            if *expected_ret != Type::Void {
                env.unify(expected_ret, &ty, expr.span);
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
            env.unify(&target_ty, &value_ty, value.span);
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
            env.unify(&Type::Scalar(ScalarType::Bool), &cond_ty, cond.span);
            env.push_scope();
            check_block(env, body, &Type::Void);
            env.pop_scope();
        }
        StmtKind::Loop { body } => {
            env.push_scope();
            check_block(env, body, &Type::Void);
            env.pop_scope();
        }
        StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Dispatch { index, targets: _, args } => {
            let _ = infer_expr(env, index);
            for arg in args {
                let _ = infer_expr(env, arg);
            }
        }
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
                env.resolve_deep(ty)
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
                    env.unify(&lhs_ty, &rhs_ty, expr.span);
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
                    // Elementwise ops require tensor operands with compatible shapes
                    match (&lhs_ty, &rhs_ty) {
                        (
                            Type::Tensor { elem: elem_l, shape: shape_l },
                            Type::Tensor { elem: elem_r, shape: shape_r },
                        ) => {
                            env.unify(elem_l, elem_r, expr.span);
                            match ShapeChecker::check_broadcast(shape_l, shape_r) {
                                Ok(result_shape) => Type::Tensor {
                                    elem: elem_l.clone(),
                                    shape: result_shape,
                                },
                                Err(e) => {
                                    env.error(expr.span, format!("{}", e));
                                    Type::Error
                                }
                            }
                        }
                        _ => {
                            env.error(expr.span, "elementwise operations require tensor operands".to_string());
                            Type::Error
                        }
                    }
                }
                BinOp::BitAnd | BinOp::BitOr | BinOp::BitXor | BinOp::Shl | BinOp::Shr => {
                    let unified = env.unify(&lhs_ty, &rhs_ty, expr.span);
                    unified.unwrap_or(lhs_ty)
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
            // Check if it's an enum variant constructor
            if let ExprKind::Ident(id) = &func.kind {
                if let Some(info) = env.variant_info.get(&id.name).cloned() {
                    if !info.field_types.is_empty() {
                        if args.len() != info.field_types.len() {
                            env.error(
                                expr.span,
                                format!(
                                    "variant `{}` expects {} fields, got {}",
                                    id.name, info.field_types.len(), args.len()
                                ),
                            );
                        } else {
                            for (arg, expected) in args.iter().zip(info.field_types.iter()) {
                                let arg_ty = infer_expr(env, arg);
                                env.unify(expected, &arg_ty, arg.span);
                            }
                        }
                    }
                    return Type::Named(info.enum_name.clone());
                }
            }

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
                        // Check argument types via unification
                        for (arg, expected) in args.iter().zip(param_types.iter()) {
                            let arg_ty = infer_expr(env, arg);
                            env.unify(expected, &arg_ty, arg.span);
                        }
                    }
                    return env.resolve_deep(&ret_type);
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
                    match ShapeChecker::check_matmul(shape_l, shape_r) {
                        Ok(result_shape) => Type::Tensor {
                            elem: elem_l.clone(),
                            shape: result_shape,
                        },
                        Err(e) => {
                            env.error(expr.span, format!("{}", e));
                            Type::Error
                        }
                    }
                }
                _ => {
                    env.error(expr.span, "matmul requires tensor operands".to_string());
                    Type::Error
                }
            }
        }

        ExprKind::FieldAccess { base, field } => {
            let base_ty = infer_expr(env, base);
            let resolved = env.resolve(&base_ty);
            if let Type::Named(ref struct_name) = resolved {
                if let Some(fields) = env.struct_fields.get(struct_name).cloned() {
                    if let Some((_, fty)) = fields.iter().find(|(n, _)| n == &field.name) {
                        return fty.clone();
                    }
                }
            }
            // Fall back to fresh var for unknown structs/fields
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
            env.unify(&Type::Scalar(ScalarType::Bool), &cond_ty, cond.span);

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

                let unified = env.unify(&then_ty, &else_ty, expr.span);
                unified.unwrap_or(then_ty)
            } else {
                Type::Void
            }
        }

        ExprKind::Range { start, end } => {
            let start_ty = infer_expr(env, start);
            let end_ty = infer_expr(env, end);
            env.unify(&start_ty, &end_ty, expr.span);
            Type::Named("Range".to_string())
        }

        ExprKind::ArrayLiteral(elems) => {
            if elems.is_empty() {
                return Type::Array {
                    elem: Box::new(env.fresh_var()),
                    size: 0,
                };
            }
            let mut elem_ty = infer_expr(env, &elems[0]);
            for elem in &elems[1..] {
                let ty = infer_expr(env, elem);
                if let Some(unified) = env.unify(&elem_ty, &ty, elem.span) {
                    elem_ty = unified;
                }
            }
            Type::Array {
                elem: Box::new(env.resolve_deep(&elem_ty)),
                size: elems.len() as u64,
            }
        }

        ExprKind::TypeCall { .. } => env.fresh_var(),

        ExprKind::Cast { expr: inner, ty } => {
            let _ = infer_expr(env, inner);
            resolve_type_expr(env, ty)
        }

        ExprKind::StructLiteral { name, fields } => {
            if let Some(def_fields) = env.struct_fields.get(&name.name).cloned() {
                for (fname, fexpr) in fields {
                    let val_ty = infer_expr(env, fexpr);
                    if let Some((_, expected_ty)) = def_fields.iter().find(|(n, _)| n == &fname.name) {
                        env.unify(expected_ty, &val_ty, fexpr.span);
                    } else {
                        env.error(fname.span, format!("struct `{}` has no field `{}`", name.name, fname.name));
                    }
                }
                for (def_name, _) in &def_fields {
                    if !fields.iter().any(|(f, _)| f.name == *def_name) {
                        env.error(expr.span, format!("missing field `{}` in struct `{}`", def_name, name.name));
                    }
                }
            } else {
                for (_, fexpr) in fields {
                    infer_expr(env, fexpr);
                }
            }
            Type::Named(name.name.clone())
        }

        ExprKind::Match { expr: match_expr, arms } => {
            let match_ty = infer_expr(env, match_expr);

            // Collect variant names referenced in patterns and check for wildcard/catch-all
            let mut covered_variants: Vec<String> = Vec::new();
            let mut has_wildcard = false;

            for arm in arms {
                match &arm.pattern {
                    Pattern::Wildcard => { has_wildcard = true; }
                    Pattern::Ident(id) => {
                        // If this identifier is a known variant name, treat as variant match
                        if env.variant_info.contains_key(&id.name) {
                            covered_variants.push(id.name.clone());
                        } else {
                            has_wildcard = true; // catch-all binding
                        }
                    }
                    Pattern::Variant { name, fields } => {
                        covered_variants.push(name.name.clone());
                        // Check variant constructor field count
                        if let Some(info) = env.variant_info.get(&name.name) {
                            if !info.field_types.is_empty() && fields.len() != info.field_types.len() {
                                env.error(
                                    arm.span,
                                    format!(
                                        "variant `{}` expects {} fields, got {}",
                                        name.name, info.field_types.len(), fields.len()
                                    ),
                                );
                            }
                        }
                    }
                    Pattern::Literal(_) => {}
                    Pattern::StructVariant { name, .. } => {
                        covered_variants.push(name.name.clone());
                    }
                    Pattern::Or(pats) => {
                        for p in pats {
                            match p {
                                Pattern::Variant { name, .. } => { covered_variants.push(name.name.clone()); }
                                Pattern::Ident(id) => {
                                    if env.variant_info.contains_key(&id.name) {
                                        covered_variants.push(id.name.clone());
                                    } else {
                                        has_wildcard = true;
                                    }
                                }
                                Pattern::Wildcard => { has_wildcard = true; }
                                _ => {}
                            }
                        }
                    }
                    Pattern::Tuple(_) => {}
                    Pattern::Rest => { has_wildcard = true; }
                }
            }

            // Exhaustiveness check: if the match target is a Named enum type,
            // check all variants are covered
            if !has_wildcard {
                if let Type::Named(ref type_name) = match_ty {
                    if let Some(all_variants) = env.enum_variants.get(type_name) {
                        let missing: Vec<&String> = all_variants.iter()
                            .filter(|v| !covered_variants.contains(v))
                            .collect();
                        if !missing.is_empty() {
                            let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
                            env.warning(
                                expr.span,
                                format!("non-exhaustive match: missing variants: {}", names.join(", ")),
                            );
                        }
                    }
                }
            }

            // Infer result type from arms — all must be compatible
            let mut arm_types: Vec<Type> = Vec::new();
            for arm in arms {
                let arm_ty = infer_expr(env, &arm.body);
                arm_types.push(arm_ty);
            }
            if arm_types.is_empty() {
                env.fresh_var()
            } else {
                let mut result_ty = arm_types[0].clone();
                for (i, arm_ty) in arm_types[1..].iter().enumerate() {
                    if let Some(unified) = env.unify(&result_ty, arm_ty, arms[i + 1].span) {
                        result_ty = unified;
                    }
                }
                env.resolve_deep(&result_ty)
            }
        }

        ExprKind::Closure { .. } => env.fresh_var(),
        ExprKind::Try(inner) => infer_expr(env, inner),
    }
}

fn check_numeric_op(env: &mut TypeEnv, span: Span, lhs: &Type, rhs: &Type) -> Type {
    let lhs = env.resolve_deep(lhs);
    let rhs = env.resolve_deep(rhs);

    // Both tensors — use shape broadcasting
    if let (
        Type::Tensor { elem: elem_l, shape: shape_l },
        Type::Tensor { elem: elem_r, shape: shape_r },
    ) = (&lhs, &rhs) {
        let unified_elem = env.unify(elem_l, elem_r, span);
        return match ShapeChecker::check_broadcast(shape_l, shape_r) {
            Ok(result_shape) => Type::Tensor {
                elem: Box::new(unified_elem.unwrap_or(*elem_l.clone())),
                shape: result_shape,
            },
            Err(e) => {
                env.error(span, format!("{}", e));
                Type::Error
            }
        };
    }

    // Both scalars — unify (handles widening)
    if is_numeric(&lhs) && is_numeric(&rhs) {
        return env.unify(&lhs, &rhs, span).unwrap_or(lhs);
    }

    // Scalar + tensor broadcasting
    if is_numeric(&lhs) && matches!(&rhs, Type::Tensor { .. }) {
        return rhs;
    }
    if matches!(&lhs, Type::Tensor { .. }) && is_numeric(&rhs) {
        return lhs;
    }

    // Named types (might be field elements or other numeric types)
    if matches!(&lhs, Type::Named(_)) || matches!(&rhs, Type::Named(_)) {
        return lhs;
    }

    // Type variables — unify
    if matches!(&lhs, Type::Var(_)) || matches!(&rhs, Type::Var(_)) {
        return env.unify(&lhs, &rhs, span).unwrap_or(lhs);
    }

    if matches!(&lhs, Type::Error) || matches!(&rhs, Type::Error) {
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

/// Given two numeric-compatible scalar types, return the wider one
fn wider_numeric_type(a: &ScalarType, b: &ScalarType) -> Type {
    use ScalarType::*;
    // Float always wins over integer
    let result = match (a, b) {
        (F64, _) | (_, F64) => F64,
        (F32, _) | (_, F32) => F32,
        (F16, _) | (_, F16) => F16,
        (I64, _) | (_, I64) => I64,
        (U64, _) | (_, U64) => U64,
        (I32, _) | (_, I32) => I32,
        (U32, _) | (_, U32) => U32,
        _ => a.clone(),
    };
    Type::Scalar(result)
}

fn types_compatible(expected: &Type, got: &Type) -> bool {
    // Type variables are compatible with anything (unification deferred)
    if matches!(expected, Type::Var(_)) || matches!(got, Type::Var(_)) {
        return true;
    }
    // Error types are compatible with anything (to avoid cascading errors)
    if matches!(expected, Type::Error) || matches!(got, Type::Error) {
        return true;
    }

    match (expected, got) {
        // Named types: compare by name
        (Type::Named(a), Type::Named(b)) => a == b,
        // Named vs non-Named: incompatible (except Var/Error handled above)
        (Type::Named(_), _) | (_, Type::Named(_)) => false,
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
        TypeExprKind::Sparse(inner) => {
            let _inner_ty = resolve_type_expr(env, inner);
            Type::Named("Sparse".to_string())
        }
        TypeExprKind::SparseIndex { .. } => {
            Type::Named("SparseIndex".to_string())
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

    #[test]
    fn test_enum_type_checking() {
        // Enum with tuple variant should register field types
        check_ok(
            "enum Option {
                None
                Some(i64)
            }
            fn test() -> i64 {
                let x = Some(42)
                return 0
            }",
        );
    }

    // --- Shape checking integration tests ---

    #[test]
    fn test_shape_matmul_batched() {
        // [B, T, D] @ [D, H] -> [B, T, H]
        check_ok(
            "fn test(x: Tensor<f32, [B, T, D]>, w: Tensor<f32, [D, H]>) -> Tensor<f32, [B, T, H]> {
                return x @ w
            }",
        );
    }

    #[test]
    fn test_shape_matmul_mismatch() {
        // [B, T, D] @ [H, D] -> error (inner dims D vs H)
        let errs = check_err(
            "fn test(x: Tensor<f32, [B, T, D]>, w: Tensor<f32, [H, D]>) -> Tensor<f32, [B, T, D]> {
                return x @ w
            }",
        );
        assert!(errs.iter().any(|d| d.message.contains("mismatch")));
    }

    #[test]
    fn test_shape_broadcast_trailing() {
        // [B, T, D] + [D] -> [B, T, D]
        check_ok(
            "fn test(x: Tensor<f32, [B, T, D]>, b: Tensor<f32, [D]>) -> Tensor<f32, [B, T, D]> {
                return x + b
            }",
        );
    }

    #[test]
    fn test_shape_broadcast_same() {
        // [B, T, D] + [B, T, D] -> [B, T, D]
        check_ok(
            "fn test(a: Tensor<f32, [B, T, D]>, b: Tensor<f32, [B, T, D]>) -> Tensor<f32, [B, T, D]> {
                return a + b
            }",
        );
    }

    #[test]
    fn test_shape_broadcast_mismatch() {
        // [B, T, D] + [B, T, H] -> error
        let errs = check_err(
            "fn test(a: Tensor<f32, [B, T, D]>, b: Tensor<f32, [B, T, H]>) -> Tensor<f32, [B, T, D]> {
                return a + b
            }",
        );
        assert!(errs.iter().any(|d| d.message.contains("mismatch")));
    }

    #[test]
    fn test_shape_matmul_concrete() {
        // [4, 128] @ [128, 64] -> [4, 64]
        check_ok(
            "fn test(a: Tensor<f32, [4, 128]>, b: Tensor<f32, [128, 64]>) -> Tensor<f32, [4, 64]> {
                return a @ b
            }",
        );
    }

    #[test]
    fn test_shape_matmul_mixed() {
        // [B, 128] @ [128, D] -> [B, D]
        check_ok(
            "fn test(a: Tensor<f32, [B, 128]>, b: Tensor<f32, [128, D]>) -> Tensor<f32, [B, D]> {
                return a @ b
            }",
        );
    }

    // --- Hindley-Milner unification tests ---

    #[test]
    fn test_unify_let_infer_i64() {
        // let x = 42 -> x inferred as i64
        check_ok("fn test() -> i64 { let x = 42\n return x }");
    }

    #[test]
    fn test_unify_let_infer_f64() {
        // let y = 3.14 -> y inferred as f64
        check_ok("fn test() -> f64 { let y = 3.14\n return y }");
    }

    #[test]
    fn test_unify_numeric_widening() {
        // i64 + f64 is not compatible (they are different scalar types without widening rule)
        // But i32 + i64 should work
        check_ok("fn test(x: i32) -> i64 { return x + 1 }");
    }

    #[test]
    fn test_unify_fn_call_type_checks() {
        check_ok("fn f(a: i64) -> i64 { return a }\nfn test() { f(42) }");
    }

    #[test]
    fn test_unify_fn_call_type_error() {
        let src = "fn f(a: i64) -> i64 { return a }\nfn test() { f(\"hi\") }";
        let tokens = lexer::lex(src);
        let program = parser::parse(tokens, 0).expect("parse failed");
        let result = check(&program, 0);
        assert!(result.is_err(), "expected type error for f(\"hi\") but got Ok");
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("type mismatch")));
    }

    #[test]
    fn test_unify_if_else_same_type() {
        check_ok("fn test() -> i64 { if true { 1 } else { 2 } }");
    }

    #[test]
    fn test_unify_if_else_type_error() {
        let src = "fn test() { let x = if true { 1 } else { \"hi\" } }";
        let tokens = lexer::lex(src);
        let program = parser::parse(tokens, 0).expect("parse failed");
        let result = check(&program, 0);
        assert!(result.is_err(), "expected type error for if/else mismatch but got Ok");
        let errs = result.unwrap_err();
        assert!(errs.iter().any(|d| d.message.contains("type mismatch")));
    }

    #[test]
    fn test_unify_array_indexing() {
        check_ok("fn test() -> i64 { let arr = [1, 2, 3]\n return arr[0] }");
    }

    #[test]
    fn test_unify_fn_return_inferred() {
        // Function with annotated return type, body returns correct type
        check_ok("fn f(x: i64) -> i64 { return x + 1 }");
    }

    #[test]
    fn test_unify_var_resolution() {
        // Type variables from unknown function calls should not cause errors
        check_ok("fn test() { let x = unknown_fn(1, 2) }");
    }

    #[test]
    fn test_match_exhaustiveness() {
        // Non-exhaustive match on an enum should produce a warning
        let tokens = lexer::lex(
            "enum Color {
                Red
                Green
                Blue
            }
            fn test() {
                let c = Red
                let x = match c {
                    Red => 1
                    Green => 2
                }
            }",
        );
        let program = parser::parse(tokens, 0).expect("parse failed");
        let result = check(&program, 0);
        // Should have a warning about missing Blue variant
        match result {
            Err(diags) => {
                let has_exhaustiveness_warning = diags.iter().any(|d| {
                    d.message.contains("non-exhaustive") || d.message.contains("missing variants")
                });
                assert!(has_exhaustiveness_warning, "expected exhaustiveness warning, got: {:?}", diags);
            }
            Ok(()) => {
                // Warnings don't cause Err in current implementation since they use Diagnostic::warning
                // which still gets added to errors vec. If it passes, that means the warning
                // was generated but the check still returned Ok because warnings are in the errors vec.
                // Actually warnings ARE in the errors vec so this should be Err.
                panic!("expected diagnostics for non-exhaustive match");
            }
        }
    }

    #[test]
    fn test_string_where_int_expected() {
        let errs = check_err(
            "fn f(x: i64) -> i64 { return x }
             fn test() { f(\"hello\") }",
        );
        assert!(errs.iter().any(|d| d.message.contains("type mismatch")));
    }

    #[test]
    fn test_arg_count_mismatch() {
        let errs = check_err(
            "fn f(a: i64, b: i64) -> i64 { return a + b }
             fn test() { f(1) }",
        );
        assert!(errs.iter().any(|d| d.message.contains("expects 2 arguments")));
    }

    #[test]
    fn test_struct_field_type_mismatch() {
        let errs = check_err(
            "struct Point { x: f64, y: f64 }
             fn test() { let p = Point { x: \"hello\", y: 1.0 } }",
        );
        assert!(errs.iter().any(|d| d.message.contains("type mismatch")));
    }

    #[test]
    fn test_struct_missing_field() {
        let errs = check_err(
            "struct Point { x: f64, y: f64 }
             fn test() { let p = Point { x: 1.0 } }",
        );
        assert!(errs.iter().any(|d| d.message.contains("missing field")));
    }

    #[test]
    fn test_named_type_strict() {
        let errs = check_err(
            "struct Foo { x: i64 }
             struct Bar { x: i64 }
             fn takes_foo(f: Foo) -> i64 { return 0 }
             fn test() {
                 let b = Bar { x: 1 }
                 takes_foo(b)
             }",
        );
        assert!(errs.iter().any(|d| d.message.contains("type mismatch")));
    }
}
