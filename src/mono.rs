//! Monomorphization pass: specializes generic functions for concrete type arguments.
//!
//! Walks the AST to find `TypeCall` sites with concrete type arguments,
//! generates specialized copies of generic functions, and rewrites call sites.

use crate::ast::*;
use crate::lexer::Span;
use std::collections::{HashMap, HashSet};

/// Monomorphize a program: find all generic call sites, generate specialized
/// copies of functions, and rewrite call sites to use them.
pub fn monomorphize(program: &Program) -> Program {
    let mut mono = MonoPass::new();
    mono.collect_generic_fns(program);
    mono.collect_call_sites(program);
    mono.generate_specializations();
    mono.rewrite_program(program)
}

struct MonoPass {
    /// Generic function definitions by name
    generic_fns: HashMap<String, Function>,
    /// Set of (fn_name, vec_of_type_args) we need to specialize
    needed: Vec<(String, Vec<TypeArg>)>,
    /// Already-generated specialization names
    generated: HashSet<String>,
    /// Specialized functions to add
    specialized: Vec<Function>,
}

impl MonoPass {
    fn new() -> Self {
        Self {
            generic_fns: HashMap::new(),
            needed: Vec::new(),
            generated: HashSet::new(),
            specialized: Vec::new(),
        }
    }

    /// Collect all generic function definitions.
    fn collect_generic_fns(&mut self, program: &Program) {
        for item in &program.items {
            if let ItemKind::Function(func) = &item.kind {
                if !func.generics.is_empty() {
                    self.generic_fns.insert(func.name.name.clone(), func.clone());
                }
            }
        }
    }

    /// Walk the AST to find all TypeCall sites that reference generic functions.
    fn collect_call_sites(&mut self, program: &Program) {
        for item in &program.items {
            match &item.kind {
                ItemKind::Function(func) => {
                    self.visit_block(&func.body);
                }
                ItemKind::Kernel(kernel) => {
                    self.visit_block(&kernel.body);
                }
                ItemKind::Impl(imp) => {
                    for method in &imp.methods {
                        if let ItemKind::Function(func) = &method.kind {
                            self.visit_block(&func.body);
                        }
                    }
                }
                _ => {}
            }
        }
    }

    fn visit_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.visit_stmt(stmt);
        }
        if let Some(expr) = &block.expr {
            self.visit_expr(expr);
        }
    }

    fn visit_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Var { value, .. } => {
                self.visit_expr(value);
            }
            StmtKind::Return(Some(expr)) => self.visit_expr(expr),
            StmtKind::Return(None) => {}
            StmtKind::Expr(expr) => self.visit_expr(expr),
            StmtKind::Assign { target, value, .. } => {
                self.visit_expr(target);
                self.visit_expr(value);
            }
            StmtKind::For { iter, body, .. } => {
                self.visit_expr(iter);
                self.visit_block(body);
            }
            StmtKind::While { cond, body } => {
                self.visit_expr(cond);
                self.visit_block(body);
            }
            StmtKind::Break | StmtKind::Continue => {}
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::TypeCall { ty, method: _, args } => {
                // For function-style generic calls like `dot<f32>(x, y)`, the parser
                // produces TypeCall with ty being Generic { name: "dot", args: [...] }
                if let TypeExprKind::Generic { name, args: type_args } = &ty.kind {
                    if self.generic_fns.contains_key(&name.name) {
                        self.needed.push((name.name.clone(), type_args.clone()));
                    }
                }
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Call { func, args } => {
                self.visit_expr(func);
                for arg in args {
                    self.visit_expr(arg);
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::Unary { expr, .. } => self.visit_expr(expr),
            ExprKind::MatMul { lhs, rhs } => {
                self.visit_expr(lhs);
                self.visit_expr(rhs);
            }
            ExprKind::FieldAccess { base, .. } => self.visit_expr(base),
            ExprKind::Index { base, indices } => {
                self.visit_expr(base);
                for idx in indices {
                    self.visit_expr(idx);
                }
            }
            ExprKind::Block(block) => self.visit_block(block),
            ExprKind::If { cond, then_block, else_block } => {
                self.visit_expr(cond);
                self.visit_block(then_block);
                if let Some(eb) = else_block {
                    self.visit_block(eb);
                }
            }
            ExprKind::Range { start, end } => {
                self.visit_expr(start);
                self.visit_expr(end);
            }
            ExprKind::ArrayLiteral(elems) => {
                for e in elems {
                    self.visit_expr(e);
                }
            }
            ExprKind::Cast { expr, .. } => self.visit_expr(expr),
            ExprKind::StructLiteral { fields, .. } => {
                for (_, fval) in fields {
                    self.visit_expr(fval);
                }
            }
            ExprKind::Match { expr, arms } => {
                self.visit_expr(expr);
                for arm in arms {
                    self.visit_expr(&arm.body);
                }
            }
            _ => {}
        }
    }

    /// Generate specialized function copies for all collected call sites.
    fn generate_specializations(&mut self) {
        for (fn_name, type_args) in std::mem::take(&mut self.needed) {
            let mangled = mangle_name(&fn_name, &type_args);
            if self.generated.contains(&mangled) {
                continue;
            }
            if let Some(generic_fn) = self.generic_fns.get(&fn_name) {
                let specialized = specialize_function(generic_fn, &type_args);
                self.generated.insert(mangled);
                self.specialized.push(specialized);
            }
        }
    }

    /// Produce a new Program with specialized functions added and call sites rewritten.
    fn rewrite_program(&self, program: &Program) -> Program {
        let mut items: Vec<Item> = Vec::new();

        for item in &program.items {
            items.push(self.rewrite_item(item));
        }

        // Append specialized functions
        for func in &self.specialized {
            items.push(Item {
                kind: ItemKind::Function(func.clone()),
                span: func.name.span.clone(),
                is_pub: false,
            });
        }

        Program { items }
    }

    fn rewrite_item(&self, item: &Item) -> Item {
        let kind = match &item.kind {
            ItemKind::Function(func) => {
                ItemKind::Function(self.rewrite_function(func))
            }
            ItemKind::Kernel(kernel) => {
                let mut k = kernel.clone();
                k.body = self.rewrite_block(&kernel.body);
                ItemKind::Kernel(k)
            }
            ItemKind::Impl(imp) => {
                let methods = imp.methods.iter().map(|m| self.rewrite_item(m)).collect();
                ItemKind::Impl(ImplBlock {
                    trait_name: imp.trait_name.clone(),
                    target: imp.target.clone(),
                    methods,
                })
            }
            other => other.clone(),
        };
        Item {
            kind,
            span: item.span.clone(),
            is_pub: item.is_pub,
        }
    }

    fn rewrite_function(&self, func: &Function) -> Function {
        Function {
            name: func.name.clone(),
            generics: func.generics.clone(),
            params: func.params.clone(),
            ret_type: func.ret_type.clone(),
            where_clause: func.where_clause.clone(),
            body: self.rewrite_block(&func.body),
        }
    }

    fn rewrite_block(&self, block: &Block) -> Block {
        Block {
            stmts: block.stmts.iter().map(|s| self.rewrite_stmt(s)).collect(),
            expr: block.expr.as_ref().map(|e| Box::new(self.rewrite_expr(e))),
            span: block.span.clone(),
        }
    }

    fn rewrite_stmt(&self, stmt: &Stmt) -> Stmt {
        let kind = match &stmt.kind {
            StmtKind::Let { name, ty, value } => StmtKind::Let {
                name: name.clone(),
                ty: ty.clone(),
                value: self.rewrite_expr(value),
            },
            StmtKind::Var { name, ty, value } => StmtKind::Var {
                name: name.clone(),
                ty: ty.clone(),
                value: self.rewrite_expr(value),
            },
            StmtKind::Return(Some(expr)) => StmtKind::Return(Some(self.rewrite_expr(expr))),
            StmtKind::Return(None) => StmtKind::Return(None),
            StmtKind::Expr(expr) => StmtKind::Expr(self.rewrite_expr(expr)),
            StmtKind::Assign { target, op, value } => StmtKind::Assign {
                target: self.rewrite_expr(target),
                op: *op,
                value: self.rewrite_expr(value),
            },
            StmtKind::For { var, iter, body } => StmtKind::For {
                var: var.clone(),
                iter: self.rewrite_expr(iter),
                body: self.rewrite_block(body),
            },
            StmtKind::While { cond, body } => StmtKind::While {
                cond: self.rewrite_expr(cond),
                body: self.rewrite_block(body),
            },
            StmtKind::Break => StmtKind::Break,
            StmtKind::Continue => StmtKind::Continue,
        };
        Stmt { kind, span: stmt.span.clone() }
    }

    fn rewrite_expr(&self, expr: &Expr) -> Expr {
        let kind = match &expr.kind {
            ExprKind::TypeCall { ty, method, args } => {
                if let TypeExprKind::Generic { name, args: type_args } = &ty.kind {
                    if self.generic_fns.contains_key(&name.name) {
                        let mangled = mangle_name(&name.name, type_args);
                        let rewritten_args: Vec<Expr> = args.iter()
                            .map(|a| self.rewrite_expr(a))
                            .collect();
                        return Expr {
                            kind: ExprKind::Call {
                                func: Box::new(Expr {
                                    kind: ExprKind::Ident(Ident::new(mangled, name.span.clone())),
                                    span: name.span.clone(),
                                }),
                                args: rewritten_args,
                            },
                            span: expr.span.clone(),
                        };
                    }
                }
                ExprKind::TypeCall {
                    ty: ty.clone(),
                    method: method.clone(),
                    args: args.iter().map(|a| self.rewrite_expr(a)).collect(),
                }
            }
            ExprKind::Call { func, args } => ExprKind::Call {
                func: Box::new(self.rewrite_expr(func)),
                args: args.iter().map(|a| self.rewrite_expr(a)).collect(),
            },
            ExprKind::Binary { lhs, op, rhs } => ExprKind::Binary {
                lhs: Box::new(self.rewrite_expr(lhs)),
                op: *op,
                rhs: Box::new(self.rewrite_expr(rhs)),
            },
            ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
                op: *op,
                expr: Box::new(self.rewrite_expr(inner)),
            },
            ExprKind::MatMul { lhs, rhs } => ExprKind::MatMul {
                lhs: Box::new(self.rewrite_expr(lhs)),
                rhs: Box::new(self.rewrite_expr(rhs)),
            },
            ExprKind::FieldAccess { base, field } => ExprKind::FieldAccess {
                base: Box::new(self.rewrite_expr(base)),
                field: field.clone(),
            },
            ExprKind::Index { base, indices } => ExprKind::Index {
                base: Box::new(self.rewrite_expr(base)),
                indices: indices.iter().map(|i| self.rewrite_expr(i)).collect(),
            },
            ExprKind::Block(block) => ExprKind::Block(self.rewrite_block(block)),
            ExprKind::If { cond, then_block, else_block } => ExprKind::If {
                cond: Box::new(self.rewrite_expr(cond)),
                then_block: self.rewrite_block(then_block),
                else_block: else_block.as_ref().map(|eb| self.rewrite_block(eb)),
            },
            ExprKind::Range { start, end } => ExprKind::Range {
                start: Box::new(self.rewrite_expr(start)),
                end: Box::new(self.rewrite_expr(end)),
            },
            ExprKind::ArrayLiteral(elems) => ExprKind::ArrayLiteral(
                elems.iter().map(|e| self.rewrite_expr(e)).collect(),
            ),
            ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
                expr: Box::new(self.rewrite_expr(inner)),
                ty: ty.clone(),
            },
            ExprKind::StructLiteral { name, fields } => ExprKind::StructLiteral {
                name: name.clone(),
                fields: fields.iter().map(|(n, v)| (n.clone(), self.rewrite_expr(v))).collect(),
            },
            ExprKind::Match { expr: inner, arms } => ExprKind::Match {
                expr: Box::new(self.rewrite_expr(inner)),
                arms: arms.iter().map(|arm| MatchArm {
                    pattern: arm.pattern.clone(),
                    body: self.rewrite_expr(&arm.body),
                    span: arm.span.clone(),
                }).collect(),
            },
            other => other.clone(),
        };
        Expr { kind, span: expr.span.clone() }
    }
}

/// Build the mangled name: `fnname_arg1_arg2`
fn mangle_name(fn_name: &str, type_args: &[TypeArg]) -> String {
    let mut name = fn_name.to_string();
    for arg in type_args {
        name.push('_');
        name.push_str(&mangle_type_arg(arg));
    }
    name
}

fn mangle_type_arg(arg: &TypeArg) -> String {
    match arg {
        TypeArg::Type(ty) => mangle_type(ty),
        TypeArg::Expr(expr) => mangle_expr(expr),
        TypeArg::Shape(dims) => {
            let parts: Vec<String> = dims.iter().map(mangle_shape_dim).collect();
            parts.join("x")
        }
    }
}

fn mangle_type(ty: &TypeExpr) -> String {
    match &ty.kind {
        TypeExprKind::Named(id) => id.name.clone(),
        TypeExprKind::Generic { name, args } => {
            let mut s = name.name.clone();
            for a in args {
                s.push('_');
                s.push_str(&mangle_type_arg(a));
            }
            s
        }
        TypeExprKind::Array { elem, size } => {
            format!("arr_{}_{}", mangle_type(elem), mangle_expr(size))
        }
        TypeExprKind::Ref { mutable, inner } => {
            if *mutable { format!("mutref_{}", mangle_type(inner)) }
            else { format!("ref_{}", mangle_type(inner)) }
        }
        TypeExprKind::Tuple(types) => {
            let parts: Vec<String> = types.iter().map(mangle_type).collect();
            format!("tup_{}", parts.join("_"))
        }
        TypeExprKind::Shape(dims) => {
            let parts: Vec<String> = dims.iter().map(mangle_shape_dim).collect();
            parts.join("x")
        }
        TypeExprKind::Fn { .. } => "fn".to_string(),
    }
}

fn mangle_expr(expr: &Expr) -> String {
    match &expr.kind {
        ExprKind::IntLiteral(n) => n.to_string(),
        ExprKind::Ident(id) => id.name.clone(),
        _ => "expr".to_string(),
    }
}

fn mangle_shape_dim(dim: &ShapeDim) -> String {
    match dim {
        ShapeDim::Lit(n) => n.to_string(),
        ShapeDim::Ident(id) => id.name.clone(),
        ShapeDim::Expr(e) => mangle_expr(e),
        ShapeDim::Dynamic => "dyn".to_string(),
    }
}

/// Create a specialized copy of a generic function with type parameters substituted.
pub fn specialize_function(func: &Function, type_args: &[TypeArg]) -> Function {
    let mut type_subst: HashMap<String, TypeExpr> = HashMap::new();
    let mut const_subst: HashMap<String, Expr> = HashMap::new();

    for (param, arg) in func.generics.iter().zip(type_args.iter()) {
        match (&param.kind, arg) {
            (GenericParamKind::Type { .. }, TypeArg::Type(ty)) => {
                type_subst.insert(param.name.name.clone(), ty.clone());
            }
            (GenericParamKind::Const { .. }, TypeArg::Expr(expr)) => {
                const_subst.insert(param.name.name.clone(), expr.clone());
            }
            _ => {}
        }
    }

    let mangled = mangle_name(&func.name.name, type_args);

    Function {
        name: Ident::new(mangled, func.name.span.clone()),
        generics: vec![],
        params: func.params.iter().map(|p| subst_param(p, &type_subst)).collect(),
        ret_type: func.ret_type.as_ref().map(|t| subst_type(t, &type_subst)),
        where_clause: vec![],
        body: subst_block(&func.body, &type_subst, &const_subst),
    }
}

fn subst_param(param: &Param, type_subst: &HashMap<String, TypeExpr>) -> Param {
    Param {
        name: param.name.clone(),
        ty: subst_type(&param.ty, type_subst),
        default: param.default.clone(),
        span: param.span.clone(),
    }
}

fn subst_type(ty: &TypeExpr, type_subst: &HashMap<String, TypeExpr>) -> TypeExpr {
    match &ty.kind {
        TypeExprKind::Named(id) => {
            if let Some(replacement) = type_subst.get(&id.name) {
                replacement.clone()
            } else {
                ty.clone()
            }
        }
        TypeExprKind::Generic { name, args } => {
            if let Some(replacement) = type_subst.get(&name.name) {
                return replacement.clone();
            }
            TypeExpr {
                kind: TypeExprKind::Generic {
                    name: name.clone(),
                    args: args.iter().map(|a| subst_type_arg(a, type_subst)).collect(),
                },
                span: ty.span.clone(),
            }
        }
        TypeExprKind::Array { elem, size } => TypeExpr {
            kind: TypeExprKind::Array {
                elem: Box::new(subst_type(elem, type_subst)),
                size: size.clone(),
            },
            span: ty.span.clone(),
        },
        TypeExprKind::Ref { mutable, inner } => TypeExpr {
            kind: TypeExprKind::Ref {
                mutable: *mutable,
                inner: Box::new(subst_type(inner, type_subst)),
            },
            span: ty.span.clone(),
        },
        TypeExprKind::Tuple(types) => TypeExpr {
            kind: TypeExprKind::Tuple(types.iter().map(|t| subst_type(t, type_subst)).collect()),
            span: ty.span.clone(),
        },
        _ => ty.clone(),
    }
}

fn subst_type_arg(arg: &TypeArg, type_subst: &HashMap<String, TypeExpr>) -> TypeArg {
    match arg {
        TypeArg::Type(ty) => TypeArg::Type(subst_type(ty, type_subst)),
        other => other.clone(),
    }
}

fn subst_block(
    block: &Block,
    type_subst: &HashMap<String, TypeExpr>,
    const_subst: &HashMap<String, Expr>,
) -> Block {
    Block {
        stmts: block.stmts.iter().map(|s| subst_stmt(s, type_subst, const_subst)).collect(),
        expr: block.expr.as_ref().map(|e| Box::new(subst_expr(e, type_subst, const_subst))),
        span: block.span.clone(),
    }
}

fn subst_stmt(
    stmt: &Stmt,
    type_subst: &HashMap<String, TypeExpr>,
    const_subst: &HashMap<String, Expr>,
) -> Stmt {
    let kind = match &stmt.kind {
        StmtKind::Let { name, ty, value } => StmtKind::Let {
            name: name.clone(),
            ty: ty.as_ref().map(|t| subst_type(t, type_subst)),
            value: subst_expr(value, type_subst, const_subst),
        },
        StmtKind::Var { name, ty, value } => StmtKind::Var {
            name: name.clone(),
            ty: ty.as_ref().map(|t| subst_type(t, type_subst)),
            value: subst_expr(value, type_subst, const_subst),
        },
        StmtKind::Return(Some(expr)) => StmtKind::Return(Some(subst_expr(expr, type_subst, const_subst))),
        StmtKind::Return(None) => StmtKind::Return(None),
        StmtKind::Expr(expr) => StmtKind::Expr(subst_expr(expr, type_subst, const_subst)),
        StmtKind::Assign { target, op, value } => StmtKind::Assign {
            target: subst_expr(target, type_subst, const_subst),
            op: *op,
            value: subst_expr(value, type_subst, const_subst),
        },
        StmtKind::For { var, iter, body } => StmtKind::For {
            var: var.clone(),
            iter: subst_expr(iter, type_subst, const_subst),
            body: subst_block(body, type_subst, const_subst),
        },
        StmtKind::While { cond, body } => StmtKind::While {
            cond: subst_expr(cond, type_subst, const_subst),
            body: subst_block(body, type_subst, const_subst),
        },
        StmtKind::Break => StmtKind::Break,
        StmtKind::Continue => StmtKind::Continue,
    };
    Stmt { kind, span: stmt.span.clone() }
}

fn subst_expr(
    expr: &Expr,
    type_subst: &HashMap<String, TypeExpr>,
    const_subst: &HashMap<String, Expr>,
) -> Expr {
    let kind = match &expr.kind {
        ExprKind::Ident(id) => {
            if let Some(replacement) = const_subst.get(&id.name) {
                return replacement.clone();
            }
            expr.kind.clone()
        }
        ExprKind::Binary { lhs, op, rhs } => ExprKind::Binary {
            lhs: Box::new(subst_expr(lhs, type_subst, const_subst)),
            op: *op,
            rhs: Box::new(subst_expr(rhs, type_subst, const_subst)),
        },
        ExprKind::Unary { op, expr: inner } => ExprKind::Unary {
            op: *op,
            expr: Box::new(subst_expr(inner, type_subst, const_subst)),
        },
        ExprKind::Call { func, args } => ExprKind::Call {
            func: Box::new(subst_expr(func, type_subst, const_subst)),
            args: args.iter().map(|a| subst_expr(a, type_subst, const_subst)).collect(),
        },
        ExprKind::MatMul { lhs, rhs } => ExprKind::MatMul {
            lhs: Box::new(subst_expr(lhs, type_subst, const_subst)),
            rhs: Box::new(subst_expr(rhs, type_subst, const_subst)),
        },
        ExprKind::FieldAccess { base, field } => ExprKind::FieldAccess {
            base: Box::new(subst_expr(base, type_subst, const_subst)),
            field: field.clone(),
        },
        ExprKind::Index { base, indices } => ExprKind::Index {
            base: Box::new(subst_expr(base, type_subst, const_subst)),
            indices: indices.iter().map(|i| subst_expr(i, type_subst, const_subst)).collect(),
        },
        ExprKind::Block(block) => ExprKind::Block(subst_block(block, type_subst, const_subst)),
        ExprKind::If { cond, then_block, else_block } => ExprKind::If {
            cond: Box::new(subst_expr(cond, type_subst, const_subst)),
            then_block: subst_block(then_block, type_subst, const_subst),
            else_block: else_block.as_ref().map(|eb| subst_block(eb, type_subst, const_subst)),
        },
        ExprKind::Range { start, end } => ExprKind::Range {
            start: Box::new(subst_expr(start, type_subst, const_subst)),
            end: Box::new(subst_expr(end, type_subst, const_subst)),
        },
        ExprKind::ArrayLiteral(elems) => ExprKind::ArrayLiteral(
            elems.iter().map(|e| subst_expr(e, type_subst, const_subst)).collect(),
        ),
        ExprKind::TypeCall { ty, method, args } => ExprKind::TypeCall {
            ty: subst_type(ty, type_subst),
            method: method.clone(),
            args: args.iter().map(|a| subst_expr(a, type_subst, const_subst)).collect(),
        },
        ExprKind::Cast { expr: inner, ty } => ExprKind::Cast {
            expr: Box::new(subst_expr(inner, type_subst, const_subst)),
            ty: subst_type(ty, type_subst),
        },
        ExprKind::StructLiteral { name, fields } => ExprKind::StructLiteral {
            name: name.clone(),
            fields: fields.iter().map(|(n, v)| (n.clone(), subst_expr(v, type_subst, const_subst))).collect(),
        },
        ExprKind::Match { expr: inner, arms } => ExprKind::Match {
            expr: Box::new(subst_expr(inner, type_subst, const_subst)),
            arms: arms.iter().map(|arm| MatchArm {
                pattern: arm.pattern.clone(),
                body: subst_expr(&arm.body, type_subst, const_subst),
                span: arm.span.clone(),
            }).collect(),
        },
        _ => expr.kind.clone(),
    };
    Expr { kind, span: expr.span.clone() }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dummy_span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn make_ident(name: &str) -> Ident {
        Ident::new(name.to_string(), dummy_span())
    }

    fn make_named_type(name: &str) -> TypeExpr {
        TypeExpr {
            kind: TypeExprKind::Named(make_ident(name)),
            span: dummy_span(),
        }
    }

    fn make_int_expr(n: u128) -> Expr {
        Expr { kind: ExprKind::IntLiteral(n), span: dummy_span() }
    }

    fn make_ident_expr(name: &str) -> Expr {
        Expr { kind: ExprKind::Ident(make_ident(name)), span: dummy_span() }
    }

    /// Helper: build a generic function `fn dot<T>(a: T, b: T) -> T { return a }`
    fn make_generic_dot() -> Function {
        Function {
            name: make_ident("dot"),
            generics: vec![GenericParam {
                name: make_ident("T"),
                kind: GenericParamKind::Type { bounds: vec![] },
                span: dummy_span(),
            }],
            params: vec![
                Param { name: make_ident("a"), ty: make_named_type("T"), default: None, span: dummy_span() },
                Param { name: make_ident("b"), ty: make_named_type("T"), default: None, span: dummy_span() },
            ],
            ret_type: Some(make_named_type("T")),
            where_clause: vec![],
            body: Block {
                stmts: vec![Stmt {
                    kind: StmtKind::Return(Some(make_ident_expr("a"))),
                    span: dummy_span(),
                }],
                expr: None,
                span: dummy_span(),
            },
        }
    }

    #[test]
    fn test_specialize_function() {
        let func = make_generic_dot();
        let type_args = vec![TypeArg::Type(make_named_type("f32"))];
        let specialized = specialize_function(&func, &type_args);

        assert_eq!(specialized.name.name, "dot_f32");
        assert!(specialized.generics.is_empty());
        assert_eq!(specialized.params.len(), 2);
        if let TypeExprKind::Named(id) = &specialized.params[0].ty.kind {
            assert_eq!(id.name, "f32");
        } else {
            panic!("Expected Named type for param");
        }
        if let Some(ret) = &specialized.ret_type {
            if let TypeExprKind::Named(id) = &ret.kind {
                assert_eq!(id.name, "f32");
            } else {
                panic!("Expected Named return type");
            }
        } else {
            panic!("Expected return type");
        }
    }

    #[test]
    fn test_monomorphize_program() {
        let dot_fn = make_generic_dot();

        let call_expr = Expr {
            kind: ExprKind::TypeCall {
                ty: TypeExpr {
                    kind: TypeExprKind::Generic {
                        name: make_ident("dot"),
                        args: vec![TypeArg::Type(make_named_type("f32"))],
                    },
                    span: dummy_span(),
                },
                method: make_ident(""),
                args: vec![
                    Expr { kind: ExprKind::FloatLiteral(1.0), span: dummy_span() },
                    Expr { kind: ExprKind::FloatLiteral(2.0), span: dummy_span() },
                ],
            },
            span: dummy_span(),
        };

        let main_fn = Function {
            name: make_ident("main"),
            generics: vec![],
            params: vec![],
            ret_type: None,
            where_clause: vec![],
            body: Block {
                stmts: vec![Stmt {
                    kind: StmtKind::Expr(call_expr),
                    span: dummy_span(),
                }],
                expr: None,
                span: dummy_span(),
            },
        };

        let program = Program {
            items: vec![
                Item { kind: ItemKind::Function(dot_fn), span: dummy_span(), is_pub: false },
                Item { kind: ItemKind::Function(main_fn), span: dummy_span(), is_pub: false },
            ],
        };

        let result = monomorphize(&program);

        assert_eq!(result.items.len(), 3);

        if let ItemKind::Function(func) = &result.items[2].kind {
            assert_eq!(func.name.name, "dot_f32");
            assert!(func.generics.is_empty());
        } else {
            panic!("Expected specialized function as last item");
        }

        if let ItemKind::Function(main) = &result.items[1].kind {
            if let StmtKind::Expr(expr) = &main.body.stmts[0].kind {
                if let ExprKind::Call { func, args } = &expr.kind {
                    if let ExprKind::Ident(id) = &func.kind {
                        assert_eq!(id.name, "dot_f32");
                    } else {
                        panic!("Expected Ident call target, got {:?}", func.kind);
                    }
                    assert_eq!(args.len(), 2);
                } else {
                    panic!("Expected Call expr, got {:?}", expr.kind);
                }
            } else {
                panic!("Expected Expr stmt");
            }
        } else {
            panic!("Expected main function");
        }
    }

    #[test]
    fn test_const_generic_substitution() {
        let func = Function {
            name: make_ident("zeros"),
            generics: vec![
                GenericParam {
                    name: make_ident("T"),
                    kind: GenericParamKind::Type { bounds: vec![] },
                    span: dummy_span(),
                },
                GenericParam {
                    name: make_ident("N"),
                    kind: GenericParamKind::Const { ty: make_named_type("usize") },
                    span: dummy_span(),
                },
            ],
            params: vec![],
            ret_type: Some(make_named_type("T")),
            where_clause: vec![],
            body: Block {
                stmts: vec![Stmt {
                    kind: StmtKind::Return(Some(make_ident_expr("N"))),
                    span: dummy_span(),
                }],
                expr: None,
                span: dummy_span(),
            },
        };

        let type_args = vec![
            TypeArg::Type(make_named_type("f32")),
            TypeArg::Expr(make_int_expr(128)),
        ];

        let specialized = specialize_function(&func, &type_args);

        assert_eq!(specialized.name.name, "zeros_f32_128");
        assert!(specialized.generics.is_empty());

        if let Some(ret) = &specialized.ret_type {
            if let TypeExprKind::Named(id) = &ret.kind {
                assert_eq!(id.name, "f32");
            } else {
                panic!("Expected Named return type");
            }
        }

        if let StmtKind::Return(Some(expr)) = &specialized.body.stmts[0].kind {
            if let ExprKind::IntLiteral(n) = &expr.kind {
                assert_eq!(*n, 128);
            } else {
                panic!("Expected IntLiteral(128), got {:?}", expr.kind);
            }
        } else {
            panic!("Expected Return statement");
        }
    }
}
