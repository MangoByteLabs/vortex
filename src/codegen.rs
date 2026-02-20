//! MLIR code generation from Vortex AST.
//!
//! Emits textual MLIR IR targeting the following dialects:
//! - `func` for functions
//! - `arith` for arithmetic
//! - `scf` for structured control flow (if/else, for, while)
//! - `gpu` for GPU kernel dispatch
//! - `memref` for memory references

use crate::ast::*;
use std::collections::HashMap;
use std::fmt::Write;

/// MLIR type representation
#[derive(Debug, Clone, PartialEq)]
enum MLIRType {
    I1,
    I8,
    I16,
    I32,
    I64,
    I128,
    F16,
    F32,
    F64,
    Index,
    MemRef(Box<MLIRType>, Vec<i64>), // memref<shape x type>
    None, // void
}

impl std::fmt::Display for MLIRType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MLIRType::I1 => write!(f, "i1"),
            MLIRType::I8 => write!(f, "i8"),
            MLIRType::I16 => write!(f, "i16"),
            MLIRType::I32 => write!(f, "i32"),
            MLIRType::I64 => write!(f, "i64"),
            MLIRType::I128 => write!(f, "i128"),
            MLIRType::F16 => write!(f, "f16"),
            MLIRType::F32 => write!(f, "f32"),
            MLIRType::F64 => write!(f, "f64"),
            MLIRType::Index => write!(f, "index"),
            MLIRType::MemRef(elem, shape) => {
                write!(f, "memref<")?;
                for dim in shape {
                    if *dim < 0 {
                        write!(f, "?x")?;
                    } else {
                        write!(f, "{}x", dim)?;
                    }
                }
                write!(f, "{}>", elem)
            }
            MLIRType::None => write!(f, "()"),
        }
    }
}

/// Code generator state
pub struct CodeGen {
    output: String,
    indent: usize,
    /// SSA value counter for generating unique names
    ssa_counter: usize,
    /// Map from Vortex variable names to MLIR SSA values
    var_map: Vec<HashMap<String, (String, MLIRType)>>,
    /// Enum variant tag assignments: "EnumName::Variant" -> tag (i32)
    enum_tags: HashMap<String, i32>,
}

impl CodeGen {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            ssa_counter: 0,
            var_map: vec![HashMap::new()],
            enum_tags: HashMap::new(),
        }
    }

    /// Generate MLIR IR from a Vortex program
    pub fn generate(&mut self, program: &Program) -> String {
        // Emit module wrapper
        self.emit_line("module {");
        self.indent += 1;

        // First pass: register enum tags
        for item in &program.items {
            if let ItemKind::Enum(e) = &item.kind {
                for (i, variant) in e.variants.iter().enumerate() {
                    let key = format!("{}::{}", e.name.name, variant.name.name);
                    self.enum_tags.insert(key, i as i32);
                }
            }
        }

        for item in &program.items {
            match &item.kind {
                ItemKind::Function(func) => self.gen_function(func),
                ItemKind::Kernel(kernel) => self.gen_kernel(kernel),
                ItemKind::Enum(e) => self.gen_enum(e),
                _ => {
                    // Skip structs, traits, etc. for now
                }
            }
        }

        self.indent -= 1;
        self.emit_line("}");

        self.output.clone()
    }

    // --- Helpers ---

    fn fresh_ssa(&mut self) -> String {
        let name = format!("%{}", self.ssa_counter);
        self.ssa_counter += 1;
        name
    }

    fn emit(&mut self, s: &str) {
        let indent_str = "  ".repeat(self.indent);
        let _ = write!(self.output, "{}{}", indent_str, s);
    }

    fn emit_line(&mut self, s: &str) {
        let indent_str = "  ".repeat(self.indent);
        let _ = writeln!(self.output, "{}{}", indent_str, s);
    }

    fn emit_raw(&mut self, s: &str) {
        self.output.push_str(s);
    }

    fn push_scope(&mut self) {
        self.var_map.push(HashMap::new());
    }

    fn pop_scope(&mut self) {
        self.var_map.pop();
    }

    fn define_var(&mut self, name: &str, ssa: &str, ty: MLIRType) {
        if let Some(scope) = self.var_map.last_mut() {
            scope.insert(name.to_string(), (ssa.to_string(), ty));
        }
    }

    fn lookup_var(&self, name: &str) -> Option<(String, MLIRType)> {
        for scope in self.var_map.iter().rev() {
            if let Some(v) = scope.get(name) {
                return Some(v.clone());
            }
        }
        None
    }

    /// Convert Vortex type expression to MLIR type
    fn resolve_type(&self, ty: &TypeExpr) -> MLIRType {
        match &ty.kind {
            TypeExprKind::Named(id) => match id.name.as_str() {
                "i8" => MLIRType::I8,
                "i16" => MLIRType::I16,
                "i32" | "int" => MLIRType::I32,
                "i64" => MLIRType::I64,
                "i128" => MLIRType::I128,
                "u8" => MLIRType::I8,
                "u16" => MLIRType::I16,
                "u32" => MLIRType::I32,
                "u64" => MLIRType::I64,
                "u128" => MLIRType::I128,
                "f16" => MLIRType::F16,
                "f32" | "float" => MLIRType::F32,
                "f64" => MLIRType::F64,
                "bool" => MLIRType::I1,
                _ => MLIRType::I64, // default for unknown types
            },
            TypeExprKind::Generic { name, .. } => {
                match name.name.as_str() {
                    "Tensor" => {
                        // For now, emit as memref
                        MLIRType::MemRef(Box::new(MLIRType::F32), vec![-1])
                    }
                    _ => MLIRType::I64,
                }
            }
            _ => MLIRType::I64,
        }
    }

    /// Infer MLIR type from Vortex expression
    fn infer_type(&self, expr: &Expr) -> MLIRType {
        match &expr.kind {
            ExprKind::IntLiteral(_) => MLIRType::I64,
            ExprKind::FloatLiteral(_) => MLIRType::F64,
            ExprKind::BoolLiteral(_) => MLIRType::I1,
            ExprKind::StringLiteral(_) => MLIRType::I64, // placeholder
            ExprKind::Ident(id) => {
                self.lookup_var(&id.name)
                    .map(|(_, ty)| ty)
                    .unwrap_or(MLIRType::I64)
            }
            ExprKind::Binary { lhs, op, .. } => {
                match op {
                    BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt
                    | BinOp::LtEq | BinOp::GtEq | BinOp::And | BinOp::Or => MLIRType::I1,
                    _ => self.infer_type(lhs),
                }
            }
            ExprKind::Unary { expr: inner, .. } => self.infer_type(inner),
            ExprKind::Call { .. } => MLIRType::I64, // default
            ExprKind::If { then_block, .. } => {
                if let Some(expr) = &then_block.expr {
                    self.infer_type(expr)
                } else {
                    MLIRType::None
                }
            }
            ExprKind::Cast { ty, .. } => self.resolve_type(ty),
            _ => MLIRType::I64,
        }
    }

    fn is_float_type(ty: &MLIRType) -> bool {
        matches!(ty, MLIRType::F16 | MLIRType::F32 | MLIRType::F64)
    }

    fn is_int_type(ty: &MLIRType) -> bool {
        matches!(ty, MLIRType::I1 | MLIRType::I8 | MLIRType::I16
            | MLIRType::I32 | MLIRType::I64 | MLIRType::I128)
    }

    // --- Enum generation ---

    fn gen_enum(&mut self, e: &EnumDef) {
        // Emit enum as a tagged union comment + type alias
        // In MLIR, enums are represented as a struct { tag: i32, payload: i64 }
        self.emit_line(&format!("// enum {} {{", e.name.name));
        for (i, variant) in e.variants.iter().enumerate() {
            let fields_str = match &variant.kind {
                EnumVariantKind::Unit => String::new(),
                EnumVariantKind::Tuple(types) => {
                    let ts: Vec<String> = types.iter().map(|t| format!("{}", t)).collect();
                    format!("({})", ts.join(", "))
                }
                EnumVariantKind::Struct(fields) => {
                    let fs: Vec<String> = fields.iter().map(|f| format!("{}: {}", f.name.name, f.ty)).collect();
                    format!(" {{ {} }}", fs.join(", "))
                }
            };
            self.emit_line(&format!("//   {} = {}{}", i, variant.name.name, fields_str));
        }
        self.emit_line(&format!("// }} (tag: i32, payload: i64)"));
        self.emit_line("");
    }

    // --- Function generation ---

    fn gen_function(&mut self, func: &Function) {
        self.push_scope();

        // Build parameter list
        let mut param_strs = Vec::new();
        let mut param_types = Vec::new();
        for param in &func.params {
            let ty = self.resolve_type(&param.ty);
            let ssa = self.fresh_ssa();
            self.define_var(&param.name.name, &ssa, ty.clone());
            param_strs.push(format!("{}: {}", ssa, ty));
            param_types.push(ty);
        }

        // Return type
        let ret_type = func.ret_type.as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(MLIRType::None);

        // Emit function signature
        let params_str = param_strs.join(", ");
        let ret_str = if ret_type == MLIRType::None {
            String::new()
        } else {
            format!(" -> {}", ret_type)
        };

        self.emit_line(&format!(
            "func.func @{}({}){} {{",
            func.name.name, params_str, ret_str
        ));
        self.indent += 1;

        // Generate body
        self.gen_block(&func.body, &ret_type);

        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.pop_scope();
    }

    fn gen_kernel(&mut self, kernel: &Kernel) {
        self.push_scope();

        // Build parameter list
        let mut param_strs = Vec::new();
        for param in &kernel.params {
            let ty = self.resolve_type(&param.ty);
            let ssa = self.fresh_ssa();
            self.define_var(&param.name.name, &ssa, ty.clone());
            param_strs.push(format!("{}: {}", ssa, ty));
        }

        // Return type
        let ret_type = kernel.ret_type.as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(MLIRType::None);

        let params_str = param_strs.join(", ");
        let ret_str = if ret_type == MLIRType::None {
            String::new()
        } else {
            format!(" -> {}", ret_type)
        };

        // Emit as gpu.func inside a gpu.module
        self.emit_line(&format!("gpu.module @{}_module {{", kernel.name.name));
        self.indent += 1;
        self.emit_line(&format!(
            "gpu.func @{}({}){} kernel {{",
            kernel.name.name, params_str, ret_str
        ));
        self.indent += 1;

        self.gen_block(&kernel.body, &ret_type);

        // GPU functions need gpu.return
        self.emit_line("gpu.return");

        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.pop_scope();
    }

    // --- Block generation ---

    fn gen_block(&mut self, block: &Block, expected_ret: &MLIRType) {
        for stmt in &block.stmts {
            self.gen_stmt(stmt);
        }

        if let Some(expr) = &block.expr {
            let (ssa, _ty) = self.gen_expr(expr);
            if *expected_ret != MLIRType::None {
                self.emit_line(&format!("func.return {}", ssa));
            }
        }
    }

    // --- Statement generation ---

    fn gen_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, value, ty } | StmtKind::Var { name, value, ty } => {
                let (ssa, inferred_ty) = self.gen_expr(value);
                let actual_ty = ty.as_ref()
                    .map(|t| self.resolve_type(t))
                    .unwrap_or(inferred_ty);
                self.define_var(&name.name, &ssa, actual_ty);
            }

            StmtKind::Return(Some(expr)) => {
                let (ssa, _ty) = self.gen_expr(expr);
                self.emit_line(&format!("func.return {}", ssa));
            }
            StmtKind::Return(None) => {
                self.emit_line("func.return");
            }

            StmtKind::Expr(expr) => {
                self.gen_expr(expr);
            }

            StmtKind::Assign { target, op, value } => {
                let (rhs_ssa, rhs_ty) = self.gen_expr(value);
                match &target.kind {
                    ExprKind::Ident(id) => {
                        let final_ssa = match op {
                            AssignOp::Assign => rhs_ssa,
                            AssignOp::AddAssign => {
                                if let Some((lhs_ssa, _)) = self.lookup_var(&id.name) {
                                    self.emit_arith_op("add", &lhs_ssa, &rhs_ssa, &rhs_ty)
                                } else {
                                    rhs_ssa
                                }
                            }
                            AssignOp::SubAssign => {
                                if let Some((lhs_ssa, _)) = self.lookup_var(&id.name) {
                                    self.emit_arith_op("sub", &lhs_ssa, &rhs_ssa, &rhs_ty)
                                } else {
                                    rhs_ssa
                                }
                            }
                            AssignOp::MulAssign => {
                                if let Some((lhs_ssa, _)) = self.lookup_var(&id.name) {
                                    self.emit_arith_op("mul", &lhs_ssa, &rhs_ssa, &rhs_ty)
                                } else {
                                    rhs_ssa
                                }
                            }
                            AssignOp::DivAssign => {
                                if let Some((lhs_ssa, _)) = self.lookup_var(&id.name) {
                                    self.emit_arith_op("div", &lhs_ssa, &rhs_ssa, &rhs_ty)
                                } else {
                                    rhs_ssa
                                }
                            }
                            _ => rhs_ssa,
                        };
                        self.define_var(&id.name, &final_ssa, rhs_ty);
                    }
                    _ => {
                        // Complex assignment targets (index, field) — emit as comment
                        self.emit_line(&format!("// TODO: complex assignment target"));
                    }
                }
            }

            StmtKind::For { var, iter, body } => {
                self.gen_for_loop(var, iter, body);
            }

            StmtKind::While { cond, body } => {
                self.gen_while_loop(cond, body);
            }

            StmtKind::Break | StmtKind::Continue => {
                self.emit_line(&format!("// {}", if matches!(stmt.kind, StmtKind::Break) { "break" } else { "continue" }));
            }
        }
    }

    // --- Expression generation (returns SSA name and type) ---

    fn gen_expr(&mut self, expr: &Expr) -> (String, MLIRType) {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                let ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "{} = arith.constant {} : i64",
                    ssa, *n as i64
                ));
                (ssa, MLIRType::I64)
            }

            ExprKind::FloatLiteral(n) => {
                let ssa = self.fresh_ssa();
                // Format float properly for MLIR
                let float_str = if n.fract() == 0.0 {
                    format!("{:.1}", n)
                } else {
                    format!("{}", n)
                };
                self.emit_line(&format!(
                    "{} = arith.constant {} : f64",
                    ssa, float_str
                ));
                (ssa, MLIRType::F64)
            }

            ExprKind::BoolLiteral(b) => {
                let ssa = self.fresh_ssa();
                let val = if *b { 1 } else { 0 };
                self.emit_line(&format!(
                    "{} = arith.constant {} : i1",
                    ssa, val
                ));
                (ssa, MLIRType::I1)
            }

            ExprKind::StringLiteral(s) => {
                // Strings are emitted as a comment + constant for now
                let ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "// string literal: \"{}\"", s
                ));
                self.emit_line(&format!(
                    "{} = arith.constant 0 : i64",
                    ssa
                ));
                (ssa, MLIRType::I64)
            }

            ExprKind::Ident(id) => {
                if let Some((ssa, ty)) = self.lookup_var(&id.name) {
                    (ssa, ty)
                } else {
                    // Undefined variable — emit a placeholder
                    let ssa = self.fresh_ssa();
                    self.emit_line(&format!(
                        "// undefined: {}", id.name
                    ));
                    self.emit_line(&format!(
                        "{} = arith.constant 0 : i64",
                        ssa
                    ));
                    (ssa, MLIRType::I64)
                }
            }

            ExprKind::Binary { lhs, op, rhs } => {
                let (l_ssa, l_ty) = self.gen_expr(lhs);
                let (r_ssa, _r_ty) = self.gen_expr(rhs);
                self.gen_binop(&l_ssa, *op, &r_ssa, &l_ty)
            }

            ExprKind::Unary { op, expr: inner } => {
                let (inner_ssa, inner_ty) = self.gen_expr(inner);
                let ssa = self.fresh_ssa();
                match op {
                    UnaryOp::Neg => {
                        if Self::is_float_type(&inner_ty) {
                            self.emit_line(&format!(
                                "{} = arith.negf {} : {}",
                                ssa, inner_ssa, inner_ty
                            ));
                        } else {
                            // Integer negation: 0 - x
                            let zero = self.fresh_ssa();
                            self.emit_line(&format!(
                                "{} = arith.constant 0 : {}", zero, inner_ty
                            ));
                            self.emit_line(&format!(
                                "{} = arith.subi {}, {} : {}",
                                ssa, zero, inner_ssa, inner_ty
                            ));
                        }
                    }
                    UnaryOp::Not => {
                        // Boolean not: xor with true
                        let one = self.fresh_ssa();
                        self.emit_line(&format!(
                            "{} = arith.constant 1 : i1", one
                        ));
                        self.emit_line(&format!(
                            "{} = arith.xori {}, {} : i1",
                            ssa, inner_ssa, one
                        ));
                    }
                    UnaryOp::BitNot => {
                        // Bitwise not: xor with -1
                        let ones = self.fresh_ssa();
                        self.emit_line(&format!(
                            "{} = arith.constant -1 : {}", ones, inner_ty
                        ));
                        self.emit_line(&format!(
                            "{} = arith.xori {}, {} : {}",
                            ssa, inner_ssa, ones, inner_ty
                        ));
                    }
                }
                (ssa, inner_ty)
            }

            ExprKind::Call { func, args } => {
                // Generate arguments
                let mut arg_ssas = Vec::new();
                let mut arg_types = Vec::new();
                for arg in args {
                    let (ssa, ty) = self.gen_expr(arg);
                    arg_ssas.push(ssa);
                    arg_types.push(ty);
                }

                let func_name = match &func.kind {
                    ExprKind::Ident(id) => id.name.clone(),
                    _ => "unknown".to_string(),
                };

                let ssa = self.fresh_ssa();
                // Emit as func.call with i64 return type (simplified)
                self.emit_line(&format!(
                    "{} = func.call @{}({}) : ({}) -> i64",
                    ssa,
                    func_name,
                    arg_ssas.join(", "),
                    arg_types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ")
                ));

                (ssa, MLIRType::I64)
            }

            ExprKind::If { cond, then_block, else_block } => {
                self.gen_if_expr(cond, then_block, else_block)
            }

            ExprKind::Block(block) => {
                self.push_scope();
                for stmt in &block.stmts {
                    self.gen_stmt(stmt);
                }
                let result = if let Some(expr) = &block.expr {
                    self.gen_expr(expr)
                } else {
                    let ssa = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                    (ssa, MLIRType::I64)
                };
                self.pop_scope();
                result
            }

            ExprKind::Range { start, end } => {
                // Ranges are handled by for loop generation
                let (s_ssa, s_ty) = self.gen_expr(start);
                let (_e_ssa, _e_ty) = self.gen_expr(end);
                (s_ssa, s_ty)
            }

            ExprKind::ArrayLiteral(elems) => {
                // Emit each element
                let mut last_ssa = String::new();
                let mut elem_ty = MLIRType::I64;
                for elem in elems {
                    let (ssa, ty) = self.gen_expr(elem);
                    last_ssa = ssa;
                    elem_ty = ty;
                }
                if last_ssa.is_empty() {
                    let ssa = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                    (ssa, MLIRType::I64)
                } else {
                    (last_ssa, elem_ty)
                }
            }

            ExprKind::Cast { expr: inner, ty } => {
                let (inner_ssa, inner_ty) = self.gen_expr(inner);
                let target_ty = self.resolve_type(ty);
                let ssa = self.fresh_ssa();

                if inner_ty == target_ty {
                    return (inner_ssa, inner_ty);
                }

                // Generate appropriate cast
                if Self::is_int_type(&inner_ty) && Self::is_float_type(&target_ty) {
                    self.emit_line(&format!(
                        "{} = arith.sitofp {} : {} to {}",
                        ssa, inner_ssa, inner_ty, target_ty
                    ));
                } else if Self::is_float_type(&inner_ty) && Self::is_int_type(&target_ty) {
                    self.emit_line(&format!(
                        "{} = arith.fptosi {} : {} to {}",
                        ssa, inner_ssa, inner_ty, target_ty
                    ));
                } else if Self::is_float_type(&inner_ty) && Self::is_float_type(&target_ty) {
                    // Float to float (extend or truncate)
                    self.emit_line(&format!(
                        "{} = arith.extf {} : {} to {}",
                        ssa, inner_ssa, inner_ty, target_ty
                    ));
                } else {
                    // Int to int (extend or truncate)
                    self.emit_line(&format!(
                        "{} = arith.extsi {} : {} to {}",
                        ssa, inner_ssa, inner_ty, target_ty
                    ));
                }

                (ssa, target_ty)
            }

            ExprKind::Index { base, indices } => {
                let (base_ssa, base_ty) = self.gen_expr(base);
                let (idx_ssa, _) = self.gen_expr(&indices[0]);
                let ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "// index: {}[{}]", base_ssa, idx_ssa
                ));
                self.emit_line(&format!(
                    "{} = arith.constant 0 : i64", ssa
                ));
                (ssa, base_ty)
            }

            ExprKind::FieldAccess { base, field } => {
                let (base_ssa, _) = self.gen_expr(base);
                let ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "// field access: {}.{}", base_ssa, field.name
                ));
                self.emit_line(&format!(
                    "{} = arith.constant 0 : i64", ssa
                ));
                (ssa, MLIRType::I64)
            }

            ExprKind::MatMul { lhs, rhs } => {
                let (l_ssa, l_ty) = self.gen_expr(lhs);
                let (r_ssa, _) = self.gen_expr(rhs);
                let ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "// matmul: {} @ {}", l_ssa, r_ssa
                ));
                self.emit_line(&format!(
                    "{} = arith.constant 0 : i64", ssa
                ));
                (ssa, l_ty)
            }

            ExprKind::StructLiteral { name, fields } => {
                self.emit_line(&format!("// struct literal: {}", name.name));
                for (fname, fexpr) in fields {
                    let (ssa, _) = self.gen_expr(fexpr);
                    self.emit_line(&format!("// .{} = {}", fname.name, ssa));
                }
                let ssa = self.fresh_ssa();
                self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                (ssa, MLIRType::I64)
            }

            ExprKind::Match { expr: match_expr, arms } => {
                let (val_ssa, _val_ty) = self.gen_expr(match_expr);
                self.emit_line(&format!("// match on {}", val_ssa));

                // Extract tag from the matched value (assume first i32 is the tag)
                let tag_ssa = self.fresh_ssa();
                self.emit_line(&format!("// extract tag from enum value"));
                self.emit_line(&format!("{} = arith.constant 0 : i32", tag_ssa));

                // Emit arms as a chain of scf.if comparisons on the tag
                let mut result_ssa = String::new();
                let mut result_ty = MLIRType::I64;

                for (i, arm) in arms.iter().enumerate() {
                    let is_wildcard = matches!(&arm.pattern, Pattern::Wildcard | Pattern::Ident(_));

                    if is_wildcard {
                        // Default arm: just emit the body
                        self.emit_line("// wildcard arm");
                        let (ssa, ty) = self.gen_expr(&arm.body);
                        result_ssa = ssa;
                        result_ty = ty;
                    } else {
                        // Get the tag value for this variant pattern
                        let tag_val = if let Pattern::Variant { name: _, .. } = &arm.pattern {
                            // Look up the tag for this variant
                            self.enum_tags.values().next().copied().unwrap_or(i as i32)
                        } else {
                            i as i32
                        };

                        let expected_tag = self.fresh_ssa();
                        self.emit_line(&format!("{} = arith.constant {} : i32", expected_tag, tag_val));
                        let cmp_ssa = self.fresh_ssa();
                        self.emit_line(&format!(
                            "{} = arith.cmpi eq, {}, {} : i32",
                            cmp_ssa, tag_ssa, expected_tag
                        ));

                        // Emit the body inside scf.if
                        self.emit_line(&format!("scf.if {} {{", cmp_ssa));
                        self.indent += 1;

                        // Bind pattern variables
                        if let Pattern::Variant { name: _, fields } = &arm.pattern {
                            for (fi, field_pat) in fields.iter().enumerate() {
                                if let Pattern::Ident(id) = field_pat {
                                    let field_ssa = self.fresh_ssa();
                                    self.emit_line(&format!(
                                        "// extract field {} as {}",
                                        fi, id.name
                                    ));
                                    self.emit_line(&format!(
                                        "{} = arith.constant 0 : i64",
                                        field_ssa
                                    ));
                                    self.define_var(&id.name, &field_ssa, MLIRType::I64);
                                }
                            }
                        }

                        let (ssa, ty) = self.gen_expr(&arm.body);
                        result_ssa = ssa.clone();
                        result_ty = ty;

                        self.indent -= 1;
                        self.emit_line("}");
                    }
                }

                if result_ssa.is_empty() {
                    let ssa = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                    (ssa, MLIRType::I64)
                } else {
                    (result_ssa, result_ty)
                }
            }

            ExprKind::TypeCall { .. } => {
                let ssa = self.fresh_ssa();
                self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                (ssa, MLIRType::I64)
            }
        }
    }

    // --- Binary operation generation ---

    fn gen_binop(&mut self, lhs: &str, op: BinOp, rhs: &str, ty: &MLIRType) -> (String, MLIRType) {
        let ssa = self.fresh_ssa();
        let is_float = Self::is_float_type(ty);

        match op {
            BinOp::Add => {
                if is_float {
                    self.emit_line(&format!("{} = arith.addf {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.addi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Sub => {
                if is_float {
                    self.emit_line(&format!("{} = arith.subf {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.subi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Mul => {
                if is_float {
                    self.emit_line(&format!("{} = arith.mulf {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.muli {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Div => {
                if is_float {
                    self.emit_line(&format!("{} = arith.divf {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.divsi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Mod => {
                self.emit_line(&format!("{} = arith.remsi {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::Pow => {
                // No direct MLIR pow for integers — emit as call to math.powf for floats
                if is_float {
                    self.emit_line(&format!("{} = math.powf {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("// TODO: integer pow"));
                    self.emit_line(&format!("{} = arith.constant 0 : {}", ssa, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Eq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf oeq, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi eq, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::NotEq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf une, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi ne, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::Lt => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf olt, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi slt, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::Gt => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf ogt, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi sgt, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::LtEq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf ole, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi sle, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::GtEq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf oge, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi sge, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::And => {
                self.emit_line(&format!("{} = arith.andi {}, {} : i1", ssa, lhs, rhs));
                (ssa, MLIRType::I1)
            }
            BinOp::Or => {
                self.emit_line(&format!("{} = arith.ori {}, {} : i1", ssa, lhs, rhs));
                (ssa, MLIRType::I1)
            }
            BinOp::BitAnd => {
                self.emit_line(&format!("{} = arith.andi {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::BitOr => {
                self.emit_line(&format!("{} = arith.ori {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::BitXor => {
                self.emit_line(&format!("{} = arith.xori {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::Shl => {
                self.emit_line(&format!("{} = arith.shli {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::Shr => {
                self.emit_line(&format!("{} = arith.shrsi {}, {} : {}", ssa, lhs, rhs, ty));
                (ssa, ty.clone())
            }
            BinOp::ElemMul | BinOp::ElemDiv => {
                self.emit_line(&format!("// elementwise op"));
                self.emit_line(&format!("{} = arith.constant 0 : {}", ssa, ty));
                (ssa, ty.clone())
            }
        }
    }

    fn emit_arith_op(&mut self, op: &str, lhs: &str, rhs: &str, ty: &MLIRType) -> String {
        let ssa = self.fresh_ssa();
        let is_float = Self::is_float_type(ty);
        let suffix = if is_float { "f" } else { "i" };
        self.emit_line(&format!(
            "{} = arith.{}{} {}, {} : {}",
            ssa, op, suffix, lhs, rhs, ty
        ));
        ssa
    }

    // --- Control flow ---

    fn gen_if_expr(&mut self, cond: &Expr, then_block: &Block, else_block: &Option<Block>) -> (String, MLIRType) {
        let (cond_ssa, _) = self.gen_expr(cond);

        // Determine result type
        let result_ty = if let Some(expr) = &then_block.expr {
            self.infer_type(expr)
        } else {
            MLIRType::None
        };

        if result_ty == MLIRType::None {
            // Statement-level if (no result)
            self.emit_line(&format!("scf.if {} {{", cond_ssa));
            self.indent += 1;
            self.push_scope();
            for stmt in &then_block.stmts {
                self.gen_stmt(stmt);
            }
            self.pop_scope();
            self.indent -= 1;

            if let Some(else_b) = else_block {
                self.emit_line("} else {");
                self.indent += 1;
                self.push_scope();
                for stmt in &else_b.stmts {
                    self.gen_stmt(stmt);
                }
                self.pop_scope();
                self.indent -= 1;
            }
            self.emit_line("}");

            let ssa = self.fresh_ssa();
            self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
            (ssa, MLIRType::I64)
        } else {
            // Expression-level if (produces a result via scf.if ... -> type)
            let ssa = self.fresh_ssa();
            self.emit_line(&format!("{} = scf.if {} -> ({}) {{", ssa, cond_ssa, result_ty));
            self.indent += 1;
            self.push_scope();
            for stmt in &then_block.stmts {
                self.gen_stmt(stmt);
            }
            if let Some(expr) = &then_block.expr {
                let (then_ssa, _) = self.gen_expr(expr);
                self.emit_line(&format!("scf.yield {}", then_ssa));
            }
            self.pop_scope();
            self.indent -= 1;

            if let Some(else_b) = else_block {
                self.emit_line(&format!("}} else {{"));
                self.indent += 1;
                self.push_scope();
                for stmt in &else_b.stmts {
                    self.gen_stmt(stmt);
                }
                if let Some(expr) = &else_b.expr {
                    let (else_ssa, _) = self.gen_expr(expr);
                    self.emit_line(&format!("scf.yield {}", else_ssa));
                }
                self.pop_scope();
                self.indent -= 1;
            }
            self.emit_line("}");

            (ssa, result_ty)
        }
    }

    fn gen_for_loop(&mut self, var: &Ident, iter: &Expr, body: &Block) {
        // Check if iter is a range expression
        if let ExprKind::Range { start, end } = &iter.kind {
            let (start_ssa, _) = self.gen_expr(start);
            let (end_ssa, _) = self.gen_expr(end);

            // Cast to index type
            let start_idx = self.fresh_ssa();
            let end_idx = self.fresh_ssa();
            let step_idx = self.fresh_ssa();

            self.emit_line(&format!("{} = arith.index_cast {} : i64 to index", start_idx, start_ssa));
            self.emit_line(&format!("{} = arith.index_cast {} : i64 to index", end_idx, end_ssa));
            self.emit_line(&format!("{} = arith.constant 1 : index", step_idx));

            let iv = self.fresh_ssa();
            self.emit_line(&format!(
                "scf.for {} = {} to {} step {} {{",
                iv, start_idx, end_idx, step_idx
            ));
            self.indent += 1;

            self.push_scope();
            // Cast induction variable back to i64 for use in body
            let iv_i64 = self.fresh_ssa();
            self.emit_line(&format!("{} = arith.index_cast {} : index to i64", iv_i64, iv));
            self.define_var(&var.name, &iv_i64, MLIRType::I64);

            for stmt in &body.stmts {
                self.gen_stmt(stmt);
            }

            self.pop_scope();
            self.indent -= 1;
            self.emit_line("}");
        } else {
            self.emit_line("// TODO: for-each over non-range iterable");
        }
    }

    fn gen_while_loop(&mut self, cond: &Expr, body: &Block) {
        self.emit_line("scf.while : () -> () {");
        self.indent += 1;

        let (cond_ssa, _) = self.gen_expr(cond);
        self.emit_line(&format!("scf.condition({})", cond_ssa));

        self.indent -= 1;
        self.emit_line("} do {");
        self.indent += 1;

        self.push_scope();
        for stmt in &body.stmts {
            self.gen_stmt(stmt);
        }
        self.pop_scope();

        self.emit_line("scf.yield");
        self.indent -= 1;
        self.emit_line("}");
    }
}

/// Generate MLIR IR from a parsed Vortex program
pub fn generate_mlir(program: &Program) -> String {
    let mut codegen = CodeGen::new();
    codegen.generate(program)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser;

    fn gen(source: &str) -> String {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).expect("parse failed");
        generate_mlir(&program)
    }

    #[test]
    fn test_simple_add() {
        let ir = gen("fn add(a: i64, b: i64) -> i64 { return a + b }");
        assert!(ir.contains("func.func @add"));
        assert!(ir.contains("arith.addi"));
        assert!(ir.contains("func.return"));
    }

    #[test]
    fn test_float_mul() {
        let ir = gen("fn mul(x: f64, y: f64) -> f64 { return x * y }");
        assert!(ir.contains("func.func @mul"));
        assert!(ir.contains("arith.mulf"));
    }

    #[test]
    fn test_if_expression() {
        let ir = gen("fn abs(x: i64) -> i64 { if x > 0 { return x } else { return 0 - x } }");
        assert!(ir.contains("arith.cmpi sgt"));
        assert!(ir.contains("scf.if"));
    }

    #[test]
    fn test_for_loop() {
        let ir = gen("fn sum(n: i64) -> i64 { var s: i64 = 0\n for i in 0..n { s += i }\n return s }");
        assert!(ir.contains("scf.for"));
        assert!(ir.contains("arith.index_cast"));
    }

    #[test]
    fn test_kernel_emission() {
        let ir = gen("kernel vadd(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> { return a + b }");
        assert!(ir.contains("gpu.module @vadd_module"));
        assert!(ir.contains("gpu.func @vadd"));
        assert!(ir.contains("kernel"));
        assert!(ir.contains("gpu.return"));
    }

    #[test]
    fn test_constants() {
        let ir = gen("fn main() { let x = 42\n let y = 3.14\n let z = true }");
        assert!(ir.contains("arith.constant 42 : i64"));
        assert!(ir.contains("arith.constant 3.14 : f64"));
        assert!(ir.contains("arith.constant 1 : i1"));
    }

    #[test]
    fn test_comparison_ops() {
        let ir = gen("fn check(a: i64, b: i64) -> bool { return a == b }");
        assert!(ir.contains("arith.cmpi eq"));
    }

    #[test]
    fn test_bitwise_ops() {
        let ir = gen("fn xor(a: i64, b: i64) -> i64 { return a ^ b }");
        assert!(ir.contains("arith.xori"));
    }
}
