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
    Tensor(Box<MLIRType>, Vec<i64>), // tensor<shape x type>
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
            MLIRType::Tensor(elem, shape) => {
                write!(f, "tensor<")?;
                for dim in shape {
                    if *dim < 0 { write!(f, "?x")?; }
                    else { write!(f, "{}x", dim)?; }
                }
                write!(f, "{}>", elem)
            }
            MLIRType::None => write!(f, "()"),
        }
    }
}

/// Stored function signature info
#[derive(Debug, Clone)]
struct FnSig {
    ret_type: MLIRType,
    _param_types: Vec<MLIRType>,
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
    /// Function signatures: name -> return type
    fn_signatures: HashMap<String, FnSig>,
    /// Whether we are currently inside a GPU kernel
    in_kernel: bool,
    /// Track which variables have unsigned source types
    unsigned_vars: HashMap<String, bool>,
    /// Nesting depth inside scf regions (if/for/while) where func.return is invalid
    scf_depth: usize,
}

impl CodeGen {
    pub fn new() -> Self {
        Self {
            output: String::new(),
            indent: 0,
            ssa_counter: 0,
            var_map: vec![HashMap::new()],
            enum_tags: HashMap::new(),
            fn_signatures: HashMap::new(),
            in_kernel: false,
            unsigned_vars: HashMap::new(),
            scf_depth: 0,
        }
    }

    /// Generate MLIR IR from a Vortex program
    pub fn generate(&mut self, program: &Program) -> String {
        // Emit module wrapper
        self.emit_line("module {");
        self.indent += 1;

        // First pass: register enum tags and function signatures
        for item in &program.items {
            match &item.kind {
                ItemKind::Enum(e) => {
                    for (i, variant) in e.variants.iter().enumerate() {
                        let key = format!("{}::{}", e.name.name, variant.name.name);
                        self.enum_tags.insert(key, i as i32);
                    }
                }
                ItemKind::Function(func) => {
                    let ret_type = func.ret_type.as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(MLIRType::None);
                    let param_types: Vec<MLIRType> = func.params.iter()
                        .map(|p| self.resolve_type(&p.ty))
                        .collect();
                    self.fn_signatures.insert(func.name.name.clone(), FnSig {
                        ret_type,
                        _param_types: param_types,
                    });
                }
                ItemKind::Kernel(kernel) => {
                    let ret_type = kernel.ret_type.as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(MLIRType::None);
                    let param_types: Vec<MLIRType> = kernel.params.iter()
                        .map(|p| self.resolve_type(&p.ty))
                        .collect();
                    self.fn_signatures.insert(kernel.name.name.clone(), FnSig {
                        ret_type,
                        _param_types: param_types,
                    });
                }
                _ => {}
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

    /// Check if a Vortex type name is unsigned
    fn is_unsigned_type_name(name: &str) -> bool {
        matches!(name, "u8" | "u16" | "u32" | "u64" | "u128")
    }

    /// Check if a type expression is unsigned
    fn is_unsigned_type_expr(ty: &TypeExpr) -> bool {
        if let TypeExprKind::Named(id) = &ty.kind {
            Self::is_unsigned_type_name(&id.name)
        } else {
            false
        }
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
            TypeExprKind::Generic { name, args } => {
                match name.name.as_str() {
                    "Tensor" => {
                        let elem_ty = if let Some(TypeArg::Type(ty)) = args.first() { self.resolve_type(ty) } else { MLIRType::F32 };
                        let shape = if let Some(TypeArg::Shape(dims)) = args.get(1) { dims.iter().map(|d| match d { ShapeDim::Lit(n) => *n as i64, _ => -1 }).collect() } else { vec![-1] }; MLIRType::Tensor(Box::new(elem_ty), shape)
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
            ExprKind::StringLiteral(_) => MLIRType::I64,
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
            ExprKind::Call { func, .. } => {
                if let ExprKind::Ident(id) = &func.kind {
                    match id.name.as_str() {
                        "softmax" | "gelu" | "layer_norm" | "attention" | "rope" => {
                            return MLIRType::Tensor(Box::new(MLIRType::F32), vec![-1]);
                        }
                        _ => {
                            if let Some(sig) = self.fn_signatures.get(&id.name) {
                                return sig.ret_type.clone();
                            }
                        }
                    }
                }
                MLIRType::I64
            }
            ExprKind::If { then_block, .. } => {
                if let Some(expr) = &then_block.expr {
                    self.infer_type(expr)
                } else {
                    MLIRType::None
                }
            }
            ExprKind::Cast { ty, .. } => self.resolve_type(ty),
            ExprKind::ArrayLiteral(elems) => {
                if let Some(first) = elems.first() {
                    let elem_ty = self.infer_type(first);
                    MLIRType::MemRef(Box::new(elem_ty), vec![elems.len() as i64])
                } else {
                    MLIRType::MemRef(Box::new(MLIRType::I64), vec![0])
                }
            }
            _ => MLIRType::I64,
        }
    }

    fn is_float_type(ty: &MLIRType) -> bool {
        match ty {
            MLIRType::F16 | MLIRType::F32 | MLIRType::F64 => true,
            MLIRType::Tensor(elem, _) => Self::is_float_type(elem),
            MLIRType::MemRef(elem, _) => Self::is_float_type(elem),
            _ => false,
        }
    }

    fn is_int_type(ty: &MLIRType) -> bool {
        matches!(ty, MLIRType::I1 | MLIRType::I8 | MLIRType::I16
            | MLIRType::I32 | MLIRType::I64 | MLIRType::I128)
    }

    /// Get the element type of a memref or tensor, or the type itself
    fn elem_type(ty: &MLIRType) -> &MLIRType {
        match ty {
            MLIRType::MemRef(elem, _) | MLIRType::Tensor(elem, _) => elem,
            _ => ty,
        }
    }

    /// Emit the appropriate return for the current context (gpu.return vs func.return)
    fn emit_return_typed(&mut self, value: Option<(&str, &MLIRType)>) {
        if self.in_kernel {
            if let Some((v, ty)) = value {
                self.emit_line(&format!("gpu.return {} : {}", v, ty));
            } else {
                self.emit_line("gpu.return");
            }
        } else if let Some((v, ty)) = value {
            self.emit_line(&format!("func.return {} : {}", v, ty));
        } else {
            self.emit_line("func.return");
        }
    }

    // --- Enum generation ---

    fn gen_enum(&mut self, e: &EnumDef) {
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

        let mut param_strs = Vec::new();
        let mut param_types = Vec::new();
        for param in &func.params {
            let ty = self.resolve_type(&param.ty);
            let ssa = self.fresh_ssa();
            self.define_var(&param.name.name, &ssa, ty.clone());
            if Self::is_unsigned_type_expr(&param.ty) {
                self.unsigned_vars.insert(param.name.name.clone(), true);
            }
            param_strs.push(format!("{}: {}", ssa, ty));
            param_types.push(ty);
        }

        let ret_type = func.ret_type.as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(MLIRType::None);

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

        self.gen_block(&func.body, &ret_type);

        // Ensure every function has a terminator
        let has_explicit_return = func.body.stmts.iter().any(|s| matches!(s.kind, StmtKind::Return(_)));
        let has_tail_expr = func.body.expr.is_some() && ret_type != MLIRType::None;
        if !has_explicit_return && !has_tail_expr {
            self.emit_line("func.return");
        }

        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.pop_scope();
    }

    fn gen_kernel(&mut self, kernel: &Kernel) {
        self.push_scope();
        self.in_kernel = true;

        // GPU kernel functions MUST have void return type per MLIR gpu dialect.
        // If the Vortex kernel has a return type, we add an output memref parameter.
        let ret_type = kernel.ret_type.as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(MLIRType::None);

        let mut param_strs = Vec::new();
        for param in &kernel.params {
            let ty = self.resolve_type(&param.ty);
            let ssa = self.fresh_ssa();
            self.define_var(&param.name.name, &ssa, ty.clone());
            param_strs.push(format!("{}: {}", ssa, ty));
        }

        // If there's a return type, add an output parameter
        let output_ssa = if ret_type != MLIRType::None {
            let ssa = self.fresh_ssa();
            let out_ty = MLIRType::MemRef(Box::new(ret_type.clone()), vec![1]);
            param_strs.push(format!("{}: {}", ssa, out_ty));
            Some((ssa, out_ty))
        } else {
            None
        };

        let params_str = param_strs.join(", ");

        self.emit_line(&format!("gpu.module @{}_module {{", kernel.name.name));
        self.indent += 1;
        // Always void return for kernel
        self.emit_line(&format!(
            "gpu.func @{}({}) kernel {{",
            kernel.name.name, params_str
        ));
        self.indent += 1;

        if let Some(ref sched) = kernel.schedule {
            for (name, _val) in &sched.params {
                self.emit_line(&format!("// schedule: {} annotation", name.name));
            }
        }

        // For kernels with return types, generate the body but intercept returns
        // to store into the output memref instead
        if ret_type != MLIRType::None {
            let has_explicit_return = kernel.body.stmts.iter().any(|s| matches!(s.kind, StmtKind::Return(_)));
            // Generate body statements
            for stmt in &kernel.body.stmts {
                self.gen_stmt_kernel_return(stmt, &output_ssa);
            }
            // Handle tail expression
            if let Some(expr) = &kernel.body.expr {
                let (val_ssa, _val_ty) = self.gen_expr(expr);
                if let Some((ref out_ssa, ref out_ty)) = output_ssa {
                    let idx = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : index", idx));
                    self.emit_line(&format!("memref.store {}, {}[{}] : {}", val_ssa, out_ssa, idx, out_ty));
                }
                self.emit_line("gpu.return");
            } else if !has_explicit_return {
                self.emit_line("gpu.return");
            }
        } else {
            self.gen_block(&kernel.body, &MLIRType::None);
            let has_explicit_return = kernel.body.stmts.iter().any(|s| matches!(s.kind, StmtKind::Return(_)));
            if !has_explicit_return {
                self.emit_line("gpu.return");
            }
        }

        self.indent -= 1;
        self.emit_line("}");
        self.indent -= 1;
        self.emit_line("}");
        self.emit_line("");

        self.in_kernel = false;
        self.pop_scope();
    }

    /// Generate a statement inside a kernel, converting return statements
    /// to memref.store + gpu.return instead of gpu.return with value.
    fn gen_stmt_kernel_return(&mut self, stmt: &Stmt, output_ssa: &Option<(String, MLIRType)>) {
        match &stmt.kind {
            StmtKind::Return(Some(expr)) => {
                let (val_ssa, _val_ty) = self.gen_expr(expr);
                if let Some((ref out_ssa, ref out_ty)) = output_ssa {
                    let idx = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : index", idx));
                    self.emit_line(&format!("memref.store {}, {}[{}] : {}", val_ssa, out_ssa, idx, out_ty));
                }
                self.emit_line("gpu.return");
            }
            StmtKind::Return(None) => {
                self.emit_line("gpu.return");
            }
            _ => self.gen_stmt(stmt),
        }
    }

    // --- Block generation ---

    fn gen_block(&mut self, block: &Block, expected_ret: &MLIRType) {
        for stmt in &block.stmts {
            self.gen_stmt(stmt);
        }

        if let Some(expr) = &block.expr {
            let (ssa, ty) = self.gen_expr(expr);
            if *expected_ret != MLIRType::None {
                self.emit_return_typed(Some((&ssa, &ty)));
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
                if let Some(t) = ty {
                    if Self::is_unsigned_type_expr(t) {
                        self.unsigned_vars.insert(name.name.clone(), true);
                    }
                }
                self.define_var(&name.name, &ssa, actual_ty);
            }

            StmtKind::Return(Some(expr)) => {
                let (ssa, ty) = self.gen_expr(expr);
                if self.scf_depth > 0 {
                    // Cannot emit func.return/gpu.return inside scf regions.
                    self.emit_line(&format!("// early return: {} (inside scf region)", ssa));
                } else {
                    self.emit_return_typed(Some((&ssa, &ty)));
                }
            }
            StmtKind::Return(None) => {
                if self.scf_depth > 0 {
                    self.emit_line("// early return (inside scf region)");
                } else {
                    self.emit_return_typed(None);
                }
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
                    ExprKind::Index { base, indices } => {
                        let (base_ssa, base_ty) = self.gen_expr(base);
                        let (idx_ssa, _) = self.gen_expr(&indices[0]);
                        let idx_index = self.fresh_ssa();
                        self.emit_line(&format!(
                            "{} = arith.index_cast {} : i64 to index",
                            idx_index, idx_ssa
                        ));

                        let final_rhs = match op {
                            AssignOp::Assign => rhs_ssa,
                            _ => {
                                let cur = self.fresh_ssa();
                                self.emit_line(&format!(
                                    "{} = memref.load {}[{}] : {}",
                                    cur, base_ssa, idx_index, base_ty
                                ));
                                let elem_ty = Self::elem_type(&base_ty).clone();
                                match op {
                                    AssignOp::AddAssign => self.emit_arith_op("add", &cur, &rhs_ssa, &elem_ty),
                                    AssignOp::SubAssign => self.emit_arith_op("sub", &cur, &rhs_ssa, &elem_ty),
                                    AssignOp::MulAssign => self.emit_arith_op("mul", &cur, &rhs_ssa, &elem_ty),
                                    AssignOp::DivAssign => self.emit_arith_op("div", &cur, &rhs_ssa, &elem_ty),
                                    _ => rhs_ssa,
                                }
                            }
                        };
                        self.emit_line(&format!(
                            "memref.store {}, {}[{}] : {}",
                            final_rhs, base_ssa, idx_index, base_ty
                        ));
                    }
                    ExprKind::FieldAccess { base, field } => {
                        let (base_ssa, base_ty) = self.gen_expr(base);
                        let field_idx = self.fresh_ssa();
                        self.emit_line(&format!(
                            "// field store: {}.{} = {}",
                            base_ssa, field.name, rhs_ssa
                        ));
                        let idx: usize = field.name.bytes().map(|b| b as usize).sum::<usize>() % 16;
                        self.emit_line(&format!(
                            "{} = arith.constant {} : index",
                            field_idx, idx
                        ));
                        self.emit_line(&format!(
                            "memref.store {}, {}[{}] : {}",
                            rhs_ssa, base_ssa, field_idx, base_ty
                        ));
                    }
                    _ => {
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

            StmtKind::Loop { body } => {
                // Emit as scf.while with constant true condition
                self.emit_line("scf.while : () -> () {");
                self.indent += 1;
                self.scf_depth += 1;
                let cond_ssa = self.fresh_ssa();
                self.emit_line(&format!("{} = arith.constant 1 : i1", cond_ssa));
                self.emit_line(&format!("scf.condition({})", cond_ssa));
                self.indent -= 1;
                self.emit_line("} do {");
                self.indent += 1;
                self.push_scope();
                for s in &body.stmts {
                    self.gen_stmt(s);
                }
                self.pop_scope();
                self.emit_line("scf.yield");
                self.scf_depth -= 1;
                self.indent -= 1;
                self.emit_line("}");
            }

            StmtKind::Break | StmtKind::Continue => {
                self.emit_line(&format!("// {}", if matches!(stmt.kind, StmtKind::Break) { "break" } else { "continue" }));
            }

            StmtKind::Dispatch { index, targets, args } => {
                let (idx_ssa, _) = self.gen_expr(index);
                let mut arg_ssas = Vec::new();
                let mut arg_types = Vec::new();
                for arg in args {
                    let (ssa, ty) = self.gen_expr(arg);
                    arg_ssas.push(ssa);
                    arg_types.push(ty);
                }
                let args_str = arg_ssas.join(", ");
                let types_str = arg_types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ");
                for (i, target) in targets.iter().enumerate() {
                    let expected = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant {} : i64", expected, i));
                    let cmp = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.cmpi eq, {}, {} : i64", cmp, idx_ssa, expected));
                    self.emit_line(&format!("scf.if {} {{", cmp));
                    self.indent += 1;
                    self.emit_line(&format!(
                        "func.call @{}({}) : ({}) -> ()",
                        target.name, args_str, types_str
                    ));
                    self.indent -= 1;
                    self.emit_line("}");
                }
            }

            StmtKind::Live { name, value } => {
                let (ssa, _) = self.gen_expr(value);
                self.emit_line(&format!("// live {} = {}", name.name, ssa));
            }
            StmtKind::Fuse { body } => {
                self.emit_line("// fuse block start");
                for s in &body.stmts { self.gen_stmt(s); }
                self.emit_line("// fuse block end");
            }
            StmtKind::GpuLet { name, value } => {
                let (ssa, _) = self.gen_expr(value);
                self.emit_line(&format!("// gpu let {} = {}", name.name, ssa));
            }
            StmtKind::Parallel { body, .. } => {
                for s in &body.stmts { self.gen_stmt(s); }
            }
            StmtKind::Train { .. } | StmtKind::Deterministic { .. } | StmtKind::Autocast { .. }
            | StmtKind::Speculate { .. } | StmtKind::Topology { .. } | StmtKind::Mmap { .. }
            | StmtKind::Explain { .. } | StmtKind::Quantize { .. } | StmtKind::Safe { .. }
            | StmtKind::Consensus { .. } | StmtKind::SymbolicBlock { .. } | StmtKind::TemporalBlock { .. }
            | StmtKind::Federated { .. } | StmtKind::SandboxBlock { .. } | StmtKind::Compress { .. }
            | StmtKind::Metacognition { .. }
            | StmtKind::TheoremBlock { .. } | StmtKind::ContinualLearn { .. }
            | StmtKind::MultimodalBlock { .. } | StmtKind::WorldModelBlock { .. }
            | StmtKind::SelfImproveBlock { .. } | StmtKind::MemoryBlock { .. }
            | StmtKind::AttentionBlock { .. } | StmtKind::CurriculumBlock { .. }
            | StmtKind::EnsembleBlock { .. } | StmtKind::AdversarialBlock { .. }
            | StmtKind::TransferBlock { .. } | StmtKind::SparseBlock { .. }
            | StmtKind::AsyncInferBlock { .. } | StmtKind::ProfileBlock { .. }
            | StmtKind::ContractBlock { .. } => {}
        }
    }

    // --- Expression generation (returns SSA name and type) ---

    fn gen_expr(&mut self, expr: &Expr) -> (String, MLIRType) {
        match &expr.kind {
            ExprKind::BigIntLiteral(_s) => {
                let ssa = self.fresh_ssa();
                self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                (ssa, MLIRType::I64)
            }
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
                let (mut r_ssa, r_ty) = self.gen_expr(rhs);
                let is_unsigned = if let ExprKind::Ident(id) = &lhs.kind {
                    self.unsigned_vars.contains_key(&id.name)
                } else {
                    false
                };
                // Insert cast if types mismatch (e.g., f32 vs f64)
                if l_ty != r_ty {
                    let cast_ssa = self.fresh_ssa();
                    if Self::is_float_type(&l_ty) && Self::is_float_type(&r_ty) {
                        // Cast rhs to match lhs type
                        let op_name = match (&r_ty, &l_ty) {
                            (MLIRType::F64, MLIRType::F32) | (MLIRType::F64, MLIRType::F16) |
                            (MLIRType::F32, MLIRType::F16) => "arith.truncf",
                            _ => "arith.extf",
                        };
                        self.emit_line(&format!(
                            "{} = {} {} : {} to {}",
                            cast_ssa, op_name, r_ssa, r_ty, l_ty
                        ));
                        r_ssa = cast_ssa;
                    } else if Self::is_int_type(&l_ty) && Self::is_int_type(&r_ty) {
                        // Cast rhs int to match lhs int type
                        self.emit_line(&format!(
                            "{} = arith.extsi {} : {} to {}",
                            cast_ssa, r_ssa, r_ty, l_ty
                        ));
                        r_ssa = cast_ssa;
                    }
                }
                self.gen_binop(&l_ssa, *op, &r_ssa, &l_ty, is_unsigned)
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

                let ret_type = self.fn_signatures.get(&func_name)
                    .map(|sig| sig.ret_type.clone())
                    .unwrap_or(MLIRType::I64);

                let ssa = self.fresh_ssa();
                let args_str = arg_ssas.join(", ");
                let types_str = arg_types.iter().map(|t| format!("{}", t)).collect::<Vec<_>>().join(", ");

                if ret_type == MLIRType::None {
                    self.emit_line(&format!(
                        "func.call @{}({}) : ({}) -> ()",
                        func_name, args_str, types_str
                    ));
                    self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                    (ssa, MLIRType::I64)
                } else {
                    self.emit_line(&format!(
                        "{} = func.call @{}({}) : ({}) -> {}",
                        ssa, func_name, args_str, types_str, ret_type
                    ));
                    (ssa, ret_type)
                }
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
                let (s_ssa, s_ty) = self.gen_expr(start);
                let (_e_ssa, _e_ty) = self.gen_expr(end);
                (s_ssa, s_ty)
            }

            ExprKind::ArrayLiteral(elems) => {
                if elems.is_empty() {
                    let ssa = self.fresh_ssa();
                    self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                    return (ssa, MLIRType::I64);
                }

                let mut elem_ssas = Vec::new();
                let mut elem_ty = MLIRType::I64;
                for elem in elems {
                    let (ssa, ty) = self.gen_expr(elem);
                    elem_ssas.push(ssa);
                    elem_ty = ty;
                }

                let len = elems.len() as i64;
                let memref_ty = MLIRType::MemRef(Box::new(elem_ty.clone()), vec![len]);

                let arr_ssa = self.fresh_ssa();
                self.emit_line(&format!(
                    "{} = memref.alloc() : {}",
                    arr_ssa, memref_ty
                ));

                for (i, elem_ssa) in elem_ssas.iter().enumerate() {
                    let idx = self.fresh_ssa();
                    self.emit_line(&format!(
                        "{} = arith.constant {} : index",
                        idx, i
                    ));
                    self.emit_line(&format!(
                        "memref.store {}, {}[{}] : {}",
                        elem_ssa, arr_ssa, idx, memref_ty
                    ));
                }

                (arr_ssa, memref_ty)
            }

            ExprKind::Cast { expr: inner, ty } => {
                let (inner_ssa, inner_ty) = self.gen_expr(inner);
                let target_ty = self.resolve_type(ty);
                let ssa = self.fresh_ssa();

                if inner_ty == target_ty {
                    return (inner_ssa, inner_ty);
                }

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
                    self.emit_line(&format!(
                        "{} = arith.extf {} : {} to {}",
                        ssa, inner_ssa, inner_ty, target_ty
                    ));
                } else {
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

                let idx_index = self.fresh_ssa();
                self.emit_line(&format!(
                    "{} = arith.index_cast {} : i64 to index",
                    idx_index, idx_ssa
                ));

                let ssa = self.fresh_ssa();
                let elem_ty = match &base_ty {
                    MLIRType::MemRef(elem, _) => (**elem).clone(),
                    MLIRType::Tensor(elem, _) => (**elem).clone(),
                    _ => base_ty.clone(),
                };

                match &base_ty {
                    MLIRType::MemRef(_, _) => {
                        self.emit_line(&format!(
                            "{} = memref.load {}[{}] : {}",
                            ssa, base_ssa, idx_index, base_ty
                        ));
                    }
                    MLIRType::Tensor(_, _) => {
                        self.emit_line(&format!(
                            "{} = tensor.extract {}[{}] : {}",
                            ssa, base_ssa, idx_index, base_ty
                        ));
                    }
                    _ => {
                        self.emit_line(&format!(
                            "// index: {}[{}]", base_ssa, idx_ssa
                        ));
                        self.emit_line(&format!(
                            "{} = arith.constant 0 : i64", ssa
                        ));
                    }
                }
                (ssa, elem_ty)
            }

            ExprKind::FieldAccess { base, field } => {
                let (base_ssa, base_ty) = self.gen_expr(base);
                let ssa = self.fresh_ssa();
                let field_idx_val: usize = field.name.bytes().map(|b| b as usize).sum::<usize>() % 16;
                let field_idx = self.fresh_ssa();
                self.emit_line(&format!(
                    "// field access: {}.{}",
                    base_ssa, field.name
                ));
                self.emit_line(&format!(
                    "{} = arith.constant {} : index",
                    field_idx, field_idx_val
                ));

                let elem_ty = match &base_ty {
                    MLIRType::MemRef(elem, _) => (**elem).clone(),
                    _ => MLIRType::I64,
                };

                match &base_ty {
                    MLIRType::MemRef(_, _) => {
                        self.emit_line(&format!(
                            "{} = memref.load {}[{}] : {}",
                            ssa, base_ssa, field_idx, base_ty
                        ));
                    }
                    _ => {
                        self.emit_line(&format!(
                            "{} = llvm.extractvalue {}[{}] : !llvm.struct<({})>",
                            ssa, base_ssa, field_idx_val, elem_ty
                        ));
                    }
                }
                (ssa, elem_ty)
            }

            ExprKind::MatMul { lhs, rhs } => {
                let (l_ssa, l_ty) = self.gen_expr(lhs);
                let (r_ssa, r_ty) = self.gen_expr(rhs);
                let ssa = self.fresh_ssa();
                match &l_ty {
                    MLIRType::Tensor(elem, _) => {
                        let out_ty = l_ty.clone();
                        let zero = self.fresh_ssa();
                        self.emit_line(&format!("{} = arith.constant 0.0 : {}", zero, elem));
                        let init = self.fresh_ssa();
                        self.emit_line(&format!("{} = linalg.fill ins({} : {}) outs({} : {}) -> {}", init, zero, elem, l_ssa, out_ty, out_ty));
                        self.emit_line(&format!("{} = linalg.matmul ins({}, {} : {}, {}) outs({} : {}) -> {}", ssa, l_ssa, r_ssa, l_ty, r_ty, init, out_ty, out_ty));
                        (ssa, out_ty)
                    }
                    _ => {
                        self.emit_line(&format!("// matmul: {} @ {}", l_ssa, r_ssa));
                        self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                        (ssa, l_ty)
                    }
                }
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

                let tag_ssa = self.fresh_ssa();
                self.emit_line(&format!("// extract tag from enum value"));
                self.emit_line(&format!("{} = arith.constant 0 : i32", tag_ssa));

                let mut result_ssa = String::new();
                let mut result_ty = MLIRType::I64;

                for (i, arm) in arms.iter().enumerate() {
                    let is_wildcard = matches!(&arm.pattern, Pattern::Wildcard | Pattern::Ident(_));

                    if is_wildcard {
                        self.emit_line("// wildcard arm");
                        let (ssa, ty) = self.gen_expr(&arm.body);
                        result_ssa = ssa;
                        result_ty = ty;
                    } else {
                        let tag_val = if let Pattern::Variant { name: _, .. } = &arm.pattern {
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

                        self.emit_line(&format!("scf.if {} {{", cmp_ssa));
                        self.indent += 1;

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

            ExprKind::TypeCall { .. } | ExprKind::Closure { .. } | ExprKind::Try(_) => {
                let ssa = self.fresh_ssa();
                self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
                (ssa, MLIRType::I64)
            }
        }
    }

    // --- Binary operation generation ---

    fn gen_binop(&mut self, lhs: &str, op: BinOp, rhs: &str, ty: &MLIRType, is_unsigned: bool) -> (String, MLIRType) {
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
                } else if is_unsigned {
                    self.emit_line(&format!("{} = arith.divui {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.divsi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Mod => {
                if is_unsigned {
                    self.emit_line(&format!("{} = arith.remui {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.remsi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::Pow => {
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
                } else if is_unsigned {
                    self.emit_line(&format!("{} = arith.cmpi ult, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi slt, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::Gt => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf ogt, {}, {} : {}", ssa, lhs, rhs, ty));
                } else if is_unsigned {
                    self.emit_line(&format!("{} = arith.cmpi ugt, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi sgt, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::LtEq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf ole, {}, {} : {}", ssa, lhs, rhs, ty));
                } else if is_unsigned {
                    self.emit_line(&format!("{} = arith.cmpi ule, {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.cmpi sle, {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, MLIRType::I1)
            }
            BinOp::GtEq => {
                if is_float {
                    self.emit_line(&format!("{} = arith.cmpf oge, {}, {} : {}", ssa, lhs, rhs, ty));
                } else if is_unsigned {
                    self.emit_line(&format!("{} = arith.cmpi uge, {}, {} : {}", ssa, lhs, rhs, ty));
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
                if is_unsigned {
                    self.emit_line(&format!("{} = arith.shrui {}, {} : {}", ssa, lhs, rhs, ty));
                } else {
                    self.emit_line(&format!("{} = arith.shrsi {}, {} : {}", ssa, lhs, rhs, ty));
                }
                (ssa, ty.clone())
            }
            BinOp::ElemMul | BinOp::ElemDiv => {
                if let MLIRType::Tensor(ref elem, ref _shape) = ty {
                    let is_elem_float = Self::is_float_type(elem);
                    let arith_op = if matches!(op, BinOp::ElemMul) {
                        if is_elem_float { "arith.mulf" } else { "arith.muli" }
                    } else {
                        if is_elem_float { "arith.divf" } else { "arith.divsi" }
                    };

                    let out = self.fresh_ssa();
                    self.emit_line(&format!("{} = linalg.generic {{", out));
                    self.indent += 1;
                    self.emit_line("indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],");
                    self.emit_line("iterator_types = [\"parallel\"]");
                    self.indent -= 1;
                    self.emit_line(&format!("}} ins({}, {} : {}, {}) outs({} : {}) {{", lhs, rhs, ty, ty, lhs, ty));
                    self.indent += 1;
                    self.emit_line(&format!("^bb0(%a: {}, %b: {}, %out: {}):", elem, elem, elem));
                    self.indent += 1;
                    self.emit_line(&format!("%result = {} %a, %b : {}", arith_op, elem));
                    self.emit_line(&format!("linalg.yield %result : {}", elem));
                    self.indent -= 2;
                    self.emit_line(&format!("}} -> {}", ty));
                    return (out, ty.clone());
                } else if Self::is_float_type(ty) {
                    let arith_op = if matches!(op, BinOp::ElemMul) { "mulf" } else { "divf" };
                    self.emit_line(&format!("{} = arith.{} {}, {} : {}", ssa, arith_op, lhs, rhs, ty));
                    (ssa, ty.clone())
                } else {
                    let arith_op = if matches!(op, BinOp::ElemMul) { "muli" } else { "divsi" };
                    self.emit_line(&format!("{} = arith.{} {}, {} : {}", ssa, arith_op, lhs, rhs, ty));
                    (ssa, ty.clone())
                }
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

        let result_ty = if let Some(expr) = &then_block.expr {
            self.infer_type(expr)
        } else {
            MLIRType::None
        };

        if result_ty == MLIRType::None {
            self.emit_line(&format!("scf.if {} {{", cond_ssa));
            self.indent += 1;
            self.scf_depth += 1;
            self.push_scope();
            for stmt in &then_block.stmts {
                self.gen_stmt(stmt);
            }
            self.pop_scope();
            self.scf_depth -= 1;
            self.indent -= 1;

            if let Some(else_b) = else_block {
                self.emit_line("} else {");
                self.indent += 1;
                self.scf_depth += 1;
                self.push_scope();
                for stmt in &else_b.stmts {
                    self.gen_stmt(stmt);
                }
                self.pop_scope();
                self.scf_depth -= 1;
                self.indent -= 1;
            }
            self.emit_line("}");

            let ssa = self.fresh_ssa();
            self.emit_line(&format!("{} = arith.constant 0 : i64", ssa));
            (ssa, MLIRType::I64)
        } else {
            let ssa = self.fresh_ssa();
            self.emit_line(&format!("{} = scf.if {} -> ({}) {{", ssa, cond_ssa, result_ty));
            self.indent += 1;
            self.scf_depth += 1;
            self.push_scope();
            for stmt in &then_block.stmts {
                self.gen_stmt(stmt);
            }
            if let Some(expr) = &then_block.expr {
                let (then_ssa, _) = self.gen_expr(expr);
                self.emit_line(&format!("scf.yield {} : {}", then_ssa, result_ty));
            }
            self.pop_scope();
            self.scf_depth -= 1;
            self.indent -= 1;

            if let Some(else_b) = else_block {
                self.emit_line(&format!("}} else {{"));
                self.indent += 1;
                self.scf_depth += 1;
                self.push_scope();
                for stmt in &else_b.stmts {
                    self.gen_stmt(stmt);
                }
                if let Some(expr) = &else_b.expr {
                    let (else_ssa, _) = self.gen_expr(expr);
                    self.emit_line(&format!("scf.yield {} : {}", else_ssa, result_ty));
                }
                self.pop_scope();
                self.scf_depth -= 1;
                self.indent -= 1;
            }
            self.emit_line("}");

            (ssa, result_ty)
        }
    }

    fn gen_for_loop(&mut self, var: &Ident, iter: &Expr, body: &Block) {
        if let ExprKind::Range { start, end } = &iter.kind {
            let (start_ssa, _) = self.gen_expr(start);
            let (end_ssa, _) = self.gen_expr(end);

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
            self.scf_depth += 1;

            self.push_scope();
            let iv_i64 = self.fresh_ssa();
            self.emit_line(&format!("{} = arith.index_cast {} : index to i64", iv_i64, iv));
            self.define_var(&var.name, &iv_i64, MLIRType::I64);

            for stmt in &body.stmts {
                self.gen_stmt(stmt);
            }

            self.pop_scope();
            self.scf_depth -= 1;
            self.indent -= 1;
            self.emit_line("}");
        } else {
            self.emit_line("// TODO: for-each over non-range iterable");
        }
    }

    fn gen_while_loop(&mut self, cond: &Expr, body: &Block) {
        self.emit_line("scf.while : () -> () {");
        self.indent += 1;
        self.scf_depth += 1;

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
        self.scf_depth -= 1;
        self.indent -= 1;
        self.emit_line("}");
    }
}

/// Generate MLIR IR from a parsed Vortex program
pub fn generate_mlir(program: &Program) -> String {
    let mut codegen = CodeGen::new();
    codegen.generate(program)
}

/// MLIR structural validation errors
#[derive(Debug, Clone)]
pub struct MLIRValidationError {
    pub line: usize,
    pub message: String,
}

/// Validate generated MLIR text for common structural errors.
/// This is not a full parser -- it catches the most common codegen bugs.
pub fn validate_mlir(mlir: &str) -> Vec<MLIRValidationError> {
    let mut errors = Vec::new();
    let lines: Vec<&str> = mlir.lines().collect();

    // Track SSA definitions and uses
    let mut defined_ssas: std::collections::HashSet<String> = std::collections::HashSet::new();
    // Track block/region nesting
    let mut region_stack: Vec<(usize, &str)> = Vec::new(); // (line, kind)
    // Track if each func/kernel region has a terminator
    let mut func_has_terminator = false;
    let mut _in_func = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        let lineno = i + 1;

        // Skip comments and empty lines
        if trimmed.starts_with("//") || trimmed.is_empty() {
            continue;
        }

        // Track SSA definitions: %N = ...
        if let Some(eq_pos) = trimmed.find(" = ") {
            let lhs = &trimmed[..eq_pos].trim();
            // Could be multiple results but typically single
            if lhs.starts_with('%') && !lhs.contains(',') {
                let ssa_name = lhs.to_string();
                if defined_ssas.contains(&ssa_name) {
                    errors.push(MLIRValidationError {
                        line: lineno,
                        message: format!("SSA value {} defined more than once", ssa_name),
                    });
                }
                defined_ssas.insert(ssa_name);
            }
        }

        // Track region opens
        if trimmed.ends_with('{') || trimmed.ends_with("kernel {") {
            let kind = if trimmed.starts_with("func.func") {
                _in_func = true;
                func_has_terminator = false;
                "func"
            } else if trimmed.starts_with("gpu.func") {
                _in_func = true;
                func_has_terminator = false;
                "gpu.func"
            } else if trimmed.starts_with("gpu.module") {
                "gpu.module"
            } else if trimmed.starts_with("scf.if") || trimmed.contains("= scf.if") {
                "scf.if"
            } else if trimmed.starts_with("scf.for") {
                "scf.for"
            } else if trimmed.starts_with("scf.while") {
                "scf.while"
            } else if trimmed.starts_with("module") {
                "module"
            } else if trimmed.starts_with("} do {") || trimmed.starts_with("} else {") {
                // continuation, not a new region
                ""
            } else {
                "block"
            };
            if !kind.is_empty() {
                region_stack.push((lineno, kind));
            }
        }

        // Check for terminators
        if trimmed.starts_with("func.return") || trimmed.starts_with("gpu.return") {
            func_has_terminator = true;

            // Check: func.return should not appear inside scf regions
            let in_scf = region_stack.iter().rev().any(|(_, k)|
                k.starts_with("scf."));
            if in_scf {
                let ret_kind = if trimmed.starts_with("func.return") { "func.return" } else { "gpu.return" };
                errors.push(MLIRValidationError {
                    line: lineno,
                    message: format!("{} inside scf region is invalid", ret_kind),
                });
            }
        }

        // Track region closes
        if trimmed == "}" {
            if let Some((open_line, kind)) = region_stack.pop() {
                // Check func/gpu.func has terminator
                if (kind == "func" || kind == "gpu.func") && !func_has_terminator {
                    errors.push(MLIRValidationError {
                        line: open_line,
                        message: format!("{} region has no terminator (func.return/gpu.return)", kind),
                    });
                }
                if kind == "func" || kind == "gpu.func" {
                    _in_func = false;
                }
            }
        }

        // Check for type mismatches in arith ops: both operands should have same type annotation
        if trimmed.contains("arith.addf") || trimmed.contains("arith.subf")
            || trimmed.contains("arith.mulf") || trimmed.contains("arith.divf")
            || trimmed.contains("arith.cmpf")
        {
            // The type after ':' should match the operand types
            // We can't fully check this without a type map, but we can check
            // that there's exactly one type annotation at the end
        }

        // Check for double terminators (two consecutive return-like statements)
        if lineno > 1 && (trimmed.starts_with("func.return") || trimmed.starts_with("gpu.return")) {
            let prev = lines[i - 1].trim();
            if prev.starts_with("func.return") || prev.starts_with("gpu.return") {
                errors.push(MLIRValidationError {
                    line: lineno,
                    message: "double terminator: consecutive return statements".to_string(),
                });
            }
        }
    }

    // Check unmatched regions
    for (line, kind) in &region_stack {
        errors.push(MLIRValidationError {
            line: *line,
            message: format!("unclosed {} region", kind),
        });
    }

    errors
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser;
    use insta::assert_snapshot;

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
        // Should NOT contain func.return inside a kernel
        assert!(!ir.contains("func.return"), "Kernel should not contain func.return, got: {}", ir);
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

    #[test]
    fn test_tensor_type_emission() {
        let ty = MLIRType::Tensor(Box::new(MLIRType::F32), vec![4, 8]);
        assert_eq!(format!("{}", ty), "tensor<4x8xf32>");
        let ty_dyn = MLIRType::Tensor(Box::new(MLIRType::F64), vec![-1]);
        assert_eq!(format!("{}", ty_dyn), "tensor<?xf64>");
    }

    #[test]
    fn test_matmul_codegen() {
        let ir = gen("fn mm(a: Tensor<f32, [4, 8]>, b: Tensor<f32, [8, 4]>) -> Tensor<f32, [4, 4]> { return a @ b }");
        assert!(ir.contains("linalg.matmul"), "Expected linalg.matmul in: {}", ir);
    }

    // --- New snapshot tests ---

    #[test]
    fn test_snapshot_array_literal() {
        let ir = gen("fn make_arr() -> i64 { let a = [1, 2, 3]\n return 0 }");
        assert_snapshot!(ir);
    }

    #[test]
    fn test_snapshot_indexing() {
        let ir = gen("fn get(a: i64) -> i64 { let arr = [10, 20, 30]\n let v = arr[1]\n return v }");
        assert_snapshot!(ir);
    }

    #[test]
    fn test_snapshot_field_access() {
        let ir = gen("fn get_field(s: i64) -> i64 { let v = s.x\n return v }");
        assert_snapshot!(ir);
    }

    #[test]
    fn test_snapshot_function_call_f64_return() {
        let ir = gen("fn square(x: f64) -> f64 { return x * x }\nfn main() -> f64 { return square(3.0) }");
        assert_snapshot!(ir);
    }

    #[test]
    fn test_snapshot_elementwise_ops() {
        let ir = gen("fn ew(a: Tensor<f32, [4]>, b: Tensor<f32, [4]>) -> Tensor<f32, [4]> { return a .* b }");
        assert_snapshot!(ir);
    }

    #[test]
    fn test_unsigned_ops() {
        let ir = gen("fn udiv(a: u64, b: u64) -> u64 { return a / b }");
        assert!(ir.contains("arith.divui"), "Expected arith.divui in: {}", ir);
    }

    #[test]
    fn test_unsigned_comparison() {
        let ir = gen("fn ucmp(a: u64, b: u64) -> bool { return a < b }");
        assert!(ir.contains("arith.cmpi ult"), "Expected arith.cmpi ult in: {}", ir);
    }

    #[test]
    fn test_unsigned_remainder() {
        let ir = gen("fn urem(a: u32, b: u32) -> u32 { return a % b }");
        assert!(ir.contains("arith.remui"), "Expected arith.remui in: {}", ir);
    }

    #[test]
    fn test_is_float_type_tensor_i32() {
        let ty = MLIRType::Tensor(Box::new(MLIRType::I32), vec![4]);
        assert!(!CodeGen::is_float_type(&ty), "Tensor<i32> should not be float");
    }

    #[test]
    fn test_is_float_type_tensor_f32() {
        let ty = MLIRType::Tensor(Box::new(MLIRType::F32), vec![4]);
        assert!(CodeGen::is_float_type(&ty), "Tensor<f32> should be float");
    }

    #[test]
    fn test_matmul_independent_operand_types() {
        let ir = gen("fn mm(a: Tensor<f32, [4, 8]>, b: Tensor<f32, [8, 4]>) -> Tensor<f32, [4, 4]> { return a @ b }");
        assert!(ir.contains("tensor<4x8xf32>, tensor<8x4xf32>"), "Expected independent operand types in: {}", ir);
    }

    // --- MLIR Validation Tests ---

    fn gen_and_validate(source: &str) -> (String, Vec<MLIRValidationError>) {
        let ir = gen(source);
        let errors = validate_mlir(&ir);
        (ir, errors)
    }

    #[test]
    fn test_validate_simple_function() {
        let (ir, errors) = gen_and_validate(
            "fn add(a: i64, b: i64) -> i64 { return a + b }"
        );
        assert!(errors.is_empty(), "Expected no errors for simple function, got: {:?}\nIR:\n{}", errors, ir);
    }

    #[test]
    fn test_validate_no_double_return_in_kernel() {
        let (ir, errors) = gen_and_validate(
            "kernel vadd(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> { return a + b }"
        );
        let double_returns: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("double terminator"))
            .collect();
        assert!(double_returns.is_empty(),
            "Should not have double terminator in kernel, got: {:?}\nIR:\n{}", double_returns, ir);
    }

    #[test]
    fn test_validate_no_func_return_in_scf() {
        let (ir, errors) = gen_and_validate(
            "fn factorial(n: i64) -> i64 { if n <= 1 { return 1 }\n return n * factorial(n - 1) }"
        );
        let scf_returns: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("inside scf region"))
            .collect();
        assert!(scf_returns.is_empty(),
            "Should not have func.return inside scf.if, got: {:?}\nIR:\n{}", scf_returns, ir);
    }

    #[test]
    fn test_validate_no_duplicate_ssa() {
        let (ir, errors) = gen_and_validate(
            "fn test() -> i64 { let a = 1\n let b = 2\n return a + b }"
        );
        let dup_ssas: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("defined more than once"))
            .collect();
        assert!(dup_ssas.is_empty(),
            "Should not have duplicate SSA defs, got: {:?}\nIR:\n{}", dup_ssas, ir);
    }

    #[test]
    fn test_validate_type_cast_in_comparison() {
        // f32 param compared with float literal (which defaults to f64)
        // Should insert a cast so types match
        let (ir, _errors) = gen_and_validate(
            "fn abs(x: f32) -> f32 { if x > 0.0 { x } else { 0.0 - x } }"
        );
        // Check that there's a truncf or extf cast
        assert!(ir.contains("arith.truncf") || !ir.contains("cmpf ogt, %"),
            "Should insert type cast for f32 vs f64 comparison\nIR:\n{}", ir);
    }

    #[test]
    fn test_validate_for_loop() {
        let (ir, errors) = gen_and_validate(
            "fn sum(n: i64) -> i64 { var s: i64 = 0\n for i in 0..n { s += i }\n return s }"
        );
        let critical: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("inside scf") || e.message.contains("no terminator"))
            .collect();
        assert!(critical.is_empty(),
            "For loop should not cause structural errors, got: {:?}\nIR:\n{}", critical, ir);
    }

    #[test]
    fn test_validate_while_loop() {
        let (ir, errors) = gen_and_validate(
            "fn countdown(n: i64) -> i64 { var x = n\n while x > 0 { x -= 1 }\n return x }"
        );
        let critical: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("inside scf") || e.message.contains("no terminator"))
            .collect();
        assert!(critical.is_empty(),
            "While loop should not cause structural errors, got: {:?}\nIR:\n{}", critical, ir);
    }

    #[test]
    fn test_validate_void_function_has_terminator() {
        let (ir, errors) = gen_and_validate(
            "fn noop() { let x = 42 }"
        );
        let no_term: Vec<_> = errors.iter()
            .filter(|e| e.message.contains("no terminator"))
            .collect();
        assert!(no_term.is_empty(),
            "Void function should have func.return terminator, got: {:?}\nIR:\n{}", no_term, ir);
    }

    /// Test that generated MLIR is accepted by mlir-opt-20 if available.
    /// This is an integration test that requires mlir-20-tools to be installed.
    #[test]
    fn test_mlir_opt_validates_simple_function() {
        let ir = gen("fn add(a: i64, b: i64) -> i64 { return a + b }");

        // Check if mlir-opt-20 is available
        let which = std::process::Command::new("which")
            .arg("mlir-opt-20")
            .output();
        if which.is_err() || !which.unwrap().status.success() {
            eprintln!("mlir-opt-20 not found, skipping integration test");
            return;
        }

        let mut child = std::process::Command::new("mlir-opt-20")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to spawn mlir-opt-20");

        use std::io::Write;
        child.stdin.take().unwrap().write_all(ir.as_bytes()).unwrap();
        let output = child.wait_with_output().unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("mlir-opt-20 rejected the IR:\n{}\n\nGenerated IR:\n{}", stderr, ir);
        }
    }

    #[test]
    fn test_mlir_opt_validates_for_loop() {
        run_mlir_opt_or_skip(&gen("fn sum(n: i64) -> i64 { var s: i64 = 0\n for i in 0..n { s += i }\n return s }"));
    }

    /// Helper: run mlir-opt-20 on IR, panic if rejected, skip if not installed.
    fn run_mlir_opt_or_skip(ir: &str) {
        let which = std::process::Command::new("which")
            .arg("mlir-opt-20")
            .output();
        if which.is_err() || !which.unwrap().status.success() {
            eprintln!("mlir-opt-20 not found, skipping");
            return;
        }

        let mut child = std::process::Command::new("mlir-opt-20")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to spawn mlir-opt-20");

        use std::io::Write;
        child.stdin.take().unwrap().write_all(ir.as_bytes()).unwrap();
        let output = child.wait_with_output().unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("mlir-opt-20 rejected the IR:\n{}\n\nGenerated IR:\n{}", stderr, ir);
        }
    }

    /// Helper: run mlir-opt-20 with full lowering passes
    fn run_mlir_opt_full_pipeline_or_skip(ir: &str) {
        let which = std::process::Command::new("which")
            .arg("mlir-opt-20")
            .output();
        if which.is_err() || !which.unwrap().status.success() {
            eprintln!("mlir-opt-20 not found, skipping");
            return;
        }

        let mut child = std::process::Command::new("mlir-opt-20")
            .args(&[
                "--canonicalize",
                "--cse",
                "--convert-scf-to-cf",
                "--convert-func-to-llvm",
                "--convert-arith-to-llvm",
                "--convert-cf-to-llvm",
                "--finalize-memref-to-llvm",
                "--reconcile-unrealized-casts",
            ])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .expect("failed to spawn mlir-opt-20");

        use std::io::Write;
        child.stdin.take().unwrap().write_all(ir.as_bytes()).unwrap();
        let output = child.wait_with_output().unwrap();

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            panic!("mlir-opt-20 full pipeline rejected the IR:\n{}\n\nGenerated IR:\n{}", stderr, ir);
        }
    }

    #[test]
    fn test_mlir_opt_full_pipeline_simple_add() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn add(a: i64, b: i64) -> i64 { return a + b }"));
    }

    #[test]
    fn test_mlir_opt_validates_void_function() {
        run_mlir_opt_or_skip(&gen("fn noop() { let x = 42 }"));
    }

    #[test]
    fn test_mlir_opt_validates_if_else() {
        run_mlir_opt_or_skip(&gen("fn abs(x: i64) -> i64 { if x > 0 { return x } else { return 0 - x } }"));
    }

    #[test]
    fn test_mlir_opt_validates_while_loop() {
        run_mlir_opt_or_skip(&gen("fn countdown(n: i64) -> i64 { var x = n\n while x > 0 { x -= 1 }\n return x }"));
    }

    #[test]
    fn test_mlir_opt_validates_bool_ops() {
        run_mlir_opt_or_skip(&gen("fn both(a: bool, b: bool) -> bool { return a && b }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_for_loop() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn sum(n: i64) -> i64 { var s: i64 = 0\n for i in 0..n { s += i }\n return s }"));
    }

    #[test]
    fn test_mlir_opt_validates_kernel_basic() {
        run_mlir_opt_or_skip(&gen("kernel nop() { let x = 42 }"));
    }

    #[test]
    fn test_mlir_opt_validates_array_literal() {
        run_mlir_opt_or_skip(&gen("fn make() -> i64 { let a = [1, 2, 3]\n return 0 }"));
    }

    #[test]
    fn test_mlir_opt_validates_cast() {
        run_mlir_opt_or_skip(&gen("fn to_float(x: i64) -> f64 { return x as f64 }"));
    }

    #[test]
    fn test_mlir_opt_validates_multiple_functions() {
        run_mlir_opt_or_skip(&gen("fn square(x: i64) -> i64 { return x * x }\nfn main() -> i64 { return square(5) }"));
    }

    #[test]
    fn test_mlir_opt_validates_negation() {
        run_mlir_opt_or_skip(&gen("fn neg(x: i64) -> i64 { return 0 - x }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_void_function() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn noop() { let x = 42 }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_if_else() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn max(a: i64, b: i64) -> i64 { if a > b { return a } else { return b } }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_while() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn countdown(n: i64) -> i64 { var x = n\n while x > 0 { x -= 1 }\n return x }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_array() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn make() -> i64 { let a = [1, 2, 3]\n return 0 }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_cast() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn to_float(x: i64) -> f64 { return x as f64 }"));
    }

    #[test]
    fn test_mlir_opt_full_pipeline_multi_fn() {
        run_mlir_opt_full_pipeline_or_skip(&gen("fn square(x: i64) -> i64 { return x * x }\nfn main() -> i64 { return square(5) }"));
    }

    #[test]
    fn test_mlir_opt_validates_fibonacci() {
        // This has if inside function with early return - tricky scf pattern
        run_mlir_opt_or_skip(&gen("fn fib(n: i64) -> i64 { if n <= 1 { return n }\n return fib(n - 1) + fib(n - 2) }"));
    }

    #[test]
    fn test_mlir_opt_validates_kernel_with_return() {
        run_mlir_opt_or_skip(&gen("kernel add(a: i64, b: i64) -> i64 { return a + b }"));
    }
}
