use crate::ast::*;

/// A warning about potential timing side-channel
#[derive(Debug, Clone)]
pub struct CTWarning {
    pub message: String,
    pub span: crate::lexer::Span,
    pub severity: CTSeverity,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CTSeverity {
    Error,   // Definite timing leak
    Warning, // Potential timing leak
    Info,    // Informational
}

impl std::fmt::Display for CTWarning {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let level = match self.severity {
            CTSeverity::Error => "ERROR",
            CTSeverity::Warning => "WARNING",
            CTSeverity::Info => "INFO",
        };
        write!(f, "[CT-{}] {}", level, self.message)
    }
}

/// Check a program for constant-time violations
/// Returns a list of warnings
pub fn check_constant_time(program: &Program) -> Vec<CTWarning> {
    let mut checker = CTChecker::new();
    checker.check_program(program);
    checker.warnings
}

struct CTChecker {
    warnings: Vec<CTWarning>,
    /// Variables marked as secret (from @constant_time annotated functions)
    secret_vars: Vec<String>,
    /// Whether we're inside a constant-time annotated function
    in_ct_function: bool,
}

impl CTChecker {
    fn new() -> Self {
        Self {
            warnings: Vec::new(),
            secret_vars: Vec::new(),
            in_ct_function: false,
        }
    }

    fn check_program(&mut self, program: &Program) {
        for item in &program.items {
            self.check_item(item);
        }
    }

    fn check_item(&mut self, item: &Item) {
        match &item.kind {
            ItemKind::Function(func) => {
                // Check if function has @constant_time annotation
                // For now, check if the function name contains "ct_" prefix
                // or if any parameter type suggests secret data
                let is_ct = func.name.name.starts_with("ct_") || self.has_secret_params(&func.params);

                if is_ct {
                    self.in_ct_function = true;
                    // All parameters are considered secret
                    self.secret_vars = func.params.iter().map(|p| p.name.name.clone()).collect();
                    self.check_block(&func.body);
                    self.in_ct_function = false;
                    self.secret_vars.clear();
                } else {
                    self.check_block(&func.body);
                }
            }
            ItemKind::Kernel(kernel) => {
                self.check_block(&kernel.body);
            }
            ItemKind::Impl(impl_block) => {
                for method in &impl_block.methods {
                    self.check_item(method);
                }
            }
            _ => {}
        }
    }

    fn has_secret_params(&self, params: &[Param]) -> bool {
        params.iter().any(|p| {
            // Heuristic: parameters named "secret", "key", "privkey", "scalar" are secret
            let name = p.name.name.to_lowercase();
            name.contains("secret") || name.contains("key") || name.contains("priv") || name.contains("scalar")
        })
    }

    fn check_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        if let Some(expr) = &block.expr {
            self.check_expr(expr);
        }
    }

    fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                // Track if the variable depends on secret data
                if self.expr_depends_on_secret(value) {
                    self.secret_vars.push(name.name.clone());
                }
                self.check_expr(value);
            }
            StmtKind::Expr(expr) => self.check_expr(expr),
            StmtKind::Return(Some(expr)) => self.check_expr(expr),
            StmtKind::Return(None) => {}
            StmtKind::Assign { target, value, .. } => {
                self.check_expr(target);
                self.check_expr(value);
            }
            StmtKind::For { iter, body, .. } => {
                if self.in_ct_function && self.expr_depends_on_secret(iter) {
                    self.warnings.push(CTWarning {
                        message: "for loop with secret-dependent iteration count".to_string(),
                        span: stmt.span,
                        severity: CTSeverity::Error,
                    });
                }
                self.check_expr(iter);
                self.check_block(body);
            }
            StmtKind::While { cond, body } => {
                if self.in_ct_function && self.expr_depends_on_secret(cond) {
                    self.warnings.push(CTWarning {
                        message: "while loop with secret-dependent condition".to_string(),
                        span: stmt.span,
                        severity: CTSeverity::Error,
                    });
                }
                self.check_expr(cond);
                self.check_block(body);
            }
            StmtKind::Loop { body } => {
                self.check_block(body);
            }
            StmtKind::Break | StmtKind::Continue => {}
            StmtKind::Dispatch { index, args, .. } => {
                self.check_expr(index);
                for arg in args {
                    self.check_expr(arg);
                }
            }
            StmtKind::Live { value, .. } => {
                self.check_expr(value);
            }
            StmtKind::Fuse { body } | StmtKind::Deterministic { body } => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::GpuLet { value, .. } => { self.check_expr(value); }
            StmtKind::Parallel { iter, body, .. } => {
                self.check_expr(iter);
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::Train { config, .. } => {
                for (_, v) in config { self.check_expr(v); }
            }
            StmtKind::Autocast { body, .. }
            | StmtKind::Speculate { body }
            | StmtKind::Explain { body }
            | StmtKind::Quantize { body, .. }  => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::Safe { body, config, .. } => {
                for (_, v) in config { self.check_expr(v); }
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::Topology { config, .. } => {
                for (_, v) in config { self.check_expr(v); }
            }
            StmtKind::Mmap { value, .. } => { self.check_expr(value); }
            StmtKind::Consensus { body }
            | StmtKind::SymbolicBlock { body }
            | StmtKind::TemporalBlock { body }
            | StmtKind::Federated { body }
            | StmtKind::SandboxBlock { body }
            | StmtKind::Metacognition { body } => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::Compress { body, .. } => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::TheoremBlock { body, .. }
            | StmtKind::ContinualLearn { body }
            | StmtKind::MultimodalBlock { body, .. }
            | StmtKind::WorldModelBlock { body }
            | StmtKind::SelfImproveBlock { body } => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::MemoryBlock { config, .. } => {
                for (_, v) in config { self.check_expr(v); }
            }
            StmtKind::AttentionBlock { body }
            | StmtKind::EnsembleBlock { body }
            | StmtKind::AdversarialBlock { body }
            | StmtKind::TransferBlock { body }
            | StmtKind::SparseBlock { body }
            | StmtKind::AsyncInferBlock { body }
            | StmtKind::ProfileBlock { body }
            | StmtKind::ContractBlock { body, .. } => {
                for s in &body.stmts { self.check_stmt(s); }
            }
            StmtKind::CurriculumBlock { config, body } => {
                for (_, v) in config { self.check_expr(v); }
                for s in &body.stmts { self.check_stmt(s); }
            }
        }
    }

    fn check_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::If { cond, then_block, else_block } => {
                if self.in_ct_function && self.expr_depends_on_secret(cond) {
                    self.warnings.push(CTWarning {
                        message: "conditional branch depends on secret data".to_string(),
                        span: expr.span,
                        severity: CTSeverity::Error,
                    });
                }
                self.check_expr(cond);
                self.check_block(then_block);
                if let Some(eb) = else_block {
                    self.check_block(eb);
                }
            }
            ExprKind::Match { expr: match_expr, arms } => {
                if self.in_ct_function && self.expr_depends_on_secret(match_expr) {
                    self.warnings.push(CTWarning {
                        message: "match expression depends on secret data".to_string(),
                        span: expr.span,
                        severity: CTSeverity::Error,
                    });
                }
                self.check_expr(match_expr);
                for arm in arms {
                    self.check_expr(&arm.body);
                }
            }
            ExprKind::Binary { lhs, op, rhs } => {
                // Check for early-exit comparisons
                if self.in_ct_function {
                    if matches!(op, BinOp::Eq | BinOp::NotEq) {
                        if self.expr_depends_on_secret(lhs) || self.expr_depends_on_secret(rhs) {
                            self.warnings.push(CTWarning {
                                message: "equality comparison with secret data may leak via timing".to_string(),
                                span: expr.span,
                                severity: CTSeverity::Warning,
                            });
                        }
                    }
                }
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            ExprKind::Call { func, args } => {
                self.check_expr(func);
                for arg in args {
                    self.check_expr(arg);
                }
            }
            ExprKind::Block(block) => self.check_block(block),
            ExprKind::Index { base, indices } => {
                if self.in_ct_function {
                    for idx in indices {
                        if self.expr_depends_on_secret(idx) {
                            self.warnings.push(CTWarning {
                                message: "array index depends on secret data (cache-timing attack)".to_string(),
                                span: expr.span,
                                severity: CTSeverity::Error,
                            });
                        }
                    }
                }
                self.check_expr(base);
                for idx in indices {
                    self.check_expr(idx);
                }
            }
            ExprKind::FieldAccess { base, .. } => self.check_expr(base),
            ExprKind::Unary { expr: inner, .. } => self.check_expr(inner),
            ExprKind::Range { start, end } => {
                self.check_expr(start);
                self.check_expr(end);
            }
            ExprKind::ArrayLiteral(elems) => {
                for e in elems { self.check_expr(e); }
            }
            ExprKind::Cast { expr: inner, .. } => self.check_expr(inner),
            ExprKind::MatMul { lhs, rhs } => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            ExprKind::StructLiteral { fields, .. } => {
                for (_, fexpr) in fields { self.check_expr(fexpr); }
            }
            _ => {}
        }
    }

    fn expr_depends_on_secret(&self, expr: &Expr) -> bool {
        if !self.in_ct_function {
            return false;
        }
        match &expr.kind {
            ExprKind::Ident(id) => self.secret_vars.contains(&id.name),
            ExprKind::Binary { lhs, rhs, .. } => {
                self.expr_depends_on_secret(lhs) || self.expr_depends_on_secret(rhs)
            }
            ExprKind::Unary { expr: inner, .. } => self.expr_depends_on_secret(inner),
            ExprKind::Call { args, .. } => args.iter().any(|a| self.expr_depends_on_secret(a)),
            ExprKind::FieldAccess { base, .. } => self.expr_depends_on_secret(base),
            ExprKind::Index { base, indices } => {
                self.expr_depends_on_secret(base) || indices.iter().any(|i| self.expr_depends_on_secret(i))
            }
            ExprKind::Cast { expr: inner, .. } => self.expr_depends_on_secret(inner),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer;
    use crate::parser;

    fn check_source(source: &str) -> Vec<CTWarning> {
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).expect("parse error");
        check_constant_time(&program)
    }

    #[test]
    fn test_no_warnings_regular_function() {
        let warnings = check_source("fn main() {\n    let x = 5\n    if x > 3 {\n        println(\"yes\")\n    }\n}\n");
        // Regular functions should have no CT warnings
        assert!(warnings.is_empty());
    }

    #[test]
    fn test_secret_dependent_branch() {
        let warnings = check_source("fn ct_verify(secret_key: u64) {\n    if secret_key > 0 {\n        println(\"has key\")\n    }\n}\nfn main() {\n}\n");
        // Should warn about secret-dependent branch
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("conditional branch")));
    }

    #[test]
    fn test_secret_dependent_index() {
        let warnings = check_source("fn ct_lookup(secret_idx: u64) {\n    let table = [1, 2, 3, 4]\n    let val = table[secret_idx]\n}\nfn main() {\n}\n");
        assert!(!warnings.is_empty());
        assert!(warnings.iter().any(|w| w.message.contains("array index")));
    }
}
