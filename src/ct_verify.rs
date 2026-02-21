use std::collections::{HashMap, HashSet};
use std::fmt;

use crate::ast::*;

/// Secrecy classification
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Secrecy {
    Public,
    Secret,
    Declassified,
}

/// Timing-unsafe operations (branching on secret values)
#[derive(Clone, Debug)]
pub enum CTViolation {
    SecretBranch { location: String, variable: String },
    SecretIndex { location: String, variable: String },
    SecretLoopBound { location: String, variable: String },
    SecretEarlyReturn { location: String, variable: String },
    SecretDivision { location: String, variable: String },
    UnverifiedCall { location: String, function: String },
}

impl fmt::Display for CTViolation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CTViolation::SecretBranch { location, variable } => {
                write!(f, "[{}] secret branch on `{}`", location, variable)
            }
            CTViolation::SecretIndex { location, variable } => {
                write!(f, "[{}] secret index `{}`", location, variable)
            }
            CTViolation::SecretLoopBound { location, variable } => {
                write!(f, "[{}] secret loop bound `{}`", location, variable)
            }
            CTViolation::SecretEarlyReturn { location, variable } => {
                write!(f, "[{}] secret early return on `{}`", location, variable)
            }
            CTViolation::SecretDivision { location, variable } => {
                write!(f, "[{}] secret division by `{}`", location, variable)
            }
            CTViolation::UnverifiedCall { location, function } => {
                write!(f, "[{}] unverified call to `{}`", location, function)
            }
        }
    }
}

pub struct CTVerifier {
    env: HashMap<String, Secrecy>,
    ct_functions: HashSet<String>,
    violations: Vec<CTViolation>,
    /// Track whether we are inside a secret-conditioned context
    in_secret_branch: bool,
}

impl CTVerifier {
    pub fn new() -> Self {
        Self {
            env: HashMap::new(),
            ct_functions: HashSet::new(),
            violations: Vec::new(),
            in_secret_branch: false,
        }
    }

    pub fn mark_secret(&mut self, name: &str) {
        self.env.insert(name.to_string(), Secrecy::Secret);
    }

    pub fn mark_public(&mut self, name: &str) {
        self.env.insert(name.to_string(), Secrecy::Public);
    }

    pub fn mark_declassified(&mut self, name: &str) {
        self.env.insert(name.to_string(), Secrecy::Declassified);
    }

    /// Classify an expression's secrecy: if any input is secret, output is secret.
    pub fn classify_expr(&self, expr: &Expr) -> Secrecy {
        match &expr.kind {
            ExprKind::IntLiteral(_)
            | ExprKind::FloatLiteral(_)
            | ExprKind::StringLiteral(_)
            | ExprKind::BoolLiteral(_) => Secrecy::Public,

            ExprKind::Ident(id) => self
                .env
                .get(&id.name)
                .cloned()
                .unwrap_or(Secrecy::Public),

            ExprKind::Binary { lhs, op: _, rhs } => {
                let l = self.classify_expr(lhs);
                let r = self.classify_expr(rhs);
                Self::join(&l, &r)
            }

            ExprKind::Unary { op: _, expr } => self.classify_expr(expr),

            ExprKind::Call { func, args } => {
                // If any arg is secret, result is secret
                let mut sec = Secrecy::Public;
                for a in args {
                    sec = Self::join(&sec, &self.classify_expr(a));
                }
                // Also consider the function expression itself
                sec = Self::join(&sec, &self.classify_expr(func));
                sec
            }

            ExprKind::MatMul { lhs, rhs } => {
                Self::join(&self.classify_expr(lhs), &self.classify_expr(rhs))
            }

            ExprKind::FieldAccess { base, .. } => self.classify_expr(base),

            ExprKind::Index { base, indices } => {
                let mut sec = self.classify_expr(base);
                for idx in indices {
                    sec = Self::join(&sec, &self.classify_expr(idx));
                }
                sec
            }

            ExprKind::Cast { expr, .. } => self.classify_expr(expr),

            ExprKind::If { cond, .. } => self.classify_expr(cond),

            ExprKind::Range { start, end } => {
                Self::join(&self.classify_expr(start), &self.classify_expr(end))
            }

            ExprKind::ArrayLiteral(elems) => {
                let mut sec = Secrecy::Public;
                for e in elems {
                    sec = Self::join(&sec, &self.classify_expr(e));
                }
                sec
            }

            ExprKind::Block(block) => {
                if let Some(e) = &block.expr {
                    self.classify_expr(e)
                } else {
                    Secrecy::Public
                }
            }

            _ => Secrecy::Public,
        }
    }

    /// Join two secrecy levels: Secret wins over everything except Declassified stays Declassified.
    fn join(a: &Secrecy, b: &Secrecy) -> Secrecy {
        match (a, b) {
            (Secrecy::Secret, Secrecy::Declassified) | (Secrecy::Declassified, Secrecy::Secret) => {
                Secrecy::Secret
            }
            (Secrecy::Secret, _) | (_, Secrecy::Secret) => Secrecy::Secret,
            (Secrecy::Declassified, _) | (_, Secrecy::Declassified) => Secrecy::Declassified,
            _ => Secrecy::Public,
        }
    }

    /// Get the "representative variable" name from an expression for error reporting.
    fn expr_var_name(expr: &Expr) -> String {
        match &expr.kind {
            ExprKind::Ident(id) => id.name.clone(),
            ExprKind::FieldAccess { base, field } => {
                format!("{}.{}", Self::expr_var_name(base), field.name)
            }
            ExprKind::Binary { lhs, .. } => Self::expr_var_name(lhs),
            ExprKind::Unary { expr, .. } => Self::expr_var_name(expr),
            _ => "<expr>".to_string(),
        }
    }

    /// Find secret variables referenced in an expression.
    fn find_secret_vars(&self, expr: &Expr) -> Vec<String> {
        let mut vars = Vec::new();
        self.collect_secret_vars(expr, &mut vars);
        vars
    }

    fn collect_secret_vars(&self, expr: &Expr, vars: &mut Vec<String>) {
        match &expr.kind {
            ExprKind::Ident(id) => {
                if self.env.get(&id.name) == Some(&Secrecy::Secret) {
                    vars.push(id.name.clone());
                }
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                self.collect_secret_vars(lhs, vars);
                self.collect_secret_vars(rhs, vars);
            }
            ExprKind::Unary { expr, .. } => {
                self.collect_secret_vars(expr, vars);
            }
            ExprKind::Call { func, args } => {
                self.collect_secret_vars(func, vars);
                for a in args {
                    self.collect_secret_vars(a, vars);
                }
            }
            ExprKind::FieldAccess { base, .. } => {
                self.collect_secret_vars(base, vars);
            }
            ExprKind::Index { base, indices } => {
                self.collect_secret_vars(base, vars);
                for idx in indices {
                    self.collect_secret_vars(idx, vars);
                }
            }
            ExprKind::Range { start, end } => {
                self.collect_secret_vars(start, vars);
                self.collect_secret_vars(end, vars);
            }
            _ => {}
        }
    }

    fn location(span: &crate::lexer::Span) -> String {
        format!("{}..{}", span.start, span.end)
    }

    /// Check an expression for CT violations (division, indexing, branches).
    fn check_expr(&mut self, expr: &Expr) {
        match &expr.kind {
            ExprKind::Binary { lhs, op, rhs } => {
                // Check for secret division
                if matches!(op, BinOp::Div | BinOp::Mod) {
                    let rhs_sec = self.classify_expr(rhs);
                    if rhs_sec == Secrecy::Secret {
                        self.violations.push(CTViolation::SecretDivision {
                            location: Self::location(&expr.span),
                            variable: Self::expr_var_name(rhs),
                        });
                    }
                }
                self.check_expr(lhs);
                self.check_expr(rhs);
            }

            ExprKind::Index { base, indices } => {
                for idx in indices {
                    let idx_sec = self.classify_expr(idx);
                    if idx_sec == Secrecy::Secret {
                        self.violations.push(CTViolation::SecretIndex {
                            location: Self::location(&expr.span),
                            variable: Self::expr_var_name(idx),
                        });
                    }
                    self.check_expr(idx);
                }
                self.check_expr(base);
            }

            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                let cond_sec = self.classify_expr(cond);
                if cond_sec == Secrecy::Secret {
                    self.violations.push(CTViolation::SecretBranch {
                        location: Self::location(&expr.span),
                        variable: Self::expr_var_name(cond),
                    });
                }
                self.check_expr(cond);
                self.check_block(then_block);
                if let Some(eb) = else_block {
                    self.check_block(eb);
                }
            }

            ExprKind::Call { func, args } => {
                // Check if calling an unverified function
                if let ExprKind::Ident(id) = &func.kind {
                    if !self.ct_functions.contains(&id.name) {
                        // Built-in ops are okay; only flag user functions
                        // We'll be conservative: flag calls to non-verified functions
                        // Skip common builtins
                        let builtins = [
                            "print", "println", "len", "push", "pop", "assert",
                            "ct_select", "ct_eq", "ct_lt", "ct_lookup", "ct_memcmp", "ct_swap",
                        ];
                        if !builtins.contains(&id.name.as_str()) {
                            self.violations.push(CTViolation::UnverifiedCall {
                                location: Self::location(&expr.span),
                                function: id.name.clone(),
                            });
                        }
                    }
                }
                for a in args {
                    self.check_expr(a);
                }
            }

            ExprKind::Match { expr: scrutinee, arms } => {
                let sec = self.classify_expr(scrutinee);
                if sec == Secrecy::Secret {
                    self.violations.push(CTViolation::SecretBranch {
                        location: Self::location(&scrutinee.span),
                        variable: Self::expr_var_name(scrutinee),
                    });
                }
                self.check_expr(scrutinee);
                for arm in arms {
                    self.check_expr(&arm.body);
                }
            }

            ExprKind::Block(block) => {
                self.check_block(block);
            }

            ExprKind::Unary { expr, .. } => self.check_expr(expr),
            ExprKind::MatMul { lhs, rhs } => {
                self.check_expr(lhs);
                self.check_expr(rhs);
            }
            ExprKind::FieldAccess { base, .. } => self.check_expr(base),
            ExprKind::Cast { expr, .. } => self.check_expr(expr),
            ExprKind::Range { start, end } => {
                self.check_expr(start);
                self.check_expr(end);
            }
            ExprKind::ArrayLiteral(elems) => {
                for e in elems {
                    self.check_expr(e);
                }
            }
            _ => {}
        }
    }

    fn check_block(&mut self, block: &Block) {
        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }
        if let Some(expr) = &block.expr {
            self.check_expr(expr);
        }
    }

    /// Check a statement for CT violations.
    pub fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::Let { name, value, .. } | StmtKind::Var { name, value, .. } => {
                let sec = self.classify_expr(value);
                self.env.insert(name.name.clone(), sec);
                self.check_expr(value);
            }

            StmtKind::Return(Some(expr)) => {
                if self.in_secret_branch {
                    let sec = self.classify_expr(expr);
                    if sec == Secrecy::Secret {
                        self.violations.push(CTViolation::SecretEarlyReturn {
                            location: Self::location(&stmt.span),
                            variable: Self::expr_var_name(expr),
                        });
                    }
                }
                self.check_expr(expr);
            }

            StmtKind::Expr(expr) => {
                self.check_expr(expr);
            }

            StmtKind::Assign { target, value, op } => {
                // Check for secret division in compound assignment
                if matches!(op, AssignOp::DivAssign) {
                    let val_sec = self.classify_expr(value);
                    if val_sec == Secrecy::Secret {
                        self.violations.push(CTViolation::SecretDivision {
                            location: Self::location(&stmt.span),
                            variable: Self::expr_var_name(value),
                        });
                    }
                }
                // Propagate taint to target
                let val_sec = self.classify_expr(value);
                let tgt_sec = self.classify_expr(target);
                let combined = Self::join(&val_sec, &tgt_sec);
                if let ExprKind::Ident(id) = &target.kind {
                    self.env.insert(id.name.clone(), combined);
                }
                self.check_expr(value);
            }

            StmtKind::For { var, iter, body } => {
                // Check if loop bound depends on secret
                let iter_sec = self.classify_expr(iter);
                if iter_sec == Secrecy::Secret {
                    let secret_vars = self.find_secret_vars(iter);
                    let var_name = if secret_vars.is_empty() {
                        Self::expr_var_name(iter)
                    } else {
                        secret_vars[0].clone()
                    };
                    self.violations.push(CTViolation::SecretLoopBound {
                        location: Self::location(&stmt.span),
                        variable: var_name,
                    });
                }
                self.env.insert(var.name.clone(), Secrecy::Public);
                self.check_expr(iter);
                self.check_block(body);
            }

            StmtKind::While { cond, body } => {
                let cond_sec = self.classify_expr(cond);
                if cond_sec == Secrecy::Secret {
                    self.violations.push(CTViolation::SecretBranch {
                        location: Self::location(&stmt.span),
                        variable: Self::expr_var_name(cond),
                    });
                }
                self.check_expr(cond);
                self.check_block(body);
            }

            StmtKind::Loop { body } => {
                self.check_block(body);
            }

            _ => {}
        }
    }

    /// Check a function body with the given parameter secrecy annotations.
    pub fn check_function(
        &mut self,
        name: &str,
        params: &[(String, Secrecy)],
        body: &[Stmt],
    ) {
        // Set up parameter environment
        for (pname, sec) in params {
            self.env.insert(pname.clone(), sec.clone());
        }

        // Check each statement
        for stmt in body {
            self.check_stmt(stmt);
        }

        // If no violations, mark function as verified
        if self.violations.is_empty() {
            self.ct_functions.insert(name.to_string());
        }
    }

    /// Check an entire program. Functions annotated with @constant_time have
    /// their "secret" params (those with type names containing "secret" or
    /// annotated) verified.
    pub fn check_program(&mut self, items: &[Item]) {
        for item in items {
            if let ItemKind::Function(func) = &item.kind {
                let has_ct_annotation = func.annotations.iter().any(|a| {
                    // We look for a @scan-style annotation named "constant_time"
                    // For now, check function name convention or annotation
                    matches!(a, Annotation::Scan(Some(params)) if params.iter().any(|(k, _)| k.name == "constant_time"))
                });

                // Also check if function name starts with "ct_"
                if has_ct_annotation || func.name.name.starts_with("ct_") {
                    let params: Vec<(String, Secrecy)> = func
                        .params
                        .iter()
                        .map(|p| {
                            // Mark params whose type name contains "Secret" as secret
                            let sec = if format!("{}", p.ty)
                                .to_lowercase()
                                .contains("secret")
                            {
                                Secrecy::Secret
                            } else {
                                Secrecy::Public
                            };
                            (p.name.name.clone(), sec)
                        })
                        .collect();

                    self.check_function(
                        &func.name.name,
                        &params,
                        &func.body.stmts,
                    );
                }
            }
        }
    }

    pub fn violations(&self) -> &[CTViolation] {
        &self.violations
    }

    pub fn format_report(&self) -> String {
        if self.violations.is_empty() {
            return "No constant-time violations found.".to_string();
        }

        let mut report = format!(
            "Constant-time verification failed: {} violation(s)\n",
            self.violations.len()
        );
        for (i, v) in self.violations.iter().enumerate() {
            report.push_str(&format!("  {}. {}\n", i + 1, v));
        }
        report
    }

    pub fn is_verified(&self, name: &str) -> bool {
        self.ct_functions.contains(name)
    }
}

/// Constant-time safe primitive operations.
pub struct CTOps;

impl CTOps {
    /// Constant-time conditional select: result = if cond { a } else { b }
    pub fn ct_select(cond: bool, a: u64, b: u64) -> u64 {
        let mask = if cond { u64::MAX } else { 0 };
        (a & mask) | (b & !mask)
    }

    /// Constant-time equality comparison
    pub fn ct_eq(a: u64, b: u64) -> bool {
        let diff = a ^ b;
        diff == 0
    }

    /// Constant-time less-than
    pub fn ct_lt(a: u64, b: u64) -> bool {
        let diff = (a as i64).wrapping_sub(b as i64);
        diff < 0
    }

    /// Constant-time array lookup (reads ALL elements, selects by mask)
    pub fn ct_lookup(table: &[u64], index: u64) -> u64 {
        let mut result = 0u64;
        for (i, &val) in table.iter().enumerate() {
            let mask = Self::ct_select(i as u64 == index, u64::MAX, 0);
            result |= val & mask;
        }
        result
    }

    /// Constant-time memcmp
    pub fn ct_memcmp(a: &[u8], b: &[u8]) -> bool {
        if a.len() != b.len() {
            return false;
        }
        let mut diff = 0u8;
        for i in 0..a.len() {
            diff |= a[i] ^ b[i];
        }
        diff == 0
    }

    /// Constant-time conditional swap
    pub fn ct_swap(cond: bool, a: &mut u64, b: &mut u64) {
        let mask = if cond { u64::MAX } else { 0 };
        let t = (*a ^ *b) & mask;
        *a ^= t;
        *b ^= t;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Span;

    fn dummy_span() -> Span {
        Span { start: 0, end: 0 }
    }

    fn make_ident(name: &str) -> Ident {
        Ident::new(name.to_string(), dummy_span())
    }

    fn make_ident_expr(name: &str) -> Expr {
        Expr {
            kind: ExprKind::Ident(make_ident(name)),
            span: dummy_span(),
        }
    }

    fn make_int_expr(n: u128) -> Expr {
        Expr {
            kind: ExprKind::IntLiteral(n),
            span: dummy_span(),
        }
    }

    fn make_binary(lhs: Expr, op: BinOp, rhs: Expr) -> Expr {
        Expr {
            kind: ExprKind::Binary {
                lhs: Box::new(lhs),
                op,
                rhs: Box::new(rhs),
            },
            span: dummy_span(),
        }
    }

    fn make_if_stmt(cond: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::Expr(Expr {
                kind: ExprKind::If {
                    cond: Box::new(cond),
                    then_block: Block {
                        stmts: vec![],
                        expr: None,
                        span: dummy_span(),
                    },
                    else_block: None,
                },
                span: dummy_span(),
            }),
            span: dummy_span(),
        }
    }

    fn make_index_expr(base: &str, idx: &str) -> Expr {
        Expr {
            kind: ExprKind::Index {
                base: Box::new(make_ident_expr(base)),
                indices: vec![make_ident_expr(idx)],
            },
            span: dummy_span(),
        }
    }

    fn make_for_stmt(var: &str, bound: Expr) -> Stmt {
        Stmt {
            kind: StmtKind::For {
                var: make_ident(var),
                iter: Expr {
                    kind: ExprKind::Range {
                        start: Box::new(make_int_expr(0)),
                        end: Box::new(bound),
                    },
                    span: dummy_span(),
                },
                body: Block {
                    stmts: vec![],
                    expr: None,
                    span: dummy_span(),
                },
            },
            span: dummy_span(),
        }
    }

    // 1. Secret branch detected
    #[test]
    fn test_secret_branch_detected() {
        let mut v = CTVerifier::new();
        v.mark_secret("secret_key");

        let cond = make_binary(make_ident_expr("secret_key"), BinOp::Gt, make_int_expr(0));
        let stmt = make_if_stmt(cond);
        v.check_stmt(&stmt);

        assert_eq!(v.violations().len(), 1);
        assert!(matches!(&v.violations()[0], CTViolation::SecretBranch { variable, .. } if variable == "secret_key"));
    }

    // 2. Secret index detected
    #[test]
    fn test_secret_index_detected() {
        let mut v = CTVerifier::new();
        v.mark_secret("secret_idx");

        let idx_expr = make_index_expr("table", "secret_idx");
        let stmt = Stmt {
            kind: StmtKind::Expr(idx_expr),
            span: dummy_span(),
        };
        v.check_stmt(&stmt);

        assert_eq!(v.violations().len(), 1);
        assert!(matches!(&v.violations()[0], CTViolation::SecretIndex { variable, .. } if variable == "secret_idx"));
    }

    // 3. Secret loop bound detected
    #[test]
    fn test_secret_loop_bound_detected() {
        let mut v = CTVerifier::new();
        v.mark_secret("secret_len");

        let stmt = make_for_stmt("i", make_ident_expr("secret_len"));
        v.check_stmt(&stmt);

        assert_eq!(v.violations().len(), 1);
        assert!(matches!(&v.violations()[0], CTViolation::SecretLoopBound { variable, .. } if variable == "secret_len"));
    }

    // 4. Secret division detected
    #[test]
    fn test_secret_division_detected() {
        let mut v = CTVerifier::new();
        v.mark_secret("secret_b");

        let div = make_binary(make_ident_expr("a"), BinOp::Div, make_ident_expr("secret_b"));
        let stmt = Stmt {
            kind: StmtKind::Expr(div),
            span: dummy_span(),
        };
        v.check_stmt(&stmt);

        assert_eq!(v.violations().len(), 1);
        assert!(matches!(&v.violations()[0], CTViolation::SecretDivision { variable, .. } if variable == "secret_b"));
    }

    // 5. Public branch OK
    #[test]
    fn test_public_branch_ok() {
        let mut v = CTVerifier::new();
        v.mark_public("public_flag");

        let cond = make_ident_expr("public_flag");
        let stmt = make_if_stmt(cond);
        v.check_stmt(&stmt);

        assert!(v.violations().is_empty());
    }

    // 6. Taint propagation
    #[test]
    fn test_taint_propagation() {
        let mut v = CTVerifier::new();
        v.mark_secret("secret");
        v.mark_public("public_val");

        // let derived = secret + public_val
        let add = make_binary(make_ident_expr("secret"), BinOp::Add, make_ident_expr("public_val"));
        let let_stmt = Stmt {
            kind: StmtKind::Let {
                name: make_ident("derived"),
                ty: None,
                value: add,
            },
            span: dummy_span(),
        };
        v.check_stmt(&let_stmt);

        // derived should be secret now
        assert_eq!(v.classify_expr(&make_ident_expr("derived")), Secrecy::Secret);

        // Using derived in a branch should trigger violation
        let cond = make_ident_expr("derived");
        let if_stmt = make_if_stmt(cond);
        v.check_stmt(&if_stmt);

        assert_eq!(v.violations().len(), 1);
        assert!(matches!(&v.violations()[0], CTViolation::SecretBranch { variable, .. } if variable == "derived"));
    }

    // 7. ct_select
    #[test]
    fn test_ct_select() {
        assert_eq!(CTOps::ct_select(true, 42, 99), 42);
        assert_eq!(CTOps::ct_select(false, 42, 99), 99);
        assert_eq!(CTOps::ct_select(true, 0, u64::MAX), 0);
        assert_eq!(CTOps::ct_select(false, 0, u64::MAX), u64::MAX);
    }

    // 8. ct_eq
    #[test]
    fn test_ct_eq() {
        assert!(CTOps::ct_eq(0, 0));
        assert!(CTOps::ct_eq(42, 42));
        assert!(CTOps::ct_eq(u64::MAX, u64::MAX));
        assert!(!CTOps::ct_eq(0, 1));
        assert!(!CTOps::ct_eq(42, 43));
    }

    // 9. ct_lookup
    #[test]
    fn test_ct_lookup() {
        let table = vec![10, 20, 30, 40, 50];
        assert_eq!(CTOps::ct_lookup(&table, 0), 10);
        assert_eq!(CTOps::ct_lookup(&table, 2), 30);
        assert_eq!(CTOps::ct_lookup(&table, 4), 50);
    }

    // 10. ct_memcmp
    #[test]
    fn test_ct_memcmp() {
        assert!(CTOps::ct_memcmp(b"hello", b"hello"));
        assert!(!CTOps::ct_memcmp(b"hello", b"world"));
        assert!(!CTOps::ct_memcmp(b"hello", b"hell"));
        assert!(CTOps::ct_memcmp(b"", b""));
    }

    // 11. ct_swap
    #[test]
    fn test_ct_swap() {
        let mut a = 10u64;
        let mut b = 20u64;
        CTOps::ct_swap(true, &mut a, &mut b);
        assert_eq!(a, 20);
        assert_eq!(b, 10);

        let mut a = 10u64;
        let mut b = 20u64;
        CTOps::ct_swap(false, &mut a, &mut b);
        assert_eq!(a, 10);
        assert_eq!(b, 20);
    }

    // 12. Declassified values don't trigger violations
    #[test]
    fn test_declassified_no_violation() {
        let mut v = CTVerifier::new();
        v.mark_declassified("safe_val");

        let cond = make_ident_expr("safe_val");
        let stmt = make_if_stmt(cond);
        v.check_stmt(&stmt);

        assert!(v.violations().is_empty());
    }
}
