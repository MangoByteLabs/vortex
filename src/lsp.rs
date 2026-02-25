//! Vortex diagnostic analyzer and symbol table.
//!
//! Provides LSP-like functionality via CLI subcommands:
//! - `vortex check <file>` — diagnostics (pretty, JSON, or GCC format)
//! - `vortex symbols <file>` — list all symbols with locations
//! - `vortex hover <file> <line> <col>` — type/info at position
//! - `vortex definition <file> <name>` — find definition of symbol

use crate::ast::*;
use crate::lexer::{self, Span, TokenKind};
use crate::parser;
use std::collections::HashMap;

// ─── Diagnostic types ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
    Hint,
}

impl std::fmt::Display for Severity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Severity::Error => write!(f, "error"),
            Severity::Warning => write!(f, "warning"),
            Severity::Info => write!(f, "info"),
            Severity::Hint => write!(f, "hint"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub severity: Severity,
    pub message: String,
    pub code: String,
}

// ─── Symbol types ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SymbolKind {
    Function,
    Struct,
    Enum,
    Trait,
    Impl,
    Variable,
    Constant,
    Kernel,
    Module,
}

impl std::fmt::Display for SymbolKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolKind::Function => write!(f, "function"),
            SymbolKind::Struct => write!(f, "struct"),
            SymbolKind::Enum => write!(f, "enum"),
            SymbolKind::Trait => write!(f, "trait"),
            SymbolKind::Impl => write!(f, "impl"),
            SymbolKind::Variable => write!(f, "variable"),
            SymbolKind::Constant => write!(f, "constant"),
            SymbolKind::Kernel => write!(f, "kernel"),
            SymbolKind::Module => write!(f, "module"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct SymbolDef {
    pub name: String,
    pub kind: SymbolKind,
    pub file: String,
    pub line: usize,
    pub col: usize,
    pub detail: Option<String>,
}

// ─── Line/column mapping ────────────────────────────────────────────────────

/// Compute byte-offset → (1-based line, 1-based col).
fn offset_to_line_col(source: &str, offset: usize) -> (usize, usize) {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in source.char_indices() {
        if i >= offset {
            break;
        }
        if ch == '\n' {
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    (line, col)
}

/// Given (1-based line, 1-based col), return byte offset into source.
fn line_col_to_offset(source: &str, target_line: usize, target_col: usize) -> Option<usize> {
    let mut line = 1usize;
    let mut col = 1usize;
    for (i, ch) in source.char_indices() {
        if line == target_line && col == target_col {
            return Some(i);
        }
        if ch == '\n' {
            if line == target_line {
                return None; // col past end of line
            }
            line += 1;
            col = 1;
        } else {
            col += 1;
        }
    }
    if line == target_line && col == target_col {
        return Some(source.len());
    }
    None
}

// ─── VortexAnalyzer ─────────────────────────────────────────────────────────

pub struct VortexAnalyzer {
    pub diagnostics: Vec<Diagnostic>,
}

impl VortexAnalyzer {
    pub fn new() -> Self {
        Self {
            diagnostics: Vec::new(),
        }
    }

    /// Run all analysis passes on a file.
    pub fn analyze(&mut self, source: &str, filename: &str) {
        self.check_lexer(source, filename);
        self.check_parser(source, filename);
        self.check_types(source, filename);

        // For passes that need parsed items, re-parse (cheap).
        let tokens = lexer::lex(source);
        if let Ok(program) = parser::parse(tokens, 0) {
            self.check_unused(&program.items, source, filename);
            self.check_style(&program.items, source, filename);
            self.check_reachability(&program.items, source, filename);
        }
    }

    // ── Pass 1: Lex errors ──────────────────────────────────────────────

    fn check_lexer(&mut self, source: &str, filename: &str) {
        let mut lex = TokenKind::lexer(source);
        use logos::Logos;
        while let Some(result) = lex.next() {
            if result.is_err() {
                let span = lex.span();
                let (line, col) = offset_to_line_col(source, span.start);
                let (end_line, end_col) = offset_to_line_col(source, span.end);
                self.diagnostics.push(Diagnostic {
                    file: filename.to_string(),
                    line,
                    col,
                    end_line,
                    end_col,
                    severity: Severity::Error,
                    message: format!("unexpected character `{}`", &source[span.start..span.end]),
                    code: "E001".to_string(),
                });
            }
        }
    }

    // ── Pass 2: Parse errors ────────────────────────────────────────────

    fn check_parser(&mut self, source: &str, filename: &str) {
        let tokens = lexer::lex(source);
        if let Err(diags) = parser::parse(tokens, 0) {
            for diag in &diags {
                // Extract span from the first label
                let (line, col, end_line, end_col) = if let Some(label) = diag.labels.first() {
                    let start = label.range.start;
                    let end = label.range.end;
                    let (l, c) = offset_to_line_col(source, start);
                    let (el, ec) = offset_to_line_col(source, end);
                    (l, c, el, ec)
                } else {
                    (1, 1, 1, 1)
                };
                self.diagnostics.push(Diagnostic {
                    file: filename.to_string(),
                    line,
                    col,
                    end_line,
                    end_col,
                    severity: Severity::Error,
                    message: diag.message.clone(),
                    code: "E002".to_string(),
                });
            }
        }
    }

    // ── Pass 3: Type errors ─────────────────────────────────────────────

    fn check_types(&mut self, source: &str, filename: &str) {
        let tokens = lexer::lex(source);
        let program = match parser::parse(tokens, 0) {
            Ok(p) => p,
            Err(_) => return, // parse errors already reported
        };
        if let Err(diags) = crate::typeck::check(&program, 0) {
            for diag in &diags {
                let sev = match diag.severity {
                    codespan_reporting::diagnostic::Severity::Error => Severity::Error,
                    codespan_reporting::diagnostic::Severity::Warning => Severity::Warning,
                    codespan_reporting::diagnostic::Severity::Note => Severity::Info,
                    codespan_reporting::diagnostic::Severity::Help => Severity::Hint,
                    _ => Severity::Error,
                };
                let (line, col, end_line, end_col) = if let Some(label) = diag.labels.first() {
                    let start = label.range.start;
                    let end = label.range.end;
                    let (l, c) = offset_to_line_col(source, start);
                    let (el, ec) = offset_to_line_col(source, end);
                    (l, c, el, ec)
                } else {
                    (1, 1, 1, 1)
                };
                self.diagnostics.push(Diagnostic {
                    file: filename.to_string(),
                    line,
                    col,
                    end_line,
                    end_col,
                    severity: sev,
                    message: diag.message.clone(),
                    code: "E003".to_string(),
                });
            }
        }
    }

    // ── Pass 4: Unused variables ────────────────────────────────────────

    fn check_unused(&mut self, items: &[Item], source: &str, filename: &str) {
        for item in items {
            if let ItemKind::Function(func) = &item.kind {
                self.check_unused_in_function(func, source, filename);
            }
            if let ItemKind::Kernel(kernel) = &item.kind {
                // Treat kernel like a function for unused-var analysis
                let fake_func = Function {
                    name: kernel.name.clone(),
                    generics: kernel.generics.clone(),
                    params: kernel.params.clone(),
                    ret_type: kernel.ret_type.clone(),
                    where_clause: kernel.where_clause.clone(),
                    body: kernel.body.clone(),
                    annotations: kernel.annotations.clone(),
                };
                self.check_unused_in_function(&fake_func, source, filename);
            }
        }
    }

    fn check_unused_in_function(&mut self, func: &Function, source: &str, filename: &str) {
        // Collect all let/var bindings
        let mut bindings: Vec<(String, Span)> = Vec::new();
        collect_bindings_block(&func.body, &mut bindings);

        // Collect all identifier references
        let mut refs: Vec<String> = Vec::new();
        collect_refs_block(&func.body, &mut refs);

        for (name, span) in &bindings {
            if name.starts_with('_') {
                continue;
            }
            if !refs.iter().any(|r| r == name) {
                let (line, col) = offset_to_line_col(source, span.start);
                let (end_line, end_col) = offset_to_line_col(source, span.end);
                self.diagnostics.push(Diagnostic {
                    file: filename.to_string(),
                    line,
                    col,
                    end_line,
                    end_col,
                    severity: Severity::Warning,
                    message: format!("unused variable `{}`", name),
                    code: "W001".to_string(),
                });
            }
        }
    }

    // ── Pass 5: Style warnings ──────────────────────────────────────────

    fn check_style(&mut self, items: &[Item], source: &str, filename: &str) {
        for item in items {
            match &item.kind {
                ItemKind::Function(func) => {
                    if !is_snake_case(&func.name.name) && func.name.name != "main" {
                        let (line, col) = offset_to_line_col(source, func.name.span.start);
                        let (end_line, end_col) = offset_to_line_col(source, func.name.span.end);
                        self.diagnostics.push(Diagnostic {
                            file: filename.to_string(),
                            line,
                            col,
                            end_line,
                            end_col,
                            severity: Severity::Hint,
                            message: format!(
                                "function `{}` should use snake_case naming",
                                func.name.name
                            ),
                            code: "S001".to_string(),
                        });
                    }
                }
                ItemKind::Struct(s) => {
                    if !is_pascal_case(&s.name.name) {
                        let (line, col) = offset_to_line_col(source, s.name.span.start);
                        let (end_line, end_col) = offset_to_line_col(source, s.name.span.end);
                        self.diagnostics.push(Diagnostic {
                            file: filename.to_string(),
                            line,
                            col,
                            end_line,
                            end_col,
                            severity: Severity::Hint,
                            message: format!(
                                "struct `{}` should use PascalCase naming",
                                s.name.name
                            ),
                            code: "S002".to_string(),
                        });
                    }
                }
                _ => {}
            }
        }
    }

    // ── Pass 6: Unreachable code ────────────────────────────────────────

    fn check_reachability(&mut self, items: &[Item], source: &str, filename: &str) {
        for item in items {
            match &item.kind {
                ItemKind::Function(func) => {
                    self.check_block_reachability(&func.body, source, filename);
                }
                ItemKind::Kernel(kernel) => {
                    self.check_block_reachability(&kernel.body, source, filename);
                }
                _ => {}
            }
        }
    }

    fn check_block_reachability(&mut self, block: &Block, source: &str, filename: &str) {
        let mut found_terminator = false;
        for (i, stmt) in block.stmts.iter().enumerate() {
            if found_terminator {
                let (line, col) = offset_to_line_col(source, stmt.span.start);
                let (end_line, end_col) = offset_to_line_col(source, stmt.span.end);
                self.diagnostics.push(Diagnostic {
                    file: filename.to_string(),
                    line,
                    col,
                    end_line,
                    end_col,
                    severity: Severity::Warning,
                    message: "unreachable code".to_string(),
                    code: "W002".to_string(),
                });
                break; // only report once per block
            }
            match &stmt.kind {
                StmtKind::Return(_) | StmtKind::Break | StmtKind::Continue => {
                    if i + 1 < block.stmts.len() || block.expr.is_some() {
                        found_terminator = true;
                    }
                }
                StmtKind::For { body, .. }
                | StmtKind::While { body, .. }
                | StmtKind::Loop { body } => {
                    self.check_block_reachability(body, source, filename);
                }
                _ => {}
            }
        }
    }

    // ── Output formatters ───────────────────────────────────────────────

    pub fn to_json(&self) -> String {
        let mut out = String::from("[");
        for (i, d) in self.diagnostics.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                concat!(
                    "{{",
                    "\"file\":\"{}\",",
                    "\"line\":{},",
                    "\"col\":{},",
                    "\"end_line\":{},",
                    "\"end_col\":{},",
                    "\"severity\":\"{}\",",
                    "\"message\":\"{}\",",
                    "\"code\":\"{}\"",
                    "}}"
                ),
                escape_json(&d.file),
                d.line,
                d.col,
                d.end_line,
                d.end_col,
                d.severity,
                escape_json(&d.message),
                d.code,
            ));
        }
        out.push(']');
        out
    }

    pub fn to_pretty(&self) -> String {
        let mut out = String::new();
        for d in &self.diagnostics {
            out.push_str(&format!(
                "{} [{}]: {}:{}:{}: {}\n",
                d.severity, d.code, d.file, d.line, d.col, d.message
            ));
        }
        if self.diagnostics.is_empty() {
            out.push_str("No diagnostics.\n");
        }
        out
    }

    pub fn to_gcc_format(&self) -> String {
        let mut out = String::new();
        for d in &self.diagnostics {
            out.push_str(&format!(
                "{}:{}:{}: {}: {}\n",
                d.file, d.line, d.col, d.severity, d.message
            ));
        }
        out
    }
}

// ─── SymbolTable ────────────────────────────────────────────────────────────

pub struct SymbolTable {
    definitions: HashMap<String, Vec<SymbolDef>>,
}

impl SymbolTable {
    pub fn from_program(items: &[Item], source: &str, filename: &str) -> Self {
        let mut defs: HashMap<String, Vec<SymbolDef>> = HashMap::new();

        for item in items {
            match &item.kind {
                ItemKind::Function(func) => {
                    let (line, col) = offset_to_line_col(source, func.name.span.start);
                    let detail = func.ret_type.as_ref().map(|t| format!("{}", t));
                    let sym = SymbolDef {
                        name: func.name.name.clone(),
                        kind: SymbolKind::Function,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(func.name.name.clone()).or_default().push(sym);

                    // Also collect params and local bindings as variable symbols
                    for param in &func.params {
                        let (pl, pc) = offset_to_line_col(source, param.name.span.start);
                        let detail = Some(format!("{}", param.ty));
                        let sym = SymbolDef {
                            name: param.name.name.clone(),
                            kind: SymbolKind::Variable,
                            file: filename.to_string(),
                            line: pl,
                            col: pc,
                            detail,
                        };
                        defs.entry(param.name.name.clone()).or_default().push(sym);
                    }
                }
                ItemKind::Kernel(kernel) => {
                    let (line, col) = offset_to_line_col(source, kernel.name.span.start);
                    let detail = kernel.ret_type.as_ref().map(|t| format!("{}", t));
                    let sym = SymbolDef {
                        name: kernel.name.name.clone(),
                        kind: SymbolKind::Kernel,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(kernel.name.name.clone()).or_default().push(sym);
                }
                ItemKind::Struct(s) => {
                    let (line, col) = offset_to_line_col(source, s.name.span.start);
                    let detail = Some(format!("{} fields", s.fields.len()));
                    let sym = SymbolDef {
                        name: s.name.name.clone(),
                        kind: SymbolKind::Struct,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(s.name.name.clone()).or_default().push(sym);
                }
                ItemKind::Enum(e) => {
                    let (line, col) = offset_to_line_col(source, e.name.span.start);
                    let detail = Some(format!("{} variants", e.variants.len()));
                    let sym = SymbolDef {
                        name: e.name.name.clone(),
                        kind: SymbolKind::Enum,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(e.name.name.clone()).or_default().push(sym);
                }
                ItemKind::Trait(t) => {
                    let (line, col) = offset_to_line_col(source, t.name.span.start);
                    let detail = Some(format!("{} methods", t.methods.len()));
                    let sym = SymbolDef {
                        name: t.name.name.clone(),
                        kind: SymbolKind::Trait,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(t.name.name.clone()).or_default().push(sym);
                }
                ItemKind::Impl(imp) => {
                    let name = format!("{}", imp.target);
                    let (line, col) = offset_to_line_col(source, item.span.start);
                    let detail = imp.trait_name.as_ref().map(|t| format!("impl {} for", t));
                    let sym = SymbolDef {
                        name: name.clone(),
                        kind: SymbolKind::Impl,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(name).or_default().push(sym);
                }
                ItemKind::Const(c) => {
                    let (line, col) = offset_to_line_col(source, c.name.span.start);
                    let detail = c.ty.as_ref().map(|t| format!("{}", t));
                    let sym = SymbolDef {
                        name: c.name.name.clone(),
                        kind: SymbolKind::Constant,
                        file: filename.to_string(),
                        line,
                        col,
                        detail,
                    };
                    defs.entry(c.name.name.clone()).or_default().push(sym);
                }
                ItemKind::TypeAlias(_) | ItemKind::Import(_) | ItemKind::FieldDef(_) | ItemKind::Static(_) => {}
            }
        }

        Self { definitions: defs }
    }

    pub fn find_definition(&self, name: &str) -> Option<&SymbolDef> {
        self.definitions.get(name).and_then(|v| v.first())
    }

    pub fn find_references(&self, name: &str) -> Vec<(String, usize, usize)> {
        self.definitions
            .get(name)
            .map(|v| v.iter().map(|s| (s.file.clone(), s.line, s.col)).collect())
            .unwrap_or_default()
    }

    pub fn all_symbols(&self) -> Vec<&SymbolDef> {
        let mut syms: Vec<&SymbolDef> = self.definitions.values().flat_map(|v| v.iter()).collect();
        syms.sort_by_key(|s| (s.line, s.col));
        syms
    }

    pub fn hover_info(&self, name: &str) -> Option<String> {
        self.find_definition(name).map(|sym| {
            let detail = sym.detail.as_deref().unwrap_or("");
            format!("{} {} {}", sym.kind, sym.name, detail)
        })
    }

    /// Find the symbol at a given (1-based) line and column.
    pub fn symbol_at(&self, source: &str, target_line: usize, target_col: usize) -> Option<&SymbolDef> {
        let offset = line_col_to_offset(source, target_line, target_col)?;
        // Find token at this offset
        let tokens = lexer::lex(source);
        let tok = tokens.iter().find(|t| t.span.start <= offset && offset < t.span.end)?;
        if tok.kind != TokenKind::Ident {
            return None;
        }
        self.find_definition(&tok.text)
    }

    pub fn to_json(&self) -> String {
        let syms = self.all_symbols();
        let mut out = String::from("[");
        for (i, s) in syms.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str(&format!(
                "{{\"name\":\"{}\",\"kind\":\"{}\",\"file\":\"{}\",\"line\":{},\"col\":{}{}}}",
                escape_json(&s.name),
                s.kind,
                escape_json(&s.file),
                s.line,
                s.col,
                s.detail
                    .as_ref()
                    .map(|d| format!(",\"detail\":\"{}\"", escape_json(d)))
                    .unwrap_or_default(),
            ));
        }
        out.push(']');
        out
    }
}

// ─── Helpers ────────────────────────────────────────────────────────────────

fn escape_json(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
        .replace('\t', "\\t")
}

fn is_snake_case(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    s.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '_')
}

fn is_pascal_case(s: &str) -> bool {
    if s.is_empty() {
        return true;
    }
    s.chars().next().unwrap().is_ascii_uppercase()
}

/// Collect all let/var binding names from a block.
fn collect_bindings_block(block: &Block, out: &mut Vec<(String, Span)>) {
    for stmt in &block.stmts {
        match &stmt.kind {
            StmtKind::Let { name, .. } | StmtKind::Var { name, .. } => {
                out.push((name.name.clone(), name.span));
            }
            StmtKind::For { var, body, .. } => {
                out.push((var.name.clone(), var.span));
                collect_bindings_block(body, out);
            }
            StmtKind::While { body, .. } | StmtKind::Loop { body } => {
                collect_bindings_block(body, out);
            }
            _ => {}
        }
    }
}

/// Collect all identifier references from a block (not definitions).
fn collect_refs_block(block: &Block, out: &mut Vec<String>) {
    for stmt in &block.stmts {
        collect_refs_stmt(stmt, out);
    }
    if let Some(expr) = &block.expr {
        collect_refs_expr(expr, out);
    }
}

fn collect_refs_stmt(stmt: &Stmt, out: &mut Vec<String>) {
    match &stmt.kind {
        StmtKind::Let { value, .. } | StmtKind::Var { value, .. } => {
            collect_refs_expr(value, out);
        }
        StmtKind::Return(Some(expr)) | StmtKind::Expr(expr) => {
            collect_refs_expr(expr, out);
        }
        StmtKind::Assign { target, value, .. } => {
            collect_refs_expr(target, out);
            collect_refs_expr(value, out);
        }
        StmtKind::For { iter, body, .. } => {
            collect_refs_expr(iter, out);
            collect_refs_block(body, out);
        }
        StmtKind::While { cond, body } => {
            collect_refs_expr(cond, out);
            collect_refs_block(body, out);
        }
        StmtKind::Loop { body } => {
            collect_refs_block(body, out);
        }
        StmtKind::Dispatch { index, args, .. } => {
            collect_refs_expr(index, out);
            for arg in args {
                collect_refs_expr(arg, out);
            }
        }
        _ => {}
    }
}

fn collect_refs_expr(expr: &Expr, out: &mut Vec<String>) {
    match &expr.kind {
        ExprKind::Ident(id) => {
            out.push(id.name.clone());
        }
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::MatMul { lhs, rhs } => {
            collect_refs_expr(lhs, out);
            collect_refs_expr(rhs, out);
        }
        ExprKind::Unary { expr: inner, .. } | ExprKind::Try(inner) | ExprKind::Cast { expr: inner, .. } => {
            collect_refs_expr(inner, out);
        }
        ExprKind::Call { func, args } => {
            collect_refs_expr(func, out);
            for arg in args {
                collect_refs_expr(arg, out);
            }
        }
        ExprKind::FieldAccess { base, .. } => {
            collect_refs_expr(base, out);
        }
        ExprKind::Index { base, indices } => {
            collect_refs_expr(base, out);
            for idx in indices {
                collect_refs_expr(idx, out);
            }
        }
        ExprKind::Block(block) => {
            collect_refs_block(block, out);
        }
        ExprKind::If { cond, then_block, else_block } => {
            collect_refs_expr(cond, out);
            collect_refs_block(then_block, out);
            if let Some(eb) = else_block {
                collect_refs_block(eb, out);
            }
        }
        ExprKind::Range { start, end } => {
            collect_refs_expr(start, out);
            collect_refs_expr(end, out);
        }
        ExprKind::ArrayLiteral(elems) => {
            for e in elems {
                collect_refs_expr(e, out);
            }
        }
        ExprKind::StructLiteral { fields, .. } => {
            for (_, val) in fields {
                collect_refs_expr(val, out);
            }
        }
        ExprKind::Match { expr: me, arms } => {
            collect_refs_expr(me, out);
            for arm in arms {
                collect_refs_expr(&arm.body, out);
            }
        }
        ExprKind::Closure { body, .. } => {
            collect_refs_expr(body, out);
        }
        _ => {}
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_error_diagnostic() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze("fn {}", "test.vx");
        let errors: Vec<_> = analyzer
            .diagnostics
            .iter()
            .filter(|d| d.severity == Severity::Error)
            .collect();
        assert!(!errors.is_empty(), "expected parse error diagnostic");
        assert!(errors[0].line >= 1);
        assert!(errors[0].col >= 1);
    }

    #[test]
    fn test_type_error_diagnostic() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze(
            "fn test() -> f32 {\n    return true\n}",
            "test.vx",
        );
        let type_errors: Vec<_> = analyzer
            .diagnostics
            .iter()
            .filter(|d| d.code == "E003")
            .collect();
        assert!(!type_errors.is_empty(), "expected type error diagnostic");
    }

    #[test]
    fn test_undefined_variable_diagnostic() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze(
            "fn test() -> f32 {\n    return undefined_var\n}",
            "test.vx",
        );
        let errors: Vec<_> = analyzer
            .diagnostics
            .iter()
            .filter(|d| d.message.contains("undefined"))
            .collect();
        assert!(!errors.is_empty(), "expected undefined variable diagnostic");
    }

    #[test]
    fn test_unused_variable_warning() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze(
            "fn test() {\n    let unused_x = 42\n}",
            "test.vx",
        );
        let warnings: Vec<_> = analyzer
            .diagnostics
            .iter()
            .filter(|d| d.code == "W001" && d.message.contains("unused_x"))
            .collect();
        assert!(
            !warnings.is_empty(),
            "expected unused variable warning, got: {:?}",
            analyzer.diagnostics
        );
    }

    #[test]
    fn test_symbol_table_finds_function() {
        let source = "fn add(a: f32, b: f32) -> f32 { return a + b }";
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).unwrap();
        let table = SymbolTable::from_program(&program.items, source, "test.vx");
        let sym = table.find_definition("add");
        assert!(sym.is_some(), "expected to find function 'add'");
        let sym = sym.unwrap();
        assert_eq!(sym.kind, SymbolKind::Function);
        assert_eq!(sym.line, 1);
    }

    #[test]
    fn test_symbol_table_finds_struct() {
        let source = "struct Point {\n    x: f32,\n    y: f32,\n}";
        let tokens = lexer::lex(source);
        let program = parser::parse(tokens, 0).unwrap();
        let table = SymbolTable::from_program(&program.items, source, "test.vx");
        let sym = table.find_definition("Point");
        assert!(sym.is_some(), "expected to find struct 'Point'");
        let sym = sym.unwrap();
        assert_eq!(sym.kind, SymbolKind::Struct);
    }

    #[test]
    fn test_json_output_structure() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze("fn main() { let x = 1 }", "test.vx");
        let json = analyzer.to_json();
        assert!(json.starts_with('['), "JSON should start with [");
        assert!(json.ends_with(']'), "JSON should end with ]");
    }

    #[test]
    fn test_gcc_format_matches_pattern() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze("fn test() -> f32 { return true }", "test.vx");
        let gcc = analyzer.to_gcc_format();
        // GCC format: file:line:col: severity: message
        for line in gcc.lines() {
            let parts: Vec<&str> = line.splitn(5, ':').collect();
            assert!(
                parts.len() >= 5,
                "GCC format line should have at least 5 colon-separated parts: {}",
                line
            );
        }
    }

    #[test]
    fn test_clean_file_zero_diagnostics() {
        let mut analyzer = VortexAnalyzer::new();
        analyzer.analyze(
            "fn add(a: f32, b: f32) -> f32 {\n    return a + b\n}",
            "test.vx",
        );
        assert!(
            analyzer.diagnostics.is_empty(),
            "expected zero diagnostics for clean file, got: {:?}",
            analyzer.diagnostics
        );
    }

    #[test]
    fn test_multiple_errors_all_reported() {
        let mut analyzer = VortexAnalyzer::new();
        // Two functions with type errors
        analyzer.analyze(
            "fn a() -> f32 { return true }\nfn b() -> i64 { return false }",
            "test.vx",
        );
        let type_errors: Vec<_> = analyzer
            .diagnostics
            .iter()
            .filter(|d| d.code == "E003")
            .collect();
        assert!(
            type_errors.len() >= 2,
            "expected at least 2 type errors, got {}",
            type_errors.len()
        );
    }
}
