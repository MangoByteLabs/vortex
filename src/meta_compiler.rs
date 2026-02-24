// Pillar 7: Self-Improving Compiler for the Vortex language.
//
// Provides a meta-compiler that can learn optimization rules, apply rewrite
// passes, perform genetic optimization of expressions, and self-improve by
// measuring which rules actually help. Operates on a simple expression language
// (not the full Vortex AST) so that users can experiment with compiler passes
// from within Vortex scripts.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ─── Helpers ────────────────────────────────────────────────────────────

fn value_to_f64(v: &Value) -> f64 {
    match v {
        Value::Float(n) => *n,
        Value::Int(n) => *n as f64,
        Value::String(s) => s.parse::<f64>().unwrap_or(0.0),
        _ => 0.0,
    }
}

fn value_to_usize(v: &Value) -> usize {
    match v {
        Value::Int(n) => *n as usize,
        Value::Float(n) => *n as usize,
        Value::String(s) => s.parse::<usize>().unwrap_or(0),
        _ => 0,
    }
}

fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Float(n) => format!("{}", n),
        Value::Int(n) => format!("{}", n),
        Value::Array(arr) => {
            let parts: Vec<String> = arr.iter().map(|x| value_to_string(x)).collect();
            format!("[{}]", parts.join(", "))
        }
        Value::HashMap(m) => {
            let parts: Vec<String> = m.iter().map(|(k, v)| format!("{}: {}", k, value_to_string(v))).collect();
            format!("{{{}}}", parts.join(", "))
        }
        Value::Bool(b) => format!("{}", b),
        Value::Void => "void".to_string(),
        _ => format!("{:?}", v),
    }
}

// ─── Deterministic xorshift PRNG ────────────────────────────────────────

struct Xorshift {
    state: u64,
}

impl Xorshift {
    fn new(seed: u64) -> Self {
        Xorshift { state: if seed == 0 { 1 } else { seed } }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        if bound == 0 { return 0; }
        (self.next_u64() % bound as u64) as usize
    }

    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

// ─── AST Pattern & Template ─────────────────────────────────────────────

#[derive(Debug, Clone)]
enum AstPattern {
    Wildcard(String),
    Literal(f64),
    FnCall { name: String, args: Vec<AstPattern> },
    BinOp { op: String, lhs: Box<AstPattern>, rhs: Box<AstPattern> },
    Sequence(Vec<AstPattern>),
    Var(String),
}

#[derive(Debug, Clone)]
enum AstTemplate {
    Literal(f64),
    FnCall { name: String, args: Vec<AstTemplate> },
    BinOp { op: String, lhs: Box<AstTemplate>, rhs: Box<AstTemplate> },
    Ref(String),
    Sequence(Vec<AstTemplate>),
}

// ─── SimpleExpr ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum SimpleExpr {
    Num(f64),
    Var(String),
    Call { name: String, args: Vec<SimpleExpr> },
    BinOp { op: String, lhs: Box<SimpleExpr>, rhs: Box<SimpleExpr> },
    Block(Vec<SimpleExpr>),
}

impl SimpleExpr {
    fn to_string_repr(&self) -> String {
        match self {
            SimpleExpr::Num(n) => {
                if *n == (*n as i64) as f64 {
                    format!("{}", *n as i64)
                } else {
                    format!("{}", n)
                }
            }
            SimpleExpr::Var(v) => v.clone(),
            SimpleExpr::Call { name, args } => {
                let arg_strs: Vec<String> = args.iter().map(|a| a.to_string_repr()).collect();
                format!("{}({})", name, arg_strs.join(", "))
            }
            SimpleExpr::BinOp { op, lhs, rhs } => {
                format!("({} {} {})", lhs.to_string_repr(), op, rhs.to_string_repr())
            }
            SimpleExpr::Block(exprs) => {
                let strs: Vec<String> = exprs.iter().map(|e| e.to_string_repr()).collect();
                format!("{{ {} }}", strs.join("; "))
            }
        }
    }

    fn complexity(&self) -> (u64, u64, u64) {
        // Returns (flops, memory_ops, depth)
        match self {
            SimpleExpr::Num(_) => (0, 1, 1),
            SimpleExpr::Var(_) => (0, 1, 1),
            SimpleExpr::Call { args, .. } => {
                let mut flops = 1u64;
                let mut mem = 1u64;
                let mut max_d = 0u64;
                for a in args {
                    let (f, m, d) = a.complexity();
                    flops += f;
                    mem += m;
                    if d > max_d { max_d = d; }
                }
                (flops, mem, max_d + 1)
            }
            SimpleExpr::BinOp { lhs, rhs, .. } => {
                let (lf, lm, ld) = lhs.complexity();
                let (rf, rm, rd) = rhs.complexity();
                (lf + rf + 1, lm + rm, std::cmp::max(ld, rd) + 1)
            }
            SimpleExpr::Block(exprs) => {
                let mut flops = 0u64;
                let mut mem = 0u64;
                let mut max_d = 0u64;
                for e in exprs {
                    let (f, m, d) = e.complexity();
                    flops += f;
                    mem += m;
                    if d > max_d { max_d = d; }
                }
                (flops, mem, max_d)
            }
        }
    }

    fn evaluate(&self, vars: &HashMap<String, f64>) -> Option<f64> {
        match self {
            SimpleExpr::Num(n) => Some(*n),
            SimpleExpr::Var(v) => vars.get(v).copied(),
            SimpleExpr::BinOp { op, lhs, rhs } => {
                let l = lhs.evaluate(vars)?;
                let r = rhs.evaluate(vars)?;
                match op.as_str() {
                    "+" => Some(l + r),
                    "-" => Some(l - r),
                    "*" => Some(l * r),
                    "/" => if r != 0.0 { Some(l / r) } else { None },
                    "%" => if r != 0.0 { Some(l % r) } else { None },
                    _ => None,
                }
            }
            SimpleExpr::Call { name, args } => {
                let evaluated: Option<Vec<f64>> = args.iter().map(|a| a.evaluate(vars)).collect();
                let vals = evaluated?;
                match name.as_str() {
                    "sqrt" if vals.len() == 1 => Some(vals[0].sqrt()),
                    "abs" if vals.len() == 1 => Some(vals[0].abs()),
                    "sin" if vals.len() == 1 => Some(vals[0].sin()),
                    "cos" if vals.len() == 1 => Some(vals[0].cos()),
                    "exp" if vals.len() == 1 => Some(vals[0].exp()),
                    "log" if vals.len() == 1 => Some(vals[0].ln()),
                    "pow" if vals.len() == 2 => Some(vals[0].powf(vals[1])),
                    "min" if vals.len() == 2 => Some(vals[0].min(vals[1])),
                    "max" if vals.len() == 2 => Some(vals[0].max(vals[1])),
                    "add" if vals.len() == 2 => Some(vals[0] + vals[1]),
                    "sub" if vals.len() == 2 => Some(vals[0] - vals[1]),
                    "mul" if vals.len() == 2 => Some(vals[0] * vals[1]),
                    "div" if vals.len() == 2 && vals[1] != 0.0 => Some(vals[0] / vals[1]),
                    "neg" if vals.len() == 1 => Some(-vals[0]),
                    _ => None,
                }
            }
            SimpleExpr::Block(exprs) => {
                let mut last = None;
                for e in exprs {
                    last = e.evaluate(vars);
                }
                last
            }
        }
    }

    fn collect_vars(&self, out: &mut Vec<String>) {
        match self {
            SimpleExpr::Var(v) => {
                if !out.contains(v) { out.push(v.clone()); }
            }
            SimpleExpr::BinOp { lhs, rhs, .. } => {
                lhs.collect_vars(out);
                rhs.collect_vars(out);
            }
            SimpleExpr::Call { args, .. } => {
                for a in args { a.collect_vars(out); }
            }
            SimpleExpr::Block(exprs) => {
                for e in exprs { e.collect_vars(out); }
            }
            SimpleExpr::Num(_) => {}
        }
    }
}

// ─── Simple expression parser ───────────────────────────────────────────

struct ExprParser {
    tokens: Vec<String>,
    pos: usize,
}

impl ExprParser {
    fn new(input: &str) -> Self {
        ExprParser { tokens: tokenize_expr(input), pos: 0 }
    }

    fn peek(&self) -> Option<&str> {
        self.tokens.get(self.pos).map(|s| s.as_str())
    }

    fn advance(&mut self) -> Option<String> {
        if self.pos < self.tokens.len() {
            let t = self.tokens[self.pos].clone();
            self.pos += 1;
            Some(t)
        } else {
            None
        }
    }

    fn expect(&mut self, s: &str) -> bool {
        if self.peek() == Some(s) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn parse_expr(&mut self) -> SimpleExpr {
        self.parse_additive()
    }

    fn parse_additive(&mut self) -> SimpleExpr {
        let mut left = self.parse_multiplicative();
        loop {
            match self.peek() {
                Some("+") | Some("-") => {
                    let op = self.advance().unwrap();
                    let right = self.parse_multiplicative();
                    left = SimpleExpr::BinOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        left
    }

    fn parse_multiplicative(&mut self) -> SimpleExpr {
        let mut left = self.parse_primary();
        loop {
            match self.peek() {
                Some("*") | Some("/") | Some("%") => {
                    let op = self.advance().unwrap();
                    let right = self.parse_primary();
                    left = SimpleExpr::BinOp {
                        op,
                        lhs: Box::new(left),
                        rhs: Box::new(right),
                    };
                }
                _ => break,
            }
        }
        left
    }

    fn parse_primary(&mut self) -> SimpleExpr {
        if self.expect("(") {
            let e = self.parse_expr();
            self.expect(")");
            return e;
        }
        if self.expect("{") {
            let mut exprs = Vec::new();
            while self.peek() != Some("}") && self.peek().is_some() {
                exprs.push(self.parse_expr());
                self.expect(";");
            }
            self.expect("}");
            return SimpleExpr::Block(exprs);
        }
        let tok = self.advance().unwrap_or_default();
        // Try number
        if let Ok(n) = tok.parse::<f64>() {
            return SimpleExpr::Num(n);
        }
        // Identifier — could be function call
        if self.peek() == Some("(") {
            self.advance(); // consume '('
            let mut args = Vec::new();
            if self.peek() != Some(")") {
                args.push(self.parse_expr());
                while self.expect(",") {
                    args.push(self.parse_expr());
                }
            }
            self.expect(")");
            return SimpleExpr::Call { name: tok, args };
        }
        SimpleExpr::Var(tok)
    }
}

fn tokenize_expr(input: &str) -> Vec<String> {
    let mut tokens = Vec::new();
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            i += 1;
            continue;
        }
        if "+-*/%(),;{}".contains(c) {
            tokens.push(c.to_string());
            i += 1;
            continue;
        }
        // Number (including negatives handled by parser via binop)
        if c.is_ascii_digit() || (c == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()) {
            let start = i;
            while i < chars.len() && (chars[i].is_ascii_digit() || chars[i] == '.') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }
        // Identifier
        if c.is_alphabetic() || c == '_' {
            let start = i;
            while i < chars.len() && (chars[i].is_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            tokens.push(chars[start..i].iter().collect());
            continue;
        }
        i += 1;
    }
    tokens
}

fn parse_simple_expr_str(input: &str) -> SimpleExpr {
    let mut parser = ExprParser::new(input);
    parser.parse_expr()
}

// ─── Pattern parsing (functional form: add(x, 0)) ──────────────────────

fn parse_pattern(input: &str) -> AstPattern {
    let trimmed = input.trim();
    // Numeric literal?
    if let Ok(n) = trimmed.parse::<f64>() {
        return AstPattern::Literal(n);
    }
    // Wildcard: _name or just _
    if trimmed.starts_with('_') {
        return AstPattern::Wildcard(trimmed.to_string());
    }
    // Function call: name(args...)
    if let Some(paren) = trimmed.find('(') {
        if trimmed.ends_with(')') {
            let name = trimmed[..paren].trim().to_string();
            let inner = &trimmed[paren + 1..trimmed.len() - 1];
            let args = split_top_level(inner);
            let arg_patterns: Vec<AstPattern> = args.iter().map(|a| parse_pattern(a)).collect();
            return AstPattern::FnCall { name, args: arg_patterns };
        }
    }
    // Variable
    AstPattern::Var(trimmed.to_string())
}

fn parse_template(input: &str) -> AstTemplate {
    let trimmed = input.trim();
    if let Ok(n) = trimmed.parse::<f64>() {
        return AstTemplate::Literal(n);
    }
    if let Some(paren) = trimmed.find('(') {
        if trimmed.ends_with(')') {
            let name = trimmed[..paren].trim().to_string();
            let inner = &trimmed[paren + 1..trimmed.len() - 1];
            let args = split_top_level(inner);
            let arg_templates: Vec<AstTemplate> = args.iter().map(|a| parse_template(a)).collect();
            return AstTemplate::FnCall { name, args: arg_templates };
        }
    }
    AstTemplate::Ref(trimmed.to_string())
}

fn split_top_level(s: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut current = String::new();
    for c in s.chars() {
        match c {
            '(' => { depth += 1; current.push(c); }
            ')' => { depth -= 1; current.push(c); }
            ',' if depth == 0 => {
                let trimmed = current.trim().to_string();
                if !trimmed.is_empty() {
                    parts.push(trimmed);
                }
                current.clear();
            }
            _ => { current.push(c); }
        }
    }
    let trimmed = current.trim().to_string();
    if !trimmed.is_empty() {
        parts.push(trimmed);
    }
    parts
}

// ─── Pattern matching on SimpleExpr ─────────────────────────────────────

fn match_pattern(pattern: &AstPattern, expr: &SimpleExpr, bindings: &mut HashMap<String, SimpleExpr>) -> bool {
    match pattern {
        AstPattern::Wildcard(name) => {
            if let Some(existing) = bindings.get(name) {
                *existing == *expr
            } else {
                bindings.insert(name.clone(), expr.clone());
                true
            }
        }
        AstPattern::Literal(n) => {
            matches!(expr, SimpleExpr::Num(v) if (*v - *n).abs() < 1e-12)
        }
        AstPattern::Var(v) => {
            // Match a pattern variable against a SimpleExpr::Var
            if let SimpleExpr::Var(ev) = expr {
                if v == ev { return true; }
            }
            // Also treat as a wildcard binding
            if let Some(existing) = bindings.get(v) {
                *existing == *expr
            } else {
                bindings.insert(v.clone(), expr.clone());
                true
            }
        }
        AstPattern::FnCall { name, args } => {
            if let SimpleExpr::Call { name: en, args: ea } = expr {
                if name == en && args.len() == ea.len() {
                    for (p, e) in args.iter().zip(ea.iter()) {
                        if !match_pattern(p, e, bindings) { return false; }
                    }
                    return true;
                }
            }
            // Also match binary ops as function calls: add(x, y) matches x + y
            if let SimpleExpr::BinOp { op, lhs, rhs } = expr {
                let fn_name = op_to_fn_name(op);
                if name == &fn_name && args.len() == 2 {
                    return match_pattern(&args[0], lhs, bindings)
                        && match_pattern(&args[1], rhs, bindings);
                }
            }
            false
        }
        AstPattern::BinOp { op, lhs, rhs } => {
            if let SimpleExpr::BinOp { op: eop, lhs: el, rhs: er } = expr {
                if op == eop {
                    return match_pattern(lhs, el, bindings)
                        && match_pattern(rhs, er, bindings);
                }
            }
            false
        }
        AstPattern::Sequence(pats) => {
            if let SimpleExpr::Block(exprs) = expr {
                if pats.len() == exprs.len() {
                    for (p, e) in pats.iter().zip(exprs.iter()) {
                        if !match_pattern(p, e, bindings) { return false; }
                    }
                    return true;
                }
            }
            false
        }
    }
}

fn op_to_fn_name(op: &str) -> String {
    match op {
        "+" => "add".to_string(),
        "-" => "sub".to_string(),
        "*" => "mul".to_string(),
        "/" => "div".to_string(),
        "%" => "mod".to_string(),
        _ => op.to_string(),
    }
}

fn fn_name_to_op(name: &str) -> Option<&str> {
    match name {
        "add" => Some("+"),
        "sub" => Some("-"),
        "mul" => Some("*"),
        "div" => Some("/"),
        "mod" => Some("%"),
        _ => None,
    }
}

fn apply_template(template: &AstTemplate, bindings: &HashMap<String, SimpleExpr>) -> SimpleExpr {
    match template {
        AstTemplate::Literal(n) => SimpleExpr::Num(*n),
        AstTemplate::Ref(name) => {
            if let Some(expr) = bindings.get(name) {
                expr.clone()
            } else {
                SimpleExpr::Var(name.clone())
            }
        }
        AstTemplate::FnCall { name, args } => {
            let resolved: Vec<SimpleExpr> = args.iter().map(|a| apply_template(a, bindings)).collect();
            // Convert add/sub/mul/div to BinOp
            if let Some(op) = fn_name_to_op(name) {
                if resolved.len() == 2 {
                    return SimpleExpr::BinOp {
                        op: op.to_string(),
                        lhs: Box::new(resolved[0].clone()),
                        rhs: Box::new(resolved[1].clone()),
                    };
                }
            }
            SimpleExpr::Call { name: name.clone(), args: resolved }
        }
        AstTemplate::BinOp { op, lhs, rhs } => {
            SimpleExpr::BinOp {
                op: op.clone(),
                lhs: Box::new(apply_template(lhs, bindings)),
                rhs: Box::new(apply_template(rhs, bindings)),
            }
        }
        AstTemplate::Sequence(templates) => {
            SimpleExpr::Block(templates.iter().map(|t| apply_template(t, bindings)).collect())
        }
    }
}

// ─── Rewrite Rules ──────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct RewriteRule {
    name: String,
    pattern: AstPattern,
    replacement: AstTemplate,
    condition: Option<String>,
}

#[derive(Debug, Clone)]
struct LearnedRule {
    rule: RewriteRule,
    success_count: u64,
    fail_count: u64,
    avg_speedup: f64,
    energy_savings: f64,
}

impl LearnedRule {
    fn success_rate(&self) -> f64 {
        let total = self.success_count + self.fail_count;
        if total == 0 { 0.5 } else { self.success_count as f64 / total as f64 }
    }
}

#[derive(Debug, Clone)]
struct PerfMeasurement {
    rule_name: String,
    before_time_ns: u64,
    after_time_ns: u64,
    before_energy: f64,
    after_energy: f64,
}

// ─── MetaCompiler ───────────────────────────────────────────────────────

struct MetaCompiler {
    rules: Vec<LearnedRule>,
    performance_db: Vec<PerfMeasurement>,
    generation: u64,
}

impl MetaCompiler {
    fn new() -> Self {
        let mut mc = MetaCompiler {
            rules: Vec::new(),
            performance_db: Vec::new(),
            generation: 0,
        };
        mc.add_default_rules();
        mc
    }

    fn add_default_rules(&mut self) {
        // x + 0 → x
        self.add_rule_from_strs("add_zero", "add(x, 0)", "x");
        // 0 + x → x
        self.add_rule_from_strs("zero_add", "add(0, x)", "x");
        // x * 1 → x
        self.add_rule_from_strs("mul_one", "mul(x, 1)", "x");
        // 1 * x → x
        self.add_rule_from_strs("one_mul", "mul(1, x)", "x");
        // x * 0 → 0
        self.add_rule_from_strs("mul_zero", "mul(x, 0)", "0");
        // 0 * x → 0
        self.add_rule_from_strs("zero_mul", "mul(0, x)", "0");
        // x - x → 0
        self.add_rule_from_strs("sub_self", "sub(x, x)", "0");
        // x - 0 → x
        self.add_rule_from_strs("sub_zero", "sub(x, 0)", "x");
        // x / 1 → x
        self.add_rule_from_strs("div_one", "div(x, 1)", "x");
        // x * 2 → x + x  (strength reduction)
        self.add_rule_from_strs("strength_mul2", "mul(x, 2)", "add(x, x)");
        // 2 * x → x + x
        self.add_rule_from_strs("strength_2mul", "mul(2, x)", "add(x, x)");
    }

    fn add_rule_from_strs(&mut self, name: &str, pattern_str: &str, replacement_str: &str) {
        let pattern = parse_pattern(pattern_str);
        let replacement = parse_template(replacement_str);
        let rule = RewriteRule {
            name: name.to_string(),
            pattern,
            replacement,
            condition: None,
        };
        self.rules.push(LearnedRule {
            rule,
            success_count: 0,
            fail_count: 0,
            avg_speedup: 1.0,
            energy_savings: 0.0,
        });
    }

    fn apply_rules_once(&self, expr: &SimpleExpr) -> (SimpleExpr, bool) {
        // Try each rule at the top level
        for lr in &self.rules {
            let mut bindings = HashMap::new();
            if match_pattern(&lr.rule.pattern, expr, &mut bindings) {
                let result = apply_template(&lr.rule.replacement, &bindings);
                return (result, true);
            }
        }
        // Recurse into sub-expressions
        match expr {
            SimpleExpr::BinOp { op, lhs, rhs } => {
                let (new_lhs, changed_l) = self.apply_rules_once(lhs);
                if changed_l {
                    return (SimpleExpr::BinOp {
                        op: op.clone(),
                        lhs: Box::new(new_lhs),
                        rhs: Box::new(rhs.as_ref().clone()),
                    }, true);
                }
                let (new_rhs, changed_r) = self.apply_rules_once(rhs);
                if changed_r {
                    return (SimpleExpr::BinOp {
                        op: op.clone(),
                        lhs: Box::new(lhs.as_ref().clone()),
                        rhs: Box::new(new_rhs),
                    }, true);
                }
            }
            SimpleExpr::Call { name, args } => {
                for (i, a) in args.iter().enumerate() {
                    let (new_a, changed) = self.apply_rules_once(a);
                    if changed {
                        let mut new_args = args.clone();
                        new_args[i] = new_a;
                        return (SimpleExpr::Call { name: name.clone(), args: new_args }, true);
                    }
                }
            }
            SimpleExpr::Block(exprs) => {
                for (i, e) in exprs.iter().enumerate() {
                    let (new_e, changed) = self.apply_rules_once(e);
                    if changed {
                        let mut new_exprs = exprs.clone();
                        new_exprs[i] = new_e;
                        return (SimpleExpr::Block(new_exprs), true);
                    }
                }
            }
            _ => {}
        }
        (expr.clone(), false)
    }

    fn constant_fold(&self, expr: &SimpleExpr) -> SimpleExpr {
        match expr {
            SimpleExpr::BinOp { op, lhs, rhs } => {
                let fl = self.constant_fold(lhs);
                let fr = self.constant_fold(rhs);
                if let (SimpleExpr::Num(l), SimpleExpr::Num(r)) = (&fl, &fr) {
                    let result = match op.as_str() {
                        "+" => Some(l + r),
                        "-" => Some(l - r),
                        "*" => Some(l * r),
                        "/" if *r != 0.0 => Some(l / r),
                        "%" if *r != 0.0 => Some(l % r),
                        _ => None,
                    };
                    if let Some(v) = result {
                        return SimpleExpr::Num(v);
                    }
                }
                SimpleExpr::BinOp { op: op.clone(), lhs: Box::new(fl), rhs: Box::new(fr) }
            }
            SimpleExpr::Call { name, args } => {
                let folded: Vec<SimpleExpr> = args.iter().map(|a| self.constant_fold(a)).collect();
                SimpleExpr::Call { name: name.clone(), args: folded }
            }
            SimpleExpr::Block(exprs) => {
                SimpleExpr::Block(exprs.iter().map(|e| self.constant_fold(e)).collect())
            }
            other => other.clone(),
        }
    }

    fn optimize(&self, expr: &SimpleExpr, max_passes: usize) -> SimpleExpr {
        let mut current = expr.clone();
        for _ in 0..max_passes {
            // Constant folding
            current = self.constant_fold(&current);
            // Rule application
            let (next, changed) = self.apply_rules_once(&current);
            if !changed {
                break;
            }
            current = next;
        }
        // Final constant fold
        self.constant_fold(&current)
    }

    fn suggest_optimizations(&self, expr: &SimpleExpr) -> Vec<String> {
        let mut suggestions = Vec::new();
        // Check each rule
        for lr in &self.rules {
            if self.check_applicable_anywhere(&lr.rule.pattern, expr) {
                suggestions.push(format!(
                    "Rule '{}' is applicable (success_rate={:.1}%, avg_speedup={:.2}x)",
                    lr.rule.name,
                    lr.success_rate() * 100.0,
                    lr.avg_speedup
                ));
            }
        }
        // Check for constant folding opportunities
        let folded = self.constant_fold(expr);
        if folded != *expr {
            suggestions.push("Constant folding opportunities detected".to_string());
        }
        // CSE check
        let cse_count = self.count_common_subexprs(expr);
        if cse_count > 0 {
            suggestions.push(format!(
                "{} common sub-expression(s) found — consider CSE pass",
                cse_count
            ));
        }
        if suggestions.is_empty() {
            suggestions.push("Expression appears already optimal".to_string());
        }
        suggestions
    }

    fn check_applicable_anywhere(&self, pattern: &AstPattern, expr: &SimpleExpr) -> bool {
        let mut bindings = HashMap::new();
        if match_pattern(pattern, expr, &mut bindings) {
            return true;
        }
        match expr {
            SimpleExpr::BinOp { lhs, rhs, .. } => {
                self.check_applicable_anywhere(pattern, lhs)
                    || self.check_applicable_anywhere(pattern, rhs)
            }
            SimpleExpr::Call { args, .. } => {
                args.iter().any(|a| self.check_applicable_anywhere(pattern, a))
            }
            SimpleExpr::Block(exprs) => {
                exprs.iter().any(|e| self.check_applicable_anywhere(pattern, e))
            }
            _ => false,
        }
    }

    fn count_common_subexprs(&self, expr: &SimpleExpr) -> usize {
        let mut seen: HashMap<String, usize> = HashMap::new();
        self.collect_subexprs(expr, &mut seen);
        seen.values().filter(|&&c| c > 1).count()
    }

    fn collect_subexprs(&self, expr: &SimpleExpr, seen: &mut HashMap<String, usize>) {
        let repr = expr.to_string_repr();
        // Only count non-trivial expressions
        match expr {
            SimpleExpr::Num(_) | SimpleExpr::Var(_) => {}
            _ => {
                *seen.entry(repr).or_insert(0) += 1;
            }
        }
        match expr {
            SimpleExpr::BinOp { lhs, rhs, .. } => {
                self.collect_subexprs(lhs, seen);
                self.collect_subexprs(rhs, seen);
            }
            SimpleExpr::Call { args, .. } => {
                for a in args { self.collect_subexprs(a, seen); }
            }
            SimpleExpr::Block(exprs) => {
                for e in exprs { self.collect_subexprs(e, seen); }
            }
            _ => {}
        }
    }

    fn learn_from_execution(&mut self, rule_name: &str, before_ns: u64, after_ns: u64) {
        let speedup = if after_ns > 0 {
            before_ns as f64 / after_ns as f64
        } else {
            1.0
        };
        let is_improvement = after_ns < before_ns;
        let energy_before = before_ns as f64 * 0.001; // simple energy model
        let energy_after = after_ns as f64 * 0.001;

        self.performance_db.push(PerfMeasurement {
            rule_name: rule_name.to_string(),
            before_time_ns: before_ns,
            after_time_ns: after_ns,
            before_energy: energy_before,
            after_energy: energy_after,
        });

        for lr in &mut self.rules {
            if lr.rule.name == rule_name {
                if is_improvement {
                    lr.success_count += 1;
                } else {
                    lr.fail_count += 1;
                }
                // Running average of speedup
                let total = lr.success_count + lr.fail_count;
                lr.avg_speedup = lr.avg_speedup * ((total - 1) as f64 / total as f64)
                    + speedup * (1.0 / total as f64);
                lr.energy_savings = lr.energy_savings * ((total - 1) as f64 / total as f64)
                    + (energy_before - energy_after) * (1.0 / total as f64);
                break;
            }
        }
    }

    fn rule_stats(&self) -> Vec<HashMap<String, String>> {
        self.rules.iter().map(|lr| {
            let mut m = HashMap::new();
            m.insert("name".to_string(), lr.rule.name.clone());
            m.insert("success_rate".to_string(), format!("{:.4}", lr.success_rate()));
            m.insert("avg_speedup".to_string(), format!("{:.4}", lr.avg_speedup));
            m.insert("success_count".to_string(), format!("{}", lr.success_count));
            m.insert("fail_count".to_string(), format!("{}", lr.fail_count));
            m.insert("energy_savings".to_string(), format!("{:.4}", lr.energy_savings));
            m
        }).collect()
    }

    fn genetic_optimize(&self, expr: &SimpleExpr, pop_size: usize, generations: usize) -> SimpleExpr {
        let mut rng = Xorshift::new(42 + self.generation);
        let mut population: Vec<SimpleExpr> = Vec::with_capacity(pop_size);
        // Seed population with the original and some mutations
        population.push(expr.clone());
        population.push(self.optimize(expr, 20));
        for _ in 2..pop_size {
            let mutated = self.mutate_expr(&population[0], &mut rng, 3);
            population.push(mutated);
        }

        for _gen in 0..generations {
            // Evaluate fitness (lower complexity = better)
            let mut scored: Vec<(f64, SimpleExpr)> = population.iter().map(|e| {
                let (flops, mem, depth) = e.complexity();
                let fitness = flops as f64 * 2.0 + mem as f64 + depth as f64 * 0.5;
                (fitness, e.clone())
            }).collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

            // Keep top half
            let survivors: Vec<SimpleExpr> = scored.iter()
                .take(std::cmp::max(pop_size / 2, 1))
                .map(|(_, e)| e.clone())
                .collect();

            population.clear();
            for s in &survivors {
                population.push(s.clone());
            }
            // Fill rest with mutations of survivors
            while population.len() < pop_size {
                let parent_idx = rng.next_usize(survivors.len());
                let child = self.mutate_expr(&survivors[parent_idx], &mut rng, 2);
                // Also apply rule-based optimization on some children
                let child = if rng.next_f64() < 0.5 {
                    self.optimize(&child, 5)
                } else {
                    child
                };
                population.push(child);
            }
        }

        // Return best
        population.iter()
            .min_by(|a, b| {
                let (af, am, ad) = a.complexity();
                let (bf, bm, bd) = b.complexity();
                let fa = af as f64 * 2.0 + am as f64 + ad as f64 * 0.5;
                let fb = bf as f64 * 2.0 + bm as f64 + bd as f64 * 0.5;
                fa.partial_cmp(&fb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned()
            .unwrap_or_else(|| expr.clone())
    }

    fn mutate_expr(&self, expr: &SimpleExpr, rng: &mut Xorshift, depth: usize) -> SimpleExpr {
        if depth == 0 {
            return expr.clone();
        }
        let choice = rng.next_usize(6);
        match choice {
            0 => {
                // Swap binary op
                if let SimpleExpr::BinOp { op, lhs, rhs } = expr {
                    let ops = ["+", "-", "*"];
                    let new_op = ops[rng.next_usize(ops.len())];
                    SimpleExpr::BinOp {
                        op: new_op.to_string(),
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                    }
                } else {
                    expr.clone()
                }
            }
            1 => {
                // Recurse into left
                if let SimpleExpr::BinOp { op, lhs, rhs } = expr {
                    SimpleExpr::BinOp {
                        op: op.clone(),
                        lhs: Box::new(self.mutate_expr(lhs, rng, depth - 1)),
                        rhs: rhs.clone(),
                    }
                } else {
                    expr.clone()
                }
            }
            2 => {
                // Recurse into right
                if let SimpleExpr::BinOp { op, lhs, rhs } = expr {
                    SimpleExpr::BinOp {
                        op: op.clone(),
                        lhs: lhs.clone(),
                        rhs: Box::new(self.mutate_expr(rhs, rng, depth - 1)),
                    }
                } else {
                    expr.clone()
                }
            }
            3 => {
                // Apply optimization rules
                self.optimize(expr, 5)
            }
            4 => {
                // Constant fold
                self.constant_fold(expr)
            }
            5 => {
                // Commute if binary op
                if let SimpleExpr::BinOp { op, lhs, rhs } = expr {
                    if op == "+" || op == "*" {
                        SimpleExpr::BinOp {
                            op: op.clone(),
                            lhs: rhs.clone(),
                            rhs: lhs.clone(),
                        }
                    } else {
                        expr.clone()
                    }
                } else {
                    expr.clone()
                }
            }
            _ => expr.clone(),
        }
    }

    fn self_improve(&mut self, test_suite: &[String]) -> String {
        self.generation += 1;
        let mut report = format!("=== Self-Improvement Report (Generation {}) ===\n", self.generation);

        let mut rule_scores: Vec<(String, f64, f64)> = Vec::new(); // (name, total_reduction, success_rate)

        for lr in &self.rules {
            let mut total_complexity_before = 0u64;
            let mut total_complexity_after = 0u64;
            let mut successes = 0u64;

            for test_str in test_suite {
                let expr = parse_simple_expr_str(test_str);
                let (cb, _, _) = expr.complexity();
                total_complexity_before += cb;

                if self.check_applicable_anywhere(&lr.rule.pattern, &expr) {
                    let optimized = self.optimize(&expr, 20);
                    let (ca, _, _) = optimized.complexity();
                    total_complexity_after += ca;
                    if ca < cb { successes += 1; }
                } else {
                    total_complexity_after += cb;
                }
            }

            let reduction = if total_complexity_before > 0 {
                1.0 - (total_complexity_after as f64 / total_complexity_before as f64)
            } else {
                0.0
            };
            let sr = if test_suite.is_empty() { 0.0 } else { successes as f64 / test_suite.len() as f64 };
            rule_scores.push((lr.rule.name.clone(), reduction, sr));
        }

        // Report on rules
        report.push_str("\nRule Performance:\n");
        for (name, reduction, sr) in &rule_scores {
            report.push_str(&format!(
                "  {} — reduction={:.2}%, test_success={:.1}%\n",
                name, reduction * 100.0, sr * 100.0
            ));
        }

        // Prune rules with negative impact
        let mut pruned = 0;
        self.rules.retain(|lr| {
            let dominated = rule_scores.iter()
                .find(|(n, _, _)| n == &lr.rule.name)
                .map(|(_, red, _)| *red < -0.1)
                .unwrap_or(false);
            if dominated && lr.success_count + lr.fail_count > 10 && lr.success_rate() < 0.2 {
                pruned += 1;
                false
            } else {
                true
            }
        });
        report.push_str(&format!("\nPruned {} underperforming rules.\n", pruned));

        // Generate new candidate rules from observed patterns
        let mut candidates_added = 0;
        // Try to discover double-negation pattern: sub(0, sub(0, x)) → x
        let has_double_neg = self.rules.iter().any(|lr| lr.rule.name == "double_neg");
        if !has_double_neg {
            self.add_rule_from_strs("double_neg", "sub(0, sub(0, x))", "x");
            candidates_added += 1;
        }
        // Try add(x, x) → mul(2, x) if not present
        let has_add_self = self.rules.iter().any(|lr| lr.rule.name == "add_self_to_mul");
        if !has_add_self {
            self.add_rule_from_strs("add_self_to_mul", "add(x, x)", "mul(2, x)");
            candidates_added += 1;
        }
        // div(x, x) → 1
        let has_div_self = self.rules.iter().any(|lr| lr.rule.name == "div_self");
        if !has_div_self {
            self.add_rule_from_strs("div_self", "div(x, x)", "1");
            candidates_added += 1;
        }
        // mul(x, mul(y, z)) → mul(mul(x, y), z) — associativity
        let has_mul_assoc = self.rules.iter().any(|lr| lr.rule.name == "mul_assoc");
        if !has_mul_assoc {
            self.add_rule_from_strs("mul_assoc", "mul(x, mul(y, z))", "mul(mul(x, y), z)");
            candidates_added += 1;
        }

        report.push_str(&format!("Generated {} new candidate rules.\n", candidates_added));
        report.push_str(&format!("Total rules: {}\n", self.rules.len()));

        report
    }
}

// ─── Global storage ─────────────────────────────────────────────────────

static COMPILERS: LazyLock<Mutex<HashMap<usize, MetaCompiler>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));
static NEXT_COMPILER_ID: LazyLock<Mutex<usize>> =
    LazyLock::new(|| Mutex::new(1));

fn alloc_compiler() -> usize {
    let mut id = NEXT_COMPILER_ID.lock().unwrap();
    let cid = *id;
    *id += 1;
    let mut map = COMPILERS.lock().unwrap();
    map.insert(cid, MetaCompiler::new());
    cid
}

fn with_compiler<F, R>(id: usize, f: F) -> Result<R, String>
where F: FnOnce(&MetaCompiler) -> R {
    let map = COMPILERS.lock().unwrap();
    let mc = map.get(&id).ok_or_else(|| format!("invalid compiler id: {}", id))?;
    Ok(f(mc))
}

fn with_compiler_mut<F, R>(id: usize, f: F) -> Result<R, String>
where F: FnOnce(&mut MetaCompiler) -> R {
    let mut map = COMPILERS.lock().unwrap();
    let mc = map.get_mut(&id).ok_or_else(|| format!("invalid compiler id: {}", id))?;
    Ok(f(mc))
}

// ─── SimpleExpr to Value conversion ─────────────────────────────────────

fn expr_to_value(expr: &SimpleExpr) -> Value {
    match expr {
        SimpleExpr::Num(n) => Value::Array(vec![
            Value::String("num".to_string()),
            Value::Float(*n),
        ]),
        SimpleExpr::Var(v) => Value::Array(vec![
            Value::String("var".to_string()),
            Value::String(v.clone()),
        ]),
        SimpleExpr::BinOp { op, lhs, rhs } => Value::Array(vec![
            Value::String("binop".to_string()),
            Value::String(op.clone()),
            expr_to_value(lhs),
            expr_to_value(rhs),
        ]),
        SimpleExpr::Call { name, args } => {
            let mut arr = vec![
                Value::String("call".to_string()),
                Value::String(name.clone()),
            ];
            for a in args {
                arr.push(expr_to_value(a));
            }
            Value::Array(arr)
        }
        SimpleExpr::Block(exprs) => {
            let mut arr = vec![Value::String("block".to_string())];
            for e in exprs {
                arr.push(expr_to_value(e));
            }
            Value::Array(arr)
        }
    }
}

// ─── Profile helper ─────────────────────────────────────────────────────

fn profile_expression(expr: &SimpleExpr, num_runs: usize) -> (u64, u64, u64) {
    // Simulate profiling via complexity-based timing model
    let (flops, mem, depth) = expr.complexity();
    let base_ns = flops * 10 + mem * 5 + depth * 2;
    let mut rng = Xorshift::new(base_ns + 7);

    let mut min_ns = u64::MAX;
    let mut max_ns = 0u64;
    let mut total_ns = 0u64;

    for _ in 0..num_runs.max(1) {
        let jitter = (rng.next_u64() % (base_ns.max(1) / 4 + 1)) as i64 - (base_ns.max(1) / 8) as i64;
        let t = (base_ns as i64 + jitter).max(1) as u64;
        if t < min_ns { min_ns = t; }
        if t > max_ns { max_ns = t; }
        total_ns += t;
    }

    let avg_ns = total_ns / num_runs.max(1) as u64;
    (avg_ns, min_ns, max_ns)
}

// ─── Builtin functions ──────────────────────────────────────────────────

fn builtin_meta_compiler_new(_env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let id = alloc_compiler();
    Ok(Value::Int(id as i128))
}

fn builtin_add_rewrite_rule(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("add_rewrite_rule requires (compiler_id, name, pattern_str, replacement_str)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let name = value_to_string(&args[1]);
    let pattern_str = value_to_string(&args[2]);
    let replacement_str = value_to_string(&args[3]);

    with_compiler_mut(id, |mc| {
        mc.add_rule_from_strs(&name, &pattern_str, &replacement_str);
    })?;
    Ok(Value::Void)
}

fn builtin_optimize_expr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("optimize_expr requires (compiler_id, expr_str)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let expr_str = value_to_string(&args[1]);

    let s = with_compiler(id, |mc| {
        let expr = parse_simple_expr_str(&expr_str);
        let optimized = mc.optimize(&expr, 50);
        optimized.to_string_repr()
    })?;
    Ok(Value::String(s))
}

fn builtin_analyze_complexity(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("analyze_complexity requires (expr_str)".to_string());
    }
    let expr_str = value_to_string(&args[0]);
    let expr = parse_simple_expr_str(&expr_str);
    let (flops, mem_ops, depth) = expr.complexity();

    let mut m = HashMap::new();
    m.insert("flops".to_string(), Value::Float(flops as f64));
    m.insert("memory_ops".to_string(), Value::Float(mem_ops as f64));
    m.insert("depth".to_string(), Value::Float(depth as f64));
    Ok(Value::HashMap(m))
}

fn builtin_suggest_optimizations(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("suggest_optimizations requires (compiler_id, expr_str)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let expr_str = value_to_string(&args[1]);

    let suggestions = with_compiler(id, |mc| {
        let expr = parse_simple_expr_str(&expr_str);
        mc.suggest_optimizations(&expr)
    })?;
    Ok(Value::Array(suggestions.into_iter().map(Value::String).collect()))
}

fn builtin_learn_from_execution(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("learn_from_execution requires (compiler_id, rule_name, before_ns, after_ns)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let rule_name = value_to_string(&args[1]);
    let before_ns = value_to_f64(&args[2]) as u64;
    let after_ns = value_to_f64(&args[3]) as u64;

    with_compiler_mut(id, |mc| {
        mc.learn_from_execution(&rule_name, before_ns, after_ns);
    })?;
    Ok(Value::Void)
}

fn builtin_rule_stats(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("rule_stats requires (compiler_id)".to_string());
    }
    let id = value_to_usize(&args[0]);

    let stats = with_compiler(id, |mc| mc.rule_stats())?;
    Ok(Value::Array(stats.into_iter().map(|m| {
        let hm: HashMap<String, Value> = m.into_iter()
            .map(|(k, v)| (k, Value::String(v)))
            .collect();
        Value::HashMap(hm)
    }).collect()))
}

fn builtin_genetic_optimize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("genetic_optimize requires (compiler_id, expr_str, population_size, generations)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let expr_str = value_to_string(&args[1]);
    let pop_size = value_to_usize(&args[2]).max(2);
    let generations = value_to_usize(&args[3]).max(1);

    let s = with_compiler(id, |mc| {
        let expr = parse_simple_expr_str(&expr_str);
        let best = mc.genetic_optimize(&expr, pop_size, generations);
        best.to_string_repr()
    })?;
    Ok(Value::String(s))
}

fn builtin_verify_rewrite(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("verify_rewrite requires (expr_str, rewritten_str, test_inputs)".to_string());
    }
    let expr_str = value_to_string(&args[0]);
    let rewritten_str = value_to_string(&args[1]);
    let test_inputs = &args[2];

    let expr = parse_simple_expr_str(&expr_str);
    let rewritten = parse_simple_expr_str(&rewritten_str);

    let mut vars = Vec::new();
    expr.collect_vars(&mut vars);
    rewritten.collect_vars(&mut vars);

    let test_sets: Vec<Vec<f64>> = match test_inputs {
        Value::Array(arr) => {
            arr.iter().map(|v| {
                match v {
                    Value::Array(inner) => inner.iter().map(|x| value_to_f64(x)).collect(),
                    Value::Float(n) => vec![*n],
                    Value::Int(n) => vec![*n as f64],
                    _ => vec![],
                }
            }).collect()
        }
        _ => vec![],
    };

    let inputs_to_check = if test_sets.is_empty() {
        let mut rng = Xorshift::new(12345);
        let mut generated = Vec::new();
        for _ in 0..10 {
            let vals: Vec<f64> = (0..vars.len()).map(|_| {
                (rng.next_u64() % 200) as f64 / 10.0 - 10.0
            }).collect();
            generated.push(vals);
        }
        generated
    } else {
        test_sets
    };

    for vals in &inputs_to_check {
        let mut env = HashMap::new();
        for (i, var) in vars.iter().enumerate() {
            if i < vals.len() {
                env.insert(var.clone(), vals[i]);
            }
        }
        let r1 = expr.evaluate(&env);
        let r2 = rewritten.evaluate(&env);
        match (r1, r2) {
            (Some(v1), Some(v2)) => {
                if (v1 - v2).abs() > 1e-9 * v1.abs().max(1.0) {
                    return Ok(Value::Bool(false));
                }
            }
            (None, None) => {}
            _ => return Ok(Value::Bool(false)),
        }
    }
    Ok(Value::Bool(true))
}

fn builtin_self_improve(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("self_improve requires (compiler_id, test_suite)".to_string());
    }
    let id = value_to_usize(&args[0]);
    let test_suite: Vec<String> = match &args[1] {
        Value::Array(arr) => arr.iter().map(|v| value_to_string(v)).collect(),
        _ => vec![value_to_string(&args[1])],
    };

    let report = with_compiler_mut(id, |mc| {
        mc.self_improve(&test_suite)
    })?;
    Ok(Value::String(report))
}

fn builtin_profile_expr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("profile_expr requires (expr_str, num_runs)".to_string());
    }
    let expr_str = value_to_string(&args[0]);
    let num_runs = if args.len() > 1 { value_to_usize(&args[1]).max(1) } else { 100 };

    let expr = parse_simple_expr_str(&expr_str);
    let (avg_ns, min_ns, max_ns) = profile_expression(&expr, num_runs);

    let mut m = HashMap::new();
    m.insert("avg_ns".to_string(), Value::Float(avg_ns as f64));
    m.insert("min_ns".to_string(), Value::Float(min_ns as f64));
    m.insert("max_ns".to_string(), Value::Float(max_ns as f64));
    Ok(Value::HashMap(m))
}

fn builtin_parse_simple_expr(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("parse_simple_expr requires (expr_str)".to_string());
    }
    let expr_str = value_to_string(&args[0]);
    let expr = parse_simple_expr_str(&expr_str);
    Ok(expr_to_value(&expr))
}

// ─── Registration ───────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert(
        "meta_compiler_new".to_string(),
        FnDef::Builtin(builtin_meta_compiler_new),
    );
    env.functions.insert(
        "add_rewrite_rule".to_string(),
        FnDef::Builtin(builtin_add_rewrite_rule),
    );
    env.functions.insert(
        "optimize_expr".to_string(),
        FnDef::Builtin(builtin_optimize_expr),
    );
    env.functions.insert(
        "analyze_complexity".to_string(),
        FnDef::Builtin(builtin_analyze_complexity),
    );
    env.functions.insert(
        "suggest_optimizations".to_string(),
        FnDef::Builtin(builtin_suggest_optimizations),
    );
    env.functions.insert(
        "learn_from_execution".to_string(),
        FnDef::Builtin(builtin_learn_from_execution),
    );
    env.functions.insert(
        "rule_stats".to_string(),
        FnDef::Builtin(builtin_rule_stats),
    );
    env.functions.insert(
        "genetic_optimize".to_string(),
        FnDef::Builtin(builtin_genetic_optimize),
    );
    env.functions.insert(
        "verify_rewrite".to_string(),
        FnDef::Builtin(builtin_verify_rewrite),
    );
    env.functions.insert(
        "self_improve".to_string(),
        FnDef::Builtin(builtin_self_improve),
    );
    env.functions.insert(
        "profile_expr".to_string(),
        FnDef::Builtin(builtin_profile_expr),
    );
    env.functions.insert(
        "parse_simple_expr".to_string(),
        FnDef::Builtin(builtin_parse_simple_expr),
    );
}

// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xorshift_deterministic() {
        let mut rng1 = Xorshift::new(42);
        let mut rng2 = Xorshift::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }

    #[test]
    fn test_xorshift_nonzero_seed() {
        let mut rng = Xorshift::new(0);
        assert_ne!(rng.next_u64(), 0);
    }

    #[test]
    fn test_parse_simple_num() {
        let e = parse_simple_expr_str("42");
        assert_eq!(e, SimpleExpr::Num(42.0));
    }

    #[test]
    fn test_parse_simple_var() {
        let e = parse_simple_expr_str("x");
        assert_eq!(e, SimpleExpr::Var("x".to_string()));
    }

    #[test]
    fn test_parse_binop() {
        let e = parse_simple_expr_str("x + 3");
        assert_eq!(e, SimpleExpr::BinOp {
            op: "+".to_string(),
            lhs: Box::new(SimpleExpr::Var("x".to_string())),
            rhs: Box::new(SimpleExpr::Num(3.0)),
        });
    }

    #[test]
    fn test_parse_fn_call() {
        let e = parse_simple_expr_str("add(x, 0)");
        assert_eq!(e, SimpleExpr::Call {
            name: "add".to_string(),
            args: vec![SimpleExpr::Var("x".to_string()), SimpleExpr::Num(0.0)],
        });
    }

    #[test]
    fn test_parse_nested() {
        let e = parse_simple_expr_str("mul(add(x, 1), 2)");
        assert_eq!(e, SimpleExpr::Call {
            name: "mul".to_string(),
            args: vec![
                SimpleExpr::Call {
                    name: "add".to_string(),
                    args: vec![SimpleExpr::Var("x".to_string()), SimpleExpr::Num(1.0)],
                },
                SimpleExpr::Num(2.0),
            ],
        });
    }

    #[test]
    fn test_constant_folding() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("3 + 4");
        let result = mc.constant_fold(&e);
        assert_eq!(result, SimpleExpr::Num(7.0));
    }

    #[test]
    fn test_constant_folding_nested() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("2 * 3 + 1");
        let result = mc.constant_fold(&e);
        assert_eq!(result, SimpleExpr::Num(7.0));
    }

    #[test]
    fn test_add_zero_rule() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x + 0");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::Var("x".to_string()));
    }

    #[test]
    fn test_mul_one_rule() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x * 1");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::Var("x".to_string()));
    }

    #[test]
    fn test_mul_zero_rule() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x * 0");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::Num(0.0));
    }

    #[test]
    fn test_sub_self_rule() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x - x");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::Num(0.0));
    }

    #[test]
    fn test_strength_reduction() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x * 2");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::BinOp {
            op: "+".to_string(),
            lhs: Box::new(SimpleExpr::Var("x".to_string())),
            rhs: Box::new(SimpleExpr::Var("x".to_string())),
        });
    }

    #[test]
    fn test_complexity() {
        let e = parse_simple_expr_str("x + y * z");
        let (flops, _mem, _depth) = e.complexity();
        assert!(flops >= 2); // at least + and *
    }

    #[test]
    fn test_evaluate() {
        let e = parse_simple_expr_str("x + y * 2");
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 3.0);
        vars.insert("y".to_string(), 4.0);
        let result = e.evaluate(&vars);
        assert_eq!(result, Some(11.0));
    }

    #[test]
    fn test_to_string_repr_roundtrip() {
        let e = parse_simple_expr_str("add(x, mul(y, 3))");
        let s = e.to_string_repr();
        assert!(s.contains("add"));
        assert!(s.contains("mul"));
    }

    #[test]
    fn test_pattern_matching() {
        let pat = parse_pattern("add(x, 0)");
        let expr = SimpleExpr::Call {
            name: "add".to_string(),
            args: vec![SimpleExpr::Var("y".to_string()), SimpleExpr::Num(0.0)],
        };
        let mut bindings = HashMap::new();
        assert!(match_pattern(&pat, &expr, &mut bindings));
        assert_eq!(bindings.get("x"), Some(&SimpleExpr::Var("y".to_string())));
    }

    #[test]
    fn test_pattern_matching_binop() {
        // add(x, 0) should match y + 0
        let pat = parse_pattern("add(x, 0)");
        let expr = parse_simple_expr_str("y + 0");
        let mut bindings = HashMap::new();
        assert!(match_pattern(&pat, &expr, &mut bindings));
    }

    #[test]
    fn test_suggest_optimizations() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x + 0");
        let suggestions = mc.suggest_optimizations(&e);
        assert!(!suggestions.is_empty());
        assert!(suggestions.iter().any(|s| s.contains("add_zero")));
    }

    #[test]
    fn test_learn_from_execution() {
        let mut mc = MetaCompiler::new();
        mc.learn_from_execution("add_zero", 1000, 500);
        let stats = mc.rule_stats();
        let add_zero = stats.iter().find(|s| s["name"] == "add_zero").unwrap();
        assert_eq!(add_zero["success_count"], "1");
    }

    #[test]
    fn test_genetic_optimize() {
        let mc = MetaCompiler::new();
        let e = parse_simple_expr_str("x + 0 + y * 1");
        let best = mc.genetic_optimize(&e, 10, 5);
        let (orig_flops, _, _) = e.complexity();
        let (opt_flops, _, _) = best.complexity();
        assert!(opt_flops <= orig_flops);
    }

    #[test]
    fn test_verify_rewrite_valid() {
        let e1 = parse_simple_expr_str("x + 0");
        let e2 = parse_simple_expr_str("x");
        let mut env = HashMap::new();
        env.insert("x".to_string(), 5.0);
        assert_eq!(e1.evaluate(&env), e2.evaluate(&env));
    }

    #[test]
    fn test_verify_rewrite_invalid() {
        let e1 = parse_simple_expr_str("x + 1");
        let e2 = parse_simple_expr_str("x");
        let mut env = HashMap::new();
        env.insert("x".to_string(), 5.0);
        assert_ne!(e1.evaluate(&env), e2.evaluate(&env));
    }

    #[test]
    fn test_self_improve() {
        let mut mc = MetaCompiler::new();
        let test_suite = vec![
            "x + 0".to_string(),
            "y * 1".to_string(),
            "3 + 4".to_string(),
            "a * 0 + b".to_string(),
        ];
        let report = mc.self_improve(&test_suite);
        assert!(report.contains("Self-Improvement Report"));
        assert!(report.contains("Generation 1"));
        assert!(mc.generation == 1);
    }

    #[test]
    fn test_profile_expr() {
        let e = parse_simple_expr_str("x + y * z");
        let (avg, min, max) = profile_expression(&e, 100);
        assert!(min <= avg);
        assert!(avg <= max);
    }

    #[test]
    fn test_expr_to_value() {
        let e = parse_simple_expr_str("x + 1");
        let v = expr_to_value(&e);
        match v {
            Value::Array(arr) => {
                assert!(matches!(&arr[0], Value::String(s) if s == "binop"));
                assert!(matches!(&arr[1], Value::String(s) if s == "+"));
            }
            _ => panic!("expected array"),
        }
    }

    #[test]
    fn test_tokenizer() {
        let tokens = tokenize_expr("add(x, 0)");
        assert_eq!(tokens, vec!["add", "(", "x", ",", "0", ")"]);
    }

    #[test]
    fn test_split_top_level() {
        let parts = split_top_level("x, add(y, z), 3");
        assert_eq!(parts, vec!["x", "add(y, z)", "3"]);
    }

    #[test]
    fn test_common_subexpr_detection() {
        let mc = MetaCompiler::new();
        // (x + y) * (x + y)
        let e = parse_simple_expr_str("(x + y) * (x + y)");
        let count = mc.count_common_subexprs(&e);
        assert!(count >= 1);
    }

    #[test]
    fn test_block_parsing() {
        let e = parse_simple_expr_str("{ x + 1; y * 2 }");
        match e {
            SimpleExpr::Block(exprs) => assert_eq!(exprs.len(), 2),
            _ => panic!("expected block"),
        }
    }

    #[test]
    fn test_alloc_and_use_compiler() {
        let id = alloc_compiler();
        let result = with_compiler(id, |mc| mc.rules.len());
        assert!(result.unwrap() > 0);
    }

    #[test]
    fn test_custom_rule() {
        let mut mc = MetaCompiler::new();
        mc.add_rule_from_strs("custom_double_neg", "sub(0, sub(0, x))", "x");
        let e = parse_simple_expr_str("0 - (0 - x)");
        let optimized = mc.optimize(&e, 10);
        assert_eq!(optimized, SimpleExpr::Var("x".to_string()));
    }

    #[test]
    fn test_multiple_passes() {
        let mc = MetaCompiler::new();
        // x + 0 + 0 should reduce to x in multiple passes
        let e = parse_simple_expr_str("(x + 0) + 0");
        let optimized = mc.optimize(&e, 20);
        assert_eq!(optimized, SimpleExpr::Var("x".to_string()));
    }

    #[test]
    fn test_operator_precedence() {
        let e = parse_simple_expr_str("x + y * z");
        // * should bind tighter than +
        match &e {
            SimpleExpr::BinOp { op, rhs, .. } => {
                assert_eq!(op, "+");
                match rhs.as_ref() {
                    SimpleExpr::BinOp { op, .. } => assert_eq!(op, "*"),
                    _ => panic!("expected binop"),
                }
            }
            _ => panic!("expected binop"),
        }
    }

    #[test]
    fn test_collect_vars() {
        let e = parse_simple_expr_str("x + y * z + x");
        let mut vars = Vec::new();
        e.collect_vars(&mut vars);
        assert!(vars.contains(&"x".to_string()));
        assert!(vars.contains(&"y".to_string()));
        assert!(vars.contains(&"z".to_string()));
        assert_eq!(vars.len(), 3); // no duplicates
    }

    #[test]
    fn test_evaluate_with_functions() {
        let e = parse_simple_expr_str("sqrt(x)");
        let mut vars = HashMap::new();
        vars.insert("x".to_string(), 9.0);
        let result = e.evaluate(&vars);
        assert!((result.unwrap() - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_rule_stats_initial() {
        let mc = MetaCompiler::new();
        let stats = mc.rule_stats();
        assert!(stats.len() >= 11);
        for s in &stats {
            assert!(s.contains_key("name"));
            assert!(s.contains_key("success_rate"));
        }
    }
}
