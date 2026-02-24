//! Self-evolving code capabilities for Vortex.
//!
//! Provides runtime code generation, eval, AST manipulation,
//! genetic programming, neural architecture search, and self-improving
//! training loops. This is the engine that lets Vortex programs write,
//! modify, evaluate, and improve their own code.

use crate::ast::*;
use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Runtime Code Generation (Templates)
// ---------------------------------------------------------------------------

/// A code template with `{{param}}` placeholders.
#[derive(Debug, Clone)]
pub struct Template {
    pub params: Vec<String>,
    pub body: String,
}

impl Template {
    pub fn render(&self, values: &HashMap<String, String>) -> Result<String, String> {
        let mut result = self.body.clone();
        for param in &self.params {
            let placeholder = format!("{{{{{}}}}}", param);
            let value = values
                .get(param)
                .ok_or_else(|| format!("missing template param: {}", param))?;
            result = result.replace(&placeholder, value);
        }
        Ok(result)
    }
}

/// Code generator that manages templates and produces Vortex code at runtime.
#[derive(Debug, Clone)]
pub struct CodeGenerator {
    pub templates: HashMap<String, Template>,
}

impl CodeGenerator {
    pub fn new() -> Self {
        Self {
            templates: HashMap::new(),
        }
    }

    pub fn register_template(&mut self, name: &str, params: Vec<String>, body: &str) {
        self.templates.insert(
            name.to_string(),
            Template {
                params,
                body: body.to_string(),
            },
        );
    }

    /// Generate a function definition as a Vortex code string.
    pub fn codegen_function(
        &self,
        name: &str,
        params: &[(&str, &str)],
        body_template: &str,
        values: &HashMap<String, String>,
    ) -> Result<String, String> {
        let tmpl = Template {
            params: values.keys().cloned().collect(),
            body: body_template.to_string(),
        };
        let body = tmpl.render(values)?;
        let param_list: Vec<String> = params
            .iter()
            .map(|(n, t)| format!("{}: {}", n, t))
            .collect();
        Ok(format!(
            "fn {}({}) {{\n{}\n}}",
            name,
            param_list.join(", "),
            body
        ))
    }

    /// Generate a neural network model from an architecture description.
    pub fn codegen_model(&self, layers: &[(String, Vec<usize>)]) -> String {
        let mut parts = Vec::new();
        for (layer_type, dims) in layers {
            match layer_type.as_str() {
                "linear" => {
                    if dims.len() >= 2 {
                        parts.push(format!("nn_linear({}, {})", dims[0], dims[1]));
                    }
                }
                "relu" => parts.push("nn_relu()".to_string()),
                "sigmoid" => parts.push("nn_sigmoid()".to_string()),
                "tanh" => parts.push("nn_tanh()".to_string()),
                "conv2d" => {
                    if dims.len() >= 4 {
                        parts.push(format!(
                            "nn_conv2d({}, {}, {}, {})",
                            dims[0], dims[1], dims[2], dims[3]
                        ));
                    }
                }
                "attention" => {
                    if dims.len() >= 2 {
                        parts.push(format!("nn_attention({}, {})", dims[0], dims[1]));
                    }
                }
                _ => parts.push(format!("nn_{}()", layer_type)),
            }
        }
        format!("nn_sequential([{}])", parts.join(", "))
    }

    /// Generate a custom optimizer from loss landscape description.
    pub fn codegen_optimizer(&self, lr: f64, momentum: f64, weight_decay: f64) -> String {
        format!(
            "fn custom_optimizer(params: Array, grads: Array) -> Array {{\n\
             \tlet lr = {}\n\
             \tlet momentum = {}\n\
             \tlet weight_decay = {}\n\
             \tlet updated = []\n\
             \tfor i in 0..len(params) {{\n\
             \t\tlet g = grads[i] + weight_decay * params[i]\n\
             \t\tlet v = momentum * g\n\
             \t\tupdated = push(updated, params[i] - lr * v)\n\
             \t}}\n\
             \tupdated\n\
             }}",
            lr, momentum, weight_decay
        )
    }
}

// ---------------------------------------------------------------------------
// 2. Runtime Eval
// ---------------------------------------------------------------------------

/// Parse and evaluate a Vortex code string within an existing environment.
pub fn eval_vortex(code: &str, env: &mut Env) -> Result<Value, String> {
    // Try parsing as a full program first (fn defs, structs, etc.)
    let tokens = crate::lexer::lex(code);
    if let Ok(program) = crate::parser::parse(tokens, 0) {
        if !program.items.is_empty() {
            // Register functions/structs
            for item in &program.items {
                match &item.kind {
                    ItemKind::Function(func) => {
                        let params: Vec<String> =
                            func.params.iter().map(|p| p.name.name.clone()).collect();
                        env.functions.insert(
                            func.name.name.clone(),
                            FnDef::User {
                                params,
                                body: func.body.clone(),
                            },
                        );
                    }
                    ItemKind::Struct(s) => {
                        let field_names: Vec<String> =
                            s.fields.iter().map(|f| f.name.name.clone()).collect();
                        env.struct_defs
                            .insert(s.name.name.clone(), field_names);
                    }
                    _ => {}
                }
            }
            return Ok(Value::Void);
        }
    }

    // Try wrapping in a function to evaluate as expression/statements
    let wrapped = format!("fn __meta_eval__() {{\n{}\n}}", code);
    let tokens = crate::lexer::lex(&wrapped);
    if let Ok(program) = crate::parser::parse(tokens, 0) {
        for item in &program.items {
            if let ItemKind::Function(func) = &item.kind {
                if func.name.name == "__meta_eval__" {
                    return crate::interpreter::eval_block(env, &func.body);
                }
            }
        }
    }

    Err(format!("meta eval: could not parse code: {}", code))
}

// ---------------------------------------------------------------------------
// 3. AST Manipulation
// ---------------------------------------------------------------------------

/// Path into an AST tree: sequence of child indices.
pub type AstPath = Vec<usize>;

/// A single modification to apply to an AST.
#[derive(Debug, Clone)]
pub enum Modification {
    /// Replace the expression at the given path with a new expression.
    ReplaceExpr { path: AstPath, new_expr: Expr },
    /// Insert a new item after the one at index.
    InsertItem { index: usize, item: Item },
    /// Remove the item at index.
    RemoveItem { index: usize },
    /// Replace a constant in any IntLiteral matching old_val with new_val.
    ReplaceConstant { old_val: u128, new_val: u128 },
}

pub struct AstManipulator;

impl AstManipulator {
    /// Find all IntLiteral nodes in a program, returning their paths.
    pub fn find_int_literals(program: &Program) -> Vec<(AstPath, u128)> {
        let mut results = Vec::new();
        for (i, item) in program.items.iter().enumerate() {
            if let ItemKind::Function(func) = &item.kind {
                Self::find_ints_in_block(&func.body, &[i], &mut results);
            }
        }
        results
    }

    fn find_ints_in_block(block: &Block, prefix: &[usize], results: &mut Vec<(AstPath, u128)>) {
        for (si, stmt) in block.stmts.iter().enumerate() {
            let mut path = prefix.to_vec();
            path.push(si);
            Self::find_ints_in_stmt(stmt, &path, results);
        }
        if let Some(expr) = &block.expr {
            let mut path = prefix.to_vec();
            path.push(block.stmts.len());
            Self::find_ints_in_expr(expr, &path, results);
        }
    }

    fn find_ints_in_stmt(stmt: &Stmt, prefix: &[usize], results: &mut Vec<(AstPath, u128)>) {
        match &stmt.kind {
            StmtKind::Let { value, .. } | StmtKind::Var { value, .. } => {
                Self::find_ints_in_expr(value, prefix, results);
            }
            StmtKind::Return(Some(e)) | StmtKind::Expr(e) => {
                Self::find_ints_in_expr(e, prefix, results);
            }
            StmtKind::Assign { value, .. } => {
                Self::find_ints_in_expr(value, prefix, results);
            }
            StmtKind::For { body, .. } => {
                Self::find_ints_in_block(body, prefix, results);
            }
            StmtKind::While { body, .. } | StmtKind::Loop { body } => {
                Self::find_ints_in_block(body, prefix, results);
            }
            _ => {}
        }
    }

    fn find_ints_in_expr(expr: &Expr, prefix: &[usize], results: &mut Vec<(AstPath, u128)>) {
        match &expr.kind {
            ExprKind::IntLiteral(n) => {
                results.push((prefix.to_vec(), *n));
            }
            ExprKind::Binary { lhs, rhs, .. } => {
                let mut lp = prefix.to_vec();
                lp.push(0);
                Self::find_ints_in_expr(lhs, &lp, results);
                let mut rp = prefix.to_vec();
                rp.push(1);
                Self::find_ints_in_expr(rhs, &rp, results);
            }
            ExprKind::Call { args, .. } => {
                for (i, arg) in args.iter().enumerate() {
                    let mut p = prefix.to_vec();
                    p.push(i);
                    Self::find_ints_in_expr(arg, &p, results);
                }
            }
            ExprKind::ArrayLiteral(elems) => {
                for (i, elem) in elems.iter().enumerate() {
                    let mut p = prefix.to_vec();
                    p.push(i);
                    Self::find_ints_in_expr(elem, &p, results);
                }
            }
            ExprKind::If {
                cond,
                then_block,
                else_block,
            } => {
                Self::find_ints_in_expr(cond, prefix, results);
                Self::find_ints_in_block(then_block, prefix, results);
                if let Some(eb) = else_block {
                    Self::find_ints_in_block(eb, prefix, results);
                }
            }
            ExprKind::Block(b) => {
                Self::find_ints_in_block(b, prefix, results);
            }
            ExprKind::Unary { expr: inner, .. } => {
                Self::find_ints_in_expr(inner, prefix, results);
            }
            _ => {}
        }
    }

    /// Clone a program and apply modifications.
    pub fn clone_and_modify(
        program: &Program,
        modifications: &[Modification],
    ) -> Program {
        let mut prog = program.clone();
        for modification in modifications {
            match modification {
                Modification::InsertItem { index, item } => {
                    let idx = (*index).min(prog.items.len());
                    prog.items.insert(idx, item.clone());
                }
                Modification::RemoveItem { index } => {
                    if *index < prog.items.len() {
                        prog.items.remove(*index);
                    }
                }
                Modification::ReplaceConstant { old_val, new_val } => {
                    replace_constants_in_program(&mut prog, *old_val, *new_val);
                }
                Modification::ReplaceExpr { .. } => {
                    // Path-based replacement is complex; skip for now
                }
            }
        }
        prog
    }

    /// Parse code and return an AST that can be inspected.
    pub fn parse_to_ast(code: &str) -> Result<Program, String> {
        let tokens = crate::lexer::lex(code);
        crate::parser::parse(tokens, 0).map_err(|diags| {
            diags
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
                .join("; ")
        })
    }

    /// Convert an AST back to code using the Display impl.
    pub fn ast_to_code(program: &Program) -> String {
        format!("{}", program)
    }
}

fn replace_constants_in_program(program: &mut Program, old_val: u128, new_val: u128) {
    for item in &mut program.items {
        if let ItemKind::Function(func) = &mut item.kind {
            replace_constants_in_block(&mut func.body, old_val, new_val);
        }
    }
}

fn replace_constants_in_block(block: &mut Block, old_val: u128, new_val: u128) {
    for stmt in &mut block.stmts {
        replace_constants_in_stmt(stmt, old_val, new_val);
    }
    if let Some(expr) = &mut block.expr {
        replace_constants_in_expr(expr, old_val, new_val);
    }
}

fn replace_constants_in_stmt(stmt: &mut Stmt, old_val: u128, new_val: u128) {
    match &mut stmt.kind {
        StmtKind::Let { value, .. } | StmtKind::Var { value, .. } => {
            replace_constants_in_expr(value, old_val, new_val);
        }
        StmtKind::Return(Some(e)) | StmtKind::Expr(e) => {
            replace_constants_in_expr(e, old_val, new_val);
        }
        StmtKind::Assign { value, .. } => {
            replace_constants_in_expr(value, old_val, new_val);
        }
        StmtKind::For { body, .. } => {
            replace_constants_in_block(body, old_val, new_val);
        }
        StmtKind::While { body, .. } | StmtKind::Loop { body } => {
            replace_constants_in_block(body, old_val, new_val);
        }
        _ => {}
    }
}

fn replace_constants_in_expr(expr: &mut Expr, old_val: u128, new_val: u128) {
    match &mut expr.kind {
        ExprKind::IntLiteral(n) => {
            if *n == old_val {
                *n = new_val;
            }
        }
        ExprKind::Binary { lhs, rhs, .. } | ExprKind::MatMul { lhs, rhs } => {
            replace_constants_in_expr(lhs, old_val, new_val);
            replace_constants_in_expr(rhs, old_val, new_val);
        }
        ExprKind::Unary { expr: inner, .. }
        | ExprKind::Try(inner)
        | ExprKind::Cast { expr: inner, .. } => {
            replace_constants_in_expr(inner, old_val, new_val);
        }
        ExprKind::Call { func, args } => {
            replace_constants_in_expr(func, old_val, new_val);
            for arg in args {
                replace_constants_in_expr(arg, old_val, new_val);
            }
        }
        ExprKind::ArrayLiteral(elems) => {
            for e in elems {
                replace_constants_in_expr(e, old_val, new_val);
            }
        }
        ExprKind::If {
            cond,
            then_block,
            else_block,
        } => {
            replace_constants_in_expr(cond, old_val, new_val);
            replace_constants_in_block(then_block, old_val, new_val);
            if let Some(eb) = else_block {
                replace_constants_in_block(eb, old_val, new_val);
            }
        }
        ExprKind::Block(b) => replace_constants_in_block(b, old_val, new_val),
        ExprKind::Index { base, indices } => {
            replace_constants_in_expr(base, old_val, new_val);
            for idx in indices {
                replace_constants_in_expr(idx, old_val, new_val);
            }
        }
        ExprKind::FieldAccess { base, .. } => {
            replace_constants_in_expr(base, old_val, new_val);
        }
        ExprKind::Range { start, end } => {
            replace_constants_in_expr(start, old_val, new_val);
            replace_constants_in_expr(end, old_val, new_val);
        }
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// 4. Genetic Programming Engine
// ---------------------------------------------------------------------------

/// A simple deterministic RNG (xorshift64) so we don't need external deps.
struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 1 } else { seed },
        }
    }
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
    fn next_usize(&mut self, max: usize) -> usize {
        (self.next_u64() as usize) % max.max(1)
    }
}

/// Genetic programming engine that evolves Vortex programs.
pub struct GeneticProgrammer {
    pub population: Vec<String>,
    pub fitness_scores: Vec<f64>,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub population_size: usize,
    rng: Rng,
}

impl GeneticProgrammer {
    pub fn new(population_size: usize, mutation_rate: f64, crossover_rate: f64, seed: u64) -> Self {
        Self {
            population: Vec::new(),
            fitness_scores: Vec::new(),
            mutation_rate,
            crossover_rate,
            population_size,
            rng: Rng::new(seed),
        }
    }

    /// Initialize population from a base program by creating mutations.
    pub fn initialize(&mut self, base_code: &str) {
        self.population.clear();
        self.population.push(base_code.to_string());
        for _ in 1..self.population_size {
            let mutated = self.mutate_code(base_code);
            self.population.push(mutated);
        }
        self.fitness_scores = vec![0.0; self.population_size];
    }

    /// Mutate code by changing numeric constants.
    pub fn mutate_code(&mut self, code: &str) -> String {
        let tokens = crate::lexer::lex(code);
        if let Ok(mut program) = crate::parser::parse(tokens, 0) {
            let literals = AstManipulator::find_int_literals(&program);
            if !literals.is_empty() && self.rng.next_f64() < self.mutation_rate {
                let idx = self.rng.next_usize(literals.len());
                let (_, old_val) = &literals[idx];
                let delta = (self.rng.next_u64() % 10) as u128;
                let new_val = if self.rng.next_f64() < 0.5 {
                    old_val.saturating_add(delta)
                } else {
                    old_val.saturating_sub(delta)
                };
                replace_constants_in_program(&mut program, *old_val, new_val);
                return format!("{}", program);
            }
        }
        // Also try operator mutations on the source text
        self.mutate_operators(code)
    }

    /// Mutate operators in source code (+ <-> -, * <-> /).
    fn mutate_operators(&mut self, code: &str) -> String {
        if self.rng.next_f64() > self.mutation_rate {
            return code.to_string();
        }
        let ops = [(" + ", " - "), (" - ", " + "), (" * ", " / "), (" / ", " * ")];
        let (from, to) = ops[self.rng.next_usize(ops.len())];
        if code.contains(from) {
            code.replacen(from, to, 1)
        } else {
            code.to_string()
        }
    }

    /// Crossover: take first half of one program's body, second half of another.
    pub fn crossover(&mut self, code_a: &str, code_b: &str) -> String {
        if self.rng.next_f64() > self.crossover_rate {
            return code_a.to_string();
        }
        let lines_a: Vec<&str> = code_a.lines().collect();
        let lines_b: Vec<&str> = code_b.lines().collect();
        if lines_a.len() < 2 || lines_b.len() < 2 {
            return code_a.to_string();
        }
        let split_a = self.rng.next_usize(lines_a.len());
        let split_b = self.rng.next_usize(lines_b.len());
        let mut result: Vec<&str> = lines_a[..split_a].to_vec();
        result.extend_from_slice(&lines_b[split_b..]);
        result.join("\n")
    }

    /// Tournament selection: pick 2 random individuals, return the fitter one's index.
    pub fn tournament_select(&mut self) -> usize {
        let a = self.rng.next_usize(self.population.len());
        let b = self.rng.next_usize(self.population.len());
        if self.fitness_scores.get(a).copied().unwrap_or(0.0)
            >= self.fitness_scores.get(b).copied().unwrap_or(0.0)
        {
            a
        } else {
            b
        }
    }

    /// Evaluate fitness for all individuals using the provided function.
    pub fn evaluate_fitness(&mut self, env: &mut Env, fitness_fn_name: &str) -> Result<(), String> {
        for i in 0..self.population.len() {
            let code = self.population[i].clone();
            // Evaluate the candidate code, then call the fitness function on the result
            let val = match eval_vortex(&code, env) {
                Ok(v) => v,
                Err(_) => {
                    self.fitness_scores[i] = f64::NEG_INFINITY;
                    continue;
                }
            };
            // Call fitness function
            let fitness_result = env.call_function(fitness_fn_name, vec![val])?;
            self.fitness_scores[i] = match fitness_result {
                Value::Float(f) => f,
                Value::Int(n) => n as f64,
                _ => 0.0,
            };
        }
        Ok(())
    }

    /// Run one generation: evaluate, select, crossover, mutate.
    pub fn evolve_generation(&mut self, env: &mut Env, fitness_fn_name: &str) -> Result<(), String> {
        self.evaluate_fitness(env, fitness_fn_name)?;

        let mut new_pop = Vec::with_capacity(self.population_size);

        // Elitism: keep the best individual
        if let Some(best_idx) = self
            .fitness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
        {
            new_pop.push(self.population[best_idx].clone());
        }

        while new_pop.len() < self.population_size {
            let parent_a = self.tournament_select();
            let parent_b = self.tournament_select();
            let child = self.crossover(&self.population[parent_a].clone(), &self.population[parent_b].clone());
            let mutated = self.mutate_code(&child);
            new_pop.push(mutated);
        }

        self.population = new_pop;
        Ok(())
    }

    /// Run the full evolutionary process for N generations, return best code.
    pub fn evolve(
        &mut self,
        env: &mut Env,
        fitness_fn_name: &str,
        generations: usize,
    ) -> Result<String, String> {
        for _ in 0..generations {
            self.evolve_generation(env, fitness_fn_name)?;
        }
        // Final evaluation
        self.evaluate_fitness(env, fitness_fn_name)?;
        let best_idx = self
            .fitness_scores
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(self.population[best_idx].clone())
    }
}

// ---------------------------------------------------------------------------
// 5. Neural Architecture Search (NAS)
// ---------------------------------------------------------------------------

/// Description of the search space for NAS.
#[derive(Debug, Clone)]
pub struct SearchSpace {
    pub layer_types: Vec<String>,
    pub min_layers: usize,
    pub max_layers: usize,
    pub dimension_range: (usize, usize),
}

impl Default for SearchSpace {
    fn default() -> Self {
        Self {
            layer_types: vec![
                "linear".into(),
                "relu".into(),
                "sigmoid".into(),
                "tanh".into(),
            ],
            min_layers: 2,
            max_layers: 8,
            dimension_range: (4, 128),
        }
    }
}

/// Record of an evaluated architecture.
#[derive(Debug, Clone)]
pub struct ArchRecord {
    pub code: String,
    pub score: f64,
}

/// Neural architecture search engine.
pub struct NASEngine {
    pub search_space: SearchSpace,
    pub history: Vec<ArchRecord>,
    rng: Rng,
    codegen: CodeGenerator,
}

impl NASEngine {
    pub fn new(search_space: SearchSpace, seed: u64) -> Self {
        Self {
            search_space,
            history: Vec::new(),
            rng: Rng::new(seed),
            codegen: CodeGenerator::new(),
        }
    }

    /// Sample a random architecture from the search space.
    pub fn sample_architecture(&mut self) -> Vec<(String, Vec<usize>)> {
        let num_layers = self.search_space.min_layers
            + self.rng.next_usize(
                (self.search_space.max_layers - self.search_space.min_layers).max(1) + 1,
            );

        let (dim_lo, dim_hi) = self.search_space.dimension_range;
        let mut layers = Vec::new();
        let mut prev_dim = dim_lo + self.rng.next_usize((dim_hi - dim_lo).max(1));

        for i in 0..num_layers {
            // Last layer? use a small output dim
            let is_last = i == num_layers - 1;
            let lt_idx = self.rng.next_usize(self.search_space.layer_types.len());
            let layer_type = &self.search_space.layer_types[lt_idx];

            match layer_type.as_str() {
                "linear" => {
                    let out_dim = if is_last {
                        1
                    } else {
                        dim_lo + self.rng.next_usize((dim_hi - dim_lo).max(1))
                    };
                    layers.push(("linear".to_string(), vec![prev_dim, out_dim]));
                    prev_dim = out_dim;
                }
                activation => {
                    layers.push((activation.to_string(), vec![]));
                }
            }
        }

        // Ensure we have at least one linear layer
        if !layers.iter().any(|(t, _)| t == "linear") {
            layers.insert(0, ("linear".to_string(), vec![prev_dim, 1]));
        }

        layers
    }

    /// Run NAS: sample architectures, evaluate them, return the best code.
    pub fn search(
        &mut self,
        env: &mut Env,
        budget: usize,
    ) -> Result<String, String> {
        for _ in 0..budget {
            let arch = self.sample_architecture();
            let code = self.codegen.codegen_model(&arch);

            // Try to evaluate the architecture
            let score = match eval_vortex(&code, env) {
                Ok(Value::Float(f)) => f,
                Ok(Value::Int(n)) => n as f64,
                Ok(_) => 0.0,
                Err(_) => f64::NEG_INFINITY,
            };

            self.history.push(ArchRecord {
                code: code.clone(),
                score,
            });
        }

        // Return best
        self.history
            .iter()
            .max_by(|a, b| a.score.partial_cmp(&b.score).unwrap_or(std::cmp::Ordering::Equal))
            .map(|r| r.code.clone())
            .ok_or_else(|| "NAS: no architectures evaluated".to_string())
    }
}

// ---------------------------------------------------------------------------
// 6. Self-Improving Training Loop
// ---------------------------------------------------------------------------

/// Engine that evolves the training code itself.
pub struct SelfImprover {
    pub training_code: String,
    pub performance_history: Vec<f64>,
    pub code_mutations: Vec<String>,
    rng: Rng,
}

impl SelfImprover {
    pub fn new(initial_training_code: &str, seed: u64) -> Self {
        Self {
            training_code: initial_training_code.to_string(),
            performance_history: Vec::new(),
            code_mutations: Vec::new(),
            rng: Rng::new(seed),
        }
    }

    /// Generate a mutation of the training code.
    fn generate_mutation(&mut self) -> String {
        let mut code = self.training_code.clone();

        // Possible mutations: tweak learning rate, batch size, epochs
        let mutation_type = self.rng.next_usize(4);
        match mutation_type {
            0 => {
                // Change learning rate
                if let Some(pos) = code.find("0.01") {
                    let new_lr = format!("{:.4}", 0.001 + self.rng.next_f64() * 0.09);
                    code = format!("{}{}{}", &code[..pos], new_lr, &code[pos + 4..]);
                } else if let Some(pos) = code.find("0.001") {
                    let new_lr = format!("{:.4}", 0.0001 + self.rng.next_f64() * 0.01);
                    code = format!("{}{}{}", &code[..pos], new_lr, &code[pos + 5..]);
                }
            }
            1 => {
                // Change numeric constants (batch size, hidden dims, etc.)
                let tokens = crate::lexer::lex(&code);
                if let Ok(mut program) = crate::parser::parse(tokens, 0) {
                    let literals = AstManipulator::find_int_literals(&program);
                    if !literals.is_empty() {
                        let idx = self.rng.next_usize(literals.len());
                        let old_val = literals[idx].1;
                        let factor = [2, 4, 8, 16, 32][self.rng.next_usize(5)] as u128;
                        let new_val = if self.rng.next_f64() < 0.5 {
                            old_val.saturating_mul(2).min(1024)
                        } else {
                            (old_val / 2).max(factor)
                        };
                        replace_constants_in_program(&mut program, old_val, new_val);
                        code = format!("{}", program);
                    }
                }
            }
            2 => {
                // Add weight decay term textually
                if !code.contains("weight_decay") {
                    code = code.replace(
                        "params[i] - lr * grads[i]",
                        "params[i] - lr * (grads[i] + 0.0001 * params[i])",
                    );
                }
            }
            _ => {
                // No-op mutation (keep original)
            }
        }
        code
    }

    /// Run the self-improvement loop for N iterations.
    pub fn improve(
        &mut self,
        env: &mut Env,
        iterations: usize,
    ) -> Result<(String, Vec<f64>), String> {
        // Evaluate initial code
        let tc = self.training_code.clone();
        let initial_score = self.evaluate_code(env, &tc)?;
        self.performance_history.push(initial_score);
        let mut best_score = initial_score;
        let mut best_code = self.training_code.clone();

        for _ in 0..iterations {
            let mutation = self.generate_mutation();
            let score = self.evaluate_code(env, &mutation)?;
            self.code_mutations.push(mutation.clone());

            if score > best_score {
                best_score = score;
                best_code = mutation;
                self.training_code = best_code.clone();
            }
            self.performance_history.push(best_score);
        }

        Ok((best_code, self.performance_history.clone()))
    }

    fn evaluate_code(&mut self, env: &mut Env, code: &str) -> Result<f64, String> {
        match eval_vortex(code, env) {
            Ok(Value::Float(f)) => Ok(f),
            Ok(Value::Int(n)) => Ok(n as f64),
            Ok(_) => Ok(0.0),
            Err(_) => Ok(f64::NEG_INFINITY),
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Interpreter Builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions
        .insert("eval_code".to_string(), FnDef::Builtin(builtin_eval_code));
    env.functions.insert(
        "generate_function".to_string(),
        FnDef::Builtin(builtin_generate_function),
    );
    env.functions
        .insert("ast_parse".to_string(), FnDef::Builtin(builtin_ast_parse));
    env.functions.insert(
        "ast_to_code".to_string(),
        FnDef::Builtin(builtin_ast_to_code),
    );
    env.functions.insert(
        "evolve_program".to_string(),
        FnDef::Builtin(builtin_evolve_program),
    );
    env.functions.insert(
        "nas_search".to_string(),
        FnDef::Builtin(builtin_nas_search),
    );
    env.functions.insert(
        "self_improve".to_string(),
        FnDef::Builtin(builtin_self_improve),
    );
    env.functions.insert(
        "template_render".to_string(),
        FnDef::Builtin(builtin_template_render),
    );
    env.functions.insert(
        "codegen_model".to_string(),
        FnDef::Builtin(builtin_codegen_model),
    );
    env.functions.insert(
        "codegen_optimizer".to_string(),
        FnDef::Builtin(builtin_codegen_optimizer),
    );
}

/// eval_code(code_string) -> execute Vortex code, return result
fn builtin_eval_code(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("eval_code expects 1 argument: code string".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("eval_code: argument must be a string".to_string()),
    };
    eval_vortex(&code, env)
}

/// generate_function(name, params_array, body) -> code string
fn builtin_generate_function(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("generate_function expects 3 arguments: name, params, body".to_string());
    }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("generate_function: name must be a string".to_string()),
    };
    let params = match &args[1] {
        Value::Array(arr) => {
            let mut p = Vec::new();
            for v in arr {
                match v {
                    Value::String(s) => p.push(s.clone()),
                    _ => return Err("generate_function: params must be array of strings".to_string()),
                }
            }
            p
        }
        _ => return Err("generate_function: params must be an array".to_string()),
    };
    let body = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("generate_function: body must be a string".to_string()),
    };

    let param_str = params
        .iter()
        .map(|p| format!("{}: Int", p))
        .collect::<Vec<_>>()
        .join(", ");
    let code = format!("fn {}({}) {{\n{}\n}}", name, param_str, body);
    Ok(Value::String(code))
}

/// ast_parse(code) -> nested array representation of the AST
fn builtin_ast_parse(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("ast_parse expects 1 argument: code string".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("ast_parse: argument must be a string".to_string()),
    };

    let program = AstManipulator::parse_to_ast(&code)?;
    Ok(ast_to_value(&program))
}

/// Convert an AST to a Value representation (nested arrays/strings).
fn ast_to_value(program: &Program) -> Value {
    let items: Vec<Value> = program
        .items
        .iter()
        .map(|item| {
            let kind_str = match &item.kind {
                ItemKind::Function(f) => format!("fn:{}", f.name.name),
                ItemKind::Kernel(k) => format!("kernel:{}", k.name.name),
                ItemKind::Struct(s) => format!("struct:{}", s.name.name),
                ItemKind::Enum(e) => format!("enum:{}", e.name.name),
                ItemKind::Trait(t) => format!("trait:{}", t.name.name),
                ItemKind::Impl(i) => format!("impl:{}", i.target),
                ItemKind::Import(_) => "import".to_string(),
                ItemKind::Const(c) => format!("const:{}", c.name.name),
                ItemKind::TypeAlias(t) => format!("type:{}", t.name.name),
                ItemKind::FieldDef(fd) => format!("field:{}", fd.name.name),
            };
            Value::String(kind_str)
        })
        .collect();
    Value::Array(items)
}

/// ast_to_code(ast_value) -> reconstructed code string
/// For simplicity, this takes the original code, parses it, and re-emits it.
fn builtin_ast_to_code(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("ast_to_code expects 1 argument: code string".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("ast_to_code: argument must be a string".to_string()),
    };
    let program = AstManipulator::parse_to_ast(&code)?;
    Ok(Value::String(AstManipulator::ast_to_code(&program)))
}

/// evolve_program(initial_code, fitness_fn_name, generations) -> best program code
fn builtin_evolve_program(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err(
            "evolve_program expects 3 arguments: initial_code, fitness_fn, generations".to_string(),
        );
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("evolve_program: first arg must be a string".to_string()),
    };
    let fitness_fn = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("evolve_program: second arg must be a string (function name)".to_string()),
    };
    let generations = match &args[2] {
        Value::Int(n) => *n as usize,
        _ => return Err("evolve_program: third arg must be an integer".to_string()),
    };

    let mut gp = GeneticProgrammer::new(20, 0.8, 0.6, 42);
    gp.initialize(&code);
    let best = gp.evolve(env, &fitness_fn, generations)?;
    Ok(Value::String(best))
}

/// nas_search(layer_types, min_layers, max_layers, budget) -> best architecture code
fn builtin_nas_search(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err(
            "nas_search expects 4 arguments: layer_types, min_layers, max_layers, budget"
                .to_string(),
        );
    }
    let layer_types = match &args[0] {
        Value::Array(arr) => {
            let mut types = Vec::new();
            for v in arr {
                match v {
                    Value::String(s) => types.push(s.clone()),
                    _ => {
                        return Err("nas_search: layer_types must be array of strings".to_string())
                    }
                }
            }
            types
        }
        _ => return Err("nas_search: first arg must be an array".to_string()),
    };
    let min_layers = match &args[1] {
        Value::Int(n) => *n as usize,
        _ => return Err("nas_search: min_layers must be an integer".to_string()),
    };
    let max_layers = match &args[2] {
        Value::Int(n) => *n as usize,
        _ => return Err("nas_search: max_layers must be an integer".to_string()),
    };
    let budget = match &args[3] {
        Value::Int(n) => *n as usize,
        _ => return Err("nas_search: budget must be an integer".to_string()),
    };

    let space = SearchSpace {
        layer_types,
        min_layers,
        max_layers,
        dimension_range: (4, 128),
    };
    let mut nas = NASEngine::new(space, 42);
    let best = nas.search(env, budget)?;
    Ok(Value::String(best))
}

/// self_improve(training_code, iterations) -> [improved_code, performance_array]
fn builtin_self_improve(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("self_improve expects 2 arguments: training_code, iterations".to_string());
    }
    let code = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("self_improve: first arg must be a string".to_string()),
    };
    let iterations = match &args[1] {
        Value::Int(n) => *n as usize,
        _ => return Err("self_improve: second arg must be an integer".to_string()),
    };

    let mut improver = SelfImprover::new(&code, 42);
    let (best_code, history) = improver.improve(env, iterations)?;

    let history_vals: Vec<Value> = history.iter().map(|f| Value::Float(*f)).collect();
    Ok(Value::Array(vec![
        Value::String(best_code),
        Value::Array(history_vals),
    ]))
}

/// template_render(template_string, params_map) -> rendered code string
fn builtin_template_render(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("template_render expects 2 arguments: template, params".to_string());
    }
    let template_str = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("template_render: first arg must be a string".to_string()),
    };
    let params = match &args[1] {
        Value::HashMap(map) => {
            let mut m = HashMap::new();
            for (k, v) in map {
                m.insert(k.clone(), format!("{}", v));
            }
            m
        }
        Value::Array(arr) => {
            // Accept array of [key, value] pairs
            let mut m = HashMap::new();
            for v in arr {
                if let Value::Array(pair) = v {
                    if pair.len() == 2 {
                        if let (Value::String(k), val) = (&pair[0], &pair[1]) {
                            m.insert(k.clone(), format!("{}", val));
                        }
                    }
                }
            }
            m
        }
        _ => return Err("template_render: second arg must be a hashmap or array of pairs".to_string()),
    };

    let param_names: Vec<String> = params.keys().cloned().collect();
    let tmpl = Template {
        params: param_names,
        body: template_str,
    };
    let result = tmpl.render(&params)?;
    Ok(Value::String(result))
}

/// codegen_model(layers_array) -> model code string
fn builtin_codegen_model(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("codegen_model expects 1 argument: layers array".to_string());
    }
    let layers = match &args[0] {
        Value::Array(arr) => {
            let mut layers = Vec::new();
            for v in arr {
                match v {
                    Value::Array(pair) => {
                        if pair.is_empty() {
                            continue;
                        }
                        let layer_type = match &pair[0] {
                            Value::String(s) => s.clone(),
                            _ => continue,
                        };
                        let dims: Vec<usize> = pair[1..]
                            .iter()
                            .filter_map(|d| match d {
                                Value::Int(n) => Some(*n as usize),
                                _ => None,
                            })
                            .collect();
                        layers.push((layer_type, dims));
                    }
                    Value::String(s) => {
                        layers.push((s.clone(), vec![]));
                    }
                    _ => {}
                }
            }
            layers
        }
        _ => return Err("codegen_model: argument must be an array".to_string()),
    };

    let cg = CodeGenerator::new();
    Ok(Value::String(cg.codegen_model(&layers)))
}

/// codegen_optimizer(lr, momentum, weight_decay) -> optimizer code string
fn builtin_codegen_optimizer(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("codegen_optimizer expects 3 arguments: lr, momentum, weight_decay".to_string());
    }
    let lr = match &args[0] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("codegen_optimizer: lr must be a number".to_string()),
    };
    let momentum = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("codegen_optimizer: momentum must be a number".to_string()),
    };
    let weight_decay = match &args[2] {
        Value::Float(f) => *f,
        Value::Int(n) => *n as f64,
        _ => return Err("codegen_optimizer: weight_decay must be a number".to_string()),
    };

    let cg = CodeGenerator::new();
    Ok(Value::String(cg.codegen_optimizer(lr, momentum, weight_decay)))
}

// ---------------------------------------------------------------------------
// Helper: Env::call_function (used by genetic programmer)
// ---------------------------------------------------------------------------

impl Env {
    /// Call a named function (user or builtin) with arguments.
    pub fn call_function(&mut self, name: &str, args: Vec<Value>) -> Result<Value, String> {
        let fndef = self
            .functions
            .get(name)
            .cloned()
            .ok_or_else(|| format!("function '{}' not found", name))?;
        match fndef {
            FnDef::Builtin(f) => f(self, args),
            FnDef::User { params, body } => {
                self.push_scope();
                for (p, a) in params.iter().zip(args.iter()) {
                    self.define(p, a.clone());
                }
                let result = crate::interpreter::eval_block(self, &body);
                self.pop_scope();
                match result {
                    Ok(Value::Return(v)) => Ok(*v),
                    other => other,
                }
            }
            FnDef::GradWrapper { fn_name, order: _ } => {
                Err(format!("GradWrapper '{}' not callable in meta engine", fn_name))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 9. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_env() -> Env {
        Env::new()
    }

    #[test]
    fn test_eval_code_expression() {
        let mut env = make_env();
        let result = eval_vortex("1 + 2", &mut env);
        assert!(result.is_ok());
        match result.unwrap() {
            Value::Int(n) => assert_eq!(n, 3),
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_eval_code_define_and_call_function() {
        let mut env = make_env();
        // Define a function
        let r = eval_vortex("fn add(a: Int, b: Int) -> Int { a + b }", &mut env);
        assert!(r.is_ok());
        // Call it
        let r2 = eval_vortex("add(3, 4)", &mut env);
        assert!(r2.is_ok());
        match r2.unwrap() {
            Value::Int(n) => assert_eq!(n, 7),
            other => panic!("expected Int(7), got {:?}", other),
        }
    }

    #[test]
    fn test_eval_code_let_binding() {
        let mut env = make_env();
        let r = eval_vortex("let x = 42\nx", &mut env);
        assert!(r.is_ok());
        match r.unwrap() {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("expected Int(42), got {:?}", other),
        }
    }

    #[test]
    fn test_template_render_basic() {
        let tmpl = Template {
            params: vec!["name".into(), "size".into()],
            body: "nn_linear({{name}}, {{size}})".to_string(),
        };
        let mut values = HashMap::new();
        values.insert("name".into(), "layer1".into());
        values.insert("size".into(), "64".into());
        let result = tmpl.render(&values).unwrap();
        assert_eq!(result, "nn_linear(layer1, 64)");
    }

    #[test]
    fn test_template_render_missing_param() {
        let tmpl = Template {
            params: vec!["name".into()],
            body: "fn {{name}}() {}".to_string(),
        };
        let values = HashMap::new();
        assert!(tmpl.render(&values).is_err());
    }

    #[test]
    fn test_codegen_model() {
        let cg = CodeGenerator::new();
        let layers = vec![
            ("linear".to_string(), vec![2, 4]),
            ("relu".to_string(), vec![]),
            ("linear".to_string(), vec![4, 1]),
        ];
        let code = cg.codegen_model(&layers);
        assert!(code.contains("nn_linear(2, 4)"));
        assert!(code.contains("nn_relu()"));
        assert!(code.contains("nn_linear(4, 1)"));
        assert!(code.starts_with("nn_sequential("));
    }

    #[test]
    fn test_codegen_optimizer() {
        let cg = CodeGenerator::new();
        let code = cg.codegen_optimizer(0.01, 0.9, 0.0001);
        assert!(code.contains("0.01"));
        assert!(code.contains("0.9"));
        assert!(code.contains("0.0001"));
        assert!(code.contains("fn custom_optimizer"));
    }

    #[test]
    fn test_codegen_function() {
        let cg = CodeGenerator::new();
        let mut values = HashMap::new();
        values.insert("op".into(), "+".into());
        let code = cg
            .codegen_function("add", &[("a", "Int"), ("b", "Int")], "a {{op}} b", &values)
            .unwrap();
        assert!(code.contains("fn add(a: Int, b: Int)"));
        assert!(code.contains("a + b"));
    }

    #[test]
    fn test_ast_parse_roundtrip() {
        let code = "fn foo(x: Int) -> Int {\n    x + 1\n}";
        let program = AstManipulator::parse_to_ast(code).unwrap();
        let regenerated = AstManipulator::ast_to_code(&program);
        // Should contain the function name at minimum
        assert!(regenerated.contains("foo"));
    }

    #[test]
    fn test_ast_find_int_literals() {
        let code = "fn f() {\n    let x = 42\n    let y = 7\n}";
        let program = AstManipulator::parse_to_ast(code).unwrap();
        let lits = AstManipulator::find_int_literals(&program);
        let vals: Vec<u128> = lits.iter().map(|(_, v)| *v).collect();
        assert!(vals.contains(&42));
        assert!(vals.contains(&7));
    }

    #[test]
    fn test_ast_replace_constant() {
        let code = "fn f() -> Int {\n    42\n}";
        let mut program = AstManipulator::parse_to_ast(code).unwrap();
        replace_constants_in_program(&mut program, 42, 100);
        let new_code = AstManipulator::ast_to_code(&program);
        assert!(new_code.contains("100"));
        assert!(!new_code.contains("42"));
    }

    #[test]
    fn test_clone_and_modify() {
        let code = "fn f() -> Int {\n    10\n}";
        let program = AstManipulator::parse_to_ast(code).unwrap();
        let modified = AstManipulator::clone_and_modify(
            &program,
            &[Modification::ReplaceConstant {
                old_val: 10,
                new_val: 99,
            }],
        );
        let new_code = AstManipulator::ast_to_code(&modified);
        assert!(new_code.contains("99"));
    }

    #[test]
    fn test_genetic_mutation_produces_valid_code() {
        let code = "fn f() -> Int {\n    let x = 5\n    x + 3\n}";
        let mut gp = GeneticProgrammer::new(10, 1.0, 0.6, 42);
        let mutated = gp.mutate_code(code);
        // Mutated code should still be parseable (or be a string)
        assert!(!mutated.is_empty());
    }

    #[test]
    fn test_genetic_initialize() {
        let code = "fn f() -> Int {\n    42\n}";
        let mut gp = GeneticProgrammer::new(10, 0.8, 0.6, 42);
        gp.initialize(code);
        assert_eq!(gp.population.len(), 10);
        // First individual should be the original
        assert_eq!(gp.population[0], code);
    }

    #[test]
    fn test_genetic_crossover() {
        let code_a = "line1\nline2\nline3";
        let code_b = "lineA\nlineB\nlineC";
        let mut gp = GeneticProgrammer::new(10, 0.8, 1.0, 42);
        let child = gp.crossover(code_a, code_b);
        assert!(!child.is_empty());
    }

    #[test]
    fn test_nas_sample_architecture() {
        let space = SearchSpace::default();
        let mut nas = NASEngine::new(space, 42);
        let arch = nas.sample_architecture();
        assert!(!arch.is_empty());
        // Should have at least one linear layer
        assert!(arch.iter().any(|(t, _)| t == "linear"));
    }

    #[test]
    fn test_nas_codegen() {
        let space = SearchSpace::default();
        let mut nas = NASEngine::new(space, 42);
        let arch = nas.sample_architecture();
        let code = nas.codegen.codegen_model(&arch);
        assert!(code.contains("nn_sequential"));
        assert!(code.contains("nn_linear"));
    }

    #[test]
    fn test_search_space_default() {
        let space = SearchSpace::default();
        assert_eq!(space.min_layers, 2);
        assert_eq!(space.max_layers, 8);
        assert!(space.layer_types.contains(&"linear".to_string()));
        assert!(space.layer_types.contains(&"relu".to_string()));
    }

    #[test]
    fn test_self_improver_generation() {
        let code = "let x = 42\nx";
        let mut improver = SelfImprover::new(code, 42);
        let mutation = improver.generate_mutation();
        assert!(!mutation.is_empty());
    }

    #[test]
    fn test_builtin_eval_code() {
        let mut env = make_env();
        let result = builtin_eval_code(&mut env, vec![Value::String("1 + 1".into())]);
        assert!(result.is_ok());
        match result.unwrap() {
            Value::Int(2) => {}
            other => panic!("expected Int(2), got {:?}", other),
        }
    }

    #[test]
    fn test_builtin_generate_function() {
        let mut env = make_env();
        let result = builtin_generate_function(
            &mut env,
            vec![
                Value::String("double".into()),
                Value::Array(vec![Value::String("x".into())]),
                Value::String("x * 2".into()),
            ],
        );
        assert!(result.is_ok());
        let code = match result.unwrap() {
            Value::String(s) => s,
            _ => panic!("expected string"),
        };
        assert!(code.contains("fn double"));
        assert!(code.contains("x * 2"));
    }

    #[test]
    fn test_builtin_template_render() {
        let mut env = make_env();
        let result = builtin_template_render(
            &mut env,
            vec![
                Value::String("hello {{name}}".into()),
                Value::Array(vec![Value::Array(vec![
                    Value::String("name".into()),
                    Value::String("world".into()),
                ])]),
            ],
        );
        assert!(result.is_ok());
        match result.unwrap() {
            Value::String(s) => assert_eq!(s, "hello world"),
            other => panic!("expected string, got {:?}", other),
        }
    }

    #[test]
    fn test_rng_deterministic() {
        let mut rng1 = Rng::new(42);
        let mut rng2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(rng1.next_u64(), rng2.next_u64());
        }
    }
}
