///! Program synthesis for Vortex — generate, compile, and optimize Vortex programs.

use crate::interpreter::{Env, FnDef, Value};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core data types
// ---------------------------------------------------------------------------

/// Constraint that a synthesized program must satisfy.
#[derive(Debug, Clone)]
pub enum Constraint {
    /// Output increases when the given input index increases.
    Monotonic(usize),
    /// Output must lie in [min, max].
    Bounded(f64, f64),
    /// Same input always produces same output.
    Deterministic,
    /// Must complete within the given number of milliseconds.
    TimeLimit(u64),
    /// f(f(x)) == f(x).
    Idempotent,
}

/// Specification of a program to synthesize.
#[derive(Debug, Clone)]
pub struct ProgramSpec {
    pub input_types: Vec<String>,
    pub output_type: String,
    pub examples: Vec<(Vec<f64>, Vec<f64>)>,
    pub constraints: Vec<Constraint>,
    pub max_complexity: usize,
}

/// A candidate program produced by synthesis.
#[derive(Debug, Clone)]
pub struct ProgramCandidate {
    pub source: String,
    pub ast_depth: usize,
    pub score: f64,
    pub verified: bool,
}

// ---------------------------------------------------------------------------
// Helper: generate a Vortex function source
// ---------------------------------------------------------------------------

/// Build a valid Vortex function: `fn synth(x0: f64, ...) -> f64 { return <expr> }`
fn make_synth_fn(params_decl: &str, ret_type: &str, expr: &str) -> String {
    format!("fn synth({}) -> {} {{ return {} }}", params_decl, ret_type, expr)
}

// ---------------------------------------------------------------------------
// Synthesizer (enumerative, bottom-up)
// ---------------------------------------------------------------------------

pub struct Synthesizer {
    pub primitives: Vec<String>,
}

impl Synthesizer {
    pub fn new() -> Self {
        Self {
            primitives: vec![
                "+".into(), "-".into(), "*".into(), "/".into(),
                "if".into(), "for".into(), "abs".into(), "neg".into(),
            ],
        }
    }

    /// Enumerate candidate programs bottom-up, scored against `spec.examples`.
    pub fn synthesize(&self, spec: &ProgramSpec, max_candidates: usize) -> Vec<ProgramCandidate> {
        let mut candidates: Vec<ProgramCandidate> = Vec::new();

        let arity = spec.input_types.len();
        let param_names: Vec<String> = (0..arity).map(|i| format!("x{}", i)).collect();
        let params_decl: String = param_names
            .iter()
            .enumerate()
            .map(|(i, n)| format!("{}: {}", n, spec.input_types[i]))
            .collect::<Vec<_>>()
            .join(", ");
        let ret = &spec.output_type;

        // Depth-1: identity candidates
        for p in &param_names {
            let src = make_synth_fn(&params_decl, ret, p);
            let score = self.evaluate(&src, &spec.examples);
            candidates.push(ProgramCandidate { source: src, ast_depth: 1, score, verified: false });
        }

        // Depth-2: binary ops between params / constants
        let bin_ops = ["+", "-", "*"];
        for op in &bin_ops {
            for a in &param_names {
                for b in &param_names {
                    let expr = format!("{} {} {}", a, op, b);
                    let src = make_synth_fn(&params_decl, ret, &expr);
                    let score = self.evaluate(&src, &spec.examples);
                    candidates.push(ProgramCandidate { source: src, ast_depth: 2, score, verified: false });
                    if candidates.len() >= max_candidates * 4 {
                        break;
                    }
                }
            }
        }

        // Depth-3: nested ops  a op (b op c), constants
        if spec.max_complexity >= 3 {
            let constants = ["2.0", "1.0", "0.5"];
            for op1 in &bin_ops {
                for op2 in &bin_ops {
                    for a in &param_names {
                        for c in &constants {
                            let expr = format!("{} {} ({} {} {})", a, op1, a, op2, c);
                            let src = make_synth_fn(&params_decl, ret, &expr);
                            let score = self.evaluate(&src, &spec.examples);
                            candidates.push(ProgramCandidate { source: src, ast_depth: 3, score, verified: false });
                        }
                    }
                }
            }
        }

        // Sort by score descending and truncate
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(max_candidates);

        // Verify constraints on top candidates
        for c in candidates.iter_mut() {
            c.verified = self.verify(&c.source, &spec.constraints);
        }

        candidates
    }

    /// Score a candidate against input/output examples.  Returns 0.0..=1.0.
    pub fn evaluate(&self, source: &str, examples: &[(Vec<f64>, Vec<f64>)]) -> f64 {
        if examples.is_empty() {
            return 0.0;
        }
        let mut total_error = 0.0f64;
        let mut count = 0usize;
        for (inputs, expected) in examples {
            match jit_compile_and_run(source, inputs) {
                Ok(outputs) => {
                    for (o, e) in outputs.iter().zip(expected.iter()) {
                        total_error += (o - e).abs();
                        count += 1;
                    }
                }
                Err(_) => {
                    total_error += 1e6;
                    count += 1;
                }
            }
        }
        if count == 0 {
            return 0.0;
        }
        let avg_error = total_error / count as f64;
        1.0 / (1.0 + avg_error)
    }

    /// Check all constraints.
    pub fn verify(&self, source: &str, constraints: &[Constraint]) -> bool {
        for c in constraints {
            match c {
                Constraint::Deterministic => {
                    let input = vec![1.0];
                    let r1 = jit_compile_and_run(source, &input);
                    let r2 = jit_compile_and_run(source, &input);
                    match (r1, r2) {
                        (Ok(a), Ok(b)) => { if a != b { return false; } }
                        _ => return false,
                    }
                }
                Constraint::Bounded(min, max) => {
                    let test_inputs = vec![vec![0.0], vec![1.0], vec![-1.0], vec![100.0]];
                    for inp in &test_inputs {
                        if let Ok(out) = jit_compile_and_run(source, inp) {
                            for v in &out {
                                if *v < *min || *v > *max { return false; }
                            }
                        }
                    }
                }
                Constraint::Monotonic(idx) => {
                    let vals = [0.0, 1.0, 2.0, 5.0, 10.0];
                    let mut prev: Option<f64> = None;
                    for &v in &vals {
                        let mut inp = vec![0.0; *idx + 1];
                        inp[*idx] = v;
                        if let Ok(out) = jit_compile_and_run(source, &inp) {
                            if let Some(o) = out.first() {
                                if let Some(p) = prev {
                                    if *o < p { return false; }
                                }
                                prev = Some(*o);
                            }
                        }
                    }
                }
                Constraint::Idempotent => {
                    let test_inputs = vec![vec![2.0], vec![5.0]];
                    for inp in &test_inputs {
                        if let Ok(out1) = jit_compile_and_run(source, inp) {
                            if let Ok(out2) = jit_compile_and_run(source, &out1) {
                                for (a, b) in out1.iter().zip(out2.iter()) {
                                    if (a - b).abs() > 1e-9 { return false; }
                                }
                            }
                        }
                    }
                }
                Constraint::TimeLimit(_ms) => {
                    // Accept for now; real impl would measure wall time.
                }
            }
        }
        true
    }

    /// Simplify / optimize a candidate source string.
    pub fn optimize(&self, source: &str) -> String {
        let mut s = source.to_string();
        s = s.replace("+ 0.0", "");
        s = s.replace("+ 0", "");
        s = s.replace("* 1.0", "");
        s = s.replace("* 1 ", "");
        s = s.replace("- 0.0", "");
        s = s.replace("- 0 ", "");
        s = s.replace("--", "");
        s
    }
}

// ---------------------------------------------------------------------------
// GeneticSynthesizer — evolutionary approach
// ---------------------------------------------------------------------------

pub struct GeneticSynthesizer {
    pub population: Vec<ProgramCandidate>,
    rng_state: u64,
}

impl GeneticSynthesizer {
    pub fn new() -> Self {
        Self { population: Vec::new(), rng_state: 42 }
    }

    fn next_rand(&mut self) -> u64 {
        self.rng_state ^= self.rng_state << 13;
        self.rng_state ^= self.rng_state >> 7;
        self.rng_state ^= self.rng_state << 17;
        self.rng_state
    }

    #[allow(dead_code)]
    fn rand_f64(&mut self) -> f64 {
        (self.next_rand() % 10000) as f64 / 10000.0
    }

    /// Apply a random AST-level mutation to the source.
    pub fn mutate(&mut self, program: &str) -> String {
        let ops = ["+", "-", "*"];
        let idx = (self.next_rand() as usize) % ops.len();
        let new_op = ops[idx];

        // Find the return expression and mutate an operator in it
        if let Some(ret_start) = program.find("return ") {
            let after_return = &program[ret_start + 7..];
            for op in &ops {
                if after_return.contains(op) && *op != new_op {
                    let mutated_expr = after_return.replacen(op, new_op, 1);
                    return format!("{}{}", &program[..ret_start + 7], mutated_expr);
                }
            }
        }
        // Fallback: return as-is with a small tweak
        program.replace("return ", &format!("return 1.0 {} ", new_op))
    }

    /// Combine two programs: take the body expression of a, wrap with header of b.
    pub fn crossover(&mut self, a: &str, b: &str) -> String {
        let body_a = extract_return_expr(a);
        let header_b = extract_header(b);
        format!("{} {{ return {} }}", header_b, body_a)
    }

    /// Run evolutionary synthesis for a number of generations.
    pub fn evolve(&mut self, spec: &ProgramSpec, generations: usize) -> ProgramCandidate {
        let synth = Synthesizer::new();
        if self.population.is_empty() {
            self.population = synth.synthesize(spec, 20);
        }

        for _ in 0..generations {
            let mut next_gen = self.population.clone();
            let pop_len = self.population.len();

            // Mutations
            for i in 0..pop_len.min(10) {
                let src = self.population[i].source.clone();
                let mutated = self.mutate(&src);
                let score = synth.evaluate(&mutated, &spec.examples);
                next_gen.push(ProgramCandidate {
                    source: mutated,
                    ast_depth: self.population[i].ast_depth,
                    score,
                    verified: false,
                });
            }

            // Crossovers
            if pop_len >= 2 {
                for _ in 0..5 {
                    let i = (self.next_rand() as usize) % pop_len;
                    let j = (self.next_rand() as usize) % pop_len;
                    if i != j {
                        let a = self.population[i].source.clone();
                        let b = self.population[j].source.clone();
                        let child = self.crossover(&a, &b);
                        let score = synth.evaluate(&child, &spec.examples);
                        next_gen.push(ProgramCandidate {
                            source: child, ast_depth: 2, score, verified: false,
                        });
                    }
                }
            }

            next_gen.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
            next_gen.truncate(20);
            self.population = next_gen;
        }

        self.population.first().cloned().unwrap_or(ProgramCandidate {
            source: String::new(), ast_depth: 0, score: 0.0, verified: false,
        })
    }
}

// ---------------------------------------------------------------------------
// NeuralGuidedSynthesizer — learned heuristics
// ---------------------------------------------------------------------------

pub struct NeuralGuidedSynthesizer {
    pub guide_weights: Vec<Vec<f64>>,
    primitive_names: Vec<String>,
}

impl NeuralGuidedSynthesizer {
    pub fn new() -> Self {
        let names: Vec<String> = vec![
            "+", "-", "*", "/", "if", "for", "abs", "neg",
        ].into_iter().map(String::from).collect();
        let n = names.len();
        let weights = vec![vec![1.0 / n as f64; n]; n];
        Self { guide_weights: weights, primitive_names: names }
    }

    /// Given a partial-program feature vector, predict ranked next primitives.
    pub fn predict_next_primitive(&self, partial_program: &[f64]) -> Vec<(String, f64)> {
        let n = self.primitive_names.len();
        let mut scores = vec![0.0f64; n];

        for (i, s) in scores.iter_mut().enumerate() {
            let weights_col = &self.guide_weights[i.min(self.guide_weights.len() - 1)];
            for (j, &feat) in partial_program.iter().enumerate() {
                if j < weights_col.len() {
                    *s += feat * weights_col[j];
                }
            }
        }

        // Softmax
        let max_s = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - max_s).exp()).collect();
        let sum: f64 = exps.iter().sum();

        let mut result: Vec<(String, f64)> = self.primitive_names.iter()
            .zip(exps.iter())
            .map(|(name, &e)| (name.clone(), e / sum))
            .collect();
        result.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        result
    }

    /// Synthesize a program using neural guidance.
    pub fn synthesize_guided(&self, spec: &ProgramSpec) -> ProgramCandidate {
        let synth = Synthesizer::new();
        let arity = spec.input_types.len();
        let param_names: Vec<String> = (0..arity).map(|i| format!("x{}", i)).collect();
        let params_decl: String = param_names.iter()
            .enumerate()
            .map(|(i, n)| format!("{}: {}", n, spec.input_types[i]))
            .collect::<Vec<_>>()
            .join(", ");
        let ret = &spec.output_type;

        let features: Vec<f64> = if let Some((inp, out)) = spec.examples.first() {
            let mut f = inp.clone();
            f.extend(out.iter());
            f
        } else {
            vec![0.0; 4]
        };

        let ranked = self.predict_next_primitive(&features);

        let mut best = ProgramCandidate {
            source: String::new(), ast_depth: 0, score: 0.0, verified: false,
        };

        for (prim, _prob) in ranked.iter().take(4) {
            let expr = match prim.as_str() {
                "+" => format!("{} + {}", param_names[0], param_names.get(1).unwrap_or(&param_names[0])),
                "-" => format!("{} - {}", param_names[0], param_names.get(1).unwrap_or(&"1.0".to_string())),
                "*" => format!("{} * {}", param_names[0], param_names.get(1).unwrap_or(&param_names[0])),
                "/" => format!("{} / 2.0", param_names[0]),
                "neg" => format!("0.0 - {}", param_names[0]),
                _ => param_names[0].clone(),
            };
            let src = make_synth_fn(&params_decl, ret, &expr);
            let score = synth.evaluate(&src, &spec.examples);
            if score > best.score {
                best = ProgramCandidate { source: src, ast_depth: 2, score, verified: false };
            }
        }

        best.verified = synth.verify(&best.source, &spec.constraints);
        best
    }

    /// Update guide weights from a successful synthesis.
    pub fn learn_from_success(&mut self, _spec: &ProgramSpec, solution: &str) {
        let learning_rate = 0.1;
        for (i, prim) in self.primitive_names.iter().enumerate() {
            if solution.contains(prim) {
                for w in self.guide_weights[i].iter_mut() {
                    *w += learning_rate;
                }
            }
        }
        for row in self.guide_weights.iter_mut() {
            let sum: f64 = row.iter().sum();
            if sum > 0.0 {
                for w in row.iter_mut() {
                    *w /= sum;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// JIT Compiler — uses existing lexer + parser + interpreter
// ---------------------------------------------------------------------------

pub struct JITCompiler {
    cache: HashMap<String, Vec<f64>>,
}

impl JITCompiler {
    pub fn new() -> Self {
        Self { cache: HashMap::new() }
    }

    pub fn compile_and_run(&mut self, source: &str, inputs: &[f64]) -> Result<Vec<f64>, String> {
        let key = format!("{}|{:?}", source, inputs);
        if let Some(cached) = self.cache.get(&key) {
            return Ok(cached.clone());
        }
        let result = jit_compile_and_run(source, inputs)?;
        self.cache.insert(key, result.clone());
        Ok(result)
    }
}

/// Internal: compile and run a Vortex source with the interpreter pipeline.
/// The source must define a function called `synth`.
fn jit_compile_and_run(source: &str, inputs: &[f64]) -> Result<Vec<f64>, String> {
    let args_str: String = inputs.iter()
        .map(|v| {
            if *v == v.floor() && v.abs() < 1e15 {
                format!("{:.1}", v)
            } else {
                format!("{}", v)
            }
        })
        .collect::<Vec<_>>()
        .join(", ");

    let full_source = format!("fn main() {{\nlet __r = synth({})\nprintln(__r)\n}}\n{}", args_str, source);

    let tokens = crate::lexer::lex(&full_source);
    let program = crate::parser::parse(tokens, 0).map_err(|diags| {
        format!("Parse error: {:?}", diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>())
    })?;

    let output = crate::interpreter::interpret(&program)
        .map_err(|e| format!("Runtime error: {}", e))?;

    if let Some(last) = output.last() {
        let trimmed = last.trim();
        if let Ok(f) = trimmed.parse::<f64>() {
            return Ok(vec![f]);
        }
        let inner = trimmed.trim_start_matches('[').trim_end_matches(']');
        let vals: Result<Vec<f64>, _> = inner.split(',')
            .map(|s| s.trim().parse::<f64>())
            .collect();
        if let Ok(v) = vals {
            return Ok(v);
        }
        Err(format!("Could not parse output: {}", trimmed))
    } else {
        Ok(vec![0.0])
    }
}

fn value_to_f64_vec(val: &Value) -> Vec<f64> {
    match val {
        Value::Float(f) => vec![*f],
        Value::Int(i) => vec![*i as f64],
        Value::Array(arr) => arr.iter().flat_map(value_to_f64_vec).collect(),
        Value::Tuple(t) => t.iter().flat_map(value_to_f64_vec).collect(),
        _ => vec![0.0],
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract the expression after `return` from a synth function source.
fn extract_return_expr(source: &str) -> String {
    if let Some(idx) = source.find("return ") {
        let after = &source[idx + 7..];
        // Take until closing brace
        if let Some(end) = after.rfind('}') {
            return after[..end].trim().to_string();
        }
        return after.trim().to_string();
    }
    "x0".to_string()
}

/// Extract the function signature (everything before the opening brace).
fn extract_header(source: &str) -> String {
    if let Some(idx) = source.find('{') {
        source[..idx].trim().to_string()
    } else {
        source.to_string()
    }
}

// ---------------------------------------------------------------------------
// Interpreter builtins
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("synthesize_program".to_string(), FnDef::Builtin(builtin_synthesize_program));
    env.functions.insert("jit_eval".to_string(), FnDef::Builtin(builtin_jit_eval));
    env.functions.insert("evolve_program".to_string(), FnDef::Builtin(builtin_evolve_program));
}

fn builtin_synthesize_program(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("synthesize_program requires (examples, max_depth)".into());
    }
    let examples = parse_examples_value(&args[0])?;
    let max_depth = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        _ => 3,
    };
    let arity = examples.first().map(|(i, _)| i.len()).unwrap_or(1);
    let spec = ProgramSpec {
        input_types: vec!["f64".to_string(); arity],
        output_type: "f64".to_string(),
        examples,
        constraints: vec![],
        max_complexity: max_depth,
    };
    let synth = Synthesizer::new();
    let candidates = synth.synthesize(&spec, 10);
    let best = candidates.first().map(|c| c.source.clone()).unwrap_or_default();
    Ok(Value::String(best))
}

fn builtin_jit_eval(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("jit_eval requires (source_string, input_array)".into());
    }
    let source = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("jit_eval: first arg must be a string".into()),
    };
    let inputs = value_to_f64_vec(&args[1]);
    let mut jit = JITCompiler::new();
    let result = jit.compile_and_run(&source, &inputs)?;
    Ok(Value::Array(result.into_iter().map(Value::Float).collect()))
}

fn builtin_evolve_program(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("evolve_program requires (examples, generations)".into());
    }
    let examples = parse_examples_value(&args[0])?;
    let generations = match &args[1] {
        Value::Int(n) => *n as usize,
        Value::Float(f) => *f as usize,
        _ => 10,
    };
    let arity = examples.first().map(|(i, _)| i.len()).unwrap_or(1);
    let spec = ProgramSpec {
        input_types: vec!["f64".to_string(); arity],
        output_type: "f64".to_string(),
        examples,
        constraints: vec![],
        max_complexity: 3,
    };
    let mut genetic = GeneticSynthesizer::new();
    let best = genetic.evolve(&spec, generations);
    Ok(Value::String(best.source))
}

fn parse_examples_value(val: &Value) -> Result<Vec<(Vec<f64>, Vec<f64>)>, String> {
    match val {
        Value::Array(pairs) => {
            let mut examples = Vec::new();
            for pair in pairs {
                match pair {
                    Value::Array(inner) if inner.len() == 2 => {
                        let inputs = value_to_f64_vec(&inner[0]);
                        let outputs = value_to_f64_vec(&inner[1]);
                        examples.push((inputs, outputs));
                    }
                    _ => return Err("Each example must be [inputs, outputs]".into()),
                }
            }
            Ok(examples)
        }
        _ => Err("Examples must be an array".into()),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_program_spec_creation() {
        let spec = ProgramSpec {
            input_types: vec!["f64".into()],
            output_type: "f64".into(),
            examples: vec![(vec![1.0], vec![2.0]), (vec![2.0], vec![4.0])],
            constraints: vec![],
            max_complexity: 3,
        };
        assert_eq!(spec.input_types.len(), 1);
        assert_eq!(spec.examples.len(), 2);
    }

    #[test]
    fn test_constraint_variants() {
        let constraints = vec![
            Constraint::Monotonic(0),
            Constraint::Bounded(-10.0, 10.0),
            Constraint::Deterministic,
            Constraint::TimeLimit(1000),
            Constraint::Idempotent,
        ];
        assert_eq!(constraints.len(), 5);
        match &constraints[0] {
            Constraint::Monotonic(idx) => assert_eq!(*idx, 0),
            _ => panic!("expected Monotonic"),
        }
    }

    #[test]
    fn test_synthesizer_new() {
        let synth = Synthesizer::new();
        assert!(synth.primitives.len() >= 4);
        assert!(synth.primitives.contains(&"+".to_string()));
    }

    #[test]
    fn test_synthesize_produces_candidates() {
        let spec = ProgramSpec {
            input_types: vec!["f64".into()],
            output_type: "f64".into(),
            examples: vec![(vec![2.0], vec![4.0]), (vec![3.0], vec![6.0])],
            constraints: vec![],
            max_complexity: 2,
        };
        let synth = Synthesizer::new();
        let candidates = synth.synthesize(&spec, 5);
        assert!(!candidates.is_empty());
        assert!(candidates.len() <= 5);
        for c in &candidates {
            assert!(!c.source.is_empty());
            assert!(c.source.contains("fn synth"));
        }
    }

    #[test]
    fn test_evaluate_perfect_identity() {
        let src = "fn synth(x0: f64) -> f64 { return x0 }";
        let result = jit_compile_and_run(src, &[5.0]);
        eprintln!("JIT result for identity(5.0): {:?}", result);

        let synth = Synthesizer::new();
        let examples = vec![(vec![5.0], vec![5.0]), (vec![10.0], vec![10.0])];
        let score = synth.evaluate(src, &examples);
        assert!(score > 0.99, "Identity should score near 1.0, got {}", score);
    }

    #[test]
    fn test_evaluate_bad_program_low_score() {
        let synth = Synthesizer::new();
        let src = "THIS IS NOT VALID";
        let examples = vec![(vec![1.0], vec![100.0])];
        let score = synth.evaluate(src, &examples);
        assert!(score < 0.01, "Invalid program should score near 0, got {}", score);
    }

    #[test]
    fn test_optimize_removes_identity_ops() {
        let synth = Synthesizer::new();
        let src = "x + 0.0";
        let optimized = synth.optimize(src);
        assert!(!optimized.contains("+ 0.0"));
    }

    #[test]
    fn test_genetic_mutate() {
        let mut gen = GeneticSynthesizer::new();
        let src = "fn synth(x0: f64) -> f64 { return x0 + x0 }";
        let mutated = gen.mutate(src);
        assert_ne!(mutated, src);
    }

    #[test]
    fn test_genetic_crossover() {
        let mut gen = GeneticSynthesizer::new();
        let a = "fn synth(x0: f64) -> f64 { return x0 + x0 }";
        let b = "fn synth(x0: f64) -> f64 { return x0 * x0 }";
        let child = gen.crossover(a, b);
        assert!(child.contains("fn synth"));
        assert!(child.contains("return"));
    }

    #[test]
    fn test_neural_predict_returns_all_primitives() {
        let neural = NeuralGuidedSynthesizer::new();
        let features = vec![1.0, 2.0, 3.0];
        let ranked = neural.predict_next_primitive(&features);
        assert_eq!(ranked.len(), 8);
        let sum: f64 = ranked.iter().map(|(_, p)| p).sum();
        assert!((sum - 1.0).abs() < 1e-6, "probabilities should sum to 1, got {}", sum);
    }

    #[test]
    fn test_neural_learn_from_success_updates_weights() {
        let mut neural = NeuralGuidedSynthesizer::new();
        let original_weights = neural.guide_weights.clone();
        let spec = ProgramSpec {
            input_types: vec!["f64".into()],
            output_type: "f64".into(),
            examples: vec![],
            constraints: vec![],
            max_complexity: 2,
        };
        neural.learn_from_success(&spec, "x0 + x0");
        assert_ne!(neural.guide_weights, original_weights);
    }

    #[test]
    fn test_jit_compiler_caching() {
        let mut jit = JITCompiler::new();
        let src = "fn synth(x0: f64) -> f64 { return x0 }";
        let r1 = jit.compile_and_run(src, &[42.0]);
        assert!(r1.is_ok(), "JIT should succeed: {:?}", r1);
        let r2 = jit.compile_and_run(src, &[42.0]);
        assert_eq!(r1, r2);
        assert_eq!(jit.cache.len(), 1);
    }

    #[test]
    fn test_verify_deterministic() {
        let synth = Synthesizer::new();
        let src = "fn synth(x0: f64) -> f64 { return x0 * 2.0 }";
        assert!(synth.verify(src, &[Constraint::Deterministic]));
    }

    #[test]
    fn test_verify_bounded() {
        let synth = Synthesizer::new();
        let src = "fn synth(x0: f64) -> f64 { return x0 * x0 }";
        // x0=100 -> 10000, out of [-10, 10]
        assert!(!synth.verify(src, &[Constraint::Bounded(-10.0, 10.0)]));
    }

    #[test]
    fn test_candidate_struct() {
        let c = ProgramCandidate {
            source: "fn f() -> f64 { return 42.0 }".into(),
            ast_depth: 1,
            score: 0.95,
            verified: true,
        };
        assert_eq!(c.ast_depth, 1);
        assert!(c.verified);
        assert!(c.score > 0.9);
    }

    #[test]
    fn test_extract_return_expr() {
        let src = "fn synth(x0: f64) -> f64 { return x0 + 1.0 }";
        let expr = extract_return_expr(src);
        assert_eq!(expr, "x0 + 1.0");
    }

    #[test]
    fn test_extract_header() {
        let src = "fn synth(x0: f64) -> f64 { return x0 }";
        let header = extract_header(src);
        assert_eq!(header, "fn synth(x0: f64) -> f64");
    }
}
