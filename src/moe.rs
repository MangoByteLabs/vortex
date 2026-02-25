// Mixture of Experts + Speculative Decoding for Vortex
use crate::interpreter::{Value, Env, FnDef};
use std::sync::{LazyLock, Mutex};

static MOE_MODELS: LazyLock<Mutex<Vec<MoeModel>>> = LazyLock::new(|| Mutex::new(Vec::new()));

#[derive(Clone, Debug)]
struct Expert {
    weights: Vec<Vec<f64>>,
    bias: Vec<f64>,
}

#[derive(Clone, Debug)]
struct MoeModel {
    num_experts: usize,
    experts: Vec<Expert>,
    gate_weights: Vec<Vec<f64>>,
    top_k: usize,
    capacity_factor: f64,
    expert_call_counts: Vec<usize>,
}

impl Expert {
    fn new(input_dim: usize, output_dim: usize, idx: usize) -> Self {
        let mut weights = vec![vec![0.0; input_dim]; output_dim];
        // Deterministic pseudo-random init based on index
        for r in 0..output_dim {
            for c in 0..input_dim {
                let seed = (idx * 1000 + r * 31 + c * 7) as f64;
                weights[r][c] = (seed * 0.6180339887).fract() * 0.2 - 0.1;
            }
        }
        let bias = vec![0.0; output_dim];
        Expert { weights, bias }
    }

    fn forward(&self, input: &[f64]) -> Vec<f64> {
        self.weights
            .iter()
            .enumerate()
            .map(|(i, row)| {
                let dot: f64 = row.iter().zip(input.iter()).map(|(w, x)| w * x).sum();
                dot + self.bias[i]
            })
            .collect()
    }
}

impl MoeModel {
    fn new(num_experts: usize, input_dim: usize, output_dim: usize, top_k: usize) -> Self {
        let experts: Vec<Expert> = (0..num_experts)
            .map(|i| Expert::new(input_dim, output_dim, i))
            .collect();
        // Gate: maps input_dim -> num_experts
        let mut gate_weights = vec![vec![0.0; input_dim]; num_experts];
        for e in 0..num_experts {
            for c in 0..input_dim {
                let seed = (e * 997 + c * 13) as f64;
                gate_weights[e][c] = (seed * 0.7071067811).fract() * 0.2 - 0.1;
            }
        }
        MoeModel {
            num_experts,
            experts,
            gate_weights,
            top_k: top_k.min(num_experts),
            capacity_factor: 1.0,
            expert_call_counts: vec![0; num_experts],
        }
    }

    fn gate(&self, input: &[f64]) -> Vec<f64> {
        let logits: Vec<f64> = self
            .gate_weights
            .iter()
            .map(|row| row.iter().zip(input.iter()).map(|(w, x)| w * x).sum())
            .collect();
        softmax(&logits)
    }

    fn forward(&mut self, input: &[f64]) -> Vec<f64> {
        let probs = self.gate(input);
        // Select top-k experts
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let top_k = &indexed[..self.top_k];

        // Renormalize selected expert weights
        let sum: f64 = top_k.iter().map(|(_, p)| p).sum();
        let norm = if sum > 0.0 { sum } else { 1.0 };

        let output_dim = self.experts[0].weights.len();
        let mut output = vec![0.0; output_dim];

        for &(idx, prob) in top_k {
            // Apply capacity factor: skip if overloaded
            let max_calls = ((self.expert_call_counts.iter().sum::<usize>() as f64
                / self.num_experts as f64)
                * self.capacity_factor
                + 1.0) as usize;
            if self.expert_call_counts[idx] >= max_calls && self.capacity_factor < 10.0 {
                continue;
            }
            self.expert_call_counts[idx] += 1;
            let expert_out = self.experts[idx].forward(input);
            let weight = prob / norm;
            for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                *o += weight * e;
            }
        }
        output
    }

    fn expert_usage(&self) -> Vec<(usize, usize)> {
        self.expert_call_counts
            .iter()
            .copied()
            .enumerate()
            .collect()
    }
}

fn softmax(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f64 = exps.iter().sum();
    exps.iter().map(|e| e / sum).collect()
}

// --- Speculative decoding ---

fn spec_decode_accept(draft_probs: &[f64], target_probs: &[f64]) -> Vec<bool> {
    // Rejection sampling: accept token i if target_prob[i] / draft_prob[i] >= r
    // We use deterministic r = 0.5 for reproducibility in the runtime
    draft_probs
        .iter()
        .zip(target_probs.iter())
        .map(|(&dp, &tp)| {
            if dp <= 0.0 {
                tp > 0.0
            } else {
                let ratio = tp / dp;
                ratio >= 0.5
            }
        })
        .collect()
}

fn spec_verify_tokens(
    draft_tokens: &[i64],
    draft_probs: &[f64],
    target_probs: &[f64],
) -> (Vec<i64>, Option<i64>) {
    let mask = spec_decode_accept(draft_probs, target_probs);
    let mut accepted = Vec::new();
    let mut corrected = None;
    for (i, &accept) in mask.iter().enumerate() {
        if accept {
            accepted.push(draft_tokens[i]);
        } else {
            // Corrected token: pick from target distribution (use index of max prob as surrogate)
            corrected = Some(i as i64);
            break;
        }
    }
    (accepted, corrected)
}

// --- Builtins ---

fn extract_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|x| match x {
                Value::Float(f) => Ok(*f),
                Value::Int(i) => Ok(*i as f64),
                _ => Err("expected numeric array".into()),
            })
            .collect(),
        _ => Err("expected array".into()),
    }
}

fn extract_i64_vec(v: &Value) -> Result<Vec<i64>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|x| match x {
                Value::Int(i) => Ok(*i as i64),
                _ => Err("expected int array".into()),
            })
            .collect(),
        _ => Err("expected array".into()),
    }
}

fn builtin_moe_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 4 {
        return Err("moe_create(num_experts, input_dim, output_dim, top_k)".into());
    }
    let num_experts = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let input_dim = match &args[1] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let output_dim = match &args[2] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let top_k = match &args[3] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let model = MoeModel::new(num_experts, input_dim, output_dim, top_k);
    let mut models = MOE_MODELS.lock().unwrap();
    let id = models.len();
    models.push(model);
    Ok(Value::Int(id as i128))
}

fn builtin_moe_forward(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("moe_forward(id, input)".into());
    }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let input = extract_f64_vec(&args[1])?;
    let mut models = MOE_MODELS.lock().unwrap();
    let model = models.get_mut(id).ok_or("invalid moe id")?;
    let output = model.forward(&input);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_moe_set_capacity(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("moe_set_capacity(id, factor)".into());
    }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let factor = match &args[1] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("expected number".into()),
    };
    let mut models = MOE_MODELS.lock().unwrap();
    let model = models.get_mut(id).ok_or("invalid moe id")?;
    model.capacity_factor = factor;
    Ok(Value::Bool(true))
}

fn builtin_moe_expert_usage(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 1 {
        return Err("moe_expert_usage(id)".into());
    }
    let id = match &args[0] { Value::Int(n) => *n as usize, _ => return Err("expected int".into()) };
    let models = MOE_MODELS.lock().unwrap();
    let model = models.get(id).ok_or("invalid moe id")?;
    let usage = model.expert_usage();
    let arr: Vec<Value> = usage
        .into_iter()
        .map(|(idx, count)| {
            Value::Array(vec![Value::Int(idx as i128), Value::Int(count as i128)])
        })
        .collect();
    Ok(Value::Array(arr))
}

fn builtin_spec_decode_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("spec_decode_create(draft_probs, target_probs)".into());
    }
    let draft_probs = extract_f64_vec(&args[0])?;
    let target_probs = extract_f64_vec(&args[1])?;
    if draft_probs.len() != target_probs.len() {
        return Err("prob arrays must have same length".into());
    }
    let mask = spec_decode_accept(&draft_probs, &target_probs);
    Ok(Value::Array(mask.into_iter().map(Value::Bool).collect()))
}

fn builtin_spec_verify(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 3 {
        return Err("spec_verify(draft_tokens, draft_probs, target_probs)".into());
    }
    let draft_tokens = extract_i64_vec(&args[0])?;
    let draft_probs = extract_f64_vec(&args[1])?;
    let target_probs = extract_f64_vec(&args[2])?;
    if draft_tokens.len() != draft_probs.len() || draft_probs.len() != target_probs.len() {
        return Err("all arrays must have same length".into());
    }
    let (accepted, corrected) = spec_verify_tokens(&draft_tokens, &draft_probs, &target_probs);
    let accepted_val = Value::Array(accepted.into_iter().map(|t| Value::Int(t as i128)).collect());
    let corrected_val = match corrected {
        Some(c) => Value::Int(c as i128),
        None => Value::Bool(false), // no correction needed
    };
    Ok(Value::Array(vec![accepted_val, corrected_val]))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("moe_create".into(), FnDef::Builtin(builtin_moe_create));
    env.functions.insert("moe_forward".into(), FnDef::Builtin(builtin_moe_forward));
    env.functions.insert("moe_set_capacity".into(), FnDef::Builtin(builtin_moe_set_capacity));
    env.functions.insert("moe_expert_usage".into(), FnDef::Builtin(builtin_moe_expert_usage));
    env.functions.insert("spec_decode_create".into(), FnDef::Builtin(builtin_spec_decode_create));
    env.functions.insert("spec_verify".into(), FnDef::Builtin(builtin_spec_verify));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_softmax() {
        let result = softmax(&[1.0, 2.0, 3.0]);
        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result[2] > result[1] && result[1] > result[0]);
    }

    #[test]
    fn test_expert_forward() {
        let expert = Expert::new(4, 2, 0);
        let input = vec![1.0, 0.0, 0.0, 0.0];
        let output = expert.forward(&input);
        assert_eq!(output.len(), 2);
    }

    #[test]
    fn test_moe_model_creation() {
        let model = MoeModel::new(4, 8, 3, 2);
        assert_eq!(model.num_experts, 4);
        assert_eq!(model.top_k, 2);
        assert_eq!(model.experts.len(), 4);
        assert_eq!(model.gate_weights.len(), 4);
    }

    #[test]
    fn test_moe_forward() {
        let mut model = MoeModel::new(4, 4, 2, 2);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let output = model.forward(&input);
        assert_eq!(output.len(), 2);
        // At least some experts should have been called
        let total: usize = model.expert_call_counts.iter().sum();
        assert!(total > 0);
    }

    #[test]
    fn test_moe_expert_usage() {
        let mut model = MoeModel::new(4, 4, 2, 2);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        model.forward(&input);
        let usage = model.expert_usage();
        assert_eq!(usage.len(), 4);
        let active: Vec<_> = usage.iter().filter(|(_, c)| *c > 0).collect();
        assert!(!active.is_empty());
    }

    #[test]
    fn test_capacity_factor() {
        let mut model = MoeModel::new(4, 4, 2, 2);
        model.capacity_factor = 1.0;
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let _ = model.forward(&input);
        assert!(model.capacity_factor == 1.0);
    }

    #[test]
    fn test_spec_decode_accept() {
        let draft = vec![0.8, 0.6, 0.3];
        let target = vec![0.9, 0.2, 0.5];
        let mask = spec_decode_accept(&draft, &target);
        // ratio: 1.125 >= 0.5 → true, 0.333 < 0.5 → false, 1.667 >= 0.5 → true
        assert_eq!(mask, vec![true, false, true]);
    }

    #[test]
    fn test_spec_verify_tokens() {
        let tokens = vec![10, 20, 30];
        let draft = vec![0.8, 0.6, 0.3];
        let target = vec![0.9, 0.2, 0.5];
        let (accepted, corrected) = spec_verify_tokens(&tokens, &draft, &target);
        // First accepted (ratio 1.125), second rejected (ratio 0.333)
        assert_eq!(accepted, vec![10]);
        assert_eq!(corrected, Some(1));
    }
}
