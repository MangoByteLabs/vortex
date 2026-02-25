// rl_training.rs — PPO/DPO/GRPO reinforcement learning for Vortex

use crate::interpreter::{Value, Env, FnDef};
use std::sync::{LazyLock, Mutex};

// --- State ---

struct PPOTrainer {
    clip_epsilon: f64,
    gamma: f64,
    lambda: f64,
}

struct RewardModel {
    weights: Vec<f64>,
    bias: f64,
}

static PPO_TRAINERS: LazyLock<Mutex<Vec<PPOTrainer>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static REWARD_MODELS: LazyLock<Mutex<Vec<RewardModel>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// --- Helpers ---

fn extract_floats(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr.iter().map(|x| match x {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected numeric array".into()),
        }).collect(),
        _ => Err("expected array".into()),
    }
}

fn floats_to_value(v: Vec<f64>) -> Value {
    Value::Array(v.into_iter().map(Value::Float).collect())
}

// --- PPO ---

fn builtin_ppo_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ppo_create(clip_epsilon, gamma, lambda)".into()); }
    let clip_epsilon = match &args[0] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("clip_epsilon must be numeric".into()) };
    let gamma = match &args[1] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("gamma must be numeric".into()) };
    let lambda = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("lambda must be numeric".into()) };
    let mut trainers = PPO_TRAINERS.lock().unwrap();
    let id = trainers.len();
    trainers.push(PPOTrainer { clip_epsilon, gamma, lambda });
    Ok(Value::Int(id as i128))
}

fn builtin_ppo_compute_advantages(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("ppo_compute_advantages(id, rewards, values)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let rewards = extract_floats(&args[1])?;
    let values = extract_floats(&args[2])?;
    if rewards.len() != values.len() { return Err("rewards and values must have same length".into()); }
    let trainers = PPO_TRAINERS.lock().unwrap();
    let t = trainers.get(id).ok_or("invalid trainer id")?;
    let n = rewards.len();
    let mut advantages = vec![0.0; n];
    let mut gae = 0.0;
    for i in (0..n).rev() {
        let next_val = if i + 1 < n { values[i + 1] } else { 0.0 };
        let delta = rewards[i] + t.gamma * next_val - values[i];
        gae = delta + t.gamma * t.lambda * gae;
        advantages[i] = gae;
    }
    Ok(floats_to_value(advantages))
}

fn builtin_ppo_loss(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("ppo_loss(id, old_probs, new_probs, advantages)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let old_probs = extract_floats(&args[1])?;
    let new_probs = extract_floats(&args[2])?;
    let advantages = extract_floats(&args[3])?;
    let n = old_probs.len();
    if n != new_probs.len() || n != advantages.len() { return Err("arrays must have same length".into()); }
    let trainers = PPO_TRAINERS.lock().unwrap();
    let t = trainers.get(id).ok_or("invalid trainer id")?;
    let eps = t.clip_epsilon;
    let mut total = 0.0;
    for i in 0..n {
        let ratio = new_probs[i] / (old_probs[i] + 1e-8);
        let clipped = ratio.clamp(1.0 - eps, 1.0 + eps);
        let surr = (ratio * advantages[i]).min(clipped * advantages[i]);
        total += surr;
    }
    Ok(Value::Float(-total / n as f64))
}

// --- DPO ---

fn builtin_dpo_loss(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("dpo_loss(chosen_logps, rejected_logps, beta)".into()); }
    let chosen = extract_floats(&args[0])?;
    let rejected = extract_floats(&args[1])?;
    let beta = match &args[2] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta must be numeric".into()) };
    if chosen.len() != rejected.len() { return Err("arrays must have same length".into()); }
    let n = chosen.len();
    let mut total = 0.0;
    for i in 0..n {
        let diff = beta * (chosen[i] - rejected[i]);
        total += -(1.0 / (1.0 + (-diff).exp())).ln();
    }
    Ok(Value::Float(total / n as f64))
}

// --- GRPO ---

fn builtin_grpo_loss(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("grpo_loss(group_rewards, logprobs, ref_logprobs, beta)".into()); }
    let rewards = extract_floats(&args[0])?;
    let logprobs = extract_floats(&args[1])?;
    let ref_logprobs = extract_floats(&args[2])?;
    let beta = match &args[3] { Value::Float(f) => *f, Value::Int(i) => *i as f64, _ => return Err("beta must be numeric".into()) };
    let n = rewards.len();
    if n != logprobs.len() || n != ref_logprobs.len() { return Err("arrays must have same length".into()); }
    // normalize rewards within group
    let mean = rewards.iter().sum::<f64>() / n as f64;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-8);
    let norm: Vec<f64> = rewards.iter().map(|r| (r - mean) / std).collect();
    let mut total = 0.0;
    for i in 0..n {
        let kl = logprobs[i] - ref_logprobs[i];
        total += -(norm[i] * logprobs[i] - beta * kl);
    }
    Ok(Value::Float(total / n as f64))
}

// --- Reward Model ---

fn builtin_reward_create(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("reward_create(input_dim)".into()); }
    let dim = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("input_dim must be int".into()) };
    let mut models = REWARD_MODELS.lock().unwrap();
    let id = models.len();
    // Xavier-like init with deterministic seed
    let weights: Vec<f64> = (0..dim).map(|i| ((i as f64 * 0.7 + 0.3).sin()) / (dim as f64).sqrt()).collect();
    models.push(RewardModel { weights, bias: 0.0 });
    Ok(Value::Int(id as i128))
}

fn builtin_reward_score(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("reward_score(id, features)".into()); }
    let id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("id must be int".into()) };
    let features = extract_floats(&args[1])?;
    let models = REWARD_MODELS.lock().unwrap();
    let m = models.get(id).ok_or("invalid reward model id")?;
    if features.len() != m.weights.len() { return Err(format!("expected {} features, got {}", m.weights.len(), features.len())); }
    let score: f64 = features.iter().zip(m.weights.iter()).map(|(f, w)| f * w).sum::<f64>() + m.bias;
    Ok(Value::Float(score.tanh()))
}

// --- Utilities ---

fn builtin_rl_normalize_rewards(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("rl_normalize_rewards(rewards)".into()); }
    let rewards = extract_floats(&args[0])?;
    let n = rewards.len();
    if n == 0 { return Ok(Value::Array(vec![])); }
    let mean = rewards.iter().sum::<f64>() / n as f64;
    let var = rewards.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n as f64;
    let std = var.sqrt().max(1e-8);
    Ok(floats_to_value(rewards.iter().map(|r| (r - mean) / std).collect()))
}

fn builtin_kl_divergence(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("kl_divergence(p_logprobs, q_logprobs)".into()); }
    let p = extract_floats(&args[0])?;
    let q = extract_floats(&args[1])?;
    if p.len() != q.len() { return Err("arrays must have same length".into()); }
    let kl: f64 = p.iter().zip(q.iter()).map(|(pi, qi)| {
        let p_prob = pi.exp();
        p_prob * (pi - qi)
    }).sum();
    Ok(Value::Float(kl))
}

// --- Registration ---

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("ppo_create".into(), FnDef::Builtin(builtin_ppo_create));
    env.functions.insert("ppo_compute_advantages".into(), FnDef::Builtin(builtin_ppo_compute_advantages));
    env.functions.insert("ppo_loss".into(), FnDef::Builtin(builtin_ppo_loss));
    env.functions.insert("dpo_loss".into(), FnDef::Builtin(builtin_dpo_loss));
    env.functions.insert("grpo_loss".into(), FnDef::Builtin(builtin_grpo_loss));
    env.functions.insert("reward_create".into(), FnDef::Builtin(builtin_reward_create));
    env.functions.insert("reward_score".into(), FnDef::Builtin(builtin_reward_score));
    env.functions.insert("rl_normalize_rewards".into(), FnDef::Builtin(builtin_rl_normalize_rewards));
    env.functions.insert("kl_divergence".into(), FnDef::Builtin(builtin_kl_divergence));
}

// --- Tests ---

#[cfg(test)]
mod tests {
    use super::*;

    fn env() -> Env { Env::new() }
    fn f(v: f64) -> Value { Value::Float(v) }
    fn arr(v: Vec<f64>) -> Value { Value::Array(v.into_iter().map(Value::Float).collect()) }

    #[test]
    fn test_ppo_create() {
        let mut e = env();
        let id = builtin_ppo_create(&mut e, vec![f(0.2), f(0.99), f(0.95)]).unwrap();
        if let Value::Int(i) = id { assert!(i >= 0); } else { panic!("expected int"); }
    }

    #[test]
    fn test_ppo_advantages() {
        let mut e = env();
        builtin_ppo_create(&mut e, vec![f(0.2), f(0.99), f(0.95)]).unwrap();
        let id_val = Value::Int(PPO_TRAINERS.lock().unwrap().len() as i128 - 1);
        let result = builtin_ppo_compute_advantages(&mut e, vec![
            id_val, arr(vec![1.0, 1.0, 1.0]), arr(vec![0.5, 0.5, 0.5]),
        ]).unwrap();
        if let Value::Array(a) = result { assert_eq!(a.len(), 3); } else { panic!("expected array"); }
    }

    #[test]
    fn test_ppo_loss() {
        let mut e = env();
        builtin_ppo_create(&mut e, vec![f(0.2), f(0.99), f(0.95)]).unwrap();
        let id_val = Value::Int(PPO_TRAINERS.lock().unwrap().len() as i128 - 1);
        let loss = builtin_ppo_loss(&mut e, vec![
            id_val, arr(vec![0.5, 0.5]), arr(vec![0.6, 0.4]), arr(vec![1.0, -1.0]),
        ]).unwrap();
        if let Value::Float(v) = loss { assert!(v.is_finite()); } else { panic!("expected float"); }
    }

    #[test]
    fn test_dpo_loss() {
        let mut e = env();
        let loss = builtin_dpo_loss(&mut e, vec![
            arr(vec![-1.0, -2.0]), arr(vec![-3.0, -4.0]), f(0.1),
        ]).unwrap();
        if let Value::Float(v) = loss { assert!(v >= 0.0); } else { panic!("expected float"); }
    }

    #[test]
    fn test_grpo_loss() {
        let mut e = env();
        let loss = builtin_grpo_loss(&mut e, vec![
            arr(vec![1.0, 0.0, -1.0]), arr(vec![-1.0, -2.0, -3.0]),
            arr(vec![-1.1, -2.1, -3.1]), f(0.1),
        ]).unwrap();
        if let Value::Float(v) = loss { assert!(v.is_finite()); } else { panic!("expected float"); }
    }

    #[test]
    fn test_reward_model() {
        let mut e = env();
        let id = builtin_reward_create(&mut e, vec![Value::Int(4)]).unwrap();
        let score = builtin_reward_score(&mut e, vec![
            id, arr(vec![1.0, 0.5, -0.5, 0.0]),
        ]).unwrap();
        if let Value::Float(v) = score { assert!(v >= -1.0 && v <= 1.0); } else { panic!("expected float"); }
    }

    #[test]
    fn test_normalize_rewards() {
        let mut e = env();
        let result = builtin_rl_normalize_rewards(&mut e, vec![arr(vec![1.0, 2.0, 3.0])]).unwrap();
        if let Value::Array(a) = result {
            let vals: Vec<f64> = a.iter().map(|v| if let Value::Float(f) = v { *f } else { panic!() }).collect();
            let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
            assert!(mean.abs() < 1e-10);
        } else { panic!("expected array"); }
    }

    #[test]
    fn test_kl_divergence() {
        let mut e = env();
        // KL(p||p) = 0
        let kl = builtin_kl_divergence(&mut e, vec![
            arr(vec![-1.0, -2.0]), arr(vec![-1.0, -2.0]),
        ]).unwrap();
        if let Value::Float(v) = kl { assert!(v.abs() < 1e-10); } else { panic!("expected float"); }
    }
}
