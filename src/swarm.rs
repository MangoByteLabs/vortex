//! Swarm intelligence: multiple Vortex instances forming a collective mind
//! with gossip learning, emergent specialization, and distributed knowledge.

use std::collections::HashMap;
use crate::interpreter::{Env, FnDef, Value};

// ---- SwarmNode ----

#[derive(Clone, Debug)]
pub struct SwarmNode {
    pub id: usize,
    pub weights: Vec<Vec<f64>>,
    pub specialization: Option<String>,
    pub knowledge_hash: u64,
    pub performance: HashMap<String, f64>,
    pub alive: bool,
}

impl SwarmNode {
    pub fn new(id: usize, width: usize) -> Self {
        // Initialize with small pseudo-random weights based on id
        let mut weights = Vec::new();
        for i in 0..width {
            let mut row = Vec::with_capacity(width);
            for j in 0..width {
                let seed = (id * 1000 + i * 37 + j * 13 + 7) as f64;
                row.push((seed * 0.0073).sin() * 0.5);
            }
            weights.push(row);
        }
        let knowledge_hash = Self::compute_hash(&weights);
        SwarmNode {
            id,
            weights,
            specialization: None,
            knowledge_hash,
            performance: HashMap::new(),
            alive: true,
        }
    }

    fn compute_hash(weights: &[Vec<f64>]) -> u64 {
        let mut h: u64 = 0xcbf29ce484222325;
        for row in weights {
            for &v in row {
                let bits = v.to_bits();
                h ^= bits;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        h
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        // Simple matrix multiply + tanh activation
        let width = self.weights.len();
        let mut output = vec![0.0; width];
        for i in 0..width {
            let mut sum = 0.0;
            for j in 0..input.len().min(width) {
                sum += self.weights[i][j] * input[j];
            }
            output[i] = sum.tanh();
        }
        output
    }

    pub fn learn(&mut self, input: &[f64], target: &[f64], lr: f64) -> f64 {
        let output = self.forward(input);
        let width = self.weights.len();
        let mut loss = 0.0;
        for i in 0..width.min(target.len()) {
            let diff = output[i] - target[i];
            loss += diff * diff;
        }
        loss /= width.max(1) as f64;

        // Gradient descent (simplified backprop for single-layer)
        for i in 0..width.min(target.len()) {
            let diff = output[i] - target[i];
            let dtanh = 1.0 - output[i] * output[i];
            for j in 0..input.len().min(width) {
                self.weights[i][j] -= lr * 2.0 * diff * dtanh * input[j] / width as f64;
            }
        }
        self.knowledge_hash = Self::compute_hash(&self.weights);
        loss
    }
}

// ---- GossipProtocol ----

pub struct GossipProtocol;

impl GossipProtocol {
    pub fn merge_weights(a: &[Vec<f64>], b: &[Vec<f64>], alpha: f64) -> Vec<Vec<f64>> {
        a.iter()
            .zip(b.iter())
            .map(|(ra, rb)| {
                ra.iter()
                    .zip(rb.iter())
                    .map(|(&va, &vb)| va * alpha + vb * (1.0 - alpha))
                    .collect()
            })
            .collect()
    }

    pub fn gossip_round(nodes: &mut [SwarmNode], pairs: &[(usize, usize)]) {
        for &(a, b) in pairs {
            if a >= nodes.len() || b >= nodes.len() {
                continue;
            }
            if !nodes[a].alive || !nodes[b].alive {
                continue;
            }
            let merged = Self::merge_weights(&nodes[a].weights, &nodes[b].weights, 0.5);
            nodes[a].weights = merged.clone();
            nodes[b].weights = merged;
            nodes[a].knowledge_hash = SwarmNode::compute_hash(&nodes[a].weights);
            nodes[b].knowledge_hash = SwarmNode::compute_hash(&nodes[b].weights);
        }
    }

    pub fn select_pairs(n_nodes: usize) -> Vec<(usize, usize)> {
        let mut pairs = Vec::new();
        let mut used = vec![false; n_nodes];
        // Deterministic pairing: pair (0,1), (2,3), ...
        let mut i = 0;
        while i + 1 < n_nodes {
            if !used[i] && !used[i + 1] {
                pairs.push((i, i + 1));
                used[i] = true;
                used[i + 1] = true;
            }
            i += 2;
        }
        pairs
    }

    pub fn selective_gossip(nodes: &[SwarmNode], node_id: usize) -> usize {
        // Pick most complementary peer (most different knowledge hash)
        let my_hash = nodes[node_id].knowledge_hash;
        let mut best_peer = 0;
        let mut best_diff = 0u64;
        for (i, node) in nodes.iter().enumerate() {
            if i == node_id || !node.alive {
                continue;
            }
            let diff = my_hash ^ node.knowledge_hash;
            if diff > best_diff {
                best_diff = diff;
                best_peer = i;
            }
        }
        best_peer
    }
}

// ---- SwarmTopology ----

#[derive(Clone, Debug)]
pub enum SwarmTopology {
    FullyConnected,
    Ring,
    SmallWorld(f64),   // rewire_prob
    ScaleFree(usize),  // m (edges per new node)
}

impl SwarmTopology {
    pub fn neighbors(&self, node_id: usize, n_nodes: usize) -> Vec<usize> {
        match self {
            SwarmTopology::FullyConnected => {
                (0..n_nodes).filter(|&i| i != node_id).collect()
            }
            SwarmTopology::Ring => {
                if n_nodes <= 1 {
                    return vec![];
                }
                let prev = if node_id == 0 { n_nodes - 1 } else { node_id - 1 };
                let next = (node_id + 1) % n_nodes;
                if prev == next {
                    vec![prev]
                } else {
                    vec![prev, next]
                }
            }
            SwarmTopology::SmallWorld(rewire_prob) => {
                // Start with ring, add long-range connections deterministically
                let mut nbrs = vec![];
                if n_nodes <= 1 {
                    return nbrs;
                }
                let prev = if node_id == 0 { n_nodes - 1 } else { node_id - 1 };
                let next = (node_id + 1) % n_nodes;
                nbrs.push(prev);
                if next != prev {
                    nbrs.push(next);
                }
                // Deterministic "rewiring": add shortcut based on hash
                let hash_val = (node_id as f64 * 0.618).fract();
                if hash_val < *rewire_prob && n_nodes > 3 {
                    let shortcut = (node_id + n_nodes / 2) % n_nodes;
                    if shortcut != node_id && !nbrs.contains(&shortcut) {
                        nbrs.push(shortcut);
                    }
                }
                nbrs
            }
            SwarmTopology::ScaleFree(m) => {
                // Simplified preferential attachment: connect to first m nodes (hubs)
                let mut nbrs = vec![];
                for i in 0..(*m).min(n_nodes) {
                    if i != node_id {
                        nbrs.push(i);
                    }
                }
                // Also connect to neighbors within distance m
                for offset in 1..=*m {
                    let next = (node_id + offset) % n_nodes;
                    if next != node_id && !nbrs.contains(&next) {
                        nbrs.push(next);
                    }
                }
                nbrs
            }
        }
    }
}

// ---- SwarmStats ----

#[derive(Clone, Debug)]
pub struct SwarmStats {
    pub n_nodes: usize,
    pub avg_loss: f64,
    pub best_node: usize,
    pub specialization_map: HashMap<String, Vec<usize>>,
    pub gossip_rounds: usize,
    pub knowledge_convergence: f64,
}

// ---- EmergentSpecialization ----

pub struct EmergentSpecialization;

impl EmergentSpecialization {
    pub fn track_performance(node: &mut SwarmNode, task_type: &str, loss: f64) {
        let entry = node.performance.entry(task_type.to_string()).or_insert(loss);
        // Exponential moving average
        *entry = *entry * 0.9 + loss * 0.1;
    }

    pub fn detect_specialization(node: &SwarmNode) -> Option<String> {
        if node.performance.is_empty() {
            return None;
        }
        node.performance
            .iter()
            .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(k, _)| k.clone())
    }

    pub fn diversity_score(swarm: &VortexSwarm) -> f64 {
        let mut specs: HashMap<String, usize> = HashMap::new();
        let mut total = 0;
        for node in &swarm.nodes {
            if let Some(ref s) = node.specialization {
                *specs.entry(s.clone()).or_insert(0) += 1;
                total += 1;
            }
        }
        if total == 0 {
            return 0.0;
        }
        // Shannon entropy normalized
        let n = total as f64;
        let mut entropy = 0.0;
        for &count in specs.values() {
            let p = count as f64 / n;
            if p > 0.0 {
                entropy -= p * p.ln();
            }
        }
        let max_entropy = (specs.len() as f64).ln().max(1.0);
        entropy / max_entropy
    }

    pub fn redundancy_score(swarm: &VortexSwarm) -> f64 {
        if swarm.nodes.len() < 2 {
            return 1.0;
        }
        let mut same_count = 0;
        let mut pair_count = 0;
        for i in 0..swarm.nodes.len() {
            for j in (i + 1)..swarm.nodes.len() {
                pair_count += 1;
                if swarm.nodes[i].knowledge_hash == swarm.nodes[j].knowledge_hash {
                    same_count += 1;
                }
            }
        }
        if pair_count == 0 {
            return 1.0;
        }
        same_count as f64 / pair_count as f64
    }
}

// ---- FaultTolerance ----

pub struct FaultTolerance;

impl FaultTolerance {
    pub fn checkpoint_swarm(swarm: &VortexSwarm) -> Vec<u8> {
        // Simple binary serialization
        let mut data = Vec::new();
        // n_nodes (u32)
        let n = swarm.nodes.len() as u32;
        data.extend_from_slice(&n.to_le_bytes());
        // topology tag
        let topo_tag: u8 = match &swarm.topology {
            SwarmTopology::FullyConnected => 0,
            SwarmTopology::Ring => 1,
            SwarmTopology::SmallWorld(_) => 2,
            SwarmTopology::ScaleFree(_) => 3,
        };
        data.push(topo_tag);
        // gossip_rounds
        data.extend_from_slice(&(swarm.gossip_rounds as u32).to_le_bytes());
        // each node: id, width, weights, alive
        for node in &swarm.nodes {
            data.extend_from_slice(&(node.id as u32).to_le_bytes());
            let width = node.weights.len() as u32;
            data.extend_from_slice(&width.to_le_bytes());
            data.push(node.alive as u8);
            for row in &node.weights {
                for &v in row {
                    data.extend_from_slice(&v.to_le_bytes());
                }
            }
        }
        data
    }

    pub fn restore_swarm(data: &[u8]) -> VortexSwarm {
        let mut pos = 0;
        let n = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let topo_tag = data[pos];
        pos += 1;
        let topology = match topo_tag {
            0 => SwarmTopology::FullyConnected,
            1 => SwarmTopology::Ring,
            2 => SwarmTopology::SmallWorld(0.1),
            _ => SwarmTopology::ScaleFree(2),
        };
        let gossip_rounds = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;
        let mut nodes = Vec::with_capacity(n);
        for _ in 0..n {
            let id = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let width = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
            pos += 4;
            let alive = data[pos] != 0;
            pos += 1;
            let mut weights = Vec::with_capacity(width);
            for _ in 0..width {
                let mut row = Vec::with_capacity(width);
                for _ in 0..width {
                    let v = f64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
                    pos += 8;
                    row.push(v);
                }
                weights.push(row);
            }
            let knowledge_hash = SwarmNode::compute_hash(&weights);
            nodes.push(SwarmNode {
                id,
                weights,
                specialization: None,
                knowledge_hash,
                performance: HashMap::new(),
                alive,
            });
        }
        VortexSwarm {
            nodes,
            topology,
            gossip_rounds,
        }
    }

    pub fn redistribute_knowledge(swarm: &mut VortexSwarm, failed_id: usize) {
        if failed_id >= swarm.nodes.len() {
            return;
        }
        let dead_weights = swarm.nodes[failed_id].weights.clone();
        swarm.nodes[failed_id].alive = false;
        let alive_ids: Vec<usize> = swarm
            .nodes
            .iter()
            .enumerate()
            .filter(|(i, n)| *i != failed_id && n.alive)
            .map(|(i, _)| i)
            .collect();
        if alive_ids.is_empty() {
            return;
        }
        let alpha = 1.0 / (alive_ids.len() as f64 + 1.0);
        for &i in &alive_ids {
            swarm.nodes[i].weights =
                GossipProtocol::merge_weights(&swarm.nodes[i].weights, &dead_weights, 1.0 - alpha);
            swarm.nodes[i].knowledge_hash = SwarmNode::compute_hash(&swarm.nodes[i].weights);
        }
    }

    pub fn health_check(swarm: &VortexSwarm) -> Vec<(usize, bool)> {
        swarm.nodes.iter().map(|n| (n.id, n.alive)).collect()
    }
}

// ---- VortexSwarm ----

#[derive(Clone, Debug)]
pub struct VortexSwarm {
    pub nodes: Vec<SwarmNode>,
    pub topology: SwarmTopology,
    pub gossip_rounds: usize,
}

impl VortexSwarm {
    pub fn new(topology: SwarmTopology) -> Self {
        VortexSwarm {
            nodes: Vec::new(),
            topology,
            gossip_rounds: 0,
        }
    }

    pub fn add_node(&mut self, width: usize) -> usize {
        let id = self.nodes.len();
        self.nodes.push(SwarmNode::new(id, width));
        id
    }

    pub fn remove_node(&mut self, id: usize) {
        if id < self.nodes.len() {
            self.nodes[id].alive = false;
        }
    }

    pub fn route_query(&self, query: &[f64]) -> usize {
        // Find best node: highest output magnitude (most confident)
        let mut best_id = 0;
        let mut best_score = f64::NEG_INFINITY;
        for (i, node) in self.nodes.iter().enumerate() {
            if !node.alive {
                continue;
            }
            let out = node.forward(query);
            let score: f64 = out.iter().map(|v| v.abs()).sum();
            if score > best_score {
                best_score = score;
                best_id = i;
            }
        }
        best_id
    }

    pub fn collective_forward(&self, query: &[f64], top_k: usize) -> Vec<f64> {
        // Gather outputs from top-k most confident nodes, average them
        let mut scored: Vec<(usize, f64, Vec<f64>)> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.alive)
            .map(|(i, n)| {
                let out = n.forward(query);
                let score: f64 = out.iter().map(|v| v.abs()).sum();
                (i, score, out)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let k = top_k.min(scored.len()).max(1);
        if scored.is_empty() {
            return vec![];
        }
        let width = scored[0].2.len();
        let mut result = vec![0.0; width];
        for (_, _, out) in scored.iter().take(k) {
            for (r, o) in result.iter_mut().zip(out.iter()) {
                *r += o;
            }
        }
        for r in &mut result {
            *r /= k as f64;
        }
        result
    }

    pub fn gossip_step(&mut self) {
        let pairs = GossipProtocol::select_pairs(self.nodes.len());
        GossipProtocol::gossip_round(&mut self.nodes, &pairs);
        self.gossip_rounds += 1;
    }

    pub fn specialize(
        &mut self,
        node_id: usize,
        task_type: &str,
        data: &[Vec<f64>],
        targets: &[Vec<f64>],
        epochs: usize,
    ) {
        if node_id >= self.nodes.len() {
            return;
        }
        for _ in 0..epochs {
            let mut total_loss = 0.0;
            for (input, target) in data.iter().zip(targets.iter()) {
                total_loss += self.nodes[node_id].learn(input, target, 0.01);
            }
            let avg = total_loss / data.len().max(1) as f64;
            EmergentSpecialization::track_performance(&mut self.nodes[node_id], task_type, avg);
        }
        self.nodes[node_id].specialization =
            EmergentSpecialization::detect_specialization(&self.nodes[node_id]);
    }

    pub fn collective_stats(&self) -> SwarmStats {
        let n_nodes = self.nodes.iter().filter(|n| n.alive).count();
        // avg loss from performance maps
        let mut total_loss = 0.0;
        let mut loss_count = 0;
        let mut best_node = 0;
        let mut best_loss = f64::INFINITY;
        let mut spec_map: HashMap<String, Vec<usize>> = HashMap::new();
        for node in &self.nodes {
            if !node.alive {
                continue;
            }
            for (task, &loss) in &node.performance {
                total_loss += loss;
                loss_count += 1;
                if loss < best_loss {
                    best_loss = loss;
                    best_node = node.id;
                }
                spec_map.entry(task.clone()).or_default().push(node.id);
            }
        }
        let avg_loss = if loss_count > 0 {
            total_loss / loss_count as f64
        } else {
            0.0
        };
        // knowledge convergence: fraction of pairs with same hash
        let convergence = EmergentSpecialization::redundancy_score(self);
        SwarmStats {
            n_nodes,
            avg_loss,
            best_node,
            specialization_map: spec_map,
            gossip_rounds: self.gossip_rounds,
            knowledge_convergence: convergence,
        }
    }
}

// ---- Interpreter builtins ----

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("swarm_new".to_string(), FnDef::Builtin(builtin_swarm_new));
    env.functions.insert("swarm_forward".to_string(), FnDef::Builtin(builtin_swarm_forward));
    env.functions.insert("swarm_gossip".to_string(), FnDef::Builtin(builtin_swarm_gossip));
    env.functions.insert("swarm_specialize".to_string(), FnDef::Builtin(builtin_swarm_specialize));
    env.functions.insert("swarm_stats".to_string(), FnDef::Builtin(builtin_swarm_stats));
    env.functions.insert("swarm_route".to_string(), FnDef::Builtin(builtin_swarm_route));
}

fn value_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        Value::Float(f) => Ok(*f as usize),
        _ => Err(format!("Expected integer, got {:?}", v)),
    }
}

fn value_to_f64_vec(v: &Value) -> Result<Vec<f64>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Float(f) => Ok(*f),
                Value::Int(n) => Ok(*n as f64),
                _ => Err("array elements must be numeric".to_string()),
            })
            .collect(),
        _ => Err("expected array".to_string()),
    }
}

fn builtin_swarm_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("swarm_new expects 2 args: (n_nodes, width)".into());
    }
    let n_nodes = value_to_usize(&args[0])?;
    let width = value_to_usize(&args[1])?;
    let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
    for _ in 0..n_nodes {
        swarm.add_node(width);
    }
    let id = env.next_swarm_id;
    env.next_swarm_id += 1;
    env.swarms.insert(id, swarm);
    Ok(Value::Int(id as i128))
}

fn builtin_swarm_forward(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("swarm_forward expects 2 args: (id, query)".into());
    }
    let id = value_to_usize(&args[0])?;
    let query = value_to_f64_vec(&args[1])?;
    let swarm = env.swarms.get(&id).ok_or("no swarm with that id")?;
    let top_k = swarm.nodes.len().min(3).max(1);
    let output = swarm.collective_forward(&query, top_k);
    Ok(Value::Array(output.into_iter().map(Value::Float).collect()))
}

fn builtin_swarm_gossip(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("swarm_gossip expects 1 arg: (id)".into());
    }
    let id = value_to_usize(&args[0])?;
    let swarm = env.swarms.get_mut(&id).ok_or("no swarm with that id")?;
    swarm.gossip_step();
    Ok(Value::Int(swarm.gossip_rounds as i128))
}

fn builtin_swarm_specialize(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 {
        return Err(
            "swarm_specialize expects 6 args: (id, node, task, data, targets, epochs)".into(),
        );
    }
    let id = value_to_usize(&args[0])?;
    let node_id = value_to_usize(&args[1])?;
    let task = match &args[2] {
        Value::String(s) => s.clone(),
        _ => return Err("task must be a string".into()),
    };
    let data: Vec<Vec<f64>> = match &args[3] {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect::<Result<_, _>>()?,
        _ => return Err("data must be array of arrays".into()),
    };
    let targets: Vec<Vec<f64>> = match &args[4] {
        Value::Array(arr) => arr.iter().map(value_to_f64_vec).collect::<Result<_, _>>()?,
        _ => return Err("targets must be array of arrays".into()),
    };
    let epochs = value_to_usize(&args[5])?;
    let swarm = env.swarms.get_mut(&id).ok_or("no swarm with that id")?;
    swarm.specialize(node_id, &task, &data, &targets, epochs);
    Ok(Value::Void)
}

fn builtin_swarm_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("swarm_stats expects 1 arg: (id)".into());
    }
    let id = value_to_usize(&args[0])?;
    let swarm = env.swarms.get(&id).ok_or("no swarm with that id")?;
    let stats = swarm.collective_stats();
    let diversity = EmergentSpecialization::diversity_score(swarm);
    Ok(Value::Array(vec![
        Value::Float(stats.n_nodes as f64),
        Value::Float(stats.avg_loss),
        Value::Float(diversity),
        Value::Float(stats.knowledge_convergence),
    ]))
}

fn builtin_swarm_route(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("swarm_route expects 2 args: (id, query)".into());
    }
    let id = value_to_usize(&args[0])?;
    let query = value_to_f64_vec(&args[1])?;
    let swarm = env.swarms.get(&id).ok_or("no swarm with that id")?;
    let best = swarm.route_query(&query);
    Ok(Value::Int(best as i128))
}

// ---- Tests ----

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_swarm_node_forward() {
        let node = SwarmNode::new(0, 4);
        let out = node.forward(&[1.0, 0.0, 0.0, 0.0]);
        assert_eq!(out.len(), 4);
        for v in &out {
            assert!(v.abs() <= 1.0, "tanh output should be in [-1,1]");
        }
    }

    #[test]
    fn test_swarm_node_learn_reduces_loss() {
        let mut node = SwarmNode::new(0, 3);
        let input = vec![1.0, 0.5, -0.3];
        let target = vec![0.1, -0.2, 0.8];
        let loss1 = node.learn(&input, &target, 0.1);
        // Train a few more times
        let mut loss = loss1;
        for _ in 0..50 {
            loss = node.learn(&input, &target, 0.1);
        }
        assert!(loss < loss1, "loss should decrease with training");
    }

    #[test]
    fn test_gossip_merge_weights() {
        let a = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let b = vec![vec![5.0, 6.0], vec![7.0, 8.0]];
        let merged = GossipProtocol::merge_weights(&a, &b, 0.5);
        assert!((merged[0][0] - 3.0).abs() < 1e-10);
        assert!((merged[1][1] - 6.0).abs() < 1e-10);
    }

    #[test]
    fn test_gossip_round_converges() {
        let mut nodes = vec![SwarmNode::new(0, 2), SwarmNode::new(1, 2)];
        let h0 = nodes[0].knowledge_hash;
        let h1 = nodes[1].knowledge_hash;
        assert_ne!(h0, h1, "nodes start with different weights");
        GossipProtocol::gossip_round(&mut nodes, &[(0, 1)]);
        assert_eq!(
            nodes[0].knowledge_hash, nodes[1].knowledge_hash,
            "after gossip, hashes should match"
        );
    }

    #[test]
    fn test_select_pairs() {
        let pairs = GossipProtocol::select_pairs(5);
        assert_eq!(pairs, vec![(0, 1), (2, 3)]);
    }

    #[test]
    fn test_selective_gossip() {
        let nodes = vec![
            SwarmNode::new(0, 2),
            SwarmNode::new(1, 2),
            SwarmNode::new(2, 2),
        ];
        let peer = GossipProtocol::selective_gossip(&nodes, 0);
        // Should pick the most different node
        assert!(peer == 1 || peer == 2);
    }

    #[test]
    fn test_topology_fully_connected() {
        let t = SwarmTopology::FullyConnected;
        let nbrs = t.neighbors(1, 4);
        assert_eq!(nbrs, vec![0, 2, 3]);
    }

    #[test]
    fn test_topology_ring() {
        let t = SwarmTopology::Ring;
        let nbrs = t.neighbors(0, 4);
        assert!(nbrs.contains(&3)); // prev wraps
        assert!(nbrs.contains(&1)); // next
        assert_eq!(nbrs.len(), 2);
    }

    #[test]
    fn test_vortex_swarm_add_and_route() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(3);
        swarm.add_node(3);
        swarm.add_node(3);
        let best = swarm.route_query(&[1.0, 0.0, 0.0]);
        assert!(best < 3);
    }

    #[test]
    fn test_collective_forward() {
        let mut swarm = VortexSwarm::new(SwarmTopology::Ring);
        for _ in 0..4 {
            swarm.add_node(3);
        }
        let out = swarm.collective_forward(&[1.0, 0.5, -0.5], 2);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_specialize_and_detect() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(2);
        let data = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let targets = vec![vec![0.5, -0.5], vec![-0.5, 0.5]];
        swarm.specialize(0, "classify", &data, &targets, 10);
        assert_eq!(swarm.nodes[0].specialization, Some("classify".to_string()));
    }

    #[test]
    fn test_fault_tolerance_checkpoint_restore() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(2);
        swarm.add_node(2);
        swarm.gossip_step();
        let data = FaultTolerance::checkpoint_swarm(&swarm);
        let restored = FaultTolerance::restore_swarm(&data);
        assert_eq!(restored.nodes.len(), 2);
        assert_eq!(restored.gossip_rounds, swarm.gossip_rounds);
        assert_eq!(
            restored.nodes[0].knowledge_hash,
            swarm.nodes[0].knowledge_hash
        );
    }

    #[test]
    fn test_redistribute_knowledge() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(2);
        swarm.add_node(2);
        swarm.add_node(2);
        let hash_before = swarm.nodes[1].knowledge_hash;
        FaultTolerance::redistribute_knowledge(&mut swarm, 0);
        assert!(!swarm.nodes[0].alive);
        // Surviving nodes should have absorbed some knowledge
        assert_ne!(swarm.nodes[1].knowledge_hash, hash_before);
    }

    #[test]
    fn test_health_check() {
        let mut swarm = VortexSwarm::new(SwarmTopology::Ring);
        swarm.add_node(2);
        swarm.add_node(2);
        swarm.remove_node(1);
        let health = FaultTolerance::health_check(&swarm);
        assert_eq!(health, vec![(0, true), (1, false)]);
    }

    #[test]
    fn test_collective_stats() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(2);
        swarm.add_node(2);
        let stats = swarm.collective_stats();
        assert_eq!(stats.n_nodes, 2);
        assert_eq!(stats.gossip_rounds, 0);
    }

    #[test]
    fn test_diversity_and_redundancy() {
        let mut swarm = VortexSwarm::new(SwarmTopology::FullyConnected);
        swarm.add_node(2);
        swarm.add_node(2);
        // No specializations yet
        assert_eq!(EmergentSpecialization::diversity_score(&swarm), 0.0);
        // Different weights = low redundancy
        let redundancy = EmergentSpecialization::redundancy_score(&swarm);
        assert!(redundancy < 1.0);
    }
}
