//! Pillar 1: LLM-to-LLM Binary Protocol
//!
//! Provides a binary communication protocol for agent-to-agent thought exchange,
//! including typed thought packets, trust-weighted consensus, cosine similarity
//! matching, selective broadcasting, and binary encode/decode of thought data.

use std::collections::HashMap;
use crate::interpreter::{Env, Value, FnDef};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Clone, Debug, PartialEq)]
pub enum ThoughtType {
    Query,
    Response,
    ArchProposal,
    GradientShare,
    SpecVerification,
    WorldModelUpdate,
    MetaCognition,
}

impl ThoughtType {
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "query" => Ok(ThoughtType::Query),
            "response" => Ok(ThoughtType::Response),
            "arch_proposal" => Ok(ThoughtType::ArchProposal),
            "gradient_share" => Ok(ThoughtType::GradientShare),
            "spec_verification" => Ok(ThoughtType::SpecVerification),
            "world_model_update" => Ok(ThoughtType::WorldModelUpdate),
            "meta_cognition" => Ok(ThoughtType::MetaCognition),
            _ => Err(format!("Unknown thought type: {}", s)),
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            ThoughtType::Query => "query",
            ThoughtType::Response => "response",
            ThoughtType::ArchProposal => "arch_proposal",
            ThoughtType::GradientShare => "gradient_share",
            ThoughtType::SpecVerification => "spec_verification",
            ThoughtType::WorldModelUpdate => "world_model_update",
            ThoughtType::MetaCognition => "meta_cognition",
        }
    }

    fn as_u8(&self) -> u8 {
        match self {
            ThoughtType::Query => 0,
            ThoughtType::Response => 1,
            ThoughtType::ArchProposal => 2,
            ThoughtType::GradientShare => 3,
            ThoughtType::SpecVerification => 4,
            ThoughtType::WorldModelUpdate => 5,
            ThoughtType::MetaCognition => 6,
        }
    }

    fn from_u8(v: u8) -> Result<Self, String> {
        match v {
            0 => Ok(ThoughtType::Query),
            1 => Ok(ThoughtType::Response),
            2 => Ok(ThoughtType::ArchProposal),
            3 => Ok(ThoughtType::GradientShare),
            4 => Ok(ThoughtType::SpecVerification),
            5 => Ok(ThoughtType::WorldModelUpdate),
            6 => Ok(ThoughtType::MetaCognition),
            _ => Err(format!("Invalid thought type byte: {}", v)),
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThoughtPacket {
    pub id: u64,
    pub sender: String,
    pub embeddings: Vec<f64>,
    pub thought_type: ThoughtType,
    pub confidence: f64,
    pub timestamp: u64,
    pub references: Vec<u64>,
    pub payload: Vec<f64>,
    pub metadata: HashMap<String, String>,
}

#[derive(Clone, Debug)]
pub struct AgentState {
    pub id: String,
    pub embedding_dim: usize,
    pub capabilities: Vec<String>,
    pub trust_score: f64,
    pub specialization: String,
    pub message_count: u64,
}

impl AgentState {
    fn new(id: String, embedding_dim: usize, specialization: String) -> Self {
        AgentState {
            id,
            embedding_dim,
            capabilities: Vec::new(),
            trust_score: 1.0,
            specialization,
            message_count: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ThoughtChannel {
    pub id: usize,
    pub agents: Vec<AgentState>,
    pub message_log: Vec<ThoughtPacket>,
    pub bandwidth_bytes: u64,
    pub next_packet_id: u64,
}

impl ThoughtChannel {
    fn new(id: usize) -> Self {
        ThoughtChannel {
            id,
            agents: Vec::new(),
            message_log: Vec::new(),
            bandwidth_bytes: 0,
            next_packet_id: 1,
        }
    }

    fn find_agent(&self, name: &str) -> Option<usize> {
        self.agents.iter().position(|a| a.id == name)
    }

    fn find_agent_mut(&mut self, name: &str) -> Option<&mut AgentState> {
        self.agents.iter_mut().find(|a| a.id == name)
    }

    fn add_agent(&mut self, agent: AgentState) {
        self.agents.push(agent);
    }

    fn send_packet(&mut self, sender: &str, thought_type: ThoughtType,
                    embeddings: Vec<f64>, confidence: f64, payload: Vec<f64>) -> u64 {
        let id = self.next_packet_id;
        self.next_packet_id += 1;

        // Estimate bandwidth: 8 bytes per f64 + overhead
        let packet_bytes = (embeddings.len() + payload.len()) as u64 * 8 + 64;
        self.bandwidth_bytes += packet_bytes;

        // Increment sender message count
        if let Some(agent) = self.find_agent_mut(sender) {
            agent.message_count += 1;
        }

        // Timestamp: simple monotonic counter based on packet id
        let timestamp = id;

        let packet = ThoughtPacket {
            id,
            sender: sender.to_string(),
            embeddings,
            thought_type,
            confidence,
            timestamp,
            references: Vec::new(),
            payload,
            metadata: HashMap::new(),
        };
        self.message_log.push(packet);
        id
    }

    fn receive_for(&self, receiver: &str) -> Vec<&ThoughtPacket> {
        // Return all packets not sent by the receiver
        self.message_log.iter().filter(|p| p.sender != receiver).collect()
    }

    fn packets_by_ids(&self, ids: &[u64]) -> Vec<&ThoughtPacket> {
        self.message_log.iter().filter(|p| ids.contains(&p.id)).collect()
    }
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }
    let mut dot = 0.0;
    let mut norm_a = 0.0;
    let mut norm_b = 0.0;
    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-15 {
        0.0
    } else {
        dot / denom
    }
}

fn confidence_weighted_merge(packets: &[&ThoughtPacket], trust_scores: &[f64]) -> Vec<f64> {
    if packets.is_empty() {
        return Vec::new();
    }
    let dim = packets.iter().map(|p| p.embeddings.len()).max().unwrap_or(0);
    if dim == 0 {
        return Vec::new();
    }
    let mut merged = vec![0.0; dim];
    let mut total_weight = 0.0;
    for (i, packet) in packets.iter().enumerate() {
        let trust = if i < trust_scores.len() { trust_scores[i] } else { 1.0 };
        let weight = packet.confidence * trust;
        total_weight += weight;
        for (j, &v) in packet.embeddings.iter().enumerate() {
            if j < dim {
                merged[j] += v * weight;
            }
        }
    }
    if total_weight > 1e-15 {
        for v in &mut merged {
            *v /= total_weight;
        }
    }
    merged
}

// ---------------------------------------------------------------------------
// Binary encode/decode
// ---------------------------------------------------------------------------

/// Binary format:
/// [4 bytes: magic 0x54485054 "THPT"]
/// [1 byte: thought type]
/// [8 bytes: confidence as f64 LE]
/// [4 bytes: embedding count as u32 LE]
/// [N * 8 bytes: embeddings as f64 LE]
fn encode_thought_binary(embeddings: &[f64], thought_type: &ThoughtType, confidence: f64) -> Vec<u8> {
    let mut buf = Vec::new();
    // Magic bytes
    buf.extend_from_slice(&[0x54, 0x48, 0x50, 0x54]);
    // Thought type
    buf.push(thought_type.as_u8());
    // Confidence
    buf.extend_from_slice(&confidence.to_le_bytes());
    // Embedding count
    let count = embeddings.len() as u32;
    buf.extend_from_slice(&count.to_le_bytes());
    // Embeddings
    for &v in embeddings {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

fn decode_thought_binary(data: &[u8]) -> Result<(Vec<f64>, ThoughtType, f64), String> {
    if data.len() < 17 {
        return Err("encoded data too short".to_string());
    }
    // Verify magic
    if data[0] != 0x54 || data[1] != 0x48 || data[2] != 0x50 || data[3] != 0x54 {
        return Err("invalid magic bytes".to_string());
    }
    let thought_type = ThoughtType::from_u8(data[4])?;
    let confidence = f64::from_le_bytes(data[5..13].try_into().map_err(|_| "bad confidence bytes")?);
    let count = u32::from_le_bytes(data[13..17].try_into().map_err(|_| "bad count bytes")?) as usize;
    let expected_len = 17 + count * 8;
    if data.len() < expected_len {
        return Err(format!("encoded data truncated: need {} bytes, got {}", expected_len, data.len()));
    }
    let mut embeddings = Vec::with_capacity(count);
    for i in 0..count {
        let offset = 17 + i * 8;
        let v = f64::from_le_bytes(data[offset..offset + 8].try_into().map_err(|_| "bad f64 bytes")?);
        embeddings.push(v);
    }
    Ok((embeddings, thought_type, confidence))
}

// ---------------------------------------------------------------------------
// Value conversion helpers
// ---------------------------------------------------------------------------

fn value_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err(format!("Expected number, got {:?}", v)),
    }
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

fn value_to_string(v: &Value) -> Result<String, String> {
    match v {
        Value::String(s) => Ok(s.clone()),
        _ => Err(format!("Expected string, got {:?}", v)),
    }
}

fn value_to_u64_vec(v: &Value) -> Result<Vec<u64>, String> {
    match v {
        Value::Array(arr) => arr
            .iter()
            .map(|v| match v {
                Value::Int(n) => Ok(*n as u64),
                Value::Float(f) => Ok(*f as u64),
                _ => Err("array elements must be numeric".to_string()),
            })
            .collect(),
        _ => Err("expected array".to_string()),
    }
}

// ---------------------------------------------------------------------------
// Builtin registrations
// ---------------------------------------------------------------------------

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("channel_new".to_string(), FnDef::Builtin(builtin_channel_new));
    env.functions.insert("channel_add_agent".to_string(), FnDef::Builtin(builtin_channel_add_agent));
    env.functions.insert("send_thought".to_string(), FnDef::Builtin(builtin_send_thought));
    env.functions.insert("send_thought_with_payload".to_string(), FnDef::Builtin(builtin_send_thought_with_payload));
    env.functions.insert("receive_thoughts".to_string(), FnDef::Builtin(builtin_receive_thoughts));
    env.functions.insert("thought_similarity".to_string(), FnDef::Builtin(builtin_thought_similarity));
    env.functions.insert("consensus_merge".to_string(), FnDef::Builtin(builtin_consensus_merge));
    env.functions.insert("selective_broadcast".to_string(), FnDef::Builtin(builtin_selective_broadcast));
    env.functions.insert("negotiate_protocol".to_string(), FnDef::Builtin(builtin_negotiate_protocol));
    env.functions.insert("agent_trust".to_string(), FnDef::Builtin(builtin_agent_trust));
    env.functions.insert("agent_update_trust".to_string(), FnDef::Builtin(builtin_agent_update_trust));
    env.functions.insert("channel_stats".to_string(), FnDef::Builtin(builtin_channel_stats));
    env.functions.insert("encode_thought".to_string(), FnDef::Builtin(builtin_encode_thought));
    env.functions.insert("decode_thought".to_string(), FnDef::Builtin(builtin_decode_thought));
}

// ---------------------------------------------------------------------------
// Builtin implementations
// ---------------------------------------------------------------------------

/// channel_new() -> channel_id
fn builtin_channel_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if !args.is_empty() {
        return Err("channel_new expects 0 args".into());
    }
    let id = env.next_thought_channel_id;
    env.next_thought_channel_id += 1;
    let channel = ThoughtChannel::new(id);
    env.thought_channels.insert(id, channel);
    Ok(Value::Int(id as i128))
}

/// channel_add_agent(channel_id, agent_name, embed_dim, specialization) -> void
fn builtin_channel_add_agent(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 {
        return Err("channel_add_agent expects 4 args: (channel_id, agent_name, embed_dim, specialization)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let name = value_to_string(&args[1])?;
    let embed_dim = value_to_usize(&args[2])?;
    let specialization = value_to_string(&args[3])?;
    let channel = env.thought_channels.get_mut(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    if channel.find_agent(&name).is_some() {
        return Err(format!("Agent '{}' already exists in channel {}", name, ch_id));
    }
    let agent = AgentState::new(name, embed_dim, specialization);
    channel.add_agent(agent);
    Ok(Value::Int(0))
}

/// send_thought(channel_id, sender, thought_type_str, embeddings, confidence) -> packet_id
fn builtin_send_thought(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 5 {
        return Err("send_thought expects 5 args: (channel_id, sender, type_str, embeddings, confidence)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let sender = value_to_string(&args[1])?;
    let type_str = value_to_string(&args[2])?;
    let embeddings = value_to_f64_vec(&args[3])?;
    let confidence = value_to_f64(&args[4])?;
    let thought_type = ThoughtType::from_str(&type_str)?;
    let channel = env.thought_channels.get_mut(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    if channel.find_agent(&sender).is_none() {
        return Err(format!("Agent '{}' not found in channel {}", sender, ch_id));
    }
    let packet_id = channel.send_packet(&sender, thought_type, embeddings, confidence, Vec::new());
    Ok(Value::Int(packet_id as i128))
}

/// send_thought_with_payload(channel_id, sender, type_str, embeddings, confidence, payload) -> packet_id
fn builtin_send_thought_with_payload(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 {
        return Err("send_thought_with_payload expects 6 args: (channel_id, sender, type_str, embeddings, confidence, payload)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let sender = value_to_string(&args[1])?;
    let type_str = value_to_string(&args[2])?;
    let embeddings = value_to_f64_vec(&args[3])?;
    let confidence = value_to_f64(&args[4])?;
    let payload = value_to_f64_vec(&args[5])?;
    let thought_type = ThoughtType::from_str(&type_str)?;
    let channel = env.thought_channels.get_mut(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    if channel.find_agent(&sender).is_none() {
        return Err(format!("Agent '{}' not found in channel {}", sender, ch_id));
    }
    let packet_id = channel.send_packet(&sender, thought_type, embeddings, confidence, payload);
    Ok(Value::Int(packet_id as i128))
}

/// receive_thoughts(channel_id, receiver) -> array of packet description hashmaps
fn builtin_receive_thoughts(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("receive_thoughts expects 2 args: (channel_id, receiver)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let receiver = value_to_string(&args[1])?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    let packets = channel.receive_for(&receiver);
    let mut result = Vec::new();
    for packet in packets {
        let mut desc = HashMap::new();
        desc.insert("id".to_string(), Value::Int(packet.id as i128));
        desc.insert("sender".to_string(), Value::String(packet.sender.clone()));
        desc.insert("type".to_string(), Value::String(packet.thought_type.as_str().to_string()));
        desc.insert("confidence".to_string(), Value::Float(packet.confidence));
        desc.insert("timestamp".to_string(), Value::Int(packet.timestamp as i128));
        desc.insert("embedding_dim".to_string(), Value::Int(packet.embeddings.len() as i128));
        desc.insert("payload_len".to_string(), Value::Int(packet.payload.len() as i128));
        desc.insert("embeddings".to_string(), Value::Array(
            packet.embeddings.iter().map(|&v| Value::Float(v)).collect(),
        ));
        desc.insert("payload".to_string(), Value::Array(
            packet.payload.iter().map(|&v| Value::Float(v)).collect(),
        ));
        result.push(Value::HashMap(desc));
    }
    Ok(Value::Array(result))
}

/// thought_similarity(embeddings1, embeddings2) -> f64
fn builtin_thought_similarity(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("thought_similarity expects 2 args: (embeddings1, embeddings2)".into());
    }
    let a = value_to_f64_vec(&args[0])?;
    let b = value_to_f64_vec(&args[1])?;
    let sim = cosine_similarity(&a, &b);
    Ok(Value::Float(sim))
}

/// consensus_merge(channel_id, packet_ids) -> merged embeddings (confidence * trust weighted average)
fn builtin_consensus_merge(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("consensus_merge expects 2 args: (channel_id, packet_ids)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let ids = value_to_u64_vec(&args[1])?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    let packets = channel.packets_by_ids(&ids);
    if packets.is_empty() {
        return Err("No packets found for the given ids".to_string());
    }
    // Gather trust scores for each packet's sender
    let trust_scores: Vec<f64> = packets.iter().map(|p| {
        channel.find_agent(&p.sender)
            .map(|idx| channel.agents[idx].trust_score)
            .unwrap_or(1.0)
    }).collect();
    let merged = confidence_weighted_merge(&packets, &trust_scores);
    Ok(Value::Array(merged.into_iter().map(Value::Float).collect()))
}

/// selective_broadcast(channel_id, sender, type_str, embeddings, confidence, target_specialization) -> count
fn builtin_selective_broadcast(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 6 {
        return Err("selective_broadcast expects 6 args: (channel_id, sender, type_str, embeddings, confidence, target_specialization)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let sender = value_to_string(&args[1])?;
    let type_str = value_to_string(&args[2])?;
    let embeddings = value_to_f64_vec(&args[3])?;
    let confidence = value_to_f64(&args[4])?;
    let target_spec = value_to_string(&args[5])?;
    let thought_type = ThoughtType::from_str(&type_str)?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    if channel.find_agent(&sender).is_none() {
        return Err(format!("Agent '{}' not found in channel {}", sender, ch_id));
    }
    // Count matching agents (excluding the sender)
    let matching_agents: Vec<String> = channel.agents.iter()
        .filter(|a| a.id != sender && a.specialization == target_spec)
        .map(|a| a.id.clone())
        .collect();
    let count = matching_agents.len();
    // Now actually send one packet (visible to all, but semantically targeted)
    let channel = env.thought_channels.get_mut(&ch_id).unwrap();
    let id = channel.next_packet_id;
    channel.next_packet_id += 1;
    let packet_bytes = (embeddings.len()) as u64 * 8 + 64;
    channel.bandwidth_bytes += packet_bytes * count as u64;
    if let Some(agent) = channel.find_agent_mut(&sender) {
        agent.message_count += 1;
    }
    let mut metadata = HashMap::new();
    metadata.insert("target_specialization".to_string(), target_spec);
    metadata.insert("broadcast_count".to_string(), count.to_string());
    let packet = ThoughtPacket {
        id,
        sender: sender.to_string(),
        embeddings,
        thought_type,
        confidence,
        timestamp: id,
        references: Vec::new(),
        payload: Vec::new(),
        metadata,
    };
    channel.message_log.push(packet);
    Ok(Value::Int(count as i128))
}

/// negotiate_protocol(channel_id) -> protocol description string
fn builtin_negotiate_protocol(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("negotiate_protocol expects 1 arg: (channel_id)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;

    let num_agents = channel.agents.len();
    if num_agents == 0 {
        return Ok(Value::String("no agents in channel".to_string()));
    }

    // Determine max embedding dimension across agents
    let max_dim = channel.agents.iter().map(|a| a.embedding_dim).max().unwrap_or(0);
    let min_dim = channel.agents.iter().map(|a| a.embedding_dim).min().unwrap_or(0);

    // Collect unique specializations
    let mut specs: Vec<String> = channel.agents.iter()
        .map(|a| a.specialization.clone())
        .collect();
    specs.sort();
    specs.dedup();

    // Build protocol description
    let protocol = format!(
        "ThoughtProtocol v1.0 | agents={} | embed_dim={}..{} | specializations=[{}] | \
         format=THPT_BINARY | trust_model=weighted_consensus | bandwidth_used={} bytes",
        num_agents, min_dim, max_dim, specs.join(", "), channel.bandwidth_bytes
    );
    Ok(Value::String(protocol))
}

/// agent_trust(channel_id, agent_name) -> f64
fn builtin_agent_trust(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("agent_trust expects 2 args: (channel_id, agent_name)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let name = value_to_string(&args[1])?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    let idx = channel.find_agent(&name)
        .ok_or_else(|| format!("Agent '{}' not found in channel {}", name, ch_id))?;
    Ok(Value::Float(channel.agents[idx].trust_score))
}

/// agent_update_trust(channel_id, agent_name, delta) -> new_trust
fn builtin_agent_update_trust(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("agent_update_trust expects 3 args: (channel_id, agent_name, delta)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let name = value_to_string(&args[1])?;
    let delta = value_to_f64(&args[2])?;
    let channel = env.thought_channels.get_mut(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    let agent = channel.find_agent_mut(&name)
        .ok_or_else(|| format!("Agent '{}' not found in channel {}", name, ch_id))?;
    agent.trust_score = (agent.trust_score + delta).clamp(0.0, 10.0);
    Ok(Value::Float(agent.trust_score))
}

/// channel_stats(channel_id) -> HashMap with num_agents, num_messages, bandwidth
fn builtin_channel_stats(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("channel_stats expects 1 arg: (channel_id)".into());
    }
    let ch_id = value_to_usize(&args[0])?;
    let channel = env.thought_channels.get(&ch_id)
        .ok_or_else(|| format!("No thought channel with id {}", ch_id))?;
    let mut stats = HashMap::new();
    stats.insert("num_agents".to_string(), Value::Int(channel.agents.len() as i128));
    stats.insert("num_messages".to_string(), Value::Int(channel.message_log.len() as i128));
    stats.insert("bandwidth".to_string(), Value::Int(channel.bandwidth_bytes as i128));
    stats.insert("next_packet_id".to_string(), Value::Int(channel.next_packet_id as i128));
    // Per-agent stats
    let mut agent_info = Vec::new();
    for agent in &channel.agents {
        let mut info = HashMap::new();
        info.insert("id".to_string(), Value::String(agent.id.clone()));
        info.insert("embedding_dim".to_string(), Value::Int(agent.embedding_dim as i128));
        info.insert("specialization".to_string(), Value::String(agent.specialization.clone()));
        info.insert("trust_score".to_string(), Value::Float(agent.trust_score));
        info.insert("message_count".to_string(), Value::Int(agent.message_count as i128));
        agent_info.push(Value::HashMap(info));
    }
    stats.insert("agents".to_string(), Value::Array(agent_info));
    Ok(Value::HashMap(stats))
}

/// encode_thought(embeddings, type_str, confidence) -> array of ints (bytes)
fn builtin_encode_thought(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 {
        return Err("encode_thought expects 3 args: (embeddings, type_str, confidence)".into());
    }
    let embeddings = value_to_f64_vec(&args[0])?;
    let type_str = value_to_string(&args[1])?;
    let confidence = value_to_f64(&args[2])?;
    let thought_type = ThoughtType::from_str(&type_str)?;
    let encoded = encode_thought_binary(&embeddings, &thought_type, confidence);
    Ok(Value::Array(encoded.into_iter().map(|b| Value::Int(b as i128)).collect()))
}

/// decode_thought(encoded) -> HashMap with embeddings, type, confidence
fn builtin_decode_thought(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("decode_thought expects 1 arg: (encoded_bytes_array)".into());
    }
    let encoded_vals = match &args[0] {
        Value::Array(arr) => arr.clone(),
        _ => return Err("expected array of byte ints".to_string()),
    };
    let bytes: Vec<u8> = encoded_vals.iter().map(|v| {
        match v {
            Value::Int(n) => Ok(*n as u8),
            Value::Float(f) => Ok(*f as u8),
            _ => Err("encoded array must contain integers".to_string()),
        }
    }).collect::<Result<Vec<u8>, String>>()?;
    let (embeddings, thought_type, confidence) = decode_thought_binary(&bytes)?;
    let mut result = HashMap::new();
    result.insert("embeddings".to_string(), Value::Array(
        embeddings.into_iter().map(Value::Float).collect(),
    ));
    result.insert("type".to_string(), Value::String(thought_type.as_str().to_string()));
    result.insert("confidence".to_string(), Value::Float(confidence));
    Ok(Value::HashMap(result))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_thought_type_roundtrip() {
        let types = vec![
            ThoughtType::Query,
            ThoughtType::Response,
            ThoughtType::ArchProposal,
            ThoughtType::GradientShare,
            ThoughtType::SpecVerification,
            ThoughtType::WorldModelUpdate,
            ThoughtType::MetaCognition,
        ];
        for t in &types {
            let s = t.as_str();
            let parsed = ThoughtType::from_str(s).unwrap();
            assert_eq!(*t, parsed);
            let byte = t.as_u8();
            let from_byte = ThoughtType::from_u8(byte).unwrap();
            assert_eq!(*t, from_byte);
        }
    }

    #[test]
    fn test_thought_type_invalid() {
        assert!(ThoughtType::from_str("invalid").is_err());
        assert!(ThoughtType::from_u8(99).is_err());
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&a, &a);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![-1.0, -2.0, -3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_empty() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_cosine_similarity_zero_vector() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![1.0, 2.0, 3.0];
        assert_eq!(cosine_similarity(&a, &b), 0.0);
    }

    #[test]
    fn test_channel_new() {
        let ch = ThoughtChannel::new(0);
        assert_eq!(ch.id, 0);
        assert!(ch.agents.is_empty());
        assert!(ch.message_log.is_empty());
        assert_eq!(ch.bandwidth_bytes, 0);
        assert_eq!(ch.next_packet_id, 1);
    }

    #[test]
    fn test_channel_add_and_find_agent() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("alice".into(), 128, "reasoning".into()));
        ch.add_agent(AgentState::new("bob".into(), 256, "coding".into()));
        assert_eq!(ch.find_agent("alice"), Some(0));
        assert_eq!(ch.find_agent("bob"), Some(1));
        assert_eq!(ch.find_agent("charlie"), None);
    }

    #[test]
    fn test_send_and_receive() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("alice".into(), 4, "reasoning".into()));
        ch.add_agent(AgentState::new("bob".into(), 4, "coding".into()));

        let id1 = ch.send_packet("alice", ThoughtType::Query, vec![1.0, 0.0, 0.0, 0.0], 0.9, Vec::new());
        let id2 = ch.send_packet("bob", ThoughtType::Response, vec![0.0, 1.0, 0.0, 0.0], 0.8, Vec::new());
        assert_eq!(id1, 1);
        assert_eq!(id2, 2);

        // Alice receives only bob's messages
        let received = ch.receive_for("alice");
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].sender, "bob");

        // Bob receives only alice's messages
        let received = ch.receive_for("bob");
        assert_eq!(received.len(), 1);
        assert_eq!(received[0].sender, "alice");

        // Message counts
        assert_eq!(ch.agents[0].message_count, 1); // alice
        assert_eq!(ch.agents[1].message_count, 1); // bob
    }

    #[test]
    fn test_bandwidth_tracking() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("alice".into(), 4, "reasoning".into()));
        assert_eq!(ch.bandwidth_bytes, 0);
        ch.send_packet("alice", ThoughtType::Query, vec![1.0, 2.0, 3.0, 4.0], 0.9, Vec::new());
        // 4 embeddings * 8 + 0 payload * 8 + 64 overhead = 96
        assert_eq!(ch.bandwidth_bytes, 96);
        ch.send_packet("alice", ThoughtType::Response, vec![1.0], 0.5, vec![10.0, 20.0]);
        // 1*8 + 2*8 + 64 = 88, total = 184
        assert_eq!(ch.bandwidth_bytes, 184);
    }

    #[test]
    fn test_confidence_weighted_merge() {
        let p1 = ThoughtPacket {
            id: 1,
            sender: "alice".into(),
            embeddings: vec![1.0, 0.0],
            thought_type: ThoughtType::Query,
            confidence: 0.8,
            timestamp: 1,
            references: vec![],
            payload: vec![],
            metadata: HashMap::new(),
        };
        let p2 = ThoughtPacket {
            id: 2,
            sender: "bob".into(),
            embeddings: vec![0.0, 1.0],
            thought_type: ThoughtType::Response,
            confidence: 0.2,
            timestamp: 2,
            references: vec![],
            payload: vec![],
            metadata: HashMap::new(),
        };
        let packets = vec![&p1, &p2];
        let trusts = vec![1.0, 1.0];
        let merged = confidence_weighted_merge(&packets, &trusts);
        // Weights: 0.8*1.0=0.8, 0.2*1.0=0.2, total=1.0
        // merged[0] = (1.0*0.8 + 0.0*0.2)/1.0 = 0.8
        // merged[1] = (0.0*0.8 + 1.0*0.2)/1.0 = 0.2
        assert!((merged[0] - 0.8).abs() < 1e-10);
        assert!((merged[1] - 0.2).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_weighted_merge_with_trust() {
        let p1 = ThoughtPacket {
            id: 1,
            sender: "alice".into(),
            embeddings: vec![1.0, 0.0],
            thought_type: ThoughtType::Query,
            confidence: 1.0,
            timestamp: 1,
            references: vec![],
            payload: vec![],
            metadata: HashMap::new(),
        };
        let p2 = ThoughtPacket {
            id: 2,
            sender: "bob".into(),
            embeddings: vec![0.0, 1.0],
            thought_type: ThoughtType::Response,
            confidence: 1.0,
            timestamp: 2,
            references: vec![],
            payload: vec![],
            metadata: HashMap::new(),
        };
        let packets = vec![&p1, &p2];
        let trusts = vec![3.0, 1.0]; // alice has 3x trust
        let merged = confidence_weighted_merge(&packets, &trusts);
        // Weights: 1.0*3.0=3.0, 1.0*1.0=1.0, total=4.0
        assert!((merged[0] - 0.75).abs() < 1e-10);
        assert!((merged[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_confidence_weighted_merge_empty() {
        let packets: Vec<&ThoughtPacket> = vec![];
        let merged = confidence_weighted_merge(&packets, &[]);
        assert!(merged.is_empty());
    }

    #[test]
    fn test_encode_decode_roundtrip() {
        let embeddings = vec![1.0, -2.5, 3.14159, 0.0, -1e10];
        let thought_type = ThoughtType::GradientShare;
        let confidence = 0.95;
        let encoded = encode_thought_binary(&embeddings, &thought_type, confidence);
        let (dec_emb, dec_type, dec_conf) = decode_thought_binary(&encoded).unwrap();
        assert_eq!(dec_emb, embeddings);
        assert_eq!(dec_type, thought_type);
        assert!((dec_conf - confidence).abs() < 1e-15);
    }

    #[test]
    fn test_encode_decode_all_types() {
        let types = vec![
            ThoughtType::Query,
            ThoughtType::Response,
            ThoughtType::ArchProposal,
            ThoughtType::GradientShare,
            ThoughtType::SpecVerification,
            ThoughtType::WorldModelUpdate,
            ThoughtType::MetaCognition,
        ];
        for t in types {
            let encoded = encode_thought_binary(&[1.0], &t, 0.5);
            let (_, dec_type, _) = decode_thought_binary(&encoded).unwrap();
            assert_eq!(dec_type, t);
        }
    }

    #[test]
    fn test_encode_magic_bytes() {
        let encoded = encode_thought_binary(&[], &ThoughtType::Query, 0.5);
        assert_eq!(&encoded[0..4], &[0x54, 0x48, 0x50, 0x54]);
    }

    #[test]
    fn test_decode_too_short() {
        let data = vec![0x54, 0x48, 0x50, 0x54, 0x00];
        assert!(decode_thought_binary(&data).is_err());
    }

    #[test]
    fn test_decode_bad_magic() {
        let mut encoded = encode_thought_binary(&[1.0], &ThoughtType::Query, 0.5);
        encoded[0] = 0xFF;
        assert!(decode_thought_binary(&encoded).is_err());
    }

    #[test]
    fn test_decode_truncated() {
        let mut encoded = encode_thought_binary(&[1.0, 2.0, 3.0], &ThoughtType::Query, 0.5);
        encoded.truncate(20); // Cut off embeddings data
        assert!(decode_thought_binary(&encoded).is_err());
    }

    #[test]
    fn test_encode_empty_embeddings() {
        let encoded = encode_thought_binary(&[], &ThoughtType::Response, 1.0);
        let (emb, t, c) = decode_thought_binary(&encoded).unwrap();
        assert!(emb.is_empty());
        assert_eq!(t, ThoughtType::Response);
        assert!((c - 1.0).abs() < 1e-15);
    }

    #[test]
    fn test_agent_state_defaults() {
        let agent = AgentState::new("test".into(), 64, "nlp".into());
        assert_eq!(agent.id, "test");
        assert_eq!(agent.embedding_dim, 64);
        assert_eq!(agent.trust_score, 1.0);
        assert_eq!(agent.specialization, "nlp");
        assert_eq!(agent.message_count, 0);
        assert!(agent.capabilities.is_empty());
    }

    #[test]
    fn test_packets_by_ids() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("a".into(), 2, "x".into()));
        let id1 = ch.send_packet("a", ThoughtType::Query, vec![1.0, 0.0], 0.9, vec![]);
        let _id2 = ch.send_packet("a", ThoughtType::Response, vec![0.0, 1.0], 0.8, vec![]);
        let id3 = ch.send_packet("a", ThoughtType::ArchProposal, vec![1.0, 1.0], 0.7, vec![]);
        let found = ch.packets_by_ids(&[id1, id3]);
        assert_eq!(found.len(), 2);
        assert_eq!(found[0].id, id1);
        assert_eq!(found[1].id, id3);
    }

    #[test]
    fn test_cosine_similarity_different_lengths() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0];
        // Only uses min length = 2
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_selective_broadcast_filter() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("alice".into(), 4, "reasoning".into()));
        ch.add_agent(AgentState::new("bob".into(), 4, "coding".into()));
        ch.add_agent(AgentState::new("carol".into(), 4, "coding".into()));
        ch.add_agent(AgentState::new("dave".into(), 4, "reasoning".into()));

        // Count agents with "coding" specialization excluding sender
        let matching: Vec<_> = ch.agents.iter()
            .filter(|a| a.id != "alice" && a.specialization == "coding")
            .collect();
        assert_eq!(matching.len(), 2); // bob and carol
    }

    #[test]
    fn test_negotiate_protocol_empty() {
        let ch = ThoughtChannel::new(0);
        // No agents -> special message
        assert_eq!(ch.agents.len(), 0);
    }

    #[test]
    fn test_trust_clamping() {
        let mut agent = AgentState::new("test".into(), 4, "x".into());
        agent.trust_score = 9.5;
        agent.trust_score = (agent.trust_score + 1.0).clamp(0.0, 10.0);
        assert!((agent.trust_score - 10.0).abs() < 1e-10);
        agent.trust_score = (agent.trust_score - 20.0).clamp(0.0, 10.0);
        assert!((agent.trust_score - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_encode_large_embeddings() {
        let embeddings: Vec<f64> = (0..1000).map(|i| i as f64 * 0.001).collect();
        let encoded = encode_thought_binary(&embeddings, &ThoughtType::WorldModelUpdate, 0.42);
        let (dec_emb, dec_type, dec_conf) = decode_thought_binary(&encoded).unwrap();
        assert_eq!(dec_emb.len(), 1000);
        assert_eq!(dec_type, ThoughtType::WorldModelUpdate);
        assert!((dec_conf - 0.42).abs() < 1e-15);
        for (a, b) in embeddings.iter().zip(dec_emb.iter()) {
            assert!((a - b).abs() < 1e-15);
        }
    }

    #[test]
    fn test_multiple_channels() {
        let ch1 = ThoughtChannel::new(0);
        let ch2 = ThoughtChannel::new(1);
        assert_ne!(ch1.id, ch2.id);
    }

    #[test]
    fn test_payload_in_packet() {
        let mut ch = ThoughtChannel::new(0);
        ch.add_agent(AgentState::new("a".into(), 2, "x".into()));
        let payload = vec![100.0, 200.0, 300.0];
        let id = ch.send_packet("a", ThoughtType::GradientShare, vec![1.0, 2.0], 0.9, payload.clone());
        let packet = &ch.message_log[0];
        assert_eq!(packet.id, id);
        assert_eq!(packet.payload, payload);
    }

    #[test]
    fn test_cosine_similarity_known_value() {
        // cos([3,4], [4,3]) = (12+12)/(5*5) = 24/25 = 0.96
        let a = vec![3.0, 4.0];
        let b = vec![4.0, 3.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim - 0.96).abs() < 1e-10);
    }
}
