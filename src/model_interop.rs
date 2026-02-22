// Model interop: load/save/convert models between ONNX, SafeTensors, NumPy, and Vortex formats.
// No external protobuf or ONNX crate dependencies.

use std::collections::HashMap;

// ─── Core Types ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DType {
    F32,
    F64,
    I32,
    I64,
    U8,
    F16,
    BF16,
}

impl DType {
    pub fn byte_size(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F64 => 8,
            DType::I32 => 4,
            DType::I64 => 8,
            DType::U8 => 1,
            DType::F16 => 2,
            DType::BF16 => 2,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::F64 => "F64",
            DType::I32 => "I32",
            DType::I64 => "I64",
            DType::U8 => "U8",
            DType::F16 => "F16",
            DType::BF16 => "BF16",
        }
    }

    pub fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "F32" | "f32" | "float32" => Ok(DType::F32),
            "F64" | "f64" | "float64" => Ok(DType::F64),
            "I32" | "i32" | "int32" => Ok(DType::I32),
            "I64" | "i64" | "int64" => Ok(DType::I64),
            "U8" | "u8" | "uint8" => Ok(DType::U8),
            "F16" | "f16" | "float16" => Ok(DType::F16),
            "BF16" | "bf16" | "bfloat16" => Ok(DType::BF16),
            _ => Err(format!("unknown dtype: {}", s)),
        }
    }

    /// ONNX TensorProto.DataType enum value
    pub fn from_onnx_type(t: u64) -> Result<Self, String> {
        match t {
            1 => Ok(DType::F32),
            2 => Ok(DType::U8),
            6 => Ok(DType::I32),
            7 => Ok(DType::I64),
            10 => Ok(DType::F16),
            11 => Ok(DType::F64),
            16 => Ok(DType::BF16),
            _ => Err(format!("unsupported ONNX data type: {}", t)),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorData {
    pub dtype: DType,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl TensorData {
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn to_f64_vec(&self) -> Vec<f64> {
        match self.dtype {
            DType::F32 => self.data.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect(),
            DType::F64 => self.data.chunks(8)
                .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]))
                .collect(),
            DType::I32 => self.data.chunks(4)
                .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64)
                .collect(),
            DType::I64 => self.data.chunks(8)
                .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f64)
                .collect(),
            DType::U8 => self.data.iter().map(|&b| b as f64).collect(),
            DType::F16 | DType::BF16 => {
                // Minimal f16 conversion
                self.data.chunks(2).map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    if self.dtype == DType::F16 {
                        f16_to_f64(bits)
                    } else {
                        bf16_to_f64(bits)
                    }
                }).collect()
            }
        }
    }

    pub fn from_f64_slice(shape: Vec<usize>, data: &[f64], dtype: DType) -> Self {
        let bytes = match dtype {
            DType::F32 => data.iter().flat_map(|&v| (v as f32).to_le_bytes()).collect(),
            DType::F64 => data.iter().flat_map(|&v| v.to_le_bytes()).collect(),
            DType::I32 => data.iter().flat_map(|&v| (v as i32).to_le_bytes()).collect(),
            DType::I64 => data.iter().flat_map(|&v| (v as i64).to_le_bytes()).collect(),
            DType::U8 => data.iter().map(|&v| v as u8).collect(),
            DType::F16 => data.iter().flat_map(|&v| f64_to_f16(v).to_le_bytes()).collect(),
            DType::BF16 => data.iter().flat_map(|&v| f64_to_bf16(v).to_le_bytes()).collect(),
        };
        TensorData { dtype, shape, data: bytes }
    }

    pub fn from_f32_bytes(shape: Vec<usize>, raw: Vec<u8>) -> Self {
        TensorData { dtype: DType::F32, shape, data: raw }
    }
}

fn f16_to_f64(bits: u16) -> f64 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x3FF) as u32;
    if exp == 0 && frac == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
    if exp == 0x1F { return if frac == 0 { if sign == 1 { f64::NEG_INFINITY } else { f64::INFINITY } } else { f64::NAN }; }
    let f32_bits = (sign << 31) | ((exp + 112) << 23) | (frac << 13);
    (f32::from_bits(f32_bits)) as f64
}

fn bf16_to_f64(bits: u16) -> f64 {
    let f32_bits = (bits as u32) << 16;
    f32::from_bits(f32_bits) as f64
}

fn f64_to_f16(v: f64) -> u16 {
    let f = v as f32;
    let bits = f.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;
    if exp == 0 { return (sign << 15) as u16; }
    if exp == 0xFF { return ((sign << 15) | 0x7C00 | if frac != 0 { 0x200 } else { 0 }) as u16; }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return ((sign << 15) | 0x7C00) as u16; }
    if new_exp <= 0 { return (sign << 15) as u16; }
    ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
}

fn f64_to_bf16(v: f64) -> u16 {
    let f = v as f32;
    (f.to_bits() >> 16) as u16
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ModelFormat {
    SafeTensors,
    Onnx,
    Npy,
    Npz,
    VortexNative,
    VortexJson,
}

impl ModelFormat {
    pub fn from_extension(path: &str) -> Result<Self, String> {
        let p = path.to_lowercase();
        if p.ends_with(".safetensors") { Ok(ModelFormat::SafeTensors) }
        else if p.ends_with(".onnx") { Ok(ModelFormat::Onnx) }
        else if p.ends_with(".npy") { Ok(ModelFormat::Npy) }
        else if p.ends_with(".npz") { Ok(ModelFormat::Npz) }
        else if p.ends_with(".vxm") { Ok(ModelFormat::VortexNative) }
        else if p.ends_with(".json") { Ok(ModelFormat::VortexJson) }
        else { Err(format!("unknown model format for: {}", path)) }
    }

    pub fn name(&self) -> &'static str {
        match self {
            ModelFormat::SafeTensors => "SafeTensors",
            ModelFormat::Onnx => "ONNX",
            ModelFormat::Npy => "NumPy .npy",
            ModelFormat::Npz => "NumPy .npz",
            ModelFormat::VortexNative => "Vortex .vxm",
            ModelFormat::VortexJson => "Vortex JSON",
        }
    }
}


// ─── SafeTensors ────────────────────────────────────────────────────────

pub fn load_safetensors(path: &str) -> Result<HashMap<String, TensorData>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    parse_safetensors(&data)
}

pub fn parse_safetensors(data: &[u8]) -> Result<HashMap<String, TensorData>, String> {
    if data.len() < 8 { return Err("safetensors file too small".into()); }
    let header_size = u64::from_le_bytes([data[0],data[1],data[2],data[3],data[4],data[5],data[6],data[7]]) as usize;
    if header_size.checked_add(8).map_or(true, |total| total > data.len()) {
        return Err("safetensors header size exceeds file".into());
    }
    let header_json = std::str::from_utf8(&data[8..8+header_size])
        .map_err(|e| format!("invalid header utf8: {}", e))?;
    let header: serde_json::Value = serde_json::from_str(header_json)
        .map_err(|e| format!("invalid header JSON: {}", e))?;
    let obj = header.as_object().ok_or("header not an object")?;
    let data_start = 8 + header_size;
    let mut result = HashMap::new();
    for (name, info) in obj {
        if name == "__metadata__" { continue; }
        let info = info.as_object().ok_or(format!("tensor {} info not object", name))?;
        let dtype_str = info.get("dtype").and_then(|v| v.as_str()).ok_or("missing dtype")?;
        let dtype = DType::from_str(dtype_str)?;
        let shape: Vec<usize> = info.get("shape").and_then(|v| v.as_array())
            .ok_or("missing shape")?
            .iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect();
        let offsets = info.get("data_offsets").and_then(|v| v.as_array()).ok_or("missing data_offsets")?;
        let start = offsets[0].as_u64().unwrap_or(0) as usize;
        let end = offsets[1].as_u64().unwrap_or(0) as usize;
        if data_start + end > data.len() {
            return Err(format!("tensor {} data out of bounds", name));
        }
        let tensor_bytes = data[data_start+start..data_start+end].to_vec();
        result.insert(name.clone(), TensorData { dtype, shape, data: tensor_bytes });
    }
    Ok(result)
}

pub fn save_safetensors(tensors: &HashMap<String, TensorData>, path: &str) -> Result<(), String> {
    let bytes = serialize_safetensors(tensors)?;
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {}", path, e))
}

pub fn serialize_safetensors(tensors: &HashMap<String, TensorData>) -> Result<Vec<u8>, String> {
    // Build header and data
    let mut sorted_names: Vec<&String> = tensors.keys().collect();
    sorted_names.sort();
    let mut header_map = serde_json::Map::new();
    let mut offset: usize = 0;
    let mut data_buf: Vec<u8> = Vec::new();
    for name in &sorted_names {
        let t = &tensors[*name];
        let byte_len = t.data.len();
        let mut info = serde_json::Map::new();
        info.insert("dtype".into(), serde_json::Value::String(t.dtype.as_str().to_string()));
        info.insert("shape".into(), serde_json::Value::Array(
            t.shape.iter().map(|&s| serde_json::Value::Number(serde_json::Number::from(s as u64))).collect()
        ));
        info.insert("data_offsets".into(), serde_json::Value::Array(vec![
            serde_json::Value::Number(serde_json::Number::from(offset as u64)),
            serde_json::Value::Number(serde_json::Number::from((offset + byte_len) as u64)),
        ]));
        header_map.insert((*name).clone(), serde_json::Value::Object(info));
        data_buf.extend_from_slice(&t.data);
        offset += byte_len;
    }
    let header_json = serde_json::to_string(&serde_json::Value::Object(header_map))
        .map_err(|e| format!("serialize header: {}", e))?;
    let header_bytes = header_json.as_bytes();
    let header_size = header_bytes.len() as u64;
    let mut out = Vec::with_capacity(8 + header_bytes.len() + data_buf.len());
    out.extend_from_slice(&header_size.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&data_buf);
    Ok(out)
}


// ─── Minimal Protobuf Reader ────────────────────────────────────────────

struct ProtobufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> ProtobufReader<'a> {
    fn new(data: &'a [u8]) -> Self { Self { data, pos: 0 } }

    fn remaining(&self) -> usize { self.data.len() - self.pos }
    fn is_empty(&self) -> bool { self.pos >= self.data.len() }

    fn read_varint(&mut self) -> Result<u64, String> {
        let mut result: u64 = 0;
        let mut shift = 0u32;
        loop {
            if self.pos >= self.data.len() { return Err("varint: unexpected end".into()); }
            let b = self.data[self.pos];
            self.pos += 1;
            result |= ((b & 0x7F) as u64) << shift;
            if b & 0x80 == 0 { return Ok(result); }
            shift += 7;
            if shift >= 64 { return Err("varint too long".into()); }
        }
    }

    fn read_tag(&mut self) -> Result<(u32, u32), String> {
        let v = self.read_varint()?;
        Ok(((v >> 3) as u32, (v & 7) as u32))
    }

    fn read_bytes(&mut self) -> Result<&'a [u8], String> {
        let len = self.read_varint()? as usize;
        if self.pos + len > self.data.len() { return Err("bytes: overflow".into()); }
        let s = &self.data[self.pos..self.pos+len];
        self.pos += len;
        Ok(s)
    }

    fn read_fixed32(&mut self) -> Result<u32, String> {
        if self.pos + 4 > self.data.len() { return Err("fixed32: overflow".into()); }
        let v = u32::from_le_bytes([self.data[self.pos], self.data[self.pos+1], self.data[self.pos+2], self.data[self.pos+3]]);
        self.pos += 4;
        Ok(v)
    }

    fn read_fixed64(&mut self) -> Result<u64, String> {
        if self.pos + 8 > self.data.len() { return Err("fixed64: overflow".into()); }
        let v = u64::from_le_bytes([
            self.data[self.pos], self.data[self.pos+1], self.data[self.pos+2], self.data[self.pos+3],
            self.data[self.pos+4], self.data[self.pos+5], self.data[self.pos+6], self.data[self.pos+7],
        ]);
        self.pos += 8;
        Ok(v)
    }

    fn skip_field(&mut self, wire_type: u32) -> Result<(), String> {
        match wire_type {
            0 => { self.read_varint()?; }
            1 => { self.read_fixed64()?; }
            2 => { self.read_bytes()?; }
            5 => { self.read_fixed32()?; }
            _ => return Err(format!("unknown wire type: {}", wire_type)),
        }
        Ok(())
    }
}


// ─── ONNX Format ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct OnnxModel {
    pub opset_version: u64,
    pub graph: OnnxGraph,
}

#[derive(Debug, Clone)]
pub struct OnnxGraph {
    pub name: String,
    pub nodes: Vec<OnnxNode>,
    pub initializers: HashMap<String, TensorData>,
    pub inputs: Vec<OnnxValueInfo>,
    pub outputs: Vec<OnnxValueInfo>,
}

#[derive(Debug, Clone)]
pub struct OnnxNode {
    pub op_type: String,
    pub name: String,
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub attributes: HashMap<String, OnnxAttr>,
}

#[derive(Debug, Clone)]
pub enum OnnxAttr {
    Int(i64),
    Float(f32),
    String(String),
    Ints(Vec<i64>),
    Floats(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct OnnxValueInfo {
    pub name: String,
}

fn parse_onnx_tensor(data: &[u8]) -> Result<(String, TensorData), String> {
    let mut reader = ProtobufReader::new(data);
    let mut dims: Vec<usize> = Vec::new();
    let mut data_type: u64 = 1; // default F32
    let mut name = String::new();
    let mut raw_data: Option<Vec<u8>> = None;
    let mut float_data: Vec<f32> = Vec::new();
    let mut int32_data: Vec<i32> = Vec::new();
    let mut int64_data: Vec<i64> = Vec::new();
    let mut double_data: Vec<f64> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, 0) => dims.push(reader.read_varint()? as usize), // dims
            (2, 0) => data_type = reader.read_varint()?,          // data_type
            (8, 2) => name = String::from_utf8_lossy(reader.read_bytes()?).to_string(), // name
            (13, 2) => raw_data = Some(reader.read_bytes()?.to_vec()), // raw_data
            (4, 2) => {
                // packed float_data
                let bytes = reader.read_bytes()?;
                for c in bytes.chunks(4) {
                    if c.len() == 4 { float_data.push(f32::from_le_bytes([c[0],c[1],c[2],c[3]])); }
                }
            }
            (4, 5) => float_data.push(f32::from_bits(reader.read_fixed32()?)),
            (5, 2) => {
                let bytes = reader.read_bytes()?;
                for c in bytes.chunks(4) {
                    if c.len() == 4 { int32_data.push(i32::from_le_bytes([c[0],c[1],c[2],c[3]])); }
                }
            }
            (5, 0) => int32_data.push(reader.read_varint()? as i32),
            (7, 2) => {
                let bytes = reader.read_bytes()?;
                for c in bytes.chunks(8) {
                    if c.len() == 8 { int64_data.push(i64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])); }
                }
            }
            (7, 0) => int64_data.push(reader.read_varint()? as i64),
            (10, 2) => {
                let bytes = reader.read_bytes()?;
                for c in bytes.chunks(8) {
                    if c.len() == 8 { double_data.push(f64::from_le_bytes([c[0],c[1],c[2],c[3],c[4],c[5],c[6],c[7]])); }
                }
            }
            (10, 1) => double_data.push(f64::from_bits(reader.read_fixed64()?)),
            _ => reader.skip_field(wire)?,
        }
    }

    let dtype = DType::from_onnx_type(data_type)?;
    let tensor_data = if let Some(raw) = raw_data {
        raw
    } else if !float_data.is_empty() {
        float_data.iter().flat_map(|f| f.to_le_bytes()).collect()
    } else if !int32_data.is_empty() {
        int32_data.iter().flat_map(|i| i.to_le_bytes()).collect()
    } else if !int64_data.is_empty() {
        int64_data.iter().flat_map(|i| i.to_le_bytes()).collect()
    } else if !double_data.is_empty() {
        double_data.iter().flat_map(|d| d.to_le_bytes()).collect()
    } else {
        Vec::new()
    };

    Ok((name, TensorData { dtype, shape: dims, data: tensor_data }))
}

fn parse_onnx_node(data: &[u8]) -> Result<OnnxNode, String> {
    let mut reader = ProtobufReader::new(data);
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut name = String::new();
    let mut op_type = String::new();
    let mut attributes = HashMap::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, 2) => inputs.push(String::from_utf8_lossy(reader.read_bytes()?).to_string()),
            (2, 2) => outputs.push(String::from_utf8_lossy(reader.read_bytes()?).to_string()),
            (3, 2) => name = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            (4, 2) => op_type = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            (5, 2) => {
                // AttributeProto
                let attr_data = reader.read_bytes()?;
                if let Ok((aname, aval)) = parse_onnx_attribute(attr_data) {
                    attributes.insert(aname, aval);
                }
            }
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(OnnxNode { op_type, name, inputs, outputs, attributes })
}

fn parse_onnx_attribute(data: &[u8]) -> Result<(String, OnnxAttr), String> {
    let mut reader = ProtobufReader::new(data);
    let mut name = String::new();
    let mut attr_type: u64 = 0;
    let mut i_val: i64 = 0;
    let mut f_val: f32 = 0.0;
    let mut s_val = String::new();
    let mut ints: Vec<i64> = Vec::new();
    let mut floats: Vec<f32> = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, 2) => name = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            (2, 5) => f_val = f32::from_bits(reader.read_fixed32()?),
            (3, 0) => i_val = reader.read_varint()? as i64,
            (4, 2) => s_val = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            (7, 2) => {
                let bytes = reader.read_bytes()?;
                let mut sub = ProtobufReader::new(bytes);
                while !sub.is_empty() { ints.push(sub.read_varint()? as i64); }
            }
            (7, 0) => ints.push(reader.read_varint()? as i64),
            (8, 2) => {
                let bytes = reader.read_bytes()?;
                for c in bytes.chunks(4) {
                    if c.len() == 4 { floats.push(f32::from_le_bytes([c[0],c[1],c[2],c[3]])); }
                }
            }
            (8, 5) => floats.push(f32::from_bits(reader.read_fixed32()?)),
            (20, 0) => attr_type = reader.read_varint()?,
            _ => reader.skip_field(wire)?,
        }
    }

    let val = match attr_type {
        1 => OnnxAttr::Float(f_val),
        2 => OnnxAttr::Int(i_val),
        3 => OnnxAttr::String(s_val),
        6 => OnnxAttr::Floats(floats),
        7 => OnnxAttr::Ints(ints),
        _ => {
            // Infer from what we found
            if !ints.is_empty() { OnnxAttr::Ints(ints) }
            else if !floats.is_empty() { OnnxAttr::Floats(floats) }
            else if i_val != 0 { OnnxAttr::Int(i_val) }
            else if f_val != 0.0 { OnnxAttr::Float(f_val) }
            else if !s_val.is_empty() { OnnxAttr::String(s_val) }
            else { OnnxAttr::Int(0) }
        }
    };

    Ok((name, val))
}

fn parse_onnx_value_info(data: &[u8]) -> Result<OnnxValueInfo, String> {
    let mut reader = ProtobufReader::new(data);
    let mut name = String::new();
    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, 2) => name = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(OnnxValueInfo { name })
}

fn parse_onnx_graph(data: &[u8]) -> Result<OnnxGraph, String> {
    let mut reader = ProtobufReader::new(data);
    let mut name = String::new();
    let mut nodes = Vec::new();
    let mut initializers = HashMap::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (1, 2) => {
                let node_data = reader.read_bytes()?;
                nodes.push(parse_onnx_node(node_data)?);
            }
            (2, 2) => name = String::from_utf8_lossy(reader.read_bytes()?).to_string(),
            (5, 2) => {
                let tensor_data = reader.read_bytes()?;
                let (tname, td) = parse_onnx_tensor(tensor_data)?;
                initializers.insert(tname, td);
            }
            (11, 2) => {
                let vi_data = reader.read_bytes()?;
                inputs.push(parse_onnx_value_info(vi_data)?);
            }
            (12, 2) => {
                let vi_data = reader.read_bytes()?;
                outputs.push(parse_onnx_value_info(vi_data)?);
            }
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(OnnxGraph { name, nodes, initializers, inputs, outputs })
}

pub fn load_onnx(path: &str) -> Result<OnnxModel, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    parse_onnx_model(&data)
}

fn parse_onnx_model(data: &[u8]) -> Result<OnnxModel, String> {
    let mut reader = ProtobufReader::new(data);
    let mut opset_version: u64 = 0;
    let mut graph = None;

    while !reader.is_empty() {
        let (field, wire) = reader.read_tag()?;
        match (field, wire) {
            (7, 2) => graph = Some(parse_onnx_graph(reader.read_bytes()?)?),
            (8, 2) => {
                // opset_import
                let oi_data = reader.read_bytes()?;
                let mut sub = ProtobufReader::new(oi_data);
                while !sub.is_empty() {
                    let (f, w) = sub.read_tag()?;
                    match (f, w) {
                        (2, 0) => opset_version = sub.read_varint()?,
                        _ => sub.skip_field(w)?,
                    }
                }
            }
            _ => reader.skip_field(wire)?,
        }
    }
    Ok(OnnxModel {
        opset_version,
        graph: graph.ok_or("ONNX model has no graph")?,
    })
}

pub fn onnx_to_vortex(onnx: &OnnxModel) -> crate::nn::Model {
    use crate::nn::{Model, Layer, Tensor};
    let mut layers = Vec::new();

    for node in &onnx.graph.nodes {
        match node.op_type.as_str() {
            "MatMul" | "Gemm" => {
                // Try to find weight initializer
                if node.inputs.len() >= 2 {
                    if let Some(w) = onnx.graph.initializers.get(&node.inputs[1]) {
                        let f64_data = w.to_f64_vec();
                        let (in_f, out_f) = if w.shape.len() == 2 { (w.shape[0], w.shape[1]) } else { (f64_data.len(), 1) };
                        let weight = Tensor::new(vec![in_f, out_f], f64_data);
                        let bias = if node.inputs.len() > 2 {
                            if let Some(b) = onnx.graph.initializers.get(&node.inputs[2]) {
                                Tensor::new(vec![b.num_elements()], b.to_f64_vec())
                            } else { Tensor::zeros(vec![out_f]) }
                        } else { Tensor::zeros(vec![out_f]) };
                        layers.push(Layer::Linear {
                            weight_grad: Tensor::zeros(vec![in_f, out_f]),
                            bias_grad: Tensor::zeros(vec![out_f]),
                            weight, bias, cache: None,
                        });
                    }
                }
            }
            "Add" => {} // Bias fused into linear
            "Relu" => layers.push(Layer::ReLU { cache: None }),
            "Sigmoid" => layers.push(Layer::Sigmoid { cache: None }),
            "Tanh" => layers.push(Layer::Tanh { cache: None }),
            "Softmax" => layers.push(Layer::Softmax { cache: None }),
            "LayerNormalization" => {
                if let Some(gamma) = node.inputs.get(1).and_then(|n| onnx.graph.initializers.get(n)) {
                    let dim = gamma.num_elements();
                    let gamma_t = Tensor::new(vec![dim], gamma.to_f64_vec());
                    let beta_t = if let Some(beta) = node.inputs.get(2).and_then(|n| onnx.graph.initializers.get(n)) {
                        Tensor::new(vec![dim], beta.to_f64_vec())
                    } else { Tensor::zeros(vec![dim]) };
                    layers.push(Layer::LayerNorm {
                        gamma: gamma_t, beta: beta_t,
                        gamma_grad: Tensor::zeros(vec![dim]),
                        beta_grad: Tensor::zeros(vec![dim]),
                        dim, cache: None,
                    });
                }
            }
            _ => {} // Skip unsupported ops
        }
    }

    if layers.is_empty() {
        layers.push(Layer::linear(1, 1));
    }
    Model { layers, name: onnx.graph.name.clone() }
}


// ─── NumPy .npy/.npz ───────────────────────────────────────────────────

const NPY_MAGIC: &[u8] = b"\x93NUMPY";

pub fn load_npy(path: &str) -> Result<TensorData, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    parse_npy(&data)
}

pub fn parse_npy(data: &[u8]) -> Result<TensorData, String> {
    if data.len() < 10 || &data[0..6] != NPY_MAGIC {
        return Err("not a valid .npy file".into());
    }
    let major = data[6];
    let _minor = data[7];
    let header_len = if major == 1 {
        u16::from_le_bytes([data[8], data[9]]) as usize
    } else {
        u32::from_le_bytes([data[8], data[9], data[10], data[11]]) as usize
    };
    let header_start = if major == 1 { 10 } else { 12 };
    let header = std::str::from_utf8(&data[header_start..header_start+header_len])
        .map_err(|_| "invalid npy header utf8")?;
    let data_start = header_start + header_len;

    // Parse header dict: {'descr': '<f4', 'fortran_order': False, 'shape': (3, 4), }
    let dtype = parse_npy_descr(header)?;
    let shape = parse_npy_shape(header)?;
    let raw = data[data_start..].to_vec();

    Ok(TensorData { dtype, shape, data: raw })
}

fn parse_npy_descr(header: &str) -> Result<DType, String> {
    let descr_start = header.find("'descr'").or_else(|| header.find("\"descr\""))
        .ok_or("npy: missing descr")?;
    let rest = &header[descr_start..];
    // Find the value between quotes after the colon
    let colon = rest.find(':').ok_or("npy: no colon after descr")?;
    let after = &rest[colon+1..];
    let q1 = after.find('\'').or_else(|| after.find('"')).ok_or("npy: no quote")?;
    let qchar = after.as_bytes()[q1] as char;
    let q2 = after[q1+1..].find(qchar).ok_or("npy: no closing quote")?;
    let descr = &after[q1+1..q1+1+q2];
    match descr {
        "<f4" | "=f4" | "f4" => Ok(DType::F32),
        "<f8" | "=f8" | "f8" => Ok(DType::F64),
        "<i4" | "=i4" | "i4" => Ok(DType::I32),
        "<i8" | "=i8" | "i8" => Ok(DType::I64),
        "|u1" | "u1" => Ok(DType::U8),
        "<f2" | "=f2" | "f2" => Ok(DType::F16),
        _ => Err(format!("unsupported npy descr: {}", descr)),
    }
}

fn parse_npy_shape(header: &str) -> Result<Vec<usize>, String> {
    let shape_start = header.find("'shape'").or_else(|| header.find("\"shape\""))
        .ok_or("npy: missing shape")?;
    let rest = &header[shape_start..];
    let paren_start = rest.find('(').ok_or("npy: no ( in shape")?;
    let paren_end = rest.find(')').ok_or("npy: no ) in shape")?;
    let inner = rest[paren_start+1..paren_end].trim();
    if inner.is_empty() {
        return Ok(vec![]); // scalar
    }
    inner.split(',').filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<usize>().map_err(|e| format!("npy shape parse: {}", e)))
        .collect()
}

pub fn save_npy(tensor: &TensorData, path: &str) -> Result<(), String> {
    let bytes = serialize_npy(tensor)?;
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {}", path, e))
}

pub fn serialize_npy(tensor: &TensorData) -> Result<Vec<u8>, String> {
    let descr = match tensor.dtype {
        DType::F32 => "<f4",
        DType::F64 => "<f8",
        DType::I32 => "<i4",
        DType::I64 => "<i8",
        DType::U8 => "|u1",
        DType::F16 => "<f2",
        DType::BF16 => "<f2", // approximate
    };
    let shape_str = if tensor.shape.is_empty() {
        "()".to_string()
    } else if tensor.shape.len() == 1 {
        format!("({},)", tensor.shape[0])
    } else {
        format!("({})", tensor.shape.iter().map(|s| s.to_string()).collect::<Vec<_>>().join(", "))
    };
    let header_dict = format!("{{'descr': '{}', 'fortran_order': False, 'shape': {}, }}", descr, shape_str);
    // Pad to align to 64 bytes
    let prefix_len = 10; // magic(6) + version(2) + header_len(2)
    let total = prefix_len + header_dict.len() + 1; // +1 for newline
    let padding = (64 - (total % 64)) % 64;
    let header_len = header_dict.len() + padding + 1;

    let mut out = Vec::new();
    out.extend_from_slice(NPY_MAGIC);
    out.push(1); // major
    out.push(0); // minor
    out.extend_from_slice(&(header_len as u16).to_le_bytes());
    out.extend_from_slice(header_dict.as_bytes());
    for _ in 0..padding { out.push(b' '); }
    out.push(b'\n');
    out.extend_from_slice(&tensor.data);
    Ok(out)
}

pub fn load_npz(path: &str) -> Result<HashMap<String, TensorData>, String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    parse_npz(&data)
}

fn parse_npz(data: &[u8]) -> Result<HashMap<String, TensorData>, String> {
    // Minimal ZIP reader: parse local file headers
    let mut result = HashMap::new();
    let mut pos = 0;
    while pos + 30 <= data.len() {
        // Local file header signature
        if data[pos..pos+4] != [0x50, 0x4B, 0x03, 0x04] { break; }
        let _version = u16::from_le_bytes([data[pos+4], data[pos+5]]);
        let _flags = u16::from_le_bytes([data[pos+6], data[pos+7]]);
        let compression = u16::from_le_bytes([data[pos+8], data[pos+9]]);
        let compressed_size = u32::from_le_bytes([data[pos+18], data[pos+19], data[pos+20], data[pos+21]]) as usize;
        let _uncompressed_size = u32::from_le_bytes([data[pos+22], data[pos+23], data[pos+24], data[pos+25]]) as usize;
        let name_len = u16::from_le_bytes([data[pos+26], data[pos+27]]) as usize;
        let extra_len = u16::from_le_bytes([data[pos+28], data[pos+29]]) as usize;
        let name = String::from_utf8_lossy(&data[pos+30..pos+30+name_len]).to_string();
        let file_start = pos + 30 + name_len + extra_len;
        let file_data = &data[file_start..file_start+compressed_size];

        if compression == 0 && name.ends_with(".npy") {
            // Stored (uncompressed)
            let tensor_name = name.trim_end_matches(".npy").to_string();
            result.insert(tensor_name, parse_npy(file_data)?);
        }

        pos = file_start + compressed_size;
    }
    Ok(result)
}

pub fn save_npz(tensors: &HashMap<String, TensorData>, path: &str) -> Result<(), String> {
    let bytes = serialize_npz(tensors)?;
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {}", path, e))
}

fn serialize_npz(tensors: &HashMap<String, TensorData>) -> Result<Vec<u8>, String> {
    // Write as stored (uncompressed) ZIP
    let mut sorted_names: Vec<&String> = tensors.keys().collect();
    sorted_names.sort();
    let mut out = Vec::new();
    let mut central_dir = Vec::new();
    let mut cd_entries = 0u16;

    for name in &sorted_names {
        let npy_bytes = serialize_npy(&tensors[*name])?;
        let fname = format!("{}.npy", name);
        let fname_bytes = fname.as_bytes();
        let offset = out.len() as u32;

        // Local file header
        out.extend_from_slice(&[0x50, 0x4B, 0x03, 0x04]); // signature
        out.extend_from_slice(&20u16.to_le_bytes()); // version needed
        out.extend_from_slice(&0u16.to_le_bytes());  // flags
        out.extend_from_slice(&0u16.to_le_bytes());  // compression (stored)
        out.extend_from_slice(&0u16.to_le_bytes());  // mod time
        out.extend_from_slice(&0u16.to_le_bytes());  // mod date
        out.extend_from_slice(&crc32_simple(&npy_bytes).to_le_bytes()); // crc32
        out.extend_from_slice(&(npy_bytes.len() as u32).to_le_bytes()); // compressed size
        out.extend_from_slice(&(npy_bytes.len() as u32).to_le_bytes()); // uncompressed size
        out.extend_from_slice(&(fname_bytes.len() as u16).to_le_bytes()); // name len
        out.extend_from_slice(&0u16.to_le_bytes()); // extra len
        out.extend_from_slice(fname_bytes);
        out.extend_from_slice(&npy_bytes);

        // Central directory entry
        central_dir.extend_from_slice(&[0x50, 0x4B, 0x01, 0x02]);
        central_dir.extend_from_slice(&20u16.to_le_bytes()); // version made by
        central_dir.extend_from_slice(&20u16.to_le_bytes()); // version needed
        central_dir.extend_from_slice(&0u16.to_le_bytes());  // flags
        central_dir.extend_from_slice(&0u16.to_le_bytes());  // compression
        central_dir.extend_from_slice(&0u16.to_le_bytes());  // mod time
        central_dir.extend_from_slice(&0u16.to_le_bytes());  // mod date
        central_dir.extend_from_slice(&crc32_simple(&npy_bytes).to_le_bytes());
        central_dir.extend_from_slice(&(npy_bytes.len() as u32).to_le_bytes());
        central_dir.extend_from_slice(&(npy_bytes.len() as u32).to_le_bytes());
        central_dir.extend_from_slice(&(fname_bytes.len() as u16).to_le_bytes());
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // extra len
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // comment len
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // disk number
        central_dir.extend_from_slice(&0u16.to_le_bytes()); // internal attrs
        central_dir.extend_from_slice(&0u32.to_le_bytes()); // external attrs
        central_dir.extend_from_slice(&offset.to_le_bytes()); // local header offset
        central_dir.extend_from_slice(fname_bytes);
        cd_entries += 1;
    }

    let cd_offset = out.len() as u32;
    out.extend_from_slice(&central_dir);
    let cd_size = central_dir.len() as u32;

    // End of central directory
    out.extend_from_slice(&[0x50, 0x4B, 0x05, 0x06]);
    out.extend_from_slice(&0u16.to_le_bytes()); // disk number
    out.extend_from_slice(&0u16.to_le_bytes()); // cd start disk
    out.extend_from_slice(&cd_entries.to_le_bytes());
    out.extend_from_slice(&cd_entries.to_le_bytes());
    out.extend_from_slice(&cd_size.to_le_bytes());
    out.extend_from_slice(&cd_offset.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // comment len

    Ok(out)
}

fn crc32_simple(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 { crc = (crc >> 1) ^ 0xEDB88320; }
            else { crc >>= 1; }
        }
    }
    !crc
}


// ─── Vortex Native Format (.vxm) ───────────────────────────────────────

const VXM_MAGIC: &[u8; 8] = b"VXMODEL\0";
const VXM_VERSION: u32 = 1;
const VXM_ALIGN: usize = 64;

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VxmMetadata {
    pub model_name: String,
    pub architecture: String,
    pub num_params: usize,
    pub created_at: String,
    pub vortex_version: String,
}

pub fn save_vxm(tensors: &HashMap<String, TensorData>, metadata: &VxmMetadata, path: &str) -> Result<(), String> {
    let bytes = serialize_vxm(tensors, metadata)?;
    std::fs::write(path, bytes).map_err(|e| format!("write {}: {}", path, e))
}

pub fn serialize_vxm(tensors: &HashMap<String, TensorData>, metadata: &VxmMetadata) -> Result<Vec<u8>, String> {
    let meta_json = serde_json::to_string(metadata).map_err(|e| e.to_string())?;
    let meta_bytes = meta_json.as_bytes();

    let mut sorted_names: Vec<&String> = tensors.keys().collect();
    sorted_names.sort();

    // Calculate layout
    // Header: magic(8) + version(4) + meta_len(4) + meta_json + tensor_table_len(4) + tensor_table + padding + data
    let mut out = Vec::new();
    out.extend_from_slice(VXM_MAGIC);
    out.extend_from_slice(&VXM_VERSION.to_le_bytes());
    out.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(meta_bytes);

    // Build tensor table entries: we'll write data offset/len later
    // First pass: compute table size
    let mut table_entries: Vec<(String, DType, Vec<usize>, usize)> = Vec::new(); // name, dtype, shape, data_len
    for name in &sorted_names {
        let t = &tensors[*name];
        table_entries.push(((*name).clone(), t.dtype, t.shape.clone(), t.data.len()));
    }

    // Serialize tensor table
    let mut table_buf = Vec::new();
    let num_tensors = table_entries.len() as u32;
    table_buf.extend_from_slice(&num_tensors.to_le_bytes());

    // We need to know data start to write offsets; compute table size first
    // Each entry: name_len(4) + name + dtype(1) + ndim(4) + shape(ndim*8) + data_offset(8) + data_len(8)
    let mut table_size = 4; // num_tensors
    for (name, _, shape, _) in &table_entries {
        table_size += 4 + name.len() + 1 + 4 + shape.len() * 8 + 8 + 8;
    }

    let header_end = out.len() + 4 + table_size; // +4 for table_len field
    let data_start = ((header_end + VXM_ALIGN - 1) / VXM_ALIGN) * VXM_ALIGN;

    let mut data_offset = 0usize;
    for (name, dtype, shape, data_len) in &table_entries {
        table_buf.extend_from_slice(&(name.len() as u32).to_le_bytes());
        table_buf.extend_from_slice(name.as_bytes());
        table_buf.push(match dtype {
            DType::F32 => 0, DType::F64 => 1, DType::I32 => 2, DType::I64 => 3,
            DType::U8 => 4, DType::F16 => 5, DType::BF16 => 6,
        });
        table_buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for &dim in shape {
            table_buf.extend_from_slice(&(dim as u64).to_le_bytes());
        }
        table_buf.extend_from_slice(&(data_offset as u64).to_le_bytes());
        table_buf.extend_from_slice(&(*data_len as u64).to_le_bytes());
        data_offset += data_len;
    }

    out.extend_from_slice(&(table_buf.len() as u32).to_le_bytes());
    out.extend_from_slice(&table_buf);

    // Pad to alignment
    while out.len() < data_start { out.push(0); }

    // Write tensor data
    for name in &sorted_names {
        out.extend_from_slice(&tensors[*name].data);
    }

    Ok(out)
}

pub fn load_vxm(path: &str) -> Result<(VxmMetadata, HashMap<String, TensorData>), String> {
    let data = std::fs::read(path).map_err(|e| format!("read {}: {}", path, e))?;
    parse_vxm(&data)
}

pub fn parse_vxm(data: &[u8]) -> Result<(VxmMetadata, HashMap<String, TensorData>), String> {
    if data.len() < 16 || &data[0..8] != VXM_MAGIC {
        return Err("not a valid .vxm file".into());
    }
    let version = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
    if version != VXM_VERSION {
        return Err(format!("unsupported .vxm version: {}", version));
    }
    let meta_len = u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
    let meta_json = std::str::from_utf8(&data[16..16+meta_len]).map_err(|_| "invalid metadata utf8")?;
    let metadata: VxmMetadata = serde_json::from_str(meta_json).map_err(|e| format!("metadata parse: {}", e))?;

    let mut pos = 16 + meta_len;
    let table_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;
    let table_end = pos + table_len;

    let num_tensors = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
    pos += 4;

    // Data starts at next 64-byte alignment after table
    let data_start = ((table_end + VXM_ALIGN - 1) / VXM_ALIGN) * VXM_ALIGN;

    let mut result = HashMap::new();
    for _ in 0..num_tensors {
        let name_len = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let name = String::from_utf8_lossy(&data[pos..pos+name_len]).to_string();
        pos += name_len;
        let dtype = match data[pos] {
            0 => DType::F32, 1 => DType::F64, 2 => DType::I32, 3 => DType::I64,
            4 => DType::U8, 5 => DType::F16, 6 => DType::BF16,
            x => return Err(format!("unknown dtype code: {}", x)),
        };
        pos += 1;
        let ndim = u32::from_le_bytes([data[pos], data[pos+1], data[pos+2], data[pos+3]]) as usize;
        pos += 4;
        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            shape.push(u64::from_le_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3],data[pos+4],data[pos+5],data[pos+6],data[pos+7]]) as usize);
            pos += 8;
        }
        let tensor_offset = u64::from_le_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3],data[pos+4],data[pos+5],data[pos+6],data[pos+7]]) as usize;
        pos += 8;
        let tensor_len = u64::from_le_bytes([data[pos],data[pos+1],data[pos+2],data[pos+3],data[pos+4],data[pos+5],data[pos+6],data[pos+7]]) as usize;
        pos += 8;

        let abs_start = data_start + tensor_offset;
        if abs_start + tensor_len > data.len() {
            return Err(format!("tensor {} data out of bounds", name));
        }
        let tensor_data = data[abs_start..abs_start+tensor_len].to_vec();
        result.insert(name, TensorData { dtype, shape, data: tensor_data });
    }

    Ok((metadata, result))
}


// ─── Model Conversion ──────────────────────────────────────────────────

pub fn detect_format(path: &str) -> Result<ModelFormat, String> {
    ModelFormat::from_extension(path)
}

pub fn load_tensors(path: &str) -> Result<HashMap<String, TensorData>, String> {
    let fmt = detect_format(path)?;
    match fmt {
        ModelFormat::SafeTensors => load_safetensors(path),
        ModelFormat::Onnx => {
            let onnx = load_onnx(path)?;
            Ok(onnx.graph.initializers)
        }
        ModelFormat::Npy => {
            let t = load_npy(path)?;
            let mut m = HashMap::new();
            m.insert("tensor".to_string(), t);
            Ok(m)
        }
        ModelFormat::Npz => load_npz(path),
        ModelFormat::VortexNative => {
            let (_, tensors) = load_vxm(path)?;
            Ok(tensors)
        }
        ModelFormat::VortexJson => Err("JSON format stores params not tensors; use nn::load_model_weights".into()),
    }
}

pub fn save_tensors(tensors: &HashMap<String, TensorData>, path: &str) -> Result<(), String> {
    let fmt = detect_format(path)?;
    match fmt {
        ModelFormat::SafeTensors => save_safetensors(tensors, path),
        ModelFormat::Npy => {
            if tensors.len() != 1 {
                return Err("npy supports only one tensor; use .npz for multiple".into());
            }
            let (_, t) = tensors.iter().next().unwrap();
            save_npy(t, path)
        }
        ModelFormat::Npz => save_npz(tensors, path),
        ModelFormat::VortexNative => {
            let total_params: usize = tensors.values().map(|t| t.num_elements()).sum();
            let meta = VxmMetadata {
                model_name: "converted".into(),
                architecture: "unknown".into(),
                num_params: total_params,
                created_at: "2026-02-22".into(),
                vortex_version: "0.1.0".into(),
            };
            save_vxm(tensors, &meta, path)
        }
        ModelFormat::Onnx => Err("ONNX export not supported (read-only)".into()),
        ModelFormat::VortexJson => Err("use nn::save_model for JSON format".into()),
    }
}

pub fn convert_model(input_path: &str, output_path: &str) -> Result<(), String> {
    let tensors = load_tensors(input_path)?;
    save_tensors(&tensors, output_path)
}


// ─── Weight Loading into nn::Model ──────────────────────────────────────

pub fn load_weights_into_model(model: &mut crate::nn::Model, tensors: &HashMap<String, TensorData>) -> Result<usize, String> {
    use crate::nn::Layer;
    let mut loaded = 0;

    // Strategy 1: try "layers.N.weight" / "layers.N.bias" naming
    for (i, layer) in model.layers.iter_mut().enumerate() {
        let weight_key = format!("layers.{}.weight", i);
        let bias_key = format!("layers.{}.bias", i);

        match layer {
            Layer::Linear { weight, bias, .. } => {
                if let Some(w) = tensors.get(&weight_key) {
                    let vals = w.to_f64_vec();
                    if vals.len() == weight.data.len() {
                        weight.data = vals;
                        loaded += 1;
                    } else {
                        return Err(format!("shape mismatch for {}: expected {} got {}", weight_key, weight.data.len(), vals.len()));
                    }
                }
                if let Some(b) = tensors.get(&bias_key) {
                    let vals = b.to_f64_vec();
                    if vals.len() == bias.data.len() {
                        bias.data = vals;
                        loaded += 1;
                    }
                }
            }
            Layer::LayerNorm { gamma, beta, .. } => {
                let gamma_key = format!("layers.{}.gamma", i);
                let beta_key = format!("layers.{}.beta", i);
                if let Some(g) = tensors.get(&gamma_key).or_else(|| tensors.get(&weight_key)) {
                    let vals = g.to_f64_vec();
                    if vals.len() == gamma.data.len() { gamma.data = vals; loaded += 1; }
                }
                if let Some(b) = tensors.get(&beta_key).or_else(|| tensors.get(&bias_key)) {
                    let vals = b.to_f64_vec();
                    if vals.len() == beta.data.len() { beta.data = vals; loaded += 1; }
                }
            }
            Layer::Embedding { weight, .. } => {
                if let Some(w) = tensors.get(&weight_key) {
                    let vals = w.to_f64_vec();
                    if vals.len() == weight.data.len() { weight.data = vals; loaded += 1; }
                }
            }
            _ => {}
        }
    }

    // Strategy 2: if nothing loaded with indexed keys, try matching by order
    if loaded == 0 && !tensors.is_empty() {
        let mut tensor_list: Vec<(&String, &TensorData)> = tensors.iter().collect();
        tensor_list.sort_by_key(|(k, _)| k.to_string());
        let mut tidx = 0;
        for layer in model.layers.iter_mut() {
            let params = layer.parameters_mut();
            for (p, _grad) in params {
                if tidx < tensor_list.len() {
                    let vals = tensor_list[tidx].1.to_f64_vec();
                    if vals.len() == p.data.len() {
                        p.data = vals;
                        loaded += 1;
                    }
                    tidx += 1;
                }
            }
        }
    }

    Ok(loaded)
}


// ─── Model Info ─────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct ModelInfo {
    pub format: ModelFormat,
    pub num_tensors: usize,
    pub num_params: usize,
    pub tensor_names: Vec<String>,
    pub tensor_shapes: Vec<(String, Vec<usize>)>,
}

pub fn model_info(path: &str) -> Result<ModelInfo, String> {
    let fmt = detect_format(path)?;
    let tensors = load_tensors(path)?;
    let num_tensors = tensors.len();
    let num_params: usize = tensors.values().map(|t| t.num_elements()).sum();
    let mut names: Vec<String> = tensors.keys().cloned().collect();
    names.sort();
    let shapes: Vec<(String, Vec<usize>)> = names.iter()
        .map(|n| (n.clone(), tensors[n].shape.clone())).collect();
    Ok(ModelInfo { format: fmt, num_tensors, num_params, tensor_names: names, tensor_shapes: shapes })
}

pub fn print_model_info(path: &str) -> Result<(), String> {
    let info = model_info(path)?;
    println!("Format:      {}", info.format.name());
    println!("Tensors:     {}", info.num_tensors);
    println!("Parameters:  {}", info.num_params);
    println!("Layers:");
    for (name, shape) in &info.tensor_shapes {
        println!("  {} {:?}", name, shape);
    }
    Ok(())
}


// ─── Interpreter Builtins ───────────────────────────────────────────────

use crate::interpreter::{Value, Env, FnDef};

fn builtin_model_load(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("model_load(path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("expected string path".into()) };
    let fmt = detect_format(&path)?;
    match fmt {
        ModelFormat::Onnx => {
            let onnx = load_onnx(&path)?;
            let model = onnx_to_vortex(&onnx);
            let id = env.nn_models.len();
            env.nn_models.insert(id, model);
            Ok(Value::Int(id as i128))
        }
        _ => {
            let tensors = load_tensors(&path)?;
            // Create a minimal model and load weights
            // Return tensor count as proxy
            Ok(Value::Int(tensors.len() as i128))
        }
    }
}

fn builtin_model_save(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("model_save(model_id, path)".into()); }
    let model_id = match &args[0] { Value::Int(i) => *i as usize, _ => return Err("expected int".into()) };
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("expected string path".into()) };
    let model = env.nn_models.get(&model_id).ok_or("no such model")?.clone();

    // Extract tensors from model
    let mut tensors = HashMap::new();
    for (i, layer) in model.layers.iter().enumerate() {
        for (j, (p, _)) in layer.parameters().iter().enumerate() {
            let name = if j == 0 { format!("layers.{}.weight", i) } else { format!("layers.{}.bias", i) };
            tensors.insert(name, TensorData::from_f64_slice(p.shape.clone(), &p.data, DType::F64));
        }
    }
    save_tensors(&tensors, &path)?;
    Ok(Value::Void)
}

fn builtin_model_convert(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("model_convert(input_path, output_path)".into()); }
    let input = match &args[0] { Value::String(s) => s.clone(), _ => return Err("expected string".into()) };
    let output = match &args[1] { Value::String(s) => s.clone(), _ => return Err("expected string".into()) };
    convert_model(&input, &output)?;
    Ok(Value::Void)
}

fn builtin_model_info(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("model_info(path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("expected string".into()) };
    let info = model_info(&path)?;
    Ok(Value::String(format!("format={}, tensors={}, params={}", info.format.name(), info.num_tensors, info.num_params)))
}

fn builtin_tensor_load(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("tensor_load(path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("expected string".into()) };
    let tensors = load_tensors(&path)?;
    // Return first tensor as flat array
    if let Some((_, t)) = tensors.iter().next() {
        let vals = t.to_f64_vec();
        Ok(Value::Array(vals.into_iter().map(Value::Float).collect()))
    } else {
        Ok(Value::Array(vec![]))
    }
}

fn builtin_tensor_save(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("tensor_save(data_array, path)".into()); }
    let data: Vec<f64> = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::Float(f) => Ok(*f),
            Value::Int(i) => Ok(*i as f64),
            _ => Err("expected number".to_string()),
        }).collect::<Result<Vec<_>, _>>()?,
        _ => return Err("expected array".into()),
    };
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("expected string path".into()) };
    let shape = vec![data.len()];
    let tensor = TensorData::from_f64_slice(shape, &data, DType::F64);
    let mut tensors = HashMap::new();
    tensors.insert("tensor".to_string(), tensor);
    save_tensors(&tensors, &path)?;
    Ok(Value::Void)
}

pub fn register_model_interop_builtins(env: &mut Env) {
    env.functions.insert("model_load".into(), FnDef::Builtin(builtin_model_load));
    env.functions.insert("model_save_interop".into(), FnDef::Builtin(builtin_model_save));
    env.functions.insert("model_convert".into(), FnDef::Builtin(builtin_model_convert));
    env.functions.insert("model_info".into(), FnDef::Builtin(builtin_model_info));
    env.functions.insert("tensor_load".into(), FnDef::Builtin(builtin_tensor_load));
    env.functions.insert("tensor_save".into(), FnDef::Builtin(builtin_tensor_save));
}


// ─── CLI ────────────────────────────────────────────────────────────────

pub fn cli_model_command(args: &[String]) -> Result<(), String> {
    if args.is_empty() {
        return Err("Usage: vortex model <info|convert> ...".into());
    }
    match args[0].as_str() {
        "info" => {
            if args.len() < 2 { return Err("Usage: vortex model info <file>".into()); }
            print_model_info(&args[1])
        }
        "convert" => {
            if args.len() < 3 { return Err("Usage: vortex model convert <input> <output>".into()); }
            convert_model(&args[1], &args[2])?;
            println!("Converted {} -> {}", args[1], args[2]);
            Ok(())
        }
        _ => Err(format!("unknown model subcommand: {}", args[0])),
    }
}


// ─── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_f32_tensor(shape: Vec<usize>, vals: &[f32]) -> TensorData {
        let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
        TensorData { dtype: DType::F32, shape, data }
    }

    fn make_f64_tensor(shape: Vec<usize>, vals: &[f64]) -> TensorData {
        TensorData::from_f64_slice(shape, vals, DType::F64)
    }

    #[test]
    fn test_safetensors_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert("weight".into(), make_f32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        tensors.insert("bias".into(), make_f32_tensor(vec![3], &[0.1, 0.2, 0.3]));

        let bytes = serialize_safetensors(&tensors).unwrap();
        let loaded = parse_safetensors(&bytes).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["weight"].shape, vec![2, 3]);
        assert_eq!(loaded["weight"].dtype, DType::F32);
        assert_eq!(loaded["weight"].data, tensors["weight"].data);
        assert_eq!(loaded["bias"].shape, vec![3]);
        assert_eq!(loaded["bias"].data, tensors["bias"].data);
    }

    #[test]
    fn test_safetensors_multiple_dtypes() {
        let mut tensors = HashMap::new();
        tensors.insert("f32_tensor".into(), make_f32_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]));
        tensors.insert("f64_tensor".into(), make_f64_tensor(vec![2], &[5.0, 6.0]));
        tensors.insert("i32_tensor".into(), TensorData {
            dtype: DType::I32, shape: vec![3],
            data: [10i32, 20, 30].iter().flat_map(|v| v.to_le_bytes()).collect(),
        });

        let bytes = serialize_safetensors(&tensors).unwrap();
        let loaded = parse_safetensors(&bytes).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded["f32_tensor"].dtype, DType::F32);
        assert_eq!(loaded["f64_tensor"].dtype, DType::F64);
        assert_eq!(loaded["i32_tensor"].dtype, DType::I32);
    }

    #[test]
    fn test_safetensors_empty() {
        let tensors: HashMap<String, TensorData> = HashMap::new();
        let bytes = serialize_safetensors(&tensors).unwrap();
        let loaded = parse_safetensors(&bytes).unwrap();
        assert_eq!(loaded.len(), 0);
    }

    #[test]
    fn test_vxm_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert("layer0.weight".into(), make_f32_tensor(vec![4, 8], &vec![1.5f32; 32]));
        tensors.insert("layer0.bias".into(), make_f32_tensor(vec![8], &vec![0.0f32; 8]));

        let meta = VxmMetadata {
            model_name: "test_model".into(),
            architecture: "mlp".into(),
            num_params: 40,
            created_at: "2026-02-22".into(),
            vortex_version: "0.1.0".into(),
        };

        let bytes = serialize_vxm(&tensors, &meta).unwrap();
        let (loaded_meta, loaded_tensors) = parse_vxm(&bytes).unwrap();

        assert_eq!(loaded_meta.model_name, "test_model");
        assert_eq!(loaded_meta.architecture, "mlp");
        assert_eq!(loaded_tensors.len(), 2);
        assert_eq!(loaded_tensors["layer0.weight"].shape, vec![4, 8]);
        assert_eq!(loaded_tensors["layer0.weight"].data, tensors["layer0.weight"].data);
    }

    #[test]
    fn test_npy_roundtrip_f32() {
        let tensor = make_f32_tensor(vec![3, 4], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let bytes = serialize_npy(&tensor).unwrap();
        let loaded = parse_npy(&bytes).unwrap();
        assert_eq!(loaded.dtype, DType::F32);
        assert_eq!(loaded.shape, vec![3, 4]);
        assert_eq!(loaded.data, tensor.data);
    }

    #[test]
    fn test_npy_roundtrip_f64() {
        let tensor = make_f64_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]);
        let bytes = serialize_npy(&tensor).unwrap();
        let loaded = parse_npy(&bytes).unwrap();
        assert_eq!(loaded.dtype, DType::F64);
        assert_eq!(loaded.shape, vec![2, 2]);
        assert_eq!(loaded.data, tensor.data);
    }

    #[test]
    fn test_npy_1d() {
        let tensor = make_f32_tensor(vec![5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        let bytes = serialize_npy(&tensor).unwrap();
        let loaded = parse_npy(&bytes).unwrap();
        assert_eq!(loaded.shape, vec![5]);
    }

    #[test]
    fn test_npz_roundtrip() {
        let mut tensors = HashMap::new();
        tensors.insert("x".into(), make_f32_tensor(vec![3], &[1.0, 2.0, 3.0]));
        tensors.insert("y".into(), make_f32_tensor(vec![2], &[4.0, 5.0]));

        let bytes = serialize_npz(&tensors).unwrap();
        let loaded = parse_npz(&bytes).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded["x"].shape, vec![3]);
        assert_eq!(loaded["y"].shape, vec![2]);
        assert_eq!(loaded["x"].data, tensors["x"].data);
    }

    #[test]
    fn test_format_detection() {
        assert_eq!(detect_format("model.safetensors").unwrap(), ModelFormat::SafeTensors);
        assert_eq!(detect_format("model.onnx").unwrap(), ModelFormat::Onnx);
        assert_eq!(detect_format("data.npy").unwrap(), ModelFormat::Npy);
        assert_eq!(detect_format("data.npz").unwrap(), ModelFormat::Npz);
        assert_eq!(detect_format("model.vxm").unwrap(), ModelFormat::VortexNative);
        assert_eq!(detect_format("model.json").unwrap(), ModelFormat::VortexJson);
        assert!(detect_format("model.xyz").is_err());
    }

    #[test]
    fn test_tensor_data_to_f64() {
        let t = make_f32_tensor(vec![3], &[1.5, 2.5, 3.5]);
        let vals = t.to_f64_vec();
        assert_eq!(vals.len(), 3);
        assert!((vals[0] - 1.5).abs() < 1e-6);
        assert!((vals[1] - 2.5).abs() < 1e-6);
        assert!((vals[2] - 3.5).abs() < 1e-6);
    }

    #[test]
    fn test_tensor_data_from_f64() {
        let t = TensorData::from_f64_slice(vec![2, 2], &[1.0, 2.0, 3.0, 4.0], DType::F32);
        assert_eq!(t.dtype, DType::F32);
        assert_eq!(t.shape, vec![2, 2]);
        assert_eq!(t.data.len(), 16); // 4 floats * 4 bytes
        let back = t.to_f64_vec();
        assert!((back[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_dtype_from_str() {
        assert_eq!(DType::from_str("F32").unwrap(), DType::F32);
        assert_eq!(DType::from_str("float64").unwrap(), DType::F64);
        assert_eq!(DType::from_str("int32").unwrap(), DType::I32);
        assert!(DType::from_str("complex128").is_err());
    }

    #[test]
    fn test_safetensors_corrupt_header() {
        let bytes = vec![0xFF; 16]; // invalid header
        assert!(parse_safetensors(&bytes).is_err());
    }

    #[test]
    fn test_safetensors_too_small() {
        assert!(parse_safetensors(&[1, 2, 3]).is_err());
    }

    #[test]
    fn test_vxm_corrupt() {
        assert!(parse_vxm(&[0; 16]).is_err());
        assert!(parse_vxm(&[]).is_err());
    }

    #[test]
    fn test_npy_corrupt() {
        assert!(parse_npy(&[0; 10]).is_err());
    }

    #[test]
    fn test_weight_loading_into_model() {
        use crate::nn::{Model, Layer};
        let mut model = Model::sequential(vec![Layer::linear(3, 2)]);

        let mut tensors = HashMap::new();
        tensors.insert("layers.0.weight".into(), TensorData::from_f64_slice(vec![3, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], DType::F64));
        tensors.insert("layers.0.bias".into(), TensorData::from_f64_slice(vec![2], &[0.1, 0.2], DType::F64));

        let loaded = load_weights_into_model(&mut model, &tensors).unwrap();
        assert_eq!(loaded, 2);

        // Verify weights were loaded
        match &model.layers[0] {
            Layer::Linear { weight, bias, .. } => {
                assert!((weight.data[0] - 1.0).abs() < 1e-10);
                assert!((weight.data[5] - 6.0).abs() < 1e-10);
                assert!((bias.data[0] - 0.1).abs() < 1e-10);
            }
            _ => panic!("expected Linear layer"),
        }
    }

    #[test]
    fn test_weight_loading_ordered_fallback() {
        use crate::nn::{Model, Layer};
        let mut model = Model::sequential(vec![Layer::linear(2, 2)]);

        // Use non-standard naming -> should fall back to ordered loading
        let mut tensors = HashMap::new();
        tensors.insert("a_kernel".into(), TensorData::from_f64_slice(vec![2, 2], &[1.0, 2.0, 3.0, 4.0], DType::F64));
        tensors.insert("b_bias".into(), TensorData::from_f64_slice(vec![2], &[0.5, 0.6], DType::F64));

        let loaded = load_weights_into_model(&mut model, &tensors).unwrap();
        assert!(loaded > 0);
    }

    #[test]
    fn test_crc32() {
        assert_eq!(crc32_simple(b""), 0x00000000);
        assert_eq!(crc32_simple(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn test_f16_conversion() {
        let v = 1.5f64;
        let bits = f64_to_f16(v);
        let back = f16_to_f64(bits);
        assert!((back - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_bf16_conversion() {
        let v = 2.0f64;
        let bits = f64_to_bf16(v);
        let back = bf16_to_f64(bits);
        assert!((back - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_safetensors_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_model.safetensors");
        let path_str = path.to_str().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("w".into(), make_f32_tensor(vec![2, 2], &[1.0, 2.0, 3.0, 4.0]));
        save_safetensors(&tensors, path_str).unwrap();
        let loaded = load_safetensors(path_str).unwrap();
        assert_eq!(loaded["w"].data, tensors["w"].data);
        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_vxm_file_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_model.vxm");
        let path_str = path.to_str().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("fc.weight".into(), make_f32_tensor(vec![4], &[1.0, 2.0, 3.0, 4.0]));
        let meta = VxmMetadata {
            model_name: "test".into(), architecture: "fc".into(), num_params: 4,
            created_at: "2026-02-22".into(), vortex_version: "0.1.0".into(),
        };
        save_vxm(&tensors, &meta, path_str).unwrap();
        let (m, t) = load_vxm(path_str).unwrap();
        assert_eq!(m.model_name, "test");
        assert_eq!(t["fc.weight"].data, tensors["fc.weight"].data);
        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_model_info_safetensors() {
        let dir = std::env::temp_dir();
        let path = dir.join("test_info.safetensors");
        let path_str = path.to_str().unwrap();

        let mut tensors = HashMap::new();
        tensors.insert("a".into(), make_f32_tensor(vec![3, 4], &vec![0.0f32; 12]));
        tensors.insert("b".into(), make_f32_tensor(vec![4], &vec![0.0f32; 4]));
        save_safetensors(&tensors, path_str).unwrap();

        let info = model_info(path_str).unwrap();
        assert_eq!(info.format, ModelFormat::SafeTensors);
        assert_eq!(info.num_tensors, 2);
        assert_eq!(info.num_params, 16);
        std::fs::remove_file(path_str).ok();
    }

    #[test]
    fn test_convert_safetensors_to_npz() {
        let dir = std::env::temp_dir();
        let st_path = dir.join("convert_test.safetensors");
        let npz_path = dir.join("convert_test.npz");

        let mut tensors = HashMap::new();
        tensors.insert("w".into(), make_f32_tensor(vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        save_safetensors(&tensors, st_path.to_str().unwrap()).unwrap();

        convert_model(st_path.to_str().unwrap(), npz_path.to_str().unwrap()).unwrap();

        let loaded = load_npz(npz_path.to_str().unwrap()).unwrap();
        assert_eq!(loaded["w"].shape, vec![2, 3]);
        std::fs::remove_file(st_path).ok();
        std::fs::remove_file(npz_path).ok();
    }
}
