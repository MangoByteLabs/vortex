// ─── GGUF Binary Format Parser ──────────────────────────────────────────────
//
// Parses GGUF (GGML Universal Format) files containing quantized model weights.
// Supports versions 2 and 3, with dequantization for Q4_0, Q4_1, Q8_0, F16, F32.

use crate::interpreter::{Env, FnDef, Value};
use crate::tensor_engine::{DType, FastTensor, Layout, f16_to_f32};

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};

// ─── Constants ──────────────────────────────────────────────────────────────

const GGUF_MAGIC: u32 = 0x46475547; // "GGUF" in little-endian
const GGUF_VERSION_2: u32 = 2;
const GGUF_VERSION_3: u32 = 3;
const GGUF_DEFAULT_ALIGNMENT: usize = 32;

// ─── Enums ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    U8 = 0,
    I8 = 1,
    U16 = 2,
    I16 = 3,
    U32 = 4,
    I32 = 5,
    F32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    U64 = 10,
    I64 = 11,
    F64 = 12,
}

impl GGUFValueType {
    fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::U8),
            1 => Ok(Self::I8),
            2 => Ok(Self::U16),
            3 => Ok(Self::I16),
            4 => Ok(Self::U32),
            5 => Ok(Self::I32),
            6 => Ok(Self::F32),
            7 => Ok(Self::Bool),
            8 => Ok(Self::String),
            9 => Ok(Self::Array),
            10 => Ok(Self::U64),
            11 => Ok(Self::I64),
            12 => Ok(Self::F64),
            _ => Err(format!("unknown GGUF value type: {}", v)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLQuantType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
}

impl GGMLQuantType {
    fn from_u32(v: u32) -> Result<Self, String> {
        match v {
            0 => Ok(Self::F32),
            1 => Ok(Self::F16),
            2 => Ok(Self::Q4_0),
            3 => Ok(Self::Q4_1),
            6 => Ok(Self::Q5_0),
            7 => Ok(Self::Q5_1),
            8 => Ok(Self::Q8_0),
            10 => Ok(Self::Q2_K),
            11 => Ok(Self::Q3_K),
            12 => Ok(Self::Q4_K),
            13 => Ok(Self::Q5_K),
            14 => Ok(Self::Q6_K),
            _ => Err(format!("unknown GGML quant type: {}", v)),
        }
    }

    /// Bytes per block for this quantization type.
    fn block_size(&self) -> usize {
        match self {
            Self::F32 => 4,
            Self::F16 => 2,
            Self::Q4_0 => 18,  // 2 (f16 scale) + 16 (32 nibbles)
            Self::Q4_1 => 20,  // 2 (f16 scale) + 2 (f16 min) + 16 (32 nibbles)
            Self::Q8_0 => 34,  // 2 (f16 scale) + 32 (i8 values)
            Self::Q5_0 => 22,
            Self::Q5_1 => 24,
            Self::Q2_K => 84,
            Self::Q3_K => 110,
            Self::Q4_K => 144,
            Self::Q5_K => 176,
            Self::Q6_K => 210,
        }
    }

    /// Number of elements per block.
    fn elements_per_block(&self) -> usize {
        match self {
            Self::F32 => 1,
            Self::F16 => 1,
            Self::Q4_0 | Self::Q4_1 => 32,
            Self::Q8_0 => 32,
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q2_K => 256,
            Self::Q3_K => 256,
            Self::Q4_K => 256,
            Self::Q5_K => 256,
            Self::Q6_K => 256,
        }
    }
}

// ─── Data structures ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GGUFValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GGUFValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GGUFValue {
    fn to_value(&self) -> Value {
        match self {
            GGUFValue::U8(v) => Value::Int(*v as i128),
            GGUFValue::I8(v) => Value::Int(*v as i128),
            GGUFValue::U16(v) => Value::Int(*v as i128),
            GGUFValue::I16(v) => Value::Int(*v as i128),
            GGUFValue::U32(v) => Value::Int(*v as i128),
            GGUFValue::I32(v) => Value::Int(*v as i128),
            GGUFValue::F32(v) => Value::Float(*v as f64),
            GGUFValue::Bool(v) => Value::Bool(*v),
            GGUFValue::String(v) => Value::String(v.clone()),
            GGUFValue::Array(arr) => {
                Value::Array(arr.iter().map(|v| v.to_value()).collect())
            }
            GGUFValue::U64(v) => Value::Int(*v as i128),
            GGUFValue::I64(v) => Value::Int(*v as i128),
            GGUFValue::F64(v) => Value::Float(*v),
        }
    }
}

pub type GGUFMetadata = HashMap<String, GGUFValue>;

#[derive(Debug, Clone)]
pub struct GGUFTensorInfo {
    pub name: String,
    pub n_dims: u32,
    pub dimensions: Vec<u64>,
    pub quant_type: GGMLQuantType,
    pub offset: u64,
}

impl GGUFTensorInfo {
    /// Total number of elements in this tensor.
    pub fn n_elements(&self) -> usize {
        self.dimensions.iter().map(|&d| d as usize).product()
    }

    /// Total bytes of quantized data for this tensor.
    pub fn data_size(&self) -> usize {
        let n = self.n_elements();
        let epb = self.quant_type.elements_per_block();
        let bs = self.quant_type.block_size();
        let n_blocks = (n + epb - 1) / epb;
        n_blocks * bs
    }
}

pub struct GGUFFile {
    pub version: u32,
    pub metadata: GGUFMetadata,
    pub tensor_infos: Vec<GGUFTensorInfo>,
    pub tensor_data_offset: usize,
    pub raw_data: Vec<u8>,
}

// ─── Binary reader helper ───────────────────────────────────────────────────

struct BufReader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl<'a> BufReader<'a> {
    fn new(data: &'a [u8]) -> Self {
        Self { data, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.data.len().saturating_sub(self.pos)
    }

    fn read_u8(&mut self) -> Result<u8, String> {
        if self.remaining() < 1 {
            return Err("unexpected EOF reading u8".into());
        }
        let v = self.data[self.pos];
        self.pos += 1;
        Ok(v)
    }

    fn read_i8(&mut self) -> Result<i8, String> {
        Ok(self.read_u8()? as i8)
    }

    fn read_u16(&mut self) -> Result<u16, String> {
        if self.remaining() < 2 {
            return Err("unexpected EOF reading u16".into());
        }
        let v = u16::from_le_bytes([self.data[self.pos], self.data[self.pos + 1]]);
        self.pos += 2;
        Ok(v)
    }

    fn read_i16(&mut self) -> Result<i16, String> {
        Ok(self.read_u16()? as i16)
    }

    fn read_u32(&mut self) -> Result<u32, String> {
        if self.remaining() < 4 {
            return Err("unexpected EOF reading u32".into());
        }
        let v = u32::from_le_bytes([
            self.data[self.pos],
            self.data[self.pos + 1],
            self.data[self.pos + 2],
            self.data[self.pos + 3],
        ]);
        self.pos += 4;
        Ok(v)
    }

    fn read_i32(&mut self) -> Result<i32, String> {
        Ok(self.read_u32()? as i32)
    }

    fn read_u64(&mut self) -> Result<u64, String> {
        if self.remaining() < 8 {
            return Err("unexpected EOF reading u64".into());
        }
        let bytes: [u8; 8] = self.data[self.pos..self.pos + 8]
            .try_into()
            .map_err(|_| "slice conversion failed")?;
        let v = u64::from_le_bytes(bytes);
        self.pos += 8;
        Ok(v)
    }

    fn read_i64(&mut self) -> Result<i64, String> {
        Ok(self.read_u64()? as i64)
    }

    fn read_f32(&mut self) -> Result<f32, String> {
        let bits = self.read_u32()?;
        Ok(f32::from_bits(bits))
    }

    fn read_f64(&mut self) -> Result<f64, String> {
        let bits = self.read_u64()?;
        Ok(f64::from_bits(bits))
    }

    fn read_bool(&mut self) -> Result<bool, String> {
        Ok(self.read_u8()? != 0)
    }

    fn read_string(&mut self) -> Result<String, String> {
        let len = self.read_u64()? as usize;
        if self.remaining() < len {
            return Err(format!(
                "unexpected EOF reading string of length {}",
                len
            ));
        }
        let s = std::str::from_utf8(&self.data[self.pos..self.pos + len])
            .map_err(|e| format!("invalid UTF-8 in GGUF string: {}", e))?
            .to_string();
        self.pos += len;
        Ok(s)
    }

    fn read_gguf_value(&mut self, vtype: GGUFValueType) -> Result<GGUFValue, String> {
        match vtype {
            GGUFValueType::U8 => Ok(GGUFValue::U8(self.read_u8()?)),
            GGUFValueType::I8 => Ok(GGUFValue::I8(self.read_i8()?)),
            GGUFValueType::U16 => Ok(GGUFValue::U16(self.read_u16()?)),
            GGUFValueType::I16 => Ok(GGUFValue::I16(self.read_i16()?)),
            GGUFValueType::U32 => Ok(GGUFValue::U32(self.read_u32()?)),
            GGUFValueType::I32 => Ok(GGUFValue::I32(self.read_i32()?)),
            GGUFValueType::F32 => Ok(GGUFValue::F32(self.read_f32()?)),
            GGUFValueType::Bool => Ok(GGUFValue::Bool(self.read_bool()?)),
            GGUFValueType::String => Ok(GGUFValue::String(self.read_string()?)),
            GGUFValueType::Array => {
                let elem_type = GGUFValueType::from_u32(self.read_u32()?)?;
                let count = self.read_u64()? as usize;
                let mut elems = Vec::with_capacity(count);
                for _ in 0..count {
                    elems.push(self.read_gguf_value(elem_type)?);
                }
                Ok(GGUFValue::Array(elems))
            }
            GGUFValueType::U64 => Ok(GGUFValue::U64(self.read_u64()?)),
            GGUFValueType::I64 => Ok(GGUFValue::I64(self.read_i64()?)),
            GGUFValueType::F64 => Ok(GGUFValue::F64(self.read_f64()?)),
        }
    }
}

// ─── Parser ─────────────────────────────────────────────────────────────────

pub fn parse_gguf(data: &[u8]) -> Result<GGUFFile, String> {
    let mut r = BufReader::new(data);

    // Header
    let magic = r.read_u32()?;
    if magic != GGUF_MAGIC {
        return Err(format!(
            "invalid GGUF magic: 0x{:08X} (expected 0x{:08X})",
            magic, GGUF_MAGIC
        ));
    }

    let version = r.read_u32()?;
    if version != GGUF_VERSION_2 && version != GGUF_VERSION_3 {
        return Err(format!("unsupported GGUF version: {}", version));
    }

    let tensor_count = r.read_u64()? as usize;
    let metadata_kv_count = r.read_u64()? as usize;

    // Metadata
    let mut metadata = GGUFMetadata::new();
    for _ in 0..metadata_kv_count {
        let key = r.read_string()?;
        let vtype = GGUFValueType::from_u32(r.read_u32()?)?;
        let value = r.read_gguf_value(vtype)?;
        metadata.insert(key, value);
    }

    // Read alignment from metadata (default 32)
    let alignment = match metadata.get("general.alignment") {
        Some(GGUFValue::U32(a)) => *a as usize,
        Some(GGUFValue::U64(a)) => *a as usize,
        Some(GGUFValue::I32(a)) => *a as usize,
        _ => GGUF_DEFAULT_ALIGNMENT,
    };

    // Tensor descriptors
    let mut tensor_infos = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.read_string()?;
        let n_dims = r.read_u32()?;
        let mut dimensions = Vec::with_capacity(n_dims as usize);
        for _ in 0..n_dims {
            dimensions.push(r.read_u64()?);
        }
        let quant_type = GGMLQuantType::from_u32(r.read_u32()?)?;
        let offset = r.read_u64()?;
        tensor_infos.push(GGUFTensorInfo {
            name,
            n_dims,
            dimensions,
            quant_type,
            offset,
        });
    }

    // Compute tensor data offset (aligned)
    let tensor_data_offset = (r.pos + alignment - 1) / alignment * alignment;

    Ok(GGUFFile {
        version,
        metadata,
        tensor_infos,
        tensor_data_offset,
        raw_data: data.to_vec(),
    })
}

// ─── Dequantization ─────────────────────────────────────────────────────────

/// Dequantize Q4_0: blocks of 18 bytes (2-byte f16 scale + 16 bytes of nibble pairs = 32 values).
pub fn dequantize_q4_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 18;
    let values_per_block = 32;
    let n_blocks = (n + values_per_block - 1) / values_per_block;
    let mut out = Vec::with_capacity(n);

    for i in 0..n_blocks {
        let block = &data[i * block_size..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);

        for j in 0..16 {
            let byte = block[2 + j];
            let lo = (byte & 0x0F) as f32 - 8.0;
            let hi = ((byte >> 4) & 0x0F) as f32 - 8.0;
            let idx = i * values_per_block + j * 2;
            if idx < n {
                out.push(lo * scale);
            }
            if idx + 1 < n {
                out.push(hi * scale);
            }
        }
    }

    out.truncate(n);
    out
}

/// Dequantize Q4_1: blocks of 20 bytes (2-byte f16 scale + 2-byte f16 min + 16 nibble bytes = 32 values).
pub fn dequantize_q4_1(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 20;
    let values_per_block = 32;
    let n_blocks = (n + values_per_block - 1) / values_per_block;
    let mut out = Vec::with_capacity(n);

    for i in 0..n_blocks {
        let block = &data[i * block_size..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let min_bits = u16::from_le_bytes([block[2], block[3]]);
        let scale = f16_to_f32(scale_bits);
        let min = f16_to_f32(min_bits);

        for j in 0..16 {
            let byte = block[4 + j];
            let lo = (byte & 0x0F) as f32;
            let hi = ((byte >> 4) & 0x0F) as f32;
            let idx = i * values_per_block + j * 2;
            if idx < n {
                out.push(lo * scale + min);
            }
            if idx + 1 < n {
                out.push(hi * scale + min);
            }
        }
    }

    out.truncate(n);
    out
}

/// Dequantize Q8_0: blocks of 34 bytes (2-byte f16 scale + 32 i8 values).
pub fn dequantize_q8_0(data: &[u8], n: usize) -> Vec<f32> {
    let block_size = 34;
    let values_per_block = 32;
    let n_blocks = (n + values_per_block - 1) / values_per_block;
    let mut out = Vec::with_capacity(n);

    for i in 0..n_blocks {
        let block = &data[i * block_size..];
        let scale_bits = u16::from_le_bytes([block[0], block[1]]);
        let scale = f16_to_f32(scale_bits);

        for j in 0..32 {
            let idx = i * values_per_block + j;
            if idx >= n {
                break;
            }
            let val = block[2 + j] as i8;
            out.push(val as f32 * scale);
        }
    }

    out.truncate(n);
    out
}

/// Dequantize F16: pairs of bytes → f32.
pub fn dequantize_f16(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 2;
        if offset + 1 >= data.len() {
            break;
        }
        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        out.push(f16_to_f32(bits));
    }
    out
}

/// Copy F32: reinterpret raw bytes as f32.
pub fn copy_f32(data: &[u8], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let offset = i * 4;
        if offset + 3 >= data.len() {
            break;
        }
        let bits = u32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ]);
        out.push(f32::from_bits(bits));
    }
    out
}

// ─── Tensor extraction ──────────────────────────────────────────────────────

/// Extract a tensor by name from a parsed GGUF file, dequantize to F32.
pub fn get_tensor(file: &GGUFFile, name: &str) -> Result<FastTensor, String> {
    let info = file
        .tensor_infos
        .iter()
        .find(|t| t.name == name)
        .ok_or_else(|| format!("tensor '{}' not found in GGUF file", name))?;

    let n_elements = info.n_elements();
    let data_start = file.tensor_data_offset + info.offset as usize;
    let data_end = data_start + info.data_size();

    if data_end > file.raw_data.len() {
        return Err(format!(
            "tensor '{}' data extends beyond file (need {} bytes at offset {}, file is {} bytes)",
            name, info.data_size(), data_start, file.raw_data.len()
        ));
    }

    let raw = &file.raw_data[data_start..data_end];

    let f32_data = match info.quant_type {
        GGMLQuantType::F32 => copy_f32(raw, n_elements),
        GGMLQuantType::F16 => dequantize_f16(raw, n_elements),
        GGMLQuantType::Q4_0 => dequantize_q4_0(raw, n_elements),
        GGMLQuantType::Q4_1 => dequantize_q4_1(raw, n_elements),
        GGMLQuantType::Q8_0 => dequantize_q8_0(raw, n_elements),
        other => {
            return Err(format!(
                "dequantization for {:?} is not yet implemented",
                other
            ))
        }
    };

    let shape: Vec<usize> = info.dimensions.iter().map(|&d| d as usize).collect();
    let strides = compute_row_major_strides(&shape);
    let byte_data: Vec<u8> = f32_data
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();

    Ok(FastTensor {
        data: byte_data,
        shape,
        strides,
        dtype: DType::F32,
        layout: Layout::RowMajor,
    })
}

fn compute_row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0usize; shape.len()];
    if shape.is_empty() {
        return strides;
    }
    strides[shape.len() - 1] = 1;
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

// ─── Global state ───────────────────────────────────────────────────────────

static LOADED_FILES: LazyLock<Mutex<Vec<GGUFFile>>> =
    LazyLock::new(|| Mutex::new(Vec::new()));

// ─── Builtins ───────────────────────────────────────────────────────────────

/// gguf_load(path: String) -> Int (handle index)
fn builtin_gguf_load(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("gguf_load expects 1 argument: path".into());
    }
    let path = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("gguf_load: path must be a string".into()),
    };

    let data = std::fs::read(&path)
        .map_err(|e| format!("gguf_load: failed to read '{}': {}", path, e))?;

    let file = parse_gguf(&data)?;

    let mut files = LOADED_FILES.lock().map_err(|e| format!("lock error: {}", e))?;
    let idx = files.len();
    files.push(file);

    Ok(Value::Int(idx as i128))
}

/// gguf_info(handle: Int) -> Struct { version, tensor_count, metadata }
fn builtin_gguf_info(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("gguf_info expects 1 argument: handle".into());
    }
    let handle = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("gguf_info: handle must be an Int".into()),
    };

    let files = LOADED_FILES.lock().map_err(|e| format!("lock error: {}", e))?;
    let file = files
        .get(handle)
        .ok_or_else(|| format!("gguf_info: invalid handle {}", handle))?;

    let mut meta_map = HashMap::new();
    for (key, val) in &file.metadata {
        meta_map.insert(key.clone(), val.to_value());
    }

    let mut fields = HashMap::new();
    fields.insert("version".to_string(), Value::Int(file.version as i128));
    fields.insert(
        "tensor_count".to_string(),
        Value::Int(file.tensor_infos.len() as i128),
    );
    fields.insert(
        "metadata".to_string(),
        Value::Struct {
            name: "GGUFMetadata".to_string(),
            fields: meta_map,
        },
    );

    Ok(Value::Struct {
        name: "GGUFInfo".to_string(),
        fields,
    })
}

/// gguf_list_tensors(handle: Int) -> Array of Struct { name, dims, quant_type }
fn builtin_gguf_list_tensors(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 {
        return Err("gguf_list_tensors expects 1 argument: handle".into());
    }
    let handle = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("gguf_list_tensors: handle must be an Int".into()),
    };

    let files = LOADED_FILES.lock().map_err(|e| format!("lock error: {}", e))?;
    let file = files
        .get(handle)
        .ok_or_else(|| format!("gguf_list_tensors: invalid handle {}", handle))?;

    let tensors: Vec<Value> = file
        .tensor_infos
        .iter()
        .map(|ti| {
            let mut fields = HashMap::new();
            fields.insert("name".to_string(), Value::String(ti.name.clone()));
            fields.insert(
                "dims".to_string(),
                Value::Array(
                    ti.dimensions
                        .iter()
                        .map(|&d| Value::Int(d as i128))
                        .collect(),
                ),
            );
            fields.insert(
                "quant_type".to_string(),
                Value::String(format!("{:?}", ti.quant_type)),
            );
            fields.insert(
                "n_elements".to_string(),
                Value::Int(ti.n_elements() as i128),
            );
            Value::Struct {
                name: "GGUFTensorInfo".to_string(),
                fields,
            }
        })
        .collect();

    Ok(Value::Array(tensors))
}

/// gguf_get_tensor(handle: Int, name: String) -> Tensor
fn builtin_gguf_get_tensor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 {
        return Err("gguf_get_tensor expects 2 arguments: handle, name".into());
    }
    let handle = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("gguf_get_tensor: handle must be an Int".into()),
    };
    let name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("gguf_get_tensor: name must be a string".into()),
    };

    let files = LOADED_FILES.lock().map_err(|e| format!("lock error: {}", e))?;
    let file = files
        .get(handle)
        .ok_or_else(|| format!("gguf_get_tensor: invalid handle {}", handle))?;

    let tensor = get_tensor(file, &name)?;
    Ok(Value::Tensor(tensor))
}

// ─── Registration ───────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert(
        "gguf_load".to_string(),
        FnDef::Builtin(builtin_gguf_load),
    );
    env.functions.insert(
        "gguf_info".to_string(),
        FnDef::Builtin(builtin_gguf_info),
    );
    env.functions.insert(
        "gguf_list_tensors".to_string(),
        FnDef::Builtin(builtin_gguf_list_tensors),
    );
    env.functions.insert(
        "gguf_get_tensor".to_string(),
        FnDef::Builtin(builtin_gguf_get_tensor),
    );
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal valid GGUF v3 binary with given metadata and tensor descriptors.
    fn build_gguf_bytes(
        metadata: &[(&str, GGUFValueType, &[u8])],
        tensors: &[(&str, &[u64], GGMLQuantType, &[u8])],
    ) -> Vec<u8> {
        let mut buf = Vec::new();

        // Magic
        buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
        // Version 3
        buf.extend_from_slice(&3u32.to_le_bytes());
        // Tensor count
        buf.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        // Metadata KV count
        buf.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

        // Metadata entries
        for &(key, _vtype, raw) in metadata {
            // Key string (length + bytes)
            buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
            buf.extend_from_slice(key.as_bytes());
            // Value type
            buf.extend_from_slice(&(_vtype as u32).to_le_bytes());
            // Raw value bytes
            buf.extend_from_slice(raw);
        }

        // Tensor descriptors
        let mut current_offset: u64 = 0;
        let mut tensor_data_blobs: Vec<&[u8]> = Vec::new();
        for &(name, dims, quant_type, data) in tensors {
            // Name
            buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
            buf.extend_from_slice(name.as_bytes());
            // n_dims
            buf.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            // dimensions
            for &d in dims {
                buf.extend_from_slice(&d.to_le_bytes());
            }
            // quant type
            buf.extend_from_slice(&(quant_type as u32).to_le_bytes());
            // offset
            buf.extend_from_slice(&current_offset.to_le_bytes());
            current_offset += data.len() as u64;
            tensor_data_blobs.push(data);
        }

        // Pad to alignment
        let alignment = GGUF_DEFAULT_ALIGNMENT;
        let padding = (alignment - (buf.len() % alignment)) % alignment;
        buf.extend(std::iter::repeat(0u8).take(padding));

        // Tensor data
        for blob in tensor_data_blobs {
            buf.extend_from_slice(blob);
        }

        buf
    }

    #[test]
    fn test_parse_header() {
        let data = build_gguf_bytes(&[], &[]);
        let file = parse_gguf(&data).unwrap();
        assert_eq!(file.version, 3);
        assert_eq!(file.tensor_infos.len(), 0);
        assert!(file.metadata.is_empty());
    }

    #[test]
    fn test_parse_with_metadata() {
        // U32 value = 42
        let val_bytes = 42u32.to_le_bytes();
        let data = build_gguf_bytes(
            &[("test.key", GGUFValueType::U32, &val_bytes)],
            &[],
        );
        let file = parse_gguf(&data).unwrap();
        match file.metadata.get("test.key") {
            Some(GGUFValue::U32(v)) => assert_eq!(*v, 42),
            other => panic!("expected U32(42), got {:?}", other),
        }
    }

    #[test]
    fn test_invalid_magic() {
        let data = vec![0u8; 32];
        assert!(parse_gguf(&data).is_err());
    }

    #[test]
    fn test_dequantize_q4_0_basic() {
        // One block: 18 bytes, 32 values
        // Scale = 1.0 as f16 = 0x3C00
        let mut block = vec![0u8; 18];
        block[0] = 0x00; // f16 1.0 = 0x3C00
        block[1] = 0x3C;
        // All nibbles = 8 (which becomes 8-8 = 0) except first byte
        for i in 2..18 {
            block[i] = 0x88; // lo=8, hi=8 → both become 0
        }
        // Set first data byte so lo=9 (val=1), hi=10 (val=2)
        block[2] = 0xA9; // lo=0x9=9, hi=0xA=10

        let result = dequantize_q4_0(&block, 32);
        assert_eq!(result.len(), 32);
        // First value: (9-8) * 1.0 = 1.0
        assert!((result[0] - 1.0).abs() < 1e-4);
        // Second value: (10-8) * 1.0 = 2.0
        assert!((result[1] - 2.0).abs() < 1e-4);
        // Rest should be 0
        assert!((result[2] - 0.0).abs() < 1e-4);
    }

    #[test]
    fn test_dequantize_q8_0_basic() {
        // One block: 34 bytes, 32 values
        // Scale = 1.0 as f16 = 0x3C00
        let mut block = vec![0u8; 34];
        block[0] = 0x00;
        block[1] = 0x3C;
        // Set some i8 values
        block[2] = 3u8;   // i8 = 3
        block[3] = 0xFD;  // i8 = -3 (two's complement)

        let result = dequantize_q8_0(&block, 32);
        assert_eq!(result.len(), 32);
        assert!((result[0] - 3.0).abs() < 1e-4);
        assert!((result[1] - (-3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_roundtrip_f32() {
        let values: Vec<f32> = vec![1.0, 2.5, -3.14, 0.0, 42.0, 1e-6, -1e6, 0.5];
        let raw: Vec<u8> = values.iter().flat_map(|f| f.to_le_bytes()).collect();

        // Build a GGUF file with one F32 tensor
        let dims: Vec<u64> = vec![8];
        let data = build_gguf_bytes(&[], &[("test_tensor", &dims, GGMLQuantType::F32, &raw)]);

        let file = parse_gguf(&data).unwrap();
        assert_eq!(file.tensor_infos.len(), 1);
        assert_eq!(file.tensor_infos[0].name, "test_tensor");

        let tensor = get_tensor(&file, "test_tensor").unwrap();
        assert_eq!(tensor.shape, vec![8]);
        assert_eq!(tensor.dtype, DType::F32);

        // Read back f32 values
        let result: Vec<f32> = tensor
            .data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert_eq!(result.len(), values.len());
        for (a, b) in result.iter().zip(values.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "mismatch: {} vs {}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_dequantize_f16_roundtrip() {
        // f16 1.0 = 0x3C00, f16 -1.0 = 0xBC00, f16 0.5 = 0x3800
        let raw = vec![0x00, 0x3C, 0x00, 0xBC, 0x00, 0x38];
        let result = dequantize_f16(&raw, 3);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 1.0).abs() < 1e-3);
        assert!((result[1] - (-1.0)).abs() < 1e-3);
        assert!((result[2] - 0.5).abs() < 1e-3);
    }
}
