// Data loading module for Vortex
// CSV/JSON/binary loading with batching, shuffle, and prefetch support

use crate::interpreter::{Env, FnDef, Value};
use crate::tensor_engine::FastTensor;
#[allow(unused_imports)]
use crate::tensor_engine::{DType, Layout};
use std::sync::{Arc, LazyLock, Mutex, mpsc};
use std::io::Read;

// ─── LCG Random Number Generator ───────────────────────────────────────────

struct LcgRng {
    state: u64,
}

impl LcgRng {
    fn new(seed: u64) -> Self {
        LcgRng { state: seed.wrapping_add(1) }
    }

    fn next_u64(&mut self) -> u64 {
        // LCG parameters from Numerical Recipes
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }

    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() % bound as u64) as usize
    }
}

// ─── Dataset ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Dataset {
    pub samples: Vec<Vec<f32>>,
    pub labels: Option<Vec<Vec<f32>>>,
    pub feature_dim: usize,
    pub label_dim: usize,
    pub len: usize,
}

impl Dataset {
    pub fn new(samples: Vec<Vec<f32>>, labels: Option<Vec<Vec<f32>>>, feature_dim: usize, label_dim: usize) -> Self {
        let len = samples.len();
        Dataset { samples, labels, feature_dim, label_dim, len }
    }

    pub fn empty() -> Self {
        Dataset { samples: vec![], labels: None, feature_dim: 0, label_dim: 0, len: 0 }
    }
}

// ─── CSV Parser ─────────────────────────────────────────────────────────────

pub fn load_csv(path: &str, has_header: bool, label_cols: &[usize]) -> Result<Dataset, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read CSV file '{}': {}", path, e))?;

    let mut lines: Vec<&str> = content.lines().collect();
    if lines.is_empty() {
        return Err("CSV file is empty".to_string());
    }

    if has_header {
        lines.remove(0);
    }

    if lines.is_empty() {
        return Err("CSV file has no data rows".to_string());
    }

    // Determine dimensions from first row
    let first_fields: Vec<&str> = lines[0].split(',').map(|s| s.trim()).collect();
    let total_cols = first_fields.len();

    let label_set: std::collections::HashSet<usize> = label_cols.iter().cloned().collect();
    let feature_dim = total_cols - label_set.len();
    let label_dim = label_set.len();

    let mut samples = Vec::with_capacity(lines.len());
    let mut labels: Vec<Vec<f32>> = if label_dim > 0 { Vec::with_capacity(lines.len()) } else { vec![] };

    for (line_idx, line) in lines.iter().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let fields: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if fields.len() != total_cols {
            return Err(format!(
                "CSV row {} has {} columns, expected {}",
                line_idx + 1, fields.len(), total_cols
            ));
        }

        let mut sample = Vec::with_capacity(feature_dim);
        let mut label = Vec::with_capacity(label_dim);

        for (col_idx, field) in fields.iter().enumerate() {
            let val: f32 = field.parse().map_err(|e| {
                format!("CSV parse error at row {}, col {}: {} (value: '{}')", line_idx + 1, col_idx, e, field)
            })?;
            if label_set.contains(&col_idx) {
                label.push(val);
            } else {
                sample.push(val);
            }
        }

        samples.push(sample);
        if label_dim > 0 {
            labels.push(label);
        }
    }

    let len = samples.len();
    let labels_opt = if label_dim > 0 { Some(labels) } else { None };

    Ok(Dataset { samples, labels: labels_opt, feature_dim, label_dim, len })
}

// ─── JSON Parser ────────────────────────────────────────────────────────────

/// Minimal JSON array-of-objects parser for numeric data.
/// Expects format: [{"feat1": 1.0, "feat2": 2.0, "label": 0.0}, ...]
pub fn load_json(path: &str, features_key: &[&str], labels_key: &[&str]) -> Result<Dataset, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("Failed to read JSON file '{}': {}", path, e))?;

    let content = content.trim();
    if !content.starts_with('[') || !content.ends_with(']') {
        return Err("JSON must be an array of objects".to_string());
    }

    let inner = &content[1..content.len() - 1];
    let objects = split_json_objects(inner)?;

    if objects.is_empty() {
        return Err("JSON array is empty".to_string());
    }

    let feature_dim = features_key.len();
    let label_dim = labels_key.len();
    let mut samples = Vec::with_capacity(objects.len());
    let mut labels: Vec<Vec<f32>> = if label_dim > 0 { Vec::with_capacity(objects.len()) } else { vec![] };

    for (obj_idx, obj) in objects.iter().enumerate() {
        let fields = parse_json_object(obj)
            .map_err(|e| format!("JSON parse error at object {}: {}", obj_idx, e))?;

        let mut sample = Vec::with_capacity(feature_dim);
        for key in features_key {
            let val = fields.get(*key).ok_or_else(|| {
                format!("Missing feature key '{}' in object {}", key, obj_idx)
            })?;
            sample.push(*val);
        }
        samples.push(sample);

        if label_dim > 0 {
            let mut label = Vec::with_capacity(label_dim);
            for key in labels_key {
                let val = fields.get(*key).ok_or_else(|| {
                    format!("Missing label key '{}' in object {}", key, obj_idx)
                })?;
                label.push(*val);
            }
            labels.push(label);
        }
    }

    let len = samples.len();
    let labels_opt = if label_dim > 0 { Some(labels) } else { None };

    Ok(Dataset { samples, labels: labels_opt, feature_dim, label_dim, len })
}

fn split_json_objects(s: &str) -> Result<Vec<String>, String> {
    let mut objects = Vec::new();
    let mut depth = 0i32;
    let mut start = None;
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0;
    let mut in_string = false;
    let mut escape_next = false;

    while i < chars.len() {
        let c = chars[i];
        if escape_next {
            escape_next = false;
            i += 1;
            continue;
        }
        if c == '\\' && in_string {
            escape_next = true;
            i += 1;
            continue;
        }
        if c == '"' {
            in_string = !in_string;
        } else if !in_string {
            if c == '{' {
                if depth == 0 {
                    start = Some(i);
                }
                depth += 1;
            } else if c == '}' {
                depth -= 1;
                if depth == 0 {
                    if let Some(s_idx) = start {
                        objects.push(s[s_idx..=i].to_string());
                        start = None;
                    }
                }
            }
        }
        i += 1;
    }

    if depth != 0 {
        return Err("Unbalanced braces in JSON".to_string());
    }

    Ok(objects)
}

fn parse_json_object(obj: &str) -> Result<std::collections::HashMap<String, f32>, String> {
    let obj = obj.trim();
    if !obj.starts_with('{') || !obj.ends_with('}') {
        return Err("Expected JSON object".to_string());
    }
    let inner = &obj[1..obj.len() - 1];
    let mut fields = std::collections::HashMap::new();

    let pairs = split_json_pairs(inner);
    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() {
            continue;
        }
        // Find the colon separating key from value
        let colon_pos = find_json_colon(pair)?;
        let key_part = pair[..colon_pos].trim();
        let val_part = pair[colon_pos + 1..].trim();

        // Extract key (strip quotes)
        if !key_part.starts_with('"') || !key_part.ends_with('"') {
            return Err(format!("Invalid JSON key: {}", key_part));
        }
        let key = key_part[1..key_part.len() - 1].to_string();

        // Parse numeric value
        let val: f32 = val_part.parse().map_err(|e| {
            format!("Cannot parse value '{}' for key '{}': {}", val_part, key, e)
        })?;
        fields.insert(key, val);
    }

    Ok(fields)
}

fn find_json_colon(s: &str) -> Result<usize, String> {
    let mut in_string = false;
    for (i, c) in s.chars().enumerate() {
        if c == '"' {
            in_string = !in_string;
        } else if c == ':' && !in_string {
            return Ok(i);
        }
    }
    Err(format!("No colon found in JSON pair: {}", s))
}

fn split_json_pairs(s: &str) -> Vec<&str> {
    let mut pairs = Vec::new();
    let mut depth = 0i32;
    let mut in_string = false;
    let mut start = 0;

    for (i, c) in s.chars().enumerate() {
        match c {
            '"' => in_string = !in_string,
            '{' | '[' if !in_string => depth += 1,
            '}' | ']' if !in_string => depth -= 1,
            ',' if !in_string && depth == 0 => {
                pairs.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    if start < s.len() {
        pairs.push(&s[start..]);
    }
    pairs
}

// ─── Binary Loader ──────────────────────────────────────────────────────────

pub fn load_binary(path: &str, feature_dim: usize, label_dim: usize, _dtype: &str) -> Result<Dataset, String> {
    let mut file = std::fs::File::open(path)
        .map_err(|e| format!("Failed to open binary file '{}': {}", path, e))?;

    let mut raw = Vec::new();
    file.read_to_end(&mut raw)
        .map_err(|e| format!("Failed to read binary file '{}': {}", path, e))?;

    let total_dim = feature_dim + label_dim;
    let f32_size = std::mem::size_of::<f32>();
    let row_bytes = total_dim * f32_size;

    if raw.len() % row_bytes != 0 {
        return Err(format!(
            "Binary file size {} is not divisible by row size {} (total_dim={} * 4)",
            raw.len(), row_bytes, total_dim
        ));
    }

    let num_samples = raw.len() / row_bytes;
    let mut samples = Vec::with_capacity(num_samples);
    let mut labels_vec: Vec<Vec<f32>> = if label_dim > 0 { Vec::with_capacity(num_samples) } else { vec![] };

    for i in 0..num_samples {
        let row_start = i * row_bytes;
        let mut all_vals = Vec::with_capacity(total_dim);
        for j in 0..total_dim {
            let offset = row_start + j * f32_size;
            let bytes: [u8; 4] = [raw[offset], raw[offset + 1], raw[offset + 2], raw[offset + 3]];
            all_vals.push(f32::from_le_bytes(bytes));
        }

        let sample = all_vals[..feature_dim].to_vec();
        samples.push(sample);

        if label_dim > 0 {
            let label = all_vals[feature_dim..].to_vec();
            labels_vec.push(label);
        }
    }

    let len = samples.len();
    let labels_opt = if label_dim > 0 { Some(labels_vec) } else { None };

    Ok(Dataset { samples, labels: labels_opt, feature_dim, label_dim, len })
}

// ─── DataLoader ─────────────────────────────────────────────────────────────

pub struct DataLoader {
    pub dataset: Arc<Dataset>,
    pub batch_size: usize,
    pub shuffle: bool,
    pub current_index: usize,
    pub indices: Vec<usize>,
    pub drop_last: bool,
    rng: LcgRng,
    prefetch_rx: Option<mpsc::Receiver<(FastTensor, Option<FastTensor>)>>,
    prefetch_active: bool,
}

impl DataLoader {
    pub fn new(dataset: Arc<Dataset>, batch_size: usize, shuffle: bool, drop_last: bool) -> Self {
        let mut loader = DataLoader {
            indices: (0..dataset.len).collect(),
            dataset,
            batch_size,
            shuffle,
            current_index: 0,
            drop_last,
            rng: LcgRng::new(42),
            prefetch_rx: None,
            prefetch_active: false,
        };
        if shuffle {
            loader.shuffle_indices();
        }
        loader
    }

    fn shuffle_indices(&mut self) {
        // Fisher-Yates shuffle
        let n = self.indices.len();
        for i in (1..n).rev() {
            let j = self.rng.next_usize(i + 1);
            self.indices.swap(i, j);
        }
    }

    pub fn next_batch(&mut self) -> Option<(FastTensor, Option<FastTensor>)> {
        // Check prefetch first
        if self.prefetch_active {
            if let Some(ref rx) = self.prefetch_rx {
                if let Ok(batch) = rx.try_recv() {
                    self.advance_index();
                    self.maybe_start_prefetch();
                    return Some(batch);
                }
            }
        }

        self.build_current_batch()
    }

    fn advance_index(&mut self) {
        self.current_index += self.batch_size;
    }

    fn build_current_batch(&mut self) -> Option<(FastTensor, Option<FastTensor>)> {
        if self.current_index >= self.dataset.len {
            return None;
        }

        let remaining = self.dataset.len - self.current_index;
        let actual_batch = if remaining < self.batch_size {
            if self.drop_last {
                return None;
            }
            remaining
        } else {
            self.batch_size
        };

        let batch_indices: Vec<usize> = self.indices[self.current_index..self.current_index + actual_batch].to_vec();
        self.current_index += actual_batch;

        // Build feature tensor [batch_size, feature_dim]
        let feature_dim = self.dataset.feature_dim;
        let mut feat_data = Vec::with_capacity(actual_batch * feature_dim);
        for &idx in &batch_indices {
            feat_data.extend_from_slice(&self.dataset.samples[idx]);
        }
        let feat_tensor = FastTensor::from_f32(vec![actual_batch, feature_dim], &feat_data);

        // Build label tensor if present
        let label_tensor = if let Some(ref labels) = self.dataset.labels {
            let label_dim = self.dataset.label_dim;
            let mut label_data = Vec::with_capacity(actual_batch * label_dim);
            for &idx in &batch_indices {
                label_data.extend_from_slice(&labels[idx]);
            }
            Some(FastTensor::from_f32(vec![actual_batch, label_dim], &label_data))
        } else {
            None
        };

        Some((feat_tensor, label_tensor))
    }

    pub fn reset(&mut self) {
        self.current_index = 0;
        if self.shuffle {
            self.shuffle_indices();
        }
        self.prefetch_active = false;
        self.prefetch_rx = None;
    }

    pub fn len(&self) -> usize {
        if self.drop_last {
            self.dataset.len / self.batch_size
        } else {
            (self.dataset.len + self.batch_size - 1) / self.batch_size
        }
    }

    pub fn from_tensors(features: &FastTensor, labels: Option<&FastTensor>, batch_size: usize) -> Result<Self, String> {
        if features.shape.len() != 2 {
            return Err("Features tensor must be 2D [num_samples, feature_dim]".to_string());
        }
        let num_samples = features.shape[0];
        let feature_dim = features.shape[1];

        // Extract f32 data from feature tensor
        let mut samples = Vec::with_capacity(num_samples);
        for i in 0..num_samples {
            let mut sample = Vec::with_capacity(feature_dim);
            for j in 0..feature_dim {
                let offset = (i * feature_dim + j) * 4;
                if offset + 4 > features.data.len() {
                    return Err("Feature tensor data too short".to_string());
                }
                let bytes: [u8; 4] = [
                    features.data[offset],
                    features.data[offset + 1],
                    features.data[offset + 2],
                    features.data[offset + 3],
                ];
                sample.push(f32::from_le_bytes(bytes));
            }
            samples.push(sample);
        }

        let (labels_vec, label_dim) = if let Some(lt) = labels {
            if lt.shape.len() != 2 {
                return Err("Labels tensor must be 2D [num_samples, label_dim]".to_string());
            }
            if lt.shape[0] != num_samples {
                return Err("Labels and features must have same number of samples".to_string());
            }
            let ld = lt.shape[1];
            let mut lvec = Vec::with_capacity(num_samples);
            for i in 0..num_samples {
                let mut label = Vec::with_capacity(ld);
                for j in 0..ld {
                    let offset = (i * ld + j) * 4;
                    if offset + 4 > lt.data.len() {
                        return Err("Label tensor data too short".to_string());
                    }
                    let bytes: [u8; 4] = [
                        lt.data[offset],
                        lt.data[offset + 1],
                        lt.data[offset + 2],
                        lt.data[offset + 3],
                    ];
                    label.push(f32::from_le_bytes(bytes));
                }
                lvec.push(label);
            }
            (Some(lvec), ld)
        } else {
            (None, 0)
        };

        let dataset = Dataset::new(samples, labels_vec, feature_dim, label_dim);
        Ok(DataLoader::new(Arc::new(dataset), batch_size, false, false))
    }

    /// Enable prefetching: spawns a background thread to pre-load the next batch
    pub fn enable_prefetch(&mut self) {
        self.prefetch_active = true;
        self.maybe_start_prefetch();
    }

    fn maybe_start_prefetch(&mut self) {
        if !self.prefetch_active {
            return;
        }
        if self.current_index >= self.dataset.len {
            return;
        }

        let remaining = self.dataset.len - self.current_index;
        let actual_batch = if remaining < self.batch_size {
            if self.drop_last {
                return;
            }
            remaining
        } else {
            self.batch_size
        };

        let dataset = Arc::clone(&self.dataset);
        let batch_indices: Vec<usize> = self.indices[self.current_index..self.current_index + actual_batch].to_vec();

        let (tx, rx) = mpsc::channel();
        self.prefetch_rx = Some(rx);

        std::thread::spawn(move || {
            let feature_dim = dataset.feature_dim;
            let mut feat_data = Vec::with_capacity(actual_batch * feature_dim);
            for &idx in &batch_indices {
                feat_data.extend_from_slice(&dataset.samples[idx]);
            }
            let feat_tensor = FastTensor::from_f32(vec![actual_batch, feature_dim], &feat_data);

            let label_tensor = if let Some(ref labels) = dataset.labels {
                let label_dim = dataset.label_dim;
                let mut label_data = Vec::with_capacity(actual_batch * label_dim);
                for &idx in &batch_indices {
                    label_data.extend_from_slice(&labels[idx]);
                }
                Some(FastTensor::from_f32(vec![actual_batch, label_dim], &label_data))
            } else {
                None
            };

            let _ = tx.send((feat_tensor, label_tensor));
        });
    }
}

// ─── Global State ───────────────────────────────────────────────────────────

static DATASETS: LazyLock<Mutex<Vec<Dataset>>> = LazyLock::new(|| Mutex::new(Vec::new()));
static LOADERS: LazyLock<Mutex<Vec<DataLoader>>> = LazyLock::new(|| Mutex::new(Vec::new()));

// ─── Builtins ───────────────────────────────────────────────────────────────

/// dataset_load(format, path, ...options) -> dataset_id
/// CSV:    dataset_load("csv", path, has_header, [label_col1, label_col2, ...])
/// JSON:   dataset_load("json", path, [feat_keys...], [label_keys...])
/// Binary: dataset_load("binary", path, feature_dim, label_dim, dtype)
fn builtin_dataset_load(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("dataset_load requires at least 2 arguments: format, path".to_string());
    }

    let format = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("dataset_load: first argument must be format string".to_string()),
    };
    let path = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("dataset_load: second argument must be path string".to_string()),
    };

    let dataset = match format.as_str() {
        "csv" => {
            let has_header = if args.len() > 2 {
                match &args[2] {
                    Value::Bool(b) => *b,
                    _ => return Err("dataset_load csv: has_header must be bool".to_string()),
                }
            } else {
                true
            };
            let label_cols: Vec<usize> = if args.len() > 3 {
                match &args[3] {
                    Value::Array(arr) => {
                        arr.iter().map(|v| match v {
                            Value::Int(i) => Ok(*i as usize),
                            _ => Err("label_cols must be array of ints".to_string()),
                        }).collect::<Result<Vec<_>, _>>()?
                    }
                    _ => return Err("dataset_load csv: label_cols must be array".to_string()),
                }
            } else {
                vec![]
            };
            load_csv(&path, has_header, &label_cols)?
        }
        "json" => {
            if args.len() < 4 {
                return Err("dataset_load json requires: path, feature_keys, label_keys".to_string());
            }
            let feat_keys: Vec<String> = match &args[2] {
                Value::Array(arr) => {
                    arr.iter().map(|v| match v {
                        Value::String(s) => Ok(s.clone()),
                        _ => Err("feature_keys must be array of strings".to_string()),
                    }).collect::<Result<Vec<_>, _>>()?
                }
                _ => return Err("dataset_load json: feature_keys must be array".to_string()),
            };
            let label_keys: Vec<String> = match &args[3] {
                Value::Array(arr) => {
                    arr.iter().map(|v| match v {
                        Value::String(s) => Ok(s.clone()),
                        _ => Err("label_keys must be array of strings".to_string()),
                    }).collect::<Result<Vec<_>, _>>()?
                }
                _ => return Err("dataset_load json: label_keys must be array".to_string()),
            };
            let feat_refs: Vec<&str> = feat_keys.iter().map(|s| s.as_str()).collect();
            let label_refs: Vec<&str> = label_keys.iter().map(|s| s.as_str()).collect();
            load_json(&path, &feat_refs, &label_refs)?
        }
        "binary" => {
            if args.len() < 5 {
                return Err("dataset_load binary requires: path, feature_dim, label_dim, dtype".to_string());
            }
            let feature_dim = match &args[2] {
                Value::Int(i) => *i as usize,
                _ => return Err("dataset_load binary: feature_dim must be int".to_string()),
            };
            let label_dim = match &args[3] {
                Value::Int(i) => *i as usize,
                _ => return Err("dataset_load binary: label_dim must be int".to_string()),
            };
            let dtype = match &args[4] {
                Value::String(s) => s.clone(),
                _ => return Err("dataset_load binary: dtype must be string".to_string()),
            };
            load_binary(&path, feature_dim, label_dim, &dtype)?
        }
        _ => return Err(format!("Unknown dataset format: '{}'", format)),
    };

    let mut datasets = DATASETS.lock().unwrap();
    let id = datasets.len();
    datasets.push(dataset);
    Ok(Value::Int(id as i128))
}

/// dataset_from_tensor(features_tensor, labels_tensor_or_none) -> dataset_id
fn builtin_dataset_from_tensor(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("dataset_from_tensor requires at least 1 argument".to_string());
    }

    let features = match &args[0] {
        Value::Tensor(t) => t.clone(),
        _ => return Err("dataset_from_tensor: first argument must be a tensor".to_string()),
    };

    if features.shape.len() != 2 {
        return Err("dataset_from_tensor: features must be 2D [samples, features]".to_string());
    }

    let num_samples = features.shape[0];
    let feature_dim = features.shape[1];

    let mut samples = Vec::with_capacity(num_samples);
    for i in 0..num_samples {
        let mut sample = Vec::with_capacity(feature_dim);
        for j in 0..feature_dim {
            let offset = (i * feature_dim + j) * 4;
            if offset + 4 > features.data.len() {
                return Err("Feature tensor data too short".to_string());
            }
            let bytes: [u8; 4] = [
                features.data[offset],
                features.data[offset + 1],
                features.data[offset + 2],
                features.data[offset + 3],
            ];
            sample.push(f32::from_le_bytes(bytes));
        }
        samples.push(sample);
    }

    let (labels_vec, label_dim) = if args.len() > 1 {
        match &args[1] {
            Value::Tensor(lt) => {
                if lt.shape.len() != 2 {
                    return Err("Labels tensor must be 2D".to_string());
                }
                if lt.shape[0] != num_samples {
                    return Err("Labels and features must have same sample count".to_string());
                }
                let ld = lt.shape[1];
                let mut lvec = Vec::with_capacity(num_samples);
                for i in 0..num_samples {
                    let mut label = Vec::with_capacity(ld);
                    for j in 0..ld {
                        let offset = (i * ld + j) * 4;
                        if offset + 4 > lt.data.len() {
                            return Err("Label tensor data too short".to_string());
                        }
                        let bytes: [u8; 4] = [
                            lt.data[offset],
                            lt.data[offset + 1],
                            lt.data[offset + 2],
                            lt.data[offset + 3],
                        ];
                        label.push(f32::from_le_bytes(bytes));
                    }
                    lvec.push(label);
                }
                (Some(lvec), ld)
            }
            Value::Option(None) | Value::Void => (None, 0),
            _ => return Err("dataset_from_tensor: second argument must be tensor or none".to_string()),
        }
    } else {
        (None, 0)
    };

    let dataset = Dataset::new(samples, labels_vec, feature_dim, label_dim);
    let mut datasets = DATASETS.lock().unwrap();
    let id = datasets.len();
    datasets.push(dataset);
    Ok(Value::Int(id as i128))
}

/// dataloader_new(dataset_id, batch_size, shuffle, drop_last) -> loader_id
fn builtin_dataloader_new(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 {
        return Err("dataloader_new requires at least 2 arguments: dataset_id, batch_size".to_string());
    }

    let dataset_id = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("dataloader_new: dataset_id must be int".to_string()),
    };
    let batch_size = match &args[1] {
        Value::Int(i) => {
            if *i <= 0 {
                return Err("dataloader_new: batch_size must be positive".to_string());
            }
            *i as usize
        }
        _ => return Err("dataloader_new: batch_size must be int".to_string()),
    };
    let shuffle = if args.len() > 2 {
        match &args[2] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };
    let drop_last = if args.len() > 3 {
        match &args[3] {
            Value::Bool(b) => *b,
            _ => false,
        }
    } else {
        false
    };

    let dataset = {
        let datasets = DATASETS.lock().unwrap();
        if dataset_id >= datasets.len() {
            return Err(format!("Invalid dataset_id: {}", dataset_id));
        }
        Arc::new(datasets[dataset_id].clone())
    };

    let loader = DataLoader::new(dataset, batch_size, shuffle, drop_last);

    let mut loaders = LOADERS.lock().unwrap();
    let id = loaders.len();
    loaders.push(loader);
    Ok(Value::Int(id as i128))
}

/// dataloader_next_batch(loader_id) -> [features_tensor, labels_tensor_or_none] or none
fn builtin_dataloader_next_batch(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("dataloader_next_batch requires 1 argument: loader_id".to_string());
    }

    let loader_id = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("dataloader_next_batch: loader_id must be int".to_string()),
    };

    let mut loaders = LOADERS.lock().unwrap();
    if loader_id >= loaders.len() {
        return Err(format!("Invalid loader_id: {}", loader_id));
    }

    match loaders[loader_id].next_batch() {
        Some((features, labels)) => {
            let label_val = match labels {
                Some(lt) => Value::Tensor(lt),
                None => Value::Option(None),
            };
            Ok(Value::Array(vec![Value::Tensor(features), label_val]))
        }
        None => Ok(Value::Option(None)),
    }
}

/// dataloader_reset(loader_id)
fn builtin_dataloader_reset(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("dataloader_reset requires 1 argument: loader_id".to_string());
    }

    let loader_id = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("dataloader_reset: loader_id must be int".to_string()),
    };

    let mut loaders = LOADERS.lock().unwrap();
    if loader_id >= loaders.len() {
        return Err(format!("Invalid loader_id: {}", loader_id));
    }

    loaders[loader_id].reset();
    Ok(Value::Void)
}

/// dataloader_len(loader_id) -> number of batches
fn builtin_dataloader_len(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.is_empty() {
        return Err("dataloader_len requires 1 argument: loader_id".to_string());
    }

    let loader_id = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("dataloader_len: loader_id must be int".to_string()),
    };

    let loaders = LOADERS.lock().unwrap();
    if loader_id >= loaders.len() {
        return Err(format!("Invalid loader_id: {}", loader_id));
    }

    Ok(Value::Int(loaders[loader_id].len() as i128))
}

// ─── Registration ───────────────────────────────────────────────────────────

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("dataset_load".into(), FnDef::Builtin(builtin_dataset_load));
    env.functions.insert("dataset_from_tensor".into(), FnDef::Builtin(builtin_dataset_from_tensor));
    env.functions.insert("dataloader_new".into(), FnDef::Builtin(builtin_dataloader_new));
    env.functions.insert("dataloader_next_batch".into(), FnDef::Builtin(builtin_dataloader_next_batch));
    env.functions.insert("dataloader_reset".into(), FnDef::Builtin(builtin_dataloader_reset));
    env.functions.insert("dataloader_len".into(), FnDef::Builtin(builtin_dataloader_len));
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicU64, Ordering as AtOrd};
    static TEST_COUNTER: AtomicU64 = AtomicU64::new(0);

    fn make_test_file(content: &str, ext: &str) -> String {
        let id = TEST_COUNTER.fetch_add(1, AtOrd::SeqCst);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("vortex_test_{}_{}.{}", std::process::id(), id, ext));
        std::fs::write(&path, content).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn test_csv_basic() {
        let path = make_test_file("a,b,c\n1.0,2.0,3.0\n4.0,5.0,6.0\n7.0,8.0,9.0\n", "csv");
        let ds = load_csv(&path, true, &[]).unwrap();
        assert_eq!(ds.len, 3);
        assert_eq!(ds.feature_dim, 3);
        assert_eq!(ds.label_dim, 0);
        assert!(ds.labels.is_none());
        assert_eq!(ds.samples[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(ds.samples[2], vec![7.0, 8.0, 9.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_csv_with_labels() {
        let path = make_test_file("f1,f2,label\n1.0,2.0,0.0\n3.0,4.0,1.0\n", "csv");
        let ds = load_csv(&path, true, &[2]).unwrap();
        assert_eq!(ds.feature_dim, 2);
        assert_eq!(ds.label_dim, 1);
        assert_eq!(ds.samples[0], vec![1.0, 2.0]);
        assert_eq!(ds.labels.as_ref().unwrap()[0], vec![0.0]);
        assert_eq!(ds.labels.as_ref().unwrap()[1], vec![1.0]);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_csv_no_header() {
        let path = make_test_file("1.0,2.0\n3.0,4.0\n", "csv");
        let ds = load_csv(&path, false, &[]).unwrap();
        assert_eq!(ds.len, 2);
        assert_eq!(ds.feature_dim, 2);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_batch_iteration() {
        let samples = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
            vec![5.0, 6.0],
            vec![7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let ds = Dataset::new(samples, None, 2, 0);
        let mut loader = DataLoader::new(Arc::new(ds), 2, false, false);

        assert_eq!(loader.len(), 3); // ceil(5/2) = 3

        let b1 = loader.next_batch().unwrap();
        assert_eq!(b1.0.shape, vec![2, 2]);
        assert!(b1.1.is_none());

        let b2 = loader.next_batch().unwrap();
        assert_eq!(b2.0.shape, vec![2, 2]);

        let b3 = loader.next_batch().unwrap();
        assert_eq!(b3.0.shape, vec![1, 2]); // last partial batch

        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_batch_drop_last() {
        let samples = vec![
            vec![1.0], vec![2.0], vec![3.0], vec![4.0], vec![5.0],
        ];
        let ds = Dataset::new(samples, None, 1, 0);
        let mut loader = DataLoader::new(Arc::new(ds), 2, false, true);

        assert_eq!(loader.len(), 2); // 5/2 = 2 (drop last)

        let _ = loader.next_batch().unwrap();
        let _ = loader.next_batch().unwrap();
        assert!(loader.next_batch().is_none()); // partial batch dropped
    }

    #[test]
    fn test_shuffle_changes_order() {
        let samples: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let ds = Dataset::new(samples, None, 1, 0);
        let mut loader = DataLoader::new(Arc::new(ds), 20, true, false);

        // After shuffle, indices should differ from sequential
        let sequential: Vec<usize> = (0..20).collect();
        // Very unlikely that shuffle produces the same order for 20 elements
        assert_ne!(loader.indices, sequential);
    }

    #[test]
    fn test_reset_resets() {
        let samples = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let ds = Dataset::new(samples, None, 1, 0);
        let mut loader = DataLoader::new(Arc::new(ds), 2, false, false);

        let _ = loader.next_batch().unwrap();
        let _ = loader.next_batch().unwrap();
        assert!(loader.next_batch().is_none());

        loader.reset();
        assert_eq!(loader.current_index, 0);

        let b = loader.next_batch().unwrap();
        assert_eq!(b.0.shape, vec![2, 1]);
    }

    #[test]
    fn test_from_tensors() {
        let feat = FastTensor::from_f32(vec![4, 3], &[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
            10.0, 11.0, 12.0,
        ]);
        let labels = FastTensor::from_f32(vec![4, 1], &[0.0, 1.0, 0.0, 1.0]);
        let mut loader = DataLoader::from_tensors(&feat, Some(&labels), 2).unwrap();

        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.0.shape, vec![2, 3]);
        assert!(batch.1.is_some());
        assert_eq!(batch.1.unwrap().shape, vec![2, 1]);
    }

    #[test]
    fn test_json_parsing() {
        let content = r#"[
            {"x": 1.0, "y": 2.0, "label": 0.0},
            {"x": 3.0, "y": 4.0, "label": 1.0}
        ]"#;
        let path = make_test_file(content, "json");

        let ds = load_json(
            &path,
            &["x", "y"],
            &["label"],
        ).unwrap();

        assert_eq!(ds.len, 2);
        assert_eq!(ds.feature_dim, 2);
        assert_eq!(ds.label_dim, 1);
        assert_eq!(ds.samples[0], vec![1.0, 2.0]);
        assert_eq!(ds.labels.as_ref().unwrap()[1], vec![1.0]);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_binary_loading() {
        // Write 3 samples, feature_dim=2, label_dim=1
        let values: Vec<f32> = vec![
            1.0, 2.0, 0.0,  // sample 0
            3.0, 4.0, 1.0,  // sample 1
            5.0, 6.0, 0.0,  // sample 2
        ];
        let mut bytes = Vec::new();
        for v in &values {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        let path = make_test_file(
            &String::from_utf8_lossy(&bytes),
            "bin",
        );
        // Rewrite as raw bytes (make_test_file writes as string)
        std::fs::write(&path, &bytes).unwrap();

        let ds = load_binary(&path, 2, 1, "f32").unwrap();
        assert_eq!(ds.len, 3);
        assert_eq!(ds.feature_dim, 2);
        assert_eq!(ds.samples[1], vec![3.0, 4.0]);
        assert_eq!(ds.labels.as_ref().unwrap()[1], vec![1.0]);

        std::fs::remove_file(&path).ok();
    }
}
