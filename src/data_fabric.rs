/// Data Fabric: First-class data citizens in Vortex.
/// Typed, versioned, lineage-tracked, semantically understood data streams.

use std::collections::HashMap;
use std::sync::{LazyLock, Mutex};
use crate::interpreter::{Env, FnDef, Value};

// ── Semantic Types ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticType {
    Raw,
    Embedding { dim: usize, model: String },
    TimeSeries { frequency: String },
    Image { channels: usize, width: usize, height: usize },
    Text { encoding: String, language: Option<String> },
    Label { classes: Vec<String> },
    Probability,
    GeoCoord,
    Timestamp,
    Currency { code: String },
    DNA { alphabet: String },
    Custom(String),
}

impl std::fmt::Display for SemanticType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SemanticType::Raw => write!(f, "Raw"),
            SemanticType::Embedding { dim, model } => write!(f, "Embedding(dim={}, model={})", dim, model),
            SemanticType::TimeSeries { frequency } => write!(f, "TimeSeries(freq={})", frequency),
            SemanticType::Image { channels, width, height } => write!(f, "Image({}x{}x{})", channels, width, height),
            SemanticType::Text { encoding, language } => {
                if let Some(lang) = language {
                    write!(f, "Text({}, {})", encoding, lang)
                } else {
                    write!(f, "Text({})", encoding)
                }
            }
            SemanticType::Label { classes } => write!(f, "Label({})", classes.join(",")),
            SemanticType::Probability => write!(f, "Probability"),
            SemanticType::GeoCoord => write!(f, "GeoCoord"),
            SemanticType::Timestamp => write!(f, "Timestamp"),
            SemanticType::Currency { code } => write!(f, "Currency({})", code),
            SemanticType::DNA { alphabet } => write!(f, "DNA({})", alphabet),
            SemanticType::Custom(s) => write!(f, "Custom({})", s),
        }
    }
}

// ── Data Types ──────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum DataType {
    Int,
    Float,
    Bool,
    String,
    DateTime,
    Json,
    Array(Box<DataType>),
}

impl std::fmt::Display for DataType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DataType::Int => write!(f, "Int"),
            DataType::Float => write!(f, "Float"),
            DataType::Bool => write!(f, "Bool"),
            DataType::String => write!(f, "String"),
            DataType::DateTime => write!(f, "DateTime"),
            DataType::Json => write!(f, "Json"),
            DataType::Array(inner) => write!(f, "Array<{}>", inner),
        }
    }
}

// ── Schema ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Constraint {
    NotNull(String),
    Unique(String),
    Range { field: String, min: f64, max: f64 },
    OneOf { field: String, values: Vec<String> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Field {
    pub name: String,
    pub dtype: DataType,
    pub nullable: bool,
    pub semantic_type: SemanticType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Schema {
    pub fields: Vec<Field>,
    pub constraints: Vec<Constraint>,
}

impl Schema {
    pub fn new() -> Self {
        Self { fields: Vec::new(), constraints: Vec::new() }
    }

    pub fn field_names(&self) -> Vec<&str> {
        self.fields.iter().map(|f| f.name.as_str()).collect()
    }

    pub fn field_by_name(&self, name: &str) -> Option<&Field> {
        self.fields.iter().find(|f| f.name == name)
    }

    pub fn hash_schema(&self) -> String {
        // Simple deterministic hash of field names + types
        let mut h: u64 = 0xcbf29ce484222325;
        for field in &self.fields {
            for b in field.name.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            let type_str = format!("{}", field.dtype);
            for b in type_str.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
        format!("{:016x}", h)
    }
}

impl std::fmt::Display for Schema {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "Schema({} fields):", self.fields.len())?;
        for field in &self.fields {
            writeln!(f, "  {} : {} [{}]{}",
                field.name, field.dtype, field.semantic_type,
                if field.nullable { " (nullable)" } else { "" }
            )?;
        }
        Ok(())
    }
}

// ── Lineage ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Lineage {
    pub id: String,
    pub source_lineages: Vec<String>,
    pub transformation: String,
    pub timestamp: String,
    pub parameters: HashMap<String, String>,
}

impl Lineage {
    pub fn root(id: &str, source: &str) -> Self {
        Self {
            id: id.to_string(),
            source_lineages: Vec::new(),
            transformation: format!("load({})", source),
            timestamp: "0".to_string(),
            parameters: HashMap::new(),
        }
    }

    pub fn derived(id: &str, parents: Vec<String>, transform: &str, params: HashMap<String, String>) -> Self {
        Self {
            id: id.to_string(),
            source_lineages: parents,
            transformation: transform.to_string(),
            timestamp: "0".to_string(),
            parameters: params,
        }
    }
}

impl std::fmt::Display for Lineage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Lineage(id={}, transform={}, parents=[{}])",
            self.id, self.transformation, self.source_lineages.join(", "))
    }
}

#[derive(Debug, Clone, Default)]
pub struct LineageGraph {
    pub nodes: HashMap<String, Lineage>,
    pub edges: Vec<(String, String, String)>,
}

impl LineageGraph {
    pub fn new() -> Self { Self::default() }

    pub fn add_node(&mut self, lineage: Lineage) {
        let id = lineage.id.clone();
        for parent in &lineage.source_lineages {
            self.edges.push((parent.clone(), id.clone(), lineage.transformation.clone()));
        }
        self.nodes.insert(id, lineage);
    }

    pub fn trace_back(&self, id: &str) -> Vec<&Lineage> {
        let mut result = Vec::new();
        let mut stack = vec![id.to_string()];
        let mut visited = std::collections::HashSet::new();
        while let Some(current) = stack.pop() {
            if !visited.insert(current.clone()) { continue; }
            if let Some(node) = self.nodes.get(&current) {
                result.push(node);
                for parent in &node.source_lineages {
                    stack.push(parent.clone());
                }
            }
        }
        result
    }

    /// Detect data leakage: check if test data shares any lineage with training data
    pub fn detect_leakage(&self, train_id: &str, test_id: &str) -> bool {
        let train_ancestors: std::collections::HashSet<String> = self.trace_back(train_id)
            .iter().map(|l| l.id.clone()).collect();
        let test_ancestors: std::collections::HashSet<String> = self.trace_back(test_id)
            .iter().map(|l| l.id.clone()).collect();
        // Leakage if test and train share a non-root ancestor that involves data content
        // For simplicity: any shared ancestor beyond the original source is suspicious
        let shared: Vec<_> = train_ancestors.intersection(&test_ancestors).collect();
        // If they share a common source that's fine; leakage is if split happens after transform
        shared.len() > 1
    }
}

// ── Transforms (Lazy Pipeline) ──────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Order {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq)]
pub enum JoinType {
    Inner,
    Left,
    Right,
    Full,
}

#[derive(Debug, Clone, PartialEq)]
pub enum AggFn {
    Sum,
    Mean,
    Count,
    Min,
    Max,
    Std,
}

#[derive(Debug, Clone, PartialEq)]
pub enum NormMethod {
    Standard,   // (x - mean) / std
    MinMax,     // (x - min) / (max - min)
    L2,         // x / ||x||
}

#[derive(Debug, Clone)]
pub enum Predicate {
    Gt(String, f64),
    Lt(String, f64),
    Eq(String, String),
    NotNull(String),
    Custom(String),
}

#[derive(Debug, Clone)]
pub enum Transform {
    Filter(Predicate),
    Map(String),           // expression string for now
    GroupBy(Vec<String>),
    Aggregate(AggFn),
    Join(usize, JoinType, Vec<String>),  // stream_id, type, on_cols
    Sort(Vec<(String, Order)>),
    Sample(f64),
    Split { train: f64, val: f64, test: f64, seed: u64 },
    Normalize { method: NormMethod },
    Tokenize { vocab_size: usize },
    Embed { model: String },
    Select(Vec<String>),
}

// ── Data Version ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DataVersion {
    pub version: u64,
    pub parent: Option<u64>,
    pub timestamp: String,
    pub description: String,
    pub schema_hash: String,
    pub data_hash: String,
    pub row_count: usize,
}

impl std::fmt::Display for DataVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "v{} (rows={}, schema={}, parent={:?})",
            self.version, self.row_count, &self.schema_hash[..8.min(self.schema_hash.len())], self.parent)
    }
}

// ── Row type ────────────────────────────────────────────────────────

pub type Row = Vec<CellValue>;

#[derive(Debug, Clone, PartialEq)]
pub enum CellValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Null,
}

impl CellValue {
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            CellValue::Float(f) => Some(*f),
            CellValue::Int(i) => Some(*i as f64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            CellValue::String(s) => Some(s),
            _ => None,
        }
    }
}

impl std::fmt::Display for CellValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CellValue::Int(n) => write!(f, "{}", n),
            CellValue::Float(v) => write!(f, "{}", v),
            CellValue::Bool(b) => write!(f, "{}", b),
            CellValue::String(s) => write!(f, "{}", s),
            CellValue::Null => write!(f, "null"),
        }
    }
}

// ── DataStream ──────────────────────────────────────────────────────

static STREAM_COUNTER: LazyLock<Mutex<usize>> = LazyLock::new(|| Mutex::new(0));
static STREAM_STORE: LazyLock<Mutex<HashMap<usize, DataStream>>> = LazyLock::new(|| Mutex::new(HashMap::new()));
static LINEAGE_GRAPH: LazyLock<Mutex<LineageGraph>> = LazyLock::new(|| Mutex::new(LineageGraph::new()));

fn next_stream_id() -> usize {
    let mut c = STREAM_COUNTER.lock().unwrap();
    let id = *c;
    *c += 1;
    id
}

#[derive(Debug, Clone)]
pub struct DataStream {
    pub id: usize,
    pub schema: Schema,
    pub source: String,
    pub data: Vec<Row>,
    pub transformations: Vec<Transform>,
    pub lineage_id: String,
    pub version: DataVersion,
    pub materialized: bool,
}

impl DataStream {
    pub fn new(schema: Schema, data: Vec<Row>, source: &str) -> Self {
        let id = next_stream_id();
        let lineage_id = format!("stream_{}", id);
        let schema_hash = schema.hash_schema();
        let row_count = data.len();
        Self {
            id,
            schema,
            source: source.to_string(),
            data,
            transformations: Vec::new(),
            lineage_id: lineage_id.clone(),
            version: DataVersion {
                version: 0,
                parent: None,
                timestamp: "0".to_string(),
                description: format!("Initial load from {}", source),
                schema_hash,
                data_hash: format!("{:016x}", hash_rows(&[])), // placeholder
                row_count,
            },
            materialized: true,
        }
    }

    pub fn row_count(&self) -> usize {
        self.data.len()
    }

    pub fn col_index(&self, name: &str) -> Option<usize> {
        self.schema.fields.iter().position(|f| f.name == name)
    }

    /// Apply pending transforms lazily — only materializes on collect
    pub fn collect(&mut self) -> &Vec<Row> {
        if !self.materialized {
            self.materialize();
        }
        &self.data
    }

    fn materialize(&mut self) {
        let transforms = self.transformations.clone();
        for t in &transforms {
            match t {
                Transform::Filter(pred) => {
                    self.data = self.data.iter().filter(|row| {
                        eval_predicate(pred, row, &self.schema)
                    }).cloned().collect();
                }
                Transform::Sort(cols) => {
                    let schema = self.schema.clone();
                    let cols = cols.clone();
                    self.data.sort_by(|a, b| {
                        for (col_name, order) in &cols {
                            if let Some(idx) = schema.fields.iter().position(|f| f.name == *col_name) {
                                let cmp = compare_cells(&a[idx], &b[idx]);
                                let cmp = match order {
                                    Order::Asc => cmp,
                                    Order::Desc => cmp.reverse(),
                                };
                                if cmp != std::cmp::Ordering::Equal { return cmp; }
                            }
                        }
                        std::cmp::Ordering::Equal
                    });
                }
                Transform::Sample(frac) => {
                    let n = (self.data.len() as f64 * frac).ceil() as usize;
                    // Deterministic sampling using stride
                    if n < self.data.len() && n > 0 {
                        let step = self.data.len() / n;
                        self.data = self.data.iter().step_by(step).take(n).cloned().collect();
                    }
                }
                Transform::Select(cols) => {
                    let indices: Vec<usize> = cols.iter()
                        .filter_map(|c| self.schema.fields.iter().position(|f| f.name == *c))
                        .collect();
                    self.data = self.data.iter().map(|row| {
                        indices.iter().map(|&i| row[i].clone()).collect()
                    }).collect();
                    self.schema.fields = indices.iter()
                        .map(|&i| self.schema.fields[i].clone())
                        .collect();
                }
                Transform::Normalize { method } => {
                    normalize_data(&mut self.data, &self.schema, method);
                }
                _ => {} // Other transforms are no-ops for now
            }
        }
        self.transformations.clear();
        self.materialized = true;
        self.version.row_count = self.data.len();
    }

    /// Add a lazy transform
    pub fn add_transform(&mut self, t: Transform) {
        self.materialized = false;
        self.transformations.push(t);
    }
}

fn eval_predicate(pred: &Predicate, row: &Row, schema: &Schema) -> bool {
    match pred {
        Predicate::Gt(col, val) => {
            if let Some(idx) = schema.fields.iter().position(|f| f.name == *col) {
                row.get(idx).and_then(|c| c.as_f64()).map_or(false, |v| v > *val)
            } else { true }
        }
        Predicate::Lt(col, val) => {
            if let Some(idx) = schema.fields.iter().position(|f| f.name == *col) {
                row.get(idx).and_then(|c| c.as_f64()).map_or(false, |v| v < *val)
            } else { true }
        }
        Predicate::Eq(col, val) => {
            if let Some(idx) = schema.fields.iter().position(|f| f.name == *col) {
                row.get(idx).map_or(false, |c| {
                    match c {
                        CellValue::String(s) => s == val,
                        other => format!("{}", other) == *val,
                    }
                })
            } else { true }
        }
        Predicate::NotNull(col) => {
            if let Some(idx) = schema.fields.iter().position(|f| f.name == *col) {
                row.get(idx).map_or(false, |c| *c != CellValue::Null)
            } else { true }
        }
        Predicate::Custom(_expr) => true, // custom predicates not evaluated yet
    }
}

fn compare_cells(a: &CellValue, b: &CellValue) -> std::cmp::Ordering {
    match (a, b) {
        (CellValue::Int(x), CellValue::Int(y)) => x.cmp(y),
        (CellValue::Float(x), CellValue::Float(y)) => x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal),
        (CellValue::String(x), CellValue::String(y)) => x.cmp(y),
        (CellValue::Null, CellValue::Null) => std::cmp::Ordering::Equal,
        (CellValue::Null, _) => std::cmp::Ordering::Less,
        (_, CellValue::Null) => std::cmp::Ordering::Greater,
        _ => std::cmp::Ordering::Equal,
    }
}

fn normalize_data(data: &mut Vec<Row>, schema: &Schema, method: &NormMethod) {
    let ncols = schema.fields.len();
    for col in 0..ncols {
        match &schema.fields[col].dtype {
            DataType::Float | DataType::Int => {}
            _ => continue,
        }
        let vals: Vec<f64> = data.iter().filter_map(|r| r.get(col).and_then(|c| c.as_f64())).collect();
        if vals.is_empty() { continue; }
        match method {
            NormMethod::Standard => {
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64).sqrt();
                if std > 1e-15 {
                    for row in data.iter_mut() {
                        if let Some(v) = row.get(col).and_then(|c| c.as_f64()) {
                            row[col] = CellValue::Float((v - mean) / std);
                        }
                    }
                }
            }
            NormMethod::MinMax => {
                let min = vals.iter().copied().fold(f64::INFINITY, f64::min);
                let max = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let range = max - min;
                if range > 1e-15 {
                    for row in data.iter_mut() {
                        if let Some(v) = row.get(col).and_then(|c| c.as_f64()) {
                            row[col] = CellValue::Float((v - min) / range);
                        }
                    }
                }
            }
            NormMethod::L2 => {
                let norm = vals.iter().map(|v| v * v).sum::<f64>().sqrt();
                if norm > 1e-15 {
                    for row in data.iter_mut() {
                        if let Some(v) = row.get(col).and_then(|c| c.as_f64()) {
                            row[col] = CellValue::Float(v / norm);
                        }
                    }
                }
            }
        }
    }
}

fn hash_rows(data: &[Row]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for row in data {
        for cell in row {
            let s = format!("{}", cell);
            for b in s.bytes() {
                h ^= b as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
        }
    }
    h
}

// ── Data Loader ─────────────────────────────────────────────────────

pub struct DataLoader;

#[derive(Debug, Clone, PartialEq)]
pub enum DataFormat {
    Csv,
    Tsv,
    Json,
    JsonLines,
    Binary,
    VortexNative,
}

impl DataLoader {
    /// Auto-detect format from content/extension
    pub fn detect_format(path: &str, content: &str) -> DataFormat {
        if path.ends_with(".csv") { return DataFormat::Csv; }
        if path.ends_with(".tsv") { return DataFormat::Tsv; }
        if path.ends_with(".json") { return DataFormat::Json; }
        if path.ends_with(".jsonl") { return DataFormat::JsonLines; }
        if path.ends_with(".vxd") { return DataFormat::VortexNative; }
        if path.ends_with(".bin") { return DataFormat::Binary; }

        // Content-based detection
        let trimmed = content.trim();
        if trimmed.starts_with('[') || trimmed.starts_with('{') {
            if trimmed.contains('\n') && !trimmed.starts_with('[') {
                return DataFormat::JsonLines;
            }
            return DataFormat::Json;
        }
        // Check if TSV (tabs more frequent than commas)
        let first_line = trimmed.lines().next().unwrap_or("");
        let tabs = first_line.chars().filter(|&c| c == '\t').count();
        let commas = first_line.chars().filter(|&c| c == ',').count();
        if tabs > commas && tabs > 0 { DataFormat::Tsv } else { DataFormat::Csv }
    }

    pub fn load(path: &str) -> Result<DataStream, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read {}: {}", path, e))?;
        let format = Self::detect_format(path, &content);
        match format {
            DataFormat::Csv => Self::load_csv(&content, ',', path),
            DataFormat::Tsv => Self::load_csv(&content, '\t', path),
            DataFormat::Json => Self::load_json(&content, path),
            DataFormat::JsonLines => Self::load_jsonl(&content, path),
            _ => Err(format!("Unsupported format for {}", path)),
        }
    }

    pub fn load_from_string(content: &str, format: DataFormat, source: &str) -> Result<DataStream, String> {
        match format {
            DataFormat::Csv => Self::load_csv(content, ',', source),
            DataFormat::Tsv => Self::load_csv(content, '\t', source),
            DataFormat::Json => Self::load_json(content, source),
            DataFormat::JsonLines => Self::load_jsonl(content, source),
            _ => Err("Unsupported format".to_string()),
        }
    }

    fn load_csv(content: &str, sep: char, source: &str) -> Result<DataStream, String> {
        let mut lines = content.lines().filter(|l| !l.trim().is_empty());
        let header = lines.next().ok_or("Empty CSV")?;
        let col_names: Vec<String> = header.split(sep).map(|s| s.trim().trim_matches('"').to_string()).collect();

        let mut rows: Vec<Vec<String>> = Vec::new();
        for line in lines {
            let cells: Vec<String> = split_csv_line(line, sep);
            rows.push(cells);
        }

        // Type inference: scan all rows
        let ncols = col_names.len();
        let mut col_types = vec![InferredType::Unknown; ncols];

        for row in &rows {
            for (i, cell) in row.iter().enumerate() {
                if i >= ncols { break; }
                let cell = cell.trim();
                if cell.is_empty() || cell == "null" || cell == "NA" || cell == "None" {
                    continue; // nullable
                }
                let inferred = infer_cell_type(cell);
                col_types[i] = merge_types(col_types[i].clone(), inferred);
            }
        }

        let fields: Vec<Field> = col_names.iter().enumerate().map(|(i, name)| {
            let (dtype, semantic) = resolved_type(&col_types[i]);
            let nullable = rows.iter().any(|r| {
                r.get(i).map_or(true, |c| {
                    let c = c.trim();
                    c.is_empty() || c == "null" || c == "NA" || c == "None"
                })
            });
            Field { name: name.clone(), dtype, nullable, semantic_type: semantic }
        }).collect();

        let schema = Schema { fields: fields.clone(), constraints: Vec::new() };

        let data: Vec<Row> = rows.iter().map(|row| {
            row.iter().enumerate().map(|(i, cell)| {
                let cell = cell.trim();
                if cell.is_empty() || cell == "null" || cell == "NA" || cell == "None" {
                    return CellValue::Null;
                }
                if i < fields.len() {
                    parse_cell(cell, &fields[i].dtype)
                } else {
                    CellValue::String(cell.to_string())
                }
            }).collect()
        }).collect();

        let mut stream = DataStream::new(schema, data, source);
        let lineage = Lineage::root(&stream.lineage_id, source);
        LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
        Ok(stream)
    }

    fn load_json(content: &str, source: &str) -> Result<DataStream, String> {
        // Simple JSON array parser: expects [{"key": val, ...}, ...]
        let trimmed = content.trim();
        if !trimmed.starts_with('[') {
            // Single object — wrap in array
            return Self::load_json(&format!("[{}]", trimmed), source);
        }
        let objects = parse_json_array(trimmed)?;
        if objects.is_empty() {
            return Ok(DataStream::new(Schema::new(), Vec::new(), source));
        }

        // Collect all field names from all objects
        let mut all_keys: Vec<String> = Vec::new();
        for obj in &objects {
            for (k, _) in obj {
                if !all_keys.contains(k) {
                    all_keys.push(k.clone());
                }
            }
        }

        // Infer types
        let mut col_types = vec![InferredType::Unknown; all_keys.len()];
        for obj in &objects {
            for (i, key) in all_keys.iter().enumerate() {
                if let Some(val) = obj.iter().find(|(k, _)| k == key).map(|(_, v)| v) {
                    if val != "null" {
                        col_types[i] = merge_types(col_types[i].clone(), infer_cell_type(val));
                    }
                }
            }
        }

        let fields: Vec<Field> = all_keys.iter().enumerate().map(|(i, name)| {
            let (dtype, semantic) = resolved_type(&col_types[i]);
            Field { name: name.clone(), dtype, nullable: true, semantic_type: semantic }
        }).collect();
        let schema = Schema { fields: fields.clone(), constraints: Vec::new() };

        let data: Vec<Row> = objects.iter().map(|obj| {
            all_keys.iter().enumerate().map(|(i, key)| {
                if let Some(val) = obj.iter().find(|(k, _)| k == key).map(|(_, v)| v) {
                    if val == "null" { CellValue::Null }
                    else { parse_cell(val, &fields[i].dtype) }
                } else {
                    CellValue::Null
                }
            }).collect()
        }).collect();

        let mut stream = DataStream::new(schema, data, source);
        let lineage = Lineage::root(&stream.lineage_id, source);
        LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
        Ok(stream)
    }

    fn load_jsonl(content: &str, source: &str) -> Result<DataStream, String> {
        let lines: Vec<&str> = content.lines().filter(|l| !l.trim().is_empty()).collect();
        let json_array = format!("[{}]", lines.join(","));
        Self::load_json(&json_array, source)
    }

    pub fn save_csv(stream: &DataStream, _path: &str) -> Result<String, String> {
        let mut out = String::new();
        // Header
        let names: Vec<&str> = stream.schema.fields.iter().map(|f| f.name.as_str()).collect();
        out.push_str(&names.join(","));
        out.push('\n');
        for row in &stream.data {
            let cells: Vec<String> = row.iter().map(|c| format!("{}", c)).collect();
            out.push_str(&cells.join(","));
            out.push('\n');
        }
        Ok(out)
    }
}

fn split_csv_line(line: &str, sep: char) -> Vec<String> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for ch in line.chars() {
        if ch == '"' {
            in_quotes = !in_quotes;
        } else if ch == sep && !in_quotes {
            result.push(current.clone());
            current.clear();
        } else {
            current.push(ch);
        }
    }
    result.push(current);
    result
}

// ── Simple JSON Parser ──────────────────────────────────────────────

type JsonObject = Vec<(String, String)>;

fn parse_json_array(input: &str) -> Result<Vec<JsonObject>, String> {
    let trimmed = input.trim();
    if !trimmed.starts_with('[') || !trimmed.ends_with(']') {
        return Err("Expected JSON array".to_string());
    }
    let inner = &trimmed[1..trimmed.len()-1].trim();
    if inner.is_empty() { return Ok(Vec::new()); }

    let mut objects = Vec::new();
    let mut depth = 0;
    let mut start = 0;
    let mut in_string = false;
    let mut escape = false;
    let bytes = inner.as_bytes();

    for i in 0..bytes.len() {
        if escape { escape = false; continue; }
        if bytes[i] == b'\\' { escape = true; continue; }
        if bytes[i] == b'"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => {
                depth -= 1;
                if depth == 0 {
                    let chunk = inner[start..=i].trim();
                    if chunk.starts_with('{') {
                        objects.push(parse_json_object(chunk)?);
                    }
                    start = i + 1;
                }
            }
            b',' if depth == 0 => {
                start = i + 1;
            }
            _ => {}
        }
    }

    Ok(objects)
}

fn parse_json_object(input: &str) -> Result<JsonObject, String> {
    let trimmed = input.trim();
    if !trimmed.starts_with('{') || !trimmed.ends_with('}') {
        return Err("Expected JSON object".to_string());
    }
    let inner = &trimmed[1..trimmed.len()-1];
    let mut result = Vec::new();

    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;
    let mut start = 0;
    let bytes = inner.as_bytes();

    let mut pairs: Vec<&str> = Vec::new();
    for i in 0..bytes.len() {
        if escape { escape = false; continue; }
        if bytes[i] == b'\\' { escape = true; continue; }
        if bytes[i] == b'"' { in_string = !in_string; continue; }
        if in_string { continue; }
        match bytes[i] {
            b'{' | b'[' => depth += 1,
            b'}' | b']' => depth -= 1,
            b',' if depth == 0 => {
                pairs.push(&inner[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    pairs.push(&inner[start..]);

    for pair in pairs {
        let pair = pair.trim();
        if pair.is_empty() { continue; }
        // Find the colon separating key and value
        if let Some(colon_pos) = find_json_colon(pair) {
            let key = pair[..colon_pos].trim().trim_matches('"');
            let val = pair[colon_pos+1..].trim();
            // Strip quotes from string values
            let val = if val.starts_with('"') && val.ends_with('"') {
                &val[1..val.len()-1]
            } else {
                val
            };
            result.push((key.to_string(), val.to_string()));
        }
    }

    Ok(result)
}

fn find_json_colon(s: &str) -> Option<usize> {
    let mut in_string = false;
    let mut escape = false;
    for (i, ch) in s.chars().enumerate() {
        if escape { escape = false; continue; }
        if ch == '\\' { escape = true; continue; }
        if ch == '"' { in_string = !in_string; continue; }
        if !in_string && ch == ':' { return Some(i); }
    }
    None
}

// ── Type Inference ──────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
enum InferredType {
    Unknown,
    Int,
    Float,
    Bool,
    DateTime,
    String,
}

fn infer_cell_type(cell: &str) -> InferredType {
    let cell = cell.trim().trim_matches('"');
    if cell.is_empty() { return InferredType::Unknown; }
    if cell == "true" || cell == "false" { return InferredType::Bool; }
    if cell.parse::<i64>().is_ok() { return InferredType::Int; }
    if cell.parse::<f64>().is_ok() { return InferredType::Float; }
    // Simple datetime detection
    if cell.len() >= 10 && cell.chars().nth(4) == Some('-') && cell.chars().nth(7) == Some('-') {
        return InferredType::DateTime;
    }
    InferredType::String
}

fn merge_types(a: InferredType, b: InferredType) -> InferredType {
    if a == b { return a; }
    match (&a, &b) {
        (InferredType::Unknown, other) | (other, InferredType::Unknown) => other.clone(),
        (InferredType::Int, InferredType::Float) | (InferredType::Float, InferredType::Int) => InferredType::Float,
        _ => InferredType::String,
    }
}

fn resolved_type(inferred: &InferredType) -> (DataType, SemanticType) {
    match inferred {
        InferredType::Int => (DataType::Int, SemanticType::Raw),
        InferredType::Float => (DataType::Float, SemanticType::Raw),
        InferredType::Bool => (DataType::Bool, SemanticType::Raw),
        InferredType::DateTime => (DataType::DateTime, SemanticType::Timestamp),
        InferredType::String | InferredType::Unknown => (DataType::String, SemanticType::Raw),
    }
}

fn parse_cell(cell: &str, dtype: &DataType) -> CellValue {
    let cell = cell.trim().trim_matches('"');
    match dtype {
        DataType::Int => cell.parse::<i64>().map(CellValue::Int).unwrap_or(CellValue::String(cell.to_string())),
        DataType::Float => cell.parse::<f64>().map(CellValue::Float).unwrap_or(CellValue::String(cell.to_string())),
        DataType::Bool => cell.parse::<bool>().map(CellValue::Bool).unwrap_or(CellValue::String(cell.to_string())),
        _ => CellValue::String(cell.to_string()),
    }
}

// ── Statistics ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ColumnStats {
    pub name: String,
    pub dtype: DataType,
    pub count: usize,
    pub null_count: usize,
    pub unique_count: usize,
    pub mean: Option<f64>,
    pub std_dev: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub q25: Option<f64>,
    pub q50: Option<f64>,
    pub q75: Option<f64>,
}

impl std::fmt::Display for ColumnStats {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} ({}): count={}, nulls={}, unique={}",
            self.name, self.dtype, self.count, self.null_count, self.unique_count)?;
        if let Some(m) = self.mean { write!(f, ", mean={:.4}", m)?; }
        if let Some(s) = self.std_dev { write!(f, ", std={:.4}", s)?; }
        if let Some(mn) = self.min { write!(f, ", min={:.4}", mn)?; }
        if let Some(mx) = self.max { write!(f, ", max={:.4}", mx)?; }
        Ok(())
    }
}

pub fn compute_statistics(stream: &DataStream) -> Vec<ColumnStats> {
    let mut stats = Vec::new();
    for (i, field) in stream.schema.fields.iter().enumerate() {
        let mut count = 0usize;
        let mut null_count = 0usize;
        let mut unique_vals = std::collections::HashSet::new();
        let mut numeric_vals = Vec::new();

        for row in &stream.data {
            if let Some(cell) = row.get(i) {
                match cell {
                    CellValue::Null => null_count += 1,
                    _ => {
                        count += 1;
                        unique_vals.insert(format!("{}", cell));
                        if let Some(v) = cell.as_f64() {
                            numeric_vals.push(v);
                        }
                    }
                }
            }
        }

        let (mean, std_dev, min, max, q25, q50, q75) = if !numeric_vals.is_empty() {
            numeric_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let n = numeric_vals.len() as f64;
            let mean = numeric_vals.iter().sum::<f64>() / n;
            let variance = numeric_vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n;
            let std = variance.sqrt();
            let min = numeric_vals.first().copied();
            let max = numeric_vals.last().copied();
            let q25 = percentile(&numeric_vals, 0.25);
            let q50 = percentile(&numeric_vals, 0.50);
            let q75 = percentile(&numeric_vals, 0.75);
            (Some(mean), Some(std), min, max, Some(q25), Some(q50), Some(q75))
        } else {
            (None, None, None, None, None, None, None)
        };

        stats.push(ColumnStats {
            name: field.name.clone(),
            dtype: field.dtype.clone(),
            count,
            null_count,
            unique_count: unique_vals.len(),
            mean, std_dev, min, max, q25, q50, q75,
        });
    }
    stats
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() { return 0.0; }
    let idx = p * (sorted.len() - 1) as f64;
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    if lo == hi { sorted[lo] }
    else {
        let frac = idx - lo as f64;
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// ── ML Operations ───────────────────────────────────────────────────

/// Simple xorshift PRNG
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self { Self(if seed == 0 { 0xdeadbeef } else { seed }) }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 { (self.next_u64() as f64) / (u64::MAX as f64) }
    fn shuffle<T>(&mut self, slice: &mut [T]) {
        for i in (1..slice.len()).rev() {
            let j = (self.next_u64() as usize) % (i + 1);
            slice.swap(i, j);
        }
    }
}

pub fn train_test_split(stream: &DataStream, train: f64, val: f64, test: f64, seed: u64)
    -> Result<(DataStream, DataStream, DataStream), String>
{
    if (train + val + test - 1.0).abs() > 0.01 {
        return Err("Split ratios must sum to 1.0".to_string());
    }
    let n = stream.data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed);
    rng.shuffle(&mut indices);

    let train_end = (n as f64 * train).round() as usize;
    let val_end = train_end + (n as f64 * val).round() as usize;

    let make_split = |idxs: &[usize], desc: &str| -> DataStream {
        let data: Vec<Row> = idxs.iter().map(|&i| stream.data[i].clone()).collect();
        let mut s = DataStream::new(stream.schema.clone(), data, &stream.source);
        s.version.description = desc.to_string();
        let lineage = Lineage::derived(
            &s.lineage_id,
            vec![stream.lineage_id.clone()],
            &format!("split({})", desc),
            HashMap::new(),
        );
        LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
        s
    };

    let train_stream = make_split(&indices[..train_end], "train");
    let val_stream = make_split(&indices[train_end..val_end], "val");
    let test_stream = make_split(&indices[val_end..], "test");

    Ok((train_stream, val_stream, test_stream))
}

pub fn cross_validate_folds(stream: &DataStream, k: usize, seed: u64)
    -> Result<Vec<(DataStream, DataStream)>, String>
{
    if k < 2 { return Err("k must be >= 2".to_string()); }
    let n = stream.data.len();
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = Rng::new(seed);
    rng.shuffle(&mut indices);

    let fold_size = n / k;
    let mut folds = Vec::new();

    for fold in 0..k {
        let test_start = fold * fold_size;
        let test_end = if fold == k - 1 { n } else { test_start + fold_size };
        let test_idxs: Vec<usize> = indices[test_start..test_end].to_vec();
        let train_idxs: Vec<usize> = indices[..test_start].iter()
            .chain(indices[test_end..].iter())
            .copied().collect();

        let train_data: Vec<Row> = train_idxs.iter().map(|&i| stream.data[i].clone()).collect();
        let test_data: Vec<Row> = test_idxs.iter().map(|&i| stream.data[i].clone()).collect();

        let train_s = DataStream::new(stream.schema.clone(), train_data, &stream.source);
        let test_s = DataStream::new(stream.schema.clone(), test_data, &stream.source);
        folds.push((train_s, test_s));
    }

    Ok(folds)
}

pub fn detect_anomalies(stream: &DataStream, threshold: f64) -> Vec<usize> {
    // Z-score based anomaly detection on numeric columns
    let stats = compute_statistics(stream);
    let mut anomaly_rows = std::collections::HashSet::new();

    for (i, stat) in stats.iter().enumerate() {
        if let (Some(mean), Some(std)) = (stat.mean, stat.std_dev) {
            if std < 1e-15 { continue; }
            for (row_idx, row) in stream.data.iter().enumerate() {
                if let Some(v) = row.get(i).and_then(|c| c.as_f64()) {
                    let z = ((v - mean) / std).abs();
                    if z > threshold {
                        anomaly_rows.insert(row_idx);
                    }
                }
            }
        }
    }

    let mut result: Vec<usize> = anomaly_rows.into_iter().collect();
    result.sort();
    result
}

// ── Interpreter Builtins ────────────────────────────────────────────

fn store_stream(stream: DataStream) -> usize {
    let id = stream.id;
    STREAM_STORE.lock().unwrap().insert(id, stream);
    id
}

fn get_stream(id: usize) -> Result<DataStream, String> {
    STREAM_STORE.lock().unwrap().get(&id).cloned()
        .ok_or_else(|| format!("Unknown data stream: {}", id))
}

fn update_stream(stream: DataStream) {
    let id = stream.id;
    STREAM_STORE.lock().unwrap().insert(id, stream);
}

fn val_to_usize(v: &Value) -> Result<usize, String> {
    match v {
        Value::Int(n) => Ok(*n as usize),
        _ => Err("Expected integer (stream id)".to_string()),
    }
}

fn val_to_f64(v: &Value) -> Result<f64, String> {
    match v {
        Value::Float(f) => Ok(*f),
        Value::Int(n) => Ok(*n as f64),
        _ => Err("Expected number".to_string()),
    }
}

fn builtin_data_load(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_load(path)".into()); }
    let path = match &args[0] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
    let stream = DataLoader::load(&path)?;
    let id = store_stream(stream);
    Ok(Value::Int(id as i128))
}

fn builtin_data_load_csv(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_load_csv(csv_string)".into()); }
    let content = match &args[0] { Value::String(s) => s.clone(), _ => return Err("arg must be string".into()) };
    let stream = DataLoader::load_from_string(&content, DataFormat::Csv, "inline_csv")?;
    let id = store_stream(stream);
    Ok(Value::Int(id as i128))
}

fn builtin_data_schema(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_schema(stream_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let stream = get_stream(id)?;
    Ok(Value::String(format!("{}", stream.schema)))
}

fn builtin_data_filter(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("data_filter(stream_id, condition_string)".into()); }
    let id = val_to_usize(&args[0])?;
    let cond = match &args[1] { Value::String(s) => s.clone(), _ => return Err("condition must be string".into()) };

    let mut stream = get_stream(id)?;

    // Parse simple conditions: "col > val", "col < val", "col == val"
    let pred = parse_predicate_string(&cond)?;
    stream.add_transform(Transform::Filter(pred));
    stream.collect();

    // Create new stream with lineage
    let mut new_stream = DataStream::new(stream.schema.clone(), stream.data.clone(), &stream.source);
    let lineage = Lineage::derived(
        &new_stream.lineage_id,
        vec![format!("stream_{}", id)],
        &format!("filter({})", cond),
        HashMap::new(),
    );
    LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
    let new_id = store_stream(new_stream);
    Ok(Value::Int(new_id as i128))
}

fn parse_predicate_string(s: &str) -> Result<Predicate, String> {
    let s = s.trim();
    if let Some(pos) = s.find(">=") {
        let col = s[..pos].trim().to_string();
        let val: f64 = s[pos+2..].trim().parse().map_err(|_| "Invalid number in predicate")?;
        // >= is approximated as > (val - epsilon)
        Ok(Predicate::Gt(col, val - 0.0001))
    } else if let Some(pos) = s.find('>') {
        let col = s[..pos].trim().to_string();
        let val: f64 = s[pos+1..].trim().parse().map_err(|_| "Invalid number in predicate")?;
        Ok(Predicate::Gt(col, val))
    } else if let Some(pos) = s.find("<=") {
        let col = s[..pos].trim().to_string();
        let val: f64 = s[pos+2..].trim().parse().map_err(|_| "Invalid number in predicate")?;
        Ok(Predicate::Lt(col, val + 0.0001))
    } else if let Some(pos) = s.find('<') {
        let col = s[..pos].trim().to_string();
        let val: f64 = s[pos+1..].trim().parse().map_err(|_| "Invalid number in predicate")?;
        Ok(Predicate::Lt(col, val))
    } else if let Some(pos) = s.find("==") {
        let col = s[..pos].trim().to_string();
        let val = s[pos+2..].trim().trim_matches('"').to_string();
        Ok(Predicate::Eq(col, val))
    } else if s.ends_with("!= null") || s.ends_with("is not null") {
        let col = s.split_whitespace().next().unwrap_or("").to_string();
        Ok(Predicate::NotNull(col))
    } else {
        Ok(Predicate::Custom(s.to_string()))
    }
}

fn builtin_data_map(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("data_map(stream_id, transform_expr)".into()); }
    let id = val_to_usize(&args[0])?;
    let expr = match &args[1] { Value::String(s) => s.clone(), _ => return Err("transform must be string".into()) };
    let stream = get_stream(id)?;
    let mut new_stream = DataStream::new(stream.schema.clone(), stream.data.clone(), &stream.source);
    new_stream.add_transform(Transform::Map(expr.clone()));
    let lineage = Lineage::derived(
        &new_stream.lineage_id,
        vec![format!("stream_{}", id)],
        &format!("map({})", expr),
        HashMap::new(),
    );
    LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
    let new_id = store_stream(new_stream);
    Ok(Value::Int(new_id as i128))
}

fn builtin_data_split(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 4 { return Err("data_split(stream_id, train, val, test)".into()); }
    let id = val_to_usize(&args[0])?;
    let train = val_to_f64(&args[1])?;
    let val = val_to_f64(&args[2])?;
    let test = val_to_f64(&args[3])?;
    let stream = get_stream(id)?;
    let (train_s, val_s, test_s) = train_test_split(&stream, train, val, test, 42)?;
    let ids = vec![
        Value::Int(store_stream(train_s) as i128),
        Value::Int(store_stream(val_s) as i128),
        Value::Int(store_stream(test_s) as i128),
    ];
    Ok(Value::Array(ids))
}

fn builtin_data_normalize(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("data_normalize(stream_id, method)".into()); }
    let id = val_to_usize(&args[0])?;
    let method_str = match &args[1] { Value::String(s) => s.clone(), _ => return Err("method must be string".into()) };
    let method = match method_str.to_lowercase().as_str() {
        "standard" | "zscore" => NormMethod::Standard,
        "minmax" | "min_max" => NormMethod::MinMax,
        "l2" => NormMethod::L2,
        _ => return Err(format!("Unknown normalization method: {}", method_str)),
    };
    let mut stream = get_stream(id)?;
    stream.add_transform(Transform::Normalize { method });
    stream.collect();
    let mut new_stream = DataStream::new(stream.schema.clone(), stream.data.clone(), &stream.source);
    let lineage = Lineage::derived(
        &new_stream.lineage_id,
        vec![format!("stream_{}", id)],
        &format!("normalize({})", method_str),
        HashMap::new(),
    );
    LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
    let new_id = store_stream(new_stream);
    Ok(Value::Int(new_id as i128))
}

fn builtin_data_stats(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_stats(stream_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let stream = get_stream(id)?;
    let stats = compute_statistics(&stream);
    let desc: Vec<String> = stats.iter().map(|s| format!("{}", s)).collect();
    Ok(Value::String(desc.join("\n")))
}

fn builtin_data_lineage(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_lineage(stream_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let stream = get_stream(id)?;
    let graph = LINEAGE_GRAPH.lock().unwrap();
    let trace = graph.trace_back(&stream.lineage_id);
    let desc: Vec<String> = trace.iter().map(|l| format!("{}", l)).collect();
    Ok(Value::String(desc.join("\n")))
}

fn builtin_data_version(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_version(stream_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let stream = get_stream(id)?;
    Ok(Value::String(format!("{}", stream.version)))
}

fn builtin_data_save(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() < 2 { return Err("data_save(stream_id, path, [format])".into()); }
    let id = val_to_usize(&args[0])?;
    let path = match &args[1] { Value::String(s) => s.clone(), _ => return Err("path must be string".into()) };
    let stream = get_stream(id)?;
    let csv = DataLoader::save_csv(&stream, &path)?;
    std::fs::write(&path, &csv).map_err(|e| format!("Failed to write: {}", e))?;
    Ok(Value::String(format!("Saved {} rows to {}", stream.data.len(), path)))
}

fn builtin_data_collect(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("data_collect(stream_id)".into()); }
    let id = val_to_usize(&args[0])?;
    let mut stream = get_stream(id)?;
    stream.collect();
    update_stream(stream.clone());
    // Convert to nested array of values
    let rows: Vec<Value> = stream.data.iter().map(|row| {
        Value::Array(row.iter().map(|cell| match cell {
            CellValue::Int(n) => Value::Int(*n as i128),
            CellValue::Float(f) => Value::Float(*f),
            CellValue::Bool(b) => Value::Bool(*b),
            CellValue::String(s) => Value::String(s.clone()),
            CellValue::Null => Value::String("null".to_string()),
        }).collect())
    }).collect();
    Ok(Value::Array(rows))
}

fn builtin_data_sample(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("data_sample(stream_id, n)".into()); }
    let id = val_to_usize(&args[0])?;
    let n = val_to_usize(&args[1])?;
    let stream = get_stream(id)?;
    let frac = n as f64 / stream.data.len().max(1) as f64;
    let mut new_stream = DataStream::new(stream.schema.clone(), stream.data.clone(), &stream.source);
    new_stream.add_transform(Transform::Sample(frac.min(1.0)));
    new_stream.collect();
    let new_id = store_stream(new_stream);
    Ok(Value::Int(new_id as i128))
}

fn builtin_data_join(_env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("data_join(stream_a, stream_b, on_col)".into()); }
    let id_a = val_to_usize(&args[0])?;
    let id_b = val_to_usize(&args[1])?;
    let on_col = match &args[2] { Value::String(s) => s.clone(), _ => return Err("on_col must be string".into()) };

    let a = get_stream(id_a)?;
    let b = get_stream(id_b)?;

    let col_a = a.col_index(&on_col).ok_or(format!("Column {} not found in stream A", on_col))?;
    let col_b = b.col_index(&on_col).ok_or(format!("Column {} not found in stream B", on_col))?;

    // Build index on B
    let mut b_index: HashMap<String, Vec<usize>> = HashMap::new();
    for (i, row) in b.data.iter().enumerate() {
        let key = format!("{}", row[col_b]);
        b_index.entry(key).or_default().push(i);
    }

    // Inner join
    let mut joined_fields = a.schema.fields.clone();
    for (i, f) in b.schema.fields.iter().enumerate() {
        if i != col_b {
            joined_fields.push(Field {
                name: format!("{}_{}", f.name, "b"),
                ..f.clone()
            });
        }
    }
    let joined_schema = Schema { fields: joined_fields, constraints: Vec::new() };

    let mut joined_data = Vec::new();
    for row_a in &a.data {
        let key = format!("{}", row_a[col_a]);
        if let Some(b_rows) = b_index.get(&key) {
            for &bi in b_rows {
                let mut new_row = row_a.clone();
                for (i, cell) in b.data[bi].iter().enumerate() {
                    if i != col_b {
                        new_row.push(cell.clone());
                    }
                }
                joined_data.push(new_row);
            }
        }
    }

    let mut new_stream = DataStream::new(joined_schema, joined_data, "join");
    let lineage = Lineage::derived(
        &new_stream.lineage_id,
        vec![format!("stream_{}", id_a), format!("stream_{}", id_b)],
        &format!("join(on={})", on_col),
        HashMap::new(),
    );
    LINEAGE_GRAPH.lock().unwrap().add_node(lineage);
    let new_id = store_stream(new_stream);
    Ok(Value::Int(new_id as i128))
}

pub fn register_builtins(env: &mut Env) {
    env.functions.insert("data_load".into(), FnDef::Builtin(builtin_data_load));
    env.functions.insert("data_load_csv".into(), FnDef::Builtin(builtin_data_load_csv));
    env.functions.insert("data_schema".into(), FnDef::Builtin(builtin_data_schema));
    env.functions.insert("data_filter".into(), FnDef::Builtin(builtin_data_filter));
    env.functions.insert("data_map".into(), FnDef::Builtin(builtin_data_map));
    env.functions.insert("data_split".into(), FnDef::Builtin(builtin_data_split));
    env.functions.insert("data_normalize".into(), FnDef::Builtin(builtin_data_normalize));
    env.functions.insert("data_stats".into(), FnDef::Builtin(builtin_data_stats));
    env.functions.insert("data_lineage".into(), FnDef::Builtin(builtin_data_lineage));
    env.functions.insert("data_version".into(), FnDef::Builtin(builtin_data_version));
    env.functions.insert("data_save".into(), FnDef::Builtin(builtin_data_save));
    env.functions.insert("data_collect".into(), FnDef::Builtin(builtin_data_collect));
    env.functions.insert("data_sample".into(), FnDef::Builtin(builtin_data_sample));
    env.functions.insert("data_join".into(), FnDef::Builtin(builtin_data_join));
}

// ── Tests ───────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_csv() -> &'static str {
        "name,age,score,active\nAlice,30,95.5,true\nBob,25,87.3,false\nCharlie,35,92.1,true\nDiana,28,78.9,true\nEve,22,88.0,false"
    }

    fn load_sample() -> DataStream {
        DataLoader::load_from_string(sample_csv(), DataFormat::Csv, "test.csv").unwrap()
    }

    #[test]
    fn test_csv_loading() {
        let stream = load_sample();
        assert_eq!(stream.row_count(), 5);
        assert_eq!(stream.schema.fields.len(), 4);
        assert_eq!(stream.schema.fields[0].name, "name");
        assert_eq!(stream.schema.fields[1].name, "age");
    }

    #[test]
    fn test_type_inference_csv() {
        let stream = load_sample();
        assert_eq!(stream.schema.fields[0].dtype, DataType::String); // name
        assert_eq!(stream.schema.fields[1].dtype, DataType::Int);    // age
        assert_eq!(stream.schema.fields[2].dtype, DataType::Float);  // score
        assert_eq!(stream.schema.fields[3].dtype, DataType::Bool);   // active
    }

    #[test]
    fn test_json_loading() {
        let json = r#"[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]"#;
        let stream = DataLoader::load_from_string(json, DataFormat::Json, "test.json").unwrap();
        assert_eq!(stream.row_count(), 2);
        assert_eq!(stream.schema.fields.len(), 2);
    }

    #[test]
    fn test_jsonl_loading() {
        let jsonl = "{\"x\": 1, \"y\": 2}\n{\"x\": 3, \"y\": 4}";
        let stream = DataLoader::load_from_string(jsonl, DataFormat::JsonLines, "test.jsonl").unwrap();
        assert_eq!(stream.row_count(), 2);
    }

    #[test]
    fn test_schema_display() {
        let stream = load_sample();
        let schema_str = format!("{}", stream.schema);
        assert!(schema_str.contains("name"));
        assert!(schema_str.contains("age"));
        assert!(schema_str.contains("4 fields"));
    }

    #[test]
    fn test_filter_gt() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Filter(Predicate::Gt("age".to_string(), 27.0)));
        stream.collect();
        // Alice(30), Charlie(35), Diana(28) pass age > 27
        assert_eq!(stream.data.len(), 3);
    }

    #[test]
    fn test_filter_lt() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Filter(Predicate::Lt("score".to_string(), 90.0)));
        stream.collect();
        // Bob(87.3), Diana(78.9), Eve(88.0) pass score < 90
        assert_eq!(stream.data.len(), 3);
    }

    #[test]
    fn test_filter_eq() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Filter(Predicate::Eq("name".to_string(), "Alice".to_string())));
        stream.collect();
        assert_eq!(stream.data.len(), 1);
    }

    #[test]
    fn test_sort() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Sort(vec![("age".to_string(), Order::Asc)]));
        stream.collect();
        // Eve(22), Bob(25), Diana(28), Alice(30), Charlie(35)
        assert_eq!(stream.data[0][1], CellValue::Int(22));
        assert_eq!(stream.data[4][1], CellValue::Int(35));
    }

    #[test]
    fn test_normalize_standard() {
        let csv = "x,y\n1,10\n2,20\n3,30\n4,40\n5,50";
        let mut stream = DataLoader::load_from_string(csv, DataFormat::Csv, "norm.csv").unwrap();
        stream.add_transform(Transform::Normalize { method: NormMethod::Standard });
        stream.collect();
        // After standard normalization, mean should be ~0
        let vals: Vec<f64> = stream.data.iter().filter_map(|r| r[0].as_f64()).collect();
        let mean: f64 = vals.iter().sum::<f64>() / vals.len() as f64;
        assert!(mean.abs() < 1e-10);
    }

    #[test]
    fn test_normalize_minmax() {
        let csv = "x\n10\n20\n30\n40\n50";
        let mut stream = DataLoader::load_from_string(csv, DataFormat::Csv, "mm.csv").unwrap();
        stream.add_transform(Transform::Normalize { method: NormMethod::MinMax });
        stream.collect();
        let vals: Vec<f64> = stream.data.iter().filter_map(|r| r[0].as_f64()).collect();
        assert!((vals[0] - 0.0).abs() < 1e-10); // min -> 0
        assert!((vals[4] - 1.0).abs() < 1e-10); // max -> 1
    }

    #[test]
    fn test_train_test_split() {
        let stream = load_sample();
        let (train, val, test) = train_test_split(&stream, 0.6, 0.2, 0.2, 42).unwrap();
        assert_eq!(train.row_count() + val.row_count() + test.row_count(), 5);
        assert!(train.row_count() >= 2);
    }

    #[test]
    fn test_cross_validate() {
        let csv = "x\n1\n2\n3\n4\n5\n6\n7\n8\n9\n10";
        let stream = DataLoader::load_from_string(csv, DataFormat::Csv, "cv.csv").unwrap();
        let folds = cross_validate_folds(&stream, 5, 42).unwrap();
        assert_eq!(folds.len(), 5);
        for (train, test) in &folds {
            assert_eq!(train.row_count() + test.row_count(), 10);
        }
    }

    #[test]
    fn test_lineage_tracking() {
        let stream = load_sample();
        let parent_id = stream.lineage_id.clone();
        let (train, _, _) = train_test_split(&stream, 0.6, 0.2, 0.2, 42).unwrap();
        let graph = LINEAGE_GRAPH.lock().unwrap();
        let trace = graph.trace_back(&train.lineage_id);
        // Should trace back to the split and the original load
        assert!(trace.len() >= 1);
    }

    #[test]
    fn test_statistics() {
        let stream = load_sample();
        let stats = compute_statistics(&stream);
        assert_eq!(stats.len(), 4);
        // age column stats
        let age_stats = &stats[1];
        assert_eq!(age_stats.name, "age");
        assert_eq!(age_stats.count, 5);
        assert_eq!(age_stats.null_count, 0);
        assert!((age_stats.mean.unwrap() - 28.0).abs() < 0.01); // (30+25+35+28+22)/5 = 28
    }

    #[test]
    fn test_format_detection_csv() {
        assert_eq!(DataLoader::detect_format("data.csv", ""), DataFormat::Csv);
        assert_eq!(DataLoader::detect_format("data.tsv", ""), DataFormat::Tsv);
        assert_eq!(DataLoader::detect_format("data.json", ""), DataFormat::Json);
    }

    #[test]
    fn test_format_detection_content() {
        assert_eq!(DataLoader::detect_format("data", "[{\"a\":1}]"), DataFormat::Json);
        assert_eq!(DataLoader::detect_format("data", "a\tb\n1\t2"), DataFormat::Tsv);
        assert_eq!(DataLoader::detect_format("data", "a,b\n1,2"), DataFormat::Csv);
    }

    #[test]
    fn test_lazy_evaluation() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Filter(Predicate::Gt("age".to_string(), 27.0)));
        // Not materialized yet
        assert!(!stream.materialized);
        assert_eq!(stream.data.len(), 5); // still has all rows
        stream.collect();
        assert!(stream.materialized);
        assert_eq!(stream.data.len(), 3); // now filtered
    }

    #[test]
    fn test_versioning() {
        let stream = load_sample();
        assert_eq!(stream.version.version, 0);
        assert_eq!(stream.version.row_count, 5);
        assert!(!stream.version.schema_hash.is_empty());
    }

    #[test]
    fn test_anomaly_detection() {
        let csv = "val\n1\n2\n3\n2\n1\n3\n2\n100\n1\n2";
        let stream = DataLoader::load_from_string(csv, DataFormat::Csv, "anom.csv").unwrap();
        let anomalies = detect_anomalies(&stream, 2.0);
        // 100 should be anomalous
        assert!(anomalies.contains(&7));
    }

    #[test]
    fn test_sample() {
        let mut stream = load_sample();
        stream.add_transform(Transform::Sample(0.4));
        stream.collect();
        assert!(stream.data.len() <= 3);
        assert!(stream.data.len() >= 1);
    }

    #[test]
    fn test_nullable_detection() {
        let csv = "a,b\n1,hello\n,world\n3,";
        let stream = DataLoader::load_from_string(csv, DataFormat::Csv, "null.csv").unwrap();
        assert!(stream.schema.fields[0].nullable); // a has empty value
        assert!(stream.schema.fields[1].nullable); // b has empty value
    }

    #[test]
    fn test_schema_hash() {
        let s1 = load_sample();
        let s2 = load_sample();
        assert_eq!(s1.schema.hash_schema(), s2.schema.hash_schema());
    }

    #[test]
    fn test_join() {
        let csv_a = "id,name\n1,Alice\n2,Bob\n3,Charlie";
        let csv_b = "id,score\n1,95\n2,87\n4,72";
        let a = DataLoader::load_from_string(csv_a, DataFormat::Csv, "a.csv").unwrap();
        let b = DataLoader::load_from_string(csv_b, DataFormat::Csv, "b.csv").unwrap();
        let id_a = store_stream(a);
        let id_b = store_stream(b);

        let a = get_stream(id_a).unwrap();
        let b = get_stream(id_b).unwrap();

        let col_a = a.col_index("id").unwrap();
        let col_b = b.col_index("id").unwrap();

        // Manual inner join
        let mut b_index: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, row) in b.data.iter().enumerate() {
            b_index.entry(format!("{}", row[col_b])).or_default().push(i);
        }
        let mut count = 0;
        for row_a in &a.data {
            let key = format!("{}", row_a[col_a]);
            if let Some(rows) = b_index.get(&key) {
                count += rows.len();
            }
        }
        assert_eq!(count, 2); // id 1 and 2 match
    }

    #[test]
    fn test_predicate_parsing() {
        let p = parse_predicate_string("age > 25").unwrap();
        assert!(matches!(p, Predicate::Gt(ref col, 25.0) if col == "age"));
        let p = parse_predicate_string("score < 90").unwrap();
        assert!(matches!(p, Predicate::Lt(ref col, 90.0) if col == "score"));
        let p = parse_predicate_string("name == \"Alice\"").unwrap();
        assert!(matches!(p, Predicate::Eq(ref col, ref val) if col == "name" && val == "Alice"));
    }

    #[test]
    fn test_semantic_types() {
        assert_eq!(format!("{}", SemanticType::Raw), "Raw");
        assert_eq!(format!("{}", SemanticType::Probability), "Probability");
        assert!(format!("{}", SemanticType::Embedding { dim: 768, model: "bert".into() }).contains("768"));
    }

    #[test]
    fn test_cell_value_display() {
        assert_eq!(format!("{}", CellValue::Int(42)), "42");
        assert_eq!(format!("{}", CellValue::Float(3.14)), "3.14");
        assert_eq!(format!("{}", CellValue::Null), "null");
    }

    #[test]
    fn test_csv_with_quotes() {
        let csv = "name,desc\n\"Alice\",\"hello, world\"\nBob,simple";
        let stream = DataLoader::load_from_string(csv, DataFormat::Csv, "q.csv").unwrap();
        assert_eq!(stream.row_count(), 2);
        assert_eq!(stream.data[0][1], CellValue::String("hello, world".to_string()));
    }
}
