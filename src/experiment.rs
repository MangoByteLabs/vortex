// Experiment tracking system for Vortex
// Automatically logs metrics, hyperparameters, model checkpoints, and loss curves.
// No external service needed — it's part of the language.

use crate::interpreter::{Env, FnDef, Value};
use crate::nn;
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

// ─── Core Types ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ParamValue {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

impl std::fmt::Display for ParamValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParamValue::Int(v) => write!(f, "{}", v),
            ParamValue::Float(v) => write!(f, "{}", v),
            ParamValue::String(v) => write!(f, "{}", v),
            ParamValue::Bool(v) => write!(f, "{}", v),
        }
    }
}

impl ParamValue {
    fn to_json(&self) -> String {
        match self {
            ParamValue::Int(v) => format!("{}", v),
            ParamValue::Float(v) => format!("{}", v),
            ParamValue::String(v) => format!("\"{}\"", v.replace('\\', "\\\\").replace('"', "\\\"")),
            ParamValue::Bool(v) => format!("{}", v),
        }
    }

    fn from_json_value(s: &str) -> Self {
        let s = s.trim();
        if s == "true" { return ParamValue::Bool(true); }
        if s == "false" { return ParamValue::Bool(false); }
        if s.starts_with('"') && s.ends_with('"') {
            return ParamValue::String(s[1..s.len()-1].replace("\\\"", "\"").replace("\\\\", "\\"));
        }
        if let Ok(i) = s.parse::<i64>() { return ParamValue::Int(i); }
        if let Ok(f) = s.parse::<f64>() { return ParamValue::Float(f); }
        ParamValue::String(s.to_string())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExperimentStatus {
    Running,
    Completed,
    Failed,
    Interrupted,
}

impl std::fmt::Display for ExperimentStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExperimentStatus::Running => write!(f, "Running"),
            ExperimentStatus::Completed => write!(f, "Completed"),
            ExperimentStatus::Failed => write!(f, "Failed"),
            ExperimentStatus::Interrupted => write!(f, "Interrupted"),
        }
    }
}

impl ExperimentStatus {
    fn to_json(&self) -> String {
        format!("\"{}\"", self)
    }

    fn from_str(s: &str) -> Self {
        match s {
            "Running" => ExperimentStatus::Running,
            "Completed" => ExperimentStatus::Completed,
            "Failed" => ExperimentStatus::Failed,
            "Interrupted" => ExperimentStatus::Interrupted,
            _ => ExperimentStatus::Running,
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetricEntry {
    pub step: usize,
    pub epoch: Option<usize>,
    pub metrics: HashMap<String, f64>,
    pub timestamp: String,
}

impl MetricEntry {
    fn to_json(&self) -> String {
        let metrics_json: Vec<String> = self.metrics.iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v))
            .collect();
        let epoch_str = match self.epoch {
            Some(e) => format!("{}", e),
            None => "null".to_string(),
        };
        format!("{{\"step\":{},\"epoch\":{},\"metrics\":{{{}}},\"timestamp\":\"{}\"}}",
            self.step, epoch_str, metrics_json.join(","), self.timestamp)
    }

    fn from_json(s: &str) -> Option<Self> {
        let step = extract_json_int(s, "step")? as usize;
        let epoch = extract_json_int(s, "epoch").map(|v| v as usize);
        let timestamp = extract_json_string(s, "timestamp").unwrap_or_default();
        let metrics = extract_json_metrics(s);
        Some(MetricEntry { step, epoch, metrics, timestamp })
    }
}

#[derive(Debug, Clone)]
pub struct Checkpoint {
    pub step: usize,
    pub path: String,
    pub metrics: HashMap<String, f64>,
    pub is_best: bool,
}

impl Checkpoint {
    fn to_json(&self) -> String {
        let metrics_json: Vec<String> = self.metrics.iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v))
            .collect();
        format!("{{\"step\":{},\"path\":\"{}\",\"metrics\":{{{}}},\"is_best\":{}}}",
            self.step, self.path, metrics_json.join(","), self.is_best)
    }
}

#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub vortex_version: String,
    pub platform: String,
    pub gpu_available: bool,
}

impl SystemInfo {
    fn current() -> Self {
        SystemInfo {
            vortex_version: "0.1.0".to_string(),
            platform: std::env::consts::OS.to_string(),
            gpu_available: false,
        }
    }

    fn to_json(&self) -> String {
        format!("{{\"vortex_version\":\"{}\",\"platform\":\"{}\",\"gpu_available\":{}}}",
            self.vortex_version, self.platform, self.gpu_available)
    }
}

#[derive(Debug, Clone)]
pub struct Experiment {
    pub id: String,
    pub name: String,
    pub created_at: String,
    pub status: ExperimentStatus,
    pub hyperparams: HashMap<String, ParamValue>,
    pub metrics: Vec<MetricEntry>,
    pub checkpoints: Vec<Checkpoint>,
    pub tags: Vec<String>,
    pub notes: String,
    pub system_info: SystemInfo,
}

impl Experiment {
    fn metadata_json(&self) -> String {
        let params_json: Vec<String> = self.hyperparams.iter()
            .map(|(k, v)| format!("\"{}\":{}", k, v.to_json()))
            .collect();
        let tags_json: Vec<String> = self.tags.iter()
            .map(|t| format!("\"{}\"", t))
            .collect();
        let checkpoints_json: Vec<String> = self.checkpoints.iter()
            .map(|c| c.to_json())
            .collect();
        format!(
            "{{\n  \"id\":\"{}\",\n  \"name\":\"{}\",\n  \"created_at\":\"{}\",\n  \"status\":{},\n  \"hyperparams\":{{{}}},\n  \"tags\":[{}],\n  \"notes\":\"{}\",\n  \"system_info\":{},\n  \"checkpoints\":[{}]\n}}",
            self.id, self.name, self.created_at, self.status.to_json(),
            params_json.join(","), tags_json.join(","),
            self.notes.replace('\\', "\\\\").replace('"', "\\\""),
            self.system_info.to_json(),
            checkpoints_json.join(",")
        )
    }
}

#[derive(Debug, Clone)]
pub struct ExperimentSummary {
    pub id: String,
    pub name: String,
    pub status: ExperimentStatus,
    pub created_at: String,
    pub num_metrics: usize,
    pub num_checkpoints: usize,
    pub best_loss: Option<f64>,
}

#[derive(Debug)]
pub struct ComparisonTable {
    pub ids: Vec<String>,
    pub names: Vec<String>,
    pub params: HashMap<String, Vec<Option<ParamValue>>>,
    pub final_metrics: Vec<HashMap<String, f64>>,
}

impl ComparisonTable {
    pub fn render(&self) -> String {
        let mut out = String::new();
        // Header
        out.push_str(&format!("{:<20}", ""));
        for (i, name) in self.names.iter().enumerate() {
            out.push_str(&format!("| {:<25}", format!("{} ({})", name, &self.ids[i][..8.min(self.ids[i].len())])));
        }
        out.push('\n');
        out.push_str(&"-".repeat(20 + self.names.len() * 27));
        out.push('\n');

        // Hyperparams
        out.push_str("HYPERPARAMETERS\n");
        for (key, vals) in &self.params {
            out.push_str(&format!("  {:<18}", key));
            for v in vals {
                match v {
                    Some(pv) => out.push_str(&format!("| {:<25}", format!("{}", pv))),
                    None => out.push_str(&format!("| {:<25}", "-")),
                }
            }
            out.push('\n');
        }

        // Final metrics
        out.push_str("METRICS (final)\n");
        let mut all_keys: Vec<String> = self.final_metrics.iter()
            .flat_map(|m| m.keys().cloned())
            .collect::<std::collections::HashSet<_>>()
            .into_iter().collect();
        all_keys.sort();
        for key in &all_keys {
            out.push_str(&format!("  {:<18}", key));
            for fm in &self.final_metrics {
                match fm.get(key) {
                    Some(v) => out.push_str(&format!("| {:<25}", format!("{:.6}", v))),
                    None => out.push_str(&format!("| {:<25}", "-")),
                }
            }
            out.push('\n');
        }

        out
    }
}

// ─── Experiment Manager ────────────────────────────────────────────────

pub struct ExperimentManager {
    pub base_dir: PathBuf,
    pub active: Option<Experiment>,
}

impl ExperimentManager {
    pub fn new() -> Self {
        let base_dir = dirs_base().join("experiments");
        ExperimentManager { base_dir, active: None }
    }

    pub fn with_base_dir(base_dir: PathBuf) -> Self {
        ExperimentManager { base_dir, active: None }
    }

    pub fn new_experiment(&mut self, name: &str) -> String {
        let id = generate_id();
        let experiment = Experiment {
            id: id.clone(),
            name: name.to_string(),
            created_at: iso_now(),
            status: ExperimentStatus::Running,
            hyperparams: HashMap::new(),
            metrics: Vec::new(),
            checkpoints: Vec::new(),
            tags: Vec::new(),
            notes: String::new(),
            system_info: SystemInfo::current(),
        };
        // Create directory structure
        let exp_dir = self.base_dir.join(&id);
        let _ = fs::create_dir_all(exp_dir.join("checkpoints"));
        // Write initial metadata
        let _ = fs::write(exp_dir.join("metadata.json"), experiment.metadata_json());
        self.active = Some(experiment);
        id
    }

    pub fn log_params(&mut self, params: HashMap<String, ParamValue>) {
        if let Some(ref mut exp) = self.active {
            exp.hyperparams.extend(params);
            self.persist_metadata();
        }
    }

    pub fn log_metric(&mut self, step: usize, name: &str, value: f64) {
        let mut metrics = HashMap::new();
        metrics.insert(name.to_string(), value);
        self.log_metrics(step, None, metrics);
    }

    pub fn log_metrics(&mut self, step: usize, epoch: Option<usize>, metrics: HashMap<String, f64>) {
        if let Some(ref mut exp) = self.active {
            let entry = MetricEntry {
                step,
                epoch,
                metrics,
                timestamp: iso_now(),
            };
            // Append to metrics.jsonl
            let exp_dir = self.base_dir.join(&exp.id);
            let metrics_path = exp_dir.join("metrics.jsonl");
            if let Ok(mut file) = fs::OpenOptions::new()
                .create(true).append(true).open(&metrics_path)
            {
                let _ = writeln!(file, "{}", entry.to_json());
            }
            exp.metrics.push(entry);
        }
    }

    pub fn save_checkpoint(&mut self, model: &nn::Model, step: usize) -> Result<String, String> {
        if let Some(ref mut exp) = self.active {
            let exp_dir = self.base_dir.join(&exp.id);
            let ckpt_path = exp_dir.join("checkpoints").join(format!("step_{}.json", step));
            let ckpt_path_str = ckpt_path.to_string_lossy().to_string();
            nn::save_model(model, &ckpt_path_str)?;

            // Determine if best
            let current_metrics: HashMap<String, f64> = if let Some(last) = exp.metrics.last() {
                last.metrics.clone()
            } else {
                HashMap::new()
            };

            let is_best = if let Some(loss) = current_metrics.get("loss") {
                exp.checkpoints.iter().all(|c| {
                    c.metrics.get("loss").map_or(true, |prev| loss <= prev)
                })
            } else {
                exp.checkpoints.is_empty()
            };

            // If this is best, also save as best.json
            if is_best {
                let best_path = exp_dir.join("checkpoints").join("best.json");
                nn::save_model(model, &best_path.to_string_lossy())?;
            }

            let checkpoint = Checkpoint {
                step,
                path: ckpt_path_str.clone(),
                metrics: current_metrics,
                is_best,
            };
            exp.checkpoints.push(checkpoint);
            self.persist_metadata();
            Ok(ckpt_path_str)
        } else {
            Err("No active experiment".to_string())
        }
    }

    pub fn finish(&mut self, status: ExperimentStatus) {
        if let Some(ref mut exp) = self.active {
            exp.status = status;
            self.persist_metadata();
            self.write_summary();
            self.active = None;
        }
    }

    pub fn list_experiments(&self) -> Vec<ExperimentSummary> {
        let mut results = Vec::new();
        if let Ok(entries) = fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    let meta_path = path.join("metadata.json");
                    if let Ok(content) = fs::read_to_string(&meta_path) {
                        let id = extract_json_string(&content, "id").unwrap_or_default();
                        let name = extract_json_string(&content, "name").unwrap_or_default();
                        let status_str = extract_json_string(&content, "status").unwrap_or_default();
                        let created_at = extract_json_string(&content, "created_at").unwrap_or_default();

                        // Count metrics
                        let metrics_path = path.join("metrics.jsonl");
                        let num_metrics = fs::read_to_string(&metrics_path)
                            .map(|c| c.lines().filter(|l| !l.trim().is_empty()).count())
                            .unwrap_or(0);

                        // Count checkpoints
                        let ckpt_dir = path.join("checkpoints");
                        let num_checkpoints = fs::read_dir(&ckpt_dir)
                            .map(|d| d.filter_map(|e| e.ok()).filter(|e| {
                                e.path().extension().map_or(false, |ext| ext == "json")
                                && e.file_name() != "best.json"
                            }).count())
                            .unwrap_or(0);

                        // Best loss from metrics
                        let best_loss = self.best_loss_from_file(&metrics_path);

                        results.push(ExperimentSummary {
                            id,
                            name,
                            status: ExperimentStatus::from_str(&status_str),
                            created_at,
                            num_metrics,
                            num_checkpoints,
                            best_loss,
                        });
                    }
                }
            }
        }
        results.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        results
    }

    pub fn get_experiment(&self, id: &str) -> Option<Experiment> {
        let exp_dir = self.base_dir.join(id);
        let meta_path = exp_dir.join("metadata.json");
        let content = fs::read_to_string(&meta_path).ok()?;

        let exp_id = extract_json_string(&content, "id")?;
        let name = extract_json_string(&content, "name").unwrap_or_default();
        let created_at = extract_json_string(&content, "created_at").unwrap_or_default();
        let status_str = extract_json_string(&content, "status").unwrap_or_default();
        let notes = extract_json_string(&content, "notes").unwrap_or_default();

        // Parse hyperparams
        let hyperparams = extract_json_params(&content);

        // Read metrics from jsonl
        let metrics_path = exp_dir.join("metrics.jsonl");
        let metrics = if let Ok(metrics_content) = fs::read_to_string(&metrics_path) {
            metrics_content.lines()
                .filter(|l| !l.trim().is_empty())
                .filter_map(|l| MetricEntry::from_json(l))
                .collect()
        } else {
            Vec::new()
        };

        Some(Experiment {
            id: exp_id,
            name,
            created_at,
            status: ExperimentStatus::from_str(&status_str),
            hyperparams,
            metrics,
            checkpoints: Vec::new(),
            tags: Vec::new(),
            notes,
            system_info: SystemInfo::current(),
        })
    }

    pub fn compare(&self, ids: &[String]) -> ComparisonTable {
        let mut names = Vec::new();
        let mut all_params: HashMap<String, Vec<Option<ParamValue>>> = HashMap::new();
        let mut final_metrics = Vec::new();

        for id in ids {
            if let Some(exp) = self.get_experiment(id) {
                names.push(exp.name.clone());
                for (k, v) in &exp.hyperparams {
                    let entry = all_params.entry(k.clone()).or_insert_with(|| vec![None; names.len() - 1]);
                    while entry.len() < names.len() - 1 { entry.push(None); }
                    entry.push(Some(v.clone()));
                }
                // Pad missing params
                for (_, vals) in all_params.iter_mut() {
                    while vals.len() < names.len() { vals.push(None); }
                }
                // Final metrics: last entry
                let fm = exp.metrics.last()
                    .map(|m| m.metrics.clone())
                    .unwrap_or_default();
                final_metrics.push(fm);
            } else {
                names.push(format!("(not found: {})", id));
                for (_, vals) in all_params.iter_mut() {
                    vals.push(None);
                }
                final_metrics.push(HashMap::new());
            }
        }

        ComparisonTable {
            ids: ids.to_vec(),
            names,
            params: all_params,
            final_metrics,
        }
    }

    pub fn best_experiment(&self, metric: &str) -> Option<ExperimentSummary> {
        let experiments = self.list_experiments();
        let mut best: Option<(ExperimentSummary, f64)> = None;

        for summary in experiments {
            if let Some(exp) = self.get_experiment(&summary.id) {
                // Find the best (lowest) value of the given metric across all entries
                for entry in &exp.metrics {
                    if let Some(&val) = entry.metrics.get(metric) {
                        match &best {
                            Some((_, best_val)) if val < *best_val => {
                                best = Some((summary.clone(), val));
                            }
                            None => {
                                best = Some((summary.clone(), val));
                            }
                            _ => {}
                        }
                        break; // Use last entry for comparison
                    }
                }
                // Actually use the last metric entry
                if let Some(last) = exp.metrics.last() {
                    if let Some(&val) = last.metrics.get(metric) {
                        match &best {
                            Some((_, best_val)) if val < *best_val => {
                                best = Some((summary.clone(), val));
                            }
                            None => {
                                best = Some((summary.clone(), val));
                            }
                            _ => {}
                        }
                    }
                }
            }
        }

        best.map(|(s, _)| s)
    }

    // ─── Auto-tracking integration ─────────────────────────────────────

    pub fn auto_track_train(
        &mut self,
        model_name: &str,
        model: &mut nn::Model,
        data: Vec<Vec<f64>>,
        labels: Vec<Vec<f64>>,
        optimizer_name: &str,
        lr: f64,
        epochs: usize,
    ) -> Vec<f64> {
        // Create experiment if none active
        if self.active.is_none() {
            self.new_experiment(&format!("auto_{}", model_name));
        }

        // Log hyperparams
        let mut params = HashMap::new();
        params.insert("model".to_string(), ParamValue::String(model_name.to_string()));
        params.insert("optimizer".to_string(), ParamValue::String(optimizer_name.to_string()));
        params.insert("learning_rate".to_string(), ParamValue::Float(lr));
        params.insert("epochs".to_string(), ParamValue::Int(epochs as i64));
        params.insert("num_samples".to_string(), ParamValue::Int(data.len() as i64));
        self.log_params(params);

        // Train with tracking
        let mut optimizer = match optimizer_name {
            "sgd" => nn::Optimizer::sgd(lr, 0.9, 0.0),
            "adamw" => nn::Optimizer::adamw(lr, 0.01),
            _ => nn::Optimizer::adam(lr),
        };

        let losses = nn::train(model, data, labels, &mut optimizer, epochs, "mse");

        // Log each epoch loss
        let mut best_loss = f64::INFINITY;
        for (i, &loss) in losses.iter().enumerate() {
            self.log_metric(i, "loss", loss);
            if loss < best_loss {
                best_loss = loss;
            }
        }

        // Save checkpoint at best
        let _ = self.save_checkpoint(model, losses.len().saturating_sub(1));

        // Finish
        self.finish(ExperimentStatus::Completed);

        losses
    }

    // ─── Private helpers ───────────────────────────────────────────────

    fn persist_metadata(&self) {
        if let Some(ref exp) = self.active {
            let exp_dir = self.base_dir.join(&exp.id);
            let _ = fs::write(exp_dir.join("metadata.json"), exp.metadata_json());
        }
    }

    fn write_summary(&self) {
        if let Some(ref exp) = self.active {
            let exp_dir = self.base_dir.join(&exp.id);
            let best_loss = exp.metrics.iter()
                .filter_map(|m| m.metrics.get("loss"))
                .cloned()
                .fold(f64::INFINITY, f64::min);
            let final_loss = exp.metrics.last()
                .and_then(|m| m.metrics.get("loss"))
                .cloned()
                .unwrap_or(0.0);
            let summary = format!(
                "{{\n  \"id\":\"{}\",\n  \"name\":\"{}\",\n  \"status\":\"{}\",\n  \"total_steps\":{},\n  \"best_loss\":{},\n  \"final_loss\":{},\n  \"num_checkpoints\":{},\n  \"created_at\":\"{}\"\n}}",
                exp.id, exp.name, exp.status, exp.metrics.len(),
                if best_loss.is_infinite() { 0.0 } else { best_loss },
                final_loss, exp.checkpoints.len(), exp.created_at
            );
            let _ = fs::write(exp_dir.join("summary.json"), summary);
        }
    }

    fn best_loss_from_file(&self, path: &PathBuf) -> Option<f64> {
        let content = fs::read_to_string(path).ok()?;
        let mut best: Option<f64> = None;
        for line in content.lines() {
            if let Some(entry) = MetricEntry::from_json(line) {
                if let Some(&loss) = entry.metrics.get("loss") {
                    best = Some(best.map_or(loss, |b: f64| b.min(loss)));
                }
            }
        }
        best
    }
}

// ─── ASCII Chart Renderer ──────────────────────────────────────────────

pub fn render_ascii_chart(values: &[f64], title: &str, width: usize, height: usize) -> String {
    if values.is_empty() {
        return format!("{}: (no data)\n", title);
    }

    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = if (max_val - min_val).abs() < 1e-12 { 1.0 } else { max_val - min_val };

    let mut out = format!("{} over {} steps:\n", title, values.len());

    // Create grid
    let chart_width = width.min(values.len());
    let mut grid = vec![vec![' '; chart_width]; height];

    // Plot points
    for col in 0..chart_width {
        let idx = if chart_width == 1 { 0 } else { col * (values.len() - 1) / (chart_width - 1) };
        let val = values[idx.min(values.len() - 1)];
        let row = ((max_val - val) / range * (height - 1) as f64).round() as usize;
        let row = row.min(height - 1);
        grid[row][col] = '*';
    }

    // Render rows with y-axis labels
    for r in 0..height {
        let y_val = max_val - (r as f64 / (height - 1) as f64) * range;
        out.push_str(&format!("{:>8.4} |", y_val));
        for c in 0..chart_width {
            out.push(grid[r][c]);
        }
        out.push('\n');
    }

    // X-axis
    out.push_str(&format!("{:>8} +", ""));
    out.push_str(&"-".repeat(chart_width));
    out.push('\n');

    // X-axis labels
    out.push_str(&format!("{:>8}  ", ""));
    let num_labels = 5.min(chart_width);
    if num_labels > 0 {
        for i in 0..num_labels {
            let pos = if num_labels == 1 { 0 } else { i * (values.len() - 1) / (num_labels - 1) };
            let label = format!("{}", pos);
            let col_pos = if num_labels == 1 { 0 } else { i * (chart_width - 1) / (num_labels - 1) };
            while out.len() < 10 + col_pos { out.push(' '); }
            out.push_str(&label);
        }
    }
    out.push('\n');

    out
}

// ─── CLI Display Functions ─────────────────────────────────────────────

pub fn cli_list_experiments(base_dir: Option<PathBuf>) {
    let mgr = match base_dir {
        Some(d) => ExperimentManager::with_base_dir(d),
        None => ExperimentManager::new(),
    };
    let experiments = mgr.list_experiments();
    if experiments.is_empty() {
        println!("No experiments found.");
        println!("Start tracking with: experiment_new(\"my_experiment\")");
        return;
    }

    println!("{:<12} {:<25} {:<12} {:<8} {:<8} {:<12}",
        "ID", "NAME", "STATUS", "METRICS", "CKPTS", "BEST LOSS");
    println!("{}", "-".repeat(77));
    for exp in &experiments {
        let short_id = if exp.id.len() > 10 { &exp.id[..10] } else { &exp.id };
        let loss_str = exp.best_loss.map_or("-".to_string(), |l| format!("{:.6}", l));
        println!("{:<12} {:<25} {:<12} {:<8} {:<8} {:<12}",
            short_id, exp.name, format!("{}", exp.status),
            exp.num_metrics, exp.num_checkpoints, loss_str);
    }
    println!("\n{} experiment(s) total.", experiments.len());
}

pub fn cli_show_experiment(id: &str, base_dir: Option<PathBuf>) {
    let mgr = match base_dir {
        Some(d) => ExperimentManager::with_base_dir(d),
        None => ExperimentManager::new(),
    };
    match mgr.get_experiment(id) {
        Some(exp) => {
            println!("Experiment: {} ({})", exp.name, exp.id);
            println!("Status: {}", exp.status);
            println!("Created: {}", exp.created_at);
            println!();

            if !exp.hyperparams.is_empty() {
                println!("Hyperparameters:");
                for (k, v) in &exp.hyperparams {
                    println!("  {}: {}", k, v);
                }
                println!();
            }

            if !exp.metrics.is_empty() {
                println!("Metrics ({} entries):", exp.metrics.len());
                // Show last 5 entries
                let start = exp.metrics.len().saturating_sub(5);
                for entry in &exp.metrics[start..] {
                    let metrics_str: Vec<String> = entry.metrics.iter()
                        .map(|(k, v)| format!("{}={:.6}", k, v))
                        .collect();
                    println!("  step {}: {}", entry.step, metrics_str.join(", "));
                }
                if exp.metrics.len() > 5 {
                    println!("  ... ({} more entries)", exp.metrics.len() - 5);
                }
                println!();

                // ASCII loss curve
                let losses: Vec<f64> = exp.metrics.iter()
                    .filter_map(|m| m.metrics.get("loss").cloned())
                    .collect();
                if !losses.is_empty() {
                    println!("{}", render_ascii_chart(&losses, "Loss", 50, 8));
                }
            }
        }
        None => {
            eprintln!("Experiment '{}' not found.", id);
        }
    }
}

pub fn cli_compare_experiments(ids: &[String], base_dir: Option<PathBuf>) {
    let mgr = match base_dir {
        Some(d) => ExperimentManager::with_base_dir(d),
        None => ExperimentManager::new(),
    };
    let table = mgr.compare(ids);
    println!("{}", table.render());
}

// ─── Interpreter Builtins ──────────────────────────────────────────────

pub fn register_experiment_builtins(env: &mut Env) {
    env.functions.insert("experiment_new".into(), FnDef::Builtin(builtin_experiment_new));
    env.functions.insert("experiment_log_param".into(), FnDef::Builtin(builtin_experiment_log_param));
    env.functions.insert("experiment_log_metric".into(), FnDef::Builtin(builtin_experiment_log_metric));
    env.functions.insert("experiment_log_metrics".into(), FnDef::Builtin(builtin_experiment_log_metrics));
    env.functions.insert("experiment_checkpoint".into(), FnDef::Builtin(builtin_experiment_checkpoint));
    env.functions.insert("experiment_finish".into(), FnDef::Builtin(builtin_experiment_finish));
    env.functions.insert("experiment_list".into(), FnDef::Builtin(builtin_experiment_list));
    env.functions.insert("experiment_best".into(), FnDef::Builtin(builtin_experiment_best));
    env.functions.insert("experiment_compare".into(), FnDef::Builtin(builtin_experiment_compare));
}

fn builtin_experiment_new(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("experiment_new(name) expects 1 argument".into()); }
    let name = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("experiment_new expects a string name".into()),
    };
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let id = mgr.new_experiment(&name);
    env.output.push(format!("Experiment started: {} ({})", name, &id[..8.min(id.len())]));
    Ok(Value::String(id))
}

fn builtin_experiment_log_param(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("experiment_log_param(key, value) expects 2 arguments".into()); }
    let key = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("key must be a string".into()),
    };
    let param_value = value_to_param(&args[1])?;
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let mut params = HashMap::new();
    params.insert(key, param_value);
    mgr.log_params(params);
    Ok(Value::Void)
}

fn builtin_experiment_log_metric(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 3 { return Err("experiment_log_metric(step, name, value) expects 3 arguments".into()); }
    let step = match &args[0] {
        Value::Int(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("step must be numeric".into()),
    };
    let name = match &args[1] {
        Value::String(s) => s.clone(),
        _ => return Err("name must be a string".into()),
    };
    let value = match &args[2] {
        Value::Float(f) => *f,
        Value::Int(i) => *i as f64,
        _ => return Err("value must be numeric".into()),
    };
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    mgr.log_metric(step, &name, value);
    Ok(Value::Void)
}

fn builtin_experiment_log_metrics(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 2 { return Err("experiment_log_metrics(step, metrics_dict) expects 2 arguments".into()); }
    let step = match &args[0] {
        Value::Int(i) => *i as usize,
        Value::Float(f) => *f as usize,
        _ => return Err("step must be numeric".into()),
    };
    let metrics = match &args[1] {
        Value::HashMap(map) => {
            let mut m = HashMap::new();
            for (k, v) in map {
                let val = match v {
                    Value::Float(f) => *f,
                    Value::Int(i) => *i as f64,
                    _ => return Err(format!("metric '{}' must be numeric", k)),
                };
                m.insert(k.clone(), val);
            }
            m
        }
        _ => return Err("metrics must be a dict/hashmap".into()),
    };
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    mgr.log_metrics(step, None, metrics);
    Ok(Value::Void)
}

fn builtin_experiment_checkpoint(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("experiment_checkpoint(model_id) expects 1 argument".into()); }
    let model_id = match &args[0] {
        Value::Int(i) => *i as usize,
        _ => return Err("model_id must be an integer".into()),
    };
    let model = env.nn_models.get(&model_id).ok_or("no such model")?.clone();
    let step = env.experiment_manager.as_ref()
        .and_then(|m| m.active.as_ref())
        .map(|e| e.metrics.len())
        .unwrap_or(0);
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let path = mgr.save_checkpoint(&model, step)?;
    Ok(Value::String(path))
}

fn builtin_experiment_finish(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let id = mgr.active.as_ref().map(|e| e.id.clone()).unwrap_or_default();
    mgr.finish(ExperimentStatus::Completed);
    env.output.push(format!("Experiment completed: {}", &id[..8.min(id.len())]));
    Ok(Value::Void)
}

fn builtin_experiment_list(env: &mut Env, _args: Vec<Value>) -> Result<Value, String> {
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let experiments = mgr.list_experiments();
    let result: Vec<Value> = experiments.iter().map(|e| {
        let mut map = HashMap::new();
        map.insert("id".to_string(), Value::String(e.id.clone()));
        map.insert("name".to_string(), Value::String(e.name.clone()));
        map.insert("status".to_string(), Value::String(format!("{}", e.status)));
        map.insert("num_metrics".to_string(), Value::Int(e.num_metrics as i128));
        map.insert("num_checkpoints".to_string(), Value::Int(e.num_checkpoints as i128));
        if let Some(loss) = e.best_loss {
            map.insert("best_loss".to_string(), Value::Float(loss));
        }
        Value::HashMap(map)
    }).collect();
    Ok(Value::Array(result))
}

fn builtin_experiment_best(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("experiment_best(metric_name) expects 1 argument".into()); }
    let metric = match &args[0] {
        Value::String(s) => s.clone(),
        _ => return Err("metric name must be a string".into()),
    };
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    match mgr.best_experiment(&metric) {
        Some(summary) => {
            let mut map = HashMap::new();
            map.insert("id".to_string(), Value::String(summary.id));
            map.insert("name".to_string(), Value::String(summary.name));
            map.insert("status".to_string(), Value::String(format!("{}", summary.status)));
            Ok(Value::HashMap(map))
        }
        None => Ok(Value::Option(None)),
    }
}

fn builtin_experiment_compare(env: &mut Env, args: Vec<Value>) -> Result<Value, String> {
    if args.len() != 1 { return Err("experiment_compare(ids_array) expects 1 argument".into()); }
    let ids = match &args[0] {
        Value::Array(arr) => arr.iter().map(|v| match v {
            Value::String(s) => Ok(s.clone()),
            _ => Err("ids must be strings".to_string()),
        }).collect::<Result<Vec<String>, String>>()?,
        _ => return Err("expected array of id strings".into()),
    };
    let mgr = env.experiment_manager.get_or_insert_with(ExperimentManager::new);
    let table = mgr.compare(&ids);
    let rendered = table.render();
    env.output.push(rendered.clone());
    Ok(Value::String(rendered))
}

// ─── Helpers ───────────────────────────────────────────────────────────

fn dirs_base() -> PathBuf {
    if let Ok(home) = std::env::var("HOME") {
        PathBuf::from(home).join(".vortex")
    } else {
        PathBuf::from(".vortex")
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let nanos = ts.as_nanos();
    // Simple hash-based ID from timestamp + random-ish bits
    let hash = nanos.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    format!("{:016x}", hash as u64)
}

fn iso_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = ts.as_secs();
    // Simple UTC timestamp (no chrono dependency)
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;

    // Calculate date from days since epoch (1970-01-01)
    let (year, month, day) = days_to_date(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", year, month, day, hours, minutes, seconds)
}

fn days_to_date(days: u64) -> (u64, u64, u64) {
    // Algorithm from http://howardhinnant.github.io/date_algorithms.html
    let z = days + 719468;
    let era = z / 146097;
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    (y, m, d)
}

fn value_to_param(v: &Value) -> Result<ParamValue, String> {
    match v {
        Value::Int(i) => Ok(ParamValue::Int(*i as i64)),
        Value::Float(f) => Ok(ParamValue::Float(*f)),
        Value::String(s) => Ok(ParamValue::String(s.clone())),
        Value::Bool(b) => Ok(ParamValue::Bool(*b)),
        _ => Err("param value must be int, float, string, or bool".into()),
    }
}

// Simple JSON field extraction (no external dependency)
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let pattern = format!("\"{}\":\"", key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let end = rest.find('"')?;
    Some(rest[..end].to_string())
}

fn extract_json_int(json: &str, key: &str) -> Option<i64> {
    let pattern = format!("\"{}\":", key);
    let start = json.find(&pattern)?;
    let rest = &json[start + pattern.len()..];
    let rest = rest.trim_start();
    if rest.starts_with("null") { return None; }
    let end = rest.find(|c: char| !c.is_ascii_digit() && c != '-' && c != '.').unwrap_or(rest.len());
    rest[..end].parse().ok()
}

fn extract_json_metrics(json: &str) -> HashMap<String, f64> {
    let mut result = HashMap::new();
    let pattern = "\"metrics\":{";
    if let Some(start) = json.find(pattern) {
        let rest = &json[start + pattern.len()..];
        if let Some(end) = rest.find('}') {
            let inner = &rest[..end];
            // Parse key:value pairs
            let mut chars = inner.chars().peekable();
            loop {
                // Skip whitespace and commas
                while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ',') {
                    chars.next();
                }
                if chars.peek().is_none() { break; }
                // Read key
                if chars.peek() != Some(&'"') { break; }
                chars.next(); // skip opening quote
                let key: String = chars.by_ref().take_while(|c| *c != '"').collect();
                // Skip colon
                while chars.peek().map_or(false, |c| c.is_whitespace() || *c == ':') {
                    chars.next();
                }
                // Read value
                let val_str: String = chars.by_ref().take_while(|c| !c.is_whitespace() && *c != ',' && *c != '}').collect();
                if let Ok(v) = val_str.parse::<f64>() {
                    result.insert(key, v);
                }
            }
        }
    }
    result
}

fn extract_json_params(json: &str) -> HashMap<String, ParamValue> {
    let mut result = HashMap::new();
    let pattern = "\"hyperparams\":{";
    if let Some(start) = json.find(pattern) {
        let rest = &json[start + pattern.len()..];
        // Find matching closing brace
        let mut depth = 1;
        let mut end = 0;
        for (i, c) in rest.chars().enumerate() {
            match c {
                '{' => depth += 1,
                '}' => { depth -= 1; if depth == 0 { end = i; break; } }
                _ => {}
            }
        }
        let inner = &rest[..end];
        // Simple parsing of "key":value pairs
        let mut pos = 0;
        let bytes = inner.as_bytes();
        while pos < bytes.len() {
            // Find key
            if let Some(key_start) = inner[pos..].find('"') {
                let ks = pos + key_start + 1;
                if let Some(key_end) = inner[ks..].find('"') {
                    let key = inner[ks..ks + key_end].to_string();
                    pos = ks + key_end + 1;
                    // Skip colon
                    while pos < bytes.len() && (bytes[pos] == b':' || bytes[pos] == b' ') { pos += 1; }
                    // Read value
                    let val_start = pos;
                    if pos < bytes.len() && bytes[pos] == b'"' {
                        // String value
                        pos += 1;
                        let mut val = String::new();
                        while pos < bytes.len() && bytes[pos] != b'"' {
                            if bytes[pos] == b'\\' && pos + 1 < bytes.len() {
                                pos += 1;
                            }
                            val.push(bytes[pos] as char);
                            pos += 1;
                        }
                        pos += 1; // skip closing quote
                        result.insert(key, ParamValue::String(val));
                    } else {
                        // Numeric or bool
                        while pos < bytes.len() && bytes[pos] != b',' && bytes[pos] != b'}' { pos += 1; }
                        let val_str = inner[val_start..pos].trim();
                        result.insert(key, ParamValue::from_json_value(val_str));
                    }
                } else {
                    break;
                }
            } else {
                break;
            }
            // Skip comma
            while pos < bytes.len() && (bytes[pos] == b',' || bytes[pos] == b' ' || bytes[pos] == b'\n') { pos += 1; }
        }
    }
    result
}

// ─── Tests ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn test_manager() -> (ExperimentManager, TempDir) {
        let tmp = TempDir::new().unwrap();
        let mgr = ExperimentManager::with_base_dir(tmp.path().to_path_buf());
        (mgr, tmp)
    }

    #[test]
    fn test_create_experiment() {
        let (mut mgr, _tmp) = test_manager();
        let id = mgr.new_experiment("test_exp");
        assert!(!id.is_empty());
        assert!(mgr.active.is_some());
        assert_eq!(mgr.active.as_ref().unwrap().name, "test_exp");
        assert_eq!(mgr.active.as_ref().unwrap().status, ExperimentStatus::Running);
    }

    #[test]
    fn test_finish_experiment() {
        let (mut mgr, _tmp) = test_manager();
        let _id = mgr.new_experiment("finish_test");
        mgr.finish(ExperimentStatus::Completed);
        assert!(mgr.active.is_none());
    }

    #[test]
    fn test_log_params() {
        let (mut mgr, _tmp) = test_manager();
        mgr.new_experiment("param_test");
        let mut params = HashMap::new();
        params.insert("lr".to_string(), ParamValue::Float(0.01));
        params.insert("epochs".to_string(), ParamValue::Int(100));
        params.insert("optimizer".to_string(), ParamValue::String("adam".to_string()));
        mgr.log_params(params);
        let exp = mgr.active.as_ref().unwrap();
        assert_eq!(exp.hyperparams.get("lr"), Some(&ParamValue::Float(0.01)));
        assert_eq!(exp.hyperparams.get("epochs"), Some(&ParamValue::Int(100)));
    }

    #[test]
    fn test_log_metric() {
        let (mut mgr, _tmp) = test_manager();
        mgr.new_experiment("metric_test");
        mgr.log_metric(0, "loss", 1.5);
        mgr.log_metric(1, "loss", 1.2);
        mgr.log_metric(2, "loss", 0.8);
        assert_eq!(mgr.active.as_ref().unwrap().metrics.len(), 3);
        assert_eq!(mgr.active.as_ref().unwrap().metrics[2].metrics.get("loss"), Some(&0.8));
    }

    #[test]
    fn test_log_metrics_batch() {
        let (mut mgr, _tmp) = test_manager();
        mgr.new_experiment("batch_metric_test");
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.5);
        metrics.insert("accuracy".to_string(), 0.92);
        mgr.log_metrics(0, Some(0), metrics);
        let entry = &mgr.active.as_ref().unwrap().metrics[0];
        assert_eq!(entry.metrics.get("accuracy"), Some(&0.92));
        assert_eq!(entry.epoch, Some(0));
    }

    #[test]
    fn test_metrics_persist_to_disk() {
        let (mut mgr, tmp) = test_manager();
        let id = mgr.new_experiment("persist_test");
        mgr.log_metric(0, "loss", 2.0);
        mgr.log_metric(1, "loss", 1.5);
        mgr.log_metric(2, "loss", 1.0);

        // Check metrics.jsonl exists and has 3 lines
        let metrics_path = tmp.path().join(&id).join("metrics.jsonl");
        let content = fs::read_to_string(&metrics_path).unwrap();
        let lines: Vec<&str> = content.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3);

        // Verify we can parse them back
        let entry = MetricEntry::from_json(lines[0]).unwrap();
        assert_eq!(entry.step, 0);
        assert_eq!(entry.metrics.get("loss"), Some(&2.0));
    }

    #[test]
    fn test_list_experiments() {
        let (mut mgr, _tmp) = test_manager();
        mgr.new_experiment("exp_a");
        mgr.log_metric(0, "loss", 1.0);
        mgr.finish(ExperimentStatus::Completed);

        mgr.new_experiment("exp_b");
        mgr.log_metric(0, "loss", 0.5);
        mgr.finish(ExperimentStatus::Completed);

        let list = mgr.list_experiments();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_get_experiment() {
        let (mut mgr, _tmp) = test_manager();
        let id = mgr.new_experiment("get_test");
        let mut params = HashMap::new();
        params.insert("lr".to_string(), ParamValue::Float(0.001));
        mgr.log_params(params);
        mgr.log_metric(0, "loss", 1.0);
        mgr.log_metric(1, "loss", 0.5);
        mgr.finish(ExperimentStatus::Completed);

        let exp = mgr.get_experiment(&id).unwrap();
        assert_eq!(exp.name, "get_test");
        assert_eq!(exp.metrics.len(), 2);
        assert_eq!(exp.hyperparams.get("lr"), Some(&ParamValue::Float(0.001)));
    }

    #[test]
    fn test_compare_experiments() {
        let (mut mgr, _tmp) = test_manager();
        let id1 = mgr.new_experiment("cmp_a");
        let mut p = HashMap::new();
        p.insert("lr".to_string(), ParamValue::Float(0.01));
        mgr.log_params(p);
        mgr.log_metric(0, "loss", 0.3);
        mgr.finish(ExperimentStatus::Completed);

        let id2 = mgr.new_experiment("cmp_b");
        let mut p = HashMap::new();
        p.insert("lr".to_string(), ParamValue::Float(0.001));
        mgr.log_params(p);
        mgr.log_metric(0, "loss", 0.1);
        mgr.finish(ExperimentStatus::Completed);

        let table = mgr.compare(&[id1, id2]);
        assert_eq!(table.names.len(), 2);
        assert!(table.params.contains_key("lr"));
        let rendered = table.render();
        assert!(rendered.contains("lr"));
    }

    #[test]
    fn test_ascii_chart_basic() {
        let values = vec![1.0, 0.8, 0.6, 0.4, 0.2, 0.1];
        let chart = render_ascii_chart(&values, "Loss", 30, 6);
        assert!(chart.contains("Loss"));
        assert!(chart.contains("*"));
    }

    #[test]
    fn test_ascii_chart_empty() {
        let chart = render_ascii_chart(&[], "Loss", 30, 6);
        assert!(chart.contains("no data"));
    }

    #[test]
    fn test_ascii_chart_single_value() {
        let chart = render_ascii_chart(&[0.5], "Loss", 30, 6);
        assert!(chart.contains("*"));
    }

    #[test]
    fn test_best_experiment() {
        let (mut mgr, _tmp) = test_manager();
        mgr.new_experiment("best_a");
        mgr.log_metric(0, "loss", 0.5);
        mgr.finish(ExperimentStatus::Completed);

        mgr.new_experiment("best_b");
        mgr.log_metric(0, "loss", 0.1);
        mgr.finish(ExperimentStatus::Completed);

        let best = mgr.best_experiment("loss").unwrap();
        assert_eq!(best.name, "best_b");
    }

    #[test]
    fn test_experiment_status_display() {
        assert_eq!(format!("{}", ExperimentStatus::Running), "Running");
        assert_eq!(format!("{}", ExperimentStatus::Completed), "Completed");
        assert_eq!(format!("{}", ExperimentStatus::Failed), "Failed");
        assert_eq!(format!("{}", ExperimentStatus::Interrupted), "Interrupted");
    }

    #[test]
    fn test_param_value_roundtrip() {
        let cases = vec![
            ParamValue::Int(42),
            ParamValue::Float(3.14),
            ParamValue::String("hello".to_string()),
            ParamValue::Bool(true),
        ];
        for pv in &cases {
            let json = pv.to_json();
            let parsed = ParamValue::from_json_value(&json);
            assert_eq!(*pv, parsed, "roundtrip failed for {:?}", pv);
        }
    }

    #[test]
    fn test_summary_written_on_finish() {
        let (mut mgr, tmp) = test_manager();
        let id = mgr.new_experiment("summary_test");
        mgr.log_metric(0, "loss", 1.0);
        mgr.log_metric(1, "loss", 0.5);
        mgr.finish(ExperimentStatus::Completed);

        let summary_path = tmp.path().join(&id).join("summary.json");
        assert!(summary_path.exists());
        let content = fs::read_to_string(&summary_path).unwrap();
        assert!(content.contains("summary_test"));
        assert!(content.contains("Completed"));
    }

    #[test]
    fn test_metadata_written() {
        let (mut mgr, tmp) = test_manager();
        let id = mgr.new_experiment("meta_test");
        let meta_path = tmp.path().join(&id).join("metadata.json");
        assert!(meta_path.exists());
        let content = fs::read_to_string(&meta_path).unwrap();
        assert!(content.contains("meta_test"));
        assert!(content.contains("Running"));
    }

    #[test]
    fn test_iso_timestamp_format() {
        let ts = iso_now();
        // Should match YYYY-MM-DDTHH:MM:SSZ pattern
        assert!(ts.len() >= 19);
        assert!(ts.contains('T'));
        assert!(ts.ends_with('Z'));
    }

    #[test]
    fn test_generate_id_unique() {
        let id1 = generate_id();
        // Small delay to ensure different timestamp
        std::thread::sleep(std::time::Duration::from_millis(1));
        let id2 = generate_id();
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_metric_entry_json_roundtrip() {
        let mut metrics = HashMap::new();
        metrics.insert("loss".to_string(), 0.42);
        metrics.insert("accuracy".to_string(), 0.95);
        let entry = MetricEntry {
            step: 10,
            epoch: Some(1),
            metrics,
            timestamp: "2026-01-01T00:00:00Z".to_string(),
        };
        let json = entry.to_json();
        let parsed = MetricEntry::from_json(&json).unwrap();
        assert_eq!(parsed.step, 10);
        assert_eq!(parsed.epoch, Some(1));
        assert_eq!(parsed.metrics.get("loss"), Some(&0.42));
    }
}
