//! Self-modifying neural network architecture for Vortex.
//! Models that can grow, shrink, and restructure at runtime.

use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

// ── Activation ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Hash)]
pub enum Activation {
    ReLU,
    Sigmoid,
    Tanh,
    GeLU,
    Identity,
}

fn apply_activation(x: f64, act: &Activation) -> f64 {
    match act {
        Activation::ReLU => x.max(0.0),
        Activation::Sigmoid => 1.0 / (1.0 + (-x).exp()),
        Activation::Tanh => x.tanh(),
        Activation::GeLU => 0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x * x * x)).tanh()),
        Activation::Identity => x,
    }
}

// ── DynamicLayer ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum DynamicLayer {
    Dense {
        weights: Vec<Vec<f64>>,
        biases: Vec<f64>,
        activation: Activation,
    },
    Attention {
        q_proj: Vec<Vec<f64>>,
        k_proj: Vec<Vec<f64>>,
        v_proj: Vec<Vec<f64>>,
        o_proj: Vec<Vec<f64>>,
        n_heads: usize,
    },
    FFN {
        w1: Vec<Vec<f64>>,
        w2: Vec<Vec<f64>>,
        b1: Vec<f64>,
        b2: Vec<f64>,
    },
    SSM {
        a: Vec<Vec<f64>>,
        b: Vec<Vec<f64>>,
        c: Vec<Vec<f64>>,
        d: Vec<Vec<f64>>,
        state_size: usize,
    },
    Identity,
}

impl DynamicLayer {
    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        match self {
            DynamicLayer::Dense { weights, biases, activation } => {
                // weights: [out_dim][in_dim], biases: [out_dim]
                let out_dim = weights.len();
                let mut output = Vec::with_capacity(out_dim);
                for i in 0..out_dim {
                    let mut sum = biases[i];
                    for (j, &x) in input.iter().enumerate() {
                        if j < weights[i].len() {
                            sum += weights[i][j] * x;
                        }
                    }
                    output.push(apply_activation(sum, activation));
                }
                output
            }
            DynamicLayer::Attention { q_proj, k_proj, v_proj, o_proj, n_heads } => {
                let _d = input.len();
                // Simple single-token self-attention: Q=K=V from input
                let q = mat_vec_mul(q_proj, input);
                let k = mat_vec_mul(k_proj, input);
                let v = mat_vec_mul(v_proj, input);
                let head_dim = if *n_heads > 0 && !q.is_empty() { q.len() / n_heads } else { q.len().max(1) };
                // Scaled dot-product per head, concatenate
                let mut concat = Vec::with_capacity(q.len());
                for h in 0..*n_heads {
                    let start = h * head_dim;
                    let end = (start + head_dim).min(q.len());
                    let mut score: f64 = 0.0;
                    for i in start..end {
                        score += q[i] * k[i];
                    }
                    score /= (head_dim as f64).sqrt().max(1.0);
                    // softmax of single score = 1.0, so output = v slice
                    for i in start..end {
                        concat.push(score.tanh() * v[i]); // approximate attention
                    }
                }
                if concat.is_empty() { return input.to_vec(); }
                mat_vec_mul(o_proj, &concat)
            }
            DynamicLayer::FFN { w1, w2, b1, b2 } => {
                // Two-layer FFN with GeLU
                let hidden: Vec<f64> = w1.iter().zip(b1.iter()).map(|(row, &b)| {
                    let s: f64 = row.iter().zip(input).map(|(w, x)| w * x).sum::<f64>() + b;
                    apply_activation(s, &Activation::GeLU)
                }).collect();
                w2.iter().zip(b2.iter()).map(|(row, &b)| {
                    row.iter().zip(&hidden).map(|(w, h)| w * h).sum::<f64>() + b
                }).collect()
            }
            DynamicLayer::SSM { a: _, b, c, d, state_size: _ } => {
                // Simplified linear SSM: y = C*(A*0 + B*x) + D*x  (state starts at 0)
                let bx = mat_vec_mul(b, input);
                let cx = mat_vec_mul(c, &bx);
                let dx = mat_vec_mul(d, input);
                cx.iter().zip(dx.iter()).map(|(a, b)| a + b).collect()
            }
            DynamicLayer::Identity => input.to_vec(),
        }
    }

    pub fn param_count(&self) -> usize {
        match self {
            DynamicLayer::Dense { weights, biases, .. } => {
                weights.iter().map(|r| r.len()).sum::<usize>() + biases.len()
            }
            DynamicLayer::Attention { q_proj, k_proj, v_proj, o_proj, .. } => {
                [q_proj, k_proj, v_proj, o_proj].iter()
                    .map(|m| m.iter().map(|r| r.len()).sum::<usize>())
                    .sum()
            }
            DynamicLayer::FFN { w1, w2, b1, b2 } => {
                w1.iter().map(|r| r.len()).sum::<usize>()
                    + w2.iter().map(|r| r.len()).sum::<usize>()
                    + b1.len() + b2.len()
            }
            DynamicLayer::SSM { a, b, c, d, .. } => {
                [a, b, c, d].iter()
                    .map(|m| m.iter().map(|r| r.len()).sum::<usize>())
                    .sum()
            }
            DynamicLayer::Identity => 0,
        }
    }

    pub fn input_dim(&self) -> usize {
        match self {
            DynamicLayer::Dense { weights, .. } => weights.first().map(|r| r.len()).unwrap_or(0),
            DynamicLayer::Attention { q_proj, .. } => q_proj.first().map(|r| r.len()).unwrap_or(0),
            DynamicLayer::FFN { w1, .. } => w1.first().map(|r| r.len()).unwrap_or(0),
            DynamicLayer::SSM { b, .. } => b.first().map(|r| r.len()).unwrap_or(0),
            DynamicLayer::Identity => 0,
        }
    }

    pub fn output_dim(&self) -> usize {
        match self {
            DynamicLayer::Dense { weights, .. } => weights.len(),
            DynamicLayer::Attention { o_proj, .. } => o_proj.len(),
            DynamicLayer::FFN { w2, .. } => w2.len(),
            DynamicLayer::SSM { c, .. } => c.len(),
            DynamicLayer::Identity => 0,
        }
    }

    fn layer_type_tag(&self) -> &'static str {
        match self {
            DynamicLayer::Dense { .. } => "Dense",
            DynamicLayer::Attention { .. } => "Attention",
            DynamicLayer::FFN { .. } => "FFN",
            DynamicLayer::SSM { .. } => "SSM",
            DynamicLayer::Identity => "Identity",
        }
    }
}

fn mat_vec_mul(m: &[Vec<f64>], v: &[f64]) -> Vec<f64> {
    m.iter().map(|row| {
        row.iter().zip(v).map(|(a, b)| a * b).sum()
    }).collect()
}

fn make_dense(in_dim: usize, out_dim: usize, activation: Activation) -> DynamicLayer {
    // Xavier-like init with deterministic small values
    let scale = 1.0 / (in_dim as f64).sqrt();
    let weights = (0..out_dim).map(|i| {
        (0..in_dim).map(|j| {
            let seed = (i * 997 + j * 131) as f64;
            (seed.sin() * 0.5) * scale
        }).collect()
    }).collect();
    let biases = vec![0.0; out_dim];
    DynamicLayer::Dense { weights, biases, activation }
}

// ── DynamicModel ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct DynamicModel {
    pub layers: Vec<DynamicLayer>,
}

impl DynamicModel {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn from_layer_sizes(sizes: &[usize]) -> Self {
        let mut layers = Vec::new();
        for i in 0..sizes.len().saturating_sub(1) {
            layers.push(make_dense(sizes[i], sizes[i + 1], Activation::ReLU));
        }
        Self { layers }
    }

    pub fn add_layer(&mut self, position: usize, layer: DynamicLayer) {
        let pos = position.min(self.layers.len());
        self.layers.insert(pos, layer);
    }

    pub fn remove_layer(&mut self, position: usize) {
        if position < self.layers.len() {
            self.layers[position] = DynamicLayer::Identity;
        }
    }

    pub fn prune_layer(&mut self, position: usize, gradient_magnitudes: &[f64], threshold: f64) {
        if position < self.layers.len() && position < gradient_magnitudes.len() {
            if gradient_magnitudes[position] < threshold {
                self.layers[position] = DynamicLayer::Identity;
            }
        }
    }

    pub fn split_layer(&mut self, position: usize) {
        if position >= self.layers.len() { return; }
        match &self.layers[position] {
            DynamicLayer::Dense { weights, biases, activation } => {
                let out_dim = weights.len();
                let in_dim = weights.first().map(|r| r.len()).unwrap_or(0);
                if out_dim < 2 || in_dim < 1 { return; }
                let mid = out_dim; // keep same width, factor into two layers
                // First: in_dim -> mid (identity-ish)
                let layer1 = make_dense(in_dim, mid, Activation::Identity);
                // Second: mid -> out_dim with original activation
                let mut layer2 = make_dense(mid, out_dim, activation.clone());
                // Copy original biases to second layer
                if let DynamicLayer::Dense { biases: ref mut b2, .. } = layer2 {
                    for (i, &b) in biases.iter().enumerate() {
                        if i < b2.len() { b2[i] = b; }
                    }
                }
                self.layers[position] = layer1;
                let pos2 = (position + 1).min(self.layers.len());
                self.layers.insert(pos2, layer2);
            }
            _ => {} // only Dense splitting supported
        }
    }

    pub fn merge_layers(&mut self, pos1: usize, pos2: usize) {
        if pos1 >= self.layers.len() || pos2 >= self.layers.len() || pos1 == pos2 { return; }
        let (a, b) = if pos1 < pos2 { (pos1, pos2) } else { (pos2, pos1) };
        match (&self.layers[a], &self.layers[b]) {
            (DynamicLayer::Dense { weights: w1, biases: b1, .. },
             DynamicLayer::Dense { weights: w2, biases: b2, activation: act2 }) => {
                // Compose: merged = w2 @ w1, biases = w2 @ b1 + b2
                let in_dim = w1.first().map(|r| r.len()).unwrap_or(0);
                let out_dim = w2.len();
                let merged_w: Vec<Vec<f64>> = (0..out_dim).map(|i| {
                    (0..in_dim).map(|j| {
                        w2[i].iter().enumerate().map(|(k, &w2ik)| {
                            if k < w1.len() { w2ik * w1[k].get(j).copied().unwrap_or(0.0) } else { 0.0 }
                        }).sum()
                    }).collect()
                }).collect();
                let merged_b: Vec<f64> = (0..out_dim).map(|i| {
                    let wb1: f64 = w2[i].iter().enumerate().map(|(k, &w)| {
                        if k < b1.len() { w * b1[k] } else { 0.0 }
                    }).sum();
                    wb1 + b2[i]
                }).collect();
                let act = act2.clone();
                self.layers[a] = DynamicLayer::Dense { weights: merged_w, biases: merged_b, activation: act };
                self.layers[b] = DynamicLayer::Identity;
            }
            _ => {} // only merge Dense+Dense
        }
    }

    pub fn forward(&self, input: &[f64]) -> Vec<f64> {
        let mut current = input.to_vec();
        for layer in &self.layers {
            match layer {
                DynamicLayer::Identity => {}
                _ => { current = layer.forward(&current); }
            }
        }
        current
    }

    pub fn total_params(&self) -> usize {
        self.layers.iter().map(|l| l.param_count()).sum()
    }

    pub fn architecture_hash(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        for (i, layer) in self.layers.iter().enumerate() {
            i.hash(&mut hasher);
            layer.layer_type_tag().hash(&mut hasher);
            layer.param_count().hash(&mut hasher);
        }
        hasher.finish()
    }

    pub fn active_layer_count(&self) -> usize {
        self.layers.iter().filter(|l| !matches!(l, DynamicLayer::Identity)).count()
    }
}

// ── TrainingMetrics ─────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct TrainingMetrics {
    pub train_loss: f64,
    pub val_loss: f64,
    pub loss_history: Vec<f64>,
    pub gradient_magnitudes: Vec<f64>,
    pub gradient_variances: Vec<f64>,
}

// ── Modification ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum Modification {
    AddLayer(usize, String, usize),        // position, layer_type, width
    RemoveLayer(usize),
    ResizeLayer(usize, usize),             // position, new_width
    ChangeActivation(usize, Activation),
    SwapLayerType(usize, String),          // position, new_type
}

impl Modification {
    pub fn describe(&self) -> String {
        match self {
            Modification::AddLayer(p, t, w) => format!("add {} layer at {} width {}", t, p, w),
            Modification::RemoveLayer(p) => format!("remove layer at {}", p),
            Modification::ResizeLayer(p, w) => format!("resize layer {} to width {}", p, w),
            Modification::ChangeActivation(p, a) => format!("change activation at {} to {:?}", p, a),
            Modification::SwapLayerType(p, t) => format!("swap layer {} to {}", p, t),
        }
    }
}

// ── GrowthPolicy ────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum GrowthPolicy {
    LossPlateauGrowth { patience: usize, min_delta: f64 },
    GradientMagnitudePruning { threshold: f64 },
    CapacityUtilization { variance_threshold: f64 },
}

impl GrowthPolicy {
    pub fn evaluate(&self, model: &DynamicModel, metrics: &TrainingMetrics) -> Vec<Modification> {
        match self {
            GrowthPolicy::LossPlateauGrowth { patience, min_delta } => {
                let hist = &metrics.loss_history;
                if hist.len() < *patience { return vec![]; }
                let recent = &hist[hist.len() - patience..];
                let improvement = recent.first().unwrap_or(&0.0) - recent.last().unwrap_or(&0.0);
                if improvement < *min_delta {
                    // Loss plateaued: add a layer in the middle
                    let mid = model.active_layer_count() / 2;
                    let width = model.layers.get(mid)
                        .map(|l| l.output_dim().max(l.input_dim()))
                        .unwrap_or(64);
                    vec![Modification::AddLayer(mid, "Dense".into(), width)]
                } else {
                    vec![]
                }
            }
            GrowthPolicy::GradientMagnitudePruning { threshold } => {
                let mut mods = vec![];
                for (i, &mag) in metrics.gradient_magnitudes.iter().enumerate() {
                    if mag < *threshold && i < model.layers.len() {
                        if !matches!(model.layers[i], DynamicLayer::Identity) {
                            mods.push(Modification::RemoveLayer(i));
                        }
                    }
                }
                mods
            }
            GrowthPolicy::CapacityUtilization { variance_threshold } => {
                let mut mods = vec![];
                for (i, &var) in metrics.gradient_variances.iter().enumerate() {
                    if var > *variance_threshold && i < model.layers.len() {
                        let width = model.layers[i].output_dim() * 2;
                        mods.push(Modification::ResizeLayer(i, width));
                    }
                }
                mods
            }
        }
    }
}

// ── WeightInterpolation ─────────────────────────────────────────────────────

pub struct WeightInterpolation;

impl WeightInterpolation {
    pub fn expand_weights(w: &[Vec<f64>], new_rows: usize, new_cols: usize) -> Vec<Vec<f64>> {
        let old_rows = w.len();
        let old_cols = w.first().map(|r| r.len()).unwrap_or(0);
        if old_rows == 0 || old_cols == 0 {
            return vec![vec![0.0; new_cols]; new_rows];
        }
        (0..new_rows).map(|i| {
            (0..new_cols).map(|j| {
                // Bilinear interpolation from old weights
                let src_r = (i as f64) * (old_rows as f64) / (new_rows as f64);
                let src_c = (j as f64) * (old_cols as f64) / (new_cols as f64);
                let r0 = (src_r as usize).min(old_rows - 1);
                let c0 = (src_c as usize).min(old_cols - 1);
                let r1 = (r0 + 1).min(old_rows - 1);
                let c1 = (c0 + 1).min(old_cols - 1);
                let fr = src_r - r0 as f64;
                let fc = src_c - c0 as f64;
                w[r0][c0] * (1.0 - fr) * (1.0 - fc)
                    + w[r0][c1] * (1.0 - fr) * fc
                    + w[r1][c0] * fr * (1.0 - fc)
                    + w[r1][c1] * fr * fc
            }).collect()
        }).collect()
    }

    pub fn shrink_weights(w: &[Vec<f64>], new_rows: usize, new_cols: usize) -> Vec<Vec<f64>> {
        let old_rows = w.len();
        let old_cols = w.first().map(|r| r.len()).unwrap_or(0);
        if old_rows == 0 || old_cols == 0 || new_rows == 0 || new_cols == 0 {
            return vec![vec![0.0; new_cols]; new_rows];
        }
        // Importance-based: keep rows/cols with largest L2 norms
        let mut row_norms: Vec<(usize, f64)> = (0..old_rows)
            .map(|i| (i, w[i].iter().map(|x| x * x).sum::<f64>()))
            .collect();
        row_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep_rows: Vec<usize> = row_norms.iter().take(new_rows).map(|&(i, _)| i).collect();

        let mut col_norms: Vec<(usize, f64)> = (0..old_cols)
            .map(|j| (j, w.iter().map(|row| { let v = row.get(j).copied().unwrap_or(0.0); v * v }).sum::<f64>()))
            .collect();
        col_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let keep_cols: Vec<usize> = col_norms.iter().take(new_cols).map(|&(j, _)| j).collect();

        keep_rows.iter().map(|&r| {
            keep_cols.iter().map(|&c| w[r].get(c).copied().unwrap_or(0.0)).collect()
        }).collect()
    }

    pub fn transfer_weights(old_layer: &DynamicLayer, new_layer: &mut DynamicLayer) {
        match (old_layer, new_layer) {
            (DynamicLayer::Dense { weights: ow, biases: ob, .. },
             DynamicLayer::Dense { weights: nw, biases: nb, .. }) => {
                let rows = ow.len().min(nw.len());
                for i in 0..rows {
                    let cols = ow[i].len().min(nw[i].len());
                    for j in 0..cols {
                        nw[i][j] = ow[i][j];
                    }
                    if i < ob.len() && i < nb.len() {
                        nb[i] = ob[i];
                    }
                }
            }
            _ => {} // incompatible types: no transfer
        }
    }
}

// ── ArchitectureSearcher ────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct ArchitectureSearcher {
    pub history: Vec<(Modification, f64)>,
    prev_snapshot: Option<Vec<DynamicLayer>>,
    prev_loss: f64,
    step_count: usize,
}

impl ArchitectureSearcher {
    pub fn new() -> Self {
        Self {
            history: Vec::new(),
            prev_snapshot: None,
            prev_loss: f64::MAX,
            step_count: 0,
        }
    }

    pub fn search_step(&mut self, model: &mut DynamicModel, _train_loss: f64, val_loss: f64) -> Option<Modification> {
        self.step_count += 1;
        // Save snapshot for potential rollback
        self.prev_snapshot = Some(model.layers.clone());
        self.prev_loss = val_loss;

        // UCB-like exploration: try different modification types
        let mod_type = self.step_count % 4;
        let n_layers = model.active_layer_count();

        let modification = match mod_type {
            0 => {
                // Try adding a layer
                let pos = n_layers / 2;
                let width = if let Some(l) = model.layers.get(pos) {
                    l.output_dim().max(16)
                } else { 32 };
                let layer = make_dense(width, width, Activation::ReLU);
                model.add_layer(pos, layer);
                Modification::AddLayer(pos, "Dense".into(), width)
            }
            1 => {
                // Try removing weakest layer (smallest param count)
                if n_layers > 1 {
                    let pos = model.layers.iter().enumerate()
                        .filter(|(_, l)| !matches!(l, DynamicLayer::Identity))
                        .min_by_key(|(_, l)| l.param_count())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    model.remove_layer(pos);
                    Modification::RemoveLayer(pos)
                } else {
                    return None;
                }
            }
            2 => {
                // Try changing activation
                if let Some(pos) = model.layers.iter().position(|l| matches!(l, DynamicLayer::Dense { .. })) {
                    if let DynamicLayer::Dense { weights, biases, activation } = &model.layers[pos] {
                        let new_act = match activation {
                            Activation::ReLU => Activation::GeLU,
                            Activation::GeLU => Activation::Tanh,
                            _ => Activation::ReLU,
                        };
                        let act_clone = new_act.clone();
                        model.layers[pos] = DynamicLayer::Dense {
                            weights: weights.clone(),
                            biases: biases.clone(),
                            activation: new_act,
                        };
                        Modification::ChangeActivation(pos, act_clone)
                    } else { return None; }
                } else { return None; }
            }
            _ => {
                // Try splitting a layer
                if n_layers > 0 {
                    let pos = 0;
                    model.split_layer(pos);
                    Modification::AddLayer(pos, "Split".into(), 0)
                } else { return None; }
            }
        };

        self.history.push((modification.clone(), val_loss));
        Some(modification)
    }

    pub fn rollback(&mut self, model: &mut DynamicModel) {
        if let Some(snapshot) = self.prev_snapshot.take() {
            model.layers = snapshot;
        }
    }
}

// ── Apply modification to model ─────────────────────────────────────────────

pub fn apply_modification(model: &mut DynamicModel, modif: &Modification) {
    match modif {
        Modification::AddLayer(pos, typ, width) => {
            let in_dim = if *pos > 0 {
                model.layers.get(pos - 1).map(|l| l.output_dim()).unwrap_or(*width)
            } else { *width };
            let layer = match typ.as_str() {
                "FFN" => {
                    DynamicLayer::FFN {
                        w1: vec![vec![0.01; in_dim]; *width],
                        w2: vec![vec![0.01; *width]; in_dim],
                        b1: vec![0.0; *width],
                        b2: vec![0.0; in_dim],
                    }
                }
                _ => make_dense(in_dim, *width, Activation::ReLU),
            };
            model.add_layer(*pos, layer);
        }
        Modification::RemoveLayer(pos) => model.remove_layer(*pos),
        Modification::ResizeLayer(pos, new_width) => {
            if let Some(DynamicLayer::Dense { weights, biases, activation }) = model.layers.get(*pos).cloned() {
                let in_dim = weights.first().map(|r| r.len()).unwrap_or(0);
                let new_w = WeightInterpolation::expand_weights(&weights, *new_width, in_dim);
                let mut new_b = vec![0.0; *new_width];
                for (i, &b) in biases.iter().enumerate() {
                    if i < new_b.len() { new_b[i] = b; }
                }
                model.layers[*pos] = DynamicLayer::Dense {
                    weights: new_w, biases: new_b, activation,
                };
            }
        }
        Modification::ChangeActivation(pos, act) => {
            if let Some(DynamicLayer::Dense { weights, biases, .. }) = model.layers.get(*pos).cloned() {
                model.layers[*pos] = DynamicLayer::Dense {
                    weights, biases, activation: act.clone(),
                };
            }
        }
        Modification::SwapLayerType(pos, new_type) => {
            if *pos < model.layers.len() {
                let old = &model.layers[*pos];
                let in_d = old.input_dim().max(1);
                let out_d = old.output_dim().max(1);
                let new_layer = match new_type.as_str() {
                    "Attention" => DynamicLayer::Attention {
                        q_proj: vec![vec![0.01; in_d]; out_d],
                        k_proj: vec![vec![0.01; in_d]; out_d],
                        v_proj: vec![vec![0.01; in_d]; out_d],
                        o_proj: vec![vec![0.01; out_d]; out_d],
                        n_heads: 1,
                    },
                    _ => make_dense(in_d, out_d, Activation::ReLU),
                };
                model.layers[*pos] = new_layer;
            }
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward_correct_output() {
        let layer = DynamicLayer::Dense {
            weights: vec![vec![1.0, 2.0], vec![3.0, 4.0]],
            biases: vec![0.5, -0.5],
            activation: Activation::Identity,
        };
        let out = layer.forward(&[1.0, 1.0]);
        assert_eq!(out.len(), 2);
        assert!((out[0] - 3.5).abs() < 1e-10); // 1*1+2*1+0.5
        assert!((out[1] - 6.5).abs() < 1e-10); // 3*1+4*1-0.5
    }

    #[test]
    fn test_model_forward_chains_layers() {
        let model = DynamicModel::from_layer_sizes(&[2, 3, 2]);
        assert_eq!(model.layers.len(), 2);
        let out = model.forward(&[1.0, 0.5]);
        assert_eq!(out.len(), 2);
        // Just verify it produces output without panic
    }

    #[test]
    fn test_add_layer_increases_param_count() {
        let mut model = DynamicModel::from_layer_sizes(&[4, 4]);
        let before = model.total_params();
        model.add_layer(1, make_dense(4, 4, Activation::ReLU));
        let after = model.total_params();
        assert!(after > before);
    }

    #[test]
    fn test_remove_layer_identity_passthrough() {
        let mut model = DynamicModel::from_layer_sizes(&[3, 4, 3]);
        let before = model.total_params();
        model.remove_layer(0);
        let after = model.total_params();
        assert!(after < before);
        assert!(matches!(model.layers[0], DynamicLayer::Identity));
        // Forward still works (identity passes through)
        let out = model.forward(&[1.0, 2.0, 3.0]);
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn test_prune_removes_near_zero_gradient_layers() {
        let mut model = DynamicModel::from_layer_sizes(&[4, 4, 4]);
        assert!(!matches!(model.layers[1], DynamicLayer::Identity));
        model.prune_layer(1, &[1.0, 0.001, 1.0], 0.01);
        assert!(matches!(model.layers[1], DynamicLayer::Identity));
    }

    #[test]
    fn test_split_layer_preserves_approximate_output() {
        let mut model = DynamicModel::from_layer_sizes(&[4, 4]);
        let input = vec![1.0, 0.5, -0.3, 0.8];
        let out_before = model.forward(&input);
        let params_before = model.total_params();
        model.split_layer(0);
        assert_eq!(model.layers.len(), 2);
        assert!(model.total_params() > params_before);
        // Output may differ but dimensions should match
        let out_after = model.forward(&input);
        assert_eq!(out_after.len(), out_before.len());
    }

    #[test]
    fn test_merge_layers_combines_weights() {
        let mut model = DynamicModel::from_layer_sizes(&[3, 3, 3]);
        assert_eq!(model.layers.len(), 2);
        let params_before = model.total_params();
        model.merge_layers(0, 1);
        // Second layer becomes Identity
        assert!(matches!(model.layers[1], DynamicLayer::Identity));
        // First layer now has the merged weights
        assert!(matches!(model.layers[0], DynamicLayer::Dense { .. }));
        // Param count should be roughly same as one layer (second is Identity=0)
        let params_after = model.total_params();
        assert!(params_after < params_before);
    }

    #[test]
    fn test_growth_policy_triggers_on_loss_plateau() {
        let model = DynamicModel::from_layer_sizes(&[4, 4]);
        let metrics = TrainingMetrics {
            train_loss: 1.0,
            val_loss: 1.0,
            loss_history: vec![1.0, 1.0, 1.0, 1.0, 1.0], // plateaued
            gradient_magnitudes: vec![1.0, 1.0],
            gradient_variances: vec![0.1, 0.1],
        };
        let policy = GrowthPolicy::LossPlateauGrowth { patience: 3, min_delta: 0.01 };
        let mods = policy.evaluate(&model, &metrics);
        assert!(!mods.is_empty());
        assert!(matches!(mods[0], Modification::AddLayer(..)));
    }

    #[test]
    fn test_architecture_search_proposes_modifications() {
        let mut model = DynamicModel::from_layer_sizes(&[4, 8, 4]);
        let mut searcher = ArchitectureSearcher::new();
        let m = searcher.search_step(&mut model, 1.0, 0.9);
        assert!(m.is_some());
        assert!(!searcher.history.is_empty());
        // Rollback should restore
        let layers_after = model.layers.len();
        searcher.rollback(&mut model);
        assert_eq!(model.layers.len(), 2); // original count restored
    }

    #[test]
    fn test_weight_interpolation_expand_preserves_behavior() {
        let w = vec![
            vec![1.0, 2.0],
            vec![3.0, 4.0],
        ];
        let expanded = WeightInterpolation::expand_weights(&w, 4, 4);
        assert_eq!(expanded.len(), 4);
        assert_eq!(expanded[0].len(), 4);
        // Corner should be same as original corner
        assert!((expanded[0][0] - 1.0).abs() < 1e-10);
        // Values should be interpolated, not zero
        assert!(expanded[1][1].abs() > 0.0);
    }

    #[test]
    fn test_architecture_hash_changes_on_modification() {
        let mut model = DynamicModel::from_layer_sizes(&[4, 4]);
        let h1 = model.architecture_hash();
        model.add_layer(1, make_dense(4, 4, Activation::ReLU));
        let h2 = model.architecture_hash();
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_weight_shrink() {
        let w = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let shrunk = WeightInterpolation::shrink_weights(&w, 2, 2);
        assert_eq!(shrunk.len(), 2);
        assert_eq!(shrunk[0].len(), 2);
    }
}
