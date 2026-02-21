use std::collections::HashMap;

/// A tunable parameter
#[derive(Clone, Debug)]
pub struct TunableParam {
    pub name: String,
    pub values: Vec<usize>,
    pub default: usize,
}

/// A kernel configuration to tune
#[derive(Clone, Debug)]
pub struct TunableKernel {
    pub name: String,
    pub params: Vec<TunableParam>,
    pub constraints: Vec<Constraint>,
}

/// Constraints on parameter combinations
#[derive(Clone, Debug)]
pub enum Constraint {
    /// param_a * param_b <= limit
    ProductLe(String, String, usize),
    /// param_a must divide value
    Divides(String, usize),
    /// param_a <= param_b
    Le(String, String),
    /// shared memory: sum of tile sizes * element_size <= limit
    SharedMemLimit(Vec<String>, usize, usize),
}

/// Result of running one configuration
#[derive(Clone, Debug)]
pub struct BenchResult {
    pub config: HashMap<String, usize>,
    pub time_us: f64,
    pub gflops: f64,
    pub memory_bandwidth_gbps: f64,
    pub occupancy: f64,
}

/// Auto-tuning engine
pub struct AutoTuner {
    pub kernel: TunableKernel,
    pub results: Vec<BenchResult>,
    pub best: Option<BenchResult>,
    cache: HashMap<String, BenchResult>,
}

#[derive(Clone, Debug)]
pub struct ProblemSize {
    pub m: usize,
    pub n: usize,
    pub k: usize,
    pub batch: usize,
}

impl AutoTuner {
    pub fn new(kernel: TunableKernel) -> Self {
        Self {
            kernel,
            results: Vec::new(),
            best: None,
            cache: HashMap::new(),
        }
    }

    /// Generate all valid configurations (respecting constraints)
    pub fn generate_configs(&self) -> Vec<HashMap<String, usize>> {
        let params = &self.kernel.params;
        if params.is_empty() {
            return vec![HashMap::new()];
        }
        let mut configs: Vec<HashMap<String, usize>> = vec![HashMap::new()];
        for p in params {
            let mut next = Vec::new();
            for cfg in &configs {
                for &v in &p.values {
                    let mut c = cfg.clone();
                    c.insert(p.name.clone(), v);
                    next.push(c);
                }
            }
            configs = next;
        }
        configs.into_iter().filter(|c| self.is_valid(c)).collect()
    }

    /// Check if a config satisfies all constraints
    pub fn is_valid(&self, config: &HashMap<String, usize>) -> bool {
        for c in &self.kernel.constraints {
            match c {
                Constraint::ProductLe(a, b, limit) => {
                    if let (Some(&va), Some(&vb)) = (config.get(a), config.get(b)) {
                        if va * vb > *limit {
                            return false;
                        }
                    }
                }
                Constraint::Divides(param, value) => {
                    if let Some(&v) = config.get(param) {
                        if v == 0 || value % v != 0 {
                            return false;
                        }
                    }
                }
                Constraint::Le(a, b) => {
                    if let (Some(&va), Some(&vb)) = (config.get(a), config.get(b)) {
                        if va > vb {
                            return false;
                        }
                    }
                }
                Constraint::SharedMemLimit(params, elem_size, limit) => {
                    let mut total = 0usize;
                    for p in params {
                        if let Some(&v) = config.get(p) {
                            total += v;
                        }
                    }
                    // Model: shared mem = sum of tile dims * tile_k * elem_size (simplified)
                    // For matmul: (BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N) * elem_size
                    // Simplified: product of consecutive pairs
                    let vals: Vec<usize> = params.iter().filter_map(|p| config.get(p).copied()).collect();
                    if vals.len() >= 3 {
                        let shared = (vals[0] * vals[1] + vals[1] * vals[2]) * elem_size;
                        if shared > *limit {
                            return false;
                        }
                    } else {
                        let shared = total * *elem_size;
                        if shared > *limit {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// Simulate running a configuration using the performance model
    pub fn simulate(&self, config: &HashMap<String, usize>, problem: &ProblemSize) -> BenchResult {
        let model = PerformanceModel::a100();
        let time = model.estimate_time(&self.kernel.name, config, problem);
        let flops = 2.0 * problem.m as f64 * problem.n as f64 * problem.k as f64 * problem.batch.max(1) as f64;
        let gflops = flops / (time * 1e-6) / 1e9;
        let bytes = ((problem.m * problem.k + problem.k * problem.n + problem.m * problem.n) * 4) as f64;
        let bw = bytes / (time * 1e-6) / 1e9;

        // Estimate occupancy based on block size and warps
        let block_m = config.get("BLOCK_M").copied().unwrap_or(64);
        let block_n = config.get("BLOCK_N").copied().unwrap_or(64);
        let warps = config.get("NUM_WARPS").copied().unwrap_or(4);
        let threads_per_block = warps * 32;
        let blocks = ((problem.m + block_m - 1) / block_m) * ((problem.n + block_n - 1) / block_n);
        let active_threads = blocks * threads_per_block;
        let max_threads = model.num_sms * model.max_threads_per_sm;
        let occupancy = (active_threads as f64 / max_threads as f64).min(1.0);

        BenchResult {
            config: config.clone(),
            time_us: time,
            gflops,
            memory_bandwidth_gbps: bw,
            occupancy,
        }
    }

    /// Find best config by exhaustive search
    pub fn tune_exhaustive(&mut self, problem: &ProblemSize) -> &BenchResult {
        let configs = self.generate_configs();
        self.results.clear();
        self.best = None;
        for cfg in configs {
            let key = format!("{}_{:?}_{:?}", self.kernel.name, cfg, problem);
            let result = if let Some(cached) = self.cache.get(&key) {
                cached.clone()
            } else {
                let r = self.simulate(&cfg, problem);
                self.cache.insert(key, r.clone());
                r
            };
            if self.best.is_none() || result.time_us < self.best.as_ref().unwrap().time_us {
                self.best = Some(result.clone());
            }
            self.results.push(result);
        }
        self.best.as_ref().unwrap()
    }

    /// Find best config by random search
    pub fn tune_random(&mut self, problem: &ProblemSize, num_trials: usize) -> &BenchResult {
        let configs = self.generate_configs();
        if configs.is_empty() {
            panic!("No valid configurations");
        }
        self.results.clear();
        self.best = None;

        // Deterministic "random" selection using a simple LCG
        let mut rng_state: u64 = 42;
        for _ in 0..num_trials {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let idx = (rng_state >> 33) as usize % configs.len();
            let cfg = &configs[idx];
            let result = self.simulate(cfg, problem);
            if self.best.is_none() || result.time_us < self.best.as_ref().unwrap().time_us {
                self.best = Some(result.clone());
            }
            self.results.push(result);
        }
        self.best.as_ref().unwrap()
    }

    /// Speedup of best over default
    pub fn speedup(&self) -> f64 {
        if let Some(best) = &self.best {
            let mut default_cfg = HashMap::new();
            for p in &self.kernel.params {
                default_cfg.insert(p.name.clone(), p.default);
            }
            let default_time = self.simulate_with_default(&default_cfg);
            default_time / best.time_us
        } else {
            1.0
        }
    }

    fn simulate_with_default(&self, config: &HashMap<String, usize>) -> f64 {
        // Use first problem from results or a default
        let problem = ProblemSize { m: 1024, n: 1024, k: 1024, batch: 1 };
        let model = PerformanceModel::a100();
        model.estimate_time(&self.kernel.name, config, &problem)
    }

    /// Get the best configuration
    pub fn best_config(&self) -> Option<&HashMap<String, usize>> {
        self.best.as_ref().map(|b| &b.config)
    }
}

/// Performance model for simulation
pub struct PerformanceModel {
    pub peak_flops: f64,
    pub memory_bandwidth: f64,
    pub l2_cache_size: usize,
    pub shared_mem_size: usize,
    pub num_sms: usize,
    pub max_threads_per_sm: usize,
}

impl PerformanceModel {
    pub fn rtx_4090() -> Self {
        Self {
            peak_flops: 82.6,       // TFLOPS FP32
            memory_bandwidth: 1008.0, // GB/s
            l2_cache_size: 73 * 1024 * 1024,
            shared_mem_size: 49152,
            num_sms: 128,
            max_threads_per_sm: 1536,
        }
    }

    pub fn a100() -> Self {
        Self {
            peak_flops: 19.5,       // TFLOPS FP32
            memory_bandwidth: 2039.0, // GB/s
            l2_cache_size: 40 * 1024 * 1024,
            shared_mem_size: 49152,
            num_sms: 108,
            max_threads_per_sm: 2048,
        }
    }

    pub fn h100() -> Self {
        Self {
            peak_flops: 67.0,       // TFLOPS FP32
            memory_bandwidth: 3350.0, // GB/s
            l2_cache_size: 50 * 1024 * 1024,
            shared_mem_size: 49152 * 2, // 98KB configurable
            num_sms: 132,
            max_threads_per_sm: 2048,
        }
    }

    pub fn mi300x() -> Self {
        Self {
            peak_flops: 81.7,       // TFLOPS FP32
            memory_bandwidth: 5300.0, // GB/s
            l2_cache_size: 256 * 1024 * 1024,
            shared_mem_size: 65536,
            num_sms: 304,           // compute units
            max_threads_per_sm: 2048,
        }
    }

    /// Estimate execution time (in microseconds) for a config
    pub fn estimate_time(&self, kernel: &str, config: &HashMap<String, usize>, problem: &ProblemSize) -> f64 {
        let m = problem.m as f64;
        let n = problem.n as f64;
        let k = problem.k as f64;
        let batch = problem.batch.max(1) as f64;

        match kernel {
            "matmul" => {
                let block_m = config.get("BLOCK_M").copied().unwrap_or(64) as f64;
                let block_n = config.get("BLOCK_N").copied().unwrap_or(64) as f64;
                let block_k = config.get("BLOCK_K").copied().unwrap_or(16) as f64;
                let warps = config.get("NUM_WARPS").copied().unwrap_or(4) as f64;
                let stages = config.get("NUM_STAGES").copied().unwrap_or(3) as f64;

                let flops = 2.0 * m * n * k * batch;
                let compute_time = flops / (self.peak_flops * 1e12) * 1e6;

                // Memory: each block loads BLOCK_M*BLOCK_K + BLOCK_K*BLOCK_N elements
                // Number of blocks = ceil(M/BLOCK_M) * ceil(N/BLOCK_N)
                // Each block iterates ceil(K/BLOCK_K) times
                let num_blocks = (m / block_m).ceil() * (n / block_n).ceil() * batch;
                let k_iters = (k / block_k).ceil();
                let bytes_per_iter = (block_m * block_k + block_k * block_n) * 4.0;
                let total_bytes = num_blocks * k_iters * bytes_per_iter;

                // Pipeline stages hide some latency
                let pipeline_factor = 1.0 / stages.sqrt();
                let mem_time = (total_bytes / (self.memory_bandwidth * 1e9)) * 1e6 * pipeline_factor;

                // Parallelism efficiency
                let threads_per_block = warps * 32.0;
                let total_threads = num_blocks * threads_per_block;
                let max_concurrent = (self.num_sms as f64) * (self.max_threads_per_sm as f64);
                let waves = (total_threads / max_concurrent).ceil();
                let parallel_efficiency = if total_threads >= max_concurrent { 1.0 } else { total_threads / max_concurrent };

                // Larger tiles = fewer blocks = less overhead, but diminishing returns
                let tile_efficiency = 1.0 - 0.1 / (block_m * block_n / 1024.0).max(0.5);

                let raw_time = compute_time.max(mem_time);
                raw_time * waves / parallel_efficiency.max(0.1) / tile_efficiency.max(0.5)
            }
            "attention" => {
                let block_q = config.get("BLOCK_Q").copied().unwrap_or(64) as f64;
                let block_kv = config.get("BLOCK_KV").copied().unwrap_or(64) as f64;
                let head_dim = config.get("HEAD_DIM").copied().unwrap_or(64) as f64;

                let seq_len = m; // sequence length
                let num_heads = n;
                // QK^T: seq*head_dim x head_dim*seq = O(seq^2 * head_dim)
                let flops = 2.0 * seq_len * seq_len * head_dim * num_heads * batch + // QK^T
                            2.0 * seq_len * seq_len * head_dim * num_heads * batch;   // softmax(QK^T)V
                let compute_time = flops / (self.peak_flops * 1e12) * 1e6;

                let blocks = (seq_len / block_q).ceil() * num_heads * batch;
                let kv_iters = (seq_len / block_kv).ceil();
                let bytes_per_iter = (block_q * head_dim + block_kv * head_dim) * 4.0;
                let total_bytes = blocks * kv_iters * bytes_per_iter;
                let mem_time = (total_bytes / (self.memory_bandwidth * 1e9)) * 1e6;

                compute_time.max(mem_time) * 1.2 // 1.2x overhead for softmax
            }
            "layernorm" | "softmax" => {
                // Memory-bound kernels
                let elements = m * n * batch;
                let bytes = elements * 4.0 * 3.0; // read + write + intermediate
                let mem_time = (bytes / (self.memory_bandwidth * 1e9)) * 1e6;
                let block_size = config.get("BLOCK_SIZE").copied().unwrap_or(256) as f64;
                let efficiency = (block_size / 1024.0).min(1.0).max(0.3);
                mem_time / efficiency
            }
            _ => {
                // Generic fallback
                let flops = 2.0 * m * n * k * batch;
                let compute_time = flops / (self.peak_flops * 1e12) * 1e6;
                let bytes = (m * n + n * k + m * k) * 4.0 * batch;
                let mem_time = (bytes / (self.memory_bandwidth * 1e9)) * 1e6;
                compute_time.max(mem_time)
            }
        }
    }

    /// Roofline model analysis
    pub fn roofline_analysis(&self, flops: f64, bytes: f64) -> RooflineResult {
        let ai = if bytes > 0.0 { flops / bytes } else { f64::INFINITY };
        // Ridge point: peak_flops (TFLOP/s) / bandwidth (GB/s) = FLOP/byte
        let ridge = self.peak_flops * 1e3 / self.memory_bandwidth; // TFLOPS*1000 / (GB/s) = GFLOP / GB = FLOP/byte
        let is_compute = ai >= ridge;
        let is_memory = !is_compute;

        let achievable_flops = if is_compute {
            self.peak_flops * 1e12
        } else {
            ai * self.memory_bandwidth * 1e9
        };
        let achieved_fraction = (achievable_flops / (self.peak_flops * 1e12)).min(1.0);

        let bottleneck = if is_compute {
            "compute".to_string()
        } else {
            "memory bandwidth".to_string()
        };

        RooflineResult {
            arithmetic_intensity: ai,
            is_compute_bound: is_compute,
            is_memory_bound: is_memory,
            achieved_fraction,
            bottleneck,
        }
    }
}

#[derive(Clone, Debug)]
pub struct RooflineResult {
    pub arithmetic_intensity: f64,
    pub is_compute_bound: bool,
    pub is_memory_bound: bool,
    pub achieved_fraction: f64,
    pub bottleneck: String,
}

/// Predefined tunable kernels
pub fn matmul_kernel() -> TunableKernel {
    TunableKernel {
        name: "matmul".to_string(),
        params: vec![
            TunableParam { name: "BLOCK_M".to_string(), values: vec![16, 32, 64, 128], default: 64 },
            TunableParam { name: "BLOCK_N".to_string(), values: vec![16, 32, 64, 128], default: 64 },
            TunableParam { name: "BLOCK_K".to_string(), values: vec![8, 16, 32], default: 16 },
            TunableParam { name: "NUM_WARPS".to_string(), values: vec![2, 4, 8], default: 4 },
            TunableParam { name: "NUM_STAGES".to_string(), values: vec![2, 3, 4, 5], default: 3 },
        ],
        constraints: vec![
            Constraint::ProductLe("BLOCK_M".into(), "BLOCK_N".into(), 16384),
            Constraint::SharedMemLimit(
                vec!["BLOCK_M".into(), "BLOCK_K".into(), "BLOCK_N".into()],
                4, 49152,
            ),
        ],
    }
}

pub fn attention_kernel() -> TunableKernel {
    TunableKernel {
        name: "attention".to_string(),
        params: vec![
            TunableParam { name: "BLOCK_Q".to_string(), values: vec![16, 32, 64, 128], default: 64 },
            TunableParam { name: "BLOCK_KV".to_string(), values: vec![16, 32, 64, 128], default: 64 },
            TunableParam { name: "HEAD_DIM".to_string(), values: vec![32, 64, 128], default: 64 },
            TunableParam { name: "NUM_WARPS".to_string(), values: vec![2, 4, 8], default: 4 },
        ],
        constraints: vec![
            Constraint::ProductLe("BLOCK_Q".into(), "BLOCK_KV".into(), 16384),
        ],
    }
}

pub fn layernorm_kernel() -> TunableKernel {
    TunableKernel {
        name: "layernorm".to_string(),
        params: vec![
            TunableParam { name: "BLOCK_SIZE".to_string(), values: vec![64, 128, 256, 512, 1024], default: 256 },
            TunableParam { name: "NUM_WARPS".to_string(), values: vec![1, 2, 4, 8], default: 4 },
        ],
        constraints: vec![],
    }
}

pub fn softmax_kernel() -> TunableKernel {
    TunableKernel {
        name: "softmax".to_string(),
        params: vec![
            TunableParam { name: "BLOCK_SIZE".to_string(), values: vec![64, 128, 256, 512, 1024], default: 256 },
            TunableParam { name: "NUM_WARPS".to_string(), values: vec![1, 2, 4, 8], default: 4 },
        ],
        constraints: vec![],
    }
}

/// Kernel scheduling: overlap compute and memory transfers
pub struct KernelScheduler {
    pub ops: Vec<ScheduledOp>,
}

#[derive(Clone, Debug)]
pub struct ScheduledOp {
    pub name: String,
    pub stream: usize,
    pub start_us: f64,
    pub end_us: f64,
    pub depends_on: Vec<usize>,
}

impl KernelScheduler {
    pub fn new() -> Self {
        Self { ops: Vec::new() }
    }

    pub fn add_compute(&mut self, name: &str, duration_us: f64, deps: Vec<usize>) -> usize {
        let idx = self.ops.len();
        self.ops.push(ScheduledOp {
            name: name.to_string(),
            stream: 0, // compute stream
            start_us: 0.0,
            end_us: duration_us,
            depends_on: deps,
        });
        idx
    }

    pub fn add_transfer(&mut self, name: &str, bytes: usize, bandwidth_gbps: f64, deps: Vec<usize>) -> usize {
        let duration_us = (bytes as f64) / (bandwidth_gbps * 1e3); // bytes / (GB/s * 1e9) * 1e6
        let idx = self.ops.len();
        self.ops.push(ScheduledOp {
            name: name.to_string(),
            stream: 1, // transfer stream
            start_us: 0.0,
            end_us: duration_us,
            depends_on: deps,
        });
        idx
    }

    /// Schedule ops to minimize total time (overlap compute + transfers on different streams)
    pub fn schedule(&mut self) -> f64 {
        // Assign streams: even indices on stream 0, but actually use the pre-assigned streams
        // Topological scheduling: each op starts after all deps finish, and after
        // previous op on the same stream finishes.
        let n = self.ops.len();
        let mut stream_end: HashMap<usize, f64> = HashMap::new();

        for i in 0..n {
            let duration = self.ops[i].end_us - self.ops[i].start_us;
            // Earliest start: after all dependencies
            let mut earliest = 0.0f64;
            for &dep in &self.ops[i].depends_on.clone() {
                if dep < n {
                    earliest = earliest.max(self.ops[dep].end_us);
                }
            }
            // Also after previous op on same stream
            let stream = self.ops[i].stream;
            if let Some(&end) = stream_end.get(&stream) {
                earliest = earliest.max(end);
            }

            self.ops[i].start_us = earliest;
            self.ops[i].end_us = earliest + duration;
            stream_end.insert(stream, self.ops[i].end_us);
        }

        // Total time = max end time
        self.ops.iter().map(|op| op.end_us).fold(0.0f64, f64::max)
    }

    /// Print Gantt chart of schedule
    pub fn print_schedule(&self) -> String {
        let mut lines = Vec::new();
        let total = self.ops.iter().map(|op| op.end_us).fold(0.0f64, f64::max);
        lines.push(format!("Schedule (total: {:.1} us):", total));
        lines.push(format!("{:<20} {:>6} {:>10} {:>10} {:>10}", "Op", "Stream", "Start", "End", "Duration"));
        lines.push("-".repeat(60));
        for op in &self.ops {
            lines.push(format!(
                "{:<20} {:>6} {:>10.1} {:>10.1} {:>10.1}",
                op.name, op.stream, op.start_us, op.end_us, op.end_us - op.start_us
            ));
        }
        lines.join("\n")
    }

    /// Compute/transfer overlap ratio
    pub fn overlap_ratio(&self) -> f64 {
        // Find total time spans for compute (stream 0) and transfer (stream 1)
        let compute_ops: Vec<&ScheduledOp> = self.ops.iter().filter(|o| o.stream == 0).collect();
        let transfer_ops: Vec<&ScheduledOp> = self.ops.iter().filter(|o| o.stream == 1).collect();

        if compute_ops.is_empty() || transfer_ops.is_empty() {
            return 0.0;
        }

        // Calculate overlap between compute and transfer time ranges
        let mut overlap = 0.0f64;
        for c in &compute_ops {
            for t in &transfer_ops {
                let start = c.start_us.max(t.start_us);
                let end = c.end_us.min(t.end_us);
                if end > start {
                    overlap += end - start;
                }
            }
        }

        let total_compute: f64 = compute_ops.iter().map(|o| o.end_us - o.start_us).sum();
        let total_transfer: f64 = transfer_ops.iter().map(|o| o.end_us - o.start_us).sum();
        let shorter = total_compute.min(total_transfer);
        if shorter > 0.0 { overlap / shorter } else { 0.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_configs_valid_only() {
        let kernel = matmul_kernel();
        let tuner = AutoTuner::new(kernel);
        let configs = tuner.generate_configs();
        assert!(!configs.is_empty());
        for cfg in &configs {
            assert!(tuner.is_valid(cfg), "Generated invalid config: {:?}", cfg);
        }
    }

    #[test]
    fn test_invalid_config_rejected() {
        let kernel = matmul_kernel();
        let tuner = AutoTuner::new(kernel);
        let mut cfg = HashMap::new();
        cfg.insert("BLOCK_M".to_string(), 128);
        cfg.insert("BLOCK_N".to_string(), 128);
        cfg.insert("BLOCK_K".to_string(), 32);
        cfg.insert("NUM_WARPS".to_string(), 4);
        cfg.insert("NUM_STAGES".to_string(), 3);
        // 128 * 128 = 16384, this is exactly at limit so valid
        // But shared mem: (128*32 + 32*128)*4 = 32768, valid
        assert!(tuner.is_valid(&cfg));

        // Now make product exceed limit
        cfg.insert("BLOCK_M".to_string(), 256);
        cfg.insert("BLOCK_N".to_string(), 128);
        // 256 * 128 = 32768 > 16384
        assert!(!tuner.is_valid(&cfg));
    }

    #[test]
    fn test_exhaustive_finds_best() {
        let kernel = matmul_kernel();
        let mut tuner = AutoTuner::new(kernel);
        let problem = ProblemSize { m: 1024, n: 1024, k: 1024, batch: 1 };
        let best = tuner.tune_exhaustive(&problem).clone();

        // Verify it's actually the best
        for r in &tuner.results {
            assert!(best.time_us <= r.time_us + 1e-9, "Found result with lower time: {:?}", r);
        }
    }

    #[test]
    fn test_random_approaches_exhaustive() {
        let kernel = matmul_kernel();
        let problem = ProblemSize { m: 1024, n: 1024, k: 1024, batch: 1 };

        let mut exhaustive = AutoTuner::new(kernel.clone());
        let best_exhaustive = exhaustive.tune_exhaustive(&problem).clone();

        let mut random = AutoTuner::new(kernel);
        let num_configs = exhaustive.results.len();
        // With enough trials (2x total configs), random should find a good result
        let best_random = random.tune_random(&problem, num_configs * 2).clone();

        // Random should be within 2x of exhaustive
        assert!(
            best_random.time_us < best_exhaustive.time_us * 2.0,
            "Random ({:.2}) much worse than exhaustive ({:.2})",
            best_random.time_us,
            best_exhaustive.time_us
        );
    }

    #[test]
    fn test_roofline_small_matmul_memory_bound() {
        let model = PerformanceModel::a100();
        // Small matmul: 32x32x32 -> low arithmetic intensity
        let flops = 2.0 * 32.0 * 32.0 * 32.0;
        let bytes = (32.0 * 32.0 + 32.0 * 32.0 + 32.0 * 32.0) * 4.0;
        let result = model.roofline_analysis(flops, bytes);
        assert!(result.is_memory_bound, "Small matmul should be memory-bound, AI={:.2}", result.arithmetic_intensity);
    }

    #[test]
    fn test_roofline_large_matmul_compute_bound() {
        let model = PerformanceModel::a100();
        // Large matmul: 4096x4096x4096 -> high arithmetic intensity
        let m = 4096.0;
        let flops = 2.0 * m * m * m;
        let bytes = (m * m + m * m + m * m) * 4.0;
        let result = model.roofline_analysis(flops, bytes);
        assert!(result.is_compute_bound, "Large matmul should be compute-bound, AI={:.2}", result.arithmetic_intensity);
    }

    #[test]
    fn test_scheduler_overlap_reduces_time() {
        let mut sched = KernelScheduler::new();
        // Transfer then compute sequentially
        let t = sched.add_transfer("upload", 1_000_000, 12.8, vec![]);
        let _c = sched.add_compute("matmul", 100.0, vec![t]);

        // Also add an independent transfer that can overlap with compute
        let t2 = sched.add_transfer("prefetch_next", 1_000_000, 12.8, vec![]);
        let _c2 = sched.add_compute("matmul2", 100.0, vec![t2, _c]);

        let total = sched.schedule();

        // Sequential would be: upload + matmul + prefetch + matmul2
        let upload_time = 1_000_000.0 / (12.8 * 1e3);
        let sequential = upload_time * 2.0 + 200.0;
        assert!(
            total < sequential,
            "Overlapped ({:.1}) should be less than sequential ({:.1})",
            total, sequential
        );
    }

    #[test]
    fn test_scheduler_dependencies_respected() {
        let mut sched = KernelScheduler::new();
        let t = sched.add_transfer("upload", 500_000, 12.8, vec![]);
        let c = sched.add_compute("kernel", 50.0, vec![t]);
        sched.schedule();

        // Compute must start after transfer ends
        assert!(
            sched.ops[c].start_us >= sched.ops[t].end_us - 1e-9,
            "Compute started at {:.1} but transfer ends at {:.1}",
            sched.ops[c].start_us, sched.ops[t].end_us
        );
    }

    #[test]
    fn test_h100_faster_than_a100() {
        let a100 = PerformanceModel::a100();
        let h100 = PerformanceModel::h100();
        let problem = ProblemSize { m: 2048, n: 2048, k: 2048, batch: 1 };
        let mut cfg = HashMap::new();
        cfg.insert("BLOCK_M".to_string(), 64);
        cfg.insert("BLOCK_N".to_string(), 64);
        cfg.insert("BLOCK_K".to_string(), 16);
        cfg.insert("NUM_WARPS".to_string(), 4);
        cfg.insert("NUM_STAGES".to_string(), 3);

        let t_a100 = a100.estimate_time("matmul", &cfg, &problem);
        let t_h100 = h100.estimate_time("matmul", &cfg, &problem);
        assert!(
            t_h100 < t_a100,
            "H100 ({:.1}us) should be faster than A100 ({:.1}us)",
            t_h100, t_a100
        );
    }

    #[test]
    fn test_speedup_greater_than_one() {
        let kernel = matmul_kernel();
        let mut tuner = AutoTuner::new(kernel);
        let problem = ProblemSize { m: 1024, n: 1024, k: 1024, batch: 1 };
        tuner.tune_exhaustive(&problem);
        let speedup = tuner.speedup();
        assert!(speedup >= 1.0, "Speedup should be >= 1.0, got {:.2}", speedup);
    }

    #[test]
    fn test_gpu_presets_reasonable() {
        let presets = vec![
            ("RTX 4090", PerformanceModel::rtx_4090()),
            ("A100", PerformanceModel::a100()),
            ("H100", PerformanceModel::h100()),
            ("MI300X", PerformanceModel::mi300x()),
        ];
        for (name, model) in &presets {
            assert!(model.peak_flops > 10.0, "{} peak_flops too low: {}", name, model.peak_flops);
            assert!(model.peak_flops < 200.0, "{} peak_flops too high: {}", name, model.peak_flops);
            assert!(model.memory_bandwidth > 500.0, "{} bandwidth too low: {}", name, model.memory_bandwidth);
            assert!(model.memory_bandwidth < 10000.0, "{} bandwidth too high: {}", name, model.memory_bandwidth);
            assert!(model.num_sms > 50, "{} num_sms too low: {}", name, model.num_sms);
            assert!(model.shared_mem_size >= 49152, "{} shared_mem too small: {}", name, model.shared_mem_size);
            assert!(model.max_threads_per_sm >= 1024, "{} max_threads too low: {}", name, model.max_threads_per_sm);
            assert!(model.l2_cache_size >= 1024 * 1024, "{} L2 too small: {}", name, model.l2_cache_size);
        }
    }
}
