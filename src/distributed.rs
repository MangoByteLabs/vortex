use std::collections::HashMap;

/// Communication topology
#[derive(Clone, Debug)]
pub enum Topology {
    Ring,
    Tree,
    Butterfly,
    Mesh2D(usize, usize),
}

/// Parallelism strategy
#[derive(Clone, Debug)]
pub enum ParallelismStrategy {
    DataParallel { world_size: usize },
    TensorParallel { tp_size: usize, axis: usize },
    PipelineParallel { num_stages: usize, micro_batches: usize },
    ExpertParallel { num_experts: usize, experts_per_device: usize },
    ZeRO { stage: u8 },
    Hybrid(Vec<ParallelismStrategy>),
}

#[derive(Clone, Debug)]
struct DeviceState {
    rank: usize,
    buffers: HashMap<String, Vec<f64>>,
}

#[derive(Clone, Debug)]
pub struct CommEvent {
    pub op: CommOp,
    pub src: usize,
    pub dst: Option<usize>,
    pub bytes: usize,
    pub timestamp_us: u64,
}

#[derive(Clone, Debug)]
pub enum CommOp {
    AllReduce,
    ReduceScatter,
    AllGather,
    Send,
    Recv,
    Broadcast,
}

pub struct DistributedRuntime {
    pub world_size: usize,
    pub rank: usize,
    devices: Vec<DeviceState>,
    pub topology: Topology,
    pub strategy: ParallelismStrategy,
    comm_log: Vec<CommEvent>,
    simulated_time_us: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub enum PipelineOp {
    Forward(usize),
    Backward(usize),
    SendActivation(usize),
    RecvActivation(usize),
    SendGradient(usize),
    RecvGradient(usize),
    Idle,
}

pub struct PipelineSchedule {
    pub stages: Vec<Vec<PipelineOp>>,
    pub bubble_ratio: f64,
}

pub struct CommStats {
    pub total_bytes: usize,
    pub num_ops: usize,
    pub all_reduce_bytes: usize,
    pub p2p_bytes: usize,
}

impl DistributedRuntime {
    pub fn new(world_size: usize, strategy: ParallelismStrategy) -> Self {
        let devices = (0..world_size)
            .map(|r| DeviceState {
                rank: r,
                buffers: HashMap::new(),
            })
            .collect();
        DistributedRuntime {
            world_size,
            rank: 0,
            devices,
            topology: Topology::Ring,
            strategy,
            comm_log: Vec::new(),
            simulated_time_us: 0,
        }
    }

    /// Set buffer on a specific rank
    pub fn set_buffer(&mut self, rank: usize, name: &str, data: Vec<f64>) {
        self.devices[rank].buffers.insert(name.to_string(), data);
    }

    /// Get buffer from a specific rank
    pub fn get_buffer(&self, rank: usize, name: &str) -> Option<&Vec<f64>> {
        self.devices[rank].buffers.get(name)
    }

    /// All-reduce sum across all ranks
    pub fn all_reduce_sum(&mut self, name: &str) {
        let len = self.devices[0]
            .buffers
            .get(name)
            .map(|b| b.len())
            .unwrap_or(0);
        if len == 0 {
            return;
        }
        let mut sum = vec![0.0f64; len];
        for dev in &self.devices {
            if let Some(buf) = dev.buffers.get(name) {
                for (i, v) in buf.iter().enumerate() {
                    sum[i] += v;
                }
            }
        }
        let bytes = len * 8;
        self.simulated_time_us += 10;
        for dev in &mut self.devices {
            dev.buffers.insert(name.to_string(), sum.clone());
        }
        self.comm_log.push(CommEvent {
            op: CommOp::AllReduce,
            src: 0,
            dst: None,
            bytes,
            timestamp_us: self.simulated_time_us,
        });
    }

    /// All-reduce average across all ranks
    pub fn all_reduce_avg(&mut self, name: &str) {
        self.all_reduce_sum(name);
        let ws = self.world_size as f64;
        for dev in &mut self.devices {
            if let Some(buf) = dev.buffers.get_mut(name) {
                for v in buf.iter_mut() {
                    *v /= ws;
                }
            }
        }
    }

    /// Reduce-scatter: each rank gets 1/N of the reduced result
    pub fn reduce_scatter(&mut self, name: &str) {
        let len = self.devices[0]
            .buffers
            .get(name)
            .map(|b| b.len())
            .unwrap_or(0);
        if len == 0 {
            return;
        }
        // Compute full sum first
        let mut sum = vec![0.0f64; len];
        for dev in &self.devices {
            if let Some(buf) = dev.buffers.get(name) {
                for (i, v) in buf.iter().enumerate() {
                    sum[i] += v;
                }
            }
        }
        let chunk = len / self.world_size;
        let bytes = chunk * 8;
        self.simulated_time_us += 8;
        for (r, dev) in self.devices.iter_mut().enumerate() {
            let start = r * chunk;
            let end = if r == self.world_size - 1 { len } else { start + chunk };
            dev.buffers.insert(name.to_string(), sum[start..end].to_vec());
        }
        self.comm_log.push(CommEvent {
            op: CommOp::ReduceScatter,
            src: 0,
            dst: None,
            bytes,
            timestamp_us: self.simulated_time_us,
        });
    }

    /// All-gather: each rank broadcasts its shard, all get the full tensor
    pub fn all_gather(&mut self, name: &str) {
        let mut full = Vec::new();
        for dev in &self.devices {
            if let Some(buf) = dev.buffers.get(name) {
                full.extend_from_slice(buf);
            }
        }
        let bytes = full.len() * 8;
        self.simulated_time_us += 8;
        for dev in &mut self.devices {
            dev.buffers.insert(name.to_string(), full.clone());
        }
        self.comm_log.push(CommEvent {
            op: CommOp::AllGather,
            src: 0,
            dst: None,
            bytes,
            timestamp_us: self.simulated_time_us,
        });
    }

    /// Broadcast from src_rank to all
    pub fn broadcast(&mut self, name: &str, src_rank: usize) {
        let data = self.devices[src_rank]
            .buffers
            .get(name)
            .cloned()
            .unwrap_or_default();
        let bytes = data.len() * 8;
        self.simulated_time_us += 5;
        for dev in &mut self.devices {
            dev.buffers.insert(name.to_string(), data.clone());
        }
        self.comm_log.push(CommEvent {
            op: CommOp::Broadcast,
            src: src_rank,
            dst: None,
            bytes,
            timestamp_us: self.simulated_time_us,
        });
    }

    /// Shard a tensor across ranks along axis 0 (flat sharding)
    pub fn shard_tensor(&mut self, name: &str, data: &[f64], _axis: usize) {
        let chunk = data.len() / self.world_size;
        for (r, dev) in self.devices.iter_mut().enumerate() {
            let start = r * chunk;
            let end = if r == self.world_size - 1 {
                data.len()
            } else {
                start + chunk
            };
            dev.buffers.insert(name.to_string(), data[start..end].to_vec());
        }
    }

    /// Gather shards back into full tensor
    pub fn gather_tensor(&self, name: &str) -> Vec<f64> {
        let mut full = Vec::new();
        for dev in &self.devices {
            if let Some(buf) = dev.buffers.get(name) {
                full.extend_from_slice(buf);
            }
        }
        full
    }

    /// Pipeline send activations from one stage to another
    pub fn pipeline_send(&mut self, name: &str, from_stage: usize, to_stage: usize) {
        let data = self.devices[from_stage]
            .buffers
            .get(name)
            .cloned()
            .unwrap_or_default();
        let bytes = data.len() * 8;
        self.simulated_time_us += 3;
        self.devices[to_stage]
            .buffers
            .insert(name.to_string(), data);
        self.comm_log.push(CommEvent {
            op: CommOp::Send,
            src: from_stage,
            dst: Some(to_stage),
            bytes,
            timestamp_us: self.simulated_time_us,
        });
    }

    /// Simulate 1F1B pipeline schedule
    pub fn pipeline_1f1b_schedule(&mut self, num_microbatches: usize) -> PipelineSchedule {
        let num_stages = match &self.strategy {
            ParallelismStrategy::PipelineParallel { num_stages, .. } => *num_stages,
            _ => self.world_size,
        };

        // Each stage uses time slots. In 1F1B:
        // - Warmup: stage s does s idle slots then starts forwards
        // - Steady: alternating 1F1B
        // - Cooldown: remaining backwards + idle
        // Total time slots = num_stages - 1 + 2 * num_microbatches
        let total_slots = (num_stages - 1) + 2 * num_microbatches;
        let mut stages: Vec<Vec<PipelineOp>> = vec![Vec::new(); num_stages];

        for stage in 0..num_stages {
            let warmup_fwds = (num_stages - 1 - stage).min(num_microbatches);
            let mut fwd_done = 0usize;
            let mut bwd_done = 0usize;

            // Warmup idle slots for this stage
            for _ in 0..stage {
                stages[stage].push(PipelineOp::Idle);
            }

            // Warmup forwards
            for _ in 0..warmup_fwds {
                stages[stage].push(PipelineOp::Forward(fwd_done));
                fwd_done += 1;
            }

            // Steady state: 1F1B pairs
            let steady_pairs = num_microbatches.saturating_sub(warmup_fwds);
            for _ in 0..steady_pairs {
                stages[stage].push(PipelineOp::Forward(fwd_done));
                fwd_done += 1;
                stages[stage].push(PipelineOp::Backward(bwd_done));
                bwd_done += 1;
            }

            // Cooldown: remaining backwards
            while bwd_done < num_microbatches {
                stages[stage].push(PipelineOp::Backward(bwd_done));
                bwd_done += 1;
            }

            // Trailing idle to fill total_slots
            while stages[stage].len() < total_slots {
                stages[stage].push(PipelineOp::Idle);
            }
        }

        // Bubble ratio: idle slots / total slots
        let total_ops: usize = stages.iter().map(|s| s.len()).sum();
        let idle_ops: usize = stages
            .iter()
            .map(|s| s.iter().filter(|op| **op == PipelineOp::Idle).count())
            .sum();
        let bubble_ratio = if total_ops > 0 {
            idle_ops as f64 / total_ops as f64
        } else {
            0.0
        };

        PipelineSchedule {
            stages,
            bubble_ratio,
        }
    }

    /// Communication statistics
    pub fn comm_stats(&self) -> CommStats {
        let mut stats = CommStats {
            total_bytes: 0,
            num_ops: 0,
            all_reduce_bytes: 0,
            p2p_bytes: 0,
        };
        for ev in &self.comm_log {
            stats.total_bytes += ev.bytes;
            stats.num_ops += 1;
            match ev.op {
                CommOp::AllReduce => stats.all_reduce_bytes += ev.bytes,
                CommOp::Send | CommOp::Recv => stats.p2p_bytes += ev.bytes,
                _ => {}
            }
        }
        stats
    }

    /// Estimate time given bandwidth in GB/s
    pub fn estimate_time(&self, bandwidth_gbps: f64) -> f64 {
        let stats = self.comm_stats();
        let bytes = stats.total_bytes as f64;
        bytes / (bandwidth_gbps * 1e9)
    }
}

/// Auto-select parallelism strategy based on model size and hardware
pub fn auto_parallel(
    model_params: usize,
    num_gpus: usize,
    gpu_memory_gb: f64,
    _interconnect_bandwidth_gbps: f64,
) -> ParallelismStrategy {
    if num_gpus == 1 {
        return ParallelismStrategy::DataParallel { world_size: 1 };
    }

    // Rough estimate: 4 bytes per param for fp32, need ~3x for optimizer states
    let model_bytes = model_params as f64 * 4.0;
    let model_gb = model_bytes / 1e9;
    let memory_with_optimizer = model_gb * 3.0;

    if memory_with_optimizer <= gpu_memory_gb {
        // Model fits in one GPU with optimizer: pure data parallel
        ParallelismStrategy::DataParallel {
            world_size: num_gpus,
        }
    } else if model_gb <= gpu_memory_gb {
        // Model fits but optimizer doesn't: ZeRO stage 1
        ParallelismStrategy::ZeRO { stage: 1 }
    } else if model_gb <= gpu_memory_gb * num_gpus as f64 {
        // Model fits across GPUs: hybrid TP + DP or pipeline
        if num_gpus >= 8 {
            ParallelismStrategy::Hybrid(vec![
                ParallelismStrategy::TensorParallel {
                    tp_size: num_gpus.min(8),
                    axis: 0,
                },
                ParallelismStrategy::DataParallel {
                    world_size: num_gpus / num_gpus.min(8),
                },
            ])
        } else {
            ParallelismStrategy::PipelineParallel {
                num_stages: num_gpus,
                micro_batches: num_gpus * 2,
            }
        }
    } else {
        // Very large model: ZeRO-3 + pipeline
        ParallelismStrategy::Hybrid(vec![
            ParallelismStrategy::ZeRO { stage: 3 },
            ParallelismStrategy::PipelineParallel {
                num_stages: num_gpus,
                micro_batches: num_gpus * 2,
            },
        ])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_reduce_sum() {
        let mut rt = DistributedRuntime::new(
            4,
            ParallelismStrategy::DataParallel { world_size: 4 },
        );
        for r in 0..4 {
            rt.set_buffer(r, "grad", vec![1.0, 2.0, 3.0]);
        }
        rt.all_reduce_sum("grad");
        for r in 0..4 {
            assert_eq!(rt.get_buffer(r, "grad").unwrap(), &vec![4.0, 8.0, 12.0]);
        }
    }

    #[test]
    fn test_reduce_scatter_allgather_equivalence() {
        // reduce-scatter + all-gather should equal all-reduce
        let data = vec![1.0, 2.0, 3.0, 4.0];
        let world_size = 4;

        // Path A: all-reduce
        let mut rt_a = DistributedRuntime::new(
            world_size,
            ParallelismStrategy::DataParallel { world_size },
        );
        for r in 0..world_size {
            rt_a.set_buffer(r, "x", data.clone());
        }
        rt_a.all_reduce_sum("x");
        let expected = rt_a.get_buffer(0, "x").unwrap().clone();

        // Path B: reduce-scatter then all-gather
        let mut rt_b = DistributedRuntime::new(
            world_size,
            ParallelismStrategy::DataParallel { world_size },
        );
        for r in 0..world_size {
            rt_b.set_buffer(r, "x", data.clone());
        }
        rt_b.reduce_scatter("x");
        rt_b.all_gather("x");
        let result = rt_b.get_buffer(0, "x").unwrap().clone();

        assert_eq!(expected, result);
    }

    #[test]
    fn test_shard_gather_roundtrip() {
        let mut rt = DistributedRuntime::new(
            4,
            ParallelismStrategy::TensorParallel { tp_size: 4, axis: 0 },
        );
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        rt.shard_tensor("w", &original, 0);
        let gathered = rt.gather_tensor("w");
        assert_eq!(original, gathered);
    }

    #[test]
    fn test_pipeline_1f1b_ops_count() {
        let mut rt = DistributedRuntime::new(
            4,
            ParallelismStrategy::PipelineParallel {
                num_stages: 4,
                micro_batches: 8,
            },
        );
        let schedule = rt.pipeline_1f1b_schedule(8);
        assert_eq!(schedule.stages.len(), 4);

        // Each stage should have exactly num_microbatches forward and backward ops
        for stage_ops in &schedule.stages {
            let fwd = stage_ops
                .iter()
                .filter(|op| matches!(op, PipelineOp::Forward(_)))
                .count();
            let bwd = stage_ops
                .iter()
                .filter(|op| matches!(op, PipelineOp::Backward(_)))
                .count();
            assert_eq!(fwd, 8, "each stage must have 8 forward ops");
            assert_eq!(bwd, 8, "each stage must have 8 backward ops");
        }
    }

    #[test]
    fn test_pipeline_bubble_decreases() {
        let mut rt4 = DistributedRuntime::new(
            4,
            ParallelismStrategy::PipelineParallel {
                num_stages: 4,
                micro_batches: 4,
            },
        );
        let s4 = rt4.pipeline_1f1b_schedule(4);

        let mut rt8 = DistributedRuntime::new(
            4,
            ParallelismStrategy::PipelineParallel {
                num_stages: 4,
                micro_batches: 8,
            },
        );
        let s8 = rt8.pipeline_1f1b_schedule(8);

        assert!(
            s8.bubble_ratio <= s4.bubble_ratio,
            "more microbatches ({}) should reduce bubble ratio: {} vs {}",
            8,
            s8.bubble_ratio,
            s4.bubble_ratio,
        );
    }

    #[test]
    fn test_zero_stage1_optimizer_sharded() {
        // ZeRO stage 1: optimizer state sharded via reduce-scatter
        let mut rt = DistributedRuntime::new(4, ParallelismStrategy::ZeRO { stage: 1 });
        // Simulate gradient all-reduce then optimizer state sharding
        for r in 0..4 {
            rt.set_buffer(r, "opt_state", vec![10.0, 20.0, 30.0, 40.0]);
        }
        rt.reduce_scatter("opt_state");
        // Each rank should hold only 1/4 of the state
        for r in 0..4 {
            let buf = rt.get_buffer(r, "opt_state").unwrap();
            assert_eq!(buf.len(), 1, "rank {} should hold 1 element shard", r);
        }
        // Values should be summed: 10*4=40, 20*4=80, etc.
        assert_eq!(rt.get_buffer(0, "opt_state").unwrap(), &vec![40.0]);
        assert_eq!(rt.get_buffer(1, "opt_state").unwrap(), &vec![80.0]);
        assert_eq!(rt.get_buffer(2, "opt_state").unwrap(), &vec![120.0]);
        assert_eq!(rt.get_buffer(3, "opt_state").unwrap(), &vec![160.0]);
    }

    #[test]
    fn test_auto_parallel_single_gpu() {
        let strat = auto_parallel(1_000_000, 1, 16.0, 100.0);
        match strat {
            ParallelismStrategy::DataParallel { world_size: 1 } => {}
            other => panic!("expected DataParallel(1), got {:?}", other),
        }
    }

    #[test]
    fn test_auto_parallel_8gpu_large_model() {
        // 70B params, 8 GPUs with 80GB each
        let strat = auto_parallel(70_000_000_000, 8, 80.0, 400.0);
        match strat {
            ParallelismStrategy::Hybrid(_) => {}
            other => panic!("expected Hybrid for large model, got {:?}", other),
        }
    }

    #[test]
    fn test_comm_stats() {
        let mut rt = DistributedRuntime::new(
            4,
            ParallelismStrategy::DataParallel { world_size: 4 },
        );
        for r in 0..4 {
            rt.set_buffer(r, "grad", vec![1.0; 100]);
        }
        rt.all_reduce_sum("grad");
        let stats = rt.comm_stats();
        assert_eq!(stats.num_ops, 1);
        assert_eq!(stats.all_reduce_bytes, 100 * 8); // 100 f64s
        assert_eq!(stats.total_bytes, 100 * 8);
        assert_eq!(stats.p2p_bytes, 0);
    }

    #[test]
    fn test_tensor_parallel_matmul() {
        // Full matmul: [1,2,3,4] * [[1,0],[0,1],[1,0],[0,1]] = [4, 6]
        // Simulated: shard weight columns across 2 GPUs
        // GPU0 gets column 0 weights: [1,0,1,0], GPU1 gets column 1: [0,1,0,1]
        let mut rt = DistributedRuntime::new(
            2,
            ParallelismStrategy::TensorParallel { tp_size: 2, axis: 1 },
        );
        let input = vec![1.0, 2.0, 3.0, 4.0];

        // Shard weights: GPU0=[1,0,1,0], GPU1=[0,1,0,1]
        rt.set_buffer(0, "input", input.clone());
        rt.set_buffer(1, "input", input.clone());
        rt.set_buffer(0, "weight", vec![1.0, 0.0, 1.0, 0.0]);
        rt.set_buffer(1, "weight", vec![0.0, 1.0, 0.0, 1.0]);

        // Each GPU computes dot product of input with its weight column
        for r in 0..2 {
            let inp = rt.get_buffer(r, "input").unwrap().clone();
            let w = rt.get_buffer(r, "weight").unwrap().clone();
            let dot: f64 = inp.iter().zip(w.iter()).map(|(a, b)| a * b).sum();
            rt.set_buffer(r, "result", vec![dot]);
        }

        // Gather partial results
        let full = rt.gather_tensor("result");
        assert_eq!(full, vec![4.0, 6.0]);
    }
}
