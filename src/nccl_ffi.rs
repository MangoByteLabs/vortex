/// NCCL FFI bindings — compiled only when feature "nccl" is enabled.
/// When disabled, falls back to the simulated distributed runtime.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// 1. Raw FFI declarations (conditional)
// ---------------------------------------------------------------------------

#[cfg(feature = "nccl")]
mod nccl_bindings {
    extern "C" {
        pub fn ncclGetVersion(version: *mut i32) -> i32;
        pub fn ncclCommInitRank(
            comm: *mut usize,
            nranks: i32,
            id: usize,
            rank: i32,
        ) -> i32;
        pub fn ncclAllReduce(
            sendbuff: *const u8,
            recvbuff: *mut u8,
            count: usize,
            datatype: i32,
            op: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclBroadcast(
            sendbuff: *const u8,
            recvbuff: *mut u8,
            count: usize,
            datatype: i32,
            root: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclReduceScatter(
            sendbuff: *const u8,
            recvbuff: *mut u8,
            recvcount: usize,
            datatype: i32,
            op: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclAllGather(
            sendbuff: *const u8,
            recvbuff: *mut u8,
            sendcount: usize,
            datatype: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclSend(
            sendbuff: *const u8,
            count: usize,
            datatype: i32,
            peer: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclRecv(
            recvbuff: *mut u8,
            count: usize,
            datatype: i32,
            peer: i32,
            comm: usize,
            stream: usize,
        ) -> i32;
        pub fn ncclCommDestroy(comm: usize) -> i32;
    }
}

// ---------------------------------------------------------------------------
// 2. NCCL enums
// ---------------------------------------------------------------------------

/// NCCL data types matching ncclDataType_t.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NcclDataType {
    Int32 = 3,
    Int64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
}

/// NCCL reduction operations matching ncclRedOp_t.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(i32)]
pub enum NcclOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

// ---------------------------------------------------------------------------
// 3. Communicator abstraction
// ---------------------------------------------------------------------------

/// Unified communicator — real NCCL or simulated.
pub enum Communicator {
    #[cfg(feature = "nccl")]
    Nccl(NcclCommunicator),
    Simulated(SimulatedCommunicator),
}

#[cfg(feature = "nccl")]
pub struct NcclCommunicator {
    pub handle: usize,
    pub rank: usize,
    pub world_size: usize,
}

// ---------------------------------------------------------------------------
// 4. Simulated communicator
// ---------------------------------------------------------------------------

pub struct SimulatedCommunicator {
    pub world_size: usize,
    pub rank: usize,
    pub buffers: HashMap<String, Vec<Vec<f64>>>,
}

impl SimulatedCommunicator {
    /// Create a vector of communicators, one per rank.
    pub fn new(world_size: usize) -> Vec<Self> {
        (0..world_size)
            .map(|r| SimulatedCommunicator {
                world_size,
                rank: r,
                buffers: HashMap::new(),
            })
            .collect()
    }

    /// All-reduce sum: every rank ends up with the element-wise sum.
    pub fn all_reduce_sum(all_comms: &mut [SimulatedCommunicator], name: &str, data: &mut [Vec<f64>]) {
        let n = data[0].len();
        let world = all_comms.len();
        let mut sum = vec![0.0f64; n];
        for r in 0..world {
            for i in 0..n {
                sum[i] += data[r][i];
            }
        }
        for r in 0..world {
            data[r].copy_from_slice(&sum);
            all_comms[r].buffers.insert(name.to_string(), vec![sum.clone()]);
        }
    }

    /// All-reduce average.
    pub fn all_reduce_avg(all_comms: &mut [SimulatedCommunicator], name: &str, data: &mut [Vec<f64>]) {
        Self::all_reduce_sum(all_comms, name, data);
        let world = all_comms.len() as f64;
        for rank_data in data.iter_mut() {
            for v in rank_data.iter_mut() {
                *v /= world;
            }
        }
    }

    /// Reduce-scatter: reduce then each rank gets 1/N of the result.
    pub fn reduce_scatter(
        all_comms: &mut [SimulatedCommunicator],
        name: &str,
        send: &[Vec<f64>],
        recv: &mut [Vec<f64>],
    ) {
        let n = send[0].len();
        let world = all_comms.len();
        let chunk = n / world;
        let mut sum = vec![0.0f64; n];
        for r in 0..world {
            for i in 0..n {
                sum[i] += send[r][i];
            }
        }
        for r in 0..world {
            let start = r * chunk;
            let end = if r == world - 1 { n } else { start + chunk };
            let shard = sum[start..end].to_vec();
            recv[r] = shard.clone();
            all_comms[r].buffers.insert(name.to_string(), vec![shard]);
        }
    }

    /// All-gather: each rank contributes its chunk, all get the full tensor.
    pub fn all_gather(
        all_comms: &mut [SimulatedCommunicator],
        name: &str,
        send: &[Vec<f64>],
        recv: &mut [Vec<f64>],
    ) {
        let mut full = Vec::new();
        for s in send.iter() {
            full.extend_from_slice(s);
        }
        for r in 0..all_comms.len() {
            recv[r] = full.clone();
            all_comms[r].buffers.insert(name.to_string(), vec![full.clone()]);
        }
    }

    /// Broadcast data from root to all ranks.
    pub fn broadcast(
        all_comms: &mut [SimulatedCommunicator],
        name: &str,
        data: &mut [Vec<f64>],
        root: usize,
    ) {
        let root_data = data[root].clone();
        for r in 0..all_comms.len() {
            data[r] = root_data.clone();
            all_comms[r].buffers.insert(name.to_string(), vec![root_data.clone()]);
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Data-parallel trainer simulation
// ---------------------------------------------------------------------------

pub struct TrainingStepResult {
    pub params_consistent: bool,
    pub total_comm_bytes: usize,
    pub gradient_norm: f64,
}

pub struct DataParallelTrainer {
    pub world_size: usize,
    pub comms: Vec<SimulatedCommunicator>,
    pub model_params: Vec<Vec<f64>>,  // per-rank parameters
    pub gradients: Vec<Vec<f64>>,     // per-rank gradients
    pub lr: f64,
    step_count: usize,
}

impl DataParallelTrainer {
    pub fn new(world_size: usize, param_size: usize, lr: f64) -> Self {
        let comms = SimulatedCommunicator::new(world_size);
        // All ranks start with identical parameters
        let init_params = vec![1.0; param_size];
        let model_params = vec![init_params; world_size];
        let gradients = vec![vec![0.0; param_size]; world_size];
        DataParallelTrainer {
            world_size,
            comms,
            model_params,
            gradients,
            lr,
            step_count: 0,
        }
    }

    /// Simulate one training step:
    /// 1. Each rank computes different gradients (simulated with deterministic noise)
    /// 2. All-reduce to average gradients
    /// 3. Each rank applies identical update
    /// 4. Verify all ranks have identical parameters
    pub fn step(&mut self) -> TrainingStepResult {
        self.step_count += 1;
        let param_size = self.model_params[0].len();

        // 1. Simulate different gradients per rank
        for r in 0..self.world_size {
            for i in 0..param_size {
                // Gradient = param_value * decay + rank-specific noise
                let base = self.model_params[r][i] * 0.1;
                let noise = ((r * 7 + i * 13 + self.step_count * 3) % 100) as f64 * 0.001;
                self.gradients[r][i] = base + noise;
            }
        }

        // 2. All-reduce average the gradients
        SimulatedCommunicator::all_reduce_avg(&mut self.comms, "grad", &mut self.gradients);

        let comm_bytes = param_size * 8; // f64 bytes

        // Compute gradient norm
        let grad_norm: f64 = self.gradients[0]
            .iter()
            .map(|g| g * g)
            .sum::<f64>()
            .sqrt();

        // 3. Apply identical update on each rank
        for r in 0..self.world_size {
            for i in 0..param_size {
                self.model_params[r][i] -= self.lr * self.gradients[r][i];
            }
        }

        // 4. Check consistency
        let consistent = self.check_consistency();

        TrainingStepResult {
            params_consistent: consistent,
            total_comm_bytes: comm_bytes,
            gradient_norm: grad_norm,
        }
    }

    /// Verify parameter consistency across ranks.
    pub fn check_consistency(&self) -> bool {
        if self.world_size <= 1 {
            return true;
        }
        let ref_params = &self.model_params[0];
        for r in 1..self.world_size {
            for (i, v) in self.model_params[r].iter().enumerate() {
                if (v - ref_params[i]).abs() > 1e-12 {
                    return false;
                }
            }
        }
        true
    }
}

// ---------------------------------------------------------------------------
// 6. ZeRO trainer simulation
// ---------------------------------------------------------------------------

pub struct ZeROMemoryReport {
    pub no_zero_per_rank: usize,
    pub zero1_per_rank: usize,
    pub zero2_per_rank: usize,
    pub zero3_per_rank: usize,
}

pub struct ZeROTrainer {
    pub stage: u8,
    pub world_size: usize,
    pub comms: Vec<SimulatedCommunicator>,
    pub params: Vec<Vec<f64>>,
    pub gradients: Vec<Vec<f64>>,
    pub optimizer_states: Vec<Vec<f64>>,
    pub param_size: usize,
    step_count: usize,
}

impl ZeROTrainer {
    pub fn new(stage: u8, world_size: usize, param_size: usize) -> Self {
        assert!(stage >= 1 && stage <= 3);
        let comms = SimulatedCommunicator::new(world_size);

        // Initialize parameters (identical across ranks for stages 1-2, sharded for stage 3)
        let params = vec![vec![1.0; param_size]; world_size];
        let gradients = vec![vec![0.0; param_size]; world_size];

        // Optimizer states: 2x param size (momentum + variance for Adam)
        // Stage 1+: sharded across ranks
        let opt_shard_size = param_size * 2 / world_size;
        let optimizer_states = vec![vec![0.0; opt_shard_size]; world_size];

        ZeROTrainer {
            stage,
            world_size,
            comms,
            params,
            gradients,
            optimizer_states,
            param_size,
            step_count: 0,
        }
    }

    pub fn step(&mut self) -> TrainingStepResult {
        self.step_count += 1;
        let param_size = self.param_size;

        // Simulate gradients
        for r in 0..self.world_size {
            for i in 0..param_size {
                let base = self.params[r][i] * 0.1;
                let noise = ((r * 7 + i * 13 + self.step_count * 3) % 100) as f64 * 0.001;
                self.gradients[r][i] = base + noise;
            }
        }

        let comm_bytes;
        match self.stage {
            1 => {
                // Stage 1: all-reduce gradients, sharded optimizer states
                SimulatedCommunicator::all_reduce_avg(
                    &mut self.comms,
                    "grad",
                    &mut self.gradients,
                );
                comm_bytes = param_size * 8;
                // Apply update
                for r in 0..self.world_size {
                    for i in 0..param_size {
                        self.params[r][i] -= 0.01 * self.gradients[r][i];
                    }
                }
            }
            2 => {
                // Stage 2: reduce-scatter gradients (each rank gets 1/N), then all-gather params
                let mut recv = vec![vec![]; self.world_size];
                SimulatedCommunicator::reduce_scatter(
                    &mut self.comms,
                    "grad",
                    &self.gradients,
                    &mut recv,
                );
                // Each rank updates its shard of parameters
                let chunk = param_size / self.world_size;
                for r in 0..self.world_size {
                    let start = r * chunk;
                    for (i, g) in recv[r].iter().enumerate() {
                        self.params[r][start + i] -= 0.01 * g / self.world_size as f64;
                    }
                }
                // All-gather to synchronize full params
                let send: Vec<Vec<f64>> = (0..self.world_size)
                    .map(|r| {
                        let start = r * chunk;
                        let end = if r == self.world_size - 1 { param_size } else { start + chunk };
                        self.params[r][start..end].to_vec()
                    })
                    .collect();
                let mut recv_params = vec![vec![]; self.world_size];
                SimulatedCommunicator::all_gather(
                    &mut self.comms,
                    "params",
                    &send,
                    &mut recv_params,
                );
                for r in 0..self.world_size {
                    self.params[r] = recv_params[r].clone();
                }
                comm_bytes = param_size * 8 * 2; // reduce-scatter + all-gather
            }
            3 => {
                // Stage 3: everything sharded
                let mut recv = vec![vec![]; self.world_size];
                SimulatedCommunicator::reduce_scatter(
                    &mut self.comms,
                    "grad",
                    &self.gradients,
                    &mut recv,
                );
                let chunk = param_size / self.world_size;
                for r in 0..self.world_size {
                    let start = r * chunk;
                    for (i, g) in recv[r].iter().enumerate() {
                        self.params[r][start + i] -= 0.01 * g / self.world_size as f64;
                    }
                }
                let send: Vec<Vec<f64>> = (0..self.world_size)
                    .map(|r| {
                        let start = r * chunk;
                        let end = if r == self.world_size - 1 { param_size } else { start + chunk };
                        self.params[r][start..end].to_vec()
                    })
                    .collect();
                let mut recv_params = vec![vec![]; self.world_size];
                SimulatedCommunicator::all_gather(
                    &mut self.comms,
                    "params",
                    &send,
                    &mut recv_params,
                );
                for r in 0..self.world_size {
                    self.params[r] = recv_params[r].clone();
                }
                comm_bytes = param_size * 8 * 3; // more comm for full sharding
            }
            _ => unreachable!(),
        }

        let grad_norm: f64 = self.gradients[0]
            .iter()
            .map(|g| g * g)
            .sum::<f64>()
            .sqrt();

        let consistent = self.check_consistency();

        TrainingStepResult {
            params_consistent: consistent,
            total_comm_bytes: comm_bytes,
            gradient_norm: grad_norm,
        }
    }

    fn check_consistency(&self) -> bool {
        let ref_params = &self.params[0];
        for r in 1..self.world_size {
            for (i, v) in self.params[r].iter().enumerate() {
                if (v - ref_params[i]).abs() > 1e-10 {
                    return false;
                }
            }
        }
        true
    }

    /// Memory per rank in bytes (for f64 parameters).
    pub fn memory_per_rank(stage: u8, param_size: usize, world_size: usize) -> usize {
        let param_bytes = param_size * 8;
        let grad_bytes = param_size * 8;
        let opt_bytes = param_size * 2 * 8; // momentum + variance
        match stage {
            0 => param_bytes + grad_bytes + opt_bytes, // no ZeRO
            1 => param_bytes + grad_bytes + opt_bytes / world_size,
            2 => param_bytes + grad_bytes / world_size + opt_bytes / world_size,
            3 => param_bytes / world_size + grad_bytes / world_size + opt_bytes / world_size,
            _ => 0,
        }
    }

    /// Compare memory usage across ZeRO stages.
    pub fn memory_comparison(param_size: usize, world_size: usize) -> ZeROMemoryReport {
        ZeROMemoryReport {
            no_zero_per_rank: Self::memory_per_rank(0, param_size, world_size),
            zero1_per_rank: Self::memory_per_rank(1, param_size, world_size),
            zero2_per_rank: Self::memory_per_rank(2, param_size, world_size),
            zero3_per_rank: Self::memory_per_rank(3, param_size, world_size),
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Network bandwidth modeling
// ---------------------------------------------------------------------------

pub struct NetworkModel {
    pub intra_node_bandwidth_gbps: f64, // GB/s
    pub inter_node_bandwidth_gbps: f64, // GB/s
    pub intra_node_latency_us: f64,
    pub inter_node_latency_us: f64,
    pub gpus_per_node: usize,
}

impl NetworkModel {
    /// 8x H100 with NVLink + InfiniBand.
    pub fn dgx_h100() -> Self {
        NetworkModel {
            intra_node_bandwidth_gbps: 900.0,
            inter_node_bandwidth_gbps: 50.0,  // 400 Gb/s = 50 GB/s
            intra_node_latency_us: 1.0,
            inter_node_latency_us: 5.0,
            gpus_per_node: 8,
        }
    }

    /// 8x A100 with NVLink + InfiniBand.
    pub fn dgx_a100() -> Self {
        NetworkModel {
            intra_node_bandwidth_gbps: 600.0,
            inter_node_bandwidth_gbps: 25.0,  // 200 Gb/s
            intra_node_latency_us: 1.5,
            inter_node_latency_us: 5.0,
            gpus_per_node: 8,
        }
    }

    /// 2x RTX 4090 over PCIe.
    pub fn consumer() -> Self {
        NetworkModel {
            intra_node_bandwidth_gbps: 32.0, // PCIe 4.0 x16
            inter_node_bandwidth_gbps: 1.0,  // Ethernet
            intra_node_latency_us: 5.0,
            inter_node_latency_us: 100.0,
            gpus_per_node: 2,
        }
    }

    /// Estimate all-reduce time in microseconds using ring algorithm.
    /// Ring all-reduce: 2*(N-1)/N * msg_size / bandwidth + latency
    pub fn all_reduce_time(&self, bytes: usize, world_size: usize) -> f64 {
        if world_size <= 1 {
            return 0.0;
        }
        let n = world_size as f64;
        let num_nodes = (world_size + self.gpus_per_node - 1) / self.gpus_per_node;
        let (bw, lat) = if num_nodes > 1 {
            (self.inter_node_bandwidth_gbps, self.inter_node_latency_us)
        } else {
            (self.intra_node_bandwidth_gbps, self.intra_node_latency_us)
        };
        let msg = bytes as f64;
        // Ring all-reduce factor
        let factor = 2.0 * (n - 1.0) / n;
        let transfer_us = (factor * msg) / (bw * 1e3); // GB/s -> bytes/us = 1e9/1e6 = 1e3
        transfer_us + lat
    }

    /// Estimate reduce-scatter time in microseconds.
    /// Reduce-scatter: (N-1)/N * msg_size / bandwidth
    pub fn reduce_scatter_time(&self, bytes: usize, world_size: usize) -> f64 {
        if world_size <= 1 {
            return 0.0;
        }
        let n = world_size as f64;
        let num_nodes = (world_size + self.gpus_per_node - 1) / self.gpus_per_node;
        let (bw, lat) = if num_nodes > 1 {
            (self.inter_node_bandwidth_gbps, self.inter_node_latency_us)
        } else {
            (self.intra_node_bandwidth_gbps, self.intra_node_latency_us)
        };
        let msg = bytes as f64;
        let factor = (n - 1.0) / n;
        let transfer_us = (factor * msg) / (bw * 1e3);
        transfer_us + lat
    }

    /// Compute/communication overlap ratio: what fraction of comm can be hidden.
    pub fn overlap_ratio(&self, compute_time_us: f64, comm_time_us: f64) -> f64 {
        if comm_time_us <= 0.0 {
            return 1.0;
        }
        // If compute >= comm, 100% overlap; otherwise partial
        (compute_time_us / comm_time_us).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// 8. Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simulated_all_reduce_4_ranks() {
        let mut comms = SimulatedCommunicator::new(4);
        let mut data = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
            vec![10.0, 11.0, 12.0],
        ];
        SimulatedCommunicator::all_reduce_sum(&mut comms, "test", &mut data);
        let expected = vec![22.0, 26.0, 30.0];
        for r in 0..4 {
            assert_eq!(data[r], expected, "rank {} mismatch", r);
        }
    }

    #[test]
    fn test_data_parallel_params_consistent_after_step() {
        let mut trainer = DataParallelTrainer::new(4, 16, 0.01);
        let result = trainer.step();
        assert!(result.params_consistent, "params should be consistent after all-reduce");
    }

    #[test]
    fn test_data_parallel_5_steps_gradient_norm_decreases() {
        let mut trainer = DataParallelTrainer::new(4, 64, 0.1);
        let r0 = trainer.step();
        // Run 4 more steps
        #[allow(unused_assignments)]
        let mut last_result = r0;
        for _ in 0..4 {
            last_result = trainer.step();
            assert!(last_result.params_consistent);
        }
        // After 5 steps with lr=0.1, params should have decreased toward 0
        // so gradient norm (proportional to params) should decrease
        // We check that final params are smaller than initial (1.0)
        let final_mean: f64 = trainer.model_params[0].iter().sum::<f64>()
            / trainer.model_params[0].len() as f64;
        assert!(
            final_mean < 1.0,
            "params should decrease from initial 1.0, got {}",
            final_mean
        );
    }

    #[test]
    fn test_zero1_memory_less_than_no_zero() {
        let report = ZeROTrainer::memory_comparison(1_000_000, 8);
        assert!(
            report.zero1_per_rank < report.no_zero_per_rank,
            "ZeRO-1 ({}) should use less memory than no ZeRO ({})",
            report.zero1_per_rank,
            report.no_zero_per_rank
        );
    }

    #[test]
    fn test_zero3_less_than_zero2_less_than_zero1() {
        let report = ZeROTrainer::memory_comparison(1_000_000, 8);
        assert!(
            report.zero3_per_rank < report.zero2_per_rank,
            "ZeRO-3 ({}) < ZeRO-2 ({})",
            report.zero3_per_rank,
            report.zero2_per_rank
        );
        assert!(
            report.zero2_per_rank < report.zero1_per_rank,
            "ZeRO-2 ({}) < ZeRO-1 ({})",
            report.zero2_per_rank,
            report.zero1_per_rank
        );
    }

    #[test]
    fn test_zero_memory_report_7b_model() {
        // 7B params, 8 GPUs
        let param_size = 7_000_000_000usize;
        let world_size = 8;
        let report = ZeROTrainer::memory_comparison(param_size, world_size);

        // No ZeRO: params + grads + optimizer = 4x param bytes (params + grads + 2x opt)
        let expected_no_zero = param_size * 8 * 4; // f64
        assert_eq!(report.no_zero_per_rank, expected_no_zero);

        // ZeRO-1: optimizer sharded
        let expected_z1 = param_size * 8 + param_size * 8 + param_size * 2 * 8 / world_size;
        assert_eq!(report.zero1_per_rank, expected_z1);

        // ZeRO-3: everything sharded
        let expected_z3 = (param_size * 8 + param_size * 8 + param_size * 2 * 8) / world_size;
        assert_eq!(report.zero3_per_rank, expected_z3);
    }

    #[test]
    fn test_network_model_dgx_h100_faster_than_consumer() {
        let h100 = NetworkModel::dgx_h100();
        let consumer = NetworkModel::consumer();
        let bytes = 1_000_000_000; // 1 GB
        let world_size = 8;
        let t_h100 = h100.all_reduce_time(bytes, world_size);
        let t_consumer = consumer.all_reduce_time(bytes, world_size);
        assert!(
            t_h100 < t_consumer,
            "H100 ({:.1} us) should be faster than consumer ({:.1} us)",
            t_h100,
            t_consumer
        );
    }

    #[test]
    fn test_all_reduce_time_increases_with_message_size() {
        let model = NetworkModel::dgx_h100();
        let t_small = model.all_reduce_time(1_000_000, 8);    // 1 MB
        let t_large = model.all_reduce_time(1_000_000_000, 8); // 1 GB
        assert!(
            t_large > t_small,
            "larger message ({:.1} us) should take more time than smaller ({:.1} us)",
            t_large,
            t_small
        );
    }

    #[test]
    fn test_broadcast_all_ranks_same_data() {
        let mut comms = SimulatedCommunicator::new(4);
        let mut data = vec![
            vec![42.0, 99.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
            vec![0.0, 0.0],
        ];
        SimulatedCommunicator::broadcast(&mut comms, "bcast", &mut data, 0);
        for r in 0..4 {
            assert_eq!(data[r], vec![42.0, 99.0], "rank {} should have root's data", r);
        }
    }

    #[test]
    fn test_reduce_scatter_all_gather_equals_all_reduce() {
        let world_size = 4;
        let n = 8; // must be divisible by world_size

        let input: Vec<Vec<f64>> = (0..world_size)
            .map(|r| (0..n).map(|i| (r * n + i + 1) as f64).collect())
            .collect();

        // Path A: all-reduce sum
        let mut data_a = input.clone();
        let mut comms_a = SimulatedCommunicator::new(world_size);
        SimulatedCommunicator::all_reduce_sum(&mut comms_a, "x", &mut data_a);
        let expected = data_a[0].clone();

        // Path B: reduce-scatter + all-gather
        let mut comms_b = SimulatedCommunicator::new(world_size);
        let mut rs_recv = vec![vec![]; world_size];
        SimulatedCommunicator::reduce_scatter(&mut comms_b, "x", &input, &mut rs_recv);
        let mut ag_recv = vec![vec![]; world_size];
        SimulatedCommunicator::all_gather(&mut comms_b, "x", &rs_recv, &mut ag_recv);

        for r in 0..world_size {
            assert_eq!(
                ag_recv[r], expected,
                "rank {} reduce-scatter+all-gather should equal all-reduce",
                r
            );
        }
    }
}
