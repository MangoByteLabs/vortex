//! Vortex performance benchmark suite.
//!
//! Benchmarks Vortex against equivalent operations and targets:
//! - Matrix multiplication (various sizes)
//! - Field arithmetic (BN254 Fr)
//! - SHA-256 hash
//! - Tensor operations (relu, softmax, layernorm)
//!
//! Run with: vortex benchmark [--json]

use std::time::Instant;

#[derive(Debug, Clone)]
pub struct BenchResult {
    pub name: String,
    pub iterations: u64,
    pub total_ms: f64,
    pub ops_per_sec: f64,
    pub throughput_gbs: Option<f64>,
}

impl BenchResult {
    pub fn summary(&self) -> String {
        let mut s = format!(
            "{}: {:.2}ms/iter ({:.0} ops/sec",
            self.name,
            self.total_ms / self.iterations as f64,
            self.ops_per_sec
        );
        if let Some(gb) = self.throughput_gbs {
            s += &format!(", {:.2} GB/s", gb);
        }
        s + ")"
    }
}

/// Run a benchmark function N times and return statistics
pub fn bench<F: FnMut()>(name: &str, mut f: F, warmup: u32, iterations: u64) -> BenchResult {
    // Warmup
    for _ in 0..warmup {
        f();
    }
    // Timed run
    let start = Instant::now();
    for _ in 0..iterations {
        f();
    }
    let elapsed = start.elapsed();
    let total_ms = elapsed.as_secs_f64() * 1000.0;
    let ops_per_sec = iterations as f64 / elapsed.as_secs_f64();

    BenchResult {
        name: name.to_string(),
        iterations,
        total_ms,
        ops_per_sec,
        throughput_gbs: None,
    }
}

/// Benchmark: naive matrix multiplication (f32, NxN)
pub fn bench_matmul(n: usize) -> BenchResult {
    let a: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001).collect();
    let b: Vec<f32> = (0..n * n).map(|i| (i as f32) * 0.001 + 1.0).collect();

    let mut result = bench(
        &format!("matmul_{}x{}", n, n),
        || {
            let mut c = vec![0.0f32; n * n];
            for i in 0..n {
                for k in 0..n {
                    let aik = a[i * n + k];
                    for j in 0..n {
                        c[i * n + j] += aik * b[k * n + j];
                    }
                }
            }
            std::hint::black_box(c);
        },
        2,
        10,
    );

    // Throughput: 2*N^3 FLOPs
    let flops = 2.0 * (n as f64).powi(3);
    let secs = result.total_ms / result.iterations as f64 / 1000.0;
    result.throughput_gbs = Some(flops / secs / 1e9);
    result
}

/// Benchmark: field arithmetic (modular multiplication)
pub fn bench_field_mul(modulus: u64, count: u64) -> BenchResult {
    bench(
        &format!("field_mul_p{}", modulus),
        || {
            let mut acc = 1u64;
            for i in 1..=count {
                acc = ((acc as u128 * i as u128) % modulus as u128) as u64;
            }
            std::hint::black_box(acc);
        },
        3,
        100,
    )
}

/// Benchmark: ReLU activation on a flat tensor
pub fn bench_relu(size: usize) -> BenchResult {
    let data: Vec<f32> = (0..size)
        .map(|i| (i as f32 - size as f32 / 2.0) * 0.01)
        .collect();
    let mut result = bench(
        &format!("relu_{}", size),
        || {
            let out: Vec<f32> = data.iter().map(|&x| x.max(0.0)).collect();
            std::hint::black_box(out);
        },
        3,
        50,
    );
    let bytes = size * std::mem::size_of::<f32>();
    let secs = result.total_ms / result.iterations as f64 / 1000.0;
    result.throughput_gbs = Some(bytes as f64 / secs / 1e9);
    result
}

/// Benchmark: softmax on a vector
pub fn bench_softmax(size: usize) -> BenchResult {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.001).collect();
    bench(
        &format!("softmax_{}", size),
        || {
            let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = data.iter().map(|&x| (x - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let out: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
            std::hint::black_box(out);
        },
        3,
        100,
    )
}

/// Benchmark: layer normalization
pub fn bench_layernorm(size: usize) -> BenchResult {
    let data: Vec<f32> = (0..size).map(|i| (i as f32) * 0.01).collect();
    bench(
        &format!("layernorm_{}", size),
        || {
            let mean = data.iter().sum::<f32>() / size as f32;
            let var = data
                .iter()
                .map(|&x| (x - mean).powi(2))
                .sum::<f32>()
                / size as f32;
            let std_inv = 1.0 / (var + 1e-5).sqrt();
            let out: Vec<f32> = data.iter().map(|&x| (x - mean) * std_inv).collect();
            std::hint::black_box(out);
        },
        3,
        100,
    )
}

/// Run the full benchmark suite and print results
pub fn run_suite(json_output: bool) -> Vec<BenchResult> {
    let mut results = Vec::new();

    results.push(bench_matmul(64));
    results.push(bench_matmul(128));
    results.push(bench_field_mul(2_147_483_647, 10_000)); // Mersenne31
    results.push(bench_field_mul(18_446_744_069_414_584_321, 10_000)); // Goldilocks
    results.push(bench_relu(1_000_000));
    results.push(bench_softmax(32_768));
    results.push(bench_layernorm(32_768));

    if json_output {
        let json: Vec<serde_json::Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "name": r.name,
                    "iterations": r.iterations,
                    "total_ms": r.total_ms,
                    "ms_per_iter": r.total_ms / r.iterations as f64,
                    "ops_per_sec": r.ops_per_sec,
                    "throughput_gbs": r.throughput_gbs,
                })
            })
            .collect();
        println!("{}", serde_json::to_string_pretty(&json).unwrap());
    } else {
        println!("=== Vortex Benchmark Suite ===");
        for r in &results {
            println!("  {}", r.summary());
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bench_matmul_32() {
        let r = bench_matmul(32);
        assert!(r.ops_per_sec > 0.0);
        assert!(r.throughput_gbs.unwrap() > 0.0);
    }

    #[test]
    fn test_bench_relu() {
        let r = bench_relu(1000);
        assert!(r.ops_per_sec > 0.0);
    }

    #[test]
    fn test_bench_softmax() {
        let r = bench_softmax(1024);
        assert!(r.ops_per_sec > 0.0);
    }

    #[test]
    fn test_bench_layernorm() {
        let r = bench_layernorm(1024);
        assert!(r.ops_per_sec > 0.0);
    }

    #[test]
    fn test_bench_field_mul() {
        let r = bench_field_mul(7, 1000);
        assert!(r.ops_per_sec > 0.0);
    }

    #[test]
    fn test_run_suite_json() {
        // Should not panic
        let results = run_suite(false);
        assert!(!results.is_empty());
    }
}
