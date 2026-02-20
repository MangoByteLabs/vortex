# 08 — Tooling & Developer Experience

## 1. Build System

### `vortex` CLI

```bash
# Compile a single file
vortex build main.vx --target sm_90

# Build a project (reads vortex.toml)
vortex build --release

# Build for multiple targets
vortex build --target sm_90,gfx942

# JIT compile and run
vortex run benchmark.vx -- --size 4096

# Auto-tune kernels for current hardware
vortex tune main.vx --iterations 100

# Format source code
vortex fmt

# Check types without compiling
vortex check main.vx
```

### Project File (vortex.toml)

```toml
[package]
name = "zk-prover"
version = "0.1.0"

[dependencies]
std-crypto = { version = "0.1", features = ["bn254", "bls12-381"] }
std-nn = { version = "0.1", features = ["attention", "quantize"] }

[build]
default-target = "sm_90"
optimization = "release"
auto-tune = true

[tune]
cache-dir = ".vortex/tune_cache"
benchmark-iterations = 50

[test]
gpu-required = true
```

---

## 2. Language Server Protocol (LSP)

### Features

- **Type hints**: Show inferred tensor shapes and memory spaces on hover
- **Shape error diagnostics**: "Expected Tensor<f16, [4096, 4096]>, got Tensor<f16, [4096, 2048]>"
- **Memory budget display**: Show shared memory / register usage inline
- **Go-to-definition**: Navigate to trait impls, field type definitions
- **Auto-complete**: Context-aware completion for kernel parameters, schedule options
- **Constant-time warnings**: Inline warnings for potential timing side channels

### Inline Annotations (Editor)

```vortex
kernel matmul(
    a: Tensor<f16, [M, K]>,        // hover: M=4096, K=4096 (from call site)
    b: Tensor<f16, [K, N]>,        // hover: K=4096, N=4096
) -> Tensor<f16, [M, N]> {        // hover: output shape [4096, 4096]
    // ▸ Shared memory: 32 KB / 228 KB
    // ▸ Registers: 96 / 255 per thread
    // ▸ Occupancy: 50%
    ...
}
```

---

## 3. Debugger

### GPU Kernel Debugging

```bash
# Debug mode: insert runtime checks, disable optimizations
vortex build --debug

# Run under debugger
vortex debug main.vx
```

### Features

- **Source-level breakpoints** in kernel code (mapped to SASS via debug info)
- **Tensor inspection**: Pretty-print tensor contents at breakpoints
- **Thread selector**: Inspect state of specific (block_id, thread_id)
- **Warp divergence visualization**: Highlight which threads are active/masked
- **Shared memory viewer**: See contents of shared memory for current block
- **Printf-style debugging** that doesn't kill performance:

```vortex
kernel debug_example(data: Tensor<f32, [N]>) {
    let val = data[thread.global_id]
    debug_print("thread {} val = {}", thread.global_id, val)
    // Compiles to: GPU printf with throttling (max 1 print per warp)
    // In release mode: stripped entirely (zero overhead)
}
```

### Simulator Mode

```bash
# Run kernel on CPU with full debugging (slow but complete)
vortex run --simulate kernel.vx
```

Executes the kernel on CPU, simulating the GPU threading model. Useful for:
- Verifying correctness before GPU execution
- Detecting race conditions via thread sanitizer
- Testing on machines without a GPU

---

## 4. Profiler Integration

### Built-in Profiling

```bash
# Profile a kernel execution
vortex profile main.vx --target sm_90

  ┌─────────────────────────────────────────────────────────┐
  │ Kernel: matmul_f16_4096x4096                            │
  │ Time: 1.23 ms                                           │
  │ TFLOPS: 223 (71% of peak)                               │
  │                                                         │
  │ Roofline Analysis:                                      │
  │   Arithmetic Intensity: 64 FLOPs/byte                   │
  │   Bound: Compute (machine balance: 295 FLOPs/byte)      │
  │                                                         │
  │ Memory:                                                 │
  │   HBM Read:  134 MB  (1.1 TB/s effective, 33% of peak) │
  │   HBM Write:  33 MB  (0.27 TB/s)                       │
  │   L2 Hit Rate: 87%                                      │
  │   Shared Memory Bank Conflicts: 0                       │
  │                                                         │
  │ Occupancy:                                              │
  │   Achieved: 48% (target: 50%)                           │
  │   Limiter: Registers (96 per thread)                    │
  │                                                         │
  │ Bottleneck: Instruction issue (ALU pipeline full)       │
  │ Suggestion: Increase tile_k to hide memory latency      │
  └─────────────────────────────────────────────────────────┘
```

### NSight Integration

```bash
# Generate NSight Compute report
vortex profile --nsight main.vx --output report.ncu-rep

# Generate NSight Systems timeline
vortex profile --timeline main.vx --output trace.nsys-rep
```

### Source-Level Annotations

The profiler maps performance data back to source lines:

```vortex
kernel attention(...) {
    let scores = q_tile @ k_tile.transpose()   // ◀ 45% of time, 720 TFLOPS
    let weights = softmax(scores)               // ◀ 12% of time (memory-bound)
    let output = weights @ v_tile               // ◀ 38% of time, 680 TFLOPS
    store(output, ...)                          // ◀ 5% of time
}
```

---

## 5. Package Manager

### Registry

```bash
# Publish a package
vortex publish

# Install a dependency
vortex add crypto-extras@0.2

# Search packages
vortex search "zk prover"
```

### Package Structure

```
my-package/
├── vortex.toml        # Package manifest
├── src/
│   ├── lib.vx         # Library root
│   └── kernels/
│       ├── ntt.vx
│       └── msm.vx
├── tests/
│   ├── test_ntt.vx
│   └── test_msm.vx
├── benches/
│   └── bench_ntt.vx
└── examples/
    └── groth16.vx
```

---

## 6. Testing Framework

```vortex
import std.test { test, assert_eq, assert_close, assert_tensor_close }

#[test]
fn test_field_mul() {
    let a: BN254_Fr = BN254_Fr.from(7)
    let b: BN254_Fr = BN254_Fr.from(6)
    assert_eq(a * b, BN254_Fr.from(42))
}

#[test]
fn test_ntt_roundtrip() {
    let poly = random_poly<BN254_Fr, 1024>()
    let ntt_poly = ntt_forward(poly)
    let recovered = ntt_inverse(ntt_poly)
    assert_tensor_close(poly, recovered, tolerance = 0)  // exact for fields
}

#[test]
fn test_attention_matches_naive() {
    let q = random_tensor<f16, [1, 8, 128, 64]>()
    let k = random_tensor<f16, [1, 8, 128, 64]>()
    let v = random_tensor<f16, [1, 8, 128, 64]>()

    let flash_out = flash_attention(q, k, v, causal = false)
    let naive_out = naive_attention(q, k, v)
    assert_tensor_close(flash_out, naive_out, rtol = 1e-2, atol = 1e-3)
}

#[test]
#[constant_time_test]  // Verify timing independence from secret values
fn test_scalar_mul_constant_time() {
    let point = BN254_G1.generator()
    // Run with multiple secret values, verify identical execution trace
    for _ in 0..100 {
        let scalar = random_secret_scalar()
        let _ = scalar_mul(scalar, point)
    }
    // Framework checks: cycle count variance < threshold
}
```

```bash
$ vortex test
  Running 42 tests on GPU (RTX 4090)...

  test_field_mul ................................. ✓ (0.1 ms)
  test_ntt_roundtrip ............................. ✓ (2.3 ms)
  test_attention_matches_naive ................... ✓ (15.7 ms)
  test_scalar_mul_constant_time .................. ✓ (timing variance: 0.02%)
  ...

  42/42 passed (230 ms total)
```
