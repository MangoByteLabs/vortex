# 00 — Design Philosophy

## Core Principles

### 1. Parallel by Default

In Vortex, every function is potentially a GPU kernel. The programmer thinks in terms
of data-parallel operations, not sequential loops. The compiler decides how to map work
to the GPU's execution hierarchy (grids → blocks → warps → threads).

**Anti-pattern we're solving:** In CUDA, you write serial C++ and manually decompose it
into threads. In Vortex, you express the algorithm and the compiler handles decomposition.

```vortex
// CUDA: you manually compute thread indices
// __global__ void add(float* a, float* b, float* c, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if (i < n) c[i] = a[i] + b[i];
// }

// Vortex: express the operation, not the threading
kernel add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> {
    return a + b  // compiler generates optimal grid/block config
}
```

### 2. Safety Without Sacrifice

Vortex provides strong safety guarantees — memory safety, type safety, side-channel
resistance — without sacrificing the low-level control that GPU programmers need.

- **Memory safety** via ownership and borrowing (Rust-inspired, adapted for GPU)
- **Type safety** via dependent types for tensor shapes and modular arithmetic
- **Timing safety** via `@constant_time` blocks with compiler verification
- **No undefined behavior** — every operation has defined semantics

### 3. Abstraction Without Penalty

High-level abstractions should compile to code as efficient as hand-written CUDA/PTX.
The compiler aggressively optimizes through abstractions.

- Generic functions monomorphize at compile time (zero runtime cost)
- High-level tensor operations lower to tiled, vectorized GPU kernels
- The `@schedule` system lets experts override compiler decisions without changing the algorithm

### 4. Two Domains, One Language

Cryptography and ML share fundamental computational patterns:
- Both are dominated by **large matrix/polynomial operations**
- Both need **modular arithmetic** (finite fields in crypto, quantized arithmetic in ML)
- Both benefit from **NTT/FFT** (polynomial multiplication in crypto, convolution in ML)
- Both are **embarrassingly parallel**

Vortex unifies these domains instead of treating them as separate ecosystems.

### 5. Progressive Disclosure of Complexity

A researcher should be able to write high-level Vortex and get 80% of peak performance.
A kernel engineer should be able to drop down to PTX-level control when needed.

```
Level 1: Algorithm      →  tensor ops, auto-scheduling
Level 2: Schedule       →  explicit tiling, memory placement, fusion hints
Level 3: Intrinsics     →  warp shuffles, tensor core MMA, async copies
Level 4: Inline ASM     →  raw PTX/AMDGCN for the last 5%
```

## What Vortex Is NOT

- **Not a general-purpose language** — it targets GPU-heavy compute. Host-side orchestration
  can use Rust/Python/C++ with Vortex FFI bindings.
- **Not a DSL embedded in Python** — it's a standalone compiled language with its own toolchain.
  (Though we may provide Python bindings for ergonomics.)
- **Not a CUDA wrapper** — it targets multiple GPU backends natively via MLIR.
- **Not an auto-parallelizing compiler for sequential code** — you write parallel algorithms,
  and the compiler optimizes the mapping to hardware.

## Design Influences

| Language/System | What We Take |
|---|---|
| **Rust** | Ownership, borrowing, no GC, algebraic types, traits |
| **Triton** | Block-level programming model, auto-tuning, MLIR backend |
| **Halide** | Separation of algorithm and schedule |
| **Futhark** | Functional GPU programming, parallel combinators (map, reduce, scan) |
| **CUDA** | Low-level control when needed, warp-level primitives |
| **Idris/Agda** | Dependent types for shape checking |
| **Zig** | Comptime evaluation, explicit allocators |
| **circom/noir** | ZK circuit DSL ideas, constraint system generation |

## Guiding Questions

Every design decision should be evaluated against:

1. **Does this make GPU programming safer?** (fewer bugs, fewer UB, fewer side channels)
2. **Does this make GPU programming faster?** (better codegen, less overhead)
3. **Does this make GPU programming more accessible?** (lower barrier to entry)
4. **Does this compose?** (can features be combined without surprising interactions)
5. **Does this perform?** (can we match hand-tuned CUDA within 5% on critical workloads)
