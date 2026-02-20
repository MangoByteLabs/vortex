# 09 — Implementation Roadmap

## Phase 0: Foundation (Months 1-3)

### Goal: Prove the concept compiles and runs on a GPU

**Deliverables:**
- [ ] Language grammar specification (EBNF)
- [ ] Lexer and parser (written in Rust)
- [ ] AST definition and pretty-printer
- [ ] Basic type checker (scalars + fixed-size tensors)
- [ ] MLIR code generation for a minimal subset:
  - Scalar arithmetic (u32, f32)
  - 1D tensor operations (add, multiply, reduce)
  - A single `kernel` function that compiles to PTX
- [ ] End-to-end test: vector addition that runs on an NVIDIA GPU
- [ ] Basic error messages with source locations

**Tech stack:**
- Compiler frontend: Rust (using `logos` for lexing, custom recursive descent parser)
- IR: Custom AST → MLIR (using `melior` or direct MLIR C API bindings from Rust)
- Backend: MLIR → LLVM → PTX (NVIDIA only initially)
- Build system: Cargo for the compiler, custom driver for `.vx` files

**Key decisions to finalize:**
- Parser strategy: recursive descent vs PEG vs parser combinator (nom/pest)
- MLIR binding approach: melior (Rust), raw C API, or shell out to mlir-opt
- Testing framework for the compiler itself

---

## Phase 1: Type System & Core Language (Months 4-7)

### Goal: A usable type system with shape checking and field types

**Deliverables:**
- [ ] Full type system implementation:
  - Parametric polymorphism (generics) with monomorphization
  - Dependent types for tensor shapes (symbolic dimensions)
  - `Field<P>` type with compile-time modulus
  - Trait system (Numeric, PrimeField, Float)
  - Ownership and borrowing (simplified Rust model)
- [ ] Memory space annotations (@shared, @global, @local)
- [ ] Control flow: if/else, match, for, while
- [ ] Module system and imports
- [ ] 2D tensor operations (matmul via naive tiled kernel)
- [ ] Compile-time evaluation (comptime)
- [ ] Comprehensive test suite for type checker

**Milestone benchmark:**
- Matrix multiplication (f32, 4096x4096) running on GPU
- Basic modular arithmetic (256-bit field elements) running on GPU

---

## Phase 2: Cryptography Primitives (Months 8-11)

### Goal: GPU-accelerated ZK proving faster than existing Rust/CUDA solutions

**Deliverables:**
- [ ] Big integer library (256-bit, 384-bit) using Montgomery multiplication
- [ ] Finite field arithmetic:
  - BN254, BLS12-381, Goldilocks, Mersenne31 fields
  - Field extensions (Fp2, Fp6, Fp12)
- [ ] Number Theoretic Transform (NTT):
  - Radix-2 and mixed-radix implementations
  - Auto-tuned for GPU shared memory size
- [ ] Multi-Scalar Multiplication (MSM):
  - Pippenger's algorithm with optimal bucket sizing
  - Warp-level cooperation for point accumulation
- [ ] Elliptic curve operations:
  - Short Weierstrass, Montgomery, twisted Edwards forms
  - Point addition, doubling, scalar multiplication
  - Pairing computation (Ate pairing for BN254/BLS12-381)
- [ ] `@constant_time` verification pass:
  - Detect data-dependent branches
  - Verify uniform warp execution
  - Flag potential timing side channels
- [ ] Polynomial operations:
  - Evaluation, interpolation, commitment
  - Division, multiplication via NTT

**Milestone benchmark:**
- MSM of 2^20 points on BN254: target < 1 second on RTX 4090
- NTT of 2^24 elements: target competitive with Icicle/CUDA
- Compare against: Icicle, gnark, arkworks, bellman

---

## Phase 3: LLM Primitives (Months 8-11, parallel with Phase 2)

### Goal: Run a full LLM inference pipeline in Vortex

**Deliverables:**
- [ ] Tensor core integration:
  - WMMA/MMA intrinsics for f16, bf16, tf32, int8
  - Auto-detection of tensor core compatibility
- [ ] FlashAttention implementation:
  - Tiled attention with online softmax
  - Causal masking, variable sequence lengths
  - Multi-query attention (MQA) and grouped-query attention (GQA)
- [ ] Optimized GEMM:
  - Hierarchical tiling (thread block, warp, thread)
  - Persistent kernel approach for large matrices
  - Split-K for tall-skinny matrices
- [ ] Quantization support:
  - INT8, INT4, FP8 quantized matmul
  - Dequantize-on-the-fly in fused kernels
  - GPTQ and AWQ weight formats
- [ ] Activation functions (fused):
  - SiLU, GELU, ReLU, softmax
- [ ] Normalization (fused):
  - RMSNorm, LayerNorm
- [ ] KV-cache management:
  - Paged allocation
  - Continuous batching support
- [ ] Kernel fusion pass:
  - Automatic fusion of elementwise ops
  - Fuse residual connections + normalization
  - Fuse linear projection + activation

**Milestone benchmark:**
- Llama 2 7B inference (single GPU): target < 50ms per token on RTX 4090
- Compare against: llama.cpp (CUDA), TensorRT-LLM, vLLM

---

## Phase 4: Compiler Optimization (Months 12-15)

### Goal: Match hand-tuned CUDA within 10% on key benchmarks

**Deliverables:**
- [ ] Auto-tuning framework:
  - Search over tile sizes, block dimensions, num_warps
  - Cache tuning results per GPU architecture
  - JIT compilation with runtime specialization
- [ ] Advanced MLIR passes:
  - Loop tiling and interchange
  - Vectorization (mapping to GPU vector instructions)
  - Software pipelining for memory latency hiding
  - Register pressure optimization
- [ ] Multi-GPU support:
  - Pipeline parallelism (pipeline/stage syntax)
  - Tensor parallelism (automatic sharding)
  - NCCL/RCCL integration for collective ops
- [ ] AMD GPU backend:
  - MLIR → AMDGCN via ROCDL dialect
  - Test on MI250X/MI300X
- [ ] Intel GPU backend:
  - MLIR → SPIR-V via spirv dialect
  - Test on Intel Data Center Max (Ponte Vecchio)

**Milestone benchmark:**
- Achieve within 10% of cuBLAS GEMM on NVIDIA
- Achieve within 10% of rocBLAS GEMM on AMD
- Full Llama 2 70B inference across 4 GPUs

---

## Phase 5: Ecosystem & Tooling (Months 16-20)

### Goal: A language people actually want to use

**Deliverables:**
- [ ] Package manager (`vortex-pkg` or similar)
- [ ] LSP server (autocomplete, go-to-definition, type hints)
- [ ] Debugger:
  - GPU kernel debugging (mapping back to source)
  - Printf-style debugging that doesn't kill performance
  - Record-replay for GPU kernels
- [ ] Profiler integration:
  - nsight-compute / rocprof integration
  - Source-level performance annotations
  - Roofline model visualization
- [ ] Standard library (`std`):
  - `std.crypto` — fields, curves, NTT, MSM, hash functions
  - `std.nn` — attention, linear, normalization, activation
  - `std.tensor` — creation, manipulation, reduction, indexing
  - `std.io` — file I/O, model loading (safetensors, GGUF)
  - `std.random` — PRNG on GPU (philox, threefry)
  - `std.comm` — multi-GPU communication primitives
- [ ] Python bindings (`pyvortex`):
  - PyTorch-compatible tensor interface
  - JIT compilation of Vortex kernels from Python
  - NumPy interop
- [ ] Documentation and tutorials
- [ ] Benchmark suite

---

## Phase 6: Production Hardening (Months 21-24)

### Goal: Ready for production use in crypto and ML infrastructure

**Deliverables:**
- [ ] Fuzzing the compiler (coverage-guided fuzzing of parser, type checker, codegen)
- [ ] Formal verification of constant-time guarantees
- [ ] Security audit of crypto primitives
- [ ] Performance regression testing (CI/CD)
- [ ] Memory leak detection and GPU resource tracking
- [ ] Error recovery in parser (don't stop at first error)
- [ ] Incremental compilation
- [ ] Cross-compilation support (compile on x86, target GPU)

---

## Success Criteria

### Cryptography
- MSM performance competitive with Icicle (within 20%)
- Correctly implement BN254 and BLS12-381 pairings
- Pass all test vectors from ZK proof system test suites
- Demonstrate a full Groth16/PLONK prover in Vortex

### LLM
- Llama-class model inference within 15% of TensorRT-LLM
- FlashAttention within 10% of the official CUDA implementation
- GEMM within 10% of cuBLAS/rocBLAS
- Demonstrate full training loop for a small model

### Language
- Compiler handles 100K+ line programs without issues
- Compilation time < 10 seconds for typical projects
- Zero compiler crashes on valid programs
- Clear, actionable error messages

---

## Team Structure (Ideal)

| Role | Count | Focus |
|---|---|---|
| Compiler engineer | 2-3 | Frontend, type system, MLIR codegen |
| GPU kernel engineer | 2 | Optimization passes, backend targets |
| Cryptography engineer | 1-2 | Crypto primitives, constant-time verification |
| ML engineer | 1-2 | LLM kernels, quantization, distributed |
| DevTools engineer | 1 | LSP, debugger, profiler, package manager |
| Technical writer | 1 | Docs, tutorials, examples |

Minimum viable team: **3 people** (compiler + GPU + domain expert)

## Open Research Questions

1. **Can we achieve constant-time guarantees on GPUs?** Warp divergence is fundamental to GPU execution. We need to verify that `@constant_time` blocks actually produce uniform execution.

2. **How do we auto-tune effectively?** The search space for GPU kernel configurations is enormous. Bayesian optimization? ML-based cost models? Polyhedral analysis?

3. **Can dependent types be made practical?** Shape-dependent types are powerful but can make type inference undecidable. Where do we draw the line?

4. **How do we handle dynamic shapes?** LLM serving has dynamic batch sizes and sequence lengths. We need runtime specialization without JIT overhead.

5. **Is MLIR mature enough?** MLIR is powerful but the GPU dialects are still evolving. We may need to contribute upstream.
