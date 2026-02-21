# Vortex: State-of-the-Art Development Plan

> From proof-of-concept (19K LOC, 190 tests) to a production GPU language
> capable of defining and training novel LLM architectures.

---

## Current State Assessment

### What Works
- **Lexer**: 60+ tokens, logos-based, robust
- **Parser**: Recursive descent + Pratt precedence, structs/enums/traits/impls/kernels/match/closures
- **Type checker**: Two-pass (collect + check), but largely permissive
- **Interpreter**: ~100 builtins, runs full programs (crypto, ML training loops, spiking nets)
- **MLIR codegen**: Emits `func.func`, `gpu.module/func`, `scf.for/while`, `linalg.matmul` — structural skeleton
- **Crypto**: BigUint256, secp256k1 (Jacobian), SHA-256, ECDSA, Schnorr, NTT, MSM, Montgomery
- **ML builtins**: SSM/Mamba scan, LIF spiking, ODE solvers, Forward-Forward, Hebbian, differentiable memory, sparse MoE
- **Autodiff**: Scalar reverse-mode tape with SGD/Adam

### Critical Gaps (Blocking GPU Training)
| Gap | Impact |
|-----|--------|
| No type unification / Hindley-Milner inference | Types don't propagate; `Named` always passes |
| MLIR codegen has ~15 stubs (indexing, field access, arrays, closures, match) | Cannot compile real programs |
| All function calls return `i64` in codegen | Wrong types in generated IR |
| No tensor autodiff | Only scalar f64 tape; can't backprop through matmul |
| GPU pipeline pass ordering wrong | `arith→LLVM` runs before `gpu→NVVM` |
| `@`/`.*`/`./` unimplemented in interpreter | Core tensor ops error at runtime |
| `attention` builtin is a stub (returns V unchanged) | No real attention mechanism |
| `break`/`continue` error at runtime | Loops can't exit early |
| Monomorphization not wired into codegen path | Generics don't compile |

---

## Development Phases

### Phase 1: Fix the Foundation (Weeks 1-4)
*Goal: Every valid Vortex program compiles to correct MLIR and interprets correctly.*

#### 1.1 Type System Overhaul
- [ ] Implement unification with substitution map (`Type::Var(n)` → concrete type)
- [ ] Wire struct field types into `FieldAccess` inference (currently returns `fresh_var`)
- [ ] Wire struct literal types (currently returns `fresh_var`)
- [ ] Remove the `Named` type hole (`typeck.rs:916` — always returns compatible)
- [ ] Match arms must unify to common type (fix last-arm-wins at `typeck.rs:839`)
- [ ] Type-check trait/impl bodies (currently skipped at `typeck.rs:282`)
- [ ] Reference types preserve mutability (currently erased at `typeck.rs:1029`)
- [ ] Tuples as distinct type from arrays (currently aliased at `parser.rs:1464`)

#### 1.2 Interpreter Correctness
- [ ] Implement `@` (matmul) for 2D arrays in interpreter
- [ ] Implement `.*` and `./` (elementwise ops)
- [ ] Fix `Cast` (currently a no-op — `interpreter.rs:2015`)
- [ ] Fix `break`/`continue` (currently error — `interpreter.rs:1729`)
- [ ] Fix multi-index `a[i,j]` (currently ignores all but first index)
- [ ] Implement real `attention` builtin (scaled dot-product + softmax)
- [ ] Parse `loop` statement (keyword exists but not parsed)
- [ ] Expose missing builtins: `rk45_solve`, `spike_attention`, `oja_update`, `chunked_scan`

#### 1.3 MLIR Codegen Fixes
- [ ] Function calls emit correct return types (not always `i64`)
- [ ] Array literals allocate `memref` and store all elements
- [ ] Indexing emits `memref.load` / `tensor.extract`
- [ ] Field access emits `llvm.extractvalue` or struct GEP
- [ ] Complex assignment targets (`a[i] = v`, `obj.field = v`) emit stores
- [ ] `break`/`continue` → `scf.execute_region` or restructure to avoid
- [ ] Fix `gpu.return` vs `func.return` double terminator
- [ ] Fix `linalg.matmul` operand types (RHS uses LHS type)
- [ ] Fix `is_float_type` for integer tensors
- [ ] Unsigned types use `divui`/`cmpi` unsigned predicates
- [ ] Match codegen: proper tag dispatch, SSA yield through `scf.if`
- [ ] Elementwise ops → `linalg.generic`
- [ ] Wire monomorphization into codegen pipeline

#### 1.4 Pipeline Pass Ordering
- [ ] Move `--gpu-to-nvvm`/`--gpu-to-rocdl` BEFORE `--convert-arith-to-llvm`
- [ ] Validate pass ordering against MLIR upstream examples

**Exit criteria**: `cargo test` passes 250+ tests; `vector_add.vx` produces valid MLIR that `mlir-opt` accepts.

---

### Phase 2: Tensor Autodiff Engine (Weeks 5-8)
*Goal: Backpropagation through tensor operations on the computation graph.*

#### 2.1 Tensor-Level Tape
- [ ] `TensorTape` with operations: `matmul`, `add`, `transpose`, `reshape`, `softmax`, `layer_norm`, `conv2d`
- [ ] Each op stores forward values + backward closure
- [ ] Backward pass: reverse accumulation over tensor ops
- [ ] Memory: gradient checkpointing for long sequences (recompute vs store tradeoff)

#### 2.2 Operator Overloading for AD
- [ ] `Tensor` values track tape participation automatically
- [ ] `@`, `+`, `-`, `*` on tracked tensors record ops
- [ ] `.backward()` method on loss scalar triggers full backprop
- [ ] Gradient accumulation for shared parameters

#### 2.3 Optimizers (Tensor-Level)
- [ ] SGD with momentum (tensor-level, not scalar loop)
- [ ] Adam / AdamW with weight decay
- [ ] Learning rate schedulers (cosine, warmup-linear)

**Exit criteria**: `llm_from_scratch.vx` trains a 2-layer transformer with tensor autodiff (not scalar tape).

---

### Phase 3: GPU Execution Pipeline (Weeks 9-16)
*Goal: Vortex programs compile and run on NVIDIA/AMD GPUs.*

#### 3.1 MLIR Toolchain Integration
- [ ] Install/bundle `mlir-opt`, `mlir-translate`, `llc` (LLVM 18+)
- [ ] End-to-end: `.vx` → MLIR → optimized MLIR → LLVM IR → PTX → cubin
- [ ] Test with `vector_add.vx` on real GPU
- [ ] AMD path: → AMDGCN → `.hsaco`

#### 3.2 Runtime Library
- [ ] CUDA FFI: `cuInit`, `cuModuleLoad`, `cuLaunchKernel`, `cuMemAlloc/Free/Copy`
- [ ] Buffer management: host↔device transfers, pinned memory
- [ ] Kernel launch configuration: grid/block dims from tensor shapes
- [ ] Multi-GPU: `cuCtxSetCurrent` for device selection

#### 3.3 Memory Model
- [ ] `memref` dialect for buffer allocation
- [ ] Automatic host↔device placement
- [ ] Lazy transfers (only copy when needed)
- [ ] Arena allocator for training (bulk free per iteration)

#### 3.4 Core GPU Kernels
- [ ] GEMM tiling (register-level blocking, shared memory)
- [ ] FlashAttention (tiled, online softmax, O(N) memory)
- [ ] Fused RMSNorm + residual
- [ ] Softmax (online, numerically stable)
- [ ] Cross-entropy loss (fused with softmax)

**Exit criteria**: A 2-layer transformer trains on GPU with correct gradients matching CPU reference.

---

### Phase 4: Novel LLM Architecture Support (Weeks 17-24)
*Goal: Define, train, and benchmark novel architectures in Vortex.*

#### 4.1 Architecture Primitives (GPU-Accelerated)
- [ ] **State Space Models**: GPU-parallel selective scan (Mamba-2/SSD style)
- [ ] **Spiking Transformers**: Sparse spike-driven attention on GPU
- [ ] **Neural ODE layers**: Adaptive RK45 with adjoint method for gradients
- [ ] **Mixture of Experts**: Top-k routing + expert dispatch across devices
- [ ] **Forward-Forward**: Local layer-wise training (no global backprop)
- [ ] **Differentiable Memory**: NTM-style read/write heads with gradient flow
- [ ] **Liquid Neural Networks**: CfC cells with ODE-based dynamics
- [ ] **Predictive Coding**: Hierarchical error-driven updates

#### 4.2 Architecture Composition Framework
- [ ] `@architecture` decorator for model definitions
- [ ] Automatic shape inference through model graph
- [ ] Mixed-precision training (FP16/BF16 forward, FP32 gradients)
- [ ] Gradient scaling for mixed precision

#### 4.3 Novel Architecture: Hybrid Spike-SSM-Attention
Design a novel architecture combining:
- SSM backbone for O(N) sequence processing
- Sparse spiking gates that activate attention only when needed
- Local Forward-Forward training for early layers, global backprop for final layers
- Liquid time-constant adaptation for variable-length sequences

```
// Example Vortex definition
@architecture
struct HybridSSMSpike {
    ssm_layers: [SelectiveSSM; 4]
    spike_gate: LIFLayer
    attention: SparseAttention
    ff_head: FFLayer      // Forward-Forward trained
    output: Linear

    fn forward(self, x: Tensor<bf16, [B, T, D]>) -> Tensor<bf16, [B, T, V]> {
        let h = self.ssm_layers.fold(x, |acc, layer| layer.scan(acc))
        let spikes = self.spike_gate.forward(h)
        let attended = where spikes {
            self.attention.forward(h)  // only compute attention where spikes fire
        } else {
            h  // pass through SSM output directly
        }
        let local = self.ff_head.forward(attended)  // local learning
        self.output.forward(local)
    }
}
```

#### 4.4 Benchmarking Suite
- [ ] Throughput: tokens/sec vs PyTorch/JAX on same hardware
- [ ] Memory: peak GPU memory vs baselines
- [ ] Convergence: loss curves on WikiText-103 / OpenWebText subset
- [ ] Latency: time-to-first-token for inference

**Exit criteria**: Train a 125M-parameter hybrid architecture on 1B tokens; publish benchmark results.

---

### Phase 5: Language Maturity (Weeks 25-36)
*Goal: Production-quality language tooling and ecosystem.*

#### 5.1 Module System
- [ ] `import` / `from` with file-based modules
- [ ] Standard library: `std.tensor`, `std.nn`, `std.optim`, `std.crypto`, `std.io`
- [ ] Package manager (simple: git-based deps)

#### 5.2 Advanced Type System
- [ ] Dependent types for tensor shapes: `Tensor<f32, [B, T, D]>` checked at compile time
- [ ] `Field<P>` type with compile-time prime verification
- [ ] Trait bounds enforced during monomorphization
- [ ] Const generics: `fn matmul<const M: usize, const N: usize, const K: usize>(...)`

#### 5.3 Developer Experience
- [ ] LSP server (go-to-definition, hover types, diagnostics)
- [ ] REPL with GPU execution
- [ ] Debugger: step through kernel execution, inspect device memory
- [ ] Profiler: roofline model, memory bandwidth utilization
- [ ] Error messages: codespan-reporting already in place, improve quality

#### 5.4 Interop
- [ ] Python bindings: call Vortex kernels from Python, load PyTorch tensors
- [ ] ONNX import: load pretrained models into Vortex runtime
- [ ] Checkpoint format: save/load model weights

#### 5.5 Optimizing Compiler
- [ ] Kernel fusion (adjacent elementwise ops → single kernel)
- [ ] Auto-tuning (tile sizes, block dims — Triton-style)
- [ ] Operator scheduling (overlap compute + memory transfers)
- [ ] Dead code elimination, constant folding, inlining

**Exit criteria**: External contributors can `pip install vortex` and run a Vortex model from Python.

---

### Phase 6: Scaling & Production (Weeks 37-52)
*Goal: Multi-GPU training at scale, competitive performance.*

#### 6.1 Distributed Training
- [ ] Data parallelism (gradient all-reduce via NCCL)
- [ ] Tensor parallelism (shard attention heads across GPUs)
- [ ] Pipeline parallelism (model stages across GPUs)
- [ ] Expert parallelism (MoE experts on different devices)
- [ ] ZeRO-style optimizer state sharding

#### 6.2 Performance Engineering
- [ ] Match cuBLAS GEMM performance (>80% of peak FLOPS)
- [ ] Match FlashAttention-2 memory efficiency
- [ ] Quantized inference (INT4/INT8/FP8)
- [ ] Speculative decoding for fast inference
- [ ] KV-cache with paged attention (vLLM-style)

#### 6.3 Safety & Verification
- [ ] `@constant_time` blocks with compiler verification
- [ ] Memory safety (ownership/borrowing or region-based)
- [ ] Formal verification of crypto primitives against specs

---

## Architecture Decision Records

### ADR-1: MLIR as IR (Decided)
MLIR provides progressive lowering, vendor-neutral GPU dialect, and access to LLVM backend.
No custom IR needed — leverage existing passes.

### ADR-2: Tensor Autodiff Strategy (Proposed)
**Option A**: Trace-based (like JAX) — record ops during forward, replay backward.
**Option B**: Source-transform (like Zygote) — generate backward code at compile time.
**Recommendation**: Option A for Phase 2 (simpler), migrate to B for Phase 5 (better optimization).

### ADR-3: Novel Architecture Focus (Proposed)
Rather than competing on standard Transformers (PyTorch/JAX are mature), focus on architectures
that are awkward in existing frameworks:
- Spiking networks (sparse, event-driven — bad fit for dense tensor ops)
- Neural ODEs (adaptive step sizes — bad fit for static graphs)
- Hybrid SSM+Attention (conditional computation — needs compiler support)
- Local learning rules (per-layer updates — no global backward pass needed)

This is where Vortex's compiler can provide 10x advantage over framework-based approaches.

### ADR-4: Crypto+ML Convergence (Unique Differentiator)
No other language targets both crypto and ML on GPU. Use cases:
- Verifiable ML inference (prove model output without revealing weights)
- Private training (MPC/FHE for federated learning)
- On-chain ML (smart contracts that run inference)

---

## Success Metrics

| Milestone | Target | Metric |
|-----------|--------|--------|
| Phase 1 complete | Week 4 | 250+ tests, valid MLIR from `mlir-opt` |
| Phase 2 complete | Week 8 | Tensor backprop matches PyTorch numerically |
| Phase 3 complete | Week 16 | Transformer trains on GPU, correct loss curve |
| Phase 4 complete | Week 24 | Novel 125M model trained, benchmark published |
| Phase 5 complete | Week 36 | `pip install vortex` works, LSP exists |
| Phase 6 complete | Week 52 | Multi-GPU training at >50% hardware utilization |

---

## Immediate Next Actions (This Week)

1. **Fix the type hole**: Remove `Named` always-compatible in `typeck.rs:916`
2. **Implement `break`/`continue`**: Use control flow enum in interpreter
3. **Fix function call return types in codegen**: Track return types properly
4. **Implement real `attention` builtin**: Scaled dot-product + softmax
5. **Fix MLIR pass ordering**: GPU lowering before LLVM conversion
</content>
</invoke>