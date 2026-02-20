# 05 — Compiler Pipeline (MLIR-Based)

## Overview

Vortex uses an MLIR-based compilation pipeline. The frontend (written in Rust) parses
`.vx` source files into an AST, then lowers through multiple IR levels before emitting
GPU binaries for NVIDIA, AMD, and Intel hardware.

```
Source (.vx)
    │
    ▼
┌─────────────────────────┐
│  Lexer + Parser (Rust)  │  logos + recursive descent
│  → AST                  │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Type Checker           │  dependent types, ownership, effects
│  → Typed AST            │
└──────────┬──────────────┘
           │
           ▼
┌─────────────────────────┐
│  Vortex IR (Custom      │  high-level: tensor ops, crypto ops, attention
│   MLIR Dialect)         │  tracks memory spaces, shapes, field types
└──────────┬──────────────┘
           │  progressive lowering
           ▼
┌─────────────────────────┐
│  MLIR Linalg/Tensor     │  tile-level: matmul, generic, fill
│  + SCF/Affine           │  loop-level: for, if, parallel
└──────────┬──────────────┘
           │  tiling, fusion, vectorization
           ▼
┌─────────────────────────┐
│  MLIR GPU + Vector      │  GPU kernels, vector operations
│  + MemRef               │  memory management, async copies
└──────────┬──────────────┘
           │  backend-specific lowering
           ├────────────────┬────────────────┐
           ▼                ▼                ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│ NVVM Dialect │  │ ROCDL Dialect│  │ SPIRV Dialect │
│    → PTX     │  │   → AMDGCN   │  │   → SPIR-V   │
│    → CUBIN   │  │   → HSA CO   │  │   → Kernel   │
└──────────────┘  └──────────────┘  └──────────────┘
   NVIDIA GPU       AMD GPU           Intel GPU
```

---

## 1. Frontend: Lexer and Parser

### Technology

- **Lexer**: `logos` crate (Rust) — fast, zero-allocation tokenizer
- **Parser**: Hand-written recursive descent (like Rust's `rustc`)
  - Predictable error messages
  - Easy error recovery (can continue parsing after errors)
  - No parser generator dependency

### AST Design

```rust
// Simplified AST types (Rust)
enum Decl {
    Kernel { name: Ident, params: Vec<Param>, ret_type: Type, body: Block, schedule: Option<Schedule> },
    Function { name: Ident, generics: Vec<GenericParam>, params: Vec<Param>, ret_type: Type, body: Block },
    Struct { name: Ident, generics: Vec<GenericParam>, fields: Vec<Field> },
    Trait { name: Ident, methods: Vec<FnSig> },
    Impl { trait_name: Option<Path>, target: Type, methods: Vec<Decl> },
    TypeAlias { name: Ident, value: Type },
    Import { path: Path, items: Vec<ImportItem> },
}

enum Expr {
    Binary { lhs: Box<Expr>, op: BinOp, rhs: Box<Expr> },
    Call { func: Box<Expr>, args: Vec<Expr> },
    Index { base: Box<Expr>, indices: Vec<Expr> },
    MatMul { lhs: Box<Expr>, rhs: Box<Expr> },  // @ operator
    TensorLiteral { shape: Vec<Expr>, data: Vec<Expr> },
    If { cond: Box<Expr>, then_: Block, else_: Option<Block> },
    For { var: Ident, range: Range, body: Block, sequential: bool },
    Block(Vec<Stmt>),
    FieldAccess { base: Box<Expr>, field: Ident },
    // ...
}

enum Type {
    Scalar(ScalarType),                            // f32, u64, bool
    Tensor { elem: Box<Type>, shape: Vec<SizeExpr>, mem_space: Option<MemSpace> },
    Field { modulus: BigUint },                     // Field<P>
    Point { curve: Path, coords: CoordSystem },     // Point<BN254, Jacobian>
    Poly { field: Box<Type>, degree: SizeExpr, repr: PolyRepr },
    Generic { name: Ident, args: Vec<TypeArg> },
    Reference { mutable: bool, inner: Box<Type> },
    Secret(Box<Type>),                              // Secret<T>
    // ...
}
```

---

## 2. Custom MLIR Dialect: `vortex`

### Why a Custom Dialect

Existing MLIR dialects (`linalg`, `tensor`) don't capture:
- Finite field arithmetic semantics
- Elliptic curve operations
- Constant-time execution requirements
- NTT/MSM as single operations (important for optimization)
- Memory space annotations on tensors
- Pipeline/stage constructs for multi-GPU

### Dialect Definition

```tablegen
// VortexDialect.td
def Vortex_Dialect : Dialect {
  let name = "vortex";
  let summary = "GPU-native operations for crypto and LLM workloads";
  let cppNamespace = "::vortex";
  let useDefaultTypePrinterParser = 1;
  let useDefaultAttributePrinterParser = 1;
}
```

### Custom Types

```tablegen
// Field element type with modulus as type parameter
def Vortex_FieldType : TypeDef<Vortex_Dialect, "Field"> {
  let mnemonic = "field";
  let parameters = (ins
    ArrayRefParameter<"uint64_t">:$modulus_limbs,
    DefaultValuedParameter<"unsigned", "8">:$num_limbs
  );
  let assemblyFormat = "`<` $modulus_limbs `,` $num_limbs `>`";
}

// Tensor with memory space annotation
def Vortex_GPUTensorType : TypeDef<Vortex_Dialect, "GPUTensor"> {
  let mnemonic = "gpu_tensor";
  let parameters = (ins
    "Type":$elementType,
    ArrayRefParameter<"int64_t">:$shape,
    DefaultValuedParameter<"StringRef", "\"global\"">:$memorySpace
  );
}
```

### Key Operations

```tablegen
// NTT as a single operation (not lowered to loops until optimization)
def Vortex_NTTOp : Vortex_Op<"ntt", [Pure]> {
  let summary = "Number Theoretic Transform";
  let arguments = (ins
    Vortex_GPUTensorType:$input,     // Poly in coefficient form
    Vortex_GPUTensorType:$roots,     // Twiddle factors
    BoolAttr:$inverse                 // Forward or inverse NTT
  );
  let results = (outs Vortex_GPUTensorType:$output);
}

// MSM as a single operation
def Vortex_MSMOp : Vortex_Op<"msm", [Pure]> {
  let summary = "Multi-Scalar Multiplication";
  let arguments = (ins
    Vortex_GPUTensorType:$scalars,   // Field elements
    Vortex_GPUTensorType:$points,    // EC points
    OptionalAttr<I32Attr>:$window_size
  );
  let results = (outs Vortex_GPUTensorType:$result);  // Single EC point
}

// FlashAttention as a single operation
def Vortex_FlashAttentionOp : Vortex_Op<"flash_attention", [Pure]> {
  let summary = "Memory-efficient attention with online softmax";
  let arguments = (ins
    Vortex_GPUTensorType:$query,
    Vortex_GPUTensorType:$key,
    Vortex_GPUTensorType:$value,
    BoolAttr:$causal,
    OptionalAttr<F32Attr>:$dropout_rate
  );
  let results = (outs Vortex_GPUTensorType:$output);
}

// Constant-time region marker
def Vortex_ConstantTimeRegionOp : Vortex_Op<"constant_time_region", [
    SingleBlockImplicitTerminator<"vortex::YieldOp">
]> {
  let summary = "Region where all operations must be constant-time";
  let regions = (region SizedRegion<1>:$body);
}
```

### Example IR

```mlir
// Vortex IR for a ZK proof kernel
vortex.kernel @prove(%witness: !vortex.gpu_tensor<f64x[1024], "global">,
                      %generators: !vortex.gpu_tensor<point<bn254>x[1024], "global">)
    -> !vortex.gpu_tensor<point<bn254>x[1], "global"> {

  // NTT of witness (stays as high-level op for optimization)
  %w_ntt = vortex.ntt %witness, %roots {inverse = false}
      : !vortex.gpu_tensor<field<bn254_fr>x[1024]>

  // MSM (stays as high-level op — Pippenger algorithm selected during lowering)
  %commitment = vortex.msm %witness, %generators {window_size = 15}
      : !vortex.gpu_tensor<point<bn254>x[1]>

  vortex.return %commitment
}
```

---

## 3. Lowering Pipeline

### Phase 1: Vortex → Linalg/Tensor

```
vortex.ntt     → tiled loop nest with butterfly operations
vortex.msm     → Pippenger bucket method with EC point addition loops
vortex.matmul  → linalg.matmul
vortex.flash_attention → tiled loop nest with online softmax
```

### Phase 2: Linalg → SCF + Vector + GPU

```
linalg.matmul        → scf.for loops + vector.contract (→ tensor core MMA)
scf.parallel          → gpu.launch with thread mapping
memref.alloc(@shared) → gpu.alloc (shared memory)
```

### Phase 3: GPU → Backend-Specific

```
gpu.launch       → nvvm.kernel / rocdl.kernel
gpu.barrier      → nvvm.barrier0 / rocdl.barrier
vector.contract  → nvvm.mma.sync / rocdl.mfma
gpu.alloc(shared)→ nvvm.sharedmem / rocdl.lds
```

### Phase 4: LLVM → Machine Code

```
nvvm.*   → llvm.nvvm.* intrinsics → PTX → CUBIN (via libNVPTXCompiler)
rocdl.*  → llvm.amdgcn.* intrinsics → AMDGCN → HSA code object
spirv.*  → SPIR-V binary → kernel
```

---

## 4. Optimization Passes

### Vortex-Level Passes (before lowering)

| Pass | Description |
|---|---|
| `NTTFusion` | Fuse NTT → pointwise → iNTT into single kernel |
| `MSMWindowOptimization` | Select optimal Pippenger window size based on N |
| `ConstantTimeVerification` | Verify no secret-dependent branches in @constant_time regions |
| `FieldRepresentationSelection` | Choose Montgomery vs Barrett reduction based on modulus |
| `KernelFusionAnalysis` | Identify fusible operation sequences |
| `MemorySpaceInference` | Propagate memory space annotations |

### Linalg/SCF-Level Passes

| Pass | Description |
|---|---|
| `TileAndFuse` | Tile operations to GPU block/warp sizes, fuse producers into consumers |
| `Vectorize` | Lower to vector ops (→ tensor core instructions) |
| `BufferizeAndPromote` | Convert tensor semantics to memref, promote to shared memory |
| `PipelineSharedMemory` | Insert async copies and double/triple buffering |
| `BankConflictAvoidance` | Insert swizzling or padding for shared memory |

### GPU-Level Passes

| Pass | Description |
|---|---|
| `GPUKernelOutlining` | Extract GPU kernel functions |
| `SharedMemoryAllocation` | Assign shared memory offsets |
| `RegisterPressureAnalysis` | Warn if register usage limits occupancy |
| `OccupancyOptimization` | Tune thread block size for occupancy |

---

## 5. Auto-Tuning Framework

Kernel performance depends heavily on hardware-specific parameters. Vortex includes
a built-in auto-tuner.

### Tunable Parameters

```vortex
// The schedule annotation defines the search space
@schedule(
    tile_m = [64, 128, 256],           // try these tile sizes
    tile_n = [64, 128, 256],
    tile_k = [32, 64],
    num_warps = [4, 8],
    num_stages = [2, 3, 4],
    use_tensor_cores = [true, false],
)
kernel my_gemm(...) { ... }
```

### Tuning Process

```
1. Generate all valid configurations (filter by resource constraints)
2. Compile each configuration
3. Benchmark on target hardware (warmup + timed runs)
4. Select best configuration
5. Cache result keyed by: (kernel_name, input_shapes, dtype, gpu_arch)
```

### Tuning Cache

```
~/.vortex/tune_cache/
  sm_90/
    gemm_f16_4096x4096x4096.json    # Best config for this shape on H100
    ntt_bn254_1048576.json           # Best NTT config for 2^20 elements
    attention_f16_bs32_seq4096.json  # Best attention config
  gfx942/                            # AMD MI300X configs
    ...
```

---

## 6. JIT Compilation

For dynamic shapes (variable batch size, sequence length), Vortex supports
runtime kernel specialization:

```vortex
// When N is not known at compile time
fn dynamic_ntt(data: &mut Tensor<Fr, [?]>) {
    // Compiler generates a JIT path:
    // 1. At runtime, determine N
    // 2. Look up pre-compiled kernel for this N (common sizes cached)
    // 3. If not cached, compile on first use (MLIR → PTX → CUBIN)
    // 4. Cache the compiled kernel for future calls
    ntt_forward(data)
}
```

---

## 7. Multi-Backend Targeting

### Shared Pipeline (backend-agnostic)

Everything above the GPU dialect is shared across backends:
- Parsing, type checking, Vortex IR
- Tiling, fusion, vectorization in Linalg/SCF
- Bufferization and memory promotion

### Backend-Specific Code Generation

| Feature | NVIDIA (sm_90) | AMD (gfx942) | Intel (PVC) |
|---|---|---|---|
| Warp size | 32 | 64 (wavefront) | 32 (subgroup) |
| Shared memory | 228 KB/SM | 64 KB/CU | 128 KB/EU |
| Tensor cores | WGMMA 64x256x16 | MFMA 32x32x8 | XMX 16x16x16 |
| Async copy | cp.async, TMA | buffer_load | LSC |
| Barrier | bar.sync | s_barrier | barrier |
| Atomics | atom.global | buffer_atomic | atomic |

The compiler parameterizes these differences:

```rust
// Backend configuration (Rust, internal)
struct BackendConfig {
    warp_size: u32,
    max_shared_memory: u32,
    max_registers_per_thread: u32,
    mma_shapes: Vec<MMAShape>,
    has_tensor_memory_accelerator: bool,
    has_async_copy: bool,
    l2_cache_size: u32,
}

impl BackendConfig {
    fn nvidia_h100() -> Self { ... }
    fn amd_mi300x() -> Self { ... }
    fn intel_pvc() -> Self { ... }
}
```

---

## 8. Compiler Implementation Plan

### Phase 0: Minimal Viable Compiler

```
Week 1-2:  Lexer (logos) + parser (recursive descent) for subset of Vortex
Week 3-4:  AST → basic MLIR emission (arith + func dialects only)
Week 5-6:  Scalar kernel: compile f32 arithmetic to PTX via MLIR → LLVM → NVPTX
Week 7-8:  Tensor ops: 1D tensor add/multiply using memref + scf loops
Week 9-10: First real kernel: vector addition running on an NVIDIA GPU
Week 11-12: GEMM: tiled matrix multiplication with shared memory
```

### Technology Stack

| Component | Technology | Rationale |
|---|---|---|
| Compiler frontend | Rust | Fast, safe, great ecosystem (logos, ariadne for errors) |
| MLIR bindings | melior (Rust crate) | Native Rust bindings to MLIR C API |
| Custom dialect | TableGen + C++ | MLIR's standard approach for dialect definition |
| Build system | Cargo (compiler) + custom driver | Rust-native for compiler dev |
| Testing | FileCheck + lit (MLIR standard) | Proven infrastructure for compiler testing |
| Benchmarking | Custom + NSight integration | Automated perf regression testing |

### Build Command

```bash
# Compile a Vortex source file
$ vortex build kernel.vx --target sm_90 --optimize
  Parsing...          ✓  (2 ms)
  Type checking...    ✓  (15 ms)
  MLIR generation...  ✓  (8 ms)
  Optimization...     ✓  (120 ms)
  Code generation...  ✓  (45 ms)

  Output: kernel.cubin (sm_90)
  Resources:
    Shared memory: 32,768 / 228,000 bytes (14%)
    Registers: 96 / 255 per thread
    Occupancy: 50% (estimated)

# Cross-compile for AMD
$ vortex build kernel.vx --target gfx942
  ...
  Output: kernel.hsaco (gfx942)
```
