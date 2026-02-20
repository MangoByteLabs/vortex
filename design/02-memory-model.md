# 02 — GPU Memory Model & Ownership

## Overview

GPU memory management is the single most impactful factor in kernel performance.
Vortex makes memory hierarchy **explicit in the type system** while providing
high-level abstractions that compile to optimal hardware instructions.

---

## GPU Memory Hierarchy (Reference)

### NVIDIA (Hopper H100)

| Level | Size | Bandwidth | Latency | Scope |
|---|---|---|---|---|
| Registers | 256 KB/SM | ~20 TB/s | 0 cycles | Per-thread |
| Shared Memory (SMEM) | 228 KB/SM | ~20 TB/s | ~20-30 cycles | Per-block |
| L1 Cache | Unified with SMEM | ~20 TB/s | ~30 cycles | Per-SM |
| L2 Cache | 50 MB | ~12 TB/s | ~200 cycles | All SMs |
| HBM3 (VRAM) | 80 GB | 3.35 TB/s | ~400 cycles | Global |
| Host (PCIe/NVLink) | System RAM | 64-900 GB/s | ~10K cycles | System |

### AMD (MI300X — CDNA 3)

| Level | Size | Bandwidth | Latency | Scope |
|---|---|---|---|---|
| VGPR (Registers) | 512 KB/CU | ~30 TB/s | 0 cycles | Per-thread |
| LDS (Local Data Share) | 64 KB/CU | ~20 TB/s | ~20 cycles | Per-workgroup |
| L1 Vector Cache | 32 KB/CU | ~10 TB/s | ~30 cycles | Per-CU (read-only) |
| L2 Cache | 256 MB | ~6 TB/s | ~150 cycles | All CUs |
| HBM3 (VRAM) | 192 GB | 5.3 TB/s | ~400 cycles | Global |

---

## Memory Spaces in Vortex

### Type-Level Memory Space Tracking

Memory spaces are tracked in the type system. Invalid cross-space operations are
compile-time errors, not runtime bugs.

```vortex
// Memory space is a type parameter
trait MemorySpace {}
struct Global  : MemorySpace {}
struct Shared  : MemorySpace {}
struct Local   : MemorySpace {}
struct Constant: MemorySpace {}

// Tensors carry their memory space
type GlobalTensor<T, Shape>  = Tensor<T, Shape, Global>
type SharedTensor<T, Shape>  = Tensor<T, Shape, Shared>
type LocalTensor<T, Shape>   = Tensor<T, Shape, Local>

// Functions are generic over memory space
fn reduce<M: MemorySpace + Readable, T: Numeric>(
    data: &Tensor<T, [N], M>
) -> T {
    // Compiler knows the memory space and can optimize accordingly
    ...
}
```

### Explicit Placement

```vortex
kernel matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    // Shared memory tiles — compiler verifies size fits SM budget
    let a_tile: Tensor<f32, [128, 32]> @shared
    let b_tile: Tensor<f32, [32, 128]> @shared

    // Register accumulator
    var acc: Tensor<f32, [8, 8]> @local = Tensor.zeros()

    // Global → Shared staging (compiler lowers to cp.async on Ampere+, TMA on Hopper)
    a_tile = stage(a[block_m..block_m+128, k..k+32])
    b_tile = stage(b[k..k+32, block_n..block_n+128])

    sync_block()  // compiler inserts appropriate fence

    // Shared → Register compute
    acc += a_tile @ b_tile  // lowers to WMMA/MMA on tensor cores

    ...
}
```

### Automatic Bank Conflict Avoidance

```vortex
// The compiler automatically applies swizzling to shared memory layouts
let tile: Tensor<f32, [32, 32]> @shared(layout = .bank_conflict_free)
// Internally applies XOR-based index swizzling:
// physical_addr = logical_addr XOR (logical_addr >> shift)

// Or use explicit padded layout
let tile: Tensor<f32, [32, 33]> @shared  // +1 padding, classic approach

// Compiler can also verify access patterns
@verify_no_bank_conflicts
let val = tile[warp.lane_id, col]  // compiler checks at compile time
```

### Coalescing Verification

```vortex
// Compiler tracks access patterns and warns on non-coalesced access
let data: Tensor<f32, [N]> @global

// OK: stride-1 access (coalesced)
let val = data[thread.global_id]

// WARNING: stride-32 access (only 12.5% bandwidth utilization)
let val = data[thread.global_id * 32]
//         ^~~ compiler warning: non-coalesced access pattern detected
//             consider using gather() or restructuring data layout

// SUGGESTION: Use Structure of Arrays
struct ParticlesSoA {
    x: Tensor<f32, [N]>,
    y: Tensor<f32, [N]>,
    z: Tensor<f32, [N]>,
}
// Instead of:
// struct Particle { x: f32, y: f32, z: f32 }
// particles: Tensor<Particle, [N]>  // AoS — non-coalesced per-field access
```

---

## Ownership and Borrowing for GPU Memory

Adapted from Rust's ownership model for the CPU/GPU memory split.

### Rules

1. Every GPU allocation has exactly **one owner**
2. Ownership can be **transferred** (moved) but not copied (unless type is `Copy`)
3. You can have either **one mutable reference** or **any number of immutable references**
4. References cannot outlive the data they point to
5. **Device ↔ Host transfers** are explicit (no hidden copies)

### Transfer Semantics

```vortex
fn main() {
    // Allocate on host
    let host_data: Vec<f32> = vec![1.0, 2.0, 3.0, ...]

    // Transfer to GPU (explicit, host_data is moved)
    let gpu_data: Tensor<f32, [N]> @global = Tensor.from_host(host_data)
    // host_data is no longer accessible here — ownership moved to GPU

    // Launch kernel (borrows gpu_data immutably)
    let result = my_kernel(&gpu_data)

    // Transfer back to host (explicit)
    let host_result: Vec<f32> = result.to_host()
    // result (GPU memory) is freed when it goes out of scope
}

kernel my_kernel(data: &Tensor<f32, [N]>) -> Tensor<f32, [N]> {
    // data is an immutable borrow — cannot modify
    return data * 2.0
}

kernel my_kernel_mut(data: &mut Tensor<f32, [N]>) {
    // data is a mutable borrow — exclusive access
    data[thread.global_id] *= 2.0
}
```

### Lifetime of Shared Memory

```vortex
kernel example() {
    // Shared memory lifetime is scoped to the block
    {
        let tile: Tensor<f32, [256]> @shared
        tile[thread.local_id] = compute()
        sync_block()
        use(tile)
    }
    // tile is deallocated here (compiler reclaims shared memory for reuse)

    {
        // This can reuse the same shared memory region
        let other: Tensor<f32, [256]> @shared
        ...
    }
}
```

---

## Asynchronous Memory Operations

```vortex
kernel pipelined_matmul(
    a: Tensor<f16, [M, K]>,
    b: Tensor<f16, [K, N]>
) -> Tensor<f16, [M, N]> {
    // Double-buffered shared memory
    let buf_a: [Tensor<f16, [128, 32]> @shared; 2]
    let buf_b: [Tensor<f16, [32, 128]> @shared; 2]

    // Pipeline: load next tile while computing current
    for (k, stage) in range(0, K, 32).enumerate() {
        let buf = stage % 2
        let next_buf = (stage + 1) % 2

        // Async load next tile (non-blocking)
        if k + 32 < K {
            async_stage(&a[block_m..+128, k+32..+32], &mut buf_a[next_buf])
            async_stage(&b[k+32..+32, block_n..+128], &mut buf_b[next_buf])
        }

        // Compute on current tile (overlaps with load)
        sync_block()
        acc += buf_a[buf] @ buf_b[buf]
    }

    return acc
}
```

---

## Compile-Time Resource Analysis

The compiler provides resource budgets as part of compilation output.

```
$ vortex build matmul.vx --target sm_90

matmul_kernel:
  Target:           SM_90 (H100)
  Shared memory:    32,768 bytes (of 228,000 available)     [14%]
  Registers:        128 per thread (of 255 max)              [50%]
  Threads/block:    256
  Blocks/SM:        2 (limited by registers)
  Occupancy:        33% (512 of 1536 max threads/SM)

  Estimated performance:
    Arithmetic intensity: 64 FLOPs/byte
    Roofline bound:       Compute-bound (H100 balance: 295 FLOPs/byte)
    Estimated throughput:  ~780 TFLOPS (79% of peak)

  Suggestions:
    - Consider reducing register usage to 96 to allow 3 blocks/SM (50% occupancy)
    - Pipeline depth of 3 would hide 95% of memory latency at current occupancy
```

---

## Memory Model for Cryptography Workloads

### Large Integer Layout

```vortex
// 256-bit integer as Structure of Arrays for coalesced access
type U256SoA = struct {
    limb0: Tensor<u32, [N]>,  // least significant
    limb1: Tensor<u32, [N]>,
    limb2: Tensor<u32, [N]>,
    limb3: Tensor<u32, [N]>,
    limb4: Tensor<u32, [N]>,
    limb5: Tensor<u32, [N]>,
    limb6: Tensor<u32, [N]>,
    limb7: Tensor<u32, [N]>,  // most significant
}

// Compiler automatically uses SoA layout for Field<P> batches
// Thread i loads limb0[i], limb1[i], ... limb7[i] — 8 coalesced transactions
kernel batch_field_mul(
    a: Tensor<Field<P>, [N]>,  // stored as SoA internally
    b: Tensor<Field<P>, [N]>,
) -> Tensor<Field<P>, [N]> {
    let tid = thread.global_id
    return a[tid] * b[tid]  // Montgomery multiplication, all in registers
}
```

### NTT Memory Strategy

```vortex
@schedule(
    stages(0..10):  global_memory, coalesced,
    stages(10..24): fused_in_shared_memory(48 * 1024),
)
kernel ntt_forward<F: PrimeField>(
    data: &mut Tensor<F, [N]>,
    roots: &Tensor<F, [N]>,
) where N: PowerOfTwo {
    // Early stages: global memory butterflies with coalesced access
    // Late stages: load block into shared memory, do all remaining stages there
    // Compiler generates the optimal split point based on N and shared memory size
    ...
}
```

---

## Memory Model for LLM Workloads

### FlashAttention Memory Strategy

```vortex
@fused(strategy = .flash_attention)
kernel attention(
    q: Tensor<f16, [B, H, S, D]>,
    k: Tensor<f16, [B, H, S, D]>,
    v: Tensor<f16, [B, H, S, D]>,
    causal: bool,
) -> Tensor<f16, [B, H, S, D]> {
    // Memory budget per block:
    //   Q tile:     BLOCK_M x D x 2 bytes = 128 x 128 x 2 = 32 KB  @shared
    //   K tile:     BLOCK_N x D x 2 bytes = 128 x 128 x 2 = 32 KB  @shared
    //   V tile:     BLOCK_N x D x 2 bytes = 128 x 128 x 2 = 32 KB  @shared
    //   Scores:     BLOCK_M x BLOCK_N x 4 bytes = 64 KB             @local (registers)
    //   Accumulator: BLOCK_M x D x 4 bytes = 64 KB                  @local (registers)
    //   Softmax stats: BLOCK_M x 2 x 4 bytes = 1 KB                 @local
    //
    // Total shared: 96 KB (fits in H100's 228 KB budget)
    // Total registers: ~130 KB (across all threads in the block)

    // The compiler handles tiling, online softmax, and pipelining
    let scores = q @ k.transpose() / sqrt(D)
    if causal { scores = scores.mask_upper_triangle(-inf) }
    let weights = softmax(scores, dim = -1)
    return weights @ v
}
```

### KV-Cache with Paging

```vortex
// Paged KV-cache for variable-length serving
struct PagedKVCache<T: Float> {
    // Physical pages: [num_pages, page_size, num_heads, head_dim]
    pages: Tensor<T, [MAX_PAGES, PAGE_SIZE, H, D]> @global,
    // Page table: maps (sequence, logical_block) -> physical_page
    page_table: Tensor<u32, [MAX_SEQ, MAX_BLOCKS_PER_SEQ]> @global,
    // Sequence lengths
    seq_lens: Tensor<u32, [MAX_SEQ]> @global,
}

kernel paged_attention(
    q: Tensor<f16, [B, H, 1, D]>,   // single query token per sequence
    cache: &PagedKVCache<f16>,
) -> Tensor<f16, [B, H, 1, D]> {
    // Indirect indexing through page table
    // Compiler optimizes scatter/gather pattern for L2 locality
    for seq in 0..B {
        let len = cache.seq_lens[seq]
        for block in 0..ceil_div(len, PAGE_SIZE) {
            let page_id = cache.page_table[seq, block]
            let k_page = cache.pages[page_id, :, :, :]  // gather
            let v_page = cache.pages[page_id, :, :, :]
            // Accumulate attention over this page
            ...
        }
    }
}
```

---

## Cross-Architecture Portability

The memory model abstracts over vendor differences:

```vortex
// This kernel compiles to optimal code on NVIDIA, AMD, and Intel
kernel generic_reduce<T: Numeric>(data: Tensor<T, [N]>) -> T {
    let tile: Tensor<T, [BLOCK_SIZE]> @shared  // SMEM on NVIDIA, LDS on AMD

    tile[thread.local_id] = data[thread.global_id]
    sync_block()

    // Tree reduction in shared memory
    for stride in [BLOCK_SIZE/2, BLOCK_SIZE/4, ..., 1] {
        if thread.local_id < stride {
            tile[thread.local_id] += tile[thread.local_id + stride]
        }
        sync_block()
    }

    return tile[0]
}

// Backend-specific lowering:
// NVIDIA: __shared__ → SMEM, __syncthreads() → bar.sync
// AMD:    __shared__ → LDS, __syncthreads() → s_barrier
// Intel:  __shared__ → SLM, __syncthreads() → barrier
```
