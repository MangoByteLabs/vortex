# 04 — LLM Primitives

## Overview

Vortex provides first-class support for the computational patterns that dominate
LLM training and inference: matrix multiplication, attention, quantization,
normalization, and distributed execution.

---

## 1. Matrix Multiplication (GEMM)

### Tensor Core Integration

```vortex
// Vortex auto-detects tensor core compatibility and selects the right instruction
kernel matmul<T: Float>(
    a: Tensor<T, [M, K]>,
    b: Tensor<T, [K, N]>,
) -> Tensor<f32, [M, N]> {
    // The @ operator compiles to:
    // - HMMA (FP16 tensor cores) if T == f16
    // - WGMMA (Hopper warp-group MMA) if target >= sm_90
    // - IMMA (INT8 tensor cores) if T == i8
    // - cuBLAS-equivalent tiled GEMM for general cases
    return a @ b
}

// Explicit precision control
kernel mixed_precision_matmul(
    a: Tensor<f8e4m3, [M, K]>,
    b: Tensor<f8e4m3, [K, N]>,
) -> Tensor<f32, [M, N]> {
    // FP8 inputs, FP32 accumulation
    // Compiler selects FP8 tensor core instructions on Hopper+
    return a @ b  // accumulation type inferred from return type
}
```

### Tiled GEMM with Schedule Control

```vortex
@schedule(
    tile_m = 128, tile_n = 256, tile_k = 64,
    num_warps = 8,
    num_stages = 4,  // pipeline depth for memory latency hiding
    use_tensor_cores = true,
)
kernel optimized_gemm(
    a: Tensor<f16, [M, K]>,
    b: Tensor<f16, [K, N]>,
) -> Tensor<f16, [M, N]> {
    let block_m = workgroup.id.x * 128
    let block_n = workgroup.id.y * 256

    var acc: Tensor<f32, [128, 256]> @local = Tensor.zeros()

    // 4-stage pipeline: load K+3, K+2, K+1 while computing K
    @pipeline(stages = 4)
    for k in range(0, K, step = 64) {
        // Async load tiles to shared memory (compiler emits cp.async / TMA)
        let a_tile = async_load(a[block_m..+128, k..+64], @shared)
        let b_tile = async_load(b[k..+64, block_n..+256], @shared)

        await a_tile, b_tile
        sync_block()

        // Warp-level MMA (compiler maps to WMMA/WGMMA)
        acc += a_tile @ b_tile
    }

    // Store with optional epilogue fusion
    store(acc.cast<f16>(), output[block_m..+128, block_n..+256])
}
```

### Performance Targets (GEMM)

For `M=N=K=4096`, FP16:
| GPU | cuBLAS | Vortex Target | % of Peak |
|---|---|---|---|
| A100 | ~280 TFLOPS | 260+ TFLOPS | >83% |
| H100 | ~800 TFLOPS | 720+ TFLOPS | >73% |
| RTX 4090 | ~165 TFLOPS | 150+ TFLOPS | >80% |

---

## 2. FlashAttention

### High-Level API

```vortex
import nn.attention { flash_attention, FlashAttentionConfig }

// Simple API — compiler generates FlashAttention kernel
let output = flash_attention(q, k, v, causal = true)

// With configuration
let config = FlashAttentionConfig {
    block_m: 128,
    block_n: 128,
    num_warps: 8,
    num_stages: 3,
    dropout: 0.0,
}
let output = flash_attention(q, k, v, causal = true, config = config)
```

### Implementation

```vortex
// FlashAttention with online softmax — never materializes N x N attention matrix
@fused(strategy = .flash_attention)
kernel attention<T: Float>(
    q: Tensor<T, [B, H, S, D]>,      // queries
    k: Tensor<T, [B, H, S_kv, D]>,   // keys (S_kv may differ for GQA)
    v: Tensor<T, [B, H, S_kv, D]>,   // values
    causal: comptime bool,
) -> Tensor<T, [B, H, S, D]> {
    const BLOCK_M = 128  // query block size
    const BLOCK_N = 128  // key/value block size

    parallel_for (b, h, q_block) in grid(B, H, ceil_div(S, BLOCK_M)) {
        let q_start = q_block * BLOCK_M

        // Load Q tile to shared memory (stays for all K/V iterations)
        let q_tile: Tensor<T, [BLOCK_M, D]> @shared = q[b, h, q_start..+BLOCK_M, :]

        // Running softmax statistics
        var m_prev: Tensor<f32, [BLOCK_M]> @local = Tensor.fill(-inf)  // row max
        var l_prev: Tensor<f32, [BLOCK_M]> @local = Tensor.zeros()     // row sum
        var acc: Tensor<f32, [BLOCK_M, D]> @local = Tensor.zeros()     // output accumulator

        let kv_end = if causal { min(S_kv, q_start + BLOCK_M) } else { S_kv }

        for kv_start in range(0, kv_end, step = BLOCK_N) {
            // Load K, V tiles
            let k_tile = k[b, h, kv_start..+BLOCK_N, :].load(@shared)
            let v_tile = v[b, h, kv_start..+BLOCK_N, :].load(@shared)
            sync_block()

            // Compute attention scores: S = Q @ K^T / sqrt(D)
            let scores: Tensor<f32, [BLOCK_M, BLOCK_N]> @local =
                (q_tile @ k_tile.transpose()) / sqrt(D as f32)

            // Apply causal mask
            if causal {
                scores.mask_where(
                    |qi, ki| q_start + qi < kv_start + ki,
                    value = -inf,
                )
            }

            // Online softmax update
            let m_new = max(m_prev, scores.row_max())
            let correction = exp(m_prev - m_new)
            let p = exp(scores - m_new.unsqueeze(-1))

            // Rescale running accumulator and add new contribution
            acc = acc * correction.unsqueeze(-1) + p @ v_tile
            l_prev = l_prev * correction + p.row_sum()
            m_prev = m_new

            sync_block()
        }

        // Final normalization
        let output_tile = acc / l_prev.unsqueeze(-1)
        store(output_tile.cast<T>(), output[b, h, q_start..+BLOCK_M, :])
    }
}
```

### Memory Analysis

For `B=1, H=32, S=4096, D=128` in FP16:

```
Naive attention:
  Q @ K^T materialized: 4096 x 4096 x 32 heads x 4 bytes = 2 GB
  Total HBM traffic: ~6 GB

FlashAttention:
  Shared memory per block: ~96 KB (Q tile + K tile + V tile)
  Total HBM traffic: ~100 MB (each element read once)
  Speedup: ~60x memory reduction, 2-4x wall-clock speedup
```

---

## 3. Quantization

### Type-Safe Quantized Types

```vortex
// Quantized types encode the scheme in the type system
type Q8 = Quantized<i8, scale = f32, zero_point = i8>           // Per-tensor INT8
type Q4_group = Quantized<i4, scale = f16, group_size = 128>    // GPTQ-style group quant
type FP8 = f8e4m3                                                // Native FP8

// Safe conversion rules (compiler-enforced)
let weights_f16: Tensor<f16, [D, D]> = load_weights()
let weights_q4: Tensor<Q4_group, [D, D]> = quantize(weights_f16, scheme = .gptq)
// weights_q4 stores: D*D/2 bytes of int4 data + D*D/128 scale factors

// Dequantize-on-the-fly during matmul (never materializes full f16 weights)
let output: Tensor<f16, [B, D]> = matmul_quantized(input_f16, weights_q4)
```

### Quantized GEMM Kernel

```vortex
// INT4 weight-only quantization with f16 activations
@schedule(
    tile_m = 128, tile_n = 128, tile_k = 64,
    dequant = .on_the_fly,  // Dequantize as part of the GEMM, not separately
)
kernel matmul_w4a16(
    activations: Tensor<f16, [M, K]>,
    weights: Tensor<Q4_group, [K, N]>,
) -> Tensor<f16, [M, N]> {
    // Each thread block:
    // 1. Loads INT4 weight tile (half the memory bandwidth of f16)
    // 2. Dequantizes to f16 in shared memory or registers
    // 3. Computes GEMM using f16 tensor cores
    // 4. Stores f16 result

    // Effective bandwidth: 2x improvement over f16 weights
    // Quality: < 0.5 perplexity increase for most models
    ...
}
```

### Supported Quantization Formats

| Format | Bits/Weight | Scale | Group Size | Use Case |
|---|---|---|---|---|
| `f16` | 16 | - | - | Training, high-quality inference |
| `bf16` | 16 | - | - | Training (larger dynamic range) |
| `f8e4m3` | 8 | per-tensor f32 | - | H100+ inference & training |
| `i8` | 8 | per-channel f32 | - | SmoothQuant inference |
| `i4` (GPTQ) | 4 | per-group f16 | 128 | Memory-efficient inference |
| `i4` (AWQ) | 4 | per-channel f16 | 128 | Activation-aware inference |
| `Q4_K_M` | 4.5 avg | block f16 | 256 | llama.cpp-compatible |

---

## 4. Fused Operations

### RMSNorm + Residual + Quantize

```vortex
// Three operations fused into one kernel (avoids 3 HBM round-trips)
@fuse
kernel fused_norm_residual_quant(
    x: Tensor<f16, [B, S, D]>,
    residual: Tensor<f16, [B, S, D]>,
    weight: Tensor<f16, [D]>,
    eps: f32,
) -> (Tensor<f8e4m3, [B, S, D]>, Tensor<f16, [B, S, D]>) {
    parallel_for (b, s) in grid(B, S) {
        // Load x and residual (1 HBM read each)
        let row = x[b, s, :] + residual[b, s, :]   // residual connection
        let rms = sqrt(mean(row * row) + eps)        // RMSNorm
        let normed = row / rms * weight              // normalize + scale
        let quantized = normed.cast<f8e4m3>()        // quantize for next layer

        // Store normalized (for residual stream) and quantized (for next matmul)
        output_residual[b, s, :] = row       // f16, feeds into next residual
        output_quantized[b, s, :] = quantized // f8, feeds into next matmul
    }
    // Total HBM traffic: 2*B*S*D*2 bytes read + B*S*D*(2+1) bytes write
    // vs unfused: 6*B*S*D*2 bytes read + 4*B*S*D*2 bytes write
}
```

### Automatic Fusion Detection

```vortex
// The compiler automatically detects fusible patterns
fn transformer_block(x: Tensor<f16, [B, S, D]>, w: Weights) -> Tensor<f16, [B, S, D]> {
    // These operations are automatically fused by the compiler:
    let normed = rms_norm(x, w.norm, eps = 1e-6)    // ─┐
    let q = normed @ w.wq                             //  ├─ Fused: norm + 3x matmul
    let k = normed @ w.wk                             //  │
    let v = normed @ w.wv                             // ─┘

    let attn = flash_attention(q, k, v, causal = true) // FlashAttention (already fused)
    let projected = attn @ w.wo                         // Standalone GEMM

    let residual = x + projected                       // ─┐
    let ff_norm = rms_norm(residual, w.ff_norm)        //  ├─ Fused: residual + norm
    let gate = silu(ff_norm @ w.gate)                  //  │  + gate projection + SiLU
    let up = ff_norm @ w.up                            //  │
    let ff_out = gate * up                             // ─┘

    return residual + (ff_out @ w.down)
}
```

---

## 5. KV-Cache Management

### Paged KV-Cache

```vortex
import nn.cache { PagedKVCache, KVCacheConfig }

// Configuration
let config = KVCacheConfig {
    page_size: 16,           // tokens per page
    max_pages: 65536,        // total physical pages
    num_layers: 80,
    num_kv_heads: 8,         // GQA: 8 KV heads for 64 query heads
    head_dim: 128,
    dtype: f16,
}

// Create cache (allocates physical page pool)
let cache = PagedKVCache.new(config)

// During generation
fn decode_step(
    cache: &mut PagedKVCache,
    sequences: &[SequenceState],
    new_tokens: Tensor<u32, [B]>,
) {
    // Allocate pages on demand as sequences grow
    for seq in sequences {
        if seq.needs_new_page() {
            cache.allocate_page(seq.id)?
        }
    }

    // Kernel uses page table for indirect KV access
    let output = paged_attention(queries, cache, sequences)
    ...
}
```

### Paged Attention Kernel

```vortex
kernel paged_attention(
    queries: Tensor<f16, [B, H_q, 1, D]>,  // single new token per sequence
    cache: &PagedKVCache,
) -> Tensor<f16, [B, H_q, 1, D]> {
    // This is memory-bandwidth-bound: loading full KV history for each query
    // Arithmetic intensity: ~2 FLOPs/byte (far below compute bound)

    parallel_for (b, h) in grid(B, H_q) {
        let kv_head = h / (H_q / cache.num_kv_heads)  // GQA head mapping
        let seq_len = cache.seq_lengths[b]

        var m = -inf, l = 0.0
        var acc: Tensor<f32, [D]> @local = Tensor.zeros()

        // Iterate over pages in this sequence's KV cache
        for page_idx in 0..ceil_div(seq_len, cache.page_size) {
            let physical_page = cache.page_table[b, page_idx]
            let k_page = cache.k_pages[physical_page, kv_head, :, :]
            let v_page = cache.v_pages[physical_page, kv_head, :, :]

            let scores = queries[b, h, 0, :] @ k_page.transpose()
            // ... online softmax accumulation (same as FlashAttention) ...
        }

        output[b, h, 0, :] = (acc / l).cast<f16>()
    }
}
```

---

## 6. Distributed Execution

### Tensor Parallelism

```vortex
import distributed { TensorParallel, AllReduce }

// Tensor parallelism splits weight matrices across GPUs
@tensor_parallel(dim = 1, group = tp_group)
kernel linear_column_parallel(
    x: Tensor<f16, [B, D]>,
    w: Tensor<f16, [D, D_out]>,  // each GPU holds D_out/TP columns
) -> Tensor<f16, [B, D_out / TP]> {
    return x @ w  // each GPU computes its shard
}

@tensor_parallel(dim = 0, group = tp_group, reduce = .all_reduce_sum)
kernel linear_row_parallel(
    x: Tensor<f16, [B, D / TP]>,   // each GPU holds partial input
    w: Tensor<f16, [D / TP, D_out]>,
) -> Tensor<f16, [B, D_out]> {
    let partial = x @ w
    return AllReduce.sum(partial, group = tp_group)  // NCCL all-reduce
}
```

### Pipeline Parallelism

```vortex
// Multi-GPU pipeline with interleaved 1F1B scheduling
pipeline llm_pipeline(
    input: Tensor<u32, [B, S]>,
    weights: DistributedWeights,
) -> Tensor<f32, [B, S, V]> {
    // 4-stage pipeline across 4 GPUs
    stage @gpu(0) {
        let x = embed(input, weights.embedding)
        for layer in weights.layers[0..20] {
            x = transformer_block(x, layer)
        }
    }
    stage @gpu(1) {
        for layer in weights.layers[20..40] {
            x = transformer_block(x, layer)
        }
    }
    stage @gpu(2) {
        for layer in weights.layers[40..60] {
            x = transformer_block(x, layer)
        }
    }
    stage @gpu(3) {
        for layer in weights.layers[60..80] {
            x = transformer_block(x, layer)
        }
        let logits = rms_norm(x, weights.final_norm) @ weights.lm_head
        return logits
    }
}
```

### Expert Parallelism (MoE)

```vortex
import distributed { AllToAll, ExpertParallel }

// Mixture of Experts with load-balanced routing
@expert_parallel(group = ep_group)
kernel moe_layer(
    x: Tensor<f16, [B * S, D]>,
    gate: Tensor<f16, [D, num_experts]>,
    experts: [FFN; num_experts],  // distributed across GPUs
    top_k: comptime u32 = 2,
) -> Tensor<f16, [B * S, D]> {
    // 1. Compute routing scores
    let scores = softmax(x @ gate, dim = -1)
    let (expert_ids, expert_weights) = top_k_gating(scores, top_k)

    // 2. All-to-all dispatch: send tokens to their assigned expert's GPU
    let dispatched = AllToAll.dispatch(x, expert_ids, group = ep_group)

    // 3. Local expert computation (each GPU processes its own experts)
    let processed = local_expert_forward(dispatched, experts[my_expert_range])

    // 4. All-to-all combine: return processed tokens
    let combined = AllToAll.combine(processed, expert_ids, group = ep_group)

    // 5. Weighted sum of expert outputs
    return weighted_sum(combined, expert_weights)
}
```

---

## 7. Speculative Decoding

```vortex
import nn.speculative { draft_and_verify, SpeculativeConfig }

// Speculative decoding: small model drafts, large model verifies in parallel
fn speculative_decode(
    target_model: &LLM,
    draft_model: &LLM,
    prompt: Tensor<u32, [S]>,
    config: SpeculativeConfig,
) -> Tensor<u32, [S + max_new_tokens]> {
    var tokens = prompt
    var kv_cache_target = PagedKVCache.new(...)
    var kv_cache_draft = PagedKVCache.new(...)

    while tokens.len() < S + max_new_tokens {
        // Step 1: Draft model generates K candidate tokens (fast, autoregressive)
        let draft_tokens = draft_model.generate(
            tokens, &mut kv_cache_draft,
            num_tokens = config.num_draft_tokens,  // typically 4-8
        )

        // Step 2: Target model verifies all K tokens in parallel (one forward pass)
        let target_logits = target_model.forward(
            concat(tokens, draft_tokens), &mut kv_cache_target,
        )

        // Step 3: Accept longest matching prefix (modified rejection sampling)
        let accepted = verify_and_sample(
            draft_tokens, target_logits,
            temperature = config.temperature,
        )

        tokens = concat(tokens, accepted)

        // Expected accepted: ~3-5 tokens per verify step for well-matched models
        // Net speedup: 2-3x for latency-sensitive single-request serving
    }
}
```

---

## 8. Model Loading and Serving

```vortex
import nn.io { load_safetensors, load_gguf }
import nn.serve { Server, ServerConfig }

// Load model weights in various formats
let weights = match format {
    .safetensors => load_safetensors("model.safetensors", dtype = .f16),
    .gguf => load_gguf("model.gguf"),  // preserves quantization from file
}

// Configure serving
let server = Server.new(ServerConfig {
    model: weights,
    max_batch_size: 64,
    max_seq_len: 8192,
    kv_cache_pages: 65536,
    tensor_parallel: 4,      // 4-way TP across GPUs 0-3
    continuous_batching: true,
    speculative_decoding: SpeculativeConfig {
        draft_model: "draft-model.safetensors",
        num_draft_tokens: 5,
    },
})

// Start OpenAI-compatible API server
server.listen("0.0.0.0:8080")
```
