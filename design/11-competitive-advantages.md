# 11 — What's Hard Today (and Why Vortex Can Win)

A technical analysis of unsolved problems in GPU programming that Vortex is uniquely positioned to address. These are not incremental improvements over CUDA — they are structural gaps that require a new language to fix.

---

## 1. Sparse and Irregular Computation on GPU

### The Problem

GPUs are designed for dense, uniform computation. When workloads are sparse or irregular, programmers are forced into one of two bad choices: waste compute on padding, or write custom CUDA kernels that fight the hardware.

**Spiking Neural Networks (SNNs):** Biological neurons fire sparsely — typically 1-5% of neurons are active at any timestep. In CUDA, you must either (a) process all neurons including the 95% that are silent, or (b) write a custom kernel that compacts active indices, dispatches only active neurons, and scatters results back. Option (a) wastes 95% of compute. Option (b) requires a different custom kernel for every network topology and sparsity pattern.

```python
# PyTorch: the "easy" way — process everything, waste 95% of compute
spikes = (membrane_potential > threshold).float()  # dense tensor, mostly zeros
output = sparse_weight_matrix @ spikes  # cuSPARSE, but format conversion overhead
```

The fundamental issue: PyTorch's `torch.sparse` tensors are second-class citizens. Most operations fall back to dense computation. JAX's sparse support (`jax.experimental.sparse`) has been "experimental" since 2021.

**Dynamic sparsity in Mixture-of-Experts (MoE):** Top-k routing sends each token to only 2 of 64 experts. This requires:
1. Computing routing scores (dense matmul — fast)
2. Top-k selection per token (ok)
3. Scatter tokens to their assigned experts (terrible — irregular memory access)
4. Process tokens per expert (variable batch sizes — load imbalanced)
5. Gather results back (terrible again)

DeepSeek-V3's MoE implementation required custom PTX for the scatter/gather steps because no framework primitive could handle the irregular communication pattern efficiently. Their DeepEP library exists solely because PyTorch cannot express this pattern without multiple kernel launches and synchronizations.

**Variable-length sequences:** Transformer inference serves requests with different sequence lengths. The standard approach is padding to the maximum length in the batch. For a batch with lengths [47, 128, 2048, 15], padding to 2048 wastes 73% of compute. PyTorch's `NestedTensor` (introduced in 2022, still partially functional in 2026) and JAX's ragged tensors remain incomplete — most operations fall back to padded dense computation.

### How Vortex Solves It

Vortex treats sparsity as a first-class compiler concern, not a library afterthought. The compiler manages format conversion, compaction, and load balancing automatically.

```vortex
// Spiking neural network: compiler-managed sparse execution
// The SparseEvent type carries sparsity metadata in the type system
kernel snn_step(
    membrane: &mut Tensor<f32, [N]>,
    weights: SparseTensor<f32, [N, N]>,  // static structural sparsity
    input_current: Tensor<f32, [N]>,
    threshold: f32,
) -> SparseEvent<u32, [?K]> {
    // Update membrane potential (dense — all neurons)
    parallel_for i in 0..N {
        membrane[i] = 0.9 * membrane[i] + input_current[i]
    }

    // Detect spikes — compiler generates compact stream
    // ?K is an existential: the number of spikes is data-dependent
    let spikes: SparseEvent<u32, [?K]> = compact_where(
        |i| membrane[i] > threshold
    )

    // Process ONLY spiking neurons — compiler generates:
    // 1. Prefix sum to compute output indices
    // 2. Scatter kernel with coalesced writes
    // 3. Sparse-dense matmul using only active rows
    parallel_for spike in spikes {
        membrane[spike.index] = 0.0  // reset
    }

    return spikes
}

// MoE routing with compiler-managed dispatch
kernel moe_forward<const E: usize, const K: usize>(
    tokens: Tensor<f16, [B, S, D]>,
    gate_weights: Tensor<f16, [D, E]>,
    experts: [FFN; E],
) -> Tensor<f16, [B, S, D]> {
    // Routing scores
    let scores = softmax(tokens @ gate_weights, dim = -1)
    let (expert_ids, expert_weights) = top_k(scores, K)

    // @balanced_dispatch: compiler generates load-balanced scatter
    // Groups tokens by expert, pads to warp-aligned sizes, dispatches
    // No manual scatter/gather — the compiler handles irregular access
    @balanced_dispatch(experts = E, top_k = K)
    let results = dispatch_to_experts(tokens, expert_ids, experts)

    // Weighted combination with compiler-managed gather
    return weighted_gather(results, expert_ids, expert_weights)
}

// Variable-length sequences: native ragged tensor support
kernel batched_attention(
    queries: RaggedTensor<f16, [B, ?, D]>,   // ? = variable per batch
    keys: RaggedTensor<f16, [B, ?, D]>,
    values: RaggedTensor<f16, [B, ?, D]>,
) -> RaggedTensor<f16, [B, ?, D]> {
    // Compiler generates per-sequence launch with:
    // 1. No padding — each sequence uses exactly its length
    // 2. Dynamic block sizing based on actual sequence length
    // 3. Automatic load balancing across SMs
    parallel_for (b, q_block) in ragged_grid(queries) {
        let seq_len = queries.len(b)
        // FlashAttention inner loop uses actual seq_len, not padded max
        let output = flash_attention_block(
            queries[b], keys[b], values[b], seq_len,
        )
        store(output, results[b])
    }
}
```

The key insight: the compiler sees `SparseEvent`, `RaggedTensor`, and `@balanced_dispatch` as optimization directives. It generates the compaction, scatter/gather, and load-balancing kernels automatically — these are not library functions but compiler-generated code paths that fuse with surrounding computation.

---

## 2. Adaptive/Dynamic Control Flow on GPU

### The Problem

GPUs execute in SIMT lockstep. When threads within a warp need different iteration counts or control paths, the hardware serializes both paths (warp divergence). This is catastrophic for algorithms where the "right" number of iterations is data-dependent.

**Neural ODEs:** Neural ODEs solve `dh/dt = f(h, t)` where `f` is a neural network. Adaptive solvers (Dormand-Prince, RK45) choose step sizes based on local error estimates. Different samples in a batch converge at different rates — one sample might need 12 steps while another needs 200. In CUDA, you either:
- Run all samples for max(steps) — wasting 95% of compute on already-converged samples
- Implement complex sample-level bookkeeping to retire converged samples — which is a custom kernel that nobody wants to maintain

```python
# PyTorch (torchdiffeq): all samples step together
# If sample 0 converges in 12 steps and sample 31 needs 200,
# sample 0 computes 188 wasted steps
from torchdiffeq import odeint_adjoint
y = odeint_adjoint(func, y0, t, method='dopri5')  # no per-sample early exit
```

**Early exit / conditional computation:** Modern efficiency techniques (SkipNet, early exit classifiers, adaptive depth) require per-token decisions about which layers to execute. In a transformer with 32 layers, a "confident" token might exit at layer 8 while uncertain tokens continue to layer 32. PyTorch has no mechanism for this. You run all tokens through all layers and mask at the end, computing 4x more than necessary.

**Liquid Neural Networks / Continuous-time models:** Liquid Time-Constant (LTC) networks have per-neuron, per-sample time constants. Each neuron's ODE evolves at its own rate. This is a worst-case scenario for GPU SIMT: every thread in a warp potentially needs a different number of iterations to reach the target time.

### How Vortex Solves It

Vortex's MLIR-based compiler uses the `scf` (Structured Control Flow) dialect to restructure divergent control flow into GPU-friendly patterns. The programmer writes the natural algorithm; the compiler handles the mapping.

```vortex
// Neural ODE with per-sample adaptive stepping
// Compiler restructures into warp-cooperative execution
kernel neural_ode_solve(
    y0: Tensor<f32, [B, D]>,
    f: fn(Tensor<f32, [D]>, f32) -> Tensor<f32, [D]>,
    t_start: f32,
    t_end: f32,
    atol: f32,
    rtol: f32,
) -> Tensor<f32, [B, D]> {
    // @adaptive: compiler generates warp-level retirement protocol
    // When a sample converges, its thread participates in useful work
    // for other samples (work stealing within warp)
    @adaptive(retirement = .warp_cooperative)
    parallel_for b in 0..B {
        var y = y0[b]
        var t = t_start
        var dt = (t_end - t_start) / 100.0  // initial step

        while t < t_end {
            // RK45 step
            let (y_new, error) = rk45_step(f, y, t, dt)

            // Adaptive step control — different per sample
            if error < atol + rtol * norm(y_new) {
                y = y_new
                t += dt
                dt = dt * min(5.0, (0.9 * (atol / error)) ** 0.2)
            } else {
                dt = dt * max(0.1, (0.9 * (atol / error)) ** 0.2)
            }
        }

        result[b] = y
    }
}

// Early exit transformer: per-token conditional depth
kernel adaptive_transformer(
    x: Tensor<f16, [B, S, D]>,
    layers: [TransformerLayer; L],
    exit_threshold: f32,
) -> Tensor<f16, [B, S, D]> {
    // @conditional_depth: compiler generates two-phase execution
    // Phase 1: run all tokens through layers 0..min_depth
    // Phase 2: compact "still-active" tokens, continue on subset
    // Avoids wasted compute on confident tokens
    @conditional_depth(min_depth = 4, granularity = .token)
    parallel_for (b, s) in grid(B, S) {
        var hidden = x[b, s, :]
        var confidence: f32 = 0.0

        for layer_idx in 0..L {
            hidden = layers[layer_idx].forward(hidden)

            // Per-token exit decision
            confidence = exit_classifier(hidden)
            if confidence > exit_threshold {
                break  // compiler handles retirement + compaction
            }
        }

        output[b, s, :] = hidden
    }
}

// Liquid neural network with per-neuron time constants
kernel liquid_network_step(
    state: &mut Tensor<f32, [B, N]>,
    input: Tensor<f32, [B, D]>,
    weights: LiquidWeights,
    tau: Tensor<f32, [N]>,         // per-neuron time constants
    target_time: f32,
) {
    // @divergent_convergence: compiler bins neurons by time constant
    // Neurons with similar tau are grouped into the same warps
    // Reduces warp divergence from O(N) to O(log(max_tau/min_tau))
    @divergent_convergence(bin_by = tau, tolerance = 0.1)
    parallel_for (b, n) in grid(B, N) {
        var s = state[b, n]
        var t = 0.0
        let dt_base = tau[n] / 10.0  // step proportional to time constant

        while t < target_time {
            let activation = sigmoid(weights.W @ state[b, :] + weights.U @ input[b, :])
            let ds = (-s + activation[n]) / tau[n]
            s += ds * dt_base
            t += dt_base
        }

        state[b, n] = s
    }
}
```

The compiler transforms these patterns using three strategies:
1. **Warp-cooperative retirement**: converged threads help execute remaining work for active threads in their warp
2. **Compaction passes**: periodically compact active work items into dense warps, eliminating idle threads
3. **Binning**: sort work items by expected iteration count before dispatch, minimizing divergence

These transforms happen at the `scf` dialect level in MLIR, so they apply uniformly regardless of backend (NVIDIA or AMD).

---

## 3. Fused Custom Operations

### The Problem

Modern ML inference is bottlenecked by memory bandwidth, not compute. An H100 has 3.35 TB/s memory bandwidth and ~990 TFLOPS (FP16). The arithmetic intensity crossover is ~295 FLOPs/byte. Most operations in a transformer (normalization, residual connections, activation functions, quantization) are memory-bound — they read data from HBM, do trivial math, and write it back. Each separate kernel launch pays the full HBM round-trip cost.

**FlashAttention as a cautionary tale:** FlashAttention fuses the entire attention computation (Q@K^T, scaling, masking, softmax, dropout, @V) into a single kernel that keeps intermediate results in SRAM. It took Tri Dao approximately 6 months of full-time work to write the first CUDA implementation (FlashAttention-1, 2022), then another year for FlashAttention-2. FlashAttention-3 (Hopper-specific) required understanding WGMMA, TMA, and warp specialization at the SASS level. A compiler should be able to derive this fusion automatically from the mathematical specification.

**The multi-kernel tax in PyTorch:** A standard transformer block in PyTorch:
```python
# Each line = separate CUDA kernel launch + HBM round-trip
normed = rms_norm(x, weight)       # read x (2 bytes/elem), write normed (2 bytes/elem)
q = normed @ W_q                    # read normed + W_q, write q
k = normed @ W_k                    # read normed + W_k, write k  (normed read AGAIN)
v = normed @ W_v                    # read normed + W_v, write v  (normed read AGAIN)
attn = flash_attention(q, k, v)    # (this one is fused, thankfully)
projected = attn @ W_o             # read attn + W_o, write projected
residual = x + projected           # read x + projected, write residual (x read AGAIN)
normed2 = rms_norm(residual, w2)   # read residual, write normed2
gate = silu(normed2 @ W_gate)      # read normed2 + W_gate, write tmp, read tmp, write gate
up = normed2 @ W_up               # read normed2 + W_up, write up (normed2 read AGAIN)
ff_out = gate * up                 # read gate + up, write ff_out
out = residual + ff_out @ W_down   # multiple reads and writes
```

That is 10+ kernel launches, with `normed` read 3 times, `residual` read twice, and `normed2` read twice from HBM. Each redundant read wastes ~B*S*D*2 bytes of bandwidth.

**Custom activation functions:** If a researcher invents a new activation function (say, `f(x) = x * sigmoid(beta * x + gamma)`), they have three options in PyTorch:
1. Implement in Python — autograd works but 10-100x slower than native CUDA
2. Write it as a `torch.autograd.Function` with manual backward — still Python overhead
3. Write a CUDA kernel — requires CUDA expertise, manual gradient computation, and integration with PyTorch's dispatch system

None of these options fuse the activation with adjacent operations.

### How Vortex Solves It

Vortex's compiler performs automatic kernel fusion through MLIR's `TileAndFuse` infrastructure, enhanced with domain-specific fusion rules.

```vortex
// Write the math. The compiler fuses automatically.
fn transformer_block(
    x: Tensor<f16, [B, S, D]>,
    w: TransformerWeights<f16>,
) -> Tensor<f16, [B, S, D]> {
    // Compiler's fusion analysis sees this entire function body
    // and generates minimal kernel launches:

    // Fused kernel 1: RMSNorm + QKV projection
    // (normed never written to HBM — stays in registers/shared memory)
    let normed = rms_norm(x, w.norm, eps = 1e-6)
    let q = normed @ w.wq
    let k = normed @ w.wk
    let v = normed @ w.wv

    // Fused kernel 2: FlashAttention (already fused by design)
    let attn = flash_attention(q, k, v, causal = true)

    // Fused kernel 3: output projection + residual + RMSNorm + SwiGLU FFN
    let projected = attn @ w.wo
    let residual = x + projected
    let ff_norm = rms_norm(residual, w.ff_norm)
    let gate = silu(ff_norm @ w.gate) .* (ff_norm @ w.up)
    let ff_out = gate @ w.down

    return residual + ff_out
    // Total: 3 kernel launches instead of 12+
    // Bandwidth saved: ~40% reduction in HBM traffic
}

// Custom activation: just write it. Compiler fuses with adjacent ops.
fn swish_beta(x: Tensor<f32, [N]>, beta: f32, gamma: f32) -> Tensor<f32, [N]> {
    return x .* sigmoid(beta * x + gamma)
}

// This entire chain compiles to ONE kernel:
@fuse
fn my_layer(x: Tensor<f16, [B, D]>, w: Tensor<f16, [D, D]>) -> Tensor<f16, [B, D]> {
    let projected = x @ w                          // GEMM epilogue fusion:
    let normed = rms_norm(projected, weight)       // norm is fused into GEMM epilogue
    let activated = swish_beta(normed, 1.5, 0.1)  // activation fused into same epilogue
    return activated
    // The GEMM writes directly to the output buffer after applying
    // normalization and activation IN REGISTERS. Zero extra HBM traffic.
}

// Even complex multi-output patterns fuse correctly:
@fuse
kernel fused_norm_residual_quantize(
    x: Tensor<f16, [B, S, D]>,
    residual: Tensor<f16, [B, S, D]>,
    weight: Tensor<f16, [D]>,
    eps: f32,
) -> (Tensor<f8e4m3, [B, S, D]>, Tensor<f16, [B, S, D]>) {
    parallel_for (b, s) in grid(B, S) {
        let row = x[b, s, :] + residual[b, s, :]
        let rms = sqrt(mean(row .* row) + eps)
        let normed = row / rms .* weight
        let quantized = normed.cast<f8e4m3>()

        // Two outputs from one kernel: residual stream + quantized for next layer
        output_residual[b, s, :] = row
        output_quantized[b, s, :] = quantized
    }
    // HBM traffic: read 2 * B*S*D*2 bytes, write B*S*D*(2+1) bytes
    // Unfused would be: read 6*B*S*D*2, write 4*B*S*D*2 bytes
    // 3.6x bandwidth reduction
}
```

The fusion system works at the MLIR level: the `KernelFusionAnalysis` pass identifies producer-consumer chains where intermediate results can stay in registers or shared memory. The `@fuse` annotation is a hint, not a requirement — the compiler can fuse automatically but the annotation guarantees it or produces a compile error explaining why fusion failed.

---

## 4. Mixed Paradigm: Crypto + ML

### The Problem

No existing framework supports both cryptographic and machine learning workloads. This is not a theoretical concern — it is an immediate practical gap:

**Verifiable inference:** As LLMs are deployed in high-stakes settings (medical, legal, financial), users need proof that a specific model produced a specific output. This requires either:
- ZK proofs of neural network execution (zkML) — requires field arithmetic + matrix multiplication in the same computation
- Trusted execution environments (TEEs) — hardware-dependent, limited GPU support

Current zkML projects (EZKL, Daniel Kang's work) compile PyTorch models to arithmetic circuits, then prove them using separate ZK libraries. The "compilation" step is a lossy transformation that loses GPU acceleration. A 7B parameter model would require astronomical proof generation time.

**Federated learning with privacy guarantees:** Secure aggregation protocols use homomorphic encryption or secret sharing to aggregate model gradients without exposing individual updates. This requires:
1. Train a model (ML framework — PyTorch/JAX)
2. Encrypt gradients (crypto library — SEAL/HElib/lattigo)
3. Aggregate encrypted gradients (custom code bridging both)
4. Decrypt aggregate (crypto library)
5. Update model (ML framework)

Steps 1 and 2 run on different systems, with data copied between GPU and CPU at every boundary. No framework handles both.

**Constant-time ML for side-channel resistance:** If an ML model processes secret data (e.g., biometric authentication on-device), the inference time should not depend on the input. In PyTorch, early exit, dynamic batching, and input-dependent sparsity all create timing side channels. There is no `@constant_time` guarantee.

### How Vortex Solves It

Vortex's unified type system handles both domains natively because it was designed for both from day one.

```vortex
import crypto.fields.bn254 { Fr }
import crypto.zk.groth16 { Proof, prove }
import nn.attention { flash_attention }
import nn.linear { matmul }

// Verifiable inference: prove that a model produced a specific output
// The SAME language, SAME GPU, SAME kernel fusion
fn verifiable_inference(
    model: &TransformerWeights<Fr>,    // Model weights as field elements
    input: Tensor<Fr, [1, S, D]>,      // Input encoded as field elements
    proving_key: &ProvingKey,
) -> (Tensor<Fr, [1, S, V]>, Proof) {
    // Forward pass in the field (same operations, different number type)
    let q = input @ model.wq   // Field<P> matmul — uses NTT-based multiplication
    let k = input @ model.wk
    let v = input @ model.wv

    let attn = attention_over_field(q, k, v)  // No softmax (not field-friendly)
                                                // Use polynomial approximation instead
    let logits = attn @ model.lm_head

    // Generate ZK proof of the computation
    let witness = extract_witness(input, logits, model)
    let proof = prove(proving_key, witness)

    return (logits, proof)
}

// Secure biometric matching with constant-time guarantee
@constant_time
kernel secure_face_match(
    template: Tensor<Secret<f32>, [128]>,    // Stored face embedding (secret)
    probe: Tensor<f32, [128]>,                // New face embedding (public)
    threshold: f32,
) -> Secret<bool> {
    // Cosine similarity — constant time because all operations
    // on Secret values use ct_select instead of branches
    var dot: Secret<f32> = 0.0
    var norm_a: Secret<f32> = 0.0
    let norm_b: f32 = 0.0

    for i in 0..128 {
        dot += template[i] * probe[i]
        norm_a += template[i] * template[i]
        norm_b += probe[i] * probe[i]
    }

    let similarity = dot / (sqrt(norm_a) * sqrt(norm_b))

    // ct_gt: constant-time greater-than comparison
    // Returns Secret<bool> — cannot be used in if/else
    return ct_gt(similarity, threshold)
}

// Homomorphic encryption + ML: encrypted inference
import crypto.fhe.ckks { CKKSCiphertext, encode, multiply_plain, rescale }

fn encrypted_linear_layer(
    encrypted_input: CKKSCiphertext<12>,          // encrypted activations
    plaintext_weights: Tensor<f64, [D_in, D_out]>, // public weights
) -> CKKSCiphertext<11> {
    // Encode weights into CKKS plaintext
    let encoded_weights = encode(plaintext_weights, scale = encrypted_input.scale)

    // Homomorphic matrix-vector multiply (NTT-based polynomial arithmetic)
    // This uses the SAME GPU, SAME NTT kernels as ZK proofs
    let product = multiply_plain(encrypted_input, encoded_weights)

    // Rescale to manage ciphertext noise (level drops: 12 → 11)
    return rescale(product)
    // Vortex generates a single fused kernel for encode + multiply + rescale
}

// The @constant_time annotation is compiler-verified:
@constant_time
fn bad_example(secret: Secret<u32>) -> u32 {
    if secret > 10 {   // COMPILE ERROR: cannot branch on Secret value
        return 1       // "Branching on Secret<u32> creates a timing side channel.
    }                   //  Use ct_select() for constant-time conditional."
    return 0
}
```

The key advantage: `Field<P>`, `Secret<T>`, `CKKSCiphertext`, and `Tensor<f16>` all live in the same type system, use the same GPU memory model, and benefit from the same kernel fusion. There is no "bridge" between crypto and ML — they are different types in the same language.

---

## 5. Type-Safe Tensor Programming

### The Problem

Shape errors are the most common bug in deep learning code, and no existing tool catches them at compile time.

**PyTorch:** All shape checking is runtime.
```python
# This looks fine. It crashes at runtime.
q = torch.randn(batch, seq_len, d_model)
k = torch.randn(batch, seq_len, d_model)
attention = q @ k  # RuntimeError: mat1 and mat2 shapes cannot be multiplied
                    # (seq_len x d_model and seq_len x d_model)
                    # Should be: q @ k.transpose(-2, -1)
```

This error is discovered only when the code runs, potentially after hours of training. In production, a shape mismatch in a rarely-executed code path can crash a serving system.

**JAX:** Same problem, runtime errors. `jax.numpy.matmul` raises at execution time, not at trace time, because shapes can be symbolic (dynamic).

**Triton:** Raw pointers with no shape information at all. A Triton kernel that reads out of bounds produces silent garbage, not an error.

**The sad state of static shape checking:** Multiple projects have attempted to add shape checking to Python:
- `torchtyping` (2021): Abandoned
- `jaxtyping` (2022): Runtime checks only, not compile-time
- `beartype` + shape annotations: Runtime overhead, incomplete coverage
- `einops`: Better notation but still runtime errors

None of these can catch the `q @ k` vs `q @ k.T` bug at compile time because Python's type system cannot express "this tensor has shape [B, S, D]."

### How Vortex Solves It

Vortex has dependent types for tensor shapes, checked at compile time. Shape errors are compile errors, not runtime crashes.

```vortex
// Shapes are part of the type. Mismatches are compile errors.
fn multi_head_attention<B: Nat, S: Nat, D: Nat, H: Nat>(
    q: Tensor<f16, [B, S, D]>,
    k: Tensor<f16, [B, S, D]>,
    v: Tensor<f16, [B, S, D]>,
) -> Tensor<f16, [B, S, D]>
where D % H == 0  // compile-time constraint: D must be divisible by H
{
    let head_dim: Nat = D / H

    // reshape is type-checked: product of dimensions must match
    let q_heads: Tensor<f16, [B, H, S, head_dim]> = reshape(q)
    let k_heads: Tensor<f16, [B, H, S, head_dim]> = reshape(k)
    let v_heads: Tensor<f16, [B, H, S, head_dim]> = reshape(v)

    // @ operator: inner dimensions checked at compile time
    // q_heads[B, H, S, head_dim] @ k_heads.T[B, H, head_dim, S]
    //                   ^^^^^^^^              ^^^^^^^^
    //                   these must match — compiler verifies
    let scores: Tensor<f32, [B, H, S, S]> =
        (q_heads @ k_heads.transpose(-2, -1)) / sqrt(head_dim as f32)

    let weights = softmax(scores, dim = -1)

    // weights[B, H, S, S] @ v_heads[B, H, S, head_dim] → [B, H, S, head_dim]
    let attended: Tensor<f16, [B, H, S, head_dim]> = weights @ v_heads

    return reshape(attended)  // [B, H, S, head_dim] → [B, S, D]
                               // product check: B*H*S*head_dim == B*S*D ✓ (since D = H * head_dim)
}

// THE BUG IS CAUGHT AT COMPILE TIME:
fn broken_attention<B: Nat, S: Nat, D: Nat>(
    q: Tensor<f16, [B, S, D]>,
    k: Tensor<f16, [B, S, D]>,
) -> Tensor<f16, [B, S, S]> {
    return q @ k  // COMPILE ERROR: cannot multiply [B, S, D] @ [B, S, D]
                   //   inner dimensions D and S do not match
                   //   hint: did you mean q @ k.transpose(-2, -1)?
}

// Existential sizes for data-dependent shapes
fn top_k_tokens<B: Nat, S: Nat, D: Nat>(
    x: Tensor<f16, [B, S, D]>,
    scores: Tensor<f32, [B, S]>,
    k: Nat,
) -> Tensor<f16, [B, k, D]> {
    // k is known at compile time — output shape is fully determined
    let indices: Tensor<u32, [B, k]> = argmax_top_k(scores, k)
    return gather(x, dim = 1, indices = indices)
    // Return type [B, k, D] is verified by the compiler
}

// Compile-time constraint checking
fn conv2d<B: Nat, C_in: Nat, H: Nat, W: Nat, C_out: Nat, K: Nat>(
    input: Tensor<f32, [B, C_in, H, W]>,
    filter: Tensor<f32, [C_out, C_in, K, K]>,
    stride: Nat,
    padding: Nat,
) -> Tensor<f32, [B, C_out, (H + 2*padding - K) / stride + 1,
                              (W + 2*padding - K) / stride + 1]>
where
    (H + 2*padding - K) % stride == 0,  // output height must be integer
    (W + 2*padding - K) % stride == 0,  // output width must be integer
{
    // Shape arithmetic is Presburger arithmetic — decidable!
    ...
}
```

The type checker uses Presburger arithmetic (addition, subtraction, divisibility) for shape constraints, which is decidable — no termination issues, no proof obligations. The programmer gets full shape checking with no annotation burden beyond writing the type signatures they would write anyway.

---

## 6. Non-Backprop Training

### The Problem

Backpropagation dominates ML training, but several promising training algorithms do not fit the autograd paradigm. Existing frameworks make these algorithms painful or impossible to implement efficiently.

**Forward-Forward Algorithm (Hinton, 2022):** Each layer has a local "goodness" function. Positive data pushes goodness up; negative data pushes goodness down. There is no backward pass — each layer trains independently. In PyTorch, you must manually break the computation graph, create per-layer optimizers, and manage separate forward passes for positive and negative data. The framework's entire architecture assumes a single global loss flowing backward.

```python
# PyTorch Forward-Forward: fighting the framework
for layer in model.layers:
    # Must detach to prevent autograd from building a graph
    pos_h = layer(pos_h.detach())
    neg_h = layer(neg_h.detach())

    # Manual per-layer loss and optimizer step
    pos_goodness = pos_h.pow(2).mean(1)
    neg_goodness = neg_h.pow(2).mean(1)
    loss = F.softplus(-pos_goodness + threshold).mean() + \
           F.softplus(neg_goodness - threshold).mean()

    layer_optimizer.zero_grad()
    loss.backward()  # backward through ONE layer only
    layer_optimizer.step()
```

This works but is slow — each `.backward()` call launches kernels for a single layer. The overhead of Python-level loop, optimizer step, and kernel launch per layer dominates.

**Hebbian / STDP learning (spiking neural networks):** Spike-Timing Dependent Plasticity updates synapses based on the relative timing of pre- and post-synaptic spikes. This is a local, per-synapse rule — the antithesis of backpropagation's global error signal. In PyTorch, there is no way to express "update weight[i,j] based on the timing difference between neuron i's spike and neuron j's spike" as a differentiable operation. You must write custom CUDA kernels for every STDP variant.

**Predictive Coding:** A hierarchy of modules, each predicting the activity of the level below. Errors propagate locally (up one level), not globally. The update rule is: minimize prediction error at each level independently. PyTorch's autograd cannot represent this because the "loss" is distributed — there is no single scalar to `.backward()` from.

### How Vortex Solves It

Vortex provides first-class local learning primitives. Not everything flows through a global autograd tape.

```vortex
// Forward-Forward: first-class local learning
// The @local_learning annotation tells the compiler to generate
// per-layer update kernels, not a global backward pass
@local_learning
fn forward_forward_train(
    layers: &mut [DenseLayer; L],
    pos_data: Tensor<f16, [B, D]>,
    neg_data: Tensor<f16, [B, D]>,
    threshold: f32,
    lr: f32,
) {
    var pos_h = pos_data
    var neg_h = neg_data

    for i in 0..L {
        // Forward pass through layer i
        pos_h = layers[i].forward(pos_h)
        neg_h = layers[i].forward(neg_h)

        // Local goodness-based update — compiler generates fused kernel:
        // 1. Compute goodness for positive and negative samples
        // 2. Compute gradient of local loss w.r.t. this layer's weights
        // 3. Apply gradient update
        // All in ONE kernel, no Python overhead, no autograd tape
        local_update(layers[i]) {
            let pos_goodness = sum(pos_h .* pos_h, dim = -1)
            let neg_goodness = sum(neg_h .* neg_h, dim = -1)
            let loss = mean(softplus(-pos_goodness + threshold))
                     + mean(softplus(neg_goodness - threshold))
            return loss
        }

        // Normalize activations for next layer (fused into the kernel above)
        pos_h = pos_h / norm(pos_h, dim = -1, keepdim = true)
        neg_h = neg_h / norm(neg_h, dim = -1, keepdim = true)
    }
}

// STDP learning rule: per-synapse updates based on spike timing
kernel stdp_update(
    weights: &mut SparseTensor<f32, [N, N]>,
    pre_spike_times: Tensor<f32, [N]>,     // last spike time of presynaptic neuron
    post_spike_times: Tensor<f32, [N]>,    // last spike time of postsynaptic neuron
    a_plus: f32,    // LTP magnitude
    a_minus: f32,   // LTD magnitude
    tau_plus: f32,  // LTP time constant
    tau_minus: f32, // LTD time constant
) {
    // Parallel over all non-zero synapses
    // SparseTensor provides efficient iteration over structural non-zeros
    parallel_for (i, j, w) in weights.nonzero_entries() {
        let dt = post_spike_times[j] - pre_spike_times[i]

        // Spike-timing dependent update
        // No autograd needed — this IS the learning rule
        let dw = if dt > 0.0 {
            a_plus * exp(-dt / tau_plus)     // LTP: pre before post
        } else {
            -a_minus * exp(dt / tau_minus)   // LTD: post before pre
        }

        weights[i, j] = clamp(w + dw, 0.0, 1.0)
    }
}

// Predictive coding: hierarchical local error minimization
@local_learning
fn predictive_coding_update(
    hierarchy: &mut [PredictiveLayer; L],
    input: Tensor<f32, [B, D_0]>,
    num_iterations: u32,   // inference iterations to converge
) {
    // Initialize representations
    var representations: [Tensor<f32, [B, ?]>; L + 1]
    representations[0] = input

    // Iterative inference: minimize prediction errors locally
    for iter in 0..num_iterations {
        for level in 0..L {
            let prediction = hierarchy[level].predict(representations[level + 1])
            let error = representations[level] - prediction

            // Update representation at this level to reduce error
            // This is a LOCAL update — no global backward pass
            representations[level + 1] += hierarchy[level].inference_lr * (
                hierarchy[level].predict_backward(error)
                - hierarchy[level].precision * representations[level + 1]
            )
        }
    }

    // Weight update: minimize prediction errors at each level
    for level in 0..L {
        let error = representations[level]
                  - hierarchy[level].predict(representations[level + 1])

        // local_update generates per-level gradient kernel
        local_update(hierarchy[level]) {
            return mean(sum(error .* error, dim = -1))
        }
    }
}
```

The `@local_learning` annotation and `local_update` block are Vortex primitives that tell the compiler: "compute gradients of this local loss with respect to this layer's parameters only." The compiler generates efficient per-layer update kernels without maintaining a global autograd tape, eliminating the memory overhead of storing activations for all layers simultaneously.

---

## 7. Hardware Abstraction Without Performance Loss

### The Problem

The GPU ecosystem is fragmenting. NVIDIA, AMD, and Intel each have different:
- Instruction sets (PTX/SASS, AMDGCN, SPIR-V)
- Memory hierarchies (SMEM vs LDS vs SLM, different sizes and banking)
- Tensor core shapes (WMMA 16x16x16, MFMA 32x32x8, XMX 16x16x16)
- Warp sizes (32 vs 64 vs 32)
- Async copy mechanisms (TMA, buffer_load, LSC)

**CUDA's lock-in:** Writing a FlashAttention kernel in CUDA means it runs on NVIDIA only. Porting to AMD requires rewriting for HIP (which is ~90% compatible but diverges on shared memory, warp primitives, and tensor cores). Intel support requires yet another rewrite.

**Triton's promise and reality:** Triton claims multi-vendor support, but as of 2026:
- NVIDIA support is excellent (Triton generates competitive PTX)
- AMD support exists but with performance gaps (10-30% slower than hand-tuned HIP on MI300X for some workloads)
- Intel support is minimal
- The `tl.dot` abstraction hides tensor core details, but architecture-specific tuning (tile sizes, pipeline depth, number of warps) still requires per-vendor configuration

**Mojo's portability promise:** Mojo claims hardware portability through its MLIR foundation, but it is closed-source. You cannot inspect the compiler, cannot contribute GPU backend improvements, and cannot verify performance claims independently. Its GPU support is also NVIDIA-first.

**ROCm's perpetual beta:** AMD's ROCm platform is open-source and improving, but the software ecosystem trails CUDA by 3-5 years. Libraries like hipBLAS and MIOpen are functional but less optimized. The real problem: researchers write CUDA first (because that is where the documentation and Stack Overflow answers are), and AMD support is always an afterthought.

### How Vortex Solves It

Vortex uses MLIR progressive lowering to generate vendor-specific code from a single source, with domain-specific optimizations applied above the hardware layer.

```vortex
// This kernel compiles to optimal code on NVIDIA, AMD, and Intel
// The programmer writes it ONCE. The compiler handles the rest.
kernel attention_block<T: Float>(
    q: Tensor<T, [B, H, S, D]>,
    k: Tensor<T, [B, H, S_kv, D]>,
    v: Tensor<T, [B, H, S_kv, D]>,
    causal: comptime bool,
) -> Tensor<T, [B, H, S, D]> {
    // Block-level programming — tile sizes are auto-tuned per target
    let BLOCK_M = comptime target_optimal_tile_m(T, D)  // 128 on H100, 64 on MI300X
    let BLOCK_N = comptime target_optimal_tile_n(T, D)  // 128 on H100, 128 on MI300X

    parallel_for (b, h, q_block) in grid(B, H, ceil_div(S, BLOCK_M)) {
        let q_start = q_block * BLOCK_M

        // Load Q tile to shared memory
        // Compiler emits: TMA on Hopper, cp.async on Ampere,
        //                  buffer_load on CDNA, LSC on PVC
        let q_tile = q[b, h, q_start..+BLOCK_M, :].load(@shared)

        var m_prev = Tensor.fill<f32, [BLOCK_M]>(-inf)
        var l_prev = Tensor.zeros<f32, [BLOCK_M]>()
        var acc = Tensor.zeros<f32, [BLOCK_M, D]>()

        for kv_start in range(0, S_kv, step = BLOCK_N) {
            let k_tile = k[b, h, kv_start..+BLOCK_N, :].load(@shared)
            let v_tile = v[b, h, kv_start..+BLOCK_N, :].load(@shared)
            sync_block()

            // @ operator maps to hardware tensor cores:
            // NVIDIA: WGMMA (Hopper) or HMMA (Ampere)
            // AMD: MFMA (CDNA 3) or WMMA (RDNA 3+)
            // Intel: XMX
            let scores = (q_tile @ k_tile.transpose()) / sqrt(D as f32)

            if causal {
                scores.mask_where(|qi, ki| q_start + qi < kv_start + ki, -inf)
            }

            // Online softmax + accumulation
            let m_new = max(m_prev, scores.row_max())
            let correction = exp(m_prev - m_new)
            let p = exp(scores - m_new.unsqueeze(-1))
            acc = acc .* correction.unsqueeze(-1) + p @ v_tile
            l_prev = l_prev .* correction + p.row_sum()
            m_prev = m_new

            sync_block()
        }

        let output_tile = (acc / l_prev.unsqueeze(-1)).cast<T>()
        output_tile.store(output[b, h, q_start..+BLOCK_M, :])
    }
}

// The SAME source compiles to three backends:
//
// $ vortex build attention.vx --target sm_90
//   → PTX with WGMMA, TMA, 228KB shared memory budget
//   → cubin (sm_90), estimated 92% of peak
//
// $ vortex build attention.vx --target gfx942
//   → AMDGCN with MFMA, buffer_load, 64KB LDS budget
//   → hsaco (MI300X), estimated 88% of peak
//
// $ vortex build attention.vx --target pvc
//   → SPIR-V with XMX, LSC, 128KB SLM budget
//   → kernel binary, estimated 85% of peak

// For the last 5%, drop down to target-specific intrinsics
@target(sm_90)
kernel hopper_optimized_gemm(
    a: Tensor<f16, [M, K]>,
    b: Tensor<f16, [K, N]>,
) -> Tensor<f16, [M, N]> {
    // Hopper-specific: warp group MMA with TMA
    let a_desc = tma_create_descriptor(a, tile = [128, 64])
    let b_desc = tma_create_descriptor(b, tile = [64, 256])

    // Warp specialization: producer warps load, consumer warps compute
    @warp_specialize(producers = 1, consumers = 3)
    {
        producer {
            tma_async_load(a_desc, a_tile_smem)
            tma_async_load(b_desc, b_tile_smem)
        }
        consumer {
            wgmma(acc, a_tile_smem, b_tile_smem)  // 256x128x64 MMA
        }
    }
}
```

The compilation strategy is hierarchical:
1. **Vortex dialect** (target-independent): attention, matmul, NTT as single operations
2. **Linalg/SCF** (target-independent): tiled loops, tensor contractions
3. **GPU dialect** (target-aware): thread mapping, shared memory, barriers
4. **Backend dialect** (target-specific): NVVM/ROCDL/SPIRV with vendor intrinsics

Domain-specific optimizations (FlashAttention tiling, NTT butterfly fusion, Montgomery multiplication scheduling) happen at levels 1-2, where they are portable. Hardware-specific tuning (tile sizes, pipeline depth, tensor core shape selection) happens at levels 3-4. This is the crucial insight: the optimizations that matter most are algorithmic, not architectural, and they should be applied once and benefit all backends.

---

## Summary: Why a New Language Is Necessary

These seven problems share a common theme: they cannot be solved by a library, a DSL embedded in Python, or a CUDA wrapper. They require changes to the **compilation model** itself:

| Problem | Why a Library Cannot Fix It |
|---|---|
| Sparse/irregular computation | Compaction and load balancing require compiler-generated code paths that fuse with surrounding computation |
| Adaptive control flow | Warp retirement and thread binning require compiler restructuring of control flow at the IR level |
| Kernel fusion | Fusing operations across function boundaries requires whole-program analysis at the IR level |
| Crypto + ML | Unified type system for `Field<P>`, `Secret<T>`, and `Tensor<f16>` requires language-level support |
| Type-safe shapes | Compile-time shape checking requires dependent types in the language's type system |
| Non-backprop training | Local learning primitives require alternatives to autograd at the compiler level |
| Hardware abstraction | Multi-backend codegen from one source requires an MLIR-based compilation pipeline |

Vortex is not an incremental improvement. It is a bet that GPU programming needs a new foundation — one designed for the workloads that matter in 2026 and beyond, not the workloads of 2012 when CUDA's programming model was established.
