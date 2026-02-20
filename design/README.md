# Vortex — A GPU-Native Language for Cryptography & LLMs

> Working name: **Vortex** (parallel flows converging toward a result)

## Project Vision

Vortex is a systems-level programming language designed from the ground up for GPU
execution, with first-class support for two high-value domains:

1. **Cryptography** — ZK proofs, homomorphic encryption, post-quantum schemes
2. **Large Language Models** — training, inference, and serving

The language treats parallelism as the default execution model, not an afterthought.
It provides domain-specific primitives (finite fields, tensors, attention) that compile
to highly optimized GPU kernels across NVIDIA, AMD, and Intel hardware.

## Design Documents

| Document | Description |
|---|---|
| [00-philosophy.md](./00-philosophy.md) | Core design philosophy and principles |
| [01-type-system.md](./01-type-system.md) | Type system design |
| [02-memory-model.md](./02-memory-model.md) | GPU memory model and ownership |
| [03-crypto-primitives.md](./03-crypto-primitives.md) | Cryptography-specific features |
| [04-llm-primitives.md](./04-llm-primitives.md) | LLM-specific features |
| [05-compiler.md](./05-compiler.md) | Compilation pipeline (MLIR-based) |
| [06-syntax.md](./06-syntax.md) | Syntax and grammar |
| [07-stdlib.md](./07-stdlib.md) | Standard library design |
| [08-tooling.md](./08-tooling.md) | Debugger, profiler, package manager |
| [09-roadmap.md](./09-roadmap.md) | Implementation roadmap |

## Quick Example

```vortex
// ZK proof kernel — multi-scalar multiplication over BN254
import crypto.curves.bn254 { G1, Fr }
import crypto.msm { pippenger }

@constant_time
kernel prove(witnesses: Tensor<Fr, [N]>, generators: Tensor<G1, [N]>) -> G1 {
    // Compiler auto-tiles this across GPU workgroups
    // using Pippenger's bucket method with optimal window size
    return pippenger(generators, witnesses)
}

// LLM inference — fused transformer block
import nn.attention { flash_attention }
import nn.linear { matmul }
import nn.norm { rms_norm }

kernel transformer(
    x: Tensor<bf16, [B, S, D]>,
    weights: TransformerWeights<bf16>,
) -> Tensor<bf16, [B, S, D]> {
    let normed = rms_norm(x, weights.norm)
    let q, k, v = matmul(normed, weights.qkv).split<3>(axis: -1)
    let attn = flash_attention(q, k, v, causal: true)
    let projected = matmul(attn, weights.out_proj)
    let residual = x + projected
    let ff = silu(matmul(rms_norm(residual, weights.ff_norm), weights.gate))
           * matmul(rms_norm(residual, weights.ff_norm), weights.up)
    return residual + matmul(ff, weights.down)
}
```
