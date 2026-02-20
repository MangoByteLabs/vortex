# 07 — Standard Library Design

## Module Structure

```
std/
├── crypto/
│   ├── fields/          # Finite field arithmetic
│   │   ├── bn254.vx     # BN254 Fp, Fr
│   │   ├── bls12_381.vx # BLS12-381 Fp, Fr
│   │   ├── goldilocks.vx# Goldilocks (2^64 - 2^32 + 1)
│   │   ├── babybear.vx  # BabyBear (2^31 - 2^27 + 1)
│   │   └── mersenne31.vx# Mersenne31 (2^31 - 1)
│   ├── curves/          # Elliptic curve operations
│   │   ├── bn254.vx     # BN254 G1, G2, pairing
│   │   ├── bls12_381.vx # BLS12-381 G1, G2, pairing
│   │   ├── ed25519.vx   # Ed25519 (signatures)
│   │   └── secp256k1.vx # secp256k1 (Bitcoin/Ethereum)
│   ├── ntt.vx           # Number Theoretic Transform
│   ├── msm.vx           # Multi-Scalar Multiplication
│   ├── hash/            # Cryptographic hash functions
│   │   ├── poseidon.vx  # ZK-friendly hash
│   │   ├── sha256.vx    # SHA-256
│   │   ├── blake3.vx    # BLAKE3
│   │   └── keccak.vx    # Keccak/SHA-3
│   ├── poly.vx          # Polynomial arithmetic
│   ├── merkle.vx        # Merkle tree construction
│   └── fhe/             # Homomorphic encryption
│       ├── tfhe.vx      # TFHE bootstrapping, key switching
│       └── ckks.vx      # CKKS encode/decode, ops
├── nn/                  # Neural network primitives
│   ├── attention.vx     # FlashAttention, paged attention
│   ├── linear.vx        # Matrix multiplication, quantized GEMM
│   ├── norm.vx          # RMSNorm, LayerNorm
│   ├── activation.vx    # SiLU, GELU, ReLU, softmax
│   ├── embedding.vx     # Token + positional embedding
│   ├── rope.vx          # Rotary positional encoding
│   ├── cache.vx         # KV-cache, paged attention
│   └── quantize.vx      # Quantization/dequantization
├── tensor/              # Tensor operations
│   ├── create.vx        # zeros, ones, arange, rand
│   ├── manipulate.vx    # reshape, transpose, concat, split
│   ├── reduce.vx        # sum, mean, max, min, argmax
│   ├── index.vx         # gather, scatter, slice
│   └── linalg.vx       # matmul, outer, inner, trace, inv
├── io/                  # I/O operations
│   ├── safetensors.vx   # SafeTensors format
│   ├── gguf.vx          # GGUF format (llama.cpp compatible)
│   └── file.vx          # File I/O utilities
├── random/              # GPU random number generation
│   ├── philox.vx        # Philox counter-based PRNG
│   └── threefry.vx      # Threefry PRNG
├── comm/                # Multi-GPU communication
│   ├── allreduce.vx     # All-reduce (sum, max, etc.)
│   ├── alltoall.vx      # All-to-all (for MoE)
│   ├── broadcast.vx     # Broadcast
│   └── p2p.vx           # Peer-to-peer send/recv
└── serve/               # LLM serving infrastructure
    ├── server.vx        # HTTP/gRPC server
    ├── scheduler.vx     # Continuous batching scheduler
    └── tokenizer.vx     # BPE tokenizer
```

---

## Key APIs

### std.crypto.fields

```vortex
pub trait PrimeField: Numeric + Eq {
    const MODULUS: [u64; LIMBS]
    const MODULUS_BITS: u32
    const GENERATOR: Self
    const TWO_ADICITY: u32      // largest k such that 2^k | (p-1)
    const ROOT_OF_UNITY: Self   // primitive 2^k-th root of unity

    fn inv(self) -> Self
    fn pow(self, exp: u64) -> Self
    fn sqrt(self) -> Option<Self>
    fn is_zero(self) -> bool
    fn to_montgomery(self) -> MontField<Self>
    fn from_bytes(bytes: &[u8]) -> Option<Self>
    fn to_bytes(self) -> [u8; LIMBS * 8]
}

// Usage
import crypto.fields.bn254 { Fr }
let x: Fr = Fr.from(42)
let y = x.inv() * x  // == Fr.one()
```

### std.nn.attention

```vortex
pub struct FlashAttentionConfig {
    pub block_m: u32 = 128,
    pub block_n: u32 = 128,
    pub num_stages: u32 = 3,
    pub causal: bool = false,
    pub dropout: f32 = 0.0,
    pub window_size: Option<u32> = None,  // sliding window attention
}

pub fn flash_attention<T: Float>(
    q: Tensor<T, [B, H, S, D]>,
    k: Tensor<T, [B, H_kv, S_kv, D]>,
    v: Tensor<T, [B, H_kv, S_kv, D]>,
    config: FlashAttentionConfig = .{},
) -> Tensor<T, [B, H, S, D]>
where H % H_kv == 0  // GQA: query heads must be divisible by KV heads

pub fn paged_attention<T: Float>(
    q: Tensor<T, [B, H, 1, D]>,  // single-token decode
    cache: &PagedKVCache<T>,
    seq_lens: Tensor<u32, [B]>,
) -> Tensor<T, [B, H, 1, D]>
```

### std.tensor

```vortex
pub fn zeros<T: Numeric, Shape>() -> Tensor<T, Shape>
pub fn ones<T: Numeric, Shape>() -> Tensor<T, Shape>
pub fn arange<T: Numeric>(start: T, end: T, step: T = T.one()) -> Tensor<T, [?]>
pub fn rand<T: Float, Shape>(rng: &mut PhiloxRng) -> Tensor<T, Shape>

pub fn reshape<T, OldShape, NewShape>(t: Tensor<T, OldShape>) -> Tensor<T, NewShape>
where product(OldShape) == product(NewShape)

pub fn transpose<T>(t: Tensor<T, [M, N]>) -> Tensor<T, [N, M]>
pub fn concat<T, N1, N2>(a: Tensor<T, [N1, D]>, b: Tensor<T, [N2, D]>) -> Tensor<T, [N1+N2, D]>

pub fn sum<T: Numeric>(t: Tensor<T, Shape>, dim: i32) -> Tensor<T, ReducedShape>
pub fn max<T: Ord>(t: Tensor<T, Shape>, dim: i32) -> Tensor<T, ReducedShape>
pub fn argmax<T: Ord>(t: Tensor<T, Shape>, dim: i32) -> Tensor<u32, ReducedShape>
```

---

## Design Principles for the Standard Library

1. **GPU-first**: All operations are designed for GPU execution. CPU fallbacks exist
   but are not the primary path.

2. **Zero hidden allocations**: Every GPU memory allocation is explicit or documented.
   No surprise VRAM consumption.

3. **Fusible by default**: Standard library operations are annotated with fusion
   compatibility. The compiler can automatically fuse chains of compatible operations.

4. **Type-safe**: Shape mismatches, field mismatches, and representation errors are
   all caught at compile time.

5. **Benchmarked**: Every operation in the standard library ships with benchmarks
   against reference implementations (cuBLAS, CUTLASS, ICICLE, FlashAttention).
