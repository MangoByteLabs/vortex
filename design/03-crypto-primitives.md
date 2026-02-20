# 03 — Cryptography Primitives

## Overview

Vortex provides first-class support for the computational primitives underlying
modern cryptographic systems: ZK proofs (Groth16, PLONK, STARKs), homomorphic
encryption (TFHE, CKKS/BGV), and elliptic curve cryptography.

---

## 1. Finite Field Arithmetic

### Built-in Field Types

```vortex
// The modulus is encoded in the type — cross-field operations are compile errors
type BN254_Fr  = Field<0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001>
type BLS12_381_Fr = Field<0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001>
type Goldilocks = Field<0xFFFFFFFF00000001>  // 2^64 - 2^32 + 1 (fits in u64!)
type BabyBear  = Field<0x78000001>           // 2^31 - 2^27 + 1 (fits in u32!)

// Operations are automatically modular — no overflow possible
let a: BN254_Fr = 42
let b: BN254_Fr = a * a + 7    // Montgomery mul + add with reduction
let c: BN254_Fr = a.inv()      // Fermat's little theorem: a^(p-2) mod p
let d: BN254_Fr = a.sqrt()?    // Returns Option<Field> (not all elements have sqrt)

// Compile-time error: cannot mix fields
let e: BLS12_381_Fr = a + b    // ERROR: type mismatch
```

### Internal Representation

Fields use Montgomery form internally for fast multiplication. The representation
is tracked in the type system:

```vortex
// Standard vs Montgomery form are distinct types
type Fr      = Field<P>              // Standard representation
type FrMont  = MontField<P>          // Montgomery form: a_mont = a * R mod P

// Conversion is explicit (prevents mixing representations)
let x: Fr = Fr.from(42)
let x_mont: FrMont = x.to_montgomery()     // x * R mod P
let x_back: Fr = x_mont.from_montgomery()  // x_mont * R^{-1} mod P

// All arithmetic on MontField uses Montgomery multiplication internally
// (compiler handles this transparently when you use Field<P>)
```

### Limb Layout for GPU Efficiency

256-bit field elements use 8x 32-bit limbs stored in **Structure of Arrays** layout
for coalesced GPU memory access:

```vortex
// When you write:
let elements: Tensor<BN254_Fr, [N]> @global

// The compiler stores this as SoA internally:
// limb0: [elem0_limb0, elem1_limb0, ..., elemN_limb0]  // contiguous
// limb1: [elem0_limb1, elem1_limb1, ..., elemN_limb1]  // contiguous
// ...
// limb7: [elem0_limb7, elem1_limb7, ..., elemN_limb7]  // contiguous

// Thread i loads limb0[i], limb1[i], ..., limb7[i] — 8 coalesced transactions
// Each Montgomery multiplication uses ~64 IMAD instructions, all in registers
```

### Field Extensions

```vortex
// Extension fields for pairing-based cryptography
type Fp2 = FieldExt<BN254_Fp, degree=2, irred=[-1, 0, 1]>  // Fp[u]/(u^2 + 1)
type Fp6 = FieldExt<Fp2, degree=3, irred=[-9, 0, 0, 1]>    // Fp2[v]/(v^3 - 9-u)
type Fp12 = FieldExt<Fp6, degree=2, irred=[-v, 0, 1]>       // Fp6[w]/(w^2 - v)

// Extension arithmetic compiles to optimized Karatsuba/Toom-Cook methods
let a: Fp12 = ...
let b: Fp12 = ...
let c: Fp12 = a * b  // Karatsuba over Fp6, then over Fp2, then base field muls
```

---

## 2. Number Theoretic Transform (NTT)

The NTT is the modular-arithmetic analog of the FFT, used for polynomial
multiplication in ZK proofs and homomorphic encryption.

### High-Level API

```vortex
import crypto.ntt { ntt_forward, ntt_inverse, ntt_multiply }

// Polynomials track their representation in the type system
type CoeffPoly<F, N> = Poly<F, N, .Coefficient>
type NTTPoly<F, N>   = Poly<F, N, .NTT>

// Conversion between representations
let poly: CoeffPoly<Fr, 1024> = ...
let poly_ntt: NTTPoly<Fr, 1024> = ntt_forward(poly)

// Pointwise multiplication only works in NTT domain (type enforced)
let product: NTTPoly<Fr, 1024> = poly_ntt * other_ntt  // OK
// let bad = poly * other_coeff  // COMPILE ERROR: cannot multiply in coefficient form

// Convert back
let result: CoeffPoly<Fr, 1024> = ntt_inverse(product)
```

### Optimized GPU Implementation

```vortex
// NTT auto-tuned for GPU memory hierarchy
@schedule(
    // Early stages: butterflies span entire array, global memory bound
    stages(0..split_point): global_memory, coalesced, prefetch_twiddles(.L2),
    // Late stages: data fits in shared memory, compute bound
    stages(split_point..log2(N)): fused_shared_memory(48 * 1024),
    // Split point auto-computed from N and shared memory size
)
kernel ntt_forward<F: PrimeField, const N: usize>(
    data: &mut Tensor<F, [N]>,
    roots: &Tensor<F, [N]>,
) -> Tensor<F, [N]>
where N: PowerOfTwo
{
    // Butterfly operation (compiler generates optimal code)
    for stage in 0..log2(N) {
        let stride = N >> (stage + 1)
        parallel_for j in 0..N/2 {
            let (a, b) = butterfly_indices(j, stride)
            let twiddle = roots[twiddle_index(stage, j)]
            let t = data[b] * twiddle
            data[b] = data[a] - t
            data[a] = data[a] + t
        }
    }
}
```

### Performance Targets

| Size | Field | GPU | Target Latency | Reference |
|---|---|---|---|---|
| 2^20 | BN254_Fr (256-bit) | RTX 4090 | < 5 ms | ICICLE baseline |
| 2^24 | BN254_Fr (256-bit) | RTX 4090 | < 80 ms | ICICLE baseline |
| 2^20 | Goldilocks (64-bit) | RTX 4090 | < 1 ms | Native u64 ops |
| 2^24 | BabyBear (32-bit)  | RTX 4090 | < 0.5 ms | Native u32 ops |

---

## 3. Multi-Scalar Multiplication (MSM)

MSM computes `Q = sum(k_i * P_i)` for scalar-point pairs. This is the dominant
cost in ZK proof generation (Groth16, PLONK, KZG commitments).

### High-Level API

```vortex
import crypto.msm { msm }
import crypto.curves.bn254 { G1, Fr }

// Simple API — compiler selects optimal algorithm
let commitment: G1 = msm(scalars, generators)

// With explicit algorithm selection
let commitment: G1 = msm(scalars, generators, algorithm = .pippenger(window_size = 15))
```

### Pippenger's Bucket Method

```vortex
// Internal implementation uses Pippenger's algorithm
// Window size w splits each scalar into ceil(256/w) windows
// Each window has 2^w buckets

@schedule(
    window_parallelism = .across_blocks,     // Each GPU block handles one window
    bucket_accumulation = .warp_cooperative,  // Warps cooperate on bucket sums
    reduction = .tree,                        // Tree reduction for window aggregation
)
kernel msm_pippenger<C: Curve>(
    scalars: Tensor<C.ScalarField, [N]>,
    points: Tensor<Point<C, .Affine>, [N]>,
    window_size: comptime u32,
) -> Point<C, .Jacobian> {
    let num_windows = ceil_div(C.SCALAR_BITS, window_size)
    let num_buckets = (1 << window_size) - 1

    // Phase 1: Bucket accumulation (parallel across windows)
    var buckets: Tensor<Point<C, .Jacobian>, [num_windows, num_buckets]> @shared

    parallel_for win in 0..num_windows {
        for i in 0..N {
            let digit = scalars[i].window(win, window_size)
            if digit != 0 {
                buckets[win, digit - 1] += points[i]  // EC point addition
            }
        }
    }

    // Phase 2: Bucket reduction (running sum within each window)
    parallel_for win in 0..num_windows {
        var running_sum = Point.identity()
        var window_sum = Point.identity()
        for b in (0..num_buckets).rev() {
            running_sum += buckets[win, b]
            window_sum += running_sum
        }
        window_results[win] = window_sum
    }

    // Phase 3: Window aggregation (Horner's method)
    var result = window_results[num_windows - 1]
    for win in (0..num_windows - 1).rev() {
        for _ in 0..window_size {
            result = ec_double(result)
        }
        result += window_results[win]
    }

    return result
}
```

### Performance Targets

| Size | Curve | GPU | Target | Baseline (ICICLE) |
|---|---|---|---|---|
| 2^16 | BN254 G1 | RTX 4090 | < 50 ms | ~80 ms |
| 2^20 | BN254 G1 | RTX 4090 | < 500 ms | ~800 ms |
| 2^20 | BLS12-381 G1 | RTX 4090 | < 800 ms | ~1.2 s |
| 2^24 | BN254 G1 | H100 | < 2 s | ~3 s |

---

## 4. Elliptic Curve Operations

### Curve Definition

```vortex
// Curve types parameterized by form and base field
trait Curve {
    type BaseField: PrimeField
    type ScalarField: PrimeField
    const A: BaseField
    const B: BaseField
    const GENERATOR: Point<Self, .Affine>
    const ORDER: ScalarField
}

// Concrete curves
struct BN254: Curve {
    type BaseField = Field<0x30644e...>
    type ScalarField = Field<0x30644e...01>
    const A = 0
    const B = 3
    ...
}

struct BLS12_381: Curve { ... }
struct Ed25519: Curve { ... }   // twisted Edwards form
struct Secp256k1: Curve { ... }
```

### Point Operations

```vortex
// Points track their coordinate system in the type
type AffinePoint<C>    = Point<C, .Affine>     // (x, y), 2 field elements
type JacobianPoint<C>  = Point<C, .Jacobian>   // (X:Y:Z), 3 field elements
type ProjectivePoint<C> = Point<C, .Projective> // (X:Y:Z), 3 field elements
type ExtendedPoint<C>  = Point<C, .Extended>    // (X:Y:T:Z), 4 field elements

// Mixed addition: Jacobian + Affine → Jacobian (most efficient for MSM)
fn ec_add_mixed<C: Curve>(
    a: JacobianPoint<C>,
    b: AffinePoint<C>,
) -> JacobianPoint<C> {
    // 7M + 4S field operations
    let u2 = b.x * a.z.square()
    let s2 = b.y * a.z.cube()
    let h = u2 - a.x
    let r = s2 - a.y
    ...
}

// Pairing (only available for pairing-friendly curves)
fn pairing<C: PairingFriendly>(a: Point<C.G1>, b: Point<C.G2>) -> C.GT {
    let f = miller_loop(a, b)
    return final_exponentiation(f)
}
```

---

## 5. Constant-Time Execution

### The @constant_time Guarantee

```vortex
// Compiler verifies: no branches on secret data, no secret-dependent
// memory access, no variable-time instructions
@constant_time
kernel batch_scalar_mul(
    scalars: Tensor<Secret<Fr>, [N]>,      // Secret scalars
    points: Tensor<Point<BN254, .Affine>, [N]>,  // Public points
) -> Tensor<Point<BN254, .Jacobian>, [N]> {
    parallel_for i in 0..N {
        var acc = Point.identity()
        var temp = points[i]

        for bit in 0..256 {
            let b: Secret<bool> = scalars[i].bit(bit)
            // ct_select: always performs both paths, selects via masking
            acc = ct_select(b, ec_add(acc, temp), acc)
            temp = ec_double(temp)
        }

        result[i] = acc
    }
}
```

### Secret Type and Information Flow

```vortex
// Secret values taint all derived computations
let key: Secret<Fr> = load_private_key()
let derived = key * public_value  // derived: Secret<Fr> (tainted)

// Compile-time errors for unsafe operations on secrets:
if key > threshold { ... }     // ERROR: branching on Secret
table[key.to_u32()] = value    // ERROR: secret-dependent index
let public: Fr = key            // ERROR: implicit declassification

// Explicit declassification (requires annotation on enclosing function)
@allow_declassify
fn verify_signature(...) -> bool {
    let computed = hash(message)
    return declassify(ct_eq(computed, expected))  // constant-time compare
}
```

### GPU-Specific Side-Channel Mitigations

```vortex
// Warp-uniform enforcement: all threads in a warp take the same path
@warp_uniform
kernel constant_time_aes_sbox(
    input: Tensor<Secret<u8>, [N]>,
    sbox: Tensor<u8, [256]>,  // Public lookup table
) -> Tensor<Secret<u8>, [N]> {
    parallel_for i in 0..N {
        // Instead of: output[i] = sbox[input[i]]  (secret-dependent index!)
        // Use constant-time table lookup: scan all 256 entries
        var result: Secret<u8> = 0
        for j in 0..256 {
            let mask = ct_eq(input[i], j as u8)
            result = ct_select(mask, sbox[j], result)
        }
        output[i] = result
    }
}
```

---

## 6. Homomorphic Encryption Support

### TFHE Primitives

```vortex
import crypto.fhe.tfhe { LWE, RLWE, RGSW, bootstrap, key_switch }

// TFHE ciphertext types
type LWECiphertext<N: usize> = struct {
    a: Tensor<u32, [N]>,  // mask
    b: u32,                // body
}

type RLWECiphertext<N: usize> = struct {
    a: Poly<u32, N, .NTT>,  // mask polynomial
    b: Poly<u32, N, .NTT>,  // body polynomial
}

// Programmable bootstrapping
@schedule(
    ntt: fused_shared_memory,
    external_product: warp_cooperative,
)
kernel programmable_bootstrap<const N: usize>(
    ct: LWECiphertext<N>,
    bsk: &BootstrappingKey,
    test_vector: Poly<u32, N, .Coefficient>,
) -> LWECiphertext<N> {
    // BlindRotate + SampleExtract + KeySwitch
    let rotated = blind_rotate(ct, bsk, test_vector)
    let extracted = sample_extract(rotated)
    return key_switch(extracted, bsk.ksk)
}
```

### CKKS/BGV Support

```vortex
import crypto.fhe.ckks { CKKSCiphertext, encode, decode, rescale }

// CKKS with RNS-optimized arithmetic
type CKKSCt<L: usize> = struct {
    c0: RNSPoly<L>,  // L RNS components
    c1: RNSPoly<L>,
    scale: f64,
}

// Homomorphic operations
kernel he_matmul(
    ct_matrix: Tensor<CKKSCt<12>, [M, N]>,
    pt_matrix: Tensor<f64, [N, K]>,
) -> Tensor<CKKSCt<11>, [M, K]> {
    // Plaintext-ciphertext multiplication fused with rescaling
    let encoded = encode(pt_matrix, scale = ct_matrix[0,0].scale)
    let product = ct_matrix @ encoded
    return rescale(product)  // Level drops: 12 → 11
}
```

---

## 7. ZK Proof Systems

### Groth16 Prover

```vortex
import crypto.zk.groth16 { Proof, ProvingKey, prove }

// Full Groth16 proof generation
@gpu_resident  // Keep proving key on GPU across proofs
fn generate_proof(
    pk: &ProvingKey<BN254>,
    witness: &[Fr],
) -> Proof<BN254> {
    // Phase 1: NTT for polynomial arithmetic
    let witness_ntt = ntt_batch(witness_polys)

    // Phase 2: Compute H polynomial
    let h = compute_h_polynomial(witness_ntt, pk)

    // Phase 3: MSM for proof elements (dominates runtime)
    let pi_a = msm(witness, pk.a_query)   // G1
    let pi_b = msm(witness, pk.b_query)   // G2 (more expensive)
    let pi_c = msm(witness, pk.c_query) + msm(h_coeffs, pk.h_query)

    return Proof { a: pi_a, b: pi_b, c: pi_c }
}
```

### STARK Prover (FRI-based)

```vortex
import crypto.zk.stark { FRI, commit, query }

// FRI commitment with auto-tuned Merkle tree construction
@schedule(
    hash: parallel_tree,            // Merkle hashing parallelized per level
    ntt: fused_shared_memory,       // Polynomial evaluation via NTT
    folding: elementwise_parallel,  // FRI folding is embarrassingly parallel
)
kernel fri_commit<F: PrimeField, const N: usize>(
    polynomial: Poly<F, N, .Coefficient>,
    domain: &EvalDomain<F, N>,
) -> FRICommitment<F> {
    var current = ntt_forward(polynomial)  // Evaluate over domain
    var commitments: Vec<MerkleRoot> = []

    for round in 0..num_rounds(N) {
        // Commit via Merkle tree (hash-based, post-quantum secure)
        commitments.push(merkle_commit(current))

        // Receive challenge (Fiat-Shamir)
        let alpha = fiat_shamir_challenge(commitments)

        // Fold polynomial (halves degree)
        current = fri_fold(current, alpha)
    }

    return FRICommitment { commitments, final_poly: current }
}
```

---

## 8. Performance Summary

### State of the Art (GPU Crypto, 2024-2025)

| Operation | Platform | Performance | Source |
|---|---|---|---|
| TFHE Bootstrap (single) | H100 | ~2 ms | TFHE-rs v1.4 |
| TFHE Bootstrap (amortized/bit) | GPU | 0.423 us | Academic |
| Montgomery mul throughput | GPU | 3.38B ops/s | ICICLE |
| CKKS Key Switching | GPU | 380x vs CPU SEAL | HEonGPU |
| Groth16 MSM | ICICLE | 63x vs CPU | Ingonyama |
| Groth16 NTT | ICICLE | 320x vs CPU | Ingonyama |
| ECDSA signing | gECC | 5.56x vs prior GPU | gECC 2025 |
| Encrypted ResNet-20 | Cheddar (RTX 5090) | 0.72 s | Cheddar 2024 |
| Circle STARK proving | ICICLE-Stwo | 3.25-7x vs CPU SIMD | Ingonyama |

### Vortex Design Goals

Match or exceed ICICLE/state-of-the-art performance while providing:
- Type-safe field arithmetic (no cross-field bugs)
- Representation tracking (NTT vs coefficient form)
- Constant-time guarantees (verified by compiler)
- Multi-backend support (NVIDIA + AMD + Intel)
