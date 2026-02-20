# Type System Design for a GPU Programming Language Targeting Cryptography and LLMs

## Table of Contents

1. [Dependent Types for Tensor Shapes](#1-dependent-types-for-tensor-shapes)
2. [Linear/Affine Types for GPU Resource Management](#2-linearaffine-types-for-gpu-resource-management)
3. [Effect Systems](#3-effect-systems)
4. [Modular Arithmetic Types](#4-modular-arithmetic-types)
5. [Fixed-Width Big Integer Types](#5-fixed-width-big-integer-types)
6. [Precision Types for ML](#6-precision-types-for-ml)
7. [Constant-Time Type Qualifiers](#7-constant-time-type-qualifiers)
8. [Ownership and Borrowing for GPU Memory](#8-ownership-and-borrowing-for-gpu-memory)
9. [Parametric Polymorphism and GPU Kernel Compilation](#9-parametric-polymorphism-and-gpu-kernel-compilation)
10. [Examples from Existing Languages](#10-examples-from-existing-languages)

---

## 1. Dependent Types for Tensor Shapes

### 1.1 The Problem

GPU kernels for both cryptography (NTT butterflies, matrix-based lattice schemes) and LLMs (attention, matmul, convolution) are fundamentally operations on multi-dimensional arrays whose shapes must satisfy algebraic constraints. A matrix multiply of dimensions `(M, K) x (K, N) -> (M, N)` is ill-defined unless the inner dimensions agree. Catching such errors at compile time eliminates an entire class of runtime failures and out-of-bounds GPU memory accesses.

### 1.2 Encoding Shapes in Types

We adopt a system of **size-dependent types** in the spirit of Futhark's size types and Dex's typed indices, but extended to the GPU setting.

#### Type Formation Rules

Introduce a sort `Nat` of compile-time natural numbers and a dependent array type constructor:

```
--------------------------------------------------
  Nat : Sort

  n : Nat
--------------------------------------------------
  Fin n : Type          -- the type {0, 1, ..., n-1}

  A : Type,  n : Nat
--------------------------------------------------
  Tensor<A, [n1, n2, ..., nk]> : Type
```

The key typing judgement for tensor operations:

```
Gamma |- a : Tensor<A, [M, K]>
Gamma |- b : Tensor<A, [K, N]>
--------------------------------------------------
Gamma |- matmul(a, b) : Tensor<A, [M, N]>
```

The size variable `K` is unified across both arguments. If the caller provides tensors with mismatched inner dimensions, the type checker rejects the program.

#### Existential Size Quantification

Following Futhark's approach, functions that produce arrays of data-dependent size return existentially quantified types:

```
-- The filter function produces an array whose length is not known at compile time.
-- The result size 'k' is existentially bound.
val filter : {n : Nat} -> (A -> Bool) -> Tensor<A, [n]> -> (k : Nat, Tensor<A, [k]>)
```

In practice, the language automatically wraps and unwraps the dependent pair, so the programmer writes:

```rust
fn filter<n: Nat>(pred: fn(A) -> Bool, xs: Tensor<A, [n]>) -> Tensor<A, [?k]>
```

where `?k` denotes an existentially bound size variable. In a `let` binding, the existential is opened:

```rust
let result = filter(is_nonzero, data);
// result : Tensor<A, [k]> for some unknown k
// k is now in scope as a runtime value
launch_kernel::<k>(result);
```

#### Arithmetic on Sizes

Size expressions support a limited arithmetic:

```
size_expr ::= n              -- variable
            | literal        -- constant natural number
            | s1 + s2        -- addition
            | s1 * s2        -- multiplication
            | s1 / s2        -- division (total, rounds down)
            | max(s1, s2)
            | min(s1, s2)
```

This is intentionally restricted to keep type checking decidable. We do **not** permit arbitrary term-level computation at the type level (unlike full dependent types in Idris/Agda), ensuring that size constraint solving remains within a decidable fragment (linear arithmetic over naturals, i.e., Presburger arithmetic).

#### Example: NTT (Number Theoretic Transform)

```rust
/// Radix-2 NTT requires the array length to be a power of 2.
/// We encode this via a log-size parameter.
fn ntt<log_n: Nat, P: Prime>(
    coeffs: Tensor<Field<P>, [pow2(log_n)]>,
    omega: Field<P>,          // primitive (2^log_n)-th root of unity
) -> Tensor<Field<P>, [pow2(log_n)]>
```

The type `pow2(log_n)` is a size expression computed from the type parameter, ensuring the array length is always a power of two. This eliminates runtime checks for NTT preconditions.

#### Example: Multi-Head Attention

```rust
fn multi_head_attention<B: Nat, S: Nat, D: Nat, H: Nat>(
    query:  Tensor<f16, [B, S, D]>,
    key:    Tensor<f16, [B, S, D]>,
    value:  Tensor<f16, [B, S, D]>,
) -> Tensor<f16, [B, S, D]>
where
    D % H == 0   // D must be divisible by number of heads
{
    let head_dim: Nat = D / H;
    let q_heads: Tensor<f16, [B, H, S, head_dim]> = reshape(query);
    // ...
}
```

### 1.3 Comparison of Approaches

| Feature | Futhark | Dex | Our Design |
|---------|---------|-----|------------|
| Size variables | First-class in types | Via typed index sets | First-class `Nat` sort |
| Existentials | Implicit packing/unpacking | Implicit | Explicit `?k` syntax, implicit unpacking |
| Arithmetic | Limited (add, sub) | Full (via Haskell-level) | Presburger fragment |
| Decidability | Yes (restricted) | No (full dependent types) | Yes (by design) |
| GPU targeting | Yes (OpenCL/CUDA) | JAX backend | Native CUDA/ROCm |

---

## 2. Linear/Affine Types for GPU Resource Management

### 2.1 Motivation

GPU programming involves explicit resource management: device memory allocations (`cudaMalloc`), streams, events, and descriptor sets must be freed exactly once. In cryptographic contexts, sensitive key material in device memory must additionally be zeroed before deallocation. Classical type systems cannot prevent:

- **Double-free**: Freeing the same device buffer twice, causing memory corruption.
- **Use-after-free**: Accessing a buffer after it has been freed.
- **Leaks**: Forgetting to free a buffer, exhausting GPU VRAM.

### 2.2 Substructural Type Discipline

We define three classes of types using substructural rules:

| Type Class | Contraction (Duplicate) | Weakening (Drop) | Description |
|------------|------------------------|-------------------|-------------|
| `Unrestricted` | Yes | Yes | Ordinary values (scalars, sizes) |
| `Affine` | No | Yes | Used at most once; may be dropped (auto-cleanup) |
| `Linear` | No | No | Used exactly once; must be explicitly consumed |

The key typing rules:

```
-- Linear introduction: creating a GPU buffer produces a linear value
Gamma |- alloc_device<A, n>(device) : Linear<DeviceBuffer<A, n>>

-- Linear elimination: the buffer must be consumed exactly once
Gamma, x : Linear<DeviceBuffer<A, n>> |- e : T
  (x appears exactly once in e)
--------------------------------------------------
Gamma |- let x = alloc_device(device) in e : T

-- Consumption via free
Gamma |- buf : Linear<DeviceBuffer<A, n>>
--------------------------------------------------
Gamma |- free(buf) : ()

-- Consumption via transfer (moves ownership)
Gamma |- buf : Linear<DeviceBuffer<A, n>>
--------------------------------------------------
Gamma |- device_to_host(buf) : (Linear<HostBuffer<A, n>>)
```

### 2.3 Practical Syntax

```rust
// Linear types are the default for GPU resources.
// The compiler ensures every DeviceBuffer is consumed exactly once.

fn encrypt_on_gpu<n: Nat, P: Prime>(
    plaintext: &HostBuffer<Field<P>, n>,   // borrowed, not consumed
    key: &DeviceBuffer<Field<P>, n>,        // borrowed, not consumed
) -> DeviceBuffer<Field<P>, n>              // returns ownership of new buffer
{
    let ct = alloc_device::<Field<P>, n>();  // linear: must be consumed
    launch_kernel!(ntt_multiply, key, plaintext, &mut ct);
    ct  // ownership transferred to caller -- this is the single consumption
}

fn main() {
    let key_buf = upload_to_device(key_data);       // Linear<DeviceBuffer<...>>
    let ct_buf = encrypt_on_gpu(&host_pt, &key_buf);

    let ct_host = device_to_host(ct_buf);           // ct_buf consumed here
    secure_free(key_buf);                            // key_buf consumed here; memory zeroed

    // ERROR if we try to use key_buf or ct_buf again:
    // launch_kernel!(foo, &key_buf);  // compile error: use of consumed linear value
}
```

### 2.4 Borrowing as Controlled Aliasing

Linear types alone are too restrictive for kernel launches that read from multiple buffers simultaneously. We introduce **borrows** (non-owning references) that temporarily alias a linear value:

```
Gamma |- buf : Linear<DeviceBuffer<A, n>>
--------------------------------------------------
Gamma |- &buf : &DeviceBuffer<A, n>        -- immutable borrow, buf still live
Gamma |- &mut buf : &mut DeviceBuffer<A, n> -- mutable borrow, exclusive access

-- Borrow rule: while &buf or &mut buf is live, buf cannot be moved or freed.
-- Multiple &buf allowed simultaneously; at most one &mut buf, exclusive with &buf.
```

### 2.5 Secure Deallocation for Cryptographic Keys

For cryptographic applications, we distinguish `SecureLinear<T>` from `Linear<T>`:

```rust
/// SecureLinear values MUST be consumed via secure_free(), which zeroes memory.
/// Dropping or freeing without zeroing is a type error.
fn load_secret_key<n: Nat, P: Prime>(
    key_data: &[u8]
) -> SecureLinear<DeviceBuffer<Field<P>, n>> {
    let buf = secure_alloc_device::<Field<P>, n>();
    upload_and_zero_host(key_data, &mut buf);
    buf
}

// The ONLY way to consume a SecureLinear is secure_free:
fn secure_free<T>(val: SecureLinear<DeviceBuffer<T>>) -> () {
    // zeros device memory, then deallocates
}
```

---

## 3. Effect Systems

### 3.1 Design Goals

A GPU programming language has several categories of side effects that interact in subtle ways:

1. **GPU memory allocation/deallocation** (`DeviceAlloc`)
2. **Host-device data transfers** (`Transfer`)
3. **Kernel launches** (`KernelLaunch`)
4. **Stream synchronization** (`Sync`)
5. **Host I/O** (`IO`)
6. **Non-termination / divergence** (`Diverge`)
7. **Random number generation** (for cryptographic nonces) (`Rand`)

We track these in a **row-polymorphic effect system** inspired by Koka's design.

### 3.2 Effect Rows

An effect row is an unordered collection of effect labels:

```
effect_row ::= <>                          -- pure
             | <E1, E2, ..., En>           -- finite set of effects
             | <E1, E2, ..., En | rho>     -- open row with row variable rho
```

Function types carry their effect row:

```
fn_type ::= (A1, A2, ..., Ak) ->{E} R
```

where `E` is an effect row. The typing rule for function application:

```
Gamma |- f : (A) ->{E} R
Gamma |- a : A
--------------------------------------------------
Gamma |- f(a) : R ! E        -- result R with effects E
```

### 3.3 Concrete Effect Definitions

```rust
effect DeviceAlloc {
    fn alloc<A: Type, n: Nat>() -> DeviceBuffer<A, n>
    fn free<A: Type, n: Nat>(buf: DeviceBuffer<A, n>) -> ()
}

effect Transfer {
    fn host_to_device<A: Type, n: Nat>(src: &HostBuffer<A, n>) -> DeviceBuffer<A, n>
    fn device_to_host<A: Type, n: Nat>(src: DeviceBuffer<A, n>) -> HostBuffer<A, n>
}

effect KernelLaunch {
    fn launch<K: Kernel>(grid: Grid, block: Block, args: K::Args) -> ()
}

effect Sync {
    fn stream_sync(stream: Stream) -> ()
    fn device_sync() -> ()
}
```

### 3.4 Effect Polymorphism

Functions that are agnostic to certain effects use row variables:

```rust
/// This function is pure except for whatever effects `f` has.
fn map<A, B, n: Nat, |rho>(
    f: (A) ->{rho} B,
    xs: Tensor<A, [n]>
) ->{rho} Tensor<B, [n]>
```

The row variable `rho` is universally quantified, meaning `map` can be used with pure functions (`rho = <>`) or effectful ones.

### 3.5 Effect Handlers for GPU Streams

Effect handlers provide a mechanism to intercept and reinterpret effects, which maps naturally to GPU stream management:

```rust
/// Execute a computation on a specific GPU stream.
/// All KernelLaunch effects within are dispatched to that stream.
fn on_stream<A, E>(
    stream: Stream,
    computation: () ->{KernelLaunch, E} A
) ->{Sync, E} A {
    handle computation {
        KernelLaunch::launch(grid, block, args) -> resume {
            launch_on_stream(stream, grid, block, args);
            resume(())
        }
    };
    stream_sync(stream);
    result
}
```

### 3.6 Typing Rule: Effect Subsumption

```
Gamma |- e : A ! E1
E1 is a subset of E2
--------------------------------------------------
Gamma |- e : A ! E2
```

A computation with fewer effects can be used where more effects are permitted (effect weakening). A pure function `() ->{<>} A` can be passed where `() ->{<DeviceAlloc>} A` is expected.

### 3.7 Purity Guarantees

The effect system provides critical guarantees:

- **GPU kernel bodies are restricted**: Inside a `__device__` function, only `DeviceAlloc` (shared memory) is permitted; `IO`, `Transfer`, and `Sync` are forbidden.
- **Pure cryptographic primitives**: Field arithmetic functions carry no effects, ensuring they can be freely reordered and parallelized.
- **Deterministic replay**: A function with only `KernelLaunch` effects (no `Rand`, no `IO`) produces deterministic results given the same inputs, critical for debugging.

---

## 4. Modular Arithmetic Types

### 4.1 The `Field<P>` Type

Cryptographic computations operate over finite fields `GF(p)` where `p` is prime. We encode the modulus as a type-level parameter:

```rust
/// A field element modulo a prime P.
/// P is a type-level natural number that must satisfy IsPrime<P>.
struct Field<P: Prime> {
    /// Internal representation: a natural number in [0, P).
    /// Invariant maintained by all operations.
    value: BigUint<limbs_for(P)>
}
```

The `Prime` trait bound:

```rust
/// Marker trait. Only types satisfying the primality proof can implement this.
/// In practice, this is verified at compile time for known constants
/// or assumed via unsafe for runtime-determined primes.
trait Prime: Nat {
    /// Witness that Self is prime.
    const PRIMALITY_CERTIFICATE: PrimalityCertificate<Self>;
}
```

### 4.2 Type-Level Primes

Common cryptographic primes are defined as type-level constants:

```rust
// BN254 scalar field prime
type BN254_R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
impl Prime for BN254_R { /* certificate */ }

// BLS12-381 scalar field prime
type BLS12_381_R = 52435875175126190479447740508185965837690552500527637822603658699938581184513;
impl Prime for BLS12_381_R { /* certificate */ }

// Goldilocks prime (2^64 - 2^32 + 1), popular in STARKs
type Goldilocks = 18446744069414584321;
impl Prime for Goldilocks { /* certificate */ }
```

### 4.3 Arithmetic Rules

All arithmetic is defined modulo `P`, and the type system prevents mixing elements from different fields:

```rust
impl<P: Prime> Add for Field<P> {
    type Output = Field<P>;
    fn add(self, rhs: Field<P>) -> Field<P> {
        Field { value: (self.value + rhs.value) % P }
    }
}

impl<P: Prime> Mul for Field<P> {
    type Output = Field<P>;
    fn mul(self, rhs: Field<P>) -> Field<P> {
        Field { value: (self.value * rhs.value) % P }
    }
}

impl<P: Prime> Inv for Field<P> {
    type Output = Field<P>;
    /// Modular inverse via Fermat's little theorem: a^{-1} = a^{p-2} mod p
    fn inv(self) -> Field<P> {
        self.pow(P - 2)
    }
}
```

The critical type safety property:

```
Gamma |- a : Field<P>
Gamma |- b : Field<Q>
P =/= Q
--------------------------------------------------
Gamma |- a + b : TYPE ERROR
// Cannot add elements from different fields
```

### 4.4 Field Extensions

For pairing-based cryptography, we need extension fields:

```rust
/// Quadratic extension field GF(p^2) = GF(p)[x] / (x^2 - beta)
struct FieldExt2<P: Prime, const BETA: Field<P>> {
    c0: Field<P>,
    c1: Field<P>,
}

/// Sextic extension for BLS12-381: GF(p^6)
type Fp6<P> = FieldExt3<FieldExt2<P, BETA_2>, BETA_6>;

/// Full extension for pairings: GF(p^12)
type Fp12<P> = FieldExt2<Fp6<P>, GAMMA>;
```

### 4.5 Montgomery Form as a Newtype

GPU implementations of modular arithmetic typically use Montgomery representation for efficient reduction. We encode this at the type level:

```rust
/// Montgomery representation of a field element.
/// The value stored is (a * R) mod P, where R = 2^(LIMB_BITS * NUM_LIMBS).
struct MontField<P: Prime> {
    mont_value: BigUint<limbs_for(P)>
}

/// Conversion between standard and Montgomery forms is explicit:
fn to_montgomery<P: Prime>(a: Field<P>) -> MontField<P>;
fn from_montgomery<P: Prime>(a: MontField<P>) -> Field<P>;

/// Arithmetic on MontField uses Montgomery multiplication.
/// You cannot accidentally mix Field<P> and MontField<P>:
impl<P: Prime> Mul for MontField<P> {
    type Output = MontField<P>;
    fn mul(self, rhs: MontField<P>) -> MontField<P> {
        // Montgomery multiplication: (a*R) * (b*R) * R^{-1} mod P = (a*b)*R mod P
        montgomery_mul(self.mont_value, rhs.mont_value)
    }
}
```

---

## 5. Fixed-Width Big Integer Types

### 5.1 Representation

Cryptographic computations require integers far wider than hardware-native 32/64-bit words. We define a family of fixed-width unsigned integer types parameterized by their bit width:

```rust
/// Fixed-width unsigned integer with W bits, stored as an array of 32-bit limbs.
/// W must be a multiple of 32.
struct UintN<const W: Nat>
where
    W % 32 == 0
{
    limbs: [u32; W / 32]
}

// Common aliases:
type uint128 = UintN<128>;  // 4 limbs
type uint256 = UintN<256>;  // 8 limbs
type uint384 = UintN<384>;  // 12 limbs (BLS12-381 field elements)
type uint512 = UintN<512>;  // 16 limbs
type uint1024 = UintN<1024>; // 32 limbs
type uint2048 = UintN<2048>; // 64 limbs (RSA)
```

### 5.2 Typing Arithmetic Operations

The result widths of arithmetic operations are computed at the type level:

```rust
/// Addition: the result may be one bit wider.
fn add<const W: Nat>(a: UintN<W>, b: UintN<W>) -> (UintN<W>, bool)
// Returns (result mod 2^W, carry_out)

/// Widening addition: no overflow possible.
fn wide_add<const W: Nat>(a: UintN<W>, b: UintN<W>) -> UintN<{W + 1}>

/// Multiplication: the result is double width.
fn mul<const W: Nat>(a: UintN<W>, b: UintN<W>) -> UintN<{W * 2}>
// uint256 * uint256 -> uint512

/// Widening operations for asymmetric widths:
fn mul_wide<const A: Nat, const B: Nat>(
    a: UintN<A>, b: UintN<B>
) -> UintN<{A + B}>

/// Shifting: result width is unchanged.
fn shl<const W: Nat>(a: UintN<W>, shift: u32) -> UintN<W>
where
    shift < W  // compile-time or runtime check
```

### 5.3 Truncation and Extension

Conversions between widths are explicit and typed:

```rust
/// Zero-extension (always safe, no information loss):
fn zext<const FROM: Nat, const TO: Nat>(a: UintN<FROM>) -> UintN<TO>
where
    TO >= FROM;

/// Truncation (potentially lossy, requires explicit call):
fn trunc<const FROM: Nat, const TO: Nat>(a: UintN<FROM>) -> UintN<TO>
where
    TO <= FROM;

/// Checked truncation (returns None if information would be lost):
fn try_trunc<const FROM: Nat, const TO: Nat>(
    a: UintN<FROM>
) -> Option<UintN<TO>>
where
    TO <= FROM;
```

### 5.4 Relationship to `Field<P>`

The internal representation of `Field<P>` is defined in terms of `UintN`:

```rust
/// Compute the number of limbs needed for a prime P.
const fn limbs_for(P: Nat) -> Nat {
    (bit_length(P) + 31) / 32
}

const fn bits_for(P: Nat) -> Nat {
    limbs_for(P) * 32
}

struct Field<P: Prime> {
    value: UintN<{bits_for(P)}>
}

// For BN254 (254-bit prime): Field<BN254_R> contains UintN<256>
// For BLS12-381 (381-bit prime): Field<BLS12_381_R> contains UintN<384>
```

### 5.5 GPU-Specific Considerations

On GPUs, limb operations map to hardware multiply-add instructions. The type system ensures correct limb counts flow through to kernel code generation:

```rust
/// A GPU kernel for big-integer multiplication.
/// The type system ensures the output buffer has the correct size.
#[kernel]
fn bigint_mul_kernel<const W: Nat>(
    a: &DeviceSlice<UintN<W>>,
    b: &DeviceSlice<UintN<W>>,
    out: &mut DeviceSlice<UintN<{W * 2}>>,
    len: u32,
) {
    let idx = thread_idx() + block_idx() * block_dim();
    if idx < len {
        out[idx] = mul(a[idx], b[idx]);
    }
}
```

---

## 6. Precision Types for ML

### 6.1 Floating-Point Type Hierarchy

LLM inference and training use a wide range of numerical precisions. We define a hierarchy of types with explicit bit-level semantics:

```rust
// IEEE 754 standard types
type f64 = Float<11, 52>;   // 1 sign + 11 exponent + 52 mantissa = 64 bits
type f32 = Float<8, 23>;    // 1 sign + 8 exponent + 23 mantissa = 32 bits
type f16 = Float<5, 10>;    // 1 sign + 5 exponent + 10 mantissa = 16 bits

// Brain floating-point
type bf16 = Float<8, 7>;    // 1 sign + 8 exponent + 7 mantissa = 16 bits

// FP8 formats (OFP8 / IEEE P3109)
type f8e4m3 = Float<4, 3>;  // 1 sign + 4 exponent + 3 mantissa = 8 bits
                              // Range: [-448, 448], no inf, has NaN
type f8e5m2 = Float<5, 2>;  // 1 sign + 5 exponent + 2 mantissa = 8 bits
                              // Range: [-57344, 57344], has inf and NaN

// Sub-byte types for weight quantization
type int4 = SintN<4>;       // Signed 4-bit integer, range [-8, 7]
type uint4 = UintN<4>;      // Unsigned 4-bit integer, range [0, 15]
type int8 = SintN<8>;       // Standard signed byte

// Parameterized float type
struct Float<const E: Nat, const M: Nat> {
    bits: UintN<{1 + E + M}>
}
```

### 6.2 Safe Conversion Rules

Not all precision conversions are safe. We define a lattice of widening conversions that never lose information, and require explicit casts for narrowing:

```
Widening (implicit, lossless):
  f8e4m3  -->  f16  -->  f32  -->  f64
  f8e5m2  -->  f16  -->  f32  -->  f64
  bf16    -->  f32  -->  f64
  int4    -->  int8  -->  f16 (exact for all int8 values)
  int8    -->  f32  (exact for all int8 values)

Narrowing (explicit cast required, potentially lossy):
  f32  --cast-->  f16       // may overflow or lose precision
  f32  --cast-->  bf16      // truncates mantissa
  f16  --cast-->  f8e4m3    // may overflow (f16 max > 448)
  f16  --cast-->  f8e5m2    // loses precision (only 2 mantissa bits)
  f32  --cast-->  int8      // rounds, may overflow
```

The typing rules:

```
-- Widening is implicit (subtyping):
Gamma |- e : f16
f16 <: f32              -- f16 widens to f32
--------------------------------------------------
Gamma |- e : f32

-- Narrowing requires an explicit cast with saturation or truncation mode:
Gamma |- e : f32
--------------------------------------------------
Gamma |- cast<f16, Saturate>(e) : f16     -- clamps to f16 range
Gamma |- cast<f16, Truncate>(e) : f16     -- truncates mantissa
Gamma |- cast<f16, RoundNearest>(e) : f16 -- rounds to nearest even
```

### 6.3 Mixed-Precision Kernel Typing

For LLM training with mixed precision, the type system enforces correct accumulation:

```rust
/// Matrix multiplication with typed precision policy.
/// Inputs in low precision, accumulation in high precision.
fn matmul_mixed<
    B: Nat, M: Nat, K: Nat, N: Nat,
    InType: FloatType,
    AccType: FloatType,
>(
    a: Tensor<InType, [B, M, K]>,
    b: Tensor<InType, [B, K, N]>,
) -> Tensor<AccType, [B, M, N]>
where
    InType: WidensTo<AccType>,  // e.g., f8e4m3 widens to f32
{
    // The hardware tensor core performs: acc += a[i,k] * b[k,j]
    // with accumulation in AccType precision
}

// Usage for FP8 training:
let logits: Tensor<f32, [B, S, V]> = matmul_mixed::<_, _, _, _, f8e4m3, f32>(
    activations,   // f8e4m3
    weights,       // f8e4m3
);
```

### 6.4 Scaling Factors

FP8 computations require per-tensor or per-block scaling factors to maintain accuracy. The type system can encode the scaling discipline:

```rust
/// A scaled tensor: the logical values are (elements * scale).
struct ScaledTensor<F: FloatType, Shape: ShapeType, ScaleType: FloatType> {
    data: Tensor<F, Shape>,
    scale: ScaleType,           // typically f32
    scale_mode: ScaleMode,      // PerTensor | PerChannel | PerBlock
}

/// Mixed-precision matmul with scaling is type-checked:
fn scaled_matmul<...>(
    a: ScaledTensor<f8e4m3, [M, K], f32>,
    b: ScaledTensor<f8e4m3, [K, N], f32>,
) -> ScaledTensor<f8e4m3, [M, N], f32>;
```

### 6.5 Format Selection Table

| Format | Bits | Exponent | Mantissa | Range | Use Case |
|--------|------|----------|----------|-------|----------|
| `f8e4m3` | 8 | 4 | 3 | ~[-448, 448] | Forward pass weights/activations |
| `f8e5m2` | 8 | 5 | 2 | ~[-57344, 57344] | Backward pass gradients |
| `bf16` | 16 | 8 | 7 | ~[-3.4e38, 3.4e38] | Training, large dynamic range |
| `f16` | 16 | 5 | 10 | ~[-65504, 65504] | Inference, higher precision |
| `int4` | 4 | - | - | [-8, 7] | Weight quantization (GPTQ/AWQ) |
| `f32` | 32 | 8 | 23 | ~[-3.4e38, 3.4e38] | Accumulation, master weights |

---

## 7. Constant-Time Type Qualifiers

### 7.1 Motivation

Cryptographic implementations must not leak secret data through timing side channels. Branching on secret data, secret-dependent memory access patterns, and variable-time instructions (division, some multiplications on certain architectures) all constitute timing leaks. The type system should make constant-time violations a compile error.

### 7.2 Secrecy Labels

We introduce two label types forming a two-point lattice:

```
Public < Secret
```

Every type is annotated with a secrecy label:

```
labeled_type ::= T @Public     -- public data (timing may depend on this)
               | T @Secret     -- secret data (timing must NOT depend on this)
```

### 7.3 Typing Rules

The core rules, inspired by FaCT (Flexible and Constant-Time Programming Language):

```
-- (T-If) Branching on secret data is forbidden:
Gamma |- cond : Bool @Public
Gamma |- e1 : T ! E
Gamma |- e2 : T ! E
--------------------------------------------------
Gamma |- if cond then e1 else e2 : T ! E

Gamma |- cond : Bool @Secret
--------------------------------------------------
Gamma |- if cond then e1 else e2 : TYPE ERROR
  "Cannot branch on secret data: timing side channel"

-- (T-Index) Array indexing by secret values is forbidden:
Gamma |- arr : Tensor<A @label, [n]>
Gamma |- idx : UintN<W> @Secret
--------------------------------------------------
Gamma |- arr[idx] : TYPE ERROR
  "Cannot index array with secret index: cache timing side channel"

-- (T-Asgn) Secret data cannot flow to public locations:
Gamma |- e : T @Secret
Gamma |- ref : &mut (T @Public)
--------------------------------------------------
Gamma |- *ref = e : TYPE ERROR
  "Cannot assign secret value to public location"

-- (T-Declassify) Explicit declassification:
Gamma |- e : T @Secret
--------------------------------------------------
Gamma |- declassify(e) : T @Public
  // Requires #[allow(declassify)] annotation on enclosing function
```

### 7.4 Constant-Time Operations

The type system restricts which operations are permitted on `@Secret` values:

```rust
// ALLOWED on @Secret values (constant-time):
impl<P: Prime> Add for Field<P> @Secret { /* constant-time modular add */ }
impl<P: Prime> Mul for Field<P> @Secret { /* constant-time Montgomery mul */ }
impl<P: Prime> Sub for Field<P> @Secret { /* constant-time modular sub */ }
fn ct_select<T>(cond: Bool @Secret, a: T @Secret, b: T @Secret) -> T @Secret;
fn ct_eq<T: Eq>(a: T @Secret, b: T @Secret) -> Bool @Secret;
fn ct_memcpy<T, n: Nat>(dst: &mut [T @Secret; n], src: &[T @Secret; n]);

// FORBIDDEN on @Secret values (variable-time):
// - Division (use Fermat inverse instead)
// - Branching (if/match on Secret)
// - Array indexing by Secret index
// - Early return based on Secret
// - Variable-iteration loops bounded by Secret
```

### 7.5 Practical Example: Constant-Time Point Multiplication

```rust
/// Scalar multiplication on an elliptic curve.
/// The scalar is secret; the implementation must be constant-time.
fn scalar_mul<P: Prime>(
    scalar: UintN<256> @Secret,
    point: ECPoint<P> @Public,
) -> ECPoint<P> @Secret
{
    let mut acc = ECPoint::identity();
    let mut temp = point;

    for i in 0..256 {
        let bit: Bool @Secret = scalar.bit(i);
        // ct_select is constant-time conditional move:
        // it always performs both branches and selects via masking
        acc = ct_select(bit, ec_add(acc, temp), acc);
        temp = ec_double(temp);
    }

    acc
}
```

### 7.6 GPU-Specific Constant-Time Considerations

GPU architectures introduce additional timing channels:

- **Warp divergence**: If threads in a warp branch differently, both paths execute sequentially. Secret-dependent branching causes divergence that leaks information across threads.
- **Memory coalescing**: Secret-dependent access patterns cause uncoalesced accesses with observable timing differences.
- **Shared memory bank conflicts**: Secret-dependent shared memory addresses cause bank conflicts.

The type system enforces uniform control flow within a warp:

```rust
#[kernel]
#[warp_uniform]  // All threads in a warp must follow the same control path
fn constant_time_kernel(
    secret_keys: &DeviceSlice<UintN<256> @Secret>,
    public_data: &DeviceSlice<ECPoint @Public>,
    results: &mut DeviceSlice<ECPoint @Secret>,
) {
    let tid = thread_idx();
    let key = secret_keys[tid];  // OK: each thread accesses its own index (public)
    let point = public_data[tid];
    results[tid] = scalar_mul(key, point);
}
```

---

## 8. Ownership and Borrowing for GPU Memory

### 8.1 Memory Spaces

GPU programming involves multiple memory spaces with different characteristics. We encode the memory space in the type:

```rust
/// Memory space marker types
enum MemSpace {
    Host,           // CPU RAM (pageable)
    HostPinned,     // CPU RAM (page-locked, DMA-capable)
    Device,         // GPU VRAM (global memory)
    DeviceShared,   // GPU on-chip shared memory (per-block)
    Unified,        // Unified Virtual Addressing (CPU + GPU)
}

/// A buffer parameterized by element type, size, and memory space.
struct Buffer<A: Type, const N: Nat, const S: MemSpace> {
    ptr: RawPtr<A, S>,
    len: N,
}
```

### 8.2 Ownership Rules

Each `Buffer` has exactly one owner. Ownership can be transferred (moved) but not duplicated:

```rust
// Ownership transfer (move):
let a: Buffer<f32, 1024, Device> = alloc_device(1024);
let b = a;  // ownership moves to b
// a is no longer valid here

// Ownership transfer across memory spaces:
fn upload<A, N>(src: Buffer<A, N, HostPinned>) -> Buffer<A, N, Device>;
fn download<A, N>(src: Buffer<A, N, Device>) -> Buffer<A, N, HostPinned>;
```

### 8.3 Borrowing Rules Adapted for GPU

The Rust-style borrowing rules are extended with memory-space-aware constraints:

```
-- Rule (Borrow-Shared): Multiple immutable borrows from any space
Gamma |- buf : Buffer<A, N, S>
--------------------------------------------------
Gamma |- &buf : &Buffer<A, N, S>     -- shared reference
// Multiple &buf allowed simultaneously

-- Rule (Borrow-Mut): Exclusive mutable borrow
Gamma |- buf : Buffer<A, N, S>
  no other borrows of buf exist
--------------------------------------------------
Gamma |- &mut buf : &mut Buffer<A, N, S>

-- Rule (Cross-Space-Borrow): A device buffer cannot be borrowed on the host
Gamma |- buf : Buffer<A, N, Device>
--------------------------------------------------
Gamma |- cpu_deref(&buf) : TYPE ERROR
  "Cannot dereference device memory on host"
```

### 8.4 Kernel Argument Passing

Kernel launches receive borrows of device buffers, not ownership:

```rust
#[kernel]
fn vector_add<N: Nat>(
    a: &DeviceSlice<f32, N>,       // immutable borrow
    b: &DeviceSlice<f32, N>,       // immutable borrow
    out: &mut DeviceSlice<f32, N>, // mutable borrow
) {
    let i = global_thread_id();
    if i < N {
        out[i] = a[i] + b[i];
    }
}

fn main() {
    let a = upload(host_a);    // Buffer<f32, 1024, Device>, owned
    let b = upload(host_b);    // Buffer<f32, 1024, Device>, owned
    let mut out = alloc_device::<f32, 1024>();

    // Kernel launch borrows the buffers; ownership remains with main
    launch!(vector_add::<1024>, grid(4), block(256), &a, &b, &mut out);

    // After synchronization, borrows are released
    device_sync();

    // We still own a, b, out and can use them:
    let result = download(out);  // moves out
    free(a);                     // explicitly free
    free(b);
    // out was moved into download, so no free needed
}
```

### 8.5 Lifetime Annotations for Async Operations

GPU operations are asynchronous. Borrows must outlive the asynchronous operation:

```rust
/// A stream-ordered operation that borrows a buffer.
/// The lifetime 'a ensures the buffer lives until stream completion.
fn async_copy<'a, A, N>(
    stream: &Stream,
    src: &'a Buffer<A, N, HostPinned>,
    dst: &'a mut Buffer<A, N, Device>,
) -> StreamEvent<'a>
where
    'a: 'stream  // buffer must outlive the stream operation
{
    // Initiates DMA transfer, returns immediately.
    // The StreamEvent holds the lifetime, preventing the
    // buffers from being freed before the transfer completes.
}
```

### 8.6 RAII and Drop for GPU Resources

When a buffer goes out of scope without being moved, its destructor runs:

```rust
impl<A, N, S: MemSpace> Drop for Buffer<A, N, S> {
    fn drop(&mut self) {
        match S {
            MemSpace::Device       => cuda_free(self.ptr),
            MemSpace::HostPinned   => cuda_free_host(self.ptr),
            MemSpace::Host         => free(self.ptr),
            MemSpace::Unified      => cuda_free_managed(self.ptr),
            MemSpace::DeviceShared => { /* freed when kernel exits */ }
        }
    }
}
```

For `SecureLinear` buffers (cryptographic keys), `Drop` is overridden to zero memory before deallocation.

---

## 9. Parametric Polymorphism and GPU Kernel Compilation

### 9.1 The Tension

Parametric polymorphism (generics) enables code reuse, but GPU compilation imposes severe constraints:

- GPU kernels are compiled to ISA-specific machine code (PTX/SASS for NVIDIA, AMDGCN for AMD).
- Register allocation, shared memory layout, and occupancy calculations depend on concrete types.
- Dynamic dispatch (vtables, function pointers) is either unsupported or extremely expensive on GPUs.
- Kernel launch configurations (grid/block dimensions) often depend on the concrete element type.

### 9.2 Monomorphization (Our Primary Strategy)

Like Rust, we use monomorphization: every instantiation of a generic function or kernel generates a specialized version:

```rust
#[kernel]
fn reduce_sum<T: Add + Zero, N: Nat>(
    input: &DeviceSlice<T, N>,
    output: &mut DeviceSlice<T, 1>,
) { /* ... */ }

// When called with concrete types:
reduce_sum::<f32, 1024>(...);    // generates reduce_sum_f32_1024
reduce_sum::<f16, 2048>(...);    // generates reduce_sum_f16_2048
reduce_sum::<Field<BN254_R>, 4096>(...);  // generates reduce_sum_FieldBN254_4096
```

Each monomorphized kernel is independently optimized:

```
reduce_sum_f32_1024:
  - Uses 32-bit float add (FADD.F32)
  - Shared memory: 1024 * 4 bytes = 4KB
  - Occupancy: 8 blocks per SM

reduce_sum_f16_2048:
  - Uses 16-bit float add (HADD2 for 2-at-a-time)
  - Shared memory: 2048 * 2 bytes = 4KB
  - Occupancy: 8 blocks per SM

reduce_sum_FieldBN254_4096:
  - Uses 256-bit modular addition (sequence of IADD3 + IMAD)
  - Shared memory: 4096 * 32 bytes = 128KB (may exceed SM limit!)
  - Occupancy: 1 block per SM
```

### 9.3 When Monomorphization Fails: Code Size

For large generic libraries, monomorphization can cause code bloat. Strategies to mitigate:

```rust
// Strategy 1: Factor out the monomorphic hot loop
#[kernel]
fn ntt_butterfly_f32(/* ... */) { /* fully specialized */ }

#[kernel]
fn ntt_butterfly_f64(/* ... */) { /* fully specialized */ }

// The generic wrapper only dispatches; it lives on the host
fn ntt<T: Field>(data: &mut DeviceBuffer<T, N>) {
    match T::TYPE_ID {
        TypeId::F32 => launch!(ntt_butterfly_f32, ...),
        TypeId::F64 => launch!(ntt_butterfly_f64, ...),
    }
}
```

```rust
// Strategy 2: Compile-time kernel registry
// All needed instantiations are registered at compile time.
#[instantiate(f16, f32, bf16, f8e4m3)]
#[kernel]
fn softmax<T: FloatType, N: Nat>(
    input: &DeviceSlice<T, N>,
    output: &mut DeviceSlice<T, N>,
) { /* ... */ }
```

### 9.4 Type Erasure: When and Why

Type erasure is appropriate for host-side code that does not touch GPU-resident data:

```rust
// Host-side graph construction can use trait objects (type-erased):
trait KernelNode {
    fn launch(&self, stream: &Stream);
    fn output_buffer(&self) -> &dyn DeviceBufferAny;
}

// The execution graph is type-erased:
struct ComputeGraph {
    nodes: Vec<Box<dyn KernelNode>>,  // type-erased
    edges: Vec<(usize, usize)>,
}

// But individual kernels are still monomorphized for GPU execution.
```

### 9.5 Formal Typing Rule for Monomorphization

```
-- Kernel generics must be fully resolvable at compile time:
Gamma |- K : forall <T1: C1, ..., Tk: Ck, N1: Nat, ..., Nm: Nat>. KernelType
Gamma |- T1 = tau1, ..., Tk = tauk    (concrete types)
Gamma |- N1 = n1, ..., Nm = nm        (concrete sizes)
--------------------------------------------------
Gamma |- K::<tau1, ..., tauk, n1, ..., nm> : ConcreteKernelType
  // Generates specialized PTX/AMDGCN code

-- Existential sizes are NOT permitted in kernel type parameters:
Gamma |- K : forall <N: Nat>. KernelType
Gamma |- n : Nat  (only known at runtime, i.e., existentially bound)
--------------------------------------------------
Gamma |- K::<n> : requires runtime compilation (JIT) or pre-compilation for all possible n
```

### 9.6 Comparison Table

| Strategy | Code Size | Runtime Perf | Compilation Time | GPU Suitability |
|----------|-----------|-------------|------------------|-----------------|
| Monomorphization | Large (O(instantiations)) | Optimal | Slow | Excellent |
| Type Erasure | Small | Overhead from indirection | Fast | Poor (no vtables on GPU) |
| JIT Specialization | N/A (runtime) | Near-optimal | Runtime cost | Good (e.g., Triton) |
| Hybrid | Medium | Good | Medium | Good |

---

## 10. Examples from Existing Languages

### 10.1 Futhark: Size-Dependent Types and Uniqueness

Futhark is a purely functional, data-parallel language that compiles to GPU code via OpenCL and CUDA backends. Its type system features:

**Size types**: Array dimensions appear in types, and the compiler checks dimensional consistency:

```futhark
-- The type of 'zip' requires both arrays to have the same length.
-- [n] is a size parameter.
val zip [n] 'a 'b : [n]a -> [n]b -> [n](a, b)

-- Matrix multiplication: inner dimensions must match
val matmul [m][n][p] : [m][n]f64 -> [n][p]f64 -> [m][p]f64

-- Existential sizes: filter returns an array of unknown length
val filter 'a : (a -> bool) -> [n]a -> ?[m].[m]a
```

**Uniqueness types** for in-place updates (preventing aliasing):

```futhark
-- The asterisk * marks a unique (non-aliased) parameter that can be updated in place.
let update [n] (xs: *[n]i32) (i: i64) (v: i32): *[n]i32 =
  xs with [i] = v

-- This is a type error:
let bad (xs: *[n]i32) =
  let ys = xs          -- xs is now aliased by ys
  let xs' = update xs 0 42  -- ERROR: xs is no longer unique
  ...
```

Futhark's uniqueness types serve the same purpose as Rust's ownership: preventing aliased mutation. The difference is that Futhark uses them primarily to enable in-place array updates within a purely functional framework, rather than for general resource management.

**Key limitation**: Futhark's size types operate only on array lengths, not on element types or more complex invariants. The arithmetic on size expressions is limited, and the compiler has historically had numerous bugs in this area, as documented in the Futhark blog post on "the biggest semantic mess."

### 10.2 Rust: Ownership and the Borrow Checker

Rust's ownership system is directly applicable to GPU resource management. Key rules:

```rust
// Rust ownership rules (standard):
// 1. Each value has exactly one owner.
// 2. When the owner goes out of scope, the value is dropped.
// 3. Ownership can be transferred (moved).
// 4. References (&T, &mut T) borrow without taking ownership.
// 5. At any time: either one &mut T, or any number of &T, never both.
```

**RustaCUDA** and **cust** demonstrate ownership applied to GPU memory:

```rust
// RustaCUDA: DeviceBox<T> owns a single value on the device.
// DeviceBuffer<T> owns a contiguous array on the device.
// Both implement Drop, which calls cudaFree.

use rustacuda::memory::DeviceBuffer;

let mut device_buf = DeviceBuffer::from_slice(&host_data)?;
// device_buf owns the GPU memory

kernel.launch(&[&device_buf])?;  // borrow for kernel launch
// device_buf is still owned here

let host_result = device_buf.as_host_vec()?;  // copy back
// device_buf dropped here -> cudaFree called
```

**wgpu** (Rust's WebGPU implementation) demonstrates ownership for cross-platform GPU resources:

```rust
// wgpu: Buffer creation returns an owned handle.
// Dropping the buffer reclaims GPU memory.
let buffer = device.create_buffer(&BufferDescriptor {
    label: Some("my_buffer"),
    size: 1024,
    usage: BufferUsages::STORAGE | BufferUsages::COPY_SRC,
    mapped_at_creation: false,
});
// buffer: wgpu::Buffer -- owned
// Usage flags are declared at creation time; wgpu inserts barriers automatically.
```

**Limitation for our design**: Rust's standard borrow checker does not reason about asynchronous GPU operations. A buffer borrow passed to an async kernel launch may be "returned" before the kernel completes. Our language extends lifetimes with stream-awareness (Section 8.5).

### 10.3 Idris and Agda: Full Dependent Types

Idris and Agda provide the theoretical gold standard for dependent types. Their type systems can express arbitrary propositions about program values.

**Agda: Length-indexed vectors with matrix multiplication**:

```agda
-- Vectors indexed by their length
data Vec (A : Set) : Nat -> Set where
  []  : Vec A 0
  _::_ : A -> Vec A n -> Vec A (suc n)

-- Matrix as vector of vectors
Mat : Set -> Nat -> Nat -> Set
Mat A m n = Vec (Vec A n) m

-- Matrix multiplication with full size correctness
matmul : {A : Set} {m n p : Nat} {{_ : Semiring A}}
       -> Mat A m n -> Mat A n p -> Mat A m p
```

**Idris 2: Quantitative Type Theory (QTT)**:

Idris 2 introduces multiplicity annotations that track how many times a variable is used, unifying linear types with dependent types:

```idris
-- (1 x : a) means x is used exactly once (linear)
-- (0 x : a) means x is erased at runtime (relevant only for types)
-- (w x : a) means x is used without restriction (unrestricted)

-- A file handle that must be closed exactly once:
openFile : (fname : String) -> (1 _ : (1 h : File) -> IO a) -> IO a

-- The continuation receives the file handle linearly (1 h : File),
-- guaranteeing it is used (and closed) exactly once.
```

This is directly relevant to our GPU resource management: QTT-style multiplicities can unify our linear buffer types with dependent size types in a single framework.

**Why we do not adopt full dependent types**: Full dependent types (as in Agda/Idris) make type checking undecidable in general, requiring termination checking and sometimes manual proofs. For a practical GPU programming language, we restrict to the decidable fragment of size-dependent types (Presburger arithmetic over natural numbers) combined with linear-type-style resource tracking. This gives most of the benefits without requiring the programmer to write proofs.

### 10.4 Triton: Python-Embedded Type Annotations for GPU Kernels

OpenAI's Triton takes a different approach: it is a Python-embedded DSL where GPU kernels are written as decorated Python functions, and type information is largely implicit.

```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr,  # compile-time constant
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Type annotations via tl.constexpr mark values known at compile time.
    # The compiler specializes the kernel for each (BLOCK_SIZE_M, N, K) tuple.
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # tl.arange(0, BLOCK_SIZE_M) returns a tl.tensor of shape (BLOCK_SIZE_M,)
    # The shape is known at compile time because BLOCK_SIZE_M is tl.constexpr.

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    # ...
```

Triton's `tl.constexpr` serves the same role as our `Nat`-kinded type parameters: values that are known at compile time and used for kernel specialization. The key differences:

| Feature | Triton | Our Language |
|---------|--------|-------------|
| Shape checking | Runtime (Python semantics) | Compile-time (dependent types) |
| Memory safety | None (raw pointers) | Ownership + borrowing |
| Effect tracking | None | Row-polymorphic effects |
| Precision control | `dtype` parameter | Type-level precision lattice |
| Constant-time | Not addressed | `@Secret` / `@Public` labels |
| Polymorphism | Python duck typing + JIT | Monomorphization + constrained generics |

Triton's strength is accessibility: Python programmers can write GPU kernels without learning a new language. Its weakness is the lack of static guarantees: shape mismatches, memory errors, and precision bugs manifest only at runtime.

### 10.5 Synthesis: How Our Design Draws from Each

```
                      Our GPU Language for Crypto + LLMs
                     /        |         |        \        \
                    /         |         |         \        \
             Futhark       Dex       Rust      Idris     Triton
              |             |         |          |         |
         Size-dependent  Typed     Ownership   QTT      JIT
         types +         index     + borrow   multi-   special-
         uniqueness      sets      checker    plicity  ization
              |             |         |          |         |
              v             v         v          v         v
         Tensor shape    Index     GPU mem    Linear    Compile-
         constraints     safety   safety +   types +   time
         in types        via      RAII for   dependent  kernel
                        Fin n    device buf  types     params
```

**From Futhark** we take: size-dependent types with existential quantification, and uniqueness types adapted into full linear/affine types.

**From Dex** we take: the concept of typed index sets (our `Fin n`) and the integration of dependent types with functional array programming.

**From Rust** we take: ownership, borrowing, and lifetimes, extended to model GPU memory spaces and asynchronous kernel lifetimes.

**From Idris/Agda** we take: the theoretical foundation of dependent types, and specifically Idris 2's quantitative type theory as inspiration for unifying linear and dependent types.

**From Triton** we take: the ergonomic design of `constexpr` parameters for kernel specialization, and the JIT compilation model as a fallback when monomorphization is insufficient.

---

## Appendix A: Summary of Type Formation Rules

```
Sorts:
  Nat : Sort                              -- natural numbers
  Type : Sort                             -- types of values
  Label : Sort                            -- {Public, Secret}
  MemSpace : Sort                         -- {Host, HostPinned, Device, ...}

Type constructors:
  UintN<W: Nat> : Type                    where W % 32 == 0
  SintN<W: Nat> : Type
  Float<E: Nat, M: Nat> : Type
  Field<P: Prime> : Type
  MontField<P: Prime> : Type
  Tensor<A: Type, shape: [Nat]> : Type
  Buffer<A: Type, N: Nat, S: MemSpace> : Type
  Fin<N: Nat> : Type

Type qualifiers:
  T @Secret : Type                        -- secret-labeled type
  T @Public : Type                        -- public-labeled type
  Linear<T> : Type                        -- must be consumed exactly once
  Affine<T> : Type                        -- may be consumed at most once

Function types:
  (A1, ..., Ak) ->{E} R                  -- function with effect row E

Effect rows:
  E ::= <> | <e1, ..., en> | <e1, ..., en | rho>

Size expressions:
  s ::= n | k | s+s | s*s | s/s | pow2(s) | max(s,s) | min(s,s)
```

## Appendix B: Type Checking Algorithm Sketch

1. **Size inference**: Unify size expressions using a constraint solver over Presburger arithmetic. Size variables are inferred by Hindley-Milner-style unification extended with arithmetic constraints.

2. **Linearity checking**: After standard type checking, perform a separate pass that counts the number of uses of each linear/affine variable. Reject programs where a linear variable is used zero or more than one times, or an affine variable is used more than one time.

3. **Effect inference**: Infer effect rows using row-polymorphic unification. Each function body's effects are computed as the union of effects of its sub-expressions, modulo effect handlers that remove effects.

4. **Secrecy checking**: Perform information flow analysis using the two-point lattice {Public < Secret}. Check that no secret data influences branch conditions, loop bounds, or memory access indices.

5. **Monomorphization**: After type checking, instantiate all generic kernels with their concrete type arguments. Reject kernel instantiations where type arguments are not fully determined at compile time (unless JIT fallback is enabled).

---

## Appendix C: References and Further Reading

- **Futhark size types**: Henriksen, T., & Elsman, M. "Towards Size-Dependent Types for Array Programming." ARRAY 2021.
- **Futhark uniqueness types**: Futhark blog, "Uniqueness Types and In-Place Updates." 2022.
- **Dex typed indices**: Paszke, A., et al. "Dex: Array Programming with Typed Indices." NeurIPS 2019 workshop.
- **FaCT constant-time DSL**: Cauligi, S., et al. "FaCT: A DSL for Timing-Sensitive Computation." PLDI 2019.
- **Idris 2 QTT**: Brady, E. "Idris 2: Quantitative Type Theory in Practice." ECOOP 2021.
- **Koka effects**: Leijen, D. "Koka: Programming with Row-Polymorphic Effect Types." 2014.
- **Rust ownership**: Weiss, A., et al. "Oxide: The Essence of Rust." arXiv 2019.
- **FP8 specification**: NVIDIA, ARM, Intel. "FP8 Formats for Deep Learning." 2022.
- **Triton**: Tillet, P., et al. "Triton: An Intermediate Language and Compiler for Tiled Neural Network Computations." MLSys 2019.
- **Monomorphization on GPUs**: Zhang, M. "Characterizing Massively Parallel Polymorphism." ISPASS 2021.
