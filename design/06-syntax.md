# 06 — Syntax & Grammar

## Design Goals

- Familiar to Rust/C++ programmers
- Minimal boilerplate for common GPU patterns
- Clear visual distinction between host and device code
- First-class support for mathematical notation where practical

## Lexical Structure

### Keywords

```
// Declarations
fn kernel struct enum trait impl type alias const let mut

// Control flow
if else match for while loop break continue return

// GPU-specific
@shared @global @local @constant_time @schedule @tile @fuse @unroll
workgroup warp thread sync_warp sync_block

// Types
Tensor Field Curve Point mod

// Modules
import from pub use as

// Other
where async await pipeline stage
```

### Operators

```
// Arithmetic (work on scalars, tensors, and field elements)
+  -  *  /  %  **

// Bitwise
&  |  ^  ~  <<  >>

// Comparison
==  !=  <  >  <=  >=

// Logical
&&  ||  !

// Tensor
@    // matrix multiply (A @ B)
.*   // elementwise multiply
./   // elementwise divide
..   // range

// Field arithmetic
%+  %-  %*  %/  %**   // modular arithmetic operators (explicit)
                        // (Field<P> types use +,-,*,/,** automatically)

// Assignment
=  +=  -=  *=  /=  @=
```

## Type Syntax

### Primitive Types

```vortex
// Integers
u8  u16  u32  u64  u128  u256  u512
i8  i16  i32  i64  i128

// Floating point
f8e4m3  f8e5m2  // FP8 formats (ML quantization)
f16  bf16       // half precision
f32  f64        // standard precision
tf32            // tensor float (NVIDIA specific)

// Boolean
bool

// Void
void
```

### Compound Types

```vortex
// Tensors (shape-typed)
Tensor<f32, [3, 4]>           // 3x4 matrix of f32
Tensor<bf16, [B, S, D]>       // dynamic dimensions (compile-time symbolic)
Tensor<u32, [N]>              // vector of length N

// Tuples
(f32, f32, f32)
(Tensor<f16, [N]>, u32)

// Arrays (fixed-size, stack-allocated)
[f32; 256]
[u64; 4]                      // used for 256-bit integers as limbs

// Slices
&[f32]
&mut [f32]

// Optional / Result
Option<T>
Result<T, E>
```

### Crypto Types

```vortex
// Finite fields — the modulus is part of the type
type Fp = Field<0x30644e72e131a029b85045b68181585d97816a916871ca8d3c208c16d87cfd47>
type Fr = Field<0x30644e72e131a029b85045b68181585d2833e84879b9709143e1f593f0000001>

// Elliptic curves
type BN254 = Curve<Weierstrass, Fp, a=0, b=3>
type G1 = Point<BN254>
type G2 = Point<BN254.twist()>

// Polynomials over fields
Poly<Fr, N>          // polynomial of degree < N over Fr
SparsePolgy<Fr>      // sparse representation
```

### ML Types

```vortex
// Quantized types
Quantized<i8, scale=f32, zero_point=i8>   // per-tensor quantization
Quantized<i4, scale=f32, group_size=128>  // group quantization (GPTQ-style)

// Structured types for model weights
struct TransformerWeights<T: Float> {
    qkv: Tensor<T, [3 * D, D]>,
    out_proj: Tensor<T, [D, D]>,
    gate: Tensor<T, [FFN, D]>,
    up: Tensor<T, [FFN, D]>,
    down: Tensor<T, [D, FFN]>,
    norm: Tensor<T, [D]>,
    ff_norm: Tensor<T, [D]>,
}
```

## Functions and Kernels

### Regular Functions (host or device, inferred)

```vortex
fn add(a: f32, b: f32) -> f32 {
    return a + b
}

// Short form for single-expression functions
fn add(a: f32, b: f32) -> f32 = a + b
```

### GPU Kernels

```vortex
// A kernel is a function that executes on the GPU
kernel vector_add(a: Tensor<f32, [N]>, b: Tensor<f32, [N]>) -> Tensor<f32, [N]> {
    return a + b
}

// Kernels can have scheduling annotations
@schedule(tile=[128], num_warps=4)
kernel matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    // Block-level programming (Triton-style)
    let block_m = workgroup.id.x * 128
    let block_n = workgroup.id.y * 128

    var acc = Tensor.zeros<f32, [128, 128]>(@shared)

    for k_start in range(0, K, step=64) {
        let a_tile = a[block_m..block_m+128, k_start..k_start+64].load(@shared)
        let b_tile = b[k_start..k_start+64, block_n..block_n+128].load(@shared)
        sync_block()
        acc += a_tile @ b_tile
        sync_block()
    }

    return acc.store(@global, [block_m..block_m+128, block_n..block_n+128])
}
```

### Generic Functions

```vortex
fn dot<T: Numeric, const N: usize>(a: Tensor<T, [N]>, b: Tensor<T, [N]>) -> T {
    return reduce(a * b, op=Add)
}

// Trait bounds
fn ntt<F: PrimeField, const N: usize>(
    poly: Tensor<F, [N]>,
    roots: Tensor<F, [N]>,
) -> Tensor<F, [N]>
where N: PowerOfTwo
{
    // NTT butterfly implementation
    ...
}
```

## Control Flow

```vortex
// If/else (expression-based)
let max = if a > b { a } else { b }

// Match (exhaustive pattern matching)
match precision {
    .f16 => quantize_f16(x),
    .bf16 => quantize_bf16(x),
    .f8e4m3 => quantize_f8(x),
}

// For loops (parallel by default in kernels)
for i in 0..N {
    output[i] = input[i] * scale
}

// Explicit sequential loop (rare — use when order matters)
@sequential
for i in 0..N {
    prefix_sum[i] = prefix_sum[i-1] + data[i]
}

// While loops (host-side or convergence loops)
while delta > epsilon {
    delta = iterate(state)
}
```

## Memory Annotations

```vortex
// Explicit memory space placement
let shared_buf: Tensor<f32, [256]> @shared
let registers: [f32; 8] @local
let global_data: Tensor<f32, [N]> @global

// Async memory operations
let tile = async_load(global_ptr, @shared)
// ... do compute ...
await tile  // ensure load is complete
```

## Modules and Imports

```vortex
// File: crypto/fields/bn254.vx
pub type Fp = Field<0x30644e...>
pub type Fr = Field<0x30644e...01>

pub fn inv(x: Fp) -> Fp { ... }
pub fn sqrt(x: Fp) -> Option<Fp> { ... }

// File: main.vx
import crypto.fields.bn254 { Fp, Fr }
import nn.attention { flash_attention }
import nn.norm { rms_norm, layer_norm }

// Wildcard import (discouraged but available)
import crypto.fields.bn254 { * }

// Aliased import
import crypto.fields.bn254 { Fp as BN254Fp }
```

## Traits (Interfaces)

```vortex
trait Numeric {
    fn zero() -> Self
    fn one() -> Self
    fn add(self, other: Self) -> Self
    fn mul(self, other: Self) -> Self
}

trait PrimeField: Numeric {
    const MODULUS: [u64; 4]
    const GENERATOR: Self

    fn inv(self) -> Self
    fn pow(self, exp: u64) -> Self
    fn to_montgomery(self) -> Self
    fn from_montgomery(self) -> Self
}

trait Float: Numeric {
    fn exp(self) -> Self
    fn log(self) -> Self
    fn sqrt(self) -> Self
    fn recip(self) -> Self
}

// Implement traits for types
impl Numeric for Field<P> {
    fn zero() -> Self = Field(0)
    fn one() -> Self = Field(1)
    fn add(self, other: Self) -> Self = self %+ other
    fn mul(self, other: Self) -> Self = self %* other
}
```

## Pipeline Syntax (Multi-GPU)

```vortex
// Pipeline parallelism for large models
pipeline llm_inference(input: Tensor<u32, [B, S]>) -> Tensor<f32, [B, S, V]> {
    stage @gpu(0) {
        let embedded = embed(input, weights.embedding)
        let x = transformer_block(embedded, weights.layers[0..16])
    }
    stage @gpu(1) {
        let x = transformer_block(x, weights.layers[16..32])
        let logits = matmul(rms_norm(x, weights.final_norm), weights.lm_head)
    }
    return logits
}
```

## Comptime (Compile-Time Evaluation)

```vortex
// Compile-time computation (Zig-inspired)
comptime fn optimal_tile_size(M: usize, N: usize, shared_mem: usize) -> (usize, usize) {
    // This runs at compile time to determine optimal tiling
    let max_tile = sqrt(shared_mem / sizeof(f32))
    return (min(M, max_tile), min(N, max_tile))
}

// Comptime-known values can be used as type parameters
const TILE = comptime optimal_tile_size(4096, 4096, 48 * 1024)
kernel my_matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    @schedule(tile=[TILE.0, TILE.1])
    ...
}
```

## Error Handling

```vortex
// Result type for fallible operations
fn load_weights(path: &str) -> Result<TransformerWeights<f16>, IOError> {
    let file = File.open(path)?   // ? operator propagates errors
    let weights = deserialize<TransformerWeights<f16>>(file)?
    return Ok(weights)
}

// GPU kernels cannot fail at runtime (no Result return)
// All validation happens before kernel launch
kernel safe_matmul(a: Tensor<f32, [M, K]>, b: Tensor<f32, [K, N]>) -> Tensor<f32, [M, N]> {
    // Shape compatibility is checked at compile time via the type system
    return a @ b
}
```

## Comments and Documentation

```vortex
// Single-line comment

/* Multi-line
   comment */

/// Documentation comment (markdown supported)
///
/// # Example
/// ```
/// let result = ntt(poly, roots)
/// ```
pub fn ntt<F: PrimeField, const N: usize>(...) -> ... { ... }
```

## File Extension

`.vx` — short, distinctive, no conflicts with existing languages.
