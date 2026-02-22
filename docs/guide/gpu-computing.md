# GPU Computing in Vortex

Vortex compiles to GPU through an MLIR-based pipeline that targets both NVIDIA and AMD hardware.

## How It Works

```
Vortex Source (.vx)
       |
   [Lexer + Parser]
       |
      AST
       |
   [Codegen]
       |
   MLIR IR (func/arith/scf/gpu dialects)
       |
   [mlir-opt]  -- optimization passes
       |
   [mlir-translate]  -- lower to LLVM IR
       |
   [llc]  -- compile to PTX or AMDGCN
       |
   GPU Binary
```

### Checking the Toolchain

```bash
vortex toolchain
```

This checks for: `mlir-opt`, `mlir-translate`, `llc`, `clang`, `nvidia-smi`.

## Writing Kernel Functions

Use the `kernel` keyword to define GPU functions:

```vortex
kernel vector_add(a: [f32], b: [f32], c: [f32], n: i32) {
    let idx = gpu_thread_id_x()
    if idx < n {
        c[idx] = a[idx] + b[idx]
    }
}
```

Kernel functions compile to `gpu.module` + `gpu.func` in MLIR output, which then lower to PTX or AMDGCN instructions.

## Generating MLIR

```bash
vortex codegen mykernel.vx
```

This prints the MLIR IR to stdout. Example output:

```mlir
module {
  gpu.module @kernels {
    gpu.func @vector_add(%a: memref<?xf32>, %b: memref<?xf32>,
                         %c: memref<?xf32>, %n: i32)
        kernel {
      %tid = gpu.thread_id x
      ...
      gpu.return
    }
  }
}
```

## Compiling to GPU

```bash
# Compile to NVIDIA PTX
vortex compile mykernel.vx ptx

# Compile to AMD GCN
vortex compile mykernel.vx amdgcn

# Compile to LLVM IR
vortex compile mykernel.vx llvm

# Compile to native object file
vortex compile mykernel.vx native

# Just output optimized MLIR
vortex compile mykernel.vx mlir
```

### Specifying GPU Architecture

```bash
vortex compile mykernel.vx ptx sm_80     # A100
vortex compile mykernel.vx ptx sm_89     # RTX 4090
vortex compile mykernel.vx amdgcn gfx90a # MI250
```

## GPU Runtime Builtins

Vortex provides GPU compute builtins that use a CPU fallback when no GPU is available:

### Memory Management

```vortex
let buf = gpu_alloc(size)           // Allocate GPU buffer
gpu_free(buf)                       // Free GPU buffer
gpu_copy_to_device(host_data)       // Upload to GPU
let data = gpu_copy_to_host(buf)    // Download from GPU
```

### Compute Operations

```vortex
let c = gpu_matmul(a, b)           // Matrix multiplication
let c = gpu_add(a, b)             // Elementwise addition
let c = gpu_mul(a, b)             // Elementwise multiplication
let c = gpu_relu(a)               // ReLU activation
let c = gpu_softmax(a)            // Softmax
```

### Status and Benchmarking

```vortex
let available = gpu_available()     // Check if GPU is present
gpu_native_matmul(a, b)           // Use native GPU matmul
gpu_train_step(model, data, lr)    // GPU-accelerated training step
gpu_benchmark(op, size, iters)     // Benchmark a GPU operation
```

## Quantization

For efficient inference on GPU:

```vortex
let q = quantize(tensor, bits)           // Quantize to N bits
let t = dequantize(q)                    // Dequantize back to float
let c = quantized_matmul(qa, qb)        // Quantized matrix multiply
let ratio = compression_ratio(q)         // Check compression ratio
```

## Flash Attention

Hardware-efficient attention computation:

```vortex
let output = flash_attention(q, k, v, block_size)
let grads = flash_attention_backward(q, k, v, grad_output, block_size)
```

## Example: GPU Matrix Multiply

```vortex
kernel matmul(A: [f32], B: [f32], C: [f32], M: i32, N: i32, K: i32) {
    let row = gpu_thread_id_y()
    let col = gpu_thread_id_x()
    if row < M {
        if col < N {
            var sum = 0.0
            for k in 0..K {
                sum = sum + A[row * K + k] * B[k * N + col]
            }
            C[row * N + col] = sum
        }
    }
}

fn main() {
    let A = gpu_copy_to_device([1.0, 2.0, 3.0, 4.0])
    let B = gpu_copy_to_device([5.0, 6.0, 7.0, 8.0])
    let C = gpu_alloc(4)

    // Launch kernel (conceptual -- actual dispatch uses MLIR pipeline)
    // matmul<<<grid, block>>>(A, B, C, 2, 2, 2)

    let result = gpu_copy_to_host(C)
    println(result)
}
```

## Multi-Backend Compilation

Vortex can target multiple backends from the same source:

```bash
# Generate MLIR and let the pipeline choose the best backend
vortex compile model.vx mlir

# Explicitly target NVIDIA
vortex compile model.vx ptx sm_80

# Explicitly target AMD
vortex compile model.vx amdgcn gfx90a
```

The compilation pipeline is modular: each stage produces an artifact, and you can inspect intermediate outputs for debugging.
