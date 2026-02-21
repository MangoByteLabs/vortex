# Vortex Compilation Pipeline

## Overview

Vortex compiles `.vx` source files through a multi-stage pipeline that targets
GPU execution via MLIR. The full pipeline is:

```
.vx source
  |
  v
[Lexer]          -- logos-based tokenizer (src/lexer.rs)
  |
  v
[Parser]         -- recursive descent + Pratt precedence (src/parser.rs)
  |
  v
[AST]            -- typed abstract syntax tree (src/ast.rs)
  |
  v
[Type Checker]   -- two-pass: collect + check (src/typeck.rs)
  |
  v
[MLIR Codegen]   -- textual MLIR IR (src/codegen.rs)
  |
  v
[mlir-opt]       -- MLIR optimization passes
  |
  v
[mlir-translate] -- MLIR to LLVM IR
  |
  v
[llc]            -- LLVM IR to target assembly (PTX / AMDGCN / x86)
  |
  v
[ptxas / ld]     -- final binary / kernel object
```

## Installing MLIR Tools

### Ubuntu / Debian (recommended)

LLVM 20 packages include MLIR tools:

```bash
# Install mlir-opt, mlir-translate, etc.
apt-get install -y mlir-20-tools

# Verify installation
mlir-opt-20 --version
```

For other LLVM versions:
```bash
apt-get install -y mlir-18-tools   # LLVM 18
apt-get install -y mlir-19-tools   # LLVM 19
```

### From LLVM releases

Download pre-built binaries from https://github.com/llvm/llvm-project/releases.
Look for `clang+llvm-*-x86_64-linux-gnu-ubuntu-*.tar.xz` which includes
`bin/mlir-opt` and `bin/mlir-translate`.

### Building from source

```bash
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release
ninja -C build mlir-opt mlir-translate
```

## Pipeline Stages

### 1. Lexer (src/lexer.rs)

Tokenizes Vortex source using the `logos` crate. Produces 60+ token types
including keywords (`fn`, `kernel`, `struct`, `enum`, `match`), operators
(`@` for matmul, `.*` / `./` for elementwise), and literals.

### 2. Parser (src/parser.rs)

Recursive descent parser with Pratt precedence climbing for expressions.
Produces a fully typed AST. Key features:
- No semicolons (newline-sensitive)
- `kernel` keyword for GPU functions
- Schedule annotations for kernel tuning
- Generic types (`Tensor<f32, [N, M]>`)

### 3. Type Checker (src/typeck.rs)

Two-pass type checking:
- **Pass 1 (collect):** Register all function signatures, struct definitions,
  enum variants, trait declarations, and impl blocks
- **Pass 2 (check):** Verify expressions, function calls, match exhaustiveness

### 4. MLIR Codegen (src/codegen.rs)

Generates textual MLIR IR using these dialects:
- **func** -- Function definitions and calls
- **arith** -- Arithmetic operations (addi, addf, cmpi, cmpf, etc.)
- **scf** -- Structured control flow (if/else, for, while)
- **gpu** -- GPU kernel dispatch (gpu.module, gpu.func)
- **memref** -- Memory references for arrays
- **linalg** -- Linear algebra operations (matmul, generic)

Key design decisions:
- Kernels emit `gpu.module` + `gpu.func` with `kernel` attribute
- Regular functions emit `func.func`
- SSA values are auto-numbered (`%0`, `%1`, ...)
- Type casts are inserted automatically when operand types mismatch
- Early returns inside `scf` regions are commented out (MLIR limitation)

### 5. mlir-opt (external tool)

Runs MLIR optimization and lowering passes. Example pass pipelines:

```bash
# Verify MLIR is syntactically correct
mlir-opt-20 input.mlir

# Lower to LLVM dialect
mlir-opt-20 input.mlir \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts

# GPU lowering pipeline
mlir-opt-20 input.mlir \
  --gpu-kernel-outlining \
  --convert-gpu-to-nvvm \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --gpu-to-llvm
```

### 6. mlir-translate (external tool)

Converts MLIR (in LLVM dialect) to LLVM IR:

```bash
mlir-translate-20 --mlir-to-llvmir output.mlir -o output.ll
```

### 7. llc (LLVM backend)

Compiles LLVM IR to target-specific assembly:

```bash
# CPU target
llc-20 output.ll -o output.s

# NVIDIA PTX
llc-20 output.ll -march=nvptx64 -mcpu=sm_80 -o output.ptx

# AMD GCN
llc-20 output.ll -march=amdgcn -mcpu=gfx90a -o output.s
```

### 8. Final assembly

```bash
# NVIDIA: assemble PTX to cubin
ptxas -arch=sm_80 output.ptx -o output.cubin

# AMD: assemble to HSA code object
ld.lld output.s -o output.hsaco
```

## Targeting Different Backends

### CPU

```bash
vortex codegen program.vx > program.mlir
mlir-opt-20 program.mlir \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --convert-arith-to-llvm \
  --reconcile-unrealized-casts \
  -o program_llvm.mlir
mlir-translate-20 --mlir-to-llvmir program_llvm.mlir -o program.ll
llc-20 program.ll -o program.s
gcc program.s -o program
```

### NVIDIA GPU

```bash
vortex codegen program.vx > program.mlir
mlir-opt-20 program.mlir \
  --gpu-kernel-outlining \
  --convert-gpu-to-nvvm \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --gpu-to-llvm \
  -o program_gpu.mlir
mlir-translate-20 --mlir-to-llvmir program_gpu.mlir -o program.ll
llc-20 program.ll -march=nvptx64 -mcpu=sm_80 -o program.ptx
ptxas -arch=sm_80 program.ptx -o program.cubin
```

### AMD GPU

```bash
vortex codegen program.vx > program.mlir
mlir-opt-20 program.mlir \
  --gpu-kernel-outlining \
  --convert-gpu-to-rocdl \
  --convert-scf-to-cf \
  --convert-func-to-llvm \
  --gpu-to-llvm \
  -o program_gpu.mlir
mlir-translate-20 --mlir-to-llvmir program_gpu.mlir -o program.ll
llc-20 program.ll -march=amdgcn -mcpu=gfx90a -o program.s
```

## Quick Start Example

```bash
# Generate MLIR from a Vortex program
cargo run -- codegen examples/vector_add.vx > vector_add.mlir

# Verify it's valid MLIR
mlir-opt-20 vector_add.mlir

# Run the interpreter instead (no MLIR tools needed)
cargo run -- run examples/vector_add.vx
```

## Validation

The codegen module includes a built-in MLIR validator (`validate_mlir()`) that
checks for common structural errors without requiring external tools:

- Duplicate SSA value definitions
- Missing block terminators (func.return / gpu.return)
- func.return / gpu.return inside scf regions (invalid in MLIR)
- Double terminators (consecutive return statements)
- Unclosed regions

Run validation tests:
```bash
cargo test -- codegen::tests::test_validate
```

Run integration tests with mlir-opt-20 (requires mlir-20-tools):
```bash
cargo test -- codegen::tests::test_mlir_opt
```
