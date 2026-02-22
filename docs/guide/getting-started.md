# Getting Started with Vortex

## Installation

Vortex is written in Rust. Build from source:

```bash
git clone https://github.com/MangoByteLabs/vortex.git
cd vortex
cargo build --release
```

The binary will be at `target/release/vortex`. Add it to your PATH:

```bash
export PATH="$PWD/target/release:$PATH"
```

### Requirements

- **Rust 1.70+** (install via [rustup](https://rustup.rs))
- **Optional for GPU**: MLIR toolchain (`mlir-opt`, `mlir-translate`), LLVM (`llc`, `clang`), NVIDIA CUDA toolkit or AMD ROCm

Check your toolchain status:

```bash
vortex toolchain
```

## Hello World

Create a file `hello.vx`:

```vortex
fn main() {
    println("Hello, Vortex!")
}
```

Run it:

```bash
vortex run hello.vx
```

Output:
```
Hello, Vortex!
```

## Running Programs

Vortex has several execution modes:

| Command | Description |
|---------|-------------|
| `vortex run file.vx` | Interpret and execute (tree-walking interpreter) |
| `vortex vm file.vx` | Execute on the bytecode VM |
| `vortex repl` | Interactive REPL |
| `vortex check file.vx` | Type-check without running |
| `vortex parse file.vx` | Print the AST |
| `vortex codegen file.vx` | Generate MLIR IR |
| `vortex compile file.vx [target]` | Compile to PTX, AMDGCN, LLVM IR, or native |
| `vortex toolchain` | Check available compilation tools |
| `vortex mcp` | Start the MCP server for AI agents |
| `vortex lsp` | Start the LSP server for editors |
| `vortex serve file.vx` | Run as an HTTP server |
| `vortex debug file.vx` | Interactive debugger |
| `vortex profile file.vx` | Profile execution |

### The REPL

```bash
vortex repl
```

```
Vortex REPL v0.1.0
Type :help for help, :quit to exit

vx> let x = 42
vx> println(x * 2)
84
vx> :env
  x = 42
vx> :quit
Goodbye!
```

## Your First Neural Network

Create `xor.vx`:

```vortex
fn main() {
    let l1 = nn_linear(2, 8)
    let l2 = nn_linear(8, 1)
    let model = nn_sequential([l1, l2])

    let data   = [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]]
    let labels = [[0.0], [1.0], [1.0], [0.0]]

    nn_train(model, data, labels, "adam", 1000, 0.01)

    for x in data {
        println(nn_predict(model, x))
    }
}
```

```bash
vortex run xor.vx
```

Five lines to define, train, and evaluate a neural network. No imports, no framework setup.

## Project Structure

For larger projects, use the package manager:

```bash
vortex init my-project        # Creates vortex.toml
vortex add some-package       # Add a dependency
vortex install                # Resolve and install dependencies
```

A typical Vortex project:

```
my-project/
  vortex.toml          # Package manifest
  src/
    main.vx            # Entry point
    lib.vx             # Library module
  examples/
    demo.vx
```

### Module System

Import other Vortex files:

```vortex
import std.tensor { Tensor }
import std.crypto { sha256 }
```

The standard library lives in `stdlib/` and provides common math, tensor, neural network, and crypto utilities.

## Next Steps

- [Language Reference](language-reference.md) -- Learn the full syntax
- [Neural Networks](neural-networks.md) -- Training models in depth
- [GPU Computing](gpu-computing.md) -- Compiling to GPU
- [Builtins Reference](builtins-reference.md) -- All 200+ built-in functions
- [Examples](examples.md) -- Real code to learn from
