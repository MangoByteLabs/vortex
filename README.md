# Vortex

**A GPU programming language built for the era of intelligent machines.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Build](https://img.shields.io/badge/build-passing-brightgreen.svg)]()
[![Rust](https://img.shields.io/badge/rust-1.70%2B-orange.svg)]()

Vortex is a programming language designed from the ground up for AI and GPU computing. It has neural network primitives, automatic differentiation, cryptographic operations, and GPU compilation built directly into the language -- no frameworks, no Python glue, no CUDA boilerplate.

## Quick Start

```bash
# 1. Clone and build
git clone https://github.com/MangoByteLabs/vortex.git
cd vortex
cargo build --release

# 2. Run your first program
echo 'fn main() { println("Hello, Vortex!") }' > hello.vx
./target/release/vortex run hello.vx

# 3. Train a neural network
./target/release/vortex run examples/train_xor.vx
```

## What Makes Vortex Different

### Train a neural network in 5 lines

```vortex
fn main() {
    let l1 = nn_linear(2, 8)
    let l2 = nn_linear(8, 1)
    let model = nn_sequential([l1, l2])
    nn_train(model, data, labels, "adam", 1000, 0.01)
    for x in data { println(nn_predict(model, x)) }
}
```

### Built-in cryptography

```vortex
fn main() {
    let sk = bigint_from_hex("DEADBEEF...")
    let pubkey = scalar_mul(sk, secp256k1_generator())
    let sig = ecdsa_sign(sk_hex, "Hello Vortex!")
    assert(ecdsa_verify(pubkey, "Hello Vortex!", sig), "invalid!")
}
```

### Compile to GPU

```vortex
kernel vector_add(a: [f32], b: [f32], c: [f32], n: i32) {
    let idx = gpu_thread_id_x()
    if idx < n { c[idx] = a[idx] + b[idx] }
}
```

```bash
vortex compile mykernel.vx ptx sm_80
```

## Features

- **Neural Networks**: Linear, Conv2D, Transformer, LSTM, GRU layers as builtins. Train with Adam/SGD/AdamW.
- **Tensor Autodiff**: Full backward-pass automatic differentiation on tensors.
- **GPU Compilation**: MLIR-based pipeline targeting NVIDIA PTX and AMD AMDGCN.
- **Cryptography**: secp256k1, ECDSA, Schnorr, SHA-256, NTT, pairings, ZK primitives.
- **MCP Server**: AI agents can train and query models via Model Context Protocol.
- **200+ Builtins**: Math, I/O, strings, arrays, functional programming, ODEs, spiking networks, and more.
- **No Semicolons**: Clean, newline-delimited syntax.
- **Package Manager**: `vortex init`, `vortex add`, `vortex install`.
- **LSP & Debugger**: Language server, interactive debugger, profiler.
- **REPL**: Interactive exploration with `vortex repl`.

## CLI Commands

| Command | Description |
|---------|-------------|
| `vortex run file.vx` | Execute a program |
| `vortex vm file.vx` | Run on bytecode VM |
| `vortex repl` | Interactive REPL |
| `vortex check file.vx` | Type-check |
| `vortex codegen file.vx` | Generate MLIR IR |
| `vortex compile file.vx [target]` | Compile to PTX/AMDGCN/LLVM |
| `vortex mcp` | Start MCP server |
| `vortex lsp` | Start LSP server |
| `vortex serve file.vx` | HTTP server |
| `vortex debug file.vx` | Debugger |
| `vortex profile file.vx` | Profiler |
| `vortex toolchain` | Check GPU toolchain |
| `vortex init [name]` | Create project |
| `vortex add <pkg>` | Add dependency |
| `vortex install` | Install dependencies |

## Documentation

Full documentation is available at [docs/](docs/index.html):

- [Getting Started](docs/guide/getting-started.md)
- [Language Reference](docs/guide/language-reference.md)
- [Neural Networks](docs/guide/neural-networks.md)
- [GPU Computing](docs/guide/gpu-computing.md)
- [MCP Server](docs/guide/mcp-server.md)
- [Builtins Reference](docs/guide/builtins-reference.md)
- [Examples](docs/guide/examples.md)
- [Architecture](docs/guide/architecture.md)

## Examples

The `examples/` directory contains runnable programs:

```bash
vortex run examples/hello_vortex.vx       # Feature tour
vortex run examples/train_xor.vx          # Neural network training
vortex run examples/transformer.vx        # Transformer inference
vortex run examples/crypto_wallet.vx      # Crypto operations
vortex run examples/spiking_network.vx    # Neuromorphic computing
```

## MCP Integration

Vortex includes a built-in MCP server for AI agent integration. Add to your Claude Desktop config:

```json
{
  "mcpServers": {
    "vortex": {
      "command": "/path/to/vortex",
      "args": ["mcp"]
    }
  }
}
```

## Requirements

- **Rust 1.70+** for building
- **Optional**: MLIR toolchain (`mlir-opt`, `mlir-translate`, `llc`) for GPU compilation
- **Optional**: NVIDIA CUDA toolkit or AMD ROCm for GPU execution

## Contributing

Contributions are welcome! Please see [Architecture](docs/guide/architecture.md) for a technical overview.

1. Fork the repository
2. Create a feature branch
3. Run `cargo test` to verify
4. Submit a pull request

## License

MIT License. See [LICENSE](LICENSE) for details.
