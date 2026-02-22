# Architecture

Technical overview of the Vortex compiler and runtime for contributors.

## Compiler Pipeline

```
                         Vortex Source Code (.vx)
                                  |
                          +-------v--------+
                          |    Lexer        |  src/lexer.rs
                          |  (logos-based)  |  60+ token types
                          +-------+--------+
                                  |
                           Token stream
                                  |
                          +-------v--------+
                          |    Parser       |  src/parser.rs
                          | Recursive desc  |  Pratt precedence climbing
                          | + Pratt parser  |  codespan-reporting errors
                          +-------+--------+
                                  |
                                 AST         (src/ast.rs)
                                  |
                    +-------------+-------------+
                    |             |             |
            +-------v---+  +-----v-----+  +---v--------+
            | TypeCheck  |  | Interpret |  |  Codegen   |
            | src/       |  | src/      |  |  src/      |
            | typeck.rs  |  | interp.rs |  |  codegen.rs|
            +-------+---+  +-----+-----+  +---+--------+
                    |             |             |
                 Errors       Output        MLIR IR
                                              |
                                      +-------v--------+
                                      |   mlir-opt     |
                                      | (optimization) |
                                      +-------+--------+
                                              |
                                      +-------v--------+
                                      | mlir-translate  |
                                      | (to LLVM IR)   |
                                      +-------+--------+
                                              |
                                  +-----------+-----------+
                                  |                       |
                          +-------v--------+      +-------v--------+
                          |   llc (PTX)    |      |  llc (AMDGCN)  |
                          |   NVIDIA GPU   |      |   AMD GPU      |
                          +----------------+      +----------------+
```

## Module Map

### Core Language

| Module | File | Description |
|--------|------|-------------|
| **Lexer** | `src/lexer.rs` | logos-based tokenizer with 60+ token types |
| **AST** | `src/ast.rs` | Full AST: structs, enums, match, traits, impls, kernels, closures |
| **Parser** | `src/parser.rs` | Recursive descent with Pratt precedence climbing |
| **Type Checker** | `src/typeck.rs` | Two-pass: collect declarations, then check bodies |
| **Interpreter** | `src/interpreter.rs` | Tree-walking interpreter with 200+ builtins |
| **VM** | `src/vm.rs` | Bytecode virtual machine (alternative to tree-walking) |
| **Codegen** | `src/codegen.rs` | MLIR IR generation (func/arith/scf/gpu dialects) |
| **Pipeline** | `src/pipeline.rs` | Full compilation pipeline: MLIR -> opt -> translate -> llc |
| **Main** | `src/main.rs` | CLI entry point: lex, parse, check, run, codegen, compile, etc. |

### Neural Networks & ML

| Module | File | Description |
|--------|------|-------------|
| **NN** | `src/nn.rs` | Neural network layers, models, training, and inference |
| **Tensor Autodiff** | `src/tensor_autodiff.rs` | Tensor-level automatic differentiation tape |
| **Scalar Autodiff** | `src/autodiff.rs` | Scalar automatic differentiation |
| **Continuous Learning** | `src/continuous_learning.rs` | Online learning while serving |
| **Architectures** | `src/architectures.rs` | Spike-SSM hybrids and advanced models |
| **Self-Modify** | `src/self_modify.rs` | Dynamic architecture search |
| **Multiscale** | `src/multiscale.rs` | Multiscale reasoning models |
| **Tiered Experts** | `src/tiered_experts.rs` | Mixture-of-experts routing |
| **Heterogeneous** | `src/heterogeneous.rs` | Mixed-type compute layers |
| **Adaptive Inference** | `src/adaptive_inference.rs` | Early-exit / adaptive depth |
| **Energy Models** | `src/energy_models.rs` | Energy-based models |

### Cryptography

| Module | File | Description |
|--------|------|-------------|
| **Crypto** | `src/crypto.rs` | BigUint256, secp256k1 (Jacobian coords), ECDSA, Schnorr |
| **ModMath** | `src/modmath.rs` | Montgomery multiplication, modular arithmetic |
| **Fields** | `src/fields.rs` | Field arithmetic |
| **NTT** | `src/ntt.rs` | Number Theoretic Transform |
| **MSM** | `src/msm.rs` | Multi-scalar multiplication |
| **Pairing** | `src/pairing.rs` | Bilinear pairings (BLS12-381) |
| **Poly** | `src/poly.rs` | Polynomial arithmetic |
| **FFT** | `src/fft.rs` | Fast Fourier Transform |

### Neuromorphic & Differential Equations

| Module | File | Description |
|--------|------|-------------|
| **Spiking** | `src/spiking.rs` | Spike trains, LIF neurons |
| **SSM** | `src/ssm.rs` | State space models (Mamba-style) |
| **ODE** | `src/ode.rs` | ODE solvers (Euler, RK4, RK45) |
| **Memory** | `src/memory.rs` | Differentiable memory (NTM-style) |

### Infrastructure

| Module | File | Description |
|--------|------|-------------|
| **MCP Server** | `src/mcp_server.rs` | Model Context Protocol for AI agents |
| **LSP Server** | `src/lsp_server.rs` | Language Server Protocol for editors |
| **LSP** | `src/lsp.rs` | Diagnostics, symbols, hover info |
| **Server** | `src/server.rs` | HTTP server for Vortex programs |
| **Module** | `src/module.rs` | Module/import resolution |
| **Package** | `src/package.rs` | Package manifest (vortex.toml) |
| **Registry** | `src/registry.rs` | Package registry and dependency resolution |
| **Python Bridge** | `src/python_bridge.rs` | Python interop |
| **GPU Pipeline** | `src/gpu_pipeline.rs` | GPU compilation pipeline |
| **GPU Runtime** | `src/gpu_runtime.rs` | GPU runtime (CPU fallback) |
| **GPU Compute** | `src/gpu_compute.rs` | Native GPU compute operations |
| **Debugger** | `src/debugger.rs` | Interactive debugger |
| **Profiler** | `src/profiler.rs` | Execution profiler |
| **Quantize** | `src/quantize.rs` | Model quantization |
| **Sparse** | `src/sparse.rs` | Sparse tensor operations |
| **DynTensor** | `src/dyntensor.rs` | Dynamic-shape tensors |
| **Flash Attention** | `src/flash_attention.rs` | Hardware-efficient attention |
| **Shape Check** | `src/shape_check.rs` | Tensor shape validation |
| **Backends** | `src/backends.rs` | Multi-backend compilation |
| **Formal Verify** | `src/formal_verify.rs` | Formal verification |
| **Prob Types** | `src/prob_types.rs` | Probabilistic type system |
| **Swarm** | `src/swarm.rs` | Swarm intelligence |
| **Synthesis** | `src/synthesis.rs` | Program synthesis |
| **Symbolic Reasoning** | `src/symbolic_reasoning.rs` | Symbolic math evaluation |
| **Diff Structures** | `src/diff_structures.rs` | Differentiable data structures |
| **Metabolic** | `src/metabolic.rs` | Metabolic computing |
| **Reversible** | `src/reversible.rs` | Reversible computation |
| **Provenance** | `src/provenance.rs` | Data provenance tracking |
| **Causal** | `src/causal.rs` | Causal inference |
| **Matrix of Thought** | `src/matrix_of_thought.rs` | Matrix-of-thought reasoning |
| **Verifiable Inference** | `src/verifiable_inference.rs` | ZK proofs for inference |

## Key Design Decisions

### No Semicolons
Vortex is newline-delimited. The parser uses newlines as statement terminators. This was a deliberate choice to reduce visual noise for ML code that tends to have many short statements.

### `self` is Not a Keyword
In method definitions, `self` is treated as a regular identifier parameter. The interpreter looks up methods as `TypeName::method_name` in the function table.

### `kernel` Produces GPU IR
The `kernel` keyword generates `gpu.module` + `gpu.func` in MLIR output, which then lower through the standard MLIR GPU pipeline to PTX or AMDGCN.

### Builtins Over Library
Most functionality (neural networks, crypto, autodiff) is implemented as built-in functions in the interpreter rather than as a standard library. This gives maximum performance and tight integration with the runtime.

### MLIR as IR
Vortex chose MLIR over LLVM IR directly because MLIR's dialect system allows domain-specific optimization. The `gpu`, `scf`, `arith`, and `func` dialects map naturally to Vortex's semantics.

## Dependencies

- **logos 0.14** -- Lexer generation
- **codespan-reporting 0.11** -- Diagnostic error messages
- **serde / serde_json** -- JSON serialization (MCP server, model save/load)
- **sha2** -- SHA-256 implementation
- **insta 1.39** -- Snapshot testing

## Testing

```bash
cargo test
```

The test suite includes snapshot tests (via insta) for parser output, interpreter results, and codegen output.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run `cargo test` to ensure all tests pass
5. Submit a pull request

Key areas where contributions are welcome:
- **Type system**: Richer types, generics, monomorphization
- **MLIR compilation**: Hooking into mlir-opt/mlir-translate for actual GPU execution
- **Optimizer**: More optimizations in the codegen pipeline
- **Standard library**: Expanding the stdlib with more utilities
- **Documentation**: More examples and tutorials
