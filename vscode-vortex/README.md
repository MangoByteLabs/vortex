# Vortex GPU Language - VSCode Extension

Language support for [Vortex](https://github.com/MangoByteLabs/vortex), a GPU programming language for LLM and Crypto workloads.

## Features

- Syntax highlighting for all Vortex keywords, types, operators, and built-in functions
- Code snippets for common patterns (kernels, structs, MoE, SSM, spiking neurons, etc.)
- Run, type-check, and generate MLIR from within VSCode
- Bracket matching and auto-closing pairs
- Basic diagnostics from type checker output

## Commands

| Command | Keybinding | Description |
|---------|-----------|-------------|
| Vortex: Run Current File | `Ctrl+Shift+R` | Run the active `.vx` file |
| Vortex: Type Check Current File | | Type check the active file |
| Vortex: Generate MLIR | | Generate MLIR IR output |

## Requirements

The `vortex` binary must be available on your PATH.

## File Extension

Vortex files use the `.vx` extension.
