# MCP Server for AI Agents

Vortex includes a built-in [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) server that allows AI agents to create, train, query, and manage neural networks through JSON-RPC 2.0.

## What is MCP?

MCP is a standard protocol that lets AI assistants (like Claude) use external tools. The Vortex MCP server exposes the full power of Vortex -- running code, training models, GPU compilation -- as tools that any MCP-compatible agent can call.

## Starting the Server

```bash
vortex mcp
```

The server communicates via JSON-RPC 2.0 on stdin/stdout, following the MCP specification.

## Available Tools

### Code Execution

| Tool | Description |
|------|-------------|
| `vortex_run` | Execute a Vortex program and return stdout output |
| `vortex_eval` | Evaluate an expression and return its result |
| `vortex_typecheck` | Type-check code without running it |
| `vortex_codegen` | Generate MLIR IR from Vortex source |
| `vortex_benchmark` | Run a program multiple times and report timing |
| `vortex_explain` | Get AST summary, types, and signatures |
| `vortex_list_builtins` | List all available builtin functions |

### Neural Network Tools

| Tool | Description |
|------|-------------|
| `nn_create_model` | Create a model (MLP, Transformer, CNN, LSTM) |
| `nn_train_model` | Train a model on provided data |
| `nn_predict` | Run inference on a model |
| `nn_save_model` | Save model weights to JSON |
| `nn_load_model` | Load model weights from JSON |
| `nn_list_models` | List all models in the session |

### System

| Tool | Description |
|------|-------------|
| `gpu_status` | Check GPU and toolchain availability |

## Tool Details

### vortex_run

```json
{
  "name": "vortex_run",
  "params": {
    "code": "fn main() { println(42 + 58) }"
  }
}
```

Or run from a file:

```json
{
  "name": "vortex_run",
  "params": {
    "file": "/path/to/program.vx"
  }
}
```

### nn_create_model

Create an MLP:

```json
{
  "name": "nn_create_model",
  "params": {
    "architecture": "mlp",
    "layers": [
      {"type": "linear", "in": 2, "out": 16},
      {"type": "relu"},
      {"type": "linear", "in": 16, "out": 1},
      {"type": "sigmoid"}
    ]
  }
}
```

Create a Transformer:

```json
{
  "name": "nn_create_model",
  "params": {
    "architecture": "transformer",
    "dim": 64,
    "num_heads": 4,
    "ff_dim": 128,
    "num_blocks": 2
  }
}
```

Create a CNN:

```json
{
  "name": "nn_create_model",
  "params": {
    "architecture": "cnn",
    "layers": [
      {"type": "conv2d", "in_channels": 1, "out_channels": 16, "kernel_size": 3},
      {"type": "relu"},
      {"type": "linear", "in": 256, "out": 10},
      {"type": "softmax"}
    ]
  }
}
```

### nn_train_model

```json
{
  "name": "nn_train_model",
  "params": {
    "model_id": "model_1",
    "data": [[0,0],[0,1],[1,0],[1,1]],
    "labels": [[0],[1],[1],[0]],
    "epochs": 500,
    "lr": 0.01,
    "optimizer": "adam",
    "loss": "mse"
  }
}
```

### nn_predict

```json
{
  "name": "nn_predict",
  "params": {
    "model_id": "model_1",
    "input": [1.0, 0.0]
  }
}
```

## Configuration for Claude Desktop

Add this to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vortex": {
      "command": "cargo",
      "args": ["run", "--manifest-path", "/path/to/vortex/Cargo.toml", "--", "mcp"],
      "description": "Vortex GPU programming language - neural net training, inference, and GPU compute"
    }
  }
}
```

Or if you have the binary built:

```json
{
  "mcpServers": {
    "vortex": {
      "command": "/path/to/vortex",
      "args": ["mcp"],
      "description": "Vortex GPU language"
    }
  }
}
```

A pre-made config file is included at `mcp_config.json` in the repo root.

## Example: AI Agent Training a Model

Here is what a conversation with an AI agent using the Vortex MCP server might look like:

**Agent:** "Create a neural network that classifies iris flowers"

The agent calls:

1. `nn_create_model` with architecture `"mlp"` and layers `[{"type":"linear","in":4,"out":16}, {"type":"relu"}, {"type":"linear","in":16,"out":3}, {"type":"softmax"}]`

2. `nn_train_model` with the iris dataset, 1000 epochs, Adam optimizer

3. `nn_predict` for each test sample

4. `nn_save_model` to persist the trained weights

All of this happens without the agent needing to write Python, install PyTorch, or manage any dependencies. The Vortex MCP server handles everything.

## Session State

The MCP server maintains session state:

- **Model registry**: All created models persist across tool calls within a session
- **Command history**: Every tool call is logged
- **Automatic IDs**: Models get sequential IDs (`model_1`, `model_2`, ...) that you use in subsequent calls

## Protocol Details

The server implements the MCP specification:

- **Transport**: stdin/stdout with JSON-RPC 2.0
- **Methods**: `initialize`, `tools/list`, `tools/call`
- **Capabilities**: `tools` (no `prompts` or `resources` currently)
