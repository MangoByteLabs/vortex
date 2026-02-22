#!/usr/bin/env bash
# MCP Server Demo - sends JSON-RPC requests to the Vortex MCP server via stdin/stdout
# Usage: ./examples/mcp_demo.sh
#
# This demonstrates the full MCP protocol flow that an AI agent would use.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Vortex MCP Server Demo ==="
echo ""

# Build first
echo "Building Vortex..."
cargo build --quiet 2>/dev/null
VORTEX="./target/debug/vortex"

# Helper: send a JSON-RPC request and print the response
send() {
    local desc="$1"
    local json="$2"
    echo "--- $desc ---"
    echo "$json" | $VORTEX mcp 2>/dev/null | head -1 | python3 -m json.tool 2>/dev/null || echo "$json" | $VORTEX mcp 2>/dev/null | head -1
    echo ""
}

# 1. Initialize
send "Initialize" '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}'

# 2. List tools
send "List Tools" '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'

# 3. Run Vortex code
send "Run Vortex Code" '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"vortex_run","arguments":{"code":"fn main() { print(2 + 3) }"}}}'

# 4. Evaluate expression
send "Eval Expression" '{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"vortex_eval","arguments":{"code":"10 * 5 + 2"}}}'

# 5. Type check
send "Type Check" '{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"vortex_typecheck","arguments":{"code":"fn add(a: i64, b: i64) -> i64 { return a + b }"}}}'

# 6. Generate MLIR
send "Codegen (MLIR)" '{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"vortex_codegen","arguments":{"code":"fn add(a: i64, b: i64) -> i64 { return a + b }"}}}'

# 7. Create an MLP model
send "Create MLP Model" '{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"nn_create_model","arguments":{"architecture":"mlp","name":"demo_mlp","layers":[{"type":"linear","in":2,"out":8},{"type":"relu"},{"type":"linear","in":8,"out":1}]}}}'

# 8. Train the model (using multi-line for the request since it is sent per-line)
send "Train Model" '{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"nn_train_model","arguments":{"model_id":"model_1","data":[[0,0],[0,1],[1,0],[1,1]],"labels":[[0],[1],[1],[0]],"epochs":50,"lr":0.01,"optimizer":"adam"}}}'

# 9. Predict
send "Predict" '{"jsonrpc":"2.0","id":9,"method":"tools/call","params":{"name":"nn_predict","arguments":{"model_id":"model_1","input":[1.0,0.0]}}}'

# 10. List models
send "List Models" '{"jsonrpc":"2.0","id":10,"method":"tools/call","params":{"name":"nn_list_models","arguments":{}}}'

# 11. GPU status
send "GPU Status" '{"jsonrpc":"2.0","id":11,"method":"tools/call","params":{"name":"gpu_status","arguments":{}}}'

# 12. List builtins
send "List Builtins" '{"jsonrpc":"2.0","id":12,"method":"tools/call","params":{"name":"vortex_list_builtins","arguments":{}}}'

echo "=== Demo Complete ==="
