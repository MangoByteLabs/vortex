"""
VortexClient - wraps the Vortex MCP server for Python usage.

Starts `vortex mcp` as a subprocess and communicates via JSON-RPC over stdio.
"""

import json
import subprocess
import os


class VortexClient:
    """Python client that communicates with the Vortex MCP server."""

    def __init__(self, binary_path=None):
        """Initialize the client.

        Args:
            binary_path: Path to the vortex binary. Defaults to $VORTEX_BIN or 'vortex'.
        """
        if binary_path is None:
            binary_path = os.environ.get("VORTEX_BIN", "vortex")
        self.binary_path = binary_path
        self._process = None
        self._request_id = 0

    def _ensure_process(self):
        """Start the MCP server subprocess if not already running."""
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                [self.binary_path, "mcp"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def _call_tool(self, tool_name, arguments):
        """Send a tool call to the MCP server and return the result."""
        self._ensure_process()
        self._request_id += 1

        request = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments,
            },
        }

        line = json.dumps(request) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

        response_line = self._process.stdout.readline()
        if not response_line:
            raise RuntimeError("MCP server closed unexpectedly")

        resp = json.loads(response_line)
        if "error" in resp:
            raise RuntimeError(resp["error"].get("message", "Unknown error"))

        result = resp.get("result", {})
        content = result.get("content", [])
        if content:
            return content[0].get("text", "")
        return ""

    def run(self, code):
        """Run Vortex code and return output.

        Args:
            code: Vortex source code string.

        Returns:
            Output from executing the code.
        """
        return self._call_tool("vortex_run", {"code": code})

    def typecheck(self, code):
        """Type-check Vortex code.

        Args:
            code: Vortex source code string.

        Returns:
            Type check result message.
        """
        return self._call_tool("vortex_typecheck", {"code": code})

    def codegen(self, code):
        """Generate MLIR from Vortex code.

        Args:
            code: Vortex source code string.

        Returns:
            Generated MLIR IR string.
        """
        return self._call_tool("vortex_codegen", {"code": code})

    def explain(self, code):
        """Parse and explain Vortex code.

        Args:
            code: Vortex source code string.

        Returns:
            AST explanation string.
        """
        return self._call_tool("vortex_explain", {"code": code})

    def close(self):
        """Shut down the MCP server subprocess."""
        if self._process and self._process.poll() is None:
            try:
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                self._process.kill()
            self._process = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
