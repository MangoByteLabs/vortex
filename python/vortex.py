"""
Vortex Python client -- call Vortex from Python via subprocess IPC.

Usage:
    import vortex

    # Run a Vortex file
    result = vortex.run_file("examples/basics.vx")

    # Evaluate Vortex code
    result = vortex.eval("let x = 42\\nprintln(to_string(x))")

    # Parse Vortex code
    ast = vortex.parse("fn add(a: int, b: int) -> int { return a + b }")

    # Type check
    diagnostics = vortex.check("let x: int = 3.14")

    # Generate MLIR
    mlir = vortex.codegen("kernel vec_add(a: Tensor<f32, [N]>) -> Tensor<f32, [N]> { return a }")
"""

import subprocess
import json
import os


class VortexError(Exception):
    """Raised when a Vortex command fails."""

    def __init__(self, message, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or []


class Vortex:
    """Client for the Vortex language bridge server."""

    def __init__(self, binary_path=None):
        if binary_path is None:
            binary_path = os.environ.get("VORTEX_BIN", "vortex")
        self.binary_path = binary_path
        self._process = None

    def _ensure_process(self):
        if self._process is None or self._process.poll() is not None:
            self._process = subprocess.Popen(
                [self.binary_path, "bridge"],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )

    def _send_command(self, cmd):
        """Send a JSON command and return the parsed response dict."""
        self._ensure_process()
        json_cmd = json.dumps(cmd) + "\n"
        self._process.stdin.write(json_cmd)
        self._process.stdin.flush()
        response_line = self._process.stdout.readline()
        if not response_line:
            raise VortexError("Bridge process closed unexpectedly")
        return json.loads(response_line)

    def _checked(self, resp):
        """Return result on success, raise VortexError on failure."""
        if resp.get("success"):
            return resp.get("result")
        raise VortexError(
            resp.get("error", "Unknown error"),
            diagnostics=resp.get("diagnostics", []),
        )

    def eval(self, source):
        """Evaluate Vortex source code and return the output string."""
        return self._checked(
            self._send_command({"command": "eval", "source": source})
        )

    def run_file(self, path):
        """Run a .vx file and return the output string."""
        return self._checked(
            self._send_command({"command": "run_file", "path": path})
        )

    def parse(self, source):
        """Parse Vortex source and return AST representation."""
        return self._checked(
            self._send_command({"command": "parse", "source": source})
        )

    def check(self, source):
        """Type-check Vortex source. Returns message on success, raises on failure."""
        return self._checked(
            self._send_command({"command": "check", "source": source})
        )

    def codegen(self, source):
        """Generate MLIR from Vortex source."""
        return self._checked(
            self._send_command({"command": "codegen", "source": source})
        )

    def call(self, function_name, *args):
        """Call a Vortex function by name with arguments."""
        return self._checked(
            self._send_command(
                {
                    "command": "call_function",
                    "name": function_name,
                    "args": [str(a) for a in args],
                }
            )
        )

    def close(self):
        """Shut down the bridge process."""
        if self._process and self._process.poll() is None:
            try:
                self._send_command({"command": "quit"})
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


# Module-level convenience functions using a shared default instance.
_default = None


def _get_default():
    global _default
    if _default is None:
        _default = Vortex()
    return _default


def eval(source):
    """Evaluate Vortex source code."""
    return _get_default().eval(source)


def run_file(path):
    """Run a .vx file."""
    return _get_default().run_file(path)


def parse(source):
    """Parse Vortex source and return AST."""
    return _get_default().parse(source)


def check(source):
    """Type-check Vortex source."""
    return _get_default().check(source)


def codegen(source):
    """Generate MLIR from Vortex source."""
    return _get_default().codegen(source)
