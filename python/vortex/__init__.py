"""
Vortex GPU Programming Language - Python bindings.

Usage:
    from vortex import VortexClient

    client = VortexClient()
    result = client.run('let x = 42\\nprintln(to_string(x))')

Or use the convenience function:

    import vortex
    output = vortex.run('let x = 42\\nprintln(to_string(x))')
"""

from .client import VortexClient

__version__ = "0.1.0"

_default_client = None


def _get_client():
    global _default_client
    if _default_client is None:
        _default_client = VortexClient()
    return _default_client


def run(code: str) -> str:
    """Run Vortex code and return output."""
    return _get_client().run(code)


def typecheck(code: str) -> str:
    """Type-check Vortex code."""
    return _get_client().typecheck(code)


def codegen(code: str) -> str:
    """Generate MLIR from Vortex code."""
    return _get_client().codegen(code)


def explain(code: str) -> str:
    """Parse and explain Vortex code."""
    return _get_client().explain(code)
