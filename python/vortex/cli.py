"""
Python CLI wrapper for Vortex.

Forwards commands to the Rust vortex binary.
"""

import os
import sys
import subprocess


def main():
    """Entry point for the vortex Python CLI wrapper."""
    binary = os.environ.get("VORTEX_BIN", "vortex")

    # Forward all arguments to the Rust binary
    args = [binary] + sys.argv[1:]

    try:
        result = subprocess.run(args)
        sys.exit(result.returncode)
    except FileNotFoundError:
        print(
            f"Error: Vortex binary not found at '{binary}'.\n"
            "Install the Rust binary first: cargo install --path .\n"
            "Or set VORTEX_BIN environment variable.",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
