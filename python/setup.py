from setuptools import setup

setup(
    name="vortex-lang",
    version="0.1.0",
    py_modules=["vortex"],
    python_requires=">=3.8",
    description="Python bindings for the Vortex GPU programming language",
    long_description="Subprocess-based Python client for the Vortex GPU language. "
    "Communicates with the Vortex compiler via JSON over stdin/stdout.",
    author="MangoByteLabs",
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Topic :: Software Development :: Compilers",
    ],
)
