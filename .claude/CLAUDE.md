# Vortex Project Rules

## RULE #1: MACHINE-LEVEL LANGUAGE — TALKS DIRECTLY TO HARDWARE

Vortex is a MACHINE-LEVEL, FULLY INDEPENDENT, SELF-HOSTING language. It talks DIRECTLY to hardware — no intermediaries, no layers, no dependencies.

- **ZERO external dependencies.** No Rust, no C, no libc, no stdlib, no frameworks.
- **Direct hardware communication.** Raw syscalls to the kernel. Raw ioctls to GPU. Raw memory management. No abstraction layers.
- **Vortex IS the stack.** From source code to bare metal binary — every byte is Vortex.

## RULE #2: ABSOLUTE PROHIBITIONS

- **NEVER create, edit, or modify any .rs file.** The Rust bootstrap is FROZEN legacy.
- **NEVER use `cargo run` as the solution.** Native compilation via vxc.vx only.
- **NEVER add Rust dependencies or crates.**
- **NEVER use external stdlib/frameworks.** No libc, no glibc, no Python, no CUDA, no ROCm, no OpenCL.
- **NEVER wrap or call external libraries.** Vortex implements EVERYTHING from scratch.
- **ALL new features must be implemented in .vx files.**

## RULE #3: HOW VORTEX TALKS TO HARDWARE

### CPU — Direct syscalls, no libc
- `syscall(num, args...)` — raw Linux syscall interface
- File I/O: `SYS_OPEN`, `SYS_READ`, `SYS_WRITE`, `SYS_CLOSE`
- Memory: `SYS_MMAP`, `SYS_MUNMAP`, `SYS_BRK`
- Process: `SYS_EXIT`, `SYS_FORK`, `SYS_EXECVE`
- Network: `SYS_SOCKET`, `SYS_BIND`, `SYS_LISTEN`, `SYS_ACCEPT`
- No libc. No glibc. Raw `syscall` instruction.

### GPU — Direct DRM ioctls, no drivers
- Vortex has its OWN GPU ISA: **VXB (Vortex Binary)**
- Direct `/dev/dri/renderD*` access via `SYS_IOCTL`
- GEM buffer management for GPU memory
- SIMT execution: 32-lane warps, barrier sync, warp shuffle
- **NEVER use CUDA, ROCm, OpenCL, Vulkan, or ANY external GPU stack.**

### Binary output — Direct ELF emission
- `elf.vx` emits raw ELF64 headers, program headers, machine code
- `x86_codegen.vx` generates raw x86-64 instruction bytes
- `linker.vx` resolves symbols and patches relocations
- The output is a bare Linux executable. No dynamic linking. No shared libraries.

## RULE #4: LANGUAGE PURPOSE — ENABLING THE IMPOSSIBLE

Vortex exists because current languages CANNOT enable what's coming:
- **Tensor as primitive type** — not a library, a LANGUAGE TYPE
- **BigInt + modular arithmetic** — native arbitrary precision, cryptographic computation
- **MatrixOfThought** — 5D cognitive reasoning spaces as data structures
- **Energy-aware computation** — the language tracks computational cost
- **Native model-to-model communication** — binary tensor transport, not serialized JSON
- **Living digital organisms** — self-modifying architectures, continual learning
- **AGI/ASI infrastructure** — the language that intelligence runs on

## RULE #5: SELF-HOSTING COMPILER PIPELINE

```
source.vx → lexer.vx → parser.vx → x86_codegen.vx → linker.vx → elf.vx → NATIVE BINARY
```

- Bootstrap compiler: `stdlib/compiler/vxc.vx`
- GPU compiler: `stdlib/gpu/vxb_codegen.vx` → VXB binary → direct GPU execution
- VERIFIED WORKING: compiles `println(40 + 2)` → native binary that prints `42`

## Vortex Syntax Rules

- No semicolons
- `push(arr, val)` returns a NEW array (does not mutate)
- `var` for mutable, `let` for immutable
- No top-level `var` — use zero-arg functions for constants
- `diff` is a reserved keyword — use `delta` instead
- String indexing: use `str_char_at(s, i)`, NOT `s[i]`
- Escape sequences work: `"\n"`, `"\t"`, `"\\"`, `"\""`
- `to_string(x)` for int→string, `int(s)` for string→int
- `float(x)` for int→float
- Arrays are dynamically typed (`[String]` annotation is not enforced at runtime)
- No nested array type annotations (`[[String]]` is invalid — use `[String]`)

## Key File Locations

- `stdlib/compiler/` — self-hosting compiler (vxc.vx, lexer.vx, parser.vx, x86_codegen.vx, linker.vx, elf.vx)
- `stdlib/ai/` — tensor ops (tensor_ops.vx), transformer inference (transformer.vx)
- `stdlib/gpu/` — VXB ISA codegen (vxb_codegen.vx), SIMT execution
- `stdlib/math/` — BigInt + modular arithmetic (bigint.vx)
- `stdlib/std/` — io, string, thread, crypto, nn, optim
