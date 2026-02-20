# GPU ISA & Intermediate Representation Analysis

## Can We Create a New Instruction Set for GPUs?

A comprehensive technical analysis of GPU execution models, instruction set architectures,
the feasibility of bypassing PTX/SASS, creating alternative intermediate representations,
and targeting open-source GPU hardware.

---

## Table of Contents

1. [NVIDIA's Execution Model: From PTX to Hardware](#1-nvidias-execution-model-from-ptx-to-hardware)
2. [Bypassing PTX: Direct SASS Generation](#2-bypassing-ptx-direct-sass-generation)
3. [Could We Create a Better IR Than PTX?](#3-could-we-create-a-better-ir-than-ptx)
4. [AMD's Open ISA Approach](#4-amds-open-isa-approach)
5. [Open-Source GPU Hardware](#5-open-source-gpu-hardware)
6. [Cross-Vendor IR: The MLIR Path](#6-cross-vendor-ir-the-mlir-path)
7. [Practical Assessment](#7-practical-assessment)
8. [Recommended Architecture for Vortex](#8-recommended-architecture-for-vortex)

---

## 1. NVIDIA's Execution Model: From PTX to Hardware

### 1.1 The Full Compilation Chain

The path from source code to GPU execution on NVIDIA hardware involves multiple layers,
each controlled by different software components:

```
Source Code (CUDA C++, Triton Python, OpenCL C)
    |
    v
[Frontend Compiler: nvcc/clang/Triton AST walker]
    |
    v
PTX (Parallel Thread Execution) -- Virtual ISA, architecture-independent
    |
    v
[ptxas -- NVIDIA's proprietary optimizing assembler]
    |
    v
SASS (Streaming ASSembler) -- Native machine code, architecture-specific
    |  packaged as...
    v
cubin (CUDA Binary) -- ELF container (EM_CUDA = 190)
    |  loaded via...
    v
[CUDA Driver API: cuModuleLoad / cuModuleLoadData]
    |  which calls into...
    v
[Resource Manager (RM) / nvidia.ko kernel module]
    |  which programs...
    v
[GPU System Processor (GSP) -- RISC-V firmware controller]
    |  which configures...
    v
[PBDMA (Push Buffer DMA) + Hardware FIFO Scheduler]
    |  which dispatches to...
    v
[Streaming Multiprocessors (SMs) -- actual compute units]
```

### 1.2 PTX: The Virtual ISA

PTX is a virtual machine ISA -- it describes an idealized GPU that represents the
common features of all NVIDIA hardware. Key characteristics:

- **Architecture-independent**: PTX compiled for compute_70 can run on SM_75, SM_80,
  SM_86, SM_89, SM_90, and SM_100 (via JIT recompilation by the driver)
- **SSA form**: The compiler output is nearly pure static single-assignment, with
  consecutive lines generally referring to consecutive virtual registers
- **Unlimited registers**: PTX uses an arbitrarily large register set; `ptxas` performs
  the actual register allocation to physical registers (typically 255 per thread max)
- **Not 1:1 with hardware**: A single PTX instruction may compile to multiple SASS
  instructions, or multiple PTX instructions may be fused into one SASS instruction
- **Instruction scheduling is deferred**: PTX does not specify instruction ordering
  constraints; `ptxas` handles all scheduling, barrier insertion, and stall management

PTX serves as NVIDIA's **forward compatibility guarantee**: an application compiled to
PTX for Turing (CC 7.5) in 2018 can run on Blackwell (CC 12.0) in 2025 via JIT
recompilation. This is the fundamental reason PTX exists as an abstraction layer.

**Current version**: PTX ISA 9.1 (as of CUDA 13.x), with ISA 8.7 introducing SM100
Blackwell support.

### 1.3 SASS: The Native Machine Code

SASS is the actual binary instruction encoding executed by GPU hardware. It is:

- **Architecture-specific**: SASS for SM_75 (Turing) will not run on SM_80 (Ampere)
- **Poorly documented**: NVIDIA provides no official SASS encoding specification
- **128-bit instructions** (Turing and later): Each instruction is 16 bytes, split into
  two 64-bit words when displayed by `nvdisasm`
- **Control-code-interleaved** (Maxwell/Pascal): 1 control instruction per 3 compute
  instructions; each control word is 64 bits containing scheduling metadata for 3
  subsequent instructions
- **Self-scheduling** (Volta+): Control information embedded within each 128-bit
  instruction word rather than in separate control instructions

#### SASS Control Code Bit Layout (Maxwell/Pascal)

From reverse engineering by the maxas project (Scott Gray / NervanaSystems):

```
64-bit control word for 3 instructions:

Instruction 1: bits [0:16]   = 17-bit control section
Instruction 2: bits [21:37]  = 17-bit control section
Instruction 3: bits [42:58]  = 17-bit control section

Reuse flags:
Instruction 1: bits [17:20]  = 4-bit reuse
Instruction 2: bits [38:41]  = 4-bit reuse
Instruction 3: bits [59:62]  = 4-bit reuse

Each 17-bit control section:
  [0:3]   Stall count (0-15 clock cycles)
  [4]     Yield hint flag (warp scheduler hint)
  [5:7]   Write dependency barrier (0-5 -> barriers 1-6)
  [8:10]  Read dependency barrier (0-5 -> barriers 1-6)
  [11:15] Wait dependency barrier mask (6 bits, one per barrier)
```

#### SASS Instruction Format (Turing/Ampere/Hopper, 128-bit)

```
128-bit instruction word:
  [0:63]    Instruction encoding (opcode + operands + modifiers)
  [64:127]  Control/scheduling metadata embedded in upper 64 bits
            including stall counts, barriers, yield, reuse flags,
            predicate guards, and scoreboard annotations

Six scoreboards (numbered 0-5) coordinate instruction ordering:
  - Write scoreboards: track variable-latency outputs (memory, DP, SFU)
  - Read scoreboards: protect source operands for non-collector instructions
  - Wait scoreboards: barrier mask indicating which scoreboards must clear

Control code fields (Turing+ embedded format):
  Reuse flags:     4 bits (register bank forwarding)
  Dependency bars: 6 bits (scoreboard 0-5 wait/set)
  Read scoreboard: "R#" sets scoreboard for source operand hold
  Write scoreboard: "W#" sets scoreboard for destination protection
  Yield:           "Y" or "-"
  Stall count:     0-15 cycles
```

### 1.4 The GPU Command Processor

Between the driver and the SMs lies the GPU's command processing infrastructure:

#### FIFO / Host Interface

The GPU HOST unit manages work scheduling. Key concepts from NVIDIA's open-gpu-doc:

- **Runlist**: An ordered list of channels the GPU reads to find work. Every enabled
  channel must belong to a runlist.
- **TSG (Time Slice Group)**: Contains multiple channels from a process. The GPU
  round-robins across TSGs with configurable timeslices.
- **Channel**: A stream of GPU commands from a process, backed by a pushbuffer
  (ring buffer in GPU-accessible memory).
- **PBDMA (Push Buffer DMA)**: Hardware unit that pulls commands from CPU-side
  pushbuffers into the GPU. Multiple PBDMA units can operate in parallel.

#### Command Submission Flow

```
1. Application calls cuLaunchKernel()
2. CUDA driver writes launch parameters to command buffer (pushbuffer)
   - Kernel address, grid dimensions, block dimensions
   - Shared memory size, parameter buffer pointer
   - These are encoded as GPU "methods" -- register writes to the
     compute engine class (e.g., NV_COMPUTE_CLASS)
3. Driver updates the channel's GP_PUT pointer
4. GPU HOST detects pending work via runlist scanning
5. PBDMA pulls method entries from the pushbuffer via DMA
6. Methods are dispatched to the compute engine
7. Compute engine configures SMs and launches thread blocks (CTAs)
```

#### Pushbuffer Method Format

From NVIDIA's open-gpu-doc (dev_pbdma.ref.txt):

```
Method data entry format:
  NV_FIFO_DMA_METHOD_ADDRESS: dword-address (LSB 2 bits not stored)
  NV_FIFO_DMA_METHOD_SUBCHANNEL: subchannel selector
  NV_FIFO_DMA_METHOD_COUNT: number of methods
  NV_FIFO_DMA_METHOD_DATA: operand for the method

GP Entry format:
  GP_ENTRY0_GET + GP_ENTRY1_GET_HI: 38-bit dword-address of pushbuffer segment
  (40-bit byte-address, 4-byte aligned, LSB 2 bits not stored)
```

#### Falcon and GSP Firmware

Modern NVIDIA GPUs (Turing+) contain multiple Falcon (FAst Logic CONtroller)
microcontrollers -- small RISC processors running signed firmware. Starting with
Turing, NVIDIA introduced the **GPU System Processor (GSP)**, which runs on a
dedicated RISC-V core:

- GSP abstracts hardware control: instead of the host driver writing individual
  control registers, it sends high-level commands to the GSP
- GSP firmware is signed and verified -- you cannot replace it
- The "Physical RM" (Resource Manager) runs on the GSP; "Kernel RM" runs in the
  host kernel module (nvidia.ko)
- This architecture means the driver increasingly talks to firmware, not directly
  to hardware registers

**Key implication**: Even if you generate valid SASS and package it in a cubin,
you still must go through NVIDIA's driver stack to load and execute it. There is
no way to bypass the driver, GSP firmware, or FIFO scheduler.

### 1.5 Entry Points: Is PTX the Only Way In?

**No.** There are multiple entry points to get code onto NVIDIA GPUs:

1. **PTX (text)** -> `cuModuleLoadData()` with PTX string -> driver JIT compiles
   to SASS via embedded `ptxas`
2. **cubin (binary)** -> `cuModuleLoad()` or `cuModuleLoadData()` -> driver loads
   pre-compiled SASS directly (no ptxas needed at runtime)
3. **fatbin (fat binary)** -> contains multiple cubins for different architectures
   plus optionally PTX for JIT fallback
4. **SPIR-V** -> via Vulkan compute shaders -> NVIDIA's Vulkan driver compiles
   SPIR-V to SASS internally
5. **OpenGL GLSL / Vulkan GLSL** -> via graphics pipeline -> compiled by driver
6. **OptiX IR** -> for ray tracing workloads -> compiled by OptiX driver

**Critical insight**: You can load a cubin containing pre-assembled SASS directly.
The driver does not require PTX. This is the entry point that CuAssembler,
turingas, and other SASS assemblers exploit.

---

## 2. Bypassing PTX: Direct SASS Generation

### 2.1 The Reverse Engineering Ecosystem

Since NVIDIA does not document SASS encoding, the community has built an impressive
ecosystem of reverse-engineering tools:

#### maxas (Maxwell Assembler)
- **Author**: Scott Gray (Nervana Systems / Intel)
- **Repository**: https://github.com/NervanaSystems/maxas
- **Architectures**: Maxwell (SM_50/52), Pascal (SM_60/61)
- **Significance**: The pioneering work. Produced hand-tuned SGEMM kernels that
  outperformed cuBLAS by 15%+ through careful control code tuning and register
  bank conflict avoidance. Demonstrated that 132 GFLOP/s -> 152 GFLOP/s (15.4%
  improvement) was achievable via SASS-level optimization.
- **Key contribution**: Documented the control code format, dependency barrier
  system, and reuse flags for the first time

#### turingas (Turing Assembler)
- **Repository**: https://github.com/daadaada/turingas
- **Architectures**: Volta (SM_70), Turing (SM_75)
- **Features**: Full assembler from SASS text to cubin binary. Can assemble
  kernels and link them into loadable cubins.
- **Status**: Functional but limited to Volta/Turing. Not updated for Ampere+.

#### CuAssembler
- **Repository**: https://github.com/cloudcores/CuAssembler
- **Architectures**: Pascal, Volta, Turing, Ampere (SM_60/61/70/75/80/86)
- **Approach**: Decomposes instructions into elemental values and position-dependent
  weights, then solves the encoding via linear algebra using known instruction
  encodings from compiled CUDA libraries as training data
- **Features**: Can modify individual instructions within an existing cubin,
  re-encode control codes, and handle most instruction types
- **Limitations**: Cannot fully recover certain instructions where modifiers are
  not shown in disassembly text (e.g., B2R on Turing, certain LDG/STG variants
  on Ampere)

#### nv_isa_solver
- **Repository**: https://github.com/kuterd/nv_isa_solver
- **Architectures**: SM_89 (Ada/RTX 4090), SM_90a (Hopper/H100)
- **Approach**: Automatically generates ISA specifications by **fuzzing nvdisasm** --
  systematically generates binary sequences, feeds them to NVIDIA's disassembler,
  and infers encoding rules from the output
- **Output**: Machine-readable JSON specification of instruction encodings
- **Online spec**: https://kuterdinel.com/nv_isa/ (human-readable SM90a ISA)
- **Significance**: Most current effort; covers Hopper architecture

#### DocumentSASS
- **Repository**: https://github.com/0xD0GF00D/DocumentSASS
- **Purpose**: Community-maintained unofficial documentation of SASS instruction sets
- **Content**: Instruction descriptions, operand formats, encoding notes

#### Other Projects
- **AsFermi**: Fermi-era assembler (SM_20)
- **KeplerAs**: Kepler assembler (SM_30/35)
- **Decuda**: Original CUDA disassembler (pre-Fermi)
- **cuasm** (gpuocelot): Pascal/Volta/Turing/Ampere assembler fork
- **SASSI** (NVlabs): NVIDIA's own SASS instrumentation framework (research)
- **AmpItUp**: https://github.com/NoxNode/AmpItUp -- Ampere encoding investigation

### 2.2 How Direct SASS Generation Works

The practical workflow for generating SASS directly:

```
Step 1: Write or generate SASS assembly text
        (using nvdisasm output format as reference)

Step 2: Use CuAssembler/turingas to encode assembly -> binary
        - Look up opcode encoding from reverse-engineered tables
        - Encode operands (registers, immediates, addresses)
        - Compute control codes (stall counts, barriers, yields)
        - Pack into 128-bit instruction words

Step 3: Package binary into cubin (ELF format)
        - Create proper ELF sections (.text.kernel_name, .nv.info, etc.)
        - Set ELF machine type to EM_CUDA (190)
        - Include kernel metadata (register count, shared memory, etc.)

Step 4: Load cubin via CUDA Driver API
        cuModuleLoadData(&module, cubin_data);
        cuModuleGetFunction(&function, module, "kernel_name");
        cuLaunchKernel(function, ...);
```

### 2.3 Practical Challenges of Direct SASS

**Architecture fragility**: SASS encoding changes with every GPU generation.
Maxwell, Pascal, Volta, Turing, Ampere, Ada, Hopper, and Blackwell each have
different instruction encodings, opcodes, and control code formats. Supporting
N architectures requires N separate encoding backends.

**Instruction scheduling**: The hardest part of SASS generation is not encoding
instructions but scheduling them correctly. You must:

- Compute correct stall counts for every instruction based on pipeline latency
- Set dependency barriers for variable-latency operations (memory loads, SFU)
- Manage the 6 scoreboard resources without conflicts
- Handle register bank conflicts (8 banks, 4 operands -> potential conflicts)
- Set yield hints for optimal warp scheduling
- Manage reuse flags for register bank forwarding

Getting scheduling wrong does not produce errors -- it produces **silently
incorrect results** or massive performance degradation.

**Undocumented features**: Some SASS instructions have modifiers or behaviors not
visible in disassembly output. CuAssembler documents cases where "some modifiers
are not shown in the assembly text, and some instructions even don't show up in
SASS assembly." This means the encoding tables may be incomplete.

**Tensor Core instructions**: The most performance-critical instructions (HMMA,
GMMA, TCGEN05.MMA on Blackwell) have complex encoding with many configuration
bits. Reverse engineering these correctly for every architecture is difficult.

**Legal considerations**: NVIDIA's CUDA EULA and terms of service may restrict
reverse engineering of their ISA. The tools listed above operate in a gray area.
The generated code must still be loaded through NVIDIA's proprietary driver.

### 2.4 The CuAsmRL Approach

A 2025 paper (arxiv:2501.08071) introduced **CuAsmRL**, which uses deep
reinforcement learning to optimize SASS instruction schedules:

- Takes ptxas-generated SASS as starting point
- Uses RL to explore alternative instruction orderings and control code settings
- Achieves measurable speedups by finding better schedules than ptxas
- Demonstrates that ptxas's scheduling is good but not optimal

This validates the premise that SASS-level control matters for performance, and
that ptxas leaves optimization opportunities on the table.

---

## 3. Could We Create a Better IR Than PTX?

### 3.1 PTX's Limitations

PTX has significant limitations that motivate the search for alternatives:

**1. Abstraction overhead**: PTX's virtual register model and architecture
independence mean the programmer cannot control:
- Physical register allocation (ptxas decides)
- Instruction scheduling and ordering
- Memory bank assignment
- Warp-level operation interleaving
- Shared memory bank conflict avoidance at the instruction level

**2. Missing hardware features**: PTX often lags behind hardware capabilities.
New instructions (like Hopper's WGMMA, TMA, or Blackwell's TCGEN05) are
initially accessible only through PTX intrinsics or inline assembly, and the
mapping from PTX to SASS is not always optimal for new features.

**3. Compiler opacity**: ptxas is a black box. You cannot:
- Guide its register allocator
- Influence its instruction scheduler
- Control its optimization passes
- Understand why it made specific decisions

**4. Single-vendor lock-in**: PTX is NVIDIA-only. Code written in PTX cannot
target AMD or Intel GPUs without complete rewriting.

**5. Scheduling information loss**: PTX intentionally discards scheduling
information, requiring ptxas to reconstruct it. For expert programmers who
know the optimal schedule, this is wasted effort and potentially worse results.

**6. Suboptimal codegen for non-standard patterns**: DeepSeek's engineers
discovered that hand-written PTX (bypassing CUDA C++ entirely) was necessary
to achieve their communication-compute overlap optimizations, allocating 20 of
132 SMs for inter-node communication while the rest computed. The CUDA compiler
could not generate these patterns.

### 3.2 What Would a Better GPU IR Look Like?

A next-generation GPU IR for crypto/LLM workloads would need:

#### For Cryptographic Workloads

- **Wide integer arithmetic primitives**: Native 256-bit and 512-bit integer
  operations (add-with-carry chains, multiply-accumulate)
- **Modular arithmetic**: Montgomery multiplication, Barrett reduction as
  first-class operations
- **Number Theoretic Transform (NTT)**: Butterfly operations with modular
  arithmetic, optimized for lattice cryptography (Kyber, Dilithium)
- **Bitwise permutation operations**: Efficient bit shuffling for hash functions
  (SHA-256, Keccak)
- **Constant-time execution guarantees**: No data-dependent branching at the IR
  level to prevent timing side channels

#### For LLM Workloads

- **Tile-based matrix operations**: First-class 2D tile types with hardware-aware
  tiling (matching tensor core shapes: 16x16, 16x8, etc.)
- **Mixed-precision semantics**: FP8 (E4M3/E5M2), BF16, TF32, FP16 with explicit
  accumulator precision control
- **Attention primitives**: Fused softmax-matmul patterns, online softmax
  (FlashAttention-style)
- **Memory hierarchy control**: Explicit async copy, prefetch, and cache
  management operations
- **Warp specialization**: Ability to assign different warps to different roles
  (producer/consumer for async pipelines)

#### General Design Principles

- **Explicit parallelism model**: Thread, warp, block, and grid levels visible
  in the IR, with explicit synchronization
- **Scheduling annotations**: Optional scheduling hints that the backend can
  use but is not required to follow
- **Target-independent core + target-specific extensions**: A stable core IR
  with vendor-specific extension mechanisms
- **MLIR-native**: Built as an MLIR dialect for composability with existing
  optimization infrastructure

### 3.3 Existing Alternative IR Approaches

#### Triton IR (TTIR / TTGIR)

OpenAI's Triton compiler has arguably already created a "better PTX" for ML workloads:

```
Compilation pipeline:
  Python -> AST -> TTIR (Triton IR, MLIR dialect)
                    |
                    v
                  TTGIR (Triton GPU IR, MLIR dialect)
                    |  with layout annotations:
                    |  - slice (tensor restructuring)
                    |  - dot_op (matrix product layout)
                    |  - shared (shared memory)
                    |  - nvidia_mma (tensor core output)
                    |  - amd_mfma (AMD matrix core)
                    |  - amd_wmma (AMD wave matrix)
                    |
                    v
                  LLVM IR
                    |
            +-------+--------+
            |                |
            v                v
          PTX              AMDGCN
            |                |
            v                v
          cubin             hsaco
```

Triton already demonstrates that:
- A higher-level IR can achieve competitive performance (FlashAttention in Triton
  matches or exceeds hand-written CUDA)
- Multi-vendor targeting is possible through MLIR (NVIDIA via PTX, AMD via AMDGCN)
- Tile-based programming abstractions map naturally to GPU hardware

**Recent development (2025)**: NVIDIA created a **CUDA Tile IR backend for Triton**,
allowing Triton to target their new Tile IR representation instead of PTX. This
suggests even NVIDIA sees value in alternative IRs.

#### ThunderKittens

Stanford's Hazy Research lab created ThunderKittens, an embedded DSL within CUDA:

- Tile-based abstractions (register tiles, shared memory tiles)
- On H100: forward FlashAttention kernel is ~30% faster than FlashAttention-2
- Abstracts warp-level matrix operations into composable primitives
- **HipKittens**: AMD port that achieves comparable performance on MI300X,
  demonstrating the abstraction's portability

#### XLA / StableHLO

Google's XLA compiler uses StableHLO as its IR for ML workloads, targeting
NVIDIA (via PTX), AMD (via AMDGCN), and TPUs. Demonstrates production viability
of multi-backend GPU compilation.

#### IREE

Google's IREE (Intermediate Representation Execution Environment) uses MLIR
throughout its compilation pipeline, with backends for Vulkan/SPIR-V, CUDA,
ROCm, and custom accelerators.

---

## 4. AMD's Open ISA Approach

### 4.1 Documentation Openness

AMD's approach to ISA documentation is fundamentally different from NVIDIA's.
AMD publishes **complete ISA reference manuals** including:

- Full instruction encoding bit layouts
- Opcode tables with binary encodings
- Microcode format specifications
- Pipeline descriptions
- Register file organization
- Memory hierarchy details

Available ISA documentation:
- **CDNA 4** (MI350+): Full ISA reference, August 2025
- **CDNA 3** (MI300): Full ISA reference, August 2025
- **CDNA 2** (MI200): Full ISA reference
- **CDNA 1** (MI100): Full ISA reference
- **RDNA 4, 3.5, 3, 2, 1**: Full ISA references
- **Vega / GCN 3**: Full ISA references

AMD also provides **machine-readable ISA specifications as XML files** with a
C++ IsaDecoder API, enabling programmatic generation of assemblers and
disassemblers.

### 4.2 AMD ISA Architecture

AMD GPUs use a **dual-unit execution model**:

```
Wavefront (64 work-items on CDNA, 32 on RDNA "wave32")
    |
    +-- Scalar ALU (SALU): one value per wavefront
    |     - Control flow, address calculation
    |     - Scalar GPRs (SGPRs)
    |
    +-- Vector ALU (VALU): one value per work-item
    |     - Arithmetic, memory operations
    |     - Vector GPRs (VGPRs), unique per work-item
    |
    +-- Matrix Core (CDNA): matrix multiply-accumulate
          - MFMA instructions (CDNA)
          - WMMA instructions (RDNA 3+)
```

#### Instruction Encoding Format

AMD GCN/CDNA/RDNA instructions use variable-length encoding:

```
32-bit base encoding:
  [31:26]  Major opcode (identifies instruction format)
  [25:0]   Format-specific fields

Common formats:
  SOP1:   Scalar, one source        (32-bit)
  SOP2:   Scalar, two sources       (32-bit)
  SOPC:   Scalar comparison          (32-bit)
  SOPK:   Scalar, inline constant   (32-bit)
  SOPP:   Scalar, program control   (32-bit)
  VOP1:   Vector, one source        (32-bit)
  VOP2:   Vector, two sources       (32-bit)
  VOP3:   Vector, three sources     (64-bit, extended encoding)
  VOP3P:  Vector, packed operations (64-bit, for mixed precision)
  MUBUF:  Memory buffer operations  (64-bit)
  FLAT:   Flat memory operations    (64-bit)
  DS:     Local data share (LDS)    (64-bit)
  MIMG:   Image/texture operations  (64-bit)
  EXP:    Export (graphics)         (64-bit)
  SMEM:   Scalar memory             (64-bit)
```

### 4.3 Targeting AMD Directly

Because AMD's ISA is documented and the compiler toolchain is open source, we can
target AMD GPUs much more directly than NVIDIA:

```
Our Custom IR
    |
    v
[Custom lowering pass]
    |
    v
LLVM IR with AMDGPU target
    |
    v
[LLVM AMDGPU backend -- fully open source in LLVM tree]
    |
    v
AMDGCN assembly (.s)
    |
    v
[llvm-mc assembler -- open source]
    |
    v
AMD GPU binary (.hsaco -- HSA Code Object, also ELF)
    |
    v
[ROCm runtime / HIP API -- open source]
    |
    v
AMD GPU execution
```

Key advantages:
- **No proprietary compiler step**: Unlike NVIDIA's ptxas, the entire AMD
  compilation chain is open source (LLVM AMDGPU backend + ROCm runtime)
- **Direct ISA control**: You can write AMDGCN assembly directly, assemble with
  `llvm-mc`, and load via ROCm
- **Documented encoding**: You know exactly how every instruction is encoded
- **Register allocation control**: LLVM's register allocator is open and
  extensible; you can write custom allocation strategies
- **Scheduling control**: LLVM's instruction scheduler is open; custom scheduling
  models can be added for specific GPU targets

### 4.4 AMD-Targeting Projects

Several projects demonstrate the viability of direct AMD GPU targeting:

- **BarraCUDA** (https://github.com/Zaneham/BarraCUDA): Compiles CUDA C directly
  to GFX11 (RDNA 3) machine code. ~1,700 lines of hand-written instruction
  selection. No HIP translation layer. Demonstrates that a minimal compiler can
  target AMD GPUs directly.

- **ZLUDA** (https://github.com/vfdev-5/zluda): Drop-in CUDA replacement that
  translates CUDA API calls to HIP/ROCm. ZLUDA v5 (December 2025) is described
  as "the most serious open source threat to CUDA yet."

- **SCALE**: Clean-room CUDA implementation using LLVM components to natively
  compile CUDA sources for AMD GPUs.

- **Triton AMD backend**: Triton's AMDGCN backend generates AMD GPU code through
  LLVM, with architecture-specific optimization passes for MFMA/WMMA instruction
  selection and shared memory layout.

---

## 5. Open-Source GPU Hardware

### 5.1 Vortex GPGPU (Georgia Tech)

The most mature open-source GPU project. Note: shares the name "Vortex" with our
project but is unrelated.

- **Repository**: https://github.com/vortexgpgpu/vortex
- **Website**: https://vortex.cc.gatech.edu/
- **ISA**: RISC-V base + 6 custom instructions for GPGPU
- **API support**: OpenCL, OpenGL
- **Implementation**: Soft GPU on FPGA (Xilinx and Altera)
- **Interface**: PCIe-based
- **Architecture**: Configurable number of cores, warps, threads
- **Published at**: MICRO 2021 ("Vortex: Extending the RISC-V ISA for GPGPU
  and 3D-Graphics Research")

The 6 added RISC-V instructions:
1. **TMC** (Thread Mask Control): Manage SIMT thread divergence
2. **WSPAWN** (Warp Spawn): Launch additional warps
3. **SPLIT/JOIN**: Handle branch divergence (split/reconverge)
4. **BAR** (Barrier): Warp/block synchronization
5. **GPU-specific CSRs**: Thread ID, warp ID, core ID registers

**Significance**: Demonstrates that a functional GPGPU can be built with
minimal ISA extensions to RISC-V. The entire software stack (compiler, driver,
runtime) is open source.

**Limitation**: FPGA implementation means orders-of-magnitude slower than
commercial GPUs. Suitable for research, not production workloads.

### 5.2 Other Open-Source GPU Projects

#### RV64X
- **Repository**: https://github.com/avl-bsuir/rv64x-base
- **Approach**: 64-bit GPU instruction extensions built on RISC-V Vector ISA
- **Status**: Early stage / specification phase

#### VeriGPU
- **Repository**: https://github.com/hughperkins/VeriGPU
- **Approach**: OpenSource GPU in Verilog, loosely RISC-V compliant
- **Target API**: HIP-compatible (for PyTorch integration)
- **Status**: Experimental

#### Rivos (Meta acquisition)
- Meta acquired RISC-V AI GPU startup Rivos
- Designing GPUs and AI accelerators on RISC-V open standard
- Could become commercially relevant but details are private

### 5.3 Viability Assessment for Open Hardware

For **research and prototyping**: Open-source GPU hardware (especially Vortex) is
excellent. You get full-stack visibility, can modify the ISA, and can experiment
with novel architectural features.

For **production crypto/LLM workloads**: Not viable in the near term.
- FPGA-based GPUs are 10-100x slower than commercial ASICs
- No open-source GPU has tensor core equivalents
- Memory bandwidth (the actual bottleneck for LLM inference) requires advanced
  HBM integration that no open project supports
- Commercial GPUs (H100, MI300X) have 1-3 TB/s memory bandwidth; FPGA designs
  achieve perhaps 10-50 GB/s

**However**: Open hardware provides a testbed for validating IR designs before
targeting commercial hardware. A custom IR proven correct on Vortex GPGPU can
then be lowered to PTX/AMDGCN for production deployment.

---

## 6. Cross-Vendor IR: The MLIR Path

### 6.1 MLIR as Foundation

MLIR (Multi-Level Intermediate Representation) is the most promising foundation
for a cross-vendor GPU IR. It provides:

- **Dialect system**: Self-contained namespaces for domain-specific abstractions
- **Progressive lowering**: High-level operations progressively decompose through
  multiple IR levels toward machine code
- **Existing GPU dialects**: `gpu`, `nvvm`, `rocdl`, `spirv`, `nvgpu`
- **Optimization infrastructure**: Pattern matching, rewriting, canonicalization
- **Active development**: Backed by Google, NVIDIA, AMD, Intel, and others

### 6.2 Proposed Multi-Backend Architecture

```
Vortex Source Code (Rust-like syntax)
    |
    v
[Vortex Frontend Compiler]
    |
    v
Vortex MLIR Dialect (high-level, domain-specific)
    |  Operations: crypto.ntt, crypto.montgomery_mul,
    |  llm.attention, llm.matmul_mixed, etc.
    |
    v
[Domain-specific optimization passes]
    |  NTT butterfly fusion, attention pattern matching,
    |  tile size selection, memory hierarchy planning
    |
    v
GPU MLIR Dialect (hardware-aware, vendor-neutral)
    |  Operations: gpu.launch, gpu.block_tile_matmul,
    |  gpu.async_copy, gpu.barrier, etc.
    |
    +---> [NVIDIA lowering path]
    |         |
    |         v
    |     NVVM Dialect + NVGPU Dialect
    |         |  WMMA/MMA ops, TMA ops, etc.
    |         v
    |     LLVM IR (NVPTX target)
    |         |
    |         v
    |     PTX assembly
    |         |
    |         v
    |     [ptxas] -> cubin -> CUDA Driver
    |
    +---> [AMD lowering path]
    |         |
    |         v
    |     ROCDL Dialect
    |         |  MFMA ops, LDS ops, etc.
    |         v
    |     LLVM IR (AMDGPU target)
    |         |
    |         v
    |     AMDGCN assembly
    |         |
    |         v
    |     [llvm-mc] -> hsaco -> ROCm Runtime
    |
    +---> [Intel lowering path]
    |         |
    |         v
    |     SPIRV Dialect
    |         |
    |         v
    |     SPIR-V binary
    |         |
    |         v
    |     [Level Zero Driver] -> Intel GPU
    |
    +---> [Open hardware path]
              |
              v
          RISC-V + Vortex GPGPU extensions
              |
              v
          [LLVM RISC-V backend] -> Vortex GPGPU
```

### 6.3 Precedent: HETOCompiler

The HETOCompiler project (arxiv:2407.09333) demonstrates MLIR-based compilation
specifically for cryptographic workloads on heterogeneous hardware:

- Created a "crypto" MLIR dialect encapsulating hash/encryption algorithms
- Supports multiple target backends through MLIR's lowering infrastructure
- Demonstrates that domain-specific crypto primitives can be represented
  effectively in MLIR and lowered to efficient GPU code

### 6.4 SPIR-V as Cross-Vendor Target

SPIR-V could serve as an alternative intermediate target:

**Advantages**:
- Vendor-neutral, standardized by Khronos
- Supported by NVIDIA (Vulkan), AMD (Vulkan/OpenCL), Intel (Vulkan/OpenCL/L0)
- Binary SSA form, retains high-level type information
- Eliminates need for vendor-specific front-end compilers in drivers

**Disadvantages**:
- Performance typically 10-30% below native paths (CUDA, ROCm)
- Limited access to vendor-specific hardware features (tensor cores)
- Compute shader model less flexible than CUDA's programming model
- No direct access to shared memory allocation or warp-level primitives in
  standard SPIR-V (requires extensions)

**Verdict**: SPIR-V is suitable as a fallback/portability target but not as the
primary path for maximum performance. Use SPIR-V for Intel GPUs (where it is
the native path) and for portable baseline implementations.

---

## 7. Practical Assessment

### 7.1 What is Actually Achievable

| Approach | Feasibility | Performance | Effort | Risk |
|----------|------------|-------------|--------|------|
| Custom MLIR dialect -> PTX -> cubin | **High** | 90-100% of CUDA | Medium | Low |
| Custom MLIR dialect -> AMDGCN -> hsaco | **High** | 90-100% of ROCm | Medium | Low |
| Custom MLIR dialect -> SPIR-V | **High** | 70-90% of native | Medium | Low |
| Direct SASS generation (bypass ptxas) | **Medium** | 100-115% of ptxas | Very High | High |
| Custom SASS IR (new ISA for NVIDIA) | **Low** | Theoretical max | Extreme | Very High |
| Open-source GPU hardware | **Medium** | 1-10% of commercial | High | Medium |
| New GPU architecture (FPGA) | **Low** | Variable | Extreme | Very High |

### 7.2 Performance vs. Effort Tradeoff

```
Performance
    ^
    |
115%|                              * Direct SASS (hand-tuned)
    |                           * CuAsmRL (RL-optimized SASS)
100%|            * PTX (via ptxas)
    |         * Triton
 95%|      * Custom MLIR->PTX
    |    * ThunderKittens
 90%|  * CUDA (standard)
    |
 80%|                    * SPIR-V (Vulkan compute)
    |
    +----+----+----+----+----+----+----+----+----> Effort
         Low       Medium      High      Extreme
```

### 7.3 Risk Analysis

**Direct SASS generation risks**:
- Architecture breaks every ~2 years (Turing->Ampere->Hopper->Blackwell)
- NVIDIA could change cubin format or add validation
- Legal gray area (reverse engineering proprietary ISA)
- Debugging is extremely difficult (no NVIDIA tooling support)
- Incorrect scheduling produces silent correctness bugs

**MLIR-to-PTX risks**:
- Dependent on ptxas quality (generally good, occasionally suboptimal)
- PTX may not expose all hardware features immediately
- Still requires NVIDIA's proprietary toolchain at some point

**AMD direct targeting risks**:
- AMD GPU market share in datacenter is growing but still minority
- ISA stability across generations (GCN -> CDNA -> RDNA split)
- ROCm software ecosystem maturity vs. CUDA

### 7.4 The DeepSeek Precedent

DeepSeek's V3 training demonstrated that hand-written PTX can unlock significant
performance improvements for specific workloads:

- Bypassed CUDA C++ entirely for communication kernels
- Used PTX to control SM allocation (20/132 SMs for communication)
- Achieved ~10x higher training efficiency than contemporary approaches
- Their DeepEP library uses custom PTX instructions for expert-parallel
  communication, with auto-tuned chunk sizes that "significantly reduce the
  use of the L2 cache"

This validates that for performance-critical paths, going below CUDA's abstraction
level to PTX (or even SASS) is a viable and valuable strategy. However, DeepSeek
still used PTX (not SASS), suggesting that PTX provides sufficient control for
most optimizations while maintaining architecture portability.

---

## 8. Recommended Architecture for Vortex

### 8.1 Recommended Path: MLIR-Based Multi-Target Compilation

Based on this analysis, the recommended architecture for Vortex's GPU backend:

```
Phase 1: MLIR Dialect Design (Foundation)
------------------------------------------
- Define `vortex.crypto` dialect with:
  - vortex.crypto.ntt (Number Theoretic Transform)
  - vortex.crypto.montgomery_mul (modular multiplication)
  - vortex.crypto.barrett_reduce (modular reduction)
  - vortex.crypto.keccak_round (hash primitive)
  - vortex.crypto.wide_mul (256/512-bit multiply)

- Define `vortex.ml` dialect with:
  - vortex.ml.matmul_tiled (tile-aware matrix multiply)
  - vortex.ml.attention (fused attention pattern)
  - vortex.ml.softmax_online (numerically stable online softmax)
  - vortex.ml.rmsnorm (fused RMS normalization)

Phase 2: NVIDIA Backend via PTX (Primary Target)
-------------------------------------------------
- Lower vortex dialects -> gpu dialect -> nvvm dialect -> LLVM IR -> PTX
- Use ptxas for SASS generation (accept its scheduling)
- For hot paths: lower to hand-tuned PTX using inline assembly patterns
- Optionally: post-process SASS with CuAssembler for scheduling optimization

Phase 3: AMD Backend via AMDGCN (Secondary Target)
---------------------------------------------------
- Lower vortex dialects -> gpu dialect -> rocdl dialect -> LLVM IR -> AMDGCN
- Full open-source toolchain, no proprietary dependencies
- Leverage documented ISA for optimal instruction selection
- Target CDNA 3/4 (MI300X/MI350) for datacenter workloads

Phase 4: Advanced Optimization (Optional)
------------------------------------------
- SASS-level post-processing for NVIDIA hot paths
  (use CuAssembler to adjust scheduling of ptxas output)
- Custom register allocation hints via PTX pragmas
- Architecture-specific kernel variants with auto-tuning
```

### 8.2 Why Not Bypass PTX Entirely?

While technically possible, bypassing PTX entirely for NVIDIA is not recommended
for a language project because:

1. **Maintenance burden**: Supporting 4+ NVIDIA architectures requires maintaining
   4+ SASS encoding backends that break with each new GPU generation
2. **Diminishing returns**: PTX + ptxas achieves ~95% of theoretical peak for most
   workloads; the remaining 5% from SASS tuning is not worth the infrastructure cost
3. **Debugging difficulty**: NVIDIA's debugging tools (cuda-gdb, compute-sanitizer,
   Nsight) all work with PTX; direct SASS has no tooling
4. **Forward compatibility**: PTX guarantees code works on future GPUs; direct SASS
   is locked to a single architecture

**Exception**: For a small number of critical hot-path kernels (NTT butterflies,
matrix multiply inner loops), SASS-level post-optimization via CuAssembler is
justified and practical.

### 8.3 The Strategic View

The most impactful investment is not in creating a new ISA or bypassing PTX, but
in creating **domain-specific optimizations at the IR level** that generate better
PTX/AMDGCN than general-purpose compilers can:

- A `crypto.ntt` operation that understands butterfly decomposition can generate
  optimal PTX for any GPU architecture
- A `ml.attention` operation that understands FlashAttention's tiling strategy
  can generate PTX that rivals hand-written CUDA
- These optimizations are **portable across vendors** and **stable across
  architecture generations**

The real competitive advantage is not in low-level ISA hacking but in
**high-level domain knowledge encoded in the compiler**.

---

## Appendix A: Key Projects and References

### Tools and Projects

| Project | URL | Purpose |
|---------|-----|---------|
| CuAssembler | https://github.com/cloudcores/CuAssembler | SASS assembler (Pascal-Ampere) |
| turingas | https://github.com/daadaada/turingas | SASS assembler (Volta/Turing) |
| maxas | https://github.com/NervanaSystems/maxas | SASS assembler (Maxwell/Pascal) |
| nv_isa_solver | https://github.com/kuterd/nv_isa_solver | ISA spec generator via fuzzing |
| DocumentSASS | https://github.com/0xD0GF00D/DocumentSASS | Unofficial SASS documentation |
| Triton | https://github.com/triton-lang/triton | ML kernel compiler (MLIR-based) |
| ThunderKittens | https://hazyresearch.stanford.edu/blog/2024-05-12-quick-tk | Tile-based GPU DSL |
| Vortex GPGPU | https://github.com/vortexgpgpu/vortex | Open-source RISC-V GPU |
| NVIDIA open-gpu-doc | https://github.com/NVIDIA/open-gpu-doc | HW interface documentation |
| NVIDIA open-gpu-kernel-modules | https://github.com/NVIDIA/open-gpu-kernel-modules | Open kernel driver |
| BarraCUDA | https://github.com/Zaneham/BarraCUDA | CUDA compiler for AMD GPUs |
| ZLUDA | https://github.com/vfdev-5/zluda | CUDA-to-HIP translation layer |
| DeepEP | https://github.com/deepseek-ai/DeepEP | DeepSeek's expert-parallel library |
| HETOCompiler | https://arxiv.org/html/2407.09333v1 | MLIR crypto compilation |

### Papers

| Paper | Venue | Relevance |
|-------|-------|-----------|
| Demystifying NVIDIA GPU Internals (Bakita & Anderson) | RTAS 2024 | GPU scheduling internals |
| CuAsmRL: Optimizing GPU SASS Schedules via Deep RL | arXiv 2501.08071 | SASS optimization via RL |
| Benchmarking and Dissecting the Nvidia Hopper Architecture | arXiv 2402.13499 | H100 microarchitecture |
| Demystifying the Nvidia Ampere Architecture | arXiv 2208.11174 | A100 microarchitecture |
| Vortex: Extending the RISC-V ISA for GPGPU (Tine et al.) | MICRO 2021 | Open-source GPU design |
| DeepSeek-V3 Technical Report | arXiv 2412.19437 | PTX optimization for LLM training |
| ML-Triton, A Multi-Level Compilation and Language | arXiv 2503.14985 | Triton compiler evolution |

### AMD ISA Documentation

| Document | Architecture |
|----------|-------------|
| CDNA 4 ISA Reference Guide (Aug 2025) | MI350+ |
| CDNA 3 ISA Reference Guide (Aug 2025) | MI300 |
| CDNA 2 ISA Reference Guide | MI200 |
| RDNA 4, 3.5, 3, 2, 1 ISA Reference Guides | Consumer GPUs |
| AMD ISA Documentation Portal | https://gpuopen.com/amd-isa-documentation/ |

---

## Appendix B: NVIDIA Architecture ISA Summary

| Generation | SM | Instruction Size | Control Codes | Key Features |
|-----------|-----|------------------|---------------|-------------|
| Fermi | SM_20 | 64-bit | Separate | First CUDA-capable |
| Kepler | SM_30/35 | 64-bit | 1 per 7 instr | Dynamic parallelism |
| Maxwell | SM_50/52 | 64-bit | 1 per 3 instr | Shared memory banks |
| Pascal | SM_60/61 | 64-bit | 1 per 3 instr | FP16 support |
| Volta | SM_70 | 128-bit | Embedded | Tensor cores (1st gen) |
| Turing | SM_75 | 128-bit | Embedded | RT cores, INT8 |
| Ampere | SM_80/86 | 128-bit | Embedded | TF32, sparsity |
| Ada | SM_89 | 128-bit | Embedded | FP8, 4th gen tensor |
| Hopper | SM_90 | 128-bit | Embedded | TMA, WGMMA, DPX |
| Blackwell | SM_100/120 | 128-bit | Embedded | TCGEN05, 5th gen tensor |

---

## Appendix C: GPU IR Comparison

| IR | Vendor | Level | Multi-Target | Open Source | Performance |
|----|--------|-------|-------------|-------------|-------------|
| PTX | NVIDIA | Low (virtual) | No | Spec only | ~95% peak |
| SASS | NVIDIA | Machine code | No | Reverse-eng. | 100% peak |
| AMDGCN | AMD | Machine code | No | Full docs + LLVM | 100% peak |
| SPIR-V | Khronos | Medium | Yes | Full spec + tools | 70-90% peak |
| Triton IR | OpenAI | High | Yes (NV/AMD) | Yes (MLIR) | 90-100% peak |
| MLIR GPU | LLVM | Medium | Yes | Yes | Depends on lowering |
| StableHLO | Google | High | Yes (NV/AMD/TPU) | Yes | 85-95% peak |
| CUDA Tile IR | NVIDIA | Medium | No | Partial | ~95% peak |

---

*Analysis completed February 2026. GPU architectures and tooling evolve rapidly;
specific version numbers and performance claims should be verified against current
documentation.*
