# Attention Mechanism Acceleration: Cross-Paradigm Analysis

## Detailed Implementation & Theory Guide

**Talk**: *Analysis of Attention Mechanisms in the Context of Computational Paradigms*
**Presented at**: Fifth Serbian International Conference on Applied Artificial Intelligence (SICAAI), Kragujevac, 20-21 May 2026.
**Author**: Uroš Savurdić, School of Electrical Engineering, University of Belgrade

---

## 1. The Attention Mechanism

### 1.1 Mathematical Formulation

The scaled dot-product attention mechanism computes:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) · V
```

We decompose this into 5 measurable steps:

| Step | Operation | Dimensions | FLOPs |
|------|-----------|-----------|-------|
| 1. QKV Projection | Q=XW_q, K=XW_k, V=XW_v | (n,d)×(d,d)→(n,d) ×3 | 6·n·d² |
| 2. Score Computation | S = QK^T / √d_k | (n,d)×(d,n)→(n,n) | 2·n²·d |
| 3. Softmax | A = softmax(S) | (n,n)→(n,n) | 5·n² |
| 4. Weighted Aggregation | O = A·V | (n,n)×(n,d)→(n,d) | 2·n²·d |
| 5. Output Projection | Y = O·W_o | (n,d)×(d,d)→(n,d) | 2·n·d² |

Where n = sequence length, d = embedding dimension.

### 1.2 W_o — Output Projection

W_o is the output projection matrix from the original "Attention is All You Need" (Vaswani et al., 2017). In multi-head attention, each head produces a d_v-dimensional output, and W_o linearly combines all heads. Even in single-head attention, W_o enables the model to learn a linear transformation of the attended values, adding representational capacity.

### 1.3 Computational Complexity

- **Total FLOPs**: 8·n·d² + 4·n²·d + 5·n²
- **Scaling**: O(n²·d) — quadratic in sequence length due to the n×n attention matrix
- **Memory**: The n×n score matrix dominates memory at large n (e.g., n=1024 → 4MB in FP32)

### 1.4 FLOP Counting Convention

For matrix multiplication (M,K) × (K,N):
- Each output element requires K multiply-accumulate (MAC) operations
- Each MAC = 1 multiply + 1 addition = 2 FLOPs
- Total: 2·M·K·N FLOPs

---

## 2. Computational Paradigms

### 2.1 P01: CPU — Multi-core Control Flow

#### Real-World Architecture
Modern CPUs (Intel i7/Xeon, AMD EPYC) execute attention through:
- **SIMD/AVX2**: 8 FP32 ops per cycle per core (256-bit vector width)
- **Cache hierarchy**: L1 (32KB, 4 cycles), L2 (256KB, 12 cycles), L3 (8-32MB, 40 cycles)
- **Multi-core**: 4-16 cores with shared L3
- **BLAS libraries**: OpenBLAS/MKL achieve near-peak FLOPS via tiled matrix multiply

#### Our Model
- **Hardware reference** (roofline only; measurements ran on the Colab T4 runtime's two-vCPU Xeon host): Intel i7 with AVX2 (8 cores × 3.8 GHz)
- **Peak FP32**: 0.5 TFLOPS (theoretical: 8 cores × 16 FP32 ops/cycle × 3.8 GHz)
- **Memory bandwidth**: 50 GB/s (DDR4-3200 dual-channel)
- **Model**: Roofline — per step, time = max(FLOPs/peak_compute, bytes/bandwidth)

#### Implementation (`paradigms/p01_cpu/paradigm.py`)
- Actually executes attention using NumPy (which calls optimized BLAS)
- Measures real wall time (perf_counter) for validation
- Computes roofline estimate as `estimated_hw_time_seconds`
- Reports per-step bottleneck: "compute" or "memory"

#### Key Insight
At d=128, matmul steps are **compute-bound** (arithmetic intensity ~29 FLOPs/byte > CPU's balance point of ~10). Softmax is **memory-bound** (low arithmetic intensity, dominated by memory reads/writes).

---

### 2.2 P02: GPU — Many-core Control Flow

#### Real-World Architecture
GPUs (NVIDIA T4/A100) achieve massive parallelism through:
- **Thousands of CUDA cores**: T4 has 2,560 FP32 cores
- **Thread blocks**: warps of 32 threads, hardware-scheduled
- **Shared memory**: 48-164 KB per SM, explicit programmer control
- **Memory bandwidth**: 320 GB/s (GDDR6, T4) to 2,000 GB/s (HBM, A100)
- **Kernel launch overhead**: ~10 µs per kernel dispatch

#### Our Model
- **Hardware reference**: NVIDIA T4 (Google Colab free tier)
- **Peak FP32**: 8.1 TFLOPS
- **Memory bandwidth**: 320 GB/s
- **Model**: Roofline + 10µs kernel launch overhead per step
- **Dual mode**: Real CUDA execution when available; roofline simulation otherwise

#### Implementation (`paradigms/p02_gpu/paradigm.py`)
- `is_available()` always returns True (simulation fallback)
- With CUDA: uses `torch.cuda.Event` for precise GPU timing + roofline estimates
- Without CUDA: NumPy reference execution + roofline estimates
- Reports `execution_mode: "real_cuda"` or `"roofline_simulation"`

#### Key Insight
In the roofline model the GPU's peak is 16x the CPU's (8.1 vs 0.5 TFLOPS). Measured on the T4, fixed per-operation cost dominates small sizes: at n=128, d=64 each of the five decomposed stages takes 33–93 µs whatever its size, and decomposed FP32 attention reaches 49% of the nominal FP32 peak only at n=2048, d=256.

---

### 2.3 P03: GaAs (RISC-V) — Minimal ISA Processor

#### Real-World Context
Gallium Arsenide (GaAs) processors were developed in the 1980s-90s under DARPA programs (e.g., Vitesse Semiconductor, Convex Computer) for military and space applications:
- **Faster switching**: GaAs transistors switch at ~1 GHz (2-3x faster than contemporary silicon)
- **Lower integration**: far fewer transistors per die (thousands vs millions)
- **Radiation hardness**: important for space/military
- **No cache, no branch prediction, no superscalar** — just raw instruction execution

The question this paradigm answers: "What is the instruction-level cost of attention on a minimal processor?"

#### Our Model
- **ISA**: RISC-V RV32I-like subset with 11 instructions (LW, SW, ADD, MUL, FADD, FMUL, FDIV, BEQ, BNE, JAL, HALT)
- **Pipeline**: simple 5-stage, no forwarding
- **Cycle costs**: ALU=1, FADD=2, FMUL=3, LW=4, FDIV=10
- **Clock**: 1 GHz (GaAs-typical)
- **No parallelism**: strictly sequential, one instruction at a time

#### Implementation (`paradigms/p03_gaas_riscv/`)
- `isa.py`: Defines opcodes, Instruction dataclass, `compile_matmul()` and `compile_softmax()` that generate instruction sequences
- `simulator.py`: Full fetch-decode-execute loop with 32 registers and flat memory
- `paradigm.py`: Uses **analytical formulas** for large sizes:
  - Per matmul element: 4K+2 instructions (K = inner dimension)
  - Per element cycle cost: 1 + K*(4+4+3+2) + 3 = 13K+4 cycles
  - Compiles 4×4 demo for instruction breakdown verification

#### Key Insight
At n=1024, d=128: total ~4.4 billion cycles → 4.4 seconds at 1 GHz, about 160 times the measured CPU time for the same work. The 1 GHz clock is itself optimistic: GaAs microprocessors of the period were designed for about 200 MHz. This demonstrates why parallelism (SIMD, multiple ALUs) is essential — raw clock speed alone cannot serve attention's computational demands.

---

### 2.4 P04: Minimal-ISA Machine (`cdte_turing`) — Fundamental Computational Limits

#### Real-World Context
Cadmium Telluride (CdTe) represents the theoretical extreme: a processor with so few transistors that only the most primitive operations are possible. This paradigm explores:
- **What is the absolute minimum computation needed for attention?**
- **How does the cost explode when you lack hardware multipliers?**

Real CdTe semiconductors are used primarily in solar cells and X-ray/gamma detectors — not computing. This is a thought experiment: "What if your chip can only skip, conditionally skip, increment, and add?"

#### Our Model
- **ISA**: Only 4 operations:
  - `SKIP` — move tape head right by 1
  - `CSKIP` — conditional skip (move if tape[head] == 0)
  - `INC` — tape[head] += 1
  - `ADD` — tape[head] += tape[head + offset]
- **Arithmetic**: Integer only (no floating-point hardware)
- **Multiplication**: Implemented as repeated addition → O(a×b) operations for a single multiply
- **Clock**: 10 MHz (minimal transistor budget)

#### Implementation (`paradigms/p04_cdte_turing/`)
- `turing_machine.py`: Tape-based machine, `program_multiply()` via repeated addition, `program_dot_product()`, analytical `estimate_matmul_ops()` and `estimate_softmax_ops()`
- `paradigm.py`: Runs tiny demo (4×4) on actual TM, uses analytical estimates at full scale
- Integer scale: float values mapped to [0, 10], so average multiply costs (5)² = 25 INC operations

#### Key Insight
At n=1024, d=128: **~9.5 billion TM operations → 949 seconds at 10 MHz**, about 2 million times the measured FP32 GPU time. The fundamental lesson: hardware multipliers provide enormous speedup. A single `FMUL` instruction (3 cycles on GaAs) replaces ~25 primitive operations. Softmax (requiring exp/div) is even more catastrophic — these transcendental functions explode into thousands of TM ops each.

---

### 2.5 P05: Dataflow (Maxeler-style) — Streaming Pipeline

#### Real-World Architecture
Maxeler Technologies builds Dataflow Engines (DFEs) — FPGA-based accelerators that:
- **Eliminate instruction fetch**: computation is spatially mapped into hardware
- **Stream data through pipelines**: each pipeline stage is a physical operator
- **Guaranteed bandwidth**: data arrives deterministically, no cache misses
- **Kernel replication**: multiple parallel pipeline instances on one chip
- **Target**: HPC, finance (Monte Carlo), seismic processing, genomics

The Mathematical Institute of SANU (Serbian Academy of Sciences) has a Maxeler machine — potential for real execution.

#### Our Model
- **Pipeline structure**: For matmul of dot-product length K → K MAC stages (each 5 cycles: FMUL=3 + FADD=2)
- **Streaming**: once pipeline fills (K×5 cycles), produces 1 result per cycle
- **Total cycles**: pipeline_depth + stream_length - 1
- **Parallel pipes**: 4 kernel instances (realistic for MAX4/MAX5 DFE)
- **Clock**: 500 MHz
- **Softmax pipeline**: max_reduce(N) → subtract → exp(8 cycles) → sum_reduce(N) → divide(10 cycles)

#### Implementation (`paradigms/p05_dataflow/`)
- `pipeline.py`: `PipelineNode(name, latency, ops_per_element)`, `DataflowPipeline` class with simulate()
- Separate pipelines for matmul (K MAC stages) and softmax (5 specialized stages)
- `paradigm.py`: Correct stream lengths (3·n·d for QKV, n² for QK^T, etc.), divides by parallel_pipes

#### Key Insight
The model reaches about 490 GFLOP/s at n=1024, d=128 at 500 MHz. Published Maxeler designs ran at 100–180 MHz with up to ~956 floating-point operations per cycle, so the model is an optimistic bound, roughly 5x above what a real engine of that generation would deliver. In the model, the advantages come from:
1. Zero instruction fetch/decode overhead
2. Guaranteed data delivery (no cache misses)
3. High utilization in steady state (all pipeline stages active simultaneously)
4. Each pipeline of K=128 stages performs 256 FLOPs/cycle internally

The disadvantage: pipeline fill/drain overhead, and the softmax pipeline is complex (requires buffering for reductions).

---

### 2.6 P06: TPU (Systolic Array) — 2D Processing Element Array

#### Real-World Architecture
Google's TPU cores from v2 to v5 each use 128×128 systolic matrix units (TPU v1 used a 256×256 8-bit array):
- **128×128 systolic array** (Matrix Multiply Unit — MXU)
- **Weight-stationary dataflow**: weights loaded once, activations stream through
- **Staggered feeding**: row i of A enters i cycles late; column j of B enters j cycles late
- **Each PE**: multiply-accumulate (MAC) every cycle
- **Peak**: 128×128×2×700 MHz = 22.9 TFLOPS
- **Vector unit**: 128-wide for non-matmul ops (softmax, element-wise)
- **Deterministic execution**: no caches, no branch prediction

#### Our Model
- **Array size**: 128×128 PE grid (TPU v2–v5 MXU scale)
- **Tiling**: matrices larger than 128 are tiled; tiles computed sequentially
- **Cycles per tile**: K_dim + array_size + array_size - 2 (streaming + fill/drain)
- **Softmax**: 5·n² ops on 128-wide vector unit → (5·n²)/128 cycles
- **Clock**: 700 MHz
- **Dual mode**: Real TPU via PyTorch XLA (Colab), or simulation

#### Implementation (`paradigms/p06_tpu_systolic/`)
- `systolic_array.py`: `SystolicArray(rows, cols)`, `matmul_cycles(M, K, N)` with tiling math, `simulate_matmul()` for correctness (at 16×16 scale)
- `paradigm.py`: Uses 128×128 for cycle estimates, 16×16 for correctness verification
- Reports PE utilization: how many of 128² PEs do useful work per tile

#### Key Insight
The measured TPU is the fastest implementation (3,410 GFLOP/s fused at n=1024, d=128; 15,772 at n=2048, d=256) because:
1. **Massive parallelism**: 16,384 MACs per cycle in steady state
2. **Data reuse**: weight-stationary means each weight is loaded once, used 128 times
3. **No instruction overhead**: fixed-function hardware, deterministic
4. **Well-matched to attention**: attention is dominated by matmul (85%+ of FLOPs)

In the systolic cycle model, softmax runs on the 128-wide vector unit rather than the 16,384-MAC array, which makes it the model's largest single stage at n=1024. The measured per-stage TPU timings cannot confirm this: each separately dispatched stage takes 137–197 µs at n=1024, d=128 whatever its FLOPs, which is dispatch, not arithmetic.

---

### 2.7 P07: Optical Computing (Photonic MZI Mesh)

#### Real-World Context
Photonic computing uses light to perform matrix multiplication at the speed of light. Mach-Zehnder Interferometer (MZI) meshes (e.g., Lightmatter, Luminous) encode matrix weights as phase shifts in optical waveguides. Input vectors are encoded as optical intensities via DACs, propagate through the mesh, and the output is read by ADCs.

Key advantages: speed-of-light propagation, massive parallelism within the mesh, energy efficiency. Key challenge: nonlinear operations (softmax) require Optical-to-Electronic-to-Optical (OEO) conversion.

#### Our Model
- **Mesh size**: 64×64 MZI array (realistic for current photonic chips)
- **Propagation time**: 0.1 ns per tile (speed of light through waveguides)
- **DAC/ADC rate**: 10 GHz (state-of-the-art converters)
- **OEO latency**: 10 ns per conversion (optical ↔ electronic domain crossing)
- **Electronic softmax clock**: 2 GHz (electronic co-processor for nonlinear ops)
- **Tiling**: matrices larger than 64 are tiled; tiles processed sequentially

#### Implementation (`paradigms/p07_optical/`)
- `mzi_mesh.py`: MZIMesh class with `tile_time()`, `matmul_time(M, K, N)` with tiling, `softmax_time(seq_len)` with OEO model
- `paradigm.py`: OpticalParadigm using MZIMesh for all 5 attention steps

#### Key Insight
The model gives 510 GFLOP/s at n=2048, d=256. Its optical matrix products take only 66 µs; softmax, computed on a scalar 2 GHz electronic unit, takes 10.5 ms, of which the per-row OEO latency is only 20 µs. The modelled time is therefore set by the electronic unit that computes the nonlinearity, not by domain conversion. The model also omits reprogramming the mesh: in attention K and V are activations, so an MZI mesh would need run-time reconfiguration, which takes 100 ns to microseconds per tile on built devices.

---

### 2.8 P08: Chemical Computing (Reaction-Diffusion CRN)

#### Real-World Context
Chemical Reaction Networks (CRNs) compute using molecular concentrations as signals and bimolecular reactions as operations. Research groups (Qian & Winfree, Caltech) have demonstrated CRN-based computation of simple functions. The key operation: two species A + B → C at a rate determined by concentrations and reaction kinetics.

For matrix multiplication, each dot product element requires K sequential bimolecular reactions. Parallelism comes from independent reaction wells (microfluidic compartments).

#### Our Model
- **Reaction rate**: 1 kHz (1 ms per bimolecular reaction — optimistic for engineered systems)
- **Well size**: 256 parallel wells (microfluidic chip scale)
- **Mixing time**: 1 ms per batch (reagent preparation between rounds)
- **Exponential rate**: 100 Hz (slower for complex multi-step exponential reactions)
- **Max problem size**: n=32, d=16 (capped due to extreme slowness)

#### Implementation (`paradigms/p08_chemical/`)
- `reaction_network.py`: ChemicalReactionNetwork with batch-round matmul and reaction-based softmax
- `paradigm.py`: ChemicalParadigm with SIM_MAX capping

#### Key Insight
Chemical CRN achieves 0.0002 GFLOP/s at n=32, d=16. Each dot product requires K sequential reaction rounds (K=16 → 16 ms) plus mixing. The parallelism (256 wells) helps but cannot overcome the millisecond timescale of individual reactions. Dense matrix multiply is a poor fit for reaction-based computing.

---

### 2.9 P09: Biological Computing (DNA Strand Displacement)

#### Real-World Context
DNA strand displacement cascades (Qian & Winfree, Science 2011; Seelig et al., Science 2006) compute using DNA hybridization. Each computation step involves toehold-mediated strand displacement: a short "toehold" sequence initiates binding, which displaces an existing strand. Multiplication requires a cascade of 6 displacement steps (for bit-serial arithmetic). Each step takes ~10 seconds at typical DNA concentrations.

#### Our Model
- **Hybridization rate**: 0.1 Hz (10 seconds per strand displacement)
- **Cascade depth per multiply**: 6 steps (bit-serial multiplication circuit)
- **Max parallelism**: 1024 (independent strands in solution)
- **Enzymatic exponential**: 20 cascade steps for exp() approximation
- **Max problem size**: n=32, d=16 (capped — 11,400 seconds even at this size)

#### Implementation (`paradigms/p09_biological/`)
- `dna_computing.py`: DNAComputer with cascade-based matmul and enzymatic softmax
- `paradigm.py`: BiologicalParadigm with SIM_MAX capping

#### Key Insight
Biological DNA computing achieves ~10⁻⁸ GFLOP/s — the slowest paradigm by orders of magnitude. A single multiply takes 60 seconds (6 × 10s displacement steps). Even with 1024 parallel strands, a 32×16 matmul requires 8 sequential rounds of 60 seconds each. This represents the fundamental speed limit of molecular computation: reactions at biological timescales.

---

### 2.10 P10: Quantum Computing (Gate-based)

#### Real-World Context
Gate-based quantum computers (IBM, Google, IonQ) use quantum gates to manipulate qubits. For arithmetic, multiplication decomposes into Toffoli gates, each requiring 7 T-gates in fault-tolerant implementations. Dense matrix multiplication offers no quantum speedup (no known quantum algorithm beats classical O(n³) for dense matmul).

The question: "What happens when you try to run dense linear algebra on a quantum computer?"

#### Our Model
- **Gate time**: 50 ns (superconducting qubit regime)
- **Measurement time**: 1 µs per qubit measurement
- **Toffoli decomposition**: 7 T-gates per Toffoli (standard fault-tolerant)
- **Qubit count**: 1000 (optimistic near-term)
- **Precision**: 16-bit → 16² = 256 Toffolis per multiply → 1,792 T-gates per multiply
- **Parallel capacity**: 1000 qubits / 32 qubits per multiply = 31 parallel multiplies
- **Softmax**: hybrid quantum-classical (measure, classical softmax, re-encode)
- **Max problem size**: n=32, d=16 (capped)

#### Implementation (`paradigms/p10_quantum/`)
- `quantum_circuit.py`: QuantumArithmeticCircuit with gate counting model
- `paradigm.py`: QuantumParadigm with SIM_MAX capping

#### Key Insight
Quantum achieves 0.0006 GFLOP/s at n=32, d=16. The massive gate overhead (1,792 T-gates per single multiply at 50 ns each ≈ 90 µs per multiply) combined with limited parallelism (31 concurrent multiplies) makes quantum computing extremely inefficient for dense matmul. This is expected — quantum advantage lies in problems with exploitable structure (factoring, search, simulation), not brute-force arithmetic.

---

### 2.11 P11: Wireless Sensor Network (Distributed Computing)

#### Real-World Context
Wireless Sensor Networks (WSNs) consist of many low-power nodes (e.g., TI CC2650, ATmega328P) communicating via IEEE 802.15.4 (ZigBee, 250 kbps). The question: "Can you distribute attention computation across a sensor network?"

This models federated inference at the extreme edge — where no single node has enough memory for the full computation.

#### Our Model
- **Nodes**: 64 sensor nodes
- **MCU clock**: 16 MHz (typical 8/16-bit MCU)
- **Cycles per MAC**: 4 (software floating-point)
- **Node RAM**: 10 KB (fits ~29×29 tile at most)
- **Wireless**: 250 kbps (IEEE 802.15.4) + 5 ms hop latency
- **Softmax sync**: 3 global rounds (broadcast max, broadcast sum, final normalize)

#### Implementation (`paradigms/p11_wsn/`)
- `sensor_network.py`: SensorNetwork with distributed tiling, communication model, sync rounds
- `paradigm.py`: WSNParadigm

#### Key Insight
WSN achieves 0.013 GFLOP/s at n=2048, d=256 (404 seconds). The bottleneck is wireless communication, not computation. Each round requires distributing tile data (at 250 kbps) and collecting results. Softmax requires 3 global sync rounds over wireless, adding significant latency. This converts a compute-bound problem into a communication-bound one.

---

### 2.12 P12: IoT Edge Computing (ARM Cortex-M4F)

#### Real-World Context
The ARM Cortex-M4F (e.g., STM32F4 at 168 MHz) is the workhorse of IoT edge computing. It has a single-precision FPU (VMLA.F32: 3 cycles per MAC), 256 KB SRAM, and 1 MB flash. The question: "Can you run attention inference on a microcontroller?"

This models TinyML deployment of attention mechanisms on resource-constrained edge devices.

#### Our Model
- **Clock**: 168 MHz (STM32F4-typical)
- **FP32 MAC**: 3 cycles (VMLA.F32 instruction)
- **SRAM**: 256 KB → max tile ~146×146 before flash tiling needed
- **Flash**: 1 MB (slower access when SRAM tiling overflows)
- **exp() LUT**: 10 cycles (lookup table approximation)
- **Division**: 14 cycles (iterative VDIV.F32)

#### Implementation (`paradigms/p12_iot_edge/`)
- `edge_device.py`: CortexM4Device with cycle-accurate model, SRAM tiling, flash stall penalties
- `paradigm.py`: IoTEdgeParadigm

#### Key Insight
IoT Edge achieves 0.11 GFLOP/s at n=2048, d=256 (49 seconds). This is below GaAs RISC-V (0.15 GFLOP/s, 35 seconds) despite having a hardware FPU — because the M4F clock (168 MHz) is much lower than GaAs (1 GHz). The SRAM tiling overhead is significant: at n=2048, matrices exceed the 146×146 tile limit and require flash-to-SRAM swapping with stall penalties.

---

## 3. Experiment Setup

### 3.1 Configuration

| Parameter | Value |
|-----------|-------|
| Sequence lengths | 128, 256, 512, 1024, 2048 |
| Embedding dimensions | 64, 128, 256 |
| Warmup runs | 2 |
| Benchmark runs | 5 (best taken) |
| Precision | FP32 (+ FP16 for GPU) |
| Measured implementations | 8 (CPU, GPU FP32 and GPU FP16, each decomposed and fused SDPA; TPU decomposed and fused) |
| Modelled implementations | 9 (Dataflow, Optical, GaAs, minimal-ISA, WSN, IoT Edge; Quantum, Chemical and Biological at n=32, d=16 only) |
| Total | 213 rows (14 implementations × 15 configurations + 3 capped models) |

### 3.2 Timing Model

Every paradigm reports two times:
- **wall_time_seconds**: actual execution time on the host machine (Python/NumPy/CUDA)
- **estimated_hw_time_seconds**: modeled time on the reference hardware

The `effective_time` property selects the best available (hw estimate for simulations, wall time for real hardware).

### 3.3 Roofline Model

For CPU and GPU, execution time per step is:
```
time = max(FLOPs / peak_compute, bytes / peak_bandwidth) + overhead
```

This captures whether a step is **compute-bound** (limited by arithmetic throughput) or **memory-bound** (limited by data movement).

---

## 4. Results

### 4.1 Performance Ranking (n=2048, d=256)

| Paradigm | Precision | Effective Time | Throughput | Source |
|----------|-----------|---------------|------------|--------|
| TPU Fused (one jit) | FP32 (6-pass bf16) | 0.342 ms | 15,771.71 GFLOP/s | measured |
| GPU FP16 Fused SDPA | FP16 | 0.575 ms | 9,368.01 GFLOP/s | measured |
| GPU FP16 Decomposed | FP16 | 0.592 ms | 9,102.71 GFLOP/s | measured |
| TPU Decomposed (five jits) | FP32 (6-pass bf16) | 1.098 ms | 4,908.19 GFLOP/s | measured |
| GPU FP32 Decomposed | FP32 | 1.347 ms | 4,001.98 GFLOP/s | measured |
| GPU FP32 Fused SDPA | FP32 | 2.289 ms | 2,354.38 GFLOP/s | measured |
| Dataflow (Maxeler-style) | – | 5.514 ms | 977.43 GFLOP/s | cycle model |
| Optical (MZI mesh) | – | 10.57 ms | 509.82 GFLOP/s | latency model |
| CPU Fused SDPA (PyTorch) | FP32 | 75.7 ms | 71.24 GFLOP/s | measured |
| CPU Decomposed (NumPy) | FP32 | 140.9 ms | 38.24 GFLOP/s | measured |
| GaAs RISC-V | – | 35.1 s | 0.15 GFLOP/s | instruction cost model |
| IoT Edge (ARM M4F) | – | 49.2 s | 0.11 GFLOP/s | MCU cycle model |
| WSN (64 nodes) | – | 404 s | 0.013 GFLOP/s | network latency model |
| Minimal ISA (`cdte_turing`) | – | 7,554 s | 0.0007 GFLOP/s | operation cost model |

*Capped paradigms (n=32, d=16): Quantum 0.00064, Chemical 0.00023, Biological 1.2×10⁻⁸ GFLOP/s — all cost models*

> **Provenance.** `combined_results.csv` stamps every row with a
> `measurement_type` and a `precision`. The CPU and GPU rows come from one Colab
> T4 session with float32 and float16 inputs; the TPU rows from a Colab TPU
> session. `build_results_table.py` performs the merge and never lets a
> modelled fallback displace a measurement.

### 4.2 Scaling Behavior

Attention is O(n²·d), but the total FLOP count also carries 8·n·d² terms that
grow only linearly in n. Going from n=128 to n=1024 at d=128, the work grows
**26.8x**, not the 64x that n² alone would suggest. Time ratios against that:

| From n=128 to n=1024 (d=128) | Time ratio | Source |
|------|-----------|--------|
| CPU (NumPy) | 38.3x | measured |
| GPU FP32 | 1.45x | measured |
| GPU FP16 | 0.93x | measured |
| TPU fused | 1.50x | measured |
| Dataflow | 23.5x | model |
| GaAs | 26.8x | model |
| Minimal ISA | 26.8x | model |

The cost models track the work almost exactly, because they have no cache and
no parallelism, so time is proportional to operations by construction.

The measured hardware does not. The CPU grows 38.3x, ahead of the work, as the
n×n score matrix outgrows cache and the single-threaded softmax grows with n².
The accelerators barely grow at all over this range, and the FP16 GPU time even
falls slightly (within session noise), because at batch size one both are
dominated by fixed per-operation costs — kernel launch, dispatch and host
synchronisation.

### 4.3 Per-Step Analysis

At n=1024, d=128 on CPU (measured, NumPy, FP32):
- **QKV projection**: 1,333 µs (4.9%)
- **QK matmul**: 6,495 µs (23.8%)
- **Softmax**: 12,494 µs (45.8%)
- **AV matmul**: 6,113 µs (22.4%)
- **Output projection**: 872 µs (3.2%)

Softmax is 0.4% of the FLOPs and **46% of the time** — the single largest line
item. NumPy computes it as five separate single-threaded passes over a matrix
that does not fit in cache, while the matrix products use a multithreaded BLAS.

On the TPU at the same size (measured, five separate jits), every stage takes
137–197 µs whatever its FLOPs. The per-stage TPU numbers measure dispatch and
host synchronisation, not the arithmetic of each stage.

### 4.4 Arithmetic Intensity Analysis

| Paradigm | Arith. Intensity (FLOPs/Byte) | Throughput (GFLOP/s) | Source |
|----------|------------------------------|---------------------|--------|
| TPU Fused | 57.75 | 15,772 | measured |
| GPU FP16 Fused | 115.51 | 9,368 | measured |
| GPU FP16 | 115.51 | 9,103 | measured |
| TPU Decomp | 57.75 | 4,908 | measured |
| GPU FP32 | 57.75 | 4,002 | measured |
| GPU FP32 Fused | 57.75 | 2,354 | measured |
| Dataflow | 57.75 | 977 | model |
| Optical | 57.75 | 510 | model |
| CPU Fused | 57.75 | 71 | measured |
| CPU | 57.75 | 38 | measured |
| GaAs | 57.75 | 0.15 | model |
| IoT Edge | 57.75 | 0.11 | model |
| WSN | 57.75 | 0.013 | model |
| Minimal ISA | 57.75 | 0.0007 | model |

The same computation moves the same data in every row (FP16 rows move half the
bytes, so their intensity doubles), yet throughput differs by orders of
magnitude. **The algorithm is fixed; the architecture and the software path
determine performance.**

---

## 4.5 What these numbers do and do not support

Five limits are worth stating before the insights section leans on any of this.

**Measured against modelled.** Eight implementations are timed on hardware and
nine are calculated. `combined_results.csv` marks every row, and no comparison
in this document draws a conclusion across that line.

**Precision.** Under NumPy 2, a float32 array multiplied by a NumPy float64
scalar becomes float64, so an input generator that scales by `1 / np.sqrt(d)`
silently produces float64 data. Every path that does not cast then runs in
FP64: on the T4, decomposed attention runs at 199 GFLOP/s in FP64 against 4,002
in FP32, and SDPA falls to its unfused math path at 104 GFLOP/s, because the
fused kernels accept only FP16 and FP32. The generator casts the scale to
float32, and every row records its precision.

**Single-head against multi-head.** The decomposed path (`reference_attention`)
computes single-head attention at the full embedding width. The fused path
(`run_fused_sdpa`) reshapes into 8 heads before calling SDPA. The FLOP counts
match apart from softmax, which is eight times larger in the fused path, so
throughput is comparable, but the two do not produce the same tensor.

**Arithmetic intensity of fused rows.** Fused implementations report the
*decomposed* memory traffic, because that is what the analytical model in
`core/attention.py` computes. Their true traffic is lower -- not materialising
the n×n score matrix is precisely what fusion buys. So the arithmetic
intensity shown for a fused row is a lower bound.

**Cost models are declared, not fitted.** The nine modelled paradigms take
clock rates, latencies and per-operation costs from `config.py` and published
device characteristics. Nothing in them is calibrated against a measurement of
the substrate they describe, and checked against the literature every one is
optimistic -- GaAs processors of the period were designed for ~200 MHz rather
than 1 GHz, built photonic engines run at ~1 GHz with weight updates at ~10
MHz, error-corrected quantum operations take microseconds rather than 50 ns,
and demonstrated DNA circuits take hours per layer rather than seconds. Every
correction makes the substrate slower, so the ordering is robust and the
absolute numbers are bounds. The roofline model is the one case where a
prediction can be checked: it puts GPU FP32 decomposed at 6,592 GFLOP/s
against a measured 4,002, an upper bound as a roofline should be.

---

## 5. Key Insights

### 5.1 Architecture-Algorithm Interaction

1. **Attention is matmul-dominated**: 85%+ of FLOPs are in matrix multiplications. Architectures optimized for matmul (TPU, GPU) excel.

2. **Softmax is the architectural discriminator**: It's memory-bound (low arithmetic intensity) and non-trivially parallelizable (requires row-wise reductions). TPU handles it on a separate vector unit; GPU uses shared memory reductions; CPU benefits from cache locality.

3. **Parallelism is non-negotiable**: the scalar GaAs model at 1 GHz reaches 0.15 GFLOP/s, while the measured two-vCPU CPU reaches 38 GFLOP/s and the measured TPU 15,772 GFLOP/s. The first is a model and the others are measurements, so read this as an order-of-magnitude argument.

4. **Unconventional computing has fundamental limits for dense LA**: Optical, quantum, chemical, and biological paradigms are optimized for different problem classes. Dense matrix multiplication — the core of attention — is their worst case. Even optical computing (which uses light-speed propagation) is bottlenecked, in the model, by the electronic unit that computes the nonlinearity.

5. **Communication dominates at the edge**: Distributing attention across constrained nodes (WSN) converts a compute-bound problem into a communication-bound one. The wireless link (250 kbps) becomes the bottleneck, not the MCU compute capacity.

### 5.2 The Transistor Budget Story

| Paradigm | Transistors (order) | Multiply Cost | Lesson |
|----------|-------------------|---------------|--------|
| Biological DNA | ~0 (molecular) | 60 seconds | Molecular timescales dominate |
| Chemical CRN | ~0 (molecular) | 16 ms (K=16) | Reaction kinetics limit speed |
| Quantum | ~10K (qubits) | 90 µs (1792 gates) | Gate overhead negates parallelism |
| Minimal ISA (`cdte_turing`) | ~100s | O(a×b) ops | Hardware multiplier = transformative |
| GaAs | ~10K | 3 cycles | Specialization matters |
| IoT M4F | ~100M | 3 cycles (+tiling) | Memory constraints force overhead |
| CPU | ~1B | 1 cycle (+ SIMD×8) | Parallelism within core |
| GPU | ~10B | 1 cycle (× 2560 cores) | Massive replication |
| TPU | ~10B | 1 cycle (× 16384 PEs) | Fixed-function = efficiency |
| Dataflow | FPGA | 1 cycle (× pipes) | Spatial computing = no fetch |
| Optical | photonic | ~0.1 ns (light) | Speed of light + electronic nonlinearity |

### 5.3 Why This Matters for AI Hardware

The attention mechanism's quadratic scaling (O(n²)) in sequence length means:
- At n=4096 (GPT-4 context): the score matrix is 64MB in FP32
- At n=100K (modern LLMs): 40GB just for attention weights
- **Memory bandwidth becomes the bottleneck** — exactly what TPU and Dataflow are designed to address

---

## 6. How to Run

```bash
# Full benchmark (produces all results)
python run_all.py

# Results location
results/
├── combined_results.csv         # canonical table (build_results_table.py)
├── results_table.md / .tex      # the same table, rendered
├── fig1_ ... fig8_*.png         # figures (generate_figures.py)
├── SICAAI-2026-report.pdf       # report (generate_report.py)
├── raw/                         # raw exports of every session
└── sweep/                       # local sweeps (not tracked)
```

### 6.1 Running on Google Colab (GPU)

```python
# Select GPU runtime: Runtime → Change runtime type → T4 GPU
!git clone <your-repo>
%cd attention_mechanism_acceleration
!pip install -r requirements.txt
!python run_all.py
# GPU paradigm will report execution_mode: "real_cuda"
```

### 6.2 Running on Google Colab (TPU)

```python
# Select TPU runtime: Runtime → Change runtime type → TPU
!pip install torch-xla
!git clone <your-repo>
%cd attention_mechanism_acceleration
!pip install -r requirements.txt
!python run_all.py
# TPU paradigm will report execution_mode: "real_tpu"
```

---

## 7. Extending the Framework

All 12 computational paradigms (17 configurations including fused/precision variants) are implemented. Adding a new paradigm requires:
1. Create `paradigms/pNN_name/` with `__init__.py` and `paradigm.py`
2. Subclass `BaseParadigm`, decorate with `@register`
3. Implement `run_attention(X, W_q, W_k, W_v, W_o) → ParadigmResult`
4. Add import to `paradigms/__init__.py`

The registry pattern auto-discovers new paradigms at startup.

---

## 8. References

1. Vaswani, A. et al. (2017). "Attention Is All You Need." NeurIPS.
2. Jouppi, N. et al. (2017). "In-Datacenter Performance Analysis of a Tensor Processing Unit." ISCA.
3. Williams, S. et al. (2009). "Roofline: An Insightful Visual Performance Model." CACM.
4. Pell, O. & Averbukh, V. (2012). "Maximum Performance Computing with Dataflow Engines." Computing in Science & Engineering.
5. Weste, N. & Harris, D. (2010). "CMOS VLSI Design: A Circuits and Systems Perspective."
6. Shen, Y. et al. (2017). "Deep learning with coherent nanophotonic circuits." Nature Photonics.
7. Qian, L. & Winfree, E. (2011). "Scaling Up Digital Circuit Computation with DNA Strand Displacement Cascades." Science.
8. Seelig, G. et al. (2006). "Enzyme-Free Nucleic Acid Logic Circuits." Science.
9. Gidney, C. & Ekera, M. (2021). "How to factor 2048 bit RSA integers in 8 hours using 20 million noisy qubits." Quantum.
10. Hjelmfelt, A. et al. (1991). "Chemical implementation of neural networks and Turing machines." PNAS.
