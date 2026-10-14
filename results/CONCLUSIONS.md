# Findings

Attention executed on 12 execution substrates in 17 implementations, spanning the
control-flow, data-flow, diffusion-flow and energy-flow computing paradigms.

**Setup.**
- Sizes: sequence lengths 128–2048, embedding dimensions 64–256, batch size one,
  one layer.
- Measured implementations: eight.
  - The CPU and GPU paths ran in one Google Colab session: an NVIDIA T4 (sm_75)
    and its host CPU (two Xeon vCPUs), PyTorch 2.11.0+cu130, NumPy 2.1.3, with
    float32 and float16 inputs.
  - The TPU paths ran in a Colab TPU session through JAX, with
    `jax_default_matmul_precision='float32'` (six bfloat16 passes per product).
- Modelled implementations: nine analytical cost models.
- Timing: GPU timings use CUDA events; CPU and TPU timings use wall clock, with
  `block_until_ready()` on the TPU. Each figure is the minimum of five
  repetitions after warm-up.

**Reading the numbers.** Every row in
[`combined_results.csv`](combined_results.csv) carries a `measurement_type` and
a `precision`.

- A `measured` row is a stopwatch reading.
- A `roofline_model`, `cycle_model` or similar row is a calculation.
- The two are never compared against each other below.

Throughput is quoted in GFLOP/s (a rate) and work in GFLOP (a count). At n=2048,
d=256 the computation is 5.39 GFLOP.

---

## 1. Architecture spans many orders of magnitude

Among measured implementations, the same 5.39 GFLOP runs at 38 GFLOP/s on the
CPU (NumPy) and 15,772 GFLOP/s on the TPU with XLA fusion, a factor of about 410.
The cost models extend the range by roughly seven more orders of magnitude at
the same size. The minimal four-instruction machine lands at 7×10⁻⁴ GFLOP/s.
The range extends further still for the DNA model, which only runs at n=32.

The algorithm, the FLOP count and the mathematical result are identical
throughout. The spread comes from architecture (parallelism, specialisation and
how far data has to move) and, among the measured rows, from the software path
(Section 3).

## 2. Softmax costs far more than its FLOPs

Softmax is **0.39%** of the FLOPs at n=2048, d=256 (21M of 5.39G). Its share of
measured runtime:

| implementation | softmax time | share of runtime |
|---|---:|---:|
| CPU, NumPy, FP32 | 49,421 µs | 35% |
| TPU, five jits | 226 µs | 21% (dispatch-bound, see below) |
| GPU FP16 | 121 µs | 20% |
| GPU FP32 | 173 µs | 13% |

On the CPU it reaches 46% at n=1024, d=128 and 59% at n=2048, d=64, where it
is 1.8% of the FLOPs.

- **CPU.** NumPy computes softmax as five separate elementwise passes (max,
  subtract, exp, sum, divide), each materialising an n×n temporary on a single
  thread. The matrix products call a multithreaded BLAS. The share reflects the
  implementation as much as the processor.
- **GPU.** Reading and writing the 2048×2048 matrix once at the T4's nominal
  bandwidth takes 105 µs in FP32 and 52 µs in FP16. Softmax takes 1.6× and 2.3×
  that, so it is memory-bound in both precisions. Its share rises in FP16
  because Tensor Cores shrink the matrix products around it.
- **TPU.** The per-stage timings are not a softmax measurement. Every stage
  takes 157–252 µs whatever its FLOPs (107–151 µs at n=128). That is dispatch and
  host synchronisation, so a 21% share is close to one fifth by construction.

In the cost models the softmax share follows from the assumptions:

| model | share of runtime |
|---|---:|
| Optical (photonic MZI) | 99.4% |
| Dataflow (Maxeler-style) | 38.1% |
| IoT edge (ARM M4F) | 1.4% |
| GaAs RISC-V, minimal-ISA machine | 0.5% |

The scalar models track operation counts, so they show softmax at its
operation share.

## 3. At batch size one, the software path matters as much as the device

| hardware | path A | path B | ratio |
|---|---:|---:|---|
| TPU | one jit 15,772 | five jits 4,908 | one jit **3.2× faster** |
| GPU FP32 | decomposed 4,002 | SDPA 2,354 | decomposed **1.7× faster** |
| GPU FP16 | decomposed 9,103 | SDPA 9,368 | within 3% |
| CPU | NumPy 38.2 | PyTorch SDPA 71.2 | SDPA 1.9× faster |

- **TPU.** At n=128, d=64, where data movement is negligible, the single jit is
  4.2× faster. The gain is mostly fewer dispatches and host synchronisations,
  with intermediate results kept on-chip.
- **GPU.** PyTorch's FlashAttention backend needs sm_80 or newer, so on the T4
  SDPA uses the memory-efficient kernel in both precisions. With batch × heads =
  8 and a head width of 32, that kernel has too little parallel work to beat
  plain cuBLAS products.
- **CPU.** The CPU comparison mixes libraries: a multithreaded PyTorch kernel
  against single-threaded NumPy elementwise passes. The gain cannot be attributed
  to fusion alone.

Repeated Colab sessions differ by up to about 24% (FP16 decomposed: 9,103–11,327
GFLOP/s), so the FP16 row is a tie.

## 4. Precision, and how easily it changes silently

FP16 is **2.3×** faster than FP32 for the decomposed path and **4.0×** through
SDPA. That is far below the 8× ratio of the T4's nominal peaks (65 against 8.1
TFLOP/s), because fixed per-operation costs and softmax take much of the time at
these sizes. FP32 decomposed attention reaches 49% of the nominal FP32 peak, and
FP16 14% of the FP16 peak.

Precision is also the easiest thing to change without noticing:

- Under NumPy 2 (NEP 50), a float32 array multiplied by a NumPy float64 scalar
  becomes float64.
- An input generator that scales by `1 / np.sqrt(d)` therefore produces float64
  data. Every PyTorch path that does not cast explicitly then runs in FP64.
- On the T4 that path runs at **199 GFLOP/s** for decomposed attention, against
  4,002 in FP32. The T4's FP64 rate is 1/32 of its FP32 rate.
- SDPA falls to its unfused math path at **104 GFLOP/s**, because the
  memory-efficient kernel accepts only FP16 and FP32.

PyTorch emits a warning to that effect only when a fused backend is forced. The
input generator here casts the scale to float32, and the results table records
the precision of every row.

## 5. Accelerators absorb quadratic growth at these sizes

From n=128 to n=2048 at d=256, FLOPs grow **64.2×**. Execution time does not
follow:

| | time growth | |
|---|---:|---|
| CPU, NumPy | 81.0× | measured |
| GPU FP32 | 4.4× | measured |
| TPU fused | 2.6× | measured |
| GPU FP16 | 1.9× | measured |

The CPU grows faster than the work does. The accelerators grow far slower,
because at batch size one they are dominated by fixed costs (kernel launch,
dispatch, synchronisation): at n=128, d=64 each decomposed GPU stage takes
33–98 µs whatever its size. The added arithmetic is absorbed by parallelism
that was already idle.

This describes this size range, not asymptotic behaviour. The algorithm is
O(n²d) everywhere.

## 6. Larger embeddings use the hardware better

At n=2048, raising d from 64 to 256:

| | d=64 | d=256 | gain |
|---|---:|---:|---|
| GPU FP16 | 2,510 | 9,103 | 3.63× |
| TPU fused | 4,855 | 15,772 | 3.25× |
| CPU, NumPy | 14.2 | 38.2 | 2.70× |
| GPU FP32 | 1,535 | 4,002 | 2.61× |

Wider matrices fill matrix units and warp schedulers more completely.

## 7. In the models, a hardware multiplier is worth two orders of magnitude

The GaAs RISC-V and the minimal-ISA machine (`cdte_turing`) models are both
scalar, sequential and cacheless. The single architectural difference is a
hardware multiply instruction, which is worth about **215×** in the models.
On a machine restricted to increment and add, one multiply costs about 25
primitive operations, with operands quantised to [0, 10]. At realistic
precisions it costs far more.

## 8. Unconventional computing meets a dense-matrix wall

All four figures below are cost models, not measurements, and the last three run
only at n=32, d=16. Every parameter was checked against the literature and every
one is optimistic, so these are optimistic estimates, not predictions.

- **Optical (photonic MZI): 510 GFLOP/s.**
  - The optical matrix products take 66 µs in the model.
  - Softmax, on a scalar 2 GHz electronic unit, takes 10.5 ms; the per-row
    optical–electronic conversion is only 20 µs of that.
  - The model also omits reprogramming the mesh for K and V, which are
    activations rather than fixed weights.
- **Quantum (gate-based): 6.4×10⁻⁴ GFLOP/s.**
  - Each 16-bit multiply needs 1,792 T-gates, at 50 ns each.
  - 50 ns is a physical gate time. Error-corrected logical operations are
    expected to take microseconds, so the real figure would be far lower.
- **Chemical (reaction network): 2.3×10⁻⁴ GFLOP/s.**
  - The model assumes 1 ms per bimolecular reaction. Programmable, DNA-based
    reaction networks take seconds to hours per reaction.
  - The exponential, and therefore softmax, lies outside the class of functions
    such networks compute independently of their rate constants.
- **Biological (DNA strand displacement): 1.2×10⁻⁸ GFLOP/s.**
  - The model assumes 10 s per cascade layer. That is one time constant of a
    single ideal reaction; demonstrated multilayer circuits take about two
    hours per layer.

These substrates are not bad at computing. They are bad at *this*: dense linear
algebra followed by a nonlinearity.

## 9. At the edge, the bottleneck stops being arithmetic

- **IoT edge (ARM Cortex-M4F): 0.11 GFLOP/s, model.**
  - The figure is the issue bound of one FP32 multiply-accumulate every three
    cycles at 168 MHz (`VMLA.F32`), with operand loads not counted.
  - Measured int8 attention on Cortex-M4 parts reaches about half that rate per
    cycle.
- **WSN (64 wireless nodes): 0.013 GFLOP/s, model.**
  - IEEE 802.15.4 at 250 kb/s dominates: communication is 97% of the modelled
    404 s.
  - The model streams 29×29 output tiles to the nodes but does not enforce node
    memory for the input strips or softmax rows, does not model the coordinator
    that holds intermediate matrices, and treats a round's transfers as concurrent
    although the nodes share one channel, all of which make it optimistic.
  - Softmax needs three global synchronisation rounds.

Distributing attention across constrained nodes converts a compute-bound problem
into a communication-bound one.

---

## Reference table, n=2048, d=256

| implementation | precision | GFLOP/s | type |
|---|---|---:|---|
| TPU, fused (one jit) | FP32 (6-pass bf16) | 15,771.71 | measured |
| GPU, fused SDPA | FP16 | 9,368.01 | measured |
| GPU, decomposed | FP16 | 9,102.71 | measured |
| TPU, decomposed (five jits) | FP32 (6-pass bf16) | 4,908.19 | measured |
| GPU, decomposed | FP32 | 4,001.98 | measured |
| GPU, fused SDPA | FP32 | 2,354.38 | measured |
| Dataflow (Maxeler-style) | – | 977.43 | cycle model |
| Optical (photonic MZI) | – | 509.82 | latency model |
| CPU, fused SDPA (PyTorch) | FP32 | 71.24 | measured |
| CPU, decomposed (NumPy) | FP32 | 38.24 | measured |
| GaAs RISC-V | – | 0.15 | instruction cost model |
| IoT edge (ARM M4F) | – | 0.11 | MCU cycle model |
| WSN, 64 nodes | – | 0.013 | network latency model |
| Minimal-ISA machine (`cdte_turing`) | – | 0.0007 | operation cost model |

Quantum, chemical and biological paradigms run only at n=32, d=16 and are not
on the same axis:

| implementation | GFLOP/s | type |
|---|---:|---|
| Quantum (gate-based) | 0.00064 | gate count model |
| Chemical (reaction network) | 0.00023 | reaction kinetics model |
| Biological (DNA) | 1.2×10⁻⁸ | strand displacement model |

## Reproducibility

`results/raw/` holds the raw export of every session:

- **CPU and GPU rows:** `colab_t4_fp32_2026-10-14.csv` is the primary record.
- **TPU rows:** they come from an earlier Colab TPU session.
- **CPU and GPU rows in the older exports:** they were produced with float64
  inputs and are superseded.

`build_results_table.py` reassembles `combined_results.csv`. It prefers a
hardware measurement over a model for any key that has both, and stamps the
measurement type and precision of every row.
