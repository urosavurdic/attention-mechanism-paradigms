# Findings

Attention executed across 12 computational paradigms and 17 implementations.

**Setup.** Sequence lengths 128–2048, embedding dimensions 64–256, 8 attention
heads. Eight implementations are timed on real hardware — Google Colab CPU, an
NVIDIA T4 (sm_75), and a TPU v5e through JAX. Nine are analytical cost models.
GPU timings use CUDA events; TPU timings use `block_until_ready()`.

**Reading the numbers.** Every row in
[`combined_results.csv`](combined_results.csv) carries a `measurement_type`. A
`measured` row is a stopwatch reading. A `roofline_model`, `cycle_model` or
similar row is a calculation. The two are never compared against each other in
what follows, because an earlier version of this document did exactly that and
produced a headline number wrong by a factor of 33.

Throughput is quoted in GFLOP/s (a rate). Work is quoted in GFLOP (a count).
At n=2048, d=256 the computation is 5.39 GFLOP.

---

## 1. Architecture spans twelve orders of magnitude

The same 5.39 GFLOP of arithmetic runs at 1.2×10⁻⁸ GFLOP/s on a DNA strand
displacement model and 15,772 GFLOP/s on a TPU with XLA fusion — a factor of
about 1.3×10¹². Restricting to electronic paradigms only, the CdTe Turing
machine model (0.0007) to the measured TPU (15,772) still spans seven orders.

The algorithm, the FLOP count and the mathematical result are identical
throughout. The spread is entirely architecture: parallelism, specialisation,
and how far data has to move.

## 2. The same PyTorch call is 92× slower in FP32 than FP16

The strongest result here, and the best evidenced.

`F.scaled_dot_product_attention` runs at **104.21 GFLOP/s in FP32** and
**9,632.69 GFLOP/s in FP16** on the same T4, the same shapes, the same call
site. A ratio of **92.4×** from a dtype change.

The cause is kernel dispatch. PyTorch silently selects a backend based on
precision and compute capability: the `math` fallback for FP32 on sm_75, the
memory-efficient kernel for FP16. The backend was logged on every run and is
recorded per row in the `sdpa_backend` column, so this is not inferred from
timing alone.

Nothing in the API surface signals the choice. One line of unchanged code sits
either side of a 92× cliff.

## 3. Fusion is a dispatch decision, not a free win

Comparing measurements against measurements only:

| | decomposed | fused | outcome |
|---|---:|---:|---|
| GPU FP32 (T4) | 199.09 | 104.21 | fused **1.9× slower** |
| GPU FP16 (T4) | 11,326.67 | 9,632.69 | fused **1.2× slower** |
| CPU | 18.14 | 17.75 | within 2%, no meaningful difference |
| TPU (JAX/XLA) | 4,908.19 | 15,771.71 | fused **3.2× faster** |

Fusion helps decisively on the TPU, where XLA eliminates intermediate memory
traffic across operation boundaries. It loses on the T4 in both precisions, and
does nothing measurable on CPU, where SDPA has no specialised backend.

Part of the GPU loss is structural rather than a dispatch failure: the fused
path partitions into 8 heads of dimension 32, while the decomposed path runs
single-head at the full width. Smaller matrices use a warp scheduler less
efficiently. The FLOP counts are identical, so throughput remains comparable,
but the two paths do not compute the same tensor.

**An earlier version of this document claimed 63× here.** That compared the
measured fused result against a *roofline estimate* of the decomposed path — a
model against a stopwatch. The measured ratio is 1.9×.

## 4. Precision changes more than precision

On the T4, moving decomposed attention from FP32 to FP16 is a **56.9×**
speedup — 199.09 to 11,326.67 GFLOP/s — as Tensor Cores become available. The
fused path gains 92.4× from the same change.

These are large enough to dominate every other tuning decision available at
this level. Precision and backend co-design is the lever; loop-level
optimisation is not.

## 5. Softmax is the architectural discriminator

Softmax is **0.39%** of the FLOPs at n=2048, d=256 — 21M of 5.39G. Its share of
runtime, by implementation:

| implementation | softmax share of runtime | |
|---|---:|---|
| Optical (photonic MZI) | 99.4% | model |
| Dataflow (Maxeler-style) | 38.1% | model |
| CPU | 34.1% | measured |
| GPU FP16 | 23.1% | measured |
| TPU systolic | 20.5% | measured |
| GPU FP32 | 7.2% | measured |
| IoT edge (ARM M4F) | 1.4% | model |
| GaAs RISC-V, CdTe Turing | 0.5% | model |

A 0.39% operation taking anywhere from 0.5% to 99% of the time is the sharpest
signal in the dataset. It measures how well an architecture handles the
irregular, non-matmul part of a workload:

- **Scalar processors** (GaAs, CdTe) track FLOPs directly — no cache, no
  parallelism, nothing to reshape the profile. They are the algorithmic control
  group, and they show softmax at its true FLOP share.
- **CPU** pays 34% because the n×n score matrix does not fit in cache.
- **Tensor Core paths** accelerate matmul and not softmax, so softmax's relative
  share *rises* as matmul gets faster. FP16's 23% against FP32's 7% is that
  effect, not a softmax regression.
- **Optical** is the extreme case: the MZI mesh multiplies at the speed of
  light, then has to convert to electronics and back to apply a nonlinearity.
  The interface dominates everything else.

## 6. Accelerators absorb quadratic growth

From n=128 to n=2048 at d=256, FLOPs grow **64.2×** — quadratic in n, since the
n²d terms come to dominate the nd² ones. Execution time does not follow:

| | time growth | |
|---|---:|---|
| CPU | 126.2× | measured |
| TPU fused | 2.6× | measured |
| GPU FP16 | 2.0× | measured |

The CPU grows *faster* than the work does, because cache pressure erodes
throughput as the score matrix grows. The accelerators grow far slower than the
work: at these sizes they are latency-bound rather than throughput-bound, so
most of the added arithmetic is absorbed by parallelism that was already
sitting idle.

This is a statement about this size range, not an asymptotic claim. The
algorithm is O(n²d) everywhere.

*An earlier version stated 341× FLOP growth. The correct figure is 64.2×.*

## 7. Larger embeddings use hardware better

At n=2048, raising d from 64 to 256:

| | d=64 | d=256 | gain |
|---|---:|---:|---|
| GPU FP16 | 2,728 | 11,327 | 4.15× |
| TPU fused | 4,855 | 15,772 | 3.25× |
| CPU | 5.9 | 18.1 | 3.06× |

Wider matrices fill a 128×128 systolic array and a warp scheduler more
completely. At d=64 the MXU is largely idle; at d=256 tiles align. Throughput
rises because utilisation improves alongside the work.

## 8. A hardware multiplier is worth more than any amount of cleverness

| | GFLOP/s | |
|---|---:|---|
| TPU fused | 15,772 | measured |
| GPU FP16 | 11,327 | measured |
| CPU with AVX2 | 18.1 | measured |
| GaAs RISC-V (has an FMUL) | 0.15 | model |
| CdTe Turing (no multiplier) | 0.0007 | model |

GaAs and CdTe are both scalar, sequential and cacheless. The single
architectural difference is a hardware multiply instruction, worth roughly
**214×** — on a Turing machine restricted to increment and add, one multiply
costs about 25 primitive operations. Adding parallelism on top is worth another
~10⁵.

## 9. Unconventional computing meets a dense-matrix wall

Attention is close to a worst case for non-electronic substrates. All four
figures below are cost models, not measurements, and the two smallest run only
at n=32, d=16.

- **Optical (photonic MZI)** — 510 GFLOP/s. Genuinely fast at matmul; the
  optical-electronic-optical conversion for softmax consumes 99% of the time.
  Without the nonlinearity it would be competitive with a GPU.
- **Quantum (gate-based)** — 0.00064 GFLOP/s. Each 16-bit multiply needs 1,792
  T-gates (256 Toffolis × 7) at 50 ns each. Quantum advantage is real for
  search and factoring, not for dense linear algebra.
- **Chemical (reaction network)** — 0.00023 GFLOP/s. Bimolecular reactions at
  1 kHz with 256-well parallelism; each dot product needs K sequential rounds
  plus mixing time.
- **Biological (DNA strand displacement)** — 1.2×10⁻⁸ GFLOP/s. Ten seconds per
  displacement, cascade depth 6 per multiply: roughly 11,400 seconds for one
  attention computation at n=32.

These substrates are not bad at computing. They are bad at *this*, which is
pure dense linear algebra — exactly what they were not designed for.

## 10. At the edge, the bottleneck stops being arithmetic

- **IoT edge (ARM Cortex-M4F)** — 0.11 GFLOP/s, model. Bounded by a 168 MHz
  clock and 256 KB of SRAM: matrices beyond roughly 146×146 must be tiled from
  flash, at 3 cycles per FP32 MAC and 10 for a table-lookup `exp`.
- **WSN (64 wireless nodes)** — 0.013 GFLOP/s, model. IEEE 802.15.4 at 250 kbps
  dominates: tiles are distributed wirelessly, computed on 16 MHz MCUs, then
  collected, and softmax needs 3 global synchronisation rounds. Over 95% of the
  time is communication.

Distributing attention across constrained nodes converts a compute-bound
problem into a communication-bound one. The unit of scarcity changes from FLOPs
to bits per second.

---

## Reference table, n=2048, d=256

| implementation | GFLOP/s | type |
|---|---:|---|
| TPU, fused (JAX/XLA) | 15,771.71 | measured |
| GPU FP16, decomposed | 11,326.67 | measured |
| GPU FP16, fused SDPA | 9,632.69 | measured |
| TPU, decomposed (per-op JIT) | 4,908.19 | measured |
| Dataflow (Maxeler-style) | 977.43 | cycle model |
| Optical (photonic MZI) | 509.82 | latency model |
| GPU FP32, decomposed | 199.09 | measured |
| GPU FP32, fused SDPA | 104.21 | measured |
| CPU, decomposed (AVX2) | 18.14 | measured |
| CPU, fused SDPA | 17.75 | measured |
| GaAs RISC-V | 0.15 | instruction cost model |
| IoT edge (ARM M4F) | 0.11 | MCU cycle model |
| WSN, 64 nodes | 0.013 | network latency model |
| CdTe Turing machine | 0.0007 | operation cost model |

Quantum, chemical and biological paradigms run only at n=32, d=16 and are not
on the same axis:

| implementation | GFLOP/s | type |
|---|---:|---|
| Quantum (gate-based) | 0.00064 | gate count model |
| Chemical (reaction network) | 0.00023 | reaction kinetics model |
| Biological (DNA) | 1.2×10⁻⁸ | strand displacement model |

## Reproducibility

`results/logs/` holds 16 timestamped experiment logs with captured
environments. All 16 are CPU-only runs: the GPU and TPU sessions ran in Colab
and their logs were not retained, so for those eight implementations the CSV
exports in `results/raw/` are the primary record. `colab_runner.ipynb` is the
path to re-measure them.

`build_results_table.py` reassembles `combined_results.csv` from
`results/raw/`, preferring a hardware measurement over a model for any paradigm
that has both.
