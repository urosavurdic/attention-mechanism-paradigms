# Attention across computational paradigms

The attention mechanism decomposed into its five stages and executed on **12
execution substrates in 17 implementations**, spanning the four computing
paradigms of control flow, data flow, diffusion flow and energy flow. Every
stage's FLOPs, memory traffic and arithmetic intensity are counted.

Eight implementations are **timed on real hardware**: an NVIDIA T4 and its host
CPU on Google Colab, and a Colab TPU through JAX. Nine are **analytical cost
models**: roofline estimates, cycle-accurate simulators, and physical-process
models for substrates that do not exist at this scale. Every row of every table
says which it is, and which precision it ran in. Those two labels are what make
the comparison credible.

## Findings

**Softmax is 0.39% of the FLOPs and 13–35% of the measured runtime** at
n=2048, d=256. On the CPU it reaches 59% at d=64.

- On the CPU, NumPy's softmax is five separate single-threaded passes over the
  n×n matrix, while the matrix products run on a multithreaded BLAS.
- On the T4, softmax is memory-bound in both precisions: 173 µs in FP32 and 121
  µs in FP16. Streaming the matrix once takes 105 µs and 52 µs. Its share
  rises in FP16 because Tensor Cores shrink the matrix products around it.
- On the TPU, the per-stage timings measure dispatch. Every stage takes
  157–252 µs whatever its FLOPs.

**At batch size one, the software path moves throughput as much as the hardware
does.**

- One XLA-compiled function on the TPU is **3.2× faster** than five separately
  dispatched ones. At n=128, where data movement is negligible, it is 4.2×.
- On the T4, PyTorch's `scaled_dot_product_attention` uses its memory-efficient
  kernel in both precisions; FlashAttention needs sm_80 or newer. The kernel is
  **1.7× slower** than five plain ops in FP32 and level with them in FP16.
- FP16 is 2.3× faster than FP32 for the decomposed path and 4.0× through SDPA.
  That is far below the 8× ratio of the T4's nominal peaks, because fixed
  per-operation costs and softmax dominate at these sizes.

**A silent precision change is the easiest way to get a wrong number.** Under
NumPy 2 (NEP 50), a float32 array multiplied by a NumPy float64 scalar becomes
float64. One `* (1 / np.sqrt(d))` in an input generator therefore turns an FP32
benchmark into an FP64 one:

- On the T4, decomposed attention runs at 199 GFLOP/s in FP64 against 4,002 in
  FP32.
- SDPA drops to its unfused math path (104 GFLOP/s), because the fused kernels
  accept only FP16 and FP32.

The generator here casts the scale to float32 for this reason, and the results
table records the precision of every row.

## Throughput at n=2048, d=256

The computation is 5.39 GFLOP of arithmetic in every row.

| Implementation | Precision | GFLOP/s | Source |
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

Quantum, chemical and DNA paradigms are implemented but not in this table.
Their models only run at n=32, d=16, so they do not belong on the same axis.
They land at 6.4×10⁻⁴, 2.3×10⁻⁴ and 1.2×10⁻⁸ GFLOP/s respectively.

The CPU paths ran on the T4 runtime's host (two Xeon vCPUs at 2.0 GHz). On TPU,
`jax_default_matmul_precision='float32'` emulates each FP32 product with six
bfloat16 passes.

Full analysis is in [`results/CONCLUSIONS.md`](results/CONCLUSIONS.md) and
[`results/ANALYSIS.md`](results/ANALYSIS.md).

## What "modelled" means, and what it is worth

A modelled row is a calculation, not a stopwatch reading. The clock rates,
latencies and per-operation costs come from `config.py` and published device
characteristics. None of them is calibrated against a measurement of the
substrate it describes, because for most of these substrates no such device
exists at this scale.

Checked against the literature, every model is optimistic:

- The GaAs model assumes 1 GHz, whereas GaAs microprocessors of the period were
  designed for about 200 MHz.
- The DNA model assumes 10 s per cascade layer, whereas demonstrated circuits
  take about two hours per layer.

Every such correction makes the substrate slower, so the ordering of substrates
is robust and the absolute numbers are bounds, not predictions.

The roofline model is the one case that can be checked against a measurement of
the same thing. It puts GPU FP32 decomposed at 6,592 GFLOP/s against a measured
4,002: an upper bound, as a roofline should be.

Two models need a note:

- **Optical:** softmax runs on a scalar 2 GHz electronic unit, and that unit,
  not the optical–electronic conversion (20 µs of 10.5 ms), sets the time.
- **`cdte_turing`:** a four-instruction counter machine, a thought experiment
  about a tiny transistor budget. CdTe itself is a photovoltaic material, not a
  logic technology.

## Running it

```bash
python -m venv venv && venv/Scripts/activate      # Windows
pip install -r requirements.txt

python run_all.py sim        # simulators only, no accelerator needed
python run_all.py            # everything available on this machine
```

A sweep writes to `results/sweep/` and leaves the published artefacts alone. It
only sees the machine it runs on, so on a laptop it would replace the measured
GPU and TPU rows with that laptop's modelled fallbacks. `build_results_table.py`
guards against that: it prefers a hardware measurement over a model wherever both
exist, and stamps every row with a `measurement_type` and a `precision`.

Then rebuild the derived artefacts:

```bash
python build_results_table.py    # canonical CSV + results tables
python generate_figures.py       # the eight figures
python generate_report.py        # the PDF
```

Experiment parameters, clock rates and roofline constants live in `config.py`.

## Reproducibility

`results/raw/` holds the raw export of every session.

- **CPU and GPU rows:** the primary record is
  `colab_t4_fp32_2026-10-14.csv` (Tesla T4 with a 2-vCPU Xeon host, PyTorch
  2.11.0+cu130, NumPy 2.1.3).
- **TPU rows:** they come from an earlier Colab TPU session.
- **CPU and GPU rows in the older exports:** they were produced with float64
  inputs and are superseded.

Colab sessions differ. The FP16 decomposed path has measured between 9,103 and
11,327 GFLOP/s across sessions, so differences under about 1.3× between GPU
paths are within session-to-session variation.

`colab_runner.ipynb` re-measures the GPU and TPU paths.

## How it is built

- `core/` — attention decomposition, FLOP and memory counting, metrics
  dataclasses, the `BaseParadigm` interface, a registry decorator.
- `paradigms/pNN_*/` — one subpackage per implementation, each returning the
  same `ParadigmResult` shape.
- `benchmarks/`, `visualization/` — sweep runner and plotting.
- `build_results_table.py` — merges the raw exports into the canonical table.

To add a paradigm:

1. Create `paradigms/pNN_name/` and subclass `BaseParadigm`.
2. Decorate it with `@register`.
3. Implement `run_attention(X, W_q, W_k, W_v, W_o) -> ParadigmResult`.
4. Add the import to `paradigms/__init__.py`.

One caveat is worth knowing before reading the code. The decomposed path
computes **single-head** attention at full width, while the fused path reshapes
into **8 heads**. The FLOP counts match apart from softmax, which is eight times
larger in the fused path. Throughput is therefore comparable, but the two paths
do not produce the same tensor.

## Talk

Presented at the Fifth Serbian International Conference on Applied Artificial
Intelligence (SICAAI), Kragujevac, 20–21 May 2026.

---

Uroš Savurdić — School of Electrical Engineering, University of Belgrade.
