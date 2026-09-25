# Attention across computational paradigms

The attention mechanism decomposed into its five stages and executed across
**12 computational paradigms and 17 implementations**, measuring FLOPs, memory
traffic and arithmetic intensity at every stage.

Eight implementations are **timed on real hardware** — Google Colab CPU, an
NVIDIA T4, and a TPU v5e through JAX. Nine are **analytical cost models**:
roofline estimates, cycle-accurate simulators, and physical-process models for
substrates that do not exist at this scale. Every row of every table says which
it is, because that distinction is the whole credibility of the comparison.

## The finding worth your time

**The same PyTorch call is 92× slower in FP32 than in FP16** — 104.21 against
9,632.69 GFLOP/s on the same T4, the same shapes, the same call site.

`F.scaled_dot_product_attention` silently dispatches to a different kernel
depending on precision and compute capability: the `math` fallback for FP32 on
sm_75, the memory-efficient kernel for FP16. Nothing in the API surface tells
you this. The backend was logged on every run and is recorded per row in the
`sdpa_backend` column, so the mechanism is evidenced, not inferred from timing.

Two more results:

- **Operator fusion is not automatically faster.** Comparing measurements
  against measurements only: on the T4 the fused SDPA path *loses* to the
  decomposed one in both precisions — 1.9× in FP32, 1.2× in FP16 — and on CPU
  the two land within 2%. On the TPU, XLA fusion wins by 3.2×. Fusion is a
  dispatch decision, not a free win.
- **Softmax is the architectural discriminator.** It is 0.39% of the FLOPs and
  between 0.5% and 99% of the runtime, depending entirely on the architecture.
  On a scalar processor it costs what its FLOP share says; on a CPU it costs
  34% because the n×n matrix leaves cache; on a photonic mesh it costs
  virtually everything, because a nonlinearity forces conversion out of the
  optical domain and back.

## Throughput at n=2048, d=256

The computation is 5.39 GFLOP of arithmetic in every row.

| Implementation | GFLOP/s | Source |
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

Quantum, chemical and DNA paradigms are implemented but not in this table:
their models only run at n=32, d=16, so they do not belong on the same axis.
They land at 6.4×10⁻⁴, 2.3×10⁻⁴ and 1.2×10⁻⁸ GFLOP/s respectively.

Full analysis in [`results/ANALYSIS.md`](results/ANALYSIS.md) and
[`results/CONCLUSIONS.md`](results/CONCLUSIONS.md).

## What "modelled" means, and what it is worth

A modelled row is a calculation, not a stopwatch. The clock rates, latencies
and per-operation costs come from `config.py` and published device
characteristics; none of them is calibrated against a measurement of the
substrate it describes, because for most of these substrates no such device
exists at this scale.

The roofline model is the one case where the prediction can be checked against
a measurement of the same thing, and it is worth reporting that it does badly:
it puts GPU FP32 decomposed at 6,592 GFLOP/s where the T4 actually delivers
**199.09**, a 33× overestimate. Peak-bandwidth roofline assumes perfect overlap
and ignores kernel launch, dispatch, and the fact that these matrices are small
enough that a T4 never reaches steady state.

So the models are order-of-magnitude arguments about architecture. They are not
predictions of what hardware would do, and this repository does not compare
them against measurements as though they were.

## Running it

```bash
python -m venv venv && venv/Scripts/activate      # Windows
pip install -r requirements.txt

python run_all.py sim        # simulators only, no accelerator needed
python run_all.py            # everything available on this machine
```

A sweep writes to `results/sweep/` and leaves the published artefacts alone. It
only sees the machine it runs on, so on a laptop it would replace measured GPU
and TPU rows with that laptop's modelled fallbacks. `build_results_table.py` guards
against that: it prefers a hardware measurement over a model wherever both exist,
and stamps every row with a `measurement_type`.

Then rebuild the derived artefacts:

```bash
python build_results_table.py    # canonical CSV + results tables
python generate_figures.py       # the eight figures
python generate_report.py        # the PDF
```

Experiment parameters, clock rates and roofline constants live in `config.py`.

## Reproducibility

`results/logs/` holds 16 timestamped experiment logs with captured
environments. **All 16 are CPU-only runs.** The GPU and TPU sessions ran in
Colab and their logs were never retained, so for those eight implementations
the CSV exports in `results/raw/` are the primary record and cannot be
regenerated locally. `colab_runner.ipynb` is the path to re-measure them.

One cross-check is available: `gpu_fused` was measured in two independent Colab
sessions and the two agree to 0.16% — 104.38 and 104.21 GFLOP/s.

## How it is built

- `core/` — attention decomposition, FLOP and memory counting, metrics
  dataclasses, the `BaseParadigm` interface, a registry decorator.
- `paradigms/pNN_*/` — one subpackage per implementation, each returning the
  same `ParadigmResult` shape.
- `benchmarks/`, `visualization/` — sweep runner and plotting.
- `build_results_table.py` — merges the raw exports into the canonical table.

To add a paradigm: create `paradigms/pNN_name/`, subclass `BaseParadigm`,
decorate with `@register`, implement
`run_attention(X, W_q, W_k, W_v, W_o) -> ParadigmResult`, and add the import to
`paradigms/__init__.py`.

One caveat worth knowing before reading the code: the decomposed path computes
**single-head** attention at full width, while the fused path reshapes into
**8 heads**. FLOP counts are identical so throughput is comparable, but the two
do not produce the same tensor, and the fused path's smaller per-head matrices
are part of why it loses on the T4.

## Talk

Presented at the Fifth Serbian International Conference on Applied Artificial
Intelligence (SICAAI), Kragujevac, 20–21 May 2026. No paper was submitted.

---

Uroš Savurdić — School of Electrical Engineering, University of Belgrade.
