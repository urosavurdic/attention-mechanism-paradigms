"""Run all benchmarks and generate results."""

import os
import sys

import config
from benchmarks.scaling import run_scaling_experiment
from visualization.plots import generate_all_plots
from visualization.tables import generate_all_tables


def main():
    os.makedirs(config.SWEEP_DIR, exist_ok=True)

    print("=" * 60)
    print("Attention Mechanism Acceleration - Benchmark Suite")
    print("=" * 60)

    from core.registry import get_available_paradigms
    all_paradigms = get_available_paradigms()

    mode = sys.argv[1] if len(sys.argv) > 1 else "all"

    if mode == "real":
        hw_hint = sys.argv[2] if len(sys.argv) > 2 else None
        paradigms = _select_real_paradigms(all_paradigms, hw_hint=hw_hint)
        print(f"\nMode: REAL HARDWARE ONLY ({hw_hint or 'auto-detect'})")
    elif mode == "sim":
        sim_names = {"cpu", "gpu", "tpu_systolic", "dataflow", "gaas_riscv", "cdte_turing",
                     "optical", "chemical", "biological", "quantum", "wsn", "iot_edge"}
        paradigms = [p for p in all_paradigms if p.short_name in sim_names]
        print(f"\nMode: SIMULATIONS ONLY")
    else:
        paradigms = all_paradigms
        print(f"\nMode: ALL PARADIGMS")

    print(f"\nRunning paradigms ({len(paradigms)}):")
    for p in paradigms:
        print(f"  - {p.name} [{p.short_name}]")

    print(f"\nSequence lengths: {config.SEQ_LENS}")
    print(f"Embedding dims: {config.EMBED_DIMS}")
    print(f"Benchmark runs: {config.BENCHMARK_RUNS}")
    print()

    df = run_scaling_experiment(paradigms=paradigms)

    print()
    generate_all_plots(df)
    print()
    generate_all_tables(df)

    print(f"\nSweep output saved to {config.SWEEP_DIR}/")
    print(f"Experiment logs in {config.LOG_DIR}/")
    print("The published tables and figures in results/ are built separately, by")
    print("build_results_table.py and generate_figures.py, and are left untouched.")
    print("=" * 60)


def _select_real_paradigms(all_paradigms, hw_hint=None):
    """Select paradigms based on available real hardware.

    hw_hint: optional 'cpu', 'gpu', or 'tpu' passed from caller
    that already detected hardware (avoids re-detection issues in subprocesses).
    """
    import torch

    if hw_hint == "tpu":
        selected = {"tpu_systolic", "tpu_fused"}
        return [p for p in all_paradigms if p.short_name in selected]

    selected = {"cpu", "cpu_fused"}

    if hw_hint == "gpu" or torch.cuda.is_available():
        selected.update({"gpu", "gpu_fused", "gpu_fp16", "gpu_fused_fp16"})

    if hw_hint is None:
        try:
            import jax
            devices = jax.devices("tpu")
            if len(devices) > 0:
                selected.update({"tpu_systolic", "tpu_fused"})
        except Exception:
            pass

    return [p for p in all_paradigms if p.short_name in selected]


if __name__ == "__main__":
    main()

