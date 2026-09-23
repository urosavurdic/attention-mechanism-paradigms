"""Scaling experiments - run sweeps and export CSV + JSON logs."""

import os
import pandas as pd

import config
from benchmarks.runner import run_benchmark
from core.experiment_logger import save_experiment


def run_scaling_experiment(paradigms=None, seq_lens=None, embed_dims=None) -> pd.DataFrame:
    """Run full scaling sweep and save to CSV + JSON log."""
    os.makedirs(config.SWEEP_DIR, exist_ok=True)

    results = run_benchmark(paradigms, seq_lens, embed_dims)
    rows = [r.to_dict() for r in results]
    df = pd.DataFrame(rows)

    csv_path = os.path.join(config.SWEEP_DIR, "scaling_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Results saved to {csv_path}")

    log_path = save_experiment(results)
    print(f"Full experiment log saved to {log_path}")

    return df
