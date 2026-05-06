"""Experiment logger - saves full results to timestamped JSON files."""

import json
import os
import platform
import sys
from datetime import datetime, timezone

import config


def _get_environment_info() -> dict:
    """Collect runtime environment details."""
    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
    }

    try:
        import numpy as np
        env["numpy_version"] = np.__version__
    except ImportError:
        pass

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        env["torch_available"] = False

    try:
        import torch_xla
        env["torch_xla_version"] = torch_xla.__version__
    except ImportError:
        pass

    return env


def save_experiment(results: list, config_snapshot: dict = None) -> str:
    """Save full experiment results to a timestamped JSON file.

    Args:
        results: List of ParadigmResult objects.
        config_snapshot: Optional config dict override.

    Returns:
        Path to the saved JSON file.
    """
    os.makedirs(config.LOG_DIR, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"experiment_{timestamp}.json"
    filepath = os.path.join(config.LOG_DIR, filename)

    if config_snapshot is None:
        config_snapshot = {
            "seq_lens": config.SEQ_LENS,
            "embed_dims": config.EMBED_DIMS,
            "num_heads": config.NUM_HEADS,
            "warmup_runs": config.WARMUP_RUNS,
            "benchmark_runs": config.BENCHMARK_RUNS,
            "sim_max_seq_len": config.SIM_MAX_SEQ_LEN,
            "sim_max_embed_dim": config.SIM_MAX_EMBED_DIM,
        }

    experiment = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "environment": _get_environment_info(),
        "config": config_snapshot,
        "results": [r.to_full_dict() for r in results],
    }

    with open(filepath, "w") as f:
        json.dump(experiment, f, indent=2, default=str)

    return filepath


def load_all_experiments() -> list[dict]:
    """Load all experiment logs from the log directory.

    Returns:
        List of experiment dicts, sorted by timestamp (newest first).
    """
    if not os.path.isdir(config.LOG_DIR):
        return []

    experiments = []
    for fname in os.listdir(config.LOG_DIR):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(config.LOG_DIR, fname)
        with open(fpath) as f:
            experiments.append(json.load(f))

    experiments.sort(key=lambda e: e.get("timestamp", ""), reverse=True)
    return experiments
