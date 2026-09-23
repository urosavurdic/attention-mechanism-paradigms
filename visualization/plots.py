"""Plotting functions for benchmark results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import config

# Consistent style
plt.rcParams.update({
    "figure.figsize": (10, 6),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

COLORS = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336", "#00BCD4",
          "#795548", "#607D8B", "#E91E63", "#3F51B5", "#CDDC39", "#FF5722",
          "#009688", "#FFC107", "#8E24AA", "#D81B60", "#1E88E5"]


def _save(fig, name: str):
    os.makedirs(config.SWEEP_DIR, exist_ok=True)
    path = os.path.join(config.SWEEP_DIR, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_paradigm_comparison(df: pd.DataFrame):
    """Grouped bar chart: wall time by paradigm, grouped by seq_len."""
    embed_dims = sorted(df["embed_dim"].unique())
    ncols = min(len(embed_dims), 3)
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6), squeeze=False)
    axes = axes[0]  # flatten to 1D

    for idx, embed_dim in enumerate(embed_dims[:ncols]):
        ax = axes[idx]
        sub = df[df["embed_dim"] == embed_dim]

        paradigms = sub["paradigm_short"].unique()
        seq_lens = sorted(sub["seq_len"].unique())
        x = np.arange(len(paradigms))
        width = 0.8 / len(seq_lens)

        for i, sl in enumerate(seq_lens):
            vals = []
            for p in paradigms:
                row = sub[(sub["paradigm_short"] == p) & (sub["seq_len"] == sl)]
                vals.append(row["total_effective_time_s"].values[0] if len(row) > 0 else 0)
            ax.bar(x + i * width, vals, width, label=f"n={sl}", color=COLORS[i % len(COLORS)])

        ax.set_xlabel("Paradigm")
        ax.set_ylabel("Estimated Execution Time (s)")
        ax.set_title(f"Paradigm Comparison (d={embed_dim})")
        ax.set_xticks(x + width * (len(seq_lens) - 1) / 2)
        ax.set_xticklabels(paradigms, rotation=45, ha="right")
        ax.legend()
        ax.set_yscale("log")

    fig.suptitle("Attention Execution Time by Paradigm\n(real HW time for CPU/GPU, estimated HW time for simulators)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, "paradigm_comparison")


def plot_scaling_curves(df: pd.DataFrame):
    """Line plot: wall time vs seq_len per paradigm."""
    embed_dims = sorted(df["embed_dim"].unique())
    fig, axes = plt.subplots(1, len(embed_dims), figsize=(7 * len(embed_dims), 6))
    if len(embed_dims) == 1:
        axes = [axes]

    for ax, embed_dim in zip(axes, embed_dims):
        sub = df[df["embed_dim"] == embed_dim]
        for i, p in enumerate(sub["paradigm_short"].unique()):
            p_data = sub[sub["paradigm_short"] == p].sort_values("seq_len")
            ax.plot(p_data["seq_len"], p_data["total_effective_time_s"],
                    "o-", label=p, color=COLORS[i % len(COLORS)], linewidth=2)

        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Estimated Execution Time (s)")
        ax.set_title(f"Scaling Behavior (d={embed_dim})")
        ax.legend()
        ax.set_yscale("log")

    fig.suptitle("Attention Scaling by Sequence Length", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "scaling_curves")


def plot_step_breakdown(df: pd.DataFrame):
    """Stacked bar chart: time per attention step."""
    step_cols = [c for c in df.columns if c.startswith("time_")]
    if not step_cols:
        return

    # Use largest config for the breakdown
    max_seq = df["seq_len"].max()
    max_dim = df["embed_dim"].max()
    sub = df[(df["seq_len"] == max_seq) & (df["embed_dim"] == max_dim)]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    paradigms = sub["paradigm_short"].values
    x = np.arange(len(paradigms))
    bottom = np.zeros(len(paradigms))

    for i, col in enumerate(step_cols):
        step_name = col.replace("time_", "")
        vals = sub[col].values
        ax.bar(x, vals, bottom=bottom, label=step_name,
               color=COLORS[i % len(COLORS)])
        bottom += vals

    ax.set_xlabel("Paradigm")
    ax.set_ylabel("Estimated Execution Time (s)")
    ax.set_title(f"Step Breakdown (n={max_seq}, d={max_dim})")
    ax.set_xticks(x)
    ax.set_xticklabels(paradigms, rotation=45, ha="right")
    ax.legend(loc="upper left")

    fig.tight_layout()
    _save(fig, "step_breakdown")


def plot_arithmetic_intensity(df: pd.DataFrame):
    """Scatter: arithmetic intensity vs throughput (roofline-style)."""
    max_seq = df["seq_len"].max()
    max_dim = df["embed_dim"].max()
    sub = df[(df["seq_len"] == max_seq) & (df["embed_dim"] == max_dim)]

    if sub.empty:
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, (_, row) in enumerate(sub.iterrows()):
        ai = row.get("arithmetic_intensity", 0)
        tp = row.get("throughput_gflops", 0)
        if ai > 0 and tp > 0:
            ax.scatter(ai, tp, s=120, color=COLORS[i % len(COLORS)],
                       label=row["paradigm_short"], zorder=5)

    ax.set_xlabel("Arithmetic Intensity (FLOPs/Byte)")
    ax.set_ylabel("Throughput (GFLOPS)")
    ax.set_title(f"Arithmetic Intensity vs Throughput (n={max_seq}, d={max_dim})")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    _save(fig, "arithmetic_intensity")


def generate_all_plots(df: pd.DataFrame):
    """Generate all plots from results DataFrame."""
    os.makedirs(config.SWEEP_DIR, exist_ok=True)
    print("Generating plots...")
    plot_paradigm_comparison(df)
    plot_scaling_curves(df)
    plot_step_breakdown(df)
    plot_arithmetic_intensity(df)
    plot_fused_vs_decomposed(df)


def plot_fused_vs_decomposed(df: pd.DataFrame):
    """Bar chart comparing fused SDPA vs decomposed attention for real hardware."""
    pairs = [
        ("cpu", "cpu_fused", "CPU"),
        ("gpu", "gpu_fused", "GPU"),
        ("tpu_systolic", "tpu_fused", "TPU"),
    ]

    max_seq = df["seq_len"].max()
    max_dim = df["embed_dim"].max()
    subset = df[(df["seq_len"] == max_seq) & (df["embed_dim"] == max_dim)]

    available_pairs = []
    for decomp, fused, label in pairs:
        d_row = subset[subset["paradigm_short"] == decomp]
        f_row = subset[subset["paradigm_short"] == fused]
        if not d_row.empty and not f_row.empty:
            available_pairs.append((
                label,
                d_row["total_effective_time_s"].values[0],
                f_row["total_effective_time_s"].values[0],
            ))

    if not available_pairs:
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(available_pairs))
    width = 0.35

    labels = [p[0] for p in available_pairs]
    decomp_times = [p[1] * 1000 for p in available_pairs]
    fused_times = [p[2] * 1000 for p in available_pairs]

    bars1 = ax.bar(x - width / 2, decomp_times, width, label="Decomposed", color=COLORS[0])
    bars2 = ax.bar(x + width / 2, fused_times, width, label="Fused SDPA", color=COLORS[1])

    ax.set_xlabel("Hardware")
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"Decomposed vs Fused Attention (n={max_seq}, d={max_dim})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    ax.set_yscale("log")

    for bar in bars1 + bars2:
        h = bar.get_height()
        ax.annotate(f"{h:.3f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    fig.tight_layout()
    _save(fig, "fused_vs_decomposed")
