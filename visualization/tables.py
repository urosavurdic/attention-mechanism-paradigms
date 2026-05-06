"""Table generation for results - Markdown and LaTeX."""

import os
import pandas as pd

import config


def generate_markdown_table(df: pd.DataFrame) -> str:
    """Generate a summary markdown table."""
    cols = ["paradigm_short", "seq_len", "embed_dim", "total_flops",
            "total_effective_time_s", "throughput_gflops", "arithmetic_intensity"]
    available = [c for c in cols if c in df.columns]
    sub = df[available].copy()

    # Format numbers
    if "total_effective_time_s" in sub.columns:
        sub["total_effective_time_s"] = sub["total_effective_time_s"].map(lambda x: f"{x:.6f}")
    if "throughput_gflops" in sub.columns:
        sub["throughput_gflops"] = sub["throughput_gflops"].map(lambda x: f"{x:.2f}")
    if "arithmetic_intensity" in sub.columns:
        sub["arithmetic_intensity"] = sub["arithmetic_intensity"].map(lambda x: f"{x:.2f}")
    if "total_flops" in sub.columns:
        sub["total_flops"] = sub["total_flops"].map(lambda x: f"{x:,.0f}")

    md = sub.to_markdown(index=False)

    path = os.path.join(config.RESULTS_DIR, "results_table.md")
    with open(path, "w") as f:
        f.write("# Benchmark Results\n\n")
        f.write(md)
    print(f"  Saved {path}")
    return md


def generate_latex_table(df: pd.DataFrame) -> str:
    """Generate a LaTeX table for the paper."""
    # Summary: one row per paradigm at max config
    max_seq = df["seq_len"].max()
    max_dim = df["embed_dim"].max()
    sub = df[(df["seq_len"] == max_seq) & (df["embed_dim"] == max_dim)].copy()

    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Attention execution across paradigms "
        f"(n={max_seq}, d={max_dim})" + r"}",
        r"\begin{tabular}{lrrrrr}",
        r"\hline",
        r"Paradigm & FLOPs & Time (s) & GFLOPS & AI \\",
        r"\hline",
    ]

    for _, row in sub.iterrows():
        name = row["paradigm_short"].replace("_", r"\_")
        flops = f"{row['total_flops']:,.0f}"
        time_s = f"{row['total_effective_time_s']:.6f}"
        gflops = f"{row['throughput_gflops']:.2f}"
        ai = f"{row['arithmetic_intensity']:.2f}"
        lines.append(f"{name} & {flops} & {time_s} & {gflops} & {ai} \\\\")

    lines.extend([
        r"\hline",
        r"\end{tabular}",
        r"\label{tab:paradigm_comparison}",
        r"\end{table}",
    ])

    latex = "\n".join(lines)
    path = os.path.join(config.RESULTS_DIR, "results_table.tex")
    with open(path, "w") as f:
        f.write(latex)
    print(f"  Saved {path}")
    return latex


def generate_all_tables(df: pd.DataFrame):
    """Generate all tables."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    print("Generating tables...")
    generate_markdown_table(df)
    generate_latex_table(df)
