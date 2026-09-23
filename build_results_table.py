"""Assemble the canonical results table from the raw run exports.

Why this script exists
----------------------
The raw CSVs in `results/raw/` come from several sessions on different
hardware, and merging them by hand went wrong once already. Two paradigms --
`gpu` (FP32 decomposed) and `tpu_systolic` (decomposed) -- have *both* a real
hardware measurement, from a Colab session with a T4 and a TPU attached, and a
modelled fallback, from a local CPU-only sweep where `torch.cuda.is_available()`
returned False and the paradigm silently took its simulation branch.

A previous merge kept the modelled rows for those two paradigms while keeping
the measured rows for everything else, and the resulting table then described
all of them as measured. A roofline estimate of 6,592 GFLOP/s sat where a
measurement of 199 belonged. Comparisons drawn across that table were
comparing a model against a stopwatch.

So the merge is a script rather than a spreadsheet, and every row carries a
`measurement_type` saying what produced it.

Precedence
----------
For each (paradigm, seq_len, embed_dim) key, prefer a real hardware run over a
modelled one; among rows of equal standing, prefer the newest source file.
Recency alone is the wrong rule here -- the newest file is the one holding the
modelled fallbacks.
"""

import csv
import os
from collections import OrderedDict

HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "results", "raw")
OUT = os.path.join(HERE, "results", "combined_results.csv")

# Oldest first. Later files win ties, but never outrank a real measurement.
SOURCES = [
    "combined_all_paradigms.csv",   # Colab: real T4 + real TPU v5e, 9 paradigms
    "scaling_results.csv",          # local CPU-only sweep, all 12 families
    "combined_all_v6.csv",          # merge of the local sweep + a later Colab session
]

# execution_mode -> (measurement_type, is_real_hardware)
MODE_KINDS = {
    "real_cpu": ("measured", True),
    "real_cpu_fused": ("measured", True),
    "real_cuda": ("measured", True),
    "real_cuda_fused": ("measured", True),
    "real_cuda_fp16": ("measured", True),
    "real_cuda_fused_fp16": ("measured", True),
    "real_tpu_jax": ("measured", True),
    "cpu_fallback": ("measured_on_cpu_fallback", True),
    "roofline_simulation": ("roofline_model", False),
    "simulation": ("cycle_model", False),
    "": ("analytical_model", False),
}

# Paradigms with no execution_mode column are analytical cost models throughout.
ANALYTICAL = {
    "gaas_riscv": "instruction_cost_model",
    "cdte_turing": "operation_cost_model",
    "dataflow": "pipeline_cycle_model",
    "optical": "photonic_latency_model",
    "chemical": "reaction_kinetics_model",
    "biological": "strand_displacement_model",
    "quantum": "gate_count_model",
    "wsn": "network_latency_model",
    "iot_edge": "mcu_cycle_model",
}


def classify(row):
    """Return (measurement_type, is_real_hardware) for one raw row."""
    mode = (row.get("info_execution_mode") or "").strip()
    if mode:
        return MODE_KINDS.get(mode, ("unknown", False))
    short = row["paradigm_short"]
    if short in ANALYTICAL:
        return (ANALYTICAL[short], False)
    return ("analytical_model", False)


def hardware(row):
    """Best available description of what the row ran on."""
    for key in ("info_gpu_name", "info_tpu_device", "info_cpu_model", "info_gpu_model"):
        value = (row.get(key) or "").strip()
        if value:
            return value
    kind, real = classify(row)
    return "CPU" if real else "n/a (model)"


# Display order for the tables: fastest first, then the small-scale-only models.
DISPLAY_ORDER = ["tpu_fused", "gpu_fp16", "gpu_fused_fp16", "tpu_systolic",
                 "dataflow", "optical", "gpu", "gpu_fused", "cpu", "cpu_fused",
                 "gaas_riscv", "iot_edge", "wsn", "cdte_turing",
                 "quantum", "chemical", "biological"]


def fmt_rate(value):
    """Throughput to 2 decimals, except where that would round it to zero."""
    return f"{value:,.2f}" if value >= 0.01 else f"{value:.2g}"


def write_tables(rows):
    """Emit results_table.md (every row) and results_table.tex (headline config)."""
    rank = {p: i for i, p in enumerate(DISPLAY_ORDER)}
    ordered = sorted(rows, key=lambda r: (rank.get(r["paradigm_short"], 99),
                                          int(r["seq_len"]), int(r["embed_dim"])))
    results_dir = os.path.dirname(OUT)

    md = ["# Benchmark results", "",
          "Generated from `combined_results.csv` by `build_results_table.py`.",
          "The `type` column distinguishes a timed run from a calculated one.", "",
          "| paradigm | n | d | FLOPs | time (s) | GFLOP/s | arith. intensity | type |",
          "|:---|---:|---:|---:|---:|---:|---:|:---|"]
    for r in ordered:
        md.append("| {} | {} | {} | {:,} | {:.6f} | {} | {:.2f} | {} |".format(
            r["paradigm_short"], r["seq_len"], r["embed_dim"], int(r["total_flops"]),
            float(r["total_effective_time_s"]), fmt_rate(float(r["throughput_gflop_s"])),
            float(r["arithmetic_intensity"]), r["measurement_type"]))
    with open(os.path.join(results_dir, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")

    big = [r for r in ordered if r["seq_len"] == "2048" and r["embed_dim"] == "256"]
    tex = [r"\begin{table}[h]", r"\centering",
           r"\caption{Attention execution across paradigms (n=2048, d=256). "
           r"`Measured' denotes a timed run on hardware; `Model' an analytical estimate.}",
           r"\begin{tabular}{lrrrl}", r"\hline",
           r"Paradigm & Time (s) & GFLOP/s & AI & Source \\", r"\hline"]
    for r in big:
        tex.append(r"{} & {:.6f} & {} & {:.2f} & {} \\".format(
            r["paradigm_short"].replace("_", r"\_"),
            float(r["total_effective_time_s"]), fmt_rate(float(r["throughput_gflop_s"])),
            float(r["arithmetic_intensity"]),
            "Measured" if r["is_real_hardware"] == "true" else "Model"))
    tex += [r"\hline", r"\end{tabular}", r"\end{table}"]
    with open(os.path.join(results_dir, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write("\n".join(tex) + "\n")

    print(f"  results_table.md ({len(ordered)} rows), results_table.tex ({len(big)} rows)")


def main():
    chosen = OrderedDict()
    provenance = {}

    for source in SOURCES:
        path = os.path.join(RAW, source)
        if not os.path.exists(path):
            print(f"  skipping missing {source}")
            continue

        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                key = (row["paradigm_short"], int(row["seq_len"]), int(row["embed_dim"]))
                kind, real = classify(row)

                incumbent = chosen.get(key)
                if incumbent is not None:
                    _, incumbent_real = classify(incumbent)
                    # A real measurement is never displaced by a model.
                    if incumbent_real and not real:
                        continue

                chosen[key] = row
                provenance[key] = (source, kind, real)

    fieldnames = ["paradigm", "paradigm_short", "seq_len", "embed_dim",
                  "measurement_type", "is_real_hardware", "hardware", "source_file",
                  "total_flops", "total_effective_time_s", "throughput_gflop_s",
                  "total_memory_bytes", "arithmetic_intensity",
                  "time_qkv_projection", "time_qk_matmul", "time_softmax",
                  "time_av_matmul", "time_output_projection", "time_fused_attention",
                  "sdpa_backend"]

    rows = []
    for key in sorted(chosen, key=lambda k: (k[0], k[1], k[2])):
        row = chosen[key]
        source, kind, real = provenance[key]
        rows.append({
            "paradigm": row["paradigm"],
            "paradigm_short": row["paradigm_short"],
            "seq_len": row["seq_len"],
            "embed_dim": row["embed_dim"],
            "measurement_type": kind,
            "is_real_hardware": str(real).lower(),
            "hardware": hardware(row),
            "source_file": source,
            "total_flops": row["total_flops"],
            "total_effective_time_s": row["total_effective_time_s"],
            "throughput_gflop_s": row["throughput_gflops"],
            "total_memory_bytes": row["total_memory_bytes"],
            "arithmetic_intensity": row["arithmetic_intensity"],
            "time_qkv_projection": row.get("time_qkv_projection", ""),
            "time_qk_matmul": row.get("time_qk_matmul", ""),
            "time_softmax": row.get("time_softmax", ""),
            "time_av_matmul": row.get("time_av_matmul", ""),
            "time_output_projection": row.get("time_output_projection", ""),
            "time_fused_attention": row.get("time_fused_attention", ""),
            "sdpa_backend": row.get("info_sdpa_backend_used", ""),
        })

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_tables(rows)

    measured = {r["paradigm_short"] for r in rows if r["is_real_hardware"] == "true"}
    modelled = {r["paradigm_short"] for r in rows} - measured
    print(f"Wrote {OUT}  ({len(rows)} rows)")
    print(f"  measured on hardware ({len(measured)}): {', '.join(sorted(measured))}")
    print(f"  modelled ({len(modelled)}): {', '.join(sorted(modelled))}")

    print("\n  rows where a measurement displaced a model:")
    for key in sorted(chosen):
        source, kind, real = provenance[key]
        if real and source == "combined_all_paradigms.csv" and key[0] in {"gpu", "tpu_systolic"}:
            if key[1] == 2048 and key[2] == 256:
                print(f"    {key[0]:<14} n={key[1]} d={key[2]}  "
                      f"{float(chosen[key]['throughput_gflops']):>10.2f} GFLOP/s  ({kind})")


if __name__ == "__main__":
    main()
