"""P01: CPU (Control Flow) - NumPy-based attention execution with roofline estimate."""

import os
import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
import config


@register
class CPUParadigm(BaseParadigm):
    name = "CPU (Control Flow)"
    short_name = "cpu"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)

        peak_flops = config.CPU_PEAK_TFLOPS_FP32 * 1e12
        mem_bw = config.CPU_MEMORY_BW_GB_S * 1e9

        steps = []

        def _record(step_name, wall_time):
            step_flops = flops[step_name]
            r, w = mem[step_name]
            total_bytes = r + w
            compute_time = step_flops / peak_flops
            memory_time = total_bytes / mem_bw
            roofline_estimate = max(compute_time, memory_time)
            steps.append(StepMetrics(
                name=step_name,
                flops=step_flops,
                memory_reads_bytes=r,
                memory_writes_bytes=w,
                wall_time_seconds=wall_time,
                estimated_hw_time_seconds=roofline_estimate,
                is_real_execution=True,
                paradigm_specific={
                    "roofline_estimate_s": roofline_estimate,
                    "compute_bound_time_s": compute_time,
                    "memory_bound_time_s": memory_time,
                    "bottleneck": "compute" if compute_time > memory_time else "memory",
                },
            ))

        # Step 1: QKV projections
        t0 = time.perf_counter()
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        t1 = time.perf_counter()
        _record(STEP_QKV, t1 - t0)

        # Step 2: QK^T / sqrt(d)
        d_k = embed_dim
        t0 = time.perf_counter()
        scores = Q @ K.T / np.sqrt(d_k)
        t1 = time.perf_counter()
        _record(STEP_QKT, t1 - t0)

        # Step 3: Softmax
        t0 = time.perf_counter()
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        t1 = time.perf_counter()
        _record(STEP_SOFTMAX, t1 - t0)

        # Step 4: Attention @ V
        t0 = time.perf_counter()
        attended = attention @ V
        t1 = time.perf_counter()
        _record(STEP_AV, t1 - t0)

        # Step 5: Output projection
        t0 = time.perf_counter()
        output = attended @ W_o
        t1 = time.perf_counter()
        _record(STEP_OUT, t1 - t0)

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "real_cpu",
                "cpu_model": "Intel i7 AVX2 (estimated)",
                "peak_tflops_fp32": config.CPU_PEAK_TFLOPS_FP32,
                "memory_bw_gb_s": config.CPU_MEMORY_BW_GB_S,
                "cpu_count": os.cpu_count(),
                "numpy_version": np.__version__,
            },
        )
