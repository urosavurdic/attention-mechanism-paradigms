"""P02: GPU (Control Flow) - Real CUDA + roofline simulation."""

import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
    reference_attention,
)
import config


def _has_cuda():
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@register
class GPUParadigm(BaseParadigm):
    name = "GPU (Control Flow)"
    short_name = "gpu"

    def is_available(self) -> bool:
        return True

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops_map = compute_step_flops(seq_len, embed_dim)
        mem_map = compute_step_memory(seq_len, embed_dim)

        peak_flops = config.GPU_PEAK_TFLOPS_FP32 * 1e12
        mem_bw = config.GPU_MEMORY_BW_GB_S * 1e9
        launch_overhead = config.GPU_KERNEL_LAUNCH_OVERHEAD_S

        step_names = [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]

        # Compute roofline estimates for all steps
        roofline = {}
        for step_name in step_names:
            sf = flops_map[step_name]
            r, w = mem_map[step_name]
            compute_time = sf / peak_flops
            memory_time = (r + w) / mem_bw
            roofline[step_name] = {
                "estimated_time": max(compute_time, memory_time) + launch_overhead,
                "compute_bound_time_s": compute_time,
                "memory_bound_time_s": memory_time,
                "bottleneck": "compute" if compute_time > memory_time else "memory",
                "launch_overhead_s": launch_overhead,
            }

        if _has_cuda():
            return self._run_with_cuda(X, W_q, W_k, W_v, W_o, flops_map, mem_map, roofline)
        return self._run_sim_only(X, W_q, W_k, W_v, W_o, flops_map, mem_map, roofline)

    def _run_with_cuda(self, X, W_q, W_k, W_v, W_o, flops_map, mem_map, roofline):
        """Real CUDA execution with roofline estimates alongside."""
        import torch

        seq_len, embed_dim = X.shape
        device = torch.device("cuda")

        X_t = torch.from_numpy(X).to(device)
        Wq_t = torch.from_numpy(W_q).to(device)
        Wk_t = torch.from_numpy(W_k).to(device)
        Wv_t = torch.from_numpy(W_v).to(device)
        Wo_t = torch.from_numpy(W_o).to(device)
        torch.cuda.synchronize()

        steps = []

        def _timed(fn, step_name):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = fn()
            end.record()
            torch.cuda.synchronize()
            wall = start.elapsed_time(end) / 1000.0
            r, w = mem_map[step_name]
            rf = roofline[step_name]
            steps.append(StepMetrics(
                name=step_name, flops=flops_map[step_name],
                memory_reads_bytes=r, memory_writes_bytes=w,
                wall_time_seconds=wall,
                estimated_hw_time_seconds=rf["estimated_time"],
                is_real_execution=True,
                paradigm_specific={
                    "roofline_estimate_s": rf["estimated_time"],
                    "compute_bound_time_s": rf["compute_bound_time_s"],
                    "memory_bound_time_s": rf["memory_bound_time_s"],
                    "bottleneck": rf["bottleneck"],
                    "real_cuda_time_s": wall,
                },
            ))
            return result

        Q, K, V = _timed(lambda: (X_t @ Wq_t, X_t @ Wk_t, X_t @ Wv_t), STEP_QKV)
        d_k = embed_dim
        scores = _timed(lambda: Q @ K.T / (d_k ** 0.5), STEP_QKT)
        attention = _timed(lambda: torch.softmax(scores, dim=-1), STEP_SOFTMAX)
        attended = _timed(lambda: attention @ V, STEP_AV)
        _timed(lambda: attended @ Wo_t, STEP_OUT)

        gpu_name = torch.cuda.get_device_name(0)
        mem_alloc = torch.cuda.memory_allocated(0)

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "real_cuda",
                "gpu_name": gpu_name,
                "memory_allocated_mb": mem_alloc / 1024 / 1024,
                "peak_tflops_fp32": config.GPU_PEAK_TFLOPS_FP32,
                "memory_bw_gb_s": config.GPU_MEMORY_BW_GB_S,
            },
        )

    def _run_sim_only(self, X, W_q, W_k, W_v, W_o, flops_map, mem_map, roofline):
        """Roofline-only estimation when no CUDA is available."""
        seq_len, embed_dim = X.shape

        t0 = time.perf_counter()
        reference_attention(X, W_q, W_k, W_v, W_o)
        total_wall = time.perf_counter() - t0

        step_names = [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]
        steps = []
        for step_name in step_names:
            r, w = mem_map[step_name]
            rf = roofline[step_name]
            steps.append(StepMetrics(
                name=step_name,
                flops=flops_map[step_name],
                memory_reads_bytes=r,
                memory_writes_bytes=w,
                wall_time_seconds=total_wall / 5,
                estimated_hw_time_seconds=rf["estimated_time"],
                paradigm_specific={
                    "compute_bound_time_s": rf["compute_bound_time_s"],
                    "memory_bound_time_s": rf["memory_bound_time_s"],
                    "bottleneck": rf["bottleneck"],
                    "roofline_estimate_s": rf["estimated_time"],
                },
            ))

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "roofline_simulation",
                "gpu_model": "NVIDIA T4 (estimated)",
                "peak_tflops_fp32": config.GPU_PEAK_TFLOPS_FP32,
                "memory_bw_gb_s": config.GPU_MEMORY_BW_GB_S,
            },
        )
