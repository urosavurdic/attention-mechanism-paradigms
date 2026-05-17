"""P02c: GPU FP16 (Control Flow) - Decomposed attention in half precision.

Same decomposed pipeline as p02_gpu but in FP16, enabling Tensor Core paths
on supported GPUs. Provides apples-to-apples precision comparison.
"""

import numpy as np
import torch

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
import config


WARMUP_RUNS = 3


def _has_cuda():
    try:
        return torch.cuda.is_available()
    except ImportError:
        return False


@register
class GPUFP16Paradigm(BaseParadigm):
    name = "GPU FP16 (Control Flow)"
    short_name = "gpu_fp16"

    def is_available(self) -> bool:
        return _has_cuda()

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        if not _has_cuda():
            return None

        seq_len, embed_dim = X.shape
        flops_map = compute_step_flops(seq_len, embed_dim)
        mem_map = compute_step_memory(seq_len, embed_dim)

        device = torch.device("cuda")
        X_t = torch.from_numpy(X).to(device, dtype=torch.float16)
        Wq_t = torch.from_numpy(W_q).to(device, dtype=torch.float16)
        Wk_t = torch.from_numpy(W_k).to(device, dtype=torch.float16)
        Wv_t = torch.from_numpy(W_v).to(device, dtype=torch.float16)
        Wo_t = torch.from_numpy(W_o).to(device, dtype=torch.float16)
        torch.cuda.synchronize()

        for _ in range(WARMUP_RUNS):
            Q = X_t @ Wq_t
            _ = Q @ Wk_t.T
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
            steps.append(StepMetrics(
                name=step_name, flops=flops_map[step_name],
                memory_reads_bytes=r // 2, memory_writes_bytes=w // 2,
                wall_time_seconds=wall,
                estimated_hw_time_seconds=0,
                is_real_execution=True,
                paradigm_specific={"real_cuda_time_s": wall},
            ))
            return result

        Q, K, V = _timed(lambda: (X_t @ Wq_t, X_t @ Wk_t, X_t @ Wv_t), STEP_QKV)
        d_k = embed_dim
        scores = _timed(lambda: Q @ K.T / (d_k ** 0.5), STEP_QKT)
        attention = _timed(lambda: torch.softmax(scores, dim=-1), STEP_SOFTMAX)
        attended = _timed(lambda: attention @ V, STEP_AV)
        _timed(lambda: attended @ Wo_t, STEP_OUT)

        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "real_cuda_fp16",
                "gpu_name": gpu_name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "precision": "float16",
                "tensor_cores": "enabled (FP16 Tensor Core path available)",
            },
        )
