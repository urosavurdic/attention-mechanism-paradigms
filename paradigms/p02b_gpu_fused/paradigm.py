"""P02b: GPU Fused SDPA - Real-world attention using PyTorch's optimized CUDA kernel.

Uses torch.nn.functional.scaled_dot_product_attention with multi-head reshape.
Detects which SDPA backend was actually dispatched (flash / mem-efficient / math).
"""

import time
import torch
import torch.nn.functional as F

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import compute_step_flops, compute_step_memory
from core.fused_attention import run_fused_sdpa, detect_sdpa_backend, is_sdpa_available
import config


WARMUP_RUNS = 3


def _has_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@register
class GPUFusedParadigm(BaseParadigm):
    name = "GPU (Fused SDPA)"
    short_name = "gpu_fused"

    def is_available(self) -> bool:
        return is_sdpa_available()

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        if _has_cuda():
            return self._run_cuda(X, W_q, W_k, W_v, W_o)
        return self._run_cpu_fallback(X, W_q, W_k, W_v, W_o)

    def _run_cuda(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        total_flops = sum(flops.values())
        total_reads = sum(r for r, _ in mem.values())
        total_writes = sum(w for _, w in mem.values())

        device = torch.device("cuda")
        X_t = torch.from_numpy(X).to(device)
        W_q_t = torch.from_numpy(W_q).to(device)
        W_k_t = torch.from_numpy(W_k).to(device)
        W_v_t = torch.from_numpy(W_v).to(device)
        W_o_t = torch.from_numpy(W_o).to(device)
        torch.cuda.synchronize()

        for _ in range(WARMUP_RUNS):
            run_fused_sdpa(X_t, W_q_t, W_k_t, W_v_t, W_o_t)
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        run_fused_sdpa(X_t, W_q_t, W_k_t, W_v_t, W_o_t)
        end.record()
        torch.cuda.synchronize()
        wall = start.elapsed_time(end) / 1000.0

        backend_used = detect_sdpa_backend(X_t, W_q_t, W_k_t, W_v_t, torch.float32)
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

        steps = [StepMetrics(
            name="fused_attention",
            flops=total_flops,
            memory_reads_bytes=total_reads,
            memory_writes_bytes=total_writes,
            wall_time_seconds=wall,
            estimated_hw_time_seconds=0,
            paradigm_specific={
                "num_heads": config.NUM_HEADS,
                "head_dim": embed_dim // config.NUM_HEADS,
                "real_cuda_time_s": wall,
            },
        )]

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "real_cuda_fused",
                "gpu_name": gpu_name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "precision": "float32",
                "sdpa_backend_used": backend_used,
                "backend": "torch.nn.functional.scaled_dot_product_attention",
                "num_heads": config.NUM_HEADS,
                "head_dim": embed_dim // config.NUM_HEADS,
                "note_if_math": (
                    "T4 (sm_75) + FP32 → SDPA uses math backend (no FlashAttention). "
                    "Flash requires Ampere+ (sm_80) and FP16/BF16."
                    if "math" in backend_used.lower() else ""
                ),
            },
        )

    def _run_cpu_fallback(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        total_flops = sum(flops.values())
        total_reads = sum(r for r, _ in mem.values())
        total_writes = sum(w for _, w in mem.values())

        X_t = torch.from_numpy(X)
        W_q_t = torch.from_numpy(W_q)
        W_k_t = torch.from_numpy(W_k)
        W_v_t = torch.from_numpy(W_v)
        W_o_t = torch.from_numpy(W_o)

        for _ in range(WARMUP_RUNS):
            run_fused_sdpa(X_t, W_q_t, W_k_t, W_v_t, W_o_t)

        t0 = time.perf_counter()
        run_fused_sdpa(X_t, W_q_t, W_k_t, W_v_t, W_o_t)
        t1 = time.perf_counter()
        wall = t1 - t0

        steps = [StepMetrics(
            name="fused_attention",
            flops=total_flops,
            memory_reads_bytes=total_reads,
            memory_writes_bytes=total_writes,
            wall_time_seconds=wall,
            estimated_hw_time_seconds=0,
            paradigm_specific={"num_heads": config.NUM_HEADS},
        )]

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "cpu_fallback",
                "backend": "torch.nn.functional.scaled_dot_product_attention",
                "num_heads": config.NUM_HEADS,
                "note": "No CUDA available, ran on CPU",
            },
        )
