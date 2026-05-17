"""P02d: GPU Fused SDPA FP16 - Fused attention in half precision.

Same SDPA pipeline as p02b_gpu_fused but in FP16, which enables
FlashAttention / memory-efficient backends on supported GPUs.
On T4 (sm_75), mem_efficient backend becomes available.
On A100/H100 (sm_80+), FlashAttention backend is available.
"""

import time
import torch
import torch.nn.functional as F

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import compute_step_flops, compute_step_memory
from core.fused_attention import detect_sdpa_backend, is_sdpa_available
import config


WARMUP_RUNS = 3


def _has_cuda():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _run_fused_sdpa_fp16(X_t, W_q_t, W_k_t, W_v_t, W_o_t):
    seq_len, embed_dim = X_t.shape
    num_heads = config.NUM_HEADS
    head_dim = embed_dim // num_heads

    Q = (X_t @ W_q_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    K = (X_t @ W_k_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    V = (X_t @ W_v_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)

    attn_out = F.scaled_dot_product_attention(Q, K, V)

    out = attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, embed_dim)
    return out @ W_o_t


@register
class GPUFusedFP16Paradigm(BaseParadigm):
    name = "GPU FP16 (Fused SDPA)"
    short_name = "gpu_fused_fp16"

    def is_available(self) -> bool:
        return is_sdpa_available() and _has_cuda()

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        if not _has_cuda():
            return None

        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        total_flops = sum(flops.values())
        total_reads = sum(r for r, _ in mem.values())
        total_writes = sum(w for _, w in mem.values())

        device = torch.device("cuda")
        X_t = torch.from_numpy(X).to(device, dtype=torch.float16)
        W_q_t = torch.from_numpy(W_q).to(device, dtype=torch.float16)
        W_k_t = torch.from_numpy(W_k).to(device, dtype=torch.float16)
        W_v_t = torch.from_numpy(W_v).to(device, dtype=torch.float16)
        W_o_t = torch.from_numpy(W_o).to(device, dtype=torch.float16)
        torch.cuda.synchronize()

        for _ in range(WARMUP_RUNS):
            _run_fused_sdpa_fp16(X_t, W_q_t, W_k_t, W_v_t, W_o_t)
            torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        _run_fused_sdpa_fp16(X_t, W_q_t, W_k_t, W_v_t, W_o_t)
        end.record()
        torch.cuda.synchronize()
        wall = start.elapsed_time(end) / 1000.0

        backend_used = detect_sdpa_backend(X_t, W_q_t, W_k_t, W_v_t, torch.float16)
        gpu_name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability(0)

        steps = [StepMetrics(
            name="fused_attention",
            flops=total_flops,
            memory_reads_bytes=total_reads // 2,
            memory_writes_bytes=total_writes // 2,
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
                "execution_mode": "real_cuda_fused_fp16",
                "gpu_name": gpu_name,
                "compute_capability": f"{cc[0]}.{cc[1]}",
                "precision": "float16",
                "sdpa_backend_used": backend_used,
                "backend": "torch.nn.functional.scaled_dot_product_attention",
                "num_heads": config.NUM_HEADS,
                "head_dim": embed_dim // config.NUM_HEADS,
            },
        )
