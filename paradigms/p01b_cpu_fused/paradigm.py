"""P01b: CPU Fused SDPA - Real-world attention using PyTorch's optimized kernel."""

import time
import torch

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import compute_step_flops, compute_step_memory
from core.fused_attention import run_fused_sdpa, is_sdpa_available
import config


WARMUP_RUNS = 3


@register
class CPUFusedParadigm(BaseParadigm):
    name = "CPU (Fused SDPA)"
    short_name = "cpu_fused"

    def is_available(self) -> bool:
        return is_sdpa_available()

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
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
                "execution_mode": "real_cpu_fused",
                "backend": "torch.nn.functional.scaled_dot_product_attention",
                "num_heads": config.NUM_HEADS,
                "torch_version": torch.__version__,
            },
        )
