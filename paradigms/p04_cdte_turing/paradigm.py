"""P04: CdTe Turing Machine - 4-operation attention execution."""

import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p04_cdte_turing.turing_machine import (
    TuringMachine, program_dot_product, estimate_matmul_ops, estimate_softmax_ops,
)
import config


DEMO_SIZE = 4  # Run actual TM simulation at this matrix size
INT_SCALE = 10  # Scale float values to integers [0, INT_SCALE]


def _hw_time(ops, clock_hz):
    """Estimated hardware time: ops / clock_frequency."""
    return ops / clock_hz


@register
class CdTeTuringParadigm(BaseParadigm):
    name = "CdTe (Turing Machine)"
    short_name = "cdte_turing"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim
        clock_hz = config.CDTE_CLOCK_HZ

        steps = []

        # Run a tiny demo on actual TM for verification
        demo_n = min(DEMO_SIZE, n)
        demo_d = min(DEMO_SIZE, d)
        X_int = np.clip(np.abs(X[:demo_n, :demo_d] * INT_SCALE), 0, INT_SCALE).astype(int)

        # Step 1: QKV projections
        t0 = time.perf_counter()
        qkv_est = estimate_matmul_ops(n, d, d, max_val=INT_SCALE)
        qkv_total_ops = qkv_est["total_ops"] * 3

        tm = TuringMachine(tape_size=8192)
        if demo_n > 0 and demo_d > 0:
            row = X_int[0].tolist()
            col = X_int[:, 0].tolist()[:demo_d]
            prog = program_dot_product(row, col)
            tm.execute(prog)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=_hw_time(qkv_total_ops, clock_hz),
            paradigm_specific={
                "estimated_tm_ops": qkv_total_ops,
                "demo_tm_steps": tm.steps,
                "demo_op_counts": tm.op_counts.copy(),
            },
        ))

        # Step 2: QK^T
        t0 = time.perf_counter()
        qkt_est = estimate_matmul_ops(n, d, n, max_val=INT_SCALE)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=_hw_time(qkt_est["total_ops"], clock_hz),
            paradigm_specific={"estimated_tm_ops": qkt_est["total_ops"]},
        ))

        # Step 3: Softmax
        t0 = time.perf_counter()
        sm_est = estimate_softmax_ops(n, max_val=INT_SCALE)
        sm_total = sm_est["total_ops"] * n
        t1 = time.perf_counter()

        r, w = mem[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=_hw_time(sm_total, clock_hz),
            paradigm_specific={"estimated_tm_ops": sm_total, "note": "exp/div extremely costly on TM"},
        ))

        # Step 4: AV
        t0 = time.perf_counter()
        av_est = estimate_matmul_ops(n, n, d, max_val=INT_SCALE)
        t1 = time.perf_counter()

        r, w = mem[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=_hw_time(av_est["total_ops"], clock_hz),
            paradigm_specific={"estimated_tm_ops": av_est["total_ops"]},
        ))

        # Step 5: Output projection
        t0 = time.perf_counter()
        out_est = estimate_matmul_ops(n, d, d, max_val=INT_SCALE)
        t1 = time.perf_counter()

        r, w = mem[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=_hw_time(out_est["total_ops"], clock_hz),
            paradigm_specific={"estimated_tm_ops": out_est["total_ops"]},
        ))

        total_tm_ops = (qkv_total_ops + qkt_est["total_ops"] + sm_total +
                        av_est["total_ops"] + out_est["total_ops"])

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "total_tm_operations": total_tm_ops,
                "estimated_hw_time_s": total_tm_ops / clock_hz,
                "clock_hz": clock_hz,
                "isa": "4 ops: SKIP, CSKIP, INC, ADD",
                "arithmetic": "integer only",
                "multiply_method": "repeated addition O(a*b)",
            },
        )
