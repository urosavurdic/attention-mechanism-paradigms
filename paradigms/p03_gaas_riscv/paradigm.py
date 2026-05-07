"""P03: GaAs RISC-V - Simulated RISC-V processor executing attention."""

import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p03_gaas_riscv.isa import (
    compile_matmul, compile_softmax, count_instructions, estimate_cycles, CYCLE_COSTS, Op,
)
import config


# Analytical instruction count formulas (avoid generating huge lists)
def _matmul_instr_count(M, K, N):
    """Instruction count for (M,K)@(K,N): per element = 1 + K*4 + 1 = 4K+2."""
    per_element = 1 + K * 4 + 1  # zero_acc + K*(LW,LW,FMUL,FADD) + SW
    return M * N * per_element + 1  # +1 for HALT

def _matmul_cycle_count(M, K, N):
    """Cycle estimate for matmul."""
    # Per element: 1 ADD + K*(4 LW + 4 LW + 3 FMUL + 2 FADD) + 3 SW
    per_element = 1 + K * (4 + 4 + 3 + 2) + 3
    return M * N * per_element + 1

def _softmax_instr_count(N):
    """Instruction count for softmax over vector of length N."""
    # Compile a small one and extrapolate
    if N <= 8:
        prog = compile_softmax(N)
        return len(prog)
    prog_small = compile_softmax(8)
    per_element = len(prog_small) / 8
    return int(per_element * N)

def _softmax_cycle_count(N):
    """Cycle estimate for softmax."""
    if N <= 8:
        prog = compile_softmax(N)
        return estimate_cycles(prog)
    prog_small = compile_softmax(8)
    per_element = estimate_cycles(prog_small) / 8
    return int(per_element * N)


DEMO_SIZE = 4  # Compile actual program only at this size for demo


@register
class GaAsRiscVParadigm(BaseParadigm):
    name = "GaAs (RISC-V)"
    short_name = "gaas_riscv"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim

        steps = []
        total_instr = 0
        total_cyc = 0
        clock_hz = config.GAAS_CLOCK_HZ

        # Compile a small demo for actual instruction breakdown
        t0 = time.perf_counter()
        demo_prog = compile_matmul(DEMO_SIZE, DEMO_SIZE, DEMO_SIZE)
        demo_breakdown = count_instructions(demo_prog)
        t1 = time.perf_counter()
        demo_time = t1 - t0

        # Step 1: QKV projections (3 matmuls: n x d @ d x d)
        t0 = time.perf_counter()
        qkv_instr = _matmul_instr_count(n, d, d) * 3
        qkv_cycles = _matmul_cycle_count(n, d, d) * 3
        t1 = time.perf_counter()

        r, w = mem[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0 + demo_time,
            estimated_hw_time_seconds=qkv_cycles / clock_hz,
            paradigm_specific={
                "instructions": qkv_instr,
                "estimated_cycles": qkv_cycles,
                "demo_instruction_breakdown": demo_breakdown,
            },
        ))
        total_instr += qkv_instr
        total_cyc += qkv_cycles

        # Step 2: QK^T (n x d @ d x n)
        t0 = time.perf_counter()
        qkt_instr = _matmul_instr_count(n, d, n)
        qkt_cycles = _matmul_cycle_count(n, d, n)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=qkt_cycles / clock_hz,
            paradigm_specific={
                "instructions": qkt_instr,
                "estimated_cycles": qkt_cycles,
            },
        ))
        total_instr += qkt_instr
        total_cyc += qkt_cycles

        # Step 3: Softmax (n rows of length n)
        t0 = time.perf_counter()
        sm_instr = _softmax_instr_count(n) * n
        sm_cycles = _softmax_cycle_count(n) * n
        t1 = time.perf_counter()

        r, w = mem[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sm_cycles / clock_hz,
            paradigm_specific={
                "instructions": sm_instr,
                "estimated_cycles": sm_cycles,
            },
        ))
        total_instr += sm_instr
        total_cyc += sm_cycles

        # Step 4: AV (n x n @ n x d)
        t0 = time.perf_counter()
        av_instr = _matmul_instr_count(n, n, d)
        av_cycles = _matmul_cycle_count(n, n, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=av_cycles / clock_hz,
            paradigm_specific={
                "instructions": av_instr,
                "estimated_cycles": av_cycles,
            },
        ))
        total_instr += av_instr
        total_cyc += av_cycles

        # Step 5: Output projection (n x d @ d x d)
        t0 = time.perf_counter()
        out_instr = _matmul_instr_count(n, d, d)
        out_cycles = _matmul_cycle_count(n, d, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=out_cycles / clock_hz,
            paradigm_specific={
                "instructions": out_instr,
                "estimated_cycles": out_cycles,
            },
        ))
        total_instr += out_instr
        total_cyc += out_cycles

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "total_instructions": total_instr,
                "total_estimated_cycles": total_cyc,
                "isa": "RISC-V RV32I subset (10 instructions)",
                "cycle_model": "1 ALU, 3 MUL, 4 LW, 10 FDIV",
            },
        )
