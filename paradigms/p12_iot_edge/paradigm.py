"""P12: IoT Edge Computing - ARM Cortex-M4F with DSP and hardware FPU."""

import time

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p12_iot_edge.edge_device import CortexM4Device
import config


@register
class IoTEdgeParadigm(BaseParadigm):
    name = "IoT Edge (ARM Cortex-M4F)"
    short_name = "iot_edge"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim

        device = CortexM4Device(
            clock_hz=config.IOT_CLOCK_HZ,
            cycles_per_mac=config.IOT_CYCLES_PER_MAC,
            sram_bytes=config.IOT_SRAM_BYTES,
            flash_bytes=config.IOT_FLASH_BYTES,
            exp_lut_cycles=config.IOT_EXP_LUT_CYCLES,
            div_cycles=config.IOT_DIV_CYCLES,
        )

        steps = []
        total_cycles = 0

        # Step 1: QKV projections (3 matmuls: n x d @ d x d)
        t0 = time.perf_counter()
        qkv_cycles = device.matmul_cycles(n, d, d) * 3
        t1 = time.perf_counter()

        r, w = mem[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=device.cycles_to_time(qkv_cycles),
            paradigm_specific={
                "estimated_cycles": qkv_cycles,
                "needs_tiling": device.needs_tiling(n, d, d),
                "tile_dim": device.max_tile_dim(),
            },
        ))
        total_cycles += qkv_cycles

        # Step 2: QK^T (n x d @ d x n)
        t0 = time.perf_counter()
        qkt_cycles = device.matmul_cycles(n, d, n)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=device.cycles_to_time(qkt_cycles),
            paradigm_specific={"estimated_cycles": qkt_cycles},
        ))
        total_cycles += qkt_cycles

        # Step 3: Softmax
        t0 = time.perf_counter()
        sm_cycles = device.softmax_cycles(n)
        t1 = time.perf_counter()

        r, w = mem[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=device.cycles_to_time(sm_cycles),
            paradigm_specific={
                "estimated_cycles": sm_cycles,
                "exp_method": "LUT approximation",
                "cycles_per_exp": config.IOT_EXP_LUT_CYCLES,
                "cycles_per_div": config.IOT_DIV_CYCLES,
            },
        ))
        total_cycles += sm_cycles

        # Step 4: AV (n x n @ n x d)
        t0 = time.perf_counter()
        av_cycles = device.matmul_cycles(n, n, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=device.cycles_to_time(av_cycles),
            paradigm_specific={"estimated_cycles": av_cycles},
        ))
        total_cycles += av_cycles

        # Step 5: Output projection (n x d @ d x d)
        t0 = time.perf_counter()
        out_cycles = device.matmul_cycles(n, d, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=device.cycles_to_time(out_cycles),
            paradigm_specific={"estimated_cycles": out_cycles},
        ))
        total_cycles += out_cycles

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "total_estimated_cycles": total_cycles,
                "estimated_hw_time_s": device.cycles_to_time(total_cycles),
                "clock_hz": config.IOT_CLOCK_HZ,
                "processor": "ARM Cortex-M4F",
                "sram_bytes": config.IOT_SRAM_BYTES,
                "dsp_mac": f"{config.IOT_CYCLES_PER_MAC} cycle/MAC",
                "execution_model": "single-core sequential with DSP",
            },
        )
