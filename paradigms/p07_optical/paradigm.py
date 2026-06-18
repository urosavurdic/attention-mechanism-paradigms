"""P07: Optical Computing - Photonic MZI mesh for attention execution."""

import time

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p07_optical.mzi_mesh import MZIMesh
import config


@register
class OpticalParadigm(BaseParadigm):
    name = "Optical (Photonic MZI)"
    short_name = "optical"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim

        mesh = MZIMesh(
            mesh_size=config.OPTICAL_MZI_MESH_SIZE,
            propagation_time_s=config.OPTICAL_PROPAGATION_TIME_S,
            dac_rate_hz=config.OPTICAL_DAC_RATE_HZ,
            oeo_latency_s=config.OPTICAL_OEO_LATENCY_S,
            electronic_softmax_clock_hz=config.OPTICAL_ELECTRONIC_SOFTMAX_CLOCK_HZ,
        )

        steps = []

        # Step 1: QKV projections (3 matmuls: n x d @ d x d)
        t0 = time.perf_counter()
        qkv_time_one, qkv_tiles = mesh.matmul_time(n, d, d)
        qkv_hw_time = qkv_time_one * 3
        t1 = time.perf_counter()

        r, w = mem[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=qkv_hw_time,
            paradigm_specific={
                "tiles_per_matmul": qkv_tiles,
                "tile_time_s": mesh.tile_time(),
                "mesh_size": config.OPTICAL_MZI_MESH_SIZE,
            },
        ))

        # Step 2: QK^T (n x d @ d x n)
        t0 = time.perf_counter()
        qkt_hw_time, qkt_tiles = mesh.matmul_time(n, d, n)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=qkt_hw_time,
            paradigm_specific={"tiles": qkt_tiles},
        ))

        # Step 3: Softmax (requires OEO conversion)
        t0 = time.perf_counter()
        sm_hw_time = mesh.softmax_time(n)
        t1 = time.perf_counter()

        r, w = mem[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sm_hw_time,
            paradigm_specific={
                "oeo_latency_s": config.OPTICAL_OEO_LATENCY_S,
                "electronic_clock_hz": config.OPTICAL_ELECTRONIC_SOFTMAX_CLOCK_HZ,
                "note": "requires optical-electrical-optical conversion",
            },
        ))

        # Step 4: AV (n x n @ n x d)
        t0 = time.perf_counter()
        av_hw_time, av_tiles = mesh.matmul_time(n, n, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=av_hw_time,
            paradigm_specific={"tiles": av_tiles},
        ))

        # Step 5: Output projection (n x d @ d x d)
        t0 = time.perf_counter()
        out_hw_time, out_tiles = mesh.matmul_time(n, d, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=out_hw_time,
            paradigm_specific={"tiles": out_tiles},
        ))

        total_hw_time = qkv_hw_time + qkt_hw_time + sm_hw_time + av_hw_time + out_hw_time

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "estimated_hw_time_s": total_hw_time,
                "mesh_size": config.OPTICAL_MZI_MESH_SIZE,
                "dac_rate_hz": config.OPTICAL_DAC_RATE_HZ,
                "oeo_latency_s": config.OPTICAL_OEO_LATENCY_S,
                "execution_model": "photonic MZI mesh, OEO for nonlinearities",
            },
        )
