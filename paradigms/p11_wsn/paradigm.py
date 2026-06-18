"""P11: Wireless Sensor Network - distributed attention across constrained nodes."""

import time

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p11_wsn.sensor_network import SensorNetwork
import config


@register
class WSNParadigm(BaseParadigm):
    name = "WSN (Wireless Sensor Network)"
    short_name = "wsn"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim

        network = SensorNetwork(
            num_nodes=config.WSN_NUM_NODES,
            mcu_clock_hz=config.WSN_MCU_CLOCK_HZ,
            cycles_per_mac=config.WSN_MCU_CYCLES_PER_MAC,
            node_ram_bytes=config.WSN_NODE_RAM_BYTES,
            wireless_bps=config.WSN_WIRELESS_BPS,
            hop_latency_s=config.WSN_HOP_LATENCY_S,
            sync_rounds_softmax=config.WSN_SYNC_ROUNDS_SOFTMAX,
        )

        steps = []

        # Step 1: QKV projections (3 distributed matmuls)
        t0 = time.perf_counter()
        qkv_time_one, qkv_info = network.matmul_time(n, d, d)
        qkv_hw_time = qkv_time_one * 3
        t1 = time.perf_counter()

        r, w = mem[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=qkv_hw_time,
            paradigm_specific=qkv_info,
        ))

        # Step 2: QK^T (n x d @ d x n)
        t0 = time.perf_counter()
        qkt_hw_time, qkt_info = network.matmul_time(n, d, n)
        t1 = time.perf_counter()

        r, w = mem[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=qkt_hw_time,
            paradigm_specific=qkt_info,
        ))

        # Step 3: Softmax (distributed with global sync)
        t0 = time.perf_counter()
        sm_hw_time, sm_info = network.softmax_time(n)
        t1 = time.perf_counter()

        r, w = mem[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sm_hw_time,
            paradigm_specific=sm_info,
        ))

        # Step 4: AV (n x n @ n x d)
        t0 = time.perf_counter()
        av_hw_time, av_info = network.matmul_time(n, n, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=av_hw_time,
            paradigm_specific=av_info,
        ))

        # Step 5: Output projection (n x d @ d x d)
        t0 = time.perf_counter()
        out_hw_time, out_info = network.matmul_time(n, d, d)
        t1 = time.perf_counter()

        r, w = mem[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=out_hw_time,
            paradigm_specific=out_info,
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
                "num_nodes": config.WSN_NUM_NODES,
                "mcu_clock_hz": config.WSN_MCU_CLOCK_HZ,
                "wireless_bps": config.WSN_WIRELESS_BPS,
                "protocol": "IEEE 802.15.4 (ZigBee)",
                "execution_model": "distributed tiling, wireless aggregation",
            },
        )
