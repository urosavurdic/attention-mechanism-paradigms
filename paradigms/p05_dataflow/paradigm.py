"""P05: Dataflow (Maxeler-style) - Streaming pipeline attention execution."""

import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p05_dataflow.pipeline import build_attention_pipelines
import config


@register
class DataflowParadigm(BaseParadigm):
    name = "Dataflow (Maxeler-style)"
    short_name = "dataflow"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem_est = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim
        clock_hz = config.DATAFLOW_CLOCK_HZ
        parallel_pipes = config.DATAFLOW_PARALLEL_PIPES

        pipelines = build_attention_pipelines(n, d)
        steps = []

        # Stream lengths = number of dot products (output elements) per step
        stream_configs = {
            STEP_QKV: ("qkv_projection", 3 * n * d),   # 3 matmuls of (n,d)@(d,d) → 3*n*d outputs
            STEP_QKT: ("qk_matmul", n * n),             # (n,d)@(d,n) → n*n outputs
            STEP_SOFTMAX: ("softmax", n * n),            # n*n elements through softmax pipeline
            STEP_AV: ("av_matmul", n * d),               # (n,n)@(n,d) → n*d outputs
            STEP_OUT: ("output_projection", n * d),      # (n,d)@(d,d) → n*d outputs
        }

        total_cycles = 0
        for step_name in [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]:
            pipe_name, stream_len = stream_configs[step_name]
            pipeline = pipelines[pipe_name]

            t0 = time.perf_counter()
            sim_result = pipeline.simulate(stream_len)
            t1 = time.perf_counter()

            # Parallel pipelines divide effective cycles
            raw_cycles = sim_result["total_cycles"]
            effective_cycles = (raw_cycles + parallel_pipes - 1) // parallel_pipes

            r, w = mem_est[step_name]
            steps.append(StepMetrics(
                name=step_name,
                flops=flops[step_name],
                memory_reads_bytes=r,
                memory_writes_bytes=w,
                wall_time_seconds=t1 - t0,
                estimated_hw_time_seconds=effective_cycles / clock_hz,
                paradigm_specific={
                    "raw_pipeline_cycles": raw_cycles,
                    "effective_cycles": effective_cycles,
                    "parallel_pipes": parallel_pipes,
                    "pipeline_depth": sim_result["pipeline_depth"],
                    "stream_length": sim_result["stream_length"],
                    "utilization": f"{sim_result['utilization']:.2%}",
                    "num_stages": sim_result["num_stages"],
                },
            ))
            total_cycles += effective_cycles

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "total_effective_cycles": total_cycles,
                "estimated_hw_time_s": total_cycles / clock_hz,
                "clock_hz": clock_hz,
                "parallel_pipes": parallel_pipes,
                "execution_model": "streaming pipeline, no instruction fetch",
            },
        )
