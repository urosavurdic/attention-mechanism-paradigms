"""P06: TPU / Systolic Array - Real TPU (JAX) + systolic simulation.

Real TPU mode: each attention sub-operation (QKV, QK^T, softmax, AV, output)
is JIT-compiled and timed SEPARATELY with block_until_ready(), giving actual
per-operation bottleneck data (not proportional estimates).
"""

import time
import numpy as np

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import (
    STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT,
    compute_step_flops, compute_step_memory,
)
from paradigms.p06_tpu_systolic.systolic_array import SystolicArray
import config


SIM_ARRAY_SIZE = 16

_tpu_available = None
_tpu_device_name = None
_jit_fns = None


def _has_tpu():
    """Check if JAX with TPU backend is available (cached)."""
    global _tpu_available, _tpu_device_name
    if _tpu_available is not None:
        return _tpu_available

    try:
        import jax
        devices = jax.devices("tpu")
        if len(devices) > 0:
            _tpu_device_name = str(devices[0])
            jax.config.update("jax_default_matmul_precision", "float32")
            print(f"  [TPU systolic] CODE VERSION: {config.BUILD_VERSION}")
            print(f"  [TPU systolic] JAX TPU detected: {_tpu_device_name}")
            print(f"  [TPU systolic] Matmul precision: float32")
            _tpu_available = True
            return True
        print("  [TPU systolic] JAX loaded but no TPU devices found")
        _tpu_available = False
        return False
    except ImportError:
        print("  [TPU systolic] JAX not installed — using simulation")
        _tpu_available = False
        return False
    except RuntimeError as e:
        print(f"  [TPU systolic] JAX TPU error: {e} — using simulation")
        _tpu_available = False
        return False
    except Exception as e:
        print(f"  [TPU systolic] TPU detection failed: {type(e).__name__}: {e}")
        _tpu_available = False
        return False


def _get_jit_fns():
    """Create separate JIT-compiled functions for each attention sub-operation."""
    global _jit_fns
    if _jit_fns is not None:
        return _jit_fns

    import jax
    import jax.numpy as jnp

    @jax.jit
    def jit_qkv(X, Wq, Wk, Wv):
        return X @ Wq, X @ Wk, X @ Wv

    @jax.jit
    def jit_qkt(Q, K):
        d = Q.shape[-1]
        return Q @ K.T / jnp.sqrt(jnp.float32(d))

    @jax.jit
    def jit_softmax(scores):
        return jax.nn.softmax(scores, axis=-1)

    @jax.jit
    def jit_av(attn_weights, V):
        return attn_weights @ V

    @jax.jit
    def jit_out(attended, Wo):
        return attended @ Wo

    _jit_fns = {
        STEP_QKV: jit_qkv,
        STEP_QKT: jit_qkt,
        STEP_SOFTMAX: jit_softmax,
        STEP_AV: jit_av,
        STEP_OUT: jit_out,
    }
    return _jit_fns


def _compute_systolic_cycles(n: int, d: int) -> dict:
    """Compute cycle estimates using realistic TPU array size."""
    array_size = config.TPU_ARRAY_SIZE
    vector_width = config.TPU_VECTOR_WIDTH
    sa = SystolicArray(rows=array_size, cols=array_size)

    qkv_info = sa.matmul_cycles(n, d, d)
    qkt_info = sa.matmul_cycles(n, d, n)
    av_info = sa.matmul_cycles(n, n, d)
    out_info = sa.matmul_cycles(n, d, d)

    softmax_cycles = (5 * n * n + vector_width - 1) // vector_width

    return {
        STEP_QKV: {
            "cycles": qkv_info["total_cycles"] * 3,
            "tiles": qkv_info["total_tiles"] * 3,
            "pe_utilization": qkv_info["pe_utilization"],
        },
        STEP_QKT: {
            "cycles": qkt_info["total_cycles"],
            "tiles": qkt_info["total_tiles"],
            "pe_utilization": qkt_info["pe_utilization"],
            "time_utilization": qkt_info["time_utilization"],
        },
        STEP_SOFTMAX: {
            "cycles": softmax_cycles,
            "note": f"vector unit (width={vector_width})",
        },
        STEP_AV: {
            "cycles": av_info["total_cycles"],
            "tiles": av_info["total_tiles"],
            "pe_utilization": av_info["pe_utilization"],
        },
        STEP_OUT: {
            "cycles": out_info["total_cycles"],
            "tiles": out_info["total_tiles"],
            "pe_utilization": out_info["pe_utilization"],
        },
    }


@register
class TPUSystolicParadigm(BaseParadigm):
    name = "TPU (Systolic Array)"
    short_name = "tpu_systolic"

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        if _has_tpu():
            return self._run_with_real_tpu(X, W_q, W_k, W_v, W_o)
        return self._run_simulation(X, W_q, W_k, W_v, W_o)

    def _run_with_real_tpu(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        """Real TPU: each sub-operation JIT-compiled and timed separately."""
        import jax
        import jax.numpy as jnp

        fns = _get_jit_fns()

        seq_len, embed_dim = X.shape
        n, d = seq_len, embed_dim
        flops = compute_step_flops(n, d)
        mem_est = compute_step_memory(n, d)
        clock_hz = config.SYSTOLIC_CLOCK_HZ
        sim_cycles = _compute_systolic_cycles(n, d)

        X_j = jnp.array(X)
        Wq_j = jnp.array(W_q)
        Wk_j = jnp.array(W_k)
        Wv_j = jnp.array(W_v)
        Wo_j = jnp.array(W_o)

        # Warmup all JIT functions (triggers compilation per shape)
        for _ in range(config.TPU_WARMUP_RUNS + 2):
            Q, K, V = fns[STEP_QKV](X_j, Wq_j, Wk_j, Wv_j)
            jax.block_until_ready((Q, K, V))
            scores = fns[STEP_QKT](Q, K)
            scores.block_until_ready()
            attn = fns[STEP_SOFTMAX](scores)
            attn.block_until_ready()
            attended = fns[STEP_AV](attn, V)
            attended.block_until_ready()
            out = fns[STEP_OUT](attended, Wo_j)
            out.block_until_ready()

        # Benchmark: time each operation separately, repeated for accuracy
        num_iters = config.BENCHMARK_RUNS
        step_times = {s: [] for s in [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]}

        for _ in range(num_iters):
            t0 = time.perf_counter()
            Q, K, V = fns[STEP_QKV](X_j, Wq_j, Wk_j, Wv_j)
            jax.block_until_ready((Q, K, V))
            step_times[STEP_QKV].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            scores = fns[STEP_QKT](Q, K)
            scores.block_until_ready()
            step_times[STEP_QKT].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            attn = fns[STEP_SOFTMAX](scores)
            attn.block_until_ready()
            step_times[STEP_SOFTMAX].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            attended = fns[STEP_AV](attn, V)
            attended.block_until_ready()
            step_times[STEP_AV].append(time.perf_counter() - t0)

            t0 = time.perf_counter()
            out = fns[STEP_OUT](attended, Wo_j)
            out.block_until_ready()
            step_times[STEP_OUT].append(time.perf_counter() - t0)

        total_sim_cycles = sum(sim_cycles[s]["cycles"] for s in sim_cycles)

        steps = []
        for step_name in [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]:
            sc = sim_cycles[step_name]
            r, w = mem_est[step_name]
            best_time = min(step_times[step_name])
            hw_est = sc["cycles"] / clock_hz

            specific = {
                "real_jax_time_s": best_time,
                "simulation_time_s": hw_est,
                "simulated_cycles": sc["cycles"],
                "all_times_s": step_times[step_name],
            }
            if "pe_utilization" in sc:
                specific["pe_utilization"] = f"{sc['pe_utilization']:.2%}"
            if "time_utilization" in sc:
                specific["time_utilization"] = f"{sc['time_utilization']:.2%}"
            if "note" in sc:
                specific["note"] = sc["note"]

            steps.append(StepMetrics(
                name=step_name, flops=flops[step_name],
                memory_reads_bytes=r, memory_writes_bytes=w,
                wall_time_seconds=best_time,
                estimated_hw_time_seconds=0,
                paradigm_specific=specific,
            ))

        decomposed_total = sum(min(step_times[s]) for s in step_times)

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "real_tpu_jax",
                "tpu_device": _tpu_device_name,
                "matmul_precision": "float32",
                "timing_method": "separate_jit_per_operation",
                "array_size": f"{config.TPU_ARRAY_SIZE}x{config.TPU_ARRAY_SIZE}",
                "clock_hz": clock_hz,
                "decomposed_total_time_s": decomposed_total,
                "simulation_time_s": total_sim_cycles / clock_hz,
                "num_timing_iterations": num_iters,
                "framework": "JAX (per-op JIT, decomposed timing)",
            },
        )

    def _run_simulation(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        """Systolic array simulation with realistic TPU-scale estimates."""
        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem_est = compute_step_memory(seq_len, embed_dim)
        n, d = seq_len, embed_dim
        clock_hz = config.SYSTOLIC_CLOCK_HZ
        array_size = config.TPU_ARRAY_SIZE

        sim_cycles = _compute_systolic_cycles(n, d)

        sa_small = SystolicArray(rows=SIM_ARRAY_SIZE, cols=SIM_ARRAY_SIZE)
        steps = []

        t0 = time.perf_counter()
        Q, _ = sa_small.simulate_matmul(X, W_q)
        K, _ = sa_small.simulate_matmul(X, W_k)
        V, _ = sa_small.simulate_matmul(X, W_v)
        t1 = time.perf_counter()

        sc = sim_cycles[STEP_QKV]
        r, w = mem_est[STEP_QKV]
        steps.append(StepMetrics(
            name=STEP_QKV, flops=flops[STEP_QKV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sc["cycles"] / clock_hz,
            paradigm_specific={
                "systolic_cycles": sc["cycles"],
                "tiles": sc["tiles"],
                "pe_utilization": f"{sc['pe_utilization']:.2%}",
            },
        ))

        t0 = time.perf_counter()
        scores, _ = sa_small.simulate_matmul(Q, K.T)
        scores = scores / np.sqrt(d)
        t1 = time.perf_counter()

        sc = sim_cycles[STEP_QKT]
        r, w = mem_est[STEP_QKT]
        steps.append(StepMetrics(
            name=STEP_QKT, flops=flops[STEP_QKT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sc["cycles"] / clock_hz,
            paradigm_specific={
                "systolic_cycles": sc["cycles"],
                "pe_utilization": f"{sc['pe_utilization']:.2%}",
                "time_utilization": f"{sc['time_utilization']:.2%}",
            },
        ))

        t0 = time.perf_counter()
        scores_max = scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
        t1 = time.perf_counter()

        sc = sim_cycles[STEP_SOFTMAX]
        r, w = mem_est[STEP_SOFTMAX]
        steps.append(StepMetrics(
            name=STEP_SOFTMAX, flops=flops[STEP_SOFTMAX],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sc["cycles"] / clock_hz,
            paradigm_specific={
                "note": sc["note"],
                "estimated_cycles": sc["cycles"],
            },
        ))

        t0 = time.perf_counter()
        attended, _ = sa_small.simulate_matmul(attention.astype(np.float32), V)
        t1 = time.perf_counter()

        sc = sim_cycles[STEP_AV]
        r, w = mem_est[STEP_AV]
        steps.append(StepMetrics(
            name=STEP_AV, flops=flops[STEP_AV],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sc["cycles"] / clock_hz,
            paradigm_specific={
                "systolic_cycles": sc["cycles"],
                "pe_utilization": f"{sc['pe_utilization']:.2%}",
            },
        ))

        t0 = time.perf_counter()
        output, _ = sa_small.simulate_matmul(attended, W_o)
        t1 = time.perf_counter()

        sc = sim_cycles[STEP_OUT]
        r, w = mem_est[STEP_OUT]
        steps.append(StepMetrics(
            name=STEP_OUT, flops=flops[STEP_OUT],
            memory_reads_bytes=r, memory_writes_bytes=w,
            wall_time_seconds=t1 - t0,
            estimated_hw_time_seconds=sc["cycles"] / clock_hz,
            paradigm_specific={
                "systolic_cycles": sc["cycles"],
                "pe_utilization": f"{sc['pe_utilization']:.2%}",
            },
        ))

        total_cycles = sum(sim_cycles[s]["cycles"] for s in sim_cycles)

        return ParadigmResult(
            paradigm_name=self.name,
            paradigm_short=self.short_name,
            seq_len=seq_len,
            embed_dim=embed_dim,
            steps=steps,
            paradigm_info={
                "execution_mode": "simulation",
                "array_size": f"{array_size}x{array_size}",
                "total_systolic_cycles": total_cycles,
                "estimated_hw_time_s": total_cycles / clock_hz,
                "clock_hz": clock_hz,
                "peak_ops_per_cycle": array_size * array_size * 2,
                "vector_width": config.TPU_VECTOR_WIDTH,
                "execution_model": "weight-stationary systolic array",
            },
        )
