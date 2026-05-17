"""P06b: TPU Fused SDPA - End-to-end fused attention on TPU via JAX.

Real TPU mode: the entire attention mechanism (QKV + QK^T + softmax + AV + output)
runs as a single JIT-compiled function. This shows the fused performance where XLA
can optimize across operation boundaries. Compare with tpu_systolic (decomposed)
to quantify the fusion benefit.
"""

import time
import numpy as np
import torch

from core.base_paradigm import BaseParadigm
from core.registry import register
from core.metrics import ParadigmResult, StepMetrics
from core.attention import compute_step_flops, compute_step_memory
from core.fused_attention import run_fused_sdpa, is_sdpa_available
import config


WARMUP_RUNS = 3

_tpu_available = None
_tpu_device_name = None
_jax_fused_fn = None


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
            print(f"  [TPU fused] CODE VERSION: {config.BUILD_VERSION}")
            print(f"  [TPU fused] JAX TPU detected: {_tpu_device_name}")
            print(f"  [TPU fused] Matmul precision: float32")
            _tpu_available = True
            return True
        print("  [TPU fused] JAX loaded but no TPU devices found")
        _tpu_available = False
        return False
    except ImportError:
        print("  [TPU fused] JAX not installed — using CPU fallback")
        _tpu_available = False
        return False
    except Exception as e:
        print(f"  [TPU fused] TPU detection failed: {type(e).__name__}: {e}")
        _tpu_available = False
        return False


def _get_jax_fused_fn():
    """Return a JIT-compiled fused attention function."""
    global _jax_fused_fn
    if _jax_fused_fn is not None:
        return _jax_fused_fn

    import jax
    import jax.numpy as jnp

    @jax.jit
    def fused_attention(X, Wq, Wk, Wv, Wo):
        Q, K, V = X @ Wq, X @ Wk, X @ Wv
        d = X.shape[-1]
        scores = Q @ K.T / jnp.sqrt(jnp.float32(d))
        attn = jax.nn.softmax(scores, axis=-1)
        return attn @ V @ Wo

    _jax_fused_fn = fused_attention
    return _jax_fused_fn


@register
class TPUFusedParadigm(BaseParadigm):
    name = "TPU (Fused SDPA)"
    short_name = "tpu_fused"

    def is_available(self) -> bool:
        return is_sdpa_available()

    def run_attention(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        if _has_tpu():
            return self._run_tpu(X, W_q, W_k, W_v, W_o)
        return self._run_cpu_fallback(X, W_q, W_k, W_v, W_o)

    def _run_tpu(self, X, W_q, W_k, W_v, W_o) -> ParadigmResult:
        """End-to-end fused attention on real TPU via JAX."""
        import jax
        import jax.numpy as jnp

        fn = _get_jax_fused_fn()

        seq_len, embed_dim = X.shape
        flops = compute_step_flops(seq_len, embed_dim)
        mem = compute_step_memory(seq_len, embed_dim)
        total_flops = sum(flops.values())
        total_reads = sum(r for r, _ in mem.values())
        total_writes = sum(w for _, w in mem.values())

        X_j = jnp.array(X)
        Wq_j = jnp.array(W_q)
        Wk_j = jnp.array(W_k)
        Wv_j = jnp.array(W_v)
        Wo_j = jnp.array(W_o)

        for _ in range(config.TPU_WARMUP_RUNS + 2):
            out = fn(X_j, Wq_j, Wk_j, Wv_j, Wo_j)
            out.block_until_ready()

        fused_times = []
        for _ in range(config.BENCHMARK_RUNS):
            t0 = time.perf_counter()
            out = fn(X_j, Wq_j, Wk_j, Wv_j, Wo_j)
            out.block_until_ready()
            fused_times.append(time.perf_counter() - t0)
        wall = min(fused_times)

        steps = [StepMetrics(
            name="fused_attention",
            flops=total_flops,
            memory_reads_bytes=total_reads,
            memory_writes_bytes=total_writes,
            wall_time_seconds=wall,
            estimated_hw_time_seconds=0,
            paradigm_specific={
                "real_tpu_time_s": wall,
                "all_times_s": fused_times,
                "num_timing_iterations": config.BENCHMARK_RUNS,
            },
        )]

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
                "timing_method": "single_fused_jit",
                "backend": "JAX fused attention (XLA-optimized)",
                "framework": "JAX (single JIT, fused timing)",
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
                "note": "No TPU available, ran on CPU",
            },
        )
