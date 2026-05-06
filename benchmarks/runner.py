"""Benchmark runner - iterates paradigms x configs, collects results."""

import time
from core.attention import generate_inputs
from core.metrics import ParadigmResult
from core.base_paradigm import BaseParadigm
import config


def run_single(paradigm: BaseParadigm, seq_len: int, embed_dim: int) -> ParadigmResult:
    """Run one paradigm on one config, with warmup."""
    X, W_q, W_k, W_v, W_o = generate_inputs(seq_len, embed_dim)

    # Warmup
    for _ in range(config.WARMUP_RUNS):
        paradigm.run_attention(X, W_q, W_k, W_v, W_o)

    # Benchmark (take the best run)
    best = None
    for _ in range(config.BENCHMARK_RUNS):
        result = paradigm.run_attention(X, W_q, W_k, W_v, W_o)
        if best is None or result.total_wall_time < best.total_wall_time:
            best = result

    return best


def run_benchmark(
    paradigms: list[BaseParadigm] = None,
    seq_lens: list[int] = None,
    embed_dims: list[int] = None,
) -> list[ParadigmResult]:
    """Run all paradigm x config combinations."""
    from core.registry import get_available_paradigms

    paradigms = paradigms or get_available_paradigms()
    seq_lens = seq_lens or config.SEQ_LENS
    embed_dims = embed_dims or config.EMBED_DIMS

    results = []

    for paradigm in paradigms:
        # Determine size limits for simulators
        max_seq = getattr(paradigm, 'max_seq_len', None)
        max_dim = getattr(paradigm, 'max_embed_dim', None)

        # Use paradigm-specific sizes if it has limits
        p_seq_lens = [s for s in seq_lens if not max_seq or s <= max_seq]
        p_embed_dims = [d for d in embed_dims if not max_dim or d <= max_dim]

        # If all standard sizes exceed limits, use the max allowed sizes
        if not p_seq_lens and max_seq:
            p_seq_lens = [max_seq]
        if not p_embed_dims and max_dim:
            p_embed_dims = [max_dim]

        for seq_len in p_seq_lens:
            for embed_dim in p_embed_dims:
                print(f"  {paradigm.short_name} "
                      f"seq={seq_len} dim={embed_dim}...", end=" ", flush=True)

                t0 = time.perf_counter()
                result = run_single(paradigm, seq_len, embed_dim)
                elapsed = time.perf_counter() - t0

                print(f"{result.total_wall_time:.4f}s (total {elapsed:.1f}s)")
                results.append(result)

    return results
