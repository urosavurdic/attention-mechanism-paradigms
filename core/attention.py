"""Reference attention decomposition, FLOP counting, and input generation."""

import numpy as np

# Step names (used as keys across all paradigms)
STEP_QKV = "qkv_projection"
STEP_QKT = "qk_matmul"
STEP_SOFTMAX = "softmax"
STEP_AV = "av_matmul"
STEP_OUT = "output_projection"

ALL_STEPS = [STEP_QKV, STEP_QKT, STEP_SOFTMAX, STEP_AV, STEP_OUT]


def generate_inputs(seq_len: int, embed_dim: int, seed: int = 42):
    """Generate deterministic random inputs for attention.

    Returns:
        X: (seq_len, embed_dim)
        W_q, W_k, W_v, W_o: (embed_dim, embed_dim)
    """
    rng = np.random.default_rng(seed)
    scale = 1.0 / np.sqrt(embed_dim)
    X = rng.standard_normal((seq_len, embed_dim)).astype(np.float32) * scale
    W_q = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
    W_k = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
    W_v = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
    W_o = rng.standard_normal((embed_dim, embed_dim)).astype(np.float32) * scale
    return X, W_q, W_k, W_v, W_o


def reference_attention(X, W_q, W_k, W_v, W_o):
    """NumPy reference implementation. Returns output and intermediates."""
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)
    # Numerically stable softmax
    scores_max = scores.max(axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attention = exp_scores / exp_scores.sum(axis=-1, keepdims=True)
    attended = attention @ V
    output = attended @ W_o
    return output


# --- Analytical FLOP and memory estimates ---

def flops_matmul(M: int, K: int, N: int) -> int:
    """FLOPs for (M,K) @ (K,N) matmul: 2*M*K*N (multiply + add)."""
    return 2 * M * K * N


def memory_matmul(M: int, K: int, N: int, dtype_bytes: int = 4) -> tuple[int, int]:
    """Memory reads/writes for (M,K) @ (K,N) in bytes.

    Returns:
        (reads, writes)
    """
    reads = (M * K + K * N) * dtype_bytes
    writes = M * N * dtype_bytes
    return reads, writes


def flops_softmax(N: int, seq_len: int) -> int:
    """FLOPs for softmax over (seq_len, seq_len) matrix.

    Per row: max (N-1 comparisons) + sub (N) + exp (N) + sum (N-1) + div (N)
    ~= 5*N ops per row, N rows.
    """
    return 5 * seq_len * seq_len


def memory_softmax(seq_len: int, dtype_bytes: int = 4) -> tuple[int, int]:
    """Memory for softmax: read scores, write attention weights."""
    reads = seq_len * seq_len * dtype_bytes
    writes = seq_len * seq_len * dtype_bytes
    return reads, writes


def compute_step_flops(seq_len: int, embed_dim: int) -> dict[str, int]:
    """Analytical FLOP count per attention step."""
    n, d = seq_len, embed_dim
    return {
        STEP_QKV: 3 * flops_matmul(n, d, d),  # Q, K, V projections
        STEP_QKT: flops_matmul(n, d, n),       # Q @ K^T
        STEP_SOFTMAX: flops_softmax(n, n),      # softmax
        STEP_AV: flops_matmul(n, n, d),         # attention @ V
        STEP_OUT: flops_matmul(n, d, d),        # output projection
    }


def compute_step_memory(seq_len: int, embed_dim: int, dtype_bytes: int = 4) -> dict[str, tuple[int, int]]:
    """Analytical memory (reads, writes) per attention step in bytes."""
    n, d = seq_len, embed_dim
    qkv_r = 3 * (n * d + d * d) * dtype_bytes
    qkv_w = 3 * (n * d) * dtype_bytes
    qkt_r, qkt_w = memory_matmul(n, d, n, dtype_bytes)
    sm_r, sm_w = memory_softmax(n, dtype_bytes)
    av_r, av_w = memory_matmul(n, n, d, dtype_bytes)
    out_r, out_w = memory_matmul(n, d, d, dtype_bytes)
    return {
        STEP_QKV: (qkv_r, qkv_w),
        STEP_QKT: (qkt_r, qkt_w),
        STEP_SOFTMAX: (sm_r, sm_w),
        STEP_AV: (av_r, av_w),
        STEP_OUT: (out_r, out_w),
    }
