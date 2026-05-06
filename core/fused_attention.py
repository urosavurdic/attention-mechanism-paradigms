"""Fused attention using torch.nn.functional.scaled_dot_product_attention."""

import torch
import torch.nn.functional as F

import config


def run_fused_sdpa(X_t, W_q_t, W_k_t, W_v_t, W_o_t):
    """Full attention using F.scaled_dot_product_attention with multi-head reshape.

    Args:
        X_t: (seq_len, embed_dim) tensor on target device
        W_q_t, W_k_t, W_v_t, W_o_t: (embed_dim, embed_dim) tensors on target device

    Returns:
        output: (seq_len, embed_dim) tensor
    """
    seq_len, embed_dim = X_t.shape
    num_heads = config.NUM_HEADS
    head_dim = embed_dim // num_heads

    Q = (X_t @ W_q_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    K = (X_t @ W_k_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    V = (X_t @ W_v_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)

    attn_out = F.scaled_dot_product_attention(Q, K, V)

    out = attn_out.squeeze(0).transpose(0, 1).reshape(seq_len, embed_dim)
    output = out @ W_o_t
    return output


def detect_sdpa_backend(X_t, W_q_t, W_k_t, W_v_t, dtype):
    """Detect which SDPA backend PyTorch actually dispatches to.

    Tries flash → mem_efficient → falls back to math.
    Returns a string naming the backend.
    """
    seq_len, embed_dim = X_t.shape
    num_heads = config.NUM_HEADS
    head_dim = embed_dim // num_heads

    Q = (X_t @ W_q_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    K = (X_t @ W_k_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)
    V = (X_t @ W_v_t).view(seq_len, num_heads, head_dim).transpose(0, 1).unsqueeze(0)

    if dtype != torch.float32:
        Q, K, V = Q.to(dtype), K.to(dtype), V.to(dtype)

    # Try each backend in isolation to see which one succeeds
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        backend_map = [
            ("flash_attention", [SDPBackend.FLASH_ATTENTION]),
            ("mem_efficient", [SDPBackend.EFFICIENT_ATTENTION]),
        ]
        for name, backends in backend_map:
            try:
                with sdpa_kernel(backends):
                    F.scaled_dot_product_attention(Q, K, V)
                    return name
            except RuntimeError:
                continue
        return "math (fallback)"
    except ImportError:
        pass

    # Older PyTorch: use sdp_kernel context manager
    try:
        for name, kwargs in [
            ("flash_attention", dict(enable_flash=True, enable_mem_efficient=False, enable_math=False)),
            ("mem_efficient", dict(enable_flash=False, enable_mem_efficient=True, enable_math=False)),
        ]:
            try:
                with torch.backends.cuda.sdp_kernel(**kwargs):
                    F.scaled_dot_product_attention(Q, K, V)
                    return name
            except RuntimeError:
                continue
    except Exception:
        pass

    return "math (fallback)"


def is_sdpa_available() -> bool:
    return hasattr(F, "scaled_dot_product_attention")
