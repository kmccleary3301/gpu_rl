from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def attention_score_kernel(
    q_ptr,
    k_ptr,
    out_ptr,
    q_row_stride,
    q_col_stride,
    k_row_stride,
    k_col_stride,
    out_row_stride,
    out_col_stride,
    n_q,
    n_k,
    head_dim,
    scale,
    mask_fill,
    BLOCK_K: tl.constexpr,
    BLOCK_D: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    row = tl.program_id(0)
    col_block = tl.program_id(1)
    cols = col_block * BLOCK_K + tl.arange(0, BLOCK_K)
    dims = tl.arange(0, BLOCK_D)

    q_ptrs = q_ptr + row * q_row_stride + dims * q_col_stride
    q_values = tl.load(q_ptrs, mask=dims < head_dim, other=0.0)

    k_ptrs = k_ptr + cols[:, None] * k_row_stride + dims[None, :] * k_col_stride
    k_values = tl.load(k_ptrs, mask=(cols[:, None] < n_k) & (dims[None, :] < head_dim), other=0.0)
    scores = tl.sum(k_values * q_values[None, :], axis=1) * scale
    if CAUSAL:
        scores = tl.where(cols <= row, scores, mask_fill)

    out_ptrs = out_ptr + row * out_row_stride + cols * out_col_stride
    tl.store(out_ptrs, scores, mask=(row < n_q) & (cols < n_k))


def triton_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
) -> torch.Tensor:
    if not q.is_cuda or not k.is_cuda:
        raise ValueError("Expected CUDA tensors for q and k")
    if q.ndim != 2 or k.ndim != 2:
        raise ValueError("Expected q[n_q, head_dim] and k[n_k, head_dim]")
    if q.shape[1] != k.shape[1]:
        raise ValueError("Expected matching head dimensions for q and k")
    n_q = int(q.shape[0])
    n_k = int(k.shape[0])
    head_dim = int(q.shape[1])
    out = torch.empty((n_q, n_k), device=q.device, dtype=torch.float32)
    block_k = 32
    block_d = triton.next_power_of_2(head_dim)
    attention_score_kernel[(n_q, triton.cdiv(n_k, block_k))](
        q,
        k,
        out,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        out.stride(0),
        out.stride(1),
        n_q,
        n_k,
        head_dim,
        1.0 / math.sqrt(float(head_dim)),
        -1.0e9,
        BLOCK_K=block_k,
        BLOCK_D=block_d,
        CAUSAL=causal,
    )
    return out


def get_build_spec() -> dict[str, object]:
    n_q, n_k, head_dim = 8, 16, 32
    q = torch.empty((n_q, head_dim), device="cuda", dtype=torch.float32)
    k = torch.empty((n_k, head_dim), device="cuda", dtype=torch.float32)
    out = torch.empty((n_q, n_k), device="cuda", dtype=torch.float32)
    block_k = 32
    return {
        "kernel": attention_score_kernel,
        "warmup_args": [
            q,
            k,
            out,
            q.stride(0),
            q.stride(1),
            k.stride(0),
            k.stride(1),
            out.stride(0),
            out.stride(1),
            n_q,
            n_k,
            head_dim,
            1.0 / math.sqrt(float(head_dim)),
            -1.0e9,
        ],
        "grid": (n_q, triton.cdiv(n_k, block_k)),
        "kwargs": {
            "BLOCK_K": block_k,
            "BLOCK_D": triton.next_power_of_2(head_dim),
            "CAUSAL": True,
        },
        "source_file": __file__,
    }
