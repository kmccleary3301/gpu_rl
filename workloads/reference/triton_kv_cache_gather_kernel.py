from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def kv_cache_gather_kernel(
    cache_ptr,
    page_ids_ptr,
    out_ptr,
    cache_row_stride,
    out_row_stride,
    head_dim,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < head_dim
    page_id = tl.load(page_ids_ptr + row)
    cache_row_ptr = cache_ptr + page_id * cache_row_stride + cols
    out_row_ptr = out_ptr + row * out_row_stride + cols
    values = tl.load(cache_row_ptr, mask=mask, other=0.0)
    tl.store(out_row_ptr, values, mask=mask)


def triton_kv_cache_gather(cache: torch.Tensor, page_ids: torch.Tensor) -> torch.Tensor:
    if not cache.is_cuda or not page_ids.is_cuda:
        raise ValueError("Expected CUDA tensors for cache and page_ids")
    if cache.ndim != 2 or page_ids.ndim != 1:
        raise ValueError("Expected cache[num_pages, head_dim] and page_ids[seq_len]")
    seq_len = int(page_ids.shape[0])
    head_dim = int(cache.shape[1])
    out = torch.empty((seq_len, head_dim), device=cache.device, dtype=cache.dtype)
    block_size = triton.next_power_of_2(head_dim)
    kv_cache_gather_kernel[(seq_len,)](
        cache,
        page_ids,
        out,
        cache.stride(0),
        out.stride(0),
        head_dim,
        BLOCK_SIZE=block_size,
    )
    return out


def get_build_spec() -> dict[str, object]:
    num_pages, head_dim, seq_len = 16, 128, 8
    cache = torch.empty((num_pages, head_dim), device="cuda", dtype=torch.float32)
    page_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32) % num_pages
    out = torch.empty((seq_len, head_dim), device="cuda", dtype=torch.float32)
    return {
        "kernel": kv_cache_gather_kernel,
        "warmup_args": [cache, page_ids, out, cache.stride(0), out.stride(0), head_dim],
        "grid": (seq_len,),
        "kwargs": {"BLOCK_SIZE": triton.next_power_of_2(head_dim)},
        "source_file": __file__,
    }
