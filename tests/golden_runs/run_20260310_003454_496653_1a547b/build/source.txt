from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def row_sum_broken_kernel(x_ptr, out_ptr, stride_row, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    effective_cols = tl.maximum(n_cols - 1, 0)
    mask = cols < effective_cols
    row_ptr = x_ptr + row * stride_row + cols
    values = tl.load(row_ptr, mask=mask, other=0.0)
    total = tl.sum(values, axis=0)
    tl.store(out_ptr + row, total)


def triton_row_sum_broken(x: torch.Tensor) -> torch.Tensor:
    if not x.is_cuda:
        raise ValueError("Expected a CUDA tensor")
    rows, cols = x.shape
    out = torch.empty((rows,), device=x.device, dtype=x.dtype)
    block_size = triton.next_power_of_2(cols)
    row_sum_broken_kernel[(rows,)](
        x,
        out,
        x.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )
    return out


def get_build_spec() -> dict[str, object]:
    rows, cols = 8, 128
    sample = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    out = torch.empty((rows,), device="cuda", dtype=torch.float32)
    return {
        "kernel": row_sum_broken_kernel,
        "warmup_args": [sample, out, sample.stride(0), cols],
        "grid": (rows,),
        "kwargs": {"BLOCK_SIZE": triton.next_power_of_2(cols)},
        "source_file": __file__,
    }
