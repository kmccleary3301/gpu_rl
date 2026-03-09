from __future__ import annotations

import triton
import triton.language as tl
import torch


@triton.jit
def routing_argmax_kernel(x_ptr, out_idx_ptr, out_val_ptr, stride_row, n_cols, BLOCK_SIZE: tl.constexpr):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < n_cols
    row_ptr = x_ptr + row * stride_row + cols
    values = tl.load(row_ptr, mask=mask, other=-float("inf"))
    max_vals, max_idx = tl.max(values, axis=0, return_indices=True, return_indices_tie_break_left=True)
    tl.store(out_idx_ptr + row, max_idx)
    tl.store(out_val_ptr + row, max_vals)


def triton_routing_argmax(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if not x.is_cuda:
        raise ValueError("Expected a CUDA tensor")
    rows, cols = x.shape
    out_idx = torch.empty((rows,), device=x.device, dtype=torch.int32)
    out_val = torch.empty((rows,), device=x.device, dtype=x.dtype)
    block_size = triton.next_power_of_2(cols)
    routing_argmax_kernel[(rows,)](
        x,
        out_idx,
        out_val,
        x.stride(0),
        cols,
        BLOCK_SIZE=block_size,
    )
    return out_idx, out_val


def get_build_spec() -> dict[str, object]:
    rows, cols = 8, 128
    sample = torch.empty((rows, cols), device="cuda", dtype=torch.float32)
    out_idx = torch.empty((rows,), device="cuda", dtype=torch.int32)
    out_val = torch.empty((rows,), device="cuda", dtype=torch.float32)
    return {
        "kernel": routing_argmax_kernel,
        "warmup_args": [sample, out_idx, out_val, sample.stride(0), cols],
        "grid": (rows,),
        "kwargs": {"BLOCK_SIZE": triton.next_power_of_2(cols)},
        "source_file": __file__,
    }
