from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_kv_cache_gather_kernel import triton_kv_cache_gather


VISIBLE_CACHE = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0],
        [13.0, 14.0, 15.0, 16.0],
    ],
    dtype=torch.float32,
)
VISIBLE_PAGE_IDS = torch.tensor([2, 0, 3], dtype=torch.int32)

HARD_HIDDEN_CACHE = torch.tensor(
    [
        [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        [3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
        [6.0, 6.5, 7.0, 7.5, 8.0, 8.5],
        [9.0, 9.5, 10.0, 10.5, 11.0, 11.5],
        [12.0, 12.5, 13.0, 13.5, 14.0, 14.5],
        [15.0, 15.5, 16.0, 16.5, 17.0, 17.5],
    ],
    dtype=torch.float32,
)
HARD_HIDDEN_PAGE_IDS = torch.tensor([4, 1, 4, 5, 2], dtype=torch.int32)


def _benchmark_inputs(num_pages: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = (torch.arange(num_pages * head_dim, device="cuda", dtype=torch.float32).reshape(num_pages, head_dim) % 257) / 29.0
    page_ids = ((torch.arange(seq_len, device="cuda", dtype=torch.int32) * 11) + 5) % num_pages
    return cache, page_ids


def _run_benchmark(repeats: int) -> None:
    cache, page_ids = _benchmark_inputs(num_pages=1536, seq_len=6144, head_dim=192)
    for _ in range(repeats):
        out = triton_kv_cache_gather(cache, page_ids)
        torch.cuda.synchronize()
    _ = out


def _serialize(cache: torch.Tensor, page_ids: torch.Tensor) -> list[list[float]]:
    out = triton_kv_cache_gather(cache.cuda(), page_ids.cuda()).cpu()
    return [[float(value) for value in row] for row in out.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_kv_rows": _serialize(VISIBLE_CACHE, VISIBLE_PAGE_IDS),
                "hidden_kv_rows": _serialize(HARD_HIDDEN_CACHE, HARD_HIDDEN_PAGE_IDS),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
