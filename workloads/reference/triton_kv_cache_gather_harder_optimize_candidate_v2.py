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

HARDER_HIDDEN_CACHE = torch.tensor(
    [
        [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75],
        [4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75],
        [6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75],
        [8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75],
        [10.0, 10.25, 10.5, 10.75, 11.0, 11.25, 11.5, 11.75],
        [12.0, 12.25, 12.5, 12.75, 13.0, 13.25, 13.5, 13.75],
        [14.0, 14.25, 14.5, 14.75, 15.0, 15.25, 15.5, 15.75],
    ],
    dtype=torch.float32,
)
HARDER_HIDDEN_PAGE_IDS = torch.tensor([6, 2, 6, 7, 3, 5], dtype=torch.int32)


def _benchmark_inputs(num_pages: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = (torch.arange(num_pages * head_dim, device="cuda", dtype=torch.float32).reshape(num_pages, head_dim) % 257) / 29.0
    page_ids = ((torch.arange(seq_len, device="cuda", dtype=torch.int32) * 13) + 7) % num_pages
    return cache, page_ids


def _run_benchmark(repeats: int) -> None:
    cache, page_ids = _benchmark_inputs(num_pages=1792, seq_len=7168, head_dim=224)
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
                "hidden_kv_rows": _serialize(HARDER_HIDDEN_CACHE, HARDER_HIDDEN_PAGE_IDS),
                "optimization_summary": {
                    "strategy_change": "supersede_triton_paged_gather_harder_candidate_with_ranked_variant",
                    "candidate_ref": "workloads/reference/triton_kv_cache_gather_harder_optimize_candidate_v2.py",
                    "kernel_ref": "workloads/reference/triton_kv_cache_gather_kernel.py",
                    "baseline_ref": "workloads/reference/triton_kv_cache_gather_baseline.py",
                    "supersedes_candidate_ref": "workloads/reference/triton_kv_cache_gather_harder_optimize_candidate.py",
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
