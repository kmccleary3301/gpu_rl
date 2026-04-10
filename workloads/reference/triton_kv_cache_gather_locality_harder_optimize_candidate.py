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

LOCALITY_HIDDEN_CACHE = torch.tensor(
    [
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
        [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9],
        [2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9],
        [3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9],
        [4.0, 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8, 4.9],
        [5.0, 5.1, 5.2, 5.3, 5.4, 5.5, 5.6, 5.7, 5.8, 5.9],
        [6.0, 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9],
        [7.0, 7.1, 7.2, 7.3, 7.4, 7.5, 7.6, 7.7, 7.8, 7.9],
        [8.0, 8.1, 8.2, 8.3, 8.4, 8.5, 8.6, 8.7, 8.8, 8.9],
        [9.0, 9.1, 9.2, 9.3, 9.4, 9.5, 9.6, 9.7, 9.8, 9.9],
    ],
    dtype=torch.float32,
)
LOCALITY_HIDDEN_PAGE_IDS = torch.tensor([8, 8, 8, 2, 2, 5, 5, 5], dtype=torch.int32)


def _benchmark_inputs(num_pages: int, seq_len: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    cache = (torch.arange(num_pages * head_dim, device="cuda", dtype=torch.float32).reshape(num_pages, head_dim) % 353) / 41.0
    cluster_ids = torch.arange(seq_len, device="cuda", dtype=torch.int32) // 8
    page_ids = ((cluster_ids * 5) + 11) % num_pages
    return cache, page_ids


def _run_benchmark(repeats: int) -> None:
    cache, page_ids = _benchmark_inputs(num_pages=1024, seq_len=8192, head_dim=256)
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
                "hidden_kv_rows": _serialize(LOCALITY_HIDDEN_CACHE, LOCALITY_HIDDEN_PAGE_IDS),
                "optimization_summary": {
                    "strategy_change": "replace_nonlocal_cpu_indexing_path_with_triton_locality_aware_gather_candidate",
                    "candidate_ref": "workloads/reference/triton_kv_cache_gather_locality_harder_optimize_candidate.py",
                    "kernel_ref": "workloads/reference/triton_kv_cache_gather_kernel.py",
                    "baseline_ref": "workloads/reference/triton_kv_cache_gather_baseline.py",
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
