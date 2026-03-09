from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_attention_score_shared import HIDDEN_K, HIDDEN_Q, MASK_FILL, VISIBLE_K, VISIBLE_Q


def _naive_scores(q: torch.Tensor, k: torch.Tensor) -> list[list[float]]:
    rows: list[list[float]] = []
    scale = 1.0 / math.sqrt(float(q.shape[1]))
    for row_idx in range(int(q.shape[0])):
        row_scores: list[float] = []
        for col_idx in range(int(k.shape[0])):
            if col_idx > row_idx:
                row_scores.append(MASK_FILL)
                continue
            total = 0.0
            for dim_idx in range(int(q.shape[1])):
                total += float(q[row_idx, dim_idx]) * float(k[col_idx, dim_idx])
            row_scores.append(total * scale)
        rows.append(row_scores)
    return rows


def _benchmark_inputs(n_q: int, n_k: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_base = torch.arange(n_q * head_dim, dtype=torch.float32)
    k_base = torch.arange(n_k * head_dim, dtype=torch.float32)
    q = ((q_base.reshape(n_q, head_dim) * 13) % 257) / 257.0
    k = ((k_base.reshape(n_k, head_dim) * 29) % 251) / 251.0
    return q, k


def _run_benchmark(repeats: int) -> None:
    q, k = _benchmark_inputs(n_q=80, n_k=80, head_dim=32)
    for _ in range(repeats):
        _ = _naive_scores(q, k)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=2)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_attention_scores": _naive_scores(VISIBLE_Q, VISIBLE_K),
                "hidden_attention_scores": _naive_scores(HIDDEN_Q, HIDDEN_K),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
