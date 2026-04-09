from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_attention_score_kernel import triton_attention_scores
from triton_attention_score_hard_shared import HIDDEN_K, HIDDEN_Q, VISIBLE_K, VISIBLE_Q


def _benchmark_inputs(n_q: int, n_k: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_base = torch.arange(n_q * head_dim, device="cuda", dtype=torch.float32)
    k_base = torch.arange(n_k * head_dim, device="cuda", dtype=torch.float32)
    q = ((q_base.reshape(n_q, head_dim) * 13) % 257) / 257.0
    k = ((k_base.reshape(n_k, head_dim) * 29) % 251) / 251.0
    return q, k


def _run_benchmark(repeats: int) -> None:
    q, k = _benchmark_inputs(n_q=768, n_k=768, head_dim=128)
    for _ in range(repeats):
        out = triton_attention_scores(q, k, causal=True)
        torch.cuda.synchronize()
    _ = out


def _serialize(q: torch.Tensor, k: torch.Tensor) -> list[list[float]]:
    scores = triton_attention_scores(q.cuda(), k.cuda(), causal=True).cpu()
    return [[round(float(value), 6) for value in row] for row in scores.tolist()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_attention_scores": _serialize(VISIBLE_Q, VISIBLE_K),
                "hidden_attention_scores": _serialize(HIDDEN_Q, HIDDEN_K)
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
