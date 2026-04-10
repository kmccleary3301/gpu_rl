from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_attention_score_harder_shared import HARDER_K, HARDER_Q, VISIBLE_K, VISIBLE_Q, reference_attention_scores, reference_attention_tensor


def _benchmark_inputs(n_q: int, n_k: int, head_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    q_base = torch.arange(n_q * head_dim, dtype=torch.float32)
    k_base = torch.arange(n_k * head_dim, dtype=torch.float32)
    q = ((q_base.reshape(n_q, head_dim) * 17) % 263) / 263.0
    k = ((k_base.reshape(n_k, head_dim) * 31) % 257) / 257.0
    return q, k


def _run_benchmark(repeats: int) -> None:
    q, k = _benchmark_inputs(n_q=896, n_k=896, head_dim=160)
    for _ in range(repeats):
        for _baseline_pass in range(4):
            out = reference_attention_tensor(q, k, causal=True)
    _ = out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=20)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_attention_scores": reference_attention_scores(VISIBLE_Q, VISIBLE_K, causal=True),
                "hidden_attention_scores": reference_attention_scores(HARDER_Q, HARDER_K, causal=True),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
