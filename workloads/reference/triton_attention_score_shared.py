from __future__ import annotations

import math

import torch

MASK_FILL = -1.0e9

VISIBLE_Q = torch.tensor(
    [
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ],
    dtype=torch.float32,
)
VISIBLE_K = torch.tensor(
    [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
    ],
    dtype=torch.float32,
)

HIDDEN_Q = torch.tensor(
    [
        [1.0, -1.0, 0.5, 2.0],
        [0.0, 1.5, -0.5, 1.0],
        [2.0, 0.5, 1.0, -1.0],
        [-1.0, 2.0, 0.0, 0.5],
    ],
    dtype=torch.float32,
)
HIDDEN_K = torch.tensor(
    [
        [0.5, 1.0, -1.0, 2.0],
        [1.5, -0.5, 0.0, 1.0],
        [2.0, 1.0, 0.5, -1.0],
        [-0.5, 1.0, 1.5, 0.0],
    ],
    dtype=torch.float32,
)


def reference_attention_tensor(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
    mask_fill: float = MASK_FILL,
) -> torch.Tensor:
    scores = torch.matmul(q.float(), k.float().transpose(0, 1)) / math.sqrt(float(q.shape[1]))
    if causal:
        row_index = torch.arange(q.shape[0], device=scores.device).unsqueeze(1)
        col_index = torch.arange(k.shape[0], device=scores.device).unsqueeze(0)
        scores = torch.where(col_index <= row_index, scores, torch.full_like(scores, mask_fill))
    return scores


def reference_attention_scores(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool = True,
    mask_fill: float = MASK_FILL,
) -> list[list[float]]:
    scores = reference_attention_tensor(q, k, causal=causal, mask_fill=mask_fill)
    return [[float(value) for value in row] for row in scores.tolist()]
