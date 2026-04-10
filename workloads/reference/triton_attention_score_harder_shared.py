from __future__ import annotations

try:
    from triton_attention_score_hard_shared import (
        MASK_FILL,
        VISIBLE_K,
        VISIBLE_Q,
        reference_attention_scores,
        reference_attention_tensor,
    )
except ModuleNotFoundError:
    from workloads.reference.triton_attention_score_hard_shared import (
        MASK_FILL,
        VISIBLE_K,
        VISIBLE_Q,
        reference_attention_scores,
        reference_attention_tensor,
    )

import torch


HARDER_Q = torch.tensor(
    [
        [1.0, -1.0, 0.5, 2.0, -0.5, 1.5, 0.25, -0.75, 1.25, 0.5],
        [0.0, 1.5, -0.5, 1.0, 2.0, -1.5, 0.75, 0.25, -0.25, 1.75],
        [2.0, 0.5, 1.0, -1.0, -0.25, 1.25, 0.5, 1.5, 0.75, -1.25],
        [-1.0, 2.0, 0.0, 0.5, 1.0, -0.5, 1.75, -1.25, 0.5, 1.25],
        [0.75, -0.25, 1.5, 0.0, -1.0, 2.0, 0.5, 0.25, 1.0, -0.75],
        [1.5, 0.25, -1.0, 1.75, 0.5, -0.5, 1.0, -1.5, 2.0, 0.5],
        [0.25, 1.0, 2.0, -0.5, 1.25, 0.75, -1.25, 1.5, 0.0, 1.0],
    ],
    dtype=torch.float32,
)

HARDER_K = torch.tensor(
    [
        [0.5, 1.0, -1.0, 2.0, 0.5, -0.5, 1.25, 0.0, -0.25, 1.0],
        [1.5, -0.5, 0.0, 1.0, -1.0, 1.5, 0.25, 0.75, 2.0, -0.25],
        [2.0, 1.0, 0.5, -1.0, 0.75, 0.25, -0.5, 1.5, 0.5, 1.25],
        [-0.5, 1.0, 1.5, 0.0, 1.25, -1.25, 0.5, 0.25, -0.75, 1.75],
        [1.0, -1.5, 0.25, 0.75, 2.0, 0.5, -0.75, 1.0, 1.25, 0.5],
        [0.25, 1.75, -0.5, 1.25, -0.25, 2.25, 0.0, -1.0, 0.75, 1.5],
        [1.25, 0.5, 1.0, -0.25, 1.5, 0.75, -1.5, 0.5, 2.0, -0.5],
    ],
    dtype=torch.float32,
)
