from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_routing_argmax_kernel import triton_routing_argmax


VISIBLE_MATRIX = torch.tensor(
    [
        [0.1, 0.7, 0.4, 0.2],
        [1.0, 1.0, 0.5, -1.0],
    ],
    dtype=torch.float32,
)
HIGHEST_HIDDEN_MATRIX = torch.tensor(
    [
        [-2.0, 3.0, 3.0, 2.75, 2.75, 1.0, -0.5, 3.0],
        [4.2, 4.2, 4.1, 4.2, 4.15, 4.0, 4.2, 3.9],
        [-5.0, -4.5, -4.0, -3.5, -3.5, -3.0, -2.5, -2.5],
        [0.501, 0.5, 0.501, 0.499, 0.501, 0.45, 0.501, 0.3],
        [7.0, 6.99, 6.98, 7.0, 6.97, 6.96, 7.0, 6.5],
    ],
    dtype=torch.float32,
)


def _benchmark_tensor(rows: int, cols: int) -> torch.Tensor:
    base = torch.arange(rows * cols, device="cuda", dtype=torch.float32)
    return ((base.reshape(rows, cols) * 31) % 601) / 601.0


def _run_benchmark(repeats: int) -> None:
    x = _benchmark_tensor(7168, 224)
    for _ in range(repeats):
        out_idx, out_val = triton_routing_argmax(x)
        torch.cuda.synchronize()
    _ = (out_idx, out_val)


def _serialize(matrix: torch.Tensor) -> dict[str, object]:
    idx, val = triton_routing_argmax(matrix.cuda())
    return {
        "indices": [int(v) for v in idx.cpu().tolist()],
        "values": [float(v) for v in val.cpu().tolist()],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_routing": _serialize(VISIBLE_MATRIX),
                "hidden_routing": _serialize(HIGHEST_HIDDEN_MATRIX),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
