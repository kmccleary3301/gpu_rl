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
TILEAWARE_HIDDEN_MATRIX = torch.tensor(
    [
        [2.5, 2.5, 2.45, 2.4, 2.1, 2.0, 1.9, 1.8, 1.2, 1.1],
        [0.2, 0.25, 0.3, 0.35, 5.0, 5.0, 4.95, 4.9, 4.5, 4.0],
        [-3.2, -3.1, -3.0, -2.9, -2.8, -2.7, -2.6, -2.5, -1.0, -1.0],
        [0.8, 0.81, 0.82, 0.83, 0.83, 0.82, 0.81, 0.8, 0.79, 0.78],
        [6.0, 5.99, 5.98, 5.97, 5.96, 5.95, 6.0, 6.0, 5.94, 5.93],
        [7.1, 7.1, 7.05, 7.0, 6.99, 6.98, 6.97, 6.96, 6.5, 6.4],
    ],
    dtype=torch.float32,
)


def _benchmark_tensor(rows: int, cols: int) -> torch.Tensor:
    base = torch.arange(rows * cols, device="cuda", dtype=torch.float32).reshape(rows, cols)
    tile_ids = (torch.arange(rows, device="cuda", dtype=torch.float32) // 16).unsqueeze(1)
    col_tiles = (torch.arange(cols, device="cuda", dtype=torch.float32) // 8).unsqueeze(0)
    bias = ((tile_ids + col_tiles) % 4 == 0).to(torch.float32) * 0.35
    return ((base * 17) % 997) / 997.0 + bias


def _run_benchmark(repeats: int) -> None:
    x = _benchmark_tensor(8192, 256)
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
                "hidden_routing": _serialize(TILEAWARE_HIDDEN_MATRIX),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
