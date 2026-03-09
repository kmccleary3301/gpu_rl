from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from triton_row_sum_kernel import triton_row_sum


VISIBLE_MATRIX = torch.tensor(
    [
        [1.0, 2.0, 3.0, 4.0],
        [0.5, -0.5, 1.5, 2.5],
    ],
    dtype=torch.float32,
)
HIDDEN_MATRIX = torch.tensor(
    [
        [2.0, 2.0, 2.0, 2.0],
        [-1.0, 3.0, 0.5, 0.5],
        [10.0, -4.0, 1.0, -1.0],
    ],
    dtype=torch.float32,
)


def _benchmark_tensor(rows: int, cols: int) -> torch.Tensor:
    base = torch.arange(rows * cols, device="cuda", dtype=torch.float32)
    return (base.reshape(rows, cols) % 97) / 97.0


def _run_benchmark(repeats: int) -> None:
    x = _benchmark_tensor(4096, 128)
    for _ in range(repeats):
        out = triton_row_sum(x)
        torch.cuda.synchronize()
    _ = out


def _payload() -> dict[str, object]:
    visible = triton_row_sum(VISIBLE_MATRIX.cuda()).cpu().tolist()
    hidden = triton_row_sum(HIDDEN_MATRIX.cuda()).cpu().tolist()
    return {
        "visible_row_sum": [float(value) for value in visible],
        "hidden_row_sum": [float(value) for value in hidden],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(json.dumps(_payload(), sort_keys=True))


if __name__ == "__main__":
    main()
