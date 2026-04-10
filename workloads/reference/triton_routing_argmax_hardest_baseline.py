from __future__ import annotations

import argparse
import json
import torch


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


def _stable_argmax(values: list[float]) -> tuple[int, float]:
    best_index = 0
    best_value = values[0]
    for index, value in enumerate(values[1:], start=1):
        if value > best_value:
            best_index = index
            best_value = value
    return best_index, float(best_value)


def _serialize(matrix: torch.Tensor) -> dict[str, object]:
    pairs = [_stable_argmax([float(v) for v in row]) for row in matrix.tolist()]
    return {
        "indices": [index for index, _ in pairs],
        "values": [value for _, value in pairs],
    }


def _benchmark_tensor(rows: int, cols: int) -> torch.Tensor:
    base = torch.arange(rows * cols, dtype=torch.float32)
    return ((base.reshape(rows, cols) * 31) % 601) / 601.0


def _run_benchmark(repeats: int) -> None:
    x = _benchmark_tensor(7168, 224)
    for _ in range(repeats):
        _serialize(x)


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
