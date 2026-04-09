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
HIDDEN_MATRIX = torch.tensor(
    [
        [-1.0, 2.0, 2.0, 1.5, 0.0, -0.5],
        [3.1, 3.0, 2.9, 3.1, 3.05, 3.1],
        [-4.0, -3.0, -2.0, -1.0, -1.0, -1.0],
        [0.25, 0.249, 0.251, 0.251, 0.2, 0.251],
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
    return ((base.reshape(rows, cols) * 29) % 521) / 521.0


def _run_benchmark(repeats: int) -> None:
    x = _benchmark_tensor(6144, 192)
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
                "hidden_routing": _serialize(HIDDEN_MATRIX),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
