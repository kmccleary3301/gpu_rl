from __future__ import annotations

import argparse
import heapq
import json


VISIBLE_SMALL = [0.12, 0.91, 0.31, 0.72, 0.44]
VISIBLE_BATCH = [
    [0.5, 0.3, 0.9, 0.1, 0.7],
    [0.05, 0.95, 0.25, 0.85, 0.15],
]
HIDDEN_NEGATIVE = [-1.5, 2.2, 0.0, 2.2, -0.2]
HIDDEN_TIES = [0.5, 0.5, 0.1, 0.5, 0.2]


def _stable_topk_heap(values: list[float], k: int) -> tuple[list[int], list[float]]:
    ranked = heapq.nlargest(k, enumerate(values), key=lambda item: (item[1], -item[0]))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return [index for index, _ in ranked], [float(score) for _, score in ranked]


def _run_benchmark(repeats: int) -> None:
    matrix = [
        [((row * 17 + col * 13) % 997) / 997.0 for col in range(512)]
        for row in range(64)
    ]
    for _ in range(repeats):
        for row in matrix:
            _stable_topk_heap(row, 8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=120)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    payload = {
        "visible_small": dict(zip(["indices", "scores"], _stable_topk_heap(VISIBLE_SMALL, 2), strict=False)),
        "visible_batch": [
            dict(zip(["indices", "scores"], _stable_topk_heap(row, 2), strict=False))
            for row in VISIBLE_BATCH
        ],
        "hidden_negative": dict(zip(["indices", "scores"], _stable_topk_heap(HIDDEN_NEGATIVE, 3), strict=False)),
        "hidden_ties": dict(zip(["indices", "scores"], _stable_topk_heap(HIDDEN_TIES, 3), strict=False)),
    }
    print(json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
