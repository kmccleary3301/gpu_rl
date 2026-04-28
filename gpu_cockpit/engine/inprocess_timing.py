from __future__ import annotations

import statistics
import time
from typing import Callable


def cuda_event_timing_ms(fn: Callable[[], object], *, warmups: int = 3, repeats: int = 20) -> dict[str, object]:
    import torch

    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    for _ in range(max(0, warmups)):
        fn()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(float(start.elapsed_time(end)))
    ordered = sorted(timings)
    return {
        "timer": "cuda_event",
        "warmups": warmups,
        "repeats": repeats,
        "ms_p50": statistics.median(ordered),
        "ms_p95": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))],
        "samples_ms": timings,
    }


def wall_clock_timing_ms(fn: Callable[[], object], *, warmups: int = 3, repeats: int = 20) -> dict[str, object]:
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    for _ in range(max(0, warmups)):
        fn()
    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        timings.append((time.perf_counter() - start) * 1000.0)
    ordered = sorted(timings)
    return {
        "timer": "wall_clock_inprocess",
        "warmups": warmups,
        "repeats": repeats,
        "ms_p50": statistics.median(ordered),
        "ms_p95": ordered[min(len(ordered) - 1, int(round((len(ordered) - 1) * 0.95)))],
        "samples_ms": timings,
    }
