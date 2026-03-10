from __future__ import annotations

import json
import math
from pathlib import Path
import statistics

from gpu_cockpit.contracts import BaselineSpec, PerfReport, TaskSpec
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = (len(ordered) - 1) * pct
    low = math.floor(rank)
    high = math.ceil(rank)
    if low == high:
        return ordered[low]
    weight = rank - low
    return ordered[low] * (1.0 - weight) + ordered[high] * weight


def _measure_command(
    executor: CommandExecutor,
    command: list[str],
    warmups: int,
    repeats: int,
    cwd: Path | None = None,
) -> tuple[list[float], list[int]]:
    for _ in range(warmups):
        executor.run(command, cwd=cwd)

    timings: list[float] = []
    exit_codes: list[int] = []
    for _ in range(repeats):
        result = executor.run(command, cwd=cwd)
        timings.append(float(result.duration_ms))
        exit_codes.append(result.exit_code)
    return timings, exit_codes


def resolve_baseline_spec(root: Path, baseline_ref: str) -> BaselineSpec:
    path = Path(baseline_ref)
    if not path.is_absolute():
        path = root / baseline_ref
    path = path.resolve()
    if path.suffix == ".json":
        return BaselineSpec.model_validate_json(path.read_text(encoding="utf-8"))
    if path.suffix == ".py":
        return BaselineSpec(baseline_id=path.stem, command=["python3", str(path)])
    if path.suffix == ".sh":
        return BaselineSpec(baseline_id=path.stem, command=["bash", str(path)])
    return BaselineSpec(baseline_id=path.stem, command=[str(path)])


def run_subprocess_benchmark(
    writer: RunBundleWriter,
    command: list[str],
    warmups: int,
    repeats: int,
    scope: str = "tool.run_benchmark",
    executor: CommandExecutor | None = None,
    baseline_id: str = "self",
    baseline_command: list[str] | None = None,
) -> PerfReport:
    executor = executor or LocalHostToolExecutor()
    started = writer.append_event(
        scope=scope,
        kind="started",
        payload={
            "command": command,
            "warmups": warmups,
            "repeats": repeats,
            "baseline_id": baseline_id,
            "baseline_command": baseline_command,
        },
    )
    timings, exit_codes = _measure_command(executor, command, warmups=warmups, repeats=repeats, cwd=writer.root)

    if any(code != 0 for code in exit_codes):
        writer.append_event(scope=scope, kind="failed", payload={"exit_codes": exit_codes})
        raise RuntimeError(f"Benchmark command failed with exit codes: {exit_codes}")

    baseline_timings: list[float] = []
    baseline_exit_codes: list[int] = []
    if baseline_command is not None:
        baseline_timings, baseline_exit_codes = _measure_command(executor, baseline_command, warmups=warmups, repeats=repeats, cwd=writer.root)
        if any(code != 0 for code in baseline_exit_codes):
            writer.append_event(scope=scope, kind="failed", payload={"baseline_exit_codes": baseline_exit_codes})
            raise RuntimeError(f"Baseline benchmark command failed with exit codes: {baseline_exit_codes}")

    candidate_p50 = _percentile(timings, 0.50)
    candidate_p95 = _percentile(timings, 0.95)
    baseline_p50 = _percentile(baseline_timings, 0.50) if baseline_timings else None
    baseline_p95 = _percentile(baseline_timings, 0.95) if baseline_timings else None
    speedup_vs_baseline = (baseline_p50 / candidate_p50) if baseline_p50 is not None and candidate_p50 > 0.0 else 1.0

    perf = PerfReport(
        baseline_id=baseline_id,
        timer="wall_clock",
        warmups=warmups,
        repeats=repeats,
        steady_state_ms_p50=candidate_p50,
        steady_state_ms_p95=candidate_p95,
        baseline_steady_state_ms_p50=baseline_p50,
        baseline_steady_state_ms_p95=baseline_p95,
        speedup_vs_baseline=speedup_vs_baseline,
        variance_pct=(statistics.pstdev(timings) / statistics.mean(timings) * 100.0) if len(timings) > 1 else 0.0,
        perf_notes=["subprocess benchmark"] + ([f"baseline:{baseline_id}"] if baseline_command is not None else []),
    )
    writer.write_artifact(
        relative_path="perf/raw_timings.json",
        kind="benchmark_timings",
        content=json.dumps(
            {
                "candidate": {"command": command, "timings_ms": timings, "exit_codes": exit_codes},
                "baseline": {
                    "baseline_id": baseline_id,
                    "command": baseline_command,
                    "timings_ms": baseline_timings,
                    "exit_codes": baseline_exit_codes,
                },
            },
            indent=2,
        )
        + "\n",
        mime="application/json",
        semantic_tags=["perf", "benchmark", "raw"],
        producer_event_id=started.event_id,
    )
    writer.write_artifact(
        relative_path="perf/benchmark.json",
        kind="perf_report",
        content=json.dumps(perf.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["perf", "benchmark", "summary"],
        producer_event_id=started.event_id,
    )
    writer.append_event(
        scope=scope,
        kind="completed",
        payload={
            "p50_ms": perf.steady_state_ms_p50,
            "p95_ms": perf.steady_state_ms_p95,
            "baseline_p50_ms": perf.baseline_steady_state_ms_p50,
            "speedup_vs_baseline": perf.speedup_vs_baseline,
        },
    )
    return perf


def run_task_benchmark(
    writer: RunBundleWriter,
    root: Path,
    task: TaskSpec,
    command: list[str],
    scope: str = "tool.run_benchmark",
    executor: CommandExecutor | None = None,
) -> PerfReport:
    baseline_id = "self"
    baseline_command: list[str] | None = None
    if task.baseline_ref:
        baseline = resolve_baseline_spec(root, task.baseline_ref)
        baseline_id = baseline.baseline_id
        baseline_command = baseline.command
    return run_subprocess_benchmark(
        writer=writer,
        command=command,
        warmups=task.perf_protocol.warmups,
        repeats=task.perf_protocol.repeats,
        scope=scope,
        executor=executor,
        baseline_id=baseline_id,
        baseline_command=baseline_command,
    )
