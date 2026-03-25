from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import os

from gpu_cockpit.contracts import BaselineSpec, PerfReport, TaskSpec
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.command_utils import local_python_build_env, normalize_python_command
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
    split_compile_from_run: bool,
    cwd: Path | None = None,
) -> tuple[int | None, list[float], list[int]]:
    normalized_command = normalize_python_command(command)
    run_env = os.environ.copy()
    run_env.update(local_python_build_env(cwd))
    cold_compile_ms: int | None = None
    if split_compile_from_run:
        cold_result = executor.run(normalized_command, cwd=cwd, env=run_env)
        cold_compile_ms = int(cold_result.duration_ms)
        if cold_result.exit_code != 0:
            return cold_compile_ms, [], [cold_result.exit_code]
    for _ in range(warmups):
        executor.run(normalized_command, cwd=cwd, env=run_env)

    timings: list[float] = []
    exit_codes: list[int] = []
    for _ in range(repeats):
        result = executor.run(normalized_command, cwd=cwd, env=run_env)
        timings.append(float(result.duration_ms))
        exit_codes.append(result.exit_code)
    return cold_compile_ms, timings, exit_codes


def _command_sha256(command: list[str]) -> str:
    return hashlib.sha256(json.dumps(command, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()


def _load_hardware_fingerprint(writer: RunBundleWriter) -> dict[str, object] | None:
    fingerprint_path = writer.run_dir / "meta" / "hardware_fingerprint.json"
    if not fingerprint_path.exists():
        return None
    try:
        payload = json.loads(fingerprint_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, dict):
        return None
    return {
        "vendor": payload.get("vendor"),
        "gpu_name": payload.get("gpu_name"),
        "arch": payload.get("arch"),
        "driver_version": payload.get("driver_version"),
        "runtime_version": payload.get("runtime_version"),
        "memory_gb": payload.get("memory_gb"),
    }


def resolve_baseline_spec(root: Path, baseline_ref: str) -> BaselineSpec:
    path = Path(baseline_ref)
    if not path.is_absolute():
        path = root / baseline_ref
    path = path.resolve()
    if path.suffix == ".json":
        return BaselineSpec.model_validate_json(path.read_text(encoding="utf-8"))
    if path.suffix == ".py":
        return BaselineSpec(baseline_id=path.stem, command=[sys.executable, str(path)])
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
    timing_method: str = "wall_clock",
    split_compile_from_run: bool = True,
) -> PerfReport:
    executor = executor or LocalHostToolExecutor()
    normalized_command = normalize_python_command(command)
    normalized_baseline_command = normalize_python_command(baseline_command) if baseline_command is not None else None
    hardware_fingerprint = _load_hardware_fingerprint(writer)
    benchmark_provenance = {
        "scope": scope,
        "baseline_id": baseline_id,
        "run_id": writer.run_spec.run_id if writer.run_spec is not None else None,
        "task_ref": writer.run_spec.task_ref if writer.run_spec is not None else None,
        "target_backend": writer.run_spec.target_backend if writer.run_spec is not None else None,
        "target_vendor": writer.run_spec.target_vendor if writer.run_spec is not None else None,
    }
    started = writer.append_event(
        scope=scope,
        kind="started",
        payload={
            "command": normalized_command,
            "warmups": warmups,
            "repeats": repeats,
            "timing_method": timing_method,
            "split_compile_from_run": split_compile_from_run,
            "baseline_id": baseline_id,
            "baseline_command": normalized_baseline_command,
        },
    )
    candidate_cold_compile_ms, timings, exit_codes = _measure_command(
        executor,
        normalized_command,
        warmups=warmups,
        repeats=repeats,
        split_compile_from_run=split_compile_from_run,
        cwd=writer.root,
    )

    if any(code != 0 for code in exit_codes):
        writer.append_event(scope=scope, kind="failed", payload={"exit_codes": exit_codes})
        raise RuntimeError(f"Benchmark command failed with exit codes: {exit_codes}")

    baseline_timings: list[float] = []
    baseline_exit_codes: list[int] = []
    baseline_cold_compile_ms: int | None = None
    if normalized_baseline_command is not None:
        baseline_cold_compile_ms, baseline_timings, baseline_exit_codes = _measure_command(
            executor,
            normalized_baseline_command,
            warmups=warmups,
            repeats=repeats,
            split_compile_from_run=split_compile_from_run,
            cwd=writer.root,
        )
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
        timing_method=timing_method,
        warmups=warmups,
        repeats=repeats,
        split_compile_from_run=split_compile_from_run,
        cold_compile_ms=candidate_cold_compile_ms,
        baseline_cold_compile_ms=baseline_cold_compile_ms,
        steady_state_ms_p50=candidate_p50,
        steady_state_ms_p95=candidate_p95,
        baseline_steady_state_ms_p50=baseline_p50,
        baseline_steady_state_ms_p95=baseline_p95,
        speedup_vs_baseline=speedup_vs_baseline,
        variance_pct=(statistics.pstdev(timings) / statistics.mean(timings) * 100.0) if len(timings) > 1 else 0.0,
        benchmark_scope=scope,
        candidate_command_sha256=_command_sha256(normalized_command),
        baseline_command_sha256=_command_sha256(normalized_baseline_command) if normalized_baseline_command is not None else None,
        hardware_fingerprint=hardware_fingerprint,
        benchmark_provenance=benchmark_provenance,
        perf_notes=["subprocess benchmark"] + ([f"baseline:{baseline_id}"] if baseline_command is not None else []),
    )
    protocol_payload = {
        "benchmark_protocol_version": perf.benchmark_protocol_version,
        "timing_method": timing_method,
        "timer": perf.timer,
        "warmups": warmups,
        "repeats": repeats,
        "split_compile_from_run": split_compile_from_run,
        "benchmark_scope": scope,
        "baseline_id": baseline_id,
        "candidate_command": normalized_command,
        "candidate_command_sha256": perf.candidate_command_sha256,
        "baseline_command": normalized_baseline_command,
        "baseline_command_sha256": perf.baseline_command_sha256,
        "hardware_fingerprint": hardware_fingerprint,
        "benchmark_provenance": benchmark_provenance,
    }
    writer.write_artifact(
        relative_path="perf/raw_timings.json",
        kind="benchmark_timings",
        content=json.dumps(
            {
                "candidate": {
                    "command": normalized_command,
                    "command_sha256": perf.candidate_command_sha256,
                    "cold_compile_ms": candidate_cold_compile_ms,
                    "timings_ms": timings,
                    "exit_codes": exit_codes,
                },
                "baseline": {
                    "baseline_id": baseline_id,
                    "command": normalized_baseline_command,
                    "command_sha256": perf.baseline_command_sha256,
                    "cold_compile_ms": baseline_cold_compile_ms,
                    "timings_ms": baseline_timings,
                    "exit_codes": baseline_exit_codes,
                },
                "protocol": protocol_payload,
            },
            indent=2,
        )
        + "\n",
        mime="application/json",
        semantic_tags=["perf", "benchmark", "raw"],
        producer_event_id=started.event_id,
    )
    writer.write_artifact(
        relative_path="perf/benchmark_protocol.json",
        kind="benchmark_protocol",
        content=json.dumps(protocol_payload, indent=2) + "\n",
        mime="application/json",
        semantic_tags=["perf", "benchmark", "protocol"],
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
            "cold_compile_ms": perf.cold_compile_ms,
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
        timing_method=task.perf_protocol.timer,
        split_compile_from_run=task.perf_protocol.split_compile_from_run,
    )
