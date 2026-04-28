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


def _benchmark_repeats_index(command: list[str]) -> int | None:
    try:
        idx = command.index("--benchmark-repeats")
    except ValueError:
        return None
    if idx + 1 >= len(command):
        return None
    return idx


def _replace_benchmark_repeats(command: list[str], repeats: int) -> list[str]:
    idx = _benchmark_repeats_index(command)
    if idx is None:
        return list(command)
    updated = list(command)
    updated[idx + 1] = str(repeats)
    return updated


def _command_benchmark_repeats(command: list[str]) -> int | None:
    idx = _benchmark_repeats_index(command)
    if idx is None:
        return None
    try:
        return int(command[idx + 1])
    except ValueError:
        return None


def _ensure_candidate_benchmark_mode(candidate_command: list[str], baseline_command: list[str] | None) -> list[str]:
    updated = list(candidate_command)
    if baseline_command is None:
        return updated
    if "--benchmark-only" in updated:
        pass
    elif "--benchmark-only" in baseline_command:
        updated = [*updated, "--benchmark-only"]
    for option in ("--benchmark-repeats", "--benchmark-profile"):
        if option in updated:
            continue
        try:
            idx = baseline_command.index(option)
        except ValueError:
            continue
        if idx + 1 < len(baseline_command):
            updated = [*updated, option, baseline_command[idx + 1]]
    return updated


def _extract_inprocess_timing(stdout: str) -> dict[str, object] | None:
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        timing = payload.get("inprocess_kernel_timing") or payload.get("inprocess_timing")
        if isinstance(timing, dict):
            return timing
        if "inprocess_kernel_ms_p50" in payload:
            return {
                "ms_p50": payload.get("inprocess_kernel_ms_p50"),
                "ms_p95": payload.get("inprocess_kernel_ms_p95"),
                "timer": payload.get("inprocess_timer", "cuda_event"),
            }
    return None


def _timing_ms_p50(timing: dict[str, object] | None) -> float | None:
    if not timing:
        return None
    value = timing.get("ms_p50")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _measure_command(
    executor: CommandExecutor,
    command: list[str],
    warmups: int,
    repeats: int,
    split_compile_from_run: bool,
    cwd: Path | None = None,
) -> tuple[int | None, list[float], list[int], dict[str, object] | None]:
    normalized_command = normalize_python_command(command)
    run_env = os.environ.copy()
    run_env.update(local_python_build_env(cwd))
    cold_compile_ms: int | None = None
    internal_repeats = _command_benchmark_repeats(normalized_command)
    zero_repeat_command = _replace_benchmark_repeats(normalized_command, 0) if internal_repeats is not None else normalized_command
    if split_compile_from_run:
        cold_result = executor.run(zero_repeat_command, cwd=cwd, env=run_env)
        cold_compile_ms = int(cold_result.duration_ms)
        if cold_result.exit_code != 0:
            return cold_compile_ms, [], [cold_result.exit_code], None
    for _ in range(warmups):
        executor.run(normalized_command, cwd=cwd, env=run_env)

    timings: list[float] = []
    exit_codes: list[int] = []
    inprocess_timing: dict[str, object] | None = None
    for _ in range(repeats):
        result = executor.run(normalized_command, cwd=cwd, env=run_env)
        duration_ms = float(result.duration_ms)
        if split_compile_from_run and internal_repeats is not None and internal_repeats > 0 and cold_compile_ms is not None:
            duration_ms = max(0.0, duration_ms - float(cold_compile_ms)) / float(internal_repeats)
        timings.append(duration_ms)
        exit_codes.append(result.exit_code)
        if inprocess_timing is None and result.exit_code == 0:
            inprocess_timing = _extract_inprocess_timing(result.stdout)
    return cold_compile_ms, timings, exit_codes, inprocess_timing


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
    baseline_kind: str | None = None,
    timing_method: str = "wall_clock",
    split_compile_from_run: bool = True,
) -> PerfReport:
    executor = executor or LocalHostToolExecutor()
    normalized_command = normalize_python_command(command)
    normalized_baseline_command = normalize_python_command(baseline_command) if baseline_command is not None else None
    candidate_internal_repeats = _command_benchmark_repeats(normalized_command)
    baseline_internal_repeats = _command_benchmark_repeats(normalized_baseline_command) if normalized_baseline_command is not None else None
    hardware_fingerprint = _load_hardware_fingerprint(writer)
    benchmark_provenance = {
        "scope": scope,
        "baseline_id": baseline_id,
        "baseline_kind": baseline_kind,
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
    candidate_cold_compile_ms, timings, exit_codes, candidate_inprocess_timing = _measure_command(
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
        baseline_cold_compile_ms, baseline_timings, baseline_exit_codes, baseline_inprocess_timing = _measure_command(
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
    else:
        baseline_inprocess_timing = None

    candidate_p50 = _percentile(timings, 0.50)
    candidate_p95 = _percentile(timings, 0.95)
    baseline_p50 = _percentile(baseline_timings, 0.50) if baseline_timings else None
    baseline_p95 = _percentile(baseline_timings, 0.95) if baseline_timings else None
    candidate_inprocess_p50 = _timing_ms_p50(candidate_inprocess_timing)
    baseline_inprocess_p50 = _timing_ms_p50(baseline_inprocess_timing)
    inprocess_speedup = (
        baseline_inprocess_p50 / candidate_inprocess_p50
        if baseline_inprocess_p50 is not None and candidate_inprocess_p50 is not None and candidate_inprocess_p50 > 0.0
        else None
    )
    process_delta_speedup = (baseline_p50 / candidate_p50) if baseline_p50 is not None and candidate_p50 > 0.0 else 1.0
    speedup_vs_baseline = inprocess_speedup if inprocess_speedup is not None else process_delta_speedup

    perf_notes = ["subprocess benchmark"] + ([f"baseline:{baseline_id}"] if baseline_command is not None else [])
    if baseline_kind:
        perf_notes.append(f"baseline_kind:{baseline_kind}")
    if split_compile_from_run and candidate_internal_repeats is not None:
        perf_notes.append("process_delta_per_internal_repeat")
    if (
        candidate_cold_compile_ms is not None
        and candidate_p50 > 0.0
        and candidate_cold_compile_ms > candidate_p50 * max(float(candidate_internal_repeats or 1), 1.0)
    ):
        perf_notes.append("candidate_startup_dominated")
    if (
        baseline_cold_compile_ms is not None
        and baseline_p50 is not None
        and baseline_p50 > 0.0
        and baseline_cold_compile_ms > baseline_p50 * max(float(baseline_internal_repeats or 1), 1.0)
    ):
        perf_notes.append("baseline_startup_dominated")
    steady_state_semantics = (
        "process_delta_per_internal_repeat"
        if split_compile_from_run and candidate_internal_repeats is not None
        else "subprocess_wall_clock"
    )
    score_surfaces = {
        "subprocess_smoke": {
            "candidate_exit_codes_ok": all(code == 0 for code in exit_codes),
            "baseline_exit_codes_ok": all(code == 0 for code in baseline_exit_codes) if baseline_command is not None else None,
        },
        "process_delta_perf": {
            "available": steady_state_semantics == "process_delta_per_internal_repeat",
            "speedup_vs_baseline": process_delta_speedup,
            "candidate_ms_p50": candidate_p50,
            "baseline_ms_p50": baseline_p50,
        },
        "inprocess_kernel_perf": {
            "available": candidate_inprocess_timing is not None and baseline_inprocess_timing is not None,
            "speedup_vs_baseline": inprocess_speedup,
            "candidate": candidate_inprocess_timing,
            "baseline": baseline_inprocess_timing,
            "reason": None
            if candidate_inprocess_timing is not None and baseline_inprocess_timing is not None
            else "not_reported_by_command",
        },
        "startup_diagnostics": {
            "candidate_cold_compile_ms": candidate_cold_compile_ms,
            "baseline_cold_compile_ms": baseline_cold_compile_ms,
            "candidate_startup_dominated": "candidate_startup_dominated" in perf_notes,
            "baseline_startup_dominated": "baseline_startup_dominated" in perf_notes,
            "steady_state_semantics": steady_state_semantics,
        },
    }

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
        perf_notes=perf_notes,
        score_surfaces=score_surfaces,
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
        "candidate_internal_repeats": candidate_internal_repeats,
        "baseline_command": normalized_baseline_command,
        "baseline_command_sha256": perf.baseline_command_sha256,
        "baseline_internal_repeats": baseline_internal_repeats,
        "steady_state_semantics": steady_state_semantics,
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
                    "inprocess_timing": candidate_inprocess_timing,
                },
                "baseline": {
                    "baseline_id": baseline_id,
                    "command": normalized_baseline_command,
                    "command_sha256": perf.baseline_command_sha256,
                    "cold_compile_ms": baseline_cold_compile_ms,
                    "timings_ms": baseline_timings,
                    "exit_codes": baseline_exit_codes,
                    "inprocess_timing": baseline_inprocess_timing,
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
        relative_path="perf/score_surfaces.json",
        kind="benchmark_score_surfaces",
        content=json.dumps(score_surfaces, indent=2) + "\n",
        mime="application/json",
        semantic_tags=["perf", "benchmark", "score_surfaces"],
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
    benchmark_command = list(command) + list(task.perf_protocol.benchmark_args)
    if task.baseline_ref:
        baseline = resolve_baseline_spec(root, task.baseline_ref)
        baseline_id = baseline.baseline_id
        baseline_command = list(baseline.command) + list(baseline.benchmark_args)
        baseline_kind = baseline.baseline_kind
    else:
        baseline_kind = None
    benchmark_command = _ensure_candidate_benchmark_mode(benchmark_command, baseline_command)
    return run_subprocess_benchmark(
        writer=writer,
        command=benchmark_command,
        warmups=task.perf_protocol.warmups,
        repeats=task.perf_protocol.repeats,
        scope=scope,
        executor=executor,
        baseline_id=baseline_id,
        baseline_command=baseline_command,
        baseline_kind=baseline_kind,
        timing_method=task.perf_protocol.timer,
        split_compile_from_run=task.perf_protocol.split_compile_from_run,
    )
