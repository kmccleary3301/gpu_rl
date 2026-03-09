from __future__ import annotations

import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from gpu_cockpit.contracts import KernelProfileMetric, KernelProfileRecord, ProfileReport, SystemTraceSummary
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _rocprof_binary() -> str | None:
    return shutil.which("rocprof") or shutil.which("rocprofv3")


def _trace_warnings(result_exit_code: int, trace_exists: bool) -> list[str]:
    warnings: list[str] = []
    if not trace_exists:
        warnings.append("rocprof trace file was not created")
    if result_exit_code != 0:
        warnings.append("rocprof trace command returned a non-zero exit code")
    return warnings


def _profile_warnings(result_exit_code: int, records: list[KernelProfileRecord], csv_exists: bool) -> list[str]:
    warnings: list[str] = []
    if not csv_exists:
        warnings.append("rocprof profile csv was not created")
    elif not records:
        warnings.append("rocprof profile csv could not be parsed into normalized kernel records")
    if result_exit_code != 0:
        warnings.append("rocprof profile command returned a non-zero exit code")
    return warnings


def _render_trace_markdown(summary: SystemTraceSummary) -> str:
    warning_lines = [f"- {warning}" for warning in summary.warnings] or ["- none"]
    return "\n".join(
        [
            f"# AMD Trace {summary.backend}",
            "",
            f"- exit_code: `{summary.exit_code}`",
            f"- duration_ms: `{summary.duration_ms}`",
            f"- report_path: `{summary.report_path}`",
            f"- stdout_path: `{summary.stdout_path}`",
            f"- stderr_path: `{summary.stderr_path}`",
            "",
            "## Warnings",
            *warning_lines,
        ]
    )


def _render_profile_markdown(report: ProfileReport) -> str:
    warning_lines = [f"- {warning}" for warning in report.warnings] or ["- none"]
    return "\n".join(
        [
            f"# AMD Profile {report.kernel_name}",
            "",
            f"- profiler: `{report.profiler}`",
            f"- backend: `{report.backend}`",
            f"- classification: `{report.classification}`",
            f"- occupancy: `{report.occupancy}`",
            f"- registers_per_thread: `{report.registers_per_thread}`",
            f"- dram_pct_peak: `{report.dram_throughput_pct_peak}`",
            f"- compute_pct_peak: `{report.compute_throughput_pct_peak}`",
            f"- roofline_position: `{report.roofline_position}`",
            f"- raw_profile_ref: `{report.raw_profile_ref}`",
            "",
            "## Warnings",
            *warning_lines,
        ]
    )


def trace_system_amd(
    writer: RunBundleWriter,
    command: list[str],
    executor: CommandExecutor | None = None,
) -> SystemTraceSummary:
    rocprof = _rocprof_binary()
    if rocprof is None:
        raise RuntimeError("rocprof is not installed.")
    executor = executor or LocalHostToolExecutor()
    started_event = writer.append_event(
        scope="tool.trace_system_amd",
        kind="started",
        payload={"command": command, "tool": Path(rocprof).name},
    )
    started = time.monotonic()
    trace_path = writer.artifacts_dir / "traces" / "system" / "rocprof_trace.json"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    result = executor.run([rocprof, "--hip-trace", "--timestamp", "on", "-o", str(trace_path), *command])
    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_artifact = writer.write_artifact(
        relative_path="traces/system/rocprof_stdout.txt",
        kind="rocprof_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["trace", "amd", "stdout"],
        producer_event_id=started_event.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path="traces/system/rocprof_stderr.txt",
        kind="rocprof_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["trace", "amd", "stderr"],
        producer_event_id=started_event.event_id,
    )
    summary = SystemTraceSummary(
        backend="amd_rocprof_trace",
        command=command,
        trace_enabled=True,
        exit_code=result.exit_code,
        duration_ms=duration_ms,
        report_path=str(trace_path.relative_to(writer.run_dir)) if trace_path.exists() else None,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
        warnings=_trace_warnings(result.exit_code, trace_path.exists()),
    )
    writer.write_artifact(
        relative_path="traces/system/summary.json",
        kind="system_trace_summary",
        content=json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["trace", "amd", "summary"],
        producer_event_id=started_event.event_id,
    )
    writer.write_artifact(
        relative_path="traces/system/summary.md",
        kind="system_trace_summary_markdown",
        content=_render_trace_markdown(summary) + "\n",
        mime="text/markdown",
        semantic_tags=["trace", "amd", "summary", "markdown"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.trace_system_amd",
        kind="completed" if result.exit_code == 0 else "failed",
        payload={"exit_code": result.exit_code, "duration_ms": duration_ms, "report_path": summary.report_path},
    )
    return summary


def _metric_value(raw_value: str) -> float | int | str | None:
    value = raw_value.strip()
    if not value:
        return None
    try:
        if any(ch in value for ch in {".", "e", "E"}):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _classify_kernel(record: dict[str, object]) -> tuple[str, str | None]:
    occupancy = record.get("occupancy")
    dram = record.get("dram_throughput_pct_peak")
    compute = record.get("compute_throughput_pct_peak")
    duration_ms = record.get("duration_ms")
    if isinstance(duration_ms, (int, float)) and duration_ms < 0.05:
        return "launch_overhead", "latency-bound"
    if isinstance(occupancy, (int, float)) and occupancy < 35.0:
        return "occupancy_limited", "below_roofline"
    if isinstance(dram, (int, float)) and dram >= 60.0 and (not isinstance(compute, (int, float)) or dram >= compute):
        return "memory_bound", "memory_roof"
    if isinstance(compute, (int, float)) and compute >= 60.0:
        return "compute_bound", "compute_roof"
    return "mixed", "below_roofline"


def _parse_rocprof_csv(raw_csv: str) -> list[KernelProfileRecord]:
    rows = list(csv.DictReader(raw_csv.splitlines()))
    grouped: dict[str, dict[str, object]] = defaultdict(dict)
    for row in rows:
        kernel_name = (row.get("KernelName") or row.get("Name") or "unknown_kernel").strip()
        bucket = grouped[kernel_name]
        bucket.setdefault("kernel_name", kernel_name)
        bucket.setdefault("raw_metrics", [])
        duration_ns = _metric_value(row.get("DurationNs") or row.get("AverageNs") or "")
        occupancy = _metric_value(row.get("OccupancyPct") or row.get("Occupancy") or "")
        vgpr = _metric_value(row.get("VGPR") or row.get("VGPRs") or "")
        dram_pct = _metric_value(row.get("DRAMPct") or row.get("MemUnitBusyPct") or "")
        compute_pct = _metric_value(row.get("ComputePct") or row.get("VALUUtilizationPct") or "")
        l2_hit = _metric_value(row.get("L2HitPct") or "")
        bucket["raw_metrics"].extend(
            [
                KernelProfileMetric(metric_name="DurationNs", value=duration_ns, unit="ns"),
                KernelProfileMetric(metric_name="OccupancyPct", value=occupancy, unit="%"),
                KernelProfileMetric(metric_name="VGPR", value=vgpr),
                KernelProfileMetric(metric_name="DRAMPct", value=dram_pct, unit="%"),
                KernelProfileMetric(metric_name="ComputePct", value=compute_pct, unit="%"),
                KernelProfileMetric(metric_name="L2HitPct", value=l2_hit, unit="%"),
            ]
        )
        bucket["duration_ms"] = float(duration_ns) / 1_000_000.0 if isinstance(duration_ns, (int, float)) else None
        bucket["occupancy"] = float(occupancy) if isinstance(occupancy, (int, float)) else None
        bucket["registers_per_thread"] = int(vgpr) if isinstance(vgpr, (int, float)) else None
        bucket["dram_throughput_pct_peak"] = float(dram_pct) if isinstance(dram_pct, (int, float)) else None
        bucket["compute_throughput_pct_peak"] = float(compute_pct) if isinstance(compute_pct, (int, float)) else None
        bucket["l2_hit_rate"] = float(l2_hit) if isinstance(l2_hit, (int, float)) else None

    total_duration = sum(
        float(bucket.get("duration_ms", 0.0))
        for bucket in grouped.values()
        if isinstance(bucket.get("duration_ms"), (int, float))
    )
    records: list[KernelProfileRecord] = []
    for bucket in grouped.values():
        classification, roofline_position = _classify_kernel(bucket)
        duration_ms = bucket.get("duration_ms")
        records.append(
            KernelProfileRecord(
                kernel_name=str(bucket["kernel_name"]),
                duration_ms=float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
                time_pct=(float(duration_ms) / total_duration * 100.0)
                if total_duration > 0 and isinstance(duration_ms, (int, float))
                else None,
                classification=classification,
                occupancy=float(bucket["occupancy"]) if isinstance(bucket.get("occupancy"), (int, float)) else None,
                registers_per_thread=int(bucket["registers_per_thread"])
                if isinstance(bucket.get("registers_per_thread"), (int, float))
                else None,
                dram_throughput_pct_peak=float(bucket["dram_throughput_pct_peak"])
                if isinstance(bucket.get("dram_throughput_pct_peak"), (int, float))
                else None,
                compute_throughput_pct_peak=float(bucket["compute_throughput_pct_peak"])
                if isinstance(bucket.get("compute_throughput_pct_peak"), (int, float))
                else None,
                l2_hit_rate=float(bucket["l2_hit_rate"]) if isinstance(bucket.get("l2_hit_rate"), (int, float)) else None,
                roofline_position=roofline_position,
                raw_metrics=list(bucket["raw_metrics"]),
            )
        )
    records.sort(key=lambda record: record.duration_ms or 0.0, reverse=True)
    return records


def profile_kernel_amd(
    writer: RunBundleWriter,
    command: list[str],
    profile_pack: str = "quick",
    executor: CommandExecutor | None = None,
) -> ProfileReport:
    del profile_pack
    rocprof = _rocprof_binary()
    if rocprof is None:
        raise RuntimeError("rocprof is not installed.")
    executor = executor or LocalHostToolExecutor()
    started_event = writer.append_event(
        scope="tool.profile_kernel_amd",
        kind="started",
        payload={"command": command, "tool": Path(rocprof).name},
    )
    started = time.monotonic()
    csv_path = writer.artifacts_dir / "profiles" / "kernel" / "rocprof_profile.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result = executor.run([rocprof, "--stats", "-o", str(csv_path), *command])
    duration_ms = int((time.monotonic() - started) * 1000)
    stdout_artifact = writer.write_artifact(
        relative_path="profiles/kernel/rocprof_stdout.txt",
        kind="rocprof_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["profile", "amd", "stdout"],
        producer_event_id=started_event.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path="profiles/kernel/rocprof_stderr.txt",
        kind="rocprof_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["profile", "amd", "stderr"],
        producer_event_id=started_event.event_id,
    )
    raw_csv = csv_path.read_text(encoding="utf-8") if csv_path.exists() else ""
    records = _parse_rocprof_csv(raw_csv)
    primary = records[0] if records else KernelProfileRecord(kernel_name="unknown_kernel", classification="mixed")
    report = ProfileReport(
        profiler="rocprof",
        backend="amd_rocprof",
        command=command,
        profile_pack="quick",
        exit_code=result.exit_code,
        duration_ms=duration_ms,
        kernel_name=primary.kernel_name,
        classification=primary.classification,
        occupancy=primary.occupancy,
        registers_per_thread=primary.registers_per_thread,
        dram_throughput_pct_peak=primary.dram_throughput_pct_peak,
        compute_throughput_pct_peak=primary.compute_throughput_pct_peak,
        l2_hit_rate=primary.l2_hit_rate,
        roofline_position=primary.roofline_position,
        top_kernels=records,
        primary_kernel_duration_ms=primary.duration_ms,
        profiled_kernel_count=len(records),
        raw_profile_ref="profiles/kernel/rocprof_profile.csv" if csv_path.exists() else None,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
        warnings=_profile_warnings(result.exit_code, records, csv_path.exists()),
    )
    writer.write_artifact(
        relative_path="profiles/kernel/summary.json",
        kind="profile_summary",
        content=json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["profile", "amd", "summary"],
        producer_event_id=started_event.event_id,
    )
    writer.write_artifact(
        relative_path="profiles/kernel/summary.md",
        kind="profile_summary_markdown",
        content=_render_profile_markdown(report) + "\n",
        mime="text/markdown",
        semantic_tags=["profile", "amd", "summary", "markdown"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.profile_kernel_amd",
        kind="completed" if result.exit_code == 0 else "failed",
        payload={"exit_code": result.exit_code, "duration_ms": duration_ms, "kernel_name": report.kernel_name},
    )
    return report
