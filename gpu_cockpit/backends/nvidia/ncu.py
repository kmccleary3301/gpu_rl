from __future__ import annotations

import csv
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

from gpu_cockpit.contracts import KernelProfileMetric, KernelProfileRecord, ProfileReport
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


PROFILE_PACK_SECTIONS: dict[str, tuple[str, ...]] = {
    "quick": ("LaunchStats", "Occupancy", "SpeedOfLight"),
    "memory": ("LaunchStats", "Occupancy", "SpeedOfLight", "MemoryWorkloadAnalysis"),
    "compute": ("LaunchStats", "Occupancy", "SpeedOfLight", "SchedulerStats"),
    "deep": ("LaunchStats", "Occupancy", "SpeedOfLight", "MemoryWorkloadAnalysis", "SchedulerStats"),
}

_FLOAT_METRICS = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "occupancy",
    "launch__occupancy_limit_active_warps_pct": "occupancy",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct_peak",
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed": "dram_throughput_pct_peak",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed": "sm_throughput_pct_peak",
    "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_active": "compute_throughput_pct_peak",
    "smsp__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active": "compute_throughput_pct_peak",
    "lts__t_sector_hit_rate.pct": "l2_hit_rate",
    "lts__request_hit_rate.pct": "l2_hit_rate",
}

_INT_METRICS = {
    "launch__registers_per_thread": "registers_per_thread",
    "launch__local_load_bytes.sum": "spill_load_bytes",
    "launch__local_store_bytes.sum": "spill_store_bytes",
}

_DURATION_METRICS = {
    "gpu__time_duration.sum",
    "gpu__time_duration.avg",
    "gpu__time_duration.max",
}


def _metric_value(raw_value: str) -> float | int | str | None:
    value = raw_value.strip()
    if not value:
        return None
    lowered = value.lower()
    if lowered in {"nan", "n/a", "none"}:
        return None
    normalized = value.replace(",", "")
    try:
        if any(ch in normalized for ch in {".", "e", "E"}):
            return float(normalized)
        return int(normalized)
    except ValueError:
        return value


def _duration_ms_from_metric(metric_name: str, metric_unit: str | None, metric_value: float | int | str | None) -> float | None:
    if metric_name not in _DURATION_METRICS:
        return None
    if not isinstance(metric_value, (int, float)):
        return None
    unit = (metric_unit or "").strip().lower()
    if unit == "ns":
        return float(metric_value) / 1_000_000.0
    if unit == "us":
        return float(metric_value) / 1_000.0
    if unit == "ms":
        return float(metric_value)
    if unit == "s":
        return float(metric_value) * 1_000.0
    return None


def _normalize_kernel_name(row: dict[str, str]) -> str:
    for key in (
        "Kernel Name",
        "Kernel Name Base",
        "Kernel Name Demangled",
        "Name",
        "Kernel",
    ):
        value = row.get(key)
        if value:
            return value.strip()
    return "unknown_kernel"


def _classify_kernel(record: dict[str, object]) -> tuple[str, str | None]:
    occupancy = record.get("occupancy")
    dram = record.get("dram_throughput_pct_peak")
    sm = record.get("sm_throughput_pct_peak")
    compute = record.get("compute_throughput_pct_peak")
    duration_ms = record.get("duration_ms")
    spill_bytes = (record.get("spill_load_bytes") or 0) + (record.get("spill_store_bytes") or 0)

    if isinstance(duration_ms, (int, float)) and duration_ms < 0.05:
        return "launch_overhead", "latency-bound"
    if isinstance(occupancy, (int, float)) and occupancy < 35.0:
        return "occupancy_limited", "below_roofline"
    if isinstance(spill_bytes, (int, float)) and spill_bytes > 0:
        return "register_pressure", "below_roofline"
    if isinstance(dram, (int, float)) and dram >= 60.0 and (not isinstance(compute, (int, float)) or dram >= compute):
        return "memory_bound", "memory_roof"
    if isinstance(sm, (int, float)) and sm >= 60.0:
        return "compute_bound", "compute_roof"
    if isinstance(compute, (int, float)) and compute >= 60.0:
        return "compute_bound", "compute_roof"
    return "mixed", "below_roofline"


def _parse_raw_csv(raw_csv: str) -> list[KernelProfileRecord]:
    lines = raw_csv.splitlines()
    header_index = next(
        (index for index, line in enumerate(lines) if "Metric Name" in line and ("Metric Value" in line or ",Value" in line)),
        None,
    )
    if header_index is None:
        return []
    rows = list(csv.DictReader(lines[header_index:]))
    grouped: dict[str, dict[str, object]] = defaultdict(dict)

    for row in rows:
        metric_name = (row.get("Metric Name") or row.get("Name") or "").strip()
        if not metric_name:
            continue
        metric_unit = (row.get("Metric Unit") or row.get("Unit") or "").strip() or None
        metric_value = _metric_value(row.get("Metric Value") or row.get("Value") or "")
        kernel_name = _normalize_kernel_name(row)
        bucket = grouped[kernel_name]
        bucket.setdefault("kernel_name", kernel_name)
        bucket.setdefault("raw_metrics", [])
        bucket.setdefault("invocation_count", 1)
        bucket["raw_metrics"].append(KernelProfileMetric(metric_name=metric_name, unit=metric_unit, value=metric_value))

        if metric_name in _FLOAT_METRICS and isinstance(metric_value, (int, float)):
            bucket[_FLOAT_METRICS[metric_name]] = float(metric_value)
        elif metric_name in _INT_METRICS and isinstance(metric_value, (int, float)):
            bucket[_INT_METRICS[metric_name]] = int(metric_value)

        duration_ms = _duration_ms_from_metric(metric_name, metric_unit, metric_value)
        if duration_ms is not None:
            bucket["duration_ms"] = max(float(bucket.get("duration_ms", 0.0)), duration_ms)

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
                invocation_count=max(1, int(bucket.get("invocation_count", 1))),
                duration_ms=float(duration_ms) if isinstance(duration_ms, (int, float)) else None,
                time_pct=(float(duration_ms) / total_duration * 100.0)
                if total_duration > 0 and isinstance(duration_ms, (int, float))
                else None,
                classification=classification,
                occupancy=float(bucket["occupancy"]) if isinstance(bucket.get("occupancy"), (int, float)) else None,
                registers_per_thread=int(bucket["registers_per_thread"])
                if isinstance(bucket.get("registers_per_thread"), (int, float))
                else None,
                spill_load_bytes=int(bucket["spill_load_bytes"])
                if isinstance(bucket.get("spill_load_bytes"), (int, float))
                else None,
                spill_store_bytes=int(bucket["spill_store_bytes"])
                if isinstance(bucket.get("spill_store_bytes"), (int, float))
                else None,
                dram_throughput_pct_peak=float(bucket["dram_throughput_pct_peak"])
                if isinstance(bucket.get("dram_throughput_pct_peak"), (int, float))
                else None,
                sm_throughput_pct_peak=float(bucket["sm_throughput_pct_peak"])
                if isinstance(bucket.get("sm_throughput_pct_peak"), (int, float))
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


def profile_kernel_nvidia(
    writer: RunBundleWriter,
    command: list[str],
    profile_pack: str = "quick",
    executor: CommandExecutor | None = None,
) -> ProfileReport:
    if shutil.which("ncu") is None:
        raise RuntimeError("ncu is not installed.")
    if profile_pack not in PROFILE_PACK_SECTIONS:
        raise ValueError(f"Unknown profile pack: {profile_pack}")

    executor = executor or LocalHostToolExecutor()
    started_event = writer.append_event(
        scope="tool.profile_kernel_nvidia",
        kind="started",
        payload={"command": command, "profile_pack": profile_pack},
    )
    started = time.monotonic()
    trace_prefix = writer.artifacts_dir / "profiles" / "kernel" / "profile"
    trace_prefix.parent.mkdir(parents=True, exist_ok=True)
    sections = PROFILE_PACK_SECTIONS[profile_pack]
    profile_cmd = [
        "ncu",
        "--target-processes",
        "all",
        "--force-overwrite",
        "--export",
        str(trace_prefix),
    ]
    for section in sections:
        profile_cmd.extend(["--section", section])
    profile_cmd.extend(command)
    result = executor.run(profile_cmd)
    duration_ms = int((time.monotonic() - started) * 1000)

    stdout_artifact = writer.write_artifact(
        relative_path="profiles/kernel/ncu_stdout.txt",
        kind="ncu_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["profile", "nvidia", "stdout"],
        producer_event_id=started_event.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path="profiles/kernel/ncu_stderr.txt",
        kind="ncu_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["profile", "nvidia", "stderr"],
        producer_event_id=started_event.event_id,
    )

    report_path = trace_prefix.with_suffix(".ncu-rep")
    warnings: list[str] = []
    if not report_path.exists():
        warnings.append("ncu report file was not created")

    raw_csv = ""
    if report_path.exists():
        import_cmd = [
            "ncu",
            "--import",
            str(report_path),
            "--page",
            "raw",
            "--csv",
            "--units",
            "base",
        ]
        import_result = executor.run(import_cmd)
        writer.write_artifact(
            relative_path="profiles/kernel/ncu_import_stdout.txt",
            kind="ncu_import_stdout",
            content=import_result.stdout,
            mime="text/plain",
            semantic_tags=["profile", "nvidia", "import", "stdout"],
            producer_event_id=started_event.event_id,
        )
        writer.write_artifact(
            relative_path="profiles/kernel/ncu_import_stderr.txt",
            kind="ncu_import_stderr",
            content=import_result.stderr,
            mime="text/plain",
            semantic_tags=["profile", "nvidia", "import", "stderr"],
            producer_event_id=started_event.event_id,
        )
        if import_result.exit_code == 0:
            raw_csv = import_result.stdout
        else:
            warnings.append("ncu import raw csv failed")

    csv_profile_ref: str | None = None
    top_kernels: list[KernelProfileRecord] = []
    if raw_csv:
        csv_profile_ref = writer.write_artifact(
            relative_path="profiles/kernel/raw_metrics.csv",
            kind="ncu_raw_metrics_csv",
            content=raw_csv,
            mime="text/csv",
            semantic_tags=["profile", "nvidia", "csv"],
            producer_event_id=started_event.event_id,
        ).path
        top_kernels = _parse_raw_csv(raw_csv)
    else:
        warnings.append("no raw kernel metrics were collected")

    primary = top_kernels[0] if top_kernels else None
    report = ProfileReport(
        profiler="ncu",
        backend="nvidia_ncu",
        command=command,
        profile_pack=profile_pack,
        exit_code=result.exit_code,
        duration_ms=duration_ms,
        kernel_name=primary.kernel_name if primary else "unknown_kernel",
        classification=primary.classification if primary else "unknown",
        occupancy=primary.occupancy if primary else None,
        registers_per_thread=primary.registers_per_thread if primary else None,
        spills_bytes=((primary.spill_load_bytes or 0) + (primary.spill_store_bytes or 0)) if primary else None,
        dram_throughput_pct_peak=primary.dram_throughput_pct_peak if primary else None,
        sm_throughput_pct_peak=primary.sm_throughput_pct_peak if primary else None,
        compute_throughput_pct_peak=primary.compute_throughput_pct_peak if primary else None,
        l2_hit_rate=primary.l2_hit_rate if primary else None,
        roofline_position=primary.roofline_position if primary else None,
        top_kernels=top_kernels,
        primary_kernel_duration_ms=primary.duration_ms if primary else None,
        profiled_kernel_count=len(top_kernels),
        raw_profile_ref=str(report_path.relative_to(writer.run_dir)) if report_path.exists() else None,
        csv_profile_ref=csv_profile_ref,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
        warnings=warnings,
    )
    writer.write_artifact(
        relative_path="profiles/kernel/summary.json",
        kind="kernel_profile_summary",
        content=json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["profile", "nvidia", "summary"],
        producer_event_id=started_event.event_id,
    )
    markdown_lines = [
        f"# Kernel Profile ({profile_pack})",
        "",
        f"- backend: `{report.backend}`",
        f"- exit_code: `{report.exit_code}`",
        f"- primary_kernel: `{report.kernel_name}`",
        f"- classification: `{report.classification}`",
        f"- occupancy: `{report.occupancy}`",
        f"- dram_pct_peak: `{report.dram_throughput_pct_peak}`",
        f"- sm_pct_peak: `{report.sm_throughput_pct_peak}`",
        f"- l2_hit_rate: `{report.l2_hit_rate}`",
        "",
        "## Top Kernels",
    ]
    if top_kernels:
        for record in top_kernels[:5]:
            markdown_lines.append(
                f"- `{record.kernel_name}`: class=`{record.classification}` duration_ms=`{record.duration_ms}` occupancy=`{record.occupancy}`"
            )
    else:
        markdown_lines.append("- none")
    writer.write_artifact(
        relative_path="profiles/kernel/summary.md",
        kind="kernel_profile_markdown",
        content="\n".join(markdown_lines) + "\n",
        mime="text/markdown",
        semantic_tags=["profile", "nvidia", "summary", "markdown"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.profile_kernel_nvidia",
        kind="completed" if result.exit_code == 0 else "failed",
        payload={
            "exit_code": result.exit_code,
            "duration_ms": duration_ms,
            "profile_pack": profile_pack,
            "profiled_kernel_count": report.profiled_kernel_count,
        },
    )
    return report
