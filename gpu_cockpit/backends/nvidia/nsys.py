from __future__ import annotations

import json
import shutil
import time
from pathlib import Path

from gpu_cockpit.contracts import SystemTraceSummary
from gpu_cockpit.executors import LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def trace_system_nvidia(
    writer: RunBundleWriter,
    command: list[str],
    executor: LocalHostToolExecutor | None = None,
) -> SystemTraceSummary:
    if shutil.which("nsys") is None:
        raise RuntimeError("nsys is not installed.")
    executor = executor or LocalHostToolExecutor()

    started_event = writer.append_event(
        scope="tool.trace_system_nvidia",
        kind="started",
        payload={"command": command},
    )
    started = time.monotonic()
    trace_prefix = writer.artifacts_dir / "traces" / "system" / "profile"
    trace_prefix.parent.mkdir(parents=True, exist_ok=True)

    profile_cmd = [
        "nsys",
        "profile",
        "--sample=none",
        "--trace=cuda,nvtx,osrt",
        "--export=none",
        "--force-overwrite=true",
        "-o",
        str(trace_prefix),
        *command,
    ]
    result = executor.run(profile_cmd)
    duration_ms = int((time.monotonic() - started) * 1000)

    stdout_artifact = writer.write_artifact(
        relative_path="traces/system/nsys_stdout.txt",
        kind="nsys_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["trace", "nvidia", "stdout"],
        producer_event_id=started_event.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path="traces/system/nsys_stderr.txt",
        kind="nsys_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["trace", "nvidia", "stderr"],
        producer_event_id=started_event.event_id,
    )

    rep_path = trace_prefix.with_suffix(".nsys-rep")
    sqlite_path: Path | None = None
    warnings: list[str] = []
    warnings.extend(
        line.strip() for line in result.stderr.splitlines() if line.strip().startswith("WARNING:")
    )

    if rep_path.exists():
        sqlite_path = trace_prefix.with_suffix(".sqlite")
        export_cmd = [
            "nsys",
            "export",
            "--type",
            "sqlite",
            "--force-overwrite=true",
            "--output",
            str(sqlite_path),
            str(rep_path),
        ]
        export_result = executor.run(export_cmd)
        writer.write_artifact(
            relative_path="traces/system/nsys_export_stdout.txt",
            kind="nsys_export_stdout",
            content=export_result.stdout,
            mime="text/plain",
            semantic_tags=["trace", "nvidia", "export", "stdout"],
            producer_event_id=started_event.event_id,
        )
        writer.write_artifact(
            relative_path="traces/system/nsys_export_stderr.txt",
            kind="nsys_export_stderr",
            content=export_result.stderr,
            mime="text/plain",
            semantic_tags=["trace", "nvidia", "export", "stderr"],
            producer_event_id=started_event.event_id,
        )
        if export_result.exit_code != 0:
            warnings.append("nsys export to sqlite failed")
            sqlite_path = None
    else:
        warnings.append("nsys report file was not created")

    summary = SystemTraceSummary(
        backend="nvidia_nsys",
        command=command,
        trace_enabled=True,
        exit_code=result.exit_code,
        duration_ms=duration_ms,
        report_path=str(rep_path.relative_to(writer.run_dir)) if rep_path.exists() else None,
        sqlite_path=str(sqlite_path.relative_to(writer.run_dir)) if sqlite_path and sqlite_path.exists() else None,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
        warnings=warnings,
    )
    writer.write_artifact(
        relative_path="traces/system/summary.json",
        kind="system_trace_summary",
        content=json.dumps(summary.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["trace", "nvidia", "summary"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.trace_system_nvidia",
        kind="completed" if result.exit_code == 0 else "failed",
        payload={"exit_code": result.exit_code, "duration_ms": duration_ms, "report_path": summary.report_path},
    )
    return summary
