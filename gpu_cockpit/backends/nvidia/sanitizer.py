from __future__ import annotations

import json
import re
import shutil
import time

from gpu_cockpit.contracts import SanitizerFinding, SanitizerReport
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter

_KERNEL_RE = re.compile(r"at\s+(?P<kernel>[^()\s]+)\([^)]*\)")
_FILE_RE = re.compile(r"in\s+(?P<path>[^:\s]+):(?P<line>\d+)")


def _categorize_message(message: str, tool: str) -> tuple[str, str, str, str | None]:
    lowered = message.lower()
    if "invalid" in lowered or "out of bounds" in lowered or "misaligned" in lowered:
        return "memory_access_error", "error", "memory_safety", "inspect memory indexing, bounds checks, and pointer alignment"
    if "race" in lowered:
        return "data_race", "error", "concurrency", "inspect shared-state writes and synchronization ordering"
    if "uninitialized" in lowered:
        return "uninitialized_access", "error", "initialization", "inspect initialization order and masked loads/stores"
    if "barrier" in lowered or "sync" in lowered:
        return "sync_error", "error", "synchronization", "inspect barrier placement and warp/block synchronization assumptions"
    if "warning" in lowered:
        return "tool_warning", "warning", "tooling", None
    return f"{tool}_issue", "error", "unknown", None


def _parse_sanitizer_findings(tool: str, raw_log: str) -> list[SanitizerFinding]:
    findings: list[SanitizerFinding] = []
    for raw_line in raw_log.splitlines():
        line = raw_line.strip()
        if not line.startswith("========="):
            continue
        payload = line.removeprefix("=========").strip()
        if not payload or payload.lower().startswith("error summary"):
            continue
        category, severity, failure_family, remediation_hint = _categorize_message(payload, tool)
        kernel_match = _KERNEL_RE.search(payload)
        file_match = _FILE_RE.search(payload)
        findings.append(
            SanitizerFinding(
                tool=tool,
                category=category,
                severity=severity,
                failure_family=failure_family,
                message=payload,
                remediation_hint=remediation_hint,
                kernel_name=kernel_match.group("kernel") if kernel_match else None,
                file_path=file_match.group("path") if file_match else None,
                line=int(file_match.group("line")) if file_match else None,
                raw_line=raw_line,
            )
        )
    return findings


def sanitize_nvidia(
    writer: RunBundleWriter,
    command: list[str],
    tool: str = "memcheck",
    executor: CommandExecutor | None = None,
) -> SanitizerReport:
    if shutil.which("compute-sanitizer") is None:
        raise RuntimeError("compute-sanitizer is not installed.")
    executor = executor or LocalHostToolExecutor()
    started_event = writer.append_event(
        scope="tool.sanitize_nvidia",
        kind="started",
        payload={"command": command, "tool": tool},
    )
    started = time.monotonic()
    sanitize_cmd = [
        "compute-sanitizer",
        "--tool",
        tool,
        "--log-file",
        str(writer.artifacts_dir / "sanitize" / f"{tool}.log"),
        *command,
    ]
    (writer.artifacts_dir / "sanitize").mkdir(parents=True, exist_ok=True)
    result = executor.run(sanitize_cmd)
    duration_ms = int((time.monotonic() - started) * 1000)

    stdout_artifact = writer.write_artifact(
        relative_path=f"sanitize/{tool}_stdout.txt",
        kind="sanitizer_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["sanitize", "nvidia", tool, "stdout"],
        producer_event_id=started_event.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path=f"sanitize/{tool}_stderr.txt",
        kind="sanitizer_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["sanitize", "nvidia", tool, "stderr"],
        producer_event_id=started_event.event_id,
    )

    log_path = writer.artifacts_dir / "sanitize" / f"{tool}.log"
    raw_log = log_path.read_text(encoding="utf-8") if log_path.exists() else result.stderr
    findings = _parse_sanitizer_findings(tool, raw_log)
    warnings: list[str] = []
    if not findings and result.exit_code != 0:
        warnings.append("sanitizer exited non-zero without parseable findings")
    severity_counts: dict[str, int] = {}
    category_counts: dict[str, int] = {}
    failure_family_counts: dict[str, int] = {}
    for finding in findings:
        severity_counts[finding.severity] = severity_counts.get(finding.severity, 0) + 1
        category_counts[finding.category] = category_counts.get(finding.category, 0) + 1
        if finding.failure_family:
            failure_family_counts[finding.failure_family] = failure_family_counts.get(finding.failure_family, 0) + 1
    dominant_failure_family = None
    if failure_family_counts:
        dominant_failure_family = max(
            sorted(failure_family_counts.items()),
            key=lambda item: item[1],
        )[0]
    triage_summary = []
    for finding in findings[:5]:
        location_parts = []
        if finding.kernel_name:
            location_parts.append(f"kernel={finding.kernel_name}")
        if finding.file_path and finding.line:
            location_parts.append(f"site={finding.file_path}:{finding.line}")
        location = f" ({', '.join(location_parts)})" if location_parts else ""
        triage_summary.append(f"{finding.failure_family or finding.category}: {finding.message}{location}")

    report = SanitizerReport(
        backend="nvidia_compute_sanitizer",
        tool=tool,
        command=command,
        exit_code=result.exit_code,
        duration_ms=duration_ms,
        passed=result.exit_code == 0 and not findings,
        error_count=sum(1 for finding in findings if finding.severity == "error"),
        warning_count=sum(1 for finding in findings if finding.severity == "warning"),
        severity_counts=severity_counts,
        category_counts=category_counts,
        failure_family_counts=failure_family_counts,
        dominant_failure_family=dominant_failure_family,
        triage_summary=triage_summary,
        findings=findings,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
        raw_log_ref=str(log_path.relative_to(writer.run_dir)) if log_path.exists() else None,
        warnings=warnings,
    )
    writer.write_artifact(
        relative_path=f"sanitize/{tool}_summary.json",
        kind="sanitizer_summary",
        content=json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["sanitize", "nvidia", tool, "summary"],
        producer_event_id=started_event.event_id,
    )
    writer.write_artifact(
        relative_path=f"sanitize/{tool}_summary.md",
        kind="sanitizer_markdown",
        content="\n".join(
            [
                f"# Compute Sanitizer ({tool})",
                "",
                f"- exit_code: `{report.exit_code}`",
                f"- passed: `{report.passed}`",
                f"- error_count: `{report.error_count}`",
                f"- warning_count: `{report.warning_count}`",
                f"- severity_counts: `{report.severity_counts}`",
                f"- category_counts: `{report.category_counts}`",
                f"- failure_family_counts: `{report.failure_family_counts}`",
                f"- dominant_failure_family: `{report.dominant_failure_family}`",
                "",
                "## Findings",
                *(
                    [
                        f"- `{finding.failure_family or finding.category}`: {finding.message}"
                        + (f" (`{finding.kernel_name}`)" if finding.kernel_name else "")
                        + (f" at `{finding.file_path}:{finding.line}`" if finding.file_path and finding.line else "")
                        for finding in findings
                    ]
                    or ["- none"]
                ),
                "",
                "## Triage Summary",
                *(triage_summary or ["- none"]),
            ]
        )
        + "\n",
        mime="text/markdown",
        semantic_tags=["sanitize", "nvidia", tool, "summary", "markdown"],
        producer_event_id=started_event.event_id,
    )
    writer.append_event(
        scope="tool.sanitize_nvidia",
        kind="completed" if report.passed else "failed",
        payload={
            "tool": tool,
            "exit_code": report.exit_code,
            "duration_ms": duration_ms,
            "error_count": report.error_count,
        },
    )
    return report
