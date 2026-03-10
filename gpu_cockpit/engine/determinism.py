from __future__ import annotations

import hashlib
import json

from gpu_cockpit.contracts import DeterminismAttempt, DeterminismReport, SystemTraceSummary
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.run_bundle import RunBundleWriter


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_attempt(attempt_index: int, exit_code: int, duration_ms: int, stdout: str, stderr: str) -> DeterminismAttempt:
    return DeterminismAttempt(
        attempt_index=attempt_index,
        exit_code=exit_code,
        duration_ms=duration_ms,
        stdout_sha256=_sha256_text(stdout),
        stderr_sha256=_sha256_text(stderr),
        stdout_bytes=len(stdout.encode("utf-8")),
        stderr_bytes=len(stderr.encode("utf-8")),
    )


def run_determinism_check(
    writer: RunBundleWriter,
    command: list[str],
    baseline: SystemTraceSummary | None,
    runs: int = 2,
    executor: CommandExecutor | None = None,
) -> DeterminismReport:
    executor = executor or LocalHostToolExecutor()
    target_runs = max(1, runs)
    attempts: list[DeterminismAttempt] = []
    warnings: list[str] = []

    baseline_stdout = ""
    baseline_stderr = ""
    baseline_exit_code: int | None = None
    if baseline is not None:
        baseline_exit_code = baseline.exit_code
        if baseline.stdout_path:
            baseline_stdout_path = writer.run_dir / baseline.stdout_path
            if baseline_stdout_path.exists():
                baseline_stdout = baseline_stdout_path.read_text(encoding="utf-8")
            else:
                warnings.append(f"Missing baseline stdout artifact: {baseline.stdout_path}")
        if baseline.stderr_path:
            baseline_stderr_path = writer.run_dir / baseline.stderr_path
            if baseline_stderr_path.exists():
                baseline_stderr = baseline_stderr_path.read_text(encoding="utf-8")
            else:
                warnings.append(f"Missing baseline stderr artifact: {baseline.stderr_path}")
        attempts.append(
            _build_attempt(
                attempt_index=1,
                exit_code=baseline.exit_code,
                duration_ms=baseline.duration_ms,
                stdout=baseline_stdout,
                stderr=baseline_stderr,
            )
        )

    started = writer.append_event(
        scope="eval.determinism",
        kind="started",
        payload={"command": command, "runs": target_runs},
    )

    next_index = len(attempts) + 1
    while len(attempts) < target_runs:
        result = executor.run(command, cwd=writer.root)
        attempts.append(
            _build_attempt(
                attempt_index=next_index,
                exit_code=result.exit_code,
                duration_ms=result.duration_ms,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        next_index += 1

    exit_codes = {attempt.exit_code for attempt in attempts}
    stdout_hashes = {attempt.stdout_sha256 for attempt in attempts}
    stderr_hashes = {attempt.stderr_sha256 for attempt in attempts}
    stable_exit_codes = len(exit_codes) == 1
    stable_stdout = len(stdout_hashes) == 1
    stable_stderr = len(stderr_hashes) == 1
    passed = stable_exit_codes and stable_stdout and stable_stderr

    if not stable_exit_codes:
        warnings.append("Exit codes changed across determinism reruns.")
    if not stable_stdout:
        warnings.append("Stdout changed across determinism reruns.")
    if not stable_stderr:
        warnings.append("Stderr changed across determinism reruns.")

    report = DeterminismReport(
        command=command,
        runs=len(attempts),
        passed=passed,
        stable_exit_codes=stable_exit_codes,
        stable_stdout=stable_stdout,
        stable_stderr=stable_stderr,
        baseline_exit_code=baseline_exit_code,
        baseline_stdout_sha256=attempts[0].stdout_sha256 if attempts else None,
        baseline_stderr_sha256=attempts[0].stderr_sha256 if attempts else None,
        attempts=attempts,
        warnings=warnings,
    )
    writer.write_artifact(
        relative_path="correctness/determinism.json",
        kind="determinism_report",
        content=json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["correctness", "determinism", "summary"],
        producer_event_id=started.event_id,
    )
    writer.append_event(
        scope="eval.determinism",
        kind="completed" if passed else "failed",
        payload={
            "runs": report.runs,
            "passed": report.passed,
            "stable_exit_codes": stable_exit_codes,
            "stable_stdout": stable_stdout,
            "stable_stderr": stable_stderr,
        },
    )
    return report
