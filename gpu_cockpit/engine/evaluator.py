from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import AntiHackReport, CorrectnessReport, DeterminismReport, EvalEnvelope, HookExecution, LearningRewardTrace, PerfReport, RewardLedger, RewardLedgerEntry, TaskSpec
from gpu_cockpit.contracts.antihack import AntiHackHit
from gpu_cockpit.contracts.trace import SystemTraceSummary
from gpu_cockpit.executors import CommandExecutor, LocalHostToolExecutor
from gpu_cockpit.engine.determinism import run_determinism_check
from gpu_cockpit.engine.run_bundle import RunBundleWriter

FAILURE_JSON_PREFIX = "GPC_FAILURE_JSON:"
EXCLUDED_GOVERNANCE_SIGNALS = [
    "required_artifact_completeness",
    "replay_completeness",
    "build_completeness",
    "profile_completeness",
    "provenance_completeness",
    "benchmark_reporting",
    "sft_collection",
    "rl_reward_trace_readiness",
]


def _resolve_hook_command(root: Path, ref: str) -> list[str]:
    path = Path(ref)
    if not path.is_absolute():
        path = root / ref
    path = path.resolve()
    if path.suffix == ".py":
        return [sys.executable, str(path)]
    if path.suffix == ".sh":
        return ["bash", str(path)]
    return [str(path)]


def _iter_scan_targets(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(candidate for candidate in path.rglob("*") if candidate.is_file())
    return []


def _read_text_file(path: Path, max_bytes: int = 1_000_000) -> str | None:
    try:
        if path.stat().st_size > max_bytes:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _pattern_category(pattern: str) -> str:
    lowered = pattern.lower()
    if "cpu_fallback" in lowered or "cpu fallback" in lowered:
        return "cpu_fallback"
    if lowered.startswith("torch.") or lowered.startswith("triton.") or lowered.startswith("cupy.") or lowered.startswith("jax."):
        return "library_shortcut"
    return "forbidden_pattern"


def run_hook(
    writer: RunBundleWriter,
    root: Path,
    name: str,
    ref: str,
    env: dict[str, str],
    executor: CommandExecutor | None = None,
) -> HookExecution:
    executor = executor or LocalHostToolExecutor()
    started = writer.append_event(scope=f"hook.{name}", kind="started", payload={"ref": ref})
    hook_command = _resolve_hook_command(root, ref)
    result = executor.run(hook_command, env=env, cwd=root)
    stdout_artifact = writer.write_artifact(
        relative_path=f"correctness/{name}_stdout.txt",
        kind="hook_stdout",
        content=result.stdout,
        mime="text/plain",
        semantic_tags=["correctness", "hook", name, "stdout"],
        producer_event_id=started.event_id,
    )
    stderr_artifact = writer.write_artifact(
        relative_path=f"correctness/{name}_stderr.txt",
        kind="hook_stderr",
        content=result.stderr,
        mime="text/plain",
        semantic_tags=["correctness", "hook", name, "stderr"],
        producer_event_id=started.event_id,
    )
    execution = HookExecution(
        name=name,
        ref=ref,
        exit_code=result.exit_code,
        passed=result.exit_code == 0,
        stdout_path=stdout_artifact.path,
        stderr_path=stderr_artifact.path,
    )
    writer.write_artifact(
        relative_path=f"correctness/{name}_summary.json",
        kind="hook_execution",
        content=json.dumps(execution.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["correctness", "hook", name, "summary"],
        producer_event_id=started.event_id,
    )
    writer.append_event(
        scope=f"hook.{name}",
        kind="completed" if execution.passed else "failed",
        payload={"exit_code": execution.exit_code},
    )
    return execution


def build_antihack_report(
    writer: RunBundleWriter,
    task: TaskSpec,
    command: list[str],
    command_summary: SystemTraceSummary | None,
    scan_paths: list[Path] | None = None,
) -> AntiHackReport:
    scan_paths = scan_paths or []
    scanned_locations: list[str] = ["command"]
    hits: list[AntiHackHit] = []
    corpus: list[tuple[str, str]] = [("command", " ".join(command))]

    if command_summary is not None:
        for rel_path in [command_summary.stdout_path, command_summary.stderr_path]:
            if rel_path is None:
                continue
            target = writer.run_dir / rel_path
            if target.exists():
                scanned_locations.append(rel_path)
                corpus.append((rel_path, target.read_text(encoding="utf-8")))

    for path in scan_paths:
        for target in _iter_scan_targets(path):
            text = _read_text_file(target)
            if text is None:
                continue
            location = str(target)
            scanned_locations.append(location)
            corpus.append((location, text))

    lower_cpu_markers = ("cpu fallback", "fall back to cpu", "cpu_fallback")
    cpu_fallback_detected = False
    seen_cpu_locations: set[str] = set()
    for location, text in corpus:
        lowered_text = text.lower()
        for marker in lower_cpu_markers:
            if marker in lowered_text:
                cpu_fallback_detected = True
                if location not in seen_cpu_locations:
                    hits.append(
                        AntiHackHit(
                            pattern="CPU_FALLBACK",
                            category="cpu_fallback",
                            location=location,
                            matched_text=marker,
                        )
                    )
                    seen_cpu_locations.add(location)
                break

    for pattern in task.forbidden_patterns:
        lowered = pattern.lower()
        for location, text in corpus:
            if lowered in text.lower():
                hits.append(
                    AntiHackHit(
                        pattern=pattern,
                        category=_pattern_category(pattern),
                        location=location,
                        matched_text=pattern,
                    )
                )

    library_shortcut_detected = any(hit.category == "library_shortcut" for hit in hits)

    report = AntiHackReport(
        passed=not hits and not cpu_fallback_detected,
        forbidden_patterns=task.forbidden_patterns,
        scanned_locations=scanned_locations,
        hits=hits,
        cpu_fallback_detected=cpu_fallback_detected,
        library_shortcut_detected=library_shortcut_detected,
        warnings=(
            (["CPU fallback markers detected."] if cpu_fallback_detected else [])
            + (["Library shortcut markers detected."] if library_shortcut_detected else [])
        ),
    )
    writer.write_artifact(
        relative_path="eval/anti_hack_report.json",
        kind="anti_hack_report",
        content=json.dumps(report.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["eval", "anti-hack", "summary"],
    )
    return report


def _extract_hook_failure_details(writer: RunBundleWriter, execution: HookExecution | None) -> tuple[str | None, dict[str, Any] | None]:
    if execution is None:
        return None, None
    lines: list[str] = []
    for rel_path in [execution.stderr_path, execution.stdout_path]:
        if rel_path is None:
            continue
        path = writer.run_dir / rel_path
        if not path.exists():
            continue
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                lines.append(line)
    summary: str | None = None
    details: dict[str, Any] | None = None
    for line in lines:
        if line.startswith(FAILURE_JSON_PREFIX):
            payload = line[len(FAILURE_JSON_PREFIX) :].strip()
            try:
                parsed = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                details = parsed
        else:
            summary = line
    return summary, details


def _build_learning_reward_trace(
    *,
    task: TaskSpec,
    correctness_gate: str,
    anti_hack_passed: bool,
    determinism_passed: bool,
    perf_gate: str,
    passes_non_perf_gates: bool,
) -> LearningRewardTrace:
    reward_components = {
        "task_success": 0.6 if passes_non_perf_gates else 0.0,
        "correctness": 0.25 if correctness_gate == "pass" else 0.0,
        "determinism": 0.1 if determinism_passed else 0.0,
        "perf_improvement": 0.05 if perf_gate == "pass" else 0.0,
    }
    shaping_components = {
        "tool_cost": 0.0,
        "compare_use_bonus": 0.0,
    }
    notes: list[str] = []
    if perf_gate in {"blocked", "not_run"}:
        notes.append(f"perf_gate:{perf_gate}")
    if not anti_hack_passed:
        notes.append("anti_hack_failed")
    reward_ledger = RewardLedger(
        task_id=task.task_id,
        task_verb=task.verb,
        task_outcome="success" if passes_non_perf_gates else "failure",
        trace_usability="trainable_positive" if passes_non_perf_gates else "analysis_only",
        entries=[
            RewardLedgerEntry(
                step_index=0,
                action_type="eval",
                reward_components=reward_components,
                shaping_components=shaping_components,
                total_delta=round(sum(reward_components.values()) + sum(shaping_components.values()), 4),
                notes=["run_level_eval_reward"],
            )
        ],
        total_reward_components=reward_components,
        total_shaping_components=shaping_components,
        total_reward=round(sum(reward_components.values()) + sum(shaping_components.values()), 4),
    )
    return LearningRewardTrace(
        task_id=task.task_id,
        task_verb=task.verb,
        terminal_state="success" if passes_non_perf_gates else "failure",
        task_outcome="success" if passes_non_perf_gates else "failure",
        trace_usability="trainable_positive" if passes_non_perf_gates else "analysis_only",
        task_success=passes_non_perf_gates,
        correctness_passed=correctness_gate == "pass",
        determinism_passed=determinism_passed,
        anti_hack_passed=anti_hack_passed,
        perf_gate=perf_gate,
        reward_components=reward_components,
        shaping_components=shaping_components,
        excluded_governance_signals=EXCLUDED_GOVERNANCE_SIGNALS,
        total_reward=round(sum(reward_components.values()) + sum(shaping_components.values()), 4),
        notes=notes,
        reward_ledger=reward_ledger,
    )


def run_evaluation_hooks(
    writer: RunBundleWriter,
    root: Path,
    task: TaskSpec,
    command: list[str],
    command_summary: SystemTraceSummary | None,
    perf_report: PerfReport | None = None,
    scan_paths: list[Path] | None = None,
    determinism_runs: int = 2,
    executor: CommandExecutor | None = None,
) -> tuple[CorrectnessReport, AntiHackReport, DeterminismReport, EvalEnvelope]:
    executor = executor or LocalHostToolExecutor()
    env = {
        "GPC_RUN_DIR": str(writer.run_dir),
        "GPC_COMMAND_JSON": json.dumps(command),
    }
    if command_summary is not None:
        if command_summary.stdout_path is not None:
            env["GPC_STDOUT_PATH"] = str(writer.run_dir / command_summary.stdout_path)
        if command_summary.stderr_path is not None:
            env["GPC_STDERR_PATH"] = str(writer.run_dir / command_summary.stderr_path)

    visible_exec = run_hook(writer, root, "visible_tests", task.visible_tests_ref, env, executor=executor) if task.visible_tests_ref else None
    hidden_exec = run_hook(writer, root, "hidden_tests", task.hidden_tests_ref, env, executor=executor) if task.hidden_tests_ref else None
    visible_failure_summary, visible_failure_details = _extract_hook_failure_details(writer, visible_exec)
    hidden_failure_summary, hidden_failure_details = _extract_hook_failure_details(writer, hidden_exec)

    failures: list[str] = []
    if visible_exec is not None and not visible_exec.passed:
        failures.append("visible_tests_failed")
    if hidden_exec is not None and not hidden_exec.passed:
        failures.append("hidden_tests_failed")

    determinism = run_determinism_check(
        writer=writer,
        command=command,
        baseline=command_summary,
        runs=determinism_runs,
        executor=executor,
    )
    if not determinism.passed:
        failures.append("determinism_failed")

    correctness = CorrectnessReport(
        compile_ok=bool(command_summary is None or command_summary.exit_code == 0),
        visible_tests_ok=visible_exec.passed if visible_exec is not None else None,
        hidden_tests_ok=hidden_exec.passed if hidden_exec is not None else None,
        determinism={
            "runs": determinism.runs,
            "passed": determinism.passed,
            "stable_exit_codes": determinism.stable_exit_codes,
            "stable_stdout": determinism.stable_stdout,
            "stable_stderr": determinism.stable_stderr,
        },
        failures=failures,
        visible_failure_summary=visible_failure_summary,
        hidden_failure_summary=hidden_failure_summary,
        failure_localization={
            key: value
            for key, value in {
                "visible_tests": visible_failure_details,
                "hidden_tests": hidden_failure_details,
            }.items()
            if isinstance(value, dict)
        },
    )
    writer.write_artifact(
        relative_path="correctness/correctness.json",
        kind="correctness_report",
        content=json.dumps(correctness.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["correctness", "summary"],
    )
    writer.write_artifact(
        relative_path="correctness/failure_localization.json",
        kind="failure_localization",
        content=json.dumps(correctness.failure_localization, indent=2) + "\n",
        mime="application/json",
        semantic_tags=["correctness", "failure", "localization"],
    )

    anti_hack = build_antihack_report(
        writer=writer,
        task=task,
        command=command,
        command_summary=command_summary,
        scan_paths=scan_paths,
    )

    correctness_gate = "pass"
    if correctness.visible_tests_ok is False or correctness.hidden_tests_ok is False:
        correctness_gate = "fail"
    if correctness.compile_ok is False:
        correctness_gate = "fail"
    determinism_gate = "pass" if determinism.passed else "fail"
    passes_non_perf_gates = correctness.compile_ok and correctness_gate == "pass" and anti_hack.passed and determinism.passed
    perf_gate = "not_run"
    perf_reward = 0.0
    binary_reward = 1.0 if passes_non_perf_gates else 0.0
    gate_reasons: list[str] = []
    if not correctness.compile_ok:
        gate_reasons.append("compile gate failed")
    if correctness.visible_tests_ok is False:
        gate_reasons.append("visible tests failed")
    if correctness.hidden_tests_ok is False:
        gate_reasons.append("hidden tests failed")
    if not anti_hack.passed:
        gate_reasons.append("anti-hack checks failed")
    if not determinism.passed:
        gate_reasons.append("determinism checks failed")
    if perf_report is not None:
        if not passes_non_perf_gates:
            perf_gate = "blocked"
            binary_reward = 0.0
            gate_reasons.append("perf gate blocked by earlier gates")
        else:
            perf_gate = "pass" if perf_report.speedup_vs_baseline >= 1.0 else "fail"
            binary_reward = 0.8
            perf_reward = 0.2 if perf_gate == "pass" else 0.0
            if perf_gate == "fail":
                gate_reasons.append("candidate slower than baseline")

    envelope = EvalEnvelope(
        compile_gate="pass" if correctness.compile_ok else "fail",
        correctness_gate=correctness_gate,
        anti_hack_gate="pass" if anti_hack.passed else "fail",
        determinism_gate=determinism_gate,
        perf_gate=perf_gate,
        reward_components={
            "binary_pass": binary_reward,
            "determinism": 0.0,
            "perf": perf_reward,
        },
        final_score=binary_reward + perf_reward if passes_non_perf_gates else 0.0,
    )
    learning_reward_trace = _build_learning_reward_trace(
        task=task,
        correctness_gate=correctness_gate,
        anti_hack_passed=anti_hack.passed,
        determinism_passed=determinism.passed,
        perf_gate=perf_gate,
        passes_non_perf_gates=passes_non_perf_gates,
    )
    writer.write_artifact(
        relative_path="eval/eval_envelope.json",
        kind="eval_envelope",
        content=json.dumps(envelope.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["eval", "summary"],
    )
    writer.write_artifact(
        relative_path="eval/learning_reward_trace.json",
        kind="learning_reward_trace",
        content=json.dumps(learning_reward_trace.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["eval", "reward", "learning"],
    )
    writer.write_artifact(
        relative_path="eval/gate_summary.json",
        kind="eval_gate_summary",
        content=json.dumps(
            {
                "compile_gate": envelope.compile_gate,
                "correctness_gate": envelope.correctness_gate,
                "anti_hack_gate": envelope.anti_hack_gate,
                "determinism_gate": envelope.determinism_gate,
                "perf_gate": envelope.perf_gate,
                "reasons": gate_reasons,
            },
            indent=2,
        )
        + "\n",
        mime="application/json",
        semantic_tags=["eval", "gates", "summary"],
    )
    return correctness, anti_hack, determinism, envelope
