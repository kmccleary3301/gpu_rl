from __future__ import annotations

import ast
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


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _resolve_existing_command_files(root: Path, command: list[str]) -> list[Path]:
    resolved: list[Path] = []
    seen: set[Path] = set()
    for token in command:
        candidate = Path(token)
        if not candidate.is_absolute():
            candidate = root / candidate
        candidate = candidate.resolve()
        if not candidate.exists() or not candidate.is_file():
            continue
        if not _path_within_root(candidate, root):
            continue
        if candidate in seen:
            continue
        seen.add(candidate)
        resolved.append(candidate)
    return resolved


def _resolve_python_import_target(importer: Path, root: Path, *, module: str | None, level: int) -> Path | None:
    module_parts = [part for part in (module or "").split(".") if part]
    candidate_paths: list[Path] = []
    if level > 0:
        base = importer.parent
        for _ in range(level - 1):
            base = base.parent
        if module_parts:
            candidate_paths.extend(
                [
                    (base / Path(*module_parts)).with_suffix(".py"),
                    base / Path(*module_parts) / "__init__.py",
                ]
            )
    else:
        for search_root in [importer.parent, root]:
            if module_parts:
                candidate_paths.extend(
                    [
                        (search_root / Path(*module_parts)).with_suffix(".py"),
                        search_root / Path(*module_parts) / "__init__.py",
                    ]
                )
    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file() and _path_within_root(candidate, root):
            return candidate.resolve()
    return None


def _resolve_import_from_alias_targets(
    importer: Path,
    root: Path,
    *,
    module: str | None,
    level: int,
    resolved_module_target: Path | None,
    alias_names: list[str],
) -> list[Path]:
    package_dir: Path | None = None
    if resolved_module_target is not None and resolved_module_target.name == "__init__.py":
        package_dir = resolved_module_target.parent
    elif level > 0:
        package_dir = importer.parent
        for _ in range(level - 1):
            package_dir = package_dir.parent
        if module:
            package_dir = package_dir / Path(*[part for part in module.split(".") if part])
    if package_dir is None:
        return []
    resolved: list[Path] = []
    for alias_name in alias_names:
        if alias_name == "*":
            continue
        for candidate in [
            (package_dir / alias_name).with_suffix(".py"),
            package_dir / alias_name / "__init__.py",
        ]:
            if candidate.exists() and candidate.is_file() and _path_within_root(candidate, root):
                resolved.append(candidate.resolve())
                break
    return resolved


def _trace_python_sources(entrypoints: list[Path], root: Path) -> tuple[list[Path], dict[str, Any]]:
    resolved_files: list[Path] = []
    edges: list[dict[str, Any]] = []
    unresolved_imports: list[dict[str, Any]] = []
    seen_files: set[Path] = set()
    seen_unresolved: set[tuple[str, int, str | None]] = set()
    queue = [path.resolve() for path in entrypoints if path.suffix == ".py"]

    while queue:
        path = queue.pop(0)
        if path in seen_files or not path.exists() or not path.is_file():
            continue
        seen_files.add(path)
        resolved_files.append(path)
        source = _read_text_file(path)
        if source is None:
            continue
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    target = _resolve_python_import_target(path, root, module=module, level=0)
                    if target is not None:
                        edges.append({"importer": str(path), "module": module, "resolved_path": str(target)})
                        if target not in seen_files:
                            queue.append(target)
                    else:
                        key = (str(path), 0, module)
                        if key not in seen_unresolved:
                            seen_unresolved.add(key)
                            unresolved_imports.append({"importer": str(path), "module": module, "level": 0})
            elif isinstance(node, ast.ImportFrom):
                module = node.module
                level = int(node.level)
                target = _resolve_python_import_target(path, root, module=module, level=level)
                if target is not None:
                    edges.append(
                        {
                            "importer": str(path),
                            "module": module,
                            "level": level,
                            "resolved_path": str(target),
                        }
                    )
                    if target not in seen_files:
                        queue.append(target)
                alias_targets = _resolve_import_from_alias_targets(
                    path,
                    root,
                    module=module,
                    level=level,
                    resolved_module_target=target,
                    alias_names=[alias.name for alias in node.names],
                )
                for alias_target in alias_targets:
                    edges.append(
                        {
                            "importer": str(path),
                            "module": module,
                            "level": level,
                            "resolved_path": str(alias_target),
                        }
                    )
                    if alias_target not in seen_files:
                        queue.append(alias_target)
                if target is None and not alias_targets:
                    key = (str(path), level, module)
                    if key not in seen_unresolved:
                        seen_unresolved.add(key)
                        unresolved_imports.append({"importer": str(path), "module": module, "level": level})

    return resolved_files, {
        "entrypoints": [str(path) for path in entrypoints],
        "resolved_python_files": [str(path) for path in resolved_files],
        "unresolved_imports": unresolved_imports,
        "edges": edges,
    }


def resolve_antihack_scan_paths(
    *,
    root: Path,
    command: list[str],
    explicit_scan_paths: list[Path] | None = None,
) -> tuple[list[Path], dict[str, Any]]:
    explicit_scan_paths = explicit_scan_paths or []
    command_files = _resolve_existing_command_files(root, command)
    traced_python_files, import_trace = _trace_python_sources(command_files, root)
    combined: list[Path] = []
    seen: set[Path] = set()
    for candidate in [*command_files, *traced_python_files, *explicit_scan_paths]:
        resolved = candidate.resolve()
        if not resolved.exists():
            continue
        if resolved in seen:
            continue
        seen.add(resolved)
        combined.append(resolved)
    return combined, import_trace


def _pattern_category(pattern: str) -> str:
    lowered = pattern.lower()
    if "cpu_fallback" in lowered or "cpu fallback" in lowered:
        return "cpu_fallback"
    if lowered.startswith("torch.") or lowered.startswith("triton.") or lowered.startswith("cupy.") or lowered.startswith("jax."):
        return "library_shortcut"
    return "forbidden_pattern"


def eval_envelope_counts_as_success(task: TaskSpec, envelope: EvalEnvelope) -> bool:
    non_perf_pass = (
        envelope.compile_gate == "pass"
        and envelope.correctness_gate == "pass"
        and envelope.anti_hack_gate == "pass"
        and envelope.determinism_gate == "pass"
    )
    if not non_perf_pass:
        return False
    if task.verb == "optimize" and task.baseline_ref:
        return envelope.perf_gate == "pass"
    return True


def classify_perf_failure(perf_report: PerfReport) -> list[str]:
    reasons: list[str] = []
    notes = set(perf_report.perf_notes)
    surfaces = perf_report.score_surfaces or {}
    startup = surfaces.get("startup_diagnostics") if isinstance(surfaces.get("startup_diagnostics"), dict) else {}
    if "candidate_startup_dominated" in notes or bool(startup.get("candidate_startup_dominated")):
        reasons.append("startup_dominated_candidate")
    if "baseline_startup_dominated" in notes or bool(startup.get("baseline_startup_dominated")):
        reasons.append("startup_dominated_baseline")
    if not bool((surfaces.get("inprocess_kernel_perf") or {}).get("available")):
        reasons.append("missing_inprocess_timer")
    baseline_kind = str(perf_report.benchmark_provenance.get("baseline_kind", "") or "")
    if "cpu" in baseline_kind:
        reasons.append("cpu_reference_mismatch")
    if not reasons:
        reasons.append("candidate_algorithm_slow")
    return reasons


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


def _default_failure_packet(
    visible_failure_details: dict[str, Any] | None,
    hidden_failure_details: dict[str, Any] | None,
) -> dict[str, Any]:
    for scope in (hidden_failure_details, visible_failure_details):
        if not isinstance(scope, dict):
            continue
        packet = {
            "failure_class": scope.get("code"),
            "benchmark_case_id": scope.get("benchmark_case_id"),
            "case_config_path": scope.get("case_config_path"),
            "observed": scope.get("observed"),
            "expected": scope.get("expected"),
            "suspected_region": scope.get("suspected_region"),
            "likely_next_actions": scope.get("likely_next_actions", []),
            "fix_family": scope.get("fix_family"),
            "confidence": scope.get("confidence"),
        }
        if any(value not in (None, [], {}) for value in packet.values()):
            return packet
    return {}


def _build_learning_reward_trace(
    *,
    task: TaskSpec,
    correctness_gate: str,
    anti_hack_passed: bool,
    determinism_passed: bool,
    perf_gate: str,
    passes_non_perf_gates: bool,
    task_success: bool,
) -> LearningRewardTrace:
    reward_components = {
        "task_success": 0.6 if task_success else 0.0,
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
    elif task.verb == "optimize" and perf_gate == "fail":
        notes.append("perf_gate:fail")
    if not anti_hack_passed:
        notes.append("anti_hack_failed")
    reward_ledger = RewardLedger(
        task_id=task.task_id,
        task_verb=task.verb,
        task_outcome="success" if task_success else "failure",
        trace_usability="trainable_positive" if task_success else "analysis_only",
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
        terminal_state="success" if task_success else "failure",
        task_outcome="success" if task_success else "failure",
        trace_usability="trainable_positive" if task_success else "analysis_only",
        task_success=task_success,
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
    resolved_scan_paths, import_trace = resolve_antihack_scan_paths(root=root, command=command, explicit_scan_paths=scan_paths)
    writer.write_artifact(
        relative_path="eval/import_trace.json",
        kind="import_trace",
        content=json.dumps(import_trace, indent=2) + "\n",
        mime="application/json",
        semantic_tags=["eval", "anti-hack", "import-trace"],
    )
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
        default_failure_packet=_default_failure_packet(visible_failure_details, hidden_failure_details),
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
        scan_paths=resolved_scan_paths,
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
                gate_reasons.extend(classify_perf_failure(perf_report))

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
    task_success = eval_envelope_counts_as_success(task, envelope)
    learning_reward_trace = _build_learning_reward_trace(
        task=task,
        correctness_gate=correctness_gate,
        anti_hack_passed=anti_hack.passed,
        determinism_passed=determinism.passed,
        perf_gate=perf_gate,
        passes_non_perf_gates=passes_non_perf_gates,
        task_success=task_success,
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
