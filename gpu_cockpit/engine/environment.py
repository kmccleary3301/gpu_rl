from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from gpu_cockpit.contracts import (
    AgentActionSpec,
    AgentEnvironmentState,
    BaselineSpec,
    RunSummary,
    TrajectoryAction,
    TrajectoryEpisode,
    TrajectoryObservation,
    TrajectoryStep,
)
from gpu_cockpit.engine.adapter_registry import get_adapter
from gpu_cockpit.engine.benchmark import run_task_benchmark
from gpu_cockpit.engine.doctor import collect_doctor_report
from gpu_cockpit.engine.evaluator import run_evaluation_hooks
from gpu_cockpit.engine.indexer import list_runs
from gpu_cockpit.engine.inspector import compare_runs, inspect_run, resolve_run_dir
from gpu_cockpit.engine.knowledge import query_knowledge
from gpu_cockpit.engine.replay import validate_run_bundle, write_replay_pack
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.runner import build_run_id, build_run_spec, run_task, write_run_summary, write_task_artifacts
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.executors import make_executor


ACTION_SPACE: tuple[AgentActionSpec, ...] = (
    AgentActionSpec(
        action_name="run",
        description="Execute a task command and create a run bundle.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
    ),
    AgentActionSpec(
        action_name="build",
        description="Compile or emit build/disassembly artifacts for a task.",
        requires_task=True,
        produces_run_bundle=True,
    ),
    AgentActionSpec(
        action_name="bench",
        description="Benchmark a task command and emit a perf bundle.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
    ),
    AgentActionSpec(
        action_name="eval",
        description="Run task evaluation hooks and emit correctness and reward artifacts.",
        requires_task=True,
        requires_command=True,
        produces_run_bundle=True,
    ),
    AgentActionSpec(
        action_name="inspect",
        description="Project a compact section from a run bundle.",
    ),
    AgentActionSpec(
        action_name="compare",
        description="Compare two run bundles.",
    ),
    AgentActionSpec(
        action_name="replay",
        description="Validate replay completeness for a run bundle.",
    ),
    AgentActionSpec(
        action_name="adapter_show",
        description="Load a benchmark case and derived task from a registered adapter.",
    ),
    AgentActionSpec(
        action_name="knowledge_query",
        description="Query the local knowledge index for related docs, tasks, and runs.",
    ),
)


def list_action_space() -> list[AgentActionSpec]:
    return [spec.model_copy(deep=True) for spec in ACTION_SPACE]


def initialize_environment_state(
    root: Path,
    task_ref: str,
    *,
    policy_id: str = "scripted_reference_v1",
    step_budget: int = 5,
) -> AgentEnvironmentState:
    task = TaskRegistry(root).get(task_ref)
    return AgentEnvironmentState(
        episode_id=f"env_episode_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}",
        policy_id=policy_id,
        task_id=task.task_id,
        step_budget_total=step_budget,
        step_budget_remaining=step_budget,
        metadata={
            "operator_family": task.operator_family,
            "verb": task.verb,
            "difficulty": task.difficulty,
            "allowed_backends": list(task.allowed_backends),
        },
    )


def _json_default(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _derive_eval_rewards(run_dir: Path) -> tuple[dict[str, float], float]:
    eval_envelope_path = run_dir / "eval" / "eval_envelope.json"
    if not eval_envelope_path.exists():
        return {"completion": 1.0 if (run_dir / "summary.json").exists() else 0.0}, 1.0 if (run_dir / "summary.json").exists() else 0.0
    payload = json.loads(eval_envelope_path.read_text(encoding="utf-8"))
    reward_components = payload.get("reward_components", {})
    normalized = {
        str(key): float(value)
        for key, value in reward_components.items()
        if isinstance(value, (int, float))
    }
    final_score = payload.get("final_score")
    return normalized, float(final_score) if isinstance(final_score, (int, float)) else 0.0


def _evidence_quality(observation: TrajectoryObservation) -> float:
    artifact_count = len(observation.artifact_refs)
    if artifact_count >= 6:
        return 0.15
    if artifact_count >= 3:
        return 0.1
    if artifact_count >= 1:
        return 0.05
    return 0.0


def _update_state_after_run(state: AgentEnvironmentState, run_ref: str, run_id: str) -> AgentEnvironmentState:
    run_history = list(state.run_history)
    run_history.append(run_ref)
    return state.model_copy(
        update={
            "steps_taken": state.steps_taken + 1,
            "step_budget_remaining": max(0, state.step_budget_remaining - 1),
            "last_run_id": run_id,
            "last_run_ref": run_ref,
            "run_history": run_history,
        }
    )


def _update_state_after_info(state: AgentEnvironmentState) -> AgentEnvironmentState:
    return state.model_copy(
        update={
            "steps_taken": state.steps_taken + 1,
            "step_budget_remaining": max(0, state.step_budget_remaining - 1),
        }
    )


def _finalize_budget(state: AgentEnvironmentState) -> AgentEnvironmentState:
    if state.step_budget_remaining > 0:
        return state
    return state.model_copy(update={"done": True, "terminal_state": "budget_exhausted"})


def _resolve_task(root: Path, task_ref: str):
    return TaskRegistry(root).get(task_ref)


def _load_baseline(root: Path, task_ref: str) -> BaselineSpec | None:
    task = _resolve_task(root, task_ref)
    if not task.baseline_ref:
        return None
    path = root / task.baseline_ref
    if not path.exists():
        return None
    return BaselineSpec.model_validate(json.loads(path.read_text(encoding="utf-8")))


def _execute_eval(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
) -> Path:
    task = _resolve_task(root, task_ref)
    doctor_report = collect_doctor_report()
    command_executor = make_executor(executor, root)
    run_spec = build_run_spec(
        task=task,
        backend=backend or (task.allowed_backends[0] if task.allowed_backends else "triton"),
        vendor=vendor or "nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    run_spec.run_id = build_run_id(prefix="eval")
    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)

    from gpu_cockpit.engine.command_runner import run_command

    command_summary = run_command(writer, command, executor=command_executor)
    perf_report = None
    if task.baseline_ref:
        perf_report = run_task_benchmark(
            writer,
            root=root,
            task=task,
            command=command,
            scope="tool.run_benchmark.eval",
            executor=command_executor,
        )

    correctness, anti_hack, determinism, envelope = run_evaluation_hooks(
        writer=writer,
        root=root,
        task=task,
        command=command,
        command_summary=command_summary,
        perf_report=perf_report,
        scan_paths=[],
        determinism_runs=determinism_runs,
        executor=command_executor,
    )

    status = "ok" if envelope.final_score > 0 else "failed"
    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "correctness/correctness.json",
        "correctness/determinism.json",
        "eval/anti_hack_report.json",
        "eval/eval_envelope.json",
        "eval/gate_summary.json",
        "summary.json",
        "summary.md",
    ]
    if doctor_report.hardware_fingerprints:
        key_artifacts.insert(4, "meta/hardware_fingerprint.json")
    if perf_report is not None:
        key_artifacts.extend(["perf/benchmark.json", "perf/raw_timings.json"])
    if command_summary is not None:
        key_artifacts.extend(["command/stdout.txt", "command/stderr.txt", "command/summary.json"])
    write_run_summary(
        writer,
        RunSummary(
            run_id=run_spec.run_id,
            task_id=task.task_id,
            status=status,
            trace_enabled=False,
            backend=run_spec.target_backend,
            vendor=run_spec.target_vendor,
            exit_code=command_summary.exit_code if command_summary is not None else None,
            duration_ms=command_summary.duration_ms if command_summary is not None else None,
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": run_spec.target_backend,
            "target_vendor": run_spec.target_vendor,
            "determinism_runs": determinism_runs,
        },
    )
    writer.append_event(scope="run", kind="completed", payload={"status": status})
    return run_dir


def _execute_bench(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    warmups: int = 1,
    repeats: int = 5,
) -> Path:
    task = _resolve_task(root, task_ref).model_copy(deep=True)
    task.perf_protocol.warmups = warmups
    task.perf_protocol.repeats = repeats
    doctor_report = collect_doctor_report()
    command_executor = make_executor(executor, root)
    run_spec = build_run_spec(
        task=task,
        backend=task.allowed_backends[0] if task.allowed_backends else "triton",
        vendor="nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    run_spec.run_id = build_run_id(prefix="bench")
    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)
    perf = run_task_benchmark(writer, root=root, task=task, command=command, executor=command_executor)
    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "perf/benchmark.json",
        "perf/raw_timings.json",
        "summary.json",
        "summary.md",
    ]
    if doctor_report.hardware_fingerprints:
        key_artifacts.insert(4, "meta/hardware_fingerprint.json")
    write_run_summary(
        writer,
        RunSummary(
            run_id=run_spec.run_id,
            task_id=task.task_id,
            status="ok",
            trace_enabled=False,
            backend=run_spec.target_backend,
            vendor=run_spec.target_vendor,
            exit_code=0,
            duration_ms=int(perf.steady_state_ms_p50),
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=command,
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": run_spec.target_backend,
            "target_vendor": run_spec.target_vendor,
            "warmups": warmups,
            "repeats": repeats,
        },
    )
    writer.append_event(scope="run", kind="completed", payload={"status": "ok"})
    return run_dir


def step_environment(
    root: Path,
    state: AgentEnvironmentState,
    *,
    action_name: str,
    task_ref: str | None = None,
    command: list[str] | None = None,
    run_ref: str | None = None,
    lhs_run_ref: str | None = None,
    rhs_run_ref: str | None = None,
    section: str = "summary",
    query: str | None = None,
    adapter_name: str | None = None,
    case_id: str | None = None,
    triton_build_spec: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
    warmups: int = 1,
    repeats: int = 5,
) -> tuple[AgentEnvironmentState, TrajectoryStep]:
    if state.done:
        raise RuntimeError("Environment is already terminal.")
    if state.step_budget_remaining <= 0:
        raise RuntimeError("Environment step budget is exhausted.")

    action_metadata: dict[str, object] = {"section": section}
    artifact_refs: list[str] = []
    observation: TrajectoryObservation
    reward_components: dict[str, float] = {}
    reward_total = 0.0
    next_state = state

    if action_name == "knowledge_query":
        task = _resolve_task(root, task_ref or state.task_id)
        rows = query_knowledge(
            root,
            query=query or f"{task.operator_family} {task.verb} {' '.join(task.feature_requirements)}",
            operator_family=task.operator_family,
            verb=task.verb,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            limit=5,
            prefer_mixed=True,
        )
        observation = TrajectoryObservation(
            observation_type="knowledge_results",
            task_id=task.task_id,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            vendor=vendor,
            projection={"results": rows},
        )
        reward_components = {"tool_use_quality": 0.05 if rows else 0.0}
        reward_total = reward_components["tool_use_quality"]
        next_state = _update_state_after_info(state)
    elif action_name == "adapter_show":
        if not adapter_name or not case_id:
            raise RuntimeError("adapter_show requires adapter_name and case_id.")
        adapter = get_adapter(adapter_name)
        case = adapter.load_case(root, case_id)
        task = adapter.load_task(root, case_id)
        observation = TrajectoryObservation(
            observation_type="adapter_case",
            task_id=task.task_id,
            backend=task.allowed_backends[0] if task.allowed_backends else None,
            projection={
                "case": case.model_dump(mode="json"),
                "task": task.model_dump(mode="json"),
            },
        )
        reward_components = {"tool_use_quality": 0.02}
        reward_total = 0.02
        next_state = _update_state_after_info(state)
    elif action_name == "run":
        if not task_ref or not command:
            raise RuntimeError("run requires task_ref and command.")
        run_dir = run_task(
            root=root,
            task_ref=task_ref,
            command=command,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
        )
        payload = inspect_run(root, str(run_dir), section=section)
        observation = TrajectoryObservation(
            observation_type="run_bundle_projection",
            run_id=str(payload.get("run_id")),
            task_id=str(payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(payload.get("backend")),
            vendor=str(payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(payload.get("key_artifacts", [])),
            projection=payload,
        )
        reward_components = {"completion": 1.0 if payload.get("status") == "ok" else 0.0, "evidence_quality": _evidence_quality(observation)}
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(payload.get("run_id")))
    elif action_name == "build":
        if not task_ref:
            raise RuntimeError("build requires task_ref.")
        run_dir = run_task(
            root=root,
            task_ref=task_ref,
            command=command or [],
            emit_disassembly=True,
            triton_build_spec=triton_build_spec,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
        )
        payload = inspect_run(root, str(run_dir), section="build")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        observation = TrajectoryObservation(
            observation_type="build_projection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            projection=payload,
        )
        reward_components = {"tool_use_quality": 0.05, "evidence_quality": _evidence_quality(observation)}
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id")))
    elif action_name == "eval":
        if not task_ref or not command:
            raise RuntimeError("eval requires task_ref and command.")
        run_dir = _execute_eval(
            root,
            task_ref,
            command,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            determinism_runs=determinism_runs,
        )
        payload = inspect_run(root, str(run_dir), section=section if section != "summary" else "eval")
        summary_payload = inspect_run(root, str(run_dir), section="summary")
        reward_components, reward_total = _derive_eval_rewards(run_dir)
        reward_components.setdefault("evidence_quality", _evidence_quality(TrajectoryObservation(observation_type="tmp", artifact_refs=list(summary_payload.get("key_artifacts", [])))))
        reward_total += reward_components["evidence_quality"]
        observation = TrajectoryObservation(
            observation_type="eval_projection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            projection=payload,
        )
        next_state = _update_state_after_run(state, str(run_dir), str(summary_payload.get("run_id"))).model_copy(
            update={
                "metadata": {**state.metadata, "last_eval_solved": summary_payload.get("status") == "ok"},
                "comparison_anchor_run_ref": state.comparison_anchor_run_ref or str(run_dir),
                "comparison_anchor_label": state.comparison_anchor_label or "primary_eval",
            }
        )
    elif action_name == "bench":
        if not task_ref or not command:
            raise RuntimeError("bench requires task_ref and command.")
        run_dir = _execute_bench(
            root,
            task_ref,
            command,
            executor=executor,
            policy_pack=policy_pack,
            warmups=warmups,
            repeats=repeats,
        )
        payload = inspect_run(root, str(run_dir), section="summary")
        observation = TrajectoryObservation(
            observation_type="bench_projection",
            run_id=str(payload.get("run_id")),
            task_id=str(payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(payload.get("backend")),
            vendor=str(payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(payload.get("key_artifacts", [])),
            projection=inspect_run(root, str(run_dir), section="full"),
        )
        reward_components = {"tool_use_quality": 0.03, "evidence_quality": _evidence_quality(observation)}
        reward_total = sum(reward_components.values())
        next_state = _update_state_after_run(state, str(run_dir), str(payload.get("run_id")))
    elif action_name == "inspect":
        target_run_ref = run_ref or state.last_run_ref
        if not target_run_ref:
            raise RuntimeError("inspect requires run_ref or a prior run-producing action.")
        payload = inspect_run(root, target_run_ref, section=section)
        summary_payload = inspect_run(root, target_run_ref, section="summary")
        observation = TrajectoryObservation(
            observation_type="inspection",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(summary_payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="summary.json",
            artifact_refs=list(summary_payload.get("key_artifacts", [])),
            projection=payload,
        )
        reward_components = {"evidence_quality": _evidence_quality(observation)}
        reward_total = reward_components["evidence_quality"]
        next_state = _update_state_after_info(state)
    elif action_name == "compare":
        lhs = lhs_run_ref or state.last_run_ref
        rhs = rhs_run_ref
        if not lhs or not rhs:
            raise RuntimeError("compare requires lhs_run_ref and rhs_run_ref, or an existing last_run_ref plus rhs_run_ref.")
        payload = compare_runs(root, lhs, rhs).model_dump(mode="json")
        observation = TrajectoryObservation(
            observation_type="comparison",
            projection=payload,
        )
        reward_components = {"tool_use_quality": 0.02}
        reward_total = reward_components["tool_use_quality"]
        next_state = _update_state_after_info(state)
    elif action_name == "replay":
        target_run_ref = run_ref or state.last_run_ref
        if not target_run_ref:
            raise RuntimeError("replay requires run_ref or a prior run-producing action.")
        payload = validate_run_bundle(root, target_run_ref)
        resolved_run_dir = resolve_run_dir(root, target_run_ref)
        summary_payload = inspect_run(root, str(resolved_run_dir), section="summary")
        artifact_refs = list(payload.get("required_artifacts", []))
        observation = TrajectoryObservation(
            observation_type="replay_validation",
            run_id=str(summary_payload.get("run_id")),
            task_id=str(summary_payload.get("task_id")),
            status=str(payload.get("status")),
            backend=str(summary_payload.get("backend")),
            vendor=str(summary_payload.get("vendor")),
            summary_ref="replay/replay_pack.json",
            artifact_refs=artifact_refs,
            projection=payload,
        )
        reward_components = {"evidence_quality": 0.15 if payload.get("status") == "ok" else 0.0}
        reward_total = reward_components["evidence_quality"]
        next_state = _update_state_after_info(state)
    else:
        raise ValueError(f"Unsupported environment action: {action_name}")

    action = TrajectoryAction(
        action_type=action_name,
        step_kind="tool_call",
        command=list(command or []),
        target_run_id=observation.run_id,
        target_task_id=observation.task_id or state.task_id,
        artifact_refs=artifact_refs,
        metadata=action_metadata,
    )
    step = TrajectoryStep(
        step_index=state.steps_taken,
        action=action,
        observation=observation,
        reward_components=reward_components,
        reward_total=reward_total,
    )
    next_state = _finalize_budget(next_state)
    return next_state, step


def run_scripted_reference_episode(
    root: Path,
    task_ref: str,
    command: list[str],
    *,
    policy_id: str = "scripted_reference_v1",
    step_budget: int = 5,
    section: str = "summary",
    include_knowledge: bool = True,
    include_build: bool = False,
    triton_build_spec: str | None = None,
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
    policy_pack: str = "balanced",
    determinism_runs: int = 2,
    workflow: str = "auto",
) -> TrajectoryEpisode:
    state = initialize_environment_state(root, task_ref, policy_id=policy_id, step_budget=step_budget)
    task = _resolve_task(root, task_ref)
    steps: list[TrajectoryStep] = []
    selected_workflow = workflow
    if selected_workflow == "auto":
        if task.verb == "reformulate":
            selected_workflow = "reformulate"
        elif task.verb == "debug":
            selected_workflow = "debug"
        elif task.verb == "diagnose":
            selected_workflow = "diagnose_first"
        else:
            selected_workflow = "standard"

    if include_knowledge and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="knowledge_query",
            task_ref=task.task_id,
        )
        steps.append(step)

    if include_build and triton_build_spec and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="build",
            task_ref=task.task_id,
            command=[],
            section="build",
            triton_build_spec=triton_build_spec,
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
        )
        steps.append(step)

    baseline = _load_baseline(root, task.task_id)
    if selected_workflow == "reformulate" and baseline and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="bench",
            task_ref=task.task_id,
            command=list(baseline.command),
            section="summary",
            executor=executor,
            policy_pack=policy_pack,
        )
        state = state.model_copy(
            update={
                "comparison_anchor_run_ref": state.last_run_ref,
                "comparison_anchor_label": "baseline_bench",
            }
        )
        steps.append(step)

    if state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="eval",
            task_ref=task.task_id,
            command=command,
            section="eval",
            backend=backend,
            vendor=vendor,
            executor=executor,
            policy_pack=policy_pack,
            determinism_runs=determinism_runs,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="inspect",
            run_ref=state.last_run_ref,
            section="quality" if selected_workflow in {"debug", "reformulate"} else section,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        state, step = step_environment(
            root,
            state,
            action_name="replay",
            run_ref=state.last_run_ref,
        )
        steps.append(step)

    if state.last_run_ref and state.step_budget_remaining > 0:
        previous_runs = [row["run_id"] for row in list_runs(root, task_id=task.task_id, limit=8) if row["run_id"] != state.last_run_id]
        lhs_ref = state.comparison_anchor_run_ref or (previous_runs[0] if previous_runs else None)
        if lhs_ref:
            state, step = step_environment(
                root,
                state,
                action_name="compare",
                lhs_run_ref=lhs_ref,
                rhs_run_ref=state.last_run_ref,
            )
            steps.append(step)

    solved = bool(state.metadata.get("last_eval_solved"))
    terminal_state = "success" if solved else "failure"
    if state.terminal_state == "budget_exhausted" and not solved:
        terminal_state = "budget_exhausted"
    if steps:
        last_step = steps[-1]
        steps[-1] = last_step.model_copy(update={"terminal": True, "terminal_state": terminal_state})
    artifact_refs = list(
        dict.fromkeys(
            artifact_ref
            for step in steps
            for artifact_ref in step.observation.artifact_refs
            if artifact_ref
        )
    )
    final_reward = sum(step.reward_total for step in steps)
    source_run_dir = resolve_run_dir(root, state.last_run_ref) if state.last_run_ref else root
    final_readiness: dict[str, object] = {}
    if state.last_run_ref:
        quality_payload = inspect_run(root, state.last_run_ref, section="quality")
        if isinstance(quality_payload, dict):
            final_readiness = dict(quality_payload.get("evidence_quality", {}))
    return TrajectoryEpisode(
        episode_id=state.episode_id,
        created_at=datetime.now(tz=UTC),
        policy_id=policy_id,
        task_id=task.task_id,
        task_verb=task.verb,
        operator_family=task.operator_family,
        source_run_id=state.last_run_id or "none",
        source_run_ref=str(source_run_dir),
        episode_kind="scripted_reference",
        steps=steps,
        final_reward=final_reward,
        terminal_state=terminal_state,
        artifact_refs=artifact_refs,
        metadata={
            **state.metadata,
            "workflow": selected_workflow,
            "action_names": [step.action.action_type for step in steps],
            "step_budget_total": state.step_budget_total,
            "step_budget_remaining": state.step_budget_remaining,
            "training_example_kind": final_readiness.get("training_example_kind", "unusable"),
            "evidence_score": final_readiness.get("overall_score", 0.0),
            "benchmark_ready": final_readiness.get("benchmark_reporting", {}).get("eligible", False)
            if isinstance(final_readiness.get("benchmark_reporting"), dict)
            else False,
            "sft_ready": final_readiness.get("sft_collection", {}).get("eligible", False)
            if isinstance(final_readiness.get("sft_collection"), dict)
            else False,
            "rl_trace_ready": final_readiness.get("rl_reward_trace", {}).get("eligible", False)
            if isinstance(final_readiness.get("rl_reward_trace"), dict)
            else False,
        },
    )
