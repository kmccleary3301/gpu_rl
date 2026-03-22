from __future__ import annotations

import difflib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from gpu_cockpit.contracts import CandidateDiffSummary, CandidateOperation, CandidateState, CandidateTransition, PatchRequest, RunSummary
from gpu_cockpit.contracts.patch import AppliedPatch, TransitionKind
from gpu_cockpit.engine.doctor import collect_doctor_report
from gpu_cockpit.engine.replay import write_replay_pack
from gpu_cockpit.engine.run_bundle import RunBundleWriter
from gpu_cockpit.engine.runner import build_run_id, build_run_spec, write_run_summary, write_task_artifacts
from gpu_cockpit.engine.task_registry import TaskRegistry


def _safe_target_path(root: Path, target_file: str) -> Path:
    target_path = (root / target_file).resolve()
    root_resolved = root.resolve()
    if not str(target_path).startswith(str(root_resolved)):
        raise RuntimeError(f"Patch target escapes workspace root: {target_file}")
    return target_path


def _count_changed_lines(before_text: str, after_text: str) -> int:
    before_lines = before_text.splitlines()
    after_lines = after_text.splitlines()
    return sum(
        1
        for line in difflib.ndiff(before_lines, after_lines)
        if line[:2] in {"+ ", "- "}
    )


def _build_unified_diff(target_file: str, before_text: str, after_text: str) -> str:
    before_lines = before_text.splitlines(keepends=True)
    after_lines = after_text.splitlines(keepends=True)
    return "".join(
        difflib.unified_diff(
            before_lines,
            after_lines,
            fromfile=f"a/{target_file}",
            tofile=f"b/{target_file}",
        )
    )


def _resolve_run_ref(root: Path, run_ref: str | None) -> Path | None:
    if not run_ref:
        return None
    candidate = Path(run_ref)
    if candidate.exists():
        return candidate.resolve()
    nested = (root / "runs" / run_ref).resolve()
    if nested.exists():
        return nested
    return None


def _lineage_depth(root: Path, parent_run_ref: str | None, parent_candidate_id: str | None) -> int:
    if parent_candidate_id is None:
        return 0
    parent_run_dir = _resolve_run_ref(root, parent_run_ref)
    if parent_run_dir is None:
        return 1
    state_path = parent_run_dir / "candidate" / "state.json"
    if not state_path.exists():
        return 1
    payload = json.loads(state_path.read_text(encoding="utf-8"))
    depth = payload.get("lineage_depth")
    return int(depth) + 1 if isinstance(depth, int) else 1


def _candidate_role_group(candidate_role: str | None) -> str | None:
    if candidate_role is None:
        return None
    if candidate_role == "baseline_candidate":
        return "baseline"
    if candidate_role in {"working_candidate", "patched_candidate", "synthesized_candidate"}:
        return "trial"
    if candidate_role == "branched_candidate":
        return "branch"
    if candidate_role == "reverted_candidate":
        return "reverted"
    if candidate_role == "promoted_candidate":
        return "promoted"
    if candidate_role == "comparison_anchor":
        return "anchor"
    return candidate_role


def _build_candidate_diff_summary(
    *,
    target_file: str,
    changed_line_count: int,
    patch_hash: str,
    intent: str,
    diff_ref: str,
) -> CandidateDiffSummary:
    return CandidateDiffSummary(
        changed_file_count=1,
        changed_line_count=changed_line_count,
        changed_files=[target_file],
        primary_target_file=target_file,
        diff_ref=diff_ref,
        patch_hash=patch_hash,
        summary=intent,
    )


def _initialize_candidate_run(
    root: Path,
    *,
    task_ref: str,
    prefix: str,
    parent_run_id: str | None,
    policy_pack: str,
    backend: str | None,
    vendor: str | None,
    executor: str,
) -> tuple[RunBundleWriter, Path, object, object]:
    registry = TaskRegistry(root)
    task = registry.get(task_ref)
    doctor_report = collect_doctor_report()
    run_spec = build_run_spec(
        task=task,
        backend=backend or (task.allowed_backends[0] if task.allowed_backends else "triton"),
        vendor=vendor or "nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    run_spec.run_id = build_run_id(prefix=prefix)
    run_spec.mode = "agent"
    run_spec.parent_run_id = parent_run_id
    run_spec.tags = list(dict.fromkeys([*run_spec.tags, "candidate-state", f"{prefix}-action"]))

    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)
    return writer, run_dir, task, doctor_report


def _write_candidate_state_bundle(
    *,
    root: Path,
    writer: RunBundleWriter,
    run_dir: Path,
    task: object,
    doctor_report: object,
    parent_run_ref: str | None,
    executor: str,
    policy_pack: str,
    target_backend: str,
    target_vendor: str,
    candidate_state: CandidateState,
    transition: CandidateTransition,
    operation: CandidateOperation,
    environment: dict[str, object],
    extra_artifacts: list[str] | None = None,
) -> None:
    writer.write_artifact(
        relative_path="candidate/state.json",
        kind="candidate_state",
        content=json.dumps(candidate_state.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "state"],
    )
    writer.write_artifact(
        relative_path="candidate/transition.json",
        kind="candidate_transition",
        content=json.dumps(transition.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "transition"],
    )
    writer.write_artifact(
        relative_path="candidate/operation.json",
        kind="candidate_operation",
        content=json.dumps(operation.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "operation"],
    )

    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "candidate/state.json",
        "candidate/transition.json",
        "candidate/operation.json",
        "summary.json",
        "summary.md",
    ]
    if getattr(doctor_report, "hardware_fingerprints", None):
        key_artifacts.insert(4, "meta/hardware_fingerprint.json")
    if extra_artifacts:
        key_artifacts[8:8] = extra_artifacts

    write_run_summary(
        writer,
        RunSummary(
            run_id=writer.run_spec.run_id if writer.run_spec is not None else run_dir.name,
            task_id=task.task_id,
            status="ok",
            trace_enabled=False,
            backend=target_backend,
            vendor=target_vendor,
            parent_run_id=writer.run_spec.parent_run_id if writer.run_spec is not None else None,
            candidate_id=candidate_state.candidate_id,
            parent_candidate_id=candidate_state.parent_candidate_id,
            patch_present=False,
            patch_kind=None,
            transition_kind=transition.transition_kind,
            candidate_role=candidate_state.candidate_role,
            candidate_status=candidate_state.status,
            candidate_origin_kind=candidate_state.origin_kind,
            candidate_operation_kind=operation.operation_kind,
            candidate_diff_ref=candidate_state.diff_summary.diff_ref if candidate_state.diff_summary is not None else None,
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=[],
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": target_backend,
            "target_vendor": target_vendor,
            **environment,
        },
        lineage={
            "candidate_id": candidate_state.candidate_id,
            "parent_candidate_id": candidate_state.parent_candidate_id,
            "source_candidate_id": candidate_state.source_candidate_id,
            "source_run_ref": parent_run_ref,
            "patch_ref": None,
            "diff_ref": candidate_state.diff_summary.diff_ref if candidate_state.diff_summary is not None else None,
            "transition_ref": "candidate/transition.json",
            "operation_ref": "candidate/operation.json",
            "candidate_role": candidate_state.candidate_role,
            "candidate_role_group": _candidate_role_group(candidate_state.candidate_role),
            "candidate_status": candidate_state.status,
            "candidate_origin_kind": candidate_state.origin_kind,
            "candidate_operation_kind": operation.operation_kind,
            "transition_kind": transition.transition_kind,
            "sibling_candidate_refs": [],
        },
    )
    writer.append_event(
        scope="candidate",
        kind="completed",
        payload={
            "candidate_id": candidate_state.candidate_id,
            "transition_kind": transition.transition_kind,
            "operation_kind": operation.operation_kind,
            "operation_id": operation.operation_id,
        },
    )


def apply_patch_candidate(
    root: Path,
    *,
    task_ref: str,
    target_file: str,
    replacement_text: str,
    intent: str,
    expected_effect: str | None = None,
    patch_kind: str = "bug_fix",
    transition_kind: TransitionKind = "patch_applied",
    parent_run_ref: str | None = None,
    parent_run_id: str | None = None,
    parent_candidate_id: str | None = None,
    policy_pack: str = "balanced",
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
) -> tuple[Path, AppliedPatch, CandidateState, CandidateTransition]:
    registry = TaskRegistry(root)
    task = registry.get(task_ref)
    doctor_report = collect_doctor_report()
    run_spec = build_run_spec(
        task=task,
        backend=backend or (task.allowed_backends[0] if task.allowed_backends else "triton"),
        vendor=vendor or "nvidia",
        executor=executor,
        policy_pack=policy_pack,
        tool_versions={tool.name: tool.version for tool in doctor_report.available_tools if tool.available and tool.version},
    )
    run_spec.run_id = build_run_id(prefix="patch")
    run_spec.mode = "agent"
    run_spec.parent_run_id = parent_run_id
    run_spec.tags = list(dict.fromkeys([*run_spec.tags, "patch-action", "candidate-state"]))

    writer = RunBundleWriter(root)
    run_dir = writer.initialize(run_spec)
    write_task_artifacts(writer, task, doctor_report)

    target_path = _safe_target_path(root, target_file)
    before_text = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    before_ref = None
    if target_path.exists():
        before_artifact = writer.write_artifact(
            relative_path="patches/before.txt",
            kind="patch_before",
            content=before_text,
            mime="text/plain",
            semantic_tags=["patch", "before"],
        )
        before_ref = before_artifact.path

    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(replacement_text, encoding="utf-8")

    after_artifact = writer.write_artifact(
        relative_path="patches/after.txt",
        kind="patch_after",
        content=replacement_text,
        mime="text/plain",
        semantic_tags=["patch", "after"],
    )
    diff_text = _build_unified_diff(target_file, before_text, replacement_text)
    diff_artifact = writer.write_artifact(
        relative_path="patches/unified_diff.patch",
        kind="patch_diff",
        content=diff_text,
        mime="text/x-diff",
        semantic_tags=["patch", "diff"],
    )
    patch_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    changed_line_count = _count_changed_lines(before_text, replacement_text)
    diff_summary = _build_candidate_diff_summary(
        target_file=target_file,
        changed_line_count=changed_line_count,
        patch_hash=patch_hash,
        intent=intent,
        diff_ref=diff_artifact.path,
    )
    request = PatchRequest(
        target_file=target_file,
        patch_kind=patch_kind,
        intent=intent,
        expected_effect=expected_effect,
        content_mode="replace",
        patch_text=replacement_text,
    )
    writer.write_artifact(
        relative_path="patches/request.json",
        kind="patch_request",
        content=json.dumps(request.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["patch", "request"],
    )
    applied_patch = AppliedPatch(
        patch_id=f"patch_{uuid4().hex[:12]}",
        target_file=target_file,
        patch_kind=patch_kind,
        intent=intent,
        expected_effect=expected_effect,
        content_mode="replace",
        accepted=True,
        patch_hash=patch_hash,
        changed_line_count=changed_line_count,
        before_ref=before_ref,
        after_ref=after_artifact.path,
        diff_ref=diff_artifact.path,
        metadata={
            "applied_at": datetime.now(tz=UTC).isoformat(),
            "workspace_target": str(target_path),
            "changed_files": [target_file],
        },
    )
    candidate_state = CandidateState(
        candidate_id=f"cand_{uuid4().hex[:12]}",
        task_id=task.task_id,
        workspace_root=str(root.resolve()),
        source_run_ref=parent_run_ref,
        source_run_id=parent_run_id,
        parent_candidate_id=parent_candidate_id,
        source_candidate_id=parent_candidate_id,
        parent_run_ref=parent_run_ref,
        patch_id=applied_patch.patch_id,
        patch_hash=applied_patch.patch_hash,
        patch_kind=applied_patch.patch_kind,
        changed_files=[target_file],
        candidate_role="patched_candidate",
        origin_kind="patch",
        status="patched",
        lineage_depth=0 if parent_candidate_id is None else 1,
        summary=intent,
        artifact_refs=[
            "patches/request.json",
            "patches/applied_patch.json",
            "patches/unified_diff.patch",
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        diff_summary=diff_summary,
        last_operation_kind="patch_apply",
        metadata={
            "intent": intent,
            "expected_effect": expected_effect,
        },
    )
    transition = CandidateTransition(
        transition_id=f"transition_{uuid4().hex[:12]}",
        transition_kind=transition_kind,
        task_id=task.task_id,
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        patch_id=applied_patch.patch_id,
        patch_hash=applied_patch.patch_hash,
        patch_kind=applied_patch.patch_kind,
        changed_files=[target_file],
        summary=intent,
        operation_kind="patch_apply",
        diff_summary=diff_summary,
        metadata={
            "expected_effect": expected_effect,
        },
    )
    operation = CandidateOperation(
        operation_id=f"op_{uuid4().hex[:12]}",
        operation_kind="patch_apply",
        task_id=task.task_id,
        actor_type="agent",
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        patch_id=applied_patch.patch_id,
        patch_hash=applied_patch.patch_hash,
        changed_files=[target_file],
        diff_summary=diff_summary,
        summary=intent,
        artifact_refs=[
            "patches/request.json",
            "patches/applied_patch.json",
            "patches/unified_diff.patch",
            "candidate/state.json",
            "candidate/transition.json",
        ],
        metadata={
            "expected_effect": expected_effect,
        },
    )
    writer.write_artifact(
        relative_path="patches/applied_patch.json",
        kind="applied_patch",
        content=json.dumps(applied_patch.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["patch", "applied"],
    )
    writer.write_artifact(
        relative_path="candidate/state.json",
        kind="candidate_state",
        content=json.dumps(candidate_state.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "state"],
    )
    writer.write_artifact(
        relative_path="candidate/transition.json",
        kind="candidate_transition",
        content=json.dumps(transition.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "transition"],
    )
    writer.write_artifact(
        relative_path="candidate/operation.json",
        kind="candidate_operation",
        content=json.dumps(operation.model_dump(mode="json"), indent=2) + "\n",
        mime="application/json",
        semantic_tags=["candidate", "operation"],
    )

    key_artifacts = [
        "manifest.json",
        "events.jsonl",
        "prompt/task_spec.json",
        "meta/task_spec_full.json",
        "meta/doctor_report.json",
        "patches/request.json",
        "patches/applied_patch.json",
        "patches/unified_diff.patch",
        "patches/after.txt",
        "candidate/state.json",
        "candidate/transition.json",
        "candidate/operation.json",
        "summary.json",
        "summary.md",
    ]
    if before_ref is not None:
        key_artifacts.insert(8, before_ref)
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
            parent_run_id=run_spec.parent_run_id,
            candidate_id=candidate_state.candidate_id,
            parent_candidate_id=candidate_state.parent_candidate_id,
            patch_present=True,
            patch_kind=applied_patch.patch_kind,
            transition_kind=transition.transition_kind,
            candidate_role=candidate_state.candidate_role,
            candidate_status=candidate_state.status,
            candidate_origin_kind=candidate_state.origin_kind,
            candidate_operation_kind=operation.operation_kind,
            candidate_diff_ref=diff_summary.diff_ref,
            key_artifacts=key_artifacts,
            warnings=[],
        ),
    )
    write_replay_pack(
        writer=writer,
        task=task,
        doctor_report=doctor_report,
        command=[],
        required_artifacts=key_artifacts,
        environment={
            "executor": executor,
            "policy_pack": policy_pack,
            "target_backend": run_spec.target_backend,
            "target_vendor": run_spec.target_vendor,
            "patch_target_file": target_file,
        },
        lineage={
            "candidate_id": candidate_state.candidate_id,
            "parent_candidate_id": candidate_state.parent_candidate_id,
            "source_candidate_id": candidate_state.source_candidate_id,
            "source_run_ref": parent_run_ref,
            "patch_ref": "patches/applied_patch.json",
            "diff_ref": diff_artifact.path,
            "transition_ref": "candidate/transition.json",
            "operation_ref": "candidate/operation.json",
            "candidate_role": candidate_state.candidate_role,
            "candidate_role_group": _candidate_role_group(candidate_state.candidate_role),
            "candidate_status": candidate_state.status,
            "candidate_origin_kind": candidate_state.origin_kind,
            "candidate_operation_kind": operation.operation_kind,
            "transition_kind": transition.transition_kind,
            "sibling_candidate_refs": [],
        },
    )
    writer.append_event(
        scope="candidate",
        kind="completed",
        payload={
            "candidate_id": candidate_state.candidate_id,
            "transition_kind": transition.transition_kind,
            "operation_kind": operation.operation_kind,
            "operation_id": operation.operation_id,
            "patch_id": applied_patch.patch_id,
        },
    )
    return run_dir, applied_patch, candidate_state, transition


def branch_candidate(
    root: Path,
    *,
    task_ref: str,
    intent: str,
    branch_label: str | None = None,
    expected_effect: str | None = None,
    parent_run_ref: str | None = None,
    parent_run_id: str | None = None,
    parent_candidate_id: str | None = None,
    policy_pack: str = "balanced",
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
) -> tuple[Path, CandidateState, CandidateTransition]:
    writer, run_dir, task, doctor_report = _initialize_candidate_run(
        root,
        task_ref=task_ref,
        prefix="branch",
        parent_run_id=parent_run_id,
        policy_pack=policy_pack,
        backend=backend,
        vendor=vendor,
        executor=executor,
    )
    lineage_depth = _lineage_depth(root, parent_run_ref, parent_candidate_id)
    candidate_state = CandidateState(
        candidate_id=f"cand_{uuid4().hex[:12]}",
        task_id=task.task_id,
        workspace_root=str(root.resolve()),
        source_run_ref=parent_run_ref,
        source_run_id=parent_run_id,
        parent_candidate_id=parent_candidate_id,
        source_candidate_id=parent_candidate_id,
        parent_run_ref=parent_run_ref,
        changed_files=[],
        candidate_role="branched_candidate",
        origin_kind="branch",
        status="draft",
        lineage_depth=lineage_depth,
        summary=intent,
        artifact_refs=[
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        last_operation_kind="branch",
        metadata={
            "intent": intent,
            "expected_effect": expected_effect,
            "branch_label": branch_label,
        },
    )
    transition = CandidateTransition(
        transition_id=f"transition_{uuid4().hex[:12]}",
        transition_kind="branched",
        operation_kind="branch",
        task_id=task.task_id,
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        status="ok",
        summary=intent,
        metadata={
            "expected_effect": expected_effect,
            "branch_label": branch_label,
        },
    )
    operation = CandidateOperation(
        operation_id=f"op_{uuid4().hex[:12]}",
        operation_kind="branch",
        task_id=task.task_id,
        actor_type="agent",
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        result_status="ok",
        summary=intent,
        artifact_refs=[
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        metadata={
            "expected_effect": expected_effect,
            "branch_label": branch_label,
        },
    )
    _write_candidate_state_bundle(
        root=root,
        writer=writer,
        run_dir=run_dir,
        task=task,
        doctor_report=doctor_report,
        parent_run_ref=parent_run_ref,
        executor=executor,
        policy_pack=policy_pack,
        target_backend=writer.run_spec.target_backend,
        target_vendor=writer.run_spec.target_vendor,
        candidate_state=candidate_state,
        transition=transition,
        operation=operation,
        environment={"branch_label": branch_label},
    )
    return run_dir, candidate_state, transition


def revert_candidate(
    root: Path,
    *,
    task_ref: str,
    target_file: str,
    replacement_text: str,
    intent: str,
    expected_effect: str | None = None,
    revert_target_candidate_id: str | None = None,
    parent_run_ref: str | None = None,
    parent_run_id: str | None = None,
    parent_candidate_id: str | None = None,
    policy_pack: str = "balanced",
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
) -> tuple[Path, CandidateState, CandidateTransition]:
    writer, run_dir, task, doctor_report = _initialize_candidate_run(
        root,
        task_ref=task_ref,
        prefix="revert",
        parent_run_id=parent_run_id,
        policy_pack=policy_pack,
        backend=backend,
        vendor=vendor,
        executor=executor,
    )
    target_path = _safe_target_path(root, target_file)
    before_text = target_path.read_text(encoding="utf-8") if target_path.exists() else ""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(replacement_text, encoding="utf-8")
    before_artifact = writer.write_artifact(
        relative_path="candidate/before.txt",
        kind="candidate_before",
        content=before_text,
        mime="text/plain",
        semantic_tags=["candidate", "before"],
    )
    after_artifact = writer.write_artifact(
        relative_path="candidate/after.txt",
        kind="candidate_after",
        content=replacement_text,
        mime="text/plain",
        semantic_tags=["candidate", "after"],
    )
    diff_text = _build_unified_diff(target_file, before_text, replacement_text)
    diff_artifact = writer.write_artifact(
        relative_path="candidate/unified_diff.patch",
        kind="candidate_diff",
        content=diff_text,
        mime="text/x-diff",
        semantic_tags=["candidate", "diff"],
    )
    diff_hash = hashlib.sha256(diff_text.encode("utf-8")).hexdigest()
    changed_line_count = _count_changed_lines(before_text, replacement_text)
    diff_summary = _build_candidate_diff_summary(
        target_file=target_file,
        changed_line_count=changed_line_count,
        patch_hash=diff_hash,
        intent=intent,
        diff_ref=diff_artifact.path,
    )
    lineage_depth = _lineage_depth(root, parent_run_ref, parent_candidate_id)
    candidate_state = CandidateState(
        candidate_id=f"cand_{uuid4().hex[:12]}",
        task_id=task.task_id,
        workspace_root=str(root.resolve()),
        source_run_ref=parent_run_ref,
        source_run_id=parent_run_id,
        parent_candidate_id=parent_candidate_id,
        source_candidate_id=revert_target_candidate_id or parent_candidate_id,
        parent_run_ref=parent_run_ref,
        changed_files=[target_file],
        candidate_role="reverted_candidate",
        origin_kind="revert",
        status="reverted",
        lineage_depth=lineage_depth,
        summary=intent,
        artifact_refs=[
            "candidate/before.txt",
            "candidate/after.txt",
            "candidate/unified_diff.patch",
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        diff_summary=diff_summary,
        last_operation_kind="revert",
        metadata={
            "intent": intent,
            "expected_effect": expected_effect,
            "revert_target_candidate_id": revert_target_candidate_id,
        },
    )
    transition = CandidateTransition(
        transition_id=f"transition_{uuid4().hex[:12]}",
        transition_kind="reverted",
        operation_kind="revert",
        task_id=task.task_id,
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        changed_files=[target_file],
        status="ok",
        summary=intent,
        diff_summary=diff_summary,
        metadata={
            "expected_effect": expected_effect,
            "revert_target_candidate_id": revert_target_candidate_id,
        },
    )
    operation = CandidateOperation(
        operation_id=f"op_{uuid4().hex[:12]}",
        operation_kind="revert",
        task_id=task.task_id,
        actor_type="agent",
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        changed_files=[target_file],
        diff_summary=diff_summary,
        result_status="ok",
        summary=intent,
        artifact_refs=[
            "candidate/before.txt",
            "candidate/after.txt",
            "candidate/unified_diff.patch",
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        metadata={
            "expected_effect": expected_effect,
            "revert_target_candidate_id": revert_target_candidate_id,
        },
    )
    _write_candidate_state_bundle(
        root=root,
        writer=writer,
        run_dir=run_dir,
        task=task,
        doctor_report=doctor_report,
        parent_run_ref=parent_run_ref,
        executor=executor,
        policy_pack=policy_pack,
        target_backend=writer.run_spec.target_backend,
        target_vendor=writer.run_spec.target_vendor,
        candidate_state=candidate_state,
        transition=transition,
        operation=operation,
        environment={"revert_target_file": target_file},
        extra_artifacts=["candidate/before.txt", "candidate/after.txt", "candidate/unified_diff.patch"],
    )
    return run_dir, candidate_state, transition


def promote_candidate(
    root: Path,
    *,
    task_ref: str,
    intent: str,
    promotion_label: str | None = None,
    expected_effect: str | None = None,
    parent_run_ref: str | None = None,
    parent_run_id: str | None = None,
    parent_candidate_id: str | None = None,
    policy_pack: str = "balanced",
    backend: str | None = None,
    vendor: str | None = None,
    executor: str = "local_host",
) -> tuple[Path, CandidateState, CandidateTransition]:
    writer, run_dir, task, doctor_report = _initialize_candidate_run(
        root,
        task_ref=task_ref,
        prefix="promote",
        parent_run_id=parent_run_id,
        policy_pack=policy_pack,
        backend=backend,
        vendor=vendor,
        executor=executor,
    )
    lineage_depth = _lineage_depth(root, parent_run_ref, parent_candidate_id)
    candidate_state = CandidateState(
        candidate_id=f"cand_{uuid4().hex[:12]}",
        task_id=task.task_id,
        workspace_root=str(root.resolve()),
        source_run_ref=parent_run_ref,
        source_run_id=parent_run_id,
        parent_candidate_id=parent_candidate_id,
        source_candidate_id=parent_candidate_id,
        parent_run_ref=parent_run_ref,
        changed_files=[],
        candidate_role="promoted_candidate",
        origin_kind="promotion",
        status="promoted",
        lineage_depth=lineage_depth,
        summary=intent,
        artifact_refs=[
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        last_operation_kind="promote",
        promotion_label=promotion_label,
        metadata={
            "intent": intent,
            "expected_effect": expected_effect,
            "promotion_label": promotion_label,
        },
    )
    transition = CandidateTransition(
        transition_id=f"transition_{uuid4().hex[:12]}",
        transition_kind="promoted",
        operation_kind="promote",
        task_id=task.task_id,
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        status="ok",
        summary=intent,
        metadata={
            "expected_effect": expected_effect,
            "promotion_label": promotion_label,
        },
    )
    operation = CandidateOperation(
        operation_id=f"op_{uuid4().hex[:12]}",
        operation_kind="promote",
        task_id=task.task_id,
        actor_type="agent",
        input_candidate_id=parent_candidate_id,
        output_candidate_id=candidate_state.candidate_id,
        source_run_ref=parent_run_ref,
        target_run_ref=str(run_dir),
        result_status="ok",
        summary=intent,
        artifact_refs=[
            "candidate/state.json",
            "candidate/transition.json",
            "candidate/operation.json",
        ],
        metadata={
            "expected_effect": expected_effect,
            "promotion_label": promotion_label,
        },
    )
    _write_candidate_state_bundle(
        root=root,
        writer=writer,
        run_dir=run_dir,
        task=task,
        doctor_report=doctor_report,
        parent_run_ref=parent_run_ref,
        executor=executor,
        policy_pack=policy_pack,
        target_backend=writer.run_spec.target_backend,
        target_vendor=writer.run_spec.target_vendor,
        candidate_state=candidate_state,
        transition=transition,
        operation=operation,
        environment={"promotion_label": promotion_label},
    )
    return run_dir, candidate_state, transition
