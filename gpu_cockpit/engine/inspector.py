from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import RunComparison, RunSummary
from gpu_cockpit.engine.evidence import assess_run_evidence


def resolve_run_dir(root: Path, run_ref: str) -> Path:
    candidate = Path(run_ref)
    if candidate.exists():
        return candidate.resolve()

    run_dir = root / "runs" / run_ref
    if run_dir.exists():
        return run_dir.resolve()

    raise FileNotFoundError(f"Run not found: {run_ref}")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_optional_json(run_dir: Path, relative_path: str) -> dict[str, Any] | None:
    path = run_dir / relative_path
    if not path.exists():
        return None
    return load_json(path)


def _sha256_for_ref(run_dir: Path, artifact_ref: str | None) -> str | None:
    if artifact_ref is None:
        return None
    path = run_dir / artifact_ref
    if not path.exists() or not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_failed_scopes(run_dir: Path) -> list[str]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return []
    failed_scopes: list[str] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if event.get("kind") == "failed":
            failed_scopes.append(str(event.get("scope", "unknown")))
    return failed_scopes


def _required_artifact_projection(run_dir: Path) -> dict[str, list[str]]:
    task_spec = _load_optional_json(run_dir, "prompt/task_spec.json") or {}
    required_artifacts = [str(path) for path in task_spec.get("required_artifacts", [])]
    missing_required_artifacts = [path for path in required_artifacts if not (run_dir / path).exists()]
    return {
        "required_artifacts": required_artifacts,
        "missing_required_artifacts": missing_required_artifacts,
    }


def _failure_triage(run_dir: Path) -> dict[str, Any]:
    task_spec = _load_optional_json(run_dir, "prompt/task_spec.json") or {}
    eval_envelope = _load_optional_json(run_dir, "eval/eval_envelope.json") or {}
    correctness = _load_optional_json(run_dir, "correctness/correctness.json") or {}
    failure_class = "ready_positive"
    likely_artifacts: list[str] = []
    if correctness.get("failures"):
        failure_class = "correctness"
        likely_artifacts.extend(["correctness/correctness.json", "command/stdout.txt", "command/stderr.txt"])
    elif eval_envelope.get("determinism_gate") == "fail":
        failure_class = "determinism"
        likely_artifacts.extend(["correctness/determinism.json", "command/stdout.txt"])
    elif eval_envelope.get("anti_hack_gate") == "fail":
        failure_class = "policy_or_shortcut"
        likely_artifacts.extend(["eval/anti_hack_report.json", "command/stdout.txt", "command/stderr.txt"])
    elif eval_envelope.get("perf_gate") == "fail":
        failure_class = "performance"
        likely_artifacts.extend(["perf/benchmark.json", "eval/gate_summary.json"])
    elif (run_dir / "build" / "tri_view.json").exists():
        failure_class = "build_or_kernel"
        likely_artifacts.extend(["build/tri_view.json", "build/build_record.json"])
    if str(task_spec.get("verb")) == "debug":
        likely_artifacts.extend(["correctness/correctness.json", "build/tri_view.json"])
    return {
        "task_verb": task_spec.get("verb"),
        "failure_class": failure_class,
        "likely_artifacts": list(dict.fromkeys([path for path in likely_artifacts if (run_dir / path).exists()])),
    }


def _candidate_projection(run_dir: Path) -> dict[str, Any] | None:
    candidate_state = _load_optional_json(run_dir, "candidate/state.json")
    transition = _load_optional_json(run_dir, "candidate/transition.json")
    applied_patch = _load_optional_json(run_dir, "patches/applied_patch.json")
    if candidate_state is None and transition is None and applied_patch is None:
        return None
    return {
        "candidate_state": candidate_state,
        "transition": transition,
        "applied_patch": applied_patch,
        "patch_present": applied_patch is not None,
        "changed_file_count": len(applied_patch.get("metadata", {}).get("changed_files", []))
        if isinstance(applied_patch, dict) and isinstance(applied_patch.get("metadata"), dict)
        else len(candidate_state.get("changed_files", []))
        if isinstance(candidate_state, dict)
        else 0,
    }


def _recommended_next_actions(projection: dict[str, Any]) -> list[str]:
    failure_triage = projection.get("failure_triage")
    if isinstance(failure_triage, dict):
        failure_class = str(failure_triage.get("failure_class", ""))
        if failure_class == "correctness":
            return ["inspect_quality", "inspect_build", "patch_candidate", "eval"]
        if failure_class == "determinism":
            return ["inspect_quality", "replay", "eval"]
        if failure_class == "performance":
            return ["inspect_profile", "compare", "patch_candidate", "bench"]
        if failure_class == "build_or_kernel":
            return ["inspect_build", "patch_candidate", "build", "eval"]
    candidate_projection = projection.get("candidate_projection")
    if isinstance(candidate_projection, dict) and candidate_projection.get("patch_present"):
        return ["build", "eval", "compare", "replay"]
    return ["inspect_quality", "compare", "replay"]


def _training_trace_triage(projection: dict[str, Any]) -> dict[str, Any]:
    evidence_quality = projection.get("evidence_quality", {})
    failure_triage = projection.get("failure_triage", {})
    candidate_projection = projection.get("candidate_projection") or {}
    training_example_kind = evidence_quality.get("training_example_kind", "unusable") if isinstance(evidence_quality, dict) else "unusable"
    reasons: list[str] = []
    if isinstance(failure_triage, dict) and failure_triage.get("failure_class") not in {None, "ready_positive"}:
        reasons.append(f"failure_class:{failure_triage.get('failure_class')}")
    if isinstance(candidate_projection, dict) and candidate_projection.get("patch_present"):
        transition = candidate_projection.get("transition") or {}
        patch = candidate_projection.get("applied_patch") or {}
        if isinstance(transition, dict) and transition.get("transition_kind") is not None:
            reasons.append(f"transition:{transition.get('transition_kind')}")
        if isinstance(patch, dict) and patch.get("patch_kind") is not None:
            reasons.append(f"patch_kind:{patch.get('patch_kind')}")
    if isinstance(evidence_quality, dict):
        benchmark = evidence_quality.get("benchmark_reporting", {})
        sft = evidence_quality.get("sft_collection", {})
        rl = evidence_quality.get("rl_reward_trace", {})
        if isinstance(benchmark, dict) and benchmark.get("eligible"):
            reasons.append("benchmark_ready")
        if isinstance(sft, dict) and sft.get("eligible"):
            reasons.append("sft_ready")
        if isinstance(rl, dict) and rl.get("eligible"):
            reasons.append("rl_trace_ready")
    return {
        "training_example_kind": training_example_kind,
        "summary": {
            "positive_sft_example": "usable positive training trace",
            "positive_rl_trace": "usable RL reward trace",
            "negative_debug_example": "usable failed repair example",
            "negative_reformulate_example": "usable failed transformation example",
            "benchmark_only": "benchmark-only trace, excluded from default training",
        }.get(str(training_example_kind), "not suitable for default training"),
        "reasons": reasons,
    }


def _summarize_triview(tri_view: dict[str, Any] | None) -> dict[str, Any] | None:
    if not tri_view:
        return None
    raw_lines = tri_view.get("lines", [])
    lines = raw_lines if isinstance(raw_lines, list) else []
    source_lines = [int(entry["source_line"]) for entry in lines if isinstance(entry, dict) and isinstance(entry.get("source_line"), int)]
    ptx_lines = [int(entry["ptx_line"]) for entry in lines if isinstance(entry, dict) and isinstance(entry.get("ptx_line"), int)]
    sass_lines = [int(entry["sass_line"]) for entry in lines if isinstance(entry, dict) and isinstance(entry.get("sass_line"), int)]
    return {
        "backend": tri_view.get("backend"),
        "correlation_method": tri_view.get("correlation_method"),
        "source_path": tri_view.get("source_path"),
        "line_count": len(lines),
        "unique_source_lines": len(set(source_lines)),
        "mapped_ptx_lines": len(ptx_lines),
        "mapped_sass_lines": len(sass_lines),
        "source_line_span": [min(source_lines), max(source_lines)] if source_lines else None,
    }


def _build_projection(run_dir: Path) -> dict[str, Any] | None:
    build_record = _load_optional_json(run_dir, "build/build_record.json")
    tri_view = _load_optional_json(run_dir, "build/tri_view.json")
    if build_record is None and tri_view is None:
        return None
    build_projection: dict[str, Any] = {
        "record": build_record,
        "tri_view": _summarize_triview(tri_view),
        "source_map_summary": _load_optional_json(run_dir, "build/source_map_summary.json"),
        "artifact_hashes": {},
    }
    refs: dict[str, str | None] = {
        "source": tri_view.get("source_ref") if tri_view else None,
        "ttir": tri_view.get("ttir_ref") if tri_view else None,
        "ttgir": tri_view.get("ttgir_ref") if tri_view else None,
        "llir": tri_view.get("llir_ref") if tri_view else None,
        "ptx": (build_record or {}).get("ptx_ref") or (tri_view.get("ptx_ref") if tri_view else None),
        "sass": (build_record or {}).get("sass_ref") or (tri_view.get("sass_ref") if tri_view else None),
    }
    for key, artifact_ref in refs.items():
        digest = _sha256_for_ref(run_dir, artifact_ref)
        if digest is not None:
            build_projection["artifact_hashes"][key] = digest
    return build_projection


def project_run_bundle(run_dir: Path) -> dict[str, Any]:
    projection = _required_artifact_projection(run_dir)
    projection["failed_scopes"] = _load_failed_scopes(run_dir)
    projection["replay_validation"] = {
        "present": (run_dir / "replay" / "replay_pack.json").exists(),
    }
    projection["profile_summary"] = _load_optional_json(run_dir, "profiles/kernel/summary.json")
    projection["sanitizer_summary"] = (
        _load_optional_json(run_dir, "sanitize/memcheck_summary.json")
        or _load_optional_json(run_dir, "sanitize/racecheck_summary.json")
        or _load_optional_json(run_dir, "sanitize/initcheck_summary.json")
        or _load_optional_json(run_dir, "sanitize/synccheck_summary.json")
    )
    projection["bottleneck_card"] = _load_optional_json(run_dir, "bottlenecks/primary.json")
    projection["correctness_summary"] = _load_optional_json(run_dir, "correctness/correctness.json")
    projection["determinism_summary"] = _load_optional_json(run_dir, "correctness/determinism.json")
    projection["anti_hack_summary"] = _load_optional_json(run_dir, "eval/anti_hack_report.json")
    projection["eval_envelope"] = _load_optional_json(run_dir, "eval/eval_envelope.json")
    projection["gate_summary"] = _load_optional_json(run_dir, "eval/gate_summary.json")
    projection["replay_pack"] = _load_optional_json(run_dir, "replay/replay_pack.json")
    projection["build_record"] = _load_optional_json(run_dir, "build/build_record.json")
    projection["tri_view"] = _load_optional_json(run_dir, "build/tri_view.json")
    projection["build_projection"] = _build_projection(run_dir)
    projection["candidate_projection"] = _candidate_projection(run_dir)
    projection["evidence_quality"] = assess_run_evidence(run_dir).model_dump(mode="json")
    projection["failure_triage"] = _failure_triage(run_dir)
    projection["recommended_next_actions"] = _recommended_next_actions(projection)
    evidence_quality = projection["evidence_quality"]
    projection["training_readiness"] = {
        "benchmark_reporting": evidence_quality.get("benchmark_reporting"),
        "sft_collection": evidence_quality.get("sft_collection"),
        "rl_reward_trace": evidence_quality.get("rl_reward_trace"),
        "training_example_kind": evidence_quality.get("training_example_kind"),
        "missing_provenance_fields": evidence_quality.get("missing_provenance_fields", []),
    }
    projection["training_trace_triage"] = _training_trace_triage(projection)
    return projection


def select_inspection_section(payload: dict[str, Any], section: str) -> dict[str, Any]:
    if section == "full":
        return payload
    projection = payload.get("projection", {})
    if not isinstance(projection, dict):
        projection = {}
    if section == "summary":
        return {
            key: payload.get(key)
            for key in [
                "run_id",
                "task_id",
                "status",
                "trace_enabled",
                "backend",
                "vendor",
                "parent_run_id",
                "candidate_id",
                "parent_candidate_id",
                "patch_present",
                "patch_kind",
                "transition_kind",
                "candidate_role",
                "exit_code",
                "duration_ms",
                "warnings",
                "key_artifacts",
            ]
        }
    section_map = {
        "build": ["build_record", "tri_view", "build_projection"],
        "eval": ["correctness_summary", "determinism_summary", "anti_hack_summary", "eval_envelope", "gate_summary"],
        "profile": ["profile_summary", "sanitizer_summary", "bottleneck_card"],
        "replay": ["replay_validation", "replay_pack"],
        "quality": ["evidence_quality", "training_readiness", "training_trace_triage", "failure_triage", "recommended_next_actions"],
        "transition": ["candidate_projection", "recommended_next_actions"],
    }
    keys = section_map.get(section)
    if keys is None:
        raise ValueError(f"Unknown inspect section: {section}")
    return {key: projection.get(key) for key in keys}


def inspect_run(root: Path, run_ref: str, *, section: str = "full") -> dict[str, Any]:
    run_dir = resolve_run_dir(root, run_ref)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        payload = load_json(summary_path)
        payload["projection"] = project_run_bundle(run_dir)
        return select_inspection_section(payload, section)
    manifest_path = run_dir / "manifest.json"
    if manifest_path.exists():
        payload = load_json(manifest_path)
        payload["projection"] = project_run_bundle(run_dir)
        return select_inspection_section(payload, section)
    raise FileNotFoundError(f"No summary or manifest found in {run_dir}")


def load_run_summary(root: Path, run_ref: str) -> RunSummary:
    run_dir = resolve_run_dir(root, run_ref)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        return RunSummary.model_validate(load_json(summary_path))

    manifest = load_json(run_dir / "manifest.json")
    return RunSummary(
        run_id=manifest["run_id"],
        task_id=manifest["task_ref"],
        status="legacy",
        trace_enabled=False,
        backend=manifest["target_backend"],
        vendor=manifest["target_vendor"],
        key_artifacts=["manifest.json", "events.jsonl"],
    )


def compare_runs(root: Path, lhs_ref: str, rhs_ref: str) -> RunComparison:
    lhs_run_dir = resolve_run_dir(root, lhs_ref)
    rhs_run_dir = resolve_run_dir(root, rhs_ref)
    lhs = load_run_summary(root, lhs_ref)
    rhs = load_run_summary(root, rhs_ref)
    lhs_warnings = set(lhs.warnings)
    rhs_warnings = set(rhs.warnings)
    lhs_projection = project_run_bundle(lhs_run_dir)
    rhs_projection = project_run_bundle(rhs_run_dir)
    lhs_evidence = assess_run_evidence(lhs_run_dir)
    rhs_evidence = assess_run_evidence(rhs_run_dir)
    lhs_eval = _load_optional_json(lhs_run_dir, "eval/eval_envelope.json") or {}
    rhs_eval = _load_optional_json(rhs_run_dir, "eval/eval_envelope.json") or {}
    lhs_perf = _load_optional_json(lhs_run_dir, "perf/benchmark.json") or {}
    rhs_perf = _load_optional_json(rhs_run_dir, "perf/benchmark.json") or {}
    lhs_profile = _load_optional_json(lhs_run_dir, "profiles/kernel/summary.json") or {}
    rhs_profile = _load_optional_json(rhs_run_dir, "profiles/kernel/summary.json") or {}
    lhs_sanitize = (
        _load_optional_json(lhs_run_dir, "sanitize/memcheck_summary.json")
        or _load_optional_json(lhs_run_dir, "sanitize/racecheck_summary.json")
        or _load_optional_json(lhs_run_dir, "sanitize/initcheck_summary.json")
        or _load_optional_json(lhs_run_dir, "sanitize/synccheck_summary.json")
        or {}
    )
    rhs_sanitize = (
        _load_optional_json(rhs_run_dir, "sanitize/memcheck_summary.json")
        or _load_optional_json(rhs_run_dir, "sanitize/racecheck_summary.json")
        or _load_optional_json(rhs_run_dir, "sanitize/initcheck_summary.json")
        or _load_optional_json(rhs_run_dir, "sanitize/synccheck_summary.json")
        or {}
    )
    lhs_build_projection = _build_projection(lhs_run_dir) or {}
    rhs_build_projection = _build_projection(rhs_run_dir) or {}
    lhs_candidate_projection = _candidate_projection(lhs_run_dir) or {}
    rhs_candidate_projection = _candidate_projection(rhs_run_dir) or {}
    lhs_build_record = lhs_build_projection.get("record") or {}
    rhs_build_record = rhs_build_projection.get("record") or {}
    lhs_candidate_state = lhs_candidate_projection.get("candidate_state") or {}
    rhs_candidate_state = rhs_candidate_projection.get("candidate_state") or {}
    lhs_transition = lhs_candidate_projection.get("transition") or {}
    rhs_transition = rhs_candidate_projection.get("transition") or {}
    lhs_patch = lhs_candidate_projection.get("applied_patch") or {}
    rhs_patch = rhs_candidate_projection.get("applied_patch") or {}
    lhs_triview = lhs_build_projection.get("tri_view") or {}
    rhs_triview = rhs_build_projection.get("tri_view") or {}
    lhs_final_score = lhs_eval.get("final_score")
    rhs_final_score = rhs_eval.get("final_score")
    lhs_perf_p50 = lhs_perf.get("steady_state_ms_p50")
    rhs_perf_p50 = rhs_perf.get("steady_state_ms_p50")
    lhs_determinism = lhs_eval.get("determinism_gate")
    rhs_determinism = rhs_eval.get("determinism_gate")
    lhs_occupancy = lhs_profile.get("occupancy")
    rhs_occupancy = rhs_profile.get("occupancy")
    lhs_dram = lhs_profile.get("dram_throughput_pct_peak")
    rhs_dram = rhs_profile.get("dram_throughput_pct_peak")
    lhs_sm = lhs_profile.get("sm_throughput_pct_peak")
    rhs_sm = rhs_profile.get("sm_throughput_pct_peak")
    lhs_regs = lhs_profile.get("registers_per_thread")
    rhs_regs = rhs_profile.get("registers_per_thread")
    lhs_triview_line_count = lhs_triview.get("line_count")
    rhs_triview_line_count = rhs_triview.get("line_count")
    lhs_unique_source_lines = lhs_triview.get("unique_source_lines")
    rhs_unique_source_lines = rhs_triview.get("unique_source_lines")
    lhs_build_binary_hash = lhs_build_record.get("binary_hash")
    rhs_build_binary_hash = rhs_build_record.get("binary_hash")
    lhs_patch_hash = lhs_patch.get("patch_hash")
    rhs_patch_hash = rhs_patch.get("patch_hash")
    lhs_changed_file_count = len(lhs_candidate_state.get("changed_files", [])) if isinstance(lhs_candidate_state.get("changed_files"), list) else 0
    rhs_changed_file_count = len(rhs_candidate_state.get("changed_files", [])) if isinstance(rhs_candidate_state.get("changed_files"), list) else 0
    lhs_correctness_gate = lhs_eval.get("correctness_gate")
    rhs_correctness_gate = rhs_eval.get("correctness_gate")
    trainworthiness_change = None
    if lhs_evidence.training_example_kind != rhs_evidence.training_example_kind:
        trainworthiness_change = f"{lhs_evidence.training_example_kind}->{rhs_evidence.training_example_kind}"
    elif lhs_evidence.overall_score != rhs_evidence.overall_score:
        direction = "improved" if rhs_evidence.overall_score > lhs_evidence.overall_score else "regressed"
        trainworthiness_change = f"evidence_{direction}"
    return RunComparison(
        lhs_run_id=lhs.run_id,
        rhs_run_id=rhs.run_id,
        lhs_status=lhs.status,
        rhs_status=rhs.status,
        lhs_duration_ms=lhs.duration_ms,
        rhs_duration_ms=rhs.duration_ms,
        duration_delta_ms=(rhs.duration_ms - lhs.duration_ms) if lhs.duration_ms is not None and rhs.duration_ms is not None else None,
        lhs_exit_code=lhs.exit_code,
        rhs_exit_code=rhs.exit_code,
        warnings_added=sorted(rhs_warnings - lhs_warnings),
        warnings_removed=sorted(lhs_warnings - rhs_warnings),
        lhs_key_artifacts=lhs.key_artifacts,
        rhs_key_artifacts=rhs.key_artifacts,
        lhs_missing_required_artifacts=lhs_projection["missing_required_artifacts"],
        rhs_missing_required_artifacts=rhs_projection["missing_required_artifacts"],
        lhs_failed_scopes=lhs_projection["failed_scopes"],
        rhs_failed_scopes=rhs_projection["failed_scopes"],
        lhs_evidence_score=lhs_evidence.overall_score,
        rhs_evidence_score=rhs_evidence.overall_score,
        evidence_score_delta=rhs_evidence.overall_score - lhs_evidence.overall_score,
        lhs_benchmark_ready=lhs_evidence.benchmark_reporting.eligible,
        rhs_benchmark_ready=rhs_evidence.benchmark_reporting.eligible,
        lhs_sft_ready=lhs_evidence.sft_collection.eligible,
        rhs_sft_ready=rhs_evidence.sft_collection.eligible,
        lhs_rl_trace_ready=lhs_evidence.rl_reward_trace.eligible,
        rhs_rl_trace_ready=rhs_evidence.rl_reward_trace.eligible,
        lhs_training_example_kind=lhs_evidence.training_example_kind,
        rhs_training_example_kind=rhs_evidence.training_example_kind,
        trainworthiness_change=trainworthiness_change,
        correctness_recovered=True
        if lhs_correctness_gate == "fail" and rhs_correctness_gate == "pass"
        else False
        if lhs_correctness_gate == "pass" and rhs_correctness_gate == "fail"
        else None,
        perf_improved=True
        if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)) and rhs_perf_p50 < lhs_perf_p50
        else False
        if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)) and rhs_perf_p50 > lhs_perf_p50
        else None,
        lhs_final_score=lhs_final_score if isinstance(lhs_final_score, (int, float)) else None,
        rhs_final_score=rhs_final_score if isinstance(rhs_final_score, (int, float)) else None,
        final_score_delta=(rhs_final_score - lhs_final_score)
        if isinstance(lhs_final_score, (int, float)) and isinstance(rhs_final_score, (int, float))
        else None,
        lhs_perf_p50_ms=lhs_perf_p50 if isinstance(lhs_perf_p50, (int, float)) else None,
        rhs_perf_p50_ms=rhs_perf_p50 if isinstance(rhs_perf_p50, (int, float)) else None,
        perf_p50_delta_ms=(rhs_perf_p50 - lhs_perf_p50)
        if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float))
        else None,
        lhs_determinism_passed=True if lhs_determinism == "pass" else False if lhs_determinism == "fail" else None,
        rhs_determinism_passed=True if rhs_determinism == "pass" else False if rhs_determinism == "fail" else None,
        lhs_profile_classification=str(lhs_profile["classification"]) if lhs_profile.get("classification") is not None else None,
        rhs_profile_classification=str(rhs_profile["classification"]) if rhs_profile.get("classification") is not None else None,
        lhs_profile_kernel=str(lhs_profile["kernel_name"]) if lhs_profile.get("kernel_name") is not None else None,
        rhs_profile_kernel=str(rhs_profile["kernel_name"]) if rhs_profile.get("kernel_name") is not None else None,
        lhs_sanitizer_passed=bool(lhs_sanitize["passed"]) if lhs_sanitize.get("passed") is not None else None,
        rhs_sanitizer_passed=bool(rhs_sanitize["passed"]) if rhs_sanitize.get("passed") is not None else None,
        lhs_sanitizer_error_count=int(lhs_sanitize["error_count"]) if lhs_sanitize.get("error_count") is not None else None,
        rhs_sanitizer_error_count=int(rhs_sanitize["error_count"]) if rhs_sanitize.get("error_count") is not None else None,
        lhs_profile_occupancy=float(lhs_occupancy) if isinstance(lhs_occupancy, (int, float)) else None,
        rhs_profile_occupancy=float(rhs_occupancy) if isinstance(rhs_occupancy, (int, float)) else None,
        occupancy_delta=(float(rhs_occupancy) - float(lhs_occupancy))
        if isinstance(lhs_occupancy, (int, float)) and isinstance(rhs_occupancy, (int, float))
        else None,
        lhs_profile_dram_pct_peak=float(lhs_dram) if isinstance(lhs_dram, (int, float)) else None,
        rhs_profile_dram_pct_peak=float(rhs_dram) if isinstance(rhs_dram, (int, float)) else None,
        dram_pct_peak_delta=(float(rhs_dram) - float(lhs_dram))
        if isinstance(lhs_dram, (int, float)) and isinstance(rhs_dram, (int, float))
        else None,
        lhs_profile_sm_pct_peak=float(lhs_sm) if isinstance(lhs_sm, (int, float)) else None,
        rhs_profile_sm_pct_peak=float(rhs_sm) if isinstance(rhs_sm, (int, float)) else None,
        sm_pct_peak_delta=(float(rhs_sm) - float(lhs_sm))
        if isinstance(lhs_sm, (int, float)) and isinstance(rhs_sm, (int, float))
        else None,
        lhs_profile_registers_per_thread=int(lhs_regs) if isinstance(lhs_regs, (int, float)) else None,
        rhs_profile_registers_per_thread=int(rhs_regs) if isinstance(rhs_regs, (int, float)) else None,
        registers_per_thread_delta=(int(rhs_regs) - int(lhs_regs))
        if isinstance(lhs_regs, (int, float)) and isinstance(rhs_regs, (int, float))
        else None,
        lhs_build_status=str(lhs_build_record["status"]) if lhs_build_record.get("status") is not None else None,
        rhs_build_status=str(rhs_build_record["status"]) if rhs_build_record.get("status") is not None else None,
        lhs_build_compiler=str(lhs_build_record["compiler"]) if lhs_build_record.get("compiler") is not None else None,
        rhs_build_compiler=str(rhs_build_record["compiler"]) if rhs_build_record.get("compiler") is not None else None,
        lhs_build_binary_hash=str(lhs_build_binary_hash) if lhs_build_binary_hash is not None else None,
        rhs_build_binary_hash=str(rhs_build_binary_hash) if rhs_build_binary_hash is not None else None,
        build_binary_hash_changed=(lhs_build_binary_hash != rhs_build_binary_hash)
        if lhs_build_binary_hash is not None and rhs_build_binary_hash is not None
        else None,
        lhs_patch_present=bool(lhs_candidate_projection.get("patch_present")),
        rhs_patch_present=bool(rhs_candidate_projection.get("patch_present")),
        lhs_patch_kind=str(lhs_patch["patch_kind"]) if lhs_patch.get("patch_kind") is not None else None,
        rhs_patch_kind=str(rhs_patch["patch_kind"]) if rhs_patch.get("patch_kind") is not None else None,
        lhs_patch_hash=str(lhs_patch_hash) if lhs_patch_hash is not None else None,
        rhs_patch_hash=str(rhs_patch_hash) if rhs_patch_hash is not None else None,
        patch_hash_changed=(lhs_patch_hash != rhs_patch_hash)
        if lhs_patch_hash is not None and rhs_patch_hash is not None
        else None,
        lhs_changed_file_count=lhs_changed_file_count,
        rhs_changed_file_count=rhs_changed_file_count,
        changed_file_count_delta=rhs_changed_file_count - lhs_changed_file_count,
        lhs_transition_kind=str(lhs_transition["transition_kind"]) if lhs_transition.get("transition_kind") is not None else None,
        rhs_transition_kind=str(rhs_transition["transition_kind"]) if rhs_transition.get("transition_kind") is not None else None,
        lhs_candidate_id=str(lhs_candidate_state["candidate_id"]) if lhs_candidate_state.get("candidate_id") is not None else None,
        rhs_candidate_id=str(rhs_candidate_state["candidate_id"]) if rhs_candidate_state.get("candidate_id") is not None else None,
        lhs_triview_present=(lhs_run_dir / "build" / "tri_view.json").exists(),
        rhs_triview_present=(rhs_run_dir / "build" / "tri_view.json").exists(),
        lhs_triview_correlation_method=str(lhs_triview["correlation_method"])
        if lhs_triview.get("correlation_method") is not None
        else None,
        rhs_triview_correlation_method=str(rhs_triview["correlation_method"])
        if rhs_triview.get("correlation_method") is not None
        else None,
        lhs_triview_source_path=str(lhs_triview["source_path"]) if lhs_triview.get("source_path") is not None else None,
        rhs_triview_source_path=str(rhs_triview["source_path"]) if rhs_triview.get("source_path") is not None else None,
        lhs_triview_line_count=int(lhs_triview_line_count) if isinstance(lhs_triview_line_count, int) else None,
        rhs_triview_line_count=int(rhs_triview_line_count) if isinstance(rhs_triview_line_count, int) else None,
        triview_line_count_delta=(int(rhs_triview_line_count) - int(lhs_triview_line_count))
        if isinstance(lhs_triview_line_count, int) and isinstance(rhs_triview_line_count, int)
        else None,
        lhs_triview_unique_source_lines=int(lhs_unique_source_lines) if isinstance(lhs_unique_source_lines, int) else None,
        rhs_triview_unique_source_lines=int(rhs_unique_source_lines) if isinstance(rhs_unique_source_lines, int) else None,
        triview_unique_source_lines_delta=(int(rhs_unique_source_lines) - int(lhs_unique_source_lines))
        if isinstance(lhs_unique_source_lines, int) and isinstance(rhs_unique_source_lines, int)
        else None,
    )
