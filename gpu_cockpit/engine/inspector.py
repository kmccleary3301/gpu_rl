from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import RunComparison, RunSummary
from gpu_cockpit.engine.evidence import assess_run_evidence
from gpu_cockpit.engine.optimize_patch_registry import get_optimize_patch_spec


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


def _load_command_payload(run_dir: Path) -> dict[str, Any] | None:
    path = run_dir / "command" / "stdout.txt"
    if not path.exists():
        return None
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    for line in reversed(lines):
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


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
    failure_localization = _load_optional_json(run_dir, "correctness/failure_localization.json") or {}
    failure_class = "ready_positive"
    likely_artifacts: list[str] = []
    primary_signal = None
    if isinstance(failure_localization, dict):
        for key in ("hidden_tests", "visible_tests"):
            payload = failure_localization.get(key)
            if isinstance(payload, dict) and payload.get("code") is not None:
                primary_signal = str(payload.get("code"))
                likely_artifacts.append("correctness/failure_localization.json")
                break
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
        "primary_signal": primary_signal,
        "likely_artifacts": list(dict.fromkeys([path for path in likely_artifacts if (run_dir / path).exists()])),
    }


def _candidate_projection(run_dir: Path) -> dict[str, Any] | None:
    candidate_state = _load_optional_json(run_dir, "candidate/state.json")
    transition = _load_optional_json(run_dir, "candidate/transition.json")
    operation = _load_optional_json(run_dir, "candidate/operation.json")
    applied_patch = _load_optional_json(run_dir, "patches/applied_patch.json")
    if candidate_state is None and transition is None and operation is None and applied_patch is None:
        return None
    return {
        "candidate_state": candidate_state,
        "transition": transition,
        "operation": operation,
        "applied_patch": applied_patch,
        "patch_present": applied_patch is not None,
        "candidate_status": candidate_state.get("status") if isinstance(candidate_state, dict) else None,
        "candidate_origin_kind": candidate_state.get("origin_kind") if isinstance(candidate_state, dict) else None,
        "candidate_operation_kind": operation.get("operation_kind")
        if isinstance(operation, dict)
        else candidate_state.get("last_operation_kind")
        if isinstance(candidate_state, dict)
        else None,
        "changed_file_count": len(applied_patch.get("metadata", {}).get("changed_files", []))
        if isinstance(applied_patch, dict) and isinstance(applied_patch.get("metadata"), dict)
        else len(candidate_state.get("changed_files", []))
        if isinstance(candidate_state, dict)
        else 0,
    }


def _public_benchmark_projection(command_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(command_payload, dict):
        return None
    benchmark_source = command_payload.get("benchmark_source")
    benchmark_case_id = command_payload.get("benchmark_case_id")
    benchmark_case_version = command_payload.get("benchmark_case_version")
    case_config_path = command_payload.get("case_config_path")
    problem_path = command_payload.get("problem_path")
    optimization_summary = command_payload.get("optimization_summary")
    if all(
        value is None
        for value in [
            benchmark_source,
            benchmark_case_id,
            benchmark_case_version,
            case_config_path,
            problem_path,
            optimization_summary,
        ]
    ):
        return None
    result: dict[str, Any] = {
        "benchmark_source": benchmark_source,
        "benchmark_case_id": benchmark_case_id,
        "benchmark_case_version": benchmark_case_version,
        "case_config_path": case_config_path,
        "problem_path": problem_path,
    }
    if isinstance(optimization_summary, dict):
        result["optimization_summary"] = optimization_summary
    return result


def _hardware_fingerprint_projection(run_dir: Path) -> dict[str, Any] | None:
    fingerprint = _load_optional_json(run_dir, "meta/hardware_fingerprint.json")
    if not isinstance(fingerprint, dict):
        return None
    return {
        "vendor": fingerprint.get("vendor"),
        "backend": fingerprint.get("backend"),
        "gpu_name": fingerprint.get("gpu_name") or fingerprint.get("device_name"),
        "compute_capability": fingerprint.get("compute_capability"),
        "driver_version": fingerprint.get("driver_version"),
    }


def _perf_localization_packet(
    run_dir: Path,
    projection: dict[str, Any],
) -> dict[str, Any]:
    perf = _load_optional_json(run_dir, "perf/benchmark.json") or {}
    profile_summary = projection.get("profile_summary") or {}
    bottleneck_card = projection.get("bottleneck_card") or {}
    hardware_fingerprint = _hardware_fingerprint_projection(run_dir) or {}
    baseline_runtime = perf.get("baseline_steady_state_ms_p50")
    candidate_runtime = perf.get("steady_state_ms_p50")
    delta = None
    if isinstance(baseline_runtime, (int, float)) and isinstance(candidate_runtime, (int, float)):
        delta = float(candidate_runtime) - float(baseline_runtime)
    return {
        "warmup_policy": {
            "warmups": perf.get("warmups"),
            "repeats": perf.get("repeats"),
            "split_compile_from_run": perf.get("split_compile_from_run"),
        },
        "timing_method": perf.get("timing_method") or perf.get("timer") or "wall_clock_benchmark",
        "compile_vs_runtime_split": {
            "cold_compile_ms": perf.get("cold_compile_ms"),
            "baseline_cold_compile_ms": perf.get("baseline_cold_compile_ms"),
            "steady_state_ms_p50": perf.get("steady_state_ms_p50"),
        },
        "baseline_runtime": baseline_runtime,
        "candidate_runtime": candidate_runtime,
        "p50": perf.get("steady_state_ms_p50"),
        "p95": perf.get("steady_state_ms_p95"),
        "delta": delta,
        "benchmark_scope": perf.get("benchmark_scope"),
        "benchmark_protocol_version": perf.get("benchmark_protocol_version"),
        "candidate_command_sha256": perf.get("candidate_command_sha256"),
        "baseline_command_sha256": perf.get("baseline_command_sha256"),
        "likely_bottleneck_class": bottleneck_card.get("primary_bottleneck") or profile_summary.get("classification"),
        "hotspot_summary": bottleneck_card.get("summary") or bottleneck_card.get("primary_bottleneck"),
        "launch_shape_summary": profile_summary.get("launch_shape_summary"),
        "benchmark_hardware_fingerprint": perf.get("hardware_fingerprint") or hardware_fingerprint,
    }


def _benchmark_provenance_packet(
    lhs: RunSummary,
    rhs: RunSummary,
    lhs_public_benchmark: dict[str, Any] | None,
    rhs_public_benchmark: dict[str, Any] | None,
) -> dict[str, Any]:
    public = rhs_public_benchmark if isinstance(rhs_public_benchmark, dict) else lhs_public_benchmark if isinstance(lhs_public_benchmark, dict) else {}
    return {
        "benchmark_source": public.get("benchmark_source"),
        "benchmark_case_id": public.get("benchmark_case_id"),
        "benchmark_case_version": public.get("benchmark_case_version"),
        "case_config_path": public.get("case_config_path"),
        "problem_path": public.get("problem_path"),
        "lhs_backend": lhs.backend,
        "rhs_backend": rhs.backend,
        "lhs_vendor": lhs.vendor,
        "rhs_vendor": rhs.vendor,
    }


def _optimization_summary_from_payload(command_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(command_payload, dict):
        return None
    optimization_summary = command_payload.get("optimization_summary")
    if isinstance(optimization_summary, dict):
        return optimization_summary
    reformulation_summary = command_payload.get("reformulation_summary")
    if not isinstance(reformulation_summary, dict):
        return None
    return {
        "strategy_change": reformulation_summary.get("strategy_change"),
        "candidate_ref": reformulation_summary.get("optimized_ref"),
        "baseline_ref": reformulation_summary.get("baseline_ref"),
        "supersedes_candidate_ref": reformulation_summary.get("supersedes_candidate_ref"),
    }


def _infer_public_benchmark_projection(
    task_id: str,
    *,
    patch_present: bool,
    patch_kind: str | None,
) -> dict[str, Any] | None:
    spec = get_optimize_patch_spec(task_id)
    if spec is None:
        return None
    public = spec.get("public_benchmark")
    if not isinstance(public, dict):
        return None
    is_positive = patch_present and patch_kind != "no_op"
    candidate_ref = str(spec["positive_patch_source_file"] if is_positive else spec["negative_patch_source_file"])
    strategy_change = public["positive_strategy_change"] if is_positive else public["negative_strategy_change"]
    return {
        "benchmark_source": public.get("benchmark_source"),
        "benchmark_case_id": public.get("benchmark_case_id"),
        "benchmark_case_version": public.get("benchmark_case_version"),
        "case_config_path": public.get("case_config_ref"),
        "problem_path": None,
        "optimization_summary": {
            "strategy_change": strategy_change,
            "candidate_ref": candidate_ref,
            "baseline_ref": public.get("baseline_ref"),
            "case_config_ref": public.get("case_config_ref"),
        },
    }


def _parent_candidate_projection(root: Path, parent_run_id: str | None) -> dict[str, Any] | None:
    if not parent_run_id:
        return None
    try:
        parent_run_dir = resolve_run_dir(root, parent_run_id)
    except FileNotFoundError:
        return None
    return _candidate_projection(parent_run_dir)


def _recommended_next_actions(projection: dict[str, Any]) -> list[str]:
    failure_localization = projection.get("failure_localization") or {}
    if isinstance(failure_localization, dict):
        hidden_details = failure_localization.get("hidden_tests")
        if isinstance(hidden_details, dict):
            likely_next_actions = hidden_details.get("likely_next_actions")
            if isinstance(likely_next_actions, list) and likely_next_actions:
                return [str(action) for action in likely_next_actions]
            code = str(hidden_details.get("code", ""))
            if code in {"missing_optimization_summary", "missing_reformulation_summary"}:
                return ["patch_candidate", "build", "eval", "compare"]
            if code in {"hidden_attention_score_mismatch", "visible_attention_score_mismatch"}:
                return ["inspect_quality", "compare", "patch_candidate", "eval"]
        visible_details = failure_localization.get("visible_tests")
        if isinstance(visible_details, dict):
            likely_next_actions = visible_details.get("likely_next_actions")
            if isinstance(likely_next_actions, list) and likely_next_actions:
                return [str(action) for action in likely_next_actions]
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


def _profile_triage(projection: dict[str, Any]) -> dict[str, Any]:
    profile_summary = projection.get("profile_summary") or {}
    bottleneck_card = projection.get("bottleneck_card") or {}
    sanitizer_summary = projection.get("sanitizer_summary") or {}
    summary_lines: list[str] = []
    classification = profile_summary.get("classification")
    kernel_name = profile_summary.get("kernel_name")
    if classification is not None:
        subject = f"`{kernel_name}`" if kernel_name else "primary kernel"
        summary_lines.append(f"{subject} is currently classified as `{classification}`.")
    primary_bottleneck = bottleneck_card.get("primary_bottleneck") if isinstance(bottleneck_card, dict) else None
    if primary_bottleneck is not None and primary_bottleneck != classification:
        summary_lines.append(f"Bottleneck card highlights `{primary_bottleneck}` as the dominant constraint.")
    dominant_sanitizer_family = sanitizer_summary.get("dominant_failure_family") if isinstance(sanitizer_summary, dict) else None
    if dominant_sanitizer_family is not None:
        summary_lines.append(f"Sanitizer findings are dominated by `{dominant_sanitizer_family}` issues.")
    occupancy = profile_summary.get("occupancy")
    dram_pct_peak = profile_summary.get("dram_throughput_pct_peak")
    sm_pct_peak = profile_summary.get("sm_throughput_pct_peak")
    if isinstance(occupancy, (int, float)):
        summary_lines.append(f"Occupancy is `{float(occupancy):.1f}` percent of peak active warps.")
    if isinstance(dram_pct_peak, (int, float)) and isinstance(sm_pct_peak, (int, float)):
        if float(dram_pct_peak) > float(sm_pct_peak):
            summary_lines.append("Memory throughput is higher than SM throughput, which supports a memory-oriented diagnosis.")
        elif float(sm_pct_peak) > float(dram_pct_peak):
            summary_lines.append("SM throughput is higher than DRAM throughput, which supports a compute-oriented diagnosis.")
    return {
        "classification": classification,
        "primary_bottleneck": primary_bottleneck,
        "dominant_sanitizer_family": dominant_sanitizer_family,
        "summary_lines": summary_lines,
    }


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


def _lineage_relationship(
    lhs_candidate_id: str | None,
    rhs_candidate_id: str | None,
    lhs_parent_candidate_id: str | None,
    rhs_parent_candidate_id: str | None,
) -> tuple[str | None, bool | None]:
    if lhs_candidate_id is None and rhs_candidate_id is None:
        return None, None
    if lhs_candidate_id is not None and rhs_candidate_id is not None and lhs_candidate_id == rhs_candidate_id:
        return "same_candidate", True
    if lhs_candidate_id is not None and rhs_parent_candidate_id is not None and lhs_candidate_id == rhs_parent_candidate_id:
        return "lhs_parent_of_rhs", True
    if rhs_candidate_id is not None and lhs_parent_candidate_id is not None and rhs_candidate_id == lhs_parent_candidate_id:
        return "rhs_parent_of_lhs", True
    if (
        lhs_parent_candidate_id is not None
        and rhs_parent_candidate_id is not None
        and lhs_parent_candidate_id == rhs_parent_candidate_id
    ):
        return "same_parent", False
    return "unrelated", False


def _candidate_role_group(candidate_role: Any) -> str | None:
    role = str(candidate_role) if candidate_role is not None else None
    if role is None:
        return None
    if role == "baseline_candidate":
        return "baseline"
    if role in {"working_candidate", "patched_candidate", "synthesized_candidate"}:
        return "trial"
    if role == "branched_candidate":
        return "branch"
    if role == "reverted_candidate":
        return "reverted"
    if role == "promoted_candidate":
        return "promoted"
    if role == "comparison_anchor":
        return "anchor"
    return role


def _hash_delta(lhs_hash: str | None, rhs_hash: str | None) -> bool | None:
    if lhs_hash is None or rhs_hash is None:
        return None
    return lhs_hash != rhs_hash


def _compare_correctness_change(lhs_gate: Any, rhs_gate: Any) -> str:
    lhs = str(lhs_gate) if lhs_gate is not None else "unknown"
    rhs = str(rhs_gate) if rhs_gate is not None else "unknown"
    if lhs == "fail" and rhs == "pass":
        return "recovered"
    if lhs == "pass" and rhs == "fail":
        return "regressed"
    if lhs == "pass" and rhs == "pass":
        return "preserved_pass"
    if lhs == "fail" and rhs == "fail":
        return "preserved_fail"
    return "unknown"


def _optimize_delta_summary(
    *,
    lhs_correctness_gate: Any,
    rhs_correctness_gate: Any,
    lhs_perf_p50: Any,
    rhs_perf_p50: Any,
    lhs_final_score: Any,
    rhs_final_score: Any,
    perf_improved: bool | None,
    lineage_relationship: str | None,
    lhs_patch_kind: Any,
    rhs_patch_kind: Any,
    rhs_patch_present: bool,
    lhs_candidate_role: Any,
    rhs_candidate_role: Any,
    lhs_public_benchmark: dict[str, Any] | None,
    rhs_public_benchmark: dict[str, Any] | None,
    lhs_optimization_summary: dict[str, Any] | None,
    rhs_optimization_summary: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[str], list[str]]:
    correctness_change = _compare_correctness_change(lhs_correctness_gate, rhs_correctness_gate)
    perf_change = "not_comparable"
    perf_delta_ms = None
    if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)):
        perf_delta_ms = float(rhs_perf_p50) - float(lhs_perf_p50)
        if perf_delta_ms < 0:
            perf_change = "improved"
        elif perf_delta_ms > 0:
            perf_change = "regressed"
        else:
            perf_change = "unchanged"
    final_score_delta = None
    if isinstance(lhs_final_score, (int, float)) and isinstance(rhs_final_score, (int, float)):
        final_score_delta = float(rhs_final_score) - float(lhs_final_score)
    focus = "baseline_to_candidate" if rhs_patch_present and not lhs_patch_kind else "candidate_delta"
    patch_change = None
    if rhs_patch_kind is not None and rhs_patch_kind != lhs_patch_kind:
        patch_change = f"{lhs_patch_kind or 'none'}->{rhs_patch_kind}"
    lhs_strategy_change = lhs_optimization_summary.get("strategy_change") if isinstance(lhs_optimization_summary, dict) else None
    rhs_strategy_change = rhs_optimization_summary.get("strategy_change") if isinstance(rhs_optimization_summary, dict) else None
    lhs_candidate_ref = lhs_optimization_summary.get("candidate_ref") if isinstance(lhs_optimization_summary, dict) else None
    rhs_candidate_ref = rhs_optimization_summary.get("candidate_ref") if isinstance(rhs_optimization_summary, dict) else None
    lhs_case_ref = lhs_optimization_summary.get("case_config_ref") if isinstance(lhs_optimization_summary, dict) else None
    rhs_case_ref = rhs_optimization_summary.get("case_config_ref") if isinstance(rhs_optimization_summary, dict) else None
    benchmark_case_id = None
    benchmark_case_changed = None
    if isinstance(lhs_public_benchmark, dict) or isinstance(rhs_public_benchmark, dict):
        lhs_case_id = lhs_public_benchmark.get("benchmark_case_id") if isinstance(lhs_public_benchmark, dict) else None
        rhs_case_id = rhs_public_benchmark.get("benchmark_case_id") if isinstance(rhs_public_benchmark, dict) else None
        benchmark_case_id = rhs_case_id or lhs_case_id
        if lhs_case_id is not None and rhs_case_id is not None:
            benchmark_case_changed = lhs_case_id != rhs_case_id
    lines: list[str] = []
    if correctness_change == "recovered":
        lines.append("Correctness recovered relative to the baseline/reference run.")
    elif correctness_change == "regressed":
        lines.append("Correctness regressed relative to the baseline/reference run.")
    elif correctness_change == "preserved_pass":
        lines.append("Correctness stayed green across both runs.")
    elif correctness_change == "preserved_fail":
        lines.append("Both runs still fail correctness; inspect the failure delta before another patch.")
    if perf_change == "improved" and perf_delta_ms is not None:
        lines.append(f"Candidate latency improved by {abs(perf_delta_ms):.3f} ms at p50.")
    elif perf_change == "regressed" and perf_delta_ms is not None:
        lines.append(f"Candidate latency regressed by {perf_delta_ms:.3f} ms at p50.")
    if patch_change is not None:
        lines.append(f"Patch lineage changed via `{patch_change}`.")
    elif lineage_relationship is not None:
        lines.append(f"Candidate lineage relation is `{lineage_relationship}`.")
    if benchmark_case_id is not None:
        if benchmark_case_changed:
            lines.append(f"Benchmark case changed from `{lhs_case_id}` to `{rhs_case_id}`.")
        else:
            lines.append(f"Compare is anchored to public benchmark case `{benchmark_case_id}`.")
    if rhs_strategy_change is not None:
        if lhs_strategy_change is not None and lhs_strategy_change != rhs_strategy_change:
            lines.append(f"Optimization strategy changed from `{lhs_strategy_change}` to `{rhs_strategy_change}`.")
        elif lhs_strategy_change is None:
            lines.append(f"Optimization strategy is `{rhs_strategy_change}`.")
    if rhs_candidate_ref is not None:
        if lhs_candidate_ref is not None and lhs_candidate_ref != rhs_candidate_ref:
            lines.append(f"Candidate reference changed from `{lhs_candidate_ref}` to `{rhs_candidate_ref}`.")
        elif lhs_candidate_ref is None:
            lines.append(f"Candidate reference is `{rhs_candidate_ref}`.")
    recommended_next_actions: list[str]
    if correctness_change in {"regressed", "preserved_fail"}:
        recommended_next_actions = ["inspect_quality", "patch_candidate", "eval"]
    elif perf_change == "regressed":
        recommended_next_actions = ["inspect_quality", "patch_candidate", "bench"]
    else:
        recommended_next_actions = ["inspect_quality", "replay", "eval"]
    lineage_scopes = {
        "baseline_delta": {
            "focus": "baseline_vs_candidate" if focus == "baseline_to_candidate" else "candidate_vs_candidate",
            "lineage_relationship": lineage_relationship,
        },
        "parent_delta": {
            "available": lineage_relationship in {"lhs_parent_of_rhs", "rhs_parent_of_lhs"},
            "lineage_relationship": lineage_relationship,
        },
        "sibling_delta": {
            "available": lineage_relationship == "same_parent",
            "lineage_relationship": lineage_relationship,
        },
    }
    return (
        {
            "focus": focus,
            "benchmark_case_id": benchmark_case_id,
            "benchmark_case_changed": benchmark_case_changed,
            "benchmark_source": rhs_public_benchmark.get("benchmark_source")
            if isinstance(rhs_public_benchmark, dict)
            else lhs_public_benchmark.get("benchmark_source")
            if isinstance(lhs_public_benchmark, dict)
            else None,
            "case_config_path": rhs_public_benchmark.get("case_config_path")
            if isinstance(rhs_public_benchmark, dict)
            else lhs_public_benchmark.get("case_config_path")
            if isinstance(lhs_public_benchmark, dict)
            else None,
            "optimization_strategy_change": rhs_strategy_change,
            "optimization_strategy_delta": f"{lhs_strategy_change}->{rhs_strategy_change}"
            if lhs_strategy_change is not None and rhs_strategy_change is not None and lhs_strategy_change != rhs_strategy_change
            else None,
            "candidate_ref": rhs_candidate_ref,
            "candidate_ref_delta": f"{lhs_candidate_ref}->{rhs_candidate_ref}"
            if lhs_candidate_ref is not None and rhs_candidate_ref is not None and lhs_candidate_ref != rhs_candidate_ref
            else None,
            "correctness_change": correctness_change,
            "perf_change": perf_change,
            "perf_p50_delta_ms": perf_delta_ms,
            "final_score_delta": final_score_delta,
            "lineage_relationship": lineage_relationship,
            "patch_change": patch_change,
            "perf_improved": perf_improved,
            "problem_path": rhs_public_benchmark.get("problem_path")
            if isinstance(rhs_public_benchmark, dict)
            else lhs_public_benchmark.get("problem_path")
            if isinstance(lhs_public_benchmark, dict)
            else None,
            "baseline_ref": rhs_optimization_summary.get("baseline_ref") if isinstance(rhs_optimization_summary, dict) else None,
            "case_config_ref": rhs_case_ref,
            "lhs_candidate_role": str(lhs_candidate_role) if lhs_candidate_role is not None else None,
            "rhs_candidate_role": str(rhs_candidate_role) if rhs_candidate_role is not None else None,
            "lhs_candidate_role_group": _candidate_role_group(lhs_candidate_role),
            "rhs_candidate_role_group": _candidate_role_group(rhs_candidate_role),
            "lineage_scopes": lineage_scopes,
        },
        lines,
        recommended_next_actions,
    )


def _candidate_delta_brief(
    *,
    lhs_candidate_id: str | None,
    rhs_candidate_id: str | None,
    lhs_parent_candidate_id: str | None,
    rhs_parent_candidate_id: str | None,
    lhs_transition_kind: Any,
    rhs_transition_kind: Any,
    lhs_operation_kind: Any,
    rhs_operation_kind: Any,
    lhs_status: Any,
    rhs_status: Any,
    lhs_candidate_role: Any,
    rhs_candidate_role: Any,
    lhs_changed_file_count: int,
    rhs_changed_file_count: int,
    lineage_relationship: str | None,
) -> dict[str, Any]:
    sibling_candidate_refs: list[str] = []
    if (
        lineage_relationship == "same_parent"
        and lhs_candidate_id is not None
        and rhs_candidate_id is not None
        and lhs_candidate_id != rhs_candidate_id
    ):
        sibling_candidate_refs = [lhs_candidate_id, rhs_candidate_id]
    return {
        "lhs_candidate_id": lhs_candidate_id,
        "rhs_candidate_id": rhs_candidate_id,
        "lhs_parent_candidate_id": lhs_parent_candidate_id,
        "rhs_parent_candidate_id": rhs_parent_candidate_id,
        "lhs_transition_kind": str(lhs_transition_kind) if lhs_transition_kind is not None else None,
        "rhs_transition_kind": str(rhs_transition_kind) if rhs_transition_kind is not None else None,
        "lhs_operation_kind": str(lhs_operation_kind) if lhs_operation_kind is not None else None,
        "rhs_operation_kind": str(rhs_operation_kind) if rhs_operation_kind is not None else None,
        "lhs_candidate_role": str(lhs_candidate_role) if lhs_candidate_role is not None else None,
        "rhs_candidate_role": str(rhs_candidate_role) if rhs_candidate_role is not None else None,
        "lhs_candidate_role_group": _candidate_role_group(lhs_candidate_role),
        "rhs_candidate_role_group": _candidate_role_group(rhs_candidate_role),
        "lhs_candidate_status": str(lhs_status) if lhs_status is not None else None,
        "rhs_candidate_status": str(rhs_status) if rhs_status is not None else None,
        "lhs_changed_file_count": lhs_changed_file_count,
        "rhs_changed_file_count": rhs_changed_file_count,
        "changed_file_count_delta": rhs_changed_file_count - lhs_changed_file_count,
        "lineage_relationship": lineage_relationship,
        "parent_candidate_ref": rhs_parent_candidate_id or lhs_parent_candidate_id,
        "sibling_candidate_refs": sibling_candidate_refs,
    }


def _compare_type(
    *,
    lineage_relationship: str | None,
    lhs_patch_present: bool,
    rhs_patch_present: bool,
    lhs_candidate_ref: str | None,
    rhs_candidate_ref: str | None,
) -> str:
    if lineage_relationship in {"lhs_parent_of_rhs", "rhs_parent_of_lhs"}:
        return "parent_child_candidate"
    if lineage_relationship == "same_parent":
        return "sibling_candidate"
    if rhs_candidate_ref is not None and lhs_candidate_ref is None:
        return "baseline_to_candidate"
    if rhs_candidate_ref is not None and lhs_candidate_ref is not None and rhs_candidate_ref != lhs_candidate_ref:
        return "baseline_to_candidate"
    if rhs_patch_present and not lhs_patch_present:
        return "baseline_to_candidate"
    if lhs_patch_present and rhs_patch_present:
        return "candidate_to_candidate"
    return "candidate_delta"


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
    projection["failure_localization"] = _load_optional_json(run_dir, "correctness/failure_localization.json")
    projection["determinism_summary"] = _load_optional_json(run_dir, "correctness/determinism.json")
    projection["anti_hack_summary"] = _load_optional_json(run_dir, "eval/anti_hack_report.json")
    projection["eval_envelope"] = _load_optional_json(run_dir, "eval/eval_envelope.json")
    projection["learning_reward_trace"] = _load_optional_json(run_dir, "eval/learning_reward_trace.json")
    projection["gate_summary"] = _load_optional_json(run_dir, "eval/gate_summary.json")
    projection["replay_pack"] = _load_optional_json(run_dir, "replay/replay_pack.json")
    projection["build_record"] = _load_optional_json(run_dir, "build/build_record.json")
    projection["tri_view"] = _load_optional_json(run_dir, "build/tri_view.json")
    projection["build_projection"] = _build_projection(run_dir)
    projection["candidate_projection"] = _candidate_projection(run_dir)
    projection["command_payload"] = _load_command_payload(run_dir)
    projection["public_benchmark_projection"] = _public_benchmark_projection(projection["command_payload"])
    projection["hardware_fingerprint"] = _hardware_fingerprint_projection(run_dir)
    projection["evidence_quality"] = assess_run_evidence(run_dir).model_dump(mode="json")
    projection["governance_score"] = projection["evidence_quality"]
    projection["failure_triage"] = _failure_triage(run_dir)
    projection["profile_triage"] = _profile_triage(projection)
    projection["perf_localization"] = _perf_localization_packet(run_dir, projection)
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
                "candidate_status",
                "candidate_origin_kind",
                "candidate_operation_kind",
                "candidate_diff_ref",
                "exit_code",
                "duration_ms",
                "warnings",
                "key_artifacts",
            ]
        }
    section_map = {
        "build": ["build_record", "tri_view", "build_projection"],
        "eval": ["correctness_summary", "failure_localization", "determinism_summary", "anti_hack_summary", "eval_envelope", "gate_summary"],
        "profile": ["profile_summary", "sanitizer_summary", "bottleneck_card", "profile_triage", "perf_localization", "hardware_fingerprint"],
        "replay": ["replay_validation", "replay_pack"],
        "quality": ["evidence_quality", "governance_score", "learning_reward_trace", "training_readiness", "training_trace_triage", "failure_triage", "failure_localization", "profile_triage", "perf_localization", "public_benchmark_projection", "hardware_fingerprint", "recommended_next_actions"],
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
    lhs_command_payload = _load_command_payload(lhs_run_dir)
    rhs_command_payload = _load_command_payload(rhs_run_dir)
    lhs_public_benchmark = _public_benchmark_projection(lhs_command_payload)
    rhs_public_benchmark = _public_benchmark_projection(rhs_command_payload)
    lhs_build_record = lhs_build_projection.get("record") or {}
    rhs_build_record = rhs_build_projection.get("record") or {}
    lhs_candidate_state = lhs_candidate_projection.get("candidate_state") or {}
    rhs_candidate_state = rhs_candidate_projection.get("candidate_state") or {}
    lhs_transition = lhs_candidate_projection.get("transition") or {}
    rhs_transition = rhs_candidate_projection.get("transition") or {}
    lhs_patch = lhs_candidate_projection.get("applied_patch") or {}
    rhs_patch = rhs_candidate_projection.get("applied_patch") or {}
    lhs_parent_candidate_projection = _parent_candidate_projection(root, lhs.parent_run_id)
    rhs_parent_candidate_projection = _parent_candidate_projection(root, rhs.parent_run_id)
    lhs_effective_candidate_projection = lhs_candidate_projection or lhs_parent_candidate_projection or {}
    rhs_effective_candidate_projection = rhs_candidate_projection or rhs_parent_candidate_projection or {}
    lhs_candidate_state = lhs_effective_candidate_projection.get("candidate_state") or lhs_candidate_state
    rhs_candidate_state = rhs_effective_candidate_projection.get("candidate_state") or rhs_candidate_state
    lhs_transition = lhs_effective_candidate_projection.get("transition") or lhs_transition
    rhs_transition = rhs_effective_candidate_projection.get("transition") or rhs_transition
    lhs_patch = lhs_effective_candidate_projection.get("applied_patch") or lhs_patch
    rhs_patch = rhs_effective_candidate_projection.get("applied_patch") or rhs_patch
    lhs_patch_present = bool(lhs_effective_candidate_projection.get("patch_present"))
    rhs_patch_present = bool(rhs_effective_candidate_projection.get("patch_present"))
    lhs_candidate_operation_kind = lhs_effective_candidate_projection.get("candidate_operation_kind")
    rhs_candidate_operation_kind = rhs_effective_candidate_projection.get("candidate_operation_kind")
    lhs_optimization_summary = _optimization_summary_from_payload(lhs_command_payload)
    rhs_optimization_summary = _optimization_summary_from_payload(rhs_command_payload)
    if lhs_public_benchmark is None:
        lhs_patch_kind = str(lhs_patch.get("patch_kind")) if lhs_patch.get("patch_kind") is not None else None
        if lhs_patch_kind is None and isinstance(lhs_parent_candidate_projection, dict):
            parent_patch = lhs_parent_candidate_projection.get("applied_patch") or {}
            lhs_patch_kind = str(parent_patch.get("patch_kind")) if parent_patch.get("patch_kind") is not None else None
        lhs_public_benchmark = _infer_public_benchmark_projection(
            lhs.task_id,
            patch_present=lhs_patch_present or bool(lhs_parent_candidate_projection and lhs_parent_candidate_projection.get("patch_present")),
            patch_kind=lhs_patch_kind,
        )
    if rhs_public_benchmark is None:
        rhs_patch_kind = str(rhs_patch.get("patch_kind")) if rhs_patch.get("patch_kind") is not None else None
        if rhs_patch_kind is None and isinstance(rhs_parent_candidate_projection, dict):
            parent_patch = rhs_parent_candidate_projection.get("applied_patch") or {}
            rhs_patch_kind = str(parent_patch.get("patch_kind")) if parent_patch.get("patch_kind") is not None else None
        rhs_public_benchmark = _infer_public_benchmark_projection(
            rhs.task_id,
            patch_present=rhs_patch_present or bool(rhs_parent_candidate_projection and rhs_parent_candidate_projection.get("patch_present")),
            patch_kind=rhs_patch_kind,
        )
    if lhs_optimization_summary is None and isinstance(lhs_public_benchmark, dict):
        summary = lhs_public_benchmark.get("optimization_summary")
        if isinstance(summary, dict):
            lhs_optimization_summary = summary
    if rhs_optimization_summary is None and isinstance(rhs_public_benchmark, dict):
        summary = rhs_public_benchmark.get("optimization_summary")
        if isinstance(summary, dict):
            rhs_optimization_summary = summary
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
    lhs_candidate_id = str(lhs_candidate_state["candidate_id"]) if lhs_candidate_state.get("candidate_id") is not None else None
    rhs_candidate_id = str(rhs_candidate_state["candidate_id"]) if rhs_candidate_state.get("candidate_id") is not None else None
    lhs_parent_candidate_id = (
        str(lhs_candidate_state["parent_candidate_id"]) if lhs_candidate_state.get("parent_candidate_id") is not None else None
    )
    rhs_parent_candidate_id = (
        str(rhs_candidate_state["parent_candidate_id"]) if rhs_candidate_state.get("parent_candidate_id") is not None else None
    )
    lhs_changed_file_count = len(lhs_candidate_state.get("changed_files", [])) if isinstance(lhs_candidate_state.get("changed_files"), list) else 0
    rhs_changed_file_count = len(rhs_candidate_state.get("changed_files", [])) if isinstance(rhs_candidate_state.get("changed_files"), list) else 0
    lineage_relationship, parent_child_related = _lineage_relationship(
        lhs_candidate_id=lhs_candidate_id,
        rhs_candidate_id=rhs_candidate_id,
        lhs_parent_candidate_id=lhs_parent_candidate_id,
        rhs_parent_candidate_id=rhs_parent_candidate_id,
    )
    lhs_correctness_gate = lhs_eval.get("correctness_gate")
    rhs_correctness_gate = rhs_eval.get("correctness_gate")
    trainworthiness_change = None
    if lhs_evidence.training_example_kind != rhs_evidence.training_example_kind:
        trainworthiness_change = f"{lhs_evidence.training_example_kind}->{rhs_evidence.training_example_kind}"
    elif lhs_evidence.overall_score != rhs_evidence.overall_score:
        direction = "improved" if rhs_evidence.overall_score > lhs_evidence.overall_score else "regressed"
        trainworthiness_change = f"evidence_{direction}"
    summary_lines: list[str] = []
    if lhs_correctness_gate == "fail" and rhs_correctness_gate == "pass":
        summary_lines.append("Correctness recovered in the rhs run.")
    elif lhs_correctness_gate == "pass" and rhs_correctness_gate == "fail":
        summary_lines.append("Correctness regressed in the rhs run.")
    if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)):
        if rhs_perf_p50 < lhs_perf_p50:
            summary_lines.append("Steady-state benchmark latency improved in the rhs run.")
        elif rhs_perf_p50 > lhs_perf_p50:
            summary_lines.append("Steady-state benchmark latency regressed in the rhs run.")
    if trainworthiness_change is not None:
        summary_lines.append(f"Trainworthiness changed: `{trainworthiness_change}`.")
    if lineage_relationship == "lhs_parent_of_rhs":
        summary_lines.append("The rhs run is a direct child candidate of the lhs run.")
    elif lineage_relationship == "rhs_parent_of_lhs":
        summary_lines.append("The lhs run is a direct child candidate of the rhs run.")
    if lhs_profile.get("classification") != rhs_profile.get("classification") and rhs_profile.get("classification") is not None:
        summary_lines.append(
            f"Profile classification changed from `{lhs_profile.get('classification')}` to `{rhs_profile.get('classification')}`."
        )
    if lhs_build_binary_hash != rhs_build_binary_hash and lhs_build_binary_hash is not None and rhs_build_binary_hash is not None:
        summary_lines.append("The compiled binary changed across the two runs.")
    if lhs_patch.get("patch_kind") != rhs_patch.get("patch_kind") and rhs_patch.get("patch_kind") is not None:
        summary_lines.append(
            f"Patch kind changed from `{lhs_patch.get('patch_kind')}` to `{rhs_patch.get('patch_kind')}`."
        )
    optimize_delta_summary, optimize_summary_lines, recommended_next_actions = _optimize_delta_summary(
        lhs_correctness_gate=lhs_correctness_gate,
        rhs_correctness_gate=rhs_correctness_gate,
        lhs_perf_p50=lhs_perf_p50,
        rhs_perf_p50=rhs_perf_p50,
        lhs_final_score=lhs_final_score,
        rhs_final_score=rhs_final_score,
        perf_improved=(
            True
            if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)) and rhs_perf_p50 < lhs_perf_p50
            else False
            if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float)) and rhs_perf_p50 >= lhs_perf_p50
            else None
        ),
        lineage_relationship=lineage_relationship,
        lhs_patch_kind=lhs_patch.get("patch_kind"),
        rhs_patch_kind=rhs_patch.get("patch_kind"),
        rhs_patch_present=rhs_patch_present,
        lhs_candidate_role=lhs_candidate_state.get("candidate_role"),
        rhs_candidate_role=rhs_candidate_state.get("candidate_role"),
        lhs_public_benchmark=lhs_public_benchmark,
        rhs_public_benchmark=rhs_public_benchmark,
        lhs_optimization_summary=lhs_optimization_summary,
        rhs_optimization_summary=rhs_optimization_summary,
    )
    candidate_delta_brief = _candidate_delta_brief(
        lhs_candidate_id=lhs_candidate_id,
        rhs_candidate_id=rhs_candidate_id,
        lhs_parent_candidate_id=lhs_parent_candidate_id,
        rhs_parent_candidate_id=rhs_parent_candidate_id,
        lhs_transition_kind=lhs_transition.get("transition_kind"),
        rhs_transition_kind=rhs_transition.get("transition_kind"),
        lhs_operation_kind=lhs_candidate_operation_kind,
        rhs_operation_kind=rhs_candidate_operation_kind,
        lhs_status=lhs_candidate_state.get("status"),
        rhs_status=rhs_candidate_state.get("status"),
        lhs_candidate_role=lhs_candidate_state.get("candidate_role"),
        rhs_candidate_role=rhs_candidate_state.get("candidate_role"),
        lhs_changed_file_count=lhs_changed_file_count,
        rhs_changed_file_count=rhs_changed_file_count,
        lineage_relationship=lineage_relationship,
    )
    compare_type = _compare_type(
        lineage_relationship=lineage_relationship,
        lhs_patch_present=lhs_patch_present,
        rhs_patch_present=rhs_patch_present,
        lhs_candidate_ref=lhs_optimization_summary.get("candidate_ref") if isinstance(lhs_optimization_summary, dict) else None,
        rhs_candidate_ref=rhs_optimization_summary.get("candidate_ref") if isinstance(rhs_optimization_summary, dict) else None,
    )
    benchmark_provenance = _benchmark_provenance_packet(lhs, rhs, lhs_public_benchmark, rhs_public_benchmark)
    perf_localization = {
        "lhs": lhs_projection.get("perf_localization"),
        "rhs": rhs_projection.get("perf_localization"),
        "delta_ms": (rhs_perf_p50 - lhs_perf_p50)
        if isinstance(lhs_perf_p50, (int, float)) and isinstance(rhs_perf_p50, (int, float))
        else None,
        "classification_change": {
            "lhs": lhs_profile.get("classification"),
            "rhs": rhs_profile.get("classification"),
        },
    }
    summary_lines.extend(line for line in optimize_summary_lines if line not in summary_lines)
    if lineage_relationship is not None:
        summary_lines.append(
            f"Candidate delta brief: relation `{lineage_relationship}`, changed files `{lhs_changed_file_count}->{rhs_changed_file_count}`."
        )
    return RunComparison(
        lhs_run_id=lhs.run_id,
        rhs_run_id=rhs.run_id,
        lhs_status=lhs.status,
        rhs_status=rhs.status,
        compare_type=compare_type,
        candidate_delta_brief=candidate_delta_brief,
        benchmark_provenance=benchmark_provenance,
        perf_localization=perf_localization,
        lhs_candidate_role=str(lhs_candidate_state.get("candidate_role")) if lhs_candidate_state.get("candidate_role") is not None else None,
        rhs_candidate_role=str(rhs_candidate_state.get("candidate_role")) if rhs_candidate_state.get("candidate_role") is not None else None,
        lhs_candidate_role_group=_candidate_role_group(lhs_candidate_state.get("candidate_role")),
        rhs_candidate_role_group=_candidate_role_group(rhs_candidate_state.get("candidate_role")),
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
        patch_hash_changed=_hash_delta(
            str(lhs_patch_hash) if lhs_patch_hash is not None else None,
            str(rhs_patch_hash) if rhs_patch_hash is not None else None,
        ),
        lhs_parent_candidate_id=lhs_parent_candidate_id,
        rhs_parent_candidate_id=rhs_parent_candidate_id,
        lineage_relationship=lineage_relationship,
        parent_child_related=parent_child_related,
        lhs_changed_file_count=lhs_changed_file_count,
        rhs_changed_file_count=rhs_changed_file_count,
        changed_file_count_delta=rhs_changed_file_count - lhs_changed_file_count,
        lhs_transition_kind=str(lhs_transition["transition_kind"]) if lhs_transition.get("transition_kind") is not None else None,
        rhs_transition_kind=str(rhs_transition["transition_kind"]) if rhs_transition.get("transition_kind") is not None else None,
        lhs_candidate_id=lhs_candidate_id,
        rhs_candidate_id=rhs_candidate_id,
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
        lhs_build_source_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("source"),
        rhs_build_source_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("source"),
        build_source_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("source"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("source"),
        ),
        lhs_build_ttir_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("ttir"),
        rhs_build_ttir_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("ttir"),
        build_ttir_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("ttir"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("ttir"),
        ),
        lhs_build_ttgir_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("ttgir"),
        rhs_build_ttgir_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("ttgir"),
        build_ttgir_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("ttgir"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("ttgir"),
        ),
        lhs_build_llir_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("llir"),
        rhs_build_llir_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("llir"),
        build_llir_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("llir"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("llir"),
        ),
        lhs_build_ptx_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("ptx"),
        rhs_build_ptx_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("ptx"),
        build_ptx_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("ptx"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("ptx"),
        ),
        lhs_build_sass_hash=(lhs_build_projection.get("artifact_hashes") or {}).get("sass"),
        rhs_build_sass_hash=(rhs_build_projection.get("artifact_hashes") or {}).get("sass"),
        build_sass_hash_changed=_hash_delta(
            (lhs_build_projection.get("artifact_hashes") or {}).get("sass"),
            (rhs_build_projection.get("artifact_hashes") or {}).get("sass"),
        ),
        optimize_delta_summary=optimize_delta_summary,
        recommended_next_actions=recommended_next_actions,
        summary_lines=summary_lines,
    )
