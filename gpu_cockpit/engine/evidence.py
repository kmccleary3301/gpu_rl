from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gpu_cockpit.contracts import EvidenceQualityReport, ReadinessDecision


def _load_optional_json(run_dir: Path, relative_path: str) -> dict[str, Any] | None:
    path = run_dir / relative_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _has_any(run_dir: Path, relative_paths: list[str]) -> bool:
    return any((run_dir / relative_path).exists() for relative_path in relative_paths)


def _ratio(present: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(present / total, 4)


def _decision(eligible: bool, reasons: list[str]) -> ReadinessDecision:
    return ReadinessDecision(eligible=eligible, reasons=reasons)


def _public_benchmark_task(task_id: str | None) -> bool:
    if task_id is None:
        return False
    return task_id.startswith("task/kernelbench/") or task_id.startswith("task/computeeval/")


def _training_example_kind(
    *,
    status: str,
    benchmark_ready: bool,
    allow_benchmark_only: bool,
    sft_ready: bool,
    rl_ready: bool,
    eval_complete: bool,
) -> str:
    if status == "ok" and rl_ready:
        return "positive_rl_trace"
    if status == "ok" and sft_ready:
        return "positive_sft_example"
    if status != "ok" and eval_complete and sft_ready:
        return "negative_debug_example"
    if benchmark_ready and allow_benchmark_only:
        return "benchmark_only"
    return "unusable"


def _task_verb(task_spec: dict[str, Any]) -> str | None:
    verb = task_spec.get("verb")
    if isinstance(verb, str) and verb:
        return verb
    return None


def _requires_rich_optimize_evidence(task_verb: str | None) -> bool:
    return task_verb in {"optimize", "reformulate"}


def _requires_diagnostic_evidence(task_verb: str | None) -> bool:
    return task_verb in {"diagnose", "debug"}


def assess_run_evidence(run_dir: Path) -> EvidenceQualityReport:
    summary_payload = _load_optional_json(run_dir, "summary.json") or _load_optional_json(run_dir, "manifest.json") or {}
    task_spec = _load_optional_json(run_dir, "prompt/task_spec.json") or {}
    replay_pack = _load_optional_json(run_dir, "replay/replay_pack.json") or {}
    eval_envelope = _load_optional_json(run_dir, "eval/eval_envelope.json") or {}

    required_artifacts = [str(path) for path in task_spec.get("required_artifacts", [])]
    missing_required_artifacts = [path for path in required_artifacts if not (run_dir / path).exists()]
    required_completeness = 1.0 if not required_artifacts else _ratio(len(required_artifacts) - len(missing_required_artifacts), len(required_artifacts))

    replay_items = [
        "replay/replay_pack.json",
        "replay/command.json",
        "replay/environment.json",
        "meta/doctor_report.json",
    ]
    build_items = [
        "build/build_record.json",
        "build/tri_view.json",
        "build/source_map_summary.json",
    ]
    profile_items = [
        "traces/system/summary.json",
        "profiles/kernel/summary.json",
        "bottlenecks/primary.json",
    ]
    replay_completeness = _ratio(sum((run_dir / path).exists() for path in replay_items), len(replay_items))
    build_completeness = _ratio(sum((run_dir / path).exists() for path in build_items), len(build_items))
    present_profile_items = sum((run_dir / path).exists() for path in profile_items)
    if _has_any(
        run_dir,
        [
            "sanitize/memcheck_summary.json",
            "sanitize/racecheck_summary.json",
            "sanitize/initcheck_summary.json",
            "sanitize/synccheck_summary.json",
        ],
    ):
        present_profile_items += 1
    profile_completeness = _ratio(present_profile_items, len(profile_items) + 1)

    eval_items = [
        "correctness/correctness.json",
        "correctness/determinism.json",
        "eval/anti_hack_report.json",
        "eval/eval_envelope.json",
        "eval/gate_summary.json",
        "perf/benchmark.json",
    ]
    eval_completeness = _ratio(sum((run_dir / path).exists() for path in eval_items), len(eval_items))

    task_id = None
    if summary_payload.get("task_id") is not None:
        task_id = str(summary_payload["task_id"])
    elif summary_payload.get("task_ref") is not None:
        task_id = str(summary_payload["task_ref"])
    task_verb = _task_verb(task_spec)

    missing_provenance_fields: list[str] = []
    if not (run_dir / "replay" / "replay_pack.json").exists():
        missing_provenance_fields.append("replay_pack")
    if not (run_dir / "replay" / "environment.json").exists():
        missing_provenance_fields.append("environment")
    if not (run_dir / "replay" / "command.json").exists():
        missing_provenance_fields.append("command")
    if not (run_dir / "meta" / "doctor_report.json").exists():
        missing_provenance_fields.append("doctor_report")
    if not (run_dir / "summary.json").exists():
        missing_provenance_fields.append("summary")
    if _public_benchmark_task(task_id):
        if not task_spec.get("baseline_ref"):
            missing_provenance_fields.append("benchmark_baseline_ref")
        if not task_spec.get("reference_impl_ref"):
            missing_provenance_fields.append("benchmark_reference_impl_ref")
    provenance_total = 5 + (2 if _public_benchmark_task(task_id) else 0)
    provenance_completeness = _ratio(provenance_total - len(missing_provenance_fields), provenance_total)

    governance_score = round(
        required_completeness * 0.35
        + eval_completeness * 0.25
        + replay_completeness * 0.20
        + build_completeness * 0.10
        + profile_completeness * 0.05
        + provenance_completeness * 0.05,
        4,
    )

    status = str(summary_payload.get("status", "unknown"))
    correctness_gate = str(eval_envelope.get("correctness_gate", "not_run"))
    determinism_gate = str(eval_envelope.get("determinism_gate", "not_run"))
    anti_hack_gate = str(eval_envelope.get("anti_hack_gate", "not_run"))
    perf_gate = str(eval_envelope.get("perf_gate", "not_run"))

    benchmark_reasons: list[str] = []
    if status != "ok":
        benchmark_reasons.append(f"run_status:{status}")
    if replay_completeness < 0.75:
        benchmark_reasons.append("replay_incomplete")
    if required_completeness < 1.0:
        benchmark_reasons.append("required_artifacts_missing")
    if eval_completeness < 1.0:
        benchmark_reasons.append("eval_artifacts_incomplete")
    if _requires_rich_optimize_evidence(task_verb) and perf_gate in {"not_run", "blocked"}:
        benchmark_reasons.append(f"perf_gate:{perf_gate}")
    if provenance_completeness < 1.0:
        benchmark_reasons.append("provenance_incomplete")

    sft_reasons: list[str] = []
    if not (run_dir / "summary.json").exists():
        sft_reasons.append("summary_missing")
    if not (run_dir / "prompt" / "task_spec.json").exists():
        sft_reasons.append("task_spec_missing")
    if replay_completeness < 0.5:
        sft_reasons.append("replay_too_sparse")
    if provenance_completeness < 0.8:
        sft_reasons.append("provenance_too_sparse")
    if _requires_rich_optimize_evidence(task_verb) and build_completeness == 0.0:
        sft_reasons.append("build_evidence_missing")
    if _requires_diagnostic_evidence(task_verb) and build_completeness == 0.0 and profile_completeness == 0.0:
        sft_reasons.append("diagnostic_evidence_missing")

    rl_reasons: list[str] = []
    if replay_completeness < 0.75:
        rl_reasons.append("replay_incomplete")
    if eval_completeness < 1.0:
        rl_reasons.append("eval_artifacts_incomplete")
    if required_completeness < 1.0:
        rl_reasons.append("required_artifacts_missing")
    if correctness_gate != "pass":
        rl_reasons.append(f"correctness_gate:{correctness_gate}")
    if determinism_gate != "pass":
        rl_reasons.append(f"determinism_gate:{determinism_gate}")
    if anti_hack_gate != "pass":
        rl_reasons.append(f"anti_hack_gate:{anti_hack_gate}")
    if provenance_completeness < 1.0:
        rl_reasons.append("provenance_incomplete")
    if _requires_rich_optimize_evidence(task_verb):
        if build_completeness < 0.67:
            rl_reasons.append("build_evidence_incomplete")
        if profile_completeness < 0.25:
            rl_reasons.append("profile_evidence_incomplete")
        if perf_gate != "pass":
            rl_reasons.append(f"perf_gate:{perf_gate}")
    elif _requires_diagnostic_evidence(task_verb):
        if build_completeness == 0.0 and profile_completeness == 0.0:
            rl_reasons.append("diagnostic_evidence_missing")

    notes: list[str] = []
    if missing_required_artifacts:
        notes.append("missing_required:" + ",".join(missing_required_artifacts))
    if build_completeness == 0.0:
        notes.append("no_build_artifacts")
    if profile_completeness == 0.0:
        notes.append("no_profile_artifacts")
    if not replay_pack:
        notes.append("no_replay_pack")

    training_example_kind = _training_example_kind(
        status=status,
        benchmark_ready=not benchmark_reasons,
        allow_benchmark_only=_public_benchmark_task(task_id) or _requires_rich_optimize_evidence(task_verb),
        sft_ready=not sft_reasons,
        rl_ready=not rl_reasons,
        eval_complete=eval_completeness >= 1.0,
    )

    return EvidenceQualityReport(
        run_id=str(summary_payload.get("run_id", run_dir.name)),
        task_id=task_id,
        required_artifact_count=len(required_artifacts),
        missing_required_artifact_count=len(missing_required_artifacts),
        required_artifact_completeness=required_completeness,
        replay_completeness=replay_completeness,
        build_completeness=build_completeness,
        profile_completeness=profile_completeness,
        eval_completeness=eval_completeness,
        provenance_completeness=provenance_completeness,
        overall_score=governance_score,
        benchmark_reporting=_decision(not benchmark_reasons, benchmark_reasons),
        sft_collection=_decision(not sft_reasons, sft_reasons),
        rl_reward_trace=_decision(not rl_reasons, rl_reasons),
        missing_provenance_fields=missing_provenance_fields,
        training_example_kind=training_example_kind,
        notes=notes,
    )
