from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.engine.trajectory import export_episode_dataset, validate_trajectory_dataset
from scripts.build_optimize_trace_training_assets import _episode_from_report


CURATED_HARD_SLICE_V2: list[dict[str, Any]] = [
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "internal_hard_positive",
        "source_block": "phase5_block_a",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "public_hard_positive",
        "source_block": "phase5_block_a",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_branch_v1/batch_v1/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "internal_hard_positive",
        "source_block": "phase5_block_b",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_branch_v1/batch_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "public_hard_positive",
        "source_block": "phase5_block_b",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_packet_v1/batch_v1/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "near_miss",
        "slice_role": "internal_hard_near_miss",
        "source_block": "phase5_block_b",
        "interface_profile": "compare_packet_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "split": "train",
        "quality_bucket": "usable_negative",
        "slice_role": "branch_revert_negative",
        "source_block": "phase3_public_multi_candidate",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "auxiliary_branch_positive",
        "source_block": "phase4_branching",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_kbv3_teacher_slice_v1/batch_v1_retry1/task__kernelbench_v3__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "kbv3_curated_positive",
        "source_block": "phase5_kbv3_teacher_slice",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_v3_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "usable_positive",
        "slice_role": "control_positive",
        "source_block": "phase5_block_a",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "near_miss",
        "slice_role": "public_hard_near_miss",
        "source_block": "phase4_two_attempt",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "near_miss",
        "slice_role": "public_hard_near_miss",
        "source_block": "phase4_final_teacher",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "split": "dev",
        "quality_bucket": "usable_negative",
        "slice_role": "branch_revert_negative",
        "source_block": "phase4_final_teacher",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_kbv3_teacher_slice_v1/batch_v1_retry1/task__kernelbench_v3__level1__23_softmax_official__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "usable_positive",
        "slice_role": "kbv3_official_positive",
        "source_block": "phase5_kbv3_teacher_slice",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_v3_official",
        "trainability": "held_out",
    },
]


def _load_payload(relative_ref: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_ref).read_text(encoding="utf-8"))


def _task_scope(task_ref: str) -> str:
    if "kernelbench_v3" in task_ref:
        return "public_kbv3"
    return "public" if "kernelbench" in task_ref else "internal"


def _candidate_mode(counters: dict[str, Any]) -> str:
    if any(int(counters.get(key, 0)) > 0 for key in ("branches", "reverts", "promotes")):
        return "multi_candidate"
    if int(counters.get("patches", 0)) > 0:
        return "single_candidate"
    return "reference_only"


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _aggregate_counters(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "patches": 0,
        "compares": 0,
        "branches": 0,
        "reverts": 0,
        "promotes": 0,
        "replays": 0,
        "bench_actions": 0,
        "eval_actions": 0,
    }
    for row in rows:
        counters = row.get("counters", {})
        if not isinstance(counters, dict):
            continue
        for key in totals:
            totals[key] += int(counters.get(key, 0))
    return totals


def _lineage_metadata(payload: dict[str, Any], counters: dict[str, Any]) -> dict[str, Any]:
    state = payload.get("state", {})
    if not isinstance(state, dict):
        state = {}
    lineage = state.get("candidate_lineage", {})
    if not isinstance(lineage, dict):
        lineage = {}
    branch_depth = int(lineage.get("current_candidate_attempt_index") or counters.get("branches", 0) or 0)
    history_length = int(lineage.get("history_length") or len(state.get("candidate_history", []) or []))
    return {
        "branch_depth": branch_depth,
        "lineage_history_length": history_length,
        "best_known_candidate_id": lineage.get("best_known_candidate_id"),
        "best_known_candidate_reason": lineage.get("best_known_candidate_reason"),
        "candidate_tree_mode": "branching" if branch_depth > 1 or history_length > 1 else "flat",
    }


def _provenance_metadata(task_ref: str, spec: dict[str, Any]) -> dict[str, Any]:
    provenance_kind = str(spec["provenance_kind"])
    benchmark_source = "internal"
    benchmark_track = "curated"
    if provenance_kind.startswith("kernelbench_v3"):
        benchmark_source = "kernelbench_v3"
        benchmark_track = "official" if provenance_kind.endswith("official") else "curated"
    elif provenance_kind.startswith("kernelbench"):
        benchmark_source = "kernelbench"
        benchmark_track = "curated"
    return {
        "provenance_kind": provenance_kind,
        "benchmark_source": benchmark_source,
        "benchmark_track": benchmark_track,
        "trainability": str(spec["trainability"]),
        "held_out": bool(spec["trainability"] == "held_out"),
    }


def _entry(spec: dict[str, Any], registry: TaskRegistry) -> dict[str, Any]:
    relative_ref = str(spec["episode_ref"])
    payload = _load_payload(relative_ref)
    task_ref = str(payload.get("task_ref"))
    task = registry.get(task_ref)
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    lineage = _lineage_metadata(payload, counters)
    provenance = _provenance_metadata(task_ref, spec)
    return {
        "episode_ref": relative_ref,
        "split": str(spec["split"]),
        "quality_bucket": str(spec["quality_bucket"]),
        "slice_role": str(spec["slice_role"]),
        "source_block": str(spec["source_block"]),
        "task_ref": task_ref,
        "variant": str(payload.get("variant", "")),
        "verb": str(payload.get("verb", "")),
        "difficulty": task.difficulty,
        "task_scope": _task_scope(task_ref),
        "candidate_mode": _candidate_mode(counters),
        "success": bool(payload.get("success")),
        "terminal_reason": str(payload.get("terminal_reason", "")),
        "step_count": int(payload.get("step_count", 0)),
        "counters": counters,
        "interface_profile": str(spec.get("interface_profile") or "compare_plus_localization_plus_branch_v1"),
        "multi_candidate_mode": payload.get("task_context", {}).get("multi_candidate_mode")
        if isinstance(payload.get("task_context"), dict)
        else None,
        "teacher_model": str(payload.get("model", "")),
        "teacher_provider": str(payload.get("provider", "")),
        **lineage,
        **provenance,
    }


def _coverage_report(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "split_counts": manifest["split_counts"],
        "quality_counts": manifest["quality_counts"],
        "task_scope_counts": manifest["task_scope_counts"],
        "difficulty_counts": manifest["difficulty_counts"],
        "candidate_mode_counts": manifest["candidate_mode_counts"],
        "slice_role_counts": manifest["slice_role_counts"],
        "source_block_counts": manifest["source_block_counts"],
        "provenance_kind_counts": _count_by(entries, "provenance_kind"),
        "trainability_counts": _count_by(entries, "trainability"),
        "interface_profile_counts": _count_by(entries, "interface_profile"),
        "branch_depth_counts": _count_by(entries, "branch_depth"),
        "teacher_model_counts": _count_by(entries, "teacher_model"),
        "aggregate_counters": manifest["aggregate_counters"],
    }


def _contamination_audit(entries: list[dict[str, Any]], manifest_id: str, created_at: str) -> dict[str, Any]:
    held_out_train_refs = [
        row["episode_ref"] for row in entries if row.get("trainability") == "held_out" and row.get("split") == "train"
    ]
    official_train_refs = [
        row["episode_ref"]
        for row in entries
        if row.get("benchmark_track") == "official" and row.get("split") == "train"
    ]
    checks = {
        "held_out_traces_excluded_from_train": not held_out_train_refs,
        "official_kbv3_traces_excluded_from_train": not official_train_refs,
        "all_rows_have_provenance_kind": all(bool(row.get("provenance_kind")) for row in entries),
        "all_rows_have_interface_profile": all(bool(row.get("interface_profile")) for row in entries),
    }
    return {
        "report_id": f"{manifest_id}_contamination_audit",
        "created_at": created_at,
        "checks": checks,
        "held_out_train_refs": held_out_train_refs,
        "official_train_refs": official_train_refs,
        "status": "pass" if all(checks.values()) else "fail",
    }


def _readiness_report(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "checks": {
            "has_internal_hard_positive": any(row["slice_role"] == "internal_hard_positive" for row in entries),
            "has_public_hard_positive": any(row["slice_role"] == "public_hard_positive" for row in entries),
            "has_kbv3_curated_positive": any(row["slice_role"] == "kbv3_curated_positive" for row in entries),
            "has_kbv3_official_positive": any(row["slice_role"] == "kbv3_official_positive" for row in entries),
            "has_near_miss": any(row["quality_bucket"] == "near_miss" for row in entries),
            "has_usable_negative": any(row["quality_bucket"] == "usable_negative" for row in entries),
            "has_branch_depth_ge_3": any(int(row.get("branch_depth", 0)) >= 3 for row in entries),
            "held_out_entries_confined_to_dev": all(
                row["split"] != "train" for row in entries if row.get("trainability") == "held_out"
            ),
        },
        "recommended_next_use": {
            "train_refs": [row["episode_ref"] for row in entries if row["split"] == "train"],
            "dev_refs": [row["episode_ref"] for row in entries if row["split"] == "dev"],
        },
    }


def _write_note(out_path: Path, manifest: dict[str, Any], contamination: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Hard Trace Freeze v2",
        "",
        "## Purpose",
        "",
        "Freeze the canonical post-contract hard-slice corpus for environment-complete Phase 5 learning and evaluation.",
        "",
        "This slice is intended to support:",
        "",
        "- artifact-feedback distillation",
        "- teacher-corrected learning views",
        "- branch-aware narrow RL and ranking exports",
        "- provenance-clean hard-slice evaluation",
        "",
        "## Composition",
        "",
        f"- dataset id: `{manifest['dataset_id']}`",
        f"- episode count: `{manifest['episode_count']}`",
        f"- train count: `{manifest['split_counts'].get('train', 0)}`",
        f"- dev count: `{manifest['split_counts'].get('dev', 0)}`",
        f"- usable positives: `{manifest['quality_counts'].get('usable_positive', 0)}`",
        f"- usable negatives: `{manifest['quality_counts'].get('usable_negative', 0)}`",
        f"- near misses: `{manifest['quality_counts'].get('near_miss', 0)}`",
        "",
        "## Provenance guarantees",
        "",
        "- records official-vs-curated benchmark provenance on every episode",
        "- records held-out-vs-trainable status on every episode",
        "- keeps official KB-v3 held-out traces out of the train split",
        f"- contamination audit status: `{contamination['status']}`",
        "",
        "## What is in the slice",
        "",
        "- internal hard positives on `attention_score`",
        "- public hard positives on `kernelbench/level1/23_softmax_wide`",
        "- branch-aware negatives and near-misses from the branch-and-rank regime",
        "- a three-attempt internal branching positive",
        "- one KB-v3 curated positive trace",
        "- one KB-v3 official held-out positive trace",
        "",
        "## Why this freeze matters",
        "",
        "- It is the first default corpus with explicit contamination and provenance semantics.",
        "- It preserves branch depth and interface-profile metadata needed for learning-view regeneration.",
        "- It cleanly separates trainable curated traces from held-out official KB-v3 traces.",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _unique_episode_id(relative_ref: str) -> str:
    ref_path = Path(relative_ref)
    parts = list(ref_path.parts)
    if parts and parts[-1].endswith(".json"):
        parts[-1] = parts[-1][:-5]
    slug = "__".join(parts[-4:]).replace("-", "_").replace("/", "__")
    return f"converted__{slug}"


def _episode_from_report_unique(relative_ref: str):
    episode = _episode_from_report(relative_ref)
    metadata = dict(episode.metadata)
    metadata["phase5_hard_trace_freeze_ref"] = relative_ref
    return episode.model_copy(
        update={
            "episode_id": _unique_episode_id(relative_ref),
            "metadata": metadata,
        }
    )


def main() -> int:
    registry = TaskRegistry(ROOT)
    out_dir = ROOT / "artifacts" / "training" / "phase5_hard_trace_freeze_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [_entry(spec, registry) for spec in CURATED_HARD_SLICE_V2]
    created_at = datetime.now(tz=UTC).isoformat()
    aggregate_counters = _aggregate_counters(entries)

    manifest = {
        "dataset_id": "phase5_hard_trace_freeze_v2",
        "created_at": created_at,
        "schema_version": "2.0.0",
        "objective": "Freeze the canonical post-contract Phase 5 hard-positive / near-miss / branch-aware corpus with KB-v3 provenance.",
        "episode_count": len(entries),
        "split_counts": _count_by(entries, "split"),
        "quality_counts": _count_by(entries, "quality_bucket"),
        "task_counts": _count_by(entries, "task_ref"),
        "task_scope_counts": _count_by(entries, "task_scope"),
        "difficulty_counts": _count_by(entries, "difficulty"),
        "candidate_mode_counts": _count_by(entries, "candidate_mode"),
        "slice_role_counts": _count_by(entries, "slice_role"),
        "source_block_counts": _count_by(entries, "source_block"),
        "aggregate_counters": aggregate_counters,
        "episodes": entries,
        "notes": [
            "Canonical Phase 5 hard-slice built from the validated hard-target positives, near-misses, branch/revert negatives, and a minimal KB-v3 teacher slice.",
            "Records official-vs-curated provenance and held-out-vs-trainable status on every episode.",
            "Replaces phase5_hard_trace_freeze_v1 as the default environment-complete corpus.",
        ],
    }

    contamination = _contamination_audit(entries, manifest["dataset_id"], created_at)
    coverage_report = _coverage_report(manifest, entries)
    readiness_report = _readiness_report(manifest, entries)

    outputs = {
        "train_trajectory": ROOT / "datasets" / "phase5_hard_trace_transition_train_v2",
        "dev_trajectory": ROOT / "datasets" / "phase5_hard_trace_transition_dev_v2",
        "train_sft": ROOT / "datasets" / "phase5_hard_trace_sft_train_v2",
        "dev_sft": ROOT / "datasets" / "phase5_hard_trace_sft_dev_v2",
    }
    train_refs = [row["episode_ref"] for row in entries if row["split"] == "train"]
    dev_refs = [row["episode_ref"] for row in entries if row["split"] == "dev"]
    train_episodes = [_episode_from_report_unique(ref) for ref in train_refs]
    dev_episodes = [_episode_from_report_unique(ref) for ref in dev_refs]

    train_manifest = export_episode_dataset(
        train_episodes,
        outputs["train_trajectory"],
        policy_id="gpt54_phase5_hard_trace_converter_v2",
        split="train",
    )
    dev_manifest = export_episode_dataset(
        dev_episodes,
        outputs["dev_trajectory"],
        policy_id="gpt54_phase5_hard_trace_converter_v2",
        split="dev",
    )
    train_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["train_trajectory"],
        outputs["train_sft"],
        split="train",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=False,
    )
    dev_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["dev_trajectory"],
        outputs["dev_sft"],
        split="dev",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=False,
    )

    training_assets_report = {
        "report_id": "phase5_hard_trace_training_assets_v2",
        "created_at": created_at,
        "dataset_id": manifest["dataset_id"],
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
        "train_trajectory_manifest": str(train_manifest.relative_to(ROOT)),
        "dev_trajectory_manifest": str(dev_manifest.relative_to(ROOT)),
        "train_sft_manifest": str(train_sft_manifest.relative_to(ROOT)),
        "dev_sft_manifest": str(dev_sft_manifest.relative_to(ROOT)),
        "train_trajectory_validation": validate_trajectory_dataset(outputs["train_trajectory"]),
        "dev_trajectory_validation": validate_trajectory_dataset(outputs["dev_trajectory"]),
        "train_sft_validation": validate_sft_dataset(outputs["train_sft"]),
        "dev_sft_validation": validate_sft_dataset(outputs["dev_sft"]),
    }

    (out_dir / "optimize_trace_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "optimize_trace_coverage_report.json").write_text(json.dumps(coverage_report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "training_readiness_report.json").write_text(json.dumps(readiness_report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "contamination_audit_report.json").write_text(json.dumps(contamination, indent=2) + "\n", encoding="utf-8")
    (out_dir / "training_assets_report.json").write_text(json.dumps(training_assets_report, indent=2) + "\n", encoding="utf-8")
    _write_note(out_dir / "phase5_hard_trace_freeze_note.md", manifest, contamination)

    print(
        json.dumps(
            {
                "manifest": manifest,
                "coverage_report": coverage_report,
                "contamination_audit": contamination,
                "training_assets_report": training_assets_report,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
