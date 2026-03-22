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


CURATED_HARD_SLICE: list[dict[str, str]] = [
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "internal_hard_positive",
        "source_block": "phase5_block_a",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "public_hard_positive",
        "source_block": "phase5_block_a",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_branch_v1/batch_v1/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "internal_hard_positive",
        "source_block": "phase5_block_b",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_branch_v1/batch_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "public_hard_positive",
        "source_block": "phase5_block_b",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_packet_v1/batch_v1/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "near_miss",
        "slice_role": "internal_hard_near_miss",
        "source_block": "phase5_block_b",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "split": "train",
        "quality_bucket": "usable_negative",
        "slice_role": "branch_revert_negative",
        "source_block": "phase3_public_multi_candidate",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "auxiliary_branch_positive",
        "source_block": "phase4_branching",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5_block_a_hard_targets_v1/batch_v4_clean/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "usable_positive",
        "slice_role": "control_positive",
        "source_block": "phase5_block_a",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "near_miss",
        "slice_role": "public_hard_near_miss",
        "source_block": "phase4_two_attempt",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "near_miss",
        "slice_role": "public_hard_near_miss",
        "source_block": "phase4_final_teacher",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "split": "dev",
        "quality_bucket": "usable_negative",
        "slice_role": "branch_revert_negative",
        "source_block": "phase4_final_teacher",
    },
]


def _load_payload(relative_ref: str) -> dict[str, Any]:
    return json.loads((ROOT / relative_ref).read_text(encoding="utf-8"))


def _task_scope(task_ref: str) -> str:
    return "public" if "kernelbench" in task_ref else "internal"


def _candidate_mode(counters: dict[str, Any]) -> str:
    if any(int(counters.get(key, 0)) > 0 for key in ("branches", "reverts", "promotes")):
        return "multi_candidate"
    return "single_candidate"


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


def _entry(spec: dict[str, str], registry: TaskRegistry) -> dict[str, Any]:
    relative_ref = spec["episode_ref"]
    payload = _load_payload(relative_ref)
    task_ref = str(payload.get("task_ref"))
    task = registry.get(task_ref)
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    return {
        "episode_ref": relative_ref,
        "split": spec["split"],
        "quality_bucket": spec["quality_bucket"],
        "slice_role": spec["slice_role"],
        "source_block": spec["source_block"],
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
        "interface_profile": payload.get("task_context", {}).get("interface_profile")
        if isinstance(payload.get("task_context"), dict)
        else None,
        "multi_candidate_mode": payload.get("task_context", {}).get("multi_candidate_mode")
        if isinstance(payload.get("task_context"), dict)
        else None,
    }


def _write_note(out_path: Path, manifest: dict[str, Any]) -> None:
    lines = [
        "# Phase 5 Hard Trace Freeze v1",
        "",
        "## Purpose",
        "",
        "Freeze the first Phase 5 hard-slice dataset directly from the validated Block A/B tranche.",
        "",
        "This slice is intended to support:",
        "",
        "- hard positive distillation",
        "- near-miss analysis and correction",
        "- branch/revert policy experiments",
        "- the first artifact-feedback distill run",
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
        "## What is in the slice",
        "",
        "- internal hard positives on `attention_score`",
        "- public hard positives on `kernelbench/level1/23_softmax_wide`",
        "- internal and public hard near-misses from the pre-branch and two-attempt interfaces",
        "- branch/revert negatives on public softmax",
        "- one stable positive control and one auxiliary three-attempt branching positive",
        "",
        "## Why this freeze matters",
        "",
        "- It isolates the exact regime where Block B showed branch-and-rank was decisive.",
        "- It keeps earlier easier traces out of the primary Phase 5 hard-learning slice.",
        "- It provides both successful and almost-successful examples on the target hard tasks.",
        "- It preserves branch/revert structure needed for the first artifact-feedback learner.",
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
    out_dir = ROOT / "artifacts" / "training" / "phase5_hard_trace_freeze_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [_entry(spec, registry) for spec in CURATED_HARD_SLICE]
    aggregate_counters = _aggregate_counters(entries)

    manifest = {
        "dataset_id": "phase5_hard_trace_freeze_v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "schema_version": "1.0.0",
        "objective": "Freeze the first Phase 5 hard-positive / near-miss / branch-revert slice from Blocks A and B.",
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
            "Focused Phase 5 hard-slice built from validated hard-target positives, near-misses, and branch/revert negatives.",
            "Uses the Block B winning interface regime as the main positive source while preserving pre-branch near-misses for contrast.",
            "Keeps a small dev split with one control positive and alternate hard near-miss/negative traces.",
        ],
    }

    coverage_report = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "split_counts": manifest["split_counts"],
        "quality_counts": manifest["quality_counts"],
        "task_scope_counts": manifest["task_scope_counts"],
        "difficulty_counts": manifest["difficulty_counts"],
        "candidate_mode_counts": manifest["candidate_mode_counts"],
        "slice_role_counts": manifest["slice_role_counts"],
        "source_block_counts": manifest["source_block_counts"],
        "aggregate_counters": manifest["aggregate_counters"],
    }

    readiness_report = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "checks": {
            "has_internal_hard_positive": any(row["slice_role"] == "internal_hard_positive" for row in entries),
            "has_public_hard_positive": any(row["slice_role"] == "public_hard_positive" for row in entries),
            "has_internal_near_miss": any(row["slice_role"] == "internal_hard_near_miss" for row in entries),
            "has_public_near_miss": any(row["slice_role"] == "public_hard_near_miss" for row in entries),
            "has_branch_revert_negative": any(row["slice_role"] == "branch_revert_negative" for row in entries),
            "has_multi_candidate_positive": any(
                row["candidate_mode"] == "multi_candidate" and row["quality_bucket"] == "usable_positive"
                for row in entries
            ),
            "has_dev_control_positive": any(row["slice_role"] == "control_positive" for row in entries),
            "block_b_winning_interface_present": any(row["source_block"] == "phase5_block_b" and row["success"] for row in entries),
        },
        "recommended_next_use": {
            "artifact_feedback_distill_train_refs": [
                row["episode_ref"] for row in entries if row["split"] == "train"
            ],
            "hard_slice_dev_refs": [
                row["episode_ref"] for row in entries if row["split"] == "dev"
            ],
        },
    }

    outputs = {
        "train_trajectory": ROOT / "datasets" / "phase5_hard_trace_transition_train_v1",
        "dev_trajectory": ROOT / "datasets" / "phase5_hard_trace_transition_dev_v1",
        "train_sft": ROOT / "datasets" / "phase5_hard_trace_sft_train_v1",
        "dev_sft": ROOT / "datasets" / "phase5_hard_trace_sft_dev_v1",
    }
    train_refs = [row["episode_ref"] for row in entries if row["split"] == "train"]
    dev_refs = [row["episode_ref"] for row in entries if row["split"] == "dev"]
    train_episodes = [_episode_from_report_unique(ref) for ref in train_refs]
    dev_episodes = [_episode_from_report_unique(ref) for ref in dev_refs]

    train_manifest = export_episode_dataset(
        train_episodes,
        outputs["train_trajectory"],
        policy_id="gpt54_phase5_hard_trace_converter_v1",
        split="train",
    )
    dev_manifest = export_episode_dataset(
        dev_episodes,
        outputs["dev_trajectory"],
        policy_id="gpt54_phase5_hard_trace_converter_v1",
        split="dev",
    )
    train_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["train_trajectory"],
        outputs["train_sft"],
        split="train",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=True,
    )
    dev_sft_manifest = package_trajectory_dataset_as_sft(
        ROOT,
        outputs["dev_trajectory"],
        outputs["dev_sft"],
        split="dev",
        governance_allowlist=["usable_positive_sft", "usable_negative_transition"],
        patch_bearing_only=True,
    )

    training_assets_report = {
        "report_id": "phase5_hard_trace_training_assets_v1",
        "created_at": manifest["created_at"],
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
    (out_dir / "training_assets_report.json").write_text(json.dumps(training_assets_report, indent=2) + "\n", encoding="utf-8")
    _write_note(out_dir / "phase5_hard_trace_freeze_note.md", manifest)

    print(json.dumps({"manifest": manifest, "training_assets_report": training_assets_report}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
