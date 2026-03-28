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
from scripts.build_phase5a_hard_trace_freeze import (
    _aggregate_counters,
    _contamination_audit,
    _count_by,
    _coverage_report,
    _entry,
    _episode_from_report_unique,
)


CURATED_PHASE5A_SLICE_V2: list[dict[str, Any]] = [
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5a_tranche1_hard_targets_v2/batch_v2_openai_rejection_trim/task__attention_score__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "internal_hard_positive",
        "source_block": "phase5a_tranche1",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
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
        "episode_ref": "artifacts/baselines/gpt54_phase5a_tranche1_hard_targets_v2/batch_v2_openai_rejection_trim/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "public_hard_positive",
        "source_block": "phase5a_tranche1",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "near_miss",
        "slice_role": "public_hard_near_miss",
        "source_block": "phase4_final_teacher",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_routing_argmax_hard_three_attempt_positive_probe_v1/batch_v5_openai/task__routing_argmax_hard__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "routing_hard_positive",
        "source_block": "phase5a_tranche1",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_kv_cache_gather_hard_three_attempt_positive_probe_v1/batch_v1_openai/task__kv_cache_gather_hard__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "near_miss",
        "slice_role": "kv_hard_near_miss",
        "source_block": "phase5a_tranche1_extension",
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
        "episode_ref": "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "split": "train",
        "quality_bucket": "usable_negative",
        "slice_role": "branch_revert_negative",
        "source_block": "phase4_final_teacher",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5a_tranche1_hard_targets_v2/batch_v1_openai/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "usable_positive",
        "slice_role": "control_positive",
        "source_block": "phase5a_tranche1",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "kernelbench_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/gpt54_phase5a_tranche1_hard_targets_v2/batch_v1_openai/task__routing_argmax_hard__eval__v1__positive.json",
        "split": "dev",
        "quality_bucket": "usable_positive",
        "slice_role": "routing_hard_positive",
        "source_block": "phase5a_tranche1",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
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
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_hard_pair_eval_base_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "base_hard_pair_analysis",
        "source_block": "phase5a_learning_trio",
        "interface_profile": "local_eval_hard_pair_v1",
        "provenance_kind": "analysis_eval",
        "trainability": "analysis_only",
    },
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_hard_pair_eval_distill_v2_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "distill_hard_pair_analysis",
        "source_block": "phase5a_learning_trio",
        "interface_profile": "local_eval_hard_pair_v1",
        "provenance_kind": "analysis_eval",
        "trainability": "analysis_only",
    },
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_hard_pair_eval_teacher_corrected_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "teacher_corrected_hard_pair_analysis",
        "source_block": "phase5a_learning_trio",
        "interface_profile": "local_eval_hard_pair_v1",
        "provenance_kind": "analysis_eval",
        "trainability": "analysis_only",
    },
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_hard_pair_eval_rwr_branchaware_v1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "rwr_hard_pair_analysis",
        "source_block": "phase5a_learning_trio",
        "interface_profile": "local_eval_hard_pair_v1",
        "provenance_kind": "analysis_eval",
        "trainability": "analysis_only",
    },
]


def _readiness_report(manifest: dict[str, Any], entries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "checks": {
            "has_internal_hard_positive": any(row["slice_role"] == "internal_hard_positive" for row in entries),
            "has_public_hard_positive": any(row["slice_role"] == "public_hard_positive" for row in entries),
            "has_routing_hard_positive": any(row["slice_role"] == "routing_hard_positive" for row in entries),
            "has_kv_hard_near_miss": any(row["slice_role"] == "kv_hard_near_miss" for row in entries),
            "has_kbv3_curated_positive": any(row["slice_role"] == "kbv3_curated_positive" for row in entries),
            "has_kbv3_official_positive": any(row["slice_role"] == "kbv3_official_positive" for row in entries),
            "has_near_miss": any(row["quality_bucket"] == "near_miss" for row in entries),
            "has_usable_negative": any(row["quality_bucket"] == "usable_negative" for row in entries),
            "has_learning_trio_analysis_rows": any(row["split"] == "analysis" for row in entries),
            "held_out_entries_confined_to_dev": all(
                row["split"] != "train" for row in entries if row.get("trainability") == "held_out"
            ),
        },
        "recommended_next_use": {
            "train_refs": [row["episode_ref"] for row in entries if row["split"] == "train"],
            "dev_refs": [row["episode_ref"] for row in entries if row["split"] == "dev"],
            "analysis_refs": [row["episode_ref"] for row in entries if row["split"] == "analysis"],
        },
    }


def _write_note(out_path: Path, manifest: dict[str, Any], contamination: dict[str, Any]) -> None:
    lines = [
        "# Phase 5A Hard Trace Freeze v2",
        "",
        "## Purpose",
        "",
        "Freeze the post-RL-comparison Phase 5A hard slice with the first harder KV successor result and explicit learner-eval analysis rows.",
        "",
        "This slice is intended to support:",
        "",
        "- the next post-Phase-5A artifact-distill refresh",
        "- the next teacher-corrected refresh",
        "- pairwise ranking refresh on the harder tranche",
        "- tranche decision-making using shared analysis-only learner rows",
        "",
        "## Composition",
        "",
        f"- dataset id: `{manifest['dataset_id']}`",
        f"- episode count: `{manifest['episode_count']}`",
        f"- train count: `{manifest['split_counts'].get('train', 0)}`",
        f"- dev count: `{manifest['split_counts'].get('dev', 0)}`",
        f"- analysis count: `{manifest['split_counts'].get('analysis', 0)}`",
        f"- usable positives: `{manifest['quality_counts'].get('usable_positive', 0)}`",
        f"- usable negatives: `{manifest['quality_counts'].get('usable_negative', 0)}`",
        f"- near misses: `{manifest['quality_counts'].get('near_miss', 0)}`",
        "",
        "## What changed from v1",
        "",
        "- adds the new `kv_cache_gather_hard` GPT-5.4 near-miss result",
        "- carries forward the validated 4/4 tranche positives and routing hard positive",
        "- keeps KB-v3 curated and official provenance explicit",
        "- records the current hard-pair local learner outcomes as analysis-only rows",
        "",
        "## Provenance guarantees",
        "",
        "- official-vs-curated benchmark provenance is recorded on every row",
        "- held-out-vs-trainable status is recorded on every row",
        "- analysis-only learner rows are excluded from train/dev exports",
        "- official KB-v3 held-out traces remain outside the train split",
        f"- contamination audit status: `{contamination['status']}`",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    registry = TaskRegistry(ROOT)
    out_dir = ROOT / "artifacts" / "training" / "phase5a_hard_trace_freeze_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [_entry(spec, registry) for spec in CURATED_PHASE5A_SLICE_V2]
    created_at = datetime.now(tz=UTC).isoformat()
    aggregate_counters = _aggregate_counters(entries)

    manifest = {
        "dataset_id": "phase5a_hard_trace_freeze_v2",
        "created_at": created_at,
        "schema_version": "2.1.0",
        "objective": "Freeze the post-learning-comparison Phase 5A hard-positive / near-miss / branch-aware corpus with the first harder KV successor and explicit analysis-only learner rows.",
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
            "Canonical Phase 5A hard slice refreshed after the first narrow RL comparison and the first harder KV successor probe.",
            "Analysis-only rows preserve current hard-pair learner outcomes without contaminating train/dev exports.",
            "Aligned source freeze for the next distill, teacher-correction, and ranking refreshes.",
        ],
    }

    contamination = _contamination_audit(entries, manifest["dataset_id"], created_at)
    coverage_report = _coverage_report(manifest, entries)
    readiness_report = _readiness_report(manifest, entries)

    outputs = {
        "train_trajectory": ROOT / "datasets" / "phase5a_hard_trace_transition_train_v2",
        "dev_trajectory": ROOT / "datasets" / "phase5a_hard_trace_transition_dev_v2",
        "train_sft": ROOT / "datasets" / "phase5a_hard_trace_sft_train_v2",
        "dev_sft": ROOT / "datasets" / "phase5a_hard_trace_sft_dev_v2",
    }
    train_refs = [row["episode_ref"] for row in entries if row["split"] == "train"]
    dev_refs = [row["episode_ref"] for row in entries if row["split"] == "dev"]
    train_episodes = [_episode_from_report_unique(ref) for ref in train_refs]
    dev_episodes = [_episode_from_report_unique(ref) for ref in dev_refs]

    train_manifest = export_episode_dataset(
        train_episodes,
        outputs["train_trajectory"],
        policy_id="gpt54_phase5a_hard_trace_converter_v2",
        split="train",
    )
    dev_manifest = export_episode_dataset(
        dev_episodes,
        outputs["dev_trajectory"],
        policy_id="gpt54_phase5a_hard_trace_converter_v2",
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
        "report_id": "phase5a_hard_trace_training_assets_v2",
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
    _write_note(out_dir / "phase5a_hard_trace_freeze_note.md", manifest, contamination)

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
