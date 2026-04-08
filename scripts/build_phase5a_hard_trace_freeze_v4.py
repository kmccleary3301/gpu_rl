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
from scripts.build_phase5a_hard_trace_freeze import (
    _aggregate_counters,
    _contamination_audit,
    _count_by,
    _coverage_report,
    _entry,
    _episode_from_report_unique,
)
from scripts.build_phase5a_hard_trace_freeze_v2 import _readiness_report


FREEZE_ROWS_V4: list[dict[str, Any]] = [
    {
        "episode_ref": "artifacts/baselines/gpt54_kv_cache_gather_hard_three_attempt_positive_probe_v2/batch_v1_openai/task__kv_cache_gather_hard__eval__v1__positive.json",
        "split": "train",
        "quality_bucket": "usable_positive",
        "slice_role": "harder_kv_positive",
        "source_block": "phase5a_kv_hard_budget_rerun",
        "interface_profile": "compare_plus_localization_plus_branch_v1",
        "provenance_kind": "internal_curated",
        "trainability": "trainable",
    },
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_tranche1_eval_distill_v5_v1/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "learner_repair_comparison",
        "source_block": "phase5a_sum_reduction_repair_turn",
        "interface_profile": "local_eval_controlfix_v2",
        "provenance_kind": "local_control_surface",
        "trainability": "analysis_only",
    },
    {
        "episode_ref": "artifacts/baselines/qwen7b_phase5a_tranche1_eval_distill_v5_v1/task__attention_score__eval__v1__positive.json",
        "split": "analysis",
        "quality_bucket": "analysis_only",
        "slice_role": "learner_repair_comparison",
        "source_block": "phase5a_sum_reduction_repair_turn",
        "interface_profile": "local_eval_controlfix_v2",
        "provenance_kind": "local_control_surface",
        "trainability": "analysis_only",
    },
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_note(out_path: Path, manifest: dict[str, Any], contamination: dict[str, Any]) -> None:
    lines = [
        "# Phase 5A Hard Trace Freeze v4",
        "",
        "## Purpose",
        "",
        "Refresh the canonical Phase 5A slice after the harder KV-budget rerun succeeded and the narrow learner repair turn was completed.",
        "",
        "This slice is intended to support:",
        "",
        "- the next artifact-distill refresh with a real harder KV success in the trainable tranche",
        "- continued teacher-correction and ranking exports off the newest shared freeze",
        "- comparison against the narrow sum-reduction repair turn without treating that learner trace as a canonical teacher win",
        "",
        "## Composition",
        "",
        f"- dataset id: `{manifest['dataset_id']}`",
        f"- episode count: `{manifest['episode_count']}`",
        f"- train count: `{manifest['split_counts'].get('train', 0)}`",
        f"- dev count: `{manifest['split_counts'].get('dev', 0)}`",
        f"- analysis count: `{manifest['split_counts'].get('analysis', 0)}`",
        f"- usable positives: `{manifest['quality_counts'].get('usable_positive', 0)}`",
        f"- contamination audit status: `{contamination['status']}`",
        "",
        "## What changed from v3",
        "",
        "- adds the successful `gpt54_kv_cache_gather_hard_three_attempt_positive_probe_v2` episode into the trainable hard tranche",
        "- preserves the broader control-surface winner rows from freeze v3",
        "- adds the narrow `distill_v5` repair outcomes as analysis-only comparison rows",
        "- keeps the previous GPT-5.4 tranche, KB-v3, and near-miss rows intact",
    ]
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    registry = TaskRegistry(ROOT)
    base_manifest_path = ROOT / "artifacts" / "training" / "phase5a_hard_trace_freeze_v3" / "optimize_trace_manifest.json"
    base_manifest = _read_json(base_manifest_path)
    base_entries = base_manifest.get("episodes", [])
    if not isinstance(base_entries, list):
        raise SystemExit(f"Invalid episode rows in {base_manifest_path}")

    out_dir = ROOT / "artifacts" / "training" / "phase5a_hard_trace_freeze_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = [row for row in base_entries if isinstance(row, dict)]
    entries.extend(_entry(spec, registry) for spec in FREEZE_ROWS_V4)

    created_at = datetime.now(tz=UTC).isoformat()
    manifest = {
        "dataset_id": "phase5a_hard_trace_freeze_v4",
        "created_at": created_at,
        "schema_version": "2.1.0",
        "objective": "Refresh the canonical Phase 5A corpus after the successful harder KV rerun and the narrow learner repair comparison.",
        "episode_count": len(entries),
        "split_counts": _count_by(entries, "split"),
        "quality_counts": _count_by(entries, "quality_bucket"),
        "task_counts": _count_by(entries, "task_ref"),
        "task_scope_counts": _count_by(entries, "task_scope"),
        "difficulty_counts": _count_by(entries, "difficulty"),
        "candidate_mode_counts": _count_by(entries, "candidate_mode"),
        "slice_role_counts": _count_by(entries, "slice_role"),
        "source_block_counts": _count_by(entries, "source_block"),
        "aggregate_counters": _aggregate_counters(entries),
        "episodes": entries,
        "notes": [
            "Canonical Phase 5A slice refreshed after the harder KV-cache gather rerun converted the prior near miss into a real GPT-5.4 success.",
            "The trainable tranche now includes both the broader local control winner rows and the successful harder KV rerun.",
            "The narrow sum-reduction repair learner turn is preserved as analysis-only comparison material rather than treated as the new canonical learner winner.",
        ],
    }

    contamination = _contamination_audit(entries, manifest["dataset_id"], created_at)
    coverage_report = _coverage_report(manifest, entries)
    readiness_report = _readiness_report(manifest, entries)

    outputs = {
        "train_trajectory": ROOT / "datasets" / "phase5a_hard_trace_transition_train_v4",
        "dev_trajectory": ROOT / "datasets" / "phase5a_hard_trace_transition_dev_v4",
        "train_sft": ROOT / "datasets" / "phase5a_hard_trace_sft_train_v4",
        "dev_sft": ROOT / "datasets" / "phase5a_hard_trace_sft_dev_v4",
    }
    train_refs = [row["episode_ref"] for row in entries if row["split"] == "train"]
    dev_refs = [row["episode_ref"] for row in entries if row["split"] == "dev"]
    train_episodes = [_episode_from_report_unique(ref) for ref in train_refs]
    dev_episodes = [_episode_from_report_unique(ref) for ref in dev_refs]

    train_manifest = export_episode_dataset(
        train_episodes,
        outputs["train_trajectory"],
        policy_id="phase5a_hard_trace_converter_v4",
        split="train",
    )
    dev_manifest = export_episode_dataset(
        dev_episodes,
        outputs["dev_trajectory"],
        policy_id="phase5a_hard_trace_converter_v4",
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
        "report_id": "phase5a_hard_trace_training_assets_v4",
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
