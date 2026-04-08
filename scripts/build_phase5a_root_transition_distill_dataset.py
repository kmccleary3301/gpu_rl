from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from build_phase5_artifact_feedback_distill_dataset import (
    _count_examples_by,
    _example_slug,
    _prompt,
    _response_json,
)
from gpu_cockpit.contracts import SFTDatasetManifest, SFTExample


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


TRANCHE_TASKS = {
    "task/attention_score/eval/v1",
    "task/kernelbench/level1/23_softmax_wide/eval/v1",
    "task/routing_argmax_hard/eval/v1",
    "task/kernelbench/level1/47_sum_reduction/eval/v1",
}


def _select_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_ref = str(row.get("task_ref"))
        quality_bucket = str(row.get("quality_bucket"))
        split = str(row.get("split"))
        if quality_bucket == "analysis_only":
            continue
        if task_ref in TRANCHE_TASKS and quality_bucket == "usable_positive":
            train_rows.append(row)
            continue
        if task_ref == "task/kv_cache_gather_hard/eval/v1" and quality_bucket == "near_miss":
            train_rows.append(row)
            continue
        if task_ref == "task/kernelbench_v3/level1/23_softmax_official/eval/v1" and quality_bucket == "usable_positive":
            dev_rows.append(row)
            continue
        if task_ref == "task/routing_argmax_hard/eval/v1" and split == "dev" and quality_bucket == "usable_positive":
            dev_rows.append(row)
    return train_rows, dev_rows


def _turn_duplication(row: dict[str, Any], turn_index: int, selected_action: str, *, split: str) -> int:
    task_ref = str(row.get("task_ref"))
    if selected_action not in {"bench", "patch_candidate", "compare", "branch_candidate"}:
        return 0
    if turn_index == 0 and selected_action == "bench":
        return 1
    if turn_index == 1 and selected_action == "patch_candidate":
        if task_ref == "task/routing_argmax_hard/eval/v1":
            return 18 if split == "train" else 3
        if task_ref == "task/kv_cache_gather_hard/eval/v1":
            return 14 if split == "train" else 2
        return 12 if split == "train" else 2
    if turn_index == 2 and selected_action == "bench":
        return 4 if split == "train" else 1
    if turn_index == 3 and selected_action == "compare":
        return 6 if split == "train" else 1
    if task_ref == "task/kv_cache_gather_hard/eval/v1" and turn_index == 4 and selected_action == "branch_candidate":
        return 5 if split == "train" else 1
    if task_ref == "task/kv_cache_gather_hard/eval/v1" and turn_index == 5 and selected_action == "patch_candidate":
        return 5 if split == "train" else 1
    return 0


def _build_examples_for_row(row: dict[str, Any], *, split: str) -> list[SFTExample]:
    payload = _read_json(ROOT / str(row["episode_ref"]))
    turns = payload.get("model_turns", [])
    if not isinstance(turns, list):
        return []
    examples: list[SFTExample] = []
    for turn_index, turn in enumerate(turns):
        if not isinstance(turn, dict):
            continue
        selected_action = str(turn.get("selected_action", ""))
        dup_count = _turn_duplication(row, turn_index, selected_action, split=split)
        if dup_count <= 0:
            continue
        observation_packet = turn.get("observation_packet", {})
        if not isinstance(observation_packet, dict):
            observation_packet = {}
        focus_band = "root_transition" if turn_index <= 1 else "early_compare"
        for dup_index in range(dup_count):
            examples.append(
                SFTExample(
                    example_id=f"phase5a_root_transition__{_example_slug(str(row['episode_ref']))}__{turn_index}__{dup_index}",
                    created_at=datetime.now(tz=UTC),
                    split=split,
                    task_id=str(row["task_ref"]),
                    prompt_family="optimize_action_json",
                    prompt=_prompt(observation_packet, row),
                    response=_response_json(turn),
                    source_episode_ref=str(row["episode_ref"]),
                    metadata={
                        "training_example_kind": "root_transition_distill",
                        "quality_bucket": row["quality_bucket"],
                        "slice_role": row["slice_role"],
                        "source_block": row.get("source_block", "phase5a_root_transition"),
                        "selected_action": selected_action,
                        "model": turn.get("model"),
                        "provider": turn.get("provider"),
                        "turn_index": turn_index,
                        "duplication_index": dup_index,
                        "focus_band": focus_band,
                    },
                )
            )
    return examples


def _write_dataset(
    *,
    rows: list[dict[str, Any]],
    split: str,
    out_dir: Path,
    source_manifest_path: Path,
) -> SFTDatasetManifest:
    out_dir.mkdir(parents=True, exist_ok=True)
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)
    examples: list[SFTExample] = []
    for row in rows:
        examples.extend(_build_examples_for_row(row, split=split))
    example_refs: list[str] = []
    for example in examples:
        example_path = examples_dir / f"{example.example_id}.json"
        example_path.write_text(json.dumps(example.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
        example_refs.append(str(example_path.relative_to(out_dir)))
    manifest = SFTDatasetManifest(
        dataset_id=f"phase5a_root_transition_distill_{split}_v1",
        created_at=datetime.now(tz=UTC),
        split=split,
        example_count=len(example_refs),
        example_refs=example_refs,
        task_ids=sorted({example.task_id for example in examples}),
        metadata={
            "source_freeze_manifest": str(source_manifest_path.relative_to(ROOT)),
            "quality_counts": _count_examples_by(examples, "quality_bucket"),
            "slice_role_counts": _count_examples_by(examples, "slice_role"),
            "action_counts": _count_examples_by(examples, "selected_action"),
            "focus_band_counts": _count_examples_by(examples, "focus_band"),
        },
    )
    (out_dir / "sft_dataset_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5a_hard_trace_freeze_v2",
    )
    parser.add_argument(
        "--train-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5a_root_transition_distill_train_v1",
    )
    parser.add_argument(
        "--dev-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5a_root_transition_distill_dev_v1",
    )
    args = parser.parse_args()

    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    report_path = args.freeze_dir / "root_transition_distill_dataset_report.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid manifest rows in {manifest_path}")

    train_rows, dev_rows = _select_rows(rows)
    train_manifest = _write_dataset(
        rows=train_rows,
        split="train",
        out_dir=args.train_out_dir,
        source_manifest_path=manifest_path,
    )
    dev_manifest = _write_dataset(
        rows=dev_rows,
        split="dev",
        out_dir=args.dev_out_dir,
        source_manifest_path=manifest_path,
    )

    train_task_counts = Counter(str(row.get("task_ref")) for row in train_rows)
    dev_task_counts = Counter(str(row.get("task_ref")) for row in dev_rows)
    report = {
        "report_id": f"{args.freeze_dir.name}_root_transition_distill_dataset",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "train_manifest": str((args.train_out_dir / "sft_dataset_manifest.json").relative_to(ROOT)),
        "dev_manifest": str((args.dev_out_dir / "sft_dataset_manifest.json").relative_to(ROOT)),
        "train_example_count": train_manifest.example_count,
        "dev_example_count": dev_manifest.example_count,
        "train_task_ids": train_manifest.task_ids,
        "dev_task_ids": dev_manifest.task_ids,
        "train_task_row_counts": dict(sorted(train_task_counts.items())),
        "dev_task_row_counts": dict(sorted(dev_task_counts.items())),
        "notes": [
            "Focused early-turn distillation dataset for the Phase 5A tranche.",
            "Overweights the root transition and early compare loop instead of full trajectories.",
            "Primary hypothesis: broader Phase 5A transfer is blocked by second-turn action selection on routing_argmax_hard rather than by general curriculum coverage alone.",
            "Includes kv_cache_gather_hard near-miss early turns as supportive branch-bearing signal.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
