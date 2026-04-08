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

from build_phase5_artifact_feedback_distill_dataset import _write_dataset


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


TRANCHE_TASKS = {
    "task/attention_score/eval/v1",
    "task/kernelbench/level1/23_softmax_wide/eval/v1",
    "task/routing_argmax_hard/eval/v1",
    "task/kernelbench/level1/47_sum_reduction/eval/v1",
}


def _row_copies(row: dict[str, Any]) -> int:
    task_ref = str(row.get("task_ref"))
    quality_bucket = str(row.get("quality_bucket"))
    slice_role = str(row.get("slice_role"))
    split = str(row.get("split"))
    if task_ref == "task/routing_argmax_hard/eval/v1" and quality_bucket == "usable_positive":
        return 3 if split == "train" else 2
    if task_ref == "task/kernelbench/level1/47_sum_reduction/eval/v1" and quality_bucket == "usable_positive":
        return 3
    if quality_bucket == "usable_positive":
        return 2
    if quality_bucket == "near_miss" and "hard" in slice_role:
        return 2
    return 1


def _select_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows: list[dict[str, Any]] = []
    dev_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        task_ref = str(row.get("task_ref"))
        split = str(row.get("split"))
        quality_bucket = str(row.get("quality_bucket"))
        if quality_bucket == "analysis_only":
            continue
        if task_ref in TRANCHE_TASKS and quality_bucket in {"usable_positive", "near_miss"}:
            train_rows.extend([row] * _row_copies(row))
            continue
        if task_ref == "task/kv_cache_gather_hard/eval/v1" and quality_bucket == "near_miss":
            dev_rows.append(row)
            continue
        if task_ref == "task/kernelbench_v3/level1/23_softmax_official/eval/v1" and quality_bucket == "usable_positive":
            dev_rows.append(row)
            continue
    return train_rows, dev_rows


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
        default=ROOT / "datasets" / "phase5a_tranche1_transfer_distill_train_v1",
    )
    parser.add_argument(
        "--dev-out-dir",
        type=Path,
        default=ROOT / "datasets" / "phase5a_tranche1_transfer_distill_dev_v1",
    )
    args = parser.parse_args()

    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    report_path = args.freeze_dir / "tranche1_transfer_distill_dataset_report.json"
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
        "report_id": f"{args.freeze_dir.name}_tranche1_transfer_distill_dataset",
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
            "Focused Phase 5A tranche-transfer artifact distill dataset.",
            "Promotes the four-task tranche targets directly into train instead of leaving the control positive in dev.",
            "Adds extra weight to routing_argmax_hard and sum_reduction positives to test whether the current transfer failure is mostly a curriculum/split problem.",
            "Keeps KB-v3 official softmax and kv_cache_gather_hard near-miss as a small out-of-tranche dev surface.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
