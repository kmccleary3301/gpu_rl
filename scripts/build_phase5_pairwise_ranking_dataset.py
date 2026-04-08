from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


QUALITY_SCORE = {
    "usable_positive": 3,
    "near_miss": 2,
    "usable_negative": 1,
}


def _row_score(row: dict[str, Any]) -> int:
    return QUALITY_SCORE.get(str(row.get("quality_bucket")), 0)


def _group_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("task_ref")), str(row.get("provenance_kind"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5_hard_trace_freeze_v2",
    )
    args = parser.parse_args()

    args.freeze_dir = args.freeze_dir.resolve()
    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    out_path = args.freeze_dir / "pairwise_ranking_dataset_report.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid manifest rows in {manifest_path}")

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        grouped.setdefault(_group_key(row), []).append(row)

    pairs: list[dict[str, Any]] = []
    for group_rows in grouped.values():
        ranked = sorted(group_rows, key=lambda row: (_row_score(row), int(row.get("branch_depth", 0))), reverse=True)
        for idx, preferred in enumerate(ranked):
            for rejected in ranked[idx + 1 :]:
                preferred_score = _row_score(preferred)
                rejected_score = _row_score(rejected)
                if preferred_score <= rejected_score:
                    continue
                pairs.append(
                    {
                        "task_ref": preferred["task_ref"],
                        "provenance_kind": preferred["provenance_kind"],
                        "preferred_episode_ref": preferred["episode_ref"],
                        "rejected_episode_ref": rejected["episode_ref"],
                        "preferred_quality_bucket": preferred["quality_bucket"],
                        "rejected_quality_bucket": rejected["quality_bucket"],
                        "preferred_branch_depth": preferred.get("branch_depth"),
                        "rejected_branch_depth": rejected.get("branch_depth"),
                        "ranking_reason": f"{preferred['quality_bucket']}_over_{rejected['quality_bucket']}",
                    }
                )

    report = {
        "report_id": f"{args.freeze_dir.name}_pairwise_ranking_dataset",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "pair_count": len(pairs),
        "task_group_count": len(grouped),
        "pairs": pairs,
        "notes": [
            "Lightweight pairwise ranking export built from the canonical Phase 5 hard freeze.",
            "Pairs prefer usable positives over near-misses and near-misses over usable negatives within matching task/provenance groups.",
            "This is a ranking substrate view, not a claim that preference training is already complete.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
