from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze-dir",
        type=Path,
        default=ROOT / "artifacts" / "training" / "phase5_hard_trace_freeze_v2",
    )
    args = parser.parse_args()

    manifest_path = args.freeze_dir / "optimize_trace_manifest.json"
    out_path = args.freeze_dir / "reward_audit_report.json"
    manifest = _read_json(manifest_path)
    rows = manifest.get("episodes", [])
    if not isinstance(rows, list):
        raise SystemExit(f"Invalid manifest rows in {manifest_path}")

    compare_without_patch: list[str] = []
    near_miss_without_compare: list[str] = []
    protocol_bonus_risk: list[str] = []
    shaped_counts = {
        "compare_bearing": 0,
        "branch_bearing": 0,
        "revert_bearing": 0,
        "promote_bearing": 0,
        "reference_only": 0,
    }
    sparse_counts = {
        "usable_positive": 0,
        "near_miss": 0,
        "usable_negative": 0,
    }

    for row in rows:
        if not isinstance(row, dict):
            continue
        counters = row.get("counters", {})
        if not isinstance(counters, dict):
            counters = {}
        episode_ref = str(row.get("episode_ref"))
        quality_bucket = str(row.get("quality_bucket"))
        sparse_counts[quality_bucket] = sparse_counts.get(quality_bucket, 0) + 1
        if int(counters.get("compares", 0)) > 0:
            shaped_counts["compare_bearing"] += 1
        if int(counters.get("branches", 0)) > 0:
            shaped_counts["branch_bearing"] += 1
        if int(counters.get("reverts", 0)) > 0:
            shaped_counts["revert_bearing"] += 1
        if int(counters.get("promotes", 0)) > 0:
            shaped_counts["promote_bearing"] += 1
        if str(row.get("candidate_mode")) == "reference_only":
            shaped_counts["reference_only"] += 1
        if int(counters.get("compares", 0)) > 0 and int(counters.get("patches", 0)) == 0:
            compare_without_patch.append(episode_ref)
        if quality_bucket == "near_miss" and int(counters.get("compares", 0)) == 0:
            near_miss_without_compare.append(episode_ref)
        if str(row.get("candidate_mode")) == "reference_only" and quality_bucket != "usable_positive":
            protocol_bonus_risk.append(episode_ref)

    report = {
        "report_id": f"{args.freeze_dir.name}_reward_audit",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "source_manifest": str(manifest_path.relative_to(ROOT)),
        "sparse_outcome_counts": sparse_counts,
        "shaped_signal_counts": shaped_counts,
        "checks": {
            "compare_without_patch_count": len(compare_without_patch),
            "near_miss_without_compare_count": len(near_miss_without_compare),
            "protocol_bonus_risk_count": len(protocol_bonus_risk),
        },
        "violations": {
            "compare_without_patch_refs": compare_without_patch,
            "near_miss_without_compare_refs": near_miss_without_compare,
            "protocol_bonus_risk_refs": protocol_bonus_risk,
        },
        "notes": [
            "This is a substrate audit, not a learned-policy evaluation.",
            "Sparse counts summarize terminal outcome classes in the freeze.",
            "Shaped counts summarize where compare/branch/revert/promote evidence exists and therefore where shaping can be meaningfully grounded.",
        ],
    }
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
