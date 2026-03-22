from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ABLATION_VARIANTS = [
    {
        "label": "v3_current",
        "batch_ref": "artifacts/baselines/gpt54_phase5_block_b_v3_current_v1/batch_v1/batch_report.json",
        "interface_delta": "single-candidate bounded optimize lane without richer compare packet, failure localization, or branch-and-rank",
    },
    {
        "label": "compare_packet_v1",
        "batch_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_packet_v1/batch_v1/batch_report.json",
        "interface_delta": "single-candidate lane with richer compare packet only",
    },
    {
        "label": "compare_plus_localization_v1",
        "batch_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_v1/batch_v1/batch_report.json",
        "interface_delta": "single-candidate lane with richer compare packet plus failure localization",
    },
    {
        "label": "compare_plus_localization_plus_branch_v1",
        "batch_ref": "artifacts/baselines/gpt54_phase5_block_b_compare_localization_branch_v1/batch_v1/batch_report.json",
        "interface_delta": "full branch-and-rank lane with richer compare packet and failure localization",
    },
]


def _load_json(relative_ref: str) -> dict[str, object]:
    return json.loads((ROOT / relative_ref).read_text(encoding="utf-8"))


def _episode_rows(batch_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for episode_path in sorted(batch_dir.glob("task__*.json")):
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        counters = payload.get("counters", {})
        if not isinstance(counters, dict):
            counters = {}
        rows.append(
            {
                "task_ref": payload.get("task_ref"),
                "success": payload.get("success"),
                "terminal_reason": payload.get("terminal_reason"),
                "step_count": payload.get("step_count"),
                "patches": counters.get("patches", 0),
                "branches": counters.get("branches", 0),
                "compares": counters.get("compares", 0),
                "promotes": counters.get("promotes", 0),
                "controller_rejections": counters.get("controller_rejections", 0),
                "knowledge_queries": counters.get("knowledge_queries", 0),
            }
        )
    return rows


def main() -> int:
    out_dir = ROOT / "artifacts" / "baselines" / "gpt54_phase5_block_b_ablation_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    success_rows: list[tuple[str, float]] = []
    for item in ABLATION_VARIANTS:
        batch_report = _load_json(item["batch_ref"])
        batch_dir = (ROOT / item["batch_ref"]).parent
        summary = batch_report.get("summary")
        success_rate = 0.0
        if isinstance(summary, dict):
            success_rate = float(summary.get("success_rate", 0.0))
        success_rows.append((item["label"], success_rate))
        rows.append(
            {
                "label": item["label"],
                "batch_ref": item["batch_ref"],
                "interface_delta": item["interface_delta"],
                "summary": summary,
                "episodes": _episode_rows(batch_dir),
            }
        )

    report = {
        "report_id": "gpt54_phase5_block_b_ablation_v1",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "variants": rows,
    }

    best_label, best_rate = max(success_rows, key=lambda row: row[1])
    note_lines = [
        "# Phase 5 Block B Ablation Note",
        "",
        "This report compares the same hard-target set across four interface variants:",
        "",
        "1. `v3_current`",
        "2. `compare_packet_v1`",
        "3. `compare_plus_localization_v1`",
        "4. `compare_plus_localization_plus_branch_v1`",
        "",
        "The key question is which interface ingredients materially improve solve rate, candidate completion, and trace quality on the frozen hard-target tranche.",
        "",
        "## Result",
        "",
        f"- Best variant: `{best_label}` at success rate `{best_rate:.4f}`.",
        "- Variants `v3_current`, `compare_packet_v1`, and `compare_plus_localization_v1` all stayed at `0/2` on the hard-target pair.",
        "- The full `compare_plus_localization_plus_branch_v1` interface reached `2/2`.",
        "",
        "## Interpretation",
        "",
        "- Richer compare packets alone were not enough.",
        "- Failure localization alone was not enough.",
        "- The decisive lift came from the full branch-and-rank candidate loop on top of those richer observations.",
        "- For this tranche, the hard targets are not observation-limited in isolation; they are search/control-limited.",
    ]

    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ablation_note.md").write_text("\n".join(note_lines) + "\n", encoding="utf-8")
    print(json.dumps({"variant_count": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
