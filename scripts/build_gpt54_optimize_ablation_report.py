from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

ABLATION_RUNS = [
    {
        "label": "attention_v2_patch_then_compare_optional",
        "episode_ref": "artifacts/baselines/gpt54_attention_score_bounded_patch_probe_v1/batch_v2_retry1/task__attention_score__eval__v1__positive.json",
        "interface_delta": "bounded patch lane active; compare not yet forced",
    },
    {
        "label": "attention_v4_compare_forced",
        "episode_ref": "artifacts/baselines/gpt54_attention_score_bounded_patch_probe_v1/batch_v4_compare_forced_retry1/task__attention_score__eval__v1__positive.json",
        "interface_delta": "compare forced; localized failure evidence active",
    },
    {
        "label": "attention_v6_eval_closeout",
        "episode_ref": "artifacts/baselines/gpt54_attention_score_bounded_patch_probe_v1/batch_v6_forced_eval_closeout_retry1/task__attention_score__eval__v1__positive.json",
        "interface_delta": "compare digest + eval-only closeout after compare",
    },
    {
        "label": "kernelbench_softmax_public_multi_candidate_negative",
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
        "interface_delta": "public branch/revert negative lane enabled",
    },
    {
        "label": "row_sum_two_attempt_positive",
        "episode_ref": "artifacts/baselines/gpt54_reduction_row_sum_two_attempt_positive_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json",
        "interface_delta": "true two-attempt bounded optimize loop with candidate A and candidate B on an internal task",
    },
    {
        "label": "kernelbench_softmax_wide_two_attempt_positive",
        "episode_ref": "artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
        "interface_delta": "harder public two-attempt loop with candidate A/B, ending as a structured near-miss",
    },
]


def _load_episode(relative_ref: str) -> dict[str, object]:
    return json.loads((ROOT / relative_ref).read_text(encoding="utf-8"))


def main() -> int:
    out_dir = ROOT / "artifacts" / "baselines" / "gpt54_optimize_ablation_v2"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    for item in ABLATION_RUNS:
        payload = _load_episode(item["episode_ref"])
        counters = payload.get("counters", {})
        if not isinstance(counters, dict):
            counters = {}
        rows.append(
            {
                "label": item["label"],
                "episode_ref": item["episode_ref"],
                "interface_delta": item["interface_delta"],
                "task_ref": payload.get("task_ref"),
                "variant": payload.get("variant"),
                "success": payload.get("success"),
                "terminal_reason": payload.get("terminal_reason"),
                "step_count": payload.get("step_count"),
                "patches": counters.get("patches", 0),
                "compares": counters.get("compares", 0),
                "branches": counters.get("branches", 0),
                "reverts": counters.get("reverts", 0),
                "controller_rejections": counters.get("controller_rejections", 0),
                "failed_tool_calls": counters.get("failed_tool_calls", 0),
            }
        )

    report = {
        "report_id": "gpt54_optimize_ablation_v2",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "rows": rows,
        "summary": {
            "attention_progression": [
                "v2 proved bounded patching but still allowed low-value tail behavior.",
                "v4 forced compare and produced the intended compare-native loop.",
                "v6 tightened post-compare closeout and reduced the loop to bench -> patch -> bench -> compare -> eval.",
                "The internal two-attempt row-sum lane solved with two real patch attempts, two compares, branch, promote, and final eval.",
                "The public softmax-wide two-attempt lane reached a structured near-miss, which is useful evidence that the harder interface is now fair even when unsolved.",
            ],
            "public_branch_result": "KernelBench public lanes now cover negative branch/revert traces plus a harder positive two-attempt near-miss.",
        },
    }
    note = "\n".join(
        [
            "# GPT-5.4 Optimize Ablation Note",
            "",
            "Key result: the tooling changes moved the model from a loosely bounded optimize loop into a cleaner candidate-engineering loop.",
            "",
            "- Attention v2 showed bounded patching but still left too much room for low-value tail actions.",
            "- Attention v4 made compare habitual by forcing it in the controller path.",
            "- Attention v6 made compare actionable by ensuring the loop closed on eval instead of drifting.",
            "- Public KernelBench softmax now supports a real branch/revert usable-negative trace, which means the multi-candidate lifecycle is no longer internal-only.",
            "- Internal row-sum now supports a true two-attempt positive loop where candidate B is only available after branching off candidate A.",
            "- Public softmax wide now supports the same two-attempt interface and yields a useful near-miss instead of an opaque failure.",
            "",
            "Interpretation: the main gains still come from tool-surface and controller design. The new two-attempt interface increases trace value even when the task remains unsolved.",
        ]
    )

    (out_dir / "ablation_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "ablation_note.md").write_text(note + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
