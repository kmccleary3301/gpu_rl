from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _episode_files(batch_dir: Path) -> list[Path]:
    return sorted(path for path in batch_dir.glob("*.json") if path.name != "batch_report.json")


def _load_episodes(batch_dirs: list[Path]) -> list[dict[str, Any]]:
    episodes: list[dict[str, Any]] = []
    for batch_dir in batch_dirs:
        for path in _episode_files(batch_dir):
            payload = _read_json(path)
            payload["_report_path"] = str(path.relative_to(ROOT))
            episodes.append(payload)
    return episodes


def _usage_totals(episode: dict[str, Any]) -> dict[str, int]:
    totals = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    for turn in episode.get("model_turns", []):
        usage = turn.get("usage", {})
        for key in totals:
            value = usage.get(key, 0)
            if isinstance(value, int):
                totals[key] += value
    return totals


def _task_aligned_outcome(episode: dict[str, Any]) -> tuple[str, list[str]]:
    task_ref = str(episode.get("task_ref"))
    variant = str(episode.get("variant"))
    success = bool(episode.get("success"))
    counters = episode.get("counters", {})
    steps = episode.get("steps", [])
    reasons: list[str] = []

    if task_ref == "task/reduction_debug/eval/v1":
        if variant == "positive":
            if success and int(counters.get("patches", 0)) >= 1:
                return "aligned", ["repair_trace_completed"]
            reasons.append("positive_debug_expected_patch_bearing_repair")
            return "misaligned", reasons
        if variant == "negative":
            if not success and int(counters.get("patches", 0)) >= 1:
                return "aligned", ["usable_negative_debug_trace_retained"]
            reasons.append("negative_debug_expected_failed_post_patch_eval")
            return "misaligned", reasons

    if task_ref == "task/profile_diagnose/eval/v1":
        if success:
            return "aligned", ["diagnostic_eval_completed"]
        return "misaligned", ["profile_diagnose_should_complete_via_eval"]

    if task_ref == "task/attention_reformulate/eval/v1":
        if success and int(counters.get("patches", 0)) == 0:
            first_action = str(steps[0].get("action_name")) if steps else "none"
            if first_action == "eval":
                return "questionable", ["eval_surface_accepts_weak_baseline_without_reformulation"]
        if variant == "positive" and success and int(counters.get("patches", 0)) >= 1:
            return "aligned", ["patch_bearing_reformulation_completed"]
        if variant == "negative" and (not success) and int(counters.get("patches", 0)) >= 1:
            return "aligned", ["negative_reformulation_trace_retained"]
        reasons.append("reformulate_surface_did_not_demonstrate_transition_requirement")
        return "questionable", reasons

    return "unknown", ["unclassified_task"]


def _pain_points(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []

    attention_questionable = [
        episode for episode in episodes
        if episode.get("task_ref") == "task/attention_reformulate/eval/v1"
        and _task_aligned_outcome(episode)[0] == "questionable"
    ]
    if attention_questionable:
        issues.append(
            {
                "rank": 1,
                "category": "eval_surface",
                "impact": "high",
                "issue": "attention_reformulate currently passes immediate eval on the weak baseline path, including the usable-negative variant.",
                "evidence": [episode["_report_path"] for episode in attention_questionable],
            }
        )

    direct_eval_opener = [
        episode for episode in episodes
        if episode.get("steps")
        and str(episode["steps"][0].get("action_name")) == "eval"
    ]
    if direct_eval_opener:
        issues.append(
            {
                "rank": 2,
                "category": "observation_shape",
                "impact": "medium",
                "issue": "The current observation packet makes immediate eval the cheapest attractive opener, so the model often skips knowledge/build/compare unless forced by failure.",
                "evidence": [episode["_report_path"] for episode in direct_eval_opener[:4]],
            }
        )

    return issues


def _strengths(episodes: list[dict[str, Any]]) -> list[str]:
    strengths: list[str] = []
    reduction_positive = next(
        (
            episode for episode in episodes
            if episode.get("task_ref") == "task/reduction_debug/eval/v1"
            and episode.get("variant") == "positive"
            and _task_aligned_outcome(episode)[0] == "aligned"
        ),
        None,
    )
    if reduction_positive is not None:
        strengths.append("GPT-5.4 recovered the reduction_debug task by failing once, inspecting, applying the scripted repair patch, and re-evaluating successfully.")
    profile_positive = next(
        (
            episode for episode in episodes
            if episode.get("task_ref") == "task/profile_diagnose/eval/v1"
            and bool(episode.get("success"))
        ),
        None,
    )
    if profile_positive is not None:
        strengths.append("GPT-5.4 handled profile_diagnose efficiently, choosing a direct eval opener and completing the task in one step.")
    return strengths


def _frictions(episodes: list[dict[str, Any]]) -> list[str]:
    frictions: list[str] = []
    questionable_attention = [
        episode for episode in episodes
        if episode.get("task_ref") == "task/attention_reformulate/eval/v1"
        and _task_aligned_outcome(episode)[0] == "questionable"
    ]
    if questionable_attention:
        frictions.append("The model can exploit the current attention_reformulate eval surface instead of demonstrating a real transformation.")
    if any(int(episode.get("counters", {}).get("patches", 0)) > 0 for episode in episodes):
        frictions.append("Patch-bearing flows work, but they currently rely on controller hints and scripted patch text rather than free-form patch synthesis.")
    return frictions


def build_report(batch_dirs: list[Path]) -> dict[str, Any]:
    episodes = _load_episodes(batch_dirs)
    aligned = 0
    questionable = 0
    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    episode_rows: list[dict[str, Any]] = []
    for episode in episodes:
        alignment, reasons = _task_aligned_outcome(episode)
        usage = _usage_totals(episode)
        prompt_tokens += usage["prompt_tokens"]
        completion_tokens += usage["completion_tokens"]
        total_tokens += usage["total_tokens"]
        if alignment == "aligned":
            aligned += 1
        elif alignment == "questionable":
            questionable += 1
        episode_rows.append(
            {
                "task_ref": episode["task_ref"],
                "variant": episode["variant"],
                "verb": episode["verb"],
                "environment_success": bool(episode["success"]),
                "task_alignment": alignment,
                "alignment_reasons": reasons,
                "step_count": int(episode.get("step_count", 0)),
                "patch_count": int(episode.get("counters", {}).get("patches", 0)),
                "provider": episode.get("provider"),
                "model": episode.get("model"),
                "report_path": episode["_report_path"],
            }
        )

    environment_successes = sum(1 for episode in episodes if bool(episode.get("success")))
    return {
        "report_id": f"gpt54_first_wave_baseline_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "batch_dirs": [str(path.relative_to(ROOT)) for path in batch_dirs],
        "provider_route": "openai_primary",
        "provider_backup": "openrouter_available",
        "model_primary": "gpt-5.4",
        "model_backup": "openai/gpt-5.4-pro",
        "summary": {
            "episode_count": len(episodes),
            "environment_success_count": environment_successes,
            "environment_success_rate": round(environment_successes / len(episodes), 4) if episodes else 0.0,
            "task_aligned_count": aligned,
            "task_aligned_rate": round(aligned / len(episodes), 4) if episodes else 0.0,
            "questionable_count": questionable,
            "avg_step_count": round(sum(int(episode.get("step_count", 0)) for episode in episodes) / len(episodes), 4) if episodes else 0.0,
            "avg_patch_count": round(sum(int(episode.get("counters", {}).get("patches", 0)) for episode in episodes) / len(episodes), 4) if episodes else 0.0,
            "avg_total_tokens": round(total_tokens / len(episodes), 2) if episodes else 0.0,
            "avg_prompt_tokens": round(prompt_tokens / len(episodes), 2) if episodes else 0.0,
            "avg_completion_tokens": round(completion_tokens / len(episodes), 2) if episodes else 0.0,
        },
        "episodes": episode_rows,
        "strengths": _strengths(episodes),
        "frictions": _frictions(episodes),
        "environment_pain_points": _pain_points(episodes),
        "handoff_question_response": {
            "what_gpt54_exploits_well": _strengths(episodes),
            "what_gpt54_fights": _frictions(episodes),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("batch_dirs", nargs="+", type=Path, help="One or more baseline batch output directories")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON report path")
    args = parser.parse_args()

    batch_dirs = [path.resolve() for path in args.batch_dirs]
    report = build_report(batch_dirs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
