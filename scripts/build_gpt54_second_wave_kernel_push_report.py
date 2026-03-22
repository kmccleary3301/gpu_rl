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


def _step_actions(episode: dict[str, Any]) -> list[str]:
    rows: list[str] = []
    for step in episode.get("steps", []):
        action = step.get("action_name")
        if isinstance(action, str):
            rows.append(action)
    return rows


def _task_aligned_outcome(episode: dict[str, Any]) -> tuple[str, list[str]]:
    task_ref = str(episode.get("task_ref"))
    success = bool(episode.get("success"))
    actions = _step_actions(episode)
    counters = episode.get("counters", {})

    if task_ref == "task/reduction_row_sum/eval/v1":
        if success and "bench" in actions:
            return "aligned", ["optimize_trace_bench_then_eval_completed"]
        if success:
            return "aligned", ["optimize_eval_completed_without_explicit_bench"]
        return "misaligned", ["row_sum_optimize_candidate_failed_eval"]

    if task_ref == "task/kernelbench/level1/47_sum_reduction/eval/v1":
        if success and "bench" in actions:
            return "aligned", ["kernelbench_optimize_trace_bench_then_eval_completed"]
        if success:
            return "aligned", ["kernelbench_optimize_eval_completed"]
        return "misaligned", ["kernelbench_sum_reduction_candidate_failed_eval"]

    if task_ref == "task/attention_score/eval/v1":
        if success and int(counters.get("patches", 0)) > 0 and "bench" in actions:
            return "aligned", ["attention_score_bounded_patch_optimize_completed"]
        if success and ("bench" in actions or "run" in actions):
            return "aligned", ["attention_score_candidate_completed_with_tooling"]
        if success:
            return "aligned", ["attention_score_candidate_eval_completed"]
        if int(counters.get("failed_tool_calls", 0)) > 0:
            return "misaligned", ["attention_score_optimize_task_hit_tool_or_eval_failure"]
        return "misaligned", ["attention_score_candidate_failed_eval"]

    return "unknown", ["unclassified_task"]


def _strengths(episodes: list[dict[str, Any]]) -> list[str]:
    strengths: list[str] = []
    row_sum = next(
        (
            episode for episode in episodes
            if episode.get("task_ref") == "task/reduction_row_sum/eval/v1"
            and _task_aligned_outcome(episode)[0] == "aligned"
        ),
        None,
    )
    if row_sum is not None:
        strengths.append("GPT-5.4 completed the direct Triton row-sum optimize task on the broader second-wave surface.")
    kernelbench = next(
        (
            episode for episode in episodes
            if episode.get("task_ref") == "task/kernelbench/level1/47_sum_reduction/eval/v1"
            and _task_aligned_outcome(episode)[0] == "aligned"
        ),
        None,
    )
    if kernelbench is not None:
        strengths.append("GPT-5.4 generalized onto a curated KernelBench sum-reduction task rather than only the hand-authored first-wave tasks.")
    return strengths


def _frictions(episodes: list[dict[str, Any]]) -> list[str]:
    frictions: list[str] = []
    if any(episode.get("task_ref") == "task/attention_score/eval/v1" and not bool(episode.get("success")) for episode in episodes):
        frictions.append("The bounded attention_score optimize lane still exposes a correctness ceiling for GPT-5.4 on the current action surface.")
    direct_eval = [episode for episode in episodes if episode.get("steps") and str(episode["steps"][0].get("action_name")) == "eval"]
    if direct_eval:
        frictions.append("Even on optimize tasks, the observation packet still makes direct eval attractive compared with a more exploratory bench-first workflow.")
    return frictions


def _pain_points(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    attention_failures = [
        episode for episode in episodes
        if episode.get("task_ref") == "task/attention_score/eval/v1" and not bool(episode.get("success"))
    ]
    if attention_failures:
        issues.append(
            {
                "rank": 1,
                "category": "task_ceiling",
                "impact": "high",
                "issue": "The bounded attention-score optimize lane still fails on some GPT-5.4 traces, which limits claims about broader open-ended kernel writing.",
                "evidence": [episode["_report_path"] for episode in attention_failures],
            }
        )
    bench_skips = [
        episode for episode in episodes
        if episode.get("task_ref") in {"task/reduction_row_sum/eval/v1", "task/kernelbench/level1/47_sum_reduction/eval/v1"}
        and "bench" not in _step_actions(episode)
    ]
    if bench_skips:
        issues.append(
            {
                "rank": 2,
                "category": "observation_shape",
                "impact": "medium",
                "issue": "The optimize packet still does not force a comparison habit, so GPT-5.4 can skip bench even when a useful baseline command exists.",
                "evidence": [episode["_report_path"] for episode in bench_skips],
            }
        )
    return issues


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
        "report_id": f"gpt54_second_wave_kernel_push_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
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
