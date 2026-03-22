from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.task_registry import TaskRegistry

CURATED_EPISODES = [
    "artifacts/baselines/gpt54_attention_score_bounded_patch_probe_v1/batch_v6_forced_eval_closeout_retry1/task__attention_score__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_bounded_patch_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kv_cache_gather_bounded_patch_probe_v1/batch_v2_retry1/task__kv_cache_gather__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_sum_reduction_bounded_patch_probe_v1/batch_v1_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_bounded_patch_probe_v1/batch_v3_public_signal_retry2/task__kernelbench__level1__23_softmax__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_multi_candidate_positive_probe_v1/batch_v2_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_positive_probe_v1/batch_v2_retry1/task__kernelbench__level1__23_softmax__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_public_negative_bounded_patch_probe_v1/batch_v2_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__negative.json",
    "artifacts/baselines/gpt54_kernelbench_public_negative_bounded_patch_probe_v1/batch_v2_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_reduction_row_sum_multi_candidate_negative_probe_v1/batch_v2_retry1/task__reduction_row_sum__eval__v1__negative.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_multi_candidate_negative_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_second_wave_kernel_push_v1/batch_v1/task__attention_score__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_two_attempt_positive_probe_v1/batch_v1_retry1/task__reduction_row_sum__eval__v1__positive.json",
    "artifacts/baselines/gpt54_kernelbench_softmax_wide_two_attempt_positive_probe_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
    "artifacts/baselines/gpt54_reduction_row_sum_three_attempt_positive_probe_v1/batch_v2_branching_task_retry1/task__reduction_row_sum_branching__eval__v1__positive.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__47_sum_reduction__eval__v1__positive.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax__eval__v1__negative.json",
    "artifacts/baselines/gpt54_final_teacher_policy_tranche_v1/batch_v1_retry1/task__kernelbench__level1__23_softmax_wide__eval__v1__positive.json",
]


def _load_episode(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _task_scope(task_ref: str) -> str:
    return "public" if "kernelbench" in task_ref else "internal"


def _candidate_mode(counters: dict[str, Any]) -> str:
    if any(int(counters.get(key, 0)) > 0 for key in ("branches", "reverts", "promotes")):
        return "multi_candidate"
    return "single_candidate"


def _perf_bucket(payload: dict[str, Any]) -> str:
    steps = payload.get("steps", [])
    if not isinstance(steps, list):
        return "unknown"
    for step in reversed(steps):
        if not isinstance(step, dict):
            continue
        if step.get("action_name") != "compare":
            continue
        observation = step.get("observation", {})
        if not isinstance(observation, dict):
            continue
        projection = observation.get("projection", {})
        if not isinstance(projection, dict):
            continue
        optimize_delta = projection.get("optimize_delta_summary", {})
        if not isinstance(optimize_delta, dict):
            continue
        perf_change = optimize_delta.get("perf_change")
        if perf_change in {"improved", "regressed", "unchanged"}:
            return str(perf_change)
    return "unknown"


def _quality_bucket(payload: dict[str, Any]) -> str:
    if bool(payload.get("success")):
        return "usable_positive"
    terminal_reason = str(payload.get("terminal_reason", ""))
    if terminal_reason in {"negative_trace_complete", "multi_candidate_negative_complete"}:
        return "usable_negative"
    if terminal_reason in {"two_attempt_positive_complete", "three_attempt_positive_complete", "post_patch_eval_failed"}:
        return "near_miss"
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    if (
        str(payload.get("verb")) == "optimize"
        and int(counters.get("failed_tool_calls", 0)) == 0
        and int(counters.get("bench_actions", 0)) >= 2
        and int(counters.get("eval_actions", 0)) >= 2
    ):
        return "near_miss"
    return "excluded"


def _trace_usability_bucket(quality_bucket: str) -> str:
    if quality_bucket == "usable_positive":
        return "trainable_positive"
    if quality_bucket == "usable_negative":
        return "trainable_negative"
    if quality_bucket == "near_miss":
        return "analysis_only"
    return "excluded"


def _episode_entry(relative_ref: str) -> dict[str, Any]:
    payload = _load_episode(ROOT / relative_ref)
    task_ref = str(payload.get("task_ref"))
    variant = str(payload.get("variant"))
    counters = payload.get("counters", {})
    if not isinstance(counters, dict):
        counters = {}
    task = TaskRegistry(ROOT).get(task_ref)
    quality_bucket = _quality_bucket(payload)
    return {
        "episode_ref": relative_ref,
        "task_ref": task_ref,
        "variant": variant,
        "verb": str(payload.get("verb", "")),
        "difficulty": task.difficulty,
        "quality_bucket": quality_bucket,
        "trace_usability": _trace_usability_bucket(quality_bucket),
        "task_scope": _task_scope(task_ref),
        "candidate_mode": _candidate_mode(counters),
        "correctness_bucket": "pass" if bool(payload.get("success")) else "fail",
        "perf_bucket": _perf_bucket(payload),
        "success": bool(payload.get("success")),
        "terminal_reason": payload.get("terminal_reason"),
        "step_count": int(payload.get("step_count", 0)),
        "counters": counters,
    }


def _count_by(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(field, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _aggregate_counters(rows: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "patches": 0,
        "compares": 0,
        "branches": 0,
        "reverts": 0,
        "promotes": 0,
        "replays": 0,
    }
    for row in rows:
        counters = row.get("counters", {})
        if not isinstance(counters, dict):
            continue
        for key in totals:
            totals[key] += int(counters.get(key, 0))
    return totals


def main() -> int:
    out_dir = ROOT / "artifacts" / "training" / "optimize_trace_dataset_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    episode_entries = [_episode_entry(relative_ref) for relative_ref in CURATED_EPISODES]
    aggregate_counters = _aggregate_counters(episode_entries)

    manifest = {
        "dataset_id": "optimize_trace_dataset_v4",
        "created_at": datetime.now(tz=UTC).isoformat(),
        "schema_version": "4.0.0",
        "episode_count": len(episode_entries),
        "quality_counts": _count_by(episode_entries, "quality_bucket"),
        "trace_usability_counts": _count_by(episode_entries, "trace_usability"),
        "variant_counts": _count_by(episode_entries, "variant"),
        "task_counts": _count_by(episode_entries, "task_ref"),
        "task_scope_counts": _count_by(episode_entries, "task_scope"),
        "difficulty_counts": _count_by(episode_entries, "difficulty"),
        "candidate_mode_counts": _count_by(episode_entries, "candidate_mode"),
        "correctness_counts": _count_by(episode_entries, "correctness_bucket"),
        "perf_counts": _count_by(episode_entries, "perf_bucket"),
        "aggregate_counters": aggregate_counters,
        "notes": [
            "Curated GPT-5.4 optimize traces spanning internal/public positive, usable-negative, and near-miss episodes.",
            "Includes branch-capable positive and negative traces for the multi-candidate loop.",
            "Adds true two-attempt bounded optimize traces on both an internal lane and a harder public lane.",
            "Adds a successful three-attempt internal branching trace and the final local teacher-policy tranche outcomes.",
        ],
        "episodes": episode_entries,
    }
    coverage_report = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "quality_counts": manifest["quality_counts"],
        "trace_usability_counts": manifest["trace_usability_counts"],
        "task_scope_counts": manifest["task_scope_counts"],
        "difficulty_counts": manifest["difficulty_counts"],
        "candidate_mode_counts": manifest["candidate_mode_counts"],
        "correctness_counts": manifest["correctness_counts"],
        "perf_counts": manifest["perf_counts"],
        "aggregate_counters": aggregate_counters,
    }
    training_readiness = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "checks": {
            "has_usable_positive": manifest["quality_counts"].get("usable_positive", 0) > 0,
            "has_usable_negative": manifest["quality_counts"].get("usable_negative", 0) > 0,
            "has_near_miss": manifest["quality_counts"].get("near_miss", 0) > 0,
            "has_internal_examples": manifest["task_scope_counts"].get("internal", 0) > 0,
            "has_public_examples": manifest["task_scope_counts"].get("public", 0) > 0,
            "has_multi_candidate_examples": manifest["candidate_mode_counts"].get("multi_candidate", 0) > 0,
            "has_positive_multi_candidate_examples": any(
                row["candidate_mode"] == "multi_candidate" and row["quality_bucket"] == "usable_positive"
                for row in episode_entries
            ),
            "has_negative_multi_candidate_examples": any(
                row["candidate_mode"] == "multi_candidate" and row["quality_bucket"] == "usable_negative"
                for row in episode_entries
            ),
            "has_two_attempt_positive_examples": any(
                row["candidate_mode"] == "multi_candidate"
                and row["quality_bucket"] == "usable_positive"
                and "two_attempt" in row["episode_ref"]
                for row in episode_entries
            ),
            "has_harder_public_near_miss": any(
                row["task_ref"] == "task/kernelbench/level1/23_softmax_wide/eval/v1"
                and row["quality_bucket"] == "near_miss"
                for row in episode_entries
            ),
            "has_three_attempt_positive_examples": any(
                row["task_ref"] == "task/reduction_row_sum_branching/eval/v1"
                and row["quality_bucket"] == "usable_positive"
                for row in episode_entries
            ),
        },
        "recommended_use": {
            "sft_positive_pool": [
                row["episode_ref"] for row in episode_entries if row["quality_bucket"] == "usable_positive"
            ],
            "analysis_negative_pool": [
                row["episode_ref"] for row in episode_entries if row["quality_bucket"] == "usable_negative"
            ],
            "analysis_near_miss_pool": [
                row["episode_ref"] for row in episode_entries if row["quality_bucket"] == "near_miss"
            ],
        },
    }
    reward_readiness = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "sufficiency": {
            "more_sft": manifest["quality_counts"].get("usable_positive", 0) >= 6 and manifest["quality_counts"].get("usable_negative", 0) >= 4,
            "preference_style_ranking": manifest["quality_counts"].get("near_miss", 0) >= 2,
            "early_rl": manifest["candidate_mode_counts"].get("multi_candidate", 0) >= 5
            and manifest["quality_counts"].get("usable_positive", 0) >= 6
            and manifest["quality_counts"].get("near_miss", 0) >= 2,
        },
        "notes": [
            "Early RL readiness is intentionally stricter than SFT readiness.",
            "Two-attempt positive traces, the branching row-sum success, and harder public near-misses are the main drivers for the v4 upgrade.",
        ],
    }
    preference_ranking = {
        "dataset_id": manifest["dataset_id"],
        "created_at": manifest["created_at"],
        "ranking_policy": {
            "usable_positive": 3,
            "near_miss": 2,
            "usable_negative": 1,
            "excluded": 0,
        },
        "ranked_episode_refs": sorted(
            [
                {
                    "episode_ref": row["episode_ref"],
                    "quality_bucket": row["quality_bucket"],
                    "trace_usability": row["trace_usability"],
                    "difficulty": row["difficulty"],
                    "score": {
                        "usable_positive": 3,
                        "near_miss": 2,
                        "usable_negative": 1,
                        "excluded": 0,
                    }.get(str(row["quality_bucket"]), 0),
                }
                for row in episode_entries
            ],
            key=lambda item: (int(item["score"]), item["difficulty"], item["episode_ref"]),
            reverse=True,
        ),
        "notes": [
            "Lightweight preference-ready ordering for future ranking or preference-style training experiments.",
            "This is a curation aid, not a claim that preference training is already complete.",
        ],
    }

    (out_dir / "optimize_trace_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    (out_dir / "optimize_trace_coverage_report.json").write_text(json.dumps(coverage_report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "training_readiness_report.json").write_text(json.dumps(training_readiness, indent=2) + "\n", encoding="utf-8")
    (out_dir / "reward_readiness_report.json").write_text(json.dumps(reward_readiness, indent=2) + "\n", encoding="utf-8")
    (out_dir / "preference_ranking_report.json").write_text(json.dumps(preference_ranking, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(coverage_report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
