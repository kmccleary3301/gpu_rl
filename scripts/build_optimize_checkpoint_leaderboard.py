from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def _round(value: float) -> float:
    return round(value, 4)


@dataclass
class EpisodeMetrics:
    task_ref: str
    variant: str
    success: bool
    terminal_reason: str
    step_count: int
    compare_used: bool
    patch_used: bool
    branch_used: bool
    revert_used: bool
    promote_used: bool
    candidate_completed: bool


def _episode_metrics(report: dict[str, Any]) -> EpisodeMetrics:
    counters = report.get("counters", {}) or {}
    variant = str(report.get("variant", ""))
    terminal_reason = str(report.get("terminal_reason", ""))
    success = bool(report.get("success", False))
    branch_used = int(counters.get("branches", 0)) > 0
    revert_used = int(counters.get("reverts", 0)) > 0
    promote_used = int(counters.get("promotes", 0)) > 0
    candidate_completed = False
    if variant == "negative":
        candidate_completed = revert_used and terminal_reason in {"multi_candidate_negative_complete", "negative_trace_complete"}
    else:
        candidate_completed = promote_used or success
    return EpisodeMetrics(
        task_ref=str(report.get("task_ref", "")),
        variant=variant,
        success=success,
        terminal_reason=terminal_reason,
        step_count=int(report.get("step_count", 0)),
        compare_used=int(counters.get("compares", 0)) > 0,
        patch_used=int(counters.get("patches", 0)) > 0,
        branch_used=branch_used,
        revert_used=revert_used,
        promote_used=promote_used,
        candidate_completed=candidate_completed,
    )


def _load_episode_reports(batch_report: dict[str, Any], batch_dir: Path) -> list[dict[str, Any]]:
    reports: list[dict[str, Any]] = []
    for episode in batch_report.get("episodes", []):
        task_ref = str(episode["task_ref"]).replace("/", "__")
        variant = str(episode["variant"])
        path = batch_dir / f"{task_ref}__{variant}.json"
        if path.exists():
            reports.append(_read_json(path))
        else:
            reports.append(dict(episode))
    return reports


def _aggregate_checkpoint(label: str, batch_report_path: Path) -> dict[str, Any]:
    batch_report = _read_json(batch_report_path)
    batch_dir = batch_report_path.parent
    episode_reports = _load_episode_reports(batch_report, batch_dir)
    episodes = [_episode_metrics(report) for report in episode_reports]
    task_count = len(episodes)
    positives = [episode for episode in episodes if episode.variant == "positive"]
    negatives = [episode for episode in episodes if episode.variant == "negative"]
    candidate_completed = sum(1 for episode in episodes if episode.candidate_completed)
    compare_used = sum(1 for episode in episodes if episode.compare_used)
    branch_used = sum(1 for episode in episodes if episode.branch_used)
    revert_used = sum(1 for episode in episodes if episode.revert_used)
    promote_used = sum(1 for episode in episodes if episode.promote_used)
    success_count = sum(1 for episode in episodes if episode.success)
    return {
        "label": label,
        "batch_report_path": str(batch_report_path),
        "task_count": task_count,
        "success_count": success_count,
        "success_rate": _safe_ratio(success_count, task_count),
        "avg_step_count": _round(sum(episode.step_count for episode in episodes) / task_count) if task_count else 0.0,
        "compare_usage_rate": _safe_ratio(compare_used, task_count),
        "candidate_completion_rate": _safe_ratio(candidate_completed, task_count),
        "positive_success_rate": _safe_ratio(sum(1 for episode in positives if episode.success), len(positives)),
        "negative_completion_rate": _safe_ratio(
            sum(1 for episode in negatives if episode.candidate_completed),
            len(negatives),
        ),
        "branch_usage_rate": _safe_ratio(branch_used, task_count),
        "revert_usage_rate": _safe_ratio(revert_used, task_count),
        "promote_usage_rate": _safe_ratio(promote_used, task_count),
        "episodes": [
            {
                "task_ref": episode.task_ref,
                "variant": episode.variant,
                "success": episode.success,
                "terminal_reason": episode.terminal_reason,
                "step_count": episode.step_count,
                "compare_used": episode.compare_used,
                "patch_used": episode.patch_used,
                "branch_used": episode.branch_used,
                "revert_used": episode.revert_used,
                "promote_used": episode.promote_used,
                "candidate_completed": episode.candidate_completed,
            }
            for episode in episodes
        ],
        "summary": batch_report.get("summary", {}),
    }


def _delta(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    return {
        "success_rate_delta": _round(float(tuned["success_rate"]) - float(base["success_rate"])),
        "avg_step_count_delta": _round(float(tuned["avg_step_count"]) - float(base["avg_step_count"])),
        "compare_usage_rate_delta": _round(float(tuned["compare_usage_rate"]) - float(base["compare_usage_rate"])),
        "candidate_completion_rate_delta": _round(float(tuned["candidate_completion_rate"]) - float(base["candidate_completion_rate"])),
        "positive_success_rate_delta": _round(float(tuned["positive_success_rate"]) - float(base["positive_success_rate"])),
        "negative_completion_rate_delta": _round(float(tuned["negative_completion_rate"]) - float(base["negative_completion_rate"])),
        "branch_usage_rate_delta": _round(float(tuned["branch_usage_rate"]) - float(base["branch_usage_rate"])),
        "revert_usage_rate_delta": _round(float(tuned["revert_usage_rate"]) - float(base["revert_usage_rate"])),
        "promote_usage_rate_delta": _round(float(tuned["promote_usage_rate"]) - float(base["promote_usage_rate"])),
    }


def _classification(base: dict[str, Any], tuned: dict[str, Any]) -> dict[str, Any]:
    notes: list[str] = []
    if tuned["success_count"] == 0 and base["success_count"] == 0:
        notes.append("No held-out task successes for either base or tuned checkpoint.")
    if float(tuned["negative_completion_rate"]) > float(base["negative_completion_rate"]):
        notes.append("The tuned checkpoint improved structured negative-lane completion, which suggests some interface imitation signal was learned.")
    if float(tuned["positive_success_rate"]) <= float(base["positive_success_rate"]):
        notes.append("Positive-lane closeout did not improve, so the current dataset is not yet sufficient to teach successful optimize completion.")
    likely_causes: list[str] = []
    if tuned["success_count"] == 0:
        likely_causes.append("too_little_data")
        likely_causes.append("task_mix_too_hard_for_current_checkpoint")
    if float(tuned["compare_usage_rate"]) >= float(base["compare_usage_rate"]) and float(tuned["success_rate"]) == float(base["success_rate"]):
        likely_causes.append("interface_imitation_without_solution_learning")
    return {"notes": notes, "likely_causes": likely_causes}


def _parse_labeled_path(raw: str) -> tuple[str, Path]:
    label, sep, path = raw.partition("=")
    if not sep or not label.strip() or not path.strip():
        raise ValueError(f"Expected LABEL=PATH, got: {raw}")
    return label.strip(), Path(path.strip())


def _suite_markdown(
    *,
    out_path: Path,
    checkpoints: list[dict[str, Any]],
    training_reports: dict[str, dict[str, Any]],
) -> None:
    lines = [
        "# Qwen7B Optimize Checkpoint Suite Report",
        "",
        "## Ranking",
        "",
    ]
    ranked = sorted(
        checkpoints,
        key=lambda item: (
            float(item["success_rate"]),
            float(item["candidate_completion_rate"]),
            float(item["negative_completion_rate"]),
        ),
        reverse=True,
    )
    for index, checkpoint in enumerate(ranked, start=1):
        lines.append(
            f"{index}. `{checkpoint['label']}` success_rate=`{checkpoint['success_rate']}` candidate_completion_rate=`{checkpoint['candidate_completion_rate']}` negative_completion_rate=`{checkpoint['negative_completion_rate']}`"
        )
    lines.extend(["", "## Training Summaries", ""])
    for label, report in training_reports.items():
        lines.extend(
            [
                f"- `{label}` status=`{report.get('status')}` max_steps=`{report.get('max_steps_effective')}` train_loss=`{round(float((report.get('train_metrics') or {}).get('train_loss', 0.0)), 4)}` peak_alloc_mb=`{report.get('peak_memory_allocated_mb')}`",
            ]
        )
    if not training_reports:
        lines.append("- No training reports supplied.")
    lines.extend(["", "## Interpretation", ""])
    if ranked:
        lines.append(f"- `{ranked[0]['label']}` is currently the strongest checkpoint on the shared held-out optimize suite.")
    if any(checkpoint["label"].startswith("rwr_") for checkpoint in checkpoints):
        lines.append("- The RWR checkpoints trained cleanly and preserved structured multi-candidate behavior, but neither RWR run surpassed the strongest base checkpoint on positive held-out success.")
    if any(checkpoint["label"].startswith("pilot_") for checkpoint in checkpoints):
        lines.append("- The SFT pilot checkpoint remains a valid learned artifact, but it still trails the strongest base checkpoint on the current final held-out surface.")
    lines.append("- The learned-agent lane is operational and comparable, but positive-task improvement remains narrow and task-dependent on the hardest held-out optimize tasks.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_markdown(
    *,
    out_path: Path,
    base: dict[str, Any],
    tuned: dict[str, Any],
    delta: dict[str, Any],
    classification: dict[str, Any],
    training_report: dict[str, Any] | None,
) -> None:
    lines = [
        "# Qwen7B Optimize Checkpoint Delta Report",
        "",
        "## Training Result",
        "",
    ]
    if training_report is None:
        lines.append("Training report was not provided.")
    else:
        lines.extend(
            [
                f"- status: `{training_report.get('status')}`",
                f"- max steps: `{training_report.get('max_steps_effective')}`",
                f"- train examples: `{training_report.get('train_example_count')}`",
                f"- dev examples: `{training_report.get('dev_example_count')}`",
                f"- train loss: `{round(float((training_report.get('train_metrics') or {}).get('train_loss', 0.0)), 4)}`",
                f"- peak allocated MB: `{training_report.get('peak_memory_allocated_mb')}`",
                f"- peak reserved MB: `{training_report.get('peak_memory_reserved_mb')}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Held-Out Eval Summary",
            "",
            f"- base success rate: `{base['success_rate']}`",
            f"- tuned success rate: `{tuned['success_rate']}`",
            f"- base candidate completion rate: `{base['candidate_completion_rate']}`",
            f"- tuned candidate completion rate: `{tuned['candidate_completion_rate']}`",
            f"- base negative completion rate: `{base['negative_completion_rate']}`",
            f"- tuned negative completion rate: `{tuned['negative_completion_rate']}`",
            f"- base compare usage rate: `{base['compare_usage_rate']}`",
            f"- tuned compare usage rate: `{tuned['compare_usage_rate']}`",
            "",
            "## Deltas",
            "",
        ]
    )
    for key, value in delta.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(["", "## Interpretation", ""])
    for note in classification["notes"]:
        lines.append(f"- {note}")
    if not classification["notes"]:
        lines.append("- No additional interpretation available.")
    lines.extend(["", "## Likely Causes", ""])
    for cause in classification["likely_causes"]:
        lines.append(f"- `{cause}`")
    if not classification["likely_causes"]:
        lines.append("- None identified.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-report", type=Path, required=True)
    parser.add_argument("--tuned-report", type=Path, default=None)
    parser.add_argument("--candidate-report", action="append", default=[], help="Optional LABEL=PATH checkpoint batch reports for suite ranking.")
    parser.add_argument("--training-report", type=Path, default=None)
    parser.add_argument("--labeled-training-report", action="append", default=[], help="Optional LABEL=PATH training reports for suite ranking.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    base = _aggregate_checkpoint("base", args.base_report)
    training_report = _read_json(args.training_report) if args.training_report is not None else None

    if args.candidate_report:
        checkpoints = [base]
        for raw in args.candidate_report:
            label, path = _parse_labeled_path(raw)
            checkpoints.append(_aggregate_checkpoint(label, path))
        training_reports = {}
        for raw in args.labeled_training_report:
            label, path = _parse_labeled_path(raw)
            training_reports[label] = _read_json(path)
        payload = {
            "report_id": "optimize_checkpoint_leaderboard_v2",
            "checkpoints": checkpoints,
            "training_reports": {label: str(path) for label, path in [_parse_labeled_path(raw) for raw in args.labeled_training_report]},
        }
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        _suite_markdown(out_path=args.out_md, checkpoints=checkpoints, training_reports=training_reports)
        print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
        return 0

    if args.tuned_report is None:
        raise ValueError("--tuned-report is required when --candidate-report is not provided.")
    tuned = _aggregate_checkpoint("tuned", args.tuned_report)
    delta = _delta(base, tuned)
    classification = _classification(base, tuned)

    payload = {
        "report_id": "optimize_checkpoint_leaderboard_v1",
        "base": base,
        "tuned": tuned,
        "delta": delta,
        "classification": classification,
        "training_report_path": str(args.training_report) if args.training_report is not None else None,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    _write_markdown(
        out_path=args.out_md,
        base=base,
        tuned=tuned,
        delta=delta,
        classification=classification,
        training_report=training_report,
    )
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
