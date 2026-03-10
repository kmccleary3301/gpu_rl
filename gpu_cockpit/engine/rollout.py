from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from gpu_cockpit.contracts import RLRolloutConfig, RolloutEvaluationReport, RolloutTaskResult
from gpu_cockpit.engine.environment import run_scripted_reference_episode
from gpu_cockpit.engine.task_registry import TaskRegistry
from gpu_cockpit.engine.trajectory import write_episode


def _default_command_for_task(root: Path, task_id: str) -> list[str]:
    task = TaskRegistry(root).get(task_id)
    if task.reference_impl_ref and task.reference_impl_ref.endswith(".py"):
        return ["python3", task.reference_impl_ref, "--benchmark-repeats", "2"]
    if task.reference_impl_ref and task.reference_impl_ref.endswith(".sh"):
        return ["bash", task.reference_impl_ref]
    if task_id == "task/smoke/eval/v1":
        return ["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"]
    raise RuntimeError(f"Cannot infer default command for task {task_id}")


def run_scripted_rollout_suite(root: Path, config: RLRolloutConfig, out_dir: Path) -> RolloutEvaluationReport:
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_dir = out_dir / "episodes"
    episodes_dir.mkdir(parents=True, exist_ok=True)
    results: list[RolloutTaskResult] = []
    final_rewards: list[float] = []
    for task_id in config.task_refs:
        command = _default_command_for_task(root, task_id)
        episode = run_scripted_reference_episode(
            root,
            task_id,
            command,
            policy_id=config.policy_id,
            step_budget=config.step_budget,
            determinism_runs=config.determinism_runs,
            workflow=config.workflow,
            section="quality",
        )
        episode_path = episodes_dir / f"{episode.episode_id}.json"
        write_episode(episode, episode_path)
        governance = episode.governance
        result = RolloutTaskResult(
            task_id=task_id,
            terminal_state=episode.terminal_state,
            final_reward=episode.final_reward,
            training_example_kind=governance.training_example_kind if governance is not None else str(episode.metadata.get("training_example_kind", "unusable")),
            episode_governance_kind=governance.episode_governance_kind if governance is not None else str(episode.metadata.get("episode_governance_kind", "unusable")),
            patch_bearing=bool(governance.patch_bearing if governance is not None else episode.metadata.get("episode_patch_bearing", False)),
            step_count=len(episode.steps),
            episode_ref=str(episode_path.relative_to(out_dir)),
        )
        results.append(result)
        final_rewards.append(episode.final_reward)
    report = RolloutEvaluationReport(
        report_id=f"rollout_report_{datetime.now(tz=UTC).strftime('%Y%m%d_%H%M%S')}",
        created_at=datetime.now(tz=UTC),
        config_id=config.config_id,
        policy_id=config.policy_id,
        task_count=len(results),
        success_count=sum(1 for result in results if result.terminal_state == "success"),
        patch_bearing_count=sum(1 for result in results if result.patch_bearing),
        avg_final_reward=round(sum(final_rewards) / len(final_rewards), 4) if final_rewards else 0.0,
        results=results,
        notes=[f"workflow:{config.workflow}", f"step_budget:{config.step_budget}"],
    )
    report_path = out_dir / "rollout_report.json"
    report_path.write_text(json.dumps(report.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")
    return report
