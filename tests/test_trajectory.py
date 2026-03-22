from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import EpisodeReadinessReport, ReadinessDecision, TrajectoryAction, TrajectoryEpisode, TrajectoryObservation, TrajectoryStep
from gpu_cockpit.engine.runner import run_task
from gpu_cockpit.engine.environment import _episode_optimize_trace_snapshots, initialize_environment_state, run_scripted_reference_episode, step_environment
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.trajectory import capture_run_episode, export_episode_dataset, export_trajectory_dataset, validate_trajectory_dataset


class TrajectoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_trajectory"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")
        shutil.copytree(ROOT / "knowledge", self.tmp_root / "knowledge")
        build_knowledge_index(self.tmp_root)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_capture_run_episode_from_simple_run(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('trajectory smoke')"],
            trace_system=False,
        )
        episode = capture_run_episode(self.tmp_root, str(run_dir), section="summary")
        self.assertEqual(episode.task_id, "task/smoke/diagnose/v1")
        self.assertEqual(len(episode.steps), 1)
        self.assertEqual(episode.steps[0].action.action_type, "run")
        self.assertEqual(episode.steps[0].observation.projection["run"]["status"], "ok")
        self.assertIsNotNone(episode.governance)
        self.assertIsNotNone(episode.governance_score)
        self.assertIsNotNone(episode.learning_reward_trace)
        self.assertIsNotNone(episode.reward_ledger)

    def test_export_and_validate_dataset(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('trajectory dataset')"],
            trace_system=False,
        )
        manifest_path = export_trajectory_dataset(self.tmp_root, [str(run_dir)], self.tmp_root / "dataset", section="summary")
        self.assertTrue(manifest_path.exists())
        validation = validate_trajectory_dataset(self.tmp_root / "dataset")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["episode_count"], 1)

    def test_export_and_validate_multistep_dataset(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/smoke/eval/v1",
            ["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            section="eval",
        )
        manifest_path = export_episode_dataset([episode], self.tmp_root / "multistep_dataset", policy_id="scripted_reference_v1", split="seed")
        self.assertTrue(manifest_path.exists())
        validation = validate_trajectory_dataset(self.tmp_root / "multistep_dataset")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["manifest"]["successful_episode_count"], 1)
        self.assertEqual(validation["manifest"]["verb_counts"]["diagnose"], 1)
        self.assertEqual(validation["manifest"]["readiness_counts"]["unusable"], 1)
        self.assertEqual(validation["manifest"]["episode_governance_counts"]["unusable"], 1)

    def test_validate_checked_in_public_collection_fixture(self) -> None:
        validation = validate_trajectory_dataset(ROOT / "tests" / "golden_datasets" / "public_collection_v1")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["manifest"]["episode_count"], 2)
        self.assertEqual(validation["manifest"]["readiness_counts"]["benchmark_only"], 2)

    def test_validate_checked_in_transition_collection_fixture(self) -> None:
        validation = validate_trajectory_dataset(ROOT / "tests" / "golden_datasets" / "transition_collection_v1")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["manifest"]["patch_bearing_episode_count"], 2)
        self.assertEqual(validation["manifest"]["lineage_safe_episode_count"], 2)
        self.assertEqual(validation["manifest"]["episode_governance_counts"]["usable_positive_sft"], 2)

    def test_validate_checked_in_negative_transition_collection_fixture(self) -> None:
        validation = validate_trajectory_dataset(ROOT / "tests" / "golden_datasets" / "transition_negative_collection_v1")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["manifest"]["patch_bearing_episode_count"], 2)
        self.assertEqual(validation["manifest"]["usable_negative_episode_count"], 2)

    def test_checked_in_reformulate_episode_contains_compare_and_inspect(self) -> None:
        episode_path = ROOT / "tests" / "golden_episodes" / "attention_reformulate_episode_v1.json"
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        actions = [step["action"]["action_type"] for step in payload["steps"]]
        self.assertIn("inspect", actions)
        self.assertIn("compare", actions)

    def test_export_dataset_counts_patch_transition_kinds(self) -> None:
        state = initialize_environment_state(self.tmp_root, "task/reduction_debug/eval/v1", step_budget=4)
        repaired_text = (self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8")
        next_state, step = step_environment(
            self.tmp_root,
            state,
            action_name="patch_candidate",
            task_ref="task/reduction_debug/eval/v1",
            patch_target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            patch_text=repaired_text,
            patch_intent="repair the reduction mask",
            patch_expected_effect="restore correct row sums",
            patch_transition_kind="repaired",
        )
        episode = TrajectoryEpisode(
            episode_id="episode_patch_transition_smoke",
            created_at=datetime.now(tz=UTC),
            policy_id="scripted_reference_v1",
            task_id="task/reduction_debug/eval/v1",
            task_verb="debug",
            operator_family="reduction_sum",
            source_run_id=next_state.last_run_id or "none",
            source_run_ref=next_state.last_run_ref or "",
            episode_kind="scripted_reference",
            steps=[step.model_copy(update={"terminal": True, "terminal_state": "success"})],
            final_reward=step.reward_total,
            terminal_state="success",
            artifact_refs=step.observation.artifact_refs,
            governance=EpisodeReadinessReport(
                episode_governance_kind="usable_positive_sft",
                training_example_kind="positive_sft_example",
                benchmark_collection=ReadinessDecision(eligible=False, reasons=[]),
                sft_collection=ReadinessDecision(eligible=True, reasons=[]),
                rl_reward_trace=ReadinessDecision(eligible=False, reasons=[]),
                has_build_evidence=True,
                has_profile_evidence=False,
                patch_bearing=True,
                reasons=["patch_bearing", "has_build_evidence"],
                notes=[],
            ),
            metadata={
                "training_example_kind": "positive_sft_example",
                "episode_governance_kind": "usable_positive_sft",
            },
        )
        manifest_path = export_episode_dataset([episode], self.tmp_root / "patch_dataset", policy_id="scripted_reference_v1", split="seed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["transition_kind_counts"]["repaired"], 1)
        self.assertEqual(manifest["patch_kind_counts"]["bug_fix"], 1)
        self.assertEqual(manifest["episode_governance_counts"]["usable_positive_sft"], 1)
        self.assertEqual(manifest["patch_bearing_episode_count"], 1)
        self.assertEqual(manifest["usable_positive_episode_count"], 1)
        self.assertEqual(manifest["lineage_safe_episode_count"], 1)

    def test_export_dataset_counts_patch_bearing_negative_episodes(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/attention_reformulate/eval/v1",
            ["python3", "workloads/reference/triton_attention_score_reformulate_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            workflow="reformulate_negative",
            step_budget=12,
        )
        manifest_path = export_episode_dataset([episode], self.tmp_root / "negative_patch_dataset", policy_id="scripted_reference_v1", split="seed")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["patch_bearing_episode_count"], 1)
        self.assertEqual(manifest["patch_bearing_negative_episode_count"], 1)
        self.assertEqual(manifest["usable_negative_episode_count"], 1)
        self.assertEqual(manifest["episode_governance_counts"]["usable_negative_transition"], 1)

    def test_optimize_episode_surfaces_reward_trace_and_snapshots(self) -> None:
        candidate_step = TrajectoryStep(
            step_index=0,
            action=TrajectoryAction(action_type="patch_candidate"),
            observation=TrajectoryObservation(
                observation_type="projection",
                run_id="run_candidate_001",
                projection={"candidate_projection": {"candidate_state": {"candidate_id": "cand_001"}}},
            ),
            reward_components={},
            reward_total=0.0,
        )
        compare_step = TrajectoryStep(
            step_index=1,
            action=TrajectoryAction(action_type="compare"),
            observation=TrajectoryObservation(
                observation_type="projection",
                run_id="cmp_001",
                projection={
                    "candidate_delta_brief": {"lineage_relationship": "same_parent"},
                    "optimize_delta_summary": {"correctness_change": "preserved_fail"},
                    "recommended_next_actions": ["patch_candidate", "eval"],
                    "summary_lines": ["Both runs still fail correctness."],
                },
            ),
            reward_components={},
            reward_total=0.0,
        )
        failure_step = TrajectoryStep(
            step_index=2,
            action=TrajectoryAction(action_type="inspect_quality"),
            observation=TrajectoryObservation(
                observation_type="projection",
                run_id="run_eval_001",
                projection={
                    "failure_localization": {
                        "hidden_tests": {"code": "hidden_attention_score_mismatch", "next_actions": ["patch_candidate", "eval"]}
                    }
                },
            ),
            reward_components={},
            reward_total=0.0,
        )
        snapshots = _episode_optimize_trace_snapshots([candidate_step, compare_step, failure_step])
        self.assertIsNotNone(snapshots)
        assert snapshots is not None
        self.assertEqual(len(snapshots.candidate_snapshots), 1)
        self.assertEqual(len(snapshots.compare_snapshots), 1)
        self.assertEqual(len(snapshots.failure_localization_snapshots), 1)

    def test_capture_run_episode_surfaces_reward_ledger(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/eval/v1",
            command=["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
            trace_system=False,
        )
        episode = capture_run_episode(self.tmp_root, str(run_dir), section="quality")
        self.assertIsNotNone(episode.reward_ledger)
        assert episode.reward_ledger is not None
        self.assertEqual(episode.reward_ledger.entries[0].action_type, "run")

    def test_validate_dataset_flags_broken_candidate_lineage(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
        )
        bad_step = episode.steps[-1].model_copy(update={"input_candidate_id": "candidate_wrong"})
        broken_episode = episode.model_copy(update={"steps": [*episode.steps[:-1], bad_step]})
        export_episode_dataset([broken_episode], self.tmp_root / "broken_lineage_dataset", policy_id="scripted_reference_v1", split="seed")
        validation = validate_trajectory_dataset(self.tmp_root / "broken_lineage_dataset")
        self.assertEqual(validation["status"], "failed")
        self.assertEqual(len(validation["broken_candidate_lineage_refs"]), 1)
