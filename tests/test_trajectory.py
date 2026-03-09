from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.runner import run_task
from gpu_cockpit.engine.environment import run_scripted_reference_episode
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
        self.assertEqual(validation["manifest"]["readiness_counts"]["positive_rl_trace"], 1)

    def test_validate_checked_in_public_collection_fixture(self) -> None:
        validation = validate_trajectory_dataset(ROOT / "tests" / "golden_datasets" / "public_collection_v1")
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["manifest"]["episode_count"], 2)
        self.assertEqual(validation["manifest"]["readiness_counts"]["positive_rl_trace"], 2)

    def test_checked_in_reformulate_episode_contains_compare_and_inspect(self) -> None:
        episode_path = ROOT / "tests" / "golden_episodes" / "attention_reformulate_episode_v1.json"
        payload = json.loads(episode_path.read_text(encoding="utf-8"))
        actions = [step["action"]["action_type"] for step in payload["steps"]]
        self.assertIn("inspect", actions)
        self.assertIn("compare", actions)
