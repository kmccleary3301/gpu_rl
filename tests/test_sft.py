from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.environment import run_scripted_reference_episode
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset
from gpu_cockpit.engine.trajectory import export_episode_dataset


class SFTPackagingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_sft"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")
        shutil.copytree(ROOT / "knowledge", self.tmp_root / "knowledge")
        build_knowledge_index(self.tmp_root)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_package_and_validate_sft_dataset(self) -> None:
        episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
        )
        trajectory_dir = self.tmp_root / "trajectory_dataset"
        export_episode_dataset([episode], trajectory_dir, policy_id="scripted_reference_v1", split="seed")
        out_dir = self.tmp_root / "sft_dataset"
        manifest_path = package_trajectory_dataset_as_sft(
            self.tmp_root,
            trajectory_dir,
            out_dir,
            split="train",
        )
        self.assertTrue(manifest_path.exists())
        validation = validate_sft_dataset(out_dir)
        self.assertEqual(validation["status"], "ok")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["example_count"], 1)
        example_path = out_dir / manifest["example_refs"][0]
        example = json.loads(example_path.read_text(encoding="utf-8"))
        self.assertEqual(example["task_id"], "task/reduction_debug/eval/v1")
        self.assertIn("knowledge_query", example["response"])
        self.assertEqual(example["metadata"]["training_example_kind"], "positive_sft_example")
        self.assertEqual(example["metadata"]["episode_governance_kind"], "usable_positive_sft")
        self.assertEqual(manifest["metadata"]["verb_counts"]["debug"], 1)
        self.assertEqual(manifest["metadata"]["operator_family_counts"]["reduction_sum"], 1)
        self.assertEqual(manifest["metadata"]["training_example_kind_counts"]["positive_sft_example"], 1)
        self.assertEqual(manifest["metadata"]["episode_governance_counts"]["usable_positive_sft"], 1)
        self.assertEqual(manifest["metadata"]["transition_kind_counts"]["repaired"], 1)
        self.assertEqual(manifest["metadata"]["patch_kind_counts"]["bug_fix"], 1)
        self.assertEqual(manifest["metadata"]["patch_bearing_example_count"], 1)

    def test_package_filters_public_benchmark_and_verb(self) -> None:
        smoke_episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/smoke/eval/v1",
            ["python3", "-c", "print('GPU_COCKPIT_SMOKE_OK')"],
        )
        public_episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/kernelbench/level1/32_hardtanh/eval/v1",
            [
                "python3",
                "workloads/reference/kernelbench_reference_runner.py",
                "--case-config",
                "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_032_hardtanh.json",
                "--benchmark-repeats",
                "2",
            ],
            section="eval",
        )
        trajectory_dir = self.tmp_root / "filtered_trajectory_dataset"
        export_episode_dataset([smoke_episode, public_episode], trajectory_dir, policy_id="scripted_reference_v1", split="seed")
        out_dir = self.tmp_root / "filtered_sft_dataset"
        public_verb = public_episode.task_verb or "optimize"
        manifest_path = package_trajectory_dataset_as_sft(
            self.tmp_root,
            trajectory_dir,
            out_dir,
            split="train",
            include_failures=False,
            only_public_benchmarks=True,
            include_benchmark_only=True,
            verb_allowlist=[public_verb],
            allowed_training_example_kinds=["benchmark_only"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["example_count"], 1)
        self.assertEqual(manifest["metadata"]["verb_allowlist"], [public_verb])

    def test_package_can_filter_patch_bearing_governed_examples(self) -> None:
        debug_episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
        )
        public_episode = run_scripted_reference_episode(
            self.tmp_root,
            "task/kernelbench/level1/32_hardtanh/eval/v1",
            [
                "python3",
                "workloads/reference/kernelbench_reference_runner.py",
                "--case-config",
                "workloads/public_benchmarks/kernelbench/v0_1/cases/level1_032_hardtanh.json",
                "--benchmark-repeats",
                "2",
            ],
            section="eval",
        )
        trajectory_dir = self.tmp_root / "patch_bearing_trajectory_dataset"
        export_episode_dataset([debug_episode, public_episode], trajectory_dir, policy_id="scripted_reference_v1", split="seed")
        out_dir = self.tmp_root / "patch_bearing_sft_dataset"
        manifest_path = package_trajectory_dataset_as_sft(
            self.tmp_root,
            trajectory_dir,
            out_dir,
            split="train",
            patch_bearing_only=True,
            governance_allowlist=["usable_positive_sft"],
            transition_kind_allowlist=["repaired"],
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.assertEqual(manifest["example_count"], 1)
        self.assertTrue(manifest["metadata"]["patch_bearing_only"])
        self.assertEqual(manifest["metadata"]["patch_bearing_example_count"], 1)
        example = json.loads((out_dir / manifest["example_refs"][0]).read_text(encoding="utf-8"))
        self.assertTrue(example["metadata"]["patch_present"])
        self.assertEqual(example["metadata"]["episode_governance_kind"], "usable_positive_sft")

    def test_validate_checked_in_transition_sft_fixture(self) -> None:
        validation = validate_sft_dataset(ROOT / "tests" / "golden_datasets" / "transition_sft_v1")
        self.assertEqual(validation["status"], "ok")
        manifest = validation["manifest"]
        self.assertEqual(manifest["example_count"], 2)
        self.assertTrue(manifest["metadata"]["patch_bearing_only"])
        self.assertEqual(manifest["metadata"]["patch_bearing_example_count"], 2)
        self.assertEqual(manifest["metadata"]["transition_kind_counts"]["repaired"], 1)
        self.assertEqual(manifest["metadata"]["transition_kind_counts"]["reformulated"], 1)

    def test_validate_checked_in_negative_transition_sft_fixture(self) -> None:
        validation = validate_sft_dataset(ROOT / "tests" / "golden_datasets" / "transition_negative_sft_v1")
        self.assertEqual(validation["status"], "ok")
        manifest = validation["manifest"]
        self.assertEqual(manifest["example_count"], 2)
        self.assertTrue(manifest["metadata"]["patch_bearing_only"])
        self.assertEqual(manifest["metadata"]["patch_bearing_example_count"], 2)
        self.assertEqual(manifest["metadata"]["training_example_kind_counts"]["negative_debug_example"], 1)
        self.assertEqual(manifest["metadata"]["training_example_kind_counts"]["negative_reformulate_example"], 1)
