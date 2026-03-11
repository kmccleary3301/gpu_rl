from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RLRolloutConfig
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.rollout import run_scripted_rollout_suite
from gpu_cockpit.engine.training import load_training_config, validate_sft_training_config, write_sft_smoke_report


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        subprocess.run(["python3", str(ROOT / "scripts" / "build_first_target_training_assets.py")], check=True)

    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_training"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")
        shutil.copytree(ROOT / "knowledge", self.tmp_root / "knowledge")
        shutil.copytree(ROOT / "tests" / "golden_datasets", self.tmp_root / "tests" / "golden_datasets")
        build_knowledge_index(self.tmp_root)

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_validate_checked_in_sft_training_config(self) -> None:
        config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
        config = load_training_config(config_path)
        validation = validate_sft_training_config(ROOT, config)
        self.assertEqual(validation["status"], "ok")
        self.assertGreaterEqual(len(validation["datasets"]), 2)

    def test_write_sft_smoke_report(self) -> None:
        config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
        out_path = self.tmp_root / "smoke_sft_report.json"
        write_sft_smoke_report(ROOT, config_path, out_path)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["config_id"], "cfg/sft/qwen32b/debug_repair_lora/v1")

    def test_build_dataset_governance_report_script(self) -> None:
        subprocess.run(["python3", str(ROOT / "scripts" / "build_dataset_governance_report.py")], check=True)
        payload = json.loads((ROOT / "artifacts" / "training" / "dataset_governance_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["report_id"], "report/dataset_governance/v1")
        self.assertEqual(payload["trajectory_datasets"]["train"]["usable_negative_episode_count"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["task_verb_counts"]["debug"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["operator_family_counts"]["reduction_sum"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["patch_bearing_example_count"], 4)

    def test_run_scripted_rollout_suite(self) -> None:
        config_path = ROOT / "configs" / "training" / "rollout_debug_repair_v1.json"
        config = RLRolloutConfig.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
        report = run_scripted_rollout_suite(self.tmp_root, config, self.tmp_root / "rollout_report")
        self.assertEqual(report.task_count, 3)
        self.assertGreaterEqual(report.patch_bearing_count, 2)
        self.assertTrue((self.tmp_root / "rollout_report" / "rollout_report.json").exists())


if __name__ == "__main__":
    unittest.main()
