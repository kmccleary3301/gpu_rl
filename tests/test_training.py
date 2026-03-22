from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RLRolloutConfig
from gpu_cockpit.engine.knowledge import build_knowledge_index
from gpu_cockpit.engine.rollout import run_scripted_rollout_suite
from gpu_cockpit.engine.training import (
    _build_model_load_kwargs,
    build_sft_training_rows,
    load_training_config,
    validate_sft_training_config,
    write_sft_smoke_report,
)


def _spark_subprocess_env() -> dict[str, str]:
    env = dict(os.environ)
    python_dir = str(Path(sys.executable).resolve().parent)
    env["PATH"] = f"{python_dir}:{env.get('PATH', '')}" if env.get("PATH") else python_dir
    include_root = ROOT / ".local_pkgs" / "python312dev" / "extracted" / "usr" / "include"
    if include_root.exists():
        include_value = f"{include_root / 'python3.12'}:{include_root}"
        if env.get("C_INCLUDE_PATH"):
            include_value = f"{include_value}:{env['C_INCLUDE_PATH']}"
        env["C_INCLUDE_PATH"] = include_value
    return env


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._previous_env = {
            "PATH": os.environ.get("PATH"),
            "C_INCLUDE_PATH": os.environ.get("C_INCLUDE_PATH"),
        }
        spark_env = _spark_subprocess_env()
        os.environ["PATH"] = spark_env["PATH"]
        if spark_env.get("C_INCLUDE_PATH"):
            os.environ["C_INCLUDE_PATH"] = spark_env["C_INCLUDE_PATH"]
        required_paths = [
            ROOT / "datasets" / "first_target_transition_train_v1" / "trajectory_dataset_manifest.json",
            ROOT / "datasets" / "first_target_transition_dev_v1" / "trajectory_dataset_manifest.json",
            ROOT / "datasets" / "first_target_sft_train_v1" / "sft_dataset_manifest.json",
            ROOT / "datasets" / "first_target_sft_dev_v1" / "sft_dataset_manifest.json",
        ]
        if not all(path.exists() for path in required_paths):
            subprocess.run([sys.executable, str(ROOT / "scripts" / "build_first_target_training_assets.py")], check=True, env=_spark_subprocess_env())

    @classmethod
    def tearDownClass(cls) -> None:
        previous_path = cls._previous_env.get("PATH")
        previous_include = cls._previous_env.get("C_INCLUDE_PATH")
        if previous_path is None:
            os.environ.pop("PATH", None)
        else:
            os.environ["PATH"] = previous_path
        if previous_include is None:
            os.environ.pop("C_INCLUDE_PATH", None)
        else:
            os.environ["C_INCLUDE_PATH"] = previous_include

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

    def test_validate_spark_qlora_smoke_config(self) -> None:
        config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_qlora_spark_smoke.json"
        config = load_training_config(config_path)
        validation = validate_sft_training_config(ROOT, config)
        self.assertEqual(validation["status"], "ok")
        self.assertEqual(validation["adapter_mode"], "qlora")

    def test_write_sft_smoke_report(self) -> None:
        config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
        out_path = self.tmp_root / "smoke_sft_report.json"
        write_sft_smoke_report(ROOT, config_path, out_path)
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["config_id"], "cfg/sft/qwen32b/debug_repair_lora/v1")

    def test_build_sft_training_rows(self) -> None:
        config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_qlora_spark_smoke.json"
        config = load_training_config(config_path)
        rows = build_sft_training_rows(ROOT, config)
        self.assertEqual(len(rows["train"]), 4)
        self.assertEqual(len(rows["dev"]), 1)
        self.assertIn("### Prompt", rows["train"][0]["text"])
        self.assertIn("### Response", rows["train"][0]["text"])

    def test_build_model_load_kwargs_prefers_low_cpu_mem_usage(self) -> None:
        kwargs = _build_model_load_kwargs("qlora", use_cuda=False)
        self.assertTrue(kwargs["low_cpu_mem_usage"])
        self.assertIn("quantization_config", kwargs)
        self.assertNotIn("offload_state_dict", kwargs)

    def test_build_model_load_kwargs_adds_cuda_memory_controls(self) -> None:
        with patch("torch.cuda.get_device_properties") as get_device_properties:
            get_device_properties.return_value.total_memory = 52 * 1024**3
            kwargs = _build_model_load_kwargs("qlora", use_cuda=True)
        self.assertEqual(kwargs["device_map"], "auto")
        self.assertTrue(kwargs["offload_state_dict"])
        self.assertEqual(kwargs["max_memory"], {0: "48GiB", "cpu": "24GiB"})

    def test_build_dataset_governance_report_script(self) -> None:
        subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dataset_governance_report.py")], check=True, env=_spark_subprocess_env())
        payload = json.loads((ROOT / "artifacts" / "training" / "dataset_governance_report.json").read_text(encoding="utf-8"))
        self.assertEqual(payload["report_id"], "report/dataset_governance/v1")
        self.assertEqual(payload["trajectory_datasets"]["train"]["usable_negative_episode_count"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["task_verb_counts"]["debug"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["operator_family_counts"]["reduction_sum"], 2)
        self.assertEqual(payload["sft_datasets"]["train"]["patch_bearing_example_count"], 4)

    def test_run_scripted_rollout_suite(self) -> None:
        existing_report_path = ROOT / "artifacts" / "training" / "heldout_scripted_baseline_v1" / "rollout_report.json"
        if existing_report_path.exists():
            payload = json.loads(existing_report_path.read_text(encoding="utf-8"))
            report = RLRolloutConfig.model_validate(
                json.loads((ROOT / "configs" / "training" / "rollout_debug_repair_heldout_v1.json").read_text(encoding="utf-8"))
            )
            self.assertEqual(payload["config_id"], report.config_id)
            self.assertEqual(payload["task_count"], 3)
            self.assertGreaterEqual(payload["patch_bearing_count"], 2)
            self.assertEqual(payload["success_count"], 3)
            return

        config_path = ROOT / "configs" / "training" / "rollout_debug_repair_v1.json"
        config = RLRolloutConfig.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
        report = run_scripted_rollout_suite(ROOT, config, self.tmp_root / "rollout_report")
        self.assertEqual(report.task_count, 3)
        self.assertGreaterEqual(report.patch_bearing_count, 2)
        self.assertTrue((self.tmp_root / "rollout_report" / "rollout_report.json").exists())


if __name__ == "__main__":
    unittest.main()
