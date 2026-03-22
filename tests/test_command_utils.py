from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.command_utils import local_python_build_env, normalize_python_command


class CommandUtilsTests(unittest.TestCase):
    def test_normalize_python_command_rewrites_python3_to_active_interpreter(self) -> None:
        command = ["python3", "workloads/reference/triton_attention_score_baseline.py", "--benchmark-repeats", "2"]

        normalized = normalize_python_command(command)

        self.assertEqual(normalized[0], sys.executable)
        self.assertEqual(normalized[1:], command[1:])

    def test_normalize_python_command_leaves_non_python_commands_unchanged(self) -> None:
        command = ["bash", "scripts/run_with_spark_env.sh"]

        normalized = normalize_python_command(command)

        self.assertEqual(normalized, command)

    def test_local_python_build_env_returns_repo_local_include_path(self) -> None:
        env = local_python_build_env(ROOT)

        include_value = env.get("C_INCLUDE_PATH")

        self.assertIsNotNone(include_value)
        assert include_value is not None
        self.assertIn(str(ROOT / ".local_pkgs" / "python312dev" / "extracted" / "usr" / "include" / "python3.12"), include_value)

    def test_local_python_build_env_returns_empty_when_headers_missing(self) -> None:
        tmp_root = ROOT / "tests" / "tmp_command_utils"
        if tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)
        tmp_root.mkdir(parents=True, exist_ok=True)
        try:
            env = local_python_build_env(tmp_root)
            self.assertEqual(env, {})
        finally:
            shutil.rmtree(tmp_root, ignore_errors=True)
