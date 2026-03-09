from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.task_registry import TaskRegistry


class TaskRegistryTests(unittest.TestCase):
    def test_load_registered_smoke_task(self) -> None:
        registry = TaskRegistry(ROOT)
        task = registry.get("task/smoke/diagnose/v1")
        self.assertEqual(task.operator_family, "smoke")
        self.assertIn("triton", task.allowed_backends)


if __name__ == "__main__":
    unittest.main()
