from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import Event, RunSpec, TaskSpec
from gpu_cockpit.contracts.common import BudgetSpec, SeedPack
from gpu_cockpit.contracts.task import NumericalPolicy, PerfProtocol
from gpu_cockpit.engine.policies import resolve_policy_pack


class ContractTests(unittest.TestCase):
    def test_run_spec_round_trip(self) -> None:
        run_spec = RunSpec(
            run_id="run_test_001",
            created_at=datetime.now(tz=UTC),
            task_ref="task/test/optimize/v1",
            mode="human",
            target_backend="triton",
            target_vendor="nvidia",
            executor="local_docker",
            policy_pack="balanced",
            budgets=BudgetSpec(
                wall_seconds=60,
                compile_attempts=1,
                bench_runs=1,
                profile_runs=1,
                artifact_mb=64,
            ),
            seed_pack=SeedPack(global_seed=1, input_seed=2),
        )
        dumped = run_spec.model_dump(mode="json")
        restored = RunSpec.model_validate(dumped)
        self.assertEqual(restored.run_id, "run_test_001")
        self.assertEqual(restored.budgets.wall_seconds, 60)

    def test_task_spec_json_schema_exports(self) -> None:
        schema = TaskSpec.model_json_schema()
        self.assertIn("properties", schema)
        self.assertIn("task_id", schema["properties"])
        self.assertIn("schema_version", schema["properties"])

    def test_event_defaults_schema_version(self) -> None:
        event = Event(
            event_id="evt_000000",
            run_id="run_test_001",
            seq=0,
            ts=datetime.now(tz=UTC),
            scope="run",
            kind="started",
        )
        self.assertEqual(event.schema_version, "1.0.0")

    def test_task_spec_defaults_include_deterministic_metadata(self) -> None:
        task = TaskSpec(
            task_id="task/test/diagnose/v1",
            verb="diagnose",
            operator_family="test",
            difficulty="L0",
            prompt="test",
            numerical_policy=NumericalPolicy(dtype_matrix=[], rtol=0.0, atol=0.0),
            perf_protocol=PerfProtocol(warmups=0, repeats=1, timer="wall_clock"),
        )
        self.assertTrue(task.deterministic_mode)
        self.assertEqual(task.feature_requirements, [])

    def test_policy_pack_resolution(self) -> None:
        conservative = resolve_policy_pack("conservative")
        exploratory = resolve_policy_pack("exploratory")
        self.assertLess(conservative.wall_seconds, exploratory.wall_seconds)
        self.assertLess(conservative.artifact_mb, exploratory.artifact_mb)


if __name__ == "__main__":
    unittest.main()
