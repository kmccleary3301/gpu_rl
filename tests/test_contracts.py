from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import CandidateDiffSummary, CandidateOperation, Event, LearningRewardTrace, OptimizeTraceSnapshot, OptimizeTraceSnapshots, RewardLedger, RewardLedgerEntry, RunSpec, TaskSpec
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

    def test_candidate_operation_round_trip(self) -> None:
        operation = CandidateOperation(
            operation_id="op_test_001",
            operation_kind="patch_apply",
            task_id="task/test/debug/v1",
            input_candidate_id="cand_parent",
            output_candidate_id="cand_child",
            diff_summary=CandidateDiffSummary(
                changed_file_count=1,
                changed_line_count=8,
                changed_files=["workloads/reference/example.py"],
                primary_target_file="workloads/reference/example.py",
                diff_ref="candidate/example.diff",
                patch_hash="deadbeef",
                summary="repair the candidate",
            ),
            summary="repair the candidate",
        )
        dumped = operation.model_dump(mode="json")
        restored = CandidateOperation.model_validate(dumped)
        self.assertEqual(restored.operation_kind, "patch_apply")
        self.assertEqual(restored.diff_summary.changed_line_count, 8)

    def test_learning_reward_trace_round_trip(self) -> None:
        trace = LearningRewardTrace(
            task_id="task/test/optimize/v1",
            task_verb="optimize",
            terminal_state="success",
            task_outcome="success",
            trace_usability="trainable_positive",
            task_success=True,
            correctness_passed=True,
            determinism_passed=True,
            anti_hack_passed=True,
            perf_gate="pass",
            compare_used=True,
            reward_components={"task_success": 0.6, "correctness": 0.25},
            shaping_components={"compare_use_bonus": 0.02, "tool_cost": -0.04},
            total_reward=0.83,
            reward_ledger=RewardLedger(
                task_id="task/test/optimize/v1",
                task_verb="optimize",
                task_outcome="success",
                trace_usability="trainable_positive",
                entries=[
                    RewardLedgerEntry(
                        step_index=0,
                        action_type="compare",
                        shaping_components={"compare_use_bonus": 0.02},
                        total_delta=0.02,
                    )
                ],
                total_reward_components={"task_success": 0.6, "correctness": 0.25},
                total_shaping_components={"compare_use_bonus": 0.02, "tool_cost": -0.04},
                total_reward=0.83,
            ),
        )
        snapshots = OptimizeTraceSnapshots(
            compare_snapshots=[
                OptimizeTraceSnapshot(
                    step_index=3,
                    action_type="compare",
                    snapshot_kind="compare",
                    run_id="cmp_001",
                    payload={"candidate_delta_brief": {"lineage_relationship": "same_parent"}},
                )
            ]
        )
        dumped = {"trace": trace.model_dump(mode="json"), "snapshots": snapshots.model_dump(mode="json")}
        restored_trace = LearningRewardTrace.model_validate(dumped["trace"])
        restored_snaps = OptimizeTraceSnapshots.model_validate(dumped["snapshots"])
        self.assertTrue(restored_trace.compare_used)
        self.assertEqual(restored_trace.reward_ledger.trace_usability, "trainable_positive")
        self.assertEqual(restored_snaps.compare_snapshots[0].snapshot_kind, "compare")


if __name__ == "__main__":
    unittest.main()
