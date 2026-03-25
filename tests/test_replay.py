from __future__ import annotations

from pathlib import Path
import shutil
import sys
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.replay import export_proof_bundle, validate_run_bundle
from gpu_cockpit.engine.runner import run_task
from gpu_cockpit.engine.patching import apply_patch_candidate, branch_candidate, promote_candidate


class ReplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_root = ROOT / "tests" / "tmp_replay"
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)
        self.tmp_root.mkdir(parents=True, exist_ok=True)
        shutil.copytree(ROOT / "workloads", self.tmp_root / "workloads")

    def tearDown(self) -> None:
        if self.tmp_root.exists():
            shutil.rmtree(self.tmp_root)

    def test_validate_run_bundle_passes_with_replay_pack(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('replay smoke')"],
            trace_system=False,
        )
        payload = validate_run_bundle(self.tmp_root, str(run_dir))
        self.assertEqual(payload["status"], "ok")
        self.assertTrue(payload["checks"]["replay_pack_present"])
        self.assertEqual(payload["missing_artifacts"], [])
        self.assertIn("evidence_quality", payload)
        self.assertGreater(payload["evidence_quality"]["overall_score"], 0.0)

    def test_export_proof_bundle_creates_zip(self) -> None:
        run_dir = run_task(
            root=self.tmp_root,
            task_ref="task/smoke/diagnose/v1",
            command=["python3", "-c", "print('proof bundle smoke')"],
            trace_system=False,
        )
        bundle_path = export_proof_bundle(self.tmp_root, str(run_dir))
        self.assertTrue(bundle_path.exists())
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
        self.assertIn("summary.json", names)
        self.assertIn("replay/replay_pack.json", names)

    def test_validate_checked_in_amd_bundle_fixture(self) -> None:
        payload = validate_run_bundle(ROOT, str(ROOT / "tests" / "golden_runs" / "amd_bundle_v1"))
        self.assertEqual(payload["status"], "ok")
        self.assertIn("profiles/kernel/summary.md", payload["required_artifacts"])
        self.assertFalse(payload["evidence_quality"]["benchmark_reporting"]["eligible"])

    def test_validate_checked_in_debug_and_reformulate_fixtures(self) -> None:
        debug_payload = validate_run_bundle(ROOT, str(ROOT / "tests" / "golden_runs" / "reduction_debug_bundle_v1"))
        reformulate_payload = validate_run_bundle(ROOT, str(ROOT / "tests" / "golden_runs" / "attention_reformulate_eval_bundle_v1"))
        self.assertEqual(debug_payload["status"], "ok")
        self.assertEqual(reformulate_payload["status"], "ok")
        self.assertEqual(debug_payload["evidence_quality"]["training_example_kind"], "unusable")
        self.assertEqual(reformulate_payload["evidence_quality"]["training_example_kind"], "benchmark_only")

    def test_patch_run_replay_pack_persists_candidate_lineage(self) -> None:
        _, applied_patch, candidate_state, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=(self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8"),
            intent="repair the row-sum kernel mask",
            expected_effect="restore the omitted column",
            patch_kind="bug_fix",
            transition_kind="repaired",
        )
        run_dir = self.tmp_root / "runs"
        patch_runs = sorted(run_dir.glob("patch_*"))
        self.assertTrue(patch_runs)
        payload = validate_run_bundle(self.tmp_root, str(patch_runs[-1]))
        replay_pack = payload["replay_pack"]
        self.assertEqual(replay_pack["candidate_id"], candidate_state.candidate_id)
        self.assertEqual(replay_pack["transition_kind"], "repaired")
        self.assertEqual(replay_pack["patch_ref"], "patches/applied_patch.json")
        self.assertEqual(replay_pack["diff_ref"], "patches/unified_diff.patch")
        self.assertEqual(replay_pack["operation_ref"], "candidate/operation.json")
        self.assertEqual(replay_pack["candidate_status"], "patched")
        self.assertEqual(replay_pack["candidate_origin_kind"], "patch")
        self.assertEqual(replay_pack["candidate_operation_kind"], "patch_apply")
        self.assertEqual(replay_pack["candidate_role_group"], "trial")
        self.assertEqual(replay_pack["candidate_tree_depth"], 0)
        self.assertEqual(replay_pack["best_known_candidate_id"], candidate_state.candidate_id)
        self.assertEqual(replay_pack["best_known_candidate_reason"], "candidate_created")
        self.assertEqual(replay_pack["legal_next_actions"], [])
        self.assertEqual(replay_pack["dominated_candidate_ids"], [])
        self.assertEqual(replay_pack["active_candidate_ids"], [candidate_state.candidate_id])
        self.assertEqual(replay_pack["archived_candidate_ids"], [])
        self.assertEqual(replay_pack["source_candidate_id"], None)
        self.assertEqual(replay_pack["sibling_candidate_refs"], [])
        self.assertTrue(payload["checks"]["patch_ref"])
        self.assertTrue(payload["checks"]["diff_ref"])
        self.assertTrue(payload["checks"]["transition_ref"])
        self.assertTrue(payload["checks"]["operation_ref"])
        self.assertTrue(payload["checks"]["candidate_lineage_present"])

    def test_validate_checked_in_patch_transition_fixture(self) -> None:
        payload = validate_run_bundle(ROOT, str(ROOT / "tests" / "golden_runs" / "reduction_debug_patch_transition_v1"))
        self.assertEqual(payload["status"], "ok")
        self.assertEqual(payload["replay_pack"]["transition_kind"], "repaired")
        self.assertEqual(payload["replay_pack"]["patch_ref"], "patches/applied_patch.json")

    def test_export_proof_bundle_for_patch_run_includes_transition_artifacts(self) -> None:
        run_dir, _, _, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=(self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8"),
            intent="repair the row-sum kernel mask",
            expected_effect="restore the omitted column",
            patch_kind="bug_fix",
            transition_kind="repaired",
        )
        bundle_path = export_proof_bundle(self.tmp_root, str(run_dir))
        with zipfile.ZipFile(bundle_path) as archive:
            names = set(archive.namelist())
        self.assertIn("candidate/state.json", names)
        self.assertIn("candidate/transition.json", names)
        self.assertIn("candidate/operation.json", names)
        self.assertIn("patches/applied_patch.json", names)
        self.assertIn("patches/unified_diff.patch", names)

    def test_deeper_candidate_tree_replay_pack_preserves_parentage(self) -> None:
        patch_run_dir, _, patched_candidate, _ = apply_patch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            target_file="workloads/reference/triton_row_sum_broken_kernel.py",
            replacement_text=(self.tmp_root / "workloads" / "reference" / "triton_row_sum_repaired_kernel.py").read_text(encoding="utf-8"),
            intent="repair the row-sum kernel mask",
            expected_effect="restore the omitted column",
            patch_kind="bug_fix",
            transition_kind="repaired",
        )
        branch_run_dir, branched_candidate, _ = branch_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            intent="branch into an alternate candidate line",
            branch_label="alt_branch",
            parent_run_ref=str(patch_run_dir),
            parent_run_id=patch_run_dir.name,
            parent_candidate_id=patched_candidate.candidate_id,
        )
        promote_run_dir, promoted_candidate, _ = promote_candidate(
            self.tmp_root,
            task_ref="task/reduction_debug/eval/v1",
            intent="promote the branched candidate",
            promotion_label="preferred_branch",
            parent_run_ref=str(branch_run_dir),
            parent_run_id=branch_run_dir.name,
            parent_candidate_id=branched_candidate.candidate_id,
        )

        payload = validate_run_bundle(self.tmp_root, str(promote_run_dir))
        replay_pack = payload["replay_pack"]
        self.assertEqual(replay_pack["candidate_id"], promoted_candidate.candidate_id)
        self.assertEqual(replay_pack["candidate_status"], "promoted")
        self.assertEqual(replay_pack["candidate_origin_kind"], "promotion")
        self.assertEqual(replay_pack["candidate_operation_kind"], "promote")
        self.assertEqual(replay_pack["parent_candidate_id"], branched_candidate.candidate_id)
        self.assertEqual(replay_pack["source_candidate_id"], branched_candidate.candidate_id)
        self.assertEqual(replay_pack["candidate_tree_depth"], 2)
        self.assertTrue(payload["checks"]["candidate_lineage_present"])


if __name__ == "__main__":
    unittest.main()
