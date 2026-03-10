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
from gpu_cockpit.engine.knowledge import build_knowledge_index, query_knowledge, retrieve_similar_for_task
from gpu_cockpit.engine.trajectory import export_episode_dataset


class KnowledgeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.out_dir = ROOT / "tests" / "tmp_knowledge_index"
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir, ignore_errors=True)

    def tearDown(self) -> None:
        if self.out_dir.exists():
            shutil.rmtree(self.out_dir, ignore_errors=True)

    def test_build_index_and_query_operator_docs(self) -> None:
        manifest_path = build_knowledge_index(ROOT, out_dir=self.out_dir)
        self.assertTrue(manifest_path.exists())
        rows = query_knowledge(ROOT, query="memory bound kv cache", limit=5, index_dir=self.out_dir)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any("kv_cache_gather" in json_row["entry_id"] or "memory-bound" in json_row["title"].lower() for json_row in rows))

    def test_retrieve_similar_for_task(self) -> None:
        build_knowledge_index(ROOT, out_dir=self.out_dir)
        rows = retrieve_similar_for_task(ROOT, "task/kv_cache_gather/eval/v1", limit=5, index_dir=self.out_dir)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(row.get("operator_family") == "kv_cache_gather" for row in rows))

    def test_query_can_mix_docs_and_run_examples(self) -> None:
        build_knowledge_index(ROOT, out_dir=self.out_dir)
        rows = query_knowledge(ROOT, query="amd memory bound run example", limit=10, index_dir=self.out_dir, prefer_mixed=True)
        kinds = {row["kind"] for row in rows}
        self.assertIn("knowledge_doc", kinds)
        self.assertIn("run_example", kinds)

    def test_query_can_filter_by_verb(self) -> None:
        build_knowledge_index(ROOT, out_dir=self.out_dir)
        rows = query_knowledge(ROOT, query="repair broken reduction mask", verb="debug", limit=10, index_dir=self.out_dir)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(any(row.get("metadata", {}).get("verb") == "debug" or row.get("metadata", {}).get("task_verb") == "debug" for row in rows))

    def test_golden_mixed_retrieval_fixture(self) -> None:
        build_knowledge_index(ROOT, out_dir=self.out_dir)
        fixture_path = ROOT / "tests" / "golden_retrieval" / "mixed_attention_reformulate_query_v1.json"
        fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
        rows = query_knowledge(
            ROOT,
            query=fixture["query"],
            verb=fixture.get("verb"),
            prefer_mixed=bool(fixture["prefer_mixed"]),
            limit=int(fixture["limit"]),
            index_dir=self.out_dir,
        )
        kinds = {row["kind"] for row in rows}
        for expected_kind in fixture["expected_kinds"]:
            self.assertIn(expected_kind, kinds)
        paths = [row["path"] for row in rows]
        for fragment in fixture["expected_path_fragments"]:
            self.assertTrue(any(fragment in path for path in paths))

    def test_query_prefers_patch_bearing_repair_examples(self) -> None:
        repo_root = self.out_dir / "repo_root"
        shutil.copytree(ROOT / "workloads", repo_root / "workloads")
        shutil.copytree(ROOT / "knowledge", repo_root / "knowledge")
        episode = run_scripted_reference_episode(
            repo_root,
            "task/reduction_debug/eval/v1",
            ["python3", "workloads/reference/triton_row_sum_debug_candidate.py", "--benchmark-repeats", "2"],
            section="quality",
            include_build=True,
            triton_build_spec="workloads/reference/triton_row_sum_repaired_kernel.py:get_build_spec",
        )
        export_episode_dataset([episode], repo_root / "datasets" / "patch_examples", policy_id="scripted_reference_v1", split="seed")
        build_knowledge_index(repo_root, out_dir=self.out_dir)
        rows = query_knowledge(repo_root, query="repair patch reduction bug fix", verb="debug", limit=10, index_dir=self.out_dir, prefer_mixed=True)
        self.assertGreaterEqual(len(rows), 1)
        self.assertTrue(
            any(
                row["kind"] in {"run_example", "episode_example"} and row.get("metadata", {}).get("patch_present") is True
                for row in rows
            )
        )
