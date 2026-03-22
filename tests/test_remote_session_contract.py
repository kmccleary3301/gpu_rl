from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts.remote_session import ArtifactTransferPolicy, RemoteSessionIdentity, RemoteSyncPolicy
from gpu_cockpit.executors.base import CommandResult
from gpu_cockpit.executors.local_host_remote_session import LocalHostRemoteSession
from gpu_cockpit.executors.remote_session import RemoteWorkspaceSession


class FakeRemoteWorkspaceSession(RemoteWorkspaceSession):
    def __init__(self) -> None:
        self._session_id = "remote_test_session"
        self.terminated = False

    @property
    def session_id(self) -> str:
        return self._session_id

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        del cwd, env, timeout
        return CommandResult(command=list(command), exit_code=0, stdout="ok", stderr="", duration_ms=1)

    def put_file(self, local_path: Path, remote_path: Path) -> None:
        del remote_path
        if not local_path.exists():
            raise FileNotFoundError(local_path)

    def get_file(self, remote_path: Path, local_path: Path) -> None:
        del remote_path
        local_path.write_text("pulled\n", encoding="utf-8")

    def sync_tree(
        self,
        local_root: Path,
        remote_root: Path,
        *,
        allowlist_roots: list[str] | None = None,
        exclude_globs: list[str] | None = None,
    ) -> list[str]:
        del remote_root, exclude_globs
        return [str(local_root / item) for item in (allowlist_roots or [])]

    def terminate(self) -> None:
        self.terminated = True


class RemoteSessionContractTests(unittest.TestCase):
    def test_remote_session_identity_defaults(self) -> None:
        identity = RemoteSessionIdentity(
            session_id="sess_001",
            executor_kind="modal_stub",
            workspace_root="/remote/workspace",
            cwd="/remote/workspace",
            environment={"CUDA_VISIBLE_DEVICES": "0"},
        )
        self.assertEqual(identity.source_compatible_local_executor, "local_host")
        self.assertIn("gpu_cockpit", identity.sync_policy.allowlist_roots)
        self.assertIn("runs", identity.artifact_policy.run_bundle_roots)
        self.assertGreater(identity.timeout_policy.command_timeout_s, 0)

    def test_remote_session_identity_round_trip(self) -> None:
        identity = RemoteSessionIdentity(
            session_id="sess_002",
            executor_kind="neutral_remote",
            workspace_root="/remote/workspace",
            cwd="/remote/workspace/subdir",
        )
        restored = RemoteSessionIdentity.model_validate(identity.model_dump(mode="json"))
        self.assertEqual(restored.cwd, "/remote/workspace/subdir")
        self.assertEqual(restored.executor_kind, "neutral_remote")

    def test_default_sync_allowlist_and_artifact_policy(self) -> None:
        sync_policy = RemoteSyncPolicy()
        artifact_policy = ArtifactTransferPolicy()
        self.assertIn("gpu_cockpit", sync_policy.allowlist_roots)
        self.assertIn("artifacts", artifact_policy.report_roots)
        self.assertIn("runs/*/replay", artifact_policy.replay_roots)

    def test_remote_workspace_session_surface(self) -> None:
        session = FakeRemoteWorkspaceSession()
        result = session.run(["echo", "ok"])
        self.assertEqual(result.exit_code, 0)
        self.assertEqual(result.stdout, "ok")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            source = tmp_root / "source.txt"
            source.write_text("payload\n", encoding="utf-8")
            pulled = tmp_root / "pulled.txt"

            session.put_file(source, Path("/remote/source.txt"))
            session.get_file(Path("/remote/source.txt"), pulled)
            synced = session.sync_tree(tmp_root, Path("/remote"), allowlist_roots=["gpu_cockpit", "workloads"])

            self.assertEqual(pulled.read_text(encoding="utf-8"), "pulled\n")
            self.assertEqual(len(synced), 2)

        session.terminate()
        self.assertTrue(session.terminated)

    def test_local_host_remote_session_bridge(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            workspace_root.mkdir(parents=True, exist_ok=True)
            source_root = Path(tmp_dir) / "source"
            source_root.mkdir(parents=True, exist_ok=True)
            (source_root / "gpu_cockpit").mkdir()
            (source_root / "gpu_cockpit" / "bridge.txt").write_text("bridge\n", encoding="utf-8")
            (source_root / "README.md").write_text("readme\n", encoding="utf-8")
            local_file = source_root / "local.txt"
            local_file.write_text("payload\n", encoding="utf-8")

            session = LocalHostRemoteSession(session_id="local_bridge", workspace_root=workspace_root)
            copied = session.sync_tree(
                source_root,
                Path("."),
                allowlist_roots=["gpu_cockpit", "README.md"],
                exclude_globs=["*.pyc"],
            )
            self.assertEqual(sorted(copied), ["README.md", "gpu_cockpit"])
            self.assertTrue((workspace_root / "gpu_cockpit" / "bridge.txt").exists())

            session.put_file(local_file, Path("uploads/local.txt"))
            self.assertEqual((workspace_root / "uploads" / "local.txt").read_text(encoding="utf-8"), "payload\n")

            pulled = Path(tmp_dir) / "pulled.txt"
            session.get_file(Path("uploads/local.txt"), pulled)
            self.assertEqual(pulled.read_text(encoding="utf-8"), "payload\n")

            result = session.run(["python3", "-c", "print('bridge_ok')"])
            self.assertEqual(result.exit_code, 0)
            self.assertIn("bridge_ok", result.stdout)

    def test_local_host_source_compatibility_expectation_is_explicit(self) -> None:
        identity = RemoteSessionIdentity(
            session_id="sess_003",
            executor_kind="local_bridge",
            workspace_root="/tmp/workspace",
            cwd="/tmp/workspace",
        )
        self.assertEqual(identity.source_compatible_local_executor, "local_host")


if __name__ == "__main__":
    unittest.main()
