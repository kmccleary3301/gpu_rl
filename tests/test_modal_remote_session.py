from __future__ import annotations

import importlib
import io
import tarfile
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch


class _FakeStream:
    def __init__(self, payload: str | bytes) -> None:
        self.payload = payload

    def read(self):
        return self.payload


class _FakeProcess:
    def __init__(self, exit_code: int = 0, stdout: str = "", stderr: str = "") -> None:
        self._exit_code = exit_code
        self.stdout = _FakeStream(stdout)
        self.stderr = _FakeStream(stderr)

    def wait(self) -> int:
        return self._exit_code


class _FakeSandbox:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.commands: list[dict[str, object]] = []
        self.terminated = False

    def exec(self, *args: str, workdir: str | None = None, env: dict[str, str] | None = None, timeout: int | None = None, **_: object):
        self.commands.append({"args": list(args), "workdir": workdir, "env": env, "timeout": timeout})
        if len(args) >= 3 and args[0] == "bash" and args[1] == "-lc" and "tar -C" in args[2]:
            command = args[2]
            if "artifacts" not in command and "runs" not in command:
                return _FakeProcess(exit_code=42)
            archive_path = command.split(" -cf ", 1)[1].split(" ", 1)[0].strip("'\"")
            payload = io.BytesIO()
            with tarfile.open(fileobj=payload, mode="w") as archive:
                for path, content in self.files.items():
                    if path.startswith("/workspace/artifacts/") or path.startswith("/workspace/runs/"):
                        info = tarfile.TarInfo(path.replace("/workspace/", "", 1))
                        info.size = len(content)
                        archive.addfile(info, io.BytesIO(content))
            self.files[archive_path] = payload.getvalue()
            return _FakeProcess()
        if list(args[:2]) == ["rm", "-f"]:
            self.files.pop(args[2], None)
            return _FakeProcess()
        return _FakeProcess(stdout="modal_ok\n")

    def mkdir(self, path: str, parents: bool = False) -> None:
        del parents
        self.files.setdefault(path.rstrip("/") + "/.dir", b"")

    def open(self, path: str, mode: str = "r"):
        if "w" in mode:
            buffer = io.BytesIO()
            sandbox = self

            class _Writer:
                def __enter__(self):
                    return buffer

                def __exit__(self, exc_type, exc, tb) -> None:
                    del exc_type, exc, tb
                    sandbox.files[path] = buffer.getvalue()

            return _Writer()
        payload = self.files.get(path, b"")
        if "b" in mode:
            stream = io.BytesIO(payload)
        else:
            stream = io.StringIO(payload.decode("utf-8"))

        class _Reader:
            def __enter__(self):
                return stream

            def __exit__(self, exc_type, exc, tb) -> None:
                del exc_type, exc, tb

        return _Reader()

    def terminate(self, *, wait: bool = False) -> int | None:
        del wait
        self.terminated = True
        return 0


class _FakeApp:
    @staticmethod
    def lookup(name: str, *, create_if_missing: bool = False, environment_name: str | None = None):
        del name, create_if_missing, environment_name
        return object()


class _FakeImage:
    def apt_install(self, *args, **kwargs):
        del args, kwargs
        return self

    def pip_install(self, *args, **kwargs):
        del args, kwargs
        return self

    def add_local_dir(self, *args, **kwargs):
        del args, kwargs
        return self

    def env(self, *args, **kwargs):
        del args, kwargs
        return self

    def workdir(self, *args, **kwargs):
        del args, kwargs
        return self


class _FakeModalModule(types.SimpleNamespace):
    def __init__(self) -> None:
        self._sandbox = _FakeSandbox()
        super().__init__(
            App=_FakeApp,
            Image=types.SimpleNamespace(debian_slim=lambda python_version=None: _FakeImage()),
            Sandbox=types.SimpleNamespace(create=lambda *args, **kwargs: self._sandbox),
        )


class ModalRemoteSessionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = importlib.import_module("gpu_cockpit.executors.modal_remote_session")

    def test_session_sync_put_get_and_terminate(self) -> None:
        fake_modal = _FakeModalModule()
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            workspace_root.mkdir()
            (workspace_root / "gpu_cockpit").mkdir()
            source = workspace_root / "gpu_cockpit" / "bridge.txt"
            source.write_text("bridge\n", encoding="utf-8")
            with patch.object(self.module, "_import_modal", return_value=fake_modal):
                session = self.module.ModalWorkspaceSession(session_id="sess_modal", workspace_root=workspace_root)
                copied = session.sync_tree(
                    workspace_root,
                    Path("."),
                    allowlist_roots=["gpu_cockpit"],
                    exclude_globs=["*.pyc"],
                )
                self.assertEqual(copied, ["gpu_cockpit"])
                self.assertEqual(fake_modal._sandbox.files["/workspace/gpu_cockpit/bridge.txt"], b"bridge\n")

                local_payload = workspace_root / "local.txt"
                local_payload.write_text("payload\n", encoding="utf-8")
                session.put_file(local_payload, Path("uploads/local.txt"))

                pulled = workspace_root / "pulled.txt"
                session.get_file(Path("uploads/local.txt"), pulled)
                self.assertEqual(pulled.read_text(encoding="utf-8"), "payload\n")

                session.terminate()
                self.assertTrue(fake_modal._sandbox.terminated)

    def test_executor_normalizes_command_and_pulls_artifacts(self) -> None:
        fake_modal = _FakeModalModule()
        fake_modal._sandbox.files["/workspace/artifacts/modal/result.json"] = b"{\"status\":\"ok\"}\n"
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir) / "workspace"
            workspace_root.mkdir()
            (workspace_root / "gpu_cockpit").mkdir()
            (workspace_root / "gpu_cockpit" / "__init__.py").write_text("", encoding="utf-8")
            with patch.object(self.module, "_import_modal", return_value=fake_modal):
                executor = self.module.ModalExecutor(workspace_root=workspace_root)
                result = executor.run(["python", "-c", "print('hi')"], env={"GPU_COCKPIT_TEST": "1"}, cwd=workspace_root)
                self.assertEqual(result.exit_code, 0)
                self.assertIn("modal_ok", result.stdout)
                self.assertTrue((workspace_root / "artifacts" / "modal" / "result.json").exists())
                command_rows = [row for row in fake_modal._sandbox.commands if row["args"] == ["python", "-c", "print('hi')"]]
                self.assertEqual(len(command_rows), 1)
                self.assertEqual(command_rows[0]["workdir"], "/workspace")
                self.assertEqual(command_rows[0]["env"], {"GPU_COCKPIT_TEST": "1"})
                executor.terminate()


if __name__ == "__main__":
    unittest.main()
