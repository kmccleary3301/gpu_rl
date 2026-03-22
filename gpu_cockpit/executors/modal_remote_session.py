from __future__ import annotations

import atexit
import fnmatch
import os
import posixpath
import shlex
import tarfile
import tempfile
import time
from io import BytesIO
from pathlib import Path, PurePosixPath
from types import ModuleType

from gpu_cockpit.contracts.remote_session import ArtifactTransferPolicy, RemoteSyncPolicy, RemoteTimeoutPolicy
from gpu_cockpit.executors.base import CommandExecutor, CommandResult
from gpu_cockpit.executors.remote_session import RemoteWorkspaceSession


def _import_modal() -> ModuleType:
    try:
        import modal  # type: ignore[import-not-found]
    except ImportError as exc:  # pragma: no cover - exercised via tests with a fake module
        raise RuntimeError("Modal support requires the optional 'modal' dependency.") from exc
    return modal


def _env_int(name: str, default: int | None = None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float | None = None) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _iter_local_files(root: Path, excludes: tuple[str, ...]) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        if any(fnmatch.fnmatch(relative, pattern) for pattern in excludes):
            continue
        rows.append((path, relative))
    return rows


def _top_level_roots(artifact_policy: ArtifactTransferPolicy, sync_policy: RemoteSyncPolicy) -> list[str]:
    roots: set[str] = set(sync_policy.artifact_pull_roots)
    for pattern in (
        artifact_policy.run_bundle_roots
        + artifact_policy.patch_roots
        + artifact_policy.replay_roots
        + artifact_policy.report_roots
    ):
        first = pattern.split("/", 1)[0].strip()
        if first:
            roots.add(first)
    return sorted(roots)


def _ensure_safe_members(archive: tarfile.TarFile) -> None:
    for member in archive.getmembers():
        target = PurePosixPath(member.name)
        if target.is_absolute() or ".." in target.parts:
            raise RuntimeError(f"Unsafe tar member from remote session: {member.name}")


class ModalWorkspaceSession(RemoteWorkspaceSession):
    def __init__(
        self,
        *,
        session_id: str,
        workspace_root: Path,
        cwd: Path | None = None,
        remote_workspace_root: str = "/workspace",
        app_name: str | None = None,
        environment_name: str | None = None,
        sync_policy: RemoteSyncPolicy | None = None,
        timeout_policy: RemoteTimeoutPolicy | None = None,
        artifact_policy: ArtifactTransferPolicy | None = None,
    ) -> None:
        self._session_id = session_id
        self.local_workspace_root = workspace_root.resolve()
        self.cwd = cwd.resolve() if cwd is not None else self.local_workspace_root
        self.remote_workspace_root = PurePosixPath(remote_workspace_root)
        self.sync_policy = sync_policy or RemoteSyncPolicy()
        self.timeout_policy = timeout_policy or RemoteTimeoutPolicy()
        self.artifact_policy = artifact_policy or ArtifactTransferPolicy()
        self._terminated = False
        self._modal = _import_modal()
        self._app_name = app_name or os.environ.get("GPU_COCKPIT_MODAL_APP_NAME", "gpu-cockpit-mvp")
        self._environment_name = environment_name or os.environ.get("GPU_COCKPIT_MODAL_ENVIRONMENT")
        self._sandbox = self._create_sandbox()
        atexit.register(self.terminate)

    @property
    def session_id(self) -> str:
        return self._session_id

    def _build_image(self):
        python_version = os.environ.get("GPU_COCKPIT_MODAL_PYTHON_VERSION", "3.12")
        pip_packages = shlex.split(os.environ.get("GPU_COCKPIT_MODAL_PIP_PACKAGES", "pydantic>=2.8,<3"))
        image = self._modal.Image.debian_slim(python_version=python_version)
        image = image.apt_install("bash", "coreutils", "findutils", "tar")
        if pip_packages:
            image = image.pip_install(*pip_packages)
        image = image.add_local_dir(
            self.local_workspace_root,
            remote_path=str(self.remote_workspace_root),
            ignore=list(self.sync_policy.exclude_globs),
        )
        return image.env({"PYTHONPATH": str(self.remote_workspace_root)}).workdir(str(self.remote_workspace_root))

    def _create_sandbox(self):
        app = self._modal.App.lookup(
            self._app_name,
            create_if_missing=True,
            environment_name=self._environment_name,
        )
        gpu = os.environ.get("GPU_COCKPIT_MODAL_GPU") or None
        cpu = _env_float("GPU_COCKPIT_MODAL_CPU")
        memory = _env_int("GPU_COCKPIT_MODAL_MEMORY_MB")
        return self._modal.Sandbox.create(
            "sleep",
            "infinity",
            app=app,
            name=self._session_id,
            image=self._build_image(),
            env={"PYTHONPATH": str(self.remote_workspace_root)},
            timeout=self.timeout_policy.session_ttl_s,
            idle_timeout=self.timeout_policy.idle_timeout_s,
            workdir=str(self.remote_workspace_root),
            gpu=gpu,
            cpu=cpu,
            memory=memory,
            environment_name=self._environment_name,
        )

    def _remote_path(self, path: Path | str) -> str:
        if isinstance(path, str):
            text = path
        else:
            text = path.as_posix()
        if text in {"", "."}:
            return str(self.remote_workspace_root)
        if text.startswith("/"):
            return text
        return str(self.remote_workspace_root / PurePosixPath(text))

    def _remote_cwd(self, cwd: Path | None) -> str:
        target = cwd.resolve() if cwd is not None else self.cwd
        try:
            relative = target.relative_to(self.local_workspace_root)
            return self._remote_path(Path(relative.as_posix()))
        except ValueError:
            return str(target)

    def run(
        self,
        command: list[str],
        *,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        if not command:
            raise RuntimeError("Modal session cannot execute an empty command.")
        started = time.perf_counter()
        process = self._sandbox.exec(
            *command,
            workdir=self._remote_cwd(cwd),
            env=env,
            timeout=int(timeout) if timeout is not None else self.timeout_policy.command_timeout_s,
        )
        exit_code = process.wait()
        stdout = "" if process.stdout is None else process.stdout.read()
        stderr = "" if process.stderr is None else process.stderr.read()
        duration_ms = int((time.perf_counter() - started) * 1000)
        return CommandResult(command=list(command), exit_code=exit_code, stdout=stdout, stderr=stderr, duration_ms=duration_ms)

    def put_file(self, local_path: Path, remote_path: Path) -> None:
        source = local_path.resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        destination = self._remote_path(remote_path)
        parent = posixpath.dirname(destination)
        if parent:
            self._sandbox.mkdir(parent, parents=True)
        with source.open("rb") as handle:
            payload = handle.read()
        with self._sandbox.open(destination, "wb") as remote_handle:
            remote_handle.write(payload)

    def get_file(self, remote_path: Path, local_path: Path) -> None:
        source = self._remote_path(remote_path)
        destination = local_path.resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        with self._sandbox.open(source, "rb") as remote_handle:
            payload = remote_handle.read()
        with destination.open("wb") as handle:
            handle.write(payload)

    def sync_tree(
        self,
        local_root: Path,
        remote_root: Path,
        *,
        allowlist_roots: list[str] | None = None,
        exclude_globs: list[str] | None = None,
    ) -> list[str]:
        copied: list[str] = []
        source_root = local_root.resolve()
        excludes = tuple(exclude_globs or [])
        remote_base = self._remote_path(remote_root)
        self._sandbox.mkdir(remote_base, parents=True)
        for relative in allowlist_roots or []:
            source = source_root / relative
            if not source.exists():
                continue
            relative_posix = Path(relative).as_posix()
            if any(fnmatch.fnmatch(relative_posix, pattern) for pattern in excludes):
                continue
            if source.is_file():
                self.put_file(source, Path(remote_base) / Path(relative_posix))
            else:
                remote_dir = Path(remote_base) / Path(relative_posix)
                self._sandbox.mkdir(self._remote_path(remote_dir), parents=True)
                for child, child_relative in _iter_local_files(source, excludes):
                    self.put_file(child, Path(remote_base) / Path(relative_posix) / Path(child_relative))
            copied.append(relative_posix)
        return copied

    def pull_tree(self, remote_path: str, local_root: Path) -> bool:
        remote_target = self._remote_path(remote_path)
        remote_parent = posixpath.dirname(remote_target)
        remote_name = posixpath.basename(remote_target)
        remote_archive = f"/tmp/{self._session_id}_{remote_name.replace('/', '_')}.tar"
        command = [
            "bash",
            "-lc",
            (
                f"if [ -e {shlex.quote(remote_target)} ]; then "
                f"tar -C {shlex.quote(remote_parent or '/')} -cf {shlex.quote(remote_archive)} {shlex.quote(remote_name)}; "
                "else exit 42; fi"
            ),
        ]
        result = self.run(command, cwd=None, timeout=self.timeout_policy.command_timeout_s)
        if result.exit_code == 42:
            return False
        if result.exit_code != 0:
            raise RuntimeError(f"Failed to archive remote path {remote_target}: {result.stderr.strip()}")
        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "remote_pull.tar"
            self.get_file(Path(remote_archive), archive_path)
            with tarfile.open(archive_path, "r") as archive:
                _ensure_safe_members(archive)
                archive.extractall(local_root, filter="data")
        self.run(["rm", "-f", remote_archive], cwd=None, timeout=30)
        return True

    def terminate(self) -> None:
        if self._terminated:
            return
        self._terminated = True
        try:
            self._sandbox.terminate(wait=True)
        except Exception:
            return None


class ModalExecutor(CommandExecutor):
    def __init__(
        self,
        *,
        workspace_root: Path,
        sync_policy: RemoteSyncPolicy | None = None,
        timeout_policy: RemoteTimeoutPolicy | None = None,
        artifact_policy: ArtifactTransferPolicy | None = None,
    ) -> None:
        self.workspace_root = workspace_root.resolve()
        self.sync_policy = sync_policy or RemoteSyncPolicy()
        self.timeout_policy = timeout_policy or RemoteTimeoutPolicy()
        self.artifact_policy = artifact_policy or ArtifactTransferPolicy()
        self.session = ModalWorkspaceSession(
            session_id=f"modal_{int(time.time())}",
            workspace_root=self.workspace_root,
            sync_policy=self.sync_policy,
            timeout_policy=self.timeout_policy,
            artifact_policy=self.artifact_policy,
        )

    def run(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        self.session.sync_tree(
            self.workspace_root,
            Path("."),
            allowlist_roots=self.sync_policy.allowlist_roots,
            exclude_globs=self.sync_policy.exclude_globs,
        )
        result = self.session.run(command, cwd=cwd or self.workspace_root, env=env, timeout=timeout)
        for root in _top_level_roots(self.artifact_policy, self.sync_policy):
            self.session.pull_tree(root, self.workspace_root)
        return result

    def terminate(self) -> None:
        self.session.terminate()
