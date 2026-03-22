from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class RemoteTimeoutPolicy(ContractModel):
    command_timeout_s: int = Field(default=900, ge=1)
    idle_timeout_s: int = Field(default=1800, ge=1)
    session_ttl_s: int = Field(default=14400, ge=1)


class RemoteSyncPolicy(ContractModel):
    allowlist_roots: list[str] = Field(
        default_factory=lambda: [
            "gpu_cockpit",
            "workloads",
            "configs",
            "knowledge",
            "scripts",
            "tests",
            "docs",
            "pyproject.toml",
            "README.md",
        ]
    )
    exclude_globs: list[str] = Field(
        default_factory=lambda: [
            ".git/**",
            ".venv/**",
            ".local_pkgs/**",
            "artifacts/**",
            "runs/**",
            "__pycache__/**",
            "*.pyc",
        ]
    )
    artifact_pull_roots: list[str] = Field(
        default_factory=lambda: [
            "runs",
            "artifacts",
        ]
    )
    notes: list[str] = Field(default_factory=list)


class ArtifactTransferPolicy(ContractModel):
    run_bundle_roots: list[str] = Field(default_factory=lambda: ["runs"])
    patch_roots: list[str] = Field(default_factory=lambda: ["runs/*/patches", "runs/*/candidate"])
    replay_roots: list[str] = Field(default_factory=lambda: ["runs/*/replay"])
    report_roots: list[str] = Field(default_factory=lambda: ["artifacts"])
    notes: list[str] = Field(default_factory=list)


class RemoteSessionIdentity(ContractModel):
    session_id: str
    executor_kind: str
    workspace_root: str
    cwd: str
    environment: dict[str, str] = Field(default_factory=dict)
    timeout_policy: RemoteTimeoutPolicy = Field(default_factory=RemoteTimeoutPolicy)
    sync_policy: RemoteSyncPolicy = Field(default_factory=RemoteSyncPolicy)
    artifact_policy: ArtifactTransferPolicy = Field(default_factory=ArtifactTransferPolicy)
    source_compatible_local_executor: str = "local_host"
    notes: list[str] = Field(default_factory=list)
