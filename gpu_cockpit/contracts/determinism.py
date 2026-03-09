from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class DeterminismAttempt(ContractModel):
    attempt_index: int = Field(ge=1)
    exit_code: int
    duration_ms: int = Field(ge=0)
    stdout_sha256: str
    stderr_sha256: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)


class DeterminismReport(ContractModel):
    command: list[str] = Field(default_factory=list)
    runs: int = Field(ge=1)
    passed: bool
    stable_exit_codes: bool
    stable_stdout: bool
    stable_stderr: bool
    baseline_exit_code: int | None = None
    baseline_stdout_sha256: str | None = None
    baseline_stderr_sha256: str | None = None
    attempts: list[DeterminismAttempt] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
