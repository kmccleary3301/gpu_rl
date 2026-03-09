from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class SanitizerFinding(ContractModel):
    tool: str
    category: str
    severity: str
    message: str
    kernel_name: str | None = None
    file_path: str | None = None
    line: int | None = Field(default=None, ge=1)
    raw_line: str | None = None


class SanitizerReport(ContractModel):
    backend: str
    tool: str
    command: list[str] = Field(default_factory=list)
    exit_code: int
    duration_ms: int = Field(ge=0)
    passed: bool
    error_count: int = Field(default=0, ge=0)
    warning_count: int = Field(default=0, ge=0)
    findings: list[SanitizerFinding] = Field(default_factory=list)
    stdout_path: str | None = None
    stderr_path: str | None = None
    raw_log_ref: str | None = None
    warnings: list[str] = Field(default_factory=list)
