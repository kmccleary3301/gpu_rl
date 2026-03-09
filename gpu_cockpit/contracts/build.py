from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class BuildRecord(ContractModel):
    compiler: str
    compiler_version: str
    flags: list[str] = Field(default_factory=list)
    status: str
    duration_ms: int = Field(ge=0)
    stdout_ref: str | None = None
    stderr_ref: str | None = None
    ptx_ref: str | None = None
    sass_ref: str | None = None
    ptxas_stats_ref: str | None = None
    binary_hash: str | None = None
