from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel
from gpu_cockpit.contracts.common import ToolVersionSet


class HardwareFingerprint(ContractModel):
    vendor: str
    gpu_name: str
    arch: str
    driver_version: str
    runtime_version: str
    memory_gb: int = Field(ge=0)
    power_limit_w: int | None = Field(default=None, ge=0)
    clock_state: str | None = None
    mig_mode: bool = False
    mps_mode: bool = False
    host_kernel: str
    container_runtime: str
    tool_versions: ToolVersionSet
