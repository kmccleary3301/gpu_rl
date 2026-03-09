from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel
from gpu_cockpit.contracts.hardware import HardwareFingerprint


class ToolStatus(ContractModel):
    name: str
    path: str | None = None
    version: str | None = None
    available: bool


class DoctorReport(ContractModel):
    host_platform: str
    host_kernel: str
    python_executable: str
    python_version: str
    container_runtime: str | None = None
    available_tools: list[ToolStatus] = Field(default_factory=list)
    hardware_fingerprints: list[HardwareFingerprint] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
