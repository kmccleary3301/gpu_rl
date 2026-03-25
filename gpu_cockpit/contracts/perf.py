from __future__ import annotations

from typing import Any

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class PerfReport(ContractModel):
    baseline_id: str
    timer: str
    timing_method: str | None = None
    warmups: int = Field(ge=0)
    repeats: int = Field(ge=1)
    split_compile_from_run: bool = True
    cold_compile_ms: int | None = Field(default=None, ge=0)
    baseline_cold_compile_ms: int | None = Field(default=None, ge=0)
    steady_state_ms_p50: float = Field(ge=0.0)
    steady_state_ms_p95: float = Field(ge=0.0)
    baseline_steady_state_ms_p50: float | None = Field(default=None, ge=0.0)
    baseline_steady_state_ms_p95: float | None = Field(default=None, ge=0.0)
    speedup_vs_baseline: float
    variance_pct: float = Field(ge=0.0)
    memory_peak_mb: int | None = Field(default=None, ge=0)
    benchmark_scope: str | None = None
    benchmark_protocol_version: str = "benchmark_v2"
    candidate_command_sha256: str | None = None
    baseline_command_sha256: str | None = None
    hardware_fingerprint: dict[str, Any] | None = None
    benchmark_provenance: dict[str, Any] = Field(default_factory=dict)
    perf_notes: list[str] = Field(default_factory=list)
