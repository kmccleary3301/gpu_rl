from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class PerfReport(ContractModel):
    baseline_id: str
    timer: str
    warmups: int = Field(ge=0)
    repeats: int = Field(ge=1)
    cold_compile_ms: int | None = Field(default=None, ge=0)
    steady_state_ms_p50: float = Field(ge=0.0)
    steady_state_ms_p95: float = Field(ge=0.0)
    baseline_steady_state_ms_p50: float | None = Field(default=None, ge=0.0)
    baseline_steady_state_ms_p95: float | None = Field(default=None, ge=0.0)
    speedup_vs_baseline: float
    variance_pct: float = Field(ge=0.0)
    memory_peak_mb: int | None = Field(default=None, ge=0)
    perf_notes: list[str] = Field(default_factory=list)
