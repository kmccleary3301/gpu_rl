from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class KernelProfileMetric(ContractModel):
    metric_name: str
    unit: str | None = None
    value: float | int | str | None = None


class KernelProfileRecord(ContractModel):
    kernel_name: str
    invocation_count: int = Field(default=1, ge=0)
    duration_ms: float | None = Field(default=None, ge=0.0)
    time_pct: float | None = Field(default=None, ge=0.0)
    classification: str
    occupancy: float | None = None
    registers_per_thread: int | None = Field(default=None, ge=0)
    spill_load_bytes: int | None = Field(default=None, ge=0)
    spill_store_bytes: int | None = Field(default=None, ge=0)
    dram_throughput_pct_peak: float | None = None
    sm_throughput_pct_peak: float | None = None
    compute_throughput_pct_peak: float | None = None
    l2_hit_rate: float | None = None
    roofline_position: str | None = None
    raw_metrics: list[KernelProfileMetric] = Field(default_factory=list)


class ProfileReport(ContractModel):
    profiler: str
    backend: str
    command: list[str] = Field(default_factory=list)
    profile_pack: str
    exit_code: int = 0
    duration_ms: int = Field(default=0, ge=0)
    kernel_name: str
    classification: str
    occupancy: float | None = None
    registers_per_thread: int | None = Field(default=None, ge=0)
    spills_bytes: int | None = Field(default=None, ge=0)
    dram_throughput_pct_peak: float | None = None
    sm_throughput_pct_peak: float | None = None
    compute_throughput_pct_peak: float | None = None
    l2_hit_rate: float | None = None
    shared_mem_conflicts: str | None = None
    warp_divergence: str | None = None
    roofline_position: str | None = None
    top_kernels: list[KernelProfileRecord] = Field(default_factory=list)
    primary_kernel_duration_ms: float | None = Field(default=None, ge=0.0)
    profiled_kernel_count: int = Field(default=0, ge=0)
    source_map_ref: str | None = None
    raw_profile_ref: str | None = None
    csv_profile_ref: str | None = None
    stdout_path: str | None = None
    stderr_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
