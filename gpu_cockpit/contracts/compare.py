from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class RunComparison(ContractModel):
    lhs_run_id: str
    rhs_run_id: str
    lhs_status: str
    rhs_status: str
    lhs_duration_ms: int | None = None
    rhs_duration_ms: int | None = None
    duration_delta_ms: int | None = None
    lhs_exit_code: int | None = None
    rhs_exit_code: int | None = None
    warnings_added: list[str] = Field(default_factory=list)
    warnings_removed: list[str] = Field(default_factory=list)
    lhs_key_artifacts: list[str] = Field(default_factory=list)
    rhs_key_artifacts: list[str] = Field(default_factory=list)
    lhs_missing_required_artifacts: list[str] = Field(default_factory=list)
    rhs_missing_required_artifacts: list[str] = Field(default_factory=list)
    lhs_failed_scopes: list[str] = Field(default_factory=list)
    rhs_failed_scopes: list[str] = Field(default_factory=list)
    lhs_evidence_score: float | None = None
    rhs_evidence_score: float | None = None
    evidence_score_delta: float | None = None
    lhs_benchmark_ready: bool | None = None
    rhs_benchmark_ready: bool | None = None
    lhs_sft_ready: bool | None = None
    rhs_sft_ready: bool | None = None
    lhs_rl_trace_ready: bool | None = None
    rhs_rl_trace_ready: bool | None = None
    lhs_training_example_kind: str | None = None
    rhs_training_example_kind: str | None = None
    trainworthiness_change: str | None = None
    correctness_recovered: bool | None = None
    perf_improved: bool | None = None
    lhs_final_score: float | None = None
    rhs_final_score: float | None = None
    final_score_delta: float | None = None
    lhs_perf_p50_ms: float | None = None
    rhs_perf_p50_ms: float | None = None
    perf_p50_delta_ms: float | None = None
    lhs_determinism_passed: bool | None = None
    rhs_determinism_passed: bool | None = None
    lhs_profile_classification: str | None = None
    rhs_profile_classification: str | None = None
    lhs_profile_kernel: str | None = None
    rhs_profile_kernel: str | None = None
    lhs_sanitizer_passed: bool | None = None
    rhs_sanitizer_passed: bool | None = None
    lhs_sanitizer_error_count: int | None = None
    rhs_sanitizer_error_count: int | None = None
    lhs_profile_occupancy: float | None = None
    rhs_profile_occupancy: float | None = None
    occupancy_delta: float | None = None
    lhs_profile_dram_pct_peak: float | None = None
    rhs_profile_dram_pct_peak: float | None = None
    dram_pct_peak_delta: float | None = None
    lhs_profile_sm_pct_peak: float | None = None
    rhs_profile_sm_pct_peak: float | None = None
    sm_pct_peak_delta: float | None = None
    lhs_profile_registers_per_thread: int | None = None
    rhs_profile_registers_per_thread: int | None = None
    registers_per_thread_delta: int | None = None
    lhs_build_status: str | None = None
    rhs_build_status: str | None = None
    lhs_build_compiler: str | None = None
    rhs_build_compiler: str | None = None
    lhs_build_binary_hash: str | None = None
    rhs_build_binary_hash: str | None = None
    build_binary_hash_changed: bool | None = None
    lhs_patch_present: bool | None = None
    rhs_patch_present: bool | None = None
    lhs_patch_kind: str | None = None
    rhs_patch_kind: str | None = None
    lhs_patch_hash: str | None = None
    rhs_patch_hash: str | None = None
    patch_hash_changed: bool | None = None
    lhs_parent_candidate_id: str | None = None
    rhs_parent_candidate_id: str | None = None
    lineage_relationship: str | None = None
    parent_child_related: bool | None = None
    lhs_changed_file_count: int | None = None
    rhs_changed_file_count: int | None = None
    changed_file_count_delta: int | None = None
    lhs_transition_kind: str | None = None
    rhs_transition_kind: str | None = None
    lhs_candidate_id: str | None = None
    rhs_candidate_id: str | None = None
    lhs_triview_present: bool | None = None
    rhs_triview_present: bool | None = None
    lhs_triview_correlation_method: str | None = None
    rhs_triview_correlation_method: str | None = None
    lhs_triview_source_path: str | None = None
    rhs_triview_source_path: str | None = None
    lhs_triview_line_count: int | None = None
    rhs_triview_line_count: int | None = None
    triview_line_count_delta: int | None = None
    lhs_triview_unique_source_lines: int | None = None
    rhs_triview_unique_source_lines: int | None = None
    triview_unique_source_lines_delta: int | None = None
    lhs_build_source_hash: str | None = None
    rhs_build_source_hash: str | None = None
    build_source_hash_changed: bool | None = None
    lhs_build_ttir_hash: str | None = None
    rhs_build_ttir_hash: str | None = None
    build_ttir_hash_changed: bool | None = None
    lhs_build_ttgir_hash: str | None = None
    rhs_build_ttgir_hash: str | None = None
    build_ttgir_hash_changed: bool | None = None
    lhs_build_llir_hash: str | None = None
    rhs_build_llir_hash: str | None = None
    build_llir_hash_changed: bool | None = None
    lhs_build_ptx_hash: str | None = None
    rhs_build_ptx_hash: str | None = None
    build_ptx_hash_changed: bool | None = None
    lhs_build_sass_hash: str | None = None
    rhs_build_sass_hash: str | None = None
    build_sass_hash_changed: bool | None = None
    summary_lines: list[str] = Field(default_factory=list)
