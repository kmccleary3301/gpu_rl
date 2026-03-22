from __future__ import annotations

from typing import Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


PatchKind = Literal["bug_fix", "perf_transform", "refactor", "instrumentation", "no_op", "rejected"]
PatchContentMode = Literal["replace", "create", "delete", "unified_diff"]
CandidateRole = Literal[
    "baseline_candidate",
    "working_candidate",
    "patched_candidate",
    "branched_candidate",
    "reverted_candidate",
    "promoted_candidate",
    "synthesized_candidate",
    "comparison_anchor",
]
CandidateOriginKind = Literal["baseline", "patch", "branch", "revert", "promotion", "synthesis", "imported", "unknown"]
CandidateStatus = Literal[
    "draft",
    "patched",
    "ready_for_build",
    "build_passed",
    "build_failed",
    "eval_passed",
    "eval_failed",
    "benchmarked",
    "profiled",
    "promoted",
    "reverted",
    "archived",
]
CandidateOperationKind = Literal["create", "edit", "patch_apply", "build", "eval", "bench", "profile", "compare", "branch", "revert", "promote"]
TransitionKind = Literal[
    "inspect_only",
    "patch_applied",
    "benchmarked",
    "evaluated",
    "repaired",
    "reformulated",
    "reverted",
    "branched",
    "promoted",
]


class PatchRequest(ContractModel):
    target_file: str
    patch_kind: PatchKind
    intent: str
    expected_effect: str | None = None
    content_mode: PatchContentMode = "replace"
    patch_text: str
    metadata: dict[str, object] = Field(default_factory=dict)


class AppliedPatch(ContractModel):
    patch_id: str
    target_file: str
    patch_kind: PatchKind
    intent: str
    expected_effect: str | None = None
    content_mode: PatchContentMode = "replace"
    accepted: bool = True
    patch_hash: str
    changed_line_count: int = Field(ge=0, default=0)
    before_ref: str | None = None
    after_ref: str | None = None
    diff_ref: str | None = None
    error_message: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateDiffSummary(ContractModel):
    changed_file_count: int = Field(ge=0, default=0)
    changed_line_count: int = Field(ge=0, default=0)
    changed_files: list[str] = Field(default_factory=list)
    primary_target_file: str | None = None
    diff_ref: str | None = None
    patch_hash: str | None = None
    summary: str | None = None


class CandidateState(ContractModel):
    candidate_id: str
    task_id: str
    workspace_root: str
    source_run_ref: str | None = None
    source_run_id: str | None = None
    parent_candidate_id: str | None = None
    source_candidate_id: str | None = None
    parent_run_ref: str | None = None
    patch_id: str | None = None
    patch_hash: str | None = None
    patch_kind: PatchKind | None = None
    changed_files: list[str] = Field(default_factory=list)
    candidate_role: CandidateRole | None = None
    origin_kind: CandidateOriginKind = "unknown"
    status: CandidateStatus = "draft"
    lineage_depth: int = Field(ge=0, default=0)
    active: bool = True
    summary: str | None = None
    hypothesis_tags: list[str] = Field(default_factory=list)
    artifact_refs: list[str] = Field(default_factory=list)
    diff_summary: CandidateDiffSummary | None = None
    last_operation_kind: CandidateOperationKind | None = None
    promotion_label: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateTransition(ContractModel):
    transition_id: str
    transition_kind: TransitionKind
    operation_kind: CandidateOperationKind | None = None
    task_id: str
    input_candidate_id: str | None = None
    output_candidate_id: str | None = None
    source_run_ref: str | None = None
    target_run_ref: str | None = None
    patch_id: str | None = None
    patch_hash: str | None = None
    patch_kind: PatchKind | None = None
    changed_files: list[str] = Field(default_factory=list)
    status: str = "ok"
    summary: str | None = None
    diff_summary: CandidateDiffSummary | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class CandidateOperation(ContractModel):
    operation_id: str
    operation_kind: CandidateOperationKind
    task_id: str
    actor_type: Literal["agent", "human", "scripted", "system"] = "agent"
    input_candidate_id: str | None = None
    output_candidate_id: str | None = None
    source_run_ref: str | None = None
    target_run_ref: str | None = None
    patch_id: str | None = None
    patch_hash: str | None = None
    changed_files: list[str] = Field(default_factory=list)
    diff_summary: CandidateDiffSummary | None = None
    result_status: str = "ok"
    summary: str | None = None
    artifact_refs: list[str] = Field(default_factory=list)
    metadata: dict[str, object] = Field(default_factory=dict)
