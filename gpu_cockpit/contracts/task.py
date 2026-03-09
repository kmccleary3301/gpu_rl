from __future__ import annotations

from typing import Literal

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class NumericalPolicy(ContractModel):
    dtype_matrix: list[str] = Field(default_factory=list)
    rtol: float = Field(ge=0.0)
    atol: float = Field(ge=0.0)


class PerfProtocol(ContractModel):
    warmups: int = Field(ge=0)
    repeats: int = Field(ge=1)
    timer: Literal["cuda_event", "hip_event", "wall_clock"]
    split_compile_from_run: bool = True


class TaskSpec(ContractModel):
    task_id: str
    verb: Literal["synthesize", "optimize", "debug", "fuse", "port", "diagnose", "reformulate"]
    operator_family: str
    difficulty: str
    prompt: str
    baseline_ref: str | None = None
    reference_impl_ref: str | None = None
    visible_tests_ref: str | None = None
    hidden_tests_ref: str | None = None
    input_generator_ref: str | None = None
    deterministic_mode: bool = True
    feature_requirements: list[str] = Field(default_factory=list)
    numerical_policy: NumericalPolicy
    perf_protocol: PerfProtocol
    allowed_backends: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    required_artifacts: list[str] = Field(default_factory=list)
