from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class CorrectnessReport(ContractModel):
    compile_ok: bool
    visible_tests_ok: bool | None = None
    hidden_tests_ok: bool | None = None
    dtype_sweep: dict[str, str] = Field(default_factory=dict)
    shape_sweep: dict[str, str] = Field(default_factory=dict)
    determinism: dict[str, int | bool | str] = Field(default_factory=dict)
    numerical_deltas_ref: str | None = None
    failures: list[str] = Field(default_factory=list)
