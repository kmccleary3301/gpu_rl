from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class TriViewLine(ContractModel):
    source_line: int | None = Field(default=None, ge=1)
    ptx_line: int | None = Field(default=None, ge=1)
    sass_line: int | None = Field(default=None, ge=1)
    source_text: str | None = None
    ptx_text: str | None = None
    sass_text: str | None = None
    anchors: list[str] = Field(default_factory=list)


class TriViewArtifact(ContractModel):
    backend: str
    correlation_method: str = "heuristic_line_zip_v1"
    source_path: str | None = None
    source_ref: str | None = None
    ttir_ref: str | None = None
    ttgir_ref: str | None = None
    llir_ref: str | None = None
    ptx_ref: str | None = None
    sass_ref: str | None = None
    line_map_ref: str | None = None
    lines: list[TriViewLine] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
