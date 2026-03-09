from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class AntiHackHit(ContractModel):
    pattern: str
    category: str
    location: str
    matched_text: str


class AntiHackReport(ContractModel):
    passed: bool
    forbidden_patterns: list[str] = Field(default_factory=list)
    scanned_locations: list[str] = Field(default_factory=list)
    hits: list[AntiHackHit] = Field(default_factory=list)
    cpu_fallback_detected: bool = False
    library_shortcut_detected: bool = False
    warnings: list[str] = Field(default_factory=list)
