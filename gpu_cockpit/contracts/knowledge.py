from __future__ import annotations

from datetime import datetime

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class KnowledgeEntry(ContractModel):
    entry_id: str
    kind: str
    title: str
    path: str
    source_type: str
    summary: str
    tags: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    operator_family: str | None = None
    backend: str | None = None
    vendor: str | None = None
    metadata: dict[str, object] = Field(default_factory=dict)


class KnowledgeIndexManifest(ContractModel):
    index_id: str
    created_at: datetime
    entry_count: int = Field(ge=0)
    source_roots: list[str] = Field(default_factory=list)
    entries_ref: str
    metadata: dict[str, object] = Field(default_factory=dict)
