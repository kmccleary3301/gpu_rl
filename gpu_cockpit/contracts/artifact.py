from __future__ import annotations

from pydantic import Field

from gpu_cockpit.contracts.base import ContractModel


class ArtifactManifest(ContractModel):
    artifact_id: str
    run_id: str
    kind: str
    path: str
    mime: str
    sha256: str
    producer_event_id: str | None = None
    semantic_tags: list[str] = Field(default_factory=list)
    size_bytes: int = Field(ge=0)
