from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from gpu_cockpit.contracts import ArtifactManifest, Event, RunSpec


class RunBundleWriter:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.run_dir = root / "runs" / "pending"
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.artifacts_dir = self.run_dir
        self.meta_dir = self.run_dir / "meta"
        self.seq = 0
        self.run_spec: RunSpec | None = None
        self.artifact_budget_bytes: int | None = None
        self.artifact_bytes_written = 0

    def _sanitize_run_id(self, run_id: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in run_id)
        return safe.strip("_") or datetime.now(tz=UTC).strftime("run_%Y%m%d_%H%M%S")

    def initialize(self, run_spec: RunSpec) -> Path:
        self.run_spec = run_spec
        self.run_dir = self.root / "runs" / self._sanitize_run_id(run_spec.run_id)
        self.events_path = self.run_dir / "events.jsonl"
        self.manifest_path = self.run_dir / "manifest.json"
        self.artifacts_dir = self.run_dir
        self.meta_dir = self.run_dir / "meta"
        self.artifact_budget_bytes = run_spec.budgets.artifact_mb * 1024 * 1024
        self.artifact_bytes_written = 0
        self.run_dir.mkdir(parents=True, exist_ok=False)
        self.meta_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(
            json.dumps(run_spec.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        self.events_path.write_text("", encoding="utf-8")
        self.append_event(scope="run", kind="started", payload={"task_ref": run_spec.task_ref})
        return self.run_dir

    def append_event(self, scope: str, kind: str, payload: dict[str, Any]) -> Event:
        if self.run_spec is None:
            raise RuntimeError("Run bundle has not been initialized.")
        event = Event(
            event_id=f"evt_{self.seq:06d}",
            run_id=self.run_spec.run_id,
            seq=self.seq,
            ts=datetime.now(tz=UTC),
            scope=scope,
            kind=kind,
            payload=payload,
        )
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.model_dump(mode="json")) + "\n")
        self.seq += 1
        return event

    def write_artifact(
        self,
        relative_path: str,
        kind: str,
        content: str | bytes,
        mime: str,
        semantic_tags: list[str] | None = None,
        producer_event_id: str | None = None,
    ) -> ArtifactManifest:
        if self.run_spec is None:
            raise RuntimeError("Run bundle has not been initialized.")
        semantic_tags = semantic_tags or []
        artifact_path = self.artifacts_dir / relative_path
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(content, bytes):
            digest = hashlib.sha256(content).hexdigest()
            size_bytes = len(content)
            if self.artifact_budget_bytes is not None and self.artifact_bytes_written + size_bytes > self.artifact_budget_bytes:
                raise RuntimeError(f"Artifact budget exceeded for run {self.run_spec.run_id}: {self.artifact_bytes_written + size_bytes} > {self.artifact_budget_bytes}")
            artifact_path.write_bytes(content)
        else:
            encoded = content.encode("utf-8")
            digest = hashlib.sha256(encoded).hexdigest()
            size_bytes = len(encoded)
            if self.artifact_budget_bytes is not None and self.artifact_bytes_written + size_bytes > self.artifact_budget_bytes:
                raise RuntimeError(f"Artifact budget exceeded for run {self.run_spec.run_id}: {self.artifact_bytes_written + size_bytes} > {self.artifact_budget_bytes}")
            artifact_path.write_text(content, encoding="utf-8")
        artifact = ArtifactManifest(
            artifact_id=f"artifact_{uuid4().hex[:12]}",
            run_id=self.run_spec.run_id,
            kind=kind,
            path=str(artifact_path.relative_to(self.run_dir)),
            mime=mime,
            sha256=digest,
            producer_event_id=producer_event_id,
            semantic_tags=semantic_tags,
            size_bytes=size_bytes,
        )
        manifest_dir = self.meta_dir / "artifacts"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"{artifact.artifact_id}.json"
        manifest_path.write_text(
            json.dumps(artifact.model_dump(mode="json"), indent=2) + "\n",
            encoding="utf-8",
        )
        self.artifact_bytes_written += size_bytes
        return artifact

    def list_artifacts(self) -> list[ArtifactManifest]:
        manifest_dir = self.meta_dir / "artifacts"
        if not manifest_dir.exists():
            return []
        manifests: list[ArtifactManifest] = []
        for path in sorted(manifest_dir.glob("*.json")):
            manifests.append(ArtifactManifest.model_validate_json(path.read_text(encoding="utf-8")))
        return manifests
