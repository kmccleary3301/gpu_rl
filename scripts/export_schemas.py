from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import (
    AntiHackReport,
    ArtifactManifest,
    BaselineSpec,
    BenchmarkCaseSpec,
    BottleneckCard,
    BuildRecord,
    RunComparison,
    CorrectnessReport,
    DeterminismAttempt,
    DeterminismReport,
    DoctorReport,
    EvidenceQualityReport,
    EvalEnvelope,
    Event,
    HardwareFingerprint,
    HookExecution,
    KnowledgeEntry,
    KnowledgeIndexManifest,
    PerfReport,
    ProfileReport,
    ReplayPack,
    RunSpec,
    RunSummary,
    SanitizerFinding,
    SanitizerReport,
    SystemTraceSummary,
    TaskSpec,
    TriViewArtifact,
    TriViewLine,
    TrajectoryAction,
    TrajectoryDatasetManifest,
    TrajectoryEpisode,
    TrajectoryObservation,
    TrajectoryStep,
)


SCHEMA_MODELS = [
    AntiHackReport,
    ArtifactManifest,
    BaselineSpec,
    BenchmarkCaseSpec,
    BottleneckCard,
    BuildRecord,
    RunComparison,
    CorrectnessReport,
    DeterminismAttempt,
    DeterminismReport,
    DoctorReport,
    EvidenceQualityReport,
    EvalEnvelope,
    Event,
    HardwareFingerprint,
    HookExecution,
    KnowledgeEntry,
    KnowledgeIndexManifest,
    PerfReport,
    ProfileReport,
    ReplayPack,
    RunSpec,
    RunSummary,
    SanitizerFinding,
    SanitizerReport,
    SystemTraceSummary,
    TaskSpec,
    TriViewArtifact,
    TriViewLine,
    TrajectoryAction,
    TrajectoryDatasetManifest,
    TrajectoryEpisode,
    TrajectoryObservation,
    TrajectoryStep,
]


def main() -> None:
    out_dir = ROOT / "gpu_cockpit" / "artifacts" / "schemas"
    out_dir.mkdir(parents=True, exist_ok=True)
    for model in SCHEMA_MODELS:
        schema = model.model_json_schema()
        out_path = out_dir / f"{model.schema_name()}.schema.json"
        out_path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
        print(out_path.relative_to(ROOT))


if __name__ == "__main__":
    main()
