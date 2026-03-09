from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import (
    AntiHackReport,
    ArtifactManifest,
    BenchmarkCaseSpec,
    BottleneckCard,
    BuildRecord,
    CorrectnessReport,
    DeterminismAttempt,
    DeterminismReport,
    DoctorReport,
    EvalEnvelope,
    Event,
    HardwareFingerprint,
    HookExecution,
    PerfReport,
    ProfileReport,
    ReplayPack,
    RunComparison,
    RunSpec,
    RunSummary,
    SanitizerFinding,
    SanitizerReport,
    SystemTraceSummary,
    TaskSpec,
    TriViewArtifact,
    TriViewLine,
)


SCHEMA_MODELS = [
    AntiHackReport,
    ArtifactManifest,
    BenchmarkCaseSpec,
    BottleneckCard,
    BuildRecord,
    RunComparison,
    CorrectnessReport,
    DeterminismAttempt,
    DeterminismReport,
    DoctorReport,
    EvalEnvelope,
    Event,
    HardwareFingerprint,
    HookExecution,
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
]


class GoldenSchemaTests(unittest.TestCase):
    def test_exported_schemas_match_current_models(self) -> None:
        schema_dir = ROOT / "gpu_cockpit" / "artifacts" / "schemas"
        for model in SCHEMA_MODELS:
            with self.subTest(model=model.__name__):
                schema_path = schema_dir / f"{model.schema_name()}.schema.json"
                self.assertTrue(schema_path.exists(), f"Missing schema file: {schema_path}")
                exported = json.loads(schema_path.read_text(encoding="utf-8"))
                current = model.model_json_schema()
                self.assertEqual(exported, current)


if __name__ == "__main__":
    unittest.main()
