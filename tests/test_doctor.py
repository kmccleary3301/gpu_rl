from __future__ import annotations

import platform
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts.doctor import ToolStatus
from gpu_cockpit.engine.doctor import _collect_nvidia_fingerprints, _parse_optional_float


class DoctorTests(unittest.TestCase):
    def test_parse_optional_float_accepts_numeric_and_missing_markers(self) -> None:
        self.assertEqual(_parse_optional_float("1024"), 1024.0)
        self.assertEqual(_parse_optional_float("[N/A]"), None)
        self.assertEqual(_parse_optional_float("Not Supported"), None)

    def test_collect_nvidia_fingerprints_tolerates_non_numeric_fields(self) -> None:
        tool_statuses = [
            ToolStatus(name="nvidia-smi", path="/usr/bin/nvidia-smi", version="580.95.05", available=True),
            ToolStatus(name="nvcc", path="/usr/local/cuda/bin/nvcc", version="Cuda compilation tools, release 13.0, V13.0.48", available=True),
        ]
        sample_query = "\n".join(
            [
                "NVIDIA GB10, 580.95.05, [N/A], [N/A]",
                "NVIDIA H200, 580.95.05, 141312, 700.00",
            ]
        )
        with (
            patch("gpu_cockpit.engine.doctor.shutil.which", side_effect=lambda name: f"/usr/bin/{name}" if name == "nvidia-smi" else None),
            patch("gpu_cockpit.engine.doctor._run_command", return_value=sample_query),
            patch("gpu_cockpit.engine.doctor.platform.release", return_value=platform.release()),
        ):
            fingerprints = _collect_nvidia_fingerprints(tool_statuses)
        self.assertEqual(len(fingerprints), 2)
        self.assertEqual(fingerprints[0].gpu_name, "NVIDIA GB10")
        self.assertEqual(fingerprints[0].memory_gb, 0)
        self.assertEqual(fingerprints[0].power_limit_w, None)
        self.assertEqual(fingerprints[1].gpu_name, "NVIDIA H200")
        self.assertEqual(fingerprints[1].memory_gb, 138)
        self.assertEqual(fingerprints[1].power_limit_w, 700)


if __name__ == "__main__":
    unittest.main()
