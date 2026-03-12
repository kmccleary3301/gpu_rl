from __future__ import annotations

import subprocess
import sys


CPU_SAFE_TEST_MODULES = [
    "tests.test_contracts",
    "tests.test_run_bundle",
    "tests.test_task_registry",
    "tests.test_schema_golden",
    "tests.test_golden_run",
    "tests.test_adapters",
    "tests.test_amd_backends",
    "tests.test_nvidia_backends",
    "tests.test_replay",
    "tests.test_inspector",
    "tests.test_runner",
    "tests.test_benchmark",
]


def main() -> int:
    command = [sys.executable, "-m", "unittest", "-v", *CPU_SAFE_TEST_MODULES]
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
