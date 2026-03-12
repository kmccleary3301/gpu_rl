from __future__ import annotations

import argparse
import subprocess
import sys


GPU_SMOKE_TEST_MODULES = [
    "tests.test_nvidia_backends",
    "tests.test_runner",
    "tests.test_triton_build",
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["smoke", "full"],
        default="smoke",
        help="Select the GPU-capable test suite to run.",
    )
    args = parser.parse_args()

    if args.suite == "full":
        command = [sys.executable, "-m", "unittest", "discover", "-s", "tests", "-v"]
    else:
        command = [sys.executable, "-m", "unittest", "-v", *GPU_SMOKE_TEST_MODULES]
    completed = subprocess.run(command)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
