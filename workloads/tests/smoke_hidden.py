from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    stdout_path = os.environ.get("GPC_STDOUT_PATH")
    stderr_path = os.environ.get("GPC_STDERR_PATH")
    if not stdout_path or not stderr_path:
        print("missing stdout/stderr paths", file=sys.stderr)
        return 2
    stdout = Path(stdout_path).read_text(encoding="utf-8").strip()
    stderr = Path(stderr_path).read_text(encoding="utf-8").strip()
    if stdout != "GPU_COCKPIT_SMOKE_OK":
        print(f"stdout mismatch: {stdout!r}", file=sys.stderr)
        return 1
    if stderr:
        print("stderr was not empty", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
