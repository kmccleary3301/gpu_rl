from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    stdout_path = os.environ.get("GPC_STDOUT_PATH")
    if not stdout_path:
        print("missing GPC_STDOUT_PATH", file=sys.stderr)
        return 2
    content = Path(stdout_path).read_text(encoding="utf-8")
    if "GPU_COCKPIT_SMOKE_OK" not in content:
        print("expected token missing from stdout", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
