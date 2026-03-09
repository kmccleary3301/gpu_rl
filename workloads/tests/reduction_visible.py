from __future__ import annotations

import json
import os
from pathlib import Path


def _load_payload() -> dict[str, object]:
    stdout_path = Path(os.environ["GPC_STDOUT_PATH"])
    lines = [line for line in stdout_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise SystemExit("No stdout payload found")
    return json.loads(lines[-1])


def main() -> None:
    payload = _load_payload()
    if payload.get("visible_row_sum") != [10.0, 4.0]:
        raise SystemExit(f"Unexpected visible row sums: {payload.get('visible_row_sum')}")


if __name__ == "__main__":
    main()
