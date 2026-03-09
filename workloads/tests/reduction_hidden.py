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
    expected = [8.0, 3.0, 6.0]
    if payload.get("hidden_row_sum") != expected:
        raise SystemExit(f"Unexpected hidden row sums: {payload.get('hidden_row_sum')} != {expected}")


if __name__ == "__main__":
    main()
