from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.training import write_sft_smoke_report


def main() -> int:
    config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
    out_path = ROOT / "artifacts" / "training" / "smoke_sft_report.json"
    write_sft_smoke_report(ROOT, config_path, out_path)
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
