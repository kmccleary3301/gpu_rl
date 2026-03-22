from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.training import run_sft_training_job


def main() -> int:
    config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_qlora_spark_smoke.json"
    report_path = run_sft_training_job(ROOT, config_path)
    print(report_path.relative_to(ROOT))
    payload = report_path.read_text(encoding="utf-8")
    print(payload)
    return 0 if '"status": "ok"' in payload else 1


if __name__ == "__main__":
    raise SystemExit(main())
