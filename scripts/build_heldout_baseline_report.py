from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.contracts import RLRolloutConfig
from gpu_cockpit.engine.rollout import run_scripted_rollout_suite


def main() -> int:
    config_path = ROOT / "configs" / "training" / "rollout_debug_repair_heldout_v1.json"
    config = RLRolloutConfig.model_validate(json.loads(config_path.read_text(encoding="utf-8")))
    out_dir = ROOT / "artifacts" / "training" / "heldout_scripted_baseline_v1"
    report = run_scripted_rollout_suite(ROOT, config, out_dir)
    print((out_dir / "rollout_report.json").relative_to(ROOT))
    print(json.dumps(report.model_dump(mode="json"), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
