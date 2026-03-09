from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.sft import package_trajectory_dataset_as_sft, validate_sft_dataset


def main() -> None:
    source_dataset = ROOT / "datasets" / "seed_reference_v2"
    out_dir = ROOT / "datasets" / "sft_seed_reference_v2"
    manifest_path = package_trajectory_dataset_as_sft(
        ROOT,
        source_dataset,
        out_dir,
        split="train",
        include_failures=True,
        verb_allowlist=["diagnose", "debug", "reformulate", "optimize"],
    )
    validation = validate_sft_dataset(out_dir)
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "validation": validation,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
