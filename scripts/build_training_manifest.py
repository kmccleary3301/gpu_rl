from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gpu_cockpit.engine.training import load_training_config, validate_sft_training_config


def _validate_inventory_refs(root: Path, payload: dict[str, object]) -> dict[str, object]:
    groups = {
        "config_refs": dict(payload.get("config_refs", {})),
        "doc_refs": dict(payload.get("doc_refs", {})),
        "script_refs": dict(payload.get("script_refs", {})),
    }
    required_fixtures = list(payload.get("required_checked_in_fixtures", []))

    missing_refs: list[str] = []
    group_summaries: dict[str, dict[str, object]] = {}
    for group_name, refs in groups.items():
        checked: dict[str, object] = {}
        for ref_name, ref_path in refs.items():
            resolved = (root / str(ref_path)).resolve()
            exists = resolved.exists()
            checked[ref_name] = {"path": str(ref_path), "exists": exists}
            if not exists:
                missing_refs.append(str(ref_path))
        group_summaries[group_name] = checked

    fixture_summary: list[dict[str, object]] = []
    for fixture_ref in required_fixtures:
        resolved = (root / str(fixture_ref)).resolve()
        exists = resolved.exists()
        fixture_summary.append({"path": str(fixture_ref), "exists": exists})
        if not exists:
            missing_refs.append(str(fixture_ref))

    return {
        "status": "ok" if not missing_refs else "failed",
        "missing_refs": sorted(set(missing_refs)),
        "groups": group_summaries,
        "required_checked_in_fixtures": fixture_summary,
    }


def main() -> int:
    training_dir = ROOT / "docs" / "training"
    training_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = ROOT / "configs" / "training" / "training_inventory_v1.json"
    split_config_path = ROOT / "configs" / "training" / "first_target_splits_v1.json"
    sft_config_path = ROOT / "configs" / "training" / "sft_qwen32b_debug_repair_lora.json"
    rollout_config_path = ROOT / "configs" / "training" / "rollout_debug_repair_v1.json"
    heldout_rollout_config_path = ROOT / "configs" / "training" / "rollout_debug_repair_heldout_v1.json"
    sft_config = load_training_config(sft_config_path)
    validation = validate_sft_training_config(ROOT, sft_config)
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    inventory_validation = _validate_inventory_refs(ROOT, payload)
    payload.update(
        {
            "model_target": sft_config.model_id,
            "inventory_ref": str(inventory_path.relative_to(ROOT)),
            "inventory_validation": inventory_validation,
            "validation": validation,
            "resolved_config_refs": {
                "splits": str(split_config_path.relative_to(ROOT)),
                "sft": str(sft_config_path.relative_to(ROOT)),
                "rollout": str(rollout_config_path.relative_to(ROOT)),
                "heldout_rollout": str(heldout_rollout_config_path.relative_to(ROOT)),
            },
        }
    )
    out_path = training_dir / "training_manifest.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(out_path.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
