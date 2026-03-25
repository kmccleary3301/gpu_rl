from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import torch


def _load_case_config(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_problem_module(path: Path):
    spec = importlib.util.spec_from_file_location(f"kernelbench_v3_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load KB-v3 problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _materialize(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float32, device=device)
    return value


def _apply_module_overrides(module: Any, overrides: dict[str, Any] | None) -> None:
    if not isinstance(overrides, dict):
        return
    for key, value in overrides.items():
        setattr(module, key, value)


def _build_split_payload(module: Any, split_spec: dict[str, Any], device: torch.device) -> tuple[list[Any], list[Any]]:
    seed = split_spec.get("seed")
    if isinstance(seed, int):
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    _apply_module_overrides(module, split_spec.get("module_overrides"))
    init_inputs = split_spec.get("init_inputs")
    if not isinstance(init_inputs, list):
        init_inputs = list(module.get_init_inputs()) if hasattr(module, "get_init_inputs") else []
    inputs = split_spec.get("inputs")
    if isinstance(inputs, list):
        materialized_inputs = [_materialize(value, device) for value in inputs]
    else:
        materialized_inputs = [_materialize(value, device) for value in module.get_inputs()]
    return init_inputs, materialized_inputs


def _run_case(problem_path: Path, case: dict[str, Any], split: str, device: torch.device) -> dict[str, Any]:
    module = _load_problem_module(problem_path)
    split_spec = case.get(split, {})
    init_inputs, inputs = _build_split_payload(module, split_spec, device)
    model = module.Model(*init_inputs).to(device)
    model.eval()
    with torch.no_grad():
        output = model(*inputs)
    output_cpu = output.detach().cpu().to(torch.float32)
    return {
        "shape": list(output_cpu.shape),
        "sum": float(output_cpu.sum().item()),
        "mean": float(output_cpu.mean().item()),
        "sha256": hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest(),
    }


def _benchmark(problem_path: Path, case: dict[str, Any], repeats: int, device: torch.device) -> None:
    module = _load_problem_module(problem_path)
    split_spec = case.get("benchmark", {})
    init_inputs, inputs = _build_split_payload(module, split_spec, device)
    model = module.Model(*init_inputs).to(device)
    model.eval()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()


def build_case_payload(
    case_config_path: Path,
    *,
    benchmark_repeats: int,
    extra_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    case = _load_case_config(case_config_path)
    problem_path = (case_config_path.parent / case["problem_relpath"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _benchmark(problem_path, case, benchmark_repeats, device)
    payload = {
        "benchmark_source": case.get("benchmark_source", "kernelbench-v3"),
        "benchmark_case_id": case["benchmark_case_id"],
        "benchmark_case_version": case["benchmark_case_version"],
        "provenance_kind": case.get("provenance_kind"),
        "official_track": case.get("official_track"),
        "problem_path": str(problem_path),
        "case_config_path": str(case_config_path),
        "visible_result": _run_case(problem_path, case, "visible", device),
        "hidden_result": _run_case(problem_path, case, "hidden", device),
    }
    if extra_payload:
        payload.update(extra_payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", required=True)
    parser.add_argument("--benchmark-repeats", type=int, default=30)
    args = parser.parse_args()

    case_config_path = Path(args.case_config).resolve()
    print(json.dumps(build_case_payload(case_config_path, benchmark_repeats=args.benchmark_repeats), sort_keys=True))


if __name__ == "__main__":
    main()
