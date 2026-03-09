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
    spec = importlib.util.spec_from_file_location(f"kernelbench_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load problem module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _materialize(value: Any, device: torch.device) -> Any:
    if isinstance(value, list):
        if value and isinstance(value[0], list):
            return torch.tensor(value, dtype=torch.float32, device=device)
        return torch.tensor(value, dtype=torch.float32, device=device)
    return value


def _build_inputs(serialized_inputs: list[Any], device: torch.device) -> list[Any]:
    return [_materialize(value, device) for value in serialized_inputs]


def _run_case(problem_path: Path, case: dict[str, Any], split: str, device: torch.device) -> dict[str, Any]:
    module = _load_problem_module(problem_path)
    init_inputs = case.get("init_inputs", [])
    model = module.Model(*init_inputs).to(device)
    model.eval()
    inputs = _build_inputs(case[f"{split}_inputs"], device)
    with torch.no_grad():
        output = model(*inputs)
    output_cpu = output.detach().cpu().to(torch.float32)
    payload = {
        "shape": list(output_cpu.shape),
        "sum": float(output_cpu.sum().item()),
        "mean": float(output_cpu.mean().item()),
        "sha256": hashlib.sha256(output_cpu.numpy().tobytes()).hexdigest(),
    }
    return payload


def _benchmark(problem_path: Path, case: dict[str, Any], repeats: int, device: torch.device) -> None:
    module = _load_problem_module(problem_path)
    benchmark = case.get("benchmark", {})
    init_inputs = benchmark.get("init_inputs", case.get("init_inputs", []))
    inputs = _build_inputs(benchmark["inputs"], device)
    model = module.Model(*init_inputs).to(device)
    model.eval()
    for _ in range(repeats):
        with torch.no_grad():
            _ = model(*inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-config", required=True)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    case_config_path = Path(args.case_config).resolve()
    case = _load_case_config(case_config_path)
    problem_path = (case_config_path.parent / case["problem_relpath"]).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _benchmark(problem_path, case, args.benchmark_repeats, device)
    print(
        json.dumps(
            {
                "benchmark_source": "kernelbench",
                "benchmark_case_id": case["benchmark_case_id"],
                "benchmark_case_version": case["benchmark_case_version"],
                "problem_path": str(problem_path),
                "case_config_path": str(case_config_path),
                "visible_result": _run_case(problem_path, case, "visible", device),
                "hidden_result": _run_case(problem_path, case, "hidden", device),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
