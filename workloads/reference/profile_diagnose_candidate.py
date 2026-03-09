from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from profile_diagnose_shared import diagnose_profile_fixture, load_fixture


VISIBLE_FIXTURE = Path("workloads/fixtures/profile_diagnose/visible_memory_bound.json")
HIDDEN_FIXTURE = Path("workloads/fixtures/profile_diagnose/hidden_occupancy_limited.json")


def _run_benchmark(repeats: int) -> None:
    visible = load_fixture(VISIBLE_FIXTURE)
    hidden = load_fixture(HIDDEN_FIXTURE)
    for _ in range(repeats):
        _ = diagnose_profile_fixture(visible)
        _ = diagnose_profile_fixture(hidden)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    args = parser.parse_args()

    _run_benchmark(args.benchmark_repeats)
    print(
        json.dumps(
            {
                "visible_diagnosis": diagnose_profile_fixture(load_fixture(VISIBLE_FIXTURE)),
                "hidden_diagnosis": diagnose_profile_fixture(load_fixture(HIDDEN_FIXTURE)),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
