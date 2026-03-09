# gpu_rl

Research and tooling substrate for training and evaluating LLM-powered code agents on GPU programming tasks.

## What Is Here

- `gpu_cockpit/`: core contracts, engines, CLI, executors, and backend integrations
- `workloads/`: internal tasks, public benchmark adapters, reference implementations, hooks, and curated benchmark inputs
- `tests/`: regression tests plus checked-in golden bundles, datasets, and episodes
- `knowledge/`: operator docs, benchmark notes, profiler playbooks, transformation cards, and hardware notes
- `scripts/`: schema export, seed trajectory collection, SFT packaging, and related utilities

## Main CLI Surfaces

The project currently exposes flows for:

- running tasks
- benchmarking
- evaluation with correctness / determinism / anti-hack / perf gates
- tracing / profiling / sanitizer-backed artifact capture
- inspect / compare / replay
- trajectory export and SFT packaging
- knowledge index build/query

## Notes

- Runtime and planning artifacts such as `runs/`, `datasets/`, and `docs_tmp/` are intentionally ignored from version control.
- Checked-in golden fixtures live under `tests/golden_*` and are part of the source tree.
