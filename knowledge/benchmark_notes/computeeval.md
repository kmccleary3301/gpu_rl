# ComputeEval Notes

The second public benchmark family in the cockpit is a curated subset of NVIDIA's ComputeEval release datapacks. The official benchmark ships as a versioned archive with `problems.jsonl` plus benchmark metadata, and each problem includes prompt text, context files, test files, build/test commands, architecture requirements, and release metadata.

The current import set is tracked in `workloads/public_benchmarks/computeeval/2025_1/manifest.json`. It now pins eight curated cases from the `2025-1` release:

- `CUDA/0`: kernel launch basics
- `CUDA/0` launch-audit variant
- `CUDA/10`: dynamic-parallel recursive reduction
- `CUDA/16`: streams and concurrency
- `CUDA/16` streams-audit variant
- `CUDA/31`: CUB device reduction
- `CUDA/35`: Thrust exclusive scan
- `CUDA/35` scan-provenance variant

Each imported case carries:

- benchmark name and version
- official task id
- curated import batch id
- eval-type tags such as `correctness-heavy`, `perf-heavy`, `memory-heavy`, `library-heavy`, and `metadata-heavy`
- a stable pointer to the checked-in normalized problem JSON

Adapter summaries now group the curated set by operator family, release, and tags, including `curated-variant`, so it is easier to see whether the imported slice is over-indexed on one CUDA pattern. The checked-in public collection fixture at `tests/golden_datasets/public_collection_v1` pairs a ComputeEval episode with a KernelBench episode to keep public-benchmark trajectory validation replayable.

On this host the curated ComputeEval flow is provenance-heavy rather than full CUDA compilation, because the machine does not currently provide `nvcc`. The shared reference runner still gives the cockpit a real benchmark-family adapter, stable task identifiers, visible/hidden hook coverage, baseline timing, and replayable artifact bundles while preserving official benchmark metadata.
