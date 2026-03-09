# KernelBench Notes

The planner explicitly called out KernelBench as the preferred first public benchmark family because it maps naturally onto the cockpit’s task/eval model. The practical use here is a curated subset with provenance, stable identifiers, and baseline wiring rather than a one-shot bulk import.

The official benchmark organizes problems into four levels, stores the canonical source problems under `KernelBench/level*`, and evaluates correctness plus speedup via the `fast_p` family of metrics. Our cockpit adapter preserves official case provenance and versioning while using a small curated local harness for stable end-to-end eval.

The current curated import set is tracked in `workloads/public_benchmarks/kernelbench/v0_1/manifest.json`. It now covers eleven curated cases and intentionally spans multiple operator families instead of only activations:

- activation: HardTanh, MinGPTNewGelu, Softmax, Softmax wide-logit variant
- normalization: LayerNorm, LayerNorm shifted-distribution variant
- reduction: sum reduction, sum reduction long-axis variant
- indexing: Argmax
- matmul: square matmul
- attention-adjacent: scaled dot-product attention

Each imported case carries:

- benchmark name and version
- official source path
- official problem id
- operator family
- eval-type tags such as `perf-heavy`, `memory-heavy`, `correctness-heavy`, and `attention-adjacent`

The operator-facing summaries also group the curated subset by operator family and tag, including `curated-variant`, so coverage gaps are visible without opening the raw case JSON. The checked-in public collection fixture at `tests/golden_datasets/public_collection_v1` includes one KernelBench episode alongside a ComputeEval episode so trajectory validation can exercise public-benchmark collection semantics end to end.
