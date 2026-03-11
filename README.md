# gpu_rl

[![Version](https://img.shields.io/badge/version-0.1.0-2f81f7)](./pyproject.toml)
[![Python](https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&logoColor=white)](./pyproject.toml)
[![Verification](https://img.shields.io/badge/verification-114%20tests%20passing-2ea043)](./tests)
[![CLI](https://img.shields.io/badge/cli-gpc-6f42c1)](./gpu_cockpit/cli/main.py)
[![Backends](https://img.shields.io/badge/backends-Triton%20%7C%20CUDA%20%7C%20HIP-8b5cf6)](./workloads)
[![Benchmarks](https://img.shields.io/badge/benchmarks-KernelBench%20%2B%20ComputeEval-f59e0b)](./workloads/public_benchmarks)
[![Training Target](https://img.shields.io/badge/training%20target-Qwen3.5--Coder--32B-eab308)](./docs/training/TRAINING_TARGET.md)
[![Status](https://img.shields.io/badge/status-training%20prep%20ready-0891b2)](./docs/training/README.md)

Research and systems substrate for **artifact-first training and evaluation of LLM-powered code agents on GPU programming tasks**.

The repository centers on one idea:

> GPU-agent work gets much easier to reason about when traces, benchmarks, build artifacts, profile summaries, candidate transitions, and training-readiness decisions are all first-class objects instead of side effects hidden in ad hoc scripts.

This repo is the resulting cockpit:

- a local execution and evaluation surface for `run`, `bench`, `eval`, `build`, `inspect`, `compare`, `replay`, and `bundle`
- a task and benchmark substrate spanning internal Triton/CUDA-style workloads and curated public benchmark imports
- a transition-aware collection layer for trajectories, patch-bearing repair traces, reformulation episodes, and SFT packaging
- a checked-in training-preparation surface for the first narrow training target on a strong sub-40B code model

> [!IMPORTANT]
> This repository is **not** a one-command dense-training stack. Its primary job is to make GPU-agent tasks, artifacts, datasets, and training-preparation state explicit enough that later SFT and narrow RL runs can start from a clean, reproducible substrate.

<details>
<summary><strong>Table of contents</strong></summary>

- [What This Repository Is](#what-this-repository-is)
- [Current Status](#current-status)
- [What You Get](#what-you-get)
- [Repository Map](#repository-map)
- [Installation and Environment](#installation-and-environment)
- [Quick Start](#quick-start)
- [CLI Walkthrough](#cli-walkthrough)
- [Task and Benchmark Surface](#task-and-benchmark-surface)
- [Artifact Model](#artifact-model)
- [Knowledge Layer](#knowledge-layer)
- [Transition-Aware Data Products](#transition-aware-data-products)
- [Training Preparation](#training-preparation)
- [Development Workflow](#development-workflow)
- [Tracked vs Generated State](#tracked-vs-generated-state)

</details>

## What This Repository Is

| Category | Description |
| --- | --- |
| Core package | [`gpu_cockpit/`](./gpu_cockpit) contains contracts, execution engines, backend adapters, CLI entrypoints, and collection logic. |
| Workload substrate | [`workloads/`](./workloads) contains internal task specs, baselines, public benchmark imports, reference implementations, fixtures, and evaluation hooks. |
| Golden verification surface | [`tests/`](./tests) contains regression tests plus checked-in golden bundles, datasets, retrieval fixtures, and multistep episode fixtures. |
| Knowledge and retrieval | [`knowledge/`](./knowledge) contains operator docs, profiler playbooks, transformation cards, benchmark notes, and hardware notes. |
| Training preparation | [`configs/training/`](./configs/training) and [`docs/training/`](./docs/training) freeze the first training target, validation splits, and smoke path. |

| It is | It is not |
| --- | --- |
| A GPU-agent cockpit and data factory | A generic LLM chat app |
| A benchmark and eval normalization layer | A leaderboard-only benchmark wrapper |
| A transition-aware trace and SFT substrate | A finished RL training stack |
| A checked-in training preparation package | A hardware-specific training recipe tied to one developer setup |

## Current Status

| Area | Status | Notes |
| --- | --- | --- |
| Contracts and schemas | 🟢 | Versioned contracts for runs, traces, profiles, replay packs, trajectories, SFT, rollout configs, patches, and candidate lineage |
| NVIDIA workflow | 🟢 | `nsys`, normalized profiling surfaces, sanitizer normalization, tri-view build artifacts, bottleneck summaries |
| AMD mirror | 🟡 | Narrow mirrored trace/profile path and golden fixtures; intentionally not broad parity yet |
| Internal task verbs | 🟢 | `diagnose`, `debug`, `reformulate`, and `optimize` are all represented |
| Public benchmark imports | 🟢 | Curated [KernelBench](./knowledge/benchmark_notes/kernelbench.md) and [ComputeEval](./knowledge/benchmark_notes/computeeval.md) adapters |
| Retrieval and knowledge | 🟢 | Docs, run examples, and patch-bearing episodes are queryable through a local index |
| Training packaging | 🟢 | Transition-rich trajectory export, SFT packaging, rollout smoke evaluation, and frozen training docs |
| Full training execution | 🔵 | Intentionally separated from the default local workflow and gated behind smoke validation |

> [!NOTE]
> The strongest path forward is **not** broadening every surface equally. The current priority is high-quality repair, diagnose, and reformulate traces that survive inspection, replay, and packaging cleanly.

## What You Get

### Core workflow surfaces

| Surface | Command family | Output |
| --- | --- | --- |
| Environment and hardware inspection | `gpc doctor` | Toolchain availability, hardware fingerprint, vendor details |
| Registry and benchmark inventory | `gpc task ...`, `gpc adapter ...` | Task listings, adapter summaries, curated case metadata |
| Run and build capture | `gpc run`, `gpc build` | Run bundle, command summary, tri-view artifacts, system traces, profiles |
| Task evaluation | `gpc eval` | Correctness, determinism, anti-hack, perf, and gate summary artifacts |
| Bundle analysis | `gpc inspect`, `gpc compare`, `gpc replay`, `gpc bundle` | Quality projections, lineage, proof bundles, replay validation |
| Agent environment | `gpc env action-space`, `gpc env scripted` | Compact observations and scripted multistep episodes |
| Offline data export | `gpc trajectory ...`, `gpc sft ...` | Trajectory datasets and packaged SFT corpora |
| Knowledge and retrieval | `gpc knowledge ...` | Mixed docs-plus-examples lookup |
| Training scaffolding | `gpc train ...`, `gpc rollout ...` | Config validation, smoke reports, held-out scripted baselines |

### Key design properties

- **Artifact-first:** every serious step emits inspectable bundle state rather than ad hoc console noise
- **Transition-aware:** episodes capture candidate lineage, patch hashes, patch kinds, and repair/reformulate transitions
- **Governed packaging:** run-level readiness and episode-level readiness are separated on purpose
- **Task-rich:** internal Triton/CUDA-style tasks coexist with curated public benchmark adapters
- **Training-targeted:** the training-preparation surface is oriented toward bounded tool-use on a strong sub-40B model, not open-ended frontier RL

## Repository Map

```text
gpu_rl/
├── README.md                                # top-level project overview, usage guide, and architecture map
├── pyproject.toml                           # package metadata and the `gpc` console entrypoint
├── .gitignore                               # runtime, dataset, and planning-artifact exclusions
├── configs/
│   └── training/
│       ├── first_target_splits_v1.json      # frozen train/dev split definition for the first training target
│       ├── rollout_debug_repair_heldout_v1.json
│       ├── rollout_debug_repair_v1.json     # scripted rollout configs for local and held-out evaluation
│       └── sft_qwen32b_debug_repair_lora.json
│                                             # first checked-in SFT smoke config for the initial training target
├── docs/
│   └── training/
│       ├── README.md                         # generic training-preparation overview and non-goals
│       ├── CHECKLIST.md                     # ordered preparation checklist before real training
│       ├── REMOTE_BOOTSTRAP.md              # bootstrap checklist for a dedicated training environment
│       ├── SMOKE_SEQUENCE.md                # exact smoke commands before real training
│       └── TRAINING_TARGET.md               # frozen first model/policy target and stop conditions
├── gpu_cockpit/
│   ├── cli/
│   │   └── main.py                          # public `gpc` CLI surface
│   ├── contracts/
│   │   ├── compare.py
│   │   ├── environment.py
│   │   ├── evidence.py
│   │   ├── patch.py
│   │   ├── replay.py
│   │   ├── summary.py
│   │   ├── training.py
│   │   └── trajectory.py                    # schema-first contract layer for bundles, episodes, and training configs
│   ├── engine/
│   │   ├── benchmark.py
│   │   ├── environment.py
│   │   ├── evaluator.py
│   │   ├── evidence.py
│   │   ├── inspector.py
│   │   ├── knowledge.py
│   │   ├── patching.py
│   │   ├── replay.py
│   │   ├── rollout.py
│   │   ├── runner.py
│   │   ├── sft.py
│   │   ├── training.py
│   │   └── trajectory.py                    # execution, eval, inspection, retrieval, and data-packaging engines
│   ├── backends/
│   │   ├── amd/                             # ROCm trace/profile normalization and AMD mirrored-path logic
│   │   └── nvidia/                          # Nsight, sanitizer, disassembly, and profile normalization
│   ├── executors/                           # local host and docker execution backends
│   ├── workloads/                           # adapter registration and workload-facing package helpers
│   └── artifacts/
│       └── schemas/
│                                             # exported JSON schemas for the contract layer
├── knowledge/
│   ├── README.md
│   ├── benchmark_notes/
│   ├── hardware_notes/
│   ├── operator_families/
│   ├── profiler_playbooks/
│   └── transformation_cards/                # human-written retrieval corpus for operators, bottlenecks, and transforms
├── scripts/
│   ├── build_training_manifest.py
│   ├── build_heldout_baseline_report.py
│   ├── build_first_target_training_assets.py
│   ├── export_schemas.py
│   ├── generate_transition_goldens.py
│   ├── run_training_preparation_verification.py
│   ├── smoke_rollout_eval.py
│   └── smoke_sft_train.py                   # reproducible builders and smoke paths for schemas, datasets, and training assets
├── tests/
│   ├── golden_datasets/
│   ├── golden_episodes/
│   ├── golden_retrieval/
│   ├── golden_runs/
│   ├── test_environment.py
│   ├── test_evaluator.py
│   ├── test_inspector.py
│   ├── test_knowledge.py
│   ├── test_sft.py
│   ├── test_training.py
│   └── test_trajectory.py                  # regression suite plus checked-in golden bundles and training-facing fixtures
└── workloads/
    ├── baselines/
    ├── benchmarks/
    ├── fixtures/
    ├── public_benchmarks/
    ├── reference/
    ├── tasks/
    └── tests/                              # task specs, baselines, curated imports, reference kernels, and hook scripts
```

## Project Documents

These documents freeze the semantics and boundaries that matter for the first training wave:

| Document | Purpose |
| --- | --- |
| [`docs/PROJECT_SCOPE.md`](./docs/PROJECT_SCOPE.md) | Defines the finished training-preparation scope, deferred training execution work, and explicit non-goals |
| [`docs/GLOSSARY.md`](./docs/GLOSSARY.md) | Freezes the shared vocabulary across runs, episodes, governance, replay, and training docs |
| [`docs/FIRST_WAVE_TASKS.md`](./docs/FIRST_WAVE_TASKS.md) | Inventory of the first-wave training task families and why each is in scope |
| [`docs/BENCHMARK_POLICY.md`](./docs/BENCHMARK_POLICY.md) | Policy for how public benchmark traces participate in packaging, reporting, and training |
| [`docs/AMD_SCOPE.md`](./docs/AMD_SCOPE.md) | Explicit narrow-scope AMD mirrored-path boundary for the current program |
| [`docs/OBSERVABILITY_SURFACE.md`](./docs/OBSERVABILITY_SURFACE.md) | Frozen local scope for build, trace, profile, sanitizer, and bottleneck artifacts |
| [`docs/REPLAY_COMPARE.md`](./docs/REPLAY_COMPARE.md) | Replay, compare, and proof-bundle semantics for transition-aware review and packaging |
| [`docs/DATA_GOVERNANCE.md`](./docs/DATA_GOVERNANCE.md) | Run-level readiness versus episode-level governance and training-example semantics |
| [`docs/POLICY_INTERFACE.md`](./docs/POLICY_INTERFACE.md) | First-wave action surface, observation model, and rollout semantics |
| [`docs/RETRIEVAL_GUIDE.md`](./docs/RETRIEVAL_GUIDE.md) | Retrieval corpus structure and recommended query patterns |
| [`docs/training/`](./docs/training) | Checked-in training-preparation package, smoke path, and training target documentation |

## Installation and Environment

### Minimal install

For schema work, bundle inspection, knowledge indexing, and non-GPU smoke paths:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

### Practical local environment

| Goal | Requirements |
| --- | --- |
| Basic package and CLI use | Python `3.12+`, `pip`, editable install |
| Triton/CUDA task execution | `torch`, `triton`, working CUDA runtime |
| Rich NVIDIA observability | `nsys`, `ncu`, `compute-sanitizer`, optional `nvcc` |
| Narrow AMD parity checks | ROCm runtime plus `rocprof` / `rocprofv3`, `rocminfo`, `rocm-smi`, `hipcc` |
| End-to-end smoke path | The above plus enough GPU support for the reference Triton tasks |

> [!TIP]
> The repository is intentionally usable in partial environments. Missing profiler or vendor tools degrade the corresponding run surfaces instead of collapsing the whole CLI.

### Compatibility matrix

| Surface | NVIDIA | AMD | CPU-only |
| --- | --- | --- | --- |
| Contracts / schemas / inspection | 🟢 | 🟢 | 🟢 |
| Knowledge index and retrieval | 🟢 | 🟢 | 🟢 |
| Trajectory export / SFT packaging | 🟢 | 🟢 | 🟢 |
| Triton internal tasks | 🟢 | 🟡 | 🔴 |
| Public CUDA benchmark adapters | 🟢 | 🔴 | 🔴 |
| `nsys` / `ncu` / sanitizer-backed runs | 🟢 | 🔴 | 🔴 |
| ROCm mirrored profile path | 🔴 | 🟡 | 🔴 |
| Local smoke training scaffolding | 🟢 | 🟢 | 🟢 |

## Quick Start

### 1. Inspect your machine

```bash
gpc doctor
```

### 2. See what tasks and benchmark adapters exist

```bash
gpc task list
gpc adapter list
gpc adapter summary kernelbench
gpc adapter summary computeeval
```

### 3. Run a real task with evaluation gates

```bash
gpc eval \
  --task task/reduction_debug/eval/v1 \
  --determinism-runs 2 \
  -- python3 workloads/reference/triton_row_sum_patchable_candidate.py --benchmark-repeats 5
```

### 4. Inspect or compare run bundles

```bash
gpc inspect runs/<run_id> --section quality
gpc compare runs/<run_a> runs/<run_b>
gpc replay runs/<run_id>
gpc bundle runs/<run_id> --full
```

### 5. Produce transition-rich data

```bash
python3 scripts/build_first_target_training_assets.py
python3 scripts/build_heldout_baseline_report.py
python3 scripts/smoke_sft_train.py
python3 scripts/smoke_rollout_eval.py
python3 scripts/run_training_preparation_verification.py
```

## CLI Walkthrough

### Command families at a glance

| Family | Subcommands / flags worth knowing |
| --- | --- |
| `doctor` | Local hardware and toolchain discovery |
| `task`, `adapter` | Task registry, adapter inventory, benchmark case summaries |
| `build`, `run`, `eval`, `bench` | Build/disassembly, execution, evaluation, benchmarking |
| `inspect`, `compare`, `replay`, `bundle`, `runs` | Bundle analysis, lineage inspection, proof export, index queries |
| `trajectory`, `env`, `sft` | Episode generation, bounded environment helpers, SFT packaging |
| `knowledge` | Build local index, free-text query, retrieve similar tasks |
| `train`, `rollout` | Config validation, smoke SFT reports, scripted rollout suites |

### Example: build-only tri-view capture

```bash
gpc build \
  --task task/attention_score/eval/v1 \
  --triton-build-spec workloads/reference/triton_attention_score_kernel.py:get_build_spec \
  --source-file workloads/reference/triton_attention_score_kernel.py
```

Useful when the immediate goal is inspecting:

- generated PTX
- SASS or disassembly output
- Triton IR stages
- source-to-PTX/SASS mapping summaries

### Example: profiled run

```bash
gpc run \
  --task task/attention_score/eval/v1 \
  --trace-system \
  --profile-kernel \
  --profile-pack quick \
  --sanitize \
  --sanitize-tool memcheck \
  --emit-disassembly \
  --triton-build-spec workloads/reference/triton_attention_score_kernel.py:get_build_spec \
  -- python3 workloads/reference/triton_attention_score_candidate.py --benchmark-repeats 5
```

### Example: bounded scripted environment episode

```bash
gpc env scripted \
  --task task/reduction_debug/eval/v1 \
  --out /tmp/reduction_debug_episode.json \
  --step-budget 12 \
  --workflow debug \
  --with-build \
  --triton-build-spec workloads/reference/triton_row_sum_kernel.py:get_build_spec \
  -- python3 workloads/reference/triton_row_sum_patchable_candidate.py --benchmark-repeats 5
```

### Example: transition-aware SFT packaging

```bash
gpc sft package \
  datasets/first_target_transition_train_v1 \
  --out-dir datasets/first_target_sft_train_v1 \
  --split train \
  --patch-bearing-only \
  --governance usable_positive_sft \
  --governance usable_negative_debug \
  --governance usable_negative_transition \
  --transition-kind repaired \
  --transition-kind reformulated \
  --transition-kind patch_applied \
  --verb debug \
  --verb reformulate
```

### Example: mixed docs-plus-example retrieval

```bash
gpc knowledge build-index

gpc knowledge query \
  --query "mask bug failed repair register pressure" \
  --verb debug \
  --prefer-mixed \
  --limit 8
```

<details>
<summary><strong>Selected CLI idioms</strong></summary>

#### Inspect only one section

```bash
gpc inspect runs/<run_id> --section build
gpc inspect runs/<run_id> --section profile
gpc inspect runs/<run_id> --section replay
gpc inspect runs/<run_id> --section quality
```

#### Adapter-driven benchmarking

```bash
gpc bench --adapter kernelbench --case case/kernelbench/level1/40_layernorm/v0_1
gpc bench --adapter computeeval --case case/computeeval/2025_1/cuda_16/v1
```

#### Training smoke validation

```bash
gpc train validate-config configs/training/sft_qwen32b_debug_repair_lora.json
gpc rollout scripted configs/training/rollout_debug_repair_heldout_v1.json --out-dir /tmp/heldout_rollout
```

</details>

## Task and Benchmark Surface

### Internal workload families

| Task family | Verb | Backend | What it exercises |
| --- | --- | --- | --- |
| `reduction_sum` | `optimize` | Triton | Row-wise reduction kernels and correctness/perf contracts |
| `reduction_debug` | `debug` | Triton | Broken masking/repair-oriented traces with patch-bearing fixes |
| `routing_argmax` | `optimize` | Triton | Routing/indexing kernels |
| `topk_router` | `optimize` | Triton | Routing top-k behavior and benchmarkable operator logic |
| `attention_score` | `optimize` | Triton | Tiled causal attention-score kernels |
| `attention_reformulate` | `reformulate` | Triton | Weak-vs-optimized strategy transitions |
| `kv_cache_gather` | `optimize` | Triton | KV-cache gather behavior and attention-adjacent memory access patterns |
| `profile_diagnose` | `diagnose` | CUDA | Bottleneck interpretation and profiler-conditioned analysis |
| `smoke` / `smoke_eval` | `diagnose` | CUDA / Triton | Minimal substrate checks |
| `amd_smoke` | `diagnose` | HIP | Narrow AMD mirrored-path validation |

### Public benchmark adapters

| Adapter | Curated cases | Source | Notes |
| --- | --- | --- | --- |
| [`kernelbench`](./knowledge/benchmark_notes/kernelbench.md) | `11` | [KernelBench](https://github.com/ScalingIntelligence/KernelBench) | Activation, normalization, reduction, indexing, matmul, attention-adjacent coverage |
| [`computeeval`](./knowledge/benchmark_notes/computeeval.md) | `8` | [ComputeEval](https://github.com/NVIDIA/compute-eval) | CUDA kernel launch, streams, reductions, CUB, Thrust, and metadata-heavy variants |

### Why both internal and public tasks exist

- **Internal tasks** carry richer semantics for repair, reformulation, hidden failures, patch transitions, and training governance.
- **Public adapters** provide external provenance, broader operator coverage, and a reality check against overfitting to bespoke tasks.
- The packaging defaults deliberately prefer transition-rich internal traces over thin public benchmark wrappers unless explicitly configured otherwise.

## Artifact Model

### What a run bundle looks like

Every serious action in the cockpit writes a run directory that can be inspected, replayed, compared, or packaged.

```text
runs/<run_id>/
├── manifest.json
├── events.jsonl
├── summary.json
├── summary.md
├── prompt/
│   └── task_spec.json
├── meta/
│   ├── doctor_report.json
│   ├── hardware_fingerprint.json
│   └── task_spec_full.json
├── command/
│   ├── stdout.txt
│   ├── stderr.txt
│   └── summary.json
├── correctness/
│   ├── correctness.json
│   ├── determinism.json
│   └── *_summary.json
├── eval/
│   ├── anti_hack_report.json
│   ├── eval_envelope.json
│   └── gate_summary.json
├── perf/
│   ├── raw_timings.json
│   └── benchmark.json
├── build/
│   ├── build_record.json
│   ├── source_map_summary.json
│   ├── tri_view.json
│   └── source_ptx_sass_map.json
├── patches/
│   ├── request.json
│   ├── unified_diff.patch
│   └── applied_patch.json
├── candidate/
│   ├── state.json
│   └── transition.json
└── replay/
    ├── command.json
    ├── environment.json
    └── replay_pack.json
```

### Important artifact families

| Family | Why it matters |
| --- | --- |
| `summary.json` / `summary.md` | Fast human and programmatic overview of run state |
| `eval/eval_envelope.json` | Core pass/fail and reward-bearing evaluation gates |
| `perf/benchmark.json` | Perf gate inputs and baseline comparisons |
| `build/*` | Triton IR, PTX, SASS, source map summaries, tri-view artifacts |
| `patches/*` and `candidate/*` | Candidate lineage and transition-aware training traces |
| `replay/*` | Rehydration metadata for reproducibility and proof bundles |

## Knowledge Layer

The local knowledge base is intentionally small and structured rather than sprawling.

| Subdirectory | Content |
| --- | --- |
| [`knowledge/operator_families/`](./knowledge/operator_families) | Operator-specific notes such as reduction, attention score, KV-cache gather, and profile diagnosis |
| [`knowledge/profiler_playbooks/`](./knowledge/profiler_playbooks) | Bottleneck-oriented interpretive guides such as memory-bound or occupancy-limited cases |
| [`knowledge/transformation_cards/`](./knowledge/transformation_cards) | Tactical strategy cards such as tiling, vectorization, staging, masking, and layout changes |
| [`knowledge/benchmark_notes/`](./knowledge/benchmark_notes) | Curated notes about imported public benchmarks |
| [`knowledge/hardware_notes/`](./knowledge/hardware_notes) | Vendor and platform-specific constraints such as AMD parity scope |

Retrieval is designed to return a **mix** of:

- docs
- prior run examples
- patch-bearing repair traces
- reformulation examples
- similar tasks

That bias is deliberate: useful training and debugging context usually mixes prose with concrete examples.

## Transition-Aware Data Products

### Why the collection layer is transition-aware

A large fraction of GPU-agent signal is not “one prompt, one answer.” The important learning unit is often:

1. inspect a candidate
2. identify a failure mode or weak strategy
3. patch or transform the candidate
4. rebuild, benchmark, compare, and re-evaluate
5. decide whether the resulting trace is usable for training

The repository therefore treats the following as first-class:

- `patch_candidate` transitions
- candidate lineage and parent-child state
- episode-level governance separate from run-level readiness
- usable negative traces for failed repair and failed reformulation

### Checked-in training-oriented fixtures

| Fixture class | Location |
| --- | --- |
| Transition datasets | [`tests/golden_datasets/transition_collection_v1`](./tests/golden_datasets/transition_collection_v1) |
| Negative transition datasets | [`tests/golden_datasets/transition_negative_collection_v1`](./tests/golden_datasets/transition_negative_collection_v1) |
| Packaged SFT examples | [`tests/golden_datasets/transition_sft_v1`](./tests/golden_datasets/transition_sft_v1) |
| Negative packaged SFT examples | [`tests/golden_datasets/transition_negative_sft_v1`](./tests/golden_datasets/transition_negative_sft_v1) |
| Episode fixtures | [`tests/golden_episodes/`](./tests/golden_episodes) |
| Run-bundle fixtures | [`tests/golden_runs/`](./tests/golden_runs) |

## Training Preparation

### First target

The first checked-in training target is documented in [docs/training/TRAINING_TARGET.md](./docs/training/TRAINING_TARGET.md):

- model family: `Qwen/Qwen3.5-Coder-32B`
- adaptation strategy: `LoRA` / PEFT-first
- first policy: bounded tool-use
- primary verbs: `debug`, `diagnose`
- secondary verb: `reformulate`
- open-ended `optimize`: intentionally deferred as a first-wave RL target

### Training-preparation contents

| File | Purpose |
| --- | --- |
| [`docs/training/README.md`](./docs/training/README.md) | Scope and non-goals of the training-preparation package |
| [`docs/training/REMOTE_BOOTSTRAP.md`](./docs/training/REMOTE_BOOTSTRAP.md) | Ordered bootstrap checklist for the target training environment |
| [`docs/training/SMOKE_SEQUENCE.md`](./docs/training/SMOKE_SEQUENCE.md) | Exact smoke commands to run before real training |
| [`configs/training/first_target_splits_v1.json`](./configs/training/first_target_splits_v1.json) | Frozen train/dev split manifest for the first target |
| [`configs/training/rollout_debug_repair_heldout_v1.json`](./configs/training/rollout_debug_repair_heldout_v1.json) | Held-out scripted rollout config |
| [`configs/training/sft_qwen32b_debug_repair_lora.json`](./configs/training/sft_qwen32b_debug_repair_lora.json) | Initial smoke SFT config |

### Smoke path

```bash
python3 scripts/build_first_target_training_assets.py
python3 scripts/build_training_manifest.py
python3 scripts/smoke_sft_train.py
python3 scripts/build_heldout_baseline_report.py
python3 scripts/smoke_rollout_eval.py
```

Supporting project docs:

- [`CONTRIBUTING.md`](./CONTRIBUTING.md)
- [`CHANGELOG.md`](./CHANGELOG.md)
- [`SECURITY.md`](./SECURITY.md)

> [!WARNING]
> Treat the smoke path as a gate before any nontrivial training run. The goal is to validate datasets, configs, and rollout assumptions mechanically before spending accelerator time.

## Development Workflow

### Package, test, and schema workflow

```bash
pip install -e .
python3 scripts/export_schemas.py
python3 -m unittest discover -s tests -v
```

### Common maintenance loops

```bash
# Rebuild checked-in transition fixtures
python3 scripts/generate_transition_goldens.py

# Rebuild train/dev assets for the first training target
python3 scripts/build_first_target_training_assets.py

# Validate training config and rollout config
gpc train validate-config configs/training/sft_qwen32b_debug_repair_lora.json
gpc rollout scripted configs/training/rollout_debug_repair_heldout_v1.json --out-dir /tmp/heldout_rollout
```

### Suggested local review order

1. Start with [`gpu_cockpit/cli/main.py`](./gpu_cockpit/cli/main.py) to see the public command surface.
2. Read [`gpu_cockpit/engine/runner.py`](./gpu_cockpit/engine/runner.py), [`gpu_cockpit/engine/evaluator.py`](./gpu_cockpit/engine/evaluator.py), and [`gpu_cockpit/engine/inspector.py`](./gpu_cockpit/engine/inspector.py) for the core run lifecycle.
3. Read [`gpu_cockpit/engine/environment.py`](./gpu_cockpit/engine/environment.py), [`gpu_cockpit/engine/trajectory.py`](./gpu_cockpit/engine/trajectory.py), and [`gpu_cockpit/engine/sft.py`](./gpu_cockpit/engine/sft.py) for the training-facing data model.
4. Read [`docs/training/TRAINING_TARGET.md`](./docs/training/TRAINING_TARGET.md) before changing training assumptions.

## Tracked vs Generated State

### Checked in

- source code
- tests
- schemas
- benchmark metadata
- golden runs / episodes / datasets used for regression coverage
- training docs and training configs

### Intentionally ignored

- `runs/`
- `datasets/`
- `docs_tmp/`
- `artifacts/`
- generated `knowledge/index/`

This split is deliberate:

- checked-in goldens define the stable verification surface
- generated runtime outputs remain local and disposable

## Closing Notes

If you are here for the shortest orientation path:

- read [`docs/training/TRAINING_TARGET.md`](./docs/training/TRAINING_TARGET.md)
- run `gpc doctor`
- inspect `gpc --help`
- execute one `gpc eval` task
- inspect a run bundle
- build the training assets

If you are here to extend the project:

- prefer richer internal repair and reformulate traces over shallow benchmark breadth
- preserve candidate lineage and governance semantics
- treat patch-bearing episodes as the highest-signal assets in the current system

If you are here to start training:

- finish validation on the checked-in smoke path
- use the checked-in training docs and configs
- validate on the intended training environment before launching larger jobs
- treat the smoke sequence as the gate to more expensive runs
