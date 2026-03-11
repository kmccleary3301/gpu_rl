# Project Scope

This document freezes the intended scope of `gpu_rl` before any real training runs are executed on dedicated accelerator hardware.

## Purpose

`gpu_rl` is a research and systems substrate for:

- GPU-agent task execution
- artifact-first benchmarking and evaluation
- patch-bearing repair and reformulate traces
- transition-aware trajectory export
- SFT packaging and rollout scaffolding
- training preparation

The project is intentionally structured so that the environment and data surface can be completed before expensive SFT or RL runs begin.

## In Scope for the Training-Preparation Program

- task registry and benchmark adapter surface
- internal Triton/CUDA/HIP-backed workload families
- curated public benchmark imports
- run bundles, proof bundles, replay packs, and compare surfaces
- correctness, determinism, anti-hack, and perf gates
- build, tri-view, disassembly, profile, and bottleneck artifacts
- knowledge and retrieval surfaces
- patch-bearing candidate transitions
- trajectory export and SFT packaging
- rollout configs, smoke training configs, and training docs
- checked-in golden datasets, run bundles, and episode fixtures

## Explicitly Deferred Until the Training Environment

- nontrivial SFT optimization runs
- online RL rollouts with learned policies
- reward-shaping iteration from real training results
- trainer-scale optimization work driven by live accelerator experiments
- distributed orchestration beyond the current smoke scaffolding

## Intentionally Out of Scope

- dense fine-tuning of frontier-scale models
- full vendor parity across every NVIDIA and AMD profiling/debug feature
- generalized multi-node orchestration platform work
- production-serving infrastructure unrelated to research or training preparation
- benchmark leaderboard maximization as an end in itself

## First-Wave Training Focus

The first training wave is defined as:

- bounded tool-use
- primary verbs: `debug`, `diagnose`
- secondary verb: `reformulate`
- `optimize` as a transfer-phase task family rather than the first RL target

## Completion Definition

The training-preparation program is complete when:

- the first-wave tasks and semantics are frozen
- the environment surface is stable enough for repeatable data collection
- the data products are governance-clean and training-ready
- the training-preparation package is mechanically executable on the target training environment
- the only major remaining work is actual SFT / RL execution and iteration
