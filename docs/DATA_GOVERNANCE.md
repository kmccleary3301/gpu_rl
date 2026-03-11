# Data Governance

This document summarizes how run-level readiness and episode-level governance are meant to be used.

## Run-Level Readiness

Run-level readiness determines whether a single run bundle is eligible for:

- benchmark reporting
- SFT collection
- RL reward traces

That decision is driven by:

- required artifact completeness
- replay completeness
- build/profile completeness
- eval completeness
- provenance completeness

## Episode-Level Governance

Episode-level governance determines whether a trajectory episode is eligible for:

- positive SFT inclusion
- usable negative inclusion
- benchmark-only retention
- exclusion

### Current governance classes

- `usable_positive_sft`
- `usable_negative_debug`
- `usable_negative_transition`
- `benchmark_only`
- `unusable`

## Packaging Rule

The packaging defaults should:

- include governed positive traces
- include governed usable negatives when explicitly configured
- exclude `benchmark_only` and `unusable` traces by default

## Why the Separation Exists

Some episodes are valuable because of the **transition**, even if the terminal run would not qualify as a high-quality RL reward trace by itself.

That is why:

- run-level readiness is stricter
- episode-level governance can still preserve useful repair and reformulate traces
