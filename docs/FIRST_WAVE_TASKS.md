# First-Wave Task Inventory

This document classifies the current task surface into the categories that matter for the first training wave.

## First-Wave Training Tasks

| Task family | Verb | Role |
| --- | --- | --- |
| `reduction_debug` | `debug` | Primary repair task with patch-bearing positive and usable-negative traces |
| `profile_diagnose` | `diagnose` | Primary diagnosis task with profiler-conditioned outputs |
| `attention_reformulate` | `reformulate` | Primary transformation task with baseline compare and patch-bearing transitions |

## Transfer-Phase Tasks

| Task family | Verb | Role |
| --- | --- | --- |
| `reduction_sum` | `optimize` | Transfer-phase optimization target |
| `routing_argmax` | `optimize` | Transfer-phase Triton indexing/routing workload |
| `topk_router` | `optimize` | Transfer-phase routing workload |
| `attention_score` | `optimize` | Transfer-phase attention kernel target |
| `kv_cache_gather` | `optimize` | Transfer-phase attention-adjacent memory target |

## Diagnostic / Substrate Tasks

| Task family | Verb | Role |
| --- | --- | --- |
| `smoke` | `diagnose` | Minimal substrate validation |
| `smoke_eval` | `diagnose` | Minimal evaluation-path validation |
| `amd_smoke` | `diagnose` | Narrow mirrored-path validation for AMD fixtures |

## Public Benchmark Tasks

Public benchmark cases remain in scope, but their default role is:

- external provenance
- retrieval examples
- benchmark reporting and transfer evaluation

They are **not** the default first-wave training source unless explicitly packaged.

### Imported families

- `KernelBench`
- `ComputeEval`

## Inclusion Rule for New Tasks

A task should only be added to the first-wave training set if it has:

- clear hidden-test semantics
- deterministic inputs
- stable build/eval surfaces
- explicit baseline/reference behavior
- meaningful positive or usable-negative transition traces
