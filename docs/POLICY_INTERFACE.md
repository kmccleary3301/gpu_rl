# Policy Interface

This document freezes the intended first-wave agent interface.

## Objective

The first policy is a bounded tool-use policy over GPU-agent tasks, not an unconstrained code-generation policy.

## First-Wave Action Set

| Action | Purpose |
| --- | --- |
| `run` | Execute a command and create a run bundle |
| `build` | Emit build/disassembly artifacts |
| `bench` | Emit a perf bundle |
| `eval` | Run correctness and reward-bearing evaluation hooks |
| `inspect` | Inspect a run bundle section |
| `inspect_build` | Focused build/disassembly inspection |
| `inspect_profile` | Focused profile/bottleneck inspection |
| `inspect_quality` | Focused evidence / trainworthiness inspection |
| `patch_candidate` | Apply a scripted patch and emit candidate-transition artifacts |
| `compare` | Compare two run bundles |
| `replay` | Validate replay completeness |
| `adapter_show` | Load a benchmark case plus derived task |
| `knowledge_query` | Query local docs, run examples, and episode examples |

## Observation Principles

Observations should be:

- compact
- bundle-backed
- stable across runs
- specific to the current step goal

### Common observation shapes

- run summary
- build projection
- eval summary
- quality projection
- profile projection
- comparison projection
- replay validation
- knowledge results
- candidate transition summary

## Terminal States

The first-wave environment uses bounded terminal semantics such as:

- `success`
- `failure`
- `blocked`
- `budget_exhausted`

## Reward Principles

The first-wave reward structure should emphasize:

- correctness
- determinism
- anti-hack compliance
- patch success
- semantic preservation
- bounded tool cost

Evidence quality remains primarily a governance/filtering signal rather than a direct reward target.
