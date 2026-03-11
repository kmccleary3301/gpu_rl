# Observability and Build Surface

This document freezes the intended local observability surface for the environment/data program.

## Goals

- make build, trace, profile, sanitizer, and bottleneck artifacts first-class
- support inspection and comparison without requiring vendor UIs
- degrade cleanly when optional tools are unavailable
- keep the artifact surface stable enough for replay, retrieval, and training-facing inspection

## Build and disassembly artifacts

The build surface is centered on Triton-oriented inspection:

- source snapshots
- TTIR / TTGIR / LLIR
- PTX
- SASS when available
- source map summary
- tri-view artifacts

These are surfaced through:

- `gpc build`
- `gpc run --emit-disassembly`
- `inspect --section build`
- build-aware compare projections

## NVIDIA observability surface

The intended local NVIDIA path includes:

- system tracing with `nsys`
- normalized kernel profiling
- normalized sanitizer findings
- bottleneck summaries and warnings
- compare projections over build/profile deltas

The contracts are stable enough for:

- run summaries
- replay packs
- checked-in golden bundles
- retrieval indexing
- transition-aware debug and reformulate episodes

## AMD mirrored surface

AMD remains intentionally narrower. The current mirrored surface includes:

- ROCm tool detection and realistic metadata parsing
- normalized AMD trace summaries
- normalized AMD profile summaries
- replay / inspect / compare support on AMD-shaped bundles

It does not claim broad feature parity with NVIDIA tooling.

See [`docs/AMD_SCOPE.md`](./AMD_SCOPE.md) for the explicit scope boundary.

## Sanitizer coverage

The project normalizes sanitizer findings into stable categories and preserves degraded-coverage warnings when tools are unavailable.

The intended local surface is:

- memcheck-style findings
- race / sync / initialization category normalization
- stable severity and category summaries for inspection and compare
- inspection and comparison projections that can distinguish correctness failures from safety or synchronization failures

This is intended for:

- failure triage
- trainworthiness filtering
- replay / proof-bundle inspection

It is not intended to guarantee identical vendor coverage on every host.

## Graceful degradation

Optional tooling is not assumed to exist everywhere.

When a tool is missing, the expected behavior is:

- keep the run bundle valid
- surface warnings explicitly
- preserve enough metadata for replay and inspection
- avoid silently fabricating complete evidence

## Exit criterion for this surface

The local observability surface is considered complete when:

- build artifacts are stable and inspectable
- trace/profile/sanitizer outputs normalize into checked-in contracts
- compare and replay can consume those artifacts
- goldens and regression tests cover the expected projections
- missing-tool behavior is explicit rather than implicit
