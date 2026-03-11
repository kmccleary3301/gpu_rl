# Replay, Compare, and Proof Bundles

This document freezes the role of replay, compare, and proof-bundle tooling in the training-preparation scope.

## Replay

Replay metadata exists to preserve:

- command invocation
- environment context
- task refs
- candidate lineage
- patch refs and transition refs
- required artifact locations

Replay is not only for rerunning commands. In this project it is also a machine-readable audit surface for:

- trainworthiness review
- bundle completeness checks
- evidence-quality projections
- golden fixture stability

## Compare

Compare is transition-aware by design.

The intended local compare surface includes:

- correctness and gate deltas
- perf deltas
- build hash and tri-view deltas
- stage-level build deltas for source / TTIR / TTGIR / LLIR / PTX / SASS when available
- patch presence and patch kind
- candidate lineage and parent-child relationships
- trainworthiness and evidence-quality changes

The important use cases are:

- repair succeeded vs repair failed
- reformulation improved perf vs regressed perf
- evidence improved vs remained thin
- unrelated runs vs parent-child candidate transitions

## Proof bundles

Proof bundles package the subset of a run that is needed for external review without requiring the full raw workspace.

They matter for:

- reproducibility
- review of benchmark claims
- review of repair or reformulate transitions
- long-lived golden fixtures

## Intended training role

Replay and compare are first-class inputs to:

- patch-bearing trajectory export
- SFT packaging metadata
- retrieval over prior failures and prior repairs
- training-preparation validation before expensive training

They are not post-hoc convenience commands bolted onto the side of the repo.

## Exit criterion for this surface

The training-preparation replay/compare surface is considered complete when:

- candidate lineage survives into replay
- inspect and compare explain training usefulness directly
- proof bundles export cleanly
- checked-in goldens exercise the important transition classes
- the training docs can rely on replay and compare semantics without private tribal knowledge
