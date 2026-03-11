# Glossary

## Core Objects

### Candidate
A concrete code state associated with a run, patch, or build/eval outcome.

### Candidate lineage
The parent-child relationship between candidates across patch, repair, revert, or reformulate steps.

### Run bundle
A directory of structured artifacts for one executed run or projection, including summaries, evaluation outputs, build outputs, and replay metadata.

### Proof bundle
A portable export of a run bundle intended for inspection or sharing without requiring the entire runtime directory tree.

### Replay pack
The structured metadata required to validate or reconstruct the execution context of a run bundle.

## Task and Episode Terms

### Task verb
The primary task mode, such as `diagnose`, `debug`, `reformulate`, or `optimize`.

### Transition
A state change between candidates or workflow phases, such as `patch_applied`, `repaired`, `reformulated`, or `reverted`.

### Patch-bearing episode
An episode that contains at least one explicit patch application step and corresponding candidate transition metadata.

### Repair episode
A patch-bearing episode whose goal is to fix a broken candidate while preserving or restoring correctness.

### Reformulate episode
A patch-bearing episode whose goal is to change implementation strategy while preserving semantics.

## Governance Terms

### Run-level readiness
A decision about whether a single run bundle has enough evidence and provenance to count toward benchmark reporting, SFT collection, or RL reward traces.

### Episode-level governance
A decision about whether a multi-step episode is usable for SFT or other collection purposes, independent from the readiness of any single terminal run.

### Positive SFT example
An episode that is governance-eligible for supervised fine-tuning as a successful trace.

### Usable negative example
An episode that failed in a meaningful, well-evidenced way and is still useful for training or analysis.

### Benchmark-only trace
A trace useful for benchmark accounting or comparison, but not included in default training packaging.

### Unusable trace
A run or episode that lacks sufficient correctness, lineage, or evidence quality to be included by default in training-facing corpora.

## Evidence Terms

### Evidence quality
The structured completeness score over required artifacts, replay metadata, build/profile artifacts, eval coverage, and provenance fields.

### Trainworthiness
A summary judgment about whether a run or episode is a good positive example, good negative example, benchmark-only item, or unusable artifact.
