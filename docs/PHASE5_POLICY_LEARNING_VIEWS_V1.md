# Phase 5 Policy-Learning Views v1

The canonical Phase 5 hard freeze now supports four explicit data-export views:

- `artifact_feedback_distill_v2`
  - Source: `artifacts/training/phase5_hard_trace_freeze_v2/optimize_trace_manifest.json`
  - Builder: `scripts/build_phase5_artifact_feedback_distill_dataset.py`
  - Intended use: teacher-conditioned distillation on observation -> action decisions with hard-slice metadata.

- `teacher_corrected_v2`
  - Source: `artifacts/training/phase5_hard_trace_freeze_v2/optimize_trace_manifest.json`
  - Builder: `scripts/build_phase5_teacher_corrected_dataset.py`
  - Intended use: on-policy teacher correction or iterative self-distillation against frozen hard traces.

- `phase5_freeze_rwr_v2`
  - Source: `artifacts/training/phase5_hard_trace_freeze_v2/optimize_trace_manifest.json`
  - Builder: `scripts/build_phase5_freeze_rwr_dataset.py`
  - Intended use: narrow RL-style regression / reward-weighted policy updates.

- `phase5_pairwise_ranking_v2`
  - Source: `artifacts/training/phase5_hard_trace_freeze_v2/optimize_trace_manifest.json`
  - Builder: `scripts/build_phase5_pairwise_ranking_dataset.py`
  - Intended use: lightweight preference/ranking experiments over matched task and provenance groups.

Why keep these separate:

- Distill should privilege strong teacher actions on the hard slice.
- Teacher correction should stay close to the same observation/action format, but remain explicitly tagged as a correction view.
- Narrow RL needs reward-scored episodes and should not be conflated with pure teacher imitation.
- Pairwise ranking should remain a sibling/task-group comparison surface, not be mixed into token-level supervision by default.

All four views are meant to be produced from the same environment semantics, provenance rules, and contamination-audited freeze. If a future task family requires bespoke repackaging, that is a substrate regression and should be treated as such.
