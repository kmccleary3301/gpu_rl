# Phase 5A Control Surface Refresh v1

## Scope

Compare the strongest broader local control surface against the strongest control-aware learner on the four-task Phase 5A tranche:

- `attention_score`
- `kernelbench/level1/23_softmax_wide`
- `routing_argmax_hard`
- `kernelbench/level1/47_sum_reduction`

## Result

- `base_controlfix_v2`: `3/4`
  - wins on `softmax_wide`, `routing_argmax_hard`, and `sum_reduction`
  - misses `attention_score`
- `artifact_feedback_distill_v4`: `3/4`
  - wins on `attention_score`, `softmax_wide`, and `routing_argmax_hard`
  - misses `sum_reduction`

## Read

The control-aware learner reached parity with the broader control-surface winner, but not clear superiority.

The remaining gap is narrow and specific:

- `distill_v4` learned the broader hard-task trio better than earlier learners
- `distill_v4` still fails to execute the `sum_reduction` closeout pattern that the broader base control winner executes cleanly

## Next Focus

Repair `sum_reduction` without giving back the new `attention_score` success.
