# Phase 5A Control Surface Refresh v2

## Broader Four-Task Tranche

- `base_controlfix_v2`: `3/4`
  - wins on `softmax_wide`, `routing_argmax_hard`, and `sum_reduction`
  - misses `attention_score`
- `artifact_feedback_distill_v4`: `3/4`
  - wins on `attention_score`, `softmax_wide`, and `routing_argmax_hard`
  - misses `sum_reduction`
- `artifact_feedback_distill_v5`: `2/4`
  - wins on `routing_argmax_hard` and `sum_reduction`
  - misses `attention_score` and `softmax_wide`

## Read

`distill_v5` repaired the exact `sum_reduction` miss it targeted, but it gave back too much elsewhere.

So the best current broader-tranche learner remains `artifact_feedback_distill_v4`, and the best current broader control surface remains `base_controlfix_v2`.

## Next Focus

Move the frontier tranche upward on the GPT-5.4 side instead of spending another immediate turn on learner reweighting.
