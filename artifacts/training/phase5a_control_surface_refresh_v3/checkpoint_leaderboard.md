# Phase 5A Control Surface Refresh v3

## Broader Four-Task Tranche

- `base_controlfix_v2`: `3/4`
  - wins on `softmax_wide`, `routing_argmax_hard`, and `sum_reduction`
  - misses `attention_score`
- `artifact_feedback_distill_v4`: `3/4`
  - wins on `attention_score`, `softmax_wide`, and `routing_argmax_hard`
  - misses `sum_reduction`
- `artifact_feedback_distill_v6`: `3/4`
  - wins on `softmax_wide`, `routing_argmax_hard`, and `sum_reduction`
  - misses `attention_score`
- `artifact_feedback_distill_v5`: `2/4`
  - wins on `routing_argmax_hard` and `sum_reduction`
  - misses `attention_score` and `softmax_wide`

## Read

`distill_v6` kept learner parity on the broader tranche after the refreshed harder-KV teacher success, but it still did not produce a strict `4/4` winner.

So the broader decision remains:

- best control surface: `base_controlfix_v2`
- best broader learners so far: `distill_v4` and `distill_v6` in a tie

## Next Focus

Check whether `distill_v6` improved the harder KV-inclusive surface, which is the main reason to keep the freeze-v4 refresh.
