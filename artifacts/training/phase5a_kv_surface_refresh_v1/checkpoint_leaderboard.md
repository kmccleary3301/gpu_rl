# Phase 5A KV Surface Refresh v1

## Tight Frontier Surface

- `artifact_feedback_distill_v3`: `1/3`
  - wins `attention_score`
  - misses `softmax_wide` and `kv_cache_gather_hard`
- `artifact_feedback_distill_v6`: `1/3`
  - wins `kv_cache_gather_hard`
  - misses `attention_score` and `softmax_wide`
- `base`: `0/3`
- `teacher_corrected_v2`: `0/3`

## Read

The freeze-v4 refresh did not raise the headline success rate on the tighter frontier surface.

It did, however, convert the learner’s single success from `attention_score` to the new harder `kv_cache_gather_hard` task, which is the specific new frontier capability added in this refresh.

## Next Focus

Preserve the new harder-KV learner success while seeking a follow-on turn or eval surface that does not give back `attention_score`.
