# Spark GB10 Troubleshooting

This note freezes the known local caveats for the DGX Spark GB10 path.

## Known Caveats

- PyTorch warns that the installed wheel officially covers capabilities through `12.0` while GB10 reports `12.1`.
- `nvdisasm` cannot decode `SM121a`, so disassembly evidence degrades even when execution still works.
- Triton helper builds require the repo-local Python headers exposed by [run_with_spark_env.sh](/home/kmccleary/projects/gpu_code_agents/gpu_rl/scripts/run_with_spark_env.sh).
- unauthenticated Hugging Face downloads work, but they are slower and more fragile than a token-backed path.

## What To Do First

- run commands through [run_with_spark_env.sh](/home/kmccleary/projects/gpu_code_agents/gpu_rl/scripts/run_with_spark_env.sh)
- prefer the `7B` local path unless a new runtime variable justifies reopening `32B`
- treat `nvdisasm fatal : Cannot decode architecture 'SM121a'` as a tooling-evidence issue unless execution also fails

## Signals That Are Usually Non-Blocking

- the PyTorch GB10 capability warning by itself
- the `SM121a` disassembly warning by itself

## Signals That Usually Matter

- OOM during model load
- failure to reach `model_load_done` or `trainer_init_done`
- missing local Python headers during Triton helper compilation
- repeated held-out regressions after a training-data change
