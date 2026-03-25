# New Task Family Onboarding Template

Use this template when adding a new optimize-task family.

## 1. Problem Framing

- Family name:
- Intended verb:
- Operator family:
- Difficulty tier:
- Why this family matters:
- Why this family is fair for the current environment:

## 2. Provenance

- Source benchmark or origin:
- Source repo:
- Source commit / release:
- Official or curated:
- Held-out or trainable:
- Related existing family or variant-of:

## 3. Runtime Surface

- Baseline command:
- Reference / candidate command:
- Required libraries:
- Expected backend(s):
- Known hardware assumptions:

## 4. Candidate Engineering

- Is this family single-candidate or multi-candidate by default?
- Expected compare anchor:
- Expected branch depth:
- Best-known semantics needed:
- Endgame actions expected:

## 5. Evaluation

- Visible test path:
- Hidden test path:
- Failure packet shape:
- Perf measurement policy:
- Benchmark provenance fields:

## 6. Learning Use

- Distill-eligible:
- Teacher-correction-eligible:
- Narrow-RL-eligible:
- Governance expectation:
- Common positive trace shape:
- Common near-miss shape:
- Common negative trace shape:

## 7. Validation

- Adapter / registry tests:
- Evaluator tests:
- Harness tests:
- Known caveats:

## 8. Exit Criteria

- What counts as “integrated”:
- What counts as “ready for GPT-5.4 hard-slice collection”:
- What counts as “ready for learned-policy training use”:
