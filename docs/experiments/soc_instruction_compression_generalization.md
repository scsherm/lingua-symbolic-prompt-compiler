# SOC Instruction Compression Generalization

Checkpoint date: 2026-07-10

## Research Question

This experiment tested whether the compiler can reduce the token count of a prompt's instruction portion while preserving structured extraction performance on unseen inputs.

The measured reduction applies only to the reusable instruction template. It does not include the runtime input inserted at `{{input}}`, and it does not claim to reduce generated completion tokens. For a repeated task, the reduced instruction cost is paid on every model call while the input remains unchanged.

## System Adaptation

The compiler was extended in two places to broaden candidate exploration without changing the target model:

1. The obligation chunker exposes smaller semantic instruction units while protecting the input placeholder and JSON schema block from unsafe rewriting.
2. The LLM proposer receives semantically perturbed versions of its rewrite request and produces pools of mixed-representation rewrites. A rewrite may combine concise English, abbreviations, symbols, compact DSL-like notation, and non-English fragments when the combination reduces target-model tokens. Candidate assembly deduplicates rewrites and favors combinations with different chunk choices.

The proposal path uses real LLM calls. There is no rule-based or synthetic proposer in this experiment.

## Data

The suite was derived from the public `witfoo/precinct6-cybersecurity` dataset. It contains three normalized SOC extraction tasks with 100 rows each:

- `incident_metadata`: extract a complete structured incident record from an investigation report.
- `raw_signal`: extract normalized security-event fields while treating conflicting raw-message text as secondary evidence.
- `scoped_investigation`: extract fields from a current incident while ignoring a related historical incident included as a distractor.

Each task was expressed using three independently written instruction styles:

- prose
- contract
- runbook

This produced nine optimization problems. The normalized datasets are under `data/hf/soc_generalization/`; the corresponding prompt templates are under `examples/soc_generalization/`.

## Experimental Design

For each of the nine problems:

- The first 20 rows were used for candidate selection.
- The proposer generated a population of eight candidates using GPT-5.4 Mini.
- GPT-5 Nano was fixed as the target model for both original and compressed prompts.
- The selected candidate was frozen before final evaluation.
- The remaining 80 rows were evaluated twice with both the original and compressed prompt.

The frozen comparison therefore contains 160 original/compressed pairs per problem and 2,880 target-model completions across the full suite.

Structured extraction was evaluated against the expected JSON values using field-level true positives, false positives, false negatives, precision, recall, F1, exact match, JSON validity, and schema validity. Embedding distance and an LLM semantic judge were recorded as diagnostics; they were not used to select extraction candidates.

## Results

| task | instruction style | instruction reduction | original F1 | compressed F1 | F1 delta |
|---|---:|---:|---:|---:|---:|
| incident metadata | prose | 19.38% | 0.9577 | 0.9532 | -0.0045 |
| incident metadata | contract | 18.55% | 0.9321 | 0.9693 | +0.0373 |
| incident metadata | runbook | 20.36% | 0.9741 | 0.9762 | +0.0021 |
| raw signal | prose | 17.01% | 1.0000 | 0.9997 | -0.0003 |
| raw signal | contract | 18.69% | 1.0000 | 1.0000 | 0.0000 |
| raw signal | runbook | 18.01% | 0.9991 | 0.9988 | -0.0003 |
| scoped investigation | prose | 18.95% | 1.0000 | 0.9979 | -0.0021 |
| scoped investigation | contract | 24.88% | 0.9995 | 0.9987 | -0.0008 |
| scoped investigation | runbook | 16.98% | 1.0000 | 1.0000 | 0.0000 |

Aggregate observations:

- Median instruction-token reduction was 18.69%.
- Every selected prompt reduced the instruction portion by 16.98% to 24.88%.
- Mean F1 delta was +0.0035.
- Two problems improved, two tied exactly, and five had negative point estimates.
- Four of the five negative deltas were smaller than 0.0021 in absolute F1.
- The strongest improvement was the incident-metadata contract prompt: +0.0373 F1 with a 95% paired bootstrap interval of [+0.0281, +0.0474].
- Three compressed prompts had lower schema-validity rates than their corresponding originals.

## Interpretation

The suite supports the narrower claim that the compiler can consistently reduce reusable instruction tokens while preserving structured extraction performance in these settings. It does not establish that compression generally improves task quality.

Prompt formulation affected the outcome. The same incident-metadata task ranged from a small regression under the prose prompt to a clear improvement under the contract prompt. This variation is evidence for evaluating optimization across multiple starting prompts rather than reporting a single favorable candidate.

The raw-signal and scoped-investigation baselines were close to or exactly F1 1.0. Those ceiling effects limited their ability to distinguish among otherwise strong candidates. They still provide evidence that substantial instruction reduction can preserve performance, but they are weak tests of quality improvement.

## Limitations

- The suite covers one public SOC dataset and one target-model family.
- Two task families had near-perfect baselines.
- Candidate selection used only 20 rows per problem.
- The experiment measures instruction-template reduction, not total request-token reduction. Actual request savings depend on the instruction-to-input token ratio.
- Schema-validity regressions show that exact output contracts should be enforced as hard acceptance criteria in future runs.

## Next Experiment

The next generalization suite should retain the frozen-selection design while introducing harder evidence boundaries, missing values, conflicting fields, ambiguous labels, and stronger distractors. Candidate acceptance should require no schema-validity regression and should report the compression-quality frontier rather than only the single lowest-loss candidate.
