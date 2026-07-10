# Hugging Face Instruction Data

This directory contains normalized instruction-data samples for prompt compression experiments.

Current sample:

```text
data/hf/no_robots_100.jsonl
data/hf/witfoo_soc_extraction_100.jsonl
data/hf/soc_generalization/incident_metadata.jsonl
data/hf/soc_generalization/raw_signal.jsonl
data/hf/soc_generalization/scoped_investigation.jsonl
```

Source dataset:

```text
HuggingFaceH4/no_robots
witfoo/precinct6-cybersecurity
```

Normalized row fields:

- `id`
- `input`
- `reference_output`
- `dataset`
- `category`
- `source_row`

`input` is the instruction text used as the variable prompt input. `reference_output` is the dataset completion supplied by the source dataset.

Refresh the sample:

```bash
python3 scripts/download_instruction_dataset.py \
  --dataset HuggingFaceH4/no_robots \
  --config default \
  --split train \
  --limit 100 \
  --output data/hf/no_robots_100.jsonl
```

The `HuggingFaceH4/no_robots` source dataset license is `cc-by-nc-4.0`.

## SOC Generalization Suite

The SOC suite contains 100 rows for each of three structured extraction tasks:

- incident metadata extracted from generated investigation reports
- normalized security-event fields extracted from alert packets
- current-incident fields extracted in the presence of a historical distractor

Each row contains `id`, `input`, `expected`, and `task`. The suite manifest maps these datasets to prose, contract, and runbook prompt formulations under `examples/soc_generalization/`.

Rebuild the normalized data and prompts with:

```bash
python3 scripts/build_soc_generalization_suite.py
```

See `docs/experiments/soc_instruction_compression_generalization.md` for the frozen-validation design and results.
