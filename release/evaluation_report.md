# EVALUATION REPORT - EXP-004-TOOL-SFT-V3

Generated: 2026-10-02T12:42:02

## EXP-004-TOOL-SFT-V3:fp32 - split `challenge` (dataset_v3)

- Source: `evaluation_challenge.json`
- Samples scored: **176**
- Mean accuracy: 0.1
- Duration: 4.7 s

| Metric | Value |
| :--- | ---: |
| duration_s | 4.7 |

| Domain | Samples | Score |
| :--- | ---: | ---: |
| code_generation | 8 | 0.0% |
| code_repair | 8 | 0.0% |
| generalization | 84 | 10.0% |
| grounding | 8 | 0.0% |
| instruction_following | 8 | 0.0% |
| language | 8 | ppl 779.7545 |
| logic | 8 | 0.0% |
| math | 12 | 0.0% |
| tool_arguments | 8 | 0.0% |
| tool_execution | 8 | 0.0% |
| tool_selection | 8 | 0.0% |
| tool_syntax | 8 | 100.0% |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:fp16 - split `challenge` (dataset_v3)

- Source: `evaluation_challenge.json`
- Samples scored: **176**
- Mean accuracy: 0.1
- Duration: 11.48 s

| Metric | Value |
| :--- | ---: |
| duration_s | 11.48 |

| Domain | Samples | Score |
| :--- | ---: | ---: |
| code_generation | 8 | 0.0% |
| code_repair | 8 | 0.0% |
| generalization | 84 | 10.0% |
| grounding | 8 | 0.0% |
| instruction_following | 8 | 0.0% |
| language | 8 | ppl 779.5647 |
| logic | 8 | 0.0% |
| math | 12 | 0.0% |
| tool_arguments | 8 | 0.0% |
| tool_execution | 8 | 0.0% |
| tool_selection | 8 | 0.0% |
| tool_syntax | 8 | 100.0% |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:int8 - split `challenge` (dataset_v3)

- Source: `evaluation_challenge.json`
- Samples scored: **176**
- Mean accuracy: 0.1
- Duration: 4.68 s

| Metric | Value |
| :--- | ---: |
| duration_s | 4.68 |

| Domain | Samples | Score |
| :--- | ---: | ---: |
| code_generation | 8 | 0.0% |
| code_repair | 8 | 0.0% |
| generalization | 84 | 10.0% |
| grounding | 8 | 0.0% |
| instruction_following | 8 | 0.0% |
| language | 8 | ppl 786.8711 |
| logic | 8 | 0.0% |
| math | 12 | 0.0% |
| tool_arguments | 8 | 0.0% |
| tool_execution | 8 | 0.0% |
| tool_selection | 8 | 0.0% |
| tool_syntax | 8 | 100.0% |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:int4 - split `challenge` (dataset_v3)

- Source: `evaluation_challenge.json`
- Samples scored: **176**
- Mean accuracy: 0.0
- Duration: 3.04 s

| Metric | Value |
| :--- | ---: |
| duration_s | 3.04 |

| Domain | Samples | Score |
| :--- | ---: | ---: |
| code_generation | 8 | 0.0% |
| code_repair | 8 | 0.0% |
| generalization | 84 | 0.0% |
| grounding | 8 | 0.0% |
| instruction_following | 8 | 0.0% |
| language | 8 | ppl 648.4466 |
| logic | 8 | 0.0% |
| math | 12 | 0.0% |
| tool_arguments | 8 | 0.0% |
| tool_execution | 8 | 0.0% |
| tool_selection | 8 | 0.0% |
| tool_syntax | 8 | 0.0% |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:fp32 - split `test` (dataset_v3)

- Source: `evaluation_test.json`
- Samples scored: **2158**
- Mean accuracy: 0.186
- Duration: 953.23 s

| Metric | Value |
| :--- | ---: |
| duration_s | 953.23 |

| Domain | Samples | Score |
| :--- | ---: | ---: |
| code | 140 | 0.0% |
| code_explanation | 92 | ppl 113.9042 |
| code_generation | 200 | 0.0% |
| code_repair | 191 | 9.4% |
| grounding | 150 | 0.0% |
| instruction_following | 142 | 0.0% |
| language | 18 | ppl 812.4003 |
| logic | 218 | 73.9% |
| math | 407 | 0.0% |
| tool_arguments | 150 | 0.0% |
| tool_execution | 150 | 20.7% |
| tool_selection | 150 | 2.0% |
| tool_syntax | 150 | 98.7% |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:fp32 - split `None` (None)

- Source: `evaluation_tools_best.json`
- Samples scored: **25**
- Mean accuracy: None
- Duration: None s

| Metric | Value |
| :--- | ---: |
| argument_accuracy | 0.6 |
| argument_validity | 1.0 |
| citation_validity | 0.0 |
| error_recovery_success | 0.0 |
| execution_success | 0.5 |
| grounded_final_answer | 0.3 |
| multi_step_success | 0.0 |
| needed_cases | 20 |
| not_needed_cases | 5 |
| task_completion | 0.0 |
| tool_name_accuracy | 0.3 |
| tool_needed_accuracy | 0.3 |
| tool_not_needed_accuracy | 0.4 |
| tool_syntax_validity | 0.84 |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:fp16 - split `None` (None)

- Source: `evaluation_tools_best.json`
- Samples scored: **25**
- Mean accuracy: None
- Duration: None s

| Metric | Value |
| :--- | ---: |
| argument_accuracy | 0.6 |
| argument_validity | 1.0 |
| citation_validity | 0.0 |
| error_recovery_success | 0.0 |
| execution_success | 0.5 |
| grounded_final_answer | 0.3 |
| multi_step_success | 0.0 |
| needed_cases | 20 |
| not_needed_cases | 5 |
| task_completion | 0.0 |
| tool_name_accuracy | 0.3 |
| tool_needed_accuracy | 0.3 |
| tool_not_needed_accuracy | 0.4 |
| tool_syntax_validity | 0.84 |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:int8 - split `None` (None)

- Source: `evaluation_tools_best.json`
- Samples scored: **25**
- Mean accuracy: None
- Duration: None s

| Metric | Value |
| :--- | ---: |
| argument_accuracy | 0.6667 |
| argument_validity | 1.0 |
| citation_validity | 0.0 |
| error_recovery_success | 0.0 |
| execution_success | 0.45 |
| grounded_final_answer | 0.25 |
| multi_step_success | 0.0 |
| needed_cases | 20 |
| not_needed_cases | 5 |
| task_completion | 0.0 |
| tool_name_accuracy | 0.3 |
| tool_needed_accuracy | 0.3 |
| tool_not_needed_accuracy | 0.4 |
| tool_syntax_validity | 0.8 |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.

## EXP-004-TOOL-SFT-V3:int4 - split `None` (None)

- Source: `evaluation_tools_best.json`
- Samples scored: **25**
- Mean accuracy: None
- Duration: None s

| Metric | Value |
| :--- | ---: |
| argument_accuracy | 0.0 |
| argument_validity | 1.0 |
| citation_validity | 0.0 |
| error_recovery_success | 0.0 |
| execution_success | 0.0 |
| grounded_final_answer | 0.2 |
| multi_step_success | 0.0 |
| needed_cases | 20 |
| not_needed_cases | 5 |
| task_completion | 0.0 |
| tool_name_accuracy | 0.0 |
| tool_needed_accuracy | 0.0 |
| tool_not_needed_accuracy | 1.0 |
| tool_syntax_validity | 0.4 |

Measurements are produced by `scripts/evaluate.py` / `scripts/evaluate_tools.py` against held-out splits only; no subset of `train.jsonl` is ever used.
