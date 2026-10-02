# EXPERIMENT LOG (`EXPERIMENT_LOG.md`)

Mirrors `EXPERIMENT_LOG.jsonl` (machine-readable, 8-point records for the
iterative interventions live in `docs/TRAINING_REPORT.md` §2). Every row cites
the artefact that holds its numbers; nothing is transcribed by hand.

| Exp ID | Type | Architecture | Dataset | Steps | Status | Val loss / ppl | Evidence |
| :--- | :--- | :--- | :--- | --: | :--- | :--- | :--- |
| `EXP-001` | pretrain (historical) | nano | — | — | HISTORICAL / INVALID-AS-HELDOUT-BASELINE | — | `EXPERIMENT_LOG.jsonl` |
| `EXP-002-PREP-VALIDATION` | pretrain (invalidated) | nano | dataset_v2 (dup-inflated) | 120 | STOPPED / SUPERSEDED | 5.1994 / 181.17 | `EXPERIMENT_LOG.jsonl` |
| `EXP-002-PRE-STRATIFIED` | pretrain (invalidated) | nano | dataset_v2 (random split) | 220 | STOPPED / SUPERSEDED | 5.0309 / 153.07 | `EXPERIMENT_LOG.jsonl` |
| `EXP-002-PRE-PROTOCOL` | pretrain (invalidated) | nano | dataset_v2 (arg drift) | 120 | STOPPED / SUPERSEDED | 5.9839 / 397.0 | `EXPERIMENT_LOG.jsonl` |
| `EXP-002-CORRECTED-NANO` | pretrain (baseline) | nano | dataset_v2 | 400 | EXECUTED / VALID BASELINE | 4.8042 / 122.0208 | `experiments/EXP-002-CORRECTED-NANO/summary.json` |
| `EXP-004-TOOL-SFT` | SFT | nano | dataset_v3 rev2 | 320 | EXECUTED / CAPABILITY FAILED | 1.9757 / 7.2117 | `experiments/EXP-004-TOOL-SFT/summary.json`, `experiments/EXP-004-TOOL-SFT/evaluation_tools_best.json` |
| `EXP-004-TOOL-SFT-V2` | SFT | nano | dataset_v3 rev3 | 700 | EXECUTED / CAPABILITY FAILED | 1.6874 / 5.4055 | `experiments/EXP-004-TOOL-SFT-V2/summary.json`, `experiments/EXP-004-TOOL-SFT-V2/evaluation_tools_best.json` |
| `EXP-004-TOOL-SFT-V3` | SFT | nano | dataset_v3 rev4 | 600 | EXECUTED / CAPABILITY FAILED | 1.5305 / 4.6206 | `experiments/EXP-004-TOOL-SFT-V3/summary.json`, `experiments/EXP-004-TOOL-SFT-V3/evaluation_tools_best.json` |
| `EXP-003-BASE-PILOT` | pretrain (feasibility) | base | dataset_v3 | 25 | EXECUTED / FEASIBILITY ONLY | 6.6931 / 806.8009 | `experiments/EXP-003-BASE-PILOT/summary.json`, `docs/CAPACITY_ANALYSIS.md` |
| `CURRIC-A-STAGED` | curriculum A pilot (pretrain+sft (staged)) | nano | dataset_v3 rev4 | 300 | EXECUTED / NEGATIVE RESULT | 4.6215 / 101.6497 | `experiments/CURRIC-A-STAGED/summary.json`, `experiments/CURRIC-A-STAGED/evaluation_test.json`, `docs/MODEL_COMPARISON.md` |
| `CURRIC-B-MIXED` | curriculum B pilot (mixed) | nano | dataset_v3 rev4 | 300 | EXECUTED / NEGATIVE RESULT | 4.4565 / 86.1883 | `experiments/CURRIC-B-MIXED/summary.json`, `experiments/CURRIC-B-MIXED/evaluation_test.json`, `docs/MODEL_COMPARISON.md` |
