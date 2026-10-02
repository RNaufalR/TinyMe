# DATA QUALITY REPORT — dataset_v3

- Raw records ingested: **11295**
- After license check: **11235**
- After content-aware preprocessing: **11233** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **11233** (dropped: {})
- After deduplication: **10200** (exact=311, normalized=5, minhash=598, code=119)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **10127**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 7220 | 718077 | 5110 |
| validation | 1422 | 128146 | 918 |
| test | 1558 | 105419 | 896 |
| challenge | 60 | 2093 | 25 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (7220 texts).
- Vocabulary: 4096; file 265717 bytes;
  sha256 `d2d47152426e0a9b307eee230e833b77…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2422 |
| python_stdlib | 192 |
| syn/algorithm_trace | 600 |
| syn/arithmetic | 1239 |
| syn/boolean | 600 |
| syn/code_explain | 599 |
| syn/code_gen | 1000 |
| syn/code_repair | 700 |
| syn/deduction | 359 |
| syn/instruction | 840 |
| syn/sequences | 428 |
| syn/tool_use | 1311 |
| syn/word_problems | 900 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 8191 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 600 |
| code_explain | 913 |
| code_gen | 1023 |
| code_repair | 708 |
| instruction | 845 |
| language | 188 |
| logic | 967 |
| math | 2594 |
| programming | 1399 |
| tool_use | 1023 |
