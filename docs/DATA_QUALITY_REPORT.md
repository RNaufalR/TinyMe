# DATA QUALITY REPORT — ci_unit

- Raw records ingested: **2817**
- After license check: **2757**
- After content-aware preprocessing: **2755** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **2755** (dropped: {})
- After deduplication: **2167** (exact=311, normalized=2, minhash=275, code=0)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **2094**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 1680 | 601383 | 9862 |
| validation | 253 | 103989 | 1695 |
| test | 234 | 76589 | 1255 |
| challenge | 60 | 0 | 0 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (1680 texts).
- Vocabulary: 4096; file 267051 bytes;
  sha256 `644e9e1fa8f43f6a76e56a845cc38ab4…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2422 |
| python_stdlib | 192 |
| syn/algorithm_trace | 6 |
| syn/arithmetic | 13 |
| syn/boolean | 6 |
| syn/code_explain | 6 |
| syn/code_gen | 10 |
| syn/code_repair | 7 |
| syn/deduction | 4 |
| syn/instruction | 10 |
| syn/sequences | 6 |
| syn/tool_use | 21 |
| syn/word_problems | 9 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 158 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 6 |
| code_explain | 466 |
| code_gen | 33 |
| code_repair | 15 |
| instruction | 18 |
| language | 188 |
| logic | 18 |
| math | 55 |
| programming | 1399 |
| tool_use | 29 |
