# DATA QUALITY REPORT — dataset_v2

- Raw records ingested: **6568**
- After license check: **6508**
- After content-aware preprocessing: **6506** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **6506** (dropped: {})
- After deduplication: **5778** (exact=311, normalized=2, minhash=392, code=23)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **5705**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 4227 | 588816 | 3585 |
| validation | 741 | 105432 | 641 |
| test | 810 | 82501 | 587 |
| challenge | 60 | 1854 | 23 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (4227 texts).
- Vocabulary: 4096; file 265717 bytes;
  sha256 `d2d47152426e0a9b307eee230e833b77…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2422 |
| python_stdlib | 192 |
| syn/algorithm_trace | 300 |
| syn/arithmetic | 591 |
| syn/boolean | 300 |
| syn/code_explain | 299 |
| syn/code_gen | 500 |
| syn/code_repair | 350 |
| syn/deduction | 188 |
| syn/instruction | 268 |
| syn/sequences | 253 |
| syn/tool_use | 350 |
| syn/word_problems | 450 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 3769 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 300 |
| code_explain | 679 |
| code_gen | 515 |
| code_repair | 363 |
| instruction | 268 |
| language | 180 |
| logic | 502 |
| math | 1328 |
| programming | 1399 |
| tool_use | 304 |
