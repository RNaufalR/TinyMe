# DATA QUALITY REPORT — dataset_v2

- Raw records ingested: **6819**
- After license check: **6759**
- After content-aware preprocessing: **6757** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **6757** (dropped: {})
- After deduplication: **3580** (exact=2900, normalized=2, minhash=275, code=0)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **3507**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 3062 | 585187 | 3070 |
| validation | 259 | 77467 | 373 |
| test | 259 | 28244 | 155 |
| challenge | 60 | 1842 | 23 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (3062 texts).
- Vocabulary: 4096; file 266195 bytes;
  sha256 `ef75489fa4946b2f719ca6b0bd49aad4…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2422 |
| python_stdlib | 192 |
| syn/algorithm_trace | 300 |
| syn/arithmetic | 650 |
| syn/boolean | 300 |
| syn/code_explain | 300 |
| syn/code_gen | 500 |
| syn/code_repair | 350 |
| syn/deduction | 200 |
| syn/instruction | 300 |
| syn/sequences | 300 |
| syn/tool_use | 450 |
| syn/word_problems | 450 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 1571 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 300 |
| code_explain | 470 |
| code_gen | 25 |
| code_repair | 19 |
| instruction | 5 |
| language | 180 |
| logic | 43 |
| math | 1093 |
| programming | 1399 |
| tool_use | 106 |
