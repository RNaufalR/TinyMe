# DATA QUALITY REPORT — dataset_v5

- Raw records ingested: **22292**
- After license check: **22202**
- After content-aware preprocessing: **22200** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **20910** (dropped: {'minified_or_generated': 1290})
- After deduplication: **19441** (exact=424, normalized=156, minhash=888, code=1)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **19398**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 14787 | 791339 | 6143 |
| validation | 2389 | 155202 | 1283 |
| test | 2265 | 102479 | 830 |
| challenge | 90 | 3211 | 46 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (14787 texts).
- Vocabulary: 4096; file 266659 bytes;
  sha256 `2ad26a6d685d58395ef4c983577cf61d…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2422 |
| python_stdlib | 192 |
| syn/algorithm_trace | 120 |
| syn/arithmetic | 259 |
| syn/boolean | 120 |
| syn/code_explain | 120 |
| syn/code_gen | 200 |
| syn/code_repair | 140 |
| syn/deduction | 79 |
| syn/instruction | 185 |
| syn/sequences | 105 |
| syn/tool_use | 327 |
| syn/word_problems | 180 |
| synv3/algorithm | 1600 |
| synv3/copy_span | 3059 |
| synv3/no_tool | 5572 |
| synv3/tool_use | 7804 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 17462 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 1720 |
| code_explain | 566 |
| code_gen | 223 |
| code_repair | 148 |
| instruction | 8487 |
| language | 188 |
| logic | 207 |
| math | 571 |
| programming | 1399 |
| tool_use | 6022 |
