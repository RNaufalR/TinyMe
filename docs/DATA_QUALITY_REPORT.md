# DATA QUALITY REPORT — dataset_v8

- Raw records ingested: **28146**
- After license check: **28056**
- After content-aware preprocessing: **28054** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **28050** (dropped: {'unsafe_content': 2, 'minified_or_generated': 2})
- After deduplication: **26742** (exact=327, normalized=15, minhash=966, code=0)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **26699**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 20088 | 1559219 | 12348 |
| validation | 2886 | 197997 | 2267 |
| test | 3768 | 208023 | 2285 |
| challenge | 90 | 5316 | 55 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (20088 texts).
- Vocabulary: 4096; file 266299 bytes;
  sha256 `8b28604653165f68d8b23a8b92d29f54…`

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
| synv3/copy_span | 6000 |
| synv3/no_tool | 7990 |
| synv3/tool_use | 8299 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 2024 |
| Synthetic-Verified | 24763 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 1720 |
| code_explain | 566 |
| code_gen | 223 |
| code_repair | 148 |
| instruction | 14151 |
| language | 188 |
| logic | 207 |
| math | 571 |
| programming | 1399 |
| tool_use | 7659 |
