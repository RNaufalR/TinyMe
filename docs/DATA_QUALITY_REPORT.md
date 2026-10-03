# DATA QUALITY REPORT — dataset_v2

- Raw records ingested: **7055**
- After license check: **6995**
- After content-aware preprocessing: **6993** (rejected: {'rejected:invalid_code': 2})
- After quality/safety filter: **6993** (dropped: {})
- After deduplication: **6207** (exact=311, normalized=2, minhash=426, code=47)
- Malformed-code rate in the final corpus: **0.000000**
- Verified records: **6137**

## Splits (group-aware, template families held out)

| split | records | active target tokens | blocks |
| :--- | ---: | ---: | ---: |
| train | 5139 | 771252 | 5004 |
| validation | 360 | 17890 | 263 |
| test | 708 | 59162 | 588 |
| challenge | 60 | 3392 | 57 |

## Contamination gate

**TRAIN/VALIDATION/TEST CONTAMINATION: PASS**

Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles, template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.

## Tokenizer

- Trained on the **train split only** (5139 texts).
- Vocabulary: 4096; file 267003 bytes;
  sha256 `dd71a73463c5405689c0da4af5746d98…`

## Sources and licences

| source | records |
| :--- | ---: |
| gsm8k | 15 |
| mbpp | 15 |
| python3.11-stdlib-extended | 2314 |
| python_stdlib | 198 |
| syn/algorithm_trace | 300 |
| syn/arithmetic | 635 |
| syn/boolean | 300 |
| syn/code_explain | 300 |
| syn/code_gen | 500 |
| syn/code_repair | 350 |
| syn/deduction | 192 |
| syn/instruction | 436 |
| syn/sequences | 253 |
| syn/tool_use | 722 |
| syn/word_problems | 450 |
| tinystories | 15 |

| licence | records |
| :--- | ---: |
| CC-BY-4.0 | 15 |
| CDLA-Sharing-1.0 | 15 |
| MIT | 15 |
| PSF-2.0 | 1939 |
| Synthetic-Verified | 4283 |

## Category distribution

| category | records |
| :--- | ---: |
| algorithm | 300 |
| code_explain | 663 |
| code_gen | 523 |
| code_repair | 357 |
| instruction | 444 |
| language | 163 |
| logic | 500 |
| math | 1365 |
| programming | 1374 |
| tool_use | 578 |
