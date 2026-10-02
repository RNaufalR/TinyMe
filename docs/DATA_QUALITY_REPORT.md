# DATA QUALITY REPORT

- **Dataset version:** `dataset_v1`
- **Generated:** 2026-10-01T13:57:28
- **Train samples:** 514
- **Eval samples:** 25
- **Verified samples:** 369 (71.8% of train)

## 1. Funnel: samples before / after each stage

| Stage | Input | Output | Removed |
| :--- | ---: | ---: | ---: |
| Ingestion + schema validation | 937 | 937 | 0 |
| License check | 937 | 937 | 0 |
| Quality / safety / language filter | 937 | 937 | 0 |
| Deduplication (4 levels) | 937 | 543 | 394 |
| Train/eval split | 543 | 514 + 25 eval | — |

## 2. Duplicate statistics

- Exact hash duplicates: **295**
- Normalized hash duplicates: **0**
- Document-similarity duplicates: **99**
- Code-similarity duplicates: **0**
- Total duplicate percentage: **42.0491%**

## 3. Contamination check (train vs evaluation)

- Evaluation samples: **25**
- Exact overlap: **0**
- Normalized overlap: **0**
- Code-shingle overlap: **0**
- **Contamination free: True**

## 4. Corpus composition

| Category | Samples | Share |
| :--- | ---: | ---: |
| language | 82 | 16.0% |
| programming | 82 | 16.0% |
| math | 72 | 14.0% |
| code_gen | 62 | 12.1% |
| algorithm | 62 | 12.1% |
| logic | 51 | 9.9% |
| code_repair | 41 | 8.0% |
| code_explain | 41 | 8.0% |
| instruction | 21 | 4.1% |

- **Code share:** 226 samples (44.0%)
- **Text share:** 103 samples (20.0%)
- **Reasoning share:** 185 samples (36.0%)

## 5. Source and license distribution

| Source | Samples |
| :--- | ---: |
| synthetic | 323 |
| python_stdlib | 89 |
| huggingface | 79 |
| github | 23 |

| License | Samples |
| :--- | ---: |
| Synthetic-Verified | 323 |
| PSF-2.0 | 89 |
| CC-BY-4.0 | 35 |
| MIT | 34 |
| CDLA-Sharing-1.0 | 33 |

## 6. Tokenization

- Tokenizer: `tok-v1`
- Total tokens: **55,508**
- Average tokens per sample: **107.99221789883268**

## 7. Provenance

Full provenance is recorded in `docs/DATA_PROVENANCE.json` and `DATA_PROVENANCE.json`.
Sources with unclear or copyleft licenses are excluded and explicitly marked.
