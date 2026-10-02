# DATA PROVENANCE (release)

- Dataset version: `dataset_v3`
- Dataset fingerprint: `e6a905f467d8d0a91ebeb75d4c5586548bcec3cb0af03f6fbb75f6eec6a6c49a`
- Tokenizer hash: `d2d47152426e0a9b307eee230e833b7763f60b50ec7316a080d2906528162230`

## Sources (records emitted per source)

| Source | Records |
| :--- | ---: |
| python3.11-stdlib-extended | 2422 |
| syn/tool_use | 1311 |
| syn/arithmetic | 1239 |
| syn/code_gen | 1000 |
| syn/word_problems | 900 |
| syn/instruction | 840 |
| syn/code_repair | 700 |
| syn/boolean | 600 |
| syn/algorithm_trace | 600 |
| syn/code_explain | 599 |
| syn/sequences | 428 |
| syn/deduction | 359 |
| python_stdlib | 192 |
| gsm8k | 15 |
| mbpp | 15 |
| tinystories | 15 |

## Licence distribution

| Licence | Records |
| :--- | ---: |
| Synthetic-Verified | 8191 |
| PSF-2.0 | 2024 |
| CDLA-Sharing-1.0 | 15 |
| CC-BY-4.0 | 15 |
| MIT | 15 |

- Contamination (train vs validation/test): **PASS**
- Malformed code rate: 0.0
- Verified samples: 10127

Full machine-readable provenance: `docs/DATA_PROVENANCE.json` and `DATA_PROVENANCE.json` in the repository.

- Packaged at: 2026-10-02T12:42:02
- Packaging environment: {"cpu_count": 2, "git_commit": "8446a0303049c49cd9b3a671011c61b4e7fa9757", "git_dirty": true, "jax": "0.10.2", "jax_backend": "cpu", "jax_devices": ["cpu:0"], "machine": "x86_64", "numpy": "2.4.6", "optax": "0.2.8", "platform": "Linux-6.1.158+-x86_64-with-glibc2.36", "python": "3.11.2", "safetensors": "0.8.0", "tokenizers": "0.23.2"}
