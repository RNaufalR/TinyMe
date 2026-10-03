# DATA PROVENANCE (release)

- Dataset version: `dataset_v2`
- Dataset fingerprint: `730f1dbdf104e23105cc4b62434a81de8bdb820e8bf03906f8f3c05ac360189f`
- Tokenizer hash: `dd71a73463c5405689c0da4af5746d98c57a41402cc750ab1464ecca2e935a7d`

## Sources (records emitted per source)

| Source | Records |
| :--- | ---: |
| python3.11-stdlib-extended | 2314 |
| syn/tool_use | 722 |
| syn/arithmetic | 635 |
| syn/code_gen | 500 |
| syn/word_problems | 450 |
| syn/instruction | 436 |
| syn/code_repair | 350 |
| syn/boolean | 300 |
| syn/algorithm_trace | 300 |
| syn/code_explain | 300 |
| syn/sequences | 253 |
| python_stdlib | 198 |
| syn/deduction | 192 |
| gsm8k | 15 |
| mbpp | 15 |
| tinystories | 15 |

## Licence distribution

| Licence | Records |
| :--- | ---: |
| Synthetic-Verified | 4283 |
| PSF-2.0 | 1939 |
| CDLA-Sharing-1.0 | 15 |
| CC-BY-4.0 | 15 |
| MIT | 15 |

- Contamination (train vs validation/test): **PASS**
- Malformed code rate: 0.0
- Verified samples: 6137

Full machine-readable provenance: `docs/DATA_PROVENANCE.json` and `DATA_PROVENANCE.json` in the repository.

- Packaged at: 2026-10-03T21:32:18
- Packaging environment: {"cpu_count": 2, "git_commit": "a1663ba474d61c82d9672a241cb584dc7e4b5344", "git_dirty": true, "jax": "0.11.2", "jax_backend": "cpu", "jax_devices": ["cpu:0"], "machine": "x86_64", "numpy": "2.5.3", "optax": "0.2.8", "platform": "Linux-6.18.44-fc-v64-x86_64-with-glibc2.39", "python": "3.13.16", "safetensors": "0.8.0", "tokenizers": "0.23.2"}
