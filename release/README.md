# TinyMe - release

TinyMe is a **small language model paired with an external tool runtime**
(a typed tool protocol, a retrieval index and an isolated code-execution
sandbox). It is not a frontier model and does not claim to be one: it is a
reproducible study of how far a few million parameters get on verified
data with a strict protocol.

- Architecture: `nano` - 2,557,632 parameters (9.76 MB fp32)
- Trained on `dataset_v3` (fingerprint `e6a905f467d8d0a91ebeb75d4c5586548bcec3cb0af03f6fbb75f6eec6a6c49a`), stage `sft`, step 600
- Tokenizer: `tok-v2`, vocab 4096, sha256 `d2d47152426e0a9b...`

## Files

| File | Purpose | Bytes |
| :--- | :--- | ---: |
| `model_fp32.safetensors` | fp32 weights | 10,233,776 |
| `model_fp16.safetensors` | fp16 weights | 5,118,480 |
| `model_int8.safetensors` | int8 weights | 2,600,976 |
| `model_int4.safetensors` | int4 weights | 1,451,272 |
| `tokenizer.json` | byte-level BPE used for training | 265,717 |
| `config.json` | model config + token special ids | - |
| `manifest.json` | hashes, fingerprints, measured sizes | - |
| `checksums.txt` | sha256 of every file in this directory | - |
| `inference.py` | standalone entry point (numpy + safetensors + tokenizers) | - |
| `evaluation_report.md` | held-out measurements (or NOT MEASURED) | - |
| `model_comparison.md` | fp32/fp16/int8/int4 + protocol comparability | - |
| `provenance.md` | data sources, licences, contamination verdict | - |

## Run it without the training environment

    pip install numpy safetensors tokenizers
    python inference.py --prompt "What is 144 / 12?" --model model_fp32.safetensors

Only `numpy`, `safetensors` and `tokenizers` are required; JAX and the
repository are not imported by `inference.py`.

## Honest limitations

- Answers come from a ~2.6M-parameter model; factual questions require the
  tool runtime (search/retrieval) and citations are only produced when a
  tool result is present.
- Quantized variants (int8/int4) are lossy; see `model_comparison.md` for
  measured round-trip error and per-variant scores (marked NOT MEASURED
  where they have not been run).
- Any comparison against experiments with a different tokenizer or dataset
  fingerprint is marked NOT COMPARABLE on purpose.
