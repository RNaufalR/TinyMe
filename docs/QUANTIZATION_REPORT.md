# TinyMe — Quantisation report (audit §10, §11)

Every row below is a measurement taken in this repository on the packaged
release artifacts. Commands are reproduced verbatim at the bottom.

## 1. Artifact sizes (from `release/manifest.json`, serialized files on disk)

| Variant | Bytes | MiB | MB | Relative RMSE | Max abs error |
| :--- | ---: | ---: | ---: | ---: | ---: |
| fp32 | 10,233,776 | 9.76 | 10.23 | 0 (reference) | 0 |
| fp16 | 5,118,480 | 4.88 | 5.12 | 2.12e-04 | 4.88e-04 |
| int8 | 2,600,976 | 2.48 | 2.60 | 6.44e-03 | 1.42e-03 |
| int4 | 1,451,272 | 1.38 | 1.45 | 9.67e-02 | 2.54e-02 |

* Raw fp32 tensor payload: 10,230,528 B; the 3,248 B difference is the
  safetensors header + metadata. Both numbers are published explicitly
  (`fp32_bytes` = serialized file, `fp32_tensor_bytes` = payload) because
  conflating them is exactly the kind of accounting error §11 asks about.
* **Total release directory: 13 files, 19,688,665 B (18.78 MiB)** — includes all
  four variants, tokenizer, config, manifest, checksums and docs.
* Limit check: `< 50 MB` holds for **all four** variants (`under_50mb` in the
  manifest). Even the sum of all variants stays at 19.2 MB.
* MB vs MiB is stated in both units everywhere; `human_bytes()` is the single
  formatter used by the packager and the tests.

## 2. Load and parameter equality

| Variant | Loads in clean env | Parameter count | Tokenizer compatible |
| :--- | :--- | ---: | :--- |
| fp32 | yes | 2,557,632 | yes (same `tokenizer.json`, hash in manifest) |
| fp16 | yes | 2,557,632 | yes |
| int8 | yes | 2,557,632 | yes |
| int4 | yes | 2,557,632 | yes |

All four files are loaded through the *released* `inference.py`
(`--model model_<variant>.safetensors`), i.e. without the training repo.

## 3. Functional regression by variant

The honest summary is that this model has **no working capability to regress
from**: the fp32 reference already fails the arithmetic/reasoning suites. What
the measurement *does* show is that quantisation is not the dominant error
source at fp16/int8, and int4 does break the little structure that exists.

### Tool-use suite (25 independent cases, `experiments/EXP-004-TOOL-SFT-V3/`)

| Metric | fp32 | fp16 | int8 | int4 |
| :--- | ---: | ---: | ---: | ---: |
| tool_syntax_validity | 0.84 | (see JSON) | 0.80 | 0.40 |
| tool_name_accuracy | 0.30 | | 0.30 | 0.00 |
| argument_validity | 1.00 | | 1.00 | 1.00 |
| execution_success | 0.50 | | 0.45 | 0.00 |
| grounded_final_answer | 0.30 | | 0.25 | 0.20 |
| task_completion | 0.00 | | 0.00 | 0.00 |

### Per-domain (challenge split, 60 records / 176 scored samples)

| Variant | Mean accuracy | math | logic | code_repair | instruction | generalization |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| fp32 | 0.1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| fp16 | 0.1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| int8 | 0.1000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.10 |
| int4 | 0.0000 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |

Language perplexity (challenge): fp32 779.75 — the structured-SFT model is a
poor plain-language model, which is stated rather than hidden.

## 4. Latency / throughput / memory (measured, NumPy backend)

| Measurement | Value |
| :--- | :--- |
| Checkpoint-served generation, challenge split | 60 calls / 3,419 tokens / 3.73 s |
| Throughput | 916.6 tokens/s |
| Latency mean | 0.062 s |
| Latency p50 / p95 | 0.062 s / 0.123 s |
| NumPy vs JAX engine (32 greedy tokens) | 409.9 tok/s vs 1.9 tok/s cold |

JAX numbers are cold-start dominated (per-call compilation) and are labelled as
such; the NumPy path is the one used for all published latency figures.

## 5. Robustness

| Check | Result |
| :--- | :--- |
| Repeat inference determinism | identical token ids for identical prompt (greedy) |
| EOS / stop handling | generation stops on `<|endoftext|>` / stop-set markers |
| Long context | see §24 matrix in `docs/audit_evidence/` |
| Empty / short prompt | no crash, documented behaviour |

## 6. Selection

Per §10 the variant is selected by **measured functional regression**, not by
size alone. Since fp32/fp16/int8 are indistinguishable on the challenge split
and int4 measurably degrades the tool suite (syntax 0.84 → 0.40), the published
release designates **fp32 as the reference** and advertises int4 as
"smallest, measurably degraded" in `README.md` and `model_comparison.md`.
No variant is claimed to be capable beyond what the tables above show.

## 7. Reproduction

```bash
python scripts/quantize.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best
python scripts/evaluate_tools.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --variants fp32,fp16,int8,int4 --max-new-tokens 96 --backend numpy
python scripts/evaluate.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --dataset dataset_v3 --split challenge --backend numpy \
    --variants fp32,fp16,int8,int4 --max-new-tokens 96
python scripts/package_model.py --experiment EXP-004-TOOL-SFT-V3 --checkpoint best \
    --dataset dataset_v3
```
