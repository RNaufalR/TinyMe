# PROGRESS REPORT — STAGE 1

**Project:** TinnyMe — a reproducible tiny AI model strictly under 50 MB with strong computational reasoning.  
**Authoritative Specification:** `TINY_AI_TRAINING_LAB.md`  
**Branch:** `arena/01a0f722-tinnyme` (repo `RNaufalR/TinnyMe`)  
**Stage:** Stage 1 — Feasibility Audit, Project Skeleton, Architecture Decision, License Policy, Initial Data Acquisition  
**Date:** 2026-10-01  
**Overall Status:** `EXECUTED` for Stage 1 deliverables (all Stage-1 items verified with live command output; no fabricated values).

---

## 1. What Was Actually Done in Stage 1

| # | Action | Result | Evidence |
| :-- | :--- | :--- | :--- |
| 1 | Located and merged the authoritative spec into the working branch | `TINY_AI_TRAINING_LAB.md` (26,001 bytes, 1,427 lines) present at repo root | `git log`, file listing |
| 2 | Ran a live feasibility audit of the sandbox (CPU/RAM/disk/Python/GPU/network/toolchain) | Complete `docs/ENVIRONMENT_REPORT.md` | `lscpu`, `free -h`, `df -h`, `python3 -m pip index versions`, `gh api rate_limit`, `curl` probes |
| 3 | Installed the ML stack and verified it executes on CPU | JAX 0.10.2 + jaxlib 0.10.2 (CPU XLA), optax 0.2.8, numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, tokenizers 0.23.2, sentencepiece 0.2.2, safetensors 0.8.0, pytest 9.1.1 | JIT matmul test returned shape `(512, 512)` |
| 4 | Created the full project skeleton per spec §30 | `src/{data,tokenizer,model,training,evaluation,inference,utils}`, `scripts/`, `data_sources/`, `datasets/{raw,processed,versions}`, `experiments/`, `checkpoints/`, `release/`, `tests/`, `docs/`, `configs/` | directory listing |
| 5 | Created the mandated tracking files | `EXECUTION_PLAN.md`, `STATE.md`, `EXPERIMENT_LOG.md`, `DECISIONS.md`, `TODO.md` | committed files |
| 6 | Evaluated all 9 required architecture/training approaches and 3 candidate model geometries | `docs/ARCHITECTURE_DECISION.md` | table of parameters, dims, raw/FP16/INT8/INT4 sizes |
| 7 | Wrote the legal/license policy and provenance schema | `docs/LICENSE_POLICY.md` | allowed/license-unclear/excluded rules, mandatory provenance fields, secret/PII/malware filters |
| 8 | Acquired real, license-verified external data slices | 45 real rows from 3 Hugging Face datasets | files below |

---

## 2. Stage 1 Evidence — Datasets Actually Acquired

All three files live in `datasets/raw/huggingface/` and were retrieved through the Hugging Face **datasets-server rows API** (`datasets-server.huggingface.co`), because the sandbox TLS egress filter blocks direct sockets to `huggingface.co` while the agent's page fetcher can reach the public rows endpoint. Each file records dataset name, URL, retrieval date, license, license URL, and per-row provenance.

| File | Dataset | License | Rows | Bytes | Purpose / Spec Category |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `datasets/raw/huggingface/tinystories_hf_slice.json` | `roneneldan/TinyStories` | `CDLA-Sharing-1.0` | 15 | 11,645 | A. Fundamental language (clean prose) |
| `datasets/raw/huggingface/gsm8k_hf_slice.json` | `openai/gsm8k` | `MIT` | 15 | 8,414 | C. Mathematics (grade-school word problems with step-by-step solutions and verifiable final answers) |
| `datasets/raw/huggingface/mbpp_hf_slice.json` | `google-research-datasets/mbpp` | `CC-BY-4.0` | 15 | 5,787 | G. Code generation + H. Code explanation (natural-language spec → Python implementation + unit tests) |

Total acquired in Stage 1: **45 real samples, 25,846 bytes.**

### External repositories inspected for licensing (candidates for Stage 2 ingestion)

| Repository | SPDX License | Verdict |
| :--- | :--- | :--- |
| `TheAlgorithms/Python` (18,230 KB) | `MIT` | **INCLUDED** in Stage 2 (Python algorithms, data structures) |
| `TheAlgorithms/Go` (2,619 KB) | `MIT` | **INCLUDED** in Stage 2 |
| `TheAlgorithms/Rust` (1,647 KB) | `MIT` | **INCLUDED** in Stage 2 |
| `keon/algorithms` (2,043 KB) | `MIT` | **INCLUDED** in Stage 2 |
| `openai/human-eval` (48 KB) | `MIT` | **INCLUDED** in Stage 2 (held out for code-generation evaluation where possible) |
| `openai/grade-school-math` | `LICENSE_UNCLEAR` per API metadata, **but** the repo ships an explicit `LICENSE` file containing the full MIT text | **INCLUDED** after manual license-file verification |
| `TheAlgorithms/JavaScript` | `GPL-3.0` | **EXCLUDED** (copyleft; would contaminate the MIT-licensed release) |
| `google-research/google-research` | `Apache-2.0` | **INCLUDED (partial)** — MBPP subset only, to avoid a 1.2 GB download |

---

## 3. Environment Reality That Shapes the Training Strategy

| Constraint | Measured Value | Consequence for the design |
| :--- | :--- | :--- |
| CPU | 2 vCPUs, Intel Xeon Ice Lake @ 2.60 GHz, **AVX-512 + VNNI** | XLA-compiled CPU training is the fastest legal path; `torch`'s CUDA wheels (>2.5 GB) are unusable |
| RAM | 3.8 GiB total, **no swap** | Model + activations + optimizer must stay under ~1.5 GiB → small batch + gradient accumulation |
| GPU | **None** (`nvidia-smi` not found) | GPU path is documented but the CPU path is the executed path |
| L3 cache | **54 MiB** | A 34 MiB FP32 or 17 MiB FP16 model fits entirely in cache → fast inference |
| Network | `pypi.org` and `api.github.com` **OPEN**; `huggingface.co` **BLOCKED** for direct sockets | Dual ingestion path: `gh`/git for GitHub, datasets-server rows API for HF |
| Disk | 21 GiB total, ~19 GiB free | Sufficient for corpus, shards, and checkpoints |

---

## 4. Stage 1 Decisions (summary of `DECISIONS.md`)

1. **DEC-001 — Backend:** JAX 0.10.2 + XLA CPU + optax for training; **`.safetensors` + pure NumPy** for a dependency-light inference/quantization path in `release/`. Verified: JIT works.
2. **DEC-002 — Architecture:** Pre-Norm decoder-only Transformer with RoPE, RMSNorm, SwiGLU, causal MHA, and **tied input/output embeddings**.
3. **DEC-003 — Data:** GitHub (`gh` API) + HF (datasets-server) + local Python 3.11 stdlib (PSF-2.0) + deterministic **execution-verified synthetic curriculum**.

---

## 5. Architecture Selected (Stage 1 output of §3)

| Metric | `TinyMe-Nano` (ablation) | **`TinyMe-Base` (selected)** | `TinyMe-Medium` |
| :--- | :--- | :--- | :--- |
| Parameters | 2,557,632 | **8,933,440** | 17,308,032 |
| `d_model` | 192 | **320** | 384 |
| Layers | 4 | **6** | 8 |
| Heads | 6 | **8** | 8 |
| `d_ff` (SwiGLU) | 512 | **896** | 1024 |
| `vocab_size` | 4096 | **4096** | 8192 |
| Context | 256 | **256** | 256 |
| **FP32 raw size** | 10.23 MB | **35.73 MB** (inside the 35–45 MB target window) | 69.23 MB → **exceeds 50 MB** |
| FP16 | 5.12 MB | **17.87 MB** | 34.62 MB |
| INT8 | 2.58 MB | **8.98 MB** | 17.35 MB |
| INT4 | 1.32 MB | **4.57 MB** | 8.81 MB |

`TinyMe-Base` is selected because it is **<50 MB even uncompressed**, and FP16/INT8/INT4 variants leave a large safety margin. 85.3% of its parameters (7.62M of 8.93M) are dedicated to the 6 reasoning layers, because the tied 4096-token embedding costs only 1.31M parameters.

---

## 6. Requirement Status After Stage 1

| Requirement | Section | Status |
| :--- | :--- | :--- |
| REQ-01 Feasibility audit + `ENVIRONMENT_REPORT.md` | §2 | **VERIFIED** |
| REQ-02 Project structure + `EXECUTION_PLAN.md`/`STATE.md`/`DECISIONS.md`/`EXPERIMENT_LOG.md`/`TODO.md` | §30, §35, §36 | **VERIFIED** |
| REQ-03 Architecture evaluation + `ARCHITECTURE_DECISION.md` | §3 | **VERIFIED** |
| REQ-04 `LICENSE_POLICY.md` + provenance schema | §5 | **VERIFIED** |
| REQ-05 Modular data engine + 14-stage pipeline | §4, §6, §7, §8 | PENDING |
| REQ-06 Dataset categories A–I + synthetic curriculum | §9, §11, §12 | PENDING |
| REQ-07 Quality scoring, dedup, contamination, `DATA_QUALITY_REPORT.md` | §13, §14 | PENDING |
| REQ-08 Tokenizer training + `TOKENIZER_REPORT.md` | §15 | PENDING |
| REQ-09 Training engine + checkpoints/resume | §16, §17, §18 | PENDING |
| REQ-10 9-domain evaluation engine | §19, §20 | PENDING |
| REQ-11 Inference interface | §25 | PENDING |
| REQ-12 Quantization + `release/` <50 MB | §23, §24 | PENDING |
| REQ-13 Automated test suite | §31 | PENDING |

---

## 7. Known Blockers (documented, not hidden)

1. **Direct HTTPS to `huggingface.co` from the sandbox is blocked** (`curl: (35) OpenSSL SSL_connect: SSL_ERROR_SYSCALL`).  
   *Closest valid fallback implemented:* the HF **datasets-server rows API** is used to fetch and cache verified dataset rows locally, and GitHub mirrors are used for code corpora. Full multi-GB HF downloads are not possible in this sandbox — this is recorded, not glossed over.
2. **`download.pytorch.org/whl/cpu` is blocked**, so a CPU-only PyPI `torch` build is unavailable.  
   *Closest valid fallback implemented:* JAX/jaxlib CPU wheels install and run (verified), with a pure-NumPy inference path for the release artifact.
3. **No GPU.** The `configs/gpu.yaml` path is documented but untested in this sandbox.

---

## 8. Next Stage (Stage 2) Plan

1. Implement `data_sources/` adapters: `github_adapter.py`, `huggingface_adapter.py`, `stdlib_adapter.py`, `synthetic_adapter.py`.
2. Implement the 14-stage pipeline in `src/data/` (ingestion → license check → extraction → normalization → quality → safety → language → classification → dedup → contamination → mixing → tokenization → shards → manifest).
3. Implement the deterministic, execution-verified synthetic generators for categories A–I.
4. Produce `docs/DATA_QUALITY_REPORT.md`, `docs/DATA_PROVENANCE.json`, and dataset version `dataset_v1`.

---

*No training, evaluation, benchmark, or model-size claim is made in this report. Nothing above is simulated: every number comes from a command executed in this sandbox on 2026-10-01.*
