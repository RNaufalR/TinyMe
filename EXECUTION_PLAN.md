# EXECUTION PLAN — TINY AI TRAINING LAB

Authoritative Specification: `TINY_AI_TRAINING_LAB.md`  
Target: Reproducible end-to-end tiny AI model (`FINAL_MODEL_SIZE < 50 MB`) with strong foundational computational reasoning, code comprehension/generation/repair, algorithmic reasoning, mathematics, and structured problem solving.

---

## Status Legend
Every requirement ends in one of:
- `PENDING`
- `IN_PROGRESS`
- `EXECUTED`
- `VERIFIED`
- `BLOCKED`

---

## P0 — Core End-to-End Working System (Required First)

| ID | Section | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **REQ-01** | §2 | Environment feasibility audit & `ENVIRONMENT_REPORT.md` | `VERIFIED` |
| **REQ-02** | §30, §35, §36 | Project directory structure + `EXECUTION_PLAN.md`, `STATE.md`, `DECISIONS.md`, `EXPERIMENT_LOG.md`, `TODO.md` | `IN_PROGRESS` |
| **REQ-03** | §3 | Architecture evaluation & `ARCHITECTURE_DECISION.md` (with parameter count, dims, precision, raw/serialized/quantized size) | `PENDING` |
| **REQ-04** | §5 | Legal/license policy & provenance schema (`LICENSE_POLICY.md`, `DATA_PROVENANCE.json`) including `LICENSE_UNCLEAR` exclusion | `PENDING` |
| **REQ-05** | §4, §6, §7, §8 | Modular Internet Data Engine (`data_sources/` adapters: GitHub, HuggingFace, Docs/Stdlib, Synthetic) + full 14-stage pipeline (`src/data/`) | `PENDING` |
| **REQ-06** | §9, §11, §12 | Dataset composition across all 9 categories (A–I) + deterministic verified synthetic curriculum generators | `PENDING` |
| **REQ-07** | §13, §14 | Automatic data quality scoring, multi-level deduplication (exact, normalized, document/code similarity), train/eval contamination check, `DATA_QUALITY_REPORT.md` | `PENDING` |
| **REQ-08** | §15 | Custom BPE Tokenizer training (`src/tokenizer/`), efficiency & footprint measurement, `TOKENIZER_REPORT.md` | `PENDING` |
| **REQ-09** | §16, §17, §18 | Resource-aware JAX/XLA Transformer Training Engine (`src/model/`, `src/training/`) with checkpoints, resume, grad accum, LR schedule, crash recovery, and `experiments/EXP-*` manifests | `PENDING` |
| **REQ-10** | §19, §20 | Multi-domain Evaluation Engine (`src/evaluation/`) covering Language (Perplexity), Logic, Math, Algorithmic Reasoning, Code Syntax/Compile, Debugging, Code Generation (sandboxed execution), Instruction Following, and Unseen Generalization + `EVALUATION_REPORT.md` | `PENDING` |
| **REQ-11** | §25 | Minimal Inference Interface (`src/inference/`, `infer.py`) supporting CLI, Python API, deterministic/sampling modes, latency & RAM measurements | `PENDING` |
| **REQ-12** | §23, §24 | Model compression (FP32, FP16, INT8, INT4 weight-only quantization) + `release/` standalone deployable package strictly `< 50 MB` with checksums | `PENDING` |
| **REQ-13** | §31 | Automated `pytest` verification suite covering all 11 required test categories | `PENDING` |

---

## P1 — Quality, Curriculum, Iterative Self-Improvement & Regression Protection

| ID | Section | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **REQ-14** | §10, §21 | Ablation studies & Curriculum Learning (10-stage curriculum vs mixed training, model/tokenizer/data ablations) | `PENDING` |
| **REQ-15** | §22 | Iterative Self-Improvement Loop (Train → Evaluate → Identify Weaknesses → Targeted Data → Finetune → Evaluate → Compare) | `PENDING` |
| **REQ-16** | §27 | Model Regression Protection & `MODEL_COMPARISON.md` | `PENDING` |
| **REQ-17** | §28 | Failure-Recovery Protocol verification (tested checkpoint crash recovery, license rejection, bad endpoint fallback) | `PENDING` |

---

## P2 — Advanced Optimization & Continual Learning

| ID | Section | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **REQ-18** | §23 | Multi-precision quantization comparison (FP32 vs FP16 vs INT8 vs INT4 size, RAM, latency, accuracy) | `PENDING` |
| **REQ-19** | §26 | Continual Internet Training subsystem with dataset versioning (`dataset_v1`, `dataset_v2`, `dataset_v3`), evaluation gate, and automatic rollback | `PENDING` |

---

## P3 — Research Reporting, Documentation & Final Audits

| ID | Section | Requirement | Status |
| :--- | :--- | :--- | :--- |
| **REQ-20** | §38, §39, §42 | Complete `README.md`, `docs/TRAINING_REPORT.md`, and `FINAL_REPORT.md` with full specification audit, functional test audit, and evidence audit | `PENDING` |
