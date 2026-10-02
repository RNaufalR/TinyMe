# EXECUTION PLAN — corrective audit (2026-10-02)

Authoritative contract: [`TinyMeAudit.md`](TinyMeAudit.md).
Supersedes the earlier `TINY_AI_TRAINING_LAB.md` plan, whose v1 pipeline is
retired (`src/data/pipeline.py` now raises with a pointer to the v2 modules).

Status labels: `PLANNED` · `EXECUTED` (ran) · `VERIFIED` (measured/tested) ·
`BLOCKED` (with the reason recorded).

## P0 — correctness and provenance (audit §5–§18)

| ID | Requirement | Status | Evidence |
| :--- | :--- | :--- | :--- |
| P0-01 | Content-aware preprocessing: preserve indentation/fences/control tokens, AST on the target only, malformed code rejected | `VERIFIED` | `src/data/preprocess.py`, `tests/test_preprocessing.py`, `tests/test_quality_filters.py` |
| P0-02 | Segmented record contract + `loss_mask` | `VERIFIED` | `src/data/records.py`, `tests/test_loss_mask.py` |
| P0-03 | Task-aware masking: tool results / tool calls as context only | `VERIFIED` | `tests/test_loss_mask.py` |
| P0-04 | Padding mask (pad → zero loss) + active/masked/padding accounting | `VERIFIED` | `tests/test_padding_mask.py`, `docs/DATA_QUALITY_REPORT.md` |
| P0-05 | Real train/validation/test split with group separation + contamination report | `VERIFIED` | `src/data/splits.py`, `docs/SPLIT_SUMMARY.md`, contamination **PASS** |
| P0-06 | Trainer loads `validation.jsonl`/`test.jsonl`, never `train[:32]` | `VERIFIED` | `scripts/train.py`, `tests/test_dataset_api.py` |
| P0-07 | Tokenizer trained on the train split only + report + startup vocab assert | `VERIFIED` | `docs/TOKENIZER_REPORT.md`, `tests/test_tokenizer.py` |
| P0-08 | One authoritative sequence builder (seq 256 primary, 512 secondary, target-aware truncation) | `VERIFIED` | `src/data/sequence.py`, `tests/test_sequence.py` |
| P0-09 | True gradient accumulation (frozen params + numerical-equivalence test) | `VERIFIED` | `tests/test_grad_accumulation.py` |
| P0-10 | Deterministic sampler with persisted state | `VERIFIED` | `src/training/sampler.py`, `tests/test_sampler.py` |
| P0-11 | Atomic checkpoints, save/load/verify, fingerprint compatibility | `VERIFIED` | `src/training/checkpoint.py`, `tests/test_checkpoint_resume.py` |
| P0-12 | Scheduler/optimizer resume continuity (LR before == after) | `VERIFIED` | `tests/test_checkpoint_resume.py` |
| P0-13 | RNG synchronisation (python/numpy/JAX/data/synthetic) | `VERIFIED` | `tests/test_rng_determinism.py` |
| P0-14 | RoPE pairing fix + full-forward vs KV-cache parity before retraining | `VERIFIED` | `docs/audit_evidence/rope_pairing_probe.out.txt`, `cache_parity_probe.out.txt`, `tests/test_rope.py`, `tests/test_kv_cache.py` |
| P0-15 | Numerical stability: finite loss, NaN/Inf guard, grad-norm logging | `VERIFIED` | `tests/test_numerical_stability.py`, training logs |
| P0-16 | `compute_dtype` affects real computation and is recorded | `VERIFIED` | `tests/test_dtype.py`, `run_config.json` |
| P0-17 | Tokenizer/checkpoint vocabulary assertion at startup | `VERIFIED` | `tests/test_inference.py` |

## P1 — data, tools, sandbox, evaluation (audit §19–§33)

| ID | Requirement | Status | Evidence |
| :--- | :--- | :--- | :--- |
| P1-01 | Density rebuild without duplicate inflation; multi-level dedup index | `VERIFIED` | `data_sources/synthetic_v2.py` reports requested/emitted/dropped; `docs/DATA_QUALITY_REPORT.md` |
| P1-02 | Code-specific quality handling (secret/PII/malware scan; no Python AST on non-Python) | `VERIFIED` | `src/data/quality_filter.py`, `tests/test_quality_filters.py` |
| P1-03 | Efficient shard writing (buffered + atomic replace) | `VERIFIED` | `src/data/shard_writer.py` |
| P1-04 | Provenance + licence policy; corpus growth beyond 55 k tokens | `VERIFIED` | 588,816 active target tokens; `docs/DATA_PROVENANCE.json`, `docs/LICENSE_POLICY.md` |
| P1-05 | Two-stage schedule (A pretrain, B instruction/reasoning/tool SFT) | `EXECUTED` A; B `PLANNED` (EXP-004) | `TrainConfig.stage`, `docs/TRAINING_REPORT.md` |
| P1-06 | Strict protocol tokens + small tool surface (search/fetch/compute/code/files) | `VERIFIED` | `src/agent/protocol.py`, `src/agent/tool_registry.py`, `tests/test_tool_protocol.py` |
| P1-07 | Protocol-exact training data (arguments + result envelopes match the runtime) | `VERIFIED` | 333/333 calls pass `validate_arguments`; `tests/test_tool_protocol.py` |
| P1-08 | Agent runtime: router, planner, executor, evidence engine, loop/budget control | `VERIFIED` | `src/agent/`, `tests/test_agent_loop.py` |
| P1-09 | Sandbox with auto-detected isolation and honest labelling | `VERIFIED` | `detect_isolation(force=True)` → `namespace(net+mount)`; `docs/SANDBOX.md` |
| P1-10 | Limits: CPU/mem/time/output/file/process, restricted FS/env, no secrets, network off | `VERIFIED` | `run_escape_suite()` 12/12; `docs/audit_evidence/sandbox_escape_suite.out.txt` |
| P1-11 | Independent domain evaluation, executable pass rate, tool metrics, >25 samples | `PLANNED` | `scripts/evaluate.py` ready; runs after EXP-002 completes |
| P1-12 | Functional quantization comparison FP32/FP16/INT8/INT4 with measured bytes | `PLANNED` | round-trip error measured; functional scores after EXP-002 |
| P1-13 | Release package with checksums, reports and standalone entry point | `PLANNED` | `scripts/package_model.py` verified on a smoke package |
| P1-14 | 21+ module test suite green before training and before release | `EXECUTED` before training (185 passed); rerun before release | `tests/` |

## P2 — stretch and reporting

| ID | Requirement | Status |
| :--- | :--- | :--- |
| P2-01 | `EXP-003-CORRECTED-BASE` feasibility pilot on 2 vCPU (documented go/no-go) | `PLANNED` |
| P2-02 | `EXP-005-QUANT` as a named quantization experiment | `PLANNED` |
| P2-03 | `FINAL_REPORT.md` with 20 sections + acceptance gate table | `PLANNED` |

## Order of operations that was actually followed

1. audit evidence probes (RoPE pairing, cache parity, dataset state) → `docs/CORRECTIVE_AUDIT.md`;
2. P0 fixes + tests; P1 sandbox/agent/tools; dataset contract (v2);
3. corpus diversity fix → splits stratification → **tool-protocol drift fix** (three invalidated
   EXP-002 attempts, each stopped, renamed and logged);
4. fresh `EXP-002-CORRECTED-NANO` pretrain on the protocol-exact `dataset_v2`;
5. held-out evaluation (test + challenge), quantization comparison, packaging;
6. documentation sync + final acceptance gates.
