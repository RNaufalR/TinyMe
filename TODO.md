# TODO — corrective audit execution

Status legend: [x] done & verified · [~] in progress · [ ] pending

## P0 — correctness (audit §5–§18)
- [x] P0-01 content-aware preprocessing (code indentation/fences/control tokens preserved, AST on target only)
- [x] P0-02 structured segmented record contract + `loss_mask`
- [x] P0-03 task-aware loss masking (tool results context-only)
- [x] P0-04 padding mask (pad_id → 0 loss, active/masked/padding reported)
- [x] P0-05 real train/validation/test split with group separation + contamination report (PASS)
- [x] P0-06 trainer loads `validation.jsonl`; never `train.jsonl[:32]`
- [x] P0-07 tokenizer trained on the train split only + `TOKENIZER_REPORT.md` + vocab assert
- [x] P0-08 single authoritative sequence builder (seq_len 256, target-aware truncation, packing)
- [x] P0-09 true gradient accumulation (params frozen + numerical-equivalence test)
- [x] P0-10 deterministic sampler with persisted state
- [x] P0-11 atomic checkpoints with load/verify + fingerprint compatibility
- [x] P0-12 scheduler/optimizer resume continuity (LR before == after)
- [x] P0-13 RNG sync (python/numpy/JAX/data/synthetic)
- [x] P0-14 RoPE + full-forward vs KV-cache parity tests
- [x] P0-15 numerical stability (finite loss, NaN/Inf guard, grad-norm logging)
- [x] P0-16 `compute_dtype` affects real computation and is recorded
- [x] P0-17 tokenizer/checkpoint vocabulary assertion at startup

## P1 — data, tools, sandbox, evaluation
- [x] density rebuild without duplicate inflation; multi-stage dedup index
- [x] code-specific quality handling (secret/PII/malware scan; no Python AST on non-Python)
- [x] efficient shard writing (buffered + atomic `os.replace`)
- [x] provenance + license policy; corpus growth beyond 55 k tokens (585 k active target tokens)
- [x] two-stage schedule (A pretrain / B instruction+reasoning+tool SFT) in the data layer
- [x] strict tool protocol + small model surface (search/fetch/compute/code/files)
- [x] agent runtime: router, planner, executor, tool registry, evidence engine, loop/budget control
- [x] sandbox: policy/limits/workspace/isolation/runner with honest auto-detected isolation
- [~] independent domain evaluation incl. executable pass rate + tool metrics (`scripts/evaluate.py` written; run after training)
- [ ] functional quantization comparison (FP32/FP16/INT8/INT4) — code ready, measurement after EXP-004
- [ ] 21+ module test suite: 22 modules exist, `test_data_pipeline.py` / `test_package_size.py` pending final artefacts
- [ ] release package (`release/` model.*, tokenizer, config, manifest, checksums, reports, inference entrypoint)

## Training / experiments
- [~] EXP-002-CORRECTED-NANO (canonical Stage-A run in progress, fresh init, never resumes EXP-001)
- [ ] EXP-003-CORRECTED-BASE feasibility pilot
- [ ] EXP-004-TOOL-SFT (Stage B, loss-masked tool/instruction training)
- [ ] EXP-005-QUANT (quantized variants + functional comparison)

## Documentation
- [x] `docs/CORRECTIVE_AUDIT.md`, `docs/audit_evidence/*`, `docs/LICENSE_POLICY.md`, `docs/DATA_QUALITY_REPORT.md`,
      `docs/TOKENIZER_REPORT.md`, `docs/SPLIT_SUMMARY.md`, `DATA_PROVENANCE.json`
- [ ] `docs/TRAINING_REPORT.md`, `docs/MODEL_COMPARISON.md` (NOT COMPARABLE where protocol changed)
- [ ] `FINAL_REPORT.md` (20 sections), `EXPERIMENT_LOG.md` entries, `README.md` rewrite
- [~] `STATE.md` / `TODO.md` / `DECISIONS.md` / `EXECUTION_PLAN.md` kept in sync
