# TODO — status after the audit execution

Legend: `[x]` done and evidenced · `[~]` executed, outcome negative but measured ·
`[ ]` open · `[<]` blocked.

## Correctness layer (P0)
- [x] Content-aware preprocessing (indentation, fences, control tokens, AST-validated targets)
- [x] Segmented record contract with `loss_mask` (`IGNORE_INDEX = -100`), tool results context-only
- [x] Padding mask + reported active/masked/padding token counts
- [x] Split on `group_id` with exact/normalized/MinHash/code-shingle/template contamination gates
- [x] True gradient accumulation (params frozen until update) + equivalence test
- [x] Atomic checkpoint save/load/verify with fingerprints, optimizer/scheduler/sampler/RNG state
- [x] Single sequence builder; prompt is an exact prefix of the training sequence
- [x] Trainer refuses `train.jsonl` for evaluation
- [x] RoPE correctness (independent rotation reference) and full-vs-cached parity
- [x] `decode(skip_special_tokens=False)` + `<|bos|>` in generation

## Data
- [x] `dataset_v3` rev4 built: train 7,220 / val 1,422 / test 1,558 / challenge 60
- [x] 718,077 active train target tokens counted from `labels != -100`
- [x] Contamination PASS on every split pair
- [x] Thought-template diversity pool (6 phrasings/family) replaces the single repeated sentence
- [ ] Corpus scale-up to ≥ 6N (15.3 M) active target tokens with genuinely diverse multi-step supervision
- [<] Live network corpora (TLS-blocked); offline cached slice + verified synthetic data only

## Training
- [x] `EXP-002-CORRECTED-NANO` 400 steps, ppl 122.02, deterministic rerun
- [~] `EXP-004-TOOL-SFT` (320), `-V2` (700), `-V3` (600) — all rejected on capability
- [x] `EXP-003-BASE-PILOT` feasibility (8.93 M params, 1,000.8 tok/s); verdict in `docs/CAPACITY_ANALYSIS.md`
- [ ] Base/Medium trained to budget — blocked by corpus, documented
- [x] Curriculum A vs B pilots (`experiments/CURRIC-A-STAGED`, `experiments/CURRIC-B-MIXED`)

## Evaluation
- [x] Per-domain evaluator (14 domains, no collapsed score) with per-domain latency/throughput
- [x] §4 tool-use suite A–H, 25 independent cases
- [x] Test split evaluated (2,158 samples, mean accuracy 0.0571)
- [x] Challenge split evaluated (held-out templates, mean accuracy 0.1000)
- [x] Quantised variants evaluated (fp32/fp16/int8/int4) on both suites
- [<] Multi-step / error-recovery capability — measured 0.00, root cause documented

## Release
- [x] `scripts/package_model.py` audits fixed (strict config, write order, inventory, README, schema-aware report)
- [x] `release/` built: 13 files, 19,688,665 B, all four variants < 50 MB
- [x] `checksums.txt` verified; `manifest.json` records dataset/tokenizer/model/config fingerprints
- [x] Clean-room gate (`env -i`, only release files) runs `inference.py`
- [x] `evaluation_report.md` rendered from the real schemas (was 0 samples / empty tables)

## Documentation
- [x] `docs/AUDIT_MATRIX.md` (§1) with per-row status + defect tables
- [x] `docs/{TRAINING_REPORT,QUANTIZATION_REPORT,CAPACITY_ANALYSIS,MODEL_COMPARISON}.md`
- [x] `FINAL_REPORT.md`
- [x] `STATE.md`, `TODO.md`, `DECISIONS.md`, `EXPERIMENT_LOG.{md,jsonl}` synchronised
