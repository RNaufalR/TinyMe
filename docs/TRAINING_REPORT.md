# TRAINING REPORT — EXP-002-CORRECTED-NANO (stage A pretrain)

Run executed 2026-10-02 on the audit-corrected data contract
(`dataset_v2`, fingerprint `a8f5c37a…`). Fresh initialisation: `EXP-001` is
HISTORICAL / INVALID-AS-HELDOUT-BASELINE and was never resumed.

## 1. Exact command

```bash
python scripts/train.py \
  --experiment EXP-002-CORRECTED-NANO \
  --experiment-name "TinyMe Nano pretrain on protocol-exact dataset_v2" \
  --stage pretrain --arch nano --dataset dataset_v2 \
  --seq-len 256 --micro-batch 8 --grad-accum 4 --max-steps 400 \
  --lr 6e-4 --dtype float32
```

## 2. Configuration (recorded in `experiments/EXP-002-CORRECTED-NANO/run_config.json`)

| Item | Value |
| :--- | :--- |
| Architecture | `nano` — d_model 192, 4 layers, 6 heads, d_ff 512, vocab 4096, ctx 512 |
| Parameters | 2,557,632 (fp32 file 10,230,528 B; checkpoint with optimizer 30,703,744 B) |
| Stage | `pretrain` (stage A) |
| seq_len | 256 (primary; 512 reserved as the secondary setting) |
| micro-batch × accumulation | 8 × 4 = effective batch 32 |
| Optimizer | AdamW via `optax.chain(clip_by_global_norm(1.0), adamw(lr=6e-4, wd=0.01))` |
| Schedule | 40-step linear warmup → cosine to 10 % of peak |
| compute_dtype | `float32` (embeddings/weights/loss all float32 — verified by `tests/test_dtype.py`) |
| Seed | 20261002 (python/numpy/JAX/data/synthetic all synced) |
| Data | `dataset_v2` 4227/741/810/60; train blocks 2203, active target tokens 494,571, padding ratio 0.1231 |
| Validation | 378 blocks, 88,000 active target tokens, padding ratio 0.0906 |

## 3. Result

| Metric | Value |
| :--- | ---: |
| Steps executed | **400 / 400** (5 epochs; `steps_this_run = 400`) |
| Final train loss | 4.5152 |
| Final validation loss | **4.8042** |
| Final validation perplexity | **122.02** |
| Best validation loss | 4.8042 (monotonic improvement at every evaluation) |
| Tokens processed | 2,872,417 |
| Wall clock | 950.88 s (15 min 51 s) |
| Throughput | 3,020.8 tokens/s (2 vCPU, no accelerator) |
| Non-finite steps | 0 |
| Status | `COMPLETED` |

Validation trajectory (evaluation every 50 steps):

| Step | val_loss | val_ppl |
| ---: | ---: | ---: |
| 50 | 6.5391 | 691.64 |
| 100 | 5.9893 | 399.13 |
| 150 | 5.3844 | 217.98 |
| 200 | 5.0833 | 161.30 |
| 250 | 4.9402 | 139.81 |
| 300 | 4.8632 | 129.44 |
| 350 | 4.8214 | 124.14 |
| 400 | **4.8042** | **122.02** |

## 4. What is verified by this run

* **Real held-out validation.** The number above comes from
  `datasets/versions/dataset_v2/validation.jsonl` (741 records, 378 blocks),
  never from a slice of the training file (`tests/test_dataset_api.py`).
* **True gradient accumulation.** Micro-batches accumulate with parameters
  frozen; the numerical-equivalence test compares 4 × micro-8 against a single
  batch-32 step (`tests/test_grad_accumulation.py`).
* **Padding and masking are real.** 12.3 % of train positions are padding and
  contribute exactly zero loss; targets marked `IGNORE_INDEX (-100)` are
  excluded, so validation loss is computed on the same active-token basis
  (88,000 tokens per evaluation).
* **Determinism.** RNG state for python/numpy/JAX/data/synthetic is persisted in
  the checkpoint; `tests/test_rng_determinism.py` proves identical draws after
  restore.
* **Checkpointing.** `best`, `latest`, `step_000200/300/400` written atomically
  with sha256 + fingerprint compatibility; `verify_checkpoint` re-checks them.
  A resume test loads `latest` and confirms LR continuity (`tests/test_checkpoint_resume.py`).
* **Numerical stability.** Every step logs a finite gradient norm (0.36–1.10 in
  this run) and no NaN/Inf occurred (`non_finite_steps = 0`).

## 5. Honest reading of the loss

Perplexity 122 on a corpus dominated by code, structured reasoning and tool
transcripts is *not* evidence of general competence. It says the model learned
the corpus statistics at a small scale; held-out task accuracy (executable code
pass rates, tool selection/arguments, exact answers) is measured separately in
`docs/MODEL_COMPARISON.md` and `release/evaluation_report.md`. Any comparison
against `EXP-001` (ppl 2.4551 on a contaminated split with a different
tokenizer and no held-out protocol) is **NOT COMPARABLE** and is labelled as
such everywhere it appears.

## 6. Intentionally not claimed

* No claim that 2.6 M parameters store factual world knowledge — factual answers
  are expected to come from the tool runtime and are grounded in citations.
* No claim of state-of-the-art anything. The reference points in this repository
  are the historical `EXP-001` artifacts, which are kept only as evidence of the
  audit findings.
