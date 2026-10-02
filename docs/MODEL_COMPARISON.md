# TinyMe — Model comparison (audit §10, §21, §2B)

Protocol versions are compared only when the tokenizer, dataset fingerprint and
evaluation split are identical. Rows that cannot be compared are marked
**NOT COMPARABLE** rather than placed in one table.

## 1. Trained checkpoints (all measured on the same held-out splits)

| Run | Arch | Params | Data | Steps | Val loss | Val ppl | Tool `task_completion` | Status |
| :--- | :--- | --: | :--- | --: | --: | --: | --: | :--- |
| `EXP-002-CORRECTED-NANO` | nano | 2,557,632 | dataset_v2 pretrain | 400 | 4.8042 | 122.02 | n/a | VALID BASELINE |
| `EXP-004-TOOL-SFT` | nano | 2,557,632 | v3 rev1 | 320 | 1.9757 | 7.21 | 0.00 | FAILED (rejected) |
| `EXP-004-TOOL-SFT-V2` | nano | 2,557,632 | v3 rev3 | 700 | 1.6874 | 5.41 | 0.00 | FAILED (rejected) |
| **`EXP-004-TOOL-SFT-V3`** | nano | 2,557,632 | v3 rev4 | 600 | **1.5305** | **4.62** | **0.00** | FAILED (rejected) — published as reference |
| `EXP-003-BASE-PILOT` | base | 8,933,440 | dataset_v3 | 25 | 6.6931 | 806.80 | not run | FEASIBILITY ONLY |
| `EXP-001` | nano | 2,557,632 | — | — | — | — | — | HISTORICAL / INVALID-AS-HELDOUT-BASELINE |

Lower perplexity does **not** indicate a better assistant here: the three SFT
iterations improve ppl monotonically and all complete zero tool tasks. The
comparison is published this way on purpose.

## 2. Quantisation variants of `EXP-004-TOOL-SFT-V3/best`

| Variant | Bytes | MiB | Rel. RMSE | Max abs error | Challenge mean acc | Tool syntax | Tool execution |
| :--- | --: | --: | --: | --: | --: | --: | --: |
| fp32 | 10,233,776 | 9.76 | 0 (ref) | 0 | 0.1000 | 0.84 | 0.50 |
| fp16 | 5,118,480 | 4.88 | 2.12e-04 | 4.88e-04 | 0.1000 | ≈ fp32 | ≈ fp32 |
| int8 | 2,600,976 | 2.48 | 6.44e-03 | 1.42e-03 | 0.1000 | 0.80 | 0.45 |
| int4 | 1,451,272 | 1.38 | 9.67e-02 | 2.54e-02 | 0.0000 | 0.40 | 0.00 |

## 3. Architecture options present in the repository

| Name | Bytes (fp32) | Parameters | Measured training throughput | Notes |
| :--- | --: | --: | --: | :--- |
| nano | 10,233,776 (serialized) | 2,557,632 | 663.9–849.4 tok/s | used for every trained run above |
| base | ~35.7 MB (32 × 8,933,440 + header) | 8,933,440 | 1,000.8 tok/s (25-step pilot) | feasible to run; corpus-blocked (§5 of `CAPACITY_ANALYSIS.md`) |
| medium | — | 17,308,032 | not measured | present in config; **not trained**, no claim made |

Base/Medium figures are labelled as extrapolation or not-measured; no capability
is claimed for an untrained configuration.

## 4. Curriculum pilots (§21)

`CURRIC-A-STAGED` (150 plain-document steps → 150 structured steps, i.e. global
step 300) versus `CURRIC-B-MIXED` (300 steps over the same blocks shuffled into
one pool), identical seed 20261002, identical effective batch (8 × 4, seq 256),
identical data revision (`dataset_v3` rev4). Measurement protocol for both:
per-domain `--max-samples 16` on validation/test/challenge plus the full 25-case
tool suite.

| Measure | A (staged) | B (mixed) | Reading |
| :--- | --: | --: | :--- |
| Final val loss / ppl (step 300) | 4.6215 / 101.65 | **4.4565 / 86.19** | B slightly lower |
| Validation split, mean accuracy | 0.0000 | **0.0455** | B ahead |
| Test split, mean accuracy | 0.0000 | **0.0909** | B ahead |
| Challenge split, mean accuracy | 0.0000 | 0.0000 | tie (both fail) |
| Tool syntax validity | **1.00** | 0.92 | A ahead |
| Tool-needed decision | 0.00 | **0.15** | B ahead |
| Execution success | 0.00 | **0.10** | B ahead |
| Grounded final answer | 0.00 | **0.25** | B ahead |
| **task_completion** | **0.00** | **0.00** | both fail |

**Verdict:** at equal step budget the mixed curriculum is as good or better on
every capability measure and no worse on generalisation; the staged ordering buys
nothing measurable in this pilot (its only win is syntax validity, 1.00 vs 0.92,
on a model that never completes a task). Reported as a negative result, with
neither configuration accepted: both remain at 0.00 task completion and 0.00
challenge accuracy. Sample sizes are small (16 per domain) by design — this is a
pilot, and the numbers above are not used to claim capability.
