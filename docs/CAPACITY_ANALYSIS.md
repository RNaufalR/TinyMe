# TinyMe — Model capacity and data-budget analysis (measured, not estimated)

This document answers a question the loss curves do not: **why does a model with
val_perplexity 4.62 still fail to answer?** All inputs are read from
`experiments/*/summary.json` and `datasets/versions/dataset_v3/manifest.json`.

## 1. The numbers

| Quantity | Value | Source |
| :--- | ---: | :--- |
| Parameters (nano) | 2,557,632 | `summary.json: parameter_count` |
| Active target tokens, train split | 718,077 | `dataset_v3/manifest.json` |
| Active validation tokens | 37,311 | training log (`eval step ... active=37311`) |
| Context tokens processed, EXP-004-V3 | 1,475,643 | `summary.json: tokens_processed` |
| Chinchilla-optimal budget (20 × N) | 51,152,640 | 20 × 2,557,632 |
| 6N lower bound | 15,345,792 | 6 × 2,557,632 |

**The training corpus supplies 1.40 % of the Chinchilla-optimal token budget**
(4.68 % of the generous 6N lower bound). Two passes (2.05 epochs) over that
corpus still leaves a ~50× data deficit.

## 2. What this explains

The model memorised the frequent template strings (a single thought sentence
appeared 382 times = 5.7 % of all target segments) because in a 0.72 M-token
corpus the highest-frequency strings *are* the distribution. It then emitted
those strings in place of task content. Template diversity (fix rev4) reduces the
most extreme case, but it cannot create the ~50× missing supervision that would
teach composition.

The arithmetic evidence is consistent:

* copy-arithmetic (`77 + 54 - 13`) needs input copying → *is* partially learned;
* multi-step and error recovery, which require composing two or three
  behaviours in one trajectory, are at 0.00 — they are the rarest patterns and
  the first casualties of a starved corpus.

## 3. Feasibility on the measured environment (§2B)

Environment: 2 vCPU, ≈3.9 GB RAM, no GPU, JAX CPU backend.

| Config | Params | Measured throughput | Time for 6N tokens | Time for 20N tokens |
| :--- | ---: | ---: | ---: | ---: |
| nano (measured, 600-step run) | 2,557,632 | 663.9–849.4 tok/s | 5.0 h | 16.7 h |
| base (measured 25-step pilot) | 8,933,440 | 1,000.8 tok/s* | 14.9 h | 49.6 h |

\* The base pilot's tokens/s is higher because the micro-batch is 4 with a
shorter ramp and because nano's 600-step run spends part of its budget in
evaluation; throughput is not monotonic in parameter count here. The pilot is a
feasibility measurement (25 steps, val_ppl 806.8 — **not** a capability claim),
recorded in `experiments/EXP-003-BASE-PILOT/summary.json`.

**Verdict for §2B:** a Base-scale *pilot* is feasible in wall-clock terms
(hours), but it is not feasible to train Base **to its token budget** on this
machine together with the token supply we have: the binding constraint is the
corpus (0.72 M active tokens), not compute. Running Base on 0.72 M tokens would
repeat the same failure with 3.5× more parameters — a documented decision, so
the audit does not mistake "we could have" for "we should have".

## 4. Capacity ceiling demonstrated independently

`EXP-002-CORRECTED-NANO` (Stage-A pretrain, 2.87 M tokens, 5 epochs) reached
val_perplexity 122.0 — the model is *undertained*, not broken: train loss 4.5152
vs val loss 4.8042 shows no train/val divergence, i.e. the run was still on the
learning part of the curve when the budget ended.

The subsequent SFT stage proves the pipeline learns structure (val_ppl
1.98 → 1.69 → 1.53 across iterations) at the same time as the independent
capability suite stays at 0.00 for composed behaviour. The two facts together,
plus the 1.4 % budget figure, are the honest explanation of the outcome.

## 5. What a realistic remediation would require (not done here)

1. **Corpus scale-up** to ≥ 15 M active target tokens (≥ 6N) with real
   diversity; the generator in `data_sources/synthetic_v2.py` can be scaled but
   the *task* diversity, not the row count, is the bottleneck.
2. **Composition supervision**: multi-step trajectories must appear thousands of
   times each, not hundreds.
3. **Capacity**: at that point a Base/Medium model is justified; on this
   CPU-only box the extrapolated cost is ~50 h per 20N pass for Base.

None of these are claimed as completed work; they are recorded so that the
failure is attributed correctly (data + capacity budget) rather than to an
unexplained model defect.
