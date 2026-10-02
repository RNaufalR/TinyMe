# TinyMe — Training report (audit §2, §10, §21, §22, §23)

All numbers in this file were produced by `scripts/train.py` runs in this
repository and are read back from `experiments/*/summary.json` and
`experiments/*/metrics.jsonl`. Nothing here is estimated.

## 1. Corrected baseline (Stage A) — `EXP-002-CORRECTED-NANO`

| Item | Value |
| :--- | :--- |
| Architecture | `nano`, 2,557,632 parameters |
| Context / tokens | seq_len 256, 2,872,417 tokens processed |
| Optimisation | micro-batch 8 × accum 4 (effective 32), lr 6e-4, warmup 50, fp32, seed 20261002 |
| Steps | 400 (5 epochs over `dataset_v2` pretrain blocks) |
| Final train loss | 4.5152 |
| Final validation loss / ppl | 4.8042 / 122.02 (88,000 validation target tokens) |
| Wall clock | 1078.66 s (2,662.9 tokens/s on 2 vCPU) |
| Checkpoint | 30,703,744 B (weights + optimizer + RNG) |

Determinism: the run was repeated end-to-end (`logs/EXP-002-CORRECTED-NANO-rerun.log`).
All 400 logged steps match the committed run exactly (0 mismatches on loss, lr,
grad_norm, val_loss, val_ppl); only wall-clock and tokens/s differ.

### Why a corrected retrain was necessary

`EXP-001` is retained as a **historical, invalid-as-held-out-baseline** experiment
(see `TinyMeAudit.md`). It is not resumed and its metrics are never compared
against the corrected runs.

## 2. Tool-SFT stage — three measured iterations (§22 loop)

The tool-SFT stage is the intervention the audit asks for: teach the protocol
(needed/not-needed tool decision, valid arguments, result interpretation,
grounded final answer, multi-step, error recovery). Three iterations were run
because the independent evaluation kept finding *real* defects:

| Iteration | Experiment | Data revision | Steps | Train loss | Val loss | Val ppl | Issue found by the *next* evaluation |
| :--- | :--- | :--- | --: | --: | --: | --: | :--- |
| 1 | `EXP-004-TOOL-SFT` | v3 rev1 (no turn opener) | 320 | 0.9878 | 1.9757 | 7.21 | runtime could not parse generations: markers stripped by the decoder, BOS missing, tool-result template leakage in prompts |
| 2 | `EXP-004-TOOL-SFT-V2` | v3 rev3 (opener fixed, varied shapes) | 700 | 0.6395 | 1.6874 | 5.41 | model emits memorised thought sentences instead of task content: the corpus had **1 distinct thought string across 382 repair records** |
| 3 | `EXP-004-TOOL-SFT-V3` | v3 rev4 (thought pool, 6 phrasings/family) | 600 | see `experiments/EXP-004-TOOL-SFT-V3/summary.json` | | | |

Loss alone never decided acceptance: iteration 2 has the better loss but
*capability* is judged by `experiments/*/evaluation_tools_best.json` (§4), and
iteration 3 was launched because the capability measurement showed the model
answering arithmetic prompts with a code-generation sentence.

### 8-point intervention record (§22)

| Field | Iteration 2 → 3 |
| :--- | :--- |
| Observed failure | Tool evaluation: `tool_name_accuracy` 0.25; finals not grounded; operand copying absent |
| Hypothesis | Target text is dominated by a single repeated sentence, so the model's prior over template text exceeds its prior over task content |
| Target behaviour | Emit task-specific content (copied operands, values from the tool result) rather than boilerplate |
| Data modification | `data_sources/synthetic_v2.py`: `_THOUGHTS` pool (6 phrasings per family) replaces 3 hard-coded sentences; compute expression shapes varied (6 shapes) |
| Training modification | 600 steps (plateau point of iteration 2), otherwise identical schedule/seed |
| Experiment ID | `EXP-004-TOOL-SFT-V3` (new id; earlier runs are kept) |
| Evaluation protocol | `scripts/evaluate_tools.py` (25 independent cases) + `scripts/evaluate.py` on test/challenge |
| Result | recorded in `experiments/EXP-004-TOOL-SFT-V3/` (see `EXPERIMENT_LOG.md` for the verdict) |

## 3. Data revisions actually used

| Build (log) | Splits (train/val/test/challenge) | Active train targets | Contamination | Thought diversity |
| :--- | :--- | --: | :--: | :--: |
| `logs/prepare_dataset_v3.log` (rev1) | 6,144 / 1,204 / 1,196 / 60 | 668,457 | PASS | no (single repeated thought sentence) |
| `logs/prepare_dataset_v3_rev2.log` | 6,144 / 1,204 / 1,196 / 60 | 668,457 | PASS | no |
| `logs/prepare_dataset_v3_rev3.log` | 6,787 / 1,711 / 1,569 / 60 | 700,451 | PASS | no |
| `logs/prepare_dataset_v3_rev4.log` | 7,220 / 1,422 / 1,558 / 60 | **718,077** | PASS | yes (6 phrasings/family) |

`EXP-004-TOOL-SFT-V3` trained on rev4; `-V2` on rev3; `-V1` on rev2 (whose train
data is identical to rev1 — only the challenge shard changed, verified by the
sha256 of every `.npy`). Shards fingerprint of the trained revision:
`fe6abc98686c7913155bd88b1cf3fd5082ffdc97577da17e259d1f9404f7b120`.

## 4. Base architecture feasibility (§2B)

The environment is CPU-only (2 vCPU, ≈3.9 GB RAM, no GPU). Base is 8,933,440
parameters (3.5× nano). Measured pilot figures are in
`experiments/EXP-003-BASE-PILOT/` and `docs/MODEL_COMPARISON.md`.

## 5. Curriculum A vs B (§21)

Two pilots on identical data/steps with identical seeds:

* **A (staged)**: Stage-A pretrain on plain documents → SFT on structured
  records.
* **B (mixed)**: a single shuffled pool (`--stage mixed`) of the same blocks.

Results and the honest verdict (measured differences, no winner declared) are in
`experiments/CURRIC-*/summary.json` and `docs/MODEL_COMPARISON.md`.

## 6. Overfitting analysis (§23)

| Signal | Observation (iteration 2) | Reading |
| :--- | :--- | :--- |
| train ↓ / val ↓ | 0.64 / 1.69 | learning, not diverging |
| val plateau | 1.645 (400) → 1.665 (500) → 1.676 (600) | mild overfit beyond step 400; `best` = step 400 |
| test vs validation | see `evaluation_test.json` | held-out template gap |
| challenge | see `evaluation_challenge.json` | hardest held-out templates |

Conclusions are recorded conservatively: the model memorises template structure
and only partially copies task content; the capability section of
`FINAL_REPORT.md` states the measured accuracy rather than claiming success.
