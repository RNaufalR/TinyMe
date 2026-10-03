#!/usr/bin/env bash
# TinyMe — the final, documented retraining order (audit §58).
#
#   bash scripts/run_final_training.sh [dataset] [base] [drill] [sft]
#
#   0. build the corpus from the committed pipeline (fresh, seeded, reproducible)
#   1. EXP-013-BASE-V9      pretrain on the plain-document stage      (nano, 512)
#   2. EXP-014-DRILL-V9     drills: single-turn, dense supervision    (nano, 512)
#   3. EXP-015-TOOL-SFT-V9  full structured mixture (drills + multi-step trajectories)
#
# Every stage initialises from the previous stage's *best* checkpoint, records its
# exact configuration in experiments/<id>/run_config.json, and never resumes a
# finished run (--no-resume on the first launch; a rerun must use a new id).
set -euo pipefail
cd "$(dirname "$0")/.."

DATASET="${1:-dataset_v9}"
BASE="${2:-EXP-013-BASE-V9}"
DRILL="${3:-EXP-014-DRILL-V9}"
SFT="${4:-EXP-015-TOOL-SFT-V9}"

step() { printf '\n=== %s ===\n' "$1"; }

step "1/4 corpus $DATASET"
python3 scripts/prepare_data_v2.py --version "$DATASET" --generators v3 \
    --synthetic-profile v6 --scale 1.0 --scale-v3 1.0 --seq-len 512

step "2/4 $BASE — pretrain (plain documents)"
python3 scripts/train.py --experiment "$BASE" --stage pretrain --arch nano \
    --dataset "$DATASET" --seq-len 512 --micro-batch 8 --grad-accum 4 \
    --max-steps 500 --lr 6e-4 --warmup 40 --eval-every 100 --checkpoint-every 50 \
    --seed 20261002 --dtype float32 --no-resume \
    --notes "v9 corpus (reproducible generator): pretrain stage, plain documents only"

step "3/4 $DRILL — curriculum stage 1 (drills)"
python3 scripts/train.py --experiment "$DRILL" --stage drill --arch nano \
    --dataset "$DATASET" --seq-len 512 --micro-batch 8 --grad-accum 4 \
    --max-steps 600 --lr 5e-4 --warmup 60 --eval-every 100 --checkpoint-every 50 \
    --seed 20261002 --dtype float32 --init-from "$BASE" --no-resume \
    --notes "DEC-013 stage 1: single-turn drills (copy spans, arithmetic, text edits, one-call tool flows)"

step "4/4 $SFT — curriculum stage 2 (full mixture)"
python3 scripts/train.py --experiment "$SFT" --stage sft --arch nano \
    --dataset "$DATASET" --seq-len 512 --micro-batch 8 --grad-accum 4 \
    --max-steps 700 --lr 3e-4 --warmup 50 --eval-every 100 --checkpoint-every 50 \
    --seed 20261002 --dtype float32 --init-from "$DRILL" --no-resume \
    --notes "DEC-013 stage 2: drills retained + multi-step tool trajectories (search/fetch/repair/recovery)"

step "done"
echo "candidate: checkpoints/$SFT/best.safetensors"
echo "next: python3 scripts/evaluate_tools.py --experiment $SFT --checkpoint best --dataset $DATASET"
