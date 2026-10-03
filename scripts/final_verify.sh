#!/usr/bin/env bash
# TinyMe — final acceptance gate (audit §60).
#
#   bash scripts/final_verify.sh [experiment] [dataset] [checkpoint]
#
# Prints one PASS/FAIL line per requirement and ends with exactly one verdict
# line: ``FINAL VERIFY: PASS`` or ``FINAL VERIFY: FAIL``.  Every step executes a
# real command against real artefacts; a step that cannot run is a failure, not
# a skip, and there is no `|| true` anywhere in this file.
set -uo pipefail
cd "$(dirname "$0")/.."

EXP="${1:-EXP-015-TOOL-SFT-V9}"
DATASET="${2:-dataset_v9}"
CKPT="${3:-best}"
LOGDIR="docs/audit_evidence/final_verify"
mkdir -p "$LOGDIR"

FAILED=0
declare -a RESULTS=()

step() {
    local name="$1"; shift
    printf '\n=== %s ===\n' "$name"
    if "$@" > "$LOGDIR/$name.log" 2>&1; then
        printf 'PASS  %s\n' "$name"
        RESULTS+=("PASS  $name")
    else
        printf 'FAIL  %s  (log: %s/%s.log)\n' "$name" "$LOGDIR" "$name"
        tail -n 12 "$LOGDIR/$name.log" | sed 's/^/      /'
        RESULTS+=("FAIL  $name")
        FAILED=1
    fi
}

step_doc() {
    # a step whose verdict is carried by a JSON artefact the previous stages wrote
    local name="$1" path="$2"
    printf '\n=== %s ===\n' "$name"
    if [[ -s "$path" ]]; then
        printf 'PASS  %s (%s)\n' "$name" "$path"
        RESULTS+=("PASS  $name")
    else
        printf 'FAIL  %s: missing %s\n' "$name" "$path"
        RESULTS+=("FAIL  $name")
        FAILED=1
    fi
}

echo "TinyMe final acceptance gate"
echo "  experiment : $EXP"
echo "  dataset    : $DATASET"
echo "  checkpoint : $CKPT"
echo "  commit     : $(git rev-parse HEAD)"
echo "  dirty      : $(git status --porcelain | wc -l) file(s)"

# 1 ── the 68-row matrix must re-derive as verified from artefacts
step "matrix-strict" python3 scripts/verify_matrix.py --strict --dataset "$DATASET"

# 2 ── the required tests
step "pytest" python3 -m pytest tests/ -q

# 3 ── dataset integrity: counts, contamination, tokenizer, shards
step "dataset-integrity" python3 scripts/verify_dataset_integrity.py --dataset "$DATASET"

# 4 ── independent tool-use capability of the final checkpoint
step "tool-capability" python3 scripts/evaluate_tools.py \
    --experiment "$EXP" --checkpoint "$CKPT" --dataset "$DATASET" \
    --variants fp32,fp16,int8,int4 --backend numpy \
    --output "experiments/$EXP/evaluation_tools_${CKPT}.json"

# 5 ── release package: loadable, checksummed, under the size gate
step "release-contract" python3 scripts/verify_release.py --release release --dataset "$DATASET"

# 6 ── standalone package: clean-room load + inference
step "local-model" python3 scripts/validate_local_model.py --package local_model \
    --out docs/audit_evidence/local_model_validation.json

# 7 ── GGUF: the artefact must exist, load in the real runtime, and generate
step "gguf-runtime" python3 scripts/verify_gguf.py --gguf release/tinyme-f16.gguf \
    --reference checkpoints/"$EXP"/"$CKPT".safetensors \
    --out docs/audit_evidence/gguf_runtime_test.json

# 8 ── fresh clone: the documented flow must work from GitHub alone
step "fresh-clone" bash scripts/fresh_clone_check.sh

echo
echo "---------------- summary ----------------"
for line in "${RESULTS[@]}"; do echo "  $line"; done
if [[ "$FAILED" -eq 0 ]]; then
    echo "FINAL VERIFY: PASS"
else
    echo "FINAL VERIFY: FAIL"
fi
exit "$FAILED"
