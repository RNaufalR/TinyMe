#!/usr/bin/env bash
# TinyMe — collect every artefact the final verification report cites.
#
#   bash scripts/collect_final_evidence.sh EXP-015-TOOL-SFT-V9 dataset_v9 [max_samples]
#
# The script is deliberately linear and loud: each stage writes its own log under
# docs/audit_evidence/logs/ and the run aborts on the first non-zero exit.  There
# is no `|| true`, no retry that hides a failure, and no stage that "passes" by
# producing an empty file — the artefacts the verification matrix parses are the
# ones written here, so a silent failure cannot be papered over later.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${1:?usage: collect_final_evidence.sh <experiment> [dataset] [max_samples]}"
DATASET="${2:-dataset_v9}"
SAMPLES="${3:-300}"
LOGDIR="docs/audit_evidence/logs"
mkdir -p "$LOGDIR"

stage() {
    local name="$1"; shift
    printf '\n===== %s =====\n' "$name"
    local log="$LOGDIR/${name}.log"
    local start
    start=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    if "$@" >"$log" 2>&1; then
        echo "[OK]   $name  ($start)  log=$log"
    else
        local rc=$?
        echo "[FAIL] $name  exit=$rc  log=$log"
        tail -25 "$log"
        exit "$rc"
    fi
}

# 1. held-out tool suite A–H, every precision variant, executed through the real runtime
stage evaluate_tools python3 scripts/evaluate_tools.py --experiment "$EXP" --checkpoint best \
    --dataset "$DATASET" --variants fp32,fp16,int8,int4 --backend numpy \
    --max-new-tokens 96 --temperature 0.0

# 2. the pre-registered gate, per case, with a first-failure taxonomy
stage diagnose_capability python3 scripts/diagnose_capability.py --experiment "$EXP"
stage diagnose_capability_int4 python3 scripts/diagnose_capability.py --experiment "$EXP" \
    --variant int4 --out docs/audit_evidence/capability_diagnosis_int4.json --md docs/CAPABILITY_DIAGNOSIS_INT4.md

# 3. per-domain evaluation on the held-out test split (stratified sample; the
#    denominator is recorded inside the artefact)
stage evaluate_test python3 scripts/evaluate.py --experiment "$EXP" --checkpoint best \
    --dataset "$DATASET" --split test --backend numpy --max-samples "$SAMPLES" \
    --variants fp32,fp16,int8,int4
stage evaluate_challenge python3 scripts/evaluate.py --experiment "$EXP" --checkpoint best \
    --dataset "$DATASET" --split challenge --backend numpy --variants fp32
stage evaluate_independent python3 scripts/evaluate.py --experiment "$EXP" --checkpoint best \
    --dataset "$DATASET" --suite datasets/evaluation/independent_v1.jsonl --backend numpy \
    --variants fp32 --tag independent_independent_v1

# 4. capability probe against the same checkpoint the release ships
stage probe_capability python3 scripts/probe_capability.py --experiment "$EXP" --checkpoint best

# 5. quantization artefacts, packaging, size gate, standalone local model
stage quantize python3 scripts/quantize.py --experiment "$EXP" --checkpoint best --arch nano --out release
stage package_model python3 scripts/package_model.py --experiment "$EXP" --checkpoint best --dataset "$DATASET"
stage build_local_model python3 scripts/build_local_model.py --release release --out local_model
stage validate_local_model python3 scripts/validate_local_model.py --package local_model

# 6. artefact-level audits and the verdict
stage verify_matrix_checks python3 scripts/verify_matrix.py --checks all --dataset "$DATASET"
stage verify_matrix python3 scripts/verify_matrix.py
stage final_report python3 scripts/final_report.py

# 7. regression suite after every artefact above exists
stage pytest python3 -m pytest tests/ -q

echo
echo "evidence collection complete for $EXP"
