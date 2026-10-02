#!/usr/bin/env bash
# TinyMe — end-to-end audit pipeline (reproduces every published number).
#
#   bash scripts/run_audit_pipeline.sh EXP-004-TOOL-SFT-V2 [checkpoint]
#
# Stages (each writes the artefact the audit matrix cites):
#   1. data contract + unit tests
#   2. sandbox escape matrix            -> docs/audit_evidence/sandbox_escape_suite.out.txt
#   3. independent tool-use evaluation  -> experiments/<exp>/evaluation_tools_<ckpt>.{json,md}
#   4. per-domain evaluation (test)     -> experiments/<exp>/evaluation_test.json
#   5. per-domain evaluation (challenge)-> experiments/<exp>/evaluation_challenge.json
#   6. quantization variants            -> experiments/<exp>/evaluation_{test,challenge}.json (variants)
#   7. packaging + size gate + release  -> release/*
#   8. clean-environment release gate   -> tests/test_release_contract.py
#
# It never fabricates a result: a failing stage aborts the run with a non-zero exit.
set -euo pipefail
cd "$(dirname "$0")/.."

EXP="${1:?usage: run_audit_pipeline.sh <experiment> [checkpoint]}"
CKPT="${2:-best}"
DATASET="${DATASET:-dataset_v3}"

step() { printf '\n=== %s ===\n' "$1"; }

step "1/8 unit + data-contract tests"
python3 -m pytest tests/test_data_pipeline.py tests/test_dataset_api.py tests/test_split_contamination.py -q

step "2/8 sandbox escape matrix"
python3 scripts/audit_sandbox.py

step "3/8 independent tool-use evaluation"
python3 scripts/evaluate_tools.py --experiment "$EXP" --checkpoint "$CKPT" \
    --variants fp32,fp16,int8,int4 --backend numpy --max-new-tokens 96

step "4/8 per-domain evaluation (test split, full set)"
python3 scripts/evaluate.py --experiment "$EXP" --checkpoint "$CKPT" --dataset "$DATASET" \
    --split test --backend numpy --variants fp32,fp16,int8,int4

step "5/8 per-domain evaluation (challenge split, full set)"
python3 scripts/evaluate.py --experiment "$EXP" --checkpoint "$CKPT" --dataset "$DATASET" \
    --split challenge --backend numpy --variants fp32

step "6/8 quantization artefact sizes"
python3 scripts/quantize.py --experiment "$EXP" --checkpoint "$CKPT" --arch nano --out release/

step "7/8 packaging + size gate (release/)"
python3 scripts/package_model.py --experiment "$EXP" --checkpoint "$CKPT" --dataset "$DATASET"

step "8/8 release contract + clean-environment gate"
python3 -m pytest tests/test_package_size.py tests/test_release_contract.py -q

echo
echo "pipeline complete — artefacts:"
ls -la release/ | head -20
