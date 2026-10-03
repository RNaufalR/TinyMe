#!/usr/bin/env bash
# TinyMe — fresh-clone verification (audit final gate E/F).
#
#   bash scripts/fresh_clone_check.sh [branch] [workdir]
#
# What it proves, in a directory that contains nothing but the pushed branch:
#   1. the documented dependency set is enough to import and test the project;
#   2. the data pipeline rebuilds the audited corpus deterministically and the
#      rebuilt manifest carries the same dataset fingerprint as the committed one;
#   3. the unit suite passes from the clone (no dependency on the author's working
#      directory, no `|| true`, no skipped-by-default gates);
#   4. the sandbox matrix passes from the clone;
#   5. the packaged release loads and generates with an empty PYTHONPATH.
#
# It refuses to claim success for anything it did not run: every stage prints its
# exit status and the script exits non-zero on the first failure.
set -uo pipefail

BRANCH="${1:-arena/01a0ff12-tinyme}"
WORKDIR="${2:-/tmp/tinyme-fresh-clone}"
REPO_URL="$(git -C "$(dirname "$0")/.." config --get remote.origin.url)"
DATASET="${DATASET:-dataset_v8}"
LOG="$WORKDIR/fresh_clone_check.log"

step() { printf '\n=== %s ===\n' "$1"; }
run() { echo "+ $*"; "$@"; local rc=$?; echo "  exit=$rc"; [ $rc -eq 0 ] || exit $rc; }

rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"

step "1/6 clone $BRANCH from $REPO_URL"
run git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR/repo"

cd "$WORKDIR/repo"
{
  set -x
  git rev-parse HEAD
  git status --porcelain
  python3 -V
} 2>&1 | tee -a "$LOG"

step "2/6 documented dependencies"
run python3 -c "import jax, numpy, safetensors, tokenizers; print('imports ok')"

step "3/6 rebuild the audited corpus from the committed pipeline"
run python3 scripts/prepare_data_v2.py --version "$DATASET" --generators v3 \
    --synthetic-profile v6 --scale 1.0 --scale-v3 1.0 --seq-len 512

step "4/6 dataset fingerprint matches the committed manifest"
# The rebuild is only meaningful if it reproduces the audited artefact: compare
# the freshly written manifest against the one committed in the clone.
run python3 - "$DATASET" <<'PY'
import json, subprocess, sys
from pathlib import Path

dataset = sys.argv[1]
local = json.loads(Path(f"datasets/versions/{dataset}/manifest.json").read_text())
head = subprocess.run(["git", "show", f"HEAD:datasets/versions/{dataset}/manifest.json"],
                      capture_output=True, text=True, check=True).stdout
committed = json.loads(head)
for key in ("splits", "train_tokens_active", "validation_tokens_active", "test_tokens_active",
            "challenge_tokens_active", "train_validation_test_contamination", "tokenizer_hash",
            "shards_fingerprint"):
    a, b = local.get(key), committed.get(key)
    status = "MATCH" if a == b else "DIFFERS"
    print(f"{key}: {status}")
    if a != b:
        print(f"  rebuilt={a}\n  committed={b}")
        raise SystemExit(1)
print("dataset fingerprint reproduced by a fresh clone")
PY

step "5/6 unit suite from the clone"
run python3 -m pytest tests/ -q

step "6/6 sandbox matrix and release smoke from the clone"
run python3 scripts/audit_sandbox.py
run python3 - <<'PY'
import subprocess, sys
proc = subprocess.run([sys.executable, "inference.py", "--prompt",
                       "<|system|>\nYou are TinyMe.\n<|user|>\nSay hello in one word.\n<|assistant|>\n",
                       "--max-new-tokens", "12"], cwd="release", capture_output=True, text=True,
                      env={"PATH": "/usr/bin:/bin", "PYTHONPATH": "", "HOME": "release"})
print("release exit:", proc.returncode)
print("release stdout:", (proc.stdout or "").strip()[-200:])
if proc.returncode != 0 or not (proc.stdout or "").strip():
    print(proc.stderr[-2000:])
    raise SystemExit(1)
PY

echo
echo "fresh-clone check complete — every stage above exited 0"
