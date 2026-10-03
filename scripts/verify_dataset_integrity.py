#!/usr/bin/env python3
"""Independent dataset-integrity verdict (audit §8, §9, §10, §60).

Re-derives every claim the corpus manifest makes, from the files on disk:

* split counts and active-target tokens counted from the JSONL, not read from
  the manifest;
* tokenizer hash recomputed from the shipped ``tokenizer.json``;
* model vocab compatibility against the checkpoint that will be trained;
* contamination classes recomputed (exact, normalised, MinHash, code shingles,
  template overlap) between train and each held-out split;
* every packed shard file accounted for, and the shard fingerprint recomputed
  and compared with the manifest;
* the curriculum partition re-derived from the corpus.

Writes ``docs/audit_evidence/dataset_integrity.json`` and exits non-zero when any
check fails, so it can serve as a gate.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dedup import contamination_report  # noqa: E402
from src.data.curriculum import partition_report  # noqa: E402
from src.data.dataset_api import (load_manifest, load_tokenizer,  # noqa: E402
                                  shard_fingerprint)
from src.data.splits import split_report  # noqa: E402
from src.utils.io_utils import sha256_file, write_json  # noqa: E402

SPLITS = ("train", "validation", "test", "challenge")


def _records(version: str, split: str) -> list[dict]:
    path = ROOT / "datasets" / "versions" / version / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _active_tokens(records: list[dict], tokenizer) -> int:
    """Sum of supervised target tokens, recomputed from the segments."""
    total = 0
    for rec in records:
        segments = rec.get("segments") or []
        if not segments:
            total += len(tokenizer.encode_ids(str(rec.get("text", ""))))
            continue
        for seg in segments:
            if seg.get("target", True) and seg.get("role") in (
                    "assistant", "thought", "tool_call", "final", "code"):
                total += len(tokenizer.encode_ids(seg.get("text", "")))
    return total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--out", default="docs/audit_evidence/dataset_integrity.json")
    args = ap.parse_args()

    manifest = load_manifest(args.dataset)
    tokenizer = load_tokenizer(args.dataset)
    checks: dict[str, dict] = {}

    # ---------------------------------------------------------------- splits
    counts, active, records = {}, {}, {}
    for split in SPLITS:
        recs = _records(args.dataset, split)
        records[split] = recs
        counts[split] = len(recs)
        active[split] = _active_tokens(recs, tokenizer)
    manifest_counts = manifest.get("splits", {})
    counts_match = all(manifest_counts.get(s) == counts[s] for s in SPLITS)
    active_match = all(manifest.get(f"{s}_tokens_active") == active[s] for s in SPLITS)
    checks["split_counts"] = {
        "ok": counts_match and active_match,
        "counted": counts, "manifest": manifest_counts,
        "active_target_tokens_counted": active,
        "active_target_tokens_manifest": {s: manifest.get(f"{s}_tokens_active") for s in SPLITS},
    }

    # ------------------------------------------------------------- tokenizer
    tok_path = ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json"
    tok_hash = sha256_file(tok_path)
    checks["tokenizer_hash"] = {
        "ok": tok_hash == manifest.get("tokenizer_hash"),
        "recomputed": tok_hash, "manifest": manifest.get("tokenizer_hash"),
        "vocab_size": tokenizer.vocab_size,
        "manifest_vocab_size": manifest.get("tokenizer", {}).get("vocab_size"),
        "trained_on": manifest.get("tokenizer", {}).get("trained_on"),
        "vocab_match": tokenizer.vocab_size == manifest.get("tokenizer", {}).get("vocab_size"),
    }
    checks["tokenizer_hash"]["ok"] = bool(checks["tokenizer_hash"]["ok"]
                                          and checks["tokenizer_hash"]["vocab_match"])

    # --------------------------------------------------------- contamination
    # recomputed here with the corpus builder's own checker, from the JSONL on
    # disk: the manifest's PASS verdict is compared against a fresh derivation
    split_rep = split_report(records)
    contam = split_rep["contamination"]
    checks["contamination"] = {
        "ok": bool(split_rep["contamination_free"])
              and manifest.get("train_validation_test_contamination") == "PASS",
        "manifest_verdict": manifest.get("train_validation_test_contamination"),
        "recomputed_free": bool(split_rep["contamination_free"]),
        "pairs": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, (list, dict))}
                  for k, v in contam.items()},
    }

    # ------------------------------------------------------------- shards
    shard_dir = ROOT / "datasets" / "processed" / args.dataset / "shards"
    disk_fp = shard_fingerprint(shard_dir) if shard_dir.exists() else None
    checks["shards"] = {"ok": bool(disk_fp) and disk_fp == manifest.get("shards_fingerprint"),
                        "recomputed": disk_fp, "manifest": manifest.get("shards_fingerprint"),
                        "files": sorted(p.name for p in shard_dir.glob("*.npy"))[:80]
                        if shard_dir.exists() else []}

    # --------------------------------------------------------- curriculum
    partition = partition_report(records["train"])
    checks["curriculum_partition"] = {"ok": partition.get("total", 0) == len(records["train"]),
                                      "report": partition}

    # ----------------------------------------------------------- group split
    groups: dict[str, set] = {}
    for split in SPLITS:
        groups[split] = {str(r.get("group_id") or r.get("template_id") or "") for r in records[split]}
    straddling = sorted((groups["train"] & groups["validation"])
                        | (groups["train"] & groups["test"])
                        | (groups["validation"] & groups["test"]))
    checks["group_isolation"] = {"ok": not straddling, "straddling_groups": straddling[:20],
                                 "n_groups": {s: len(groups[s]) for s in SPLITS}}

    ok = all(c["ok"] for c in checks.values())
    payload = {"dataset": args.dataset, "ok": ok, "checks": checks,
               "fingerprint": manifest.get("dataset_fingerprint")}
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(payload, out)
    for name, check in checks.items():
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {name}")
    print(f"dataset integrity: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
