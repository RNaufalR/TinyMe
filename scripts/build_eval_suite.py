#!/usr/bin/env python3
"""Build an independent evaluation suite (audit §36/§37).

Two kinds of held-out evidence are needed and they are not the same thing:

* **held-out instances of a trained family** — the ``validation``/``test`` splits
  of the corpus (every multi-phrasing family is represented there; see
  ``python scripts/verify_matrix.py --checks family_coverage``);
* **instances generated after the training corpus was frozen** — this suite.

The suite is generated with a fresh seed that appears in no training corpus and
written to ``datasets/evaluation/``.  Independence is then *measured*, not
assumed: every instance whose exact / normalised / MinHash / code-shingle
signature already exists in the training split is removed, and the drop counts
plus the final overlap numbers are stored in the suite manifest.  Template-level
overlap is expected and reported separately: template-level generalisation is
measured on the corpus ``challenge`` split.

Usage::

    python scripts/build_eval_suite.py --seed 777001 --version independent_v1
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_sources import synthetic_v2, synthetic_v3  # noqa: E402
from src.data.dedup import contamination_report  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402

log = logging.getLogger("tinyme.eval.suite")

SUITE_COUNTS_V3 = {"copy_span": 160, "no_tool": 260, "tool_use": 320, "algorithm": 90}
SUITE_COUNTS_V2 = {"arithmetic": 60, "logic": 40, "math_word": 40, "sequence": 30,
                   "code_gen": 40, "code_repair": 30, "code_explain": 24,
                   "instruction": 40}


def build(seed: int) -> tuple[list[dict], dict]:
    v3_records, v3_prov = synthetic_v3.generate_corpus_v3(seed=seed, counts=SUITE_COUNTS_V3,
                                                          scale=1.0, profile="v6")
    v2_records, v2_prov = synthetic_v2.generate_corpus(seed=seed + 1, counts=SUITE_COUNTS_V2,
                                                       scale=1.0)
    challenge = list(synthetic_v2.gen_challenge(__import__("random").Random(seed + 2), 30))
    challenge += list(synthetic_v3.gen_tool_challenge(__import__("random").Random(seed + 3), 20))
    records = [r.to_dict() for r in list(v3_records) + list(v2_records) + challenge]
    prov = {"v3": v3_prov, "v2": v2_prov, "seed": seed,
            "challenge": {"v2_templates": 30, "v3_templates": 20}}
    return records, prov


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=777001,
                    help="fresh seed; must not equal any training seed (20261003 etc.)")
    ap.add_argument("--version", default="independent_v1")
    ap.add_argument("--train-split", default=None,
                    help="training JSONL to prove independence against (default: newest dataset_v*)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")

    t0 = time.time()
    versions = sorted([d for d in (ROOT / "datasets" / "versions").glob("dataset_v*")
                       if d.name[9:].isdigit() and (d / "manifest.json").exists()],
                      key=lambda d: int(d.name[9:]))
    train_split = Path(args.train_split) if args.train_split else (versions[-1] / "train.jsonl")
    records, prov = build(args.seed)

    train = [json.loads(line) for line in train_split.open(encoding="utf-8") if line.strip()]

    # ------------------------------------------------------------------ filter
    # A fresh seed still regenerates some strings (small arithmetic, short
    # algorithm traces).  Anything whose signature already exists in training is
    # not independent evidence and is removed, with the drop counts recorded.
    from src.data.dedup import MinHashLSH, code_tokens, exact_hash, normalized_hash, shingles

    train_exact = {exact_hash(r["text"]) for r in train}
    train_norm = {normalized_hash(r["text"]) for r in train}
    text_index = MinHashLSH(num_perm=64, bands=16, threshold=0.85, seed=98)
    code_index = MinHashLSH(num_perm=64, bands=16, threshold=0.90, seed=99)
    for r in train:
        sh = shingles(r["text"])
        if sh:
            text_index.add(r["record_id"], sh)
        ct = code_tokens(r["text"])
        if ct:
            code_index.add(r["record_id"], ct)
    kept, dropped = [], {"exact": 0, "normalized": 0, "minhash": 0, "code": 0}
    for rec in records:
        if exact_hash(rec["text"]) in train_exact:
            dropped["exact"] += 1
            continue
        if normalized_hash(rec["text"]) in train_norm:
            dropped["normalized"] += 1
            continue
        sh = shingles(rec["text"])
        if sh and text_index.is_duplicate(sh) is not None:
            dropped["minhash"] += 1
            continue
        ct = code_tokens(rec["text"])
        if ct and code_index.is_duplicate(ct) is not None:
            dropped["code"] += 1
            continue
        kept.append(rec)
    raw_overlap = dict(dropped)
    records = kept
    report = contamination_report(train, records)

    out_dir = ROOT / "datasets" / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    suite_path = out_dir / f"{args.version}.jsonl"
    with suite_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")

    by_category = Counter(r.get("category", "unknown") for r in records)
    by_family = Counter(r.get("family") or r.get("template_id") for r in records)
    manifest = {
        "version": args.version,
        "seed": args.seed,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "records": len(records),
        "path": str(suite_path.relative_to(ROOT)),
        "train_split_compared": str(train_split.relative_to(ROOT)),
        "categories": dict(by_category),
        "families": {k: v for k, v in sorted(by_family.items())},
        "contamination_vs_train": {k: v for k, v in report.items() if not isinstance(v, dict)},
        "dropped_for_independence": raw_overlap,
        "raw_generated": sum(raw_overlap.values()) + len(records),
        "independence": {
            "exact_overlap": report["exact_overlap"],
            "normalized_overlap": report["normalized_overlap"],
            "minhash_overlap": report["minhash_overlap"],
            "code_shingle_overlap": report["code_shingle_overlap"],
            "template_overlap_expected": report["template_overlap"],
            "template_level_evidence": "corpus challenge split (held-out templates)",
            "note": ("template/group overlap is by construction: the suite re-uses the "
                     "generator templates with a fresh seed so that *instances* are new."),
        },
        "provenance": prov,
        "seconds": round(time.time() - t0, 1),
    }
    write_json(manifest, out_dir / f"{args.version}.manifest.json")
    log.info("suite %s: %d records, exact=%d normalized=%d minhash=%d code=%d",
             args.version, len(records), report["exact_overlap"], report["normalized_overlap"],
             report["minhash_overlap"], report["code_shingle_overlap"])
    print(json.dumps({k: manifest[k] for k in ("version", "records", "categories",
                                               "independence", "dropped_for_independence")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
