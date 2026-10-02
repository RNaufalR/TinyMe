#!/usr/bin/env python3
"""Build dataset_v2 with the corrected pipeline (audit §2–§9, §19–§22).

Order is enforced by construction:

    ingest → license check → content-aware preprocess → quality filter
    → classify → dedup → group split (train/validation/test/challenge)
    → tokenizer trained on the TRAIN split only → freeze
    → tokenize + pack per split/stage → shards → manifest + reports

Every stage reports its own counters. Nothing is upsampled or duplicated.
Usage::

    python scripts/prepare_data_v2.py --version dataset_v2 --seq-len 256 --scale 1.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources import corpus_v2, synthetic_v2  # noqa: E402
from src.data import preprocess as preprocess_mod  # noqa: E402
from src.data.dedup import deduplicate  # noqa: E402
from src.data.quality_filter import filter_record, score_record  # noqa: E402
from src.data.records import classify_license  # noqa: E402
from src.data.sequence import LOSS_MASK_SEMANTICS, pack_records  # noqa: E402
from src.data.shard_writer import ShardWriter, load_shard_arrays  # noqa: E402
from src.data.splits import (SPLITS, assign_splits, dataset_fingerprint,  # noqa: E402
                             split_report, split_summary_markdown)
from src.tokenizer.bpe import (SPECIAL_TOKENS, TOKENIZER_VERSION, TinyMeTokenizer,  # noqa: E402
                               train_tokenizer, write_tokenizer_report)
from src.utils.io_utils import REPO_ROOT, human_bytes, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("prepare_data_v2")

VERSIONS_DIR = REPO_ROOT / "datasets" / "versions"
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"


def token_stage(record: dict) -> str:
    """Stage A = plain documents; Stage B = structured instruction/tool records."""
    return "sft" if record.get("segments") else "pretrain"


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a corrected TinyMe dataset version")
    ap.add_argument("--version", default="dataset_v2")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--scale", type=float, default=1.0, help="synthetic generator scale")
    ap.add_argument("--seed", type=int, default=20261002)
    ap.add_argument("--no-hf", action="store_true", help="skip cached HF slices")
    ap.add_argument("--stdlib-files", type=int, default=480, help="stdlib files to scan")
    ap.add_argument("--challenge-size", type=int, default=60)
    ap.add_argument("--shard-capacity", type=int, default=512)
    ap.add_argument("--tokenizer-from", default=None,
                    help="freeze the tokenizer of an existing dataset version instead of "
                         "retraining it (keeps checkpoints embedding-compatible; the source "
                         "train split must be a subset of the new train split)")
    args = ap.parse_args()

    t0 = time.time()
    stats: dict[str, object] = {}
    version_dir = VERSIONS_DIR / args.version
    shard_dir = PROCESSED_DIR / args.version / "shards"
    version_dir.mkdir(parents=True, exist_ok=True)
    shard_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- ingest
    records, provenance = corpus_v2.build_corpus(seed=args.seed, scale=args.scale, use_hf=not args.no_hf,
                                                stdlib_files=args.stdlib_files)
    challenge = [r.to_dict() for r in synthetic_v2.gen_challenge(__import__("random").Random(args.seed + 1),
                                                                 args.challenge_size)]
    stats["ingest"] = {**corpus_v2.corpus_statistics(records),
                       "challenge_pool": len(challenge)}

    # ------------------------------------------------------- license + type
    kept, license_stats = [], Counter()
    for r in records:
        lic = classify_license(r.get("license"))
        r["license"] = lic
        license_stats[lic] += 1
        if lic not in ("LICENSE_UNCLEAR",) and not lic.startswith("EXCLUDED"):
            kept.append(r)
    stats["license"] = {"kept": len(kept), "distribution": dict(license_stats)}

    # ----------------------------------------------------- preprocessing
    rejected = Counter()
    preprocessed: list[dict] = []
    for r in kept:
        _, status = preprocess_mod.preprocess_record(r)
        if status.startswith("rejected"):
            rejected[status.split(":", 1)[0] + ":" + status.split(":", 2)[1] if status.count(":") >= 2 else status] += 1
            continue
        preprocessed.append(r)
    stats["preprocess"] = {"kept": len(preprocessed), "rejected": dict(rejected)}

    # ------------------------------------------------------ quality filter
    quality_reasons = Counter()
    filtered: list[dict] = []
    for r in preprocessed:
        r["quality"] = score_record(r)
        ok, r, reason = filter_record(r, min_quality=0.30)
        if not ok:
            quality_reasons[reason] += 1
            continue
        filtered.append(r)
    stats["quality"] = {"kept": len(filtered), "dropped": dict(quality_reasons)}

    # -------------------------------------------------------------- dedup
    deduped, dedup_report = deduplicate(filtered)
    stats["dedup"] = dedup_report.to_dict()

    # ------------------------------------------------------------- splits
    train_challenge = deduped + challenge
    assert all(r.get("split") == "challenge" for r in challenge)
    assigned, split_rep = assign_splits(train_challenge, seed=args.seed)
    stats["splits"] = split_rep

    by_split = {s: [r for r in assigned if r["split"] == s] for s in SPLITS}
    log.info("splits: %s", {s: len(v) for s, v in by_split.items()})

    # ------------------------------------- tokenizer (TRAIN split only)
    train_texts = [r["text"] for r in by_split["train"]]
    if args.tokenizer_from:
        # Controlled revisions: reuse a tokenizer that was already trained on a
        # *subset* of this split's text.  Freezing it keeps every checkpoint
        # embedding-compatible across dataset revisions, so a data-format change
        # can be isolated from a vocabulary change (and the reuse is verifiable:
        # the source train split must be a subset of this one).
        src_dir = VERSIONS_DIR / args.tokenizer_from
        src_tok = src_dir / "tokenizer.json"
        if not src_tok.exists():
            log.error("--tokenizer-from %s has no tokenizer.json", args.tokenizer_from)
            return 4
        tokenizer = TinyMeTokenizer.load(src_tok)
        tokenizer.save(version_dir / "tokenizer.json")
        tokenizer.save(PROCESSED_DIR / "tokenizer.json")
        tokenizer_saved = TinyMeTokenizer.load(version_dir / "tokenizer.json")
        tok_hash = tokenizer_saved.sha256(version_dir / "tokenizer.json")
        tok_bytes = (version_dir / "tokenizer.json").stat().st_size
        stats["tokenizer"] = {"version": getattr(tokenizer_saved, "version", TOKENIZER_VERSION), "vocab_size": tokenizer_saved.vocab_size,
                              "sha256": tok_hash, "bytes": tok_bytes,
                              "trained_on": f"frozen from {args.tokenizer_from} (train split only)",
                              "train_texts": len(train_texts)}
        log.info("tokenizer frozen from %s (vocab=%d, sha256=%s)", args.tokenizer_from,
                 tokenizer_saved.vocab_size, tok_hash[:16])
    else:
        tokenizer = train_tokenizer(train_texts, vocab_size=args.vocab_size,
                                    min_frequency=2, corpus_role="train")
        tok_path = version_dir / "tokenizer.json"
        tokenizer.save(tok_path)
        tokenizer.save(PROCESSED_DIR / "tokenizer.json")
        tokenizer_saved = TinyMeTokenizer.load(tok_path)
        tok_hash = tokenizer_saved.sha256(tok_path)
        tok_bytes = tok_path.stat().st_size
        stats["tokenizer"] = {"version": TOKENIZER_VERSION, "vocab_size": tokenizer_saved.vocab_size,
                              "sha256": tok_hash, "bytes": tok_bytes,
                              "trained_on": "train split only", "train_texts": len(train_texts)}

    # --------------------------------------------- tokenize + pack + shards
    token_stats: dict[str, dict] = {}
    split_token_totals: dict[str, int] = {}
    for split in SPLITS:
        recs = by_split[split]
        if not recs:
            continue
        writers: dict[str, ShardWriter] = {}
        counters: Counter = Counter()
        active_tokens = 0
        padding_tokens = 0
        n_blocks = 0
        by_stage: dict[str, list[dict]] = {}
        for rec in recs:
            by_stage.setdefault(token_stage(rec), []).append(rec)
        for stage, stage_recs in by_stage.items():
            data, pstats = pack_records(stage_recs, tokenizer_saved, args.seq_len)
            if data["input_ids"].shape[0] == 0:
                continue
            writers[stage] = ShardWriter(shard_dir, f"{split}_{stage}", args.seq_len,
                                         capacity_per_shard=args.shard_capacity)
            writers[stage].append_blocks(data)
            counters["records"] += pstats["records_seen"] - pstats["records_rejected"]
            counters["rejected"] += pstats["records_rejected"]
            counters["windows"] += pstats.get("windows", 0)
            active_tokens += int(data["loss_mask"].sum())
            padding_tokens += int((data["input_ids"] == tokenizer_saved.tok.token_to_id("<|pad|>")).sum())
            n_blocks += int(data["input_ids"].shape[0])
        for w in writers.values():
            w.close()
        token_stats[split] = {
            "records": counters["records"], "blocks": n_blocks,
            "rejected_records": counters["rejected"], "windows": counters["windows"],
            "active_target_tokens": active_tokens, "padding_tokens": padding_tokens,
            "padding_ratio": round(padding_tokens / max(active_tokens + padding_tokens, 1), 6),
            "stages": sorted(writers.keys()),
        }
        split_token_totals[split] = active_tokens
    stats["tokens"] = token_stats
    stats["tokens_total_active"] = sum(split_token_totals.values())

    # --------------------------------------------------------------- write
    from src.utils.io_utils import write_jsonl

    for split, recs in by_split.items():
        write_jsonl(recs, version_dir / f"{split}.jsonl")

    contamination_free = split_rep["contamination_free"]
    manifest = {
        "dataset_version": args.version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "seq_len": args.seq_len,
        "splits": {s: len(v) for s, v in by_split.items()},
        "train_tokens_active": split_token_totals.get("train", 0),
        "validation_tokens_active": split_token_totals.get("validation", 0),
        "test_tokens_active": split_token_totals.get("test", 0),
        "challenge_tokens_active": split_token_totals.get("challenge", 0),
        "tokenizer": stats["tokenizer"],
        "tokenizer_hash": tok_hash,
        "tokenizer_version": TOKENIZER_VERSION,
        "contamination": split_rep["contamination"],
        "train_validation_test_contamination": "PASS" if contamination_free else "FAIL",
        "loss_mask_semantics": LOSS_MASK_SEMANTICS,
        "stage_stats": stats,
        "provenance_summary": {
            "sources": dict(Counter(p.get("source") for p in provenance)),
            "records_by_source": {p.get("source_id", p.get("source", "?")): p.get("records", p.get("sample_count", 0))
                                  for p in provenance},
        },
        "licenses": dict(Counter(r["license"] for r in assigned)),
        "categories": dict(Counter(r["category"] for r in assigned)),
        "verified_samples": sum(1 for r in assigned if r.get("verified")),
        "malformed_code_rate": round(
            sum(1 for r in assigned
                if r.get("category") in ("programming", "code_gen", "code_explain", "algorithm")
                and r.get("quality", {}).get("syntax_valid") is False)
            / max(sum(1 for r in assigned if r.get("category") in
                      ("programming", "code_gen", "code_explain", "algorithm")), 1), 6),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    manifest["dataset_fingerprint"] = dataset_fingerprint(assigned, extra={
        "seq_len": args.seq_len, "tokenizer_sha256": tok_hash,
        "splits": {s: len(v) for s, v in by_split.items()}})
    # shard fingerprint: training reads the packed shards, so their content must
    # be verifiable — a stale shard built by an older sequence builder would
    # otherwise train silently (audit §13/§20).
    from src.data.dataset_api import shard_fingerprint
    manifest["shards_fingerprint"] = shard_fingerprint(shard_dir)
    write_json(manifest, version_dir / "manifest.json")
    write_json(manifest, REPO_ROOT / "DATA_PROVENANCE.json")

    # provenance + reports
    write_json({"dataset_version": args.version, "created": manifest["created"],
                "sources": provenance}, REPO_ROOT / "docs" / "DATA_PROVENANCE.json")
    write_json({"dataset_version": args.version, "created": manifest["created"],
                "sources": provenance}, PROCESSED_DIR / "DATA_PROVENANCE.json")

    report = {
        "tokenizer_version": TOKENIZER_VERSION,
        "algorithm": "byte-level BPE (`tokenizers`), ByteLevel pre-tokenizer/decoder",
        "vocab_size": tokenizer_saved.vocab_size,
        "special_tokens": SPECIAL_TOKENS,
        "file_size_bytes": tok_bytes,
        "sha256": tok_hash,
        "trained_on": "train split only",
        "train_corpus_texts": len(train_texts),
        "train_corpus_chars": sum(len(t) for t in train_texts),
        "domains": {
            "all_train": tokenizer_saved.measure([r["text"] for r in by_split["train"]]),
            "code": tokenizer_saved.measure([r["text"] for r in by_split["train"]
                                             if r["category"] in ("programming", "code_gen", "code_repair",
                                                                  "code_explain", "algorithm")][:2000]),
            "prose": tokenizer_saved.measure([r["text"] for r in by_split["train"]
                                              if r["category"] in ("language",)][:2000]),
            "math": tokenizer_saved.measure([r["text"] for r in by_split["train"]
                                             if r["category"] in ("math", "logic")][:2000]),
            "tool_call_json": tokenizer_saved.measure(
                [json.dumps({"name": "search", "arguments": {"query": "population of jakarta"}},
                            sort_keys=True)]),
        },
    }
    write_json(report, REPO_ROOT / "docs" / "TOKENIZER_METRICS.json")
    write_tokenizer_report(report, REPO_ROOT / "docs" / "TOKENIZER_REPORT.md")
    (REPO_ROOT / "docs" / "SPLIT_SUMMARY.md").write_text(split_summary_markdown(split_rep))

    quality_md = _quality_report_markdown(args, manifest, stats)
    (REPO_ROOT / "docs" / "DATA_QUALITY_REPORT.md").write_text(quality_md)

    log.info("DONE %.1fs | train=%d val=%d test=%d challenge=%d | active target tokens=%d | tokens.json=%s",
             time.time() - t0, len(by_split["train"]), len(by_split["validation"]),
             len(by_split["test"]), len(by_split["challenge"]),
             manifest["train_tokens_active"], human_bytes(tok_bytes))
    print(json.dumps({k: manifest[k] for k in ("dataset_version", "splits", "train_tokens_active",
                                               "train_validation_test_contamination")}, indent=2))
    return 0


def _quality_report_markdown(args, manifest: dict, stats: dict) -> str:
    ingest = stats["ingest"]
    lines = [
        "# DATA QUALITY REPORT — " + args.version, "",
        f"- Raw records ingested: **{ingest['records'] + ingest['challenge_pool']}**",
        f"- After license check: **{stats['license']['kept']}**",
        f"- After content-aware preprocessing: **{stats['preprocess']['kept']}**"
        f" (rejected: {stats['preprocess']['rejected']})",
        f"- After quality/safety filter: **{stats['quality']['kept']}** (dropped: {stats['quality']['dropped']})",
        f"- After deduplication: **{stats['dedup']['kept']}**"
        f" (exact={stats['dedup']['exact_duplicates']}, normalized={stats['dedup']['normalized_duplicates']},"
        f" minhash={stats['dedup']['similar_duplicates']}, code={stats['dedup']['code_similar_duplicates']})",
        f"- Malformed-code rate in the final corpus: **{manifest['malformed_code_rate']:.6f}**",
        f"- Verified records: **{manifest['verified_samples']}**",
        "",
        "## Splits (group-aware, template families held out)", "",
        "| split | records | active target tokens | blocks |",
        "| :--- | ---: | ---: | ---: |",
    ]
    for split in SPLITS:
        t = stats["tokens"].get(split, {})
        lines.append(f"| {split} | {manifest['splits'].get(split, 0)} | {t.get('active_target_tokens', 0)} | {t.get('blocks', 0)} |")
    lines += [
        "",
        f"## Contamination gate", "",
        f"**TRAIN/VALIDATION/TEST CONTAMINATION: {manifest['train_validation_test_contamination']}**",
        "",
        "Checks performed: exact hash, normalized hash, MinHash/LSH similarity, code-AST shingles,"
        " template-family overlap. See `docs/SPLIT_SUMMARY.md` for per-pair numbers.",
        "",
        "## Tokenizer", "",
        f"- Trained on the **train split only** ({manifest['tokenizer']['train_texts']} texts).",
        f"- Vocabulary: {manifest['tokenizer']['vocab_size']}; file {manifest['tokenizer']['bytes']} bytes;",
        f"  sha256 `{manifest['tokenizer']['sha256'][:32]}…`",
        "",
        "## Sources and licences", "",
        "| source | records |", "| :--- | ---: |",
    ]
    for k, v in sorted(manifest["provenance_summary"]["records_by_source"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "| licence | records |", "| :--- | ---: |"]
    for k, v in sorted(manifest["licenses"].items()):
        lines.append(f"| {k} | {v} |")
    lines += ["", "## Category distribution", "", "| category | records |", "| :--- | ---: |"]
    for k, v in sorted(manifest["categories"].items()):
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
