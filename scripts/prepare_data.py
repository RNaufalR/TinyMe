#!/usr/bin/env python3
"""Prepare the TinyMe dataset (spec §4, §6, §7, §8, §9, §11, §13, §14, §15).

Usage:
    python scripts/prepare_data.py --version dataset_v1 --vocab-size 4096
    python scripts/prepare_data.py --no-github --no-hf --synthetic-only
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from data_sources import github_adapter, huggingface_adapter, stdlib_adapter, synthetic_adapter  # noqa: E402
from src.data import pipeline  # noqa: E402
from src.tokenizer.bpe import TinyMeTokenizer, train_tokenizer, tokenizer_report, write_tokenizer_report  # noqa: E402
from src.utils.io_utils import REPO_ROOT, Timer, human_bytes, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="TinnyMe data preparation pipeline")
    ap.add_argument("--version", default="dataset_v1")
    ap.add_argument("--vocab-size", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--synthetic-counts", type=int, default=120,
                    help="synthetic examples per generator")
    ap.add_argument("--no-hf", action="store_true")
    ap.add_argument("--no-github", action="store_true")
    ap.add_argument("--no-stdlib", action="store_true")
    ap.add_argument("--synthetic-only", action="store_true")
    ap.add_argument("--target-total", type=int, default=None)
    ap.add_argument("--min-quality", type=float, default=0.35)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log = logging.getLogger("prepare_data")
    t0 = time.time()

    all_records = []
    provenance: list[dict] = []

    if not args.synthetic_only and not args.no_hf:
        with Timer("huggingface ingestion"):
            recs, prov = huggingface_adapter.ingest(rows_per_dataset=20)
            all_records.extend(r.to_dict() for r in recs)
            provenance.extend(prov)

    if not args.synthetic_only and not args.no_github:
        with Timer("github ingestion"):
            recs, prov = github_adapter.ingest(files_per_repo=8, records_per_repo=8)
            all_records.extend(r.to_dict() for r in recs)
            provenance.extend(prov)

    if not args.synthetic_only and not args.no_stdlib:
        with Timer("python stdlib ingestion"):
            recs, prov = stdlib_adapter.ingest(max_per_module=8)
            all_records.extend(r.to_dict() for r in recs)
            provenance.extend(prov)

    with Timer("synthetic ingestion"):
        recs, prov = synthetic_adapter.ingest(
            seed=args.seed, counts={k: args.synthetic_counts for k in synthetic_adapter.GENERATORS})
        all_records.extend(r.to_dict() for r in recs)
        provenance.extend(prov)

    log.info("total raw records: %d", len(all_records))
    write_json(provenance, REPO_ROOT / "docs" / "DATA_PROVENANCE.json")

    # ---- tokenizer training on the pre-filter corpus (raw text only)
    texts = [r["text"] for r in all_records]
    log.info("training tokenizer (vocab_size=%d) on %d texts", args.vocab_size, len(texts))
    tokenizer = train_tokenizer(texts, vocab_size=args.vocab_size, min_frequency=2)
    tok_path = REPO_ROOT / "datasets" / "processed" / "tokenizer.json"
    tokenizer.save(tok_path)
    tok_bytes = tok_path.stat().st_size
    log.info("tokenizer saved: %s (%s)", tok_path, human_bytes(tok_bytes))

    # ---- held-out evaluation pool (disjoint from training by construction)
    heldout = synthetic_adapter.gen_heldout_eval(n_per_task=2, seed=args.seed + 1)
    log.info("held-out evaluation samples: %d", len(heldout))

    # ---- run the 14-stage pipeline
    with Timer("pipeline"):
        manifest = pipeline.run_pipeline(
            all_records, provenance, version=args.version, tokenizer=tokenizer,
            seq_len=args.seq_len, target_total=args.target_total,
            min_quality=args.min_quality, seed=args.seed,
            eval_extra=[r.to_dict() for r in heldout],
        )

    # ---- tokenizer report
    report = tokenizer_report(
        tokenizer,
        {"all": texts, "code": [r["text"] for r in all_records if r["category"] in
                                ("programming", "code_gen", "code_repair", "code_explain")],
         "math_logic": [r["text"] for r in all_records if r["category"] in ("math", "logic", "algorithm")]},
        file_size_bytes=tok_bytes,
    )
    write_json(report, REPO_ROOT / "docs" / "TOKENIZER_METRICS.json")
    write_tokenizer_report(report, REPO_ROOT / "docs" / "TOKENIZER_REPORT.md")

    # ---- data quality report
    from src.data.quality_report import write_quality_report
    write_quality_report(manifest, REPO_ROOT / "docs" / "DATA_QUALITY_REPORT.md")

    log.info("DONE in %.1fs: train=%d eval=%d", time.time() - t0,
             manifest["train_samples"], manifest["eval_samples"])
    print(f"\nDataset version : {args.version}")
    print(f"Train samples   : {manifest['train_samples']}")
    print(f"Eval samples    : {manifest['eval_samples']}")
    print(f"Tokenizer       : {tok_path} ({human_bytes(tok_bytes)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
