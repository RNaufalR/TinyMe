"""The 14-stage data pipeline (spec §4).

SOURCE -> INGESTION -> LICENSE CHECK -> CONTENT EXTRACTION -> NORMALIZATION
-> QUALITY FILTER -> SAFETY FILTER -> LANGUAGE FILTER -> CODE/TEXT CLASSIFICATION
-> DEDUPLICATION -> CONTAMINATION CHECK -> DATA MIXING -> TOKENIZATION
-> TRAINING SHARDS -> MANIFEST

Each stage is a pure function over lists of record dicts, so the pipeline is
fully testable and resumable. A stage report is emitted for every stage.
"""
from __future__ import annotations

import logging
import random
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..utils.io_utils import REPO_ROOT, Timer, write_json, write_jsonl
from .dedup import contamination_report, deduplicate
from .quality_filter import filter_record, score_record
from .records import TrainingRecord, classify_license, validate_record

logger = logging.getLogger("tinyme.pipeline")

PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
VERSIONS_DIR = REPO_ROOT / "datasets" / "versions"

# Target mixture (fractions) for the final training corpus.
DEFAULT_MIX = {
    "language": 0.16,
    "logic": 0.10,
    "math": 0.14,
    "algorithm": 0.12,
    "programming": 0.16,
    "code_repair": 0.08,
    "code_gen": 0.12,
    "code_explain": 0.08,
    "instruction": 0.04,
}


# ------------------------------------------------------------------ stage 1
def stage_ingest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept, rejected = [], []
    for r in records:
        ok, reason = validate_record(r)
        (kept if ok else rejected).append(r)
    logger.info("[1/14] ingestion: %d valid, %d rejected", len(kept), len(rejected))
    return kept


# ------------------------------------------------------------------ stage 2
def stage_license_check(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from .records import ALLOWED_LICENSES

    kept, stats = [], Counter()
    for r in records:
        lic = classify_license(r.get("license"))
        r["license"] = lic
        if lic in ALLOWED_LICENSES:
            kept.append(r)
            stats[lic] += 1
        elif lic == "LICENSE_UNCLEAR":
            stats["LICENSE_UNCLEAR"] += 1
        else:
            stats[f"EXCLUDED:{lic}"] += 1
    logger.info("[2/14] license check: kept %d, %s", len(kept), dict(stats))
    return kept, {"kept": len(kept), "license_distribution": dict(stats)}


# ------------------------------------------------------------------ stage 3
def stage_extract(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Normalize unicode, strip control characters, collapse whitespace runs."""
    for r in records:
        t = r.get("text", "")
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = "".join(ch for ch in t if ch == "\n" or ch == "\t" or ord(ch) >= 32)
        t = re.sub(r"[ \t]{3,}", "  ", t)
        t = re.sub(r"\n{4,}", "\n\n\n", t)
        r["text"] = t.strip()
    logger.info("[3/14] content extraction/normalization applied to %d records", len(records))
    return records


# ------------------------------------------------------------------ stage 4
def stage_quality_filter(records: list[dict[str, Any]], source_quality: dict[str, float] | None = None,
                         min_quality: float = 0.35) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source_quality = source_quality or {"huggingface": 1.0, "github": 0.9, "python_stdlib": 0.95, "synthetic": 1.0}
    kept, reasons = [], Counter()
    for r in records:
        q = score_record(r, source_quality=source_quality.get(r.get("source", "synthetic"), 0.8))
        r["quality"] = q
        ok, r, reason = filter_record(r, min_quality=min_quality)
        if ok:
            kept.append(r)
        else:
            reasons[reason] += 1
    logger.info("[4/14] quality+safety+language filter: kept %d, dropped %s", len(kept), dict(reasons))
    return kept, {"kept": len(kept), "dropped_by_reason": dict(reasons)}


# ------------------------------------------------------------------ stage 5
def stage_classify(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Heuristic code/text classification for records lacking a category."""
    from .quality_filter import CODE_HINTS

    for r in records:
        cat = r.get("category")
        if cat and cat != "unknown":
            continue
        text = r.get("text", "")
        codeish = sum(1 for ln in text.split("\n") if any(h in ln for h in CODE_HINTS))
        r["category"] = "programming" if codeish >= 2 else "language"
    logger.info("[5/14] code/text classification: %s", dict(Counter(r["category"] for r in records)))
    return records


# ------------------------------------------------------------------ stage 6
def stage_dedup(records: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kept, report = deduplicate(records)
    logger.info("[6/14] dedup: kept %d of %d (%s)", len(kept), len(records), report.to_dict())
    return kept, report.to_dict()


# ------------------------------------------------------------------ stage 7
def stage_contamination(records: list[dict[str, Any]], eval_ratio: float = 0.06,
                        seed: int = 1234,
                        extra_eval: list[dict[str, Any]] | None = None) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Split train/eval *before* mixing so no eval sample is duplicated into train.

    Contaminated evaluation samples are REMOVED (not silently kept) so the
    evaluation set is provably disjoint from the training set. Additional
    held-out evaluation samples (unseen problem templates) are appended.
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    train, eval_ = [], []
    for cat, items in by_cat.items():
        rng.shuffle(items)
        n_eval = max(1, int(len(items) * eval_ratio)) if len(items) >= 8 else 0
        eval_.extend(items[:n_eval])
        train.extend(items[n_eval:])

    report = contamination_report(train, eval_)
    if not report["contamination_free"]:
        contaminated = set(report["contaminated_record_ids"])
        before = len(eval_)
        eval_ = [r for r in eval_ if r.get("record_id") not in contaminated]
        report["eval_samples_removed_for_contamination"] = before - len(eval_)
        report = contamination_report(train, eval_)
        report["eval_samples_removed_for_contamination"] = before - len(eval_)

    if extra_eval:
        # Held-out evaluation samples are generated from a DISJOINT task pool.
        held = contamination_report(train, extra_eval)
        kept_extra = [r for r in extra_eval if r.get("record_id") not in set(held["contaminated_record_ids"])]
        eval_.extend(kept_extra)
        report["heldout_eval_samples"] = len(kept_extra)
        report["heldout_contamination_removed"] = len(extra_eval) - len(kept_extra)
        report = contamination_report(train, eval_)
        report["eval_samples_removed_for_contamination"] = report.get("eval_samples_removed_for_contamination", 0)
        report["heldout_eval_samples"] = len(kept_extra)

    report["train_samples"] = len(train)
    report["eval_samples"] = len(eval_)
    logger.info("[7/14] contamination check: %s", report)
    return train, eval_, report


# ------------------------------------------------------------------ stage 8
def stage_mix(records: list[dict[str, Any]], mix: dict[str, float] | None = None,
              target_total: int | None = None, seed: int = 1234) -> list[dict[str, Any]]:
    mix = mix or DEFAULT_MIX
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for r in records:
        by_cat.setdefault(r["category"], []).append(r)
    for items in by_cat.values():
        rng.shuffle(items)

    total = target_total or len(records)
    out: list[dict[str, Any]] = []
    for cat, frac in mix.items():
        want = int(round(total * frac))
        have = by_cat.get(cat, [])
        if want <= len(have):
            out.extend(have[:want])
        else:
            # Upsample with repetition but keep it bounded and reported.
            if have:
                reps = (want + len(have) - 1) // len(have)
                out.extend((have * reps)[:want])
            by_cat[cat] = []
    rng.shuffle(out)
    logger.info("[8/14] data mixing: %d samples (%s)", len(out), dict(Counter(r["category"] for r in out)))
    return out


# ------------------------------------------------------------------ stage 9
def stage_tokenize(records: list[dict[str, Any]], tokenizer: Any | None = None,
                   seq_len: int = 256) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Attach token ids/lengths when a tokenizer is supplied."""
    if tokenizer is None:
        lengths = [len(r["text"]) for r in records]
        return records, {"tokenizer": None, "avg_chars": sum(lengths) / max(len(lengths), 1)}
    encode = tokenizer.encode if hasattr(tokenizer, "encode") else tokenizer
    total_tokens, kept = 0, []
    for r in records:
        try:
            ids = encode(r["text"]).ids if hasattr(encode(r["text"]), "ids") else list(encode(r["text"]))
        except Exception:
            continue
        r["token_ids"] = [int(i) for i in ids][:seq_len * 4]
        r["n_tokens"] = len(r["token_ids"])
        total_tokens += r["n_tokens"]
        kept.append(r)
    logger.info("[9/14] tokenization: %d samples, %d tokens", len(kept), total_tokens)
    return kept, {"tokenizer": getattr(tokenizer, "name", "custom"), "total_tokens": total_tokens,
                  "avg_tokens_per_sample": total_tokens / max(len(kept), 1)}


# ----------------------------------------------------------------- stage 10
def stage_shards(records: list[dict[str, Any]], out_dir: Path, shard_size: int = 2048,
                 seq_len: int = 256) -> list[Path]:
    """Write fixed-length token shards as uint16 memmap-ready .npy files."""
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    buf: list[int] = []
    shard_idx = 0
    bos = 1
    eos = 2
    for r in records:
        ids = r.get("token_ids")
        if ids is None:
            continue
        buf.extend([bos] + list(ids) + [eos])
        while len(buf) >= seq_len:
            chunk, buf = buf[:seq_len], buf[seq_len:]
            arr = np.array(chunk, dtype=np.uint16)
            if len(arr) == seq_len:
                if not paths or paths[-1].stat().st_size >= shard_size * seq_len * 2:
                    path = out_dir / f"shard_{shard_idx:04d}.npy"
                    np.save(path, np.empty((0, seq_len), dtype=np.uint16))
                    shard_idx += 1
                    paths.append(path)
                existing = np.load(paths[-1], allow_pickle=False)
                np.save(paths[-1], np.vstack([existing, arr[None, :]]))
    # Drop empty trailing shards.
    paths = [p for p in paths if np.load(p, allow_pickle=False).shape[0] > 0]
    logger.info("[10/14] sharding: %d shards in %s", len(paths), out_dir)
    return paths


# ---------------------------------------------------------------- stage 11
def stage_manifest(version: str, records: list[dict[str, Any]], eval_records: list[dict[str, Any]],
                   provenance: list[dict[str, Any]], stats: dict[str, Any],
                   tokenizer_info: dict[str, Any]) -> dict[str, Any]:
    manifest = {
        "dataset_version": version,
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "train_samples": len(records),
        "eval_samples": len(eval_records),
        "categories": dict(Counter(r["category"] for r in records)),
        "sources": dict(Counter(r["source"] for r in records)),
        "licenses": dict(Counter(r["license"] for r in records)),
        "verified_samples": sum(1 for r in records if r.get("verified")),
        "tokenizer": tokenizer_info,
        "stage_stats": stats,
        "provenance": provenance,
    }
    logger.info("[11/14] manifest written for %s", version)
    return manifest


# ---------------------------------------------------------------- the runner
def run_pipeline(records: list[dict[str, Any]], provenance: list[dict[str, Any]],
                 version: str = "dataset_v1", tokenizer: Any | None = None,
                 seq_len: int = 256, mix: dict[str, float] | None = None,
                 target_total: int | None = None, min_quality: float = 0.35,
                 seed: int = 1234, write_shards: bool = True,
                 eval_extra: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Execute all stages in order and persist the dataset version."""
    stats: dict[str, Any] = {}
    t0 = time.time()

    recs = stage_ingest(records)
    stats["ingestion"] = {"input": len(records), "valid": len(recs)}
    recs, lic_stats = stage_license_check(recs)
    stats["license_check"] = lic_stats
    recs = stage_extract(recs)
    recs, q_stats = stage_quality_filter(recs, min_quality=min_quality)
    stats["quality_filter"] = q_stats
    recs = stage_classify(recs)
    recs, dd_stats = stage_dedup(recs)
    stats["dedup"] = dd_stats
    train, eval_, contam = stage_contamination(recs, seed=seed, extra_eval=eval_extra)
    stats["contamination"] = contam
    train = stage_mix(train, mix=mix, target_total=target_total, seed=seed)
    train, tok_stats = stage_tokenize(train, tokenizer=tokenizer, seq_len=seq_len)
    stats["tokenization"] = tok_stats
    eval_ = [r for r in eval_ if "token_ids" in r or True]
    for r in eval_:
        r.pop("token_ids", None)
        r.pop("n_tokens", None)

    version_dir = VERSIONS_DIR / version
    version_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(train, version_dir / "train.jsonl")
    write_jsonl(eval_, version_dir / "eval.jsonl")
    write_json(provenance, PROCESSED_DIR / "DATA_PROVENANCE.json")

    shard_paths: list[str] = []
    if write_shards and tokenizer is not None:
        paths = stage_shards(train, PROCESSED_DIR / version / "shards", seq_len=seq_len)
        shard_paths = [str(p) for p in paths]
    stats["shards"] = shard_paths

    manifest = stage_manifest(version, train, eval_, provenance, stats, tok_stats)
    manifest["elapsed_seconds"] = round(time.time() - t0, 2)
    manifest["dataset_dir"] = str(version_dir)
    write_json(manifest, version_dir / "manifest.json")
    write_json(manifest, REPO_ROOT / "DATA_PROVENANCE.json")
    return manifest
