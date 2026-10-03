"""Clean dataset API (corrective audit P0-06).

``load_split(version, split)`` is the only supported way to obtain training,
validation, test or challenge records. There is deliberately no function that
defaults to ``train.jsonl`` while being usable as a validation loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..utils.io_utils import REPO_ROOT, read_json, read_jsonl, sha256_file
from .sequence import IGNORE_INDEX, build_packed_batch, pack_records

VERSIONS_DIR = REPO_ROOT / "datasets" / "versions"
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
SPLIT_FILES = {
    "train": "train.jsonl",
    "validation": "validation.jsonl",
    "test": "test.jsonl",
    "challenge": "challenge.jsonl",
}


def version_dir(version: str) -> Path:
    d = VERSIONS_DIR / version
    if not d.exists():
        raise FileNotFoundError(f"dataset version {version!r} not found at {d}")
    return d


def load_split_records(version: str, split: str) -> list[dict[str, Any]]:
    """Load the raw records of one split (JSONL)."""
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    path = version_dir(version) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"split {split!r} missing for dataset version {version!r}: {path}")
    return read_jsonl(path)


def load_manifest(version: str) -> dict[str, Any]:
    return read_json(version_dir(version) / "manifest.json")


def load_tokenizer(version: str, prefer_release: bool = False):
    from ..tokenizer.bpe import TinyMeTokenizer

    candidates = []
    if prefer_release:
        candidates.append(REPO_ROOT / "release" / "tokenizer.json")
    candidates += [version_dir(version) / "tokenizer.json",
                   PROCESSED_DIR / "tokenizer.json"]
    for c in candidates:
        if c.exists():
            return TinyMeTokenizer.load(c)
    raise FileNotFoundError(f"no tokenizer found for dataset version {version!r}")


def dataset_fingerprint(version: str) -> dict[str, Any]:
    """Content fingerprint of a dataset version (used by checkpoints)."""
    d = version_dir(version)
    files = {}
    for name in ("train.jsonl", "validation.jsonl", "test.jsonl", "challenge.jsonl",
                 "manifest.json", "tokenizer.json"):
        p = d / name
        if p.exists():
            files[name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    return {"version": version, "files": files}


def load_split_arrays(version: str, split: str, stage: str = "sft") -> dict[str, np.ndarray]:
    """Load pre-built packed shards for ``(split, stage)`` (fast path)."""
    from .shard_writer import load_shard_arrays

    shard_dir = PROCESSED_DIR / version / "shards"
    return load_shard_arrays(shard_dir, f"{split}_{stage}")


def load_split(version: str, split: str, seq_len: int = 256, tokenizer=None,
               max_records: int | None = None, mode: str = "packed",
               stage: str = "sft",
               record_filter: Callable[[dict[str, Any]], bool] | None = None,
               ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build model tensors for ``split`` of ``version``.

    Returns ``(arrays, stats)`` where ``arrays`` holds ``input_ids``, ``labels``,
    ``loss_mask`` and ``doc_ids``. Loss masking and padding masking are applied
    here — the trainer never sees an unmasked label.

    ``stage`` selects the pre-built shard family (``pretrain``/``sft``); when
    shards exist they are memory-mapped instead of re-tokenising JSONL.
    """
    if stage == "mixed":
        # Curriculum-B control: a single shuffled pool of plain documents *and*
        # structured records, so the schedule cannot present them in an order.
        try:
            plain = load_split_arrays(version, split, "pretrain")
            structured = load_split_arrays(version, split, "sft")
        except FileNotFoundError:
            plain = structured = None
        if plain is not None and structured is not None:
            data = {k: np.concatenate([plain[k], structured[k]], axis=0)
                    for k in ("input_ids", "labels", "loss_mask", "doc_ids")}
            order = np.random.default_rng(0).permutation(data["input_ids"].shape[0])
            data = {k: v[order] for k, v in data.items()}
            stats = {"split": split, "version": version, "stage": stage, "source": "shards(mixed)",
                     "blocks": int(data["input_ids"].shape[0]),
                     "active_target_tokens": int(data["loss_mask"].sum()),
                     "padding_ratio": round(float((~data["loss_mask"]).mean()), 6)}
            return data, stats
    if record_filter is not None:
        # Curriculum stage (DEC-013): select records from the raw JSONL and pack
        # them in memory.  There is deliberately no shard fast-path here — a
        # stale shard family must never be mistaken for a curriculum selection.
        records = load_split_records(version, split)
        selected = [r for r in records if record_filter(r)]
        if not selected:
            raise ValueError(
                f"{version}/{split}: the curriculum filter selected no record "
                f"(of {len(records)}) — refusing to train on an empty stage")
        if max_records:
            selected = selected[:max_records]
        if tokenizer is None:
            tokenizer = load_tokenizer(version)
        # Use the *same* packer as the corpus builder (scripts/prepare_data_v2.py):
        # it expands an over-long trajectory into one example per supervised turn
        # instead of truncating it, so a curriculum stage sees exactly the records
        # the pipeline packed — only the selection differs.
        data, stats = pack_records(selected, tokenizer, seq_len)
        stats.update({"split": split, "version": version, "stage": stage,
                      "source": "jsonl(curriculum)", "n_records": len(selected),
                      "n_records_available": len(records),
                      "blocks": int(data["input_ids"].shape[0])})
        if data["input_ids"].shape[0]:
            stats["active_target_tokens"] = int(data["loss_mask"].sum())
            stats["padding_ratio"] = round(float((~data["loss_mask"]).mean()), 6)
        return data, stats
    try:
        data = load_split_arrays(version, split, stage)
        stats = {"split": split, "version": version, "stage": stage,
                 "source": "shards", "blocks": int(data["input_ids"].shape[0])}
        # never train on shards built by a different sequence contract
        recorded = load_manifest(version).get("shards_fingerprint")
        actual = shard_fingerprint(PROCESSED_DIR / version / "shards")
        if recorded and recorded != actual:
            raise RuntimeError(
                f"{version}: packed shards are stale (manifest {recorded[:12]}… != disk {actual[:12]}…). "
                "Rebuild with scripts/prepare_data_v2.py — refusing to train on shards that do not "
                "match the recorded data contract.")
        stats["active_target_tokens"] = int(data["loss_mask"].sum())
        stats["padding_ratio"] = round(float((~data["loss_mask"]).mean()), 6)
        return data, stats
    except FileNotFoundError:
        pass
    records = load_split_records(version, split)
    if max_records:
        records = records[:max_records]
    if tokenizer is None:
        tokenizer = load_tokenizer(version)
    data, stats = build_packed_batch(records, tokenizer, seq_len, max_examples=None)
    stats["split"] = split
    stats["version"] = version
    stats["n_records"] = len(records)
    if data["input_ids"].shape[0]:
        active = data["loss_mask"].sum()
        stats["active_target_tokens"] = int(active)
        stats["padding_ratio"] = round(float((data["input_ids"] == tokenizer.tok.token_to_id("<|pad|>")).mean()), 6)
        stats["masked_positions"] = int((data["loss_mask"] == False).sum())  # noqa: E712
    return data, stats


def load_shard_tokens(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    """Load a token shard written by the pipeline (memmap-friendly)."""
    arr = np.load(Path(path), allow_pickle=False)
    return arr.astype(np.int32)


def shard_fingerprint(shard_dir: str | Path) -> str:
    shard_dir = Path(shard_dir)
    h = hashlib.sha256()
    for p in sorted(shard_dir.glob("*.npy")):
        if p.name.endswith(".meta.json"):
            continue
        arr = np.load(p, allow_pickle=False, mmap_mode="r")
        h.update(p.name.encode())
        h.update(str(arr.shape).encode())
        h.update(str(arr.dtype).encode())
        h.update(np.ascontiguousarray(arr).tobytes())   # full content, not a sample
    return h.hexdigest()


def iter_packed_batches(arrays: dict[str, np.ndarray], batch_size: int, order: np.ndarray | None = None):
    """Yield masked batches ``(input_ids, labels, loss_mask, doc_ids)``."""
    n = arrays["input_ids"].shape[0]
    idx = np.arange(n) if order is None else np.asarray(order)
    for start in range(0, n, batch_size):
        sel = idx[start:start + batch_size]
        if sel.size == 0:
            continue
        yield (arrays["input_ids"][sel], arrays["labels"][sel],
               arrays["loss_mask"][sel], arrays["doc_ids"][sel])


def stats_from_labels(labels: np.ndarray, loss_mask: np.ndarray) -> dict[str, Any]:
    active = int(loss_mask.sum())
    total = int(labels.size)
    return {"active_target_tokens": active, "masked_tokens": total - active,
            "padding_ratio": round(1.0 - active / max(total, 1), 6),
            "ignore_index": IGNORE_INDEX}


def write_split(version: str, split: str, records: list[dict[str, Any]]) -> Path:
    from ..utils.io_utils import write_jsonl

    path = version_dir(version) / SPLIT_FILES[split]
    write_jsonl(records, path)
    return path


def describe_version(version: str) -> dict[str, Any]:
    out: dict[str, Any] = {"version": version, "splits": {}}
    try:
        manifest = load_manifest(version)
        out.update({"dataset_fingerprint": manifest.get("dataset_fingerprint"),
                    "tokenizer_hash": manifest.get("tokenizer_hash"),
                    "tokenizer_version": manifest.get("tokenizer_version"),
                    "seq_len": manifest.get("seq_len"),
                    "licenses": manifest.get("licenses"),
                    "contamination": manifest.get("train_validation_test_contamination"),
                    "verified_samples": manifest.get("verified_samples"),
                    "created": manifest.get("created")})
    except FileNotFoundError:
        pass
    for split in SPLIT_FILES:
        try:
            recs = load_split_records(version, split)
        except FileNotFoundError:
            continue
        out["splits"][split] = {
            "records": len(recs),
            "chars": sum(len(r.get("text", "")) for r in recs),
            "families": len({r.get("group_id") or r.get("template_id") for r in recs}),
        }
    return out
