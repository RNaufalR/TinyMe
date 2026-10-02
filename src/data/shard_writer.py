"""Efficient shard writer / reader (corrective audit P1-04).

The baseline re-loaded and re-wrote the whole ``.npy`` file for every sample
(``np.vstack`` inside the loop), which is quadratic. This module writes
*append-only chunk shards*: a fixed-capacity memmap is filled in place and a new
chunk file is opened only when the previous one is full. Nothing is ever
rewritten, and the reader memory-maps the shards instead of loading them.

Shard layout (one directory per split)::

    <dir>/<split>_<index:04d>.npy        # int32 blocks, shape (n, seq_len)
    <dir>/index.json                     # shard inventory + fingerprints
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterator

import numpy as np


class ShardWriter:
    """Append-only shard writer with a bounded in-memory buffer.

    Blocks are buffered up to ``capacity_per_shard`` and then written with a
    single ``np.save``. Nothing is ever re-read or rewritten, so the writer is
    linear in the number of blocks (the baseline was quadratic).
    """

    def __init__(self, directory: str | Path, split: str, seq_len: int,
                 capacity_per_shard: int = 512, columns: int = 4):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.split = split
        self.seq_len = seq_len
        self.capacity = capacity_per_shard
        self.columns = columns           # (input_ids, labels, loss_mask, doc_ids)
        self.shards: list[dict[str, Any]] = []
        self._buffer: list[np.ndarray] = []
        self._total = 0

    # ------------------------------------------------------------- writing
    def append(self, input_ids: np.ndarray, labels: np.ndarray, loss_mask: np.ndarray,
               doc_ids: np.ndarray) -> None:
        block = np.stack([np.asarray(input_ids, dtype=np.int64), np.asarray(labels, dtype=np.int64),
                          np.asarray(loss_mask, dtype=np.int64), np.asarray(doc_ids, dtype=np.int64)])
        self._buffer.append(block)
        self._total += 1
        if len(self._buffer) >= self.capacity:
            self._flush()

    def extend(self, data: dict[str, np.ndarray]) -> None:
        for i in range(data["input_ids"].shape[0]):
            self.append(data["input_ids"][i], data["labels"][i], data["loss_mask"][i],
                        data["doc_ids"][i])

    def append_blocks(self, data: dict[str, np.ndarray]) -> None:
        """Append a batch of pre-built blocks in one vectorised stack."""
        n = int(data["input_ids"].shape[0])
        if n == 0:
            return
        blocks = np.stack([
            np.asarray(data["input_ids"], dtype=np.int64),
            np.asarray(data["labels"], dtype=np.int64),
            np.asarray(data["loss_mask"], dtype=np.int64),
            np.asarray(data["doc_ids"], dtype=np.int64),
        ], axis=1)                                # (n, 4, seq_len)
        for start in range(0, n, self.capacity):
            chunk = blocks[start:start + self.capacity]
            self._buffer.extend(list(chunk))
            self._total += chunk.shape[0]
            if len(self._buffer) >= self.capacity:
                self._flush()

    def _flush(self) -> None:
        if not self._buffer:
            return
        index = len(self.shards)
        path = self.directory / f"{self.split}_{index:04d}.npy"
        arr = np.stack(self._buffer)                    # (n, 4, seq_len)
        tmp = path.with_suffix(".tmp.npy")
        np.save(tmp, arr)
        os.replace(tmp, path)                           # atomic publish
        self.shards.append({"file": path.name, "used": int(arr.shape[0]),
                            "bytes": path.stat().st_size, "sha256": _sha256(path)})
        self._buffer = []

    def close(self) -> dict[str, Any]:
        self._flush()
        index = {
            "split": self.split, "seq_len": self.seq_len, "blocks": self._total,
            "shards": self.shards,
        }
        index_path = self.directory / f"{self.split}_index.json"
        index_path.write_text(json.dumps(index, indent=2) + "\n", encoding="utf-8")
        return index

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------ reading
def iter_shards(directory: str | Path, split: str) -> Iterator[np.ndarray]:
    directory = Path(directory)
    for path in sorted(directory.glob(f"{split}_[0-9][0-9][0-9][0-9].npy")):
        yield np.load(path, mmap_mode="r")


def load_shard_arrays(directory: str | Path, split: str) -> dict[str, np.ndarray]:
    """Load (memory-mapped) shard arrays for one split."""
    arrays = [arr for arr in iter_shards(directory, split)]
    if not arrays:
        raise FileNotFoundError(f"no shards for split {split!r} in {directory}")
    stacked = np.concatenate(arrays, axis=0)
    return {
        "input_ids": stacked[:, 0].astype(np.int32),
        "labels": stacked[:, 1].astype(np.int32),
        "loss_mask": stacked[:, 2].astype(bool),
        "doc_ids": stacked[:, 3].astype(np.int32),
    }


def shard_index(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def total_blocks(directory: str | Path, split: str) -> int:
    return sum(arr.shape[0] for arr in iter_shards(directory, split))
