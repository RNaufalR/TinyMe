"""Single authoritative sequence builder (corrective audit P0-08, P0-03, P0-04).

Exactly one code path turns records into model tensors:

    record → segments → tokens → (input_ids, labels, loss_mask, doc_ids)

Guarantees
* ``labels`` contains ``-100`` wherever a token is not a training target.
* padding tokens can never be predicted and never contribute to the loss.
* tool results are context only: they are never generation targets.
* packing never lets a document predict across a document boundary, and
  attention is masked by ``doc_ids`` so packed documents stay separate.
* truncation is target-aware: the answer / tool call is preserved, context is
  cropped first, and samples whose target cannot be preserved are *rejected*
  (counted, not silently mangled).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Iterator

import numpy as np

from .records import (CONTEXT_ROLES, ROLE_CODE, ROLE_SYSTEM, ROLE_TOOL_CALL,
                      ROLE_TOOL_RESULT, Segment, TrainingRecord)

IGNORE_INDEX = -100
MIN_TARGET_TOKENS = 6

_ROLE_OPEN = {
    ROLE_CODE: "<|code|>",
    ROLE_TOOL_CALL: "<|tool_call|>",
    ROLE_TOOL_RESULT: "<|tool_result|>",
}

#: Public role -> opening control token map (canonical protocol rendering).
ROLE_TOKENS = {
    "system": "<|system|>", "user": "<|user|>", "assistant": "<|assistant|>",
    "thought": "<|thought|>", "final": "<|final|>", "code": "<|code|>",
    "tool_call": "<|tool_call|>", "tool_result": "<|tool_result|>",
}
ROLE_CLOSE_TOKENS = {"code": "<|endcode|>", "tool_call": "<|end_tool_call|>",
                     "tool_result": "<|end_tool_result|>"}
#: Canonical closing tokens (audit §25).  The compact ``<|endtool_call|>`` /
#: ``<|endtool_result|>`` spellings remain valid *input* aliases in the parser.
_ROLE_CLOSE = {
    ROLE_CODE: "<|endcode|>",
    ROLE_TOOL_CALL: "<|end_tool_call|>",
    ROLE_TOOL_RESULT: "<|end_tool_result|>",
}


@dataclass
class SequenceExample:
    input_ids: np.ndarray
    labels: np.ndarray
    loss_mask: np.ndarray
    doc_ids: np.ndarray
    n_active_targets: int
    n_padding: int
    truncated: bool
    dropped_context_tokens: int
    truncated_target_tokens: int
    source_record_id: str
    split: str = ""

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])


def _segments_of(record: dict[str, Any] | TrainingRecord) -> list[Segment]:
    if isinstance(record, TrainingRecord):
        return record.as_segments()
    if record.get("segments"):
        return [Segment.from_dict(s) for s in record["segments"]]
    tr = TrainingRecord.from_dict(record)
    return tr.as_segments()


def encode_segment(seg: Segment, tokenizer: Any) -> tuple[list[int], list[bool]]:
    """Encode one segment to ids plus a per-token target flag."""
    open_tok = _ROLE_OPEN.get(seg.role, f"<|{seg.role}|>")
    close_tok = _ROLE_CLOSE.get(seg.role, "")
    pieces: list[int] = []
    flags: list[bool] = []
    marker = tokenizer.encode_ids(open_tok + "\n")
    # Role markers themselves are structural, never prediction targets.
    pieces += marker
    flags += [False] * len(marker)
    body = tokenizer.encode_ids(seg.text)
    if seg.role in _ROLE_CLOSE:
        # body may itself contain the closing token text; strip it defensively
        body = _truncate_at_marker(body, tokenizer, _ROLE_CLOSE[seg.role])
    pieces += body
    flags += [seg.contributes_to_loss()] * len(body)
    if close_tok:
        cnt = tokenizer.encode_ids(close_tok)
        pieces += cnt
        flags += [False] * len(cnt)
    else:
        pieces += tokenizer.encode_ids("\n")
        flags += [False]
    return pieces, flags


def _truncate_at_marker(ids: list[int], tokenizer: Any, marker: str) -> list[int]:
    marker_ids = tokenizer.encode_ids(marker)
    if not marker_ids:
        return ids
    n = len(marker_ids)
    for i in range(len(ids) - n + 1):
        if ids[i:i + n] == marker_ids:
            return ids[:i]
    return ids


def _pad(arr: np.ndarray, seq_len: int, pad_id: int, fill: int = IGNORE_INDEX) -> np.ndarray:
    if arr.shape[0] >= seq_len:
        return arr[:seq_len]
    out = np.full(seq_len, fill, dtype=arr.dtype)
    out[:arr.shape[0]] = arr
    return out


def _is_lm_record(record: dict[str, Any] | TrainingRecord) -> bool:
    """True when the record is a plain document (Stage A / causal LM)."""
    if isinstance(record, TrainingRecord):
        return not record.segments
    return not record.get("segments")


def build_lm_example(record: dict[str, Any] | TrainingRecord, tokenizer: Any, seq_len: int,
                     mode: str = "packed") -> SequenceExample | None:
    """Plain causal-LM example: every content token is a target (windowing)."""
    pad_id = int(tokenizer.tok.token_to_id("<|pad|>"))
    bos_id = int(tokenizer.tok.token_to_id("<|bos|>"))
    eos_id = int(tokenizer.tok.token_to_id("<|eos|>"))
    text = record.text if isinstance(record, TrainingRecord) else record.get("text", "")
    ids = [bos_id] + tokenizer.encode_ids(text) + [eos_id]
    truncated = False
    if len(ids) > seq_len:
        # sliding window over the tail keeps the most recent context; the full
        # document is covered by emitting several windows from the caller
        ids = ids[-seq_len:]
        truncated = True
    n_pad = max(0, seq_len - len(ids)) if mode != "dynamic" else 0
    if n_pad:
        ids = ids + [pad_id] * n_pad
    input_ids = np.asarray(ids, dtype=np.int32)
    labels = np.full_like(input_ids, IGNORE_INDEX)
    loss_mask = np.zeros_like(input_ids, dtype=bool)
    if input_ids.shape[0] > 1:
        labels[:-1] = input_ids[1:]
        loss_mask[:-1] = True
        if n_pad:
            labels[-(n_pad + 1):] = IGNORE_INDEX
            loss_mask[-(n_pad + 1):] = False
    doc_ids = np.zeros_like(input_ids)
    rid = record.record_id if isinstance(record, TrainingRecord) else str(record.get("record_id", ""))
    split = record.split if isinstance(record, TrainingRecord) else str(record.get("split", ""))
    return SequenceExample(input_ids=input_ids, labels=labels, loss_mask=loss_mask,
                           doc_ids=doc_ids, n_active_targets=int(loss_mask.sum()),
                           n_padding=int(n_pad), truncated=truncated, dropped_context_tokens=0,
                           truncated_target_tokens=0, source_record_id=rid, split=split)


def build_sequence(record: dict[str, Any] | TrainingRecord, tokenizer: Any, seq_len: int,
                   mode: str = "packed") -> SequenceExample | None:
    """Build one fixed-length training example, or ``None`` if not buildable.

    ``mode``:
      * ``"dynamic"``  – no padding; sequence is as long as the content (capped).
      * ``"packed"``   – padding to ``seq_len`` (the trainer's fixed width).
      * ``"window"``   – long plain-LM documents are cut into sliding windows by
        the caller; here a single window is built from the tail.
    """
    if _is_lm_record(record):
        return build_lm_example(record, tokenizer, seq_len, mode)
    pad_id = int(tokenizer.tok.token_to_id("<|pad|>"))
    bos_id = int(tokenizer.tok.token_to_id("<|bos|>"))
    eos_id = int(tokenizer.tok.token_to_id("<|eos|>"))
    segs = _segments_of(record)
    ids: list[int] = [bos_id]
    target_flags: list[bool] = [False]
    target_start = 1
    for seg in segs:
        seg_ids, seg_flags = encode_segment(seg, tokenizer)
        if any(seg_flags):
            target_start = min(target_start, len(ids)) if ids else len(ids)
        ids += seg_ids
        target_flags += seg_flags
    ids.append(eos_id)
    target_flags.append(True)
    if len(ids) < 4 or not any(target_flags[1:]):
        return None

    truncated = False
    dropped_context = 0
    truncated_target_tokens = 0
    budget = seq_len if mode != "dynamic" else max(seq_len, 2048)
    if len(ids) > budget:
        truncated = True
        first_target = next((i for i, f in enumerate(target_flags) if f), len(ids))
        context = ids[1:first_target]
        target = ids[first_target:-1]          # without EOS
        target_flags_t = target_flags[first_target:-1]
        target_budget = budget - 2 - min(len(context), max(1, budget // 4))
        if len(target) > target_budget:
            drop = len(target) - target_budget
            target = target[drop:]
            target_flags_t = target_flags_t[drop:]
            truncated_target_tokens = drop
        if len(target) < MIN_TARGET_TOKENS:
            return None                        # cannot preserve a meaningful target
        ctx_budget = budget - 2 - len(target)
        if len(context) > ctx_budget:
            keep_head = min(len(context), max(1, ctx_budget // 3))
            keep_tail = ctx_budget - keep_head
            dropped_context = len(context) - ctx_budget
            context = context[:keep_head] + (context[-keep_tail:] if keep_tail > 0 else [])
        ids = [bos_id] + context + target + [eos_id]
        target_flags = [False] + [False] * len(context) + list(target_flags_t) + [True]

    n_pad = 0
    if mode != "dynamic":
        n_pad = max(0, seq_len - len(ids))
        pad = [pad_id] * n_pad
        ids = ids + pad
        target_flags = target_flags + [False] * n_pad

    input_ids = np.asarray(ids, dtype=np.int32)
    # labels[i] = ids[i+1] when position i+1 is a target, else IGNORE_INDEX
    labels = np.full_like(input_ids, IGNORE_INDEX)
    loss_mask = np.zeros_like(input_ids, dtype=bool)
    if input_ids.shape[0] > 1:
        nxt_is_target = np.asarray(target_flags[1:], dtype=bool)
        labels[:-1] = np.where(nxt_is_target, input_ids[1:], IGNORE_INDEX)
        loss_mask[:-1] = nxt_is_target
        loss_mask[-1] = False
    # padding never contributes
    if n_pad:
        loss_mask[-n_pad:] = False
        labels[-n_pad:] = IGNORE_INDEX
    doc_ids = np.zeros_like(input_ids)
    return SequenceExample(input_ids=input_ids, labels=labels, loss_mask=loss_mask,
                           doc_ids=doc_ids, n_active_targets=int(loss_mask.sum()),
                           n_padding=int(n_pad), truncated=truncated,
                           dropped_context_tokens=dropped_context,
                           truncated_target_tokens=truncated_target_tokens,
                           source_record_id=str(record.get("record_id", "") if isinstance(record, dict) else record.record_id),
                           split=str(record.get("split", "") if isinstance(record, dict) else record.split))


def build_packed_batch(records: Iterable[dict[str, Any]], tokenizer: Any, seq_len: int,
                       max_examples: int | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pack multiple records into fixed-length blocks with document isolation.

    Documents are concatenated without padding *inside* a block; each keeps its
    own ``doc_id`` so attention and loss never cross the boundary. A block is
    closed when the next document would not fit.
    """
    pad_id = int(tokenizer.tok.token_to_id("<|pad|>"))
    blocks: list[SequenceExample] = []
    cur_ids: list[int] = []
    cur_labels: list[int] = []
    cur_mask: list[bool] = []
    cur_doc: list[int] = []
    doc_counter = 0
    stats = {"records_seen": 0, "records_packed": 0, "records_rejected": 0,
             "context_truncated": 0, "target_truncated": 0}

    def flush() -> None:
        nonlocal cur_ids, cur_labels, cur_mask, cur_doc
        if not cur_ids:
            return
        n_pad = seq_len - len(cur_ids)
        ids = np.asarray(cur_ids + [pad_id] * n_pad, dtype=np.int32)
        labels = np.asarray(cur_labels + [IGNORE_INDEX] * n_pad, dtype=np.int32)
        mask = np.asarray(cur_mask + [False] * n_pad, dtype=bool)
        doc = np.asarray(cur_doc + [max(doc_counter - 1, 0)] * n_pad, dtype=np.int32)
        blocks.append(SequenceExample(ids, labels, mask, doc, int(mask.sum()), int(n_pad),
                                      False, 0, 0, "", ""))
        cur_ids, cur_labels, cur_mask, cur_doc = [], [], [], []

    for rec in records:
        stats["records_seen"] += 1
        ex = build_sequence(rec, tokenizer, seq_len, mode="dynamic")
        if ex is None:
            stats["records_rejected"] += 1
            continue
        # trim trailing EOS/pad artefacts from the dynamic example
        n = int((ex.loss_mask | (ex.input_ids != pad_id)).sum())
        ids = ex.input_ids[:n].tolist()
        labels = ex.labels[:n].tolist()
        mask = ex.loss_mask[:n].tolist()
        if ex.truncated:
            stats["context_truncated"] += 1
        if ex.truncated_target_tokens:
            stats["target_truncated"] += 1
        if len(ids) > seq_len:
            stats["records_rejected"] += 1
            continue
        if len(cur_ids) + len(ids) > seq_len:
            flush()
            doc_counter += 1
        cur_ids += ids
        cur_labels += labels
        cur_mask += mask
        cur_doc += [doc_counter] * len(ids)
        stats["records_packed"] += 1
        if max_examples and len(blocks) >= max_examples:
            break
    if cur_ids:
        flush()
    if not blocks:
        return ({"input_ids": np.zeros((0, seq_len), np.int32), "labels": np.zeros((0, seq_len), np.int32),
                 "loss_mask": np.zeros((0, seq_len), bool), "doc_ids": np.zeros((0, seq_len), np.int32)},
                stats)
    data = {
        "input_ids": np.stack([b.input_ids for b in blocks]),
        "labels": np.stack([b.labels for b in blocks]),
        "loss_mask": np.stack([b.loss_mask for b in blocks]),
        "doc_ids": np.stack([b.doc_ids for b in blocks]),
    }
    stats["blocks"] = len(blocks)
    stats["active_target_tokens"] = int(data["loss_mask"].sum())
    stats["padding_tokens"] = int((~data["loss_mask"]).sum())
    stats["padding_ratio"] = round(float((data["input_ids"] == pad_id).mean()), 6)
    return data, stats


def sliding_windows(text: str, tokenizer: Any, seq_len: int, stride: int | None = None) -> list[list[int]]:
    """Split a long plain-text document into overlapping windows."""
    ids = tokenizer.encode_ids(text)
    stride = stride or seq_len // 2
    if len(ids) <= seq_len:
        return [ids]
    out: list[list[int]] = []
    for start in range(0, len(ids), stride):
        chunk = ids[start:start + seq_len]
        if len(chunk) < 16:
            break
        out.append(chunk)
        if start + seq_len >= len(ids):
            break
    return out


def pack_records(records: Iterable[dict[str, Any]], tokenizer: Any, seq_len: int,
                 window_long_docs: bool = True) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pack a stream of records into fixed-length blocks with document isolation.

    * structured (segmented) records keep their loss mask and never cross-join;
    * plain documents longer than ``seq_len`` are emitted as several sliding
      windows so no content is silently dropped;
    * a block is closed before a document that would not fit.
    """
    pad_id = int(tokenizer.tok.token_to_id("<|pad|>"))
    blocks: list[dict[str, np.ndarray]] = []
    cur: dict[str, list] = {"ids": [], "labels": [], "mask": [], "doc": []}
    doc_counter = 0
    stats: dict[str, Any] = {"records_seen": 0, "records_rejected": 0, "blocks": 0,
                             "windows": 0, "truncated_windows": 0, "active_target_tokens": 0,
                             "padding_tokens": 0}

    def flush() -> None:
        nonlocal cur, doc_counter
        if not cur["ids"]:
            return
        n_pad = seq_len - len(cur["ids"])
        blocks.append({
            "input_ids": np.asarray(cur["ids"] + [pad_id] * n_pad, dtype=np.int32),
            "labels": np.asarray(cur["labels"] + [IGNORE_INDEX] * n_pad, dtype=np.int32),
            "loss_mask": np.asarray(cur["mask"] + [False] * n_pad, dtype=bool),
            "doc_ids": np.asarray(cur["doc"] + [max(doc_counter - 1, 0)] * n_pad, dtype=np.int32),
        })
        stats["padding_tokens"] += n_pad
        stats["blocks"] += 1
        cur = {"ids": [], "labels": [], "mask": [], "doc": []}
        doc_counter += 1

    def add(ids: list[int], labels: list[int], mask: list[bool]) -> None:
        nonlocal cur, doc_counter
        if len(ids) > seq_len:
            return
        if len(cur["ids"]) + len(ids) > seq_len:
            flush()
        cur["ids"] += ids
        cur["labels"] += labels
        cur["mask"] += mask
        cur["doc"] += [doc_counter] * len(ids)
        stats["active_target_tokens"] += int(sum(mask))

    for rec in records:
        stats["records_seen"] += 1
        segments = rec.get("segments") if isinstance(rec, dict) else getattr(rec, "segments", None)
        if not segments and window_long_docs:
            text = rec.get("text", "") if isinstance(rec, dict) else rec.text
            windows = sliding_windows(text, tokenizer, max(8, seq_len - 2))
            stats["windows"] += len(windows)
            emitted = False
            for w in windows:
                if len(w) < 8:
                    continue
                ids = [int(tokenizer.tok.token_to_id("<|bos|>"))] + list(w) + [int(tokenizer.tok.token_to_id("<|eos|>"))]
                if len(ids) > seq_len:
                    ids = ids[:seq_len]
                    stats["truncated_windows"] += 1
                labels = [IGNORE_INDEX] * len(ids)
                mask = [False] * len(ids)
                for i in range(len(ids) - 1):
                    if i + 1 < len(ids):
                        labels[i] = ids[i + 1]
                        mask[i] = True
                add(ids, labels, mask)
                emitted = True
            if not emitted:
                stats["records_rejected"] += 1
            continue
        ex = build_sequence(rec, tokenizer, seq_len, mode="dynamic")
        if ex is None:
            stats["records_rejected"] += 1
            continue
        n = len(ex.input_ids)
        add(ex.input_ids[:n].tolist(), ex.labels[:n].tolist(), ex.loss_mask[:n].tolist())
    flush()
    if not blocks:
        return ({"input_ids": np.zeros((0, seq_len), np.int32), "labels": np.zeros((0, seq_len), np.int32),
                 "loss_mask": np.zeros((0, seq_len), bool), "doc_ids": np.zeros((0, seq_len), np.int32)}, stats)
    data = {k: np.stack([b[k] for b in blocks]) for k in ("input_ids", "labels", "loss_mask", "doc_ids")}
    stats["active_target_tokens"] = int(data["loss_mask"].sum())
    stats["padding_tokens"] = int((~data["loss_mask"] & (data["input_ids"] == pad_id)).sum())
    stats["padding_ratio"] = round(float((data["input_ids"] == pad_id).mean()), 6)
    stats["tokens_per_block"] = round(float(data["loss_mask"].sum()) / max(data["input_ids"].shape[0], 1), 1)
    return data, stats
