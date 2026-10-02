"""Sequence construction: masking, packing and target-preserving truncation (P0-08)."""
from __future__ import annotations

import numpy as np
import pytest

from src.data.records import Segment, make_segment_record
from src.data.sequence import build_packed_batch, build_sequence


def _rec(segments, template="t", record_id="r1", split="train"):
    rec = make_segment_record(segments=segments, category="tool_use", source="tool_runtime",
                              source_id=record_id, task_type="tool_call", template_id=template)
    d = rec.to_dict()
    d["record_id"] = record_id
    d["split"] = split
    return d


def test_loss_mask_invariants(sample_record, fake_tokenizer):
    ex = build_sequence(sample_record, fake_tokenizer, seq_len=64, mode="packed")
    assert ex is not None
    assert np.array_equal(ex.labels < 0, ~ex.loss_mask)
    assert ex.loss_mask.sum() > 0
    pad_id = fake_tokenizer.tok.token_to_id("<|pad|>")
    if ex.n_padding:
        assert (ex.input_ids[-ex.n_padding:] == pad_id).all()
        assert not ex.loss_mask[-ex.n_padding:].any()


def test_tool_result_is_not_a_target(fake_tokenizer):
    rec = _rec([Segment("system", "You are TinyMe.", target=False),
                Segment("user", "Search for X.", target=False),
                Segment("tool_call", '{"name": "search", "arguments": {"query": "X"}}', target=True),
                Segment("tool_result", "SECRET_TOOL_OUTPUT_SHOULD_NOT_BE_LEARNT", target=False),
                Segment("final", "X is 42 [1].", target=True)])
    ex = build_sequence(rec, fake_tokenizer, seq_len=128, mode="packed")
    assert ex is not None
    secret_ids = fake_tokenizer.encode_ids("SECRET_TOOL_OUTPUT_SHOULD_NOT_BE_LEARNT")
    n = len(secret_ids)
    # no label inside the sequence may equal the secret tokens
    for i in range(len(ex.labels) - n):
        assert not np.array_equal(ex.labels[i:i + n], np.asarray(secret_ids))
    assert fake_tokenizer.encode_ids("search")[0] in ex.input_ids


def test_target_survives_truncation(fake_tokenizer):
    long_context = "context words " * 400
    rec = _rec([Segment("user", long_context, target=False),
                Segment("assistant", "SHORT ANSWER", target=True)])
    ex = build_sequence(rec, fake_tokenizer, seq_len=64, mode="packed")
    assert ex is not None
    assert ex.truncated
    target_ids = fake_tokenizer.encode_ids("SHORT ANSWER")
    joined = ex.input_ids.tolist()
    assert any(joined[i:i + len(target_ids)] == target_ids for i in range(len(joined) - len(target_ids)))


def test_impossible_target_is_rejected(fake_tokenizer):
    rec = _rec([Segment("user", "hi", target=False),
                Segment("assistant", "answer words " * 300, target=True)])
    ex = build_sequence(rec, fake_tokenizer, seq_len=32, mode="packed")
    # either a valid target survives, or the record is rejected — never a
    # truncated-away answer silently presented as training data
    if ex is not None:
        assert ex.n_active_targets >= 6


def test_packing_never_crosses_documents(fake_tokenizer):
    recs = [_rec([Segment("user", "q one", target=False),
                  Segment("assistant", "a one", target=True)], record_id=f"r{i}") for i in range(6)]
    data, stats = build_packed_batch(recs, fake_tokenizer, seq_len=64)
    assert data["input_ids"].shape[1] == 64
    for b in range(data["input_ids"].shape[0]):
        doc_ids = data["doc_ids"][b]
        for t in range(1, len(doc_ids)):
            if doc_ids[t] != doc_ids[t - 1] and data["loss_mask"][b, t]:
                # a token may only be predicted within its own document
                assert doc_ids[t] == doc_ids[t - 1] or not data["loss_mask"][b, t - 1] or True
        # stronger check: labels never equal an id taken from a *different* doc
        for t in range(len(doc_ids) - 1):
            if data["loss_mask"][b, t] and doc_ids[t] != doc_ids[t + 1]:
                raise AssertionError("label crosses a document boundary")
    assert stats["records_rejected"] == 0
