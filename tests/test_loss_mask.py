"""Loss masking: ignored tokens and tool results never become targets (P0-03)."""
from __future__ import annotations

import jax
import numpy as np
import pytest

from src.training.trainer import masked_cross_entropy


def test_masked_positions_contribute_zero_loss():
    import jax.numpy as jnp

    rng = np.random.default_rng(0)
    logits = jnp.asarray(rng.normal(size=(1, 4, 11)).astype(np.float32))
    labels = jnp.asarray(np.array([[1, 2, 3, 4]], dtype=np.int32))
    all_mask = jnp.ones((1, 4), dtype=bool)
    no_mask = jnp.zeros((1, 4), dtype=bool)
    total_all, count_all = masked_cross_entropy(logits, labels, all_mask)
    total_none, count_none = masked_cross_entropy(logits, labels, no_mask)
    assert float(count_none) == 0.0
    assert float(total_none) == 0.0
    assert float(count_all) == 4.0
    assert float(total_all) > 0.0


def test_ignore_index_labels_do_not_contribute():
    import jax.numpy as jnp

    rng = np.random.default_rng(1)
    logits = jnp.asarray(rng.normal(size=(1, 5, 13)).astype(np.float32))
    labels = jnp.asarray(np.array([[3, -100, 7, -100, 9]], dtype=np.int32))
    mask = jnp.asarray(np.array([[True, True, True, True, True]]))
    total, count = masked_cross_entropy(logits, labels, mask)
    # Only the target labels contributing are 3, 7, 9; recompute by hand
    logp = np.asarray(jax.nn.log_softmax(logits, axis=-1))[0]
    manual = -(logp[0, 3] + logp[2, 7] + logp[4, 9])
    assert np.isclose(float(total), manual, atol=1e-4)


def test_masked_matches_manual_softmax_ce():
    import jax.numpy as jnp

    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.normal(size=(2, 6, 17)).astype(np.float32))
    labels = jnp.asarray(rng.integers(0, 17, size=(2, 6)).astype(np.int32))
    mask = jnp.asarray(rng.random((2, 6)) > 0.5)
    total, count = masked_cross_entropy(logits, labels, mask)
    logp = np.asarray(jax.nn.log_softmax(logits, axis=-1))
    m = np.asarray(mask)
    manual = -logp[np.arange(2)[:, None], np.arange(6)[None, :], np.asarray(labels)][m].sum()
    assert np.isclose(float(total), manual, atol=1e-3)
    assert int(count) == int(m.sum())


def test_tool_result_is_context_only(sample_record):
    """A record whose tool result is large must not gain target tokens from it."""
    from src.data.records import make_segment_record

    terse = make_segment_record(
        segments=[__import__("src.data.records", fromlist=["Segment"]).Segment("user", "hi"),
                  __import__("src.data.records", fromlist=["Segment"]).Segment("tool_result", "TOOL OUTPUT"),
                  __import__("src.data.records", fromlist=["Segment"]).Segment("final", "answer")],
        category="tool_use", source="tool_runtime", source_id="unit/tool/1", task_type="tool_call")
    segs = terse.as_segments()
    result_seg = [s for s in segs if s.role == "tool_result"][0]
    assert result_seg.contributes_to_loss() is False
    final_seg = [s for s in segs if s.role == "final"][0]
    assert final_seg.contributes_to_loss() is True


def test_loss_mask_alignment_with_labels():
    """Every masked label must be exactly -100 and vice versa."""
    from src.data.sequence import build_sequence

    class FakeTok:
        class _tok:
            @staticmethod
            def token_to_id(name):
                return {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2}.get(name, 3)

        tok = _tok()

        def encode_ids(self, text):
            return [10 + (ord(c) % 20) for c in text] or [3]

    rec = {"segments": [{"role": "user", "text": "hello", "target": False},
                        {"role": "assistant", "text": "world", "target": True}],
           "record_id": "x"}
    ex = build_sequence(rec, FakeTok(), seq_len=64, mode="packed")
    assert ex is not None
    # exact invariant: labels < 0  <=>  loss_mask False
    assert np.array_equal(ex.labels < 0, ~ex.loss_mask)
    assert ex.n_active_targets > 0
