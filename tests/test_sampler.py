"""Sampler: complete coverage, determinism, resumability (P0-10)."""
from __future__ import annotations

import numpy as np
import pytest

from src.training.sampler import EpochSampler, SamplerState


def test_epoch_covers_every_index_exactly_once():
    """Every index appears exactly once in the epoch *permutation*; batches are
    always full, with the epoch tail wrapped (documented `tail_padding`)."""
    s = EpochSampler(dataset_size=1031, batch_size=32, seed=7)
    order = s.epoch_order(0)
    assert sorted(order.tolist()) == list(range(1031))
    assert s.batches_in_epoch == 33  # ceil(1031/32)

    seen: list[int] = []
    state = s.initial_state()
    for idx, state in s.iter_batches(state, max_batches=s.batches_in_epoch):
        assert idx.size == 32, "every batch must be complete so the training reshape is valid"
        seen.extend(idx.tolist())
    assert state.epoch == 1 and state.position == 0
    assert set(seen) == set(range(1031)), "no index may be skipped"
    pad = s.tail_padding()
    assert pad["wrapped_indices_per_epoch"] == 32 - (1031 % 32)
    assert len(seen) - 1031 == pad["wrapped_indices_per_epoch"]

    idx, _ = next(s.iter_batches(state, max_batches=1))
    assert len(idx) == 32


def test_deterministic_order():
    a = EpochSampler(500, 16, seed=42).epoch_order(3)
    b = EpochSampler(500, 16, seed=42).epoch_order(3)
    c = EpochSampler(500, 16, seed=43).epoch_order(3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_no_replacement_within_epoch():
    s = EpochSampler(100, 10, seed=1)
    order = s.epoch_order(0)
    assert len(set(order.tolist())) == 100


def test_resume_mid_epoch_continues():
    s = EpochSampler(64, 8, seed=5)
    state = s.initial_state()
    idx_first, state = next(s.iter_batches(state, max_batches=1))
    idx_second, _ = next(s.iter_batches(state, max_batches=1))
    resumed = SamplerState.from_dict(state.to_dict())
    idx_resumed, _ = next(s.iter_batches(resumed, max_batches=1))
    assert np.array_equal(idx_second, idx_resumed)
    assert not np.array_equal(idx_first, idx_second)


def test_partial_batches_are_opt_in():
    """allow_partial=False drops the tail batch; allow_partial=True wraps it so
    the batch is full (never a ragged array)."""
    s = EpochSampler(10, 4, seed=0, allow_partial=False)
    assert s.batches_in_epoch == 2
    with pytest.raises(IndexError):
        s.batch_indices(0, 2)

    p = EpochSampler(10, 4, seed=0, allow_partial=True)
    assert p.batches_in_epoch == 3
    last = p.batch_indices(0, 2)
    assert last.size == 4
    assert sorted(last.tolist()) == sorted(p.epoch_order(0)[8:].tolist() + [p.epoch_order(0)[0], p.epoch_order(0)[1]])
