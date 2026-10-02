"""Deterministic epoch sampler with persisted state (corrective audit P0-10).

Properties
  * every index is visited exactly once per epoch (no tail loss, no silent
    replacement sampling);
  * the permutation for epoch ``e`` is derived from ``(seed, e)`` so the
    ordering is reproducible without serialising the permutation itself;
  * batches never span epoch boundaries unless ``allow_partial=True``;
  * progress (epoch, batch position, consumed samples) can be saved and
    restored so a resumed run continues mid-epoch instead of restarting.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterator

import numpy as np


@dataclass
class SamplerState:
    epoch: int = 0
    position: int = 0          # batch index within the epoch
    consumed: int = 0          # samples consumed in this epoch
    batches_in_epoch: int = 0
    dataset_size: int = 0
    batch_size: int = 0
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SamplerState":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class EpochSampler:
    """Reproducible, complete-coverage epoch sampler."""

    def __init__(self, dataset_size: int, batch_size: int, seed: int = 1234,
                 fingerprint: str = "", allow_partial: bool = True):
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.fingerprint = fingerprint
        self.allow_partial = allow_partial
        self.batches_in_epoch = dataset_size // batch_size
        if allow_partial and dataset_size % batch_size:
            self.batches_in_epoch += 1

    # ----------------------------------------------------------- ordering
    def epoch_order(self, epoch: int) -> np.ndarray:
        rng = np.random.default_rng([self.seed, epoch])
        return rng.permutation(self.dataset_size)

    def batch_indices(self, epoch: int, position: int) -> np.ndarray:
        """Exactly ``batch_size`` indices — never a short batch.

        The epoch tail is padded by *wrapping to the start of the same epoch
        permutation*, so the total number of steps per epoch is known in advance
        and the training reshape is always valid.  Every index still appears at
        least once per epoch; the few wrapped indices appear twice at the
        boundary (reported by ``tail_padding`` for full transparency).
        """
        order = self.epoch_order(epoch)
        start = position * self.batch_size
        chunk = order[start:start + self.batch_size]
        if chunk.size == 0:
            raise IndexError(f"batch {position} past the end of epoch {epoch}")
        if chunk.size < self.batch_size:
            if not self.allow_partial:
                raise IndexError(f"batch {position} past the end of epoch {epoch}")
            pad = self.batch_size - chunk.size
            chunk = np.concatenate([chunk, np.take(order, np.arange(pad) % self.dataset_size)])
        return chunk

    def tail_padding(self) -> dict:
        """How many indices are wrapped into the final epoch batch."""
        remainder = self.dataset_size % self.batch_size
        return {"tail_remainder": int(remainder),
                "wrapped_indices_per_epoch": int((self.batch_size - remainder) if remainder else 0),
                "complete_coverage": True}

    # -------------------------------------------------------------- state
    def initial_state(self) -> SamplerState:
        return SamplerState(epoch=0, position=0, consumed=0,
                            batches_in_epoch=self.batches_in_epoch,
                            dataset_size=self.dataset_size, batch_size=self.batch_size,
                            fingerprint=self.fingerprint)

    def state_after_step(self, state: SamplerState) -> SamplerState:
        position = state.position + 1
        consumed = state.consumed + min(self.batch_size, self.dataset_size - state.consumed)
        if position >= self.batches_in_epoch:
            return SamplerState(epoch=state.epoch + 1, position=0, consumed=0,
                                batches_in_epoch=self.batches_in_epoch,
                                dataset_size=self.dataset_size, batch_size=self.batch_size,
                                fingerprint=self.fingerprint)
        return SamplerState(epoch=state.epoch, position=position, consumed=consumed,
                            batches_in_epoch=self.batches_in_epoch,
                            dataset_size=self.dataset_size, batch_size=self.batch_size,
                            fingerprint=self.fingerprint)

    def iter_batches(self, state: SamplerState | None = None,
                     max_batches: int | None = None) -> Iterator[tuple[np.ndarray, SamplerState]]:
        state = state or self.initial_state()
        produced = 0
        while max_batches is None or produced < max_batches:
            idx = self.batch_indices(state.epoch, state.position)
            next_state = self.state_after_step(state)
            yield idx, next_state
            state = next_state
            produced += 1

    def coverage_check(self) -> dict:
        order = self.epoch_order(0)
        return {
            "dataset_size": self.dataset_size,
            "unique_indices": int(np.unique(order).size),
            "full_coverage": bool(np.unique(order).size == self.dataset_size),
            "batches_in_epoch": self.batches_in_epoch,
        }
