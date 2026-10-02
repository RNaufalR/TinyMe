"""Single RNG service for python/numpy/JAX (corrective audit P0-13).

Every stochastic decision (parameter init, shuffling, sampling, dropout) draws
from this service, and its state is serialisable so a resumed run continues the
exact same random stream.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class RNGState:
    seed: int = 1234
    python_state: list = field(default_factory=list)
    numpy_state: list = field(default_factory=list)
    jax_keys: list = field(default_factory=list)
    epoch: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RNGState":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class RNGService:
    def __init__(self, seed: int = 1234):
        self.state = RNGState(seed=int(seed))
        self.seed(int(seed))

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed % (2**32))
        self.state.seed = int(seed)
        self._refresh()

    def _refresh(self) -> None:
        self.state.python_state = list(random.getstate()[1])
        self.state.numpy_state = np.random.get_state()[1].tolist()[:8]

    def numpy_generator(self, stream: int | str = 0) -> np.random.Generator:
        tag = stream if isinstance(stream, int) else abs(hash(stream)) % (2**31)
        return np.random.default_rng([self.state.seed, self.state.epoch, int(tag)])

    def next_epoch(self) -> int:
        self.state.epoch += 1
        return self.state.epoch

    def jax_key(self, stream: int | str = 0) -> Any:
        import jax

        tag = stream if isinstance(stream, int) else abs(hash(stream)) % (2**31)
        return jax.random.PRNGKey((self.state.seed * 1000003 + self.state.epoch * 7919 + int(tag)) % (2**31))

    def save_state(self) -> dict[str, Any]:
        self._refresh()
        return self.state.to_dict()

    def load_state(self, d: dict[str, Any] | None) -> None:
        if not d:
            return
        self.state = RNGState.from_dict(d)
        # numpy state tuple is (name, keys, pos, has_gauss, cached_gaussian)
        keys = np.asarray((d.get("numpy_state") or []) + [0] * (624 - len(d.get("numpy_state") or [])),
                          dtype=np.uint32)
        np.random.set_state(("MT19937", keys, 0, 0.0, 0.0))
        if d.get("python_state"):
            random.setstate((3, tuple(d["python_state"]), None))


def set_global_seeds(seed: int) -> RNGService:
    return RNGService(seed)
