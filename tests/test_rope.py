"""RoPE must be the mathematical rotation, not just self-consistent (P0-14)."""
from __future__ import annotations

import numpy as np
import pytest


def reference_rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    """Explicit 2-D rotation: pair (2i, 2i+1) rotated by pos * base^(-2i/D)."""
    head_dim = x.shape[-1]
    half = head_dim // 2
    pos = np.arange(x.shape[-2], dtype=np.float64)
    inv = base ** (-np.arange(half, dtype=np.float64) / half)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


@pytest.mark.parametrize("head_dim", [4, 6, 8])
def test_rope_matches_reference(head_dim):
    import jax.numpy as jnp

    from src.model import apply_rope, rope_frequencies

    rng = np.random.default_rng(head_dim)
    x = rng.normal(size=(2, 3, 7, head_dim))
    got = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(rope_frequencies(head_dim, 7))))
    expected = reference_rope(x)
    assert np.abs(got - expected).max() < 1e-5


def test_rope_frequency_pairing():
    """Pair i must use frequency f_i in both slots (the baseline used f_0 twice)."""
    from src.model import rope_frequencies

    freqs = np.asarray(rope_frequencies(8, 3))
    half = freqs[:, ::2]
    # values: f_i = t * theta^(-2i/D)
    expected = np.outer(np.arange(3), 10000.0 ** (-np.arange(4) / 4))
    assert np.allclose(half, expected, rtol=1e-5)
    # duplication is *within* a pair, never across pairs
    assert np.allclose(freqs[:, 0::2], freqs[:, 1::2])
    assert not np.allclose(freqs[:, 1], freqs[:, 2])


def test_rope_is_norm_preserving():
    import jax.numpy as jnp

    from src.model import apply_rope, rope_frequencies

    x = np.random.default_rng(3).normal(size=(1, 2, 9, 6))
    got = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(rope_frequencies(6, 9))))
    assert np.abs(np.linalg.norm(got, axis=-1) - np.linalg.norm(x, axis=-1)).max() < 1e-5


def test_rope_relative_position_property():
    """<RoPE(q,pos+d), RoPE(k,pos'+d)> is independent of the absolute offset d."""
    import jax.numpy as jnp

    from src.model import apply_rope, rope_frequencies

    rng = np.random.default_rng(11)
    q = rng.normal(size=(1, 1, 1, 8))
    k = rng.normal(size=(1, 1, 1, 8))
    table = np.asarray(rope_frequencies(8, 40))
    scores = []
    for offset in (0, 5, 17):
        qr = np.asarray(apply_rope(jnp.asarray(q), jnp.asarray(table[10 + offset:11 + offset])))
        kr = np.asarray(apply_rope(jnp.asarray(k), jnp.asarray(table[4 + offset:5 + offset])))
        scores.append(float(np.sum(qr * kr)))
    assert max(scores) - min(scores) < 1e-5


def test_rope_odd_head_dim_rejected():
    from src.model import rope_frequencies

    with pytest.raises(ValueError):
        rope_frequencies(5, 4)
