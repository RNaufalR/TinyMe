"""Probe (re-run through the test suite): is RoPE implemented as a true 2-D rotation?

Compares ``src.model.transformer.apply_rope`` against an *independent* explicit
rotation written with plain NumPy.  This is deliberately not self-referential:
the reference does not call any repository code.

The recorded output shows the post-fix values (the pre-fix defect measured
max|delta| ≈ 4.076 and is preserved in `rope_pairing_probe.out.txt.history`).

Run: python docs/audit_evidence/rope_pairing_probe_current.py
"""
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.model.transformer import apply_rope, rope_frequencies  # noqa: E402


def reference_rope(x: np.ndarray, base: float = 10000.0) -> np.ndarray:
    d = x.shape[-1]
    half = d // 2
    pos = np.arange(x.shape[-2], dtype=np.float64)
    inv = base ** (-np.arange(half, dtype=np.float64) / half)
    ang = pos[:, None] * inv[None, :]
    cos, sin = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


if __name__ == "__main__":
    import jax.numpy as jnp

    worst = 0.0
    for head_dim in (4, 6, 8):
        for seq_len in (7, 33):
            x = np.random.default_rng(head_dim * 100 + seq_len).normal(size=(2, 3, seq_len, head_dim))
            got = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(rope_frequencies(head_dim, seq_len))))
            delta = float(np.abs(got - reference_rope(x)).max())
            worst = max(worst, delta)
            print(f"head_dim={head_dim} seq_len={seq_len}: max|delta| = {delta:.3e}")
    # norm preservation is a consequence of being a rotation
    x = np.random.default_rng(1).normal(size=(1, 2, 5, 8))
    got = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(rope_frequencies(8, 5))))
    norm_delta = float(np.abs(np.linalg.norm(got, axis=-1) - np.linalg.norm(x, axis=-1)).max())
    print(f"norm preservation: max|delta| = {norm_delta:.3e}")
    print(f"worst-case parity vs independent reference: {worst:.3e} (tolerance 1e-5)")
