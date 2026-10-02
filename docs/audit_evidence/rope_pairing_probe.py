"""Probe: is the baseline RoPE frequency pairing mathematically correct?
Compares src.model.transformer.apply_rope with an explicit 2-D rotation reference."""
import sys, numpy as np
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from src.model.transformer import rope_frequencies, apply_rope
import jax.numpy as jnp

def reference_rope(x, base=10000.0):
    D = x.shape[-1]; half = D // 2
    pos = np.arange(x.shape[-2], dtype=np.float64)
    inv = base ** (-np.arange(half, dtype=np.float64) / half)
    ang = pos[:, None] * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = x1 * c - x2 * s
    out[..., 1::2] = x1 * s + x2 * c
    return out

for head_dim in (4, 6, 8):
    x = np.random.default_rng(head_dim).normal(size=(2, 3, 7, head_dim))
    got = np.asarray(apply_rope(jnp.asarray(x), jnp.asarray(rope_frequencies(head_dim, 7))))
    ref = reference_rope(x)
    print(f"head_dim={head_dim}: max|delta| = {np.abs(got - ref).max():.6f}")
