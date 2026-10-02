"""Probe: does the baseline KV-cache decode match the full forward pass?"""
import sys, numpy as np
import pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
import jax.numpy as jnp
from src.model.transformer import init_params, init_cache, forward, forward_with_cache, TinyMeConfig
cfg = TinyMeConfig.from_name("nano"); p = init_params(cfg, seed=7)
toks = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, 12)).astype(np.int32)
full = np.asarray(forward(p, jnp.asarray(toks), cfg))[0, -1]
cache = init_cache(cfg, 1); last = None
for t in range(toks.shape[1]):
    last = np.asarray(forward_with_cache(p, jnp.asarray(toks[:, t]), cache, cfg))[0, -1]
print(f"full vs cached max|delta| = {np.abs(last - full).max():.3e}; argmax equal = {np.argmax(last) == np.argmax(full)}")
