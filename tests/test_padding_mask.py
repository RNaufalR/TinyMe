"""Padding must never contribute to loss or metrics (P0-04)."""
from __future__ import annotations

import numpy as np


def test_padding_shifts_loss():
    """Padding a sequence must not change its loss value."""
    import jax.numpy as jnp

    from src.model import TinyMeConfig, init_params, forward
    from src.training.trainer import masked_cross_entropy

    cfg = TinyMeConfig(name="pad-test", vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=32)
    params = init_params(cfg, seed=0)
    rng = np.random.default_rng(0)
    real = rng.integers(0, cfg.vocab_size, size=(1, 9)).astype(np.int32)
    labels = real.copy()

    def loss_of(ids, lbls, mask):
        logits = forward(params, jnp.asarray(ids), cfg)
        return masked_cross_entropy(logits, jnp.asarray(lbls), jnp.asarray(mask))

    padded = np.full((1, 16), 0, dtype=np.int32)
    padded[0, :9] = real[0]
    labels_p = np.full((1, 16), -100, dtype=np.int32)
    labels_p[0, :9] = labels[0]
    mask_p = np.zeros((1, 16), dtype=bool)
    mask_p[0, :9] = True

    loss_pad, count_pad = loss_of(padded, labels_p, mask_p)
    # unpadded prefix: add one context token so both cover the same targets
    loss_ref, count_ref = loss_of(padded[:, :9], labels[:, :9], np.ones((1, 9), dtype=bool))
    assert int(count_pad) == int(count_ref)
    assert abs(float(loss_pad) - float(loss_ref)) < 1e-4


def test_padding_does_not_change_gradients():
    import jax
    import jax.numpy as jnp

    from src.model import TinyMeConfig, init_params, forward
    from src.training.trainer import masked_cross_entropy

    cfg = TinyMeConfig(name="pad-grad", vocab_size=64, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=32)
    params = init_params(cfg, seed=1)
    rng = np.random.default_rng(7)
    T, P = 8, 5
    tokens = rng.integers(0, cfg.vocab_size, size=(1, T + P)).astype(np.int32)
    labels = tokens.copy()
    labels[0, T:] = -100
    mask = np.zeros((1, T + P), dtype=bool)
    mask[0, :T] = True

    padded_tokens = tokens.copy()
    padded_tokens[0, T:] = 0

    def loss(p, ids, lbls, msk):
        return masked_cross_entropy(forward(p, jnp.asarray(ids), cfg),
                                    jnp.asarray(lbls), jnp.asarray(msk))[0]

    g_pad = jax.grad(loss)(params, padded_tokens, labels, mask)
    g_short = jax.grad(loss)(params, tokens[:, :T], labels[:, :T], mask[:, :T])
    for a, b in zip(jax.tree_util.tree_leaves(g_pad), jax.tree_util.tree_leaves(g_short)):
        assert np.abs(np.asarray(a) - np.asarray(b)).max() < 1e-5


def test_label_invariant_ignores_masked_labels():
    """Even if a masked slot's label is garbage, it must contribute nothing."""
    import jax.numpy as jnp

    from src.training.trainer import masked_cross_entropy

    rng = np.random.default_rng(3)
    logits = jnp.asarray(rng.normal(size=(1, 6, 9)).astype(np.float32))
    mask = jnp.asarray(np.array([[True, True, False, False, False, False]]))
    labels_a = jnp.asarray(np.array([[1, 2, 3, 4, 5, 6]], dtype=np.int32))
    labels_b = jnp.asarray(np.array([[1, 2, 999, 999, 999, 999]], dtype=np.int32))
    # labels >= vocab size must not blow up masked positions
    a, ca = masked_cross_entropy(logits, labels_a, mask)
    b, cb = masked_cross_entropy(logits, labels_b, mask)
    assert int(ca) == int(cb) == 2
