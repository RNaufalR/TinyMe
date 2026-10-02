"""Gradient accumulation must equal one update on the combined batch (P0-09)."""
from __future__ import annotations

import jax
import numpy as np


def _opt_counts(state):
    from src.training.checkpoint import tree_to_flat

    return [int(np.asarray(v)) for k, v in tree_to_flat(state).items()
            if k.endswith("count") and np.asarray(v).ndim == 0]


def _accum_step(trainer, rng, B=6, T=16):
    import jax.numpy as jnp

    cfg = trainer.model_cfg
    batch = (jnp.asarray(rng.integers(0, cfg.vocab_size, size=(3, 2, T)).astype(np.int32)),
             jnp.asarray(rng.integers(0, cfg.vocab_size, size=(3, 2, T)).astype(np.int32)),
             jnp.asarray(rng.random((3, 2, T)) > 0.3),
             jnp.asarray(np.zeros((3, 2, T), dtype=np.int32)))
    grads, _, _ = trainer._accum_grads(trainer.params, batch)
    trainer.params, trainer.opt_state = trainer._apply_update(trainer.params, trainer.opt_state, grads)


def _make_trainer(tmp_path, monkeypatch, tokenizer, accum=3, micro=2):
    from src.training.trainer import TrainConfig, Trainer

    from src.model import TinyMeConfig

    cfg = TrainConfig(experiment_id="TEST-GRADACC", architecture="nano", seq_len=16,
                      micro_batch_size=micro, grad_accum_steps=accum, max_steps=4,
                      warmup_steps=1, learning_rate=1e-3, compute_dtype="float32")
    monkeypatch.setattr("src.training.trainer.CHECKPOINTS_DIR", tmp_path / "ckpt")
    monkeypatch.setattr("src.training.trainer.EXPERIMENTS_DIR", tmp_path / "exp")
    model_cfg = TinyMeConfig(name="tiny", vocab_size=int(tokenizer.vocab_size), d_model=32,
                             n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
    return Trainer(cfg, model_cfg=model_cfg, tokenizer=tokenizer)


def test_accumulated_grads_equal_single_batch_grads(tmp_path, monkeypatch, fake_tokenizer):
    import jax
    import jax.numpy as jnp

    from src.training.trainer import masked_cross_entropy
    from src.model import forward

    trainer = _make_trainer(tmp_path, monkeypatch, fake_tokenizer)
    cfg = trainer.model_cfg
    rng = np.random.default_rng(0)
    B, T = 6, 16
    tokens = rng.integers(0, cfg.vocab_size, size=(B, T)).astype(np.int32)
    labels = rng.integers(0, cfg.vocab_size, size=(B, T)).astype(np.int32)
    mask = rng.random((B, T)) > 0.4

    def whole_batch_loss(p):
        logits = forward(p, jnp.asarray(tokens), cfg)
        total, count = masked_cross_entropy(logits, jnp.asarray(labels), jnp.asarray(mask))
        return total / jnp.maximum(count, 1.0)

    ref_grads = jax.grad(whole_batch_loss)(trainer.params)

    micro = 2
    chunks = [(tokens[i:i + micro], labels[i:i + micro], mask[i:i + micro])
              for i in range(0, B, micro)]
    acc = jax.tree_util.tree_map(jnp.zeros_like, trainer.params)
    total, count = 0.0, 0.0
    for t, l, m in chunks:
        total_, count_ = masked_cross_entropy(forward(trainer.params, jnp.asarray(t), cfg),
                                              jnp.asarray(l), jnp.asarray(m))
        grads = jax.grad(lambda p: masked_cross_entropy(forward(p, jnp.asarray(t), cfg),
                                                        jnp.asarray(l), jnp.asarray(m))[0])(
            trainer.params)
        acc = jax.tree_util.tree_map(lambda a, b: a + b, acc, grads)
        total += float(total_)
        count += float(count_)
    acc = jax.tree_util.tree_map(lambda g: g / max(count, 1.0), acc)

    diffs = [float(np.abs(np.asarray(a) - np.asarray(b)).max())
             for a, b in zip(jax.tree_util.tree_leaves(ref_grads),
                             jax.tree_util.tree_leaves(acc))]
    rel = max(diffs) / max(1e-9, max(float(np.abs(np.asarray(g)).max())
                                     for g in jax.tree_util.tree_leaves(ref_grads)))
    assert rel < 1e-4, f"accumulated vs combined-batch gradient mismatch: {rel}"


def test_parameters_frozen_within_accumulation(tmp_path, monkeypatch, fake_tokenizer):
    """The compiled accumulation step must not mutate parameters."""
    import jax.numpy as jnp

    trainer = _make_trainer(tmp_path, monkeypatch, fake_tokenizer)
    cfg = trainer.model_cfg
    rng = np.random.default_rng(1)
    B, T = 6, 16
    batch = (jnp.asarray(rng.integers(0, cfg.vocab_size, size=(3, 2, T)).astype(np.int32)),
             jnp.asarray(rng.integers(0, cfg.vocab_size, size=(3, 2, T)).astype(np.int32)),
             jnp.asarray(rng.random((3, 2, T)) > 0.3),
             jnp.asarray(np.zeros((3, 2, T), dtype=np.int32)))
    before = [np.asarray(v).copy() for v in jax.tree_util.tree_leaves(trainer.params)]
    trainer._accum_grads(trainer.params, batch)
    after = [np.asarray(v) for v in jax.tree_util.tree_leaves(trainer.params)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))


def test_optimizer_state_updates_once_per_accumulation(tmp_path, monkeypatch, fake_tokenizer):
    """One accumulation step == exactly one optimizer update (count +1)."""
    trainer = _make_trainer(tmp_path, monkeypatch, fake_tokenizer)
    rng = np.random.default_rng(2)
    start = _opt_counts(trainer.opt_state)
    _accum_step(trainer, rng)
    after_one = _opt_counts(trainer.opt_state)
    assert [a - b for a, b in zip(after_one, start)] == [1, 1]
    _accum_step(trainer, rng)
    after_two = _opt_counts(trainer.opt_state)
    assert [a - b for a, b in zip(after_two, start)] == [2, 2]
