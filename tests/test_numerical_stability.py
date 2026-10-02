"""Numerical stability: finite losses, NaN/Inf guards, clipping (audit P0-15)."""
from __future__ import annotations

import numpy as np
import pytest

from src.model import TinyMeConfig, forward, init_params
from src.training.trainer import TrainConfig, Trainer, masked_cross_entropy


def _batch(cfg, batch=2, seq=16, seed=0):
    tokens = np.random.default_rng(seed).integers(0, cfg.vocab_size, (batch, seq)).astype(np.int32)
    labels = tokens.copy()
    mask = np.ones_like(tokens, dtype=bool)
    mask[:, -4:] = False
    labels[:, -4:] = -100
    return tokens, labels, mask


def test_masked_loss_is_finite_for_extreme_logits():
    import jax.numpy as jnp

    logits = jnp.asarray(np.array([[[1e4, -1e4, 0.0], [1e4, 1e4, 1e4]]], dtype=np.float32))
    labels = jnp.asarray(np.array([[0, 2]], dtype=np.int32))
    mask = jnp.asarray(np.ones((1, 2), dtype=bool))
    total, count = masked_cross_entropy(logits, labels, mask)
    assert np.isfinite(float(total)) and np.isfinite(float(count))
    assert float(count) == 2.0


def test_loss_ignores_masked_positions_entirely():
    import jax.numpy as jnp

    logits = jnp.asarray(np.random.default_rng(0).normal(size=(1, 4, 8)).astype(np.float32))
    labels = jnp.asarray(np.array([[1, 2, 3, 4]], dtype=np.int32))
    full = masked_cross_entropy(logits, labels, jnp.ones((1, 4), bool))
    partial = masked_cross_entropy(logits, labels, jnp.asarray([[True, True, False, False]]))
    assert float(partial[1]) == 2.0
    assert not np.isclose(float(full[0]), float(partial[0]))


def test_trainer_skips_non_finite_updates_and_logs_a_non_finite_event(tmp_path, monkeypatch,
                                                                      fake_tokenizer):
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(trainer_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = TrainConfig(experiment_id="NAN-UNIT", max_steps=1, micro_batch_size=2,
                      grad_accum_steps=1, seq_len=16, eval_every=1, checkpoint_every=1)
    model_cfg = TinyMeConfig(name="nan-tiny", vocab_size=fake_tokenizer.vocab_size, d_model=32,
                             n_layers=1, n_heads=4, d_ff=64, max_seq_len=32)
    tr = Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)
    poisoned = dict(tr.params)
    poisoned["ln_f"] = tr.params["ln_f"] * np.nan   # poison the final norm
    tr.params = poisoned

    tokens, labels, mask = _batch(model_cfg, batch=2, seq=16)
    data = {"input_ids": tokens, "labels": labels, "loss_mask": mask,
            "doc_ids": np.zeros_like(tokens, dtype=np.int32)}
    before = np.asarray(tr.params["tok_emb"]).copy()
    summary = tr.fit(data, data, resume=False)
    assert summary["steps_executed"] == 1
    assert any(e.get("event") == "NON_FINITE" for e in tr.events), tr.events
    assert np.allclose(np.asarray(tr.params["tok_emb"]), before, equal_nan=True)
    assert summary["non_finite_steps"] == 1


def test_gradient_clipping_bounds_the_update(tmp_path, monkeypatch, fake_tokenizer):
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(trainer_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    cfg = TrainConfig(experiment_id="CLIP-UNIT", max_steps=1, micro_batch_size=2,
                      grad_accum_steps=1, seq_len=16, grad_clip_norm=0.5, learning_rate=0.1,
                      warmup_steps=1)
    model_cfg = TinyMeConfig(name="clip-tiny", vocab_size=fake_tokenizer.vocab_size, d_model=32,
                             n_layers=1, n_heads=4, d_ff=64, max_seq_len=32)
    tr = Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)
    import jax.numpy as jnp

    tokens, labels, mask = _batch(model_cfg)
    batch = (jnp.asarray(tokens[None]), jnp.asarray(labels[None]),
             jnp.asarray(mask[None]), jnp.zeros((1, 2, 16), jnp.int32))
    grads, _, _ = tr._accum_grads(tr.params, batch)
    import optax

    raw_norm = float(optax.global_norm(grads))
    assert raw_norm > cfg.grad_clip_norm, "test needs a gradient large enough to be clipped"
    clipped, _ = optax.clip_by_global_norm(cfg.grad_clip_norm).update(grads, None)
    assert abs(float(optax.global_norm(clipped)) - cfg.grad_clip_norm) < 1e-4

    new_params, _ = tr._apply_update(tr.params, tr.opt_state, grads)
    delta = float(np.max(np.abs(np.asarray(new_params["tok_emb"]) - np.asarray(tr.params["tok_emb"]))))
    # AdamW's normalised update is bounded by ~learning_rate per coordinate
    # regardless of the gradient magnitude — that is the stability guarantee.
    assert delta <= cfg.learning_rate * 1.01, (raw_norm, delta)


def test_forward_stays_finite_in_low_precision(tiny_model_cfg, tiny_params):
    tokens = np.random.default_rng(0).integers(0, tiny_model_cfg.vocab_size, (2, 32)).astype(np.int32)
    for dtype in ("bfloat16", "float16"):
        logits = np.asarray(forward(tiny_params, tokens, tiny_model_cfg, dtype=dtype))
        assert np.isfinite(logits).all(), dtype
        assert np.abs(logits).max() < 1e4, dtype
