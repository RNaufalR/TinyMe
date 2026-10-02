"""Checkpoint integrity, compatibility and LR continuity on resume (P0-11/P0-12)."""
from __future__ import annotations

import json

import numpy as np
import pytest


def _opt_counts(state):
    from src.training.checkpoint import tree_to_flat

    return [int(np.asarray(v)) for k, v in tree_to_flat(state).items()
            if k.endswith("count") and np.asarray(v).ndim == 0]


def _do_updates(trainer, n, seed=0):
    import jax.numpy as jnp

    rng = np.random.default_rng(seed)
    cfg = trainer.model_cfg
    T = 16
    for _ in range(n):
        batch = (jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, 2, T)).astype(np.int32)),
                 jnp.asarray(rng.integers(0, cfg.vocab_size, size=(2, 2, T)).astype(np.int32)),
                 jnp.asarray(rng.random((2, 2, T)) > 0.3),
                 jnp.asarray(np.zeros((2, 2, T), dtype=np.int32)))
        grads, _, _ = trainer._accum_grads(trainer.params, batch)
        trainer.params, trainer.opt_state = trainer._apply_update(trainer.params, trainer.opt_state, grads)
        trainer.step += 1


def _tiny_trainer(tmp_path, monkeypatch, tokenizer, exp_id="TEST-CKPT"):
    from src.model import TinyMeConfig, init_params
    from src.training.trainer import TrainConfig, Trainer

    monkeypatch.setattr("src.training.trainer.CHECKPOINTS_DIR", tmp_path / "ckpt")
    monkeypatch.setattr("src.training.trainer.EXPERIMENTS_DIR", tmp_path / "exp")
    cfg = TrainConfig(experiment_id=exp_id, architecture="nano", seq_len=16,
                      micro_batch_size=2, grad_accum_steps=2, max_steps=6,
                      warmup_steps=2, learning_rate=1e-3, eval_every=100,
                      checkpoint_every=100, keep_last_checkpoints=2)
    model_cfg = TinyMeConfig(name="tiny", vocab_size=int(tokenizer.vocab_size), d_model=32,
                             n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
    return Trainer(cfg, model_cfg=model_cfg, tokenizer=tokenizer)


def test_checkpoint_round_trip(tmp_path, monkeypatch, fake_tokenizer):
    import jax

    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    trainer.step = 3
    trainer.best_val_loss = 1.25
    path = trainer.save("latest")
    before = [np.asarray(v).copy() for v in jax.tree_util.tree_leaves(trainer.params)]
    before_opt = [np.asarray(v).copy() for v in jax.tree_util.tree_leaves(trainer.opt_state)]

    # mutate in place, then resume
    trainer.params = jax.tree_util.tree_map(lambda x: x * 0, trainer.params)
    trainer.step = 0
    assert trainer.resume_from_checkpoint("latest") is True
    after = [np.asarray(v) for v in jax.tree_util.tree_leaves(trainer.params)]
    after_opt = [np.asarray(v) for v in jax.tree_util.tree_leaves(trainer.opt_state)]
    assert all(np.array_equal(a, b) for a, b in zip(before, after))
    assert all(np.array_equal(a, b) for a, b in zip(before_opt, after_opt))
    assert trainer.step == 3
    assert trainer.best_val_loss == pytest.approx(1.25)
    assert path.exists()


def test_lr_continuity_on_resume(tmp_path, monkeypatch, fake_tokenizer):
    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    _do_updates(trainer, 3)
    assert _opt_counts(trainer.opt_state) == [3, 3]
    trainer.save("latest")
    lr_before = trainer.lr_at(trainer.step)
    next_lr_before = trainer.lr_at(trainer.step + 1)

    trainer2 = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    assert trainer2.resume_from_checkpoint("latest")
    assert trainer2.step == trainer.step
    assert trainer2.lr_at(trainer2.step) == pytest.approx(lr_before)
    assert trainer2.lr_at(trainer2.step + 1) == pytest.approx(next_lr_before)
    # the optimizer's own schedule counter is restored, not reset to zero
    assert _opt_counts(trainer2.opt_state) == [3, 3]


def test_corrupted_checkpoint_is_detected(tmp_path, monkeypatch, fake_tokenizer):
    from src.training.checkpoint import verify_checkpoint

    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    trainer.save("latest")
    report = verify_checkpoint(trainer.ckpt_dir, "latest")
    assert report["ok"] is True

    path = trainer.ckpt_dir / "latest.safetensors"
    data = bytearray(path.read_bytes())
    data[-8:] = b"\x00" * 8
    path.write_bytes(bytes(data))
    report = verify_checkpoint(trainer.ckpt_dir, "latest")
    assert report["ok"] is False
    with pytest.raises(ValueError):
        trainer.resume_from_checkpoint("latest")


def test_incompatible_fingerprint_rejected(tmp_path, monkeypatch, fake_tokenizer):
    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    trainer.save("latest")
    # pretend the dataset changed
    monkeypatch.setattr("src.training.trainer.load_manifest", lambda v: {"dataset_fingerprint": "CHANGED"})
    with pytest.raises(ValueError, match="incompatibility"):
        trainer.resume_from_checkpoint("latest")


def test_atomic_write_preserves_previous_checkpoint(tmp_path, monkeypatch, fake_tokenizer):
    from src.training.checkpoint import save_checkpoint, sha256_file

    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    trainer.save("latest")
    good_sha = sha256_file(trainer.ckpt_dir / "latest.safetensors")

    # an interrupted write leaves a temporary file but never a torn final file
    tmp = trainer.ckpt_dir / ".latest.safetensors.tmp"
    tmp.write_bytes(b"garbage")
    assert sha256_file(trainer.ckpt_dir / "latest.safetensors") == good_sha
    assert trainer.resume_from_checkpoint("latest") is True

    # a corrupt *final* file must be refused instead of silently loaded
    (trainer.ckpt_dir / "latest.safetensors").write_bytes(b"garbage")
    with pytest.raises(ValueError):
        trainer.resume_from_checkpoint("latest")


def test_metadata_contains_required_fingerprints(tmp_path, monkeypatch, fake_tokenizer):
    trainer = _tiny_trainer(tmp_path, monkeypatch, fake_tokenizer)
    trainer.save("latest")
    meta = json.loads((trainer.ckpt_dir / "latest.json").read_text())
    for key in ("model_config_hash", "dataset_fingerprint", "tokenizer_hash", "experiment_id",
                "architecture", "seq_len", "step", "sampler_state", "rng_state", "environment"):
        assert key in meta, key
