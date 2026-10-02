"""compute_dtype must change real computation and be recorded (audit P0-16)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.model import forward, init_params
from src.training.trainer import TrainConfig, Trainer


def test_forward_accepts_every_declared_dtype(tiny_model_cfg, tiny_params):
    tokens = np.arange(8, dtype=np.int32).reshape(1, -1)
    outputs = {}
    for name in ("float32", "bfloat16", "float16"):
        logits = forward(tiny_params, tokens, tiny_model_cfg, dtype=name)
        assert logits.dtype == np.float32, "logits must always be returned in float32"
        assert np.isfinite(np.asarray(logits)).all(), name
        outputs[name] = np.asarray(logits)
    # bf16 truncates the computation: the result must *differ* from fp32, proving
    # the dtype really reaches the arithmetic instead of being cosmetic.
    assert not np.array_equal(outputs["float32"], outputs["bfloat16"])
    assert np.allclose(outputs["float32"], outputs["bfloat16"], atol=5e-2)


def test_unknown_dtype_is_rejected(tiny_model_cfg, tiny_params):
    import pytest

    with pytest.raises(ValueError):
        forward(tiny_params, np.zeros((1, 4), np.int32), tiny_model_cfg, dtype="float8")


def test_trainer_records_the_compute_dtype_in_config_and_checkpoint(tmp_path, monkeypatch, fake_tokenizer):
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(trainer_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    from src.model import TinyMeConfig

    cfg = TrainConfig(experiment_id="DTYPE-UNIT", compute_dtype="bfloat16", max_steps=1,
                      micro_batch_size=1, grad_accum_steps=1, seq_len=16)
    model_cfg = TinyMeConfig(name="unit-tiny", vocab_size=fake_tokenizer.vocab_size,
                             d_model=32, n_layers=2, n_heads=4, d_ff=64, max_seq_len=32)
    tr = Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)
    assert tr.cfg.compute_dtype == "bfloat16"
    tokens, labels, mask = _batch(tr.model_cfg, batch=1, seq=16)
    import jax.numpy as jnp

    batch = (jnp.asarray(tokens[None]), jnp.asarray(labels[None]),
             jnp.asarray(mask[None]), jnp.zeros((1, 1, 16), jnp.int32))
    grads, loss_sum, count_sum = tr._accum_grads(tr.params, batch)
    assert np.isfinite(float(loss_sum)) and float(count_sum) > 0
    tr.step = 1
    tr.save("latest")
    meta = json.loads((tr.ckpt_dir / "latest.json").read_text())
    assert meta["config"]["compute_dtype"] == "bfloat16"


def _batch(cfg, batch=2, seq=16):
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, (batch, seq)).astype(np.int32)
    labels = tokens.copy()
    mask = np.ones_like(tokens, dtype=bool)
    mask[:, -4:] = False
    labels[:, -4:] = -100
    return tokens, labels, mask
