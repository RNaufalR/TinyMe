"""RNG synchronisation and reproducibility (audit P0-13)."""
from __future__ import annotations

import numpy as np

from src.model import TinyMeConfig, init_params
from src.training.trainer import TrainConfig, Trainer
from src.utils.rng import RNGService, set_global_seeds


def test_python_numpy_jax_streams_are_reproducible_and_independent():
    a, b = RNGService(1234), RNGService(1234)
    assert a.save_state() == b.save_state()
    assert a.numpy_generator("sampler").random() == b.numpy_generator("sampler").random()

    keys_a = [np.asarray(k) for k in (a.jax_key("init"), a.jax_key("drop"))]
    keys_b = [np.asarray(k) for k in (b.jax_key("init"), b.jax_key("drop"))]
    assert all(np.array_equal(x, y) for x, y in zip(keys_a, keys_b))
    assert not np.array_equal(keys_a[0], keys_a[1]), "distinct streams must give distinct keys"

    c = RNGService(999)
    assert not np.array_equal(np.asarray(c.jax_key("init")), keys_a[0])


def test_global_seeding_covers_python_numpy_and_data():
    import random

    np.random.seed(0)
    set_global_seeds(7)
    first_random, first_np = random.random(), np.random.random()
    set_global_seeds(7)
    assert random.random() == first_random
    assert np.random.random() == first_np


def test_sampler_orders_are_seed_dependent_and_stable():
    from src.training.sampler import EpochSampler

    a = EpochSampler(500, 32, seed=5).epoch_order(2)
    b = EpochSampler(500, 32, seed=5).epoch_order(2)
    c = EpochSampler(500, 32, seed=6).epoch_order(2)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert not np.array_equal(EpochSampler(500, 32, seed=5).epoch_order(3), a)


def test_state_roundtrip_resumes_the_same_stream():
    svc = RNGService(42)
    _ = svc.numpy_generator("data").random(5)
    saved = svc.save_state()
    expected = svc.numpy_generator("data").random(5)
    resumed = RNGService(0)
    resumed.load_state(saved)
    assert resumed.save_state() == saved
    assert np.allclose(resumed.numpy_generator("data").random(5), expected)


def test_two_training_runs_with_the_same_seed_produce_identical_weights(tmp_path, monkeypatch,
                                                                        fake_tokenizer):
    from src.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "CHECKPOINTS_DIR", tmp_path / "checkpoints")
    monkeypatch.setattr(trainer_mod, "EXPERIMENTS_DIR", tmp_path / "experiments")
    model_cfg = TinyMeConfig(name="rng-tiny", vocab_size=fake_tokenizer.vocab_size, d_model=32,
                             n_layers=1, n_heads=4, d_ff=64, max_seq_len=32)
    data = _data(model_cfg, n=8)
    finals = []
    for run in range(2):
        cfg = TrainConfig(experiment_id=f"RNG-{run}", max_steps=3, micro_batch_size=2,
                          grad_accum_steps=2, seq_len=32, seed=2026, eval_every=10,
                          checkpoint_every=10, warmup_steps=1)
        tr = Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)
        tr.fit(data, data, resume=False)
        finals.append({k: np.asarray(v).copy() if not isinstance(v, list)
                       else [np.asarray(b["ln1"]).copy() for b in v] for k, v in tr.params.items()})
    assert np.array_equal(finals[0]["tok_emb"], finals[1]["tok_emb"])

    cfg = TrainConfig(experiment_id="RNG-diff", max_steps=3, micro_batch_size=2,
                      grad_accum_steps=2, seq_len=32, seed=99, eval_every=10,
                      checkpoint_every=10, warmup_steps=1)
    tr = Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)
    tr.fit(data, data, resume=False)
    assert not np.array_equal(np.asarray(tr.params["tok_emb"]), finals[0]["tok_emb"])


def _data(cfg, n=8, seq=32, seed=3):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, cfg.vocab_size, (n, seq)).astype(np.int32)
    labels = tokens.copy()
    mask = np.ones_like(tokens, dtype=bool)
    labels[:, -6:] = -100
    mask[:, -6:] = False
    return {"input_ids": tokens, "labels": labels, "loss_mask": mask,
            "doc_ids": np.zeros_like(tokens, dtype=np.int32)}
