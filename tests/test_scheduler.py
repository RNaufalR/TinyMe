"""Learning-rate schedule (audit §16: scheduler coverage).

Checked against an independent closed-form reference — not by calling the
implementation twice.  The schedule must warm up linearly, decay on a cosine to
the configured floor, and stay a pure function of the *global* step so a resume
never restarts warm-up (audit §27).
"""
from __future__ import annotations

import math

import pytest

from src.training.trainer import TrainConfig, Trainer


@pytest.fixture(scope="module")
def trainer(fake_tokenizer, tiny_model_cfg):
    import dataclasses

    cfg = TrainConfig(experiment_id="TEST-SCHEDULER", learning_rate=3e-4, warmup_steps=40,
                      max_steps=320, min_lr_ratio=0.1, dataset_version="unit")
    # the model config must accept the unit tokenizer's vocab size
    model_cfg = dataclasses.replace(tiny_model_cfg, vocab_size=int(fake_tokenizer.vocab_size))
    return Trainer(cfg, model_cfg=model_cfg, tokenizer=fake_tokenizer)


def _closed_form(step: int, *, base=3e-4, warmup=40, total=320, floor=0.1) -> float:
    """Independent reference, including the trainer's warm-up clamp of 10% of the run."""
    warmup = max(1, min(warmup, total // 10))
    if step <= warmup:
        return base * step / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress)))


def test_warmup_is_monotonic_and_matches_the_reference(trainer):
    values = [trainer.lr_at(s) for s in range(0, 41)]
    # the schedule must not dip during warm-up (the first decay step after the
    # peak is numerically ~equal, so allow float32 slack)
    assert all(b >= a * (1 - 1e-3) for a, b in zip(values, values[1:])), values[:8]
    for step in (0, 7, 20, 39, 40):
        assert trainer.lr_at(step) == pytest.approx(_closed_form(step), rel=1e-5), step


def test_decay_is_monotone_and_reaches_the_floor(trainer):
    values = [trainer.lr_at(s) for s in range(40, 321, 20)]
    assert all(b <= a * (1 + 1e-5) for a, b in zip(values, values[1:]))
    assert trainer.lr_at(320) == pytest.approx(3e-5, rel=1e-4)


def test_schedule_is_a_pure_function_of_the_global_step(trainer):
    assert trainer.lr_at(123) == trainer.lr_at(123)
    assert trainer.lr_at(123) > trainer.lr_at(124)


def test_resume_does_not_restart_warmup(trainer):
    for step in (120, 200, 319):
        assert trainer.lr_at(step) == pytest.approx(_closed_form(step), rel=1e-5)
    # a restarted warm-up would sit at the peak (~3e-4) at step 120; the real
    # schedule is already decaying but above the floor
    assert 3e-5 < trainer.lr_at(120) < 0.9 * 3e-4
