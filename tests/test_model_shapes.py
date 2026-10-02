"""Shape/parameter-count contracts for the transformer (audit §13)."""
from __future__ import annotations

import numpy as np
import pytest

from src.model import (MODEL_CONFIGS, TinyMeConfig, assert_vocab_compatible, count_parameters,
                       estimate_sizes, forward, init_cache, init_params, prefill_cache)


def test_forward_shapes(tiny_model_cfg, tiny_params):
    tokens = np.zeros((2, 16), dtype=np.int32)
    logits = forward(tiny_params, tokens, tiny_model_cfg)
    assert logits.shape == (2, 16, tiny_model_cfg.vocab_size)
    assert logits.dtype == np.float32


def test_parameter_counts_are_measured_not_guessed(tiny_model_cfg, tiny_params):
    measured = count_parameters(tiny_params)
    estimated = estimate_sizes(tiny_model_cfg)["total_parameters"]
    assert measured == estimated, "count_parameters must agree with the analytical estimate"


def test_official_configs_match_the_audit_estimates():
    """Nano ≈ 2.56 M and Base ≈ 8.93 M parameters (audit §13)."""
    nano, base = TinyMeConfig.from_name("nano"), TinyMeConfig.from_name("base")
    assert 2.5e6 < nano.parameter_count() < 2.7e6, nano.parameter_count()
    assert 8.8e6 < base.parameter_count() < 9.1e6, base.parameter_count()
    assert nano.fp32_bytes() < 16 * 1024 * 1024
    # the measured parameter count for the real initialisation must match
    assert count_parameters(init_params(nano, seed=0)) == nano.parameter_count()


def test_sequence_length_guard(tiny_model_cfg, tiny_params):
    too_long = np.zeros((1, tiny_model_cfg.max_seq_len + 1), dtype=np.int32)
    with pytest.raises(ValueError):
        forward(tiny_params, too_long, tiny_model_cfg)


def test_vocab_mismatch_is_rejected_loudly(tiny_model_cfg):
    with pytest.raises(ValueError):
        assert_vocab_compatible(tiny_model_cfg, tiny_model_cfg.vocab_size + 1, context="unit-test")


def test_doc_ids_block_cross_document_attention(tiny_model_cfg, tiny_params):
    """Two packed documents must not attend across the boundary (§13)."""
    tokens = np.tile(np.arange(8, dtype=np.int32), (1, 2))[:, :16]
    doc_ids = np.concatenate([np.zeros((1, 8), np.int32), np.ones((1, 8), np.int32)], axis=1)
    combined = forward(tiny_params, tokens, tiny_model_cfg, doc_ids=doc_ids)
    second_only = forward(tiny_params, tokens[:, 8:], tiny_model_cfg)
    assert np.allclose(combined[0, 8:], second_only[0], atol=1e-5)


def test_kv_cache_shapes_match_full_forward(tiny_model_cfg, tiny_params):
    tokens = np.arange(12, dtype=np.int32).reshape(1, -1) % tiny_model_cfg.vocab_size
    cache = init_cache(tiny_model_cfg, batch_size=1)
    logits = prefill_cache(tiny_params, tokens, cache, tiny_model_cfg)
    assert logits.shape == (1, 12, tiny_model_cfg.vocab_size)
    assert cache["pos"] == 12
    assert cache["k"][0].shape[2] >= 12


def test_config_validation_rejects_bad_head_geometry():
    with pytest.raises(ValueError):
        TinyMeConfig(name="bad", vocab_size=64, d_model=30, n_layers=1, n_heads=4, d_ff=32).assert_valid()


def test_model_hash_is_stable_and_sensitive():
    a = TinyMeConfig.from_name("nano")
    b = TinyMeConfig.from_name("nano")
    c = TinyMeConfig(**{**a.to_dict(), "d_model": a.d_model + 1})
    assert a.model_hash() == b.model_hash()
    assert a.model_hash() != c.model_hash()
