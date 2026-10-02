"""NumPy reference engine vs JAX implementation parity (audit P1-01)."""
from __future__ import annotations

import numpy as np

from src.model import forward, init_cache, prefill_cache
from src.inference.engine import InferenceEngine


def test_numpy_engine_matches_jax_forward(tiny_release):
    engine = InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                             tiny_release["config"], backend="numpy")
    tokens = np.asarray([[1, 5, 9, 12, 13, 20, 21, 30]], dtype=np.int32)
    np_logits = engine.forward_numpy(tokens)
    np_params = {k: np.asarray(v) for k, v in engine.params.items()}
    jax_logits = np.asarray(forward(np_params, tokens, engine.config))
    assert np_logits.shape == jax_logits.shape
    assert np.allclose(np_logits, jax_logits, atol=1e-4), np.abs(np_logits - jax_logits).max()


def test_cached_prefill_matches_full_forward_literally(tiny_release):
    engine = InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                             tiny_release["config"], backend="numpy")
    tokens = np.asarray([[1, 7, 11, 15, 19, 23]], dtype=np.int32)
    full = engine.forward_numpy(tokens)
    cache = init_cache(engine.config, batch_size=1)
    prefill = np.asarray(prefill_cache({k: np.asarray(v) for k, v in engine.params.items()},
                                       tokens, cache, engine.config))
    # Both implementations run the same operations; float32 associativity means the
    # NumPy and JAX orderings differ at ~1e-7.  The *bit-for-bit* requirement applies
    # to JAX-vs-JAX (see tests/test_kv_cache.py).
    assert np.allclose(full, prefill, atol=1e-5)
    parity = engine.verify_parity(tokens, tol=1e-4)
    assert parity["ok"], parity
    assert parity["jax_cache_parity_pass"], parity
    assert parity["jax_numpy_parity_pass"], parity
    assert parity["numpy_cache_parity_pass"], parity


def test_engine_forward_dispatch_agrees_across_backends(tiny_release):
    numpy_engine = InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                                   tiny_release["config"], backend="numpy")
    jax_engine = InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                                 tiny_release["config"], backend="jax")
    tokens = np.asarray([[2, 3, 4, 5]], dtype=np.int32)
    a, b = numpy_engine.forward(tokens), np.asarray(jax_engine.forward(tokens))
    assert np.allclose(a, b, atol=1e-4)


def test_weight_dtype_variants_keep_shapes(tiny_release):
    from src.quantization.quantize import (dequantize_tree_int4, dequantize_tree_int8,
                                           quantize_tree_int4, quantize_tree_int8,
                                           quantize_tree_to_fp16)

    engine = InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                             tiny_release["config"], backend="numpy")
    tokens = np.asarray([[1, 2, 3, 4, 5, 6]], dtype=np.int32)
    base = engine.forward_numpy(tokens)
    fp16 = quantize_tree_to_fp16(engine.params)
    q8, s8 = quantize_tree_int8(engine.params)
    q4, s4, hint = quantize_tree_int4(engine.params)
    for name, params in (("fp16", fp16), ("int8", dequantize_tree_int8(q8, s8)),
                         ("int4", dequantize_tree_int4(q4, s4, hint))):
        engine.set_params(params)
        got = engine.forward_numpy(tokens)
        assert got.shape == base.shape, name
        assert np.isfinite(got).all(), name
        assert np.allclose(got, base, atol=2.0), f"{name} diverged beyond tolerance"
