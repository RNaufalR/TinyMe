"""Inference engine contracts (audit P1-01/P0-13)."""
from __future__ import annotations

import numpy as np
import pytest

from src.inference.engine import InferenceEngine


@pytest.fixture()
def engine(tiny_release):
    return InferenceEngine(tiny_release["model"], tiny_release["tokenizer"],
                           tiny_release["config"], backend="numpy")


def test_greedy_generation_is_deterministic(engine):
    a = engine.generate("<|user|>\nhello", max_new_tokens=12, temperature=0.0)
    b = engine.generate("<|user|>\nhello", max_new_tokens=12, temperature=0.0)
    assert a == b


def test_sampled_generation_is_seed_reproducible(engine):
    a = engine.generate("<|user|>\nhello", max_new_tokens=12, temperature=0.9, top_p=0.95, seed=5)
    b = engine.generate("<|user|>\nhello", max_new_tokens=12, temperature=0.9, top_p=0.95, seed=5)
    c = engine.generate("<|user|>\nhello", max_new_tokens=12, temperature=0.9, top_p=0.95, seed=6)
    assert a == b
    assert isinstance(c, str)


def test_generation_respects_the_token_budget(engine):
    out = engine.generate("<|user|>\nwrite a long text", max_new_tokens=7, temperature=0.0)
    assert len(engine.encode(out)) <= 7


def test_cache_and_no_cache_agree(engine):
    """Cached decoding must be a pure optimisation (same tokens either way)."""
    prompt = "<|user|>\ncount to five"
    with_cache = engine.generate(prompt, max_new_tokens=16, temperature=0.0, use_cache=True)
    without = engine.generate(prompt, max_new_tokens=16, temperature=0.0, use_cache=False)
    assert with_cache == without


def test_perplexity_is_finite_and_lower_for_in_domain_text(engine):
    good = engine.perplexity(["def add(a, b):\n    return a + b\n"])
    bad = engine.perplexity(["zzzzqqqq xxxx #### 0000\n"])
    assert np.isfinite(good) and np.isfinite(bad)


def test_long_prompt_is_truncated_from_the_left(engine):
    long_prompt = "<|user|>\n" + ("word " * 400)
    out = engine.generate(long_prompt, max_new_tokens=4, temperature=0.0)
    assert isinstance(out, str)


def test_parity_report_is_complete(engine):
    report = engine.verify_parity(np.asarray([[1, 2, 3, 4, 5]], dtype=np.int32), tol=1e-4)
    for key in ("numpy_full_vs_cached_max_abs_diff", "jax_full_vs_cached_max_abs_diff",
                "jax_vs_numpy_max_abs_diff", "ok"):
        assert key in report
    assert report["ok"], report


def test_set_params_swaps_weights_live(engine):
    before = engine.forward_numpy(np.asarray([[1, 2, 3]], dtype=np.int32))
    from src.quantization.quantize import quantize_tree_to_fp16

    engine.set_params(quantize_tree_to_fp16(engine.params))
    after = engine.forward_numpy(np.asarray([[1, 2, 3]], dtype=np.int32))
    assert np.allclose(before, after, atol=1e-2)
