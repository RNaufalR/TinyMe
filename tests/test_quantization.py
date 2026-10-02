"""Functional quantization tests (audit §10 / §16).

These exercise the real quantize/dequantize round trip on real model weights and
measure the *functional* consequences (logits and perplexity), not just the file
size.  A variant is only acceptable if the dequantized model still produces
finite, close logits — the release decision is made from measurements like these.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.model import TinyMeConfig, forward, init_params
from src.quantization.quantize import (dequantize_int4, dequantize_int8, flatten_params,
                                       quantize_int4, quantize_int8, quantize_to_fp16,
                                       quantized_size_bytes, unflatten_params)


@pytest.fixture(scope="module")
def params():
    cfg = TinyMeConfig(name="quant-tiny", vocab_size=96, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=32)
    return cfg, init_params(cfg, seed=3)


def test_int8_round_trip_is_close(params):
    _cfg, p = params
    flat = flatten_params(p)
    q, s = quantize_int8(flat)
    deq = dequantize_int8(q, s)
    for key, original in flat.items():
        got = deq[key]
        assert got.shape == original.shape, key
        if original.ndim == 2 and original.size >= 256:
            assert q[key].dtype == np.int8
            rel = np.abs(got - original).max() / max(np.abs(original).max(), 1e-9)
            assert rel < 0.02, f"int8 relative error too large for {key}: {rel}"


def test_int4_round_trip_is_close(params):
    _cfg, p = params
    flat = flatten_params(p)
    q, s = quantize_int4(flat)
    deq = dequantize_int4(q, s, shape_hint={k: v.shape for k, v in flat.items()})
    for key, original in flat.items():
        got = deq[key]
        assert got.shape == original.shape, key
        if original.ndim == 2 and original.size >= 512:
            assert q[key].dtype == np.uint8
            # 4-bit: an order of magnitude coarser than int8 but still bounded
            rel = np.abs(got - original).max() / max(np.abs(original).max(), 1e-9)
            assert rel < 0.20, f"int4 relative error too large for {key}: {rel}"


def test_quantized_payloads_shrink():
    cfg = TinyMeConfig(name="quant-size", vocab_size=256, d_model=64, n_layers=2, n_heads=4,
                       d_ff=128, max_seq_len=32)
    flat = flatten_params(init_params(cfg, seed=5))
    fp32 = sum(v.nbytes for v in flat.values())
    q8, s8 = quantize_int8(flat)
    q4, s4 = quantize_int4(flat)
    fp16 = sum(v.nbytes for v in quantize_to_fp16(flat).values())
    size8 = quantized_size_bytes(q8, s8)
    size4 = quantized_size_bytes(q4, s4)
    assert fp16 < fp32
    assert size8 < fp16, "int8 must be smaller than fp16"
    assert size4 < size8, "int4 must be smaller than int8"


def test_dequantized_model_still_runs_with_bounded_logit_drift(params):
    """A quantized model must still be a *functioning* model.

    On randomly initialised weights the logits carry no structure, so argmax
    agreement is not a meaningful threshold; what is measurable is that the
    forward pass stays finite and that the logit drift is bounded and ordered
    (fp16 < int8 < int4).  The trained-model functional comparison (perplexity,
    tool protocol validity) is measured separately by ``scripts/evaluate.py
    --variants`` (EXP-005) where the logits are structured.
    """
    cfg, p = params
    flat = flatten_params(p)
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(2, 16)).astype(np.int32)

    baseline = np.asarray(forward(p, tokens, cfg), dtype=np.float32)
    scale = float(np.abs(baseline).mean()) or 1.0

    drifts = {}
    for name, (q, s) in (("int8", quantize_int8(flat)), ("int4", quantize_int4(flat))):
        deq = (dequantize_int8(q, s) if name == "int8"
               else dequantize_int4(q, s, shape_hint={k: v.shape for k, v in flat.items()}))
        rebuilt = unflatten_params(deq)
        logits = np.asarray(forward(rebuilt, tokens, cfg), dtype=np.float32)
        assert np.all(np.isfinite(logits)), f"{name} produced non-finite logits"
        drifts[name] = float(np.abs(logits - baseline).mean() / scale)

    fp16_drift = float(np.abs(np.asarray(
        forward(unflatten_params({k: v.astype(np.float16).astype(np.float32)
                                  for k, v in flat.items()}), tokens, cfg), dtype=np.float32)
        - baseline).mean() / scale)
    # Upper bounds measured on randomly initialised weights (worst case: the
    # logits carry no structure and their mean magnitude is small).  Trained
    # weights drift far less; the trained-model numbers are the ones reported.
    assert drifts["int8"] < 0.10, drifts
    assert drifts["int4"] < 0.60, drifts
    assert fp16_drift <= drifts["int8"] <= drifts["int4"], (fp16_drift, drifts)


def test_int4_shape_hint_restores_original_shapes(params):
    _cfg, p = params
    flat = flatten_params(p)
    q, s = quantize_int4(flat)
    hint = {k: v.shape for k, v in flat.items()}
    deq = dequantize_int4(q, s, shape_hint=hint)
    for k, v in flat.items():
        assert deq[k].shape == v.shape
