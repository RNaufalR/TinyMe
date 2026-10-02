"""Full-forward vs cached-decode parity (P0-14 / P1-01 regression test)."""
from __future__ import annotations

import numpy as np
import pytest

TOL = 5e-4


@pytest.fixture(scope="module")
def payload():
    from src.model import TinyMeConfig, init_params

    cfg = TinyMeConfig(name="cache-test", vocab_size=96, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=32)
    params = init_params(cfg, seed=3)
    tokens = np.random.default_rng(0).integers(0, cfg.vocab_size, size=(1, 14)).astype(np.int32)
    return cfg, params, tokens


def test_cached_logits_match_full_forward(payload):
    import jax.numpy as jnp

    from src.model import forward, forward_with_cache, init_cache, prefill_cache

    cfg, params, tokens = payload
    full = np.asarray(forward(params, jnp.asarray(tokens), cfg))
    cache = init_cache(cfg, 1)
    prefill = np.asarray(prefill_cache(params, jnp.asarray(tokens[:, :8]), cache, cfg))
    assert np.abs(prefill - full[:, :8]).max() < TOL

    step_logits = []
    for t in range(8, tokens.shape[1]):
        step_logits.append(np.asarray(forward_with_cache(
            params, jnp.asarray(tokens[:, t:t + 1]), cache, cfg))[0, -1])
    step_logits = np.stack(step_logits)
    assert np.abs(step_logits - full[0, 8:]).max() < TOL
    assert np.array_equal(step_logits.argmax(-1), full[0, 8:].argmax(-1))


def test_cache_positions_are_absolute(payload):
    """Decoding after a prefill uses absolute positions, not 0-based re-indexing."""
    import jax.numpy as jnp

    from src.model import forward, forward_with_cache, init_cache, prefill_cache

    cfg, params, tokens = payload
    full = np.asarray(forward(params, jnp.asarray(tokens), cfg))
    cache = init_cache(cfg, 1)
    prefill_cache(params, jnp.asarray(tokens[:, :12]), cache, cfg)
    logits = np.asarray(forward_with_cache(params, jnp.asarray(tokens[:, 12:13]), cache, cfg))[0, -1]
    assert np.abs(logits - full[0, 12]).max() < TOL


def test_numpy_cached_matches_numpy_full():
    import tempfile
    """The standalone (release) numpy path must agree with its own full forward."""
    import importlib.util
    import pathlib

    # scratch space must stay outside the repository (tmp_path fixture)
    root = pathlib.Path(tempfile.mkdtemp(prefix="tinyme-kv-cache-"))
    from src.inference.engine import InferenceEngine
    from src.model import TinyMeConfig, init_params
    from safetensors.numpy import save_file

    from src.tokenizer.bpe import SPECIAL_TOKENS, TinyMeTokenizer, build_tokenizer
    from tokenizers import pre_tokenizers, trainers

    tok = build_tokenizer(96)
    trainer = trainers.BpeTrainer(vocab_size=96, min_frequency=1, special_tokens=SPECIAL_TOKENS,
                                  show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    trainer_texts = [" ".join(SPECIAL_TOKENS)] * 4
    tok.train_from_iterator(trainer_texts, trainer=trainer, length=len(trainer_texts))
    vocab_size = tok.get_vocab_size()
    cfg = TinyMeConfig(name="np-cache", vocab_size=vocab_size, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=32)
    params = init_params(cfg, seed=5)
    flat = {}
    for i, blk in enumerate(params["blocks"]):
        for sub in ("attn", "mlp"):
            for k, v in blk[sub].items():
                flat[f"blocks/{i}/{sub}/{k}"] = np.asarray(v, dtype=np.float32)
        flat[f"blocks/{i}/ln1"] = np.asarray(blk["ln1"], dtype=np.float32)
        flat[f"blocks/{i}/ln2"] = np.asarray(blk["ln2"], dtype=np.float32)
    flat["tok_emb"] = np.asarray(params["tok_emb"], dtype=np.float32)
    flat["ln_f"] = np.asarray(params["ln_f"], dtype=np.float32)
    tmp = pathlib.Path(root) / "checkpoints" / "_test_np"
    tmp.mkdir(parents=True, exist_ok=True)
    st = tmp / "model.safetensors"
    save_file(flat, str(st))
    (tmp / "config.json").write_text(__import__("json").dumps(cfg.to_dict()))
    tok_path = tmp / "tokenizer.json"
    TinyMeTokenizer(tok).save(tok_path)

    engine = InferenceEngine(st, tok_path, tmp / "config.json", backend="numpy")
    tokens = np.random.default_rng(9).integers(0, vocab_size, size=(1, 10)).astype(np.int32)
    full = engine.forward_numpy(tokens)
    logits, cache = engine.prefill_numpy(tokens[:, :7])
    assert np.abs(logits - full[:, :7]).max() < TOL
    step = engine.decode_step_numpy(tokens[:, 7:8], cache)
    assert np.abs(step[0, -1] - full[0, 7]).max() < TOL
