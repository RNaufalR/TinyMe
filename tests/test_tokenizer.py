"""Tokenizer: special tokens, vocab compatibility, train-only rule (P0-07/P0-17)."""
from __future__ import annotations

import numpy as np
import pytest


def test_special_tokens_present(fake_tokenizer):
    from src.tokenizer.bpe import SPECIAL_TOKENS

    for tok in SPECIAL_TOKENS:
        assert fake_tokenizer.tok.token_to_id(tok) is not None, tok
    for tok in ("<|tool_call|>", "<|tool_result|>", "<|final|>", "<|endtool_call|>",
                "<|endtool_result|>"):
        assert tok in SPECIAL_TOKENS


def test_zero_unknown_rate(fake_tokenizer):
    ids = fake_tokenizer.encode_ids("def f(x):\n    return x → 42 ✓")
    unk = fake_tokenizer.tok.token_to_id("<|unk|>")
    assert unk not in ids


def test_vocab_match_and_mismatch(fake_tokenizer):
    from src.model import TinyMeConfig, assert_vocab_compatible

    cfg = TinyMeConfig(name="x", vocab_size=fake_tokenizer.vocab_size)
    assert_vocab_compatible(cfg, fake_tokenizer.vocab_size)
    with pytest.raises(ValueError, match="vocabulary mismatch"):
        assert_vocab_compatible(cfg, fake_tokenizer.vocab_size + 1)


def test_tokenizer_hash_is_stable(fake_tokenizer, tmp_path):
    path = fake_tokenizer.save(tmp_path / "tok.json")
    h1 = fake_tokenizer.sha256(path)
    h2 = fake_tokenizer.sha256(path)
    assert h1 == h2 and len(h1) == 64


def test_train_tokenizer_rejects_non_train_corpus():
    from src.tokenizer.bpe import train_tokenizer

    with pytest.raises(ValueError, match="train split"):
        train_tokenizer(["hello world"], vocab_size=64, corpus_role="test")


def test_roundtrip_encode_decode(fake_tokenizer):
    text = "def add(a, b):\n    return a + b"
    ids = fake_tokenizer.encode_ids(text)
    assert fake_tokenizer.decode(ids) == text
