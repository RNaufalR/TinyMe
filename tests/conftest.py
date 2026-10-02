"""Shared pytest fixtures. Tests run with the repository root on sys.path."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def tiny_model_cfg():
    from src.model import TinyMeConfig

    return TinyMeConfig(name="test-tiny", vocab_size=128, d_model=32, n_layers=2,
                        n_heads=4, d_ff=64, max_seq_len=64)


@pytest.fixture(scope="session")
def tiny_params(tiny_model_cfg):
    from src.model import init_params

    return init_params(tiny_model_cfg, seed=0)


@pytest.fixture(scope="session")
def fake_tokenizer(tmp_path_factory):
    """A small but real byte-level BPE tokenizer trained on toy data."""
    from src.tokenizer.bpe import SPECIAL_TOKENS, TinyMeTokenizer, build_tokenizer
    from tokenizers import trainers
    from tokenizers import pre_tokenizers

    texts = [
        "def add(a, b):\n    return a + b\n",
        "def mul(a, b):\n    return a * b\n",
        "<|system|>\nYou are TinyMe.\n<|user|>\nCompute 2 + 3.\n<|assistant|>\n5\n",
        "the quick brown fox jumps over the lazy dog",
    ] * 4
    tok = build_tokenizer(vocab_size=512)
    trainer = trainers.BpeTrainer(vocab_size=512, min_frequency=1, special_tokens=SPECIAL_TOKENS,
                                  show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    return TinyMeTokenizer(tok, name="tok-test")


def make_record(**overrides):
    from src.data.records import make_segment_record, Segment

    base = dict(
        segments=[Segment("user", "Compute 2 + 3.", target=False),
                  Segment("assistant", "5", target=True)],
        category="math", source="synthetic", source_id="unit/1", template_id="unit",
    )
    base.update(overrides)
    return make_segment_record(**base)


@pytest.fixture
def sample_record():
    return make_record()


@pytest.fixture
def tiny_release(tmp_path):
    """Write a tiny, self-consistent release (weights + tokenizer + config)."""
    import json

    import numpy as np
    from safetensors.numpy import save_file
    from tokenizers import pre_tokenizers, trainers

    from src.model import TinyMeConfig, init_params
    from src.tokenizer.bpe import SPECIAL_TOKENS, TinyMeTokenizer, build_tokenizer

    tok = build_tokenizer(96)
    trainer = trainers.BpeTrainer(vocab_size=96, min_frequency=1, special_tokens=SPECIAL_TOKENS,
                                  show_progress=False,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet())
    texts = [" ".join(SPECIAL_TOKENS), "def f(x):\n    return x + 1"] * 3
    tok.train_from_iterator(texts, trainer=trainer, length=len(texts))
    vocab_size = tok.get_vocab_size()
    cfg = TinyMeConfig(name="unit-tiny", vocab_size=vocab_size, d_model=32, n_layers=2,
                       n_heads=4, d_ff=64, max_seq_len=32)
    params = init_params(cfg, seed=11)

    flat: dict[str, np.ndarray] = {"tok_emb": np.asarray(params["tok_emb"], np.float32),
                                   "ln_f": np.asarray(params["ln_f"], np.float32)}
    for i, blk in enumerate(params["blocks"]):
        flat[f"blocks/{i}/ln1"] = np.asarray(blk["ln1"], np.float32)
        flat[f"blocks/{i}/ln2"] = np.asarray(blk["ln2"], np.float32)
        for sub in ("attn", "mlp"):
            for k, v in blk[sub].items():
                flat[f"blocks/{i}/{sub}/{k}"] = np.asarray(v, np.float32)
    model_path = tmp_path / "model_fp32.safetensors"
    save_file(flat, str(model_path))
    (tmp_path / "config.json").write_text(json.dumps(cfg.to_dict()))
    tok_path = tmp_path / "tokenizer.json"
    TinyMeTokenizer(tok).save(tok_path)
    return {"model": model_path, "tokenizer": tok_path, "config": tmp_path / "config.json",
            "cfg": cfg, "params": params, "vocab_size": vocab_size, "dir": tmp_path}


def masked_batch_inputs(cfg, batch_size=2, seq_len=16, seed=0):
    rng = np.random.default_rng(seed)
    tokens = rng.integers(0, cfg.vocab_size, size=(batch_size, seq_len)).astype(np.int32)
    labels = tokens.copy()
    mask = np.ones_like(tokens, dtype=bool)
    mask[:, -4:] = False
    labels[:, -4:] = -100
    return tokens, labels, mask


@pytest.fixture(scope="session")
def packed_dataset():
    """Build a small dataset with the *real* builder and return its version name.

    Packed shards are regenerable and git-ignored, so a fresh clone (and CI) has
    no shards. The shard-first loading tests must not silently skip (audit §17)
    and must not assert against an artefact CI cannot have, so this fixture runs
    ``scripts/prepare_data_v2.py`` — the production pipeline, including license,
    preprocessing, quality, dedup, split and packing stages — over a small scale
    and returns the version it built. The build is skipped when the recorded
    ``shards_fingerprint`` still matches the shards on disk.
    """
    import json
    import subprocess

    from src.data import dataset_api
    from src.data.dataset_api import shard_fingerprint

    version = "ci_unit"
    manifest_path = ROOT / "datasets" / "versions" / version / "manifest.json"
    shard_dir = dataset_api.PROCESSED_DIR / version / "shards"
    if manifest_path.exists() and shard_dir.exists():
        recorded = json.loads(manifest_path.read_text(encoding="utf-8")).get("shards_fingerprint")
        if recorded and shard_fingerprint(shard_dir) == recorded:
            return version
    subprocess.run([sys.executable, "scripts/prepare_data_v2.py", "--version", version,
                    "--scale", "0.05", "--seq-len", "64"],
                   cwd=ROOT, check=True, capture_output=True)
    return version
