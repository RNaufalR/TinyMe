"""TinyMe transformer core (corrected).

Fixes relative to the audited baseline:
  * **RoPE pairing fixed** (P0-14): the angle vector is the *paired*
    representation (``repeat_interleave``), so interleaved pairs ``(x0,x1),
    (x2,x3), …`` use frequencies ``f0, f1, …`` exactly as the 2-D rotation
    definition requires. The baseline concatenated ``[f, f]``, which paired
    ``(x0,x1)`` and ``(x2,x3)`` both with ``f0``.
  * **Document-boundary attention masking** for packed sequences (P0-08).
  * **Real compute dtype** (P0-16): activations/logits are computed in the
    requested dtype; weights stay float32 master parameters.
  * **No silent vocabulary mismatch** (P0-17): ``assert_vocab_compatible``.
  * Numerically stable softmax using ``jax.nn.softmax`` with ``-inf`` fill.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import random

MODEL_CONFIGS: dict[str, dict[str, Any]] = {
    "nano": dict(vocab_size=4096, d_model=192, n_layers=4, n_heads=6, d_ff=512, max_seq_len=512),
    "base": dict(vocab_size=4096, d_model=320, n_layers=6, n_heads=8, d_ff=896, max_seq_len=512),
    "medium": dict(vocab_size=8192, d_model=384, n_layers=8, n_heads=8, d_ff=1024, max_seq_len=512),
}

DTYPE_NAMES = {"float32": jnp.float32, "bfloat16": jnp.bfloat16, "float16": jnp.float16}


@dataclass
class TinyMeConfig:
    name: str = "base"
    vocab_size: int = 4096
    d_model: int = 320
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 896
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    dropout: float = 0.0

    @classmethod
    def from_name(cls, name: str) -> "TinyMeConfig":
        if name not in MODEL_CONFIGS:
            raise KeyError(f"unknown architecture {name!r}; have {sorted(MODEL_CONFIGS)}")
        return cls(name=name, **MODEL_CONFIGS[name])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def model_hash(self) -> str:
        import hashlib

        payload = f"{self.name}|{self.vocab_size}|{self.d_model}|{self.n_layers}|{self.n_heads}|{self.d_ff}|{self.max_seq_len}|{self.rope_theta}|{self.rms_norm_eps}|{self.tie_word_embeddings}"
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def parameter_count(self) -> int:
        """Analytical parameter count (validates the measured ``count_parameters``)."""
        return int(estimate_sizes(self)["total_parameters"])

    def fp32_bytes(self) -> int:
        return int(estimate_sizes(self)["fp32_bytes"])

    def assert_valid(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError(f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")
        head_dim = self.d_model // self.n_heads
        if head_dim % 2:
            raise ValueError(f"head_dim ({head_dim}) must be even for RoPE")


ModelConfig = TinyMeConfig


def assert_vocab_compatible(cfg: TinyMeConfig, tokenizer_vocab_size: int, context: str = "") -> None:
    """Fail loudly if model and tokenizer vocabularies disagree (P0-17)."""
    if int(cfg.vocab_size) != int(tokenizer_vocab_size):
        raise ValueError(
            f"vocabulary mismatch{' (' + context + ')' if context else ''}: "
            f"model vocab_size={cfg.vocab_size} tokenizer vocab_size={tokenizer_vocab_size}")


# ------------------------------------------------------------------ init
def init_params(cfg: TinyMeConfig, seed: int = 0) -> dict[str, Any]:
    cfg.assert_valid()
    rng = random.PRNGKey(seed)
    params: dict[str, Any] = {}
    k = random.split(rng, 4 * cfg.n_layers + 4)
    idx = 0

    params["tok_emb"] = (random.normal(k[idx], (cfg.vocab_size, cfg.d_model)) * 0.02).astype(jnp.float32)
    idx += 1

    params["blocks"] = []
    for _ in range(cfg.n_layers):
        blk: dict[str, Any] = {}
        blk["ln1"] = jnp.ones((cfg.d_model,), dtype=jnp.float32)
        blk["ln2"] = jnp.ones((cfg.d_model,), dtype=jnp.float32)
        blk["attn"] = {
            "q": random.normal(k[idx], (cfg.d_model, cfg.d_model)) / math.sqrt(cfg.d_model),
            "k": random.normal(k[idx + 1], (cfg.d_model, cfg.d_model)) / math.sqrt(cfg.d_model),
            "v": random.normal(k[idx + 2], (cfg.d_model, cfg.d_model)) / math.sqrt(cfg.d_model),
            "o": random.normal(k[idx + 3], (cfg.d_model, cfg.d_model)) / math.sqrt(cfg.d_model),
        }
        idx += 4
        blk["mlp"] = {
            "gate": random.normal(k[idx], (cfg.d_model, cfg.d_ff)) / math.sqrt(cfg.d_model),
            "up": random.normal(k[idx + 1], (cfg.d_model, cfg.d_ff)) / math.sqrt(cfg.d_model),
            "down": random.normal(k[idx + 2], (cfg.d_ff, cfg.d_model)) / math.sqrt(cfg.d_model),
        }
        idx += 3
        params["blocks"].append(blk)

    params["ln_f"] = jnp.ones((cfg.d_model,), dtype=jnp.float32)
    if not cfg.tie_word_embeddings:
        params["lm_head"] = random.normal(k[idx], (cfg.d_model, cfg.vocab_size)) * 0.02
    return params


def count_parameters(params: dict[str, Any]) -> int:
    return int(sum(int(np.prod(v.shape)) for v in jax.tree_util.tree_leaves(params)))


def estimate_sizes(cfg: TinyMeConfig) -> dict[str, int]:
    d, L, V, ff = cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.d_ff
    emb = V * d
    attn = L * 4 * d * d
    mlp = L * 3 * d * ff
    norms = L * 2 * d + d
    head = 0 if cfg.tie_word_embeddings else V * d
    total = emb + attn + mlp + norms + head
    return {
        "embedding_params": emb, "attention_params": attn, "mlp_params": mlp,
        "norm_params": norms, "lm_head_params": head, "total_parameters": total,
        "fp32_bytes": total * 4, "fp16_bytes": total * 2, "int4_bytes": total // 2 + total // 32,
    }


# ------------------------------------------------------------------ ops
def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    xf = x.astype(jnp.float32)
    normed = xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)
    return normed.astype(x.dtype) * weight


def rope_frequencies(dim: int, max_seq: int, theta: float = 10000.0) -> jnp.ndarray:
    """Angle table ``[max_seq, dim]`` in the **paired** representation.

    Row ``t`` is ``[w0, w0, w1, w1, …, w_{dim/2-1}, w_{dim/2-1}]`` so that the
    interleaved rotation applied by :func:`apply_rope` (which slices ``0::2`` /
    ``1::2``) rotates pair ``(x_{2i}, x_{2i+1})`` by ``t * w_i``.
    """
    if dim % 2:
        raise ValueError("RoPE requires an even head dimension")
    half = dim // 2
    inv = 1.0 / (theta ** (jnp.arange(0, half, dtype=jnp.float32) / half))
    pos = jnp.arange(max_seq, dtype=jnp.float32)
    freqs = jnp.outer(pos, inv)                     # [T, half]
    return jnp.repeat(freqs, 2, axis=-1)            # [T, dim] paired


def apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    """Rotate interleaved pairs of ``x`` ([..., T, D]) by the angle table."""
    cos = jnp.cos(freqs)[None, None, :, :].astype(x.dtype)
    sin = jnp.sin(freqs)[None, None, :, :].astype(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = jnp.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos + rot * sin


def build_attention_mask(T: int, doc_ids: jnp.ndarray | None = None) -> jnp.ndarray:
    """Causal mask, optionally restricted to same-document positions."""
    causal = jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]
    if doc_ids is None:
        return causal
    same_doc = doc_ids[:, None, :] == doc_ids[:, :, None]      # [B, 1, T, T]
    return causal & same_doc[:, None, :, :]


def attention(x: jnp.ndarray, w: dict[str, jnp.ndarray], n_heads: int,
              freqs: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    B, T, D = x.shape
    hd = D // n_heads
    q = (x @ w["q"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    k = (x @ w["k"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    v = (x @ w["v"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    q, k = apply_rope(q, freqs), apply_rope(k, freqs)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(hd)
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
    attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ w["o"]


def swiglu(x: jnp.ndarray, w: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return (jax.nn.silu(x @ w["gate"]) * (x @ w["up"])) @ w["down"]


def causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]


def _dtype(dtype: Any) -> Any:
    if dtype is None:
        return jnp.float32
    if isinstance(dtype, str):
        if dtype not in DTYPE_NAMES:
            raise ValueError(f"unsupported compute dtype {dtype!r}; have {sorted(DTYPE_NAMES)}")
        return DTYPE_NAMES[dtype]
    return dtype


def forward(params: dict[str, Any], tokens: jnp.ndarray, cfg: TinyMeConfig,
            dtype: Any = None, doc_ids: jnp.ndarray | None = None) -> jnp.ndarray:
    """tokens: [B, T] int32 -> logits [B, T, V] (float32 logits)."""
    cfg.assert_valid()
    dt = _dtype(dtype)
    B, T = tokens.shape
    if T > cfg.max_seq_len:
        raise ValueError(f"sequence length {T} exceeds max_seq_len {cfg.max_seq_len}")
    x = params["tok_emb"][tokens].astype(dt)
    freqs = rope_frequencies(cfg.d_model // cfg.n_heads, T, cfg.rope_theta)
    mask = build_attention_mask(T, None if doc_ids is None else jnp.asarray(doc_ids))
    for blk in params["blocks"]:
        w = {k: v.astype(dt) for k, v in blk["attn"].items()}
        h = rms_norm(x, blk["ln1"].astype(dt), cfg.rms_norm_eps)
        x = x + attention(h, w, cfg.n_heads, freqs, mask).astype(dt)
        h = rms_norm(x, blk["ln2"].astype(dt), cfg.rms_norm_eps)
        mw = {k: v.astype(dt) for k, v in blk["mlp"].items()}
        x = x + swiglu(h, mw).astype(dt)
    x = rms_norm(x, params["ln_f"].astype(dt), cfg.rms_norm_eps)
    if cfg.tie_word_embeddings:
        logits = x.astype(jnp.float32) @ params["tok_emb"].astype(jnp.float32).T
    else:
        logits = x.astype(jnp.float32) @ params["lm_head"].astype(jnp.float32)
    return logits


# ------------------------------------------------------- KV-cached decoding
def init_cache(cfg: TinyMeConfig, batch_size: int = 1) -> dict[str, Any]:
    hd = cfg.d_model // cfg.n_heads
    return {
        "k": [jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, hd)) for _ in range(cfg.n_layers)],
        "v": [jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, hd)) for _ in range(cfg.n_layers)],
        "pos": 0,
    }


def prefill_cache(params: dict[str, Any], tokens: jnp.ndarray, cache: dict[str, Any],
                  cfg: TinyMeConfig, dtype: Any = None) -> jnp.ndarray:
    """Prefill the cache with a prompt; returns logits for every position.

    Runs exactly the same layer stack as :func:`forward` (the previous version
    re-used the layer-0 input for every layer, which broke cache parity).
    """
    cfg.assert_valid()
    dt = _dtype(dtype)
    B, T = tokens.shape
    hd = cfg.d_model // cfg.n_heads
    x = params["tok_emb"][tokens].astype(dt)
    freqs = rope_frequencies(hd, T, cfg.rope_theta)
    mask = build_attention_mask(T)
    for li, blk in enumerate(params["blocks"]):
        h = rms_norm(x, blk["ln1"].astype(dt), cfg.rms_norm_eps)
        q = (h @ blk["attn"]["q"].astype(dt)).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        k = (h @ blk["attn"]["k"].astype(dt)).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        v = (h @ blk["attn"]["v"].astype(dt)).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        q, k = apply_rope(q, freqs), apply_rope(k, freqs)
        cache["k"][li] = cache["k"][li].at[:, :, :T, :].set(k)
        cache["v"][li] = cache["v"][li].at[:, :, :T, :].set(v)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(dt)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
        x = x + (out @ blk["attn"]["o"].astype(dt)).astype(dt)
        h = rms_norm(x, blk["ln2"].astype(dt), cfg.rms_norm_eps)
        mw = {k2: v2.astype(dt) for k2, v2 in blk["mlp"].items()}
        x = x + swiglu(h, mw).astype(dt)
    cache["pos"] = T
    x = rms_norm(x, params["ln_f"].astype(dt), cfg.rms_norm_eps)
    if cfg.tie_word_embeddings:
        return x.astype(jnp.float32) @ params["tok_emb"].astype(jnp.float32).T
    return x.astype(jnp.float32) @ params["lm_head"].astype(jnp.float32)


def forward_with_cache(params: dict[str, Any], token: jnp.ndarray, cache: dict[str, Any],
                       cfg: TinyMeConfig, dtype: Any = None) -> jnp.ndarray:
    """Single-token cached decode step. Returns logits for the last position."""
    T = 1
    dt = _dtype(dtype)
    x = params["tok_emb"][token].astype(dt)
    hd = cfg.d_model // cfg.n_heads
    pos = int(cache["pos"])
    freqs = rope_frequencies(hd, pos + 1, cfg.rope_theta)[pos:pos + 1]
    for li, blk in enumerate(params["blocks"]):
        w = {k: v.astype(dt) for k, v in blk["attn"].items()}
        h = rms_norm(x, blk["ln1"].astype(dt), cfg.rms_norm_eps)
        q = (h @ w["q"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        k_new = (h @ w["k"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        v_new = (h @ w["v"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        q, k_new = apply_rope(q, freqs), apply_rope(k_new, freqs)
        cache["k"][li] = cache["k"][li].at[:, :, pos:pos + 1, :].set(k_new)
        cache["v"][li] = cache["v"][li].at[:, :, pos:pos + 1, :].set(v_new)
        k = cache["k"][li][:, :, :pos + 1, :]
        v = cache["v"][li][:, :, :pos + 1, :]
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        attn = jax.nn.softmax(scores.astype(jnp.float32), axis=-1).astype(scores.dtype)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(-1, T, cfg.d_model)
        x = x + (out @ w["o"]).astype(dt)
        h = rms_norm(x, blk["ln2"].astype(dt), cfg.rms_norm_eps)
        mw = {k: v.astype(dt) for k, v in blk["mlp"].items()}
        x = x + swiglu(h, mw).astype(dt)
    cache["pos"] = pos + 1
    x = rms_norm(x, params["ln_f"].astype(dt), cfg.rms_norm_eps)
    if cfg.tie_word_embeddings:
        return x.astype(jnp.float32) @ params["tok_emb"].astype(jnp.float32).T
    return x.astype(jnp.float32) @ params["lm_head"].astype(jnp.float32)
