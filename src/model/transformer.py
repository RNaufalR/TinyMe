"""TinyMe-Reasoner: a Pre-Norm decoder-only Transformer in pure JAX.

Design (see docs/ARCHITECTURE_DECISION.md):
  * RMSNorm
  * Rotary Position Embeddings (RoPE, theta=10000)
  * Causal multi-head self-attention (bias-free q/k/v/o)
  * SwiGLU feed-forward network (bias-free)
  * Tied input embedding / output LM head

Pure functional style: parameters are a nested dict of jnp arrays, which makes
safetensors export, INT8/INT4 quantization, and pure-NumPy inference trivial.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
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
        return cls(name=name, **MODEL_CONFIGS[name])

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


ModelConfig = TinyMeConfig  # backwards-compatible alias


# ------------------------------------------------------------------ init
def _linear(rng, in_dim: int, out_dim: int) -> tuple[jnp.ndarray, dict[str, Any]]:
    # Fan-in scaled normal init (bias-free linear layers).
    std = 1.0 / math.sqrt(in_dim)
    w = random.normal(rng, (in_dim, out_dim), dtype=jnp.float32) * std
    return w, {}


def init_params(cfg: TinyMeConfig, seed: int = 0) -> dict[str, Any]:
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
    total = 0
    for v in jax.tree_util.tree_leaves(params):
        total += int(np.prod(v.shape))
    return total


def estimate_sizes(cfg: TinyMeConfig) -> dict[str, int]:
    """Analytic parameter count + serialized sizes in bytes."""
    d, L, V, ff = cfg.d_model, cfg.n_layers, cfg.vocab_size, cfg.d_ff
    emb = V * d
    attn = L * 4 * d * d
    mlp = L * 3 * d * ff
    norms = L * 2 * d + d
    head = 0 if cfg.tie_word_embeddings else V * d
    total = emb + attn + mlp + norms + head
    return {
        "embedding_params": emb,
        "attention_params": attn,
        "mlp_params": mlp,
        "norm_params": norms,
        "lm_head_params": head,
        "total_parameters": total,
        "fp32_bytes": total * 4,
        "fp16_bytes": total * 2,
        "int8_bytes": total + (total // cfg.d_model if cfg.tie_word_embeddings else 0) * 0,
        "int4_bytes": total // 2,
    }


# ------------------------------------------------------------------ ops
def rms_norm(x: jnp.ndarray, weight: jnp.ndarray, eps: float = 1e-5) -> jnp.ndarray:
    return x * jax.lax.rsqrt(jnp.mean(jnp.square(x), axis=-1, keepdims=True) + eps) * weight


def rope_frequencies(dim: int, max_seq: int, theta: float = 10000.0) -> jnp.ndarray:
    inv = 1.0 / (theta ** (jnp.arange(0, dim, 2, dtype=jnp.float32) / dim))
    pos = jnp.arange(max_seq, dtype=jnp.float32)
    freqs = jnp.outer(pos, inv)                      # [T, dim/2]
    return jnp.concatenate([freqs, freqs], axis=-1)  # [T, dim]


def apply_rope(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    # x: [B, H, T, D]; freqs: [T, D]
    cos = jnp.cos(freqs)[None, None, :, :]
    sin = jnp.sin(freqs)[None, None, :, :]
    x1, x2 = x[..., 0::2], x[..., 1::2]
    rot = jnp.stack([-x2, x1], axis=-1).reshape(x.shape)
    return x * cos + rot * sin


def attention(x: jnp.ndarray, w: dict[str, jnp.ndarray], n_heads: int,
              freqs: jnp.ndarray, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    B, T, D = x.shape
    hd = D // n_heads
    q = (x @ w["q"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    k = (x @ w["k"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    v = (x @ w["v"]).reshape(B, T, n_heads, hd).transpose(0, 2, 1, 3)
    q, k = apply_rope(q, freqs), apply_rope(k, freqs)
    scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
    if mask is not None:
        scores = jnp.where(mask, scores, -1e10)
    scores = scores - jnp.max(scores, axis=-1, keepdims=True)
    attn = jnp.exp(scores)
    attn = attn / (jnp.sum(attn, axis=-1, keepdims=True) + 1e-9)
    out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
    return out @ w["o"]


def swiglu(x: jnp.ndarray, w: dict[str, jnp.ndarray]) -> jnp.ndarray:
    return (jax.nn.silu(x @ w["gate"]) * (x @ w["up"])) @ w["down"]


def causal_mask(T: int) -> jnp.ndarray:
    return jnp.tril(jnp.ones((T, T), dtype=bool))[None, None, :, :]


def forward(params: dict[str, Any], tokens: jnp.ndarray, cfg: TinyMeConfig) -> jnp.ndarray:
    """tokens: [B, T] int32 -> logits [B, T, V]."""
    B, T = tokens.shape
    x = params["tok_emb"][tokens]
    freqs = rope_frequencies(cfg.d_model // cfg.n_heads, T, cfg.rope_theta)
    mask = causal_mask(T)
    for blk in params["blocks"]:
        h = rms_norm(x, blk["ln1"], cfg.rms_norm_eps)
        x = x + attention(h, blk["attn"], cfg.n_heads, freqs, mask)
        h = rms_norm(x, blk["ln2"], cfg.rms_norm_eps)
        x = x + swiglu(h, blk["mlp"])
    x = rms_norm(x, params["ln_f"], cfg.rms_norm_eps)
    if cfg.tie_word_embeddings:
        return x @ params["tok_emb"].T
    return x @ params["lm_head"]


# ------------------------------------------------------- KV-cached decoding
def init_cache(cfg: TinyMeConfig, batch_size: int = 1) -> dict[str, Any]:
    return {
        "k": [jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, cfg.d_model // cfg.n_heads)) for _ in range(cfg.n_layers)],
        "v": [jnp.zeros((batch_size, cfg.n_heads, cfg.max_seq_len, cfg.d_model // cfg.n_heads)) for _ in range(cfg.n_layers)],
        "pos": 0,
    }


def forward_with_cache(params: dict[str, Any], token: jnp.ndarray, cache: dict[str, Any],
                       cfg: TinyMeConfig) -> jnp.ndarray:
    """Single-token step using a KV cache. Returns logits for the last position."""
    T = 1
    x = params["tok_emb"][token]
    hd = cfg.d_model // cfg.n_heads
    pos = cache["pos"]
    freqs = rope_frequencies(hd, cfg.max_seq_len, cfg.rope_theta)[pos:pos + 1]
    mask = jnp.ones((1, 1, 1, pos + 1), dtype=bool)
    for li, blk in enumerate(params["blocks"]):
        h = rms_norm(x, blk["ln1"], cfg.rms_norm_eps)
        q = (h @ blk["attn"]["q"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        k_new = (h @ blk["attn"]["k"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        v_new = (h @ blk["attn"]["v"]).reshape(-1, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
        q, k_new = apply_rope(q, freqs), apply_rope(k_new, freqs)
        cache["k"][li] = cache["k"][li].at[:, :, pos:pos + 1, :].set(k_new)
        cache["v"][li] = cache["v"][li].at[:, :, pos:pos + 1, :].set(v_new)
        k = cache["k"][li][:, :, :pos + 1, :]
        v = cache["v"][li][:, :, :pos + 1, :]
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        scores = jnp.where(mask, scores, -1e10)
        attn = jax.nn.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(-1, T, cfg.d_model)
        x = x + out @ blk["attn"]["o"]
        h = rms_norm(x, blk["ln2"], cfg.rms_norm_eps)
        x = x + swiglu(h, blk["mlp"])
    cache["pos"] = pos + 1
    x = rms_norm(x, params["ln_f"], cfg.rms_norm_eps)
    if cfg.tie_word_embeddings:
        return x @ params["tok_emb"].T
    return x @ params["lm_head"]
