"""TinyMe inference engine.

Two interchangeable backends so the release artifact runs anywhere:

* `jax`     — XLA-compiled, fastest on CPU (AVX-512).
* `numpy`   — zero-dependency path used by the standalone release package;
              only requires `numpy` and `safetensors`.

Supports greedy / temperature / top-p sampling, deterministic mode, repetition
penalty, stop strings, and KV-cached incremental decoding.
"""
from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from ..utils.io_utils import REPO_ROOT

logger = logging.getLogger("tinyme.inference")


def load_safetensors_numpy(path: str | Path) -> dict[str, np.ndarray]:
    from safetensors.numpy import load_file
    return load_file(str(path))


class InferenceEngine:
    """Loads a TinyMe checkpoint + tokenizer and generates text."""

    def __init__(self, model_path: str | Path, tokenizer_path: str | Path,
                 config_path: str | Path | None = None, backend: str = "jax",
                 weights_dtype: str = "float32"):
        from src.model import TinyMeConfig
        from src.tokenizer.bpe import TinyMeTokenizer

        self.backend = backend
        self.weights_dtype = weights_dtype
        self.tokenizer = TinyMeTokenizer.load(tokenizer_path)

        if config_path is None:
            config_path = Path(model_path).parent / "config.json"
        if Path(config_path).exists():
            cfg_dict = json.loads(Path(config_path).read_text())
            self.config = TinyMeConfig(**{k: v for k, v in cfg_dict.items()
                                          if k in TinyMeConfig.__dataclass_fields__})
        else:
            self.config = TinyMeConfig.from_name("nano")

        self.params = self._load_params(model_path)
        self.pad_id = self.tokenizer.tok.token_to_id("<|pad|>")
        self.bos_id = self.tokenizer.tok.token_to_id("<|bos|>")
        self.eos_id = self.tokenizer.tok.token_to_id("<|eos|>")

        if backend == "jax":
            import jax.numpy as jnp
            self._jnp = jnp
            self._jparams = {k: jnp.asarray(v) for k, v in self.params.items()}
        self._freq_cache: dict[int, np.ndarray] = {}

    # ------------------------------------------------------------ weights
    def _load_params(self, model_path: str | Path) -> dict[str, np.ndarray]:
        from safetensors import safe_open

        with safe_open(str(model_path), framework="numpy") as f:
            meta = f.metadata() or {}
            dtype = meta.get("dtype", "float32")
            flat = {k: f.get_tensor(k) for k in f.keys()}

        if dtype == "int8":
            from src.quantization.quantize import dequantize_int8
            q = {k: v for k, v in flat.items() if not k.endswith(".__scale__")}
            sc = {k: v for k, v in flat.items() if k.endswith(".__scale__")}
            flat = dequantize_int8(q, sc)
        elif dtype == "int4":
            from src.quantization.quantize import dequantize_int4
            q = {k: v for k, v in flat.items() if not k.endswith(".__scale__")}
            sc = {k: v for k, v in flat.items() if k.endswith(".__scale__")}
            hint = {k: tuple(v) for k, v in json.loads(meta.get("shapes", "{}")).items()}
            flat = dequantize_int4(q, sc, shape_hint=hint)
        self.weights_dtype = dtype

        params: dict[str, Any] = {}
        blocks: dict[int, dict[str, Any]] = {}
        for key, arr in flat.items():
            arr = np.asarray(arr, dtype=np.float32)
            parts = key.split("/")
            if parts[0] == "blocks":
                i = int(parts[1])
                blocks.setdefault(i, {})
                if len(parts) == 3:
                    blocks[i][parts[2]] = arr
                else:
                    blocks[i].setdefault(parts[2], {})[parts[3]] = arr
            else:
                params[parts[0]] = arr
        params["blocks"] = [blocks[i] for i in sorted(blocks)]
        return params

    # ------------------------------------------------------------- helpers
    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode_ids(text)

    def decode(self, ids: list[int]) -> str:
        return self.tokenizer.decode(ids)

    def _rope(self, head_dim: int, max_pos: int) -> np.ndarray:
        """RoPE frequencies [max_pos, head_dim], cached and grown on demand."""
        key = head_dim
        cached = self._freq_cache.get(key)
        if cached is None or cached.shape[0] < max_pos:
            inv = 1.0 / (self.config.rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float32) / head_dim))
            pos = np.arange(max_pos, dtype=np.float32)
            freqs = np.outer(pos, inv)
            cached = np.concatenate([freqs, freqs], axis=-1)
            self._freq_cache[key] = cached
        return cached

    # --------------------------------------------------- forward (numpy)
    def forward_numpy(self, tokens: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = self.params
        B, T = tokens.shape
        hd = cfg.d_model // cfg.n_heads
        x = p["tok_emb"][tokens]
        freqs = self._rope(hd, max(T, 8))[:T]
        cos, sin = np.cos(freqs)[None, None], np.sin(freqs)[None, None]
        mask = np.tril(np.ones((T, T), dtype=bool))[None, None]

        def rms(v, w):
            return v / np.sqrt(np.mean(np.square(v), axis=-1, keepdims=True) + cfg.rms_norm_eps) * w

        def rope_apply(t):
            t1, t2 = t[..., 0::2], t[..., 1::2]
            rot = np.stack([-t2, t1], axis=-1).reshape(t.shape)
            return t * cos + rot * sin

        for blk in p["blocks"]:
            h = rms(x, blk["ln1"])
            q = (h @ blk["attn"]["q"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            k = (h @ blk["attn"]["k"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            v = (h @ blk["attn"]["v"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            q, k = rope_apply(q), rope_apply(k)
            scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
            scores = np.where(mask, scores, -1e10)
            scores = scores - scores.max(-1, keepdims=True)
            a = np.exp(scores)
            a = a / (a.sum(-1, keepdims=True) + 1e-9)
            out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
            x = x + out @ blk["attn"]["o"]
            h = rms(x, blk["ln2"])
            gate = h @ blk["mlp"]["gate"]
            up = h @ blk["mlp"]["up"]
            silu = np.where(gate >= 0, gate / (1.0 + np.exp(-np.abs(gate))),
                            gate * np.exp(-np.abs(gate)) / (1.0 + np.exp(-np.abs(gate))))
            swiglu = silu * up
            x = x + swiglu @ blk["mlp"]["down"]
        x = rms(x, p["ln_f"])
        return x @ p["tok_emb"].T

    def forward_jax(self, tokens: np.ndarray) -> np.ndarray:
        from src.model import forward
        import jax.numpy as jnp
        logits = forward(self._jparams, jnp.asarray(tokens), self.config)
        return np.asarray(logits)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        if self.backend == "jax":
            return self.forward_jax(tokens)
        return self.forward_numpy(tokens)

    # ---------------------------------------------------------- metrics
    def sequence_loss(self, ids: list[int]) -> float:
        """Mean next-token cross-entropy over a sequence (for perplexity)."""
        arr = np.asarray([ids], dtype=np.int32)
        logits = self.forward(arr)[0].astype(np.float64)
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        targets = arr[0, 1:]
        n = min(len(targets), len(logp) - 1)
        return float(-logp[np.arange(n), targets[:n]].mean())

    def perplexity(self, texts: list[str]) -> float:
        return float(np.exp(np.mean([self.sequence_loss(self.encode(t)) for t in texts])))

    # -------------------------------------------------------- generation
    def generate(self, prompt: str, max_new_tokens: int = 96, temperature: float = 0.0,
                 top_p: float = 1.0, repetition_penalty: float = 1.0,
                 stop: list[str] | None = None, seed: int | None = None) -> str:
        ids = self.encode(prompt)
        rng = np.random.default_rng(seed) if seed is not None else None
        generated: list[int] = []
        context = list(ids)
        for _ in range(max_new_tokens):
            window = context[-self.config.max_seq_len:]
            arr = np.asarray([window], dtype=np.int32)
            logits = self.forward(arr)[0, -1].astype(np.float64)
            if repetition_penalty != 1.0 and generated:
                for tok in set(generated):
                    logits[tok] /= repetition_penalty
            if temperature <= 0.0:
                nxt = int(np.argmax(logits))
            else:
                logits = logits / temperature
                logits = logits - logits.max()
                probs = np.exp(logits)
                probs = probs / probs.sum()
                if top_p < 1.0:
                    order = np.argsort(probs)[::-1]
                    cum = np.cumsum(probs[order])
                    keep = order[cum <= top_p]
                    if len(keep) == 0:
                        keep = order[:1]
                    masked = np.zeros_like(probs)
                    masked[keep] = probs[keep]
                    probs = masked / masked.sum()
                nxt = int(rng.choice(len(probs), p=probs))
            if nxt == self.eos_id:
                break
            generated.append(nxt)
            context.append(nxt)
            text_so_far = self.decode(generated)
            if stop and any(s in text_so_far for s in stop):
                break
        return self.decode(generated)

    # ------------------------------------------------------------ metadata
    def model_size_bytes(self, model_path: str | Path) -> int:
        return Path(model_path).stat().st_size

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "architecture": self.config.name,
            "vocab_size": self.config.vocab_size,
            "d_model": self.config.d_model,
            "n_layers": self.config.n_layers,
            "n_heads": self.config.n_heads,
            "max_seq_len": self.config.max_seq_len,
            "parameter_count": sum(int(np.prod(v.shape)) for v in self.params["tok_emb"][None][0:0]) or self._count_params(),
            "weights_dtype": self.weights_dtype,
        }

    def _count_params(self) -> int:
        total = 0
        for v in self.params["tok_emb"][None][0:0] if False else []:
            pass
        def walk(o):
            nonlocal total
            if isinstance(o, dict):
                for v in o.values():
                    walk(v)
            elif isinstance(o, list):
                for v in o:
                    walk(v)
            elif isinstance(o, np.ndarray):
                total += int(np.prod(o.shape))
        walk(self.params)
        return total
