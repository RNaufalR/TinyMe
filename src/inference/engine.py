"""TinyMe inference engine (corrected, P0-13/P1-01/P1-02/P1-03/P0-14).

Two interchangeable backends:
  * ``jax``    — XLA-compiled, uses the shared model module (single source of
                 truth for RoPE/attention semantics);
  * ``numpy``  — standalone release path (numpy + safetensors only), with an
                 explicit cached decoder whose numbers match the JAX path.

Generation uses **prefill + KV-cached decode** instead of re-running the full
context for every token. Sampling is deterministic when ``seed`` is omitted
(seed 0), nucleus sampling keeps the smallest prefix whose cumulative mass
reaches ``top_p``, and the repetition penalty is sign-aware.
"""
from __future__ import annotations

import json
import logging
import math
import time
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
                 weights_dtype: str | None = None):
        from ..model import TinyMeConfig, assert_vocab_compatible
        from ..tokenizer.bpe import TinyMeTokenizer

        self.backend = backend
        self.tokenizer = TinyMeTokenizer.load(tokenizer_path)
        config_path = Path(config_path) if config_path else Path(model_path).parent / "config.json"
        if Path(config_path).exists():
            cfg_dict = json.loads(Path(config_path).read_text())
            self.config = TinyMeConfig(**{k: v for k, v in cfg_dict.items()
                                          if k in TinyMeConfig.__dataclass_fields__})
        else:
            self.config = TinyMeConfig.from_name("nano")
        assert_vocab_compatible(self.config, int(self.tokenizer.vocab_size), context="inference load")
        self.params = self._load_params(model_path)
        self.weights_dtype = weights_dtype or "float32"

        self.pad_id = int(self.tokenizer.tok.token_to_id("<|pad|>"))
        self.bos_id = int(self.tokenizer.tok.token_to_id("<|bos|>"))
        self.eos_id = int(self.tokenizer.tok.token_to_id("<|eos|>"))
        self._jparams = None
        if backend == "jax":
            import jax.numpy as jnp

            self._jnp = jnp
            self._jparams = self._to_jax(self.params)

    # ------------------------------------------------------------ parameter IO
    @staticmethod
    def _to_jax(params):
        """Recursively convert a nested NumPy parameter tree to a JAX pytree.

        A naive ``{k: jnp.asarray(v)}`` fails on the ``blocks`` list-of-dicts
        (that was a real defect: the JAX inference backend could not load any
        checkpoint).
        """
        import jax.numpy as jnp

        if isinstance(params, dict):
            return {k: InferenceEngine._to_jax(v) for k, v in params.items()}
        if isinstance(params, (list, tuple)):
            return [InferenceEngine._to_jax(v) for v in params]
        return jnp.asarray(params)

    def set_params(self, params: dict[str, Any]) -> None:
        """Replace weights in both the NumPy and JAX representations."""
        self.params = params
        if self.backend == "jax":
            self._jparams = self._to_jax(params)

    @classmethod
    def from_params(cls, params: dict[str, Any], tokenizer_path: str | Path,
                    cfg=None, backend: str = "numpy") -> "InferenceEngine":
        """Build an engine from in-memory weights (used for quantized variants)."""
        from ..tokenizer.bpe import TinyMeTokenizer
        from ..model import TinyMeConfig, assert_vocab_compatible

        self = cls.__new__(cls)
        self.backend = backend
        self.tokenizer = TinyMeTokenizer.load(tokenizer_path)
        self.config = cfg or TinyMeConfig.from_name("nano")
        assert_vocab_compatible(self.config, int(self.tokenizer.vocab_size), context="from_params")
        self.params = params
        self.weights_dtype = "float32"
        self.pad_id = int(self.tokenizer.tok.token_to_id("<|pad|>"))
        self.bos_id = int(self.tokenizer.tok.token_to_id("<|bos|>"))
        self.eos_id = int(self.tokenizer.tok.token_to_id("<|eos|>"))
        self._jparams = None
        if backend == "jax":
            import jax.numpy as jnp
            self._jnp = jnp
            self._jparams = self._to_jax(params)
        return self

    # ------------------------------------------------------------ weights
    @staticmethod
    def _normalize_key(key: str) -> str:
        """``[blocks]/[0]/[ln1]`` (jax tree path) -> ``blocks/0/ln1``."""
        return "/".join(part.strip("[]") for part in key.split("/"))

    @staticmethod
    def _decode_flat(flat: dict[str, np.ndarray], dtypes: dict[str, str]) -> dict[str, np.ndarray]:
        """Cast tensors back to the dtypes recorded at save time (bf16-safe)."""
        out: dict[str, np.ndarray] = {}
        for key, value in flat.items():
            want = dtypes.get(key, str(value.dtype))
            if want in ("bfloat16", "float16"):
                try:
                    out[key] = np.asarray(value).astype(np.float32)
                except TypeError:                      # ml_dtypes not installed
                    out[key] = (np.asarray(value).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            elif want == "bool":
                out[key] = np.asarray(value).astype(bool)
            else:
                out[key] = np.asarray(value, dtype=np.float32)
        return out

    def _load_params(self, model_path: str | Path) -> dict[str, np.ndarray]:
        from safetensors import safe_open

        with safe_open(str(model_path), framework="numpy") as f:
            meta = f.metadata() or {}
            dtype = meta.get("dtype", "float32")
            flat = {k: f.get_tensor(k) for k in f.keys()}

        if any(k.startswith("params/") for k in flat):
            # ``tinyme-ckpt-v2`` training checkpoint: ``params/<jax tree path>``
            # plus optimizer state.  Strip the prefix and the bracket notation,
            # then restore the stored dtypes.  (Without this the engine built a
            # bogus top level ``{"params": ..., "opt": ...}`` tree and every
            # forward pass raised ``KeyError: 'tok_emb'``.)
            dtypes = {self._normalize_key(k): v for k, v in (meta.get("param_dtypes") or {}).items()}
            params_flat = {self._normalize_key(k[len("params/"):]): v
                           for k, v in flat.items() if k.startswith("params/")}
            flat = self._decode_flat(params_flat, dtypes)
            dtype = "float32"

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

    def _rope_table(self, T: int) -> np.ndarray:
        """Paired RoPE table matching :func:`src.model.transformer.rope_frequencies`."""
        head_dim = self.config.d_model // self.config.n_heads
        half = head_dim // 2
        inv = 1.0 / (self.config.rope_theta ** (np.arange(0, half, dtype=np.float32) / half))
        pos = np.arange(T, dtype=np.float32)
        freqs = np.outer(pos, inv)                # [T, half]
        return np.repeat(freqs, 2, axis=-1)       # [T, head_dim] paired

    def _rope_apply(self, t: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        cos = np.cos(freqs)[None, None].astype(np.float32)
        sin = np.sin(freqs)[None, None].astype(np.float32)
        t1, t2 = t[..., 0::2], t[..., 1::2]
        rot = np.stack([-t2, t1], axis=-1).reshape(t.shape)
        return t * cos + rot * sin

    @staticmethod
    def _rms(v: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
        return v / np.sqrt(np.mean(np.square(v), axis=-1, keepdims=True) + eps) * w

    @staticmethod
    def _silu(g: np.ndarray) -> np.ndarray:
        return g / (1.0 + np.exp(-g))

    # --------------------------------------------------- forward (numpy)
    def forward_numpy(self, tokens: np.ndarray) -> np.ndarray:
        cfg = self.config
        p = self.params
        B, T = tokens.shape
        hd = cfg.d_model // cfg.n_heads
        x = p["tok_emb"][tokens]
        freqs = self._rope_table(T)
        mask = np.tril(np.ones((T, T), dtype=bool))[None, None]
        for blk in p["blocks"]:
            h = self._rms(x, blk["ln1"], cfg.rms_norm_eps)
            q = (h @ blk["attn"]["q"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            k = (h @ blk["attn"]["k"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            v = (h @ blk["attn"]["v"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            q, k = self._rope_apply(q, freqs), self._rope_apply(k, freqs)
            scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
            scores = np.where(mask, scores, np.finfo(np.float32).min)
            a = np.exp(scores - scores.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
            x = x + out @ blk["attn"]["o"]
            h = self._rms(x, blk["ln2"], cfg.rms_norm_eps)
            x = x + (self._silu(h @ blk["mlp"]["gate"]) * (h @ blk["mlp"]["up"])) @ blk["mlp"]["down"]
        x = self._rms(x, p["ln_f"], cfg.rms_norm_eps)
        return x @ p["tok_emb"].T

    def forward_jax(self, tokens: np.ndarray) -> np.ndarray:
        from ..model import forward

        logits = forward(self._jparams, self._jnp.asarray(tokens), self.config)
        return np.asarray(logits)

    def forward(self, tokens: np.ndarray) -> np.ndarray:
        return self.forward_jax(tokens) if self.backend == "jax" else self.forward_numpy(tokens)

    # ------------------------------------------- cached forward (numpy)
    def prefill_numpy(self, tokens: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        """Prefill a numpy KV cache; returns logits for all positions."""
        cfg = self.config
        p = self.params
        B, T = tokens.shape
        hd = cfg.d_model // cfg.n_heads
        cache: dict[str, Any] = {"k": [], "v": [], "pos": 0}
        x = p["tok_emb"][tokens]
        freqs = self._rope_table(T)
        mask = np.tril(np.ones((T, T), dtype=bool))[None, None]
        for blk in p["blocks"]:
            h = self._rms(x, blk["ln1"], cfg.rms_norm_eps)
            q = (h @ blk["attn"]["q"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            k = (h @ blk["attn"]["k"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            v = (h @ blk["attn"]["v"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            q, k = self._rope_apply(q, freqs), self._rope_apply(k, freqs)
            cache["k"].append(k)
            cache["v"].append(v)
            scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
            scores = np.where(mask, scores, np.finfo(np.float32).min)
            a = np.exp(scores - scores.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
            x = x + out @ blk["attn"]["o"]
            h = self._rms(x, blk["ln2"], cfg.rms_norm_eps)
            x = x + (self._silu(h @ blk["mlp"]["gate"]) * (h @ blk["mlp"]["up"])) @ blk["mlp"]["down"]
        x = self._rms(x, p["ln_f"], cfg.rms_norm_eps)
        cache["pos"] = T
        return x @ p["tok_emb"].T, cache

    def decode_step_numpy(self, token: np.ndarray, cache: dict[str, Any]) -> np.ndarray:
        """Single-token cached decode step (numpy). Mutates ``cache``."""
        cfg = self.config
        p = self.params
        B, T = token.shape
        hd = cfg.d_model // cfg.n_heads
        pos = int(cache["pos"])
        freqs = self._rope_table(pos + T)[pos:pos + T]
        x = p["tok_emb"][token]
        for li, blk in enumerate(p["blocks"]):
            h = self._rms(x, blk["ln1"], cfg.rms_norm_eps)
            q = (h @ blk["attn"]["q"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            k_new = (h @ blk["attn"]["k"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            v_new = (h @ blk["attn"]["v"]).reshape(B, T, cfg.n_heads, hd).transpose(0, 2, 1, 3)
            q, k_new = self._rope_apply(q, freqs), self._rope_apply(k_new, freqs)
            cache["k"][li] = np.concatenate([cache["k"][li][:, :, :pos], k_new], axis=2)
            cache["v"][li] = np.concatenate([cache["v"][li][:, :, :pos], v_new], axis=2)
            k, v = cache["k"][li], cache["v"][li]
            scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
            a = np.exp(scores - scores.max(-1, keepdims=True))
            a = a / a.sum(-1, keepdims=True)
            out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg.d_model)
            x = x + out @ blk["attn"]["o"]
            h = self._rms(x, blk["ln2"], cfg.rms_norm_eps)
            x = x + (self._silu(h @ blk["mlp"]["gate"]) * (h @ blk["mlp"]["up"])) @ blk["mlp"]["down"]
        cache["pos"] = pos + T
        x = self._rms(x, p["ln_f"], cfg.rms_norm_eps)
        return x @ p["tok_emb"].T

    # ------------------------------------------------- metrics / generation
    def sequence_loss(self, ids: list[int]) -> float:
        # Documents longer than the context window are truncated to their head:
        # positions beyond max_seq_len were never trained and would dominate the
        # mean with out-of-distribution loss.
        max_ctx = int(self.config.max_seq_len)
        if len(ids) > max_ctx:
            ids = ids[:max_ctx]
        arr = np.asarray([ids], dtype=np.int32)
        logits = self.forward(arr)[0].astype(np.float64)
        logits = logits - logits.max(-1, keepdims=True)
        logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
        targets = arr[0, 1:]
        n = min(len(targets), len(logp) - 1)
        if n <= 0:
            return float("nan")
        return float(-logp[np.arange(n), targets[:n]].mean())

    def perplexity(self, texts: list[str]) -> float:
        vals = [self.sequence_loss(self.encode(t)) for t in texts]
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.exp(np.mean(vals))) if vals else float("nan")

    @staticmethod
    def _apply_repetition_penalty(logits: np.ndarray, tokens: list[int], penalty: float) -> np.ndarray:
        if penalty == 1.0 or not tokens:
            return logits
        for tok in set(tokens):
            if logits[tok] > 0:
                logits[tok] /= penalty
            else:
                logits[tok] *= penalty
        return logits

    @staticmethod
    def _nucleus(logits: np.ndarray, top_p: float) -> np.ndarray:
        """Return a probability vector for the smallest prefix with cum >= top_p."""
        order = np.argsort(logits)[::-1]
        sorted_logits = logits[order]
        probs = np.exp(sorted_logits - sorted_logits.max())
        probs = probs / probs.sum()
        cum = np.cumsum(probs)
        k = int(np.searchsorted(cum, top_p, side="left")) + 1
        k = max(1, min(k, probs.size))
        keep = order[:k]
        masked = np.full_like(probs, -np.inf)
        masked[:k] = sorted_logits[:k]
        out = np.exp(masked - masked.max())
        out = out / out.sum()
        full = np.zeros_like(logits)
        full[keep] = out[:k]
        return full

    def generate(self, prompt: str, max_new_tokens: int = 96, temperature: float = 0.0,
                 top_p: float = 1.0, repetition_penalty: float = 1.1,
                 stop: list[str] | None = None, seed: int | None = None,
                 use_cache: bool = True) -> str:
        """Generate a continuation.

        Deterministic when ``temperature <= 0``; when sampling, ``seed=None``
        uses seed 0 so behaviour is still reproducible (P0-13).
        """
        ids = self.encode(prompt)
        if not ids:
            ids = [self.bos_id]
        max_ctx = self.config.max_seq_len
        if len(ids) > max_ctx - 1:
            ids = ids[-(max_ctx - 1):]                       # keep the prompt tail
        rng = np.random.default_rng(0 if seed is None else seed)
        generated: list[int] = []
        context_len = len(ids)

        def sample(logits: np.ndarray) -> int:
            logits = np.asarray(logits, dtype=np.float64).copy()
            logits = self._apply_repetition_penalty(logits, generated, repetition_penalty)
            if temperature <= 0.0:
                return int(np.argmax(logits))
            scaled = logits / max(temperature, 1e-6)
            if top_p < 1.0:
                probs = self._nucleus(scaled, top_p)
            else:
                scaled = scaled - scaled.max()
                probs = np.exp(scaled)
                probs = probs / probs.sum()
            return int(rng.choice(len(probs), p=probs))

        if use_cache:
            arr = np.asarray([ids], dtype=np.int32)
            if self.backend == "jax":
                from ..model import init_cache, prefill_cache, forward_with_cache

                cache = init_cache(self.config, 1)
                logits = np.asarray(prefill_cache(self._jparams, self._jnp.asarray(arr),
                                                  cache, self.config))[0, -1]
                for _ in range(max_new_tokens):
                    nxt = sample(logits)
                    if nxt == self.eos_id:
                        break
                    generated.append(nxt)
                    context_len += 1
                    text_so_far = self.decode(generated)
                    if stop and any(s in text_so_far for s in stop):
                        break
                    if context_len >= max_ctx:
                        break
                    logits = np.asarray(forward_with_cache(
                        self._jparams, self._jnp.asarray([[nxt]], dtype=self._jnp.int32),
                        cache, self.config))[0, -1]
            else:
                logits, cache = self.prefill_numpy(arr)
                logits = logits[0, -1]
                for _ in range(max_new_tokens):
                    nxt = sample(logits)
                    if nxt == self.eos_id:
                        break
                    generated.append(nxt)
                    context_len += 1
                    text_so_far = self.decode(generated)
                    if stop and any(s in text_so_far for s in stop):
                        break
                    if context_len >= max_ctx:
                        break
                    logits = self.decode_step_numpy(np.asarray([[nxt]], dtype=np.int32), cache)[0, -1]
        else:  # pragma: no cover - kept for parity tests
            for _ in range(max_new_tokens):
                window = ids[len(ids) - max_ctx:] if len(ids) > max_ctx else ids
                arr = np.asarray([window], dtype=np.int32)
                logits = self.forward(arr)[0, -1]
                nxt = sample(logits)
                if nxt == self.eos_id:
                    break
                generated.append(nxt)
                ids.append(nxt)
                context_len += 1
                text_so_far = self.decode(generated)
                if stop and any(s in text_so_far for s in stop):
                    break
                if context_len >= max_ctx:
                    break
        return self.decode(generated)

    # ------------------------------------------------------------ metadata
    def verify_parity(self, tokens: np.ndarray | None = None, tol: float = 2e-4) -> dict[str, Any]:
        """Compare full forward vs cached decode and JAX vs NumPy backends."""
        rng = np.random.default_rng(0)
        if tokens is None:
            tokens = rng.integers(0, self.config.vocab_size, size=(1, 12)).astype(np.int32)
        out: dict[str, Any] = {"tokens": tokens.shape[1]}
        full_np = self.forward_numpy(tokens)
        logits, cache = self.prefill_numpy(tokens[:, :-1])
        step = self.decode_step_numpy(tokens[:, -1:], cache)
        out["numpy_full_vs_cached_max_abs_diff"] = float(np.abs(full_np[0, -1] - step[0, -1]).max())
        out["numpy_cache_parity_pass"] = out["numpy_full_vs_cached_max_abs_diff"] <= tol
        # The parity check is a first-class artefact: build the JAX view lazily so
        # it works for a NumPy-backend engine too.
        if self._jparams is None:
            import jax.numpy as jnp

            self._jnp = jnp
            self._jparams = self._to_jax(self.params)
        if self._jparams is not None:
            from ..model import forward_with_cache, init_cache, prefill_cache

            full_jax = self.forward_jax(tokens)
            cache = init_cache(self.config, 1)
            prefill_cache(self._jparams, self._jnp.asarray(tokens[:, :-1]), cache, self.config)
            cached_jax = np.asarray(forward_with_cache(
                self._jparams, self._jnp.asarray(tokens[:, -1:]), cache, self.config))[0, -1]
            out["jax_full_vs_cached_max_abs_diff"] = float(
                np.abs(full_jax[0, -1] - cached_jax).max())
            out["jax_cache_parity_pass"] = out["jax_full_vs_cached_max_abs_diff"] <= tol
            out["jax_vs_numpy_max_abs_diff"] = float(np.abs(full_jax - full_np).max())
            out["jax_numpy_parity_pass"] = out["jax_vs_numpy_max_abs_diff"] <= tol
        out["ok"] = all(v for k, v in out.items() if k.endswith("_pass"))
        out["tolerance"] = tol
        return out
