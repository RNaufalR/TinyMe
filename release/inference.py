#!/usr/bin/env python3
"""Standalone TinyMe inference entry point.

Requires only: numpy, safetensors, tokenizers.  No JAX, no repository imports.
Reads ``config.json`` (model config), ``tokenizer.json`` and one weight file
(``model_fp32.safetensors`` by default; fp16/int8/int4 are dequantized on load).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_flat(path):
    from safetensors import safe_open

    with safe_open(str(path), framework="numpy") as f:
        meta = f.metadata() or {}
        dtype = meta.get("dtype", "float32")
        flat = {k: f.get_tensor(k) for k in f.keys()}
    if dtype == "int8":
        flat = _dequantize_int8(flat)
    elif dtype == "int4":
        shapes = {k: tuple(v) for k, v in json.loads(meta.get("shapes", "{}")).items()}
        flat = _dequantize_int4(flat, shapes)
    return {k: np.asarray(v, dtype=np.float32) for k, v in flat.items()}, dtype


def _dequantize_int8(flat):
    out = {}
    for k, v in flat.items():
        if k.endswith(".__scale__"):
            continue
        scale = flat.get(k + ".__scale__")
        if v.dtype == np.int8 and scale is not None:
            out[k] = v.astype(np.float32) * scale.astype(np.float32).reshape(1, -1)
        else:
            out[k] = v.astype(np.float32)
    return out


def _dequantize_int4(flat, shape_hint, group=64):
    out = {}
    for k, v in flat.items():
        if k.endswith(".__scale__"):
            continue
        scale = flat.get(k + ".__scale__")
        if v.dtype == np.uint8 and scale is not None:
            rows, ngroups = scale.shape
            v = v.reshape(rows, ngroups, -1)
            low = (v & 0x0F).astype(np.int8)
            high = ((v >> 4) & 0x0F).astype(np.int8)
            low = np.where(low >= 8, low - 16, low)
            high = np.where(high >= 8, high - 16, high)
            vals = np.stack([low, high], axis=-1).reshape(rows, ngroups, group)
            deq = (vals * scale[:, :, None]).reshape(rows, -1)
            if k in shape_hint:
                deq = deq[:, : shape_hint[k][1]]
            out[k] = deq.astype(np.float32)
        else:
            out[k] = v.astype(np.float32)
    return out


def build_params(flat):
    params, blocks = {}, {}
    for key, arr in flat.items():
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


def rms(x, w, eps):
    return x / np.sqrt(np.mean(np.square(x), axis=-1, keepdims=True) + eps) * w


def silu(x):
    return x / (1.0 + np.exp(-x))


def rope_table(T, head_dim, theta):
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) / half))
    freqs = np.outer(np.arange(T, dtype=np.float32), inv)
    return np.repeat(freqs, 2, axis=-1)


def rope_apply(t, freqs):
    cos, sin = np.cos(freqs)[None, None].astype(np.float32), np.sin(freqs)[None, None].astype(np.float32)
    t1, t2 = t[..., 0::2], t[..., 1::2]
    rot = np.stack([-t2, t1], axis=-1).reshape(t.shape)
    return t * cos + rot * sin


def forward(cfg, params, tokens):
    B, T = tokens.shape
    hd = cfg["d_model"] // cfg["n_heads"]
    x = params["tok_emb"][tokens]
    freqs = rope_table(T, hd, cfg["rope_theta"])
    mask = np.tril(np.ones((T, T), dtype=bool))[None, None]
    for blk in params["blocks"]:
        h = rms(x, blk["ln1"], cfg["rms_norm_eps"])
        q = (h @ blk["attn"]["q"]).reshape(B, T, cfg["n_heads"], hd).transpose(0, 2, 1, 3)
        k = (h @ blk["attn"]["k"]).reshape(B, T, cfg["n_heads"], hd).transpose(0, 2, 1, 3)
        v = (h @ blk["attn"]["v"]).reshape(B, T, cfg["n_heads"], hd).transpose(0, 2, 1, 3)
        q, k = rope_apply(q, freqs), rope_apply(k, freqs)
        scores = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
        scores = np.where(mask, scores, np.finfo(np.float32).min)
        a = np.exp(scores - scores.max(-1, keepdims=True))
        a = a / a.sum(-1, keepdims=True)
        out = (a @ v).transpose(0, 2, 1, 3).reshape(B, T, cfg["d_model"])
        x = x + out @ blk["attn"]["o"]
        h = rms(x, blk["ln2"], cfg["rms_norm_eps"])
        x = x + (silu(h @ blk["mlp"]["gate"]) * (h @ blk["mlp"]["up"])) @ blk["mlp"]["down"]
    x = rms(x, params["ln_f"], cfg["rms_norm_eps"])
    return x @ params["tok_emb"].T


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="model_fp32.safetensors")
    ap.add_argument("--config", default="config.json")
    ap.add_argument("--tokenizer", default="tokenizer.json")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--system", default="You are TinyMe, a small model that answers with verified steps.")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    here = Path(__file__).resolve().parent
    from tokenizers import Tokenizer

    cfg = json.loads((here / args.config).read_text())
    tok = Tokenizer.from_file(str(here / args.tokenizer))
    params, dtype = load_flat(here / args.model)
    params = build_params(params)

    text = f"<|system|>\n{args.system}\n<|user|>\n{args.prompt}\n<|assistant|>\n"
    ids = tok.encode(text, add_special_tokens=False).ids
    rng = np.random.default_rng(0)
    out_ids = []
    for _ in range(args.max_new_tokens):
        logits = forward(cfg, params, np.asarray([ids], dtype=np.int64))[0, -1]
        if args.temperature and args.temperature > 0:
            probs = np.exp((logits - logits.max()) / args.temperature)
            probs = probs / probs.sum()
            nxt = int(rng.choice(len(probs), p=probs))
        else:
            nxt = int(np.argmax(logits))
        if nxt == tok.token_to_id("<|eos|>"):
            break
        ids.append(nxt)
        out_ids.append(nxt)
    def _count(tree):
        if isinstance(tree, dict):
            return sum(_count(v) for v in tree.values())
        if isinstance(tree, (list, tuple)):
            return sum(_count(v) for v in tree)
        return int(np.prod(tree.shape)) if hasattr(tree, "shape") else 0

    print(f"[weights={dtype} params={_count(params):,}]")
    print(tok.decode(out_ids, skip_special_tokens=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
