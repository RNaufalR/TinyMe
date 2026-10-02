"""Weight-only post-training quantization (spec §23).

Supported precisions:
  * float32 (baseline)
  * float16 (2x smaller, ~lossless)
  * int8    (per-channel symmetric, 4x smaller)
  * int4    (per-group symmetric, group=64, 8x smaller)

Scales/zero-points are stored alongside the packed integer weights, so the
actual serialized artifact size is measured, never estimated.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

INT4_GROUP = 64


def quantize_to_fp16(params: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: v.astype(np.float16) for k, v in params.items()}


def quantize_int8(params: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-output-channel symmetric INT8 quantization of 2-D weights."""
    q: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for k, v in params.items():
        if v.ndim == 2 and v.size >= 256:
            absmax = np.max(np.abs(v), axis=0, keepdims=True)
            scale = np.maximum(absmax / 127.0, 1e-12).astype(np.float32)
            q[k] = np.clip(np.round(v / scale), -127, 127).astype(np.int8)
            scales[k + ".__scale__"] = scale.astype(np.float32).reshape(-1)
        else:
            q[k] = v.astype(np.float32)
    return q, scales


def dequantize_int8(q: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in q.items():
        s = scales.get(k + ".__scale__")
        if v.dtype == np.int8 and s is not None:
            out[k] = v.astype(np.float32) * s.reshape(1, -1)
        else:
            out[k] = v.astype(np.float32)
    return out


def quantize_int4(params: dict[str, Any], group: int = INT4_GROUP) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Per-group symmetric INT4 quantization; two values packed per byte."""
    q: dict[str, np.ndarray] = {}
    scales: dict[str, np.ndarray] = {}
    for k, v in params.items():
        if v.ndim == 2 and v.size >= 512:
            rows, cols = v.shape
            pad = (-cols) % group
            if pad:
                v = np.concatenate([v, np.zeros((rows, pad), dtype=v.dtype)], axis=1)
            vg = v.reshape(rows, -1, group)
            absmax = np.max(np.abs(vg), axis=-1, keepdims=True)
            scale = np.maximum(absmax / 7.0, 1e-12).astype(np.float32)
            qi = np.clip(np.round(vg / scale), -7, 7).astype(np.int8)
            packed = (qi[:, :, 0::2] & 0x0F) | ((qi[:, :, 1::2] & 0x0F) << 4)
            q[k] = packed.reshape(rows, -1).astype(np.uint8)
            scales[k + ".__scale__"] = scale.astype(np.float32).reshape(rows, -1)
        else:
            q[k] = v.astype(np.float32)
    return q, scales


def dequantize_int4(q: dict[str, np.ndarray], scales: dict[str, np.ndarray],
                    group: int = INT4_GROUP, shape_hint: dict[str, tuple[int, int]] | None = None) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    shape_hint = shape_hint or {}
    for k, v in q.items():
        s = scales.get(k + ".__scale__")
        if v.dtype == np.uint8 and s is not None:
            rows, ngroups = s.shape
            v = v.reshape(rows, ngroups, -1)
            low = (v & 0x0F).astype(np.int8)
            high = ((v >> 4) & 0x0F).astype(np.int8)
            low = np.where(low >= 8, low - 16, low)
            high = np.where(high >= 8, high - 16, high)
            vals = np.stack([low, high], axis=-1).reshape(rows, ngroups, group).reshape(rows, -1)
            # Broadcast each group's scale across its `group` values.
            deq = (vals.reshape(rows, ngroups, group) * s[:, :, None]).reshape(rows, -1)
            if k in shape_hint:
                deq = deq[:, : shape_hint[k][1]]
            out[k] = deq.astype(np.float32)
        else:
            out[k] = v.astype(np.float32)
    return out


def quantized_size_bytes(q: dict[str, np.ndarray], scales: dict[str, np.ndarray]) -> int:
    total = 0
    for v in q.values():
        total += v.nbytes
    for v in scales.values():
        total += v.nbytes
    return total


# --------------------------------------------------------------- tree helpers
def flatten_params(params: dict[str, Any], prefix: str = "") -> dict[str, np.ndarray]:
    """Flatten the nested parameter tree into ``blocks/0/attn/q`` style keys."""
    flat: dict[str, np.ndarray] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten_params(value, prefix=f"{name}/"))
        elif isinstance(value, (list, tuple)):
            for idx, item in enumerate(value):
                if isinstance(item, dict):
                    flat.update(flatten_params(item, prefix=f"{name}/{idx}/"))
                else:
                    flat[f"{name}/{idx}"] = np.asarray(item)
        else:
            flat[name] = np.asarray(value)
    return flat


def unflatten_params(flat: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rebuild the nested tree from flat keys (inverse of :func:`flatten_params`)."""
    params: dict[str, Any] = {}
    blocks: dict[int, dict[str, Any]] = {}
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


def quantize_tree_to_fp16(params: dict[str, Any]) -> dict[str, Any]:
    return unflatten_params(quantize_to_fp16(flatten_params(params)))


def quantize_tree_int8(params: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    q, scales = quantize_int8(flatten_params(params))
    return unflatten_params(q), scales


def dequantize_tree_int8(q: dict[str, Any], scales: dict[str, Any]) -> dict[str, Any]:
    return unflatten_params(dequantize_int8(flatten_params(q), scales))


def quantize_tree_int4(params: dict[str, Any], group: int = INT4_GROUP):
    flat = flatten_params(params)
    q, scales = quantize_int4(flat, group=group)
    return q, scales, {k: v.shape for k, v in flat.items()}


def dequantize_tree_int4(q: dict[str, Any], scales: dict[str, Any],
                         shape_hint: dict[str, tuple[int, ...]] | None = None) -> dict[str, Any]:
    return unflatten_params(dequantize_int4(q, scales, shape_hint=shape_hint))


def tree_size_bytes(tree: dict[str, Any]) -> int:
    """Serialized byte size of a (possibly nested) parameter tree."""
    return int(sum(v.nbytes for v in flatten_params(tree).values()))


def quantized_tree_size_bytes(q: dict[str, Any], scales: dict[str, Any]) -> int:
    return tree_size_bytes(q) + tree_size_bytes(scales)
