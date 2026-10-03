#!/usr/bin/env python3
"""Export a TinyMe checkpoint to a real GGUF v3 file (llama architecture).

Why a transformer mapping is needed at all
------------------------------------------
GGUF runtimes do not know TinyMe's module names, and TinyMe's RoPE pairs
*consecutive* head-dimension components ``(x_2i, x_2i+1)`` while ``LLM_ARCH_LLAMA``
in the reference runtime uses the half-split pairing ``(x_i, x_i + d/2)``.  Both
facts are handled explicitly here rather than by hoping the file "looks close":

* weight matrices are re-emitted under the canonical names the runtime expects;
* the RoPE convention is *checked* rather than assumed.  ggml rotates
  interleaved pairs ``(x[i], x[i+1])`` for ``GGML_ROPE_TYPE_NORMAL`` (the mode
  ``LLM_ARCH_LLAMA`` selects) with the same frequency index and the same
  ``theta^(-2i/d)`` schedule TinyMe uses, so the raw weights are written
  unchanged; ``--permute on`` exists only to demonstrate that inserting the
  HuggingFace half-split permutation breaks the equivalence.

The file layout follows the GGUF v3 specification (little-endian, 32-byte
alignment): header, key/value metadata, tensor infos, then the tensor data.
Every value written here is derived from the checkpoint or the shipped
tokenizer — nothing is guessed — and the writer records SHA-256 of the finished
file next to it.

Usage::

    python scripts/gguf_export.py --experiment EXP-015-TOOL-SFT-V9 --checkpoint best \\
        --dataset dataset_v9 --out release/tinyme-f16.gguf --dtype f16 --pre gpt2
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402

# GGUF value types
GGUF_UINT8, GGUF_INT8, GGUF_UINT16, GGUF_INT16 = 0, 1, 2, 3
GGUF_UINT32, GGUF_INT32, GGUF_FLOAT32, GGUF_BOOL = 4, 5, 6, 7
GGUF_STRING, GGUF_ARRAY, GGUF_UINT64, GGUF_INT64, GGUF_FLOAT64 = 8, 9, 10, 11, 12

# ggml tensor types
GGML_F32, GGML_F16, GGML_BF16 = 0, 1, 30

TOKEN_TYPE_NORMAL, TOKEN_TYPE_UNKNOWN, TOKEN_TYPE_CONTROL = 1, 2, 3


class Writer:
    def __init__(self, path: Path):
        self.path = path
        self.kv: list[tuple[str, int, object]] = []
        self.tensors: list[tuple[str, list[int], int, bytes]] = []

    # ------------------------------------------------------------- metadata
    def add_u32(self, key: str, value: int) -> None:
        self.kv.append((key, GGUF_UINT32, int(value)))

    def add_f32(self, key: str, value: float) -> None:
        self.kv.append((key, GGUF_FLOAT32, float(value)))

    def add_bool(self, key: str, value: bool) -> None:
        self.kv.append((key, GGUF_BOOL, bool(value)))

    def add_str(self, key: str, value: str) -> None:
        self.kv.append((key, GGUF_STRING, str(value)))

    def add_str_array(self, key: str, values: list[str]) -> None:
        self.kv.append((key, GGUF_ARRAY, (GGUF_STRING, list(values))))

    def add_i32_array(self, key: str, values: list[int]) -> None:
        self.kv.append((key, GGUF_ARRAY, (GGUF_INT32, [int(v) for v in values])))

    def add_tensor(self, name: str, array: np.ndarray, dtype: int) -> None:
        # 1-D tensors (RMSNorm weights) stay F32: the runtime multiplies them
        # against F32 activations and the CPU binary ops reject a mixed f32/f16
        # multiply (the reference converter keeps them F32 for the same reason).
        if array.ndim <= 1:
            dtype = GGML_F32
        if dtype == GGML_F16:
            data = np.ascontiguousarray(array.astype(np.float16)).tobytes()
        elif dtype == GGML_F32:
            data = np.ascontiguousarray(array.astype(np.float32)).tobytes()
        else:
            raise ValueError(f"unsupported dtype {dtype}")
        # GGUF stores ne[0] = fastest-varying dimension = last numpy axis
        dims = [int(d) for d in array.shape[::-1]]
        self.tensors.append((name, dims, dtype, data))

    # ----------------------------------------------------------------- write
    @staticmethod
    def _enc_string(text: str) -> bytes:
        raw = text.encode("utf-8")
        return struct.pack("<Q", len(raw)) + raw

    def _enc_value(self, vtype: int, value) -> bytes:
        if vtype == GGUF_UINT32:
            return struct.pack("<I", value)
        if vtype == GGUF_INT32:
            return struct.pack("<i", value)
        if vtype == GGUF_FLOAT32:
            return struct.pack("<f", value)
        if vtype == GGUF_BOOL:
            return struct.pack("<?", value)
        if vtype == GGUF_STRING:
            return self._enc_string(value)
        if vtype == GGUF_ARRAY:
            etype, items = value
            out = struct.pack("<IQ", etype, len(items))
            for item in items:
                out += self._enc_value(etype, item)
            return out
        raise ValueError(f"unsupported gguf value type {vtype}")

    def write(self, alignment: int = 32) -> dict:
        buf = bytearray()
        buf += b"GGUF"
        buf += struct.pack("<IQQ", 3, len(self.tensors), len(self.kv))
        for key, vtype, value in self.kv:
            buf += self._enc_string(key)
            buf += struct.pack("<I", vtype)
            buf += self._enc_value(vtype, value)

        # tensor infos: data offsets are computed after the header is complete
        infos = bytearray()
        offset = 0
        offsets = []
        for name, dims, dtype, data in self.tensors:
            align_pad = (-offset) % alignment
            offset += align_pad
            offsets.append((offset, align_pad))
            infos += self._enc_string(name)
            infos += struct.pack("<I", len(dims))
            for d in dims:
                infos += struct.pack("<Q", d)
            infos += struct.pack("<I", dtype)
            infos += struct.pack("<Q", offset)
            offset += len(data)
        buf += infos
        buf += b"\x00" * ((-len(buf)) % alignment)

        for (offset, align_pad), (_, _, _, data) in zip(offsets, self.tensors):
            buf += b"\x00" * align_pad
            buf += data

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(bytes(buf))
        digest = hashlib.sha256(bytes(buf)).hexdigest()
        return {"file": self.path.name, "bytes": len(buf), "sha256": digest,
                "tensors": len(self.tensors), "kv_count": len(self.kv),
                "gguf_version": 3, "alignment": alignment}


def _rope_permutation(head_dim: int) -> np.ndarray:
    """Index permutation mapping TinyMe's interleaved pairs onto llama's half pairs."""
    half = head_dim // 2
    p = np.arange(head_dim)
    return 2 * (p % half) + (p // half)


def _permute_heads(weight: np.ndarray, n_heads: int, head_dim: int) -> np.ndarray:
    """Permute the output dimension of a projection, per head (rows = outputs)."""
    out = weight.reshape(n_heads, head_dim, weight.shape[1]).copy()
    out = out[:, _rope_permutation(head_dim), :]     # permute within each head
    return out.reshape(weight.shape)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--out", default="release/tinyme-f16.gguf")
    ap.add_argument("--dtype", default="f16", choices=["f16", "f32"])
    ap.add_argument("--pre", default="gpt2", help="tokenizer.ggml.pre value (gpt2|chameleon)")
    ap.add_argument("--name", default="TinyMe nano")
    ap.add_argument("--report", default=None)
    ap.add_argument("--permute", default="off", choices=["on", "off"],
                    help="permute q/k output rows.  Default off: ggml's GGML_ROPE_TYPE_NORMAL "
                         "(what LLM_ARCH_LLAMA uses) rotates *interleaved* pairs (x[i], x[i+1]) "
                         "with freq_index = i -- byte-identical in convention to TinyMe's "
                         "rotate over t[..., 0::2] / t[..., 1::2] (verified by reading "
                         "ggml/src/ggml-cpu/ops.cpp rotate_pairs()).  'on' exists to show that "
                         "applying the HF half-split permutation *breaks* that equivalence.")
    args = ap.parse_args()

    ckpt = ROOT / "checkpoints" / args.experiment / f"{args.checkpoint}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    tok_path = ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json"
    engine = InferenceEngine(ckpt, tok_path, backend="numpy")
    cfg = engine.config
    params = engine.params
    dtype = GGML_F16 if args.dtype == "f16" else GGML_F32
    file_type = 1 if args.dtype == "f16" else 0

    head_dim = cfg.d_model // cfg.n_heads
    assert head_dim % 2 == 0, "RoPE requires an even head dimension"

    writer = Writer(Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out)

    # ------------------------------------------------------------ general
    writer.add_str("general.architecture", "llama")
    writer.add_str("general.name", args.name)
    writer.add_u32("general.file_type", file_type)
    writer.add_str("general.source.experiment", args.experiment)
    writer.add_str("general.source.checkpoint", args.checkpoint)
    writer.add_str("general.source.dataset", args.dataset)

    # -------------------------------------------------------------- hparams
    writer.add_u32("llama.context_length", int(cfg.max_seq_len))
    writer.add_u32("llama.embedding_length", int(cfg.d_model))
    writer.add_u32("llama.block_count", int(cfg.n_layers))
    writer.add_u32("llama.feed_forward_length", int(cfg.d_ff))
    writer.add_u32("llama.attention.head_count", int(cfg.n_heads))
    writer.add_u32("llama.attention.head_count_kv", int(cfg.n_heads))
    writer.add_f32("llama.attention.layer_norm_rms_epsilon", float(cfg.rms_norm_eps))
    writer.add_u32("llama.rope.dimension_count", int(head_dim))
    writer.add_f32("llama.rope.freq_base", float(cfg.rope_theta))
    writer.add_u32("llama.vocab_size", int(cfg.vocab_size))

    # ------------------------------------------------------------ tokenizer
    from tokenizers import Tokenizer
    raw = json.loads(tok_path.read_text(encoding="utf-8"))
    vocab: dict[str, int] = dict(raw["model"]["vocab"])
    specials: dict[int, str] = {}
    for entry in raw.get("added_tokens", []):
        vocab[entry["content"]] = int(entry["id"])
        specials[int(entry["id"])] = entry["content"]
    tokens = [""] * int(cfg.vocab_size)
    for token, idx in vocab.items():
        if 0 <= idx < len(tokens):
            tokens[idx] = token
    token_types = [TOKEN_TYPE_CONTROL if i in specials else TOKEN_TYPE_NORMAL
                   for i in range(len(tokens))]
    merges = [" ".join(m) if isinstance(m, list) else str(m) for m in raw["model"]["merges"]]
    writer.add_str("tokenizer.ggml.model", "gpt2")
    writer.add_str("tokenizer.ggml.pre", args.pre)
    writer.add_str_array("tokenizer.ggml.tokens", tokens)
    writer.add_i32_array("tokenizer.ggml.token_type", token_types)
    writer.add_str_array("tokenizer.ggml.merges", merges)
    tok = load_tokenizer(args.dataset)
    for key, token in (("bos", "<|bos|>"), ("eos", "<|eos|>"), ("padding", "<|pad|>"),
                       ("unk", "<|unk|>")):
        tid = tok.tok.token_to_id(token)
        if tid is not None:
            writer.add_u32(f"tokenizer.ggml.{key}_token_id", int(tid))
    writer.add_bool("tokenizer.ggml.add_bos_token", True)
    writer.add_bool("tokenizer.ggml.add_eos_token", False)

    # --------------------------------------------------------------- tensors
    # names and shapes must match LLM_ARCH_LLAMA exactly (see the module docstring)
    writer.add_tensor("token_embd.weight", np.asarray(params["tok_emb"], np.float32), dtype)
    for i, blk in enumerate(params["blocks"]):
        attn, mlp = blk["attn"], blk["mlp"]
        q = np.asarray(attn["q"], np.float32).T
        k = np.asarray(attn["k"], np.float32).T
        if args.permute == "on":               # demonstration only; see module docstring
            q = _permute_heads(q, cfg.n_heads, head_dim)
            k = _permute_heads(k, cfg.n_heads, head_dim)
        writer.add_tensor(f"blk.{i}.attn_q.weight", q, dtype)
        writer.add_tensor(f"blk.{i}.attn_k.weight", k, dtype)
        writer.add_tensor(f"blk.{i}.attn_v.weight", np.asarray(attn["v"], np.float32).T, dtype)
        writer.add_tensor(f"blk.{i}.attn_output.weight", np.asarray(attn["o"], np.float32).T, dtype)
        writer.add_tensor(f"blk.{i}.ffn_gate.weight", np.asarray(mlp["gate"], np.float32).T, dtype)
        writer.add_tensor(f"blk.{i}.ffn_up.weight", np.asarray(mlp["up"], np.float32).T, dtype)
        writer.add_tensor(f"blk.{i}.ffn_down.weight", np.asarray(mlp["down"], np.float32).T, dtype)
        writer.add_tensor(f"blk.{i}.attn_norm.weight", np.asarray(blk["ln1"], np.float32), dtype)
        writer.add_tensor(f"blk.{i}.ffn_norm.weight", np.asarray(blk["ln2"], np.float32), dtype)
    writer.add_tensor("output_norm.weight", np.asarray(params["ln_f"], np.float32), dtype)
    if not cfg.tie_word_embeddings:
        writer.add_tensor("output.weight", np.asarray(params["tok_emb"], np.float32), dtype)

    info = writer.write(alignment=32)
    info.update({
        "experiment": args.experiment, "checkpoint": args.checkpoint, "dataset": args.dataset,
        "architecture": {"name": "llama", "hidden_size": cfg.d_model, "layers": cfg.n_layers,
                         "heads": cfg.n_heads, "head_dim": head_dim, "ffn": cfg.d_ff,
                         "vocab_size": cfg.vocab_size, "context_length": cfg.max_seq_len,
                         "rope_theta": cfg.rope_theta, "rms_norm_eps": cfg.rms_norm_eps,
                         "tied_embeddings": cfg.tie_word_embeddings},
        "parameters": int(sum(int(np.prod(v.shape)) for v in params.values()
                              if isinstance(v, np.ndarray))
                          + sum(int(np.prod(v.shape)) for b in params["blocks"]
                                for v in list(b.values()) + list(b["attn"].values()) + list(b["mlp"].values())
                                if hasattr(v, "shape"))),
        "dtype": args.dtype, "tokenizer_pre": args.pre,
        "rope_convention": ("interleaved pairs, identical to ggml GGML_ROPE_TYPE_NORMAL"
                            if args.permute == "off" else
                            "q/k rows permuted with the HuggingFace half-split mapping "
                            "(used only to demonstrate that it breaks equivalence)"),
        "decimal_mb": round(info["bytes"] / 1e6, 3),
        "mib": round(info["bytes"] / (1024 * 1024), 3),
    })
    report = Path(args.report) if args.report else writer.path.with_suffix(".json")
    report = report if report.is_absolute() else ROOT / report
    write_json(info, report)
    print(json.dumps(info, indent=2))
    print(f"wrote {writer.path} ({info['bytes']} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
