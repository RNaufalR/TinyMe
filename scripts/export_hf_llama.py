#!/usr/bin/env python3
"""Export a TinyMe checkpoint to a HuggingFace-compatible llama-architecture tree.

The GGUF ecosystem (llama.cpp, and the runtimes built on it) does not know the
TinyMe module names, so the weights are re-emitted under the canonical
llama names (``embed_tokens``, ``q_proj`` … ``down_proj``, ``input_layernorm``,
``post_attention_layernorm``) together with a ``config.json`` that carries the
real hyper-parameters (RoPE base, RMSNorm eps, head count, context length).
Nothing about the architecture is approximated: TinyMe *is* a pre-norm
decoder-only transformer with RMSNorm + RoPE + SwiGLU and tied embeddings, which
is exactly the llama layout.

The export is written to ``<out>/model.safetensors`` plus
``config.json``/``tokenizer.json``/``tokenizer_config.json``/
``special_tokens_map.json`` so the stock ``convert_hf_to_gguf.py`` from the
chosen runtime can consume the directory unchanged — that keeps the conversion
step itself auditable instead of hand-rolling a container format.

Usage::

    python scripts/export_hf_llama.py --experiment EXP-015-TOOL-SFT-V9 \\
        --checkpoint best --dataset dataset_v9 --out build/hf_export
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402


def _hf_state_dict(params: dict, cfg) -> dict[str, np.ndarray]:
    """Map the TinyMe parameter tree onto llama-style tensor names."""
    out: dict[str, np.ndarray] = {"model.embed_tokens.weight": np.asarray(params["tok_emb"], np.float32)}
    for i, blk in enumerate(params["blocks"]):
        # TinyMe stores Linear weights as (in, out); HF/GGUF want (out, in).
        out[f"model.layers.{i}.self_attn.q_proj.weight"] = np.asarray(blk["attn"]["q"], np.float32).T
        out[f"model.layers.{i}.self_attn.k_proj.weight"] = np.asarray(blk["attn"]["k"], np.float32).T
        out[f"model.layers.{i}.self_attn.v_proj.weight"] = np.asarray(blk["attn"]["v"], np.float32).T
        out[f"model.layers.{i}.self_attn.o_proj.weight"] = np.asarray(blk["attn"]["o"], np.float32).T
        out[f"model.layers.{i}.mlp.gate_proj.weight"] = np.asarray(blk["mlp"]["gate"], np.float32).T
        out[f"model.layers.{i}.mlp.up_proj.weight"] = np.asarray(blk["mlp"]["up"], np.float32).T
        out[f"model.layers.{i}.mlp.down_proj.weight"] = np.asarray(blk["mlp"]["down"], np.float32).T
        out[f"model.layers.{i}.input_layernorm.weight"] = np.asarray(blk["ln1"], np.float32)
        out[f"model.layers.{i}.post_attention_layernorm.weight"] = np.asarray(blk["ln2"], np.float32)
    out["model.norm.weight"] = np.asarray(params["ln_f"], np.float32)
    if not cfg.tie_word_embeddings:
        out["lm_head.weight"] = np.asarray(params["tok_emb"], np.float32)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="EXP-001")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--out", default="build/hf_export")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16"])
    args = ap.parse_args()

    from safetensors.numpy import save_file

    ckpt = ROOT / "checkpoints" / args.experiment / f"{args.checkpoint}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    tok_path = ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json"
    engine = InferenceEngine(ckpt, tok_path, backend="numpy")
    cfg = engine.config

    out_dir = Path(args.out)
    out_dir = out_dir if out_dir.is_absolute() else ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    state = _hf_state_dict(engine.params, cfg)
    if args.dtype == "float16":
        state = {k: v.astype(np.float16) for k, v in state.items()}
    save_file(state, str(out_dir / "model.safetensors"),
              metadata={"format": "pt", "architecture": "LlamaForCausalLM",
                        "source_experiment": args.experiment, "source_checkpoint": args.checkpoint})

    hf_cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "vocab_size": int(cfg.vocab_size),
        "hidden_size": int(cfg.d_model),
        "intermediate_size": int(cfg.d_ff),
        "num_hidden_layers": int(cfg.n_layers),
        "num_attention_heads": int(cfg.n_heads),
        "num_key_value_heads": int(cfg.n_heads),
        "hidden_act": "silu",
        "max_position_embeddings": int(cfg.max_seq_len),
        "initializer_range": 0.02,
        "rms_norm_eps": float(cfg.rms_norm_eps),
        "rope_theta": float(cfg.rope_theta),
        "rope_scaling": None,
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "use_cache": True,
        "attention_bias": False,
        "mlp_bias": False,
        "torch_dtype": "float32" if args.dtype == "float32" else "float16",
        "transformers_version": "n/a-tinyme-export",
    }
    write_json(hf_cfg, out_dir / "config.json")

    # tokenizer: the shipped file is already a HuggingFace tokenizers JSON; the
    # runtime reads it directly, so it is copied next to the weights unchanged.
    shutil.copyfile(tok_path, out_dir / "tokenizer.json")
    tok = load_tokenizer(args.dataset)
    ids = {name: int(tok.tok.token_to_id(name)) for name in
           ("<|pad|>", "<|bos|>", "<|eos|>", "<|unk|>") if tok.tok.token_to_id(name) is not None}
    write_json({"tokenizer_class": "PreTrainedTokenizerFast", "model_max_length": int(cfg.max_seq_len),
                "bos_token": "<|bos|>", "eos_token": "<|eos|>", "pad_token": "<|pad|>",
                "unk_token": "<|unk|>",
                "added_tokens_decoder": {str(v): {"content": k, "special": True}
                                         for k, v in ids.items()}},
               out_dir / "tokenizer_config.json")
    write_json({"bos_token": "<|bos|>", "eos_token": "<|eos|>", "pad_token": "<|pad|>",
                "unk_token": "<|unk|>"}, out_dir / "special_tokens_map.json")
    # generation config: greedy by default; the runtime is what stops the turn
    write_json({"bos_token_id": ids.get("<|bos|>"), "eos_token_id": ids.get("<|eos|>"),
                "pad_token_id": ids.get("<|pad|>"), "do_sample": False,
                "transformers_version": "n/a-tinyme-export"}, out_dir / "generation_config.json")

    n_params = sum(int(np.prod(v.shape)) for v in state.values())
    print(f"exported {len(state)} tensors, {n_params} parameters -> {out_dir}")
    print(f"  hidden={cfg.d_model} layers={cfg.n_layers} heads={cfg.n_heads} ffn={cfg.d_ff} "
          f"vocab={cfg.vocab_size} ctx={cfg.max_seq_len} rope_theta={cfg.rope_theta}")
    print(f"  special ids: {ids}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
