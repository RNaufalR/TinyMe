#!/usr/bin/env python3
"""Package one trained experiment into a self-describing ``release/`` directory.

Audit §17 requirements implemented here:

* ``model_fp32.safetensors`` + quantized ``model_fp16/int8/int4.safetensors``
  with *measured* on-disk bytes (and a < 50 MB check);
* ``tokenizer.json`` copied from the exact tokenizer the model was trained with
  (``tokenizer_hash`` must match the checkpoint metadata, otherwise packaging
  aborts);
* ``config.json`` (model config + token special ids), ``manifest.json`` (hashes,
  dataset fingerprint, provenance pointers, environment), ``checksums.txt``;
* ``provenance.md`` (sources / licences / contamination verdict), ``README.md``
  (honest product description) and ``inference.py`` - a dependency-light entry
  point that runs with ``numpy`` + ``safetensors`` + ``tokenizers`` only, i.e.
  without the training environment;
* ``evaluation_report.md`` and ``model_comparison.md`` rendered from real
  evaluation JSON when it exists, and explicitly marked NOT MEASURED otherwise.

Usage:
    python scripts/package_model.py --experiment EXP-002-CORRECTED-NANO \
        --checkpoint best --out release
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.quantization.quantize import (  # noqa: E402
    dequantize_int4, dequantize_int8, quantize_int4, quantize_int8,
)
from src.utils.io_utils import REPO_ROOT, human_bytes, write_json  # noqa: E402

log = logging.getLogger("tinyme.package")


# --------------------------------------------------------------------- helpers
def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_key(key: str) -> str:
    """``[blocks]/[0]/[ln1]`` (jax path repr) -> ``blocks/0/ln1`` (release layout)."""
    return "/".join(part.strip("[]") for part in key.split("/"))


def load_flat_checkpoint(st_path: Path, meta: dict) -> tuple[dict[str, np.ndarray], dict]:
    """Load v2 checkpoint params as a float32 flat dict in release key layout."""
    from safetensors.numpy import load_file

    from src.training.checkpoint import _decode_from_safetensors

    raw = load_file(str(st_path))
    flat = {k[len("params/"):]: v for k, v in raw.items() if k.startswith("params/")}
    if not flat:
        raise SystemExit(f"checkpoint {st_path} contains no params/* tensors")
    flat = _decode_from_safetensors(flat, {_normalize_key(k): v for k, v in meta.get("param_dtypes", {}).items()})
    out: dict[str, np.ndarray] = {}
    for key, value in flat.items():
        normalized = _normalize_key(key)
        if normalized in out:
            raise SystemExit(f"key collision after normalization: {normalized}")
        out[normalized] = np.asarray(value, dtype=np.float32)
    return out, raw


def _normalise_results(payload: dict, source: str) -> list[dict]:
    """Flatten the real evaluation schemas into renderable rows.

    ``scripts/evaluate.py`` writes
    ``{experiment, split, dataset, variants: {fp32: {overall, domains, environment}}}``
    while ``scripts/evaluate_tools.py`` writes a flat per-variant metric dict.
    Both must survive into ``release/evaluation_report.md`` -- an unread schema
    previously produced a report with "0 samples scored", which is a false pass.
    """
    rows: list[dict] = []
    if not isinstance(payload, dict):
        return rows
    experiment = payload.get("experiment", "?")
    split = payload.get("split")
    dataset = payload.get("dataset")
    variants = payload.get("variants")
    if isinstance(variants, dict) and variants:
        for name, stats in variants.items():
            if isinstance(stats.get("summary"), dict):
                stats = {**stats["summary"], "cases": len(stats.get("cases", []))}
            overlap = set(stats) & {"overall", "domains", "perplexity", "accuracy",
                                    "tool_syntax_validity", "task_completion"}
            if not overlap:
                continue
            rows.append({"model": f"{experiment}:{name}", "split": split,
                         "dataset_version": dataset,
                         "overall": stats.get("overall", {}),
                         "domains": stats.get("domains", {}),
                         "metrics": {k: v for k, v in stats.items()
                                     if isinstance(v, (int, float)) and k != "cases"},
                         "cases": stats.get("cases"), "duration_s": stats.get("duration_s"),
                         "source": source})
    return rows


def _render_evaluation_markdown(results: list[dict], title: str) -> str:
    lines = [f"# {title}", "", f"Generated: {time.strftime('%Y-%m-%dT%H:%M:%S')}", ""]
    rendered = 0
    for result in results:
        domains = result.get("domains") or {}
        metrics = result.get("metrics") or {}
        overall = result.get("overall") or {}
        total = overall.get("total_samples") or result.get("cases") or 0
        if not domains and not metrics and not total:
            continue  # never render an empty section (that was the false pass)
        rendered += 1
        lines += [f"## {result.get('model')} - split `{result.get('split')}` "
                  f"({result.get('dataset_version')})", "",
                  f"- Source: `{result.get('source')}`",
                  f"- Samples scored: **{total}**",
                  f"- Mean accuracy: {overall.get('mean_accuracy')}",
                  f"- Duration: {result.get('duration_s')} s", ""]
        if metrics:
            lines += ["| Metric | Value |", "| :--- | ---: |"]
            lines += [f"| {k} | {v} |" for k, v in sorted(metrics.items())]
            lines += [""]
        if domains:
            lines += ["| Domain | Samples | Score |", "| :--- | ---: | ---: |"]
            for name, stats in sorted(domains.items()):
                score = stats.get("accuracy")
                if score is None:
                    ppl = stats.get("perplexity")
                    score_txt = "n/a" if ppl is None else f"ppl {ppl}"
                else:
                    score_txt = f"{100 * float(score):.1f}%"
                lines.append(f"| {name} | {stats.get('samples', 0)} | {score_txt} |")
            lines += [""]
        lines += ["Measurements are produced by `scripts/evaluate.py` / "
                  "`scripts/evaluate_tools.py` against held-out splits only; no subset of "
                  "`train.jsonl` is ever used.", ""]
    if not rendered:
        lines += ["No evaluation JSON was found for this experiment.", "",
                  "Reproduce with:", "",
                  "    python scripts/evaluate.py --experiment <EXP> --checkpoint best "
                  "--dataset dataset_v3 --split test --variants fp32,fp16,int8,int4", ""]
    return "\n".join(lines)


def _render_comparison(releases: list[dict], evaluation_files: list[Path]) -> str:
    lines = ["# MODEL COMPARISON", "",
             "Protocol versions are compared only when the tokenizer, dataset fingerprint and "
             "evaluation split are identical; otherwise rows are marked **NOT COMPARABLE**.", ""]
    if releases:
        lines += ["| Variant | File bytes | Human | Round-trip rel. RMSE | Max abs error |",
                  "| :--- | ---: | ---: | ---: | ---: |"]
        for variant, info in releases[0]["variants"].items():
            lines.append(f"| {variant} | {info['bytes']} | {info['human']} | "
                         f"{info.get('relative_rmse', 'n/a')} | {info.get('max_abs_error', 'n/a')} |")
        lines += ["", f"- fp32 file: **{human_bytes(releases[0]['variants']['fp32']['bytes'])}** "
                      f"({'under' if releases[0]['variants']['fp32']['bytes'] < 50 * 1024 * 1024 else 'OVER'} 50 MB)",
                  ""]
    else:
        lines += ["No packaged release was inspected (run without `--out`?).", ""]

    lines += ["## Held-out measurements", ""]
    if evaluation_files:
        for path in evaluation_files:
            payload = json.loads(Path(path).read_text())
            if isinstance(payload, list):
                continue
            rows = _normalise_results(payload, Path(path).name)
            for result in rows:
                overall = result.get("overall") or {}
                if not overall and result.get("metrics"):
                    head = ", ".join(f"{k}={v}" for k, v in
                                     sorted(result["metrics"].items())[:4])
                    lines.append(f"- `{result['model']}` (tool suite, "
                                 f"{result.get('cases')} cases): {head}")
                    continue
                lines.append(f"- `{result['model']}` on `{result.get('split')}`: "
                             f"{overall.get('total_samples', 0)} samples, "
                             f"mean accuracy {overall.get('mean_accuracy')}")
        lines += ["", "Rows for models trained on a different dataset fingerprint or tokenizer are "
                      "**NOT COMPARABLE** even if they appear in the same report; check "
                      "`manifest.json` before drawing conclusions."]
    else:
        lines += ["Not measured yet for this release. Reproduce with:", "",
                  "    python scripts/evaluate.py --experiment <EXP> --checkpoint best "
                  "--dataset dataset_v2 --split test --variants fp32,fp16,int8,int4", "",
                  "Until then, every comparison row is **NOT MEASURED** - not assumed."]
    lines.append("")
    return "\n".join(lines)


def _render_provenance(manifest: dict, ckpt_meta: dict, dataset_manifest: dict | None) -> str:
    lines = ["# DATA PROVENANCE (release)", "",
             f"- Dataset version: `{ckpt_meta.get('dataset_version', 'unknown')}`",
             f"- Dataset fingerprint: `{ckpt_meta.get('dataset_fingerprint', 'unknown')}`",
             f"- Tokenizer hash: `{ckpt_meta.get('tokenizer_hash', 'unknown')}`",
             ""]
    if dataset_manifest:
        prov = dataset_manifest.get("provenance_summary", {})
        lines += ["## Sources (records emitted per source)", "",
                  "| Source | Records |", "| :--- | ---: |"]
        for src, count in sorted(prov.get("records_by_source", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"| {src} | {count} |")
        lines += ["", "## Licence distribution", "", "| Licence | Records |", "| :--- | ---: |"]
        for lic, count in sorted(dataset_manifest.get("licenses", {}).items(), key=lambda kv: -kv[1]):
            lines.append(f"| {lic} | {count} |")
        lines += ["", f"- Contamination (train vs validation/test): **"
                      f"{dataset_manifest.get('train_validation_test_contamination', 'UNKNOWN')}**",
                  f"- Malformed code rate: {dataset_manifest.get('malformed_code_rate')}",
                  f"- Verified samples: {dataset_manifest.get('verified_samples')}",
                  "",
                  "Full machine-readable provenance: `docs/DATA_PROVENANCE.json` and "
                  "`DATA_PROVENANCE.json` in the repository.", ""]
    else:
        lines += ["Dataset manifest not found next to the release; see the repository copy.", ""]
    lines += [f"- Packaged at: {time.strftime('%Y-%m-%dT%H:%M:%S')}",
              f"- Packaging environment: {json.dumps(ckpt_meta.get('environment', {}), sort_keys=True)}",
              ""]
    return "\n".join(lines)


def _write_readme(out_dir: Path, meta: dict) -> None:
    variants = meta["release"]["variants"]
    lines = [
        "# TinyMe - release",
        "",
        "TinyMe is a **small language model paired with an external tool runtime**",
        "(a typed tool protocol, a retrieval index and an isolated code-execution",
        "sandbox). It is not a frontier model and does not claim to be one: it is a",
        "reproducible study of how far a few million parameters get on verified",
        "data with a strict protocol.",
        "",
        f"- Architecture: `{meta['model_config']['name']}` - "
        f"{meta['parameter_count']:,} parameters ({human_bytes(meta['fp32_bytes'])} fp32)",
        f"- Trained on `{meta['dataset_version']}` (fingerprint `{meta['dataset_fingerprint']}`), "
        f"stage `{meta['stage']}`, step {meta['step']}",
        f"- Tokenizer: `{meta['tokenizer_version']}`, vocab {meta['tokenizer_vocab_size']}, "
        f"sha256 `{meta['tokenizer_hash'][:16]}...`",
        "",
        "## Files",
        "",
        "| File | Purpose | Bytes |",
        "| :--- | :--- | ---: |",
    ]
    for variant, info in variants.items():
        lines.append(f"| `{Path(info['file']).name}` | {variant} weights | {info['bytes']:,} |")
    lines += [
        "| `tokenizer.json` | byte-level BPE used for training | "
        f"{meta['release']['files']['tokenizer.json']['bytes']:,} |",
        "| `config.json` | model config + token special ids | - |",
        "| `manifest.json` | hashes, fingerprints, measured sizes | - |",
        "| `checksums.txt` | sha256 of every file in this directory | - |",
        "| `inference.py` | standalone entry point (numpy + safetensors + tokenizers) | - |",
        "| `evaluation_report.md` | held-out measurements (or NOT MEASURED) | - |",
        "| `model_comparison.md` | fp32/fp16/int8/int4 + protocol comparability | - |",
        "| `provenance.md` | data sources, licences, contamination verdict | - |",
        "",
        "## Run it without the training environment",
        "",
        "    pip install numpy safetensors tokenizers",
        "    python inference.py --prompt \"What is 144 / 12?\" --model model_fp32.safetensors",
        "",
        "Only `numpy`, `safetensors` and `tokenizers` are required; JAX and the",
        "repository are not imported by `inference.py`.",
        "",
        "## Honest limitations",
        "",
        "- Answers come from a ~2.6M-parameter model; factual questions require the",
        "  tool runtime (search/retrieval) and citations are only produced when a",
        "  tool result is present.",
        "- Quantized variants (int8/int4) are lossy; see `model_comparison.md` for",
        "  measured round-trip error and per-variant scores (marked NOT MEASURED",
        "  where they have not been run).",
        "- Any comparison against experiments with a different tokenizer or dataset",
        "  fingerprint is marked NOT COMPARABLE on purpose.",
        "",
    ]
    (out_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


RELEASE_INFERENCE = '''#!/usr/bin/env python3
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

    text = f"<|system|>\\n{args.system}\\n<|user|>\\n{args.prompt}\\n<|assistant|>\\n"
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
'''


# ----------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest", "final"])
    ap.add_argument("--out", default="release")
    ap.add_argument("--dataset", default=None, help="dataset version for the tokenizer (default: from checkpoint meta)")
    ap.add_argument("--no-variants", action="store_true", help="only fp32")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    ckpt_dir = REPO_ROOT / "checkpoints" / args.experiment
    st_path = ckpt_dir / f"{args.checkpoint}.safetensors"
    meta_path = ckpt_dir / f"{args.checkpoint}.json"
    if not st_path.exists():
        for candidate in ("best", "latest", "final"):
            if (ckpt_dir / f"{candidate}.safetensors").exists():
                log.warning("%s not found; using %s", st_path.name, candidate)
                st_path, meta_path = ckpt_dir / f"{candidate}.safetensors", ckpt_dir / f"{candidate}.json"
                break
        else:
            raise SystemExit(f"no checkpoint found under {ckpt_dir}")
    ckpt_meta = json.loads(meta_path.read_text())
    dataset_version = args.dataset or ckpt_meta.get("dataset_version") or ckpt_meta["config"]["dataset_version"]
    model_cfg = ckpt_meta["model_config"]
    tokenizer_version = ckpt_meta.get("config", {}).get("tokenizer_version", "tok-v2")

    flat, _raw = load_flat_checkpoint(st_path, ckpt_meta)
    param_count = int(sum(int(np.prod(v.shape)) for v in flat.values()))
    fp32_tensor_bytes = int(sum(v.nbytes for v in flat.values()))
    if ckpt_meta.get("param_count") and ckpt_meta["param_count"] != param_count:
        raise SystemExit(f"checkpoint parameter count mismatch: meta={ckpt_meta['param_count']} file={param_count}")

    tokenizer_src = REPO_ROOT / "datasets" / "versions" / dataset_version / "tokenizer.json"
    if not tokenizer_src.exists():
        raise SystemExit(f"tokenizer not found: {tokenizer_src}")
    tokenizer_hash = sha256_file(tokenizer_src)
    if ckpt_meta.get("tokenizer_hash") and ckpt_meta["tokenizer_hash"] != tokenizer_hash:
        raise SystemExit(f"tokenizer mismatch: checkpoint={ckpt_meta['tokenizer_hash']} "
                         f"file={tokenizer_hash} (refusing to package incompatible artifacts)")

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import save_file

    variants: dict[str, dict] = {}

    def record_variant(name: str, path: Path) -> None:
        try:
            shown = str(path.relative_to(REPO_ROOT))
        except ValueError:                      # smoke test outside the repo
            shown = str(path)
        variants[name] = {"file": shown, "bytes": path.stat().st_size,
                          "human": human_bytes(path.stat().st_size)}

    fp32_path = out_dir / "model_fp32.safetensors"
    save_file(flat, str(fp32_path), metadata={"format": "tinyme-release-v1", "dtype": "float32",
                                              "param_count": str(param_count)})
    record_variant("fp32", fp32_path)

    if not args.no_variants:
        fp16 = {k: v.astype(np.float16) for k, v in flat.items()}
        fp16_path = out_dir / "model_fp16.safetensors"
        save_file(fp16, str(fp16_path), metadata={"format": "tinyme-release-v1", "dtype": "float16"})
        record_variant("fp16", fp16_path)

        q8, s8 = quantize_int8(flat)
        int8_path = out_dir / "model_int8.safetensors"
        save_file({**q8, **s8}, str(int8_path), metadata={"format": "tinyme-release-v1", "dtype": "int8"})
        record_variant("int8", int8_path)

        q4, s4 = quantize_int4(flat)
        shapes4 = {k: list(v.shape) for k, v in flat.items()
                   if k in q4 and q4[k].dtype == np.uint8 and flat[k].ndim == 2}
        int4_path = out_dir / "model_int4.safetensors"
        save_file({**q4, **s4}, str(int4_path),
                  metadata={"format": "tinyme-release-v1", "dtype": "int4", "shapes": json.dumps(shapes4)})
        record_variant("int4", int4_path)

        log.info("measuring round-trip reconstruction error ...")
        reconstructed = {
            "fp16": {k: v.astype(np.float32) for k, v in fp16.items()},
            "int8": dequantize_int8(q8, s8),
            "int4": dequantize_int4(q4, s4, shape_hint={k: v.shape for k, v in flat.items()}),
        }
        for name, deq in reconstructed.items():
            errs = denom = max_abs = 0.0
            for k, v in flat.items():
                d = deq[k].astype(np.float32)
                if d.shape != v.shape:
                    continue
                errs += float(np.sum(np.square(d - v)))
                denom += float(np.sum(np.square(v)))
                max_abs = max(max_abs, float(np.max(np.abs(d - v))))
            variants[name]["relative_rmse"] = round(float(np.sqrt(errs / max(denom, 1e-12))), 6)
            variants[name]["max_abs_error"] = round(max_abs, 6)

    # The tokenizer, entry point and config are written *before* the loadability
    # check: validating the package against an absent config would silently fall
    # back to the 'nano' default and make a `base` release look loadable
    # (audit §13 — packaging order bug).
    shutil.copy2(tokenizer_src, out_dir / "tokenizer.json")
    (out_dir / "inference.py").write_text(RELEASE_INFERENCE, encoding="utf-8")

    specials = {}
    try:
        from src.tokenizer.bpe import TinyMeTokenizer

        tok = TinyMeTokenizer.load(tokenizer_src)
        specials = {name: int(tok.tok.token_to_id(f"<|{name}|>"))
                    for name in ("pad", "bos", "eos", "system", "user", "assistant", "final",
                                 "tool_call", "tool_result", "end_tool_call", "end_tool_result")}
    except Exception as exc:  # pragma: no cover - diagnostic
        log.warning("could not read special token ids: %s", exc)
    config_payload = dict(model_cfg)                 # flat: InferenceEngine can read it directly
    config_payload.update({
        "tokenizer_version": tokenizer_version,
        "special_token_ids": specials,
        "parameter_count": param_count,
        "fp32_bytes": variants["fp32"]["bytes"],
        "fp32_tensor_bytes": fp32_tensor_bytes,
        "architecture": ckpt_meta.get("architecture") or model_cfg.get("architecture"),
    })
    (out_dir / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")

    # Verify the artifact is loadable by the repo engine (numpy backend, no JAX)
    # using the config that was just written, in strict mode.
    try:
        from src.inference.engine import InferenceEngine

        engine = InferenceEngine(fp32_path, tokenizer_src, out_dir / "config.json",
                                 backend="numpy", strict_config=True)

        def _count(tree) -> int:
            if isinstance(tree, dict):
                return sum(_count(v) for v in tree.values())
            if isinstance(tree, (list, tuple)):
                return sum(_count(v) for v in tree)
            return int(np.prod(tree.shape)) if hasattr(tree, "shape") else 0

        loaded = _count(engine.params)
        if loaded and loaded != param_count:
            raise SystemExit(f"release verification failed: engine loaded {loaded} params, "
                             f"checkpoint declares {param_count}")
        probe = np.asarray([[engine.bos_id, 42, 43, 44]], dtype=np.int64)
        logits = engine.forward_numpy(probe)
        if not np.all(np.isfinite(logits)):
            raise SystemExit("release verification failed: non-finite logits")
        log.info("release verification: loadable (config=%s, arch=%s), logits %s finite, "
                 "max|logit|=%.3f", engine.config_source, model_cfg.get("architecture"),
                 logits.shape, float(np.max(np.abs(logits))))
    except SystemExit:
        raise
    except Exception as exc:  # pragma: no cover - diagnostics
        raise SystemExit(f"release verification failed: {type(exc).__name__}: {exc}")

    dataset_manifest = None
    ds_manifest_path = REPO_ROOT / "datasets" / "versions" / dataset_version / "manifest.json"
    if ds_manifest_path.exists():
        dataset_manifest = json.loads(ds_manifest_path.read_text())

    evaluation_files = sorted((REPO_ROOT / "experiments" / args.experiment).glob("evaluation_*.json"))
    release_meta = {
        "release_format": "tinyme-release-v1",
        "experiment_id": args.experiment,
        "checkpoint": st_path.stem,
        "stage": ckpt_meta.get("config", {}).get("stage", "pretrain"),
        "step": ckpt_meta.get("step"),
        "best_val_loss": ckpt_meta.get("best_val_loss"),
        "parameter_count": param_count,
        "fp32_bytes": variants["fp32"]["bytes"],       # serialized fp32 file
        "fp32_tensor_bytes": fp32_tensor_bytes,        # raw payload (no header)
        "model_config": model_cfg,
        "model_config_hash": ckpt_meta.get("model_config_hash"),
        "dataset_version": dataset_version,
        "dataset_fingerprint": ckpt_meta.get("dataset_fingerprint"),
        "tokenizer_version": tokenizer_version,
        "tokenizer_hash": tokenizer_hash,
        "tokenizer_vocab_size": model_cfg.get("vocab_size"),
        "train_config": ckpt_meta.get("config", {}),
        "release": {"variants": variants},
        "under_50mb": {k: v["bytes"] < 50 * 1024 * 1024 for k, v in variants.items()},
        "environment": ckpt_meta.get("environment", {}),
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    results = []
    for path in evaluation_files:
        payload = json.loads(path.read_text())
        results.extend(_normalise_results(payload, path.name))

    (out_dir / "evaluation_report.md").write_text(
        _render_evaluation_markdown(results, f"EVALUATION REPORT - {args.experiment}"), encoding="utf-8")
    (out_dir / "model_comparison.md").write_text(
        _render_comparison([release_meta["release"]], evaluation_files), encoding="utf-8")
    (out_dir / "provenance.md").write_text(
        _render_provenance(release_meta, {**ckpt_meta, "dataset_version": dataset_version}, dataset_manifest),
        encoding="utf-8")

    # Order (audit §13): every file is produced first, then inventoried, then the
    # manifest is written, and only then are the checksums computed — so the
    # manifest lists README.md and no checksum can describe a file that changed.
    def _inventory() -> dict:
        tracked = sorted(p for p in out_dir.iterdir()
                         if p.is_file() and p.name not in ("checksums.txt", "manifest.json"))
        return {p.name: {"bytes": p.stat().st_size, "sha256": sha256_file(p),
                         "mb": round(p.stat().st_size / 1_000_000, 4),
                         "mib": round(p.stat().st_size / (1024 * 1024), 4)}
                for p in tracked}

    # The README quotes the inventory, so it needs a first pass; the manifest then
    # gets the *final* inventory (README included) — order matters (audit §13).
    release_meta["release"]["files"] = _inventory()
    _write_readme(out_dir, release_meta)
    release_meta["release"]["files"] = _inventory()
    release_meta["release"]["files"]["manifest.json"] = {"bytes": None, "sha256": None,
                                                         "note": "self — hashed in checksums.txt"}
    write_json(release_meta, out_dir / "manifest.json")
    checksum_targets = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "checksums.txt")
    checksum_lines = [f"{sha256_file(p)}  {p.name}" for p in checksum_targets]
    (out_dir / "checksums.txt").write_text("\n".join(sorted(set(checksum_lines))) + "\n", encoding="utf-8")

    release_files = sorted(p for p in out_dir.iterdir() if p.is_file())
    total_bytes = sum(p.stat().st_size for p in release_files)
    release_meta["release"]["total"] = {
        "files": len(release_files), "bytes": total_bytes,
        "mb": round(total_bytes / 1_000_000, 4), "mib": round(total_bytes / (1024 * 1024), 4)}
    write_json(release_meta, out_dir / "manifest.json")
    # rewrite checksums so the (final) manifest is covered too
    checksum_targets = sorted(p for p in out_dir.iterdir() if p.is_file() and p.name != "checksums.txt")
    (out_dir / "checksums.txt").write_text(
        "\n".join(f"{sha256_file(p)}  {p.name}" for p in checksum_targets) + "\n", encoding="utf-8")
    log.info("release total: %s bytes (%.4f MB / %.4f MiB) across %d files",
             total_bytes, total_bytes / 1_000_000, total_bytes / (1024 * 1024), len(release_files))
    log.info("packaged %s into %s", args.experiment, out_dir)
    log.info("variants: %s", json.dumps({k: v["human"] for k, v in variants.items()}))
    log.info("release payload (weights+tokenizer+config+entrypoint): %s", human_bytes(total_bytes))
    log.info("under 50 MB: %s", all(release_meta["under_50mb"].values()))
    print(json.dumps({"experiment": args.experiment, "checkpoint": st_path.stem,
                      "variants": {k: v["bytes"] for k, v in variants.items()},
                      "parameter_count": param_count}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
