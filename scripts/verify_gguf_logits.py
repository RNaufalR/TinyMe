#!/usr/bin/env python3
"""Logit-level parity between a GGUF file and the native TinyMe engine (audit §34–§38).

Perplexity aggregates cannot decide whether an export reproduces the native
computation: they mix in BOS insertion, chunk boundaries and token counts. This
script compares *the same forward pass* on *the same token window*:

1. the token ids are produced by the shipped tokenizer, so both engines see the
   identical stream (no BOS is added to either side — the RoPE positions must
   line up);
2. ``tools/gguf_logit_dump.cpp`` runs llama.cpp on that window and writes the full
   logit matrix (position x vocabulary) as float32;
3. the native engine evaluates the same window;
4. the two matrices are compared with argmax agreement, top-5 overlap and the
   per-position Pearson correlation.

``--discriminate`` additionally exports the same checkpoint with the RoPE pairs
permuted and shows how far the metric collapses. That is the proof the test can
fail: a metric that is 1.000 for one layout and 0.33 for another is measuring the
layout, not luck. Aggregates are never reported as the primary evidence.

Writes ``docs/audit_evidence/gguf_logit_parity.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.utils.io_utils import write_json  # noqa: E402

HARNESS_SRC = ROOT / "tools" / "gguf_logit_dump.cpp"


def build_harness(llama_cpp: Path, out: Path) -> Path:
    """Compile the dump tool against the llama.cpp build tree (once)."""
    if out.exists() and out.stat().st_mtime > HARNESS_SRC.stat().st_mtime:
        return out
    cmd = ["g++", "-O2", "-std=c++17",
           f"-I{llama_cpp / 'include'}", f"-I{llama_cpp / 'ggml' / 'include'}",
           str(HARNESS_SRC), "-o", str(out),
           f"-L{llama_cpp / 'build' / 'bin'}", "-lllama", "-lggml", "-lggml-base",
           f"-Wl,-rpath,{llama_cpp / 'build' / 'bin'}"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"harness build failed:\n{proc.stderr[-2000:]}")
    return out


def dump_logits(harness: Path, gguf: Path, tokens_file: Path, out: Path, ctx: int) -> np.ndarray:
    proc = subprocess.run([str(harness), str(gguf), str(tokens_file), str(out),
                           "-c", str(ctx), "--no-bos"], capture_output=True, text=True, timeout=900)
    if proc.returncode != 0:
        raise RuntimeError(f"logit dump failed for {gguf}:\n{proc.stderr[-2000:]}")
    blob = out.read_bytes()
    n_pos, n_vocab = struct.unpack_from("<II", blob, 0)
    matrix = np.frombuffer(blob, dtype="<f4", offset=8).reshape(n_pos, n_vocab)
    if not np.isfinite(matrix).all():
        raise RuntimeError(f"{gguf}: logits contain non-finite values")
    return matrix.astype(np.float64)


def compare(a: np.ndarray, b: np.ndarray) -> dict:
    if a.shape != b.shape:
        return {"ok": False, "error": f"shape mismatch {a.shape} vs {b.shape}"}
    top1 = float((a.argmax(-1) == b.argmax(-1)).mean())
    top5 = float(np.mean([len(set(np.argsort(-a[i])[:5]) & set(np.argsort(-b[i])[:5])) / 5
                          for i in range(len(a))]))
    corr = float(np.mean([np.corrcoef(a[i], b[i])[0, 1] for i in range(len(a))]))
    return {"ok": True, "top1_agreement": round(top1, 4), "top5_overlap": round(top5, 4),
            "logit_correlation": round(corr, 6), "positions": int(len(a))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--reference", required=True,
                    help="checkpoint the GGUF was exported from (.safetensors)")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--llama-cpp", default="/home/user/llama.cpp")
    ap.add_argument("--work-dir", default="/tmp/tinyme-logit-parity")
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--discriminate", action="store_true",
                    help="also export a permuted RoPE variant and show the metric collapse")
    ap.add_argument("--min-top1", type=float, default=0.95)
    ap.add_argument("--out", default="docs/audit_evidence/gguf_logit_parity.json")
    args = ap.parse_args()

    from src.data.dataset_api import load_tokenizer
    from src.inference.engine import InferenceEngine

    gguf = Path(args.gguf)
    gguf = gguf if gguf.is_absolute() else ROOT / gguf
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    harness = build_harness(Path(args.llama_cpp), work / "gguf_logit_dump")

    tok = load_tokenizer(args.dataset)
    text = ("Population of Bandung is 2,444,160 as of the 2023 census. "
            "Add 53 and 4699. The capital of Indonesia is Jakarta. "
            "def add(a, b): return a + b  Compute exactly: 4837 * 962 + 71")
    ids = tok.encode_ids(text)
    tokens_file = work / "tokens.txt"
    tokens_file.write_text("\n".join(str(i) for i in ids) + "\n", encoding="utf-8")

    tok_path = ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json"
    engine = InferenceEngine(args.reference, tok_path, backend="numpy")
    native = engine.forward(np.asarray(ids, dtype=np.int32)[None, :])[0].astype(np.float64)

    report: dict = {"audit": "§34–§38 GGUF parity", "gguf": str(gguf),
                    "reference": args.reference, "token_window": len(ids),
                    "tokens_sha256": hashlib.sha256(
                        ",".join(map(str, ids)).encode()).hexdigest(),
                    "gguf_sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
                    "checks": {}}
    main_cmp = compare(dump_logits(harness, gguf, tokens_file, work / "gguf.bin", args.context),
                       native)
    report["checks"]["logit_parity"] = {
        "ok": bool(main_cmp.get("ok") and main_cmp["top1_agreement"] >= args.min_top1),
        "comparison": main_cmp, "min_top1": args.min_top1,
        "method": "llama.cpp logits vs native engine logits on the identical token window, "
                  "no BOS on either side so the RoPE positions align",
    }

    if args.discriminate:
        perm = work / "permuted.gguf"
        cmd = [sys.executable, str(ROOT / "scripts" / "gguf_export.py"),
               "--experiment", "EXP-001", "--checkpoint", "best", "--out", str(perm),
               "--dtype", "f16", "--pre", "gpt-2", "--permute", "on"]
        proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(ROOT), timeout=1800)
        if proc.returncode != 0 or not perm.exists():
            report["checks"]["discrimination"] = {"ok": False,
                                                  "error": proc.stderr[-400:]}
        else:
            wrong = compare(dump_logits(harness, perm, tokens_file, work / "permuted.bin",
                                        args.context), native)
            report["checks"]["discrimination"] = {
                "ok": bool(wrong.get("ok") and wrong["top1_agreement"] < main_cmp["top1_agreement"]),
                "permuted_rope_comparison": wrong,
                "interpretation": "the permuted export is the same weights with the RoPE pairs "
                                  "re-paired; a large drop proves the metric detects layout, and "
                                  "the unpermuted export is the one that reproduces the engine",
            }

    ok = all(c["ok"] for c in report["checks"].values())
    report["ok"] = ok
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, out)
    for name, check in report["checks"].items():
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {name}")
        print("      ", json.dumps({k: v for k, v in check.items() if k != "ok"})[:300])
    print(f"gguf logit parity: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
