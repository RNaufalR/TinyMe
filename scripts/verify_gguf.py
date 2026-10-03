#!/usr/bin/env python3
"""GGUF artefact + runtime verification (audit §34–§40, §60).

Nothing here trusts the exporter: the file is parsed again from its bytes with an
independent reader, then handed to the *real* runtime binaries and compared
against the native TinyMe engine.

Checks
------
1. ``structure``      — magic/version/tensor count parsed independently; every
                        required llama metadata key present; the tensor list and
                        shapes must match the reference checkpoint exactly.
2. ``tokenizer``      — ``llama-tokenize`` ids vs the shipped tokenizer on a fixed
                        prompt set (the match rate is reported, not assumed).
3. ``perplexity``     — ``llama-perplexity`` on a fixed text vs the native
                        engine's chunked NLL (same tokens, same window).
4. ``generation``     — ``llama-cli`` greedy continuations are non-empty and are
                        stored verbatim for inspection.
5. ``quant_variants`` — ``llama-quantize`` variants are produced, loaded and
                        allowed to generate; bytes/sha256 recorded per variant.

Usage::

    python scripts/verify_gguf.py --gguf release/tinyme-f16.gguf \\
        --reference checkpoints/EXP-015-TOOL-SFT-V9/best.safetensors \\
        --runtime /home/user/llama.cpp/build/bin
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402

GGUF_TYPE_SIZES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
GGUF_STRING, GGUF_ARRAY = 8, 9

REQUIRED_KEYS = (
    "general.architecture", "general.file_type", "general.name",
    "llama.context_length", "llama.embedding_length", "llama.block_count",
    "llama.feed_forward_length", "llama.attention.head_count",
    "llama.attention.head_count_kv", "llama.attention.layer_norm_rms_epsilon",
    "llama.rope.dimension_count", "llama.rope.freq_base", "llama.vocab_size",
    "tokenizer.ggml.model", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges", "tokenizer.ggml.bos_token_id",
)


def _read_gguf(path: Path) -> dict:
    """Independent GGUF v3 reader (header, metadata, tensor infos)."""
    data = path.read_bytes()
    pos = 0
    magic = data[pos:pos + 4]
    pos += 4
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", data, pos)
    pos += 20
    meta: dict = {}
    for _ in range(n_kv):
        klen = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        key = data[pos:pos + klen].decode("utf-8")
        pos += klen
        vtype = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        if vtype == GGUF_STRING:
            slen = struct.unpack_from("<Q", data, pos)[0]
            pos += 8
            meta[key] = data[pos:pos + slen].decode("utf-8")
            pos += slen
        elif vtype == GGUF_ARRAY:
            etype = struct.unpack_from("<I", data, pos)[0]
            count = struct.unpack_from("<Q", data, pos + 4)[0]
            pos += 12
            if etype == GGUF_STRING:
                values = []
                for _ in range(count):
                    slen = struct.unpack_from("<Q", data, pos)[0]
                    pos += 8
                    values.append(data[pos:pos + slen].decode("utf-8"))
                    pos += slen
                meta[key] = values
            else:
                size = GGUF_TYPE_SIZES[etype]
                fmt = {4: "<I", 5: "<i", 6: "<f", 7: "<?", 2: "<H", 3: "<h", 0: "<B", 1: "<b"}[etype]
                meta[key] = [struct.unpack_from(fmt, data, pos + i * size)[0] for i in range(count)]
                pos += size * count
        else:
            fmt = {4: "<I", 5: "<i", 6: "<f", 7: "<?", 2: "<H", 3: "<h", 0: "<B", 1: "<b",
                   10: "<Q", 11: "<q", 12: "<d"}[vtype]
            meta[key] = struct.unpack_from(fmt, data, pos)[0]
            pos += GGUF_TYPE_SIZES[vtype]
    tensors = {}
    for _ in range(n_tensors):
        nlen = struct.unpack_from("<Q", data, pos)[0]
        pos += 8
        name = data[pos:pos + nlen].decode("utf-8")
        pos += nlen
        n_dims = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        dims = list(struct.unpack_from("<" + "Q" * n_dims, data, pos))
        pos += 8 * n_dims
        ttype, offset = struct.unpack_from("<IQ", data, pos)
        pos += 12
        tensors[name] = {"dims": dims, "type": ttype, "offset": offset}
    return {"magic": magic, "version": version, "n_tensors": n_tensors, "n_kv": n_kv,
            "metadata": meta, "tensors": tensors}


def _expected_tensors(engine: InferenceEngine) -> dict[str, list[int]]:
    cfg = engine.config
    expect = {"token_embd.weight": [cfg.d_model, cfg.vocab_size],
              "output_norm.weight": [cfg.d_model]}
    for i in range(cfg.n_layers):
        expect[f"blk.{i}.attn_q.weight"] = [cfg.d_model, cfg.d_model]
        expect[f"blk.{i}.attn_k.weight"] = [cfg.d_model, cfg.d_model]
        expect[f"blk.{i}.attn_v.weight"] = [cfg.d_model, cfg.d_model]
        expect[f"blk.{i}.attn_output.weight"] = [cfg.d_model, cfg.d_model]
        expect[f"blk.{i}.ffn_gate.weight"] = [cfg.d_model, cfg.d_ff]
        expect[f"blk.{i}.ffn_up.weight"] = [cfg.d_model, cfg.d_ff]
        expect[f"blk.{i}.ffn_down.weight"] = [cfg.d_ff, cfg.d_model]
        expect[f"blk.{i}.attn_norm.weight"] = [cfg.d_model]
        expect[f"blk.{i}.ffn_norm.weight"] = [cfg.d_model]
    if not cfg.tie_word_embeddings:
        expect["output.weight"] = [cfg.d_model, cfg.vocab_size]
    return expect


def _run(binary: Path, args: list[str], timeout: int = 1800) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PATH"] = "/usr/bin:/bin"
    env["LD_LIBRARY_PATH"] = str(binary.parent)
    return subprocess.run([str(binary), *args], capture_output=True, text=True,
                          timeout=timeout, env=env)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="release/tinyme-f16.gguf")
    ap.add_argument("--reference", default=None,
                    help="checkpoint the GGUF was exported from (defaults to the manifest)")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--runtime", default="/home/user/llama.cpp/build/bin")
    ap.add_argument("--out", default="docs/audit_evidence/gguf_runtime_test.json")
    ap.add_argument("--quantize-variants", default="Q8_0,Q6_K,Q5_K_M,Q4_K_M")
    ap.add_argument("--skip-quantize", action="store_true")
    args = ap.parse_args()

    gguf = Path(args.gguf)
    gguf = gguf if gguf.is_absolute() else ROOT / gguf
    runtime = Path(args.runtime)
    tok = load_tokenizer(args.dataset)
    report: dict = {"gguf": str(gguf), "runtime": str(runtime), "checks": {}}
    checks = report["checks"]

    if not gguf.exists():
        print(f"FAIL  gguf missing: {gguf}")
        return 1
    if not (runtime / "llama-cli").exists():
        print(f"FAIL  runtime binaries missing in {runtime}")
        return 1

    # ------------------------------------------------------------------ structure
    parsed = _read_gguf(gguf)
    meta = parsed["metadata"]
    missing_keys = [k for k in REQUIRED_KEYS if k not in meta]
    reference = Path(args.reference) if args.reference else None
    if reference is None or not reference.exists():
        reference = ROOT / "checkpoints" / meta.get("general.source.experiment", "") / \
            f"{meta.get('general.source.checkpoint', 'best')}.safetensors"
    engine = InferenceEngine(reference, ROOT / "datasets" / "versions" / args.dataset /
                             "tokenizer.json", backend="numpy") if reference.exists() else None
    shape_report = {}
    if engine is not None:
        expect = _expected_tensors(engine)
        for name, dims in expect.items():
            got = parsed["tensors"].get(name)
            shape_report[name] = "ok" if got and got["dims"] == dims else \
                f"expected {dims}, got {got['dims'] if got else None}"
    shapes_ok = all(v == "ok" for v in shape_report.values()) if shape_report else False
    checks["structure"] = {
        "ok": (parsed["magic"] == b"GGUF" and parsed["version"] == 3 and not missing_keys
               and shapes_ok and meta.get("general.architecture") == "llama"),
        "magic": parsed["magic"].decode("latin-1"), "version": parsed["version"],
        "tensor_count": parsed["n_tensors"], "kv_count": parsed["n_kv"],
        "missing_metadata_keys": missing_keys,
        "tensor_shapes": {k: v for k, v in shape_report.items() if v != "ok"} or "all match",
        "sha256": hashlib.sha256(gguf.read_bytes()).hexdigest(),
        "bytes": gguf.stat().st_size,
        "decimal_mb": round(gguf.stat().st_size / 1e6, 3),
        "mib": round(gguf.stat().st_size / (1024 * 1024), 3),
        "key_metadata": {k: (v if not isinstance(v, list) else f"[{len(v)} items]")
                         for k, v in meta.items() if k in REQUIRED_KEYS},
    }

    # ---------------------------------------------------------------- tokenizer
    prompts = ["What is 21 + 34?", "Add 53 and 4699.",
               "Transcribe only the code that follows; ticket 7936 is irrelevant. "
               "Code: 4a26e73e-88a9-ccf1-ec6e-d68aeba6fd26",
               "def add(a, b):\n    return a + b\n",
               "Compute exactly: 4837 * 962 + 71", "Fetch source_id FACT-2082DD and summarise it.",
               "Reverse the string 'abc'.", "Jakarta has 10,679,951 inhabitants",
               "symmetric_difference_update", "Execute this and tell me the printed number: 14"]
    tok_results, matched = [], 0
    for prompt in prompts:
        proc = _run(runtime / "llama-tokenize", ["-m", str(gguf), "-p", prompt, "--ids"], timeout=300)
        line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
        try:
            ids = json.loads(re.sub(r"^.*?(\[)", r"\1", line))
        except Exception:                                   # noqa: BLE001
            ids = None
        mine = tok.encode_ids(prompt)
        ok = bool(ids) and ids[1:] == mine                  # llama adds BOS; TinyMe's encoder does not
        matched += int(ok)
        tok_results.append({"prompt": prompt[:60], "llama_ids": ids[:12] if ids else None,
                            "tinyme_ids": mine[:12], "match_excluding_bos": ok})
    checks["tokenizer"] = {"ok": matched == len(prompts), "matched": matched,
                           "total": len(prompts), "cases": tok_results}

    # --------------------------------------------------------------- perplexity
    if engine is not None:
        texts = []
        with (ROOT / "datasets" / "versions" / args.dataset / "validation.jsonl").open() as fh:
            for line in fh:
                rec = json.loads(line)
                if not rec.get("segments"):
                    texts.append(rec["text"].replace("\n", " "))
                if sum(len(t) for t in texts) > 8000:
                    break
        text = "\n".join(texts)
        ppl_file = Path("/tmp/tinyme_gguf_ppl.txt")
        ppl_file.write_text(text, encoding="utf-8")
        ids = tok.encode_ids(text)
        ctx = int(engine.config.max_seq_len)
        vals = []
        for start in range(0, len(ids) - 1, ctx):
            win = ids[start:start + ctx]
            if len(win) < 8:
                break
            logits = engine.forward(np.asarray(win, dtype=np.int32)[None, :])[0].astype(np.float64)
            logits -= logits.max(-1, keepdims=True)
            logp = logits - np.log(np.exp(logits).sum(-1, keepdims=True))
            target = np.asarray(win[1:])
            vals.extend((-logp[np.arange(len(target)), target]).tolist())
        native_nll = float(np.mean(vals)) if vals else float("nan")
        proc = _run(runtime / "llama-perplexity",
                    ["-m", str(gguf), "-f", str(ppl_file), "-c", str(ctx), "-t", "2"], timeout=3600)
        # llama.cpp logs to stderr; search both streams so a missing line is a real failure
        match = re.search(r"Final estimate: PPL =\s*([0-9.]+)", proc.stdout + proc.stderr)
        gguf_ppl = float(match.group(1)) if match else None
        gguf_nll = float(np.log(gguf_ppl)) if gguf_ppl else None
        rel = abs(gguf_nll - native_nll) / abs(native_nll) if gguf_nll else None
        checks["perplexity_parity"] = {
            "ok": bool(gguf_nll) and rel is not None and rel < 0.05,
            "native_mean_nll": round(native_nll, 5), "native_ppl": round(float(np.exp(native_nll)), 3),
            "gguf_ppl": gguf_ppl, "gguf_mean_nll": round(gguf_nll, 5) if gguf_nll else None,
            "relative_difference": round(rel, 5) if rel is not None else None,
            "tokens": len(vals), "context": ctx,
            "note": "small differences are expected: the GGUF stores f16 weights while the "
                    "native reference runs f32",
        }

    # --------------------------------------------------------------- generation
    gens = []
    for prompt in prompts[:4]:
        proc = _run(runtime / "llama-cli",
                    ["-m", str(gguf), "-p", prompt, "-n", "24", "--temp", "0",
                     "--no-display-prompt", "--no-warmup", "-t", "2", "--seed", "0",
                     "--ignore-eos", "--no-jinja"], timeout=900)
        raw = proc.stdout.strip()
        # the CLI echoes the prompt and a timing line; cut both so the record holds
        # only the continuation the runtime actually produced
        body = raw.split(prompt, 1)[-1] if prompt in raw else raw
        body = body.split("[ Prompt:")[0].strip("\n> ")
        native = engine.generate(prompt, max_new_tokens=24, temperature=0.0,
                                 repetition_penalty=1.0) if engine is not None else ""
        gens.append({"prompt": prompt[:60], "gguf": body[:200], "native": native[:200]})
    checks["generation"] = {"ok": all(g["gguf"].strip() for g in gens), "cases": gens}

    # ----------------------------------------------------------- quant variants
    if not args.skip_quantize:
        variants = {}
        base = gguf
        for variant in args.quantize_variants.split(","):
            out = gguf.with_name(f"{gguf.stem}-{variant.lower()}.gguf")
            proc = _run(runtime / "llama-quantize", [str(base), str(out), variant], timeout=1800)
            entry: dict = {"returncode": proc.returncode,
                           "log_tail": (proc.stdout + proc.stderr).strip().splitlines()[-3:]}
            if out.exists():
                entry.update({"bytes": out.stat().st_size,
                              "decimal_mb": round(out.stat().st_size / 1e6, 3),
                              "mib": round(out.stat().st_size / (1024 * 1024), 3),
                              "sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
                              "relative_to_f16": round(out.stat().st_size / gguf.stat().st_size, 4)})
                smoke = _run(runtime / "llama-cli",
                             ["-m", str(out), "-p", "Add 53 and 4699.", "-n", "8", "--temp", "0",
                              "--no-display-prompt", "--no-warmup", "-t", "2", "--ignore-eos",
                              "--no-jinja"], timeout=900)
                entry["loads_and_runs"] = smoke.returncode == 0
                entry["smoke_tail"] = smoke.stdout.strip()[-120:]
            else:
                entry["loads_and_runs"] = False
            variants[variant] = entry
        checks["quant_variants"] = {"ok": all(v.get("loads_and_runs") for v in variants.values()),
                                    "variants": variants}

    ok = all(c["ok"] for c in checks.values())
    report["ok"] = ok
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, out)
    for name, check in checks.items():
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {name}")
    print(f"gguf verification: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
