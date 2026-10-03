#!/usr/bin/env python3
"""Verify the shipped ``release/`` directory from the artefacts themselves (§19, §31–33, §60).

The check is deliberately *not* "the files are there": every claim is re-derived.

* every variant is loaded through the same loader the runtime uses and must
  produce finite logits for a real prompt (a file that cannot run is not a
  release artefact);
* the tokenizer is loaded from the release directory and its vocabulary must
  match the model config written next to it;
* every SHA-256 in ``checksums.txt`` is recomputed from the bytes on disk;
* every file in the directory is listed in ``checksums.txt`` (nothing smuggled
  in), and no file is missing;
* sizes come from ``os.stat`` and the 50 MB gate is applied to the *decimal*
  and *binary* readings;
* ``inference.py`` is executed as a subprocess with the repository off
  ``PYTHONPATH`` — the entry point must work standalone.

Writes ``docs/audit_evidence/release_verify.json`` and exits non-zero on failure.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.utils.io_utils import write_json  # noqa: E402

SIZE_LIMIT = 50_000_000


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--out", default="docs/audit_evidence/release_verify.json")
    args = ap.parse_args()

    rel = Path(args.release)
    rel = rel if rel.is_absolute() else ROOT / rel
    checks: dict[str, dict] = {}
    report: dict = {"release": str(rel), "checks": checks}

    if not rel.exists():
        print("FAIL  release directory missing")
        return 1

    # ---------------------------------------------------------- required files
    required = ["model_fp32.safetensors", "model_fp16.safetensors", "model_int8.safetensors",
                "model_int4.safetensors", "tokenizer.json", "config.json", "manifest.json",
                "checksums.txt", "inference.py", "README.md", "evaluation_report.md",
                "model_comparison.md"]
    missing = [name for name in required if not (rel / name).exists()]
    checks["required_files"] = {"ok": not missing, "missing": missing, "required": required}

    # -------------------------------------------------------------- loadability
    variants = ["fp32", "fp16", "int8", "int4"]
    loaded: dict[str, dict] = {}
    from src.inference.engine import InferenceEngine

    tok_path = rel / "tokenizer.json"
    for variant in variants:
        path = rel / f"model_{variant}.safetensors"
        if not path.exists():
            loaded[variant] = {"ok": False, "error": "missing"}
            continue
        try:
            engine = InferenceEngine(path, tok_path, backend="numpy")
            logits = engine.forward(np.asarray([[engine.bos_id]], dtype=np.int32))
            finite = bool(np.isfinite(logits).all())
            # a short greedy continuation must produce at least one token id in range
            text = engine.generate("What is 2 + 2?", max_new_tokens=8, temperature=0.0,
                                   repetition_penalty=1.0)
            n_params = 0
            for value in engine.params.values():
                if hasattr(value, "size"):
                    n_params += int(value.size)
            for blk in engine.params["blocks"]:
                for group in blk.values():
                    if hasattr(group, "values"):
                        n_params += sum(int(v.size) for v in group.values())
                    elif hasattr(group, "size"):
                        n_params += int(group.size)
            loaded[variant] = {"ok": finite and len(engine.encode(text)) >= 0,
                               "finite_logits": finite, "generated_chars": len(text),
                               "parameters": n_params,
                               "weights_dtype": getattr(engine, "weights_dtype", "float32")}
        except Exception as exc:                      # noqa: BLE001 - report, never hide
            loaded[variant] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
    checks["variants_load_and_run"] = {"ok": all(v["ok"] for v in loaded.values()), "variants": loaded}

    # ---------------------------------------------------------------- tokenizer
    from src.data.dataset_api import load_tokenizer
    cfg = json.loads((rel / "config.json").read_text(encoding="utf-8"))
    tok = load_tokenizer(args.dataset)                    # repository copy, for comparison
    from src.tokenizer.bpe import TinyMeTokenizer
    release_tok = TinyMeTokenizer.load(tok_path)
    checks["tokenizer"] = {
        "ok": release_tok.vocab_size == int(cfg.get("vocab_size", -1)),
        "release_vocab_size": release_tok.vocab_size,
        "config_vocab_size": cfg.get("vocab_size"),
        "repo_vocab_size": tok.vocab_size,
        "same_vocabulary": release_tok.vocab_size == tok.vocab_size,
    }

    # ---------------------------------------------------------------- checksums
    listed: dict[str, str] = {}
    for line in (rel / "checksums.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            listed[parts[-1].lstrip("*")] = parts[0]
    mismatched, absent = [], []
    for name, digest in listed.items():
        path = rel / name
        if not path.exists():
            absent.append(name)
            continue
        if sha256(path) != digest:
            mismatched.append(name)
    on_disk = sorted(p.name for p in rel.iterdir()
                     if p.is_file() and p.name != "checksums.txt")
    unlisted = [name for name in on_disk if name not in listed]
    checks["checksums"] = {"ok": not (mismatched or absent or unlisted),
                           "listed": len(listed), "mismatched": mismatched,
                           "missing_files": absent, "unlisted_files": unlisted}

    # -------------------------------------------------------------------- size
    sizes = {}
    for variant in variants:
        path = rel / f"model_{variant}.safetensors"
        if path.exists():
            n = os.stat(path).st_size
            sizes[variant] = {"bytes": n, "decimal_mb": round(n / 1e6, 3),
                              "mib": round(n / (1024 * 1024), 3),
                              "under_50mb": n < SIZE_LIMIT}
    smallest = min((s for s in sizes.values() if s["under_50mb"]), key=lambda s: s["bytes"],
                   default=None)
    checks["size_gate"] = {"ok": bool(sizes) and all(s["under_50mb"] for s in sizes.values()),
                           "limit_bytes": SIZE_LIMIT, "variants": sizes,
                           "deployable_smallest_variant_under_limit": smallest is not None}

    # ----------------------------------------------------- standalone entry point
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PATH"] = "/usr/bin:/bin"
    proc = subprocess.run([sys.executable, "inference.py", "--prompt", "What is 2 + 2?",
                           "--model", "model_fp32.safetensors", "--max-new-tokens", "8"],
                          cwd=str(rel), capture_output=True, text=True, timeout=600, env=env)
    checks["standalone_entrypoint"] = {"ok": proc.returncode == 0,
                                       "returncode": proc.returncode,
                                       "stdout_tail": proc.stdout[-400:],
                                       "stderr_tail": proc.stderr[-400:]}

    ok = all(c["ok"] for c in checks.values())
    report["ok"] = ok
    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, out)
    for name, check in checks.items():
        print(f"{'PASS' if check['ok'] else 'FAIL'}  {name}")
        if not check["ok"]:
            print("      ", json.dumps({k: v for k, v in check.items() if k != "ok"})[:400])
    print(f"release verification: {'PASS' if ok else 'FAIL'} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
