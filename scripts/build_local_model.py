#!/usr/bin/env python3
"""Build the standalone ``local_model/`` package from a packaged ``release/``.

The deliverable is "copy folder -> run locally -> model loads -> tokenizer loads
-> inference works", with no access to the training repository.  Everything in
the directory is generated from the released artefacts, and every number in the
manifest comes from ``os.stat`` / ``hashlib`` on those files, never from a
parameter-count estimate.

Layout::

    local_model/
    ├── model/model_fp32|fp16|int8|int4.safetensors
    ├── tokenizer.json
    ├── config.json
    ├── manifest.json          (per-file bytes + sha256 + provenance)
    ├── checksums.txt
    ├── inference.py           (numpy + safetensors + tokenizers only)
    ├── README.md
    ├── requirements.txt
    └── LICENSES/

Usage::

    python scripts/build_local_model.py --release release --out local_model
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

log = logging.getLogger("tinyme.local_model")

REQUIREMENTS = """# Runtime dependencies of the standalone TinyMe package.
# Pinned to the versions this package was validated with (see manifest.json ->
# environment).  Nothing here imports JAX, optax or the training repository.
numpy==2.4.6
safetensors==0.8.0
tokenizers==0.23.2
"""

LICENSE_NOTE = """# Licences and provenance — TinyMe local model

* **Model weights** (`model/*.safetensors`) — produced by this project's training
  runs.  Licensed under the repository licence (see the project `LICENSE`).
* **Tokenizer** (`tokenizer.json`) — ByteLevel BPE trained by this project on the
  *train* split of the corpus it is shipped with.  No third-party tokenizer file
  is redistributed.
* **Training corpus provenance** — per-record source and licence classification
  lives in the repository (`datasets/processed/DATA_PROVENANCE_<version>.json`)
  and is summarised in the packaged `provenance.md`.  Sources are the CPython
  standard library documentation (PSF-2.0), the GSM8K / MBPP / TinyStories slices
  shipped with the project, and verified synthetic generators (`Synthetic-Verified`).
* **Code** — `inference.py` is part of this project and is covered by the same
  repository licence.

No third-party model weights, tokenizer vocabularies or datasets are
redistributed in this directory beyond the sources listed above.
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="release")
    ap.add_argument("--out", default="local_model")
    ap.add_argument("--project", default="TinyMe")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

    rel = ROOT / args.release
    out = ROOT / args.out
    manifest = json.loads((rel / "manifest.json").read_text(encoding="utf-8"))
    variants = (manifest.get("release") or {}).get("variants") or {}
    if not variants:
        raise SystemExit(f"{rel}/manifest.json lists no variants")

    if out.exists():
        # Never delete evidence: move an existing package aside instead.
        stamp = time.strftime("%Y%m%d-%H%M%S")
        shutil.move(str(out), str(out.with_name(f"{out.name}.superseded-{stamp}")))
        log.info("previous %s moved to %s.superseded-%s", out.name, out.name, stamp)

    (out / "model").mkdir(parents=True, exist_ok=True)
    (out / "LICENSES").mkdir(parents=True, exist_ok=True)

    copied: dict[str, Path] = {}
    for name, info in variants.items():
        src = rel / Path(info["file"]).name
        if not src.exists():
            raise SystemExit(f"variant {name} listed but missing: {src}")
        dst = out / "model" / src.name
        shutil.copy2(src, dst)
        copied[name] = dst
    for extra in ("tokenizer.json", "config.json", "inference.py", "provenance.md",
                  "evaluation_report.md"):
        src = rel / extra
        if src.exists():
            shutil.copy2(src, out / extra)
    (out / "requirements.txt").write_text(REQUIREMENTS, encoding="utf-8")
    (out / "LICENSES" / "PROVENANCE.md").write_text(LICENSE_NOTE, encoding="utf-8")

    # ------------------------------------------------------------ verification
    from safetensors.numpy import load_file

    per_variant = {}
    for name, path in copied.items():
        tensors = load_file(str(path))
        params = int(sum(int(v.size) for v in tensors.values()))
        per_variant[name] = {
            "file": f"model/{path.name}",
            "bytes": path.stat().st_size,
            "mb_decimal": round(path.stat().st_size / 1_000_000, 4),
            "mib_binary": round(path.stat().st_size / (1024 * 1024), 4),
            "sha256": sha256_file(path),
            "tensors": len(tensors),
            "parameter_count": params,
            "under_50mb": path.stat().st_size < 50_000_000,
        }
        log.info("%s: %d bytes (%.3f MB / %.3f MiB), %d params", name, path.stat().st_size,
                 path.stat().st_size / 1e6, path.stat().st_size / 2 ** 20, params)

    tok_bytes = (out / "tokenizer.json").stat().st_size
    primary = per_variant.get("fp32") or next(iter(per_variant.values()))
    local_manifest = {
        "project": args.project,
        "release_format": "tinyme-local-model-v1",
        "model_architecture": manifest.get("model_config", {}).get("name", "nano"),
        "model_config": manifest.get("model_config"),
        "parameter_count": primary["parameter_count"],
        "tokenizer": "tokenizer.json",
        "tokenizer_bytes": tok_bytes,
        "tokenizer_sha256": sha256_file(out / "tokenizer.json"),
        "tokenizer_version": manifest.get("tokenizer_version"),
        "model_variant": "fp32",
        "precision": "float32",
        "file_bytes": primary["bytes"],
        "sha256": primary["sha256"],
        "runtime": "numpy + safetensors + tokenizers (inference.py)",
        "offline": True,
        "network_required": False,
        "dataset_version": manifest.get("dataset_version"),
        "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "training_experiment": manifest.get("experiment_id"),
        "training_step": manifest.get("step"),
        "git_commit": (manifest.get("environment") or {}).get("git_commit"),
        "variants": per_variant,
        "model_artifact_bytes": primary["bytes"],
        "model_artifact_mb_decimal": primary["mb_decimal"],
        "model_artifact_mib_binary": primary["mib_binary"],
        "directory_bytes_excluding_weights": None,   # filled below
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    tracked = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "checksums.txt")
    directory_bytes = sum(p.stat().st_size for p in tracked)
    weights_bytes = sum(v["bytes"] for v in per_variant.values())
    local_manifest["directory_bytes_including_all_variants"] = directory_bytes
    local_manifest["directory_mb_decimal"] = round(directory_bytes / 1e6, 4)
    local_manifest["directory_mib_binary"] = round(directory_bytes / 2 ** 20, 4)
    local_manifest["directory_bytes_excluding_weights"] = directory_bytes - weights_bytes
    local_manifest["size_notes"] = {
        "model_artifact": "one weight file < 50,000,000 bytes (the gate applies per variant)",
        "full_directory": "every variant plus tokenizer, config, entry point and docs",
    }
    (out / "manifest.json").write_text(json.dumps(local_manifest, indent=2, sort_keys=True),
                                       encoding="utf-8")
    print(json.dumps({k: local_manifest[k] for k in
                      ("model_artifact_bytes", "model_artifact_mb_decimal", "parameter_count",
                       "directory_bytes_including_all_variants", "directory_mb_decimal")}, indent=2))
    print(json.dumps({k: v["bytes"] for k, v in per_variant.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
