#!/usr/bin/env python3
"""Quantize a checkpoint and measure the actual deployable artifact size.

Usage:
    python scripts/quantize.py --experiment EXP-001 --checkpoint best --out release/
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from src.quantization.quantize import (  # noqa: E402
    dequantize_int4, dequantize_int8, quantize_int4, quantize_int8,
)
from src.utils.io_utils import REPO_ROOT, human_bytes, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="EXP-001")
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--out", default="release")
    ap.add_argument("--arch", default="nano")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    log = logging.getLogger("quantize")

    ckpt = REPO_ROOT / "checkpoints" / args.experiment / f"{args.checkpoint}.safetensors"
    if not ckpt.exists():
        log.error("checkpoint not found: %s", ckpt)
        return 1
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    from safetensors.numpy import load_file, save_file

    flat = {k: v.astype(np.float32) for k, v in load_file(str(ckpt)).items()}
    n_params = sum(int(np.prod(v.shape)) for v in flat.values())

    results: dict = {
        "experiment": args.experiment, "checkpoint": args.checkpoint,
        "architecture": args.arch, "parameter_count": n_params, "variants": {},
    }

    def record(name: str, path: Path) -> None:
        results["variants"][name] = {
            "file": str(path.relative_to(REPO_ROOT)), "bytes": path.stat().st_size,
            "human": human_bytes(path.stat().st_size),
        }

    fp32_path = out_dir / "model_fp32.safetensors"
    save_file(flat, str(fp32_path), metadata={"format": "pt", "dtype": "float32"})
    record("fp32", fp32_path)

    fp16 = {k: v.astype(np.float16) for k, v in flat.items()}
    fp16_path = out_dir / "model_fp16.safetensors"
    save_file(fp16, str(fp16_path), metadata={"format": "pt", "dtype": "float16"})
    record("fp16", fp16_path)

    q8, s8 = quantize_int8(flat)
    payload8 = {**q8, **s8}
    int8_path = out_dir / "model_int8.safetensors"
    save_file(payload8, str(int8_path), metadata={"format": "pt", "dtype": "int8"})
    record("int8", int8_path)

    q4, s4 = quantize_int4(flat)
    payload4 = {**q4, **s4}
    int4_path = out_dir / "model_int4.safetensors"
    shapes4 = {k: list(v.shape) for k, v in flat.items() if k in q4 and q4[k].dtype == np.uint8}
    save_file(payload4, str(int4_path),
              metadata={"format": "pt", "dtype": "int4", "shapes": json.dumps(shapes4)})
    record("int4", int4_path)

    log.info("verifying round-trip reconstruction error ...")
    variants = {
        "fp16": {k: v.astype(np.float32) for k, v in fp16.items()},
        "int8": dequantize_int8(q8, s8),
        "int4": dequantize_int4(q4, s4, shape_hint={k: v.shape for k, v in flat.items()}),
    }
    for name, deq in variants.items():
        errs, denom, max_abs = 0.0, 0.0, 0.0
        for k, v in flat.items():
            d = deq[k].astype(np.float32)
            if d.shape != v.shape:
                continue
            errs += float(np.sum(np.square(d - v)))
            denom += float(np.sum(np.square(v)))
            max_abs = max(max_abs, float(np.max(np.abs(d - v))))
        results["variants"][name]["relative_rmse"] = round(float(np.sqrt(errs / max(denom, 1e-12))), 6)
        results["variants"][name]["max_abs_error"] = round(max_abs, 6)

    results["under_50mb"] = {k: v["bytes"] < 50 * 1024 * 1024
                             for k, v in results["variants"].items()}
    write_json(results, out_dir / "quantization_report.json")
    log.info("quantization report:\n%s", json.dumps(results["variants"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
