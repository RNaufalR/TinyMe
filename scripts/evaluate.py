#!/usr/bin/env python3
"""Evaluate a checkpoint on the independent held-out splits (audit §19/§20).

* reads ``test``/``challenge`` from the dataset version — never ``train``;
* no sample cap by default (``--max-samples`` only ever *reduces* the set and is
  recorded in the report);
* measures language, math, logic, algorithms, code generation/repair,
  instruction following, tool syntax/selection/arguments/execution, grounding
  and generalization to held-out templates;
* can evaluate the quantized variants of the same checkpoint
  (``--variants fp32,fp16,int8,int4``) for the functional quantization report.

Examples::

    python scripts/evaluate.py --experiment EXP-002-CORRECTED-NANO --split test
    python scripts/evaluate.py --experiment EXP-004-TOOL-SFT --split challenge \
        --variants fp32,fp16,int8,int4 --max-samples 40
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.dataset_api import load_manifest, load_split_records, load_tokenizer  # noqa: E402
from src.evaluation.evaluator import (  # noqa: E402
    EvaluationResult, build_suite, evaluate_model, write_evaluation_report)
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import TinyMeConfig, count_parameters  # noqa: E402
from src.quantization.quantize import (dequantize_int4, dequantize_int8, flatten_params,  # noqa: E402
                                       quantize_int4, quantize_int8, unflatten_params)
from src.sandbox.runner import detected_summary  # noqa: E402
from src.training.checkpoint import verify_checkpoint  # noqa: E402
from src.utils.io_utils import REPO_ROOT, human_bytes, read_json, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("tinyme.eval.cli")


def _model_config_for(params: dict, tokenizer, architecture: str) -> TinyMeConfig:
    cfg = TinyMeConfig.from_name(architecture)
    cfg.vocab_size = int(tokenizer.vocab_size)
    actual = count_parameters(params)
    log.info("model config %s: %d params (checkpoint has %d)", architecture,
             cfg.parameter_count(), actual)
    return cfg


def _read_json_or(path: Path, default: dict) -> dict:
    """``read_json`` with a default (the shared helper takes no default)."""
    try:
        return read_json(path)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _load_engine(experiment: str, checkpoint: str, tokenizer, backend: str,
                 params_override: dict | None = None, arch: str | None = None) -> InferenceEngine:
    ckpt_dir = REPO_ROOT / "checkpoints" / experiment
    report = verify_checkpoint(ckpt_dir, checkpoint)
    if not report.get("ok"):
        raise SystemExit(f"checkpoint {experiment}/{checkpoint} failed verification: {report['errors']}")
    run_cfg = _read_json_or(REPO_ROOT / "experiments" / experiment / "run_config.json", {})
    architecture = arch or (run_cfg.get("model_config") or {}).get("architecture") or "nano"
    params_file = ckpt_dir / f"{checkpoint}.safetensors"
    engine = InferenceEngine(params_file, REPO_ROOT / "datasets" / "versions" /
                             (run_cfg.get("dataset_version") or "dataset_v2") / "tokenizer.json",
                             backend=backend)
    engine.config = _model_config_for(engine.params, tokenizer, architecture)
    if params_override is not None:
        engine.set_params(params_override)
    return engine


def _f32_list(node):
    """Yield a float32-cast copy of a nested block dict (blocks are lists of dicts)."""
    if isinstance(node, dict):
        return [{k: (v.astype(np.float32) if hasattr(v, "astype") else v) for k, v in node.items()}]
    if isinstance(node, list):
        return [x for item in node for x in _f32_list(item)]
    return [node]


def quantized_variants(params: dict, names: list[str]) -> dict[str, dict]:
    """Apply the flat quantization API to the nested tree and rebuild it.

    ``quantize_int8``/``quantize_int4`` operate on flat ``blocks/0/attn/q`` keys,
    while the engine holds the nested tree - flatten -> quantize -> unflatten
    (the direct call used to raise ``AttributeError: 'list' object has no
    attribute 'ndim'``).
    """
    out: dict[str, dict] = {}
    for name in names:
        if name == "fp32":
            out[name] = params
            continue
        flat = flatten_params(params)
        if name == "fp16":
            out[name] = unflatten_params({k: v.astype(np.float16) for k, v in flat.items()})
        elif name == "int8":
            q, s = quantize_int8(flat)
            out[name] = unflatten_params(dequantize_int8(q, s))
        elif name == "int4":
            q, s = quantize_int4(flat)
            out[name] = unflatten_params(dequantize_int4(q, s))
        else:
            raise SystemExit(f"unknown variant {name}")
        # keep float32 everywhere the NumPy forward touches (nested blocks included)
        parts = out[name].get("blocks")
        out[name] = {k: (v.astype(np.float32) if hasattr(v, "astype") else v)
                     for k, v in out[name].items()}
        if isinstance(parts, list):
            out[name]["blocks"] = [inner for blk in parts for inner in _f32_list(blk)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--split", default="test", choices=["test", "challenge", "validation"])
    ap.add_argument("--backend", default="jax", choices=["jax", "numpy"])
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--max-samples", type=int, default=0, help="0 = evaluate every held-out record")
    ap.add_argument("--variants", default="",
                    help="comma list of fp32,fp16,int8,int4 for the quantization comparison")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="")
    ap.add_argument("--release", action="store_true",
                    help="copy the primary-variant report into release/evaluation_report.md")
    args = ap.parse_args()

    run_cfg = _read_json_or(REPO_ROOT / "experiments" / args.experiment / "run_config.json", {})
    version = args.dataset or run_cfg.get("dataset_version") or "dataset_v2"
    manifest = load_manifest(version)
    tokenizer = load_tokenizer(version)
    records = build_suite(version, args.split, max_samples=args.max_samples or None)
    log.info("evaluating %d held-out records from %s/%s (dataset fingerprint %s)", len(records),
             version, args.split, manifest.get("dataset_fingerprint"))

    engine = _load_engine(args.experiment, args.checkpoint, tokenizer, args.backend)
    variant_names = [v.strip() for v in args.variants.split(",") if v.strip()] or ["fp32"]
    params_by_variant = quantized_variants(engine.params, variant_names)

    results: list[EvaluationResult] = []
    summaries: dict[str, dict] = {}
    for name in variant_names:
        engine.set_params(params_by_variant[name])
        generate = lambda prompt, max_new_tokens=args.max_new_tokens: engine.generate(
            prompt, max_new_tokens=max_new_tokens, temperature=0.0, use_cache=True)
        result = evaluate_model(generate, records, model_name=f"{args.experiment}:{name}",
                                dataset_version=version, split=args.split,
                                max_new_tokens=args.max_new_tokens,
                                token_count_fn=lambda text: max(1, len(engine.encode(text))),
                                perplexity_fn=lambda text: engine.perplexity([text]))
        # merge, never replace: evaluate_model() already recorded the measured
        # generation latency/throughput for this variant
        result.environment.update({"backend": args.backend, "isolation": detected_summary(),
                              "checkpoint": f"{args.experiment}/{args.checkpoint}",
                              "checkpoint_bytes": int((REPO_ROOT / "checkpoints" / args.experiment /
                                                       f"{args.checkpoint}.safetensors").stat().st_size),
                              "max_new_tokens": args.max_new_tokens,
                              "max_samples": args.max_samples or None,
                              "held_out_records": len(records),
                              "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        results.append(result)
        summaries[name] = result.to_dict()
        log.info("[%s] mean_accuracy=%s domains=%d (%.1fs)", name,
                 result.overall.get("mean_accuracy"), len(result.domains), result.duration_s)

    tag = args.tag or args.split
    out_dir = Path(args.out) if args.out else REPO_ROOT / "experiments" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json({"experiment": args.experiment, "split": args.split, "dataset": version,
                "dataset_fingerprint": manifest.get("dataset_fingerprint"),
                "variants": summaries}, out_dir / f"evaluation_{tag}.json")
    report_path = out_dir / f"evaluation_{tag}.md"
    primary = [r for r in results if r.model.endswith(":fp32")] or results
    write_evaluation_report(primary, report_path,
                            title=f"Evaluation report — {args.experiment} ({args.split} split)")
    if args.release and args.split == "test":
        write_evaluation_report(primary, REPO_ROOT / "release" / "evaluation_report.md",
                               title=f"Evaluation report — {args.experiment} (held-out test split)")
    print(json.dumps({name: s["overall"] for name, s in summaries.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
