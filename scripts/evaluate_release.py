#!/usr/bin/env python3
"""Run the independent A-H tool suite against the shipped *release* artifacts.

``scripts/evaluate_tools.py`` evaluates a training checkpoint.  This script
evaluates what a user actually downloads: ``release/model_<variant>.safetensors``
+ ``release/tokenizer.json`` + ``release/config.json``.  It exists because the
audit must not accept "the checkpoint would pass" as evidence, and because the
release is the artefact the capability claim is about.

Usage::

    python scripts/evaluate_release.py --variant fp32 --output experiments/release_eval_tools.json
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

from scripts.evaluate_tools import SYSTEM_PROMPT, run_cases  # noqa: E402
from src.evaluation.tool_cases import summarise  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import count_parameters  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger("tinyme.eval.release")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release-dir", default=str(ROOT / "release"))
    ap.add_argument("--variant", default="fp32", choices=["fp32", "fp16", "int8", "int4"])
    ap.add_argument("--dataset-for-tokenizer", default=None,
                    help="fall back to a dataset version's tokenizer when the release "
                         "tokenizer is a different tokenizer generation")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--output", default=str(ROOT / "experiments" / "release_eval_tools.json"))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    rel = Path(args.release_dir)
    weights = rel / f"model_{args.variant}.safetensors"
    tokenizer = rel / "tokenizer.json"
    if args.dataset_for_tokenizer:
        tokenizer = ROOT / "datasets" / "versions" / args.dataset_for_tokenizer / "tokenizer.json"
    engine = InferenceEngine(weights, tokenizer, rel / "config.json", backend="numpy",
                             strict_config=True)
    log.info("loaded %s: %d params (config from %s)", weights.name,
             count_parameters(engine.params), engine.config_source)

    started = time.time()
    rows, transcripts = run_cases(engine, max_new_tokens=args.max_new_tokens,
                                  temperature=args.temperature, verbose=args.verbose)
    summary = summarise(rows)
    payload = {"release_dir": str(rel), "variant": args.variant, "weights": weights.name,
               "tokenizer": str(tokenizer), "temperature": args.temperature,
               "max_new_tokens": args.max_new_tokens, "suite": "independent tool cases A-H",
               "summary": summary, "cases": rows, "transcripts": transcripts,
               "duration_s": round(time.time() - started, 2),
               "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("summary: %s", json.dumps({k: v for k, v in summary.items() if k != "by_family"}))
    log.info("wrote %s", out)

    for row in rows:
        log.info("%s %-4s stop=%-12s tools=%s final=%r", row["case_id"], row["family"],
                 row.get("stop_reason"), row.get("tools_used"), (row.get("final") or "")[:90])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
