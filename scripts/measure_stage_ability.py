#!/usr/bin/env python3
"""Stage-level ability on *held-out* records (audit §48/§50).

The capability gate lives in ``scripts/evaluate_tools.py`` (independent A–H
cases).  This script answers a narrower question that the curriculum decision
needs: for a given checkpoint, how often does the model reproduce the supervised
target of a held-out record of a *known family* — copying a span, doing the small
arithmetic, emitting the tool call for a compute question?

It never uses a training record: the split is loaded through the dataset API and
the prompt is rendered by the evaluator's own ``render_prompt``, so the number is
comparable across checkpoints and across the curriculum stages.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.evaluation.evaluator import render_prompt  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402

DEFAULT_TEMPLATES = ("copy/span/kind3", "no_tool/arith/add", "no_tool/arith/mul",
                     "no_tool/text_edit/upper", "tool/compute", "tool/search_single")


def expected_turn(record: dict) -> str:
    return "\n".join(seg["text"] for seg in record.get("segments", [])
                     if seg.get("target", True) and seg["role"] in
                     ("assistant", "final", "tool_call", "code", "thought"))


def normalise(text: str) -> str:
    text = re.sub(r"<\|[a-z_]+\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--split", default="test")
    ap.add_argument("--templates", default=",".join(DEFAULT_TEMPLATES))
    ap.add_argument("--per-template", type=int, default=12)
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ckpt = ROOT / "checkpoints" / args.experiment / f"{args.checkpoint}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    tokenizer_path = ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json"
    engine = InferenceEngine(ckpt, tokenizer_path, backend="numpy")

    records = [json.loads(line) for line in
               (ROOT / "datasets" / "versions" / args.dataset / f"{args.split}.jsonl")
               .read_text(encoding="utf-8").splitlines() if line.strip()]
    by_template: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_template[str(record.get("template_id") or "")].append(record)

    wanted = [t.strip() for t in args.templates.split(",") if t.strip()]
    out: dict[str, dict] = {}
    for template in wanted:
        pool = by_template.get(template, [])[:args.per_template]
        if not pool:
            out[template] = {"samples": 0, "exact_match": None, "note": "no held-out record"}
            print(f"{template:26s} no held-out record in split {args.split}")
            continue
        exact, contains, samples = 0, 0, []
        for record in pool:
            generated = engine.generate(render_prompt(record), max_new_tokens=args.max_new_tokens,
                                        temperature=0.0, repetition_penalty=1.0)
            gold = normalise(expected_turn(record))
            got = normalise(generated)
            hit_exact = bool(gold) and gold == got
            hit_contains = bool(gold) and (gold == got or gold in got)
            exact += hit_exact
            contains += hit_contains
            samples.append({"record_id": record.get("record_id"), "gold": gold[:120],
                            "generated": got[:120], "contains_target": hit_contains,
                            "exact": hit_exact})
        out[template] = {"samples": len(pool), "exact_match": round(exact / len(pool), 4),
                         "contains_target": round(contains / len(pool), 4),
                         "examples": samples[:3]}
        print(f"{template:26s} exact={out[template]['exact_match']:.2f} "
              f"contains={out[template]['contains_target']:.2f} (n={len(pool)})")

    scored = [v["exact_match"] for v in out.values() if v.get("samples")]
    payload = {"experiment": args.experiment, "checkpoint": args.checkpoint, "dataset": args.dataset,
               "split": args.split, "backend": "numpy", "temperature": 0.0,
               "repetition_penalty": 1.0, "max_new_tokens": args.max_new_tokens,
               "mean_exact_match": round(sum(scored) / len(scored), 4) if scored else None,
               "templates": out}
    out_path = Path(args.out) if args.out else (
        ROOT / "docs" / "audit_evidence" / f"stage_ability_{args.experiment}_{args.checkpoint}.json")
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"mean exact match {payload['mean_exact_match']} -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
