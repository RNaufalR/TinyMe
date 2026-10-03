#!/usr/bin/env python3
"""Diagnostic probe: does a checkpoint *copy* from its context?

This is the probe that diagnosed the capability failure (see
``docs/audit_evidence/tokenizer_context_copy_probe.*``).  It measures, with real
model execution and no thresholds:

1. **argument copy rate** — for N held-in ``tool/compute`` training records, the
   model is prompted with everything up to ``<|assistant|>`` and the emitted
   tool call is parsed; the expression it asked the tool to compute either
   equals the one in the prompt or it does not;
2. **grounded compute accuracy** — the emitted expression is actually evaluated
   and compared with the record's gold value, so a "copied but wrong" answer is
   not counted as success;
3. **template reproduction** — first-30-character match rate on held-in records,
   the memorisation signal;
4. **no-tool direct answering** — for arithmetic prompts, whether the model
   answers without calling a tool.

Usage::

    python scripts/probe_capability.py --experiment EXP-008-TOOL-SFT-V5 --checkpoint best
    python scripts/probe_capability.py --release release/model_fp32.safetensors \\
        --tokenizer release/tokenizer.json --config release/config.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.tool_cases import _norm  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.tools.compute import compute as compute_tool  # noqa: E402
from src.tools.context import ToolContext  # noqa: E402

CALL_RE = re.compile(r"<\|tool_call\|>\s*(?P<body>.*?)\s*<\|end_tool_call\|>", re.DOTALL)


def _load(experiment: str | None, checkpoint: str, weights: str | None,
          tokenizer: str, config: str, dataset: str) -> InferenceEngine:
    if weights:
        return InferenceEngine(Path(weights), Path(tokenizer), Path(config),
                               backend="numpy", strict_config=False)
    ckpt = ROOT / "checkpoints" / experiment / f"{checkpoint}.safetensors"
    run_cfg = ROOT / "experiments" / experiment / "run_config.json"
    dataset_version = dataset
    if not dataset_version and run_cfg.exists():
        dataset_version = json.loads(run_cfg.read_text())["dataset_version"]
    tokenizer_path = ROOT / "datasets" / "versions" / dataset_version / "tokenizer.json"
    return InferenceEngine(ckpt, tokenizer_path, backend="numpy")


def _generate(engine: InferenceEngine, prompt: str, max_new: int = 96) -> str:
    return engine.generate(prompt, max_new_tokens=max_new, temperature=0.0, top_p=1.0,
                           repetition_penalty=1.0,
                           stop=["<|end_tool_call|>", "<|end_tool_result|>", "<|endcode|>"])


def _records(version: str, limit: int, template: str) -> list[dict]:
    rows = [json.loads(l) for l in (ROOT / "datasets" / "versions" / version / "train.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    picked = [r for r in rows if r.get("template_id") == template]
    random.Random(7).shuffle(picked)
    return picked[:limit]


def _prompt_of(text: str) -> str:
    marker = "<|assistant|>"
    return text[:text.index(marker) + len(marker)] + "\n"


def _parse_call(text: str) -> dict | None:
    m = CALL_RE.search(text)
    if not m:
        return None
    try:
        return json.loads(m.group("body"))
    except json.JSONDecodeError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest", "final"])
    ap.add_argument("--release", default=None, help="evaluate a release safetensors file instead")
    ap.add_argument("--release-tokenizer", default=str(ROOT / "release" / "tokenizer.json"))
    ap.add_argument("--release-config", default=str(ROOT / "release" / "config.json"))
    ap.add_argument("--dataset", default="dataset_v5")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "docs" / "audit_evidence" / "capability_probe.json"))
    args = ap.parse_args()

    engine = _load(args.experiment, args.checkpoint, args.release, args.release_tokenizer,
                   args.release_config, args.dataset)
    print(f"loaded {count_parameters(engine.params)} params (config: {engine.config_source})")

    report: dict = {"params": count_parameters(engine.params), "probes": {}}

    # ---------------------------------------------------------- 1. compute copy
    ctx = ToolContext.create(with_workspace=False)
    records = _records(args.dataset, args.n, "tool/compute")
    copied = correct = called = 0
    rows = []
    for rec in records:
        prompt = _prompt_of(rec["text"])
        gold_expr = json.loads(rec["segments"][2]["text"])["arguments"]["expression"] \
            if rec.get("segments") and rec["segments"][2]["role"] == "tool_call" else None
        out = _generate(engine, prompt)
        call = _parse_call(out)
        args_ok = bool(call and call.get("name") == "compute"
                       and isinstance(call.get("arguments"), dict)
                       and isinstance(call["arguments"].get("expression"), str))
        expr = call["arguments"]["expression"] if args_ok else ""
        called += args_ok
        copied += bool(expr and gold_expr and expr == gold_expr)
        value = None
        if args_ok:
            try:
                value = compute_tool(ctx, expr)["value"]
            except Exception:
                value = None
        correct += bool(value is not None and str(value) in out)
        rows.append({"record": rec["source_id"], "gold_expression": gold_expr,
                     "emitted_expression": expr or None, "call_valid": args_ok,
                     "copied": bool(expr and gold_expr and expr == gold_expr),
                     "value_in_output": correct and rows and bool(value is not None)})
    report["probes"]["tool_compute_copy"] = {
        "records": len(records), "valid_calls": called, "copied_expression": copied,
        "tool_result_value_used": correct, "rows": rows}
    print(f"compute: {called}/{len(records)} valid calls, {copied}/{len(records)} copied the "
          f"operand, {correct}/{len(records)} used the tool value")

    # ------------------------------------------------- 2. template reproduction
    held = _records(args.dataset, 30, "no_tool/arith/add")
    match = 0
    for rec in held:
        prompt = _prompt_of(rec["text"])
        out = _generate(engine, prompt)
        gold = rec["text"].split("<|assistant|>", 1)[1].lstrip("\n")[:30]
        match += _norm(out)[:30] == _norm(gold)[:30]
    report["probes"]["held_in_reproduction"] = {"records": len(held), "first30_match": match}
    print(f"held-in arithmetic records reproduced (first 30 chars): {match}/{len(held)}")

    # ------------------------------------------------------- 3. tool decision
    direct = 0
    direct_correct = 0
    for rec in held:
        out = _generate(engine, _prompt_of(rec["text"]))
        called_tool = bool(CALL_RE.search(out))
        direct += not called_tool
        answer = re.search(r"<\|final\|>\s*(?P<body>.{0,60})", out)
        gold = str(rec.get("answer") or "")
        direct_correct += bool(not called_tool and gold and gold in (answer.group("body") if answer else ""))
    report["probes"]["no_tool_decision"] = {
        "records": len(held), "answered_without_tool": direct, "direct_answer_correct": direct_correct}
    print(f"no-tool prompts answered without a tool: {direct}/{len(held)} "
          f"({direct_correct} of them correct)")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
