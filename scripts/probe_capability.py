#!/usr/bin/env python3
"""Diagnostic probe: does a checkpoint actually *use* its context and tools?

This is the probe that diagnosed the capability failure.  It measures, with real
model execution and real tools, and it never uses a threshold as evidence:

1. **call -> execute -> answer** (``tool_compute_copy``) — for N held-in
   ``tool/compute`` records the model is prompted with everything up to
   ``<|assistant|>``; its emitted call is parsed, **executed by the shipped
   tool**, its result is rendered back into the prompt exactly as the runtime
   does, and the model writes the second turn.  Recorded: whether the call is
   valid, whether the operand was copied verbatim, whether a ``<|final|>`` turn
   followed, and whether the tool's value appears in it.

   The earlier version of this probe looked for the tool value in the *first*
   turn's text, which no model can produce (the result is not in the prompt yet);
   it reported 0/10 for that structural reason alone.  It also read the gold
   expression from segment index 2, which is not guaranteed to be the tool call.

2. **fresh operands** (``fresh_operand_generalisation``) — arithmetic prompts
   generated at probe time with a seed that never appears in the corpus, so a
   memorised template cannot help.  This is the measurement that exposed the
   "shape rewrite" failure (``11 + 22`` answered as ``111 + 22``).

3. **held-in reproduction** (``held_in_reproduction``) — first-30-character match
   on held-in records: a memorisation signal, not a capability signal.

4. **no-tool decision** (``no_tool_decision``) — arithmetic prompts whose gold
   answer lives in the record: does the model answer directly, and is it right.

Usage::

    python scripts/probe_capability.py --experiment EXP-010-TOOL-SFT-V7 --checkpoint best
    python scripts/probe_capability.py --release release/model_fp32.safetensors \\
        --release-tokenizer release/tokenizer.json --release-config release/config.json
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

from src.agent.protocol import render_tool_result  # noqa: E402
from src.evaluation.tool_cases import _norm  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import count_parameters  # noqa: E402
from src.agent.tool_registry import ToolRegistry  # noqa: E402
from src.tools.context import ToolContext  # noqa: E402

CALL_RE = re.compile(r"<\|tool_call\|>\s*(?P<body>.*?)\s*<\|end_tool_call\|>", re.DOTALL)
FINAL_RE = re.compile(r"<\|final\|>\s*(?P<body>.*?)(?:<\|eos\|>|$)", re.DOTALL)

_SYSTEM = ("You are TinyMe, a controller for external tools. Use a tool only when needed, "
           "then answer from the evidence.")


def _load(experiment: str | None, checkpoint: str, weights: str | None,
          tokenizer: str, config: str, dataset: str) -> InferenceEngine:
    if weights:
        return InferenceEngine(Path(weights), Path(tokenizer or ROOT / "release" / "tokenizer.json"),
                               Path(config or ROOT / "release" / "config.json"),
                               backend="numpy", strict_config=False)
    ckpt = ROOT / "checkpoints" / experiment / f"{checkpoint}.safetensors"
    run_cfg = ROOT / "experiments" / experiment / "run_config.json"
    dataset_version = dataset
    if not dataset_version and run_cfg.exists():
        dataset_version = json.loads(run_cfg.read_text())["dataset_version"]
    cfg_path = ckpt.with_suffix(".json")
    tokenizer_path = ROOT / "datasets" / "versions" / dataset_version / "tokenizer.json"
    if cfg_path.exists():
        return InferenceEngine(ckpt, tokenizer_path, cfg_path, backend="numpy", strict_config=False)
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


def _gold_expression(rec: dict) -> str | None:
    """The gold call is the segment whose *role* is tool_call (never a fixed index)."""
    for seg in rec.get("segments", []):
        if seg.get("role") == "tool_call":
            try:
                return json.loads(seg["text"]).get("arguments", {}).get("expression")
            except json.JSONDecodeError:
                return None
    return None


def _corpus_text(version: str) -> str:
    """All split text of one dataset version, used to reject accidental overlaps."""
    parts = []
    for split in ("train", "validation", "test", "challenge"):
        path = ROOT / "datasets" / "versions" / version / f"{split}.jsonl"
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


def _fresh_expressions(n: int, seed: int = 424242, corpus: str | None = None) -> list[str]:
    """Arithmetic prompts that appear in no corpus build (checked, not assumed)."""
    rng = random.Random(seed)
    out: list[str] = []
    rejected = 0
    guard = 0
    while len(out) < n:
        digits = rng.randint(1, 4)
        op = rng.choice(["+", "-", "*", "//"])
        a = rng.randint(1, 10 ** digits - 1)
        b = rng.randint(1, 10 ** rng.randint(1, 3) - 1)
        if op == "//":
            a = max(b, a - (a % b))
            if a == 0:
                continue
        expr = f"{a} {op} {b}"
        try:
            eval(expr)                                       # noqa: S307 - digits only
        except ZeroDivisionError:
            continue
        guard += 1
        if guard > 20000:
            break
        if expr in out:
            continue
        if corpus is not None and expr in corpus:
            rejected += 1
            continue
        out.append(expr)
    if corpus is not None:
        _fresh_expressions.last_rejected = rejected       # type: ignore[attr-defined]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--checkpoint", default="best", choices=["best", "latest", "final"])
    ap.add_argument("--release", default=None, help="evaluate a release safetensors file instead")
    ap.add_argument("--release-tokenizer", default=str(ROOT / "release" / "tokenizer.json"))
    ap.add_argument("--release-config", default=str(ROOT / "release" / "config.json"))
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default=str(ROOT / "docs" / "audit_evidence" / "capability_probe.json"))
    args = ap.parse_args()

    if not args.experiment and not args.release:
        raise SystemExit("pass --experiment or --release")

    version = args.dataset
    if version is None and args.experiment:
        run_cfg = ROOT / "experiments" / args.experiment / "run_config.json"
        version = json.loads(run_cfg.read_text())["dataset_version"] if run_cfg.exists() else "dataset_v7"
    if args.release:
        man = ROOT / "release" / "manifest.json"
        if man.exists():
            version = json.loads(man.read_text()).get("dataset_version", "dataset_v7")
    version = version or "dataset_v7"

    engine = _load(args.experiment, args.checkpoint, args.release, args.release_tokenizer,
                   args.release_config, version)
    params = count_parameters(engine.params)
    print(f"loaded {params} params (config: {engine.config_source}); dataset={version}")
    report: dict = {"params": params, "dataset": version,
                    "checkpoint": args.release or f"{args.experiment}/{args.checkpoint}",
                    "probes": {}}

    ctx = ToolContext.create(with_workspace=False)
    registry = ToolRegistry.default(context=ctx)

    def run_call(expr_expected: str, prompt: str) -> dict:
        """One full two-turn episode for a compute prompt."""
        out = _generate(engine, prompt)
        call = _parse_call(out)
        valid = bool(call and call.get("name") == "compute"
                     and isinstance(call.get("arguments"), dict)
                     and isinstance(call["arguments"].get("expression"), str))
        emitted = call["arguments"]["expression"] if valid else None
        value, tool_out, second = None, None, ""
        if valid:
            result = registry.call("compute", {"expression": emitted})
            tool_out = result.to_dict()
            if result.ok:
                value = result.payload.get("value")
        if value is not None:
            second = _generate(engine, prompt + render_tool_result(tool_out)
                               + "\n<|assistant|>\n", max_new=64)
        return {"call_valid": valid, "emitted_expression": emitted,
                "copied": bool(emitted and emitted == expr_expected),
                "tool_value": value, "tool_ok": bool(tool_out and tool_out.get("ok")),
                "tool_error": (tool_out or {}).get("error"),
                "final_turn": bool(FINAL_RE.search(second)),
                "value_in_final": bool(value is not None and str(value) in second),
                "second_turn_text": second[:200]}

    # ---------------------------------------------- 1. held-in compute trajectory
    records = _records(version, args.n, "tool/compute")
    rows, called, copied, final_turns, grounded = [], 0, 0, 0, 0
    for rec in records:
        prompt = _prompt_of(rec["text"])
        gold = _gold_expression(rec)
        r = run_call(gold or "", prompt)
        called += r["call_valid"]
        copied += r["copied"]
        final_turns += r["final_turn"]
        grounded += r["value_in_final"]
        rows.append({"record": rec.get("source_id"), "gold_expression": gold, **r})
    report["probes"]["tool_compute_trajectory"] = {
        "records": len(records), "valid_calls": called, "copied_expression": copied,
        "final_turn_emitted": final_turns, "tool_result_value_used": grounded, "rows": rows}
    print(f"compute trajectory: {called}/{len(records)} valid calls, {copied}/{len(records)} copied "
          f"the operand, {final_turns}/{len(records)} produced a final turn, "
          f"{grounded}/{len(records)} put the tool value in it")

    # --------------------------------------------- 2. fresh operands (never seen)
    fresh_rows, f_calls, f_copied, f_grounded = [], 0, 0, 0
    corpus_text = _corpus_text(version)
    fresh_prompts = _fresh_expressions(args.n, corpus=corpus_text)
    print(f"fresh operands: generated {len(fresh_prompts)} prompts that occur nowhere in "
          f"{version} ({getattr(_fresh_expressions, 'last_rejected', 0)} candidates rejected as "
          f"already present)")
    for expr in fresh_prompts:
        prompt = (f"<|system|>\n{_SYSTEM}\n<|user|>\nEvaluate the expression {expr} exactly "
                  f"with a tool.\n<|assistant|>\n")
        r = run_call(expr, prompt)
        f_calls += r["call_valid"]
        f_copied += r["copied"]
        f_grounded += r["value_in_final"]
        fresh_rows.append({"expression": expr, **r})
    n_fresh = len(fresh_rows)
    report["probes"]["fresh_operand_generalisation"] = {
        "prompts": n_fresh, "valid_calls": f_calls, "copied_expression": f_copied,
        "tool_result_value_used": f_grounded,
        "candidates_rejected_as_present_in_corpus": getattr(_fresh_expressions, "last_rejected", 0),
        "corpus_checked": version, "rows": fresh_rows}
    print(f"fresh operands: {f_calls}/{n_fresh} valid calls, {f_copied}/{n_fresh} copied exactly, "
          f"{f_grounded}/{n_fresh} grounded in the tool value")

    # --------------------------------------------------- 3. template reproduction
    held = _records(version, 30, "no_tool/arith/add")
    match = 0
    for rec in held:
        out = _generate(engine, _prompt_of(rec["text"]))
        gold = rec["text"].split("<|assistant|>", 1)[1].lstrip("\n")[:30]
        match += _norm(out)[:30] == _norm(gold)[:30]
    report["probes"]["held_in_reproduction"] = {"records": len(held), "first30_match": match}
    print(f"held-in arithmetic records reproduced (first 30 chars): {match}/{len(held)}")

    # ------------------------------------------------------ 4. no-tool decision
    direct = direct_correct = 0
    for rec in held:
        out = _generate(engine, _prompt_of(rec["text"]))
        called_tool = bool(CALL_RE.search(out))
        direct += not called_tool
        answer = FINAL_RE.search(out)
        gold = str(rec.get("answer") or "")
        direct_correct += bool(not called_tool and gold
                               and gold in (answer.group("body") if answer else ""))
    report["probes"]["no_tool_decision"] = {
        "records": len(held), "answered_without_tool": direct, "direct_answer_correct": direct_correct}
    print(f"no-tool prompts answered without a tool: {direct}/{len(held)} "
          f"({direct_correct} of them correct)")

    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
