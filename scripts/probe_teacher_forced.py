#!/usr/bin/env python3
"""Teacher-forced vs greedy diagnostic on the *training* token stream (§48/§50).

Two numbers per family separate the possible causes of a capability failure:

* **teacher-forced accuracy** — argmax(logits[i]) == input_ids[i+1] on the
  positions the loss mask supervises.  High accuracy with wrong generation
  points at the decoding/serving path; low accuracy points at the recipe.
* **greedy exact match** — the released ``InferenceEngine.generate`` from the
  prefix of the *same* example (byte-identical prompt tokens), so a formatting
  difference between training and serving shows up as a lower second number only
  when it really is a serving-path bug.

Prompt parity is reported per family: the training prefix (from
``build_sequence``) is compared with the evaluator's ``render_prompt``
tokenisation, so a render divergence is visible instead of silently depressing
the score.
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

import numpy as np  # noqa: E402

from src.data.dataset_api import load_tokenizer  # noqa: E402
from src.data.sequence import build_sequence  # noqa: E402
from src.evaluation.evaluator import render_prompt  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402

CONTROL_TOKENS = ("<|pad|>", "<|bos|>", "<|eos|>", "<|system|>", "<|user|>", "<|assistant|>",
                  "<|thought|>", "<|tool_call|>", "<|end_tool_call|>", "<|tool_result|>",
                  "<|end_tool_result|>", "<|final|>", "<|code|>", "<|endcode|>")


def _records(dataset: str, split: str, prefix: str, limit: int) -> list[dict]:
    path = ROOT / "datasets" / "versions" / dataset / f"{split}.jsonl"
    out = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if prefix and not str(rec.get("template_id") or "").startswith(prefix):
                continue
            out.append(rec)
            if len(out) >= limit:
                break
    return out


def _gold_text(record: dict) -> str:
    return "\n".join(seg["text"] for seg in record.get("segments", [])
                     if seg.get("target", True) and seg["role"] in
                     ("assistant", "final", "tool_call", "code", "thought"))


def _normalise(text: str) -> str:
    text = re.sub(r"<\|[a-z_]+\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--checkpoint", default="best")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--split", default="train")
    ap.add_argument("--template-prefix", default="")
    ap.add_argument("--limit", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tok = load_tokenizer(args.dataset)
    ckpt = ROOT / "checkpoints" / args.experiment / f"{args.checkpoint}.safetensors"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")
    engine = InferenceEngine(ckpt, ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json",
                             backend="numpy")

    pad_id = int(tok.tok.token_to_id("<|pad|>"))
    control_ids = {int(tok.tok.token_to_id(t)) for t in CONTROL_TOKENS}

    records = _records(args.dataset, args.split, args.template_prefix, args.limit)
    if not records:
        raise SystemExit("no records matched")

    families: dict[str, dict] = defaultdict(
        lambda: {"n": 0, "target_tokens": 0, "tf_hits": 0, "control_hits": 0, "control_n": 0,
                 "content_hits": 0, "content_n": 0, "exact": 0, "parity_ok": 0, "examples": []})
    for record in records:
        family = str(record.get("template_id") or "?")
        ex = build_sequence(record, tok, args.seq_len)
        if ex is None:
            continue
        ids = ex.input_ids
        n = int((ex.loss_mask | (ids != pad_id)).sum())
        ids = np.asarray(ids[:n], dtype=np.int32)
        mask = np.asarray(ex.loss_mask[:n], dtype=bool)
        # forward over the real training tokens: (1, T)
        logits = engine.forward(ids[None, :])[0]
        gold = ids[1:]

        stats = families[family]
        stats["n"] += 1
        first_target = int(np.argmax(mask)) if mask.any() else 0
        serve_ids = tok.encode_ids(render_prompt(record))
        train_prefix = [int(t) for t in ids[1:1 + len(serve_ids)]]
        stats["parity_ok"] += int(train_prefix == serve_ids)
        for i, is_target in enumerate(mask[:-1]):
            if not is_target:
                continue
            hit = int(int(logits[i].argmax()) == int(gold[i]))
            stats["target_tokens"] += 1
            stats["tf_hits"] += hit
            if int(gold[i]) in control_ids:
                stats["control_n"] += 1
                stats["control_hits"] += hit
            else:
                stats["content_n"] += 1
                stats["content_hits"] += hit

        prompt_text = tok.decode([int(t) for t in ids[:first_target]])
        generated = engine.generate(prompt_text, max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, repetition_penalty=1.0)
        gold_text = _gold_text(record)
        exact = int(_normalise(gold_text) == _normalise(generated))
        stats["exact"] += exact
        if len(stats["examples"]) < 3:
            stats["examples"].append({"record_id": record.get("record_id"),
                                      "gold": _normalise(gold_text)[:120],
                                      "generated": _normalise(generated)[:120], "exact": exact})

    report = {"experiment": args.experiment, "checkpoint": args.checkpoint, "dataset": args.dataset,
              "split": args.split, "template_prefix": args.template_prefix, "backend": "numpy",
              "families": {}}
    for family, s in sorted(families.items()):
        report["families"][family] = {
            "records": s["n"],
            "prompt_parity": f"{s['parity_ok']}/{s['n']}",
            "teacher_forced_accuracy": round(s["tf_hits"] / max(1, s["target_tokens"]), 4),
            "control_token_accuracy": round(s["control_hits"] / max(1, s["control_n"]), 4),
            "content_token_accuracy": round(s["content_hits"] / max(1, s["content_n"]), 4),
            "greedy_exact_match": round(s["exact"] / max(1, s["n"]), 4),
            "examples": s["examples"],
        }
        r = report["families"][family]
        print(f"{family:28s} n={r['records']:3d} parity={r['prompt_parity']:>5s} "
              f"tf={r['teacher_forced_accuracy']:.3f} control={r['control_token_accuracy']:.3f} "
              f"content={r['content_token_accuracy']:.3f} exact={r['greedy_exact_match']:.3f}")

    out_path = Path(args.out) if args.out else (
        ROOT / "docs" / "audit_evidence" /
        f"teacher_forced_{args.experiment}_{args.checkpoint}_{args.split}.json")
    out_path = out_path if out_path.is_absolute() else ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
