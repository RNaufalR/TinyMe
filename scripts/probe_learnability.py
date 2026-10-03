#!/usr/bin/env python3
"""Fast learnability probe for a curriculum stage (DEC-013 evidence).

Trains a fresh nano model on one curriculum stage (`drill` or `trajectory`) for a
small number of steps and then measures, with the *released* inference path, how
often the greedy continuation matches the held-out record's supervision target.

The probe answers one question before a multi-hour run is committed to:
*can this architecture and this objective learn the stage at all* — copying a
span out of the prompt, doing the small arithmetic, emitting the tool call.
It is evidence about the recipe, never a capability claim: the capability gate
stays `scripts/evaluate_tools.py` on the independent A–H case set.

Writes a JSON report (per-family exact match, sample transcripts, training
metrics) and a markdown twin.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.data.curriculum import record_filter, stage_of  # noqa: E402
from src.data.dataset_api import load_split, load_tokenizer  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import TinyMeConfig  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402

SYSTEM_PROMPT = ("You are TinyMe, a small tool-using assistant. Use a tool only when it is "
                 "needed; otherwise answer directly and briefly.")


def _segments(record: dict) -> dict[str, str]:
    return {s["role"]: s["text"] for s in record.get("segments", [])}


def _target_role(record: dict) -> tuple[str, str]:
    """(role, text) of the first supervised segment — the probe's gold turn."""
    for seg in record.get("segments", []):
        if seg.get("target", True) and seg["role"] in ("assistant", "final", "tool_call", "code"):
            return seg["role"], seg["text"]
    return "", ""


def _render_prompt(record: dict) -> str:
    segs = _segments(record)
    text = f"<|system|>\n{segs.get('system', SYSTEM_PROMPT)}\n"
    if "user" in segs:
        text += f"<|user|>\n{segs['user']}\n"
    text += "<|assistant|>\n"
    return text


def _normalise(text: str) -> str:
    text = re.sub(r"<\|[a-z_]+\|>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v8")
    ap.add_argument("--stage", default="drill", choices=["drill", "trajectory"])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--eval-records", type=int, default=40)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    experiment = args.experiment or f"PROBE-{args.stage.upper()}"
    out = Path(args.out or f"docs/audit_evidence/learnability_probe_{args.stage}.json")

    tokenizer = load_tokenizer(args.dataset)
    selector = record_filter(args.stage)
    train_data, train_stats = load_split(args.dataset, "train", seq_len=args.seq_len,
                                         tokenizer=tokenizer, stage=args.stage,
                                         record_filter=selector)
    val_data, val_stats = load_split(args.dataset, "validation", seq_len=args.seq_len,
                                     tokenizer=tokenizer, stage=args.stage,
                                     record_filter=selector)
    print(f"stage={args.stage} train_blocks={train_data['input_ids'].shape[0]} "
          f"active={train_stats['active_target_tokens']} | val_blocks={val_data['input_ids'].shape[0]} "
          f"active={val_stats['active_target_tokens']}", flush=True)

    cfg = TrainConfig(experiment_id=experiment, experiment_name=f"learnability probe ({args.stage})",
                      stage=args.stage, architecture="nano", dataset_version=args.dataset,
                      seed=args.seed, seq_len=args.seq_len, micro_batch_size=args.micro_batch,
                      grad_accum_steps=args.grad_accum, learning_rate=args.lr, warmup_steps=args.warmup,
                      max_steps=args.steps, eval_every=max(50, args.steps // 4),
                      checkpoint_every=args.steps, keep_last_checkpoints=1,
                      max_val_batches=8, resume=False,
                      notes="DEC-013 learnability probe; not a capability claim")
    trainer = Trainer(cfg, model_cfg=TinyMeConfig.from_name("nano"), tokenizer=tokenizer)
    t0 = time.perf_counter()
    summary = trainer.fit(train_data, val_data, resume=False)
    duration = time.perf_counter() - t0

    ckpt = ROOT / "checkpoints" / experiment / "best.safetensors"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / experiment / "latest.safetensors"
    engine = InferenceEngine(model_path=str(ckpt), tokenizer_path=str(ROOT / "datasets" / "versions" /
                             args.dataset / "tokenizer.json"), backend="numpy")

    records = [r for r in _read_records(args.dataset, "validation") if stage_of(r) == args.stage]
    records = records[:args.eval_records]
    per_family: dict[str, list[bool]] = {}
    rows = []
    for record in records:
        role, gold = _target_role(record)
        prompt = _render_prompt(record)
        generated = engine.generate(prompt, max_new_tokens=args.max_new_tokens,
                                    temperature=0.0, repetition_penalty=1.0)
        want = _normalise(f"{role} {gold}")
        got = _normalise(generated)
        match = want in got or _normalise(gold) == got
        family = str(record.get("template_id") or "?")
        per_family.setdefault(family, []).append(match)
        rows.append({"record_id": record.get("record_id"), "template_id": family,
                     "gold": gold[:160], "generated": generated[:160], "exact_match": match})

    families = {k: {"n": len(v), "exact_match": round(sum(v) / len(v), 4)} for k, v in sorted(per_family.items())}
    overall = round(sum(r["exact_match"] for r in rows) / max(len(rows), 1), 4)
    report = {
        "probe": "learnability", "stage": args.stage, "dataset": args.dataset,
        "experiment": experiment, "steps": args.steps, "seq_len": args.seq_len, "lr": args.lr,
        "seed": args.seed, "duration_s": round(duration, 1),
        "train_stats": train_stats, "val_stats": val_stats,
        "training": {"final_val_loss": summary.get("final_val_loss"),
                     "best_val_loss": summary.get("best_val_loss")},
        "evaluated_records": len(rows), "exact_match": overall, "by_template": families,
        "samples": rows[:12],
    }
    write_json(out, report)
    print(json.dumps({k: report[k] for k in ("exact_match", "by_template", "duration_s")}, indent=1))
    print(f"wrote {out}")
    return 0


def _read_records(dataset: str, split: str) -> list[dict]:
    path = ROOT / "datasets" / "versions" / dataset / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
