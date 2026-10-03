#!/usr/bin/env python3
"""Can the stack fit a *single* supervised task? (audit §48/§50 diagnostic)

``probe_learnability.py`` answers "does a curriculum stage move the metric"; this
probe answers the question underneath it: with every confound removed (one task
family, a handful of records, hundreds of steps, no mixture), can this model
reproduce the supervised continuation at all?

A fresh nano is trained on ``--records`` records (repeated), and afterwards the
released inference path greedily continues the same records.  ``exact_match`` is
measured on the model's own training data, so a flat zero is a
plumbing/optimisation result and can never be blamed on held-out difficulty.
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

from src.data.dataset_api import load_tokenizer, pack_records  # noqa: E402
from src.evaluation.evaluator import render_prompt  # noqa: E402
from src.inference.engine import InferenceEngine  # noqa: E402
from src.model import TinyMeConfig  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<\|[a-z_]+\|>", " ", text)).strip().lower()


def _gold(record: dict) -> str:
    return "\n".join(seg["text"] for seg in record.get("segments", [])
                     if seg.get("target", True) and seg["role"] in
                     ("assistant", "final", "tool_call", "code"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--split", default="train")
    ap.add_argument("--template-prefix", default="copy/")
    ap.add_argument("--records", type=int, default=8)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--micro-batch", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--seed", type=int, default=777001)
    ap.add_argument("--experiment", default="PROBE-MEMORISE")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tokenizer = load_tokenizer(args.dataset)
    records = []
    for line in (ROOT / "datasets" / "versions" / args.dataset /
                 f"{args.split}.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        if str(rec.get("template_id") or "").startswith(args.template_prefix):
            records.append(rec)
        if len(records) >= args.records:
            break
    repeated = records * max(1, (args.micro_batch + len(records) - 1) // len(records))
    data, stats = pack_records(repeated, tokenizer, args.seq_len)
    print(f"records={len(records)} blocks={data['input_ids'].shape[0]} "
          f"active={stats.get('active_target_tokens')}", flush=True)

    cfg = TrainConfig(experiment_id=args.experiment, experiment_name="memorisation probe",
                      stage="sft", architecture="nano", dataset_version=args.dataset,
                      seed=args.seed, seq_len=args.seq_len, micro_batch_size=args.micro_batch,
                      grad_accum_steps=args.grad_accum, learning_rate=args.lr,
                      warmup_steps=args.warmup, max_steps=args.steps,
                      eval_every=max(10, args.steps // 6), checkpoint_every=max(10, args.steps // 6),
                      keep_last_checkpoints=1, max_val_batches=2, resume=False,
                      notes="single-task memorisation probe; diagnostic only")
    trainer = Trainer(cfg, model_cfg=TinyMeConfig.from_name("nano"), tokenizer=tokenizer)
    t0 = time.perf_counter()
    trainer.fit(data, data, resume=False)
    duration = time.perf_counter() - t0

    ckpt = ROOT / "checkpoints" / args.experiment / "latest.safetensors"
    if not ckpt.exists():
        ckpt = ROOT / "checkpoints" / args.experiment / "best.safetensors"
    engine = InferenceEngine(ckpt, ROOT / "datasets" / "versions" / args.dataset / "tokenizer.json",
                             backend="numpy")
    rows = []
    for record in records:
        generated = engine.generate(render_prompt(record), max_new_tokens=64, temperature=0.0,
                                    repetition_penalty=1.0)
        rows.append({"record_id": record.get("record_id"), "template_id": record.get("template_id"),
                     "gold": _norm(_gold(record))[:120], "generated": _norm(generated)[:120],
                     "exact": int(_norm(_gold(record)) == _norm(generated))})
    exact = round(sum(r["exact"] for r in rows) / max(1, len(rows)), 4)
    report = {"probe": "memorise", "dataset": args.dataset, "template_prefix": args.template_prefix,
              "records": len(records), "steps": args.steps, "seq_len": args.seq_len,
              "duration_s": round(duration, 1), "exact_match": exact, "samples": rows}
    out = Path(args.out) if args.out else ROOT / "docs" / "audit_evidence" / "memorise_probe.json"
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"exact_match": exact, "duration_s": report["duration_s"]}, indent=1))
    for r in rows[:4]:
        print("  gold:", r["gold"], "| gen:", r["generated"], "|", r["exact"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
