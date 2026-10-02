#!/usr/bin/env python3
"""Train a TinyMe model (spec §16, §17, §18).

Usage:
    python scripts/train.py --config configs/cpu.yaml
    python scripts/train.py --experiment EXP-001 --steps 300 --arch nano
    python scripts/train.py --resume --experiment EXP-001 --steps 600
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.training.trainer import TrainConfig, Trainer, load_dataset  # noqa: E402
from src.utils.io_utils import REPO_ROOT, human_bytes, write_json  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="TinnyMe training")
    ap.add_argument("--config", default=None)
    ap.add_argument("--experiment", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--arch", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--notes", default="")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
    log = logging.getLogger("train")

    if args.config:
        cfg = TrainConfig.from_yaml(REPO_ROOT / args.config)
    else:
        cfg = TrainConfig()
    for key, val in (("experiment_id", args.experiment), ("experiment_name", args.name),
                     ("architecture", args.arch), ("dataset_version", args.dataset),
                     ("max_steps", args.steps), ("seq_len", args.seq_len),
                     ("micro_batch_size", args.batch_size), ("learning_rate", args.lr),
                     ("seed", args.seed)):
        if val is not None:
            setattr(cfg, key, val)
    if args.notes:
        cfg.notes = args.notes

    log.info("config: %s", cfg.to_dict())
    data = load_dataset(cfg.dataset_version, cfg.seq_len, max_samples=args.max_samples)
    train = data["tokens"]
    eval_ = load_dataset(cfg.dataset_version, cfg.seq_len)["tokens"][:32]

    trainer = Trainer(cfg)
    log.info("model: %s, %d params (~%s FP32)", cfg.architecture, trainer.param_count,
             human_bytes(trainer.param_count * 4))

    summary = trainer.fit(train, eval_, resume=not args.no_resume)

    # append to the experiment log
    from src.utils.io_utils import read_jsonl
    log_path = REPO_ROOT / "EXPERIMENT_LOG.jsonl"
    existing = read_jsonl(log_path) if log_path.exists() else []
    existing.append({
        "experiment_id": cfg.experiment_id, "name": cfg.experiment_name,
        "arch": cfg.architecture, "params": trainer.param_count,
        "dataset": cfg.dataset_version, "seed": cfg.seed,
        "steps": summary["steps_executed"],
        "train_loss": summary["final_train_loss"],
        "val_loss": summary["final_val_loss"],
        "val_ppl": summary["final_val_perplexity"],
        "wall_clock_s": summary["wall_clock_seconds"],
        "tokens_per_sec": summary["avg_tokens_per_sec"],
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    from src.utils.io_utils import write_jsonl
    write_jsonl(existing, log_path)

    print("\n=== TRAINING SUMMARY ===")
    for k in ("experiment_id", "architecture", "parameter_count", "steps_executed",
              "final_train_loss", "final_val_loss", "final_val_perplexity",
              "avg_tokens_per_sec", "wall_clock_seconds", "checkpoint_bytes"):
        print(f"{k:24s}: {summary[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
