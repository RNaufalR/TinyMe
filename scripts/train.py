#!/usr/bin/env python3
"""Corrected training entrypoint (never silently resumes an invalid run).

Loads the real ``validation`` split — never ``train.jsonl`` — and refuses to
start unless the validation split exists and is contamination-free.

Examples::

    # Stage A — domain pretraining from fresh initialisation
    python scripts/train.py --config configs/cpu.yaml --experiment EXP-002-CORRECTED-NANO \
        --stage pretrain --max-steps 350

    # Stage B — instruction / reasoning / tool-use SFT (loss-masked)
    python scripts/train.py --experiment EXP-004-TOOL-SFT --stage sft \
        --init-from EXP-002-CORRECTED-NANO --max-steps 180 --lr 3e-4

    # Base feasibility pilot
    python scripts/train.py --experiment EXP-003-CORRECTED-BASE-PILOT --arch base --max-steps 30
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

from src.data.dataset_api import load_manifest, load_split, load_tokenizer  # noqa: E402
from src.model import TinyMeConfig, count_parameters  # noqa: E402
from src.training.checkpoint import environment_report, load_checkpoint  # noqa: E402
from src.training.trainer import TrainConfig, Trainer  # noqa: E402
from src.utils.io_utils import REPO_ROOT, human_bytes, read_json, write_json  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
log = logging.getLogger("tinyme.train.cli")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None, help="YAML base config")
    ap.add_argument("--experiment", required=True, help="experiment id (never reuse)")
    ap.add_argument("--experiment-name", default="")
    ap.add_argument("--stage", default="pretrain", choices=["pretrain", "sft", "mixed"],
                    help=("pretrain = plain documents, sft = structured (loss-masked) records, "
                          "mixed = both shuffled together (curriculum-B control)"))
    ap.add_argument("--arch", default=None, choices=["nano", "base", "medium"])
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--micro-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--eval-every", type=int, default=None)
    ap.add_argument("--checkpoint-every", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--init-from", default=None,
                    help="copy weights from another experiment (transfer learning)")
    ap.add_argument("--vocab-size", type=int, default=None,
                    help="override (only for controlled tokenizer ablations)")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--notes", default="")
    args = ap.parse_args()

    cfg = TrainConfig.from_yaml(args.config) if args.config else TrainConfig()
    for attr, value in (("architecture", args.arch), ("dataset_version", args.dataset),
                        ("seq_len", args.seq_len), ("micro_batch_size", args.micro_batch),
                        ("grad_accum_steps", args.grad_accum), ("max_steps", args.max_steps),
                        ("learning_rate", args.lr), ("warmup_steps", args.warmup),
                        ("eval_every", args.eval_every), ("checkpoint_every", args.checkpoint_every),
                        ("seed", args.seed), ("compute_dtype", args.dtype)):
        if value is not None:
            setattr(cfg, attr, value)
    cfg.experiment_id = args.experiment
    cfg.experiment_name = args.experiment_name or f"{args.experiment} ({cfg.stage})"
    cfg.stage = args.stage
    cfg.notes = args.notes
    if args.no_resume:
        cfg.resume = False

    manifest = load_manifest(cfg.dataset_version)
    if manifest.get("train_validation_test_contamination") != "PASS":
        log.error("refusing to train: contamination gate is %s",
                  manifest.get("train_validation_test_contamination"))
        return 2
    tokenizer = load_tokenizer(cfg.dataset_version)
    model_cfg = TinyMeConfig.from_name(cfg.architecture)
    if args.vocab_size:
        model_cfg = TinyMeConfig(**{**model_cfg.to_dict(), "vocab_size": args.vocab_size})
    model_cfg.vocab_size = int(tokenizer.vocab_size)

    train_data, train_stats = load_split(cfg.dataset_version, "train", seq_len=cfg.seq_len,
                                         tokenizer=tokenizer, stage=cfg.stage)
    val_data, val_stats = load_split(cfg.dataset_version, "validation", seq_len=cfg.seq_len,
                                     tokenizer=tokenizer, stage=cfg.stage)
    if train_data["input_ids"].shape[0] == 0 or val_data["input_ids"].shape[0] == 0:
        log.error("empty split/stage (train=%s val=%s); nothing to do",
                  train_data["input_ids"].shape, val_data["input_ids"].shape)
        return 3
    log.info("train blocks=%d active_targets=%d | val blocks=%d active_targets=%d",
             train_data["input_ids"].shape[0], train_stats["active_target_tokens"],
             val_data["input_ids"].shape[0], val_stats["active_target_tokens"])

    trainer = Trainer(cfg, model_cfg=model_cfg, tokenizer=tokenizer)
    init_fingerprints = None
    if args.init_from:
        ckpt_dir = REPO_ROOT / "checkpoints" / args.init_from
        if not ckpt_dir.exists():
            log.error("--init-from experiment %s has no checkpoints", args.init_from)
            return 4
        name = "best" if (ckpt_dir / "best.safetensors").exists() else "latest"
        # Deliberate transfer: architecture, tokenizer and sequence length must
        # match exactly; the *dataset* fingerprint is allowed to differ (that is
        # the point of a controlled transfer to a revised data revision) but the
        # difference is reported, never hidden.
        params, _, meta = load_checkpoint(
            ckpt_dir, name, trainer.params, None,
            expected_fingerprints={"model_config_hash": model_cfg.model_hash(),
                                   "tokenizer_hash": manifest.get("tokenizer_hash", ""),
                                   "seq_len": cfg.seq_len},
            require_optimizer=False)
        if not meta.get("experiment_id"):
            raise SystemExit("init-from checkpoint has no experiment identity")
        if meta.get("dataset_fingerprint") != manifest.get("dataset_fingerprint"):
            log.warning("transfer learning: init checkpoint was trained on dataset fingerprint %s, "
                        "this run uses %s (recorded in run_config.json)",
                        str(meta.get("dataset_fingerprint"))[:16],
                        str(manifest.get("dataset_fingerprint"))[:16])
        trainer.params = params
        trainer.param_count = count_parameters(params)
        init_fingerprints = {k: meta.get(k) for k in
                             ("experiment_id", "step", "dataset_fingerprint", "tokenizer_hash",
                              "model_config_hash", "best_val_loss")}
        log.info("initialised weights from %s/%s (step %s) — new experiment id %s",
                 args.init_from, name, meta.get("step"), cfg.experiment_id)

    # record the exact run configuration before training starts
    run_meta = {
        "experiment_id": cfg.experiment_id, "stage": cfg.stage,
        "dataset_version": cfg.dataset_version, "dataset_fingerprint": manifest.get("dataset_fingerprint"),
        "tokenizer_hash": manifest.get("tokenizer_hash"), "tokenizer_vocab": tokenizer.vocab_size,
        "model_config": model_cfg.to_dict(), "train_config": cfg.to_dict(),
        "train_stats": train_stats, "val_stats": val_stats,
        "environment": environment_report(), "init_from": args.init_from,
        "init_from_fingerprints": (init_fingerprints if args.init_from else None),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    exp_dir = REPO_ROOT / "experiments" / cfg.experiment_id
    write_json(run_meta, exp_dir / "run_config.json")

    summary = trainer.fit(train_data, val_data, resume=not args.no_resume)
    summary["train_stats"] = train_stats
    summary["val_stats"] = val_stats
    write_json(summary, exp_dir / "summary.json")
    log.info("checkpoint: %s", human_bytes(int(summary["checkpoint_bytes"])))
    print(json.dumps({k: summary[k] for k in (
        "experiment_id", "stage", "parameter_count", "steps_executed", "final_val_loss",
        "final_val_perplexity", "tokens_processed", "wall_clock_seconds", "avg_tokens_per_sec")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
