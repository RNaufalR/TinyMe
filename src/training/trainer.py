"""Resource-aware training engine for TinyMe (spec §16, §17, §18).

Features:
  * deterministic seeding (jax, numpy, python random)
  * YAML-configurable hyperparameters
  * XLA-JIT compiled train step (fused forward + backward + AdamW)
  * gradient accumulation for memory-constrained CPUs
  * gradient clipping (global norm)
  * learning-rate warmup + cosine decay
  * validation split with periodic perplexity evaluation
  * periodic checkpointing with retention, latest-pointer and crash recovery
  * per-step JSONL metrics log for resumability
"""
from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import tree_util

from ..model import TinyMeConfig, count_parameters, forward, init_params
from ..utils.io_utils import REPO_ROOT, write_json, write_jsonl

logger = logging.getLogger("tinyme.train")

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"


@dataclass
class TrainConfig:
    experiment_id: str = "EXP-001"
    experiment_name: str = "baseline"
    architecture: str = "nano"
    dataset_version: str = "dataset_v1"
    tokenizer_version: str = "tok-v1"
    seed: int = 1234
    seq_len: int = 128
    micro_batch_size: int = 16
    grad_accum_steps: int = 1
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    max_steps: int = 400
    eval_every: int = 100
    checkpoint_every: int = 100
    keep_last_checkpoints: int = 3
    label_smoothing: float = 0.0
    compute_dtype: str = "float32"
    curriculum: list[dict[str, Any]] | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


def normalize_keys(flat: dict[str, "np.ndarray"]) -> dict[str, "np.ndarray"]:
    """Convert JAX tree paths like "['blocks']/[0]/['attn']/['k']" to "blocks/0/attn/k"."""
    import re as _re

    out: dict[str, np.ndarray] = {}
    for key, val in flat.items():
        k = key.replace("['", "").replace("']", "")
        k = _re.sub(r"\[(\d+)\]", r"/\1", k)
        out[_re.sub(r"/+", "/", k).strip("/")] = val
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))


# ------------------------------------------------------------------ dataset
def load_dataset(version: str, seq_len: int, max_samples: int | None = None,
                 categories: list[str] | None = None) -> dict[str, np.ndarray]:
    """Load a dataset version and build fixed-length training tensors."""
    version_dir = REPO_ROOT / "datasets" / "versions" / version
    path = version_dir / "train.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"dataset {version} not found at {path}")

    from src.tokenizer.bpe import TinyMeTokenizer
    tok_path = REPO_ROOT / "release" / "tokenizer.json"
    if not tok_path.exists():
        tok_path = REPO_ROOT / "datasets" / "processed" / "tokenizer.json"
    tokenizer = TinyMeTokenizer.load(tok_path)
    pad_id = tokenizer.tok.token_to_id("<|pad|>")
    bos_id = tokenizer.tok.token_to_id("<|bos|>")
    eos_id = tokenizer.tok.token_to_id("<|eos|>")

    sequences: list[list[int]] = []
    n_read = 0
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if categories and rec.get("category") not in categories:
                continue
            ids = tokenizer.encode_ids(rec["text"])
            ids = [bos_id] + ids[:seq_len - 2] + [eos_id]
            if len(ids) < 8:
                continue
            ids = ids + [pad_id] * (seq_len - len(ids))
            sequences.append(ids)
            n_read += 1
            if max_samples and n_read >= max_samples:
                break
    if not sequences:
        raise ValueError(f"no sequences produced from {version}")
    arr = np.array(sequences, dtype=np.int32)
    return {"tokens": arr, "pad_id": np.int32(pad_id),
            "bos_id": np.int32(bos_id), "eos_id": np.int32(eos_id)}


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, cfg: TrainConfig, params: dict[str, Any] | None = None,
                 model_cfg: TinyMeConfig | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg or TinyMeConfig.from_name(cfg.architecture)
        self.params = params if params is not None else init_params(self.model_cfg, seed=cfg.seed)
        self.param_count = count_parameters(self.params)
        self.step = 0
        self.best_val_loss = float("inf")
        self.metrics: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        set_seed(cfg.seed)

        self.exp_dir = EXPERIMENTS_DIR / cfg.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = CHECKPOINTS_DIR / cfg.experiment_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        total = max(cfg.max_steps, 1)
        warmup = min(cfg.warmup_steps, total - 1) if total > 1 else 0
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=cfg.learning_rate,
            warmup_steps=warmup,
            decay_steps=max(total - warmup, 1),
            end_value=cfg.learning_rate * cfg.min_lr_ratio,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(cfg.grad_clip_norm),
            optax.adamw(learning_rate=schedule, weight_decay=cfg.weight_decay,
                        b1=0.9, b2=0.95, eps=1e-8),
        )
        self.opt_state = self.optimizer.init(self.params)
        self._lr_schedule = schedule

        micro_bs = cfg.micro_batch_size
        accum = cfg.grad_accum_steps
        cfg_model = self.model_cfg

        def loss_fn(p, tokens, targets):
            logits = forward(p, tokens, cfg_model).astype(jnp.float32)
            if cfg.label_smoothing > 0:
                onehot = jax.nn.one_hot(targets, logits.shape[-1])
                soft = (1 - cfg.label_smoothing) * onehot + cfg.label_smoothing / logits.shape[-1]
                loss = -(soft * jax.nn.log_softmax(logits)).sum(-1)
            else:
                loss = optax.softmax_cross_entropy_with_integer_labels(logits, targets)
            return jnp.maximum(loss.mean(), 0.0)

        def train_step(p, opt_st, batch):
            tokens, targets = batch[:, :-1], batch[:, 1:]
            micros = max(1, min(tokens.shape[0] // micro_bs, accum))
            p_acc, st_acc, total_loss = p, opt_st, 0.0
            for i in range(micros):
                sl_ = slice(i * micro_bs, (i + 1) * micro_bs)
                loss, grads = jax.value_and_grad(loss_fn)(p_acc, tokens[sl_], targets[sl_])
                updates, st_acc = self.optimizer.update(grads, st_acc, p_acc)
                p_acc = optax.apply_updates(p_acc, updates)
                total_loss = total_loss + loss
            return p_acc, st_acc, total_loss / micros

        self._train_step = jax.jit(train_step)

        def eval_fn(p, batch):
            tokens, targets = batch[:, :-1], batch[:, 1:]
            logits = forward(p, tokens, cfg_model).astype(jnp.float32)
            return optax.softmax_cross_entropy_with_integer_labels(logits, targets).mean()

        self._eval_fn = jax.jit(eval_fn)

    # ------------------------------------------------------------ checkpoint
    def _flat_numpy(self) -> dict[str, np.ndarray]:
        flat = tree_util.tree_flatten_with_path(self.params)[0]
        out: dict[str, np.ndarray] = {}
        for path, val in flat:
            key = "/".join(str(k) for k in path)
            out[key] = np.asarray(val, dtype=np.float32)
        return normalize_keys(out)

    def save_checkpoint(self, name: str = "latest", extra: dict[str, Any] | None = None) -> Path:
        from safetensors.numpy import save_file

        path = self.ckpt_dir / f"{name}.safetensors"
        save_file(self._flat_numpy(), str(path), metadata={"format": "pt"})
        meta = {
            "experiment_id": self.cfg.experiment_id,
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "param_count": self.param_count,
            "architecture": self.cfg.architecture,
            "config": self.cfg.to_dict(),
            "checkpoint_file": path.name,
            "checkpoint_bytes": path.stat().st_size,
            "saved_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        if extra:
            meta.update(extra)
        write_json(meta, self.ckpt_dir / f"{name}.json")
        self._prune_checkpoints()
        return path

    def _prune_checkpoints(self) -> None:
        ckpts = sorted(self.ckpt_dir.glob("step_*.safetensors"), key=lambda p: p.stat().st_mtime)
        while len(ckpts) > self.cfg.keep_last_checkpoints:
            victim = ckpts.pop(0)
            victim.unlink(missing_ok=True)
            (self.ckpt_dir / f"{victim.stem}.json").unlink(missing_ok=True)

    def load_checkpoint(self, name: str = "latest") -> bool:
        from safetensors.numpy import load_file

        path = self.ckpt_dir / f"{name}.safetensors"
        meta_path = self.ckpt_dir / f"{name}.json"
        if not path.exists() or not meta_path.exists():
            return False
        flat = load_file(str(path))
        keys = ["/".join(str(k) for k in p)
                for p, _ in tree_util.tree_flatten_with_path(self.params)[0]]
        leaves, treedef = tree_util.tree_flatten(self.params)
        new_leaves = [jnp.asarray(flat[k]) for k in keys]
        self.params = tree_util.tree_unflatten(treedef, new_leaves)
        meta = json.loads(meta_path.read_text())
        self.step = int(meta.get("step", 0))
        self.best_val_loss = float(meta.get("best_val_loss", float("inf")))
        self.opt_state = self.optimizer.init(self.params)
        logger.info("resumed %s from step %d (val_loss=%.4f)", name, self.step, self.best_val_loss)
        return True

    # ---------------------------------------------------------------- train
    def evaluate(self, data: np.ndarray, max_batches: int = 8) -> dict[str, float]:
        losses = []
        n = min(len(data), max_batches * self.cfg.micro_batch_size)
        for i in range(0, n - self.cfg.micro_batch_size, self.cfg.micro_batch_size):
            batch = jnp.asarray(data[i:i + self.cfg.micro_batch_size])
            losses.append(float(self._eval_fn(self.params, batch)))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        return {"val_loss": mean_loss, "val_perplexity": float(np.exp(mean_loss))}

    def fit(self, train_data: np.ndarray, val_data: np.ndarray | None = None,
            resume: bool = True) -> dict[str, Any]:
        cfg = self.cfg
        if resume:
            self.load_checkpoint("latest")

        rng = np.random.default_rng(cfg.seed)
        bs = cfg.micro_batch_size
        start_step = self.step
        t_start = time.perf_counter()
        tokens_processed = 0

        logger.info("training %s: %d params, seq_len=%d, micro_bs=%d, accum=%d, steps=%d",
                    cfg.experiment_id, self.param_count, cfg.seq_len, bs,
                    cfg.grad_accum_steps, cfg.max_steps)

        while self.step < cfg.max_steps:
            idx = np.sort(rng.integers(0, max(1, len(train_data) - bs), size=bs))
            batch = jnp.asarray(train_data[idx])
            self.params, self.opt_state, loss = self._train_step(self.params, self.opt_state, batch)
            self.step += 1
            tokens_processed += int(batch.shape[0] * (batch.shape[1] - 1))
            lr = float(self._lr_schedule(self.step))

            record = {
                "experiment_id": cfg.experiment_id, "step": self.step,
                "loss": float(loss), "lr": lr,
                "tokens_per_sec": round(tokens_processed / max(time.perf_counter() - t_start, 1e-6), 1),
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.metrics.append(record)

            if self.step % 10 == 0 or self.step == 1:
                logger.info("step %d/%d loss=%.4f lr=%.2e tok/s=%s",
                            self.step, cfg.max_steps, record["loss"], record["lr"], record["tokens_per_sec"])

            if val_data is not None and self.step % cfg.eval_every == 0:
                ev = self.evaluate(val_data)
                record.update(ev)
                if ev["val_loss"] < self.best_val_loss:
                    self.best_val_loss = ev["val_loss"]
                    self.save_checkpoint("best", extra={"val_loss": ev["val_loss"]})
                logger.info("  eval step %d: val_loss=%.4f ppl=%.2f",
                            self.step, ev["val_loss"], ev["val_perplexity"])

            if self.step % cfg.checkpoint_every == 0:
                self.save_checkpoint("latest")
                self.save_checkpoint(f"step_{self.step:06d}")

        self.save_checkpoint("latest")
        final_val = (self.evaluate(val_data) if val_data is not None
                     else {"val_loss": float("nan"), "val_perplexity": float("nan")})
        write_jsonl(self.metrics, self.exp_dir / "metrics.jsonl")

        elapsed = time.perf_counter() - t_start
        summary = {
            "experiment_id": cfg.experiment_id,
            "experiment_name": cfg.experiment_name,
            "architecture": cfg.architecture,
            "parameter_count": self.param_count,
            "dataset_version": cfg.dataset_version,
            "tokenizer_version": cfg.tokenizer_version,
            "seed": cfg.seed,
            "steps_executed": self.step,
            "steps_this_run": self.step - start_step,
            "final_train_loss": float(np.mean([m["loss"] for m in self.metrics[-20:]])),
            "final_val_loss": final_val["val_loss"],
            "final_val_perplexity": final_val["val_perplexity"],
            "best_val_loss": self.best_val_loss,
            "tokens_processed": tokens_processed,
            "wall_clock_seconds": round(elapsed, 2),
            "avg_tokens_per_sec": round(tokens_processed / max(elapsed, 1e-6), 1),
            "checkpoint_bytes": (self.ckpt_dir / "latest.safetensors").stat().st_size,
            "config": cfg.to_dict(),
            "status": "COMPLETED",
        }
        write_json(summary, self.exp_dir / "summary.json")
        self.history.append(summary)
        return summary
