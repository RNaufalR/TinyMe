"""Corrected TinyMe training engine (P0-03, P0-04, P0-09, P0-10, P0-11, P0-12, P0-15, P0-16).

What changed relative to the audited baseline:

* **true gradient accumulation** — microbatches are scanned with frozen
  parameters, gradients are summed (``jax.lax.scan``), and the optimizer is
  updated **once** per accumulation step;
* **masked loss** — every label carries ``-100`` where the token is not a
  target; tool results, prompts and padding contribute exactly zero;
* **padding never trains or evaluates** — the evaluation mean is normalised by
  active target tokens;
* **deterministic sampler** with persisted epoch/position;
* **complete checkpoint/resume** (params + optimizer + scheduler + RNG +
  sampler + fingerprints) through :mod:`src.training.checkpoint`;
* **LR continuity on resume** — the schedule is a pure function of the global
  step, so ``lr(step)`` is identical before and after a resume;
* **numerical stability** — finite-loss checks, gradient-norm logging, and
  explicit ``NON_FINITE`` events instead of silent NaN propagation;
* **honest ``compute_dtype``** — activations run in the requested dtype and the
  actually-used dtypes are logged.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import yaml
from jax import tree_util

from ..data.dataset_api import load_manifest, load_split, load_tokenizer
from ..model import TinyMeConfig, assert_vocab_compatible, count_parameters, forward, init_params
from ..utils.io_utils import REPO_ROOT, human_bytes, write_json, write_jsonl
from ..utils.rng import RNGService
from .checkpoint import environment_report, load_checkpoint, prune_checkpoints, save_checkpoint
from .sampler import EpochSampler, SamplerState

logger = logging.getLogger("tinyme.train")

CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
EXPERIMENTS_DIR = REPO_ROOT / "experiments"
IGNORE_INDEX = -100


@dataclass
class TrainConfig:
    experiment_id: str = "EXP-002-CORRECTED-NANO"
    experiment_name: str = "corrected baseline"
    stage: str = "pretrain"                     # pretrain | sft
    architecture: str = "nano"
    dataset_version: str = "dataset_v2"
    tokenizer_version: str = "tok-v2"
    seed: int = 20261002
    seq_len: int = 256
    micro_batch_size: int = 8
    grad_accum_steps: int = 4
    learning_rate: float = 6e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 50
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    max_steps: int = 400
    eval_every: int = 50
    checkpoint_every: int = 100
    keep_last_checkpoints: int = 3
    label_smoothing: float = 0.0
    compute_dtype: str = "float32"
    max_val_batches: int = 16
    notes: str = ""
    resume: bool = True
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TrainConfig":
        data = yaml.safe_load(Path(path).read_text()) or {}
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# ------------------------------------------------------------------- loss
def masked_cross_entropy(logits: jnp.ndarray, labels: jnp.ndarray,
                         loss_mask: jnp.ndarray, label_smoothing: float = 0.0) -> jnp.ndarray:
    """Per-token cross-entropy; masked positions contribute exactly zero.

    Returns the *sum* over active positions together with the active count, so
    callers can normalise globally rather than per-microbatch.
    """
    logits = logits.astype(jnp.float32)
    target = jnp.clip(jnp.where(labels < 0, 0, labels), 0, logits.shape[-1] - 1)
    logp = jax.nn.log_softmax(logits, axis=-1)
    nll = -jnp.take_along_axis(logp, target[..., None], axis=-1)[..., 0]
    if label_smoothing > 0:
        smooth = -logp.mean(axis=-1)
        nll = (1 - label_smoothing) * nll + label_smoothing * smooth
    # label -100 is the ignore index; it must contribute zero even if the
    # caller's mask is inconsistent with the labels.
    mask = loss_mask.astype(jnp.float32) * (labels >= 0).astype(jnp.float32)
    return (nll * mask).sum(), mask.sum()


# ------------------------------------------------------------------ trainer
class Trainer:
    def __init__(self, cfg: TrainConfig, model_cfg: TinyMeConfig | None = None,
                 params: dict[str, Any] | None = None, tokenizer: Any | None = None):
        self.cfg = cfg
        self.model_cfg = model_cfg or TinyMeConfig.from_name(cfg.architecture)
        self.rng = RNGService(cfg.seed)
        self.tokenizer = tokenizer or load_tokenizer(cfg.dataset_version)
        assert_vocab_compatible(self.model_cfg, int(self.tokenizer.vocab_size), context="trainer init")
        self.params = params if params is not None else init_params(self.model_cfg, seed=cfg.seed)
        self.param_count = count_parameters(self.params)
        self.step = 0
        self.best_val_loss = float("inf")
        self.metrics: list[dict[str, Any]] = []
        self.events: list[dict[str, Any]] = []
        self.sampler_state: SamplerState | None = None
        self.opt_state: Any = None

        self.exp_dir = EXPERIMENTS_DIR / cfg.experiment_id
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = CHECKPOINTS_DIR / cfg.experiment_id
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.optimizer = self._build_optimizer()
        self.opt_state = self.optimizer.init(self.params)
        self._compile()

    # --------------------------------------------------------------- setup
    def _lr_jnp(self, step: Any) -> jnp.ndarray:
        """Warmup + cosine LR as a pure (jittable) function of the global step."""
        cfg = self.cfg
        warmup = float(max(1, min(cfg.warmup_steps, max(cfg.max_steps // 10, 1))))
        s = jnp.asarray(step, jnp.float32)
        warm = cfg.learning_rate * s / warmup
        progress = jnp.clip((s - warmup) / max(1.0, cfg.max_steps - warmup), 0.0, 1.0)
        cosine = 0.5 * (1.0 + jnp.cos(jnp.pi * progress))
        decay = cfg.learning_rate * (cfg.min_lr_ratio + (1 - cfg.min_lr_ratio) * cosine)
        return jnp.where(s <= warmup, warm, decay)

    def lr_at(self, step: int) -> float:
        """Learning rate at a global step (identical before and after resume)."""
        return float(self._lr_jnp(step))

    def _build_optimizer(self) -> optax.GradientTransformation:
        # The schedule reads Optax's internal step count, which is part of the
        # optimizer state and therefore survives checkpoints: LR continuity on
        # resume is structural, not accidental (P0-12).
        return optax.chain(
            optax.clip_by_global_norm(self.cfg.grad_clip_norm),
            optax.adamw(learning_rate=lambda count: self._lr_jnp(count + 1),
                        weight_decay=self.cfg.weight_decay, b1=0.9, b2=0.95, eps=1e-8),
        )

    def _compile(self) -> None:
        cfg, model_cfg, dtype = self.cfg, self.model_cfg, self.cfg.compute_dtype

        def per_microb_loss(p, tokens, labels, mask, doc_ids):
            logits = forward(p, tokens, model_cfg, dtype=dtype, doc_ids=doc_ids)
            total, count = masked_cross_entropy(logits, labels, mask, cfg.label_smoothing)
            return total, count

        def accum_grads(p, batch):
            zeros = tree_util.tree_map(jnp.zeros_like, p)

            def scan_body(carry, microbatch):
                grad_acc, loss_acc, count_acc = carry
                t, l, m, d = microbatch
                (total, count), grads = jax.value_and_grad(per_microb_loss, has_aux=True)(p, t, l, m, d)
                grad_acc = tree_util.tree_map(lambda a, b: a + b, grad_acc, grads)
                return (grad_acc, loss_acc + total, count_acc + count), None

            (grad_sum, loss_sum, count_sum), _ = jax.lax.scan(
                scan_body, (zeros, jnp.zeros((), jnp.float32), jnp.zeros((), jnp.float32)), batch)
            mean_grads = tree_util.tree_map(lambda g: g / jnp.maximum(count_sum, 1.0), grad_sum)
            return mean_grads, loss_sum, count_sum

        def apply_update(p, opt_state, grads):
            updates, new_state = self.optimizer.update(grads, opt_state, p)
            return optax.apply_updates(p, updates), new_state

        self._accum_grads = jax.jit(accum_grads)
        self._apply_update = jax.jit(apply_update)

        def eval_loss(p, tokens, labels, mask, doc_ids):
            logits = forward(p, tokens, model_cfg, dtype=dtype, doc_ids=doc_ids)
            total, count = masked_cross_entropy(logits, labels, mask, 0.0)
            return total, count

        self._eval = jax.jit(eval_loss)

    # ---------------------------------------------------------- checkpoint
    def _fingerprints(self) -> dict[str, Any]:
        try:
            manifest = load_manifest(self.cfg.dataset_version)
        except FileNotFoundError:
            manifest = {}
        return {
            "experiment_id": self.cfg.experiment_id,
            "architecture": self.cfg.architecture,
            "model_config_hash": self.model_cfg.model_hash(),
            "tokenizer_hash": manifest.get("tokenizer_hash", ""),
            "dataset_fingerprint": manifest.get("dataset_fingerprint", ""),
            "seq_len": self.cfg.seq_len,
        }

    def save(self, name: str = "latest", extra: dict[str, Any] | None = None) -> Path:
        meta: dict[str, Any] = {
            "step": self.step,
            "best_val_loss": self.best_val_loss,
            "param_count": self.param_count,
            "config": self.cfg.to_dict(),
            "model_config": self.model_cfg.to_dict(),
            "sampler_state": (self.sampler_state or SamplerState()).to_dict(),
            "rng_state": self.rng.save_state(),
            "environment": environment_report(),
            **self._fingerprints(),
        }
        if extra:
            meta.update(extra)
        path = save_checkpoint(self.ckpt_dir, name, self.params, self.opt_state, meta)
        prune_checkpoints(self.ckpt_dir, keep_last=self.cfg.keep_last_checkpoints)
        return path

    def resume_from_checkpoint(self, name: str = "latest") -> bool:
        try:
            params, opt_state, meta = load_checkpoint(
                self.ckpt_dir, name, self.params, self.opt_state,
                expected_fingerprints=self._fingerprints(), require_optimizer=True)
        except FileNotFoundError:
            logger.info("no checkpoint %r to resume", name)
            return False
        self.params = params
        self.opt_state = opt_state
        self.step = int(meta.get("step", 0))
        self.best_val_loss = float(meta.get("best_val_loss", float("inf")))
        self.sampler_state = SamplerState.from_dict(meta.get("sampler_state", {}))
        self.rng.load_state(meta.get("rng_state"))
        logger.info("resumed %s from step %d (best_val=%.4f, lr=%.3e)",
                    self.cfg.experiment_id, self.step, self.best_val_loss, self.lr_at(self.step))
        return True

    # ----------------------------------------------------------- evaluation
    def evaluate(self, data: dict[str, np.ndarray], max_batches: int | None = None) -> dict[str, float]:
        n = data["input_ids"].shape[0]
        bs = self.cfg.micro_batch_size * self.cfg.grad_accum_steps
        max_batches = max_batches or self.cfg.max_val_batches
        total, count, batches = 0.0, 0.0, 0
        for start in range(0, n, bs):
            if batches >= max_batches:
                break
            sl = slice(start, min(start + bs, n))
            if sl.stop - sl.start < 2:
                continue
            t, c = self._eval(self.params, jnp.asarray(data["input_ids"][sl]),
                              jnp.asarray(data["labels"][sl]), jnp.asarray(data["loss_mask"][sl]),
                              jnp.asarray(data["doc_ids"][sl]))
            total += float(t)
            count += float(c)
            batches += 1
        if count == 0:
            return {"val_loss": float("nan"), "val_perplexity": float("nan"),
                    "active_tokens": 0.0}
        loss = total / count
        return {"val_loss": loss, "val_perplexity": float(np.exp(min(loss, 20.0))),
                "active_tokens": count}

    # ---------------------------------------------------------------- train
    def fit(self, train_data: dict[str, np.ndarray], val_data: dict[str, np.ndarray] | None = None,
            resume: bool | None = None) -> dict[str, Any]:
        cfg = self.cfg
        resume = cfg.resume if resume is None else resume
        if resume and self.opt_state is not None and self.step == 0:
            self.resume_from_checkpoint("latest")

        n = train_data["input_ids"].shape[0]
        batch_size = cfg.micro_batch_size * cfg.grad_accum_steps
        sampler = EpochSampler(n, batch_size, seed=cfg.seed,
                               fingerprint=self._fingerprints()["dataset_fingerprint"])
        if self.sampler_state is None or self.sampler_state.dataset_size != n:
            self.sampler_state = sampler.initial_state()

        if val_data is None or val_data["input_ids"].shape[0] == 0:
            raise ValueError("a real validation split is required (train split cannot be used)")

        t_start = time.perf_counter()
        tokens_processed = 0
        skipped = 0
        logger.info("training %s [%s]: %d params, seq_len=%d, micro_bs=%d, accum=%d, "
                    "effective_batch=%d, steps=%d, dtype=%s",
                    cfg.experiment_id, cfg.stage, self.param_count, cfg.seq_len,
                    cfg.micro_batch_size, cfg.grad_accum_steps, batch_size, cfg.max_steps,
                    cfg.compute_dtype)
        logger.info("dtypes: embeddings=%s weights=%s loss=float32",
                    self.params["tok_emb"].dtype, self.params["tok_emb"].dtype)

        start_step = self.step
        while self.step < cfg.max_steps:
            idx = sampler.batch_indices(self.sampler_state.epoch, self.sampler_state.position)
            batch = {k: v[idx] for k, v in train_data.items()}
            n_micro = cfg.micro_batch_size
            expected = cfg.grad_accum_steps * n_micro
            if batch["input_ids"].shape[0] != expected:
                raise ValueError(f"sampler returned {batch['input_ids'].shape[0]} rows; "
                                 f"expected {expected} (grad_accum*micro_batch)")
            tokens = batch["input_ids"].reshape(cfg.grad_accum_steps, n_micro, cfg.seq_len)
            labels = batch["labels"].reshape(cfg.grad_accum_steps, n_micro, cfg.seq_len)
            mask = batch["loss_mask"].reshape(cfg.grad_accum_steps, n_micro, cfg.seq_len)
            doc_ids = batch["doc_ids"].reshape(cfg.grad_accum_steps, n_micro, cfg.seq_len)

            grads, loss_sum, count_sum = self._accum_grads(
                self.params, (jnp.asarray(tokens), jnp.asarray(labels),
                              jnp.asarray(mask), jnp.asarray(doc_ids)))
            loss = float(loss_sum) / max(float(count_sum), 1.0)
            grad_norm = float(optax.global_norm(grads))
            finite = bool(np.isfinite(loss)) and bool(np.isfinite(grad_norm))

            self.step += 1
            if finite:
                self.params, self.opt_state = self._apply_update(self.params, self.opt_state, grads)
            else:
                skipped += 1
                self.events.append({"step": self.step, "event": "NON_FINITE", "loss": loss,
                                    "grad_norm": grad_norm, "config": cfg.to_dict()})
                logger.error("step %d: non-finite loss/grad (loss=%s grad_norm=%s) — update skipped",
                             self.step, loss, grad_norm)

            self.sampler_state = sampler.state_after_step(self.sampler_state)
            tokens_processed += int(count_sum)

            record = {
                "experiment_id": cfg.experiment_id, "stage": cfg.stage, "step": self.step,
                "loss": loss, "lr": float(self.lr_at(self.step)), "grad_norm": grad_norm,
                "active_target_tokens": int(count_sum),
                "tokens_per_sec": round(tokens_processed / max(time.perf_counter() - t_start, 1e-6), 1),
                "elapsed_sec": round(time.perf_counter() - t_start, 2),
                "epoch": self.sampler_state.epoch,
                "batch_position": self.sampler_state.position,
                "non_finite": not finite,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            self.metrics.append(record)

            if self.step % 10 == 0 or self.step == start_step + 1:
                logger.info("step %d/%d loss=%.4f lr=%.2e grad_norm=%.3f tok/s=%s epoch=%d",
                            self.step, cfg.max_steps, loss, record["lr"], grad_norm,
                            record["tokens_per_sec"], self.sampler_state.epoch)

            if val_data is not None and self.step % cfg.eval_every == 0:
                ev = self.evaluate(val_data)
                record.update(ev)
                if ev["val_loss"] < self.best_val_loss:
                    self.best_val_loss = ev["val_loss"]
                    self.save("best", extra={"val_loss": ev["val_loss"]})
                logger.info("  eval step %d: val_loss=%.4f val_ppl=%.2f (active=%d)",
                            self.step, ev["val_loss"], ev["val_perplexity"], int(ev["active_tokens"]))

            if self.step % cfg.checkpoint_every == 0:
                self.save("latest")
                self.save(f"step_{self.step:06d}")

        self.save("latest")
        final_val = self.evaluate(val_data)
        write_jsonl(self.metrics, self.exp_dir / "metrics.jsonl")
        if self.events:
            write_jsonl(self.events, self.exp_dir / "events.jsonl")

        elapsed = time.perf_counter() - t_start
        summary = {
            "experiment_id": cfg.experiment_id,
            "experiment_name": cfg.experiment_name,
            "stage": cfg.stage,
            "architecture": cfg.architecture,
            "parameter_count": self.param_count,
            "dataset_version": cfg.dataset_version,
            "tokenizer_version": cfg.tokenizer_version,
            "seed": cfg.seed,
            "seq_len": cfg.seq_len,
            "effective_batch_size": batch_size,
            "grad_accum_steps": cfg.grad_accum_steps,
            "compute_dtype": cfg.compute_dtype,
            "steps_executed": self.step,
            "steps_this_run": self.step - start_step,
            "final_train_loss": float(np.mean([m["loss"] for m in self.metrics[-20:]])),
            "final_val_loss": final_val["val_loss"],
            "final_val_perplexity": final_val["val_perplexity"],
            "final_val_active_tokens": final_val["active_tokens"],
            "best_val_loss": self.best_val_loss,
            "tokens_processed": tokens_processed,
            "non_finite_steps": skipped,
            "epochs_completed": self.sampler_state.epoch,
            "wall_clock_seconds": round(elapsed, 2),
            "avg_tokens_per_sec": round(tokens_processed / max(elapsed, 1e-6), 1),
            "checkpoint_bytes": (self.ckpt_dir / "latest.safetensors").stat().st_size,
            "fingerprints": self._fingerprints(),
            "environment": environment_report(),
            "config": cfg.to_dict(),
            "status": "COMPLETED",
        }
        write_json(summary, self.exp_dir / "summary.json")
        logger.info("done: %d steps, val_loss=%.4f, val_ppl=%.2f, %s artifacts",
                    self.step, final_val["val_loss"], final_val["val_perplexity"],
                    human_bytes(int(summary["checkpoint_bytes"])))
        return summary


# ------------------------------------------------------------- compatibility
def load_dataset_for_stage(cfg: TrainConfig, split: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Load a split with the corrected API (no implicit train.jsonl)."""
    return load_split(cfg.dataset_version, split, seq_len=cfg.seq_len,
                      tokenizer=load_tokenizer(cfg.dataset_version))
