"""Safe, atomic, verifiable checkpoints (corrective audit P0-11).

A checkpoint contains everything needed to continue a run *exactly*:

* model parameters (float32 master weights),
* optimizer state (all Optax leaves, dtypes restored),
* scheduler/global step, epoch, sampler position,
* RNG state (python/numpy/jax),
* best validation metric and full training/model configs,
* tokenizer hash, dataset fingerprint, git commit, environment.

Format: ``safetensors`` for arrays (no pickle), JSON for scalars/metadata, a
sha256 manifest for integrity, and atomic ``tmp → fsync → os.replace`` writes so
an interrupted write can never destroy the previous valid checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import jax
import numpy as np

from ..utils.io_utils import REPO_ROOT, sha256_file, write_json


# --------------------------------------------------------------- tree utils
def tree_to_flat(tree: Any) -> dict[str, np.ndarray]:
    """Flatten a pytree of arrays to ``{"a/b/0": ndarray}`` (arrays only)."""
    flat: dict[str, np.ndarray] = {}
    for path, value in jax.tree_util.tree_flatten_with_path(tree)[0]:
        if value is None:
            continue
        key = "/".join(str(k).replace("'", "") for k in path)
        flat[key] = np.asarray(value)
    return flat


def restore_tree(flat: dict[str, np.ndarray], template: Any) -> Any:
    """Rebuild a pytree shaped like ``template`` from a flat dict."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    paths = ["/".join(str(k).replace("'", "") for k in p)
             for p, _ in jax.tree_util.tree_flatten_with_path(template)[0]]
    new_leaves = []
    for leaf, path in zip(leaves, paths):
        if path not in flat:
            raise KeyError(f"checkpoint is missing parameter {path!r}")
        arr = flat[path]
        if hasattr(leaf, "dtype") and np.asarray(leaf).dtype != arr.dtype:
            arr = arr.astype(np.asarray(leaf).dtype)
        new_leaves.append(jax.numpy.asarray(arr))
    return jax.tree_util.tree_unflatten(treedef, new_leaves)


def _encode_for_safetensors(flat: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    out: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    for k, v in flat.items():
        dtypes[k] = str(v.dtype)
        if v.dtype == np.bool_:
            out[k] = v.astype(np.uint8)
        elif v.dtype == np.float64:
            out[k] = v.astype(np.float32)
        elif v.dtype == np.int32:
            out[k] = v.astype(np.int64)
        elif v.dtype == np.uint32:
            out[k] = v.astype(np.int64)
        else:
            out[k] = v
    return out, dtypes


def _decode_from_safetensors(flat: dict[str, np.ndarray], dtypes: dict[str, str]) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for k, v in flat.items():
        dt = dtypes.get(k, str(v.dtype))
        if dt == "bool":
            out[k] = v.astype(bool)
        else:
            out[k] = v.astype(np.dtype(dt))
    return out


# ------------------------------------------------------------------ environ
def environment_report() -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    try:
        info["jax"] = jax.__version__
        info["jax_backend"] = jax.default_backend()
        info["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception as exc:  # pragma: no cover
        info["jax_error"] = str(exc)
    try:
        import optax

        info["optax"] = optax.__version__
    except Exception:  # pragma: no cover
        pass
    for mod in ("numpy", "tokenizers", "safetensors"):
        try:
            m = __import__(mod)
            info[mod] = getattr(m, "__version__", "unknown")
        except Exception:  # pragma: no cover
            pass
    try:
        info["git_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip()
        info["git_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, capture_output=True,
            text=True, timeout=10).stdout.strip())
    except Exception:
        info["git_commit"] = "unknown"
    return info


# -------------------------------------------------------------- save / load
def save_checkpoint(directory: str | Path, name: str, params: Any, opt_state: Any,
                    meta: dict[str, Any]) -> Path:
    """Atomically write ``name.safetensors`` + ``name.json`` into ``directory``."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    from safetensors.numpy import save_file

    params_flat, params_dtypes = _encode_for_safetensors(tree_to_flat(params))
    opt_flat, opt_dtypes = _encode_for_safetensors(tree_to_flat(opt_state))
    tmp_st = directory / f".{name}.safetensors.tmp"
    final_st = directory / f"{name}.safetensors"
    payload = {f"params/{k}": v for k, v in params_flat.items()}
    payload.update({f"opt/{k}": v for k, v in opt_flat.items()})
    save_file(payload, str(tmp_st), metadata={"format": "tinyme-ckpt-v2"})
    _fsync_file(tmp_st)

    meta = dict(meta)
    meta["checkpoint_format"] = "tinyme-ckpt-v2"
    meta["safetensors_sha256"] = sha256_file(tmp_st)
    meta["param_dtypes"] = params_dtypes
    meta["opt_dtypes"] = opt_dtypes
    meta["saved_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    tmp_meta = directory / f".{name}.json.tmp"
    tmp_meta.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    _fsync_file(tmp_meta)

    os.replace(tmp_st, final_st)          # atomic
    os.replace(tmp_meta, directory / f"{name}.json")
    return final_st


def load_checkpoint(directory: str | Path, name: str, params_template: Any,
                    opt_template: Any | None = None,
                    expected_fingerprints: dict[str, Any] | None = None,
                    require_optimizer: bool = True) -> tuple[Any, Any | None, dict[str, Any]]:
    """Load and verify a checkpoint. Raises on corruption/incompatibility."""
    from safetensors.numpy import load_file

    directory = Path(directory)
    st_path, meta_path = directory / f"{name}.safetensors", directory / f"{name}.json"
    if not st_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"checkpoint {name!r} not found in {directory}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("checkpoint_format") != "tinyme-ckpt-v2":
        raise ValueError(f"unsupported checkpoint format: {meta.get('checkpoint_format')!r}")
    digest = sha256_file(st_path)
    if meta.get("safetensors_sha256") not in (None, digest):
        raise ValueError("checkpoint integrity check failed (sha256 mismatch)")

    raw = load_file(str(st_path))
    params_flat = {k[len("params/"):]: v for k, v in raw.items() if k.startswith("params/")}
    opt_flat = {k[len("opt/"):]: v for k, v in raw.items() if k.startswith("opt/")}
    params_flat = _decode_from_safetensors(params_flat, meta.get("param_dtypes", {}))
    opt_flat = _decode_from_safetensors(opt_flat, meta.get("opt_dtypes", {}))

    if expected_fingerprints:
        for key, expected in expected_fingerprints.items():
            actual = meta.get(key)
            if actual != expected:
                raise ValueError(
                    f"checkpoint incompatibility on {key!r}: checkpoint={actual!r} expected={expected!r}")

    params = restore_tree(params_flat, params_template)
    opt_state = None
    if opt_flat and opt_template is not None:
        opt_state = restore_tree(opt_flat, opt_template)
    elif require_optimizer:
        raise ValueError("checkpoint does not contain optimizer state")
    return params, opt_state, meta


def verify_checkpoint(directory: str | Path, name: str) -> dict[str, Any]:
    """Integrity + completeness report for one checkpoint."""
    directory = Path(directory)
    st_path, meta_path = directory / f"{name}.safetensors", directory / f"{name}.json"
    report: dict[str, Any] = {"name": name, "exists": st_path.exists() and meta_path.exists(),
                              "ok": False}
    if not report["exists"]:
        # A report without an explicit verdict is easy to misread as success:
        # ``None`` is falsy but a caller that only checks the key's presence
        # would proceed.  Always answer the question.
        report["errors"] = ["missing .safetensors or .json"]
        return report
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    digest = sha256_file(st_path)
    report["sha256_match"] = meta.get("safetensors_sha256") in (None, digest)
    report["bytes"] = st_path.stat().st_size
    report["step"] = meta.get("step")
    required = ("model_config_hash", "dataset_fingerprint", "tokenizer_hash", "experiment_id")
    report["missing_metadata"] = [k for k in required if k not in meta]
    report["ok"] = report["sha256_match"] and not report["missing_metadata"]
    return report


def _fsync_file(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def prune_checkpoints(directory: str | Path, keep_last: int = 3,
                      protected: tuple[str, ...] = ("best",)) -> list[str]:
    """Delete old ``step_*`` checkpoints, never ``best``/``latest``."""
    directory = Path(directory)
    victims: list[str] = []
    steps = sorted(directory.glob("step_*.safetensors"), key=lambda p: p.stat().st_mtime)
    while len(steps) > keep_last:
        victim = steps.pop(0)
        if victim.stem.split(".")[0] in protected:
            continue
        victims.append(victim.stem)
        victim.unlink(missing_ok=True)
        (directory / f"{victim.stem}.json").unlink(missing_ok=True)
    return victims


def copy_checkpoint(src_dir: str | Path, dst_dir: str | Path, name: str) -> Path:
    src_dir, dst_dir = Path(src_dir), Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".safetensors", ".json"):
        shutil.copy2(src_dir / f"{name}{suffix}", dst_dir / f"{name}{suffix}")
    return dst_dir / f"{name}.safetensors"


def checkpoint_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]
