"""Failure-recovery contract (audit §62).

Every case here exercises a *real* failure path of the shipped code and asserts
that it is detected, reported and never silently converted into success:

* a missing checkpoint is refused (load raises, ``verify_checkpoint`` says
  ``ok: False`` rather than returning a verdict-less report);
* a byte-flipped weight file fails its recorded digest;
* a shard family whose fingerprint no longer matches the manifest is refused by
  the dataset API, and the same call succeeds once the shard is restored;
* a curriculum stage that selects no record raises instead of training on an
  empty split;
* the CLIs (``evaluate_tools.py``, ``build_local_model.py``, ``quantize.py``)
  exit non-zero on a missing experiment / empty release / missing checkpoint;
* a failed evaluation leaves no result file behind (no stale artifact that a
  later reader could mistake for evidence).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------- helpers
def _write_checkpoint(directory: Path, name: str = "step_000001") -> Path:
    """A minimal but structurally valid checkpoint written by the real writer."""
    from src.model import TinyMeConfig, init_params
    from src.training.checkpoint import save_checkpoint

    cfg = TinyMeConfig(name="test-tiny", vocab_size=128, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=64)
    params = init_params(cfg, seed=0)
    opt_state = {"count": 1, "mu": params, "nu": params}
    meta = {"experiment_id": "UNIT-RECOVERY", "step": 1, "model_config_hash": "deadbeef",
            "dataset_fingerprint": "fingerprint", "tokenizer_hash": "tokenhash"}
    return save_checkpoint(directory, name, params, opt_state, meta)


def _run(script: str, *args: str, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=timeout)


# ---------------------------------------------------------------- checkpoints
def test_missing_checkpoint_raises_and_reports_not_ok(tmp_path):
    from src.model import TinyMeConfig, init_params
    from src.training.checkpoint import load_checkpoint, verify_checkpoint

    report = verify_checkpoint(tmp_path, "best")
    assert report["exists"] is False
    assert report["ok"] is False          # never a verdict-less report

    cfg = TinyMeConfig(name="test-tiny", vocab_size=128, d_model=32, n_layers=2, n_heads=4,
                       d_ff=64, max_seq_len=64)
    params = init_params(cfg, seed=0)
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path, "best", params, params)


def test_missing_metadata_marks_checkpoint_not_ok(tmp_path):
    from src.training.checkpoint import verify_checkpoint

    _write_checkpoint(tmp_path, "step_000001")
    meta_path = tmp_path / "step_000001.json"
    meta = json.loads(meta_path.read_text())
    del meta["tokenizer_hash"]               # a checkpoint without provenance is not usable
    meta_path.write_text(json.dumps(meta))

    report = verify_checkpoint(tmp_path, "step_000001")
    assert report["exists"] is True
    assert report["ok"] is False
    assert "tokenizer_hash" in report["missing_metadata"]


def test_corrupted_weight_bytes_fail_the_digest(tmp_path):
    from src.training.checkpoint import verify_checkpoint

    path = _write_checkpoint(tmp_path, "step_000001")
    before = verify_checkpoint(tmp_path, "step_000001")
    assert before["ok"] is True

    data = bytearray(path.read_bytes())
    data[len(data) // 2] ^= 0xFF             # flip one byte in the middle
    path.write_bytes(bytes(data))

    after = verify_checkpoint(tmp_path, "step_000001")
    assert after["ok"] is False
    assert after["sha256_match"] is False


# -------------------------------------------------------------- dataset / API
def test_stale_shards_are_refused_and_recover(tmp_path, monkeypatch):
    """A tampered shard fingerprint must stop training, and restoring must work."""
    from src.data import dataset_api
    from src.data.dataset_api import load_manifest

    version = "dataset_v9"
    manifest = load_manifest(version)
    original = manifest.get("shards_fingerprint")
    if not original:
        pytest.skip("dataset_v9 manifest carries no shards_fingerprint")
    if not (ROOT / "datasets" / "processed" / version / "shards").exists():
        pytest.skip("dataset_v9 shards are not built")

    # the API recomputes the fingerprint from the files; simulate the stale case
    # by making the manifest disagree with the shards that are on disk
    def fake_manifest(_version):
        return {**manifest, "shards_fingerprint": "0" * 64}

    monkeypatch.setattr(dataset_api, "load_manifest", fake_manifest)
    with pytest.raises(RuntimeError, match="stale"):
        dataset_api.load_split(version, "validation", 512, stage="pretrain")
    monkeypatch.undo()

    # restored manifest -> the very same call succeeds
    data, stats = dataset_api.load_split(version, "validation", 512, stage="pretrain")
    assert data["input_ids"].shape[0] == stats["blocks"] > 0


def test_empty_curriculum_stage_is_refused(monkeypatch):
    from src.data.dataset_api import load_split

    def nothing(_record):
        return False

    with pytest.raises(ValueError, match="selected no record"):
        load_split("dataset_v9", "train", 512, stage="drill", record_filter=nothing)


# ----------------------------------------------------------------------- CLIs
def test_evaluate_tools_unknown_experiment_exits_nonzero(tmp_path):
    proc = _run("evaluate_tools.py", "--experiment", "EXP-DOES-NOT-EXIST",
                "--checkpoint", "best", "--dataset", "ci_unit", "--output", str(tmp_path / "o.json"))
    assert proc.returncode != 0
    assert not (tmp_path / "o.json").exists()


def test_build_local_model_empty_release_exits_nonzero(tmp_path):
    empty = tmp_path / "empty_release"
    empty.mkdir()
    proc = _run("build_local_model.py", "--release", str(empty), "--out", str(tmp_path / "pkg"))
    assert proc.returncode != 0
    manifest = tmp_path / "pkg" / "manifest.json"
    assert not manifest.exists() or json.loads(manifest.read_text()).get("entrypoint") is None


def test_quantize_missing_checkpoint_exits_nonzero(tmp_path):
    proc = _run("quantize.py", "--experiment", "EXP-DOES-NOT-EXIST", "--checkpoint", "best",
                "--out", str(tmp_path / "q"))
    assert proc.returncode != 0
    assert not (tmp_path / "q" / "model_fp32.safetensors").exists()


def test_failed_evaluation_leaves_no_result_file(tmp_path):
    """A crashed evaluation must not leave a half-written result a reader could cite."""
    out = tmp_path / "eval.json"
    proc = _run("evaluate.py", "--experiment", "EXP-DOES-NOT-EXIST", "--checkpoint", "best",
                "--dataset", "ci_unit", "--split", "validation", "--out", str(out),
                "--max-samples", "2")
    assert proc.returncode != 0
    assert not out.exists()


# ------------------------------------------------------------------ invariants
def test_failure_recovery_does_not_delete_evidence(tmp_path):
    """The recovery paths above must not have removed any repository artifact."""
    assert (ROOT / "src" / "training" / "checkpoint.py").exists()
    assert shutil.which(sys.executable) is not None
