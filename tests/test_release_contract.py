"""Release contract and clean-environment gate (audit §12, §13, §14).

These tests are artefact-gated on purpose: they exercise the *packaged release*
(``release/``), never the training tree.  When no release has been produced they
skip with an explicit reason (a skip is a statement that the evidence does not
exist yet, never a silent pass).
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RELEASE = ROOT / "release"
REQUIRED = [
    "model_fp32.safetensors", "model_fp16.safetensors", "model_int8.safetensors",
    "model_int4.safetensors", "tokenizer.json", "config.json", "manifest.json",
    "checksums.txt", "evaluation_report.md", "model_comparison.md", "provenance.md",
    "README.md", "inference.py",
]

pytestmark = pytest.mark.skipif(
    not (RELEASE / "manifest.json").exists(),
    reason="release/ not packaged yet — run scripts/package_model.py")


def _manifest() -> dict:
    return json.loads((RELEASE / "manifest.json").read_text(encoding="utf-8"))


def test_release_contains_exactly_the_declared_files():
    present = {p.name for p in RELEASE.iterdir() if p.is_file()}
    missing = [name for name in REQUIRED if name not in present]
    assert not missing, f"release is missing {missing}"
    unexpected = sorted(present - set(REQUIRED))
    assert not unexpected, f"release has undeclared files: {unexpected}"


def test_no_stale_or_hidden_artifacts():
    for path in RELEASE.rglob("*"):
        assert ".tmp" not in path.name and not path.name.startswith("~"), f"stale artifact {path}"
        assert path.suffix != ".pyc", f"stale artifact {path}"
    assert not (RELEASE / "__pycache__").exists()


def test_checksums_cover_every_file_and_match():
    import hashlib
    lines = [l for l in (RELEASE / "checksums.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    recorded = {}
    for line in lines:
        digest, _, name = line.partition("  ")
        recorded[name.strip()] = digest.strip()
    for name in REQUIRED:
        if name == "checksums.txt":
            continue
        assert name in recorded, f"{name} is not covered by checksums.txt"
        actual = hashlib.sha256((RELEASE / name).read_bytes()).hexdigest()
        assert actual == recorded[name], f"checksum mismatch for {name}"
    assert set(recorded) - {"checksums.txt"} == set(REQUIRED) - {"checksums.txt"}


def test_inference_entrypoint_has_no_repository_dependency():
    source = (RELEASE / "inference.py").read_text(encoding="utf-8")
    for forbidden in ("from src", "import src", "checkpoints", "jax", "optax", "datasets/"):
        assert forbidden not in source, f"inference.py still references {forbidden!r}"


def test_model_files_stay_under_the_size_gate():
    total = sum((RELEASE / name).stat().st_size for name in REQUIRED)
    assert total < 50 * 1000 * 1000, f"release totals {total} bytes"


def test_clean_environment_release_gate(tmp_path):
    """Copy ONLY the release, install nothing, run inference.py for real.

    The subprocess runs with ``PYTHONPATH`` cleared and from an unrelated
    directory, so any hidden dependency on the training tree fails loudly.
    """
    sandbox = tmp_path / "clean"
    sandbox.mkdir()
    for name in REQUIRED:
        shutil.copy2(RELEASE / name, sandbox / name)
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path),
           "PYTHONPATH": "", "PYTHONDONTWRITEBYTECODE": "1"}
    prompt = "Compute exactly: 12 * 12"
    proc = subprocess.run([sys.executable, "inference.py", "--prompt", prompt,
                           "--max-new-tokens", "16"],
                          cwd=sandbox, env=env, capture_output=True, text=True, timeout=600)
    assert proc.returncode == 0, f"clean-release inference failed:\n{proc.stderr[-2000:]}"
    assert proc.stdout.strip(), "inference.py produced no output"
    # the entrypoint must also refuse to silently fall back to a default config
    assert "config" not in proc.stderr.lower() or "error" not in proc.stderr.lower()
