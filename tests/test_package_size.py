"""Release packaging: measured bytes, checksums, self-containment (audit §17).

These tests run against a packaged ``release/`` directory.  When no release has
been built yet they skip with an explicit reason instead of passing silently.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
RELEASE = REPO_ROOT / "release"
MANIFEST = RELEASE / "manifest.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@pytest.fixture(scope="module")
def manifest() -> dict:
    if not MANIFEST.exists():
        pytest.skip("release/ not packaged yet - run scripts/package_model.py")
    return json.loads(MANIFEST.read_text())


def test_required_release_files_exist(manifest):
    required = ["model_fp32.safetensors", "tokenizer.json", "config.json", "manifest.json",
                "checksums.txt", "evaluation_report.md", "model_comparison.md", "provenance.md",
                "README.md", "inference.py"]
    missing = [name for name in required if not (RELEASE / name).exists()]
    assert not missing, f"release is missing: {missing}"


def test_quantized_variants_are_present_and_smaller(manifest):
    variants = manifest["release"]["variants"]
    assert {"fp32", "fp16", "int8", "int4"} <= set(variants)
    by_bytes = {name: variants[name]["bytes"] for name in ("fp32", "fp16", "int8", "int4")}
    assert by_bytes["fp16"] < by_bytes["fp32"]
    assert by_bytes["int8"] < by_bytes["fp16"]
    assert by_bytes["int4"] < by_bytes["int8"]
    for name, info in variants.items():
        measured = (REPO_ROOT / info["file"]).stat().st_size
        assert measured == info["bytes"], (name, measured, info["bytes"])


def test_measured_bytes_are_under_fifty_megabytes(manifest):
    limit = 50 * 1024 * 1024
    for name, info in manifest["release"]["variants"].items():
        assert info["bytes"] < limit, (name, info["bytes"])
        if name == "fp32":
            assert info["bytes"] == manifest["fp32_bytes"], (info["bytes"], manifest["fp32_bytes"])
            assert manifest["fp32_tensor_bytes"] <= manifest["fp32_bytes"]
    assert manifest["under_50mb"] and all(manifest["under_50mb"].values())


def test_checksums_match_every_listed_file(manifest):
    entries = {}
    for line in (RELEASE / "checksums.txt").read_text().splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        entries[name.strip()] = digest
    assert "manifest.json" in entries
    for name, digest in entries.items():
        path = RELEASE / name
        assert path.exists(), name
        assert _sha256(path) == digest, name
    listed = {p.name for p in RELEASE.iterdir() if p.is_file() and p.name != "checksums.txt"}
    assert listed <= set(entries), sorted(listed - set(entries))


def test_manifest_records_fingerprints_and_provenance(manifest):
    for key in ("experiment_id", "checkpoint", "parameter_count", "model_config_hash",
                "dataset_version", "dataset_fingerprint", "tokenizer_hash", "step"):
        assert manifest.get(key) not in (None, ""), key
    assert manifest["parameter_count"] > 0
    assert (RELEASE / "provenance.md").read_text().strip()


def test_standalone_entrypoint_does_not_import_the_training_repo(manifest):
    source = (RELEASE / "inference.py").read_text()
    forbidden = [r"^\s*import\s+jax", r"^\s*from\s+jax", r"^\s*import\s+torch",
                 r"from\s+src\.", r"import\s+src\."]
    for pattern in forbidden:
        assert not re.search(pattern, source, flags=re.MULTILINE), pattern
    for allowed in ("numpy", "safetensors", "tokenizers"):
        assert allowed in source


def test_config_is_loadable_by_the_engine(manifest):
    from src.model import TinyMeConfig

    payload = json.loads((RELEASE / "config.json").read_text())
    cfg = TinyMeConfig(**{k: v for k, v in payload.items()
                          if k in TinyMeConfig.__dataclass_fields__})
    assert cfg.vocab_size == manifest["model_config"]["vocab_size"]
    assert cfg.parameter_count() == manifest["parameter_count"]


def test_evaluation_report_is_honest_when_not_measured(manifest):
    text = (RELEASE / "evaluation_report.md").read_text()
    assert "NOT MEASURED" in text or "Samples scored" in text
    comparison = (RELEASE / "model_comparison.md").read_text()
    assert "NOT COMPARABLE" in comparison or "NOT MEASURED" in comparison
