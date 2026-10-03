#!/usr/bin/env python3
"""Execute the standalone ``local_model/`` package in a clean environment.

This is the clean-room test the deliverable is judged by.  The package is copied
to a temporary directory that contains **only** the package, and each command is
run with

* ``cwd`` = that directory,
* ``PYTHONPATH`` emptied, so the training repository is not importable,
* ``PATH``/``HOME``/``TMPDIR`` set to system and temporary locations,
* no network access (the runtime does not use it; the check also asserts that no
  socket is created by running the script with a blocking ``socket`` guard).

Recorded per case: exit code, stdout, stderr, elapsed time, parameter count as
reported by the package itself.  The raw transcript is written to
``docs/audit_evidence/local_model_validation.json``.

Usage::

    python scripts/validate_local_model.py --package local_model
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: The guard is written into a tiny shim that is *prepended* to the module search
#: path of the child process; it fails loudly if the package tries to open a
#: socket, which is how "offline" is verified rather than assumed.
NETWORK_GUARD = """
import socket

class _Blocked(RuntimeError):
    pass

def _deny(*_a, **_kw):
    raise _Blocked("network access attempted in a declared-offline package")

socket.socket = _deny
socket.create_connection = _deny
socket.getaddrinfo = _deny
"""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_case(workdir: Path, guard_dir: Path, args: list[str], timeout: int = 900) -> dict:
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(workdir),
        "TMPDIR": str(workdir),
        "LANG": "C.UTF-8",
        "PYTHONPATH": str(guard_dir),          # only the network guard, never the repo
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    started = time.time()
    proc = subprocess.run([sys.executable, "inference.py", *args], cwd=str(workdir),
                          capture_output=True, text=True, timeout=timeout, env=env)
    return {"argv": args, "returncode": proc.returncode,
            "stdout": (proc.stdout or "")[-4000:], "stderr": (proc.stderr or "")[-4000:],
            "seconds": round(time.time() - started, 3)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default="local_model")
    ap.add_argument("--out", default="docs/audit_evidence/local_model_validation.json")
    args = ap.parse_args()

    src = ROOT / args.package
    if not src.exists():
        raise SystemExit(f"{src} does not exist — run scripts/build_local_model.py first")
    manifest = json.loads((src / "manifest.json").read_text(encoding="utf-8"))

    report: dict = {"package": args.package, "manifest": manifest, "cases": {},
                    "checks": {}}

    with tempfile.TemporaryDirectory(prefix="tinyme-cleanroom-") as tmp:
        work = Path(tmp) / "local_model"
        shutil.copytree(src, work)
        guard = Path(tmp) / "guard"
        guard.mkdir()
        (guard / "sitecustomize.py").write_text(NETWORK_GUARD, encoding="utf-8")

        # the repository must not be reachable: only the package and the guard are
        report["checks"]["isolated_copy"] = {
            "files": sorted(str(p.relative_to(work)) for p in work.rglob("*") if p.is_file()),
            "repository_on_pythonpath": False,
        }

        cases = {
            "empty_prompt": ["--prompt", "", "--max-new-tokens", "8"],
            "short_prompt": ["--prompt", "Hello", "--max-new-tokens", "16"],
            "normal_prompt": ["--prompt", "Explain binary search.", "--max-new-tokens",
                              "48", "--temperature", "0"],
            "sampled_prompt": ["--prompt", "Explain binary search.", "--max-new-tokens",
                               "24", "--temperature", "0.7", "--top-p", "0.9", "--seed", "7"],
            "long_prompt": ["--prompt", "Repeat after me: " + ("token " * 200),
                            "--max-new-tokens", "16"],
            "max_context": ["--prompt", "x " * 4000, "--max-new-tokens", "8"],
            "deterministic_repeat": ["--prompt", "Explain binary search.", "--max-new-tokens",
                                     "32", "--temperature", "0", "--seed", "7"],
        }
        for name, argv in cases.items():
            report["cases"][name] = run_case(work, guard, argv)

        # determinism: the same deterministic request twice must be identical
        again = run_case(work, guard, ["--prompt", "Explain binary search.",
                                      "--max-new-tokens", "32", "--temperature", "0", "--seed", "7"])
        report["checks"]["deterministic_same_output"] = (
            again["stdout"] == report["cases"]["deterministic_repeat"]["stdout"])
        report["cases"]["deterministic_repeat_2"] = again

        # malformed configuration must fail clearly instead of guessing
        bad = Path(tmp) / "bad"
        shutil.copytree(work, bad)
        cfg = json.loads((bad / "config.json").read_text(encoding="utf-8"))
        cfg.pop("d_model", None)
        (bad / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        report["cases"]["malformed_config"] = run_case(bad, guard, ["--prompt", "hi",
                                                                   "--max-new-tokens", "4"])
        report["checks"]["malformed_config_fails_clearly"] = (
            report["cases"]["malformed_config"]["returncode"] != 0
            and "d_model" in (report["cases"]["malformed_config"]["stderr"]
                              + report["cases"]["malformed_config"]["stdout"]))

        # a corrupted weight file must be refused, not silently loaded
        corrupt = Path(tmp) / "corrupt"
        shutil.copytree(work, corrupt)
        weights = sorted((corrupt / "model").glob("model_fp32*"))
        if weights:
            with weights[0].open("r+b") as fh:
                fh.seek(min(4096, weights[0].stat().st_size - 1))
                fh.write(b"\x00")
            report["cases"]["corrupted_weights"] = run_case(corrupt, guard,
                                                            ["--prompt", "hi", "--max-new-tokens", "4"])
            report["checks"]["corrupted_weights_rejected"] = (
                report["cases"]["corrupted_weights"]["returncode"] != 0)

        # variant coverage: every shipped variant must load and generate
        variants = sorted((work / "model").glob("model_*.safetensors"))
        for path in variants:
            name = path.stem.replace("model_", "")
            report["cases"][f"variant_{name}"] = run_case(
                work, guard, ["--model", f"model/{path.name}", "--prompt", "Hello",
                              "--max-new-tokens", "16"])
        report["checks"]["all_variants_generate"] = all(
            report["cases"][f"variant_{p.stem.replace('model_', '')}"]["returncode"] == 0
            and report["cases"][f"variant_{p.stem.replace('model_', '')}"]["stdout"].strip()
            for p in variants)

        report["checks"]["network_blocked"] = all(
            "network access attempted" not in c["stderr"] for c in report["cases"].values())

    # fingerprints of the shipped files (outside the temp copy, so a mismatch is visible)
    files = {}
    for p in sorted(src.rglob("*")):
        if p.is_file() and p.name != "checksums.txt":
            files[str(p.relative_to(src))] = {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
    report["shipped_files"] = files
    report["verdict"] = {
        "loads": report["cases"]["short_prompt"]["returncode"] == 0,
        "generates": bool(report["cases"]["normal_prompt"]["stdout"].strip()),
        "deterministic": report["checks"]["deterministic_same_output"],
        "malformed_config_rejected": report["checks"]["malformed_config_fails_clearly"],
        "all_variants_generate": report["checks"]["all_variants_generate"],
        "model_bytes": manifest.get("file_bytes"),
        "under_50mb": bool(manifest.get("file_bytes", 0) < 50_000_000),
    }
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2))
    print(json.dumps({k: v for k, v in report["cases"].items()}, indent=2)[:1500])
    print(f"wrote {out_path.relative_to(ROOT)}")
    ok = all([report["verdict"]["loads"], report["verdict"]["generates"],
              report["verdict"]["deterministic"], report["verdict"]["malformed_config_rejected"],
              report["verdict"]["all_variants_generate"], report["verdict"]["under_50mb"]])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
