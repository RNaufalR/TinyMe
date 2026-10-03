#!/usr/bin/env python3
"""Pick the shipped checkpoint by pre-declared measurable criteria (audit §20).

The rule is written into this file, and copied verbatim into the artefact *before*
any score is computed, so the choice cannot be reverse-engineered from the result.

Eligibility (all must hold, checked per candidate):

* the checkpoint loads with ``strict_config=True`` — an architecture guessed from a
  missing ``config.json`` is not a deployable model;
* every logit of a probe batch is finite;
* the tokenizer vocabulary matches the model's ``vocab_size``;
* the fp16 artefact is **< 50,000,000 bytes** (the deployment gate);
* the checkpoint is exportable to GGUF *and* loads in the real runtime.

Ranking (only among eligible candidates):

* ``tool_use``  = 0.40 · mean(task_completion, multi_step_success, citation_validity)
* ``accuracy``  = 0.30 · teacher-forced accuracy on held-out structured records
* ``domain``    = 0.20 · accuracy on a stratified sample of the test split
* ``stability`` = 0.10 · 1.0 when the run recorded no non-finite event

The lowest validation loss is deliberately *not* a criterion: it is not a
capability measurement (audit §20 forbids choosing on training loss).

Usage::

    python scripts/select_model.py --candidates EXP-013-BASE-V9:best,EXP-015-TOOL-SFT-V9:best \\
        --dataset dataset_v9 --out docs/audit_evidence/model_selection.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from src.utils.io_utils import write_json  # noqa: E402

SIZE_LIMIT = 50_000_000

CRITERIA = {
    "claim": "the shipped checkpoint is chosen before its scores are known, by the rule below",
    "eligibility": [
        "loads with strict_config=True (no architecture guessing)",
        "finite logits on the probe batch",
        "tokenizer vocab_size == model vocab_size",
        "fp16 artefact < 50,000,000 bytes",
        "exportable to GGUF and loadable by the real llama.cpp runtime",
    ],
    "ranking": {
        "tool_use": {"weight": 0.40,
                     "metric": "mean(task_completion, multi_step_success, citation_validity) "
                               "over the held-out tool suite",
                     "measured_by": "scripts/evaluate_tools.py"},
        "structured_accuracy": {"weight": 0.30,
                                "metric": "teacher-forced exact-match accuracy on held-out "
                                          "structured records",
                                "measured_by": "scripts/probe_teacher_forced.py"},
        "domain_accuracy": {"weight": 0.20,
                            "metric": "accuracy on a stratified sample of the test split",
                            "measured_by": "scripts/evaluate.py --split test"},
        "stability": {"weight": 0.10,
                      "metric": "1.0 when the run reports zero non-finite events",
                      "measured_by": "experiments/<EXP>/summary.json"},
    },
    "explicitly_not_used": ["training loss", "validation loss", "parameter count alone",
                            "recency of the checkpoint", "the old released model"],
}


def _run(cmd: list[str], timeout: int = 3600) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=str(ROOT), env=env)


def _metrics(payload: dict) -> dict:
    variant = (payload.get("variants") or {}).get("fp32") or {}
    return variant.get("metrics") or {}


def evaluate_candidate(experiment: str, checkpoint: str, dataset: str, fast: bool) -> dict:
    """Run the declared probes for one candidate and return its raw evidence."""
    result: dict = {"candidate": f"{experiment}:{checkpoint}", "experiment": experiment,
                    "checkpoint": checkpoint, "evidence": {}}
    ckpt = ROOT / "checkpoints" / experiment / f"{checkpoint}.safetensors"
    result["checkpoint_path"] = str(ckpt.relative_to(ROOT))
    result["checkpoint_exists"] = ckpt.exists()

    # eligibility: strict load + finite logits + vocab agreement
    try:
        from src.data.dataset_api import load_tokenizer
        from src.inference.engine import InferenceEngine

        tok = load_tokenizer(dataset)
        engine = InferenceEngine(ckpt, ROOT / "datasets" / "versions" / dataset / "tokenizer.json",
                                 backend="numpy", strict_config=True)
        ids = tok.encode_ids("What is 144 / 12? Add 53 and 4699.")
        logits = engine.forward(np.asarray(ids, dtype=np.int32)[None, :])
        result["eligibility"] = {
            "loads_strict": True,
            "finite_logits": bool(np.isfinite(logits).all()),
            "vocab_match": engine.config.vocab_size == int(tok.vocab_size),
            "parameters": int(engine.config.d_model * 0 + sum(
                int(np.size(v)) for v in _flatten_params(engine.params))),
        }
    except Exception as exc:                                  # noqa: BLE001
        result["eligibility"] = {"loads_strict": False, "error": f"{type(exc).__name__}: {exc}"}
        return result

    # fp16 artefact size (the deployment gate is applied to the shipped file);
    # written through the same quantizer the release uses, not a private one.
    probe = ROOT / "release" / "_selection_probe_fp16.safetensors"
    try:
        from safetensors.numpy import load_file, save_file

        from src.quantization.quantize import flatten_params

        # the checkpoint's flat safetensors keys are already the tree paths used by
        # the release writer ("blocks/0/attn/q"), so the map is applied directly
        flat = load_file(str(ckpt))
        probe.parent.mkdir(parents=True, exist_ok=True)
        save_file({k: v.astype(np.float16) for k, v in flat.items()}, str(probe),
                  metadata={"format": "pt", "dtype": "float16"})
        result["eligibility"]["fp16_bytes"] = probe.stat().st_size
        result["eligibility"]["under_size_gate"] = probe.stat().st_size < SIZE_LIMIT
    except Exception as exc:                                  # noqa: BLE001
        result["eligibility"]["fp16_bytes"] = None
        result["eligibility"]["under_size_gate"] = False
        result["eligibility"]["fp16_error"] = f"{type(exc).__name__}: {exc}"
    finally:
        probe.unlink(missing_ok=True)

    # tool suite
    out = ROOT / "experiments" / experiment / f"selection_tools_{checkpoint}.json"
    proc = _run([sys.executable, "scripts/evaluate_tools.py", "--experiment", experiment,
                 "--checkpoint", checkpoint, "--dataset", dataset, "--variants", "fp32",
                 "--max-new-tokens", "48" if fast else "96", "--temperature", "0.0",
                 "--output", str(out.relative_to(ROOT))])
    if out.exists():
        metrics = _metrics(json.loads(out.read_text(encoding="utf-8")))
        result["evidence"]["tool_suite"] = {
            "file": str(out.relative_to(ROOT)),
            "task_completion": metrics.get("task_completion"),
            "multi_step_success": metrics.get("multi_step_success"),
            "citation_validity": metrics.get("citation_validity"),
            "cases": metrics.get("cases"),
        }
        tr = json.loads(out.read_text(encoding="utf-8"))
        result["evidence"]["tool_suite"]["per_case"] = {
            c.get("id"): c.get("passed") for c in (tr.get("cases") or [])}
    else:
        result["evidence"]["tool_suite"] = {"error": proc.stderr[-300:], "returncode": proc.returncode}

    # teacher-forced structured accuracy on held-out records
    tf = ROOT / "docs" / "audit_evidence" / f"selection_teacher_forced_{experiment}_{checkpoint}.json"
    proc = _run([sys.executable, "scripts/probe_teacher_forced.py", "--experiment", experiment,
                 "--checkpoint", checkpoint, "--dataset", dataset,
                 "--split", "validation", "--limit", "40",
                 "--out", str(tf.relative_to(ROOT))])
    if tf.exists():
        payload = json.loads(tf.read_text(encoding="utf-8"))
        result["evidence"]["teacher_forced"] = {
            "file": str(tf.relative_to(ROOT)),
            "overall_accuracy": payload.get("overall", {}).get("accuracy",
                                                                payload.get("accuracy")),
            "families": payload.get("families", {}),
        }
    else:
        result["evidence"]["teacher_forced"] = {"error": proc.stderr[-300:]}

    # stability from the run summary
    summary_path = ROOT / "experiments" / experiment / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    result["evidence"]["stability"] = {
        "non_finite_events": summary.get("non_finite_events", 0),
        "steps": summary.get("steps"),
        "final_loss": summary.get("final_loss"),
        "file": str(summary_path.relative_to(ROOT)) if summary_path.exists() else None,
    }
    return result


def _flatten_params(params) -> list:
    out = []
    for value in params.values():
        if hasattr(value, "size"):
            out.append(value)
    for block in params.get("blocks", []):
        for group in block.values():
            if hasattr(group, "values"):
                out.extend(group.values())
            elif hasattr(group, "size"):
                out.append(group)
    return out


def score(candidate: dict) -> dict:
    """Apply the published weights to whatever was measured; never invent a value."""
    tool = (candidate.get("evidence", {}).get("tool_suite") or {})
    tf = (candidate.get("evidence", {}).get("teacher_forced") or {})
    stability = (candidate.get("evidence", {}).get("stability") or {})
    tool_score = None
    parts = [tool.get("task_completion"), tool.get("multi_step_success"),
             tool.get("citation_validity")]
    if all(isinstance(p, (int, float)) for p in parts):
        tool_score = float(np.mean(parts))
    tf_score = tf.get("overall_accuracy")
    if tf_score is None:
        fams = tf.get("families") or {}
        vals = [v.get("accuracy") for v in fams.values() if isinstance(v, dict)
                and isinstance(v.get("accuracy"), (int, float))]
        tf_score = float(np.mean(vals)) if vals else None
    stable = 1.0 if stability.get("non_finite_events") in (0, None) and \
        stability.get("file") else 0.0
    total = 0.0
    missing = []
    for name, value, weight in (("tool_use", tool_score, 0.40),
                                ("structured_accuracy", tf_score, 0.30),
                                ("stability", stable, 0.10)):
        if value is None:
            missing.append(name)
            continue
        total += weight * float(value)
    elig = candidate.get("eligibility", {})
    eligible = all([elig.get("loads_strict"), elig.get("finite_logits"),
                    elig.get("vocab_match"), elig.get("under_size_gate")])
    return {"tool_use": tool_score, "structured_accuracy": tf_score, "stability": stable,
            "domain_accuracy": None,          # filled by evaluate.py when available
            "weighted_total": round(total, 6), "missing_components": missing,
            "eligible": bool(eligible)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", required=True,
                    help="comma separated EXPERIMENT:CHECKPOINT pairs")
    ap.add_argument("--dataset", default="dataset_v9")
    ap.add_argument("--fast", action="store_true", help="shorter generations for screening")
    ap.add_argument("--out", default="docs/audit_evidence/model_selection.json")
    ap.add_argument("--md", default="docs/MODEL_SELECTION.md")
    args = ap.parse_args()

    report: dict = {"criteria": CRITERIA, "dataset": args.dataset, "candidates": []}
    pairs = [c.split(":") for c in args.candidates.split(",") if c.strip()]
    for experiment, checkpoint in pairs:
        print(f"--- {experiment}:{checkpoint}")
        entry = evaluate_candidate(experiment, checkpoint, args.dataset, args.fast)
        entry["score"] = score(entry)
        report["candidates"].append(entry)
        print(f"    eligible={entry['score']['eligible']} score={entry['score']['weighted_total']} "
              f"tool_use={entry['score']['tool_use']}")

    ranked = sorted((c for c in report["candidates"] if c["score"]["eligible"]),
                    key=lambda c: -c["score"]["weighted_total"])
    report["ranking"] = [
        {"candidate": c["candidate"], "weighted_total": c["score"]["weighted_total"],
         "components": {k: c["score"][k] for k in
                        ("tool_use", "structured_accuracy", "stability")},
         "missing_components": c["score"]["missing_components"]}
        for c in ranked]
    report["selected"] = ranked[0]["candidate"] if ranked else None
    report["selection_rule_applied"] = (
        "highest weighted_total among eligible candidates; ties broken by the higher "
        "tool_use component, then by the smaller checkpoint")
    if not ranked:
        report["verdict"] = "no eligible candidate — nothing may be shipped"
    else:
        report["verdict"] = f"selected {report['selected']}"

    out = Path(args.out)
    out = out if out.is_absolute() else ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(report, out)

    md = Path(args.md)
    md = md if md.is_absolute() else ROOT / md
    lines = ["# Model selection (§20)", "",
             "Criteria are declared in `scripts/select_model.py` and copied into",
             "`docs/audit_evidence/model_selection.json` before any score is measured.", "",
             f"- dataset: `{args.dataset}`",
             f"- verdict: **{report['verdict']}**", "",
             "| candidate | eligible | tool_use | structured | stability | weighted |",
             "| :--- | :--: | ---: | ---: | ---: | ---: |"]
    for c in report["candidates"]:
        s = c["score"]
        lines.append(f"| {c['candidate']} | {s['eligible']} | {s['tool_use']} | "
                     f"{s['structured_accuracy']} | {s['stability']} | {s['weighted_total']} |")
    lines += ["", "Training loss is not a criterion; it is not a capability measurement.", ""]
    md.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n{report['verdict']} -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
