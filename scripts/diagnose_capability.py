#!/usr/bin/env python3
"""Per-case failure triage for the held-out tool suite (audit §45, §49, §62).

The tool evaluation reports aggregate metrics.  A failure loop needs the *stage*
at which each case died: protocol, tool choice, arguments, execution, grounding,
answer, recovery.  This script reads the raw per-case rows that
``scripts/evaluate_tools.py`` already wrote and assigns every failing case to the
first stage that failed, then checks the pre-registered gate from ``DECISIONS.md``
(DEC-011).

It never re-grades with a hand-rolled matcher: the aggregate metrics come from
``src.evaluation.tool_cases.summarise`` — the same function the evaluation used —
so the diagnosis cannot disagree with the reported numbers.

Usage::

    python scripts/diagnose_capability.py --experiment EXP-012-TOOL-SFT-V8
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.tool_cases import summarise  # noqa: E402

#: The pre-registered bar (DECISIONS.md DEC-011).  Kept here as data so the
#: diagnosis prints it on every run and cannot be quietly retuned.
GATE = {
    "task_completion": 0.50,
    "tool_syntax_validity": 0.90,
    "argument_validity": 0.90,
    "tool_not_needed_accuracy": 0.80,
    "citation_validity": 0.80,
}

GROUNDED_FAMILIES = {"C", "D", "G"}


def first_failure(row: dict) -> str | None:
    """The first stage of the pipeline this case failed at, or None if it passed."""
    if not row["tool_needed"]:
        if not row.get("tool_not_needed_ok"):
            return "called_tool_when_not_needed"
        if not row.get("answer_correct"):
            return "direct_answer_wrong"
        return None
    if not row.get("tool_syntax_valid", True):
        return "protocol_error"
    if int(row.get("tool_calls") or 0) == 0:
        return "no_tool_call"
    if not row.get("tool_name_correct"):
        return f"wrong_tool({','.join(str(t) for t in (row.get('tools_used') or []))[:40]})"
    if not row.get("args_valid", True):
        return "invalid_arguments"
    if not row.get("exec_success"):
        return "execution_failed"
    if row["family"] in GROUNDED_FAMILIES and not row.get("grounded_final", row.get("grounded")):
        return "not_grounded"
    if row.get("answers_with_citations") and not row.get("citation_valid"):
        return "invalid_citation"
    if not row.get("answer_correct"):
        return "answer_wrong"
    if row["family"] == "G" and not row.get("recovery_success"):
        return "recovery_not_observed"
    if row["family"] == "D" and not row.get("multi_step_success"):
        return "multi_step_incomplete"
    if not row.get("correct"):
        return "other"
    return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="EXP-012-TOOL-SFT-V8")
    ap.add_argument("--variant", default="fp32")
    ap.add_argument("--out", default="docs/audit_evidence/capability_diagnosis.json")
    ap.add_argument("--md", default="docs/CAPABILITY_DIAGNOSIS.md")
    args = ap.parse_args()

    src = ROOT / "experiments" / args.experiment / "evaluation_tools_best.json"
    if not src.exists():
        print(f"missing {src.relative_to(ROOT)} — run scripts/evaluate_tools.py first")
        return 2
    payload = json.loads(src.read_text(encoding="utf-8"))
    variant = (payload.get("variants") or {}).get(args.variant)
    if not variant:
        print(f"variant {args.variant} absent from {src.name}")
        return 2
    rows = variant["cases"]
    metrics = variant.get("metrics") or summarise(rows)

    failures = Counter()
    per_case = []
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        stage = first_failure(row)
        per_case.append({
            "case_id": row["case_id"], "family": row["family"], "tool_needed": row["tool_needed"],
            "correct": bool(row.get("correct")), "first_failure": stage,
            "tools_used": row.get("tools_used"), "stop_reason": row.get("stop_reason"),
            "final": (row.get("final") or "")[:400],
            "expectation": row.get("expectation"), "notes": row.get("notes"),
        })
        by_family[row["family"]].append(row)
        if stage:
            failures[stage.split("(")[0]] += 1

    fam_summary = {}
    for fam, fam_rows in sorted(by_family.items()):
        fam_summary[fam] = {
            "cases": len(fam_rows),
            "passed": sum(1 for r in fam_rows if r.get("correct")),
            "tool_name_accuracy": round(sum(1 for r in fam_rows if r.get("tool_name_correct")) /
                                        len(fam_rows), 4) if fam_rows else None,
            "exec_success": round(sum(1 for r in fam_rows if r.get("exec_success")) /
                                  len(fam_rows), 4) if fam_rows else None,
            "dominant_failure": (Counter(first_failure(r) or "passed" for r in fam_rows)
                                 .most_common(1)[0][0]),
        }

    gate = {}
    for key, threshold in GATE.items():
        value = metrics.get(key)
        gate[key] = {"value": value, "threshold": threshold,
                     "pass": bool(value is not None and value >= threshold)}
    passed = all(g["pass"] for g in gate.values())

    diagnosis = {
        "experiment": args.experiment, "variant": args.variant,
        "source": str(src.relative_to(ROOT)),
        "cases": len(rows), "metrics": metrics,
        "gate": gate, "gate_pass": passed, "gate_definition": "DECISIONS.md DEC-011",
        "failure_stages": dict(failures.most_common()),
        "by_family": fam_summary, "per_case": per_case,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(diagnosis, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Capability diagnosis — {args.experiment} ({args.variant})", "",
             f"Source: `{diagnosis['source']}` · cases: {len(rows)} · "
             f"pre-registered gate: DEC-011 · **gate {'PASS' if passed else 'FAIL'}**", "",
             "## Pre-registered gate", "",
             "| Criterion | Measured | Threshold | Result |", "| :--- | --: | --: | :--- |"]
    for key, g in gate.items():
        lines.append(f"| {key} | {g['value']} | ≥ {g['threshold']} | "
                     f"{'PASS' if g['pass'] else 'FAIL'} |")
    lines += ["", "## Failure stages (first stage that failed)", "",
              "| Stage | Cases |", "| :--- | --: |"]
    for stage, count in failures.most_common():
        lines.append(f"| {stage} | {count} |")
    lines += ["", "## Per family", "",
              "| Family | Cases | Passed | tool_name | exec | Dominant failure |",
              "| :--- | --: | --: | --: | --: | :--- |"]
    for fam, info in fam_summary.items():
        lines.append(f"| {fam} | {info['cases']} | {info['passed']} | "
                     f"{info['tool_name_accuracy']} | {info['exec_success']} | "
                     f"{info['dominant_failure']} |")
    lines += ["", "## Per case", "",
              "| Case | Family | Correct | First failure | Tools used |", "| :--- | :--- | :--- | :--- | :--- |"]
    for row in per_case:
        lines.append(f"| {row['case_id']} | {row['family']} | {row['correct']} | "
                     f"{row['first_failure'] or '—'} | {row['tools_used']} |")
    lines += ["", "Raw: `" + args.out + "`", ""]
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"cases={len(rows)} gate={'PASS' if passed else 'FAIL'}")
    for key, g in gate.items():
        print(f"  {key}: {g['value']} (need ≥ {g['threshold']}) {'ok' if g['pass'] else 'MISS'}")
    print("failure stages:", json.dumps(dict(failures.most_common())))
    print(f"wrote {args.out} and {args.md}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
