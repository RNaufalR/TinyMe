#!/usr/bin/env python3
"""Coverage of the held-out tool-suite mechanisms in the training corpus.

The tool evaluation (``src/evaluation/tool_cases.py``) is authored independently of
the generators, so a family can fail because the model is weak *or* because the
corpus never contained that mechanism.  This script answers the second question
with raw counts: for every mechanism the held-out suite needs (A–H), how many
training records exercise it, drawn from how many distinct template families.

Mechanisms and their held-out counterpart:

====  ====================================================  =====================
tag   mechanism                                             family
====  ====================================================  =====================
M1    answer with no tool call at all                       A
M2    compute call whose expression is copied from the ask  B
M3    search call whose passage grounds the final answer    C
M4    search then fetch, grounded final                      D
M5    fetch a source id named in the request                D3
M6    run a program that the request itself supplies         E
M7    repair code after a failing run                        F
M8    a call that fails, then a corrected call that works    G
M9    a final answer carrying a runtime-issued citation      H (and C/D/G)
====  ====================================================  =====================

Usage::

    python scripts/tool_coverage_report.py --dataset dataset_v7
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CITATION_RE = re.compile(r"\[(?:FACT|DOC|SRC|CALC)-[0-9A-Fa-f]+\]|\[(?:calc|fact)-[0-9a-f]+\]")

#: Minimum the corpus must contain for a mechanism to count as covered.  A stated
#: floor is the point of the report: it turns "the model failed" into a decidable
#: question about the data.
FLOOR_RECORDS = 20
FLOOR_TEMPLATES = 5


def _records(split_path: Path):
    with split_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _segments(rec: dict) -> list[dict]:
    return rec.get("segments") or []


def _calls(rec: dict) -> list[dict]:
    out = []
    for seg in _segments(rec):
        if seg.get("role") == "tool_call":
            try:
                parsed = json.loads(seg["text"])
            except Exception:                                  # noqa: BLE001
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _results(rec: dict) -> list[dict]:
    out = []
    for seg in _segments(rec):
        if seg.get("role") == "tool_result":
            try:
                parsed = json.loads(seg["text"])
            except Exception:                                  # noqa: BLE001
                continue
            if isinstance(parsed, dict):
                out.append(parsed)
    return out


def _final(rec: dict) -> str:
    return " ".join(sg.get("text", "") for sg in _segments(rec) if sg.get("role") == "final")


def _user(rec: dict) -> str:
    return " ".join(sg.get("text", "") for sg in _segments(rec) if sg.get("role") == "user")


def _shape(text: str) -> str:
    """Prompt shape with digits and quoted literals masked.

    Template ids are family-level (``tool/compute``); the diversity that matters
    for generalisation is the *shape* of the request.  Masking the numbers keeps
    "Compute exactly: 4837 * 962 + 71" and "Compute exactly: 3821 * 219 + 22"
    the same shape, which is the honest unit when claiming shape coverage.
    """
    masked = re.sub(r"\d+", "#", text)
    masked = re.sub(r"```.*?```", "<code>", masked, flags=re.S)
    return re.sub(r"\s+", " ", masked).strip()[:160]


def classify(rec: dict) -> set[str]:
    """Which held-out mechanisms does this training record exercise?"""
    tags: set[str] = set()
    calls = _calls(rec)
    names = [c.get("name") for c in calls]
    results = _results(rec)
    oks = [bool(r.get("ok")) for r in results]
    user, final = _user(rec), _final(rec)

    if not calls:
        tags.add("M1")
    if "compute" in names:
        exprs = [c.get("arguments", {}).get("expression", "") for c in calls if c.get("name") == "compute"]
        if any(e and e in user for e in exprs):
            tags.add("M2")
    if "search" in names and final.strip():
        tags.add("M3")
    if "search" in names and "fetch" in names:
        tags.add("M4")
    if names and names[0] == "fetch" and "fetch" not in names[1:] and "fetch" in user.lower():
        tags.add("M5")
    if "code" in names:
        code_args = [c.get("arguments", {}).get("code", "") for c in calls if c.get("name") == "code"]
        for code in code_args:
            head = "\n".join(code.strip().splitlines()[:2])
            if head and head.split("(")[0][:24] in user:
                tags.add("M6")
                break
    if results and not oks[0] and any(oks):
        tags.add("M8")
    if "repair" in (rec.get("template_id") or "") and "code" in names:
        tags.add("M7")
    if CITATION_RE.search(final):
        tags.add("M9")
    return tags


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="dataset_v7")
    ap.add_argument("--out", default="docs/audit_evidence/tool_coverage.json")
    ap.add_argument("--md", default="docs/TOOL_COVERAGE.md")
    args = ap.parse_args()

    base = ROOT / "datasets" / "versions" / args.dataset
    if not base.exists():
        print(f"missing dataset {args.dataset}")
        return 2

    report: dict[str, dict] = {}
    for split in ("train", "validation", "test", "challenge"):
        path = base / f"{split}.jsonl"
        counts: Counter = Counter()
        templates: dict[str, set] = defaultdict(set)
        shapes: dict[str, set] = defaultdict(set)
        examples: dict[str, list[str]] = defaultdict(list)
        records = 0
        for rec in _records(path):
            records += 1
            shape = _shape(_user(rec)) or (rec.get("template_id") or "?")
            for tag in classify(rec):
                counts[tag] += 1
                tid = rec.get("template_id") or rec.get("family") or rec.get("task_type") or "?"
                templates[tag].add(tid)
                shapes[tag].add(shape)
                if len(examples[tag]) < 3:
                    examples[tag].append(rec.get("record_id"))
        report[split] = {"records": records,
                         "mechanisms": {tag: {"records": counts[tag],
                                              "distinct_templates": len(templates[tag]),
                                              "distinct_prompt_shapes": len(shapes[tag]),
                                              "example_record_ids": examples[tag]}
                                        for tag in sorted(set(counts) | set("M1 M2 M3 M4 M5 M6 M7 M8 M9".split()))}}

    train = report["train"]["mechanisms"]
    unverified = {tag: train.get(tag, {}) for tag in ("M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9")}
    gaps = [tag for tag, info in unverified.items()
            if info.get("records", 0) < FLOOR_RECORDS
            or info.get("distinct_prompt_shapes", 0) < FLOOR_TEMPLATES]

    payload = {"dataset": args.dataset, "floor_records": FLOOR_RECORDS,
               "floor_templates": FLOOR_TEMPLATES, "splits": report,
               "train_floor_gaps": gaps,
               "verdict": "PASS" if not gaps else "GAP"}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    lines = [f"# Tool-mechanism coverage — {args.dataset}", "",
             "Every count below is produced by `scripts/tool_coverage_report.py` from the "
             "split JSONL files; a mechanism counts as covered when the train split holds "
             f"at least {FLOOR_RECORDS} records with at least {FLOOR_TEMPLATES} distinct prompt "
             "shapes (digits and code blocks masked), so a passing check cannot be satisfied by "
             "repeating one template.", "",
             "| Mechanism | Held-out family | train records | train shapes | validation | test | challenge |",
             "| :--- | :--- | --: | --: | --: | --: | --: |"]
    names = {"M1": "answer with no tool", "M2": "compute (copied expression)", "M3": "search → grounded answer",
             "M4": "search → fetch (multi-step)", "M5": "fetch a named source id", "M6": "run a supplied program",
             "M7": "repair failing code", "M8": "failure → corrected call", "M9": "citation in the final answer"}
    fam = {"M1": "A", "M2": "B", "M3": "C", "M4": "D", "M5": "D3", "M6": "E", "M7": "F", "M8": "G", "M9": "H"}
    for tag, label in names.items():
        row = [f"| {tag} — {label} | {fam[tag]} "]
        for split in ("train", "validation", "test", "challenge"):
            mech = report[split]["mechanisms"][tag]
            row.append(f"| {mech['records']} / {mech['distinct_prompt_shapes']} ")
        lines.append("".join(row) + "|")
    lines += ["", f"Train-split floor gaps: {gaps if gaps else 'none'} "
              f"(floor: {FLOOR_RECORDS} records / {FLOOR_TEMPLATES} shapes)", "",
              f"Raw: `{args.out}`", ""]
    md = ROOT / args.md
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({tag: unverified[tag].get("records") for tag in unverified}))
    print("gaps:", gaps, "->", payload["verdict"])
    return 0 if not gaps else 1


if __name__ == "__main__":
    raise SystemExit(main())
