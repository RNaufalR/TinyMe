#!/usr/bin/env python3
"""Recorded sandbox security evidence (audit §8, §9).

Runs the adversarial probe matrix against the real sandbox backend and writes a
human-auditable table to ``docs/audit_evidence/sandbox_escape_suite.out.txt``.

The report is deliberately honest about the isolation level:

* ``blocked``            — the action did not happen (enforced by kernel limits)
* ``contained``          — the action happened but stayed inside the run workspace
* ``allowed``            — the action is permitted *by design* (e.g. subprocess spawn)
* ``visible``            — shared host state that is intentionally readable

Run:  python scripts/audit_sandbox.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.sandbox import detected_summary, run_escape_suite  # noqa: E402

OUT = ROOT / "docs" / "audit_evidence" / "sandbox_escape_suite.out.txt"
JSON_OUT = ROOT / "docs" / "audit_evidence" / "sandbox_escape_suite.json"


def main() -> int:
    started = time.time()
    caps = detected_summary()
    probes = run_escape_suite()
    summary = probes.pop("_summary")

    lines = ["# Sandbox escape suite — recorded evidence", "",
             f"date: {time.strftime('%Y-%m-%d %H:%M:%S')} | wall: {time.time() - started:.2f}s",
             f"isolation capabilities: {json.dumps(caps)}",
             f"honest label: {caps.get('label') or caps.get('isolation') or 'namespace'}",
             "",
             f"{'case':26s} {'expectation':12s} {'observed':12s} {'passed':7s} {'exit':>5s}  detail",
             ]
    for name, p in probes.items():
        if not isinstance(p, dict):
            continue
        detail = (p.get("stdout") or p.get("stderr_tail") or "").replace("\n", " ")[:70]
        lines.append(f"{name:26s} {p['expectation']:12s} {p['observed']:12s} "
                     f"{str(p['passed']):7s} {p['exit_code']:>5}  {detail}")
    lines += ["", f"summary: {json.dumps(summary)}"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Raw per-case records next to the human-readable table, so the numbers in
    # docs/SANDBOX.md can be re-derived from an artefact rather than re-read.
    raw = {"date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "wall_seconds": round(time.time() - started, 3),
           "isolation": caps, "summary": summary, "probes": probes}
    JSON_OUT.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if summary["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
