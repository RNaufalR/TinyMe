#!/usr/bin/env python3
"""Grounding / no-hallucination contract evidence (audit §61).

Exercises the *shipped* evidence machinery (``EvidenceStore.verify`` through
``AgentRuntime.solve``) with six scripted episodes.  Nothing here is a model
capability claim: the scripted policy replaces the model so the runtime's
verdict can be read in isolation from generation quality.

Cases
-----
1. real citation            -> accepted (grounded, no unsupported ids)
2. forged citation id       -> reported as ``unsupported_citations``
3. quote never returned     -> reported as ``unsupported_quotes``
4. citation with no tool use-> ``citations_without_evidence`` plus the id itself
                               reported unsupported, so an answer cannot cite thin air
5. unknown information      -> the runtime accepts "I don't know" style answers
6. uncited answer           -> grounded, but ``evidence_items=0``: the caller
                               can tell "no claim" from "verified claim"
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent.executor import AgentRuntime, scripted_policy  # noqa: E402
from src.agent.protocol import render_final, render_tool_call  # noqa: E402
from src.utils.io_utils import write_json  # noqa: E402

SYSTEM = "You are TinyMe, a controller for external tools."
OUT = ROOT / "docs" / "audit_evidence" / "grounding_mechanism.json"


def _call(name: str, **arguments) -> str:
    # the runtime requires the closing marker; without it the turn is a
    # protocol error and no grounding verdict is produced
    return render_tool_call(name, arguments)


def _episode(script: list[str], request: str) -> dict:
    runtime = AgentRuntime(max_steps=6, max_seconds=60.0, system_prompt=SYSTEM)
    traj = runtime.solve(request, scripted_policy(script))
    return traj.to_dict()


def _verdict(traj: dict) -> dict:
    return traj.get("grounding") or {}


def main() -> int:
    cases: list[dict] = []

    # 1. a real search + citation of the returned source
    t1 = _episode([_call("search", query="population of jakarta", k=3),
                   render_final("Jakarta has 10,679,951 inhabitants [FACT-988002].")],
                  "What is the population of jakarta?")
    g1 = _verdict(t1)
    cases.append({"case": "real_citation_accepted", "request": t1["request"],
                  "grounding": g1, "final": t1["final"],
                  "pass": bool(g1.get("grounded")) and not g1.get("unsupported_citations")})

    # 2. forged identifier the runtime never issued
    t2 = _episode([_call("search", query="population of jakarta", k=3),
                   render_final("Jakarta has 10,679,951 inhabitants [FACT-000000].")],
                  "What is the population of jakarta? (forged id)")
    g2 = _verdict(t2)
    cases.append({"case": "forged_citation_rejected", "request": t2["request"],
                  "grounding": g2, "final": t2["final"],
                  "pass": "FACT-000000" in (g2.get("unsupported_citations") or [])})

    # 3. a quote that no tool ever returned
    t3 = _episode([_call("search", query="population of jakarta", k=3),
                   render_final('The city "contains 42 million llamas" [FACT-988002].')],
                  "Quote the stored source.")
    g3 = _verdict(t3)
    cases.append({"case": "unsupported_quote_rejected", "request": t3["request"],
                  "grounding": g3, "final": t3["final"],
                  "pass": len(g3.get("unsupported_quotes") or []) >= 1})

    # 4. a citation with zero tool calls in the episode
    t4 = _episode([render_final("Jakarta has 10,679,951 inhabitants [FACT-988002].")],
                  "What is the population of jakarta? (no tool use)")
    g4 = _verdict(t4)
    cases.append({"case": "citation_without_evidence_rejected", "request": t4["request"],
                  "grounding": g4, "final": t4["final"],
                  "pass": "citations_without_evidence" in (g4.get("problems") or [])})

    # 5. information that simply is not in the corpus: the runtime must accept
    #    an honest non-answer rather than force a fabricated citation
    t5 = _episode([render_final("I could not find this in the available sources.")],
                  "What is the population of the planet Zorgon?")
    g5 = _verdict(t5)
    cases.append({"case": "unknown_information_honest", "request": t5["request"],
                  "grounding": g5, "final": t5["final"],
                  "pass": bool(g5.get("grounded")) and not g5.get("unsupported_citations")})

    # 6. an uncited answer: grounded, but with zero evidence items — explicitly
    #    *not* a verified claim
    t6 = _episode([_call("compute", expression="2 + 2"), render_final("Four.")],
                  "What is 2 + 2?")
    g6 = _verdict(t6)
    cases.append({"case": "uncited_answer_not_verified", "request": t6["request"],
                  "grounding": g6, "final": t6["final"],
                  "pass": bool(g6.get("grounded")) and int(g6.get("evidence_items", 0)) == 0})

    payload = {
        "audit": "§61 no-hallucination rule",
        "mechanism": "src/agent/evidence.py::EvidenceStore.verify via AgentRuntime.solve",
        "scripted_policy": "src/agent/executor.py::scripted_policy (the model is replaced by "
                           "fixed outputs so the runtime verdict is measured in isolation)",
        "protocol": "tool calls use <|tool_call|>\\n{json}<|end_tool_call|>; the closing marker "
                    "is required or the turn is a protocol error and no verdict is produced",
        "scope": ("the runtime guarantees that a citation which resolves to no runtime-issued "
                  "evidence is reported; it does NOT guarantee that the model never states "
                  "something false in prose, and a grounded-but-uncited answer is explicitly "
                  "marked evidence_items=0"),
        "cases": cases,
        "passed": sum(1 for c in cases if c["pass"]),
        "total": len(cases),
    }
    write_json(payload, OUT)
    for case in cases:
        print(f"{'PASS' if case['pass'] else 'FAIL'}  {case['case']}")
    print(f"{payload['passed']}/{payload['total']} grounding cases pass -> {OUT}")
    return 0 if payload["passed"] == payload["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
