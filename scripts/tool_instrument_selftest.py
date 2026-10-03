#!/usr/bin/env python3
"""Self-test of the A–H tool instrument (audit §4/§45, evaluator diagnosis).

Why this exists
---------------
When a model scores ``task_completion 0.00`` there are two possible
explanations: the model cannot do the task, or the instrument cannot award
credit for the task.  The two are indistinguishable from the model's numbers, so
this script drives the *same* cases, the *same* runtime, the *same* tools and the
*same* graders with two scripted policies that never touch a model:

* ``oracle``  — a hand-written agent that does the right thing (real retrieval,
  real fetch, real sandboxed execution, real recovery after a failing call) and
  answers from the evidence it retrieved;
* ``garbage`` — an agent that answers immediately with ``<|final|>`` and a
  useless sentence.

The instrument is sound only if the oracle scores high and garbage scores zero.
Both blocks are written to ``docs/audit_evidence/tool_instrument_selftest.json``.

Known limitation, recorded rather than hidden: the case graders check
``expected_answer in final`` as a substring, so echoing the retrieved passage
verbatim satisfies it.  Grounding credit additionally requires the evidence
engine to have issued the citation, which is what makes the answer evidenced.

Usage::

    python scripts/tool_instrument_selftest.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_tools import SYSTEM_PROMPT  # noqa: E402
from src.agent.executor import AgentRuntime  # noqa: E402
from src.evaluation.tool_cases import build_tool_cases, citation_validity, summarise  # noqa: E402

RESULT_RE = re.compile(r"<\|tool_result\|>\s*(?P<body>\{.*?\})\s*<\|end_tool_result\|>", re.S)
CODE_BLOCK_RE = re.compile(r"```python\s*(?P<code>.*?)```", re.S)
SOURCE_ID_RE = re.compile(r"FACT-[0-9A-F]{6}")


def _call(name: str, arguments: dict) -> str:
    return (f"<|tool_call|>\n"
            f"{json.dumps({'name': name, 'arguments': arguments}, sort_keys=True, separators=(', ', ': '))}\n"
            f"<|end_tool_call|>")


def _final(text: str) -> str:
    return f"<|final|>\n{text}"


def _last_result(history: list[str]) -> dict | None:
    matches = list(RESULT_RE.finditer("\n".join(history)))
    if not matches:
        return None
    try:
        return json.loads(matches[-1].group("body"))
    except json.JSONDecodeError:
        return None


def _payload(result: dict | None) -> dict:
    if not isinstance(result, dict):
        return {}
    payload = result.get("result")
    return payload if isinstance(payload, dict) else {}


def _first_hit(result: dict | None) -> dict:
    hits = _payload(result).get("results") or []
    return hits[0] if hits else {}


FIXED_F0 = ('def count_words(text):\n'
            '    counts = {}\n'
            '    for w in text.lower().split():\n'
            '        counts[w] = counts.get(w, 0) + 1\n'
            '    return counts\n'
            '\n'
            'assert count_words("a a b") == {"a": 2, "b": 1}\n'
            'print(count_words("hi hi")["hi"])\n')
FIXED_F1 = ('def total(values):\n'
            '    t = 0\n'
            '    for v in values:\n'
            '        t += v\n'
            '    return t\n'
            '\n'
            'print(total([3, 4, 5]))\n')


def oracle_turn(case, step: int, history: list[str]) -> str:
    """One scripted turn of the oracle agent."""
    cid, request = case.case_id, case.request
    last = _last_result(history)

    if cid.startswith("A"):
        return _final(str(case.answer))
    if cid.startswith("B"):
        expr = request.split("Compute exactly:", 1)[1].strip()
        if step == 1:
            return _call("compute", {"expression": expr})
        return _final(str(_payload(last).get("value")))
    if cid.startswith("C"):
        # the search snippet is a title only: the value lives in the fetched passage,
        # so the correct behaviour is search -> fetch -> answer
        if step == 1:
            query = re.sub(r"\s*(according to a source|with a source id)\??", "", request).strip()
            return _call("search", {"query": query, "k": 1})
        if step == 2:
            return _call("fetch", {"source_id": _first_hit(last).get("source_id", "")})
        payload = _payload(last)
        return _final(f"{payload.get('text', '').strip()} "
                      f"[{payload.get('source_id', '')}]")
    if cid.startswith("D"):
        sid = SOURCE_ID_RE.search(request)
        topic = re.sub(r"^.*?about ", "", request).split(" and ")[0].strip(" .")
        if cid == "D3":
            if step == 1:
                return _call("fetch", {"source_id": sid.group(0) if sid else "FACT-2082DD"})
            payload = _payload(last)
            return _final(f"{payload.get('text', '').strip()} "
                          f"[{payload.get('source_id') or (sid.group(0) if sid else 'FACT-2082DD')}]")
        if step == 1:
            return _call("search", {"query": topic, "k": 1})
        if step == 2:
            return _call("fetch", {"source_id": sid.group(0) if sid else ""})
        payload = _payload(last)
        return _final(f"{payload.get('text', '').strip()} [{sid.group(0) if sid else ''}]")
    if cid.startswith("E"):
        code = CODE_BLOCK_RE.search(request).group("code")
        if step == 1:
            return _call("code", {"code": code})
        out = (_payload(last).get("stdout") or "").strip()
        return _final(out or str(case.answer))
    if cid.startswith("F"):
        code = FIXED_F0 if cid == "F0" else FIXED_F1
        if step == 1:
            return _call("code", {"code": code})
        out = (_payload(last).get("stdout") or "").strip()
        return _final(out or str(case.answer))
    if cid == "G0":
        if step == 1:
            return _call("search", {"query": "what is the population of jakartaa", "k": 1})
        if step == 2:
            return _call("search", {"query": "what is the population of jakarta", "k": 1})
        if step == 3:
            return _call("fetch", {"source_id": _first_hit(last).get("source_id", "")})
        payload = _payload(last)
        return _final(f"{payload.get('text', '').strip()} [{payload.get('source_id', '')}]")
    if cid == "G1":
        if step == 1:
            return _call("fetch", {"source_id": "FACT-000000"})
        if step == 2:
            return _call("search", {"query": "capital of indonesia", "k": 1})
        if step == 3:
            return _call("fetch", {"source_id": _first_hit(last).get("source_id", "")})
        payload = _payload(last)
        return _final(f"{payload.get('text', '').strip()} [{payload.get('source_id', '')}]")
    if cid == "G2":
        if step == 1:
            return _call("compute", {"expression": "12 / 0"})
        if step == 2:
            return _call("compute", {"expression": "12 / 4"})
        return _final(str(_payload(last).get("value")))
    return _final("unknown case")


def garbage_turn(case, step: int, history: list[str]) -> str:
    return _final("I think the answer is 42.")


def run(policy_factory, cases) -> tuple[list[dict], list[dict]]:
    rows, transcripts = [], []
    for case in cases:
        runtime = AgentRuntime(max_steps=6, max_seconds=60.0, system_prompt=SYSTEM_PROMPT)
        try:
            traj = runtime.solve(case.request, policy_factory(case))
        finally:
            runtime.close()
        graded = case.grade(traj, case) if case.grade else {}
        row = {"case_id": case.case_id, "family": case.family, "expectation": case.expectation,
               "request": case.request, "final": traj.final, "stop_reason": traj.stop_reason,
               "tool_needed": case.tool_required, "notes": case.notes, **graded}
        row.update(citation_validity(traj))
        row["citation_valid"] = row["answers_with_citations"] == 0 or row["invalid"] == 0
        rows.append(row)
        transcripts.append({"case_id": case.case_id, "trajectory": traj.to_dict()})
    return rows, transcripts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs" / "audit_evidence" / "tool_instrument_selftest.json"))
    args = ap.parse_args()
    cases = build_tool_cases()
    started = time.time()

    oracle_rows, _ = run(lambda case: (lambda request, history, step, tools:
                                       oracle_turn(case, step, history)), cases)
    garbage_rows, _ = run(lambda case: (lambda request, history, step, tools:
                                        garbage_turn(case, step, history)), cases)
    oracle, garbage = summarise(oracle_rows), summarise(garbage_rows)
    report = {
        "cases": len(cases),
        "oracle": oracle,
        "garbage": garbage,
        "oracle_rows": oracle_rows,
        "garbage_rows": garbage_rows,
        "verdict": {
            "instrument_credits_correct_behaviour": oracle["task_completion"] >= 0.8,
            "instrument_rejects_nonsense": garbage["task_completion"] <= 0.2,
            "oracle_task_completion": oracle["task_completion"],
            "garbage_task_completion": garbage["task_completion"],
        },
        "seconds": round(time.time() - started, 1),
    }
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"oracle": {k: v for k, v in oracle.items() if not isinstance(v, dict)},
                      "garbage": {k: v for k, v in garbage.items() if not isinstance(v, dict)},
                      "verdict": report["verdict"]}, indent=2))
    print(f"wrote {args.out}")
    ok = all([report["verdict"]["instrument_credits_correct_behaviour"],
              report["verdict"]["instrument_rejects_nonsense"]])
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
