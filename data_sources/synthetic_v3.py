"""Capability-focused synthetic generators (v3, 2026-10-03).

Why a v3 module instead of editing :mod:`data_sources.synthetic_v2`
-------------------------------------------------------------------
``dataset_v3`` is a published artefact and stays reproducible from
``synthetic_v2``; the corrective generators therefore live in a new module and
``corpus_v2.build_corpus`` selects between them.  Nothing in v2 is deleted.

What the measurement said (evidence under ``docs/audit_evidence/``)
-------------------------------------------------------------------
The tool-use block of ``dataset_v3`` contained only ~410 tool-call supervision
segments in total (~42 ``compute`` calls, 8 ``search→fetch`` chains, 8 error
recovery chains after splitting).  The released model reproduced 60/60 memorised
training templates but achieved **0/10** operand copying on its own training
``tool/compute`` records: fluent protocol scaffolding with invented operands.
A behaviour cannot be learned from a handful of instances, so v3 scales every
audited workflow by an order of magnitude and adds an explicit transcription
curriculum.

Design rules (unchanged, per the audit)
---------------------------------------
* generator → independent solve → independent verify → accept/reject;
* tool calls use the runtime's exact argument names and result envelope;
* an answer is never copied from the prompt without being recomputed, read from
  the shipped retrieval index, or executed;
* ``group_id`` is ``"<template>#<phrasing-index>"`` so the group-aware splitter
  holds out *phrasings*, not whole workflows (holding out a whole workflow would
  mean the model never trains on that behaviour at all).

The copy invariant
------------------
Every user turn that requests a tool action contains the argument value as a
verbatim character span, and every final answer copies a verbatim span out of
the rendered tool result.  Together with the tok-v3 individual-digit tokenizer
this makes both copies *token-identity* copies rather than re-tokenisations.
"""
from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

from src.data.records import Segment, make_segment_record

from .synthetic_v2 import (  # noqa: F401  (re-exported helpers keep one source of truth)
    EVIDENCE_DB, TOOL_SPECS, _BUG_PATTERNS, _FACT_KEYS, _INSTANCE_MAKERS, _SPEC_LIBRARY,
    _WORD_POOL, _mock_code, _mock_compute, _mock_fetch, _mock_search, _rand_int_list,
    _rand_sentence, _tool_trajectory as _v2_tool_trajectory, _verified_solution, run_python,
)

TOOL_SYSTEM = ("You are TinyMe, a controller for external tools. Use a tool only when needed, "
               "then answer from the evidence.")

# ---------------------------------------------------------------- phrase pools
#: Every pool is long (8-16 phrasings) on purpose: repeated target text is a
#: defect (DEC-008), and a long pool means the splitter's phrasing holdout costs
#: each family only ~1/N of its supervision.
_COMPUTE_PHRASINGS = [
    "Compute exactly: {e}",
    "What is {e}? Use a tool and give the exact value.",
    "Evaluate exactly: {e}",
    "Work out {e} exactly and report the value.",
    "Give me the exact value of {e} using a tool.",
    "Calculate {e} exactly.",
    "I need the exact result of {e}. Use a tool.",
    "Use the compute tool to evaluate {e} exactly.",
    "Please compute {e} exactly and tell me the value.",
    "Evaluate the expression {e} exactly with a tool.",
    "What does {e} equal? Give the exact value.",
    "Compute exactly: {e} (report the value only).",
]

_SEARCH_PHRASINGS = [
    "What is the current {q}?",
    "Look up {q} and answer with a citation.",
    "According to a source, what is the {q}?",
    "Tell me the {q} and cite where it comes from.",
    "Which source supports the {q}?",
    "Find a source for the {q} and quote it.",
    "Search the local corpus for the {q} and answer with a citation.",
    "Give me the {q} with a source id.",
    "What is the {q}? Cite the source you used.",
    "Retrieve a source about the {q} and answer with its id.",
    "Use the search tool for the {q} and quote the result.",
    "Answer with a citation: what is the {q}?",
]

_FETCH_PHRASINGS = [
    "Read the stored source {sid} about the {q} and quote its key sentence.",
    "Fetch the passage with source id {sid} and summarise it.",
    "Use the stored source {sid} for the {q} and report the key sentence.",
    "Retrieve source {sid} and quote the sentence it stores.",
    "Open the stored passage {sid} and give me its key sentence.",
    "Fetch source_id {sid} and report what it says about the {q}.",
    "Quote the stored sentence in {sid} about the {q}.",
    "Read source {sid} from the index and summarise it.",
]

_FETCH_FIRST_PHRASINGS = [
    "Read the detailed passage about the {q} and summarise it with a citation.",
    "Use the stored source for the {q} and quote the key sentence.",
    "Get the full passage about the {q} from the index and cite it.",
    "Find then read the stored passage about the {q} and quote it with a citation.",
    "Retrieve the detailed source for the {q} and report its key sentence.",
]

_CODE_RUN_PHRASINGS = [
    "Run this program in the sandbox and report the result:\n```python\n{c}\n```",
    "Execute this code and tell me what it prints:\n```python\n{c}\n```",
    "Use the code tool on this program and report its output:\n```python\n{c}\n```",
    "Run the following program in the sandbox and report what it prints:\n```python\n{c}\n```",
    "Please execute this snippet and report the output:\n```python\n{c}\n```",
]

_CODE_REPAIR_PHRASINGS = [
    "The tests fail for this code:\n```python\n{c}\n```\nFix it, run the tests in the sandbox, "
    "and report the result.",
    "This program fails its assertions:\n```python\n{c}\n```\nRepair it and run it with the code tool.",
    "Debug this program so the assertions pass, then execute it:\n```python\n{c}\n```",
    "This code does not satisfy its tests:\n```python\n{c}\n```\nCorrect it and run the tests.",
    "Repair the broken implementation below and verify it by running the tests:\n```python\n{c}\n```",
]

_NO_TOOL_ADD = ["What is {a} + {b}?", "Add {a} and {b}.", "What is the sum of {a} and {b}?",
                "Compute {a} + {b} without a tool.", "What do you get when you add {a} and {b}?",
                "{a} + {b} = ?"]
_NO_TOOL_SUB = ["What is {a} - {b}?", "Subtract {b} from {a}.", "What is {a} minus {b}?",
                "Compute {a} - {b} without a tool.", "What is {a} less {b}?", "{a} - {b} = ?"]
_NO_TOOL_MUL = ["What is {a} * {b}?", "Multiply {a} by {b}.", "What is the product of {a} and {b}?",
                "Compute {a} * {b} without a tool.", "What is {a} times {b}?", "{a} * {b} = ?"]
_NO_TOOL_UPPER = ["Capitalise the word '{w}'.", "Uppercase the word '{w}'.",
                  "Write the word '{w}' in capital letters.", "Convert '{w}' to uppercase.",
                  "Make '{w}' all caps."]
_NO_TOOL_REVERSE = ["Reverse the string '{w}'.", "Reverse the word '{w}'.",
                    "What is '{w}' reversed?", "Write '{w}' backwards.",
                    "Reverse the characters of '{w}'."]

_FINAL_STYLES = ["{v}", "The answer is {v}.", "{v}.", "Result: {v}"]


def _pick(rng: random.Random, pool: list[str], **kw: Any) -> str:
    return rng.choice(pool).format(**kw)


def _pick_idx(rng: random.Random, pool: list[str], **kw: Any) -> tuple[str, int]:
    """Pick a phrasing and report its index (the index becomes part of group_id)."""
    idx = rng.randrange(len(pool))
    return pool[idx].format(**kw), idx


def _gid(template: str, idx: int) -> str:
    return f"{template}#{idx}"


def _tool_trajectory(*args: Any, group_id: str | None = None, **kw: Any) -> Any:
    """v2 trajectory builder plus an explicit ``group_id``."""
    rec = _v2_tool_trajectory(*args, **kw)
    if group_id:
        rec.group_id = group_id
    return rec


# --------------------------------------------------------------- transcription
_SYSTEM_COPY = "You are TinyMe, a precise assistant. Answer with the requested value only."
_COPY_POOLS = [
    ["The recorded identifier is {v}. Repeat it exactly.",
     "Reproduce the token {v} character for character.",
     "Copy the value {v} as it appears.",
     "Echo the identifier {v} without changing anything.",
     "Repeat the code {v} exactly.",
     "Return the literal {v} unchanged.",
     "Write out {v} verbatim.",
     "Copy {v} and nothing else."],
    ["The serial is {v}. Return the serial only.",
     "Copy the code {v} exactly.",
     "Write out {v} verbatim.",
     "Repeat the tag {v} character by character.",
     "Echo {v} exactly as printed.",
     "Return the string {v} unchanged."],
    ["A measurement recorded {a} units. Report the measurement exactly.",
     "The log line contains the value {a}. Repeat that value.",
     "The counter reads {a}. State the counter value.",
     "A reading of {a} was stored. Reproduce the reading.",
     "The sensor printed {a}. Copy that number.",
     "Repeat the recorded number {a} exactly."],
    ["The tag is {v}. Repeat the tag exactly.",
     "Copy the tag {v} and nothing else.",
     "Return the tag {v} verbatim.",
     "Echo the label {v} unchanged."],
]


def gen_copy_span(rng: random.Random, n: int) -> list[Any]:
    """Verbatim-transcription curriculum (the skill tool use depends on).

    Disjoint system prompt and phrasings from every tool family, so this block
    cannot contradict the tool/no-tool decision.  Each answer is asserted to be a
    verbatim span of the user turn, so a record whose answer is not literally
    present is rejected at generation time.
    """
    out: list[Any] = []
    for i in range(n):
        kind = i % 4
        pool = _COPY_POOLS[kind]
        if kind == 0:
            span = f"{rng.randint(1000, 99999)}-{rng.randint(1000, 99999)}"
            user, ph = _pick_idx(rng, pool, v=span)
        elif kind == 1:
            span = rng.choice(_WORD_POOL) + "-" + str(rng.randint(100, 999))
            user, ph = _pick_idx(rng, pool, v=span)
        elif kind == 2:
            span = str(rng.randint(10000, 999999))
            user, ph = _pick_idx(rng, pool, a=span)
        else:
            span = rng.choice(_WORD_POOL)
            user, ph = _pick_idx(rng, pool, v=span)
        assert span in user, (span, user)
        template = f"copy/span/kind{kind}"
        segs = [Segment("system", _SYSTEM_COPY, target=False),
                Segment("user", user, target=False),
                Segment("assistant", f"<|final|>\n{span}", target=True)]
        out.append(make_segment_record(segments=segs, category="instruction", source="synthetic",
                                       source_id=f"syn/copy/{kind}/{i}", task_type="instruction",
                                       template_id=template, group_id=_gid(template, ph),
                                       verified=True, verifier="verbatim_containment_check",
                                       answer=span))
    return out


# --------------------------------------------------------------- arithmetic core
def _expression(shape: str, a: int, b: int, c: int, rng: random.Random) -> tuple[str, Any]:
    """Build one arithmetic expression and its independently computed value."""
    if shape == "mul_add":
        expr, value = f"{a} * {b} + {c}", a * b + c
    elif shape == "add_mul":
        expr, value = f"{a} + {b} * {c}", a + b * c
    elif shape == "mul":
        expr, value = f"{a} * {b}", a * b
    elif shape == "add_sub":
        expr, value = f"{a} + {b} - {c}", a + b - c
    elif shape == "paren":
        expr, value = f"({a} + {b}) * {c}", (a + b) * c
    else:
        divisor = rng.randint(7, 29)
        expr, value = f"{a * divisor} // {divisor}", (a * divisor) // divisor
    assert eval(expr) == value, (expr, value)          # independent recomputation
    return expr, value


def instance_examples_v3(spec: dict[str, Any], rng: random.Random,
                         k: int = 2) -> tuple[str, str]:
    """Concrete instances whose values are produced by running the reference.

    Fixes a real defect in the v2 helper: ``print(value)`` renders a *string*
    result without quotes, so ``assert caesar('golf', 25) == fnke`` was emitted
    as a test (invalid Python, and it silently poisoned the stored ``tests``
    metadata of every string-returning spec -- see the audit defect table).
    Here the harness prints ``repr(value)``, so the emitted assertion is
    executable for strings, dicts, lists, bools and numbers alike, and it is
    re-executed before being accepted.
    """
    maker = _INSTANCE_MAKERS.get(spec["name"])
    if maker is None:
        return "", ""
    calls = maker(rng, k)
    harness = spec["solution"] + "\n" + "\n".join(f"print(repr({call}))" for call in calls)
    ok, out = run_python(harness)
    assert ok, f"instance harness failed for {spec['name']}: {out[:200]}"
    values = out.strip().splitlines()
    assert len(values) == len(calls), (spec["name"], values, calls)
    display = "\n".join(f"- {call}  ->  {value}" for call, value in zip(calls, values))
    tests = "\n".join(f"assert {call} == {value}" for call, value in zip(calls, values))
    good, txt = run_python(spec["solution"], tests)
    assert good, (spec["name"], tests, txt[:200])
    return display, tests


def _code_id(code: str) -> str:
    from src.tools.code import code_citation
    return code_citation(code)


def _buggy_program(rng: random.Random, *, seed_i: int) -> tuple[dict[str, Any], str]:
    """Return (spec, buggy_source) where the injected defect really fails tests."""
    for attempt in range(200):
        spec = _SPEC_LIBRARY[(seed_i + attempt) % len(_SPEC_LIBRARY)]
        candidates = [(name, fn) for name, fn, names in _BUG_PATTERNS if spec["name"] in names]
        if not candidates:
            continue
        _name, fn = rng.choice(candidates)
        buggy = fn(spec["solution"])
        if buggy == spec["solution"]:
            continue
        fails, _ = run_python(buggy, spec["tests"])
        if fails:
            continue
        return spec, buggy
    raise AssertionError("no failing bug pattern available")


def _typo(text: str, rng: random.Random) -> str:
    """Mutate one alphanumeric character (never a space) and guarantee a change."""
    positions = [k for k in range(1, len(text) - 1) if text[k].isalnum()]
    for _ in range(64):
        pos = rng.choice(positions)
        mode = rng.choice(["drop", "swap", "dup"])
        if mode == "drop":
            bad = text[:pos] + text[pos + 1:]
        elif mode == "swap":
            if not text[pos + 1].isalnum() or text[pos] == text[pos + 1]:
                continue
            bad = text[:pos] + text[pos + 1] + text[pos] + text[pos + 2:]
        else:
            bad = text[:pos] + text[pos] + text[pos:]
        if bad != text:
            return bad
    raise AssertionError("could not build a typo")


# ------------------------------------------------- runtime-backed fact retrieval
#: Every corpus document the shipped BM25 index returns **first for its own key**
#: is a usable fact.  The generator verifies that with the real index instead of
#: trusting a hand-written list, which is what makes the trajectories match what
#: the runtime returns at evaluation time (and gives ~400 distinct facts rather
#: than the eight curated ones).
def _load_fact_pool() -> list[dict[str, str]]:
    from src.tools.retrieval import load_default_index

    index = load_default_index()
    if index is None:                                    # pragma: no cover
        return []
    corpus = Path(__file__).resolve().parents[1] / "datasets" / "retrieval" / "corpus.jsonl"
    pool: list[dict[str, str]] = []
    for line in corpus.read_text(encoding="utf-8").splitlines():
        doc = json.loads(line)
        text = doc.get("text") or ""
        if "\n" not in text:
            continue
        key = text.split("\n", 1)[0].strip().rstrip(".").lower()
        if not key:
            continue
        hits = index.search(key, k=1)
        if not hits or hits[0]["source_id"] != doc["source_id"]:
            continue          # ambiguous key: the runtime would return another doc
        pool.append({"key": key, "source_id": str(doc["source_id"]), "title": str(doc.get("title", "")),
                     "url": str(doc.get("url", "")), "source": str(doc.get("source", "curated-facts")),
                     "license": str(doc.get("license", "CC0-1.0")),
                     "passage": text.split("\n", 1)[1].strip(),
                     "snippet": str(hits[0].get("snippet", ""))})
    return pool


_FACT_POOL: list[dict[str, str]] = _load_fact_pool()
if not _FACT_POOL:                                       # pragma: no cover - defensive
    _FACT_POOL = [{"key": k, "source_id": EVIDENCE_DB[k]["source_id"],
                   "title": EVIDENCE_DB[k]["title"], "url": EVIDENCE_DB[k]["url"],
                   "source": EVIDENCE_DB[k].get("source", "curated-facts"),
                   "license": EVIDENCE_DB[k].get("license", "CC0-1.0"),
                   "passage": EVIDENCE_DB[k]["excerpt"], "snippet": EVIDENCE_DB[k]["snippet"]}
                  for k in _FACT_KEYS]

_TOOL_CTX = None


def _envelope(name: str, **args: Any) -> str:
    """The exact ``<|tool_result|>`` payload the *runtime* renders for a call.

    Running the shipped tool (BM25 search / exact arithmetic) instead of a mock
    means the training transcript is byte-identical to what the evaluation-time
    agent sees, including scores, ids and citation hashes.
    """
    global _TOOL_CTX
    from src.agent.tool_registry import ToolRegistry
    from src.tools.context import ToolContext

    if _TOOL_CTX is None:
        _TOOL_CTX = ToolContext.create(with_workspace=False)
    result = ToolRegistry.default(context=_TOOL_CTX).call(name, args)
    return json.dumps(result.to_dict(), ensure_ascii=False, sort_keys=True)


def _retrieve(entry: dict[str, str], k: int = 3) -> tuple[str, str, str]:
    """Search for a fact then fetch the top hit; return (search, fetch, passage)."""
    search_env = _envelope("search", query=entry["key"], k=k)
    payload = json.loads(search_env)
    assert payload["ok"] and payload["result"]["results"], entry
    top = payload["result"]["results"][0]
    assert top["source_id"] == entry["source_id"], (top["source_id"], entry["source_id"])
    fetch_env = _envelope("fetch", source_id=top["source_id"])
    fetched = json.loads(fetch_env)
    assert fetched["ok"], fetched
    passage = fetched["result"]["text"].split("\n", 1)[1].strip()
    assert passage == entry["passage"], (passage, entry["passage"])
    return search_env, fetch_env, passage


#: Generic source mutations.  The library patterns above only cover five specs,
#: which produced ~27 distinct repair tasks for 1000 samples; these mutations
#: apply to any solution and are accepted only when the mutated programme really
#: fails the (extended) tests while the reference still passes them.
def _swap_int_literal(source: str, rng: random.Random) -> str:
    matches = list(re.finditer(r"(?<![\w.])(\d+)(?![\w.])", source))
    if not matches:
        return source
    m = rng.choice(matches)
    new = str(max(0, int(m.group(1)) + rng.choice([-2, -1, 1, 2])))
    if new == m.group(1):
        return source
    return source[:m.start(1)] + new + source[m.end(1):]


def _swap_comparison(source: str, rng: random.Random) -> str:
    pairs = [("==", "!="), ("<=", "<"), (">=", ">"), ("<", "<="), (">", ">="), ("!=", "==")]
    rng.shuffle(pairs)
    for a, b in pairs:
        if a in source:
            return source.replace(a, b, 1)
    return source


def _flip_bool(source: str, rng: random.Random) -> str:
    return source.replace("True", "False", 1) if "True" in source else source


def _shift_index(source: str, rng: random.Random) -> str:
    for a, b in (("[0]", "[1]"), ("[1:]", "[:1]"), ("[:-1]", "[1:]"), ("[-1]", "[0]")):
        if a in source:
            return source.replace(a, b, 1)
    return source


def _bump_constant(source: str, rng: random.Random) -> str:
    return source.replace("+ 1", "+ 2", 1) if "+ 1" in source else source


def _mutated_program(spec: dict[str, Any], tests: str, rng: random.Random,
                     *, seed_i: int) -> str | None:
    """A source variant that really fails ``tests`` (verified by execution).

    Up to 24 candidate mutations are built, then all of them are executed in a
    *single* subprocess (each in its own namespace, exceptions caught) — running
    one interpreter per candidate made the dataset build take half an hour.
    """
    mutations = [_swap_int_literal, _swap_comparison, _flip_bool, _shift_index, _bump_constant]
    variants: list[str] = []
    for _ in range(24):
        variant = spec["solution"]
        for fn in rng.sample(mutations, rng.choice([1, 1, 2])):
            variant = fn(variant, rng)
        if variant and variant != spec["solution"] and variant not in variants:
            variants.append(variant)
    if not variants:
        return None

    probe = ["_failures = []"]
    for idx, variant in enumerate(variants):
        program = variant + chr(10) + chr(10) + tests
        probe += ["try:", f"    exec(compile({program!r}, 'v{idx}', 'exec'), {{}})",
                  "except Exception:", f"    _failures.append({idx})"]
    probe += ["print('FAILURES', _failures)"]
    import subprocess
    import sys as _sys

    try:
        proc = subprocess.run([_sys.executable, "-c", "\n".join(probe)], capture_output=True,
                              text=True, timeout=30)
    except subprocess.TimeoutExpired:                        # pragma: no cover
        return None
    failing: list[int] = []
    for line in proc.stdout.splitlines():
        if line.startswith("FAILURES "):
            try:
                failing = eval(line.split(" ", 1)[1])
            except Exception:                                # pragma: no cover
                failing = []
    # the reference must pass the same (extended) tests
    ok_ref, _ = run_python(spec["solution"], tests)
    if not ok_ref:                                           # pragma: no cover
        return None
    if not failing:
        return None
    first = variants[failing[0]]
    ok_again, _ = run_python(first, tests)
    return first if not ok_again else None


# ------------------------------------------------------------------- tool use
def _failing_query(entry: dict[str, str], rng: random.Random) -> str | None:
    """A query that the *real* index returns nothing for, verified by execution.

    BM25 is forgiving: a one-character typo often still retrieves something, so
    the generator cannot assume a typo misses.  It tries several mutations and
    accepts only one the shipped index really returns zero results for.
    """
    for _ in range(40):
        candidate = _typo(entry["key"], rng)
        if json.loads(_envelope("search", query=candidate, k=1))["result"]["results"] == []:
            return candidate
    words = entry["key"].split()
    for mangled in (f"{' '.join(words[:-1])} zzqx", f"zzqx {' '.join(words[1:])}",
                    f"zzqx{words[0]}"):
        if json.loads(_envelope("search", query=mangled, k=1))["result"]["results"] == []:
            return mangled
    return None


def gen_tool_use_v3(rng: random.Random, n: int) -> list[Any]:
    """The nine audited tool workflows, scaled to thousands of instances.

    0 grounded search · 1 search→fetch · 2 multi-source search ·
    3 compute · 4 sandboxed code execution · 5 code repair (fail→fix→pass) ·
    6 error recovery (empty / unknown-id / zero-division) · 7 no tool needed ·
    8 fetch by given id
    """
    out: list[Any] = []
    kinds = [0, 1, 2, 3, 4, 5, 6, 7, 8] * (n // 9 + 1)
    rng.shuffle(kinds)
    for i in range(n):
        kind = kinds[i]
        if kind == 0:
            entry = rng.choice(_FACT_POOL)
            q = entry["key"]
            user, ph = _pick_idx(rng, _SEARCH_PHRASINGS, q=q)
            assert q in user
            r1, r2, passage = _retrieve(entry)
            template = "tool/search_single"
            # the shipped BM25 snippet is the document's key line, so a grounded
            # answer must fetch the passage; the final quotes it and cites the id
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q, "k": 3}, "result": r1},
                {"name": "fetch", "arguments": {"source_id": entry["source_id"]}, "result": r2}],
                f"{passage} [{entry['source_id']}]", template,
                f"syn/tool3/search/{i}", "search", [entry["source_id"]],
                group_id=_gid(template, ph)))
        elif kind == 1:
            entry = rng.choice(_FACT_POOL)
            q = entry["key"]
            user, ph = _pick_idx(rng, _FETCH_FIRST_PHRASINGS, q=q)
            r1, r2, passage = _retrieve(entry, k=1)
            template = "tool/search_fetch"
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q, "k": 1}, "result": r1},
                {"name": "fetch", "arguments": {"source_id": entry["source_id"]}, "result": r2},
            ], f"{passage} [{entry['source_id']}]", template, f"syn/tool3/fetch/{i}", "fetch",
                [entry["source_id"]], group_id=_gid(template, ph)))
        elif kind == 2:
            e1, e2 = rng.sample(_FACT_POOL, 2)
            q1, q2 = e1["key"], e2["key"]
            user, ph = _pick_idx(rng, [
                "Verify two independent facts: {q1}, and {q2}. Cite both sources.",
                "Cross-check the two claims {q1} and {q2}, then cite both sources.",
                "Look up {q1} and {q2}, then report both with their source ids.",
                "Search for {q1} and {q2} and answer with both citations.",
                "Find sources for {q1} and for {q2} and quote both with ids.",
                "Compare what the index says about {q1} and about {q2}, with citations.",
            ], q1=q1, q2=q2)
            template = "tool/search_multi"
            r1, f1, pass1 = _retrieve(e1, k=1)
            r2, f2, pass2 = _retrieve(e2, k=1)
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q1, "k": 1}, "result": r1},
                {"name": "fetch", "arguments": {"source_id": e1["source_id"]}, "result": f1},
                {"name": "search", "arguments": {"query": q2, "k": 1}, "result": r2},
                {"name": "fetch", "arguments": {"source_id": e2["source_id"]}, "result": f2},
            ], f"{pass1} [{e1['source_id']}] Also: {pass2} [{e2['source_id']}]",
                template, f"syn/tool3/multi/{i}", "search",
                [e1["source_id"], e2["source_id"]], group_id=_gid(template, ph)))
        elif kind == 3:
            shape = rng.choice(["mul_add", "mul", "add_mul", "add_sub", "paren", "div_exact"])
            expr, value = _expression(shape, rng.randint(37, 987), rng.randint(11, 89),
                                      rng.randint(3, 40), rng)
            r1 = _mock_compute(expr, value)
            calc_id = r1["result"]["citation"]
            user, ph = _pick_idx(rng, _COMPUTE_PHRASINGS, e=expr)
            assert expr in user
            template = "tool/compute"
            out.append(_tool_trajectory(user, [
                {"name": "compute", "arguments": {"expression": expr},
                 "result": json.dumps(r1, ensure_ascii=False)}],
                f"{expr} = {value} [{calc_id}]", template, f"syn/tool3/compute/{i}", "compute",
                [calc_id], group_id=_gid(template, ph)))
        elif kind == 4:
            spec = rng.choice(_SPEC_LIBRARY)
            ok, txt = _verified_solution(spec)
            assert ok, txt
            _examples, extra_tests = instance_examples_v3(spec, rng, k=2)
            tests = spec["tests"] + ("\n" + extra_tests if extra_tests else "")
            tests_all = tests
            program = f"{spec['solution'].strip()}\n\n{tests}"
            ran, stdout = run_python(program)
            assert ran, stdout
            r1 = _mock_code((stdout + "\n") if stdout else "all assertions passed\n", code=program)
            user, ph = _pick_idx(rng, _CODE_RUN_PHRASINGS, c=program)
            run_id = r1["result"]["citation"]
            final = f"Exit code 0; stdout was {stdout!r} [{run_id}]."
            template = "tool/code_run"
            out.append(_tool_trajectory(user, [
                {"name": "code", "arguments": {"code": program, "timeout_s": 8},
                 "result": json.dumps(r1, ensure_ascii=False)}],
                final, template, f"syn/tool3/code/{i}", "code", [run_id],
                group_id=_gid(template, ph)))
        elif kind == 5:
            if i % 2 == 0:
                spec, buggy = _buggy_program(rng, seed_i=i)
                tests_all = spec["tests"]
            else:
                spec = _SPEC_LIBRARY[i % len(_SPEC_LIBRARY)]
                _examples, extra = instance_examples_v3(spec, rng, k=2)
                tests_all = spec["tests"] + ("\n" + extra if extra else "")
                buggy = _mutated_program(spec, tests_all, rng, seed_i=i)
                if buggy is None:                       # pragma: no cover - defensive
                    spec, buggy = _buggy_program(rng, seed_i=i)
                    tests_all = spec["tests"]
            ok_bad, out_bad = run_python(buggy, tests_all)
            assert not ok_bad, (spec["name"], out_bad)
            ok_good, out_good = run_python(spec["solution"], tests_all)
            assert ok_good, out_good
            r_bad = _mock_code("", exit_code=1)
            r_bad["result"] = {**r_bad["result"], "code": buggy, "citation": _code_id(buggy),
                               "stdout": "", "stderr": (out_bad.splitlines()[-1] if out_bad else "AssertionError")}
            r_good = _mock_code((out_good + "\n") if out_good else "all assertions passed\n",
                                code=spec["solution"])
            user, ph = _pick_idx(rng, _CODE_REPAIR_PHRASINGS, c=buggy)
            final = (f"All assertions passed after the repair; stdout was {out_good!r} "
                     f"[{r_good['result']['citation']}].")
            template = "tool/code_repair"
            out.append(_tool_trajectory(user, [
                {"name": "code", "arguments": {"code": buggy, "timeout_s": 8},
                 "result": json.dumps(r_bad, ensure_ascii=False)},
                {"name": "code", "arguments": {"code": spec["solution"], "timeout_s": 8},
                 "result": json.dumps(r_good, ensure_ascii=False)},
            ], final, template, f"syn/tool3/repair/{i}", "code", [r_good["result"]["citation"]],
                group_id=_gid(template, ph)))
        elif kind == 6:
            mode = i % 3
            if mode == 0:
                entry = rng.choice(_FACT_POOL)
                good = entry["key"]
                bad = _failing_query(entry, rng)
                if bad is None:                       # pragma: no cover - defensive
                    continue
                r_bad = _envelope("search", query=bad, k=1)
                assert json.loads(r_bad)["result"]["results"] == [], bad
                r_good, f_good, passage = _retrieve(entry, k=3)
                hit = {"source_id": entry["source_id"], "snippet": entry["snippet"]}
                user, ph = _pick_idx(rng, [
                    "Look up the {q} with the misspelled query {bad!r}; if the search finds nothing, "
                    "correct the query and answer with a citation.",
                    "Search for {bad!r}; if that returns nothing, find the {q} and cite the source.",
                    "Try the query {bad!r} first; if the index finds nothing, search for the {q} and "
                    "answer with a citation.",
                ], q=good, bad=bad)
                calls = [{"name": "search", "arguments": {"query": bad, "k": 1}, "result": r_bad},
                         {"name": "search", "arguments": {"query": good, "k": 3}, "result": r_good},
                         {"name": "fetch", "arguments": {"source_id": entry["source_id"]},
                          "result": f_good}]
                final, cites, expected = (f"{passage} [{entry['source_id']}]",
                                          [entry["source_id"]], "search")
            elif mode == 1:
                entry = rng.choice(_FACT_POOL)
                q = entry["key"]
                r_fail = _envelope("fetch", source_id="FACT-000000")
                assert not json.loads(r_fail)["ok"], r_fail
                r1, r2, passage = _retrieve(entry, k=1)
                hit = {"source_id": entry["source_id"]}
                user, ph = _pick_idx(rng, [
                    "Fetch source_id FACT-000000; if that fails, find the real id for the {q} and fetch it.",
                    "Try to fetch FACT-000000. If the id is unknown, locate the id for the {q} and read it.",
                    "Fetch FACT-000000 and, if it does not exist, search for the {q} and fetch the real id.",
                ], q=q)
                calls = [{"name": "fetch", "arguments": {"source_id": "FACT-000000"}, "result": r_fail},
                         {"name": "search", "arguments": {"query": q, "k": 1}, "result": r1},
                         {"name": "fetch", "arguments": {"source_id": entry["source_id"]}, "result": r2}]
                final, cites, expected = f"{passage} [{entry['source_id']}]", [entry["source_id"]], "fetch"
            else:
                divisor = rng.choice([2, 3, 4, 5, 6, 8])
                a = rng.randint(12, 400) * divisor
                r_fail = {"ok": False, "name": "compute", "result": {}, "error": "zero_division",
                          "duration_s": 0.0001}
                ok, printed = run_python(f"print({a} // {divisor})")
                assert ok and printed.strip() == str(a // divisor)
                value = a // divisor
                r_ok = _mock_compute(f"{a} // {divisor}", value)
                user, ph = _pick_idx(rng, [
                    "Evaluate {a} / 0 with the compute tool; if it errors, evaluate {a} // {d} instead "
                    "and report the value.",
                    "Try to compute {a} / 0; on failure use {a} // {d} and give the value.",
                    "Compute {a} / 0. If the tool reports an error, compute {a} // {d} and report that value.",
                ], a=a, d=divisor)
                calls = [{"name": "compute", "arguments": {"expression": f"{a} / 0"},
                          "result": json.dumps(r_fail, ensure_ascii=False)},
                         {"name": "compute", "arguments": {"expression": f"{a} // {divisor}"},
                          "result": json.dumps(r_ok, ensure_ascii=False)}]
                final, cites, expected = f"{value} [{r_ok['result']['citation']}]", [r_ok["result"]["citation"]], "compute"
            template = "tool/error_recovery"
            out.append(_tool_trajectory(user, calls, final, template, f"syn/tool3/recovery/{i}",
                                        expected, cites, group_id=_gid(template, mode * 4 + ph)))
        elif kind == 7:
            sub = i % 3
            if sub == 0:
                a, b = rng.randint(11, 98), rng.randint(11, 98)
                user, ph = _pick_idx(rng, _NO_TOOL_ADD, a=a, b=b)
                ans = a + b
                assert eval(f"{a} + {b}") == ans
                template = "no_tool/simple_math_add"
            elif sub == 1:
                a, b = rng.randint(21, 999), rng.randint(1, 20)
                user, ph = _pick_idx(rng, _NO_TOOL_SUB, a=a, b=b)
                ans = a - b
                assert eval(f"{a} - {b}") == ans
                template = "no_tool/simple_math_sub"
            else:
                a, b = rng.randint(2, 9), rng.randint(2, 9)
                user, ph = _pick_idx(rng, _NO_TOOL_MUL, a=a, b=b)
                ans = a * b
                assert eval(f"{a} * {b}") == ans
                template = "no_tool/simple_math_mul"
            segs = [Segment("system", TOOL_SYSTEM, target=False),
                    Segment("user", user, target=False),
                    Segment("assistant", f"<|final|>\n{rng.choice(_FINAL_STYLES).format(v=ans)}",
                            target=True)]
            out.append(make_segment_record(segments=segs, category="tool_use", source="synthetic",
                                           source_id=f"syn/tool3/notool/{i}", task_type="no_tool",
                                           template_id=template, group_id=_gid(template, ph),
                                           verified=True, verifier="python_recomputation",
                                           expected_tool="none", answer=str(ans)))
        else:
            entry = rng.choice(_FACT_POOL)
            q, sid = entry["key"], entry["source_id"]
            user, ph = _pick_idx(rng, _FETCH_PHRASINGS, sid=sid, q=q)
            assert sid in user
            fetch_env = _envelope("fetch", source_id=sid)
            fetched = json.loads(fetch_env)
            assert fetched["ok"], fetched
            passage = fetched["result"]["text"].split("\n", 1)[1].strip()
            template = "tool/fetch_direct"
            out.append(_tool_trajectory(user, [
                {"name": "fetch", "arguments": {"source_id": sid}, "result": fetch_env}],
                f"{passage} [{sid}]", template, f"syn/tool3/fetchid/{i}", "fetch",
                [sid], group_id=_gid(template, ph)))
    return out


# ---------------------------------------------------------------- algorithms
_ALG_PHRASINGS = [
    "Trace binary search for {t} in the sorted list {arr}.",
    "Show the binary-search steps that find {t} in {arr}.",
    "Run a binary search for {t} over {arr} and report the index it returns.",
    "Walk through binary search looking for {t} in {arr}, step by step.",
    "What does binary search do while looking for {t} in {arr}? Give the trace.",
    "Trace the search for {t} in {arr} and give the final index.",
]

_MAX_PHRASINGS = [
    "Trace a single pass computing the running maximum of {arr} and report it.",
    "Walk through {arr} once and give the running maximum.",
    "Show the running maximum of {arr} after one left-to-right pass.",
    "Compute the running maximum of the list {arr} step by step.",
]


def gen_algorithm_v3(rng: random.Random, n: int) -> list[Any]:
    """Execution-verified algorithm traces with instance-level split groups.

    Two sub-families (binary search, running maximum).  Every trace is produced
    by *running* the reference implementation in a subprocess and is asserted
    against it, and the group id is per phrasing so the splitter can hold out
    phrasings instead of dumping the whole family into train (which is what left
    the historical ``algorithm`` category with a single group of 120 records).
    """
    out: list[Any] = []
    for i in range(n):
        if i % 3:
            n_items = rng.randint(4, 8)
            values = rng.sample(range(1, 95), n_items)
            target = rng.choice(values)
            arr = sorted(values)
            lo, hi, steps = 0, len(arr) - 1, []
            while lo <= hi:
                mid = (lo + hi) // 2
                steps.append(f"lo={lo} hi={hi} mid={mid} value={arr[mid]}")
                if arr[mid] == target:
                    break
                if arr[mid] < target:
                    lo = mid + 1
                else:
                    hi = mid - 1
            idx = arr.index(target)
            code = ("def binary_search(a, x):\n"
                    "    lo, hi = 0, len(a) - 1\n"
                    "    while lo <= hi:\n"
                    "        mid = (lo + hi) // 2\n"
                    "        if a[mid] == x:\n"
                    "            return mid\n"
                    "        if a[mid] < x:\n"
                    "            lo = mid + 1\n"
                    "        else:\n"
                    "            hi = mid - 1\n"
                    "    return -1\n")
            ok, out_txt = run_python(code + f"\nassert binary_search({arr}, {target}) == {idx}\n"
                                            f"print('verified')")
            assert ok, out_txt
            user, ph = _pick_idx(rng, _ALG_PHRASINGS, t=target, arr=arr)
            assert str(target) in user and str(arr)[1:-1].split(",")[0].strip() in user
            template = "algorithm/binary_search_trace"
            answer = f"index {idx}"
            body = ("<|thought|>\n" + "\n".join(steps) +
                    f"\nThe matching index is {idx}.\n<|answer|>\n{answer}")
            tests = [f"assert binary_search({arr}, {target}) == {idx}"]
            verifier = "subprocess_execution"
        else:
            n_items = rng.randint(4, 8)
            arr = [rng.randint(1, 95) for _ in range(n_items)]
            best, steps = arr[0], []
            for j, value in enumerate(arr):
                if value > best:
                    best = value
                steps.append(f"i={j} value={value} best={best}")
            user, ph = _pick_idx(rng, _MAX_PHRASINGS, arr=arr)
            template = "algorithm/running_max_trace"
            answer = f"maximum {best}"
            body = ("<|thought|>\n" + "\n".join(steps) +
                    f"\nThe running maximum is {best}.\n<|answer|>\n{answer}")
            tests = [f"assert max({arr}) == {best}"]
            verifier = "python_recomputation"
        segs = [Segment("system", "You are TinyMe. Trace algorithms step by step.", target=False),
                Segment("user", user, target=False),
                Segment("assistant", body, target=True)]
        out.append(make_segment_record(segments=segs, category="algorithm", source="synthetic",
                                       source_id=f"syn/alg3/{i}", task_type="reasoning",
                                       template_id=template, group_id=_gid(template, ph),
                                       verified=True, verifier=verifier, tests=tests,
                                       answer=answer))
    return out


# ---------------------------------------------------------------- direct tasks
def gen_no_tool_v3(rng: random.Random, n: int) -> list[Any]:
    """Direct answering (no tool) using the audited phrasings.

    Sub-families: two-digit addition, subtraction, single-digit multiplication,
    upper-casing, string reversal.  Every answer is recomputed.
    """
    out: list[Any] = []
    for i in range(n):
        sub = i % 5
        if sub == 0:
            a, b = rng.randint(11, 98), rng.randint(11, 98)
            user, ph = _pick_idx(rng, _NO_TOOL_ADD, a=a, b=b)
            ans = a + b
            assert eval(f"{a} + {b}") == ans
            template, verifier = "no_tool/arith/add", "python_recomputation"
        elif sub == 1:
            a, b = rng.randint(21, 999), rng.randint(1, 20)
            user, ph = _pick_idx(rng, _NO_TOOL_SUB, a=a, b=b)
            ans = a - b
            assert eval(f"{a} - {b}") == ans
            template, verifier = "no_tool/arith/sub", "python_recomputation"
        elif sub == 2:
            a, b = rng.randint(2, 9), rng.randint(2, 9)
            user, ph = _pick_idx(rng, _NO_TOOL_MUL, a=a, b=b)
            ans = a * b
            assert eval(f"{a} * {b}") == ans
            template, verifier = "no_tool/arith/mul", "python_recomputation"
        elif sub == 3:
            word = rng.choice(_WORD_POOL) + str(rng.randint(0, 99))
            user, ph = _pick_idx(rng, _NO_TOOL_UPPER, w=word)
            ans = word.upper()
            template, verifier = "no_tool/text_edit/upper", "upper_recomputation"
        else:
            word = rng.choice(_WORD_POOL) + str(rng.randint(0, 99))
            user, ph = _pick_idx(rng, _NO_TOOL_REVERSE, w=word)
            ans = word[::-1]
            template, verifier = "no_tool/text_edit/reverse", "reverse_recomputation"
        segs = [Segment("system", TOOL_SYSTEM, target=False),
                Segment("user", user, target=False),
                Segment("assistant", f"<|final|>\n{rng.choice(_FINAL_STYLES).format(v=ans)}",
                        target=True)]
        out.append(make_segment_record(segments=segs, category="instruction", source="synthetic",
                                       source_id=f"syn/notool3/{sub}/{i}", task_type="no_tool",
                                       template_id=template, group_id=_gid(template, ph),
                                       verified=True, verifier=verifier, expected_tool="none",
                                       answer=str(ans)))
    return out


# ----------------------------------------------------------------- challenge
def gen_tool_challenge(rng: random.Random, n: int) -> list[Any]:
    """Held-out tool templates: these phrasings exist **only** here.

    Answerable by the same runtime behaviour, but with sentence shapes that never
    appear in training, so a passing score cannot come from template similarity.
    """
    out: list[Any] = []
    i = 0
    while len(out) < n:
        i += 1
        kind = i % 3
        idx = len(out)
        if kind == 0:
            a, b, c = rng.randint(120, 9800), rng.randint(11, 90), rng.randint(3, 60)
            expr = f"{a} // {b} + {c}"
            value = a // b + c
            assert eval(expr) == value
            r1 = _mock_compute(expr, value)
            user = (f"An operator asks for the exact value of the integer expression {expr} "
                    f"(evaluate it left to right). Report it with a citation.")
            final = f"{value} [{r1['result']['citation']}]"
            rec = _tool_trajectory(user, [
                {"name": "compute", "arguments": {"expression": expr},
                 "result": json.dumps(r1, ensure_ascii=False)}],
                final, "challenge/tool/arith_chain", f"challenge/tool3/{idx}", "compute",
                [r1["result"]["citation"]])
        elif kind == 1:
            entry = rng.choice(_FACT_POOL)
            sid = entry["source_id"]
            fetch_env = _envelope("fetch", source_id=sid)
            fetched = json.loads(fetch_env)
            assert fetched["ok"], fetched
            passage = fetched["result"]["text"].split("\n", 1)[1].strip()
            user = (f"An audit needs the stored passage identified by {sid}. Quote the "
                    f"single sentence it stores and give the source id in brackets.")
            final = f"{passage} [{sid}]"
            rec = _tool_trajectory(user, [
                {"name": "fetch", "arguments": {"source_id": sid}, "result": fetch_env}],
                final, "challenge/tool/quote_passage", f"challenge/tool3/{idx}", "fetch",
                [sid])
        else:
            a, b = rng.randint(11, 79), rng.randint(11, 79)
            value = a * b
            assert value == a * b
            r1 = _mock_compute(f"{a} * {b}", value)
            user = (f"Multiply {a} by {b} using the compute tool and answer with the citation "
                    f"the tool returned.")
            final = f"{value} [{r1['result']['citation']}]"
            rec = _tool_trajectory(user, [
                {"name": "compute", "arguments": {"expression": f"{a} * {b}"},
                 "result": json.dumps(r1, ensure_ascii=False)}],
                final, "challenge/tool/multiply", f"challenge/tool3/{idx}", "compute",
                [r1["result"]["citation"]])
        rec.split = "challenge"
        out.append(rec)
    return out


GENERATORS_V3: dict[str, tuple[Any, int, str]] = {
    "copy_span": (gen_copy_span, 4000, "instruction"),
    "no_tool": (gen_no_tool_v3, 6000, "instruction"),
    "tool_use": (gen_tool_use_v3, 9000, "tool"),
    "algorithm": (gen_algorithm_v3, 1600, "algorithm"),
}


def generate_corpus_v3(seed: int = 20261003, counts: dict[str, int] | None = None,
                       scale: float = 1.0) -> tuple[list[Any], list[dict[str, Any]]]:
    """Generate the v3 blocks with the same duplicate-inflation guard as v2."""
    rng = random.Random(seed)
    records: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for name, (fn, default, family) in GENERATORS_V3.items():
        want = int(round((counts or {}).get(name, default) * scale))
        if want <= 0:
            continue
        produced = fn(rng, want)
        assert len(produced) == want, (name, len(produced), want)
        seen: set[str] = set()
        unique = []
        for rec in produced:
            key = hashlib.sha256(rec.text.encode("utf-8")).hexdigest()
            if key in seen:
                continue
            seen.add(key)
            unique.append(rec)
        records.extend(unique)
        provenance.append({
            "requested": want, "emitted": len(unique), "duplicates_dropped": want - len(unique),
            "source": "synthetic", "source_id": f"synv3/{name}",
            "source_url": "local://data_sources/synthetic_v3.py", "retrieval_date": "generated",
            "license": "Synthetic-Verified", "license_url": "", "category_family": family,
            "records": len(unique), "verified": True,
            "verifier": "python_recomputation | subprocess_execution | index_containment",
            "generator": f"synthetic_v3.{fn.__name__}", "seed": seed,
            "notes": ("capability-scaled generator: verbatim-copy invariant, independent "
                      "verification, duplicate inflation rejected"),
        })
    return records, provenance
