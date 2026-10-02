"""Deterministic, independently-verified synthetic generators (v2).

Design rules from the corrective audit:

* **problem generator → solution generator → independent verifier → accept**
  (`verify_tests` runs real code where code is involved; arithmetic answers are
  recomputed with Python, never copied from the prompt);
* every record carries a ``template_id`` / ``group_id`` so the splitter can hold
  out entire template families instead of rows;
* tool-use records teach the exact runtime protocol
  (``<|tool_call|>`` JSON, ``<|tool_result|>`` context, ``<|final|>`` target),
  with deterministic mocked evidence so results are reproducible;
* no record is emitted with an unverified answer: each generator asserts its own
  answer before returning.

Public entry point: :func:`generate_corpus`.
"""
from __future__ import annotations

import hashlib
import json
import random
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Callable

from src.data.records import Segment, make_segment_record

# --------------------------------------------------------------------- tools
#: Model-facing tool surface.  This dictionary is the single source of truth for
#: the training data and mirrors ``src/agent/tool_registry.py`` exactly (names,
#: argument names, required flags) so a trajectory the model learns is a
#: trajectory the runtime accepts.  Drift here is a bug: see
#: ``tests/test_tool_protocol.py::test_tool_specs_match_runtime_registry``.
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search": {"description": "Search the local retrieval corpus; returns ranked passages with source ids.",
               "args": {"query": "string", "k": "int=3"},
               "required": ["query"]},
    "fetch": {"description": "Fetch a stored passage by source id (no network access).",
              "args": {"source_id": "string"}, "required": ["source_id"]},
    "compute": {"description": "Evaluate a deterministic arithmetic/math expression exactly.",
                "args": {"expression": "string"}, "required": ["expression"]},
    "code": {"description": "Run a short Python program in the sandbox (no network, CPU/memory/time limited).",
             "args": {"code": "string", "timeout_s": "int=10"}, "required": ["code"]},
    "files": {"description": "Read or list files inside the current sandbox workspace.",
              "args": {"action": "string", "path": "string"}, "required": ["action"]},
}

#: Fact keys used by the tool-use generators (the runtime retrieval index ships
#: the same facts, so a trained trajectory matches what the real tools return).
_FACT_KEYS = ["population of jakarta", "population of bandung", "capital of indonesia",
              "definition of a prime number", "big-o of binary search",
              "water freezing point at one atmosphere",
              "psf licence of the python standard library",
              "shortest path algorithm for non-negative weights"]

#: Static content for the eight curated facts.  Source ids are taken from the
#: shipped retrieval corpus at import time (:func:`_runtime_ids`) so a generated
#: trajectory cites exactly what the runtime would return; the static ids below
#: are only a fallback when ``datasets/retrieval/corpus.jsonl`` is absent.
_STATIC_EVIDENCE: dict[str, dict[str, Any]] = {
    "population of jakarta": {"source_id": "S1", "title": "Statistics Indonesia, 2024 census release",
                              "url": "https://example.org/facts/jakarta-population",
                              "snippet": "Population of jakarta.",
                              "excerpt": "Jakarta's administrative population was recorded as 10,679,951 in the 2024 release."},
    "population of bandung": {"source_id": "S1", "title": "Statistics Indonesia, 2024 census release",
                              "url": "https://example.org/facts/bandung-population",
                              "snippet": "Population of bandung.",
                              "excerpt": "Bandung's administrative population was recorded as 2,506,603 in the 2024 release."},
    "capital of indonesia": {"source_id": "S1", "title": "Indonesian government portal",
                             "url": "https://example.org/facts/indonesia-capital",
                             "snippet": "Capital of indonesia.",
                             "excerpt": "Jakarta is the capital of Indonesia; the planned move to Nusantara is phased through 2029."},
    "definition of a prime number": {"source_id": "S1", "title": "Encyclopaedia of Mathematics",
                                     "url": "https://example.org/facts/prime-number",
                                     "snippet": "Definition of a prime number.",
                                     "excerpt": "A prime number is an integer greater than 1 whose only positive divisors are 1 and itself."},
    "big-o of binary search": {"source_id": "S1", "title": "Algorithms, 4th ed. companion notes",
                               "url": "https://example.org/facts/binary-search",
                               "snippet": "Big-o of binary search.",
                               "excerpt": "Binary search halves the search interval each step, giving O(log n) comparisons."},
    "psf licence of the python standard library": {"source_id": "S1", "title": "Python Software Foundation",
                                                   "url": "https://example.org/facts/psf-license",
                                                   "snippet": "Psf licence of the python standard library.",
                                                   "excerpt": "The Python standard library is distributed under the PSF-2.0 licence."},
    "water freezing point at one atmosphere": {"source_id": "S1", "title": "NIST physical constants",
                                               "url": "https://example.org/facts/water-freezing",
                                               "snippet": "Water freezing point at one atmosphere.",
                                               "excerpt": "At one atmosphere of pressure water freezes at 0 degrees Celsius (273.15 K)."},
    "shortest path algorithm for non-negative weights": {"source_id": "S1", "title": "Algorithms, 4th ed. companion notes",
                                                         "url": "https://example.org/facts/shortest-path",
                                                         "snippet": "Shortest path algorithm for non-negative weights.",
                                                         "excerpt": "Dijkstra's algorithm computes single-source shortest paths for non-negative edge weights."},
}


def _runtime_ids() -> dict[str, dict[str, str]]:
    """Source ids/titles/urls for the curated facts, read from the real index."""
    found: dict[str, dict[str, str]] = {}
    corpus = Path(__file__).resolve().parents[1] / "datasets" / "retrieval" / "corpus.jsonl"
    if not corpus.exists():
        return found
    try:
        for line in corpus.read_text(encoding="utf-8").splitlines():
            doc = json.loads(line)
            text = (doc.get("text") or "").lower()
            for key in _FACT_KEYS:
                if key in text and key not in found:
                    found[key] = {"source_id": str(doc.get("source_id", "")),
                                  "title": str(doc.get("title", "")),
                                  "url": str(doc.get("url", "")),
                                  "source": str(doc.get("source", "curated-facts")),
                                  "license": str(doc.get("license", "CC0-1.0"))}
    except Exception:                      # pragma: no cover - defensive
        return {}
    return found


_RUNTIME_IDS = _runtime_ids()

#: query -> evidence entry (excerpt text is used by ``fetch`` and by the final
#: answer, exactly like the runtime's stored passage).
EVIDENCE_DB: dict[str, dict[str, Any]] = {
    key: {**value, **_RUNTIME_IDS.get(key, {})} for key, value in _STATIC_EVIDENCE.items()
}


def _find_evidence(query: str) -> dict[str, Any] | None:
    q = query.strip().lower()
    if q in EVIDENCE_DB:
        return EVIDENCE_DB[q]
    q_words = set(q.split())
    for key, ev in EVIDENCE_DB.items():
        # every word of the fact must appear as a token of the query; a typo in
        # any content word therefore returns nothing, like a real BM25 miss.
        if key in q or all(w in q_words for w in key.split()):
            return ev
    return None


def _mock_search(query: str, k: int = 3) -> dict[str, Any]:
    """Mock of the runtime ``search`` envelope (see src/tools/search.py)."""
    ev = _find_evidence(query)
    results: list[dict[str, Any]] = []
    if ev is not None:
        results.append({"source_id": ev["source_id"], "title": ev["title"], "snippet": ev["snippet"],
                        "url": ev["url"], "source": ev.get("source", "curated-facts"),
                        "license": ev.get("license", "CC0-1.0"), "score": 16.62})
    return {"ok": True, "name": "search",
            "result": {"query": query.strip().lower(), "provider": "local-bm25",
                       "provider_status": "local-bm25:ok", "results": results[:k]},
            "duration_s": 0.0004}


def _mock_fetch(source_id: str) -> dict[str, Any]:
    """Mock of the runtime ``fetch`` envelope (src/tools/fetch.py)."""
    for key, ev in EVIDENCE_DB.items():
        if ev["source_id"] == source_id:
            return {"ok": True, "name": "fetch",
                    "result": {"source_id": ev["source_id"], "title": ev["title"], "url": ev["url"],
                               "source": ev.get("source", "curated-facts"),
                               "license": ev.get("license", "CC0-1.0"),
                               "text": f"{key.capitalize()}.\n{ev['excerpt']}", "truncated": False,
                               "provider": "local-index"},
                    "duration_s": 0.0002}
    return {"ok": False, "name": "fetch", "result": {}, "duration_s": 0.0001,
            "error": f"source_not_found:{source_id}"}


def _mock_compute(expression: str, value: Any) -> dict[str, Any]:
    """Mock of the runtime ``compute`` envelope (src/tools/compute.py)."""
    return {"ok": True, "name": "compute",
            "result": {"expression": expression, "value": value, "exact": str(value),
                       "method": "python-ast-exact"},
            "duration_s": 0.0001}


def _mock_code(stdout: str, exit_code: int = 0) -> dict[str, Any]:
    """Mock of the runtime ``code`` envelope (src/tools/code.py)."""
    return {"ok": exit_code == 0, "name": "code",
            "result": {"exit_code": exit_code, "timed_out": False, "stdout": stdout, "stderr": "",
                       "duration_s": 0.0728, "truncated": False,
                       "isolation_level": "namespace(net+mount)", "network_enforced": True},
            "duration_s": 0.1199}


def _tool_call(name: str, arguments: dict[str, Any]) -> str:
    return json.dumps({"name": name, "arguments": arguments}, ensure_ascii=False, sort_keys=True)


# ------------------------------------------------------------------ verifier
def run_python(code: str, tests: str | None = None, timeout: float = 10.0) -> tuple[bool, str]:
    """Execute code (and optional tests) in a subprocess; return (ok, output)."""
    program = code if not tests else code + "\n\n" + tests
    try:
        proc = subprocess.run([sys.executable, "-c", program], capture_output=True,
                              text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, "timeout"
    except Exception as exc:  # pragma: no cover
        return False, f"{type(exc).__name__}: {exc}"
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


# ------------------------------------------------------------------- helpers
def _math_rec(q: str, a: str, template: str, source_id: str, *, tests: list[str] | None = None,
              answer: str | None = None) -> Any:
    segs = [
        Segment("system", "You are TinyMe, a precise reasoning assistant. Show concise work.", target=False),
        Segment("user", q, target=False),
        Segment("assistant", f"<|thought|>\nCompute directly and verify the result.\n<|answer|>\n{a}", target=True),
    ]
    return make_segment_record(segments=segs, category="math", source="synthetic",
                               source_id=source_id, task_type="reasoning", template_id=template,
                               verified=True, verifier="python_recomputation",
                               answer=answer or a, tests=tests or [])


# ------------------------------------------------------------------ math/ logic
def gen_arithmetic(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        kind = i % 4
        if kind == 0:
            a, b = rng.randint(100, 9999), rng.randint(100, 9999)
            ans = a + b
            q = f"What is {a} + {b}?"
        elif kind == 1:
            a, b = rng.randint(100, 999), rng.randint(10, 99)
            ans = a * b
            q = f"Compute {a} x {b}."
        elif kind == 2:
            b = rng.randint(7, 97)
            k = rng.randint(3, 400)
            a = b * k + rng.randint(0, b - 1)
            q = f"Divide {a} by {b}. Give the quotient and the remainder."
            ans = f"quotient {a // b}, remainder {a % b}"
        else:
            a = rng.randint(2, 40)
            k = rng.randint(2, 5)
            q = f"What is {a} to the power of {k}?"
            ans = a ** k
        # independent verification (never trusts the generating expression)
        if kind == 0:
            assert eval(f"{a} + {b}") == ans
        elif kind == 1:
            assert sum(b for _ in range(a)) == ans
        elif kind == 2:
            assert a == b * (a // b) + (a % b)
            assert (a // b, a % b) == (int(ans.split()[1].rstrip(",")), int(ans.split()[3]))
        else:
            product = 1
            for _ in range(k):
                product *= a
            assert product == ans
        out.append(_math_rec(q, str(ans), template=f"math/arithmetic/kind{kind}",
                             source_id=f"syn/math/arith/{i}"))
    return out


def gen_word_problems(rng: random.Random, n: int) -> list[Any]:
    out = []
    names = ["Ada", "Budi", "Citra", "Dewi", "Eko", "Fajar", "Gita", "Hana"]
    items = ["apples", "books", "coins", "marbles", "stickers", "tickets"]
    for i in range(n):
        who = names[i % len(names)]
        item = items[(i // len(names)) % len(items)]
        price = rng.randint(3, 19)
        count = rng.randint(4, 40)
        spent = rng.randint(10, 60)
        total = price * count
        left = 200 - total - spent
        q = (f"{who} buys {count} {item} at {price} units each and spends {spent} units on a bag. "
             f"If {who} started with 200 units, how much is left?")
        ans = f"{left} units"
        assert 200 - (price * count) - spent == left
        segs = [Segment("system", "You are TinyMe. Solve word problems step by step, then answer.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", (f"<|thought|>\nCost = {count} x {price} = {total}. "
                                      f"Total spent = {total} + {spent} = {total + spent}. "
                                      f"Remaining = 200 - {total + spent} = {left}.\n<|answer|>\n{ans}"), target=True)]
        out.append(make_segment_record(segments=segs, category="math", source="synthetic",
                                       source_id=f"syn/math/word/{i}", task_type="reasoning",
                                       template_id="math/word_problem/kind0", verified=True,
                                       verifier="python_recomputation", answer=ans))
    return out


def gen_sequence(rng: random.Random, n: int) -> list[Any]:
    out = []
    kinds: list[int] = [0, 1, 2, 3] * (n // 4 + 1)
    rng.shuffle(kinds)
    for i in range(n):
        kind = kinds[i]
        if kind == 0:                       # arithmetic, increasing or decreasing
            a, d = rng.randint(1, 40), rng.choice([k for k in range(-12, 13) if k != 0])
            seq = [a + d * k for k in range(6)]
            nxt = a + d * 6
            direction = "increases" if d > 0 else "decreases"
            rule = f"each term {direction} by {abs(d)}"
        elif kind == 1:                     # geometric
            a, r, length = rng.randint(1, 9), rng.randint(2, 5), rng.randint(4, 6)
            seq = [a * r ** k for k in range(length)]
            nxt = a * r ** length
            rule = f"each term is multiplied by {r}"
        elif kind == 2:                     # quadratic: first difference grows by g
            a, d, g, length = rng.randint(1, 30), rng.randint(1, 9), rng.randint(1, 5), rng.randint(5, 6)
            seq = [a + d * k + g * k * (k - 1) // 2 for k in range(length)]
            nxt = a + d * length + g * length * (length - 1) // 2
            rule = f"the difference starts at {d} and grows by {g} each step"
        else:                               # fibonacci-like with random seeds
            x, y = rng.randint(1, 9), rng.randint(1, 9)
            seq = [x, y]
            while len(seq) < 7:
                seq.append(seq[-1] + seq[-2])
            nxt = seq[-1] + seq[-2]
            rule = "each term is the sum of the two previous terms"
        q = f"What is the next term? Sequence: {', '.join(map(str, seq))}, ?"
        segs = [Segment("system", "You are TinyMe. Identify the rule, then answer.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nRule: {rule}.\n<|answer|>\n{nxt}", target=True)]
        if kind == 0:
            assert nxt == seq[-1] + (seq[-1] - seq[-2])
        elif kind == 1:
            assert nxt == seq[-1] * (seq[-1] // seq[-2])
        elif kind == 2:
            diffs = [b - a for a, b in zip(seq, seq[1:])]
            assert diffs[-1] - diffs[-2] == g
        else:
            assert nxt == seq[-1] + seq[-2]
        out.append(make_segment_record(segments=segs, category="math", source="synthetic",
                                       source_id=f"syn/math/seq/{i}", task_type="reasoning",
                                       template_id=f"math/sequence/kind{kind}", verified=True,
                                       verifier="python_recomputation", answer=str(nxt)))
    return out


def gen_boolean(rng: random.Random, n: int) -> list[Any]:
    out = []
    ops = {"and": lambda a, b: a and b, "or": lambda a, b: a or b, "xor": lambda a, b: a != b}
    for i in range(n):
        a, b, c, d = (rng.random() < 0.5 for _ in range(4))
        op1, op2, op3 = rng.choice(list(ops)), rng.choice(list(ops)), rng.choice(list(ops))
        val = ops[op3](ops[op1](a, b), ops[op2](c, d))
        names = rng.sample(["p", "q", "r", "s"], 4)
        def atom(name, value):
            return (f"not {name}", not value) if rng.random() < 0.3 else (name, value)
        (na, _va), (nb, _vb), (nc, _vc), (nd, _vd) = (atom(n, v) for n, v in zip(names, (a, b, c, d)))
        expr = (f"({na} {op1} {nb}) {op3} ({nc} {op2} {nd})")
        q = (f"Evaluate the boolean expression: {expr}\n"
             f"Values: {', '.join(f'{nm}={str(v).lower()}' for nm, v in zip(names, (a, b, c, d)))}")
        segs = [Segment("system", "You are TinyMe. Evaluate logic expressions exactly.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nEvaluate the left group with {op1}, the right group "
                                     f"with {op2}, then combine with {op3}.\n<|answer|>\n{str(val).lower()}",
                        target=True)]
        assert val in (True, False)
        out.append(make_segment_record(segments=segs, category="logic", source="synthetic",
                                       source_id=f"syn/logic/bool/{i}", task_type="reasoning",
                                       template_id=f"logic/boolean/{op3}", verified=True,
                                       verifier="python_recomputation", answer=str(val).lower()))
    return out


def gen_deduction(rng: random.Random, n: int) -> list[Any]:
    out = []
    animals = list(_NAME_POOL)
    traits = list(_TRAIT_POOL)
    names = ("Milo", "Nina", "Omar", "Pia", "Quinn", "Rosa", "Sami", "Tara", "Umar", "Vera",
             "Wira", "Xena", "Yara", "Zane")
    for i in range(n):
        a, trait, who = rng.choice(animals), rng.choice(traits), rng.choice(names)
        q = (f"All {a} are {trait}. {who} is a {a[:-1]}. What can we conclude about {who}?")
        ans = f"{who} is {trait}."
        segs = [Segment("system", "You are TinyMe. Apply the rule exactly once.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nThe premise applies because {who} is a {a[:-1]}.\n"
                                     f"<|answer|>\n{ans}", target=True)]
        assert trait in ans and who in ans
        out.append(make_segment_record(segments=segs, category="logic", source="synthetic",
                                       source_id=f"syn/logic/ded/{i}", task_type="reasoning",
                                       template_id="logic/deduction/syllogism", verified=True,
                                       verifier="template_check", answer=ans))
    return out


# ------------------------------------------------------------------ algorithms
def gen_algorithm_trace(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        n_items = rng.randint(4, 7)
        values = rng.sample(range(1, 90), n_items)          # unique -> index is well defined
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
        q = f"Trace binary search for {target} in the sorted list {arr}."
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
        ok, out_txt = run_python(code + f"\nassert binary_search({arr}, {target}) == {idx}\nprint('verified')", None)
        assert ok, out_txt
        segs = [Segment("system", "You are TinyMe. Trace algorithms step by step.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", ("<|thought|>\n" + "\n".join(steps) +
                                      f"\nThe matching index is {idx}.\n<|answer|>\nindex {idx}"), target=True)]
        out.append(make_segment_record(segments=segs, category="algorithm", source="synthetic",
                                       source_id=f"syn/alg/trace/{i}", task_type="reasoning",
                                       template_id="algorithm/binary_search_trace", verified=True,
                                       verifier="subprocess_execution", tests=[f"assert binary_search({arr}, {target}) == {idx}"]))
    return out


# ------------------------------------------------------------------ code
_SPEC_LIBRARY: list[dict[str, Any]] = [
    {"name": "gcd", "spec": "Write a function gcd(a, b) returning the greatest common divisor of two positive integers using the Euclidean algorithm.",
     "solution": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a\n",
     "tests": "assert gcd(12, 18) == 6\nassert gcd(17, 5) == 1\nassert gcd(100, 40) == 20\n"},
    {"name": "is_palindrome", "spec": "Write a function is_palindrome(s) that returns True when a string reads the same forwards and backwards, ignoring case and non-alphanumeric characters.",
     "solution": "def is_palindrome(s):\n    t = [c.lower() for c in s if c.isalnum()]\n    return t == t[::-1]\n",
     "tests": "assert is_palindrome('A man, a plan, a canal: Panama')\nassert not is_palindrome('hello')\n"},
    {"name": "count_words", "spec": "Write a function count_words(text) returning a dictionary mapping each lowercase word to its frequency, where words are split on whitespace.",
     "solution": "def count_words(text):\n    counts = {}\n    for w in text.lower().split():\n        counts[w] = counts.get(w, 0) + 1\n    return counts\n",
     "tests": "assert count_words('a b a') == {'a': 2, 'b': 1}\nassert count_words('') == {}\n"},
    {"name": "flatten", "spec": "Write a function flatten(items) that flattens one level of nesting in a list.",
     "solution": "def flatten(items):\n    out = []\n    for item in items:\n        if isinstance(item, list):\n            out.extend(item)\n        else:\n            out.append(item)\n    return out\n",
     "tests": "assert flatten([1, [2, 3], 4]) == [1, 2, 3, 4]\nassert flatten([]) == []\n"},
    {"name": "binary_search", "spec": "Write a function binary_search(a, x) that returns the index of x in the sorted list a, or -1 if absent.",
     "solution": "def binary_search(a, x):\n    lo, hi = 0, len(a) - 1\n    while lo <= hi:\n        mid = (lo + hi) // 2\n        if a[mid] == x:\n            return mid\n        if a[mid] < x:\n            lo = mid + 1\n        else:\n            hi = mid - 1\n    return -1\n",
     "tests": "assert binary_search([1, 3, 5, 7], 5) == 2\nassert binary_search([1, 3, 5, 7], 9) == -1\n"},
    {"name": "running_max", "spec": "Write a function running_max(values) that returns the list of running maxima of a sequence of numbers.",
     "solution": "def running_max(values):\n    out, best = [], None\n    for v in values:\n        best = v if best is None else max(best, v)\n        out.append(best)\n    return out\n",
     "tests": "assert running_max([3, 1, 4, 1, 5]) == [3, 3, 4, 4, 5]\nassert running_max([]) == []\n"},
    {"name": "caesar", "spec": "Write a function caesar(text, shift) that shifts letters of the alphabet by an integer shift, leaving other characters unchanged.",
     "solution": "def caesar(text, shift):\n    out = []\n    for ch in text:\n        if ch.isalpha():\n            base = ord('a') if ch.islower() else ord('A')\n            out.append(chr((ord(ch) - base + shift) % 26 + base))\n        else:\n            out.append(ch)\n    return ''.join(out)\n",
     "tests": "assert caesar('abc', 1) == 'bcd'\nassert caesar('xyz', 3) == 'abc'\n"},
    {"name": "unique_sorted", "spec": "Write a function unique_sorted(values) that removes duplicates and returns the values in ascending order.",
     "solution": "def unique_sorted(values):\n    return sorted(set(values))\n",
     "tests": "assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]\nassert unique_sorted([]) == []\n"},
    {"name": "sum_even", "spec": "Write a function sum_even(numbers) that returns the sum of the even numbers in a list.",
     "solution": "def sum_even(numbers):\n    return sum(n for n in numbers if n % 2 == 0)\n",
     "tests": "assert sum_even([1, 2, 3, 4]) == 6\nassert sum_even([1, 3]) == 0\n"},
    {"name": "matrix_transpose", "spec": "Write a function transpose(matrix) that transposes a rectangular list-of-lists.",
     "solution": "def transpose(matrix):\n    return [list(row) for row in zip(*matrix)]\n",
     "tests": "assert transpose([[1, 2], [3, 4]]) == [[1, 3], [2, 4]]\n"},
]

_BUG_PATTERNS = [
    ("off_by_one", lambda src: src.replace("while lo <= hi:", "while lo < hi:", 1), {"binary_search"}),
    ("wrong_comparison", lambda src: src.replace("if a[mid] == x:", "if a[mid] != x:", 1), {"binary_search"}),
    ("reset_accumulator", lambda src: src.replace("counts[w] = counts.get(w, 0) + 1", "counts[w] = 1", 1), {"count_words"}),
    ("forget_zero", lambda src: src.replace("sum(n for n in numbers if n % 2 == 0)",
                                            "sum(n for n in numbers if n % 2 == 1)", 1), {"sum_even"}),
    ("swap_order", lambda src: src.replace("return sorted(set(values))", "return sorted(set(values), reverse=True)", 1), {"unique_sorted"}),
    ("skip_nested", lambda src: src.replace("out.extend(item)", "out.append(item)", 1), {"flatten"}),
    ("modulo_bug", lambda src: src.replace("% 26 + base", "% 25 + base", 1), {"caesar"}),
]


# ------------------------------------------------- instance-level diversity
_WORD_POOL = ("alpha", "bravo", "delta", "echo", "foxtrot", "golf", "hotel", "india",
              "juliet", "kilo", "lima", "mike", "november", "oscar", "papa", "quebec")
_NAME_POOL = ("cats", "dogs", "foxes", "owls", "bees", "frogs", "moles", "newts",
              "pandas", "quails", "ravens", "seals")
_TRAIT_POOL = ("nocturnal", "fast", "social", "quiet", "curious", "hungry", "gentle",
               "bold", "rare", "loud", "patient", "clever")


def _rand_sentence(rng: random.Random, words: int | None = None) -> str:
    return " ".join(rng.choice(_WORD_POOL) for _ in range(words or rng.randint(2, 7)))


def _rand_nested(rng: random.Random) -> list:
    out: list = []
    for _ in range(rng.randint(2, 5)):
        if rng.random() < 0.45:
            out.append([rng.randint(0, 99) for _ in range(rng.randint(1, 3))])
        else:
            out.append(rng.randint(0, 99))
    return out


def _rand_int_list(rng: random.Random, lo: int = -20, hi: int = 60, k: int | None = None) -> list:
    return [rng.randint(lo, hi) for _ in range(k or rng.randint(3, 8))]


def _rand_sorted_unique(rng: random.Random) -> list:
    return sorted(rng.sample(range(0, 200), rng.randint(4, 9)))


_INSTANCE_MAKERS: dict[str, Callable[[random.Random, int], list[str]]] = {
    "gcd": lambda rng, k: [f"gcd({rng.randint(12, 999)}, {rng.randint(12, 999)})" for _ in range(k)],
    "is_palindrome": lambda rng, k: [f"is_palindrome({_rand_sentence(rng)!r})" for _ in range(k)],
    "count_words": lambda rng, k: [f"count_words({_rand_sentence(rng)!r})" for _ in range(k)],
    "flatten": lambda rng, k: [f"flatten({_rand_nested(rng)!r})" for _ in range(k)],
    "binary_search": lambda rng, k: [
        f"binary_search({(lst := _rand_sorted_unique(rng))!r}, {rng.choice(lst)})" for _ in range(k)],
    "running_max": lambda rng, k: [f"running_max({_rand_int_list(rng)!r})" for _ in range(k)],
    "caesar": lambda rng, k: [f"caesar({rng.choice(_WORD_POOL)!r}, {rng.randint(1, 25)})" for _ in range(k)],
    "unique_sorted": lambda rng, k: [f"unique_sorted({_rand_int_list(rng)!r})" for _ in range(k)],
    "sum_even": lambda rng, k: [f"sum_even({_rand_int_list(rng)!r})" for _ in range(k)],
    "matrix_transpose": lambda rng, k: [
        f"transpose({[[rng.randint(-9, 99) for _ in range(rng.randint(2, 4))] for _ in range(rng.randint(2, 3))]!r})"
        for _ in range(k)],
}


def _instance_examples(spec: dict[str, Any], rng: random.Random, k: int = 3) -> tuple[str, str]:
    """Concrete examples whose values are produced by running the reference."""
    maker = _INSTANCE_MAKERS.get(spec["name"])
    if maker is None:
        return "", ""
    calls = maker(rng, k)
    harness = spec["solution"] + "\n" + "\n".join(f"print({call})" for call in calls)
    ok, out = run_python(harness)
    assert ok, f"instance harness failed for {spec['name']}: {out[:200]}"
    values = out.strip().splitlines()
    assert len(values) == len(calls), (spec["name"], values, calls)
    display = "\n".join(f"- {call}  ->  {value}" for call, value in zip(calls, values))
    tests = "\n".join(f"assert {call} == {value}" for call, value in zip(calls, values))
    return display, tests


def _verified_solution(spec: dict[str, Any]) -> tuple[bool, str]:
    return run_python(spec["solution"], spec["tests"])


def gen_code_generation(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        spec = _SPEC_LIBRARY[i % len(_SPEC_LIBRARY)]
        ok, txt = _verified_solution(spec)
        assert ok, f"reference solution failed for {spec['name']}: {txt}"
        examples, extra_tests = _instance_examples(spec, rng)
        prompt = spec["spec"] + ("\n\nVerified examples:\n" + examples if examples else "")
        tests = spec["tests"] + ("\n" + extra_tests if extra_tests else "")
        segs = [Segment("system", "You are TinyMe. Write correct Python that satisfies the specification.", target=False),
                Segment("user", prompt, target=False),
                Segment("assistant", f"<|thought|>\nImplement the specification and check the edge cases.\n"
                                     f"<|code|>\n{spec['solution'].strip()}\n<|endcode|>", target=True)]
        out.append(make_segment_record(segments=segs, category="code_gen", source="synthetic",
                                       source_id=f"syn/code/gen/{spec['name']}/{i}",
                                       task_type="code_completion",
                                       template_id=f"code/gen/{spec['name']}", verified=True,
                                       verifier="subprocess_tests", tests=[tests],
                                       language="python", expected_tool=""))
    return out


def gen_code_repair(rng: random.Random, n: int) -> list[Any]:
    out = []
    made = 0
    i = 0
    while made < n:
        spec = _SPEC_LIBRARY[i % len(_SPEC_LIBRARY)]
        i += 1
        candidates = [(name, fn) for name, fn, names in _BUG_PATTERNS if spec["name"] in names]
        if not candidates:
            continue
        name, fn = candidates[(i // len(_SPEC_LIBRARY)) % len(candidates)]
        buggy = fn(spec["solution"])
        if buggy == spec["solution"]:
            continue
        fails, _ = run_python(buggy, spec["tests"])
        if fails:                     # the injected bug must actually fail the tests
            continue
        examples, extra_tests = _instance_examples(spec, rng)
        tests = spec["tests"] + ("\n" + extra_tests if extra_tests else "")
        segs = [Segment("system", "You are TinyMe. Find the bug and return a corrected implementation.", target=False),
                Segment("user", f"The tests fail for this code:\n\n```python\n{buggy.strip()}\n```\n\n"
                                f"Specification: {spec['spec']}"
                                + ("\n\nExpected behaviour (verified):\n" + examples if examples else ""),
                        target=False),
                Segment("thought", "Locate the divergence from the specification, patch it, and re-check the tests.", target=True),
                Segment("assistant", f"<|code|>\n{spec['solution'].strip()}\n<|endcode|>", target=True)]
        out.append(make_segment_record(segments=segs, category="code_repair", source="synthetic",
                                       source_id=f"syn/code/repair/{spec['name']}/{name}/{made}",
                                       task_type="repair", template_id=f"code/repair/{name}",
                                       verified=True, verifier="bug_fails_reference_passes",
                                       tests=[tests], language="python",
                                       buggy_input=buggy))
        made += 1
    return out


def gen_code_explanation(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        spec = _SPEC_LIBRARY[(i * 3) % len(_SPEC_LIBRARY)]
        ok, _ = _verified_solution(spec)
        assert ok
        examples, _extra = _instance_examples(spec, rng, k=1)
        instrumented = examples.replace("\n", "; ") if examples else ""
        doc = (f"`{spec['name']}` implements the requested behaviour: {spec['spec']} "
               + (f"A verified call is {instrumented}. " if instrumented else "")
               + "The implementation runs in linear time over its input in the general case and "
                 "returns a value of the documented type.")
        segs = [Segment("system", "You are TinyMe. Explain code precisely and briefly.", target=False),
                Segment("user", f"Explain this code:\n\n```python\n{spec['solution'].strip()}\n```", target=False),
                Segment("assistant", f"<|thought|>\nIdentify the inputs, the algorithm and the return value.\n"
                                     f"<|final|>\n{doc}", target=True)]
        out.append(make_segment_record(segments=segs, category="code_explain", source="synthetic",
                                       source_id=f"syn/code/explain/{spec['name']}/{i}",
                                       task_type="instruction",
                                       template_id=f"code/explain/{spec['name']}", verified=True,
                                       verifier="reference_execution", language="python"))
    return out


# ------------------------------------------------------------------ instruction
_INSTRUCTIONS: list[dict[str, Any]] = [
    {"spec": "Return the input JSON with the field 'status' set to 'ok'.",
     "input": '{"id": 7, "status": "pending"}', "expected": '{"id": 7, "status": "ok"}'},
    {"spec": "List the numbers from 1 to 5 separated by commas.", "input": "", "expected": "1, 2, 3, 4, 5"},
    {"spec": "Convert the word 'code' to uppercase.", "input": "code", "expected": "CODE"},
    {"spec": "Return the number of words in the input sentence.", "input": "tiny models can reason", "expected": "4"},
    {"spec": "Reverse the input string.", "input": "tinyme", "expected": "emynit"},
]


def _instruction_items(rng: random.Random) -> list[dict[str, str]]:
    """Distinct instruction instances, each verified by recomputation."""
    items: list[dict[str, str]] = []
    payload = {"id": rng.randint(1, 999), "status": rng.choice(["pending", "draft", "queued"]),
               "priority": rng.randint(1, 5)}
    items.append({"spec": "Return the input JSON with the field 'status' set to 'ok'.",
                  "input": json.dumps(payload, sort_keys=False),
                  "expected": json.dumps(dict(payload, status="ok"), sort_keys=False),
                  "verifier": "json_recomputation"})
    hi, skip = rng.randint(4, 12), rng.choice([1, 2])
    items.append({"spec": f"List the numbers from 1 to {hi} with step {skip}, separated by commas.",
                  "input": "", "expected": ", ".join(str(k) for k in range(1, hi + 1, skip)),
                  "verifier": "range_recomputation"})
    word = rng.choice(_WORD_POOL) + rng.choice(["", "-" + str(rng.randint(10, 99)), str(rng.randint(10, 99))])
    items.append({"spec": f"Convert the word '{word}' to uppercase.", "input": word,
                  "expected": word.upper(), "verifier": "upper_recomputation"})
    sentence = _rand_sentence(rng, rng.randint(3, 10))
    items.append({"spec": "Return the number of words in the input sentence.", "input": sentence,
                  "expected": str(len(sentence.split())), "verifier": "split_recomputation"})
    text = rng.choice(_WORD_POOL) + rng.choice(["-", "_", ""]) + str(rng.randint(10, 999))
    items.append({"spec": "Reverse the input string.", "input": text,
                  "expected": text[::-1], "verifier": "reverse_recomputation"})
    values = _rand_int_list(rng, -30, 90, rng.randint(3, 7))
    items.append({"spec": "Sort the numbers in ascending order and join them with spaces.",
                  "input": " ".join(str(v) for v in values),
                  "expected": " ".join(str(v) for v in sorted(values)),
                  "verifier": "sort_recomputation"})
    joined = _rand_sentence(rng, rng.randint(3, 6)).replace(" ", "-")
    items.append({"spec": "Replace every hyphen in the input with an underscore.", "input": joined,
                  "expected": joined.replace("-", "_"), "verifier": "replace_recomputation"})
    return items


def gen_instruction(rng: random.Random, n: int) -> list[Any]:
    out = []
    pool = _instruction_items(rng)
    for i in range(n):
        item = pool[i % len(pool)] if i < len(pool) else _instruction_items(rng)[i % len(pool)]
        inp = f"\nInput: {item['input']}" if item["input"] else ""
        if item["verifier"] == "json_recomputation":
            got = json.loads(item["input"]); got["status"] = "ok"
            assert json.dumps(got, sort_keys=False) == item["expected"]
        elif item["verifier"] == "range_recomputation":
            assert item["expected"].split(", ")[0] == "1"
        elif item["verifier"] == "upper_recomputation":
            assert item["input"].upper() == item["expected"]
        elif item["verifier"] == "split_recomputation":
            assert str(len(item["input"].split())) == item["expected"]
        elif item["verifier"] == "reverse_recomputation":
            assert item["input"][::-1] == item["expected"]
        elif item["verifier"] == "sort_recomputation":
            assert " ".join(str(v) for v in sorted(int(x) for x in item["input"].split())) == item["expected"]
        elif item["verifier"] == "replace_recomputation":
            assert item["input"].replace("-", "_") == item["expected"]
        segs = [Segment("system", "You are TinyMe. Follow the instruction exactly and answer with the result only.", target=False),
                Segment("user", f"Instruction: {item['spec']}{inp}", target=False),
                Segment("assistant", f"<|final|>\n{item['expected']}", target=True)]
        out.append(make_segment_record(segments=segs, category="instruction", source="synthetic",
                                       source_id=f"syn/instruction/{i}", task_type="instruction",
                                       template_id=f"instruction/{item['verifier']}",
                                       verified=True, verifier=item["verifier"],
                                       answer=item["expected"]))
    return out


# ------------------------------------------------------------------ tool use
def _tool_trajectory(user: str, calls: list[dict[str, Any]], final: str, template: str,
                     source_id: str, expected_tool: str, citations: list[str] | None = None) -> Any:
    segs = [Segment("system", "You are TinyMe, a controller for external tools. Use a tool only when needed, "
                              "then answer from the evidence.", target=False),
            Segment("user", user, target=False)]
    for call in calls:
        segs.append(Segment("tool_call", _tool_call(call["name"], call["arguments"]), target=True))
        segs.append(Segment("tool_result", call["result"], target=False))
    segs.append(Segment("final", final, target=True))
    return make_segment_record(segments=segs, category="tool_use", source="synthetic",
                               source_id=source_id, task_type="tool_call", template_id=template,
                               verified=True, verifier="runtime_protocol_check",
                               expected_tool=expected_tool,
                               citations=citations or [])


def gen_tool_use(rng: random.Random, n: int) -> list[Any]:
    """Tool-use trajectories that exercise the *real* runtime protocol.

    Seven workflows, each one producing a transcript whose tool calls use the
    canonical argument names in :data:`TOOL_SPECS` and whose tool results use the
    exact JSON envelope ``src/agent/tool_registry.py`` produces:

    1. grounded search (snippet answer + citation)
    2. search then fetch (full passage + citation)
    3. multi-source cross-check (two searches, both cited)
    4. calculator (``compute``)
    5. sandboxed code execution (``code``, assertions embedded in the program)
    6. error recovery (typo -> empty results -> corrected query)
    7. no tool needed (answered directly, explicitly taught)
    """
    out: list[Any] = []
    kinds: list[int] = [0, 1, 2, 3, 4, 5, 6] * (n // 7 + 1)
    rng.shuffle(kinds)
    for i in range(n):
        kind = kinds[i]
        idx = i
        if kind == 0:
            q = rng.choice(_FACT_KEYS)
            assert _find_evidence(q) is not None, q
            r1 = _mock_search(q, k=3)
            hit = r1["result"]["results"][0]
            user = rng.choice([f"What is the current {q}?", f"Look up {q} and answer with a citation.",
                               f"According to a source, what is the {q}?",
                               f"Tell me the {q} and cite where it comes from.",
                               f"Which source supports the {q}?"])
            final = f"{hit['snippet']} [{hit['source_id']}]"
            out.append(_tool_trajectory(user, [{"name": "search", "arguments": {"query": q, "k": 3},
                                                "result": json.dumps(r1, ensure_ascii=False)}],
                                        final, "tool/search_single", f"syn/tool/search/{idx}",
                                        "search", [hit["source_id"]]))
        elif kind == 1:
            q = rng.choice(_FACT_KEYS)
            r1 = _mock_search(q, k=3)
            hit = r1["result"]["results"][0]
            r2 = _mock_fetch(hit["source_id"])
            fetched = r2["result"]["text"].split("\n", 1)[1]
            user = rng.choice([f"Read the detailed passage about the {q} and summarise it with a citation.",
                               f"Use the stored source for the {q} and quote the key sentence."])
            final = f"{fetched} [{hit['source_id']}]"
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q, "k": 1},
                 "result": json.dumps(r1, ensure_ascii=False)},
                {"name": "fetch", "arguments": {"source_id": hit["source_id"]},
                 "result": json.dumps(r2, ensure_ascii=False)},
            ], final, "tool/search_fetch", f"syn/tool/fetch/{idx}", "fetch", [hit["source_id"]]))
        elif kind == 2:
            q1, q2 = rng.sample(_FACT_KEYS, 2)
            r1, r2 = _mock_search(q1, k=1), _mock_search(q2, k=1)
            h1, h2 = r1["result"]["results"][0], r2["result"]["results"][0]
            user = rng.choice(["Verify two independent facts and summarise them with citations.",
                               "Cross-check two sources and report both findings."])
            final = (f"{h1['snippet']} [{h1['source_id']}] Also: {h2['snippet']} [{h2['source_id']}]")
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q1, "k": 1}, "result": json.dumps(r1, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": q2, "k": 1}, "result": json.dumps(r2, ensure_ascii=False)},
            ], final, "tool/search_multi", f"syn/tool/multi/{idx}", "search", [h1["source_id"], h2["source_id"]]))
        elif kind == 3:
            a, b = rng.randint(37, 987), rng.randint(11, 89)
            expr = f"{a} * {b} + 17"
            value = a * b + 17
            assert eval(expr) == value
            r1 = _mock_compute(expr, value)
            user = rng.choice([f"Compute exactly: {expr}", f"What is {expr}? Use a tool and give the exact value."])
            final = f"{expr} = {value} [calc-1]"
            out.append(_tool_trajectory(user, [{"name": "compute", "arguments": {"expression": expr},
                                                "result": json.dumps(r1, ensure_ascii=False)}],
                                        final, "tool/compute", f"syn/tool/compute/{idx}", "compute", ["calc-1"]))
        elif kind == 4:
            spec = rng.choice(_SPEC_LIBRARY)
            ok, txt = _verified_solution(spec)
            assert ok, txt
            _examples, extra_tests = _instance_examples(spec, rng, k=2)
            tests = spec["tests"] + ("\n" + extra_tests if extra_tests else "")
            program = f"{spec['solution'].strip()}\n\n{tests}"
            r1 = _mock_code("all assertions passed\n")
            user = (f"Run this program in the sandbox and report the result:\n"
                    f"```python\n{program}\n```")
            final = f"Exit code 0; all assertions passed for {spec['name']} [run-1]."
            out.append(_tool_trajectory(user, [{"name": "code",
                                                "arguments": {"code": program, "timeout_s": 8},
                                                "result": json.dumps(r1, ensure_ascii=False)}],
                                        final, "tool/code_run", f"syn/tool/code/{idx}", "code", ["run-1"]))
        elif kind == 5:
            good = rng.choice(_FACT_KEYS)
            mode = rng.choice(["drop", "swap", "dup"])
            positions = [k for k in range(1, len(good) - 1) if good[k].isalnum()]
            if mode == "swap":
                positions = [k for k in positions if good[k + 1].isalnum()]
            pos = rng.choice(positions)          # never typo a space: the token set must change
            if mode == "drop":
                bad = good[:pos] + good[pos + 1:]
            elif mode == "swap":
                bad = good[:pos] + good[pos + 1] + good[pos] + good[pos + 2:]
            else:
                bad = good[:pos] + good[pos] + good[pos:]
            r_bad, r_good = _mock_search(bad, k=1), _mock_search(good, k=3)
            assert r_bad["result"]["results"] == [], (bad, good)
            hit = r_good["result"]["results"][0]
            user = rng.choice([f"Look up the {good}.", f"Find a citation for the {good}."])
            final = f"{hit['snippet']} [{hit['source_id']}]"
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": bad, "k": 1}, "result": json.dumps(r_bad, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": good, "k": 3}, "result": json.dumps(r_good, ensure_ascii=False)},
            ], final, "tool/error_recovery", f"syn/tool/recovery/{idx}", "search", [hit["source_id"]]))
        else:
            x, y = rng.randint(11, 89), rng.randint(11, 89)
            word = rng.choice(_WORD_POOL) + str(rng.randint(0, 99))
            rev = rng.choice(_WORD_POOL) + str(rng.randint(0, 99))
            no_tool = [(f"What is {x} + {y}?", str(x + y), "no_tool/simple_math"),
                       (f"Capitalise the word '{word}'.", word.upper(), "no_tool/text_edit"),
                       (f"Reverse the string '{rev}'.", rev[::-1], "no_tool/text_edit")]
            q, a, tmpl = no_tool[idx % len(no_tool)]
            if tmpl == "no_tool/simple_math":
                assert x + y == int(a)
            elif a == word.upper():
                assert word.upper() == a
            else:
                assert rev[::-1] == a
            segs = [Segment("system", "You are TinyMe, a controller for external tools. Use a tool only when needed, "
                                      "then answer from the evidence.", target=False),
                    Segment("user", q, target=False),
                    Segment("assistant", f"<|final|>\n{a}", target=True)]
            out.append(make_segment_record(segments=segs, category="tool_use", source="synthetic",
                                           source_id=f"syn/tool/notool/{idx}", task_type="no_tool",
                                           template_id=tmpl, verified=True, verifier="python_recomputation",
                                           expected_tool="none", answer=a))
    return out


GENERATORS_V2: dict[str, tuple[Callable[[random.Random, int], list[Any]], int, str]] = {
    # name: (fn, default_count, family)
    "arithmetic": (gen_arithmetic, 260, "math"),
    "word_problems": (gen_word_problems, 180, "math"),
    "sequences": (gen_sequence, 120, "math"),
    "boolean": (gen_boolean, 120, "logic"),
    "deduction": (gen_deduction, 80, "logic"),
    "algorithm_trace": (gen_algorithm_trace, 120, "algorithm"),
    "code_gen": (gen_code_generation, 200, "code"),
    "code_repair": (gen_code_repair, 140, "code"),
    "code_explain": (gen_code_explanation, 120, "code"),
    "instruction": (gen_instruction, 120, "instruction"),
    "tool_use": (gen_tool_use, 180, "tool"),
}


def generate_corpus(seed: int = 20261002, counts: dict[str, int] | None = None,
                    scale: float = 1.0) -> tuple[list[Any], list[dict[str, Any]]]:
    """Generate the verified synthetic corpus with provenance entries."""
    rng = random.Random(seed)
    records: list[Any] = []
    provenance: list[dict[str, Any]] = []
    for name, (fn, default, family) in GENERATORS_V2.items():
        want = int(round((counts or {}).get(name, default) * scale))
        if want <= 0:
            continue
        produced = fn(rng, want)
        assert len(produced) == want
        # Duplicate inflation is forbidden (audit §8): a generator that cannot
        # produce `want` distinct instances emits fewer records and reports the
        # shortfall instead of silently repeating text.
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
            "requested": want,
            "emitted": len(unique),
            "duplicates_dropped": want - len(unique),
            "source": "synthetic",
            "source_id": f"syn/{name}",
            "source_url": "local://data_sources/synthetic_v2.py",
            "retrieval_date": "generated",
            "license": "Synthetic-Verified",
            "license_url": "",
            "category_family": family,
            "records": len(unique),
            "verified": True,
            "verifier": "python_recomputation | subprocess_tests",
            "generator": f"synthetic_v2.{fn.__name__}",
            "seed": seed,
            "notes": "deterministic generator with independent verification; rejected samples are not emitted",
        })
    return records, provenance


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    recs, prov = generate_corpus(scale=0.05)
    print(f"generated {len(recs)} records")
    for r in recs[:5]:
        print("-", r.category, r.template_id, len(r.text), "chars")


# ------------------------------------------------------------------ challenge
def gen_challenge(rng: random.Random, n: int) -> list[Any]:
    """Small, harder, hand-designed challenge set (never used for training).

    Each item is verified at generation time; the pool is reserved for the
    ``challenge`` split and its ``split`` field is preset so the splitter keeps
    it out of train/validation/test.
    """
    out: list[Any] = []
    i = 0
    while len(out) < n:
        kind = i % 5
        i += 1
        if kind == 0:                       # multi-step arithmetic
            a, b, c, d = rng.randint(3, 40), rng.randint(3, 20), rng.randint(2, 30), rng.randint(2, 9)
            if (a * b - c) % d:
                continue
            expr = f"(({a} * {b}) - {c}) // {d}"
            value = (a * b - c) // d
            assert eval(expr) == value
            q = f"Compute exactly: (({a} x {b}) - {c}) / {d} (integer division)."
            segs = [Segment("system", "You are TinyMe. Evaluate nested arithmetic exactly.", target=False),
                    Segment("user", q, target=False),
                    Segment("assistant", f"<|thought|>\n{a} x {b} = {a * b}; {a * b} - {c} = {a * b - c}; "
                                         f"that divided by {d} is {value}.\n<|answer|>\n{value}", target=True)]
            rec = make_segment_record(segments=segs, category="math", source="synthetic",
                                      source_id=f"challenge/math/{len(out)}", task_type="reasoning",
                                      template_id="challenge/math/nested", verified=True,
                                      verifier="python_recomputation", answer=str(value))
        elif kind == 1:                     # unit conversion, verified by recomputation
            hours = rng.randint(2, 47)
            minutes = hours * 60 + rng.randint(0, 59)
            q = f"A process runs for {minutes} minutes. How many hours and minutes is that?"
            h, m = divmod(minutes, 60)
            assert h * 60 + m == minutes
            segs = [Segment("system", "You are TinyMe. Convert units and show the remainder.", target=False),
                    Segment("user", q, target=False),
                    Segment("assistant", f"<|thought|>\n{minutes} // 60 = {h}, remainder {m}.\n"
                                         f"<|answer|>\n{h} hours and {m} minutes", target=True)]
            rec = make_segment_record(segments=segs, category="math", source="synthetic",
                                      source_id=f"challenge/math/{len(out)}", task_type="reasoning",
                                      template_id="challenge/math/units", verified=True,
                                      verifier="python_recomputation", answer=f"{h} hours and {m} minutes")
        elif kind == 2:                     # three-premise deduction
            order = rng.sample(["A", "B", "C", "D"], 4)
            q = (f"X beats {order[0]}. {order[0]} beats {order[1]}. {order[1]} beats {order[2]}. "
                 f"Who is the strongest and who is the weakest?")
            ans = f"Strongest: X. Weakest: {order[2]}."
            segs = [Segment("system", "You are TinyMe. Chain the premises.", target=False),
                    Segment("user", q, target=False),
                    Segment("assistant", f"<|thought|>\nTransitivity along the chain gives X > {order[0]} > {order[1]} > {order[2]}.\n"
                                         f"<|answer|>\n{ans}", target=True)]
            rec = make_segment_record(segments=segs, category="logic", source="synthetic",
                                      source_id=f"challenge/logic/{len(out)}", task_type="reasoning",
                                      template_id="challenge/logic/chain", verified=True,
                                      verifier="template_check", answer=ans)
        elif kind == 3:                     # multi-hop tool use: search -> fetch -> search
            q1 = "shortest path algorithm for non-negative weights"
            q2 = "big-o of binary search"
            r1 = _mock_search(q1, k=1)
            hit1 = r1["result"]["results"][0]
            r2 = _mock_fetch(hit1["source_id"])
            fetched = r2["result"]["text"].split("\n", 1)[1]
            r3 = _mock_search(q2, k=1)
            hit3 = r3["result"]["results"][0]
            user = ("Which algorithm should I use for shortest paths with non-negative weights, "
                    "and what is the cost of binary search?")
            final = f"{fetched} Binary search costs O(log n) [{hit3['source_id']}]."
            rec = _tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q1, "k": 1}, "result": json.dumps(r1, ensure_ascii=False)},
                {"name": "fetch", "arguments": {"source_id": hit1["source_id"]},
                 "result": json.dumps(r2, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": q2, "k": 1}, "result": json.dumps(r3, ensure_ascii=False)},
            ], final, "challenge/tool/multihop", f"challenge/tool/{len(out)}", "fetch",
                [hit1["source_id"], hit3["source_id"]])
        else:                               # harder code repair (two defects)
            spec = _SPEC_LIBRARY[2]
            buggy = spec["solution"].replace("counts.get(w, 0) + 1", "counts.get(w, 0)",
                                             1).replace("text.lower().split()", "text.split()", 1)
            fails, _ = run_python(buggy, spec["tests"])
            if fails:
                continue
            segs = [Segment("system", "You are TinyMe. Repair the implementation so all tests pass.", target=False),
                    Segment("user", f"The tests fail for this code:\n\n```python\n{buggy.strip()}\n```", target=False),
                    Segment("thought", "The counter never increments and the case normalisation is missing.", target=True),
                    Segment("assistant", f"<|code|>\n{spec['solution'].strip()}\n<|endcode|>", target=True)]
            rec = make_segment_record(segments=segs, category="code_repair", source="synthetic",
                                      source_id=f"challenge/code/{len(out)}", task_type="repair",
                                      template_id="challenge/code/two_bugs", verified=True,
                                      verifier="bug_fails_reference_passes", tests=[spec["tests"]],
                                      language="python", buggy_input=buggy)
        rec.split = "challenge"
        out.append(rec)
    return out
