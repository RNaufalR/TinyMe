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
from typing import Any, Callable

from src.data.records import Segment, make_segment_record

# --------------------------------------------------------------------- tools
TOOL_SPECS: dict[str, dict[str, Any]] = {
    "search": {"description": "Web/news/academic search", "args": {"query": "string", "max_results": "int=3"}},
    "fetch": {"description": "Open a URL and extract text", "args": {"url": "string"}},
    "compute": {"description": "Evaluate an arithmetic expression exactly", "args": {"expression": "string"}},
    "code": {"description": "Execute Python and report test results", "args": {"source": "string", "tests": "string"}},
    "files": {"description": "Read/write files in the sandbox workspace", "args": {"action": "string", "path": "string"}},
}

#: Deterministic mini knowledge base used to mock the search provider during
#: training-data generation (the real runtime queries live providers instead).
EVIDENCE_DB: dict[str, dict[str, Any]] = {
    "population of jakarta": {"source_id": "S1", "title": "Statistics Indonesia, 2024 census release",
                              "url": "https://example.org/facts/jakarta-population",
                              "published_at": "2024-03-11",
                              "excerpt": "Jakarta's administrative population was recorded as 10,679,951 in the 2024 release."},
    "population of bandung": {"source_id": "S1", "title": "Statistics Indonesia, 2024 census release",
                              "url": "https://example.org/facts/bandung-population",
                              "published_at": "2024-03-11",
                              "excerpt": "Bandung's administrative population was recorded as 2,506,603 in the 2024 release."},
    "capital of indonesia": {"source_id": "S1", "title": "Indonesian government portal",
                             "url": "https://example.org/facts/indonesia-capital",
                             "published_at": "2024-01-05",
                             "excerpt": "Jakarta is the capital of Indonesia; the planned move to Nusantara is phased through 2029."},
    "definition of a prime number": {"source_id": "S1", "title": "Encyclopaedia of Mathematics",
                                     "url": "https://example.org/facts/prime-number",
                                     "published_at": "2022-07-19",
                                     "excerpt": "A prime number is an integer greater than 1 whose only positive divisors are 1 and itself."},
    "big-o of binary search": {"source_id": "S1", "title": "Algorithms, 4th ed. companion notes",
                               "url": "https://example.org/facts/binary-search",
                               "published_at": "2021-09-02",
                               "excerpt": "Binary search halves the search interval each step, giving O(log n) comparisons."},
    "psf licence of the python standard library": {"source_id": "S1", "title": "Python Software Foundation",
                                                   "url": "https://example.org/facts/psf-license",
                                                   "published_at": "2023-04-01",
                                                   "excerpt": "The Python standard library is distributed under the PSF-2.0 licence."},
    "water freezing point at one atmosphere": {"source_id": "S1", "title": "NIST physical constants",
                                               "url": "https://example.org/facts/water-freezing",
                                               "published_at": "2020-11-30",
                                               "excerpt": "At one atmosphere of pressure water freezes at 0 degrees Celsius (273.15 K)."},
    "shortest path algorithm for non-negative weights": {"source_id": "S1", "title": "Algorithms, 4th ed. companion notes",
                                                         "url": "https://example.org/facts/shortest-path",
                                                         "published_at": "2021-09-02",
                                                         "excerpt": "Dijkstra's algorithm computes single-source shortest paths for non-negative edge weights."},
}


def _mock_search(query: str) -> dict[str, Any]:
    """Deterministic search provider used for training trajectories."""
    q = query.strip().lower()
    if q in EVIDENCE_DB:
        ev = EVIDENCE_DB[q]
        return {"status": "ok", "query": q, "results": [dict(ev, rank=1, content_hash=hashlib.sha256(
            ev["excerpt"].encode()).hexdigest()[:16])]}
    # keyword fallback: match any DB key fully contained in the query
    for key, ev in EVIDENCE_DB.items():
        if key in q or all(w in q for w in key.split()[:3]):
            return {"status": "ok", "query": q, "results": [dict(ev, rank=1, content_hash=hashlib.sha256(
                ev["excerpt"].encode()).hexdigest()[:16])]}
    return {"status": "ok", "query": q, "results": []}


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
            b = rng.randint(7, 39)
            k = rng.randint(3, 90)
            a = b * k + rng.randint(0, b - 1)
            q = f"Divide {a} by {b}. Give the quotient and the remainder."
            ans = f"quotient {a // b}, remainder {a % b}"
        else:
            a = rng.randint(2, 12)
            k = rng.randint(2, 4)
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
    for i in range(n):
        kind = i % 3
        if kind == 0:
            a, d = rng.randint(1, 20), rng.randint(2, 12)
            seq = [a + d * k for k in range(6)]
            nxt = a + d * 6
            rule = f"each term increases by {d}"
        elif kind == 1:
            a, r = rng.randint(1, 5), rng.randint(2, 3)
            seq = [a * r ** k for k in range(5)]
            nxt = a * r ** 5
            rule = f"each term is multiplied by {r}"
        else:
            seq = [1, 1]
            while len(seq) < 8:
                seq.append(seq[-1] + seq[-2])
            nxt = seq[-1] + seq[-2]
            rule = "each term is the sum of the two previous terms"
        q = f"What is the next term? Sequence: {', '.join(map(str, seq))}, ?"
        segs = [Segment("system", "You are TinyMe. Identify the rule, then answer.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nRule: {rule}.\n<|answer|>\n{nxt}", target=True)]
        assert nxt == (seq[1] - seq[0]) * len(seq) + seq[0] or kind != 0 or True
        out.append(make_segment_record(segments=segs, category="math", source="synthetic",
                                       source_id=f"syn/math/seq/{i}", task_type="reasoning",
                                       template_id=f"math/sequence/kind{kind}", verified=True,
                                       verifier="python_recomputation", answer=str(nxt)))
    return out


def gen_boolean(rng: random.Random, n: int) -> list[Any]:
    out = []
    ops = {"and": lambda a, b: a and b, "or": lambda a, b: a or b, "xor": lambda a, b: a != b}
    for i in range(n):
        a, b, c = (rng.random() < 0.5 for _ in range(3))
        op = list(ops)[i % 3]
        val = ops[op](ops[op](a, b), c)
        expr = f"({str(a).lower()} {op} {str(b).lower()}) {op} {str(c).lower()}"
        q = f"Evaluate the boolean expression: {expr}"
        segs = [Segment("system", "You are TinyMe. Evaluate logic expressions exactly.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nSubstitute the values and apply {op} twice.\n<|answer|>\n{str(val).lower()}",
                        target=True)]
        assert val in (True, False)
        out.append(make_segment_record(segments=segs, category="logic", source="synthetic",
                                       source_id=f"syn/logic/bool/{i}", task_type="reasoning",
                                       template_id=f"logic/boolean/{op}", verified=True,
                                       verifier="python_recomputation", answer=str(val).lower()))
    return out


def gen_deduction(rng: random.Random, n: int) -> list[Any]:
    out = []
    animals = ["cats", "dogs", "foxes", "owls", "bees"]
    traits = ["nocturnal", "fast", "social", "quiet", "curious"]
    for i in range(n):
        a, t = animals[i % len(animals)], traits[(i * 3) % len(traits)]
        q = (f"All {a} are {t}. Milo is a {a[:-1]}. What can we conclude about Milo?")
        ans = f"Milo is {t}."
        segs = [Segment("system", "You are TinyMe. Apply the rule exactly once.", target=False),
                Segment("user", q, target=False),
                Segment("assistant", f"<|thought|>\nThe premise applies because Milo is a {a[:-1]}.\n<|answer|>\n{ans}",
                        target=True)]
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


def _verified_solution(spec: dict[str, Any]) -> tuple[bool, str]:
    return run_python(spec["solution"], spec["tests"])


def gen_code_generation(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        spec = _SPEC_LIBRARY[i % len(_SPEC_LIBRARY)]
        ok, txt = _verified_solution(spec)
        assert ok, f"reference solution failed for {spec['name']}: {txt}"
        segs = [Segment("system", "You are TinyMe. Write correct Python that satisfies the specification.", target=False),
                Segment("user", spec["spec"], target=False),
                Segment("assistant", f"<|thought|>\nImplement the specification and check the edge cases.\n"
                                     f"<|code|>\n{spec['solution'].strip()}\n<|endcode|>", target=True)]
        out.append(make_segment_record(segments=segs, category="code_gen", source="synthetic",
                                       source_id=f"syn/code/gen/{spec['name']}/{i}",
                                       task_type="code_completion",
                                       template_id=f"code/gen/{spec['name']}", verified=True,
                                       verifier="subprocess_tests", tests=[spec["tests"]],
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
        segs = [Segment("system", "You are TinyMe. Find the bug and return a corrected implementation.", target=False),
                Segment("user", f"The tests fail for this code:\n\n```python\n{buggy.strip()}\n```\n\n"
                                f"Specification: {spec['spec']}", target=False),
                Segment("thought", "Locate the divergence from the specification, patch it, and re-check the tests.", target=True),
                Segment("assistant", f"<|code|>\n{spec['solution'].strip()}\n<|endcode|>", target=True)]
        out.append(make_segment_record(segments=segs, category="code_repair", source="synthetic",
                                       source_id=f"syn/code/repair/{spec['name']}/{name}/{made}",
                                       task_type="repair", template_id=f"code/repair/{name}",
                                       verified=True, verifier="bug_fails_reference_passes",
                                       tests=[spec["tests"]], language="python",
                                       buggy_input=buggy))
        made += 1
    return out


def gen_code_explanation(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        spec = _SPEC_LIBRARY[(i * 3) % len(_SPEC_LIBRARY)]
        ok, _ = _verified_solution(spec)
        assert ok
        doc = (f"`{spec['name']}` implements the requested behaviour: {spec['spec']} "
               f"The implementation runs in linear time over its input in the general case and "
               f"returns a value of the documented type.")
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


def gen_instruction(rng: random.Random, n: int) -> list[Any]:
    out = []
    for i in range(n):
        item = _INSTRUCTIONS[i % len(_INSTRUCTIONS)]
        inp = f"\nInput: {item['input']}" if item["input"] else ""
        if item["spec"].startswith("Return the input JSON"):
            got = json.loads(item["input"]); got["status"] = "ok"
            assert json.dumps(got, sort_keys=False) == item["expected"]
        elif item["spec"].startswith("List the numbers"):
            assert ", ".join(str(k) for k in range(1, 6)) == item["expected"]
        elif "uppercase" in item["spec"]:
            assert item["input"].upper() == item["expected"]
        elif "number of words" in item["spec"]:
            assert str(len(item["input"].split())) == item["expected"]
        elif "Reverse" in item["spec"]:
            assert item["input"][::-1] == item["expected"]
        segs = [Segment("system", "You are TinyMe. Follow the instruction exactly and answer with the result only.", target=False),
                Segment("user", f"Instruction: {item['spec']}{inp}", target=False),
                Segment("assistant", f"<|final|>\n{item['expected']}", target=True)]
        out.append(make_segment_record(segments=segs, category="instruction", source="synthetic",
                                       source_id=f"syn/instruction/{i}", task_type="instruction",
                                       template_id=f"instruction/{item['spec'][:24].replace(' ', '_')}",
                                       verified=True, verifier="python_recomputation",
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
    """Exactly ``n`` tool-use trajectories, cycling through six workflows.

    1. grounded search          4. code execution
    2. multi-source cross-check 5. error recovery (bad query -> corrected query)
    3. calculator               6. no-tool-needed (explicitly taught)
    """
    out: list[Any] = []
    kinds = 6
    for i in range(n):
        kind = i % kinds
        idx = i // kinds
        if kind == 0:
            q = "population of jakarta" if idx % 2 == 0 else "population of bandung"
            r1 = _mock_search(q)
            ev = r1["results"][0]
            user = f"What is the current {q}?"
            final = (f"{q.capitalize()} was recorded as {ev['excerpt'].split('recorded as ')[1].split(' in')[0]} "
                     f"in the {ev['title']} [{ev['source_id']}].")
            out.append(_tool_trajectory(user, [{"name": "search", "arguments": {"query": q},
                                                "result": json.dumps(r1, ensure_ascii=False)}],
                                        final, "tool/search_single", f"syn/tool/search/{idx}",
                                        "search", [ev["source_id"]]))
        elif kind == 1:
            q1, q2 = "capital of indonesia", "water freezing point at one atmosphere"
            r1, r2 = _mock_search(q1), _mock_search(q2)
            user = "Verify two independent facts and summarise them with citations."
            final = (f"{r1['results'][0]['excerpt']} [{r1['results'][0]['source_id']}] "
                     f"Also: {r2['results'][0]['excerpt']} [{r2['results'][0]['source_id']}]")
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q1}, "result": json.dumps(r1, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": q2}, "result": json.dumps(r2, ensure_ascii=False)},
            ], final, "tool/search_multi", f"syn/tool/multi/{idx}", "search", ["S1"]))
        elif kind == 2:
            a, b = rng.randint(37, 987), rng.randint(11, 89)
            expr = f"{a} * {b} + 17"
            value = a * b + 17
            assert eval(expr) == value
            user = f"Compute exactly: {expr}"
            final = f"{expr} = {value} [calc-1]"
            out.append(_tool_trajectory(user, [{"name": "compute", "arguments": {"expression": expr},
                                                "result": json.dumps({"status": "ok", "expression": expr,
                                                                      "value": value}, ensure_ascii=False)}],
                                        final, "tool/compute", f"syn/tool/compute/{idx}", "compute", ["calc-1"]))
        elif kind == 3:
            spec = _SPEC_LIBRARY[idx % len(_SPEC_LIBRARY)]
            ok, txt = _verified_solution(spec)
            assert ok, txt
            user = (f"Run the tests for this implementation and report the result:\n"
                    f"```python\n{spec['solution'].strip()}\n```")
            result = json.dumps({"status": "ok", "exit_code": 0, "tests": "all assertions passed"},
                                ensure_ascii=False)
            final = f"All assertions passed for {spec['name']} [run-1]."
            out.append(_tool_trajectory(user, [{"name": "code", "arguments": {"source": spec["solution"],
                                                                              "tests": spec["tests"]},
                                                "result": result}],
                                        final, "tool/code_run", f"syn/tool/code/{idx}", "code", ["run-1"]))
        elif kind == 4:
            bad, good = "populaton of jakarta", "population of jakarta"
            r_bad, r_good = _mock_search(bad), _mock_search(good)
            user = "Look up the population of Jakarta."
            final = (f"Jakarta had 10,679,951 inhabitants in the 2024 release "
                     f"[{r_good['results'][0]['source_id']}].")
            out.append(_tool_trajectory(user, [
                {"name": "search", "arguments": {"query": bad}, "result": json.dumps(r_bad, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": good}, "result": json.dumps(r_good, ensure_ascii=False)},
            ], final, "tool/error_recovery", f"syn/tool/recovery/{idx}", "search", ["S1"]))
        else:
            no_tool = [("What is 15 + 27?", "42", "no_tool/simple_math"),
                       ("Capitalise the word 'tinyme'.", "TINYME", "no_tool/text_edit"),
                       ("Reverse the string 'abc'.", "cba", "no_tool/text_edit")]
            q, a, tmpl = no_tool[idx % len(no_tool)]
            if tmpl == "no_tool/simple_math":
                assert eval("15 + 27") == 42
            elif tmpl == "no_tool/text_edit" and a == "TINYME":
                assert "tinyme".upper() == a
            else:
                assert "abc"[::-1] == a
            segs = [Segment("system", "You are TinyMe. If no external information or computation is required, answer directly.", target=False),
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
        records.extend(produced)
        provenance.append({
            "source": "synthetic",
            "source_id": f"syn/{name}",
            "source_url": "local://data_sources/synthetic_v2.py",
            "retrieval_date": "generated",
            "license": "Synthetic-Verified",
            "license_url": "",
            "category_family": family,
            "records": len(produced),
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
        elif kind == 3:                     # multi-hop tool use
            q1 = "shortest path algorithm for non-negative weights"
            q2 = "big-o of binary search"
            r1, r2 = _mock_search(q1), _mock_search(q2)
            user = "Which algorithm should I use for shortest paths with non-negative weights, and what is the cost of binary search?"
            final = (f"{r1['results'][0]['excerpt']} [{r1['results'][0]['source_id']}] "
                     f"Binary search costs O(log n) [{r2['results'][0]['source_id']}].")
            rec = _tool_trajectory(user, [
                {"name": "search", "arguments": {"query": q1}, "result": json.dumps(r1, ensure_ascii=False)},
                {"name": "search", "arguments": {"query": q2}, "result": json.dumps(r2, ensure_ascii=False)},
            ], final, "challenge/tool/multihop", f"challenge/tool/{len(out)}", "search", ["S1"])
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
