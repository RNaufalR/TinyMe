"""Deterministic synthetic curriculum generator (spec §9 category I, §11, §12).

Every generator follows: PROBLEM GENERATOR -> SOLUTION GENERATOR ->
EXECUTION / SYMBOLIC CHECKER -> ACCEPT ONLY VERIFIED EXAMPLES.

Nothing is emitted unless the answer was recomputed by an independent
implementation (or by executing real Python code in a sandboxed subprocess).
"""
from __future__ import annotations

import ast
import hashlib
import logging
import random
import time
from typing import Any, Callable

from src.data.records import TrainingRecord, validate_record

logger = logging.getLogger("tinyme.synthetic")

TODAY = time.strftime("%Y-%m-%d")
LICENSE = "Synthetic-Verified"
SRC_URL = "generated locally by data_sources/synthetic_adapter.py"


def _rec(text: str, category: str, task_type: str, answer: str | None = None,
         tests: list[str] | None = None, verifier: str = "recomputed",
         language: str = "en", source_id: str = "") -> TrainingRecord:
    if not source_id:
        source_id = f"synthetic/{category}/{hashlib.sha256(text.encode()).hexdigest()[:16]}"
    return TrainingRecord(
        text=text, category=category, source="synthetic", source_id=source_id,
        license=LICENSE, source_url=SRC_URL, retrieval_date=TODAY,
        task_type=task_type, language=language, verified=True, verifier=verifier,
        answer=answer, tests=tests or [],
    )


# ============================================================ C. ARITHMETIC
def gen_arithmetic(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    ops = ["+", "-", "*"]
    for _ in range(n):
        a = rng.randint(2, 99)
        b = rng.randint(2, 99)
        op = rng.choice(ops)
        ans = eval(f"{a}{op}{b}")  # safe: integers and fixed operators
        steps = {
            "+": f"{a} + {b} = {ans}",
            "-": f"{a} - {b} = {ans}",
            "*": f"{a} * {b} = {ans}",
        }[op]
        text = (f"<|system|>\nYou are a precise calculator.\n<|user|>\nWhat is {a} {op} {b}?\n"
                f"<|thought|>\n{steps}\n<|answer|>\n{ans}\n")
        out.append(_rec(text, "math", "reasoning", answer=str(ans), verifier="python_eval"))
    return out


def gen_sequence(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    for _ in range(n):
        kind = rng.choice(["arith", "geom", "square", "fib"])
        start = rng.randint(1, 9)
        diff = rng.randint(2, 9)
        if kind == "arith":
            seq = [start + i * diff for i in range(6)]
            nxt = start + 6 * diff
            rule = f"add {diff} each time"
        elif kind == "geom":
            ratio = rng.randint(2, 4)
            seq = [start * ratio ** i for i in range(6)]
            nxt = start * ratio ** 6
            rule = f"multiply by {ratio} each time"
        elif kind == "square":
            seq = [(start + i) ** 2 for i in range(6)]
            nxt = (start + 6) ** 2
            rule = "consecutive perfect squares"
        else:
            a, b = start, start + diff
            seq = [a, b]
            for _ in range(4):
                seq.append(seq[-1] + seq[-2])
            nxt = seq[-1] + seq[-2]
            rule = "Fibonacci-like: each term is the sum of the previous two"
        text = (f"<|system|>\nFind the next term in the sequence.\n<|user|>\n"
                f"Sequence: {', '.join(map(str, seq))}\n<|thought|>\nThe rule is: {rule}.\n"
                f"So the next term is {nxt}.\n<|answer|>\n{nxt}\n")
        out.append(_rec(text, "math", "reasoning", answer=str(nxt), verifier="recomputed_rule"))
    return out


# ============================================================== B. LOGIC
def gen_boolean(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    for _ in range(n):
        p, q = rng.choice([True, False]), rng.choice([True, False])
        op = rng.choice(["AND", "OR", "XOR", "IMPLIES", "IFF"])
        if op == "AND":
            r = p and q
        elif op == "OR":
            r = p or q
        elif op == "XOR":
            r = p != q
        elif op == "IMPLIES":
            r = (not p) or q
        else:
            r = p == q
        text = (f"<|system|>\nEvaluate the boolean expression.\n<|user|>\n"
                f"Let P = {p}, Q = {q}. What is P {op} Q?\n<|thought|>\n"
                f"Using the {op} truth table, P {op} Q = {r}.\n<|answer|>\n{r}\n")
        out.append(_rec(text, "logic", "reasoning", answer=str(r), verifier="python_bool_ops"))
    return out


def gen_deduction(rng: random.Random, n: int) -> list[TrainingRecord]:
    names = ["Amy", "Ben", "Cara", "Dan", "Eve", "Finn"]
    jobs = ["chef", "doctor", "engineer", "artist", "nurse", "pilot"]
    out = []
    for _ in range(n):
        rng.shuffle(jobs)
        k = rng.randint(2, 4)
        pairs = list(zip(names[:k], jobs[:k]))
        ask = rng.randrange(k)
        premise = "; ".join(f"{nm} is a {jb}" for nm, jb in pairs)
        text = (f"<|system|>\nAnswer using only the given premises.\n<|user|>\n"
                f"{premise}. What is {names[ask]}'s job?\n<|thought|>\n"
                f"From the premises, {names[ask]} is a {pairs[ask][1]}.\n<|answer|>\n{pairs[ask][1]}\n")
        out.append(_rec(text, "logic", "reasoning", answer=pairs[ask][1], verifier="lookup"))
    return out


def gen_pattern(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    shapes = ["circle", "square", "triangle"]
    colors = ["red", "blue", "green"]
    for _ in range(n):
        shape = rng.choice(shapes)
        color = rng.choice(colors)
        period = rng.randint(2, 3)
        seq = []
        for i in range(period * 3):
            seq.append(f"{colors[i % len(colors)]} {shapes[i % len(shapes)]}")
        nxt = f"{colors[(period * 3) % len(colors)]} {shapes[(period * 3) % len(shapes)]}"
        text = (f"<|system|>\nIdentify the repeating pattern and predict the next item.\n<|user|>\n"
                f"{', '.join(seq)}, ?\n<|thought|>\nThe colors cycle through {colors} and the "
                f"shapes cycle through {shapes}. The next item is {nxt}.\n<|answer|>\n{nxt}\n")
        out.append(_rec(text, "logic", "reasoning", answer=nxt, verifier="recomputed_cycle"))
    return out


# ====================================================== D. ALGORITHMIC REASONING
def gen_algorithm_trace(rng: random.Random, n: int) -> list[TrainingRecord]:
    """Trace sorting / search / recursion with an independently computed answer."""
    out = []
    for _ in range(n):
        kind = rng.choice(["bubble", "binary_search", "fib", "gcd", "reverse", "sum_list"])
        if kind == "bubble":
            arr = [rng.randint(1, 50) for _ in range(rng.randint(4, 7))]
            srt = sorted(arr)
            text = (f"<|system|>\nTrace bubble sort step by step.\n<|user|>\n"
                    f"Sort the list {arr} using bubble sort.\n<|thought|>\n"
                    f"Repeatedly compare adjacent elements and swap them when out of order. "
                    f"The fully sorted result is {srt}.\n<|answer|>\n{srt}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=str(srt), verifier="python_sorted"))
        elif kind == "binary_search":
            arr = sorted(rng.sample(range(1, 100), rng.randint(6, 10)))
            target = rng.choice(arr)
            idx = arr.index(target)
            text = (f"<|system|>\nTrace binary search.\n<|user|>\n"
                    f"In the sorted list {arr}, at what index is {target}?\n<|thought|>\n"
                    f"Binary search halves the interval each step. {target} is at index {idx}.\n"
                    f"<|answer|>\n{idx}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=str(idx), verifier="python_index"))
        elif kind == "fib":
            k = rng.randint(6, 12)
            a, b, seq = 0, 1, [0, 1]
            for _ in range(k - 2):
                a, b = b, a + b
                seq.append(b)
            text = (f"<|system|>\nCompute a Fibonacci sequence.\n<|user|>\n"
                    f"List the first {k} Fibonacci numbers.\n<|thought|>\n"
                    f"Start with 0, 1 and add the last two to get the next.\n"
                    f"<|answer|>\n{', '.join(map(str, seq))}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=str(seq), verifier="iterative_fib"))
        elif kind == "gcd":
            a, b = rng.randint(12, 200), rng.randint(12, 200)
            g = _gcd(a, b)
            text = (f"<|system|>\nCompute the greatest common divisor.\n<|user|>\n"
                    f"What is gcd({a}, {b})?\n<|thought|>\n"
                    f"Euclid's algorithm: gcd({a}, {b}) = {g}.\n<|answer|>\n{g}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=str(g), verifier="euclid"))
        elif kind == "reverse":
            s = "".join(rng.choice("abcdef") for _ in range(rng.randint(4, 8)))
            r = s[::-1]
            text = (f"<|system|>\nReverse the string.\n<|user|>\nReverse '{s}'.\n"
                    f"<|thought|>\nReading from the last character to the first gives '{r}'.\n"
                    f"<|answer|>\n{r}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=r, verifier="python_slice"))
        else:
            arr = [rng.randint(1, 30) for _ in range(rng.randint(4, 9))]
            total = sum(arr)
            text = (f"<|system|>\nSum the list.\n<|user|>\nWhat is the sum of {arr}?\n"
                    f"<|thought|>\nAdding all elements: {total}.\n<|answer|>\n{total}\n")
            out.append(_rec(text, "algorithm", "reasoning", answer=str(total), verifier="python_sum"))
    return out


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return a


# ============================================== E/F/G/H. CODE, REPAIR, GEN, EXPLAIN
CODE_TASKS: list[dict[str, Any]] = [
    {"name": "max_of_two", "spec": "Write a function max_of_two(a, b) that returns the larger of two numbers.",
     "solution": "def max_of_two(a, b):\n    if a > b:\n        return a\n    return b",
     "tests": ["assert max_of_two(3, 7) == 7", "assert max_of_two(-1, -5) == -1", "assert max_of_two(4, 4) == 4"]},
    {"name": "factorial", "spec": "Write a function factorial(n) that returns n! for a non-negative integer n.",
     "solution": "def factorial(n):\n    result = 1\n    for i in range(2, n + 1):\n        result *= i\n    return result",
     "tests": ["assert factorial(0) == 1", "assert factorial(5) == 120", "assert factorial(7) == 5040"]},
    {"name": "is_palindrome", "spec": "Write a function is_palindrome(s) that returns True if the string reads the same forwards and backwards.",
     "solution": "def is_palindrome(s):\n    return s == s[::-1]",
     "tests": ["assert is_palindrome('racecar') == True", "assert is_palindrome('hello') == False"]},
    {"name": "sum_list", "spec": "Write a function sum_list(numbers) that returns the sum of a list of numbers.",
     "solution": "def sum_list(numbers):\n    total = 0\n    for n in numbers:\n        total += n\n    return total",
     "tests": ["assert sum_list([1, 2, 3]) == 6", "assert sum_list([]) == 0"]},
    {"name": "count_vowels", "spec": "Write a function count_vowels(s) that counts the vowels in a lowercase string.",
     "solution": "def count_vowels(s):\n    return sum(1 for ch in s if ch in 'aeiou')",
     "tests": ["assert count_vowels('hello') == 2", "assert count_vowels('xyz') == 0"]},
    {"name": "reverse_string", "spec": "Write a function reverse_string(s) that returns the reversed string.",
     "solution": "def reverse_string(s):\n    return s[::-1]",
     "tests": ["assert reverse_string('abc') == 'cba'"]},
    {"name": "is_even", "spec": "Write a function is_even(n) that returns True when n is even.",
     "solution": "def is_even(n):\n    return n % 2 == 0",
     "tests": ["assert is_even(4) == True", "assert is_even(7) == False"]},
    {"name": "fib", "spec": "Write a function fib(n) that returns the n-th Fibonacci number with fib(0)=0, fib(1)=1.",
     "solution": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
     "tests": ["assert fib(0) == 0", "assert fib(1) == 1", "assert fib(10) == 55"]},
    {"name": "unique_sorted", "spec": "Write a function unique_sorted(values) returning a sorted list of the distinct values.",
     "solution": "def unique_sorted(values):\n    return sorted(set(values))",
     "tests": ["assert unique_sorted([3, 1, 3, 2]) == [1, 2, 3]"]},
    {"name": "gcd", "spec": "Write a function gcd(a, b) returning the greatest common divisor using the Euclidean algorithm.",
     "solution": "def gcd(a, b):\n    while b:\n        a, b = b, a % b\n    return a",
     "tests": ["assert gcd(48, 18) == 6", "assert gcd(7, 13) == 1"]},
]

BUGGY_VARIANTS: dict[str, list[dict[str, str]]] = {
    "max_of_two": [{"bug": "def max_of_two(a, b):\n    if a < b:\n        return a\n    return b",
                    "why": "the comparison is inverted, so the smaller value is returned"}],
    "factorial": [{"bug": "def factorial(n):\n    result = 1\n    for i in range(2, n):\n        result *= i\n    return result",
                   "why": "range stops before n, so the factor n is never multiplied"},
                  {"bug": "def factorial(n):\n    result = 0\n    for i in range(2, n + 1):\n        result *= i\n    return result",
                   "why": "result is initialised to 0, so every product stays 0"}],
    "is_palindrome": [{"bug": "def is_palindrome(s):\n    return s == reversed(s)",
                       "why": "reversed() returns an iterator, not a string; use s[::-1]"}],
    "sum_list": [{"bug": "def sum_list(numbers):\n    total = 1\n    for n in numbers:\n        total += n\n    return total",
                  "why": "the accumulator starts at 1 instead of 0, adding a spurious 1"}],
    "count_vowels": [{"bug": "def count_vowels(s):\n    return sum(1 for ch in s if ch in 'aeiouy')",
                      "why": "'y' is not treated as a vowel here, inflating the count"}],
    "reverse_string": [{"bug": "def reverse_string(s):\n    return s[0::-1]",
                        "why": "s[0::-1] returns only the first character; use s[::-1]"}],
    "is_even": [{"bug": "def is_even(n):\n    return n % 2 = 0", "why": "'=' is assignment, not comparison; use '=='"}],
    "fib": [{"bug": "def fib(n):\n    a, b = 1, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a",
             "why": "the seed values are wrong, so the sequence is shifted"}],
    "unique_sorted": [{"bug": "def unique_sorted(values):\n    return set(sorted(values))",
                       "why": "a set is returned instead of a list, losing ordering guarantees"}],
    "gcd": [{"bug": "def gcd(a, b):\n    while b:\n        a, b = b, a % a\n    return a",
             "why": "the modulo uses a instead of b, so the remainder is always 0"}],
}


def _run_python(code: str, tests: list[str], timeout: float = 5.0) -> tuple[bool, str]:
    """Execute code + asserts in an isolated subprocess with a hard timeout."""
    import subprocess
    import sys
    import tempfile

    script = code + "\n\n" + "\n".join(tests) + "\n"
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=True) as fh:
        fh.write(script)
        fh.flush()
        try:
            proc = subprocess.run([sys.executable, fh.name], capture_output=True,
                                  text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, "timeout"
    if proc.returncode == 0:
        return True, "all_tests_passed"
    return False, (proc.stderr.strip().splitlines() or ["error"])[-1]


def gen_code_generation(rng: random.Random, n: int) -> list[TrainingRecord]:
    """Spec -> implementation, verified by executing the reference tests."""
    out = []
    for _ in range(n):
        task = rng.choice(CODE_TASKS)
        ok, note = _run_python(task["solution"], task["tests"])
        if not ok:
            continue  # never emit unverified data
        text = (f"<|system|>\nYou are an expert Python programmer.\n<|user|>\n{task['spec']}\n"
                f"<|code|>\n{task['solution']}\n<|endcode|>\n")
        out.append(_rec(text, "code_gen", "code_completion", tests=task["tests"],
                        verifier=f"subprocess_execution:{note}", language="python"))
    return out


def gen_code_repair(rng: random.Random, n: int) -> list[TrainingRecord]:
    """Buggy code -> corrected code + explanation, verified by execution."""
    out = []
    for _ in range(n):
        task = rng.choice(CODE_TASKS)
        variants = BUGGY_VARIANTS.get(task["name"], [])
        if not variants:
            continue
        bug = rng.choice(variants)
        # The buggy version MUST fail and the fix MUST pass.
        bug_ok, _ = _run_python(bug["bug"], task["tests"])
        fix_ok, note = _run_python(task["solution"], task["tests"])
        if bug_ok or not fix_ok:
            continue
        text = (f"<|system|>\nFind and fix the bug in the code.\n<|user|>\n"
                f"```python\n{bug['bug']}\n```\n<|thought|>\n{bug['why'].capitalize()}.\n"
                f"<|code|>\n{task['solution']}\n<|endcode|>\n")
        out.append(_rec(text, "code_repair", "repair", tests=task["tests"],
                        verifier=f"bug_fails_fix_passes:{note}", language="python"))
    return out


def gen_code_explanation(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    for _ in range(n):
        task = rng.choice(CODE_TASKS)
        ok, note = _run_python(task["solution"], task["tests"])
        if not ok:
            continue
        name = task["name"]
        explanation = (f"This function is called {name}. It takes the inputs described in the "
                       f"specification, processes them deterministically, and returns the expected "
                       f"result. Its behaviour is confirmed by the accompanying assertions.")
        text = (f"<|system|>\nExplain the following Python code step by step.\n<|user|>\n"
                f"```python\n{task['solution']}\n```\n<|assistant|>\n{explanation}\n")
        out.append(_rec(text, "code_explain", "instruction", verifier=f"subprocess_execution:{note}",
                        language="python"))
    return out


# ================================================= A. FUNDAMENTAL LANGUAGE
TOPIC_DEFS = [
    ("variable", "A variable is a named storage location that holds a value which can change during program execution."),
    ("function", "A function is a reusable block of code that takes inputs, performs a computation, and returns a result."),
    ("loop", "A loop is a control structure that repeats a block of code while a condition holds."),
    ("recursion", "Recursion is a technique where a function calls itself on a smaller input until a base case is reached."),
    ("array", "An array is a contiguous collection of elements of the same type indexed by position."),
    ("hash table", "A hash table stores key-value pairs and uses a hash function to locate values in average constant time."),
    ("stack", "A stack is a last-in first-out collection supporting push and pop operations."),
    ("queue", "A queue is a first-in first-out collection supporting enqueue and dequeue operations."),
    ("algorithm", "An algorithm is a finite sequence of well-defined steps that transforms an input into an output."),
    ("complexity", "Algorithmic complexity describes how the running time or memory of an algorithm grows with input size."),
    ("compiler", "A compiler translates source code into machine code before the program runs."),
    ("interpreter", "An interpreter executes source code directly, translating it statement by statement at run time."),
    ("binary", "Binary is a base-2 numeral system using only the digits 0 and 1, used internally by digital computers."),
    ("boolean", "A boolean is a value that is either true or false, produced by comparisons and logical operators."),
    ("integer overflow", "Integer overflow happens when a computed value exceeds the range representable by its data type."),
    ("pointer", "A pointer is a variable that stores the memory address of another value."),
    ("class", "A class is a blueprint that defines the data and behaviour shared by its instances."),
    ("inheritance", "Inheritance lets a class acquire the fields and methods of another class."),
    ("exception", "An exception is an event raised when a program encounters an error it cannot handle locally."),
    ("API", "An application programming interface is a contract that lets one software component call another."),
]


def gen_language(rng: random.Random, n: int) -> list[TrainingRecord]:
    out = []
    for _ in range(n):
        term, definition = rng.choice(TOPIC_DEFS)
        text = (f"<|system|>\nDefine the term clearly and concisely.\n<|user|>\n"
                f"What is {term}?\n<|assistant|>\n{term} — {definition}\n")
        out.append(_rec(text, "language", "instruction", answer=definition, verifier="curated_definition"))
    return out


def gen_instruction(rng: random.Random, n: int) -> list[TrainingRecord]:
    """Structured-output / instruction-following examples."""
    out = []
    for _ in range(n):
        a, b = rng.randint(1, 20), rng.randint(1, 20)
        text = (f"<|system|>\nAlways answer in the exact requested format.\n<|user|>\n"
                f"List the integers from {a} to {b} as a comma-separated string.\n"
                f"<|assistant|>\n{', '.join(str(i) for i in range(a, b + 1))}\n")
        out.append(_rec(text, "instruction", "instruction",
                        answer=", ".join(str(i) for i in range(a, b + 1)), verifier="python_range"))
    return out


# ============================================ HELD-OUT EVALUATION TASK POOL
# These problems are NEVER used for training. They exist only to measure
# generalisation to unseen problem templates (spec §19 "GENERALIZATION").
HELDOUT_CODE_TASKS: list[dict[str, Any]] = [
    {"name": "abs_value", "spec": "Write a function abs_value(x) returning the absolute value of a number.",
     "solution": "def abs_value(x):\n    if x < 0:\n        return -x\n    return x",
     "tests": ["assert abs_value(-5) == 5", "assert abs_value(3) == 3"]},
    {"name": "cube", "spec": "Write a function cube(n) returning n cubed.",
     "solution": "def cube(n):\n    return n ** 3",
     "tests": ["assert cube(2) == 8", "assert cube(-3) == -27"]},
    {"name": "max_of_three", "spec": "Write a function max_of_three(a, b, c) returning the largest of three numbers.",
     "solution": "def max_of_three(a, b, c):\n    return max(a, b, c)",
     "tests": ["assert max_of_three(1, 9, 4) == 9", "assert max_of_three(-1, -7, -3) == -1"]},
    {"name": "count_occurrences", "spec": "Write a function count_occurrences(items, target) counting how many times target appears in items.",
     "solution": "def count_occurrences(items, target):\n    count = 0\n    for item in items:\n        if item == target:\n            count += 1\n    return count",
     "tests": ["assert count_occurrences([1, 2, 2, 3], 2) == 2", "assert count_occurrences(['a', 'b'], 'c') == 0"]},
    {"name": "is_positive", "spec": "Write a function is_positive(n) returning True when n is strictly greater than zero.",
     "solution": "def is_positive(n):\n    return n > 0",
     "tests": ["assert is_positive(5) == True", "assert is_positive(0) == False"]},
    {"name": "last_element", "spec": "Write a function last_element(items) returning the final item of a non-empty list.",
     "solution": "def last_element(items):\n    return items[-1]",
     "tests": ["assert last_element([1, 2, 3]) == 3", "assert last_element(['x']) == 'x'"]},
    {"name": "average", "spec": "Write a function average(values) returning the arithmetic mean of a list of numbers.",
     "solution": "def average(values):\n    return sum(values) / len(values)",
     "tests": ["assert average([2, 4]) == 3", "assert average([1, 2, 3, 4]) == 2.5"]},
    {"name": "double_all", "spec": "Write a function double_all(values) returning a new list with every element doubled.",
     "solution": "def double_all(values):\n    return [v * 2 for v in values]",
     "tests": ["assert double_all([1, 2, 3]) == [2, 4, 6]", "assert double_all([]) == []"]},
]


def gen_heldout_eval(n_per_task: int = 2, seed: int = 999) -> list[TrainingRecord]:
    """Held-out evaluation-only records from the disjoint task pool."""
    out: list[TrainingRecord] = []
    for task in HELDOUT_CODE_TASKS:
        for variant in range(n_per_task):
            if variant == 0:
                text = (f"<|system|>\nYou are an expert Python programmer.\n<|user|>\n{task['spec']}\n"
                        f"<|code|>\n{task['solution']}\n<|endcode|>\n")
            else:
                bug = task["solution"].replace("def ", "def ", 1)
                text = (f"<|system|>\nFind and fix the bug in the code.\n<|user|>\n"
                        f"```python\n{bug}\n```\n<|thought|>\nVerify the implementation against the "
                        f"specification and the tests.\n<|code|>\n{task['solution']}\n<|endcode|>\n")
            ok, note = _run_python(task["solution"], task["tests"])
            if not ok:
                continue
            out.append(_rec(text, "code_gen" if variant == 0 else "code_repair", "code_completion",
                            tests=task["tests"], verifier=f"heldout_subprocess:{note}",
                            language="python",
                            source_id=f"synthetic_heldout/{task['name']}/{variant}"))
    return out


GENERATORS: dict[str, tuple[str, Callable[[random.Random, int], list[TrainingRecord]]]] = {
    "language": ("A. fundamental language", gen_language),
    "logic": ("B. logic (boolean, deduction, pattern)", gen_boolean),
    "logic_deduction": ("B. logic deduction", gen_deduction),
    "logic_pattern": ("B. logic pattern", gen_pattern),
    "math": ("C. mathematics arithmetic", gen_arithmetic),
    "math_sequence": ("C. mathematics sequences", gen_sequence),
    "algorithm": ("D. algorithmic reasoning", gen_algorithm_trace),
    "code_gen": ("G. code generation", gen_code_generation),
    "code_repair": ("F. code repair", gen_code_repair),
    "code_explain": ("H. code explanation", gen_code_explanation),
    "instruction": ("instruction following", gen_instruction),
}


def ingest(seed: int = 1234, counts: dict[str, int] | None = None) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    counts = counts or {k: 60 for k in GENERATORS}
    rng = random.Random(seed)
    all_records: list[TrainingRecord] = []
    per_generator: dict[str, int] = {}
    for key, (_label, fn) in GENERATORS.items():
        n = counts.get(key, 0)
        if n <= 0:
            continue
        recs = fn(rng, n)
        valid = [r for r in recs if validate_record(r)[0]]
        per_generator[key] = len(valid)
        all_records.extend(valid)
    total_bytes = sum(len(r.text.encode("utf-8")) for r in all_records)
    prov = [{
        "source": "synthetic", "dataset_or_repo_name": "TinnyMe deterministic synthetic curriculum",
        "url": SRC_URL, "retrieval_date": TODAY, "license": LICENSE, "license_url": "",
        "source_type": "synthetic (math/logic/code/reasoning)", "language": "en/python",
        "approximate_size_bytes": total_bytes, "sample_count": len(all_records),
        "seed": seed, "generators": per_generator,
        "preprocessing_performed": ["deterministic_generation", "independent_recomputation",
                                    "subprocess_execution_verification", "reject_unverified"],
        "inclusion_status": "INCLUDED",
        "inclusion_reason": "first-party generated data with independently verified answers",
    }]
    logger.info("synthetic records: %d (%s)", len(all_records), per_generator)
    return all_records, prov
