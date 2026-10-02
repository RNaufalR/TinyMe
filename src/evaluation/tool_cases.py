"""Independent tool-use evaluation cases (audit §4, cases A-H).

These cases are *independent of the training generator*: they are written by
hand against the shipped retrieval index and the real tool runtime, so passing
them means the model actually uses the runtime rather than reproducing a
memorised template.  Every case is executed through
:class:`src.agent.executor.AgentRuntime` with real tools (BM25 search, exact
arithmetic, the sandboxed code runner, the evidence engine).

Families
--------
A  tool not needed      → answer directly, no tool call
B  compute              → ``compute`` with valid arguments, exact value
C  search               → ``search`` with valid arguments, grounded answer
D  search → fetch       → multi-step retrieval, grounded answer
E  code                 → ``code`` in the sandbox, result interpreted
F  code repair          → failing program → corrected program → tests pass
G  error recovery       → invalid/failed call → corrected call → success
H  citation validity    → every citation in a final answer resolves (graded
                          globally across all cases that produce citations)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable

from ..agent.evidence import CITATION_ID_RE
from ..agent.executor import Trajectory


@dataclass
class ToolCase:
    case_id: str
    family: str
    request: str
    expectation: str
    tool_required: bool
    acceptable_tools: tuple[str, ...] = ()
    answer: str | None = None          # exact expected answer, when applicable
    grade: Callable[[Trajectory], dict[str, Any]] | None = None
    notes: str = ""


# --------------------------------------------------------------------- helpers
def _first_call(traj: Trajectory) -> dict | None:
    for step in traj.steps:
        if step.call:
            return step.call
    return None


def _all_calls(traj: Trajectory) -> list[dict]:
    return [s.call for s in traj.steps if s.call]


def _calls_by_name(traj: Trajectory, name: str) -> list[dict]:
    return [c for c in _all_calls(traj) if c.get("name") == name]


def _executed_ok(traj: Trajectory, name: str | None = None) -> int:
    n = 0
    for step in traj.steps:
        if step.call and step.result is not None and step.result.get("ok"):
            if name is None or step.call.get("name") == name:
                n += 1
    return n


def _norm(text: Any) -> str:
    return re.sub(r"[\s,]+", "", str(text).strip().lower()).rstrip(".")


def _call_matches(call: dict | None, expected: tuple[str, ...]) -> bool:
    return bool(call) and call.get("name") in expected


def _args_valid(traj: Trajectory) -> bool:
    """A call is argument-valid when the runtime accepted it (schema validation)."""
    for step in traj.steps:
        if step.call and step.result is not None:
            err = str(step.result.get("error") or "")
            if err.startswith(("unknown_tool", "missing_argument", "bad_type", "unknown_arguments",
                               "below_minimum", "above_maximum", "too_long", "not_a_choice")):
                return False
    return True


def _grade_common(traj: Trajectory) -> dict[str, Any]:
    return {
        "solved": bool(traj.solved),
        "steps": len(traj.steps),
        "tool_calls": len(_all_calls(traj)),
        "tools_used": [c.get("name") for c in _all_calls(traj)],
        "tool_syntax_valid": all(step.error is None or
                                 not str(step.error).startswith("protocol_error") for step in traj.steps),
        "args_valid": _args_valid(traj),
        "exec_success": _executed_ok(traj) > 0,
        "grounded": bool(traj.grounding.get("grounded")) if traj.grounding else False,
        "stop_reason": traj.stop_reason,
    }


def _grade_direct(traj: Trajectory, case: ToolCase) -> dict[str, Any]:
    metrics = _grade_common(traj)
    metrics["tool_needed"] = False
    metrics["called_tool"] = bool(_all_calls(traj))
    metrics["tool_not_needed_ok"] = not metrics["called_tool"]
    metrics["answer_correct"] = bool(
        case.answer is not None and traj.final is not None
        and _norm(case.answer) in _norm(traj.final))
    metrics["correct"] = metrics["tool_not_needed_ok"] and (
        metrics["answer_correct"] if case.answer is not None else bool(traj.final))
    return metrics


def _grade_tool(traj: Trajectory, case: ToolCase) -> dict[str, Any]:
    metrics = _grade_common(traj)
    metrics["tool_needed"] = True
    first = _first_call(traj)
    metrics["first_tool_correct"] = _call_matches(first, case.acceptable_tools)
    metrics["tool_name_correct"] = any(_call_matches(c, case.acceptable_tools) for c in _all_calls(traj))
    metrics["answer_correct"] = bool(
        case.answer is not None and traj.final is not None
        and _norm(case.answer) in _norm(traj.final))
    metrics["grounded_final"] = bool(traj.final) and bool(traj.grounding.get("grounded"))
    metrics["correct"] = bool(metrics["tool_name_correct"] and metrics["exec_success"]
                              and (metrics["answer_correct"] or case.answer is None))
    return metrics


def _grade_multistep(traj: Trajectory, case: ToolCase) -> dict[str, Any]:
    metrics = _grade_tool(traj, case)
    metrics["multi_step"] = len(_calls_by_name(traj, "search")) >= 1 and len(_calls_by_name(traj, "fetch")) >= 1
    metrics["multi_step_success"] = bool(metrics["multi_step"] and metrics["exec_success"]
                                         and metrics["grounded_final"])
    metrics["correct"] = bool(metrics["multi_step_success"]
                              and (metrics["answer_correct"] or case.answer is None))
    return metrics


def _grade_recovery(traj: Trajectory, case: ToolCase) -> dict[str, Any]:
    metrics = _grade_tool(traj, case)
    results = [s.result for s in traj.steps if s.result is not None]
    failed = [r for r in results if not r.get("ok")]
    recovered = len(results) >= 2 and bool(results[-1].get("ok"))
    metrics["saw_failure"] = bool(failed)
    metrics["recovery_success"] = bool(recovered and metrics["exec_success"])
    metrics["correct"] = bool(metrics["recovery_success"] and metrics["tool_name_correct"]
                              and (metrics["answer_correct"] or case.answer is None))
    return metrics


def _grade_code(traj: Trajectory, case: ToolCase) -> dict[str, Any]:
    metrics = _grade_tool(traj, case)
    metrics["code_executed"] = _executed_ok(traj, "code") > 0
    metrics["correct"] = bool(metrics["code_executed"] and metrics["exec_success"]
                              and (metrics["answer_correct"] or case.answer is None))
    return metrics


# ----------------------------------------------------------------------- cases
def build_tool_cases() -> list[ToolCase]:
    """The concrete case set (A-H), all authored independently of training data."""
    cases: list[ToolCase] = []

    # ---- A: tool not needed (direct answer) --------------------------------
    for i, (q, a) in enumerate([
        ("What is 21 + 34?", "55"),
        ("What is 8 * 7?", "56"),
        ("Capitalise the word 'tinyme'.", "TINYME"),
        ("Reverse the string 'abc'.", "cba"),
        ("What is 100 - 45?", "55"),
    ]):
        cases.append(ToolCase(f"A{i}", "A", q, "no tool call; direct correct answer",
                              tool_required=False, answer=a, grade=_grade_direct,
                              notes="answerable without the runtime"))

    # ---- B: compute --------------------------------------------------------
    # Operands are held out (none of them appears in the training generator), and
    # the expression shapes match the shapes the data covers, so a correct answer
    # requires *copying the operands out of the prompt* — the exact behaviour the
    # tool-SFT intervention targets.
    compute_cases = [(4837, 962, 71), (7123, 481, 13), (9059, 337, 29), (6271, 1543, 47),
                     (3821, 219, 22)]
    for i, (a, b, c) in enumerate(compute_cases):
        shape = ("mul_add", "mul_add", "add_mul", "mul", "add_sub")[i]
        if shape == "mul_add":
            expr, value = f"{a} * {b} + {c}", a * b + c
        elif shape == "add_mul":
            expr, value = f"{a} + {b} * {c}", a + b * c
        elif shape == "mul":
            expr, value = f"{a} * {b}", a * b
        else:
            expr, value = f"{a} + {b} - {c}", a + b - c
        cases.append(ToolCase(f"B{i}", "B", f"Compute exactly: {expr}", "compute with valid args",
                              tool_required=True, acceptable_tools=("compute",),
                              answer=str(value), grade=_grade_tool,
                              notes=f"{shape} shape; operands held out from training data"))

    # ---- C: search ---------------------------------------------------------
    search_cases = [
        ("What is the population of jakarta according to a source?", "10,679,951", "FACT-988002"),
        ("Which source supports the capital of indonesia?", "Jakarta", "FACT-8334A1"),
        ("What is the big-o of binary search?", "O(log n)", "FACT-2082DD"),
        ("What licence applies to the python standard library?", "PSF-2.0", "FACT-16880E"),
    ]
    for i, (q, a, _sid) in enumerate(search_cases):
        cases.append(ToolCase(f"C{i}", "C", q, "search with valid args; grounded answer",
                              tool_required=True, acceptable_tools=("search",), answer=a,
                              grade=_grade_tool, notes="requires the retrieval index"))

    # ---- D: search -> fetch ------------------------------------------------
    fetch_targets = [("FACT-988002", "population of jakarta"),
                     ("FACT-FCD206", "definition of a prime number"),
                     ("FACT-0B1DC0", "shortest path algorithm")]
    for i, (sid, topic) in enumerate(fetch_targets):
        cases.append(ToolCase(f"D{i}", "D",
                              f"Read the stored source {sid} about the {topic} and quote its key sentence.",
                              "search then fetch; grounded answer",
                              tool_required=True, acceptable_tools=("search", "fetch"),
                              grade=_grade_multistep, notes="multi-step retrieval"))
    # a fetch-only case: the id is given, so a single fetch is the right call
    cases.append(ToolCase("D3", "D", "Fetch the passage with source_id FACT-2082DD and summarise it.",
                          "fetch with the given id", tool_required=True,
                          acceptable_tools=("fetch",), grade=_grade_tool))

    # ---- E: code execution -------------------------------------------------
    program = ("def is_prime(n):\n"
               "    if n < 2:\n        return False\n"
               "    for d in range(2, int(n ** 0.5) + 1):\n"
               "        if n % d == 0:\n            return False\n"
               "    return True\n\n"
               "assert is_prime(97)\n"
               "assert not is_prime(91)\n"
               "print(is_prime(97), is_prime(91))\n")
    cases.append(ToolCase("E0", "E", f"Run this program in the sandbox and report the result:\n```python\n{program}```",
                          "code tool executes; result interpreted", tool_required=True,
                          acceptable_tools=("code",), answer="True False", grade=_grade_code))
    program2 = ("nums = [4, 9, 16, 25]\n"
                "roots = [int(n ** 0.5) for n in nums]\n"
                "assert roots == [2, 3, 4, 5]\n"
                "print(sum(roots))\n")
    cases.append(ToolCase("E1", "E", f"Execute this and tell me the printed number:\n```python\n{program2}```",
                          "code tool executes; result interpreted", tool_required=True,
                          acceptable_tools=("code",), answer="14", grade=_grade_code))

    # ---- F: code repair ----------------------------------------------------
    buggy = ("def count_words(text):\n"
             "    counts = {}\n"
             "    for w in text.lower().split():\n"
             "        counts[w] = 1\n"
             "    return counts\n")
    spec = ("Write count_words(text) returning each lowercase whitespace-separated word mapped to "
            "its frequency. The current implementation returns 1 for every word.")
    cases.append(ToolCase("F0", "F", f"The tests fail for this code:\n```python\n{buggy}```\n"
                                  f"Specification: {spec}\n"
                                  "Fix it, run the tests, and report the result.",
                          "code tool runs a corrected implementation; tests pass",
                          tool_required=True, acceptable_tools=("code",), answer="2",
                          grade=_grade_code,
                          notes="the repaired program must count duplicate words"))
    buggy2 = ("def total(values):\n"
              "    t = 0\n"
              "    for v in values:\n"
              "        t = v\n"
              "    return t\n")
    cases.append(ToolCase("F1", "F", f"This function is wrong:\n```python\n{buggy2}```\n"
                                   "It must return the sum of the list. Fix it, run it on "
                                   "[3, 4, 5] and report the number.",
                          "code tool runs a corrected implementation",
                          tool_required=True, acceptable_tools=("code",), answer="12",
                          grade=_grade_code))

    # ---- G: error recovery -------------------------------------------------
    cases.append(ToolCase("G0", "G", "Look up the population of jakartaa (note the typo) and if the "
                                     "search finds nothing, correct the query and answer with a citation.",
                          "failed/empty call then a corrected successful call",
                          tool_required=True, acceptable_tools=("search",), answer="10,679,951",
                          grade=_grade_recovery, notes="typo query returns no results"))
    cases.append(ToolCase("G1", "G", "Fetch source_id FACT-000000; if that fails, find the real id for "
                                     "the capital of indonesia and fetch it.",
                          "invalid call then a corrected successful call",
                          tool_required=True, acceptable_tools=("fetch", "search"),
                          grade=_grade_recovery, notes="unknown source id must be recovered from"))
    cases.append(ToolCase("G2", "G", "Evaluate 12 / 0 with the compute tool; if it errors, evaluate 12 / 4 "
                                     "instead and report the value.",
                          "error then corrected call", tool_required=True,
                          acceptable_tools=("compute",), answer="3", grade=_grade_recovery))

    return cases


# ------------------------------------------------------------------- reporting
TOOL_METRICS = ["tool_needed_accuracy", "tool_not_needed_accuracy", "tool_syntax_validity",
                "tool_name_accuracy", "argument_validity", "argument_accuracy",
                "execution_success", "multi_step_success", "error_recovery_success",
                "grounded_final_answer", "citation_validity", "task_completion"]


def citation_validity(trajectory: Trajectory) -> dict[str, Any]:
    """Every citation in a final answer must resolve to runtime-issued evidence."""
    if not trajectory.final:
        return {"answers_with_citations": 0, "valid": 0, "invalid": 0, "invalid_ids": []}
    cited = set(CITATION_ID_RE.findall(trajectory.final))
    if not cited:
        return {"answers_with_citations": 0, "valid": 0, "invalid": 0, "invalid_ids": []}
    known = set(trajectory.grounding.get("cited_ids", [])) - set(
        trajectory.grounding.get("unsupported_citations", []))
    invalid = sorted(i for i in cited if i not in known)
    return {"answers_with_citations": 1, "valid": 1 if not invalid else 0,
            "invalid": 1 if invalid else 0, "invalid_ids": invalid}


def summarise(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-case rows into the metric block required by audit §2/§4."""
    def acc(pred, key="correct") -> float:
        sel = [r for r in rows if pred(r)]
        return round(sum(1 for r in sel if r.get(key)) / len(sel), 4) if sel else 0.0

    needed = [r for r in rows if r["tool_needed"]]
    not_needed = [r for r in rows if not r["tool_needed"]]
    multi = [r for r in rows if r["family"] == "D"]
    recovery = [r for r in rows if r["family"] == "G"]
    citable = [r for r in rows if r.get("answers_with_citations")]
    return {
        "cases": len(rows),
        "tool_needed_accuracy": acc(lambda r: r["tool_needed"], "tool_name_correct"),
        "tool_not_needed_accuracy": acc(lambda r: not r["tool_needed"], "tool_not_needed_ok"),
        "tool_syntax_validity": acc(lambda r: True, "tool_syntax_valid"),
        "tool_name_accuracy": acc(lambda r: r["tool_needed"], "tool_name_correct"),
        "argument_validity": acc(lambda r: r["tool_needed"] and r.get("tool_calls", 0) > 0, "args_valid"),
        "argument_accuracy": acc(lambda r: r["tool_needed"] and r.get("tool_calls", 0) > 0,
                                 "first_tool_correct"),
        "execution_success": acc(lambda r: r["tool_needed"], "exec_success"),
        "multi_step_success": acc(lambda r: r in multi, "multi_step_success"),
        "error_recovery_success": acc(lambda r: r in recovery, "recovery_success"),
        "grounded_final_answer": acc(lambda r: r["tool_needed"], "grounded_final"),
        "citation_validity": acc(lambda r: r in citable, "citation_valid"),
        "task_completion": acc(lambda r: True, "correct"),
        "by_family": {fam: acc(lambda r, f=fam: r["family"] == f) for fam in sorted({r["family"] for r in rows})},
        "needed_cases": len(needed), "not_needed_cases": len(not_needed),
    }
