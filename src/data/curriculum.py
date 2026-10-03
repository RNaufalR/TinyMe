"""Curriculum stages for the capability intervention (DEC-013).

The SFT corpus mixes two very different kinds of records:

* **drills** — short, single-turn tasks whose answer *is* the content of the
  prompt (copy a span, do a small sum, upper-case a word, run one tool call and
  read its result).  Most supervised tokens are model output, so the signal per
  token is high;
* **trajectories** — long multi-step tool episodes (search → fetch → answer,
  failing program → repaired program → tests) where most tokens are *context*
  (tool results) and only the model turns are supervised.

Training them as one undifferentiated mixture (EXP-012, `--stage sft`) left the
model able to emit the protocol but unable to copy an operand out of its own
prompt: on a held-in copy record it produced ``58a7e6d…`` for
``4a26e73e-88a9-ccf1-ec6e-d68aeba6fd26`` and on a held-in compute record
``"9 - 43 * 7"`` for the gold ``"1 - 6445 * 4"``.  The stages below let the
trainer present the drills first (dense, unambiguous supervision) and the
trajectories afterwards, which is the intervention DEC-013 records.

The stage partition is *total and disjoint* over the structured (SFT) records:
every record carrying segments belongs to exactly one of the two stages, and
`tests/test_curriculum.py` re-derives that property from the corpus on disk so
no record can silently fall out of training.
"""
from __future__ import annotations

from typing import Any, Callable

STAGE_DRILL = "drill"
STAGE_TRAJECTORY = "trajectory"

#: Template-id prefixes of the single-turn drills (0 or 1 tool call).
DRILL_TEMPLATE_PREFIXES: tuple[str, ...] = (
    "copy/",                 # copy a span verbatim out of the prompt
    "no_tool/",              # small arithmetic / text edits answered directly
    "algorithm/",            # trace a tiny algorithm step by step
    "logic/",                # boolean arithmetic, syllogisms
    "math/",                 # arithmetic, sequences, word problems
    "instruction/",          # follow an instruction over the given values
    "code/",                 # write, repair (in-head) or explain a snippet
    "hf/gsm8k/",             # word-problem slice
    "hf/mbpp/",              # code-completion slice
    "stdlib/",               # standard-library documentation explanations
    "tool/compute",          # one compute call, then read the real result
    "tool/fetch_direct",     # the id is given: one fetch, then the passage
    "tool/code_run",         # write code, execute once, report the output
)

#: Template-id prefixes of the multi-step episodes (>= 2 tool calls).
TRAJECTORY_TEMPLATE_PREFIXES: tuple[str, ...] = (
    "tool/search_single",    # search -> fetch -> grounded, cited answer
    "tool/search_fetch",     # search -> fetch -> grounded answer
    "tool/search_multi",     # several searches, comparison answer
    "tool/code_repair",      # failing program -> repair -> tests pass
    "tool/error_recovery",   # invalid/failed call -> corrected call -> success
)


def template_id(record: dict[str, Any]) -> str:
    """Template id of a record (``""`` when the record carries none)."""
    return str(record.get("template_id") or record.get("template") or "")


def tool_call_count(record: dict[str, Any]) -> int:
    """Number of ``tool_call`` segments in a structured record."""
    return sum(1 for s in record.get("segments") or [] if s.get("role") == "tool_call")


def stage_of(record: dict[str, Any]) -> str | None:
    """Return the curriculum stage of a structured record, or ``None``.

    The partition is **structural**, not a hard-coded list of template ids: a
    record is a trajectory when it contains at least two tool calls (so the
    model has to read a real tool result and decide again), and a drill
    otherwise.  That makes the partition total and disjoint by construction —
    a new generator family cannot silently fall out of training — while the
    prefix tables above stay the human-readable description of which families
    currently land in which stage (``tests/test_curriculum.py`` asserts that the
    two agree).

    Plain documents (no ``segments``) are the pretraining stage and return
    ``None`` here: the curriculum only partitions the instruction corpus.
    """
    if not record.get("segments"):
        return None
    return STAGE_TRAJECTORY if tool_call_count(record) >= 2 else STAGE_DRILL


def record_filter(stage: str) -> Callable[[dict[str, Any]], bool]:
    """Predicate selecting the records of ``stage`` (``"sft"`` keeps them all)."""
    if stage in ("sft", "mixed"):
        return lambda record: bool(record.get("segments"))
    if stage == STAGE_DRILL:
        return lambda record: stage_of(record) == STAGE_DRILL
    if stage == STAGE_TRAJECTORY:
        return lambda record: stage_of(record) == STAGE_TRAJECTORY
    raise ValueError(f"unknown curriculum stage {stage!r}; expected drill/trajectory/sft")


def partition_report(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Count structured records per stage and flag any unassigned template."""
    counts = {STAGE_DRILL: 0, STAGE_TRAJECTORY: 0}
    unassigned: dict[str, int] = {}
    structured = 0
    for record in records:
        if not record.get("segments"):
            continue
        structured += 1
        stage = stage_of(record)
        if stage is None:
            unassigned[template_id(record)] = unassigned.get(template_id(record), 0) + 1
        else:
            counts[stage] += 1
    return {"structured_records": structured, "drill": counts[STAGE_DRILL],
            "trajectory": counts[STAGE_TRAJECTORY],
            "assigned": counts[STAGE_DRILL] + counts[STAGE_TRAJECTORY],
            "unassigned_templates": unassigned}
