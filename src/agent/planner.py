"""Bounded planner with loop detection (audit §31).

Plans are short (``max_steps`` small), every step is a concrete tool call, and
repeated identical calls are detected *before* execution so a small model cannot
spin: the executor refuses to re-run the same signature twice and terminates
with an explicit reason.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .protocol import ToolCall
from .router import route

DEFAULT_MAX_STEPS = 4


@dataclass
class Plan:
    goal: str
    steps: list[dict] = field(default_factory=list)
    max_steps: int = DEFAULT_MAX_STEPS
    rationale: str = ""

    def to_dict(self) -> dict:
        return {"goal": self.goal, "steps": self.steps, "max_steps": self.max_steps,
                "rationale": self.rationale}


def signature(call: ToolCall | dict) -> str:
    if isinstance(call, ToolCall):
        name, args = call.name, call.arguments
    else:
        name, args = call.get("name", ""), call.get("arguments", {})
    return f"{name}:{json.dumps(args, sort_keys=True, default=str)}"


def plan_for(request: str, *, max_steps: int = DEFAULT_MAX_STEPS) -> Plan:
    """Deterministic first-step plan (the executor replans after each result)."""
    decision = route(request)
    if decision.tool is None:
        return Plan(goal=request, steps=[], max_steps=max_steps,
                    rationale=decision.reason)
    return Plan(goal=request,
                steps=[{"name": decision.tool, "arguments": decision.arguments,
                        "why": decision.reason}],
                max_steps=max_steps, rationale=f"router:{decision.reason}")


class LoopDetector:
    """Detects repeated tool-call signatures and repeated no-progress answers."""

    def __init__(self, max_repeats: int = 1):
        self.seen: dict[str, int] = {}
        self.max_repeats = max_repeats
        self.triggered: list[str] = []

    def check(self, call: ToolCall | dict) -> bool:
        sig = signature(call)
        self.seen[sig] = self.seen.get(sig, 0) + 1
        if self.seen[sig] > self.max_repeats:
            self.triggered.append(sig)
            return True
        return False

    def reset(self) -> None:
        self.seen.clear()

    def report(self) -> dict:
        return {"repeat_counts": dict(self.seen), "triggered": list(self.triggered)}
