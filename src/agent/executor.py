"""Bounded agent executor with an auditable trajectory (audit §31).

The executor drives one episode: it asks the policy (a real model, or a scripted
policy in tests) for a turn, parses the strict protocol, validates the call
against the registry, executes it inside the sandboxed tool runtime, feeds the
result back as **context-only** (never as a loss target), and stops on a final
answer, on budget exhaustion, or on loop detection.  Every step is recorded.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..sandbox.policy import SandboxPolicy
from ..tools.context import ToolContext
from .evidence import EvidenceStore
from .planner import LoopDetector
from .protocol import FINAL, ProtocolError, parse_model_output, render_tool_result, validate_output
from .tool_registry import ToolRegistry

DEFAULT_MAX_STEPS = 4
DEFAULT_MAX_SECONDS = 60.0


@dataclass
class Step:
    index: int
    raw_output: str
    output: dict
    call: dict | None = None
    result: dict | None = None
    duration_s: float = 0.0
    error: str | None = None

    def to_dict(self) -> dict:
        return {"index": self.index, "raw_output": self.raw_output[:2000], "output": self.output,
                "call": self.call, "result": self.result, "duration_s": round(self.duration_s, 4),
                "error": self.error}


@dataclass
class Trajectory:
    request: str
    steps: list[Step] = field(default_factory=list)
    final: str | None = None
    stop_reason: str = "unknown"
    evidence: dict = field(default_factory=dict)
    grounding: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.final is not None and self.stop_reason in ("final", "budget_after_final")

    def to_dict(self) -> dict:
        return {"request": self.request, "final": self.final, "stop_reason": self.stop_reason,
                "solved": self.solved, "steps": [s.to_dict() for s in self.steps],
                "evidence": self.evidence, "grounding": self.grounding, "metrics": self.metrics}


def render_transcript(trajectory: Trajectory, *, system: str | None = None) -> str:
    """Deterministic textual transcript of an episode (used by docs/tests)."""
    parts = []
    if system:
        parts.append(f"<|system|>{system}")
    parts.append(f"<|user|>{trajectory.request}")
    for step in trajectory.steps:
        parts.append(step.raw_output)
        if step.result is not None:
            parts.append(render_tool_result(step.result))
    if trajectory.final is not None:
        parts.append(f"{FINAL}{trajectory.final}")
    return "\n".join(parts)


class AgentRuntime:
    def __init__(self, *, registry: ToolRegistry | None = None, context: ToolContext | None = None,
                 max_steps: int = DEFAULT_MAX_STEPS, max_seconds: float = DEFAULT_MAX_SECONDS,
                 max_context_chars: int = 24_000, policy: SandboxPolicy | None = None,
                 system_prompt: str | None = None):
        context = context or ToolContext.create(policy=policy)
        if context.workspace is None:
            from ..sandbox.workspace import Workspace
            context.workspace = Workspace.create(context.policy)
        self.context = context
        self.registry = registry or ToolRegistry.default(context=context)
        self.max_steps = max_steps
        self.max_seconds = max_seconds
        self.max_context_chars = max_context_chars
        self.evidence = EvidenceStore()
        self.loops = LoopDetector()
        self.system_prompt = system_prompt or _DEFAULT_SYSTEM

    # ------------------------------------------------------------------ helpers
    def tool_schemas(self) -> list[dict]:
        return self.registry.schemas()

    def _truncate_context(self, history: list[str]) -> list[str]:
        total, kept = 0, []
        for chunk in reversed(history):
            total += len(chunk)
            if total > self.max_context_chars:
                kept.append(chunk[: max(0, self.max_context_chars - total + len(chunk))])
                break
            kept.append(chunk)
        return list(reversed(kept))

    # -------------------------------------------------------------------- solve
    def solve(self, request: str, policy_fn) -> Trajectory:
        """Run one episode.  ``policy_fn(request, history, step, tools) -> str``."""
        started = time.monotonic()
        traj = Trajectory(request=request)
        history: list[str] = [f"<|user|>{request}"]
        for step_index in range(1, self.max_steps + 1):
            if time.monotonic() - started > self.max_seconds:
                traj.stop_reason = f"wall_budget_exhausted({self.max_seconds}s)"
                break
            step_started = time.monotonic()
            raw = policy_fn(request, self._truncate_context(history), step_index, self.tool_schemas())
            history.append(raw)
            try:
                parsed = parse_model_output(raw)
            except ProtocolError as exc:
                traj.steps.append(Step(index=step_index, raw_output=raw, output={},
                                       error=f"protocol_error:{exc.reason}", duration_s=time.monotonic() - step_started))
                traj.stop_reason = f"protocol_error:{exc.reason}"
                break
            problems = validate_output(raw, registry=self.registry)
            step = Step(index=step_index, raw_output=raw, output=parsed.to_dict(),
                        duration_s=time.monotonic() - step_started)

            if parsed.final is not None:
                traj.final = parsed.final
                traj.grounding = self.evidence.verify(parsed.final)
                traj.stop_reason = "final"
                traj.steps.append(step)
                break

            call = parsed.tool_calls[0]
            step.call = call.to_dict()
            if self.loops.check(call):
                step.error = "loop_detected"
                traj.stop_reason = "loop_detected"
                traj.steps.append(step)
                break
            budget = self.context.budget_exhausted()
            if budget:
                step.error = budget
                traj.stop_reason = budget
                traj.steps.append(step)
                break
            result = self.registry.call(call.name, call.arguments)
            self.context.calls += 1
            self.context.seconds += result.duration_s
            self.evidence.add_tool_result(call.name, result)
            step.result = result.to_dict()
            step.duration_s = time.monotonic() - step_started
            if problems:
                step.error = ";".join(problems)
            traj.steps.append(step)
            history.append(render_tool_result(result.to_dict()))
        else:
            traj.stop_reason = f"max_steps_reached({self.max_steps})"

        traj.evidence = self.evidence.to_dict()
        if not traj.grounding:
            traj.grounding = {"grounded": None, "evidence_items": len(self.evidence.items),
                              "problems": ["no_final_answer"]}
        traj.metrics = {
            "steps": len(traj.steps),
            "tool_calls": sum(1 for s in traj.steps if s.call),
            "errors": [s.error for s in traj.steps if s.error],
            "total_seconds": round(time.monotonic() - started, 4),
            "context": self.context.summary(),
            "loops": self.loops.report(),
        }
        traj.steps = list(traj.steps)
        return traj

    def close(self) -> None:
        self.context.close()


def scripted_policy(outputs: list[str]):
    """Return a policy function that replays fixed model outputs (tests/docs)."""
    state = {"i": 0}

    def _policy(request, history, step, tools):
        idx = min(state["i"], len(outputs) - 1)
        state["i"] += 1
        return outputs[idx]

    return _policy


_DEFAULT_SYSTEM = (
    "You are TinyMe, a tiny language model paired with an external tool runtime. "
    "Call a tool when the answer needs computation, retrieval or code execution; "
    "otherwise answer directly with <|final|>."
)
