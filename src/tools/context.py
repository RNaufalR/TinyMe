"""Shared execution context for the model-facing tools.

Holds the retrieval index, the search provider, the sandbox policy and the
per-episode sandbox workspace, plus the call/step budget that makes the tool
runtime bounded (audit §31).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..sandbox.policy import SandboxPolicy
from ..sandbox.workspace import Workspace
from .retrieval import RetrievalIndex, load_default_index


@dataclass
class ToolContext:
    index: RetrievalIndex | None = None
    policy: SandboxPolicy = field(default_factory=SandboxPolicy)
    workspace: Workspace | None = None
    max_tool_calls: int = 12
    max_total_seconds: float = 90.0
    max_context_chars: int = 24_000

    #: accounting (enforced by the executor, reported in the trajectory)
    calls: int = 0
    seconds: float = 0.0
    events: list[dict] = field(default_factory=list)

    # ------------------------------------------------------------------ set-up
    @classmethod
    def create(cls, *, policy: SandboxPolicy | None = None, index_path: str | Path | None = None,
               with_workspace: bool = True, **kw) -> "ToolContext":
        policy = policy or SandboxPolicy()
        ctx = cls(index=load_default_index(index_path), policy=policy, **kw)
        if with_workspace:
            ctx.workspace = Workspace.create(policy)
        return ctx

    def close(self) -> None:
        if self.workspace is not None:
            self.workspace.cleanup()
            self.workspace = None

    # --------------------------------------------------------------- accounting
    def budget_exhausted(self) -> str | None:
        if self.calls >= self.max_tool_calls:
            return f"tool_call_budget_exhausted({self.max_tool_calls})"
        if self.seconds >= self.max_total_seconds:
            return f"tool_time_budget_exhausted({self.max_total_seconds}s)"
        return None

    def note(self, kind: str, **payload) -> None:
        self.events.append({"kind": kind, **payload})

    def summary(self) -> dict:
        return {"tool_calls": self.calls, "tool_seconds": round(self.seconds, 3),
                "max_tool_calls": self.max_tool_calls, "max_total_seconds": self.max_total_seconds,
                "retrieval_index_documents": len(self.index) if self.index else 0,
                "workspace": str(self.workspace.path) if self.workspace else None}
