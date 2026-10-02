"""Tool registry: a small, strictly validated, model-facing surface (audit §25).

Only five tools are ever exposed to the model (``search``, ``fetch``,
``compute``, ``code``, ``files``).  Every call is validated against the tool's
JSON-schema-style argument spec before execution; unknown tools, missing
arguments, wrong types and out-of-range values are rejected with structured
errors instead of reaching the implementation.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .protocol import ToolResult

log = logging.getLogger("tinyme.tools")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, dict]
    fn: Callable[..., dict]
    network: bool = False
    sandboxed: bool = False
    max_seconds: float = 15.0
    returns_evidence: bool = False

    def schema(self) -> dict:
        return {"name": self.name, "description": self.description,
                "parameters": {"type": "object", "properties": self.parameters,
                               "required": sorted(k for k, v in self.parameters.items()
                                                  if v.get("required", False))}}


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    call_log: list[dict] = field(default_factory=list)
    context: object | None = None

    # ------------------------------------------------------------- registration
    def register(self, spec: ToolSpec) -> None:
        if spec.name in self.specs:
            raise ValueError(f"duplicate tool {spec.name}")
        self.specs[spec.name] = spec

    def names(self) -> list[str]:
        return sorted(self.specs)

    def schemas(self) -> list[dict]:
        """The exact JSON surface advertised to the model."""
        return [self.specs[n].schema() for n in self.names()]

    # -------------------------------------------------------------- validation
    def validate_arguments(self, name: str, args: dict[str, Any]) -> list[str]:
        if name not in self.specs:
            return [f"unknown_tool:{name}"]
        spec = self.specs[name]
        problems: list[str] = []
        unknown = set(args) - set(spec.parameters)
        if unknown:
            problems.append(f"unknown_arguments:{sorted(unknown)}")
        for key, meta in spec.parameters.items():
            if meta.get("required") and key not in args:
                problems.append(f"missing_argument:{key}")
                continue
            if key not in args:
                continue
            value = args[key]
            expected = meta.get("type", "string")
            if expected == "integer" and (not isinstance(value, int) or isinstance(value, bool)):
                problems.append(f"bad_type:{key}:expected integer")
            elif expected == "number" and not isinstance(value, (int, float)):
                problems.append(f"bad_type:{key}:expected number")
            elif expected == "string" and not isinstance(value, str):
                problems.append(f"bad_type:{key}:expected string")
            elif expected == "array" and not isinstance(value, list):
                problems.append(f"bad_type:{key}:expected array")
            elif expected in ("integer", "number") and "min" in meta and value < meta["min"]:
                problems.append(f"below_minimum:{key}")
            elif expected in ("integer", "number") and "max" in meta and value > meta["max"]:
                problems.append(f"above_maximum:{key}")
            elif expected == "string" and "max_length" in meta and len(value) > meta["max_length"]:
                problems.append(f"too_long:{key}")
            elif expected in ("string", "array") and meta.get("choices"):
                if value not in meta["choices"]:
                    problems.append(f"not_a_choice:{key}")
        return problems

    # --------------------------------------------------------------- execution
    def call(self, name: str, args: dict[str, Any]) -> ToolResult:
        started = time.monotonic()
        problems = self.validate_arguments(name, args)
        if problems:
            result = ToolResult(name=name, ok=False, error=";".join(problems),
                                duration_s=time.monotonic() - started)
            self.call_log.append({"tool": name, "args": args, "ok": False, "error": result.error})
            return result
        spec = self.specs[name]
        try:
            payload = spec.fn(**args)
            ok = bool(payload.pop("ok", True))
            evidence = payload.pop("evidence", []) if spec.returns_evidence else []
            result = ToolResult(name=name, ok=ok, payload=payload if ok else {},
                                error=None if ok else str(payload.get("error", "tool reported failure")),
                                duration_s=time.monotonic() - started, evidence=evidence)
        except Exception as exc:  # tools must never crash the agent loop
            log.warning("tool %s failed: %s: %s", name, type(exc).__name__, exc)
            result = ToolResult(name=name, ok=False, error=f"{type(exc).__name__}: {exc}",
                                duration_s=time.monotonic() - started)
        self.call_log.append({"tool": name, "args": args, "ok": result.ok,
                              "error": result.error, "duration_s": result.duration_s})
        return result

    @classmethod
    def default(cls, *, context=None, policy=None) -> "ToolRegistry":
        """The canonical small surface wired to the real implementations.

        Tool functions are bound to ``context`` (retrieval index + sandbox
        workspace + budgets); ``policy`` only customises the sandbox policy of a
        freshly created context.
        """
        from functools import partial

        from ..tools.code import run_code as _run_code
        from ..tools.compute import compute as _compute
        from ..tools.context import ToolContext
        from ..tools.fetch import fetch as _fetch
        from ..tools.files import files as _files
        from ..tools.search import search as _search

        context = context or ToolContext.create(policy=policy)
        search = partial(_search, context)
        fetch = partial(_fetch, context)
        compute = partial(_compute)
        run_code = partial(_run_code, context)
        files = partial(_files, context)

        registry = cls(context=context)
        registry.register(ToolSpec(
            name="search",
            description="Search the local retrieval corpus; returns ranked passages with source ids.",
            parameters={"query": {"type": "string", "required": True, "max_length": 300},
                        "k": {"type": "integer", "required": False, "min": 1, "max": 8}},
            fn=search, network=False, returns_evidence=True))
        registry.register(ToolSpec(
            name="fetch",
            description="Fetch a stored passage by source id (no network access).",
            parameters={"source_id": {"type": "string", "required": True, "max_length": 120}},
            fn=fetch, network=False, returns_evidence=True))
        registry.register(ToolSpec(
            name="compute",
            description="Evaluate a deterministic arithmetic/math expression exactly.",
            parameters={"expression": {"type": "string", "required": True, "max_length": 400}},
            fn=compute, network=False))
        registry.register(ToolSpec(
            name="code",
            description="Run a short Python program in the sandbox (no network, CPU/memory/time limited).",
            parameters={"code": {"type": "string", "required": True, "max_length": 8000},
                        "timeout_s": {"type": "integer", "required": False, "min": 1, "max": 10}},
            fn=run_code, network=False, sandboxed=True))
        registry.register(ToolSpec(
            name="files",
            description="Read or list files inside the current sandbox workspace.",
            parameters={"action": {"type": "string", "required": True, "choices": ["list", "read"]},
                        "path": {"type": "string", "required": False, "max_length": 200}},
            fn=files, network=False, sandboxed=True))
        if policy is not None:
            registry.specs["code"].max_seconds = policy.wall_timeout_s
        return registry
