"""Model-facing code-execution tool — the sandboxed surface (audit §25/§32).

Everything the model writes runs through :func:`src.sandbox.runner.run_python`
with a strict policy: no network namespace, rlimits, scrubbed environment,
bounded output and a wall-clock timeout.  The tool never reports success for a
program that timed out or crashed.
"""
from __future__ import annotations

from ..sandbox.policy import SandboxPolicy
from ..sandbox.runner import run_python
from ..sandbox.workspace import Workspace
from .context import ToolContext


def run_code(ctx: ToolContext, code: str, timeout_s: int | None = None) -> dict:
    policy = SandboxPolicy.from_dict(ctx.policy.to_dict())
    if timeout_s:
        policy.wall_timeout_s = float(min(int(timeout_s), 10))
        policy.cpu_timeout_s = int(min(int(timeout_s), 10)) + 1
    ctx.calls += 1
    if ctx.workspace is None:
        ctx.workspace = Workspace.create(policy)
    result = run_python(code, policy=policy, workspace=ctx.workspace, keep_workspace=True,
                        script="_tool_code.py")
    ctx.seconds += result.duration_s
    ctx.note("code", ok=result.ok, exit_code=result.exit_code, timed_out=result.timed_out,
             isolation=result.isolation.get("level"))
    payload = {
        "ok": result.ok,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "duration_s": result.duration_s,
        "truncated": result.truncated,
        "isolation_level": result.isolation.get("level"),
        "network_enforced": result.isolation.get("network_enforced", False),
    }
    if not result.ok and not result.timed_out:
        payload["error"] = result.error or f"exit_code={result.exit_code}"
    return payload
