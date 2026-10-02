"""Workspace-scoped file tool (audit §25/§32).

The model may only list and read files inside its own ephemeral sandbox
workspace; path traversal and absolute paths are rejected before any filesystem
call, and every listed path is reported relative to the workspace root.
"""
from __future__ import annotations

from pathlib import Path

from ..sandbox.workspace import WorkspaceError, _safe_relpath
from .context import ToolContext

MAX_READ_CHARS = 8000


def files(ctx: ToolContext, action: str, path: str | None = None) -> dict:
    ctx.calls += 1
    if ctx.workspace is None:
        return {"ok": False, "error": "no_workspace_for_this_episode", "action": action}
    if action == "list":
        listing = []
        for rel, size in sorted(ctx.workspace.list_outputs().items()):
            if rel.startswith("tmp/"):
                continue
            listing.append({"path": rel, "bytes": size})
        ctx.note("files", action="list", count=len(listing))
        return {"ok": True, "action": "list", "workspace": str(ctx.workspace.path), "files": listing}
    if action == "read":
        if not path:
            return {"ok": False, "error": "missing_argument:path", "action": action}
        try:
            rel = _safe_relpath(path)
        except WorkspaceError as exc:
            return {"ok": False, "error": f"unsafe_path: {exc}", "action": action, "path": path}
        target = ctx.workspace.path / rel
        if not target.is_file():
            return {"ok": False, "error": f"not_a_file:{rel}", "action": action, "path": path}
        text = target.read_text(errors="replace")
        ctx.note("files", action="read", path=str(rel), chars=len(text))
        return {"ok": True, "action": "read", "path": str(rel), "text": text[:MAX_READ_CHARS],
                "truncated": len(text) > MAX_READ_CHARS}
    return {"ok": False, "error": f"unsupported_action:{action}", "action": action}
