"""Ephemeral sandbox workspace management (audit §32).

A workspace is a private directory created outside the host home tree
(``/var/tmp/tinyme-sandbox/<token>``) that holds the code under test and any
files it may read.  Paths are validated so a caller cannot ask the sandbox to
write outside the workspace.
"""
from __future__ import annotations

import os
import secrets
import shutil
from dataclasses import dataclass, field
from pathlib import Path

SANDBOX_ROOT = Path("/var/tmp/tinyme-sandbox")


class WorkspaceError(RuntimeError):
    pass


def _safe_relpath(name: str) -> Path:
    if not name or name.startswith(("/", "~")) or ".." in Path(name).parts:
        raise WorkspaceError(f"unsafe path in sandbox workspace: {name!r}")
    return Path(name)


@dataclass
class Workspace:
    path: Path
    policy: object
    created: bool = True
    _files_written: int = field(default=0)

    # ------------------------------------------------------------------ setup
    @classmethod
    def create(cls, policy, *, token: str | None = None) -> "Workspace":
        token = token or f"run-{secrets.token_hex(8)}"
        path = SANDBOX_ROOT / token
        if path.exists():
            shutil.rmtree(path)
        (path / "tmp").mkdir(parents=True, exist_ok=True)
        os.chmod(path, 0o700)
        return cls(path=path, policy=policy)

    def write(self, name: str, content: str | bytes) -> Path:
        rel = _safe_relpath(name)
        self._files_written += 1
        if self._files_written > getattr(self.policy, "max_files", 64):
            raise WorkspaceError("sandbox workspace file-count limit exceeded")
        target = self.path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        data = content.encode("utf-8") if isinstance(content, str) else content
        if len(data) > getattr(self.policy, "max_file_bytes", 8 * 1024 * 1024):
            raise WorkspaceError(f"sandbox workspace file too large: {name}")
        with open(target, "wb") as fh:
            fh.write(data)
        return target

    def write_many(self, files: dict[str, str | bytes]) -> list[Path]:
        return [self.write(name, content) for name, content in files.items()]

    # --------------------------------------------------------------- inspect
    def list_outputs(self, *, skip: set[str] | None = None) -> dict[str, int]:
        skip = skip or set()
        out: dict[str, int] = {}
        for root, _dirs, names in os.walk(self.path):
            for name in names:
                p = Path(root) / name
                rel = str(p.relative_to(self.path))
                if rel in skip:
                    continue
                out[rel] = p.stat().st_size
        return out

    def read_text(self, name: str, limit: int | None = None) -> str:
        data = (self.path / _safe_relpath(name)).read_text(errors="replace")
        return data[:limit] if limit else data

    # --------------------------------------------------------------- cleanup
    def cleanup(self) -> None:
        shutil.rmtree(self.path, ignore_errors=True)
        self.created = False

    def __enter__(self) -> "Workspace":
        return self

    def __exit__(self, *exc) -> None:
        self.cleanup()
