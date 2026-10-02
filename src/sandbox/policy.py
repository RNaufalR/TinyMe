"""Sandbox policy (corrective audit §32).

A policy is a pure description of the limits applied to one execution; it is
serialised into every result so the evidence trail records exactly what
enforcement was in place.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class SandboxPolicy:
    # time
    wall_timeout_s: float = 10.0
    cpu_timeout_s: int = 5
    # memory
    memory_mb: int = 512
    # output / files
    max_output_bytes: int = 64 * 1024
    max_file_bytes: int = 8 * 1024 * 1024
    max_files: int = 64
    # processes / descriptors
    max_processes: int = 16
    max_open_files: int = 64
    # isolation
    network: bool = False
    isolate_filesystem: bool = True
    hide_host_home: bool = True
    # environment
    inherit_env: bool = False
    extra_env: dict[str, str] = field(default_factory=dict)
    # workspace
    writable_workspace: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SandboxPolicy":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def describe_enforcement(self) -> list[str]:
        out = [f"rlimit CPU={self.cpu_timeout_s}s", f"rlimit AS={self.memory_mb}MB",
               f"rlimit FSIZE={self.max_file_bytes}B", f"rlimit NOFILE={self.max_open_files}",
               f"rlimit NPROC={self.max_processes}",
               f"wall timeout={self.wall_timeout_s}s", f"output cap={self.max_output_bytes}B",
               "process-group kill on timeout",
               "environment scrubbed" if not self.inherit_env else "environment inherited"]
        out.append("network namespace (loopback only)" if not self.network else "network allowed")
        out.append("mount namespace hides host home/project" if self.isolate_filesystem and self.hide_host_home
                   else "filesystem isolation unavailable")
        return out
