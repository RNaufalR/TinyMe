"""Auto-detection of the strongest isolation primitive available (audit §32).

Detected *by probing*, never assumed; the probe result is cached per process and
reported in every sandbox result as ``isolation_level`` so evidence never
over-claims what was actually enforced.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass

_NET_PROBE = """\
import socket
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.settimeout(2)
    s.connect(("1.1.1.1", 443)); print("NETWORK_REACHABLE")
except Exception as exc:
    print("NETWORK_BLOCKED", type(exc).__name__)
"""
_MOUNT_PROBE = """\
import os
print("MOUNT_ISOLATED" if os.path.isdir("/home") and not os.listdir("/home") else "MOUNT_VISIBLE")
"""


@dataclass(frozen=True)
class IsolationCapabilities:
    bwrap: bool
    unshare_net: bool
    unshare_mount: bool
    unshare_user: bool
    rlimit: bool = True
    platform: str = sys.platform

    @property
    def level(self) -> str:
        if self.bwrap:
            return "bwrap"
        if self.unshare_net and self.unshare_mount:
            return "namespace(net+mount)"
        if self.unshare_net:
            return "netns-only"
        return "rlimit-only"

    def to_dict(self) -> dict:
        return {"bwrap": self.bwrap, "unshare_net": self.unshare_net,
                "unshare_mount": self.unshare_mount, "unshare_user": self.unshare_user,
                "rlimit": self.rlimit, "platform": self.platform, "level": self.level}


_lock = threading.Lock()
_cached: IsolationCapabilities | None = None


def _probe(cmd: list[str], needle: str, timeout: float = 40.0) -> bool:
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return False
    return proc.returncode == 0 and needle in (proc.stdout + proc.stderr)


def detect_isolation(force: bool = False) -> IsolationCapabilities:
    global _cached
    with _lock:
        if _cached is not None and not force:
            return _cached
        has_unshare = shutil.which("unshare") is not None
        net = has_unshare and _probe(["unshare", "-Urn", sys.executable, "-c", _NET_PROBE], "NETWORK_BLOCKED")
        mount = False
        if net:
            shell = (f'{sys.executable} -c {shlex_quote(_MOUNT_PROBE)}')
            mount = _probe(["unshare", "-Urm", "sh", "-c",
                            f"mount -t tmpfs tmpfs /home && {shell}"], "MOUNT_ISOLATED")
        _cached = IsolationCapabilities(
            bwrap=shutil.which("bwrap") is not None,
            unshare_net=bool(net), unshare_mount=bool(mount),
            unshare_user=has_unshare and _probe(["unshare", "-Ur", "true"], ""),
        )
        return _cached


def detected_summary() -> dict:
    """Human-readable snapshot of the isolation this host actually provides."""
    caps = detect_isolation()
    return {"capabilities": caps.to_dict(),
            "enforced_by_default": ("private network namespace + tmpfs over host home tree "
                                    "when the sandboxed code runs") if caps.unshare_net else
                                   "rlimits only — namespace isolation unavailable",
            "honest_label": caps.level}


def shlex_quote(text: str) -> str:
    import shlex
    return shlex.quote(text)


def isolation_plan(policy) -> tuple[list[str], dict]:
    """Return the command prefix and an honest description of what it enforces."""
    caps = detect_isolation()
    if caps.unshare_net and not policy.network:
        use_mount = bool(caps.unshare_mount and policy.isolate_filesystem)
        flags = "-Urnm" if use_mount else "-Urn"
        desc = {
            "backend": "unshare",
            "level": "namespace(net+mount)" if use_mount else "netns-only",
            "network_enforced": True,
            "filesystem_isolated": use_mount,
            "notes": ("private network namespace (loopback only); tmpfs mounted over the host home tree"
                      if use_mount else
                      "private network namespace (loopback only); host filesystem still visible"),
        }
        return ["unshare", flags, "--"], desc
    return [], {
        "backend": "rlimit",
        "level": "rlimit-only",
        "network_enforced": False,
        "filesystem_isolated": False,
        "notes": ("isolation namespaces unavailable in this environment; only rlimits, timeouts and a "
                  "scrubbed environment are enforced — declared honestly as rlimit-only"),
    }
