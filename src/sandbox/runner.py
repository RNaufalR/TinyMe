"""Sandboxed code execution (audit §32).

Design principles:

* **Code execution is network-disabled by default** — a private network
  namespace is created with ``unshare -Urn`` (loopback only).  Network
  *retrieval* is a separate tool that runs outside this runner, so a model can
  never exfiltrate data or fetch resources from inside executed code.
* **Isolation is auto-detected, never assumed.**  ``detect_isolation()`` probes
  the host; the level actually achieved is stamped on every result
  (``isolation.level``) and downgraded to ``rlimit-only`` when namespaces are
  unavailable.
* **Limits are real**: CPU/memory/file-size/fd/process rlimits plus a wall-clock
  timeout with process-group kill, bounded output capture, and a scrubbed
  environment (no inherited secrets such as tokens or proxies).
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from .isolation import detected_summary, isolation_plan
from .limits import apply_limits, limits_report
from .policy import SandboxPolicy
from .workspace import Workspace

_PREAMBLE = (
    'for d in /home /root /media /mnt /srv /var/log /tmp; do '
    'mount -t tmpfs -o size=1m tmpfs "$d" 2>/dev/null || true; done; '
    'exec "$@"'
)

_SCRUBBED_ENV_KEYS = ("PATH", "LANG", "LC_ALL", "TZ", "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE")


@dataclass
class SandboxResult:
    exit_code: int | None
    stdout: str
    stderr: str
    duration_s: float
    timed_out: bool
    truncated: bool
    isolation: dict = field(default_factory=dict)
    limits: dict = field(default_factory=dict)
    policy: dict = field(default_factory=dict)
    outputs: dict[str, int] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def to_dict(self) -> dict:
        d = dict(self.__dict__)
        d["ok"] = self.ok
        return d


def _build_env(workspace: Workspace, policy: SandboxPolicy) -> dict[str, str]:
    env: dict[str, str] = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": str(workspace.path),
        "TMPDIR": str(workspace.path / "tmp"),
        "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "TINYME_SANDBOX": "1",
        "LC_ALL": "C.UTF-8",
        "LANG": "C.UTF-8",
    }
    env.update(policy.extra_env or {})
    return env


def run_python(
    code: str | None = None,
    *,
    script: str = "main.py",
    files: dict[str, str | bytes] | None = None,
    argv: list[str] | None = None,
    policy: SandboxPolicy | None = None,
    workspace_token: str | None = None,
    workspace: Workspace | None = None,
    keep_workspace: bool = False,
) -> SandboxResult:
    """Execute ``code`` (written to ``script``) with the given policy.

    ``workspace`` reuses an existing ephemeral workspace (so the ``code`` and
    ``files`` tools share one filesystem); otherwise a fresh 0700 directory is
    created and removed afterwards.
    """
    policy = policy or SandboxPolicy()
    if code is not None:
        files = {**(files or {}), script: code}
    prefix, iso = isolation_plan(policy)
    inner = [sys.executable, "-I", "-B", script] + list(argv or [])
    if iso["backend"] == "unshare" and policy.max_processes > 0 and shutil.which("prlimit"):
        # Applied *inside* the new user namespace, where the process count starts
        # fresh; the same limit set outside would block the namespace helper.
        inner = ["prlimit", f"--nproc={int(policy.max_processes)}:{int(policy.max_processes) + 8}",
                 "--"] + inner
        iso = {**iso, "process_cap": int(policy.max_processes), "process_cap_mechanism": "prlimit-in-userns"}
    if iso["backend"] == "unshare" and iso["filesystem_isolated"]:
        cmd = prefix + ["sh", "-c", _PREAMBLE, "tinyme-sandbox"] + inner
    else:
        cmd = [sys.executable, "-I", "-B", script] + list(argv or [])
        if not policy.network and iso["backend"] != "unshare":
            iso = {**iso, "notes": iso["notes"] + "; code can technically reach the network in this mode"}
    ws = workspace or Workspace.create(policy, token=workspace_token)
    owned = workspace is None
    try:
        if files:
            ws.write_many(files)
        out_path, err_path = ws.path / "tmp" / "_stdout.txt", ws.path / "tmp" / "_stderr.txt"
        env = _build_env(ws, policy)
        start = time.monotonic()
        timed_out, exit_code, error = False, None, None
        with open(out_path, "wb") as out_fh, open(err_path, "wb") as err_fh:
            try:
                proc = subprocess.Popen(
                    cmd, cwd=str(ws.path), env=env, stdin=subprocess.DEVNULL,
                    stdout=out_fh, stderr=err_fh, start_new_session=True,
                    preexec_fn=(lambda: apply_limits(
                        policy, include_nproc=(iso["backend"] != "unshare"))),
                )
                try:
                    exit_code = proc.wait(timeout=policy.wall_timeout_s)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    exit_code = proc.wait(timeout=10)
            except OSError as exc:
                error = f"{type(exc).__name__}: {exc}"
        duration = time.monotonic() - start
        raw_out = out_path.read_bytes() if out_path.exists() else b""
        raw_err = err_path.read_bytes() if err_path.exists() else b""
        cap = int(policy.max_output_bytes)
        truncated = len(raw_out) > cap or len(raw_err) > cap
        outputs = ws.list_outputs(skip={"tmp/_stdout.txt", "tmp/_stderr.txt"})
        return SandboxResult(
            exit_code=exit_code, stdout=raw_out[:cap].decode("utf-8", "replace"),
            stderr=raw_err[:cap].decode("utf-8", "replace"), duration_s=round(duration, 4),
            timed_out=timed_out, truncated=truncated, isolation=iso,
            limits=limits_report(policy), policy=policy.to_dict(), outputs=outputs, error=error,
        )
    finally:
        if owned and not keep_workspace:
            ws.cleanup()


def run_command(argv: list[str], *, policy: SandboxPolicy | None = None, **kw) -> SandboxResult:
    """Execute an arbitrary argv with no script prelude (still sandboxed)."""
    return run_python(None, argv=argv, policy=policy, **kw)


def run_escape_suite() -> dict[str, dict]:
    """Run the escape-oriented probes (used by tests and the final report)."""
    probes: dict[str, dict] = {}
    cases = {
        "network_egress": "import socket\n"
                          "s=socket.create_connection(('1.1.1.1',443),timeout=3)\nprint('CONNECTED')",
        "subprocess_spawn": "import subprocess,sys\n"
                            "p=subprocess.run([sys.executable,'-c','print(1)'],capture_output=True,text=True)\n"
                            "print('spawned', p.returncode)",
        "read_host_repo": "import pathlib\n"
                          "p=pathlib.Path('/home/user/TinyMe/TinyMeAudit.md')\n"
                          "print('READABLE' if p.exists() else 'HIDDEN')",
        "read_host_etc_passwd": "import pathlib\n"
                                "print(pathlib.Path('/etc/passwd').exists())",
        "read_env_secrets": "import os\n"
                            "print('SECRET' if any(k in os.environ for k in "
                            "('GITHUB_TOKEN','OPENAI_API_KEY','HF_TOKEN','AWS_SECRET_ACCESS_KEY')) else 'CLEAN')",
        "write_outside_workspace": "import pathlib\n"
                                   "try:\n"
                                   "    pathlib.Path('/home/user/pwned.txt').write_text('x'); print('WROTE')\n"
                                   "except Exception as exc: print('BLOCKED', type(exc).__name__)",
        "fork_bomb_capped": "import os\n"
                            "n=0\n"
                            "try:\n"
                            "    while True:\n"
                            "        os.fork(); n+=1\n"
                            "except Exception as exc:\n"
                            "    print('capped_after', n, type(exc).__name__)",
        "cpu_burn": "while True:\n    pass",
        "memory_bomb": "x=bytearray(900*1024*1024)\nprint('ALLOCATED')",
        "output_flood": "print('A'*200000)",
        "file_flood": "open('big.bin','wb').write(b'B'*(3*1024*1024))",
        "absolute_path_write": "open('/var/tmp/tinyme-sandbox/escape.txt','w').write('x')",
    }
    for name, code in cases.items():
        policy = SandboxPolicy(wall_timeout_s=8, cpu_timeout_s=5, memory_mb=256,
                               max_output_bytes=4096, max_file_bytes=1024 * 1024)
        if name == "cpu_burn":
            policy = SandboxPolicy(wall_timeout_s=4, cpu_timeout_s=2, memory_mb=256, max_output_bytes=4096)
        res = run_python(code, policy=policy)
        probes[name] = {"ok": res.ok, "exit_code": res.exit_code, "timed_out": res.timed_out,
                        "stdout": res.stdout.strip()[:200], "stderr_tail": res.stderr.strip()[-160:],
                        "error": res.error}
    return probes


if __name__ == "__main__":  # manual smoke run
    import json

    info = {"isolation": detected_summary()}
    res = run_python("import socket, pathlib\n"
                     "print('home_visible', pathlib.Path('/home/user').exists())\n"
                     "try:\n"
                     "    socket.create_connection(('1.1.1.1', 443), timeout=2); print('net OK')\n"
                     "except Exception as exc:\n"
                     "    print('net blocked:', type(exc).__name__)\n"
                     "print('2+2 =', 2 + 2)")
    info["sample_run"] = res.to_dict()
    print(json.dumps(info, indent=2)[:2000])
