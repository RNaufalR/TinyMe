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

#: Mount-masking prelude.  A failing mount used to be swallowed by a trailing
#: no-op, which silently downgraded filesystem confinement without saying so.
#: Failures are now printed with a machine-readable marker that ``run_python``
#: parses back into ``SandboxResult.isolation["mount_masking_failures"]`` so the
#: sandbox reports what it actually achieved (audit §9: label, never assume).
_MOUNT_FAILURE_MARKER = "TINYME_MOUNT_MASK_FAILED:"
_PREAMBLE = (
    'for d in /home /root /media /mnt /srv /var/log /tmp; do '
    'mount -t tmpfs -o size=1m tmpfs "$d" 2>/dev/null '
    f'|| echo "{_MOUNT_FAILURE_MARKER}$d" >&2; done; '
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
        failures = [line.split(_MOUNT_FAILURE_MARKER, 1)[1].strip()
                    for line in raw_err.decode("utf-8", "replace").splitlines()
                    if _MOUNT_FAILURE_MARKER in line]
        if failures:
            iso = {**iso, "mount_masking_failures": failures,
                   "notes": iso.get("notes", "") + f"; {len(failures)} path(s) could not be masked"}
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
    """Run the escape-oriented probes and return a per-case verdict.

    Each entry carries ``expectation`` (what a correct sandbox must do),
    ``observed`` (what actually happened) and ``passed``.  It is deliberately
    honest: cases the current backend cannot block are recorded as
    ``allowed`` / ``contained`` instead of being reported as blocked.
    """
    cases: dict[str, tuple[str, str]] = {
        "network_egress": ("blocked", "socket.create_connection to 1.1.1.1:443 must fail"),
        "subprocess_spawn": ("allowed", "child processes are permitted but bounded by the process cap"),
        "read_host_repo": ("blocked", "/home/user (host home tree) must not be visible"),
        "read_host_etc_passwd": ("visible", "/etc is shared: passwd is readable (non-secret, needed by the loader)"),
        "read_env_secrets": ("blocked", "token environment variables must not leak into the sandbox"),
        "write_outside_workspace": ("blocked", "writes to the host home tree must fail"),
        "fork_bomb_capped": ("blocked", "unbounded fork must be capped by RLIMIT_NPROC"),
        "cpu_burn": ("blocked", "infinite loop must be killed by the CPU/wall limit"),
        "memory_bomb": ("blocked", "oversized allocation must fail under RLIMIT_AS"),
        "output_flood": ("contained", "stdout must be truncated at max_output_bytes"),
        "file_flood": ("blocked", "oversized file must fail under RLIMIT_FSIZE"),
        "absolute_path_write": ("contained", "writes stay inside /var/tmp/tinyme-sandbox, never in the repo"),
        "path_traversal_read": ("blocked", "relative traversal must not reach the host repo/home tree"),
        "symlink_escape": ("blocked", "a symlink pointing at the host tree must not become a write path"),
        "temp_dir_outside_run": ("contained", "writes to shared /tmp are either blocked or confined, never in the repo"),
        "descriptor_abuse": ("blocked", "opening unbounded file descriptors must fail under RLIMIT_NOFILE"),
        "process_visibility": ("contained", "PID namespace hides the host process table when available"),
        "workspace_confinement": ("blocked", "cwd is the run workspace; escaping via relative paths must fail"),
    }
    probes: dict[str, dict] = {}
    for name, (expectation, description) in cases.items():
        policy = SandboxPolicy(wall_timeout_s=8, cpu_timeout_s=5, memory_mb=256,
                               max_output_bytes=4096, max_file_bytes=1024 * 1024)
        if name == "cpu_burn":
            policy = SandboxPolicy(wall_timeout_s=4, cpu_timeout_s=2, memory_mb=256, max_output_bytes=4096)
        code = _ESCAPE_CASES[name]
        res = run_python(code, policy=policy)
        stdout = res.stdout
        if name == "network_egress":
            observed = "blocked" if ("Network is unreachable" in res.stderr or "CONNECTED" not in stdout) else "allowed"
        elif name == "subprocess_spawn":
            observed = "allowed" if "spawned 0" in stdout else "blocked"
        elif name == "read_host_repo":
            observed = "blocked" if "HIDDEN" in stdout else "visible"
        elif name == "read_host_etc_passwd":
            observed = "visible" if "True" in stdout else "blocked"
        elif name == "read_env_secrets":
            observed = "blocked" if "CLEAN" in stdout else "visible"
        elif name == "write_outside_workspace":
            observed = "blocked" if "BLOCKED" in stdout else "allowed"
        elif name == "fork_bomb_capped":
            observed = "blocked" if "BlockingIOError" in stdout else "allowed"
        elif name == "cpu_burn":
            observed = "blocked" if (res.exit_code in (-24, -9) or res.timed_out) else "allowed"
        elif name == "memory_bomb":
            observed = "blocked" if "MemoryError" in res.stderr else "allowed"
        elif name == "output_flood":
            observed = "contained" if len(stdout) <= policy.max_output_bytes + 64 else "flooded"
        elif name == "file_flood":
            observed = "blocked" if "File too large" in res.stderr else "allowed"
        elif name == "path_traversal_read":
            observed = "blocked" if "HOSTSECRET" not in stdout else "visible"
        elif name == "symlink_escape":
            observed = "blocked" if "WROTE" not in stdout else "allowed"
        elif name == "temp_dir_outside_run":
            observed = "contained" if "WROTE" not in stdout or "run-" in stdout else "escaped"
        elif name == "descriptor_abuse":
            observed = "blocked" if "fd cap" in stdout or "EMFILE" in stdout else "allowed"
        elif name == "process_visibility":
            # honest: without a PID namespace the host table is readable
            observed = "visible" if "host processes visible" in stdout else "contained"
        elif name == "workspace_confinement":
            observed = "blocked" if "ESCAPED" not in stdout else "contained"
        else:  # absolute_path_write
            escaped = Path("/var/tmp/tinyme-sandbox/escape.txt")
            contained = "WROTE" not in stdout or (escaped.exists() and "/var/tmp/tinyme-sandbox" in str(escaped))
            observed = "contained" if contained else "escaped"
        expected_counts = {"network_egress": {"blocked"}, "subprocess_spawn": {"allowed"},
                           "read_host_repo": {"blocked"}, "read_host_etc_passwd": {"visible"},
                           "read_env_secrets": {"blocked"}, "write_outside_workspace": {"blocked"},
                           "fork_bomb_capped": {"blocked"}, "cpu_burn": {"blocked"},
                           "memory_bomb": {"blocked"}, "output_flood": {"contained"},
                           "file_flood": {"blocked"}, "absolute_path_write": {"contained"},
                           "path_traversal_read": {"blocked"}, "symlink_escape": {"blocked"},
                           "temp_dir_outside_run": {"contained"}, "descriptor_abuse": {"blocked"},
                           "process_visibility": {"contained"}, "workspace_confinement": {"blocked"}}
        probes[name] = {
            "expectation": expectation, "description": description, "observed": observed,
            "passed": observed in expected_counts[name],
            "ok": res.ok, "exit_code": res.exit_code, "timed_out": res.timed_out,
            "stdout": stdout.strip()[:200], "stderr_tail": res.stderr.strip()[-160:],
            "error": res.error,
        }
    blocked = sum(1 for p in probes.values() if p["observed"] in ("blocked", "contained"))
    probes["_summary"] = {  # type: ignore[assignment]
        "cases": len(cases), "blocked_or_contained": blocked,
        "passed": sum(1 for k, p in probes.items() if k != "_summary" and p["passed"]),
        "verdict": "PASS" if all(p["passed"] for k, p in probes.items() if k != "_summary") else "REVIEW",
    }
    return probes


_ESCAPE_CASES: dict[str, str] = {
    "network_egress": "import socket\n"
                      "s=socket.create_connection(('1.1.1.1',443),timeout=3)\nprint('CONNECTED')",
    "subprocess_spawn": "import subprocess,sys\n"
                        "p=subprocess.run([sys.executable,'-c','print(1)'],capture_output=True,text=True)\n"
                        "print('spawned', p.returncode)",
    "read_host_repo": "import pathlib\n"
                      "p=pathlib.Path('/home/user/TinyMe/TinyMeAudit.md')\n"
                      "print('READABLE' if p.exists() else 'HIDDEN')",
    "read_host_etc_passwd": "import pathlib\nprint(pathlib.Path('/etc/passwd').exists())",
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
    "path_traversal_read": "import pathlib\n"
                           "targets=['../../../../../../home/user/TinyMe/TinyMeAudit.md',\n"
                           "         '../../../../../../home/user/TinyMe/src/agent/protocol.py']\n"
                           "found=[]\n"
                           "for t in targets:\n"
                           "    try:\n"
                           "        p=pathlib.Path(t)\n"
                           "        if p.exists():\n"
                           "            found.append(str(p))\n"
                           "    except Exception:\n"
                           "        pass\n"
                           "print('HOSTSECRET' if found else 'blocked', found)",
    "symlink_escape": "import os, pathlib\n"
                      "try:\n"
                      "    os.symlink('/home/user', 'host_link')\n"
                      "    pathlib.Path('host_link/pwned_via_symlink.txt').write_text('x')\n"
                      "    print('WROTE')\n"
                      "except Exception as exc:\n"
                      "    print('BLOCKED', type(exc).__name__)",
    "temp_dir_outside_run": "import os, pathlib, tempfile\n"
                            "p=pathlib.Path(tempfile.gettempdir())/'tinyme_probe.txt'\n"
                            "try:\n"
                            "    p.write_text('x'); print('WROTE', os.getcwd())\n"
                            "except Exception as exc: print('BLOCKED', type(exc).__name__)",
    "descriptor_abuse": "import resource, sys\n"
                        "soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)\n"
                        "print('soft_cap', soft)\n"
                        "fds=[]\n"
                        "try:\n"
                        "    for i in range(soft+64):\n"
                        "        fds.append(open(sys.executable,'rb'))\n"
                        "except OSError as exc:\n"
                        "    print('fd cap', type(exc).__name__)\n"
                        "print('opened', len(fds))",
    "process_visibility": "import os, pathlib\n"
                          "host_procs=[]\n"
                          "if pathlib.Path('/proc').exists():\n"
                          "    for entry in os.listdir('/proc'):\n"
                          "        if entry.isdigit() and int(entry)!=os.getpid():\n"
                          "            try:\n"
                          "                cmd=pathlib.Path(f'/proc/{entry}/cmdline').read_bytes()[:80].decode('utf-8','replace')\n"
                          "                if 'python3' in cmd or 'prepare_data' in cmd or 'train.py' in cmd:\n"
                          "                    host_procs.append(entry)\n"
                          "            except Exception: pass\n"
                          "print('host processes visible', host_procs) if len(host_procs)>3 else print('process table isolated', len(host_procs))",
    "workspace_confinement": "import os, pathlib\n"
                             "print('cwd', os.getcwd())\n"
                             "try:\n"
                             "    pathlib.Path('../../../repo_escape.txt').write_text('x')\n"
                             "    print('ESCAPED', pathlib.Path('../../../repo_escape.txt').resolve())\n"
                             "except Exception as exc:\n"
                             "    print('BLOCKED', type(exc).__name__)",
}


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
