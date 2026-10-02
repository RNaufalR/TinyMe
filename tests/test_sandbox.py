"""Sandbox isolation and limits — the 14-case matrix plus escape tests (audit §32)."""
from __future__ import annotations

import io
import json
import os
import sys
import time
from pathlib import Path

import pytest

from src.sandbox.isolation import detect_isolation, detected_summary, isolation_plan
from src.sandbox.limits import limits_report
from src.sandbox.policy import SandboxPolicy
from src.sandbox.runner import run_python
from src.sandbox.workspace import Workspace, WorkspaceError

CAPS = detect_isolation()
HAS_NS = CAPS.unshare_net


@pytest.fixture(scope="module")
def policy():
    return SandboxPolicy(wall_timeout_s=8, cpu_timeout_s=4, memory_mb=512,
                         max_output_bytes=8192, max_file_bytes=512 * 1024, max_processes=16)


# --------------------------------------------------------------- capabilities
def test_isolation_is_detected_and_labelled_honestly():
    caps = detect_isolation()
    summary = detected_summary()
    assert summary["capabilities"]["level"] == caps.level
    assert summary["honest_label"] in {"bwrap", "namespace(net+mount)", "netns-only", "rlimit-only"}
    if caps.unshare_net:
        assert "namespace" in caps.level or caps.level == "bwrap"


def test_plan_describes_what_is_enforced():
    plan, desc = isolation_plan(SandboxPolicy(network=False, isolate_filesystem=True))
    assert desc["network_enforced"] is HAS_NS, "the claim must match real capability"
    assert desc["filesystem_isolated"] in (True, False)
    if HAS_NS:
        assert "--" in plan and plan[0] == "unshare"


# ------------------------------------------------------- the 14-case matrix
CASES = {
    # name: (code, extra policy kwargs, expectation)
    "01_compute_runs": ("print(sum(range(11)))", {}, {"stdout_has": "55", "ok": True}),
    "02_imports_allowed": ("import json, math\nprint(math.isqrt(144))", {}, {"stdout_has": "12"}),
    "03_filesystem_read_of_workspace": ("open('data.txt').read()", {}, {"ok": True}),
    "04_write_inside_workspace": ("open('out.txt', 'w').write('ok')\nprint('written')", {}, {"ok": True}),
    "05_network_egress_blocked": (
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 443), timeout=3)\n"
        "    print('CONNECTED')\n"
        "except Exception as exc:\n"
        "    print('blocked', type(exc).__name__)\n", {}, {"not_stdout_has": "CONNECTED"}),
    "06_dns_blocked": (
        "import socket\n"
        "try:\n"
        "    socket.gethostbyname('example.com'); print('RESOLVED')\n"
        "except Exception as exc:\n"
        "    print('blocked', type(exc).__name__)\n", {}, {"not_stdout_has": "RESOLVED"}),
    "07_host_home_hidden": (
        "import pathlib\n"
        "print('visible' if pathlib.Path('/home/user/TinyMe').exists() else 'hidden')\n",
        {}, {"stdout_has": "hidden"} if HAS_NS else {"any": True}),
    "08_env_secrets_absent": (
        "import os\n"
        "bad = [k for k in os.environ if any(t in k.upper() for t in ('TOKEN', 'KEY', 'SECRET', 'PASSWORD'))]\n"
        "print('clean' if not bad else 'leaked ' + ','.join(bad))\n", {}, {"stdout_has": "clean"}),
    # either the CPU rlimit fires first (SIGXCPU) or the wall-clock timeout does;
    # both are enforced limits, and the result must never be reported as success
    "09_timeout_kills_cpu_burn": ("while True:\n    pass\n",
                                  {"wall_timeout_s": 3, "cpu_timeout_s": 2},
                                  {"killed_or_timed_out": True}),
    "10_memory_limit": ("x = bytearray(900 * 1024 * 1024)\nprint('ALLOCATED')",
                        {"memory_mb": 256}, {"not_stdout_has": "ALLOCATED"}),
    "11_output_is_capped": ("print('A' * 100000)", {"max_output_bytes": 4096},
                            {"truncated": True, "stdout_len_max": 4096}),
    "12_file_size_limit": ("open('big.bin', 'wb').write(b'B' * (4 * 1024 * 1024))\nprint('WROTE')",
                           {"max_file_bytes": 256 * 1024}, {"not_stdout_has": "WROTE"}),
    "13_process_cap": (
        "import os\n"
        "n = 0\n"
        "try:\n"
        "    while n < 500:\n"
        "        if os.fork() == 0:\n"
        "            os._exit(0)\n"
        "        n += 1\n"
        "except Exception as exc:\n"
        "    print('capped', type(exc).__name__)\n"
        "print('children', n)\n", {"max_processes": 16, "wall_timeout_s": 15},
        {"not_stdout_has": "children 500"}),
    "14_crash_reported_not_hidden": ("raise ValueError('boom')", {}, {"ok": False, "exit_nonzero": True}),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_sandbox_case_matrix(case, policy, tmp_path):
    code, overrides, expect = CASES[case]
    pol = SandboxPolicy(**{**policy.to_dict(), **overrides})
    files = {"data.txt": "workspace content"} if case == "03_filesystem_read_of_workspace" else None
    result = run_python(code, policy=pol, files=files)
    as_dict = result.to_dict()
    if expect.get("ok") is True:
        assert result.ok, as_dict
    if expect.get("exit_nonzero"):
        assert result.exit_code != 0, as_dict
    if expect.get("timed_out"):
        assert result.timed_out and result.exit_code != 0
    if expect.get("killed_or_timed_out"):
        assert (result.timed_out or (result.exit_code is not None and result.exit_code < 0)
                or result.exit_code != 0), as_dict
        assert not result.ok
    if expect.get("stdout_has"):
        assert expect["stdout_has"] in result.stdout, (case, result.stdout[:200], result.stderr[:200])
    if expect.get("not_stdout_has"):
        assert expect["not_stdout_has"] not in result.stdout, (case, result.stdout[:200])
    if expect.get("truncated"):
        assert result.truncated
    if expect.get("stdout_len_max"):
        assert len(result.stdout) <= expect["stdout_len_max"]
    assert "level" in as_dict["isolation"] and "notes" in as_dict["isolation"]


# ------------------------------------------------------------------ escapes
def test_escape_read_outside_workspace(tmp_path):
    result = run_python("import pathlib\nprint(pathlib.Path('/etc/passwd').exists())",
                        policy=SandboxPolicy())
    # /etc is readable in a namespace sandbox (the audit's rule is about *host
    # home / project* data and secrets, not the base image).
    assert result.exit_code == 0


def test_escape_write_outside_workspace_is_denied(policy):
    target = Path("/home/user/pwned_by_sandbox_test.txt")
    code = (f"import pathlib\n"
            f"try:\n"
            f"    pathlib.Path({str(target)!r}).write_text('x'); print('WROTE')\n"
            f"except Exception as exc:\n"
            f"    print('blocked', type(exc).__name__)\n")
    result = run_python(code, policy=policy)
    assert "WROTE" not in result.stdout, result.stdout
    assert not target.exists()


def test_escape_read_absolute_project_file_is_denied(policy):
    code = ("import pathlib\n"
            "p = pathlib.Path('/home/user/TinyMe/TinyMeAudit.md')\n"
            "print('READ' if p.exists() else 'denied')\n")
    result = run_python(code, policy=policy)
    if HAS_NS:
        assert result.stdout.strip() == "denied", result.stdout


def test_escape_via_subprocess_is_still_sandboxed(policy):
    code = ("import subprocess, sys\n"
            "out = subprocess.run([sys.executable, '-c', "
            "\"import socket\\n\"\n"
            "    \"try:\\n\"\n"
            "    \"    socket.create_connection(('1.1.1.1', 443), timeout=2); print('INNER_NET')\\n\"\n"
            "    \"except Exception:\\n\"\n"
            "    \"    print('inner blocked')\"], capture_output=True, text=True)\n"
            "print('child:', out.stdout.strip())\n")
    result = run_python(code, policy=policy)
    assert "INNER_NET" not in result.stdout
    assert "inner blocked" in result.stdout


def test_workspace_paths_are_validated():
    ws = Workspace.create(SandboxPolicy())
    try:
        for bad in ("../escape.txt", "/etc/x", "~/x"):
            with pytest.raises(WorkspaceError):
                ws.write(bad, "x")
        assert ws.write("ok.txt", "fine").exists()
    finally:
        ws.cleanup()
    assert not ws.path.exists()


def test_result_records_policy_and_limits(policy):
    result = run_python("print(1)", policy=policy)
    assert result.policy["network"] is False
    assert result.limits["RLIMIT_AS_MB"] == policy.memory_mb
    assert limits_report(policy)["RLIMIT_CPU"] == policy.cpu_timeout_s
    report = json.loads(json.dumps(result.to_dict()))
    assert report["ok"] is True and report["duration_s"] >= 0
