"""POSIX resource limits applied to sandboxed children (audit §32).

``apply_rlimits`` is intended for ``subprocess``'s ``preexec_fn`` so the limits
are installed between fork and exec and are inherited by the whole process tree
created by the unshare/exec chain.
"""
from __future__ import annotations

import os
import resource


def _count_uid_processes() -> int:
    """Number of processes owned by this uid (used to calibrate RLIMIT_NPROC).

    RLIMIT_NPROC is enforced per *real user id* across the whole machine, not per
    process tree, so a fixed small value would kill the sandbox immediately in a
    container where the uid already owns dozens of processes.  The limit is
    therefore expressed as "current count + policy.max_processes", which still
    bounds how many processes the sandbox itself can create.
    """
    uid = os.getuid()
    count = 0
    try:
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status", "r") as fh:
                    for line in fh:
                        if line.startswith("Uid:"):
                            if int(line.split()[1]) == uid:
                                count += 1
                            break
            except OSError:
                continue
    except OSError:
        return 64
    return max(count, 1)


def apply_limits(policy, *, include_nproc: bool = False) -> None:  # pragma: no cover
    """Set hard+soft rlimits for the child process.

    ``RLIMIT_NPROC`` is *not* applied by default: it is enforced per real uid
    across the whole machine, so a small value would prevent the namespace
    helper from forking at all.  The runner applies the process cap with
    ``prlimit`` *inside* a fresh user namespace instead (verified: a fork bomb
    stops after ~max_processes children); ``include_nproc=True`` is the fallback
    when namespaces are unavailable.
    """
    if policy.cpu_timeout_s > 0:
        cpu = int(policy.cpu_timeout_s)
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 2))
    if policy.memory_mb > 0:
        mem = int(policy.memory_mb) * 1024 * 1024
        resource.setrlimit(resource.RLIMIT_AS, (mem, mem))
    if policy.max_file_bytes > 0:
        resource.setrlimit(resource.RLIMIT_FSIZE, (int(policy.max_file_bytes),) * 2)
    if policy.max_open_files > 0:
        resource.setrlimit(resource.RLIMIT_NOFILE, (int(policy.max_open_files),) * 2)
    if policy.max_processes > 0 and include_nproc:
        try:
            budget = _count_uid_processes() + int(policy.max_processes)
            resource.setrlimit(resource.RLIMIT_NPROC, (budget, budget + 8))
        except (ValueError, OSError):
            pass  # not permitted on some platforms/kernels
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def limits_report(policy) -> dict:
    return {"RLIMIT_CPU": policy.cpu_timeout_s, "RLIMIT_AS_MB": policy.memory_mb,
            "RLIMIT_FSIZE_B": policy.max_file_bytes, "RLIMIT_NOFILE": policy.max_open_files,
            "RLIMIT_NPROC": (policy.max_processes
                             if policy.max_processes else 0), "RLIMIT_CORE": 0,
            "wall_timeout_s": policy.wall_timeout_s, "max_output_bytes": policy.max_output_bytes}
