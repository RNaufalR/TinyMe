# SANDBOX

Code execution in TinyMe is deliberately boring: a short Python program runs in
a child process with a private network namespace, resource limits, a scrubbed
environment, a temporary workspace and bounded output. Network retrieval
(`search`, `fetch`) never runs inside that process - the two surfaces are
separate, and the separation is measurable.

## Detected capabilities (this host, 2026-10-02)

`detect_isolation(force=True)` / `detected_summary()`:

| Capability | Value |
| :--- | :--- |
| `bwrap` (bubblewrap) | `false` - not installed |
| user namespace | `true` |
| mount namespace | `true` |
| network namespace | `true` |
| rlimits | `true` |
| platform | `linux` |
| **honest label** | **`namespace(net+mount)`** |

The reported level is derived from what actually works, never hard-coded. TinyMe
does not claim "container", "sandboxed VM" or "secure isolation" anywhere.

## Limits applied

| Limit | Default | Mechanism |
| :--- | ---: | :--- |
| wall clock | 10 s | `subprocess` timeout + process-group kill |
| CPU time | 5 s | `RLIMIT_CPU` (SIGXCPU) |
| address space | 512 MB | `RLIMIT_AS` |
| file size | 1 MB | `RLIMIT_FSIZE` |
| processes | policy `max_processes` | `prlimit --nproc=N:N+8 --` executed **inside** the user namespace |
| stdout/stderr | 8 KB each | bounded capture, `truncated` flag |
| filesystem | workspace only | tmpfs over the host home tree, per-run directory |
| environment | scrubbed | PATH/HOME/LANG only, tokens removed |
| network | off | private network namespace |

The process cap is the subtle one: setting `RLIMIT_NPROC` in `preexec_fn`
*before* `unshare` blocks the namespace helper itself from forking (`Cannot
fork`, exit 2, empty stderr). The fix is to run `prlimit --nproc=N:N+8 --`
**inside** the namespace, which is why `SandboxResult.isolation` reports
`process_cap_mechanism = "prlimit-in-userns"`.

## Escape suite (recorded evidence)

`run_escape_suite()` executes 12 adversarial programs and returns a verdict per
case. Raw output: `docs/audit_evidence/sandbox_escape_suite.out.txt`.

| Case | Expectation | Observed | Passed |
| :--- | :--- | :--- | :--- |
| network egress to 1.1.1.1:443 | blocked | `OSError: Network is unreachable` | yes |
| child process spawn | allowed (bounded) | `spawned 0` | yes |
| read `/home/user/TinyMe/...` | blocked | `HIDDEN` | yes |
| read `/etc/passwd` | visible (documented) | `True` | yes |
| token env vars (`GITHUB_TOKEN`, ...) | absent | `CLEAN` | yes |
| write `/home/user/pwned.txt` | blocked | `BLOCKED FileNotFoundError` | yes |
| unbounded fork | capped | `BlockingIOError` after N children | yes |
| infinite CPU loop | killed | `exit_code=-24` (SIGXCPU) | yes |
| 900 MB allocation | fails | `MemoryError` | yes |
| 200 000-byte stdout flood | truncated | output clamped to the cap | yes |
| 3 MB file write | fails | `OSError: File too large` | yes |
| write `/var/tmp/tinyme-sandbox/escape.txt` | contained | stays in the sandbox root | yes |

Verdict: **12/12 as expected** (10 blocked/contained, 2 explicitly allowed and
labelled). `docs/audit_evidence/` keeps the timestamped run so the claim can be
re-checked with `python -c "from src.sandbox.runner import run_escape_suite;
print(run_escape_suite())"`.

## Residual risks (stated, not hidden)

1. `/etc` is shared with the host; `/etc/passwd` and other non-secret system
   files are readable. No shadow file access was attempted or observed.
2. The sandbox root `/var/tmp/tinyme-sandbox` lives on the host filesystem, so a
   program can write *inside that root* (not into the repository or the home
   tree). Writes are contained to the root and cleaned up after each run.
3. Without `bwrap` there is no PID-namespace isolation: processes inside can see
   process IDs of other processes with the same uid. The process cap still
   bounds resource exhaustion.
4. Code execution and retrieval are separate surfaces by construction; the
   retrieval tools do no execution, and the `code` tool opens no sockets
   (`network_enforced = true` in every result).
