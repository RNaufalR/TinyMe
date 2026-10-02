#!/usr/bin/env python3
"""DEPRECATED entry point - forwards to the corrected v2 pipeline.

Kept so an older command line cannot silently rebuild v1 data: every invocation
prints a notice and then runs ``scripts/prepare_data_v2.py`` with the same
arguments.  Use the v2 script directly in new work.
"""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    print("NOTICE: scripts/prepare_data.py is retired and now forwards to "
          "scripts/prepare_data_v2.py (corrected pipeline, audit 2026-10-02).",
          file=sys.stderr)
    runpy.run_path(str(REPO_ROOT / "scripts" / "prepare_data_v2.py"), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
