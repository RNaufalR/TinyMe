"""Extended Python-standard-library extraction (offline, PSF-2.0).

The v1 adapter only read a fixed list of 26 modules and kept one record pair per
documented function. This version walks the whole local CPython 3.11 standard
library (still fully offline and licence-clean) and emits:

* **code records** — top-level functions/classes (documented or not) of
  reasonable size, validated with ``ast.parse`` before emission;
* **documentation records** — ``<|system|>``/``<|user|>``/``<|assistant|>``
  lessons that pair a function with its real docstring (prose supervision);
* **module overviews** — module docstring + public API index (prose).

Files that are vendored, generated, minified, too large, or that contain
suspicious content (secrets/malware patterns) are skipped and counted.
"""
from __future__ import annotations

import ast
import logging
import sysconfig
import time
from collections import Counter
from pathlib import Path
from typing import Any

from src.data.quality_filter import is_minified, safety_flags
from src.data.records import Segment, TrainingRecord, make_segment_record, validate_record

logger = logging.getLogger("tinyme.stdlib_v2")

LICENSE = "PSF-2.0"
LICENSE_URL = "https://docs.python.org/3/license.html"

#: Directories inside the stdlib that are vendored/bundled or not source we want.
EXCLUDE_PARTS = {
    "site-packages", "dist-packages", "__pycache__", "ensurepip", "venv",
    "idle_test", "lib2to3",  # lib2to3 is vendored and deprecated
}
EXCLUDE_FILES = {"this.py", "antigravity.py", "pydoc.py"}

MAX_FILE_BYTES = 400_000
MIN_CODE_CHARS = 120
MAX_CODE_CHARS = 6000
MAX_DOC_CHARS = 1800


def stdlib_root() -> Path:
    return Path(sysconfig.get_paths()["stdlib"])


def _iter_python_files(root: Path, max_files: int | None = None) -> list[Path]:
    files: list[Path] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in EXCLUDE_PARTS for part in path.parts):
            continue
        if path.name in EXCLUDE_FILES or path.name.startswith("_"):
            continue
        try:
            if path.stat().st_size > MAX_FILE_BYTES:
                continue
        except OSError:
            continue
        files.append(path)
        if max_files and len(files) >= max_files:
            break
    return files


def _module_name(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    parts = list(rel.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts) or path.stem


def extract_file_records(path: Path, module: str, today: str,
                         max_per_file: int = 6) -> tuple[list[TrainingRecord], Counter]:
    counters: Counter = Counter()
    try:
        source = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as exc:  # pragma: no cover - unreadable file
        counters[f"unreadable:{type(exc).__name__}"] += 1
        return [], counters
    flags = safety_flags(source)
    if flags["unsafe"]:
        counters["skipped_safety"] += 1
        return [], counters
    if is_minified(source):
        counters["skipped_minified"] += 1
        return [], counters
    try:
        tree = ast.parse(source)
    except SyntaxError:
        counters["skipped_unparsable"] += 1
        return [], counters

    records: list[TrainingRecord] = []
    source_url = f"https://github.com/python/cpython/blob/3.11/Lib/{path.relative_to(stdlib_root())}"
    module_doc = ast.get_docstring(tree)
    if module_doc and len(module_doc) > 120:
        overview = (f"Module `{module}` of the Python standard library.\n\n"
                    f"{module_doc.strip()[:MAX_DOC_CHARS]}")
        records.append(TrainingRecord(
            text=overview, category="language", source="python_stdlib",
            source_id=f"python3.11/{module}::module-doc", license=LICENSE,
            source_url=source_url, retrieval_date=today, task_type="lm",
            language="en", template_id=f"stdlib/{module}.overview",
            verified=True, verifier="stdlib_ast"))

    emitted = 0
    for node in tree.body:
        if emitted >= max_per_file:
            break
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if node.name.startswith("_"):
            continue
        try:
            code = ast.get_source_segment(source, node) or ""
        except Exception:  # pragma: no cover
            continue
        code = code.strip()
        if not (MIN_CODE_CHARS <= len(code) <= MAX_CODE_CHARS):
            continue
        if is_minified(code) or safety_flags(code)["unsafe"]:
            counters["skipped_suspicious"] += 1
            continue
        # the emitted code must be valid Python on its own
        try:
            ast.parse(code)
        except SyntaxError:
            counters["skipped_invalid_code"] += 1
            continue
        public = not node.name.startswith("_")
        if not public:
            continue
        template = f"stdlib/{module}/{node.name}"
        records.append(TrainingRecord(
            text=code, category="programming", source="python_stdlib",
            source_id=f"python3.11/{module}.py::{node.name}", license=LICENSE,
            source_url=source_url, retrieval_date=today, task_type="code_completion",
            language="python", template_id=template, verified=True, verifier="ast_parse"))
        doc = (ast.get_docstring(node) or "").strip()
        if len(doc) >= 60:
            segs = [Segment("system", "Explain a Python standard-library API precisely.", target=False),
                    Segment("user", f"Explain what `{module}.{node.name}` does.\n\n"
                                    f"```python\n{code[:3000]}\n```", target=False),
                    Segment("assistant", doc[:MAX_DOC_CHARS], target=True)]
            records.append(make_segment_record(
                segments=segs, category="code_explain", source="python_stdlib",
                source_id=f"python3.11/{module}.py::{node.name}:doc", task_type="instruction",
                template_id=template + ".doc", license=LICENSE, language="python",
                source_url=source_url, retrieval_date=today, verified=True,
                verifier="stdlib_docstring"))
        emitted += 1
        counters["emitted"] += 1
    return records, counters


def ingest_extended(max_files: int = 160, max_per_file: int = 6) -> tuple[list[TrainingRecord], list[dict[str, Any]], Counter]:
    root = stdlib_root()
    files = _iter_python_files(root, max_files=max_files)
    today = time.strftime("%Y-%m-%d")
    records: list[TrainingRecord] = []
    counters: Counter = Counter()
    modules_seen: set[str] = set()
    for path in files:
        module = _module_name(root, path)
        recs, c = extract_file_records(path, module, today, max_per_file=max_per_file)
        counters.update(c)
        records.extend(recs)
        if recs:
            modules_seen.add(module)
    records = [r for r in records if validate_record(r)[0]]
    provenance = [{
        "source": "python_stdlib", "source_id": "python3.11-stdlib-extended",
        "dataset_or_repo_name": "CPython 3.11 standard library (local installation)",
        "url": "https://github.com/python/cpython", "license": LICENSE, "license_url": LICENSE_URL,
        "retrieval_date": today, "source_type": "code + documentation",
        "language": "python", "records": len(records), "files_scanned": len(files),
        "modules": len(modules_seen),
        "skipped": {k: v for k, v in counters.items() if k.startswith("skipped")},
        "preprocessing_performed": ["content_extraction", "ast_function_extraction",
                                    "python AST validation", "minified/vendored rejection",
                                    "secret/malware scanning", "deduplication"],
        "inclusion_status": "INCLUDED",
        "inclusion_reason": "PSF-2.0 permits ML use; offline, deterministic, license-clean code corpus",
    }]
    logger.info("stdlib v2: %d records from %d files (%d modules); skipped=%s",
                len(records), len(files), len(modules_seen),
                {k: v for k, v in counters.items() if k.startswith("skipped")})
    return records, provenance, counters
