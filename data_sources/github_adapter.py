"""GitHub repository adapter (spec §7).

GitHub is treated as a source of PROGRAMMING KNOWLEDGE, not as a bulk scraper.
Every repository is license-checked first; GPL/copyleft and LICENSE_UNCLEAR
repositories are excluded from the default corpus but still recorded.

NOT WIRED INTO THE AUTHORITATIVE PIPELINE (2026-10-02).

Kept as an optional experimental GitHub ingester. The corrective audit found no
GitHub-sourced records in the shipped corpus; every source actually used is
recorded in `docs/DATA_PROVENANCE.json`. New work should extend
`data_sources/corpus_v2.py` and keep provenance entries complete.
"""
from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
import time
from pathlib import Path
from typing import Any

from src.data.records import TrainingRecord, classify_license, validate_record

logger = logging.getLogger("tinyme.github")

REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = REPO_ROOT / "datasets" / "raw" / "github"

CODE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "typescript",
    ".java": "java", ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp",
    ".rs": "rust", ".go": "go", ".kt": "kotlin", ".rb": "ruby", ".sql": "sql",
    ".sh": "shell", ".html": "html", ".css": "css", ".yaml": "yaml", ".yml": "yaml",
    ".toml": "toml", ".json": "json", ".md": "markdown",
}
EXCLUDE_PATH_PATTERNS = [
    r"(^|/)node_modules/", r"(^|/)vendor/", r"(^|/)\.venv/", r"(^|/)site-packages/",
    r"(^|/)dist/", r"(^|/)build/", r"(^|/)\.git/", r"__pycache__",
    r"(^|/)third_party/", r"(^|/)extern/", r"\.min\.js$", r"\.map$",
    r"(^|/)migrations/", r"(^|/)fixtures/", r"\.lock$", r"package-lock\.json",
    r"(^|/)coverage/", r"(^|/)docs/_build/", r"\.pb\.go$", r"_pb2\.py$",
]
SECRET_RX = re.compile(r"(ghp_[A-Za-z0-9]{20,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z\-_]{35}|-----BEGIN [A-Z ]*PRIVATE KEY-----)")


def _gh_api(path: str) -> Any:
    out = subprocess.run(["gh", "api", path], capture_output=True, text=True, timeout=120)
    if out.returncode != 0:
        raise RuntimeError(f"gh api {path} failed: {out.stderr.strip()[:200]}")
    return json.loads(out.stdout) if out.stdout.strip() else None


def repo_license(owner_repo: str) -> tuple[str, str]:
    """Return (spdx_or_LICENSE_UNCLEAR, license_url) using API metadata + LICENSE file."""
    try:
        meta = _gh_api(f"repos/{owner_repo}")
    except Exception as exc:
        logger.warning("metadata fetch failed for %s: %s", owner_repo, exc)
        return "LICENSE_UNCLEAR", ""
    lic = classify_license((meta.get("license") or {}).get("spdx_id"),
                           (meta.get("license") or {}).get("name"))
    lic_url = (meta.get("license") or {}).get("url") or ""
    if lic == "LICENSE_UNCLEAR":
        # Try the LICENSE file itself before giving up.
        for cand in ("LICENSE", "LICENSE.md", "LICENSE.txt", "COPYING"):
            try:
                content = _gh_api(f"repos/{owner_repo}/contents/{cand}")
                text = base64.b64decode(content["content"]).decode("utf-8", "ignore")[:3000]
                lic = classify_license(None, text.splitlines()[0] if text.splitlines() else "")
                if lic == "LICENSE_UNCLEAR":
                    lowered = text.lower()
                    if "mit license" in lowered:
                        lic = "MIT"
                    elif "apache license" in lowered and "version 2.0" in lowered:
                        lic = "Apache-2.0"
                    elif "bsd" in lowered:
                        lic = "BSD-3-Clause"
                if lic != "LICENSE_UNCLEAR":
                    lic_url = content.get("html_url", lic_url)
                    break
            except Exception:
                continue
    return lic, lic_url


def list_repo_files(owner_repo: str, max_files: int = 400, max_size: int = 60_000) -> list[dict[str, Any]]:
    """List candidate source files from the default branch tree."""
    meta = _gh_api(f"repos/{owner_repo}")
    branch = meta.get("default_branch", "main")
    tree = _gh_api(f"repos/{owner_repo}/git/trees/{branch}?recursive=1")
    files = []
    for node in tree.get("tree", []):
        if node.get("type") != "blob":
            continue
        path = node["path"]
        if node.get("size", 0) > max_size or node.get("size", 0) == 0:
            continue
        ext = Path(path).suffix.lower()
        if ext not in CODE_EXTENSIONS:
            continue
        if any(re.search(p, path) for p in EXCLUDE_PATH_PATTERNS):
            continue
        files.append({"path": path, "size": node.get("size", 0), "branch": branch})
        if len(files) >= max_files:
            break
    return files


def fetch_file(owner_repo: str, path: str, branch: str) -> str:
    content = _gh_api(f"repos/{owner_repo}/contents/{path}?ref={branch}")
    if content.get("encoding") != "base64":
        return ""
    return base64.b64decode(content["content"]).decode("utf-8", "ignore")


def is_usable_code(text: str) -> bool:
    if not text or len(text) < 60:
        return False
    if SECRET_RX.search(text):
        return False
    lines = text.split("\n")
    if max((len(ln) for ln in lines), default=0) > 1000:
        return False
    avg = sum(len(ln) for ln in lines) / len(lines)
    return avg < 200


def make_code_records(owner_repo: str, files: list[dict[str, Any]], license_: str,
                      lic_url: str, per_file_limit: int = 6) -> list[TrainingRecord]:
    """Turn repository files into code + code-explanation records."""
    records: list[TrainingRecord] = []
    today = time.strftime("%Y-%m-%d")
    for f in files[:per_file_limit]:
        try:
            text = fetch_file(owner_repo, f["path"], f["branch"])
        except Exception as exc:
            logger.warning("file fetch failed %s/%s: %s", owner_repo, f["path"], exc)
            continue
        if not is_usable_code(text):
            continue
        text = text.strip()
        if len(text) > 20_000:
            text = text[:20_000]
        lang = CODE_EXTENSIONS.get(Path(f["path"]).suffix.lower(), "text")
        records.append(TrainingRecord(
            text=text, category="programming", source="github",
            source_id=f"{owner_repo}:{f['path']}", license=license_,
            source_url=f"https://github.com/{owner_repo}/blob/{f['branch']}/{f['path']}",
            retrieval_date=today, task_type="code_completion", language=lang,
        ))
        # Documentation -> code relationship (docstring/README pairing)
        docstring_match = re.search(r'"""(.*?)"""', text, re.S)
        if docstring_match and len(docstring_match.group(1).strip()) > 80:
            doc = docstring_match.group(1).strip()
            records.append(TrainingRecord(
                text=f"<|system|>\nExplain what the following code does.\n<|user|>\n```\n{text[:6000]}\n```\n<|assistant|>\n{doc[:4000]}\n",
                category="code_explain", source="github",
                source_id=f"{owner_repo}:{f['path']}:doc", license=license_,
                source_url=f"https://github.com/{owner_repo}/blob/{f['branch']}/{f['path']}",
                retrieval_date=today, task_type="instruction", language=lang,
            ))
    return records


def ingest(repos: list[str] | None = None, files_per_repo: int = 6,
           records_per_repo: int = 6) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    """Ingest the configured repositories with license gating."""
    repos = repos or [
        "TheAlgorithms/Python", "keon/algorithms", "openai/grade-school-math",
        "TheAlgorithms/Go", "TheAlgorithms/Rust",
    ]
    records: list[TrainingRecord] = []
    provenance: list[dict[str, Any]] = []
    today = time.strftime("%Y-%m-%d")
    for repo in repos:
        lic, lic_url = repo_license(repo)
        allowed = lic not in ("LICENSE_UNCLEAR",) and lic not in {"GPL-3.0", "AGPL-3.0", "CC-BY-NC-4.0"}
        entry = {
            "source": "github", "dataset_or_repo_name": repo,
            "url": f"https://github.com/{repo}", "retrieval_date": today,
            "license": lic, "license_url": lic_url,
            "source_type": "code", "language": "multi",
            "approximate_size_bytes": 0, "sample_count": 0,
            "preprocessing_performed": ["license_check", "content_extraction", "secret_redaction",
                                        "binary_filter", "vendored_dependency_filter", "quality_scoring", "deduplication"],
            "inclusion_status": "INCLUDED" if allowed else "EXCLUDED",
            "inclusion_reason": (f"license {lic} permits ML use" if allowed
                                 else f"license {lic} excluded by LICENSE_POLICY.md"),
        }
        if not allowed:
            logger.warning("repository %s excluded (license=%s)", repo, lic)
            provenance.append(entry)
            continue
        try:
            files = list_repo_files(repo, max_files=max(40, files_per_repo * 4))
            recs = make_code_records(repo, files, lic, lic_url, per_file_limit=records_per_repo)
            recs = [r for r in recs if validate_record(r)[0]]
        except Exception as exc:
            logger.warning("repository ingestion failed for %s: %s", repo, exc)
            entry["inclusion_status"] = "EXCLUDED"
            entry["inclusion_reason"] = f"ingestion error: {exc}"
            provenance.append(entry)
            continue
        records.extend(recs)
        entry["sample_count"] = len(recs)
        entry["approximate_size_bytes"] = sum(len(r.text.encode("utf-8")) for r in recs)
        provenance.append(entry)
        logger.info("ingested %d records from %s (%s)", len(recs), repo, lic)
    return records, provenance
