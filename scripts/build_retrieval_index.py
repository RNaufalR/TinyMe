#!/usr/bin/env python3
"""Build the offline retrieval corpus + BM25 index (audit §30).

Corpus composition (all provenance-carrying, all offline):

* the curated fact table used to generate tool-use training data
  (``data_sources.synthetic_v2.EVIDENCE_DB``);
* CPython 3.11 standard-library module summaries extracted with the same
  safety filters as the training corpus (license ``PSF-2.0``).

Usage::

    python scripts/build_retrieval_index.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_sources.stdlib_v2 import stdlib_root  # noqa: E402
from data_sources.synthetic_v2 import EVIDENCE_DB  # noqa: E402
from src.tools.retrieval import Document, RetrievalIndex, write_corpus  # noqa: E402
from src.utils.io_utils import REPO_ROOT, write_json  # noqa: E402


def curated_documents() -> list[Document]:
    docs = []
    for key, entry in sorted(EVIDENCE_DB.items()):
        docs.append(Document(
            source_id="FACT-" + hashlib.sha256(key.encode()).hexdigest()[:6].upper(),
            title=entry["title"], text=f"{key.capitalize()}.\n{entry['excerpt']}",
            url=entry["url"], source="curated-facts", license="CC0-1.0",
            published_at=entry.get("published_at", "")))
    return docs


def stdlib_documents(limit: int = 400) -> list[Document]:
    root = stdlib_root()
    if root is None:
        return []
    docs: list[Document] = []
    for path in sorted(root.rglob("*.py")):
        if any(part in {"test", "tests", "idlelib", "lib2to3", "site-packages", "__pycache__"} for part in path.parts):
            continue
        rel = path.relative_to(root).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
            tree = ast.parse(source)
        except (SyntaxError, OSError):
            continue
        docstring = ast.get_docstring(tree) or ""
        public = [n.name for n in tree.body
                  if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and not n.name.startswith("_")]
        if not docstring and not public:
            continue
        summary = docstring.strip().split("\n\n")[0][:600]
        text = (f"Module {rel} (CPython 3.11 standard library).\n{summary}\n"
                f"Public API: {', '.join(public[:24])}.")
        docs.append(Document(
            source_id=f"STDLIB-{rel.replace('/', '.').removesuffix('.py')}",
            title=f"Python stdlib: {rel}", text=text,
            url=f"https://docs.python.org/3.11/library/{rel.replace('/', '.').removesuffix('.py')}.html",
            source="CPython 3.11 stdlib", license="PSF-2.0"))
        if len(docs) >= limit:
            break
    return docs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "datasets" / "retrieval"))
    args = ap.parse_args()

    docs = curated_documents() + stdlib_documents(args.limit)
    out_dir = Path(args.out_dir)
    corpus_path = write_corpus(docs, out_dir / "corpus.jsonl")
    index = RetrievalIndex(docs)
    index_path = index.save(out_dir / "index.json")
    manifest = {
        "version": index.to_dict()["version"],
        "documents": len(docs),
        "sources": sorted({d.source for d in docs}),
        "licenses": sorted({d.license for d in docs}),
        "corpus_bytes": corpus_path.stat().st_size,
        "index_bytes": index_path.stat().st_size,
        "corpus_path": str(corpus_path.relative_to(REPO_ROOT)),
        "index_path": str(index_path.relative_to(REPO_ROOT)),
    }
    write_json(manifest, out_dir / "manifest.json")
    probes = {q: [r["source_id"] for r in index.search(q, k=2)]
              for q in ("binary search complexity", "jakarta population", "what is a prime number")}
    print(json.dumps({**manifest, "probe_queries": probes}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
