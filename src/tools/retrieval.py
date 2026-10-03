"""Offline lexical retrieval index (audit §30).

A deterministic BM25 ranking over a local corpus — no network, no external
embedding service, so retrieval works in this restricted environment and its
results are reproducible.  Every document carries a provenance record
(``source``/``license``/``url``) so downstream citations can be traced and never
invented.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..utils.io_utils import REPO_ROOT

INDEX_VERSION = "retrieval-v1"
TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset("a an and are as at be by for from has have in is it its of on or that the to was were with".split())


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


@dataclass
class Document:
    source_id: str
    title: str
    text: str
    url: str = ""
    source: str = "unknown"
    license: str = "unknown"
    published_at: str = ""

    def to_dict(self) -> dict:
        return {"source_id": self.source_id, "title": self.title, "text": self.text, "url": self.url,
                "source": self.source, "license": self.license, "published_at": self.published_at}


class RetrievalIndex:
    """BM25 index; small enough to search by brute force, fully deterministic."""

    K1 = 1.5
    B = 0.75

    def __init__(self, documents: list[Document]):
        self.documents = documents
        self.doc_tokens: list[Counter] = [Counter(tokenize(d.title + " \n " + d.text)) for d in documents]
        self.doc_len = [sum(c.values()) or 1 for c in self.doc_tokens]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)
        self.df: Counter = Counter()
        for counts in self.doc_tokens:
            self.df.update(counts.keys())

    # ------------------------------------------------------------------ search
    def search(self, query: str, k: int = 3) -> list[dict]:
        q_tokens = tokenize(query)
        if not q_tokens or not self.documents:
            return []
        n = len(self.documents)
        scores: list[tuple[float, int]] = []
        for idx, counts in enumerate(self.doc_tokens):
            length = self.doc_len[idx]
            score = 0.0
            # sorted(): a set's iteration order depends on hash randomisation, and the
            # float accumulation below is order-sensitive — the same seed produced a
            # different corpus on every process until this was deterministic.
            for term in sorted(set(q_tokens)):
                tf = counts.get(term, 0)
                if not tf:
                    continue
                idf = math.log(1 + (n - self.df[term] + 0.5) / (self.df[term] + 0.5))
                score += idf * tf * (self.K1 + 1) / (tf + self.K1 * (1 - self.B + self.B * length / self.avgdl))
            if score > 0:
                scores.append((score, idx))
        scores.sort(key=lambda p: (-p[0], self.documents[p[1]].source_id))
        out = []
        for score, idx in scores[: max(1, min(int(k), 8))]:
            doc = self.documents[idx]
            snippet = _best_snippet(doc.text, q_tokens)
            out.append({"source_id": doc.source_id, "title": doc.title, "snippet": snippet,
                        "url": doc.url, "source": doc.source, "license": doc.license,
                        "score": round(score, 4)})
        return out

    def get(self, source_id: str) -> Document | None:
        for doc in self.documents:
            if doc.source_id == source_id:
                return doc
        return None

    # ------------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return {"version": INDEX_VERSION, "documents": [d.to_dict() for d in self.documents]}

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RetrievalIndex":
        data = json.loads(Path(path).read_text())
        return cls([Document(**d) for d in data["documents"]])

    def __len__(self) -> int:
        return len(self.documents)


def _best_snippet(text: str, q_tokens: list[str], width: int = 240) -> str:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    for sent in sentences:
        low = sent.lower()
        if any(tok in low for tok in q_tokens):
            return sent[:width]
    return text.strip()[:width]


def default_index_path() -> Path:
    return REPO_ROOT / "datasets" / "retrieval" / "index.json"


_cached: RetrievalIndex | None = None


def load_default_index(path: str | Path | None = None) -> RetrievalIndex | None:
    """Load the shipped index, or return ``None`` when it has not been built."""
    global _cached
    path = Path(path) if path else default_index_path()
    if not path.exists():
        return None
    if _cached is None or Path(path) != default_index_path():
        idx = RetrievalIndex.load(path)
        if Path(path) == default_index_path():
            _cached = idx
        return idx
    return _cached


def build_documents(corpus_path: str | Path) -> list[Document]:
    docs = []
    for line in Path(corpus_path).read_text().splitlines():
        if line.strip():
            docs.append(Document(**json.loads(line)))
    return docs


def write_corpus(documents: Iterable[Document], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for doc in documents:
            fh.write(json.dumps(doc.to_dict(), ensure_ascii=False) + "\n")
    return path
