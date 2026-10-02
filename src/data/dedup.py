"""Multi-level deduplication with a real MinHash/LSH index (corrective audit P1-05).

Levels
  1. exact hash            (sha256 of raw text)
  2. normalized hash       (case/whitespace/punctuation-insensitive sha256)
  3. MinHash/LSH similarity (banded index, `num_perm` permutations, Jaccard check
     on the candidates the index returns; no linear scan over the corpus)
  4. code similarity       (AST-token shingles for Python, MinHash-LSH indexed)

The previous implementation compared every new document against the last
``max_compare`` kept documents and contained a placeholder ``for key in
bucket_keys: pass`` loop; both are gone.
"""
from __future__ import annotations

import ast
import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

_MASK61 = (1 << 61) - 1
_MERSENNE31 = (1 << 31) - 1          # a*x + b fits comfortably in int64
_WORD_RE = re.compile(r"[a-z0-9_]+")
_WS_RE = re.compile(r"\s+")


# ------------------------------------------------------------------ hashes
def exact_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    t = _WS_RE.sub(" ", text.strip().lower())
    t = re.sub(r"[^\w\s]", "", t)
    return t.strip()


def normalized_hash(text: str) -> str:
    return hashlib.sha256(normalize_text(text).encode("utf-8")).hexdigest()


def shingles(text: str, k: int = 5) -> set[str]:
    toks = _WORD_RE.findall(text.lower())
    if len(toks) < k:
        return {" ".join(toks)} if toks else set()
    return {" ".join(toks[i:i + k]) for i in range(len(toks) - k + 1)}


def code_tokens(code: str) -> set[str]:
    """Identifier-aware shingles of a Python snippet."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return shingles(code, 3)
    toks: list[str] = []
    for n in ast.walk(tree):
        toks.append(type(n).__name__)
        name = getattr(n, "name", None) or getattr(n, "id", None) or getattr(n, "attr", None)
        if isinstance(name, str):
            toks.append(f"id:{name}")
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, str, bool)):
            toks.append(f"const:{n.value!r}")
    if len(toks) < 4:
        return set(toks)
    return {" ".join(toks[i:i + 4]) for i in range(len(toks) - 3)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return inter / len(a | b)


# ------------------------------------------------------------------ MinHash
def _hash64(value: str, seed: bytes = b"") -> int:
    h = (hashlib.blake2b(value.encode("utf-8"), digest_size=8, key=seed)
         if seed else hashlib.blake2b(value.encode("utf-8"), digest_size=8))
    return int.from_bytes(h.digest(), "little")


def _hash31(value: str) -> int:
    """31-bit shingle hash (fits the Mersenne-31 MinHash arithmetic)."""
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=4).digest()
    v = int.from_bytes(digest, "little") % _MERSENNE31
    return v or 1


class MinHashLSH:
    """Banded MinHash index.

    ``num_perm`` permutations are grouped into ``bands`` bands of
    ``rows = num_perm // bands`` rows. Two documents become candidates when any
    band signature collides; Jaccard is then verified exactly.
    """

    def __init__(self, num_perm: int = 64, bands: int = 16, threshold: float = 0.85,
                 seed: int = 1234):
        if num_perm % bands:
            raise ValueError("num_perm must be divisible by bands")
        self.num_perm = num_perm
        self.bands = bands
        self.rows = num_perm // bands
        self.threshold = threshold
        rng = np.random.default_rng(seed)
        self._p = _MERSENNE31
        self._a = rng.integers(1, self._p, size=num_perm, dtype=np.int64)
        self._b = rng.integers(0, self._p, size=num_perm, dtype=np.int64)
        self._buckets: dict[tuple[int, int], list[int]] = {}
        self._docs: list[tuple[str, frozenset[str]]] = []

    # -- signatures --------------------------------------------------------
    def signature(self, values: Iterable[str]) -> np.ndarray:
        """Vectorised MinHash signature over the shingle set."""
        vals = list(values)
        if not vals:
            return np.full(self.num_perm, self._p, dtype=np.int64)
        hashed = np.fromiter((_hash31(v) for v in vals), dtype=np.int64, count=len(vals))
        # (num_perm, n): a*x + b mod p, taken row-wise min — no Python loop over shingles
        products = (self._a[:, None] * hashed[None, :] + self._b[:, None]) % self._p
        return products.min(axis=1)

    # -- index -------------------------------------------------------------
    def add(self, key: str, values: Iterable[str]) -> list[int]:
        """Index a value set; return candidate ids sharing at least one band."""
        vals = frozenset(values)
        sig = self.signature(vals)
        doc_id = len(self._docs)
        candidates: set[int] = set()
        for band in range(self.bands):
            band_sig = sig[band * self.rows:(band + 1) * self.rows]
            band_key = (band, _hash64(band_sig.tobytes().hex()))
            candidates.update(self._buckets.get(band_key, ()))
            self._buckets.setdefault(band_key, []).append(doc_id)
        self._docs.append((key, vals))
        return sorted(candidates)

    def candidates_for(self, values: Iterable[str]) -> list[int]:
        vals = frozenset(values)
        sig = self.signature(vals)
        out: set[int] = set()
        for band in range(self.bands):
            band_key = (band, _hash64(sig[band * self.rows:(band + 1) * self.rows].tobytes().hex()))
            out.update(self._buckets.get(band_key, ()))
        return sorted(out)

    def is_duplicate(self, values: Iterable[str], threshold: float | None = None) -> int | None:
        """Return the id of a near-duplicate document, or ``None``."""
        thr = self.threshold if threshold is None else threshold
        vals = frozenset(values)
        for cid in self.candidates_for(vals):
            if jaccard(vals, self._docs[cid][1]) >= thr:
                return cid
        return None

    @property
    def size(self) -> int:
        return len(self._docs)


@dataclass
class DedupReport:
    total: int = 0
    exact_duplicates: int = 0
    normalized_duplicates: int = 0
    similar_duplicates: int = 0
    code_similar_duplicates: int = 0
    kept: int = 0
    lsh_index_size: int = 0
    duplicate_examples: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = dict(self.__dict__)
        d["duplicate_percentage"] = round(
            100.0 * (self.exact_duplicates + self.normalized_duplicates
                     + self.similar_duplicates + self.code_similar_duplicates)
            / max(self.total, 1), 4)
        return d


def deduplicate(records: Iterable[dict[str, Any]], similarity_threshold: float = 0.85,
                code_threshold: float = 0.90, num_perm: int = 64, bands: int = 16,
                seed: int = 1234,
                code_categories: Iterable[str] = ("programming", "code_gen", "code_repair",
                                                  "code_explain", "algorithm", "tool_use")
                ) -> tuple[list[dict[str, Any]], DedupReport]:
    """Streaming multi-level dedup with a banded MinHash index."""
    report = DedupReport()
    code_cats = set(code_categories)
    seen_exact: set[str] = set()
    seen_norm: set[str] = set()
    text_index = MinHashLSH(num_perm=num_perm, bands=bands, threshold=similarity_threshold, seed=seed)
    code_index = MinHashLSH(num_perm=num_perm, bands=bands, threshold=code_threshold, seed=seed + 1)
    kept: list[dict[str, Any]] = []

    for rec in records:
        report.total += 1
        text = rec.get("text", "") or ""
        eh = exact_hash(text)
        if eh in seen_exact:
            report.exact_duplicates += 1
            continue
        nh = normalized_hash(text)
        if nh in seen_norm:
            report.normalized_duplicates += 1
            continue

        sh = shingles(text)
        if sh and text_index.is_duplicate(sh) is not None:
            report.similar_duplicates += 1
            if len(report.duplicate_examples) < 10:
                report.duplicate_examples.append(rec.get("record_id", ""))
            continue

        if rec.get("category") in code_cats:
            ct = code_tokens(text)
            if ct and code_index.is_duplicate(ct) is not None:
                report.code_similar_duplicates += 1
                continue
            if ct:
                code_index.add(rec.get("record_id", "") or nh, ct)

        if sh:
            text_index.add(rec.get("record_id", "") or eh, sh)
        seen_exact.add(eh)
        seen_norm.add(nh)
        kept.append(rec)

    report.kept = len(kept)
    report.lsh_index_size = text_index.size
    return kept, report


# ------------------------------------------------------- contamination check
def contamination_report(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]],
                         similarity_threshold: float = 0.85) -> dict[str, Any]:
    """Detect overlap between train and eval at every dedup level."""
    train_exact = {exact_hash(r["text"]) for r in train_records}
    train_norm = {normalized_hash(r["text"]) for r in train_records}
    train_code: set[str] = set()
    code_index = MinHashLSH(num_perm=64, bands=16, threshold=0.90, seed=99)
    text_index = MinHashLSH(num_perm=64, bands=16, threshold=similarity_threshold, seed=98)
    for r in train_records:
        train_code |= code_tokens(r["text"])
        sh = shingles(r["text"])
        if sh:
            text_index.add(r.get("record_id", ""), sh)
    for r in train_records:
        ct = code_tokens(r["text"])
        if ct:
            code_index.add(r.get("record_id", ""), ct)

    train_groups = {r.get("group_id") or r.get("template_id") for r in train_records}
    train_groups.discard(None)
    train_groups.discard("")

    exact_hits, norm_hits, sim_hits, code_hits, group_hits = [], [], [], [], []
    for r in eval_records:
        rid = r.get("record_id", "")
        if exact_hash(r["text"]) in train_exact:
            exact_hits.append(rid)
            continue
        if normalized_hash(r["text"]) in train_norm:
            norm_hits.append(rid)
            continue
        sh = shingles(r["text"])
        if sh and text_index.is_duplicate(sh) is not None:
            sim_hits.append(rid)
            continue
        et = code_tokens(r["text"])
        if et and code_index.is_duplicate(et) is not None:
            code_hits.append(rid)
            continue
        gid = r.get("group_id") or r.get("template_id")
        if gid and gid in train_groups:
            group_hits.append(rid)

    contaminated = exact_hits + norm_hits + sim_hits + code_hits + group_hits
    return {
        "eval_samples": len(eval_records),
        "train_samples": len(train_records),
        "exact_overlap": len(exact_hits),
        "normalized_overlap": len(norm_hits),
        "minhash_overlap": len(sim_hits),
        "code_shingle_overlap": len(code_hits),
        "template_overlap": len(group_hits),
        "contaminated_record_ids": contaminated[:50],
        "contamination_free": not contaminated,
    }
