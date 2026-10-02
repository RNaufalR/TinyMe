"""Deduplication and contamination: real MinHash/LSH, no placeholders (P1-05)."""
from __future__ import annotations

from src.data.dedup import (MinHashLSH, code_tokens, contamination_report,
                            deduplicate, exact_hash, jaccard, normalized_hash,
                            shingles)


def _rec(text, rid, category="language", group="g1"):
    return {"text": text, "record_id": rid, "category": category, "group_id": group,
            "template_id": group}


def test_exact_and_normalized_duplicates():
    base = "The quick brown fox jumps over the lazy dog. " * 4
    recs = [_rec(base, "a"), _rec(base, "b"),
            _rec("  THE   QUICK brown fox   jumps over the lazy dog. " * 4, "c")]
    kept, report = deduplicate(recs)
    assert report.exact_duplicates == 1
    assert report.normalized_duplicates == 1
    assert len(kept) == 1


def test_minhash_lsh_detects_near_duplicates():
    para = " ".join(f"word{i}" for i in range(120))
    near = " ".join(f"word{i}" for i in range(118)) + " extra tokens appended here"
    recs = [_rec(para, "a"), _rec(near, "b"), _rec("totally unrelated " * 30, "c")]
    kept, report = deduplicate(recs, similarity_threshold=0.8)
    assert report.similar_duplicates == 1
    assert {r["record_id"] for r in kept} == {"a", "c"}
    assert report.lsh_index_size == 2


def test_minhash_index_is_banded_not_linear():
    idx = MinHashLSH(num_perm=32, bands=8, threshold=0.9, seed=0)
    idx.add("a", shingles(" ".join(f"t{i}" for i in range(50))))
    assert idx.size == 1
    # identical content -> candidate found
    assert idx.is_duplicate(shingles(" ".join(f"t{i}" for i in range(50)))) == 0
    # unrelated content -> no candidate
    assert idx.is_duplicate(shingles(" ".join(f"z{i}" for i in range(50)))) is None


def test_code_similarity_uses_ast():
    code = "def add(a, b):\n    return a + b\n" * 3
    recs = [_rec(code, "a", "code_gen"), _rec(code, "b", "code_gen")]
    _, report = deduplicate(recs, code_threshold=0.9)
    assert report.exact_duplicates + report.code_similar_duplicates == 1
    assert len(code_tokens(code)) > 0


def test_contamination_report_finds_template_overlap():
    train = [_rec("q one a one", "t1", group="family_a")]
    test = [_rec("q two a two", "e1", group="family_a"),
            _rec("completely different topic entirely", "e2", group="family_b")]
    report = contamination_report(train, test)
    assert report["template_overlap"] == 1
    assert report["contaminated_record_ids"] == ["e1"]
    assert report["contamination_free"] is False


def test_jaccard_and_hashes():
    assert jaccard({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard({"a"}, {"b"}) == 0.0
    assert exact_hash("x") != exact_hash("y")
    assert normalized_hash("Hello   World") == normalized_hash("hello world")
