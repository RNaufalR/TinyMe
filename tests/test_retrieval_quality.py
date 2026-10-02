"""Retrieval quality and provenance (audit §7).

Not merely "BM25 returns something": exact/paraphrased/multi-term/irrelevant
queries, tie-breaking, ``k`` boundaries, metadata, provenance and deterministic
ranking are all measured against the shipped index.
"""
from __future__ import annotations

import pytest

from src.tools.retrieval import RetrievalIndex, load_default_index, tokenize

index = load_default_index()
pytestmark = pytest.mark.skipif(index is None or len(index) == 0,
                                reason="no retrieval index built (scripts/build_retrieval_index.py)")


def _ids(results):
    return [r["source_id"] for r in results]


def test_exact_query_hits_the_expected_document():
    results = index.search("population of jakarta", k=3)
    assert results, "exact query returned nothing"
    assert "FACT-988002" in _ids(results)
    top = results[0]
    assert top["score"] > 0
    assert "jakarta" in (top["title"] + top["snippet"]).lower()


def test_paraphrased_query_still_retrieves_the_same_fact():
    direct = _ids(index.search("capital of indonesia", k=1))
    paraphrased = _ids(index.search("which city is the capital of indonesia", k=3))
    assert direct[0] in paraphrased, (direct, paraphrased)


def test_multi_term_query_ranks_the_matching_document_first():
    results = index.search("psf licence python standard library", k=3)
    assert results and results[0]["source_id"] == "FACT-16880E", _ids(results)


def test_irrelevant_query_returns_nothing_rather_than_nonsense():
    results = index.search("zzzzq xyzzy plugh foobar", k=3)
    assert results == [], _ids(results)


def test_repeated_identical_queries_are_deterministic():
    a = index.search("binary search complexity", k=5)
    b = index.search("binary search complexity", k=5)
    assert [r["source_id"] for r in a] == [r["source_id"] for r in b]
    assert [round(r["score"], 6) for r in a] == [round(r["score"], 6) for r in b]


def test_k_boundaries_are_respected():
    assert len(index.search("population", k=1)) <= 1
    assert len(index.search("population", k=3)) <= 3
    many = index.search("python stdlib module function", k=5)
    assert 0 < len(many) <= 5


def test_results_carry_source_metadata_and_provenance():
    for r in index.search("binary search complexity", k=3):
        assert r["source_id"]
        assert r["url"].startswith("http")
        assert r["title"]
        assert r["snippet"]
        assert r["license"]
        assert r["source"]


def test_evidence_presence_for_citation_ready_queries():
    """Every hit must be citable: id + url + a non-empty quoted span."""
    hits = index.search("definition of a prime number", k=3)
    assert hits
    for r in hits:
        assert r["source_id"] and r["url"]
        assert len(r["snippet"]) >= 20


def test_tie_breaking_is_stable_by_source_id():
    results = index.search("python module", k=8)
    scored = [(r["score"], r["source_id"]) for r in results]
    ranked = sorted(scored, key=lambda x: (-x[0], x[1]))
    assert scored == ranked, "ties must break deterministically by source id"


def test_tokenizer_drops_stopwords_and_short_tokens():
    assert tokenize("the a of an and x") == []
    assert "population" in tokenize("The Population of Jakarta!")


def test_index_round_trips_through_disk(tmp_path):
    path = tmp_path / "index.json"
    index.save(path)
    loaded = RetrievalIndex.load(path)
    assert len(loaded) == len(index)
    assert _ids(loaded.search("capital of indonesia", k=1)) == _ids(index.search("capital of indonesia", k=1))


def test_hit_rate_over_a_fixed_query_set():
    """Aggregate measurement, not a single example: report the hit rate."""
    queries = {
        "population of jakarta": "FACT-988002",
        "population of bandung": "FACT-F56310",
        "capital of indonesia": "FACT-8334A1",
        "definition of a prime number": "FACT-FCD206",
        "big-o of binary search": "FACT-2082DD",
        "water freezing point at one atmosphere": "FACT-E850FA",
        "psf licence of the python standard library": "FACT-16880E",
        "shortest path algorithm for non-negative weights": "FACT-0B1DC0",
    }
    hits = 0
    for query, expected in queries.items():
        if expected in _ids(index.search(query, k=1)):
            hits += 1
    hit_rate = hits / len(queries)
    assert hit_rate >= 0.875, f"hit rate {hit_rate:.3f} ({hits}/{len(queries)})"
