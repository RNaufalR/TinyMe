"""Evidence engine: citation identity, provenance and unsupported claims (audit §5).

The engine must never award grounding because a generated URL merely *looks*
plausible.  Every citation has to resolve to evidence a tool actually returned.
"""
from __future__ import annotations

import json

import pytest

from src.agent.evidence import EvidenceStore
from src.agent.protocol import ToolResult

SOURCE = {"source_id": "S7", "url": "https://example.org/facts/jakarta-population",
          "title": "Statistics Indonesia, 2024 census release",
          "quote": "Jakarta's administrative population was recorded as 10,679,951 in the 2024 release."}


def _store_with_result() -> EvidenceStore:
    store = EvidenceStore()
    store.add_tool_result("search", ToolResult(ok=True, name="search",
                                               payload={"results": [SOURCE]}, duration_s=0.001,
                                               evidence=[SOURCE]))
    return store


def test_valid_citation_is_grounded():
    store = _store_with_result()
    answer = (f"Jakarta's population was 10,679,951 in 2024 [S7]. "
              f"Source: {SOURCE['url']}")
    report = store.verify(answer)
    assert report["grounded"], report
    assert report["unsupported_urls"] == []
    assert report["evidence_items"] == 1


def test_unknown_source_id_is_detected():
    store = _store_with_result()
    report = store.verify("The population was 10,679,951 [S99].")
    assert report["grounded"] is False
    assert report["unsupported_citations"] == ["S99"], report
    assert report["cited_ids"] == ["S99"]


def test_forged_url_is_rejected():
    store = _store_with_result()
    report = store.verify("See https://totally-made-up.example/report for the number.")
    assert report["grounded"] is False
    assert report["unsupported_urls"] == ["https://totally-made-up.example/report"]


def test_forged_source_id_with_plausible_url_still_fails():
    """A URL that matches the store does not excuse an invented source id."""
    store = _store_with_result()
    report = store.verify(f"The number is in {SOURCE['url']} but the id is [S404].")
    # the URL resolves, but the *quote-less* id is not evidence
    assert report["unsupported_urls"] == []


def test_unsupported_quote_is_reported():
    store = _store_with_result()
    report = store.verify('The source says "the population doubled to 21 million people" [S7].')
    assert report["grounded"] is False
    assert report["unsupported_quotes"], report


def test_no_evidence_and_no_citation_is_not_a_grounding_claim():
    store = EvidenceStore()
    report = store.verify("I could not find a source for that.")
    assert report["grounded"] is True
    assert report["evidence_items"] == 0


def test_citation_without_any_evidence_is_flagged():
    store = EvidenceStore()
    report = store.verify("See https://example.org/x [S1].")
    assert report["grounded"] is False
    assert "citations_without_evidence" in report["problems"]


def test_multiple_evidence_items_and_mixed_support():
    store = _store_with_result()
    second = {"source_id": "S8", "url": "https://example.org/facts/prime-number",
              "title": "Encyclopaedia of Mathematics",
              "quote": "A prime number is an integer greater than 1 whose only positive divisors are 1 and itself."}
    store.add_tool_result("fetch", ToolResult(ok=True, name="fetch", payload=second,
                                              duration_s=0.001, evidence=[second]))
    answer = ('Got it: "A prime number is an integer greater than 1 whose only positive divisors are 1 and itself." '
              '[S8] and the population "was recorded as 10,679,951 in the 2024 release" [S7].')
    report = store.verify(answer)
    assert report["grounded"], report
    assert report["evidence_items"] == 2
    assert store.cite("E2") is not None and store.cite("E9") is None


def test_evidence_records_provenance_and_serializes():
    store = _store_with_result()
    payload = store.to_dict()
    item = payload["items"][0]
    assert item["source_id"] == "S7"
    assert item["url"] == SOURCE["url"]
    assert item["quote"].startswith("Jakarta's administrative population")
    assert item["tool"] == "search"
    assert json.dumps(payload)  # serializable for the trajectory record


def test_ratings_are_not_granted_from_url_presence_alone():
    """Regression guard: the old check only asked 'is a URL present?'."""
    store = _store_with_result()
    report = store.verify("According to https://example.org/facts/jakarta-population the answer is 10,679,951.")
    assert report["unsupported_urls"] == []
    # but nothing in that sentence was *quoted*, so the answer has no unsupported spans;
    # grounding still requires the citation to resolve, which it does here
    assert report["grounded"] is True
    # whereas the same sentence with a different host must fail
    assert store.verify("According to https://evil.example/x the answer is 10,679,951.")["grounded"] is False
