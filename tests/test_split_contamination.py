"""Group-aware splitting and contamination gates (P0-05)."""
from __future__ import annotations

import numpy as np

from src.data.splits import SPLITS, assign_splits, dataset_fingerprint, split_report


def _records(n_families=12, per_family=5):
    out = []
    for f in range(n_families):
        tokens = " ".join(f"topic{f}x{j}" for j in range(24))
        for i in range(per_family):
            out.append({"text": f"family {f} sample {i}: {tokens}",
                        "record_id": f"rec-{f}-{i}", "category": "language",
                        "group_id": f"family-{f}", "template_id": f"family-{f}",
                        "split": ""})
    return out


def test_groups_never_split_across_splits():
    recs, report = assign_splits(_records(), seed=1234)
    by_group: dict[str, set[str]] = {}
    for r in recs:
        by_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert report["counts"]["train"] > 0
    assert report["counts"]["validation"] > 0
    assert report["counts"]["test"] > 0


def test_deterministic_given_seed():
    a, _ = assign_splits(_records(), seed=7)
    b, _ = assign_splits(_records(), seed=7)
    assert [r["split"] for r in a] == [r["split"] for r in b]
    # the seed drives the family ordering: different seeds give different orders
    from src.data.splits import _stable_order

    own = [f"family-{i}" for i in range(60)]
    assert _stable_order(own, 7) != _stable_order(own, 8)


def test_no_contamination_between_splits():
    recs, report = assign_splits(_records(), seed=99)
    assert report["contamination_free"] is True
    for key, c in report["contamination"].items():
        assert c["exact_overlap"] == 0
        assert c["normalized_overlap"] == 0
        assert c["minhash_overlap"] == 0
        assert c["template_overlap"] == 0


def test_fingerprint_is_stable_and_content_sensitive():
    recs, _ = assign_splits(_records(), seed=1)
    f1 = dataset_fingerprint(recs)
    f2 = dataset_fingerprint(recs)
    assert f1 == f2
    recs[0]["split"] = "test"
    assert dataset_fingerprint(recs) != f1


def test_report_markdown_contains_gate():
    recs, report = assign_splits(_records(), seed=3)
    from src.data.splits import split_summary_markdown

    md = split_summary_markdown(report)
    assert "TRAIN/VALIDATION/TEST CONTAMINATION: PASS" in md
