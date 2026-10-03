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


def test_every_multi_phrasing_family_is_measurable_held_out():
    """A family with >=3 phrasings must appear in train *and* in both held-out splits.

    Measured defect (2026-10-03): the category-level loop put whole families in
    one split, so 1610 of 1993 families had no held-out instance - including every
    multi-step tool family, which made held-out tool capability unmeasurable.
    """
    recs = []
    for fam, groups in (("tool/search", 4), ("copy/span", 5)):
        for g in range(groups):
            for i in range(3):
                recs.append({"text": f"{fam} group {g} item {i} unique text {fam}{g}{i}",
                             "record_id": f"{fam}-{g}-{i}", "category": "tool_use",
                             "family": fam, "group_id": f"{fam}#{g}", "template_id": fam,
                             "split": ""})
    out, report = assign_splits(recs, seed=1234)
    per_family = {}
    for r in out:
        per_family.setdefault(r["family"], set()).add(r["split"])
    for fam in ("tool/search", "copy/span"):
        assert {"train", "validation", "test"} <= per_family[fam], per_family[fam]
    by_group: dict[str, set[str]] = {}
    for r in out:
        by_group.setdefault(r["group_id"], set()).add(r["split"])
    assert all(len(v) == 1 for v in by_group.values())
    assert report["contamination_free"] is True
    assert report["family_coverage"]["families_with_held_out_instances"] >= 2


def test_held_out_instance_counts_are_recorded_per_family():
    recs, report = assign_splits(_records(n_families=4, per_family=6), seed=5)
    assert report["family_coverage"]["families_with_held_out_instances"] + \
        report["family_coverage"]["families_assigned_wholly"] == 4
    assert report["counts"]["train"] > 0 and report["counts"]["validation"] > 0
