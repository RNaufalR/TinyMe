"""Curriculum partition tests (DEC-013).

The staged SFT curriculum is only meaningful if the partition is *total and
disjoint* over the structured corpus: every instruction/tool record must land in
exactly one stage, and a new generator family must not silently fall out of
training.  These tests re-derive that property from the real generators instead
of from a hand-written list, so a generator that starts emitting an unassigned
template fails the build.
"""
from __future__ import annotations

import pytest

from data_sources import synthetic_v2 as s2
from data_sources import synthetic_v3 as s3
from src.data.curriculum import (DRILL_TEMPLATE_PREFIXES, STAGE_DRILL, STAGE_TRAJECTORY,
                                 TRAJECTORY_TEMPLATE_PREFIXES, partition_report, record_filter,
                                 stage_of, template_id, tool_call_count)

#: Small explicit draws: enough to exercise every branch of both blocks without
#: turning the unit suite into a corpus build.
V2_COUNTS = {"arithmetic": 6, "word_problems": 4, "sequences": 4, "boolean": 4, "deduction": 4,
             "algorithm_trace": 4, "code_gen": 6, "code_repair": 4, "code_explain": 4,
             "instruction": 6, "tool_use": 12}
V3_COUNTS = {"copy_span": 12, "no_tool": 24, "tool_use": 24, "algorithm": 6}


def _as_dicts(records):
    return [r.to_dict() if hasattr(r, "to_dict") else r for r in records]


#: Prefix tables are the human-readable description of the corpus; the *stage*
#: itself is derived from the record structure, because two generator blocks can
#: render the same template id with a different number of tool calls
#: (``tool/search_single`` is one call in the v2 block, search + fetch in v3).
ALL_PREFIXES = DRILL_TEMPLATE_PREFIXES + TRAJECTORY_TEMPLATE_PREFIXES


def _assert_prefixes_cover_templates(records):
    """Every structured template must be described by one of the prefix tables."""
    for record in records:
        if not record.get("segments"):
            continue
        template = template_id(record)
        assert any(template.startswith(p) for p in ALL_PREFIXES), template
        stage = stage_of(record)
        assert stage in (STAGE_DRILL, STAGE_TRAJECTORY)


def test_v3_block_templates_are_all_assigned():
    records, _ = s3.generate_corpus_v3(seed=20261003, counts=V3_COUNTS, scale=1.0, profile="v6")
    dicts = _as_dicts(records)
    report = partition_report(dicts)
    assert report["structured_records"] > 0
    assert report["unassigned_templates"] == {}, report["unassigned_templates"]
    assert report["assigned"] == report["structured_records"]
    assert report["drill"] > 0 and report["trajectory"] > 0
    _assert_prefixes_cover_templates(dicts)


def test_v2_block_templates_are_all_assigned():
    records, _ = s2.generate_corpus(seed=20261002, counts=V2_COUNTS, scale=1.0)
    dicts = _as_dicts(records)
    report = partition_report(dicts)
    assert report["structured_records"] > 0
    assert report["unassigned_templates"] == {}, report["unassigned_templates"]
    assert report["assigned"] == report["structured_records"]
    assert report["drill"] > 0 and report["trajectory"] > 0
    _assert_prefixes_cover_templates(dicts)


def test_partition_is_disjoint_and_stage_specific():
    """A drill record must never be selected by the trajectory filter, and back."""
    drill = {"segments": [{"role": "user", "text": "x"}], "template_id": "copy/span/kind0"}
    trajectory = {"segments": [{"role": "user", "text": "x"}, {"role": "tool_call", "text": "{}"},
                               {"role": "tool_result", "text": "{}"}, {"role": "tool_call", "text": "{}"}],
                  "template_id": "tool/search_fetch"}
    plain = {"text": "a document"}
    assert stage_of(drill) == STAGE_DRILL
    assert stage_of(trajectory) == STAGE_TRAJECTORY
    assert stage_of(plain) is None
    assert record_filter(STAGE_DRILL)(drill) and not record_filter(STAGE_DRILL)(trajectory)
    assert record_filter(STAGE_TRAJECTORY)(trajectory) and not record_filter(STAGE_TRAJECTORY)(drill)
    assert record_filter("sft")(drill) and record_filter("sft")(trajectory)
    assert not record_filter("sft")(plain)
    with pytest.raises(ValueError):
        record_filter("nonsense")


def test_prefixes_are_disjoint():
    """The two prefix lists must not overlap, which would make the partition ambiguous."""
    for a in DRILL_TEMPLATE_PREFIXES:
        for b in TRAJECTORY_TEMPLATE_PREFIXES:
            assert not a.startswith(b) and not b.startswith(a), (a, b)


def test_template_id_falls_back_to_legacy_field():
    assert template_id({"template_id": "copy/span/kind3"}) == "copy/span/kind3"
    assert template_id({"template": "tool/compute"}) == "tool/compute"
    assert template_id({}) == ""
