"""Tests for the capability-scaled synthetic generators (v3).

These are *behavioural* checks on the generated training data, not existence
checks.  The properties asserted here are the ones the audit demands:

* the argument a tool call carries is a verbatim span of the user turn (the
  copy invariant that makes tool use learnable);
* generated tool calls are accepted by the **real** runtime registry (a
  trajectory the runtime would reject is worthless supervision);
* a repair trajectory really contains a failing run followed by a passing one;
* an error-recovery trajectory really contains a failure envelope followed by a
  successful call;
* the canonical supervised text (opening marker + body + closing token) carries
  every marker the inference-time protocol needs;
* no generator inflates the corpus by repeating text.
"""
from __future__ import annotations

import json
import random
import re
from collections import Counter

import pytest

from data_sources import synthetic_v3 as s3
from data_sources.synthetic_v2 import _FACT_KEYS, _mock_fetch, _mock_search, run_python
from src.agent.protocol import parse_model_output
from src.agent.tool_registry import ToolRegistry
from src.data.sequence import encode_segment, ROLE_CLOSE_TOKENS

MARKERS = ("<|system|>", "<|user|>", "<|assistant|>", "<|thought|>", "<|tool_call|>",
           "<|tool_result|>", "<|final|>", "<|code|>")


def _segment_bodies(text: str, role: str) -> list[str]:
    """Split raw record text on role markers and return the bodies of ``role``."""
    parts = re.split(r"(<\|[a-z_]+\|>)", text)
    out: list[str] = []
    for i in range(1, len(parts) - 1, 2):
        if parts[i] == f"<|{role}|>":
            out.append(parts[i + 1].strip("\n"))
    return out


def _calls(text: str) -> list[dict]:
    return [json.loads(b) for b in _segment_bodies(text, "tool_call")]


def _results(text: str) -> list[dict]:
    return [json.loads(b) for b in _segment_bodies(text, "tool_result")]


def _user_turn(text: str) -> str:
    body = _segment_bodies(text, "user")[0]
    return body


def _final(text: str) -> str:
    return _segment_bodies(text, "final")[0]


def _supervised_text(rec) -> str:
    """The marker/body composition the loss sees, via the production builder."""
    return "\n".join("".join(encode_segment(s, _TOKENIZER)[0]) for s in rec.as_segments())


class _IdTokenizer:
    """Minimal stand-in: encode_ids returns the text wrapped in a tuple marker.

    The generator does not obtain a tokenizer; tests only need the marker/body
    composition rule from ``encode_segment``, so a lossless identity encoder is
    faithful for structure checks.
    """

    def encode_ids(self, text: str) -> list[str]:  # noqa: D401
        return [text]


_TOKENIZER = _IdTokenizer()


# --------------------------------------------------------------------- copy span
def test_copy_span_answers_are_verbatim_spans_of_the_user_turn():
    records = s3.gen_copy_span(random.Random(11), 40)
    assert len(records) == 40
    for rec in records:
        answer = rec.answer
        assert answer and answer in _user_turn(rec.text), (answer, rec.text)
        assert _final(rec.text) == answer


def test_copy_span_has_no_single_dominant_target_string():
    """Repeated target text is a defect (DEC-008): measure it, do not assume."""
    records = s3.gen_copy_span(random.Random(3), 600)
    finals = Counter(_final(r.text) for r in records)
    assert finals.most_common(1)[0][1] <= len(records) * 0.15, finals.most_common(3)


# ---------------------------------------------------------------------- tool use
def test_generated_tool_calls_are_accepted_by_the_real_registry():
    """Every generated call must survive the runtime's own schema validation."""
    registry = ToolRegistry.default()
    records = s3.gen_tool_use_v3(random.Random(5), 36)
    checked = 0
    for rec in records:
        for call in _calls(rec.text):
            problems = registry.validate_arguments(call["name"], call["arguments"])
            assert not problems, (call, problems)
            checked += 1
    assert checked >= 30, checked


def test_tool_arguments_are_verbatim_spans_of_available_context():
    """The copy invariant: every argument is literally in the context seen so far.

    Multi-step workflows may copy from a *tool result* rather than the user turn
    (that is what makes search->fetch learnable), so the check walks the record
    in order and accumulates the context that precedes each call.
    """
    records = s3.gen_tool_use_v3(random.Random(23), 36)
    checked = 0
    for rec in records:
        parts = re.split(r"(<\|[a-z_]+\|>)", rec.text)
        context = ""
        for i in range(1, len(parts) - 1, 2):
            role, body = parts[i], parts[i + 1]
            if role == "<|tool_call|>":
                call = json.loads(body)
                args = call["arguments"]
                for key, value in args.items():
                    if key in ("timeout_s", "k"):
                        continue  # generator-set hyper-parameters, not copy targets
                    if key == "code":
                        first = value.split("\n")[0].strip()
                        hay = re.sub(r"\s+", " ", context)
                        assert re.sub(r"\s+", " ", first) in hay, (first, context[:200])
                    else:
                        assert str(value) in context, (key, value, context[:200])
                    checked += 1
            else:
                context += body
    assert checked >= 40, checked


def test_supervised_text_carries_every_protocol_marker():
    """The closing tokens the parser needs must be *supervised*, i.e. present."""
    rec = s3.gen_tool_use_v3(random.Random(31), 9)[0]
    supervised = _supervised_text(rec)
    assert "<|tool_call|>" in supervised
    assert ROLE_CLOSE_TOKENS["tool_call"] in supervised
    assert ROLE_CLOSE_TOKENS["tool_result"] in supervised
    assert "<|final|>" in supervised


def test_repair_trajectories_contain_a_real_failure_then_a_real_success():
    records = [r for r in s3.gen_tool_use_v3(random.Random(9), 180)
               if r.template_id == "tool/code_repair"]
    assert records, "generator produced no repair trajectories"
    for rec in records:
        results = _results(rec.text)
        assert len(results) == 2, results
        assert results[0]["ok"] is False and results[1]["ok"] is True, results
        calls = _calls(rec.text)
        assert calls[0]["arguments"]["code"] != calls[1]["arguments"]["code"]
        # find the spec whose solution the repair emitted, then re-run both
        # programs against that spec's own tests (independent of the generator)
        fixed = calls[1]["arguments"]["code"]
        spec = next((s for s in s3._SPEC_LIBRARY if s["solution"].strip() == fixed.strip()), None)
        assert spec is not None, "repaired code is not the reference solution"
        ok_good, out_good = run_python(spec["solution"], spec["tests"])
        assert ok_good, out_good
        ok_bad, out_bad = run_python(calls[0]["arguments"]["code"], spec["tests"])
        assert not ok_bad, ("the 'buggy' program actually passes its tests", out_bad)


def test_recovery_trajectories_contain_a_failure_envelope_then_success():
    records = [r for r in s3.gen_tool_use_v3(random.Random(13), 180)
               if r.template_id == "tool/error_recovery"]
    assert records
    for rec in records:
        results = _results(rec.text)
        assert len(results) >= 2, results
        assert any(r.get("ok") is False or r["result"].get("results") == [] for r in results)
        assert results[-1]["ok"] is True, results


def test_multi_step_search_fetch_records_call_search_before_fetch():
    records = [r for r in s3.gen_tool_use_v3(random.Random(17), 180)
               if r.template_id == "tool/search_fetch"]
    assert records
    for rec in records:
        names = [c["name"] for c in _calls(rec.text)]
        assert names == ["search", "fetch"], names
        fetched = _calls(rec.text)[1]["arguments"]["source_id"]
        first_search = _results(rec.text)[0]["result"]["results"][0]["source_id"]
        assert fetched == first_search


def test_no_tool_records_answer_directly_and_correctly():
    records = s3.gen_no_tool_v3(random.Random(2), 50)
    assert records
    for rec in records:
        assert not _calls(rec.text), "a no-tool record must not call a tool"
        assert rec.answer and rec.answer in _final(rec.text)


def test_error_recovery_typo_query_really_returns_no_results():
    """The 'failure' must be observable, not a stylistic claim."""
    for seed in range(20):
        rng = random.Random(seed)
        key = rng.choice(_FACT_KEYS)
        bad = s3._typo(key, random.Random(seed + 100))
        assert bad != key
        assert _mock_search(bad, k=1)["result"]["results"] == []
        assert _mock_search(key, k=1)["result"]["results"] != []


def test_tool_result_envelopes_match_the_runtime_shape():
    envs = {"search": _mock_search(_FACT_KEYS[0], 1), "fetch": _mock_fetch("FACT-000000"),
            "compute": s3._mock_compute("1 + 1", 2), "code": s3._mock_code("ok\n", code="print(1)")}
    for name, env in envs.items():
        assert set(env) >= {"ok", "name", "result"}
        assert env["name"] == name


def test_generate_corpus_v3_reports_duplicate_drops_instead_of_padding():
    records, provenance = s3.generate_corpus_v3(seed=1, counts={"copy_span": 40, "no_tool": 40,
                                                                "tool_use": 0})
    assert provenance
    for entry in provenance:
        assert entry["requested"] == entry["emitted"] + entry["duplicates_dropped"]
        assert entry["emitted"] == entry["records"]
    total = sum(e["records"] for e in provenance)
    assert len(records) == total
    assert len(set(r.record_id for r in records)) == len(records), "duplicate record ids emitted"


def test_challenge_tool_templates_never_appear_in_the_training_generators():
    """A held-out template that leaks into training is not held out."""
    train_templates = {r.template_id for r in s3.gen_tool_use_v3(random.Random(4), 90)}
    challenge_templates = {r.template_id for r in s3.gen_tool_challenge(random.Random(4), 9)}
    assert challenge_templates
    assert not (train_templates & challenge_templates)


def test_generated_target_text_parses_through_the_protocol_parser():
    """A trajectory the parser cannot read would teach an unusable format."""
    from src.data.dataset_api import load_tokenizer

    tok = load_tokenizer("dataset_v3")  # noqa: F841  (tokeniser parity guard below)
    for rec in s3.gen_tool_use_v3(random.Random(29), 18):
        # parse the *supervised* turn: closing markers are added by the sequence
        # builder, not stored in the raw record text
        supervised = _supervised_text(rec)
        body = supervised.split("<|assistant|>", 1)[1]
        body = body.split("<|tool_result|>", 1)[0]
        parsed = parse_model_output(body)
        assert parsed.tool_calls or parsed.final is not None, body[:200]
        for call in parsed.tool_calls:
            assert call.name in {"search", "fetch", "compute", "code", "files"}
        # the call body must be tokenisable in one piece under the shipped
        # tokenizer (a body the tokenizer mangles cannot be supervised)
        bodies = _segment_bodies(rec.text, "tool_call")
        if bodies:
            marker = tok.encode_ids("<|tool_call|>\n")
            body_ids = tok.encode_ids(bodies[0] + "<|end_tool_call|>")
            assert len(body_ids) > 0 and marker[-1] != body_ids[0]


@pytest.mark.parametrize("pool_name", ["_COMPUTE_PHRASINGS", "_SEARCH_PHRASINGS",
                                       "_FETCH_PHRASINGS", "_FETCH_FIRST_PHRASINGS",
                                       "_CODE_RUN_PHRASINGS", "_CODE_REPAIR_PHRASINGS",
                                       "_NO_TOOL_ADD", "_NO_TOOL_SUB", "_NO_TOOL_MUL",
                                       "_NO_TOOL_UPPER", "_NO_TOOL_REVERSE"])
def test_phrase_pools_are_long_enough_to_avoid_target_repetition(pool_name):
    pool = getattr(s3, pool_name)
    assert len(pool) >= 5, (pool_name, len(pool))
    assert len(set(pool)) == len(pool), f"{pool_name} contains duplicates"


def test_v6_profile_widens_the_shape_distribution():
    """The v6 profile must break the fixed operand shape the model memorised.

    Measured defect: the v5-trained model answered the held-out prompt
    ``11 + 22`` with ``111 + 22`` - it rewrote unseen operands back into the
    training shape (fixed 2-3 digit operands).  The v6 profile is only useful if
    it actually produces a wider distribution, so that is asserted here.
    """
    import random as _random

    from data_sources import synthetic_v3 as S

    S.set_profile("v6")
    try:
        rng = _random.Random(20261003)
        exprs = [S._expression_v6(rng)[0] for _ in range(2000)]
        digits = [len(e.split()[0]) for e in exprs]
        assert min(digits) == 1 and max(digits) >= 4, (min(digits), max(digits))
        assert len(set(digits)) >= 3, sorted(set(digits))
        values = [S._expression_v6(rng)[1] for _ in range(200)]
        assert all(isinstance(v, int) for v in values)

        spans = S.gen_copy_span_v6(_random.Random(5), 160)
        kinds = {r.template_id for r in spans}
        assert len(kinds) == 8, sorted(kinds)
        assert len({r.answer for r in spans}) == 160, "spans must not repeat"
        for rec in spans:
            segs = rec.as_segments()
            user = next(s.text for s in segs if s.role == "user")
            assert rec.answer in user, "verbatim invariant"
            assert rec.verifier == "verbatim_containment_check"

        no_tool = S.gen_no_tool_v3(_random.Random(9), 100)
        assert len(no_tool) == 100
        assert all(r.answer for r in no_tool)
    finally:
        S.set_profile("v5")


def test_v6_profile_is_opt_in():
    """v5 must stay the default so dataset_v5 remains byte-reproducible."""
    import random as _random

    from data_sources import synthetic_v3 as S

    assert S.PROFILE == "v5"
    spans = S.gen_copy_span(_random.Random(3), 8)
    assert {r.template_id for r in spans} == {"copy/span/kind0", "copy/span/kind1",
                                              "copy/span/kind2", "copy/span/kind3"}
