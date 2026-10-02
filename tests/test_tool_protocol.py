"""Strict tool protocol: syntax, validation, rendering (audit §25)."""
from __future__ import annotations

import json

import pytest

from src.agent import protocol as P
from src.agent.tool_registry import ToolRegistry


def test_canonical_round_trip():
    text = P.render_tool_call("compute", {"expression": "2 + 2"})
    assert text.startswith(P.TOOL_CALL) and text.endswith(P.END_TOOL_CALL)
    calls = P.parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "compute"
    assert calls[0].arguments == {"expression": "2 + 2"}
    assert json.loads(text[len(P.TOOL_CALL):-len(P.END_TOOL_CALL)])["name"] == "compute"


def test_compact_aliases_are_accepted_on_input():
    text = f'{P.TOOL_CALL}{{"name": "compute", "arguments": {{"expression": "1+1"}}}}<|endtool_call|>'
    call = P.parse_tool_calls(text)[0]
    assert call.name == "compute"


def test_malformed_calls_raise_with_a_reason():
    cases = {
        "call_not_json": f"{P.TOOL_CALL}{{not json}}{P.END_TOOL_CALL}",
        "not_an_object": f"{P.TOOL_CALL}[1, 2, 3]{P.END_TOOL_CALL}",
        "bad_tool_name": f'{P.TOOL_CALL}{{"name": 7, "arguments": {{}}}}{P.END_TOOL_CALL}',
        "arguments_not_object": f'{P.TOOL_CALL}{{"name": "compute", "arguments": []}}{P.END_TOOL_CALL}',
        "unknown_keys": f'{P.TOOL_CALL}{{"name": "compute", "arguments": {{}}, "extra": 1}}{P.END_TOOL_CALL}',
    }
    for expected, text in cases.items():
        with pytest.raises(P.ProtocolError) as exc:
            P.parse_tool_calls(text)
        assert exc.value.reason == expected, (expected, exc.value.reason)


def test_unterminated_call_is_not_silently_accepted():
    text = f'{P.TOOL_CALL}{{"name": "compute", "arguments": {{"expression": "1+1"}}}}'
    assert P.parse_tool_calls(text) == []
    parsed = P.parse_model_output(text)
    assert parsed.tool_calls == [] and parsed.final is None


def test_final_and_thought_extraction():
    text = f"{P.THOUGHT}\nUse compute.\n{P.FINAL}\nThe answer is 4."
    parsed = P.parse_model_output(text)
    assert parsed.thought == "Use compute."
    assert parsed.final == "The answer is 4."
    assert parsed.tool_calls == []


def test_validate_output_flags_problems():
    assert P.validate_output(f"{P.FINAL}\n42") == []
    assert "tool_call_and_final_in_same_turn" in P.validate_output(
        f'{P.TOOL_CALL}{{"name": "compute", "arguments": {{"expression": "1"}}}}{P.END_TOOL_CALL}{P.FINAL} x')
    assert "neither_tool_call_nor_final" in P.validate_output("just some text")
    problems = P.validate_output(f'{P.TOOL_CALL}{{"name": "nope", "arguments": {{}}}}{P.END_TOOL_CALL}',
                                 registry=ToolRegistry.default())
    assert any("unknown_tool" in p for p in problems)


def test_result_rendering_is_valid_json_and_reparseable():
    text = P.render_tool_result({"ok": True, "name": "compute", "result": {"value": 4}})
    payload = P.parse_tool_results(text)[0]
    assert payload["ok"] is True and payload["result"]["value"] == 4


def test_protocol_tokens_are_registered_in_the_tokenizer():
    from src.tokenizer.bpe import SPECIAL_TOKENS

    for token in (P.TOOL_CALL, P.END_TOOL_CALL, P.TOOL_RESULT, P.END_TOOL_RESULT, P.FINAL):
        assert token in SPECIAL_TOKENS, token


def test_tokenizer_round_trips_protocol_text(fake_tokenizer):
    text = P.render_tool_call("compute", {"expression": "17*3"}) + P.render_final("51")
    ids = fake_tokenizer.encode_ids(text)
    assert fake_tokenizer.decode(ids, skip_special_tokens=False) == text
    # special tokens must be encoded as single ids, not spelled out byte by byte
    assert len(ids) < len(text)


def test_tool_specs_match_runtime_registry():
    """Training data must teach exactly the arguments the runtime accepts.

    Regression guard for the drift that was found during the audit: the
    synthetic corpus used to advertise ``search.max_results``, ``fetch.url`` and
    ``code.source``/``code.tests`` while ``ToolRegistry`` validates
    ``query``/``k``, ``source_id`` and ``code`` - so every trajectory the model
    learned was rejected by the runtime.
    """
    from data_sources.synthetic_v2 import TOOL_SPECS

    registry = ToolRegistry.default()
    assert set(TOOL_SPECS) == set(registry.names())
    for name, spec in TOOL_SPECS.items():
        runtime = registry.specs[name].parameters
        assert set(spec["args"]) == set(runtime), (
            name, sorted(spec["args"]), sorted(runtime))
        for arg, meta in runtime.items():
            if meta.get("required"):
                assert arg in spec.get("required", []), (name, arg)
        type_names = {"string": "string", "int": "integer", "integer": "integer",
                      "float": "number", "bool": "boolean"}
        for arg, meta in spec["args"].items():
            declared = type_names.get(meta.split("=")[0], meta.split("=")[0])
            assert declared == runtime[arg]["type"], (name, arg, meta, runtime[arg])


def test_generated_tool_calls_pass_runtime_validation():
    """Every tool call in the shipped corpus must satisfy the registry schema."""
    import random

    from data_sources.synthetic_v2 import gen_tool_use

    records = gen_tool_use(random.Random(20261002), 21)
    registry = ToolRegistry.default()
    checked = 0
    for record in records:
        if record.category != "tool_use":
            continue
        for segment in record.segments:
            if segment["role"] != "tool_call":
                continue
            # The segment body is bare JSON; the sequence builder wraps it in the
            # canonical protocol tokens.  Rendering it here proves the corpus text
            # is parseable by the same parser the agent runtime uses.
            rendered = P.render_tool_call(**{k: v for k, v in
                                             __import__("json").loads(segment["text"]).items()})
            for call in P.parse_tool_calls(rendered):
                checked += 1
                assert registry.validate_arguments(call.name, call.arguments) == [], (
                    call.name, call.arguments)
    assert checked > 0
