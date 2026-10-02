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
