"""Tool registry: the strict, validated model-facing surface (audit §4/§16).

The registry is the security boundary between model output and real execution:
anything that does not match the declared schema must be refused *before* the
tool implementation runs.
"""
from __future__ import annotations

import pytest

from src.agent.tool_registry import ToolRegistry, ToolSpec


def _registry(calls: list[dict]) -> ToolRegistry:
    reg = ToolRegistry()

    def echo(**kwargs):
        calls.append(kwargs)
        return {"ok": True, "echo": kwargs}

    reg.register(ToolSpec(
        name="probe", description="test tool", fn=echo,
        parameters={
            "query": {"type": "string", "required": True, "max_length": 8},
            "k": {"type": "integer", "required": False, "min": 1, "max": 5},
            "mode": {"type": "string", "required": False, "choices": ["fast", "slow"]},
            "flags": {"type": "array", "required": False},
        }))
    return reg


def test_schema_advertises_required_arguments_only():
    reg = _registry([])
    (schema,) = reg.schemas()
    assert schema["name"] == "probe"
    assert schema["parameters"]["required"] == ["query"]
    assert set(schema["parameters"]["properties"]) == {"query", "k", "mode", "flags"}


def test_valid_arguments_reach_the_tool():
    calls: list[dict] = []
    reg = _registry(calls)
    result = reg.call("probe", {"query": "abc", "k": 3, "mode": "fast"})
    assert result.ok and calls == [{"query": "abc", "k": 3, "mode": "fast"}]


@pytest.mark.parametrize("args,expected", [
    ({}, "missing_argument:query"),
    ({"query": "abc", "nope": 1}, "unknown_arguments"),
    ({"query": 5}, "bad_type:query"),
    ({"query": "abc", "k": "3"}, "bad_type:k"),
    ({"query": "abc", "k": 0}, "below_minimum:k"),
    ({"query": "abc", "k": 99}, "above_maximum:k"),
    ({"query": "waytoolongquery"}, "too_long:query"),
    ({"query": "abc", "mode": "warp"}, "not_a_choice:mode"),
    ({"query": "abc", "flags": "x"}, "bad_type:flags"),
])
def test_invalid_arguments_are_refused_before_execution(args, expected):
    calls: list[dict] = []
    reg = _registry(calls)
    result = reg.call("probe", args)
    assert not result.ok, (args, result)
    assert expected in (result.error or ""), (args, result.error)
    assert calls == [], "the tool implementation must not run for an invalid call"


def test_unknown_tool_is_refused():
    calls: list[dict] = []
    reg = _registry(calls)
    result = reg.call("rm_rf", {"path": "/"})
    assert not result.ok and "unknown_tool" in result.error
    assert calls == []


def test_call_log_records_success_and_failure():
    calls: list[dict] = []
    reg = _registry(calls)
    reg.call("probe", {"query": "ok"})
    reg.call("probe", {})
    assert len(reg.call_log) == 2
    assert reg.call_log[0]["ok"] is True
    assert reg.call_log[1]["ok"] is False and "missing_argument" in reg.call_log[1]["error"]


def test_tool_exceptions_become_structured_errors_not_crashes():
    reg = ToolRegistry()

    def boom(**_):
        raise RuntimeError("kaboom")

    reg.register(ToolSpec(name="boom", description="explodes", fn=boom,
                          parameters={"x": {"type": "string", "required": True}}))
    result = reg.call("boom", {"x": "1"})
    assert not result.ok
    assert "RuntimeError" in (result.error or "") and "kaboom" in (result.error or "")


def test_duplicate_registration_is_rejected():
    reg = _registry([])
    with pytest.raises(ValueError):
        reg.register(ToolSpec(name="probe", description="dup", fn=lambda **_: {"ok": True},
                              parameters={}))


def test_boolean_is_not_accepted_as_an_integer():
    calls: list[dict] = []
    reg = _registry(calls)
    result = reg.call("probe", {"query": "abc", "k": True})
    assert not result.ok and "bad_type:k" in result.error
    assert calls == []


def test_default_registry_exposes_the_five_documented_tools():
    reg = ToolRegistry.default()
    assert reg.names() == ["code", "compute", "fetch", "files", "search"]
    for schema in reg.schemas():
        assert schema["parameters"]["type"] == "object"
        assert schema["description"]
