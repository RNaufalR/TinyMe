"""Strict tool protocol (audit §25).

The model must emit machine-parseable tool calls; the runtime never guesses.
Canonical grammar (the ``end`` spellings follow the audit literally; the compact
aliases shipped with tok-v2 are accepted on input for backwards compatibility)::

    <|tool_call|>{"name": "compute", "arguments": {"expression": "2+2"}}<|end_tool_call|>
    <|tool_result|>{"ok": true, "name": "compute", "result": {"value": 4}}<|end_tool_result|>
    <|final|>The answer is 4.

Everything the model emits is validated; anything malformed is reported as a
``ProtocolError`` with a machine-readable reason instead of being silently
accepted.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

TOOL_CALL = "<|tool_call|>"
END_TOOL_CALL = "<|end_tool_call|>"
END_TOOL_CALL_ALIAS = "<|endtool_call|>"
TOOL_RESULT = "<|tool_result|>"
END_TOOL_RESULT = "<|end_tool_result|>"
END_TOOL_RESULT_ALIAS = "<|endtool_result|>"
FINAL = "<|final|>"
THOUGHT = "<|thought|>"
ASSISTANT = "<|assistant|>"
END_OF_TURN = "<|endoftext|>"

PROTOCOL_TOKENS = [TOOL_CALL, END_TOOL_CALL, TOOL_RESULT, END_TOOL_RESULT, FINAL, THOUGHT]

_CALL_RE = re.compile(
    re.escape(TOOL_CALL) + r"\s*(?P<body>.*?)"
    + r"(?:" + re.escape(END_TOOL_CALL) + "|" + re.escape(END_TOOL_CALL_ALIAS) + r")",
    re.DOTALL)
_RESULT_RE = re.compile(
    re.escape(TOOL_RESULT) + r"\s*(?P<body>.*?)"
    + r"(?:" + re.escape(END_TOOL_RESULT) + "|" + re.escape(END_TOOL_RESULT_ALIAS) + r")",
    re.DOTALL)
_FINAL_RE = re.compile(re.escape(FINAL) + r"\s*(?P<body>.*)", re.DOTALL)
_THOUGHT_RE = re.compile(re.escape(THOUGHT) + r"\s*(?P<body>.*?)(?=<\||\Z)", re.DOTALL)


class ProtocolError(ValueError):
    """Raised for structurally invalid model output."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(f"{reason}: {detail}" if detail else reason)
        self.reason = reason
        self.detail = detail


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict[str, Any]
    raw: str = ""

    def to_dict(self) -> dict:
        return {"name": self.name, "arguments": self.arguments}

    def render(self) -> str:
        return render_tool_call(self.name, self.arguments)


@dataclass
class ToolResult:
    name: str
    ok: bool
    payload: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    duration_s: float = 0.0
    evidence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = {"ok": self.ok, "name": self.name}
        if self.ok:
            d["result"] = self.payload
        else:
            d["error"] = self.error or "unknown error"
        if self.duration_s:
            d["duration_s"] = round(self.duration_s, 4)
        return d

    def render(self) -> str:
        return render_tool_result(self.to_dict())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(", ", ": "))


def render_tool_call(name: str, arguments: dict[str, Any]) -> str:
    return f"{TOOL_CALL}{_canonical_json({'name': name, 'arguments': arguments})}{END_TOOL_CALL}"


def render_tool_result(result: dict[str, Any]) -> str:
    return f"{TOOL_RESULT}{_canonical_json(result)}{END_TOOL_RESULT}"


def render_final(text: str) -> str:
    return f"{FINAL}{text}"


def validate_call_payload(payload: Any) -> tuple[str, dict]:
    if not isinstance(payload, dict):
        raise ProtocolError("not_an_object", f"tool call payload is {type(payload).__name__}")
    if set(payload) - {"name", "arguments"}:
        raise ProtocolError("unknown_keys", f"unexpected keys {sorted(set(payload) - {'name', 'arguments'})}")
    name = payload.get("name")
    if not isinstance(name, str) or not name or not re.fullmatch(r"[a-z_][a-z0-9_]{0,31}", name):
        raise ProtocolError("bad_tool_name", repr(name))
    args = payload.get("arguments", {})
    if not isinstance(args, dict):
        raise ProtocolError("arguments_not_object", type(args).__name__)
    return name, args


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Extract every well-formed tool call; malformed blocks raise."""
    calls: list[ToolCall] = []
    for m in _CALL_RE.finditer(text):
        body = m.group("body").strip()
        try:
            payload = json.loads(body)
        except json.JSONDecodeError as exc:
            raise ProtocolError("call_not_json", f"{exc.msg} at {exc.pos}") from exc
        name, args = validate_call_payload(payload)
        calls.append(ToolCall(name=name, arguments=args, raw=m.group(0)))
    return calls


def parse_tool_results(text: str) -> list[dict]:
    out = []
    for m in _RESULT_RE.finditer(text):
        try:
            out.append(json.loads(m.group("body").strip()))
        except json.JSONDecodeError as exc:
            raise ProtocolError("result_not_json", str(exc)) from exc
    return out


@dataclass
class ParsedOutput:
    thought: str | None
    tool_calls: list[ToolCall]
    final: str | None
    noise: str

    @property
    def is_final(self) -> bool:
        return self.final is not None

    def to_dict(self) -> dict:
        return {"thought": self.thought, "tool_calls": [c.to_dict() for c in self.tool_calls],
                "final": self.final, "noise": self.noise[:200]}


def parse_model_output(text: str) -> ParsedOutput:
    """Parse one model turn. Raises ProtocolError on malformed tool syntax."""
    thought_match = _THOUGHT_RE.search(text)
    thought = thought_match.group("body").strip() if thought_match else None
    tool_calls = parse_tool_calls(text)
    final_match = _FINAL_RE.search(text)
    final = final_match.group("body").strip() if final_match else None
    # Noise = anything outside the protocol constructs.  Every construct is
    # removed with a regex (a literal replace missed the newline after the
    # marker, which made valid finals look like stray text).
    stripped = _CALL_RE.sub(" ", text)
    stripped = _RESULT_RE.sub(" ", stripped)
    stripped = _THOUGHT_RE.sub(" ", stripped)
    stripped = _FINAL_RE.sub(" ", stripped)
    stripped = stripped.replace(THOUGHT, " ")
    noise = stripped.strip()
    return ParsedOutput(thought=thought, tool_calls=tool_calls, final=final, noise=noise)


def validate_output(text: str, *, registry=None) -> list[str]:
    """Return a list of human-readable problems (empty = fully valid)."""
    problems: list[str] = []
    try:
        parsed = parse_model_output(text)
    except ProtocolError as exc:
        return [f"protocol_error:{exc.reason}:{exc.detail}"]
    if parsed.tool_calls and parsed.final is not None:
        problems.append("tool_call_and_final_in_same_turn")
    if not parsed.tool_calls and parsed.final is None:
        problems.append("neither_tool_call_nor_final")
    if parsed.noise:
        problems.append(f"unparsed_text:{parsed.noise[:60]!r}")
    if registry is not None:
        for call in parsed.tool_calls:
            problems.extend(registry.validate_arguments(call.name, call.arguments))
    return problems
