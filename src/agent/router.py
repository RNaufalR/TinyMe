"""Deterministic tool router (audit §28).

The router is intentionally simple and auditable: it inspects the user request
and decides whether the runtime should answer directly, compute, retrieve,
fetch or execute code.  Heuristics are explicit (regex / keyword classes), so
routing accuracy can be measured and the decision can be logged with a reason.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

MATH_RE = re.compile(r"^[\s\d\.\+\-\*/%\(\)\^,eE]+$|^\s*[\d\.]+\s*[\+\-\*/%]\s*[\d\.]+")
CODE_HINTS = ("run this code", "execute", "what does this program output", "trace the code",
              "code:", "def ", "print(")
RETRIEVAL_HINTS = ("who", "what", "when", "where", "which", "how many", "how much", "why",
                   "capital", "population", "latest", "current", "according to", "source",
                   "definition", "license", "licence", "complexity", "big-o")
FETCH_HINTS = ("open the source", "fetch the passage", "read more about")
COMPUTE_VERBS = ("compute", "calculate", "evaluate", "what is", "how much is", "arithmetic")


@dataclass
class RouteDecision:
    tool: str | None           # None => answer directly
    arguments: dict
    reason: str
    confidence: float

    def to_dict(self) -> dict:
        return {"tool": self.tool, "arguments": self.arguments, "reason": self.reason,
                "confidence": self.confidence}


_QUESTION_VERBS = re.compile(
    r"(?i)^\s*(what\s+is|what\s+are|how\s+much\s+is|compute|calculate|evaluate|"
    r"what\s+does|how\s+many\s+is|solve)\s+")
_NUMBER_WORDS = ("plus", "minus", "times", "divided by", "multiplied by", "percent of", "% of")


def _looks_like_arithmetic(text: str) -> bool:
    """True when the request is a closed arithmetic question (never a lookup)."""
    stripped = text.strip().rstrip("?.! ")
    lowered = stripped.lower()
    for phrase in _NUMBER_WORDS:
        lowered = lowered.replace(phrase, " ")
    core = _QUESTION_VERBS.sub("", lowered).strip()
    if not core or not re.search(r"\d", core):
        return False
    # remove words that are pure filler so "of", "the", "sum of" do not hide the math
    core = re.sub(r"[a-z]{1,4}\b", " ", core)
    core = core.replace(",", " ").strip()
    return bool(core) and bool(MATH_RE.search(core)) or bool(re.fullmatch(r"[\d\s\.\+\-\*/%]+", core))


_DIMENSIONLESS = {"it", "this", "that", "the expression", "the result"}


def route(request: str) -> RouteDecision:
    """Classify a user request into the smallest sufficient tool (or none)."""
    text = request.strip()
    low = text.lower()

    if any(h in low for h in FETCH_HINTS):
        return RouteDecision("fetch", {"source_id": _extract_id(text)},
                            "request explicitly asks to open a stored passage", 0.7)
    if any(h in low for h in CODE_HINTS):
        return RouteDecision("code", {"code": _extract_code(text)},
                            "request contains or asks about executable code", 0.8)
    if _looks_like_arithmetic(text):
        expr = _extract_expression(text)
        return RouteDecision("compute", {"expression": expr},
                            "request is a closed arithmetic/math expression", 0.85)
    if any(h in low for h in RETRIEVAL_HINTS):
        return RouteDecision("search", {"query": text},
                            "factual question needs a sourced passage", 0.6)
    return RouteDecision(None, {}, "no tool required: answer from parametric knowledge", 0.5)


def _extract_expression(text: str) -> str:
    text = re.sub(r"(?i)^\s*(what is|compute|calculate|evaluate|how much is)\s*", "", text.strip())
    text = text.rstrip("?.! ").replace("^", "**").replace("×", "*").replace("÷", "/")
    return re.sub(r"[^0-9a-zA-Z_\.\+\-\*/%\(\)\s,]", "", text).strip()


def _extract_code(text: str) -> str:
    fence = re.search(r"```(?:python)?\n(.*?)```", text, re.DOTALL)
    if fence:
        return fence.group(1)
    return text


def _extract_id(text: str) -> str:
    match = re.search(r"\b([A-Za-z]{1,3}\d{1,4})\b", text)
    return match.group(1) if match else text.strip()
