"""Structured record contract shared by every data adapter (corrective audit P0-02).

A training record is no longer an opaque ``text`` string. It is a list of
``Segment`` objects with explicit roles, so the sequence builder knows which
tokens are *context* and which are *targets*::

    SYSTEM        -> context
    USER          -> context
    ASSISTANT     -> target
    THOUGHT       -> target (decided explicitly; concise reasoning is learned)
    TOOL_CALL     -> target (the model must learn to emit valid calls)
    TOOL_RESULT   -> context ONLY (the model must never learn to reproduce it)
    FINAL         -> target
    CODE          -> target

TinyMe control tokens are structural and are never normalised away::

    <|system|> <|user|> <|assistant|> <|thought|> <|answer|> <|tool_call|>
    <|tool_result|> <|final|> <|end_tool_call|> <|end_tool_result|> <|code|>
    <|endcode|> <|endoftext|> <|pad|> <|bos|> <|eos|> <|unk|>

The serialized ``text`` is still emitted for portability (and for data that has
no segmentation), but the trainer consumes the structured form whenever it is
available.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from ..utils.io_utils import sha256_text

# --------------------------------------------------------------------- enums
ROLE_SYSTEM = "system"
ROLE_USER = "user"
ROLE_ASSISTANT = "assistant"
ROLE_THOUGHT = "thought"
ROLE_TOOL_CALL = "tool_call"
ROLE_TOOL_RESULT = "tool_result"
ROLE_FINAL = "final"
ROLE_CODE = "code"
ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_ASSISTANT, ROLE_THOUGHT,
         ROLE_TOOL_CALL, ROLE_TOOL_RESULT, ROLE_FINAL, ROLE_CODE)

#: Roles whose tokens contribute to the loss by default.
TARGET_ROLES = (ROLE_ASSISTANT, ROLE_THOUGHT, ROLE_TOOL_CALL, ROLE_FINAL, ROLE_CODE)
#: Roles that are conditioning context only.
CONTEXT_ROLES = (ROLE_SYSTEM, ROLE_USER, ROLE_TOOL_RESULT)

CATEGORIES = (
    "language",        # A. clean prose / technical text
    "logic",           # B. boolean logic, deduction, pattern, symbolic reasoning
    "math",            # C. arithmetic, algebra, equations, word problems
    "algorithm",       # D. pseudocode, algorithms, complexity, data structures
    "programming",     # E. source code, documentation, examples, tests
    "code_repair",     # F. buggy code -> corrected code + explanation
    "code_gen",        # G. natural-language spec -> implementation
    "code_explain",    # H. code -> structured explanation
    "synthetic",       # I. programmatically generated, execution-verified
    "instruction",     # general instruction following / structured output
    "tool_use",        # J. tool selection / argument construction / grounding
)

TASK_TYPES = (
    "lm",               # plain next-token prediction
    "instruction",      # <|system|><|user|>...<|assistant|>...
    "code_completion",  # code prefix -> code suffix
    "repair",           # buggy code -> fix
    "reasoning",        # problem -> <|thought|> -> <|answer|>
    "tool_call",        # user -> tool call -> result -> final
    "grounded_qa",      # answer grounded in retrieved evidence
    "no_tool",          # explicitly teach that no tool is required
)

SOURCE_TYPES = ("huggingface", "github", "python_stdlib", "synthetic", "docs", "tool_runtime")

ALLOWED_LICENSES = {
    "MIT", "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "PSF-2.0",
    "CC0-1.0", "Unlicense", "0BSD", "Public-Domain",
    "CDLA-Sharing-1.0", "CDLA-Permissive-2.0", "CC-BY-4.0", "CC-BY-SA-4.0", "ODC-By",
    "Synthetic-Verified",
}
EXCLUDED_LICENSES = {"CC-BY-NC-4.0", "GPL-3.0", "AGPL-3.0", "Proprietary"}
UNCLEAR_TOKENS = ("", "unknown", "none", "other", "custom", "license_unclear", "null", "see-license", "nolicense")


def classify_license(spdx: str | None, license_name: str | None = None) -> str:
    """Normalize a license string into ALLOWED / EXCLUDED / LICENSE_UNCLEAR."""
    raw = (spdx or "").strip()
    if not raw:
        raw = (license_name or "").strip()
    if not raw or raw.lower() in UNCLEAR_TOKENS:
        return "LICENSE_UNCLEAR"
    if raw in ALLOWED_LICENSES:
        return raw
    if raw in EXCLUDED_LICENSES:
        return raw
    lowered = raw.lower()
    for lic in ALLOWED_LICENSES:
        if lic.lower() in lowered:
            return lic
    for lic in EXCLUDED_LICENSES:
        if lic.lower() in lowered:
            return lic
    return "LICENSE_UNCLEAR"


# ------------------------------------------------------------------ segments
@dataclass
class Segment:
    """One role-tagged span of a record."""

    role: str
    text: str
    target: bool | None = None

    def contributes_to_loss(self) -> bool:
        if self.target is not None:
            return bool(self.target)
        return self.role in TARGET_ROLES

    def to_dict(self) -> dict[str, Any]:
        return {"role": self.role, "text": self.text,
                "target": self.contributes_to_loss()}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Segment":
        return cls(role=d["role"], text=d.get("text", ""),
                   target=d.get("target") if "target" in d else None)


def serialize_segments(segments: Iterable[Segment | dict[str, Any]]) -> str:
    """Canonical on-disk/on-wire encoding: ``<|role|>\\n<text>`` per segment."""
    out: list[str] = []
    for seg in segments:
        s = seg if isinstance(seg, Segment) else Segment.from_dict(seg)
        out.append(f"<|{s.role}|>\n{s.text}")
    return "\n".join(out) + "\n"


def parse_segments(text: str) -> list[Segment]:
    """Best-effort inverse of :func:`serialize_segments` for legacy records."""
    import re

    parts = re.split(r"<\|([a-z_]+)\|>\n", text)
    segs: list[Segment] = []
    # parts = [prefix, role, body, role, body, ...]
    for i in range(1, len(parts) - 1, 2):
        role, body = parts[i], parts[i + 1]
        if role in ROLES:
            segs.append(Segment(role=role, text=body.rstrip("\n")))
    return segs


# ------------------------------------------------------------ record schema
@dataclass
class TrainingRecord:
    """One standardized training/evaluation sample."""

    text: str
    category: str
    source: str
    source_id: str
    license: str = "Synthetic-Verified"
    source_url: str = ""
    license_url: str = ""
    retrieval_date: str = ""
    task_type: str = "lm"
    language: str = "en"
    verified: bool = False
    verifier: str = ""
    answer: str | None = None
    tests: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""
    # --- corrective additions -------------------------------------------------
    segments: list[dict[str, Any]] = field(default_factory=list)
    template_id: str = ""
    family: str = ""
    split: str = ""
    buggy_input: str = ""
    expected_tool: str = ""
    citations: list[str] = field(default_factory=list)
    group_id: str = ""            # split unit: never split a group across splits
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.retrieval_date:
            self.retrieval_date = time.strftime("%Y-%m-%d")
        if not self.family:
            self.family = self.template_id or f"{self.category}:{self.task_type}"
        if not self.group_id:
            self.group_id = self.template_id or self.family
        if not self.record_id:
            self.record_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"{self.source}:{self.source_id}:{sha256_text(self.text)}"
            ).hex

    # ------------------------------------------------------------- helpers
    def as_segments(self) -> list[Segment]:
        if self.segments:
            return [Segment.from_dict(s) for s in self.segments]
        if "<|" in self.text:
            parsed = parse_segments(self.text)
            if parsed:
                return parsed
        # plain document: a single target span, never a duplicated user/assistant pair
        return [Segment(role=ROLE_ASSISTANT, text=self.text, target=True)]

    def target_text(self) -> str:
        segs = [s for s in self.as_segments() if s.contributes_to_loss()]
        return "\n".join(s.text for s in segs)

    def context_text(self) -> str:
        segs = [s for s in self.as_segments() if not s.contributes_to_loss()]
        return "\n".join(s.text for s in segs)

    def to_dict(self) -> dict[str, Any]:
        d = {
            "record_id": self.record_id,
            "text": self.text,
            "category": self.category,
            "task_type": self.task_type,
            "source": self.source,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "license_url": self.license_url,
            "retrieval_date": self.retrieval_date,
            "license": self.license,
            "language": self.language,
            "template_id": self.template_id,
            "family": self.family,
            "group_id": self.group_id,
            "split": self.split,
            "verified": self.verified,
            "verifier": self.verifier,
            "answer": self.answer,
            "tests": self.tests,
            "quality": self.quality,
            "tags": self.tags,
        }
        if self.segments:
            d["segments"] = [Segment.from_dict(s).to_dict() for s in self.segments]
        if self.buggy_input:
            d["buggy_input"] = self.buggy_input
        if self.expected_tool:
            d["expected_tool"] = self.expected_tool
        if self.citations:
            d["citations"] = self.citations
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainingRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


REQUIRED_FIELDS = ("text", "category", "source", "source_id", "license", "task_type")


def validate_record(rec: Any) -> tuple[bool, str]:
    """Return (ok, reason). Used by the pipeline and by the test suite."""
    d = rec.to_dict() if isinstance(rec, TrainingRecord) else rec
    if not isinstance(d, dict):
        return False, "record is not a dict"
    for f in REQUIRED_FIELDS:
        if not d.get(f):
            return False, f"missing required field: {f}"
    if d["category"] not in CATEGORIES:
        return False, f"unknown category: {d['category']}"
    if d["task_type"] not in TASK_TYPES:
        return False, f"unknown task_type: {d['task_type']}"
    if d["source"] not in SOURCE_TYPES:
        return False, f"unknown source: {d['source']}"
    if not str(d["text"]).strip() and not d.get("segments"):
        return False, "empty text"
    if len(str(d["text"])) > 100_000:
        return False, "text too long (>100k chars)"
    return True, "ok"


def make_segment_record(*, segments: list[Segment], category: str, source: str, source_id: str,
                        task_type: str = "instruction", license: str = "Synthetic-Verified",
                        language: str = "python", template_id: str = "", **kw: Any) -> TrainingRecord:
    """Convenience constructor for segmented (instruction/tool-use) records."""
    text = serialize_segments(segments)
    return TrainingRecord(text=text, category=category, source=source, source_id=source_id,
                          task_type=task_type, license=license, language=language,
                          template_id=template_id,
                          segments=[s.to_dict() for s in segments], **kw)


def record_from_hf(dataset: str, url: str, row: dict[str, Any], category: str,
                   task_type: str, license: str, language: str = "en",
                   text_field: str = "text") -> TrainingRecord:
    return TrainingRecord(
        text=str(row.get(text_field, "")).strip(),
        category=category,
        source="huggingface",
        source_id=f"{dataset}:row_{row.get('row_idx', 0)}",
        license=license,
        source_url=url,
        retrieval_date=time.strftime("%Y-%m-%d"),
        task_type=task_type,
        language=language,
    )
