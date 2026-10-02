"""Standardized training-record schema shared across every data adapter.

A record is a plain dict so it serializes to JSONL without extra machinery.
All adapters MUST produce records that validate against `validate_record`.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from ..utils.io_utils import sha256_text

CATEGORIES = (
    "language",       # A. fundamental language / clean prose / technical text
    "logic",          # B. boolean logic, deduction, pattern, symbolic reasoning
    "math",           # C. arithmetic, algebra, equations, word problems
    "algorithm",      # D. pseudocode, algorithms, complexity, data structures
    "programming",    # E. source code, comments, documentation, examples, tests
    "code_repair",    # F. buggy code -> corrected code + explanation
    "code_gen",       # G. natural-language spec -> implementation
    "code_explain",   # H. code -> structured explanation
    "synthetic",      # I. programmatically generated, execution-verified curriculum
    "instruction",    # general instruction following / structured output
)

TASK_TYPES = (
    "lm",              # plain next-token prediction
    "instruction",     # <|system|><|user|>...<|assistant|>...
    "code_completion",  # code prefix -> code suffix
    "repair",          # buggy code -> fix
    "reasoning",       # problem -> structured <|thought|> -> answer
)

SOURCE_TYPES = ("huggingface", "github", "python_stdlib", "synthetic", "docs")

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


@dataclass
class TrainingRecord:
    """One standardized training/evaluation sample."""

    text: str
    category: str
    source: str
    source_id: str
    license: str = "Synthetic-Verified"
    source_url: str = ""
    retrieval_date: str = ""
    task_type: str = "lm"
    language: str = "en"
    verified: bool = False
    verifier: str = ""
    answer: str | None = None
    tests: list[str] = field(default_factory=list)
    quality: dict[str, Any] = field(default_factory=dict)
    record_id: str = ""

    def __post_init__(self) -> None:
        if not self.retrieval_date:
            self.retrieval_date = time.strftime("%Y-%m-%d")
        if not self.record_id:
            self.record_id = uuid.uuid5(
                uuid.NAMESPACE_URL, f"{self.source}:{self.source_id}:{sha256_text(self.text)}"
            ).hex

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "text": self.text,
            "category": self.category,
            "task_type": self.task_type,
            "source": self.source,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "retrieval_date": self.retrieval_date,
            "license": self.license,
            "language": self.language,
            "verified": self.verified,
            "verifier": self.verifier,
            "answer": self.answer,
            "tests": self.tests,
            "quality": self.quality,
        }

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
    if not str(d["text"]).strip():
        return False, "empty text"
    if len(str(d["text"])) > 100_000:
        return False, "text too long (>100k chars)"
    return True, "ok"


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
