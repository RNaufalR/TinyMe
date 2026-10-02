"""Content-aware text preprocessing (corrective audit P0-01).

The historical pipeline collapsed whitespace runs globally, which destroys
Python indentation and code-fence content. This module replaces that with a
content-aware normalizer:

* ``prose``      - Unicode NFC, line-ending normalisation, control-character
                   sanitation, intra-line whitespace collapsed, blank-line runs
                   bounded. Never applied to code.
* ``code``       - line-ending normalisation and control-character sanitation
                   ONLY. Indentation, tabs, repeated spaces and newlines are
                   preserved byte-for-byte otherwise. Trailing whitespace is
                   preserved because it can be semantic inside string literals.
* ``markdown``   - fence-aware: text outside ``` fences is normalised as prose,
                   everything inside a fence is preserved exactly.
* ``structured`` - JSON/JSONL/YAML/TOML-ish payloads: never re-indent, never
                   collapse; only newline/control sanitation.

Python code is additionally validated with ``ast.parse`` *after* preprocessing;
``syntax_valid == false`` is a hard rejection for categories that claim to be
valid source code (repair tasks keep the buggy input and the target code in
separate fields, so only the target must parse).
"""
from __future__ import annotations

import json
import re
import unicodedata
from typing import Any, Iterable

__all__ = [
    "KIND_PROSE", "KIND_CODE", "KIND_MARKDOWN", "KIND_STRUCTURED",
    "detect_content_kind", "normalize_text", "validate_code",
    "code_categories", "preprocess_record",
]

KIND_PROSE = "prose"
KIND_CODE = "code"
KIND_MARKDOWN = "markdown"
KIND_STRUCTURED = "structured"

CODE_CATEGORIES = {"programming", "code_gen", "code_repair", "code_explain", "algorithm"}
_LANGUAGE_ALIASES = {
    "py": "python", "python3": "python", "python": "python",
    "json": "json", "jsonl": "json",
    "md": "markdown", "markdown": "markdown",
    "text": "text", "en": "text", "": "text",
}
# Languages for which a real parser/validator exists in this project.
SUPPORTED_CODE_LANGUAGES = ("python", "json")
# Declared-but-not-parsed languages: we do not pretend to validate them.
DECLARED_CODE_LANGUAGES = (
    "javascript", "typescript", "rust", "go", "c", "cpp", "java", "kotlin",
    "shell", "sql", "html", "css", "yaml", "toml",
)

_FENCE_RE = re.compile(r"^(\s*)(`{3,}|~{3,})(.*)$")
_INLINE_COMMENT_LANGS = {"python", "shell", "yaml", "toml"}
_CONTROL_RE = re.compile(r"[\u0000-\u0008\u000b\u000c\u000e-\u001f\u007f]")


def code_categories() -> set[str]:
    return set(CODE_CATEGORIES)


def detect_content_kind(category: str | None, language: str | None = None,
                        text: str = "") -> str:
    """Classify a record so the right normalizer is used."""
    cat = (category or "").strip().lower()
    lang = _LANGUAGE_ALIASES.get((language or "").strip().lower(), (language or "").strip().lower())
    if cat in ("code_gen", "code_repair", "code_explain", "programming", "algorithm"):
        return KIND_CODE
    if cat in ("tool_use", "tool_data"):
        return KIND_STRUCTURED
    if lang in ("markdown",):
        return KIND_MARKDOWN
    if lang in ("json",):
        return KIND_STRUCTURED
    if cat == "language" and text.lstrip().startswith(("```", "~~~")):
        return KIND_MARKDOWN
    return KIND_PROSE


# --------------------------------------------------------------- primitives
def _sanitize_controls(text: str) -> str:
    """Drop C0/C1 control characters except newline and tab (never semantic)."""
    return _CONTROL_RE.sub("", text)


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _normalize_prose_block(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    out_lines: list[str] = []
    for line in text.split("\n"):
        # Collapse intra-line whitespace runs (indentation is not semantic in prose).
        line = re.sub(r"[ \t\f\v]+", " ", line).strip()
        out_lines.append(line)
    text = "\n".join(out_lines)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    return text.strip("\n")


def _normalize_code_block(text: str) -> str:
    """Preserve code exactly; only strip a leading/trailing empty-line margin."""
    return text.strip("\n")


def _split_fences(text: str) -> list[tuple[bool, str]]:
    """Split markdown into [(is_code, chunk)] preserving fence delimiters."""
    lines = text.split("\n")
    chunks: list[tuple[bool, list[str]]] = [(False, [])]
    in_fence = False
    for line in lines:
        m = _FENCE_RE.match(line)
        if m:
            fence_marker = m.group(2)[0]
            if not in_fence:
                in_fence = True
                chunks.append((True, [line]))
                continue
            # closing fence (must use the same marker family)
            if line.strip().startswith(fence_marker):
                chunks[-1][1].append(line)
                in_fence = False
                chunks.append((False, []))
                continue
        chunks[-1][1].append(line)
    return [(flag, "\n".join(ls)) for flag, ls in chunks if "\n".join(ls).strip("\n") or flag]


def normalize_text(text: str, kind: str = KIND_PROSE) -> str:
    """Normalize ``text`` according to its content kind."""
    if text is None:
        return ""
    text = _normalize_newlines(_sanitize_controls(text))
    if kind == KIND_CODE or kind == KIND_STRUCTURED:
        return _normalize_code_block(text)
    if kind == KIND_MARKDOWN:
        parts: list[str] = []
        for is_code, chunk in _split_fences(text):
            parts.append(chunk if is_code else _normalize_prose_block(chunk))
        return "\n".join(p for p in parts if p != "").strip("\n")
    return _normalize_prose_block(text)


# -------------------------------------------------------------- validation
def validate_code(text: str, language: str = "python") -> tuple[bool, str]:
    """Language-aware syntax validation.

    Only languages with a real validator are parsed (``python``, ``json``);
    everything else returns ``not_applicable`` rather than pretending.
    """
    lang = _LANGUAGE_ALIASES.get((language or "").strip().lower(), (language or "").strip().lower())
    stripped = text.strip()
    if not stripped:
        return False, "empty"
    if lang == "python":
        import ast

        try:
            ast.parse(stripped)
            return True, "ast_ok"
        except SyntaxError as exc:
            return False, f"syntax_error: {exc.msg} (line {exc.lineno})"
        except (ValueError, MemoryError) as exc:  # pragma: no cover - defensive
            return False, f"parse_error: {type(exc).__name__}"
    if lang == "json":
        try:
            json.loads(stripped)
            return True, "json_ok"
        except json.JSONDecodeError as exc:
            return False, f"json_error: {exc.msg} (line {exc.lineno})"
    return True, "not_applicable"


_CONTROL_CODE_RE = re.compile(r"<\|code\|>\s*\n?(.*?)(?:<\|endcode\|>|<\|endoftext\|>|\Z)", re.DOTALL)

#: Categories whose *target* is expected to be code rather than prose.
CODE_TARGET_CATEGORIES = {"programming", "code_gen", "code_repair"}


def extract_code_blocks(text: str) -> list[str]:
    """Every explicit code block: markdown fences **and** ``<|code|>`` spans."""
    blocks = [c for is_code, c in _split_fences(text) if is_code]
    cleaned = []
    for b in blocks:
        lines = b.split("\n")
        if lines and _FENCE_RE.match(lines[0]):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith(("```", "~~~")):
            lines = lines[:-1]
        cleaned.append("\n".join(lines))
    for match in _CONTROL_CODE_RE.finditer(text):
        body = match.group(1).strip("\n")
        if body:
            cleaned.append(body)
    return cleaned


def extract_python_source(text: str) -> str:
    """Return the Python source contained in a record (fences/control tokens stripped).

    When the text holds no explicit code block the original text is returned
    unchanged, so callers that *know* the payload is code (e.g. a stdlib file
    record) still validate the whole thing.
    """
    blocks = extract_code_blocks(text)
    return "\n\n".join(blocks) if blocks else text


def extract_target_code(record: dict[str, Any]) -> str:
    """Extract the *code to validate* from a record, or ``""`` if there is none.

    Validation policy (audit §8):

    * explicit code blocks in the target (fences or ``<|code|>`` spans) are
      validated — prose, thoughts and tool JSON around them are ignored;
    * a record that has no segments and *is* source code by construction
      (stdlib files, ``programming`` category) is validated as a whole;
    * prose targets (explanations, reasoning traces) contain no code and are
      therefore **not** Python-AST-policed.
    """
    category = record.get("category")
    if record.get("segments"):
        target = "\n".join(seg.get("text", "") for seg in record["segments"] if seg.get("target"))
        blocks = extract_code_blocks(target)
        if blocks:
            return "\n\n".join(blocks)
        if category == "programming":
            return target
        return ""
    text = record.get("text", "") or ""
    blocks = extract_code_blocks(text)
    if blocks:
        return "\n\n".join(blocks)
    if category in CODE_TARGET_CATEGORIES or record.get("content_kind") == "code":
        return text
    return ""


def preprocess_record(record: dict[str, Any], *, hard_reject_invalid_code: bool = True
                      ) -> tuple[dict[str, Any], str]:
    """Preprocess one record in place-safe fashion.

    Returns ``(record, status)`` where status is one of ``ok``, ``rejected:...``.
    """
    category = record.get("category")
    language = record.get("language", "python" if category in CODE_CATEGORIES else "text")
    text = record.get("text", "") or ""
    kind = detect_content_kind(category, language, text)

    if "segments" in record and record["segments"]:
        for seg in record["segments"]:
            seg_kind = KIND_CODE if seg.get("role") in ("code", "tool_call", "tool_result") else kind
            seg["text"] = normalize_text(seg.get("text", ""), seg_kind)
        record["text"] = serialize_segments(record["segments"]) if not record.get("text") else record["text"]
        text = record["text"]
    else:
        text = normalize_text(text, kind)
        record["text"] = text

    record["content_kind"] = kind

    if hard_reject_invalid_code and category in CODE_CATEGORIES:
        lang = _LANGUAGE_ALIASES.get((language or "").strip().lower(), (language or "").strip().lower())
        # Only the *target code* is validated.  Prompts legitimately contain
        # prose, deliberately-broken code (repair tasks) or non-code content, and
        # prose explanations contain no code at all, so validating whole record
        # texts (or whole targets) was a false-positive machine that rejected
        # perfectly good records (audit §8).
        candidate = extract_target_code(record) if lang == "python" else (
            record.get("answer") or text if lang == "json" else "")
        if lang in SUPPORTED_CODE_LANGUAGES and candidate.strip():
            ok, note = validate_code(candidate, lang)
            record["syntax_valid"] = bool(ok)
            record["syntax_note"] = note
            if not ok:
                return record, f"rejected:invalid_code:{note}"
        else:
            # Non-Python payload, or a target with no code to parse: no Python AST.
            record["syntax_valid"] = None
            record["syntax_note"] = "no_code_in_target" if lang == "python" else "language_not_checked"
    else:
        record.setdefault("syntax_valid", None)
        record.setdefault("syntax_note", "not_applicable")
    return record, "ok"


def serialize_segments(segments: Iterable[dict[str, Any]]) -> str:
    """Canonical on-disk encoding of a segmented record."""
    out: list[str] = []
    for seg in segments:
        role = seg.get("role", "assistant")
        out.append(f"<|{role}|>\n{seg.get('text', '')}")
    return "\n".join(out) + "\n"
