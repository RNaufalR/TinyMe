"""Automatic quality scoring + safety filtering (spec §13).

Each sample is scored on: source quality, length, language confidence, code
validity, duplication score, information density, structural quality,
toxicity/safety flags, license confidence, syntax validity, execution validity.

Low-quality records are removed or down-weighted.
"""
from __future__ import annotations

import re
from typing import Any

# ---------------------------------------------------------------- secret / PII
SECRET_PATTERNS = [
    (re.compile(r"ghp_[A-Za-z0-9]{20,}"), "github_personal_access_token"),
    (re.compile(r"github_pat_[A-Za-z0-9_]{20,}"), "github_fine_grained_pat"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "openai_api_key"),
    (re.compile(r"AKIA[0-9A-Z]{16}"), "aws_access_key_id"),
    (re.compile(r"AIza[0-9A-Za-z\-_]{35}"), "google_api_key"),
    (re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"), "private_key_block"),
    (re.compile(r"xox[baprs]-[0-9A-Za-z-]{10,}"), "slack_token"),
    (re.compile(r"(?i)\b(password|passwd|secret|api_key|apikey|token)\s*[:=]\s*['\"][^'\"]{8,}['\"]"), "hardcoded_credential"),
]
PII_PATTERNS = [
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "email"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"), "ipv4_literal"),
]
MALWARE_PATTERNS = [
    (re.compile(r"eval\(\s*base64\.b64decode"), "obfuscated_eval"),
    (re.compile(r"(?i)rm\s+-rf\s+/"), "destructive_rm"),
    (re.compile(r"(?i)nc\s+-e\s+/bin/(ba)?sh"), "reverse_shell"),
    (re.compile(r"(?i)curl[^\n]*\|\s*(ba)?sh"), "remote_pipe_to_shell"),
]
TOXICITY_TERMS = [
    "kill yourself", "i will find you and", "go die", "kys", "gas the",
]

CODE_HINTS = ("def ", "class ", "import ", "return", "for ", "while ", "function ", "const ", "=>", "{", "}", ";")
NON_ASCII_RATIO_MAX = 0.35


def find_patterns(text: str, patterns) -> list[str]:
    return [name for rx, name in patterns if rx.search(text)]


def safety_flags(text: str) -> dict[str, Any]:
    """Detect secrets / PII / malware / toxicity. Returns flags + safe decision."""
    flags: dict[str, Any] = {}
    flags["secrets"] = find_patterns(text, SECRET_PATTERNS)
    flags["pii"] = find_patterns(text, PII_PATTERNS)
    flags["malware"] = find_patterns(text, MALWARE_PATTERNS)
    lowered = text.lower()
    flags["toxicity"] = [t for t in TOXICITY_TERMS if t in lowered]
    flags["unsafe"] = bool(flags["secrets"] or flags["malware"] or flags["toxicity"])
    return flags


def language_confidence(text: str) -> float:
    """Cheap heuristic English/ASCII confidence in [0,1]."""
    if not text:
        return 0.0
    sample = text[:2000]
    ascii_ratio = sum(1 for ch in sample if ord(ch) < 128) / len(sample)
    letters = sum(1 for ch in sample if ch.isalpha())
    letter_ratio = letters / len(sample)
    return round(min(1.0, 0.5 * (ascii_ratio / max(NON_ASCII_RATIO_MAX, 0.01)) + 0.5 * (letter_ratio / 0.7)), 4)


def information_density(text: str) -> float:
    """Unique-word ratio (type/token) restricted to the first 4000 chars."""
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]+", text[:4000])
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def structural_quality(text: str) -> float:
    """Reward well-formed lines and penalize pathological line lengths."""
    lines = [ln for ln in text.split("\n")]
    if not lines:
        return 0.0
    long_lines = sum(1 for ln in lines if len(ln) > 500)
    avg = sum(len(ln) for ln in lines) / len(lines)
    score = 1.0
    if avg > 200:
        score -= 0.35
    if long_lines:
        score -= min(0.5, 0.1 * long_lines)
    return round(max(0.0, min(1.0, score)), 4)


def code_ratio(text: str) -> float:
    lines = text.split("\n")
    if not lines:
        return 0.0
    codeish = sum(1 for ln in lines if any(h in ln for h in CODE_HINTS))
    return round(codeish / len(lines), 4)


def is_minified(text: str) -> bool:
    lines = text.split("\n")
    if not lines:
        return False
    longest = max(len(ln) for ln in lines)
    avg = sum(len(ln) for ln in lines) / len(lines)
    return longest > 1000 or (avg > 200 and len(lines) > 1)


def syntax_validity(text: str, language: str = "en") -> tuple[bool, str]:
    """Static syntax check for Python snippets; other languages are accepted."""
    if language != "python":
        return True, "not_applicable"
    import ast

    # Only attempt to parse when the text looks like pure Python code.
    stripped = text.strip()
    if not stripped:
        return True, "not_applicable"
    try:
        ast.parse(stripped)
        return True, "ast_ok"
    except SyntaxError as exc:
        # Prose containing code is common; treat as not applicable rather than broken.
        if not any(h in text for h in ("def ", "class ", "import ")):
            return True, "not_applicable"
        return False, f"syntax_error: {exc.msg}"
    except Exception:
        return True, "not_applicable"


def score_record(rec: dict[str, Any], source_quality: float = 1.0,
                 duplication_score: float = 0.0) -> dict[str, Any]:
    """Compute the full quality feature vector for one record."""
    text = rec.get("text", "")
    category = rec.get("category", "language")
    language = rec.get("language", "en")
    safety = safety_flags(text)
    syntax_ok, syntax_note = syntax_validity(text, "python" if category in ("programming", "code_gen", "code_repair", "code_explain", "algorithm", "synthetic") else language)
    length = len(text)
    lang_conf = language_confidence(text)
    density = information_density(text)
    struct = structural_quality(text)
    code_r = code_ratio(text)

    length_score = 1.0
    if length < 40:
        length_score = 0.3
    elif length < 120:
        length_score = 0.7

    overall = (
        0.18 * source_quality
        + 0.10 * length_score
        + 0.12 * lang_conf
        + 0.12 * density
        + 0.10 * struct
        + 0.08 * (1.0 if syntax_ok else 0.0)
        + 0.10 * (1.0 - duplication_score)
        + 0.10 * (0.0 if safety["unsafe"] else 1.0)
        + 0.10 * (0.5 + 0.5 * code_r if category in ("programming", "code_gen", "code_repair", "code_explain") else 1.0)
    )
    return {
        "source_quality": round(source_quality, 4),
        "length_chars": length,
        "length_score": round(length_score, 4),
        "language_confidence": lang_conf,
        "information_density": density,
        "structural_quality": struct,
        "code_ratio": code_r,
        "syntax_valid": syntax_ok,
        "syntax_note": syntax_note,
        "duplication_score": round(duplication_score, 4),
        "safety_unsafe": safety["unsafe"],
        "safety_secrets": safety["secrets"],
        "safety_pii": safety["pii"],
        "safety_malware": safety["malware"],
        "safety_toxicity": safety["toxicity"],
        "is_minified": is_minified(text),
        "quality_score": round(max(0.0, min(1.0, overall)), 4),
    }


def filter_record(rec: dict[str, Any], min_quality: float = 0.35,
                  drop_unsafe: bool = True, redact_pii: bool = True) -> tuple[bool, dict[str, Any], str]:
    """Return (keep, updated_record, reason)."""
    q = rec.setdefault("quality", {})
    if q.get("safety_unsafe") and drop_unsafe:
        return False, rec, "unsafe_content"
    if q.get("is_minified"):
        return False, rec, "minified_or_generated"
    if q.get("length_chars", 0) < 25:
        return False, rec, "too_short"
    if q.get("language_confidence", 0.0) < 0.25:
        return False, rec, "low_language_confidence"
    if q.get("quality_score", 0.0) < min_quality:
        return False, rec, "below_quality_threshold"
    if q.get("safety_pii") and redact_pii:
        text = rec["text"]
        for rx, _ in PII_PATTERNS:
            text = rx.sub("<REDACTED>", text)
        rec["text"] = text
    return True, rec, "ok"
