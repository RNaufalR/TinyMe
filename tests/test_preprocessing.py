"""Content-aware preprocessing: code survives, prose is still normalised (P0-01)."""
from __future__ import annotations

import pytest

from src.data.preprocess import (KIND_CODE, KIND_MARKDOWN, KIND_PROSE,
                                 detect_content_kind, normalize_text,
                                 preprocess_record, validate_code)

PY_SNIPPET = '''def classify(x):
    if x > 0:
        return "positive"
    else:
        return "non-positive"
'''


def test_python_indentation_survives():
    out = normalize_text(PY_SNIPPET, KIND_CODE)
    assert out == PY_SNIPPET.strip("\n")
    assert "        return \"positive\"" in out
    import ast

    ast.parse(out)


def test_baseline_style_collapsing_is_gone():
    text = "def f():\n    x = 1\n        y = 2\n    return x"
    out = normalize_text(text, KIND_CODE)
    assert out == text
    # the historical normalizer collapsed runs of 3+ spaces; verify we do not
    assert "        y = 2" in out


def test_markdown_fences_are_preserved_exactly():
    md = "## Title\n\nSome   prose    here.\n\n```python\ndef f():\n        return 1\n```\n\nEnd.\n"
    out = normalize_text(md, KIND_MARKDOWN)
    assert "def f():\n        return 1" in out
    assert "Some prose here." in out          # prose part normalised
    assert "## Title" in out


def test_prose_is_normalised():
    out = normalize_text("Hello    world.\n\n\n\n\nSecond line.  ", KIND_PROSE)
    assert out == "Hello world.\n\n\nSecond line."


def test_unicode_survives_prose():
    text = "  café  \u2014  naïve  "
    out = normalize_text(text, KIND_PROSE)
    assert "café" in out and "naïve" in out and "\u2014" in out


def test_control_characters_removed_but_tabs_kept_in_code():
    text = "def f():\n\tif True:\n\t\tpass\x00\x07\n"
    out = normalize_text(text, KIND_CODE)
    assert "\x00" not in out and "\x07" not in out
    assert "\tif True:" in out


def test_python_syntax_validation_hard_rejects_broken_code():
    ok, note = validate_code("def f(:\n  return 1", "python")
    assert ok is False and "syntax_error" in note
    ok2, note2 = validate_code("def f():\n    return 1", "python")
    assert ok2 is True and note2 == "ast_ok"


def test_non_python_is_not_ast_parsed():
    rust = "fn main() { let x = 1; println!(\"{}\", x); }"
    ok, note = validate_code(rust, "rust")
    assert ok is True and note == "not_applicable"


def test_preprocess_record_rejects_malformed_python():
    rec = {"text": "def f(:\n  return 1", "category": "code_gen", "language": "python",
           "task_type": "code_completion", "source": "synthetic", "source_id": "x", "license": "MIT"}
    _, status = preprocess_record(rec)
    assert status.startswith("rejected:invalid_code")


def test_repair_task_keeps_buggy_input_but_validates_target():
    rec = {"text": "def f():\n    return 1", "category": "code_repair", "language": "python",
           "task_type": "repair", "buggy_input": "def f(:\n    return 1",
           "source": "synthetic", "source_id": "x", "license": "MIT"}
    _, status = preprocess_record(rec)
    assert status == "ok"


@pytest.mark.parametrize("category,language,expected", [
    ("code_gen", "python", KIND_CODE),
    ("language", "en", KIND_PROSE),
    ("language", "markdown", KIND_MARKDOWN),
    ("tool_use", "python", None),
])
def test_detect_content_kind(category, language, expected):
    kind = detect_content_kind(category, language)
    if expected:
        assert kind == expected
    else:
        assert kind in (KIND_CODE, "structured")
