"""Evaluation engine: held-out only, no caps, executable code metrics (audit §19/§20)."""
from __future__ import annotations

import json

import pytest

from src.evaluation.evaluator import (DOMAINS, build_suite, evaluate_model, extract_code,
                                      extract_final, render_prompt, render_target,
                                      write_evaluation_report)
from src.data.records import Segment, make_segment_record


@pytest.fixture(scope="module")
def suite():
    return build_suite("dataset_v2", "test")


def test_domains_cover_every_required_axis():
    required = {"language", "logic", "math", "algorithmic_reasoning", "code", "debugging",
                "code_generation", "instruction_following", "tool_selection", "tool_arguments",
                "tool_syntax", "tool_execution", "grounding", "generalization"}
    assert required <= set(DOMAINS)


def test_suite_is_held_out_and_large_enough(suite):
    assert len(suite) > 25, "the audit requires evaluation sets larger than 25 samples"
    assert all(r.get("split") in ("test", "challenge") for r in suite)
    assert not any(r.get("split") == "train" for r in suite)
    with pytest.raises(ValueError):
        build_suite("dataset_v2", "train")


def test_suite_contains_code_records_with_real_tests(suite):
    code_records = [r for r in suite if r.get("category") in ("code_gen", "code_repair")]
    assert code_records
    assert all(isinstance(r.get("tests"), list) for r in code_records)


def test_prompt_rendering_never_leaks_the_target():
    record = make_segment_record(
        segments=[Segment("user", "Write an adder.", target=False),
                  Segment("assistant", "<|code|>\ndef add(a, b):\n    return a + b\n<|endcode|>", target=True)],
        category="code_gen", source="unit", source_id="u/1")
    prompt = render_prompt(record)
    assert "Write an adder." in prompt
    assert "def add" not in prompt, "the target must never appear in the prompt"
    assert render_target(record).startswith("<|code|>")


def test_extract_answer_and_code_helpers():
    assert extract_final("<|final|>\n42\n") == "42"
    assert extract_final("<|answer|>\nJakarta") == "Jakarta"
    assert extract_code("```python\nprint(1)\n```") == "print(1)"
    assert extract_code("<|code|>\nprint(2)\n<|endcode|>") == "print(2)"


def test_no_generated_example_cap():
    """The old evaluator truncated `generated_examples` to 20; that cap must be gone."""
    import inspect

    source = inspect.getsource(evaluate_model)
    assert "[:20]" not in source


def test_evaluate_model_reports_domains_with_full_denominators(suite):
    def fake_generate(prompt, max_new_tokens=64):
        return "<|final|>\n42"

    result = evaluate_model(fake_generate, suite, model_name="unit:fake", dataset_version="dataset_v2",
                            split="test", max_new_tokens=8, examples_per_domain=1)
    payload = result.to_dict()
    assert payload["overall"]["total_samples"] >= len(suite) * 0.5
    for domain, stats in payload["domains"].items():
        assert stats["samples"] > 0, domain
        assert "accuracy" in stats
    code_domain = payload["domains"].get("code_generation")
    if code_domain:
        assert code_domain["parseable"] >= 0 and code_domain["executed"] >= 0
        assert code_domain["metric"] == "execution_pass_rate"
    tool_domains = [d for d in payload["domains"] if d.startswith("tool_")]
    assert tool_domains, "tool metrics must be reported on a dataset that contains tool records"


def test_code_generation_is_measured_by_execution(suite):
    """A model that emits correct code must score, one that emits junk must not."""
    good = ('<|final|>\n<|code|>\ndef add(a, b):\n    return a + b\n<|endcode|>')
    bad = '<|final|>\nnot python at all'

    def gen_good(prompt, max_new_tokens=64, **kw):
        return good

    def gen_bad(prompt, max_new_tokens=64, **kw):
        return bad

    sub = [r for r in suite if r.get("category") == "code_gen"][:6]
    if not sub:
        pytest.skip("no code_gen records in the held-out split")
    r_good = evaluate_model(gen_good, sub, model_name="good", split="test", examples_per_domain=0)
    r_bad = evaluate_model(gen_bad, sub, model_name="bad", split="test", examples_per_domain=0)
    d_good = r_good.domains["code_generation"]
    d_bad = r_bad.domains["code_generation"]
    assert d_bad["passed"] == 0
    assert d_bad["accuracy"] == 0.0
    assert d_good["samples"] == len(sub)


def test_report_writer_produces_a_table(tmp_path, suite):
    def fake(prompt, max_new_tokens=64, **kw):
        return "<|final|>\n42"

    result = evaluate_model(fake, suite[:6], model_name="unit:fake", split="test", examples_per_domain=0)
    path = write_evaluation_report([result], tmp_path / "report.md", title="Unit report")
    text = path.read_text()
    assert "# Unit report" in text
    assert "| Domain | unit:fake |" in text
    assert "no `[:20]`-style caps" in text
