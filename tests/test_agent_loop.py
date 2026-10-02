"""Agent runtime: routing, bounded execution, loop/step control, grounding (audit §28-§31)."""
from __future__ import annotations

import json

import pytest

from src.agent import AgentRuntime, LoopDetector, plan_for, route, scripted_policy
from src.agent.executor import render_transcript
from src.agent.protocol import FINAL, TOOL_CALL, render_final, render_tool_call, render_tool_result
from src.agent.tool_registry import ToolRegistry


@pytest.fixture(scope="module")
def retrieved_index(tmp_path_factory):
    from src.tools.retrieval import Document, RetrievalIndex, write_corpus

    docs = [Document(source_id="D1", title="Binary search",
                     text="Binary search halves the interval each step, giving O(log n) comparisons.",
                     url="https://example.org/bs", source="unit", license="CC0-1.0")]
    path = tmp_path_factory.mktemp("retr") / "index.json"
    write_corpus(docs, path.parent / "corpus.jsonl")
    RetrievalIndex(docs).save(path)
    return path


@pytest.fixture()
def runtime(retrieved_index):
    from src.tools.context import ToolContext

    ctx = ToolContext.create(index_path=retrieved_index)
    rt = AgentRuntime(context=ctx, max_steps=4)
    yield rt
    rt.close()


# ------------------------------------------------------------------- routing
def test_router_classifies_requests_deterministically():
    assert route("What is 35% of 1200 plus 19?").tool == "compute"
    assert route("compute 17 * 3").tool == "compute"
    assert route("Trace the code:\nprint(1)").tool == "code"
    assert route("According to sources, what is the complexity?").tool == "search"
    assert route("Say hello politely.").tool is None
    assert route("What is 2 + 2") == route("What is 2 + 2")


def test_plan_for_builds_bounded_plans():
    plan = plan_for("What is 8 * 9?", max_steps=3)
    assert plan.max_steps == 3
    assert len(plan.steps) == 1 and plan.steps[0]["name"] == "compute"
    assert plan.steps[0]["why"]
    assert plan_for("hello").steps == []


def test_loop_detector_flags_repeats():
    det = LoopDetector(max_repeats=1)
    call = {"name": "compute", "arguments": {"expression": "1+1"}}
    assert det.check(call) is False
    assert det.check(call) is True
    assert det.triggered and det.report()["triggered"]


# ------------------------------------------------------------------ episodes
def test_single_tool_call_then_final(runtime):
    outputs = [render_tool_call("compute", {"expression": "1200*0.35+19"}),
               render_final("439.0")]
    traj = runtime.solve("What is 35% of 1200 plus 19?", scripted_policy(outputs))
    assert traj.solved and traj.stop_reason == "final"
    assert traj.final == "439.0"
    assert traj.metrics["tool_calls"] == 1
    assert traj.steps[0].result["ok"] is True
    assert "439" in json.dumps(traj.steps[0].result)


def test_answer_directly_without_tools(runtime):
    traj = runtime.solve("Say hello.", scripted_policy([render_final("Hello!")]))
    assert traj.solved and traj.metrics["tool_calls"] == 0


def test_retrieval_episode_produces_citable_evidence(runtime):
    outputs = [render_tool_call("search", {"query": "binary search complexity", "k": 2}),
               render_final("Binary search is O(log n) per https://example.org/bs")]
    traj = runtime.solve("What is the complexity of binary search?", scripted_policy(outputs))
    assert traj.solved
    assert traj.evidence["items"], traj.evidence
    assert traj.grounding["grounded"] is True, traj.grounding
    assert traj.evidence["items"][0]["source_id"] == "D1"


def test_invented_citation_is_caught_by_the_evidence_engine(runtime):
    outputs = [render_tool_call("search", {"query": "binary search complexity", "k": 2}),
               render_final("Complexity is O(log n) per https://made-up.example.org/paper")]
    traj = runtime.solve("complexity?", scripted_policy(outputs))
    assert traj.grounding["grounded"] is False
    assert traj.grounding["unsupported_urls"] == ["https://made-up.example.org/paper"]


def test_loop_is_detected_and_the_episode_stops(runtime):
    same = render_tool_call("compute", {"expression": "1+1"})
    traj = runtime.solve("loop please", scripted_policy([same, same, same]))
    assert traj.stop_reason == "loop_detected"
    assert traj.metrics["tool_calls"] == 1, "the repeated call must not be executed twice"
    assert not traj.solved


def test_malformed_protocol_output_stops_with_a_reason(runtime):
    traj = runtime.solve("bad", scripted_policy([f"{TOOL_CALL}{{oops}}<|end_tool_call|>"]))
    assert traj.stop_reason.startswith("protocol_error:")
    assert traj.solved is False
    assert traj.steps[-1].error


def test_max_steps_is_respected(runtime):
    outputs = [render_tool_call("compute", {"expression": f"{i}+1"}) for i in range(10)]
    traj = runtime.solve("many steps", scripted_policy(outputs))
    assert traj.stop_reason == f"max_steps_reached({runtime.max_steps})"
    assert len(traj.steps) == runtime.max_steps


def test_tool_call_budget_is_enforced(retrieved_index):
    from src.tools.context import ToolContext

    ctx = ToolContext.create(index_path=retrieved_index)
    ctx.max_tool_calls = 1
    rt = AgentRuntime(context=ctx, max_steps=5)
    try:
        outputs = [render_tool_call("compute", {"expression": f"{i}+1"}) for i in range(5)]
        traj = rt.solve("budget", scripted_policy(outputs))
        assert traj.stop_reason.startswith("tool_call_budget_exhausted")
        assert traj.metrics["tool_calls"] == 1
    finally:
        rt.close()


def test_unknown_tool_is_reported_to_the_model_not_executed(runtime):
    traj = runtime.solve("x", scripted_policy([render_tool_call("nope", {}), render_final("done")]))
    assert traj.steps[0].result["ok"] is False
    assert "unknown_tool" in traj.steps[0].result["error"]
    assert traj.solved, "the model must be able to recover after a rejected call"


def test_transcript_is_replayable_and_renders_both_protocol_sides(runtime):
    outputs = [render_tool_call("compute", {"expression": "2+2"}), render_final("4")]
    traj = runtime.solve("2+2?", scripted_policy(outputs))
    transcript = render_transcript(traj, system="You are TinyMe.")
    assert transcript.startswith("<|system|>")
    assert "<|user|>2+2?" in transcript
    assert "<|tool_call|>" in transcript
    assert "<|tool_result|>" in transcript
    assert transcript.rstrip().endswith("<|final|>4")
    assert json.loads(json.dumps(traj.to_dict()))["solved"] is True


def test_code_tool_episode_runs_in_the_sandbox(runtime):
    outputs = [render_tool_call("code", {"code": "print(sum(i*i for i in range(5)))"}),
               render_final("30")]
    traj = runtime.solve("What does this print: sum of squares 0..4?", scripted_policy(outputs))
    assert traj.steps[0].result["result"]["stdout"].strip() == "30"
    assert traj.steps[0].result["result"]["isolation_level"] in {
        "namespace(net+mount+pid)", "namespace(net+mount)", "netns-only", "rlimit-only", "bwrap"}


def test_registry_schemas_are_the_model_facing_surface(runtime):
    schemas = runtime.tool_schemas()
    assert [s["name"] for s in schemas] == ["code", "compute", "fetch", "files", "search"]
    # the surface must stay small: no nested objects, no arrays of objects
    for schema in schemas:
        for spec in schema["parameters"]["properties"].values():
            assert spec["type"] in ("string", "integer", "number"), (schema["name"], spec)
