"""Tool implementations: compute, retrieval, search, fetch, code, files (audit §25-§32)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.agent.tool_registry import ToolRegistry
from src.tools.compute import compute as compute_fn
from src.tools.context import ToolContext
from src.tools.retrieval import Document, RetrievalIndex, write_corpus


@pytest.fixture()
def ctx(tmp_path):
    docs = [
        Document(source_id="D1", title="Binary search", text="Binary search halves the interval, giving O(log n).",
                 url="https://example.org/bs", source="unit", license="CC0-1.0"),
        Document(source_id="D2", title="Dijkstra", text="Dijkstra computes shortest paths for non-negative weights.",
                 url="https://example.org/dij", source="unit", license="CC0-1.0"),
        Document(source_id="D3", title="Jakarta", text="Jakarta's administrative population was 10,679,951 in 2024.",
                 url="https://example.org/jkt", source="unit", license="CC0-1.0"),
    ]
    write_corpus(docs, tmp_path / "corpus.jsonl")
    index = RetrievalIndex(docs)
    index.save(tmp_path / "index.json")
    context = ToolContext.create(index_path=tmp_path / "index.json")
    yield context
    context.close()


# --------------------------------------------------------------------- compute
def test_compute_exact_arithmetic():
    assert compute_fn("(127*3+19)/7")["value"] == pytest.approx(400 / 7)
    assert compute_fn("2**10")["value"] == 1024
    assert compute_fn("factorial(6)")["value"] == 720
    assert compute_fn("sqrt(144)")["value"] == 12.0


def test_compute_refuses_non_numeric_and_dangerous_input():
    for expression in ("__import__('os').system('ls')", "open('/etc/passwd').read()", "x + 1", "", "1 if 2 else 3"):
        result = compute_fn(expression)
        assert result["ok"] is False, expression
        assert "error" in result
    assert compute_fn("9**9**9")["ok"] is False  # guarded against huge exponents


# ------------------------------------------------------------------- retrieval
def test_retrieval_is_deterministic_and_relevant(ctx):
    first = ctx.index.search("how fast is binary search", k=2)
    second = ctx.index.search("how fast is binary search", k=2)
    assert first == second
    assert first[0]["source_id"] == "D1"
    assert first[0]["score"] > 0
    assert first[0]["license"] == "CC0-1.0", "passages must carry provenance"


def test_search_returns_citation_ready_evidence(ctx):
    from src.tools.search import search

    result = search(ctx, "population of jakarta", k=2)
    assert result["ok"] and result["provider"] == "local-bm25"
    assert result["evidence"][0]["source_id"] == "D3"
    assert result["evidence"][0]["url"].startswith("https://")


def test_search_without_an_index_is_unavailable_not_fabricated():
    from src.tools.search import search

    empty = ToolContext(index=None, workspace=None)
    result = search(empty, "anything")
    assert result["ok"] is False and result["results"] == []
    assert "no_search_provider" in result["error"]


def test_fetch_resolves_ids_and_rejects_urls(ctx):
    from src.tools.fetch import fetch

    ok = fetch(ctx, "D2")
    assert ok["ok"] and "Dijkstra" in ok["text"] and ok["evidence"][0]["source_id"] == "D2"
    assert fetch(ctx, "https://example.org/x")["ok"] is False
    assert fetch(ctx, "NOPE")["ok"] is False


# ----------------------------------------------------------------------- code
@pytest.mark.skipif(
    not __import__("src.sandbox.isolation", fromlist=["x"]).detected_summary()["capabilities"].get("unshare_net"),
    reason="network egress blocking requires a network namespace; the host provides "
           "rlimit-only isolation — audit §9 environment-dependent")
def test_code_tool_runs_but_cannot_reach_the_network(ctx):
    from src.tools.code import run_code

    result = run_code(ctx, "print(sum(range(11)))\n"
                           "import socket\n"
                           "try:\n"
                           "    socket.create_connection(('1.1.1.1', 443), timeout=2)\n"
                           "    print('NETWORK_OK')\n"
                           "except Exception as exc:\n"
                           "    print('network blocked', type(exc).__name__)")
    assert result["ok"], result
    assert "55" in result["stdout"]
    assert "NETWORK_OK" not in result["stdout"]
    assert result["network_enforced"] is True


def test_code_tool_reports_timeouts_and_crashes_honestly(ctx):
    from src.tools.code import run_code

    timeout = run_code(ctx, "while True:\n    pass", timeout_s=2)
    assert timeout["ok"] is False and timeout["timed_out"] is True
    crash = run_code(ctx, "raise SystemExit(3)")
    assert crash["ok"] is False and crash["exit_code"] == 3 and "error" in crash


# ---------------------------------------------------------------------- files
def test_files_tool_is_confined_to_the_workspace(ctx):
    from src.tools.files import files

    ctx.workspace.write("notes.txt", "hello workspace")
    listing = files(ctx, "list")
    assert any(f["path"] == "notes.txt" for f in listing["files"])
    read = files(ctx, "read", path="notes.txt")
    assert read["ok"] and read["text"] == "hello workspace"
    for path in ("../secret.txt", "/etc/passwd", "~/x"):
        assert files(ctx, "read", path=path)["ok"] is False


# ------------------------------------------------------------------- registry
def test_registry_exposes_exactly_five_documented_tools(ctx):
    registry = ToolRegistry.default(context=ctx)
    assert registry.names() == ["code", "compute", "fetch", "files", "search"]
    for schema in registry.schemas():
        assert schema["description"] and schema["parameters"]["type"] == "object"
        assert "$schema" not in json.dumps(schema)  # keep the surface small


def test_registry_rejects_bad_arguments_without_calling_the_tool(ctx):
    registry = ToolRegistry.default(context=ctx)
    assert registry.call("nope", {}) .error.startswith("unknown_tool")
    assert "missing_argument" in registry.call("compute", {}).error
    assert "bad_type" in registry.call("compute", {"expression": 5}).error
    assert "unknown_arguments" in registry.call("compute", {"expression": "1", "z": 1}).error
    assert "not_a_choice" in registry.call("files", {"action": "erase"}).error


def test_registry_counts_and_logs_calls(ctx):
    registry = ToolRegistry.default(context=ctx)
    registry.call("compute", {"expression": "1+1"})
    registry.call("fetch", {"source_id": "NOPE"})
    assert len(registry.call_log) == 2
    assert [c["ok"] for c in registry.call_log] == [True, False]
