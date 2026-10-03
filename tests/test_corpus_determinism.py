"""Corpus identity must not depend on anything a clock can change (audit §6).

The training corpus is described by fingerprints (dataset, tokenizer, shards).
Those hashes are only meaningful if regenerating the corpus on a different day,
on a busier machine, or in a different process produces the same bytes. The
failure mode this file guards against is subtle: a tool transcript that embeds a
*runtime-measured* value (``duration_s``) is different every time it is built,
so the corpus — and every fingerprint of it — silently stops being reproducible.

Two independent guarantees are checked:

1. the envelopes the generators splice into records are built through a filter
   that removes timing fields — demonstrated by running the *same* call with the
   clock forced to jump, and requiring byte-identical output;
2. every ``duration_s`` that does appear in the built corpus is one of the
   frozen literals declared by the mock builders, i.e. content, not measurement.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DURATION_RE = re.compile(r'\\?"duration_s\\?":\s*([0-9.eE+-]+)')

#: Literals the mock envelope builders are allowed to use (data_sources/synthetic_v2.py).
FROZEN_MOCK_DURATIONS = {"0.0001", "0.0002", "0.0004", "0.0728", "0.1199"}


def _frozen_set_from_source() -> set[str]:
    """Read the literals back out of the mock builders instead of trusting them."""
    from data_sources import synthetic_v2 as v2

    known_id = next(iter(__import__("data_sources.synthetic_v2", fromlist=["EVIDENCE_DB"])
                         .EVIDENCE_DB.values()))["source_id"]
    values = {str(v2._mock_search("x")["duration_s"]),
              str(v2._mock_fetch(known_id)["duration_s"]),          # hit branch
              str(v2._mock_fetch("FACT-000000")["duration_s"]),     # miss branch
              str(v2._mock_compute("1 + 1", 2)["duration_s"]),
              str(v2._mock_code("")["duration_s"]),
              str(v2._mock_code("")["result"]["duration_s"])}
    return values


def test_mock_envelopes_declare_constants():
    values = _frozen_set_from_source()
    assert values == FROZEN_MOCK_DURATIONS, (
        f"the mock builders changed their timing literals: {sorted(values)} — every corpus "
        "fingerprint in the repository is invalidated by that, on purpose or not")


def test_runtime_envelope_ignores_the_clock(monkeypatch):
    """A tool that 'takes' an hour must serialize exactly like a fast one."""
    from data_sources import synthetic_v3 as v3

    baseline = v3._envelope("search", query="population of jakarta", k=1)
    assert "duration_s" not in baseline, "the runtime envelope still leaks its duration"

    import src.agent.tool_registry as registry
    real_monotonic = registry.time.monotonic
    ticks = iter(range(0, 10_000_000, 3_600))

    def fake_monotonic():                      # every tool call 'takes an hour'
        return float(next(ticks, 9_999_999))

    monkeypatch.setattr(registry.time, "monotonic", fake_monotonic)
    slow = v3._envelope("search", query="population of jakarta", k=1)
    monkeypatch.setattr(registry.time, "monotonic", real_monotonic)

    assert slow == baseline, "an envelope changed when only the clock changed"


def test_runtime_and_mock_envelopes_agree_on_the_payload():
    """The transcript the model reads must match what the runtime really returns."""
    from data_sources import synthetic_v2 as v2
    from data_sources import synthetic_v3 as v3

    runtime = json.loads(v3._envelope("compute", expression="53 + 4699"))
    mock = v2._mock_compute("53 + 4699", 4699 + 53)
    assert runtime["ok"] == mock["ok"] and runtime["name"] == mock["name"]
    assert runtime["result"] == mock["result"], (
        "the mock compute payload drifted from the runtime payload; records would teach a "
        "shape the agent never sees at inference time")

    # The corpus builders search with the catalogue key itself, which is also the
    # only query form the mocked lookup accepts.
    key = "population of jakarta"
    got = json.loads(v3._envelope("search", query=key, k=1))
    expected = v2._mock_search(key, k=1)
    assert [r["source_id"] for r in got["result"]["results"]] == \
           [r["source_id"] for r in expected["result"]["results"]], (
        "the runtime search and the mocked search disagree on which source a query retrieves")


@pytest.mark.parametrize("split", ["train", "validation", "test", "challenge"])
def test_built_corpus_contains_no_measured_duration(split: str):
    path = ROOT / "datasets" / "versions" / "dataset_v9" / f"{split}.jsonl"
    if not path.exists():
        pytest.skip(f"{path} not built in this checkout")
    allowed = _frozen_set_from_source()
    seen: dict[str, int] = {}
    supervised = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        for segment in record.get("segments") or []:
            matches = DURATION_RE.findall(segment.get("text", ""))
            for value in matches:
                seen[value] = seen.get(value, 0) + 1
            if matches and segment.get("target", True):
                supervised += 1
    unexpected = {v: n for v, n in seen.items() if v not in allowed}
    assert not unexpected, (
        f"{split}: values that are not frozen literals: {unexpected} — the corpus would "
        "not reproduce and its fingerprint would not identify it")
    assert supervised == 0, (
        f"{split}: {supervised} supervised segments carry a duration; the model would be "
        "trained to emit a timing value the runtime cannot promise")
    if split == "train":
        assert seen, "expected the tool transcripts to contain the frozen literals"


def test_generator_output_is_stable_across_processes(tmp_path: Path):
    """Two fresh interpreters, same seed, same bytes (cheap subset of the corpus)."""
    script = (
        "import hashlib, json, sys;"
        "sys.path.insert(0, r'%s');"
        "from data_sources import synthetic_v3 as v3;"
        "import random;"
        "rng = random.Random(20261002);"
        "rows = [v3._envelope('search', query='FACT-2082DD', k=1),"
        "        v3._envelope('compute', expression='53 + 4699'),"
        "        v3._envelope('fetch', source_id='FACT-2082DD')];"
        "print(hashlib.sha256(json.dumps(rows).encode()).hexdigest())" % ROOT)
    digests = []
    for _ in range(2):
        import subprocess

        proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                              timeout=300, env={"PATH": "/usr/bin:/bin", "PYTHONPATH": ""})
        assert proc.returncode == 0, proc.stderr[-500:]
        digests.append(proc.stdout.strip())
    assert digests[0] == digests[1], f"generator output differs between processes: {digests}"
    assert hashlib.sha256(digests[0].encode()).hexdigest()
