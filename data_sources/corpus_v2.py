"""Corpus assembly for dataset_v2.

Sources (all offline, licensed, with provenance):

* ``python_stdlib`` — documented functions/classes extracted from the local
  CPython 3.11 standard library (PSF-2.0).
* ``huggingface`` — locally cached, previously verified slices of GSM8K (MIT),
  MBPP (CC-BY-4.0 via bigcode) and TinyStories (CDLA-Sharing-1.0).
* ``synthetic`` — deterministic verified generators from
  :mod:`data_sources.synthetic_v2` (Synthetic-Verified).

No network access is required or attempted; if a source is unavailable it is
skipped and reported, never faked.
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

from src.data.records import Segment, TrainingRecord, make_segment_record
from src.utils.io_utils import REPO_ROOT

from . import stdlib_adapter, stdlib_v2
from .synthetic_v2 import generate_corpus
from .synthetic_v3 import generate_corpus_v3

logger = logging.getLogger("tinyme.corpus")

RAW_DIR = REPO_ROOT / "datasets" / "raw" / "huggingface"


# ------------------------------------------------------------------ slicess
def _load_slice(name: str) -> dict[str, Any] | None:
    path = RAW_DIR / name
    if not path.exists():
        logger.warning("cached slice %s missing — source skipped", name)
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def ingest_gsm8k(limit: int = 15) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    data = _load_slice("gsm8k_hf_slice.json")
    if not data:
        return [], []
    records: list[TrainingRecord] = []
    for row in data.get("rows", [])[:limit]:
        question = str(row.get("question", "")).strip()
        answer = str(row.get("answer", "")).strip()
        if "####" not in answer:
            continue
        reasoning, final = answer.split("####", 1)
        final = final.strip()
        segs = [Segment("system", "You are TinyMe. Solve grade-school mathematics step by step.", target=False),
                Segment("user", question, target=False),
                Segment("assistant", f"<|thought|>\n{reasoning.strip()}\n<|answer|>\n{final}", target=True)]
        rec = make_segment_record(segments=segs, category="math", source="huggingface",
                                  source_id=f"gsm8k/{row.get('row_idx')}",
                                  task_type="reasoning", template_id="hf/gsm8k/v1",
                                  license=data.get("license", "MIT"), language="en",
                                  source_url=data.get("url", ""),
                                  retrieval_date=data.get("retrieval_date", ""),
                                  verified=True, verifier="published_dataset",
                                  answer=final)
        records.append(rec)
    prov = [{
        "source": "huggingface", "source_id": "gsm8k", "dataset_or_repo_name": "openai/gsm8k",
        "url": data.get("url"), "license": data.get("license", "MIT"),
        "license_url": data.get("license_url", ""), "retrieval_date": data.get("retrieval_date", ""),
        "source_type": "text + reasoning", "records": len(records),
        "preprocessing_performed": ["content_extraction", "segment_split", "quality_filter", "dedup"],
        "inclusion_status": "INCLUDED",
        "inclusion_reason": "permissive licence (MIT); reference chain-of-thought supervision for arithmetic word problems",
    }]
    return records, prov


def ingest_mbpp(limit: int = 20) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    data = _load_slice("mbpp_hf_slice.json")
    if not data:
        return [], []
    records: list[TrainingRecord] = []
    for row in data.get("rows", [])[:limit]:
        text = str(row.get("text", row.get("prompt", ""))).strip()
        code = str(row.get("code", "")).strip()
        tests = row.get("test_list") or []
        if not text or not code:
            continue
        test_src = "\n".join(str(t) for t in tests)
        segs = [Segment("system", "You are TinyMe. Write correct Python that satisfies the tests.", target=False),
                Segment("user", f"{text}", target=False),
                Segment("assistant", f"<|thought|>\nImplement the specification and check the provided tests.\n"
                                     f"<|code|>\n{code}\n<|endcode|>", target=True)]
        records.append(make_segment_record(segments=segs, category="code_gen", source="huggingface",
                                           source_id=f"mbpp/{row.get('row_idx')}", task_type="code_completion",
                                           template_id="hf/mbpp/v1", license=data.get("license", "CC-BY-4.0"),
                                           language="python", source_url=data.get("url", ""),
                                           retrieval_date=data.get("retrieval_date", ""),
                                           verified=bool(tests), verifier="published_tests",
                                           tests=[test_src] if test_src else []))
    prov = [{
        "source": "huggingface", "source_id": "mbpp", "dataset_or_repo_name": "google-research/mbpp",
        "url": data.get("url"), "license": data.get("license", "CC-BY-4.0"),
        "license_url": data.get("license_url", ""), "retrieval_date": data.get("retrieval_date", ""),
        "source_type": "code", "records": len(records),
        "preprocessing_performed": ["content_extraction", "segment_split", "python AST validation", "dedup"],
        "inclusion_status": "INCLUDED",
        "inclusion_reason": "permissively licensed code dataset with executable tests",
    }]
    return records, prov


def ingest_tinystories(limit: int = 40) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    data = _load_slice("tinystories_hf_slice.json")
    if not data:
        return [], []
    records: list[TrainingRecord] = []
    for row in data.get("rows", [])[:limit]:
        text = str(row.get("text", row.get("story", ""))).strip()
        if len(text) < 80:
            continue
        records.append(TrainingRecord(
            text=text, category="language", source="huggingface",
            source_id=f"tinystories/{row.get('row_idx')}", license=data.get("license", "CDLA-Sharing-1.0"),
            source_url=data.get("url", ""), retrieval_date=data.get("retrieval_date", ""),
            task_type="lm", language="en", template_id="hf/tinystories/v1",
            verified=True, verifier="published_dataset"))
    prov = [{
        "source": "huggingface", "source_id": "tinystories", "dataset_or_repo_name": "roneneldan/TinyStories",
        "url": data.get("url"), "license": data.get("license", "CDLA-Sharing-1.0"),
        "license_url": data.get("license_url", ""), "retrieval_date": data.get("retrieval_date", ""),
        "source_type": "text", "records": len(records),
        "preprocessing_performed": ["content_extraction", "quality_filter", "dedup"],
        "inclusion_status": "INCLUDED",
        "inclusion_reason": "synthetic but well-formed English prose for language grounding",
    }]
    return records, prov


# ---------------------------------------------------------------- assembly
def build_corpus(seed: int = 20261002, scale: float = 1.0,
                 synthetic_counts: dict[str, int] | None = None,
                 use_hf: bool = True, use_stdlib: bool = True,
                 stdlib_modules: int = 24, stdlib_files: int | None = None,
                 generator_set: str = "v2",
                 synthetic_counts_v3: dict[str, int] | None = None,
                 scale_v3: float = 1.0) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Assemble the raw corpus (records + provenance).

    ``generator_set`` selects the synthetic block:
      * ``"v2"`` — the generators that produced ``dataset_v3`` (kept so that
        dataset stays reproducible);
      * ``"v3"`` — the capability-scaled generators
        (:mod:`data_sources.synthetic_v3`) that add the transcription
        curriculum, direct-answer arithmetic/text edits and an order-of-magnitude
        larger tool-workflow block.  The v2 blocks are still assembled, so the
        general-purpose domains do not regress.
    """
    records: list[TrainingRecord] = []
    provenance: list[dict[str, Any]] = []

    if use_stdlib:
        # v1: curated documented functions from a fixed module list
        stdlib_recs, stdlib_prov = stdlib_adapter.ingest(max_per_module=max(4, stdlib_modules // 3))
        for rec in stdlib_recs:
            module = rec.source_id.split("/")[-1].split(".")[0]
            rec.template_id = f"stdlib/{module}"
            rec.family = f"stdlib/{module}"
        records.extend(stdlib_recs)
        provenance.extend(stack_provenance(stdlib_prov, "python_stdlib"))
        # v2: extended walk over the local stdlib (files/records budgeted)
        ext_recs, ext_prov, _counters = stdlib_v2.ingest_extended(
            max_files=stdlib_files or max(stdlib_modules * 8, 40), max_per_file=6)
        records.extend(ext_recs)
        provenance.extend(ext_prov)

    if use_hf:
        for fn in (ingest_gsm8k, ingest_mbpp, ingest_tinystories):
            recs, prov = fn()
            records.extend(recs)
            provenance.extend(prov)

    syn_recs, syn_prov = generate_corpus(seed=seed, counts=synthetic_counts, scale=scale)
    # synthetic families are already set via template_id
    if generator_set == "v3":
        # The v2 tool block is superseded by the v3 runtime-backed generator: its
        # search results are mocked (stale scores) and its "grounded" finals quote
        # the query echo rather than the fetched passage, which is exactly the
        # behaviour that made the model fail the independent tool suite.  Keeping
        # both blocks would supervise contradictory targets.
        dropped = [r for r in syn_recs if r.category == "tool_use"]
        syn_recs = [r for r in syn_recs if r.category != "tool_use"]
        logger.info("v3 build: dropped %d superseded v2 tool records", len(dropped))
    records.extend(syn_recs)
    provenance.extend(syn_prov)

    if generator_set == "v3":
        recs3, prov3 = generate_corpus_v3(seed=seed + 7, counts=synthetic_counts_v3,
                                          profile=synthetic_profile,
                                          scale=scale_v3)
        records.extend(recs3)
        provenance.extend(prov3)
    elif generator_set != "v2":
        raise ValueError(f"unknown generator_set {generator_set!r}")

    out = [r.to_dict() for r in records]
    logger.info("assembled %d raw records from %d provenance entries", len(out), len(provenance))
    return out, provenance


def stack_provenance(entries: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    for e in entries:
        e.setdefault("source", source)
    return entries


def corpus_statistics(records: list[dict[str, Any]]) -> dict[str, Any]:
    from collections import Counter

    return {
        "records": len(records),
        "chars": sum(len(r.get("text", "")) for r in records),
        "categories": dict(Counter(r.get("category") for r in records)),
        "sources": dict(Counter(r.get("source") for r in records)),
        "licenses": dict(Counter(r.get("license") for r in records)),
        "families": len({r.get("group_id") or r.get("template_id") for r in records}),
        "verified_records": sum(1 for r in records if r.get("verified")),
    }
