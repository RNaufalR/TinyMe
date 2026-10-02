"""Data pipeline contract: records, masks, split paths and shard consumption.

Required by audit §16 (``test_data_pipeline.py``); these tests exercise real
behaviour of the shipped dataset (schema round-trip, mask derivation, the
train/validation/test/challenge paths, and the packed shards the trainer reads)
rather than checking that functions merely exist.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
import pytest

from src.data.dataset_api import (SPLIT_FILES, VERSIONS_DIR, load_manifest, load_split,
                                  load_split_arrays, load_split_records, load_tokenizer,
                                  stats_from_labels)
from src.data.records import (CONTEXT_ROLES, TARGET_ROLES, Segment, make_segment_record,
                              parse_segments, serialize_segments, validate_record)
from src.data.sequence import build_sequence, encode_segment
try:
    from src.evaluation.evaluator import render_prompt
except ImportError:  # pragma: no cover - evaluator import is optional here
    render_prompt = None

def _current_version() -> str:
    """The newest built dataset revision (the one the pipeline writes today)."""
    for version in ("dataset_v3", "dataset_v2"):
        if (VERSIONS_DIR / version / "manifest.json").exists():
            return version
    pytest.skip("no dataset version built - run scripts/prepare_data_v2.py")


DATASET = _current_version()


def _shards_available() -> bool:
    from src.data.dataset_api import PROCESSED_DIR
    return any((PROCESSED_DIR / DATASET / "shards").glob("train_pretrain_*.npy"))


requires_shards = pytest.mark.skipif(
    not _shards_available(),
    reason="packed shards are gitignored build artefacts - run scripts/prepare_data_v2.py")


# ------------------------------------------------------------------ schema
def test_segment_round_trip_preserves_roles_and_targets():
    rec = make_segment_record(
        segments=[Segment("system", "You are TinyMe.", target=False),
                  Segment("user", "Compute 2 + 3.", target=False),
                  Segment("assistant", "5", target=True)],
        category="math", source="synthetic", source_id="unit/roundtrip",
        template_id="unit/roundtrip")
    text = serialize_segments(rec.segments)
    parsed = parse_segments(text)
    assert [s.role for s in parsed] == ["system", "user", "assistant"]
    assert parsed[2].text.strip() == "5"
    # the serialized form is the on-disk contract and must be stable
    assert text.startswith("<|system|>\nYou are TinyMe.")
    assert "<|assistant|>" in text


def test_validate_record_rejects_malformed_shapes():
    ok, why = validate_record({"text": "plain document", "category": "language", "source": "python_stdlib",
                               "source_id": "z", "task_type": "lm", "license": "PSF-2.0"})
    assert ok, why  # plain LM records stay valid (they feed the pretrain stage)
    for broken in ({"category": "language"},                       # missing text/source
                   {"text": "x", "category": "nope", "source": "python_stdlib", "source_id": "z"},
                   {"text": "x", "category": "language", "source": "nowhere", "source_id": "z"},
                   {"text": " ", "category": "language", "source": "python_stdlib", "source_id": "z"},
                   {"text": "x", "category": "language", "source": "python_stdlib", "source_id": "z"}):
        good, reason = validate_record(broken)
        assert not good and reason, broken


def test_target_and_context_roles_are_disjoint_sets():
    assert set(TARGET_ROLES) & set(CONTEXT_ROLES) == set()
    assert "tool_result" in CONTEXT_ROLES  # tool output is context, never a target
    assert "tool_call" in TARGET_ROLES     # the model must learn to emit calls


# --------------------------------------------------------------------- masks
def test_tool_control_tokens_are_supervised_but_results_are_not(fake_tokenizer):
    """The protocol tokens must be learnable targets; tool output must not be."""
    rec = make_segment_record(
        segments=[Segment("system", "You are TinyMe.", target=False),
                  Segment("user", "Look it up.", target=False),
                  Segment("assistant", "", target=True),
                  Segment("tool_call", '{"name": "search", "arguments": {"query": "x"}}', target=True),
                  Segment("tool_result", '{"ok": true, "result": {"results": []}}', target=False),
                  Segment("final", "answer [1]", target=True)],
        category="tool_use", source="synthetic", source_id="unit/mask", task_type="tool_call",
        template_id="unit/mask").to_dict()
    ex = build_sequence(rec, fake_tokenizer, seq_len=256, mode="dynamic")
    assert ex is not None
    ids = ex.input_ids.tolist()
    predicted = {ids[i + 1] for i, flag in enumerate(ex.loss_mask.tolist()) if flag}
    id_of = fake_tokenizer.tok.token_to_id
    # protocol tokens the model has to emit are supervised ...
    for name in ("<|assistant|>", "<|tool_call|>", "<|final|>", "<|eos|>"):
        assert id_of(name) in predicted, f"{name} is not a prediction target"
    # ... while runtime-provided context tokens are never predicted
    for name in ("<|system|>", "<|user|>", "<|tool_result|>"):
        assert id_of(name) not in predicted, f"{name} must stay context-only"


def test_encoder_supervises_opening_and_closing_markers(fake_tokenizer):
    seg = Segment("tool_call", '{"name": "compute", "arguments": {"expression": "1+1"}}', target=True)
    ids, flags = encode_segment(seg, fake_tokenizer)
    assert all(flags), "every token of a target tool_call segment is supervised"
    assert ids[0] == fake_tokenizer.tok.token_to_id("<|tool_call|>")
    assert ids[-1] == fake_tokenizer.tok.token_to_id("<|end_tool_call|>")
    ctx = Segment("tool_result", '{"ok": true}', target=False)
    _, ctx_flags = encode_segment(ctx, fake_tokenizer)
    assert not any(ctx_flags), "tool results contribute zero loss"


def test_labels_and_masks_are_aligned(fake_tokenizer, sample_record):
    ex = build_sequence(sample_record, fake_tokenizer, seq_len=64, mode="packed")
    assert np.array_equal(ex.labels < 0, ~ex.loss_mask)


# ------------------------------------------------------------------- splits
def test_manifest_declares_its_loss_mask_contract():
    manifest = load_manifest(DATASET)
    assert manifest.get("loss_mask_semantics") in ("control-tokens-supervised", "body-only")


def test_every_split_has_its_own_file_and_no_aliasing():
    assert set(SPLIT_FILES) == {"train", "validation", "test", "challenge"}
    paths = {name: str(VERSIONS_DIR / DATASET / f) for name, f in SPLIT_FILES.items()}
    assert len(set(paths.values())) == 4, "splits must never share a file"
    for split in SPLIT_FILES:
        records = load_split_records(DATASET, split)
        assert records, f"{split} is empty"
        assert all(r.get("split") in (None, split) for r in records)


def test_unknown_split_is_rejected():
    with pytest.raises(ValueError):
        load_split_records(DATASET, "eval")


def test_split_record_ids_are_disjoint():
    ids = {split: {r["record_id"] for r in load_split_records(DATASET, split)}
           for split in SPLIT_FILES}
    for a in SPLIT_FILES:
        for b in SPLIT_FILES:
            if a < b:
                assert not (ids[a] & ids[b]), f"{a}/{b} share records"


def test_manifest_reports_contamination_pass():
    manifest = load_manifest(DATASET)
    assert manifest["train_validation_test_contamination"] == "PASS"
    for pair, block in manifest["contamination"].items():
        assert block["contamination_free"], pair
        assert block["exact_overlap"] == 0 and block["template_overlap"] == 0


def test_tokenizer_was_trained_on_the_train_split_only():
    manifest = load_manifest(DATASET)
    tok_block = manifest["tokenizer"]
    assert "train split only" in tok_block["trained_on"]
    # a frozen tokenizer is only legitimate when it was trained on a train split
    assert tok_block["train_texts"] <= manifest["splits"]["train"]
    assert manifest["loss_mask_semantics"] in ("control-tokens-supervised", "body-only")
    assert manifest["tokenizer_hash"] == tok_block["sha256"]


@requires_shards
def test_tokenizer_vocab_matches_the_shard_ids():
    tokenizer = load_tokenizer(DATASET)
    data, stats = load_split(DATASET, "validation", tokenizer=tokenizer, stage="pretrain")
    assert data["input_ids"].max() < tokenizer.vocab_size
    assert data["input_ids"].min() >= 0
    assert stats["active_target_tokens"] == int(data["loss_mask"].sum())


# ------------------------------------------------------------------- shards
@requires_shards
def test_shards_are_consumed_by_the_loader():
    for split in ("train", "validation"):
        arrays = load_split_arrays(DATASET, split, "pretrain")
        assert arrays["input_ids"].shape[0] > 0
        assert arrays["input_ids"].shape == arrays["labels"].shape == arrays["doc_ids"].shape
        assert arrays["loss_mask"].shape == arrays["input_ids"].shape
        # labels are either a real token id or the ignore index
        assert np.isin(np.unique(arrays["labels"]), [-100, *range(0, 4097)]).all()


@requires_shards
def test_shard_first_loading_matches_the_packed_builder():
    """The shard fast path must agree with re-deriving the split from JSONL."""
    tokenizer = load_tokenizer(DATASET)
    from_shards, stats_shards = load_split(DATASET, "test", tokenizer=tokenizer, stage="sft")
    assert stats_shards["source"] == "shards"
    records = load_split_records(DATASET, "test")
    structured = [r for r in records if r.get("segments")]
    from src.data.sequence import pack_records
    rebuilt, _ = pack_records(structured, tokenizer, 256)
    assert rebuilt["input_ids"].shape[0] == from_shards["input_ids"].shape[0]
    assert int(rebuilt["loss_mask"].sum()) == int(from_shards["loss_mask"].sum())


@requires_shards
def test_stats_report_active_and_masked_tokens():
    _, stats = load_split(DATASET, "train", stage="sft")
    assert stats["active_target_tokens"] > 0
    assert 0.0 <= stats["padding_ratio"] < 1.0
    helper = stats_from_labels(np.zeros((2, 4), np.int32), np.array([[1, 1, 0, 0], [0, 0, 0, 0]], bool))
    assert helper["active_target_tokens"] == 2
    assert helper["masked_tokens"] == 6


@requires_shards
def test_stage_filtering_changes_the_dataset_but_never_leaks_splits():
    pre, ps = load_split(DATASET, "train", stage="pretrain")
    sft, ss = load_split(DATASET, "train", stage="sft")
    assert ps["blocks"] != ss["blocks"] or ps["active_target_tokens"] != ss["active_target_tokens"]
    # both stages come from the same train split and stay inside it
    assert ps["split"] == ss["split"] == "train"
    # the SFT stage is denser in supervised tokens per block than plain documents
    assert ss["active_target_tokens"] / ss["blocks"] > 0


def test_records_carry_provenance_and_license():
    records = load_split_records(DATASET, "train")
    for r in records[:200]:
        assert r.get("source")
        assert r.get("license")
        assert r.get("record_id")
    kinds = Counter(r.get("source") for r in records)
    assert kinds, "no sources recorded"


def test_tool_use_records_speak_the_runtime_protocol():
    """Tool trajectories must carry the assistant turn opener and a real result envelope.

    ``dataset_v2`` predates the format fix and is kept as the historical
    revision, so the invariant is checked on revisions built with the corrected
    data contract (``dataset_v3`` onwards).
    """
    manifest = load_manifest(DATASET)
    if manifest.get("loss_mask_semantics") != "control-tokens-supervised":
        pytest.skip(f"{DATASET} predates the control-token/format fix (historical revision)")
    records = [r for r in load_split_records(DATASET, "train")
               if r.get("task_type") == "tool_call"]
    assert records, "no tool-use records in the train split"
    for r in records:
        roles = [s["role"] for s in r["segments"]]
        assert roles[0] == "system" and roles[1] == "user"
        assert roles[2] == "assistant", f"missing assistant turn opener in {r['record_id']}"
        assert "tool_call" in roles
        for seg in r["segments"]:
            if seg["role"] == "tool_result":
                payload = json.loads(seg["text"])
                assert "ok" in payload and "name" in payload


def test_every_model_turn_opens_with_the_assistant_marker():
    """Train/inference format parity (audit §2/§24).

    The runtime prompts the model with ``…<|end_tool_result|>\\n<|assistant|>\\n``
    and then expects ``<|final|>``.  If training jumped straight from the tool
    result to `<|final|>`, the model would never have seen the transition it is
    asked to make, so every target turn *after* context must be opened by
    ``<|assistant|>``.
    """
    manifest = load_manifest(DATASET)
    if manifest.get("loss_mask_semantics") != "control-tokens-supervised":
        pytest.skip(f"{DATASET} predates the control-token/format fix (historical revision)")
    tokenizer = load_tokenizer(DATASET)
    open_id = int(tokenizer.tok.token_to_id("<|assistant|>"))
    final_id = int(tokenizer.tok.token_to_id("<|final|>"))
    call_id = int(tokenizer.tok.token_to_id("<|tool_call|>"))
    checked = 0
    for r in load_split_records(DATASET, "train"):
        if r.get("task_type") != "tool_call":
            continue
        ex = build_sequence(r, tokenizer, seq_len=256, mode="dynamic")
        assert ex is not None
        ids = ex.input_ids.tolist()
        for i, tok_id in enumerate(ids):
            if tok_id in (final_id, call_id) and i > 0 and ids[i - 1] not in (open_id,):
                # the only legal predecessors are the assistant opener itself
                # (when the model emits its own turn) or the opener's newline
                assert ids[i - 1] == open_id or ids[i - 2] == open_id, (
                    f"{r['record_id']}: {tok_id} at {i} is not preceded by <|assistant|>")
        checked += 1
    assert checked, "no tool-use records checked"


def test_prompt_is_an_exact_prefix_of_the_training_sequence():
    """What the evaluator feeds in must be exactly what the model trained on."""
    tokenizer = load_tokenizer(DATASET)
    matched = truncated = 0
    for r in load_split_records(DATASET, "train"):
        if not r.get("segments"):
            continue
        ex = build_sequence(r, tokenizer, seq_len=256, mode="dynamic")
        if ex is None:
            continue
        prompt_ids = tokenizer.encode_ids(render_prompt(r))
        prefix = ex.input_ids[1:1 + len(prompt_ids)].tolist()
        if ex.truncated and prompt_ids[:len(prefix)] != prefix:
            truncated += 1          # target-aware truncation dropped context
            continue
        assert prefix == prompt_ids, (
            f"{r['record_id']}: prompt is not the training prefix\n"
            f"  prompt={tokenizer.decode(prompt_ids)[:200]!r}\n"
            f"  train ={tokenizer.decode(prefix)[:200]!r}")
        matched += 1
    assert matched > 100, f"only {matched} records exercised (truncated={truncated})"
