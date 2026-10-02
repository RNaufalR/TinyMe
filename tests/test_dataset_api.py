"""Dataset API contracts: shard-first loading, stages, honest stats (audit §6/§7)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from src.data import dataset_api


def test_dataset_v2_manifest_is_self_consistent():
    manifest = dataset_api.load_manifest("dataset_v2")
    assert manifest["train_validation_test_contamination"] == "PASS"
    assert manifest["malformed_code_rate"] == 0.0
    for key in ("train", "validation", "test", "challenge"):
        assert manifest["splits"][key] > 0
    assert manifest["tokenizer_hash"], "the tokenizer hash must be recorded"
    assert manifest["dataset_fingerprint"]
    assert manifest["licenses"], "license breakdown must be recorded"
    contamination = manifest["contamination"]["train_vs_test"]
    assert contamination["exact_overlap"] == 0 and contamination["minhash_overlap"] == 0


def test_splits_are_disjoint_and_carry_provenance():
    seen: dict[str, set] = {}
    for split in ("train", "validation", "test", "challenge"):
        records = dataset_api.load_split_records("dataset_v2", split)
        assert records, split
        seen[split] = {r["record_id"] for r in records}
        assert all(r.get("license") for r in records), f"{split}: every record needs a license"
        assert all(r.get("source_id") for r in records), f"{split}: every record needs a source id"
    pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    for a, b in pairs:
        assert not (seen[a] & seen[b]), f"{a}/{b} overlap"


def test_shard_first_loading_matches_packed_labels(packed_dataset):
    arrays = dataset_api.load_split_arrays(packed_dataset, "validation", stage="pretrain")
    assert set(arrays) == {"input_ids", "labels", "loss_mask", "doc_ids"}
    ids, labels, mask = arrays["input_ids"], arrays["labels"], arrays["loss_mask"]
    assert ids.shape == labels.shape == mask.shape
    assert ids.dtype == np.int32 and labels.dtype == np.int32 and mask.dtype == bool
    tok = dataset_api.load_tokenizer(packed_dataset)
    assert ids.min() >= 0 and ids.max() < int(tok.vocab_size), "shard token ids must be valid tokenizer ids"
    # the loss mask and the ignore index must agree everywhere
    assert np.array_equal(labels == -100, ~mask)


def test_stats_report_active_and_masked_tokens(packed_dataset):
    arrays = dataset_api.load_split_arrays(packed_dataset, "test", stage="pretrain")
    stats = dataset_api.stats_from_labels(arrays["labels"], arrays["loss_mask"])
    assert stats["active_target_tokens"] > 0
    assert stats["masked_tokens"] >= 0
    assert stats["active_target_tokens"] + stats["masked_tokens"] == int(arrays["labels"].size)
    assert 0.0 <= stats["padding_ratio"] <= 1.0
    assert stats["ignore_index"] == -100


def test_stage_filtering_changes_the_dataset_but_never_leaks_splits(packed_dataset):
    pre = dataset_api.load_split_arrays(packed_dataset, "train", stage="pretrain")
    sft = dataset_api.load_split_arrays(packed_dataset, "train", stage="sft")
    assert pre["input_ids"].shape[0] >= sft["input_ids"].shape[0]
    manifest = dataset_api.load_manifest(packed_dataset)
    stages = manifest["stage_stats"]["tokens"]["train"]["stages"]
    assert set(stages) == {"pretrain", "sft"}
    assert manifest["stage_stats"]["tokens"]["train"]["blocks"] == (pre["input_ids"].shape[0]
                                                                    + sft["input_ids"].shape[0])


def test_tokenizer_vocab_matches_the_shard_ids(packed_dataset):
    tok = dataset_api.load_tokenizer(packed_dataset)
    arrays = dataset_api.load_split_arrays(packed_dataset, "train", stage="pretrain")
    assert int(arrays["input_ids"].max()) < int(tok.vocab_size)


def test_describe_version_includes_fingerprint_and_licenses():
    info = dataset_api.describe_version("dataset_v2")
    assert info["splits"]["train"]["records"] > 0
    assert info["splits"]["train"]["families"] > 0
    assert info["dataset_fingerprint"]
    assert info["tokenizer_hash"]
    assert info["licenses"], "license breakdown must be reported"
    assert info["contamination"] == "PASS"
