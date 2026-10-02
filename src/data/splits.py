"""Deterministic group-aware train/validation/test splits (corrective audit P0-05).

Row-level random splitting is not acceptable for synthetic data: paraphrases of
the same template land on both sides and the "held-out" number is meaningless.
This module splits by ``group_id`` (template family / generator family / source
family), so an entire family is either train, validation or test.

The split is deterministic given ``(seed, dataset fingerprint)`` and never
depends on the order in which records were produced.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from typing import Any, Iterable

from .dedup import contamination_report

SPLITS = ("train", "validation", "test", "challenge")
DEFAULT_RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}


def _group_key(rec: dict[str, Any]) -> str:
    return str(rec.get("group_id") or rec.get("template_id") or rec.get("family") or rec.get("record_id") or "")


def _stable_order(items: Iterable[str], seed: int) -> list[str]:
    return sorted(items, key=lambda g: hashlib.sha256(f"{seed}:{g}".encode("utf-8")).hexdigest())


def assign_splits(records: list[dict[str, Any]], seed: int = 1234,
                  ratios: dict[str, float] | None = None,
                  respect_preset: bool = True) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Assign ``split`` to every record by group and return ``(records, report)``."""
    ratios = ratios or DEFAULT_RATIOS
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    preset: list[dict[str, Any]] = []
    for rec in records:
        if respect_preset and rec.get("split") in SPLITS:
            preset.append(rec)
            continue
        groups[_group_key(rec)].append(rec)

    gids = _stable_order(groups.keys(), seed)
    total = sum(len(v) for v in groups.values()) or 1
    targets = {s: max(1, int(round(total * ratios.get(s, 0.0)))) for s in ("train", "validation", "test")}
    assigned: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}
    for gid in gids:
        items = groups[gid]
        # greedy: place the family where the current deficit is largest
        best = max(("train", "validation", "test"),
                   key=lambda s: targets[s] - len(assigned[s]))
        assigned[best].extend(items)
        for rec in items:
            rec["split"] = best
    for rec in preset:
        assigned[rec["split"]].append(rec)

    out = [r for s in SPLITS for r in assigned[s]]
    report = split_report(assigned, ratios)
    return out, report


def split_report(assigned: dict[str, list[dict[str, Any]]],
                 ratios: dict[str, float] | None = None) -> dict[str, Any]:
    report: dict[str, Any] = {"counts": {}, "groups": {}, "categories": {}, "contamination": {}}
    for split in SPLITS:
        recs = assigned.get(split, [])
        report["counts"][split] = len(recs)
        report["groups"][split] = len({_group_key(r) for r in recs})
        report["categories"][split] = dict(Counter(r.get("category", "unknown") for r in recs))
    train = assigned.get("train", [])
    for other in ("validation", "test", "challenge"):
        recs = assigned.get(other, [])
        if recs:
            report["contamination"][f"train_vs_{other}"] = contamination_report(train, recs)
    validation = assigned.get("validation", [])
    for other in ("test", "challenge"):
        recs = assigned.get(other, [])
        if recs and validation:
            report["contamination"][f"validation_vs_{other}"] = contamination_report(validation, recs)
    report["contamination_free"] = all(
        c.get("contamination_free", True) for c in report["contamination"].values())
    report["ratios_requested"] = ratios or DEFAULT_RATIOS
    return report


def dataset_fingerprint(records: list[dict[str, Any]], extra: dict[str, Any] | None = None) -> str:
    """Stable fingerprint of the dataset content (record ids + splits)."""
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: str(r.get("record_id", ""))):
        h.update(str(rec.get("record_id", "")).encode())
        h.update(str(rec.get("split", "")).encode())
        h.update(str(rec.get("text", ""))[:512].encode("utf-8", "ignore"))
    if extra:
        h.update(json.dumps(extra, sort_keys=True, default=str).encode())
    return h.hexdigest()


def split_summary_markdown(report: dict[str, Any]) -> str:
    lines = ["# SPLIT SUMMARY", ""]
    for split in SPLITS:
        if report["counts"].get(split):
            lines.append(f"- **{split}**: {report['counts'][split]} records / "
                         f"{report['groups'][split]} template families")
    lines += ["", "## Category distribution", "", "| split | " +
              " | ".join(sorted({c for v in report["categories"].values() for c in v})) + " |",
              "| :--- | " + " | ".join(":---:" for _ in sorted({c for v in report["categories"].values() for c in v})) + " |"]
    cats = sorted({c for v in report["categories"].values() for c in v})
    for split in SPLITS:
        if not report["counts"].get(split):
            continue
        row = [str(report["categories"][split].get(c, 0)) for c in cats]
        lines.append(f"| {split} | " + " | ".join(row) + " |")
    lines += ["", "## Contamination", "",
              f"- TRAIN/VALIDATION/TEST CONTAMINATION: "
              f"{'PASS' if report['contamination_free'] else 'FAIL'}"]
    for key, c in report["contamination"].items():
        lines.append(f"  - `{key}`: exact={c['exact_overlap']} normalized={c['normalized_overlap']} "
                     f"minhash={c['minhash_overlap']} code={c['code_shingle_overlap']} "
                     f"template={c['template_overlap']}")
    return "\n".join(lines) + "\n"
