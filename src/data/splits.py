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

    assigned: dict[str, list[dict[str, Any]]] = {s: [] for s in SPLITS}

    # ---------------------------------------------------------------- stratified
    # Assign per *category*, so every split (most importantly the held-out test
    # and validation sets) contains every task family that exists in the corpus.
    # A plain global greedy split can starve whole domains — that actually
    # happened on the first dataset_v2 build, where `test` had no code_gen,
    # code_repair, tool_use or instruction records and those domains were
    # therefore unmeasurable.
    by_category: dict[str, list[str]] = defaultdict(list)
    for gid, items in groups.items():
        by_category[str(items[0].get("category", "unknown"))].append(gid)

    per_category: dict[str, dict[str, int]] = {}
    family_coverage: dict[str, dict[str, int]] = {}
    for category in sorted(by_category):
        gids = _stable_order(by_category[category], seed)
        total_records = sum(len(groups[g]) for g in gids)
        target_validation = int(round(total_records * ratios.get("validation", 0.1)))
        target_test = int(round(total_records * ratios.get("test", 0.1)))

        # ------------------------------------------------------- per *family*
        # A family is a template family (``family``/``template_id``).  A family
        # with several phrasings keeps one phrasing in validation and one in
        # test, so *every* measurable family is represented in the held-out
        # splits.  The earlier category-only loop put whole families into train
        # (1610 of 1993 families had no held-out instance at all - including
        # every multi-step tool family, which is why tool capability could not be
        # measured held-out).  Single-phrasing families are still assigned wholly
        # to one split, because the contamination gate counts a shared group as
        # contamination and splitting one would therefore be leakage.
        by_family: dict[str, list[str]] = defaultdict(list)
        for gid in gids:
            first = groups[gid][0]
            fam = str(first.get("family") or first.get("template_id") or gid)
            by_family[fam].append(gid)
        validation_gids: list[str] = []
        test_gids: list[str] = []
        train_gids: list[str] = []
        for fam in _stable_order(by_family, seed):
            fam_gids = _stable_order(by_family[fam], seed)
            n_fam_records = sum(len(groups[g]) for g in fam_gids)
            if len(fam_gids) >= 3:
                validation_gids.append(fam_gids[0])
                test_gids.append(fam_gids[1])
                train_gids.extend(fam_gids[2:])
                family_coverage[fam] = {"groups": len(fam_gids), "records": n_fam_records}
            else:
                key = int(hashlib.sha256(f"{seed}:{fam}".encode("utf-8")).hexdigest(), 16) % 10
                if key == 0 and n_fam_records >= 8 and \
                        sum(len(groups[g]) for g in validation_gids) < target_validation:
                    validation_gids.extend(fam_gids)
                elif key == 1 and n_fam_records >= 8 and \
                        sum(len(groups[g]) for g in test_gids) < target_test:
                    test_gids.extend(fam_gids)
                else:
                    train_gids.extend(fam_gids)
        # Deterministic coverage floor: a category must never end up with an
        # empty held-out split just because the hash order put every family in
        # train.  Whole groups are moved, so group integrity (and therefore the
        # contamination gate) is untouched.
        while not validation_gids and len(train_gids) >= 2:
            validation_gids.append(train_gids.pop(0))
        while not test_gids and len(train_gids) >= 2:
            test_gids.append(train_gids.pop(0))
        train_gids = [g for g in train_gids if g not in set(validation_gids) | set(test_gids)]
        for gid in validation_gids:
            for rec in groups[gid]:
                rec["split"] = "validation"
        for gid in test_gids:
            for rec in groups[gid]:
                rec["split"] = "test"
        for gid in train_gids:
            for rec in groups[gid]:
                rec["split"] = "train"
        per_category[category] = {
            "groups": len(gids), "records": total_records,
            "train": sum(len(groups[g]) for g in train_gids),
            "validation": sum(len(groups[g]) for g in validation_gids),
            "test": sum(len(groups[g]) for g in test_gids),
        }
        for gid in gids:
            assigned[groups[gid][0]["split"]].extend(groups[gid])

    for rec in preset:
        assigned[rec["split"]].append(rec)

    out = [r for s in SPLITS for r in assigned[s]]
    report = split_report(assigned, ratios)
    report["per_category"] = per_category
    report["family_coverage"] = {
        "families_with_held_out_instances": len(family_coverage),
        "families_assigned_wholly": len({str(groups[g][0].get("family") or groups[g][0].get("template_id") or g)
                                        for g in groups}) - len(family_coverage),
        "detail": family_coverage}
    report["category_coverage"] = {
        split: sorted({r.get("category", "unknown") for r in assigned.get(split, [])})
        for split in SPLITS}
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
    lines += ["", "## Category coverage per split", ""]
    for split in SPLITS:
        if report["counts"].get(split):
            lines.append(f"- **{split}**: {', '.join(report.get('category_coverage', {}).get(split, []))}")
    lines += ["", "## Contamination", "",
              f"- TRAIN/VALIDATION/TEST CONTAMINATION: "
              f"{'PASS' if report['contamination_free'] else 'FAIL'}"]
    for key, c in report["contamination"].items():
        lines.append(f"  - `{key}`: exact={c['exact_overlap']} normalized={c['normalized_overlap']} "
                     f"minhash={c['minhash_overlap']} code={c['code_shingle_overlap']} "
                     f"template={c['template_overlap']}")
    return "\n".join(lines) + "\n"
