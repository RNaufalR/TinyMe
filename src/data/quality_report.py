"""Render DATA_QUALITY_REPORT.md from the pipeline manifest (spec §13)."""
from __future__ import annotations

from pathlib import Path
from typing import Any


def write_quality_report(manifest: dict[str, Any], path: str | Path) -> None:
    stats = manifest.get("stage_stats", {})
    cats = manifest.get("categories", {})
    sources = manifest.get("sources", {})
    licenses = manifest.get("licenses", {})
    total = manifest.get("train_samples", 0)

    def pct(n: int) -> str:
        return f"{100.0 * n / max(total, 1):.1f}%"

    code_total = sum(v for k, v in cats.items()
                     if k in ("programming", "code_gen", "code_repair", "code_explain"))
    text_total = sum(v for k, v in cats.items() if k in ("language", "instruction"))
    reasoning_total = sum(v for k, v in cats.items() if k in ("math", "logic", "algorithm"))

    ingestion = stats.get("ingestion", {})
    lic = stats.get("license_check", {})
    qf = stats.get("quality_filter", {})
    dd = stats.get("dedup", {})
    contam = stats.get("contamination", {})
    tok = stats.get("tokenization", {})

    lines = [
        "# DATA QUALITY REPORT",
        "",
        f"- **Dataset version:** `{manifest.get('dataset_version')}`",
        f"- **Generated:** {manifest.get('created')}",
        f"- **Train samples:** {total}",
        f"- **Eval samples:** {manifest.get('eval_samples')}",
        f"- **Verified samples:** {manifest.get('verified_samples')} "
        f"({pct(manifest.get('verified_samples', 0))} of train)",
        "",
        "## 1. Funnel: samples before / after each stage",
        "",
        "| Stage | Input | Output | Removed |",
        "| :--- | ---: | ---: | ---: |",
        f"| Ingestion + schema validation | {ingestion.get('input', 0)} | {ingestion.get('valid', 0)} "
        f"| {ingestion.get('input', 0) - ingestion.get('valid', 0)} |",
        f"| License check | {ingestion.get('valid', 0)} | {lic.get('kept', 0)} "
        f"| {ingestion.get('valid', 0) - lic.get('kept', 0)} |",
        f"| Quality / safety / language filter | {lic.get('kept', 0)} | {qf.get('kept', 0)} "
        f"| {lic.get('kept', 0) - qf.get('kept', 0)} |",
        f"| Deduplication (4 levels) | {qf.get('kept', 0)} | {dd.get('kept', 0)} "
        f"| {qf.get('kept', 0) - dd.get('kept', 0)} |",
        f"| Train/eval split | {dd.get('kept', 0)} | {total} + {manifest.get('eval_samples')} eval | — |",
        "",
        "## 2. Duplicate statistics",
        "",
        f"- Exact hash duplicates: **{dd.get('exact_duplicates', 0)}**",
        f"- Normalized hash duplicates: **{dd.get('normalized_duplicates', 0)}**",
        f"- Document-similarity duplicates: **{dd.get('similar_duplicates', 0)}**",
        f"- Code-similarity duplicates: **{dd.get('code_similar_duplicates', 0)}**",
        f"- Total duplicate percentage: **{dd.get('duplicate_percentage', 0)}%**",
        "",
        "## 3. Contamination check (train vs evaluation)",
        "",
        f"- Evaluation samples: **{contam.get('eval_samples', 0)}**",
        f"- Exact overlap: **{contam.get('exact_overlap', 0)}**",
        f"- Normalized overlap: **{contam.get('normalized_overlap', 0)}**",
        f"- Code-shingle overlap: **{contam.get('code_shingle_overlap', 0)}**",
        f"- **Contamination free: {contam.get('contamination_free', False)}**",
        "",
        "## 4. Corpus composition",
        "",
        "| Category | Samples | Share |",
        "| :--- | ---: | ---: |",
    ]
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} | {pct(v)} |")

    lines += [
        "",
        f"- **Code share:** {code_total} samples ({pct(code_total)})",
        f"- **Text share:** {text_total} samples ({pct(text_total)})",
        f"- **Reasoning share:** {reasoning_total} samples ({pct(reasoning_total)})",
        "",
        "## 5. Source and license distribution",
        "",
        "| Source | Samples |",
        "| :--- | ---: |",
    ]
    for k, v in sorted(sources.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines += ["", "| License | Samples |", "| :--- | ---: |"]
    for k, v in sorted(licenses.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")

    lines += [
        "",
        "## 6. Tokenization",
        "",
        f"- Tokenizer: `{tok.get('tokenizer')}`",
        f"- Total tokens: **{tok.get('total_tokens', 0):,}**",
        f"- Average tokens per sample: **{tok.get('avg_tokens_per_sample', 0)}**",
        "",
        "## 7. Provenance",
        "",
        "Full provenance is recorded in `docs/DATA_PROVENANCE.json` and `DATA_PROVENANCE.json`.",
        "Sources with unclear or copyleft licenses are excluded and explicitly marked.",
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")
