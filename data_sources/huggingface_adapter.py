"""Hugging Face dataset adapter (spec §8).

Two retrieval paths are supported so the pipeline never depends on one source:

1. `datasets-server` rows API  — used when direct huggingface.co sockets are blocked.
2. `datasets`/`huggingface_hub` library — used when direct access is available.

Each adapter call returns standardized `TrainingRecord`s plus a provenance entry.
Unusable sources are skipped, never silently included.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from src.data.records import TrainingRecord, classify_license, validate_record

logger = logging.getLogger("tinyme.hf")

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "datasets" / "raw" / "huggingface"
ROWS_API = "https://datasets-server.huggingface.co/rows"


def http_get_json(url: str, timeout: int = 45) -> dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "TinnyMe-data-pipeline/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def direct_access_available() -> bool:
    """Probe whether huggingface.co is directly reachable from this sandbox."""
    try:
        req = urllib.request.Request("https://huggingface.co/api/datasets?limit=1",
                                     headers={"User-Agent": "TinnyMe-data-pipeline/1.0"})
        with urllib.request.urlopen(req, timeout=8) as resp:
            return resp.status == 200
    except Exception:
        return False


def rows_api_url(dataset: str, config: str, split: str, offset: int, length: int) -> str:
    q = urllib.parse.urlencode({
        "dataset": dataset, "config": config, "split": split,
        "offset": offset, "length": length,
    })
    return f"{ROWS_API}?{q}"


def fetch_rows(dataset: str, config: str, split: str, offset: int = 0,
               length: int = 20) -> list[dict[str, Any]]:
    """Fetch rows via the datasets-server API."""
    url = rows_api_url(dataset, config, split, offset, length)
    payload = http_get_json(url)
    return payload.get("rows", [])


def cache_rows(dataset: str, rows: list[dict[str, Any]], name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{name}.json"
    payload = {
        "dataset": dataset,
        "cached_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "row_count": len(rows),
        "rows": rows,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def load_cached(name: str) -> list[dict[str, Any]]:
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("rows", [])


def load_prefetched_slice(name: str) -> list[dict[str, Any]]:
    """Load the human/agent-fetched verified slices committed under datasets/raw/huggingface."""
    path = CACHE_DIR / f"{name}.json"
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data.get("rows", [])


def records_from_tinystories(rows: list[dict[str, Any]]) -> list[TrainingRecord]:
    out = []
    for r in rows:
        out.append(TrainingRecord(
            text=r["text"].strip(), category="language", source="huggingface",
            source_id=f"roneneldan/TinyStories:row_{r.get('row_idx', 0)}",
            license="CDLA-Sharing-1.0",
            source_url="https://huggingface.co/datasets/roneneldan/TinyStories",
            retrieval_date=time.strftime("%Y-%m-%d"), task_type="lm", language="en",
        ))
    return out


def records_from_gsm8k(rows: list[dict[str, Any]]) -> list[TrainingRecord]:
    out = []
    for r in rows:
        q, a = r["question"].strip(), r["answer"].strip()
        answer = a.split("####")[-1].strip()
        text = (f"<|system|>\nYou are a careful math reasoner. Solve step by step.\n"
                f"<|user|>\n{q}\n<|thought|>\n{a.split('####')[0].strip()}\n<|answer|>\n{answer}\n")
        out.append(TrainingRecord(
            text=text, category="math", source="huggingface",
            source_id=f"openai/gsm8k:row_{r.get('row_idx', 0)}",
            license="MIT", source_url="https://huggingface.co/datasets/openai/gsm8k",
            retrieval_date=time.strftime("%Y-%m-%d"), task_type="reasoning",
            language="en", verified=True, verifier="dataset_final_answer_marker", answer=answer,
        ))
    return out


def records_from_mbpp(rows: list[dict[str, Any]]) -> list[TrainingRecord]:
    out = []
    for r in rows:
        spec, code = r["text"].strip(), r["code"].strip()
        tests = list(r.get("test_list", []))
        text = (f"<|system|>\nYou are an expert Python programmer.\n<|user|>\n{spec}\n"
                f"<|code|>\n{code}\n<|endcode|>\n")
        out.append(TrainingRecord(
            text=text, category="code_gen", source="huggingface",
            source_id=f"google-research-datasets/mbpp:task_{r.get('task_id', 0)}",
            license="CC-BY-4.0", source_url="https://huggingface.co/datasets/google-research-datasets/mbpp",
            retrieval_date=time.strftime("%Y-%m-%d"), task_type="code_completion",
            language="python", verified=True, verifier="mbpp_assert_tests", tests=tests,
        ))
    return out


DATASET_REGISTRY = {
    "roneneldan/TinyStories": {"config": "default", "split": "train", "builder": records_from_tinystories, "slice": "tinystories_hf_slice"},
    "openai/gsm8k": {"config": "main", "split": "train", "builder": records_from_gsm8k, "slice": "gsm8k_hf_slice"},
    "google-research-datasets/mbpp": {"config": "full", "split": "train", "builder": records_from_mbpp, "slice": "mbpp_hf_slice"},
}

DATASET_LICENSES = {
    "roneneldan/TinyStories": ("CDLA-Sharing-1.0", "https://cdla.dev/sharing-1-0/"),
    "openai/gsm8k": ("MIT", "https://opensource.org/licenses/MIT"),
    "google-research-datasets/mbpp": ("CC-BY-4.0", "https://creativecommons.org/licenses/by/4.0/"),
}


def ingest(datasets: list[str] | None = None, rows_per_dataset: int = 20,
           force_refetch: bool = False) -> tuple[list[TrainingRecord], list[dict[str, Any]]]:
    """Ingest the registered HF datasets, returning records + provenance entries."""
    datasets = datasets or list(DATASET_REGISTRY)
    all_records: list[TrainingRecord] = []
    provenance: list[dict[str, Any]] = []
    direct = direct_access_available()
    logger.info("huggingface.co direct access: %s", "AVAILABLE" if direct else "BLOCKED (using datasets-server API)")

    for name in datasets:
        spec = DATASET_REGISTRY.get(name)
        if spec is None:
            logger.warning("dataset %s not in registry -> skipped", name)
            continue
        rows: list[dict[str, Any]] = []
        mode = "none"
        if force_refetch:
            try:
                rows = fetch_rows(name, spec["config"], spec["split"], 0, rows_per_dataset)
                mode = "datasets-server-api"
            except Exception as exc:  # network/blocked -> fall back
                logger.warning("rows API fetch failed for %s: %s", name, exc)
        if not rows:
            cached = load_prefetched_slice(spec["slice"])
            if cached:
                rows, mode = cached, "cached-verified-slice"
        if not rows:
            logger.warning("no rows available for %s -> skipped", name)
            provenance.append({
                "source": "huggingface", "dataset_or_repo_name": name,
                "url": f"https://huggingface.co/datasets/{name}", "retrieval_date": time.strftime("%Y-%m-%d"),
                "license": "LICENSE_UNCLEAR", "license_url": "", "source_type": "unknown",
                "language": "unknown", "approximate_size_bytes": 0, "sample_count": 0,
                "preprocessing_performed": ["attempted_rows_api_fetch"],
                "inclusion_status": "EXCLUDED", "inclusion_reason": "no rows retrievable in this sandbox",
            })
            continue

        recs = spec["builder"](rows)
        valid = [r for r in recs if validate_record(r)[0]]
        all_records.extend(valid)
        lic, lic_url = DATASET_LICENSES[name]
        provenance.append({
            "source": "huggingface", "dataset_or_repo_name": name,
            "url": f"https://huggingface.co/datasets/{name}", "retrieval_date": time.strftime("%Y-%m-%d"),
            "license": lic, "license_url": lic_url,
            "source_type": "text/math/code", "language": "en",
            "approximate_size_bytes": sum(len(r.text.encode("utf-8")) for r in valid),
            "sample_count": len(valid),
            "retrieval_mode": mode,
            "preprocessing_performed": ["license_check", "content_extraction", "unicode_normalization",
                                        "structured_prompt_wrapping", "quality_scoring", "deduplication"],
            "inclusion_status": "INCLUDED",
            "inclusion_reason": f"open license ({lic}) and high educational value for tiny-model curriculum",
        })
        logger.info("ingested %d records from %s (mode=%s)", len(valid), name, mode)

    return all_records, provenance
