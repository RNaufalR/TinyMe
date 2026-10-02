"""DEPRECATED (audit corrective action): the v1 14-stage pipeline.

The v1 flow this module implemented is **not** authoritative any more.  It
shuffled records, split train/eval as ``train[:32]``, tokenized over the whole
corpus and wrote memmap shards without a padding mask - exactly the defects the
corrective audit (2026-10-02) documents in ``docs/CORRECTIVE_AUDIT.md``.

The authoritative pipeline is
``scripts/prepare_data_v2.py`` built from
``src/data/{preprocess,records,quality_filter,dedup,splits,sequence,shard_writer}.py``.
Calling anything here raises immediately so a stale script can never silently
reproduce v1 data.
"""
from __future__ import annotations

from typing import Any

MESSAGE = (
    "src.data.pipeline is retired. Use scripts/prepare_data_v2.py "
    "(src/data/{preprocess,quality_filter,dedup,splits,sequence,shard_writer}.py). "
    "See docs/CORRECTIVE_AUDIT.md and docs/DATA_QUALITY_REPORT.md."
)


def _retired(*_args: Any, **_kwargs: Any) -> Any:
    raise RuntimeError(MESSAGE)


#: Every v1 entry point, kept only so imports fail loudly with the explanation.
STAGE_NAMES = ("stage_ingest", "stage_license_check", "stage_extract", "stage_quality_filter",
               "stage_classify", "stage_dedup", "stage_contamination", "stage_mix",
               "stage_tokenize", "stage_shards", "stage_manifest", "run_pipeline")
globals().update({name: _retired for name in STAGE_NAMES})

__all__ = list(STAGE_NAMES)
