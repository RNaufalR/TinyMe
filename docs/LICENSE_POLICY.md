# LEGAL, LICENSE & DATA PROVENANCE POLICY (`LICENSE_POLICY.md`)

**Date:** 2026-10-01  
**Status:** `VERIFIED`  
**Authoritative Reference:** `TINY_AI_TRAINING_LAB.md` §4, §5, §6, §7, §8

---

## 1. Core Legal & Ethical Principle

TinnyMe strictly prohibits indiscriminate scraping or training on arbitrary copyrighted material without clear legal permission. Every external dataset, repository, documentation source, and synthetic generator must pass automated license verification and provenance logging before entering the training corpus.

---

## 2. License Classification & Filtering Rules

### A. Permitted Licenses (`INCLUDED`)
Sources bearing any of the following explicit SPDX / standard open licenses are permitted for ingestion into the default training and evaluation corpora:
1. **Public Domain / Dedication**: `CC0-1.0`, `Unlicense`, `0BSD`, `Public-Domain`
2. **Permissive Software & Code Licenses**: `MIT`, `Apache-2.0`, `BSD-2-Clause`, `BSD-3-Clause`, `ISC`, `PSF-2.0` (Python Software Foundation License v2)
3. **Open Data & ML Research Licenses**: `CDLA-Sharing-1.0`, `CDLA-Permissive-2.0`, `CC-BY-4.0`, `CC-BY-SA-4.0` (with attribution preserved in `DATA_PROVENANCE.json`), `ODC-By`
4. **First-Party / Programmatically Verified Synthetic Data**: `MIT` / `Synthetic-Verified` (generated deterministically by in-repo algorithms and verified via execution/symbolic checkers)

### B. Unclear or Missing Licenses (`LICENSE_UNCLEAR` — EXCLUDED)
- Any external repository, dataset, or webpage whose license metadata is `null`, `"unknown"`, `"other"`, `"custom"`, or missing an explicit license declaration MUST be flagged with:
  - `"license": "LICENSE_UNCLEAR"`
  - `"inclusion_status": "EXCLUDED"`
- `LICENSE_UNCLEAR` sources are **automatically blocked** by the license gate (`src/data/license_filter.py`) from entering the training shards, while still being recorded in `DATA_PROVENANCE.json` for audit transparency.

### C. Restrictive / Copyleft / Non-Commercial Licenses (`EXCLUDED`)
- Sources with `CC-BY-NC-*`, `GPL-3.0`, `AGPL-3.0`, or proprietary terms are excluded from the default training corpus to guarantee that the resulting `<50 MB` model package in `release/` can be freely redistributed under the MIT License.

---

## 3. Mandatory Provenance Schema (`DATA_PROVENANCE.json`)

Every inspected source (both `INCLUDED` and `EXCLUDED`) is recorded in `docs/DATA_PROVENANCE.json` (and mirrored at root `DATA_PROVENANCE.json`) with the following mandatory fields:

```json
{
  "source_id": "unique-identifier",
  "source": "huggingface | github | python_stdlib | synthetic",
  "dataset_or_repo_name": "owner/name",
  "url": "https://...",
  "retrieval_date": "2026-10-01",
  "license": "MIT | Apache-2.0 | PSF-2.0 | CC-BY-4.0 | CDLA-Sharing-1.0 | LICENSE_UNCLEAR",
  "license_url": "https://...",
  "source_type": "code | text | math | reasoning | instruction | synthetic",
  "language": "en | python | javascript | multi",
  "approximate_size_bytes": 0,
  "sample_count": 0,
  "preprocessing_performed": [
    "license_check",
    "content_extraction",
    "unicode_normalization",
    "secret_redaction",
    "quality_scoring",
    "deduplication"
  ],
  "inclusion_status": "INCLUDED | EXCLUDED",
  "inclusion_reason": "Explicit explanation of why the source was included or excluded"
}
```

---

## 4. Safety, Privacy & Secret Filtering

Even within permissively licensed code and text, the ingestion pipeline enforces automated sanitization (`src/data/quality_filter.py`):
1. **Secret & Credential Redaction/Rejection**: Rejects or redacts files matching patterns for API keys (`ghp_`, `sk-`, `AKIA`, `AIza`), private keys (`-----BEGIN.*PRIVATE KEY-----`), tokens, or hardcoded passwords.
2. **PII Filtering**: Strips email addresses and IPv4/IPv6 literals from code comments and prose.
3. **Binary / Minified / Generated Code Rejection**: Rejects binary blobs, lockfiles, minified JS/CSS (line length > 1000 chars or average line > 200 chars), and auto-generated protobuf/schema dumps.
4. **Malicious Payload Filter**: Rejects scripts containing obfuscated shell/eval payloads (`eval(base64...)`, destructive filesystem commands, network reverse shells).
