"""Passage fetch by source id — network-free (audit §30).

In this environment outbound HTTP is firewall-blocked, so ``fetch`` resolves
ids against the shipped retrieval index only and *says so*: the payload reports
``provider="local-index"``.  A URL-shaped argument is rejected rather than
silently attempted.
"""
from __future__ import annotations

from .context import ToolContext

MAX_CHARS = 4000


def fetch(ctx: ToolContext, source_id: str) -> dict:
    if "://" in source_id:
        return {"ok": False, "error": "network_fetch_disabled: pass a source_id from search results, not a URL",
                "source_id": source_id}
    index = ctx.index
    if index is None:
        return {"ok": False, "error": "no_retrieval_index", "source_id": source_id}
    doc = index.get(source_id)
    if doc is None:
        return {"ok": False, "error": f"unknown_source_id:{source_id}", "source_id": source_id}
    text = doc.text[:MAX_CHARS]
    ctx.note("fetch", source_id=source_id, chars=len(text))
    return {"ok": True, "source_id": doc.source_id, "title": doc.title, "url": doc.url,
            "source": doc.source, "license": doc.license, "text": text,
            "truncated": len(doc.text) > MAX_CHARS, "provider": "local-index",
            "evidence": [{"evidence_id": "E1", "source_id": doc.source_id, "url": doc.url,
                          "title": doc.title, "quote": text[:240]}]}
