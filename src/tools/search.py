"""Provider-agnostic search (audit §30).

Three provider kinds share one interface:

* ``local`` — BM25 over the shipped retrieval index; always available, offline,
  fully deterministic (the default, and the one used in every measured run here);
* ``http`` — a configurable JSON endpoint (outbound network is **disabled** in
  this environment, so it is reported as ``unavailable`` rather than faked);
* ``none`` — no provider configured.

The tool never invents results: an empty or unavailable provider returns an
explicitly empty result list plus the provider status.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Protocol

from .context import ToolContext


class SearchProvider(Protocol):
    name: str

    def available(self) -> tuple[bool, str]:
        ...

    def query(self, query: str, k: int) -> list[dict]:
        ...


@dataclass
class LocalIndexProvider:
    index: Any
    name: str = "local-bm25"

    def available(self) -> tuple[bool, str]:
        if self.index is None or len(self.index) == 0:
            return False, "no retrieval index built (run scripts/build_retrieval_index.py)"
        return True, "ok"

    def query(self, query: str, k: int) -> list[dict]:
        return self.index.search(query, k=k)


@dataclass
class HttpJsonProvider:
    """Generic endpoint adapter: ``GET {url}?q=...`` returning ``{"results": [...]}``."""

    url_template: str
    name: str = "http-json"
    timeout_s: float = 6.0
    api_key_env: str | None = None
    headers: dict[str, str] = field(default_factory=dict)

    def available(self) -> tuple[bool, str]:
        if not self.url_template:
            return False, "no endpoint configured"
        if os.environ.get("TINYME_ALLOW_NETWORK", "0") != "1":
            return False, "network disabled (set TINYME_ALLOW_NETWORK=1 to enable the http provider)"
        return True, "ok"

    def query(self, query: str, k: int) -> list[dict]:
        url = self.url_template.replace("{query}", urllib.parse.quote_plus(query)).replace("{k}", str(k))
        headers = dict(self.headers)
        if self.api_key_env and os.environ.get(self.api_key_env):
            headers["Authorization"] = f"Bearer {os.environ[self.api_key_env]}"
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:  # noqa: S310 - explicit provider
            payload = json.loads(resp.read().decode("utf-8", "replace"))
        return list(payload.get("results", []))[:k]


def resolve_provider(ctx: ToolContext) -> tuple[SearchProvider | None, str]:
    """Pick the best available provider (local first; http only if configured)."""
    candidates: list[SearchProvider] = []
    template = os.environ.get("TINYME_SEARCH_ENDPOINT", "")
    if template:
        candidates.append(HttpJsonProvider(url_template=template,
                                           api_key_env=os.environ.get("TINYME_SEARCH_KEY_ENV", "") or None))
    candidates.append(LocalIndexProvider(index=ctx.index))
    reasons = []
    for provider in candidates:
        ok, why = provider.available()
        reasons.append(f"{provider.name}:{why}")
        if ok:
            return provider, "; ".join(reasons)
    return None, "; ".join(reasons)


def search(ctx: ToolContext, query: str, k: int = 3) -> dict:
    """Tool entrypoint. Returns ranked, citation-ready passages."""
    provider, status = resolve_provider(ctx)
    if provider is None:
        return {"ok": False, "error": f"no_search_provider_available ({status})", "query": query,
                "provider": "unavailable", "results": []}
    results = provider.query(query, k)
    evidence = [{"evidence_id": f"E{idx + 1}", "source_id": r.get("source_id", ""),
                 "url": r.get("url", ""), "title": r.get("title", ""),
                 "quote": r.get("snippet", "")[:240], "score": r.get("score", 0.0)}
                for idx, r in enumerate(results) if r.get("snippet")]
    ctx.note("search", query=query, hits=len(results), provider=provider.name)
    return {"ok": True, "query": query, "provider": provider.name, "provider_status": status,
            "results": results, "evidence": evidence}
