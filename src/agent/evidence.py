"""Evidence engine: grounded answers, no invented citations (audit §30).

Every fact the agent may cite is first recorded in an :class:`EvidenceStore`
with its source id, url and exact quote.  ``verify_citations`` then checks a
final answer against the store: URLs and quoted spans that cannot be traced back
are reported as *unsupported*, and an answer with unsupported citations cannot
be labelled grounded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

URL_RE = re.compile(r"https?://[^\s)\]\"'>]+")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,200})[\"”]")


@dataclass
class EvidenceItem:
    evidence_id: str
    source_id: str
    url: str
    title: str
    quote: str
    tool: str

    def to_dict(self) -> dict:
        return {"evidence_id": self.evidence_id, "source_id": self.source_id, "url": self.url,
                "title": self.title, "quote": self.quote, "tool": self.tool}


@dataclass
class EvidenceStore:
    items: list[EvidenceItem] = field(default_factory=list)

    def add_tool_result(self, tool: str, result) -> None:
        """Index every evidence record a tool returned."""
        for raw in getattr(result, "evidence", []) or []:
            item = EvidenceItem(
                evidence_id=f"E{len(self.items) + 1}", source_id=str(raw.get("source_id", "")),
                url=str(raw.get("url", "")), title=str(raw.get("title", "")),
                quote=str(raw.get("quote", ""))[:400], tool=tool)
            if item.source_id or item.url:
                self.items.append(item)

    # ------------------------------------------------------------------ checks
    def source_ids(self) -> set[str]:
        return {i.source_id for i in self.items}

    def urls(self) -> set[str]:
        return {i.url for i in self.items}

    def quotes(self) -> list[str]:
        return [i.quote for i in self.items]

    def verify(self, answer: str, *, require_support_for_claims: bool = True) -> dict:
        cited_urls = set(URL_RE.findall(answer))
        unsupported_urls = sorted(u for u in cited_urls if u not in self.urls())
        quoted = [q.strip() for q in QUOTE_RE.findall(answer)]
        unsupported_quotes = [q for q in quoted
                              if not any(q.lower()[:60] in ev.lower() for ev in self.quotes())]
        problems = []
        if unsupported_urls:
            problems.append(f"unsupported_urls:{unsupported_urls}")
        if unsupported_quotes:
            problems.append(f"unsupported_quotes:{len(unsupported_quotes)}")
        if require_support_for_claims and not self.items and (cited_urls or quoted):
            problems.append("citations_without_evidence")
        return {"grounded": not problems, "evidence_items": len(self.items),
                "cited_urls": sorted(cited_urls), "unsupported_urls": unsupported_urls,
                "unsupported_quotes": unsupported_quotes, "problems": problems}

    def cite(self, evidence_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def to_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self.items]}
