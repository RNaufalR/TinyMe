"""Evidence engine: grounded answers, no invented citations (audit §5/§30).

Every fact the agent may cite is first recorded in an :class:`EvidenceStore`
with its source id, url and exact quote.  ``verify`` then checks a final answer
against the store:

* every generated URL must resolve to a url a tool actually returned;
* every quoted span must appear in returned evidence;
* every bracketed citation id (``[S7]``, ``[calc-1a2b3c4d]``, ``[run-…]``) must
  resolve to a source id or a citation id the runtime issued.

An answer with any unresolvable citation can never be labelled grounded — a
plausible-looking URL or id earns nothing on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

URL_RE = re.compile(r"""https?://[^\s)\]"'>]+""")
QUOTE_RE = re.compile(r"[\"“]([^\"”]{12,200})[\"”]")
#: Citation ids are bracketed tokens that contain at least one letter and one
#: digit (``[S7]``, ``[S99]``, ``[calc-1a2b3c4d]``, ``[run-deadbeef]``).  Plain
#: numbers (``[1]``) and bare words are not treated as citations.
CITATION_ID_RE = re.compile(r"\[([A-Za-z][A-Za-z0-9_-]*\d[A-Za-z0-9_-]*)\]")
#: Payload keys whose value is a citable id.
CITABLE_KEYS = ("citation", "source_id", "evidence_id")


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


def _collect_citable_ids(value, out: set[str]) -> None:
    """Walk a tool payload and collect every id the runtime is willing to vouch for."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in CITABLE_KEYS and isinstance(item, (str, int)) and str(item):
                out.add(str(item))
            else:
                _collect_citable_ids(item, out)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _collect_citable_ids(item, out)


@dataclass
class EvidenceStore:
    items: list[EvidenceItem] = field(default_factory=list)
    #: ids (``calc-…``/``run-…``) issued by tools that do not return evidence
    #: records, e.g. the exact-arithmetic and sandbox tools
    citation_ids: set[str] = field(default_factory=set)

    def add_tool_result(self, tool: str, result) -> None:
        """Index every evidence record a tool returned, plus its citation ids."""
        for raw in getattr(result, "evidence", []) or []:
            item = EvidenceItem(
                evidence_id=f"E{len(self.items) + 1}", source_id=str(raw.get("source_id", "")),
                url=str(raw.get("url", "")), title=str(raw.get("title", "")),
                quote=str(raw.get("quote", ""))[:400], tool=tool)
            if item.source_id or item.url:
                self.items.append(item)
        _collect_citable_ids(getattr(result, "payload", {}) or {}, self.citation_ids)

    # ------------------------------------------------------------------ checks
    def source_ids(self) -> set[str]:
        return {i.source_id for i in self.items}

    def urls(self) -> set[str]:
        return {i.url for i in self.items}

    def quotes(self) -> list[str]:
        return [i.quote for i in self.items]

    def known_ids(self) -> set[str]:
        """Every id a citation may legitimately reference."""
        return self.source_ids() | self.citation_ids | {i.evidence_id for i in self.items}

    def verify(self, answer: str, *, require_support_for_claims: bool = True) -> dict:
        cited_urls = set(URL_RE.findall(answer))
        unsupported_urls = sorted(u for u in cited_urls if u not in self.urls())
        quoted = [q.strip() for q in QUOTE_RE.findall(answer)]
        unsupported_quotes = [q for q in quoted
                              if not any(q.lower()[:60] in ev.lower() for ev in self.quotes())]
        cited_ids = set(CITATION_ID_RE.findall(answer))
        # ``[E1]`` refers to the store's own evidence numbering, which is a
        # runtime id too; everything else must have been issued by a tool.
        unsupported_ids = sorted(i for i in cited_ids if i not in self.known_ids())
        problems = []
        if unsupported_urls:
            problems.append(f"unsupported_urls:{unsupported_urls}")
        if unsupported_quotes:
            problems.append(f"unsupported_quotes:{len(unsupported_quotes)}")
        if unsupported_ids:
            problems.append(f"unsupported_citations:{unsupported_ids}")
        if require_support_for_claims and not self.items and (cited_urls or quoted or cited_ids):
            if not (cited_ids and cited_ids <= self.known_ids()):
                problems.append("citations_without_evidence")
        return {"grounded": not problems, "evidence_items": len(self.items),
                "cited_urls": sorted(cited_urls), "unsupported_urls": unsupported_urls,
                "cited_ids": sorted(cited_ids), "unsupported_citations": unsupported_ids,
                "unsupported_quotes": unsupported_quotes, "problems": problems}

    def cite(self, evidence_id: str) -> EvidenceItem | None:
        for item in self.items:
            if item.evidence_id == evidence_id:
                return item
        return None

    def to_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self.items],
                "citation_ids": sorted(self.citation_ids)}
