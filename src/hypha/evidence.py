"""Verification layer: independently check a proposed A-C link against the
real literature / web, attach citations, and assign a verdict.

The discovery engine *proposes* novel links from OpenAlex concept co-occurrence.
That signal is noisy, so before a hypothesis is trusted we run an independent
**evidence search** for the A-C pair and ask: how much does the literature
already discuss these two together?

* **open**        - essentially no direct co-mention -> genuinely undiscovered.
* **emerging**    - a handful of papers -> early signal, worth pursuing.
* **established** - widely co-studied -> not novel; demote / flag.

Providers (pluggable, BYOK):

* ``OpenAlexEvidenceProvider`` - free, no key (strict title/abstract AND count).
* ``ParallelEvidenceProvider`` - Parallel Search API (``PARALLEL_API_KEY``).
* ``PaperclipEvidenceProvider`` - GXL Paperclip full text + trials + FDA
  (``PAPERCLIP_API_KEY`` + ``gxl_paperclip``); experimental.
* ``FixtureEvidenceProvider``  - offline, deterministic (for tests/demo).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Optional, Protocol

import httpx

from hypha.models import EvidenceItem, Hypothesis
from hypha.sources.openalex import OpenAlexSource


@dataclass
class EvidenceResult:
    direct_hits: int  # co-mention count; -1 means "unknown"
    items: list[EvidenceItem] = field(default_factory=list)
    provider: str = "none"
    query: str = ""


# --------------------------------------------------------------------------- #
#  Providers
# --------------------------------------------------------------------------- #
class EvidenceProvider(Protocol):
    name: str

    def gather(self, a_name: str, c_name: str, max_items: int = 4) -> EvidenceResult:
        ...


def _reconstruct_abstract(inv: Optional[dict], limit: int = 240) -> str:
    """Rebuild an abstract from OpenAlex's inverted index, truncated."""
    if not inv:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inv.items():
        for i in idxs:
            positions.append((i, word))
    positions.sort()
    text = " ".join(w for _, w in positions)
    return (text[:limit] + "…") if len(text) > limit else text


class OpenAlexEvidenceProvider:
    name = "openalex"

    def __init__(self, source: Optional[OpenAlexSource] = None) -> None:
        self.source = source or OpenAlexSource()

    def gather(self, a_name: str, c_name: str, max_items: int = 4) -> EvidenceResult:
        count = self.source.comention_count(a_name, c_name)
        items: list[EvidenceItem] = []
        if count:
            for w in self.source.comention_works(a_name, c_name, limit=max_items):
                items.append(
                    EvidenceItem(
                        title=w.get("title") or "(untitled)",
                        url=w.get("id"),
                        snippet=_reconstruct_abstract(w.get("abstract_inverted_index")),
                        source="openalex",
                        year=w.get("publication_year"),
                        kind="paper",
                    )
                )
        return EvidenceResult(direct_hits=count, items=items, provider=self.name,
                              query=f"{a_name} AND {c_name}")


class ParallelEvidenceProvider:
    """Parallel Search API (https://docs.parallel.ai/api-reference/search)."""

    name = "parallel"
    URL = "https://api.parallel.ai/v1/search"

    def __init__(self, api_key: Optional[str] = None, timeout: float = 45.0) -> None:
        self.api_key = api_key or os.environ.get("PARALLEL_API_KEY")
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def gather(self, a_name: str, c_name: str, max_items: int = 4) -> EvidenceResult:
        objective = (
            f"Determine whether the scientific literature has directly studied a "
            f"relationship, mechanism, or therapeutic link between '{a_name}' and "
            f"'{c_name}'. Return the most relevant primary sources."
        )
        payload = {
            "objective": objective,
            "search_queries": [
                f"{a_name} {c_name} relationship",
                f"{a_name} {c_name} mechanism study",
                f"effect of {c_name} on {a_name}",
            ],
        }
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                self.URL,
                headers={"x-api-key": self.api_key, "Content-Type": "application/json"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
        a_l, c_l = a_name.lower(), c_name.lower()
        items: list[EvidenceItem] = []
        direct = 0
        for r in data.get("results", [])[: max_items * 2]:
            blob = (r.get("title", "") + " " + " ".join(r.get("excerpts", []))).lower()
            mentions_both = a_l in blob and c_l in blob
            if mentions_both:
                direct += 1
            if len(items) < max_items:
                items.append(
                    EvidenceItem(
                        title=r.get("title") or r.get("url", "(web result)"),
                        url=r.get("url"),
                        snippet=(r.get("excerpts") or [""])[0][:240],
                        source="parallel",
                        year=_year_from(r.get("publish_date")),
                        kind="web",
                    )
                )
        return EvidenceResult(direct_hits=direct, items=items, provider=self.name,
                              query=objective)


class PaperclipEvidenceProvider:
    """GXL Paperclip (11M+ full-text papers, clinical trials, FDA docs).

    Experimental: requires ``pip install gxl_paperclip`` and ``PAPERCLIP_API_KEY``.
    Uses count mode for the verdict and a small result set for citations. All
    calls are defensively wrapped so a schema change degrades gracefully.
    """

    name = "paperclip"

    def __init__(self, api_key: Optional[str] = None) -> None:
        self.api_key = api_key or os.environ.get("PAPERCLIP_API_KEY")
        self._client = None
        if self.api_key:
            try:  # pragma: no cover - optional dependency
                from gxl_paperclip import Paperclip  # type: ignore

                self._client = Paperclip(api_key=self.api_key)
            except Exception:  # noqa: BLE001
                self._client = None

    @property
    def available(self) -> bool:
        return self._client is not None

    def gather(self, a_name: str, c_name: str, max_items: int = 4) -> EvidenceResult:  # pragma: no cover - needs key
        query = f'"{a_name}" "{c_name}"'
        items: list[EvidenceItem] = []
        direct = -1
        try:
            res = self._client.search(query, n=max_items)  # type: ignore[union-attr]
            results = getattr(res, "results", res) or []
            direct = len(results)
            for r in results[:max_items]:
                get = (lambda k: r.get(k) if isinstance(r, dict) else getattr(r, k, None))
                items.append(
                    EvidenceItem(
                        title=get("title") or "(untitled)",
                        url=get("doi") or get("url"),
                        snippet=(get("tldr") or get("abstract") or "")[:240],
                        source=get("source") or "paperclip",
                        year=_year_from(get("date") or get("year")),
                        kind="paper",
                    )
                )
        except Exception:  # noqa: BLE001
            return EvidenceResult(direct_hits=-1, items=[], provider=self.name, query=query)
        return EvidenceResult(direct_hits=direct, items=items, provider=self.name, query=query)


class FixtureEvidenceProvider:
    """Deterministic offline provider backed by the fixture co-occurrence graph."""

    name = "fixture"

    def __init__(self, source=None) -> None:
        from hypha.sources.fixture import FixtureSource

        self.source = source or FixtureSource()

    def gather(self, a_name: str, c_name: str, max_items: int = 4) -> EvidenceResult:
        a = self.source.resolve_concept(a_name)
        c = self.source.resolve_concept(c_name)
        if a is None or c is None:
            return EvidenceResult(direct_hits=0, items=[], provider=self.name)
        count = self.source.cooccurrence_count(a, c)
        items = [
            EvidenceItem(title=w.title, url=w.url, source="fixture", year=w.year)
            for w in self.source.representative_works([a, c], limit=max_items)
        ] if count else []
        return EvidenceResult(direct_hits=count, items=items, provider=self.name,
                              query=f"{a_name} AND {c_name}")


# --------------------------------------------------------------------------- #
#  Verifier
# --------------------------------------------------------------------------- #
@dataclass
class VerifierConfig:
    emerging_max: int = 30      # <= this many co-mentions => "emerging"
    open_max: int = 0           # <= this many => "open" (undiscovered)


class Verifier:
    def __init__(self, provider: EvidenceProvider, config: Optional[VerifierConfig] = None):
        self.provider = provider
        self.cfg = config or VerifierConfig()

    def verify(self, hyp: Hypothesis) -> Hypothesis:
        a = hyp.bridge.source.name
        c = hyp.bridge.target.name
        try:
            res = self.provider.gather(a, c)
        except Exception as exc:  # noqa: BLE001 - verification is best-effort
            hyp.verdict = "unverified"
            hyp.verification_note = f"verification failed ({self.provider.name}): {exc}"
            return hyp

        hits = res.direct_hits
        hyp.evidence = res.items
        if hits < 0:
            hyp.verdict = "unverified"
            hyp.verification_note = (
                f"{res.provider}: returned {len(res.items)} related sources "
                f"(no reliable co-mention count)."
            )
            return hyp

        if hits <= self.cfg.open_max:
            hyp.verdict = "open"
            hyp.verified_novelty = 0.95
        elif hits <= self.cfg.emerging_max:
            hyp.verdict = "emerging"
            # scale 0.85 -> 0.5 across the emerging band
            span = max(self.cfg.emerging_max - self.cfg.open_max, 1)
            frac = (hits - self.cfg.open_max) / span
            hyp.verified_novelty = round(0.85 - 0.35 * frac, 3)
        else:
            hyp.verdict = "established"
            hyp.verified_novelty = round(min(0.4, 30.0 / hits), 3)

        hyp.verification_note = (
            f"{res.provider}: {hits} document(s) directly co-mention "
            f"'{a}' and '{c}' -> verdict '{hyp.verdict}'."
        )
        return hyp


# --------------------------------------------------------------------------- #
#  Factory
# --------------------------------------------------------------------------- #
def get_evidence_provider(
    offline: bool = False,
    source: Optional[OpenAlexSource] = None,
    prefer: Optional[str] = None,
) -> EvidenceProvider:
    """Pick the best available evidence provider.

    Priority (unless ``prefer`` forces one): Paperclip > Parallel > OpenAlex.
    OpenAlex is always available (free, no key), so verification works out of
    the box and is enhanced when keys are present.
    """
    if offline:
        return FixtureEvidenceProvider()

    if prefer == "openalex":
        return OpenAlexEvidenceProvider(source)
    if prefer in (None, "paperclip"):
        pc = PaperclipEvidenceProvider()
        if pc.available:
            return pc
        if prefer == "paperclip":
            return OpenAlexEvidenceProvider(source)
    if prefer in (None, "parallel"):
        par = ParallelEvidenceProvider()
        if par.available:
            return par
        if prefer == "parallel":
            return OpenAlexEvidenceProvider(source)
    return OpenAlexEvidenceProvider(source)


def _year_from(value) -> Optional[int]:
    if not value:
        return None
    s = str(value)
    for i in range(len(s) - 3):
        chunk = s[i : i + 4]
        if chunk.isdigit() and chunk.startswith(("19", "20")):
            return int(chunk)
    return None
