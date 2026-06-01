"""The Hypha discovery agent.

Orchestrates the full agentic loop:

    plan -> resolve -> gather bridges -> link -> rank ->
    gather citations -> hypothesize -> critique -> report

Every step appends to a transparent trace so the output is fully auditable
(an explicit design goal: literature-based discovery is only useful if a
human can follow and check the reasoning chain).
"""

from __future__ import annotations

from typing import Optional

from hypha.discovery import DiscoveryConfig, discover_links
from hypha.models import DiscoveryReport, Hypothesis, TraceEvent
from hypha.reasoning import FallbackReasoner, get_reasoner
from hypha.sources.base import ScholarSource
from hypha.sources.openalex import OpenAlexSource


class DiscoveryAgent:
    def __init__(
        self,
        source: Optional[ScholarSource] = None,
        reasoner: object | None = None,
        config: Optional[DiscoveryConfig] = None,
        prefer_llm: bool = True,
    ) -> None:
        self.source = source or OpenAlexSource()
        self.reasoner = reasoner or get_reasoner(prefer_llm=prefer_llm)
        self.config = config or DiscoveryConfig()

    def discover(self, topic: str, max_hypotheses: Optional[int] = None) -> DiscoveryReport:
        cfg = self.config
        if max_hypotheses is not None:
            cfg = DiscoveryConfig(**{**cfg.__dict__, "max_results": max_hypotheses})

        source_concept, links, stats, trace = discover_links(self.source, topic, cfg)
        report = DiscoveryReport(
            topic=topic,
            source_concept=source_concept,
            trace=list(trace),
            source_name=getattr(self.source, "name", "unknown"),
            reasoner=getattr(self.reasoner, "name", "fallback"),
            stats=stats,
        )
        if source_concept is None or not links:
            report.trace.append(
                TraceEvent(step="done", detail="No novel cross-domain links surfaced.")
            )
            return report

        hypotheses: list[Hypothesis] = []
        for link in links:
            works = self._citations(link)
            hyp = self.reasoner.reason(link, works)
            hyp = self._critique(hyp)
            hypotheses.append(hyp)

        report.hypotheses = hypotheses
        report.trace.append(
            TraceEvent(
                step="hypothesize",
                detail=f"Drafted {len(hypotheses)} hypotheses via reasoner '{report.reasoner}'.",
            )
        )
        return report

    # ------------------------------------------------------------------
    def _citations(self, link) -> list:
        """Collect representative works grounding each leg of the bridge."""
        works = []
        seen = set()
        for bridge in link.bridges[:2]:
            for legs in ([link.source, bridge], [bridge, link.target]):
                try:
                    for w in self.source.representative_works(legs, limit=1):
                        if w.id not in seen:
                            seen.add(w.id)
                            works.append(w)
                except Exception:  # noqa: BLE001 - citations are best-effort
                    continue
        return works

    def _critique(self, hyp: Hypothesis) -> Hypothesis:
        """A lightweight self-check that adjusts confidence and flags weak links."""
        link = hyp.bridge
        if link.bridge_support <= 1:
            hyp.plausibility_score *= 0.8
            hyp.rationale += (
                " [critique] Only a single bridge supports this link; treat as "
                "speculative until corroborated by an independent intermediate."
            )
        if link.direct_cooccurrence == 0 and link.bridge_support >= 3:
            hyp.rationale += (
                " [critique] Strong multi-bridge support with zero direct co-mention "
                "makes this a high-value 'undiscovered public knowledge' candidate."
            )
        hyp.plausibility_score = round(min(1.0, hyp.plausibility_score), 3)
        return hyp


def run_discovery(
    topic: str,
    *,
    offline: bool = False,
    max_hypotheses: int = 8,
    prefer_llm: bool = True,
) -> DiscoveryReport:
    """Convenience entry point used by the CLI and API."""
    if offline:
        from hypha.sources.fixture import FixtureSource

        agent = DiscoveryAgent(
            source=FixtureSource(),
            reasoner=FallbackReasoner(),
            prefer_llm=False,
        )
    else:
        agent = DiscoveryAgent(prefer_llm=prefer_llm)
    return agent.discover(topic, max_hypotheses=max_hypotheses)
