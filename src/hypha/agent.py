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
from hypha.evidence import EvidenceProvider, Verifier
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
        evidence_provider: Optional[EvidenceProvider] = None,
    ) -> None:
        self.source = source or OpenAlexSource()
        self.reasoner = reasoner or get_reasoner(prefer_llm=prefer_llm)
        self.config = config or DiscoveryConfig()
        self.evidence_provider = evidence_provider

    def discover(
        self,
        topic: str,
        max_hypotheses: Optional[int] = None,
        verify: bool = False,
    ) -> DiscoveryReport:
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

        report.trace.append(
            TraceEvent(
                step="hypothesize",
                detail=f"Drafted {len(hypotheses)} hypotheses via reasoner '{report.reasoner}'.",
            )
        )

        if verify and self.evidence_provider is not None:
            hypotheses = self._verify(hypotheses, report)

        report.hypotheses = hypotheses
        return report

    def _verify(self, hypotheses: list[Hypothesis], report: DiscoveryReport) -> list[Hypothesis]:
        verifier = Verifier(self.evidence_provider)
        for hyp in hypotheses:
            verifier.verify(hyp)
        # Re-rank: prioritise verified novelty x plausibility, pushing
        # already-"established" links down even if their structural score was high.
        hypotheses.sort(
            key=lambda h: h.ranking_novelty() * max(h.plausibility_score, 0.05),
            reverse=True,
        )
        counts: dict[str, int] = {}
        for h in hypotheses:
            counts[h.verdict] = counts.get(h.verdict, 0) + 1
        report.reasoner = report.reasoner
        report.stats = {**report.stats, "verdicts": counts}
        report.trace.append(
            TraceEvent(
                step="verify",
                detail=(
                    f"Verified {len(hypotheses)} links via "
                    f"'{getattr(self.evidence_provider, 'name', 'evidence')}' and re-ranked "
                    f"by verified novelty. Verdicts: {counts}."
                ),
            )
        )
        return hypotheses

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
    verify: bool = False,
    evidence: Optional[str] = None,
) -> DiscoveryReport:
    """Convenience entry point used by the CLI and API."""
    from hypha.evidence import get_evidence_provider

    if offline:
        from hypha.sources.fixture import FixtureSource

        agent = DiscoveryAgent(
            source=FixtureSource(),
            reasoner=FallbackReasoner(),
            prefer_llm=False,
            evidence_provider=get_evidence_provider(offline=True) if verify else None,
        )
    else:
        source = OpenAlexSource()
        provider = (
            get_evidence_provider(offline=False, source=source, prefer=evidence)
            if verify
            else None
        )
        agent = DiscoveryAgent(
            source=source, prefer_llm=prefer_llm, evidence_provider=provider
        )
    return agent.discover(topic, max_hypotheses=max_hypotheses, verify=verify)
