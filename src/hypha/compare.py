"""Side-by-side comparison: OpenAlex ABC discovery vs Paperclip search mining."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hypha.discovery import DiscoveryConfig, discover_links
from hypha.filters import _bad_parenthetical, _is_generic
from hypha.evidence import Verifier, get_evidence_provider
from hypha.models import Hypothesis
from hypha.paperclip_client import PaperclipClient
from hypha.reasoning import FallbackReasoner, get_reasoner
from hypha.sources.openalex import GENERIC_STOPLIST, OpenAlexSource
from hypha.sources.paperclip import PaperclipDiscovery, proposal_to_link


@dataclass
class CompareRow:
    pipeline: str
    target: str
    bridges: str
    score: float
    verdict: str
    citations: int
    nonsense: bool
    actionable: bool
    note: str = ""


@dataclass
class CompareReport:
    topic: str
    openalex_source: str = ""
    rows: list[CompareRow] = field(default_factory=list)
    summary: dict[str, dict[str, float]] = field(default_factory=dict)


def _is_nonsense_target(name: str, level: int = 3) -> bool:
    low = name.strip().lower()
    if low in GENERIC_STOPLIST:
        return True
    if _is_generic(name) or _bad_parenthetical(name):
        return True
    if level < 2:
        return True
    # Obvious homonym / out-of-domain noise seen in bad OpenAlex runs
    if "(linguistics)" in name or "(graphical)" in name.lower():
        return True
    if name.lower() in {"medicine", "government", "perspective", "key"}:
        return True
    return False


def _is_actionable(hyp: Hypothesis) -> bool:
    mech = (hyp.mechanism or "").strip()
    exp = (hyp.experiment or "").strip()
    return len(mech) > 20 and len(exp) > 15


def _summarize(rows: list[CompareRow]) -> dict[str, float]:
    if not rows:
        return {"count": 0, "nonsense_rate": 0.0, "actionable_rate": 0.0, "avg_citations": 0.0}
    n = len(rows)
    nonsense = sum(1 for r in rows if r.nonsense)
    actionable = sum(1 for r in rows if r.actionable)
    cites = sum(r.citations for r in rows)
    return {
        "count": float(n),
        "nonsense_rate": nonsense / n,
        "actionable_rate": actionable / n,
        "avg_citations": cites / n,
    }


def run_compare(
    topic: str,
    *,
    max_rows: int = 6,
    verify: bool = False,
    offline: bool = False,
    prefer_llm: bool = True,
    evidence: Optional[str] = None,
) -> CompareReport:
    report = CompareReport(topic=topic)
    oa_rows, oa_label = _openalex_rows(topic, max_rows, verify, offline, prefer_llm, evidence)
    pc_rows = _paperclip_rows(topic, max_rows, verify, offline, prefer_llm, evidence)
    report.rows = oa_rows + pc_rows
    report.openalex_source = oa_label
    report.summary = {
        "openalex": _summarize(oa_rows),
        "paperclip": _summarize(pc_rows),
    }
    return report


def _openalex_rows(
    topic: str,
    max_rows: int,
    verify: bool,
    offline: bool,
    prefer_llm: bool,
    evidence: Optional[str],
) -> tuple[list[CompareRow], str]:
    if offline:
        from hypha.sources.fixture import FixtureSource

        source = FixtureSource()
    else:
        source = OpenAlexSource()
    a, links, _stats, _trace = discover_links(source, topic, DiscoveryConfig(max_results=max_rows))
    if a is None:
        return [
            CompareRow(
                pipeline="openalex",
                target="(none)",
                bridges="",
                score=0.0,
                verdict="—",
                citations=0,
                nonsense=True,
                actionable=False,
                note="Could not resolve source concept A.",
            )
        ], topic
    reasoner = FallbackReasoner() if offline or not prefer_llm else get_reasoner(prefer_llm=prefer_llm)
    verifier = Verifier(get_evidence_provider(offline=offline, source=source if isinstance(source, OpenAlexSource) else None, prefer=evidence)) if verify else None

    rows: list[CompareRow] = []
    for link in links[:max_rows]:
        hyp = reasoner.reason(link, source.representative_works([link.source] + link.bridges + [link.target], limit=2) if hasattr(source, "representative_works") else [])
        if verifier:
            hyp = verifier.verify(hyp)
        rows.append(
            CompareRow(
                pipeline="openalex",
                target=link.target.name,
                bridges=", ".join(b.name for b in link.bridges[:3]),
                score=link.score,
                verdict=hyp.verdict if verify else "—",
                citations=len(hyp.evidence) + len(hyp.supporting_works),
                nonsense=_is_nonsense_target(link.target.name, link.target.level),
                actionable=_is_actionable(hyp),
                note=f"A={a.name} (L{a.level})",
            )
        )
    return rows, a.name


def _paperclip_rows(
    topic: str,
    max_rows: int,
    verify: bool,
    offline: bool,
    prefer_llm: bool,
    evidence: Optional[str],
) -> list[CompareRow]:
    if offline:
        return []

    discovery = PaperclipDiscovery()
    if not discovery.available:
        return [
            CompareRow(
                pipeline="paperclip",
                target="(skipped)",
                bridges="",
                score=0.0,
                verdict="—",
                citations=0,
                nonsense=False,
                actionable=False,
                note="Set PAPERCLIP_API_KEY to enable Paperclip pipeline.",
            )
        ]

    label, proposals = discovery.propose(topic, limit=max_rows)
    reasoner = get_reasoner(prefer_llm=prefer_llm)
    provider = get_evidence_provider(prefer=evidence or "paperclip")
    verifier = Verifier(provider) if verify else None
    client = PaperclipClient()

    rows: list[CompareRow] = []
    for prop in proposals:
        link = proposal_to_link(label, prop)
        hyp = reasoner.reason(link, prop.supporting)
        if verify and verifier:
            if client.available:
                try:
                    prop.direct_hits = client.count_comention(label, prop.target)
                except Exception:  # noqa: BLE001
                    prop.direct_hits = -1
            hyp = verifier.verify(hyp)
        rows.append(
            CompareRow(
                pipeline="paperclip",
                target=prop.target,
                bridges=" | ".join(prop.bridges[:2]) or "(search hits)",
                score=prop.score,
                verdict=hyp.verdict if verify else "—",
                citations=len(hyp.evidence) + len(prop.supporting),
                nonsense=_is_nonsense_target(prop.target),
                actionable=_is_actionable(hyp),
                note=f"papers={prop.paper_count}",
            )
        )
    if not rows:
        rows.append(
            CompareRow(
                pipeline="paperclip",
                target="(none)",
                bridges="",
                score=0.0,
                verdict="—",
                citations=0,
                nonsense=False,
                actionable=False,
                note="No intervention phrases mined from Paperclip search.",
            )
        )
    return rows
