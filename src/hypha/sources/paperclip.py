"""Experimental Paperclip-backed open discovery.

Unlike OpenAlex concept co-occurrence, Paperclip proposes targets by mining
full-text search hits for recurring intervention / entity phrases. This is
noisier but grounded in document text — useful for side-by-side comparison.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from hypha.filters import _bad_parenthetical, _is_generic
from hypha.models import BridgeLink, Concept, EvidenceItem, SupportingWork
from hypha.paperclip_client import PaperclipClient, parse_papers

# Terms that often introduce a candidate intervention in paper titles.
_INTERVENTION_MARKERS = re.compile(
    r"\b(?:"
    r"drug|drugs|therapy|therapies|treatment|inhibitor|agonist|antagonist|"
    r"supplement|vitamin|steroid|peptide|antibody|vaccine|repurposing|"
    r"lanosterol|n-acetylcarnosine|carnosine|metformin|statins?"
    r")\b",
    re.I,
)

_TITLE_STOP = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "in",
    "on",
    "for",
    "with",
    "to",
    "from",
    "by",
    "via",
    "using",
    "based",
    "study",
    "studies",
    "review",
    "meta",
    "analysis",
    "clinical",
    "trial",
    "patients",
    "patient",
    "human",
    "humans",
    "mouse",
    "mice",
    "model",
    "models",
    "effect",
    "effects",
    "role",
    "novel",
    "new",
    "potential",
    "early",
    "late",
    "stage",
    "stages",
}


@dataclass
class PaperclipProposal:
    """A target C mined from Paperclip search, with supporting papers."""

    target: str
    score: float
    paper_count: int
    bridges: list[str] = field(default_factory=list)
    supporting: list[SupportingWork] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    direct_hits: int = -1


class PaperclipDiscovery:
    name = "paperclip"

    def __init__(self, client: PaperclipClient | None = None) -> None:
        self.client = client or PaperclipClient()

    @property
    def available(self) -> bool:
        return self.client.available

    def propose(self, topic: str, limit: int = 8) -> tuple[str, list[PaperclipProposal]]:
        """Return (resolved topic label, ranked target proposals)."""
        if not self.available:
            return topic, []

        queries = [
            f'"{topic}" treatment OR therapy OR drug',
            f'"{topic}" inhibitor OR supplement OR repurposing',
            f"{topic} mechanism pathway",
        ]
        papers: list[dict[str, str]] = []
        topic_terms = _topic_terms(topic)
        for q in queries:
            try:
                res = self.client.search(q, limit=25)
                papers.extend(parse_papers(res.output))
            except Exception:  # noqa: BLE001
                continue

        phrases = Counter[str]()
        phrase_papers: dict[str, list[dict[str, str]]] = {}
        for p in papers:
            title = p.get("title", "")
            if not title or not _mentions_topic(title, topic_terms):
                continue
            for phrase in _mine_phrases(title, topic_terms):
                if _is_bad_target(phrase, topic_terms):
                    continue
                phrases[phrase] += 1
                phrase_papers.setdefault(phrase, []).append(p)

        proposals: list[PaperclipProposal] = []
        for phrase, count in phrases.most_common(limit * 3):
            support = phrase_papers.get(phrase, [])
            proposals.append(
                PaperclipProposal(
                    target=phrase,
                    score=float(count),
                    paper_count=count,
                    bridges=[p.get("title", "")[:80] for p in support[:2]],
                    supporting=[
                        SupportingWork(
                            id=p.get("id", p.get("url", phrase)),
                            title=p.get("title", ""),
                            year=_year_from(p.get("date", "")),
                            url=p.get("url"),
                            links=[topic, phrase],
                        )
                        for p in support[:2]
                    ],
                )
            )
        proposals.sort(key=lambda x: x.score, reverse=True)
        return topic, proposals[:limit]


def _topic_terms(topic: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", topic.lower()) if len(t) >= 3 and t not in _TITLE_STOP}


def _mentions_topic(text: str, topic_terms: set[str]) -> bool:
    if not topic_terms:
        return True
    low = text.lower()
    return any(t in low for t in topic_terms)


def _mine_phrases(title: str, topic_terms: set[str]) -> list[str]:
    """Extract short intervention-like phrases from a paper title."""
    out: list[str] = []
    low = title.lower()
    if not _INTERVENTION_MARKERS.search(low):
        return out
    # Parenthetical drug names: "...(lanosterol)"
    for m in re.finditer(r"\(([A-Za-z][\w\-]+(?:\s[\w\-]+){0,3})\)", title):
        phrase = m.group(1).strip()
        if phrase.lower() not in topic_terms:
            out.append(_normalize_phrase(phrase))
    # "X therapy/treatment/drug" patterns
    for m in re.finditer(
        r"([A-Za-z][\w\-]+(?:\s[A-Za-z][\w\-]+){0,2})\s+"
        r"(?:therapy|treatment|drug|inhibitor|supplement|vaccine)s?",
        title,
        re.I,
    ):
        phrase = m.group(1).strip()
        if phrase.lower() not in topic_terms:
            out.append(_normalize_phrase(phrase))
    return out


def _normalize_phrase(phrase: str) -> str:
    return " ".join(phrase.split())


def _is_bad_target(phrase: str, topic_terms: set[str]) -> bool:
    if len(phrase) < 4:
        return True
    if _is_generic(phrase) or _bad_parenthetical(phrase):
        return True
    tokens = {t for t in re.split(r"[^a-z0-9]+", phrase.lower()) if t}
    if tokens & topic_terms:
        return True
    if tokens <= _TITLE_STOP:
        return True
    return False


def _year_from(value: str) -> int | None:
    for i in range(max(0, len(value) - 3)):
        chunk = value[i : i + 4]
        if chunk.isdigit() and chunk.startswith(("19", "20")):
            return int(chunk)
    return None


def proposal_to_link(topic: str, proposal: PaperclipProposal) -> BridgeLink:
    """Build pseudo bridge link for reasoner / verifier (topic -> target)."""
    a = Concept(id=f"paperclip:topic:{topic}", name=topic, level=3, works_count=0)
    c = Concept(id=f"paperclip:target:{proposal.target}", name=proposal.target, level=3)
    return BridgeLink(
        source=a,
        target=c,
        bridges=[Concept(id=f"paperclip:paper:{i}", name=b, level=3) for i, b in enumerate(proposal.bridges)],
        bridge_support=max(1, len(proposal.bridges)),
        direct_cooccurrence=proposal.direct_hits if proposal.direct_hits >= 0 else 0,
        novelty=0.7,
        score=proposal.score,
    )
