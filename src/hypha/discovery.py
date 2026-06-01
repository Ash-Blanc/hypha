"""The ABC literature-based discovery engine.

Given a source concept **A**, we:

1. find the concepts **B** that co-occur strongly with A (the bridge layer);
2. for each B, find the concepts **C** that co-occur with B;
3. keep the C concepts that are *not* already directly linked to A;
4. score each surviving A-C pair by how many independent bridges connect them
   (bridge support), how strong those two-hop paths are (path strength), and how
   rarely A and C are mentioned together directly (novelty).

A high-scoring A-C pair is a candidate piece of "undiscovered public
knowledge" - a connection implied by the literature but not yet stated in it.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field

from hypha.models import BridgeLink, Concept, TraceEvent
from hypha.sources.base import ScholarSource

# Vague umbrella concepts that are technically mid-level in OpenAlex but carry
# no discovery value as bridges or targets.
NAME_STOPLIST = {
    "disease",
    "syndrome",
    "disorder",
    "phenomenon",
    "context",
    "context (archaeology)",
    "systemic disease",
    "pathology",
    "etiology",
    "pathogenesis",
    "clinical significance",
    "prevalence",
    "incidence (epidemiology)",
    "comorbidity",
    "quality of life",
    "mortality",
    "prognosis",
    "epidemiology",
    "complication",
    "differential diagnosis",
    "intensive care medicine",
    "physical therapy",
    "in patient",
    "population",
    "young adult",
    "gerontology",
    "environmental health",
    # Ubiquitous methodology / measurement / omics terms that co-occur with
    # almost everything and carry no discovery value as a target or bridge.
    "gene expression",
    "in vitro",
    "in vivo",
    "in silico",
    "biomarker",
    "antigen",
    "antibody",
    "cytokine",
    "phenotype",
    "genotype",
    "messenger rna",
    "complementary dna",
    "peptide sequence",
    "amino acid",
    "cell",
    "cell culture",
    "cell biology",
    "computational biology",
    "bioinformatics",
    "immune system",
    "receptor",
    "enzyme",
    "protein",
    "gene",
    "transcription factor",
    "cell growth",
    "intracellular",
    "cohort",
    "clinical trial",
    "randomized controlled trial",
    "placebo",
    "meta analysis",
    "systematic review",
    "odds ratio",
    "confidence interval",
    "logistic regression",
    "retrospective cohort study",
    "case control study",
    "multivariate analysis",
    "regression analysis",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "algorithm",
    "function (biology)",
    "test (biology)",
    "perception",
    "mental health",
    "rehabilitation",
    "public health",
    "nursing",
    "well-being",
    "attention",
    "behavior",
    "stress (mechanics)",
}

# Parenthetical sense tags that indicate an out-of-domain homonym in OpenAlex's
# (frozen) concept set - e.g. "Scleroderma (fungus)" or "Amyloid (mycology)" in
# a biomedical query. Such concepts are almost always noise.
HOMONYM_TAGS = {
    "mycology",
    "fungus",
    "archaeology",
    "geology",
    "programming language",
    "album",
    "band",
    "software",
    "journal",
    "given name",
    "surname",
    "plant",
    "insect",
    "moth",
    "genus",
    "color",
    "mathematics",
    "geometry",
    "opera",
    "film",
    "novel",
    "tv series",
    "magazine",
    "company",
    "river",
    "city",
}


def _bad_parenthetical(name: str) -> bool:
    m = re.search(r"\(([^)]+)\)", name)
    return bool(m and m.group(1).strip().lower() in HOMONYM_TAGS)

# Tokens stripped when computing a concept's "distinctive" terms (used to drop
# near-synonyms of the source concept A).
GENERIC_TOKENS = {
    "disease",
    "diseases",
    "syndrome",
    "disorder",
    "disorders",
    "the",
    "of",
    "and",
    "in",
    "a",
    "an",
    "s",
    "phenomenon",
    "primary",
    "secondary",
    "acute",
    "chronic",
}


def _tokens(name: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}


def _distinctive_tokens(name: str) -> set[str]:
    return _tokens(name) - GENERIC_TOKENS


def _is_generic(name: str) -> bool:
    n = name.strip().lower()
    if n in NAME_STOPLIST:
        return True
    return bool(_distinctive_tokens(name)) is False


@dataclass
class DiscoveryConfig:
    n_bridges: int = 10
    n_candidates_checked: int = 25
    max_results: int = 8
    min_level: int = 2
    # Novelty is measured by *lift* (observed direct co-occurrence / expected by
    # chance) rather than a raw count, so it works for both tiny and huge
    # fields. A pair is "novel" when it co-occurs no more than ``max_lift`` times
    # what independence would predict.
    max_lift: float = 2.0
    corpus_size: int = 250_000_000
    bridge_pool: int = 200


@dataclass
class _Candidate:
    concept: Concept
    bridges: list[Concept] = field(default_factory=list)
    path_strength: float = 0.0


def _normalize(values: dict[str, float]) -> dict[str, float]:
    if not values:
        return {}
    hi = max(values.values())
    if hi <= 0:
        return {k: 0.0 for k in values}
    return {k: v / hi for k, v in values.items()}


def _ensure_levels(source: ScholarSource, concepts: list[Concept]) -> None:
    """Fill in concept level + works_count via the source if unknown."""
    unknown = [c for c in concepts if c.level == 0 or c.works_count == 0]
    hydrate = getattr(source, "hydrate_meta", None)
    if unknown and callable(hydrate):
        meta = hydrate(unknown)
        for c in concepts:
            info = meta.get(c.short_id())
            if info:
                c.level = info["level"]
                c.works_count = info["works_count"]


def _specificity(concept: Concept) -> float:
    """IDF-style weight: rare (specific) concepts score higher than ubiquitous ones."""
    return 1.0 / (1.0 + math.log10(max(concept.works_count, 10)))


def discover_links(
    source: ScholarSource,
    topic: str,
    config: DiscoveryConfig | None = None,
) -> tuple[Concept | None, list[BridgeLink], dict, list[TraceEvent]]:
    cfg = config or DiscoveryConfig()
    trace: list[TraceEvent] = []

    a = source.resolve_concept(topic)
    if a is None:
        trace.append(TraceEvent(step="resolve", detail=f"No concept found for '{topic}'."))
        return None, [], {}, trace
    trace.append(
        TraceEvent(
            step="resolve",
            detail=f"Resolved '{topic}' -> '{a.name}' (level {a.level}, {a.works_count} works).",
            data={"concept": a.model_dump()},
        )
    )

    # --- Layer B: concepts directly co-occurring with A -------------------
    a_cooc = source.cooccurring_concepts(a, limit=cfg.bridge_pool)
    _ensure_levels(source, a_cooc)
    a_cooc_ids = {c.short_id() for c in a_cooc}
    a_cooc_score = {c.short_id(): c.score for c in a_cooc}
    a_norm = _normalize(a_cooc_score)

    # Distinctive tokens of A let us drop its own near-synonyms (e.g. the
    # source "Raynaud disease" vs the variant concept "Raynaud's disease").
    a_terms = _distinctive_tokens(a.name)

    def _usable(c: Concept) -> bool:
        if c.short_id() == a.short_id() or c.level < cfg.min_level:
            return False
        if _is_generic(c.name) or _bad_parenthetical(c.name):
            return False
        if a_terms and (_distinctive_tokens(c.name) & a_terms):
            return False
        return True

    usable_bridges = [c for c in a_cooc if _usable(c)]
    # Prefer bridges that are both strongly associated with A and reasonably
    # specific, rather than just the most frequent (often near-synonyms).
    usable_bridges.sort(
        key=lambda c: a_norm.get(c.short_id(), 0.0) * _specificity(c),
        reverse=True,
    )
    bridges = usable_bridges[: cfg.n_bridges]
    trace.append(
        TraceEvent(
            step="bridges",
            detail=f"Selected {len(bridges)} bridge (B) concepts from {len(a_cooc)} co-occurring concepts.",
            data={"bridges": [b.name for b in bridges]},
        )
    )
    if not bridges:
        return a, [], {"bridges": 0}, trace

    bridge_ids = {b.short_id() for b in bridges}

    # --- Layer C: concepts co-occurring with each B, not already linked to A
    candidates: dict[str, _Candidate] = {}
    for b in bridges:
        b_cooc = source.cooccurring_concepts(b, limit=cfg.bridge_pool)
        b_norm = _normalize({c.short_id(): c.score for c in b_cooc})
        ab = a_norm.get(b.short_id(), 0.0)
        for c in b_cooc:
            cid = c.short_id()
            if cid == a.short_id() or cid in a_cooc_ids or cid in bridge_ids:
                continue
            bc = b_norm.get(cid, 0.0)
            leg = math.sqrt(max(ab, 1e-6) * max(bc, 1e-6))
            cand = candidates.get(cid)
            if cand is None:
                cand = _Candidate(concept=c)
                candidates[cid] = cand
            cand.bridges.append(b)
            cand.path_strength += leg

    trace.append(
        TraceEvent(
            step="candidates",
            detail=f"Found {len(candidates)} indirect (C) candidates reachable through the bridge layer.",
        )
    )
    if not candidates:
        return a, [], {"bridges": len(bridges), "candidates": 0}, trace

    _ensure_levels(source, [c.concept for c in candidates.values()])
    filtered = [c for c in candidates.values() if _usable(c.concept)]

    # Preliminary ranking before the (costlier) direct-co-occurrence check.
    filtered.sort(key=lambda c: (len(c.bridges), c.path_strength), reverse=True)
    short = filtered[: cfg.n_candidates_checked]

    # --- Novelty: how rarely do A and C actually appear together? ---------
    path_norm = _normalize({c.concept.short_id(): c.path_strength for c in short})
    links: list[BridgeLink] = []
    for cand in short:
        direct = source.cooccurrence_count(a, cand.concept)
        # Expected co-occurrence under independence, and the resulting lift.
        expected = (a.works_count * cand.concept.works_count) / max(cfg.corpus_size, 1)
        lift = direct / max(expected, 0.5)
        if lift > cfg.max_lift:
            continue  # already meaningfully co-discussed -> not novel
        novelty = 1.0 / (1.0 + lift)
        support = len(cand.bridges) / max(len(bridges), 1)
        p = path_norm.get(cand.concept.short_id(), 0.0)
        spec = _specificity(cand.concept)
        score = (0.6 * p + 0.4 * support) * novelty * spec
        links.append(
            BridgeLink(
                source=a,
                target=cand.concept,
                bridges=sorted(cand.bridges, key=lambda b: a_norm.get(b.short_id(), 0), reverse=True),
                direct_cooccurrence=direct,
                bridge_support=len(cand.bridges),
                path_strength=cand.path_strength,
                novelty=novelty,
                score=score,
            )
        )

    links.sort(key=lambda link: link.score, reverse=True)
    links = links[: cfg.max_results]
    trace.append(
        TraceEvent(
            step="rank",
            detail=f"Ranked {len(links)} novel A-C links after the direct co-occurrence filter.",
            data={"links": [link.explain() for link in links]},
        )
    )

    stats = {
        "bridges": len(bridges),
        "candidates": len(candidates),
        "checked": len(short),
        "results": len(links),
    }
    return a, links, stats, trace
