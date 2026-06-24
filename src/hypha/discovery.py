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
from dataclasses import dataclass, field
from typing import Optional

from hypha.filters import (
    RelevanceFilter,
    _bad_parenthetical,
    _distinctive_tokens,
    _is_generic,
    make_default_filter,
)
from hypha.models import BridgeLink, Concept, TraceEvent
from hypha.novelty import (
    attach_diversity,
    blend_novelty,
    compute_bridge_diversity,
    compute_novelty_for_link,
)
from hypha.sources.base import ScholarSource


@dataclass
class DiscoveryConfig:
    n_bridges: int = 10
    n_candidates_checked: int = 25
    max_results: int = 8
    min_level: int = 2
    # Optional MeSH category letters the target C must belong to (e.g. {"D"} for
    # drugs/chemicals). ``None`` = no type constraint.
    target_categories: Optional[frozenset[str]] = None
    n_typed_pool: int = 80  # how many top candidates to type before filtering
    # Novelty is measured by *lift* (observed direct co-occurrence / expected by
    # chance) rather than a raw count, so it works for both tiny and huge
    # fields. A pair is "novel" when it co-occurs no more than ``max_lift`` times
    # what independence would predict.
    max_lift: float = 2.0
    corpus_size: int = 250_000_000
    bridge_pool: int = 200

    # --- Generalizability & novelty upgrades (additive, defaults = v0 behavior) ---
    # Filter profile or explicit instance. "biomed" reproduces historical lists
    # exactly. "general" uses a much smaller stop set for non-biomed domains.
    filter_profile: str = "biomed"
    filters: Optional[RelevanceFilter] = None

    # Embedder for semantic (vector) filtering + diversity.
    # When present, RelevanceFilter uses cosine-to-prototypes for is_generic /
    # A-similarity instead of (or in addition to) giant static lists, and
    # compute_bridge_diversity uses real embedding distances.
    # Create with filters.make_openai_embedder() or your own Callable[[list[str]], list[list[float]]].
    # This is the main lever for *far less hardcoding* of stoplists.
    embedder: Any = None

    # How to compute/weight novelty for ranking (used in proposal stage).
    # "concept_lift": original behavior (concept cooc lift only).
    # "comention": prefer title/abstract (or fulltext) comention count when the
    #             source provides comention_count(a.name, c.name).
    # "hybrid": blend lift + comention signals (recommended for most real topics).
    novelty_mode: str = "concept_lift"  # "concept_lift" | "comention" | "hybrid"
    # Relative weights for a simple linear blend when mode="hybrid".
    novelty_weights: dict[str, float] = field(
        default_factory=lambda: {"lift": 0.55, "comention": 0.45}
    )
    # If > 0, reward A-C links whose supporting bridges are semantically diverse.
    # When embedder is present this uses vector cosine (much stronger than token Jaccard).
    bridge_diversity_weight: float = 0.0
    # Scale used when turning a raw comention count into a novelty-ish score in
    # comention/hybrid modes (higher count => lower novelty; mirrors Verifier logic).
    comention_novelty_scale: float = 30.0
    # Minimum number of bridges required before a link can be emitted (quality gate).
    min_bridge_support: int = 1


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


def _get_filter(cfg: DiscoveryConfig) -> RelevanceFilter:
    """Return a RelevanceFilter instance for this config (cached construction)."""
    if cfg.filters is not None:
        # If caller gave an explicit filter but we also have a top-level embedder,
        # make sure the filter sees it (unless it already has one).
        if getattr(cfg, "embedder", None) and not getattr(cfg.filters, "embedder", None):
            cfg.filters.embedder = cfg.embedder
        return cfg.filters
    flt = make_default_filter(cfg.filter_profile or "biomed")
    if getattr(cfg, "embedder", None):
        flt.embedder = cfg.embedder
    return flt


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
    typer: object | None = None,
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

    flt = _get_filter(cfg)
    # Distinctive tokens of A let us drop its own near-synonyms (e.g. the
    # source "Raynaud disease" vs the variant concept "Raynaud's disease").
    a_terms = _distinctive_tokens(a.name)  # still used for legacy path; filter also has it

    def _usable(c: Concept) -> bool:
        # Delegate to the (possibly custom) filter; fall back to legacy min_level
        # behavior for exact v0 compatibility when using default biomed filter.
        return flt.is_usable(c, a=a, min_level=cfg.min_level)

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
    filtered = [c for c in candidates.values() if _usable(c.concept)]  # re-uses the closure above (flt + a)

    # Preliminary ranking before the (costlier) direct-co-occurrence check.
    filtered.sort(key=lambda c: (len(c.bridges), c.path_strength), reverse=True)

    # Optional MeSH type constraint on the target C (e.g. drugs only).
    if cfg.target_categories and typer is not None:
        pool = filtered[: cfg.n_typed_pool]
        try:
            types = typer.type_many([c.concept.name for c in pool])
        except Exception:  # noqa: BLE001 - fail open if typing is unavailable
            types = {}
        if any(types.values()):
            kept = [
                c for c in pool
                if types.get(c.concept.name, set()) & cfg.target_categories
            ]
            trace.append(
                TraceEvent(
                    step="type",
                    detail=(
                        f"Constrained targets to MeSH categories "
                        f"{sorted(cfg.target_categories)}: kept {len(kept)} of "
                        f"{len(pool)} typed candidates."
                    ),
                    data={"kept": [c.concept.name for c in kept][:20]},
                )
            )
            filtered = kept if kept else filtered
        else:
            trace.append(
                TraceEvent(
                    step="type",
                    detail="MeSH typing unavailable; proceeding without a type filter.",
                )
            )

    short = filtered[: cfg.n_candidates_checked]

    # --- Novelty: how rarely do A and C actually appear together? ---------
    # We always compute the classic lift for the gate + backward compat, but in
    # "comention" / "hybrid" modes we also pull comention (when the source
    # supports it) and blend. Bridge diversity (lightweight) is opt-in.
    path_norm = _normalize({c.concept.short_id(): c.path_strength for c in short})
    links: list[BridgeLink] = []
    comention_available = False
    for cand in short:
        direct = source.cooccurrence_count(a, cand.concept)
        # Expected co-occurrence under independence, and the resulting lift.
        expected = (getattr(a, "works_count", 0) or 10) * (getattr(cand.concept, "works_count", 0) or 10) / max(
            getattr(cfg, "corpus_size", 250_000_000), 1
        )
        lift = direct / max(expected, 0.5)

        # Mode-aware gate: concept_lift keeps the historical strict filter for
        # exact v0 behavior on defaults/fixture. Richer modes are more permissive
        # here and let the blended novelty + verification do the heavy lifting.
        mode = getattr(cfg, "novelty_mode", "concept_lift")
        if mode == "concept_lift" and lift > getattr(cfg, "max_lift", 2.0):
            continue

        support = len(cand.bridges) / max(len(bridges), 1)
        p = path_norm.get(cand.concept.short_id(), 0.0)
        spec = _specificity(cand.concept)
        base_struct = (0.6 * p + 0.4 * support) * spec

        # Classic novelty (always available)
        classic_novelty = 1.0 / (1.0 + lift)

        # Build the link first with classic values (keeps explain() etc. happy)
        link = BridgeLink(
            source=a,
            target=cand.concept,
            bridges=sorted(cand.bridges, key=lambda b: a_norm.get(b.short_id(), 0), reverse=True),
            direct_cooccurrence=direct,
            bridge_support=len(cand.bridges),
            path_strength=cand.path_strength,
            novelty=classic_novelty,
            score=base_struct * classic_novelty,
        )

        # Richer novelty (comention + blend) when requested and available
        if mode in ("comention", "hybrid"):
            try:
                blended, sigdict, _sigs = compute_novelty_for_link(link, cfg, source=source)
                # Use blended for scoring in rich modes; keep classic in .novelty for compat
                link.novelty = blended  # historical field gets the one we actually ranked by
                link.signals = {**getattr(link, "signals", {}), **sigdict}
                comention_available = True
            except Exception:  # noqa: BLE001 - never break discovery on novelty
                pass

        # Optional bridge diversity (generalizability / quality signal)
        if getattr(cfg, "bridge_diversity_weight", 0.0) > 0 or mode != "concept_lift":
            try:
                emb = getattr(cfg, "embedder", None)
                div = compute_bridge_diversity(link.bridges, embedder=emb)
                attach_diversity(link, div)
                dw = getattr(cfg, "bridge_diversity_weight", 0.0)
                if dw > 0:
                    # Gentle multiplicative boost for diverse bridge sets
                    link.score = link.score * (1.0 + dw * max(0.0, (div - 0.4)))
            except Exception:  # noqa: BLE001
                pass

        # Final score (may have been adjusted by rich novelty above)
        if mode in ("comention", "hybrid"):
            # Re-apply a structural * novelty blend so very low-support things don't dominate
            link.score = base_struct * (getattr(link, "novelty", classic_novelty) or classic_novelty)
        else:
            link.score = base_struct * classic_novelty

        # Quality gate
        if getattr(cfg, "min_bridge_support", 1) > 1 and link.bridge_support < cfg.min_bridge_support:
            continue

        links.append(link)

    links.sort(key=lambda link: link.score, reverse=True)
    links = links[: cfg.max_results]

    rank_detail = f"Ranked {len(links)} novel A-C links (mode={getattr(cfg, 'novelty_mode', 'concept_lift')})."
    if comention_available:
        rank_detail += " Comention signal was used for ranking."
    trace.append(
        TraceEvent(
            step="rank",
            detail=rank_detail,
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


def explain_link(
    source: ScholarSource,
    a_topic: str,
    c_topic: str,
    config: DiscoveryConfig | None = None,
) -> tuple[Concept | None, Concept | None, BridgeLink | None, list[TraceEvent]]:
    """Closed discovery: given A *and* C, find the bridge concepts B that connect
    them and explain the implied path.

    Where :func:`discover_links` answers "what novel C relates to A?", this
    answers "*why* might A and C be related?" - useful for explaining a
    repurposing candidate (drug C, disease A) or an observed association.
    """
    cfg = config or DiscoveryConfig()
    trace: list[TraceEvent] = []

    a = source.resolve_concept(a_topic)
    c = source.resolve_concept(c_topic)
    if a is None or c is None:
        missing = a_topic if a is None else c_topic
        trace.append(TraceEvent(step="resolve", detail=f"Could not resolve '{missing}'."))
        return a, c, None, trace
    trace.append(
        TraceEvent(step="resolve", detail=f"A='{a.name}', C='{c.name}'.")
    )

    a_cooc = source.cooccurring_concepts(a, limit=cfg.bridge_pool)
    c_cooc = source.cooccurring_concepts(c, limit=cfg.bridge_pool)
    _ensure_levels(source, a_cooc)
    _ensure_levels(source, c_cooc)
    a_norm = _normalize({x.short_id(): x.score for x in a_cooc})
    c_norm = _normalize({x.short_id(): x.score for x in c_cooc})
    c_by_id = {x.short_id(): x for x in c_cooc}

    flt = _get_filter(cfg)
    a_terms = _distinctive_tokens(a.name)
    c_terms = _distinctive_tokens(c.name)

    shared: list[tuple[Concept, float]] = []
    for b in a_cooc:
        bid = b.short_id()
        if bid not in c_by_id or bid in (a.short_id(), c.short_id()):
            continue
        if b.level < cfg.min_level or flt.is_generic(b.name) or flt.is_bad_parenthetical(b.name):
            continue
        bt = _distinctive_tokens(b.name)
        if (a_terms and bt & a_terms) or (c_terms and bt & c_terms):
            continue
        strength = math.sqrt(max(a_norm.get(bid, 0), 1e-6) * max(c_norm.get(bid, 0), 1e-6))
        shared.append((b, strength))

    shared.sort(key=lambda t: t[1], reverse=True)
    top = shared[: cfg.n_bridges]
    trace.append(
        TraceEvent(
            step="bridges",
            detail=f"Found {len(shared)} shared intermediate concepts; using top {len(top)}.",
            data={"bridges": [b.name for b, _ in top]},
        )
    )
    if not top:
        return a, c, None, trace

    direct = source.cooccurrence_count(a, c)
    link = BridgeLink(
        source=a,
        target=c,
        bridges=[b for b, _ in top],
        direct_cooccurrence=direct,
        bridge_support=len(top),
        path_strength=sum(s for _, s in top),
        novelty=1.0 / (1.0 + direct),
        score=sum(s for _, s in top),
    )
    return a, c, link, trace
