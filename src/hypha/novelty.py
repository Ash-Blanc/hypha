"""Pluggable novelty estimation for literature-based discovery.

The goal (generalizability + novelty first): proposal and ranking should be
able to use multiple independent signals of "how rarely / surprisingly are A
and C discussed together", not just coarse concept co-occurrence lift.

This module provides:
- ConceptLiftNoveltyScorer (the original math, generalized)
- ComentionNoveltyScorer (title/abstract or full-text direct co-mention count)
- Helpers to blend them and attach rich signals to BridgeLink for auditing.

Sources can implement `comention_count(a_name: str, c_name: str) -> int` (duck
typed) to participate in better novelty modes without changing their core
co-occurrence graph contract.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from hypha.models import BridgeLink, Concept


@dataclass
class NoveltySignal:
    """One contributing signal to an overall novelty assessment."""

    name: str
    value: float  # in [0,1] or raw; higher = more novel (less observed)
    weight: float = 1.0
    note: str = ""


class NoveltyScorer(Protocol):
    def score(self, a: Concept, c: Concept, *, source: Any = None, **kw: Any) -> NoveltySignal:
        ...


def _safe_div(num: float, den: float, default: float = 0.5) -> float:
    if den <= 0:
        return default
    return num / den


@dataclass
class ConceptLiftNoveltyScorer:
    """Original lift-based novelty using concept co-occurrence counts.

    lift = observed_direct / expected
    novelty = 1 / (1 + lift)   (clamped so lift <= max_lift is filtered upstream)
    Works even if works_count is missing (falls back conservatively).
    """

    corpus_size: int = 250_000_000
    max_lift: float = 2.0
    name: str = "lift"

    def score(
        self, a: Concept, c: Concept, *, source: Any = None, direct: Optional[int] = None, **kw: Any
    ) -> NoveltySignal:
        if direct is None and source is not None:
            # best-effort
            try:
                direct = int(source.cooccurrence_count(a, c))
            except Exception:  # noqa: BLE001
                direct = 0
        direct = max(0, int(direct or 0))

        wa = max(getattr(a, "works_count", 0) or 10, 10)
        wc = max(getattr(c, "works_count", 0) or 10, 10)
        expected = (wa * wc) / max(self.corpus_size, 1)
        lift = _safe_div(direct, max(expected, 0.5), default=0.0)
        # Clamp for stability; callers may have already filtered > max_lift
        lift = min(lift, self.max_lift)
        novelty = 1.0 / (1.0 + lift)
        return NoveltySignal(
            name=self.name,
            value=round(novelty, 4),
            weight=1.0,
            note=f"direct={direct} lift={lift:.2f}",
        )


@dataclass
class ComentionNoveltyScorer:
    """Novelty derived from direct title/abstract (or full-text) co-mention count.

    This is the "comention-native" signal that the back-test and verification
    layers already treat as ground truth for "was this link known?".

    Lower observed comention => higher novelty. We scale similarly to the
    Verifier (open/ emerging / established bands) but return a [0,1] novelty.
    """

    scale: float = 30.0  # comention count at which novelty bottoms out ~0.03
    emerging_max: int = 30
    name: str = "comention"

    def score(
        self, a: Concept, c: Concept, *, source: Any = None, direct: Optional[int] = None, **kw: Any
    ) -> NoveltySignal:
        hits = direct
        if hits is None and source is not None:
            fn = getattr(source, "comention_count", None)
            if callable(fn):
                try:
                    # Prefer name-based (works for synthetic/paperclip entities too)
                    hits = int(fn(a.name, c.name))
                except Exception:  # noqa: BLE001
                    hits = None
        if hits is None:
            # No signal available; neutral / unknown
            return NoveltySignal(name=self.name, value=0.5, weight=0.0, note="no-comention-signal")

        hits = max(0, int(hits))
        if hits == 0:
            novelty = 0.95
        elif hits <= self.emerging_max:
            span = max(self.emerging_max, 1)
            frac = hits / span
            novelty = round(0.85 - 0.35 * frac, 3)
        else:
            novelty = round(min(0.4, self.scale / max(hits, 1)), 3)

        return NoveltySignal(
            name=self.name,
            value=novelty,
            weight=1.0,
            note=f"comention_hits={hits}",
        )


def blend_novelty(signals: list[NoveltySignal], weights: Optional[dict[str, float]] = None) -> tuple[float, list[NoveltySignal]]:
    """Weighted average of available (weight>0) signals. Returns (blended, signals)."""
    if not signals:
        return 0.5, []
    active = [s for s in signals if s.weight > 0]
    if not active:
        # all signals had weight 0 (e.g. no comention available) -> fall back to first
        return signals[0].value, signals

    if weights:
        # Apply caller-provided relative weights (e.g. {"lift": 0.55, "comention": 0.45})
        total_w = 0.0
        acc = 0.0
        used: list[NoveltySignal] = []
        for s in active:
            w = float(weights.get(s.name, s.weight))
            if w <= 0:
                continue
            acc += s.value * w
            total_w += w
            used.append(NoveltySignal(s.name, s.value, w, s.note))
        if total_w <= 0:
            return active[0].value, active
        return round(acc / total_w, 4), used
    else:
        # uniform among active
        total = sum(s.value * s.weight for s in active)
        wsum = sum(s.weight for s in active)
        return round(total / max(wsum, 1e-9), 4), active


def compute_novelty_for_link(
    link: BridgeLink,
    cfg,  # DiscoveryConfig to avoid circular
    source: Any = None,
) -> tuple[float, dict[str, float], list[NoveltySignal]]:
    """Compute a (possibly blended) novelty for an already-populated BridgeLink.

    Returns (final_novelty, signals_dict, signal_objects).
    Mutates link.comention_count and link.signals as a side effect when possible.
    """
    from hypha.discovery import DiscoveryConfig  # local to avoid top-level cycle

    if not isinstance(cfg, DiscoveryConfig):
        # tolerate plain dicts in some call sites
        mode = getattr(cfg, "novelty_mode", "concept_lift")
        w = getattr(cfg, "novelty_weights", {"lift": 0.55, "comention": 0.45})
        cscale = getattr(cfg, "comention_novelty_scale", 30.0)
    else:
        mode = cfg.novelty_mode
        w = cfg.novelty_weights or {"lift": 0.55, "comention": 0.45}
        cscale = cfg.comention_novelty_scale

    signals: list[NoveltySignal] = []
    comention_hits = 0

    # Always compute lift (cheap if direct already on link, else ask source)
    lift_scorer = ConceptLiftNoveltyScorer(
        corpus_size=getattr(cfg, "corpus_size", 250_000_000),
        max_lift=getattr(cfg, "max_lift", 2.0),
    )
    direct_for_lift = getattr(link, "direct_cooccurrence", 0)
    lift_sig = lift_scorer.score(
        link.source, link.target, source=source, direct=direct_for_lift
    )
    signals.append(lift_sig)

    # Comention if mode wants it and source supports
    if mode in ("comention", "hybrid"):
        cscore = ComentionNoveltyScorer(scale=cscale)
        c_sig = cscore.score(link.source, link.target, source=source)
        if c_sig.weight > 0:
            signals.append(c_sig)
            # try to parse the note back to an int for storage
            try:
                comention_hits = int(c_sig.note.split("=")[-1])
            except Exception:  # noqa: BLE001
                comention_hits = 0

    blended, used = blend_novelty(signals, weights=w if mode == "hybrid" else None)

    # Attach to link for downstream (ranking, trace, report)
    try:
        link.comention_count = max(getattr(link, "comention_count", 0) or 0, comention_hits)
        sigdict = {s.name: s.value for s in used}
        link.signals = {**getattr(link, "signals", {}), **sigdict}
    except Exception:  # noqa: BLE001 - best effort
        pass

    # The "novelty" field on link has historically been the lift-style one.
    # We keep it for compat but callers doing rich ranking should prefer the
    # blended value we return here (or the one stored in signals).
    return blended, {s.name: s.value for s in used}, used


def attach_diversity(link: BridgeLink, diversity: float) -> None:
    """Helper to set the diversity field (and incorporate into signals if present)."""
    try:
        link.diversity = round(max(0.0, min(1.0, float(diversity))), 3)
        if hasattr(link, "signals") and isinstance(link.signals, dict):
            link.signals["bridge_diversity"] = link.diversity
    except Exception:  # noqa: BLE001
        pass


def compute_bridge_diversity(
    bridges: list[Concept],
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None,
) -> float:
    """Bridge diversity.

    - If `embedder` is supplied: mean pairwise (1 - cosine) of the *embeddings*
      of the bridge names. This is true semantic diversity of mechanisms /
      intermediates (the original roadmap desire).
    - Otherwise: lightweight token-Jaccard fallback (no extra deps).

    Higher = more diverse (good). Used when bridge_diversity_weight > 0 in config.
    """
    if len(bridges) <= 1:
        return 1.0

    if embedder:
        try:
            names = [b.name for b in bridges]
            vecs = embedder(names)
            pairs = 0
            acc = 0.0
            for i in range(len(vecs)):
                for j in range(i + 1, len(vecs)):
                    sim = sum(x * y for x, y in zip(vecs[i], vecs[j])) / (
                        (sum(x * x for x in vecs[i]) ** 0.5 or 1e-9) *
                        (sum(y * y for y in vecs[j]) ** 0.5 or 1e-9)
                    )
                    acc += 1.0 - sim
                    pairs += 1
            return round(acc / max(pairs, 1), 3)
        except Exception:  # noqa: BLE001 - fall back silently
            pass

    # Token fallback (original behavior)
    def toks(c: Concept) -> set[str]:
        # reuse the (shim) distinctive logic; falls back to all tokens if needed
        from hypha.filters import _distinctive_tokens as dt
        import re as _re

        t = dt(c.name)
        return t if t else {x for x in _re.split(r"[^a-z0-9]+", c.name.lower()) if x}

    pairs = 0
    acc = 0.0
    for i in range(len(bridges)):
        for j in range(i + 1, len(bridges)):
            ti, tj = toks(bridges[i]), toks(bridges[j])
            inter = len(ti & tj)
            union = len(ti | tj) or 1
            jacc = inter / union
            acc += 1.0 - jacc
            pairs += 1
    return round(acc / max(pairs, 1), 3)