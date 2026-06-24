"""Time-sliced back-testing: does Hypha actually *predict* discoveries?

The honest validation question for any literature-based-discovery engine is:
*if you'd run it in the past, would its top predictions have come true?*

This harness answers it quantitatively:

1. Run discovery on a topic using **only literature published up to a split
   year Y** (via OpenAlex publication-date windowing).
2. Take the top-K novel A-C links predicted "as of Y".
3. Look at the **future window (Y+1 .. now)** and check whether each predicted
   pair actually started being co-mentioned in the literature.
4. A *hit* = a link that was novel as of Y and emerged afterwards.

The headline metric is **precision@k**: the fraction of Hypha's top-K
predicted-novel links that later became real. This is the number to show YC /
VCs and to put in front of design partners.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from hypha.discovery import DiscoveryConfig, discover_links
from hypha.models import Concept


@dataclass
class BacktestLink:
    target: str
    bridges: list[str]
    past_comentions: int      # title/abstract co-mentions as of Y
    future_comentions: int    # title/abstract co-mentions in (Y, now]
    novel_as_of_split: bool
    emerged: bool
    score: float

    @property
    def hit(self) -> bool:
        return self.novel_as_of_split and self.emerged


@dataclass
class BacktestResult:
    topic: str
    split_year: int
    k: int
    emergence_threshold: int
    source_concept: Optional[str] = None
    pool_size: int = 0           # total candidate links scored
    n_novel_in_pool: int = 0     # how many of those were novel as of Y
    links: list[BacktestLink] = field(default_factory=list)  # top-k NOVEL predictions

    @property
    def n_hits(self) -> int:
        return sum(1 for l in self.links if l.hit)

    @property
    def precision_at_k(self) -> float:
        """Of Hypha's top-k genuinely-novel predictions, how many emerged."""
        return self.n_hits / len(self.links) if self.links else 0.0

    def summary(self) -> dict:
        return {
            "topic": self.topic,
            "split_year": self.split_year,
            "k": self.k,
            "candidates_scored": self.pool_size,
            "novel_as_of_split": self.n_novel_in_pool,
            "evaluated_predictions": len(self.links),
            "hits": self.n_hits,
            "precision_at_k": round(self.precision_at_k, 3),
        }


def backtest(
    topic: str,
    split_year: int,
    *,
    k: int = 10,
    emergence_threshold: int = 3,
    novelty_ceiling: int = 2,
    config: Optional[DiscoveryConfig] = None,
    past_source: object | None = None,
    future_source: object | None = None,
) -> BacktestResult:
    """Run a time-sliced back-test for ``topic`` at ``split_year``.

    ``past_source`` / ``future_source`` can be injected (for tests); otherwise
    OpenAlex sources windowed to ``<= split_year`` and ``> split_year`` are used.
    """
    if past_source is None or future_source is None:
        from hypha.sources.openalex import OpenAlexSource

        past_source = past_source or OpenAlexSource(to_year=split_year)
        future_source = future_source or OpenAlexSource(from_year=split_year + 1)

    # Scan a deep candidate pool so genuinely-novel links (which sit below the
    # already-studied ones in raw structural rank) are included. We disable the
    # concept-pair lift gate here and judge novelty independently by the
    # title/abstract co-mention count - the same signal used for emergence -
    # so "novel as of Y" and "emerged after Y" are measured on equal footing.
    pool = max(k * 8, 80)
    base = config or DiscoveryConfig()
    # Preserve any richer novelty/filter settings the caller passed; force a deep
    # candidate pool and disable the classic lift gate (back-test judges novelty
    # via comention/emergence for honesty, as before).
    cfg = DiscoveryConfig(
        **{
            **base.__dict__,
            "max_results": pool,
            "n_candidates_checked": pool,
            "max_lift": 10**9,
        }
    )

    a, links, _stats, _trace = discover_links(past_source, topic, cfg)
    result = BacktestResult(
        topic=topic,
        split_year=split_year,
        k=k,
        emergence_threshold=emergence_threshold,
        source_concept=a.name if a else None,
    )
    if a is None:
        return result

    scored: list[BacktestLink] = []
    for link in links:
        past = future_source_safe_comention(past_source, a, link.target)
        future = future_source_safe_comention(future_source, a, link.target)
        scored.append(
            BacktestLink(
                target=link.target.name,
                bridges=[b.name for b in link.bridges],
                past_comentions=past,
                future_comentions=future,
                novel_as_of_split=past <= novelty_ceiling,
                emerged=future >= emergence_threshold,
                score=link.score,
            )
        )

    result.pool_size = len(scored)
    novel = [l for l in scored if l.novel_as_of_split]
    result.n_novel_in_pool = len(novel)
    # The product's actual output: the top-k genuinely-novel predictions.
    novel.sort(key=lambda l: l.score, reverse=True)
    result.links = novel[:k]
    return result


def future_source_safe_comention(source: object, a: Concept, c: Concept) -> int:
    """Best-effort title/abstract co-mention count via a source."""
    fn = getattr(source, "comention_count", None)
    if callable(fn):
        try:
            return int(fn(a.name, c.name))
        except Exception:  # noqa: BLE001
            return 0
    # Fallback for sources without comention support (e.g. fixtures).
    fn2 = getattr(source, "cooccurrence_count", None)
    if callable(fn2):
        try:
            return int(fn2(a, c))
        except Exception:  # noqa: BLE001
            return 0
    return 0
