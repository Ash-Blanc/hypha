"""Tests for the time-sliced back-test harness (offline, with fake sources)."""

from hypha.backtest import backtest
from hypha.models import Concept


class FakePastSource:
    """A tiny windowed source that proposes two candidate links from A."""

    name = "fake-past"

    def __init__(self):
        self._concepts = {
            "a": Concept(id="A", name="Disease A", level=2, works_count=1000),
            "b": Concept(id="B", name="Bridge B", level=3, works_count=2000),
            "c1": Concept(id="C1", name="Drug Hit", level=3, works_count=1500),
            "c2": Concept(id="C2", name="Drug Miss", level=3, works_count=1500),
        }

    def resolve_concept(self, topic):
        return self._concepts["a"]

    def cooccurring_concepts(self, concept, limit=200):
        if concept.short_id() == "A":
            b = self._concepts["b"]; b.score = 100.0
            return [b]
        if concept.short_id() == "B":
            c1 = self._concepts["c1"]; c1.score = 80.0
            c2 = self._concepts["c2"]; c2.score = 70.0
            return [c1, c2]
        return []

    def cooccurrence_count(self, a, c):
        return 0  # nothing co-occurs directly as of the split year

    def comention_count(self, a_name, c_name):
        return 0  # novel as of split year


class FakeFutureSource:
    """Future window: 'Drug Hit' emerged, 'Drug Miss' did not."""

    name = "fake-future"

    def comention_count(self, a_name, c_name):
        return 12 if c_name == "Drug Hit" else 0


def test_backtest_scores_precision():
    res = backtest(
        "Disease A",
        2010,
        k=10,
        emergence_threshold=3,
        past_source=FakePastSource(),
        future_source=FakeFutureSource(),
    )
    assert res.source_concept == "Disease A"
    targets = {l.target: l for l in res.links}
    assert targets["Drug Hit"].hit is True
    assert targets["Drug Miss"].hit is False
    # 2 candidates, both novel as of split, 1 emerged -> precision 0.5
    assert res.n_novel_in_pool == 2
    assert res.n_hits == 1
    assert res.precision_at_k == 0.5


def test_backtest_summary_serialises():
    res = backtest(
        "Disease A", 2010,
        past_source=FakePastSource(), future_source=FakeFutureSource(),
    )
    s = res.summary()
    assert s["split_year"] == 2010
    assert s["hits"] == 1
    assert 0.0 <= s["precision_at_k"] <= 1.0


def test_backtest_unknown_topic_is_empty():
    class Empty:
        name = "empty"
        def resolve_concept(self, t): return None
        def cooccurring_concepts(self, c, limit=200): return []
        def cooccurrence_count(self, a, c): return 0
        def comention_count(self, a, c): return 0

    res = backtest("nope", 2010, past_source=Empty(), future_source=Empty())
    assert res.links == []
    assert res.precision_at_k == 0.0
