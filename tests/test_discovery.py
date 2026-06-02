"""Tests for the ABC discovery engine using the offline fixture.

The fixture reproduces Swanson's two textbook literature-based discoveries, so
a correct engine must rediscover them: Raynaud disease -> fish oil, and
Migraine -> magnesium.
"""

from hypha.discovery import DiscoveryConfig, discover_links
from hypha.sources.fixture import FixtureSource


def _targets(topic: str) -> list[str]:
    source = FixtureSource()
    _a, links, _stats, _trace = discover_links(
        source, topic, DiscoveryConfig(min_level=2)
    )
    return [link.target.name for link in links]


def test_rediscovers_raynaud_fish_oil():
    targets = _targets("Raynaud disease")
    assert "Fish oil" in targets
    # Fish oil should be the strongest link (3 independent bridges, 0 direct).
    assert targets[0] == "Fish oil"


def test_rediscovers_migraine_magnesium():
    targets = _targets("Migraine")
    assert "Magnesium" in targets


def test_excludes_directly_linked_concepts():
    # Vasospasm co-occurs directly with Raynaud, so it must not appear as a
    # "novel" target C.
    assert "Vasospasm" not in _targets("Raynaud disease")


def test_generic_concepts_filtered_out():
    # Level-0/1 concepts like "Medicine"/"Dermatology" must never be hypotheses.
    targets = _targets("Raynaud disease")
    assert "Medicine" not in targets
    assert "Dermatology" not in targets


def test_unknown_topic_returns_empty():
    source = FixtureSource()
    a, links, _stats, _trace = discover_links(source, "nonexistent xyzzy topic")
    assert a is None
    assert links == []


def test_links_have_novelty_and_bridges():
    source = FixtureSource()
    _a, links, _stats, _trace = discover_links(source, "Raynaud disease")
    assert links
    for link in links:
        assert link.bridge_support >= 1
        assert 0.0 < link.novelty <= 1.0
        assert link.score > 0.0
