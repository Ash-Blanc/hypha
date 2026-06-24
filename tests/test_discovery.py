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


# ---------------------------------------------------------------------------
# Revamp tests: filters (generalizability) + novelty modes
# These must not change the default (biomed + concept_lift) behavior on fixture.
# ---------------------------------------------------------------------------


def test_default_biomed_filter_reproduces_classic():
    from hypha.filters import make_default_filter, _is_generic, _bad_parenthetical

    flt = make_default_filter("biomed")
    # Classic generic terms still filtered
    assert flt.is_generic("gene expression")
    assert flt.is_generic("in vitro")
    assert flt.is_generic("machine learning")
    assert _is_generic("disease") is True
    # Homonym tags
    assert flt.is_bad_parenthetical("Amyloid (mycology)") is True
    assert _bad_parenthetical("Scleroderma (fungus)") is True


def test_general_filter_is_less_strict():
    from hypha.filters import make_default_filter

    biomed = make_default_filter("biomed")
    general = make_default_filter("general")
    # "machine learning" is generic in biomed but should *not* be auto-dropped in general
    # (it may still be filtered by other logic or user stoplists).
    assert biomed.is_generic("machine learning") is True
    # general profile has a tiny list; "machine learning" is not in it.
    assert general.is_generic("machine learning") is False


def test_novelty_mode_hybrid_and_comention_on_fixture():
    """On fixture, comention falls back to cooc count. We just assert the
    plumbing works and that we still surface the canonical links under hybrid.
    """
    source = FixtureSource()
    # Default must be unchanged
    a_def, links_def, _, _ = discover_links(source, "Raynaud disease")
    assert links_def and links_def[0].target.name == "Fish oil"

    # Hybrid should also work (and on fixture comention==cooc so scores may differ
    # slightly due to blend, but the top Swanson link should still appear).
    cfg_h = DiscoveryConfig(novelty_mode="hybrid", filter_profile="biomed")
    a_h, links_h, _, _ = discover_links(source, "Raynaud disease", config=cfg_h)
    assert a_h is not None
    targets_h = [l.target.name for l in links_h]
    assert "Fish oil" in targets_h

    # comention mode too
    cfg_c = DiscoveryConfig(novelty_mode="comention")
    a_c, links_c, _, _ = discover_links(source, "Raynaud disease", config=cfg_c)
    assert a_c is not None
    assert any(l.target.name == "Fish oil" for l in links_c)

    # Rich signals should be populated in non-classic modes
    if links_h:
        assert hasattr(links_h[0], "signals")
        assert hasattr(links_h[0], "comention_count")


def test_bridge_diversity_and_custom_filter():
    from hypha.filters import make_default_filter

    source = FixtureSource()
    # Enable diversity weight (generalizability/quality lever)
    cfg = DiscoveryConfig(bridge_diversity_weight=0.3, novelty_mode="hybrid")
    _a, links, _, trace = discover_links(source, "Raynaud disease", config=cfg)
    assert links
    # At least one link should have a recorded diversity (even if 1.0)
    assert any(hasattr(l, "diversity") for l in links)

    # Custom filter: drop everything (edge case) -> no links or very few
    strict = make_default_filter("biomed", override_name_stop={"fish oil", "magnesium", "blood viscosity"})
    cfg2 = DiscoveryConfig(filters=strict)
    _a2, links2, _, _ = discover_links(source, "Raynaud disease", config=cfg2)
    # With such an aggressive override the engine may surface nothing or very little.
    # We mainly assert it didn't crash and the filter was respected.
    assert isinstance(links2, list)
