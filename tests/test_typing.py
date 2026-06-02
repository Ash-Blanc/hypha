"""Tests for MeSH target typing (offline via a stub typer)."""

from hypha.discovery import DiscoveryConfig, discover_links
from hypha.mesh import TARGET_PRESETS, resolve_target
from hypha.sources.fixture import FixtureSource


class StubTyper:
    """Deterministic stand-in for MeshTyper used in tests."""

    def __init__(self, mapping):
        self.mapping = mapping

    def type_many(self, names):
        return {n: set(self.mapping.get(n, set())) for n in names}


def test_resolve_target_presets():
    assert resolve_target("drug") == frozenset({"D"})
    assert resolve_target("disease") == frozenset({"C"})
    assert resolve_target("any") is None
    assert resolve_target(None) is None
    assert resolve_target("C") == frozenset({"C"})  # raw tree letter
    assert resolve_target("nonsense") is None


def test_target_presets_cover_drug_and_disease():
    assert TARGET_PRESETS["drug"] == frozenset({"D"})
    assert TARGET_PRESETS["disease"] == frozenset({"C"})


def test_type_filter_keeps_only_matching_targets():
    source = FixtureSource()
    typer = StubTyper(
        {"Fish oil": {"D"}, "Eicosapentaenoic acid": {"D"}, "Serotonin": set()}
    )
    cfg = DiscoveryConfig(target_categories=frozenset({"D"}))
    _a, links, _s, trace = discover_links(source, "Raynaud disease", cfg, typer=typer)
    targets = [l.target.name for l in links]
    assert "Fish oil" in targets
    assert "Serotonin" not in targets  # untyped -> excluded under a type filter
    assert any(ev.step == "type" for ev in trace)


def test_type_filter_fails_open_when_typing_unavailable():
    source = FixtureSource()
    typer = StubTyper({})  # types nothing -> simulate typing unavailable
    cfg = DiscoveryConfig(target_categories=frozenset({"D"}))
    _a, links, _s, trace = discover_links(source, "Raynaud disease", cfg, typer=typer)
    # Should not collapse to empty; the unconstrained results remain.
    assert links
    assert any(ev.step == "type" for ev in trace)


def test_no_typer_means_no_type_filtering():
    source = FixtureSource()
    cfg = DiscoveryConfig(target_categories=frozenset({"D"}))
    _a, links, _s, _t = discover_links(source, "Raynaud disease", cfg, typer=None)
    assert links  # typer absent -> filter silently skipped
