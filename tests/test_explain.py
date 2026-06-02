"""Tests for closed discovery (explain A <-> C) using the offline fixture."""

from hypha.agent import DiscoveryAgent, run_explain
from hypha.discovery import explain_link
from hypha.reasoning import FallbackReasoner
from hypha.sources.fixture import FixtureSource


def test_explain_finds_shared_bridges():
    source = FixtureSource()
    a, c, link, trace = explain_link(source, "Raynaud disease", "Fish oil")
    assert a and c and link
    names = {b.name for b in link.bridges}
    # Fish oil connects to Raynaud via these hemorheologic intermediates.
    assert {"Blood viscosity", "Platelet aggregation", "Vasospasm"} & names
    assert link.bridge_support >= 1


def test_explain_unknown_concept_returns_none():
    source = FixtureSource()
    a, c, link, _ = explain_link(source, "Raynaud disease", "nonexistent xyz")
    assert link is None


def test_agent_explain_builds_hypothesis():
    agent = DiscoveryAgent(
        source=FixtureSource(), reasoner=FallbackReasoner(), prefer_llm=False
    )
    report = agent.explain("Migraine", "Magnesium")
    assert report.hypotheses
    h = report.hypotheses[0]
    assert "Magnesium" in h.statement
    assert h.mechanism and h.experiment
    assert any(ev.step == "explain" for ev in report.trace)


def test_run_explain_offline_helper():
    report = run_explain("Raynaud disease", "Fish oil", offline=True)
    assert report.hypotheses
    assert report.topic == "Raynaud disease \u2194 Fish oil"
