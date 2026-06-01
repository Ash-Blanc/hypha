"""End-to-end tests for the discovery agent + fallback reasoner (offline)."""

from hypha.agent import DiscoveryAgent, run_discovery
from hypha.reasoning import FallbackReasoner
from hypha.sources.fixture import FixtureSource


def _offline_agent() -> DiscoveryAgent:
    return DiscoveryAgent(
        source=FixtureSource(), reasoner=FallbackReasoner(), prefer_llm=False
    )


def test_agent_produces_hypotheses_with_evidence():
    report = _offline_agent().discover("Raynaud disease")
    assert report.hypotheses
    top = report.hypotheses[0]
    assert "Fish oil" in top.statement
    assert top.experiment
    assert top.mechanism
    assert top.supporting_works  # citations attached
    assert top.generated_by == "fallback"


def test_report_is_serialisable_and_traced():
    report = _offline_agent().discover("Migraine")
    blob = report.model_dump()
    assert blob["topic"] == "Migraine"
    assert any(ev["step"] == "rank" for ev in blob["trace"])
    assert report.stats["results"] >= 1


def test_run_discovery_offline_helper():
    report = run_discovery("Raynaud disease", offline=True, max_hypotheses=3)
    assert len(report.hypotheses) <= 3
    assert report.source_name == "fixture"


def test_critique_flags_single_bridge_links():
    report = _offline_agent().discover("Migraine")
    for h in report.hypotheses:
        if h.bridge.bridge_support <= 1:
            assert "critique" in h.rationale
