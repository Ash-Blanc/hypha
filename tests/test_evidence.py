"""Tests for the verification layer (offline / deterministic)."""

from hypha.agent import DiscoveryAgent, run_discovery
from hypha.discovery import discover_links
from hypha.evidence import (
    EvidenceResult,
    FixtureEvidenceProvider,
    Verifier,
    VerifierConfig,
    get_evidence_provider,
)
from hypha.models import EvidenceItem
from hypha.reasoning import FallbackReasoner
from hypha.sources.fixture import FixtureSource


class StubProvider:
    """Returns a fixed co-mention count to exercise verdict thresholds."""

    name = "stub"

    def __init__(self, hits: int):
        self.hits = hits

    def gather(self, a_name, c_name, max_items=4):
        items = [EvidenceItem(title=f"{a_name}/{c_name} study", source="stub")] if self.hits else []
        return EvidenceResult(direct_hits=self.hits, items=items, provider=self.name)


def _one_hypothesis():
    source = FixtureSource()
    _a, links, _s, _t = discover_links(source, "Raynaud disease")
    return FallbackReasoner().reason(links[0], [])


def test_verdict_open_for_zero_hits():
    h = Verifier(StubProvider(0)).verify(_one_hypothesis())
    assert h.verdict == "open"
    assert h.verified_novelty and h.verified_novelty > 0.9


def test_verdict_emerging_for_few_hits():
    h = Verifier(StubProvider(10)).verify(_one_hypothesis())
    assert h.verdict == "emerging"
    assert 0.4 < h.verified_novelty < 0.9
    assert h.evidence


def test_verdict_established_for_many_hits():
    h = Verifier(StubProvider(5000)).verify(_one_hypothesis())
    assert h.verdict == "established"
    assert h.verified_novelty <= 0.4


def test_unknown_count_is_unverified():
    h = Verifier(StubProvider(-1)).verify(_one_hypothesis())
    assert h.verdict == "unverified"


def test_fixture_provider_marks_link_open():
    # Fish oil never co-occurs directly with Raynaud in the fixture graph.
    provider = FixtureEvidenceProvider()
    res = provider.gather("Raynaud disease", "Fish oil")
    assert res.direct_hits == 0


def test_agent_verify_offline_sets_verdicts_and_reranks():
    agent = DiscoveryAgent(
        source=FixtureSource(),
        reasoner=FallbackReasoner(),
        prefer_llm=False,
        evidence_provider=FixtureEvidenceProvider(),
    )
    report = agent.discover("Raynaud disease", verify=True)
    assert report.hypotheses
    assert all(h.verdict != "unverified" for h in report.hypotheses)
    assert "verdicts" in report.stats
    assert any(ev.step == "verify" for ev in report.trace)


def test_run_discovery_offline_verify_helper():
    report = run_discovery("Migraine", offline=True, verify=True, max_hypotheses=3)
    assert report.hypotheses
    assert all(h.verdict == "open" for h in report.hypotheses)


def test_factory_offline_returns_fixture():
    assert get_evidence_provider(offline=True).name == "fixture"
