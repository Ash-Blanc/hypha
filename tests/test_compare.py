"""Tests for compare harness and Paperclip parsing (offline)."""

from hypha.compare import CompareRow, _is_nonsense_target, _summarize, run_compare
from hypha.paperclip_client import parse_papers


SAMPLE_OUTPUT = """
  1. Lanosterol treatment of cataract
     pmc123 · pmc · 2020
     https://doi.org/10.1000/example
     "Lanosterol dissolves protein aggregates in lens cells."

  2. N-acetylcarnosine eye drops for cataract
     pmc456 · pmc · 2019
     https://doi.org/10.1000/example2
"""


def test_parse_papers_extracts_titles():
    papers = parse_papers(SAMPLE_OUTPUT)
    assert len(papers) == 2
    assert "Lanosterol" in papers[0]["title"]


def test_nonsense_detector():
    assert _is_nonsense_target("Medicine", level=0)
    assert _is_nonsense_target("Government (linguistics)")
    assert not _is_nonsense_target("Fish oil", level=3)


def test_summarize_metrics():
    rows = [
        CompareRow("openalex", "Medicine", "", 1.0, "—", 0, True, False),
        CompareRow("openalex", "Fish oil", "", 2.0, "—", 2, False, True),
    ]
    s = _summarize(rows)
    assert s["nonsense_rate"] == 0.5
    assert s["actionable_rate"] == 0.5


def test_compare_offline_openalex_only():
    report = run_compare("Raynaud disease", max_rows=3, offline=True)
    assert report.summary["openalex"]["count"] >= 1
    assert report.summary["paperclip"]["count"] == 0
    assert any(r.pipeline == "openalex" for r in report.rows)
