"""Tests for Paperclip HTTP client."""

import pytest

from hypha.evidence import PaperclipEvidenceProvider
from hypha.paperclip_client import PaperclipAuthError, PaperclipClient


def test_client_requires_key(monkeypatch):
    monkeypatch.delenv("PAPERCLIP_API_KEY", raising=False)
    client = PaperclipClient(api_key=None)
    assert not client.available
    with pytest.raises(PaperclipAuthError):
        client.execute("search", "test")


def test_evidence_provider_unavailable_without_key(monkeypatch):
    monkeypatch.delenv("PAPERCLIP_API_KEY", raising=False)
    provider = PaperclipEvidenceProvider(api_key=None)
    assert not provider.available
    res = provider.gather("A", "B")
    assert res.direct_hits == -1
