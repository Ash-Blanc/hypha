"""Smoke tests for the FastAPI layer (offline, no network)."""

from fastapi.testclient import TestClient

from hypha.api import app

client = TestClient(app)


def test_index_served():
    r = client.get("/")
    assert r.status_code == 200
    assert "Hypha" in r.text


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_discover_offline_endpoint():
    r = client.post(
        "/api/discover",
        json={"topic": "Raynaud disease", "offline": True, "max_hypotheses": 3},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["topic"] == "Raynaud disease"
    assert body["hypotheses"]
    assert any("Fish oil" in h["statement"] for h in body["hypotheses"])
    assert body["trace"]
