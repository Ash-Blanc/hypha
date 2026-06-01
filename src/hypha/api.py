"""FastAPI backend + static web UI for Hypha."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from hypha.agent import run_discovery
from hypha.reasoning import detect_provider

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(
    title="Hypha",
    description="Agentic literature-based discovery engine.",
    version="0.1.0",
)


class DiscoverRequest(BaseModel):
    topic: str
    offline: bool = False
    max_hypotheses: int = 6


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (WEB_DIR / "index.html").read_text()


@app.get("/api/health")
def health() -> dict:
    provider = detect_provider()
    return {
        "status": "ok",
        "reasoner": provider.name if provider else "fallback (no API key set)",
        "model": provider.model if provider else None,
    }


@app.post("/api/discover")
def discover(req: DiscoverRequest) -> JSONResponse:
    report = run_discovery(
        req.topic,
        offline=req.offline,
        max_hypotheses=max(1, min(req.max_hypotheses, 12)),
    )
    return JSONResponse(report.model_dump())


@app.get("/api/discover")
def discover_get(
    topic: str = Query(...),
    offline: bool = Query(False),
    max_hypotheses: int = Query(6),
) -> JSONResponse:
    return discover(
        DiscoverRequest(topic=topic, offline=offline, max_hypotheses=max_hypotheses)
    )
