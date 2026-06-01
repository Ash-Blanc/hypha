"""Core data models for the Hypha discovery engine.

The vocabulary follows Swanson's ABC model of literature-based discovery:

* **A** - the source concept the user starts from (e.g. "Raynaud disease").
* **B** - intermediate / bridge concepts that co-occur with A in the
  literature (e.g. "blood viscosity", "platelet aggregation").
* **C** - target concepts that co-occur with some B but are *not* directly
  linked to A. An A-C pair with strong bridge support but little or no direct
  co-occurrence is a candidate piece of "undiscovered public knowledge".
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class Concept(BaseModel):
    """A scholarly concept / topic returned by a source."""

    id: str
    name: str
    level: int = 0
    works_count: int = 0
    score: float = 0.0

    def short_id(self) -> str:
        """Return the bare OpenAlex-style id (last path segment)."""
        return self.id.rstrip("/").split("/")[-1]


class SupportingWork(BaseModel):
    """A concrete publication that grounds part of a bridge chain."""

    id: str
    title: str
    year: Optional[int] = None
    doi: Optional[str] = None
    url: Optional[str] = None
    links: list[str] = Field(default_factory=list)


class BridgeLink(BaseModel):
    """A candidate hidden A-C link with the B concepts that connect them."""

    source: Concept
    target: Concept
    bridges: list[Concept] = Field(default_factory=list)
    direct_cooccurrence: int = 0
    bridge_support: int = 0
    path_strength: float = 0.0
    novelty: float = 0.0
    score: float = 0.0

    def explain(self) -> str:
        names = ", ".join(b.name for b in self.bridges[:5])
        return (
            f"{self.source.name} -> [{names}] -> {self.target.name} "
            f"(bridges={self.bridge_support}, direct={self.direct_cooccurrence}, "
            f"score={self.score:.3f})"
        )


class Hypothesis(BaseModel):
    """A natural-language, testable hypothesis built from a bridge link."""

    statement: str
    rationale: str
    mechanism: str = ""
    experiment: str = ""
    novelty_score: float = 0.0
    plausibility_score: float = 0.0
    bridge: BridgeLink
    supporting_works: list[SupportingWork] = Field(default_factory=list)
    generated_by: str = "fallback"


class TraceEvent(BaseModel):
    """A single step in the transparent agent trace."""

    step: str
    detail: str
    data: dict = Field(default_factory=dict)


class DiscoveryReport(BaseModel):
    """The full output of a discovery run."""

    topic: str
    source_concept: Optional[Concept] = None
    hypotheses: list[Hypothesis] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    source_name: str = "openalex"
    reasoner: str = "fallback"
    stats: dict = Field(default_factory=dict)
