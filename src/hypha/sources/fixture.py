"""An offline, deterministic source backed by a small co-occurrence graph.

This makes the discovery engine fully testable without network access and is
used by the unit tests (it reproduces Swanson's Raynaud->fish-oil and
migraine->magnesium discoveries).
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Optional

from hypha.models import Concept, SupportingWork

_DEFAULT_FIXTURE = Path(__file__).resolve().parents[3] / "data" / "fixture.json"


class FixtureSource:
    name = "fixture"

    def __init__(self, path: Optional[str | Path] = None) -> None:
        if path is not None:
            raw = Path(path).read_text()
        elif _DEFAULT_FIXTURE.exists():
            raw = _DEFAULT_FIXTURE.read_text()
        else:  # installed as package data
            raw = resources.files("data").joinpath("fixture.json").read_text()
        blob = json.loads(raw)
        self._concepts: dict[str, dict] = blob["concepts"]
        self._edges: dict[tuple[str, str], int] = {}
        for key, count in blob["edges"].items():
            a, b = key.split("|")
            self._edges[(a, b)] = count
            self._edges[(b, a)] = count

    def _concept(self, cid: str) -> Concept:
        c = self._concepts[cid]
        return Concept(
            id=c["id"],
            name=c["name"],
            level=c.get("level", 0),
            works_count=c.get("works_count", 0),
        )

    def resolve_concept(self, topic: str) -> Concept | None:
        t = topic.strip().lower()
        best = None
        for cid, c in self._concepts.items():
            name = c["name"].lower()
            if t == name:
                return self._concept(cid)
            if t in name or name in t:
                if best is None or c.get("works_count", 0) > self._concepts[best].get("works_count", 0):
                    best = cid
        return self._concept(best) if best else None

    def cooccurring_concepts(self, concept: Concept, limit: int = 200) -> list[Concept]:
        sid = concept.short_id()
        out: list[Concept] = []
        for (a, b), count in self._edges.items():
            if a == sid:
                c = self._concept(b)
                c.score = float(count)
                out.append(c)
        out.sort(key=lambda c: c.score, reverse=True)
        return out[:limit]

    def cooccurrence_count(self, a: Concept, c: Concept) -> int:
        return self._edges.get((a.short_id(), c.short_id()), 0)

    def comention_count(self, a_name: str, c_name: str) -> int:
        """For the offline fixture we treat the synthetic co-occurrence count
        as a proxy for "direct" mentions. This lets novelty_mode=hybrid/comention
        and back-test-style logic run deterministically without network, while
        the canonical Raynaud/Migraine fixture tests continue to pass under
        default (concept_lift) mode.
        """
        a = self.resolve_concept(a_name)
        c = self.resolve_concept(c_name)
        if a is None or c is None:
            return 0
        return self.cooccurrence_count(a, c)

    def representative_works(
        self, concepts: list[Concept], limit: int = 2
    ) -> list[SupportingWork]:
        names = " + ".join(c.name for c in concepts)
        return [
            SupportingWork(
                id=f"fixture://{'-'.join(c.short_id() for c in concepts)}",
                title=f"Representative study linking {names}",
                year=2011,
                url=None,
                links=[c.name for c in concepts],
            )
        ][:limit]
