"""OpenAlex-backed scholarly source.

OpenAlex (https://openalex.org) is a free, fully open index of ~250M scholarly
works with no API key required. We use three of its features:

* ``/concepts?search=`` to resolve a topic to a concept.
* ``/works?filter=concepts.id:X&group_by=concepts.id`` to get the concepts that
  co-occur with X (one request returns up to 200 ranked concepts).
* ``/works?filter=concepts.id:X,concepts.id:Y`` whose ``meta.count`` gives the
  number of works mentioning *both* concepts (AND semantics).

All responses are cached on disk so repeated/!offline runs are cheap and
deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional

import httpx

from hypha.models import Concept, SupportingWork

OPENALEX_BASE = "https://api.openalex.org"

_STOP_TERMS = {"disease", "syndrome", "disorder", "the", "of", "and", "s", "a", "an"}


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9]+", text.lower()) if t and t not in _STOP_TERMS}

# OpenAlex concept levels run 0 (broadest, e.g. "Medicine", "Physics") to 5.
# Bridges drawn from very broad concepts are uninformative, so we keep a small
# stoplist of root domains as an extra guard alongside a minimum-level filter.
GENERIC_STOPLIST = {
    "medicine",
    "biology",
    "chemistry",
    "physics",
    "mathematics",
    "computer science",
    "engineering",
    "materials science",
    "psychology",
    "economics",
    "business",
    "geology",
    "political science",
    "sociology",
    "philosophy",
    "art",
    "history",
    "geography",
    "environmental science",
}


class OpenAlexSource:
    name = "openalex"

    def __init__(
        self,
        mailto: Optional[str] = None,
        cache_dir: Optional[str] = None,
        timeout: float = 30.0,
        min_interval: float = 0.1,
    ) -> None:
        self.mailto = mailto or os.environ.get("OPENALEX_MAILTO", "hypha@example.org")
        self.cache_dir = Path(
            cache_dir or os.environ.get("HYPHA_CACHE_DIR", ".hypha_cache")
        )
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        self.min_interval = min_interval
        self._last_call = 0.0
        self._client = httpx.Client(
            timeout=timeout,
            headers={"User-Agent": f"hypha/0.1 (mailto:{self.mailto})"},
        )

    # ----------------------------------------------------------------- http
    def _cache_path(self, url: str) -> Path:
        digest = hashlib.sha256(url.encode()).hexdigest()[:32]
        return self.cache_dir / f"{digest}.json"

    def _get(self, path: str, params: dict[str, Any]) -> dict:
        params = {**params, "mailto": self.mailto}
        request = self._client.build_request("GET", f"{OPENALEX_BASE}{path}", params=params)
        url = str(request.url)
        cache_file = self._cache_path(url)
        if cache_file.exists():
            return json.loads(cache_file.read_text())

        # polite rate limiting
        delta = time.monotonic() - self._last_call
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        resp = self._client.send(request)
        self._last_call = time.monotonic()
        resp.raise_for_status()
        data = resp.json()
        cache_file.write_text(json.dumps(data))
        return data

    # ------------------------------------------------------------ interface
    def resolve_concept(self, topic: str) -> Concept | None:
        # 1) Try the (frozen, sometimes unreliable) concepts search index.
        data = self._get("/concepts", {"search": topic, "per_page": 10})
        results = data.get("results", [])
        topic_terms = {t for t in _tokenize(topic)}
        scored = []
        for c in results:
            terms = _tokenize(c["display_name"])
            overlap = len(topic_terms & terms)
            scored.append((overlap, c.get("level", 0) >= 2, c.get("works_count", 0), c))
        scored.sort(key=lambda t: (t[0], t[1], t[2]), reverse=True)
        if scored and scored[0][0] > 0:
            top = scored[0][3]
            return Concept(
                id=top["id"],
                name=top["display_name"],
                level=top.get("level", 0),
                works_count=top.get("works_count", 0),
                score=float(top.get("relevance_score") or 0.0),
            )

        # 2) Fallback: OpenAlex froze Concepts and the search index misses many
        # entities. Derive A from the concepts that dominate a works search.
        return self._resolve_from_works(topic, topic_terms)

    def _resolve_from_works(self, topic: str, topic_terms: set[str]) -> Concept | None:
        data = self._get(
            "/works",
            {"search": topic, "group_by": "concepts.id", "per_page": 200},
        )
        groups = data.get("group_by", [])
        if not groups:
            return None
        # Prefer a concept whose name overlaps the query (e.g. "Alzheimer's
        # disease" for the query "Alzheimer disease"); fall back to the most
        # frequent non-generic concept.
        best = None
        for g in groups:
            name = g.get("key_display_name", "")
            terms = _tokenize(name)
            overlap = len(topic_terms & terms)
            key = (overlap, g.get("count", 0))
            if best is None or key > best[0]:
                best = (key, g)
        if best is None:
            return None
        g = best[1]
        concept = Concept(id=g["key"], name=g["key_display_name"], score=float(g.get("count", 0)))
        meta = self.hydrate_meta([concept])
        info = meta.get(concept.short_id())
        if info:
            concept.level = info["level"]
            concept.works_count = info["works_count"]
        return concept

    def cooccurring_concepts(self, concept: Concept, limit: int = 200) -> list[Concept]:
        data = self._get(
            "/works",
            {
                "filter": f"concepts.id:{concept.short_id()}",
                "group_by": "concepts.id",
                "per_page": 200,
            },
        )
        groups = data.get("group_by", [])
        out: list[Concept] = []
        for g in groups[:limit]:
            name = g.get("key_display_name", "")
            if not name or name.strip().lower() in GENERIC_STOPLIST:
                continue
            out.append(
                Concept(
                    id=g["key"],
                    name=name,
                    score=float(g.get("count", 0)),
                )
            )
        return out

    def cooccurrence_count(self, a: Concept, c: Concept) -> int:
        data = self._get(
            "/works",
            {
                "filter": f"concepts.id:{a.short_id()},concepts.id:{c.short_id()}",
                "per_page": 1,
            },
        )
        return int(data.get("meta", {}).get("count", 0))

    def hydrate_meta(self, concepts: list[Concept]) -> dict[str, dict]:
        """Fetch concept level + global works_count in batches.

        ``works_count`` is needed for specificity (IDF-style) weighting so that
        globally ubiquitous concepts ("gene expression", "in vitro") do not
        dominate the results.
        """
        meta: dict[str, dict] = {}
        ids = [c.short_id() for c in concepts]
        for i in range(0, len(ids), 50):
            chunk = ids[i : i + 50]
            data = self._get(
                "/concepts",
                {
                    "filter": "ids.openalex:" + "|".join(chunk),
                    "per_page": 50,
                    "select": "id,level,works_count",
                },
            )
            for item in data.get("results", []):
                sid = item["id"].rstrip("/").split("/")[-1]
                meta[sid] = {
                    "level": item.get("level", 0),
                    "works_count": item.get("works_count", 0),
                }
        return meta

    def representative_works(
        self, concepts: list[Concept], limit: int = 2
    ) -> list[SupportingWork]:
        ids = ",".join(f"concepts.id:{c.short_id()}" for c in concepts)
        data = self._get(
            "/works",
            {
                "filter": ids,
                "per_page": limit,
                "sort": "cited_by_count:desc",
                "select": "id,title,publication_year,doi",
            },
        )
        works: list[SupportingWork] = []
        for w in data.get("results", []):
            works.append(
                SupportingWork(
                    id=w["id"],
                    title=w.get("title") or "(untitled)",
                    year=w.get("publication_year"),
                    doi=w.get("doi"),
                    url=w["id"],
                    links=[c.name for c in concepts],
                )
            )
        return works

    def close(self) -> None:
        self._client.close()
