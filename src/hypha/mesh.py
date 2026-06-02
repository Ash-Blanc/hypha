"""MeSH-based semantic typing of concepts.

OpenAlex froze its Concepts taxonomy and never exposed reliable types, which is
the main precision ceiling for literature-based discovery: a "novel" link to a
*method* or *phenomenon* is rarely as actionable as one to a *drug* or a
*disease*. This module restores typing by mapping a concept name to its MeSH
descriptor and reading the first letter of its MeSH **tree number**:

    A Anatomy            B Organisms          C Diseases
    D Chemicals & Drugs  E Techniques         F Psychiatry/Psychology
    G Phenomena          ... (full list in TREE_CATEGORY)

Pipeline (free, no key required; an optional ``NCBI_API_KEY`` raises limits):

1. NCBI E-utilities ``esearch`` on the ``mesh`` db translates a free-text name
   to its canonical MeSH heading (parsed from ``querytranslation``), which is
   robust to plurals/synonyms (e.g. "fish oil" -> "fish oils").
2. The NLM MeSH SPARQL endpoint maps those exact headings to tree numbers.

This makes drug-repurposing a first-class mode: ``--target drug`` keeps only
chemical/drug targets (tree ``D``).
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import httpx

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
MESH_SPARQL = "https://id.nlm.nih.gov/mesh/sparql"

TREE_CATEGORY = {
    "A": "Anatomy",
    "B": "Organisms",
    "C": "Diseases",
    "D": "Chemicals & Drugs",
    "E": "Techniques & Equipment",
    "F": "Psychiatry & Psychology",
    "G": "Phenomena & Processes",
    "H": "Disciplines & Occupations",
    "I": "Anthropology & Social Sciences",
    "J": "Technology & Food",
    "K": "Humanities",
    "L": "Information Science",
    "M": "Named Groups",
    "N": "Health Care",
    "V": "Publication Characteristics",
    "Z": "Geographicals",
}

# Friendly target presets -> the MeSH tree letters they map to.
TARGET_PRESETS: dict[str, Optional[frozenset[str]]] = {
    "any": None,
    "drug": frozenset({"D"}),
    "chemical": frozenset({"D"}),
    "compound": frozenset({"D"}),
    "disease": frozenset({"C"}),
    "condition": frozenset({"C"}),
    "anatomy": frozenset({"A"}),
    "organism": frozenset({"B"}),
    "technique": frozenset({"E"}),
    "intervention": frozenset({"E"}),
    "psychology": frozenset({"F"}),
    "phenomenon": frozenset({"G"}),
    "process": frozenset({"G"}),
}

_HEADING_RE = re.compile(r'"([^"]+)"\[MeSH Terms\]', re.IGNORECASE)


class MeshTyper:
    def __init__(
        self,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 30.0,
        min_interval: float = 0.34,  # ~3 req/s NCBI limit without a key
    ) -> None:
        self.cache_dir = Path(cache_dir or os.environ.get("HYPHA_CACHE_DIR", ".hypha_cache")) / "mesh"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.api_key = api_key or os.environ.get("NCBI_API_KEY")
        self.timeout = timeout
        self.min_interval = 0.11 if self.api_key else min_interval
        self._last = 0.0
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "hypha/0.2"})

    # ------------------------------------------------------------------ util
    def _cache(self, key: str) -> Path:
        return self.cache_dir / (hashlib.sha256(key.encode()).hexdigest()[:24] + ".json")

    def _throttle(self) -> None:
        delta = time.monotonic() - self._last
        if delta < self.min_interval:
            time.sleep(self.min_interval - delta)
        self._last = time.monotonic()

    # -------------------------------------------------------------- pipeline
    def canonical_heading(self, name: str) -> Optional[str]:
        """Translate a free-text concept name to its canonical MeSH heading."""
        cache = self._cache("heading:" + name.lower())
        if cache.exists():
            return json.loads(cache.read_text()).get("heading")
        params = {"db": "mesh", "term": name, "retmode": "json"}
        if self.api_key:
            params["api_key"] = self.api_key
        heading = None
        try:
            self._throttle()
            r = self._client.get(NCBI_ESEARCH, params=params)
            r.raise_for_status()
            trans = r.json().get("esearchresult", {}).get("querytranslation", "")
            m = _HEADING_RE.search(trans)
            if m:
                heading = m.group(1).lower()
        except Exception:  # noqa: BLE001 - typing is best-effort
            heading = None
        cache.write_text(json.dumps({"heading": heading}))
        return heading

    def _tree_letters(self, headings: Iterable[str]) -> dict[str, set[str]]:
        """Batched SPARQL: canonical heading (lowercased) -> set of tree letters."""
        headings = sorted({h for h in headings if h})
        if not headings:
            return {}
        result: dict[str, set[str]] = {}
        # chunk to keep the query small
        for i in range(0, len(headings), 25):
            chunk = headings[i : i + 25]
            cache = self._cache("tree:" + "|".join(chunk))
            if cache.exists():
                for k, v in json.loads(cache.read_text()).items():
                    result[k] = set(v)
                continue
            values = " ".join(f'"{h}"' for h in chunk)
            query = (
                "PREFIX meshv: <http://id.nlm.nih.gov/mesh/vocab#> "
                "PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> "
                "SELECT ?l ?tn WHERE { ?d a meshv:TopicalDescriptor . "
                "?d rdfs:label ?lab . BIND(LCASE(STR(?lab)) AS ?l) "
                f"VALUES ?l {{ {values} }} ?d meshv:treeNumber ?tn . }}"
            )
            local: dict[str, set[str]] = {}
            try:
                self._throttle()
                r = self._client.get(
                    MESH_SPARQL, params={"query": query, "format": "JSON"}
                )
                r.raise_for_status()
                for b in r.json().get("results", {}).get("bindings", []):
                    label = b["l"]["value"]
                    tn = b["tn"]["value"].split("/")[-1]
                    local.setdefault(label, set()).add(tn[0])
            except Exception:  # noqa: BLE001
                local = {}
            cache.write_text(json.dumps({k: sorted(v) for k, v in local.items()}))
            result.update(local)
        return result

    def type_many(self, names: Iterable[str]) -> dict[str, set[str]]:
        """Map each concept name to its set of MeSH category letters (may be empty)."""
        names = list(dict.fromkeys(names))
        name_to_heading = {n: self.canonical_heading(n) for n in names}
        letters = self._tree_letters(name_to_heading.values())
        out: dict[str, set[str]] = {}
        for n, h in name_to_heading.items():
            out[n] = letters.get(h, set()) if h else set()
        return out

    def close(self) -> None:
        self._client.close()


def resolve_target(target: Optional[str]) -> Optional[frozenset[str]]:
    """Map a ``--target`` preset (or a raw tree letter) to MeSH category letters."""
    if not target or target == "any":
        return None
    t = target.strip().lower()
    if t in TARGET_PRESETS:
        return TARGET_PRESETS[t]
    if len(t) == 1 and t.upper() in TREE_CATEGORY:
        return frozenset({t.upper()})
    return None
