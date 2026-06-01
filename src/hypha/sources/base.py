"""Abstract interface every scholarly source must implement.

The discovery engine is source-agnostic: it only needs to (1) resolve a
free-text topic to a concept, (2) ask which concepts co-occur with a given
concept, (3) count direct co-occurrence between two concepts, and (4) fetch a
few representative works for a set of concepts. Any backend (OpenAlex,
Semantic Scholar, PubMed, a local corpus, or an offline fixture) can satisfy
this contract.
"""

from __future__ import annotations

from typing import Protocol

from hypha.models import Concept, SupportingWork


class ScholarSource(Protocol):
    name: str

    def resolve_concept(self, topic: str) -> Concept | None:
        """Map a free-text topic to the best matching concept."""
        ...

    def cooccurring_concepts(self, concept: Concept, limit: int = 200) -> list[Concept]:
        """Return concepts that co-occur with ``concept`` across works.

        The returned ``score`` field holds the co-occurrence count.
        """
        ...

    def cooccurrence_count(self, a: Concept, c: Concept) -> int:
        """Return the number of works mentioning both ``a`` and ``c``."""
        ...

    def representative_works(
        self, concepts: list[Concept], limit: int = 2
    ) -> list[SupportingWork]:
        """Return a few works that mention all of ``concepts``."""
        ...
