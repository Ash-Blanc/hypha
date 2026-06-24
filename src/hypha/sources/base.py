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

    # --- Optional extensions (duck-typed; enable richer novelty & verification) ---
    # Implement these (taking *names* for A/C so they work with synthetic entities
    # too) to participate in comention-native novelty scoring and better
    # verification. The core ABC engine will discover and use them when
    # novelty_mode="comention" or "hybrid".
    #
    # def comention_count(self, a_name: str, c_name: str) -> int:
    #     """Direct title/abstract (or full-text) co-mention count for A and C.
    #     Return 0 when unknown. Higher values mean the link is less novel.
    #     """
    #     ...
    #
    # def comention_works(self, a_name: str, c_name: str, limit: int = 4) -> list[dict]:
    #     """Representative documents that mention both (for evidence snippets)."""
    #     ...
