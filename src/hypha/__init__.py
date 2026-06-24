"""Hypha - an agentic literature-based discovery engine.

Hypha grows connections through the scientific literature (like fungal hyphae
through soil) to surface *undiscovered public knowledge*: novel, testable,
cross-domain hypotheses that no single paper has yet stated, following Don
Swanson's ABC model of literature-based discovery.
"""

from hypha.models import (
    BridgeLink,
    Concept,
    DiscoveryReport,
    EvidenceItem,
    Hypothesis,
    SupportingWork,
)

# Core revamp abstractions (advanced use / extension)
from hypha.discovery import DiscoveryConfig, discover_links  # noqa: F401
from hypha.filters import RelevanceFilter, make_default_filter, make_openai_embedder  # noqa: F401
from hypha.novelty import ComentionNoveltyScorer, ConceptLiftNoveltyScorer  # noqa: F401

__version__ = "0.3.0"  # revamp: generalizability (filters/profiles) + novelty (comention/hybrid + diversity) first

__all__ = [
    "BridgeLink",
    "ComentionNoveltyScorer",
    "Concept",
    "ConceptLiftNoveltyScorer",
    "DiscoveryConfig",
    "DiscoveryReport",
    "EvidenceItem",
    "Hypothesis",
    "RelevanceFilter",
    "SupportingWork",
    "discover_links",
    "make_default_filter",
    "make_openai_embedder",
    "__version__",
]
