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

__version__ = "0.2.0"

__all__ = [
    "BridgeLink",
    "Concept",
    "DiscoveryReport",
    "EvidenceItem",
    "Hypothesis",
    "SupportingWork",
    "__version__",
]
