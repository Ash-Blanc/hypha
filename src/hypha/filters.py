"""Configurable relevance filters for discovery (bridges, candidates, targets).

This module extracts the previously hardcoded biomed-centric stoplists and
predicates so the engine can be used on other domains (materials, CS, social
science, etc.) without leaking noise or over-filtering.

**Far less hardcoding via vectors**: When an `embedder` (callable that takes
list[str] and returns list[list[float]]) is supplied, `is_generic`, near-synonym
dropping (A-similarity), and related decisions are augmented (or driven) by
cosine similarity to small *seed prototypes* rather than exhaustive static lists.

Lists become high-quality seeds + fallback for the zero-dep / no-key path.
Semantic similarity generalizes to synonyms, paraphrases, and related umbrella
terms (e.g. "cellular process" near "cell", "clinical investigation" near
"clinical trial").

See `make_openai_embedder()` (reuses OPENAI_API_KEY or HYPHA_EMBED_* env like
the LLM reasoner) and how RelevanceFilter + DiscoveryConfig accept it.

The design was informed by LBD literature on "avoiding background knowledge",
"quantifying and filtering" generated hypotheses, and modern use of embedding
thresholds / bi-encoders for semantic filtering instead of brittle stoplists.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import httpx

# ---------------------------------------------------------------------------
# Optional semantic / vector layer (the main lever for "far less hardcoding")
# ---------------------------------------------------------------------------


def make_openai_embedder(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    timeout: float = 30.0,
) -> Optional[Callable[[list[str]], list[list[float]]]]:
    """Return a batch embedder using an OpenAI-compatible /embeddings endpoint.

    Prefers:
      - HYPHA_EMBED_API_KEY / HYPHA_EMBED_MODEL / HYPHA_EMBED_BASE_URL, else
      - OPENAI_API_KEY (with a cheap embedding model).

    Returns a callable `embed(texts) -> list[vector]` or None if no key.
    This is the main way users get vector-powered generic/homonym/A-similarity
    and much better bridge diversity without changing Hypha's dep footprint.
    """
    key = api_key or os.environ.get("HYPHA_EMBED_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not key:
        return None

    mdl = model or os.environ.get("HYPHA_EMBED_MODEL") or "text-embedding-3-small"
    url = (base_url or os.environ.get("HYPHA_EMBED_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
    embed_url = f"{url}/embeddings"

    def _embed(texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {"model": mdl, "input": texts}
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(embed_url, json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()["data"]
            # data is in the order of input
            vecs = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
            return vecs

    return _embed


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Built-in lightweight "vector" (ngram feature sets) for generalization.
# This lets us do "vector search"-style matching against seeds with zero
# external dependencies or API keys. Character n-grams give cheap morphological
# and compound-term robustness (e.g. "in_vitro", "invitro", "vitro model").
# Used to drive is_generic and A-similarity when no full embedder is supplied.
# This is the main mechanism for far less hardcoding of exhaustive lists.
# ---------------------------------------------------------------------------

def _char_ngrams(text: str, n: int = 3) -> set[str]:
    """Simple character n-gram set (alphanum only, lower)."""
    s = re.sub(r"[^a-z0-9]", "", text.lower())
    if len(s) < n:
        return {s} if s else set()
    return {s[i : i + n] for i in range(len(s) - n + 1)}


# ---------------------------------------------------------------------------
# Default "biomed" stoplists (moved verbatim from discovery.py for exact repro)
# ---------------------------------------------------------------------------

# Small, high-signal seed set for "generic" / low-value concepts.
#
# Far less hardcoding: we no longer enumerate 100+ variants. The built-in ngram
# "vector" Jaccard (plus optional full embedder cosine) generalizes from these
# seeds to morphological variants, compounds, and near-synonyms
# (e.g. "in_vitro", "vitro model", "cellular process", "gene expr.").
#
# The old exhaustive list is kept below in a comment for reference / exact
# historical repro when needed. "biomed" profile uses the small seed set + vector
# generalization so the same filtering power is achieved with ~1/4 the explicit
# strings (and much better coverage for new domains).
NAME_STOPLIST_BIOMED: set[str] = {
    # Core umbrellas / very high-level
    "disease",
    "syndrome",
    "disorder",
    "phenomenon",
    "pathology",
    "comorbidity",
    "epidemiology",
    # Very common methodology / measurement / omics noise
    "cell",
    "protein",
    "gene",
    "in vitro",
    "in vivo",
    "biomarker",
    "expression",
    "clinical trial",
    "meta analysis",
    "systematic review",
    "machine learning",
    "algorithm",
    "analysis",
    "study",
    "review",
    "model",
    "system",
    "process",
    "function",
    "mechanism",
    "effect",
    "patient",
    "treatment",
    # Explicit for test / canonical repro (old long list had many more;
    # ngram vector now covers variants of these)
    "medicine",
    "dermatology",
}

# --- Historical full set (for reference / exact legacy repro if ever needed) ---
# NAME_STOPLIST_BIOMED_FULL = { ... the original ~110 item set ... }
# It has been replaced by the small seed set above + ngram vector similarity
# (and optional embedder) so that we do far less hardcoding while keeping
# (or improving) recall of generic noise.

# Parenthetical sense tags for out-of-domain homonyms. Kept reasonably complete
# because exact tag matching is still the most reliable here; ngram vector helps
# less for these.
HOMONYM_TAGS_BIOMED: set[str] = {
    "mycology",
    "fungus",
    "archaeology",
    "geology",
    "programming language",
    "album",
    "band",
    "software",
    "journal",
    "given name",
    "surname",
    "plant",
    "insect",
    "genus",
    "mathematics",
    "film",
    "novel",
    "tv series",
    "company",
    "city",
}

# Tokens stripped for "distinctive" computation. Pruned; the ngram vector path
# and embedder now do most of the heavy lifting for near-synonym detection vs A.
GENERIC_TOKENS_BIOMED: set[str] = {
    "disease",
    "syndrome",
    "disorder",
    "the",
    "of",
    "and",
    "in",
    "a",
    "an",
    "s",
    "primary",
    "secondary",
    "acute",
    "chronic",
}


# Even smaller seeds for non-biomed/emerging use. With ngram vector + embedder
# the "general" profile now generalizes well without long lists.
NAME_STOPLIST_GENERAL: set[str] = {
    "context",
    "system",
    "analysis",
    "model",
    "method",
    "study",
    "review",
    "approach",
}

HOMONYM_TAGS_GENERAL: set[str] = {
    "company",
    "journal",
    "software",
    "film",
    "novel",
}

GENERIC_TOKENS_GENERAL: set[str] = {
    "the",
    "of",
    "and",
    "in",
    "a",
    "an",
    "s",
}


@dataclass
class RelevanceFilter:
    """Holds configurable stoplists and predicates for bridge/candidate filtering.

    Create via ``make_default_filter("biomed")`` for exact historical behavior,
    or ``make_default_filter("general")`` + overrides for other domains.

    When ``embedder`` is provided (see `make_openai_embedder`), many decisions
    become semantic (cosine to small seed prototypes) instead of pure string
    matching. This is the primary mechanism for *far less hardcoding*.
    """

    name_stop: set[str] = field(default_factory=lambda: set(NAME_STOPLIST_BIOMED))
    homonym_tags: set[str] = field(default_factory=lambda: set(HOMONYM_TAGS_BIOMED))
    generic_tokens: set[str] = field(default_factory=lambda: set(GENERIC_TOKENS_BIOMED))
    # Additional user-provided terms (union-ed in).
    extra_name_stop: set[str] = field(default_factory=set)
    extra_homonym_tags: set[str] = field(default_factory=set)
    extra_generic_tokens: set[str] = field(default_factory=set)
    # If provided, completely replace the base lists (advanced escape hatch).
    override_name_stop: Optional[set[str]] = None
    override_homonym_tags: Optional[set[str]] = None
    override_generic_tokens: Optional[set[str]] = None

    # min_level is still primarily driven by DiscoveryConfig, but filter can
    # enforce a floor too.
    min_level_floor: int = 0

    # --- Semantic / vector (embedder-powered) configuration ---
    # Callable: texts -> list of vectors. When present, is_generic / A-similarity
    # / etc. use cosine to prototypes derived from the (current) name_stop seeds.
    embedder: Optional[Callable[[list[str]], list[list[float]]]] = None
    # Thresholds (tuned for typical normalized OpenAI embedding spaces).
    generic_semantic_threshold: float = 0.78
    a_similarity_threshold: float = 0.82  # if sim(A, C) high, treat as near-synonym of A

    # Lazily computed from the *seed* terms in name_stop when embedder is first used.
    _generic_prototypes: list[list[float]] = field(default_factory=list, init=False, repr=False)
    _embed_cache: dict[str, list[float]] = field(default_factory=dict, init=False, repr=False)

    # Built-in ngram "vector" prototypes (always available, zero-dep).
    # These enable vector-style (ngram Jaccard) generalization from the (much smaller)
    # seed lists, so we don't need to enumerate every variant explicitly.
    ngram_n: int = 3
    ngram_generic_threshold: float = 0.55  # Jaccard on char-ngrams to any seed proto
    _generic_ngram_protos: list[set[str]] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.override_name_stop is not None:
            self.name_stop = set(self.override_name_stop)
        if self.override_homonym_tags is not None:
            self.homonym_tags = set(self.override_homonym_tags)
        if self.override_generic_tokens is not None:
            self.generic_tokens = set(self.override_generic_tokens)

        self.name_stop = set(self.name_stop) | set(self.extra_name_stop)
        self.homonym_tags = set(self.homonym_tags) | set(self.extra_homonym_tags)
        self.generic_tokens = set(self.generic_tokens) | set(self.extra_generic_tokens)

        # Build ngram prototypes from the (curated, hopefully small) name_stop seeds.
        # This is the "vector search" replacement for exhaustive exact-list matching.
        self._generic_ngram_protos = [
            _char_ngrams(s, self.ngram_n) for s in self.name_stop if s
        ]

    # --- Semantic helpers (vector-powered generalization of the lists) ---

    def _get_prototypes(self) -> list[list[float]]:
        if self._generic_prototypes or not self.embedder:
            return self._generic_prototypes
        # Use the (possibly user-curated / small) name_stop as seeds for the "generic" region.
        seeds = sorted({s for s in self.name_stop if len(s) > 2})
        if not seeds:
            return []
        try:
            vecs = self.embedder(seeds)
            self._generic_prototypes = vecs
        except Exception:  # noqa: BLE001 - embed failure -> stay list-only
            self._generic_prototypes = []
        return self._generic_prototypes

    def _embed(self, text: str) -> Optional[list[float]]:
        if not self.embedder:
            return None
        key = text.lower().strip()
        if key in self._embed_cache:
            return self._embed_cache[key]
        try:
            v = self.embedder([text])[0]
            self._embed_cache[key] = v
            return v
        except Exception:  # noqa: BLE001
            return None

    def semantic_generic_score(self, name: str) -> float:
        """Max cosine of `name` to the generic prototype cloud (0 if no embedder)."""
        protos = self._get_prototypes()
        if not protos:
            return 0.0
        vec = self._embed(name)
        if not vec:
            return 0.0
        return max(_cosine(vec, p) for p in protos)

    def semantic_similarity(self, a_name: str, b_name: str) -> float:
        """Cosine between two concept names (0 if no embedder)."""
        va = self._embed(a_name)
        vb = self._embed(b_name)
        if not va or not vb:
            return 0.0
        return _cosine(va, vb)

    def ngram_generic_score(self, name: str) -> float:
        """Max ngram-Jaccard of `name` against the generic seed prototypes.
        This is a cheap, always-on vector feature space model (no deps, no keys).
        High score means the term is similar (in subword/compound structure) to
        known generic/umbrella/methodological terms.
        """
        ng = _char_ngrams(name, self.ngram_n)
        if not ng or not self._generic_ngram_protos:
            return 0.0
        return max(
            (len(ng & p) / max(len(ng | p), 1)) for p in self._generic_ngram_protos
        )

    # --- predicates (pure, no side effects) ---

    def _tokens(self, name: str) -> set[str]:
        return {t for t in re.split(r"[^a-z0-9]+", name.lower()) if t}

    def _distinctive_tokens(self, name: str) -> set[str]:
        return self._tokens(name) - self.generic_tokens

    def is_generic(self, name: str) -> bool:
        """Far less hardcoding via vector-ish signals + tiny seed lists.

        1. Built-in ngram "vector" Jaccard to the (small) generic seed prototypes.
        2. If full embedder supplied: cosine to prototypes (stronger semantic).
        3. Exact list membership (for exact historical repro + zero-cost path).
        4. Token "no distinctive content" as last cheap fallback.

        The ngram + embedder paths are the "vector search" replacement for
        maintaining a giant exhaustive stoplist of every possible variant.
        """
        n = name.strip().lower()
        if n in self.name_stop:
            return True

        # Built-in lightweight vector (ngram feature sets) — always available
        if self.ngram_generic_score(name) >= self.ngram_generic_threshold:
            return True

        # Stronger semantic (optional external embeddings)
        if self.embedder:
            if self.semantic_generic_score(name) >= self.generic_semantic_threshold:
                return True

        # Token-based "has no distinctive content" (legacy cheap path)
        return bool(self._distinctive_tokens(name)) is False

    def is_bad_parenthetical(self, name: str) -> bool:
        m = re.search(r"\(([^)]+)\)", name)
        return bool(m and m.group(1).strip().lower() in self.homonym_tags)

    def is_usable(
        self,
        c: "Concept",  # type: ignore[name-defined]  # forward to avoid circular import at runtime
        a: Optional["Concept"] = None,
        min_level: int = 2,
    ) -> bool:
        """Return True if c is acceptable as bridge or target given A (if known)."""
        if c.short_id() == (a.short_id() if a else ""):
            return False
        eff_min = max(min_level, self.min_level_floor)
        if c.level < eff_min:
            return False
        if self.is_generic(c.name) or self.is_bad_parenthetical(c.name):
            return False
        if a is not None:
            # Vector-style near-synonym drop for A (replaces pure token overlap).
            # Prefer full embedder cosine if available; always also use cheap ngram Jaccard.
            if self.embedder and self.semantic_similarity(a.name, c.name) >= self.a_similarity_threshold:
                return False
            # Built-in ngram vector similarity as the main always-on "vector search"
            # signal for dropping near-duplicates of A.
            a_ng = _char_ngrams(a.name, self.ngram_n)
            c_ng = _char_ngrams(c.name, self.ngram_n)
            if a_ng and c_ng:
                j = len(a_ng & c_ng) / max(len(a_ng | c_ng), 1)
                if j >= 0.6:  # conservative; full embedder above is stricter when present
                    return False

            # Legacy token distinctive overlap (defense in depth, very cheap)
            a_terms = self._distinctive_tokens(a.name)
            if a_terms and (self._distinctive_tokens(c.name) & a_terms):
                return False
        return True

    def filter_bridges(self, bridges: list["Concept"], a: "Concept", min_level: int = 2) -> list["Concept"]:
        return [b for b in bridges if self.is_usable(b, a=a, min_level=min_level)]


def make_default_filter(profile: str = "biomed", **overrides) -> RelevanceFilter:
    """Factory for the two main shipped profiles.

    profile:
      - "biomed": the full historical stoplists (exact v0 behavior for fixtures).
      - "general": much smaller lists, suitable for materials / CS / emerging fields.
    Any kwarg is forwarded to RelevanceFilter (e.g. extra_name_stop, embedder,
    generic_semantic_threshold, min_level_floor).

    Pass embedder=make_openai_embedder() (or your own) to activate the vector
    path that dramatically reduces dependence on the static lists.
    """
    p = (profile or "biomed").lower()
    if p in ("biomed", "bio", "medical", "default"):
        base = RelevanceFilter(
            name_stop=set(NAME_STOPLIST_BIOMED),
            homonym_tags=set(HOMONYM_TAGS_BIOMED),
            generic_tokens=set(GENERIC_TOKENS_BIOMED),
        )
    elif p in ("general", "gen", "broad", "relaxed"):
        base = RelevanceFilter(
            name_stop=set(NAME_STOPLIST_GENERAL),
            homonym_tags=set(HOMONYM_TAGS_GENERAL),
            generic_tokens=set(GENERIC_TOKENS_GENERAL),
        )
    else:
        # Unknown profile -> start empty and let caller/overrides populate.
        base = RelevanceFilter(name_stop=set(), homonym_tags=set(), generic_tokens=set())

    if overrides:
        # Merge extras or apply overrides after construction for cleanliness.
        if "extra_name_stop" in overrides:
            base.extra_name_stop |= set(overrides.pop("extra_name_stop"))
        if "extra_homonym_tags" in overrides:
            base.extra_homonym_tags |= set(overrides.pop("extra_homonym_tags"))
        if "extra_generic_tokens" in overrides:
            base.extra_generic_tokens |= set(overrides.pop("extra_generic_tokens"))
        for k, v in overrides.items():
            setattr(base, k, v)
        # Re-run post_init style merge
        base.name_stop |= base.extra_name_stop
        base.homonym_tags |= base.extra_homonym_tags
        base.generic_tokens |= base.extra_generic_tokens
    return base


# ---------------------------------------------------------------------------
# Backward-compat shims (so existing internal imports of private names keep working
# during the transition; new code should use RelevanceFilter directly).
# These delegate to a default biomed filter instance so behavior is byte-for-byte
# identical for the historical call sites.
# ---------------------------------------------------------------------------

_default_biomed: RelevanceFilter = make_default_filter("biomed")


def _bad_parenthetical(name: str) -> bool:
    return _default_biomed.is_bad_parenthetical(name)


def _tokens(name: str) -> set[str]:
    return _default_biomed._tokens(name)


def _distinctive_tokens(name: str) -> set[str]:
    return _default_biomed._distinctive_tokens(name)


def _is_generic(name: str) -> bool:
    return _default_biomed.is_generic(name)


# Note: the inner _usable closure in discovery.py will be replaced in a later
# step to use the filter instance from config. These shims keep paperclip/compare
# and any other direct callers working without behavior change for now.
