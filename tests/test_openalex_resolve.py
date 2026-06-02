"""Tests for OpenAlex concept resolution (reject generic level-0 sources)."""

from hypha.sources.openalex import GENERIC_STOPLIST, MIN_SOURCE_LEVEL, OpenAlexSource, _is_generic_source


class FakeOpenAlex(OpenAlexSource):
    """OpenAlex source with canned HTTP responses (no network)."""

    def __init__(self, responses: dict[str, dict]) -> None:
        super().__init__(cache_dir="/tmp/hypha-test-cache-unused")
        self._responses = responses
        self._client = None  # type: ignore[assignment]

    def _get(self, path: str, params: dict) -> dict:
        key = path
        if path == "/works" and "search" in params:
            key = "works_search"
        elif path == "/concepts" and "search" in params:
            key = "concepts_search"
        elif path == "/concepts" and "filter" in params:
            key = "concepts_filter"
        return self._responses[key]


def test_rejects_generic_medicine_from_concepts_search():
    source = FakeOpenAlex(
        {
            "concepts_search": {
                "results": [
                    {
                        "id": "https://openalex.org/C71924100",
                        "display_name": "Medicine",
                        "level": 0,
                        "works_count": 86_000_000,
                        "relevance_score": 100,
                    },
                    {
                        "id": "https://openalex.org/C123",
                        "display_name": "Cataract",
                        "level": 2,
                        "works_count": 50_000,
                        "relevance_score": 50,
                    },
                ]
            },
            "works_search": {"group_by": []},
        }
    )
    c = source.resolve_concept("cataract early stages")
    assert c is not None
    assert c.name == "Cataract"
    assert c.level >= MIN_SOURCE_LEVEL


def test_works_fallback_skips_generic_without_overlap():
    source = FakeOpenAlex(
        {
            "concepts_search": {"results": []},
            "works_search": {
                "group_by": [
                    {"key": "https://openalex.org/C71924100", "key_display_name": "Medicine", "count": 9000},
                    {"key": "https://openalex.org/C123", "key_display_name": "Cataract", "count": 120},
                ]
            },
            "concepts_filter": {
                "results": [
                    {"id": "https://openalex.org/C71924100", "level": 0, "works_count": 86_000_000},
                    {"id": "https://openalex.org/C123", "level": 2, "works_count": 50_000},
                ]
            },
        }
    )
    c = source.resolve_concept("cataract")
    assert c is not None
    assert c.name == "Cataract"


def test_is_generic_source_helper():
    assert _is_generic_source("Medicine", 0)
    assert "medicine" in GENERIC_STOPLIST
