"""Tests for LLM provider detection (incl. Fireworks)."""

import pytest

from hypha.reasoning import detect_provider

_ALL_KEYS = [
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "OPENROUTER_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "FIREWORKS_API_KEY",
    "HYPHA_MODEL",
]


@pytest.fixture
def clean_env(monkeypatch):
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_no_keys_means_no_provider(clean_env):
    assert detect_provider() is None


def test_fireworks_detected(clean_env):
    clean_env.setenv("FIREWORKS_API_KEY", "fw_test")
    p = detect_provider()
    assert p is not None
    assert p.name == "fireworks"
    assert p.style == "openai"
    assert p.url == "https://api.fireworks.ai/inference/v1/chat/completions"
    assert p.model.startswith("accounts/fireworks/models/")


def test_fireworks_model_override(clean_env):
    clean_env.setenv("FIREWORKS_API_KEY", "fw_test")
    clean_env.setenv("HYPHA_MODEL", "accounts/fireworks/models/qwen3p6-plus")
    assert detect_provider().model == "accounts/fireworks/models/qwen3p6-plus"


def test_openai_takes_priority_over_fireworks(clean_env):
    clean_env.setenv("OPENAI_API_KEY", "sk-x")
    clean_env.setenv("FIREWORKS_API_KEY", "fw_test")
    assert detect_provider().name == "openai"
