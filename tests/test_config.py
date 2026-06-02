"""Tests for the zero-dependency .env loader."""

import os

from hypha.config import load_env


def test_load_env_parses_and_does_not_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text(
        "# a comment\n"
        "\n"
        "export OPENAI_API_KEY=sk-from-file\n"
        'HYPHA_MODEL="gpt-4o-mini"\n'
        "PARALLEL_API_KEY='pk-quoted'\n"
        "EXISTING_VAR=should-not-win\n"
    )
    monkeypatch.setenv("EXISTING_VAR", "already-set")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("HYPHA_MODEL", raising=False)

    found = load_env(env)

    assert found["OPENAI_API_KEY"] == "sk-from-file"
    assert found["HYPHA_MODEL"] == "gpt-4o-mini"  # quotes stripped
    assert found["PARALLEL_API_KEY"] == "pk-quoted"
    # real env wins over the file
    assert os.environ["EXISTING_VAR"] == "already-set"
    # new keys are injected
    assert os.environ["OPENAI_API_KEY"] == "sk-from-file"


def test_load_env_missing_file_is_noop(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_load_env_override(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text("FOO=fromfile\n")
    monkeypatch.setenv("FOO", "fromenv")
    load_env(env, override=True)
    assert os.environ["FOO"] == "fromfile"


def test_example_file_documents_key_vars():
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env.example"
    text = example.read_text()
    for var in [
        "OPENAI_API_KEY",
        "PARALLEL_API_KEY",
        "PAPERCLIP_API_KEY",
        "NCBI_API_KEY",
        "OPENALEX_MAILTO",
        "HYPHA_CACHE_DIR",
    ]:
        assert var in text
