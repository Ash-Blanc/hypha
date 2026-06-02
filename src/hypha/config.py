"""Lightweight, dependency-free ``.env`` loading.

Hypha reads all configuration from environment variables (see ``.env.example``).
To make a local ``.env`` file "just work" without adding a dependency, this
module provides a minimal parser. Values already present in the real
environment always win, so secrets injected by a deployment or CI are never
overridden by a checked-out ``.env``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_LOADED = False


def load_env(path: Optional[str | Path] = None, *, override: bool = False) -> dict[str, str]:
    """Load KEY=VALUE pairs from a ``.env`` file into ``os.environ``.

    Returns the dict of values found. Idempotent: only the first call without an
    explicit ``path`` actually reads from disk (subsequent calls are no-ops).
    """
    global _LOADED
    if path is None:
        if _LOADED:
            return {}
        _LOADED = True
        path = _discover()
        if path is None:
            return {}

    p = Path(path)
    if not p.exists():
        return {}

    found: dict[str, str] = {}
    for raw in p.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.lower().startswith("export "):
            line = line[len("export ") :]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        found[key] = value
        if override or key not in os.environ:
            os.environ[key] = value
    return found


def _discover() -> Optional[Path]:
    """Find a ``.env`` from CWD upward (then the package's project root)."""
    candidates = []
    cwd = Path.cwd()
    candidates.append(cwd / ".env")
    candidates.extend(parent / ".env" for parent in cwd.parents)
    candidates.append(Path(__file__).resolve().parents[2] / ".env")
    for c in candidates:
        if c.exists():
            return c
    return None
