"""Thin HTTP client for GXL Paperclip (https://paperclip.gxl.ai).

The published ``gxl-paperclip`` wheel (0.1.x) is a CLI-only package; the Python
SDK described in the docs may ship in a newer release. Hypha talks to the same
REST endpoint the CLI uses: ``POST /api/cli/execute``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import httpx

DEFAULT_BASE_URL = "https://paperclip.gxl.ai"


@dataclass
class ExecuteResult:
    output: str
    exit_code: int = 0
    elapsed_ms: Optional[int] = None
    result_id: Optional[str] = None
    raw: dict[str, Any] | None = None


class PaperclipError(Exception):
    """Base error for Paperclip HTTP failures."""


class PaperclipAuthError(PaperclipError):
    pass


class PaperclipClient:
    """Minimal Paperclip API client using an API key or OAuth bearer token."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 120.0,
    ) -> None:
        self.api_key = api_key or os.environ.get("PAPERCLIP_API_KEY")
        self.base_url = (base_url or os.environ.get("PAPERCLIP_BASE_URL", DEFAULT_BASE_URL)).rstrip(
            "/"
        )
        self.timeout = timeout

    @property
    def available(self) -> bool:
        return bool(self.api_key)

    def _headers(self) -> dict[str, str]:
        if not self.api_key:
            raise PaperclipAuthError("PAPERCLIP_API_KEY is not set")
        # Docs: API keys via env; MCP uses X-API-Key — send both for compatibility.
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "X-API-Key": self.api_key,
        }

    def execute(self, command: str, raw: str = "") -> ExecuteResult:
        url = f"{self.base_url}/api/cli/execute"
        payload = {"command": command, "raw": raw}
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(url, json=payload, headers=self._headers())
        if resp.status_code in (401, 403):
            raise PaperclipAuthError(f"Paperclip authentication failed ({resp.status_code})")
        if resp.status_code == 429:
            raise PaperclipError("Paperclip rate limit (429)")
        if resp.status_code != 200:
            detail = resp.text[:500]
            try:
                detail = resp.json().get("detail", detail)
            except Exception:  # noqa: BLE001
                pass
            raise PaperclipError(f"Paperclip error {resp.status_code}: {detail}")
        data = resp.json()
        return ExecuteResult(
            output=data.get("output", "") or "",
            exit_code=int(data.get("exit_code", 0)),
            elapsed_ms=data.get("elapsed_ms"),
            result_id=data.get("result_id"),
            raw=data,
        )

    def search(self, query: str, *, limit: int = 20, source: Optional[str] = None) -> ExecuteResult:
        parts = [f'"{query}"' if " " in query else query]
        if limit:
            parts.append(f"-n {limit}")
        if source:
            parts.append(f"-s {source}")
        return self.execute("search", " ".join(parts))

    def count_comention(self, a_name: str, c_name: str) -> int:
        """Approximate direct co-mention count via a tight boolean search."""
        query = f'"{a_name}" "{c_name}"'
        res = self.search(query, limit=1)
        # If the server returns structured counts in output, parse them; else use hits.
        m = re.search(r"(\d[\d,]*)\s+(?:results?|papers?|hits?)", res.output, re.I)
        if m:
            return int(m.group(1).replace(",", ""))
        papers = parse_papers(res.output)
        return len(papers)


def parse_papers(output: str) -> list[dict[str, str]]:
    """Parse formatted search output into paper records (mirrors gxl_paperclip CLI)."""
    text = re.sub(r"\033\[[0-9;]*m", "", output)
    records: list[dict[str, str]] = []
    entries = re.split(r"\n(?=\s+\d+\.\s)", text)
    for entry in entries:
        lines = entry.strip().split("\n")
        if not lines:
            continue
        m = re.match(r"\s*(\d+)\.\s+(.+)", lines[0])
        if not m:
            continue
        record: dict[str, str] = {"title": m.group(2).strip()}
        for line in (ln.strip() for ln in lines[1:] if ln.strip()):
            if line.startswith("https://"):
                record.setdefault("url", line)
            elif line.startswith("doi:"):
                record.setdefault("url", f"https://doi.org/{line[4:]}")
            elif line.startswith('"'):
                record["abstract"] = line.strip('"')
            elif "\u00b7" in line or "·" in line:
                sep = "\u00b7" if "\u00b7" in line else "·"
                parts = [p.strip() for p in line.split(sep)]
                if parts:
                    record["id"] = parts[0]
                if len(parts) >= 2:
                    record["source"] = parts[1]
                if len(parts) >= 3:
                    record["date"] = parts[2]
            elif "authors" not in record:
                record["authors"] = line
        records.append(record)
    return records
