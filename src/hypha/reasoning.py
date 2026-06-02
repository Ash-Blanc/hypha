"""Turn structured bridge links into natural-language, testable hypotheses.

Two reasoners are provided:

* :class:`LLMReasoner` - a thin, dependency-light "bring your own key" client
  that auto-detects an available provider from the environment
  (OpenAI / Anthropic / Gemini / OpenRouter / Groq) and asks it to draft a
  mechanism + experiment for each link.
* :class:`FallbackReasoner` - a fully deterministic, offline reasoner that
  templates a hypothesis directly from the bridge evidence. It needs no API
  key, so the engine always produces grounded output.

Both share the same interface: ``reason(link, works) -> Hypothesis``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional

import httpx

from hypha.models import BridgeLink, Hypothesis, SupportingWork

SYSTEM_PROMPT = (
    "You are a rigorous research scientist performing literature-based discovery. "
    "You are given two scientific concepts (A and C) that are connected indirectly "
    "through shared bridge concepts (B) but are rarely or never discussed together "
    "directly. Propose a single, specific, falsifiable hypothesis about how A and C "
    "may be related, the plausible biological/physical mechanism via the bridges, and "
    "a concrete experiment to test it. Be honest about uncertainty. Respond ONLY with "
    "minified JSON: {\"statement\":str,\"mechanism\":str,\"experiment\":str,"
    "\"plausibility\":float between 0 and 1}."
)


@dataclass
class ProviderConfig:
    name: str
    url: str
    model: str
    api_key: str
    style: str  # "openai" | "anthropic" | "gemini"


def detect_provider() -> Optional[ProviderConfig]:
    """Pick the first usable provider from environment variables."""
    env = os.environ
    if env.get("OPENAI_API_KEY"):
        return ProviderConfig(
            "openai",
            "https://api.openai.com/v1/chat/completions",
            env.get("HYPHA_MODEL", "gpt-4o-mini"),
            env["OPENAI_API_KEY"],
            "openai",
        )
    if env.get("ANTHROPIC_API_KEY"):
        return ProviderConfig(
            "anthropic",
            "https://api.anthropic.com/v1/messages",
            env.get("HYPHA_MODEL", "claude-3-5-sonnet-latest"),
            env["ANTHROPIC_API_KEY"],
            "anthropic",
        )
    if env.get("OPENROUTER_API_KEY"):
        return ProviderConfig(
            "openrouter",
            "https://openrouter.ai/api/v1/chat/completions",
            env.get("HYPHA_MODEL", "openai/gpt-4o-mini"),
            env["OPENROUTER_API_KEY"],
            "openai",
        )
    if env.get("GROQ_API_KEY"):
        return ProviderConfig(
            "groq",
            "https://api.groq.com/openai/v1/chat/completions",
            env.get("HYPHA_MODEL", "llama-3.3-70b-versatile"),
            env["GROQ_API_KEY"],
            "openai",
        )
    if env.get("GEMINI_API_KEY") or env.get("GOOGLE_API_KEY"):
        key = env.get("GEMINI_API_KEY") or env["GOOGLE_API_KEY"]
        model = env.get("HYPHA_MODEL", "gemini-1.5-flash")
        return ProviderConfig(
            "gemini",
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            model,
            key,
            "gemini",
        )
    return None


def _link_prompt(link: BridgeLink, works: list[SupportingWork]) -> str:
    bridges = ", ".join(b.name for b in link.bridges)
    cites = "\n".join(f"- {w.title} ({w.year})" for w in works) or "- (none)"
    return (
        f"A = {link.source.name}\n"
        f"C = {link.target.name}\n"
        f"Bridge concepts (B) connecting them: {bridges}\n"
        f"Direct co-mentions of A and C in the literature: {link.direct_cooccurrence}\n"
        f"Independent bridges: {link.bridge_support}\n"
        f"Representative supporting works:\n{cites}\n"
    )


class FallbackReasoner:
    name = "fallback"
    available = True

    def reason(self, link: BridgeLink, works: list[SupportingWork]) -> Hypothesis:
        bridges = [b.name for b in link.bridges]
        bridge_phrase = _join(bridges)
        statement = (
            f"{link.source.name} and {link.target.name} are linked through "
            f"{bridge_phrase}; modulating {link.target.name} may therefore influence "
            f"{link.source.name}."
        )
        mechanism = (
            f"The literature connects {link.source.name} to "
            f"{bridge_phrase}, and independently connects {bridge_phrase} to "
            f"{link.target.name}. These shared intermediates form {link.bridge_support} "
            f"independent two-hop path(s) from {link.source.name} to "
            f"{link.target.name}, yet the two are mentioned together directly in only "
            f"{link.direct_cooccurrence} work(s) - the signature of a connection that "
            f"is implied but not yet stated."
        )
        experiment = (
            f"Test whether intervening on {link.target.name} produces a measurable "
            f"change in a marker of {link.source.name} that is mediated by "
            f"{bridges[0] if bridges else 'the bridge concept'}. Use a controlled "
            f"design with the bridge variable measured as the mediator, and pre-register "
            f"the predicted direction of effect."
        )
        return Hypothesis(
            statement=statement,
            rationale=mechanism,
            mechanism=mechanism,
            experiment=experiment,
            novelty_score=link.novelty,
            plausibility_score=min(1.0, 0.4 + 0.15 * link.bridge_support),
            bridge=link,
            supporting_works=works,
            generated_by="fallback",
        )


class LLMReasoner:
    name = "llm"

    def __init__(self, provider: Optional[ProviderConfig] = None, timeout: float = 60.0):
        self.provider = provider or detect_provider()
        self.timeout = timeout
        self._fallback = FallbackReasoner()

    @property
    def available(self) -> bool:
        return self.provider is not None

    def reason(self, link: BridgeLink, works: list[SupportingWork]) -> Hypothesis:
        if not self.provider:
            return self._fallback.reason(link, works)
        try:
            raw = self._complete(_link_prompt(link, works))
            parsed = _extract_json(raw)
            return Hypothesis(
                statement=parsed["statement"],
                rationale=parsed.get("mechanism", ""),
                mechanism=parsed.get("mechanism", ""),
                experiment=parsed.get("experiment", ""),
                novelty_score=link.novelty,
                plausibility_score=float(parsed.get("plausibility", 0.5)),
                bridge=link,
                supporting_works=works,
                generated_by=f"llm:{self.provider.name}:{self.provider.model}",
            )
        except Exception:  # noqa: BLE001 - degrade gracefully to the offline reasoner
            hyp = self._fallback.reason(link, works)
            hyp.generated_by = "fallback(llm-error)"
            return hyp

    def _complete(self, user_prompt: str) -> str:
        p = self.provider
        assert p is not None
        with httpx.Client(timeout=self.timeout) as client:
            if p.style == "openai":
                resp = client.post(
                    p.url,
                    headers={"Authorization": f"Bearer {p.api_key}"},
                    json={
                        "model": p.model,
                        "messages": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": user_prompt},
                        ],
                        "temperature": 0.7,
                    },
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            if p.style == "anthropic":
                resp = client.post(
                    p.url,
                    headers={
                        "x-api-key": p.api_key,
                        "anthropic-version": "2023-06-01",
                    },
                    json={
                        "model": p.model,
                        "max_tokens": 1024,
                        "system": SYSTEM_PROMPT,
                        "messages": [{"role": "user", "content": user_prompt}],
                    },
                )
                resp.raise_for_status()
                return resp.json()["content"][0]["text"]
            # gemini
            resp = client.post(
                p.url,
                params={"key": p.api_key},
                json={
                    "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
                    "contents": [{"parts": [{"text": user_prompt}]}],
                },
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


def get_reasoner(prefer_llm: bool = True) -> object:
    """Return an LLM reasoner if a key is configured, else the fallback."""
    if prefer_llm:
        llm = LLMReasoner()
        if llm.available:
            return llm
    return FallbackReasoner()


def _join(items: list[str]) -> str:
    if not items:
        return "shared intermediate mechanisms"
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + f" and {items[-1]}"


def _extract_json(text: str) -> dict:
    text = text.strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON found in model output")
    return json.loads(match.group(0))
