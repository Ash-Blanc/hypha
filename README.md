# Hypha

**An agentic literature-based discovery engine that surfaces novel, testable,
cross-domain hypotheses — the connections the literature *implies* but no single
paper has stated yet.**

```
   Raynaud disease ──▶ [blood viscosity · platelet aggregation · vasospasm] ──▶ Fish oil
        (A)                              (B, bridges)                            (C, novel)
```

Hypha grows connections through the scientific literature the way fungal hyphae
grow through soil, looking for *undiscovered public knowledge*: a concept **A**
and a concept **C** that are each well-connected to the same intermediate
concepts **B**, yet are almost never discussed together directly. That gap is
the signature of a hypothesis waiting to be made — the same structure behind Don
Swanson's classic discoveries that **fish oil** may help **Raynaud's disease**
and that **magnesium** may help **migraine**, both later supported by trials.

---

## Why this, why now (the thesis)

The agentic-R&D gold rush is crowded at one end and empty at the other.

- **Crowded:** "deep research" agents and AI co-scientists (Google's AI
  co-scientist, Agent Laboratory, AI Scientist, Zochi, Elicit, Undermind…)
  are excellent at telling you **what is already known** — summarizing,
  retrieving, and drafting. They compress the literature.
- **Empty:** almost nothing productizes **what is *not yet* known but is already
  implied** by the literature. Literature-based discovery (LBD) — Swanson's
  1986 insight — is a 40-year-old academic method (SemNet, FieldSHIFT, ARROWSMITH)
  that never became a clean, transparent, bring-your-own-key product.

Three tailwinds make now the moment: (1) open scholarly graphs like
**OpenAlex** (~250M works, free, no key) give planetary-scale co-occurrence data;
(2) LLMs can turn a structural A–B–C link into a mechanistic, testable
hypothesis; (3) the cost of being wrong is low and the payoff of being right is
enormous (drug repurposing, materials, cross-field analogy).

**Hypha's wedge:** a focused, *auditable* discovery engine — every hypothesis
ships with its bridge chain, novelty math, and citations — instead of an opaque
"trust me" research bot. Discovery, not summarization.

> ⚠️ Hypotheses are machine-generated **leads for human review**, not
> established facts. Hypha is a hypothesis *generator*, not a truth oracle.

---

## What it does

Given a topic **A**, Hypha:

1. **Resolves** A to a scholarly concept (with a robust fallback for OpenAlex's
   frozen concept index).
2. **Finds bridges (B):** the concepts that co-occur with A, kept specific via
   IDF-style weighting and generic/homonym filtering.
3. **Expands candidates (C):** the concepts that co-occur with each B but are
   *not* already linked to A.
4. **Scores novelty** by **lift** — how rarely A and C appear together compared
   to chance — so it works for both tiny and huge fields.
5. **Ranks** A–C links by bridge support × path strength × novelty × specificity.
6. **Reasons** each top link into a hypothesis (statement + mechanism +
   experiment) and **self-critiques** confidence.
7. Returns a fully **traced, cited** report.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Offline demo — reproduces Swanson's two textbook discoveries, no network:
hypha discover "Raynaud disease" --offline
hypha discover "Migraine" --offline

# Live discovery over OpenAlex (free, no API key):
hypha discover "type 2 diabetes"
hypha discover "Alzheimer disease" --json

# Web UI + JSON API:
hypha serve            # → http://127.0.0.1:8000
```

### Bring your own LLM key (optional)

With **no key**, Hypha uses a deterministic, fully offline reasoner that
templates a grounded hypothesis from the bridge evidence. Set any one of these
to get richer, model-written mechanisms and experiments:

```bash
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY /
                                 #    OPENROUTER_API_KEY / GROQ_API_KEY
export HYPHA_MODEL=gpt-4o-mini   # optional model override
```

The provider is auto-detected; if a call fails, Hypha degrades gracefully back
to the offline reasoner.

---

## Example (live, real OpenAlex data)

`hypha discover "type 2 diabetes"` surfaces, among others:

> **Type 2 diabetes ↔ Breast cancer** via *insulin, obesity, insulin resistance,
> body-mass index, metformin.*

— a genuine, much-studied cross-domain link (the metformin–cancer connection),
recovered purely from co-occurrence structure.

---

## Architecture

```
              ┌──────────────┐
  topic  ───▶ │ DiscoveryAgent│  plan ▸ resolve ▸ bridge ▸ link ▸ rank ▸
              └──────┬───────┘            hypothesize ▸ critique ▸ report
                     │
     ┌───────────────┼───────────────────┐
     ▼               ▼                   ▼
 ScholarSource   discovery.py        reasoning.py
 (OpenAlex /     ABC engine:         BYOK LLM client
  Fixture)       bridges, lift,      (OpenAI/Anthropic/
                 IDF, filtering      Gemini/…) + offline
                                     fallback
```

| Module | Responsibility |
| --- | --- |
| `hypha/sources/` | Pluggable scholarly backends (`OpenAlexSource`, offline `FixtureSource`). Swap in PubMed/Semantic Scholar by implementing `ScholarSource`. |
| `hypha/discovery.py` | The ABC engine: bridge finding, candidate expansion, lift-based novelty, IDF specificity, generic/homonym filters. |
| `hypha/reasoning.py` | Turns a `BridgeLink` into a `Hypothesis`. LLM (BYOK) or deterministic fallback. |
| `hypha/agent.py` | Orchestration, citation gathering, self-critique, transparent trace. |
| `hypha/api.py` + `hypha/web/` | FastAPI backend and a single-page UI. |
| `hypha/cli.py` | `hypha discover` / `hypha serve`. |

---

## How novelty is scored

For a candidate link A→C reached through bridges B:

- **bridge support** = number of independent B's connecting A and C.
- **path strength** = Σ over bridges of `√(assoc(A,B) · assoc(B,C))`.
- **novelty** via **lift**: `lift = direct(A,C) / expected(A,C)` where
  `expected = works(A)·works(C)/N`. A pair with `lift ≤ 2` (co-occurs no more
  than ~2× chance) is treated as novel; `novelty = 1/(1+lift)`.
- **specificity** (IDF): `1/(1+log10(works(C)))` downweights ubiquitous concepts.
- **score** = `(0.6·pathⁿᵒʳᵐ + 0.4·support) · novelty · specificity`.

---

## Validation

- **Offline fixture** reproduces Swanson's canonical discoveries: the engine
  ranks **fish oil** #1 for *Raynaud disease* and surfaces **magnesium** for
  *migraine* (`tests/test_discovery.py`).
- **Live** OpenAlex runs recover known cross-domain links (e.g. diabetes ↔
  breast cancer via metformin).

```bash
pytest -q          # 13 tests, fully offline
```

---

## Honest limitations & roadmap

Hypha v0 is built on **OpenAlex "concepts"**, which OpenAlex has **frozen** in
favor of Topics. Consequences we handle but don't fully solve:

- Concept co-occurrence is noisy: high-frequency methodology terms and
  homonyms ("Amyloid (mycology)") leak in. We mitigate with IDF weighting and
  generic/homonym stoplists, but precision is bounded by the underlying data.
- Co-occurrence ≠ causation; novelty ≠ correctness. Output is for triage.

**Roadmap (in priority order):**

1. **Typed entities** — swap OpenAlex concepts for PubMed/MeSH + UMLS semantic
   types so C can be constrained (e.g. *disease → drug* for repurposing). This
   is the single biggest precision unlock.
2. **Embedding-based bridge diversity** — reward links supported by
   *semantically diverse* bridges, not synonym clusters.
3. **Closed discovery** — given A *and* C, explain *why* (find the B path).
4. **Time-sliced back-testing** — train on literature up to year *Y*, measure
   how many ranked links became real co-publications after *Y* (a real metric).
5. **Multi-source fusion** — OpenAlex + Semantic Scholar + patents + clinical
   trials.

---

## License

MIT.
