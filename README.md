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
7. **Verifies (optional, `--verify`):** independently checks each proposed link
   against the real literature/web and assigns a **verdict** — see below.
8. Returns a fully **traced, cited** report.

## Verification: propose, then check

Concept co-occurrence is a great *idea generator* but a noisy *novelty judge*.
So Hypha separates the two. After the engine proposes links, an independent
**evidence provider** searches for the A–C pair and asks how much the literature
*already* says about it:

| Verdict | Meaning | Effect |
| --- | --- | --- |
| **open** | ~no direct co-mention | genuinely undiscovered → boosted |
| **emerging** | a handful of papers | early signal worth pursuing |
| **established** | widely co-studied | not novel → demoted / flagged |

Hypha then **re-ranks by verified novelty**, so already-known links sink and the
real leads rise. Example (live):

> `hypha discover "type 2 diabetes" --verify` flags **diabetes ↔ breast cancer**
> as `ESTABLISHED` (4,846 direct co-mentions) — correctly catching a link the
> structural score alone would have presented as novel. Meanwhile
> `hypha discover "Raynaud disease" --verify` surfaces **Raynaud ↔ COVID-19** as
> `EMERGING` (7 co-mentions) and ranks it above the established autoimmune links.

Evidence providers are **pluggable / bring-your-own-key**:

- **OpenAlex** *(default, free, no key)* — strict title/abstract AND co-mention
  count + top cited papers.
- **[Parallel](https://parallel.ai) Search API** (`PARALLEL_API_KEY`) —
  web-scale, LLM-optimized evidence excerpts.
- **[GXL Paperclip](https://paperclip.gxl.ai)** (`PAPERCLIP_API_KEY` +
  `pip install gxl_paperclip`) — 11M+ full-text papers, 1M+ clinical trials, and
  FDA/EMA/PMDA regulatory documents. *(experimental)*

Priority when multiple are configured: **Paperclip → Parallel → OpenAlex**
(override with `--evidence`).

## Typed targets (drug repurposing) & closed discovery

**Typed targets** (`--target`) constrain the discovered concept **C** to a MeSH
semantic category, turning open discovery into a focused mode — most usefully
*disease → drug* repurposing. OpenAlex can't type concepts, so Hypha resolves
each candidate's type through MeSH (NCBI term translation → NLM MeSH tree
number; free, key-less, cached):

```bash
uv run hypha discover "Alzheimer disease" --target drug   # keep MeSH chemical/drug targets
uv run hypha discover "psoriasis" --target disease        # comorbidity discovery
```

Presets map to MeSH tree letters: `drug`/`chemical`→D, `disease`→C, `anatomy`→A,
`organism`→B, `technique`→E, `psychology`→F, `phenomenon`→G.

**Closed discovery** (`hypha explain A C`) answers the complementary question —
not "what novel C relates to A?" but "*why* might A and C be related?" — by
finding the bridge concepts **B** shared by both and explaining the implied
mechanism. Great for sanity-checking a repurposing candidate:

```bash
uv run hypha explain "Alzheimer disease" "Metformin"
# → bridges: Diabetes mellitus, Insulin, Inflammation, Oxidative stress,
#            Signal transduction …  (the insulin-signalling rationale), and
#   flags it as already heavily studied (77 direct co-mentions).
```

---

## Quickstart

Hypha uses [uv](https://docs.astral.sh/uv/). `uv run` auto-creates the
environment from `uv.lock` on first use — no manual venv/activate needed.

```bash
uv sync                       # install deps into .venv from the lockfile
cp .env.example .env          # optional — add API keys to unlock richer output

# Offline demo — reproduces Swanson's two textbook discoveries, no network:
uv run hypha discover "Raynaud disease" --offline
uv run hypha discover "Migraine" --offline

# Live discovery over OpenAlex (free, no API key):
uv run hypha discover "type 2 diabetes"
uv run hypha discover "Alzheimer disease" --json

# Propose AND verify each link against the literature, then re-rank:
uv run hypha discover "Raynaud disease" --verify
uv run hypha discover "type 2 diabetes" --verify --evidence parallel   # needs PARALLEL_API_KEY

# Drug-repurposing mode — constrain targets to MeSH chemicals/drugs:
uv run hypha discover "Alzheimer disease" --target drug

# Closed discovery — explain WHY two concepts might be linked (find the B path):
uv run hypha explain "Alzheimer disease" "Metformin"
uv run hypha explain "Raynaud disease" "Fish oil" --offline

# Back-test — would past predictions have come true? (precision@k of discovery)
uv run hypha backtest "multiple sclerosis" --year 2013 --k 10

# Compare OpenAlex ABC discovery vs Paperclip search mining (A/B metrics):
uv run hypha compare "cataract" --verify

# Web UI + JSON API (verify toggle + target dropdown in the UI):
uv run hypha serve            # → http://127.0.0.1:8000
```

> No `uv`? Install it with `curl -fsSL https://astral.sh/uv/install.sh | sh`
> (or `pip install uv`). Or use plain pip: `pip install -e .` then drop the
> `uv run` prefix.

### Bring your own LLM key (optional)

With **no key**, Hypha uses a deterministic, fully offline reasoner that
templates a grounded hypothesis from the bridge evidence. Set any one of these
to get richer, model-written mechanisms and experiments:

```bash
# Hypothesis writing (optional):
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY / GEMINI_API_KEY /
                                 #    FIREWORKS_API_KEY / OPENROUTER_API_KEY / GROQ_API_KEY
export HYPHA_MODEL=gpt-4o-mini   # optional model override

# Example — use Fireworks AI:
export FIREWORKS_API_KEY=fw_...
export HYPHA_MODEL=accounts/fireworks/models/gpt-oss-120b

# Verification evidence providers (optional; OpenAlex is the free default):
export PARALLEL_API_KEY=...      # parallel.ai Search API
export PAPERCLIP_API_KEY=...     # gxl.ai Paperclip (also: pip install gxl_paperclip)
```

Everything is auto-detected; if any call fails, Hypha degrades gracefully back
to the free defaults (offline reasoner / OpenAlex evidence). All of these can
also be put in a local `.env` (copy `.env.example`) — Hypha loads it
automatically, and real environment variables always take precedence. See
[`.env.example`](.env.example) for the full list.

---

## Example (live, real OpenAlex data)

`uv run hypha discover "type 2 diabetes"` surfaces, among others:

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
| `hypha/discovery.py` | The ABC engine: bridge finding, candidate expansion, lift-based novelty, IDF specificity, filters, + closed discovery (`explain_link`). |
| `hypha/mesh.py` | `MeshTyper`: free MeSH semantic typing (NCBI translation → NLM MeSH tree) powering `--target`. |
| `hypha/evidence.py` | Verification layer: `EvidenceProvider`s (OpenAlex / Parallel / Paperclip / fixture) + `Verifier` (verdict + re-rank). |
| `hypha/reasoning.py` | Turns a `BridgeLink` into a `Hypothesis`. LLM (BYOK) or deterministic fallback. |
| `hypha/agent.py` | Orchestration, citation gathering, self-critique, verification, transparent trace. |
| `hypha/backtest.py` | Time-sliced validation harness (`hypha backtest`): precision@k of discovery. |
| `hypha/config.py` | Zero-dependency `.env` loader (real env always wins). |
| `hypha/api.py` + `hypha/web/` | FastAPI backend and a single-page UI. |
| `hypha/cli.py` | `hypha discover` / `explain` / `compare` / `backtest` / `serve`. |
| `hypha/compare.py` | Side-by-side OpenAlex vs Paperclip with nonsense/actionable metrics. |
| `hypha/paperclip_client.py` | Paperclip REST client (`PAPERCLIP_API_KEY`). |

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
  breast cancer via metformin), and `--verify` correctly labels them
  `established` rather than novel.

```bash
uv run pytest      # 37 tests, fully offline
```

### Does it actually *predict* discoveries? (`hypha backtest`)

A time-sliced back-test runs discovery using **only literature up to a split
year Y**, then checks how many of the top-k *genuinely-novel* predictions
actually emerged as real co-publications afterwards — an honest **precision@k**
of discovery:

```bash
uv run hypha backtest "multiple sclerosis" --year 2013 --k 10
```

What it reveals (and we report honestly): pure concept co-occurrence
**over-selects already-known links** on heavily-studied fields — the structural
top-k are mostly already connected, so true novelty lives deeper and is noisier.
This is exactly why **verification and typing are not optional**, and the
back-test is the metric that drives that work. See [`PITCH.md`](PITCH.md) and
[`docs/design-partners.md`](docs/design-partners.md) for how this anchors the
fundraising and validation story.

---

## Honest limitations & roadmap

Hypha v0 is built on **OpenAlex "concepts"**, which OpenAlex has **frozen** in
favor of Topics. Consequences we handle but don't fully solve:

- Concept co-occurrence is noisy: high-frequency methodology terms and
  homonyms ("Amyloid (mycology)") leak in. We mitigate with IDF weighting and
  generic/homonym stoplists, but precision is bounded by the underlying data.
- Co-occurrence ≠ causation; novelty ≠ correctness. Output is for triage.

**Shipped:** verification layer (OpenAlex/Parallel/Paperclip providers);
MeSH-typed targets (`--target drug`) for repurposing; closed discovery
(`hypha explain A C`); time-sliced back-testing (`hypha backtest`).

**Roadmap (in priority order):**

1. **Comention-native novelty** — make the engine rank by the independent
   title/abstract co-mention signal (used by `--verify`/back-test), not just
   concept-pair lift, so genuinely-novel links surface in the top-k. (The
   back-test shows this is the key precision lever.)
2. **Sharper drug typing** — restrict `--target drug` to MeSH pharmacologic
   subtrees / "PA" actions (exclude biopolymers like RNA), and add UMLS
   semantic types for finer control.
3. **Deeper verification** — promote Paperclip full-text + clinical-trial +
   FDA signals into the verdict (e.g. "0 papers but 2 active trials"), and add
   an LLM `supported / untested / refuted` read over the retrieved evidence.
4. **Embedding-based bridge diversity** — reward links supported by
   *semantically diverse* bridges, not synonym clusters.
5. **Multi-source fusion** — OpenAlex + Semantic Scholar + patents + clinical
   trials (Paperclip already unlocks much of this).

---

## License

MIT.
