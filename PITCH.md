# Hypha — pitch & go-to-market

> **Everyone is building AI that summarizes what science already knows.
> Hypha finds what science doesn't know yet — but already implies.**

Hypha is a **hypothesis engine** (not a research summarizer). It mines the
global literature for cross-domain links that are strongly implied — multiple
independent intermediate "bridges" — yet never stated (near-zero direct
co-mention), then **verifies** each against the literature and proposes a
testable experiment. Everything is auditable: every hypothesis ships with its
bridge chain, novelty math, and citations.

Flagship wedge: **drug repurposing** (disease → existing drug).

---

## The problem

Scientific knowledge is expanding faster than any lab can read, verify, and
connect. The bottleneck is no longer ideas, data, or compute — it's
**attention and connection**. ~$2T/yr of global R&D produces millions of
findings a year that are never linked to the findings that would make them
actionable. Don Swanson showed in 1986 that some of those missing links are
real discoveries hiding in plain sight (fish oil → Raynaud's; magnesium →
migraine). That method — literature-based discovery (LBD) — stayed academic.

## The insight & product

Connect concept **A** to concept **C** through the intermediate concepts **B**
they both relate to, and surface A–C pairs with strong multi-bridge support but
no direct literature link. Then run the loop:

**propose → verify → type → explain**

- **Propose:** ABC discovery over OpenAlex (250M works), lift-based novelty.
- **Verify:** independent literature/web search assigns each link a verdict
  (open / emerging / established) and re-ranks by *verified* novelty. Providers:
  OpenAlex (free), Parallel Search, GXL Paperclip (11M full-text + trials + FDA).
- **Type:** MeSH semantic typing constrains targets (e.g. `--target drug`) —
  this is what turns it into a repurposing engine.
- **Explain:** closed discovery — given A and C, reconstruct the mechanism.

## Why now

1. Open planetary-scale scholarly graphs (OpenAlex) — free, no key.
2. LLMs turn a structural link into a mechanism + experiment.
3. Repurposing economics: a validated lead is worth millions; a query costs cents.

## Why us

A working, tested, **auditable** system already exists — verification, MeSH
typing, closed discovery, a quantitative back-test, CLI + API + UI, BYOK,
near-zero COGS. Most "AI for science" teams have a deck; we have the loop.

---

## Validation: we measure discovery honestly

We built a **time-sliced back-test** (`hypha backtest`): run discovery using
only literature up to year *Y*, then check how many of the top-k
*genuinely-novel* predictions actually emerged as real co-publications after *Y*
(precision@k).

What it already tells us (and we say this out loud):

- Hypha **rediscovers** Swanson's textbook links from first principles.
- Pure concept co-occurrence **over-selects already-known links** on
  heavily-studied diseases — which is precisely why **verification and typing
  are not optional**. The back-test is the metric that drives that R&D, and
  it's the number we'll put in front of partners and investors.

This is the most fundable artifact we can own: a repeatable, quantitative answer
to "does it actually discover?" — plus design-partner wet-lab outcomes (below).

---

## Business model

- **Seats** for researchers (discovery + explain + verify).
- **Per-program / success-linked** pricing for repurposing engagements.
- **API metering** as the long-term "discovery layer" every research org calls.
- BYOK keeps early COGS near zero.

## Moat

Models commoditize. Our compounding asset is the **validation data flywheel** —
*which hypotheses were tested and what happened* — plus auditability/trust and
multi-source fusion (full text + trials + FDA + patents). The back-test and
partner outcomes feed ranking, which improves hit-rate, which wins more
partners.

## The ask

Pre-seed/seed to fund (1) validation partnerships, (2) a domain scientist, and
(3) GTM into repurposing foundations + biotech R&D. Use of funds is weighted to
*proving discovery*, not headcount.

---

## The one objection — and our answer

> "Aren't these just plausible-sounding correlations?"

Yes, unverified co-occurrence is noisy — so we don't ship that. We ship
*propose → verify*, we **measure** precision@k with the back-test, and we
validate top candidates with design partners who can run an assay. We're honest
that open discovery on saturated fields is hard; the value is in
under-explored/young fields, cross-domain analogies, and typed repurposing —
and we have the eval to prove where it works.

See [`README.md`](README.md) for the working system and
[`docs/design-partners.md`](docs/design-partners.md) for the partner playbook.
