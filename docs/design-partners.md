# Hypha — design-partner playbook

The make-or-break for a discovery tool is proving it surfaces **real,
non-obvious, testable** hypotheses. So pick partners by one criterion above all:

> **Can they cheaply test a hypothesis and tell us whether it was any good?**

Tight feedback loop > logo prestige.

---

## Ideal partner profile (ICP)

- A **focused problem area** (a named disease, target, or material class).
- An **in-house way to test cheaply** (assay, registry, clinical/real-world data).
- A **scientist champion** who personally feels the "what do we try next?" pain.
- **Willingness to share outcomes** (ratings + what happened when they tested).

~5–8 partners is plenty.

## Who to target (ranked)

**Tier 1 — best (cheap, fast validation + real pain)**
- **Drug-repurposing groups & rare-disease / patient-advocacy foundations.**
  A named disease, strong motivation to find *any* approved drug, often existing
  assays/registries, and they publicize wins. Repurposing is the perfect wedge:
  the target is an approved drug → fast, low-risk to test.
- **Academic translational labs with a wet lab** in a focused area (oncology,
  neuro, immunology). They can screen our top candidates; co-authored validation
  is proof for both sides.

**Tier 2 — high value, slower**
- **Small/mid biotech & TechBio** discovery / indication-expansion teams (pay +
  credibility, longer cycles).
- **Computational-biology / bioinformatics cores** already doing knowledge-graph
  / target-ID work (stress-test methodology, integrate fast).

**Tier 3 — expansion after biomed proof**
- Materials / battery / catalysis / agritech R&D (cross-domain analogy shines).
- IP / patent-landscape & R&D-strategy teams (whitespace finding).

**Avoid as first partners:** big-pharma central R&D (procurement death-by-pilot)
and pure-summarization buyers (wrong job-to-be-done).

---

## How to lock them in

Structure matters more than the pitch. Use a **free-but-committed pilot** that
builds the moat.

1. **Concrete & bounded.** "Give us your disease/target. We return 20 ranked,
   verified, cited hypotheses with proposed experiments. You rate them, and
   ideally test the top 2–3."
2. **Trade access for data + feedback, not cash (yet).** Free pilot in exchange
   for (a) expert ratings per hypothesis and (b) outcomes of anything tested.
   That data is the defensible flywheel and the fundraising proof.
3. **Co-define success up front** (in writing) so the pilot can't be "meh":
   e.g. *"≥3 of 20 hypotheses rated novel & worth testing"* and/or *"1 advances
   to an assay."*
4. **Land with a champion, expand to the org.** Make them a hero (co-authored
   validation / a board-deck win), then expand to seats/program deals.
5. **One-page agreement** (template below): scope, data rights, IP (hypotheses
   are leads, not claims — they own what they test; we retain anonymized
   feedback/outcome rights), confidentiality, convert-to-paid trigger.
6. **Urgency + scarcity:** "founding design partners" get roadmap input,
   preferential pricing, and co-publication.

### Outreach that works

Cold email with a **personalized live result**: run Hypha on their exact disease
and paste the top 3 *verified* hypotheses (verdict + bridges + citations). One
real result beats any deck.

---

## Pilot agreement (one-page template)

```
HYPHA DESIGN-PARTNER PILOT — <Partner> × Hypha

Scope:        Hypha delivers up to <N> ranked, verified, cited hypotheses for
              the area "<disease/target>", each with bridges + a proposed test.
Term:         <fixed-scope pilot; ends on delivery + evaluation>.
Partner does: rate each hypothesis (novel? plausible? would you test?); share
              outcomes of any hypotheses tested.
Success:      <e.g. >=3/N rated "novel & worth testing"; >=1 advanced to assay>.
Fees:         $0 for the pilot. Convert to <seat / per-program> on success.
Data & IP:    Partner owns all results of experiments they run. Hypotheses are
              research leads, not validated claims. Hypha may use anonymized
              ratings & outcomes to improve ranking. Mutual NDA applies.
Logo/quote:   Partner permits "design partner" reference + an optional quote.
```

## Outcome-tracking template (one row per hypothesis)

| id | A (topic) | C (target) | bridges | verdict | novelty | partner_novel? | partner_plausible? | would_test? | tested? | outcome | notes |
|----|-----------|------------|---------|---------|---------|----------------|--------------------|-------------|---------|--------|-------|
| 1  |           |            |         |         |         | y/n            | y/n                | y/n         | y/n     | hit/miss/pending | |

Aggregate KPIs to report to investors:
- **% rated novel & worth testing** (expert precision)
- **# advanced to an assay / trial**
- **hit-rate of tested hypotheses**
- **back-test precision@k** (`hypha backtest`) as the offline complement
