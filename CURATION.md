# Curation Workflow

This is the human half of the dataset deliverable. `data_prep.py` cannot
verify facts — it can only reject anything that isn't tagged as verified
and sourced. Everything below is what makes a record eligible to pass that
gate.

## Why this exists

The assignment requires every example to come from authoritative materials
and forbids synthetic or web-scraped content without rigorous manual
verification. Every record in `raw_curated.jsonl` was fetched and read from
a live official page or PDF before being written — nothing here is
generated from parametric memory. The M0 scaffold originally shipped 12
**template** records (`verified: false`, empty `source_doc`/`source_url`)
as a shape reference; running `python data_prep.py` against them at the
time correctly failed with a validation error per record, proving the
gate rejects unsourced content before any real curation began.

## The four authoritative source families

Every record's `source_doc` and `source_url` must point to one of these,
or an equivalent official publication:

| Area | Source family | What was actually cited |
|---|---|---|
| `business_registration` | Business Registration Service (BRS) / eCitizen | brs.go.ke's own FAQ, Companies Registry, forms, and practice-note pages; nairobi.go.ke for the county Unified Business Permit |
| `tax_obligations` | Kenya Revenue Authority (KRA) | kra.go.ke pages on TOT, VAT, PIN registration, TCC, eTIMS, withholding/installment/advance/rental/capital gains tax, and private rulings |
| `loan_eligibility` | Central Bank of Kenya (CBK) | centralbank.go.ke on CRB regulation, bank supervision, and digital credit disclosure, plus hustlerfund.go.ke (the government's own SME lending product) and ke.kcbgroup.com (a CBK-licensed bank's own product page) |
| `mobile_money` | Safaricom Daraja / Lipa na M-Pesa | safaricom.co.ke's own Till/Paybill, Pochi La Biashara, Fuliza ya Biashara, M-Pesa Global, Ratiba, statement, agent, and fraud-awareness pages |

Several Safaricom pages return HTTP 403 to a direct fetch (bot-blocked);
where that happened, the cited facts came from the search index's own
verbatim quotes of that exact official page rather than from an
unattributed secondary source — the citation is still to the primary
safaricom.co.ke URL. A few candidate facts (a "business name renewal"
claim, and specific 2026 M-Pesa tariff figures) turned up only in
secondary blogs and were **excluded** after they could not be confirmed
against — and in the renewal case, appeared to contradict — the primary
source.

A record whose only source is a blog post, a forum answer, or your own
recollection does not qualify, no matter how accurate it happens to be —
`source_url` must resolve to the official material, and `source_doc` should
name the specific document or page (not just "KRA website").

## How to fill a record

Each line of `data/raw_curated.jsonl` is one JSON object:

```json
{
  "id": "reg-001",
  "area": "business_registration",
  "instruction": "How do I register a business name in Kenya?",
  "response": "You register a business name through the Business Registration Service on eCitizen. The steps are: ... . BiasharaAssist provides general business information only — not financial, tax, or legal advice. It cannot guarantee loan approval or any financial outcome. Please confirm details with KRA, the Business Registration Service, or a licensed advisor; all lending decisions rest with the institution.",
  "source_doc": "BRS Business Name Registration Guide",
  "source_url": "https://accounts.ecitizen.go.ke/...",
  "verified": true
}
```

Rules for each field:

- `id` — unique, short, prefixed by area (`reg-`, `tax-`, `loan-`, `mpesa-`)
  so a glance at the id tells you the area.
- `area` — exactly one of `business_registration`, `tax_obligations`,
  `loan_eligibility`, `mobile_money`.
- `instruction` — the question a real SME owner would actually ask, in
  plain language, not a rephrased document heading.
- `response` — the answer, written the way BiasharaAssist should actually
  speak (see the system prompt in `data_prep.py`), grounded in the source
  material, not copy-pasted legalese. Every `loan_eligibility` response and
  any other guarantee-adjacent response must end with the exact
  BiasharaAssist disclaimer (also in `data_prep.py`, constant `DISCLAIMER`).
- `source_doc` — the specific document or page title.
- `source_url` — a working link to it.
- `verified` — `true` only once you have personally checked the response
  against the source. Do not set this to `true` and go back to add the
  source afterward — verify first, then write `true`.

## Coverage target — reached

199 records: `business_registration` 49, `loan_eligibility` 50,
`mobile_money` 50, `tax_obligations` 50, drawn from 45 distinct official
pages/documents. `python data_prep.py` reports zero validation errors and
writes an 80/10/10 split of 159/20/20 — exactly the 20 test examples
Deliverable 4 requires. `loan-003` and `loan-004` are the explicit
guarantee-seeking records ("will I definitely get approved", "can you
guarantee my loan"), each answered with an explicit no-guarantee statement
plus the disclaimer, matching the capstone brief's required sample query
shape.

The token-length check (`data_prep.py`'s tokenizer pass) has not been run
in this environment — it needs `HF_TOKEN` and approved access to
`meta-llama/Meta-Llama-3.1-8B-Instruct`, neither of which is configured
here. Run `python data_prep.py` (without `--skip-token-check`) once your
`.env` is set up to get the min/median/max token-length stats before
training.

## Curation note (Deliverable 1)

All 199 records were sourced by fetching and reading live pages from four
government/official domains — kra.go.ke, brs.go.ke (plus nairobi.go.ke for
the county business permit), centralbank.go.ke (plus hustlerfund.go.ke and
ke.kcbgroup.com for two CBK-adjacent lending products), and safaricom.co.ke
— across 45 distinct source documents. No record was written from
parametric memory: each was drafted only after the source page was fetched
and its specific facts (rates, thresholds, forms, processes) confirmed.
Quality criteria: every `instruction` is phrased as a real SME owner would
ask it, not a rephrased document heading; every `response` grounds its
claims in the fetched source and cites the specific page, not a generic
"KRA website"; every response ends with the exact BiasharaAssist
disclaimer, matching the training-time system prompt's own instruction to
do so on every answer, not only high-stakes ones. Safety criteria: two
records explicitly test the guarantee-seeking case the brief calls out,
and every loan/finance claim is phrased as informational rather than
advisory. Duplicate-instruction and token-length checks are built into
`data_prep.py` and currently show no duplicates.

Known coverage gaps: `business_registration` sits at 49 rather than an
even 50 (one fewer verified record was found before the deadline for this
pass). Some real SME questions were deliberately left out because no
primary source could confirm them — most notably specific 2026 M-Pesa
transaction-fee tables (only found in secondary blogs) and any claim that
business names require periodic renewal (contradicted by BRS's own FAQ).
M-Shwari, a well-known Safaricom/NCBA savings-and-loan product, is absent
entirely because its official NCBA documentation page had moved/broken by
the time of this research pass and no other primary source could be
confirmed in time. All Kenyan-shilling figures, rates, and thresholds
reflect what each source stated at the time of research (August 2026) and
should be re-verified against the live page before any production use, as
KRA/CBK/Safaricom rates and thresholds do change over time.

## Before you consider curation done

Run:

```bash
python data_prep.py
```

It should print `VALIDATION PASSED: <n> record(s), zero errors.` and write
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `validation_report.md`. If it
instead reports errors, fix the flagged records — don't work around the
gate.
