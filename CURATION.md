# Curation Workflow

This is the human half of the dataset deliverable. `data_prep.py` cannot
verify facts — it can only reject anything that isn't tagged as verified
and sourced. Everything below is what makes a record eligible to pass that
gate.

## Why this exists

The assignment requires every example to come from authoritative materials
and forbids synthetic or web-scraped content without rigorous manual
verification. `raw_curated.jsonl` currently holds 12 **template** records
(`verified: false`, empty `source_doc`/`source_url`) as a shape reference
only — running `python data_prep.py` against them today correctly fails
with a validation error per record. That failure is the proof the gate
works, not a bug to fix by relaxing the check.

## The four authoritative source families

Every record's `source_doc` and `source_url` must point to one of these,
or an equivalent official publication:

| Area | Source family | What to cite |
|---|---|---|
| `business_registration` | Business Registration Service (BRS) / eCitizen | Official BRS guides, eCitizen help pages, the Companies Act 2015 where procedure is described |
| `tax_obligations` | Kenya Revenue Authority (KRA) | KRA's official guides on PIN registration, Turnover Tax, VAT thresholds, iTax help pages |
| `loan_eligibility` | Central Bank of Kenya (CBK) | CBK guidance on lending, microfinance regulation, credit reference bureaus |
| `mobile_money` | Safaricom Daraja / Lipa na M-Pesa | Official Daraja API docs, Safaricom's published Lipa na M-Pesa business guidance |

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

## Coverage target

~200 records, stratified across the four areas (roughly 50 each so the
80/10/10 split lands on the target 160/20/20 with 20 test examples per
Deliverable 4). Include at least a few explicitly guarantee-seeking
questions in `loan_eligibility` (e.g. "will I definitely get the loan?",
"can you guarantee my business loan will be approved?") — these are what
`local_inference.py`'s safety gate is built to catch, and the capstone
brief calls for at least one sample query in this shape.

## Before you consider curation done

Run:

```bash
python data_prep.py
```

It should print `VALIDATION PASSED: <n> record(s), zero errors.` and write
`train.jsonl`, `val.jsonl`, `test.jsonl`, and `validation_report.md`. If it
instead reports errors, fix the flagged records — don't work around the
gate. Read `validation_report.md`'s token-length and duplicate-instruction
sections too; those are warnings, not hard failures, but they inform the
300-word curation note the assignment asks for (what quality/safety
criteria you applied, what coverage gaps remain).
