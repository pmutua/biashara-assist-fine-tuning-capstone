# Memo: BiasharaAssist Fine-Tuned SME Advisory Assistant

**To:** Operations/Product Director, [microfinance institution — name pending]
**From:** Philip Mutua
**Date:** 2026-08-29
**Re:** Fine-tuned assistant for SME financial advisory — results and recommendation

## What we built

We trained a version of our AI assistant specifically on Kenyan SME
questions about business registration, tax, loans, and M-Pesa, using 159
real, verified examples sourced from KRA, the Business Registration
Service, the Central Bank of Kenya, and Safaricom documentation. Tested
against 20 held-out questions it never saw during training, the assistant
now matches our verified answers' phrasing and structure far more closely
than the general-purpose model it started from — but on tax and loan
questions specifically, it also became less reliable at sticking to facts
it was actually trained on. Both results are explained below, with a
concrete fix recommended before this assistant handles those two topic
areas unsupervised.

## Quality improvement

| Metric | Before (base model) | After (fine-tuned) | Change |
|---|---|---|---|
| ROUGE-L (avg) | 0.330 | 0.532 | **+61%** |
| LLM judge score (avg / 5) | 4.27 | 3.70 | **−13%** |
| Groundedness (avg / 5) | 3.80 | 3.10 | **−18%** |

*Groundedness measures how often the assistant's answer sticks to facts it
was actually trained on, versus making something up that sounds
plausible — the single number that matters most for a lending-adjacent
product.*

In plain language: the assistant got noticeably better at sounding like
our verified answers — matching their wording and structure improved by
about 61%. But on questions involving specific tax and loan numbers
(thresholds, repayment terms, fees), it got measurably worse at staying
factually accurate: our fact-checking score dropped about 18%, and 8 of
the 20 test questions (40%) fell below our minimum safety bar for factual
grounding. This clusters by topic — 60% of tax and loan questions failed
that bar, versus only 20–25% of M-Pesa and registration questions — which
points to a specific, fixable cause: those two topic areas need more
training examples per specific numeric fact, not just more examples
overall. This is not a reason to abandon the approach; it is a signal
about where to invest before wider deployment.

## Compute cost

Training ran on a RunPod A40 GPU pod at $0.44/hour, covering the full
pipeline — environment setup, training (202 seconds of actual training
time across 3 epochs), merging, inference, and evaluation, through to
pod termination. Per RunPod's own exported daily billing summary
(`runpod-billing-summary-daily-2026-08-01_2026-08-29.csv`), the 2026-08-29
session's final settled cost was **$0.603** ($0.596 GPU compute + $0.007
storage) — well under $1 for the entire pipeline, consistent with
`RUNPOD_GUIDE.md`'s cost estimate for a small, LoRA-only fine-tune on
this dataset size.

## Recommended next actions

1. **Expand the `tax_obligations` and `loan_eligibility` dataset before
   the next training run** — 60% of tax and loan test questions (6 of 10)
   fell below the groundedness floor after fine-tuning, versus 20–25% for
   M-Pesa/registration; these two areas need more training examples per
   specific numeric fact (tax thresholds, loan repayment terms), not just
   a larger total dataset.
2. **Require human review for tax/loan-specific numeric answers before
   deployment** — 7 of the 8 groundedness failures were introduced by
   fine-tuning itself (the base model scored acceptably on the same
   question), so until dataset coverage improves, route any
   `tax_obligations`/`loan_eligibility` response scoring below the 3.0
   groundedness floor to a human reviewer before it reaches a customer.

## Risk and mitigation

**Risk:** Hallucination — the model may state a fact (a fee, a deadline, an
eligibility rule) that is wrong or out of date, with the same confident
tone as a correct answer. This is not hypothetical: it already happened on
8 of 20 test questions in this evaluation, concentrated in tax and loan
numeric-fact questions.

**Mitigation:** Human-in-the-loop review of every new example before it is
added to the training set (see `CURATION.md`), plus the groundedness score
above as an ongoing check: any new batch of responses scoring below
3.0/5 on groundedness should be reviewed before the model is retrained on
it — and, per the action above, `tax_obligations`/`loan_eligibility`
responses specifically should route to human review until the next
training round measurably improves on this evaluation's baseline.
