# Memo: BiasharaAssist Fine-Tuned SME Advisory Assistant

**To:** {{STAKEHOLDER_NAME}}, Operations/Product Director
**From:** {{YOUR_NAME}}
**Date:** {{DATE}}
**Re:** Fine-tuned assistant for SME financial advisory — results and recommendation

<!--
Fill every {{PLACEHOLDER}} with a real number pulled from
comparison_results.csv and the training run. No unexplained jargon — every
claim in this memo must be backed by a number. Keep this to one page.
-->

## What we built

{{ONE_PARAGRAPH_JARGON_FREE_SUMMARY}}

<!-- e.g. "We trained a version of our AI assistant specifically on Kenyan
SME questions about business registration, tax, loans, and M-Pesa, using
{{N_TRAIN}} real, verified examples sourced from KRA, the Business
Registration Service, CBK, and Safaricom documentation. In testing, it
answers these questions more accurately and consistently than the
general-purpose model we started from." -->

## Quality improvement

| Metric | Before (base model) | After (fine-tuned) | Improvement |
|---|---|---|---|
| ROUGE-L (avg) | {{BASE_ROUGE_L}} | {{FT_ROUGE_L}} | {{ROUGE_L_PCT_IMPROVEMENT}}% |
| LLM judge score (avg / 5) | {{BASE_JUDGE}} | {{FT_JUDGE}} | {{JUDGE_PCT_IMPROVEMENT}}% |
| Groundedness (avg / 5) | {{BASE_GROUNDED}} | {{FT_GROUNDED}} | {{GROUNDED_PCT_IMPROVEMENT}}% |

*Groundedness measures how often the assistant's answer sticks to facts it
was actually trained on, versus making something up that sounds
plausible — the single number that matters most for a lending-adjacent
product.*

In plain language: {{PLAIN_LANGUAGE_INTERPRETATION_OF_THE_TABLE}}

## Compute cost

Training ran on a RunPod {{GPU_TYPE}} pod for {{TRAINING_HOURS}} hours at
${{RUNPOD_HOURLY_RATE}}/hour, for a total compute cost of approximately
**${{TOTAL_COMPUTE_COST}}**.

## Recommended next actions

1. **{{ACTION_1_TITLE}}** — {{ACTION_1_RATIONALE_WITH_A_NUMBER}}
2. **{{ACTION_2_TITLE}}** — {{ACTION_2_RATIONALE_WITH_A_NUMBER}}

## Risk and mitigation

**Risk:** Hallucination — the model may state a fact (a fee, a deadline, an
eligibility rule) that is wrong or out of date, with the same confident
tone as a correct answer.

**Mitigation:** Human-in-the-loop review of every new example before it is
added to the training set (see `CURATION.md`), plus the groundedness score
above as an ongoing check: any new batch of responses scoring below
{{GROUNDEDNESS_FLOOR}}/5 on groundedness should be reviewed before the
model is retrained on it.
