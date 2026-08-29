# BiasharaAssist — Fine-Tuned SME Financial Advisory Assistant

## Overview

BiasharaAssist is a LLaMA 3.1 8B assistant fine-tuned with QLoRA for **Kenyan
SME financial advisory**: routine questions from small business owners about
business registration, tax obligations, loan eligibility, and mobile money
(M-Pesa / Daraja) integration. It is a Week 4 fine-tuning capstone —
non-advisory and informational only, never a substitute for KRA, the
Business Registration Service, a licensed financial advisor, or the
lending institution's own decision.

Stakeholder: the microfinance institution's operations/product director
(see `memo.md`).

**A note on compute platform:** the assignment brief names Nebius
specifically for the training deliverable. This project trains on RunPod
instead because Nebius does not support payment from Kenya — a real
compute-access constraint, not a preference. See `RUNPOD_GUIDE.md`'s "Why
RunPod instead of Nebius" section for the full reasoning.

## Tech stack

| Layer | Tool / platform | Why |
|---|---|---|
| Base model | [`meta-llama/Meta-Llama-3.1-8B-Instruct`](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) — gated, hosted on Hugging Face | The Instruct variant already follows the chat/messages format and instruction-following behaviour this dataset is built around |
| Fine-tuning method | QLoRA — 4-bit NF4 quantization + LoRA adapters over `q_proj`/`k_proj`/`v_proj`/`o_proj` | Fits an 8B model's training footprint on a single 24GB consumer/prosumer GPU instead of a multi-GPU full fine-tune |
| Training stack | `transformers==4.43.3`, `trl==0.8.6` (`SFTTrainer`), `peft==0.11.1`, `bitsandbytes>=0.46.1`, `accelerate==0.33.0` | Exact pinned combination confirmed to run this training job end to end on a RunPod CUDA 12.8 template — see `RUNPOD_GUIDE.md`'s troubleshooting table for what breaks if any one of these drifts |
| Compute platform | [RunPod](https://runpod.io) GPU pod — Ampere-or-newer, 24GB+ VRAM (A40 or RTX 4090 recommended) | The brief names Nebius, but Nebius does not support payment from Kenya — see the compute-platform note above and `RUNPOD_GUIDE.md` |
| Model access / hosting | Hugging Face Hub (`huggingface_hub`) | Downloads the gated base model; hosts the merged model's config/tokenizer format |
| Dataset format | LLaMA chat/`messages`-format JSONL, hand-curated with per-record provenance (`source_doc`/`source_url`/`verified`) | Matches the brief's required training format; provenance fields let `data_prep.py` mechanically reject unsourced content |
| Dataset tooling | `data_prep.py` — Python standard library only (`argparse`, `json`, `random`, `statistics`) | No extra dependency needed for validation/splitting; keeps the dataset gate runnable anywhere, no GPU or network required |
| Lexical evaluation | `rouge-score` (ROUGE-L) + a hand-rolled token-F1 function | Fast, free, local metrics that don't depend on an external API being up |
| LLM-as-judge evaluation | [OpenRouter](https://openrouter.ai) serving `openai/gpt-4o-mini`, called via the `openai>=1.0` Python client (OpenAI-API-compatible) | One quality dimension automated metrics can't capture — correctness/groundedness/relevance/helpfulness scored by a second model |
| Loss visualization | `matplotlib` (`plot_loss.py`) | Renders `trainer_state.json`'s `log_history` into the required train-vs-validation loss curve |
| Data wrangling / reporting | `pandas`, `tabulate` | Builds `comparison_results.csv` and the printed comparison table in `evaluate_models.py` |
| Config / secrets | `python-dotenv` + a gitignored `.env` (see `.env.example`) | Keeps `HF_TOKEN`/`JUDGE_API_KEY` out of source control while still loadable by local scripts |
| Language / runtime | Python 3.10+ | Matches the `from __future__ import annotations` + modern type-hint syntax (`list[dict]`, `tuple[...]`) used throughout every script |
| Version control / hosting | Git + GitHub | Repository hosting for submission; `git clone` is also the fastest way to get this repo onto a RunPod pod (see `RUNPOD_GUIDE.md` Step 3) |
| Testing | `pytest` (`tests/`) | Unit-tests the pure logic in every script — dataset splitting, loss-curve diagnosis, the safety gate, the scoring functions — without needing a GPU, `HF_TOKEN`, or a real merged model; see `WORKFLOW.md`'s "Testing" section |

`requirements.txt` is the single source of truth for exact package pins —
this table explains *why* each one is here, not just *that* it's here.

## Project status

This repo is built in milestones; each stops for human input before the
next starts (see the brief this repo was built from).

| Milestone | What it produces | Status |
|---|---|---|
| M0 — Scaffold | Repo layout, scripts below, 12 unverified template dataset records | Done |
| M1 — Dataset | `train/val/test.jsonl`, `validation_report.md`, zero validation errors | Done — 199 sourced records (49/50/50/50 across areas), 159/20/20 split; see `CURATION.md` |
| M2 — Training | Completed training run on RunPod: `adapter/`, `trainer_state.json`, `loss_curve.png` | Script written (`fine_tune.py`), not yet run — Llama 3.1 access approved and RunPod account funded; only remaining step is pasting a real `HF_TOKEN` into `.env` and launching per `RUNPOD_GUIDE.md` |
| M3 — Merge & inference | `merged-model/`, 5+ verified sample responses | Scripts written (`merge_model.py`, `local_inference.py`), pending M2's real `adapter/` output |
| M4 — Evaluation | `comparison_results.csv`, comparison table, top/bottom-3 analysis | Script written (`evaluate_models.py`), pending M3's real `merged-model/` |
| M5 — Memo | `memo.md` filled with real numbers | Pending M4 |

## Project structure

- `data/raw_curated.jsonl` — human-curated dataset with provenance (source of truth)
- `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` — generated by `data_prep.py`
- `data_prep.py` — validates `raw_curated.jsonl` and writes the 80/10/10 split
- `tests/` — `pytest` unit tests for every script's pure logic (see `WORKFLOW.md`)
- `fine_tune.py` — QLoRA training script (runs on a RunPod GPU pod)
- `plot_loss.py` — reads `trainer_state.json`, plots `loss_curve.png`
- `merge_model.py` — merges the trained adapter into the base model (RunPod)
- `local_inference.py` — inference pipeline with the safety/disclaimer gate
- `evaluate_models.py` — base vs. fine-tuned evaluation on `test.jsonl`
- `comparison_results.csv` — evaluation metrics (produced by `evaluate_models.py`)
- `memo.md` — one-page stakeholder recommendation
- `CURATION.md` — how records are sourced and verified
- `validation_report.md` — produced by `data_prep.py`
- `WORKFLOW.md` — day-by-day runbook: exact commands, which machine each
  step runs on, and the "done when" checklist for each milestone
- `RUNPOD_GUIDE.md` — detailed, click-by-click RunPod walkthrough for
  Day 2/3, written for a developer who has never used RunPod before

## How to reproduce

For the day-by-day version of these same steps — what to check before
moving on, where each script runs, and rough time/cost expectations per
stage — see `WORKFLOW.md`.

1. **Environment setup**

   ```bash
   python -m venv .venv
   # Windows: .venv\Scripts\activate   |   macOS/Linux: source .venv/bin/activate
   pip install -r requirements.txt
   cp .env.example .env   # then fill in HF_TOKEN and JUDGE_API_KEY
   ```

   `requirements.txt` covers both the local side (this machine) and the
   RunPod side (training/merge); it deliberately does not pin `torch` —
   see the comment at the top of that file for why.

2. **Run the tests** — confirms the codebase itself is healthy before you
   spend any time or RunPod money on it:

   ```bash
   pytest
   ```

   Should print `45 passed, 1 skipped` (the skip is `evaluate_response()`'s
   ROUGE-L test if `rouge-score` isn't installed yet — harmless). None of
   this needs a GPU, `HF_TOKEN`, or network access.

3. **Data**: curate `data/raw_curated.jsonl` per `CURATION.md`, then:

   ```bash
   python data_prep.py
   ```

   This must report zero validation errors before you move on — it writes
   `data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl`, and
   `validation_report.md`.

4. **Fine-tuning**: copy the repo to a RunPod GPU pod (24GB minimum — RTX
   4090 / A5000 / A100), then, inside `tmux` so the run survives a dropped
   connection:

   ```bash
   tmux new -s finetune
   pip install -r requirements.txt
   python fine_tune.py
   # detach: Ctrl-b then d ; reattach later: tmux attach -t finetune
   ```

   **Stop the pod from the RunPod console the moment training finishes** —
   you are billed for pod uptime, not just training time.

5. **Loss curve**: on the pod or after copying `adapter/trainer_state.json`
   back locally:

   ```bash
   python plot_loss.py
   ```

6. **Merging** (on the pod, before you terminate it — needs the full-size
   base model weights in memory):

   ```bash
   python merge_model.py
   ```

   Then download `merged-model/` back to this machine (or continue
   locally if your machine can hold an 8B model in memory).

7. **Local inference**:

   ```bash
   python local_inference.py
   ```

8. **Evaluation**:

   ```bash
   python evaluate_models.py
   ```

   Writes `comparison_results.csv` and prints the comparison table plus the
   top-3/bottom-3 per-question breakdown.

## Safety and ethics design

The brief requires the domain's disclaimer to be built into both the
system prompt and the validation gate, not just mentioned in a memo.
Concretely, that's four separate layers, each in a different file:

1. **Dataset gate** (`data_prep.py`) — a record cannot reach the training
   split unless `verified: true` and both `source_doc`/`source_url` are
   filled in; unsourced or unverified records are rejected before
   training ever sees them (see `CURATION.md`).
2. **Training-time system prompt** (`data_prep.py`'s `SYSTEM_PROMPT` and
   `DISCLAIMER` constants) — every training example's assistant turn ends
   with the exact disclaimer, and the system prompt explicitly instructs
   the model never to promise a loan approval, guarantee a financial
   outcome, or issue a binding tax/legal ruling. Two dataset records
   (`loan-003`, `loan-004`) exist specifically to teach the
   guarantee-seeking refusal case.
3. **Inference-time verification gate** (`local_inference.py`'s
   `passes_safety_gate`) — every response is checked for the disclaimer's
   presence and scanned against a list of outcome-promising phrases
   (`"your loan is approved"`, `"100% approval"`, etc.). That phrase list
   is deliberately *not* the same words the disclaimer itself uses
   (`"guarantee"`, `"financial advice"`, `"legal advice"`) — the
   disclaimer negates those words in every single correct answer, so a
   naive keyword filter reusing them would flag every compliant response
   as unsafe. The gate checks for affirmative phrasing that would
   *contradict* the disclaimer, not for the disclaimer's own vocabulary.
4. **Evaluation-time compliance floor** (`evaluate_models.py`'s
   `GROUNDEDNESS_FLOOR = 3.0`) — any fine-tuned response the LLM judge
   scores below 3/5 on groundedness is printed as a `[ALERT]` during the
   run and counted toward a compliance warning in the final summary,
   independent of how well it scores on the other three dimensions.

## Evaluation methodology

`evaluate_models.py` prompts the base model and `merged-model/` identically
on all 20 held-out `data/test.jsonl` questions, then scores every response
two ways:

- **Lexical overlap** — ROUGE-L (longest common subsequence F1) and token
  F1 against the reference answer, computed locally with `rouge-score`, no
  API cost.
- **LLM judge** — `openai/gpt-4o-mini` via OpenRouter (`JUDGE_PROVIDER`/
  `JUDGE_MODEL`/`JUDGE_API_KEY` in `.env`) scores each response 1–5 on four
  dimensions: **correctness** (factually right), **groundedness**
  (supported by the reference, not hallucinated), **relevance** (answers
  the actual question), and **helpfulness** (usable by a small business
  owner), at `temperature=0` for repeatable scores.

The per-question breakdown ranks all 20 questions by `judge_delta`
(fine-tuned score minus base score) and prints the 3 biggest and 3
smallest improvements — the pattern worth writing up in the memo is
usually that the biggest gains land on narrow, fact-lookup questions
(rates, deadlines, thresholds) the base model has no way to know, while
the smallest gains land on questions the base model could already answer
reasonably from general knowledge (e.g. generic "how do I register a
business" phrasing).

## Disclaimer

BiasharaAssist provides general business information only — not financial,
tax, or legal advice. It cannot guarantee loan approval or any financial
outcome. Please confirm details with KRA, the Business Registration
Service, or a licensed advisor; all lending decisions rest with the
institution.
