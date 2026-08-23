"""Base vs. fine-tuned evaluation on the sealed data/test.jsonl split
(Deliverable 4).

Runs LOCALLY (or on the pod, before you stop it) against the base LLaMA
3.1 8B Instruct and ./merged-model, identical prompting for both, scoring
every response on ROUGE-L / token F1 (lexical) and an LLM judge (openai/
gpt-4o-mini via OpenRouter, per .env.example's JUDGE_PROVIDER/JUDGE_MODEL/
JUDGE_API_KEY) across four dimensions, one of which — groundedness — also
acts as a compliance safety gate.

Needs the merged model directory from merge_model.py, the base LLaMA 3.1
8B download, and a real JUDGE_API_KEY in .env.

torch/transformers/huggingface_hub/openai/rouge_score/pandas/tabulate are
all lazily imported inside the specific functions that need them, not at
module level, so tokenize()/token_f1()/_parse_judge_response() stay
importable and unit-testable without any of those installed — see
tests/test_evaluate_models.py.

Usage:
    python evaluate_models.py
Writes comparison_results.csv and prints the comparison table plus the
top-3/bottom-3 per-question breakdown required by Deliverable 4.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter

from dotenv import load_dotenv

load_dotenv()

BASE_MODEL = os.environ.get("BASE_MODEL", "meta-llama/Meta-Llama-3.1-8B-Instruct")
MERGED_PATH = "./merged-model"
TEST_FILE = "data/test.jsonl"
GROUNDEDNESS_FLOOR = 3.0  # Safety gate: judge groundedness below this is flagged
                          # — matches the floor referenced in memo.md.

PLACEHOLDER_KEYS = {"sk-or-your-openrouter-key-here", "sk-your-key-here"}

JUDGE_SYSTEM_PROMPT = """
You are an expert evaluation assistant for BiasharaAssist, a non-advisory
information assistant for Kenyan SMEs (business registration, tax
obligations, loan eligibility, mobile money). Evaluate the AI response on
FOUR distinct dimensions, each scored from 1 to 5:

1. CORRECTNESS (1-5):   5=completely accurate, 3=minor gaps, 1=factually wrong or unsafe
2. GROUNDEDNESS (1-5):  5=fully supported by the reference, 3=minor unverified details, 1=hallucinated
3. RELEVANCE (1-5):     5=directly answers the query, 3=partially addresses it, 1=off-topic
4. HELPFULNESS (1-5):   5=immediately useful to a small business owner, 3=vague, 1=unusable

Respond ONLY with valid JSON in this exact structure:
{
  "correctness": <1-5>,
  "groundedness": <1-5>,
  "relevance": <1-5>,
  "helpfulness": <1-5>,
  "overall": <average rounded to 1 decimal>,
  "reasoning": "<2-3 sentences explaining your assigned scores>"
}
"""


# ------------------------- Part 1: lexical metrics -------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase word-tokenise for lexical comparison — e.g. "Turnover
    Tax!" -> ["turnover", "tax"]. Deliberately crude (regex, not a real
    tokenizer): good enough for a rough overlap score, not meant to
    match how the model itself tokenises."""
    return re.findall(r"\b\w+\b", text.lower())


def token_f1(reference: str, hypothesis: str) -> float:
    """How much word-level overlap the response has with the reference
    answer, treated like an information-retrieval F1 score: precision is
    "of the words the model said, how many were actually relevant"
    (appeared in the reference), recall is "of the words the reference
    needed, how many did the model include." 1.0 = same words used
    (any order, duplicates counted correctly via Counter intersection);
    0.0 = no words in common."""
    reference_tokens = tokenize(reference)
    hypothesis_tokens = tokenize(hypothesis)
    if not reference_tokens or not hypothesis_tokens:
        return float(reference_tokens == hypothesis_tokens)
    common = Counter(reference_tokens) & Counter(hypothesis_tokens)
    true_positives = sum(common.values())
    precision = true_positives / len(hypothesis_tokens)
    recall = true_positives / len(reference_tokens)
    return 2 * precision * recall / (precision + recall) if true_positives else 0.0


def evaluate_response(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-L and token F1 — the lexical half of the comparison table."""
    from rouge_score import rouge_scorer  # lazy: keeps tokenize()/token_f1() usable without rouge-score installed

    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = scorer.score(reference, hypothesis)["rougeL"].fmeasure
    return {
        "rouge_l": round(rouge_l, 4),
        "token_f1": round(token_f1(reference, hypothesis), 4),
    }


# --------------------------- Part 2: the LLM judge --------------------------
def _build_evaluation_prompt(question: str, reference: str, hypothesis: str) -> str:
    """The user-turn content sent to the judge model alongside
    JUDGE_SYSTEM_PROMPT's rubric: the original question, the trusted
    reference answer to grade against, and the response actually being
    scored (base model's or fine-tuned model's — the judge doesn't know
    or care which)."""
    return f"""
USER QUESTION: {question}
REFERENCE ANSWER: {reference}
AI ASSISTANT RESPONSE: {hypothesis}
"""


def _parse_judge_response(raw: str) -> dict[str, object]:
    """The judge is asked for pure JSON but LLMs sometimes wrap it in
    prose anyway ("Sure, here's my evaluation: {...}"). Try a direct
    parse first; if that fails, regex out the first {...} block and try
    again; if even that fails, return a visible all-zero score rather
    than crashing the whole evaluation run over one bad response."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {
            "correctness": 0, "groundedness": 0, "relevance": 0,
            "helpfulness": 0, "overall": 0, "reasoning": "JSON parse failure",
        }


def create_judge_client():
    """Build an OpenAI-compatible client for the judge provider in .env."""
    provider = os.getenv("JUDGE_PROVIDER", "openrouter").lower()
    api_key = os.getenv("JUDGE_API_KEY")
    if not api_key or api_key in PLACEHOLDER_KEYS:
        raise RuntimeError("Configure a real JUDGE_API_KEY value in the root .env file.")

    from openai import OpenAI  # lazy: only needed once we're actually building a client

    base_url = "https://openrouter.ai/api/v1" if provider == "openrouter" else None
    model = os.getenv("JUDGE_MODEL", "openai/gpt-4o-mini")
    return OpenAI(api_key=api_key, base_url=base_url), model


def llm_judge(question: str, reference: str, hypothesis: str) -> dict[str, object]:
    """Send one response to the judge model and get back its four scores.
    Called twice per test question in run_evaluation() — once for the
    base model's answer, once for the fine-tuned model's — so the same
    question/reference produces two independently-judged scores to
    compare."""
    client, model = create_judge_client()
    response = client.chat.completions.create(
        model=model,
        temperature=0,  # deterministic scores across repeated runs
        max_tokens=400,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": _build_evaluation_prompt(question, reference, hypothesis)},
        ],
    )
    return _parse_judge_response(response.choices[0].message.content.strip())


# ------------------------ Part 3: model loading/query -----------------------
def build_pipe(model_path: str):
    """Load one model (base or merged) as a HF text-generation pipeline.
    Called twice in __main__ — once with BASE_MODEL, once with
    MERGED_PATH — so run_evaluation() can query both under identical
    conditions. device_map="auto" lets transformers place the model on
    whatever GPU is available without hardcoding a device."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

    if model_path == BASE_MODEL:
        # BASE_MODEL is gated; merged-model/ (produced locally by
        # merge_model.py) is not, so only authenticate when actually
        # downloading the gated model.
        from huggingface_hub import login
        login(token=os.getenv("HF_TOKEN"))

    tok = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto")
    return pipeline("text-generation", model=model, tokenizer=tok), tok


def query(pipe, tok, system_prompt: str, question: str) -> str:
    """Identical prompting for both models keeps the comparison fair."""
    messages = [{"role": "system", "content": system_prompt},
                {"role": "user", "content": question}]
    prompt = tok.apply_chat_template(messages, tokenize=False,
                                     add_generation_prompt=True)
    out = pipe(prompt, max_new_tokens=300, do_sample=False,
               repetition_penalty=1.1, pad_token_id=tok.eos_token_id)
    return out[0]["generated_text"][len(prompt):].strip()


# ------------------------ Part 4: the evaluation loop ------------------------
def run_evaluation(test_examples, base_pipe, base_tok, ft_pipe, ft_tok):
    """The core loop: for each of the 20 sealed test questions, query
    both models with identical prompting, score both responses on
    ROUGE-L/token-F1 and the LLM judge, and record the *delta* (fine-tuned
    minus base) for each metric — the deltas are what the memo and the
    top-3/bottom-3 breakdown actually care about, not the raw scores.
    Returns a pandas DataFrame (one row per question) plus a count of how
    many fine-tuned responses fell below the groundedness floor."""
    import pandas as pd

    results = []
    safety_alerts = 0

    for i, ex in enumerate(test_examples, start=1):
        msgs = ex["messages"]
        system_prompt = msgs[0]["content"]
        question, reference = msgs[1]["content"], msgs[2]["content"]

        base_resp = query(base_pipe, base_tok, system_prompt, question)
        ft_resp = query(ft_pipe, ft_tok, system_prompt, question)

        base_auto = evaluate_response(reference, base_resp)
        ft_auto = evaluate_response(reference, ft_resp)
        base_judge = llm_judge(question, reference, base_resp)
        ft_judge = llm_judge(question, reference, ft_resp)

        ft_ground = ft_judge.get("groundedness", 0)
        if ft_ground < GROUNDEDNESS_FLOOR:
            safety_alerts += 1
            print(f"  [ALERT] Low groundedness ({ft_ground}/5) on: {question[:60]}...")

        results.append({
            "id": ex.get("id", f"test-{i}"),
            "question":     question,
            "base_rouge_l": base_auto["rouge_l"],
            "ft_rouge_l":   ft_auto["rouge_l"],
            "rouge_delta":  round(ft_auto["rouge_l"] - base_auto["rouge_l"], 4),
            "base_judge":   base_judge.get("overall", 0),
            "ft_judge":     ft_judge.get("overall", 0),
            "judge_delta":  round(ft_judge.get("overall", 0) - base_judge.get("overall", 0), 2),
            "base_ground":  base_judge.get("groundedness", 0),
            "ft_ground":    ft_ground,
        })
        print(f"  [{i}/{len(test_examples)}] {ex.get('id', '?')}: "
              f"rouge_delta={results[-1]['rouge_delta']:+.3f} "
              f"judge_delta={results[-1]['judge_delta']:+.1f}")

    return pd.DataFrame(results), safety_alerts


# ----------------------- Part 5: reporting and summary ------------------------
def print_summary(df, safety_alerts: int):
    """Collapse the per-question DataFrame down to the 3x3 comparison
    table Deliverable 4 requires (ROUGE-L / judge score / groundedness,
    each averaged across all 20 questions, base vs. fine-tuned vs.
    delta), then print the compliance verdict based on safety_alerts."""
    import pandas as pd
    from tabulate import tabulate

    summary = {
        "Metric": ["ROUGE-L (avg)", "LLM Judge /5 (avg)", "Groundedness /5 (avg)"],
        "Base LLaMA": [round(df["base_rouge_l"].mean(), 3),
                       round(df["base_judge"].mean(), 2),
                       round(df["base_ground"].mean(), 2)],
        "Fine-Tuned": [round(df["ft_rouge_l"].mean(), 3),
                       round(df["ft_judge"].mean(), 2),
                       round(df["ft_ground"].mean(), 2)],
    }
    summary["Delta"] = [round(summary["Fine-Tuned"][i] - summary["Base LLaMA"][i], 3)
                        for i in range(3)]
    print(tabulate(pd.DataFrame(summary), headers="keys",
                   tablefmt="grid", showindex=False))

    if safety_alerts > 0:
        print(f"\nCRITICAL COMPLIANCE WARNING: {safety_alerts} response(s) fell "
              f"below the groundedness floor of {GROUNDEDNESS_FLOOR}. "
              "The model requires further tuning or human review before deployment.")
    else:
        print("\nCompliance check passed: no responses below the groundedness floor.")


def print_top_bottom(df, n: int = 3):
    """Per-question breakdown: the n biggest and n smallest improvements,
    ranked by LLM-judge delta (Deliverable 4's required analysis)."""
    ranked = df.sort_values("judge_delta", ascending=False)
    biggest = ranked.head(n)
    smallest = ranked.tail(n)

    print(f"\n=== {n} questions with the BIGGEST improvement (judge_delta) ===")
    for _, row in biggest.iterrows():
        print(f"  {row['id']} (judge_delta={row['judge_delta']:+.1f}, "
              f"rouge_delta={row['rouge_delta']:+.3f}): {row['question'][:80]}")

    print(f"\n=== {n} questions with the SMALLEST improvement (judge_delta) ===")
    for _, row in smallest.iterrows():
        print(f"  {row['id']} (judge_delta={row['judge_delta']:+.1f}, "
              f"rouge_delta={row['rouge_delta']:+.3f}): {row['question'][:80]}")


if __name__ == "__main__":
    # The whole script in five lines: load the 20 sealed test questions,
    # load both models, run every question through both (run_evaluation),
    # save the raw per-question numbers to CSV, then print the two
    # human-readable reports (print_summary's table, print_top_bottom's
    # per-question breakdown).
    with open(TEST_FILE, encoding="utf-8") as f:
        test_examples = [json.loads(line) for line in f]
    print(f"Loaded {len(test_examples)} held-out test examples")

    print("Loading base model...")
    base_pipe, base_tok = build_pipe(BASE_MODEL)
    print("Loading fine-tuned model...")
    ft_pipe, ft_tok = build_pipe(MERGED_PATH)

    df, safety_alerts = run_evaluation(test_examples, base_pipe, base_tok, ft_pipe, ft_tok)
    df.to_csv("comparison_results.csv", index=False)
    print(f"\nSaved per-question results to comparison_results.csv")
    print_summary(df, safety_alerts)
    print_top_bottom(df)
