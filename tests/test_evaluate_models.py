"""Unit tests for evaluate_models.py's pure scoring/parsing logic.

tokenize/token_f1/_parse_judge_response/_build_evaluation_prompt need no
extra dependencies. evaluate_response() lazily imports rouge_score, so
those tests are skipped (not failed) in an environment without it
installed -- see the module's own lazy-import docstring for why."""

from __future__ import annotations

import pytest

import evaluate_models as em


# -------------------------------- tokenize --------------------------------
def test_tokenize_lowercases_and_strips_punctuation():
    assert em.tokenize("Hello, World! Is TOT 1.5%?") == [
        "hello", "world", "is", "tot", "1", "5",
    ]


def test_tokenize_empty_string_is_empty_list():
    assert em.tokenize("") == []


# -------------------------------- token_f1 --------------------------------
def test_token_f1_is_one_for_identical_text():
    assert em.token_f1("Turnover Tax is 1.5%", "Turnover Tax is 1.5%") == 1.0


def test_token_f1_is_zero_for_disjoint_text():
    assert em.token_f1("apples and oranges", "completely different words") == 0.0


def test_token_f1_handles_empty_reference_or_hypothesis():
    assert em.token_f1("", "something") == 0.0
    assert em.token_f1("something", "") == 0.0
    assert em.token_f1("", "") == 1.0


def test_token_f1_partial_overlap_is_between_zero_and_one():
    score = em.token_f1("Turnover Tax is one point five percent", "Turnover Tax rate")
    assert 0.0 < score < 1.0


# ----------------------------- evaluate_response -----------------------------
def test_evaluate_response_returns_rouge_l_and_token_f1():
    rouge_score = pytest.importorskip("rouge_score")  # noqa: F841 -- skip if not installed
    result = em.evaluate_response(
        "Turnover Tax is a flat 1.5% on gross sales.",
        "Turnover Tax is a flat 1.5% on gross sales.",
    )
    assert set(result.keys()) == {"rouge_l", "token_f1"}
    assert result["rouge_l"] == 1.0
    assert result["token_f1"] == 1.0


# --------------------------- _build_evaluation_prompt ---------------------------
def test_build_evaluation_prompt_includes_all_three_fields():
    prompt = em._build_evaluation_prompt("Q?", "reference answer", "actual response")
    assert "Q?" in prompt
    assert "reference answer" in prompt
    assert "actual response" in prompt


# ---------------------------- _parse_judge_response ----------------------------
def test_parse_judge_response_handles_direct_json():
    raw = '{"correctness": 5, "groundedness": 4, "relevance": 5, "helpfulness": 4, "overall": 4.5, "reasoning": "good"}'
    result = em._parse_judge_response(raw)
    assert result["overall"] == 4.5
    assert result["reasoning"] == "good"


def test_parse_judge_response_recovers_json_wrapped_in_prose():
    raw = 'Sure, here is my evaluation:\n{"correctness": 3, "groundedness": 3, "relevance": 3, "helpfulness": 3, "overall": 3.0, "reasoning": "ok"}\nHope that helps!'
    result = em._parse_judge_response(raw)
    assert result["overall"] == 3.0


def test_parse_judge_response_falls_back_to_zero_scores_on_unparseable_output():
    result = em._parse_judge_response("I refuse to answer in JSON.")
    assert result["overall"] == 0
    assert result["reasoning"] == "JSON parse failure"


# ------------------------------ create_judge_client ------------------------------
def test_create_judge_client_rejects_a_placeholder_key(monkeypatch):
    monkeypatch.setenv("JUDGE_API_KEY", "sk-or-your-openrouter-key-here")
    with pytest.raises(RuntimeError):
        em.create_judge_client()


def test_create_judge_client_rejects_a_missing_key(monkeypatch):
    monkeypatch.delenv("JUDGE_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        em.create_judge_client()
