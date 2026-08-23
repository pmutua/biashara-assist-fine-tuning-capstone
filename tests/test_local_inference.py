"""Unit tests for local_inference.py's verification gate — no GPU, no
torch/transformers, and no real merged-model/ directory needed, because
the model itself is loaded lazily (see _get_pipeline() in the module)."""

from __future__ import annotations

from data_prep import DISCLAIMER
import local_inference


def _compliant_response(body: str) -> str:
    return f"{body} {DISCLAIMER}"


def test_passes_safety_gate_accepts_a_compliant_response():
    response = _compliant_response(
        "Register through eCitizen using your KRA PIN and a certificate of "
        "incorporation."
    )
    passed, problems = local_inference.passes_safety_gate(response)
    assert passed is True
    assert problems == []


def test_passes_safety_gate_rejects_a_response_missing_the_disclaimer():
    response = "Register through eCitizen using your KRA PIN."
    passed, problems = local_inference.passes_safety_gate(response)
    assert passed is False
    assert any("disclaimer" in p for p in problems)


def test_passes_safety_gate_rejects_an_outcome_promising_phrase():
    response = _compliant_response("Don't worry, your loan is approved already.")
    passed, problems = local_inference.passes_safety_gate(response)
    assert passed is False
    assert any("prohibited" in p for p in problems)


def test_passes_safety_gate_phrase_check_is_case_insensitive():
    response = _compliant_response("YOUR LOAN IS APPROVED, congratulations.")
    passed, problems = local_inference.passes_safety_gate(response)
    assert passed is False


def test_prohibited_phrases_do_not_collide_with_the_disclaimer_itself():
    """Regression guard for the bug this list was deliberately designed to
    avoid: the mandated disclaimer itself contains "guarantee" and
    "financial/legal advice" in negated form. A compliant response that
    is ONLY the disclaimer (no separate promise) must still pass."""
    passed, problems = local_inference.passes_safety_gate(DISCLAIMER)
    assert passed is True
    assert problems == []


def test_passes_safety_gate_reports_both_problems_at_once():
    response = "Your loan is approved! No disclaimer here."
    passed, problems = local_inference.passes_safety_gate(response)
    assert passed is False
    assert len(problems) == 2
