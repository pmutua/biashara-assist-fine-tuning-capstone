"""Unit tests for plot_loss.py — no GPU, no matplotlib needed for these
(the plot() function itself needs matplotlib and is not exercised here;
this covers the parsing/diagnosis logic, which is the graded part)."""

from __future__ import annotations

import json

import pytest

import plot_loss


# ----------------------------- load_log_history -----------------------------
def test_load_log_history_raises_on_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        plot_loss.load_log_history(tmp_path / "missing.json")


def test_load_log_history_raises_on_empty_history(tmp_path):
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": []}), encoding="utf-8")
    with pytest.raises(ValueError):
        plot_loss.load_log_history(path)


def test_load_log_history_returns_the_log_history_list(tmp_path):
    history = [{"step": 10, "loss": 2.5}, {"step": 10, "eval_loss": 1.5}]
    path = tmp_path / "trainer_state.json"
    path.write_text(json.dumps({"log_history": history}), encoding="utf-8")
    assert plot_loss.load_log_history(path) == history


# ------------------------------- split_series -------------------------------
def test_split_series_separates_train_and_eval_points():
    history = [
        {"step": 10, "loss": 2.7},
        {"step": 10, "eval_loss": 1.5},
        {"step": 20, "loss": 1.1},
        {"step": 20, "eval_loss": 0.9},
        {"step": 20},  # neither key present -- should be ignored, not crash
    ]
    train_points, eval_points = plot_loss.split_series(history)
    assert train_points == [(10, 2.7), (20, 1.1)]
    assert eval_points == [(10, 1.5), (20, 0.9)]


def test_split_series_handles_empty_history():
    assert plot_loss.split_series([]) == ([], [])


# ---------------------------- heuristic_diagnosis ----------------------------
def test_heuristic_diagnosis_flags_missing_eval_points():
    result = plot_loss.heuristic_diagnosis([(10, 1.0)], [])
    assert "UNKNOWN" in result


def test_heuristic_diagnosis_flags_underfitting_when_eval_loss_barely_moves():
    train_points = [(10, 2.0), (20, 1.9), (30, 1.85)]
    eval_points = [(10, 2.0), (20, 1.99), (30, 1.98)]  # < 2% drop
    result = plot_loss.heuristic_diagnosis(train_points, eval_points)
    assert "UNDERFITTING" in result


def test_heuristic_diagnosis_flags_overfitting_on_a_large_train_eval_gap():
    train_points = [(10, 2.0), (20, 1.0), (30, 0.3)]
    eval_points = [(10, 2.0), (20, 1.5), (30, 1.2)]  # gap = 1.2 - 0.3 = 0.9 > 0.5
    result = plot_loss.heuristic_diagnosis(train_points, eval_points)
    assert "OVERFITTING" in result


def test_heuristic_diagnosis_reports_healthy_when_both_fall_together():
    train_points = [(10, 2.7), (20, 1.1), (30, 0.85)]
    eval_points = [(10, 1.5), (20, 0.9), (30, 0.82)]  # matches the course's own healthy example
    result = plot_loss.heuristic_diagnosis(train_points, eval_points)
    assert "HEALTHY" in result
