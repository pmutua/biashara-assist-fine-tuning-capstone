"""Plot train/eval loss from a completed fine_tune.py run.

Runs LOCALLY or on the RunPod pod (no GPU needed — this only reads a JSON
log). Reads trainer_state.json's log_history, plots training and
validation loss against step to loss_curve.png, and prints a heuristic
health read. The heuristic is a starting point, not the graded diagnosis —
Deliverable 2 requires your own written healthy/overfit/underfit call using
the patterns from this course's Week 4 Tuesday loss-curve lesson.

Usage:
    python plot_loss.py --trainer-state adapter/trainer_state.json \
        --output loss_curve.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Command-line flags. Defaults match fine_tune.py's OUTPUT_DIR
    ("adapter"), so `python plot_loss.py` with no arguments just works
    once you've copied adapter/trainer_state.json back from RunPod."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer-state", type=Path, default=Path("adapter/trainer_state.json")
    )
    parser.add_argument("--output", type=Path, default=Path("loss_curve.png"))
    return parser.parse_args()


def load_log_history(trainer_state_path: Path) -> list[dict]:
    """Read trainer_state.json (written by HF Trainer/SFTTrainer at the
    end of fine_tune.py) and return its log_history list — one dict per
    logging/eval event during training, e.g. {"step": 10, "loss": 2.74}
    or {"step": 10, "eval_loss": 1.51}. Raises loudly instead of
    returning an empty result, since a missing or empty file almost
    always means "you forgot to copy this back from the pod," not "there
    is nothing to plot."""
    if not trainer_state_path.exists():
        raise FileNotFoundError(
            f"{trainer_state_path} not found. Copy it back from the RunPod "
            "pod's adapter output directory after training completes."
        )
    with trainer_state_path.open("r", encoding="utf-8") as fh:
        state = json.load(fh)
    log_history = state.get("log_history", [])
    if not log_history:
        raise ValueError(f"{trainer_state_path} has an empty log_history")
    return log_history


def split_series(log_history: list[dict]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """log_history is one flat list mixing two kinds of entries — training
    steps (which have "loss") and evaluation steps (which have
    "eval_loss") — logged at different cadences (logging_steps vs.
    eval_steps in fine_tune.py's TrainingArguments). Split it into two
    separate (step, value) point lists, one per line the plot draws."""
    train_points = [
        (entry["step"], entry["loss"])
        for entry in log_history
        if "loss" in entry and "step" in entry
    ]
    eval_points = [
        (entry["step"], entry["eval_loss"])
        for entry in log_history
        if "eval_loss" in entry and "step" in entry
    ]
    return train_points, eval_points


def heuristic_diagnosis(
    train_points: list[tuple[float, float]], eval_points: list[tuple[float, float]]
) -> str:
    """A rough automated read — not a substitute for the written diagnosis.
    Three checks, in order: (1) if eval_loss hasn't dropped at least 2%
    from its first to last value, the model probably hasn't learned much
    yet (UNDERFITTING); (2) else, if the final eval_loss sits more than
    0.5 above the final train loss, the model fits the training examples
    much better than unseen ones (OVERFITTING); (3) otherwise, both losses
    are falling together, which is what a healthy run looks like."""
    if not eval_points:
        return "UNKNOWN — no eval_loss entries found; validation loss was not monitored."

    first_eval = eval_points[0][1]
    last_eval = eval_points[-1][1]
    last_train = train_points[-1][1] if train_points else None

    if last_eval > first_eval * 0.98:
        return (
            "Possible UNDERFITTING — eval loss barely moved from "
            f"{first_eval:.3f} to {last_eval:.3f}. Consider more epochs, a "
            "higher LoRA rank, or a higher learning rate."
        )

    if last_train is not None and last_eval - last_train > 0.5:
        return (
            f"Possible OVERFITTING — final train loss {last_train:.3f} vs "
            f"eval loss {last_eval:.3f} (gap {last_eval - last_train:.3f}). "
            "Consider fewer epochs, more dropout, or more training data."
        )

    return (
        f"Looks HEALTHY on this heuristic — eval loss fell from "
        f"{first_eval:.3f} to {last_eval:.3f} and stays close to train "
        f"loss. Confirm by eye on {{output}} before writing it up."
    )


def plot(
    train_points: list[tuple[float, float]],
    eval_points: list[tuple[float, float]],
    output_path: Path,
) -> None:
    """Draw both loss curves on one chart and save it to output_path.
    matplotlib is imported lazily here (not at module top) so the rest of
    this file — load_log_history/split_series/heuristic_diagnosis, the
    parts covered by tests/test_plot_loss.py — stays importable even
    without matplotlib installed. matplotlib.use("Agg") selects a
    file-output backend instead of trying to pop up a GUI window, which
    would fail on a headless RunPod pod or CI anyway."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    if train_points:
        steps, losses = zip(*train_points)
        ax.plot(steps, losses, label="train loss", marker="o", markersize=3)
    if eval_points:
        steps, losses = zip(*eval_points)
        ax.plot(steps, losses, label="eval (validation) loss", marker="s", markersize=4)
    ax.set_xlabel("step")
    ax.set_ylabel("loss")
    ax.set_title("BiasharaAssist fine-tuning: train vs. validation loss")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"Saved {output_path}")


def main() -> int:
    """Entry point: load the trainer state -> split into train/eval point
    series -> render loss_curve.png -> print the heuristic read. Always
    returns 0 (nothing here is a pass/fail gate like data_prep.py's
    validation — a "possible overfitting" read is information for you to
    act on, not a script failure)."""
    args = parse_args()
    log_history = load_log_history(args.trainer_state)
    train_points, eval_points = split_series(log_history)
    print(f"{len(train_points)} train-loss point(s), {len(eval_points)} eval-loss point(s)")
    plot(train_points, eval_points, args.output)
    print(heuristic_diagnosis(train_points, eval_points).format(output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
