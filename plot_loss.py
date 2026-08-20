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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer-state", type=Path, default=Path("adapter/trainer_state.json")
    )
    parser.add_argument("--output", type=Path, default=Path("loss_curve.png"))
    return parser.parse_args()


def load_log_history(trainer_state_path: Path) -> list[dict]:
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
    """A rough automated read — not a substitute for the written diagnosis."""
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
    args = parse_args()
    log_history = load_log_history(args.trainer_state)
    train_points, eval_points = split_series(log_history)
    print(f"{len(train_points)} train-loss point(s), {len(eval_points)} eval-loss point(s)")
    plot(train_points, eval_points, args.output)
    print(heuristic_diagnosis(train_points, eval_points).format(output=args.output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
