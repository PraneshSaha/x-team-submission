"""Step 4: pick a metric that cannot be gamed, then make the class-imbalance choice.

Part one scores classifiers that ignore the input entirely, which exposes each metric's
floor. Part two compares the standard imbalance treatments on paired fold seeds, judged by
the direction of the difference rather than its size. Writes results/04_degenerate.csv,
results/04_methods.csv and results/04_paired.csv.
Run with `uv run python analysis/04_metric_and_imbalance.py`.
"""

from math import comb
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold

from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.model import build_router

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "train.csv"
RESULTS = REPO / "results"

FOLDS = 5
SEED_REPEATS = 10
TARGET_CLASS = "fraud-report"

METHODS = {
    "plain": {},
    "class weights": {"class_weight": "balanced"},
    "oversample minority": {"resample": "over"},
    "undersample majority": {"resample": "under"},
}


def degenerate_baselines(labels: np.ndarray) -> pd.DataFrame:
    """Score classifiers that ignore the input, to expose what each metric rewards.

    Args:
        labels: True labels.

    Returns:
        One row per strategy with every candidate metric. Whatever a strategy that has
        learned nothing can score is that metric's floor, and a metric with a high floor
        flatters the model.
    """
    rng = np.random.default_rng(0)
    classes = np.array(sorted(set(labels)))
    strategies = {
        "always general": np.array(["general"] * len(labels)),
        "always fraud-report": np.array([TARGET_CLASS] * len(labels)),
        "uniform random": rng.choice(classes, len(labels)),
        "random at class rates": rng.choice(labels, len(labels)),
    }
    return pd.DataFrame(
        [
            {
                "strategy": name,
                "accuracy": accuracy_score(labels, predicted),
                "micro_f1": f1_score(labels, predicted, average="micro"),
                "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
                "weighted_f1": f1_score(
                    labels, predicted, average="weighted", zero_division=0
                ),
                "balanced_accuracy": balanced_accuracy_score(labels, predicted),
            }
            for name, predicted in strategies.items()
        ]
    )


def resample_indices(
    indices: np.ndarray, labels: np.ndarray, mode: str, rng: np.random.Generator
) -> np.ndarray:
    """Draw a class-balanced index set from a training fold.

    Args:
        indices: Positions of the training rows.
        labels: True labels for the whole dataset.
        mode: `over` to raise every class to the largest, `under` to cut every class to
            the smallest.
        rng: Generator used to draw the sample.

    Returns:
        Resampled positions, with every class present in equal number.
    """
    groups = {label: indices[labels[indices] == label] for label in set(labels[indices])}
    size = (
        max(len(group) for group in groups.values())
        if mode == "over"
        else min(len(group) for group in groups.values())
    )
    drawn = [
        rng.choice(group, size=size, replace=len(group) < size)
        for group in groups.values()
    ]
    return np.concatenate(drawn)


def score_methods(texts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Evaluate each imbalance treatment on the same folds.

    Args:
        texts: Raw ticket text.
        labels: True labels.

    Returns:
        One row per method and fold seed, carrying the threshold-free metrics, the
        target class's precision and recall, and log loss. Every method sees identical
        folds, so differences are paired rather than confounded with the split.
    """
    classes = np.array(sorted(set(labels)))
    rows = []
    for seed in range(SEED_REPEATS):
        folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=seed).split(texts, labels))
        for name, options in METHODS.items():
            mode = options.get("resample")
            kwargs = {k: v for k, v in options.items() if k != "resample"}
            rng = np.random.default_rng(seed)
            predicted = np.empty(len(labels), dtype=object)
            probabilities = np.zeros((len(labels), len(classes)))
            for train_index, test_index in folds:
                if mode:
                    train_index = resample_indices(train_index, labels, mode, rng)
                model = build_router(**kwargs).fit(texts[train_index], labels[train_index])
                predicted[test_index] = model.predict(texts[test_index])
                probabilities[test_index] = model.predict_proba(texts[test_index])
            rows.append(
                {
                    "method": name,
                    "seed": seed,
                    "macro_f1": f1_score(labels, predicted, average="macro"),
                    "balanced_accuracy": balanced_accuracy_score(labels, predicted),
                    "target_recall": recall_score(
                        labels, predicted, labels=[TARGET_CLASS], average="macro"
                    ),
                    "target_precision": precision_score(
                        labels,
                        predicted,
                        labels=[TARGET_CLASS],
                        average="macro",
                        zero_division=0,
                    ),
                    "log_loss": log_loss(labels, probabilities, labels=list(classes)),
                }
            )
    return pd.DataFrame(rows)


def sign_test(wins: int, losses: int) -> float:
    """Give the two-sided probability of a win/loss split this lopsided under chance.

    Args:
        wins: Seeds on which the method beat the reference.
        losses: Seeds on which it lost.

    Returns:
        The exact two-sided sign-test p-value, ignoring ties. At this sample size the
        direction across paired seeds carries the evidence, not the size of the gap.
    """
    trials = wins + losses
    if trials == 0:
        return 1.0
    extreme = max(wins, losses)
    tail = sum(comb(trials, k) for k in range(extreme, trials + 1)) / 2**trials
    return min(1.0, 2 * tail)


LOWER_IS_BETTER = {"log_loss"}


def pair_against_plain(scores: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Count how often each method beats plain on the same seeds.

    Args:
        scores: The per-method, per-seed score table.
        metric: Column to compare on.

    Returns:
        One row per method with its mean, the win, tie and loss counts against plain,
        and the sign-test p-value for that split. A win always means better, so the
        comparison is flipped for metrics where lower is better.
    """
    reference = scores[scores["method"] == "plain"].set_index("seed")[metric]
    direction = -1 if metric in LOWER_IS_BETTER else 1
    rows = []
    for name in METHODS:
        values = scores[scores["method"] == name].set_index("seed")[metric]
        difference = direction * (values - reference)
        wins = int((difference > 1e-12).sum())
        losses = int((difference < -1e-12).sum())
        rows.append(
            {
                "method": name,
                "metric": metric,
                "mean": values.mean(),
                "sd": values.std(),
                "wins": wins,
                "ties": len(difference) - wins - losses,
                "losses": losses,
                "sign_test_p": sign_test(wins, losses),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Expose the metric floors, then compare the imbalance treatments."""
    frame = load_tickets(DATA)
    texts = frame[TEXT_COLUMN].to_numpy()
    labels = frame[LABEL_COLUMN].to_numpy()
    RESULTS.mkdir(exist_ok=True)

    degenerate = degenerate_baselines(labels)
    degenerate.to_csv(RESULTS / "04_degenerate.csv", index=False)

    scores = score_methods(texts, labels)
    scores.to_csv(RESULTS / "04_methods.csv", index=False)

    paired = pd.concat(
        [
            pair_against_plain(scores, metric)
            for metric in ("macro_f1", "target_recall", "log_loss")
        ]
    )
    paired.to_csv(RESULTS / "04_paired.csv", index=False)

    print("classifiers that ignore the input")
    print(degenerate.to_string(index=False, float_format="{:.3f}".format), end="\n\n")
    print(f"imbalance treatments, {SEED_REPEATS} paired fold seeds")
    print(
        scores.groupby("method")[
            ["macro_f1", "balanced_accuracy", "target_recall", "target_precision", "log_loss"]
        ]
        .mean()
        .to_string(float_format="{:.4f}".format),
        end="\n\n",
    )
    for metric in ("macro_f1", "target_recall", "log_loss"):
        print(f"paired against plain on {metric}")
        print(
            paired[paired["metric"] == metric]
            .drop(columns="metric")
            .to_string(index=False, float_format="{:.4f}".format),
            end="\n\n",
        )
    print(f"[written] {RESULTS}/04_*.csv")


if __name__ == "__main__":
    main()
