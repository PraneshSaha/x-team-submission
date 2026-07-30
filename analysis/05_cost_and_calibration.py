"""Step 5: decide by expected cost, and find out what that needs from the probabilities.

Routing by argmax treats every mistake as equally bad. The brief says one of them is not.
Replacing argmax with a cost rule is a two-line change that reads the probability values
rather than their order, which is the first thing in this repo to do so. Writes
results/05_regularisation.csv, results/05_calibration.csv, results/05_cost_sweep.csv
and results/05_separability_stress.csv.
Run with `uv run python analysis/05_cost_and_calibration.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)

from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.model import build_router

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "train.csv"
RESULTS = REPO / "results"

FOLDS = 5
SEED_REPEATS = 5
TARGET_CLASS = "fraud-report"
COST_GRID = [1, 2, 5, 10, 26, 100]
REGULARISATION_GRID = [0.1, 1.0, 10.0, 100.0, 1000.0]
DIAGNOSTIC_COST = 26
CLIP = 1e-6


def expected_calibration_error(
    probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray, bins: int = 10
) -> float:
    """Measure the gap between how confident the model is and how often it is right.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.
        bins: Number of equal-width confidence bins.

    Returns:
        The expected calibration error, a support-weighted mean of the absolute gap
        between mean confidence and accuracy within each bin. Zero means a stated 0.7
        is right 70% of the time.
    """
    confidence = probabilities.max(axis=1)
    correct = (classes[probabilities.argmax(axis=1)] == labels).astype(float)
    edges = np.linspace(0, 1, bins + 1)
    error = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        inside = (confidence > low) & (confidence <= high)
        if inside.any():
            error += inside.mean() * abs(
                correct[inside].mean() - confidence[inside].mean()
            )
    return float(error)


def brier_score(
    probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> float:
    """Measure squared error between the predicted distribution and the truth.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.

    Returns:
        The multiclass Brier score. Like log loss it is a proper scoring rule, so it is
        minimised only by honest probabilities, which accuracy and macro-F1 are not.
    """
    truth = (labels[:, None] == classes[None, :]).astype(float)
    return float(((probabilities - truth) ** 2).sum(axis=1).mean())


def cost_matrix(classes: np.ndarray, target: str, missed_cost: float) -> np.ndarray:
    """Build the cost of every true-class and predicted-class pairing.

    Args:
        classes: Class names in the column order used by predict_proba.
        target: The class whose misrouting is expensive.
        missed_cost: Cost of routing a target-class ticket anywhere else, relative to a
            cost of 1 for every other mistake.

    Returns:
        A square matrix indexed [true, predicted], zero on the diagonal. The numbers
        come from the business, never from the class frequencies.
    """
    matrix = np.ones((len(classes), len(classes))) - np.eye(len(classes))
    row = list(classes).index(target)
    matrix[row, :] = missed_cost
    matrix[row, row] = 0.0
    return matrix


def decide(probabilities: np.ndarray, classes: np.ndarray, costs: np.ndarray) -> np.ndarray:
    """Route each ticket to whichever queue has the lowest expected cost.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        classes: Class names in the column order used by predict_proba.
        costs: A [true, predicted] cost matrix.

    Returns:
        One predicted label per ticket. With a zero-one cost matrix this reduces to
        argmax, so argmax is the special case that assumes every mistake costs the same.
    """
    return classes[(probabilities @ costs).argmin(axis=1)]


def summarise(
    probabilities: np.ndarray,
    predicted: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    costs: np.ndarray,
) -> dict:
    """Describe one decision rule's behaviour on one set of predictions.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        predicted: Labels chosen by the decision rule.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.
        costs: The [true, predicted] cost matrix the rule was given.

    Returns:
        Errors, realised cost, target-class precision and recall, the share of tickets
        routed to the target queue, and macro-F1.
    """
    rows = np.array([list(classes).index(label) for label in labels])
    columns = np.array([list(classes).index(label) for label in predicted])
    return {
        "errors": int((predicted != labels).sum()),
        "cost": float(costs[rows, columns].sum()),
        "target_recall": recall_score(
            labels, predicted, labels=[TARGET_CLASS], average="macro"
        ),
        "target_precision": precision_score(
            labels, predicted, labels=[TARGET_CLASS], average="macro", zero_division=0
        ),
        "flagged": float((predicted == TARGET_CLASS).mean()),
        "macro_f1": f1_score(labels, predicted, average="macro", zero_division=0),
    }


def out_of_fold(model, texts: np.ndarray, labels: np.ndarray, seed: int) -> np.ndarray:
    """Score every ticket with a model that never saw it.

    Args:
        model: An unfitted estimator exposing predict_proba.
        texts: Raw ticket text.
        labels: True labels.
        seed: Shuffle seed for the fold split.

    Returns:
        An (n_tickets, n_classes) array of out-of-fold predicted probabilities.
    """
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(model, texts, labels, cv=folds, method="predict_proba")


def regularisation_sweep(
    texts: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> pd.DataFrame:
    """Trace what regularisation does to confidence while accuracy stays put.

    Args:
        texts: Raw ticket text.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.

    Returns:
        One row per inverse-regularisation strength with accuracy, mean confidence and
        the errors the expensive cost rule makes. Accuracy is blind to everything the
        cost rule depends on.
    """
    costs = cost_matrix(classes, TARGET_CLASS, DIAGNOSTIC_COST)
    rows = []
    for strength in REGULARISATION_GRID:
        probabilities = out_of_fold(build_router(C=strength), texts, labels, 0)
        accuracy = accuracy_score(labels, classes[probabilities.argmax(axis=1)])
        rows.append(
            {
                "C": strength,
                "accuracy": accuracy,
                "mean_confidence": float(probabilities.max(axis=1).mean()),
                "confidence_gap": accuracy - float(probabilities.max(axis=1).mean()),
                "errors_at_cost": summarise(
                    probabilities,
                    decide(probabilities, classes, costs),
                    labels,
                    classes,
                    costs,
                )["errors"],
            }
        )
    return pd.DataFrame(rows)


def calibrators() -> dict:
    """Name the candidate probability post-processors.

    Returns:
        A mapping from label to an unfitted estimator, with the uncalibrated pipeline
        as the reference.
    """
    return {
        "raw": build_router(),
        "platt": CalibratedClassifierCV(build_router(), method="sigmoid", cv=FOLDS),
        "isotonic": CalibratedClassifierCV(build_router(), method="isotonic", cv=FOLDS),
    }


def calibration_table(
    texts: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare calibrators on probability quality and on cost-rule behaviour.

    Args:
        texts: Raw ticket text.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.

    Returns:
        A pair of tables. The first carries per-calibrator probability quality and the
        cost rule's behaviour at the diagnostic cost; the second sweeps the cost across
        the grid for every calibrator.
    """
    quality_rows = []
    sweep_rows = []
    for name, estimator in calibrators().items():
        for seed in range(SEED_REPEATS):
            probabilities = out_of_fold(estimator, texts, labels, seed)
            clipped = np.clip(probabilities, CLIP, 1)
            clipped = clipped / clipped.sum(axis=1, keepdims=True)
            costs = cost_matrix(classes, TARGET_CLASS, DIAGNOSTIC_COST)
            quality_rows.append(
                {
                    "calibrator": name,
                    "seed": seed,
                    "accuracy": accuracy_score(
                        labels, classes[probabilities.argmax(axis=1)]
                    ),
                    "mean_confidence": float(probabilities.max(axis=1).mean()),
                    "ece": expected_calibration_error(probabilities, labels, classes),
                    "log_loss": log_loss(labels, clipped, labels=list(classes)),
                    "brier": brier_score(probabilities, labels, classes),
                    **summarise(
                        probabilities,
                        decide(probabilities, classes, costs),
                        labels,
                        classes,
                        costs,
                    ),
                }
            )
            for missed_cost in COST_GRID:
                costs = cost_matrix(classes, TARGET_CLASS, missed_cost)
                sweep_rows.append(
                    {
                        "calibrator": name,
                        "seed": seed,
                        "missed_fraud_cost": missed_cost,
                        **summarise(
                            probabilities,
                            decide(probabilities, classes, costs),
                            labels,
                            classes,
                            costs,
                        ),
                    }
                )
    return pd.DataFrame(quality_rows), pd.DataFrame(sweep_rows)


def separability_stress(
    texts: np.ndarray, labels: np.ndarray, classes: np.ndarray
) -> pd.DataFrame:
    """Check whether isotonic's advantage survives when the classes stop separating.

    Args:
        texts: Raw ticket text.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.

    Returns:
        One row per calibrator and stress condition. Isotonic fits a free-form monotone
        step function, which is said to overfit small calibration sets; it wins here
        because separable classes make the true mapping nearly a step. These conditions
        remove that property, by shrinking the data and by injecting label noise, and
        ask whether the win goes with it.
    """
    rng = np.random.default_rng(0)
    noisy = labels.copy()
    corrupted = rng.choice(len(labels), int(0.15 * len(labels)), replace=False)
    noisy[corrupted] = rng.choice(classes, len(corrupted))
    conditions = {"none": (texts, labels)}
    for fraction in (0.5, 0.3):
        sample, _, sample_labels, _ = train_test_split(
            texts, labels, train_size=fraction, stratify=labels, random_state=0
        )
        conditions[f"{fraction:.0%} of the data"] = (sample, sample_labels)
    conditions["15% label noise"] = (texts, noisy)

    costs = cost_matrix(classes, TARGET_CLASS, DIAGNOSTIC_COST)
    rows = []
    for condition, (stressed_texts, stressed_labels) in conditions.items():
        for name, estimator in calibrators().items():
            if name == "raw":
                continue
            for seed in range(SEED_REPEATS):
                probabilities = out_of_fold(
                    estimator, stressed_texts, stressed_labels, seed
                )
                rows.append(
                    {
                        "condition": condition,
                        "calibrator": name,
                        "seed": seed,
                        "accuracy": accuracy_score(
                            stressed_labels, classes[probabilities.argmax(axis=1)]
                        ),
                        "ece": expected_calibration_error(
                            probabilities, stressed_labels, classes
                        ),
                        "brier": brier_score(probabilities, stressed_labels, classes),
                        "errors": summarise(
                            probabilities,
                            decide(probabilities, classes, costs),
                            stressed_labels,
                            classes,
                            costs,
                        )["errors"],
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    """Run the cost rule, show what it needs, and sweep the cost it is given."""
    frame = load_tickets(DATA)
    texts = frame[TEXT_COLUMN].to_numpy()
    labels = frame[LABEL_COLUMN].to_numpy()
    classes = np.array(sorted(set(labels)))
    RESULTS.mkdir(exist_ok=True)

    regularisation = regularisation_sweep(texts, labels, classes)
    regularisation.to_csv(RESULTS / "05_regularisation.csv", index=False)

    quality, sweep = calibration_table(texts, labels, classes)
    quality.to_csv(RESULTS / "05_calibration.csv", index=False)
    sweep.to_csv(RESULTS / "05_cost_sweep.csv", index=False)

    print(f"regularisation sweep, cost of a missed fraud = {DIAGNOSTIC_COST}")
    print(regularisation.to_string(index=False, float_format="{:.4f}".format), end="\n\n")

    print(f"probability quality and the cost rule at cost {DIAGNOSTIC_COST}, "
          f"{SEED_REPEATS} seeds")
    print(
        quality.groupby("calibrator")[
            ["accuracy", "mean_confidence", "ece", "log_loss", "brier",
             "errors", "cost", "target_recall", "target_precision", "flagged"]
        ]
        .mean()
        .to_string(float_format="{:.4f}".format),
        end="\n\n",
    )

    print("cost sweep, mean over seeds")
    for column in ("errors", "target_recall", "target_precision", "flagged"):
        print(f"  {column}")
        print(
            sweep.groupby(["missed_fraud_cost", "calibrator"])[column]
            .mean()
            .unstack()
            .to_string(float_format="{:.3f}".format),
            end="\n\n",
        )
    stress = separability_stress(texts, labels, classes)
    stress.to_csv(RESULTS / "05_separability_stress.csv", index=False)
    print("calibrators as separability is removed")
    print(
        stress.groupby(["condition", "calibrator"])[["accuracy", "ece", "brier", "errors"]]
        .mean()
        .to_string(float_format="{:.4f}".format),
        end="\n\n",
    )
    print(f"true rate of {TARGET_CLASS} in the data: {(labels == TARGET_CLASS).mean():.3f}")
    print(f"[written] {RESULTS}/05_*.csv")


if __name__ == "__main__":
    main()
