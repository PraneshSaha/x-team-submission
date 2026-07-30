"""Step 2: fit a baseline with nothing done about imbalance and read its errors.

Writes results/02_scores.csv, results/02_per_class.csv, results/02_confusion.csv,
results/02_errors.csv, results/02_learning_curve.csv and results/02_split_variance.csv.
Run with `uv run python analysis/02_baseline.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    StratifiedKFold,
    cross_val_predict,
    learning_curve,
    train_test_split,
)

from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.model import build_baseline

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "train.csv"
RESULTS = REPO / "results"

FOLDS = 5
SEED = 0
SEED_REPEATS = 10
SPLIT_SEEDS = 40
TEST_SIZE = 0.2
TARGET_CLASS = "fraud-report"


def out_of_fold_probabilities(
    texts: np.ndarray, labels: np.ndarray, seed: int
) -> np.ndarray:
    """Score every ticket with a model that never saw it during training.

    Args:
        texts: Raw ticket text.
        labels: True labels, used only for the stratified fold assignment.
        seed: Shuffle seed for the fold split.

    Returns:
        An (n_tickets, n_classes) array of predicted probabilities, columns ordered
        by sorted class name.
    """
    folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    return cross_val_predict(
        build_baseline(), texts, labels, cv=folds, method="predict_proba"
    )


def threshold_predictions(
    probabilities: np.ndarray, classes: np.ndarray, target: str, cut: float
) -> np.ndarray:
    """Predict the target class whenever its score clears a cut, else the best of the rest.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        classes: Class names in the column order used by predict_proba.
        target: The class the cut applies to.
        cut: Score above which the target class is predicted.

    Returns:
        One predicted label per ticket.
    """
    column = list(classes).index(target)
    others = [index for index in range(len(classes)) if index != column]
    fallback = classes[others][probabilities[:, others].argmax(axis=1)]
    return np.where(probabilities[:, column] > cut, target, fallback)


def pick_threshold(
    probabilities: np.ndarray, labels: np.ndarray, classes: np.ndarray, target: str
) -> float:
    """Choose the target-score cut that makes the fewest errors, breaking ties centrally.

    Args:
        probabilities: An (n_tickets, n_classes) array of predicted probabilities.
        labels: True labels for those tickets.
        classes: Class names in the column order used by predict_proba.
        target: The class the cut applies to.

    Returns:
        The median of the cuts that tie for fewest errors, which sits furthest from
        either edge of the winning interval.
    """
    grid = np.linspace(0.01, 0.99, 197)
    errors = np.array(
        [
            (threshold_predictions(probabilities, classes, target, cut) != labels).sum()
            for cut in grid
        ]
    )
    return float(np.median(grid[errors == errors.min()]))


def score_separation(
    texts: np.ndarray, labels: np.ndarray, classes: np.ndarray, target: str
) -> pd.DataFrame:
    """Test whether the errors on one class are an overlap problem or a decision problem.

    Args:
        texts: Raw ticket text.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.
        target: The class whose errors are under investigation.

    Returns:
        One row per fold seed giving the errors the argmax rule makes, the lowest score
        the target class receives, the highest score any other class receives, the
        errors left by the best cut chosen on the same predictions, and the errors left
        by a cut chosen inside each training fold and applied to the held-out fold. A
        positive gap means the two groups are separable, so no information is missing.
        The two threshold columns differ by the optimism of fitting the cut on the data
        it is scored on.
    """
    column = list(classes).index(target)
    rows = []
    for seed in range(SEED_REPEATS):
        probabilities = out_of_fold_probabilities(texts, labels, seed)
        scores = probabilities[:, column]
        is_target = labels == target
        in_sample_cut = pick_threshold(probabilities, labels, classes, target)
        rows.append(
            {
                "seed": seed,
                "argmax_errors": int(
                    (classes[probabilities.argmax(axis=1)] != labels).sum()
                ),
                "lowest_score_on_target": float(scores[is_target].min()),
                "highest_score_on_others": float(scores[~is_target].max()),
                "gap": float(scores[is_target].min() - scores[~is_target].max()),
                "in_sample_threshold_errors": int(
                    (
                        threshold_predictions(
                            probabilities, classes, target, in_sample_cut
                        )
                        != labels
                    ).sum()
                ),
                "nested_threshold_errors": nested_threshold_errors(
                    texts, labels, classes, target, seed
                ),
            }
        )
    return pd.DataFrame(rows)


def nested_threshold_errors(
    texts: np.ndarray,
    labels: np.ndarray,
    classes: np.ndarray,
    target: str,
    seed: int,
) -> int:
    """Score a threshold rule without letting it see the data it is judged on.

    Args:
        texts: Raw ticket text.
        labels: True labels.
        classes: Class names in the column order used by predict_proba.
        target: The class the cut applies to.
        seed: Shuffle seed for both the outer and inner fold splits.

    Returns:
        Total errors over all held-out folds, where each fold's cut was chosen on
        out-of-fold predictions from that fold's training data only.
    """
    outer = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
    errors = 0
    for train_index, test_index in outer.split(texts, labels):
        inner = out_of_fold_probabilities(texts[train_index], labels[train_index], seed)
        cut = pick_threshold(inner, labels[train_index], classes, target)
        model = build_baseline().fit(texts[train_index], labels[train_index])
        held_out = model.predict_proba(texts[test_index])
        predictions = threshold_predictions(held_out, classes, target, cut)
        errors += int((predictions != labels[test_index]).sum())
    return errors


def single_split_spread(texts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Score the same model on many different single train/test splits.

    Args:
        texts: Raw ticket text.
        labels: True labels.

    Returns:
        One row per split seed with the macro-F1 that split would have reported.
    """
    rows = []
    for seed in range(SPLIT_SEEDS):
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, labels, test_size=TEST_SIZE, stratify=labels, random_state=seed
        )
        model = build_baseline().fit(train_texts, train_labels)
        rows.append(
            {
                "seed": seed,
                "macro_f1": f1_score(
                    test_labels, model.predict(test_texts), average="macro"
                ),
            }
        )
    return pd.DataFrame(rows)


def data_hunger(texts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Trace macro-F1 against training-set size to see whether the model is saturated.

    Args:
        texts: Raw ticket text.
        labels: True labels.

    Returns:
        One row per training size with mean and standard deviation of macro-F1 across
        folds. A curve still climbing at the largest size means the score is being
        read off the steep part, so more data would still buy accuracy.
    """
    sizes, _, test_scores = learning_curve(
        build_baseline(),
        texts,
        labels,
        train_sizes=np.linspace(0.2, 1.0, 9),
        cv=StratifiedKFold(FOLDS, shuffle=True, random_state=SEED),
        scoring="f1_macro",
    )
    return pd.DataFrame(
        {
            "train_size": sizes,
            "macro_f1": test_scores.mean(axis=1),
            "macro_f1_sd": test_scores.std(axis=1),
        }
    )


def main() -> None:
    """Run the baseline diagnostic and write every table it produces."""
    frame = load_tickets(DATA)
    texts = frame[TEXT_COLUMN].to_numpy()
    labels = frame[LABEL_COLUMN].to_numpy()
    classes = np.array(sorted(set(labels)))
    RESULTS.mkdir(exist_ok=True)

    probabilities = out_of_fold_probabilities(texts, labels, SEED)
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    accuracy = accuracy_score(labels, predictions)

    scores = pd.DataFrame(
        [
            {
                "accuracy": accuracy,
                "macro_f1": f1_score(labels, predictions, average="macro"),
                "balanced_accuracy": balanced_accuracy_score(labels, predictions),
                "log_loss": log_loss(labels, probabilities, labels=list(classes)),
                "mean_confidence": float(confidence.mean()),
                "confidence_gap": float(accuracy - confidence.mean()),
                "errors": int((predictions != labels).sum()),
            }
        ]
    )
    scores.to_csv(RESULTS / "02_scores.csv", index=False)

    report = classification_report(
        labels, predictions, output_dict=True, zero_division=0
    )
    per_class = (
        pd.DataFrame(report)
        .transpose()
        .loc[list(classes)]
        .reset_index(names="label")
    )
    per_class.to_csv(RESULTS / "02_per_class.csv", index=False)

    confusion = pd.DataFrame(
        confusion_matrix(labels, predictions, labels=list(classes)),
        index=pd.Index(classes, name="true"),
        columns=pd.Index(classes, name="predicted"),
    )
    confusion.to_csv(RESULTS / "02_confusion.csv")

    wrong = predictions != labels
    errors = pd.DataFrame(
        {
            "text": texts[wrong],
            "true": labels[wrong],
            "predicted": predictions[wrong],
            "confidence": confidence[wrong],
        }
    )
    errors.to_csv(RESULTS / "02_errors.csv", index=False)

    separation = score_separation(texts, labels, classes, TARGET_CLASS)
    separation.to_csv(RESULTS / "02_score_separation.csv", index=False)

    curve = data_hunger(texts, labels)
    curve.to_csv(RESULTS / "02_learning_curve.csv", index=False)

    spread = single_split_spread(texts, labels)
    spread.to_csv(RESULTS / "02_split_variance.csv", index=False)

    print(scores.to_string(index=False, float_format="{:.4f}".format), end="\n\n")
    print(per_class.to_string(index=False, float_format="{:.3f}".format), end="\n\n")
    print(confusion.to_string(), end="\n\n")
    print(f"errors: {int(wrong.sum())} of {len(labels)}")
    print(errors["true"].value_counts().to_string(), end="\n\n")
    print(separation.to_string(index=False, float_format="{:.3f}".format), end="\n\n")
    print(
        f"gap positive on {int((separation['gap'] > 0).sum())} of {SEED_REPEATS} seeds\n"
        f"errors per seed, mean: argmax {separation['argmax_errors'].mean():.1f}  "
        f"in-sample cut {separation['in_sample_threshold_errors'].mean():.1f}  "
        f"nested cut {separation['nested_threshold_errors'].mean():.1f}",
        end="\n\n",
    )
    print(curve.to_string(index=False, float_format="{:.4f}".format), end="\n\n")
    print(
        f"single 80/20 split over {SPLIT_SEEDS} seeds: "
        f"macro-F1 min {spread['macro_f1'].min():.3f} "
        f"max {spread['macro_f1'].max():.3f} "
        f"mean {spread['macro_f1'].mean():.3f} "
        f"sd {spread['macro_f1'].std():.3f}"
    )
    print(f"[written] {RESULTS}/02_*.csv")


if __name__ == "__main__":
    main()
