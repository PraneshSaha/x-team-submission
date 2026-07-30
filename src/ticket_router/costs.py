"""Turning probabilities into routing decisions under an asymmetric cost."""

import numpy as np


def cost_matrix(classes: np.ndarray, target: str, missed_cost: float) -> np.ndarray:
    """Build the cost of every true-class and predicted-class pairing.

    Args:
        classes: Class names in the column order used by predict_proba.
        target: The class whose misrouting is expensive.
        missed_cost: Cost of routing a target-class ticket anywhere else, relative to a
            cost of 1 for every other mistake.

    Returns:
        A square matrix indexed [true, predicted], zero on the diagonal.

    Raises:
        ValueError: If `target` is not one of `classes`, or `missed_cost` is below 1.
    """
    if target not in classes:
        raise ValueError(f"{target!r} is not one of the classes: {sorted(classes)}")
    if missed_cost < 1:
        raise ValueError(f"missed_cost must be at least 1, got {missed_cost}")
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
