"""The shipped router: calibrated probabilities routed by expected cost."""

from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from ticket_router.costs import cost_matrix, decide
from ticket_router.model import build_router

DEFAULT_TARGET_CLASS = "fraud-report"
DEFAULT_MISSED_COST = 10.0
CALIBRATION_FOLDS = 5
MAX_TICKET_CHARACTERS = 10_000


class TicketRouter:
    """Routes a support ticket to one of the labelled queues by expected cost."""

    def __init__(
        self,
        missed_target_cost: float = DEFAULT_MISSED_COST,
        target_class: str = DEFAULT_TARGET_CLASS,
    ) -> None:
        """Configure the router's cost trade-off.

        Args:
            missed_target_cost: Cost of routing a `target_class` ticket to the wrong
                queue, relative to 1 for every other mistake. 1 reproduces argmax. The
                default of 10 is the smallest value at which no fraud report was missed
                in step 5, and is a placeholder for a number the business should set.
            target_class: The class whose misrouting carries that cost.
        """
        self.missed_target_cost = missed_target_cost
        self.target_class = target_class
        self.pipeline_ = None
        self.classes_ = None

    def fit(self, texts, labels) -> "TicketRouter":
        """Fit the calibrated pipeline on labelled tickets.

        Args:
            texts: Raw ticket text, one per ticket.
            labels: The queue each ticket belongs to.

        Returns:
            This router, fitted.

        Raises:
            ValueError: If the target class is absent, or a class has too few examples
                to calibrate on.
        """
        texts = np.asarray(texts, dtype=object)
        labels = np.asarray(labels, dtype=object)
        classes, counts = np.unique(labels, return_counts=True)
        if self.target_class not in classes:
            raise ValueError(
                f"target_class {self.target_class!r} is absent from the training "
                f"labels: {sorted(classes)}"
            )
        if counts.min() < CALIBRATION_FOLDS:
            scarce = classes[counts.argmin()]
            raise ValueError(
                f"class {scarce!r} has {counts.min()} examples, fewer than the "
                f"{CALIBRATION_FOLDS} calibration folds require"
            )
        self.pipeline_ = CalibratedClassifierCV(
            build_router(), method="isotonic", cv=CALIBRATION_FOLDS
        ).fit(texts, labels)
        self.classes_ = np.asarray(self.pipeline_.classes_)
        return self

    def predict(self, text: str) -> str:
        """Route one ticket.

        Args:
            text: The raw ticket text.

        Returns:
            The name of the queue to route it to.

        Raises:
            RuntimeError: If the router has not been fitted.
            TypeError: If `text` is not a string.
            ValueError: If `text` is blank once stripped, or implausibly long.
        """
        return self.predict_many([text])[0]

    def predict_many(self, texts) -> list[str]:
        """Route a batch of tickets.

        Args:
            texts: An iterable of raw ticket texts.

        Returns:
            One queue name per ticket, in the order given.

        Raises:
            RuntimeError: If the router has not been fitted.
            TypeError: If any ticket is not a string.
            ValueError: If any ticket is blank once stripped, or implausibly long.
        """
        cleaned = np.asarray([self._validate(text) for text in texts], dtype=object)
        if not len(cleaned):
            return []
        probabilities = self._require_fitted().predict_proba(cleaned)
        costs = cost_matrix(
            self.classes_, self.target_class, self.missed_target_cost
        )
        return list(decide(probabilities, self.classes_, costs))

    def predict_proba(self, text: str) -> dict[str, float]:
        """Give the calibrated probability of each queue for one ticket.

        Args:
            text: The raw ticket text.

        Returns:
            A mapping from queue name to probability. Exposed because the routing
            decision reads these values, so anything auditing the decision needs them.

        Raises:
            RuntimeError: If the router has not been fitted.
            TypeError: If `text` is not a string.
            ValueError: If `text` is blank once stripped, or implausibly long.
        """
        cleaned = self._validate(text)
        probabilities = self._require_fitted().predict_proba([cleaned])[0]
        return dict(zip(self.classes_, (float(value) for value in probabilities)))

    def save(self, path: str | Path) -> Path:
        """Write the fitted router to disk.

        Args:
            path: Destination file.

        Returns:
            The path written.

        Raises:
            RuntimeError: If the router has not been fitted.
        """
        self._require_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path: str | Path) -> "TicketRouter":
        """Read a fitted router from disk.

        Args:
            path: File written by `save`.

        Returns:
            The fitted router.

        Raises:
            FileNotFoundError: If no file exists at `path`.
        """
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"No saved router at {path}")
        return joblib.load(path)

    def _require_fitted(self):
        """Return the fitted pipeline, or explain that there is not one.

        Returns:
            The fitted calibrated pipeline.

        Raises:
            RuntimeError: If `fit` has not been called.
        """
        if self.pipeline_ is None:
            raise RuntimeError("TicketRouter is not fitted; call fit() or load() first")
        return self.pipeline_

    @staticmethod
    def _validate(text: str) -> str:
        """Check one ticket is usable and return it stripped.

        Args:
            text: The raw ticket text.

        Returns:
            The text with surrounding whitespace removed.

        Raises:
            TypeError: If `text` is not a string.
            ValueError: If it is blank once stripped, or longer than the cap. A blank
                ticket vectorises to all zeros and would be routed to whichever class
                the intercepts favour, silently.
        """
        if not isinstance(text, str):
            raise TypeError(f"ticket text must be a string, got {type(text).__name__}")
        stripped = text.strip()
        if not stripped:
            raise ValueError("ticket text is empty once stripped")
        if len(stripped) > MAX_TICKET_CHARACTERS:
            raise ValueError(
                f"ticket text is {len(stripped)} characters, over the "
                f"{MAX_TICKET_CHARACTERS} cap"
            )
        return stripped
