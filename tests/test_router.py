"""Tests for the shipped router.

Three, chosen for the failures that would be silent rather than loud. A blank ticket
vectorises to all zeros and gets routed anyway; a cost parameter that does not actually
reach the decision leaves the model quietly running on argmax; and a scoring run that
drops or reorders rows misattributes every prediction.
"""

from pathlib import Path

import pandas as pd
import pytest

from ticket_router.cli import main
from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.router import TicketRouter

REPO = Path(__file__).resolve().parents[1]
TRAIN_CSV = REPO / "train.csv"

FRAUD_TICKET = "There is a suspicious transfer to an unknown wallet that I did not make."


@pytest.fixture(scope="module")
def frame():
    return load_tickets(TRAIN_CSV)


@pytest.fixture(scope="module")
def router(frame):
    return TicketRouter().fit(frame[TEXT_COLUMN], frame[LABEL_COLUMN])


def test_predict_returns_a_known_label_and_rejects_unusable_input(router, frame):
    """A ticket routes to a real queue, and unusable input raises rather than routes."""
    assert router.predict(FRAUD_TICKET) in set(frame[LABEL_COLUMN])

    with pytest.raises(ValueError, match="empty"):
        router.predict("   ")
    with pytest.raises(TypeError, match="string"):
        router.predict(None)
    with pytest.raises(RuntimeError, match="not fitted"):
        TicketRouter().predict(FRAUD_TICKET)


def test_missed_cost_widens_the_target_queue_monotonically(router, frame):
    """Raising the cost of a missed fraud can only route more tickets to fraud.

    The cost enters as an additive offset in log-odds, so the set of tickets routed to
    the target class must grow with it. If the parameter were ignored, or applied to the
    wrong axis of the matrix, this would not hold.
    """
    texts = frame[TEXT_COLUMN].tolist()
    flagged = []
    for cost in (1, 5, 10, 26, 100):
        router.missed_target_cost = cost
        routed = {
            index
            for index, label in enumerate(router.predict_many(texts))
            if label == router.target_class
        }
        flagged.append(routed)
    router.missed_target_cost = 10

    for smaller, larger in zip(flagged, flagged[1:]):
        assert smaller <= larger
    assert len(flagged[0]) < len(flagged[-1])


def test_scoring_a_csv_preserves_every_row_in_order(router, tmp_path, frame):
    """The holdout entry point writes one prediction per input row, order intact."""
    source = tmp_path / "messages.csv"
    destination = tmp_path / "predictions.csv"
    messages = pd.DataFrame(
        {TEXT_COLUMN: [FRAUD_TICKET, "I cannot log in to my account.", "Thanks!"]}
    )
    messages.to_csv(source, index=False)
    model = router.save(tmp_path / "model.joblib")

    exit_code = main(
        ["score", "--input", str(source), "--output", str(destination),
         "--model", str(model)]
    )

    assert exit_code == 0
    written = pd.read_csv(destination)
    assert len(written) == len(messages)
    assert written[TEXT_COLUMN].tolist() == messages[TEXT_COLUMN].tolist()
    assert set(written["predicted_label"]) <= set(frame[LABEL_COLUMN])
