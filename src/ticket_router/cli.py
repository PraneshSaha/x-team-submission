"""Command line entry points for training a router and scoring a holdout CSV."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.router import DEFAULT_MISSED_COST, TicketRouter

REPO = Path(__file__).resolve().parents[2]
DEFAULT_TRAINING_DATA = REPO / "train.csv"


def train(data: Path, model: Path, missed_cost: float) -> TicketRouter:
    """Fit a router on labelled tickets and save it.

    Args:
        data: CSV with `text` and `label` columns.
        model: Destination for the fitted router.
        missed_cost: Cost of misrouting the target class, relative to 1 for other errors.

    Returns:
        The fitted router.
    """
    frame = load_tickets(data)
    router = TicketRouter(missed_target_cost=missed_cost).fit(
        frame[TEXT_COLUMN], frame[LABEL_COLUMN]
    )
    router.save(model)
    print(f"trained on {len(frame)} tickets from {data}", file=sys.stderr)
    print(f"saved to {model}", file=sys.stderr)
    return router


def score(
    router: TicketRouter, source: Path, destination: Path, text_column: str
) -> pd.DataFrame:
    """Route every message in a CSV and write the predictions.

    Args:
        router: A fitted router.
        source: CSV of messages to route.
        destination: Where to write the predictions.
        text_column: Name of the column holding the message text.

    Returns:
        The frame that was written, one row per input row in the original order.

    Raises:
        FileNotFoundError: If `source` does not exist.
        ValueError: If `source` has no column named `text_column`.
    """
    if not source.is_file():
        raise FileNotFoundError(f"No input CSV at {source}")
    frame = pd.read_csv(source)
    if text_column not in frame.columns:
        raise ValueError(
            f"{source} has no {text_column!r} column. Found: {sorted(frame.columns)}"
        )
    texts = [str(value) for value in frame[text_column]]
    predictions = router.predict_many(texts)
    output = frame.copy()
    output["predicted_label"] = predictions
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    print(f"wrote {len(output)} predictions to {destination}", file=sys.stderr)
    return output


def load_or_fit(model: Path | None, data: Path, missed_cost: float) -> TicketRouter:
    """Load a saved router, or fit one from the training data if none was given.

    Args:
        model: Path to a saved router, or None.
        data: CSV to fit from when there is no saved router.
        missed_cost: Cost of misrouting the target class, applied either way so the
            trade-off can be changed without retraining.

    Returns:
        A fitted router.
    """
    if model and model.is_file():
        router = TicketRouter.load(model)
        router.missed_target_cost = missed_cost
        return router
    frame = load_tickets(data)
    print(f"no saved model given, fitting on {data}", file=sys.stderr)
    return TicketRouter(missed_target_cost=missed_cost).fit(
        frame[TEXT_COLUMN], frame[LABEL_COLUMN]
    )


def build_parser() -> argparse.ArgumentParser:
    """Describe the command line interface.

    Returns:
        The configured argument parser.
    """
    parser = argparse.ArgumentParser(
        prog="ticket-router", description="Route support tickets to one of four queues."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    trainer = subcommands.add_parser("train", help="fit a router and save it")
    trainer.add_argument("--data", type=Path, default=DEFAULT_TRAINING_DATA)
    trainer.add_argument("--model", type=Path, default=REPO / "model.joblib")
    trainer.add_argument("--missed-cost", type=float, default=DEFAULT_MISSED_COST)

    single = subcommands.add_parser("predict", help="route one message given as text")
    single.add_argument("message")
    single.add_argument("--model", type=Path, default=None)
    single.add_argument("--data", type=Path, default=DEFAULT_TRAINING_DATA)
    single.add_argument("--missed-cost", type=float, default=DEFAULT_MISSED_COST)
    single.add_argument(
        "--probabilities", action="store_true", help="also print each queue's probability"
    )

    scorer = subcommands.add_parser("score", help="route the messages in a CSV")
    scorer.add_argument("--input", type=Path, required=True)
    scorer.add_argument("--output", type=Path, required=True)
    scorer.add_argument("--model", type=Path, default=None)
    scorer.add_argument("--data", type=Path, default=DEFAULT_TRAINING_DATA)
    scorer.add_argument("--text-column", default=TEXT_COLUMN)
    scorer.add_argument("--missed-cost", type=float, default=DEFAULT_MISSED_COST)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the command line interface.

    Args:
        argv: Arguments to parse, defaulting to the process arguments.

    Returns:
        A process exit code, 0 on success and 1 on a handled error.
    """
    arguments = build_parser().parse_args(argv)
    try:
        if arguments.command == "train":
            train(arguments.data, arguments.model, arguments.missed_cost)
            return 0
        router = load_or_fit(
            arguments.model, arguments.data, arguments.missed_cost
        )
        if arguments.command == "predict":
            print(router.predict(arguments.message))
            if arguments.probabilities:
                for queue, probability in sorted(
                    router.predict_proba(arguments.message).items(),
                    key=lambda item: -item[1],
                ):
                    print(f"  {queue:22s} {probability:.3f}", file=sys.stderr)
            return 0
        score(router, arguments.input, arguments.output, arguments.text_column)
        return 0
    except (FileNotFoundError, ValueError, TypeError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
