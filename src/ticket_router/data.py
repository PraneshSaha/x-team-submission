"""Loading and validating the ticket dataset."""

from pathlib import Path

import pandas as pd

TEXT_COLUMN = "text"
LABEL_COLUMN = "label"


def load_tickets(path: str | Path) -> pd.DataFrame:
    """Read a ticket CSV and return it validated and whitespace-stripped.

    Args:
        path: Path to a CSV carrying at least a `text` and a `label` column.

    Returns:
        A frame with exactly the `text` and `label` columns, no nulls, no blank text.

    Raises:
        FileNotFoundError: If no file exists at `path`.
        ValueError: If a required column is missing, a cell is null, or a ticket is
            blank once stripped.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"No ticket CSV at {path}")

    frame = pd.read_csv(path)

    missing = {TEXT_COLUMN, LABEL_COLUMN} - set(frame.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required column(s): {sorted(missing)}. "
            f"Found: {sorted(frame.columns)}"
        )

    nulls = frame[[TEXT_COLUMN, LABEL_COLUMN]].isna().sum()
    if nulls.any():
        raise ValueError(f"{path} has null cells: {nulls[nulls > 0].to_dict()}")

    stripped = {
        column: [str(value).strip() for value in frame[column]]
        for column in (TEXT_COLUMN, LABEL_COLUMN)
    }
    blank = sum(1 for text in stripped[TEXT_COLUMN] if not text)
    if blank:
        raise ValueError(f"{path} has {blank} row(s) whose text is blank once stripped")

    return pd.DataFrame(stripped)


def class_counts(frame: pd.DataFrame) -> pd.Series:
    """Count the rows of each label, largest class first.

    Args:
        frame: A frame carrying a `label` column.

    Returns:
        Counts indexed by label, sorted descending.
    """
    return frame[LABEL_COLUMN].value_counts()


def imbalance_ratio(frame: pd.DataFrame) -> float:
    """Measure how much larger the biggest class is than the smallest.

    Args:
        frame: A frame carrying a `label` column.

    Returns:
        The largest class count divided by the smallest.
    """
    counts = class_counts(frame)
    return int(counts.iloc[0]) / int(counts.iloc[-1])
