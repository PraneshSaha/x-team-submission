"""Step 1: does class imbalance exist, and is the smallest class representative?

Writes results/01_class_counts.csv, results/01_coverage.csv and
results/01_vocab_growth.csv. Run with `uv run python analysis/01_distribution.py`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from ticket_router.data import (
    LABEL_COLUMN,
    TEXT_COLUMN,
    class_counts,
    imbalance_ratio,
    load_tickets,
)
from ticket_router.text import missing_mass, tokenize

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "train.csv"
RESULTS = REPO / "results"

SHUFFLES = 200
SEED = 0


def coverage_at(
    documents: list[str], size: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Measure vocabulary growth and open vocabulary at a fixed sample size.

    Args:
        documents: Tickets belonging to a single class.
        size: Documents to draw per repeat, held equal across classes so the result
            compares coverage rather than restating class size.
        rng: Generator used to draw the SHUFFLES random subsets and orderings.

    Returns:
        A pair of the mean rarefaction curve, distinct vocabulary after 1, 2, ...
        `size` documents, and the per-repeat missing masses. A curve still climbing
        at its right-hand end means new documents are still bringing new words.
    """
    curves = np.zeros((SHUFFLES, size))
    masses = np.zeros(SHUFFLES)
    for shuffle in range(SHUFFLES):
        drawn = rng.permutation(len(documents))[:size]
        seen: set[str] = set()
        for step, position in enumerate(drawn):
            seen.update(tokenize(documents[position]))
            curves[shuffle, step] = len(seen)
        masses[shuffle] = missing_mass([documents[position] for position in drawn])
    return curves.mean(axis=0), masses


def main() -> None:
    """Measure class balance and per-class vocabulary coverage, and write the tables."""
    frame = load_tickets(DATA)
    counts = class_counts(frame)
    labels = [str(label) for label in counts.index]
    n_smallest = int(counts.iloc[-1])
    rng = np.random.default_rng(SEED)
    RESULTS.mkdir(exist_ok=True)

    always_majority = [labels[0]] * len(frame)
    truth = frame[LABEL_COLUMN]

    counts_table = pd.DataFrame(
        {
            "label": labels,
            "count": [int(counts[label]) for label in labels],
            "share": [int(counts[label]) / len(frame) for label in labels],
        }
    )
    counts_table.to_csv(RESULTS / "01_class_counts.csv", index=False)

    coverage_rows = []
    growth_rows = []
    for label in labels:
        documents = frame.loc[frame[LABEL_COLUMN] == label, TEXT_COLUMN].tolist()
        tokens = [token for document in documents for token in tokenize(document)]
        curve, masses = coverage_at(documents, n_smallest, rng)
        coverage_rows.append(
            {
                "label": label,
                "documents": len(documents),
                "tokens": len(tokens),
                "vocabulary": len(set(tokens)),
                "mean_words": float(np.mean([len(tokenize(d)) for d in documents])),
                "missing_mass_full": missing_mass(documents),
                "missing_mass": float(masses.mean()),
                "missing_mass_sd": float(masses.std()),
                "vocabulary_at_equal_n": float(curve[-1]),
                "new_words_on_last_doc": float(curve[-1] - curve[-2]),
            }
        )
        growth_rows.extend(
            {"label": label, "documents_seen": step + 1, "vocabulary": float(size)}
            for step, size in enumerate(curve)
        )

    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(RESULTS / "01_coverage.csv", index=False)
    pd.DataFrame(growth_rows).to_csv(RESULTS / "01_vocab_growth.csv", index=False)

    lengths = frame[TEXT_COLUMN].map(lambda text: len(tokenize(text)))
    least_covered = coverage.loc[coverage["missing_mass"].idxmax()]

    print(f"rows {len(frame)}  classes {len(labels)}")
    print(counts_table.to_string(index=False, float_format="{:.3f}".format), end="\n\n")
    print(f"imbalance ratio {imbalance_ratio(frame):.1f}:1  smallest class {n_smallest}", end="\n\n")
    print(
        f"always-{labels[0]} baseline: "
        f"accuracy {accuracy_score(truth, always_majority):.3f}  "
        f"macro-F1 {f1_score(truth, always_majority, average='macro'):.3f}",
    end = "\n\n"
    )
    print(
        f"duplicate texts {int(frame[TEXT_COLUMN].duplicated().sum())}  "
        f"words mean {lengths.mean():.1f} min {int(lengths.min())} "
        f"max {int(lengths.max())}",
        end="\n\n"
    )
    print(coverage.to_string(index=False, float_format="{:.3f}".format), end="\n\n")
    print(f"least covered at equal sample size: {least_covered['label']}", end="\n\n")
    print(f"[written] {RESULTS}/01_class_counts.csv, 01_coverage.csv, 01_vocab_growth.csv")


if __name__ == "__main__":
    main()
