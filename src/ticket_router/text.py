"""Tokenisation and vocabulary-coverage measures shared by the analysis steps."""

import re
from collections import Counter

TOKEN = re.compile(r"\b\w\w+\b")


def tokenize(text: str) -> list[str]:
    """Split text into the tokens the model will see.

    Args:
        text: A raw ticket.

    Returns:
        Lowercased tokens matching scikit-learn's default token pattern.
    """
    return TOKEN.findall(text.lower())


def missing_mass(documents: list[str]) -> float:
    """Estimate the chance that the next token drawn is one never seen before.

    Args:
        documents: Tickets belonging to a single class.

    Returns:
        The Good-Turing missing mass, the share of the corpus made of tokens occurring
        exactly once. Higher means the vocabulary is still open.
    """
    counts = Counter(token for document in documents for token in tokenize(document))
    total = sum(counts.values())
    hapax = sum(1 for count in counts.values() if count == 1)
    return hapax / total if total else float("nan")
