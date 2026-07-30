"""Collapsing open-ended surface forms to fixed tokens.

Money amounts, times and bare numbers are unbounded, so a new ticket can always carry one
never seen in training, and none of them decides the route. Collapsing them needs no word
list to be kept up to date.
"""

import re

AMOUNT_TOKEN = "moneyamount"
NUMBER_TOKEN = "numbervalue"
TIME_TOKEN = "clocktime"

MONEY = re.compile(r"[$€£]\s?\d[\d,]*(?:\.\d+)?")
TIME = re.compile(r"\b\d{1,2}(?::\d{2})?\s?[ap]\.?m\.?", re.IGNORECASE)
NUMBER = re.compile(r"(?<![\w$€£])\d[\d,]*(?:\.\d+)?(?![\w])")


def normalize(text: str) -> str:
    """Replace money amounts, clock times and bare numbers with fixed tokens.

    Args:
        text: A raw ticket.

    Returns:
        The ticket with those surface forms replaced. Replacement tokens are plain words
        so they survive scikit-learn's default token pattern, and the guards on the
        number pattern leave things like `2FA` intact.
    """
    text = MONEY.sub(f" {AMOUNT_TOKEN} ", text)
    text = TIME.sub(f" {TIME_TOKEN} ", text)
    text = NUMBER.sub(f" {NUMBER_TOKEN} ", text)
    return " ".join(text.split())
