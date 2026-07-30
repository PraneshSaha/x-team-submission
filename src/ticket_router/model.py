"""The classifier, in the raw form steps 1 to 3 diagnose and the form that ships."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

from ticket_router.normalize import normalize

RANDOM_STATE = 0
LOGISTIC_DEFAULTS = {"max_iter": 5000, "C": 1}


def preprocess(text: str) -> str:
    """Prepare one ticket for vectorising.

    Args:
        text: A raw ticket.

    Returns:
        The ticket with amounts, times and bare numbers collapsed and then lowercased.
        A custom preprocessor replaces scikit-learn's own, which is why the lowercasing
        it would normally do has to happen here.
    """
    return normalize(text).lower()


def build_baseline(**logistic_regression_kwargs) -> Pipeline:
    """Build the pipeline on raw text, with nothing done about imbalance.

    Args:
        **logistic_regression_kwargs: Overrides passed to LogisticRegression, used by
            later steps to add `class_weight` or change `C`.

    Returns:
        An unfitted pipeline mapping raw ticket text to a label. This is the step 1 to 3
        diagnostic, kept unchanged so those results stay reproducible.
    """
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)),
        LogisticRegression(**{**LOGISTIC_DEFAULTS, **logistic_regression_kwargs}),
    )


def build_router(**logistic_regression_kwargs) -> Pipeline:
    """Build the shipped pipeline, which normalises amounts before vectorising.

    Args:
        **logistic_regression_kwargs: Overrides passed to LogisticRegression, used by
            later steps to add `class_weight` or change `C`.

    Returns:
        An unfitted pipeline mapping raw ticket text to a label. Step 3 found money
        normalisation buys no accuracy but costs nothing and removes an unbounded
        surface form, so it is in; the asset vocabulary it also measured is not.
    """
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), preprocessor=preprocess),
        LogisticRegression(**{**LOGISTIC_DEFAULTS, **logistic_regression_kwargs}),
    )
