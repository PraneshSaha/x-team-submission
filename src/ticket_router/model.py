"""The classifier under test."""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline

RANDOM_STATE = 0


def build_baseline(**logistic_regression_kwargs) -> Pipeline:
    """Build the TF-IDF and logistic regression pipeline with nothing done about imbalance.

    Args:
        **logistic_regression_kwargs: Overrides passed to LogisticRegression, used by
            later steps to add `class_weight` or change `C` without redefining the
            pipeline.

    Returns:
        An unfitted pipeline mapping raw ticket text to a label.
    """
    return make_pipeline(
        TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2)),
        LogisticRegression(max_iter=5000, C=1, **logistic_regression_kwargs),
    )
