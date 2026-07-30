"""Step 3: does normalising amounts and asset names buy anything?

Judged three ways: vocabulary coverage, accuracy on the data as it stands, and accuracy
when held-out tickets name assets the training set never mentioned. Writes
results/03_coverage.csv, results/03_clean_scores.csv and results/03_robustness.csv.
Run with `uv run python analysis/03_normalization.py`.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import StratifiedKFold

from ticket_router.data import LABEL_COLUMN, TEXT_COLUMN, load_tickets
from ticket_router.model import build_baseline
from ticket_router.normalize import (
    MONEY,
    NUMBER,
    TIME,
    AMOUNT_TOKEN,
    NUMBER_TOKEN,
    TIME_TOKEN,
    normalize,
)
from ticket_router.text import missing_mass

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "train.csv"
RESULTS = REPO / "results"

FOLDS = 5
SEED_REPEATS = 10
TARGET_CLASS = "fraud-report"

IN_VOCABULARY_ASSETS = ["AVAX", "MATIC", "DOT", "LINK", "ATOM", "ARB", "TIA", "SUI"]
UNKNOWN_ASSETS = ["MOONX", "VLTR", "ZYNQ", "KRBT", "DFLX", "OMNA"]
UNSEEN_AMOUNTS = ["$73,412", "$8.99", "$1,250,000", "$62", "$9,340", "$412.50"]

ASSET_TOKEN = "cryptoasset"

ASSET_NAMES = {
    "bitcoin cash": "BCH",
    "bitcoin": "BTC",
    "ethereum classic": "ETC",
    "ethereum": "ETH",
    "tether": "USDT",
    "usd coin": "USDC",
    "binance coin": "BNB",
    "ripple": "XRP",
    "solana": "SOL",
    "cardano": "ADA",
    "dogecoin": "DOGE",
    "tron": "TRX",
    "toncoin": "TON",
    "avalanche": "AVAX",
    "shiba inu": "SHIB",
    "polkadot": "DOT",
    "chainlink": "LINK",
    "polygon": "MATIC",
    "litecoin": "LTC",
    "uniswap": "UNI",
    "stellar": "XLM",
    "cosmos": "ATOM",
    "monero": "XMR",
    "filecoin": "FIL",
    "hedera": "HBAR",
    "arbitrum": "ARB",
    "vechain": "VET",
    "algorand": "ALGO",
    "tezos": "XTZ",
    "aptos": "APT",
    "near protocol": "NEAR",
    "internet computer": "ICP",
    "injective": "INJ",
    "immutable": "IMX",
    "thorchain": "RUNE",
    "fantom": "FTM",
    "decentraland": "MANA",
    "the sandbox": "SAND",
    "axie infinity": "AXS",
    "chiliz": "CHZ",
    "zcash": "ZEC",
    "worldcoin": "WLD",
    "kaspa": "KAS",
    "celestia": "TIA",
    "stacks": "STX",
    "pancakeswap": "CAKE",
    "synthetix": "SNX",
    "apecoin": "APE",
    "starknet": "STRK",
    "kusama": "KSM",
    "zilliqa": "ZIL",
    "loopring": "LRC",
    "sushiswap": "SUSHI",
    "decred": "DCR",
    "multiversx": "EGLD",
}

EXTRA_TICKERS = {
    "BUSD", "TUSD", "USDD", "FRAX", "PYUSD", "FDUSD", "USDE", "GUSD", "WBTC", "WETH",
    "STETH", "LEO", "OKB", "CRO", "MNT", "TAO", "WIF", "PYTH", "JTO", "EIGEN", "DAI",
    "SUI", "SEI", "ONDO", "ENA", "BLUR", "GALA", "COMP", "AAVE", "MKR", "GRT", "QNT",
    "LDO", "CRV", "OP", "BCH", "ETC",
}

ASSET_TICKERS = set(ASSET_NAMES.values()) | EXTRA_TICKERS

ASSET_NAME = re.compile(
    r"\b(?:" + "|".join(re.escape(name) for name in ASSET_NAMES) + r")\b",
    re.IGNORECASE,
)
ASSET_TICKER = re.compile(
    r"\b(?:" + "|".join(sorted(ASSET_TICKERS, key=len, reverse=True)) + r")\b"
)


def normalize_with_assets(text: str, collapse: bool = True) -> str:
    """Normalise as the shipped path does, and additionally rewrite crypto asset names.

    Args:
        text: A raw ticket.
        collapse: If True every known asset becomes one shared token, so an unseen asset
            behaves like a seen one. If False each asset name is rewritten to its ticker,
            which unifies spellings but keeps assets distinct.

    Returns:
        The ticket with amounts and assets replaced. This variant is measured here and
        deliberately not shipped, because the vocabulary it needs goes stale.
    """
    text = MONEY.sub(f" {AMOUNT_TOKEN} ", text)
    text = TIME.sub(f" {TIME_TOKEN} ", text)
    if collapse:
        text = ASSET_NAME.sub(f" {ASSET_TOKEN} ", text)
        text = ASSET_TICKER.sub(f" {ASSET_TOKEN} ", text)
    else:
        text = ASSET_NAME.sub(lambda m: f" {ASSET_NAMES[m.group(0).lower()]} ", text)
    text = NUMBER.sub(f" {NUMBER_TOKEN} ", text)
    return " ".join(text.split())


VARIANTS = {
    "raw": lambda text: text,
    "money": normalize,
    "money + assets": normalize_with_assets,
    "money + tickers": lambda text: normalize_with_assets(text, collapse=False),
}


def swap_assets(text: str, replacements: list[str], rng: np.random.Generator) -> str:
    """Rename every crypto asset in a ticket to one the training set never mentioned.

    Args:
        text: A raw ticket.
        replacements: Asset names to draw from.
        rng: Generator used to pick each replacement.

    Returns:
        The ticket with asset names and tickers substituted, simulating a holdout
        ticket about an asset the model was never trained on.
    """
    swapped = ASSET_NAME.sub(lambda _: rng.choice(replacements), text)
    return ASSET_TICKER.sub(lambda _: rng.choice(replacements), swapped)


def swap_amounts(text: str, rng: np.random.Generator) -> str:
    """Rewrite every money amount in a ticket to one the training set never mentioned.

    Args:
        text: A raw ticket.
        rng: Generator used to pick each replacement.

    Returns:
        The ticket with money amounts substituted.
    """
    return MONEY.sub(lambda _: rng.choice(UNSEEN_AMOUNTS), text)


def perturbations(texts: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    """Build the held-out text variants the models are stress-tested on.

    Args:
        texts: Raw ticket text.
        seed: Generator seed, so each fold seed perturbs differently.

    Returns:
        A mapping from condition name to a perturbed copy of `texts`.
    """
    rng = np.random.default_rng(seed)
    return {
        "clean": texts,
        "unseen asset, in vocabulary": np.array(
            [swap_assets(text, IN_VOCABULARY_ASSETS, rng) for text in texts]
        ),
        "unseen asset, out of vocabulary": np.array(
            [swap_assets(text, UNKNOWN_ASSETS, rng) for text in texts]
        ),
        "unseen amounts": np.array([swap_amounts(text, rng) for text in texts]),
    }


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare per-class open vocabulary before and after normalising.

    Args:
        frame: The validated ticket frame.

    Returns:
        One row per class with missing mass under each text variant, which measures
        the intervention without involving a model at all.
    """
    rows = []
    for label, group in frame.groupby(LABEL_COLUMN)[TEXT_COLUMN]:
        documents = group.tolist()
        row = {"label": str(label), "documents": len(documents)}
        for name, transform in VARIANTS.items():
            row[name] = missing_mass([transform(text) for text in documents])
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate(texts: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """Score every text variant on clean and on perturbed held-out tickets.

    Args:
        texts: Raw ticket text.
        labels: True labels.

    Returns:
        One row per variant, condition and fold seed, carrying macro-F1 and recall on
        the target class. Models always train on unperturbed text, so the perturbation
        acts only on the held-out side, as a hidden holdout would.
    """
    rows = []
    for seed in range(SEED_REPEATS):
        conditions = perturbations(texts, seed)
        folds = StratifiedKFold(FOLDS, shuffle=True, random_state=seed)
        for name, transform in VARIANTS.items():
            transformed = {
                condition: np.array([transform(text) for text in variant])
                for condition, variant in conditions.items()
            }
            predictions = {condition: np.empty(len(labels), dtype=object) for condition in conditions}
            for train_index, test_index in folds.split(texts, labels):
                model = build_baseline().fit(
                    transformed["clean"][train_index], labels[train_index]
                )
                for condition in conditions:
                    predictions[condition][test_index] = model.predict(
                        transformed[condition][test_index]
                    )
            for condition, predicted in predictions.items():
                rows.append(
                    {
                        "variant": name,
                        "condition": condition,
                        "seed": seed,
                        "macro_f1": f1_score(labels, predicted, average="macro"),
                        "target_recall": recall_score(
                            labels, predicted, labels=[TARGET_CLASS], average="macro"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def paired_summary(scores: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Compare each variant against raw on the same fold seeds.

    Args:
        scores: The per-seed score table.
        condition: The held-out condition to compare within.

    Returns:
        One row per variant with mean macro-F1 and target recall, and the number of
        seeds on which it beats, ties and loses to raw. Direction across paired seeds
        is what carries evidence at this sample size, not the size of the mean gap.
    """
    subset = scores[scores["condition"] == condition]
    reference = subset[subset["variant"] == "raw"].set_index("seed")
    rows = []
    for name in VARIANTS:
        variant = subset[subset["variant"] == name].set_index("seed")
        difference = variant["macro_f1"] - reference["macro_f1"]
        rows.append(
            {
                "variant": name,
                "macro_f1": variant["macro_f1"].mean(),
                "macro_f1_sd": variant["macro_f1"].std(),
                "target_recall": variant["target_recall"].mean(),
                "wins": int((difference > 0).sum()),
                "ties": int((difference == 0).sum()),
                "losses": int((difference < 0).sum()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    """Measure normalisation on coverage, on clean accuracy and under perturbation."""
    frame = load_tickets(DATA)
    texts = frame[TEXT_COLUMN].to_numpy()
    labels = frame[LABEL_COLUMN].to_numpy()
    RESULTS.mkdir(exist_ok=True)

    coverage = coverage_table(frame)
    coverage.to_csv(RESULTS / "03_coverage.csv", index=False)

    scores = evaluate(texts, labels)
    clean = paired_summary(scores, "clean")
    clean.to_csv(RESULTS / "03_clean_scores.csv", index=False)

    robustness = (
        scores.groupby(["condition", "variant"])[["macro_f1", "target_recall"]]
        .mean()
        .reset_index()
    )
    robustness.to_csv(RESULTS / "03_robustness.csv", index=False)

    print("missing mass by class")
    print(coverage.to_string(index=False, float_format="{:.4f}".format), end="\n\n")
    print(f"clean held-out text, {SEED_REPEATS} fold seeds, paired against raw")
    print(clean.to_string(index=False, float_format="{:.4f}".format), end="\n\n")
    print("held-out text perturbed")
    print(
        robustness.pivot(
            index="condition", columns="variant", values="macro_f1"
        ).to_string(float_format="{:.4f}".format),
        end="\n\n",
    )
    print("target-class recall under the same conditions")
    print(
        robustness.pivot(
            index="condition", columns="variant", values="target_recall"
        ).to_string(float_format="{:.4f}".format),
        end="\n\n",
    )
    print(f"[written] {RESULTS}/03_*.csv")


if __name__ == "__main__":
    main()
