# x-team-submission

Submission for x-team problem: route a customer support ticket to one of four teams.

## Setup

Dependencies are managed with [uv](https://docs.astral.sh/uv/). `uv.lock` is committed, so
the environment below is the exact one every number in this README was produced on.

```bash
uv sync            # creates .venv and installs the locked dependency set
uv run pytest      # run the tests
```

Anything prefixed with `uv run` executes inside that environment. No manual activation needed.

## How to read this repo

The commit history is the argument. Each commit takes one assumption about the data,
tests it, and records what came back in `results/`. The README grows the same way, so
reading it top to bottom is the same path as reading the commits in order.

---

## Step 1: does imbalance exist, and is the smallest class covered?

```bash
uv run python analysis/01_distribution.py
```

Writes `results/01_class_counts.csv`, `results/01_coverage.csv`,
`results/01_vocab_growth.csv`.

400 rows, four classes: `general` 160, `account-access` 100, `transaction-dispute` 90,
`fraud-report` 50. Largest over smallest is **3.2:1**.

**Does imbalance exist?** Yes, but mildly. The reflex worry is the one where a constant
predictor scores well on accuracy. Here always predicting `general` gives 40% accuracy
and 0.143 macro-F1, so that specific failure mode is not what is at stake. A ratio alone
cannot tell whether the imbalance costs anything. That depends on whether the classes
overlap, which only a fitted model can answer.

**Is the smallest class representative?** This is the question the ratio cannot reach,
and is worth knowing. Because otherwise we have out-of-distribution examples in the hidden set.
3.2:1 with 50 minority examples is a different problem from 100:1 with 10,000, 
because what limits coverage is the absolute count.

So here we measure it directly, with Good-Turing missing mass: the share of a class's tokens
that appear exactly once, which estimates the probability that the next ticket in that
class uses a word the class has never used. High means the vocabulary is still open.

I need it two ways, because they answer different questions.

| class | docs | tokens | missing mass, all docs | missing mass, 50 docs each |
| --- | ---: | ---: | ---: | ---: |
| general | 160 | 1933 | 0.001 | **0.031** ± 0.012 |
| account-access | 100 | 1605 | 0.002 | 0.012 ± 0.006 |
| transaction-dispute | 90 | 1473 | 0.003 | 0.016 ± 0.006 |
| fraud-report | 50 | 762 | **0.020** | 0.020 |

The two columns disagree, and the disagreement is the finding.

**On the data we actually have**, `fraud-report` is the least covered class by an order
of magnitude, 0.020 against 0.001 to 0.003. That is the number that matters operationally.

**At equal sample size**, all concepts appear equal, and more specifically fraud-report is not different than others.
It is narrower than `general`. Its coverage gap is a pure sample-size effect: it is the only
class that never gets past 50 documents. `general` starts out broader and still ends up
better covered, because it gets 160.

That distinction changes what would fix it. If fraud were intrinsically unbounded, more
examples would not close the gap. Since it is not, more fraud examples would, and
`general`'s curve says roughly how many: 160 documents took the broadest class down to
0.001.

