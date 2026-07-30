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


---

## Step 2: a baseline with nothing done about imbalance

```bash
uv run python analysis/02_baseline.py
```

Writes `results/02_scores.csv`, `02_per_class.csv`, `02_confusion.csv`, `02_errors.csv`,
`02_score_separation.csv`, `02_learning_curve.csv`, `02_split_variance.csv`.

TF-IDF (1-2 grams, sublinear) plus logistic regression -> scored on 5-fold stratified cross val.

This tells us whether we -
1. need to worry about imbalance
2. have clean separation or not
3. can trust the probabilities, from log loss and the confidence gap

| | |
| --- | ---: |
| accuracy | 0.9925 |
| macro-F1 | 0.9893 |
| balanced accuracy | 0.9850 |
| log loss | 0.4019 |
| mean confidence | 0.6841 |
| confidence gap | +0.3084 |
| errors | 3 of 400 |

Per class:

| class | precision | recall | support |
| --- | ---: | ---: | ---: |
| account-access | 1.000 | 1.000 | 100 |
| general | 0.988 | 1.000 | 160 |
| transaction-dispute | 0.989 | 1.000 | 90 |
| fraud-report | **1.000** | **0.940** | 50 |

**All three errors are `fraud-report`.** Every other class is perfect. So, imbalance is not significant
and error is in the class that is highest priority.

Also, worth noting the fraud failures: precision 1.000, recall 0.940. 
The model never wrongly classified the other classes as fraud. 
It only ever misses fraud. That implies under-confidence in fraud classification.

### Is that a data problem or a model problem?

If `fraud-report` overlaps the other classes in feature space, or is
starved of examples, the fix is: more data, better features, resampling. If the
signal is present and only the decision rule is wrong, the fix is downstream: a threshold.

The test is whether the fraud score alone separates the classes:

| | lowest score on a true fraud | highest score on anything else | gap |
| --- | ---: | ---: | ---: |
| seed 0 | 0.227 | 0.147 | +0.080 |

Across 10 fold seeds the gap is **positive on 10 of 10**. The two groups never overlap on
this one score.

Replacing `argmax` with a cut on that score, mean errors per seed over the 10:

| rule | errors |
| --- | ---: |
| argmax | 3.7 |
| best cut, chosen on the same predictions | 0.5 |
| cut chosen inside the training folds only | **1.1** |

Only the last row is honest. The middle one picks the cut on the data it is then scored
on, so it is a lower bound and not an estimate. The nested number is the one to quote: a
70% error reduction, and it beats `argmax` on 9 of the 10 seeds.

So the information is already in the model. `argmax` is the wrong decision rule for a
class where the costs are asymmetric, and resampling or reweighting can only imitate what
a threshold does directly. This is a prior and cost problem, not a scarcity and overlap
problem.

The cut here was still fitted to minimise errors, which is not the right objective either,
given errors on all classes are not same - fraud is more important.

That contradicts the reading of step 1. `fraud-report` did have the thinnest
vocabulary coverage, and it is the class that fails, but the failure is not caused by
the coverage gap.

### Two reasons not to trust 0.9893

**It is data-limited, not converged.** The learning curve is still climbing steeply at the
largest training size available: 0.934 at 224 examples, 0.975 at 288, 0.989 at 320. 
This model is not saturated.

**A single split cannot resolve anything at this size.** The same model on 40 different
stratified 80/20 splits reports macro-F1 anywhere from **0.923 to 1.000**, sd 0.019. This
7.7-point range is wider than any effect any of the candidate methods will produce. 

### Consequences for the rest of the work

- The imbalance-handling toolbox aimed at scarcity and overlap is the wrong drawer. It
  cannot beat a threshold at fixing a threshold problem.
- Since the fix is a decision rule reading probability values, those values have to mean
  something. Mean confidence is 0.684 against accuracy 0.993, a gap of +0.31, so they
  currently do not.
- macro-F1 will not detect any of that, because it sees only the argmax.
