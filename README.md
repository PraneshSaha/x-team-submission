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

---

## Step 3: normalising amounts and asset names

```bash
uv run python analysis/03_normalization.py
```

Writes `results/03_coverage.csv`, `03_clean_scores.csv`, `03_robustness.csv`.

Step-1 showed `fraud-report` has the thinnest vocabulary coverage. Its once-only tokens are
`15 3am 880 activity browser cardano computer desktop litecoin missing usdt wasn`, and
`transaction-dispute`'s are `120 500 75 880`. Amounts, times and asset names. All three
step-2 errors mention a ticker and an amount.

As the holdout is hidden removing these might help. Both
of these are open-ended surface forms: a new ticket can always carry an amount or an asset
we have never seen, and neither decides the route. So there are two candidate
interventions, and they are not equally cheap.

| | needs a word list | goes stale |
| --- | --- | --- |
| `$10,000` -> `moneyamount`, `3am` -> `clocktime` | no, a regex | no |
| `SOL`, `litecoin` -> `cryptoasset` | yes, ~90 names and tickers | yes |

### Measurements

**Coverage.** Both interventions do what they claim. Missing mass:

| class | raw | money | money + assets |
| --- | ---: | ---: | ---: |
| fraud-report | 0.0197 | 0.0171 | **0.0131** |
| transaction-dispute | 0.0027 | **0.0000** | 0.0000 |
| account-access | 0.0025 | 0.0025 | 0.0025 |
| general | 0.0005 | 0.0005 | 0.0005 |

Money alone empties `transaction-dispute`'s open vocabulary: all four of its once-only
tokens were amounts.

**Accuracy on the data as it stands.** 10 fold seeds, paired against raw:

| variant | macro-F1 | fraud recall | wins | ties | losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0.9874 | 0.9360 | 0 | 10 | 0 |
| money | 0.9871 | 0.9340 | 0 | 9 | 1 |
| money + assets | 0.9875 | 0.9320 | 2 | 6 | 2 |
| money + tickers | 0.9873 | 0.9340 | 1 | 8 | 1 |

Nothing moves. Nine or ten ties out of ten.

**Under perturbation, which is the surprise.** Models train on clean text; held-out
tickets are rewritten to name assets the training set never saw. The swap changes
**182 of 400** tickets.

| held-out condition | raw | money | money + assets |
| --- | ---: | ---: | ---: |
| clean | 0.9874 | 0.9871 | 0.9875 |
| unseen amounts | 0.9876 | 0.9871 | 0.9875 |
| unseen asset, in our vocabulary | 0.9882 | 0.9884 | 0.9875 |
| unseen asset, unknown to us | 0.9878 | 0.9880 | 0.9888 |

Rewriting 182 tickets to talk about assets the model has never heard of does not move
macro-F1 at all. The hypothesis this step was built to test is false.

**Why.** The top-weighted fraud features are `someone, account and, my account, was,
account, fraud, think, never, unauthorized, recognize, money`. Not one asset name, not one
amount. 

### Decision

**Money normalisation ships. The asset list does not.**

**What this does not rule out.** The perturbation moves the axis the model turns out not
to use. A holdout that differs in *complaint phrasing* rather than in nouns would hurt,
and neither intervention helps with that. That risk is real and remains untested.

---

## Step 4: choosing a metric, then choosing what to do about imbalance

```bash
uv run python analysis/04_metric_and_imbalance.py
```

Writes `results/04_degenerate.csv`, `04_methods.csv`, `04_paired.csv`.

These are one step because the metric decides the imbalance question. 

### Which metric

The sharpest test of a metric is to score a classifier that ignores the input. Whatever it
gets is that metric's floor, and a high floor flatters the model.

| strategy | accuracy | micro-F1 | macro-F1 | weighted-F1 | balanced acc |
| --- | ---: | ---: | ---: | ---: | ---: |
| always `general` | 0.400 | 0.400 | 0.143 | 0.229 | 0.250 |
| always `fraud-report` | 0.125 | 0.125 | 0.056 | 0.028 | 0.250 |
| uniform random | 0.270 | 0.270 | 0.256 | 0.280 | 0.263 |
| random at class rates | 0.235 | 0.235 | 0.201 | 0.237 | 0.202 |

Three things fall out.

**Accuracy and micro-F1 are the same number**, both no-go (and same in case of multi-class classification).

**Balanced accuracy gives every constant classifier exactly 0.250**, which is 1/K. It
averages per-class recall and a constant classifier gets recall 1 on one class and 0 on
the rest. That fixed floor makes it readable, which macro-F1 has no equivalent of.

**Macro-F1 prefers uniform random (0.256) to always-`general` (0.143).**  it refuses to reward a model for ignoring small
classes, and `general` is 40% of the data.

Weighted-F1 is the bad one. It scores always-`general` at 0.229 against macro-F1's
0.143, because it weights by support.

**Choice: macro-F1 as the headline, balanced accuracy beside it, log loss alongside both.**

And the caveat that matters. **Macro-F1 and balanced accuracy both
weight the four classes equally.** The brief says `fraud-report` is the highest-stakes
route to get wrong. An equal-weight metric is the right instrument for asking "is the model ignoring a small class",
and the wrong one for asking "is the model making the expensive mistake".

Log loss is there because macro-F1 and balanced accuracy see only the argmax. 

### The imbalance choice

Four options, on identical folds, 10 seeds:

| method | macro-F1 | balanced acc | fraud recall | fraud precision | log loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| plain | 0.9871 | 0.9822 | 0.9340 | 1.0000 | 0.3938 |
| class weights | **0.9992** | 0.9990 | 0.9960 | 1.0000 | 0.4018 |
| oversample minority | 0.9990 | 0.9988 | 0.9960 | 1.0000 | **0.3069** |
| undersample majority | 0.9716 | 0.9778 | 0.9960 | 0.9170 | 0.6305 |

Paired against plain, counting direction rather than size:

| method | macro-F1 wins/ties/losses | sign test | fraud recall | sign test |
| --- | ---: | ---: | ---: | ---: |
| class weights | 10 / 0 / 0 | p = 0.002 | 10 / 0 / 0 | p = 0.002 |
| oversample minority | 10 / 0 / 0 | p = 0.002 | 10 / 0 / 0 | p = 0.002 |
| undersample majority | 1 / 0 / 9 | p = 0.022 | 10 / 0 / 0 | p = 0.002 |


**Class weights and oversampling are the same intervention.** 0.9992 against 0.9990,
identical fraud recall, and both move the same three errors. Both change
how often each class is drawn without changing what a fraud ticket looks like, so both
shift the decision boundary by the same offset and leave the ranking alone.

**Undersampling is strictly worse.** Same recall gain, but precision falls 1.000 to 0.917
and log loss nearly doubles. It throws away 110 of 160 `general` tickets to fix a problem
that was never about the ratio.

**Log loss disagrees with macro-F1** Class weights
win 10/10 on macro-F1 while losing 10/10 on log loss. It most improves
the argmax but makes the probabilities worse, because it biases them. Anything
downstream reading a probability rather than a label is paying for that improvement.

### Decision

**Ship class weights? No, not yet.** They win, and they win for the wrong reason.

Step 2 established this is a decision problem: the fraud score already separates the
classes on 10 of 10 seeds, and only `argmax` is wrong. `class_weight='balanced'` fixes that
by shifting the boundary, and it picks the size of the shift from the class ratio,
`160/50`. Nothing about 3.2 encodes what missing a fraud report costs. It works here only
because the rarest class happens to be the expensive one, which is a coincidence of this
dataset and not a property we should build on.
