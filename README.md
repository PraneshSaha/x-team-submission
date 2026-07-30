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
| oversample minority | 0.9967 | 0.9958 | 0.9920 | 1.0000 | **0.3101** |
| undersample majority | 0.9813 | 0.9858 | 0.9960 | 0.9556 | 0.6313 |

Paired against plain, counting direction rather than size:

| method | macro-F1 wins/ties/losses | sign test | fraud recall | sign test |
| --- | ---: | ---: | ---: | ---: |
| class weights | 10 / 0 / 0 | p = 0.002 | 10 / 0 / 0 | p = 0.002 |
| oversample minority | 10 / 0 / 0 | p = 0.002 | 10 / 0 / 0 | p = 0.002 |
| undersample majority | 3 / 0 / 7 | p = 0.34 | 10 / 0 / 0 | p = 0.002 |


**Class weights and oversampling are the same intervention.** 0.9992 against 0.9990,
identical fraud recall, and both move the same three errors. Both change
how often each class is drawn without changing what a fraud ticket looks like, so both
shift the decision boundary by the same offset and leave the ranking alone.

**Undersampling buys the recall and loses everywhere else.** Same recall gain, but
precision falls 1.000 to 0.956 and log loss nearly doubles, and on macro-F1 it is 3 wins
to 7 losses, p = 0.34, which is no evidence either way. It throws away 110 of 160
`general` tickets to fix a problem that was never about the ratio.

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

---

## Step 5: decide by expected cost, and calibrate so the cost rule works

```bash
uv run python analysis/05_cost_and_calibration.py
```

Writes `results/05_regularisation.csv`, `05_calibration.csv`, `05_cost_sweep.csv`,
`05_separability_stress.csv`.

Everything up to here used `argmax`, which assumes all four mistakes cost the same.
But they don't. So, we introduce cost of failing: give the model a matrix of what each mistake
costs, and pick the queue with the lowest expected cost.

    C[true, predicted]      zero on the diagonal
    C[fraud-report, *] = M  every other mistake costs 1
    route to argmin over k of  sum_j  P(j|x) * C[j, k]

`argmax` is the special case M = 1. The numbers come from the business, never from class
frequencies, which is the whole difference from step 4. Two lines of code.

### It broke immediately

I ran it at M = 26 expecting a modest recall gain. Instead:

| | errors | fraud recall | fraud precision | flagged as fraud |
| --- | ---: | ---: | ---: | ---: |
| raw probabilities | **350 of 400** | 1.000 | 0.125 | **100%** |

Every ticket in the dataset routed to the fraud queue. Precision 0.125 is exactly the base
rate, which is what flagging everything gets you.

The rule is not wrong. The inputs are. Step 2 recorded mean confidence 0.684 against
accuracy 0.993; this is what that number was worth. L2
regularisation shrinks logits toward uniform, so `p(fraud | x)` sits above 1/27 for almost
every ticket even when the model is sure it is something else. A threshold derived from
real costs was applied to numbers that do not mean what they claim.

Every metric so far reads only the ordering of the probabilities. The moment anything reads their
*values*, all of that evidence goes silent.

### Proving accuracy cannot see it

If that claim is right we can improve just by sweeping the regularisation strength:

| C | accuracy | mean confidence | gap | errors at M=26 |
| ---: | ---: | ---: | ---: | ---: |
| 0.1 | 0.630 | 0.413 | +0.217 | 350 |
| 1 | 0.993 | 0.685 | +0.307 | 350 |
| 10 | 1.000 | 0.923 | +0.078 | 31 |
| 100 | 1.000 | 0.985 | +0.015 | 1 |
| 1000 | 1.000 | 0.997 | +0.003 | 0 |

Accuracy saturates at C = 10 and never moves again. Everything after that column is
invisible to accuracy and macro-F1, and it is the difference between 31 errors and 0.

This is the answer to "how would you know if it were hurting you". From macro-F1, you
would not.

Now the overfitting. The table says C = 1000 fixes everything. It fixes it by fitting 400
templated tickets almost exactly, and step 2's learning curve showed the model still
climbing steeply at 320 examples, which is exactly where a hidden holdout will fail.
Buying calibration by deleting regularisation is a bad trade in small data. We fix
the probabilities instead, and leave the model alone.

### Calibration

5 fold seeds, cost rule at M = 26 throughout, C left at 1:

| | accuracy | mean conf | ECE | log loss | Brier | errors at M=26 | flagged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| raw | 0.9885 | 0.690 | 0.300 | 0.392 | 0.151 | 350 | 100.0% |
| Platt | 0.9990 | 0.916 | 0.084 | 0.092 | 0.017 | 65 | 28.8% |
| isotonic | 0.9990 | 0.991 | **0.008** | **0.012** | **0.005** | **6** | **14.0%** |

True fraud rate is 12.5%. Isotonic flags 14.0%.

If we look at the accuracy column: 0.9885 to 0.9990, one ticket in a hundred, while the cost-rule
errors move by a factor of 58. **Log loss separates these three by a factor of 34 and
macro-F1 separates them by nothing.** Log loss and Brier are proper scoring rules, minimised
only by honest probabilities; accuracy, F1 and AUC are not. That is why none of them
noticed, and it is why a proper scoring rule is in the report.

### The isotonic result looks too good - and isotonic is notorious

Isotonic fits a free-form monotone step function. It overfits below roughly a thousand calibration points, 
and each fold gives it about 80. It should be the fragile choice, and it beat Platt tenfold on ECE. 
That is the pattern that usually means a measurement error.

Two checks.

**The log loss is partly an artifact.** Isotonic is piecewise constant, so a pure block
gets value exactly 0 or 1, and log loss is then dominated by wherever we clip.
Everything above clips at 1e-6 and renormalises, and Brier is reported alongside 
because it is bounded and does not care.

**The advantage is conditional on separability** If
isotonic wins only because the classes are cleanly separated, then a step function is the
easy shape to fit and the overfit premise never applied. Shrinking the data and
injecting label noise should break it:

| condition | accuracy | Platt ECE | isotonic ECE | Platt errors | isotonic errors |
| --- | ---: | ---: | ---: | ---: | ---: |
| none | 0.999 | 0.084 | **0.008** | 65 | **6** |
| 50% of the data | 0.966 | 0.160 | **0.026** | 106 | **27** |
| 30% of the data | 0.920 | 0.220 | **0.050** | 99 | **35** |
| 15% label noise | 0.879 | 0.113 | **0.054** | 254 | **143** |

It does not break. Isotonic's margin *widens* as conditions worsen. Platt has exactly one sigmoid to fit with, 
and once the distortion stops being sigmoid-shaped that rigidity is the liability, not isotonic's
flexibility.

**What this still does not test**: all four conditions keep
calibration and scoring on the same distribution. A holdout whose scores fall outside the
range isotonic saw would pin to a constant where Platt keeps moving. That argues for
watching Brier in production, as otherwise I am guessing.

### What M should be

M is a business input, not something to fit.. So here is the
operating curve instead, isotonic-calibrated:

| M | errors | fraud recall | fraud precision | flagged |
| ---: | ---: | ---: | ---: | ---: |
| 1 (`argmax`) | 0.4 | 0.992 | 1.000 | 12.4% |
| 2 | 0.6 | 0.992 | 0.996 | 12.5% |
| 5 | 2.0 | 0.996 | 0.965 | 12.9% |
| 10 | 3.8 | **1.000** | 0.930 | 13.5% |
| 26 | 6.0 | 1.000 | 0.893 | 14.0% |
| 100 | 10.4 | 1.000 | 0.829 | 15.1% |

Catching every fraud report costs about 3.4 extra misroutes in 400, and 1% of tickets
arriving in the fraud queue that do not belong there. Past M = 10 we pay more for nothing,
because recall is already 1.000.

**Shipping: isotonic calibration, cost rule, M = 10 as the default**, with M exposed as a
parameter. M = 10 is the smallest cost at which no fraud report is missed. It is a
placeholder for a real number.

### Where this leaves the imbalance question

Step 4 measured `class_weight='balanced'` winning 10 of 10 seeds.
The cost rule does the same job in the same direction, with the size of the
shift set by what a missed fraud costs rather than by 160/50, and it moves sensibly when
that number is revised. Adding class weights on top would stack an uncontrolled shift on a
controlled one.

---

## Step 6: ship

```bash
uv run ticket-router train --model model.joblib
uv run ticket-router score --input messages.csv --output predictions.csv --model model.joblib
uv run pytest
```

`score` also works without `--model`, in which case it fits on `train.csv` first, so a
reviewer can run one command.

```python
from ticket_router.router import TicketRouter

router = TicketRouter().fit(texts, labels)      # or TicketRouter.load("model.joblib")
router.predict("A transfer left my wallet that I did not authorise.")
# 'fraud-report'
router.predict_proba("...")                     # the values the routing decision reads
```

The shipped model is the accumulation of the five decisions above: TF-IDF 1-2 grams with
money normalisation, logistic regression at C = 1, isotonic calibration, and routing by
expected cost with `missed_target_cost = 10`.

Trained on all 400 rows, and step 2's learning curve showed the model is still gaining from every example it gets. 
So, more data will help

### Validation, and why these three

`load_tickets` rejects missing columns, nulls and blank text. `predict` rejects non-strings,
blank text and anything over 10,000 characters. `fit` rejects a missing target class or a
class too small to calibrate.

Sanity check on four messages written by hand.

| message | routed to |
| --- | --- |
| "suspicious USDC transfer to a wallet I do not recognise, $10,000 gone" | `fraud-report` |
| "I am locked out and my SMS code never arrives" | `account-access` |
| "How do I change my email address?" | `general` |
| "I was charged twice for the same order, please refund" | `transaction-dispute` |

### Cost of running it

| | |
| --- | ---: |
| fit on 400 tickets | 0.13 s |
| model on disk | 395 KB |
| single prediction | 4.4 ms median, 11.7 ms p95 |
| batch throughput | ~10,500 tickets/sec, single core |
| needed for 10,000 requests/minute | 167 tickets/sec |

At 10k requests a minute we need 167 predictions a second and one core does sixty times
that.

### Trade-offs
1. Skipped fastapi, docker mostly because they are trivial extensions to the problem and choose to spend more time on data.
2. We could have tested LLMs/language-models. One easy one can be introducing sentence-embeddings instead of tf-ids. But would need more time.
3. Logging
4. The tickets are templated: zero exact duplicates, but mean length 14 words and heavily repeated phrasing. 
Vocabulary saturates far faster than a real support queue would, so 0.989 is optimistic and every coverage number in the README is a lower bound on how open the true vocabulary is.