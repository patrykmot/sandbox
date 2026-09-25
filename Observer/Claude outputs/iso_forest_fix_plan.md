# `_calculate_n_estimators` — what changed

Final record. Supersedes the earlier draft of this document, which proposed a *fraction* scale;
you chose **percent**, and that is what shipped.

## The bugs

**1. Units.** `__init__` held `n_tree_training_coverage = 0.99`; the method's `coverage_percent`
expects 0–100 and divides by 100. The forest was asked for **0.99%** coverage:

```
c = 0.99 / 100 = 0.0099
E = ln(1 - 0.0099) / ln(1 - 256/3000) = 0.11  →  ceil → 1
```

**2. The `samples_per_tree == dataset_size` branch returned 1** before the formula ran, catching
every dataset of 256 rows or fewer — including both failing tests.

Measured on the failing tests' data (N=64, 20 seeds): **1 tree caught the outlier 30% of the
time; 100 trees caught it every time.** False positives stayed ~2/30 either way, so extra trees
cost only CPU.

## Change 1 — `MIN_TREES`, new module-level constant

```python
MIN_TREES = 100
```

Module level rather than a class attribute, because `_calculate_n_estimators` is a
`@staticmethod` and couldn't reach a class attribute unqualified. Matches `_MIN_STD` in
`autoencoder_detector.py`.

## Change 2 — `_calculate_n_estimators`

| | before | after |
|---|---|---|
| `max_samples` default | `1024` | `256` — what `fit()` actually passes |
| validation | `0 < coverage_percent < 100` | same range, message explains why 100 is out |
| `S == N` branch | `return 1` | `return MIN_TREES` |
| return | `math.ceil(n_estimators)` | `max(MIN_TREES, math.ceil(n_estimators))` |
| sub-1% values | silent | `logger.warning` |

The percent scale and the `c = coverage_percent / 100.0` conversion stay exactly as you wrote
them. The formula is unchanged.

**100 is rejected, not clamped:**

```python
raise ValueError(
    "coverage_percent is a percentage strictly between 0 and 100. "
    "100 is excluded because full coverage would take infinitely "
    f"many trees - ask for 99.99 instead. Got {coverage_percent!r}."
)
```

Random subsampling never guarantees every row is seen, so 100% is a limit, not a value —
`ln(1 - 1.0)` is `ln(0)`. It diverges slowly, so 99.99 gets you almost all of it (715 trees at
N=20,000, against 358 for 99%).

**The new warning**, because the floor now hides the old mistake rather than collapsing to one
tree:

```python
if coverage_percent < 1:
    logger.warning(
        "coverage_percent=%r asks for less than 1%% coverage. If a fraction "
        "was meant, pass %g instead.", coverage_percent, coverage_percent * 100,
    )
```

## Change 3 — the constructor

```python
# before
n_tree_training_coverage: float = 0.99, # Percentage! Useful range for tree is from 0.1 to 0.99 %

# after
# Percent, strictly between 0 and 100 - 99 means 99%, not 0.99.
# The unit is in the name on purpose: reading it as a fraction is what
# once made this forest a single tree.
n_tree_training_coverage_percent: float = 99.0,
```

This one is **not optional** on the percent scale. Left at `0.99` it would still request 0.99%
coverage — masked by the floor into 100 trees instead of 1, but still wrong above ~5,700
samples. Nothing outside the class referenced the old name.

## Change 4 — the log line in `fit()`

```python
# before
logger.info(f"Calculated trees_number = {trees_number}")

# after
logger.info(
    "n_estimators=%d for %d samples at %g%% coverage (max_samples=%d, floor=%d).",
    trees_number, len(x_train), self._n_tree_training_coverage_percent, max_samples, MIN_TREES,
)
```

A bare `= 1` is what made this hard to spot; now the number explains itself.

## Change 5 — `tests/test_iso_forest_detector.py` (new, 9 tests)

The floor (`S == N`, and `N = S + 1` where the raw formula returns 1 — asserted explicitly, so
the test fails loudly if that premise ever changes), a large N proving the floor isn't a
constant, monotonicity, `100` rejected, out-of-range rejected, zero-size rejected, the sub-1%
warning, and one wiring test fitting a real model and asserting `n_estimators == 100`.

## Results

| N (samples) | returned |
|---:|---:|
| 64 — the failing tests | **100** |
| 257 | 100 |
| 3,000 — your `collection_target_value` | 100 |
| 5,000 | 100 |
| 5,700 | 101 ← coverage takes over here |
| 20,000 | 358 |

Detection on the failing tests' data, 20 seeds:

| | outlier caught | false positives / 30 |
|---|---:|---:|
| before (1 tree) | 30% | 1.8 |
| after (100 trees) | **100%** | 2.0 |

`python -m pytest -q` → **94 passed, 1 skipped**. The skip is the torch module; this machine's
test VM can't install torch, unchanged from before.

## Still open

`isolation_forest_contamination` in `src/config.py` is `0.0000001`, down from `0.05`. That tells
the forest to expect essentially no anomalies in training, pushing the decision threshold far
out — possibly far enough to suppress alarms in a live run. The tests pass `0.05` explicitly, so
they never see it. Deliberate, or left over from debugging the tree count?
