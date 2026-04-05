# greyhound-autoresearch

Autonomous research loop for greyhound race winner prediction, inspired by
[karpathy/autoresearch](https://github.com/karpathy/autoresearch). An AI agent
edits a single trainer file, trains a model on historic races, evaluates on
held-out races, keeps winners, discards losers, and iterates.

## Objective

Maximize **top-1 win-rate accuracy on the held-out `test/` split** — the
fraction of races where the runner with the highest predicted score actually
won. No odds, no ROI, no betting logic. Pure picking-the-winner.

## Data

```
train/   2025-10 … 2026-01   ~15.5k historic races    — fit the model
val/     2026-02             ~3.5k historic races     — select hyperparams
test/    2026-03-17 … 04-04  ~19 days of snapshots    — ground-truth metric
```

`train/` and `val/` use one JSON layout (`race_info.results`, `meta.*`),
`test/` uses another (`temp_result.results`, flat top-level keys). Both are
normalized by `prepare.py`.

## Files

| File | Modified by | Purpose |
|---|---|---|
| `prepare.py` | never | data loading, normalization, fixed `evaluate()` metric |
| `train.py`   | agent | feature extraction, model, training loop |
| `program.md` | human | agent skill / loop protocol |
| `pyproject.toml` | agent (rarely) | dependencies |

## Metric (defined in `prepare.py`, not negotiable)

```
top1_accuracy = fraction of races where argmax(scores) == winner
log_loss      = -log p(winner) under field-softmax (secondary diagnostic)
```

Primary: **`test_top1_accuracy`** — higher is better.
Selection during the loop uses `val_top1_accuracy`; `test_top1_accuracy` is
reported every run but not used for decisions.

## Quick start

```bash
# 1. install deps (CPU-only, M2-friendly)
uv sync                      # or: pip install -e .

# 2. sanity-check data is visible
python prepare.py
#   train: 15xxx usable races from .../train
#   val:    3xxx usable races from .../val
#   test:    xxx usable races from .../test

# 3. train the baseline
python train.py
```

Expected output ends with a grep-friendly summary:

```
---
val_top1_accuracy:   0.3xxxxx
val_log_loss:        1.xxxxxx
val_races:           3xxx
test_top1_accuracy:  0.3xxxxx
test_log_loss:       1.xxxxxx
test_races:          xxx
train_seconds:       xx.x
total_seconds:       xx.x
time_budget:         300
num_train_rows:      1xxxxx
num_features:        23
```

## Running the agent

Start Claude Code (or equivalent) in this directory and prompt:

```
Read program.md and let's start a new autoresearch run.
```

The agent will create a branch, commit experiments one-by-one, keep the ones
that improve val_top1, and track everything in `results.tsv`.

## Philosophy

- **EDA before complexity.** Box number has statistical signal on its own.
  Understand the data before reaching for bigger models.
- **Start simple, get funky.** The baseline is a CatBoost classifier on ~23
  hand features. The loop can grow into rankers, Elo, per-dog sequence models,
  time-series foundation models (Chronos / TimesFM) as feature extractors, or
  hand-written re-rank filters — anything that beats val_top1 stays.
- **No odds ever.** `prepare.py` strips `sp` from runners and form history so
  it cannot accidentally become a feature.
- **Fixed wall-clock budget.** 5 min training cap per experiment. Don't
  rescale the budget to "win" by training longer.

## License

MIT (inherits from upstream karpathy/autoresearch).
