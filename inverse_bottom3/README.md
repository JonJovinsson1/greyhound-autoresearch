# Inverse Bottom-3 Research

This is a clean experiment lane for predicting the **bottom 3 finishers in a
race, in any order**.

It is intentionally separate from the main winner model in
[`train.py`](/Users/jonjovinsson/Documents/GitHub/greyhounds-research-data/autoresearch/train.py),
so we can test feature families one at a time and then combine them without
turning the main pipeline into a junk drawer.

## Goal

For each race, score every runner by its chance of finishing in the **bottom 3**.
Then pick the 3 runners with the highest bottom-3 scores.

We report:

- `slot_accuracy`: average overlap between predicted and actual bottom 3, divided by 3
- `exact_accuracy`: fraction of races where the predicted bottom-3 set matches exactly
- `runner_auc`: standard binary AUC on the one-row-per-runner bottom-3 label

## Why this folder exists

- isolate experiments from the main winner stack
- test one feature family at a time
- test combinations cleanly
- make it easy to swap groups in and out

## Feature groups

Current groups are defined in
[`features.py`](/Users/jonjovinsson/Documents/GitHub/greyhounds-research-data/autoresearch/inverse_bottom3/features.py):

- `race_context`
- `availability`
- `recent_form`
- `specialization`
- `career`
- `speed`
- `relative`

## Run one experiment

```bash
cd /Users/jonjovinsson/Documents/GitHub/greyhounds-research-data/autoresearch
./.venv/bin/python inverse_bottom3/run_experiment.py \
  --groups race_context,recent_form,career \
  --tag baseline
```

## Search feature groups

Singles:

```bash
./.venv/bin/python inverse_bottom3/search_groups.py \
  --mode singles \
  --results-path inverse_bottom3/results/singles.tsv
```

Greedy forward search:

```bash
./.venv/bin/python inverse_bottom3/search_groups.py \
  --mode greedy \
  --results-path inverse_bottom3/results/greedy.tsv
```

Pairs:

```bash
./.venv/bin/python inverse_bottom3/search_groups.py \
  --mode pairs \
  --results-path inverse_bottom3/results/pairs.tsv
```
