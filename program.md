# greyhound-autoresearch — agent program

You are running an autonomous research loop on greyhound race prediction. This
file is your persistent skill. Read it first every run.

## Objective (read this twice)

Maximize **top-1 win-rate accuracy on `test/`** — fraction of races in
`2026-03-17 … 2026-04-04` where the runner with your highest predicted score
actually won. That is THE number.

- `train/` (2025-10 → 2026-01) fits the model.
- `val/` (2026-02) selects hyperparameters / decides keep-vs-discard in the loop.
- `test/` is the ultimate judge. Evaluate on it every run, report it, but don't
  tune on it.

**No odds. No SP. No ROI.** `prepare.py` already strips odds off runners. Do
not add them back. Do not use `time` / `margin` / `winner` from the current
race (only from prior runs in `form_history`). Any post-race leakage invalidates
the run.

## The two files

| File | Who edits | What's in it |
|---|---|---|
| `prepare.py` | nobody (frozen) | data loaders, `evaluate(predict_fn, split)`, the fixed metric |
| `train.py` | **you** | features, model, training loop |

You may create scratch analysis files (e.g. `eda.py`, `scratch_*.py`), just
don't commit them to the branch.

## Loop

```
SETUP (once):
  1. Agree on a run tag with the human (e.g. `apr5`).
  2. git checkout -b autoresearch/<tag>
  3. Read AGENT.md + this file + prepare.py + train.py.
  4. Sanity check: `python prepare.py` — should list usable race counts for
     train/val/test. If any split is 0 or missing, stop and tell the human.
  5. Initialize `results.tsv` with the header:
       commit\tval_top1\ttest_top1\tval_ll\ttest_ll\tsecs\tstatus\tnotes
  6. Run the baseline as-is to lock in a reference row.

LOOP FOREVER (after setup, do NOT stop to ask "should I keep going?"):
  1. Form a single concrete hypothesis. Write it down (one sentence) in notes.
  2. Edit train.py. git commit with a short message.
  3. Run: `python train.py > run.log 2>&1`
  4. Read results: `grep "^val_top1_accuracy:\|^test_top1_accuracy:\|^val_log_loss:\|^test_log_loss:\|^total_seconds:" run.log`
  5. Append a row to results.tsv. Do NOT commit results.tsv (keep it untracked).
  6. Keep/discard rule (primary = val_top1_accuracy):
       - val_top1 strictly improves → keep commit, advance branch.
       - val_top1 ties best → use val_log_loss as tiebreaker (lower wins).
       - val_top1 worse → `git reset --hard HEAD~1`.
  7. Push to remote so the human can see progress: `git push origin HEAD`.
```

## Hard rules

- Time budget is 5 minutes wall-clock training (`TIME_BUDGET` in prepare.py).
  If a run exceeds 10 min total, kill it and mark as `crash`. Do not widen the
  budget to "win."
- Never read from `test/` for hyperparameter selection. Only report its number.
- Don't modify `prepare.py`. If you think the metric is wrong, stop and say so
  to the human — do not silently change it.
- Don't install new packages without a good reason. If you do add one, update
  `pyproject.toml` and mention it in notes.
- Don't touch `research/` — it's read-only prior art. You may read it for
  ideas (there's years of CatBoost work in `research/catboostResearch/`).

## EDA first, complexity second (this is important)

The human explicitly asked for **data analysis over model complexity**. Before
you stack more layers or swap models, answer questions like:

- What's the base rate per box? Does box 1 actually win more than box 8?
  What's the statistical significance? (Even box is signal.)
- How does win rate vary by race_num? R1–R3 vs R10+ — is there a distribution
  shift the model is missing?
- How does win rate depend on field_size, distance, track?
- Are there tracks where the baseline is strong and tracks where it collapses?
  Look at per-track top-1 accuracy on val.
- What fraction of winners come from dogs with 0 form_history entries?
- Where does the model lose? Sample 50 races where the top-pick lost and eyeball them.

When you find a real signal, it should show up as a cheap, interpretable
feature that moves val_top1 by more than noise. A +0.2pp feature that adds
3 lines is a keep. A +0.2pp trick that adds 100 lines is a discard.

## Directions worth trying (start simple, get funky)

**Simple (do these first):**
- Within-race feature normalization (rank / z-score each feature across the
  field, so the model sees relative-to-field, not absolute).
- Softmax over the field at predict time with a learnable temperature, fit on val.
- LightGBM / XGBoost ranker (`rank:pairwise`, groups = races) instead of a
  binary classifier — this directly optimizes the per-race ordering, which is
  what we're measured on.
- Per-track or per-distance-bucket calibration (Platt / isotonic).
- Dog-level features aggregated from ALL their historic form in train (career
  win rate vs field, time z-score vs track/distance).

**Medium:**
- Elo / Glicko ratings updated game-by-game on train, frozen at val/test time.
- Gradient-boosted ranker with monotone constraints (more recent-wins ↑,
  worse avg_position ↓).
- Ensemble CatBoost + LightGBM + Elo.
- Target-encode categorical columns (track, grade, trainer, sire/dam) with
  leave-one-fold-out to avoid leakage.
- **Trainer metrics** — trainer win rate overall / at this track / at this
  distance / over last 30 days, computed only from races strictly before
  race_dt. Also pairwise: "this trainer's win rate vs the other trainers in
  the field" (trainer-head-to-head). Same idea for sire / dam / kennel.

**Funky (go here if the simple stuff plateaus):**
- Per-dog sequence model over form_history: small transformer / GRU / Mamba-SSM
  consuming (position, distance, time, first_split, margin, weight, days_ago)
  per past run → dog embedding → race-level softmax. PyTorch-CPU on an M2 16GB
  is fine for a tiny model.
- Pretrained time-series foundation models (Chronos, TimesFM, Moirai) as
  feature extractors over each dog's (time, first_split) sequence. Freeze the
  backbone, just use the embedding. Treat this as a feature factory feeding
  into the boosted model — it adds complexity, so it needs to pay for itself
  in val_top1.
- Learned race-level set-transformer over runner features (permutation-invariant
  attention across the field, softmax over runners).
- Contrastive dog embeddings: train an encoder s.t. the winner's embedding
  scores higher than each loser's in the same race (pairwise hinge / InfoNCE).

**Ambition ≠ progress.** Every addition must beat the baseline on val_top1
or get reverted. Keep the winner, keep it simple, keep moving.

## Re-rank / tiebreaker filters are fair game

If the model's top-1 pick is often losing to a runner within ~0.3 probability
of it, you're allowed to post-process the pick with a hand rule, as long as
the rule only uses **pre-race** info (sex, age, weight, trainer, dog_id,
career line, form_history, box, distance, grade, track, race_num, field_size —
anything that would be knowable before the race is run).

Example shapes:
- "If the top-1 and top-2 are within 0.05 and top-2 is sex=D in a distance
  ≥ 500m, swap them."
- "If top-1 has 0 prior runs at this distance but top-2 has ≥3, swap them."
- "If field_size ≥ 9 and top-1 is in box 8, demote to top-2."

These rules compete against the ML model on a level playing field: they're
kept only if they beat the current val_top1. They must not touch odds,
results, time, margin, winner, or any field of the race being predicted.
Grid-search small rule spaces on val, then check test.

## Output contract

`train.py` must end by printing (grep-able):

```
---
val_top1_accuracy:   <float>
val_log_loss:        <float>
val_races:           <int>
test_top1_accuracy:  <float>
test_log_loss:       <float>
test_races:          <int>
train_seconds:       <float>
total_seconds:       <float>
time_budget:         <int>
num_train_rows:      <int>
num_features:        <int>
```

If the run crashes, log `status=crash` in results.tsv with a one-line reason.

## Platform note

Target machine is an M2 Mac, 16GB. Prefer CPU-friendly frameworks: CatBoost,
LightGBM, XGBoost, scikit-learn, PyTorch-CPU with small models. Do not assume
CUDA or MPS without checking. Small models, fast iterations, lots of EDA.

## NEVER STOP

Once the loop has begun, don't pause to ask the human "should I continue?" or
"is this a good place to stop?". If you run out of ideas, re-read EDA notes,
look at failure cases, combine past near-misses, try a more radical direction
from the list above. The loop runs until manually interrupted.
