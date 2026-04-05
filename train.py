"""
Greyhound autoresearch trainer — single-file, single-machine (M2 / CPU-friendly).

This is the ONE file the agent edits. Everything here is fair game: features,
model family, hyperparameters, how train/val data is split, calibration, etc.

Protocol (see program.md for the full loop):
    1. Train on train/ (2025-10 … 2026-01).
    2. Score on val/ (2026-02) — this is what the loop uses to keep/discard.
    3. Score on test/ (2026-03-17 … 2026-04-04) — the headline number.
       TEST TOP-1 ACCURACY IS THE ULTIMATE GOAL. Val is just a selector.

Primary metric: top1_accuracy on test — "fraction of races where the runner
with the highest predicted score actually won." No odds, no ROI.

Baseline deliberately kept small and simple:
    - A handful of per-runner features derived from form_history + box + race_num.
    - CatBoost binary classifier (CPU, no GPU, runs fine on an M2 16GB).
    - Per-race softmax over the raw margins to turn scores into win probabilities.

The agent should start with feature analysis (see program.md "EDA first") before
reaching for bigger models. Simpler wins are more valuable than complex ones.

Usage: python train.py
"""

import math
import re
import time
from datetime import datetime

import numpy as np
from catboost import CatBoostClassifier, Pool

from prepare import TIME_BUDGET, load_races, evaluate

# ---------------------------------------------------------------------------
# Hyperparameters — edit these directly
# ---------------------------------------------------------------------------

# Model
ITERATIONS     = 700
LEARNING_RATE  = 0.04
DEPTH          = 5
L2_LEAF_REG    = 5.0
RANDOM_SEED    = 42

# Feature engineering
MAX_FORM_RUNS  = 5       # how many recent form_history entries to consume per dog
MIN_TRAIN_DATE = None    # e.g. "2025-10-01" to restrict train window; None = all

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _parse_float(x):
    if x is None or x == "" or x == "—":
        return None
    try:
        return float(str(x).replace("$", "").strip())
    except Exception:
        return None


def _parse_pos(s):
    """'3rd/7' -> (3, 7), 'fell' -> (None, None)."""
    if not s:
        return None, None
    m = re.match(r"(\d+)(?:st|nd|rd|th)?(?:/(\d+))?", str(s).strip())
    if not m:
        return None, None
    pos = int(m.group(1)) if m.group(1) else None
    field = int(m.group(2)) if m.group(2) else None
    return pos, field


def _parse_form_date(s):
    try:
        return datetime.strptime(s, "%d/%m/%Y")
    except Exception:
        return None


def runner_features(runner, race):
    """
    Minimal, interpretable features for one runner. Starts deliberately simple —
    the research loop should grow this based on what EDA reveals actually moves
    the needle, not on speculation.
    """
    race_dt = race["race_dt"]
    race_distance = race["distance"]
    race_track = race["track"]

    box = runner.get("box") or 0

    # Walk form history — only runs STRICTLY BEFORE race_dt (no leakage).
    form = runner.get("form_history") or []
    valid = []
    for h in form:
        d = _parse_form_date(h.get("date", ""))
        if d is None or (race_dt is not None and d >= race_dt):
            continue
        valid.append((h, d))
        if len(valid) >= MAX_FORM_RUNS:
            break

    n = len(valid)
    wins = places = 0
    positions, times, margins, first_splits = [], [], [], []
    td_runs = td_wins = 0
    dist_runs = dist_wins = 0

    for h, _ in valid:
        p, _ = _parse_pos(h.get("position"))
        if p is not None:
            positions.append(p)
            if p == 1:
                wins += 1
            if p <= 3:
                places += 1
        t = _parse_float(h.get("time"))
        if t:
            times.append(t)
        fs = _parse_float(h.get("first_split"))
        if fs:
            first_splits.append(fs)
        mg = _parse_float(h.get("margin"))
        if mg is not None:
            margins.append(mg)
        # track/distance specialization
        ht = (h.get("track") or "").strip().lower()
        hd = h.get("distance")
        if hd and race_distance and int(hd) == race_distance:
            dist_runs += 1
            if p == 1:
                dist_wins += 1
            if ht and race_track and ht.startswith(race_track[:4]):
                td_runs += 1
                if p == 1:
                    td_wins += 1

    # Days since last run (recency).
    if valid and race_dt is not None:
        days_since = (race_dt - valid[0][1]).days
    else:
        days_since = -1

    latest = valid[0][0] if valid else None
    latest_pos, latest_field = _parse_pos(latest.get("position")) if latest else (None, None)
    latest_margin = _parse_float(latest.get("margin")) if latest else None
    latest_time = _parse_float(latest.get("time")) if latest else None
    latest_first_split = _parse_float(latest.get("first_split")) if latest else None

    def _avg(xs): return float(np.mean(xs)) if xs else 0.0
    def _min(xs): return float(np.min(xs)) if xs else 0.0

    # Career line (e.g. "59:26-5-6" = starts:wins-2nds-3rds)
    ca = runner.get("career_all") or ""
    m = re.match(r"(\d+):(\d+)-(\d+)-(\d+)", ca)
    if m:
        c_starts, c_wins, c_2, c_3 = (int(m.group(i)) for i in range(1, 5))
    else:
        c_starts = c_wins = c_2 = c_3 = 0

    best_time = _parse_float(runner.get("best_time")) or 0.0
    best_split = _parse_float(runner.get("best_split")) or 0.0

    return {
        # Race context
        "box": box,
        "race_num": race["race_num"] or 0,
        "distance": race_distance or 0,
        "field_size": len(race["runners"]),
        # Availability
        "has_form": 1 if n else 0,
        "has_form_time": 1 if times else 0,
        "has_form_first_split": 1 if first_splits else 0,
        "has_last_pos": 1 if latest_pos is not None else 0,
        "has_last_margin": 1 if latest_margin is not None else 0,
        "has_last_time": 1 if latest_time is not None else 0,
        "has_last_first_split": 1 if latest_first_split is not None else 0,
        "has_best_time": 1 if best_time > 0 else 0,
        "has_best_split": 1 if best_split > 0 else 0,
        # Recent form
        "form_runs": n,
        "last_start_win": 1.0 if latest_pos == 1 else 0.0,
        "last_start_pos": float(latest_pos) if latest_pos is not None else 0.0,
        "last_start_pos_ratio": (latest_pos / latest_field) if latest_pos and latest_field else 0.0,
        "last_start_margin": latest_margin if latest_margin is not None else 0.0,
        "last_start_time": latest_time if latest_time is not None else 0.0,
        "last_start_first_split": latest_first_split if latest_first_split is not None else 0.0,
        "form_win_rate": (wins / n) if n else 0.0,
        "form_place_rate": (places / n) if n else 0.0,
        "form_avg_pos": _avg(positions),
        "form_avg_margin": _avg(margins),
        "form_avg_time": _avg(times),
        "form_best_time": _min(times),
        "form_avg_first_split": _avg(first_splits),
        "form_days_since": days_since,
        # Track / distance specialization (from form)
        "form_dist_runs": dist_runs,
        "form_dist_win_rate": (dist_wins / dist_runs) if dist_runs else 0.0,
        "form_td_runs": td_runs,
        "form_td_win_rate": (td_wins / td_runs) if td_runs else 0.0,
        # Career
        "career_starts": c_starts,
        "career_win_rate": (c_wins / c_starts) if c_starts else 0.0,
        "career_place_rate": ((c_wins + c_2 + c_3) / c_starts) if c_starts else 0.0,
        "best_time": best_time,
        "best_split": best_split,
    }


RELATIVE_FEATURE_SPECS = [
    ("box", "min", lambda f: False),
    ("form_runs", "max", lambda f: False),
    ("last_start_win", "max", lambda f: not f["has_last_pos"]),
    ("last_start_pos", "min", lambda f: not f["has_last_pos"]),
    ("last_start_pos_ratio", "min", lambda f: not f["has_last_pos"]),
    ("last_start_margin", "min", lambda f: not f["has_last_margin"]),
    ("last_start_time", "min", lambda f: not f["has_last_time"]),
    ("last_start_first_split", "min", lambda f: not f["has_last_first_split"]),
    ("form_win_rate", "max", lambda f: False),
    ("form_place_rate", "max", lambda f: False),
    ("form_avg_pos", "min", lambda f: not f["has_form"]),
    ("form_avg_time", "min", lambda f: not f["has_form_time"]),
    ("form_best_time", "min", lambda f: not f["has_form_time"]),
    ("form_avg_first_split", "min", lambda f: not f["has_form_first_split"]),
    ("form_dist_runs", "max", lambda f: False),
    ("form_dist_win_rate", "max", lambda f: False),
    ("form_td_runs", "max", lambda f: False),
    ("form_td_win_rate", "max", lambda f: False),
    ("career_starts", "max", lambda f: False),
    ("career_win_rate", "max", lambda f: False),
    ("career_place_rate", "max", lambda f: False),
    ("best_time", "min", lambda f: not f["has_best_time"]),
    ("best_split", "min", lambda f: not f["has_best_split"]),
]

RELATIVE_NUMERIC_FEATURES = []
for _name, _direction, _missing_fn in RELATIVE_FEATURE_SPECS:
    RELATIVE_NUMERIC_FEATURES.extend([
        f"{_name}_field_rank",
        f"{_name}_field_gap",
    ])


def add_relative_features(feature_rows):
    """Augment per-runner rows with within-race rank/gap features."""
    n = len(feature_rows)
    if not n:
        return feature_rows

    for name, direction, missing_fn in RELATIVE_FEATURE_SPECS:
        scored = []
        for idx, row in enumerate(feature_rows):
            missing = missing_fn(row)
            value = row[name]
            if missing:
                sort_value = math.inf if direction == "min" else -math.inf
            else:
                sort_value = value
            scored.append((idx, sort_value, missing))

        order = sorted(
            scored,
            key=lambda item: item[1],
            reverse=(direction == "max"),
        )
        denom = max(n - 1, 1)
        finite_values = [value for _, value, _ in scored if math.isfinite(value)]
        if finite_values:
            best = min(finite_values) if direction == "min" else max(finite_values)
            worst = max(finite_values) if direction == "min" else min(finite_values)
            span = abs(worst - best)
        else:
            best = 0.0
            span = 0.0
        span = max(span, 1e-9)

        for rank, (idx, value, missing) in enumerate(order):
            feature_rows[idx][f"{name}_field_rank"] = 1.0 - (rank / denom)
            if missing or not math.isfinite(value):
                feature_rows[idx][f"{name}_field_gap"] = 1.0
            elif direction == "min":
                feature_rows[idx][f"{name}_field_gap"] = (value - best) / span
            else:
                feature_rows[idx][f"{name}_field_gap"] = (best - value) / span

    return feature_rows


def race_feature_rows(race):
    feature_rows = []
    for runner in race["runners"]:
        features = runner_features(runner, race)
        features["track"] = race["track"] or "unknown"
        feature_rows.append(features)
    return add_relative_features(feature_rows)


NUMERIC_FEATURES = [
    "box", "race_num", "distance", "field_size",
    "has_form", "has_form_time", "has_form_first_split",
    "has_last_pos", "has_last_margin", "has_last_time", "has_last_first_split",
    "has_best_time", "has_best_split",
    "form_runs", "last_start_win", "last_start_pos", "last_start_pos_ratio",
    "last_start_margin", "last_start_time", "last_start_first_split",
    "form_win_rate", "form_place_rate", "form_avg_pos",
    "form_avg_margin", "form_avg_time", "form_best_time", "form_avg_first_split",
    "form_days_since",
    "form_dist_runs", "form_dist_win_rate", "form_td_runs", "form_td_win_rate",
    "career_starts", "career_win_rate", "career_place_rate", "best_time", "best_split",
] + RELATIVE_NUMERIC_FEATURES
CAT_FEATURES = ["track"]
ALL_FEATURES = NUMERIC_FEATURES + CAT_FEATURES


def build_split(split):
    """Materialize (X, y) + the race grouping needed to evaluate top-1 per race."""
    X, y = [], []
    for race in load_races(split):
        if MIN_TRAIN_DATE and split == "train" and race["date"] < MIN_TRAIN_DATE:
            continue
        feature_rows = race_feature_rows(race)
        for runner, features in zip(race["runners"], feature_rows):
            X.append([features[k] for k in ALL_FEATURES])
            y.append(1 if runner["box"] == race["winner_box"] else 0)
    return X, y


# ---------------------------------------------------------------------------
# Train + evaluate
# ---------------------------------------------------------------------------

def main():
    t0 = time.time()

    print("Loading train split...")
    X_train, y_train = build_split("train")
    print(f"  {len(X_train):,} rows, base rate = {sum(y_train)/len(y_train):.4f}")

    print("Training CatBoost...")
    cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
    train_pool = Pool(X_train, y_train, cat_features=cat_idx)
    model = CatBoostClassifier(
        iterations=ITERATIONS,
        learning_rate=LEARNING_RATE,
        depth=DEPTH,
        l2_leaf_reg=L2_LEAF_REG,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=RANDOM_SEED,
        verbose=100,
        task_type="CPU",
        thread_count=-1,
    )
    model.fit(train_pool)
    t_train = time.time() - t0

    # Build predict_fn for the fixed evaluator: scores are the model's P(win=1)
    # for each runner. evaluate() applies softmax over the field for log-loss and
    # takes the argmax for top-1. So using raw P(win=1) is fine.
    def predict_fn(race):
        feature_rows = race_feature_rows(race)
        rows = [[features[k] for k in ALL_FEATURES] for features in feature_rows]
        pool = Pool(rows, cat_features=cat_idx)
        probs = model.predict_proba(pool)[:, 1]
        return probs.tolist()

    print("\nScoring val...")
    val = evaluate(predict_fn, "val")
    print("Scoring test...")
    test = evaluate(predict_fn, "test")

    t_total = time.time() - t0

    # The summary the experiment loop greps on.
    print("---")
    print(f"val_top1_accuracy:   {val['top1_accuracy']:.6f}")
    print(f"val_log_loss:        {val['log_loss']:.6f}")
    print(f"val_races:           {val['n_races']}")
    print(f"test_top1_accuracy:  {test['top1_accuracy']:.6f}")
    print(f"test_log_loss:       {test['log_loss']:.6f}")
    print(f"test_races:          {test['n_races']}")
    print(f"train_seconds:       {t_train:.1f}")
    print(f"total_seconds:       {t_total:.1f}")
    print(f"time_budget:         {TIME_BUDGET}")
    print(f"num_train_rows:      {len(X_train)}")
    print(f"num_features:        {len(ALL_FEATURES)}")


if __name__ == "__main__":
    main()
