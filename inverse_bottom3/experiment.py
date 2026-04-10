import itertools
import math
import time

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

from inverse_bottom3.data import load_races
from inverse_bottom3.features import CAT_FEATURES, feature_names_for_groups, race_feature_rows


DEFAULT_ITERATIONS = 400
DEFAULT_LEARNING_RATE = 0.05
DEFAULT_DEPTH = 5
DEFAULT_L2 = 5.0
DEFAULT_RANDOM_SEED = 42
DEFAULT_GROUPS = ("relative", "specialization", "race_context")


def race_runner_table(split, groups):
    selected_features = feature_names_for_groups(groups)
    rows = []
    race_rows = []

    for race_idx, race in enumerate(load_races(split)):
        feature_rows = race_feature_rows(race)
        row_indices = []
        bottom3_boxes = set(race["bottom3_boxes"])
        for runner, features in zip(race["runners"], feature_rows):
            row = {name: features[name] for name in selected_features}
            row["race_id"] = race["race_id"]
            row["box"] = runner["box"]
            row["label_bottom3"] = int(runner["box"] in bottom3_boxes)
            row["finish_rank"] = race["finish_rank"].get(runner["box"], len(race["runners"]) + 1)
            rows.append(row)
            row_indices.append(len(rows) - 1)
        race_rows.append(
            {
                "race_id": race["race_id"],
                "boxes": [runner["box"] for runner in race["runners"]],
                "bottom3_boxes": bottom3_boxes,
                "row_indices": row_indices,
            }
        )

    frame = pd.DataFrame(rows)
    return frame, race_rows, selected_features


def build_catboost_inputs(frame, feature_names):
    X = frame[feature_names].copy()
    y = frame["label_bottom3"].astype(int).to_numpy()
    cat_idx = [feature_names.index(name) for name in CAT_FEATURES if name in feature_names]
    return X, y, cat_idx


def fit_bottom3_model(
    train_frame,
    feature_names,
    iterations=DEFAULT_ITERATIONS,
    learning_rate=DEFAULT_LEARNING_RATE,
    depth=DEFAULT_DEPTH,
    l2_leaf_reg=DEFAULT_L2,
    random_seed=DEFAULT_RANDOM_SEED,
):
    X_train, y_train, cat_idx = build_catboost_inputs(train_frame, feature_names)
    model = CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        loss_function="Logloss",
        eval_metric="Logloss",
        random_seed=random_seed,
        verbose=False,
        task_type="CPU",
        thread_count=-1,
    )
    model.fit(Pool(X_train, y_train, cat_features=cat_idx))
    return model, cat_idx


def score_frame(model, frame, feature_names, cat_idx):
    X = frame[feature_names].copy()
    probs = model.predict_proba(Pool(X, cat_features=cat_idx))[:, 1]
    scored = frame.copy()
    scored["bottom3_score"] = probs
    return scored


def build_race_frame(race, feature_names):
    feature_rows = race_feature_rows(race)
    frame = pd.DataFrame([{name: row[name] for name in feature_names} for row in feature_rows])
    return frame, feature_rows


def train_bottom3_side_model(
    groups=DEFAULT_GROUPS,
    split="train",
    iterations=DEFAULT_ITERATIONS,
    learning_rate=DEFAULT_LEARNING_RATE,
    depth=DEFAULT_DEPTH,
    l2_leaf_reg=DEFAULT_L2,
    random_seed=DEFAULT_RANDOM_SEED,
):
    frame, _, feature_names = race_runner_table(split, groups)
    model, cat_idx = fit_bottom3_model(
        train_frame=frame,
        feature_names=feature_names,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
    )
    return {
        "model": model,
        "groups": list(groups),
        "feature_names": feature_names,
        "cat_idx": cat_idx,
        "train_rows": int(len(frame)),
    }


def score_race_bottom3(race, bundle):
    frame, feature_rows = build_race_frame(race, bundle["feature_names"])
    probs = bundle["model"].predict_proba(Pool(frame, cat_features=bundle["cat_idx"]))[:, 1]
    return probs, feature_rows


def _safe_auc(y_true, y_score):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = int(y_true.sum())
    neg = int((1 - y_true).sum())
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)
    pos_ranks = ranks[y_true == 1].sum()
    return (pos_ranks - (pos * (pos + 1) / 2.0)) / (pos * neg)


def evaluate_bottom3(scored_frame, race_rows):
    race_lookup = scored_frame.reset_index(drop=True)
    total_overlap = 0
    exact_hits = 0
    n_races = 0

    for race in race_rows:
        race_slice = race_lookup.iloc[race["row_indices"]]
        ranked = race_slice.sort_values(["bottom3_score", "box"], ascending=[False, True])
        predicted = set(ranked["box"].head(3).tolist())
        actual = set(race["bottom3_boxes"])
        overlap = len(predicted & actual)
        total_overlap += overlap
        exact_hits += int(predicted == actual)
        n_races += 1

    y_true = race_lookup["label_bottom3"].to_numpy()
    y_score = race_lookup["bottom3_score"].to_numpy()
    return {
        "slot_accuracy": (total_overlap / (3 * n_races)) if n_races else 0.0,
        "exact_accuracy": (exact_hits / n_races) if n_races else 0.0,
        "runner_auc": _safe_auc(y_true, y_score),
        "n_races": n_races,
    }


def run_experiment(
    groups,
    iterations=DEFAULT_ITERATIONS,
    learning_rate=DEFAULT_LEARNING_RATE,
    depth=DEFAULT_DEPTH,
    l2_leaf_reg=DEFAULT_L2,
    random_seed=DEFAULT_RANDOM_SEED,
):
    t0 = time.time()
    train_frame, _, feature_names = race_runner_table("train", groups)
    val_frame, val_races, _ = race_runner_table("val", groups)
    test_frame, test_races, _ = race_runner_table("test", groups)

    model, cat_idx = fit_bottom3_model(
        train_frame=train_frame,
        feature_names=feature_names,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        l2_leaf_reg=l2_leaf_reg,
        random_seed=random_seed,
    )
    train_seconds = time.time() - t0

    val_scored = score_frame(model, val_frame, feature_names, cat_idx)
    test_scored = score_frame(model, test_frame, feature_names, cat_idx)
    val_metrics = evaluate_bottom3(val_scored, val_races)
    test_metrics = evaluate_bottom3(test_scored, test_races)
    total_seconds = time.time() - t0

    return {
        "groups": list(groups),
        "feature_names": feature_names,
        "num_train_rows": int(len(train_frame)),
        "num_features": int(len(feature_names)),
        "train_seconds": train_seconds,
        "total_seconds": total_seconds,
        "val": val_metrics,
        "test": test_metrics,
    }


def group_combinations(mode, group_names):
    if mode == "singles":
        return [(name,) for name in group_names]
    if mode == "pairs":
        return list(itertools.combinations(group_names, 2))
    raise ValueError(f"unsupported mode: {mode}")
