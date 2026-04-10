import math
import re

import numpy as np


MAX_FORM_RUNS = 5


def parse_float(value):
    if value is None or value == "" or value == "—":
        return None
    try:
        return float(str(value).replace("$", "").replace(",", "").strip())
    except Exception:
        return None


def parse_pos(value):
    if not value:
        return None, None
    match = re.match(r"(\d+)(?:st|nd|rd|th)?(?:/(\d+))?", str(value).strip(), re.IGNORECASE)
    if not match:
        return None, None
    pos = int(match.group(1)) if match.group(1) else None
    field = int(match.group(2)) if match.group(2) else None
    return pos, field


def parse_form_date(value):
    from datetime import datetime

    try:
        return datetime.strptime(value, "%d/%m/%Y")
    except Exception:
        return None


def parse_career_line(value):
    match = re.match(r"(\d+):(\d+)-(\d+)-(\d+)", value or "")
    if not match:
        return 0, 0, 0, 0
    return tuple(int(match.group(i)) for i in range(1, 5))


def normalize_text(value):
    return (value or "").strip().lower()


def same_track(history_track, race_track):
    history_track = normalize_text(history_track)
    race_track = normalize_text(race_track)
    return bool(history_track and race_track and history_track.startswith(race_track[:4]))


def runner_features(runner, race):
    race_dt = race["race_dt"]
    race_distance = race["distance"]
    race_track = race["track"]

    valid = []
    for history in runner.get("form_history") or []:
        history_date = parse_form_date(history.get("date", ""))
        if history_date is None or (race_dt is not None and history_date >= race_dt):
            continue
        valid.append((history, history_date))
        if len(valid) >= MAX_FORM_RUNS:
            break

    n = len(valid)
    wins = 0
    places = 0
    positions = []
    times = []
    margins = []
    first_splits = []
    td_runs = 0
    td_wins = 0
    dist_runs = 0
    dist_wins = 0

    for history, _ in valid:
        pos, _ = parse_pos(history.get("position"))
        if pos is not None:
            positions.append(pos)
            wins += int(pos == 1)
            places += int(pos <= 3)
        time_value = parse_float(history.get("time"))
        if time_value:
            times.append(time_value)
        split_value = parse_float(history.get("first_split"))
        if split_value:
            first_splits.append(split_value)
        margin_value = parse_float(history.get("margin"))
        if margin_value is not None:
            margins.append(margin_value)

        hist_distance = history.get("distance")
        if hist_distance and race_distance and int(hist_distance) == race_distance:
            dist_runs += 1
            dist_wins += int(pos == 1)
            if same_track(history.get("track"), race_track):
                td_runs += 1
                td_wins += int(pos == 1)

    if valid and race_dt is not None:
        days_since = (race_dt - valid[0][1]).days
    else:
        days_since = -1

    latest = valid[0][0] if valid else None
    latest_pos, latest_field = parse_pos(latest.get("position")) if latest else (None, None)
    latest_margin = parse_float(latest.get("margin")) if latest else None
    latest_time = parse_float(latest.get("time")) if latest else None
    latest_first_split = parse_float(latest.get("first_split")) if latest else None

    def avg(xs):
        return float(np.mean(xs)) if xs else 0.0

    def minv(xs):
        return float(np.min(xs)) if xs else 0.0

    starts, wins_all, seconds_all, thirds_all = parse_career_line(runner.get("career_all"))
    td_starts, td_wins_all, _, _ = parse_career_line(runner.get("career_td"))

    best_time = parse_float(runner.get("best_time")) or 0.0
    best_split = parse_float(runner.get("best_split")) or 0.0

    return {
        "track": race["track"] or "unknown",
        "box": runner.get("box") or 0,
        "race_num": race["race_num"] or 0,
        "distance": race_distance or 0,
        "field_size": len(race["runners"]),
        "has_form": 1 if n else 0,
        "has_form_time": 1 if times else 0,
        "has_form_first_split": 1 if first_splits else 0,
        "has_last_pos": 1 if latest_pos is not None else 0,
        "has_last_margin": 1 if latest_margin is not None else 0,
        "has_last_time": 1 if latest_time is not None else 0,
        "has_last_first_split": 1 if latest_first_split is not None else 0,
        "has_best_time": 1 if best_time > 0 else 0,
        "has_best_split": 1 if best_split > 0 else 0,
        "form_runs": n,
        "last_start_win": 1.0 if latest_pos == 1 else 0.0,
        "last_start_pos": float(latest_pos) if latest_pos is not None else 0.0,
        "last_start_pos_ratio": (latest_pos / latest_field) if latest_pos and latest_field else 0.0,
        "last_start_margin": latest_margin if latest_margin is not None else 0.0,
        "last_start_time": latest_time if latest_time is not None else 0.0,
        "last_start_first_split": latest_first_split if latest_first_split is not None else 0.0,
        "form_win_rate": (wins / n) if n else 0.0,
        "form_place_rate": (places / n) if n else 0.0,
        "form_avg_pos": avg(positions),
        "form_avg_margin": avg(margins),
        "form_avg_time": avg(times),
        "form_best_time": minv(times),
        "form_avg_first_split": avg(first_splits),
        "form_days_since": days_since,
        "form_dist_runs": dist_runs,
        "form_dist_win_rate": (dist_wins / dist_runs) if dist_runs else 0.0,
        "form_td_runs": td_runs,
        "form_td_win_rate": (td_wins / td_runs) if td_runs else 0.0,
        "career_starts": starts,
        "career_win_rate": (wins_all / starts) if starts else 0.0,
        "career_place_rate": ((wins_all + seconds_all + thirds_all) / starts) if starts else 0.0,
        "career_td_starts": td_starts,
        "career_td_win_rate": (td_wins_all / td_starts) if td_starts else 0.0,
        "best_time": best_time,
        "best_split": best_split,
    }


RELATIVE_FEATURE_SPECS = [
    ("box", "min", lambda row: False),
    ("form_runs", "max", lambda row: False),
    ("last_start_pos", "min", lambda row: not row["has_last_pos"]),
    ("last_start_pos_ratio", "min", lambda row: not row["has_last_pos"]),
    ("last_start_margin", "min", lambda row: not row["has_last_margin"]),
    ("last_start_time", "min", lambda row: not row["has_last_time"]),
    ("last_start_first_split", "min", lambda row: not row["has_last_first_split"]),
    ("form_win_rate", "max", lambda row: False),
    ("form_place_rate", "max", lambda row: False),
    ("form_avg_pos", "min", lambda row: not row["has_form"]),
    ("form_avg_time", "min", lambda row: not row["has_form_time"]),
    ("form_best_time", "min", lambda row: not row["has_form_time"]),
    ("form_avg_first_split", "min", lambda row: not row["has_form_first_split"]),
    ("form_dist_runs", "max", lambda row: False),
    ("form_dist_win_rate", "max", lambda row: False),
    ("form_td_runs", "max", lambda row: False),
    ("form_td_win_rate", "max", lambda row: False),
    ("career_starts", "max", lambda row: False),
    ("career_win_rate", "max", lambda row: False),
    ("career_place_rate", "max", lambda row: False),
    ("best_time", "min", lambda row: not row["has_best_time"]),
    ("best_split", "min", lambda row: not row["has_best_split"]),
]


def add_relative_features(feature_rows):
    n = len(feature_rows)
    if not n:
        return feature_rows

    for name, direction, missing_fn in RELATIVE_FEATURE_SPECS:
        scored = []
        for idx, row in enumerate(feature_rows):
            missing = missing_fn(row)
            value = row[name]
            sort_value = math.inf if (missing and direction == "min") else (-math.inf if missing else value)
            scored.append((idx, sort_value, missing))

        order = sorted(scored, key=lambda item: item[1], reverse=(direction == "max"))
        denom = max(n - 1, 1)
        finite_values = [value for _, value, _ in scored if math.isfinite(value)]
        if finite_values:
            best = min(finite_values) if direction == "min" else max(finite_values)
            worst = max(finite_values) if direction == "min" else min(finite_values)
            span = max(abs(worst - best), 1e-9)
        else:
            best = 0.0
            span = 1.0

        for rank, (idx, value, missing) in enumerate(order):
            feature_rows[idx][f"{name}_field_rank"] = 1.0 - (rank / denom)
            if missing or not math.isfinite(value):
                feature_rows[idx][f"{name}_field_gap"] = 1.0
            elif direction == "min":
                feature_rows[idx][f"{name}_field_gap"] = (value - best) / span
            else:
                feature_rows[idx][f"{name}_field_gap"] = (best - value) / span
    return feature_rows


FEATURE_GROUPS = {
    "race_context": [
        "box",
        "race_num",
        "distance",
        "field_size",
        "track",
    ],
    "availability": [
        "has_form",
        "has_form_time",
        "has_form_first_split",
        "has_last_pos",
        "has_last_margin",
        "has_last_time",
        "has_last_first_split",
        "has_best_time",
        "has_best_split",
    ],
    "recent_form": [
        "form_runs",
        "last_start_win",
        "last_start_pos",
        "last_start_pos_ratio",
        "last_start_margin",
        "last_start_time",
        "last_start_first_split",
        "form_win_rate",
        "form_place_rate",
        "form_avg_pos",
        "form_avg_margin",
        "form_avg_time",
        "form_best_time",
        "form_avg_first_split",
        "form_days_since",
    ],
    "specialization": [
        "form_dist_runs",
        "form_dist_win_rate",
        "form_td_runs",
        "form_td_win_rate",
        "career_td_starts",
        "career_td_win_rate",
    ],
    "career": [
        "career_starts",
        "career_win_rate",
        "career_place_rate",
    ],
    "speed": [
        "best_time",
        "best_split",
    ],
    "relative": [
        f"{name}_{suffix}"
        for name, _, _ in RELATIVE_FEATURE_SPECS
        for suffix in ("field_rank", "field_gap")
    ],
}


CAT_FEATURES = ["track"]


def feature_names_for_groups(groups):
    seen = []
    for group in groups:
        for name in FEATURE_GROUPS[group]:
            if name not in seen:
                seen.append(name)
    return seen


def race_feature_rows(race):
    rows = []
    for runner in race["runners"]:
        rows.append(runner_features(runner, race))
    return add_relative_features(rows)
