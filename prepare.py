"""
Fixed data-layer + evaluation harness for greyhound autoresearch.

Imported by train.py. DO NOT MODIFY during experiments — this file defines the
ground-truth metric and the data contract. If you change constants here, you
break comparability with past runs.

Metric philosophy (deliberate):
- The goal is PURE top-1 win-rate accuracy. "Did the model's highest-probability
  runner actually win this race?" — averaged over races.
- Odds / starting price are NOT used. They are not features, not a target, and
  not part of the metric. We intentionally do not expose `sp` on runners so the
  agent cannot accidentally leak post-race information into a feature.
- Log-loss on the true winner is reported as a secondary diagnostic, not the
  objective.

Exposes:
    TRAIN_DIR, VAL_DIR, TEST_DIR   — data roots (historic vs pre-race layouts)
    TIME_BUDGET                    — wall-clock training cap (seconds)
    load_races(split)              — generator over normalized race dicts
    evaluate(predict_fn, split)    — the fixed metric: top-1 accuracy + log-loss
"""

import json
import math
import re
from datetime import datetime
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "train"     # 2025-10 … 2026-01, historic (race_info.results)
VAL_DIR   = ROOT / "val"       # 2026-02, historic
TEST_DIR  = ROOT / "test"      # 2026-03-17 … 2026-04-04, pre-race (temp_result.results)

TIME_BUDGET = 300              # 5 min wall-clock training cap (honor this)
MAX_BOX = 8                    # drop reserve boxes (>8)

# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_pos_1st(s):
    """True if position string indicates a win ('1st' or '1st=')."""
    return bool(s) and str(s).strip().startswith("1")


def _parse_distance(gd):
    m = re.search(r"(\d{3,4})\s*m", gd or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# Race loading — handles both historic and pre-race JSON variants
# ---------------------------------------------------------------------------

def _normalize(raw, src_path):
    """
    Unify historic (race_info.results, meta.*) and pre-race (temp_result.results,
    top-level track/date) formats into a single dict. Returns None if the race
    can't be used (no results, no winner, too few runners).

    We DELIBERATELY strip all odds/sp fields from the returned runner dicts so
    they cannot be used as features. Odds are post-race info for historic data
    and out-of-scope regardless.
    """
    # Meta can be nested under 'meta' (historic) or flat at top level (pre-race).
    meta = raw.get("meta") or {}
    track = meta.get("track") or raw.get("track") or ""
    date  = meta.get("date")  or raw.get("date")  or ""
    race_num = meta.get("race_number") or raw.get("race_num")

    # Results live under race_info.results (historic) or temp_result.results (pre-race).
    ri = raw.get("race_info") or {}
    results = ri.get("results") or []
    gd = ri.get("grade_distance") or raw.get("grade_distance") or ""
    if not results:
        tr = raw.get("temp_result") or {}
        results = tr.get("results") or []
    if not results:
        return None

    runners = raw.get("runners") or []
    runners = [r for r in runners if isinstance(r.get("box"), int) and 1 <= r["box"] <= MAX_BOX]
    if len(runners) < 2:
        return None

    # Winner by box.
    winner_box = None
    for r in results:
        if _parse_pos_1st(r.get("position")) and isinstance(r.get("box"), int):
            winner_box = r["box"]
            break
    if winner_box is None:
        return None

    # Strip odds from runners and their form history so they can't leak as features.
    for r in runners:
        r.pop("sp", None)
        for fh in (r.get("form_history") or []):
            fh.pop("sp", None)

    try:
        race_dt = datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        race_dt = None

    try:
        race_num_int = int(race_num) if race_num is not None else None
    except Exception:
        race_num_int = None

    return {
        "race_id": src_path.stem,
        "track": (track or "").lower().strip(),
        "date": date,
        "race_dt": race_dt,
        "race_num": race_num_int,
        "grade_distance": gd,
        "distance": _parse_distance(gd),
        "runners": runners,
        "winner_box": winner_box,
    }


def _iter_json_files(root):
    """Yield .json files under any subdirectory structure (train/YYYY-MM, test/YYYY-MM-DD)."""
    if not root.is_dir():
        return
    for sub in sorted(p for p in root.iterdir() if p.is_dir()):
        for f in sorted(sub.glob("*.json")):
            yield f


def load_races(split):
    """Generator of normalized race dicts for 'train' | 'val' | 'test'."""
    root = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split]
    for path in _iter_json_files(root):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        race = _normalize(raw, path)
        if race is not None:
            yield race


# ---------------------------------------------------------------------------
# Evaluation — THIS IS THE FIXED METRIC. Do not reinvent in train.py.
# ---------------------------------------------------------------------------

def evaluate(predict_fn, split):
    """
    predict_fn(race) -> list of scores, one per runner in race['runners'],
                        in the same order. Higher score = more likely to win.
                        Scores can be un-normalized; log-loss uses softmax-normalized
                        probabilities over the field.

    Primary metric (maximize): top1_accuracy — fraction of races where the
    runner with the highest predicted score actually won.

    Secondary (minimize): log_loss — mean negative log-likelihood of the true
    winner under field-normalized probabilities. Useful as a tie-breaker and
    for diagnosing over-confident-but-wrong models.

    Ties in top-score are broken by picking the first runner with the max score.
    """
    n_races = 0
    n_top1 = 0
    total_log_loss = 0.0
    eps = 1e-9

    for race in load_races(split):
        runners = race["runners"]
        scores = predict_fn(race)
        if scores is None or len(scores) != len(runners):
            raise ValueError(
                f"predict_fn returned {None if scores is None else len(scores)} scores "
                f"for {len(runners)} runners in race {race['race_id']}"
            )

        scores = [float(s) for s in scores]

        # Top-1 pick (first argmax)
        best_i = max(range(len(scores)), key=lambda i: scores[i])
        if runners[best_i]["box"] == race["winner_box"]:
            n_top1 += 1

        # Log-loss via softmax over the field (numerically stable)
        m = max(scores)
        exps = [math.exp(s - m) for s in scores]
        z = sum(exps)
        winner_p = eps
        for r, e in zip(runners, exps):
            if r["box"] == race["winner_box"]:
                winner_p = max(e / z, eps)
                break
        total_log_loss += -math.log(winner_p)
        n_races += 1

    return {
        "top1_accuracy": (n_top1 / n_races) if n_races else 0.0,
        "log_loss": (total_log_loss / n_races) if n_races else float("inf"),
        "n_races": n_races,
        "n_top1": n_top1,
    }


# ---------------------------------------------------------------------------
# Quick health-check (run this file directly to verify data availability)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for split, root in [("train", TRAIN_DIR), ("val", VAL_DIR), ("test", TEST_DIR)]:
        if not root.is_dir():
            print(f"{split:5s}: MISSING — {root}")
            continue
        n = sum(1 for _ in load_races(split))
        print(f"{split:5s}: {n:>5d} usable races from {root}")
    print(f"\nTIME_BUDGET = {TIME_BUDGET}s")
    print("Metric: primary=top1_accuracy (higher=better), secondary=log_loss")
    print("No odds / SP used anywhere.")
