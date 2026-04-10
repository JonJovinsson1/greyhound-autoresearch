import json
import re
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent.parent
TRAIN_DIR = ROOT / "train"
VAL_DIR = ROOT / "val"
TEST_DIR = ROOT / "test"
MAX_BOX = 8

NON_FINISHER_CODES = {"F", "FELL", "ABD", "FTF", "P", "OFF", "-"}
SCRATCH_CODES = {"SCR", "L/SCR"}


def parse_distance(grade_distance):
    match = re.search(r"(\d{3,4})\s*m", grade_distance or "")
    return int(match.group(1)) if match else None


def parse_race_date(date_str):
    try:
        return datetime.strptime(date_str, "%Y-%m-%d")
    except Exception:
        return None


def parse_position_token(token):
    text = (token or "").strip().upper()
    match = re.match(r"^(\d+)(?:ST|ND|RD|TH)?=?$", text)
    if match:
        return ("finish", int(match.group(1)))
    if text in SCRATCH_CODES:
        return ("scratch", None)
    if text in NON_FINISHER_CODES:
        return ("nonfinish", None)
    return ("other", None)


def iter_json_files(root):
    if not root.is_dir():
        return
    for subdir in sorted(path for path in root.iterdir() if path.is_dir()):
        for path in sorted(subdir.glob("*.json")):
            yield path


def _normalize(raw, src_path):
    meta = raw.get("meta") or {}
    track = meta.get("track") or raw.get("track") or ""
    date = meta.get("date") or raw.get("date") or ""
    race_num = meta.get("race_number") or raw.get("race_num")
    race_info = raw.get("race_info") or {}
    temp_result = raw.get("temp_result") or {}
    results = race_info.get("results") or temp_result.get("results") or []
    grade_distance = race_info.get("grade_distance") or raw.get("grade_distance") or ""

    if not results:
        return None

    runners = raw.get("runners") or []
    runners = [runner for runner in runners if isinstance(runner.get("box"), int) and 1 <= runner["box"] <= MAX_BOX]
    if len(runners) < 4:
        return None

    runner_by_box = {runner["box"]: runner for runner in runners}
    finish_rows = []
    winner_box = None

    for result in results:
        box = result.get("box")
        if box not in runner_by_box:
            continue
        kind, position = parse_position_token(result.get("position"))
        if kind == "scratch":
            continue
        if kind == "finish" and position == 1 and winner_box is None:
            winner_box = box
        if kind == "finish":
            sort_key = (0, position, box)
        elif kind == "nonfinish":
            sort_key = (1, len(runners) + 1, box)
        else:
            sort_key = (2, len(runners) + 2, box)
        finish_rows.append((box, kind, position, sort_key))

    if winner_box is None or len(finish_rows) < 3:
        return None

    finish_rows.sort(key=lambda row: row[3])
    ranked_boxes = [box for box, _, _, _ in finish_rows]
    if len(ranked_boxes) < 3:
        return None

    bottom3_boxes = set(ranked_boxes[-3:])
    finish_rank = {box: idx + 1 for idx, box in enumerate(ranked_boxes)}

    try:
        race_num_int = int(race_num) if race_num is not None else None
    except Exception:
        race_num_int = None

    for runner in runners:
        runner.pop("sp", None)
        for history in runner.get("form_history") or []:
            history.pop("sp", None)

    return {
        "race_id": src_path.stem,
        "track": (track or "").strip().lower(),
        "date": date,
        "race_dt": parse_race_date(date),
        "race_num": race_num_int,
        "grade_distance": grade_distance,
        "distance": parse_distance(grade_distance),
        "runners": runners,
        "winner_box": winner_box,
        "finish_rank": finish_rank,
        "bottom3_boxes": bottom3_boxes,
    }


def load_races(split):
    root = {"train": TRAIN_DIR, "val": VAL_DIR, "test": TEST_DIR}[split]
    for path in iter_json_files(root):
        try:
            raw = json.loads(path.read_text())
        except Exception:
            continue
        race = _normalize(raw, path)
        if race is not None:
            yield race
