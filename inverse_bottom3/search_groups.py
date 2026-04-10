import argparse
from pathlib import Path
import sys

import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse_bottom3.experiment import group_combinations, run_experiment
from inverse_bottom3.features import FEATURE_GROUPS


def append_row(results_path, row):
    results_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    write_header = not results_path.exists()
    frame.to_csv(results_path, sep="\t", index=False, mode="a", header=write_header)


def row_from_result(mode, result):
    return {
        "mode": mode,
        "groups": ",".join(result["groups"]),
        "val_slot": result["val"]["slot_accuracy"],
        "val_exact": result["val"]["exact_accuracy"],
        "val_auc": result["val"]["runner_auc"],
        "test_slot": result["test"]["slot_accuracy"],
        "test_exact": result["test"]["exact_accuracy"],
        "test_auc": result["test"]["runner_auc"],
        "seconds": result["total_seconds"],
        "num_features": result["num_features"],
    }


def greedy_search(results_path, iterations, learning_rate, depth, l2_leaf_reg, random_seed):
    remaining = list(FEATURE_GROUPS)
    chosen = []
    best_val = -1.0

    while remaining:
        best_candidate = None
        best_result = None
        for group in remaining:
            candidate = chosen + [group]
            result = run_experiment(
                groups=candidate,
                iterations=iterations,
                learning_rate=learning_rate,
                depth=depth,
                l2_leaf_reg=l2_leaf_reg,
                random_seed=random_seed,
            )
            append_row(results_path, row_from_result("greedy", result))
            if result["val"]["slot_accuracy"] > best_val and (
                best_result is None or result["val"]["slot_accuracy"] > best_result["val"]["slot_accuracy"]
            ):
                best_candidate = group
                best_result = result

        if best_candidate is None:
            break
        chosen.append(best_candidate)
        remaining.remove(best_candidate)
        best_val = best_result["val"]["slot_accuracy"]
        print(
            f"keep {best_candidate}: val_slot={best_result['val']['slot_accuracy']:.6f} "
            f"test_slot={best_result['test']['slot_accuracy']:.6f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["singles", "pairs", "greedy"], required=True)
    parser.add_argument("--results-path", required=True)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    results_path = Path(args.results_path)

    if args.mode == "greedy":
        greedy_search(
            results_path=results_path,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.random_seed,
        )
        return

    combinations = group_combinations(args.mode, list(FEATURE_GROUPS))
    for groups in combinations:
        result = run_experiment(
            groups=groups,
            iterations=args.iterations,
            learning_rate=args.learning_rate,
            depth=args.depth,
            l2_leaf_reg=args.l2_leaf_reg,
            random_seed=args.random_seed,
        )
        append_row(results_path, row_from_result(args.mode, result))
        print(
            f"{','.join(result['groups'])}: "
            f"val_slot={result['val']['slot_accuracy']:.6f} "
            f"test_slot={result['test']['slot_accuracy']:.6f}"
        )


if __name__ == "__main__":
    main()
