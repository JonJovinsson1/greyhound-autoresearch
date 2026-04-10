import argparse
import sys
from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inverse_bottom3.experiment import run_experiment
from inverse_bottom3.features import FEATURE_GROUPS


def parse_groups(raw_groups):
    groups = [part.strip() for part in raw_groups.split(",") if part.strip()]
    unknown = [group for group in groups if group not in FEATURE_GROUPS]
    if unknown:
        raise SystemExit(f"Unknown groups: {', '.join(unknown)}")
    return groups


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", required=True, help="Comma-separated feature groups.")
    parser.add_argument("--tag", default="", help="Optional run label for the output.")
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--l2-leaf-reg", type=float, default=5.0)
    parser.add_argument("--random-seed", type=int, default=42)
    args = parser.parse_args()

    groups = parse_groups(args.groups)
    result = run_experiment(
        groups=groups,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        depth=args.depth,
        l2_leaf_reg=args.l2_leaf_reg,
        random_seed=args.random_seed,
    )

    tag_prefix = f"{args.tag}: " if args.tag else ""
    print("---")
    print(f"{tag_prefix}groups:                {','.join(result['groups'])}")
    print(f"val_bottom3_slot_accuracy:   {result['val']['slot_accuracy']:.6f}")
    print(f"val_bottom3_exact_accuracy:  {result['val']['exact_accuracy']:.6f}")
    print(f"val_runner_auc:              {result['val']['runner_auc']:.6f}")
    print(f"val_races:                   {result['val']['n_races']}")
    print(f"test_bottom3_slot_accuracy:  {result['test']['slot_accuracy']:.6f}")
    print(f"test_bottom3_exact_accuracy: {result['test']['exact_accuracy']:.6f}")
    print(f"test_runner_auc:             {result['test']['runner_auc']:.6f}")
    print(f"test_races:                  {result['test']['n_races']}")
    print(f"train_seconds:               {result['train_seconds']:.1f}")
    print(f"total_seconds:               {result['total_seconds']:.1f}")
    print(f"num_train_rows:              {result['num_train_rows']}")
    print(f"num_features:                {result['num_features']}")


if __name__ == "__main__":
    main()
