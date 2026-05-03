import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _default_label(path):
    name = path.stem
    for prefix in ("ours_policy_diversity_", "ours_policy_", "ours_best_diversity_", "ours_best_"):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _read_series(path):
    df = pd.read_csv(path)
    required = {"samples", "prob_failure", "reward", "dist_uni"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {', '.join(sorted(missing))}")

    out = df.copy()
    out["prob_failure"] = out["prob_failure"].astype(float).abs()
    out["reward"] = out["reward"].astype(float)
    out["dist_uni"] = out["dist_uni"].astype(float)
    out["samples"] = out["samples"].astype(float)
    return out


def main():
    parser = argparse.ArgumentParser(description="Plot ApproPO experiment metrics against environment samples.")
    parser.add_argument("csvs", nargs="+", type=Path, help="Result CSV files from ApproPO/RCPO runs")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels, one per CSV")
    parser.add_argument("--output", type=Path, default=Path("results/learning_curves.png"))
    parser.add_argument("--failure-threshold", type=float, default=0.2)
    parser.add_argument("--reward-threshold", type=float, default=-0.17)
    parser.add_argument("--diversity-threshold", type=float, default=0.12)
    parser.add_argument("--xmax", type=float, default=None)
    parser.add_argument("--title", type=str, default=None)
    args = parser.parse_args()

    if args.labels is not None and len(args.labels) not in (0, len(args.csvs)):
        raise ValueError("--labels must have the same number of entries as CSV files")

    labels = args.labels if args.labels else [_default_label(path) for path in args.csvs]
    series = [(_read_series(path), label) for path, label in zip(args.csvs, labels)]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 9.5), sharex=True, constrained_layout=True)
    specs = [
        ("prob_failure", "probability of failure", args.failure_threshold, "upper"),
        ("reward", "average reward", args.reward_threshold, "lower"),
        ("dist_uni", "distance from uniform", args.diversity_threshold, "upper"),
    ]

    for ax, (column, ylabel, threshold, _) in zip(axes, specs):
        for df, label in series:
            ax.plot(df["samples"], df[column], linewidth=2.0, label=label)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1.8, label="Constraint")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        if args.xmax is not None:
            ax.set_xlim(left=0, right=args.xmax)
        else:
            ax.set_xlim(left=0)

    axes[-1].set_xlabel("samples")
    axes[-1].legend(loc="best", frameon=True, fontsize=9)
    if args.title:
        fig.suptitle(args.title)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
