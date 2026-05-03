import argparse
import ast
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from ApproPO.envs.gym_frozenmarsrover.envs.maps import MAPS


def _load_visitation(csv_path, row):
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError(f"{csv_path} does not contain any rows")

    row_idx = row if row >= 0 else len(df) + row
    if row_idx < 0 or row_idx >= len(df):
        raise IndexError(f"row {row} is out of range for {len(df)} rows")

    obs_cell = ast.literal_eval(df.iloc[row_idx]["obs"])
    values = np.array(obs_cell["obs"], dtype=float)
    side = int(np.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"expected a square grid, got {values.size} values")
    return values.reshape(side, side), df.iloc[row_idx]


def _draw_base_grid(ax, map_rows):
    nrow = len(map_rows)
    ncol = len(map_rows[0])
    colors = {"S": "#8f8f86", "G": "#078b19", "H": "#f2a900", "F": "#eef6f7"}

    grid = np.zeros((nrow, ncol), dtype=int)
    palette = ["#eef6f7", colors["S"], colors["G"], colors["H"]]
    for r, row in enumerate(map_rows):
        for c, cell in enumerate(row):
            grid[r, c] = {"F": 0, "S": 1, "G": 2, "H": 3}[cell]

    ax.imshow(grid, cmap=ListedColormap(palette), vmin=0, vmax=3)
    _style_grid(ax, nrow, ncol)

    for r, row in enumerate(map_rows):
        for c, cell in enumerate(row):
            if cell == "S":
                ax.text(c, r, "*", ha="center", va="center", color="yellow", fontsize=18, fontweight="bold")
            elif cell == "G":
                ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=10)
            elif cell == "H":
                ax.text(c, r, "R", ha="center", va="center", color="black", fontsize=10)


def _draw_visitation_grid(ax, map_rows, visitation, vmax):
    nrow, ncol = visitation.shape
    cmap = plt.get_cmap("YlGnBu").copy()
    cmap.set_bad("#ffffff")

    values = visitation.copy()
    for r, row in enumerate(map_rows):
        for c, cell in enumerate(row):
            if cell == "H":
                values[r, c] = np.nan

    im = ax.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    _style_grid(ax, nrow, ncol)

    for r, row in enumerate(map_rows):
        for c, cell in enumerate(row):
            if cell == "S":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#8f8f86"))
                ax.text(c, r, "*", ha="center", va="center", color="yellow", fontsize=18, fontweight="bold")
            elif cell == "G":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#078b19"))
                ax.text(c, r, "G", ha="center", va="center", color="white", fontsize=10)
            elif cell == "H":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="#f2a900"))
                ax.text(c, r, "R", ha="center", va="center", color="black", fontsize=10)

    return im


def _style_grid(ax, nrow, ncol):
    ax.set_xticks(np.arange(-0.5, ncol, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrow, 1), minor=True)
    ax.grid(which="minor", color="black", linewidth=1.2)
    ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(nrow - 0.5, -0.5)


def main():
    parser = argparse.ArgumentParser(description="Plot Mars rover visitation probabilities from an ApproPO CSV.")
    parser.add_argument("csv", type=Path, help="CSV produced by ApproPO, e.g. results/ours_policy_diversity_tmp.csv")
    parser.add_argument("--row", type=int, default=-1, help="CSV row to plot; defaults to the last row")
    parser.add_argument("--output", type=Path, default=Path("results/visitation_grid.png"))
    parser.add_argument("--map", choices=MAPS.keys(), default="8x8")
    parser.add_argument("--vmax", type=float, default=None, help="Color scale maximum; defaults to observed max")
    args = parser.parse_args()

    visitation, row = _load_visitation(args.csv, args.row)
    map_rows = MAPS[args.map]
    vmax = args.vmax if args.vmax is not None else max(float(np.nanmax(visitation)), 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6), constrained_layout=True)
    _draw_base_grid(axes[0], map_rows)
    axes[0].set_title("Mars rover map")

    im = _draw_visitation_grid(axes[1], map_rows, visitation, vmax)
    axes[1].set_title(f"Visitation probabilities, epoch {int(row['ep'])}")

    divider = make_axes_locatable(axes[1])
    cax = divider.append_axes("right", size="5%", pad=0.12)
    fig.colorbar(im, cax=cax)

    fig.suptitle(
        f"failure={abs(float(row['prob_failure'])):.3f}, "
        f"reward={float(row['reward']):.3f}, "
        f"dist_uni={float(row['dist_uni']):.3f}",
        y=1.03,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=200, bbox_inches="tight")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
