"""Exploratory data analysis figures for the reptile SDM thesis (Chapter 3).

Produces four publication-ready figures from the 641-plot labelled dataset:

  1. eda_histograms.pdf   -- key continuous predictors, split by presence/absence
  2. eda_correlation.pdf  -- correlation heatmap of the numeric predictors
  3. eda_presence_map.pdf -- plot locations coloured by presence, per species
  4. eda_gradient_map.pdf -- an environmental gradient (rainfall) over space

Run from the repo root:

    python eda_plots.py --data "Train and Test/train_split.csv" --out figures

Notes
-----
* The "train_split.csv" file contains all 641 labelled plots; the competition
  test set was unlabelled and is not used here, so this is the complete data.
* One row per species-per-plot. For plot-level views (maps, predictor
  distributions) we deduplicate to one row per plot, since the eight predictors
  are identical across a plot's species-rows. Presence is species-specific, so
  the histograms and presence map keep the species dimension.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# --- thesis-friendly styling: serif, muted palette, no chartjunk ---------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "serif",
    "font.size": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
})

CONTINUOUS = ["rainann", "tempann", "soildepth", "soilfert", "topo"]
NUMERIC_FOR_CORR = ["rainann", "tempann", "soildepth", "soilfert", "topo",
                    "disturb", "easting", "northing"]
ABSENCE_COLOR = "#4C72B0"
PRESENCE_COLOR = "#C44E52"


def load(path):
    df = pd.read_csv(path)
    df["pres.abs"] = df["pres.abs"].astype(int)
    return df


def fig_histograms(df, out):
    """Continuous predictors, presence vs absence (pooled across species)."""
    n = len(CONTINUOUS)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 3 * nrows))
    axes = axes.ravel()
    for ax, col in zip(axes, CONTINUOUS):
        present = df.loc[df["pres.abs"] == 1, col].dropna()
        absent = df.loc[df["pres.abs"] == 0, col].dropna()
        bins = np.linspace(
            df[col].min(), df[col].max(), 30
        )
        # density=True so the rare presence class is visible despite ~6% prevalence
        ax.hist(absent, bins=bins, density=True, alpha=0.6,
                color=ABSENCE_COLOR, label="Absence")
        ax.hist(present, bins=bins, density=True, alpha=0.6,
                color=PRESENCE_COLOR, label="Presence")
        ax.set_xlabel(col)
        ax.set_ylabel("Density")
    for ax in axes[n:]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Predictor distributions by presence / absence (all species pooled)",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "eda_histograms.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_correlation(df, out):
    """Correlation heatmap of numeric predictors (plot-level, deduplicated)."""
    plots = df.drop_duplicates("plot")[NUMERIC_FOR_CORR]
    corr = plots.corr()
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(NUMERIC_FOR_CORR)))
    ax.set_yticks(range(len(NUMERIC_FOR_CORR)))
    ax.set_xticklabels(NUMERIC_FOR_CORR, rotation=45, ha="right")
    ax.set_yticklabels(NUMERIC_FOR_CORR)
    ax.grid(False)
    for i in range(len(NUMERIC_FOR_CORR)):
        for j in range(len(NUMERIC_FOR_CORR)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    color="black" if abs(corr.iloc[i, j]) < 0.6 else "white",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson correlation")
    ax.set_title("Correlation among numeric predictors (plot level)")
    fig.tight_layout()
    fig.savefig(out / "eda_correlation.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_presence_map(df, out):
    """Plot locations coloured by presence, one panel per species."""
    species = sorted(df["Species"].unique())
    ncols = 4
    nrows = int(np.ceil(len(species) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(13, 3.2 * nrows),
                             sharex=True, sharey=True)
    axes = axes.ravel()
    for ax, sp in zip(axes, species):
        sub = df[df["Species"] == sp]
        absent = sub[sub["pres.abs"] == 0]
        present = sub[sub["pres.abs"] == 1]
        ax.scatter(absent["easting"], absent["northing"], s=6,
                   color=ABSENCE_COLOR, alpha=0.3, linewidths=0, label="Absent")
        ax.scatter(present["easting"], present["northing"], s=22,
                   color=PRESENCE_COLOR, edgecolors="black", linewidths=0.3,
                   label="Present")
        ax.set_title(f"{sp}\n({int(present.shape[0])} present)",
                     fontsize=9, style="italic")
        ax.set_aspect("equal", adjustable="box")
    for ax in axes[len(species):]:
        ax.set_visible(False)
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Presence / absence by species over space (easting vs northing)",
                 y=1.01, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "eda_presence_map.pdf", bbox_inches="tight")
    plt.close(fig)


def fig_gradient_map(df, out):
    """A continuous environmental variable (rainfall) over space."""
    plots = df.drop_duplicates("plot")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, (col, label) in zip(
        axes, [("rainann", "Mean annual rainfall"),
               ("tempann", "Mean annual temperature")]
    ):
        sc = ax.scatter(plots["easting"], plots["northing"], c=plots[col],
                        cmap="viridis", s=18, linewidths=0)
        ax.set_xlabel("Easting")
        ax.set_ylabel("Northing")
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(label)
        fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Environmental gradients over space", y=1.02, fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "eda_gradient_map.pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Train and Test/train_split.csv")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load(args.data)
    print(f"Loaded {df.shape[0]} rows, {df['plot'].nunique()} plots, "
          f"{df['Species'].nunique()} species.")
    fig_histograms(df, out)
    fig_correlation(df, out)
    fig_presence_map(df, out)
    fig_gradient_map(df, out)
    print(f"Saved 4 figures to {out.resolve()}")


if __name__ == "__main__":
    main()
