
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("figures", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "font.size": 11, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.bbox": "tight"})

SPECIES_SHORT = {  # for readable labels
    "Eulamprus murrayi": "E. murrayi", "Saltuarius swaini": "S. swaini",
    "Egernia mcpheei": "E. mcpheei", "Pseudechis porphyricaus": "P. porphyricaus",
    "Cacophis kreftii": "C. kreftii", "Calyptotis scutirostrum": "C. scutirostrum",
    "Coeranoscincus reticulatus": "C. reticulatus", "Ophioscincus truncatus": "O. truncatus",
}
RELIABLE_MIN_DEV = 15   # species below this are noise-dominated; drawn hollow


def fig_morans_gap():
    if not os.path.exists("direction_e_results.csv"):
        print("skip fig1: direction_e_results.csv not found"); return
    d = pd.read_csv("direction_e_results.csv")
    # tolerate either column naming from the two script versions
    gap_col = "optimism_gap" if "optimism_gap" in d else "gap"
    mi_col = "morans_I" if "morans_I" in d else "morans_i"
    dev_col = "dev_pos" if "dev_pos" in d else "dev_presences"
    d = d.dropna(subset=[gap_col, mi_col])

    reliable = d[dev_col] >= RELIABLE_MIN_DEV
    rho, p = stats.spearmanr(d[mi_col], d[gap_col])

    fig, ax = plt.subplots(figsize=(6.4, 5))
    # unreliable species: hollow grey; reliable: filled
    ax.scatter(d.loc[~reliable, mi_col], d.loc[~reliable, gap_col],
               s=70, facecolors="none", edgecolors="#999", linewidths=1.4,
               label=f"< {RELIABLE_MIN_DEV} presences (noise-dominated)")
    ax.scatter(d.loc[reliable, mi_col], d.loc[reliable, gap_col],
               s=90, color="#c0392b", edgecolors="k", linewidths=0.6, zorder=3,
               label=f"$\\geq$ {RELIABLE_MIN_DEV} presences")
    for _, r in d.iterrows():
        ax.annotate(SPECIES_SHORT.get(r["species"], r["species"]),
                    (r[mi_col], r[gap_col]), fontsize=8,
                    xytext=(5, 3), textcoords="offset points",
                    style="italic", color="#333")
    ax.axhline(0, color="#bbb", lw=0.8, zorder=0)
    ax.axvline(0, color="#bbb", lw=0.8, zorder=0)
    ax.set_xlabel("Moran's $I$ on out-of-fold residuals")
    ax.set_ylabel("Optimism gap (spatial $-$ random log loss)")
    ax.set_title(f"Residual spatial autocorrelation predicts CV optimism\n"
                 f"Spearman $\\rho = {rho:.2f}$, $p = {p:.3f}$", fontsize=11)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.savefig("figures/fig_morans_gap.png")
    plt.close(fig)
    print(f"wrote figures/fig_morans_gap.png  (rho={rho:.3f}, p={p:.3f})")

def fig_cv_flip():
    if not os.path.exists("mlp_oof.csv"):
        print("skip fig2: mlp_oof.csv not found"); return
    d = pd.read_csv("mlp_oof.csv")
    delta_col = "delta_multi_minus_single" if "delta_multi_minus_single" in d else "delta"
    piv = d.pivot(index="species", columns="scheme", values=delta_col)
    # order by dev presences if available, else by spatial delta
    if "dev_pos" in d.columns:
        order = d.groupby("species")["dev_pos"].first().sort_values().index
        piv = piv.loc[order]

    labels = [SPECIES_SHORT.get(s, s) for s in piv.index]
    x = np.arange(len(piv)); w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ax.bar(x - w/2, piv["random"], w, label="Random CV", color="#5b8bd0")
    ax.bar(x + w/2, piv["spatial"], w, label="Spatial CV", color="#e08a3c")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right",
                                         style="italic", fontsize=9)
    ax.set_ylabel("log loss:  multi $-$ single")
    n_rand = (d[(d.scheme=="random")][delta_col] < -0.002).sum()
    n_spat = (d[(d.scheme=="spatial")][delta_col] < -0.002).sum()
    ax.set_title("Multi-species advantage depends on validation scheme\n"
                 f"multi better (below zero) on {n_rand}/8 species under random CV, "
                 f"{n_spat}/8 under spatial CV", fontsize=11)
    ax.text(0.015, 0.04, "below 0 = multi-species better", transform=ax.transAxes,
            fontsize=8, color="#555", style="italic")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig("figures/fig_cv_flip.png")
    plt.close(fig)
    print(f"wrote figures/fig_cv_flip.png  (multi better: {n_rand}/8 random, {n_spat}/8 spatial)")


def _nearest(A, B):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)


def fig_spatial_holdout():
    need = ["train_plotlevel.csv", "test_plotlevel.csv", "kaggle_test.csv"]
    if not all(os.path.exists(f) for f in need):
        print(f"skip fig3: need {need}"); return
    tr = pd.read_csv("train_plotlevel.csv").drop_duplicates("plot")
    te = pd.read_csv("test_plotlevel.csv").drop_duplicates("plot")
    kt = pd.read_csv("kaggle_test.csv").drop_duplicates("plot")
    A = tr[["easting", "northing"]].values
    d_int = _nearest(te[["easting", "northing"]].values, A)
    d_kag = _nearest(kt[["easting", "northing"]].values, A)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bins = np.linspace(0, max(d_int.max(), d_kag.max()), 40)
    ax.hist(d_int, bins=bins, alpha=0.75, color="#5b8bd0",
            label=f"Internal test (median {np.median(d_int):,.0f} m)")
    ax.hist(d_kag, bins=bins, alpha=0.75, color="#c0392b",
            label=f"Kaggle test (median {np.median(d_kag):,.0f} m)")
    ax.axvline(1000, color="#333", ls="--", lw=1)
    ax.text(1000, ax.get_ylim()[1]*0.9, " 1 km", fontsize=8, color="#333")
    ax.set_xlabel("Distance to nearest training plot (m)")
    ax.set_ylabel("Number of plots")
    ax.set_title("The Kaggle test set is a spatial holdout\n"
                 "internal test plots sit among the training plots; Kaggle plots do not",
                 fontsize=11)
    ax.legend(frameon=False, fontsize=9)
    fig.savefig("figures/fig_spatial_holdout.png")
    plt.close(fig)
    print(f"wrote figures/fig_spatial_holdout.png  "
          f"(median internal {np.median(d_int):.0f} m, Kaggle {np.median(d_kag):.0f} m)")


if __name__ == "__main__":
    fig_morans_gap()
    fig_cv_flip()
    fig_spatial_holdout()
    print("\nDone. calibration_plot.png is produced separately by class_weight_ablation.py.")
