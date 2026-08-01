
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

os.makedirs("slides", exist_ok=True)

# light theme, big fonts, presentation sizing
plt.rcParams.update({
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 18, "axes.titlesize": 20, "axes.labelsize": 18,
    "xtick.labelsize": 15, "ytick.labelsize": 15, "legend.fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 1.3, "lines.linewidth": 2.5,
})
BLUE, RED, ORANGE, GREY = "#2c6fbb", "#c0392b", "#e08a3c", "#9aa0a6"
FIGSIZE = (11, 6.2)  # 16:9-ish

SHORT = {"Eulamprus murrayi":"E. murrayi","Saltuarius swaini":"S. swaini",
         "Egernia mcpheei":"E. mcpheei","Pseudechis porphyricaus":"P. porphyricaus",
         "Cacophis kreftii":"C. kreftii","Calyptotis scutirostrum":"C. scutirostrum",
         "Coeranoscincus reticulatus":"C. reticulatus","Ophioscincus truncatus":"O. truncatus"}


def _nearest(A, B):
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1))
    return d.min(axis=1)


def slide_spatial_holdout():
    need = ["train_plotlevel.csv", "test_plotlevel.csv", "kaggle_test.csv"]
    if not all(os.path.exists(f) for f in need):
        print(f"skip spatial_holdout: need {need}"); return
    tr = pd.read_csv("train_plotlevel.csv").drop_duplicates("plot")
    te = pd.read_csv("test_plotlevel.csv").drop_duplicates("plot")
    kt = pd.read_csv("kaggle_test.csv").drop_duplicates("plot")
    A = tr[["easting","northing"]].values
    d_int = _nearest(te[["easting","northing"]].values, A) / 1000
    d_kag = _nearest(kt[["easting","northing"]].values, A) / 1000

    fig, ax = plt.subplots(figsize=FIGSIZE)
    bins = np.linspace(0, max(d_int.max(), d_kag.max()), 35)
    ax.hist(d_int, bins=bins, alpha=0.8, color=BLUE,
            label=f"Our test plots  (median {np.median(d_int):.1f} km)")
    ax.hist(d_kag, bins=bins, alpha=0.8, color=RED,
            label=f"Competition plots  (median {np.median(d_kag):.1f} km)")
    ax.set_xlabel("Distance to nearest training plot (km)")
    ax.set_ylabel("Number of plots")
    ax.set_title("The competition asks us to predict FAR from where we trained",
                 fontweight="bold")
    ax.legend(frameon=False)
    fig.savefig("slides/slide_spatial_holdout.png")
    plt.close(fig)
    print("wrote slides/slide_spatial_holdout.png")


def slide_cv_flip():
    if not os.path.exists("mlp_oof.csv"):
        print("skip cv_flip: mlp_oof.csv not found"); return
    d = pd.read_csv("mlp_oof.csv")
    dcol = "delta_multi_minus_single" if "delta_multi_minus_single" in d else "delta"
    piv = d.pivot(index="species", columns="scheme", values=dcol)
    if "dev_pos" in d.columns:
        order = d.groupby("species")["dev_pos"].first().sort_values().index
        piv = piv.loc[order]
    labels = [SHORT.get(s, s) for s in piv.index]
    x = np.arange(len(piv)); w = 0.4
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(x - w/2, piv["random"], w, label="Random validation", color=BLUE)
    ax.bar(x + w/2, piv["spatial"], w, label="Spatial validation", color=ORANGE)
    ax.axhline(0, color="k", lw=1.2)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right",
                                         style="italic", fontsize=13)
    ax.set_ylabel("multi-species better  $\\leftarrow$   $\\rightarrow$  single better")
    nr = (d[d.scheme=="random"][dcol] < -0.002).sum()
    ns = (d[d.scheme=="spatial"][dcol] < -0.002).sum()
    ax.set_title(f"Pooling species helps {nr}/8 species... until we validate honestly ({ns}/8)",
                 fontweight="bold", fontsize=18)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig("slides/slide_cv_flip.png")
    plt.close(fig)
    print(f"wrote slides/slide_cv_flip.png ({nr}/8 -> {ns}/8)")


def slide_weighting():
    if not os.path.exists("ablation_results.csv"):
        print("skip weighting: ablation_results.csv not found (using calibration_plot.png instead is fine)")
        return
    a = pd.read_csv("ablation_results.csv")
    # tolerate different column layouts: need a weighted flag and log_loss
    wcol = next((c for c in a.columns if "weight" in c.lower()), None)
    lcol = next((c for c in a.columns if c.lower() in ("log_loss","logloss","ll")), None)
    acol = next((c for c in a.columns if c.lower()=="auc"), None)
    if not (wcol and lcol):
        print("skip weighting: could not find weight/log_loss columns in ablation_results.csv"); return
    g = a.groupby(wcol)[[c for c in [lcol, acol] if c]].mean()
    # map booleans to labels
    def lab(v): return "Class weighting" if v in (True,"True",1,"balanced") else "No weighting"
    order = sorted(g.index, key=lambda v: 0 if lab(v)=="No weighting" else 1)
    labels = [lab(v) for v in order]
    ll = [g.loc[v, lcol] for v in order]
    fig, ax = plt.subplots(figsize=(9, 6.2))
    bars = ax.bar(labels, ll, color=[BLUE, RED], width=0.55)
    for b, v in zip(bars, ll):
        ax.text(b.get_x()+b.get_width()/2, v+0.005, f"{v:.3f}",
                ha="center", fontsize=17, fontweight="bold")
    ax.set_ylabel("Log loss  (lower = better)")
    pct = (ll[1]-ll[0])/ll[0]*100 if ll[0] else 0
    auc_note = ""
    if acol:
        au = [g.loc[v, acol] for v in order]
        auc_note = f"\n(ranking ability unchanged: AUC {au[0]:.2f} vs {au[1]:.2f})"
    ax.set_title(f"A standard imbalance 'fix' makes probabilities {pct:.0f}% worse{auc_note}",
                 fontweight="bold", fontsize=17)
    ax.set_ylim(0, max(ll)*1.25)
    fig.savefig("slides/slide_weighting.png")
    plt.close(fig)
    print(f"wrote slides/slide_weighting.png (+{pct:.0f}% log loss)")


def slide_morans():
    if not os.path.exists("direction_e_results.csv"):
        print("skip morans: direction_e_results.csv not found"); return
    d = pd.read_csv("direction_e_results.csv")
    gcol = "optimism_gap" if "optimism_gap" in d else "gap"
    mcol = "morans_I" if "morans_I" in d else "morans_i"
    dcol = "dev_pos" if "dev_pos" in d else "dev_presences"
    d = d.dropna(subset=[gcol, mcol])
    rel = d[dcol] >= 15
    rho, p = stats.spearmanr(d[mcol], d[gcol])
    fig, ax = plt.subplots(figsize=FIGSIZE)
    # only label the reliable, data-rich species on a slide (readability)
    ax.scatter(d.loc[~rel, mcol], d.loc[~rel, gcol], s=90, facecolors="none",
               edgecolors=GREY, linewidths=1.6)
    ax.scatter(d.loc[rel, mcol], d.loc[rel, gcol], s=150, color=RED,
               edgecolors="k", linewidths=0.8, zorder=3)
    for _, r in d[rel].iterrows():
        ax.annotate(SHORT.get(r["species"], r["species"]), (r[mcol], r[gcol]),
                    fontsize=13, xytext=(7,4), textcoords="offset points",
                    style="italic")
    ax.axhline(0, color="#ccc", lw=1); ax.axvline(0, color="#ccc", lw=1)
    ax.set_xlabel("Leftover spatial pattern in errors (Moran's $I$)")
    ax.set_ylabel("How much random validation over-estimates")
    ax.set_title(f"WHY random validation lies: the two track each other ($\\rho={rho:.2f}$)",
                 fontweight="bold", fontsize=18)
    fig.savefig("slides/slide_morans.png")
    plt.close(fig)
    print(f"wrote slides/slide_morans.png (rho={rho:.2f})")


if __name__ == "__main__":
    slide_spatial_holdout()
    slide_cv_flip()
    slide_weighting()
    slide_morans()
    print("\nDone. Figures in slides/ — check each renders legibly at slide size before use.")
