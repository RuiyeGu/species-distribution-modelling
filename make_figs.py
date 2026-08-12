import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

import argparse

_p = argparse.ArgumentParser()
_p.add_argument("--protocol-dir", type=Path, default=None)
_p.add_argument("--out-dir", type=Path, default=Path("figures"))
_args = _p.parse_args()


def _find_protocol(explicit):
    if explicit is not None:
        return explicit.resolve()
    here = Path.cwd()
    for cand in (here / "protocol", here.parent / "protocol", here):
        if (cand / "protocol.py").exists() or (cand / "GAM").is_dir():
            return cand.resolve()
    raise SystemExit("Could not find protocol/ - run from the repo root or pass --protocol-dir")


ROOT = _find_protocol(_args.protocol_dir)
OUT = _args.out_dir.resolve()
OUT.mkdir(parents=True, exist_ok=True)
print(f"protocol dir : {ROOT}")
print(f"output dir   : {OUT}")

FAMILIES = {
    "Logistic Regression": ("Logistic Regression", "Logistic"),
    "GAM": ("GAM", "GAM"),
    "Random Forest": ("Random Forest", "RF"),
    "XGBoost": ("XGBoost", "XGBoost"),
    "MLP": ("MLP", "MLP"),
}
SPECIES_ORDER = [
    "Eulamprus murrayi", "Saltuarius swaini", "Egernia mcpheei",
    "Pseudechis porphyricaus", "Cacophis kreftii", "Calyptotis scutirostrum",
    "Coeranoscincus reticulatus", "Ophioscincus truncatus",
]
COLORS = {"single": "#2980b9", "multi": "#e67e22", "multi_interactions": "#27ae60"}
EPS = 1e-15


def load_metrics(coord="with_coords"):
    frames = []
    for label, (folder, prefix) in FAMILIES.items():
        f = ROOT / folder / "Output" / f"{prefix}_{coord}_all_metrics.csv"
        if not f.exists():
            print(f"  MISSING {f}")
            continue
        d = pd.read_csv(f)
        d["family"] = label
        d["formulation"] = (d["model"].str.replace(f"_{coord}", "", regex=False)
                            .str.replace(f"{prefix}_", "", regex=False))
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def load_predictions(coord="with_coords"):
    frames = []
    for label, (folder, prefix) in FAMILIES.items():
        for split in ["random", "spatial"]:
            f = ROOT / folder / "Output" / f"{prefix}_{coord}_{split}_predictions.csv"
            if not f.exists():
                continue
            d = pd.read_csv(f)
            d["family"] = label
            d["formulation"] = (d["model"].str.replace(f"_{coord}", "", regex=False)
                                .str.replace(f"{prefix}_", "", regex=False))
            frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else None


def pooled_logloss(g):
    return log_loss(g["y_true"], np.clip(g["y_prob"], EPS, 1 - EPS), labels=[0, 1])


def bootstrap_ci(g, n_boot=400, seed=42):
    """Bootstrap over PLOTS (not rows), so species-within-plot stay together."""
    rng = np.random.default_rng(seed)
    plots = g["plot"].unique()
    by_plot = {p: sub for p, sub in g.groupby("plot")}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(plots, size=len(plots), replace=True)
        sample = pd.concat([by_plot[p] for p in pick], ignore_index=True)
        if sample["y_true"].nunique() < 2:
            continue
        vals.append(pooled_logloss(sample))
    return (np.percentile(vals, 2.5), np.percentile(vals, 97.5)) if vals else (np.nan, np.nan)


# figure 1
def fig_heatmaps(met, coord, split="spatial"):
    """Species x family heatmap of single - multi (log loss and F1)."""
    sub = met[met.split_type == split]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6))
    for ax, metric, title in [
        (axes[0], "log_loss", "(a) Log-loss difference (single $-$ multi)"),
        (axes[1], "f1", "(b) $F_1$ difference (single $-$ multi)"),
    ]:
        mat, used_int = [], []
        for sp in SPECIES_ORDER:
            row = []
            for fam in FAMILIES:
                d = sub[(sub.family == fam) & (sub.species == sp)]
                s = d[d.formulation == "single"][metric]
                m = d[d.formulation == "multi"][metric]
                if m.empty:  # XGBoost: only multi_interactions exists
                    m = d[d.formulation == "multi_interactions"][metric]
                    if fam not in used_int:
                        used_int.append(fam)
                row.append(np.nan if (s.empty or m.empty)
                           else float(s.iloc[0]) - float(m.iloc[0]))
            mat.append(row)
        mat = np.array(mat, dtype=float)
        lim = np.nanmax(np.abs(mat)) or 1.0
        im = ax.imshow(mat, cmap="RdBu_r", norm=TwoSlopeNorm(0, -lim, lim), aspect="auto")
        ax.set_xticks(range(len(FAMILIES)))
        ax.set_xticklabels(FAMILIES, rotation=30, ha="right", fontsize=9)
        ax.set_yticks(range(len(SPECIES_ORDER)))
        ax.set_yticklabels(SPECIES_ORDER, fontsize=9, style="italic")
        ax.set_title(title, fontsize=11)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                            fontsize=7.5,
                            color="white" if abs(mat[i, j]) > 0.6 * lim else "black")
        plt.colorbar(im, ax=ax, fraction=0.046)
    note = ("Red: single-species better. Blue: multi-species better. "
            f"{split.capitalize()} split, {coord.replace('_', ' ')}. "
            "XGBoost uses multi (interactions); MLP has no interactions variant.")
    fig.suptitle("Single- vs multi-species performance by species and model family",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.text(0.5, -0.06, note, ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    p = OUT / f"heatmap_diff_{coord}_{split}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# figure 2
def fig_per_family(met, coord, split="spatial"):
    """Per-species log loss by formulation, one panel per family."""
    sub = met[met.split_type == split]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    axes = axes.ravel()
    for ax, fam in zip(axes, FAMILIES):
        d = sub[sub.family == fam]
        forms = [f for f in ["single", "multi", "multi_interactions"]
                 if f in d.formulation.unique()]
        width = 0.8 / len(forms)
        x = np.arange(len(SPECIES_ORDER))
        for k, f in enumerate(forms):
            vals = [d[(d.species == sp) & (d.formulation == f)]["log_loss"]
                    for sp in SPECIES_ORDER]
            vals = [float(v.iloc[0]) if not v.empty else np.nan for v in vals]
            ax.bar(x + k * width - 0.4 + width / 2, vals, width,
                   label=f.replace("_", " + "), color=COLORS[f], edgecolor="white",
                   linewidth=0.5)
        pos = [int(d[d.species == sp]["positives"].iloc[0]) if not d[d.species == sp].empty
               else 0 for sp in SPECIES_ORDER]
        ax.set_xticks(x)
        ax.set_xticklabels([f"{sp}\n(n$^+$={p})" for sp, p in zip(SPECIES_ORDER, pos)],
                           rotation=40, ha="right", fontsize=7.5, style="italic")
        ax.set_ylabel("Log loss (lower is better)", fontsize=9)
        ax.set_title(fam, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.25)
        for i, p in enumerate(pos):
            if p < 5:
                ax.axvspan(i - 0.5, i + 0.5, color="grey", alpha=0.10, zorder=0)
    axes[-1].axis("off")
    axes[-1].text(0.5, 0.5,
                  "Shaded columns: fewer than 5 test\npresences — metrics are\n"
                  "noise-dominated and not\ninterpretable per species.\n\n"
                  f"{split.capitalize()} split, {coord.replace('_', ' ')}.",
                  ha="center", va="center", fontsize=10, style="italic")
    fig.suptitle(f"Per-species log loss by formulation ({split} split)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    p = OUT / f"per_family_logloss_{coord}_{split}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


# figure 3
def fig_overall(pred, coord):
    """Pooled overall log loss, single vs multi, random and spatial splits."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    table = []
    for ax, split in zip(axes, ["random", "spatial"]):
        d = pred[pred.split_type == split]
        forms = ["single", "multi", "multi_interactions"]
        width = 0.26
        x = np.arange(len(FAMILIES))
        for k, f in enumerate(forms):
            vals, los, his = [], [], []
            for fam in FAMILIES:
                g = d[(d.family == fam) & (d.formulation == f)]
                if g.empty:
                    vals.append(np.nan); los.append(np.nan); his.append(np.nan); continue
                v = pooled_logloss(g)
                lo, hi = bootstrap_ci(g)
                vals.append(v); los.append(v - lo); his.append(hi - v)
                table.append(dict(coord=coord, split=split, family=fam, formulation=f,
                                  log_loss=round(v, 4), ci_lo=round(lo, 4),
                                  ci_hi=round(hi, 4)))
            ax.bar(x + k * width - width, vals, width, label=f.replace("_", " + "),
                   color=COLORS[f], edgecolor="white", linewidth=0.5,
                   yerr=[np.nan_to_num(los), np.nan_to_num(his)],
                   error_kw=dict(lw=1, capsize=3, ecolor="#444444"))
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"({'a' if split=='random' else 'b'}) {split.capitalize()} split"
                     f" — {'interpolation' if split=='random' else 'spatial extrapolation'}",
                     fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Pooled test log loss (lower is better)", fontsize=10)
    fig.suptitle("Overall test log loss by model family and formulation",
                 fontsize=13, fontweight="bold")
    fig.text(0.5, -0.04, "Error bars: 95% bootstrap CI resampling test PLOTS "
             f"(400 replicates). {coord.replace('_', ' ')}.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    p = OUT / f"overall_logloss_{coord}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p, pd.DataFrame(table)


# figure 4
def fig_optimism(pred, coord):
    """Optimism gap: spatial minus random pooled log loss, per family/formulation."""
    fig, ax = plt.subplots(figsize=(11, 5.2))
    forms = ["single", "multi", "multi_interactions"]
    width = 0.26
    x = np.arange(len(FAMILIES))
    rows = []
    for k, f in enumerate(forms):
        gaps = []
        for fam in FAMILIES:
            r = pred[(pred.family == fam) & (pred.formulation == f)
                     & (pred.split_type == "random")]
            s = pred[(pred.family == fam) & (pred.formulation == f)
                     & (pred.split_type == "spatial")]
            if r.empty or s.empty:
                gaps.append(np.nan); continue
            gap = pooled_logloss(s) - pooled_logloss(r)
            gaps.append(gap)
            rows.append(dict(coord=coord, family=fam, formulation=f,
                             random=round(pooled_logloss(r), 4),
                             spatial=round(pooled_logloss(s), 4), gap=round(gap, 4)))
        ax.bar(x + k * width - width, gaps, width, label=f.replace("_", " + "),
               color=COLORS[f], edgecolor="white", linewidth=0.5)
    ax.axhline(0, color="black", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(FAMILIES, rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Optimism gap: spatial $-$ random log loss", fontsize=10)
    ax.set_title("Optimism gap by model family — how much random evaluation flatters",
                 fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.text(0.5, -0.04, "Positive: random split reports better performance than the "
             f"spatial split. {coord.replace('_', ' ')}.",
             ha="center", fontsize=8.5, style="italic")
    fig.tight_layout()
    p = OUT / f"optimism_gap_{coord}.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p, pd.DataFrame(rows)


# figure 5
def fig_coords(met_with, met_no):
    """Effect of dropping coordinates, per family (Direction C)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=True)
    for ax, split in zip(axes, ["random", "spatial"]):
        x = np.arange(len(FAMILIES))
        for k, (lab, met, col) in enumerate(
                [("with coords", met_with, "#2980b9"), ("no coords", met_no, "#c0392b")]):
            vals = []
            for fam in FAMILIES:
                d = met[(met.family == fam) & (met.split_type == split)
                        & (met.formulation == "single")]
                vals.append(d["log_loss"].mean() if not d.empty else np.nan)
            ax.bar(x + k * 0.38 - 0.19, vals, 0.38, label=lab, color=col,
                   edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILIES, rotation=25, ha="right", fontsize=9)
        ax.set_title(f"{split.capitalize()} split", fontsize=11)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Mean per-species log loss", fontsize=10)
    fig.suptitle("Effect of dropping easting/northing (single-species formulation)",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    p = OUT / "coordinate_effect.png"
    fig.savefig(p, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return p


if __name__ == "__main__":
    met_w = load_metrics("with_coords")
    met_n = load_metrics("no_coords")
    pred_w = load_predictions("with_coords")
    print(f"metrics rows: {len(met_w)} | prediction rows: {len(pred_w)}")

    made = []
    for split in ["spatial", "random"]:
        made.append(fig_heatmaps(met_w, "with_coords", split))
        made.append(fig_per_family(met_w, "with_coords", split))
    p, tbl = fig_overall(pred_w, "with_coords")
    made.append(p)
    tbl.to_csv(OUT / "overall_logloss_table.csv", index=False)
    p, gaps = fig_optimism(pred_w, "with_coords")
    made.append(p)
    gaps.to_csv(OUT / "optimism_gap_table.csv", index=False)
    made.append(fig_coords(met_w, met_n))

    print("\nFIGURES WRITTEN:")
    for m in made:
        print("  ", m.name)
    print("\nOVERALL POOLED LOG LOSS (with_coords):")
    print(tbl.to_string(index=False))
    print("\nOPTIMISM GAPS (with_coords):")
    print(gaps.to_string(index=False))
