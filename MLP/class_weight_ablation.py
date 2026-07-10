
import numpy as np, pandas as pd
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

TARGET, SPECIES_COL = "pres.abs", "Species"
NUM = ["disturb", "rainann", "soildepth", "soilfert", "tempann", "topo",
       "easting", "northing"]
SEEDS = range(10)
COMMON = ["Ophioscincus truncatus", "Coeranoscincus reticulatus",
          "Calyptotis scutirostrum"]


def build_X(d, cols=None):
    du = pd.get_dummies(d[SPECIES_COL], prefix="sp")
    X = pd.concat([d[NUM].reset_index(drop=True),
                   du.reset_index(drop=True)], axis=1)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


def plot_level_split(df, seed):
    rng = np.random.default_rng(seed)
    plots = df["plot"].unique().copy(); rng.shuffle(plots)
    test = set(plots[:int(round(0.2 * len(plots)))])
    return df[~df["plot"].isin(test)], df[df["plot"].isin(test)]


def fit_predict(train, test, use_weight):
    Xtr = build_X(train); ytr = train[TARGET].astype(int).reset_index(drop=True)
    Xte = build_X(test, list(Xtr.columns))
    spw = ((len(ytr) - ytr.sum()) / max(int(ytr.sum()), 1)) if use_weight else 1.0
    m = xgb.XGBClassifier(objective="binary:logistic", eval_metric="logloss",
                          n_estimators=500, learning_rate=0.03, max_depth=4,
                          min_child_weight=3, subsample=0.8, colsample_bytree=0.8,
                          reg_lambda=2.0, scale_pos_weight=spw, tree_method="hist",
                          random_state=42, n_jobs=-1)
    m.fit(Xtr, ytr, verbose=False)
    return np.clip(m.predict_proba(Xte)[:, 1], 1e-15, 1 - 1e-15)


def run():
    df = pd.read_csv("train.csv")
    rows = []
    # keep one representative seed's predictions for the calibration figure
    fig_data = {}
    for seed in SEEDS:
        train, test = plot_level_split(df, seed)
        y = test[TARGET].astype(int).values
        for use_weight in (True, False):
            p = fit_predict(train, test, use_weight)
            rows.append({"seed": seed, "weighted": use_weight, "scope": "OVERALL",
                         "prevalence": y.mean(), "mean_pred": p.mean(),
                         "log_loss": log_loss(y, p, labels=[0, 1]),
                         "brier": brier_score_loss(y, p),
                         "auc": roc_auc_score(y, p)})
            for sp in df[SPECIES_COL].unique():
                mask = (test[SPECIES_COL] == sp).values
                ys, ps = y[mask], p[mask]
                rows.append({"seed": seed, "weighted": use_weight, "scope": sp,
                             "prevalence": ys.mean(), "mean_pred": ps.mean(),
                             "log_loss": log_loss(ys, ps, labels=[0, 1]),
                             "brier": brier_score_loss(ys, ps),
                             "auc": roc_auc_score(ys, ps) if ys.min() != ys.max() else np.nan})
            if seed == 0:
                fig_data[use_weight] = (test.copy(), p)
    res = pd.DataFrame(rows)
    res.to_csv("ablation_results.csv", index=False)

    # ---- summary printed to console ----
    ov = res[res.scope == "OVERALL"].groupby("weighted")[["log_loss", "brier", "auc"]].mean()
    print("OVERALL (mean over seeds):")
    print(ov.round(4).to_string())
    print()
    print("Common species log-loss (mean over seeds):")
    cs = (res[res.scope.isin(COMMON)]
          .groupby(["scope", "weighted"])["log_loss"].mean().unstack())
    print(cs.round(4).to_string())

    make_figure(df, fig_data)
    return res


def make_figure(df, fig_data):
    species = list(df[SPECIES_COL].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # (a) pooled reliability curve
    for use_weight, style in [(True, dict(c="#c0392b", marker="o", label="with class weighting")),
                              (False, dict(c="#2980b9", marker="s", label="no weighting"))]:
        test, p = fig_data[use_weight]
        y = test[TARGET].astype(int).values
        bins = np.linspace(0, 1, 11)
        idx = np.digitize(p, bins) - 1
        xs, ys = [], []
        for b in range(10):
            m = idx == b
            if m.sum() >= 10:
                xs.append(p[m].mean()); ys.append(y[m].mean())
        ax1.plot(xs, ys, **style)
    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
    ax1.set_xlabel("mean predicted probability"); ax1.set_ylabel("observed frequency")
    ax1.set_title("(a) Reliability curve (all species pooled)")
    ax1.legend(fontsize=9); ax1.set_xlim(0, 1); ax1.set_ylim(0, 1)

    # (b) mean predicted vs observed prevalence, per species
    for use_weight, col in [(True, "#c0392b"), (False, "#2980b9")]:
        test, p = fig_data[use_weight]
        prevs, preds = [], []
        for sp in species:
            m = (test[SPECIES_COL] == sp).values
            prevs.append(test[TARGET].values[m].mean()); preds.append(p[m].mean())
        ax2.scatter(prevs, preds, c=col, s=60,
                    label="with class weighting" if use_weight else "no weighting",
                    zorder=3, edgecolor="white")
    lim = 0.25
    ax2.plot([0, lim], [0, lim], "k--", lw=1, label="perfect calibration")
    ax2.set_xlabel("observed prevalence (per species)")
    ax2.set_ylabel("mean predicted probability")
    ax2.set_title("(b) Per-species calibration")
    ax2.legend(fontsize=9); ax2.set_xlim(0, lim); ax2.set_ylim(0, lim)

    fig.suptitle("Class weighting distorts probabilities under a log-loss objective",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig("calibration_plot.png", dpi=150, bbox_inches="tight")
    print("\nSaved calibration_plot.png")


if __name__ == "__main__":
    run()
