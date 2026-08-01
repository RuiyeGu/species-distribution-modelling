
import warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from scipy import stats
import sdm_protocol as P

warnings.filterwarnings("ignore")

FEATURES = P.ENV_FEATURES + P.COORD_FEATURES
# Fixed architecture (median of the tuned per-species configs) -- held constant
# so the ONLY difference between conditions is the fold assignment.
FIXED = dict(hidden_layer_sizes=(16,), alpha=1e-1, learning_rate_init=1e-2,
             activation="relu", solver="adam", max_iter=500,
             random_state=P.RANDOM_SEED)


def oof_by_scheme(wide, splits, scheme):
    """Out-of-fold predictions + per-fold log loss for every species."""
    per_fold, oof = [], {sp: {"y": [], "p": [], "plot": []} for sp in P.SPECIES}
    for k, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xva = sc.transform(tr[FEATURES]), sc.transform(va[FEATURES])
        for sp in P.SPECIES:
            ytr = tr[sp].values
            if len(np.unique(ytr)) < 2:
                p = np.full(len(va), ytr.mean())
            else:
                p = MLPClassifier(**FIXED).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            ll = P.evaluate(va[sp].values, p)["log_loss"]
            per_fold.append({"scheme": scheme, "fold": k, "species": sp,
                             "n_val": len(va), "n_pos": int(va[sp].sum()),
                             "log_loss": ll})
            oof[sp]["y"].append(va[sp].values); oof[sp]["p"].append(p)
            oof[sp]["plot"].append(va[P.PLOT_COL].values)
    for sp in P.SPECIES:
        for key in ("y", "p", "plot"):
            oof[sp][key] = np.concatenate(oof[sp][key])
    return pd.DataFrame(per_fold), oof


def morans_i(values, coords, k=8, n_perm=499, seed=42):
    """Moran's I with k-nearest-neighbour row-standardised weights.
    Returns (I, expected_I, pseudo p-value from permutation)."""
    n = len(values)
    d = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    W = np.zeros((n, n))
    rows = np.repeat(np.arange(n), k)
    W[rows, nn.ravel()] = 1.0
    W = W / W.sum(axis=1, keepdims=True)          # row-standardised

    z = values - values.mean()
    denom = (z ** 2).sum()
    if denom == 0:
        return np.nan, np.nan, np.nan

    def _I(zv):
        return (n / W.sum()) * (zv @ (W @ zv)) / (zv ** 2).sum()

    I = _I(z)
    rng = np.random.RandomState(seed)
    null = np.array([_I(rng.permutation(z)) for _ in range(n_perm)])
    p = (1 + (np.abs(null) >= abs(I)).sum()) / (n_perm + 1)
    return float(I), float(-1 / (n - 1)), float(p)


def main():
    wide = P.load_wide(); splits = P.load_splits()
    dev = P.get_dev(wide, splits)
    print(f"Direction E — random vs spatial CV | {len(dev)} dev plots")
    print(f"Fixed architecture (model held constant): {FIXED['hidden_layer_sizes']}, "
          f"alpha={FIXED['alpha']}\nFeatures: {FEATURES}\n")

    # ---- Part 1: per-fold log loss under each scheme ----
    rows_r, oof_r = oof_by_scheme(wide, splits, "random")
    rows_s, oof_s = oof_by_scheme(wide, splits, "spatial")
    folds = pd.concat([rows_r, rows_s], ignore_index=True)

    # ---- Part 2: per-species optimism gap, with paired test ----
    summ = []
    for sp in P.SPECIES:
        r = rows_r[rows_r.species == sp]["log_loss"].values
        s = rows_s[rows_s.species == sp]["log_loss"].values
        gap = s.mean() - r.mean()
        # paired t-test across the 5 folds (same dev plots, different assignment)
        t, pval = stats.ttest_rel(s, r) if len(s) == len(r) else (np.nan, np.nan)
        summ.append({"species": sp, "dev_pos": int(oof_r[sp]["y"].sum()),
                     "random_ll_mean": r.mean(), "random_ll_sd": r.std(ddof=1),
                     "spatial_ll_mean": s.mean(), "spatial_ll_sd": s.std(ddof=1),
                     "optimism_gap": gap,
                     "gap_pct": 100 * gap / r.mean() if r.mean() else np.nan,
                     "paired_t": t, "p_value": pval})
    # overall (pooled across species, per fold)
    r_all = rows_r.groupby("fold")["log_loss"].mean().values
    s_all = rows_s.groupby("fold")["log_loss"].mean().values
    t_all, p_all = stats.ttest_rel(s_all, r_all)
    summ.append({"species": "OVERALL(macro)", "dev_pos": np.nan,
                 "random_ll_mean": r_all.mean(), "random_ll_sd": r_all.std(ddof=1),
                 "spatial_ll_mean": s_all.mean(), "spatial_ll_sd": s_all.std(ddof=1),
                 "optimism_gap": s_all.mean() - r_all.mean(),
                 "gap_pct": 100 * (s_all.mean() - r_all.mean()) / r_all.mean(),
                 "paired_t": t_all, "p_value": p_all})
    summary = pd.DataFrame(summ)

    print("=" * 78)
    print("PART 1-2: OPTIMISM GAP (spatial - random), paired across folds")
    print("=" * 78)
    print(summary[["species", "dev_pos", "random_ll_mean", "spatial_ll_mean",
                   "optimism_gap", "gap_pct", "p_value"]].round(4).to_string(index=False))

    # ---- Part 3: MECHANISM — Moran's I on out-of-fold residuals ----
    coords = wide.set_index(P.PLOT_COL).loc[oof_r[P.SPECIES[0]]["plot"]][["easting", "northing"]].values
    mrows = []
    for sp in P.SPECIES:
        resid = oof_r[sp]["y"] - oof_r[sp]["p"]      # residuals under random CV
        c = wide.set_index(P.PLOT_COL).loc[oof_r[sp]["plot"]][["easting", "northing"]].values
        I, EI, pv = morans_i(resid, c, k=8)
        mrows.append({"species": sp, "dev_pos": int(oof_r[sp]["y"].sum()),
                      "morans_I": I, "expected_I": EI, "p_value": pv})
    morans = pd.DataFrame(mrows)

    print("\n" + "=" * 78)
    print("PART 3: MECHANISM — Moran's I on out-of-fold RESIDUALS (k=8 NN)")
    print("  I > 0 => residuals spatially clustered => covariates missed spatial")
    print("  signal => random CV leaks it => optimism. I ~ 0 => schemes agree.")
    print("=" * 78)
    print(morans.round(4).to_string(index=False))

    # ---- does the gap track the autocorrelation? ----
    m = summary[summary.species != "OVERALL(macro)"].merge(morans[["species", "morans_I"]], on="species")
    valid = m.dropna(subset=["morans_I", "optimism_gap"])
    if len(valid) >= 3:
        rho, prho = stats.spearmanr(valid["morans_I"], valid["optimism_gap"])
        print(f"\nSpearman(Moran's I, optimism gap) across species: rho={rho:.3f}, p={prho:.3f}")
        print("  (positive rho = species with more residual autocorrelation show a")
        print("   larger random-vs-spatial gap — the mechanism, quantified)")

    # ---- ONE tidy output: gap + mechanism side by side ----
    out = summary.merge(morans[["species", "morans_I", "p_value"]]
                        .rename(columns={"p_value": "morans_p"}), on="species", how="left")
    out["reliable"] = out["dev_pos"] >= 15
    out.to_csv("direction_e_results.csv", index=False)
    print("\nSaved direction_e_results.csv (gap + mechanism in one table)")


if __name__ == "__main__":
    main()
