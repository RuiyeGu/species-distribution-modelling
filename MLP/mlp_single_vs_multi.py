
import itertools, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sdm_protocol as P

warnings.filterwarnings("ignore")

USE_COORDS = True
FEATURES = P.ENV_FEATURES + (P.COORD_FEATURES if USE_COORDS else [])
CV_SCHEME = "spatial"
KAGGLE_TEST_FILE = "kaggle_test.csv"     # unlabelled Kaggle test set (optional)
GRID = {
    "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
    "learning_rate_init": [1e-2, 1e-3],
}
MAX_ITER = 500


def build(hidden_layer_sizes, alpha, learning_rate_init):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         learning_rate_init=learning_rate_init, activation="relu",
                         solver="adam", max_iter=MAX_ITER, random_state=P.RANDOM_SEED)


# ---------- strategy 1: single-species ----------
def single_cv_logloss(wide, splits, sp, cfg):
    losses = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        ytr = tr[sp].values
        if len(np.unique(ytr)) < 2:
            p = np.full(len(va), ytr.mean())
        else:
            sc = StandardScaler().fit(tr[FEATURES])
            p = build(**cfg).fit(sc.transform(tr[FEATURES]), ytr).predict_proba(sc.transform(va[FEATURES]))[:, 1]
        losses.append(P.evaluate(va[sp].values, p)["log_loss"])
    return float(np.nanmean(losses))


def single_fit_predict(dev, target, cfg, X_pred):
    ytr = dev[target].values
    if len(np.unique(ytr)) < 2:
        return np.full(len(X_pred), ytr.mean())
    sc = StandardScaler().fit(dev[FEATURES])
    return build(**cfg).fit(sc.transform(dev[FEATURES]), ytr).predict_proba(sc.transform(X_pred[FEATURES]))[:, 1]


# ---------- strategy 2: shared-trunk multi-output ----------
def multi_cv_logloss(wide, splits, cfg):
    per_fold = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        sc = StandardScaler().fit(tr[FEATURES])
        m = build(**cfg).fit(sc.transform(tr[FEATURES]), tr[P.SPECIES].values)
        proba = np.array(m.predict_proba(sc.transform(va[FEATURES])))
        per_fold.append(np.nanmean([P.evaluate(va[sp].values, proba[:, j])["log_loss"]
                                    for j, sp in enumerate(P.SPECIES)]))
    return float(np.nanmean(per_fold))


def multi_fit_predict(dev, cfg, X_pred):
    sc = StandardScaler().fit(dev[FEATURES])
    m = build(**cfg).fit(sc.transform(dev[FEATURES]), dev[P.SPECIES].values)
    return np.array(m.predict_proba(sc.transform(X_pred[FEATURES])))  # (n, 8)


# ---------- strategy 3: pooled one-hot (long format) ----------
def _long(frame):
    rows = []
    for _, r in frame.iterrows():
        for sp in P.SPECIES:
            d = {f: r[f] for f in FEATURES}
            d.update({"Species": sp, "y": r[sp], "plot": r[P.PLOT_COL]})
            rows.append(d)
    return pd.DataFrame(rows)


def _onehot(L, cols=None):
    X = pd.concat([L[FEATURES].reset_index(drop=True),
                   pd.get_dummies(L["Species"], prefix="sp", dtype=float).reset_index(drop=True)], axis=1)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


def multiint_cv_logloss(wide, splits, cfg):
    per_fold = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        Xtr = _onehot(_long(tr)); Xva = _onehot(_long(va), Xtr.columns)
        ytr = _long(tr)["y"].values; yva = _long(va)["y"].values
        sc = StandardScaler().fit(Xtr)
        p = build(**cfg).fit(sc.transform(Xtr), ytr).predict_proba(sc.transform(Xva))[:, 1]
        per_fold.append(P.evaluate(yva, p)["log_loss"])
    return float(np.nanmean(per_fold))


def grid_select(score_fn):
    keys = list(GRID); best, bl = None, np.inf
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo)); loss = score_fn(cfg)
        if loss < bl:
            best, bl = cfg, loss
    return best, bl


# ---------- OOF comparison (trustworthy single vs multi) ----------
def oof_single_vs_multi(wide, splits, cfg):
    """Architecture held fixed; only difference is parameter sharing. Full dev
    presences per species out of fold."""
    st, ss, sm = {sp: [] for sp in P.SPECIES}, {sp: [] for sp in P.SPECIES}, {sp: [] for sp in P.SPECIES}
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xva = sc.transform(tr[FEATURES]), sc.transform(va[FEATURES])
        pm = np.array(build(**cfg).fit(Xtr, tr[P.SPECIES].values).predict_proba(Xva))
        for j, sp in enumerate(P.SPECIES):
            ytr = tr[sp].values
            ps = (np.full(len(va), ytr.mean()) if len(np.unique(ytr)) < 2
                  else build(**cfg).fit(Xtr, ytr).predict_proba(Xva)[:, 1])
            st[sp].append(va[sp].values); ss[sp].append(ps); sm[sp].append(pm[:, j])
    rows = {}
    for sp in P.SPECIES:
        yt = np.concatenate(st[sp])
        rows[sp] = {"dev_pos": int(yt.sum()),
                    "single_ll": P.evaluate(yt, np.concatenate(ss[sp]))["log_loss"],
                    "multi_ll":  P.evaluate(yt, np.concatenate(sm[sp]))["log_loss"]}
    df = pd.DataFrame(rows).T
    df["delta"] = df["multi_ll"] - df["single_ll"]
    return df


def main():
    wide = P.load_wide(); splits = P.load_splits()
    dev, test = P.get_dev(wide, splits), P.get_test(wide, splits)
    print(f"features: {FEATURES}\ndev: {len(dev)} plots | test: {len(test)} plots\n")

    # ---- tune ----
    single_cfgs = {sp: grid_select(lambda c, s=sp: single_cv_logloss(wide, splits, s, c))[0] for sp in P.SPECIES}
    multi_cfg, _ = grid_select(lambda c: multi_cv_logloss(wide, splits, c))
    multiint_cfg, _ = grid_select(lambda c: multiint_cv_logloss(wide, splits, c))

    # ---- test-set predictions ----
    single_p = {sp: single_fit_predict(dev, sp, single_cfgs[sp], test) for sp in P.SPECIES}
    multi_p = multi_fit_predict(dev, multi_cfg, test)
    L_dev, L_test = _long(dev), _long(test)
    Xdev = _onehot(L_dev); Xte = _onehot(L_test, Xdev.columns)
    sc_mi = StandardScaler().fit(Xdev)
    mi_model = build(**multiint_cfg).fit(sc_mi.transform(Xdev), L_dev["y"].values)
    L_test = L_test.copy(); L_test["p"] = mi_model.predict_proba(sc_mi.transform(Xte))[:, 1]

    # ---- per-species test table (Ang-style) ----
    rows = []
    for j, sp in enumerate(P.SPECIES):
        y = test[sp].values
        ll_s = P.evaluate(y, single_p[sp])["log_loss"]
        ll_m = P.evaluate(y, multi_p[:, j])["log_loss"]
        mi = L_test[L_test["Species"] == sp]
        ll_mi = P.evaluate(mi["y"].values, mi["p"].values)["log_loss"]
        best = min([("single", ll_s), ("multi", ll_m), ("multi_int", ll_mi)], key=lambda x: x[1])[0]
        rows.append({"species": sp, "n_val": len(y), "presence_rate": float(y.mean()),
                     "log_loss_single": ll_s, "log_loss_multi": ll_m,
                     "log_loss_multi_int": ll_mi, "best_approach": best})
    res = pd.DataFrame(rows); res.to_csv("mlp_single_vs_multi_results.csv", index=False)

    # ---- overall summary ----
    y_all = np.concatenate([test[sp].values for sp in P.SPECIES])
    summ = [{"approach": "single", **{k: P.evaluate(y_all, np.concatenate([single_p[sp] for sp in P.SPECIES]))[k] for k in ["log_loss", "auc", "brier"]}},
            {"approach": "multi", **{k: P.evaluate(y_all, np.concatenate([multi_p[:, j] for j in range(len(P.SPECIES))]))[k] for k in ["log_loss", "auc", "brier"]}},
            {"approach": "multi_int", **{k: P.evaluate(L_test["y"].values, L_test["p"].values)[k] for k in ["log_loss", "auc", "brier"]}}]
    summary = pd.DataFrame(summ); summary.to_csv("mlp_single_vs_multi_summary.csv", index=False)

    # ---- OOF comparison (trustworthy) ----
    oof = oof_single_vs_multi(wide, splits, multi_cfg)
    oof.to_csv("mlp_single_vs_multi_oof.csv")

    # ---- configs ----
    cfgs = {"multi_shared_trunk": multi_cfg, "multi_int_pooled": multiint_cfg,
            **{f"single::{sp}": single_cfgs[sp] for sp in P.SPECIES}}
    pd.DataFrame(cfgs).T.to_csv("mlp_selected_configs.csv")

    # ---- Kaggle submissions (if the unlabelled test file is present) ----
    import os
    if os.path.exists(KAGGLE_TEST_FILE):
        kt = pd.read_csv(KAGGLE_TEST_FILE)
        # single: per-species predict on matching rows
        sc_s = {sp: StandardScaler().fit(dev[FEATURES]) for sp in P.SPECIES}
        sub_s = kt[["id"]].copy(); preds = np.zeros(len(kt))
        for sp in P.SPECIES:
            mask = (kt["Species"] == sp).values
            preds[mask] = single_fit_predict(dev, sp, single_cfgs[sp], kt.loc[mask])
        sub_s["pred"] = np.clip(preds, 1e-6, 1 - 1e-6)
        sub_s.sort_values("id").to_csv("submission_mlp_single.csv", index=False)
        # multi: shared-trunk predict per row via its species column
        Xk = _onehot(kt.assign(**{sp: 0 for sp in P.SPECIES}).rename(columns={}))  # placeholder not used
        # shared-trunk needs per-plot wide input; predict per plot then map back
        kt_wide = kt.drop_duplicates("plot").set_index("plot")
        sc_m = StandardScaler().fit(dev[FEATURES])
        mm = build(**multi_cfg).fit(sc_m.transform(dev[FEATURES]), dev[P.SPECIES].values)
        proba = np.array(mm.predict_proba(sc_m.transform(kt_wide[FEATURES])))  # (n_plots, 8)
        prob_map = {(plot, sp): proba[i, j] for i, plot in enumerate(kt_wide.index) for j, sp in enumerate(P.SPECIES)}
        sub_m = kt[["id"]].copy()
        sub_m["pred"] = np.clip([prob_map[(r.plot, r.Species)] for r in kt.itertuples()], 1e-6, 1 - 1e-6)
        sub_m.sort_values("id").to_csv("submission_mlp_multi.csv", index=False)
        print("Saved submission_mlp_single.csv and submission_mlp_multi.csv")
    else:
        print(f"(No {KAGGLE_TEST_FILE} found — skipped Kaggle submissions.)")

    print("\nPER-SPECIES (test log loss):"); print(res.round(4).to_string(index=False))
    print("\nOVERALL (test):"); print(summary.round(4).to_string(index=False))
    print("\nOUT-OF-FOLD single vs multi (trustworthy):"); print(oof.round(4).to_string())
    n_better = int((oof["delta"] < -0.002).sum())
    print(f"Multi better on {n_better}/{len(P.SPECIES)} species out of fold.")
    print("\nSaved all result CSVs.")


if __name__ == "__main__":
    main()
