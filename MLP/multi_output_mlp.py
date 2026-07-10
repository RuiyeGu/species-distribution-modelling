
import itertools, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import sdm_protocol as P

warnings.filterwarnings("ignore")

FEATURES = P.ENV_FEATURES          # env only, no coords -> clean single-vs-multi
CV_SCHEME = "spatial"
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


def multi_cv_logloss(wide, splits, cfg):
    """Mean spatial-CV log-loss ACROSS species for one shared-trunk config."""
    per_fold = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xva = sc.transform(tr[FEATURES]), sc.transform(va[FEATURES])
        Ytr = tr[P.SPECIES].values
        m = build(**cfg).fit(Xtr, Ytr)
        proba = np.array(m.predict_proba(Xva))          # (n_val, 8)
        lls = []
        for j, sp in enumerate(P.SPECIES):
            lls.append(P.evaluate(va[sp].values, proba[:, j])["log_loss"])
        per_fold.append(np.nanmean(lls))
    return float(np.nanmean(per_fold))


def select_multi(wide, splits):
    keys = list(GRID)
    best, best_loss = None, np.inf
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        loss = multi_cv_logloss(wide, splits, cfg)
        if loss < best_loss:
            best, best_loss = cfg, loss
    return best, best_loss


def multi_test(wide, splits, cfg):
    dev, tst = P.get_dev(wide, splits), P.get_test(wide, splits)
    sc = StandardScaler().fit(dev[FEATURES])
    m = build(cfg["hidden_layer_sizes"], cfg["alpha"],
              cfg["learning_rate_init"]).fit(sc.transform(dev[FEATURES]),
                                             dev[P.SPECIES].values)
    proba = np.array(m.predict_proba(sc.transform(tst[FEATURES])))
    prob_df = pd.DataFrame(proba, columns=P.SPECIES, index=tst.index)
    return P.evaluate_all_species(tst[P.SPECIES], prob_df)


def oof_controlled(wide, splits, cfg):
   
    fixed = dict(cfg, activation="relu", solver="adam",
                 max_iter=MAX_ITER, random_state=P.RANDOM_SEED)
    s_true = {sp: [] for sp in P.SPECIES}
    s_single = {sp: [] for sp in P.SPECIES}
    s_multi = {sp: [] for sp in P.SPECIES}
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xva = sc.transform(tr[FEATURES]), sc.transform(va[FEATURES])
        pm = np.array(MLPClassifier(**fixed).fit(Xtr, tr[P.SPECIES].values).predict_proba(Xva))
        for j, sp in enumerate(P.SPECIES):
            ytr = tr[sp].values
            if len(np.unique(ytr)) < 2:
                ps = np.full(len(va), ytr.mean())
            else:
                ps = MLPClassifier(**fixed).fit(Xtr, ytr).predict_proba(Xva)[:, 1]
            s_true[sp].append(va[sp].values)
            s_single[sp].append(ps); s_multi[sp].append(pm[:, j])
    rows = {}
    for sp in P.SPECIES:
        yt = np.concatenate(s_true[sp])
        rows[sp] = {"dev_pos": int(yt.sum()),
                    "single_ll": P.evaluate(yt, np.concatenate(s_single[sp]))["log_loss"],
                    "multi_ll":  P.evaluate(yt, np.concatenate(s_multi[sp]))["log_loss"]}
    df = pd.DataFrame(rows).T
    df["delta"] = df["multi_ll"] - df["single_ll"]
    return df


def main():
    wide = P.load_wide("train.csv")
    splits = P.load_splits("protocol_splits.json")

    cfg, cv_loss = select_multi(wide, splits)
    print(f"Best MULTI-output config: {cfg}  (mean CV log-loss across species = {cv_loss:.3f})\n")
    multi = multi_test(wide, splits, cfg)

    # load the per-species (single) results for side-by-side
    single = pd.read_csv("mlp_test_results.csv", index_col=0)

    comp = pd.DataFrame({
        "prevalence": single.loc[P.SPECIES, "prevalence"],
        "single_logloss": single.loc[P.SPECIES, "log_loss"],
        "multi_logloss":  multi.loc[P.SPECIES, "log_loss"],
        "single_auc": single.loc[P.SPECIES, "auc"],
        "multi_auc":  multi.loc[P.SPECIES, "auc"],
    })
    comp["ll_delta(multi-single)"] = comp["multi_logloss"] - comp["single_logloss"]
    comp.to_csv("multi_vs_single_mlp.csv")

    print("Per-species TEST comparison (lower log-loss better):")
    print(comp.round(3).to_string())
    print()
    # only the two learnable species carry interpretable signal
    learn = ["Coeranoscincus reticulatus", "Ophioscincus truncatus"]
    print("Learnable species only (the trustworthy comparison):")
    print(comp.loc[learn].round(3).to_string())
    print(f"\nMACRO log-loss  single={single.loc['MACRO_AVG','log_loss']:.3f}  "
          f"multi={multi.loc['MACRO_AVG','log_loss']:.3f}")
    print(f"PREV-WEIGHTED   single={single.loc['PREV_WEIGHTED','log_loss']:.3f}  "
          f"multi={multi.loc['PREV_WEIGHTED','log_loss']:.3f}")

    print("\n" + "=" * 60)
    print("TRUSTWORTHY COMPARISON: out-of-fold, architecture held fixed")
    print("(only difference = parameter sharing; full dev presences per species)")
    print("=" * 60)
    oof = oof_controlled(wide, splits, cfg)
    oof.to_csv("multi_vs_single_oof.csv")
    print(oof.round(3).to_string())
    n_better = (oof["delta"] < -0.002).sum()
    print(f"\nMulti-output better on {n_better}/{len(P.SPECIES)} species out of fold.")


if __name__ == "__main__":
    main()
