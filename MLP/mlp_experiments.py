
import itertools, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import sdm_protocol as P

warnings.filterwarnings("ignore")

USE_COORDS = True
FEATURES = P.ENV_FEATURES + (P.COORD_FEATURES if USE_COORDS else [])
SCHEMES = ["random", "spatial"]
STRATEGIES = ["single", "multi", "multi_int"]
GRID = {
    "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
    "learning_rate_init": [1e-2, 1e-3],
}
MAX_ITER = 500
THRESHOLDS = np.linspace(0.05, 0.95, 19)
OOF_FIXED = dict(hidden_layer_sizes=(16,), alpha=1e-1, learning_rate_init=1e-2)
KAGGLE_TEST = "kaggle_test.csv"
METRICS = ["log_loss", "brier", "auc", "f1", "sensitivity", "specificity"]
# Below this many TEST presences the threshold-dependent metrics rest on too few
# positives to mean anything, so the reported table suppresses them. Single
# source of truth for both the `reliable` column and that suppression.
MIN_POS_REPORT = 5
# Threshold-dependent -> suppressed for unreliable species. Log loss, Brier and
# AUC are threshold-free and stay visible for every row.
SUPPRESSED_IF_UNRELIABLE = ["f1", "sensitivity", "specificity", "threshold"]


def build(hidden_layer_sizes, alpha, learning_rate_init):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         learning_rate_init=learning_rate_init, activation="relu",
                         solver="adam", max_iter=MAX_ITER, random_state=P.RANDOM_SEED)


def _long(frame):
    rows = []
    for _, r in frame.iterrows():
        for sp in P.SPECIES:
            d = {f: r[f] for f in FEATURES}
            d.update({"Species": sp, "y": r[sp]})
            rows.append(d)
    return pd.DataFrame(rows)


def _onehot(L, cols=None):
    X = pd.concat([L[FEATURES].reset_index(drop=True),
                   pd.get_dummies(L["Species"], prefix="sp", dtype=float).reset_index(drop=True)], axis=1)
    return X.reindex(columns=cols, fill_value=0) if cols is not None else X


# ---------- CV scoring per strategy ----------
def cv_single(wide, splits, sp, cfg, scheme):
    yt, yp = [], []
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        ytr = tr[sp].values
        if len(np.unique(ytr)) < 2:
            p = np.full(len(va), ytr.mean())
        else:
            sc = StandardScaler().fit(tr[FEATURES])
            p = build(**cfg).fit(sc.transform(tr[FEATURES]), ytr).predict_proba(sc.transform(va[FEATURES]))[:, 1]
        yt.append(va[sp].values); yp.append(p)
    return np.concatenate(yt), np.concatenate(yp)


def cv_multi(wide, splits, cfg, scheme):
    per = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        sc = StandardScaler().fit(tr[FEATURES])
        pr = np.array(build(**cfg).fit(sc.transform(tr[FEATURES]), tr[P.SPECIES].values)
                      .predict_proba(sc.transform(va[FEATURES])))
        per.append(np.nanmean([P.evaluate(va[sp].values, pr[:, j])["log_loss"]
                               for j, sp in enumerate(P.SPECIES)]))
    return float(np.nanmean(per))


def cv_multiint(wide, splits, cfg, scheme):
    per = []
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        Ltr, Lva = _long(tr), _long(va)
        Xtr = _onehot(Ltr); Xva = _onehot(Lva, Xtr.columns)
        sc = StandardScaler().fit(Xtr)
        p = build(**cfg).fit(sc.transform(Xtr), Ltr["y"].values).predict_proba(sc.transform(Xva))[:, 1]
        per.append(P.evaluate(Lva["y"].values, p)["log_loss"])
    return float(np.nanmean(per))


def oof_multi(wide, splits, cfg, scheme):
    """Per-species out-of-fold predictions from the shared-trunk model, so its
    cutoffs can be tuned on the same dev folds the single strategy uses."""
    yt = {sp: [] for sp in P.SPECIES}; yp = {sp: [] for sp in P.SPECIES}
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        sc = StandardScaler().fit(tr[FEATURES])
        pr = np.array(build(**cfg).fit(sc.transform(tr[FEATURES]), tr[P.SPECIES].values)
                      .predict_proba(sc.transform(va[FEATURES])))
        for j, sp in enumerate(P.SPECIES):
            yt[sp].append(va[sp].values); yp[sp].append(pr[:, j])
    return ({sp: np.concatenate(yt[sp]) for sp in P.SPECIES},
            {sp: np.concatenate(yp[sp]) for sp in P.SPECIES})


def oof_multiint(wide, splits, cfg, scheme):
    """Same, for the pooled one-hot model: split its single OOF probability
    vector back out per species."""
    yt = {sp: [] for sp in P.SPECIES}; yp = {sp: [] for sp in P.SPECIES}
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        Ltr, Lva = _long(tr), _long(va)
        Xtr = _onehot(Ltr); Xva = _onehot(Lva, Xtr.columns)
        sc = StandardScaler().fit(Xtr)
        p = build(**cfg).fit(sc.transform(Xtr), Ltr["y"].values).predict_proba(sc.transform(Xva))[:, 1]
        for sp in P.SPECIES:
            m = (Lva["Species"] == sp).values
            yt[sp].append(Lva["y"].values[m]); yp[sp].append(p[m])
    return ({sp: np.concatenate(yt[sp]) for sp in P.SPECIES},
            {sp: np.concatenate(yp[sp]) for sp in P.SPECIES})


def grid_select(fn):
    keys = list(GRID); best, bl = None, np.inf
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo)); loss = fn(cfg)
        if loss < bl:
            best, bl = cfg, loss
    return best, bl


def tune_threshold(yt, yp):
    bt, bf = 0.5, -1.0
    for t in THRESHOLDS:
        f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
        if f > bf: bt, bf = t, f
    return float(bt)


def best_oof_f1(yt, yp):
    """Best F1 reachable on these OOF predictions. 0.0 means tune_threshold
    never beat its bf=-1.0 sentinel with a real score, so the cutoff it
    returned is its first candidate (0.05), not a selected optimum."""
    return max(f1_score(yt, (yp >= t).astype(int), zero_division=0) for t in THRESHOLDS)


def reported_table(res_df):
    """Reporting view of the per-species results.

    Threshold-dependent metrics (F1, sensitivity, specificity) and the cutoff
    that produced them are shown as '-' for species with fewer than
    MIN_POS_REPORT test presences: with that few positives the confusion matrix
    is noise, so a number there invites a conclusion the data cannot support.
    Log loss, Brier and AUC are threshold-free and always shown; AUC is '-' only
    where it is undefined (a species with no test presences).
    """
    cols = ["scheme", "strategy", "species", "n_pos",
            "log_loss", "brier", "auc", "f1", "sensitivity", "specificity", "threshold"]
    out = res_df[cols].copy()
    reliable = res_df["n_pos"] >= MIN_POS_REPORT

    for c in ["log_loss", "brier", "auc"]:
        out[c] = ["-" if pd.isna(v) else f"{v:.4f}" for v in res_df[c]]
    for c in SUPPRESSED_IF_UNRELIABLE:
        fmt = "{:.2f}" if c == "threshold" else "{:.3f}"
        out[c] = ["-" if (not ok or pd.isna(v)) else fmt.format(v)
                  for ok, v in zip(reliable, res_df[c])]
    return out


def oof_fixed(wide, splits, scheme):
    st, ss, sm = ({sp: [] for sp in P.SPECIES} for _ in range(3))
    for _, tr, va in P.iter_cv(wide, splits, scheme=scheme):
        sc = StandardScaler().fit(tr[FEATURES])
        Xtr, Xva = sc.transform(tr[FEATURES]), sc.transform(va[FEATURES])
        pm = np.array(build(**OOF_FIXED).fit(Xtr, tr[P.SPECIES].values).predict_proba(Xva))
        for j, sp in enumerate(P.SPECIES):
            ytr = tr[sp].values
            ps = (np.full(len(va), ytr.mean()) if len(np.unique(ytr)) < 2
                  else build(**OOF_FIXED).fit(Xtr, ytr).predict_proba(Xva)[:, 1])
            st[sp].append(va[sp].values); ss[sp].append(ps); sm[sp].append(pm[:, j])
    rows = []
    for sp in P.SPECIES:
        yt = np.concatenate(st[sp])
        s = P.evaluate(yt, np.concatenate(ss[sp]))["log_loss"]
        m = P.evaluate(yt, np.concatenate(sm[sp]))["log_loss"]
        rows.append({"scheme": scheme, "species": sp, "dev_pos": int(yt.sum()),
                     "single_ll": s, "multi_ll": m, "delta_multi_minus_single": m - s,
                     "better": "multi" if m < s - 0.002 else ("single" if m > s + 0.002 else "tie")})
    return rows


def main():
    wide = P.load_wide(); splits = P.load_splits()
    dev, test = P.get_dev(wide, splits), P.get_test(wide, splits)
    print(f"dev {len(dev)} plots | test {len(test)} plots | features {FEATURES}")
    print(f"design: {STRATEGIES} x {SCHEMES} | no class weighting\n")

    results, summary, oof_rows, compare = [], [], [], []
    keep_preds = {}

    for scheme in SCHEMES:
        print(f"--- {scheme.upper()} CV ---")
        # ---- tune ----
        s_cfg, s_thr, s_cv = {}, {}, {}
        fallback = []   # (strategy, species) whose cutoff is tune_threshold's fallback
        for sp in P.SPECIES:
            c, _ = grid_select(lambda cf, x=sp: P.evaluate(*cv_single(wide, splits, x, cf, scheme))["log_loss"])
            yt, yp = cv_single(wide, splits, sp, c, scheme)
            s_cfg[sp] = c; s_thr[sp] = tune_threshold(yt, yp)
            s_cv[sp] = P.evaluate(yt, yp)["log_loss"]
            if best_oof_f1(yt, yp) == 0.0:
                fallback.append(("single", sp))
        m_cfg, m_cv = grid_select(lambda c: cv_multi(wide, splits, c, scheme))
        mi_cfg, mi_cv = grid_select(lambda c: cv_multiint(wide, splits, c, scheme))

        # Per-species cutoffs for the pooled strategies, tuned on the SAME dev
        # out-of-fold predictions the single strategy uses (P.iter_cv drops the
        # test plots), so cross-strategy F1 stays like-for-like.
        myt, myp = oof_multi(wide, splits, m_cfg, scheme)
        m_thr = {sp: tune_threshold(myt[sp], myp[sp]) for sp in P.SPECIES}
        miyt, miyp = oof_multiint(wide, splits, mi_cfg, scheme)
        mi_thr = {sp: tune_threshold(miyt[sp], miyp[sp]) for sp in P.SPECIES}
        for name, oy, op in [("multi", myt, myp), ("multi_int", miyt, miyp)]:
            fallback += [(name, sp) for sp in P.SPECIES if best_oof_f1(oy[sp], op[sp]) == 0.0]

        # ---- fit on dev, predict test ----
        sc_env = StandardScaler().fit(dev[FEATURES])
        s_prob = pd.DataFrame(index=test.index); 
        for sp in P.SPECIES:
            ytr = dev[sp].values
            s_prob[sp] = (np.full(len(test), ytr.mean()) if len(np.unique(ytr)) < 2
                          else build(**s_cfg[sp]).fit(sc_env.transform(dev[FEATURES]), ytr)
                               .predict_proba(sc_env.transform(test[FEATURES]))[:, 1])
        m_arr = np.array(build(**m_cfg).fit(sc_env.transform(dev[FEATURES]), dev[P.SPECIES].values)
                         .predict_proba(sc_env.transform(test[FEATURES])))
        m_prob = pd.DataFrame(m_arr, columns=P.SPECIES, index=test.index)
        Ldev, Lte = _long(dev), _long(test)
        Xd = _onehot(Ldev); Xt = _onehot(Lte, Xd.columns)
        scmi = StandardScaler().fit(Xd)
        Lte = Lte.copy()
        Lte["p"] = build(**mi_cfg).fit(scmi.transform(Xd), Ldev["y"].values).predict_proba(scmi.transform(Xt))[:, 1]
        mi_prob = pd.DataFrame({sp: Lte[Lte.Species == sp]["p"].values for sp in P.SPECIES}, index=test.index)

        if scheme == "spatial":
            # Kaggle's test set is a SPATIAL HOLDOUT (median distance to the
            # nearest labelled plot is 9.1 km vs 0.8 km between labelled plots,
            # an 11x ratio; only 3% of Kaggle plots lie within 1 km of a
            # labelled plot). Spatial CV therefore estimates the quantity the
            # competition actually scores, so the submission model is selected
            # by SPATIAL CV log loss -- never by test-set performance, which
            # would be test-set selection.
            keep_preds = {"single_cfg": s_cfg, "multi_cfg": m_cfg,
                          "multiint_cfg": mi_cfg,
                          "cv": {"single": float(np.mean([s_cv[sp] for sp in P.SPECIES])),
                                 "multi": m_cv, "multi_int": mi_cv}}

        # ---- per-species rows ----
        for strat, prob, cfg_of, cv_of, thr in [
            ("single", s_prob, lambda sp: s_cfg[sp], lambda sp: s_cv[sp], s_thr),
            ("multi", m_prob, lambda sp: m_cfg, lambda sp: m_cv, m_thr),
            ("multi_int", mi_prob, lambda sp: mi_cfg, lambda sp: mi_cv, mi_thr)]:
            # old behaviour (fixed 0.5) kept alongside so the change is auditable
            res_half = P.evaluate_all_species(test[P.SPECIES], prob)
            res = P.evaluate_all_species(test[P.SPECIES], prob, thr)
            for sp in P.SPECIES:
                c = cfg_of(sp)
                results.append({"scheme": scheme, "strategy": strat, "species": sp,
                                "n_pos": int(res.loc[sp, "n_pos"]),
                                "prevalence": float(res.loc[sp, "prevalence"]),
                                **{k: float(res.loc[sp, k]) for k in METRICS},
                                "hidden_layer_sizes": str(c["hidden_layer_sizes"]),
                                "alpha": c["alpha"], "learning_rate_init": c["learning_rate_init"],
                                "threshold": float(res.loc[sp, "threshold"]),
                                "reliable": res.loc[sp, "n_pos"] >= MIN_POS_REPORT})
                compare.append({"scheme": scheme, "strategy": strat, "species": sp,
                                "n_pos": int(res.loc[sp, "n_pos"]),
                                "threshold": float(res.loc[sp, "threshold"]),
                                "is_fallback_cutoff": (strat, sp) in fallback,
                                **{f"{k}_before": float(res_half.loc[sp, k])
                                   for k in ("f1", "sensitivity", "specificity")},
                                **{f"{k}_after": float(res.loc[sp, k])
                                   for k in ("f1", "sensitivity", "specificity")}})
            y_all = np.concatenate([test[sp].values for sp in P.SPECIES])
            p_all = np.concatenate([prob[sp].values for sp in P.SPECIES])
            ov = P.evaluate(y_all, p_all)
            cv_est = np.mean([cv_of(sp) for sp in P.SPECIES]) if strat == "single" else cv_of(None)
            summary.append({"scheme": scheme, "strategy": strat,
                            "test_log_loss": ov["log_loss"], "test_auc": ov["auc"],
                            "test_brier": ov["brier"],
                            "cv_log_loss_estimate": cv_est,
                            "optimism_test_minus_cv": ov["log_loss"] - cv_est,
                            "macro_log_loss": float(res.loc["MACRO_AVG", "log_loss"]),
                            "prev_weighted_log_loss": float(res.loc["PREV_WEIGHTED", "log_loss"])})
        oof_rows += oof_fixed(wide, splits, scheme)
        print(f"  done ({scheme})")

    res_df = pd.DataFrame(results)
    res_df.to_csv("mlp_results.csv", index=False)          # full-precision raw record
    summ = pd.DataFrame(summary); summ.to_csv("mlp_summary.csv", index=False)
    oof = pd.DataFrame(oof_rows); oof.to_csv("mlp_oof.csv", index=False)

    # ---- threshold change: before (fixed 0.5) vs after (per-species, dev-OOF tuned) ----
    cmp_df = pd.DataFrame(compare)
    cmp_df.to_csv("threshold_fix_comparison.csv", index=False)
    FOCUS = ["Calyptotis scutirostrum", "Coeranoscincus reticulatus",
             "Ophioscincus truncatus"]
    print("\n" + "=" * 88)
    print("THRESHOLD CHANGE -- before (fixed 0.5) vs after (per-species, tuned on dev OOF)")
    print("=" * 88)
    for scheme in SCHEMES:
        print(f"\n--- {scheme} CV ---")
        print(f"{'strategy':10s} {'species':28s} {'thr':>5} {'n+':>3}"
              f" {'F1':>16} {'Sensitivity':>16} {'Specificity':>16}")
        d = cmp_df[(cmp_df.scheme == scheme) & (cmp_df.species.isin(FOCUS))]
        for _, r in d.iterrows():
            print(f"{r.strategy:10s} {r.species:28s} {r.threshold:5.2f} {r.n_pos:3d}"
                  f"   {r.f1_before:6.3f}->{r.f1_after:6.3f}"
                  f"   {r.sensitivity_before:6.3f}->{r.sensitivity_after:6.3f}"
                  f"   {r.specificity_before:6.3f}->{r.specificity_after:6.3f}")
    fb = cmp_df[cmp_df.is_fallback_cutoff]
    print(f"\nFallback cutoffs: {len(fb)} of {len(cmp_df)} rows. Best dev-OOF F1 was 0, so"
          f" tune_threshold\nreturned its first candidate rather than a selected optimum:")
    for _, r in fb.iterrows():
        print(f"   {r.scheme:8s} {r.strategy:10s} {r.species:28s} "
              f"thr={r.threshold:.2f} n+={r.n_pos:2d} "
              f"F1 {r.f1_before:.3f}->{r.f1_after:.3f}")
    print("\nWrote threshold_fix_comparison.csv (all rows, before + after)")

    # ---- reporting view: suppress metrics the test set cannot support ----
    rep = reported_table(res_df)
    rep.to_csv("mlp_results_reported.csv", index=False)
    n_sup = int((res_df["n_pos"] < MIN_POS_REPORT).sum())
    print("\n" + "=" * 88)
    print(f"REPORTED PER-SPECIES RESULTS  (F1/sensitivity/specificity/threshold shown as '-'"
          f"\nwhere test presences < {MIN_POS_REPORT}; log loss, Brier and AUC always shown)")
    print("=" * 88)
    for scheme in SCHEMES:
        for strat in STRATEGIES:
            d = rep[(rep.scheme == scheme) & (rep.strategy == strat)]
            print(f"\n{scheme} CV / {strat}")
            print(d.drop(columns=["scheme", "strategy"]).to_string(index=False))
    print(f"\n{n_sup} of {len(res_df)} rows have suppressed threshold-dependent metrics.")
    print("Wrote mlp_results_reported.csv (this view); mlp_results.csv keeps the raw values.")

    # ---- Kaggle submission: the single best model, selected by SPATIAL CV ----
    import os
    cvs = keep_preds["cv"]
    best_strategy = min(cvs, key=cvs.get)
    print("\n" + "=" * 72)
    print("KAGGLE SUBMISSION — model selected by SPATIAL CV log loss")
    print("=" * 72)
    for k, v in sorted(cvs.items(), key=lambda x: x[1]):
        print(f"  {k:10s} spatial-CV log loss = {v:.4f}{'   <-- selected' if k == best_strategy else ''}")
    print("  (selection uses CV, NOT test performance — the test set is not a")
    print("   model-selection tool. Kaggle is a spatial holdout, so spatial CV")
    print("   estimates the quantity it scores.)")

    if os.path.exists(KAGGLE_TEST):
        kt = pd.read_csv(KAGGLE_TEST)
        sc_env = StandardScaler().fit(dev[FEATURES])
        pred = np.zeros(len(kt))

        if best_strategy == "single":
            for sp in P.SPECIES:
                mask = (kt["Species"] == sp).values
                ytr = dev[sp].values
                pred[mask] = (np.full(mask.sum(), ytr.mean()) if len(np.unique(ytr)) < 2
                              else build(**keep_preds["single_cfg"][sp])
                                   .fit(sc_env.transform(dev[FEATURES]), ytr)
                                   .predict_proba(sc_env.transform(kt.loc[mask, FEATURES]))[:, 1])
        elif best_strategy == "multi":
            ktw = kt.drop_duplicates("plot").set_index("plot")
            mm = build(**keep_preds["multi_cfg"]).fit(sc_env.transform(dev[FEATURES]),
                                                      dev[P.SPECIES].values)
            proba = np.array(mm.predict_proba(sc_env.transform(ktw[FEATURES])))
            pmap = {(pl, sp): proba[i, j] for i, pl in enumerate(ktw.index)
                    for j, sp in enumerate(P.SPECIES)}
            pred = np.array([pmap[(r.plot, r.Species)] for r in kt.itertuples()])
        else:  # multi_int  (pooled one-hot)
            Ldev = _long(dev); Xd = _onehot(Ldev)
            Xk = pd.concat([kt[FEATURES].reset_index(drop=True),
                            pd.get_dummies(kt["Species"], prefix="sp", dtype=float)
                              .reset_index(drop=True)], axis=1).reindex(columns=Xd.columns, fill_value=0)
            scmi = StandardScaler().fit(Xd)
            mi = build(**keep_preds["multiint_cfg"]).fit(scmi.transform(Xd), Ldev["y"].values)
            pred = mi.predict_proba(scmi.transform(Xk))[:, 1]

        sub = kt[["id"]].copy()
        sub["pred"] = np.clip(pred, 1e-6, 1 - 1e-6)
        sub.sort_values("id").to_csv("submission_mlp.csv", index=False)
        assert len(sub) == len(kt) and sub["pred"].notna().all() and not sub["id"].duplicated().any()
        print(f"\n  Saved submission_mlp.csv  ({best_strategy}, {len(sub)} rows, "
              f"mean pred={sub['pred'].mean():.4f} vs data base rate 0.063)")
    else:
        print(f"  (No {KAGGLE_TEST} found — skipped submission.)")

    print("\n" + "=" * 72)
    print("HEADLINE — test log loss by strategy x CV scheme")
    print("=" * 72)
    print(summ.pivot(index="strategy", columns="scheme", values="test_log_loss").round(4).to_string())
    print("\nCV estimate vs truth (optimism = test - cv; +ve => CV was optimistic):")
    print(summ.pivot(index="strategy", columns="scheme", values="optimism_test_minus_cv").round(4).to_string())
    print("\nOOF single vs multi (architecture fixed):")
    for s in SCHEMES:
        d = oof[oof.scheme == s]
        print(f"  {s:8s}: multi better on {(d.better=='multi').sum()}/8, "
              f"single better on {(d.better=='single').sum()}/8, tie {(d.better=='tie').sum()}/8")
    print("\nSaved: mlp_results.csv, mlp_summary.csv, mlp_oof.csv")


if __name__ == "__main__":
    main()
