"""
MLP Model — Small Reptile Species Distribution (EcoStat Modelling, DATA5925)

Per-species MLP on the team's CANONICAL plot-level split (loaded via
sdm_protocol). Structure mirrors the other team scripts; modelling choices
follow the shared evaluation protocol, with each divergence from a naive
default flagged inline.

Spec compliance:
  * Split      : canonical plot-level (Ang's), loaded — not re-made.
  * Features   : 6 environmental + easting/northing (USE_COORDS), matching the
                 team's shared feature set (Nickson: prefer easting/northing).
  * Weighting  : NONE. Class weighting ~doubles log loss (see ablation); we
                 optimise calibrated cross-entropy and threshold-tune for F1.
  * Tuning     : grid search by SPATIAL 5-fold CV log loss, dev plots only.
  * Metrics    : full per-species suite + macro + prevalence-weighted.
"""
import itertools, warnings
import numpy as np, pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, average_precision_score,
                             brier_score_loss, confusion_matrix, f1_score,
                             log_loss, roc_auc_score)
import sdm_protocol as P

warnings.filterwarnings("ignore", category=UserWarning)

# ---- configuration ----
USE_COORDS = True     # spec default: include easting/northing (set False for
                      # the Direction C no-coordinate experiment)
FEATURES = P.ENV_FEATURES + (P.COORD_FEATURES if USE_COORDS else [])
CV_SCHEME = "spatial"
GRID = {
    "hidden_layer_sizes": [(8,), (16,), (32,), (16, 8)],
    "alpha":              [1e-4, 1e-3, 1e-2, 1e-1],   # L2 regularisation
    "learning_rate_init": [1e-2, 1e-3],
}
MAX_ITER = 500
THRESHOLDS = np.linspace(0.05, 0.95, 19)


def build_mlp(hidden_layer_sizes, alpha, learning_rate_init):
    return MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, alpha=alpha,
                         learning_rate_init=learning_rate_init, activation="relu",
                         solver="adam", max_iter=MAX_ITER, random_state=P.RANDOM_SEED)


def cv_predict(wide, splits, sp, cfg):
    yt, yp = [], []
    for _, tr, va in P.iter_cv(wide, splits, scheme=CV_SCHEME):
        ytr = tr[sp].values
        if len(np.unique(ytr)) < 2:
            p = np.full(len(va), ytr.mean())
        else:
            sc = StandardScaler().fit(tr[FEATURES])
            m = build_mlp(**cfg).fit(sc.transform(tr[FEATURES]), ytr)
            p = m.predict_proba(sc.transform(va[FEATURES]))[:, 1]
        yt.append(va[sp].values); yp.append(p)
    return np.concatenate(yt), np.concatenate(yp)


def select_config(wide, splits, sp):
    keys = list(GRID); best, best_loss = None, np.inf
    for combo in itertools.product(*[GRID[k] for k in keys]):
        cfg = dict(zip(keys, combo))
        yt, yp = cv_predict(wide, splits, sp, cfg)
        loss = P.evaluate(yt, yp)["log_loss"]
        if loss < best_loss:
            best, best_loss = cfg, loss
    return best, best_loss


def tune_threshold(yt, yp):
    best_t, best_f1 = 0.5, -1.0
    for t in THRESHOLDS:
        f = f1_score(yt, (yp >= t).astype(int), zero_division=0)
        if f > best_f1:
            best_t, best_f1 = t, f
    return best_t


def main():
    wide = P.load_wide()
    splits = P.load_splits(P.SPLIT_FILE)
    dev, test = P.get_dev(wide, splits), P.get_test(wide, splits)
    print(f"dev : {len(dev):>4} plots | test: {len(test):>4} plots | features: {FEATURES}")
    print(f"tuning by {CV_SCHEME} {splits['n_folds']}-fold CV, no class weighting\n")

    configs = {}
    prob_df = pd.DataFrame(index=test.index)
    for sp in P.SPECIES:
        cfg, cv_loss = select_config(wide, splits, sp)
        yt, yp = cv_predict(wide, splits, sp, cfg)
        thr = tune_threshold(yt, yp)
        configs[sp] = {**cfg, "cv_log_loss": round(cv_loss, 4), "threshold": thr}
        ytr = dev[sp].values
        if len(np.unique(ytr)) < 2:
            prob_df[sp] = np.full(len(test), ytr.mean())
        else:
            sc = StandardScaler().fit(dev[FEATURES])
            m = build_mlp(**cfg).fit(sc.transform(dev[FEATURES]), ytr)
            prob_df[sp] = m.predict_proba(sc.transform(test[FEATURES]))[:, 1]

    # overall pooled test performance
    y_all = np.concatenate([test[sp].values for sp in P.SPECIES])
    p_all = np.clip(np.concatenate([prob_df[sp].values for sp in P.SPECIES]), 1e-15, 1 - 1e-15)
    pred_all = (p_all >= 0.5).astype(int)
    print("=" * 45); print("TEST PERFORMANCE (pooled)"); print("=" * 45)
    print(f"  Log loss (primary) : {log_loss(y_all, p_all):.4f}")
    print(f"  ROC-AUC            : {roc_auc_score(y_all, p_all):.4f}")
    print(f"  PR-AUC (avg prec)  : {average_precision_score(y_all, p_all):.4f}")
    print(f"  Brier score        : {brier_score_loss(y_all, p_all):.4f}")
    print(f"  F1 (thr=0.5)       : {f1_score(y_all, pred_all, zero_division=0):.4f}")
    print(f"  Accuracy           : {accuracy_score(y_all, pred_all):.4f}")
    print("\n  Confusion matrix [[TN, FP], [FN, TP]]:")
    print(confusion_matrix(y_all, pred_all))

    # full per-species suite
    results = P.evaluate_all_species(test[P.SPECIES], prob_df)
    mc = ["log_loss", "brier", "auc", "f1", "sensitivity", "specificity"]
    print("\nPER-SPECIES PERFORMANCE (thr=0.5)")
    print("  (rare species: metrics rest on <5 test presences -> treat as noise)")
    print(results[["n_pos", "prevalence"] + mc].astype(float).round(3).to_string())

    pd.DataFrame(configs).T.to_csv("mlp_selected_configs.csv")
    results.to_csv("mlp_test_results.csv")
    print("\nSaved mlp_selected_configs.csv and mlp_test_results.csv")


if __name__ == "__main__":
    main()
